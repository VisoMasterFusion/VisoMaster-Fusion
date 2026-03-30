"""
FP16 PyTorch reimplementation of occluder.onnx
(ResNet-encoder U-Net binary occluder — 256 px input/output).

Architecture: ResNet-style encoder (no BatchNorm) with 4 stages
              (64→128→256→512) + symmetric U-Net decoder with nearest-
              neighbour upsampling and skip connections.

Input:  (1, 3, 256, 256) float32 — RGB face crop, values in [0, 1]
Output: (1, 1, 256, 256) float32 — raw logits  (positive = face / not occluded)

All 31 Conv2d weights are loaded positionally from the ONNX graph's topological
Conv node order → PyTorch forward-execution order.
No normalization layers exist in the ONNX; biases are stored directly in Conv nodes.

Usage:
    model  = OccluderTorch.from_onnx("model_assets/occluder.onnx").cuda().eval()
    runner = build_cuda_graph_runner(model)   # fixed 256×256 → single captured graph

    with torch.no_grad():
        logits = runner(x)   # (1,3,256,256) float32 → (1,1,256,256) float32
    mask = logits.squeeze() > 0  # (256,256) bool — True = face
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Encoder building blocks
# ---------------------------------------------------------------------------


class _BasicBlock(nn.Module):
    """
    ResNet BasicBlock — no BatchNorm (pure Conv + bias + ReLU).

    Forward execution order (governs positional weight loading):
        conv1 → relu1 → conv2 → [downsample] → add → relu2
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=True)
        self.downsample: Optional[nn.Conv2d] = (
            nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=True)
            if (stride != 1 or in_ch != out_ch)
            else None
        )
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu1(self.conv1(x))
        out = self.conv2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu2(out + identity)


# ---------------------------------------------------------------------------
# Decoder building block
# ---------------------------------------------------------------------------


class _DecBlock(nn.Module):
    """
    Decoder block: nearest-neighbour 2× upsample → optional skip-cat → 2× Conv+ReLU.

    skip_ch = 0 means no skip connection (dec4 — final block).
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        concat_ch = in_ch + skip_ch
        self.conv1 = nn.Conv2d(concat_ch, out_ch, 3, padding=1, bias=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=True)
        self.relu2 = nn.ReLU(inplace=True)
        self._has_skip = skip_ch > 0

    def forward(
        self, x: torch.Tensor, skip: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self._has_skip and skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.relu2(self.conv2(self.relu1(self.conv1(x))))


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------


class OccluderTorch(nn.Module):
    """
    FP16 PyTorch occluder matching occluder.onnx exactly.

    compute_dtype: torch.float16 (default, fastest) or torch.float32.
    Input always accepted as float32; cast happens inside forward().
    Output always returned as float32.
    """

    def __init__(self, compute_dtype: torch.dtype = torch.float16) -> None:
        super().__init__()
        self._compute_dtype = compute_dtype

        # ── Encoder stem ─────────────────────────────────────────────────
        self.stem_conv = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=True)
        self.stem_relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        # ── Encoder stages ───────────────────────────────────────────────
        self.layer1 = nn.Sequential(
            _BasicBlock(64, 64),
            _BasicBlock(64, 64),
        )
        self.layer2 = nn.Sequential(
            _BasicBlock(64, 128, stride=2),
            _BasicBlock(128, 128),
        )
        self.layer3 = nn.Sequential(
            _BasicBlock(128, 256, stride=2),
            _BasicBlock(256, 256),
        )
        self.layer4 = nn.Sequential(
            _BasicBlock(256, 512, stride=2),
            _BasicBlock(512, 512),
        )

        # ── Decoder stages ───────────────────────────────────────────────
        # dec0: bottleneck(512) + skip layer3(256) → 256
        self.dec0 = _DecBlock(512, 256, 256)
        # dec1: dec0(256)       + skip layer2(128) → 128
        self.dec1 = _DecBlock(256, 128, 128)
        # dec2: dec1(128)       + skip layer1(64)  → 64
        self.dec2 = _DecBlock(128, 64, 64)
        # dec3: dec2(64)        + skip stem_relu(64) → 32
        self.dec3 = _DecBlock(64, 64, 32)
        # dec4: dec3(32)        no skip             → 16
        self.dec4 = _DecBlock(32, 0, 16)

        # ── Output head ──────────────────────────────────────────────────
        self.head = nn.Conv2d(16, 1, 3, padding=1, bias=True)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self._compute_dtype)

        # Encoder
        s0 = self.stem_relu(self.stem_conv(x))  # (1, 64, 128, 128) — skip for dec3
        x = self.maxpool(s0)  # (1, 64,  64,  64)
        x = self.layer1(x)  # (1, 64,  64,  64) — skip for dec2
        s1 = x
        x = self.layer2(x)  # (1,128,  32,  32) — skip for dec1
        s2 = x
        x = self.layer3(x)  # (1,256,  16,  16) — skip for dec0
        s3 = x
        x = self.layer4(x)  # (1,512,   8,   8) — bottleneck

        # Decoder
        x = self.dec0(x, s3)  # resize + cat → (1,256, 16, 16)
        x = self.dec1(x, s2)  # resize + cat → (1,128, 32, 32)
        x = self.dec2(x, s1)  # resize + cat → (1, 64, 64, 64)
        x = self.dec3(x, s0)  # resize + cat → (1, 32,128,128)
        x = self.dec4(x)  # resize (no skip) → (1, 16,256,256)

        return self.head(x).float()  # raw logits, (1,1,256,256) float32

    # ------------------------------------------------------------------
    @classmethod
    def from_onnx(
        cls,
        onnx_path: str,
        compute_dtype: torch.dtype = torch.float16,
    ) -> "OccluderTorch":
        """Build and return an OccluderTorch with weights loaded from *onnx_path*."""
        import onnx

        onnx_model = onnx.load(onnx_path)
        model = cls(compute_dtype=compute_dtype)
        _load_all_params(model, onnx_model)
        model._visomaster_onnx_path = str(onnx_path)
        return model.to(compute_dtype)


# ---------------------------------------------------------------------------
# Weight-loading helpers
# ---------------------------------------------------------------------------


def _conv_modules_in_forward_order(model: OccluderTorch) -> list:
    """
    Return all 31 Conv2d modules in the exact order the ONNX graph executes them
    (= PyTorch forward-execution order).

    Index layout
    ────────────
    [0]            stem_conv
    [1–4]          layer1 (2 blocks, no downsample):
                     [1] l1b0.conv1  [2] l1b0.conv2
                     [3] l1b1.conv1  [4] l1b1.conv2
    [5–9]          layer2 (block0 with downsample, block1):
                     [5] l2b0.conv1  [6] l2b0.conv2  [7] l2b0.downsample
                     [8] l2b1.conv1  [9] l2b1.conv2
    [10–14]        layer3 (block0 with downsample, block1):
                     [10] l3b0.conv1 [11] l3b0.conv2 [12] l3b0.downsample
                     [13] l3b1.conv1 [14] l3b1.conv2
    [15–19]        layer4 (block0 with downsample, block1):
                     [15] l4b0.conv1 [16] l4b0.conv2 [17] l4b0.downsample
                     [18] l4b1.conv1 [19] l4b1.conv2
    [20–21]        dec0.conv1  dec0.conv2
    [22–23]        dec1.conv1  dec1.conv2
    [24–25]        dec2.conv1  dec2.conv2
    [26–27]        dec3.conv1  dec3.conv2
    [28–29]        dec4.conv1  dec4.conv2
    [30]           head
    """
    mods: list = []

    # Stem
    mods.append(model.stem_conv)

    # Encoder stages
    for stage in (model.layer1, model.layer2, model.layer3, model.layer4):
        for blk in stage:
            mods.append(blk.conv1)
            mods.append(blk.conv2)
            if blk.downsample is not None:
                mods.append(blk.downsample)

    # Decoder stages
    for dec in (model.dec0, model.dec1, model.dec2, model.dec3, model.dec4):
        mods.append(dec.conv1)
        mods.append(dec.conv2)

    # Output head
    mods.append(model.head)

    assert len(mods) == 31, f"Expected 31 Conv modules, got {len(mods)}"
    return mods


def _load_all_params(model: OccluderTorch, onnx_model) -> None:
    """
    Load all Conv2d weights and biases positionally from the ONNX graph.

    Strategy: iterate ONNX nodes in topological order; for each Conv node,
    copy weight (and bias if present) to the next positional Conv2d module.
    """
    from onnx import numpy_helper

    init_map: dict = {
        init.name: numpy_helper.to_array(init) for init in onnx_model.graph.initializer
    }

    conv_mods = _conv_modules_in_forward_order(model)
    conv_idx = 0

    for node in onnx_model.graph.node:
        if node.op_type != "Conv":
            continue

        mod = conv_mods[conv_idx]

        # Weight
        w_arr = init_map[node.input[1]]
        mod.weight.data.copy_(torch.from_numpy(w_arr.copy()))

        # Bias (all occluder Conv nodes carry an explicit bias)
        if len(node.input) > 2 and node.input[2] in init_map:
            b_arr = init_map[node.input[2]]
            mod.bias.data.copy_(torch.from_numpy(b_arr.copy()))

        conv_idx += 1

    assert conv_idx == 31, f"Expected 31 Conv nodes in ONNX, found {conv_idx}"


# ---------------------------------------------------------------------------
# CUDA graph runner
# ---------------------------------------------------------------------------


class _CapturedGraph:
    """Single CUDA graph captured for fixed input (1, 3, 256, 256)."""

    def __init__(self, model: OccluderTorch, warmup: int = 3) -> None:
        self._model = model
        device = next(model.parameters()).device

        self._inp = torch.zeros(1, 3, 256, 256, dtype=torch.float32, device=device)
        self._out: Optional[torch.Tensor] = None

        # Warm-up on the default stream — primes cuDNN workspace caches
        # (MaxPool2d, interpolate) before the graph capture begins.
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(self._inp)

        # Explicit device sync before capture ensures no in-flight GPU work
        # interferes with cudaStreamBeginCapture.  A dedicated capture stream
        # is used so the capture is isolated from the default stream.
        self._graph = torch.cuda.CUDAGraph()
        self._stream = torch.cuda.Stream()
        torch.cuda.synchronize()
        with (
            torch.no_grad(),
            torch.cuda.graph(
                self._graph, stream=self._stream, capture_error_mode="relaxed"
            ),
        ):
            self._out = model(self._inp)
        torch.cuda.synchronize()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        self._inp.copy_(x, non_blocking=True)
        self._graph.replay()
        return self._out.clone()  # type: ignore[union-attr, return-value]


def build_cuda_graph_runner(
    model: OccluderTorch,
    warmup: int = 3,
    torch_compile: bool = False,
) -> _CapturedGraph:
    """
    Return a CUDA-graph-backed runner for fixed (1, 3, 256, 256) input.

    Falls back gracefully: caller should catch exceptions and use the eager
    model directly if capture fails.
    """
    if torch_compile:
        try:
            from custom_kernels.compile_utils import apply_torch_compile
            device = next(model.parameters()).device
            example_inp = torch.zeros((1, 3, 256, 256), dtype=torch.float32, device=device)
            compiled = apply_torch_compile(model, example_inp)
            print("[OccluderTorch] torch.compile warmup done.")
            return compiled  # CUDA graph on top of torch.compile fails on Windows
        except Exception as e:
            print(f"[OccluderTorch] torch.compile failed ({e!s:.120}), falling back to CUDA graph.")
    return _CapturedGraph(model, warmup=warmup)
