"""
w600k_r50_torch.py — FP16 PyTorch reimplementation of w600k_r50.onnx
=====================================================================

Architecture (reverse-engineered from ONNX topological order):

  Input : (1, 3, 112, 112) float32  — normalised face crop ([0,255]−127.5)/127.5
  Output: (1, 512)          float32  — L2-normalised ArcFace embedding

  Backbone : IResNet-50 (InsightFace) — live BatchNorm (NOT folded), PReLU activations
  Head     : BN2 → Flatten → FC(25088→512) → BN_features → 512-dim embedding

  Model    : w600k_r50 (trained on WebFace600K, used by Inswapper128ArcFace)

Weight loading strategy
-----------------------
  * 53 Conv2d  — loaded POSITIONALLY in ONNX topological order (integer initialiser names)
  * 25 PReLU   — loaded POSITIONALLY in ONNX topological order (integer initialiser names)
  * 26 BatchNorm — loaded BY NAME  (e.g. "layer1.0.bn1.weight")
  * FC (Gemm)  — loaded by name   ("fc.weight", "fc.bias")
  * BN_features — loaded by name  ("features.weight" etc.)

No Triton kernels are required — FP16 speed-up comes entirely from cuDNN
TensorCore kernels for Conv2d on Ampere/Ada GPUs.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# IBasicBlock (single-BN pre-activation block from InsightFace w600k_r50)
# ---------------------------------------------------------------------------


class _IBasicBlock(nn.Module):
    """InsightFace IBasicBlock — single BN, PReLU, no post-block BN.

    Forward: BN1(x) → Conv1(3×3,s=1) → PReLU → Conv2(3×3,stride) → Add(shortcut)
    Shortcut: identity (in_ch==out_ch, stride==1) or Conv1×1(stride) on pre-BN input x.

    All Conv2d declared with bias=True; those lacking an ONNX bias will have
    their bias zero-initialised (numerically equivalent to bias=False).
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_ch, eps=1e-5, momentum=0.1, affine=True)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1, bias=True)
        self.prelu = nn.PReLU(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=stride, padding=1, bias=True)
        self.downsample: Optional[nn.Conv2d] = (
            nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=True)
            if (stride != 1 or in_ch != out_ch)
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.prelu(out)
        out = self.conv2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        return out + identity


def _make_layer(in_ch: int, out_ch: int, num_blocks: int, stride: int) -> nn.Sequential:
    """Build one IResNet stage.  Block0 always has a downsample (stride or channel change)."""
    blocks: List[_IBasicBlock] = [_IBasicBlock(in_ch, out_ch, stride=stride)]
    for _ in range(1, num_blocks):
        blocks.append(_IBasicBlock(out_ch, out_ch, stride=1))
    return nn.Sequential(*blocks)


# ---------------------------------------------------------------------------
# Full IResNet-50 model
# ---------------------------------------------------------------------------


class IResNet50Torch(nn.Module):
    """FP16-capable PyTorch reimplementation of w600k_r50.onnx (IResNet-50 / ArcFace).

    Input : (1, 3, 112, 112) float32
    Output: (1, 512)          float32  — ArcFace embedding (raw logits; app L2-normalises)
    """

    def __init__(self) -> None:
        super().__init__()

        # Stem — Conv(3→64, 3×3, s=1, p=1, bias=True) + PReLU; NO BatchNorm, NO MaxPool
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=True)
        self.prelu0 = nn.PReLU(64)

        # IResNet-50 stages: layers=[3,4,14,3], all stride=2
        self.layer1 = _make_layer(64, 64, num_blocks=3, stride=2)  # → 56×56
        self.layer2 = _make_layer(64, 128, num_blocks=4, stride=2)  # → 28×28
        self.layer3 = _make_layer(128, 256, num_blocks=14, stride=2)  # → 14×14
        self.layer4 = _make_layer(256, 512, num_blocks=3, stride=2)  # →  7×7

        # Head
        self.bn2 = nn.BatchNorm2d(512, eps=1e-5, momentum=0.1, affine=True)
        self.fc = nn.Linear(512 * 7 * 7, 512, bias=True)  # 25088 → 512
        self.features = nn.BatchNorm1d(512, eps=1e-5, affine=True)

        # Compute dtype is set by from_onnx(); inputs cast inside forward()
        self._compute_dtype: torch.dtype = torch.float32

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self._compute_dtype)

        # Stem
        x = self.prelu0(self.conv1(x))

        # Backbone stages
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Head: BN2 → Flatten → FC → BN_features
        x = self.bn2(x)
        x = torch.flatten(x, 1)  # (1, 25088)
        x = self.fc(x)  # (1, 512)
        x = self.features(x)  # (1, 512)

        return x.float()

    # ------------------------------------------------------------------
    # Class-method constructor
    # ------------------------------------------------------------------

    @classmethod
    def from_onnx(
        cls,
        onnx_path: str | Path,
        compute_dtype: torch.dtype = torch.float16,
    ) -> "IResNet50Torch":
        """Build IResNet50Torch and load weights from the w600k_r50 ONNX file.

        Args:
            onnx_path:     Path to w600k_r50.onnx.
            compute_dtype: dtype for internal computation (default: float16).

        Returns:
            IResNet50Torch instance with weights loaded, converted to compute_dtype.
        """
        import onnx  # type: ignore

        onnx_path = Path(onnx_path)
        print(f"[IResNet50Torch] Loading ONNX model from {onnx_path} …")
        onnx_model = onnx.load(str(onnx_path))

        model = cls()
        _load_all_params(model, onnx_model)

        model._compute_dtype = compute_dtype
        model = model.to(compute_dtype)
        model.eval()

        # FIX: keep BatchNorm running statistics in FP32 regardless of compute dtype.
        # model.to(fp16) converts ALL buffers including running_mean / running_var,
        # which only have 10-bit mantissa in FP16.  The BN normalisation
        # (x - mean) / sqrt(var + eps) then suffers catastrophic cancellation for
        # similar activation values, making the per-frame ArcFace embedding less
        # stable and causing visible colour flickering in the swap output.
        # Restoring them to FP32 has zero performance cost: BN eval is a single
        # fused CUDA kernel and the dtype cast is done once at load time.
        if compute_dtype != torch.float32:
            for m in model.modules():
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    m.running_mean = m.running_mean.float()
                    m.running_var = m.running_var.float()
                    if m.weight is not None:
                        m.weight = nn.Parameter(m.weight.float())
                    if m.bias is not None:
                        m.bias = nn.Parameter(m.bias.float())

        total_params = sum(p.numel() for p in model.parameters())
        print(
            f"[IResNet50Torch] Loaded {total_params:,} parameters"
            f" | compute dtype: {compute_dtype}"
        )
        model._visomaster_onnx_path = str(onnx_path)
        return model


# ---------------------------------------------------------------------------
# Weight-loading helpers
# ---------------------------------------------------------------------------


def _np(init):
    """Extract float32 numpy array from an ONNX TensorProto initialiser."""
    import numpy as np

    if init.raw_data:
        return np.frombuffer(init.raw_data, dtype=np.float32).copy()
    return np.array(init.float_data, dtype=np.float32)


def _conv_modules_in_forward_order(model: IResNet50Torch) -> List[nn.Conv2d]:
    """Return all 53 Conv2d modules in ONNX topological (= forward execution) order.

    Breakdown:
         1  stem conv1
         7  layer1  (3 blocks: block0 conv1+conv2+downsample, blocks1-2 conv1+conv2)
         9  layer2  (4 blocks: block0 +downsample, blocks1-3 plain)
        29  layer3  (14 blocks: block0 +downsample, blocks1-13 plain)
         7  layer4  (3 blocks: block0 +downsample, blocks1-2 plain)
        --
        53  total
    """
    mods: List[nn.Conv2d] = [model.conv1]

    for layer in (model.layer1, model.layer2, model.layer3, model.layer4):
        for block in layer:
            mods.append(block.conv1)
            mods.append(block.conv2)
            if block.downsample is not None:
                mods.append(block.downsample)

    assert len(mods) == 53, f"Expected 53 Conv2d, got {len(mods)}"
    return mods


def _prelu_modules_in_forward_order(model: IResNet50Torch) -> List[nn.PReLU]:
    """Return all 25 PReLU modules in ONNX topological (= forward execution) order.

    Breakdown:
         1  stem prelu0
         3  layer1 (1 per block)
         4  layer2
        14  layer3
         3  layer4
        --
        25  total
    """
    mods: List[nn.PReLU] = [model.prelu0]

    for layer in (model.layer1, model.layer2, model.layer3, model.layer4):
        for block in layer:
            mods.append(block.prelu)

    assert len(mods) == 25, f"Expected 25 PReLU, got {len(mods)}"
    return mods


def _load_all_params(model: IResNet50Torch, onnx_model) -> None:
    """Load all weights from the ONNX model into model (in float32; caller converts).

    Strategy:
      - Conv2d weights/biases: loaded positionally by matching ONNX Conv nodes in
        topological order to _conv_modules_in_forward_order().
      - PReLU slopes: loaded positionally by matching ONNX PRelu nodes in
        topological order to _prelu_modules_in_forward_order().
      - BatchNorm, FC (Linear), BN_features: loaded by name from state_dict.
    """

    init_map = {init.name: init for init in onnx_model.graph.initializer}
    state = model.state_dict()

    conv_mods = _conv_modules_in_forward_order(model)
    prelu_mods = _prelu_modules_in_forward_order(model)
    conv_idx = 0
    prelu_idx = 0

    # ── Step 1: Positional Conv + PReLU loading ──────────────────────────────
    for node in onnx_model.graph.node:
        if node.op_type == "Conv":
            if conv_idx >= len(conv_mods):
                raise RuntimeError(
                    f"More Conv nodes in ONNX than expected ({len(conv_mods)})"
                )
            mod = conv_mods[conv_idx]
            conv_idx += 1

            w_init = init_map[node.input[1]]
            w = _np(w_init).reshape(list(w_init.dims))
            with torch.no_grad():
                mod.weight.copy_(torch.from_numpy(w))

            if len(node.input) > 2 and node.input[2]:
                b = _np(init_map[node.input[2]])
                with torch.no_grad():
                    mod.bias.copy_(torch.from_numpy(b))
            else:
                with torch.no_grad():
                    mod.bias.zero_()

        elif node.op_type == "PRelu":
            if prelu_idx >= len(prelu_mods):
                raise RuntimeError(
                    f"More PRelu nodes in ONNX than expected ({len(prelu_mods)})"
                )
            mod = prelu_mods[prelu_idx]
            prelu_idx += 1

            slope_init = init_map[node.input[1]]
            slope = _np(slope_init).flatten()  # [C,1,1] → [C]
            with torch.no_grad():
                mod.weight.copy_(torch.from_numpy(slope))

    if conv_idx != len(conv_mods):
        raise RuntimeError(
            f"Conv count mismatch: expected {len(conv_mods)}, found {conv_idx}"
        )
    if prelu_idx != len(prelu_mods):
        raise RuntimeError(
            f"PRelu count mismatch: expected {len(prelu_mods)}, found {prelu_idx}"
        )

    # ── Step 2: Named BatchNorm loading ──────────────────────────────────────
    bn_suffixes = (".weight", ".bias", ".running_mean", ".running_var")
    for init_name, init in init_map.items():
        if not any(init_name.endswith(s) for s in bn_suffixes):
            continue
        if init_name not in state:
            continue
        data = _np(init).reshape(list(init.dims) if init.dims else [-1])
        with torch.no_grad():
            state[init_name].copy_(torch.from_numpy(data))

    # ── Step 3: Named FC (Gemm) loading ──────────────────────────────────────
    for node in onnx_model.graph.node:
        if node.op_type != "Gemm":
            continue
        w_init = init_map[node.input[1]]
        w = _np(w_init).reshape(list(w_init.dims))
        with torch.no_grad():
            model.fc.weight.copy_(torch.from_numpy(w))
        if len(node.input) > 2 and node.input[2]:
            b = _np(init_map[node.input[2]])
            with torch.no_grad():
                model.fc.bias.copy_(torch.from_numpy(b))
        break  # only one Gemm node


# ---------------------------------------------------------------------------
# CUDA graph runner
# ---------------------------------------------------------------------------


class _CapturedGraph:
    """Single fixed-size CUDA graph for (1, 3, 112, 112) input."""

    def __init__(self, model: IResNet50Torch, warmup: int = 3) -> None:
        device = next(model.parameters()).device
        self._inp = torch.zeros(1, 3, 112, 112, dtype=torch.float32, device=device)

        with torch.no_grad():
            for _ in range(warmup):
                _ = model(self._inp)

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
        self._inp.copy_(x)
        self._graph.replay()
        return self._out.clone()


def build_cuda_graph_runner(
    model: IResNet50Torch,
    warmup: int = 3,
    torch_compile: bool = False,
) -> _CapturedGraph:
    """Capture a CUDA graph for model and return a callable runner.

    Args:
        torch_compile: If True, wrap the model with ``torch.compile`` before
                       capturing the CUDA graph.  Requires Triton; adds ~30 s
                       one-time compile overhead.
    """
    if torch_compile:
        try:
            from custom_kernels.compile_utils import apply_torch_compile
            device = next(model.parameters()).device
            example_inp = torch.zeros(1, 3, 112, 112, dtype=torch.float32, device=device)
            # reduce-overhead avoids the Triton MLIR AV crash (0xC0000005 on Windows sm_89)
            # that mode='default' triggers in the subprocess ptxas optimizer.
            compiled = apply_torch_compile(model, example_inp, compile_mode="reduce-overhead")
            print("[w600k_r50] torch.compile reduce-overhead done.")
            return compiled  # CUDA graph on top of torch.compile fails on Windows
        except Exception as e:
            print(f"[w600k_r50] torch.compile failed ({e!s:.120}), falling back to CUDA graph.")

    return _CapturedGraph(model, warmup)
