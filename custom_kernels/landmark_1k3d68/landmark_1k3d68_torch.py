"""
landmark_1k3d68_torch.py — FP16 PyTorch reimplementation of 1k3d68.onnx
========================================================================

Architecture (reverse-engineered from ONNX topological order):

  Input : (1, 3, 192, 192) float32 — ImageNet-normalised face crop
  Output: (1, 3309)         float32 — raw regressed values; caller takes
                                       the last 68×3 slice for 3-D landmarks

  Backbone : Pre-activation ResNet-50 (MXNet / InsightFace style)
               – stage depths  3-4-6-3
               – channel widths 64→256 → 512 → 1024 → 2048
               – BatchNorm NOT folded; 18 live BN layers
  Head     : Conv2d(2048→256, 3×3, stride=2) → Flatten → Linear(2304→3309)

Pre-activation bottleneck order
--------------------------------
  BN(in_ch) → ReLU → Conv1×1(bias) → ReLU → Conv3×3(stride, bias)
           → ReLU → Conv1×1(no-bias) + shortcut(Conv1×1 from shared ReLU,
             no-bias) or identity → Add

Weight loading strategy
-----------------------
  Conv weights  : positional — 54 Conv2d modules listed in the same
                  topological order the ONNX graph visits them.
  BN parameters : named — MXNet-style ONNX initialiser names
                  (e.g. 'stage1_unit1_bn1_gamma') mapped explicitly
                  to each nn.BatchNorm2d.
  FC parameters : named — 'fc1_weight' / 'fc1_bias'.

No Triton kernels are required — speed-up comes from FP16 cuDNN
TensorCore convolutions + a single captured CUDA graph.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Pre-activation bottleneck
# ---------------------------------------------------------------------------


class _PreActBottleneck(nn.Module):
    """
    MXNet-style pre-activation bottleneck:

        BN(in_ch) → ReLU → Conv1×1 → ReLU → Conv3×3(stride)
                 → ReLU → Conv1×1(no-bias) [+ shortcut] → Add

    For the first unit of each stage (has_skip_conv=True) the shortcut is
    a strided Conv1×1 applied to the shared BN+ReLU output (h), so both
    branches start from the same activations — matching the ONNX graph.

    For subsequent units the shortcut is a plain identity (x itself, not h).
    """

    def __init__(
        self,
        in_ch: int,
        mid_ch: int,
        out_ch: int,
        stride: int = 1,
        has_skip_conv: bool = False,
    ) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_ch, eps=2e-5, momentum=0.9)
        self.conv1 = nn.Conv2d(in_ch, mid_ch, 1, bias=True)
        self.conv2 = nn.Conv2d(mid_ch, mid_ch, 3, stride=stride, padding=1, bias=True)
        self.conv3 = nn.Conv2d(mid_ch, out_ch, 1, bias=False)
        self.shortcut: Optional[nn.Conv2d] = (
            nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False)
            if has_skip_conv
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Shared BN + ReLU — used by both main path and (when present) shortcut
        h = F.relu(self.bn1(x), inplace=True)
        out = F.relu(self.conv1(h), inplace=True)
        out = F.relu(self.conv2(out), inplace=True)
        out = self.conv3(out)
        sc = self.shortcut(h) if self.shortcut is not None else x
        return out + sc


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------


class Landmark1k3d68Torch(nn.Module):
    """
    FP16 PyTorch reimplementation of 1k3d68.onnx.

    compute_dtype : torch.float16 (default) or torch.float32.
    Input  always accepted as float32; cast happens inside forward().
    Output always returned as float32 — (1, 3309).
    """

    def __init__(self, compute_dtype: torch.dtype = torch.float16) -> None:
        super().__init__()
        self._compute_dtype = compute_dtype

        # ── Input BatchNorm (applied to raw 3-channel normalised input) ───
        self.input_bn = nn.BatchNorm2d(3, eps=2e-5, momentum=0.9)

        # ── Stem ──────────────────────────────────────────────────────────
        self.stem_conv = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=True)
        self.stem_pool = nn.MaxPool2d(3, stride=2, padding=1)

        # ── Stage 1: 3 units, 64 → 256, no spatial downsampling ──────────
        self.stage1 = nn.Sequential(
            _PreActBottleneck(64, 64, 256, stride=1, has_skip_conv=True),
            _PreActBottleneck(256, 64, 256, stride=1, has_skip_conv=False),
            _PreActBottleneck(256, 64, 256, stride=1, has_skip_conv=False),
        )

        # ── Stage 2: 4 units, 256 → 512, stride-2 in first unit ──────────
        self.stage2 = nn.Sequential(
            _PreActBottleneck(256, 128, 512, stride=2, has_skip_conv=True),
            _PreActBottleneck(512, 128, 512, stride=1, has_skip_conv=False),
            _PreActBottleneck(512, 128, 512, stride=1, has_skip_conv=False),
            _PreActBottleneck(512, 128, 512, stride=1, has_skip_conv=False),
        )

        # ── Stage 3: 6 units, 512 → 1024, stride-2 in first unit ─────────
        self.stage3 = nn.Sequential(
            _PreActBottleneck(512, 256, 1024, stride=2, has_skip_conv=True),
            _PreActBottleneck(1024, 256, 1024, stride=1, has_skip_conv=False),
            _PreActBottleneck(1024, 256, 1024, stride=1, has_skip_conv=False),
            _PreActBottleneck(1024, 256, 1024, stride=1, has_skip_conv=False),
            _PreActBottleneck(1024, 256, 1024, stride=1, has_skip_conv=False),
            _PreActBottleneck(1024, 256, 1024, stride=1, has_skip_conv=False),
        )

        # ── Stage 4: 3 units, 1024 → 2048, stride-2 in first unit ────────
        self.stage4 = nn.Sequential(
            _PreActBottleneck(1024, 512, 2048, stride=2, has_skip_conv=True),
            _PreActBottleneck(2048, 512, 2048, stride=1, has_skip_conv=False),
            _PreActBottleneck(2048, 512, 2048, stride=1, has_skip_conv=False),
        )

        # ── Final BN (bn8) ────────────────────────────────────────────────
        self.final_bn = nn.BatchNorm2d(2048, eps=2e-5, momentum=0.9)

        # ── Head: Conv3×3 stride-2 → ReLU → Flatten → FC ─────────────────
        # After stage4: 192/32 = 6 × 6 spatial; after head_conv(s=2): 3 × 3
        # Flatten → 256 × 3 × 3 = 2304  (matches Gemm weight (3309, 2304))
        self.head_conv = nn.Conv2d(2048, 256, 3, stride=2, padding=1, bias=True)
        self.fc = nn.Linear(2304, 3309, bias=True)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self._compute_dtype)

        x = self.input_bn(x)
        x = F.relu(self.stem_conv(x), inplace=True)
        x = self.stem_pool(x)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = F.relu(self.final_bn(x), inplace=True)

        x = F.relu(self.head_conv(x), inplace=True)
        x = x.flatten(1)  # (1, 256*3*3) = (1, 2304)
        x = self.fc(x)  # (1, 3309)

        return x.float()

    # ------------------------------------------------------------------
    @classmethod
    def from_onnx(
        cls,
        onnx_path: str | Path,
        compute_dtype: torch.dtype = torch.float16,
        verbose: bool = False,
    ) -> "Landmark1k3d68Torch":
        """Build and return a Landmark1k3d68Torch with weights from *onnx_path*."""
        import onnx as _onnx  # type: ignore

        onnx_path = Path(onnx_path)
        print(f"[1k3d68Torch] Loading ONNX from {onnx_path} …")
        onnx_model = _onnx.load(str(onnx_path))

        model = cls(compute_dtype=torch.float32)  # load in fp32, cast later
        _load_all_params(model, onnx_model, verbose=verbose)

        model._compute_dtype = compute_dtype
        model = model.to(compute_dtype).eval()

        n_params = sum(p.numel() for p in model.parameters())
        print(
            f"[1k3d68Torch] {n_params:,} parameters loaded"
            f" | compute dtype: {compute_dtype}"
        )
        model._visomaster_onnx_path = str(onnx_path)
        return model


# ---------------------------------------------------------------------------
# Weight-loading helpers
# ---------------------------------------------------------------------------


def _conv_modules_in_forward_order(model: Landmark1k3d68Torch) -> List[nn.Conv2d]:
    """
    Return all 54 Conv2d modules in the same topological order the ONNX
    graph visits Conv nodes.

    Index layout
    ────────────
     [0]        stem_conv
     [1–4]      stage1[0]:  conv1, conv2, conv3, shortcut
     [5–7]      stage1[1]:  conv1, conv2, conv3
     [8–10]     stage1[2]:  conv1, conv2, conv3
     [11–14]    stage2[0]:  conv1, conv2, conv3, shortcut
     [15–17]    stage2[1]:  conv1, conv2, conv3
     [18–20]    stage2[2]:  conv1, conv2, conv3
     [21–23]    stage2[3]:  conv1, conv2, conv3
     [24–27]    stage3[0]:  conv1, conv2, conv3, shortcut
     [28–30]    stage3[1]:  conv1, conv2, conv3
     [31–33]    stage3[2]:  conv1, conv2, conv3
     [34–36]    stage3[3]:  conv1, conv2, conv3
     [37–39]    stage3[4]:  conv1, conv2, conv3
     [40–42]    stage3[5]:  conv1, conv2, conv3
     [43–46]    stage4[0]:  conv1, conv2, conv3, shortcut
     [47–49]    stage4[1]:  conv1, conv2, conv3
     [50–52]    stage4[2]:  conv1, conv2, conv3
     [53]       head_conv
    """
    mods: List[nn.Conv2d] = []

    mods.append(model.stem_conv)

    for stage in (model.stage1, model.stage2, model.stage3, model.stage4):
        for unit in stage:
            mods.append(unit.conv1)
            mods.append(unit.conv2)
            mods.append(unit.conv3)
            if unit.shortcut is not None:
                mods.append(unit.shortcut)

    mods.append(model.head_conv)

    assert len(mods) == 54, f"Expected 54 Conv2d modules, got {len(mods)}"
    return mods


def _load_all_params(
    model: Landmark1k3d68Torch,
    onnx_model,  # onnx.ModelProto
    verbose: bool = False,
) -> None:
    """
    Load all weights from the ONNX model:
      1. Conv2d weights + biases — positionally (by ONNX Conv node order).
      2. BatchNorm parameters   — by MXNet-style ONNX initialiser names.
      3. Linear (FC) weights    — by ONNX initialiser names.
    """
    from onnx import numpy_helper  # type: ignore

    init_map: dict = {
        init.name: numpy_helper.to_array(init) for init in onnx_model.graph.initializer
    }

    # ── 1. Conv2d (positional) ────────────────────────────────────────
    conv_mods = _conv_modules_in_forward_order(model)
    conv_idx = 0
    for node in onnx_model.graph.node:
        if node.op_type != "Conv":
            continue
        mod = conv_mods[conv_idx]
        conv_idx += 1

        w = init_map[node.input[1]]
        mod.weight.data.copy_(torch.from_numpy(w.copy()))

        if len(node.input) > 2 and node.input[2] in init_map:
            b = init_map[node.input[2]]
            mod.bias.data.copy_(torch.from_numpy(b.copy()))  # type: ignore[union-attr]

    assert conv_idx == 54, f"Expected 54 Conv nodes in ONNX, found {conv_idx}"

    # ── 2. BatchNorm (by MXNet name) ──────────────────────────────────
    # Maps (PyTorch BN module, ONNX name prefix)
    bn_map: List[Tuple[nn.BatchNorm2d, str]] = [
        (model.input_bn, "bn_data"),
        (model.stage1[0].bn1, "stage1_unit1_bn1"),
        (model.stage1[1].bn1, "stage1_unit2_bn1"),
        (model.stage1[2].bn1, "stage1_unit3_bn1"),
        (model.stage2[0].bn1, "stage2_unit1_bn1"),
        (model.stage2[1].bn1, "stage2_unit2_bn1"),
        (model.stage2[2].bn1, "stage2_unit3_bn1"),
        (model.stage2[3].bn1, "stage2_unit4_bn1"),
        (model.stage3[0].bn1, "stage3_unit1_bn1"),
        (model.stage3[1].bn1, "stage3_unit2_bn1"),
        (model.stage3[2].bn1, "stage3_unit3_bn1"),
        (model.stage3[3].bn1, "stage3_unit4_bn1"),
        (model.stage3[4].bn1, "stage3_unit5_bn1"),
        (model.stage3[5].bn1, "stage3_unit6_bn1"),
        (model.stage4[0].bn1, "stage4_unit1_bn1"),
        (model.stage4[1].bn1, "stage4_unit2_bn1"),
        (model.stage4[2].bn1, "stage4_unit3_bn1"),
        (model.final_bn, "bn8"),
    ]
    for bn_mod, prefix in bn_map:
        bn_mod.weight.data.copy_(torch.from_numpy(init_map[f"{prefix}_gamma"].copy()))
        bn_mod.bias.data.copy_(torch.from_numpy(init_map[f"{prefix}_beta"].copy()))
        bn_mod.running_mean.copy_(
            torch.from_numpy(init_map[f"{prefix}_moving_mean"].copy())
        )
        bn_mod.running_var.copy_(
            torch.from_numpy(init_map[f"{prefix}_moving_var"].copy())
        )

    # ── 3. FC (by name) ───────────────────────────────────────────────
    model.fc.weight.data.copy_(torch.from_numpy(init_map["fc1_weight"].copy()))
    model.fc.bias.data.copy_(torch.from_numpy(init_map["fc1_bias"].copy()))

    if verbose:
        n = sum(p.numel() for p in model.parameters())
        print(f"[1k3d68Torch] _load_all_params: {n:,} parameters populated.")


# ---------------------------------------------------------------------------
# CUDA graph runner
# ---------------------------------------------------------------------------


class Landmark1k3d68CUDAGraphRunner:
    """
    Wraps Landmark1k3d68Torch in a CUDA graph for minimal kernel-launch overhead.

    The model is called with a fixed (1, 3, 192, 192) buffer; output is cloned
    so every call returns an independent tensor.
    """

    def __init__(self, model: Landmark1k3d68Torch, warmup: int = 3) -> None:
        device = next(model.parameters()).device
        self._inp = torch.zeros(1, 3, 192, 192, dtype=torch.float32, device=device)

        # Warm-up: primes cuDNN auto-tuner and Triton JIT (if used)
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
        """x: (1, 3, 192, 192) float32 — same device as model."""
        self._inp.copy_(x, non_blocking=True)
        self._graph.replay()
        return self._out.clone()


def build_cuda_graph_runner(
    model: Landmark1k3d68Torch,
    warmup: int = 3,
    torch_compile: bool = False,
) -> "Landmark1k3d68CUDAGraphRunner | Landmark1k3d68Torch":
    """
    Capture a CUDA graph for fixed (1, 3, 192, 192) input.
    Falls back to eager model if capture fails (e.g. CPU-only environment).

    Args:
        model         : Landmark1k3d68Torch on CUDA, already in eval() mode.
        warmup        : number of warm-up passes before graph capture.
        torch_compile : if True, apply torch.compile (mode='default') to the
                        model before CUDA graph capture.  Triggers Triton JIT
                        on first call (~30 s); compiled kernels are then
                        captured in the graph.
    """
    if torch_compile:
        try:
            from custom_kernels.compile_utils import apply_torch_compile
            device = next(model.parameters()).device
            example_inp = torch.zeros((1, 3, 192, 192), dtype=torch.float32, device=device)
            compiled = apply_torch_compile(model, example_inp, compile_mode="default")
            print("[landmark_1k3d68] torch.compile default done.")
            return compiled
        except Exception as e:
            print(f"[landmark_1k3d68] torch.compile failed ({e!s:.120}), falling back to CUDA graph.")
    try:
        runner = Landmark1k3d68CUDAGraphRunner(model, warmup=warmup)
        print("[1k3d68Torch] CUDA graph captured successfully.")
        return runner
    except Exception as exc:
        print(f"[1k3d68Torch] CUDA graph capture failed — using eager FP16: {exc}")
        return model
