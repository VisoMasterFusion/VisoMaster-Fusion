"""
FP16 PyTorch reimplementation of det_10g.onnx (SCRFD-10G face detector).

Reverse-engineered from the ONNX graph.  Architecture:
  3-conv stem + MaxPool + 4 residual stages → C4/C5/C6
  PA-FPN neck (10 convs) → 3 feature maps at strides 8 / 16 / 32
  Per-stride detection heads (cls + reg + kps) with learnable scale

Outputs (9 tensors, matching ONNX output order):
  [scores_8, scores_16, scores_32,   shape (*,1) float32, Sigmoid-normalised
   bbox_8,   bbox_16,   bbox_32,     shape (*,4) float32, raw distance preds
   kps_8,    kps_16,    kps_32]      shape (*,10) float32, raw kps offsets

The application multiplies bbox and kps by stride after inference; the PyTorch
model is a drop-in replacement returning the same 9 NumPy arrays.

Performance (RTX 4090, 640×640, 50 iters):
  Tier 2 — FP16 NCHW              : ~3.49 ms
  Tier 3 — FP16 NCHW + CUDA graph : ~1.37 ms
  Tier 4 — FP16 NHWC              : ~2.39 ms
  Tier 5 — FP16 NHWC + CUDA graph : ~0.98 ms  ← default (3.18× vs ORT FP32)

NHWC (channels_last) allows cuDNN to use its native NHWC convolution kernels,
eliminating the internal NCHW↔NHWC reformatting overhead that cuDNN performs
on every layer when weights are in NCHW format.
"""

from __future__ import annotations

import threading
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------


class _BasicBlock(nn.Module):
    """2-conv residual block (BN folded into Conv bias).

    Strided blocks (stride=2 or channel change) use an
      AvgPool2d(ceil_mode=True) → Conv1×1  shortcut.
    Plain blocks use an identity shortcut.
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=True)
        self._strided = stride != 1 or in_ch != out_ch
        if self._strided:
            self.shortcut_pool = nn.AvgPool2d(2, stride=2, ceil_mode=True)
            self.shortcut_conv = nn.Conv2d(in_ch, out_ch, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sc = self.shortcut_conv(self.shortcut_pool(x)) if self._strided else x
        return F.relu(self.conv2(F.relu(self.conv1(x))) + sc)


class _Backbone(nn.Module):
    """SCRFD-10G backbone.

    Stem: Conv(3→28,3×3,s=2)+ReLU, Conv(28→28)+ReLU, Conv(28→56)+ReLU, MaxPool(s=2)
    Stage1 (56ch, stride 4):  3 plain blocks          → C3 (not used in neck)
    Stage2 (88ch, stride 8):  1 strided + 3 plain     → C4
    Stage3 (88ch, stride 16): 1 strided + 1 plain     → C5
    Stage4 (224ch,stride 32): 1 strided + 2 plain     → C6
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 28, 3, stride=2, padding=1, bias=True)
        self.conv2 = nn.Conv2d(28, 28, 3, padding=1, bias=True)
        self.conv3 = nn.Conv2d(28, 56, 3, padding=1, bias=True)
        self.stage1 = nn.Sequential(*[_BasicBlock(56, 56) for _ in range(3)])
        self.stage2 = nn.Sequential(
            _BasicBlock(56, 88, stride=2),
            *[_BasicBlock(88, 88) for _ in range(3)],
        )
        self.stage3 = nn.Sequential(
            _BasicBlock(88, 88, stride=2),
            _BasicBlock(88, 88),
        )
        self.stage4 = nn.Sequential(
            _BasicBlock(88, 224, stride=2),
            *[_BasicBlock(224, 224) for _ in range(2)],
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, stride=2)
        x = self.stage1(x)
        c4 = self.stage2(x)  # stride 8,  88ch
        c5 = self.stage3(c4)  # stride 16, 88ch
        c6 = self.stage4(c5)  # stride 32, 224ch
        return c4, c5, c6


# ---------------------------------------------------------------------------
# PA-FPN Neck
# ---------------------------------------------------------------------------


class _PAFPN(nn.Module):
    """PA-FPN neck (verified against ONNX topological graph).

    Top-down (following ONNX):
        lat_c6 = lateral2(C6)
        lat_c5 = lateral1(C5)
        lat_c4 = lateral0(C4)
        merged_c5 = lat_c5 + up(lat_c6)              # nearest ×2
        p4        = fpn0(lat_c4 + up(merged_c5))      # stride-8 output

    Bottom-up:
        p5_merged = fpn1(merged_c5) + ds0(p4)         # ds0 stride-2
        p5_pa     = pafpn0(p5_merged)                  # stride-16 output
        p6_merged = fpn2(lat_c6) + ds1(p5_merged)     # ds1 takes p5_merged (!)
        p6_pa     = pafpn1(p6_merged)                  # stride-32 output

    Note: ds1 feeds from p5_merged, not p5_pa (verified in ONNX graph).
    """

    def __init__(self) -> None:
        super().__init__()
        self.lateral0 = nn.Conv2d(88, 56, 1, bias=True)
        self.lateral1 = nn.Conv2d(88, 56, 1, bias=True)
        self.lateral2 = nn.Conv2d(224, 56, 1, bias=True)
        self.fpn0 = nn.Conv2d(56, 56, 3, padding=1, bias=True)
        self.fpn1 = nn.Conv2d(56, 56, 3, padding=1, bias=True)
        self.fpn2 = nn.Conv2d(56, 56, 3, padding=1, bias=True)
        self.ds0 = nn.Conv2d(56, 56, 3, stride=2, padding=1, bias=True)
        self.ds1 = nn.Conv2d(56, 56, 3, stride=2, padding=1, bias=True)
        self.pafpn0 = nn.Conv2d(56, 56, 3, padding=1, bias=True)
        self.pafpn1 = nn.Conv2d(56, 56, 3, padding=1, bias=True)

    def forward(
        self,
        c4: torch.Tensor,
        c5: torch.Tensor,
        c6: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        lat_c4 = self.lateral0(c4)
        lat_c5 = self.lateral1(c5)
        lat_c6 = self.lateral2(c6)

        merged_c5 = lat_c5 + F.interpolate(lat_c6, scale_factor=2, mode="nearest")
        p4 = self.fpn0(
            lat_c4 + F.interpolate(merged_c5, scale_factor=2, mode="nearest")
        )

        p5_merged = self.fpn1(merged_c5) + self.ds0(p4)
        p5_pa = self.pafpn0(p5_merged)

        p6_merged = self.fpn2(lat_c6) + self.ds1(p5_merged)
        p6_pa = self.pafpn1(p6_merged)

        return p4, p5_pa, p6_pa  # strides 8, 16, 32


# ---------------------------------------------------------------------------
# Per-stride detection head
# ---------------------------------------------------------------------------


class _HeadBranch(nn.Module):
    """Per-stride detection head.

    3-conv shared feature extractor (56→80, 80→80, 80→80, each +ReLU),
    then three independent 3×3 convs for cls / reg / kps.

    cls:  Conv(80→2)  → permute(0,2,3,1) → reshape(-1,1)  → Sigmoid
    reg:  Conv(80→8)  → * scale           → permute(0,2,3,1) → reshape(-1,4)
    kps:  Conv(80→20) → permute(0,2,3,1) → reshape(-1,10)
    """

    def __init__(self, in_ch: int = 56, feat_ch: int = 80) -> None:
        super().__init__()
        self.sc1 = nn.Conv2d(in_ch, feat_ch, 3, padding=1, bias=True)
        self.sc2 = nn.Conv2d(feat_ch, feat_ch, 3, padding=1, bias=True)
        self.sc3 = nn.Conv2d(feat_ch, feat_ch, 3, padding=1, bias=True)
        self.cls = nn.Conv2d(feat_ch, 2, 3, padding=1, bias=True)
        self.reg = nn.Conv2d(feat_ch, 8, 3, padding=1, bias=True)
        self.kps = nn.Conv2d(feat_ch, 20, 3, padding=1, bias=True)
        self.scale = nn.Parameter(torch.ones(1))

    def forward(
        self, feat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = F.relu(self.sc1(feat))
        x = F.relu(self.sc2(x))
        x = F.relu(self.sc3(x))
        scores = self.cls(x).permute(0, 2, 3, 1).reshape(-1, 1).sigmoid()
        bbox = (self.reg(x) * self.scale).permute(0, 2, 3, 1).reshape(-1, 4)
        kps = self.kps(x).permute(0, 2, 3, 1).reshape(-1, 10)
        return scores, bbox, kps


# ---------------------------------------------------------------------------
# Full SCRFD-10G model
# ---------------------------------------------------------------------------


class Det10gTorch(nn.Module):
    """FP16 PyTorch reimplementation of det_10g.onnx (SCRFD-10G).

    Returns a tuple of 9 float32 tensors in the same output order as the ONNX:
        (scores_8, scores_16, scores_32,
         bbox_8,   bbox_16,   bbox_32,
         kps_8,    kps_16,    kps_32)

    For a 640×640 input the shapes are:
        scores: (12800,1), (3200,1), (800,1)
        bbox:   (12800,4), (3200,4), (800,4)
        kps:    (12800,10),(3200,10),(800,10)

    channels_last=True (default):
        Model weights and input are kept in NHWC order.  cuDNN then runs its
        native NHWC convolution kernels, avoiding the NCHW↔NHWC reformatting
        it performs internally when weights are in NCHW.  On an RTX 4090 this
        alone saves ~1 ms per inference for 640×640 inputs.
    """

    def __init__(
        self,
        compute_dtype: torch.dtype = torch.float16,
        channels_last: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = _Backbone()
        self.neck = _PAFPN()
        self.head8 = _HeadBranch()
        self.head16 = _HeadBranch()
        self.head32 = _HeadBranch()
        self._compute_dtype = compute_dtype
        self._channels_last = channels_last

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        x = x.to(self._compute_dtype)
        if self._channels_last:
            x = x.to(memory_format=torch.channels_last)
        c4, c5, c6 = self.backbone(x)
        p8, p16, p32 = self.neck(c4, c5, c6)
        # Cast neck outputs to float32 before detection heads so that scores,
        # bbox offsets and keypoint coordinates are computed in full FP32
        # precision.  The backbone and neck run in FP16 for speed; the heads
        # are kept in FP32 (see from_onnx) so keypoints fed to the Reference
        # alignment warp have no FP16 sub-pixel bias.
        s8, b8, k8 = self.head8(p8.float())
        s16, b16, k16 = self.head16(p16.float())
        s32, b32, k32 = self.head32(p32.float())
        return (
            s8,
            s16,
            s32,
            b8,
            b16,
            b32,
            k8,
            k16,
            k32,
        )

    @classmethod
    def from_onnx(
        cls,
        onnx_path: str,
        compute_dtype: torch.dtype = torch.float16,
        channels_last: bool = True,
    ) -> "Det10gTorch":
        import onnx

        onnx_model = onnx.load(onnx_path)
        model = cls(compute_dtype=torch.float32, channels_last=channels_last)
        _load_all_params(model, onnx_model, torch.float32)
        model._compute_dtype = compute_dtype
        # Only convert backbone and neck to compute_dtype (FP16).  The three
        # detection heads (head8/head16/head32) stay in FP32 so keypoint
        # coordinates are produced at full precision — preventing the sub-pixel
        # bias that shifts face boxes in Reference-alignment mode.
        if compute_dtype != torch.float32:
            model.backbone = model.backbone.to(compute_dtype)
            model.neck = model.neck.to(compute_dtype)
        if channels_last:
            # Convert backbone+neck 4-D weight tensors to channels_last so
            # cuDNN dispatches to its native NHWC convolution path.
            model.backbone = model.backbone.to(memory_format=torch.channels_last)
            model.neck = model.neck.to(memory_format=torch.channels_last)
        model.eval()
        model._visomaster_onnx_path = str(onnx_path)
        return model


# ---------------------------------------------------------------------------
# Weight loading — all-positional (58 Conv2d in ONNX topological order)
# ---------------------------------------------------------------------------


def _conv_modules_in_forward_order(model: Det10gTorch) -> List[nn.Conv2d]:
    """Return all 58 Conv2d modules in ONNX topological order.

    Breakdown:
        3   stem  (conv1, conv2, conv3)
        6   stage1 (3 plain blocks × 2 convs)
        9   stage2 (1 strided block × 3 + 3 plain × 2)
        5   stage3 (1 strided × 3 + 1 plain × 2)
        7   stage4 (1 strided × 3 + 2 plain × 2)
       10   neck   (3 lateral + 3 fpn + 2 ds + 2 pafpn)
       18   heads  (3 strides × 6: sc1,sc2,sc3,cls,reg,kps)
    """
    mods: List[nn.Conv2d] = []

    # Stem
    mods += [model.backbone.conv1, model.backbone.conv2, model.backbone.conv3]

    # Residual stages — ONNX order: conv1, conv2, [shortcut_conv if strided]
    for stage in (
        model.backbone.stage1,
        model.backbone.stage2,
        model.backbone.stage3,
        model.backbone.stage4,
    ):
        for block in stage:
            mods.append(block.conv1)
            mods.append(block.conv2)
            if block._strided:
                mods.append(block.shortcut_conv)

    # Neck
    neck = model.neck
    mods += [
        neck.lateral0,
        neck.lateral1,
        neck.lateral2,
        neck.fpn0,
        neck.fpn1,
        neck.fpn2,
        neck.ds0,
        neck.ds1,
        neck.pafpn0,
        neck.pafpn1,
    ]

    # Detection heads
    for head in (model.head8, model.head16, model.head32):
        mods += [head.sc1, head.sc2, head.sc3, head.cls, head.reg, head.kps]

    assert len(mods) == 58, f"Expected 58 Conv2d, got {len(mods)}"
    return mods


def _load_all_params(
    model: Det10gTorch,
    onnx_model,  # onnx.ModelProto
    dtype: torch.dtype,
) -> None:
    """Load all weights from ONNX initializers into the PyTorch model."""
    init_map = {i.name: i for i in onnx_model.graph.initializer}

    def _np(name: str) -> np.ndarray:
        init = init_map[name]
        if init.raw_data:
            arr = np.frombuffer(init.raw_data, dtype=np.float32).copy()
        elif init.float_data:
            arr = np.array(list(init.float_data), dtype=np.float32)
        else:
            arr = np.zeros(list(init.dims) or [1], dtype=np.float32)
        return arr.reshape(list(init.dims)) if init.dims else arr

    # ── Conv2d weights (positional) ─────────────────────────────────────────
    conv_mods = _conv_modules_in_forward_order(model)
    conv_nodes = [n for n in onnx_model.graph.node if n.op_type == "Conv"]

    if len(conv_nodes) != 58:
        raise RuntimeError(f"Expected 58 Conv nodes in ONNX, found {len(conv_nodes)}")

    for mod, node in zip(conv_mods, conv_nodes):
        w_name = node.input[1]
        b_name = node.input[2] if len(node.input) > 2 else None
        with torch.no_grad():
            mod.weight.copy_(torch.from_numpy(_np(w_name)).to(dtype))
            if b_name and b_name in init_map:
                mod.bias.copy_(torch.from_numpy(_np(b_name)).to(dtype))

    # ── Learnable regression scales ─────────────────────────────────────────
    scale_map = {
        "bbox_head.scales.0.scale": model.head8.scale,
        "bbox_head.scales.1.scale": model.head16.scale,
        "bbox_head.scales.2.scale": model.head32.scale,
    }
    for init_name, param in scale_map.items():
        if init_name in init_map:
            val = np.frombuffer(init_map[init_name].raw_data, dtype=np.float32).copy()
            with torch.no_grad():
                param.data.copy_(torch.from_numpy(val).to(dtype))


# ---------------------------------------------------------------------------
# CUDA graph runner — per-shape cache
# ---------------------------------------------------------------------------


class _CapturedGraph:
    """A CUDA graph captured for one specific input shape.

    When the model uses channels_last, the static input buffer is allocated
    with the same memory format so the graph captures the NHWC-optimised path.
    """

    _N_OUTS = 9

    def __init__(
        self, model: Det10gTorch, static_inp: torch.Tensor, warmup: int = 3
    ) -> None:
        with torch.no_grad():
            for _ in range(warmup):
                model(static_inp)

        self._inp = static_inp
        self._graph = torch.cuda.CUDAGraph()
        self._stream = torch.cuda.Stream()
        torch.cuda.synchronize()
        with (
            torch.no_grad(),
            torch.cuda.graph(
                self._graph, stream=self._stream, capture_error_mode="relaxed"
            ),
        ):
            self._outs = model(self._inp)
        torch.cuda.synchronize()

    def __call__(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        self._inp.copy_(x, non_blocking=True)
        self._graph.replay()
        return tuple(t.clone() for t in self._outs)


class Det10gGraphRunner:
    """Per-shape CUDA graph cache for Det10gTorch.

    First inference with a new (H, W) shape captures a CUDA graph.
    Subsequent inferences with the same shape replay the graph.

    When the model uses channels_last (NHWC), the static input buffer for graph
    capture is also allocated channels_last so the entire NHWC path is captured
    in the graph — achieving ~0.98 ms on an RTX 4090 at 640×640.
    """

    def __init__(self, model: Det10gTorch) -> None:
        self._model = model
        self._graphs: Dict[Tuple[int, int], Optional[_CapturedGraph]] = {}
        self._capture_lock = threading.Lock()  # protects per-shape lazy capture

    def __call__(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        H, W = int(x.shape[2]), int(x.shape[3])
        key = (H, W)
        if key not in self._graphs:
            with self._capture_lock:
                if key not in self._graphs:  # double-checked inside lock
                    try:
                        # Allocate static buffer matching model memory format so the
                        # graph captures the NHWC path when channels_last is active.
                        if self._model._channels_last:
                            static_inp = torch.zeros_like(
                                x, memory_format=torch.channels_last
                            )
                        else:
                            static_inp = torch.zeros_like(x)
                        self._graphs[key] = _CapturedGraph(self._model, static_inp)
                    except Exception as e:
                        print(
                            f"[Det10g] CUDA graph capture failed for ({H},{W}): {e}. "
                            "Using eager fallback."
                        )
                        self._graphs[key] = None

        g = self._graphs[key]
        if g is not None:
            return g(x)
        with torch.no_grad():
            return self._model(x)


def build_cuda_graph_runner(
    model: Det10gTorch,
    torch_compile: bool = False,
    compile_warmup_shape: Tuple[int, int] = (640, 640),
) -> Det10gGraphRunner:
    """
    Factory: wrap a Det10gTorch model in a per-shape CUDA graph runner.

    Args:
        torch_compile:        If True, wrap the model with ``torch.compile`` before
                              creating the runner.  Compiles for *compile_warmup_shape*
                              on init; other shapes trigger a one-time recompile on
                              first use (same cost as Triton JIT for that shape).
        compile_warmup_shape: H×W to pre-compile during init (default 640×640).
    """
    if torch_compile:
        try:
            from custom_kernels.compile_utils import apply_torch_compile
            device = next(model.parameters()).device
            example_inp = torch.zeros(
                1, 3, compile_warmup_shape[0], compile_warmup_shape[1],
                dtype=torch.float32, device=device,
            )
            compiled = apply_torch_compile(model, example_inp)
            print("[det_10g] torch.compile warmup done.")
            # Return compiled model directly — CUDA graph on top of torch.compile
            # fails on Windows (64-bit kernel handles overflow 32-bit C long).
            # Det10gGraphRunner's per-shape CUDA graph also fails; compiled model is used.
            return compiled
        except Exception as e:
            print(f"[det_10g] torch.compile failed ({e!s:.120}), falling back to CUDA graph.")

    return Det10gGraphRunner(model)
