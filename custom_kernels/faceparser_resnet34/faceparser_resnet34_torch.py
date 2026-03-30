"""
FP16 PyTorch reimplementation of faceparser_resnet34.onnx
(BiSeNet v1 with ResNet-34 backbone, 19-class face parsing).

Architecture: ResNet-34 backbone + Context Path (ARM32 + ARM16) + FFM + three output heads.
Input:  (1, 3, 512, 512) float32 — ImageNet-normalised
Output: (1, 19, 512, 512) float32 — class logits (primary head only)

All 52 Conv2d weights are loaded positionally from the ONNX graph's topological Conv
node order → PyTorch forward-execution order.  BN is fully folded into Conv bias.
No Triton kernels required (pure Conv/ReLU/Sigmoid — cuDNN TensorCore dispatched).

Usage:
    model  = FaceParserResnet34Torch.from_onnx("model_assets/faceparser_resnet34.onnx").cuda().eval()
    runner = build_cuda_graph_runner(model)   # fixed 512×512 → single captured graph

    with torch.no_grad():
        logits = runner(x)   # (1,3,512,512) float32 → (1,19,512,512) float32
    labels = logits.argmax(dim=1).squeeze(0)  # (512,512) long
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Shared primitives
# ---------------------------------------------------------------------------


class _CBR(nn.Module):
    """
    BN-folded Conv2d + optional in-place ReLU.
    The Conv2d is stored as *.conv* so positional weight loading works cleanly.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int,
        s: int = 1,
        p: int = 0,
        act: bool = True,
        bias: bool = True,
    ):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, p, bias=bias)
        self.act = act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return F.relu(x, inplace=True) if self.act else x


# ---------------------------------------------------------------------------
# ResNet-34 backbone
# ---------------------------------------------------------------------------


class _BasicBlock(nn.Module):
    """ResNet-34 BasicBlock (BN folded into conv bias)."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = _CBR(in_ch, out_ch, 3, stride, 1, act=True)
        self.conv2 = _CBR(out_ch, out_ch, 3, 1, 1, act=False)
        # 1×1 downsampling shortcut when stride>1 or channel change
        self.downsample: Optional[_CBR] = (
            _CBR(in_ch, out_ch, 1, stride, 0, act=False)
            if stride != 1 or in_ch != out_ch
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv2(self.conv1(x))
        residual = self.downsample(x) if self.downsample is not None else x
        return F.relu(out + residual, inplace=True)


class _Backbone(nn.Module):
    """ResNet-34: stem + layer1–4.  Returns (c3, c4, c5) = (layer2, layer3, layer4)."""

    def __init__(self):
        super().__init__()
        # Stem: Conv 7×7 stride-2 + MaxPool stride-2 → 128×128
        self.conv1 = _CBR(3, 64, 7, s=2, p=3, act=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        # Layer 1: 3 × BasicBlock(64→64, stride=1)
        self.layer1 = nn.Sequential(
            _BasicBlock(64, 64),
            _BasicBlock(64, 64),
            _BasicBlock(64, 64),
        )
        # Layer 2: 4 × BasicBlock(64→128 first, then 128→128)  →  64×64
        self.layer2 = nn.Sequential(
            _BasicBlock(64, 128, stride=2),
            _BasicBlock(128, 128),
            _BasicBlock(128, 128),
            _BasicBlock(128, 128),
        )
        # Layer 3: 6 × BasicBlock(128→256 first, then 256→256)  →  32×32
        self.layer3 = nn.Sequential(
            _BasicBlock(128, 256, stride=2),
            _BasicBlock(256, 256),
            _BasicBlock(256, 256),
            _BasicBlock(256, 256),
            _BasicBlock(256, 256),
            _BasicBlock(256, 256),
        )
        # Layer 4: 3 × BasicBlock(256→512 first, then 512→512)  →  16×16
        self.layer4 = nn.Sequential(
            _BasicBlock(256, 512, stride=2),
            _BasicBlock(512, 512),
            _BasicBlock(512, 512),
        )

    def forward(self, x: torch.Tensor):
        x = self.maxpool(self.conv1(x))  # 64ch, 128×128
        x = self.layer1(x)  # 64ch, 128×128  (not used by neck)
        c3 = self.layer2(x)  # 128ch, 64×64   → FFM spatial path
        c4 = self.layer3(c3)  # 256ch, 32×32   → ARM16
        c5 = self.layer4(c4)  # 512ch, 16×16   → ARM32 + conv_avg
        return c3, c4, c5


# ---------------------------------------------------------------------------
# Context Path: ARM + FFM
# ---------------------------------------------------------------------------


class _ARM(nn.Module):
    """
    Attention Refinement Module.
    conv_block(3×3) → channel-SE attention (avgpool→1×1→sigmoid) → Mul(feat, attn).
    The Add with global context is performed in _ContextPath.forward().
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv_block = _CBR(in_ch, out_ch, 3, 1, 1, act=True)
        # 1×1 conv for channel attention — has bias (BN folded)
        self.attention = nn.Conv2d(out_ch, out_ch, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv_block(x)
        attn = torch.sigmoid(self.attention(F.adaptive_avg_pool2d(feat, 1)))
        return feat * attn


class _FFM(nn.Module):
    """
    Feature Fusion Module.
    Concat(spatial [128ch], context [128ch]) → 256ch
    → Conv1×1 (256→256) + SE attention (256→64→256) → residual add.
    """

    def __init__(self):
        super().__init__()
        # ffm.conv_block: 1×1 conv + ReLU (has bias — BN folded)
        self.conv_block = _CBR(256, 256, 1, act=True, bias=True)
        # SE squeeze/excite — no bias (exported with bias=False)
        self.conv1 = nn.Conv2d(256, 64, 1, bias=False)  # ffm.conv1.weight
        self.conv2 = nn.Conv2d(64, 256, 1, bias=False)  # ffm.conv2.weight

    def forward(self, spatial: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        x = torch.cat([spatial, context], 1)  # (B, 256, H, W)
        feat = self.conv_block(x)
        # SE attention
        attn = F.adaptive_avg_pool2d(feat, 1)
        attn = F.relu(self.conv1(attn), inplace=True)
        attn = torch.sigmoid(self.conv2(attn))
        return feat + feat * attn


class _ContextPath(nn.Module):
    """
    BiSeNet Context Path: conv_avg (global context) + ARM32 + ARM16 + FFM.
    Returns (ffm_out [256ch,64×64], head16_out [128ch,64×64], head32_out [128ch,32×32]).
    """

    def __init__(self):
        super().__init__()
        # Global context: GlobalAvgPool(layer4) → Conv1×1(512→128) + ReLU
        self.conv_avg = _CBR(512, 128, 1, act=True)
        # ARM modules
        self.arm32 = _ARM(512, 128)
        self.conv_head32 = _CBR(128, 128, 3, 1, 1, act=True)
        self.arm16 = _ARM(256, 128)
        self.conv_head16 = _CBR(128, 128, 3, 1, 1, act=True)
        # Feature Fusion Module
        self.ffm = _FFM()

    def forward(self, c3, c4, c5):
        # c3 = layer2 output: 128ch, 64×64
        # c4 = layer3 output: 256ch, 32×32
        # c5 = layer4 output: 512ch, 16×16

        # Global context: GAP → conv → resize to layer4 spatial
        ctx = self.conv_avg(F.adaptive_avg_pool2d(c5, 1))  # 128ch, 1×1
        ctx = F.interpolate(ctx, size=c5.shape[2:], mode="nearest")  # 128ch, 16×16

        # ARM32: attend layer4, add global context → resize ×2 → conv_head32
        arm32_out = self.arm32(c5) + ctx  # 128ch, 16×16
        head32_out = self.conv_head32(
            F.interpolate(arm32_out, scale_factor=2, mode="nearest")
        )  # 128ch, 32×32

        # ARM16: attend layer3, add arm32 context → resize ×2 → conv_head16
        arm16_out = self.arm16(c4) + head32_out  # 128ch, 32×32
        head16_out = self.conv_head16(
            F.interpolate(arm16_out, scale_factor=2, mode="nearest")
        )  # 128ch, 64×64

        # FFM: fuse layer2 (spatial path) with head16 (context path)
        ffm_out = self.ffm(c3, head16_out)  # 256ch, 64×64

        return ffm_out, head16_out, head32_out


# ---------------------------------------------------------------------------
# Output heads
# ---------------------------------------------------------------------------


class _OutputHead(nn.Module):
    """
    conv_block(3×3, in→mid, bias) + ReLU + Conv1×1(mid→19, no bias) + Resize to 512×512.
    """

    def __init__(self, in_ch: int, mid_ch: int, num_classes: int = 19):
        super().__init__()
        self.conv_block = _CBR(in_ch, mid_ch, 3, 1, 1, act=True, bias=True)
        self.conv = nn.Conv2d(mid_ch, num_classes, 1, bias=False)  # no bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(self.conv_block(x))
        return F.interpolate(x, size=(512, 512), mode="bilinear", align_corners=True)


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------


class FaceParserResnet34Torch(nn.Module):
    """
    FP16 BiSeNet-v1 / ResNet-34 face parser.

    Returns only the *primary* output (1, 19, 512, 512) float32 — the two
    auxiliary training outputs are omitted (not used at inference time).

    Load from ONNX::
        model = FaceParserResnet34Torch.from_onnx(
            "model_assets/faceparser_resnet34.onnx"
        ).cuda().eval()
    """

    def __init__(self, compute_dtype: torch.dtype = torch.float16):
        super().__init__()
        self.backbone = _Backbone()
        self.context = _ContextPath()
        # Primary output: 256ch FFM → 19 classes
        self.conv_out = _OutputHead(256, 256)
        # Auxiliary outputs (kept for weight loading completeness; not returned)
        self.conv_out16 = _OutputHead(128, 64)
        self.conv_out32 = _OutputHead(128, 64)
        self._compute_dtype = compute_dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (1,3,512,512) float32 → (1,19,512,512) float32"""
        x = x.to(self._compute_dtype)
        c3, c4, c5 = self.backbone(x)
        ffm_out, head16_out, head32_out = self.context(c3, c4, c5)
        return self.conv_out(ffm_out).float()  # always return float32

    # ------------------------------------------------------------------
    @classmethod
    def from_onnx(
        cls,
        onnx_path: str,
        compute_dtype: torch.dtype = torch.float16,
    ) -> "FaceParserResnet34Torch":
        """Load all 52 Conv2d weights positionally from faceparser_resnet34.onnx."""
        import onnx

        model_onnx = onnx.load(onnx_path)

        def _np(name: str) -> np.ndarray:
            init = {i.name: i for i in model_onnx.graph.initializer}[name]
            if init.raw_data:
                arr = np.frombuffer(init.raw_data, dtype=np.float32).copy()
            elif init.float_data:
                arr = np.array(list(init.float_data), dtype=np.float32)
            else:
                arr = np.zeros(list(init.dims) or [1], dtype=np.float32)
            return arr.reshape(list(init.dims)) if init.dims else arr

        model = cls(compute_dtype=compute_dtype)

        # --- Positional Conv loading ---
        conv_nodes = [n for n in model_onnx.graph.node if n.op_type == "Conv"]
        pt_convs = _conv_modules_in_forward_order(model)
        assert len(conv_nodes) == len(pt_convs), (
            f"ONNX has {len(conv_nodes)} Conv nodes but model has {len(pt_convs)}"
        )

        with torch.no_grad():
            for onnx_node, pt_conv in zip(conv_nodes, pt_convs):
                w = torch.from_numpy(_np(onnx_node.input[1]))
                pt_conv.weight.copy_(w)
                if len(onnx_node.input) > 2 and onnx_node.input[2]:
                    b = torch.from_numpy(_np(onnx_node.input[2]))
                    if pt_conv.bias is not None:
                        pt_conv.bias.copy_(b)

        # Convert to compute dtype (FP16 for TensorCore dispatch)
        model = model.to(compute_dtype)
        model._compute_dtype = compute_dtype
        model._visomaster_onnx_path = str(onnx_path)
        return model


# ---------------------------------------------------------------------------
# Weight enumeration helper
# ---------------------------------------------------------------------------


def _conv_modules_in_forward_order(model: FaceParserResnet34Torch) -> list[nn.Conv2d]:
    """
    Returns all 52 Conv2d modules in ONNX topological (= forward execution) order.

    Backbone (36 convs):
      stem(1) + layer1(6) + layer2(9) + layer3(13) + layer4(7)
    Context path + heads (16 convs):
      conv_avg(1) + arm32(2) + conv_head32(1) + arm16(2) + conv_head16(1) +
      ffm(3) + conv_out(2) + conv_out16(2) + conv_out32(2)
    """
    bb = model.backbone
    ct = model.context

    convs: list[nn.Conv2d] = []

    # Stem
    convs.append(bb.conv1.conv)

    # Layer 1 — no downsampling blocks
    for blk in bb.layer1:
        convs += [blk.conv1.conv, blk.conv2.conv]

    # Layer 2 — block 0 has downsample; blocks 1-3 don't
    b0 = bb.layer2[0]
    convs += [b0.conv1.conv, b0.conv2.conv, b0.downsample.conv]
    for blk in list(bb.layer2)[1:]:
        convs += [blk.conv1.conv, blk.conv2.conv]

    # Layer 3 — block 0 has downsample; blocks 1-5 don't
    b0 = bb.layer3[0]
    convs += [b0.conv1.conv, b0.conv2.conv, b0.downsample.conv]
    for blk in list(bb.layer3)[1:]:
        convs += [blk.conv1.conv, blk.conv2.conv]

    # Layer 4 — block 0 has downsample; blocks 1-2 don't
    b0 = bb.layer4[0]
    convs += [b0.conv1.conv, b0.conv2.conv, b0.downsample.conv]
    for blk in list(bb.layer4)[1:]:
        convs += [blk.conv1.conv, blk.conv2.conv]

    # Context path
    convs += [
        ct.conv_avg.conv,
        ct.arm32.conv_block.conv,
        ct.arm32.attention,  # plain nn.Conv2d (no _CBR wrapper)
        ct.conv_head32.conv,
        ct.arm16.conv_block.conv,
        ct.arm16.attention,  # plain nn.Conv2d
        ct.conv_head16.conv,
        ct.ffm.conv_block.conv,
        ct.ffm.conv1,  # plain nn.Conv2d, bias=False
        ct.ffm.conv2,  # plain nn.Conv2d, bias=False
    ]

    # Output heads (primary + two auxiliary)
    convs += [
        model.conv_out.conv_block.conv,
        model.conv_out.conv,  # plain nn.Conv2d, bias=False
        model.conv_out16.conv_block.conv,
        model.conv_out16.conv,  # plain nn.Conv2d, bias=False
        model.conv_out32.conv_block.conv,
        model.conv_out32.conv,  # plain nn.Conv2d, bias=False
    ]

    return convs


# ---------------------------------------------------------------------------
# CUDA graph runner (fixed 512×512 — single captured graph)
# ---------------------------------------------------------------------------


class _CapturedGraph:
    """Wraps a single CUDA-graph capture for the fixed 512×512 input."""

    def __init__(self, model: FaceParserResnet34Torch, device: torch.device):
        self._inp = torch.zeros(1, 3, 512, 512, dtype=torch.float32, device=device)

        # Warmup passes — must run before capture to settle cuDNN algorithm selection
        with torch.no_grad():
            for _ in range(3):
                model(self._inp)

        self._stream = torch.cuda.Stream()
        torch.cuda.synchronize()
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(
            self._graph, stream=self._stream, capture_error_mode="relaxed"
        ):
            self._out = model(self._inp)  # (1, 19, 512, 512) float32
        torch.cuda.synchronize()

    def __call__(self, inp: torch.Tensor) -> torch.Tensor:
        self._inp.copy_(inp)
        self._graph.replay()
        return self._out.clone()


def build_cuda_graph_runner(
    model: FaceParserResnet34Torch,
    torch_compile: bool = False,
) -> "_CapturedGraph | FaceParserResnet34Torch":
    """
    Capture a single CUDA graph for the fixed 512×512 input.

    Args:
        model:         FaceParserResnet34Torch already on CUDA in eval mode.
        torch_compile: If True, wrap the model with ``torch.compile``
                       (``reduce-overhead`` mode) before capturing CUDA graphs.
                       reduce-overhead avoids the Triton MLIR AV crash that
                       ``mode='default'`` triggers on Windows (sm_89 / Triton
                       3.4.0) and is ~1.67× faster than CUDA graph (0.80 ms/iter
                       on RTX 4090).  30 warmup calls are used because BiSeNet's
                       multi-branch structure requires more graph captures than
                       simpler models.  The compiled model is returned directly
                       (no outer _CapturedGraph — reduce-overhead has its own).

    Returns a :class:`_CapturedGraph` callable (torch_compile=False) or the
    torch-compiled model (torch_compile=True), or the original *model* as an
    eager-FP16 fallback if graph capture fails.
    """
    if torch_compile:
        try:
            from custom_kernels.compile_utils import apply_torch_compile
            device = next(model.parameters()).device
            example_inp = torch.zeros(1, 3, 512, 512, dtype=torch.float32, device=device)
            # reduce-overhead: avoids manual CUDA graph capture conflicts and is
            # faster (~1.67x vs CUDA graph at 0.80 ms/iter on RTX 4090).
            # warmup=30: BiSeNet has multiple branches; cudagraph_trees needs ~30
            # calls to capture all internal CUDA graphs before hitting fast path.
            compiled = apply_torch_compile(model, example_inp,
                                           compile_mode="reduce-overhead",
                                           warmup=30)
            print("[faceparser_resnet34] torch.compile reduce-overhead done.")
            # reduce-overhead manages its own CUDA graphs — return directly.
            return compiled
        except Exception as e:
            print(f"[faceparser_resnet34] torch.compile failed ({e!s:.120}), falling back to CUDA graph.")

    device = next(model.parameters()).device
    try:
        runner = _CapturedGraph(model, device)
        print("[faceparser_resnet34] CUDA graph captured successfully.")
        return runner
    except Exception as e:
        print(f"[faceparser_resnet34] CUDA graph capture failed, using FP16 eager: {e}")
        return model
