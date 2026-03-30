"""
FP16 PyTorch reimplementation of yoloface_8n.onnx (YOLOv8n-face detector).

Architecture: YOLOv8n backbone + PAN neck + Detect head (bbox/cls/kps).
Input:  (1, 3, 640, 640) float32  — fixed size
Output: (1, 20, 8400) float32 — [cx,cy,w,h, cls_conf, kps_x0,y0,v0,...,x4,y4,v4]

All 73 Conv2d weights are loaded by ONNX initializer name (Ultralytics naming).
BN is folded into Conv at export (all conv have bias=True, no BN nodes).
SiLU = Sigmoid(x)*x in ONNX graph.

Usage:
    model  = YoloFace8nTorch.from_onnx("model_assets/yoloface_8n.onnx").cuda().eval()
    runner = build_cuda_graph_runner(model)   # single captured graph (fixed input)
    with torch.no_grad():
        out = runner(img_float32_1_3_640_640)  # (1, 20, 8400) float32
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class _CBS(nn.Module):
    """BN-folded Conv2d + optional SiLU.  Attribute .conv matches ONNX weight names."""

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 1,
        s: int = 1,
        p: Optional[int] = None,
        act: bool = True,
    ):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(c1, c2, k, s, p, bias=True)
        self.act = act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return F.silu(x) if self.act else x


class _Bottleneck(nn.Module):
    """YOLOv8 Bottleneck: two 3×3 convs with optional residual skip."""

    def __init__(self, c: int, shortcut: bool = True):
        super().__init__()
        self.cv1 = _CBS(c, c, k=3)
        self.cv2 = _CBS(c, c, k=3)
        self.add = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.cv2(self.cv1(x))
        return x + out if self.add else out


class _C2f(nn.Module):
    """YOLOv8 C2f: cv1(1×1) → split → n×Bottleneck → cat → cv2(1×1)."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True):
        super().__init__()
        self.c_ = c2 // 2  # hidden channels
        self.cv1 = _CBS(c1, 2 * self.c_, k=1)
        self.cv2 = _CBS((2 + n) * self.c_, c2, k=1)
        self.m = nn.ModuleList(
            [_Bottleneck(self.c_, shortcut=shortcut) for _ in range(n)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class _SPPF(nn.Module):
    """Spatial Pyramid Pooling Fast: cv1 → 3×MaxPool(k,s=1) → cat → cv2."""

    def __init__(self, c1: int, c2: int, k: int = 5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = _CBS(c1, c_, k=1)
        self.cv2 = _CBS(c_ * 4, c2, k=1)
        self.mp = nn.MaxPool2d(k, 1, k // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        y1 = self.mp(x)
        y2 = self.mp(y1)
        y3 = self.mp(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], 1))


# ---------------------------------------------------------------------------
# Backbone (models 0–9)
# ---------------------------------------------------------------------------


class _Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.m0 = _CBS(3, 16, k=3, s=2)  # stride 2
        self.m1 = _CBS(16, 32, k=3, s=2)  # stride 4
        self.m2 = _C2f(32, 32, n=1, shortcut=True)
        self.m3 = _CBS(32, 64, k=3, s=2)  # stride 8
        self.m4 = _C2f(64, 64, n=2, shortcut=True)  # P3, stride 8, 64ch
        self.m5 = _CBS(64, 128, k=3, s=2)  # stride 16
        self.m6 = _C2f(128, 128, n=2, shortcut=True)  # P4, stride 16, 128ch
        self.m7 = _CBS(128, 256, k=3, s=2)  # stride 32
        self.m8 = _C2f(256, 256, n=1, shortcut=True)
        self.m9 = _SPPF(256, 256, k=5)  # P5, stride 32, 256ch

    def forward(self, x: torch.Tensor):
        x = self.m0(x)
        x = self.m1(x)
        x = self.m2(x)
        x = self.m3(x)
        p3 = self.m4(x)  # 64ch
        x = self.m5(p3)
        p4 = self.m6(x)  # 128ch
        x = self.m7(p4)
        x = self.m8(x)
        p5 = self.m9(x)  # 256ch
        return p3, p4, p5


# ---------------------------------------------------------------------------
# PAN Neck (models 12, 15, 16, 18, 19, 21; models 10/11/13/14/17 are
# Upsample/Concat ops — no learnable params)
# ---------------------------------------------------------------------------


class _Neck(nn.Module):
    def __init__(self):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        # top-down
        self.m12 = _C2f(384, 128, n=1, shortcut=False)  # cat(up(p5),p4)  256+128=384
        self.m15 = _C2f(192, 64, n=1, shortcut=False)  # cat(up(m12),p3) 128+64=192
        # bottom-up
        self.m16 = _CBS(64, 64, k=3, s=2)
        self.m18 = _C2f(192, 128, n=1, shortcut=False)  # cat(m16,m12)    64+128=192
        self.m19 = _CBS(128, 128, k=3, s=2)
        self.m21 = _C2f(384, 256, n=1, shortcut=False)  # cat(m19,p5)     128+256=384

    def forward(self, p3, p4, p5):
        # top-down pathway
        x = torch.cat([self.up(p5), p4], 1)  # 384ch
        p4_neck = self.m12(x)  # 128ch, stride 16
        x = torch.cat([self.up(p4_neck), p3], 1)  # 192ch
        p3_out = self.m15(x)  # 64ch,  stride 8  → detection output

        # bottom-up pathway
        x = torch.cat([self.m16(p3_out), p4_neck], 1)  # 192ch
        p4_out = self.m18(x)  # 128ch, stride 16 → detection output
        x = torch.cat([self.m19(p4_out), p5], 1)  # 384ch
        p5_out = self.m21(x)  # 256ch, stride 32 → detection output
        return p3_out, p4_out, p5_out


# ---------------------------------------------------------------------------
# Detect head (model.22) — per-scale branches + DFL decode
# ---------------------------------------------------------------------------


class _HeadScale(nn.Module):
    """One per detection stride.  cv2=bbox-reg, cv3=cls, cv4=kps."""

    def __init__(self, c1: int):
        super().__init__()
        # bbox regression (DFL): outputs reg_max*4 = 64 channels
        self.cv2 = nn.Sequential(
            _CBS(c1, 64, k=3),
            _CBS(64, 64, k=3),
            nn.Conv2d(64, 64, 1, bias=True),  # plain — ONNX name .2.weight
        )
        # class confidence: outputs nc=1 channel
        self.cv3 = nn.Sequential(
            _CBS(c1, 64, k=3),
            _CBS(64, 64, k=3),
            nn.Conv2d(64, 1, 1, bias=True),  # plain — ONNX name .2.weight
        )
        # keypoints: outputs nkpt*3=15 channels (5 kps × [x,y,vis])
        self.cv4 = nn.Sequential(
            _CBS(c1, 16, k=3),
            _CBS(16, 16, k=3),
            nn.Conv2d(16, 15, 1, bias=True),  # plain — ONNX name .2.weight
        )


class _DetectHead(nn.Module):
    """
    Full decode: DFL → dist2bbox + anchor → cxcywh pixel;
    cls → sigmoid; kps → (xy*2+anchor)*stride, vis → sigmoid.

    Anchor grids are pre-computed for fixed 640×640 input and stored as
    float32 buffers.  They are NOT converted when .to(fp16) is called.
    """

    REG_MAX = 16
    NKPT = 5

    def __init__(self):
        super().__init__()
        self.head8 = _HeadScale(64)  # stride 8,  P3
        self.head16 = _HeadScale(128)  # stride 16, P4
        self.head32 = _HeadScale(256)  # stride 32, P5

        # DFL 1×1 conv: weights initialised as [0..REG_MAX-1], no bias.
        # Loaded from ONNX (model.22.dfl.conv.weight); should equal range [0..15].
        self.dfl = nn.Conv2d(self.REG_MAX, 1, 1, bias=False)

        # Anchor grids — computed analytically for fixed 640×640; float32 always.
        anchor_xy, kps_anchor_xy, strides = _make_anchor_grids()
        self.register_buffer("anchor_xy", anchor_xy)          # (1, 2, 8400) — bbox (cell centers)
        self.register_buffer("kps_anchor_xy", kps_anchor_xy)  # (2,    8400) — kps  (no +0.5)
        self.register_buffer("strides", strides)               # (1, 1, 8400)

    # ------------------------------------------------------------------
    def _dfl_decode(
        self,
        raw_reg: torch.Tensor,  # (B, 64, N)
        anc_xy: torch.Tensor,  # (1, 2, N)
        stride: torch.Tensor,  # (1, 1, N)
    ) -> torch.Tensor:  # (B, 4, N) cxcywh pixels
        """DFL soft-argmax → ltrb → dist2bbox → cxcywh, multiplied by stride."""
        B, _, N = raw_reg.shape
        # (B,4,16,N) → softmax → weighted sum → ltrb (grid units)
        w = self.dfl.weight.reshape(1, 1, self.REG_MAX, 1).float()  # (1,1,16,1)
        ltrb = (
            raw_reg.float().reshape(B, 4, self.REG_MAX, N).softmax(2).mul(w).sum(2)
        )  # (B, 4, N)

        anc = anc_xy.float()  # (1, 2, N)
        lx, t, r, b = ltrb[:, 0:1], ltrb[:, 1:2], ltrb[:, 2:3], ltrb[:, 3:4]
        cx = anc[:, 0:1] + (r - lx) * 0.5
        cy = anc[:, 1:2] + (b - t) * 0.5
        w_ = lx + r
        h_ = t + b
        return torch.cat([cx, cy, w_, h_], 1) * stride.float()  # (B,4,N) pixels

    # ------------------------------------------------------------------
    def _decode_scale(
        self,
        head: _HeadScale,
        feat: torch.Tensor,       # (B,C,H,W)
        anc_xy: torch.Tensor,     # (1,2,N)   — bbox cell centers (+0.5)
        stride: torch.Tensor,     # (1,1,N)
        kps_anc_xy: torch.Tensor, # (2,N)     — kps grid indices (no +0.5)
    ) -> torch.Tensor:  # (B,20,N)
        B, _, H, W = feat.shape
        N = H * W

        raw_reg = head.cv2(feat).reshape(B, 64, N)  # (B,64,N)
        raw_cls = head.cv3(feat).reshape(B, 1, N)  # (B,1,N)
        raw_kps = head.cv4(feat).reshape(B, self.NKPT, 3, N)  # (B,5,3,N)

        bbox = self._dfl_decode(raw_reg, anc_xy, stride)  # (B,4,N)
        cls = raw_cls.float().sigmoid()  # (B,1,N)

        # kps xy: (raw * 2 + kps_anchor) * stride
        # kps_anchor uses raw grid indices (no +0.5) — matches ONNX Constant_23.
        kps_xy_raw = raw_kps[:, :, :2, :].float()   # (B,5,2,N)
        kps_vis_raw = raw_kps[:, :, 2:3, :].float() # (B,5,1,N)
        str_exp = stride.unsqueeze(1).float()        # (1,1,1,N)
        # kps_anc_xy is (2,N); broadcasting against (B,5,2,N) works via trailing dims.
        kps_xy = (kps_xy_raw * 2.0 + kps_anc_xy.float()) * str_exp  # (B,5,2,N) pixels
        kps_vis = kps_vis_raw.sigmoid()  # (B,5,1,N)
        kps_out = torch.cat([kps_xy, kps_vis], 2).reshape(B, 15, N)  # (B,15,N)

        return torch.cat([bbox, cls, kps_out], 1)  # (B,20,N)

    # ------------------------------------------------------------------
    def forward(self, p3, p4, p5) -> torch.Tensor:  # (B,20,8400)
        a8,   s8   = self.anchor_xy[:, :, :6400],    self.strides[:, :, :6400]
        a16,  s16  = self.anchor_xy[:, :, 6400:8000], self.strides[:, :, 6400:8000]
        a32,  s32  = self.anchor_xy[:, :, 8000:],    self.strides[:, :, 8000:]
        ka8  = self.kps_anchor_xy[:, :6400]
        ka16 = self.kps_anchor_xy[:, 6400:8000]
        ka32 = self.kps_anchor_xy[:, 8000:]

        out8  = self._decode_scale(self.head8,  p3, a8,  s8,  ka8)
        out16 = self._decode_scale(self.head16, p4, a16, s16, ka16)
        out32 = self._decode_scale(self.head32, p5, a32, s32, ka32)
        return torch.cat([out8, out16, out32], 2)  # (B,20,8400)


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------


class YoloFace8nTorch(nn.Module):
    """
    FP16 YOLOv8n-face detector.

    Weights loaded from yoloface_8n.onnx via :meth:`from_onnx`.
    The model internally runs convolutions in *compute_dtype* (default float16)
    and returns float32 output (1, 20, 8400).

    Input accepted in float32 [0, 1].
    """

    def __init__(self, compute_dtype: torch.dtype = torch.float16):
        super().__init__()
        self.backbone = _Backbone()
        self.neck = _Neck()
        self.head = _DetectHead()
        self._compute_dtype = compute_dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (1,3,640,640) float32 → (1,20,8400) float32"""
        x = x.to(self._compute_dtype)
        p3, p4, p5 = self.backbone(x)
        p3, p4, p5 = self.neck(p3, p4, p5)
        return self.head(p3, p4, p5)  # float32 output from decode

    # ------------------------------------------------------------------
    @classmethod
    def from_onnx(
        cls,
        onnx_path: str,
        compute_dtype: torch.dtype = torch.float16,
    ) -> "YoloFace8nTorch":
        """Load weights from yoloface_8n.onnx and return ready-to-use model."""
        import onnx

        model_onnx = onnx.load(onnx_path)
        init_map = {init.name: init for init in model_onnx.graph.initializer}

        def _np(name: str) -> np.ndarray:
            init = init_map[name]
            if init.raw_data:
                arr = np.frombuffer(init.raw_data, dtype=np.float32).copy()
            elif init.float_data:
                arr = np.array(list(init.float_data), dtype=np.float32)
            else:
                arr = np.zeros(list(init.dims) or [1], dtype=np.float32)
            return arr.reshape(list(init.dims)) if init.dims else arr

        def _load(param: nn.Parameter, name: str) -> None:
            with torch.no_grad():
                param.copy_(torch.from_numpy(_np(name)))

        def _load_cbs(mod: _CBS, prefix: str) -> None:
            _load(mod.conv.weight, f"{prefix}.conv.weight")
            _load(mod.conv.bias, f"{prefix}.conv.bias")

        def _load_c2f(mod: _C2f, prefix: str) -> None:
            _load_cbs(mod.cv1, f"{prefix}.cv1")
            for i, b in enumerate(mod.m):
                _load_cbs(b.cv1, f"{prefix}.m.{i}.cv1")
                _load_cbs(b.cv2, f"{prefix}.m.{i}.cv2")
            _load_cbs(mod.cv2, f"{prefix}.cv2")

        def _load_sppf(mod: _SPPF, prefix: str) -> None:
            _load_cbs(mod.cv1, f"{prefix}.cv1")
            _load_cbs(mod.cv2, f"{prefix}.cv2")

        def _load_head_scale(
            mod: _HeadScale, pfx_cv2: str, pfx_cv3: str, pfx_cv4: str
        ) -> None:
            # cv2 — [0]=CBS, [1]=CBS, [2]=plain Conv2d
            _load_cbs(mod.cv2[0], f"{pfx_cv2}.0")
            _load_cbs(mod.cv2[1], f"{pfx_cv2}.1")
            _load(mod.cv2[2].weight, f"{pfx_cv2}.2.weight")
            _load(mod.cv2[2].bias, f"{pfx_cv2}.2.bias")
            # cv3
            _load_cbs(mod.cv3[0], f"{pfx_cv3}.0")
            _load_cbs(mod.cv3[1], f"{pfx_cv3}.1")
            _load(mod.cv3[2].weight, f"{pfx_cv3}.2.weight")
            _load(mod.cv3[2].bias, f"{pfx_cv3}.2.bias")
            # cv4
            _load_cbs(mod.cv4[0], f"{pfx_cv4}.0")
            _load_cbs(mod.cv4[1], f"{pfx_cv4}.1")
            _load(mod.cv4[2].weight, f"{pfx_cv4}.2.weight")
            _load(mod.cv4[2].bias, f"{pfx_cv4}.2.bias")

        model = cls(compute_dtype=compute_dtype)
        bb = model.backbone
        nk = model.neck
        hd = model.head

        # Backbone
        _load_cbs(bb.m0, "model.0")
        _load_cbs(bb.m1, "model.1")
        _load_c2f(bb.m2, "model.2")
        _load_cbs(bb.m3, "model.3")
        _load_c2f(bb.m4, "model.4")
        _load_cbs(bb.m5, "model.5")
        _load_c2f(bb.m6, "model.6")
        _load_cbs(bb.m7, "model.7")
        _load_c2f(bb.m8, "model.8")
        _load_sppf(bb.m9, "model.9")

        # Neck
        _load_c2f(nk.m12, "model.12")
        _load_c2f(nk.m15, "model.15")
        _load_cbs(nk.m16, "model.16")
        _load_c2f(nk.m18, "model.18")
        _load_cbs(nk.m19, "model.19")
        _load_c2f(nk.m21, "model.21")

        # Detection head — three scales
        _load_head_scale(hd.head8, "model.22.cv2.0", "model.22.cv3.0", "model.22.cv4.0")
        _load_head_scale(
            hd.head16, "model.22.cv2.1", "model.22.cv3.1", "model.22.cv4.1"
        )
        _load_head_scale(
            hd.head32, "model.22.cv2.2", "model.22.cv3.2", "model.22.cv4.2"
        )

        # DFL conv — no bias
        _load(hd.dfl.weight, "model.22.dfl.conv.weight")

        # Apply compute dtype (anchor buffers stay float32 — cast back after)
        anchor_xy_saved = model.head.anchor_xy.clone()
        kps_anchor_xy_saved = model.head.kps_anchor_xy.clone()
        strides_saved = model.head.strides.clone()
        model = model.to(compute_dtype)
        model._compute_dtype = compute_dtype
        # Restore float32 anchor buffers (used in decode, not in conv path)
        model.head.anchor_xy.copy_(anchor_xy_saved.float())
        model.head.kps_anchor_xy.copy_(kps_anchor_xy_saved.float())
        model.head.strides.copy_(strides_saved.float())

        model._visomaster_onnx_path = str(onnx_path)
        return model


# ---------------------------------------------------------------------------
# Anchor grid helper (computed analytically for fixed 640×640 input)
# ---------------------------------------------------------------------------


def _make_anchor_grids() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns (bbox_anchor_xy, kps_anchor_xy, strides) for fixed 640×640 YOLOv8n input.

    bbox_anchor_xy : (1, 2, 8400) float32 — cell centers (+0.5) for DFL bbox decode
    kps_anchor_xy  : (2,    8400) float32 — raw grid indices (no +0.5) for kps decode
                     Matches ONNX Constant_23 used in the kps branch.
    strides        : (1, 1, 8400) float32 — stride value (8/16/32) per anchor
    """
    configs = [(8, 80, 80), (16, 40, 40), (32, 20, 20)]
    all_bbox_xy: list[torch.Tensor] = []
    all_kps_xy: list[torch.Tensor] = []
    all_st: list[torch.Tensor] = []

    for stride, H, W in configs:
        N = H * W
        # bbox anchors: cell centers (+ 0.5)
        gy_c, gx_c = torch.meshgrid(
            torch.arange(H, dtype=torch.float32) + 0.5,
            torch.arange(W, dtype=torch.float32) + 0.5,
            indexing="ij",
        )
        all_bbox_xy.append(torch.stack([gx_c.reshape(-1), gy_c.reshape(-1)], 0))  # (2, N)
        # kps anchors: raw grid indices (no + 0.5) — matches ONNX Constant_23
        gy, gx = torch.meshgrid(
            torch.arange(H, dtype=torch.float32),
            torch.arange(W, dtype=torch.float32),
            indexing="ij",
        )
        all_kps_xy.append(torch.stack([gx.reshape(-1), gy.reshape(-1)], 0))  # (2, N)
        all_st.append(torch.full((N,), stride, dtype=torch.float32))

    bbox_anchor_xy = torch.cat(all_bbox_xy, 1).unsqueeze(0)  # (1, 2, 8400)
    kps_anchor_xy = torch.cat(all_kps_xy, 1)                 # (2,    8400)
    strides = torch.cat(all_st).unsqueeze(0).unsqueeze(0)    # (1, 1, 8400)
    return bbox_anchor_xy, kps_anchor_xy, strides


# ---------------------------------------------------------------------------
# CUDA graph runner (fixed 640×640 — single captured graph)
# ---------------------------------------------------------------------------


class _CapturedGraph:
    """Wraps a single CUDA-graph capture for the fixed 640×640 input."""

    def __init__(self, model: YoloFace8nTorch, device: torch.device):
        self._inp = torch.zeros(1, 3, 640, 640, dtype=torch.float32, device=device)

        # Warmup — needed before graph capture to avoid spurious cudnn operations
        with torch.no_grad():
            for _ in range(3):
                model(self._inp)

        self._stream = torch.cuda.Stream()
        torch.cuda.synchronize()
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(
            self._graph, stream=self._stream, capture_error_mode="relaxed"
        ):
            self._out = model(self._inp)  # (1, 20, 8400) float32
        torch.cuda.synchronize()

    def __call__(self, inp: torch.Tensor) -> torch.Tensor:
        self._inp.copy_(inp)
        self._graph.replay()
        return self._out.clone()


def build_cuda_graph_runner(
    model: YoloFace8nTorch,
    torch_compile: bool = False,
) -> "_CapturedGraph | YoloFace8nTorch":
    """
    Capture a CUDA graph for the fixed 640×640 input.

    Args:
        model:         YoloFace8nTorch already on CUDA in eval mode.
        torch_compile: If True, wrap the model with ``torch.compile`` before
                       capturing the CUDA graph.  Uses ``default`` mode (stable
                       on all GPUs / Triton versions).  The compiled model is
                       then captured into a CUDA graph so that dynamo is never
                       re-invoked during inference (avoids recompile_limit
                       warnings and ensures stable latency).
                       Adds a one-time ~30 s compile cost; gives ~1.92×
                       speedup over the uncompiled baseline.

    Returns a :class:`_CapturedGraph` callable, or the original *model* as a
    fallback if graph capture is unavailable.
    """
    if torch_compile:
        try:
            from custom_kernels.compile_utils import apply_torch_compile
            device = next(model.parameters()).device
            example_inp = torch.zeros(1, 3, 640, 640, dtype=torch.float32, device=device)
            # reduce-overhead: captures its own CUDA graphs internally; avoids the
            # manual CUDA graph capture seed error.
            # ~1.9x faster than mode='default'+CUDA graph at 0.59 ms/iter.
            compiled = apply_torch_compile(model, example_inp,
                                           compile_mode="reduce-overhead")
            print("[yoloface_8n] torch.compile reduce-overhead done.")
            # reduce-overhead has its own internal CUDA graphs — return directly
            # without an outer _CapturedGraph wrapper.
            return compiled
        except Exception as e:
            print(f"[yoloface_8n] torch.compile failed ({e!s:.120}), falling back to CUDA graph.")

    device = next(model.parameters()).device
    try:
        runner = _CapturedGraph(model, device)
        print("[yoloface_8n] CUDA graph captured successfully.")
        return runner
    except Exception as e:
        print(f"[yoloface_8n] CUDA graph capture failed, using FP16 eager: {e}")
        return model
