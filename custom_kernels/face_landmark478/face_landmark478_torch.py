"""
FP16 PyTorch reimplementation of
``model_assets/face_landmarks_detector_Nx3x256x256.onnx`` — the MobileNet-style
478-point face landmark detector used for ``FaceLandmark478``.

Architecture
------------
    Input   : (1, 3, 256, 256)  float32
    Outputs : Identity          (1, 1, 1, 1434)  float32  — 478 × (x, y, z) landmarks
              Identity_1        (1, 1, 1,    1)  float32  — visibility score
              Identity_2        (1,          1)  float32  — face presence (sigmoid)

    Backbone: 7-stage hierarchical depthwise-separable network with
              squeeze-expand residual blocks (MobileNet-style).
    Stages  : 128→64→32→16→8→4→2 spatial (channel: 16→32→64→128→128→128→128)
    Head    : 3 parallel Conv(128→C, 2×2) on 2×2 spatial feature

Block types
-----------
    _FirstBlock(C):
        PW_sq(C→C/2) → PReLU → DW(C/2) → PW_ex(C/2→C) → Add(input)
        Used only as the first block of stage 1 (no leading PReLU).

    _ResBlock(C):
        PReLU(C) → PW_sq(C→C/2) → PReLU(C/2) → DW(C/2) → PW_ex(C/2→C) → Add(PReLU_out)
        Used for blocks 2+ in every stage.

Stage transitions
-----------------
    Stages 1→2, 2→3, 3→4 (channel doubling):
        h_act  = stage_prelu(h)
        skip   = F.pad(MaxPool(h_act, 2×2), +C)        # zero-pad MaxPool C→2C
        conv   = Conv(h_act, C→C, 2×2, s=2)            # strided conv is main path
        # First block of next stage:
        h      = PW_ex(DW(PReLU(conv))) + skip          # C→2C via expand

    Stages 4→5, 5→6, 6→7 (channel constant at 128):
        h_act  = stage_prelu(h)
        skip   = MaxPool(h_act, 2×2)                   # 128 ch
        conv   = Conv(h_act, 128→64, 2×2, s=2)
        # First block of next stage:
        h      = PW_ex(DW(PReLU(conv))) + skip          # 64→128 via expand

Weight loading
--------------
    All 256 initializers are named ``const_fold_opt__NNNN`` (no anonymous onnx:: tensors).
    Loaded positionally: ONNX nodes are traversed in topological order; Conv weight/bias
    and PReLU slope tensors are collected in encounter order and assigned to the
    corresponding PyTorch parameters in the same structural order.
    Pad pads arrays (int64) and Reshape shape constants are skipped.
"""

from __future__ import annotations

import pathlib
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class _FirstBlock(nn.Module):
    """First block of stage 1 — no leading PReLU (stem PReLU serves as activation)."""

    def __init__(self, C: int):
        super().__init__()
        H = C // 2
        self.pw_sq = nn.Conv2d(C, H, 1, bias=True)
        self.prelu = nn.PReLU(H)
        self.dw = nn.Conv2d(H, H, 3, padding=1, groups=H, bias=True)
        self.pw_ex = nn.Conv2d(H, C, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pw_ex(self.dw(self.prelu(self.pw_sq(x))))


class _ResBlock(nn.Module):
    """
    Standard residual block with leading pre-activation PReLU.

    PReLU(C) → PW_sq(C→H) → PReLU(H) → DW(H, 3×3) → PW_ex(H→C) → Add(PReLU_out)
    The skip is the PReLU output (pre-activation residual pattern).
    """

    def __init__(self, C: int):
        super().__init__()
        H = C // 2
        self.prelu_in = nn.PReLU(C)
        self.pw_sq = nn.Conv2d(C, H, 1, bias=True)
        self.prelu_sq = nn.PReLU(H)
        self.dw = nn.Conv2d(H, H, 3, padding=1, groups=H, bias=True)
        self.pw_ex = nn.Conv2d(H, C, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.prelu_in(x)
        return h + self.pw_ex(self.dw(self.prelu_sq(self.pw_sq(h))))


# ---------------------------------------------------------------------------
# Full FaceLandmark478 model
# ---------------------------------------------------------------------------


class FaceLandmark478Torch(nn.Module):
    """
    MobileNet-style 7-stage depthwise-separable network for 478-point face landmarks.

    Input  : (1, 3, 256, 256)  float32 CUDA
    Outputs: (landmarks (1,1,1,1434), visibility (1,1,1,1), score (1,1))  float32
             Downstream code reshapes landmarks → (1, 478, 3).
    """

    def __init__(self, compute_dtype: torch.dtype = torch.float16):
        super().__init__()
        self.compute_dtype = compute_dtype

        # ── Stem ─────────────────────────────────────────────────────────────
        # Conv(3→16, 3×3, stride=2) — asymmetric pad [top=0,left=0,bot=1,right=1]
        # applied via F.pad before the conv (padding=0 here)
        self.stem_conv = nn.Conv2d(3, 16, 3, stride=2, padding=0, bias=True)
        self.stem_prelu = nn.PReLU(16)

        # ── Stage 1: 128×128, 16 ch — 4 blocks ──────────────────────────────
        self.s1_b1 = _FirstBlock(16)
        self.s1_b2 = _ResBlock(16)
        self.s1_b3 = _ResBlock(16)
        self.s1_b4 = _ResBlock(16)
        self.s1_act = nn.PReLU(16)  # pre-transition activation

        # ── Transition 1→2: MaxPool + Conv(16→16, 2×2) + Pad(16→32) ─────────
        self.t12_conv = nn.Conv2d(16, 16, 2, stride=2, padding=0, bias=True)

        # ── Stage 2: 64×64, 32 ch — 5 blocks (first = expand-only) ──────────
        self.s2_tb_prelu = nn.PReLU(16)
        self.s2_tb_dw = nn.Conv2d(16, 16, 3, padding=1, groups=16, bias=True)
        self.s2_tb_pwex = nn.Conv2d(16, 32, 1, bias=True)
        self.s2_b2 = _ResBlock(32)
        self.s2_b3 = _ResBlock(32)
        self.s2_b4 = _ResBlock(32)
        self.s2_b5 = _ResBlock(32)
        self.s2_act = nn.PReLU(32)

        # ── Transition 2→3: MaxPool + Conv(32→32, 2×2) + Pad(32→64) ─────────
        self.t23_conv = nn.Conv2d(32, 32, 2, stride=2, padding=0, bias=True)

        # ── Stage 3: 32×32, 64 ch — 5 blocks ────────────────────────────────
        self.s3_tb_prelu = nn.PReLU(32)
        self.s3_tb_dw = nn.Conv2d(32, 32, 3, padding=1, groups=32, bias=True)
        self.s3_tb_pwex = nn.Conv2d(32, 64, 1, bias=True)
        self.s3_b2 = _ResBlock(64)
        self.s3_b3 = _ResBlock(64)
        self.s3_b4 = _ResBlock(64)
        self.s3_b5 = _ResBlock(64)
        self.s3_act = nn.PReLU(64)

        # ── Transition 3→4: MaxPool + Conv(64→64, 2×2) + Pad(64→128) ────────
        self.t34_conv = nn.Conv2d(64, 64, 2, stride=2, padding=0, bias=True)

        # ── Stage 4: 16×16, 128 ch — 5 blocks ───────────────────────────────
        self.s4_tb_prelu = nn.PReLU(64)
        self.s4_tb_dw = nn.Conv2d(64, 64, 3, padding=1, groups=64, bias=True)
        self.s4_tb_pwex = nn.Conv2d(64, 128, 1, bias=True)
        self.s4_b2 = _ResBlock(128)
        self.s4_b3 = _ResBlock(128)
        self.s4_b4 = _ResBlock(128)
        self.s4_b5 = _ResBlock(128)
        self.s4_act = nn.PReLU(128)

        # ── Transition 4→5: MaxPool + Conv(128→64, 2×2) [no Pad] ─────────────
        # MaxPool output (128ch) is the skip; Conv output (64ch) is processed.
        self.t45_conv = nn.Conv2d(128, 64, 2, stride=2, padding=0, bias=True)

        # ── Stage 5: 8×8, 128 ch — 5 blocks ─────────────────────────────────
        self.s5_tb_prelu = nn.PReLU(64)
        self.s5_tb_dw = nn.Conv2d(64, 64, 3, padding=1, groups=64, bias=True)
        self.s5_tb_pwex = nn.Conv2d(64, 128, 1, bias=True)
        self.s5_b2 = _ResBlock(128)
        self.s5_b3 = _ResBlock(128)
        self.s5_b4 = _ResBlock(128)
        self.s5_b5 = _ResBlock(128)
        self.s5_act = nn.PReLU(128)

        # ── Transition 5→6: MaxPool + Conv(128→64, 2×2) ─────────────────────
        self.t56_conv = nn.Conv2d(128, 64, 2, stride=2, padding=0, bias=True)

        # ── Stage 6: 4×4, 128 ch — 5 blocks ─────────────────────────────────
        self.s6_tb_prelu = nn.PReLU(64)
        self.s6_tb_dw = nn.Conv2d(64, 64, 3, padding=1, groups=64, bias=True)
        self.s6_tb_pwex = nn.Conv2d(64, 128, 1, bias=True)
        self.s6_b2 = _ResBlock(128)
        self.s6_b3 = _ResBlock(128)
        self.s6_b4 = _ResBlock(128)
        self.s6_b5 = _ResBlock(128)
        self.s6_act = nn.PReLU(128)

        # ── Transition 6→7: MaxPool + Conv(128→64, 2×2) ─────────────────────
        self.t67_conv = nn.Conv2d(128, 64, 2, stride=2, padding=0, bias=True)

        # ── Stage 7: 2×2, 128 ch — 5 blocks (no separate stage_act) ─────────
        self.s7_tb_prelu = nn.PReLU(64)
        self.s7_tb_dw = nn.Conv2d(64, 64, 3, padding=1, groups=64, bias=True)
        self.s7_tb_pwex = nn.Conv2d(64, 128, 1, bias=True)
        self.s7_b2 = _ResBlock(128)
        self.s7_b3 = _ResBlock(128)
        self.s7_b4 = _ResBlock(128)
        self.s7_b5 = _ResBlock(128)

        # ── Head (operates on 2×2 spatial) ───────────────────────────────────
        self.head_act = nn.PReLU(128)  # node 214
        self.head_score = nn.Conv2d(128, 1, 2, bias=True)  # node 215 → score
        self.head_vis = nn.Conv2d(128, 1, 2, bias=True)  # node 216 → vis
        self.head_lmk = nn.Conv2d(128, 1434, 2, bias=True)  # node 217 → landmarks

    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor):
        """
        Args:
            x : (1, 3, 256, 256)  float32 CUDA
        Returns:
            landmarks : (1, 1, 1, 1434)  float32  (reshape to (1,478,3) downstream)
            visibility: (1, 1, 1,    1)  float32
            score     : (1,          1)  float32
        """
        N = x.shape[0]
        h = x.to(self.compute_dtype)

        # Stem — asymmetric spatial pad: top=0, left=0, bottom=1, right=1
        h = F.pad(h, (0, 1, 0, 1))
        h = self.stem_prelu(self.stem_conv(h))  # (N, 16, 128, 128)

        # Stage 1
        h = self.s1_b1(h)
        h = self.s1_b2(h)
        h = self.s1_b3(h)
        h = self.s1_b4(h)
        h = self.s1_act(h)  # (N, 16, 128, 128)

        # Transition 1→2: MaxPool → Pad = skip; strided Conv → PReLU → DW → PW_ex = main
        skip = F.pad(
            F.max_pool2d(h, 2, stride=2), (0, 0, 0, 0, 0, 16)
        )  # (N, 32, 64, 64)
        h = self.s2_tb_pwex(self.s2_tb_dw(self.s2_tb_prelu(self.t12_conv(h)))) + skip
        h = self.s2_b2(h)
        h = self.s2_b3(h)
        h = self.s2_b4(h)
        h = self.s2_b5(h)
        h = self.s2_act(h)  # (N, 32, 64, 64)

        # Transition 2→3: MaxPool → Pad = skip; strided Conv → PReLU → DW → PW_ex = main
        skip = F.pad(
            F.max_pool2d(h, 2, stride=2), (0, 0, 0, 0, 0, 32)
        )  # (N, 64, 32, 32)
        h = self.s3_tb_pwex(self.s3_tb_dw(self.s3_tb_prelu(self.t23_conv(h)))) + skip
        h = self.s3_b2(h)
        h = self.s3_b3(h)
        h = self.s3_b4(h)
        h = self.s3_b5(h)
        h = self.s3_act(h)  # (N, 64, 32, 32)

        # Transition 3→4: MaxPool → Pad = skip; strided Conv → PReLU → DW → PW_ex = main
        skip = F.pad(
            F.max_pool2d(h, 2, stride=2), (0, 0, 0, 0, 0, 64)
        )  # (N, 128, 16, 16)
        h = self.s4_tb_pwex(self.s4_tb_dw(self.s4_tb_prelu(self.t34_conv(h)))) + skip
        h = self.s4_b2(h)
        h = self.s4_b3(h)
        h = self.s4_b4(h)
        h = self.s4_b5(h)
        h = self.s4_act(h)  # (N, 128, 16, 16)

        # Transition 4→5: MaxPool (skip=128ch) + strided Conv (128→64, processed)
        skip = F.max_pool2d(h, 2, stride=2)  # (N, 128, 8, 8)
        h = self.s5_tb_pwex(self.s5_tb_dw(self.s5_tb_prelu(self.t45_conv(h)))) + skip
        h = self.s5_b2(h)
        h = self.s5_b3(h)
        h = self.s5_b4(h)
        h = self.s5_b5(h)
        h = self.s5_act(h)  # (N, 128, 8, 8)

        # Transition 5→6
        skip = F.max_pool2d(h, 2, stride=2)  # (N, 128, 4, 4)
        h = self.s6_tb_pwex(self.s6_tb_dw(self.s6_tb_prelu(self.t56_conv(h)))) + skip
        h = self.s6_b2(h)
        h = self.s6_b3(h)
        h = self.s6_b4(h)
        h = self.s6_b5(h)
        h = self.s6_act(h)  # (N, 128, 4, 4)

        # Transition 6→7
        skip = F.max_pool2d(h, 2, stride=2)  # (N, 128, 2, 2)
        h = self.s7_tb_pwex(self.s7_tb_dw(self.s7_tb_prelu(self.t67_conv(h)))) + skip
        h = self.s7_b2(h)
        h = self.s7_b3(h)
        h = self.s7_b4(h)
        h = self.s7_b5(h)  # (N, 128, 2, 2)

        # Head
        h = self.head_act(h)  # (N, 128, 2, 2)
        score_raw = self.head_score(h)  # (N, 1, 1, 1)
        vis = self.head_vis(h).float()  # (N, 1, 1, 1)
        lmk_raw = self.head_lmk(h)  # (N, 1434, 1, 1)

        score = torch.sigmoid(score_raw).reshape(N, 1).float()  # (N, 1)
        lmk = lmk_raw.permute(0, 2, 3, 1).reshape(N, 1, 1, 1434).float()  # (N,1,1,1434)

        return lmk, vis, score

    # ------------------------------------------------------------------

    @classmethod
    def from_onnx(
        cls,
        onnx_path: Union[str, pathlib.Path],
        compute_dtype: torch.dtype = torch.float16,
    ) -> "FaceLandmark478Torch":
        """
        Construct a FaceLandmark478Torch and load all weights from the ONNX model.

        All 256 initializers are named ``const_fold_opt__NNNN`` — no anonymous
        onnx:: tensors.  Weights are loaded positionally by traversing ONNX
        nodes in topological (index) order and collecting:
          - Conv  nodes: weight + bias → assigned to successive Conv2d params
          - PRelu nodes: slope         → assigned to successive PReLU params
          - Pad / Reshape / MaxPool / Add / Sigmoid nodes: skipped
        """
        import onnx
        from onnx import numpy_helper

        proto = onnx.load(str(onnx_path))
        g = proto.graph
        init_map = {init.name: numpy_helper.to_array(init) for init in g.initializer}

        # ── Collect Conv weights and PReLU slopes in node order ───────────
        conv_params: list = []  # (weight_arr, bias_arr | None)
        prelu_params: list = []  # slope_arr

        for node in g.node:
            if node.op_type == "Conv":
                w = init_map.get(node.input[1]) if len(node.input) > 1 else None
                b = init_map.get(node.input[2]) if len(node.input) > 2 else None
                if w is not None:
                    conv_params.append((w, b))
            elif node.op_type == "PRelu":
                s = init_map.get(node.input[1]) if len(node.input) > 1 else None
                if s is not None:
                    prelu_params.append(s)

        # ── Build model ───────────────────────────────────────────────────
        m = cls(compute_dtype=compute_dtype)
        ci = 0  # conv index
        pi = 0  # prelu index

        def _c(layer):
            """Assign next Conv weight+bias."""
            nonlocal ci
            w, b = conv_params[ci]
            ci += 1
            layer.weight.data = torch.from_numpy(w.copy()).to(compute_dtype)
            if b is not None and layer.bias is not None:
                layer.bias.data = torch.from_numpy(b.copy()).to(compute_dtype)

        def _p(layer):
            """Assign next PReLU slope (ONNX shape (1,C,1,1) → PyTorch (C,))."""
            nonlocal pi
            s = prelu_params[pi]
            pi += 1
            layer.weight.data = torch.from_numpy(s.reshape(-1).copy()).to(compute_dtype)

        # Stem
        _c(m.stem_conv)
        _p(m.stem_prelu)

        # Stage 1 — block 1 (FirstBlock: pw_sq, prelu, dw, pw_ex)
        _c(m.s1_b1.pw_sq)
        _p(m.s1_b1.prelu)
        _c(m.s1_b1.dw)
        _c(m.s1_b1.pw_ex)

        # Stage 1 — blocks 2-4 (ResBlock)
        for blk in (m.s1_b2, m.s1_b3, m.s1_b4):
            _p(blk.prelu_in)
            _c(blk.pw_sq)
            _p(blk.prelu_sq)
            _c(blk.dw)
            _c(blk.pw_ex)

        _p(m.s1_act)  # s1 final activation

        # Transition 1→2
        _c(m.t12_conv)  # Pad node pads are int64 — no weight to assign

        # Stage 2 — transition block (prelu, dw, pw_ex)
        _p(m.s2_tb_prelu)
        _c(m.s2_tb_dw)
        _c(m.s2_tb_pwex)

        # Stage 2 — blocks 2-5
        for blk in (m.s2_b2, m.s2_b3, m.s2_b4, m.s2_b5):
            _p(blk.prelu_in)
            _c(blk.pw_sq)
            _p(blk.prelu_sq)
            _c(blk.dw)
            _c(blk.pw_ex)

        _p(m.s2_act)

        # Transition 2→3
        _c(m.t23_conv)

        # Stage 3
        _p(m.s3_tb_prelu)
        _c(m.s3_tb_dw)
        _c(m.s3_tb_pwex)
        for blk in (m.s3_b2, m.s3_b3, m.s3_b4, m.s3_b5):
            _p(blk.prelu_in)
            _c(blk.pw_sq)
            _p(blk.prelu_sq)
            _c(blk.dw)
            _c(blk.pw_ex)
        _p(m.s3_act)

        # Transition 3→4
        _c(m.t34_conv)

        # Stage 4
        _p(m.s4_tb_prelu)
        _c(m.s4_tb_dw)
        _c(m.s4_tb_pwex)
        for blk in (m.s4_b2, m.s4_b3, m.s4_b4, m.s4_b5):
            _p(blk.prelu_in)
            _c(blk.pw_sq)
            _p(blk.prelu_sq)
            _c(blk.dw)
            _c(blk.pw_ex)
        _p(m.s4_act)

        # Transition 4→5
        _c(m.t45_conv)

        # Stage 5
        _p(m.s5_tb_prelu)
        _c(m.s5_tb_dw)
        _c(m.s5_tb_pwex)
        for blk in (m.s5_b2, m.s5_b3, m.s5_b4, m.s5_b5):
            _p(blk.prelu_in)
            _c(blk.pw_sq)
            _p(blk.prelu_sq)
            _c(blk.dw)
            _c(blk.pw_ex)
        _p(m.s5_act)

        # Transition 5→6
        _c(m.t56_conv)

        # Stage 6
        _p(m.s6_tb_prelu)
        _c(m.s6_tb_dw)
        _c(m.s6_tb_pwex)
        for blk in (m.s6_b2, m.s6_b3, m.s6_b4, m.s6_b5):
            _p(blk.prelu_in)
            _c(blk.pw_sq)
            _p(blk.prelu_sq)
            _c(blk.dw)
            _c(blk.pw_ex)
        _p(m.s6_act)

        # Transition 6→7
        _c(m.t67_conv)

        # Stage 7 (no separate stage_act — head_act follows immediately)
        _p(m.s7_tb_prelu)
        _c(m.s7_tb_dw)
        _c(m.s7_tb_pwex)
        for blk in (m.s7_b2, m.s7_b3, m.s7_b4, m.s7_b5):
            _p(blk.prelu_in)
            _c(blk.pw_sq)
            _p(blk.prelu_sq)
            _c(blk.dw)
            _c(blk.pw_ex)

        # Head
        _p(m.head_act)
        _c(m.head_score)  # node 215
        _c(m.head_vis)  # node 216
        _c(m.head_lmk)  # node 217

        print(f"[face_landmark478] loaded: {ci} Conv + {pi} PReLU weight tensors")
        assert ci == len(conv_params), (
            f"Conv mismatch: assigned {ci}/{len(conv_params)}"
        )
        assert pi == len(prelu_params), (
            f"PReLU mismatch: assigned {pi}/{len(prelu_params)}"
        )
        m._visomaster_onnx_path = str(onnx_path)
        return m


# ---------------------------------------------------------------------------
# CUDA graph runner
# ---------------------------------------------------------------------------


class FaceLandmark478CUDAGraphRunner:
    """Wraps FaceLandmark478Torch in a CUDA graph for minimal kernel-launch overhead."""

    def __init__(
        self, model: FaceLandmark478Torch, input_shape: tuple = (1, 3, 256, 256)
    ):
        self.model = model
        self.device = next(model.parameters()).device

        self._x_buf = torch.zeros(input_shape, dtype=torch.float32, device=self.device)

        # Warm-up (cuDNN auto-tune, workspace allocation)
        with torch.no_grad():
            for _ in range(3):
                _ = model(self._x_buf)
        torch.cuda.synchronize()

        # Capture
        self._stream = torch.cuda.Stream()
        self._graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        with torch.no_grad():
            with torch.cuda.graph(
                self._graph, stream=self._stream, capture_error_mode="relaxed"
            ):
                self._out = model(self._x_buf)  # (lmk, vis, score)
        torch.cuda.synchronize()

    def __call__(self, x: torch.Tensor):
        """
        Args:
            x : (1, 3, 256, 256)  float32 CUDA
        Returns:
            (landmarks (1,1,1,1434), visibility (1,1,1,1), score (1,1))  float32
        """
        self._x_buf.copy_(x)
        self._graph.replay()
        return tuple(o.clone() for o in self._out)


def build_cuda_graph_runner(
    model: FaceLandmark478Torch,
    input_shape: tuple = (1, 3, 256, 256),
    torch_compile: bool = False,
) -> FaceLandmark478CUDAGraphRunner:
    """
    Build and return a CUDA-graph-backed runner for FaceLandmark478Torch.

    Args:
        model         : FaceLandmark478Torch on CUDA in eval() mode.
        input_shape   : fixed input shape (default (1, 3, 256, 256)).
        torch_compile : if True, apply torch.compile before building the CUDA graph.
    """
    if torch_compile:
        try:
            from custom_kernels.compile_utils import apply_torch_compile
            device = next(model.parameters()).device
            example_inp = torch.zeros(input_shape, dtype=torch.float32, device=device)
            compiled = apply_torch_compile(model, example_inp)
            print("[face_landmark478] torch.compile warmup done.")
            return compiled  # CUDA graph on top of torch.compile fails on Windows
        except Exception as e:
            print(f"[face_landmark478] torch.compile failed ({e!s:.120}), falling back to CUDA graph.")
    return FaceLandmark478CUDAGraphRunner(model, input_shape=input_shape)
