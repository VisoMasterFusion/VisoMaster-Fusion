"""
FP16 PyTorch reimplementation of
``model_assets/face_blendshapes_Nx146x2.onnx`` — the MLP-Mixer network that
predicts 52 ARKit blendshape coefficients from 146 selected 2-D facial
landmarks.

Architecture
------------
    Input   : (N, 146, 2)  float32 — 2-D landmark coords (un-normalised)
    Output  : (N, 52)      float32 — blendshape weights ∈ (0, 1)  (Sigmoid)

    Preprocessing:
        1.  Subtract per-sample centroid  (mean over 146 landmarks).
        2.  Divide by mean L2 norm of the centred landmarks × 0.5.
        3.  Reshape to NCHW (N, 146, 1, 2) for Conv2d compatibility.

    Embedding:
        Conv2d(146→96, 1×1)   — mix across the 146 landmark channels
        Transpose [0,3,2,1]   — (N, 96, 1, 2) → (N, 2, 1, 96)
        Conv2d(2→64,   1×1)   — project coordinate pair to 64-d embedding
        Prepend CLS token     — (N, 64, 1, 96) → (N, 64, 1, 97)

    4 × MixerBlock:
        LayerNorm(64) per token (over C=64 dim)
        Token mixing  : permute→Conv2d(97→384→97, 1×1)→ReLU→permute  +skip
        LayerNorm(64) per token
        Channel mixing: Conv2d(64→256→64, 1×1) → ReLU                +skip

    Head:
        Take CLS token  (W-index 0)  →  (N, 64, 1, 1)
        Conv2d(64→52, 1×1) → Sigmoid → reshape → (N, 52)

Tensor format throughout: (N, C=64, H=1, W=97)

LayerNorm
---------
    The ONNX graph implements LN manually (TF export artefact) with a
    learnable scale γ (shape 64,) but *no* learnable bias (β = 0).
    Epsilon  = 1.013279e-06 (from ONNX initialiser ``Transpose__290_0``).
    Equivalent to: F.layer_norm(x_perm, [64], weight=γ, bias=None, eps=…)

Weight loading
--------------
    59 ONNX initialisers:
      • 19 Conv weight/bias pairs  → loaded positionally in node order.
      •  8 LN scale vectors (64,)  → loaded by name.
      •  1 CLS token (1, 64, 1, 1) → loaded by name.
      • 31 shape/constant tensors   → skipped.
"""

from __future__ import annotations

import pathlib
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Constants (from ONNX graph)
# ---------------------------------------------------------------------------

_LN_EPS = 1.013279e-06  # Transpose__290_0 value

_LN1_GAMMA_NAMES = [
    "const_fold_opt__452",  # MixerBlock_0 LN1
    "const_fold_opt__435",  # MixerBlock_1 LN1
    "const_fold_opt__416",  # MixerBlock_2 LN1
    "const_fold_opt__384",  # MixerBlock_3 LN1
]
_LN2_GAMMA_NAMES = [
    "const_fold_opt__396",  # MixerBlock_0 LN2
    "const_fold_opt__434",  # MixerBlock_1 LN2
    "const_fold_opt__415",  # MixerBlock_2 LN2
    "const_fold_opt__386",  # MixerBlock_3 LN2
]
_CLS_TOKEN_NAME = "tile_Constant_4_output_0"


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------


class FaceBlendShapesTorch(nn.Module):
    """
    FP16-ready PyTorch reimplementation of face_blendshapes_Nx146x2.onnx.

    Instantiate via :meth:`from_onnx` to load pretrained weights.
    """

    N_LANDMARKS = 146
    N_PROJ = 96  # landmark projection dim
    D_MODEL = 64  # token embedding dim
    N_TOKENS = 97  # 1 CLS + 96 landmark tokens
    TOKEN_HIDDEN = 384
    CHANNEL_HIDDEN = 256
    N_CLASSES = 52

    def __init__(self, compute_dtype: torch.dtype = torch.float16):
        super().__init__()
        self.compute_dtype = compute_dtype

        # ── Embedding ──────────────────────────────────────────────────────
        # Step 1: mix across the 146 landmark channels → 96
        self.emb1 = nn.Conv2d(self.N_LANDMARKS, self.N_PROJ, 1, bias=True)
        # Step 2: project the 2-d coord pair → 64
        self.emb2 = nn.Conv2d(2, self.D_MODEL, 1, bias=True)
        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, self.D_MODEL, 1, 1))

        # ── MixerBlock parameters (4 blocks) ─────────────────────────────
        # LN scale (no bias; model has β = 0 throughout)
        self.ln1_gamma = nn.ParameterList(
            [nn.Parameter(torch.ones(self.D_MODEL)) for _ in range(4)]
        )
        self.ln2_gamma = nn.ParameterList(
            [nn.Parameter(torch.ones(self.D_MODEL)) for _ in range(4)]
        )

        # Token mixing MLP — operates on transposed (N, 97, 1, 64) tensor
        self.tok_up = nn.ModuleList(
            [
                nn.Conv2d(self.N_TOKENS, self.TOKEN_HIDDEN, 1, bias=True)
                for _ in range(4)
            ]
        )
        self.tok_down = nn.ModuleList(
            [
                nn.Conv2d(self.TOKEN_HIDDEN, self.N_TOKENS, 1, bias=True)
                for _ in range(4)
            ]
        )

        # Channel mixing MLP — operates on (N, 64, 1, 97)
        self.ch_up = nn.ModuleList(
            [
                nn.Conv2d(self.D_MODEL, self.CHANNEL_HIDDEN, 1, bias=True)
                for _ in range(4)
            ]
        )
        self.ch_down = nn.ModuleList(
            [
                nn.Conv2d(self.CHANNEL_HIDDEN, self.D_MODEL, 1, bias=True)
                for _ in range(4)
            ]
        )

        # ── Output head ───────────────────────────────────────────────────
        self.head = nn.Conv2d(self.D_MODEL, self.N_CLASSES, 1, bias=True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _ln(self, x: torch.Tensor, gamma: nn.Parameter) -> torch.Tensor:
        """
        Layer normalisation over the C=64 channel dimension.

        x     : (N, 64, 1, 97)
        gamma : (64,)  — learnable scale, no bias
        returns (N, 64, 1, 97)
        """
        x_t = x.permute(0, 2, 3, 1)  # (N, 1, 97, 64)
        x_n = F.layer_norm(x_t, [self.D_MODEL], weight=gamma, bias=None, eps=_LN_EPS)
        return x_n.permute(0, 3, 1, 2)  # (N, 64, 1, 97)

    def _mixer_block(self, x: torch.Tensor, i: int) -> torch.Tensor:
        """
        One MixerBlock.

        x : (N, 64, 1, 97)
        returns (N, 64, 1, 97)
        """
        # ── Token mixing ──────────────────────────────────────────────────
        h = self._ln(x, self.ln1_gamma[i])  # (N, 64, 1, 97)
        h = h.permute(0, 3, 2, 1)  # (N, 97, 1, 64)
        h = F.relu(self.tok_up[i](h))  # (N, 384, 1, 64)
        h = self.tok_down[i](h)  # (N, 97, 1, 64)
        h = h.permute(0, 3, 2, 1)  # (N, 64, 1, 97)
        x = x + h

        # ── Channel mixing ────────────────────────────────────────────────
        h = self._ln(x, self.ln2_gamma[i])  # (N, 64, 1, 97)
        h = F.relu(self.ch_up[i](h))  # (N, 256, 1, 97)
        h = self.ch_down[i](h)  # (N, 64, 1, 97)
        x = x + h

        return x

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (N, 146, 2)  float32 — raw 2-D landmark coordinates

        Returns:
            (N, 52)  float32 — blendshape coefficients in (0, 1)
        """
        dtype = self.compute_dtype

        # ── Preprocessing ────────────────────────────────────────────────
        # Cast early so all arithmetic runs in compute_dtype
        x = x.to(dtype)

        # 1. Centre: subtract mean landmark position per sample
        centroid = x.mean(dim=1, keepdim=True)  # (N, 1, 2)
        x = x - centroid  # (N, 146, 2)

        # 2. Scale: divide by mean L2 norm of centred landmarks, then × 0.5
        norms = x.pow(2).sum(dim=2, keepdim=True).sqrt()  # (N, 146, 1)
        scale = norms.mean(dim=1, keepdim=True)  # (N, 1, 1)
        x = x / (scale + 1e-12) * 0.5  # (N, 146, 2)

        # 3. Reshape to NCHW: (N, 146, 1, 2)
        N = x.shape[0]
        x = x.reshape(N, self.N_LANDMARKS, 1, 2)

        # ── Embedding ────────────────────────────────────────────────────
        x = self.emb1(x)  # (N, 96, 1, 2)
        x = x.permute(0, 3, 2, 1)  # (N, 2, 1, 96)
        x = self.emb2(x)  # (N, 64, 1, 96)

        # Prepend CLS token (broadcast over batch)
        cls = self.cls_token.expand(N, -1, -1, -1)  # (N, 64, 1, 1)
        x = torch.cat([cls, x], dim=3)  # (N, 64, 1, 97)

        # ── MixerBlocks ──────────────────────────────────────────────────
        for i in range(4):
            x = self._mixer_block(x, i)

        # ── Head ─────────────────────────────────────────────────────────
        # Extract CLS token (W-index 0)
        cls_out = x[:, :, :, 0:1]  # (N, 64, 1, 1)
        out = self.head(cls_out)  # (N, 52, 1, 1)
        out = torch.sigmoid(out)
        return out.reshape(N, self.N_CLASSES).float()  # (N, 52) float32

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_onnx(
        cls,
        onnx_path: Union[str, pathlib.Path],
        compute_dtype: torch.dtype = torch.float16,
    ) -> "FaceBlendShapesTorch":
        """
        Construct a FaceBlendShapesTorch and load all weights from the ONNX
        model.

        Conv weight/bias pairs are loaded positionally (19 total, in node
        traversal order).  LN scale vectors and the CLS token are loaded by
        their ONNX initialiser names.
        """
        import onnx
        from onnx import numpy_helper

        proto = onnx.load(str(onnx_path))
        g = proto.graph
        init_map = {init.name: numpy_helper.to_array(init) for init in g.initializer}

        # ── Collect Conv weights in node order ───────────────────────────
        conv_params: list = []
        for node in g.node:
            if node.op_type == "Conv":
                w = init_map.get(node.input[1]) if len(node.input) > 1 else None
                b = init_map.get(node.input[2]) if len(node.input) > 2 else None
                if w is not None:
                    conv_params.append((w, b))

        # ── Build model and assign ────────────────────────────────────────
        m = cls(compute_dtype=compute_dtype)
        ci = 0

        def _c(layer: nn.Conv2d) -> None:
            nonlocal ci
            w, b = conv_params[ci]
            ci += 1
            layer.weight.data = torch.from_numpy(w.copy()).to(compute_dtype)
            if b is not None and layer.bias is not None:
                layer.bias.data = torch.from_numpy(b.copy()).to(compute_dtype)

        # Embedding
        _c(m.emb1)
        _c(m.emb2)

        # 4 MixerBlocks — ONNX Conv order: tok_up, tok_down, ch_up, ch_down
        for i in range(4):
            _c(m.tok_up[i])
            _c(m.tok_down[i])
            _c(m.ch_up[i])
            _c(m.ch_down[i])

        # Output head
        _c(m.head)

        assert ci == len(conv_params), (
            f"[face_blendshapes] Conv mismatch: assigned {ci}/{len(conv_params)}"
        )

        # ── LN scale parameters ───────────────────────────────────────────
        for i, name in enumerate(_LN1_GAMMA_NAMES):
            m.ln1_gamma[i].data = torch.from_numpy(init_map[name].copy()).to(
                compute_dtype
            )
        for i, name in enumerate(_LN2_GAMMA_NAMES):
            m.ln2_gamma[i].data = torch.from_numpy(init_map[name].copy()).to(
                compute_dtype
            )

        # ── CLS token ────────────────────────────────────────────────────
        m.cls_token.data = torch.from_numpy(init_map[_CLS_TOKEN_NAME].copy()).to(
            compute_dtype
        )

        print(
            f"[face_blendshapes] loaded: {ci} Conv + 8 LN-scale + 1 CLS weight tensors"
        )
        m._visomaster_onnx_path = str(onnx_path)
        return m


# ---------------------------------------------------------------------------
# CUDA graph runner
# ---------------------------------------------------------------------------


class FaceBlendShapesCUDAGraphRunner:
    """Wraps FaceBlendShapesTorch in a CUDA graph for minimal kernel-launch overhead."""

    def __init__(
        self,
        model: FaceBlendShapesTorch,
        input_shape: tuple = (1, 146, 2),
    ):
        self.model = model
        self.device = next(model.parameters()).device

        self._x_buf = torch.zeros(input_shape, dtype=torch.float32, device=self.device)

        # Warm-up: cuDNN auto-tune + workspace allocation
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
                self._out = model(self._x_buf)  # (1, 52)
        torch.cuda.synchronize()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (1, 146, 2)  float32 CUDA
        Returns:
            (1, 52)  float32 — blendshape coefficients
        """
        self._x_buf.copy_(x)
        self._graph.replay()
        return self._out.clone()


def build_cuda_graph_runner(
    model: FaceBlendShapesTorch,
    input_shape: tuple = (1, 146, 2),
    torch_compile: bool = False,
) -> FaceBlendShapesCUDAGraphRunner:
    """
    Build and return a CUDA-graph-backed runner for FaceBlendShapesTorch.

    Args:
        model         : FaceBlendShapesTorch on CUDA in eval() mode.
        input_shape   : fixed input shape (default (1, 146, 2)).
        torch_compile : if True, apply torch.compile before building the CUDA graph.
    """
    if torch_compile:
        try:
            from custom_kernels.compile_utils import apply_torch_compile
            device = next(model.parameters()).device
            example_inp = torch.zeros(input_shape, dtype=torch.float32, device=device)
            compiled = apply_torch_compile(model, example_inp)
            print("[face_blendshapes] torch.compile warmup done.")
            return compiled  # CUDA graph on top of torch.compile fails on Windows
        except Exception as e:
            print(f"[face_blendshapes] torch.compile failed ({e!s:.120}), falling back to CUDA graph.")
    return FaceBlendShapesCUDAGraphRunner(model, input_shape=input_shape)
