"""
FP16 PyTorch reimplementation of ``model_assets/landmark.onnx`` — the
ConvNeXt-Tiny 203-point face landmark detector used for ``FaceLandmark203``.

Architecture
------------
    Input  : (1, 3, 224, 224)  float32
    Outputs: (1, 214), (1, 262), (1, 406)   float32
              ↑ coeff   ↑ lmk    ↑ pts (only output[2] used by the application)

Weight loading
--------------
    Named   — 150 initializers whose ONNX names match PyTorch state_dict keys
              directly (dwconv, block norm/gamma, head norms, Gemm heads).
    Anon-LN — 4 × (Mul weight + Add bias) pairs with shapes (C,1,1), loaded
              positionally into the four _ChannelsFirstLN layers.
    Anon-MM — 36 anonymous MatMul weights for pwconv1/pwconv2 in all 18 blocks,
              loaded positionally in forward-pass order; each transposed (.T)
              before assignment to nn.Linear.weight.
"""

from __future__ import annotations

import pathlib
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Optional Triton LayerNorm
# ---------------------------------------------------------------------------
try:
    from custom_kernels.triton_ops import triton_layernorm, TRITON_AVAILABLE
except ImportError:
    TRITON_AVAILABLE = False
    triton_layernorm = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class _ChannelsFirstLN(nn.Module):
    """
    LayerNorm for NCHW tensors — normalises over dim=1 (channel axis) at each
    spatial position (h, w).  Weight and bias stored as (C, 1, 1) to match the
    anonymous ONNX Mul/Add initialiser shapes for direct in-place loading.
    """

    def __init__(self, C: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(C, 1, 1))
        self.bias = nn.Parameter(torch.zeros(C, 1, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # FP32 accumulation for numerical stability; keep output dtype as input.
        xf = x.float()
        mean = xf.mean(dim=1, keepdim=True)
        var = ((xf - mean) ** 2).mean(dim=1, keepdim=True)
        xn = (xf - mean) / (var + self.eps).sqrt()
        return (xn * self.weight.float() + self.bias.float()).to(x.dtype)


class _ConvNeXtBlock(nn.Module):
    """
    Standard ConvNeXt block.

    Structure:
        dwconv (depth-wise 7×7) → permute NCHW→NHWC →
        LayerNorm (over C, last dim) →
        pwconv1 (Linear dim→4*dim) → GELU →
        pwconv2 (Linear 4*dim→dim) →
        gamma (layer-scale) → permute NHWC→NCHW →
        residual add.
    """

    def __init__(self, dim: int, use_triton_ln: bool = False):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(1e-6 * torch.ones(dim))
        self._use_triton_ln = use_triton_ln and TRITON_AVAILABLE

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.dwconv(x)  # (N, C, H, W)
        h = h.permute(0, 2, 3, 1)  # (N, H, W, C)
        if self._use_triton_ln and h.dtype == torch.float16:
            h = triton_layernorm(h, self.norm.weight, self.norm.bias, eps=self.norm.eps)
        else:
            h = self.norm(h.float()).to(x.dtype)
        h = self.pwconv1(h)
        h = F.gelu(h)
        h = self.pwconv2(h)
        h = self.gamma * h
        h = h.permute(0, 3, 1, 2)  # (N, C, H, W)
        return x + h


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------


class Landmark203Torch(nn.Module):
    """
    ConvNeXt-Tiny face landmark model (landmark.onnx).

    Channels : [96, 192, 384, 768]
    Depths   : [3, 3, 9, 3]
    """

    CHANNELS = [96, 192, 384, 768]
    DEPTHS = [3, 3, 9, 3]

    def __init__(
        self, compute_dtype: torch.dtype = torch.float16, use_triton_ln: bool = True
    ):
        super().__init__()
        C = self.CHANNELS
        D = self.DEPTHS
        self.compute_dtype = compute_dtype
        uln = use_triton_ln

        # -- Downsampling / stem layers ------------------------------------------
        # layers[0]: Conv2d(3→96, k=4, s=4) + _ChannelsFirstLN(96)
        # layers[1]: _ChannelsFirstLN(96) + Conv2d(96→192, k=2, s=2)
        # layers[2]: _ChannelsFirstLN(192) + Conv2d(192→384, k=2, s=2)
        # layers[3]: _ChannelsFirstLN(384) + Conv2d(384→768, k=2, s=2)
        self.downsample_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(3, C[0], kernel_size=4, stride=4),
                    _ChannelsFirstLN(C[0]),
                ),
                nn.Sequential(
                    _ChannelsFirstLN(C[0]),
                    nn.Conv2d(C[0], C[1], kernel_size=2, stride=2),
                ),
                nn.Sequential(
                    _ChannelsFirstLN(C[1]),
                    nn.Conv2d(C[1], C[2], kernel_size=2, stride=2),
                ),
                nn.Sequential(
                    _ChannelsFirstLN(C[2]),
                    nn.Conv2d(C[2], C[3], kernel_size=2, stride=2),
                ),
            ]
        )

        # -- Stages --------------------------------------------------------------
        self.stages = nn.ModuleList(
            [
                nn.Sequential(
                    *[_ConvNeXtBlock(C[i], use_triton_ln=uln) for _ in range(D[i])]
                )
                for i in range(4)
            ]
        )

        # -- Output norms --------------------------------------------------------
        # norm_s3 normalises stage-3 (384-ch) GAP features before fc_pts.
        # norm    normalises stage-4 (768-ch) GAP features for all heads.
        self.norm_s3 = nn.LayerNorm(C[2], eps=1e-6)  # 384-ch
        self.norm = nn.LayerNorm(C[3], eps=1e-6)  # 768-ch

        # -- Output heads --------------------------------------------------------
        # fc_pts takes concatenated (384 + 768 = 1152) features
        self.fc_coeff = nn.Linear(C[3], 214)
        self.fc_lmk = nn.Linear(C[3], 262)
        self.fc_pts = nn.Linear(C[2] + C[3], 406)

    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor):
        """
        Args:
            x : (1, 3, 224, 224)  float32 CUDA  — pixel values in [0, 1]
        Returns:
            tuple of three tensors (all float32):
                out_coeff : (1, 214)
                out_lmk   : (1, 262)
                out_pts   : (1, 406)   ← used by detect_face_landmark_203
        """
        x = x.to(self.compute_dtype)

        # Stem + stage 0  →  (1, 96, 56, 56)
        x = self.downsample_layers[0](x)
        x = self.stages[0](x)

        # Transition 1 + stage 1  →  (1, 192, 28, 28)
        x = self.downsample_layers[1](x)
        x = self.stages[1](x)

        # Transition 2 + stage 2  →  (1, 384, 14, 14)
        x = self.downsample_layers[2](x)
        x = self.stages[2](x)

        # Global average pool + norm for stage-3 features
        # norm_s3/norm are FP32 LayerNorm; cast output back to compute_dtype for FC
        feat3 = self.norm_s3(x.float().mean([-2, -1])).to(
            self.compute_dtype
        )  # (1, 384)

        # Transition 3 + stage 3  →  (1, 768, 7, 7)
        x = self.downsample_layers[3](x)
        x = self.stages[3](x)

        # Global average pool + norm for stage-4 features
        feat4 = self.norm(x.float().mean([-2, -1])).to(self.compute_dtype)  # (1, 768)

        # Heads
        out_coeff = self.fc_coeff(feat4)  # (1, 214)
        out_lmk = self.fc_lmk(feat4)  # (1, 262)
        out_pts = self.fc_pts(torch.cat([feat3, feat4], -1))  # (1, 406)

        return out_coeff.float(), out_lmk.float(), out_pts.float()

    # ------------------------------------------------------------------

    @classmethod
    def from_onnx(
        cls,
        onnx_path: Union[str, pathlib.Path],
        compute_dtype: torch.dtype = torch.float16,
        use_triton_ln: bool = True,
    ) -> "Landmark203Torch":
        """
        Build a Landmark203Torch and load weights from the ONNX model.

        Weight mapping
        --------------
        Named    : 150 initializers whose ONNX names match PyTorch state_dict
                   keys directly → loaded via load_state_dict(strict=False).
        Anon-CLN : 4 × (Mul weight + Add bias) (C,1,1) tensors in forward order:
                   stem-LN, dl1-LN, dl2-LN, dl3-LN → _ChannelsFirstLN params.
        Anon-MM  : 36 anonymous MatMul weights, one per (block, pwconv) in
                   forward-pass order; transposed to (out, in) for nn.Linear.
        """
        import onnx
        from onnx import numpy_helper

        model_proto = onnx.load(str(onnx_path))

        # ---- Build initialiser lookup ----------------------------------------
        init_map = {init.name: init for init in model_proto.graph.initializer}

        # Collect anonymous tensors in ONNX node topological order
        named_inits: dict = {}
        anon_cln_mul: list = []  # (C,1,1) channel-first LN scale
        anon_cln_add: list = []  # (C,1,1) channel-first LN bias
        anon_matmul: list = []  # anonymous MatMul weight matrices

        # First pass: categorize all initializers
        anon_mul_names: set[str] = set()
        anon_add_names: set[str] = set()
        anon_mm_names: set[str] = set()

        for init in model_proto.graph.initializer:
            name = init.name
            arr = numpy_helper.to_array(init)
            if name.startswith("onnx::Mul_") and arr.ndim == 3:
                anon_mul_names.add(name)
            elif name.startswith("onnx::Add_") and arr.ndim == 3:
                anon_add_names.add(name)
            elif name.startswith("onnx::MatMul_"):
                anon_mm_names.add(name)
            else:
                named_inits[name] = arr

        # Second pass: collect in topological order from graph nodes
        seen_mul: set[str] = set()
        seen_add: set[str] = set()
        seen_mm: set[str] = set()

        for node in model_proto.graph.node:
            if node.op_type == "Mul":
                for inp in node.input:
                    if inp in anon_mul_names and inp not in seen_mul:
                        anon_cln_mul.append(numpy_helper.to_array(init_map[inp]))
                        seen_mul.add(inp)
            elif node.op_type == "Add":
                for inp in node.input:
                    if inp in anon_add_names and inp not in seen_add:
                        anon_cln_add.append(numpy_helper.to_array(init_map[inp]))
                        seen_add.add(inp)
            elif node.op_type == "MatMul":
                # Weight is the second input of MatMul
                if len(node.input) >= 2:
                    w_name = node.input[1]
                    if w_name in anon_mm_names and w_name not in seen_mm:
                        anon_matmul.append(numpy_helper.to_array(init_map[w_name]))
                        seen_mm.add(w_name)

        # ---- Construct model -------------------------------------------------
        m = cls(compute_dtype=compute_dtype, use_triton_ln=use_triton_ln)

        # ---- Load named params via direct assignment --------------------------
        # load_state_dict is avoided because it uses copy_() internally, which
        # coerces the source dtype to match the destination (FP32 buffers would
        # silently undo our FP16 cast).  Direct .data = assignment replaces the
        # buffer in-place with the correctly-typed tensor.
        #
        # Identify all LayerNorm parameter names so they stay FP32 (PyTorch
        # layer_norm requires weight/bias to match the FP32 input dtype).
        ln_param_names: set = set()
        for mod_name, mod in m.named_modules():
            if isinstance(mod, nn.LayerNorm):
                for p_name in mod._parameters:
                    full = f"{mod_name}.{p_name}" if mod_name else p_name
                    ln_param_names.add(full)

        all_params = dict(m.named_parameters())
        all_buffers = dict(m.named_buffers())
        all_tensors = {**all_params, **all_buffers}
        loaded, skipped = [], []
        for name, arr in named_inits.items():
            if name in all_tensors:
                t = torch.tensor(arr, dtype=torch.float32)  # always start FP32
                target = all_tensors[name]
                if t.shape == target.data.shape:
                    # LayerNorm weight/bias stay FP32; everything else → compute_dtype.
                    target.data = t if name in ln_param_names else t.to(compute_dtype)
                    loaded.append(name)
                else:
                    skipped.append((name, tuple(t.shape), tuple(target.data.shape)))
            # else: ONNX-internal constant not in our model — silently skip

        if skipped:
            for name, got, exp in skipped:
                print(f"[landmark_203] shape mismatch: {name} got {got} expected {exp}")

        # ---- Load anonymous channel-first LN weights -------------------------
        cln_targets = [
            m.downsample_layers[0][1],  # stem LN  (after Conv2d 3→96)
            m.downsample_layers[1][0],  # dl1  LN  (before Conv2d 96→192)
            m.downsample_layers[2][0],  # dl2  LN  (before Conv2d 192→384)
            m.downsample_layers[3][0],  # dl3  LN  (before Conv2d 384→768)
        ]

        if len(anon_cln_mul) == 4 and len(anon_cln_add) == 4:
            for cln, mul_w, add_b in zip(cln_targets, anon_cln_mul, anon_cln_add):
                cln.weight.data = torch.from_numpy(mul_w.copy()).to(
                    compute_dtype
                )  # (C,1,1)
                cln.bias.data = torch.from_numpy(add_b.copy()).to(
                    compute_dtype
                )  # (C,1,1)
        else:
            print(
                f"[landmark_203] WARNING: expected 4 anon-CLN Mul/Add pairs, "
                f"got {len(anon_cln_mul)} Mul / {len(anon_cln_add)} Add"
            )

        # ---- Load anonymous MatMul weights (pwconv1, pwconv2) ----------------
        # ONNX MatMul weight shape: (in_features, out_features)
        # nn.Linear weight shape : (out_features, in_features) — needs .T
        if len(anon_matmul) == 36:
            mm_idx = 0
            for stage in m.stages:
                for block in stage:
                    w1 = (
                        torch.from_numpy(anon_matmul[mm_idx].copy())
                        .T.contiguous()
                        .to(compute_dtype)
                    )
                    mm_idx += 1
                    w2 = (
                        torch.from_numpy(anon_matmul[mm_idx].copy())
                        .T.contiguous()
                        .to(compute_dtype)
                    )
                    mm_idx += 1
                    block.pwconv1.weight.data = w1
                    block.pwconv2.weight.data = w2
        else:
            print(
                f"[landmark_203] WARNING: expected 36 anon MatMul weights, "
                f"got {len(anon_matmul)}"
            )

        total_named = len(loaded)
        print(
            f"[landmark_203] loaded: {total_named} named + "
            f"{len(anon_cln_mul) * 2} anon-CLN + {len(anon_matmul)} anon-MM "
            f"initializers"
        )
        m._visomaster_onnx_path = str(onnx_path)
        return m


# ---------------------------------------------------------------------------
# CUDA graph runner
# ---------------------------------------------------------------------------


class Landmark203CUDAGraphRunner:
    """Wraps a Landmark203Torch in a CUDA graph for minimal kernel-launch overhead."""

    def __init__(self, model: Landmark203Torch, input_shape: tuple = (1, 3, 224, 224)):
        self.model = model
        self.device = next(model.parameters()).device
        dtype = torch.float32  # input is always float32

        # Static input buffer
        self._x_buf = torch.zeros(input_shape, dtype=dtype, device=self.device)

        # Warm-up: run once outside the graph to allocate cuDNN workspace etc.
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
                self._out = model(self._x_buf)  # tuple of 3 tensors
        torch.cuda.synchronize()

    def __call__(self, x: torch.Tensor):
        """
        Args:
            x : (1, 3, 224, 224) float32 CUDA
        Returns:
            tuple (out_coeff, out_lmk, out_pts) — each a fresh CPU-ready tensor
        """
        self._x_buf.copy_(x)
        self._graph.replay()
        # Clone all outputs so callers get stable data independent of the buffer.
        return tuple(o.clone() for o in self._out)


def build_cuda_graph_runner(
    model: Landmark203Torch,
    input_shape: tuple = (1, 3, 224, 224),
    torch_compile: bool = False,
) -> Landmark203CUDAGraphRunner:
    """
    Build and return a CUDA-graph-backed runner for Landmark203Torch.

    Args:
        model         : Landmark203Torch on CUDA, already in eval() mode.
        input_shape   : fixed input shape (default (1, 3, 224, 224)).
        torch_compile : if True, apply torch.compile (mode='default') to the
                        model before CUDA graph capture.  Triggers Triton JIT
                        on first call (~30 s); subsequent calls use compiled
                        kernels inside the captured graph.
    """
    if torch_compile:
        try:
            from custom_kernels.compile_utils import apply_torch_compile
            device = next(model.parameters()).device
            example_inp = torch.zeros((1, 3, 224, 224), dtype=torch.float32, device=device)
            compiled = apply_torch_compile(model, example_inp)
            print("[landmark_203] torch.compile warmup done.")
            return compiled  # CUDA graph on top of torch.compile fails on Windows
        except Exception as e:
            print(f"[landmark_203] torch.compile failed ({e!s:.120}), falling back to CUDA graph.")
    return Landmark203CUDAGraphRunner(model, input_shape=input_shape)
