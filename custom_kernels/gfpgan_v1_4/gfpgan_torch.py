"""
GFPGANTorch — FP16 PyTorch implementation of GFPGANv1.4 (512) and GFPGAN-1024.

Speedup tiers vs ORT FP32 baseline (~14 ms on RTX-class GPU):
  Tier 0: ORT FP32 CUDA EP                                       ~14 ms   1.00x  (baseline)
  Tier 0b: ORT TensorRT EP                                       ~11 ms   1.28x
  Tier 1: PyTorch FP32 pure-ops                                  ~13 ms   1.12x
  Tier 2: PyTorch FP16 + Triton demod + Triton fused-act         ~9.2 ms  1.57x
  Tier 3: PyTorch FP16 + Triton + CUDA graph                     ~7.5 ms  1.91x

Supports both variants via GFPGANTorch.from_onnx():
    model = GFPGANTorch.from_onnx("model_assets/GFPGANv1.4.onnx").cuda().eval()
    model = GFPGANTorch.from_onnx("model_assets/gfpgan-1024.onnx").cuda().eval()
    output = model(input_f32_cuda)  # [1,3,H,W] float32 in/out
"""

from __future__ import annotations

import importlib.util
import math
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Triton kernels (preferred — no MSVC required, works on Windows)
# ---------------------------------------------------------------------------
try:
    from custom_kernels.triton_ops import (
        TRITON_AVAILABLE as _TRITON_AVAILABLE,
        triton_demod as _triton_demod,
        triton_fused_gfpgan_act as _triton_gfpgan_act,
    )
except Exception:
    _TRITON_AVAILABLE = False
    _triton_demod = None  # type: ignore[assignment]
    _triton_gfpgan_act = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Fused weight-demodulation CUDA kernel
# ---------------------------------------------------------------------------
# Per output channel co:
#   w_mod[co,ci,h,w] = weight[co,ci,h,w] * style[ci]
#   demod            = rsqrt( sum_{ci,h,w}(w_mod^2) + eps )
#   result[co,...]   = w_mod[co,...] * demod
# Grid: (C_out,)  Block: 256 threads  warp-shuffle reduction
# ---------------------------------------------------------------------------

_EXT_NAME = "gfpgan_demod_ext"
_LEGACY_BUILD_DIR = Path(__file__).parent / "_demod_build"
# Primary location: multi-arch fat binary in model_assets/custom_kernels/
_SHARED_DIR = Path(__file__).parent.parent.parent / "model_assets" / "custom_kernels"
_SHARED_DIR.mkdir(parents=True, exist_ok=True)

_CUDA_SRC = r"""
#include <cuda_runtime.h>
#include <math.h>

__device__ __forceinline__ float warp_reduce(float v) {
    for (int d = 16; d > 0; d >>= 1) v += __shfl_down_sync(0xffffffff, v, d);
    return v;
}

__global__ void fused_demod_kernel(
    const float* __restrict__ weight,   // [C_out, n]   n = C_in * kH * kW
    const float* __restrict__ style,    // [C_in]
    float*       __restrict__ out,      // [C_out, n]
    int C_out, int n, int kHkW, float eps)
{
    int co = blockIdx.x;
    if (co >= C_out) return;
    const float* w = weight + (long long)co * n;
    float*       o = out    + (long long)co * n;

    float sumsq = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float ws = w[i] * style[i / kHkW];
        sumsq += ws * ws;
    }
    sumsq = warp_reduce(sumsq);
    __shared__ float smem[8];
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) smem[warp] = sumsq;
    __syncthreads();
    if (warp == 0) {
        sumsq = (lane < blockDim.x / 32) ? smem[lane] : 0.f;
        sumsq = warp_reduce(sumsq);
    }
    __shared__ float demod_s;
    if (threadIdx.x == 0) demod_s = rsqrtf(sumsq + eps);
    __syncthreads();
    float demod = demod_s;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        o[i] = w[i] * style[i / kHkW] * demod;
}
"""

_CPP_SRC = r"""
#include <torch/extension.h>
void fused_demod_kernel(const float*, const float*, float*, int, int, int, float);

torch::Tensor fused_demod(
    torch::Tensor weight,
    torch::Tensor style,
    float eps)
{
    TORCH_CHECK(weight.is_cuda() && style.is_cuda());
    TORCH_CHECK(weight.dim() == 4 && weight.scalar_type() == torch::kFloat32);
    int C_out = weight.size(0);
    int C_in  = weight.size(1);
    int kHkW  = weight.size(2) * weight.size(3);
    auto out  = torch::empty_like(weight);
    fused_demod_kernel<<<C_out, 256>>>(
        weight.data_ptr<float>(), style.data_ptr<float>(),
        out.data_ptr<float>(), C_out, C_in * kHkW, kHkW, eps);
    return out;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_demod", &fused_demod, "Fused weight demod (CUDA)");
}
"""

_ext_obj = False  # False = not tried, None = failed, module = success


def _add_dll_dirs():
    if sys.platform == "win32":
        try:
            torch_lib = Path(torch.__file__).parent / "lib"
            if torch_lib.is_dir():
                os.add_dll_directory(str(torch_lib))
        except Exception:
            pass


def _try_load_pyd(pyd_path: Path, tag: str) -> bool:
    """Try to load a pre-built .pyd; return True on success."""
    global _ext_obj
    if not pyd_path.exists():
        return False
    try:
        spec = importlib.util.spec_from_file_location(_EXT_NAME, str(pyd_path))
        assert spec is not None
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        _ext_obj = mod  # type: ignore[assignment]
        print(f"[GFPGANTorch] Demod extension loaded ({tag}).")
        return True
    except Exception as e:
        print(f"[GFPGANTorch] Load failed ({tag}): {e}")
        return False


def _load_ext():
    global _ext_obj
    _add_dll_dirs()

    # 1. Shared multi-arch binary (model_assets/custom_kernels/)
    if _try_load_pyd(_SHARED_DIR / f"{_EXT_NAME}.pyd", "shared multi-arch"):
        return

    # 2. Legacy per-module build dir
    if _try_load_pyd(_LEGACY_BUILD_DIR / f"{_EXT_NAME}.pyd", "legacy build dir"):
        return

    # 3. JIT compile (fallback — requires MSVC on Windows / GCC on Linux)
    import sys as _sys
    import shutil as _shutil

    if (
        _sys.platform == "win32"
        and _shutil.which("cl") is None
        and _shutil.which("cl.exe") is None
    ):
        print(
            "[GFPGANTorch] Demod CUDA extension not found and CL.exe is unavailable.\n"
            "  Run: python custom_kernels/build_kernels.py  (requires Visual Studio Build Tools)\n"
            "  -> Using Triton / pure-PyTorch demod fallback."
        )
        _ext_obj = None
        return
    print("[GFPGANTorch] Compiling demod extension for current GPU (requires MSVC)...")
    try:
        from torch.utils.cpp_extension import load_inline

        _SHARED_DIR.mkdir(parents=True, exist_ok=True)
        _ext_obj = load_inline(
            name=_EXT_NAME,
            cuda_sources=[_CUDA_SRC],
            cpp_sources=[_CPP_SRC],
            build_directory=str(_SHARED_DIR),
            extra_cuda_cflags=["--use_fast_math", "-O3"],
            verbose=False,
        )
        print("[GFPGANTorch] Demod extension compiled and ready.")
    except Exception as e:
        print(f"[GFPGANTorch] Compile failed: {e}  ->  using pure-PyTorch fallback.")
        _ext_obj = None


def _get_demod_fn():
    global _ext_obj
    if _ext_obj is False:
        _load_ext()
    return None if _ext_obj is None else _ext_obj.fused_demod


def _fused_demod(
    w: torch.Tensor, style: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """
    w     : [C_out, C_in, kH, kW]  FP16 or FP32  CUDA
    style : [C_in]                  FP32  CUDA
    returns: [C_out, C_in, kH, kW]  same dtype as w  CUDA  (demodulated)
    """
    # When being traced by torch.compile/dynamo, skip custom Triton/@triton.jit and
    # CUDA C++ ops — they create graph breaks and trigger Triton MLIR compilation in
    # the subprocess, which crashes on Windows (sm_89 MLIR optimizer AV).  Fall
    # through to the pure PyTorch path so inductor can fuse the entire demod op into
    # a single efficient kernel with no graph breaks.
    # In eager mode (CUDA graph capture / direct inference) the fast custom paths below
    # are still used.
    if not torch.compiler.is_compiling():
        # Priority 1: Triton (Windows-friendly, no MSVC needed)
        if _TRITON_AVAILABLE and _triton_demod is not None:
            return _triton_demod(
                w.to(torch.float16).contiguous(), style.contiguous().float(), eps
            )
        # Priority 2: CUDA C++ extension
        fn = _get_demod_fn()
        if fn is not None:
            return fn(w.contiguous().float(), style.contiguous().float(), eps)
    # Priority 3: Pure PyTorch — fully traceable by dynamo, fused by inductor
    wm = w.float() * style.view(1, -1, 1, 1)
    return (wm * torch.rsqrt(wm.pow(2).sum([1, 2, 3], keepdim=True) + eps)).to(w.dtype)


# ---------------------------------------------------------------------------
# ONNX helpers
# ---------------------------------------------------------------------------


def _load_onnx(path: str) -> dict:
    """Load ONNX initializers and alias auto-named to_rgb biases to standard keys."""
    import onnx
    from onnx import numpy_helper

    m = onnx.load(path)
    w = {i.name: numpy_helper.to_array(i).copy() for i in m.graph.initializer}

    # The to_rgb Conv nodes store their bias as auto-named initializers
    # (e.g. _v_818) instead of the expected dot-name pattern.
    # Detect them in graph order and register under standard alias keys.
    init_shapes = {i.name: numpy_helper.to_array(i).shape for i in m.graph.initializer}
    torgb_bias_keys: list = []
    for node in m.graph.node:
        if node.op_type != "Conv":
            continue
        ks = None
        for attr in node.attribute:
            if attr.name == "kernel_shape":
                ks = list(attr.ints)
        if ks != [1, 1]:
            continue
        if len(node.input) >= 3 and node.input[2] in init_shapes:
            if init_shapes[node.input[2]] == (3,):
                torgb_bias_keys.append(node.input[2])

    if torgb_bias_keys:
        w["_torgb_bias_rgb1"] = w[torgb_bias_keys[0]]
        for i, k in enumerate(torgb_bias_keys[1:8]):
            w[f"_torgb_bias_rgbs{i}"] = w[k]
        if len(torgb_bias_keys) >= 9:
            w["_torgb_bias_final_rgb"] = w[torgb_bias_keys[8]]

    # StyleGAN constant input: shape [1, 512, 4, 4].
    # The initializer name varies across ONNX export versions:
    #   - Constant-folded Tile output: '/stylegan_decoderdotconstant_input/Tile_output_0'
    #   - Plain parameter export:      'stylegan_decoder.constant_input.input'
    # Detect by trying known names first, then fall back to unique-shape search.
    _SG_CONST_KNOWN = (
        "/stylegan_decoderdotconstant_input/Tile_output_0",
        "stylegan_decoder.constant_input.input",
        "/stylegan_decoder/constant_input/Tile_output_0",
    )
    sg_const_src = next((k for k in _SG_CONST_KNOWN if k in w), None)
    if sg_const_src is None:
        # Shape-based fallback: the constant input is the only [1, 512, 4, 4] initializer
        sg_const_src = next(
            (k for k, v in w.items() if getattr(v, "shape", None) == (1, 512, 4, 4)),
            None,
        )
    if sg_const_src is not None:
        w["_sg_const"] = w[sg_const_src]
    else:
        print(
            "[GFPGANTorch] WARNING: Could not locate StyleGAN constant input "
            "in ONNX model. Available shape-(1,512,4,4) keys: "
            + str(
                [k for k, v in w.items() if getattr(v, "shape", None) == (1, 512, 4, 4)]
            )
        )

    return w


def _h16(a: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(a).to(torch.float16).contiguous()


def _f32(a: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(a).to(torch.float32).contiguous()


# ---------------------------------------------------------------------------
# GFPGANTorch
# ---------------------------------------------------------------------------


class GFPGANTorch(torch.nn.Module):
    """
    FP16 PyTorch reimplementation of GFPGAN.
    Works for v1.4 (out_size=512) and GFPGAN-1024 (out_size=1024).
    """

    def __init__(self, out_size: int = 512, compute_dtype: torch.dtype = torch.float16):
        super().__init__()
        assert out_size in (512, 1024)
        self.out_size = out_size
        self.log_size = int(math.log2(out_size))  # 9 or 10
        self.num_latents = self.log_size * 2 - 2  # 16 or 18
        self.is_1024 = out_size == 1024
        self.n_loop = 7  # same for both variants
        self.compute_dtype = compute_dtype  # float16 (fast) or float32 (reference)

    @classmethod
    def from_onnx(
        cls, onnx_path: str, compute_dtype: torch.dtype = torch.float16
    ) -> "GFPGANTorch":
        w = _load_onnx(onnx_path)
        out_size = 1024 if "final_extend_linear.weight" in w else 512
        model = cls(out_size, compute_dtype=compute_dtype)
        model._load_weights(w)
        model._visomaster_onnx_path = str(onnx_path)
        return model

    # ------------------------------------------------------------------
    # Weight registration
    # ------------------------------------------------------------------

    def _load_weights(self, w: dict):
        rb = self.register_buffer
        cdtype = self.compute_dtype  # float16 or float32

        def h16(k):
            return torch.from_numpy(w[k]).to(cdtype).contiguous()

        def f32(k):
            return _f32(w[k])

        def h16_opt(k, shape):
            """Load with compute_dtype or fall back to zeros if key absent."""
            if k in w:
                return torch.from_numpy(w[k]).to(cdtype).contiguous()
            return torch.zeros(shape, dtype=cdtype)

        def h16_rgb_bias(alias_key):
            """Load a to_rgb bias from the aliased key (shape [3] -> [1,3,1,1])."""
            if alias_key in w:
                return (
                    torch.from_numpy(w[alias_key])
                    .to(cdtype)
                    .view(1, 3, 1, 1)
                    .contiguous()
                )
            return torch.zeros(1, 3, 1, 1, dtype=cdtype)

        # ── U-Net encoder ─────────────────────────────────────────────────
        rb("cbf_w", h16("conv_body_first.weight"))
        rb("cbf_b", h16("conv_body_first.bias"))
        for i in range(7):
            for attr in (
                "conv1.weight",
                "conv1.bias",
                "conv2.weight",
                "conv2.bias",
                "skip.weight",
            ):
                rb(
                    f"cbd_{i}_" + attr.replace(".", "_"),
                    h16(f"conv_body_down.{i}.{attr}"),
                )

        rb("fc_w", h16("final_conv.weight"))
        rb("fc_b", h16("final_conv.bias"))

        # ── Latent projections (FP32 for accuracy) ────────────────────────
        rb("fl_w", f32("final_linear.weight"))
        rb("fl_b", f32("final_linear.bias"))
        if self.is_1024:
            rb("fel_w", f32("final_extend_linear.weight"))
            rb("fel_b", f32("final_extend_linear.bias"))

        # ── U-Net decoder ─────────────────────────────────────────────────
        for i in range(7):
            for attr in (
                "conv1.weight",
                "conv1.bias",
                "conv2.weight",
                "conv2.bias",
                "skip.weight",
            ):
                rb(
                    f"cbu_{i}_" + attr.replace(".", "_"),
                    h16(f"conv_body_up.{i}.{attr}"),
                )

        # ── SFT condition networks  (condition_scale / condition_shift) ───
        for branch, short in (("condition_scale", "cs"), ("condition_shift", "ch")):
            for i in range(7):
                for li, ls in ((0, "0"), (2, "2")):
                    for p in ("weight", "bias"):
                        rb(f"{short}_{i}_{ls}_{p}", h16(f"{branch}.{i}.{li}.{p}"))

        # ── 1024-only: final_body_up + final_scale / final_shift ──────────
        if self.is_1024:
            for attr in (
                "conv1.weight",
                "conv1.bias",
                "conv2.weight",
                "conv2.bias",
                "skip.weight",
            ):
                rb("fbu_" + attr.replace(".", "_"), h16(f"final_body_up.{attr}"))
            for branch, short in (("final_scale", "fs"), ("final_shift", "fh")):
                for li, ls in ((0, "0"), (2, "2")):
                    for p in ("weight", "bias"):
                        rb(f"{short}_{ls}_{p}", h16(f"{branch}.{li}.{p}"))

        # ── StyleGAN constant input [1, 512, 4, 4] ────────────────────────
        rb("sg_const", h16("_sg_const"))

        # ── style_conv1 (prefix "init_sc" avoids collision with sc1 in the loop) ──
        pf = "stylegan_decoderdotstyle_conv1dot"
        rb("init_sc_mod_w", f32(f"{pf}modulated_convdotmodulation.weight"))
        rb("init_sc_mod_b", f32(f"{pf}modulated_convdotmodulation.bias"))
        rb("init_sc_w", h16(f"{pf}modulated_convdotweight"))
        rb("init_sc_b", h16(f"{pf}bias"))

        # ── to_rgb1 ───────────────────────────────────────────────────────
        pf = "stylegan_decoderdotto_rgb1dot"
        rb("rgb1_mod_w", f32(f"{pf}modulated_convdotmodulation.weight"))
        rb("rgb1_mod_b", f32(f"{pf}modulated_convdotmodulation.bias"))
        rb("rgb1_w", h16(f"{pf}modulated_convdotweight"))
        rb("rgb1_b", h16_rgb_bias("_torgb_bias_rgb1"))

        # ── style_convsdot 0-13 ───────────────────────────────────────────
        for i in range(14):
            pf = f"stylegan_decoderdotstyle_convsdot{i}dot"
            rb(f"sc{i}_mod_w", f32(f"{pf}modulated_convdotmodulation.weight"))
            rb(f"sc{i}_mod_b", f32(f"{pf}modulated_convdotmodulation.bias"))
            rb(f"sc{i}_w", h16(f"{pf}modulated_convdotweight"))
            C_out_i = w[f"{pf}modulated_convdotweight"].shape[1]
            rb(f"sc{i}_b", h16_opt(f"{pf}bias", (1, C_out_i, 1, 1)))

        # ── to_rgbs 0-6 ───────────────────────────────────────────────────
        for i in range(7):
            pf = f"stylegan_decoderdotto_rgbsdot{i}dot"
            rb(f"rgbs{i}_mod_w", f32(f"{pf}modulated_convdotmodulation.weight"))
            rb(f"rgbs{i}_mod_b", f32(f"{pf}modulated_convdotmodulation.bias"))
            rb(f"rgbs{i}_w", h16(f"{pf}modulated_convdotweight"))
            rb(f"rgbs{i}_b", h16_rgb_bias(f"_torgb_bias_rgbs{i}"))

        # ── 1024-only: final_conv1 / final_conv2 / final_rgb ─────────────
        if self.is_1024:
            for cname in ("final_conv1", "final_conv2"):
                pf = f"stylegan_decoderdot{cname}dot"
                rb(f"{cname}_mod_w", f32(f"{pf}modulated_convdotmodulation.weight"))
                rb(f"{cname}_mod_b", f32(f"{pf}modulated_convdotmodulation.bias"))
                rb(f"{cname}_w", h16(f"{pf}modulated_convdotweight"))
                C_out_c = w[f"{pf}modulated_convdotweight"].shape[1]
                rb(f"{cname}_b", h16_opt(f"{pf}bias", (1, C_out_c, 1, 1)))
            # final_rgb is a to_rgb layer (bias uses auto-named key)
            pf = "stylegan_decoderdotfinal_rgbdot"
            rb("final_rgb_mod_w", f32(f"{pf}modulated_convdotmodulation.weight"))
            rb("final_rgb_mod_b", f32(f"{pf}modulated_convdotmodulation.bias"))
            rb("final_rgb_w", h16(f"{pf}modulated_convdotweight"))
            rb("final_rgb_b", h16_rgb_bias("_torgb_bias_final_rgb"))

        # ── Fixed noise buffers ───────────────────────────────────────────
        # Noise index mapping:
        #   0        → style_conv1
        #   1..14    → style_convsdot 0..13
        #   15, 16   → final_conv1, final_conv2  (1024 only)
        if self.is_1024:
            noise_keys = [
                "onnx::Add_2180",  # sc1
                "onnx::Add_2231",
                "onnx::Add_2260",  # scdot 0, 1
                "onnx::Add_2312",
                "onnx::Add_2341",  # scdot 2, 3
                "onnx::Add_2393",
                "onnx::Add_2422",  # scdot 4, 5
                "onnx::Add_2474",
                "onnx::Add_2503",  # scdot 6, 7
                "onnx::Add_2555",
                "onnx::Add_2584",  # scdot 8, 9
                "onnx::Add_2636",
                "onnx::Add_2665",  # scdot10,11
                "onnx::Add_2717",
                "onnx::Add_2746",  # scdot12,13
                "onnx::Add_2798",
                "onnx::Add_2827",  # final_conv1,2
            ]
        else:
            noise_keys = [
                "onnx::Add_1341",  # sc1
                "onnx::Add_1349",
                "onnx::Add_1351",  # scdot 0, 1
                "onnx::Add_1365",
                "onnx::Add_1367",  # scdot 2, 3
                "onnx::Add_1381",
                "onnx::Add_1383",  # scdot 4, 5
                "onnx::Add_1397",
                None,  # type: ignore[list-item]  # scdot 6 (has), 7 (missing)
                "onnx::Add_1413",
                "onnx::Add_1415",  # scdot 8, 9
                "onnx::Add_1424",
                "onnx::Add_1426",  # scdot10,11
                "onnx::Add_1435",
                "onnx::Add_1437",  # scdot12,13
            ]
        for idx, nk in enumerate(noise_keys):
            val = h16(nk) if (nk is not None and nk in w) else None
            self.register_buffer(f"noise_{idx}", val)

    # ------------------------------------------------------------------
    # Primitive helpers
    # ------------------------------------------------------------------

    def _resblock(self, x: torch.Tensor, pfx: str, up: bool) -> torch.Tensor:
        g = self._buffers
        scale = 2.0 if up else 0.5
        out = F.leaky_relu(
            F.conv2d(x, g[f"{pfx}_conv1_weight"], g[f"{pfx}_conv1_bias"], padding=1),
            0.2,
        )
        out = F.interpolate(
            out, scale_factor=scale, mode="bilinear", align_corners=False
        )
        out = F.leaky_relu(
            F.conv2d(out, g[f"{pfx}_conv2_weight"], g[f"{pfx}_conv2_bias"], padding=1),
            0.2,
        )
        x = F.interpolate(x, scale_factor=scale, mode="bilinear", align_corners=False)
        return out + F.conv2d(x, g[f"{pfx}_skip_weight"])

    def _sft_net(self, feat: torch.Tensor, short: str, idx: int) -> torch.Tensor:
        """2-layer conv condition net: e.g. condition_scale.i → sft_scale."""
        g = self._buffers
        out = F.leaky_relu(
            F.conv2d(
                feat,
                g[f"{short}_{idx}_0_weight"],
                g[f"{short}_{idx}_0_bias"],
                padding=1,
            ),
            0.2,
        )
        return F.conv2d(
            out, g[f"{short}_{idx}_2_weight"], g[f"{short}_{idx}_2_bias"], padding=1
        )

    def _style_conv(
        self,
        x: torch.Tensor,  # [1, C_in, H, W]  FP16
        w_pfx: str,  # e.g. "sc0"
        latent: torch.Tensor,  # [1, 512]          FP32
        noise_idx: int,  # index into noise_* (-1 = none)
        upsample: bool = False,
        sft_scale: Optional[torch.Tensor] = None,
        sft_shift: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Modulated conv with weight demodulation, optional noise, bias,
        leaky-relu, and SFT (sft_half=True: applied to second half of channels).
        """
        g = self._buffers

        weight = g[f"{w_pfx}_w"]  # [1, C_out, C_in, kH, kW] FP16
        mod_w = g[f"{w_pfx}_mod_w"]  # [C_in, 512]               FP32
        mod_b = g[f"{w_pfx}_mod_b"]  # [C_in]                    FP32
        bias = g[f"{w_pfx}_b"]  # [1, C_out, 1, 1]          FP16

        C_out, _C_in, kH = weight.shape[1], weight.shape[2], weight.shape[3]

        style = F.linear(latent, mod_w, mod_b)  # [1, C_in] FP32
        w_f32 = weight[0].float()  # [C_out, C_in, kH, kW] always FP32
        w_comp = _fused_demod(w_f32, style[0]).to(self.compute_dtype)

        if upsample:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

        out = F.conv2d(x, w_comp, padding=kH // 2)

        noise = None
        if noise_idx >= 0:
            n_buf = g[f"noise_{noise_idx}"]
            if n_buf is not None:
                noise = n_buf

        if (
            _TRITON_AVAILABLE
            and _triton_gfpgan_act is not None
            and out.dtype == torch.float16
            and not torch.compiler.is_compiling()  # skip custom @triton.jit during tracing
        ):
            # Triton kernel now handles broadcasting of noise/bias internally
            out = _triton_gfpgan_act(out, noise, bias, 0.2, 2.0**0.5)
        else:
            out = out * (2.0**0.5)
            if noise is not None:
                out = out + noise
            out = out + bias
            out = F.leaky_relu(out, 0.2)

        if sft_scale is not None:
            half = C_out // 2
            out = torch.cat(
                [out[:, :half], out[:, half:] * sft_scale + sft_shift], dim=1
            )
        return out

    def _torgb_conv(
        self,
        x: torch.Tensor,  # [1, C_in, H, W]  FP16
        w_pfx: str,
        latent: torch.Tensor,  # [1, 512]          FP32
        skip: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        to-RGB layer: modulated 1×1 conv, no demod.
        Optionally adds upsampled skip RGB.
        """
        g = self._buffers

        weight = g[f"{w_pfx}_w"]  # [1, 3, C_in, 1, 1] FP16
        mod_w = g[f"{w_pfx}_mod_w"]  # [C_in, 512]          FP32
        mod_b = g[f"{w_pfx}_mod_b"]  # [C_in]               FP32
        bias = g[f"{w_pfx}_b"]  # [1, 3, 1, 1]         FP16

        C_in = weight.shape[2]

        style = F.linear(latent, mod_w, mod_b)  # [1, C_in] FP32
        w_comp = (weight[0].float() * style[0].view(1, C_in, 1, 1)).to(
            self.compute_dtype
        )

        rgb = F.conv2d(x, w_comp) + bias  # [1, 3, H, W]

        if skip is not None:
            skip = F.interpolate(
                skip, scale_factor=2, mode="bilinear", align_corners=False
            )
            rgb = rgb + skip

        return rgb

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x:       [1, 3, H, W]  float32 CUDA
        returns: [1, 3, H, W]  float32 CUDA
        """
        x = x.to(self.compute_dtype)
        g = self._buffers

        # ── U-Net Encoder ─────────────────────────────────────────────────
        feat = F.leaky_relu(F.conv2d(x, g["cbf_w"], g["cbf_b"]), 0.2)
        skips: list[torch.Tensor] = []
        for i in range(7):
            feat = self._resblock(feat, f"cbd_{i}", up=False)
            skips.insert(0, feat)  # skips[0] = deepest (4×4)

        feat = F.leaky_relu(F.conv2d(feat, g["fc_w"], g["fc_b"], padding=1), 0.2)

        # ── Latent codes ──────────────────────────────────────────────────
        feat_flat = feat.float().view(1, -1)
        style_code = F.linear(feat_flat, g["fl_w"], g["fl_b"])
        if self.is_1024:
            style_code = torch.cat(
                [style_code, F.linear(feat_flat, g["fel_w"], g["fel_b"])], dim=1
            )
        latent = style_code.view(1, self.num_latents, 512)  # [1, L, 512]

        # ── U-Net Decoder + SFT conditions ────────────────────────────────
        sft_scales: list[torch.Tensor] = []
        sft_shifts: list[torch.Tensor] = []
        for i in range(7):
            feat = feat + skips[i]
            feat = self._resblock(feat, f"cbu_{i}", up=True)
            sft_scales.append(self._sft_net(feat, "cs", i))
            sft_shifts.append(self._sft_net(feat, "ch", i))

        if self.is_1024:
            feat = self._resblock(feat, "fbu", up=True)
            final_sft_s = self._sft_net(feat, "fs", -1)  # uses fs_0_*, fs_2_*
            final_sft_h = self._sft_net(feat, "fh", -1)  # uses fh_0_*, fh_2_*

        # ── StyleGAN Decoder ──────────────────────────────────────────────
        sg = g["sg_const"].expand(1, -1, -1, -1).clone()

        # style_conv1 at 4×4  (noise_0, latent[0])
        sg = self._style_conv(sg, "init_sc", latent[:, 0], noise_idx=0)

        # to_rgb1  (latent[1], no skip yet)
        skip_rgb = self._torgb_conv(sg, "rgb1", latent[:, 1])

        sg_li = 1  # current latent index for style_convs in the loop

        # 7 main iterations: resolutions 8, 16, 32, 64, 128, 256, 512
        for li in range(self.n_loop):
            ci = li * 2  # first style_conv index of this pair (0,2,4,6,8,10,12)

            # noise for scdot_ci  = noise_{ci+1}  (noise_0 is sc1, so offset by 1)
            # noise for scdot_ci+1 = noise_{ci+2}
            sg = self._style_conv(
                sg,
                f"sc{ci}",
                latent[:, sg_li],
                noise_idx=ci + 1,
                upsample=True,
                sft_scale=sft_scales[li],
                sft_shift=sft_shifts[li],
            )

            sg = self._style_conv(
                sg, f"sc{ci + 1}", latent[:, sg_li + 1], noise_idx=ci + 2
            )

            skip_rgb = self._torgb_conv(
                sg, f"rgbs{li}", latent[:, sg_li + 2], skip=skip_rgb
            )
            sg_li += 2

        # 1024 final stage (1024×1024)
        if self.is_1024:
            sg = self._style_conv(
                sg,
                "final_conv1",
                latent[:, sg_li],
                noise_idx=15,
                upsample=True,
                sft_scale=final_sft_s,
                sft_shift=final_sft_h,
            )

            sg = self._style_conv(sg, "final_conv2", latent[:, sg_li + 1], noise_idx=16)

            skip_rgb = self._torgb_conv(
                sg, "final_rgb", latent[:, sg_li + 2], skip=skip_rgb
            )

        return skip_rgb.float()

    def _sft_net(self, feat: torch.Tensor, short: str, idx: int) -> torch.Tensor:  # type: ignore[no-redef]
        """
        2-layer conv condition net.
        short="cs"/"ch"/"fs"/"fh", idx=0..6 (or -1 for final_scale/shift).
        """
        g = self._buffers
        if idx >= 0:
            w0k = f"{short}_{idx}_0_weight"
            b0k = f"{short}_{idx}_0_bias"
            w2k = f"{short}_{idx}_2_weight"
            b2k = f"{short}_{idx}_2_bias"
        else:
            # final_scale / final_shift: keys are {short}_0_weight etc.
            w0k = f"{short}_0_weight"
            b0k = f"{short}_0_bias"
            w2k = f"{short}_2_weight"
            b2k = f"{short}_2_bias"
        out = F.leaky_relu(F.conv2d(feat, g[w0k], g[b0k], padding=1), 0.2)
        return F.conv2d(out, g[w2k], g[b2k], padding=1)


# ---------------------------------------------------------------------------
# CUDA Graph runner
# ---------------------------------------------------------------------------


def build_cuda_graph_runner(
    model: GFPGANTorch,
    inp_shape: tuple = (1, 3, 512, 512),
    torch_compile: bool = False,
):
    """
    Capture model() as a CUDAGraph for repeated fixed-shape inference.
    Returns a callable: (x: float32 CUDA) → float32 CUDA.

    Args:
        model:         GFPGANTorch instance (.cuda().eval())
        inp_shape:     Input tensor shape (default: (1, 3, 512, 512))
        torch_compile: If True, apply torch.compile() before CUDA graph capture
                       for additional kernel fusion (~1.2-1.5x over Tier 3).
    """
    dev = next(iter(model.buffers())).device
    static_inp = torch.zeros(inp_shape, dtype=torch.float32, device=dev)

    if torch_compile:
        try:
            from custom_kernels.compile_utils import apply_torch_compile
            example_inp = torch.zeros(inp_shape, dtype=torch.float32, device=dev)
            compiled = apply_torch_compile(model, example_inp)
            print("[GFPGANTorch] torch.compile warmup done.")
            # Fall through to CUDA graph capture with the compiled model.
            # Previously returned early to avoid the 64-bit CUDA handle overflow in
            # _StaticCudaLauncher (ctypes c_long = 32-bit on Windows).  That root
            # cause is now fixed via use_static_cuda_launcher=False.
            # Capturing a CUDA graph on top of the compiled model ensures that inference
            # always replays a fixed CUDA graph — dynamo is never re-invoked, so the
            # recompile_limit=8 warning (from _style_conv/_resblock called with varying
            # channel widths) is fully avoided.
            # The compiled model uses pure-PyTorch demod/act (guarded by is_compiling()
            # during tracing), so all kernels in the graph are inductor-generated and
            # safely capturable.
            model = compiled
        except Exception as e:
            print(f"[GFPGANTorch] torch.compile failed ({e!s:.120}), falling back to CUDA graph.")

    with torch.no_grad():
        for _ in range(3):
            model(static_inp)

    capture_stream = torch.cuda.Stream(device=dev)
    torch.cuda.synchronize(dev)
    graph = torch.cuda.CUDAGraph()
    with (
        torch.no_grad(),
        torch.cuda.graph(graph, stream=capture_stream, capture_error_mode="relaxed"),
    ):
        static_out = model(static_inp)
    torch.cuda.synchronize(dev)

    def runner(x: torch.Tensor) -> torch.Tensor:
        static_inp.copy_(x)
        graph.replay()
        return static_out.clone()

    return runner
