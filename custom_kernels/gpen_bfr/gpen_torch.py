"""
GPENTorch — FP16 PyTorch implementation of GPEN-BFR face restoration.

Supports all four GPEN variants:
    GPEN-BFR-256   (256×256  → 256×256)
    GPEN-BFR-512   (512×512  → 512×512)
    GPEN-BFR-1024  (512×512  → 1024×1024)
    GPEN-BFR-2048  (512×512  → 2048×2048)

Architecture overview
─────────────────────
• Encoder: ecd0 (1×1 conv, no stride) + N strided 3×3 downsampling convs
  (ecd1…ecdN) that reduce spatial resolution to 4×4.
• Bottleneck linear: flatten → FC → [1,512]
• Style MLP: 9 × (Linear 512→512 + LeakyReLU)
  The MLP output is a single style vector tiled 14 (or more) times.
• StyleGAN2 generator (const 4×4 → output resolution):
    - conv1:   modulated 3×3 + noise-inject + activate → [1,2*C,4,4]
    - N pairs of (styled_conv_up, styled_conv) + to_rgb at each scale
  "Noise injection" = replace random noise with encoder features:
      activate_input = cat([conv_out, noise_weight * enc_feat], dim=1)
  This doubles channels; subsequent convs take the doubled-channel input.

Reuses the fused demod CUDA kernel from custom_kernels/gfpgan_v1_4/_demod_build/.

Usage:
    model = GPENTorch.from_onnx("model_assets/GPEN-BFR-512.onnx").cuda().eval()
    output = model(input_f32_cuda)   # [1,3,H,W] float32 in/out
"""

from __future__ import annotations

import importlib.util
import math
import os
import sys
from pathlib import Path
from typing import List, Optional

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
        triton_fused_gpen_act as _triton_gpen_act,
    )
except Exception:
    _TRITON_AVAILABLE = False
    _triton_demod = None  # type: ignore[assignment]
    _triton_gpen_act = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Reuse the fused demod kernel from gfpgan_v1_4
# ---------------------------------------------------------------------------

_GFPGAN_DIR = Path(__file__).parent.parent / "gfpgan_v1_4"
_LEGACY_BUILD_DIR = _GFPGAN_DIR / "_demod_build"
_EXT_NAME = "gfpgan_demod_ext"
# Primary location: multi-arch fat binary in model_assets/custom_kernels/
_SHARED_DIR = Path(__file__).parent.parent.parent / "model_assets" / "custom_kernels"
_SHARED_DIR.mkdir(parents=True, exist_ok=True)

_ext_obj = False  # False = not tried, None = failed, module = loaded


def _add_dll_dirs():
    if sys.platform == "win32":
        try:
            import torch as _t

            tlib = Path(_t.__file__).parent / "lib"
            if tlib.is_dir():
                os.add_dll_directory(str(tlib))
        except Exception:
            pass


def _try_load_pyd(pyd_path: Path, tag: str) -> bool:
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
        print(f"[GPENTorch] Demod extension loaded ({tag}).")
        return True
    except Exception as e:
        print(f"[GPENTorch] Load failed ({tag}): {e}")
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

    # 3. JIT compile for current GPU only (saves to shared dir)
    _CUDA_SRC = r"""
#include <cuda_runtime.h>
#include <math.h>
__device__ __forceinline__ float warp_reduce(float v) {
    for (int d=16;d>0;d>>=1) v+=__shfl_down_sync(0xffffffff,v,d);
    return v;
}
__global__ void fused_demod_kernel(const float* __restrict__ w,
    const float* __restrict__ s, float* __restrict__ o,
    int C_out,int n,int kHkW,float eps){
    int co=blockIdx.x; if(co>=C_out) return;
    const float* wp=w+(long long)co*n; float* op=o+(long long)co*n;
    float sq=0.f;
    for(int i=threadIdx.x;i<n;i+=blockDim.x){float ws=wp[i]*s[i/kHkW];sq+=ws*ws;}
    sq=warp_reduce(sq);
    __shared__ float sm[8]; int lane=threadIdx.x&31,warp=threadIdx.x>>5;
    if(lane==0)sm[warp]=sq; __syncthreads();
    if(warp==0){sq=(lane<blockDim.x/32)?sm[lane]:0.f;sq=warp_reduce(sq);}
    __shared__ float dm; if(threadIdx.x==0)dm=rsqrtf(sq+eps); __syncthreads();
    float d=dm;
    for(int i=threadIdx.x;i<n;i+=blockDim.x) op[i]=wp[i]*s[i/kHkW]*d;
}
"""
    _CPP_SRC = r"""
#include <torch/extension.h>
void fused_demod_kernel(const float*,const float*,float*,int,int,int,float);
torch::Tensor fused_demod(torch::Tensor w,torch::Tensor s,float eps){
    TORCH_CHECK(w.is_cuda()&&s.is_cuda());
    TORCH_CHECK(w.dim()==4&&w.scalar_type()==torch::kFloat32);
    int Co=w.size(0),Ci=w.size(1),kHkW=w.size(2)*w.size(3);
    auto out=torch::empty_like(w);
    fused_demod_kernel<<<Co,256>>>(w.data_ptr<float>(),s.data_ptr<float>(),
        out.data_ptr<float>(),Co,Ci*kHkW,kHkW,eps);
    return out;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("fused_demod",&fused_demod,"Fused demod");}
"""
    import sys as _sys
    import shutil as _shutil

    if (
        _sys.platform == "win32"
        and _shutil.which("cl") is None
        and _shutil.which("cl.exe") is None
    ):
        print(
            "[GPENTorch] Demod CUDA extension not found and CL.exe is unavailable.\n"
            "  Run: python custom_kernels/build_kernels.py  (requires Visual Studio Build Tools)\n"
            "  -> Using pure-PyTorch demod fallback."
        )
        _ext_obj = None
        return
    print("[GPENTorch] Compiling demod extension for current GPU (requires MSVC)...")
    try:
        from torch.utils.cpp_extension import load_inline

        _ext_obj = load_inline(
            name=_EXT_NAME,
            cuda_sources=[_CUDA_SRC],
            cpp_sources=[_CPP_SRC],
            build_directory=str(_SHARED_DIR),
            extra_cuda_cflags=["--use_fast_math", "-O3"],
            verbose=False,
        )
        print("[GPENTorch] Demod extension compiled and ready.")
    except Exception as e:
        print(f"[GPENTorch] Demod compile failed: {e}  -> using pure-PyTorch fallback.")
        _ext_obj = None


def _get_demod_fn():
    global _ext_obj
    if _ext_obj is False:
        _load_ext()
    return None if _ext_obj is None else _ext_obj.fused_demod


def _fused_demod(
    w: torch.Tensor, style: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    # Priority 1: Triton (Windows-friendly, no MSVC needed)
    if _TRITON_AVAILABLE and _triton_demod is not None:
        return _triton_demod(
            w.to(torch.float16).contiguous(), style.contiguous().float(), eps
        )
    # Priority 2: CUDA C++ extension (if pre-built or JIT-compiled)
    fn = _get_demod_fn()
    if fn is not None:
        return fn(w.contiguous().float(), style.contiguous().float(), eps)
    # Priority 3: Pure PyTorch fallback
    wm = w.float() * style.view(1, -1, 1, 1)
    return (wm * torch.rsqrt(wm.pow(2).sum([1, 2, 3], keepdim=True) + eps)).to(w.dtype)


# ---------------------------------------------------------------------------
# ONNX weight extraction helpers
# ---------------------------------------------------------------------------


def _load_onnx_weights(path: str) -> dict:
    """
    Load all ONNX initializers by name, then attach additional positional
    aliases for encoder convs, linear layers, style MLP, and generator
    convs/to_rgbs — so the model loader can address them by role rather
    than auto-generated numeric names.
    """
    import onnx
    from onnx import numpy_helper

    m = onnx.load(path)
    w = {i.name: numpy_helper.to_array(i).copy() for i in m.graph.initializer}
    init_shapes = {k: v.shape for k, v in w.items()}

    # ── Detect model I/O sizes ──────────────────────────────────────────
    # Input shape from graph input[0]
    in_shape = None
    for inp in m.graph.input:
        if inp.type.HasField("tensor_type"):
            dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
            if len(dims) == 4:
                in_shape = dims  # [1, 3, H, W]
                break
    out_shape = None
    for outp in m.graph.output:
        if outp.type.HasField("tensor_type"):
            dims = [d.dim_value for d in outp.type.tensor_type.shape.dim]
            if len(dims) == 4:
                out_shape = dims
                break
    w["_meta_in_hw"] = np.array(in_shape[2:4] if in_shape else [0, 0])
    w["_meta_out_hw"] = np.array(out_shape[2:4] if out_shape else [0, 0])

    # ── Collect Conv nodes (ignore 4×4 FIR blur convs that have no bias) ──
    # Encoder strided convs have shape [C_out, C_in, 3, 3] and bias [C_out]
    enc_conv_weights: list = []  # (weight_key, bias_key)
    enc_conv0_weight: Optional[str] = None  # 1×1 first layer
    enc_conv0_bias: Optional[str] = None

    for node in m.graph.node:
        if node.op_type != "Conv":
            continue
        if len(node.input) < 3:
            continue  # FIR blur conv has only weight (no bias input)
        wk = node.input[1]
        bk = node.input[2]
        if wk not in init_shapes or bk not in init_shapes:
            continue
        ws = init_shapes[wk]
        bs = init_shapes[bk]
        if len(ws) != 4 or len(bs) != 1:
            continue
        C_out, C_in, kH, kW = ws
        if C_in == 1:
            continue  # skip depthwise FIR kernels (they'd be [C,1,4,4])
        if kH == 1 and kW == 1 and C_in == 3:
            # ecd0: 1×1 conv, 3 input channels
            enc_conv0_weight = wk
            enc_conv0_bias = bk
        elif kH == 3 and kW == 3 and node.input[0].startswith("/ecd"):
            enc_conv_weights.append((wk, bk))

    if enc_conv0_weight:
        w["_enc0_w"] = w[enc_conv0_weight]
        w["_enc0_b"] = w[enc_conv0_bias]
    for i, (wk, bk) in enumerate(enc_conv_weights):
        w[f"_enc{i + 1}_w"] = w[wk]
        w[f"_enc{i + 1}_b"] = w[bk]
    w["_num_enc_strided"] = np.array(len(enc_conv_weights))

    # ── Bottleneck FC ────────────────────────────────────────────────────
    # The final_linear node: first Gemm/MatMul with output containing
    # "final_linear" whose weight has shape [large_in, 512].
    for node in m.graph.node:
        if node.op_type not in ("Gemm", "MatMul"):
            continue
        out_name = node.output[0] if node.output else ""
        if "final_linear" not in out_name and "final_linear" not in "".join(node.input):
            continue
        wk = node.input[1] if len(node.input) > 1 else ""
        if wk not in init_shapes:
            continue
        sh = init_shapes[wk]
        if len(sh) == 2 and sh[1] == 512:
            w["_fc_w"] = w[wk]
            if len(node.input) > 2 and node.input[2] in init_shapes:
                w["_fc_b"] = w[node.input[2]]
            break

    # ── Style MLP ────────────────────────────────────────────────────────
    # Detect by output name containing "/style/" path (GPEN style sub-net).
    mlp_count = 0
    for node in m.graph.node:
        if node.op_type not in ("Gemm", "MatMul"):
            continue
        out_name = node.output[0] if node.output else ""
        if "/style/" not in out_name:
            continue
        wk = node.input[1] if len(node.input) > 1 else ""
        if wk not in init_shapes:
            continue
        sh = init_shapes[wk]
        if len(sh) == 2:
            w[f"_mlp{mlp_count}_w"] = w[wk]
            if len(node.input) > 2 and node.input[2] in init_shapes:
                w[f"_mlp{mlp_count}_b"] = w[node.input[2]]
            mlp_count += 1
    w["_num_mlp"] = np.array(mlp_count)

    # ── Generator constant ──────────────────────────────────────────────
    for k, v in w.items():
        if "Tile_output_0" in k and v.ndim == 4:
            w["_gen_const"] = v
            break

    # ── Generator conv weights (position order) ─────────────────────────
    # Identify Mul initializers with shape [1,C_out,C_in,kH,kW]
    gen_mul_keys: list = []  # (key, C_out, C_in, kH, kW)
    for node in m.graph.node:
        if node.op_type != "Mul":
            continue
        for inp in node.input:
            if inp in init_shapes and len(init_shapes[inp]) == 5:
                sh = init_shapes[inp]
                if sh[0] == 1:
                    gen_mul_keys.append((inp, sh[1], sh[2], sh[3], sh[4]))

    w["_gen_mul_keys"] = gen_mul_keys  # type: ignore[assignment]  # list kept for reference

    # ── Generator modulation Gemms ────────────────────────────────────────
    # Detect by first input being a Gather output ("/generator/Gather_*").
    # This uniquely identifies modulation Gemms (style MLP Gemms use /style/
    # paths and the FC uses Reshape inputs).
    gen_mod_pairs: list = []  # (mod_w_key, mod_b_key_or_None)
    for node in m.graph.node:
        if node.op_type not in ("Gemm", "MatMul"):
            continue
        if not node.input:
            continue
        first_input = node.input[0]
        if (
            "/generator/Gather" not in first_input
            and "Gather_output" not in first_input
        ):
            continue
        wk = node.input[1] if len(node.input) > 1 else ""
        if wk not in init_shapes:
            continue
        sh = init_shapes[wk]
        if len(sh) != 2 or sh[1] != 512:
            continue
        bk = None
        if len(node.input) > 2 and node.input[2] in init_shapes:
            bk = node.input[2]
        gen_mod_pairs.append((wk, bk))

    # conv1 uses a named bias 'generator.conv1.conv.modulation.bias'
    # conv1's modulation is the FIRST gen_mod_pairs entry
    # Store aliases
    for i, (mwk, mbk) in enumerate(gen_mod_pairs):
        w[f"_gmod{i}_w"] = w[mwk]
        if mbk:
            w[f"_gmod{i}_b"] = w[mbk]
    w["_num_gen_mods"] = np.array(len(gen_mod_pairs))

    # ── Activate biases: [1,C,1,1] initializers in generator section ──
    # These appear as Add node inputs in the generator after styled conv
    act_bias_keys: list = []
    in_generator = False
    for node in m.graph.node:
        if "/generator/" in "".join(node.input) or "/generator/" in "".join(
            node.output
        ):
            in_generator = True
        if not in_generator:
            continue
        if node.op_type != "Add":
            continue
        for inp in node.input:
            if inp in init_shapes:
                sh = init_shapes[inp]
                if len(sh) == 4 and sh[0] == 1 and sh[2] == 1 and sh[3] == 1:
                    act_bias_keys.append(inp)
    for i, k in enumerate(act_bias_keys):
        w[f"_gact{i}"] = w[k]
    w["_num_gact"] = np.array(len(act_bias_keys))

    # ── to_rgb Conv biases: [3] initializers in generator section ────
    torgb_bias_keys: list = []
    in_generator = False
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
    for i, k in enumerate(torgb_bias_keys):
        w[f"_trgb_b{i}"] = w[k]
    w["_num_trgb"] = np.array(len(torgb_bias_keys))

    # ── Noise weights ────────────────────────────────────────────────────
    # Use direct name lookup to guarantee correct numeric ordering:
    #   _noise0 = conv1,  _noise1 = convs.0, ..., _noise{N} = convs.{N-1}
    _noise_idx = 0
    if "generator.conv1.noise.weight" in w:
        w[f"_noise{_noise_idx}"] = w["generator.conv1.noise.weight"]
        _noise_idx += 1
    for _ci in range(200):  # up to 200 generator convs
        nk = f"generator.convs.{_ci}.noise.weight"
        if nk not in w:
            break
        w[f"_noise{_noise_idx}"] = w[nk]
        _noise_idx += 1

    return w


# ---------------------------------------------------------------------------
# GPENTorch
# ---------------------------------------------------------------------------


class GPENTorch(torch.nn.Module):
    """
    FP16 PyTorch reimplementation of GPEN-BFR.
    Supports 256/512/1024/2048 output sizes.
    """

    def __init__(
        self,
        in_size: int,
        out_size: int,
        n_enc: int,
        n_loop: int,
        compute_dtype: torch.dtype = torch.float16,
    ):
        """
        in_size:       spatial input H (= W), e.g. 256 or 512
        out_size:      spatial output H (= W), e.g. 256, 512, 1024, 2048
        n_enc:         number of strided encoder stages (ecd1..ecdN)
        n_loop:        number of generator up-sampling pairs (= log2(out_size)-2)
        compute_dtype: torch.float16 (fast) or torch.float32 (reference)
        """
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.n_enc = n_enc
        self.n_loop = n_loop
        self.compute_dtype = compute_dtype

    @classmethod
    def from_onnx(
        cls, onnx_path: str, compute_dtype: torch.dtype = torch.float16
    ) -> "GPENTorch":
        w = _load_onnx_weights(onnx_path)

        in_hw = int(w["_meta_in_hw"][0])
        out_hw = int(w["_meta_out_hw"][0])
        n_enc = int(w["_num_enc_strided"])
        n_loop = int(math.log2(out_hw)) - 2  # 4×4 → out_hw

        model = cls(in_hw, out_hw, n_enc, n_loop, compute_dtype)
        model._load_weights(w)
        model._visomaster_onnx_path = str(onnx_path)
        return model

    # ------------------------------------------------------------------
    # Weight registration
    # ------------------------------------------------------------------

    def _load_weights(self, w: dict):
        rb = self.register_buffer
        cdtype = self.compute_dtype

        def h(k):
            return torch.from_numpy(w[k]).to(cdtype).contiguous()

        def f32(k):
            return torch.from_numpy(w[k]).to(torch.float32).contiguous()

        def h_opt(k, shape):
            if k in w:
                return torch.from_numpy(w[k]).to(cdtype).contiguous()
            return torch.zeros(shape, dtype=cdtype)

        # ── Encoder ───────────────────────────────────────────────────────
        rb("enc0_w", h("_enc0_w"))
        rb("enc0_b", h("_enc0_b"))
        for i in range(1, self.n_enc + 1):
            rb(f"enc{i}_w", h(f"_enc{i}_w"))
            rb(f"enc{i}_b", h(f"_enc{i}_b"))

        # ── Bottleneck linear (FP32 for accuracy) ────────────────────────
        # ONNX Gemm stores weight as [in, out] (A @ B format);
        # F.linear needs [out, in], so transpose.
        rb("fc_w", f32("_fc_w").T.contiguous())  # MatMul A@B: B[in,out] → T → [out,in]
        rb("fc_b", f32("_fc_b").squeeze())

        # ── Style MLP (FP32) ─────────────────────────────────────────────
        # ONNX Gemm with transB=0: C = A @ B (NOT A @ B^T).
        # F.linear computes x @ W.T, so we must store W = B.T.
        n_mlp = int(
            w["_num_mlp"].item() if hasattr(w["_num_mlp"], "item") else w["_num_mlp"]
        )
        for i in range(n_mlp):
            rb(f"mlp{i}_w", f32(f"_mlp{i}_w").T.contiguous())  # transB=0 → need .T
            rb(f"mlp{i}_b", f32(f"_mlp{i}_b").squeeze())
        self._n_mlp = n_mlp

        # ── Generator constant ───────────────────────────────────────────
        rb("gen_const", h("_gen_const"))

        # ── Generator modulation weights (FP32 for accuracy) ─────────────
        # ONNX Gemm with transB=1: weight [C_mod, 512] already in [out, in]
        # format for F.linear(style[1,512], W[C_mod,512]) → [1, C_mod].
        n_mods = int(
            w["_num_gen_mods"].item()
            if hasattr(w["_num_gen_mods"], "item")
            else w["_num_gen_mods"]
        )
        conv1_mod_b_key = "generator.conv1.conv.modulation.bias"
        rb("gmod0_w", f32("_gmod0_w"))
        if conv1_mod_b_key in w:
            rb("gmod0_b", f32(conv1_mod_b_key))
        else:
            rb("gmod0_b", f32("_gmod0_b").squeeze())
        for i in range(1, n_mods):
            rb(f"gmod{i}_w", f32(f"_gmod{i}_w"))
            rb(f"gmod{i}_b", f32(f"_gmod{i}_b").squeeze())
        self._n_mods = n_mods

        # ── Generator conv kernels ────────────────────────────────────────
        gen_mul_keys = w["_gen_mul_keys"]  # list of (key, C_out, C_in, kH, kW)
        for i, (k, C_out, C_in, kH, kW) in enumerate(gen_mul_keys):
            rb(f"gconv{i}_w", h(k))
        self._n_gconv = len(gen_mul_keys)
        # Record shapes for index lookup
        self._gconv_shapes = [
            (C_out, C_in, kH, kW) for (_, C_out, C_in, kH, kW) in gen_mul_keys
        ]

        # ── Activate biases ───────────────────────────────────────────────
        n_gact = int(
            w["_num_gact"].item() if hasattr(w["_num_gact"], "item") else w["_num_gact"]
        )
        for i in range(n_gact):
            rb(f"gact{i}", h(f"_gact{i}"))
        self._n_gact = n_gact

        # ── to_rgb biases [3] ─────────────────────────────────────────────
        n_trgb = int(
            w["_num_trgb"].item() if hasattr(w["_num_trgb"], "item") else w["_num_trgb"]
        )
        for i in range(n_trgb):
            arr = w[f"_trgb_b{i}"]
            rb(
                f"trgb_b{i}",
                torch.from_numpy(arr).to(cdtype).view(1, 3, 1, 1).contiguous(),
            )
        self._n_trgb = n_trgb

        # ── Noise (encoder-feature injection) weights ─────────────────────
        # 1 per styled conv (conv1 + all convs.*)
        n_styled = 1 + 2 * self.n_loop
        for i in range(n_styled):
            nk = f"_noise{i}"
            val = torch.from_numpy(w[nk]).to(cdtype).contiguous() if nk in w else None
            rb(f"noise{i}", val)

        # ── FIR kernels ───────────────────────────────────────────────────
        # Encoder FIR: [1,3,3,1]/8 outer product /8 = bilinear blur for downsample
        _enc_fir = torch.tensor(
            [
                0.015625,
                0.046875,
                0.046875,
                0.015625,
                0.046875,
                0.140625,
                0.140625,
                0.046875,
                0.046875,
                0.140625,
                0.140625,
                0.046875,
                0.015625,
                0.046875,
                0.046875,
                0.015625,
            ],
            dtype=cdtype,
        ).view(1, 1, 4, 4)
        rb("enc_fir", _enc_fir)
        # Generator FIR: [1,3,3,1]/4 outer product /4 = bilinear blur for upsample
        _gen_fir = torch.tensor(
            [
                0.0625,
                0.1875,
                0.1875,
                0.0625,
                0.1875,
                0.5625,
                0.5625,
                0.1875,
                0.1875,
                0.5625,
                0.5625,
                0.1875,
                0.0625,
                0.1875,
                0.1875,
                0.0625,
            ],
            dtype=cdtype,
        ).view(1, 1, 4, 4)
        rb("gen_fir", _gen_fir)

        # ── Build index maps (fixed once we know shapes) ──────────────────
        #
        # Generator conv ordering in ONNX graph:
        #   gconv0:           conv1         (C_in = gen_const.C)
        #   gconv1..2*n_loop: convs.0..N   (paired: up+same per resolution)
        #
        # to_rgb ordering:
        #   to_rgb0:          to_rgb1       (4×4)
        #   to_rgb1..n_loop:  to_rgbs.0..M
        #
        # Modulation (gmod) ordering:
        #   gmod0:  conv1
        #   gmod1:  to_rgb1  (shared latent with convs.0)
        #   gmod2:  convs.0
        #   gmod3:  convs.1
        #   gmod4:  to_rgbs.0  (shared latent with convs.2)
        #   gmod5:  convs.2
        #   ... pattern: (to_rgbs[i], convs[2i], convs[2i+1]) repeat
        #
        # Latent index (single style vector tiled):
        #   latent[0]:          conv1
        #   latent[1]:          to_rgb1 AND convs.0  (shared)
        #   latent[2]:          convs.1
        #   latent[3]:          to_rgbs.0 AND convs.2  (shared)
        #   latent[4]:          convs.3
        #   latent[2k-1]:       to_rgbs.k-1 AND convs.2(k-1)  (shared)
        #   latent[2k]:         convs.2k-1
        #   ...
        # Total latents = 1 + 1 + 2*(n_loop) = 2 + 2*n_loop
        #   (conv1=1, then n_loop pairs of (shared_torgb+convup, conv))
        self._n_latents = 2 + 2 * self.n_loop

    # ------------------------------------------------------------------
    # FIR helpers
    # ------------------------------------------------------------------

    def _enc_fir_down(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encoder FIR anti-aliasing: zero-pad 2 on each side, then depthwise 4×4 conv.
        Input  [1, C, H, W]  →  Output [1, C, H+1, W+1]
        After this, apply strided 3×3 enc conv with stride=2, padding=0.
        """
        C = x.shape[1]
        fir = self._buffers["enc_fir"].expand(C, 1, 4, 4)
        x = F.pad(x, (2, 2, 2, 2))  # [1, C, H+4, W+4]
        return F.conv2d(x, fir, groups=C)  # [1, C, H+1, W+1]

    def _gen_fir_blur(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generator FIR anti-aliasing after ConvTranspose: pad 1 on each side,
        then depthwise 4×4 conv.
        Input  [1, C, H, W]  →  Output [1, C, H-2, W-2]  (because 4×4 no-pad eats 3 pixels)
        With pad 1: net effect H → H-2, since pad=+2, fir=-3 → -1... let me verify:
        H+2 - 4 + 1 = H - 1. So output is H-1.
        For conv_transpose output of 9 → pad→11 → fir→8:  9+2-4+1=8. ✓
        """
        C = x.shape[1]
        fir = self._buffers["gen_fir"].expand(C, 1, 4, 4)
        x = F.pad(x, (1, 1, 1, 1))  # [1, C, H+2, W+2]
        return F.conv2d(x, fir, groups=C)  # [1, C, H-1, W-1]

    # ------------------------------------------------------------------
    # Primitive helpers
    # ------------------------------------------------------------------

    def _styled_conv(
        self,
        x: torch.Tensor,  # [1, C_in,  H,  W]  FP16
        gconv_idx: int,  # index into gconv* buffers
        gmod_idx: int,  # index into gmod* buffers
        gact_idx: int,  # index into gact* buffers
        latent: torch.Tensor,  # [1, 512] FP32
        enc_feat: torch.Tensor,  # [1, C_out, H', W'] FP16
        noise_w: Optional[torch.Tensor],  # scalar [1] FP16 or None
        upsample: bool = False,
    ) -> torch.Tensor:
        """
        Modulated conv (with demod) + noise-inject via Concat + activate.
        Returns [1, 2*C_out, H', W'].
        """
        g = self._buffers

        weight = g[f"gconv{gconv_idx}_w"]  # [1, C_out, C_in, kH, kW] FP16
        mod_w = g[f"gmod{gmod_idx}_w"]  # [C_in, 512] FP32
        mod_b = g[f"gmod{gmod_idx}_b"]  # [C_in] FP32
        act_b = g[f"gact{gact_idx}"]  # [1, 2*C_out, 1, 1] FP16

        _C_out = weight.shape[1]
        _C_in = weight.shape[2]
        kH = weight.shape[3]

        # Modulation: style [1, C_in]
        style = F.linear(latent, mod_w, mod_b)  # [1, C_in] FP32
        w_f32 = weight[0].float()  # [C_out, C_in, kH, kW]
        w_dem = _fused_demod(w_f32, style[0]).to(self.compute_dtype)

        if upsample:
            # GPEN uses ConvTranspose (stride=2, no-pad) then FIR anti-alias blur.
            # Transpose kernel [C_out, C_in, kH, kW] → [C_in, C_out, kH, kW].
            w_up = w_dem.permute(1, 0, 2, 3).contiguous()  # [C_in, C_out, kH, kW]
            out = F.conv_transpose2d(x, w_up, stride=2, padding=0)
            out = self._gen_fir_blur(out)  # [1, C_out, 2H, 2H]
        else:
            out = F.conv2d(x, w_dem, padding=kH // 2)  # [1, C_out, H', W']

        # Noise injection: cat([conv_out, noise_weight * enc_feat], dim=1)
        if noise_w is not None and enc_feat is not None:
            noise_term = noise_w * enc_feat  # Triton kernel handles broadcasting
        elif enc_feat is not None:
            noise_term = enc_feat
        else:
            noise_term = torch.zeros_like(out)

        if (
            _TRITON_AVAILABLE
            and _triton_gpen_act is not None
            and out.dtype == torch.float16
        ):
            # Triton kernel now handles broadcasting of noise_term and act_b internally
            return _triton_gpen_act(out, noise_term, act_b, 0.2, 2.0**0.5)

        doubled = torch.cat([out, noise_term], dim=1)  # [1, 2*C_out, H', W']
        doubled = doubled + act_b
        doubled = F.leaky_relu(doubled, 0.2)
        doubled = doubled * (2.0**0.5)
        return doubled

    def _torgb_conv(
        self,
        x: torch.Tensor,  # [1, C_in, H, W] FP16  (2*C_out from styled conv)
        gconv_idx: int,
        gmod_idx: int,
        trgb_idx: int,
        latent: torch.Tensor,  # [1, 512] FP32
        skip: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        to-RGB 1×1 modulated conv (no demod) + bias + optional skip upsample.
        """
        g = self._buffers

        weight = g[f"gconv{gconv_idx}_w"]  # [1, 3, C_in, 1, 1] FP16
        mod_w = g[f"gmod{gmod_idx}_w"]  # [C_in, 512] FP32
        mod_b = g[f"gmod{gmod_idx}_b"]  # [C_in] FP32
        bias = g[f"trgb_b{trgb_idx}"]  # [1, 3, 1, 1] FP16

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
        x:       [1, 3, H, W] float32 CUDA
        returns: [1, 3, H, W] float32 CUDA  (H = W = out_size)
        """
        x = x.to(self.compute_dtype)
        g = self._buffers

        _SQRT2 = 2.0**0.5

        # ── Encoder ───────────────────────────────────────────────────────
        # ecd0: 1×1 conv (no downsample): Conv → LeakyRelu → Mul(sqrt(2))
        feat = F.leaky_relu(F.conv2d(x, g["enc0_w"], g["enc0_b"]), 0.2) * _SQRT2
        enc_feats: List[torch.Tensor] = [feat]  # enc_feats[0] = ecd0 output

        # ecd1..N: FIR blur (pad 2 + depthwise 4×4) → 3×3 strided conv no-pad
        # ONNX: FIR(x) → [1,C,H+1,W+1] → Conv(stride=2,pad=0) → [1,C_out,H/2,W/2]
        for i in range(1, self.n_enc + 1):
            feat_fir = self._enc_fir_down(feat)  # [1, C, H+1, W+1]
            feat = (
                F.leaky_relu(
                    F.conv2d(
                        feat_fir, g[f"enc{i}_w"], g[f"enc{i}_b"], stride=2, padding=0
                    ),
                    0.2,
                )
                * _SQRT2
            )
            enc_feats.append(feat)
        # enc_feats ordering: [ecd0, ecd1, ..., ecd_N]  (deepest = last)

        # ── Bottleneck linear ─────────────────────────────────────────────
        # ONNX: Gemm → LeakyRelu → Mul(sqrt(2))  (NOT just a linear layer)
        feat_flat = feat.float().view(1, -1)  # flatten 4×4 deepest feat
        z = F.leaky_relu(F.linear(feat_flat, g["fc_w"], g["fc_b"]), 0.2) * _SQRT2

        # ── Style MLP ─────────────────────────────────────────────────────
        # Input: pixel normalization of FC output (normalize to unit RMS)
        # Each layer: Gemm → LeakyRelu → Mul(sqrt(2))
        rms = torch.sqrt(z.pow(2).mean(dim=1, keepdim=True) + 1e-9)
        style = z / rms
        for i in range(self._n_mlp):
            style = (
                F.leaky_relu(F.linear(style, g[f"mlp{i}_w"], g[f"mlp{i}_b"]), 0.2)
                * _SQRT2
            )
        # style: [1, 512]

        # Single style vector, tiled to [1, n_latents, 512]
        latent = style.unsqueeze(1).expand(1, self._n_latents, 512)  # all same

        # ── StyleGAN2 Generator ───────────────────────────────────────────
        # Encoder features are injected deepest-first:
        #   conv1 (4×4)         ← enc_feats[n_enc]   (deepest)
        #   convs.0/1 (8×8)     ← enc_feats[n_enc-1]
        #   convs.2/3 (16×16)   ← enc_feats[n_enc-2]
        #   ...
        #   convs[-2]/[-1]      ← enc_feats[0] or enc_feats[1]

        sg = g["gen_const"].expand(1, -1, -1, -1).clone()  # [1, C, 4, 4]

        # ONNX graph interleaved ordering:
        #   gconv/gmod 0:           conv1         (styled)
        #   gconv/gmod 1:           to_rgb1       (torgb)
        #   gconv/gmod 2+li*3:      convs[2*li]   (styled, upsample)
        #   gconv/gmod 3+li*3:      convs[2*li+1] (styled, same-res)
        #   gconv/gmod 4+li*3:      to_rgbs[li]   (torgb)
        #
        # gact layout (styled only, no torgb):
        #   gact 0: conv1
        #   gact 1+li*2:  convs[2*li]
        #   gact 2+li*2:  convs[2*li+1]
        #
        # Latent indices (single tiled style vector):
        #   latent[0]:      conv1
        #   latent[1]:      to_rgb1 AND convs[0]   (shared)
        #   latent[2*li+2]: convs[2*li+1]
        #   latent[2*li+3]: to_rgbs[li] AND convs[2*(li+1)]  (shared)
        #
        # Noise ordering: _noise0=conv1, _noise1=convs.0, _noise2=convs.1, ...

        # conv1 (4×4)
        enc_feat = enc_feats[self.n_enc].to(self.compute_dtype)  # deepest
        nw = g.get("noise0")
        sg = self._styled_conv(
            sg,
            gconv_idx=0,
            gmod_idx=0,
            gact_idx=0,
            latent=latent[:, 0],
            enc_feat=enc_feat,
            noise_w=nw,
            upsample=False,
        )
        # sg: [1, 2*C_const, 4, 4]

        # to_rgb1
        skip_rgb = self._torgb_conv(
            sg, gconv_idx=1, gmod_idx=1, trgb_idx=0, latent=latent[:, 1]
        )

        for li in range(self.n_loop):
            enc_depth = self.n_enc - 1 - li  # enc feature for this resolution
            enc_feat = enc_feats[enc_depth].to(self.compute_dtype)

            ci_up = 2 + li * 3  # gconv/gmod for convs[2*li]   (upsample)
            ci_same = 3 + li * 3  # gconv/gmod for convs[2*li+1] (same-res)
            ci_trgb = 4 + li * 3  # gconv/gmod for to_rgbs[li]

            lat_up = 2 * li + 1  # latent[1,3,5,7,9,11]
            lat_same = 2 * li + 2  # latent[2,4,6,8,10,12]
            lat_trgb = 2 * li + 3  # latent[3,5,7,9,11,13]

            noise_up = 1 + li * 2  # _noise1,3,5,7,9,11
            noise_same = 2 + li * 2  # _noise2,4,6,8,10,12

            gact_up = 1 + li * 2
            gact_same = 2 + li * 2

            nw = g.get(f"noise{noise_up}")
            sg = self._styled_conv(
                sg,
                gconv_idx=ci_up,
                gmod_idx=ci_up,
                gact_idx=gact_up,
                latent=latent[:, lat_up],
                enc_feat=enc_feat,
                noise_w=nw,
                upsample=True,
            )

            nw = g.get(f"noise{noise_same}")
            sg = self._styled_conv(
                sg,
                gconv_idx=ci_same,
                gmod_idx=ci_same,
                gact_idx=gact_same,
                latent=latent[:, lat_same],
                enc_feat=enc_feat,
                noise_w=nw,
                upsample=False,
            )

            skip_rgb = self._torgb_conv(
                sg,
                gconv_idx=ci_trgb,
                gmod_idx=ci_trgb,
                trgb_idx=1 + li,
                latent=latent[:, lat_trgb],
                skip=skip_rgb,
            )

        return skip_rgb.float()


# ---------------------------------------------------------------------------
# CUDA Graph runner  (same API as GFPGAN)
# ---------------------------------------------------------------------------


def build_cuda_graph_runner(
    model: GPENTorch,
    inp_shape: tuple,
    torch_compile: bool = False,
):
    """
    Capture model() as a CUDAGraph for repeated fixed-shape inference.
    Returns a callable: (x: float32 CUDA) → float32 CUDA.

    Args:
        model:         GPENTorch instance (.cuda().eval())
        inp_shape:     Input tensor shape, e.g. (1, 3, 512, 512)
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
            print("[GPENTorch] torch.compile warmup done.")
            # Return compiled model directly — CUDA graph on top of torch.compile
            # fails on Windows (64-bit kernel handles overflow 32-bit C long).
            return compiled
        except Exception as e:
            print(f"[GPENTorch] torch.compile failed ({e!s:.120}), falling back to CUDA graph.")

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
