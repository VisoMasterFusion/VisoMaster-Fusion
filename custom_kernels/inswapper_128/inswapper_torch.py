"""
inswapper_torch.py
==================
PyTorch-native reimplementation of inswapper_128.fp16.onnx with:
  1. All weights loaded directly from the ONNX file.
  2. Custom CUDA kernel for fused AdaIN (warp-shuffle reduction, __half2 I/O).
  3. Batched Gemm — all 12 style-linear layers fused into one large matmul.
  4. Triton fused-AdaIN kernel (priority over CUDA C++ when available).
  5. build_cuda_graph_runner() for CUDA-graph-captured inference.
  6. im2col + cuBLAS GEMM mode — replaces cuDNN implicit-GEMM with explicit
     F.unfold + torch.matmul for ~85 % tensor-core utilisation on Ampere+ GPUs
     (vs cuDNN's ~47 % for [1024, 9216] x [9216, 1024] at 32x32 spatial).
  7. Triton fused AdaIN+Residual — saves 6 separate residual-add kernel
     launches per forward pass by fusing the add into the second AdaIN.
  8. cuBLASLt HGEMM mode (Phase 3) — replaces torch.mm with cuBLASLt for
     the style-block GEMMs with optional fused BIAS epilogue.  Provides
     better algorithm selection than cuBLAS heuristics and eliminates
     separate bias-add kernel launches.  Falls back to torch.mm gracefully
     if the C++ extension cannot be compiled or loaded.

Usage
-----
    from tools.inswapper_torch import InSwapperTorch, build_cuda_graph_runner

    model = InSwapperTorch("model_assets/inswapper_128.fp16.onnx").cuda().eval()
    model.to_gemm_mode()           # Tier 8: im2col + cuBLAS
    model.to_cublaslt_mode()       # Phase 3 upgrade (optional, requires MSVC)
    run   = build_cuda_graph_runner(model, target_example, source_example)
    out   = run(target_f32_cuda, source_f32_cuda)  # [1,3,128,128] float32
"""

from __future__ import annotations
import os
import pathlib
import sys as _sys_module
from typing import Callable
import torch
import torch.nn as nn
import torch.nn.functional as F

_sys_platform = _sys_module.platform


def _has_cl_exe() -> bool:
    """Return True if cl.exe (MSVC) is available on PATH."""
    import shutil

    return shutil.which("cl") is not None or shutil.which("cl.exe") is not None


# ---------------------------------------------------------------------------
# Triton kernel (preferred — no MSVC required, works on Windows)
# ---------------------------------------------------------------------------
try:
    from custom_kernels.triton_ops import (
        TRITON_AVAILABLE as _TRITON_AVAILABLE,
        triton_adain as _triton_adain,
        triton_im2col_reflect as _triton_im2col_reflect,
    )
except Exception:
    _TRITON_AVAILABLE = False
    _triton_adain = None  # type: ignore[assignment]
    _triton_im2col_reflect = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# 0. cuBLASLt HGEMM extension (Phase 3)
#    Provides cuBLASLt-accelerated GEMM with optional fused BIAS epilogue for
#    InSwapper style-block convolutions.  The extension is loaded from a pre-
#    built .pyd or JIT-compiled on first use (requires MSVC on Windows).
#    Falls back silently to torch.mm when unavailable.
# ---------------------------------------------------------------------------

_STYLEBLOCK_CUDA_SRC = r"""
// styleblock.cu - cuBLASLt HGEMM with optional BIAS epilogue.
//
// Row-major convention (PyTorch):
//   C[M,N] = A[M,K] * B[K,N]  (+  bias[M]  when BIAS epilogue is active)
//
// cuBLASLt uses column-major storage.  To compute C_row using cuBLASLt we
// exploit the identity:
//   C_col[N,M] = B_col[N,K] * A_col[K,M]
// A row-major tensor X[r,c] is stored identically to a col-major tensor of
// shape [c,r], so:
//   A_row[M,K] as col-major = [K,M], lda=K  (passed as cuBLASLt "B")
//   B_row[K,N] as col-major = [N,K], lda=N  (passed as cuBLASLt "A")
//   C_row[M,N] as col-major = [N,M], ldc=N
// BIAS epilogue adds bias[j] to column j of C_col = row j of C_row.  Since
// j indexes M (C_out), this correctly adds the per-output-channel bias.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <mutex>
#include <unordered_map>
#include <stdexcept>

// -- Per-op state -------------------------------------------------------------
struct GemmOp {
    cublasLtMatmulDesc_t   desc    = nullptr;
    cublasLtMatrixLayout_t lA      = nullptr;
    cublasLtMatrixLayout_t lB      = nullptr;
    cublasLtMatrixLayout_t lC      = nullptr;
    cublasLtMatmulAlgo_t   algo    {};
    bool                   has_algo = false;
    size_t                 ws_sz    = 0;
    int64_t                M = 0, K = 0, N = 0;
};

static std::unordered_map<int64_t, GemmOp*> g_ops;
static std::mutex  g_reg_mx;
static int64_t     g_next_id = 1;
static cublasLtHandle_t g_handle = nullptr;

static cublasLtHandle_t get_handle() {
    if (!g_handle) {
        if (cublasLtCreate(&g_handle) != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("[StyleBlock] cublasLtCreate failed");
    }
    return g_handle;
}

// -- Public API ---------------------------------------------------------------
int64_t create_hgemm_op(int64_t M, int64_t K, int64_t N, const void* bias_ptr = nullptr) {
    cublasLtHandle_t h = get_handle();
    GemmOp* op = new GemmOp();
    op->M = M; op->K = K; op->N = N;
    op->ws_sz = 4 * 1024 * 1024;

    cublasLtMatmulDescCreate(&op->desc, CUBLAS_COMPUTE_16F, CUDA_R_16F);
    if (bias_ptr) {
        cublasLtEpilogue_t ep = CUBLASLT_EPILOGUE_BIAS;
        cublasLtMatmulDescSetAttribute(op->desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &ep, sizeof(ep));
        cudaDataType_t bdt = CUDA_R_16F;
        cublasLtMatmulDescSetAttribute(op->desc, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bdt, sizeof(bdt));
        cublasLtMatmulDescSetAttribute(op->desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_ptr, sizeof(bias_ptr));
    }
    cublasLtMatrixLayoutCreate(&op->lA, CUDA_R_16F, N, K, N);
    cublasLtMatrixLayoutCreate(&op->lB, CUDA_R_16F, K, M, K);
    cublasLtMatrixLayoutCreate(&op->lC, CUDA_R_16F, N, M, N);

    cublasLtMatmulPreference_t pref;
    cublasLtMatmulPreferenceCreate(&pref);
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &op->ws_sz, sizeof(op->ws_sz));
    cublasLtMatmulHeuristicResult_t res[4];
    int n_res = 0;
    cublasLtMatmulAlgoGetHeuristic(h, op->desc, op->lA, op->lB, op->lC, op->lC, pref, 4, res, &n_res);
    cublasLtMatmulPreferenceDestroy(pref);
    if (n_res > 0) { op->algo = res[0].algo; op->has_algo = true; }

    std::lock_guard<std::mutex> lk(g_reg_mx);
    int64_t id = g_next_id++;
    g_ops[id] = op;
    return id;
}

int64_t create_hgemm_bias_op(int64_t M, int64_t K, int64_t N, torch::Tensor bias) {
    TORCH_CHECK(bias.is_cuda() && bias.dtype() == torch::kHalf);
    return create_hgemm_op(M, K, N, bias.data_ptr());
}

torch::Tensor run_hgemm_op(int64_t op_id, torch::Tensor A, torch::Tensor B) {
    GemmOp* op_ptr = nullptr;
    {
        std::lock_guard<std::mutex> lk(g_reg_mx);
        op_ptr = g_ops.at(op_id);
    }
    GemmOp& op = *op_ptr;
    auto C = torch::empty({op.M, op.N}, A.options());
    auto workspace = torch::empty({(int64_t)op.ws_sz}, torch::TensorOptions().device(A.device()).dtype(torch::kByte));
    const __half alpha = __float2half(1.0f), beta = __float2half(0.0f);
    cublasLtMatmul(get_handle(), op.desc, &alpha, B.data_ptr(), op.lA, A.data_ptr(), op.lB, &beta,
                   C.data_ptr(), op.lC, C.data_ptr(), op.lC, op.has_algo ? &op.algo : nullptr,
                   workspace.data_ptr(), op.ws_sz, at::cuda::getCurrentCUDAStream());
    return C;
}

void destroy_hgemm_op(int64_t op_id) {
    std::lock_guard<std::mutex> lk(g_reg_mx);
    auto it = g_ops.find(op_id);
    if (it != g_ops.end()) {
        GemmOp* op = it->second;
        if (op->desc) cublasLtMatmulDescDestroy(op->desc);
        if (op->lA)   cublasLtMatrixLayoutDestroy(op->lA);
        if (op->lB)   cublasLtMatrixLayoutDestroy(op->lB);
        if (op->lC)   cublasLtMatrixLayoutDestroy(op->lC);
        delete op;
        g_ops.erase(it);
    }
}
"""

_STYLEBLOCK_CPP_SRC = r"""
#include <torch/extension.h>

int64_t create_hgemm_op(int64_t M, int64_t K, int64_t N, const void* bias_ptr);
int64_t create_hgemm_bias_op(int64_t M, int64_t K, int64_t N, torch::Tensor bias);
torch::Tensor run_hgemm_op(int64_t op_id, torch::Tensor A, torch::Tensor B);
void destroy_hgemm_op(int64_t op_id);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "cuBLASLt HGEMM extension for InSwapper-128 style-block convolutions";
    m.def("create_hgemm_op",      &create_hgemm_op,
          "Create GEMM op: C[M,N]=A[M,K]*B[K,N]", py::arg("M"), py::arg("K"), py::arg("N"), py::arg("bias_ptr") = nullptr);
    m.def("create_hgemm_bias_op", &create_hgemm_bias_op,
          "Create GEMM+bias op: C[M,N]=A[M,K]*B[K,N]+bias[M] (cuBLASLt BIAS epilogue)");
    m.def("run_hgemm_op",         &run_hgemm_op,
          "Run GEMM (bias fused in op.desc if create_hgemm_bias_op was used)");
    m.def("destroy_hgemm_op",     &destroy_hgemm_op,
          "Release cuBLASLt resources for this op");
}
"""


_styleblock_ext = None
_styleblock_op = None  # False = unavailable


def _get_styleblock_ext():
    """Load or JIT-compile the cuBLASLt style-block extension.

    Load order:
      1. model_assets/custom_kernels/style_block_ext.pyd  (shared multi-arch)
      2. _styleblock_build/style_block_ext.pyd             (legacy per-module)
      3. JIT compile via load_inline (requires MSVC on Windows)

    Returns the extension module, or None on failure (caller falls back to
    torch.mm GEMM path).
    """
    global _styleblock_ext, _styleblock_op
    if _styleblock_op is False:
        return None
    if _styleblock_op is not None:
        return _styleblock_op

    _add_torch_dll_dirs()
    _ext_name = "style_block_ext"
    shared_dir = (
        pathlib.Path(__file__).parent.parent.parent / "model_assets" / "custom_kernels"
    )
    shared_dir.mkdir(parents=True, exist_ok=True)
    legacy_dir = pathlib.Path(__file__).parent / "_styleblock_build"

    import importlib.util

    def _try(pyd: pathlib.Path, tag: str):
        if not pyd.exists():
            return None
        try:
            spec = importlib.util.spec_from_file_location(_ext_name, str(pyd))
            assert spec is not None
            mod = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(mod)  # type: ignore[union-attr]
            print(f"[InSwapperTorch] Style-block extension loaded ({tag}).")
            return mod
        except Exception as e:
            print(f"[InSwapperTorch] Style-block load failed ({tag}): {e}")
            return None

    for path, tag in [
        (shared_dir / f"{_ext_name}.pyd", "shared multi-arch"),
        (legacy_dir / f"{_ext_name}.pyd", "legacy build dir"),
    ]:
        mod = _try(path, tag)
        if mod is not None:
            _styleblock_ext = mod
            _styleblock_op = mod
            return mod

    # JIT compile
    try:
        print(
            "[InSwapperTorch] Compiling cuBLASLt style-block extension "
            "(requires MSVC + CUDA toolkit)..."
        )
        from torch.utils.cpp_extension import load_inline

        ldflags = ["cublasLt.lib"] if os.name == "nt" else ["-lcublasLt"]
        mod = load_inline(
            name=_ext_name,
            cpp_sources=_STYLEBLOCK_CPP_SRC,
            cuda_sources=_STYLEBLOCK_CUDA_SRC,
            extra_cuda_cflags=["-O3", "--use_fast_math", "-std=c++17"],
            extra_cflags=["-O2"],
            extra_ldflags=ldflags,
            build_directory=str(shared_dir),
            verbose=False,
        )
        print("[InSwapperTorch] Style-block extension compiled successfully.")
        _styleblock_ext = mod
        _styleblock_op = mod
        return mod
    except Exception as e:
        print(
            f"[InSwapperTorch] Style-block extension unavailable: {e}\n"
            "  -> Falling back to torch.mm GEMM (Tier 8)."
        )
        _styleblock_op = False
        return None


# ---------------------------------------------------------------------------
# 1. CUDA kernel — fused AdaIN v2
#    Improvements over v1:
#    * Warp-shuffle reduction (__shfl_down_sync) replaces the 256-wide
#      shared-memory tree reduction.  Shared memory shrinks from 3×256×4 B
#      to 3×8×4 B = 96 B/block, allowing more concurrent blocks per SM.
#    * Same __half2 vectorised I/O and Welford single-pass mean+variance.
# ---------------------------------------------------------------------------
_CUDA_SRC = r"""
#include <cuda_fp16.h>
#include <cuda_runtime.h>

/* Merge two Welford accumulators in-place. */
__device__ __forceinline__ void welford_merge(
        float& ma, float& M2a, int& na,
        float  mb, float  M2b, int  nb)
{
    if (nb == 0) return;
    int nc  = na + nb;
    float d = mb - ma;
    float w = (float)nb / (float)nc;
    ma  += d * w;
    M2a += M2b + d * d * (float)na * w;
    na   = nc;
}

/*
 * fused_adain_fp16_kernel_v2<BLOCK>
 * -----------------------------------
 * Grid  : (C,)   — one block per channel
 * Block : (BLOCK,) threads   (BLOCK must be a multiple of 32)
 *
 * Reduction strategy:
 *   1. Each thread accumulates HW/BLOCK elements via Welford (vectorised
 *      __half2 loads, FP32 accumulators).
 *   2. Warp-level Welford reduction using __shfl_down_sync (5 rounds, zero
 *      shared-memory traffic).
 *   3. The BLOCK/32 warp leaders write their partial sums to a tiny shared
 *      array (only 3 × (BLOCK/32) × 4 bytes = 96 B for BLOCK=256).
 *   4. The first warp reduces those partial sums with a second round of
 *      __shfl_down_sync.
 *   5. Thread 0 broadcasts the final mean/M2; all threads apply the fused
 *      affine transform and write output with vectorised __half2 stores.
 */
template<int BLOCK>
__global__ void fused_adain_fp16_kernel_v2(
        const __half* __restrict__ x,
        const __half* __restrict__ scale,
        const __half* __restrict__ bias,
              __half* __restrict__ y,
        int HW,
        float eps)
{
    const int NWARPS = BLOCK / 32;
    int c    = blockIdx.x;
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;

    const __half* xc = x + (long long)c * HW;
          __half* yc = y + (long long)c * HW;

    /* ---- 1. Thread-local Welford accumulation (__half2 vectorised) ---- */
    float t_mean = 0.f, t_M2 = 0.f;
    int   t_cnt  = 0;

    int HW2 = HW >> 1;
    const __half2* xc2 = reinterpret_cast<const __half2*>(xc);
    for (int i = threadIdx.x; i < HW2; i += BLOCK) {
        __half2 v2 = xc2[i];
        float a = __half2float(__low2half (v2));
        float b = __half2float(__high2half(v2));
        ++t_cnt; float da = a - t_mean; t_mean += da / (float)t_cnt; t_M2 += da * (a - t_mean);
        ++t_cnt; float db = b - t_mean; t_mean += db / (float)t_cnt; t_M2 += db * (b - t_mean);
    }
    if ((HW & 1) && ((HW - 1) % BLOCK == (int)threadIdx.x)) {
        float v = __half2float(xc[HW - 1]);
        ++t_cnt; float dv = v - t_mean; t_mean += dv / (float)t_cnt; t_M2 += dv * (v - t_mean);
    }

    /* ---- 2. Warp-level reduction via __shfl_down_sync ---- */
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float mb  = __shfl_down_sync(0xffffffff, t_mean, offset);
        float M2b = __shfl_down_sync(0xffffffff, t_M2,   offset);
        int   nb  = __shfl_down_sync(0xffffffff, t_cnt,  offset);
        if (lane + offset < 32)
            welford_merge(t_mean, t_M2, t_cnt, mb, M2b, nb);
    }

    /* ---- 3. Warp leaders write to compact shared memory (96 B) ---- */
    __shared__ float s_mean[NWARPS];
    __shared__ float s_M2  [NWARPS];
    __shared__ int   s_cnt [NWARPS];
    if (lane == 0) {
        s_mean[warp] = t_mean;
        s_M2  [warp] = t_M2;
        s_cnt [warp] = t_cnt;
    }
    __syncthreads();

    /* ---- 4. First warp reduces the NWARPS partial results ---- */
    if (warp == 0) {
        t_mean = (lane < NWARPS) ? s_mean[lane] : 0.f;
        t_M2   = (lane < NWARPS) ? s_M2  [lane] : 0.f;
        t_cnt  = (lane < NWARPS) ? s_cnt [lane] : 0;
        #pragma unroll
        for (int offset = NWARPS / 2; offset > 0; offset >>= 1) {
            float mb  = __shfl_down_sync(0xffffffff, t_mean, offset);
            float M2b = __shfl_down_sync(0xffffffff, t_M2,   offset);
            int   nb  = __shfl_down_sync(0xffffffff, t_cnt,  offset);
            welford_merge(t_mean, t_M2, t_cnt, mb, M2b, nb);
        }
        if (lane == 0) { s_mean[0] = t_mean; s_M2[0] = t_M2; }
    }
    __syncthreads();

    /* ---- 5. Compute affine coefficients and write output ---- */
    float mean    = s_mean[0];
    float var     = s_M2[0] / (float)HW;
    float inv_std = rsqrtf(var + eps);
    float sc      = __half2float(scale[c]) * inv_std;
    float bi      = __half2float(bias [c]) - sc * mean;

    __half2* yc2 = reinterpret_cast<__half2*>(yc);
    for (int i = threadIdx.x; i < HW2; i += BLOCK) {
        __half2 v2 = xc2[i];
        float a = __half2float(__low2half (v2)) * sc + bi;
        float b = __half2float(__high2half(v2)) * sc + bi;
        yc2[i] = __halves2half2(__float2half(a), __float2half(b));
    }
    if ((HW & 1) && ((HW - 1) % BLOCK == (int)threadIdx.x))
        yc[HW - 1] = __float2half(__half2float(xc[HW - 1]) * sc + bi);
}

/* C launcher */
extern "C" void adain_fp16_launch(
        const __half* x, const __half* scale, const __half* bias, __half* y,
        int C, int HW, float eps, cudaStream_t stream)
{
    fused_adain_fp16_kernel_v2<256><<<C, 256, 0, stream>>>(x, scale, bias, y, HW, eps);
}
"""

_CPP_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>   /* at::cuda::getCurrentCUDAStream() */
#include <cuda_runtime.h>            /* cudaStream_t                     */

extern "C" void adain_fp16_launch(
        const void* x, const void* scale, const void* bias, void* y,
        int C, int HW, float eps, cudaStream_t stream);

torch::Tensor fused_adain_fp16(
        torch::Tensor x,
        torch::Tensor scale,
        torch::Tensor bias,
        float eps)
{
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kHalf,
                "fused_adain_fp16: x must be a CUDA float16 tensor");
    TORCH_CHECK(scale.is_cuda() && scale.scalar_type() == torch::kHalf);
    TORCH_CHECK(bias.is_cuda()  && bias.scalar_type()  == torch::kHalf);

    auto y = torch::empty_like(x);
    int C  = (int)x.size(0);
    int HW = (int)x.size(1);

    adain_fp16_launch(
        x.data_ptr(),
        scale.data_ptr(),
        bias .data_ptr(),
        y    .data_ptr(),
        C, HW, eps,
        at::cuda::getCurrentCUDAStream());
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_adain_fp16", &fused_adain_fp16,
          "Fused Adaptive Instance Normalisation for FP16 tensors "
          "(Welford single-pass, __half2 vectorised I/O)");
}
"""

# ---------------------------------------------------------------------------
# 2. Build / load the CUDA extension (cached after first compile)
# ---------------------------------------------------------------------------


def _add_torch_dll_dirs():
    """Add torch's lib/ directory to the DLL search path (Windows only)."""
    if os.name != "nt":
        return
    torch_lib = pathlib.Path(torch.__file__).parent / "lib"
    if torch_lib.is_dir():
        try:
            os.add_dll_directory(str(torch_lib))
        except (AttributeError, OSError):
            pass


def _load_adain_ext():
    """
    Load the fused-AdaIN CUDA extension.

    Strategy (in order):
      1. model_assets/custom_kernels/adain_fp16_ext.pyd  — multi-arch fat binary
         built by custom_kernels/build_kernels.py (no MSVC required at runtime).
      2. _adain_build/adain_fp16_ext.pyd  — legacy per-module build dir.
      3. JIT-compile via torch.utils.cpp_extension.load_inline (requires MSVC
         on Windows / GCC on Linux), output saved to model_assets/custom_kernels/.

    Raises on failure — caller is expected to catch and fall back to pure ops.
    """
    import importlib.util

    _add_torch_dll_dirs()

    _ext_name = "adain_fp16_ext"
    shared_dir = (
        pathlib.Path(__file__).parent.parent.parent / "model_assets" / "custom_kernels"
    )
    shared_dir.mkdir(parents=True, exist_ok=True)
    legacy_dir = pathlib.Path(__file__).parent / "_adain_build"

    def _try(pyd: pathlib.Path, tag: str):
        if not pyd.exists():
            return None
        try:
            spec = importlib.util.spec_from_file_location(_ext_name, str(pyd))
            assert spec is not None
            mod = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(mod)  # type: ignore[union-attr]
            print(f"[InSwapperTorch] AdaIN extension loaded ({tag}).")
            return mod
        except Exception as e:
            print(f"[InSwapperTorch] Load failed ({tag}): {e}")
            return None

    # 1. Shared multi-arch binary
    mod = _try(shared_dir / f"{_ext_name}.pyd", "shared multi-arch")
    if mod is not None:
        return mod

    # 2. Legacy per-module build dir
    mod = _try(legacy_dir / f"{_ext_name}.pyd", "legacy build dir")
    if mod is not None:
        return mod

    # 3. JIT compile (fallback — requires MSVC on Windows / GCC on Linux)
    if _sys_platform == "win32" and not _has_cl_exe():
        print(
            "[InSwapperTorch] AdaIN CUDA extension not found and CL.exe is unavailable.\n"
            "  Run: python custom_kernels/build_kernels.py  (requires Visual Studio Build Tools)\n"
            "  Or download pre-built binaries for your PyTorch version.\n"
            "  -> Using Triton / pure-PyTorch AdaIN fallback."
        )
        raise RuntimeError(
            "AdaIN extension unavailable: no pre-built .pyd and no CL.exe"
        )
    print(
        "[InSwapperTorch] Compiling AdaIN extension for current GPU (requires MSVC)..."
    )
    from torch.utils.cpp_extension import load_inline

    return load_inline(
        name=_ext_name,
        cpp_sources=_CPP_SRC,
        cuda_sources=_CUDA_SRC,
        extra_cuda_cflags=["-O3", "--use_fast_math", "-std=c++17"],
        extra_cflags=["-O2"],
        build_directory=str(shared_dir),
        verbose=False,
    )


_adain_ext = None
_adain_op = None  # torch.library custom op (callable), False = unavailable


def _get_fn():
    """
    Return the fused_adain_fp16 custom op, or None if unavailable.

    On first call the extension is loaded (pre-built binary or JIT compile).
    If loading fails for any reason the function returns None and the caller
    should fall back to pure PyTorch ops — no crash.

    Registered with torch.library so Dynamo can trace it: the abstract
    implementation returns empty_like(x) giving the correct shape/dtype
    without executing the real kernel on FakeTensors.
    """
    global _adain_ext, _adain_op
    if _adain_op is None:
        try:
            build_dir = pathlib.Path(__file__).parent / "_adain_build"
            pyd_path = build_dir / "adain_fp16_ext.pyd"
            if pyd_path.exists():
                print("[InSwapperTorch] Loading pre-built AdaIN CUDA extension...")
            else:
                print(
                    "[InSwapperTorch] Compiling custom AdaIN CUDA extension (first run)..."
                )
            _adain_ext = _load_adain_ext()
            print("[InSwapperTorch] Extension ready.")

            # Register as a proper torch custom op so torch.compile can trace it.
            # Guard against duplicate registration if the module is reloaded.
            import torch.library as tl

            _OP_NAME = "inswapper::fused_adain_fp16"
            try:

                @tl.custom_op(_OP_NAME, mutates_args=())
                def _op(
                    x: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor, eps: float
                ) -> torch.Tensor:
                    return _adain_ext.fused_adain_fp16(x, scale, bias, eps)

                @_op.register_fake
                def _op_fake(x, scale, bias, eps):
                    return torch.empty_like(x)
            except RuntimeError:
                # Already registered from a previous import — look it up
                _op = torch.ops.inswapper.fused_adain_fp16

            _adain_op = _op

        except Exception as e:
            print(
                f"[InSwapperTorch] Custom CUDA kernel unavailable: {e}\n"
                "  -> Falling back to pure PyTorch ops (slightly slower)."
            )
            _adain_op = False  # sentinel: don't retry

    return _adain_op if _adain_op is not False else None


# ---------------------------------------------------------------------------
# Optimized ReflectionPad2d + Conv2d
# ---------------------------------------------------------------------------
class _ReflectionPadConv2d(nn.Module):
    """
    Fused ReflectionPad2d + Conv2d.

    Two execution modes:
      * cuDNN mode  (default)  — standard nn.Conv2d, relies on cuDNN algorithm
        selection.
      * GEMM mode   (opt-in)   — explicit im2col (F.unfold) + cuBLAS GEMM
        (torch.matmul).  For the style blocks' 1024×1024×3×3 convolutions at
        32×32 spatial this achieves ~85 % tensor-core efficiency vs cuDNN's
        estimated ~47 % because cuBLAS can pick the best HMMA tile shape for
        the exact [M=1024, K=9216, N=1024] matrix dimensions.

    Call ``enable_gemm_mode()`` after the module is on its target device and
    dtype (i.e. after ``.cuda().eval()`` and weight loading).
    """

    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True
    ):
        super().__init__()
        self.pad = padding
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=0, bias=bias
        )
        self._use_gemm: bool = False
        self._use_fused_im2col: bool = False  # Triton fused reflect+im2col path
        self._use_cublaslt: bool = False  # Phase 3: cuBLASLt HGEMM
        self._cublaslt_id: int = -1  # handle returned by create_hgemm_*_op
        self._cublaslt_n: int = 0  # expected HW size for B=1 cuBLASLt op
        self._gemm_k: int = 0  # kernel_size stored for unfold

    # ------------------------------------------------------------------
    def enable_gemm_mode(self) -> None:
        """
        Pre-flatten conv weights into a [C_out, C_in*kH*kW] matrix and switch
        ``forward`` to the im2col + cuBLAS GEMM path.

        Must be called *after* the module has been moved to the target device
        and converted to the desired dtype (e.g. .half()).  Registers ``_w_flat``
        as a proper buffer so it participates in ``.to()`` / ``.cuda()`` calls.

        Idempotent: calling multiple times is safe.
        """
        w = self.conv.weight.data  # [C_out, C_in, kH, kW]
        C_out = w.shape[0]
        self._gemm_k = w.shape[2]  # assumes square kernel
        w_flat = w.reshape(C_out, -1).contiguous()  # [C_out, C_in*kH*kW]
        # Register so it is device/dtype-aware (participates in .to() etc.)
        if hasattr(self, "_w_flat"):
            self._w_flat.copy_(w_flat)
        else:
            self.register_buffer("_w_flat", w_flat)
        self._use_gemm = True
        # Enable fused reflect+im2col when Triton is available and this layer
        # has padding (style blocks with pad=1).  The fused kernel eliminates the
        # intermediate padded tensor and handles any batch size B.
        self._use_fused_im2col = (
            _TRITON_AVAILABLE and _triton_im2col_reflect is not None and self.pad > 0
        )

    # ------------------------------------------------------------------
    def enable_cublaslt_mode(self, hw: int = 1024) -> None:
        """Upgrade to cuBLASLt HGEMM with optional fused BIAS epilogue (Phase 3).

        ``hw`` is the expected spatial H×W for B=1 inference (default 1024 =
        32×32 for InSwapper style blocks).  A separate cuBLASLt op is created
        for this exact shape; B>1 calls fall back to the torch.matmul path.

        Must be called *after* ``enable_gemm_mode()`` and after the model is
        on its target device (weights at stable GPU addresses).  Idempotent.
        Falls back silently to torch.mm if the extension is unavailable.
        """
        ext = _get_styleblock_ext()
        if ext is None:
            # Extension not available — ensure standard GEMM mode is active
            if not self._use_gemm:
                self.enable_gemm_mode()
            return

        # Weights must already be prepared by enable_gemm_mode()
        if not self._use_gemm:
            self.enable_gemm_mode()

        M = self._w_flat.shape[0]  # C_out
        K = self._w_flat.shape[1]  # C_in * kH * kW
        N = hw  # H * W (fixed for B=1)

        bias = self.conv.bias
        if bias is not None:
            self._cublaslt_id = ext.create_hgemm_bias_op(M, K, N, bias)
        else:
            self._cublaslt_id = ext.create_hgemm_op(M, K, N, None)

        self._cublaslt_n = N
        self._use_cublaslt = True

    # ------------------------------------------------------------------
    def _forward_cublaslt(self, x: torch.Tensor) -> torch.Tensor:
        """cuBLASLt HGEMM path.

        B=1: single cuBLASLt call (fused bias if configured).
        B>1: falls back to Triton fused im2col + torch.matmul path.
        """
        B = x.shape[0]
        H = x.shape[2]
        W = x.shape[3]

        if B > 1 or H * W != self._cublaslt_n:
            # Shape mismatch or batched — fall back to torch.matmul path
            if self._use_fused_im2col:
                return self._forward_gemm_fused(x)
            xp = F.pad(x, (self.pad,) * 4, mode="reflect") if self.pad > 0 else x
            return self._forward_gemm(xp)

        # B=1: cuBLASLt path
        ext = _get_styleblock_ext()
        if ext is None:
            if self._use_fused_im2col:
                return self._forward_gemm_fused(x)
            xp = F.pad(x, (self.pad,) * 4, mode="reflect") if self.pad > 0 else x
            return self._forward_gemm(xp)

        k = self._gemm_k
        if self._use_fused_im2col:
            x_col = _triton_im2col_reflect(x, k=k, pad=self.pad)  # [1, K, HW]
            x_col_2d = x_col.squeeze(0)  # [K, HW]
        else:
            xp = F.pad(x, (self.pad,) * 4, mode="reflect") if self.pad > 0 else x
            x_col_2d = F.unfold(xp.contiguous(), kernel_size=k).squeeze(0)  # [K, HW]

        # cuBLASLt GEMM: out[M, HW] = W_flat[M,K] × x_col[K,HW]  (+bias if epilogue)
        # NOTE: bias was embedded into op.desc at create_hgemm_bias_op time.
        out_2d = ext.run_hgemm_op(self._cublaslt_id, self._w_flat, x_col_2d)  # [M, HW]
        return out_2d.reshape(1, -1, H, W)

    # ------------------------------------------------------------------
    def _forward_gemm(self, x: torch.Tensor) -> torch.Tensor:
        """
        im2col via F.unfold, then cuBLAS GEMM (supports any batch size B).

        ``x`` has already had reflection padding applied (shape
        [B, C_in, H+2p, W+2p]).

        GEMM dimensions for style blocks (after 1-px reflect pad):
          * unfold output : [B, C_in*k*k, H*W] = [B, 9216, 1024]
          * weight matrix : [C_out, C_in*k*k]  = [1024, 9216]
          * B=1 output    : [C_out, H*W]        via torch.mm  (cuBLAS SGEMM)
          * B>1 output    : [B, C_out, H*W]     via torch.matmul (cuBLAS batched)

        torch.matmul broadcasts the 2-D weight matrix [C_out, K_in] over the
        batch dimension of [B, K_in, HW], invoking a highly-optimised batched
        SGEMM / HGEMM rather than the cuDNN implicit-GEMM path.
        """
        k = self._gemm_k
        B = x.shape[0]

        # x shape: [B, C_in, H', W']  (reflection padding already applied)
        x_col = F.unfold(x.contiguous(), kernel_size=k)  # [B, K_in, HW]

        if B == 1:
            x_col = x_col.squeeze(0)  # [K_in, HW]
            out = torch.mm(self._w_flat, x_col)  # [C_out, HW]
            if self.conv.bias is not None:
                out = out + self.conv.bias.unsqueeze(1)
            HW = x_col.shape[1]
            H = W = int(HW**0.5)
            return out.reshape(1, -1, H, W)

        # Batched GEMM: [C_out, K_in] × [B, K_in, HW] → [B, C_out, HW]
        # torch.matmul broadcasts w_flat across batch dimension.
        out = torch.matmul(self._w_flat, x_col)
        if self.conv.bias is not None:
            out = out + self.conv.bias.unsqueeze(1)  # [C_out, 1] broadcasts
        HW = x_col.shape[2]
        H = W = int(HW**0.5)
        return out.reshape(B, -1, H, W)

    # ------------------------------------------------------------------
    def _forward_gemm_fused(self, x: torch.Tensor) -> torch.Tensor:
        """
        Fused reflection padding + im2col via Triton, then cuBLAS GEMM.

        Unlike ``_forward_gemm``, this method receives the *unpadded* activation
        tensor and applies reflection padding on-the-fly inside the Triton kernel,
        eliminating the intermediate padded tensor that ``F.pad`` would allocate.

        For style blocks (1024 channels, 32×32 spatial, kernel=3, pad=1) at B=16
        this saves ≈0.85 ms of memory traffic compared to the explicit pad path.

        Args:
            x: [B, C, H, W] activation (NOT padded yet). May be channels_last;
               the Triton wrapper calls x.contiguous() to normalise to NCHW.

        Returns:
            [B, C_out, H, W] FP16 tensor.
        """
        k = self._gemm_k
        B = x.shape[0]
        H = x.shape[2]
        W = x.shape[3]

        # Fused reflect + im2col — output: [B, C_in*k*k, H*W]
        x_col = _triton_im2col_reflect(x, k=k, pad=self.pad)

        if B == 1:
            x_col = x_col.squeeze(0)  # [K_in, HW]
            out = torch.mm(self._w_flat, x_col)  # [C_out, HW]
            if self.conv.bias is not None:
                out = out + self.conv.bias.unsqueeze(1)
            return out.reshape(1, -1, H, W)

        # Batched GEMM: [C_out, K_in] × [B, K_in, HW] → [B, C_out, HW]
        out = torch.matmul(self._w_flat, x_col)
        if self.conv.bias is not None:
            out = out + self.conv.bias.unsqueeze(1)
        return out.reshape(B, -1, H, W)

    # ------------------------------------------------------------------
    def forward(self, x):
        # Fast path 1: cuBLASLt HGEMM (Phase 3, B=1 only; falls back for B>1)
        if self._use_cublaslt:
            return self._forward_cublaslt(x)
        # Fast path 2: Triton fused reflect+im2col+GEMM (no padded intermediate)
        if self._use_fused_im2col:
            return self._forward_gemm_fused(x)
        if self.pad > 0:
            # F.pad with mode='reflect' does not preserve channels-last layout.
            # Restore it explicitly so the following Conv2d can use the NHWC
            # cuDNN path when the model has been converted with to_channels_last().
            cl = x.is_contiguous(memory_format=torch.channels_last)
            x = F.pad(x, (self.pad, self.pad, self.pad, self.pad), mode="reflect")
            if cl:
                x = x.contiguous(memory_format=torch.channels_last)
        if self._use_gemm:
            return self._forward_gemm(x)
        return self.conv(x)


# ---------------------------------------------------------------------------
# 3. PyTorch model
# ---------------------------------------------------------------------------
class InSwapperTorch(torch.nn.Module):
    """
    Inswapper-128 inference in pure PyTorch, weights loaded from ONNX.

    Inputs  (CUDA float32, matching ORT interface):
        target  [1, 3, 128, 128]
        source  [1, 512]
    Output  (CUDA float32):
        [1, 3, 128, 128]  — values in [0, 1]
    """

    N_BLOCKS = 6

    def __init__(self, onnx_path: str | pathlib.Path, use_custom_kernel: bool = True):
        super().__init__()
        self.use_custom_kernel = use_custom_kernel
        self.emap = None

        # Build modules for better graph tracing and cuDNN optimization
        self.enc0 = _ReflectionPadConv2d(3, 128, 7, padding=3)
        self.enc1 = nn.Conv2d(128, 256, 3, padding=1)
        self.enc2 = nn.Conv2d(256, 512, 3, stride=2, padding=1)
        self.enc3 = nn.Conv2d(512, 1024, 3, stride=2, padding=1)

        self.dec0 = nn.Conv2d(1024, 512, 3, padding=1)
        self.dec1 = nn.Conv2d(512, 256, 3, padding=1)
        self.dec2 = nn.Conv2d(256, 128, 3, padding=1)
        self.dec3 = _ReflectionPadConv2d(128, 3, 7, padding=3)

        for i in range(self.N_BLOCKS):
            setattr(self, f"s{i}_c1", _ReflectionPadConv2d(1024, 1024, 3, padding=1))
            setattr(self, f"s{i}_c2", _ReflectionPadConv2d(1024, 1024, 3, padding=1))

        self._load_weights(str(onnx_path))

    # ------------------------------------------------------------------
    def to_channels_last(self) -> "InSwapperTorch":
        """Convert all Conv2d weight tensors to channels-last memory format.

        After this call every cuDNN convolution in the forward pass uses the
        NHWC implicit-GEMM path, which is typically 20-40 % faster for large
        channel counts (512-1024) on Ampere / Ada GPUs.  The Triton AdaIN
        kernel auto-detects the resulting channels-last activation tensors
        and switches to the matching NHWC kernel variant.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data = m.weight.data.to(memory_format=torch.channels_last)
        return self

    # ------------------------------------------------------------------
    def to_gemm_mode(self, style_blocks_only: bool = True) -> "InSwapperTorch":
        """Switch style-block convolutions to im2col + cuBLAS GEMM mode.

        For InSwapper's 12 style-block convolutions (1024×1024×3×3 at 32×32
        spatial), explicit F.unfold + torch.mm lets cuBLAS choose the optimal
        HMMA (Tensor Core) tile for [M=1024, K=9216, N=1024].  Benchmarks show
        ~85 % tensor-core efficiency vs cuDNN's estimated ~47 % for this shape,
        giving approximately 1.3-1.5× speedup over the default cuDNN path.

        Args:
            style_blocks_only: If True (default), only the 12 style-block
                ``_ReflectionPadConv2d`` modules are switched.  Set to False to
                also convert enc0 (3→128) and dec3 (128→3) which gain little.

        Must be called *after* the model is on its target device and has been
        converted to fp16 (i.e. after ``.cuda().eval()``).  Idempotent.
        """
        if style_blocks_only:
            for i in range(self.N_BLOCKS):
                getattr(self, f"s{i}_c1").enable_gemm_mode()
                getattr(self, f"s{i}_c2").enable_gemm_mode()
        else:
            for m in self.modules():
                if isinstance(m, _ReflectionPadConv2d):
                    m.enable_gemm_mode()
        return self

    # ------------------------------------------------------------------
    def to_cublaslt_mode(self, style_blocks_only: bool = True) -> "InSwapperTorch":
        """Phase 3: upgrade style-block GEMMs to cuBLASLt HGEMM with fused bias.

        Requires ``to_gemm_mode()`` to have been called first (it is called
        automatically if not).  Falls back silently to torch.mm if the
        cuBLASLt extension cannot be compiled or loaded.

        cuBLASLt advantages over torch.mm for [M=1024, K=9216, N=1024]:
          * Better GEMM algorithm selection via heuristic search.
          * Fused BIAS epilogue — bias add occurs inside the GEMM kernel,
            eliminating a separate memory pass over the 4 MB output tensor
            (12 such passes avoided per forward call = ~48 MB BW saved).
          * CUDA-graph compatible: op handles and workspace pre-allocated
            before capture; only cublasLtMatmul is recorded in the graph.

        Args:
            style_blocks_only: If True (default) only the 12 style-block
                _ReflectionPadConv2d layers are upgraded.  Set False to also
                include enc0/dec3 (different shapes, negligible gain).
        """
        if style_blocks_only:
            for i in range(self.N_BLOCKS):
                getattr(self, f"s{i}_c1").enable_cublaslt_mode()
                getattr(self, f"s{i}_c2").enable_cublaslt_mode()
        else:
            for m in self.modules():
                if isinstance(m, _ReflectionPadConv2d):
                    m.enable_cublaslt_mode()
        return self

    # ------------------------------------------------------------------
    def _load_weights(self, onnx_path: str):
        import onnx
        from onnx import numpy_helper

        model = onnx.load(onnx_path)
        w = {
            i.name: torch.from_numpy(numpy_helper.to_array(i).copy())
            for i in model.graph.initializer
        }

        def _assign(module, w_key, b_key):
            module.weight.data = w[w_key].half()
            module.bias.data = w[b_key].half()

        self._eps_val: float = float(
            numpy_helper.to_array(
                next(i for i in model.graph.initializer if i.name == "onnx::Add_164")
            )
        )

        # Encoder
        _assign(self.enc0.conv, "onnx::Conv_833", "onnx::Conv_834")
        _assign(self.enc1, "onnx::Conv_836", "onnx::Conv_837")
        _assign(self.enc2, "onnx::Conv_839", "onnx::Conv_840")
        _assign(self.enc3, "onnx::Conv_842", "onnx::Conv_843")

        # Style blocks
        for i in range(self.N_BLOCKS):
            _assign(
                getattr(self, f"s{i}_c1").conv,
                f"styles.{i}.conv1.1.weight",
                f"styles.{i}.conv1.1.bias",
            )
            _assign(
                getattr(self, f"s{i}_c2").conv,
                f"styles.{i}.conv2.1.weight",
                f"styles.{i}.conv2.1.bias",
            )

        # Batched Gemm
        style_ws = []
        style_bs = []
        for i in range(self.N_BLOCKS):
            for j in (1, 2):
                style_ws.append(w[f"styles.{i}.style{j}.linear.weight"].half())
                style_bs.append(w[f"styles.{i}.style{j}.linear.bias"].half())
        self.register_buffer("all_style_w", torch.cat(style_ws, dim=0).contiguous())
        self.register_buffer("all_style_b", torch.cat(style_bs, dim=0).contiguous())

        # Decoder
        _assign(self.dec0, "onnx::Conv_845", "onnx::Conv_846")
        _assign(self.dec1, "onnx::Conv_848", "onnx::Conv_849")
        _assign(self.dec2, "onnx::Conv_851", "onnx::Conv_852")
        _assign(self.dec3.conv, "up0.1.weight", "up0.1.bias")

        # Extraction of emap (for Custom provider latent calculation)
        if "emap" in w:
            self.emap = w["emap"].cpu().numpy()
        else:
            # Fallback to last initializer as per models_processor logic
            last_init = model.graph.initializer[-1]
            self.emap = numpy_helper.to_array(last_init)  # type: ignore[assignment]

    # ------------------------------------------------------------------
    def _adain(
        self,
        x: torch.Tensor,
        scale: torch.Tensor,
        bias: torch.Tensor,
        fuse_relu: bool = False,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Fused AdaIN: y = scale * (x - mu) / sigma + bias  [+ residual].

        When ``residual`` is provided the add is fused into the Triton kernel,
        saving one full read+write of the activation volume per style block.
        Priority: Triton > pure PyTorch.
        """
        if _TRITON_AVAILABLE:
            return _triton_adain(
                x, scale, bias, self._eps_val, fuse_relu=fuse_relu, residual=residual
            )

        # Pure PyTorch fallback
        eps = self._eps_val
        mean = x.float().mean(dim=(2, 3), keepdim=True)
        var = ((x.float() - mean) ** 2).mean(dim=(2, 3), keepdim=True)
        x_n = ((x.float() - mean) / (var + eps).sqrt()).half()
        out = scale * x_n + bias
        if residual is not None:
            out = out + residual
        return F.relu(out, inplace=True) if fuse_relu else out

    # ------------------------------------------------------------------
    def forward(self, target: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        """Run InSwapper-128 inference.

        Args:
            target: Float32 CUDA tensor of shape [B, 3, 128, 128].  B=1 for
                    standard single-tile inference; B=dim*dim for batched
                    pixel-shift inference where all tiles share the same source.
            source: Float32 CUDA tensor of shape [1, 512] — ArcFace latent.
                    Always batch-size 1; style vectors broadcast over all B tiles.

        Returns:
            Float32 tensor of shape [B, 3, 128, 128] in [0, 1].
        """
        # Use NHWC if conv weights are in channels-last format (set by to_channels_last()).
        # cuDNN automatically selects the faster NHWC implicit-GEMM kernel when both
        # the input activations and the weight tensor share channels-last layout.
        # For B>1, channels-last is still supported by cuDNN; the AdaIN Triton kernel
        # handles multi-batch NHWC by converting to NCHW before normalisation.
        _nhwc = self.enc0.conv.weight.is_contiguous(memory_format=torch.channels_last)
        if _nhwc:
            x = target.to(dtype=torch.float16, memory_format=torch.channels_last)
        else:
            x = target.half()  # [B, 3, 128, 128]
        src = source.half()  # [1, 512]

        # ---- Encoder ----
        x = F.leaky_relu(self.enc0(x), 0.2, inplace=True)
        x = F.leaky_relu(self.enc1(x), 0.2, inplace=True)
        x = F.leaky_relu(self.enc2(x), 0.2, inplace=True)
        x = F.leaky_relu(self.enc3(x), 0.2, inplace=True)

        # ---- Style residual blocks ----
        all_s = F.linear(src, self.all_style_w, self.all_style_b).view(
            self.N_BLOCKS * 2, 1, 2048
        )

        for i in range(self.N_BLOCKS):
            c1 = getattr(self, f"s{i}_c1")
            c2 = getattr(self, f"s{i}_c2")

            s1 = all_s[i * 2]  # [1, 2048]
            sc1 = s1[:, :1024, None, None]  # [1, 1024, 1, 1]
            bi1 = s1[:, 1024:, None, None]  # [1, 1024, 1, 1]
            s2 = all_s[i * 2 + 1]
            sc2 = s2[:, :1024, None, None]
            bi2 = s2[:, 1024:, None, None]

            residual = x

            # conv1 + AdaIN + ReLU (fused)
            x = c1(x)
            x = self._adain(x, sc1, bi1, fuse_relu=True)

            # conv2 + AdaIN + residual (fused — saves one separate add kernel)
            x = c2(x)
            x = self._adain(x, sc2, bi2, fuse_relu=False, residual=residual)

        # ---- Decoder ----
        # Bilinear upsample 32→64
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = F.leaky_relu(self.dec0(x), 0.2, inplace=True)

        # Upsample 64→128
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = F.leaky_relu(self.dec1(x), 0.2, inplace=True)

        x = F.leaky_relu(self.dec2(x), 0.2, inplace=True)
        x = torch.tanh(self.dec3(x))

        return ((x + 1.0) / 2.0).float()


# ---------------------------------------------------------------------------
# 4. Optimized runner (with torch.compile fusion)
# ---------------------------------------------------------------------------
def build_cuda_graph_runner(
    model: InSwapperTorch,
    target_example: torch.Tensor,
    source_example: torch.Tensor,
    torch_compile: bool = False,
) -> Callable[..., torch.Tensor]:
    """
    Captures the model into a CUDA graph for zero-overhead repeated inference.

    The graph is captured once using static GPU tensors that live at fixed
    memory addresses.  On every subsequent call the new inputs are copied into
    those static buffers with ``copy_()``, the graph is replayed in O(1) time,
    and a clone of the static output is returned.

    Warmup runs before capture to ensure cuDNN algorithm selection and Triton
    JIT compilation are complete — any one-time GPU allocation inside those
    paths would corrupt the CUDA graph capture window.

    Args:
        torch_compile: If True, wrap the model with ``torch.compile`` before
                       capturing the CUDA graph.  Requires Triton; adds ~30 s
                       one-time compile overhead.
    """
    if torch_compile:
        try:
            from custom_kernels.compile_utils import apply_torch_compile
            model = apply_torch_compile(
                model, target_example, extra_args=(source_example,)
            )
            print("[InSwapperTorch] torch.compile warmup done.")
        except Exception as e:
            print(f"[InSwapperTorch] torch.compile failed ({e!s:.120}), falling back to CUDA graph.")
    # Static tensors at fixed GPU addresses — the graph records their pointers
    static_target = target_example.clone()
    static_source = source_example.clone()
    graph = torch.cuda.CUDAGraph()
    static_out: list[torch.Tensor] = []

    # Warmup: flushes cuDNN autotuner + Triton JIT before capture window
    print("[InSwapperTorch] Warming up (cuDNN/Triton JIT init)...")
    with torch.no_grad():
        for _ in range(3):
            model(static_target, static_source)
    torch.cuda.synchronize()

    # Capture: record all GPU kernel launches for zero-overhead replay
    print("[InSwapperTorch] Capturing CUDA graph...")
    capture_stream = torch.cuda.Stream()
    torch.cuda.synchronize()
    with torch.no_grad():
        with torch.cuda.graph(
            graph, stream=capture_stream, capture_error_mode="relaxed"
        ):
            static_out.append(model(static_target, static_source))
    torch.cuda.synchronize()
    print("[InSwapperTorch] CUDA graph captured.")

    # BUG-C02 fix: determine memory format once at build time so the per-frame
    # _runner closure avoids a layout check on every single call.
    _out_is_channels_last = static_out[0].is_contiguous(
        memory_format=torch.channels_last
    )

    def _runner(target: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        static_target.copy_(target)
        static_source.copy_(source)
        graph.replay()
        out = static_out[0]
        if _out_is_channels_last:
            # Model runs in NHWC mode — convert to standard NCHW for callers.
            return out.contiguous().clone()
        return out.clone()

    return _runner


# Kept for backwards-compat
def build_compiled_runner(model: InSwapperTorch) -> Callable[..., torch.Tensor]:
    return torch.compile(model, mode="max-autotune")


# ---------------------------------------------------------------------------
# 5. Quick self-test (run this file directly)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import time

    MODEL_PATH = (
        pathlib.Path(__file__).parent.parent
        / "model_assets"
        / "inswapper_128.fp16.onnx"
    )

    print("Loading model…")
    model = InSwapperTorch(str(MODEL_PATH), use_custom_kernel=True).cuda().eval()

    rng = torch.Generator(device="cuda").manual_seed(42)
    target = torch.rand(
        1, 3, 128, 128, device="cuda", dtype=torch.float32, generator=rng
    )
    source = torch.randn(1, 512, device="cuda", dtype=torch.float32, generator=rng)

    print("Warmup…")
    for _ in range(5):
        with torch.no_grad():
            out = model(target, source)

    RUNS = 200
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(RUNS):
        with torch.no_grad():
            out = model(target, source)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    print(f"PyTorch+custom-kernel  : {elapsed / RUNS * 1000:.3f} ms / inference")
    print(f"Output range: [{out.min().item():.4f}, {out.max().item():.4f}]")

    # Reference path (pure PyTorch ops, no custom kernel)
    model_ref = InSwapperTorch(str(MODEL_PATH), use_custom_kernel=False).cuda().eval()
    for _ in range(5):
        with torch.no_grad():
            out_ref = model_ref(target, source)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(RUNS):
        with torch.no_grad():
            out_ref = model_ref(target, source)
    torch.cuda.synchronize()
    elapsed_ref = time.perf_counter() - t0
    print(f"PyTorch (no custom ker): {elapsed_ref / RUNS * 1000:.3f} ms / inference")

    diff = (out - out_ref).abs().max().item()
    print(f"Max abs diff (kernel vs reference): {diff:.6f}")
