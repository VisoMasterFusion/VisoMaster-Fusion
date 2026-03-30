"""
build_kernels.py
================
Compile all custom CUDA C++ extensions as multi-architecture fat binaries and
place them in model_assets/custom_kernels/ where the application finds them.

Run once (requires MSVC on Windows / GCC on Linux, plus CUDA toolkit):

    .venv/Scripts/python custom_kernels/build_kernels.py

The resulting .pyd files contain device code for all supported GPU generations
(sm_70 through sm_120), so a single binary works across GPU cards without
needing to recompile.  Building sm_120 (Blackwell / RTX 50xx) requires
CUDA Toolkit 12.8+; if the toolkit is older, sm_120 is skipped gracefully.

After a successful build the Triton JIT cache is also warmed up for the current
GPU so the first inference is fast.

Output directory:  model_assets/custom_kernels/
  adain_fp16_ext.pyd      -- InSwapper fused AdaIN
  gfpgan_demod_ext.pyd    -- GFPGAN / GPEN fused weight demodulation
  style_block_ext.pyd     -- InSwapper cuBLASLt HGEMM with fused BIAS epilogue (Phase 3)
  triton_cache/           -- Triton JIT compiled kernels (GPU-arch specific)
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Venv guard — re-exec with the project venv's Python if the user ran this
# script with a different interpreter (e.g. the bare `python` on PATH).
# Running with the wrong Python version causes a "SystemError: unknown opcode"
# crash because the venv's site-packages contain bytecode compiled for a
# different Python version.
# ---------------------------------------------------------------------------
_VENV_PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
if _VENV_PYTHON.exists():
    try:
        Path(sys.executable).resolve().relative_to((ROOT / ".venv").resolve())
    except ValueError:
        # sys.executable is not inside the project venv — re-exec with it.
        import subprocess as _sp
        sys.exit(_sp.run([str(_VENV_PYTHON)] + sys.argv, env=os.environ.copy()).returncode)


# ---------------------------------------------------------------------------
# On Windows, ensure MSVC (cl.exe) is on PATH before torch.utils.cpp_extension
# tries to use it.  We search common VS 2019/2022 BuildTools / Community
# install paths and inject the first cl.exe we find.
# ---------------------------------------------------------------------------
def _find_hostx64_cl() -> str | None:
    """Return the path to the best available Hostx64 cl.exe, or None.

    Search order (highest priority first):
      1. vswhere.exe  — drive-agnostic, finds VS on any drive letter (D:, E:, …)
      2. glob C:\\Program Files*  — fallback for non-standard / side-by-side installs

    Only ``Hostx64`` variants are considered.  The ``HostX86`` (32-bit host)
    crashes with INTERNAL COMPILER ERROR on large PyTorch headers.
    """
    import subprocess
    import glob as _glob

    cl_dirs: list[str] = []

    # --- 1. vswhere (canonical, drive-agnostic) ---
    vswhere = Path(
        r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
    )
    if vswhere.exists():
        try:
            # Ask vswhere for all VS installs (latest first) that provide
            # the x64 host → x64 target MSVC toolset.
            result = subprocess.run(
                [
                    str(vswhere),
                    "-all",  # all products (Community, BuildTools, …)
                    "-prerelease",  # include Preview if present
                    "-requires",
                    "Microsoft.VisualCpp.Tools.HostX64.TargetX64",
                    "-find",
                    r"VC\Tools\MSVC\**\bin\Hostx64\x64\cl.exe",
                ],
                capture_output=True,
                text=True,
                timeout=15,
            )
            for line in result.stdout.strip().splitlines():
                p = Path(line.strip())
                if p.exists():
                    cl_dirs.append(str(p.parent))
        except Exception as e:
            print(f"[build_kernels] vswhere query failed: {e}")

    # --- 2. glob fallback (C:\Program Files* only) ---
    for pat in [
        r"C:\Program Files*\Microsoft Visual Studio\20*\*\VC\Tools\MSVC\*\bin\Hostx64\x64",
        r"C:\Program Files*\Microsoft Visual Studio\20*\*\VC\Tools\MSVC\*\bin\HostX64\x64",
    ]:
        cl_dirs.extend(d for d in _glob.glob(pat) if Path(d, "cl.exe").exists())

    if not cl_dirs:
        return None

    # Deduplicate, sort by MSVC version number (newest first).
    # The version segment looks like "14.37.32822" in the path.
    def _msvc_ver(d: str) -> tuple[int, ...]:
        for part in Path(d).parts:
            segments = part.split(".")
            if len(segments) == 3 and segments[0] == "14":
                try:
                    return tuple(int(x) for x in segments)
                except ValueError:
                    pass
        return (0,)

    unique = sorted(set(cl_dirs), key=_msvc_ver, reverse=True)
    return str(Path(unique[0]) / "cl.exe")


def _ensure_msvc_on_path() -> None:
    """Configure the MSVC build environment for CUDA C++ compilation.

    Key problem: NVCC auto-detects the host C++ compiler via the Windows registry
    and may pick ``HostX86\\x64\\cl.exe`` (32-bit MSVC host) even when the correct
    ``Hostx64\\x64\\cl.exe`` is on PATH.  The 32-bit host exhausts its address
    space on large PyTorch headers → INTERNAL COMPILER ERROR / ACCESS_VIOLATION
    in cudafe++.  VS 2022 may also be installed on a non-C: drive, so a simple
    glob of ``C:\\Program Files*`` silently falls back to an older VS version.

    Fixes applied:
      * Use ``vswhere.exe`` to find the installation on any drive.
      * Search only ``Hostx64`` directories — never ``HostX86``.
      * Set ``CUDAHOSTCXX`` to override NVCC's own registry lookup.
      * Set ``DISTUTILS_USE_SDK`` / ``MSSdk`` so setuptools uses our environment.
    """
    if sys.platform != "win32":
        return

    cl_x64 = _find_hostx64_cl()

    if cl_x64 is None:
        import shutil

        cl_x64 = shutil.which("cl")  # last-resort: whatever is on PATH

    if cl_x64 is None:
        print("[build_kernels] WARNING: cl.exe not found. CUDA C++ builds may fail.")
        return

    cl_dir = str(Path(cl_x64).parent)
    os.environ["PATH"] = cl_dir + os.pathsep + os.environ.get("PATH", "")

    # Inject INCLUDE / LIB for the selected MSVC version.
    # Path structure: …\VC\Tools\MSVC\<ver>\bin\Hostx64\x64\cl.exe
    # cl_dir       = …\VC\Tools\MSVC\<ver>\bin\Hostx64\x64
    # msvc_root    = …\VC\Tools\MSVC\<ver>  (three levels up from cl_dir)
    msvc_root = Path(cl_dir).parent.parent.parent
    inc = str(msvc_root / "include")
    lib = str(msvc_root / "lib" / "x64")

    # Windows SDK — locate via registry (drive-agnostic), then pick highest version.
    sdk_incs: list[str] = []
    sdk_libs: list[str] = []
    _kits_root: Path | None = None
    try:
        import winreg

        for _hive in [winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER]:
            for _kp in [
                r"SOFTWARE\Microsoft\Windows Kits\Installed Roots",
                r"SOFTWARE\WOW6432Node\Microsoft\Windows Kits\Installed Roots",
            ]:
                try:
                    _k = winreg.OpenKey(_hive, _kp)
                    _val, _ = winreg.QueryValueEx(_k, "KitsRoot10")
                    _candidate = Path(_val)
                    if (_candidate / "Include").exists():
                        _kits_root = _candidate
                        raise StopIteration  # break both loops
                except (OSError, FileNotFoundError):
                    pass
    except StopIteration:
        pass
    except Exception:
        pass

    # Fallback: well-known hard-coded paths (C: and non-x86)
    if _kits_root is None:
        for _wk in [
            Path(r"C:\Program Files\Windows Kits\10"),
            Path(r"C:\Program Files (x86)\Windows Kits\10"),
        ]:
            if (_wk / "Include").exists():
                _kits_root = _wk
                break

    if _kits_root is not None:
        _sdk_candidates: list[Path] = sorted(
            (_kits_root / "Include").glob("10.*"), reverse=True
        )
        for sdk_ver_dir in _sdk_candidates[:1]:
            sdk_incs.append(str(sdk_ver_dir / "ucrt"))
            sdk_incs.append(str(sdk_ver_dir / "shared"))
            sdk_incs.append(str(sdk_ver_dir / "um"))
            _wk_lib = _kits_root / "Lib" / sdk_ver_dir.name
            sdk_libs.extend(
                [str(_wk_lib / "ucrt" / "x64"), str(_wk_lib / "um" / "x64")]
            )

    os.environ["INCLUDE"] = os.pathsep.join(
        [inc] + sdk_incs + [os.environ.get("INCLUDE", "")]
    )
    os.environ["LIB"] = os.pathsep.join([lib] + sdk_libs + [os.environ.get("LIB", "")])

    # Override NVCC's registry-based host-compiler detection (the registry may
    # point to a HostX86 binary even when Hostx64 is on PATH).
    os.environ["CUDAHOSTCXX"] = cl_x64
    # Tell setuptools to trust the SDK environment above.
    os.environ.setdefault("DISTUTILS_USE_SDK", "1")
    os.environ.setdefault("MSSdk", "1")

    print(f"[build_kernels] MSVC (Hostx64) : {cl_x64}")
    print(f"[build_kernels] CUDAHOSTCXX    : {cl_x64}")
    if sdk_incs:
        print(f"[build_kernels] Windows SDK inc: {sdk_incs[0]}")
    else:
        print(
            "[build_kernels] WARNING: Windows SDK not found — stddef.h may be missing."
        )


_ensure_msvc_on_path()

OUT_DIR = ROOT / "model_assets" / "custom_kernels"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Target GPU architectures (Turing → Blackwell).
# Each entry adds a separate PTX + SASS code object to the fat binary.
# sm_70 (Volta/V100) is NOT included — CUDA 12.9 deprecates offline compilation
# for sm_70 and older; building it triggers nvcc warnings and increases compile time.
# sm_75: RTX 2000   sm_80: A100   sm_86: RTX 3000   sm_89: RTX 4000
# sm_90: H100       sm_120: RTX 5000 (Blackwell — requires CUDA 12.8+)
# ---------------------------------------------------------------------------
ARCH_FLAGS = [
    "-gencode=arch=compute_75,code=sm_75",
    "-gencode=arch=compute_80,code=sm_80",
    "-gencode=arch=compute_86,code=sm_86",
    "-gencode=arch=compute_89,code=sm_89",
    "-gencode=arch=compute_90,code=sm_90",
    "-gencode=arch=compute_120,code=sm_120",  # Blackwell (RTX 50xx) — requires CUDA 12.8+
    # Forward compatibility: embed PTX for the highest supported arch so future
    # GPUs can JIT-compile from PTX at first use.
    "-gencode=arch=compute_120,code=compute_120",
]

COMMON_CUDA_FLAGS = [
    "--use_fast_math",
    "-O3",
    "-std=c++17",
    # Workaround for MSVC 14.37 Internal Compiler Error triggered by PyTorch 2.8
    # headers (TensorOptions.h / std::make_optional noexcept template chain) when
    # cl.exe is invoked as NVCC's host compiler.  /d2SSAOptimizer- disables the
    # specific SSA optimization pass that causes the ICE.
    "-Xcompiler",
    "/d2SSAOptimizer-",
] + ARCH_FLAGS


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _add_dll_dirs():
    if sys.platform == "win32":
        try:
            import torch as _t

            tlib = Path(_t.__file__).parent / "lib"
            if tlib.is_dir():
                os.add_dll_directory(str(tlib))
        except Exception:
            pass


def compile_ext(
    name: str, cuda_src: str, cpp_src: str, extra_ldflags: "list[str] | None" = None
) -> bool:
    """Compile a CUDA extension in a fresh subprocess and save the .pyd to OUT_DIR.

    Running each kernel in its own subprocess prevents CUDA 12.9 cicc.exe
    ACCESS_VIOLATION crashes that occur when two kernels are compiled back-to-back
    in the same Python process (process-memory / temp-file state pollution).
    """
    import subprocess
    import tempfile
    import json
    import textwrap

    pyd = OUT_DIR / f"{name}.pyd"
    build_dir = OUT_DIR / f"_build_{name}"
    build_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[build_kernels] Compiling {name} ...")
    print(f"  Target archs : {', '.join(f'sm_{a}' for a in [75, 80, 86, 89, 90, 120])}")
    print(f"  Output       : {pyd}")
    t0 = time.perf_counter()

    # Serialise sources and flags into a temp JSON so the subprocess can read them
    # without shell-escaping headaches.
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as fh:
        json.dump(
            {
                "name": name,
                "cuda_src": cuda_src,
                "cpp_src": cpp_src,
                "cuda_flags": COMMON_CUDA_FLAGS,
                "extra_ldflags": extra_ldflags or [],
                "build_dir": str(build_dir),
                "root": str(ROOT),
            },
            fh,
        )
        spec_path = fh.name

    worker_code = textwrap.dedent(r"""
        import sys, os, json, pathlib, glob, shutil, hashlib
        spec_path = sys.argv[1]
        with open(spec_path, encoding="utf-8") as f:
            spec = json.load(f)

        build_dir = pathlib.Path(spec["build_dir"])
        name      = spec["name"]
        dst       = str(build_dir.parent / (name + ".pyd"))

        # ------------------------------------------------------------------
        # Fast path: if the source + flags hash matches the last successful
        # build, skip load_inline entirely and just copy the cached .pyd.
        # This avoids:
        #   1. Unnecessary MSVC/NVCC recompilation on repeated runs.
        #   2. PyTorch 2.8 load_inline importing the existing .pyd inside the
        #      subprocess (which occasionally causes a STATUS_ACCESS_VIOLATION
        #      in the CUDA extension-loading machinery on Windows).
        # ------------------------------------------------------------------
        src_hash = hashlib.sha256(
            (spec["cuda_src"] + spec["cpp_src"]
             + repr(spec["cuda_flags"]) + repr(spec["extra_ldflags"])).encode()
        ).hexdigest()
        hash_file   = build_dir / ".build_hash"
        cached_pyds = sorted(build_dir.glob(name + "*.pyd"))
        if cached_pyds and hash_file.exists() and hash_file.read_text().strip() == src_hash:
            pyd = str(cached_pyds[0])
            shutil.copy2(pyd, dst)
            print(f"Copied {pyd} -> {dst} (cached, no rebuild needed)", flush=True)
            sys.exit(0)

        # ------------------------------------------------------------------
        # Slow path: source changed (or first build) — compile with load_inline.
        # Delete any stale .pyd first so load_inline won't try to import it
        # before recompilation completes (that import is what crashes the
        # subprocess with 0xC0000005 when a stale binary is present).
        # ------------------------------------------------------------------
        for old in cached_pyds:
            old.unlink(missing_ok=True)
        hash_file.unlink(missing_ok=True)

        sys.path.insert(0, spec["root"])
        from torch.utils.cpp_extension import load_inline
        # If cuda_src is empty, compile purely as C++ (skip NVCC entirely).
        # This avoids cicc.exe ACCESS_VIOLATION crashes on sm_120 when the source
        # contains complex STL headers but no actual CUDA device code.
        cuda_srcs = [spec["cuda_src"]] if spec.get("cuda_src") else []
        # /d2SSAOptimizer- suppresses the MSVC 14.37 SSA-optimizer ICE (C1001) that
        # fires on pybind11/PyTorch headers.  For CUDA builds this is already passed
        # via -Xcompiler inside extra_cuda_cflags; for pure-C++ builds we must add it
        # directly to extra_cflags so MSVC receives it.
        cpp_cflags = ["-O2", "/d2SSAOptimizer-"] if sys.platform == "win32" else ["-O2"]
        ldflags = spec.get("extra_ldflags") or []
        if not cuda_srcs:
            # No CUDA device code -- inject CUDA toolkit include/lib paths so
            # MSVC/GCC can resolve <cuda_fp16.h>, <cublasLt.h>, etc. directly.
            try:
                import os as _os
                from torch.utils.cpp_extension import CUDA_HOME
                if CUDA_HOME:
                    cpp_cflags.append("-I" + _os.path.join(CUDA_HOME, "include"))
                    if sys.platform == "win32":
                        cuda_lib = _os.path.join(CUDA_HOME, "lib", "x64")
                        ldflags = ["/LIBPATH:" + cuda_lib] + ldflags
                    else:
                        cuda_lib = _os.path.join(CUDA_HOME, "lib64")
                        ldflags = ["-L" + cuda_lib] + ldflags
                else:
                    print("  Warning: CUDA_HOME not set; cuda_fp16.h may not be found", flush=True)
            except Exception as _e:
                print("  Warning: CUDA path injection failed: " + str(_e), flush=True)
        load_inline(
            name=name,
            cuda_sources=cuda_srcs,
            cpp_sources=[spec["cpp_src"]],
            extra_cuda_cflags=spec["cuda_flags"] if cuda_srcs else [],
            extra_cflags=cpp_cflags,
            extra_ldflags=ldflags,
            build_directory=str(build_dir),
            verbose=True,
        )
        built = sorted(build_dir.glob(name + "*.pyd"))
        if built:
            shutil.copy2(str(built[0]), dst)
            print(f"Copied {built[0]} -> {dst}", flush=True)
            # Save hash so the next run skips recompilation.
            hash_file.write_text(src_hash)
    """).strip()

    # Write worker to a temp .py file (avoids -c command-line length limits on Windows)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as fw:
        fw.write("# -*- coding: utf-8 -*-\n" + worker_code)
        worker_path = fw.name

    try:
        result = subprocess.run(
            [sys.executable, worker_path, spec_path],
            env=os.environ.copy(),
            timeout=480,
        )
        Path(spec_path).unlink(missing_ok=True)
        Path(worker_path).unlink(missing_ok=True)
        if result.returncode != 0:
            print(f"  FAILED: subprocess exited {result.returncode}")
            return False
        elapsed = time.perf_counter() - t0
        print(f"  Done in {elapsed:.1f}s  ->  {pyd}")
        return True
    except Exception as e:
        Path(spec_path).unlink(missing_ok=True)
        Path(worker_path).unlink(missing_ok=True)
        print(f"  FAILED: {e}")
        return False


# ---------------------------------------------------------------------------
# Kernel 1 — fused AdaIN (InSwapper)
# ---------------------------------------------------------------------------
_ADAIN_CUDA_SRC = r"""
#include <cuda_fp16.h>
#include <cuda_runtime.h>

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

    float t_mean = 0.f, t_M2 = 0.f;
    int   t_cnt  = 0;

    // Vectorised 2-at-a-time Welford using __ldg for both __half2 elements.
    // Avoid reinterpret_cast<__half2*> which triggers a CUDA 12.9 cicc
    // code-gen crash when targeting sm_120 (Blackwell).
    int HW2 = HW >> 1;
    for (int i = threadIdx.x; i < HW2; i += BLOCK) {
        __half a_h = __ldg(xc + 2 * i);
        __half b_h = __ldg(xc + 2 * i + 1);
        float a = __half2float(a_h);
        float b = __half2float(b_h);
        ++t_cnt; float da = a - t_mean; t_mean += da / (float)t_cnt; t_M2 += da * (a - t_mean);
        ++t_cnt; float db = b - t_mean; t_mean += db / (float)t_cnt; t_M2 += db * (b - t_mean);
    }
    if ((HW & 1) && ((HW - 1) % BLOCK == (int)threadIdx.x)) {
        float v = __half2float(__ldg(xc + HW - 1));
        ++t_cnt; float dv = v - t_mean; t_mean += dv / (float)t_cnt; t_M2 += dv * (v - t_mean);
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float mb  = __shfl_down_sync(0xffffffff, t_mean, offset);
        float M2b = __shfl_down_sync(0xffffffff, t_M2,   offset);
        int   nb  = __shfl_down_sync(0xffffffff, t_cnt,  offset);
        if (lane + offset < 32)
            welford_merge(t_mean, t_M2, t_cnt, mb, M2b, nb);
    }

    __shared__ float s_mean[NWARPS];
    __shared__ float s_M2  [NWARPS];
    __shared__ int   s_cnt [NWARPS];
    if (lane == 0) { s_mean[warp] = t_mean; s_M2[warp] = t_M2; s_cnt[warp] = t_cnt; }
    __syncthreads();

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

    float mean    = s_mean[0];
    float var     = s_M2[0] / (float)HW;
    float inv_std = rsqrtf(var + eps);
    float sc      = __half2float(scale[c]) * inv_std;
    float bi      = __half2float(bias [c]) - sc * mean;

    // Write-back: avoid reinterpret_cast<__half2*> (same sm_120 cicc workaround).
    for (int i = threadIdx.x; i < HW2; i += BLOCK) {
        float a = __half2float(__ldg(xc + 2 * i))     * sc + bi;
        float b = __half2float(__ldg(xc + 2 * i + 1)) * sc + bi;
        yc[2 * i]     = __float2half(a);
        yc[2 * i + 1] = __float2half(b);
    }
    if ((HW & 1) && ((HW - 1) % BLOCK == (int)threadIdx.x))
        yc[HW - 1] = __float2half(__half2float(__ldg(xc + HW - 1)) * sc + bi);
}

extern "C" void adain_fp16_launch(
        const __half* x, const __half* scale, const __half* bias, __half* y,
        int C, int HW, float eps, cudaStream_t stream)
{
    fused_adain_fp16_kernel_v2<256><<<C, 256, 0, stream>>>(x, scale, bias, y, HW, eps);
}
"""

_ADAIN_CPP_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

extern "C" void adain_fp16_launch(
        const void* x, const void* scale, const void* bias, void* y,
        int C, int HW, float eps, cudaStream_t stream);

torch::Tensor fused_adain_fp16(
        torch::Tensor x,
        torch::Tensor scale,
        torch::Tensor bias,
        float eps)
{
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kHalf);
    TORCH_CHECK(scale.is_cuda() && scale.scalar_type() == torch::kHalf);
    TORCH_CHECK(bias.is_cuda()  && bias.scalar_type()  == torch::kHalf);
    auto y = torch::empty_like(x);
    adain_fp16_launch(
        x.data_ptr(), scale.data_ptr(), bias.data_ptr(), y.data_ptr(),
        (int)x.size(0), (int)x.size(1), eps,
        at::cuda::getCurrentCUDAStream());
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_adain_fp16", &fused_adain_fp16,
          "Fused Adaptive Instance Normalisation (Welford, __half2, warp-shuffle)");
}
"""


# ---------------------------------------------------------------------------
# Kernel 2 — fused weight demodulation (GFPGAN / GPEN shared)
# ---------------------------------------------------------------------------
_DEMOD_CUDA_SRC = r"""
#include <cuda_runtime.h>
#include <math.h>

__device__ __forceinline__ float warp_reduce(float v) {
    for (int d = 16; d > 0; d >>= 1) v += __shfl_down_sync(0xffffffff, v, d);
    return v;
}

// Templatized on BLOCK so shared-memory sizes and warp-count are compile-time
// constants.  This avoids a CUDA 12.9 cicc code-gen crash on sm_120 that is
// triggered by runtime-variable shared-array indexing (blockDim.x / 32).
template<int BLOCK>
__global__ void fused_demod_kernel(
    const float* __restrict__ weight,
    const float* __restrict__ style,
    float*       __restrict__ out,
    int C_out, int n, int kHkW, float eps)
{
    const int NWARPS = BLOCK / 32;
    int co   = blockIdx.x;
    if (co >= C_out) return;
    const float* w = weight + (long long)co * n;
    float*       o = out    + (long long)co * n;

    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;

    float sumsq = 0.f;
    for (int i = threadIdx.x; i < n; i += BLOCK) {
        float ws = w[i] * style[i / kHkW];
        sumsq += ws * ws;
    }
    sumsq = warp_reduce(sumsq);

    __shared__ float smem[NWARPS];
    if (lane == 0) smem[warp] = sumsq;
    __syncthreads();

    if (warp == 0) {
        sumsq = (lane < NWARPS) ? smem[lane] : 0.f;
        sumsq = warp_reduce(sumsq);
    }

    __shared__ float demod_s;
    if (threadIdx.x == 0) demod_s = rsqrtf(sumsq + eps);
    __syncthreads();
    float demod = demod_s;

    for (int i = threadIdx.x; i < n; i += BLOCK)
        o[i] = w[i] * style[i / kHkW] * demod;
}

extern "C" void fused_demod_launch(
    const float* weight, const float* style, float* out,
    int C_out, int n, int kHkW, float eps, cudaStream_t stream)
{
    fused_demod_kernel<256><<<C_out, 256, 0, stream>>>(weight, style, out, C_out, n, kHkW, eps);
}
"""

_DEMOD_CPP_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

extern "C" void fused_demod_launch(
    const float* weight, const float* style, float* out,
    int C_out, int n, int kHkW, float eps, cudaStream_t stream);

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
    fused_demod_launch(
        weight.data_ptr<float>(), style.data_ptr<float>(),
        out.data_ptr<float>(), C_out, C_in * kHkW, kHkW, eps,
        at::cuda::getCurrentCUDAStream());
    return out;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_demod", &fused_demod, "Fused weight demodulation (warp-shuffle)");
}
"""


# ---------------------------------------------------------------------------
# Triton warm-up
# ---------------------------------------------------------------------------
def warmup_triton():
    """Trigger Triton JIT compilation for current GPU, populate triton_cache/.

    Runs in a subprocess to isolate native crashes (e.g. 0xC0000005 access
    violation inside libtriton.pyd MLIR passes).  Python try/except cannot
    catch native process crashes, so isolation is the only reliable guard.
    """
    print("\n[build_kernels] Warming up Triton kernels for current GPU ...")

    import subprocess
    import tempfile
    import textwrap

    worker_code = textwrap.dedent(r"""
        import sys, os
        sys.path.insert(0, sys.argv[1])  # project root

        import torch
        from custom_kernels.triton_ops import (
            TRITON_AVAILABLE,
            triton_demod,
            triton_fused_gpen_act,
            triton_fused_gfpgan_act,
            triton_adain,
            triton_group_norm_silu,
            triton_rmsnormmax,
            triton_pixel_shift_extract,
            triton_pixel_shift_insert,
            triton_im2col_reflect,
            _TRITON_CACHE,
        )

        if not TRITON_AVAILABLE:
            print("  Triton not available — skipping warm-up.", flush=True)
            sys.exit(0)

        dev = "cuda"
        dtype = torch.float16

        # demod — GFPGAN / GPEN weight demodulation
        w = torch.randn(256, 256, 3, 3, dtype=dtype, device=dev)
        s = torch.randn(256, device=dev)
        triton_demod(w, s)

        # gpen_act — GPEN fused noise-inject + activate
        c = torch.randn(1, 128, 32, 32, dtype=dtype, device=dev)
        n = torch.randn(1, 128, 32, 32, dtype=dtype, device=dev)
        b = torch.randn(1, 256, 1, 1, dtype=dtype, device=dev)
        triton_fused_gpen_act(c, n, b)

        # gfpgan_act — GFPGAN fused activate (with + without noise)
        x = torch.randn(1, 128, 32, 32, dtype=dtype, device=dev)
        bias = torch.randn(1, 128, 1, 1, dtype=dtype, device=dev)
        triton_fused_gfpgan_act(x, n[:, :128], bias)
        triton_fused_gfpgan_act(x, None, bias)

        # adain — InSwapper Adaptive Instance Normalization (B=1 and batched B>1)
        xa1 = torch.randn(1, 1024, 32, 32, dtype=dtype, device=dev)
        sa = torch.randn(1, 1024, 1, 1, dtype=dtype, device=dev)
        triton_adain(xa1, sa, sa)
        xa4 = torch.randn(4, 1024, 32, 32, dtype=dtype, device=dev)
        triton_adain(xa4, sa, sa)
        xa9 = torch.randn(9, 1024, 32, 32, dtype=dtype, device=dev)
        triton_adain(xa9, sa, sa)
        xa16 = torch.randn(16, 1024, 32, 32, dtype=dtype, device=dev)
        triton_adain(xa16, sa, sa)

        # im2col_reflect — fused reflection padding + im2col
        for B_im2col in [1, 4, 16]:
            x_im = torch.randn(B_im2col, 1024, 32, 32, dtype=dtype, device=dev)
            triton_im2col_reflect(x_im, k=3, pad=1)

        # pixel_shift_extract / insert — InSwapper batched resolution tiles
        for dim in [2, 3, 4]:
            H = dim * 128
            img_hwc = torch.randn(H, H, 3, dtype=torch.float32, device=dev)
            tiles = triton_pixel_shift_extract(img_hwc, dim)
            triton_pixel_shift_insert(tiles, img_hwc, dim)

        # group_norm_silu — ReF-LDM VAE / UNet GroupNorm + optional SiLU
        for C, H in [(128, 32), (256, 16), (512, 8)]:
            xg = torch.randn(1, C, H, H, dtype=dtype, device=dev)
            wg = torch.randn(C, dtype=dtype, device=dev)
            bg = torch.randn(C, dtype=dtype, device=dev)
            triton_group_norm_silu(xg, wg, bg, num_groups=32, fuse_silu=False)
            triton_group_norm_silu(xg, wg, bg, num_groups=32, fuse_silu=True)

        # rmsnormmax — XSeg per-channel RMS norm + affine + max-floor
        for C, H in [(32, 256), (64, 128), (128, 64), (256, 32), (256, 16), (256, 8)]:
            xn = torch.randn(1, C, H, H, dtype=dtype, device=dev)
            gamma = torch.randn(1, C, 1, 1, dtype=dtype, device=dev)
            beta = torch.randn(1, C, 1, 1, dtype=dtype, device=dev)
            maxval = torch.randn(1, C, 1, 1, dtype=dtype, device=dev)
            triton_rmsnormmax(xn, gamma, beta, maxval, eps=0.5)

        print("  Triton warm-up complete. Kernels cached in:", flush=True)
        print(f"    {_TRITON_CACHE}", flush=True)
    """).strip()

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as fw:
        fw.write("# -*- coding: utf-8 -*-\n" + worker_code)
        worker_path = fw.name

    try:
        result = subprocess.run(
            [sys.executable, worker_path, str(ROOT)],
            env=os.environ.copy(),
            timeout=300,
        )
        if result.returncode != 0:
            print(
                f"  WARNING: Triton warm-up subprocess exited with code "
                f"{result.returncode} (possible native crash in libtriton.pyd).\n"
                "  Kernels will be JIT-compiled on first inference instead."
            )
        else:
            pass  # success messages already printed by subprocess
    except subprocess.TimeoutExpired:
        print("  WARNING: Triton warm-up timed out — skipping.")
    except Exception as e:
        print(f"  WARNING: Triton warm-up failed: {e}")
    finally:
        Path(worker_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    import torch

    cuda_ver = torch.version.cuda or "unknown"
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    print(f"\n{'=' * 60}")
    print("VisoMaster-Fusion — Custom CUDA Kernel Builder")
    print(f"{'=' * 60}")
    print(f"  CUDA        : {cuda_ver}")
    print(f"  GPU         : {gpu_name}")
    print(f"  PyTorch     : {torch.__version__}")
    print(f"  Output dir  : {OUT_DIR}")
    print("  Arch targets: sm_75 sm_80 sm_86 sm_89 sm_90 sm_120")
    print()

    # Import cuBLASLt sources from inswapper_torch (single source of truth)
    try:
        from custom_kernels.inswapper_128.inswapper_torch import (
            _STYLEBLOCK_CUDA_SRC,
            _STYLEBLOCK_CPP_SRC,
        )

        _have_styleblock_src = True
    except Exception as e:
        print(f"[build_kernels] Could not import style_block sources: {e}")
        _have_styleblock_src = False

    results = {}

    results["adain_fp16_ext"] = compile_ext(
        "adain_fp16_ext", _ADAIN_CUDA_SRC, _ADAIN_CPP_SRC
    )
    results["gfpgan_demod_ext"] = compile_ext(
        "gfpgan_demod_ext", _DEMOD_CUDA_SRC, _DEMOD_CPP_SRC
    )

    if _have_styleblock_src:
        # When compiled as C++ (no cuda_sources), PyTorch does not auto-link the
        # CUDA runtime or CUDA-aware torch libs.  Add them explicitly.
        ldflags = (
            ["cublasLt.lib", "cudart.lib", "c10_cuda.lib", "torch_cuda.lib"]
            if sys.platform == "win32"
            else ["-lcublasLt", "-lcudart", "-lc10_cuda", "-ltorch_cuda"]
        )
        # style_block has no CUDA device code (no __global__ kernels) — compile
        # as C++ via MSVC/GCC to bypass NVCC cicc.exe, which crashes with
        # STATUS_ACCESS_VIOLATION on sm_120 when parsing complex STL headers
        # (<mutex>, <unordered_map>, <stdexcept>) during device-code extraction.
        # Include-order fix: on Windows, <cublasLt.h> pulls in <windows.h> before
        # <torch/extension.h>, and the resulting macros (min/max, etc.) corrupt
        # PyTorch C++ template instantiation in cloneable.h.  Prepend a preamble
        # that defines NOMINMAX and includes <torch/extension.h> first so that the
        # later duplicate includes are no-ops (include guards).
        style_preamble = (
            "#define NOMINMAX\n"
            "#include <torch/extension.h>\n"
            "#include <ATen/cuda/CUDAContext.h>\n"
        )
        combined_cpp = (
            style_preamble + _STYLEBLOCK_CUDA_SRC + "\n" + _STYLEBLOCK_CPP_SRC
        )
        results["style_block_ext"] = compile_ext(
            "style_block_ext",
            "",  # empty → no CUDA device code, skip NVCC
            combined_cpp,
            extra_ldflags=ldflags,
        )
    else:
        print("\n[build_kernels] Skipping style_block_ext (sources unavailable).")
        results["style_block_ext"] = False

    warmup_triton()

    print(f"\n{'=' * 60}")
    print("Build summary:")
    for name, ok in results.items():
        status = "OK " if ok else "FAIL"
        pyd = OUT_DIR / f"{name}.pyd"
        size_kb = f"{pyd.stat().st_size // 1024} KB" if pyd.exists() else "N/A"
        print(f"  [{status}] {name}.pyd  ({size_kb})")
    print(f"{'=' * 60}")

    # style_block_ext failure is non-fatal — falls back to torch.mm GEMM (Tier 8)
    critical = {k: v for k, v in results.items() if k != "style_block_ext"}
    if not all(critical.values()):
        print("\nSome kernels failed to compile. Check MSVC / CUDA toolkit setup.")
        print("The application will fall back to Triton or pure-PyTorch ops.")
        sys.exit(1)
    elif not results.get("style_block_ext"):
        print("\nCritical kernels built. style_block_ext failed — InSwapper will use")
        print("torch.mm GEMM (Tier 8) instead of cuBLASLt (Phase 3).")
        print(
            "Multi-arch fat binaries work on any NVIDIA GPU (Volta through Blackwell)."
        )
        print("Note: sm_120 (RTX 50xx) requires CUDA Toolkit 12.8+.")
    else:
        print("\nAll kernels built successfully.")
        print(
            "Multi-arch fat binaries work on any NVIDIA GPU (Volta through Blackwell)."
        )
        print("Note: sm_120 (RTX 50xx) requires CUDA Toolkit 12.8+.")


if __name__ == "__main__":
    main()
