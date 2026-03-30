"""
Benchmark GFPGANv1.4 — 4-tier latency comparison.

Run via:
    custom_kernels/gfpgan_v1_4/run_with_msvc.bat custom_kernels/gfpgan_v1_4/benchmark_gfpgan.py
  or plain Python (Tier 3 disabled if MSVC unavailable):
    .venv/Scripts/python custom_kernels/gfpgan_v1_4/benchmark_gfpgan.py
"""

# ── TensorRT DLL discovery (must be before onnxruntime import) ────────────
import os as _os
import sys as _sys
from pathlib import Path as _Path

_REPO_ROOT = _Path(__file__).resolve().parents[2]
for _candidate in [
    _REPO_ROOT / ".venv" / "Lib" / "site-packages" / "tensorrt_libs",
    _REPO_ROOT / ".venv" / "Lib" / "site-packages" / "nvidia" / "cuda_runtime" / "bin",
]:
    if _candidate.exists():
        _os.environ["PATH"] = (
            str(_candidate) + _os.pathsep + _os.environ.get("PATH", "")
        )
del _candidate, _REPO_ROOT, _os, _sys, _Path

import sys as _sys_enc
if hasattr(_sys_enc.stdout, 'reconfigure'):
    _sys_enc.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(_sys_enc.stderr, 'reconfigure'):
    _sys_enc.stderr.reconfigure(encoding='utf-8', errors='replace')
del _sys_enc
# ──────────────────────────────────────────────────────────────────────────

import sys
import time
import os
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
MODEL_PATH = str(ROOT / "model_assets" / "GFPGANv1.4.onnx")
WARMUP = int(os.environ.get("WARMUP", 10))
RUNS = int(os.environ.get("ITERS", 50))

import numpy as np  # noqa: E402
import torch  # noqa: E402


def ms(t0, t1, n):
    return (t1 - t0) / n * 1000


def run_ort():
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(
        MODEL_PATH,
        so,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        provider_options=[{"device_id": "0"}, {}],
    )
    inp = np.random.default_rng(0).random((1, 3, 512, 512)).astype(np.float32)
    for _ in range(WARMUP):
        sess.run(None, {"input": inp})
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(RUNS):
        sess.run(None, {"input": inp})
    torch.cuda.synchronize()
    return ms(t0, time.perf_counter(), RUNS)


def run_ort_trt():
    """ORT TensorRT EP — matches the application's provider config.
    Always attempts to build/load engine using a temporary cache dir.
    Returns None and prints a reason if TRT EP is unavailable."""
    import tempfile
    import shutil
    import onnxruntime as ort

    if "TensorrtExecutionProvider" not in ort.get_available_providers():
        print("  SKIP: TensorrtExecutionProvider not available")
        return None
    _trt_tmp = tempfile.mkdtemp(prefix="ort_trt_bench_")
    try:
        trt_opts = {
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": _trt_tmp,
            "trt_timing_cache_enable": True,
            "trt_timing_cache_path": _trt_tmp,
            "trt_layer_norm_fp32_fallback": True,
            "trt_max_workspace_size": 8589934592,
            "trt_builder_optimization_level": 5,
        }
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        print("  Building TRT engine (first run may take 1-5 min)...")
        sess = ort.InferenceSession(
            MODEL_PATH,
            so,
            providers=[
                ("TensorrtExecutionProvider", trt_opts),
                ("CUDAExecutionProvider", {"device_id": "0"}),
                ("CPUExecutionProvider", {}),
            ],
        )
        inp = np.random.default_rng(0).random((1, 3, 512, 512)).astype(np.float32)
        for _ in range(WARMUP):
            sess.run(None, {"input": inp})
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(RUNS):
            sess.run(None, {"input": inp})
        torch.cuda.synchronize()
        return ms(t0, time.perf_counter(), RUNS)
    except Exception as e:
        print(f"  FAILED: {e}")
        return None
    finally:
        shutil.rmtree(_trt_tmp, ignore_errors=True)


def run_pytorch(fp16: bool):
    from custom_kernels.gfpgan_v1_4.gfpgan_torch import GFPGANTorch

    dtype = torch.float16 if fp16 else torch.float32
    model = GFPGANTorch.from_onnx(MODEL_PATH, compute_dtype=dtype).cuda().eval()
    inp = torch.randn(1, 3, 512, 512, device="cuda")
    with torch.no_grad():
        for _ in range(WARMUP):
            model(inp)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(RUNS):
            model(inp)
    torch.cuda.synchronize()
    return ms(t0, time.perf_counter(), RUNS)


def run_cuda_graph():
    from custom_kernels.gfpgan_v1_4.gfpgan_torch import (
        GFPGANTorch,
        build_cuda_graph_runner,
    )

    model = GFPGANTorch.from_onnx(MODEL_PATH).cuda().eval()
    inp = torch.randn(1, 3, 512, 512, device="cuda")
    with torch.no_grad():
        runner = build_cuda_graph_runner(model, inp_shape=(1, 3, 512, 512))
    for _ in range(WARMUP):
        runner(inp)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(RUNS):
        runner(inp)
    torch.cuda.synchronize()
    return ms(t0, time.perf_counter(), RUNS)


def run_compile_cuda_graph():
    from custom_kernels.gfpgan_v1_4.gfpgan_torch import (
        GFPGANTorch,
        build_cuda_graph_runner,
    )

    model = GFPGANTorch.from_onnx(MODEL_PATH).cuda().eval()
    inp = torch.randn(1, 3, 512, 512, device="cuda")
    with torch.no_grad():
        runner = build_cuda_graph_runner(
            model, inp_shape=(1, 3, 512, 512), torch_compile=True
        )
    for _ in range(WARMUP):
        runner(inp)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(RUNS):
        runner(inp)
    torch.cuda.synchronize()
    return ms(t0, time.perf_counter(), RUNS)


def main():
    print(f"\n=== GFPGANv1.4 Benchmark ({RUNS} runs, RTX GPU) ===\n")
    results = []

    print("Tier 0: ORT FP32 CUDA EP ...")
    t = run_ort()
    results.append(("ORT FP32 CUDA EP", t))
    print(f"  {t:.2f} ms")

    print("Tier 0b: ORT TensorRT EP (app default) ...")
    t_trt = run_ort_trt()
    if t_trt is not None:
        results.append(("ORT TensorRT EP (app default)", t_trt))
        print(f"  {t_trt:.2f} ms")

    print("Tier 1: PyTorch FP32 ...")
    t = run_pytorch(fp16=False)
    results.append(("PyTorch FP32", t))
    print(f"  {t:.2f} ms")

    print("Tier 2: PyTorch FP16 + Triton demod + Triton fused-act ...")
    t = run_pytorch(fp16=True)
    results.append(("PyTorch FP16 + Triton demod", t))
    print(f"  {t:.2f} ms")

    print("Tier 3: PyTorch FP16 + Triton demod + CUDA graph ...")
    try:
        t = run_cuda_graph()
        results.append(("PyTorch FP16 + Triton + CUDAGraph", t))
        print(f"  {t:.2f} ms")
    except Exception as e:
        print(f"  SKIPPED: {e}")

    print("Tier 4: PyTorch FP16 + Triton demod + torch.compile + CUDA graph ...")
    try:
        t = run_compile_cuda_graph()
        results.append(("PT FP16 + Triton + compile + CUDAGraph", t))
        print(f"  {t:.2f} ms")
    except Exception as e:
        print(f"  SKIPPED: {e}")

    print("Tier 4b: torch.compile(reduce-overhead) — no extra CUDA graph ...")
    if not int(os.environ.get("GFPGAN_TORCH_COMPILE", "0")):
        print("  Skipped — reduce-overhead may crash (MLIR segfault) on this model on Windows/sm_89.")
        print("  Set GFPGAN_TORCH_COMPILE=1 to attempt.")
    else:
        try:
            from custom_kernels.gfpgan_v1_4.gfpgan_torch import GFPGANTorch
            from custom_kernels.compile_utils import apply_torch_compile
            import time as _time
            _m4b = GFPGANTorch.from_onnx(MODEL_PATH, compute_dtype=torch.float16).cuda().eval()
            _inp4b = torch.randn(1, 3, 512, 512, device="cuda")
            _m4b_compiled = apply_torch_compile(_m4b, _inp4b, compile_mode="reduce-overhead")
            with torch.no_grad():
                for _ in range(WARMUP):
                    _m4b_compiled(_inp4b)
            torch.cuda.synchronize()
            _t0_4b = _time.perf_counter()
            with torch.no_grad():
                for _ in range(RUNS):
                    _m4b_compiled(_inp4b)
            torch.cuda.synchronize()
            t_4b = (_time.perf_counter() - _t0_4b) / RUNS * 1000
            results.append(("torch.compile reduce-overhead (no CUDA graph)", t_4b))
            print(f"  {t_4b:.2f} ms")
        except Exception as e:
            print(f"  SKIPPED: {e}")
            import traceback; traceback.print_exc()

    cuda_ep_t = results[0][1]
    trt_ep_t = t_trt  # may be None if TRT not available
    hdr = f"{'Method':<38} {'ms':>8} {'vs CUDA EP':>11}"
    if trt_ep_t:
        hdr += f" {'vs TRT EP':>10}"
    print(f"\n{hdr}")
    print("-" * (len(hdr)))
    for name, t in results:
        row = f"  {name:<38} {t:>8.2f} {cuda_ep_t / t:>10.2f}x"
        if trt_ep_t:
            row += f" {trt_ep_t / t:>9.2f}x"
        print(row)

    out = Path(__file__).parent / "benchmark_results.txt"
    with open(out, "w") as f:
        for name, t in results:
            f.write(f"{name}: {t:.2f} ms  (vs CUDA EP: {cuda_ep_t / t:.2f}x")
            if trt_ep_t:
                f.write(f", vs TRT EP: {trt_ep_t / t:.2f}x")
            f.write(")\n")
    print(f"\n[Saved -> {out}]")


if __name__ == "__main__":
    main()
