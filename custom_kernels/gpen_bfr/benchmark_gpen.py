"""
Benchmark all four GPEN-BFR models — 4-tier latency comparison.

Run via:
    custom_kernels/gpen_bfr/run_with_msvc.bat custom_kernels/gpen_bfr/benchmark_gpen.py
  or plain Python:
    .venv/Scripts/python custom_kernels/gpen_bfr/benchmark_gpen.py [256|512|1024|2048]
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

MODELS = {
    256: "GPEN-BFR-256.onnx",
    512: "GPEN-BFR-512.onnx",
    1024: "GPEN-BFR-1024.onnx",
    2048: "GPEN-BFR-2048.onnx",
}

WARMUP = int(os.environ.get("WARMUP", 10))
RUNS = int(os.environ.get("ITERS", 50))


def ms(t0, t1, n):
    return (t1 - t0) / n * 1000


def run_ort(model_path, inp_hw):
    import numpy as np
    import onnxruntime as ort
    import torch

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(
        model_path,
        so,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        provider_options=[{"device_id": "0"}, {}],
    )
    inp_name = sess.get_inputs()[0].name
    inp = np.random.default_rng(0).random((1, 3, inp_hw, inp_hw)).astype("float32")
    for _ in range(WARMUP):
        sess.run(None, {inp_name: inp})
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(RUNS):
        sess.run(None, {inp_name: inp})
    torch.cuda.synchronize()
    return ms(t0, time.perf_counter(), RUNS)


def run_ort_trt(model_path, inp_hw, model_stem):
    import tempfile
    import shutil
    import numpy as np
    import onnxruntime as ort
    import torch

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
            model_path,
            so,
            providers=[
                ("TensorrtExecutionProvider", trt_opts),
                ("CUDAExecutionProvider", {"device_id": "0"}),
                ("CPUExecutionProvider", {}),
            ],
        )
        inp_name = sess.get_inputs()[0].name
        inp = np.random.default_rng(0).random((1, 3, inp_hw, inp_hw)).astype("float32")
        for _ in range(WARMUP):
            sess.run(None, {inp_name: inp})
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(RUNS):
            sess.run(None, {inp_name: inp})
        torch.cuda.synchronize()
        return ms(t0, time.perf_counter(), RUNS)
    except Exception as e:
        print(f"  FAILED: {e}")
        return None
    finally:
        shutil.rmtree(_trt_tmp, ignore_errors=True)


def run_pytorch(model_path, inp_hw, fp16: bool):
    import torch
    from custom_kernels.gpen_bfr.gpen_torch import GPENTorch

    dtype = torch.float16 if fp16 else torch.float32
    model = GPENTorch.from_onnx(model_path, compute_dtype=dtype).cuda().eval()
    inp = torch.randn(1, 3, inp_hw, inp_hw, device="cuda")
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


def run_cuda_graph(model_path, inp_hw):
    import torch
    from custom_kernels.gpen_bfr.gpen_torch import GPENTorch, build_cuda_graph_runner

    model = GPENTorch.from_onnx(model_path, compute_dtype=torch.float16).cuda().eval()
    inp = torch.randn(1, 3, inp_hw, inp_hw, device="cuda")
    with torch.no_grad():
        runner = build_cuda_graph_runner(model, inp_shape=(1, 3, inp_hw, inp_hw))
    for _ in range(WARMUP):
        runner(inp)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(RUNS):
        runner(inp)
    torch.cuda.synchronize()
    return ms(t0, time.perf_counter(), RUNS)


def run_compile_cuda_graph(model_path, inp_hw):
    import torch
    from custom_kernels.gpen_bfr.gpen_torch import GPENTorch, build_cuda_graph_runner

    model = GPENTorch.from_onnx(model_path, compute_dtype=torch.float16).cuda().eval()
    inp = torch.randn(1, 3, inp_hw, inp_hw, device="cuda")
    with torch.no_grad():
        runner = build_cuda_graph_runner(
            model, inp_shape=(1, 3, inp_hw, inp_hw), torch_compile=True
        )
    for _ in range(WARMUP):
        runner(inp)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(RUNS):
        runner(inp)
    torch.cuda.synchronize()
    return ms(t0, time.perf_counter(), RUNS)


def benchmark_one(size: int):
    model_name = MODELS[size]
    model_path = str(ROOT / "model_assets" / model_name)
    model_stem = Path(model_name).stem

    print(f"\n=== GPEN-BFR-{size} Benchmark ({RUNS} runs) ===\n")

    # Determine input size from ONNX
    import onnx

    m = onnx.load(model_path)
    in_dims = [d.dim_value for d in m.graph.input[0].type.tensor_type.shape.dim]
    inp_hw = in_dims[2] if len(in_dims) >= 3 else size

    results = []

    print("Tier 0: ORT FP32 CUDA EP ...")
    t = run_ort(model_path, inp_hw)
    results.append(("ORT FP32 CUDA EP", t))
    print(f"  {t:.2f} ms")

    print("Tier 0b: ORT TensorRT EP (app default) ...")
    t_trt = run_ort_trt(model_path, inp_hw, model_stem)
    if t_trt is not None:
        results.append(("ORT TensorRT EP (app default)", t_trt))
        print(f"  {t_trt:.2f} ms")

    print("Tier 1: PyTorch FP32 ...")
    t = run_pytorch(model_path, inp_hw, fp16=False)
    results.append(("PyTorch FP32", t))
    print(f"  {t:.2f} ms")

    print("Tier 2: PyTorch FP16 + fused demod ...")
    t = run_pytorch(model_path, inp_hw, fp16=True)
    results.append(("PyTorch FP16 + demod", t))
    print(f"  {t:.2f} ms")

    print("Tier 3: PyTorch FP16 + fused demod + CUDA graph ...")
    try:
        t = run_cuda_graph(model_path, inp_hw)
        results.append(("PyTorch FP16 + demod + CUDAGraph", t))
        print(f"  {t:.2f} ms")
    except Exception as e:
        print(f"  SKIPPED: {e}")

    print("Tier 4: PyTorch FP16 + fused demod + torch.compile + CUDA graph ...")
    if not int(os.environ.get("GPEN_TORCH_COMPILE", "0")):
        print("  Skipped — torch.compile crashes (access violation) on this model on Windows/sm_89.")
        print("  Set GPEN_TORCH_COMPILE=1 to attempt.")
    else:
        try:
            t = run_compile_cuda_graph(model_path, inp_hw)
            results.append(("PT FP16 + demod + compile + CUDAGraph", t))
            print(f"  {t:.2f} ms")
        except Exception as e:
            print(f"  SKIPPED: {e}")

    print("Tier 4b: torch.compile(reduce-overhead) — no extra CUDA graph ...")
    if not int(os.environ.get("GPEN_TORCH_COMPILE", "0")):
        print("  Skipped — reduce-overhead may crash (MLIR segfault) on this model on Windows/sm_89.")
        print("  Set GPEN_TORCH_COMPILE=1 to attempt.")
    else:
        try:
            import torch as _torch
            from custom_kernels.gpen_bfr.gpen_torch import GPENTorch
            from custom_kernels.compile_utils import apply_torch_compile
            import time as _time
            _m4b = GPENTorch.from_onnx(model_path, compute_dtype=_torch.float16).cuda().eval()
            _inp4b = _torch.randn(1, 3, inp_hw, inp_hw, device="cuda")
            _m4b_compiled = apply_torch_compile(_m4b, _inp4b, compile_mode="reduce-overhead")
            with _torch.no_grad():
                for _ in range(WARMUP):
                    _m4b_compiled(_inp4b)
            _torch.cuda.synchronize()
            _t0_4b = _time.perf_counter()
            with _torch.no_grad():
                for _ in range(RUNS):
                    _m4b_compiled(_inp4b)
            _torch.cuda.synchronize()
            t_4b = (_time.perf_counter() - _t0_4b) / RUNS * 1000
            results.append(("torch.compile reduce-overhead (no CUDA graph)", t_4b))
            print(f"  {t_4b:.2f} ms")
        except Exception as e:
            print(f"  SKIPPED: {e}")
            import traceback; traceback.print_exc()

    cuda_ep_t = results[0][1]
    hdr = f"{'Method':<40} {'ms':>8} {'vs CUDA EP':>11}"
    if t_trt:
        hdr += f" {'vs TRT EP':>10}"
    print(f"\n{hdr}")
    print("-" * len(hdr))
    for name, t in results:
        row = f"  {name:<40} {t:>8.2f} {cuda_ep_t / t:>10.2f}x"
        if t_trt:
            row += f" {t_trt / t:>9.2f}x"
        print(row)

    out_path = Path(__file__).parent / f"benchmark_results_{size}.txt"
    with open(out_path, "w") as f:
        for name, t in results:
            f.write(f"{name}: {t:.2f} ms  (vs CUDA EP: {cuda_ep_t / t:.2f}x")
            if t_trt:
                f.write(f", vs TRT EP: {t_trt / t:.2f}x")
            f.write(")\n")
    print(f"\n[Saved -> {out_path}]")


def main():
    if len(sys.argv) > 1:
        sizes = [int(s) for s in sys.argv[1:]]
    else:
        sizes = list(MODELS.keys())

    for size in sizes:
        mp = ROOT / "model_assets" / MODELS[size]
        if not mp.exists():
            print(f"\nSKIP: {MODELS[size]} not found")
            continue
        benchmark_one(size)


if __name__ == "__main__":
    main()
