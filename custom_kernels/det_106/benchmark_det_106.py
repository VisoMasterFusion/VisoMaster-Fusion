"""
Benchmark — det_106 custom kernel vs ORT baseline.

Usage (from repo root):
    .venv\\Scripts\\python custom_kernels\\det_106\\benchmark_det_106.py

Optional env vars:
    ONNX_PATH=model_assets/2d106det.onnx
    WARMUP=50
    ITERS=500
"""

from __future__ import annotations

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

import os
import sys
import time
import pathlib

import numpy as np
import torch

# ---------------------------------------------------------------------------
ROOT = pathlib.Path(__file__).parent.parent.parent
ONNX_PATH = pathlib.Path(
    os.environ.get(
        "ONNX_PATH",
        ROOT / "model_assets" / "2d106det.onnx",
    )
)
WARMUP = int(os.environ.get("WARMUP", 50))
ITERS = int(os.environ.get("ITERS", 500))

sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------


def _bench(fn, warmup: int = WARMUP, iters: int = ITERS) -> float:
    """Return mean ms/iteration (GPU-synchronised wall-clock)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0


def _ort_session(path: str, provider: str = "CUDAExecutionProvider"):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), sess_options=opts, providers=[provider])


def _print_row(tier, label, ms_val, ref_ms):
    speedup = ref_ms / ms_val if ms_val > 0 else float("nan")
    print(f"  Tier {tier} | {label:<44} | {ms_val:8.3f} ms | {speedup:6.2f}×")


# ---------------------------------------------------------------------------


def bench():
    print("\n=== det_106 landmark detector — (1,3,192,192) → (1,212) ===")
    print(f"  warm-up={WARMUP}, iters={ITERS}\n")
    print(f"  {'Tier':<6} | {'Method':<44} | {'ms':>8} | {'speedup':>7}")
    print(f"  {'-' * 6}-+-{'-' * 44}-+-{'-' * 8}-+-{'-' * 7}")

    # ORT expects the same preprocessing the app applies: no-op normalize (float32
    # in [0,255]); the ONNX model applies Sub(127.5)+Mul(0.0078125) internally.
    inp = torch.randint(0, 256, (1, 3, 192, 192), dtype=torch.float32, device="cuda")
    inp_np = inp.cpu().numpy()

    # ── Tier 0: ORT FP32 CUDA EP ─────────────────────────────────────────
    sess0 = _ort_session(ONNX_PATH, "CUDAExecutionProvider")
    in_name = sess0.get_inputs()[0].name
    out_names = [o.name for o in sess0.get_outputs()]
    t0 = _bench(lambda: sess0.run(out_names, {in_name: inp_np}))
    _print_row("0", "ORT FP32 CUDA EP (baseline)", t0, t0)

    # ── Tier 0b: ORT TensorRT EP ─────────────────────────────────────────
    t0b = t0
    import tempfile
    import shutil
    import onnxruntime as _ort

    if "TensorrtExecutionProvider" not in _ort.get_available_providers():
        print("  Tier 0b | TRT EP — skipped (provider not available)")
    else:
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
            so = _ort.SessionOptions()
            so.graph_optimization_level = _ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            print(
                "  Tier 0b | ORT TRT EP — building engine (first run may take 1-5 min)..."
            )
            sess0b = _ort.InferenceSession(
                str(ONNX_PATH),
                so,
                providers=[
                    ("TensorrtExecutionProvider", trt_opts),
                    ("CUDAExecutionProvider", {"device_id": "0"}),
                    ("CPUExecutionProvider", {}),
                ],
            )
            for _ in range(WARMUP):
                sess0b.run(out_names, {in_name: inp_np})
            t0b = _bench(lambda: sess0b.run(out_names, {in_name: inp_np}))
            _print_row("0b", "ORT TRT EP FP32", t0b, t0)
            trt_out = sess0b.run(out_names, {in_name: inp_np})
            ref_out_0b = sess0.run(out_names, {in_name: inp_np})
            for i, (a, b) in enumerate(zip(ref_out_0b, trt_out)):
                diff = abs(np.array(a) - np.array(b)).max()
                print(f"  TRT vs CUDA EP output[{i}]: max|diff| = {diff:.5e}")
        except Exception as e:
            print(f"  Tier 0b | TRT EP — failed: {e}")
        finally:
            shutil.rmtree(_trt_tmp, ignore_errors=True)

    # ── Tier 1: PyTorch FP32 eager ────────────────────────────────────────
    from custom_kernels.det_106.det_106_torch import Det106Torch

    m_fp32 = Det106Torch.from_onnx(ONNX_PATH, compute_dtype=torch.float32).cuda().eval()
    with torch.no_grad():
        t1 = _bench(lambda: m_fp32(inp))
    _print_row("1", "PyTorch FP32 eager", t1, t0)

    # ── Tier 2: PyTorch FP16 eager ────────────────────────────────────────
    m_fp16 = Det106Torch.from_onnx(ONNX_PATH, compute_dtype=torch.float16).cuda().eval()
    with torch.no_grad():
        t2 = _bench(lambda: m_fp16(inp))
    _print_row("2", "PyTorch FP16 eager", t2, t0)

    # ── Tier 3: PyTorch FP16 + CUDA graph ─────────────────────────────────
    from custom_kernels.det_106.det_106_torch import build_cuda_graph_runner

    runner = build_cuda_graph_runner(m_fp16)
    t3 = _bench(lambda: runner(inp))
    _print_row("3", "PyTorch FP16 + CUDA graph", t3, t0)

    # ── Tier 4 — torch.compile + FP16 + CUDA graph ────────────────────────
    print("\n  [Tier 4] torch.compile(mode='default') + CUDA graph")
    print("  One-time compile cost: ~30 s on first run (Triton JIT).")
    try:
        m4 = Det106Torch.from_onnx(ONNX_PATH, compute_dtype=torch.float16).cuda().eval()
        runner4 = build_cuda_graph_runner(m4, torch_compile=True)
        t4 = _bench(lambda: runner4(inp))
        _print_row("4", "torch.compile + FP16 + CUDA graph", t4, t0)
    except Exception as e:
        print(f"  Tier 4 | torch.compile — failed: {e}")
        import traceback; traceback.print_exc()

    # ── Tier 4b — torch.compile reduce-overhead (no separate CUDA graph) ────
    print("\n  [Tier 4b] torch.compile(mode='reduce-overhead') — no extra CUDA graph")
    if not int(os.environ.get("DET106_TORCH_COMPILE", "0")):
        print("  Skipped — reduce-overhead may crash on this model on Windows/sm_89.")
        print("  Set DET106_TORCH_COMPILE=1 to attempt.")
    else:
        try:
            from custom_kernels.compile_utils import apply_torch_compile
            m4b = Det106Torch.from_onnx(ONNX_PATH, compute_dtype=torch.float16).cuda().eval()
            m4b_compiled = apply_torch_compile(
                m4b,
                inp,
                compile_mode="reduce-overhead",
            )
            with torch.no_grad():
                t4b = _bench(lambda: m4b_compiled(inp))
            _print_row("4b", "torch.compile reduce-overhead (no CUDA graph)", t4b, t0)
        except Exception as e:
            print(f"  Tier 4b | reduce-overhead — failed: {e}")
            import traceback; traceback.print_exc()

    print()

    # ── Numerical accuracy check ──────────────────────────────────────────
    print("=== Numerical accuracy (ORT FP32 vs PyTorch FP16 + CUDA graph) ===")
    ref_outs = sess0.run(out_names, {in_name: inp_np})
    # ref_outs[0] shape: (1, 212) or (N, 212) — 106 (x,y) pairs in model space
    ref_pred = ref_outs[0].reshape(106, 2)
    with torch.no_grad():
        pt_out = runner(inp)
    pt_pred = pt_out.detach().cpu().numpy().reshape(106, 2)

    max_err = float(np.abs(ref_pred - pt_pred).max())
    mean_err = float(np.abs(ref_pred - pt_pred).mean())
    print(f"  max  |Δ| = {max_err:.4f}  (landmark x/y in model space [-1,1])")
    print(f"  mean |Δ| = {mean_err:.6f}")
    if max_err < 0.01:
        print("  ✓ Accuracy OK (max error < 0.01 in model space)")
    else:
        print("  ✗ Accuracy WARNING — large deviation detected")
    print()


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available.")
        sys.exit(1)

    print(f"\nDevice : {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    try:
        import onnxruntime as ort

        print(f"ORT    : {ort.__version__}")
    except ImportError:
        print("ORT    : not available — install onnxruntime-gpu")
        sys.exit(1)

    if not ONNX_PATH.exists():
        print(f"ERROR: ONNX not found: {ONNX_PATH}")
        sys.exit(1)

    bench()
