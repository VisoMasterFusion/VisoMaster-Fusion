"""
Benchmark — XSeg_model (SN256_XSeg face segmentation U-Net) custom kernels vs ORT.

Tests the full segmentation pipeline:
  Input : (1, 3, 256, 256) float32  — normalised RGB face crop [0, 1]
  Output: (1, 1, 256, 256) float32  — sigmoid segmentation mask

Usage (from repo root):
    .venv\\Scripts\\python custom_kernels\\xseg\\benchmark_xseg.py

Optional env vars:
    ONNX_DIR=model_assets      path to ONNX model files
    WARMUP=5                   warm-up iterations
    ITERS=30                   timed iterations
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
# Config
# ---------------------------------------------------------------------------
ROOT = pathlib.Path(__file__).parent.parent.parent
ONNX_DIR = pathlib.Path(os.environ.get("ONNX_DIR", ROOT / "model_assets"))
WARMUP = int(os.environ.get("WARMUP", 5))
ITERS = int(os.environ.get("ITERS", 30))

XSEG_ONNX = str(ONNX_DIR / "XSeg_model.onnx")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _ms(elapsed_s: float, iters: int) -> float:
    return elapsed_s / iters * 1000.0


def _bench(fn, warmup: int = WARMUP, iters: int = ITERS) -> float:
    """Return ms/iter (GPU-synchronised)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return _ms(time.perf_counter() - t0, iters)


def _ort_session(onnx_path: str, provider: str = "CUDAExecutionProvider"):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(onnx_path, sess_options=opts, providers=[provider])


def _print_row(tier, label, ms_val, ref_ms):
    speedup = ref_ms / ms_val if ms_val > 0 else float("nan")
    print(f"  Tier {tier} | {label:<46} | {ms_val:8.2f} ms | {speedup:6.2f}x")


# ---------------------------------------------------------------------------
# Numerical accuracy check
# ---------------------------------------------------------------------------
def _check_accuracy(ort_out: np.ndarray, pt_out: torch.Tensor) -> None:
    """Compare ORT FP32 mask vs PyTorch FP16 mask."""
    a = ort_out.flatten().astype(np.float32)
    b = pt_out.detach().cpu().numpy().flatten().astype(np.float32)
    diff = np.abs(a - b)
    # Per-pixel binary agreement at threshold 0.5
    a_bin = (a > 0.5).astype(np.uint8)
    b_bin = (b > 0.5).astype(np.uint8)
    agree = (a_bin == b_bin).mean() * 100
    print("\n  Numerical accuracy (FP16 PyTorch vs ORT FP32):")
    print(f"    MAE={diff.mean():.2e}  MaxAbsErr={diff.max():.2e}")
    print(f"    Binary pixel agreement (threshold 0.5): {agree:.2f}%  (target > 99%)")


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------
def bench_xseg():
    print("\n=== XSeg_model / SN256_XSeg U-Net  (1,3,256,256) → (1,1,256,256) ===")
    print(f"  warm-up={WARMUP}, iters={ITERS}\n")
    print(f"  {'Tier':<6} | {'Method':<46} | {'ms':>8} | {'speedup':>7}")
    print(f"  {'-' * 6}-+-{'-' * 46}-+-{'-' * 8}-+-{'-' * 7}")

    # Normalised face-crop input ([0, 1])
    inp_f32 = torch.rand(1, 3, 256, 256, dtype=torch.float32, device="cuda")
    inp_np = inp_f32.cpu().numpy()

    # ── Tier 0 — ORT FP32 CUDA EP ────────────────────────────────────────
    sess0 = _ort_session(XSEG_ONNX, "CUDAExecutionProvider")
    in_name = sess0.get_inputs()[0].name  # "in_face:0"
    t0 = _bench(lambda: sess0.run(None, {in_name: inp_np}))
    _print_row("0", "ORT FP32 CUDA EP", t0, t0)

    # ── Tier 0b — ORT TensorRT EP ─────────────────────────────────────────
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
                XSEG_ONNX,
                so,
                providers=[
                    ("TensorrtExecutionProvider", trt_opts),
                    ("CUDAExecutionProvider", {"device_id": "0"}),
                    ("CPUExecutionProvider", {}),
                ],
            )
            for _ in range(WARMUP):
                sess0b.run(None, {in_name: inp_np})
            t0b = _bench(lambda: sess0b.run(None, {in_name: inp_np}))
            _print_row("0b", "ORT TRT EP FP32", t0b, t0)
            trt_out = sess0b.run(None, {in_name: inp_np})[0]
            ref_out_0b = sess0.run(None, {in_name: inp_np})[0]
            diff = abs(np.array(ref_out_0b) - np.array(trt_out)).max()
            print(f"  TRT vs CUDA EP output[0]: max|diff| = {diff:.5e}")
        except Exception as e:
            print(f"  Tier 0b | TRT EP — failed: {e}")
        finally:
            shutil.rmtree(_trt_tmp, ignore_errors=True)

    # ── Tier 1 — PyTorch FP32 ────────────────────────────────────────────
    sys.path.insert(0, str(ROOT))
    from custom_kernels.xseg.xseg_torch import XSegTorch, build_cuda_graph_runner

    m_fp32 = XSegTorch.from_onnx(XSEG_ONNX, compute_dtype=torch.float32).cuda().eval()
    with torch.no_grad():
        t1 = _bench(lambda: m_fp32(inp_f32))
    _print_row("1", "PyTorch FP32 pure ops", t1, t0)

    # ── Tier 2 — PyTorch FP16 ─────────────────────────────────────────────
    m_fp16 = XSegTorch.from_onnx(XSEG_ONNX, compute_dtype=torch.float16).cuda().eval()
    with torch.no_grad():
        t2 = _bench(lambda: m_fp16(inp_f32))
    _print_row("2", "PyTorch FP16 (TensorCore conv dispatch)", t2, t0)

    # ── Accuracy check ────────────────────────────────────────────────────
    ort_out = sess0.run(None, {in_name: inp_np})[0]  # (1, 1, 256, 256)
    with torch.no_grad():
        pt_out = m_fp16(inp_f32)
    _check_accuracy(ort_out, pt_out)

    # ── Tier 3 — FP16 + CUDA graph ────────────────────────────────────────
    runner3 = None
    try:
        runner3 = build_cuda_graph_runner(m_fp16)
        with torch.no_grad():
            _ = runner3(inp_f32)  # verify
        t3 = _bench(lambda: runner3(inp_f32))
        _print_row("3", "FP16 + CUDA graph (single captured graph)", t3, t0)
    except Exception as e:
        print(f"  Tier 3 | CUDA graph — skipped ({e})")

    # ── Tier 4 — Triton RMSNormMax + FP16 + CUDA graph ───────────────────
    try:
        from custom_kernels.triton_ops import TRITON_AVAILABLE

        if not TRITON_AVAILABLE:
            print("  Tier 4 | Triton — skipped (Triton not available)")
        else:
            # Build a fresh fp16 model; Triton path is enabled automatically
            # inside _RMSNormMax.forward() when _TRITON_RMSNORMMAX is set.
            m_triton = (
                XSegTorch.from_onnx(XSEG_ONNX, compute_dtype=torch.float16)
                .cuda()
                .eval()
            )
            runner4 = build_cuda_graph_runner(m_triton)
            with torch.no_grad():
                _ = runner4(inp_f32)  # verify
            t4 = _bench(lambda: runner4(inp_f32))
            _print_row("4", "FP16 + Triton RMSNormMax + CUDA graph", t4, t0)
    except Exception as e:
        print(f"  Tier 4 | Triton — skipped ({e})")

    # ── Tier 5 — torch.compile + FP16 + CUDA graph ────────────────────────
    print("\n  [Tier 5] torch.compile(mode='default') + CUDA graph")
    print("  One-time compile cost: ~30 s on first run (Triton JIT).")
    try:
        m5 = XSegTorch.from_onnx(XSEG_ONNX, compute_dtype=torch.float16).cuda().eval()
        runner5 = build_cuda_graph_runner(m5, torch_compile=True)
        t5 = _bench(lambda: runner5(inp_f32))
        _print_row("5", "torch.compile + FP16 + CUDA graph", t5, t0)
    except Exception as e:
        print(f"  Tier 5 | torch.compile — failed: {e}")
        import traceback; traceback.print_exc()

    # ── Tier 5b — torch.compile reduce-overhead (no separate CUDA graph) ────
    print("\n  [Tier 5b] torch.compile(mode='reduce-overhead') — no extra CUDA graph")
    if not int(os.environ.get("XSEG_TORCH_COMPILE", "0")):
        print("  Skipped — reduce-overhead may crash on this model on Windows/sm_89.")
        print("  Set XSEG_TORCH_COMPILE=1 to attempt.")
    else:
        try:
            from custom_kernels.compile_utils import apply_torch_compile
            m5b = XSegTorch.from_onnx(XSEG_ONNX, compute_dtype=torch.float16).cuda().eval()
            m5b_compiled = apply_torch_compile(
                m5b,
                inp_f32,
                compile_mode="reduce-overhead",
            )
            with torch.no_grad():
                t5b = _bench(lambda: m5b_compiled(inp_f32))
            _print_row("5b", "torch.compile reduce-overhead (no CUDA graph)", t5b, t0)
        except Exception as e:
            print(f"  Tier 5b | reduce-overhead — failed: {e}")
            import traceback; traceback.print_exc()

    print()
    return t0, t0b


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available.")
        sys.exit(1)

    dev = torch.cuda.get_device_name(0)
    print(f"\nDevice: {dev}")
    print(f"PyTorch: {torch.__version__}")
    try:
        import onnxruntime as ort

        print(f"ORT:     {ort.__version__}")
    except ImportError:
        print("ORT:     not available (skipping Tier 0/0b)")
        sys.exit(1)

    import pathlib as _pl

    if not _pl.Path(XSEG_ONNX).exists():
        print(f"ERROR: XSeg ONNX not found: {XSEG_ONNX}")
        sys.exit(1)

    bench_xseg()
