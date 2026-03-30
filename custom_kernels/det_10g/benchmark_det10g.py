"""
Benchmark — det_10g (SCRFD-10G face detector) custom kernels vs ORT baseline.

Tests the full SCRFD pipeline:
  Input : (1, 3, H, W) float32   (default 640×640)
  Outputs: 9 tensors — scores×3, bbox×3, kps×3

Usage (from repo root):
    .venv\\Scripts\\python custom_kernels\\det_10g\\benchmark_det10g.py

Optional env vars:
    ONNX_DIR=model_assets          path to ONNX model files
    INPUT_H=640                    input height  (default 640)
    INPUT_W=640                    input width   (default 640)
    WARMUP=10                      warm-up iterations
    ITERS=50                       timed iterations
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
INPUT_H = int(os.environ.get("INPUT_H", 640))
INPUT_W = int(os.environ.get("INPUT_W", 640))
WARMUP = int(os.environ.get("WARMUP", 10))
ITERS = int(os.environ.get("ITERS", 50))

DET_ONNX = str(ONNX_DIR / "det_10g.onnx")


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
def _check_accuracy(
    ort_outs: list,
    pt_outs: tuple,
) -> None:
    """Compare ORT FP32 outputs vs PyTorch FP16 outputs."""
    # ONNX output order: scores_8/16/32, bbox_8/16/32, kps_8/16/32
    out_names = [
        "scores_8",
        "scores_16",
        "scores_32",
        "bbox_8",
        "bbox_16",
        "bbox_32",
        "kps_8",
        "kps_16",
        "kps_32",
    ]
    print("\n  Numerical accuracy (FP16 PyTorch vs ORT FP32):")
    max_errs = []
    for name, ort_out, pt_out in zip(out_names, ort_outs, pt_outs):
        a = ort_out.astype(np.float32)
        b = pt_out.detach().cpu().numpy().astype(np.float32)
        if a.shape != b.shape:
            print(f"    {name}: shape mismatch ORT={a.shape} PT={b.shape}")
            continue
        mae = float(np.abs(a - b).mean())
        maxe = float(np.abs(a - b).max())
        max_errs.append(maxe)
        print(f"    {name}: MAE={mae:.2e}  MaxAbsErr={maxe:.2e}")
    if max_errs:
        print(f"  Overall max abs error: {max(max_errs):.2e}  (target < 1e-2)")


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------
def bench_det10g():
    print(f"\n=== det_10g / SCRFD-10G  (1,3,{INPUT_H},{INPUT_W}) → 9 outputs ===")
    print(f"  warm-up={WARMUP}, iters={ITERS}\n")
    print(f"  {'Tier':<6} | {'Method':<46} | {'ms':>8} | {'speedup':>7}")
    print(f"  {'-' * 6}-+-{'-' * 46}-+-{'-' * 8}-+-{'-' * 7}")

    inp = torch.randn(1, 3, INPUT_H, INPUT_W, dtype=torch.float32, device="cuda")
    inp_np = inp.cpu().numpy()

    ORT_OUT_NAMES = ["448", "471", "494", "451", "474", "497", "454", "477", "500"]

    # ── Tier 0 — ORT FP32 CUDA EP ────────────────────────────────────────
    sess0 = _ort_session(DET_ONNX, "CUDAExecutionProvider")
    in_name = sess0.get_inputs()[0].name  # "input.1"
    t0 = _bench(lambda: sess0.run(ORT_OUT_NAMES, {in_name: inp_np}))
    _print_row("0", "ORT FP32 CUDA EP", t0, t0)

    # ── Tier 0b — ORT TensorRT EP ─────────────────────────────────────────
    t0b = t0
    import tempfile
    import shutil
    import onnxruntime as ort

    if "TensorrtExecutionProvider" not in ort.get_available_providers():
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
            so = ort.SessionOptions()
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            print(
                "  Tier 0b | ORT TRT EP — building engine (first run may take 1-5 min)..."
            )
            sess0b = ort.InferenceSession(
                DET_ONNX,
                so,
                providers=[
                    ("TensorrtExecutionProvider", trt_opts),
                    ("CUDAExecutionProvider", {"device_id": "0"}),
                    ("CPUExecutionProvider", {}),
                ],
            )
            for _ in range(WARMUP):
                sess0b.run(ORT_OUT_NAMES, {in_name: inp_np})
            t0b = _bench(lambda: sess0b.run(ORT_OUT_NAMES, {in_name: inp_np}))
            _print_row("0b", "ORT TRT EP FP32", t0b, t0)
            trt_out = sess0b.run(ORT_OUT_NAMES, {in_name: inp_np})
            ref_out_0 = sess0.run(ORT_OUT_NAMES, {in_name: inp_np})
            for i, (a, b) in enumerate(zip(ref_out_0, trt_out)):
                diff = abs(np.array(a) - np.array(b)).max()
                print(f"  TRT vs CUDA EP output[{i}]: max|diff| = {diff:.5e}")
        except Exception as e:
            print(f"  Tier 0b | TRT EP — failed: {e}")
        finally:
            shutil.rmtree(_trt_tmp, ignore_errors=True)

    # ── Tier 1 — PyTorch FP32 ────────────────────────────────────────────
    sys.path.insert(0, str(ROOT))
    from custom_kernels.det_10g.det10g_torch import Det10gTorch, build_cuda_graph_runner

    m_fp32 = Det10gTorch.from_onnx(DET_ONNX, compute_dtype=torch.float32).cuda().eval()
    with torch.no_grad():
        t1 = _bench(lambda: m_fp32(inp))
    _print_row("1", "PyTorch FP32 pure ops", t1, t0)

    # ── Tier 2 — PyTorch FP16 ─────────────────────────────────────────────
    m_fp16 = Det10gTorch.from_onnx(DET_ONNX, compute_dtype=torch.float16).cuda().eval()
    with torch.no_grad():
        t2 = _bench(lambda: m_fp16(inp))
    _print_row("2", "PyTorch FP16 (TensorCore conv dispatch)", t2, t0)

    # ── Accuracy check ────────────────────────────────────────────────────
    ort_outs = sess0.run(ORT_OUT_NAMES, {in_name: inp_np})
    with torch.no_grad():
        pt_outs = m_fp16(inp)
    _check_accuracy(ort_outs, pt_outs)

    # ── Tier 3 — FP16 + CUDA graph ────────────────────────────────────────
    try:
        runner = build_cuda_graph_runner(m_fp16)
        with torch.no_grad():
            _ = runner(inp)  # trigger first-shape graph capture
        t3 = _bench(lambda: runner(inp))
        _print_row("3", "FP16 NCHW + CUDA graph (per-shape cache)", t3, t0)
    except Exception as e:
        print(f"  Tier 3 | CUDA graph — skipped ({e})")
        t3 = t2

    # ── Tier 4 — FP16 NHWC (channels_last) ───────────────────────────────
    from custom_kernels.det_10g.det10g_torch import Det10gTorch as _Det10gTorch

    m_nhwc = (
        _Det10gTorch.from_onnx(
            DET_ONNX, compute_dtype=torch.float16, channels_last=True
        )
        .cuda()
        .eval()
    )
    with torch.no_grad():
        t4 = _bench(lambda: m_nhwc(inp))
    _print_row("4", "FP16 NHWC (channels_last, cuDNN native path)", t4, t0)

    # ── Tier 5 — FP16 NHWC + CUDA graph  (recommended) ──────────────────
    try:
        runner_nhwc = build_cuda_graph_runner(m_nhwc)
        with torch.no_grad():
            _ = runner_nhwc(inp)  # first call captures graph
        t5 = _bench(lambda: runner_nhwc(inp))
        _print_row("5", "FP16 NHWC + CUDA graph (recommended)", t5, t0)
    except Exception as e:
        print(f"  Tier 5 | NHWC + CUDA graph — skipped ({e})")
        t5 = t4

    # ── Accuracy check (NHWC vs ORT FP32) ────────────────────────────────
    with torch.no_grad():
        pt_outs_nhwc = m_nhwc(inp)
    print("\n  Accuracy (FP16 NHWC vs ORT FP32 — should match FP16 NCHW):")
    _check_accuracy(ort_outs, pt_outs_nhwc)

    # ── Tier 6 — torch.compile + FP16 NHWC + CUDA graph ──────────────────
    t6 = t5
    print("\n  [Tier 6] torch.compile(mode='default') + FP16 NHWC + CUDA graph")
    print("  One-time compile cost: ~30 s on first run (Triton JIT).")
    try:
        m6 = (
            Det10gTorch.from_onnx(DET_ONNX, compute_dtype=torch.float16)
            .to(memory_format=torch.channels_last)
            .cuda()
            .eval()
        )
        runner6 = build_cuda_graph_runner(m6, torch_compile=True)
        with torch.no_grad():
            _ = runner6(inp)  # first call: trigger graph capture for INPUT_H×INPUT_W
        t6 = _bench(lambda: runner6(inp))
        _print_row("6", "torch.compile + FP16 NHWC + CUDA graph", t6, t0)
    except Exception as e:
        print(f"  Tier 6 | torch.compile — failed: {e}")
        import traceback; traceback.print_exc()

    # ── Tier 4b — torch.compile reduce-overhead (no separate CUDA graph) ────
    print("\n  [Tier 4b] torch.compile(mode='reduce-overhead') — no extra CUDA graph")
    try:
        from custom_kernels.compile_utils import apply_torch_compile
        inp_f32 = inp
        m4b = Det10gTorch.from_onnx(DET_ONNX, compute_dtype=torch.float16).cuda().eval()
        m4b_compiled = apply_torch_compile(
            m4b,
            inp_f32,
            compile_mode="reduce-overhead",
        )
        with torch.no_grad():
            t4b = _bench(lambda: m4b_compiled(inp_f32))
        _print_row("4b", "torch.compile reduce-overhead (no CUDA graph)", t4b, t0)
    except Exception as e:
        print(f"  Tier 4b | reduce-overhead — failed: {e}")
        import traceback; traceback.print_exc()

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n  {'─' * 72}")
    print(f"  SUMMARY  (input {INPUT_H}×{INPUT_W}, {ITERS} iters)")
    print(f"  {'─' * 72}")
    print(f"  {'Tier':<6} | {'Method':<46} | {'ms':>8} | {'speedup vs ORT FP32':>20}")
    print(f"  {'-' * 6}-+-{'-' * 46}-+-{'-' * 8}-+-{'-' * 20}")
    rows = [
        ("0", "ORT FP32 CUDA EP", t0),
        ("0b", "ORT TensorRT EP", t0b),
        ("2", "FP16 NCHW pure ops", t2),
        ("3", "FP16 NCHW + CUDA graph", t3),
        ("4", "FP16 NHWC (channels_last)", t4),
        ("5", "FP16 NHWC + CUDA graph  ← recommended", t5),
        ("6", "torch.compile + FP16 NHWC + CUDA graph", t6),
    ]
    for tier, label, ms_val in rows:
        speedup = t0 / ms_val if ms_val > 0 else float("nan")
        marker = " ***" if tier == "6" else ""
        print(f"  {tier:<6} | {label:<46} | {ms_val:8.3f} ms | {speedup:6.2f}×{marker}")
    print(f"  {'─' * 72}")

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

    if not pathlib.Path(DET_ONNX).exists():
        print(f"ERROR: det_10g ONNX not found: {DET_ONNX}")
        sys.exit(1)

    bench_det10g()
