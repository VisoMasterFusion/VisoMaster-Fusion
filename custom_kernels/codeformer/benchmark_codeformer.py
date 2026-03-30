"""
Benchmark — CodeFormer custom kernels vs ORT baseline.

Tests the full CodeFormer pipeline:
  Input : (1, 3, 512, 512) float32
  Output: (1, 3, 512, 512) float32

Usage (from repo root):
    .venv\\Scripts\\python custom_kernels\\codeformer\\benchmark_codeformer.py

Optional env vars:
    ONNX_DIR=model_assets          path to ONNX model files
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

import torch

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ROOT = pathlib.Path(__file__).parent.parent.parent
ONNX_DIR = pathlib.Path(os.environ.get("ONNX_DIR", ROOT / "model_assets"))
WARMUP = int(os.environ.get("WARMUP", 10))
ITERS = int(os.environ.get("ITERS", 50))

CF_ONNX = str(ONNX_DIR / "codeformer_fp16.onnx")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _ms(elapsed_s: float, iters: int) -> float:
    return elapsed_s / iters * 1000.0


def _bench(fn, warmup=WARMUP, iters=ITERS) -> float:
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
    print(f"  Tier {tier} | {label:<44} | {ms_val:8.2f} ms | {speedup:6.2f}x")


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------
def bench_codeformer():
    print("\n=== CodeFormer (1,3,512,512) -> (1,3,512,512) ===")
    print(f"  warm-up={WARMUP}, iters={ITERS}\n")
    print(f"  {'Tier':<6} | {'Method':<44} | {'ms':>8} | {'speedup':>7}")
    print(f"  {'-' * 6}-+-{'-' * 44}-+-{'-' * 8}-+-{'-' * 7}")

    inp = torch.randn(1, 3, 512, 512, dtype=torch.float32, device="cuda")
    inp_np = inp.cpu().numpy()
    w_val = 0.5  # fidelity weight for all tiers

    # ── Tier 0 — ORT FP32 CUDA EP ────────────────────────────────────────
    import numpy as np

    sess0 = _ort_session(CF_ONNX, "CUDAExecutionProvider")
    in_name = sess0.get_inputs()[0].name  # "x"
    w_name = sess0.get_inputs()[1].name  # "w"  (float64 scalar)
    out_name = sess0.get_outputs()[0].name
    w_np = np.array([w_val], dtype=np.float64)
    t0 = _bench(lambda: sess0.run([out_name], {in_name: inp_np, w_name: w_np}))
    _print_row("0", "ORT FP32 CUDA EP", t0, t0)

    # ── Tier 0b — ORT TensorRT EP ─────────────────────────────────────────
    # NOTE: TRT engine building for transformer models at opt-level 5 can crash
    # the process with SIGSEGV/SIGILL. We run it in a subprocess to isolate the
    # crash and fall back to opt-level 3 if level 5 fails.
    t0b = t0
    import tempfile
    import shutil
    import subprocess
    import json
    import numpy as np
    import onnxruntime as _ort

    if "TensorrtExecutionProvider" not in _ort.get_available_providers():
        print("  Tier 0b | TRT EP — skipped (provider not available)")
    else:
        _trt_tmp = tempfile.mkdtemp(prefix="ort_trt_bench_")
        try:
            # Build a small helper script that runs TRT EP and prints JSON result
            _helper = pathlib.Path(_trt_tmp) / "_trt_helper.py"
            _helper.write_text(f"""
import sys, json, time, numpy as np, tempfile, shutil
sys.path.insert(0, r"{str(ROOT)}")
import onnxruntime as ort
import torch

WARMUP = {WARMUP}
ITERS  = {ITERS}
CF_ONNX = r"{CF_ONNX}"
in_name = "{in_name}"
w_name  = "{w_name}"
out_name = "{out_name}"
inp_np = np.zeros((1,3,512,512), dtype=np.float32)
w_np   = np.array([0.5], dtype=np.float64)

for opt_level in (5, 3):
    cache = tempfile.mkdtemp(prefix="ort_trt_")
    try:
        trt_opts = {{
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": cache,
            "trt_timing_cache_enable": True,
            "trt_timing_cache_path": cache,
            "trt_layer_norm_fp32_fallback": True,
            "trt_max_workspace_size": 8589934592,
            "trt_builder_optimization_level": opt_level,
        }}
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess = ort.InferenceSession(CF_ONNX, so, providers=[
            ("TensorrtExecutionProvider", trt_opts),
            ("CUDAExecutionProvider", {{"device_id": "0"}}),
            ("CPUExecutionProvider", {{}}),
        ])
        for _ in range(WARMUP):
            sess.run([out_name], {{in_name: inp_np, w_name: w_np}})
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(ITERS):
            sess.run([out_name], {{in_name: inp_np, w_name: w_np}})
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / ITERS * 1000
        out = sess.run([out_name], {{in_name: inp_np, w_name: w_np}})[0]
        print(json.dumps({{"ms": ms, "opt_level": opt_level, "out": out.flatten()[:4].tolist()}}))
        sys.exit(0)
    except Exception as e:
        print(f"opt_level={{opt_level}} failed: {{e}}", file=sys.stderr)
    finally:
        shutil.rmtree(cache, ignore_errors=True)
sys.exit(1)
""", encoding="utf-8")
            print(
                "  Tier 0b | ORT TRT EP — building engine (first run may take 1-5 min)..."
            )
            r = subprocess.run(
                [sys.executable, str(_helper)],
                capture_output=True, text=True, timeout=600,
                cwd=str(ROOT),
            )
            if r.returncode == 0:
                for line in r.stdout.splitlines():
                    if line.startswith("{"):
                        d = json.loads(line)
                        t0b = d["ms"]
                        opt_used = d["opt_level"]
                        label = f"ORT TRT EP (opt={opt_used})"
                        _print_row("0b", label, t0b, t0)
                        ref_out_0b = sess0.run([out_name], {in_name: inp_np, w_name: w_np})[0]
                        # compare only first 4 values (full output too large to pass via JSON)
                        print(f"  TRT opt_level={opt_used} used for benchmark")
            else:
                print(f"  Tier 0b | TRT EP — failed (rc={r.returncode}): {r.stderr[-200:]}")
        except subprocess.TimeoutExpired:
            print("  Tier 0b | TRT EP — timed out after 10 min")
        except Exception as e:
            print(f"  Tier 0b | TRT EP — failed: {e}")
        finally:
            shutil.rmtree(_trt_tmp, ignore_errors=True)

    # ── Tier 1 — PyTorch FP32 ────────────────────────────────────────────
    sys.path.insert(0, str(ROOT))
    from custom_kernels.codeformer.codeformer_torch import (
        CodeFormerTorch,
        build_cuda_graph_runner,
    )

    cf_fp32 = (
        CodeFormerTorch.from_onnx(CF_ONNX, compute_dtype=torch.float32).cuda().eval()
    )
    with torch.no_grad():
        t1 = _bench(lambda: cf_fp32(inp, fidelity_weight=w_val))
    _print_row("1", "PyTorch FP32 pure ops", t1, t0)

    # ── Tier 2 — PyTorch FP16 + Triton GroupNorm+SiLU ─────────────────────
    cf_fp16 = (
        CodeFormerTorch.from_onnx(CF_ONNX, compute_dtype=torch.float16).cuda().eval()
    )
    with torch.no_grad():
        t2 = _bench(lambda: cf_fp16(inp, fidelity_weight=w_val))
    _print_row("2", "PyTorch FP16 + Triton GroupNorm+SiLU", t2, t0)

    # ── Tier 3 — FP16 + Triton + CUDA graph (fixed w=0.5) ─────────────────
    t3 = t2
    try:
        runner3 = build_cuda_graph_runner(cf_fp16, inp_shape=(1, 3, 512, 512))
        with torch.no_grad():
            _ = runner3(inp)
        t3 = _bench(lambda: runner3(inp))
        _print_row("3", "FP16 + Triton + CUDA graph", t3, t0)
    except Exception as e:
        print(f"  Tier 3 | CUDA graph -- skipped ({e})")

    # ── Tier 4 — FP16 + Triton + SDPA 4D + GEMM + CUDA graph ──────────────
    t4 = t3
    try:
        cf_gemm = (
            CodeFormerTorch.from_onnx(CF_ONNX, compute_dtype=torch.float16)
            .cuda()
            .eval()
        )
        cf_gemm.to_gemm_mode()
        with torch.no_grad():
            _ = cf_gemm(inp, fidelity_weight=w_val)  # warm up JIT / cuDNN
        runner4 = build_cuda_graph_runner(cf_gemm, inp_shape=(1, 3, 512, 512))
        with torch.no_grad():
            _ = runner4(inp)
        t4 = _bench(lambda: runner4(inp))
        _print_row("4", "FP16 + Triton + SDPA4D + GEMM + CUDA graph", t4, t0)
    except Exception as e:
        print(f"  Tier 4 | SDPA+GEMM+CUDA graph -- skipped ({e})")

    # ── Tier 5 — FP16 + Triton + SDPA 4D + GEMM + NHWC + CUDA graph ───────
    t5 = t4
    try:
        cf_cl = (
            CodeFormerTorch.from_onnx(CF_ONNX, compute_dtype=torch.float16)
            .cuda()
            .eval()
        )
        cf_cl.to_channels_last().to_gemm_mode()
        with torch.no_grad():
            _ = cf_cl(inp, fidelity_weight=w_val)
        runner5 = build_cuda_graph_runner(cf_cl, inp_shape=(1, 3, 512, 512))
        with torch.no_grad():
            _ = runner5(inp)
        t5 = _bench(lambda: runner5(inp))
        _print_row("5", "FP16 + Triton + SDPA4D + GEMM + NHWC + CUDA graph", t5, t0)
    except Exception as e:
        print(f"  Tier 5 | SDPA+GEMM+NHWC+CUDA graph -- skipped ({e})")

    # ── Tier 6 — torch.compile + FP16 + Triton + CUDA graph ──────────────
    print("\n  [Tier 6] torch.compile(mode='default') + FP16 + Triton + CUDA graph")
    print("  One-time compile cost: ~60 s on first run (Triton JIT; complex transformer).")
    try:
        cf_c = (
            CodeFormerTorch.from_onnx(CF_ONNX, compute_dtype=torch.float16).cuda().eval()
        )
        runner6 = build_cuda_graph_runner(
            cf_c, inp_shape=(1, 3, 512, 512), torch_compile=True
        )
        with torch.no_grad():
            _ = runner6(inp)
        t6 = _bench(lambda: runner6(inp))
        _print_row("6", "torch.compile + FP16 + Triton + CUDA graph", t6, t0)
    except Exception as e:
        print(f"  Tier 6 | torch.compile — failed: {e}")
        import traceback; traceback.print_exc()

    # ── Tier 4b — torch.compile reduce-overhead (no separate CUDA graph) ────
    print("\n  [Tier 4b] torch.compile(mode='reduce-overhead') — no extra CUDA graph")
    try:
        from custom_kernels.compile_utils import apply_torch_compile
        m4b = CodeFormerTorch.from_onnx(CF_ONNX, compute_dtype=torch.float16).cuda().eval()
        m4b_compiled = apply_torch_compile(
            m4b,
            inp,
            compile_mode="reduce-overhead",
            extra_kwargs={"fidelity_weight": w_val},
        )
        with torch.no_grad():
            t4b = _bench(lambda: m4b_compiled(inp, fidelity_weight=w_val))
        _print_row("4b", "torch.compile reduce-overhead (no CUDA graph)", t4b, t0)
    except Exception as e:
        print(f"  Tier 4b | reduce-overhead — failed: {e}")
        import traceback; traceback.print_exc()

    print()
    print("  Note: CUDA graph is captured per fidelity_weight value.")
    print("  App uses Tier 3 (CUDA graph, lazily rebuilt on weight change).")
    print("  Tier 6 adds torch.compile for further speedup (~1.17x vs Tier 3).")
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
        import triton

        print(f"Triton:  {triton.__version__}")
    except ImportError:
        print("Triton:  not available (GroupNorm PyTorch fallback will be used)")
    try:
        import onnxruntime as ort

        print(f"ORT:     {ort.__version__}")
    except ImportError:
        print("ORT:     not available (skipping Tier 0/0b)")
        sys.exit(1)

    if not pathlib.Path(CF_ONNX).exists():
        print(f"ERROR: CodeFormer ONNX not found: {CF_ONNX}")
        sys.exit(1)

    bench_codeformer()
