"""
benchmark_inswapper.py
======================
Compares inference strategies for inswapper_128.fp16.onnx:

  Tier 0  -- ORT CUDA EP (baseline)
  Tier 0b -- ORT TensorRT EP (app default)
  Tier 1  -- PyTorch pure ops  (no custom kernel)
  Tier 2  -- PyTorch + Triton fused-AdaIN
  Tier 3  -- PyTorch + CUDA C++ fused-AdaIN kernel
  Tier 4  -- PyTorch pure ops + CUDA graph
  Tier 5  -- PyTorch + Triton fused-AdaIN + CUDA graph  [recommended single-tile]
  Tier 6  -- PyTorch + CUDA C++ kernel + CUDA graph
  Tier 7  -- channels-last (NHWC) + Triton NHWC AdaIN + CUDA graph
  Tier 8  -- im2col + cuBLAS GEMM (style blocks) + Triton AdaIN + CUDA graph

  --- Batched Pixel-Shift Resolution Tiers (Custom provider optimisation) ---
  Tier 9  -- Sequential B=4  (dim=2, 256px):  4 × single-tile calls
  Tier 10 -- Batched   B=4  (dim=2, 256px):  1 × 4-tile batched call
  Tier 11 -- Sequential B=9  (dim=3, 384px):  9 × single-tile calls
  Tier 12 -- Batched   B=9  (dim=3, 384px):  1 × 9-tile batched call
  Tier 13 -- Sequential B=16 (dim=4, 512px): 16 × single-tile calls
  Tier 14 -- Batched   B=16 (dim=4, 512px):  1 × 16-tile batched call

Also verifies numerical correctness against Tier 0 (ORT).
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
import os
import pathlib
import time
import numpy as np
import torch

ROOT = pathlib.Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

MODEL_PATH = str(ROOT / "model_assets" / "inswapper_128.fp16.onnx")
WARMUP = int(os.environ.get("WARMUP", 20))
RUNS = int(os.environ.get("ITERS", 300))


# ============================================================
# Helpers
# ============================================================
def timed(fn, runs, sync=True):
    if sync:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(runs):
        fn()
    if sync:
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / runs * 1_000  # ms


def stats(arr):
    return f"min={arr.min():.4f}  max={arr.max():.4f}  mean={arr.mean():.4f}"


# ============================================================
# Tier 0 -- ORT CUDA EP
# ============================================================
def make_ort_session():
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(
        MODEL_PATH,
        so,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        provider_options=[{"device_id": "0"}, {}],
    )
    return sess


def ort_run(sess, target_np, source_np):
    io = sess.io_binding()
    io.bind_cpu_input("target", target_np)
    io.bind_cpu_input("source", source_np)
    io.bind_output("output", "cuda")
    sess.run_with_iobinding(io)
    return np.array(io.copy_outputs_to_cpu()[0])


# ============================================================
# PyTorch model helpers
# ============================================================
def make_torch_model(use_custom_kernel: bool):
    from custom_kernels.inswapper_128.inswapper_torch import InSwapperTorch

    return InSwapperTorch(MODEL_PATH, use_custom_kernel=use_custom_kernel).cuda().eval()


# Temporarily disable Triton so we can measure CUDA C++ kernel in isolation


# ============================================================
# Main
# ============================================================
def main():
    rng = np.random.default_rng(0)
    target_np = rng.random((1, 3, 128, 128)).astype(np.float32)
    src_raw = rng.standard_normal((1, 512)).astype(np.float32)
    source_np = (src_raw / np.linalg.norm(src_raw)).astype(np.float32)

    target_gpu = torch.from_numpy(target_np).cuda()
    source_gpu = torch.from_numpy(source_np).cuda()

    sep = "=" * 72

    # ------------------------------------------------------------------ Tier 0
    print(sep)
    print("Tier 0 -- ORT CUDA EP (baseline)")
    sess = make_ort_session()
    for _ in range(WARMUP):
        ort_run(sess, target_np, source_np)
    out_ort = ort_run(sess, target_np, source_np)
    ms0 = timed(lambda: ort_run(sess, target_np, source_np), RUNS, sync=False)
    print(f"  Latency : {ms0:.3f} ms")
    print(f"  Output  : {stats(out_ort)}")

    # ------------------------------------------------------------------ Tier 0b
    print(sep)
    print("Tier 0b -- ORT TensorRT EP (app default)")
    ms0b = float("nan")
    import tempfile
    import shutil
    import onnxruntime as _ort_trt

    if "TensorrtExecutionProvider" not in _ort_trt.get_available_providers():
        print("  SKIP: TensorrtExecutionProvider not available")
    else:
        _trt_tmp = tempfile.mkdtemp(prefix="ort_trt_bench_")
        try:
            _trt_opts = {
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": _trt_tmp,
                "trt_timing_cache_enable": True,
                "trt_timing_cache_path": _trt_tmp,
                "trt_layer_norm_fp32_fallback": True,
                "trt_max_workspace_size": 8589934592,
                "trt_builder_optimization_level": 5,
            }
            _so = _ort_trt.SessionOptions()
            _so.graph_optimization_level = (
                _ort_trt.GraphOptimizationLevel.ORT_ENABLE_ALL
            )
            print("  Building TRT engine (first run may take 1-5 min)...")
            _sess_trt = _ort_trt.InferenceSession(
                MODEL_PATH,
                _so,
                providers=[
                    ("TensorrtExecutionProvider", _trt_opts),
                    ("CUDAExecutionProvider", {"device_id": "0"}),
                    ("CPUExecutionProvider", {}),
                ],
            )
            for _ in range(WARMUP):
                ort_run(_sess_trt, target_np, source_np)
            ms0b = timed(
                lambda: ort_run(_sess_trt, target_np, source_np), RUNS, sync=False
            )
            out_trt = ort_run(_sess_trt, target_np, source_np)
            diff_trt = np.abs(out_trt - out_ort).max()
            print(f"  Latency : {ms0b:.3f} ms  ({ms0 / ms0b:.2f}x vs ORT CUDA EP)")
            print(f"  Max |diff| vs ORT CUDA EP: {diff_trt:.5f}")
        except Exception as e:
            print(f"  FAILED: {e}")
        finally:
            shutil.rmtree(_trt_tmp, ignore_errors=True)

    # ------------------------------------------------------------------ Tier 1
    print(sep)
    print("Tier 1 -- PyTorch pure ops (no custom kernel)")
    m1 = make_torch_model(use_custom_kernel=False)
    with torch.no_grad():
        for _ in range(WARMUP):
            m1(target_gpu, source_gpu)
        out_t1 = m1(target_gpu, source_gpu).detach().cpu().numpy()
        ms1 = timed(lambda: m1(target_gpu, source_gpu), RUNS)
    diff1 = np.abs(out_t1 - out_ort).max()
    print(f"  Latency : {ms1:.3f} ms  ({ms0 / ms1:.2f}x vs ORT)")
    print(f"  Max |diff| vs ORT: {diff1:.5f}")

    # ------------------------------------------------------------------ Tier 2
    print(sep)
    print("Tier 2 -- PyTorch + Triton fused-AdaIN")
    from custom_kernels.triton_ops import TRITON_AVAILABLE

    if not TRITON_AVAILABLE:
        print("  Triton not available -- skipping")
        ms2, diff2 = float("nan"), float("nan")
    else:
        m2 = make_torch_model(use_custom_kernel=True)  # Triton is priority 1
        with torch.no_grad():
            for _ in range(WARMUP):
                m2(target_gpu, source_gpu)
            out_t2 = m2(target_gpu, source_gpu).detach().cpu().numpy()
            ms2 = timed(lambda: m2(target_gpu, source_gpu), RUNS)
        diff2 = np.abs(out_t2 - out_ort).max()
        print(f"  Latency : {ms2:.3f} ms  ({ms0 / ms2:.2f}x vs ORT)")
        print(f"  Max |diff| vs ORT: {diff2:.5f}")

    # ------------------------------------------------------------------ Tier 3
    print(sep)
    print("Tier 3 -- PyTorch + CUDA C++ fused-AdaIN kernel")
    # Patch out Triton temporarily so CUDA C++ kernel is used
    import custom_kernels.inswapper_128.inswapper_torch as _mod

    _saved_triton = _mod._TRITON_AVAILABLE
    _mod._TRITON_AVAILABLE = False
    try:
        m3 = make_torch_model(use_custom_kernel=True)
        with torch.no_grad():
            for _ in range(WARMUP):
                m3(target_gpu, source_gpu)
            out_t3 = m3(target_gpu, source_gpu).detach().cpu().numpy()
            ms3 = timed(lambda: m3(target_gpu, source_gpu), RUNS)
        diff3 = np.abs(out_t3 - out_ort).max()
        print(f"  Latency : {ms3:.3f} ms  ({ms0 / ms3:.2f}x vs ORT)")
        print(f"  Max |diff| vs ORT: {diff3:.5f}")
    except Exception as e:
        ms3, diff3 = float("nan"), float("nan")
        print(f"  FAILED: {e}")
    finally:
        _mod._TRITON_AVAILABLE = _saved_triton

    # ------------------------------------------------------------------ Tier 4
    print(sep)
    print("Tier 4 -- PyTorch pure ops + CUDA graph")
    from custom_kernels.inswapper_128.inswapper_torch import build_cuda_graph_runner

    print("  Capturing CUDA graph...")
    run4 = build_cuda_graph_runner(m1, target_gpu, source_gpu)
    torch.cuda.synchronize()
    out_t4 = run4(target_gpu, source_gpu).detach().cpu().numpy()
    ms4 = timed(lambda: run4(target_gpu, source_gpu), RUNS)
    diff4 = np.abs(out_t4 - out_ort).max()
    print(f"  Latency : {ms4:.3f} ms  ({ms0 / ms4:.2f}x vs ORT)")
    print(f"  Max |diff| vs ORT: {diff4:.5f}")

    # ------------------------------------------------------------------ Tier 5
    print(sep)
    print("Tier 5 -- PyTorch + Triton fused-AdaIN + CUDA graph  [recommended]")
    if not TRITON_AVAILABLE:
        print("  Triton not available -- skipping")
        ms5, diff5 = float("nan"), float("nan")
    else:
        print("  Capturing CUDA graph...")
        run5 = build_cuda_graph_runner(m2, target_gpu, source_gpu)
        torch.cuda.synchronize()
        out_t5 = run5(target_gpu, source_gpu).detach().cpu().numpy()
        ms5 = timed(lambda: run5(target_gpu, source_gpu), RUNS)
        diff5 = np.abs(out_t5 - out_ort).max()
        print(f"  Latency : {ms5:.3f} ms  ({ms0 / ms5:.2f}x vs ORT)")
        print(f"  Max |diff| vs ORT: {diff5:.5f}")

    # ------------------------------------------------------------------ Tier 6
    print(sep)
    print("Tier 6 -- PyTorch + CUDA C++ kernel + CUDA graph")
    try:
        _mod._TRITON_AVAILABLE = False
        print("  Capturing CUDA graph...")
        run6 = build_cuda_graph_runner(m3, target_gpu, source_gpu)
        torch.cuda.synchronize()
        out_t6 = run6(target_gpu, source_gpu).detach().cpu().numpy()
        ms6 = timed(lambda: run6(target_gpu, source_gpu), RUNS)
        diff6 = np.abs(out_t6 - out_ort).max()
        print(f"  Latency : {ms6:.3f} ms  ({ms0 / ms6:.2f}x vs ORT)")
        print(f"  Max |diff| vs ORT: {diff6:.5f}")
    except Exception as e:
        ms6, diff6 = float("nan"), float("nan")
        print(f"  FAILED: {e}")
    finally:
        _mod._TRITON_AVAILABLE = _saved_triton

    # ------------------------------------------------------------------ Tier 7
    print(sep)
    print(
        "Tier 7 -- channels-last (NHWC) + Triton NHWC AdaIN + CUDA graph  [TRT-fusion]"
    )
    ms7, diff7 = float("nan"), float("nan")
    if not TRITON_AVAILABLE:
        print("  Triton not available -- skipping")
    else:
        try:
            m7 = make_torch_model(use_custom_kernel=True)
            m7.to_channels_last()  # convert all Conv2d weights to NHWC
            print("  Capturing CUDA graph (channels-last NHWC mode)...")
            run7 = build_cuda_graph_runner(m7, target_gpu, source_gpu)
            torch.cuda.synchronize()
            out_t7 = run7(target_gpu, source_gpu).detach().cpu().numpy()
            ms7 = timed(lambda: run7(target_gpu, source_gpu), RUNS)
            diff7 = np.abs(out_t7 - out_ort).max()
            print(f"  Latency : {ms7:.3f} ms  ({ms0 / ms7:.2f}x vs ORT)")
            print(f"  Max |diff| vs ORT: {diff7:.5f}")
        except Exception as e:
            import traceback

            traceback.print_exc()
            print(f"  FAILED: {e}")

    # ------------------------------------------------------------------ Tier 8
    print(sep)
    print("Tier 8 -- im2col + cuBLAS GEMM (style blocks) + Triton AdaIN + CUDA graph")
    ms8, diff8 = float("nan"), float("nan")
    if not TRITON_AVAILABLE:
        print("  Triton not available -- skipping")
    else:
        try:
            m8 = make_torch_model(use_custom_kernel=True)
            m8.to_gemm_mode()  # switch 12 style-block convs to F.unfold + torch.mm
            print("  Capturing CUDA graph (GEMM mode)...")
            run8 = build_cuda_graph_runner(m8, target_gpu, source_gpu)
            torch.cuda.synchronize()
            out_t8 = run8(target_gpu, source_gpu).detach().cpu().numpy()
            ms8 = timed(lambda: run8(target_gpu, source_gpu), RUNS)
            diff8 = np.abs(out_t8 - out_ort).max()
            print(f"  Latency : {ms8:.3f} ms  ({ms0 / ms8:.2f}x vs ORT)")
            print(f"  Max |diff| vs ORT: {diff8:.5f}")
        except Exception as e:
            import traceback

            traceback.print_exc()
            print(f"  FAILED: {e}")

    # ------------------------------------------------------------------ Tier 8b (Phase 3: cuBLASLt)
    print(sep)
    print(
        "Tier 8b -- cuBLASLt HGEMM + fused bias + Triton AdaIN+residual + CUDA graph  [Phase 3]"
    )
    ms8b, diff8b = float("nan"), float("nan")
    if not TRITON_AVAILABLE:
        print("  Triton not available -- skipping")
    else:
        try:
            m8b = make_torch_model(use_custom_kernel=True)
            m8b.to_gemm_mode()
            m8b.to_cublaslt_mode()  # Phase 3: cuBLASLt HGEMM with fused bias
            print("  Capturing CUDA graph (cuBLASLt mode)...")
            run8b = build_cuda_graph_runner(m8b, target_gpu, source_gpu)
            torch.cuda.synchronize()
            out_t8b = run8b(target_gpu, source_gpu).detach().cpu().numpy()
            ms8b = timed(lambda: run8b(target_gpu, source_gpu), RUNS)
            diff8b = np.abs(out_t8b - out_ort).max()
            print(f"  Latency : {ms8b:.3f} ms  ({ms0 / ms8b:.2f}x vs ORT)")
            print(f"  Max |diff| vs ORT: {diff8b:.5f}")
        except Exception as e:
            import traceback

            traceback.print_exc()
            print(f"  FAILED (cuBLASLt unavailable or extension not compiled): {e}")

    # ------------------------------------------------------------------ Tier 18 (torch.compile + Triton AdaIN + CUDA graph)
    # NOTE: torch.compile crashes the process (MLIR optimizer segfault in libtriton.pyd)
    # for inswapper_128 on Windows/sm_89 due to the custom Triton AdaIN fused kernel.
    # Set INSWAPPER_TORCH_COMPILE=1 to attempt (process may crash).
    print(sep)
    print("Tier 18 -- torch.compile + Triton AdaIN + CUDA graph  [compile+CG, single-tile]")
    ms18, diff18 = float("nan"), float("nan")
    if not TRITON_AVAILABLE:
        print("  Triton not available -- skipping")
    elif not int(os.environ.get("INSWAPPER_TORCH_COMPILE", "0")):
        print("  Skipped — torch.compile crashes (MLIR segfault) on this model on Windows/sm_89.")
        print("  Set INSWAPPER_TORCH_COMPILE=1 to attempt.")
    else:
        try:
            print("  [Tier 18] torch.compile(mode='default') — one-time ~30 s Triton JIT.")
            m18 = make_torch_model(use_custom_kernel=True)
            run18 = build_cuda_graph_runner(m18, target_gpu, source_gpu, torch_compile=True)
            torch.cuda.synchronize()
            out_t18 = run18(target_gpu, source_gpu).detach().cpu().numpy()
            ms18 = timed(lambda: run18(target_gpu, source_gpu), RUNS)
            diff18 = np.abs(out_t18 - out_ort).max()
            print(f"  Latency : {ms18:.3f} ms  ({ms0 / ms18:.2f}x vs ORT)")
            print(f"  Max |diff| vs ORT: {diff18:.5f}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  FAILED: {e}")

    # ------------------------------------------------------------------ Tier 18b (reduce-overhead, no CUDA graph)
    print(sep)
    print("Tier 18b -- torch.compile(reduce-overhead) — no extra CUDA graph  [single-tile]")
    ms18b, diff18b = float("nan"), float("nan")
    if not TRITON_AVAILABLE:
        print("  Triton not available -- skipping")
    elif not int(os.environ.get("INSWAPPER_TORCH_COMPILE", "0")):
        print("  Skipped — torch.compile crashes (MLIR segfault) on this model on Windows/sm_89.")
        print("  Set INSWAPPER_TORCH_COMPILE=1 to attempt.")
    else:
        try:
            from custom_kernels.compile_utils import apply_torch_compile
            print("  [Tier 18b] torch.compile(mode='reduce-overhead') — one-time ~30 s Triton JIT.")
            m18b = make_torch_model(use_custom_kernel=True)
            m18b_compiled = apply_torch_compile(
                m18b,
                target_gpu,
                extra_args=(source_gpu,),
                compile_mode="reduce-overhead",
            )
            with torch.no_grad():
                out_t18b = m18b_compiled(target_gpu, source_gpu).detach().cpu().numpy()
            ms18b = timed(lambda: m18b_compiled(target_gpu, source_gpu), RUNS)
            diff18b = np.abs(out_t18b - out_ort).max()
            print(f"  Latency : {ms18b:.3f} ms  ({ms0 / ms18b:.2f}x vs ORT)")
            print(f"  Max |diff| vs ORT: {diff18b:.5f}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  FAILED: {e}")

    # ------------------------------------------------------------------ Batched pixel-shift tiers
    from custom_kernels.triton_ops import TRITON_AVAILABLE as _T

    if not _T:
        print(f"\n{sep}")
        print("Tiers 9-17 (batched pixel-shift / GEMM) — SKIP: Triton not available")
        ms9 = ms10 = ms11 = ms12 = ms13 = ms14 = float("nan")
        ms15 = ms16 = ms17 = float("nan")
        diff_b4 = diff_b9 = diff_b16 = float("nan")
        diff_g4 = diff_g9 = diff_g16 = float("nan")
    else:
        # Best single-tile model for Custom provider: m2 (Triton AdaIN, eager)
        # The CUDA graph model (m2 + run5) is for B=1; for batched we use eager m2.

        def run_sequential(model, target, source, B):
            """Simulate dim*dim sequential single-tile calls (ORT-style)."""
            for _ in range(B):
                with torch.no_grad():
                    model(target, source)
            torch.cuda.synchronize()

        def run_batched(model, target_B, source):
            """Single batched forward for B tiles."""
            with torch.no_grad():
                model(target_B, source)
            torch.cuda.synchronize()

        for B, dim, tier_seq, tier_bat in [
            (4, 2, "9 ", "10"),
            (9, 3, "11", "12"),
            (16, 4, "13", "14"),
        ]:
            # Create a batch of B random tiles (all with same source)
            rng_b = torch.Generator(device="cuda").manual_seed(B)
            target_B = torch.rand(
                B, 3, 128, 128, device="cuda", dtype=torch.float32, generator=rng_b
            )
            source_1 = source_gpu.clone()

            # Warm-up
            with torch.no_grad():
                for _ in range(WARMUP):
                    m2(target_B[0:1], source_1)
                for _ in range(WARMUP):
                    m2(target_B, source_1)
            torch.cuda.synchronize()

            # Sequential: B separate single-tile calls
            print(sep)
            print(
                f"Tier {tier_seq} -- Sequential {B} single-tile calls (dim={dim}, {dim * 128}px equivalent)"
            )
            ms_seq = timed(lambda: run_sequential(m2, target_B[0:1], source_1, B), RUNS)
            print(
                f"  Latency (total {B} calls): {ms_seq:.3f} ms  ({ms_seq / B:.3f} ms/call)"
            )

            # Batched: one forward with B tiles
            print(sep)
            print(
                f"Tier {tier_bat} -- Batched B={B} (dim={dim}, {dim * 128}px equivalent)  [NEW]"
            )
            ms_bat = timed(lambda: run_batched(m2, target_B, source_1), RUNS)
            # Check accuracy: batched[0] vs single-tile[0]
            with torch.no_grad():
                out_single = m2(target_B[0:1], source_1).cpu().numpy()
                out_batch = m2(target_B, source_1)[0:1].cpu().numpy()
            diff_bat = float(np.abs(out_single - out_batch).max())
            speedup = ms_seq / ms_bat
            print(
                f"  Latency (1 batched call): {ms_bat:.3f} ms  ({speedup:.2f}x vs sequential)"
            )
            print(f"  Max |diff| tile[0] single vs batched: {diff_bat:.6f}")

            if B == 4:
                ms9, ms10, diff_b4 = ms_seq, ms_bat, diff_bat
            elif B == 9:
                ms11, ms12, diff_b9 = ms_seq, ms_bat, diff_bat
            else:
                ms13, ms14, diff_b16 = ms_seq, ms_bat, diff_bat

        # ---- GEMM-mode batched (Phase 1 optimisation: torch.matmul for B>1) ----
        print(sep)
        print("--- GEMM-mode batched (Phase 1: torch.matmul, no cuDNN fallback) ---")
        ms15 = ms16 = ms17 = float("nan")
        diff_g4 = diff_g9 = diff_g16 = float("nan")
        try:
            m_gemm = make_torch_model(use_custom_kernel=True)
            m_gemm.to_gemm_mode()
            with torch.no_grad():
                for _ in range(WARMUP):
                    m_gemm(target_gpu, source_gpu)
            torch.cuda.synchronize()

            for B, dim, ms_slot, diff_slot in [
                (4, 2, "ms15", "diff_g4"),
                (9, 3, "ms16", "diff_g9"),
                (16, 4, "ms17", "diff_g16"),
            ]:
                rng_b = torch.Generator(device="cuda").manual_seed(B)
                target_B = torch.rand(
                    B, 3, 128, 128, device="cuda", dtype=torch.float32, generator=rng_b
                )
                source_1 = source_gpu.clone()
                with torch.no_grad():
                    for _ in range(WARMUP):
                        m_gemm(target_B, source_1)
                torch.cuda.synchronize()
                ms_bat_g = timed(
                    lambda: (m_gemm(target_B, source_1), torch.cuda.synchronize()),
                    RUNS,
                    sync=False,
                )
                with torch.no_grad():
                    out_single_g = m_gemm(target_B[0:1], source_1).cpu().numpy()
                    out_batch_g = m_gemm(target_B, source_1)[0:1].cpu().numpy()
                diff_g = float(np.abs(out_single_g - out_batch_g).max())
                print(
                    f"  GEMM B={B:2d} (dim={dim}, {dim * 128}px): "
                    f"{ms_bat_g:.3f} ms  diff={diff_g:.6f}"
                )
                if B == 4:
                    ms15 = ms_bat_g
                    diff_g4 = diff_g
                elif B == 9:
                    ms16 = ms_bat_g
                    diff_g9 = diff_g
                else:
                    ms17 = ms_bat_g
                    diff_g16 = diff_g
        except Exception as e:
            import traceback

            traceback.print_exc()
            print(f"  GEMM batched tiers FAILED: {e}")

    # ------------------------------------------------------------------ Summary
    print(sep)
    print("SUMMARY")
    print(f"  {'Tier':<65} {'ms':>7} {'speedup':>9} {'max|diff|':>12}")
    print(f"  {'-' * 65} {'-' * 7} {'-' * 9} {'-' * 12}")
    _nan = float("nan")
    rows = [
        ("0   ORT CUDA EP (baseline)", ms0, 1.0, 0.0),
        (
            "0b  ORT TensorRT EP (app default)",
            ms0b,
            ms0 / ms0b if ms0b == ms0b else _nan,
            0.0,
        ),
        ("1   PyTorch pure ops", ms1, ms0 / ms1, diff1),
        ("2   PyTorch + Triton AdaIN", ms2, ms0 / ms2, diff2),
        ("3   PyTorch + CUDA C++ AdaIN", ms3, ms0 / ms3, diff3),
        ("4   PyTorch pure ops + CUDA graph", ms4, ms0 / ms4, diff4),
        (
            "5   PyTorch + Triton AdaIN + CUDA graph  [best single-tile]",
            ms5,
            ms0 / ms5,
            diff5,
        ),
        ("6   PyTorch + CUDA C++ AdaIN + CUDA graph", ms6, ms0 / ms6, diff6),
        ("7   NHWC + Triton NHWC AdaIN + CUDA graph", ms7, ms0 / ms7, diff7),
        (
            "8   im2col+cuBLAS GEMM + Triton AdaIN+res + CUDA graph",
            ms8,
            ms0 / ms8,
            diff8,
        ),
        (
            "8b  cuBLASLt HGEMM+bias + Triton AdaIN+res + CUDA graph [Phase 3]",
            ms8b,
            ms0 / ms8b if ms8b == ms8b else _nan,
            diff8b,
        ),
        (
            "18  torch.compile + Triton AdaIN + CUDA graph  [compile+CG]",
            ms18,
            ms0 / ms18 if ms18 == ms18 else _nan,
            diff18,
        ),
        (
            "18b torch.compile reduce-overhead (no CUDA graph)",
            ms18b,
            ms0 / ms18b if ms18b == ms18b else _nan,
            diff18b,
        ),
        ("--- Batched pixel-shift (256px / dim=2) ---", _nan, _nan, _nan),
        ("9   Sequential B=4  (4× single calls)", ms9, ms0 / (ms9 or _nan), _nan),
        (
            "10  Batched   B=4  (1× 4-tile call)  [NEW]",
            ms10,
            ms0 / (ms10 or _nan),
            diff_b4,
        ),
        ("--- Batched pixel-shift (384px / dim=3) ---", _nan, _nan, _nan),
        ("11  Sequential B=9  (9× single calls)", ms11, ms0 / (ms11 or _nan), _nan),
        (
            "12  Batched   B=9  (1× 9-tile call)  [NEW]",
            ms12,
            ms0 / (ms12 or _nan),
            diff_b9,
        ),
        ("--- Batched pixel-shift (512px / dim=4) ---", _nan, _nan, _nan),
        ("13  Sequential B=16 (16× single calls)", ms13, ms0 / (ms13 or _nan), _nan),
        ("14  Batched   B=16 (1× 16-tile call)", ms14, ms0 / (ms14 or _nan), diff_b16),
        ("--- GEMM-mode batched (Phase 1: batched matmul) ---", _nan, _nan, _nan),
        ("15  GEMM Batched B=4  (torch.matmul)", ms15, ms0 / (ms15 or _nan), diff_g4),
        ("16  GEMM Batched B=9  (torch.matmul)", ms16, ms0 / (ms16 or _nan), diff_g9),
        ("17  GEMM Batched B=16 (torch.matmul)", ms17, ms0 / (ms17 or _nan), diff_g16),
    ]
    for label, ms, spd, dif in rows:
        if ms != ms:  # nan — section header or skipped
            print(f"  {label:<65}")
        else:
            dif_str = f"{dif:12.5f}" if dif == dif else f"{'N/A':>12}"
            print(f"  {label:<65} {ms:7.3f} {spd:9.2f}x {dif_str}")
    print(sep)

    # Save results
    out_path = pathlib.Path(__file__).parent / "benchmark_results.txt"
    with open(out_path, "w") as f:
        f.write("InSwapper-128 Benchmark Results\n")
        f.write(f"{'Tier':<65} {'ms':>7} {'speedup':>9} {'max|diff|':>12}\n")
        f.write(f"{'-' * 65} {'-' * 7} {'-' * 9} {'-' * 12}\n")
        for label, ms, spd, dif in rows:
            if ms != ms:
                f.write(f"{label}\n")
            else:
                dif_str = f"{dif:12.5f}" if dif == dif else f"{'N/A':>12}"
                f.write(f"{label:<65} {ms:7.3f} {spd:9.2f}x {dif_str}\n")
    print(f"[Saved -> {out_path}]")


if __name__ == "__main__":
    main()
