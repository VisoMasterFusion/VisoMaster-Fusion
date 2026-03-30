"""
Benchmark — ReF-LDM custom kernels vs ORT baseline.

Tests three model components separately:
  VAE Encoder  : (1,3,512,512)f32  -> (1,8,64,64)f32
  VAE Decoder  : (1,8,64,64)f32   -> (1,3,512,512)f32
  UNet denoiser: (1,16,64,64)f32  + K/V -> (1,8,64,64)f32

Usage (from repo root):
    .venv\\Scripts\\python custom_kernels\\ref_ldm\\benchmark_ref_ldm.py

Optional env vars:
    ONNX_DIR=model_assets          path to ONNX model files
    WARMUP=50                      warm-up iterations
    ITERS=200                      timed iterations
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
sys.path.insert(0, str(ROOT))   # needed by all sub-benchmarks when run as subprocess

ONNX_DIR = pathlib.Path(os.environ.get("ONNX_DIR", ROOT / "model_assets"))
WARMUP = int(os.environ.get("WARMUP", 50))
ITERS = int(os.environ.get("ITERS", 200))

ENC_ONNX = str(ONNX_DIR / "ref_ldm_vae_encoder.onnx")
DEC_ONNX = str(ONNX_DIR / "ref_ldm_vae_decoder.onnx")
UNET_ONNX = str(ONNX_DIR / "ref_ldm_unet_external_kv.onnx")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _ms(elapsed_s: float, iters: int) -> float:
    return elapsed_s / iters * 1000.0


def _bench(fn, warmup=WARMUP, iters=ITERS) -> float:
    """Return ms/iter (GPU-timed)."""
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


import tempfile as _tempfile
import shutil as _shutil


def _make_trt_session(onnx_path: str):
    """Create an ORT TensorRT EP session using a fresh temporary cache dir.

    Returns (session, trt_tmp_dir) on success, or (None, None) on failure.
    The caller is responsible for calling shutil.rmtree(trt_tmp_dir) when done.
    """
    import onnxruntime as ort

    if (
        "TensorrtExecutionProvider" not in ort.get_available_providers()
        or os.environ.get("SKIP_TRT") == "1"
    ):
        return None, None

    trt_tmp = _tempfile.mkdtemp(prefix="ort_trt_bench_")
    try:
        opts = {
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": trt_tmp,
            "trt_timing_cache_enable": True,
            "trt_timing_cache_path": trt_tmp,
            "trt_layer_norm_fp32_fallback": True,
            "trt_max_workspace_size": 8589934592,
            "trt_builder_optimization_level": 5,
        }
        print(
            "  Tier 0b | ORT TRT EP — building engine (first run may take 1-5 min)..."
        )
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess = ort.InferenceSession(
            onnx_path,
            so,
            providers=[
                ("TensorrtExecutionProvider", opts),
                ("CUDAExecutionProvider", {"device_id": "0"}),
                ("CPUExecutionProvider", {}),
            ],
        )
        return sess, trt_tmp
    except Exception as _e:
        _shutil.rmtree(trt_tmp, ignore_errors=True)
        print(f"  [WARN] TRT session build failed: {_e}")
        return None, None


def _ort_trt_session(onnx_path: str, ctx_name: str):
    """Compatibility wrapper: builds a fresh TRT session using tempfile.
    The ctx_name parameter is ignored (kept for API compat).
    Returns (session, error_string_or_None).
    """
    sess, trt_tmp = _make_trt_session(onnx_path)
    if sess is None:
        import onnxruntime as ort

        if "TensorrtExecutionProvider" not in ort.get_available_providers():
            return (
                None,
                "TensorrtExecutionProvider not in ort.get_available_providers()",
            )
        return None, "TRT engine build failed"
    # Store trt_tmp on sess so callers can clean it up
    sess._trt_tmp = trt_tmp
    return sess, None


def _print_row(tier, label, ms_val, ref_ms):
    speedup = ref_ms / ms_val if ms_val > 0 else float("nan")
    print(f"  Tier {tier} | {label:<42} | {ms_val:8.2f} ms | {speedup:6.2f}x")


# ---------------------------------------------------------------------------
# VAE Encoder benchmark
# ---------------------------------------------------------------------------
def bench_encoder():
    print("\n=== VAE Encoder (1,3,512,512) -> (1,8,64,64) ===")
    print(f"  warm-up={WARMUP}, iters={ITERS}\n")
    print(f"  {'Tier':<6} | {'Method':<42} | {'ms':>8} | {'speedup':>7}")
    print(f"  {'-' * 6}-+-{'-' * 42}-+-{'-' * 8}-+-{'-' * 7}")

    inp = torch.randn(1, 3, 512, 512, dtype=torch.float32, device="cuda")

    # Tier 0  — ORT FP32 CUDA EP
    sess0 = _ort_session(ENC_ONNX, "CUDAExecutionProvider")
    inp_np = inp.cpu().numpy()
    in_name = sess0.get_inputs()[0].name
    out_name = sess0.get_outputs()[0].name
    t0 = _bench(lambda: sess0.run([out_name], {in_name: inp_np}))
    _print_row("0", "ORT FP32 CUDA EP", t0, t0)

    # Tier 0b — ORT TensorRT EP
    t0b = t0
    sess0b, err0b = _ort_trt_session(ENC_ONNX, "ref_ldm_vae_encoder_ctx.onnx")
    if sess0b is None:
        print(f"  Tier 0b | ORT TRT EP — skipped ({err0b})")
    else:
        try:
            for _ in range(WARMUP):
                sess0b.run([out_name], {in_name: inp_np})
            t0b = _bench(lambda: sess0b.run([out_name], {in_name: inp_np}))
            _print_row("0b", "ORT TRT EP FP32", t0b, t0)
        except Exception as e:
            print(f"  Tier 0b | TRT EP — failed: {e}")
        finally:
            _shutil.rmtree(getattr(sess0b, "_trt_tmp", None) or "", ignore_errors=True)

    # Tier 1  — PyTorch FP32
    from custom_kernels.ref_ldm.ref_ldm_torch import RefLDMEncoderTorch

    enc_fp32 = (
        RefLDMEncoderTorch.from_onnx(ENC_ONNX, compute_dtype=torch.float32)
        .cuda()
        .eval()
    )
    with torch.no_grad():
        t1 = _bench(lambda: enc_fp32(inp))
    _print_row("1", "PyTorch FP32 pure ops", t1, t0)

    # Tier 2  — PyTorch FP16 + Triton GroupNorm+SiLU
    enc_fp16 = (
        RefLDMEncoderTorch.from_onnx(ENC_ONNX, compute_dtype=torch.float16)
        .cuda()
        .eval()
    )
    with torch.no_grad():
        t2 = _bench(lambda: enc_fp16(inp))
    _print_row("2", "PyTorch FP16 + Triton GroupNorm+SiLU", t2, t0)

    # Tier 3  — FP16 + Triton + CUDA graph
    from custom_kernels.ref_ldm.ref_ldm_torch import build_cuda_graph_runner

    runner = build_cuda_graph_runner(enc_fp16, inp_shape=(1, 3, 512, 512))
    t3 = _bench(lambda: runner(inp))
    _print_row("3", "PyTorch FP16 + Triton + CUDA graph", t3, t0)

    # Tier 4  — FP16 + Triton + CUDA graph + NHWC (channels-last)
    enc_cl = (
        RefLDMEncoderTorch.from_onnx(ENC_ONNX, compute_dtype=torch.float16)
        .cuda()
        .eval()
    )
    enc_cl.to_channels_last()
    runner_cl = build_cuda_graph_runner(enc_cl, inp_shape=(1, 3, 512, 512))
    t4 = _bench(lambda: runner_cl(inp))
    _print_row("4", "FP16 + Triton + CUDA graph + NHWC", t4, t0)

    # Tier 5 — torch.compile(mode='default') + CUDA graph
    print("\n  [Tier 5] torch.compile(mode='default') + CUDA graph")
    print("  One-time compile cost: ~30–60 s on first run (Triton JIT).")
    try:
        enc_c5 = (
            RefLDMEncoderTorch.from_onnx(ENC_ONNX, compute_dtype=torch.float16)
            .cuda()
            .eval()
        )
        runner_c5 = build_cuda_graph_runner(enc_c5, inp_shape=(1, 3, 512, 512), torch_compile=True)
        t5 = _bench(lambda: runner_c5(inp))
        _print_row("5", "FP16 + Triton + CUDA graph + torch.compile", t5, t0)
    except Exception as e:
        print(f"  Tier 5 | torch.compile — failed ({type(e).__name__}: {e})")
        import traceback; traceback.print_exc()

    # Tier 5b — torch.compile reduce-overhead (no separate CUDA graph)
    print("\n  [Tier 5b] torch.compile(mode='reduce-overhead') — no extra CUDA graph")
    if not int(os.environ.get("REFLDM_TORCH_COMPILE", "0")):
        print("  Skipped — reduce-overhead may crash on this model on Windows/sm_89.")
        print("  Set REFLDM_TORCH_COMPILE=1 to attempt.")
    else:
        try:
            from custom_kernels.compile_utils import apply_torch_compile
            enc_5b = (
                RefLDMEncoderTorch.from_onnx(ENC_ONNX, compute_dtype=torch.float16)
                .cuda()
                .eval()
            )
            enc_5b_compiled = apply_torch_compile(
                enc_5b,
                inp,
                compile_mode="reduce-overhead",
            )
            with torch.no_grad():
                t5b = _bench(lambda: enc_5b_compiled(inp))
            _print_row("5b", "torch.compile reduce-overhead (no CUDA graph)", t5b, t0)
        except Exception as e:
            print(f"  Tier 5b | reduce-overhead — failed: {e}")
            import traceback; traceback.print_exc()

    print()
    return t0, t0b


# ---------------------------------------------------------------------------
# VAE Decoder benchmark
# ---------------------------------------------------------------------------
def bench_decoder():
    print("\n=== VAE Decoder (1,8,64,64) -> (1,3,512,512) ===")
    print(f"  warm-up={WARMUP}, iters={ITERS}\n")
    print(f"  {'Tier':<6} | {'Method':<42} | {'ms':>8} | {'speedup':>7}")
    print(f"  {'-' * 6}-+-{'-' * 42}-+-{'-' * 8}-+-{'-' * 7}")

    lat = torch.randn(1, 8, 64, 64, dtype=torch.float32, device="cuda")

    # Tier 0  — ORT FP32 CUDA EP
    sess0 = _ort_session(DEC_ONNX, "CUDAExecutionProvider")
    lat_np = lat.cpu().numpy()
    in_name = sess0.get_inputs()[0].name
    out_name = sess0.get_outputs()[0].name
    t0 = _bench(lambda: sess0.run([out_name], {in_name: lat_np}))
    _print_row("0", "ORT FP32 CUDA EP", t0, t0)

    # Tier 0b — ORT TensorRT EP
    t0b = t0
    sess0b, err0b = _ort_trt_session(DEC_ONNX, "ref_ldm_vae_decoder_ctx.onnx")
    if sess0b is None:
        print(f"  Tier 0b | ORT TRT EP — skipped ({err0b})")
    else:
        try:
            for _ in range(WARMUP):
                sess0b.run([out_name], {in_name: lat_np})
            t0b = _bench(lambda: sess0b.run([out_name], {in_name: lat_np}))
            _print_row("0b", "ORT TRT EP FP32", t0b, t0)
        except Exception as e:
            print(f"  Tier 0b | TRT EP — failed: {e}")
        finally:
            _shutil.rmtree(getattr(sess0b, "_trt_tmp", None) or "", ignore_errors=True)

    # Tier 1  — PyTorch FP32
    from custom_kernels.ref_ldm.ref_ldm_torch import RefLDMDecoderTorch

    dec_fp32 = (
        RefLDMDecoderTorch.from_onnx(DEC_ONNX, compute_dtype=torch.float32)
        .cuda()
        .eval()
    )
    with torch.no_grad():
        t1 = _bench(lambda: dec_fp32(lat))
    _print_row("1", "PyTorch FP32 pure ops", t1, t0)

    # Tier 2  — PyTorch FP16 + Triton GroupNorm+SiLU
    dec_fp16 = (
        RefLDMDecoderTorch.from_onnx(DEC_ONNX, compute_dtype=torch.float16)
        .cuda()
        .eval()
    )
    with torch.no_grad():
        t2 = _bench(lambda: dec_fp16(lat))
    _print_row("2", "PyTorch FP16 + Triton GroupNorm+SiLU", t2, t0)

    # Tier 3  — FP16 + Triton + CUDA graph
    from custom_kernels.ref_ldm.ref_ldm_torch import build_cuda_graph_runner

    runner = build_cuda_graph_runner(dec_fp16, inp_shape=(1, 8, 64, 64))
    t3 = _bench(lambda: runner(lat))
    _print_row("3", "PyTorch FP16 + Triton + CUDA graph", t3, t0)

    # Tier 4  — FP16 + Triton + CUDA graph + NHWC (channels-last)
    dec_cl = (
        RefLDMDecoderTorch.from_onnx(DEC_ONNX, compute_dtype=torch.float16)
        .cuda()
        .eval()
    )
    dec_cl.to_channels_last()
    runner_cl = build_cuda_graph_runner(dec_cl, inp_shape=(1, 8, 64, 64))
    t4 = _bench(lambda: runner_cl(lat))
    _print_row("4", "FP16 + Triton + CUDA graph + NHWC", t4, t0)

    # Tier 5 — torch.compile(mode='default') + CUDA graph
    print("\n  [Tier 5] torch.compile(mode='default') + CUDA graph")
    print("  One-time compile cost: ~30–60 s on first run (Triton JIT).")
    try:
        dec_c5 = (
            RefLDMDecoderTorch.from_onnx(DEC_ONNX, compute_dtype=torch.float16)
            .cuda()
            .eval()
        )
        runner_c5 = build_cuda_graph_runner(dec_c5, inp_shape=(1, 8, 64, 64), torch_compile=True)
        t5 = _bench(lambda: runner_c5(lat))
        _print_row("5", "FP16 + Triton + CUDA graph + torch.compile", t5, t0)
    except Exception as e:
        print(f"  Tier 5 | torch.compile — failed ({type(e).__name__}: {e})")
        import traceback; traceback.print_exc()

    # Tier 5b — torch.compile reduce-overhead (no separate CUDA graph)
    print("\n  [Tier 5b] torch.compile(mode='reduce-overhead') — no extra CUDA graph")
    if not int(os.environ.get("REFLDM_TORCH_COMPILE", "0")):
        print("  Skipped — reduce-overhead may crash on this model on Windows/sm_89.")
        print("  Set REFLDM_TORCH_COMPILE=1 to attempt.")
    else:
        try:
            from custom_kernels.compile_utils import apply_torch_compile
            dec_5b = (
                RefLDMDecoderTorch.from_onnx(DEC_ONNX, compute_dtype=torch.float16)
                .cuda()
                .eval()
            )
            dec_5b_compiled = apply_torch_compile(
                dec_5b,
                lat,
                compile_mode="reduce-overhead",
            )
            with torch.no_grad():
                t5b = _bench(lambda: dec_5b_compiled(lat))
            _print_row("5b", "torch.compile reduce-overhead (no CUDA graph)", t5b, t0)
        except Exception as e:
            print(f"  Tier 5b | reduce-overhead — failed: {e}")
            import traceback; traceback.print_exc()

    print()
    return t0, t0b


# ---------------------------------------------------------------------------
# UNet benchmark
# ---------------------------------------------------------------------------
def _build_dummy_kv_map():
    """Build a representative K/V map for the UNet benchmark.

    UNet: mc=160, mult=(1,2,2,4), nhc=32  ->  n_heads = ch // 32
      Level 1 (32×32, 320ch):  input 4-5,  output 3-5  -> n_heads=10
      Level 2 (16×16, 320ch):  input 7-8,  output 6-8  -> n_heads=10
      Level 3 (8×8,  640ch):   input 10-11, middle, output 0-2 -> n_heads=20
    External seq_len is arbitrary (64 reference tokens used here).
    """
    # (path, n_heads)
    ch_per_head = 32
    seq_len = 64
    attn_blocks = [
        # Level 1 — 320ch -> 10 heads
        ("input_blocks.4.1.attention", 10),
        ("input_blocks.5.1.attention", 10),
        # Level 2 — 320ch -> 10 heads
        ("input_blocks.7.1.attention", 10),
        ("input_blocks.8.1.attention", 10),
        # Level 3 — 640ch -> 20 heads
        ("input_blocks.10.1.attention", 20),
        ("input_blocks.11.1.attention", 20),
        ("middle_block.1.attention", 20),
        ("output_blocks.0.1.attention", 20),
        ("output_blocks.1.1.attention", 20),
        ("output_blocks.2.1.attention", 20),
        # Level 2 (decoder) — 320ch -> 10 heads
        ("output_blocks.3.1.attention", 10),
        ("output_blocks.4.1.attention", 10),
        ("output_blocks.5.1.attention", 10),
    ]
    kv_map = {}
    for path, n_heads in attn_blocks:
        kv_map[path] = {
            "k": torch.randn(
                n_heads, ch_per_head, seq_len, device="cuda", dtype=torch.float32
            ),
            "v": torch.randn(
                n_heads, ch_per_head, seq_len, device="cuda", dtype=torch.float32
            ),
        }
    return kv_map


def bench_unet():
    print("\n=== UNet denoiser (1,16,64,64) + K/V -> (1,8,64,64) ===")
    print(f"  warm-up={WARMUP}, iters={ITERS}\n")
    print(f"  {'Tier':<6} | {'Method':<42} | {'ms':>8} | {'speedup':>7}")
    print(f"  {'-' * 6}-+-{'-' * 42}-+-{'-' * 8}-+-{'-' * 7}")

    x = torch.randn(1, 16, 64, 64, dtype=torch.float32, device="cuda")
    ts = torch.tensor([500], dtype=torch.int64, device="cuda")
    kv_map = _build_dummy_kv_map()

    # Tier 0  — ORT FP32 CUDA EP (simplified: no K/V, baseline timing only)
    sess0 = _ort_session(UNET_ONNX, "CUDAExecutionProvider")
    onnx_inputs = sess0.get_inputs()
    _in_names = [i.name for i in onnx_inputs]

    def _build_ort_feeds():
        feeds = {
            "x_noisy_plus_lq_latent": x.cpu().numpy(),
            "timesteps": ts.cpu().numpy(),
            "is_ref_flag_input": np.array([True], dtype=bool),
            "use_reference_exclusive_path_globally_input": np.array([True], dtype=bool),
        }
        # Zero-fill K/V inputs
        for inp in onnx_inputs:
            if inp.name.endswith("_k_ext") or inp.name.endswith("_v_ext"):
                shape = tuple(
                    d if isinstance(d, int) and d > 0 else 1 for d in inp.shape
                )
                feeds[inp.name] = np.zeros(shape, dtype=np.float32)
        return feeds

    feeds = _build_ort_feeds()
    out_name = sess0.get_outputs()[0].name
    t0 = _bench(lambda: sess0.run([out_name], feeds))
    _print_row("0", "ORT FP32 CUDA EP (no K/V)", t0, t0)

    # Tier 0b — ORT TensorRT EP
    t0b = t0
    sess0b, err0b = _ort_trt_session(UNET_ONNX, "ref_ldm_unet_external_kv_ctx.onnx")
    if sess0b is None:
        print(f"  Tier 0b | ORT TRT EP — skipped ({err0b})")
    else:
        try:
            for _ in range(WARMUP):
                sess0b.run([out_name], feeds)
            t0b = _bench(lambda: sess0b.run([out_name], feeds))
            _print_row("0b", "ORT TRT EP FP32", t0b, t0)
        except Exception as e:
            print(f"  Tier 0b | TRT EP — failed: {e}")
        finally:
            _shutil.rmtree(getattr(sess0b, "_trt_tmp", None) or "", ignore_errors=True)

    # Tier 1  — PyTorch FP32
    from custom_kernels.ref_ldm.ref_ldm_torch import RefLDMUNetTorch

    unet_fp32 = (
        RefLDMUNetTorch.from_onnx(UNET_ONNX, compute_dtype=torch.float32).cuda().eval()
    )
    with torch.no_grad():
        t1 = _bench(lambda: unet_fp32(x, ts, kv_map=kv_map, use_exclusive=True))
    _print_row("1", "PyTorch FP32 pure ops", t1, t0)

    # Tier 2  — PyTorch FP16 + Triton GroupNorm+SiLU (no CUDA graph)
    unet_fp16 = (
        RefLDMUNetTorch.from_onnx(UNET_ONNX, compute_dtype=torch.float16).cuda().eval()
    )
    with torch.no_grad():
        t2 = _bench(lambda: unet_fp16(x, ts, kv_map=kv_map, use_exclusive=True))
    _print_row("2", "PyTorch FP16 + Triton GroupNorm+SiLU", t2, t0)

    # Tier 3  — FP16 + Triton + CUDA graph (static K/V buffers)
    from custom_kernels.ref_ldm.ref_ldm_torch import build_unet_cuda_graph_runner

    unet_runner = build_unet_cuda_graph_runner(
        unet_fp16,
        x_shape=(1, 16, 64, 64),
        ts_example=ts,
        kv_map_template=kv_map,
        use_exclusive=True,
    )
    t3 = _bench(lambda: unet_runner(x, ts, kv_map, use_exclusive=True))
    _print_row("3", "FP16 + Triton + CUDA graph", t3, t0)

    # Tier 4  — FP16 + Triton + CUDA graph + NHWC (channels-last Conv2d)
    unet_cl = (
        RefLDMUNetTorch.from_onnx(UNET_ONNX, compute_dtype=torch.float16).cuda().eval()
    )
    unet_cl.to_channels_last()
    unet_cl_runner = build_unet_cuda_graph_runner(
        unet_cl,
        x_shape=(1, 16, 64, 64),
        ts_example=ts,
        kv_map_template=kv_map,
        use_exclusive=True,
    )
    t4 = _bench(lambda: unet_cl_runner(x, ts, kv_map, use_exclusive=True))
    _print_row("4", "FP16 + Triton + CUDA graph + NHWC", t4, t0)

    # Tier 5 — torch.compile(mode='default') + CUDA graph
    print("\n  [Tier 5] torch.compile(mode='default') + CUDA graph")
    print("  One-time compile cost: ~60–120 s on first run (UNet is a complex transformer).")
    try:
        unet_c5 = (
            RefLDMUNetTorch.from_onnx(UNET_ONNX, compute_dtype=torch.float16)
            .cuda()
            .eval()
        )
        unet_c5_runner = build_unet_cuda_graph_runner(
            unet_c5,
            x_shape=(1, 16, 64, 64),
            ts_example=ts,
            kv_map_template=kv_map,
            use_exclusive=True,
            torch_compile=True,
        )
        t5 = _bench(lambda: unet_c5_runner(x, ts, kv_map, use_exclusive=True))
        _print_row("5", "FP16 + Triton + CUDA graph + torch.compile", t5, t0)
    except Exception as e:
        print(f"  Tier 5 | torch.compile — failed ({type(e).__name__}: {e})")
        import traceback; traceback.print_exc()

    # Tier 5b — torch.compile reduce-overhead (no separate CUDA graph)
    print("\n  [Tier 5b] torch.compile(mode='reduce-overhead') — no extra CUDA graph")
    if not int(os.environ.get("REFLDM_TORCH_COMPILE", "0")):
        print("  Skipped — reduce-overhead may crash on this model on Windows/sm_89.")
        print("  Set REFLDM_TORCH_COMPILE=1 to attempt.")
    else:
        try:
            from custom_kernels.compile_utils import apply_torch_compile
            unet_5b = (
                RefLDMUNetTorch.from_onnx(UNET_ONNX, compute_dtype=torch.float16)
                .cuda()
                .eval()
            )
            unet_5b_compiled = apply_torch_compile(
                unet_5b,
                x,
                extra_args=(ts,),
                extra_kwargs={"kv_map": kv_map, "use_exclusive": True},
                compile_mode="reduce-overhead",
            )
            with torch.no_grad():
                t5b = _bench(lambda: unet_5b_compiled(x, ts, kv_map=kv_map, use_exclusive=True))
            _print_row("5b", "torch.compile reduce-overhead (no CUDA graph)", t5b, t0)
        except Exception as e:
            print(f"  Tier 5b | reduce-overhead — failed: {e}")
            import traceback; traceback.print_exc()

    print()
    return t0, t0b


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _check_prerequisites():
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
        print("Triton:  not available (FP32 fallback will be used)")
    try:
        import onnxruntime as ort
        print(f"ORT:     {ort.__version__}")
    except ImportError:
        print("ORT:     not available (skipping Tier 0/0b)")
        sys.exit(1)
    for path, name in [
        (ENC_ONNX, "VAE encoder"),
        (DEC_ONNX, "VAE decoder"),
        (UNET_ONNX, "UNet"),
    ]:
        if not pathlib.Path(path).exists():
            print(f"ERROR: {name} ONNX not found: {path}")
            sys.exit(1)


if __name__ == "__main__":
    # Each sub-benchmark can be run standalone via --encoder / --decoder / --unet.
    # When invoked without flags, each is spawned as an isolated subprocess so that
    # native TRT engine-build crashes (0xC0000005 access violations when building
    # multiple TRT engines in one process) cannot kill the overall benchmark run.
    import subprocess

    _MODE = None
    if "--encoder" in sys.argv:
        _MODE = "encoder"
    elif "--decoder" in sys.argv:
        _MODE = "decoder"
    elif "--unet" in sys.argv:
        _MODE = "unet"

    if _MODE is not None:
        # Subprocess mode: run only the requested sub-benchmark.
        _check_prerequisites()
        if _MODE == "encoder":
            bench_encoder()
        elif _MODE == "decoder":
            bench_decoder()
        else:
            bench_unet()
        sys.exit(0)

    # Orchestrator mode: spawn each sub-benchmark in its own process.
    # This prevents a TRT build crash in one component from killing the others.
    _check_prerequisites()
    _script = pathlib.Path(__file__).resolve()
    for _flag in ["--encoder", "--decoder", "--unet"]:
        print(f"\n{'='*60}")
        print(f"  Running {_flag[2:].upper()} sub-benchmark in isolated subprocess...")
        print(f"{'='*60}")
        _env = os.environ.copy()
        _result = subprocess.run(
            [sys.executable, str(_script), _flag],
            env=_env,
            timeout=3600,
        )
        if _result.returncode not in (0, 1):
            print(f"  [WARN] {_flag[2:].upper()} subprocess exited with code {_result.returncode}")
