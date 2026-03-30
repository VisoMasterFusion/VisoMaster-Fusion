"""
Probe torch.compile modes on yoloface and faceparser.
Tests reduce-overhead and cudagraphs backends (max-autotune failed with ptxas sm_89 error).

Usage (from repo root):
    .venv\\Scripts\\python custom_kernels\\probe_compile.py
"""
from __future__ import annotations

import os as _os, sys as _sys
from pathlib import Path as _Path

_REPO_ROOT = _Path(__file__).resolve().parents[1]
for _candidate in [
    _REPO_ROOT / ".venv" / "Lib" / "site-packages" / "tensorrt_libs",
    _REPO_ROOT / ".venv" / "Lib" / "site-packages" / "nvidia" / "cuda_runtime" / "bin",
]:
    if _candidate.exists():
        _os.environ["PATH"] = str(_candidate) + _os.pathsep + _os.environ.get("PATH", "")

import sys as _sys_enc
if hasattr(_sys_enc.stdout, 'reconfigure'):
    _sys_enc.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(_sys_enc.stderr, 'reconfigure'):
    _sys_enc.stderr.reconfigure(encoding='utf-8', errors='replace')

import sys, time, traceback, pathlib
import torch
import numpy as np

ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
ONNX_DIR = ROOT / "model_assets"

WARMUP = 10
ITERS  = 50

def _bench(fn, warmup=WARMUP, iters=ITERS):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0


def _try_compile_cg(model, inp, label):
    """Try torch.compile(model) + CUDA graph capture. Returns ms or error string."""
    try:
        compiled = torch.compile(model, mode="reduce-overhead", fullgraph=False)
        # warm up compiled model (triggers compilation)
        with torch.no_grad():
            for _ in range(5):
                compiled(inp)
        torch.cuda.synchronize()

        # Capture CUDA graph of the compiled model
        g = torch.cuda.CUDAGraph()
        inp_static = inp.clone()
        out_static = None
        with torch.cuda.graph(g):
            with torch.no_grad():
                out_static = compiled(inp_static)
        torch.cuda.synchronize()

        def _run():
            inp_static.copy_(inp)
            g.replay()

        ms = _bench(_run)
        return ms, None
    except Exception as e:
        return None, traceback.format_exc()[-800:]


def _try_compile_eager(model, inp, mode, label):
    """Try torch.compile(model, mode=mode) eager (no CUDA graph). Returns ms or error."""
    try:
        compiled = torch.compile(model, mode=mode, fullgraph=False)
        with torch.no_grad():
            for _ in range(5):
                compiled(inp)
        torch.cuda.synchronize()
        ms = _bench(lambda: compiled(inp) if True else None)
        return ms, None
    except Exception as e:
        return None, traceback.format_exc()[-800:]


def _baseline_cg(model, inp):
    """Plain CUDA graph baseline (no compile)."""
    with torch.no_grad():
        for _ in range(3): model(inp)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    inp_s = inp.clone()
    with torch.cuda.graph(g):
        with torch.no_grad():
            out_s = model(inp_s)
    torch.cuda.synchronize()
    def _run():
        inp_s.copy_(inp)
        g.replay()
    return _bench(_run)


# ── YOLOFACE ─────────────────────────────────────────────────────────────────
def probe_yoloface():
    print("\n=== yoloface_8n probe ===")
    from custom_kernels.yoloface_8n.yoloface8n_torch import YoloFace8nTorch
    onnx = str(ONNX_DIR / "yoloface_8n.onnx")
    inp  = torch.rand(1, 3, 640, 640, dtype=torch.float32, device="cuda")

    m16 = YoloFace8nTorch.from_onnx(onnx, compute_dtype=torch.float16).cuda().eval()

    base_ms = _baseline_cg(m16, inp)
    print(f"  Baseline CUDA graph (FP16)         : {base_ms:7.3f} ms")

    # reduce-overhead eager
    ms, err = _try_compile_eager(m16, inp, "reduce-overhead", "reduce-overhead eager")
    if err:
        print(f"  compile(reduce-overhead) eager     : FAILED")
        print(f"    {err[:300]}")
    else:
        print(f"  compile(reduce-overhead) eager     : {ms:7.3f} ms  ({base_ms/ms:.2f}x vs baseline)")

    # reduce-overhead + CUDA graph
    ms2, err2 = _try_compile_cg(m16, inp, "compile+CG")
    if err2:
        print(f"  compile(reduce-overhead) + CG      : FAILED")
        print(f"    {err2[:300]}")
    else:
        print(f"  compile(reduce-overhead) + CG      : {ms2:7.3f} ms  ({base_ms/ms2:.2f}x vs baseline)")

    # default mode eager
    ms3, err3 = _try_compile_eager(m16, inp, "default", "default eager")
    if err3:
        print(f"  compile(default) eager             : FAILED")
        print(f"    {err3[:300]}")
    else:
        print(f"  compile(default) eager             : {ms3:7.3f} ms  ({base_ms/ms3:.2f}x vs baseline)")


# ── FACEPARSER ────────────────────────────────────────────────────────────────
def probe_faceparser():
    print("\n=== faceparser_resnet34 probe ===")
    from custom_kernels.faceparser_resnet34.faceparser_resnet34_torch import FaceParserResnet34Torch
    onnx = str(ONNX_DIR / "faceparser_resnet34.onnx")
    inp  = torch.randn(1, 3, 512, 512, dtype=torch.float32, device="cuda")

    m16 = FaceParserResnet34Torch.from_onnx(onnx, compute_dtype=torch.float16).cuda().eval()

    base_ms = _baseline_cg(m16, inp)
    print(f"  Baseline CUDA graph (FP16)         : {base_ms:7.3f} ms")

    ms, err = _try_compile_eager(m16, inp, "reduce-overhead", "reduce-overhead eager")
    if err:
        print(f"  compile(reduce-overhead) eager     : FAILED")
        print(f"    {err[:300]}")
    else:
        print(f"  compile(reduce-overhead) eager     : {ms:7.3f} ms  ({base_ms/ms:.2f}x vs baseline)")

    ms2, err2 = _try_compile_cg(m16, inp, "compile+CG")
    if err2:
        print(f"  compile(reduce-overhead) + CG      : FAILED")
        print(f"    {err2[:300]}")
    else:
        print(f"  compile(reduce-overhead) + CG      : {ms2:7.3f} ms  ({base_ms/ms2:.2f}x vs baseline)")

    ms3, err3 = _try_compile_eager(m16, inp, "default", "default eager")
    if err3:
        print(f"  compile(default) eager             : FAILED")
        print(f"    {err3[:300]}")
    else:
        print(f"  compile(default) eager             : {ms3:7.3f} ms  ({base_ms/ms3:.2f}x vs baseline)")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available"); sys.exit(1)
    print(f"Device : {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    try:
        import triton; print(f"Triton : {triton.__version__}")
    except Exception: print("Triton : not available")

    probe_yoloface()
    probe_faceparser()
    print("\nDone.")
