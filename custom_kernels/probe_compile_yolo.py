"""FaceParser: warmup sweep on FX cache hit path."""
from __future__ import annotations
import os, sys, time, tempfile, shutil, warnings
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
for _c in [_ROOT/".venv"/"Lib"/"site-packages"/"tensorrt_libs"]:
    if _c.exists(): os.environ["PATH"]=str(_c)+os.pathsep+os.environ.get("PATH","")
if hasattr(sys.stdout,'reconfigure'): sys.stdout.reconfigure(encoding='utf-8',errors='replace')
if hasattr(sys.stderr,'reconfigure'): sys.stderr.reconfigure(encoding='utf-8',errors='replace')

import torch
print(f"PyTorch {torch.__version__} | GPU: {torch.cuda.get_device_name(0)}")

ONNX = str(_ROOT/"model_assets"/"faceparser_resnet34.onnx")
_TMP = tempfile.mkdtemp(prefix="probe_fp_")
os.environ["TORCHINDUCTOR_CACHE_DIR"] = _TMP

try:
    from custom_kernels.compile_utils import (
        apply_torch_compile, setup_compile_env, _compute_compile_sentinel_path,
    )
    from custom_kernels.faceparser_resnet34.faceparser_resnet34_torch import (
        FaceParserResnet34Torch, build_cuda_graph_runner,
    )
    setup_compile_env(cache_dir=_TMP)
    inp = torch.rand(1,3,512,512,dtype=torch.float32,device="cuda")

    # Baseline
    m_base = FaceParserResnet34Torch.from_onnx(ONNX, compute_dtype=torch.float16).cuda().eval()
    runner_base = build_cuda_graph_runner(m_base, torch_compile=False)
    with torch.no_grad():
        for _ in range(5): runner_base(inp)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(20): runner_base(inp)
    torch.cuda.synchronize()
    base_ms = (time.perf_counter()-t0)/20*1000
    print(f"CUDA graph baseline: {base_ms:.2f} ms/iter\n")

    # Populate FX cache (subprocess simulation)
    m1 = FaceParserResnet34Torch.from_onnx(ONNX, compute_dtype=torch.float16).cuda().eval()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        compiled1 = apply_torch_compile(m1, inp, compile_mode="reduce-overhead",
                                        _subprocess_mode=True, warmup=30)
    sentinel = _compute_compile_sentinel_path(m1, _TMP)
    sentinel.parent.mkdir(parents=True, exist_ok=True)
    sentinel.write_text("ok", encoding="utf-8")
    print("FX cache populated. Warmup sweep on FX cache hit path:")

    # Warmup sweep: compile fresh model from cache, add N extra warmup calls, measure
    for extra in [0, 20, 50, 100, 200]:
        m = FaceParserResnet34Torch.from_onnx(ONNX, compute_dtype=torch.float16).cuda().eval()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            compiled = apply_torch_compile(m, inp, compile_mode="reduce-overhead",
                                           warmup=30)  # 30 in apply_torch_compile
        with torch.no_grad():
            for _ in range(extra): compiled(inp)  # extra warmup
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(20): compiled(inp)
        torch.cuda.synchronize()
        ms = (time.perf_counter()-t0)/20*1000
        print(f"  warmup={30+extra:3d}: {ms:.2f} ms/iter  ({base_ms/ms:.2f}x)")

finally:
    shutil.rmtree(_TMP, ignore_errors=True)
