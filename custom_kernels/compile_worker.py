
import importlib
import json
import os
import sys
from pathlib import Path

def run_worker(spec_path: str):
    with open(spec_path, "r", encoding="utf-8") as f:
        spec = json.load(f)

    root = spec.get("root_path", "")
    if root and root not in sys.path:
        sys.path.insert(0, root)

    # SET ENVIRONMENT BEFORE ANY IMPORTS
    os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"
    os.environ["TORCHINDUCTOR_USE_STATIC_CUDA_LAUNCHER"] = "0"
    os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
    os.environ["TORCHINDUCTOR_MULTI_KERNEL"] = "0"
    os.environ["PYTHONUTF8"] = "1"

    cache_dir = spec.get("cache_dir")
    if cache_dir:
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = cache_dir

    # Redirect Triton JIT cache to project-local root directory — MUST happen before
    # any triton import.  ALWAYS force (not setdefault) so the worker uses the same
    # root path as setup_compile_env(), overriding any version-tagged path that
    # triton_ops.py may inject via setdefault later during model import.
    _triton_cache = Path(root) / "model_assets" / "custom_kernels" / "triton_cache"
    try:
        _triton_cache.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    os.environ["TRITON_CACHE_DIR"] = str(_triton_cache)

    sys.setrecursionlimit(50000)

    # SCRUB Shiboken/PySide6 import hooks before ANY other imports
    _to_remove = []
    for hook in sys.meta_path:
        _h_str = str(hook).lower()
        if "shiboken" in _h_str or "pyside" in _h_str:
            _to_remove.append(hook)
    for hook in _to_remove:
        try: sys.meta_path.remove(hook)
        except ValueError: pass
    sys.modules.pop("shiboken", None)
    sys.modules.pop("PySide6", None)

    import torch
    import torch.nn as nn

    compile_mode = spec.get("compile_mode", "default")

    try:
        import torch._inductor.config as ind_cfg
        ind_cfg.triton.descriptive_names = False
        ind_cfg.triton.use_block_ptr = False
        ind_cfg.triton.enable_eviction_policy = False

        # Consistent with compile_utils.py setup_compile_env()
        if compile_mode == "reduce-overhead":
            ind_cfg.triton.cudagraphs = True
        else:
            ind_cfg.triton.cudagraphs = False

        ind_cfg.triton.cudagraph_trees = False

        # MANDATORY FOR WINDOWS: Avoid OverflowError: Python int too large to convert to C long
        ind_cfg.use_static_cuda_launcher = False

        ind_cfg.fx_graph_cache = True
        ind_cfg.compile_threads = 1
    except Exception:
        pass

    from custom_kernels.compile_utils import apply_torch_compile, setup_compile_env
    setup_compile_env(cache_dir=cache_dir, compile_mode=compile_mode)

    # Import model
    module = importlib.import_module(spec["model_module"])
    ModelClass = getattr(module, spec["model_class"])
    model = ModelClass.from_onnx(spec["onnx_path"])

    device = spec.get("device", "cuda")
    model = model.to(device).eval()

    # Input
    inp_spec = spec["input_spec"]
    example_inp = torch.zeros(
        inp_spec["shape"],
        dtype=getattr(torch, inp_spec.get("dtype", "float32")),
        device=device,
    )

    # Extra args
    extra_args = None
    if spec.get("extra_args_spec"):
        extra_args = tuple(
            torch.zeros(
                a["shape"],
                dtype=getattr(torch, a.get("dtype", "float32")),
                device=device,
            )
            for a in spec["extra_args_spec"]
        )

    # Extra kwargs
    raw_kwargs = spec.get("extra_kwargs") or {}
    extra_kwargs = raw_kwargs if raw_kwargs else None

    # COMPILE
    print(f"[worker] Starting torch.compile for {spec['model_class']} (mode={compile_mode})...")
    apply_torch_compile(
        model,
        example_inp,
        warmup=spec.get("warmup", 10),
        extra_args=extra_args,
        extra_kwargs=extra_kwargs,
        compile_mode=compile_mode,
        _subprocess_mode=True,
    )
    torch.cuda.synchronize()
    print(f"[worker] Compilation successful.")
    os._exit(0)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(1)
    run_worker(sys.argv[1])
