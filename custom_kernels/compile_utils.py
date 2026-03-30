"""
Shared torch.compile infrastructure for all custom-kernel models.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable, Optional, TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
_compile_env_set_up: bool = False
_poll_fn: Optional[Callable[[], None]] = None
_show_fn: Optional[Callable[[str, str], None]] = None
_hide_fn: Optional[Callable[[], None]] = None

def register_compile_callbacks(
    poll_fn: Optional[Callable[[], None]] = None,
    show_fn: Optional[Callable[[str, str], None]] = None,
    hide_fn: Optional[Callable[[], None]] = None,
) -> None:
    global _poll_fn, _show_fn, _hide_fn
    if poll_fn is not None: _poll_fn = poll_fn
    if show_fn is not None: _show_fn = show_fn
    if hide_fn is not None: _hide_fn = hide_fn

def setup_compile_env(cache_dir: Optional[str] = None, compile_mode: str = "default") -> None:
    global _compile_env_set_up

    # Fix: Python default recursion limit of 1000 is too low for torch._dynamo regex
    # compilation on Windows/Python 3.11 (PYTORCH_TRIE and similar patterns).
    # Must be set in every process that calls torch.compile — worker sets it too.
    sys.setrecursionlimit(50000)

    # Fix: Windows cp1252 encoding issues with Triton arrows
    os.environ["PYTHONUTF8"] = "1"

    # CRITICAL: Disable Static CUDA Launcher on Windows to avoid 32-bit c_long handle overflows.
    # This must be set in BOTH env and config to override 'reduce-overhead' defaults.
    os.environ["TORCHINDUCTOR_USE_STATIC_CUDA_LAUNCHER"] = "0"
    os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
    os.environ.setdefault("TORCHINDUCTOR_MULTI_KERNEL", "0")

    if cache_dir is None and not os.environ.get("TORCHINDUCTOR_CACHE_DIR"):
        _project_root = Path(__file__).parent.parent
        cache_dir = str(_project_root / "model_assets" / "torch_compile_cache")
    os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", cache_dir)

    # Redirect Triton JIT cache to project-local directory.
    # ALWAYS force this path — triton_ops.py uses setdefault with a version-tagged
    # subdirectory which may run before setup_compile_env in the main process (at model
    # import time).  Both the compilation worker and the main inference process must use
    # the SAME TRITON_CACHE_DIR or the worker-compiled kernels won't be found and
    # Triton will recompile (crashing ptxas on complex models).
    _project_root = Path(__file__).parent.parent
    _triton_cache = _project_root / "model_assets" / "custom_kernels" / "triton_cache"
    try:
        _triton_cache.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    os.environ["TRITON_CACHE_DIR"] = str(_triton_cache)

    # Workaround: Python 3.11 re module cannot compile the massive PYTORCH_TRIE regex in
    # torch/utils/hipify/hipify_python.py (triggered by aoti_hipify_utils ← cpp_wrapper_cpu
    # ← cuda_kernel when torch._inductor codegen runs or when loading from FX graph cache).
    # HIP/ROCm hipify is unused on NVIDIA CUDA; stub it out before any inductor import.
    # This stub must be installed in EVERY process that calls torch.compile (worker AND app).
    if "torch.utils.hipify.hipify_python" not in sys.modules:
        import types as _types
        _hip_stub = _types.ModuleType("torch.utils.hipify.hipify_python")
        _hip_stub.PYTORCH_MAP = {}
        _hip_stub.PYTORCH_TRIE = type("_FakeTrie", (), {
            "export_to_regex": lambda self: "VISOMASTER_HIPIFY_STUB_NEVER_MATCHES",
        })()
        _hip_stub.RE_PYTORCH_PREPROCESSOR = None
        sys.modules["torch.utils.hipify.hipify_python"] = _hip_stub
        del _types, _hip_stub

    # Locate ptxas
    _ptxas: Optional[str] = None
    try:
        import triton as _triton_pkg
        _bundled = os.path.join(os.path.dirname(_triton_pkg.__file__), "backends", "nvidia", "bin", "ptxas.exe")
        if os.path.exists(_bundled): _ptxas = _bundled
    except Exception: pass
    if _ptxas is None:
        _sys_ptxas = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/bin/ptxas.exe"
        if os.path.exists(_sys_ptxas): _ptxas = _sys_ptxas
    if _ptxas is not None: os.environ.setdefault("TRITON_PTXAS.EXE_PATH", _ptxas)

    # Inductor config hardening
    try:
        import torch._inductor.config as _ind_cfg
        _ind_cfg.triton.use_block_ptr = False

        # Stability logic: reduce-overhead REQUIRES cudagraphs.
        if compile_mode == "reduce-overhead":
            _ind_cfg.triton.cudagraphs = True
        else:
            _ind_cfg.triton.cudagraphs = False

        _ind_cfg.triton.cudagraph_trees = False
        _ind_cfg.triton.descriptive_names = False
        _ind_cfg.triton.enable_eviction_policy = False

        # MANDATORY FOR WINDOWS: Avoid OverflowError: Python int too large to convert to C long
        _ind_cfg.use_static_cuda_launcher = False

        _ind_cfg.fx_graph_cache = True
        _ind_cfg.compile_threads = 1
    except Exception: pass

    torch.set_float32_matmul_precision("high")
    _compile_env_set_up = True

def _compute_compile_sentinel_path(model: nn.Module, cache_dir: Optional[str] = None) -> Path:
    try:
        cc = torch.cuda.get_device_capability()
        arch = f"sm_{cc[0]}{cc[1]}"
    except Exception: arch = "cpu"
    tv = torch.__version__.replace("+", "_").replace(".", "_")
    pv = f"py{sys.version_info.major}{sys.version_info.minor}"
    onnx_path = getattr(model, "_visomaster_onnx_path", "")
    path_hash = hashlib.md5(onnx_path.encode()).hexdigest()[:8]
    model_class = type(model).__name__
    # _nsl21: forced use_static_cuda_launcher=False for Windows.
    sentinel_name = f"{model_class}_{arch}_{tv}_{pv}_{path_hash}_nsl21.ok"
    if cache_dir: base = Path(cache_dir)
    else:
        env_dir = os.environ.get("TORCHINDUCTOR_CACHE_DIR", "")
        if env_dir: base = Path(env_dir)
        else:
            localappdata = os.environ.get("LOCALAPPDATA", "")
            if localappdata: base = Path(localappdata) / "torch" / "inductor"
            else: base = Path(tempfile.gettempdir()) / "torch_inductor"
    return base / "visomaster_sentinels" / sentinel_name

# These Windows AV/illegal-instruction codes are kept for log-comment purposes only;
# the subprocess logic now treats ANY non-zero exit as fatal.
_FATAL_AV_EXIT_CODES: frozenset = frozenset({3221225477, -1073741819, 3221225501})

def _ensure_compile_cache_subprocess(model: nn.Module, example_inp: torch.Tensor, warmup: int, extra_args: Optional[tuple], extra_kwargs: Optional[dict], compile_mode: str, cache_dir: Optional[str] = None) -> Optional[bool]:
    sentinel = _compute_compile_sentinel_path(model, cache_dir)
    if sentinel.exists():
        try: content = sentinel.read_text(encoding="utf-8").strip()
        except Exception: content = "ok"
        if content == "fatal": return None
        return True

    model_class = type(model).__name__
    spec: dict = {
        "root_path": str(Path(__file__).parent.parent),
        "model_class": model_class,
        "model_module": type(model).__module__,
        "onnx_path": getattr(model, "_visomaster_onnx_path", ""),
        "device": str(example_inp.device),
        "input_spec": {"shape": list(example_inp.shape), "dtype": str(example_inp.dtype).replace("torch.", "")},
        "extra_args_spec": [{"shape": list(a.shape), "dtype": str(a.dtype).replace("torch.", "")} for a in (extra_args or [])],
        "extra_kwargs": {k: v for k, v in (extra_kwargs or {}).items() if isinstance(v, (int, float, bool, str, type(None)))},
        "compile_mode": compile_mode,
        "warmup": warmup,
        "cache_dir": cache_dir or os.environ.get("TORCHINDUCTOR_CACHE_DIR") or "",
    }

    spec_fd, spec_path = tempfile.mkstemp(suffix="_compile_spec.json")
    try:
        with os.fdopen(spec_fd, "w", encoding="utf-8") as f: json.dump(spec, f)
    except Exception:
        try: os.close(spec_fd)
        except OSError: pass
        raise

    exitcode: Optional[int] = None
    success = False
    try:
        if _show_fn is not None: _show_fn("Compiling Custom Kernels", f"First-time torch.compile for {model_class}...")

        import subprocess
        worker_script = os.path.join(os.path.dirname(__file__), "compile_worker.py")
        cmd = [sys.executable, worker_script, spec_path]
        print(f"[compile_utils] Spawning worker: {' '.join(cmd)}")

        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        env["TORCHINDUCTOR_USE_STATIC_CUDA_LAUNCHER"] = "0"
        # Propagate TRITON_CACHE_DIR so the worker uses the project-local Triton
        # cache from Python startup — before Triton initialises and locks the path.
        if "TRITON_CACHE_DIR" in os.environ:
            env["TRITON_CACHE_DIR"] = os.environ["TRITON_CACHE_DIR"]

        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", env=env)
        while process.poll() is None:
            line = process.stdout.readline()
            if line: print(f"  [{model_class}] {line.strip()}")
            if _poll_fn: _poll_fn()
            time.sleep(0.01)
        remaining = process.stdout.read()
        if remaining:
            for rline in remaining.splitlines(): print(f"  [{model_class}] {rline.strip()}")
        exitcode = process.returncode
        success = (exitcode == 0)
    finally:
        if _hide_fn: _hide_fn()
        try: os.unlink(spec_path)
        except OSError: pass

    if success:
        sentinel.parent.mkdir(parents=True, exist_ok=True)
        sentinel.write_text("ok", encoding="utf-8")
        return True

    # ANY non-zero exit (Python exception, AV, illegal instruction, OOM, …)
    # means the compiled kernel state is unreliable — mark as fatal so the app
    # falls back to the CUDA-graph path instead of crashing at inference time.
    av_note = " (AV/illegal-instruction)" if exitcode in _FATAL_AV_EXIT_CODES else ""
    print(f"[compile_utils] Worker exited with code {exitcode}{av_note} — marking as fatal.")
    try:
        sentinel.parent.mkdir(parents=True, exist_ok=True)
        sentinel.write_text("fatal", encoding="utf-8")
    except Exception:
        pass
    return None

def apply_torch_compile(model: nn.Module, example_inp: torch.Tensor, warmup: int = 10, extra_args: Optional[tuple] = None, extra_kwargs: Optional[dict] = None, compile_mode: str = "default", _subprocess_mode: bool = False) -> nn.Module:
    setup_compile_env(compile_mode=compile_mode)
    _need_sentinel = False
    if not _subprocess_mode and hasattr(model, "_visomaster_onnx_path"):
        cache_dir = os.environ.get("TORCHINDUCTOR_CACHE_DIR") or None
        cache_ready = _ensure_compile_cache_subprocess(model, example_inp, warmup, extra_args, extra_kwargs, compile_mode, cache_dir)
        if cache_ready is None:
            raise RuntimeError(f"torch.compile fatally failed for {type(model).__name__} (Triton AV).")
        if not cache_ready:
            _need_sentinel = True

    # Fast FX cache hit
    compiled = torch.compile(model, mode=compile_mode, fullgraph=False, dynamic=False)

    if not _subprocess_mode:
        if _need_sentinel:
            try:
                cache_dir = os.environ.get("TORCHINDUCTOR_CACHE_DIR") or None
                sentinel = _compute_compile_sentinel_path(model, cache_dir)
                sentinel.parent.mkdir(parents=True, exist_ok=True)
                sentinel.write_text("ok", encoding="utf-8")
            except Exception: pass
        return compiled

    # Warmup (worker only)
    args = (example_inp,)
    if extra_args: args = args + extra_args
    kwargs = extra_kwargs or {}
    with torch.no_grad():
        for _ in range(warmup): compiled(*args, **kwargs)
    torch.cuda.synchronize()
    return compiled
