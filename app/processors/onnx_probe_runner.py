"""Standalone ONNX/TensorRT engine-build probe.

``ModelsProcessor.load_model()`` spawns this script as a separate process
instead of building the TensorRT engine in-process:

* A fatal C++/CUDA crash during the build kills only this process, never the
  main application.
* Because it is launched with piped stdout/stderr (instead of a
  ``multiprocessing`` spawn that inherits the console), every line it prints
  — including ONNX Runtime / TensorRT build messages — is streamed back to
  the parent, which feeds it to the build progress dialog.

Usage::

    python onnx_probe_runner.py <config.json>

where the config JSON carries the model path, the provider list, the
TensorRT provider options and the session options. Exit code 0 means the
engine cache was built and flushed to disk; 1 means failure.
"""

import json
import os
import sys
import traceback


def main() -> int:
    if len(sys.argv) < 2:
        print("[ONNX Prober]: ERROR - no config file argument supplied.", flush=True)
        return 1

    config_path = sys.argv[1]
    try:
        with open(config_path, "r", encoding="utf-8") as handle:
            config = json.load(handle)
    except Exception:
        print(
            f"[ONNX Prober]: ERROR - could not read config file: {config_path}",
            flush=True,
        )
        traceback.print_exc()
        return 1

    # Make `app.*` imports resolvable when this script is launched directly
    # (rather than via ``python -m``) from any working directory.
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    try:
        import onnxruntime
        import torch

        model_path = config["model_path"]
        providers_list = config["providers"]
        trt_options = config.get("trt_options") or {}
        session_options_dict = config.get("session_options") or {}

        # Create the SessionOptions object *inside* the worker process.
        session_options = onnxruntime.SessionOptions()
        for key, value in session_options_dict.items():
            setattr(session_options, key, value)
        # ONNX Runtime defaults to WARNING, which hides every message the
        # TensorRT builder emits. INFO is chatty enough to expose the build
        # phases (parsing, tactic autotuning, serialization) without the
        # multi-megabyte spam VERBOSE would produce, and the parent only
        # surfaces a tail of it in the progress dialog anyway.
        session_options.log_severity_level = 1  # INFO

        # Set the CUDA device to match the TRT provider's device_id.
        gpu_id = trt_options.get("device_id", 0)
        if gpu_id != 0 and torch.cuda.is_available():
            torch.cuda.set_device(gpu_id)

        # Reconstruct the providers list (JSON turns tuples into lists).
        providers = []
        for entry in providers_list:
            name = entry[0] if isinstance(entry, (tuple, list)) else entry
            if name == "TensorrtExecutionProvider":
                providers.append((name, trt_options))
            elif isinstance(entry, (tuple, list)) and len(entry) > 1:
                providers.append(tuple(entry))
            else:
                providers.append(name)

        print(
            f"[ONNX Prober]: Attempting to load {os.path.basename(model_path)}...",
            flush=True,
        )
        # This call is the one that triggers the build/cache generation.
        session = onnxruntime.InferenceSession(
            model_path, sess_options=session_options, providers=providers
        )

        # Wait until all CUDA operations (i.e., the engine build and the
        # serialization to disk) are *fully* complete before exiting.
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # If we get here, the load and the synchronization worked.
        del session
        print(
            "[ONNX Prober]: Load successful. TRT engine cache built and flushed.",
            flush=True,
        )
        return 0
    except Exception:
        print("[ONNX Prober]: ERROR during model load probe.", flush=True)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
