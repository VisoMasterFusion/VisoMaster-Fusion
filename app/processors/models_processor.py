import threading
import os
import subprocess as sp
import gc
import traceback
import multiprocessing
import re
import time
from typing import Dict, TYPE_CHECKING, Any
from packaging import version
import numpy as np

try:
    import onnxruntime
except ImportError as _ort_err:
    print("\n" + "=" * 70)
    print("[FATAL] onnxruntime failed to import.")
    print(f"  Error: {_ort_err}")
    print()
    print("  COMMON FIXES:")
    print("  1. Install Visual C++ Redistributable 2019 (x64) from Microsoft.")
    print("     Download: https://aka.ms/vs/17/release/vc_redist.x64.exe")
    print("  2. Ensure CUDA 12.x runtime DLLs are present (cudart64_12.dll etc.).")
    print(
        "     Install CUDA Toolkit 12.x from https://developer.nvidia.com/cuda-downloads"
    )
    print("  3. On Windows 10 older than 1903: update Windows or install KB4571756.")
    print("  4. Portable install: run 'Check / Update Dependencies' in the Launcher.")
    print("=" * 70 + "\n")
    raise

import torch
import onnx
from torchvision.transforms import v2

from app.processors.utils import faceutil

from PySide6 import QtCore

# TENSORRT IMPORT
try:
    import tensorrt as trt

    TENSORRT_AVAILABLE = True
except ModuleNotFoundError:
    print("[WARN] No TensorRT Found")
    TENSORRT_AVAILABLE = False
    trt = None

from app.processors.utils.dfm_model import DFMModel
from app.processors.models_data import (
    models_list,
    fp16_safe_models_list,
    tensorrt_shape_infer_models,
    ARCFACE_DST,
    FFHQ_KPS,
    LANDMARKS_SUBSET_IDXS,
)

if TYPE_CHECKING:
    from app.ui.main_ui import MainWindow

# --- Global Configuration ---

onnxruntime.set_default_logger_severity(4)
onnxruntime.log_verbosity_level = -1


# --- Isolated Process Workers ---
# These functions run in a separate process to prevent fatal C++/CUDA
# crashes (like segmentation faults) from killing the main application.
def _probe_onnx_model_worker(
    model_path, providers_list, trt_options, session_options_dict
):
    """
    Worker function to be run in an isolated process to "warm up"
    an ONNX model, especially for the TensorRT provider.
    This triggers the engine cache build without freezing the main thread.
    """
    # Move all imports to top of function so sys.exit(1) is always available
    import os
    import sys
    import traceback
    import onnxruntime
    import torch

    try:
        # Create the SessionOptions object *inside* the worker process.
        session_options = onnxruntime.SessionOptions()
        if session_options_dict:
            for key, value in session_options_dict.items():
                # Use setattr to configure the SessionOptions object
                setattr(session_options, key, value)

        # Set the CUDA device to match the TRT provider's device_id
        gpu_id = trt_options.get("device_id", 0)
        if gpu_id != 0 and torch.cuda.is_available():
            torch.cuda.set_device(gpu_id)

        # Reconstruct the providers tuple
        providers = []
        for p in providers_list:
            name = p[0] if isinstance(p, tuple) else p
            if name == "TensorrtExecutionProvider":
                providers.append((name, trt_options))
            elif isinstance(p, tuple) and len(p) > 1:
                providers.append(p)
            else:
                providers.append(name)

        print(f"[ONNX Prober]: Attempting to load {os.path.basename(model_path)}...")
        # This line is the one that triggers the build/cache generation
        session = onnxruntime.InferenceSession(
            model_path, sess_options=session_options, providers=providers
        )

        # Force this prober process to wait until all CUDA operations
        # (i.e., the engine build and serialization to disk)
        # are *fully* complete before this process exits.
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # If we get here, the load and the synchronization worked.
        del session
        print("[ONNX Prober]: Load successful. TRT engine cache built and flushed.")
        sys.exit(0)  # Success
    except Exception:
        print("[ONNX Prober]: ERROR during model load probe.")
        traceback.print_exc()
        sys.exit(1)  # Failure


class ModelsProcessor(QtCore.QObject):
    """
    Central hub for managing AI models (ONNX, TensorRT, PyTorch).
    Handles:
    - Model Loading/Unloading (Thread-safe)
    - TensorRT Engine compilation and caching
    - Inference wrapper methods for various tasks (detection, swapping, restoration)
    - GPU memory management
    """

    processing_complete = QtCore.Signal()
    model_loaded = QtCore.Signal()  # Signal emitted with Onnx InferenceSession

    # Signal to request the GUI thread to show the build dialog
    # Arguments: (str: window_title, str: label_text)
    show_build_dialog = QtCore.Signal(str, str)
    # Signal to request the GUI thread to hide the build dialog
    hide_build_dialog = QtCore.Signal()

    def __init__(self, main_window: "MainWindow", device: str = "cuda") -> None:
        """
        Initialises the ModelsProcessor.

        Sets up all model dictionaries, TensorRT options, provider lists,
        and helper state (locks, sync vectors). Sub-processors are managed externally.

        Args:
            main_window: The application's MainWindow, used to access UI controls and signals.
            device: Torch/ONNX device string — ``"cuda"`` or ``"cpu"``.
        """
        super().__init__()
        self.main_window = main_window
        self.gpu_id = getattr(main_window, "gpu_id", 0)
        self.provider_name = "TensorRT"

        # NOTE: internal_deep_copied_kv_map / internal_kv_map_source_filename were
        # placeholder attributes for a planned per-session KV-map cache.  They are
        # currently unused (never written after __init__).  If a future feature
        # populates them, ensure a matching cleanup path is added to the force-unload
        # path (delete_models_dfm / force_unload path) so the tensors are freed.
        self.internal_deep_copied_kv_map: Dict[str, Dict[str, torch.Tensor]] | None = (
            None
        )
        self.internal_kv_map_source_filename: str | None = None
        self.device = f"{device}:{self.gpu_id}" if device != "cpu" else device
        self.device_type = device
        if self.gpu_id != 0 and device != "cpu":
            torch.cuda.set_device(self.gpu_id)
        self.model_lock = threading.RLock()  # Reentrant lock for model access

        self.cuda_graph_capture_lock = threading.Lock()

        # --- TENSORRT WORKSPACE ---
        MIN_WORKSPACE_SIZE = 1073741824  # 1 GB
        FALLBACK_WORKSPACE_SIZE = 4294967296  # 4 GB

        try:
            # Get total GPU memory in bytes
            total_vram = torch.cuda.get_device_properties(self.gpu_id).total_memory
            # Safely allocate 40% of total VRAM for TensorRT workspace
            calculated_workspace = int(total_vram * 0.40)
            # Enforce a minimum of 1 GB to avoid compilation failures on very low-end GPUs
            workspace_size = max(calculated_workspace, MIN_WORKSPACE_SIZE)
        except Exception:
            # Fallback to 4GB if PyTorch fails to detect the GPU
            workspace_size = FALLBACK_WORKSPACE_SIZE

        # Default TensorRT options
        self.trt_ep_options: Dict[str, Any] = {
            "device_id": self.gpu_id,
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": "tensorrt-engines",
            "trt_timing_cache_enable": True,
            "trt_timing_cache_path": "tensorrt-engines",
            "trt_dump_ep_context_model": True,
            "trt_ep_context_file_path": "tensorrt-engines",
            "trt_layer_norm_fp32_fallback": True,
            "trt_max_workspace_size": workspace_size,
            "trt_builder_optimization_level": 5,
        }

        # A set to keep track of models that have been loaded but
        # have not had their engine built (lazy build).
        self.models_pending_build: set = set()
        self.providers: list = [
            ("TensorrtExecutionProvider", self.trt_ep_options),
            ("CUDAExecutionProvider", {"device_id": self.gpu_id}),
            ("CPUExecutionProvider"),
        ]
        self.syncvec = torch.empty((1, 1), dtype=torch.float32, device=self.device)
        self.nThreads = 1

        # Initialize models and models_path dictionaries
        self.models: Dict[str, Any] = {}
        self.models_path: Dict[str, str] = {}
        self.models_data: Dict[str, Dict[str, Any]] = {}

        for model_data in models_list:
            model_name, model_path = model_data["model_name"], model_data["local_path"]
            self.models[model_name] = None  # Model Instance placeholder
            self.models_path[model_name] = model_path
            self.models_data[model_name] = {
                "local_path": model_data["local_path"],
                "hash": model_data["hash"],
                "url": model_data.get("url"),
            }

        self.dfm_models: Dict[str, DFMModel] = {}
        self.dfm_inference_lock = threading.Lock()
        self.force_unload_in_progress = False

        # --- SMART UNLOAD STATE ---
        self.deferred_unloads: Dict[str, Dict[str, Any]] = {}

        # Initialize Mask Latent
        self.lp_mask_crop_latent = faceutil.create_faded_inner_mask(
            size=(64, 64),
            border_thickness=3,
            fade_thickness=8,
            blur_radius=3,
            device=self.device,
        )
        self.lp_mask_crop_latent = torch.unsqueeze(
            self.lp_mask_crop_latent, 0
        )  # Shape: [1, 64, 64]

        # Initialize Clip
        self.clip_session: list = []

        # --- Face Analysis Constants (ArcFace/Landmarks) ---
        self.arcface_dst: np.ndarray = ARCFACE_DST
        self.FFHQ_kps: np.ndarray = FFHQ_KPS
        self.LandmarksSubsetIdxs: list[int] = LANDMARKS_SUBSET_IDXS
        self.mean_lmk: list = []
        self.anchors: list = []
        self.emap: list = []

        self.normalize = v2.Normalize(
            mean=[0.0, 0.0, 0.0], std=[1 / 1.0, 1 / 1.0, 1 / 1.0]
        )

    @property
    def binding_device_id(self) -> int:
        return self.gpu_id if self.device_type != "cpu" else 0

    def _ensure_trt_ready_onnx(self, model_name: str, onnx_path: str) -> str:
        """Return an ONNX path that the TensorRT EP can build an engine from.

        Some models (see ``tensorrt_shape_infer_models``) contain ops — notably
        5-D ``GridSample`` in the PerformRecast warping module — whose output
        tensors carry no static shape. The TensorRT EP refuses such graphs with
        "has no shape specified. Please run shape inference on the onnx model
        first." We fix this once by pinning the batch dimension to 1 (the app
        always feeds a single face) and running ONNX Runtime's symbolic shape
        inference, then caching the result next to the original as
        ``*.trtshape.onnx``. The cached file is reused unless the source ONNX is
        newer. For models not in the list, the original path is returned as-is.
        """
        if model_name not in tensorrt_shape_infer_models:
            return onnx_path
        if not onnx_path.lower().endswith(".onnx"):
            return onnx_path

        sidecar_path = onnx_path[: -len(".onnx")] + ".trtshape.onnx"
        try:
            if os.path.exists(sidecar_path) and (
                os.path.getmtime(sidecar_path) >= os.path.getmtime(onnx_path)
            ):
                return sidecar_path

            print(
                f"[INFO] Preparing TensorRT-ready (shape-inferred) ONNX for {model_name}..."
            )
            # This shape-inference pass can take a noticeable amount of time for
            # large graphs (the PerformRecast warping module is ~200 MB) and runs
            # *before* the engine-build probe, so without a dialog the UI looks
            # frozen with no indication of what is happening. Surface a dialog for
            # this preprocessing step too. It is only paid once (result cached).
            self.show_build_dialog.emit(
                "Preparing TensorRT Model",
                f"Running shape inference for:\n{model_name}\n\n"
                f"This one-time step prepares the model for the TensorRT engine "
                f"build and may take a moment.",
            )
            try:
                from onnxruntime.tools.onnx_model_utils import make_dim_param_fixed
                from onnxruntime.tools.symbolic_shape_infer import (
                    SymbolicShapeInference,
                )

                model = onnx.load(onnx_path)
                # Pin the dynamic 'batch' axis to 1 so symbolic dims (e.g.
                # "50*batch") resolve to concrete values the TensorRT builder
                # accepts.
                try:
                    make_dim_param_fixed(model.graph, "batch", 1)
                except Exception as dim_err:
                    # Not fatal — symbolic shape inference may still add shapes.
                    print(f"[WARN] Could not pin batch dim for {model_name}: {dim_err}")
                model = SymbolicShapeInference.infer_shapes(
                    model, auto_merge=True, guess_output_rank=True
                )
                onnx.save(model, sidecar_path)
                del model
                gc.collect()
            finally:
                self.hide_build_dialog.emit()
            print(f"[INFO] Wrote shape-inferred ONNX: {os.path.basename(sidecar_path)}")
            return sidecar_path
        except Exception as e:
            print(
                f"[WARN] Shape-inference preprocessing failed for {model_name} ({e}). "
                f"Falling back to the original ONNX."
            )
            traceback.print_exc()
            return onnx_path

    def _check_tensorrt_cache_state(
        self, model_name: str, onnx_path: str
    ) -> str | None:
        """
        Checks if a valid TensorRT cache (ctx and engine file) exists for the given model.

        Returns:
            "LEGACY": if a valid legacy cache (generic TensorrtExecutionProvider_ naming) is found.
            "EXPLICIT": if a valid explicit cache (custom model_name prefix) is found.
            None: if no valid cache is found.
        """
        try:
            cache_dir = "tensorrt-engines"
            base_onnx_name = os.path.splitext(os.path.basename(onnx_path))[0]

            # Support both UI model names (explicit prefix) and base ONNX file names (legacy prefix)
            possible_prefixes = list(dict.fromkeys([model_name, base_onnx_name]))

            for prefix in possible_prefixes:
                ctx_file_name = f"{prefix}_ctx.onnx"
                ctx_file_path = os.path.join(cache_dir, ctx_file_name)

                if os.path.exists(ctx_file_path) and os.path.isfile(ctx_file_path):
                    with open(ctx_file_path, "rb") as f:
                        content = f.read()

                    # Look for the engine name embedded in the context file using regex
                    match = re.search(rb"[A-Za-z0-9_.-]+\.engine", content)
                    if not match:
                        continue  # Keep searching next prefix instead of failing early

                    engine_name = match.group(0).decode("utf-8")
                    engine_subdirectory_name = os.path.basename(cache_dir)

                    # Check root cache directory and subdirectory
                    engine_file_path_root = os.path.join(cache_dir, engine_name)
                    engine_file_path_sub = os.path.join(
                        cache_dir, engine_subdirectory_name, engine_name
                    )

                    if os.path.exists(engine_file_path_root) or os.path.exists(
                        engine_file_path_sub
                    ):
                        if engine_name.startswith("TensorrtExecutionProvider_"):
                            return "LEGACY"
                        return "EXPLICIT"

            return None  # No valid engine found after checking all prefixes

        except Exception as e:
            print(f"[ERROR] Failed TensorRT cache state check for {model_name}: {e}")
            return None

    def _clean_tensorrt_cache(
        self, onnx_path: str, trt_options: Dict[str, Any]
    ) -> None:
        """
        Cleans up potentially corrupted TensorRT cache files for a specific model.
        Safely handles both legacy (generic ORT naming) and explicit prefixed caches.

        Args:
            onnx_path (str): The local path to the ONNX model.
            trt_options (Dict[str, Any]): The TensorRT options dictionary.
        """
        cache_dir = trt_options.get("trt_engine_cache_path", "tensorrt-engines")
        base_onnx_name = os.path.splitext(os.path.basename(onnx_path))[0]

        # Extract the explicit prefix if available
        target_prefix = trt_options.get("trt_engine_cache_prefix")

        possible_prefixes: list[str] = []
        if target_prefix:
            possible_prefixes.append(target_prefix)
        possible_prefixes.append(base_onnx_name)
        possible_prefixes = list(dict.fromkeys(possible_prefixes))

        engine_file_paths_to_check: list[str] = []

        # 1. Read context files across all candidate prefixes to extract referenced engine paths
        for prefix in possible_prefixes:
            ctx_file_name = f"{prefix}_ctx.onnx"
            ctx_file_path = os.path.join(cache_dir, ctx_file_name)

            if os.path.exists(ctx_file_path) and os.path.isfile(ctx_file_path):
                try:
                    with open(ctx_file_path, "rb") as f:
                        content = f.read()

                    # Extract the engine name using the broader regex
                    match = re.search(rb"[A-Za-z0-9_.-]+\.engine", content)
                    if match:
                        engine_name = match.group(0).decode("utf-8")

                        # Failsafe: ORT pathing behavior varies.
                        engine_subdirectory_name = os.path.basename(cache_dir)
                        engine_file_paths_to_check.extend(
                            [
                                os.path.join(cache_dir, engine_name),
                                os.path.join(
                                    cache_dir, engine_subdirectory_name, engine_name
                                ),
                            ]
                        )
                except Exception as e:
                    print(
                        f"[WARN] Could not read context file {ctx_file_path} to find engine name: {e}"
                    )

            # 2. Delete context file safely
            if os.path.exists(ctx_file_path) and os.path.isfile(ctx_file_path):
                try:
                    os.remove(ctx_file_path)
                    print(f"[INFO] Deleted TensorRT context file: {ctx_file_path}")
                except Exception as e:
                    print(
                        f"[WARN] Failed to delete {ctx_file_path} (locked or missing): {e}"
                    )

        # 3. Delete referenced engine files
        for engine_path in set(engine_file_paths_to_check):
            if (
                engine_path
                and os.path.exists(engine_path)
                and os.path.isfile(engine_path)
            ):
                try:
                    os.remove(engine_path)
                    print(f"[INFO] Deleted TensorRT engine file: {engine_path}")
                except Exception as e:
                    print(f"[WARN] Failed to delete engine file {engine_path}: {e}")

        # 4. Clean up auxiliary / profile / timing cache files
        if os.path.exists(cache_dir) and os.path.isdir(cache_dir):
            try:
                for file_name in os.listdir(cache_dir):
                    # Catch model-specific files tracking all prefixes
                    is_model_specific = any(
                        file_name.startswith(p) for p in possible_prefixes
                    ) and (
                        file_name.endswith(".profile")
                        or file_name.endswith(".cache")
                        or file_name.endswith(".timing")
                    )

                    # Catch exact generic names (like DFM's "timing.cache")
                    is_generic_timing = file_name == "timing.cache"

                    # Catch ORT's global architecture-based timing caches
                    is_ort_global_timing = file_name.startswith(
                        "TensorrtExecutionProvider_"
                    ) and (
                        file_name.endswith(".timing") or file_name.endswith(".profile")
                    )

                    if is_model_specific or is_generic_timing or is_ort_global_timing:
                        target_path = os.path.join(cache_dir, file_name)
                        if os.path.isfile(target_path):
                            try:
                                os.remove(target_path)
                                print(
                                    f"[INFO] Deleted TensorRT auxiliary file: {target_path}"
                                )
                            except Exception as e:
                                print(
                                    f"[WARN] Failed to delete auxiliary file {target_path}: {e}"
                                )
            except Exception as e:
                print(f"[WARN] Failed to clean auxiliary files in {cache_dir}: {e}")

    def load_model(self, model_name: str, session_options: Any = None) -> Any | None:
        """
        Loads an AI model (ONNX) with thread safety.
        Handles checking for existing TensorRT caches and launching the build probe if needed.
        """
        with self.model_lock:
            if self.models.get(model_name):
                return self.models[model_name]

            model_instance = None
            onnx_path = self.models_path.get(model_name)
            if not onnx_path:
                print(
                    f"[ERROR] Model path for '{model_name}' not found in models_data."
                )
                return None

            # Some models need a shape-inferred graph before the TensorRT EP can
            # build an engine. This transparently swaps in a cached sidecar; the
            # original path stays untouched for download/integrity checks.
            onnx_path = self._ensure_trt_ready_onnx(model_name, onnx_path)

            build_was_triggered = (
                False  # MP-05: flag to track if build dialog was shown
            )

            # --- DYNAMIC PRECISION CONFIGURATION (WHITELIST FP16) ---
            model_trt_options = dict(self.trt_ep_options)

            # --- DETECT TRT CACHE STATE ---
            cache_state = self._check_tensorrt_cache_state(model_name, onnx_path)

            if cache_state == "LEGACY":
                print(
                    f"[INFO] Legacy TRT cache detected for {model_name}. Bypassing explicit prefix."
                )
                # Remove prefix so ONNX Runtime loads generic TensorrtExecutionProvider_ files
                model_trt_options.pop("trt_engine_cache_prefix", None)
            else:
                # For EXPLICIT caches or brand new builds (None), strictly set custom prefix
                model_trt_options["trt_engine_cache_prefix"] = model_name

            # Check if the model is explicitly marked as safe for FP16 in models_data.py
            if model_name in fp16_safe_models_list:
                model_trt_options["trt_fp16_enable"] = True
                print(f"[INFO] FP16 Acceleration ENABLED for {model_name}")
            else:
                model_trt_options["trt_fp16_enable"] = False

            # Reconstruct the providers with model-specific options
            model_providers = []
            for p in self.providers:
                if isinstance(p, tuple) and p[0] == "TensorrtExecutionProvider":
                    model_providers.append(
                        ("TensorrtExecutionProvider", model_trt_options)
                    )
                elif p == "TensorrtExecutionProvider":
                    model_providers.append(
                        ("TensorrtExecutionProvider", model_trt_options)
                    )
                else:
                    model_providers.append(p)

            is_tensorrt_load = any(
                (p[0] if isinstance(p, tuple) else p) == "TensorrtExecutionProvider"
                for p in model_providers
            )

            if onnx_path.lower().endswith(".onnx"):
                # Only run the isolated probe if TensorRT is the target provider
                if is_tensorrt_load:
                    cache_is_valid = cache_state is not None

                    # If no engine config file or cache file exists run the probe
                    if not cache_is_valid:
                        print(
                            f"[INFO] TensorRT load detected for {model_name}. Running isolated probe..."
                        )

                        try:
                            # We emit signals to ask the main GUI thread to show the dialog.
                            dialog_title = "Building TensorRT Cache"
                            dialog_text = (
                                f"Building TensorRT engine cache for:\n"
                                f"{os.path.basename(onnx_path)}\n\n"
                                f"This may take several minutes.\n"
                                f"The application will continue once finished."
                            )

                            # The trt engine build worker process use this SessionOptions
                            # to use only 1 thread for building engines
                            sess_options_dict = {"intra_op_num_threads": 1}

                            # Ask the main thread to show the dialog
                            self.show_build_dialog.emit(dialog_title, dialog_text)

                            probe_successful = False
                            last_exit_code = None
                            max_retries = 3

                            for attempt in range(max_retries):
                                print(
                                    f"[INFO] Probe attempt {attempt + 1} of {max_retries} for {model_name}..."
                                )

                                # Use 'spawn' context for CUDA/TRT safety
                                ctx = multiprocessing.get_context("spawn")
                                # Pass full providers list (with tuples) so the worker
                                # can reconstruct them with device_id options.
                                current_providers_list = list(model_providers)
                                probe_process = ctx.Process(
                                    target=_probe_onnx_model_worker,
                                    args=(
                                        onnx_path,
                                        current_providers_list,
                                        model_trt_options,
                                        sess_options_dict,
                                    ),
                                )

                                try:
                                    probe_process.start()
                                    build_was_triggered = True

                                    # Timeout at 15 minutes to recover if compiler locks up
                                    probe_process.join(timeout=900)

                                    if probe_process.is_alive():
                                        print(
                                            f"[ERROR] Probe process for {model_name} timed out! Terminating."
                                        )
                                        probe_process.terminate()
                                        probe_process.join()

                                        # Clean up corrupted caches caused by the timeout before raising
                                        print(
                                            f"[INFO] Cleaning up corrupted TensorRT cache for {model_name} due to timeout..."
                                        )
                                        self._clean_tensorrt_cache(
                                            onnx_path, model_trt_options
                                        )

                                        raise RuntimeError(
                                            "TensorRT Engine build timed out."
                                        )
                                except Exception as e:
                                    print(f"[ERROR] Process execution failed: {e}")

                                # Process finished, get exit code
                                exitcode = probe_process.exitcode
                                last_exit_code = exitcode

                                if exitcode == 0:
                                    print(
                                        f"[INFO] Probe successful for {model_name}. Cache should be built."
                                    )
                                    probe_successful = True
                                    break  # Exit the retry loop on success
                                else:
                                    print(
                                        f"[WARN] Probe attempt {attempt + 1} failed with exit code {exitcode}."
                                    )

                                    # Wipe corrupted artifacts before attempting the next retry
                                    print(
                                        f"[INFO] Cleaning up potentially corrupted TensorRT cache for {model_name}..."
                                    )
                                    self._clean_tensorrt_cache(
                                        onnx_path, model_trt_options
                                    )

                                    if attempt < max_retries - 1:
                                        print("[INFO] Retrying in 2 seconds...")
                                        time.sleep(2.0)

                            if not probe_successful:
                                raise RuntimeError(
                                    f"[ERROR] ONNX/TensorRT probe process failed after {max_retries} attempts. Last exit code: {last_exit_code}"
                                )

                        except Exception:
                            # MP-05: only emit hide_build_dialog when build was triggered
                            if build_was_triggered:
                                self.hide_build_dialog.emit()

                            print(f"[ERROR] Isolated probe failed for {model_name}.")
                            print(
                                "[ERROR] The model will not be loaded. This is likely a fatal TensorRT/CUDA error."
                            )
                            traceback.print_exc()
                            self.models[model_name] = (
                                None  # Ensure it's marked as not loaded
                            )
                            return None  # Abort the load

            # Now, proceed with the *actual* load in the main thread.
            try:
                # MP-01: Double-checked load after re-acquiring the lock.
                # Another thread may have loaded this model while we were in the probe.
                if self.models.get(model_name):
                    print(
                        f"[INFO] Skipped loading: {model_name} is already loaded in memory (post-probe check)."
                    )
                    return self.models.get(model_name)

                if session_options is None:
                    session_options = onnxruntime.SessionOptions()

                # Force log_severity_level to 3 (ERROR) for the actual load as well, to suppress non-critical warnings from ONNX Runtime that can clutter the console.
                session_options.log_severity_level = 3

                model_instance = onnxruntime.InferenceSession(
                    onnx_path,
                    sess_options=session_options,
                    providers=model_providers,
                )

                # This ensures the CUDA context is synchronized after a new TRT
                # engine build, before we try to load it.
                if build_was_triggered:
                    if torch.cuda.is_available():
                        # Only synchronise current stream
                        torch.cuda.current_stream().synchronize()

                    # Check cache AGAIN.
                    # If the probe succeeded BUT the cache STILL doesn't exist,
                    # it's a "Lazy Build" model.
                    if self._check_tensorrt_cache_state(model_name, onnx_path) is None:
                        print(
                            f"[INFO] Model {model_name} requires a lazy build (engine not found after probe)."
                        )
                        self.models_pending_build.add(model_name)

                self.models[model_name] = model_instance
                print(
                    f"[INFO] Loading model: {model_name} with provider: {self.provider_name}"
                )
                if model_name == "Inswapper128":
                    graph = onnx.load(self.models_path[model_name]).graph
                    emap_initializer = None
                    for initializer in graph.initializer:
                        if initializer.name == "emap":
                            emap_initializer = initializer
                            break

                    if emap_initializer:
                        self.emap = onnx.numpy_helper.to_array(emap_initializer)
                    else:
                        self.emap = onnx.numpy_helper.to_array(graph.initializer[-1])
                    # MP-17: release large ONNX graph object after emap extraction
                    del graph
                    gc.collect()
                return model_instance

            except Exception:
                # This catch is still valuable for non-fatal errors
                print(f"[ERROR] Failed to load model {model_name} (even after probe).")
                traceback.print_exc()
                if model_instance is not None:
                    del model_instance
                    gc.collect()
                self.models[model_name] = None
                return None

            finally:
                # MP-05: Only emit hide_build_dialog when a build was triggered.
                if build_was_triggered:
                    self.hide_build_dialog.emit()

    def check_and_clear_pending_build(self, model_name: str) -> bool:
        """
        Checks if a model is pending its first-run lazy build.
        If it is, it clears the flag and returns True.
        """
        with self.model_lock:
            if model_name in self.models_pending_build:
                print(
                    f"[INFO] Model '{model_name}' is triggering its first-run lazy build."
                )
                # MP-08: use discard for atomic, safe removal (no KeyError)
                self.models_pending_build.discard(model_name)
                return True
        return False

    def load_dfm_model(self, dfm_model):
        """Loads a DeepFaceLab model instance."""
        with self.model_lock:
            if self.dfm_models.get(dfm_model):
                return self.dfm_models[dfm_model]

            self.main_window.model_loading_signal.emit()
            try:
                max_models_to_keep = self.main_window.control["MaxDFMModelsSlider"]
                total_loaded_models = len(self.dfm_models)
                # Ensure max_models_to_keep > 0 to avoid evicting when set to 0 (unlimited)
                if total_loaded_models >= max_models_to_keep and max_models_to_keep > 0:
                    print("[INFO] Clearing DFM Model (max capacity reached)")
                    model_name, model_instance = list(self.dfm_models.items())[0]
                    del model_instance
                    self.dfm_models.pop(model_name)
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                # --- Isolate TensorRT cache and bypass DFM internal garbage names ---
                import copy
                import re
                import os

                dfm_providers = copy.deepcopy(self.providers)

                if (
                    dfm_providers
                    and isinstance(dfm_providers[0], tuple)
                    and dfm_providers[0][0] == "TensorrtExecutionProvider"
                ):
                    trt_options = dict(dfm_providers[0][1])

                    # 1. Clean the filename to create a safe string
                    safe_name = re.sub(r"[^A-Za-z0-9_.-]", "", dfm_model)
                    if safe_name.lower().endswith(".dfm"):
                        safe_name = safe_name[:-4]

                    # 2. Use absolute paths for the dedicated cache directory to avoid OS pathing bugs
                    cache_base = os.path.abspath("tensorrt-engines")
                    dedicated_cache_dir = os.path.join(
                        cache_base, "dfm_caches", safe_name
                    )
                    os.makedirs(dedicated_cache_dir, exist_ok=True)

                    # 3. Route cache paths
                    trt_options["trt_engine_cache_path"] = dedicated_cache_dir
                    trt_options["trt_timing_cache_path"] = os.path.join(
                        dedicated_cache_dir, "timing.cache"
                    )

                    # 4. Override the internal ONNX model name.
                    trt_options["trt_engine_cache_prefix"] = safe_name

                    # 5. Disable Context dumping for DFM.
                    trt_options["trt_dump_ep_context_model"] = False
                    if "trt_ep_context_file_path" in trt_options:
                        del trt_options["trt_ep_context_file_path"]

                    dfm_providers[0] = ("TensorrtExecutionProvider", trt_options)

                self.dfm_models[dfm_model] = DFMModel(
                    self.main_window.dfm_model_manager.get_models_data()[dfm_model],
                    dfm_providers,
                    self.device,
                    self.gpu_id,
                )
            except Exception:
                print(f"[ERROR] Failed to load DFM model {dfm_model}.")
                traceback.print_exc()
                self.dfm_models[dfm_model] = None
            finally:
                self.main_window.model_loaded_signal.emit()

            return self.dfm_models.get(dfm_model)

    def delete_models(self):
        """Unloads all ONNX models."""
        model_names_to_unload = list(self.models.keys())
        for model_name in model_names_to_unload:
            self.unload_model(model_name)
        self.clip_session = []

    def delete_models_dfm(self):
        """Unloads all DFM models."""
        model_names_to_unload = list(self.dfm_models.keys())
        for model_name in model_names_to_unload:
            self.unload_dfm_model(model_name)

    def unload_dfm_model(self, model_name_to_unload, force_immediate=False):
        """
        Unloads a single DFM model instance from memory.

        Respects the KeepModelsAliveToggle control unless a force-unload is in progress.
        Frees the Python object, runs gc.collect(), and clears the CUDA cache.
        """
        # Check if unloading should be skipped
        if not self.force_unload_in_progress:
            if self.main_window.control.get("KeepModelsAliveToggle", False):
                return  # Skip unloading

        # --- SMART UNLOAD: Intercept if video is playing ---
        if not force_immediate and not self.force_unload_in_progress:
            vp = getattr(self.main_window, "video_processor", None)
            if vp and getattr(vp, "processing", False):
                # Video is playing, get the feeder's current frame and defer
                target_frame = getattr(vp, "current_frame_number", 0) + 1
                with self.model_lock:
                    self.deferred_unloads[model_name_to_unload] = {
                        "type": "dfm",
                        "target_frame": target_frame,
                    }
                print(
                    f"[INFO] Smart Unload: Deferring DFM '{model_name_to_unload}' unload after frame {target_frame}"
                )
                return

        with self.model_lock:
            if (
                model_name_to_unload
                and model_name_to_unload in self.dfm_models
                and self.dfm_models.get(model_name_to_unload) is not None
            ):
                print(f"[INFO] Unloading DFM model: {model_name_to_unload}")
                model_instance = self.dfm_models.pop(model_name_to_unload, None)
                if model_instance:
                    del model_instance
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    def unload_model(self, model_name_to_unload, force_immediate=False):
        """
        Unloads a single ONNX model from memory.

        Handles the ``self.models`` (ONNX) dictionary. Respects the KeepModelsAliveToggle
        control unless a force-unload is in progress. Frees the Python object, runs
        gc.collect(), and clears the CUDA cache when something was actually unloaded.
        """
        # Check if unloading should be skipped
        if not self.force_unload_in_progress:
            if self.main_window.control.get("KeepModelsAliveToggle", False):
                return  # Skip unloading

        # --- SMART UNLOAD: Intercept if video is playing ---
        if not force_immediate and not self.force_unload_in_progress:
            vp = getattr(self.main_window, "video_processor", None)
            if vp and getattr(vp, "processing", False):
                # Video is playing, get the feeder's current frame and defer
                target_frame = getattr(vp, "current_frame_number", 0) + 1
                with self.model_lock:
                    self.deferred_unloads[model_name_to_unload] = {
                        "type": "onnx",
                        "target_frame": target_frame,
                    }
                print(
                    f"[INFO] Smart Unload: Deferring ONNX '{model_name_to_unload}' unload after frame {target_frame}"
                )
                return

        with self.model_lock:
            unloaded = False

            # Handle ONNX models (for CUDA, CPU, and TensorRT providers)
            if model_name_to_unload and model_name_to_unload in self.models:
                model_instance = self.models[model_name_to_unload]

                if model_instance is not None:
                    print(f"[INFO] Unloading ONNX model: {model_name_to_unload}")
                    # MP-06: set dict entry to None first, then del the instance
                    self.models[model_name_to_unload] = None
                    # Explicitly delete the object to trigger its __del__ method
                    del model_instance
                    unloaded = True
                else:
                    self.models[model_name_to_unload] = None

            if unloaded:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    def is_model_active_in_ui(self, model_name: str) -> bool:
        """
        Live Verification JIT: Checks the UI state dynamically to see if a model is still needed.
        Used exclusively before a deferred unload to prevent accidental purging.
        """
        from app.ui.widgets.models_toggle_data import MODELS_TOGGLE_MAP

        # Thread Safety: Use dict() to safely create shallow copies and avoid 'dictionary changed size' errors
        try:
            params = dict(getattr(self.main_window, "parameters", {}))
            ctrl = dict(getattr(self.main_window, "control", {}))
        except Exception:
            params = getattr(self.main_window, "parameters", {})
            ctrl = getattr(self.main_window, "control", {})

        # Create a unified flat dictionary for global lookups
        live_global = (
            {**params, **ctrl}
            if isinstance(params, dict) and isinstance(ctrl, dict)
            else params
        )

        # 1. SPECIAL CASE: FACE RESTORERS
        if hasattr(self.main_window.function_worker, "face_restorers") and hasattr(
            self.main_window.function_worker.face_restorers, "model_map"
        ):
            expected_combo = None
            for (
                combo_str,
                ort_name,
            ) in self.main_window.function_worker.face_restorers.model_map.items():
                if ort_name == model_name:
                    expected_combo = combo_str
                    break

            if expected_combo is not None:

                def is_restorer_requested(p) -> bool:
                    if not hasattr(p, "get"):
                        return False

                    if (
                        p.get("FaceRestorerEnableToggle", False)
                        and p.get("FaceRestorerTypeSelection") == expected_combo
                    ):
                        return True
                    if p.get("FaceRestorerEnable2Toggle", False) and not p.get(
                        "FaceRestorerEnable2EndToggle", False
                    ):
                        if p.get("FaceRestorerType2Selection") == expected_combo:
                            return True
                    if (
                        p.get("FaceRestorerEnable2EndToggle", False)
                        and p.get("FaceRestorerType2Selection") == expected_combo
                    ):
                        return True
                    return False

                # 1. Global check
                if is_restorer_requested(live_global):
                    return True

                # 2. Exhaustive verification for each active face in memory
                if hasattr(params, "items"):
                    for face_id, face_params in params.items():
                        if is_restorer_requested(face_params):
                            return True

                return False  # No face requested this restorer, we can safely unload it

        # 2. GENERAL CASE: MODELS TOGGLE MAP
        toggles = MODELS_TOGGLE_MAP.get(model_name)
        if not toggles:
            return True  # Core models without UI toggles are always assumed needed

        for toggle in toggles:
            target_key = toggle.key

            if hasattr(ctrl, "get") and ctrl.get(target_key, False):
                return True

            if hasattr(params, "get") and params.get(target_key, False):
                return True

            if hasattr(params, "items"):
                for face_id, face_params in params.items():
                    if hasattr(face_params, "get") and face_params.get(
                        target_key, False
                    ):
                        return True

        return False

    def check_deferred_unloads(self, current_displayed_frame: int):
        """
        Checks if any pending model unloads have reached their failsafe trigger.
        Includes a Just-In-Time (JIT) UI state verification.
        """
        if not self.deferred_unloads:
            return

        with self.model_lock:
            to_unload = []
            for model_name, data in self.deferred_unloads.items():
                if current_displayed_frame >= data["target_frame"]:
                    to_unload.append((model_name, data["type"]))

            for model_name, m_type in to_unload:
                # Remove from pending list regardless
                del self.deferred_unloads[model_name]

                # --- JIT LIVE CHECK ---
                if self.is_model_active_in_ui(model_name):
                    print(
                        f"[INFO] Smart Unload JIT: Option re-enabled for '{model_name}'. Cancelling unload."
                    )
                    continue  # Safe! We skip the unload.

                print(
                    f"[INFO] Smart Unload: Trigger reached. Actually releasing '{model_name}'."
                )
                if m_type == "onnx":
                    self.unload_model(model_name, force_immediate=True)
                elif m_type == "dfm":
                    self.unload_dfm_model(model_name, force_immediate=True)
                elif m_type == "kv":
                    self.face_denoiser.unload_kv_extractor(force_immediate=True)

    def execute_all_deferred_unloads(self):
        """
        Forces the immediate unload of all deferred models.
        Ideal for the 'Stop' function (Smart Stop).
        Includes JIT verification to keep re-enabled models.
        """
        with self.model_lock:
            if not self.deferred_unloads:
                return

            print("[INFO] Smart Stop: Executing pending UI unloads...")
            for model_name, data in list(self.deferred_unloads.items()):
                del self.deferred_unloads[model_name]

                # --- LIVE UI VERIFICATION (JIT) ---
                if self.is_model_active_in_ui(model_name):
                    print(
                        f"[INFO] Smart Stop JIT: Option is active for '{model_name}'. Keeping model in VRAM."
                    )
                    continue  # We skip the unload, leaving the model ready for the next run!

                print(f"[INFO] Smart Stop: Releasing '{model_name}'.")
                if data["type"] == "onnx":
                    self.unload_model(model_name, force_immediate=True)
                elif data["type"] == "dfm":
                    self.unload_dfm_model(model_name, force_immediate=True)
                elif data["type"] == "kv":
                    self.face_denoiser.unload_kv_extractor(force_immediate=True)

    def showModelLoadingProgressBar(self):
        """Shows the model-loading progress dialog in the UI."""
        self.main_window.model_load_dialog.show()

    def hideModelLoadProgressBar(self):
        """Closes the model-loading progress dialog if it is open."""
        if self.main_window.model_load_dialog:
            self.main_window.model_load_dialog.close()

    def set_number_of_threads(self, value):
        """Sets the ONNX thread count. TRT engine reloading is no longer needed here."""
        self.nThreads = value

    def get_gpu_memory(self):
        """
        Returns GPU memory usage as ``(used_MB, total_MB)``.

        Queries nvidia-smi for accuracy; falls back to ``torch.cuda`` device properties
        if nvidia-smi is unavailable.  Returns ``(0, 0)`` when no GPU is detected.
        """
        # MP-13: use a single nvidia-smi call for both total and free memory
        try:
            command = f"nvidia-smi --id={self.gpu_id} --query-gpu=memory.total,memory.free --format=csv,noheader,nounits"
            output = sp.check_output(command.split()).decode("ascii").strip()
            # Output format: "total, free" (one line per GPU)
            first_line = output.split("\n")[0]
            parts = first_line.split(",")
            memory_total_val = int(parts[0].strip())
            memory_free_val = int(parts[1].strip())
            memory_used = memory_total_val - memory_free_val
            return memory_used, memory_total_val
        except Exception:
            # Fallback to torch.cuda if nvidia-smi is unavailable
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(self.gpu_id)
                memory_total_val = props.total_memory // (1024 * 1024)
                memory_free_val = (
                    props.total_memory - torch.cuda.memory_reserved(self.gpu_id)
                ) // (1024 * 1024)
                memory_used = memory_total_val - memory_free_val
                return memory_used, memory_total_val
            return 0, 0

    def update_provider_configuration(self, provider_name: str) -> str:
        """
        Updates the internal device and provider lists.
        Called exclusively by the FunctionWorker after it safely unloads active models.
        """
        match provider_name:
            case "TensorRT" | "TensorRT-Engine":
                if not TENSORRT_AVAILABLE or trt is None:
                    raise RuntimeError("TensorRT is not installed.")
                providers = [
                    ("TensorrtExecutionProvider", self.trt_ep_options),
                    ("CUDAExecutionProvider", {"device_id": self.gpu_id}),
                    ("CPUExecutionProvider"),
                ]
                self.device = f"cuda:{self.gpu_id}"
                self.device_type = "cuda"
                if (
                    version.parse(trt.__version__) < version.parse("10.2.0")
                    and provider_name == "TensorRT-Engine"
                ):
                    print(
                        "[WARN] TensorRT-Engine provider cannot be used when TensorRT version is lower than 10.2.0."
                    )
                    provider_name = "TensorRT"

            case "CPU":
                providers = ["CPUExecutionProvider"]
                self.device = "cpu"
                self.device_type = "cpu"

            case "CUDA":
                providers = [
                    ("CUDAExecutionProvider", {"device_id": self.gpu_id}),
                    ("CPUExecutionProvider"),
                ]
                self.device = f"cuda:{self.gpu_id}"
                self.device_type = "cuda"

            case _:
                raise ValueError(f"Unknown provider: {provider_name}")

        self.providers = providers
        self.provider_name = provider_name
        return self.provider_name
