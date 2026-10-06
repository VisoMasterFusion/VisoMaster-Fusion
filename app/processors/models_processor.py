import threading
import os
import subprocess as sp
import gc
import json
import re
import sys
import tempfile
import time
import traceback
from typing import Dict, TYPE_CHECKING, Any, Optional
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
from app.processors.utils import platform_support

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
from app.helpers.build_progress import BuildTimeStore, StageTracker
from app.processors.models_data import (
    models_dir,
    models_list,
    compound_models_mapping,
    restorer_model_mapping,
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


# --- Isolated Process Worker ---
# The ONNX/TensorRT engine-build probe lives in
# app/processors/onnx_probe_runner.py and is launched as a subprocess with
# piped output (see _run_build_probe), so the build log can be streamed to
# the progress dialog while the build runs.


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

    # Rich TensorRT engine-build progress dialog (app/ui/widgets/trt_build_dialog.py).
    # Emitted from arbitrary worker threads and connected to the internal
    # slots below; because this QObject has GUI-thread affinity, the slots run
    # in the GUI thread and are the only place the dialog is touched.
    # Arguments: (window_title, model_label, expected_build_seconds, build_number)
    build_dialog_show = QtCore.Signal(str, str, float, int)
    build_dialog_log = QtCore.Signal(str)  # one streamed probe log line
    build_dialog_stage = QtCore.Signal(str)  # human-readable build phase
    build_dialog_hide = QtCore.Signal()

    def __init__(self, main_window: "MainWindow", device: str = "") -> None:
        """
        Initialises the ModelsProcessor.

        Sets up all model dictionaries, TensorRT options, provider lists,
        and helper state (locks, sync vectors). Sub-processors are managed externally.

        Args:
            main_window: The application's MainWindow, used to access UI controls and signals.
            device: Torch/ONNX device string — ``"cuda"``, ``"mps"`` or ``"cpu"``.
                Defaults to the best backend this machine actually has.
        """
        super().__init__()
        self.main_window = main_window
        self.gpu_id = getattr(main_window, "gpu_id", 0)
        device = device or platform_support.default_torch_device()
        self.provider_name = platform_support.default_execution_provider()

        # NOTE: internal_deep_copied_kv_map / internal_kv_map_source_filename were
        # placeholder attributes for a planned per-session KV-map cache.  They are
        # currently unused (never written after __init__).  If a future feature
        # populates them, ensure a matching cleanup path is added to the force-unload
        # path (delete_models_dfm / force_unload path) so the tensors are freed.
        self.internal_deep_copied_kv_map: Dict[str, Dict[str, torch.Tensor]] | None = (
            None
        )
        self.internal_kv_map_source_filename: str | None = None
        # `device` may arrive bare ("cuda") from a caller or already indexed
        # ("cuda:0") from default_torch_device(); normalise before use.
        # device_type must stay bare: it is handed straight to ONNX Runtime as
        # io_binding(device_type=...), which rejects "cuda:0".
        device_type = device.split(":", 1)[0]
        # Only CUDA has addressable per-index devices; "mps" and "cpu" are bare.
        self.device = (
            f"{device_type}:{self.gpu_id}" if device_type == "cuda" else device_type
        )
        self.device_type = device_type
        if self.gpu_id != 0 and device_type == "cuda":
            torch.cuda.set_device(self.gpu_id)
        self.model_lock = threading.RLock()  # Reentrant lock for model access

        self.cuda_graph_capture_lock = threading.Lock()

        # --- TENSORRT WORKSPACE ---
        MIN_WORKSPACE_SIZE = 1073741824  # 1 GB
        FALLBACK_WORKSPACE_SIZE = 4294967296  # 4 GB

        workspace_size = FALLBACK_WORKSPACE_SIZE

        # Prevent silent C++ driver crashes by ensuring CUDA is requested
        # and physically available before querying device properties.
        if self.device_type == "cuda" and torch.cuda.is_available():
            try:
                # Get total GPU memory in bytes
                total_vram = torch.cuda.get_device_properties(self.gpu_id).total_memory
                # Safely allocate 40% of total VRAM for TensorRT workspace
                calculated_workspace = int(total_vram * 0.40)
                # Enforce a minimum of 1 GB to avoid compilation failures on very low-end GPUs
                workspace_size = max(calculated_workspace, MIN_WORKSPACE_SIZE)
            except Exception as e:
                print(f"[WARN] Failed to retrieve CUDA properties: {e}")
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

        # Default CoreML options (macOS). MLProgram is the modern model format and
        # is required for fp16 compute; ALL lets CoreML place ops on the Neural
        # Engine / GPU / CPU as it sees fit.
        #
        # RequireStaticInputShapes is not optional. Several models here (RetinaFace
        # / det_10g among them) have dynamic dimensions, and CoreML silently
        # produces wrong-shaped outputs for those subgraphs — inference fails with
        # "Invalid shape for output feature". Restricting CoreML to statically
        # shaped subgraphs makes it hand the dynamic ones back to the CPU EP, which
        # is both correct and, for those specific graphs, no slower.
        self.coreml_ep_options: Dict[str, Any] = {
            "ModelFormat": "MLProgram",
            "MLComputeUnits": "ALL",
            "RequireStaticInputShapes": "1",
            "AllowLowPrecisionAccumulationOnGPU": "1",
        }

        # A set to keep track of models that have been loaded but
        # have not had their engine built (lazy build).
        self.models_pending_build: set = set()

        # --- TENSORRT BUILD PROGRESS STATE ---
        # The dialog is created lazily on first show, always in the GUI thread.
        self._trt_build_dialog: Any = None
        # Set by the dialog's Cancel button; polled by the probe wait loop.
        self._build_cancel_event = threading.Event()
        # How many engine builds were triggered this session (dialog counter).
        self._trt_build_session_count = 0
        # Remembers past per-model build durations for the dialog's estimate.
        self._build_time_store = BuildTimeStore(
            os.path.join(str(models_dir), "trt_build_times.json")
        )
        self.build_dialog_show.connect(self._on_build_dialog_show)
        self.build_dialog_log.connect(self._on_build_dialog_log)
        self.build_dialog_stage.connect(self._on_build_dialog_stage)
        self.build_dialog_hide.connect(self._on_build_dialog_hide)
        self.providers: list = self._default_providers()
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
        self.emap: list[Any] | np.ndarray = []
        self.face_denoiser: Any = None

        self.normalize = v2.Normalize(
            mean=[0.0, 0.0, 0.0], std=[1 / 1.0, 1 / 1.0, 1 / 1.0]
        )

    @property
    def binding_device_id(self) -> int:
        return self.gpu_id if self.device_type != "cpu" else 0

    # --- TensorRT build progress dialog plumbing ---
    # These slots are connected in __init__ and, thanks to this QObject's
    # GUI-thread affinity, always execute in the GUI thread no matter which
    # worker thread emitted the signal.

    @QtCore.Slot(str, str, float, int)
    def _on_build_dialog_show(
        self, title: str, model_label: str, expected_seconds: float, build_number: int
    ) -> None:
        if self._trt_build_dialog is None:
            # Imported here so that importing this module never requires a
            # running QApplication (unit tests construct ModelsProcessor
            # headlessly).
            from app.ui.widgets.trt_build_dialog import TrtBuildDialog

            self._trt_build_dialog = TrtBuildDialog(self.main_window)
            self._trt_build_dialog.cancel_requested.connect(
                self._build_cancel_event.set
            )
        self._build_cancel_event.clear()
        self._trt_build_dialog.start_build(
            title, model_label, expected_seconds, build_number
        )

    @QtCore.Slot(str)
    def _on_build_dialog_log(self, line: str) -> None:
        if self._trt_build_dialog is not None:
            self._trt_build_dialog.append_log_line(line)

    @QtCore.Slot(str)
    def _on_build_dialog_stage(self, stage_label: str) -> None:
        if self._trt_build_dialog is not None:
            self._trt_build_dialog.set_stage(stage_label)

    @QtCore.Slot()
    def _on_build_dialog_hide(self) -> None:
        if self._trt_build_dialog is not None:
            self._trt_build_dialog.finish()

    @staticmethod
    def _terminate_probe(probe_process) -> None:
        """Terminate a probe subprocess, escalating to kill() if needed."""
        probe_process.terminate()
        try:
            probe_process.wait(timeout=10)
        except sp.TimeoutExpired:
            probe_process.kill()
            try:
                probe_process.wait(timeout=10)
            except sp.TimeoutExpired:
                pass

    def _run_build_probe(
        self,
        onnx_path: str,
        providers_list: list,
        trt_options: Dict[str, Any],
        session_options_dict: Dict[str, Any],
        model_name: str,
        timeout_seconds: int = 900,
    ) -> int:
        """Run the isolated engine-build probe as a subprocess.

        Returns the probe's exit code (0 = cache built). While the probe runs,
        its output is streamed line by line to the console (as before) *and*
        into the progress dialog, and the wait loop keeps the UI responsive
        and honours the dialog's Cancel button.
        """
        config = {
            "model_path": onnx_path,
            "providers": providers_list,
            "trt_options": trt_options,
            "session_options": session_options_dict,
        }

        config_path = None
        probe_process = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".json",
                prefix="vm_probe_",
                delete=False,
                encoding="utf-8",
            ) as handle:
                json.dump(config, handle)
                config_path = handle.name

            runner_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "onnx_probe_runner.py"
            )
            # -u: unbuffered output, so build log lines reach the dialog
            # immediately instead of accumulating in the pipe buffer.
            probe_process = sp.Popen(
                [sys.executable, "-u", runner_path, config_path],
                stdout=sp.PIPE,
                stderr=sp.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )

            # Dedicated reader thread: echo lines to the console (previous
            # behavior), feed the dialog's log tail, and advance the build
            # stage label when a known phase marker scrolls by.
            stage_tracker = StageTracker()

            def _pump_output() -> None:
                assert probe_process.stdout is not None
                for line in probe_process.stdout:
                    stripped = line.rstrip("\r\n")
                    print(stripped, flush=True)  # keep console behavior
                    self.build_dialog_log.emit(stripped)
                    new_stage = stage_tracker.update(stripped)
                    if new_stage is not None:
                        self.build_dialog_stage.emit(new_stage)

            reader = threading.Thread(target=_pump_output, daemon=True)
            reader.start()

            deadline = time.monotonic() + timeout_seconds
            app = QtCore.QCoreApplication.instance()
            on_gui_thread = (
                app is not None and QtCore.QThread.currentThread() == app.thread()
            )
            while probe_process.poll() is None:
                if self._build_cancel_event.is_set():
                    print(f"[INFO] Engine build for {model_name} cancelled by user.")
                    self._terminate_probe(probe_process)
                    self._clean_tensorrt_cache(onnx_path, trt_options)
                    raise RuntimeError("TensorRT engine build cancelled by user.")
                if time.monotonic() > deadline:
                    # Recover if the compiler locks up.
                    print(
                        f"[ERROR] Probe process for {model_name} timed out! Terminating."
                    )
                    self._terminate_probe(probe_process)
                    # Clean up corrupted caches caused by the timeout.
                    print(
                        f"[INFO] Cleaning up corrupted TensorRT cache for {model_name} due to timeout..."
                    )
                    self._clean_tensorrt_cache(onnx_path, trt_options)
                    raise RuntimeError("TensorRT Engine build timed out.")
                if on_gui_thread:
                    # load_model is sometimes called synchronously from the GUI
                    # thread; keep the event loop pumping so Windows never
                    # marks the window "Not Responding" during long builds.
                    QtCore.QCoreApplication.processEvents()
                time.sleep(0.1)

            reader.join(timeout=5)
            return (
                probe_process.returncode
                if probe_process.returncode is not None
                else 1
            )
        finally:
            if probe_process is not None and probe_process.poll() is None:
                self._terminate_probe(probe_process)
            if config_path:
                try:
                    os.remove(config_path)
                except OSError:
                    pass

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

    def load_model(
        self,
        model_name: str | tuple[str, ...] | list[str],
        session_options: Any = None,
    ) -> Any | None:
        """
        Loads an AI model (ONNX) with thread safety.
        Handles checking for existing TensorRT caches and launching the build probe if needed.
        Recursively delegates compound pipelines (e.g. OSDFace) or sequences to their constituent sub-models.
        """
        # Defensive recursion: unpack sequence collections
        if isinstance(model_name, (tuple, list, set, frozenset)):
            compound_sessions: Dict[str, Any] = {}
            all_successful: bool = True
            for sub_name in model_name:
                sub_session = self.load_model(sub_name, session_options=session_options)
                if sub_session is None:
                    all_successful = False
                compound_sessions[sub_name] = sub_session
            return compound_sessions if all_successful else None

        canonical_name: str = restorer_model_mapping.get(model_name, model_name)

        # Decompose compound pipelines into atomic ONNX sessions
        if canonical_name in compound_models_mapping:
            compound_sessions = {}
            all_successful = True
            for sub_model_name in compound_models_mapping[canonical_name]:
                sub_session = self.load_model(
                    sub_model_name, session_options=session_options
                )
                if sub_session is None:
                    all_successful = False
                    print(
                        f"[WARN] Sub-model '{sub_model_name}' of compound '{canonical_name}' failed to load."
                    )
                compound_sessions[sub_model_name] = sub_session

            # Purge non-active restorers once the entire compound pipeline is safely resident
            if all_successful:
                self.purge_unused_restorers()

            return compound_sessions if all_successful else None

        with self.model_lock:
            if self.models.get(canonical_name):
                return self.models[canonical_name]

            model_instance = None
            onnx_path = self.models_path.get(canonical_name)
            if not onnx_path:
                print(
                    f"[ERROR] Model path for '{canonical_name}' not found in models_data."
                )
                return None

            # Some models need a shape-inferred graph before the TensorRT EP can
            # build an engine. This transparently swaps in a cached sidecar; the
            # original path stays untouched for download/integrity checks.
            onnx_path = self._ensure_trt_ready_onnx(canonical_name, onnx_path)

            build_was_triggered = (
                False  # MP-05: flag to track if build dialog was shown
            )

            # --- DYNAMIC PRECISION CONFIGURATION (WHITELIST FP16) ---
            model_trt_options = dict(self.trt_ep_options)

            # --- DETECT TRT CACHE STATE ---
            cache_state = self._check_tensorrt_cache_state(canonical_name, onnx_path)

            if cache_state == "LEGACY":
                print(
                    f"[INFO] Legacy TRT cache detected for {canonical_name}. Bypassing explicit prefix."
                )
                # Remove prefix so ONNX Runtime loads generic TensorrtExecutionProvider_ files
                model_trt_options.pop("trt_engine_cache_prefix", None)
            else:
                # For EXPLICIT caches or brand new builds (None), strictly set custom prefix
                model_trt_options["trt_engine_cache_prefix"] = canonical_name

            # Check if the model is explicitly marked as safe for FP16 in models_data.py
            if canonical_name in fp16_safe_models_list:
                model_trt_options["trt_fp16_enable"] = True
                print(f"[INFO] FP16 Acceleration ENABLED for {canonical_name}")
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
                            f"[INFO] TensorRT load detected for {canonical_name}. Running isolated probe..."
                        )

                        try:
                            # The trt engine build worker process use this SessionOptions
                            # to use only 1 thread for building engines
                            sess_options_dict = {"intra_op_num_threads": 1}

                            self._trt_build_session_count += 1
                            expected_seconds = (
                                self._build_time_store.expected_seconds(canonical_name)
                                or -1.0
                            )

                            # Ask the GUI thread to show the progress dialog.
                            self.build_dialog_show.emit(
                                "Building TensorRT Cache",
                                "Building TensorRT engine cache for:\n"
                                f"{os.path.basename(onnx_path)}",
                                float(expected_seconds),
                                self._trt_build_session_count,
                            )
                            build_was_triggered = True

                            probe_successful = False
                            last_exit_code = None
                            max_retries = 3

                            for attempt in range(max_retries):
                                print(
                                    f"[INFO] Probe attempt {attempt + 1} of {max_retries} for {canonical_name}..."
                                )
                                if attempt > 0:
                                    self.build_dialog_stage.emit(
                                        f"Retrying the build (attempt {attempt + 1} of {max_retries})..."
                                    )

                                build_started = time.monotonic()
                                # Pass the full providers list (with tuples) so the
                                # probe can reconstruct them with device_id options.
                                # Timeout of 15 minutes recovers from compiler lockups;
                                # the dialog's Cancel button aborts via RuntimeError.
                                exitcode = self._run_build_probe(
                                    onnx_path,
                                    list(model_providers),
                                    model_trt_options,
                                    sess_options_dict,
                                    canonical_name,
                                    timeout_seconds=900,
                                )
                                last_exit_code = exitcode

                                if exitcode == 0:
                                    # Remember how long the build took so the
                                    # dialog can estimate remaining time on
                                    # the next rebuild of this model.
                                    self._build_time_store.record(
                                        canonical_name, time.monotonic() - build_started
                                    )
                                    print(
                                        f"[INFO] Probe successful for {canonical_name}. Cache should be built."
                                    )
                                    probe_successful = True
                                    break  # Exit the retry loop on success
                                else:
                                    print(
                                        f"[WARN] Probe attempt {attempt + 1} failed with exit code {exitcode}."
                                    )

                                    # Wipe corrupted artifacts before attempting the next retry
                                    print(
                                        f"[INFO] Cleaning up potentially corrupted TensorRT cache for {canonical_name}..."
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
                                self.build_dialog_hide.emit()

                            print(
                                f"[ERROR] Isolated probe failed for {canonical_name}."
                            )
                            print(
                                "[ERROR] The model will not be loaded. This is likely a fatal TensorRT/CUDA error."
                            )
                            traceback.print_exc()
                            self.models[canonical_name] = (
                                None  # Ensure it's marked as not loaded
                            )
                            return None  # Abort the load

            # Now, proceed with the *actual* load in the main thread.
            try:
                # MP-01: Double-checked load after re-acquiring the lock.
                # Another thread may have loaded this model while we were in the probe.
                if self.models.get(canonical_name):
                    print(
                        f"[INFO] Skipped loading: {canonical_name} is already loaded in memory (post-probe check)."
                    )
                    return self.models.get(canonical_name)

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
                    if (
                        self._check_tensorrt_cache_state(canonical_name, onnx_path)
                        is None
                    ):
                        print(
                            f"[INFO] Model {canonical_name} requires a lazy build (engine not found after probe)."
                        )
                        self.models_pending_build.add(canonical_name)

                self.models[canonical_name] = model_instance
                print(
                    f"[INFO] Loading model: {canonical_name} with provider: {self.provider_name}"
                )
                if canonical_name == "Inswapper128":
                    graph = onnx.load(self.models_path[canonical_name]).graph
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

                # If an atomic restorer was loaded, purge any previously loaded restorers that are no longer active
                if canonical_name in restorer_model_mapping.values():
                    self.purge_unused_restorers()

                return model_instance

            except Exception:
                print(
                    f"[ERROR] Failed to load model {canonical_name} (even after probe)."
                )
                traceback.print_exc()
                if model_instance is not None:
                    del model_instance
                    gc.collect()
                self.models[canonical_name] = None
                return None

            finally:
                # MP-05: Only emit hide_build_dialog when a build was triggered.
                if build_was_triggered:
                    self.build_dialog_hide.emit()

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

    def unload_model(
        self,
        model_name_to_unload: str | tuple[str, ...] | list[str],
        force_immediate: bool = False,
    ) -> None:
        """
        Unloads a single ONNX model, collection of models, or compound pipeline from memory.
        Canonicalizes UI display names and decomposes compound multi-model architectures.
        """
        # Defensive recursion: unpack sequence collections
        if isinstance(model_name_to_unload, (tuple, list, set, frozenset)):
            for sub_name in model_name_to_unload:
                self.unload_model(sub_name, force_immediate=force_immediate)
            return

        if not self.force_unload_in_progress:
            if self.main_window.control.get("KeepModelsAliveToggle", False):
                return  # Skip unloading

        canonical_name: str = restorer_model_mapping.get(
            model_name_to_unload, model_name_to_unload
        )

        # Decompose compound pipelines into atomic sub-models
        if canonical_name in compound_models_mapping:
            if not force_immediate and not self.force_unload_in_progress:
                vp = getattr(self.main_window, "video_processor", None)
                if vp and getattr(vp, "processing", False):
                    target_frame = getattr(vp, "current_frame_number", 0) + 1
                    with self.model_lock:
                        self.deferred_unloads[canonical_name] = {
                            "type": "onnx",
                            "target_frame": target_frame,
                        }
                    print(
                        f"[INFO] Smart Unload: Deferring compound '{canonical_name}' unload after frame {target_frame}"
                    )
                    return

            for sub_name in compound_models_mapping[canonical_name]:
                self.unload_model(sub_name, force_immediate=force_immediate)
            return

        # Smart Unload: Intercept if video is playing
        if not force_immediate and not self.force_unload_in_progress:
            vp = getattr(self.main_window, "video_processor", None)
            if vp and getattr(vp, "processing", False):
                target_frame = getattr(vp, "current_frame_number", 0) + 1
                with self.model_lock:
                    self.deferred_unloads[canonical_name] = {
                        "type": "onnx",
                        "target_frame": target_frame,
                    }
                print(
                    f"[INFO] Smart Unload: Deferring ONNX '{canonical_name}' unload after frame {target_frame}"
                )
                return

        with self.model_lock:
            unloaded: bool = False

            if canonical_name and canonical_name in self.models:
                model_instance = self.models[canonical_name]

                if model_instance is not None:
                    print(f"[INFO] Unloading ONNX model: {canonical_name}")
                    # MP-06: set dict entry to None first, then del the instance
                    self.models[canonical_name] = None
                    # Explicitly delete the object to trigger its __del__ method
                    del model_instance
                    unloaded = True
                else:
                    self.models[canonical_name] = None

            if unloaded:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    def purge_unused_restorers(self) -> None:
        """
        Scans all loaded face restorer models against current UI state and unloads any that
        are not actively selected in Slot 1 or Slot 2 across any face or global configuration.
        """
        all_restorers = set(restorer_model_mapping.values())
        with self.model_lock:
            for canonical_name in all_restorers:
                if not isinstance(canonical_name, str):
                    continue

                if canonical_name in compound_models_mapping:
                    is_loaded = any(
                        self.models.get(sub) is not None
                        for sub in compound_models_mapping[canonical_name]
                    )
                else:
                    is_loaded = self.models.get(canonical_name) is not None

                if is_loaded and not self.is_model_active_in_ui(canonical_name):
                    print(
                        f"[INFO] Restorer '{canonical_name}' is no longer active in UI. Purging from VRAM."
                    )
                    self.unload_model(canonical_name, force_immediate=True)

    def is_model_active_in_ui(
        self, model_name: str | tuple[str, ...] | list[str]
    ) -> bool:
        """
        Live Verification JIT: Checks UI state dynamically before a deferred unload or purge.
        Accurately inspects default_parameters, per-face parameters, and control dicts.
        """
        from app.ui.widgets.models_toggle_data import MODELS_TOGGLE_MAP

        # Defensive recursion: if a collection of model names is passed, check if any is active
        if isinstance(model_name, (tuple, list, set, frozenset)):
            return any(self.is_model_active_in_ui(m) for m in model_name)

        default_params: Dict[str, Any] = {}
        if hasattr(self.main_window, "default_parameters"):
            dp = getattr(self.main_window, "default_parameters")
            if hasattr(dp, "data") and isinstance(dp.data, dict):
                default_params = dict(dp.data)
            elif isinstance(dp, dict):
                default_params = dict(dp)

        try:
            params = dict(getattr(self.main_window, "parameters", {}))
            ctrl = dict(getattr(self.main_window, "control", {}))
        except Exception:
            params = getattr(self.main_window, "parameters", {})
            ctrl = getattr(self.main_window, "control", {})

        live_global: Dict[str, Any] = {
            **default_params,
            **(ctrl if isinstance(ctrl, dict) else {}),
            **(params if isinstance(params, dict) else {}),
        }

        def _is_truthy(val: Any) -> bool:
            if isinstance(val, str):
                return val.lower() in ("true", "1", "yes", "on")
            return bool(val)

        # 1. SPECIAL CASE: FACE RESTORERS (Canonical & Compound Resolution)
        expected_combo: Optional[str] = None

        if model_name in compound_models_mapping:
            expected_combo = model_name
        else:
            for compound_parent, sub_models in compound_models_mapping.items():
                if model_name in sub_models:
                    expected_combo = compound_parent
                    break

        if expected_combo is None:
            for ui_combo, canonical in restorer_model_mapping.items():
                if canonical == model_name or ui_combo == model_name:
                    expected_combo = ui_combo
                    break

        if expected_combo is not None:

            def is_restorer_requested(p: Any) -> bool:
                if not hasattr(p, "get"):
                    return False

                if (
                    _is_truthy(p.get("FaceDetailerEnableToggle", False))
                    and p.get("FaceDetailerRestorerTypeSelection", "GPEN-1024")
                    == expected_combo
                ):
                    return True

                # Slot 1 Verification: Toggle enabled AND combo selection matches
                if (
                    _is_truthy(p.get("FaceRestorerEnableToggle", False))
                    and p.get("FaceRestorerTypeSelection") == expected_combo
                ):
                    return True

                # Slot 2 Verification: Toggle enabled AND combo selection matches
                if (
                    _is_truthy(p.get("FaceRestorerEnable2Toggle", False))
                    and p.get("FaceRestorerType2Selection") == expected_combo
                ):
                    return True

                return False

            if is_restorer_requested(live_global):
                return True
            if is_restorer_requested(default_params):
                return True
            if is_restorer_requested(ctrl):
                return True

            # Exhaustive verification for each active face in memory
            if hasattr(params, "values"):
                for face_params in params.values():
                    if isinstance(face_params, dict):
                        if is_restorer_requested(face_params):
                            return True
                    elif hasattr(face_params, "data") and isinstance(
                        face_params.data, dict
                    ):
                        if is_restorer_requested(face_params.data):
                            return True

            return False  # No face requested this restorer, we can safely unload it

        # 2. GENERAL CASE: MODELS TOGGLE MAP
        toggles = MODELS_TOGGLE_MAP.get(model_name)
        if not toggles:
            return True  # Core models without UI toggles are always assumed needed

        for toggle in toggles:
            target_key = toggle.key

            if hasattr(ctrl, "get") and _is_truthy(ctrl.get(target_key, False)):
                return True
            if hasattr(params, "get") and _is_truthy(params.get(target_key, False)):
                return True
            if hasattr(default_params, "get") and _is_truthy(
                default_params.get(target_key, False)
            ):
                return True

            if hasattr(params, "values"):
                for face_params in params.values():
                    if hasattr(face_params, "get") and _is_truthy(
                        face_params.get(target_key, False)
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

    def _default_providers(self) -> list:
        """ONNX Runtime provider list matching this machine's best provider."""
        match platform_support.default_execution_provider():
            case "TensorRT" | "TensorRT-Engine":
                return [
                    ("TensorrtExecutionProvider", self.trt_ep_options),
                    ("CUDAExecutionProvider", {"device_id": self.gpu_id}),
                    ("CPUExecutionProvider"),
                ]
            case "CUDA":
                return [
                    ("CUDAExecutionProvider", {"device_id": self.gpu_id}),
                    ("CPUExecutionProvider"),
                ]
            case "CoreML":
                return [
                    ("CoreMLExecutionProvider", self.coreml_ep_options),
                    ("CPUExecutionProvider"),
                ]
            case _:
                return ["CPUExecutionProvider"]

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

            case "CoreML":
                if not platform_support.has_coreml():
                    raise RuntimeError(
                        "CoreML execution provider is not available in this "
                        "onnxruntime build."
                    )
                providers = [
                    ("CoreMLExecutionProvider", self.coreml_ep_options),
                    ("CPUExecutionProvider"),
                ]
                self.device, self.device_type = (
                    platform_support.torch_device_for_provider("CoreML", self.gpu_id)
                )

            case "CUDA":
                # A workspace saved on an NVIDIA machine can carry "CUDA" here.
                # Without this check the device would be set to cuda:N on a host
                # that has no CUDA, and the failure would surface later as an
                # opaque crash on the first tensor allocation.
                if not platform_support.has_cuda():
                    raise RuntimeError(
                        "CUDA execution provider requested but no CUDA device is "
                        "available on this machine."
                    )
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
