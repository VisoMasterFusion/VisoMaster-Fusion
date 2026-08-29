import traceback
import threading
import queue
import copy
import contextlib
import time
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, cast

import torch
from skimage import transform as trans
import kornia.geometry.transform as kgm

from torchvision.transforms import v2
import torchvision

import numpy as np

from app.processors.utils import faceutil
from app.processors.utils import platform_support

from app.helpers.miscellaneous import (
    find_best_target_match,
    get_scaling_transforms,
)
from app.processors.workers.frame_worker_vr import VRProcessor
from app.processors.workers.frame_worker_standard import StandardProcessor
from app.processors.workers.frame_worker_pipeline import PipelineProcessor
from app.helpers.typing_helper import ParametersTypes

if TYPE_CHECKING:
    from app.ui.main_ui import MainWindow

torchvision.disable_beta_transforms_warning()


class FrameWorker(threading.Thread):
    """
    Worker thread responsible for processing a single video frame or image.
    Can operate in two modes:
    1. Pool Worker: Persistently runs, fetching tasks from a queue.
    2. Single Frame Worker: Runs once for a specific frame (e.g., UI preview).

    Handles the entire pipeline: Detection -> Swapping -> Enhancement -> Post-processing.
    """

    # FW-QUAL-10: frozenset of GhostFace model names for cleaner membership tests
    GHOSTFACE_MODELS: frozenset[str] = frozenset(
        {"GhostFace-v1", "GhostFace-v2", "GhostFace-v3"}
    )

    # Q-IMP-01: interpolation mode constants for face alignment
    _BICUBIC_INTERP: str = "bicubic"
    _BILINEAR_INTERP: str = "bilinear"

    # Q-QUAL-01: EMA alpha for keypoint smoothing
    _KPS_EMA_ALPHA: float = 0.35

    # Q-IMP-04: minimum face bounding-box side length (pixels) to process
    _MIN_FACE_PIXELS: int = 20
    preview_generation: int

    def __init__(
        self,
        main_window: "MainWindow",
        # Pool worker args (frame_queue is a task queue)
        frame_queue: queue.Queue | None = None,
        worker_id: int = -1,
        worker_stream: torch.cuda.Stream | None = None,
        # Single-frame worker args
        frame: np.ndarray | None = None,
        frame_number: int = -1,
        is_single_frame: bool = False,
    ) -> None:
        """
        Initialises a FrameWorker thread.

        The worker operates in one of two modes:
          - **Pool mode** (frame_queue is not None, worker_id >= 0): runs continuously,
            pulling (frame_number, frame, params, control) tasks from the shared queue.
          - **Single-frame mode** (frame_queue is None): processes one frame supplied at
            construction time and then exits.

        Args:
            main_window:    Application MainWindow — provides access to models, parameters,
                            target faces, and control settings.
            frame_queue:    Shared task queue for pool-mode workers; ``None`` in single-frame mode.
            worker_id:      Integer identifier used to name the thread; ``-1`` in single-frame mode.
            worker_stream:  Explicitly injected CUDA stream for VRAM pooling and task scheduling.
            frame:          Pre-read frame (RGB ndarray) for single-frame mode; ``None`` in pool mode.
            frame_number:   Frame index corresponding to *frame*; ``-1`` in pool mode.
            is_single_frame: ``True`` when this is a one-shot single-frame worker.
        """
        super().__init__()
        # This event will be used to signal the thread to stop
        self.stop_event = threading.Event()

        # Instantiate the decoupled VR Processor
        self.vr_processor = VRProcessor(self)

        # Instantiate the decoupled Standard Processor
        self.standard_processor = StandardProcessor(self)

        # Instantiate the decoupled Pipeline Processor (Swap_Core)
        self.pipeline_processor = PipelineProcessor(self)

        # Scaling transforms (initialized in set_scaling_transforms)
        self.t512: v2.Resize | None = None
        self.t384: v2.Resize | None = None
        self.t256: v2.Resize | None = None
        self.t128: v2.Resize | None = None
        self.interpolation_get_cropped_face_kps: v2.InterpolationMode | None = None
        self.interpolation_original_face_128_384: v2.InterpolationMode | None = None
        self.interpolation_original_face_512: v2.InterpolationMode | None = None
        self.interpolation_Untransform: v2.InterpolationMode | None = None
        self.interpolation_scaleback: v2.InterpolationMode | None = None
        self.t256_face: v2.Resize | None = None
        self.interpolation_expression_faceeditor_back: v2.InterpolationMode | None = (
            None
        )
        self.interpolation_block_shift: v2.InterpolationMode | None = None

        # FW-PERF-5: promote frequently-used inline Resize objects to instance
        # attributes so they are not reconstructed on every swap_core call
        self.t512_mask: v2.Resize | None = None
        self.t128_mask: v2.Resize | None = None
        self.t256_near: v2.Resize | None = None

        # FW-MEM-02: resize-object cache — bounded LRU to avoid accumulating
        # v2.Resize instances for every unique (H, W, interp) combination seen
        # across variable-resolution inputs.
        self._resize_cache: OrderedDict[tuple, v2.Resize] = OrderedDict()
        self._RESIZE_CACHE_MAX: int = 16

        # --- Architecture References ---
        self.main_window = main_window
        self.models_processor = main_window.models_processor
        self.video_processor = main_window.video_processor
        self.function_worker = main_window.function_worker

        # Mode-specific args
        self.frame_queue = frame_queue  # This is now the TASK queue
        self.worker_id = worker_id

        # Single-frame data
        self.frame = frame  # Will be None in pool mode until a task is dequeued
        self.frame_number = (
            frame_number  # Will be -1 in pool mode until a task is dequeued
        )
        self.is_single_frame = is_single_frame
        self.preview_generation = 0

        # Determine mode & Thread Name
        self.is_pool_worker = (frame_queue is not None) and (worker_id != -1)

        if self.is_pool_worker:
            self.name = f"FrameWorker-Pool-{worker_id}"
        else:
            self.name = f"FrameWorker-Single-{frame_number}"

        self.parameters: dict[
            str, ParametersTypes
        ] = {}  # Will be populated from main_window.parameters or task

        self.last_processed_frame_number: int = -1
        self.last_detected_faces: list[dict[str, Any]] = []

        self.is_view_face_compare: bool = False
        self.is_view_face_mask: bool = False
        self.lock = threading.Lock()
        self._tracker_lock = threading.Lock()  # BT-06: guard self.tracker access

        # FW-ROBUST-05: initialize feeder state dict so worker-thread reads never see NameError
        self.local_control_state_from_feeder: dict[str, Any] = {}

        # Dirty-check cache for set_scaling_transforms (FW-PERF-07)
        self._last_scaling_control: dict[str, Any] | None = None

        # Q-QUAL-01: EMA-smoothed keypoints to reduce detection-interval flicker
        self._smoothed_kps: dict[int, np.ndarray] = {}

        # Precalculater bboxes and kps from the video_processor in chronological order
        self.precomputed_bboxes: list[np.ndarray] | None = None
        self.precomputed_kpss_5: list[np.ndarray] | None = None
        self.precomputed_kpss: list[np.ndarray] | None = None
        self.precomputed_kpss_203: list[np.ndarray] | None = None

        # --- ISOLATED CUDA STREAM (VRAM OPTIMIZED) ---
        # The WorkerPoolManager now explicitly passes the shared stream
        self.worker_stream = worker_stream

        if (
            torch.cuda.is_available()
            and self.is_pool_worker
            and self.worker_stream is None
        ):
            # Failsafe fallback just in case the orchestrator failed to pass one
            self.worker_stream = torch.cuda.Stream()

    def set_scaling_transforms(self, control_params: dict[str, Any]) -> None:
        """Initializes the torchvision transforms based on user interpolation settings."""
        (
            self.t512,
            self.t384,
            self.t256,
            self.t128,
            self.interpolation_get_cropped_face_kps,
            self.interpolation_original_face_128_384,
            self.interpolation_original_face_512,
            self.interpolation_Untransform,
            self.interpolation_scaleback,
            self.t256_face,
            self.interpolation_expression_faceeditor_back,
            self.interpolation_block_shift,
        ) = get_scaling_transforms(control_params)

        # Pass relevant transforms to FrameEdits helper via the thread-safe facade
        self.function_worker.set_frame_edits_transforms(
            self.t256_face, self.interpolation_expression_faceeditor_back
        )

        # FW-PERF-5: initialize promoted inline transforms once here
        # Removed antialias=True from mask transforms to prevent ringing artifacts
        # and sub-pixel haloing when multiplied against the swapped face.
        self.t512_mask = v2.Resize(
            (512, 512), interpolation=v2.InterpolationMode.BILINEAR, antialias=False
        )
        self.t128_mask = v2.Resize(
            (128, 128), interpolation=v2.InterpolationMode.BILINEAR, antialias=False
        )
        self.t256_near = v2.Resize(
            (256, 256), interpolation=v2.InterpolationMode.NEAREST, antialias=False
        )

    def swap_core(self, *args, **kwargs):
        """Delegates heavy rendering inference to the PipelineProcessor."""
        return self.pipeline_processor.swap_core(*args, **kwargs)

    def _apply_auto_mouth(self, *args, **kwargs):
        """Delegates mouth occlusion analysis to the PipelineProcessor."""
        return self.pipeline_processor._apply_auto_mouth(*args, **kwargs)

    def get_border_mask(self, *args, **kwargs):
        """Delegates border mask generation to the PipelineProcessor."""
        return self.pipeline_processor.get_border_mask(*args, **kwargs)

    def run(self) -> None:
        """
        Main thread execution loop.
        - In Pool Mode: Loops, gets tasks from self.frame_queue, calls process_and_emit_task().
        - In Single-Frame Mode: Calls process_and_emit_task() just once.
        """
        if self.is_pool_worker:
            # --- Pool Worker Mode ---

            # FW-MYPY-FIX: Assert the queue exists for static typing and runtime safety.
            # If the worker is a pool worker, the orchestrator MUST provide a queue.
            assert self.frame_queue is not None, (
                "Pool worker initialized without a frame_queue."
            )

            # VRAM check throttling state
            vram_check_counter: int = 0
            was_vram_throttled: bool = False

            while not self.stop_event.is_set():
                task = None  # Ensure task is defined for 'finally'
                try:
                    # Block until a task is available or a poison pill is received
                    # Use a timeout to periodically check the stop_event
                    task = self.frame_queue.get(timeout=1.0)

                    if task is None:
                        # Poison pill received: Exit the loop
                        print(f"[INFO] {self.name} received poison pill. Exiting.")
                        break  # 'finally' will call task_done()

                    if self.stop_event.is_set():
                        # Stopped while waiting, discard task
                        break  # 'finally' will call task_done()

                    # Unpack the task which includes frame, specific parameters AND precomputed detections
                    (
                        self.frame_number,
                        self.frame,
                        local_params_from_feeder,
                        local_control_from_feeder,
                        self.precomputed_bboxes,
                        self.precomputed_kpss_5,
                        self.precomputed_kpss,
                        self.precomputed_kpss_203,
                    ) = task

                    # Store them locally in the worker
                    self.parameters = local_params_from_feeder
                    self.local_control_state_from_feeder = local_control_from_feeder

                    # --- Prevent Unified Memory Thrashing (VRAM Paging) ---
                    # Throttled VRAM Probe: Query driver every 8th frame or if recently throttled
                    vram_check_counter += 1
                    should_check_vram: bool = (
                        vram_check_counter % 8 == 0
                    ) or was_vram_throttled

                    if (
                        should_check_vram
                        and torch.cuda.is_available()
                        and torch.cuda.is_initialized()
                    ):
                        device_id = torch.cuda.current_device()
                        free_vram_os, total_vram = torch.cuda.mem_get_info(device_id)

                        # CACHING ALLOCATOR TRAP: OS reports VRAM as "used" even if PyTorch is just
                        # caching it for reuse. We MUST add PyTorch's cached memory back to the true free count.
                        pytorch_cached_free = torch.cuda.memory_reserved(
                            device_id
                        ) - torch.cuda.memory_allocated(device_id)
                        actual_free_vram = free_vram_os + pytorch_cached_free

                        # Reserve 5% of total VRAM or at least 1.2 GB to prevent driver crashes
                        min_required_vram = max(total_vram * 0.05, 1288490188)

                        if actual_free_vram < min_required_vram:
                            was_vram_throttled = True
                            backoff_delay: float = 0.05
                            while actual_free_vram < min_required_vram:
                                if self.stop_event.is_set():
                                    break
                                time.sleep(backoff_delay)
                                # Progressive backoff up to 200ms to reduce polling pressure
                                backoff_delay = min(0.2, backoff_delay * 1.5)

                                free_vram_os, _ = torch.cuda.mem_get_info(device_id)
                                pytorch_cached_free = torch.cuda.memory_reserved(
                                    device_id
                                ) - torch.cuda.memory_allocated(device_id)
                                actual_free_vram = free_vram_os + pytorch_cached_free
                        else:
                            was_vram_throttled = False

                    if not self.stop_event.is_set():
                        # Process the frame
                        self.process_and_emit_task()

                except queue.Empty:
                    # Timeout occurred, just loop again to check stop_event
                    continue
                except Exception as e:
                    # An error happened *during* processing
                    print(
                        f"[ERROR] Error in {self.name} (frame {self.frame_number}): {e}"
                    )
                    traceback.print_exc()

                finally:
                    # This block executes *no matter what* (success, exception, or break)
                    if task is not None and self.frame_queue is not None:
                        try:
                            self.frame_queue.task_done()
                        except ValueError:
                            # Safe to ignore if queue was cleared externally
                            pass
                    # --- FIX RAM LEAK: Clear heavy objects while thread sleeps ---
                    self.frame = None
                    self.precomputed_bboxes = None
                    self.precomputed_kpss_5 = None
                    self.precomputed_kpss = None
                    self.precomputed_kpss_203 = None
                    self.parameters = {}
                    self.local_control_state_from_feeder = {}
                    task = None

            # --- Break Circular References & Clear Caches ---
            # We explicitly wipe the heavy PyTorch VRAM caches when the thread dies to ensure
            # immediate memory release back to the OS.
            if hasattr(self, "pipeline_processor"):
                if hasattr(self.pipeline_processor, "_drift_cache"):
                    self.pipeline_processor._drift_cache.clear()
                if hasattr(self.pipeline_processor, "_gabor_kernels_cache"):
                    self.pipeline_processor._gabor_kernels_cache.clear()
                if hasattr(self.pipeline_processor, "_gabor_kernels_expanded_cache"):
                    self.pipeline_processor._gabor_kernels_expanded_cache.clear()
                if hasattr(self.pipeline_processor, "_color_stats_ema"):
                    self.pipeline_processor._color_stats_ema.clear()
                self.pipeline_processor._kernel_lap = None
                self.pipeline_processor._kernel_sobel_x = None
                self.pipeline_processor._kernel_sobel_y = None

        else:
            # --- Single-Frame Mode ---
            if self.stop_event.is_set():
                print(f"[WARN] {self.name} cancelled before start.")
                return
            try:
                # Single-Frame worker MUST use the *current* global state
                # to reflect immediate UI changes.
                with self.main_window.models_processor.model_lock:
                    local_parameters_copy = copy.deepcopy(self.main_window.parameters)
                    local_control_copy = copy.deepcopy(self.main_window.control)

                # Ensure parameter dicts exist (failsafe for new faces)
                active_target_face_ids = list(self.main_window.target_faces.keys())
                for face_id_key in active_target_face_ids:
                    if str(face_id_key) not in local_parameters_copy:
                        local_parameters_copy[str(face_id_key)] = (
                            self.main_window.default_parameters.copy()
                        )

                # Store locally
                self.parameters = local_parameters_copy
                self.local_control_state_from_feeder = local_control_copy

                # Run once
                self.process_and_emit_task()
            except Exception as e:
                print(f"[ERROR] Error in {self.name}: {e}")
                traceback.print_exc()

    def process_and_emit_task(self) -> None:
        """
        Processes self.frame using the configured parameters and emits the result signal.
        Does NOT interact with the task queue.
        """
        if self.frame is None:
            print(f"[WARN] {self.name} received an empty frame. Skipping.")
            return

        # Snapshot original RGB frame BEFORE the try block so the except handler
        # always has a valid fallback regardless of where the exception is thrown.
        _fallback_frame_rgb = self.frame
        try:
            # Setup the dedicated asynchronous context for HPC parallel processing
            stream_context = (
                torch.cuda.stream(self.worker_stream)
                if self.worker_stream
                else contextlib.nullcontext()
            )

            with stream_context:
                local_control_state = self.local_control_state_from_feeder

                # FW-RACE-04: use local variables instead of instance-level assignments
                # to prevent cross-frame data races in the pool worker.
                is_view_face_compare = self.main_window.faceCompareCheckBox.isChecked()
                is_view_face_mask = self.main_window.faceMaskCheckBox.isChecked()
                # Keep instance attrs consistent so downstream helpers that still reference
                # them (e.g. swap_core) see the same values.
                self.is_view_face_compare = is_view_face_compare
                self.is_view_face_mask = is_view_face_mask

                # FW-RACE-02: snapshot Qt button states into the feeder dict so worker
                # threads never call .isChecked() concurrently on the same Qt object.
                local_control_state["swap_enabled"] = (
                    self.main_window.swapfacesButton.isChecked()
                )
                local_control_state["edit_enabled"] = (
                    self.main_window.editFacesButton.isChecked()
                )

                # Determine if processing is needed
                needs_processing = (
                    local_control_state.get("swap_enabled", True)
                    or local_control_state.get("edit_enabled", True)
                    or local_control_state.get("FrameEnhancerEnableToggle", False)
                    or local_control_state.get(
                        "ModeEnableToggle", False
                    )  # Always processes in this mode
                    or is_view_face_compare
                    or is_view_face_mask
                )

                if needs_processing:
                    # Ensure input frame is C-contiguous for PyTorch/OpenCV compatibility
                    if not self.frame.flags["C_CONTIGUOUS"]:
                        self.frame = np.ascontiguousarray(self.frame)

                    # Process Frame (returns BGR, uint8)
                    processed_frame_bgr_np_uint8 = self.process_frame(
                        local_control_state, self.stop_event
                    )

                    # Ensure output is C-contiguous for Qt display
                    self.frame = np.ascontiguousarray(processed_frame_bgr_np_uint8)
                else:
                    # If no processing, just convert RGB to BGR for display
                    self.frame = self.frame[..., ::-1]
                    self.frame = np.ascontiguousarray(self.frame)

                # Sync thread before returning to cpu.
                # Non-spinning wait: this fires once per frame, so the event
                # wake-up cost is noise against the frame itself.
                if self.worker_stream:
                    platform_support.blocking_stream_sync(self.worker_stream)

            # Check stop event again
            if self.stop_event.is_set():
                print(f"[WARN] {self.name} cancelled during process_frame.")
                return

            # Emit Signals directly with numpy.ndarray (JIT QPixmap creation is now handled by the GUI Thread)
            if self.video_processor.file_type == "webcam" and not self.is_single_frame:
                self.video_processor.webcam_frame_processed_signal.emit(self.frame)
            elif not self.is_single_frame:
                self.video_processor.frame_processed_signal.emit(
                    self.frame_number, self.frame
                )
            else:  # Single frame processing (image or paused video)
                self.video_processor.single_frame_processed_signal.emit(
                    getattr(self, "preview_generation", 0),
                    self.frame_number,
                    self.frame,
                    None,
                )

        except Exception as e:
            # FW-CUDA-LOG: Distinguish CUDA / TensorRT errors so they are loud and
            # actionable, since they typically mean the entire CUDA context is
            # corrupted and continuing to process will produce more cryptic crashes
            # downstream. The visible message points users at the launcher's
            # Update / Maintenance page where they can switch the provider.
            _err_msg = str(e)
            _is_cuda_error = (
                "CUDA" in _err_msg
                or "cuda" in _err_msg.lower()
                or "TensorRT" in _err_msg
                or "tensorrt" in _err_msg.lower()
                or e.__class__.__name__ == "AcceleratorError"
            )
            if _is_cuda_error:
                print(
                    f"[FATAL] CUDA error in {self.name} (frame {self.frame_number}): {e}"
                )
                print(
                    "[FATAL] CUDA context may be corrupted. If this persists, open "
                    "Launcher → Update / Maintenance → switch provider (TensorRT ↔ CUDA) "
                    "or rebuild the TensorRT engines."
                )
                try:
                    self.video_processor.fatal_processing_error_signal.emit(
                        f"{self.name} frame {self.frame_number}: {e}"
                    )
                except Exception:
                    pass
            else:
                print(f"[ERROR] Error in {self.name} (frame {self.frame_number}): {e}")
            traceback.print_exc()
            # Emit the original (unprocessed) frame as a fallback so the recording
            # metronome is never blocked waiting for a frame that will never arrive.
            # Only do this for pool/video mode AND webcam mode — single-frame has its
            # own error recovery path.
            if (
                not self.stop_event.is_set()
                and not self.is_single_frame
                and isinstance(_fallback_frame_rgb, np.ndarray)
            ):
                try:
                    fallback_bgr = np.ascontiguousarray(_fallback_frame_rgb[..., ::-1])
                    if self.video_processor.file_type == "webcam":
                        self.video_processor.webcam_frame_processed_signal.emit(
                            fallback_bgr
                        )
                    else:
                        self.video_processor.frame_processed_signal.emit(
                            self.frame_number, fallback_bgr
                        )
                except Exception as fb_err:
                    print(
                        f"[WARN] Fallback emit also failed for frame "
                        f"{self.frame_number}: {fb_err}"
                    )

    def _find_best_target_match(
        self,
        detected_embedding_np,
        control_global,
        target_faces_snapshot: dict | None = None,
        is_sequentially_tracked: bool = False,
    ):
        """Finds the best matching source face for a detected target face.

        Args:
            detected_embedding_np: ArcFace embedding of the detected face.
            control_global: Global control dict.
            target_faces_snapshot: Optional pre-snapshotted copy of target_faces dict
                taken under self.lock.  If None, falls back to reading the live dict
                (single-frame / legacy callers).
        """
        # FW-RACE-01: use the snapshot when available; otherwise fall back to live dict.
        if target_faces_snapshot is not None:
            faces_to_iterate = dict(target_faces_snapshot)
        else:
            with self.lock:
                faces_to_iterate = dict(self.main_window.target_faces)

        # FW-RACE-05: snapshot default_parameters under lock once
        with self.lock:
            default_params_dict = dict(self.main_window.default_parameters.data)

        # FW-ARCH-FIX: Trust the Sequential Detector
        # If the face was already rescued temporally, we bypass the UI threshold
        current_params = copy.deepcopy(self.parameters)
        if is_sequentially_tracked:
            for face_id, p in current_params.items():
                if isinstance(p, dict):
                    p["SimilarityThresholdSlider"] = 0.0
            if "SimilarityThresholdSlider" in default_params_dict:
                default_params_dict["SimilarityThresholdSlider"] = 0.0

        return find_best_target_match(
            detected_embedding_np,
            self.function_worker,
            faces_to_iterate,
            current_params,
            cast(dict, default_params_dict),
            str(control_global["RecognitionModelSelection"]),
        )

    def process_frame(self, control: dict, stop_event: threading.Event):
        """
        Routing method:
        - Checks inputs.
        - Sets up transforms.
        - Dispatches to either VR180 or Standard processing logic.
        - Applies common global enhancers.
        """
        # Check 1: At the very beginning
        if stop_event.is_set():
            assert self.frame is not None
            return self.frame[..., ::-1]  # Return original BGR frame

        # Keep last frame number for reference (this is the single-frame path read; pool
        # workers update last_processed_frame_number under lock in _process_frame_standard)
        self.last_processed_frame_number = self.frame_number

        # FW-PERF-07: dirty-check — only rebuild scaling transforms when interpolation
        # settings actually change (the underlying get_scaling_transforms has its own
        # module-level cache but calling set_scaling_transforms also rebuilds the mask
        # transform objects, so we guard it too).
        _scaling_keys = {
            k: control.get(k)
            for k in (
                "get_cropped_face_kpsTypeSelection",
                "original_face_128_384TypeSelection",
                "original_face_512TypeSelection",
                "UntransformTypeSelection",
                "ScalebackFrameTypeSelection",
                "expression_faceeditor_t256TypeSelection",
                "expression_faceeditor_backTypeSelection",
                "block_shiftTypeSelection",
                "AntialiasTypeSelection",
            )
        }
        if self._last_scaling_control != _scaling_keys:
            self.set_scaling_transforms(control)
            self._last_scaling_control = _scaling_keys
        img_numpy_rgb_uint8 = self.frame
        assert img_numpy_rgb_uint8 is not None, "frame must be set before processing"

        # Prepare the base tensor
        img_chw = np.ascontiguousarray(img_numpy_rgb_uint8.transpose(2, 0, 1))

        processed_tensor_rgb_uint8 = torch.from_numpy(img_chw).to(
            self.models_processor.device
        )

        # --- ROUTING LOGIC ---
        if control.get("VR180ModeEnableToggle", False):
            # Delegate entirely to the VR Worker
            processed_tensor_rgb_uint8 = self.vr_processor.process_frame_vr180(
                processed_tensor_rgb_uint8, img_numpy_rgb_uint8, control, stop_event
            )
        else:
            # Delegate entirely to the Standard Worker
            processed_tensor_rgb_uint8 = self.standard_processor.process_standard_frame(
                processed_tensor_rgb_uint8, control, stop_event
            )

        # --- Common Post-Processing (Enhancers, etc.) ---
        compare_mode_active = self.is_view_face_mask or self.is_view_face_compare

        if control.get("FrameEnhancerEnableToggle", False) and not compare_mode_active:
            # Check 5: Before final heavy operation
            if stop_event.is_set():
                assert img_numpy_rgb_uint8 is not None
                return img_numpy_rgb_uint8[..., ::-1]

            processed_tensor_rgb_uint8 = self.function_worker.enhance_core(
                processed_tensor_rgb_uint8, control=control
            )

        final_img_np_rgb_uint8 = (
            processed_tensor_rgb_uint8.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        )
        if not final_img_np_rgb_uint8.flags["C_CONTIGUOUS"]:
            final_img_np_rgb_uint8 = np.ascontiguousarray(final_img_np_rgb_uint8)

        return final_img_np_rgb_uint8[..., ::-1]

    def get_face_similarity_tform(
        self, swapper_model: str, kps_5: np.ndarray
    ) -> trans.SimilarityTransform:
        """
        Computes the similarity transform that maps the detected 5-point keypoints
        to the canonical face template for the given *swapper_model*.

        Different swapper architectures use different alignment templates
        (ArcFace 128 crop, pose-aware ArcFace map crop, or FFHQ-aligned crop).

        Args:
            swapper_model: The name of the active swapper (e.g. ``"Inswapper128"``).
            kps_5:         Detected 5-point keypoints, shape [5, 2].

        Returns:
            Fitted ``skimage.transform.SimilarityTransform``.

        Raises:
            ValueError: If the transform estimation fails (degenerate face geometry).
        """
        # FW-QUAL-10: use GHOSTFACE_MODELS frozenset instead of chained != comparisons
        uses_pose_aware_template = (
            swapper_model in self.GHOSTFACE_MODELS or swapper_model == "AlphaFace"
        )
        if not uses_pose_aware_template and swapper_model != "CSCS":
            dst = faceutil.get_arcface_template(image_size=512, mode="arcface128")
            dst = np.squeeze(dst)
            # Use instance initialization + .estimate() for older skimage versions
            if hasattr(trans.SimilarityTransform, "from_estimate"):
                tform = trans.SimilarityTransform.from_estimate(kps_5, dst)
                if np.any(np.isnan(tform.params)) or np.any(np.isinf(tform.params)):
                    raise ValueError(
                        "Similarity transform estimation produced NaN/Inf (degenerate face geometry)"
                    )
            else:
                tform = trans.SimilarityTransform()
                # FW-ROBUST-11: check return value of tform.estimate()
                if not tform.estimate(kps_5, dst):
                    raise ValueError("Similarity transform estimation failed for face")
        elif swapper_model == "CSCS":
            # Use instance initialization + .estimate() for older skimage versions
            if hasattr(trans.SimilarityTransform, "from_estimate"):
                tform = trans.SimilarityTransform.from_estimate(
                    kps_5, self.models_processor.FFHQ_kps
                )
                if np.any(np.isnan(tform.params)) or np.any(np.isinf(tform.params)):
                    raise ValueError(
                        "Similarity transform estimation produced NaN/Inf (degenerate face geometry)"
                    )
            else:
                tform = trans.SimilarityTransform()
                # FW-ROBUST-11: check return value
                if not tform.estimate(kps_5, self.models_processor.FFHQ_kps):
                    raise ValueError(
                        "Similarity transform estimation failed for CSCS face"
                    )
        else:
            # FW-QUAL-10: swapper_model in GHOSTFACE_MODELS, or AlphaFace
            tform = trans.SimilarityTransform()
            dst = faceutil.get_arcface_template(image_size=512, mode="arcfacemap")
            if swapper_model == "AlphaFace":
                # AlphaFace only matches the five yaw templates. The two trailing
                # pitch templates can select a noticeably different crop scale for
                # near-profile faces, which pops between frames.
                dst = dst[:5]
                # Scale the 112-based templates to match the 128-based Inswapper crop.
                # This zooms the crop out.
                dst = dst * (112.0 / 128.0)
                # Re-center the face horizontally by shifting X coordinates
                dst[:, :, 0] += (512.0 / 128.0) * 8.0

            M, _ = faceutil.estimate_norm_arcface_template(kps_5, src=dst)
            if M is None or np.any(np.isnan(M)) or np.any(np.isinf(M)):
                raise ValueError(
                    f"{swapper_model} transform estimation failed "
                    "(degenerate face geometry)"
                )
            tform.params[0:2] = M
        return tform

    @torch.no_grad()
    def get_transformed_and_scaled_faces(
        self, tform, img, interp_mode: str = "bilinear", only_256: bool = False
    ):
        """
        Applies the similarity transform to extract aligned face crops at four resolutions.
        OPTIMIZED: GPU warping directly from transformation matrix using Kornia.

        Args:
            tform:       Fitted ``SimilarityTransform`` from ``get_face_similarity_tform``.
            img:         Full-frame CHW uint8 tensor.
            interp_mode: Interpolation mode for warp_affine (e.g. "bilinear" or "bicubic").
            only_256:    Skip the 384px and 128px resizes and alias those slots onto
                         the 512px/256px crops. Inswapper128 is the only swapper that
                         reads those two crops, so this is safe for any other model
                         (currently used by AlphaFace).

        Returns:
            Tuple ``(face_512, face_384, face_256, face_128)``, all CHW uint8 tensors.
        """
        M_tensor = (
            torch.from_numpy(cast(np.ndarray, tform.params)[0:2])
            .float()
            .unsqueeze(0)
            .to(img.device)
        )

        # Cast to float32 for Kornia's grid_sample compatibility on CUDA
        img_b = img.unsqueeze(0) if img.dim() == 3 else img
        img_b_float = img_b.float()

        original_face_512 = kgm.warp_affine(
            img_b_float,
            M_tensor,
            dsize=(512, 512),
            mode=interp_mode,
            align_corners=True,
        ).squeeze(0)

        # Convert back to original dtype (uint8) before passing to torchvision resizers
        original_face_512 = original_face_512.to(img.dtype)

        assert self.t256 is not None, (
            "t256 must be initialized via set_scaling_transforms"
        )
        original_face_256 = self.t256(original_face_512)
        if only_256:
            return (
                original_face_512,
                original_face_512,
                original_face_256,
                original_face_256,
            )

        assert self.t384 is not None, (
            "t384 must be initialized via set_scaling_transforms"
        )
        assert self.t128 is not None, (
            "t128 must be initialized via set_scaling_transforms"
        )
        original_face_384 = self.t384(original_face_512)
        original_face_128 = self.t128(original_face_256)
        return (
            original_face_512,
            original_face_384,
            original_face_256,
            original_face_128,
        )

    @staticmethod
    def _is_kps_valid(kps: np.ndarray, img_h: int, img_w: int) -> bool:
        """Returns False if any keypoint is NaN, Inf, or outside image bounds."""
        if kps is None or kps.size == 0:
            return False
        if np.any(np.isnan(kps)) or np.any(np.isinf(kps)):
            return False

        """
        # WE DO NOT CHECK image bounds here anymore.
        if np.any(kps[:, 0] < 0) or np.any(kps[:, 0] >= img_w):
            return False
        if np.any(kps[:, 1] < 0) or np.any(kps[:, 1] >= img_h):
            return False
        """
        return True
