import traceback
from typing import TYPE_CHECKING, Any, Dict, Optional, cast
import threading
import queue
import copy
import math
from math import ceil
from app.ui.widgets import widget_components
import torch
from skimage import transform as trans
import kornia.enhance as ke
import kornia.color as kc
import kornia.geometry.transform as kgm

from torchvision.transforms import v2
import torchvision

import numpy as np
import torch.nn.functional as F

from app.processors.utils import faceutil

from app.helpers.miscellaneous import (
    ParametersDict,
    find_best_target_match,
    get_scaling_transforms,
    draw_bounding_boxes_on_detected_faces,
    paint_landmarks_on_image,
    is_detected_face_eligible_for_matching,
    keypoints_adjustments,
    get_grid_for_pasting,
)
from app.helpers.vr_utils import EquirectangularConverter, PerspectiveConverter
from app.helpers.typing_helper import ParametersTypes
from app.processors.frame_enhancers import FrameEnhancers
from app.processors.frame_edits import FrameEdits

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
    GHOSTFACE_MODELS = frozenset({"GhostFace-v1", "GhostFace-v2", "GhostFace-v3"})

    # Q-IMP-01: interpolation mode constants for face alignment
    _BICUBIC_INTERP = "bicubic"
    _BILINEAR_INTERP = "bilinear"

    # Q-QUAL-01: EMA alpha for keypoint smoothing
    _KPS_EMA_ALPHA: float = 0.35

    # Q-QUAL-03: EMA alpha for AutoColor reference statistics
    _COLOR_EMA_ALPHA: float = 0.30

    # P3-06: reusable GaussianBlur for _fix_drift_and_texture (stateless, device-agnostic)
    _DRIFT_TEXTURE_BLUR = v2.GaussianBlur(kernel_size=5, sigma=1.0)

    # Q-IMP-04: minimum face bounding-box side length (pixels) to process
    _MIN_FACE_PIXELS: int = 20

    def __init__(
        self,
        main_window: "MainWindow",
        # Pool worker args (frame_queue is a task queue)
        frame_queue: queue.Queue | None = None,
        worker_id: int = -1,
        # Single-frame worker args
        frame: np.ndarray | None = None,
        frame_number: int = -1,
        is_single_frame: bool = False,
    ):
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
            frame:          Pre-read frame (RGB ndarray) for single-frame mode; ``None`` in pool mode.
            frame_number:   Frame index corresponding to *frame*; ``-1`` in pool mode.
            is_single_frame: ``True`` when this is a one-shot single-frame worker.
        """
        super().__init__()
        # This event will be used to signal the thread to stop
        self.stop_event = threading.Event()

        # Scaling transforms (initialized in set_scaling_transforms)
        self.t512: Optional[v2.Resize] = None
        self.t384: Optional[v2.Resize] = None
        self.t256: Optional[v2.Resize] = None
        self.t128: Optional[v2.Resize] = None
        self.interpolation_get_cropped_face_kps: Optional[v2.InterpolationMode] = None
        self.interpolation_original_face_128_384: Optional[v2.InterpolationMode] = None
        self.interpolation_original_face_512: Optional[v2.InterpolationMode] = None
        self.interpolation_Untransform: Optional[v2.InterpolationMode] = None
        self.interpolation_scaleback: Optional[v2.InterpolationMode] = None
        self.t256_face: Optional[v2.Resize] = None
        self.interpolation_expression_faceeditor_back: Optional[
            v2.InterpolationMode
        ] = None
        self.interpolation_block_shift: Optional[v2.InterpolationMode] = None
        # FW-PERF-5: promote frequently-used inline Resize objects to instance
        # attributes so they are not reconstructed on every swap_core call
        self.t512_mask: Optional[v2.Resize] = None
        self.t128_mask: Optional[v2.Resize] = None
        self.t256_near: Optional[v2.Resize] = None
        # FW-MEM-01: Gabor kernel cache as LRU-bounded OrderedDict
        from collections import OrderedDict as _OrderedDict

        self._gabor_kernels_cache: _OrderedDict = (
            _OrderedDict()
        )  # keyed by (kernel_size,sigma,lambd,gamma,psi,N,device_str)
        # FW-MEM-02: resize-object cache — bounded LRU to avoid accumulating
        # v2.Resize instances for every unique (H, W, interp) combination seen
        # across variable-resolution inputs.
        self._resize_cache: _OrderedDict = _OrderedDict()
        self._RESIZE_CACHE_MAX = 16

        # --- Architecture References ---
        self.main_window = main_window
        self.models_processor = main_window.models_processor
        self.video_processor = main_window.video_processor

        # Initialize Helpers
        self.frame_enhancers = FrameEnhancers(self.models_processor)
        self.frame_edits = FrameEdits(self.models_processor)

        # Mode-specific args
        self.frame_queue = frame_queue  # This is now the TASK queue
        self.worker_id = worker_id

        # Single-frame data
        self.frame = frame  # Will be None in pool mode until a task is dequeued
        self.frame_number = (
            frame_number  # Will be -1 in pool mode until a task is dequeued
        )
        self.is_single_frame = is_single_frame

        # Determine mode & Thread Name
        self.is_pool_worker = (frame_queue is not None) and (worker_id != -1)

        if self.is_pool_worker:
            self.name = f"FrameWorker-Pool-{worker_id}"
        else:
            self.name = f"FrameWorker-Single-{frame_number}"

        self.parameters: Dict[
            str, ParametersTypes
        ] = {}  # Will be populated from main_window.parameters or task

        self.last_processed_frame_number = -1
        self.last_detected_faces: list = []

        # VR-specific tracking state (kept separate from standard-mode state so
        # switching modes does not corrupt either path's detection interval logic)
        self.last_detected_faces_vr: list = []
        self.last_processed_frame_number_vr: int = -1

        # VR specific constants
        self.VR_PERSPECTIVE_RENDER_SIZE = 512  # Pixels, for rendering perspective crops
        # VR-13: renamed from VR_DYNAMIC_FOV_PADDING_FACTOR and set to 1.5
        self.VR_FOV_SCALE_FACTOR = 1.5  # Scale factor for dynamic FOV calculation
        self.is_view_face_compare: bool = False
        self.is_view_face_mask: bool = False
        self.lock = threading.Lock()
        self._tracker_lock = threading.Lock()  # BT-06: guard self.tracker access

        # FW-ROBUST-05: initialize feeder state dict so worker-thread reads never see NameError
        self.local_control_state_from_feeder: dict = {}

        # VR converter cache (VR-08)
        self._vr_converter: Optional[EquirectangularConverter] = None
        self._vr_frame_size: Optional[tuple] = None
        # Improvement K: cache PerspectiveConverter across frames (same lifetime as E2P converter)
        self._vr_p2e_converter: Optional[PerspectiveConverter] = None
        # Tracked separately from _vr_frame_size: _vr_frame_size is updated by the E2P branch
        # before the P2E check executes, making them always equal and preventing P2E recreation
        # on resolution change.  This dedicated attribute fixes that race.
        self._vr_p2e_frame_size: Optional[tuple] = None
        # VR-MEM-03: per-worker frame counter for periodic CUDA allocator flush
        self._vr_processed_count: int = 0

        # Dirty-check cache for set_scaling_transforms (FW-PERF-07)
        self._last_scaling_control: dict | None = None
        # Separate dirty-check for VR path — keys differ from standard path
        # so they must not share the same variable or they constantly invalidate each other.
        self._last_vr_scaling_control: dict | None = None

        # Q-QUAL-01: EMA-smoothed keypoints to reduce detection-interval flicker
        self._smoothed_kps: dict[int, np.ndarray] = {}

        # Q-QUAL-03: EMA over per-face AutoColor reference statistics to reduce flicker.
        # Bounded LRU: keyed by embedding bytes (one entry per unique target face seen).
        # Typical usage: 1–10 entries.  Cap at 32 to handle large session edge cases.
        self._color_stats_ema: _OrderedDict = _OrderedDict()
        self._COLOR_STATS_EMA_MAX = 32

        # Mouth action detection score (set per-face call in _detect_mouth_action_score)
        self._mouth_action_score: float = 0.0

        # Precalculater bboxes and kps from the video_processor in chronological order
        self.precomputed_bboxes: Optional[list] = None
        self.precomputed_kpss_5: Optional[list] = None
        self.precomputed_kpss: Optional[list] = None

        # --- OPTIMIZATION: Cached Convolution Kernels (VRAM) ---
        # Pre-allocating mathematical filters prevents massive CPU-to-GPU
        # allocation overheads during the binary search loops (sharpness_score).
        device = self.models_processor.device
        self.kernel_lap = torch.tensor(
            [[0, 1, 0], [1, -4, 1], [0, 1, 0]], device=device, dtype=torch.float32
        ).view(1, 1, 3, 3)

        self.kernel_sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=device, dtype=torch.float32
        ).view(1, 1, 3, 3)
        self.kernel_sobel_y = self.kernel_sobel_x.transpose(2, 3)

        # Thread-safety. Create a frame_worker stream as cpu uses multithreaded streams but GPU makes a queue.
        self.worker_stream = None  # torch.cuda.Stream() if self.models_processor.device == "cuda" else None

    def set_scaling_transforms(self, control_params):
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

        # Pass relevant transforms to FrameEdits helper
        self.frame_edits.set_transforms(
            self.t256_face, self.interpolation_expression_faceeditor_back
        )

        # FW-PERF-5: initialize promoted inline transforms once here
        self.t512_mask = v2.Resize(
            (512, 512), interpolation=v2.InterpolationMode.BILINEAR, antialias=True
        )
        self.t128_mask = v2.Resize(
            (128, 128), interpolation=v2.InterpolationMode.BILINEAR, antialias=True
        )
        self.t256_near = v2.Resize(
            (256, 256), interpolation=v2.InterpolationMode.NEAREST, antialias=False
        )

    def run(self):
        """
        Main thread execution loop.
        - In Pool Mode: Loops, gets tasks from self.frame_queue, calls process_and_emit_task().
        - In Single-Frame Mode: Calls process_and_emit_task() just once.
        """
        if self.is_pool_worker:
            # --- Pool Worker Mode ---
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
                    ) = task

                    # Store them locally in the worker
                    self.parameters = local_params_from_feeder
                    self.local_control_state_from_feeder = local_control_from_feeder

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

    def process_and_emit_task(self):
        """
        Processes self.frame using the configured parameters and emits the result signal.
        Does NOT interact with the task queue.
        """
        # Snapshot original RGB frame BEFORE the try block so the except handler
        # always has a valid fallback regardless of where the exception is thrown.
        _fallback_frame_rgb = self.frame
        try:
            import contextlib

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

                # Sync thread before returning to cpu
                if self.worker_stream:
                    self.worker_stream.synchronize()

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
                    self.frame_number, self.frame
                )

        except Exception as e:
            print(f"[ERROR] Error in {self.name} (frame {self.frame_number}): {e}")
            traceback.print_exc()
            # Emit the original (unprocessed) frame as a fallback so the recording
            # metronome is never blocked waiting for a frame that will never arrive.
            # Only do this for pool/video mode — single-frame and webcam have their
            # own error recovery paths.
            if (
                not self.stop_event.is_set()
                and not self.is_single_frame
                and self.video_processor.file_type != "webcam"
                and isinstance(_fallback_frame_rgb, np.ndarray)
            ):
                try:
                    fallback_bgr = np.ascontiguousarray(_fallback_frame_rgb[..., ::-1])
                    self.video_processor.frame_processed_signal.emit(
                        self.frame_number, fallback_bgr
                    )
                except Exception as fb_err:
                    print(
                        f"[WARN] Fallback emit also failed for frame "
                        f"{self.frame_number}: {fb_err}"
                    )

    def _apply_denoiser_pass(
        self,
        image_tensor_cxhxw_uint8: torch.Tensor,
        control: dict,
        pass_suffix: str,
        kv_map: Dict | None,
    ) -> torch.Tensor:
        """Helper to run the diffusion-based denoiser (Ref-LDM).

        FW-QUAL-13: pass_suffix convention:
          - "Before"     → DenoiserUNetEnableBeforeRestorersToggle (before Restoration 1)
          - "AfterFirst" → DenoiserAfterFirstRestorerToggle (between Restoration 1 and 2)
          - "After"      → DenoiserAfterRestorersToggle (after Restoration 2)
        """
        use_exclusive_path = control.get("UseReferenceExclusivePathToggle", False)
        denoiser_seed_from_slider_val = int(control.get("DenoiserBaseSeedSlider", 1))

        denoiser_mode_key = f"DenoiserModeSelection{pass_suffix}"
        denoiser_mode_val = control.get(denoiser_mode_key, "Single Step (Fast)")

        ddim_steps_key = f"DenoiserDDIMStepsSlider{pass_suffix}"
        ddim_steps_val = int(control.get(ddim_steps_key, 20))

        cfg_scale_key = f"DenoiserCFGScaleDecimalSlider{pass_suffix}"
        cfg_scale_val = float(control.get(cfg_scale_key, 1.0))

        single_step_t_key = f"DenoiserSingleStepTimestepSlider{pass_suffix}"
        single_step_t_val = int(control.get(single_step_t_key, 1))

        sharpen_key = f"DenoiserLatentSharpeningDecimalSlider{pass_suffix}"
        sharpen_val = float(control.get(sharpen_key, 0.0))

        if not kv_map:
            if use_exclusive_path:
                if control.get("CommandLineDebugEnableToggle", False):
                    print(
                        f"[ERROR] Denoiser {pass_suffix}: No source face for K/V, but 'Exclusive Reference Path' is ON. Skipping."
                    )
                return image_tensor_cxhxw_uint8

        denoised_image = self.models_processor.apply_denoiser_unet(
            image_tensor_cxhxw_uint8,
            reference_kv_map=kv_map,
            use_reference_exclusive_path=use_exclusive_path,
            denoiser_mode=denoiser_mode_val,
            base_seed=denoiser_seed_from_slider_val,
            denoiser_single_step_t=single_step_t_val,
            denoiser_ddim_steps=ddim_steps_val,
            denoiser_cfg_scale=cfg_scale_val,
            latent_sharpening_strength=sharpen_val,
        )
        return torch.clamp(denoised_image, 0, 255)

    def _find_best_target_match(
        self,
        detected_embedding_np,
        control_global,
        target_faces_snapshot: dict | None = None,
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

        return find_best_target_match(
            detected_embedding_np,
            self.models_processor,
            faces_to_iterate,
            self.parameters,
            cast(dict, default_params_dict),
            str(control_global["RecognitionModelSelection"]),
        )

    def _process_single_vr_perspective_crop_multi(
        self,
        perspective_crop_torch_rgb_uint8: torch.Tensor,
        target_face_button: "widget_components.TargetFaceCardButton",
        parameters_for_face: ParametersDict,
        control_global: dict,
        kps_5_on_crop_param: np.ndarray,
        kps_all_on_crop_param: np.ndarray | None,
        swap_button_is_checked_global: bool,
        edit_button_is_checked_global: bool,
        eye_side_for_debug: str = "",
        kv_map_for_swap: Dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Processes a single perspective crop extracted from a VR frame.

        Returns:
            (processed_crop, swap_mask_1x512x512_float_or_None)
            The mask is only populated when self.is_view_face_mask is True.
        """
        # VR-12: assert crop is square with the configured render size so downstream
        # swap_core assumptions hold.  Improvement F allows 512/768/1024 via UI.
        _expected_crop_size = self.VR_PERSPECTIVE_RENDER_SIZE
        assert perspective_crop_torch_rgb_uint8.shape[-2:] == (
            _expected_crop_size,
            _expected_crop_size,
        ), (
            f"VR perspective crop must be {_expected_crop_size}×{_expected_crop_size}, "
            f"got {perspective_crop_torch_rgb_uint8.shape}"
        )
        processed_crop_torch_rgb_uint8 = perspective_crop_torch_rgb_uint8.clone()
        if kps_5_on_crop_param is None or kps_5_on_crop_param.size == 0:
            return processed_crop_torch_rgb_uint8, None

        if not (swap_button_is_checked_global or edit_button_is_checked_global):
            return processed_crop_torch_rgb_uint8, None

        arcface_model_for_swap = self.models_processor.get_arcface_model(
            parameters_for_face["SwapModelSelection"]
        )
        s_e_for_swap_np = None
        if swap_button_is_checked_global:
            _vr_reaging_on = parameters_for_face.get("FaceReagingEnableToggle", False)
            _vr_aged_emb = getattr(target_face_button, "aged_input_embedding", {})
            if _vr_reaging_on and _vr_aged_emb:
                s_e_for_swap_np = _vr_aged_emb.get(arcface_model_for_swap)
            else:
                s_e_for_swap_np = target_face_button.assigned_input_embedding.get(
                    arcface_model_for_swap
                )
            if (
                s_e_for_swap_np is None
                or not isinstance(s_e_for_swap_np, np.ndarray)
                or s_e_for_swap_np.size == 0
                or np.isnan(s_e_for_swap_np).any()
                or np.isinf(s_e_for_swap_np).any()
            ):
                s_e_for_swap_np = None

        t_e_for_swap_np = target_face_button.get_embedding(arcface_model_for_swap)
        dfm_model_instance_local = None
        if parameters_for_face["SwapModelSelection"] == "DeepFaceLive (DFM)":
            dfm_model_name = parameters_for_face["DFMModelSelection"]
            if dfm_model_name:
                dfm_model_instance_local = self.models_processor.load_dfm_model(
                    dfm_model_name
                )

        s_e_for_swap_core = s_e_for_swap_np if swap_button_is_checked_global else None
        vr_swap_mask_for_compare: torch.Tensor | None = None

        if (
            swap_button_is_checked_global
            and (
                s_e_for_swap_core is not None
                or (
                    parameters_for_face["SwapModelSelection"] == "DeepFaceLive (DFM)"
                    and dfm_model_instance_local is not None
                )
            )
        ) or edit_button_is_checked_global:
            source_kps = None
            if target_face_button and target_face_button.assigned_input_faces:
                first_input_id = list(target_face_button.assigned_input_faces.keys())[0]
                store = target_face_button.assigned_input_faces[first_input_id]
                if "kps_5" in store:
                    source_kps = store["kps_5"]

            kps_5_on_crop_param = keypoints_adjustments(
                kps_5_on_crop_param,
                cast(dict, parameters_for_face),
                source_kps=source_kps,  # type: ignore[arg-type]
            )
            _params_for_swap_vr = self._apply_auto_mouth(
                parameters_for_face.data,  # plain dict — VR convention
                target_face_button,
                vr_crop_chw=perspective_crop_torch_rgb_uint8,  # already a focused face region
            )
            try:
                (
                    swapped_face_512_torch_rgb_uint8,
                    comprehensive_mask_1x512x512_from_swap_core,
                    _,
                ) = self.swap_core(
                    perspective_crop_torch_rgb_uint8,
                    kps_5_on_crop_param,
                    kps=kps_all_on_crop_param,
                    s_e=s_e_for_swap_core,
                    t_e=t_e_for_swap_np,
                    parameters=_params_for_swap_vr,
                    control=control_global,
                    dfm_model_name=parameters_for_face["DFMModelSelection"],
                    is_perspective_crop=True,
                    kv_map=kv_map_for_swap,
                )
            except Exception as e_swap_core:
                print(
                    f"[ERROR] Error in swap_core for VR crop {eye_side_for_debug}: {e_swap_core}"
                )
                traceback.print_exc()
                swapped_face_512_torch_rgb_uint8 = cast(v2.Resize, self.t512)(
                    perspective_crop_torch_rgb_uint8
                )
                comprehensive_mask_1x512x512_from_swap_core = torch.zeros(
                    (1, 512, 512),
                    dtype=torch.float32,
                    device=perspective_crop_torch_rgb_uint8.device,
                )

            tform_persp_to_512template = self.get_face_similarity_tform(
                parameters_for_face["SwapModelSelection"], kps_5_on_crop_param
            )

            # FW-PERF-5: use promoted instance-attribute transform for masks
            t512_mask = self.t512_mask
            assert t512_mask is not None, (
                "t512_mask must be initialized via set_scaling_transforms"
            )

            if (
                comprehensive_mask_1x512x512_from_swap_core is None
                or comprehensive_mask_1x512x512_from_swap_core.numel() == 0
            ):
                # Fallback mask logic
                persp_final_combined_mask_1x512x512_float_for_paste = (
                    t512_mask(self.get_border_mask(parameters_for_face.data)[0]).float()
                    if swap_button_is_checked_global
                    else torch.zeros(
                        (1, 512, 512),
                        dtype=torch.float32,
                        device=perspective_crop_torch_rgb_uint8.device,
                    )
                )
                vr_swap_mask_for_compare = None
            else:
                # Primary path
                persp_final_combined_mask_1x512x512_float_for_paste = (
                    comprehensive_mask_1x512x512_from_swap_core.float()
                )
                vr_swap_mask_for_compare = (
                    comprehensive_mask_1x512x512_from_swap_core
                    if self.is_view_face_mask
                    else None
                )

                if parameters_for_face.get("BordermaskEnableToggle", False):
                    border_mask_128, _ = self.get_border_mask(parameters_for_face.data)
                    border_mask_512 = t512_mask(border_mask_128)
                    persp_final_combined_mask_1x512x512_float_for_paste *= (
                        border_mask_512
                    )

            persp_final_combined_mask_3x512x512_float_for_paste = (
                persp_final_combined_mask_1x512x512_float_for_paste.repeat(3, 1, 1)
            )
            masked_swapped_face_to_paste_float = (
                swapped_face_512_torch_rgb_uint8.float()
                * persp_final_combined_mask_3x512x512_float_for_paste
            )

            crop_h, crop_w = (
                perspective_crop_torch_rgb_uint8.shape[1],
                perspective_crop_torch_rgb_uint8.shape[2],
            )
            _, source_grid_normalized_xy_persp = get_grid_for_pasting(
                tform_persp_to_512template,
                crop_h,
                crop_w,
                512,
                512,
                perspective_crop_torch_rgb_uint8.device,
            )
            # Bug 5 fix: use align_corners=True to match the E2P/P2E projection convention.
            # Previously align_corners=False introduced a sub-pixel shift (~0.2% for 512px)
            # at the face-paste step, causing slight misalignment at crop boundaries.
            pasted_face_on_persp_float = torch.nn.functional.grid_sample(
                masked_swapped_face_to_paste_float.unsqueeze(0),
                source_grid_normalized_xy_persp,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            ).squeeze(0)
            transformed_mask_on_persp_float = torch.nn.functional.grid_sample(
                persp_final_combined_mask_3x512x512_float_for_paste.unsqueeze(0),
                source_grid_normalized_xy_persp,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            ).squeeze(0)
            blended_persp_crop_float = (
                pasted_face_on_persp_float
                + perspective_crop_torch_rgb_uint8.float()
                * (1.0 - transformed_mask_on_persp_float)
            )
            processed_crop_torch_rgb_uint8 = torch.clamp(
                blended_persp_crop_float, 0, 255
            ).byte()

            if edit_button_is_checked_global:
                _, _, kps_all_for_editor_list = self.models_processor.run_detect(
                    processed_crop_torch_rgb_uint8,
                    control_global["DetectorModelSelection"],
                    max_num=1,
                    score=control_global["DetectorScoreSlider"] / 100.0,
                    input_size=(
                        processed_crop_torch_rgb_uint8.shape[1],
                        processed_crop_torch_rgb_uint8.shape[2],
                    ),
                    use_landmark_detection=True,
                    landmark_detect_mode="203",
                    landmark_score=control_global["LandmarkDetectScoreSlider"] / 100.0,
                    from_points=True,
                    rotation_angles=[0],
                    use_mean_eyes=control_global.get("LandmarkMeanEyesToggle", False),
                )
                kps_all_for_editor_on_crop = (
                    kps_all_for_editor_list[0]
                    if len(kps_all_for_editor_list) > 0
                    else None
                )
                if (
                    kps_all_for_editor_on_crop is not None
                    and kps_all_for_editor_on_crop.size > 0
                ):
                    processed_crop_torch_rgb_uint8 = (
                        self.frame_edits.swap_edit_face_core(
                            processed_crop_torch_rgb_uint8,
                            processed_crop_torch_rgb_uint8,
                            parameters_for_face.data,
                            control_global,
                        )
                    )
                    if any(
                        parameters_for_face.get(f, False)
                        for f in (
                            "FaceMakeupEnableToggle",
                            "HairMakeupEnableToggle",
                            "EyeBrowsMakeupEnableToggle",
                            "LipsMakeupEnableToggle",
                        )
                    ):
                        assert (
                            kps_all_on_crop_param is not None
                        )  # guarded by outer landmark check
                        processed_crop_torch_rgb_uint8 = (
                            self.frame_edits.swap_edit_face_core_makeup(
                                processed_crop_torch_rgb_uint8,
                                kps_all_on_crop_param,
                                parameters_for_face.data,
                                control_global,
                            )
                        )

        return processed_crop_torch_rgb_uint8, vr_swap_mask_for_compare

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
        processed_tensor_rgb_uint8 = (
            torch.from_numpy(img_numpy_rgb_uint8)
            .to(self.models_processor.device)
            .permute(2, 0, 1)
        )

        # --- ROUTING LOGIC ---
        if control.get("VR180ModeEnableToggle", False):
            processed_tensor_rgb_uint8 = self._process_frame_vr180(
                processed_tensor_rgb_uint8, img_numpy_rgb_uint8, control, stop_event
            )
        else:
            processed_tensor_rgb_uint8 = self._process_frame_standard(
                processed_tensor_rgb_uint8, control, stop_event
            )

        # --- Common Post-Processing (Enhancers, etc.) ---
        compare_mode_active = self.is_view_face_mask or self.is_view_face_compare

        if control["FrameEnhancerEnableToggle"] and not compare_mode_active:
            # Check 5: Before final heavy operation
            if stop_event.is_set():
                assert img_numpy_rgb_uint8 is not None
                return img_numpy_rgb_uint8[..., ::-1]

            processed_tensor_rgb_uint8 = self.frame_enhancers.enhance_core(
                processed_tensor_rgb_uint8, control=control
            )

        final_img_np_rgb_uint8 = (
            processed_tensor_rgb_uint8.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        )
        if not final_img_np_rgb_uint8.flags["C_CONTIGUOUS"]:
            final_img_np_rgb_uint8 = np.ascontiguousarray(final_img_np_rgb_uint8)

        return final_img_np_rgb_uint8[..., ::-1]

    # ------------------------------------------------------------------
    # VR180 helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _rodrigues_np(axis_angle: np.ndarray) -> np.ndarray:
        """Compute a 3×3 rotation matrix from an axis-angle vector (Rodrigues formula).
        Pure numpy — no cv2 dependency needed for the VR bbox projection helper."""
        angle = float(np.linalg.norm(axis_angle))
        if angle < 1e-10:
            return np.eye(3, dtype=np.float64)
        k = axis_angle / angle
        K = np.array(
            [[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]],
            dtype=np.float64,
        )
        return np.eye(3) + math.sin(angle) * K + (1.0 - math.cos(angle)) * (K @ K)

    @staticmethod
    def _project_crop_bbox_to_equirect(
        bbox_crop: np.ndarray,
        theta: float,
        phi: float,
        fov: float,
        crop_size: int,
        eq_h: int,
        eq_w: int,
    ) -> "np.ndarray | None":
        """Project a crop-space bounding box back to equirectangular pixel coordinates.

        Samples 9 points (4 corners + 4 edge midpoints + centre) through the same
        forward rotation used by E2P, then takes the axis-aligned bounding box of
        the projected points.

        Returns (x1, y1, x2, y2) float32 in equirect pixel space, or None if the
        resulting box is degenerate (<2 px on either side).
        """
        y_axis = np.array([0.0, 1.0, 0.0], np.float64)
        z_axis = np.array([0.0, 0.0, 1.0], np.float64)
        R1 = FrameWorker._rodrigues_np(z_axis * math.radians(theta))
        R2 = FrameWorker._rodrigues_np(np.dot(R1, y_axis) * math.radians(-phi))
        R = R2 @ R1

        w_len = math.tan(math.radians(fov / 2.0))
        equ_cx = (eq_w - 1) / 2.0
        equ_cy = (eq_h - 1) / 2.0

        x1_c, y1_c, x2_c, y2_c = (
            float(bbox_crop[0]),
            float(bbox_crop[1]),
            float(bbox_crop[2]),
            float(bbox_crop[3]),
        )
        mx_c = (x1_c + x2_c) / 2.0
        my_c = (y1_c + y2_c) / 2.0

        sample_pts = [
            (x1_c, y1_c),
            (mx_c, y1_c),
            (x2_c, y1_c),
            (x1_c, my_c),
            (mx_c, my_c),
            (x2_c, my_c),
            (x1_c, y2_c),
            (mx_c, y2_c),
            (x2_c, y2_c),
        ]

        lon_pxs: list[float] = []
        lat_pxs: list[float] = []
        for u, v in sample_pts:
            px_norm = 2.0 * u / (crop_size - 1) - 1.0
            py_norm = 2.0 * v / (crop_size - 1) - 1.0
            d = np.array([1.0, w_len * px_norm, -w_len * py_norm], np.float64)
            d_norm = d / np.linalg.norm(d)
            d_world = R @ d_norm
            lon = math.atan2(d_world[1], d_world[0])
            lat = math.asin(float(np.clip(d_world[2], -1.0, 1.0)))
            lon_pxs.append((lon / math.pi) * equ_cx + equ_cx)
            lat_pxs.append((-lat / (math.pi / 2.0)) * equ_cy + equ_cy)

        x1_eq = float(np.clip(min(lon_pxs), 0.0, eq_w - 1))
        y1_eq = float(np.clip(min(lat_pxs), 0.0, eq_h - 1))
        x2_eq = float(np.clip(max(lon_pxs), 0.0, eq_w - 1))
        y2_eq = float(np.clip(max(lat_pxs), 0.0, eq_h - 1))

        if x2_eq - x1_eq < 2.0 or y2_eq - y1_eq < 2.0:
            return None
        return np.array([x1_eq, y1_eq, x2_eq, y2_eq], dtype=np.float32)

    def _detect_faces_vr_tiled(
        self,
        equirect_converter: EquirectangularConverter,
        control: dict,
        eq_h: int,
        eq_w: int,
        crop_size: int,
        stop_event: threading.Event | None = None,
    ) -> list:
        """Detect faces in a grid of undistorted perspective crops.

        Catches faces that are invisible to the equirect-domain detector because
        of projection distortion (high elevation, near the ±180° seam, or very
        large near-camera faces).

        Returns a list of float32 numpy arrays, each (4,) or (5,) in equirect
        pixel coordinates.
        """
        # Full-sphere tile grid: (theta_deg, phi_deg)
        # 60° horizontal spacing at each latitude band gives ~30° overlap with a 90° FOV tile.
        TILE_GRID = [
            # equator band
            (-150, 0),
            (-90, 0),
            (-30, 0),
            (30, 0),
            (90, 0),
            (150, 0),
            # upper band ~40°
            (-150, 40),
            (-90, 40),
            (-30, 40),
            (30, 40),
            (90, 40),
            (150, 40),
            # lower band ~-40°
            (-150, -40),
            (-90, -40),
            (-30, -40),
            (30, -40),
            (90, -40),
            (150, -40),
            # near-pole tiles — 3 per pole is sufficient given the 90° FOV
            (-90, 70),
            (0, 70),
            (90, 70),
            (-90, -70),
            (0, -70),
            (90, -70),
        ]
        tile_fov = 90.0
        det_score = control["DetectorScoreSlider"] / 100.0
        found: list = []

        for theta, phi in TILE_GRID:
            if stop_event is not None and stop_event.is_set():
                break
            tile_crop = equirect_converter.get_perspective_crop(
                tile_fov, theta, phi, crop_size, crop_size
            )
            if tile_crop is None or tile_crop.numel() == 0:
                continue

            bboxes_tile, _, _ = self.models_processor.run_detect(
                tile_crop,
                control["DetectorModelSelection"],
                max_num=control["MaxFacesToDetectSlider"],
                score=det_score,
                use_landmark_detection=False,
                landmark_detect_mode=control["LandmarkDetectModelSelection"],
                landmark_score=control["LandmarkDetectScoreSlider"] / 100.0,
                from_points=False,
                rotation_angles=[0],
                use_mean_eyes=False,
                previous_detections=None,
            )

            if not isinstance(bboxes_tile, np.ndarray):
                bboxes_tile = np.array(bboxes_tile)
            if bboxes_tile.ndim == 1 and bboxes_tile.shape[0] in (4, 5):
                bboxes_tile = bboxes_tile.reshape(1, -1)
            if bboxes_tile.ndim != 2 or bboxes_tile.shape[0] == 0:
                continue

            for bbox_tile in bboxes_tile:
                eq_bbox = self._project_crop_bbox_to_equirect(
                    bbox_tile[:4], theta, phi, tile_fov, crop_size, eq_h, eq_w
                )
                if eq_bbox is None:
                    continue
                if bboxes_tile.shape[1] >= 5:
                    found.append(
                        np.append(eq_bbox, float(bbox_tile[4])).astype(np.float32)
                    )
                else:
                    found.append(eq_bbox)

        return found

    def _process_frame_vr180(
        self,
        original_equirect_tensor_for_vr: torch.Tensor,
        img_numpy_rgb_uint8: np.ndarray,
        control: dict,
        stop_event: threading.Event,
    ) -> torch.Tensor:
        """
        Handles the specific logic for VR180/360 frames:
        - Equirectangular detection
        - Perspective cropping
        - Processing per crop
        - Stitching back
        """
        # FW-RACE-02: read from snapshotted feeder state instead of live Qt button
        swap_button_is_checked_global = self.local_control_state_from_feeder.get(
            "swap_enabled", True
        )
        edit_button_is_checked_global = self.local_control_state_from_feeder.get(
            "edit_enabled", True
        )

        # VR-08: cache the EquirectangularConverter instance at worker level.
        # When the frame size is unchanged (common for video), reuse the cached
        # instance and update its image data in-place to avoid redundant Python-side
        # object allocation.
        _cur_frame_size = (img_numpy_rgb_uint8.shape[0], img_numpy_rgb_uint8.shape[1])
        if self._vr_converter is None or self._vr_frame_size != _cur_frame_size:
            self._vr_converter = EquirectangularConverter(
                img_numpy_rgb_uint8, device=self.models_processor.device
            )
            self._vr_frame_size = _cur_frame_size
        else:
            # VR-MEM-01: update in-place to avoid new GPU allocation per frame.
            self._vr_converter.equirect_tensor_cxhxw_rgb_uint8.copy_(
                torch.from_numpy(img_numpy_rgb_uint8).permute(2, 0, 1)
            )
            # VR-PERF-11: invalidate the per-frame float cache so GetPerspective
            # recomputes it once from the fresh uint8 data (instead of 24+ times).
            self._vr_converter.e2p_instance._img_float = None
        equirect_converter = self._vr_converter
        assert equirect_converter is not None, (
            "VR converter must be initialized before use"
        )

        # VR-11: always use rotation_angles=[0] in VR mode — non-zero angles produce
        # geometrically invalid spherical coordinates.
        vr_rotation_angles = [0]

        # Improvement F: read perspective crop resolution from UI (512/768/1024).
        self.VR_PERSPECTIVE_RENDER_SIZE = int(
            control.get("VR180CropResolutionSelection", "512")
        )
        # VR-15: ensure scaling transforms are set for this resolution.
        # FW-PERF-07b: mirror the standard-path dirty-check so set_scaling_transforms
        # is not called on every VR frame when settings have not changed.
        _vr_scaling_keys = {
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
        _vr_scaling_keys["VR180CropResolutionSelection"] = (
            self.VR_PERSPECTIVE_RENDER_SIZE
        )
        if self._last_vr_scaling_control != _vr_scaling_keys:
            self.set_scaling_transforms(control)
            self._last_vr_scaling_control = _vr_scaling_keys

        # Detection interval / previous-detections setup (mirrors standard-mode logic).
        _vr_detection_interval = int(control.get("FaceDetectionIntervalSlider", 1))
        _previous_faces_vr: list | None = None
        with self.lock:
            _last_detected_vr = self.last_detected_faces_vr
            _last_frame_no_vr = self.last_processed_frame_number_vr
        if (
            len(_last_detected_vr) > 0
            and self.frame_number % _vr_detection_interval != 0
            and self.frame_number == _last_frame_no_vr + 1
        ):
            _previous_faces_vr = _last_detected_vr

        # Improvement D: in Both-Eyes mode, detect on each eye-half separately.
        # A standard VR180 SBS equirect is 2:1 (W=2H), so each half is square (H×H).
        # Detecting on each 1:1 half gives the detector 4× more pixels per face vs
        # letterboxing the full 2:1 equirect to 512×256.
        _eq_h = img_numpy_rgb_uint8.shape[0]
        _eq_w = img_numpy_rgb_uint8.shape[1]
        _is_both_eyes = (
            control.get("VR180EyeModeSelection", "Both Eyes") != "Single Eye"
        )
        _aspect = _eq_w / max(_eq_h, 1)
        _do_per_eye_detect = _is_both_eyes and _aspect >= 1.8

        _det_score = control["DetectorScoreSlider"] / 100.0
        _det_mode = control["DetectorModelSelection"]
        _det_max = control["MaxFacesToDetectSlider"]
        _lm_mode = control["LandmarkDetectModelSelection"]
        _lm_score = control["LandmarkDetectScoreSlider"] / 100.0
        _use_mean_eyes = control.get("LandmarkMeanEyesToggle", False)

        def _run_det(tensor, prev_dets=None, bypass_bytetrack=False):
            return self.models_processor.run_detect(
                tensor,
                _det_mode,
                max_num=_det_max,
                score=_det_score,
                use_landmark_detection=False,
                landmark_detect_mode=_lm_mode,
                landmark_score=_lm_score,
                from_points=False,
                rotation_angles=vr_rotation_angles,
                use_mean_eyes=_use_mean_eyes,
                previous_detections=prev_dets,
                bypass_bytetrack=bypass_bytetrack,
            )

        def _norm_bboxes(arr):
            if not isinstance(arr, np.ndarray):
                arr = np.array(arr)
            if arr.ndim == 1 and arr.shape[0] in (4, 5):
                arr = arr.reshape(1, -1)
            return arr

        if _do_per_eye_detect:
            _half_w = original_equirect_tensor_for_vr.shape[2] // 2
            _left_tensor = original_equirect_tensor_for_vr[:, :, :_half_w]
            _right_tensor = original_equirect_tensor_for_vr[:, :, _half_w:]

            # Split previous_detections into per-eye coordinate spaces
            _prev_left: list | None = None
            _prev_right: list | None = None
            if _previous_faces_vr is not None:
                _prev_left = [
                    f
                    for f in _previous_faces_vr
                    if (f["bbox"][0] + f["bbox"][2]) / 2.0 < _half_w
                ]
                _prev_right = [
                    {
                        "bbox": [
                            f["bbox"][0] - _half_w,
                            f["bbox"][1],
                            f["bbox"][2] - _half_w,
                            f["bbox"][3],
                        ],
                        "score": f.get("score", 1.0),
                    }
                    for f in _previous_faces_vr
                    if (f["bbox"][0] + f["bbox"][2]) / 2.0 >= _half_w
                ]

            # bypass_bytetrack=True: per-eye detection uses half-width coordinate spaces,
            # which would corrupt the tracker's Kalman state if ByteTrack ran on both halves.
            # Simple fallback tracking (via previous_detections) handles the interval skip.
            _bboxes_left = _norm_bboxes(
                _run_det(_left_tensor, _prev_left, bypass_bytetrack=True)[0]
            )
            _bboxes_right = _norm_bboxes(
                _run_det(_right_tensor, _prev_right, bypass_bytetrack=True)[0]
            )

            # Offset right-half x-coordinates into full-equirect space
            if _bboxes_right.ndim == 2 and _bboxes_right.shape[0] > 0:
                _bboxes_right = _bboxes_right.copy()
                _bboxes_right[:, 0] += _half_w
                _bboxes_right[:, 2] += _half_w

            _pieces = []
            if _bboxes_left.ndim == 2 and _bboxes_left.shape[0] > 0:
                _pieces.append(_bboxes_left)
            if _bboxes_right.ndim == 2 and _bboxes_right.shape[0] > 0:
                _pieces.append(_bboxes_right)
            bboxes_eq_np = np.vstack(_pieces) if _pieces else np.array([])
        else:
            # Single Eye or non-2:1 equirect — detect on the full frame as before.
            # Use the standard (512, 512) input size — TRT engines are compiled for this shape.
            bboxes_eq_np = _norm_bboxes(
                _run_det(original_equirect_tensor_for_vr, _previous_faces_vr)[0]
            )

        if not isinstance(bboxes_eq_np, np.ndarray):
            bboxes_eq_np = np.array(bboxes_eq_np)

        # FW-ROBUST-07: reshape 1-D bbox (e.g. single face returned as shape (4,) or (5,))
        if bboxes_eq_np.ndim == 1 and bboxes_eq_np.shape[0] in (4, 5):
            bboxes_eq_np = bboxes_eq_np.reshape(1, -1)

        # Improvement A: tiled perspective detection — catch faces missed by equirect
        # detection (distorted at high elevation, near ±180° seam, or large near-camera faces).
        # Guard: only run on detection keyframes (same interval as equirect detection) to avoid
        # running 24 full inference passes every frame, which causes major slowdowns.
        _is_detection_keyframe = (
            _vr_detection_interval <= 1
            or self.frame_number % _vr_detection_interval == 0
        )
        # Custom provider serializes all CUDA graph runs through per-runner locks.
        # 24 tiles × serial execution = unacceptable latency for VR recording.
        # Skip tile detection for Custom provider entirely.
        _skip_tile_det = self.models_processor.provider_name == "Custom"
        if (
            control.get("VR180TileDetectionToggle", False)
            and _is_detection_keyframe
            and not _skip_tile_det
        ):
            _tile_bboxes = self._detect_faces_vr_tiled(
                equirect_converter,
                control,
                _eq_h,
                _eq_w,
                self.VR_PERSPECTIVE_RENDER_SIZE,
                stop_event=stop_event,
            )
            if _tile_bboxes:
                _tile_arr = np.array(_tile_bboxes, dtype=np.float32)
                if _tile_arr.ndim == 1 and _tile_arr.shape[0] in (4, 5):
                    _tile_arr = _tile_arr.reshape(1, -1)
                if _tile_arr.ndim == 2 and _tile_arr.shape[0] > 0:
                    _dev = self.models_processor.device
                    _tile_boxes_t = torch.from_numpy(
                        _tile_arr[:, :4].astype(np.float32)
                    ).to(_dev)

                    # Step 1 — intra-tile NMS: the same face is often detected by
                    # multiple overlapping tiles (90° FOV with 60° spacing gives 30°
                    # of overlap).  Run a tight NMS to merge these near-duplicates
                    # before comparing against the equirect detections.
                    if _tile_arr.shape[0] > 1:
                        _t_scores = (
                            torch.from_numpy(_tile_arr[:, 4].astype(np.float32))
                            .to(_dev)
                            .clamp(min=1e-6)
                            if _tile_arr.shape[1] >= 5
                            else (
                                (_tile_boxes_t[:, 2] - _tile_boxes_t[:, 0])
                                * (_tile_boxes_t[:, 3] - _tile_boxes_t[:, 1])
                            ).clamp(min=0.0)
                        )
                        _t_keep = torchvision.ops.nms(
                            _tile_boxes_t, _t_scores, iou_threshold=0.3
                        )
                        _tile_arr = _tile_arr[_t_keep.cpu().numpy()]
                        _tile_boxes_t = torch.from_numpy(
                            _tile_arr[:, :4].astype(np.float32)
                        ).to(_dev)

                    # Step 2 — suppress tile detections that already have a
                    # corresponding equirect detection.  If a tile bbox overlaps any
                    # equirect bbox by IoU ≥ 0.3, it represents the same face and
                    # adding it would cause double-processing (two swaps stitched on
                    # top of each other → artifacts, wrong size, colour shift).
                    # Only genuinely NEW detections (IoU < 0.3 with all equirect
                    # bboxes) are merged into the candidate list.
                    if bboxes_eq_np.ndim == 2 and bboxes_eq_np.shape[0] > 0:
                        _eq_boxes_t = torch.from_numpy(
                            bboxes_eq_np[:, :4].astype(np.float32)
                        ).to(_dev)
                        # pairwise IoU matrix: shape (N_tile, N_equirect)
                        _iou_tile_vs_eq = torchvision.ops.box_iou(
                            _tile_boxes_t, _eq_boxes_t
                        )
                        _novel_mask = (
                            (_iou_tile_vs_eq.max(dim=1).values < 0.3).cpu().numpy()
                        )
                        _novel_tile = _tile_arr[_novel_mask]
                        if _novel_tile.shape[0] > 0:
                            # Pad score column with 0.9 default where missing
                            _n_cols = max(bboxes_eq_np.shape[1], _novel_tile.shape[1])
                            if bboxes_eq_np.shape[1] < _n_cols:
                                bboxes_eq_np = np.hstack(
                                    [
                                        bboxes_eq_np,
                                        np.full(
                                            (
                                                bboxes_eq_np.shape[0],
                                                _n_cols - bboxes_eq_np.shape[1],
                                            ),
                                            0.9,
                                            dtype=np.float32,
                                        ),
                                    ]
                                )
                            if _novel_tile.shape[1] < _n_cols:
                                _novel_tile = np.hstack(
                                    [
                                        _novel_tile,
                                        np.full(
                                            (
                                                _novel_tile.shape[0],
                                                _n_cols - _novel_tile.shape[1],
                                            ),
                                            0.9,
                                            dtype=np.float32,
                                        ),
                                    ]
                                )
                            bboxes_eq_np = np.vstack([bboxes_eq_np, _novel_tile])
                    else:
                        # No equirect detections at all — use all (intra-NMS'd) tile bboxes
                        bboxes_eq_np = _tile_arr

        # VR-03 / Bug 2 fix: IoU-NMS using detector confidence score when available,
        # falling back to bbox area only when scores are not present.
        if bboxes_eq_np.ndim == 2 and bboxes_eq_np.shape[0] > 1:
            _boxes_t = torch.from_numpy(bboxes_eq_np[:, :4].astype(np.float32)).to(
                self.models_processor.device
            )
            if bboxes_eq_np.shape[1] >= 5:
                # Use detector confidence score — keeps the highest-confidence detection
                _scores_t = (
                    torch.from_numpy(bboxes_eq_np[:, 4].astype(np.float32))
                    .to(self.models_processor.device)
                    .clamp(min=1e-6)
                )
            else:
                # Fall back to area when no confidence score column is present
                _scores_t = (
                    (_boxes_t[:, 2] - _boxes_t[:, 0])
                    * (_boxes_t[:, 3] - _boxes_t[:, 1])
                ).clamp(min=0.0)
            _keep = torchvision.ops.nms(_boxes_t, _scores_t, iou_threshold=0.5)
            bboxes_eq_np = bboxes_eq_np[_keep.cpu().numpy()]

        # Update VR tracking state for the next frame. Done here — post-NMS so the
        # stored bboxes are clean, pre-VR-MIRROR so synthetic bboxes are not tracked.
        _vr_state_for_next: list = (
            [
                {"bbox": row[:4], "score": float(row[4]) if len(row) >= 5 else 1.0}
                for row in bboxes_eq_np
            ]
            if bboxes_eq_np.ndim == 2 and bboxes_eq_np.shape[0] > 0
            else []
        )
        with self.lock:
            self.last_detected_faces_vr = _vr_state_for_next
            self.last_processed_frame_number_vr = self.frame_number

        if len(bboxes_eq_np) == 0:
            return original_equirect_tensor_for_vr

        # VR-MIRROR: in Both-Eyes mode, if a face is detected on one eye side but not
        # the other, synthesize a mirrored bbox on the missing side using the same
        # relative position and size. This prevents one-eye-only swaps when the detector
        # fires on one half but not the other due to marginal confidence or rendering
        # differences between the two stereo views.
        # Only applies to Both-Eyes VR180 (not Single Eye, not VR360 panoramic).
        if (
            control.get("VR180EyeModeSelection", "Both Eyes") != "Single Eye"
            and bboxes_eq_np.ndim == 2
            and bboxes_eq_np.shape[0] > 0
        ):
            _half_w = equirect_converter.width / 2.0
            _cx = (bboxes_eq_np[:, 0] + bboxes_eq_np[:, 2]) / 2.0
            _is_left = _cx < _half_w
            _mirrored_bboxes: list[np.ndarray] = []
            for _i in range(len(bboxes_eq_np)):
                _b = bboxes_eq_np[_i]
                _tol = max(_b[2] - _b[0], _b[3] - _b[1]) * 0.5
                if _is_left[_i]:
                    _mx1, _mx2 = _b[0] + _half_w, _b[2] + _half_w
                    _other_mask = ~_is_left
                else:
                    _mx1, _mx2 = _b[0] - _half_w, _b[2] - _half_w
                    _other_mask = _is_left
                _mcx = (_mx1 + _mx2) / 2.0
                _mcy = (_b[1] + _b[3]) / 2.0
                _found = False
                for _j in np.where(_other_mask)[0]:
                    _ocx = (bboxes_eq_np[_j, 0] + bboxes_eq_np[_j, 2]) / 2.0
                    _ocy = (bboxes_eq_np[_j, 1] + bboxes_eq_np[_j, 3]) / 2.0
                    if abs(_ocx - _mcx) < _tol and abs(_ocy - _mcy) < _tol:
                        _found = True
                        break
                if not _found:
                    _mb = _b.copy()
                    _mb[0] = _mx1
                    _mb[2] = _mx2
                    _mirrored_bboxes.append(_mb)
            if _mirrored_bboxes:
                bboxes_eq_np = np.vstack([bboxes_eq_np, np.array(_mirrored_bboxes)])

        # VR-07: use list of namedtuple-style tuples instead of a string-keyed dict
        # to avoid string allocation for each face and simplify downstream iteration
        processed_perspective_crops_details = []  # each entry: (eye_side, tensor, theta, phi, fov)
        analyzed_faces_for_vr = []
        # Crops for detected faces that had no matching target — shown as-is in compare/mask mode
        unmatched_compare_crops: list[torch.Tensor] = []
        compare_mode_vr_early = self.is_view_face_mask or self.is_view_face_compare

        # Phase 1: extract all perspective crops and run landmark detection.
        # Failures are kept (kps_on_crop=None) so Phase 2 can attempt to fill them
        # from the partner eye view before discarding.
        _crop_landmark_results: list[dict] = []
        for bbox_eq_single in bboxes_eq_np:
            if stop_event.is_set():
                break

            theta, phi = equirect_converter.calculate_theta_phi_from_bbox(
                bbox_eq_single
            )
            original_eye_side = (
                "L"
                if (bbox_eq_single[0] + bbox_eq_single[2]) / 2
                < equirect_converter.width / 2
                else "R"
            )
            angular_width_deg = (
                (bbox_eq_single[2] - bbox_eq_single[0])
                / equirect_converter.width
                * 360.0
            )
            angular_height_deg = (
                (bbox_eq_single[3] - bbox_eq_single[1])
                / equirect_converter.height
                * 180.0
            )

            # VR-13: use renamed VR_FOV_SCALE_FACTOR (was VR_DYNAMIC_FOV_PADDING_FACTOR)
            # VR-02: correct angular_width_deg for latitude compression in equirectangular
            # projection: true angular width ≈ equirect_width / cos(phi_radians)
            phi_radians = math.radians(phi)
            cos_phi = math.cos(phi_radians)
            # Avoid division by near-zero at poles; clamp cos_phi to at least 0.1
            angular_width_deg_corrected = angular_width_deg / max(cos_phi, 0.1)
            # Improvement C: use VR180MaxFOVSlider instead of the previous hard-cap of
            # 100°.  Near-camera faces can subtend >100° and were being clipped.
            _vr_max_fov = float(control.get("VR180MaxFOVSlider", 120))
            dynamic_fov_for_crop = float(
                np.clip(
                    max(angular_width_deg_corrected, angular_height_deg)
                    * self.VR_FOV_SCALE_FACTOR,
                    15.0,
                    _vr_max_fov,
                )
            )

            face_crop_tensor = equirect_converter.get_perspective_crop(
                dynamic_fov_for_crop,
                theta,
                phi,
                self.VR_PERSPECTIVE_RENDER_SIZE,
                self.VR_PERSPECTIVE_RENDER_SIZE,
            )
            if face_crop_tensor is None or face_crop_tensor.numel() == 0:
                continue

            # Landmark detection on the crop
            crop_size = self.VR_PERSPECTIVE_RENDER_SIZE
            padding = int(crop_size * 0.025)
            dummy_bbox_on_crop = np.array(
                [padding, padding, crop_size - padding, crop_size - padding]
            )

            kpss_5_crop_list, kpss_crop_list, _ = (
                self.models_processor.run_detect_landmark(
                    img=face_crop_tensor,
                    bbox=dummy_bbox_on_crop,
                    det_kpss=[],
                    detect_mode=control["LandmarkDetectModelSelection"],
                    score=control["LandmarkDetectScoreSlider"] / 100.0,
                    from_points=False,
                    use_mean_eyes=control.get("LandmarkMeanEyesToggle", False),
                )
            )

            kpss_5_crop = [kpss_5_crop_list] if len(kpss_5_crop_list) > 0 else []
            kpss_crop = [kpss_crop_list] if len(kpss_crop_list) > 0 else []

            # FW-BUG-07: fixed check — kpss_crop is a list not ndarray; use truthiness
            landmark_ok = (
                isinstance(kpss_5_crop, np.ndarray)
                and kpss_5_crop.shape[0] > 0
                or isinstance(kpss_5_crop, list)
                and len(kpss_5_crop) > 0
            )
            _crop_landmark_results.append(
                {
                    "theta": theta,
                    "phi": phi,
                    "original_eye_side": original_eye_side,
                    "face_crop_tensor": face_crop_tensor,
                    "fov_used_for_crop": dynamic_fov_for_crop,
                    "kps_on_crop": kpss_5_crop[0] if landmark_ok else None,
                    "kps_all_on_crop": (
                        kpss_crop[0] if (landmark_ok and kpss_crop) else None
                    ),
                }
            )

        # Improvement I / VR-LANDMARK-MIRROR:
        # Step 1: for faces that failed landmark detection, retry with a reduced
        # score threshold (half of user setting) before falling back to stereo copy.
        # This avoids importing wrong-eye keypoints when the face is clearly present
        # in the crop but the detector is marginally below threshold.
        _lm_score_orig = control["LandmarkDetectScoreSlider"] / 100.0
        _lm_score_retry = max(0.05, _lm_score_orig * 0.5)
        _lm_detect_mode = control["LandmarkDetectModelSelection"]
        _lm_use_mean = control.get("LandmarkMeanEyesToggle", False)
        for _fd in _crop_landmark_results:
            if _fd["kps_on_crop"] is not None:
                continue
            _cs = _fd["face_crop_tensor"].shape[-1]
            _pad = int(_cs * 0.025)
            _dummy_bb = np.array([_pad, _pad, _cs - _pad, _cs - _pad])
            _r5, _rall, _ = self.models_processor.run_detect_landmark(
                img=_fd["face_crop_tensor"],
                bbox=_dummy_bb,
                det_kpss=[],
                detect_mode=_lm_detect_mode,
                score=_lm_score_retry,
                from_points=False,
                use_mean_eyes=_lm_use_mean,
            )
            if len(_r5) > 0:
                _fd["kps_on_crop"] = _r5
                _fd["kps_all_on_crop"] = _rall if len(_rall) > 0 else None

        # Step 2: VR-LANDMARK-MIRROR — if a face crop still has no landmarks but the
        # partner eye view (same face, other VR180 half) succeeded, reuse those
        # keypoints as a last resort. In the equirectangular projection the partner's
        # theta is exactly ±180° away (mirrored x-offset = width/2).
        if control.get("VR180EyeModeSelection", "Both Eyes") != "Single Eye":
            _kps_cache: dict[tuple[float, float], tuple] = {}
            for _fd in _crop_landmark_results:
                if _fd["kps_on_crop"] is not None:
                    _kps_cache[(_fd["theta"], _fd["phi"])] = (
                        _fd["kps_on_crop"],
                        _fd["kps_all_on_crop"],
                    )
            for _fd in _crop_landmark_results:
                if _fd["kps_on_crop"] is None:
                    _mir_theta = (
                        _fd["theta"] + 180.0
                        if "L" in _fd["original_eye_side"]
                        else _fd["theta"] - 180.0
                    )
                    for (_ct, _cp), _ckps in _kps_cache.items():
                        if abs(_ct - _mir_theta) < 1.0 and abs(_cp - _fd["phi"]) < 0.5:
                            _fd["kps_on_crop"], _fd["kps_all_on_crop"] = _ckps
                            break

        # Phase 2: recognition and target matching for all faces with valid landmarks.
        for _fd in _crop_landmark_results:
            if stop_event.is_set():
                break

            if _fd["kps_on_crop"] is None:
                # Improvement H: log faces that are detected but cannot be swapped
                # (landmark detection failed even after retry and stereo fallback).
                print(
                    f"[VR] Skipping face at theta={_fd['theta']:.1f}° phi={_fd['phi']:.1f}° "
                    f"({_fd['original_eye_side']}) — landmark detection failed."
                )
                del _fd["face_crop_tensor"]
                continue

            kps_on_crop = _fd["kps_on_crop"]
            kps_all_on_crop = _fd["kps_all_on_crop"]
            face_crop_tensor = _fd["face_crop_tensor"]
            theta = _fd["theta"]
            phi = _fd["phi"]
            original_eye_side = _fd["original_eye_side"]
            dynamic_fov_for_crop = _fd["fov_used_for_crop"]

            face_emb_crop, _ = self.models_processor.run_recognize_direct(
                face_crop_tensor,
                kps_on_crop,
                control["SimilarityTypeSelection"],
                control["RecognitionModelSelection"],
            )

            best_target_button_vr, best_params_for_target_vr, _ = (
                self._find_best_target_match(face_emb_crop, control)
            )

            if best_target_button_vr:
                denoiser_on = (
                    control.get("DenoiserUNetEnableBeforeRestorersToggle", False)
                    or control.get("DenoiserAfterFirstRestorerToggle", False)
                    or control.get("DenoiserAfterRestorersToggle", False)
                )
                if (
                    denoiser_on
                    and best_target_button_vr.assigned_kv_map is None
                    and best_target_button_vr.assigned_input_faces
                ):
                    with self.models_processor.model_lock:
                        if (
                            best_target_button_vr.assigned_kv_map is None
                        ):  # Double Check inside lock
                            best_target_button_vr.calculate_assigned_input_embedding()

                analyzed_faces_for_vr.append(
                    {
                        "theta": theta,
                        "phi": phi,
                        "original_eye_side": original_eye_side,
                        "face_crop_tensor": face_crop_tensor,
                        "kps_on_crop": kps_on_crop,
                        "kps_all_on_crop": kps_all_on_crop,
                        "target_button": best_target_button_vr,
                        "params": best_params_for_target_vr,
                        "fov_used_for_crop": dynamic_fov_for_crop,
                    }
                )
            else:
                if compare_mode_vr_early:
                    # No target match — save the raw crop for compare display (shown as-is)
                    unmatched_compare_crops.append(face_crop_tensor)
                else:
                    del face_crop_tensor

        # Process collected faces
        compare_mode_active_vr = self.is_view_face_mask or self.is_view_face_compare
        vr_compare_crops: list[
            tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]
        ] = []
        # Include unmatched detected faces in compare view (original shown on both sides, no mask)
        for _unmatched in unmatched_compare_crops:
            vr_compare_crops.append((_unmatched, _unmatched, None))

        for item_data in analyzed_faces_for_vr:
            original_crop_for_compare = (
                item_data["face_crop_tensor"].clone()
                if compare_mode_active_vr
                else None
            )
            if swap_button_is_checked_global or edit_button_is_checked_global:
                processed_crop_for_stitching, vr_crop_swap_mask = (
                    self._process_single_vr_perspective_crop_multi(
                        item_data["face_crop_tensor"],
                        item_data["target_button"],
                        item_data["params"],
                        control,
                        kps_5_on_crop_param=item_data["kps_on_crop"],
                        kps_all_on_crop_param=item_data["kps_all_on_crop"],
                        swap_button_is_checked_global=swap_button_is_checked_global,
                        edit_button_is_checked_global=edit_button_is_checked_global,
                        eye_side_for_debug=item_data["original_eye_side"],
                        kv_map_for_swap=(
                            getattr(item_data["target_button"], "aged_kv_map", None)
                            if item_data["params"].get("FaceReagingEnableToggle", False)
                            and getattr(item_data["target_button"], "aged_kv_map", None)
                            is not None
                            else item_data["target_button"].assigned_kv_map
                        ),
                    )
                )
            else:
                processed_crop_for_stitching = item_data["face_crop_tensor"]
                vr_crop_swap_mask = None

            if compare_mode_active_vr and original_crop_for_compare is not None:
                vr_compare_crops.append(
                    (
                        original_crop_for_compare,
                        processed_crop_for_stitching,
                        vr_crop_swap_mask,
                    )
                )

            # VR-07: append (eye_side, tensor, theta, phi, fov) tuple instead of dict entry
            processed_perspective_crops_details.append(
                (
                    item_data["original_eye_side"],
                    processed_crop_for_stitching,
                    item_data["theta"],
                    item_data["phi"],
                    item_data["fov_used_for_crop"],
                )
            )
            del item_data["face_crop_tensor"]

        # Stitch back
        # FW-PERF-12: skip PerspectiveConverter instantiation when there are no crops to stitch
        if not processed_perspective_crops_details:
            _result_no_stitch = original_equirect_tensor_for_vr
            _det_faces_eq = [{"bbox": row[:4]} for row in bboxes_eq_np]
            if control.get("ShowAllDetectedFacesBBoxToggle", False) and _det_faces_eq:
                _result_no_stitch = draw_bounding_boxes_on_detected_faces(
                    _result_no_stitch, _det_faces_eq
                )
            if control.get("ShowByteTrackBBoxToggle", False) and _det_faces_eq:
                _result_no_stitch = draw_bounding_boxes_on_detected_faces(
                    _result_no_stitch, _det_faces_eq, color_rgb=[255, 165, 0]
                )
            return _result_no_stitch

        final_equirect_torch_cxhxw_rgb_uint8 = original_equirect_tensor_for_vr.clone()
        vr_single_eye_mode = (
            control.get("VR180EyeModeSelection", "Both Eyes") == "Single Eye"
        )
        # Improvement K: cache PerspectiveConverter across frames (like E2P converter).
        # PerspectiveConverter only uses img_numpy_rgb_uint8 for its dimensions; the
        # per-frame image content is not stored in the converter itself (stitch_single_perspective
        # receives the target tensor directly).  We can therefore reuse the cached instance
        # whenever the frame resolution is unchanged, avoiding repeated kernel allocation.
        # NOTE: deliberately uses _vr_p2e_frame_size (NOT _vr_frame_size) for this check.
        # _vr_frame_size is updated by the E2P branch above, so by the time we reach this
        # point it always equals _cur_frame_size.  A separate tracking attribute ensures the
        # P2E converter is properly recreated when the frame resolution changes.
        if self._vr_p2e_converter is None or self._vr_p2e_frame_size != _cur_frame_size:
            self._vr_p2e_converter = PerspectiveConverter(
                img_numpy_rgb_uint8, device=self.models_processor.device
            )
            self._vr_p2e_frame_size = _cur_frame_size
        p2e_converter = self._vr_p2e_converter

        # VR-15: check stop_event in stitching loop so we can abort early
        # VR-07: iterate over list of (eye_side, tensor, theta, phi, fov) tuples
        for (
            _eye_side,
            _crop_tensor,
            _theta,
            _phi,
            _fov,
        ) in processed_perspective_crops_details:
            if stop_event is not None and stop_event.is_set():
                break
            p2e_converter.stitch_single_perspective(
                target_equirect_torch_cxhxw_rgb_uint8=final_equirect_torch_cxhxw_rgb_uint8,
                processed_crop_torch_cxhxw_rgb_uint8=_crop_tensor,
                theta=_theta,
                phi=_phi,
                fov=_fov,
                is_left_eye=(None if vr_single_eye_mode else ("L" in _eye_side)),
            )
            # FW-MEM-02: release the crop tensor immediately after stitching
            del _crop_tensor

        processed_tensor_rgb_uint8 = final_equirect_torch_cxhxw_rgb_uint8

        # --- Bounding box overlays on the stitched equirectangular image ---
        _det_faces_eq = [{"bbox": row[:4]} for row in bboxes_eq_np]
        if control.get("ShowAllDetectedFacesBBoxToggle", False) and _det_faces_eq:
            processed_tensor_rgb_uint8 = draw_bounding_boxes_on_detected_faces(
                processed_tensor_rgb_uint8, _det_faces_eq
            )
        if control.get("ShowByteTrackBBoxToggle", False) and _det_faces_eq:
            processed_tensor_rgb_uint8 = draw_bounding_boxes_on_detected_faces(
                processed_tensor_rgb_uint8, _det_faces_eq, color_rgb=[255, 165, 0]
            )

        # Cleanup
        # VR-08: equirect_converter is now self._vr_converter (cached); do not del it
        # Improvement K: p2e_converter is now self._vr_p2e_converter (cached); do not del it
        del processed_perspective_crops_details, analyzed_faces_for_vr
        # FW-MEM-1: removed torch.cuda.empty_cache() — calling it per-frame
        # defeats the CUDA caching allocator and causes unnecessary overhead.

        # VR-MEM-03: periodic CUDA allocator flush to prevent VRAM fragmentation
        # over long recording sessions. Per-worker counter
        # staggers flushes across 8 pool workers so they don't all flush simultaneously.
        # empty_cache() returns fragmented free blocks to CUDA; gc.collect() clears
        # Python cyclic garbage that may hold tensor references.
        self._vr_processed_count += 1
        if self._vr_processed_count % 300 == 0:
            import gc

            torch.cuda.empty_cache()
            gc.collect()

        # --- VR Compare/Mask view ---
        # Build side-by-side crop strips when Face Compare or Face Mask is active.
        if compare_mode_active_vr and vr_compare_crops:
            imgs_to_vstack = []
            for original_chw, processed_chw, mask_1chw_float in vr_compare_crops:
                imgs_to_cat: list[torch.Tensor] = []
                if self.is_view_face_compare:
                    imgs_to_cat.append(original_chw)
                    imgs_to_cat.append(processed_chw)
                elif not self.is_view_face_mask:
                    # Neither compare nor mask — shouldn't reach here, but show processed
                    imgs_to_cat.append(processed_chw)
                if self.is_view_face_mask and mask_1chw_float is not None:
                    mask_display = torch.sub(1, mask_1chw_float).repeat(3, 1, 1)
                    mask_display = torch.mul(mask_display, 255.0).type(torch.uint8)
                    imgs_to_cat.append(mask_display)
                if imgs_to_cat:
                    imgs_to_vstack.append(torch.cat(imgs_to_cat, dim=2))
            if imgs_to_vstack:
                max_width = max(t.size(2) for t in imgs_to_vstack)
                padded = [
                    torch.nn.functional.pad(t, (0, max_width - t.size(2), 0, 0))
                    for t in imgs_to_vstack
                ]
                return torch.cat(padded, dim=1)

        return processed_tensor_rgb_uint8

    def _process_frame_standard(
        self,
        processed_tensor_rgb_uint8: torch.Tensor,
        control: dict,
        stop_event: threading.Event,
    ) -> torch.Tensor:
        """
        Handles the standard (flat) processing logic:
        - Rotation
        - Detection
        - Swapping/Editing
        - Overlays (BBox, Landmarks, Comparison)
        """
        # FW-RACE-02: read button state from snapshotted feeder dict, not live Qt buttons
        swap_button_is_checked_global = self.local_control_state_from_feeder.get(
            "swap_enabled", True
        )
        edit_button_is_checked_global = self.local_control_state_from_feeder.get(
            "edit_enabled", True
        )

        # FW-RACE-01: snapshot target_faces under lock so the worker thread never
        # iterates the live dict while the UI thread may be modifying it.
        with self.lock:
            target_faces_snapshot = dict(self.main_window.target_faces)

        det_faces_data_for_display = []

        img = processed_tensor_rgb_uint8
        # FW-QUAL-11: rename img_x/img_y to img_w/img_h for clarity
        img_w, img_h = img.size(2), img.size(1)
        scale_applied = False

        # FW-BUG-06: Upscale small frames so the shorter side is at least 512px before detection
        if img_w < 512 or img_h < 512:
            if img_w <= img_h:
                new_h, new_w = int(512 * img_h / img_w), 512
            else:
                new_h, new_w = 512, int(512 * img_w / img_h)
            # FW-ROBUST-2: respect the user's interpolation setting rather than
            # always using the default (NEAREST / BILINEAR depending on build)
            # FW-PERF-08 / FW-MEM-02: LRU-bounded cache of v2.Resize objects to avoid
            # re-constructing the same transform on every frame (max 16 entries).
            _up_key = (new_h, new_w, self.interpolation_scaleback, False)
            if _up_key in self._resize_cache:
                self._resize_cache.move_to_end(_up_key)
            else:
                if len(self._resize_cache) >= self._RESIZE_CACHE_MAX:
                    self._resize_cache.popitem(last=False)
                self._resize_cache[_up_key] = v2.Resize(
                    (new_h, new_w),
                    interpolation=self.interpolation_scaleback,
                    antialias=False,
                )
            img = self._resize_cache[_up_key](img)
            scale_applied = True
            # Upscale Feeder KPS if necessary
            if self.precomputed_bboxes is not None:
                ratio_w = new_w / img_w
                ratio_h = new_h / img_h

                self.precomputed_bboxes = [b.copy() for b in self.precomputed_bboxes]
                for b in self.precomputed_bboxes:
                    b[0] *= ratio_w
                    b[2] *= ratio_w
                    b[1] *= ratio_h
                    b[3] *= ratio_h

                if self.precomputed_kpss_5 is not None:
                    self.precomputed_kpss_5 = [
                        k.copy() for k in self.precomputed_kpss_5
                    ]
                    for k in self.precomputed_kpss_5:
                        k[:, 0] *= ratio_w
                        k[:, 1] *= ratio_h

                if self.precomputed_kpss is not None:
                    self.precomputed_kpss = [
                        k.copy() if k is not None else None
                        for k in self.precomputed_kpss
                    ]
                    for k in self.precomputed_kpss:
                        if k is not None:
                            k[:, 0] *= ratio_w
                            k[:, 1] *= ratio_h

        # Manual Rotation
        if control["ManualRotationEnableToggle"]:
            img = v2.functional.rotate(
                img,
                angle=control["ManualRotationAngleSlider"],
                interpolation=v2.InterpolationMode.BILINEAR,
                expand=True,
            )

        # --- DETECTION PHASE ---
        # The workers are now "Stateless Render Engines". They no longer track time or state.
        # They consume perfectly sequenced and EMA-smoothed detections from the Feeder thread.

        if self.precomputed_bboxes is not None and self.precomputed_kpss_5 is not None:
            # 1. Primary Path (Video/Webcam): Use the sequentially precomputed detections
            bboxes = self.precomputed_bboxes
            kpss_5 = self.precomputed_kpss_5
            kpss = self.precomputed_kpss
        else:
            # 2. Fallback Path (Single-Frame Mode / Static Images)
            use_landmark, landmark_mode, from_points = (
                control.get("LandmarkDetectToggle", True),
                control.get("LandmarkDetectModelSelection", "203"),
                control.get("DetectFromPointsToggle", False),
            )
            if edit_button_is_checked_global:
                use_landmark, landmark_mode, from_points = True, "203", True

            # Run detection strictly for this single image
            bboxes, kpss_5, kpss = self.models_processor.run_detect(
                img,
                control.get("DetectorModelSelection", "RetinaFace"),
                max_num=control.get("MaxFacesToDetectSlider", 1),
                score=control.get("DetectorScoreSlider", 50) / 100.0,
                input_size=(512, 512),
                use_landmark_detection=use_landmark,
                landmark_detect_mode=landmark_mode,
                landmark_score=control.get("LandmarkDetectScoreSlider", 50) / 100.0,
                from_points=from_points,
                rotation_angles=[0]
                if not control.get("AutoRotationToggle", False)
                else [0, 90, 180, 270],
                use_mean_eyes=control.get("LandmarkMeanEyesToggle", False),
                previous_detections=None,  # No historical tracking for single images
                bypass_bytetrack=True,  # Bypassing ByteTrack to avoid corrupting the global Kalman filter state
            )

        if (
            isinstance(kpss_5, np.ndarray)
            and kpss_5.shape[0] > 0
            or isinstance(kpss_5, list)
            and len(kpss_5) > 0
        ):
            for i in range(len(kpss_5)):
                _bbox_i = bboxes[i]
                if not is_detected_face_eligible_for_matching(
                    kpss_5[i], _bbox_i, self._MIN_FACE_PIXELS
                ):
                    continue  # too small to produce meaningful swap
                face_emb, _ = self.models_processor.run_recognize_direct(
                    img,
                    kpss_5[i],
                    control["SimilarityTypeSelection"],
                    control["RecognitionModelSelection"],
                )
                # FW-BUG-01: bounds check before indexing kpss
                kps_all_i = kpss[i] if kpss is not None and i < len(kpss) else None
                det_faces_data_for_display.append(
                    {
                        "kps_5": kpss_5[i],
                        "kps_all": kps_all_i,
                        "embedding": face_emb,
                        "bbox": bboxes[i],
                        "original_face": None,
                        "swap_mask": None,
                        "matched_target": None,  # FW-BUG-09: cache slot
                    }
                )

        # Swapping / Editing Loop
        if det_faces_data_for_display:
            if control["SwapOnlyBestMatchEnableToggle"]:
                # --- Branch: Swap Only Best Match ---
                # FW-RACE-01: iterate the snapshot, not the live dict
                for _, target_face in target_faces_snapshot.items():
                    if stop_event.is_set():
                        break

                    # FW-ROBUST-04: use .get() with default_parameters as fallback
                    with self.lock:
                        _default_params = dict(self.main_window.default_parameters.data)
                    # FIX: Force face_id as string to prevent silent parameter fallback
                    face_id_str = str(target_face.face_id)
                    params = ParametersDict(
                        self.parameters.get(face_id_str, _default_params),
                        _default_params,
                    )
                    best_fface, best_score = None, -1.0

                    for fface in det_faces_data_for_display:
                        # FW-BUG-09: pass snapshot; cache result on fface for downstream reuse
                        tgt, tgt_params, score = self._find_best_target_match(
                            fface["embedding"], control, target_faces_snapshot
                        )
                        fface["matched_target"] = tgt
                        if tgt and tgt.face_id == target_face.face_id:
                            if (
                                score >= tgt_params["SimilarityThresholdSlider"]
                                and score > best_score
                            ):
                                best_score = score
                                best_fface = fface

                    if best_fface is not None and (
                        swap_button_is_checked_global or edit_button_is_checked_global
                    ):
                        denoiser_on = (
                            control.get(
                                "DenoiserUNetEnableBeforeRestorersToggle", False
                            )
                            or control.get("DenoiserAfterFirstRestorerToggle", False)
                            or control.get("DenoiserAfterRestorersToggle", False)
                        )
                        if (
                            denoiser_on
                            and target_face.assigned_kv_map is None
                            and target_face.assigned_input_faces
                        ):
                            with self.models_processor.model_lock:
                                if (
                                    target_face.assigned_kv_map is None
                                ):  # Double Check inside lock
                                    target_face.calculate_assigned_input_embedding()

                        # --- MORPHING: Swap Only Best Match ---
                        source_kps = None
                        if target_face and target_face.assigned_input_faces:
                            first_input_id = list(
                                target_face.assigned_input_faces.keys()
                            )[0]
                            store = target_face.assigned_input_faces[first_input_id]
                            if "kps_5" in store:
                                source_kps = store["kps_5"]

                        best_fface["kps_5"] = keypoints_adjustments(
                            best_fface["kps_5"],
                            cast(dict, params),
                            source_kps=source_kps,  # type: ignore[arg-type]
                        )

                        s_e = None
                        arcface_model = self.models_processor.get_arcface_model(
                            params["SwapModelSelection"]
                        )
                        if (
                            swap_button_is_checked_global
                            and params["SwapModelSelection"] != "DeepFaceLive (DFM)"
                        ):
                            _reaging_on = params.get("FaceReagingEnableToggle", False)
                            _aged_emb = getattr(target_face, "aged_input_embedding", {})
                            if _reaging_on and _aged_emb:
                                s_e = _aged_emb.get(arcface_model)
                            else:
                                s_e = target_face.assigned_input_embedding.get(
                                    arcface_model
                                )
                            if s_e is not None and np.isnan(s_e).any():
                                s_e = None

                        _aged_kv = getattr(target_face, "aged_kv_map", None)
                        _reaging_kv = (
                            _aged_kv
                            if params.get("FaceReagingEnableToggle", False)
                            and _aged_kv is not None
                            else target_face.assigned_kv_map
                        )
                        _params_for_swap_a = self._apply_auto_mouth(
                            cast(dict, params),
                            target_face,
                            face_bbox=best_fface["bbox"],
                        )
                        try:
                            (
                                img,
                                best_fface["original_face"],
                                best_fface["swap_mask"],
                            ) = self.swap_core(
                                img,
                                best_fface["kps_5"],
                                best_fface["kps_all"],
                                s_e=s_e,
                                t_e=target_face.get_embedding(arcface_model),
                                parameters=_params_for_swap_a,
                                control=control,
                                dfm_model_name=params["DFMModelSelection"],
                                kv_map=_reaging_kv,
                            )
                            if edit_button_is_checked_global and any(
                                params[f]
                                for f in (
                                    "FaceMakeupEnableToggle",
                                    "HairMakeupEnableToggle",
                                    "EyeBrowsMakeupEnableToggle",
                                    "LipsMakeupEnableToggle",
                                )
                            ):
                                img = self.frame_edits.swap_edit_face_core_makeup(
                                    img, best_fface["kps_all"], params.data, control
                                )
                        except Exception as e:
                            print(
                                f"[ERROR] Standard mode swap_core failed for best face: {e}"
                            )
                            continue  # Ignore this face but save the frame

            else:
                # --- Branch: Swap All Matches ---
                for fface in det_faces_data_for_display:
                    if stop_event.is_set():
                        break
                    # FW-BUG-09: pass the target_faces snapshot to avoid re-iterating live dict
                    best_target, params, _ = self._find_best_target_match(
                        fface["embedding"], control, target_faces_snapshot
                    )
                    # FW-BUG-09: cache matched target so downstream helpers can reuse it
                    fface["matched_target"] = best_target

                    if best_target and (
                        swap_button_is_checked_global or edit_button_is_checked_global
                    ):
                        denoiser_on = (
                            control.get(
                                "DenoiserUNetEnableBeforeRestorersToggle", False
                            )
                            or control.get("DenoiserAfterFirstRestorerToggle", False)
                            or control.get("DenoiserAfterRestorersToggle", False)
                        )
                        if (
                            denoiser_on
                            and best_target.assigned_kv_map is None
                            and best_target.assigned_input_faces
                        ):
                            with self.models_processor.model_lock:
                                if (
                                    best_target.assigned_kv_map is None
                                ):  # Double Check inside lock
                                    best_target.calculate_assigned_input_embedding()

                        # --- MORPHING: Branch Swap All Matches ---
                        source_kps = None
                        if best_target and best_target.assigned_input_faces:
                            first_input_id = list(
                                best_target.assigned_input_faces.keys()
                            )[0]
                            store = best_target.assigned_input_faces[first_input_id]
                            if "kps_5" in store:
                                source_kps = store["kps_5"]

                        fface["kps_5"] = keypoints_adjustments(
                            fface["kps_5"], params, source_kps=source_kps
                        )

                        arcface_model = self.models_processor.get_arcface_model(
                            params["SwapModelSelection"]
                        )
                        s_e = None
                        if (
                            swap_button_is_checked_global
                            and params["SwapModelSelection"] != "DeepFaceLive (DFM)"
                        ):
                            _reaging_on = params.get("FaceReagingEnableToggle", False)
                            _aged_emb_bt = getattr(
                                best_target, "aged_input_embedding", {}
                            )
                            if _reaging_on and _aged_emb_bt:
                                s_e = _aged_emb_bt.get(arcface_model)
                            else:
                                s_e = best_target.assigned_input_embedding.get(
                                    arcface_model
                                )
                            if s_e is not None and np.isnan(s_e).any():
                                s_e = None

                        _aged_kv_bt = getattr(best_target, "aged_kv_map", None)
                        _reaging_kv = (
                            _aged_kv_bt
                            if params.get("FaceReagingEnableToggle", False)
                            and _aged_kv_bt is not None
                            else best_target.assigned_kv_map
                        )
                        _params_for_swap_b = self._apply_auto_mouth(
                            cast(dict, params),
                            best_target,
                            face_bbox=fface["bbox"],
                        )
                        try:
                            img, fface["original_face"], fface["swap_mask"] = (
                                self.swap_core(
                                    img,
                                    fface["kps_5"],
                                    fface["kps_all"],
                                    s_e=s_e,
                                    t_e=best_target.get_embedding(arcface_model),
                                    parameters=_params_for_swap_b,
                                    control=control,
                                    dfm_model_name=params["DFMModelSelection"],
                                    kv_map=_reaging_kv,
                                )
                            )
                            if edit_button_is_checked_global and any(
                                params[f]
                                for f in (
                                    "FaceMakeupEnableToggle",
                                    "HairMakeupEnableToggle",
                                    "EyeBrowsMakeupEnableToggle",
                                    "LipsMakeupEnableToggle",
                                )
                            ):
                                img = self.frame_edits.swap_edit_face_core_makeup(
                                    img, fface["kps_all"], params.data, control
                                )
                        except Exception as e:
                            print(
                                f"[ERROR] Standard mode swap_core failed for a face: {e}"
                            )
                            continue

        # Undo Rotation / Scaling
        if control["ManualRotationEnableToggle"]:
            img = v2.functional.rotate(
                img,
                angle=-control["ManualRotationAngleSlider"],
                interpolation=v2.InterpolationMode.BILINEAR,
                expand=True,
            )
        if scale_applied:
            # FW-QUAL-11: use renamed img_h/img_w variables
            # FW-PERF-08 / FW-MEM-02: LRU-bounded cache for the scale-back transform.
            _down_key = (img_h, img_w, self.interpolation_scaleback, False)
            if _down_key in self._resize_cache:
                self._resize_cache.move_to_end(_down_key)
            else:
                if len(self._resize_cache) >= self._RESIZE_CACHE_MAX:
                    self._resize_cache.popitem(last=False)
                self._resize_cache[_down_key] = v2.Resize(
                    (img_h, img_w),
                    interpolation=self.interpolation_scaleback,
                    antialias=False,
                )
            img = self._resize_cache[_down_key](img)

        processed_tensor_rgb_uint8 = img

        # --- Overlays ---
        if control["ShowAllDetectedFacesBBoxToggle"] and det_faces_data_for_display:
            processed_tensor_rgb_uint8 = draw_bounding_boxes_on_detected_faces(
                processed_tensor_rgb_uint8, det_faces_data_for_display
            )

        if control.get("ShowByteTrackBBoxToggle", False) and det_faces_data_for_display:
            processed_tensor_rgb_uint8 = draw_bounding_boxes_on_detected_faces(
                processed_tensor_rgb_uint8,
                det_faces_data_for_display,
                color_rgb=[255, 165, 0],
            )

        if control["ShowLandmarksEnableToggle"] and det_faces_data_for_display:
            landmarks_data = self._resolve_landmarks_to_draw(
                det_faces_data_for_display, control
            )
            if landmarks_data:
                temp_permuted = processed_tensor_rgb_uint8.permute(1, 2, 0)
                temp_permuted = paint_landmarks_on_image(temp_permuted, landmarks_data)
                processed_tensor_rgb_uint8 = temp_permuted.permute(2, 0, 1)

        compare_mode_active = self.is_view_face_mask or self.is_view_face_compare
        if compare_mode_active and det_faces_data_for_display:
            processed_tensor_rgb_uint8 = self.get_compare_faces_image(
                processed_tensor_rgb_uint8, det_faces_data_for_display, control
            )

        return processed_tensor_rgb_uint8

    def _resolve_landmarks_to_draw(self, det_faces_data: list, control: dict) -> list:
        """
        Helper to determine which landmarks to draw and in what color based on matches.
        """
        landmarks_to_draw = []
        for fface_data in det_faces_data:
            # FW-BUG-09: use cached matched_target if available to avoid re-running match
            cached_tgt = fface_data.get("matched_target")
            if cached_tgt is not None:
                with self.lock:
                    _default_params = dict(self.main_window.default_parameters.data)
                _face_id_1 = getattr(cached_tgt, "face_id", None)
                face_specific: dict = cast(
                    dict,
                    self.parameters.get(_face_id_1, {})  # type: ignore[arg-type]
                    if _face_id_1 is not None
                    else {},
                )
                matched_params = ParametersDict(dict(face_specific), _default_params)
            else:
                _, matched_params, _ = self._find_best_target_match(
                    fface_data["embedding"], control
                )
            if matched_params:
                use_adj = matched_params["LandmarksPositionAdjEnableToggle"]
                keypoints = (
                    fface_data.get("kps_5") if use_adj else fface_data.get("kps_all")
                )
                kcolor = (255, 0, 0) if use_adj else (0, 255, 255)

                if keypoints is not None:
                    landmarks_to_draw.append({"kps": keypoints, "color": kcolor})

        return landmarks_to_draw

    def get_compare_faces_image(
        self, img: torch.Tensor, det_faces_data: list, control: dict
    ) -> torch.Tensor:
        """
        Builds a side-by-side comparison image for all detected faces.

        For each detected face that has a matched target, creates a horizontal strip
        containing: [original_face | swapped_face | swap_mask (if available)].
        All strips are vertically stacked and returned as a single CHW tensor.
        Returns the original *img* unchanged if no matched faces are found.
        """
        imgs_to_vstack = []
        for _, fface in enumerate(det_faces_data):
            # FW-BUG-09 / FW-PERF-01/02/03: use cached match result when available
            cached_tgt = fface.get("matched_target")
            if cached_tgt is not None:
                with self.lock:
                    _default_params = dict(self.main_window.default_parameters.data)
                _face_id_2 = getattr(cached_tgt, "face_id", None)
                face_specific: dict = cast(
                    dict,
                    self.parameters.get(_face_id_2, {})  # type: ignore[arg-type]
                    if _face_id_2 is not None
                    else {},
                )
                parameters_for_face = ParametersDict(
                    dict(face_specific), _default_params
                )
                best_target_for_compare = cached_tgt
            else:
                best_target_for_compare, parameters_for_face, _ = (
                    self._find_best_target_match(fface["embedding"], control)
                )
            if best_target_for_compare and parameters_for_face:
                modified_face = self.get_cropped_face_using_kps(
                    img, fface["kps_5"], cast(dict, parameters_for_face)
                )
                # FW-PERF-03: skip enhance_core in compare view — diagnostic only,
                # enhancement is too expensive to run twice per face.
                imgs_to_cat_horizontally = []
                original_face_from_swap_core = fface.get("original_face")
                if original_face_from_swap_core is not None:
                    imgs_to_cat_horizontally.append(
                        original_face_from_swap_core.permute(2, 0, 1)
                    )
                imgs_to_cat_horizontally.append(modified_face)
                swap_mask_from_swap_core = fface.get("swap_mask")
                if swap_mask_from_swap_core is not None:
                    mask_chw = swap_mask_from_swap_core.permute(2, 0, 1)
                    if mask_chw.shape[0] == 1:
                        mask_chw = mask_chw.repeat(3, 1, 1)
                    imgs_to_cat_horizontally.append(mask_chw)
                if imgs_to_cat_horizontally:
                    min_h = min(t.shape[1] for t in imgs_to_cat_horizontally)
                    resized_imgs_to_cat = []
                    for t_img in imgs_to_cat_horizontally:
                        if t_img.shape[1] != min_h:
                            aspect_ratio = t_img.shape[2] / t_img.shape[1]
                            new_w = (
                                int(min_h * aspect_ratio)
                                if aspect_ratio > 0
                                else t_img.shape[2]
                            )
                            resized_imgs_to_cat.append(
                                v2.Resize((min_h, new_w), antialias=True)(t_img)
                            )
                        else:
                            resized_imgs_to_cat.append(t_img)
                    imgs_to_vstack.append(torch.cat(resized_imgs_to_cat, dim=2))

        if imgs_to_vstack:
            max_width_for_vstack = max(
                img_strip.size(2) for img_strip in imgs_to_vstack
            )
            padded_strips_for_vstack = [
                torch.nn.functional.pad(
                    img_strip, (0, max_width_for_vstack - img_strip.size(2), 0, 0)
                )
                for img_strip in imgs_to_vstack
            ]
            return torch.cat(padded_strips_for_vstack, dim=1)
        return img

    def get_cropped_face_using_kps(
        self,
        img: torch.Tensor,
        kps_5: np.ndarray,
        parameters: dict,
        interp_mode: str = "bilinear",
    ) -> torch.Tensor:
        """
        Aligns and crops a 512×512 face patch from *img* using the 5-point keypoints.
        OPTIMIZED: Uses Kornia to warp directly on the GPU, avoiding CPU decomposition.

        Args:
            img:         Full-frame CHW uint8 tensor.
            kps_5:       5-point facial keypoints (numpy array, shape [5, 2]).
            parameters:  Per-face parameter dict containing at least ``SwapModelSelection``.
            interp_mode: Interpolation mode for warp_affine (e.g. "bilinear" or "bicubic").

        Returns:
            CHW uint8 tensor of shape [3, 512, 512].
        """
        tform = self.get_face_similarity_tform(parameters["SwapModelSelection"], kps_5)

        M_tensor = (
            torch.from_numpy(cast(np.ndarray, tform.params)[0:2])
            .float()
            .unsqueeze(0)
            .to(img.device)
        )

        # Cast to float32 for Kornia
        img_b = img.unsqueeze(0) if img.dim() == 3 else img
        img_b_float = img_b.float()

        face_512_aligned = kgm.warp_affine(
            img_b_float,
            M_tensor,
            dsize=(512, 512),
            mode=interp_mode,
            align_corners=True,
        ).squeeze(0)

        # Convert back to original dtype (uint8)
        return face_512_aligned.to(img.dtype)

    def get_face_similarity_tform(
        self, swapper_model: str, kps_5: np.ndarray
    ) -> trans.SimilarityTransform:
        """
        Computes the similarity transform that maps the detected 5-point keypoints
        to the canonical face template for the given *swapper_model*.

        Different swapper architectures use different alignment templates
        (ArcFace 128 crop, ArcFace map crop, or FFHQ-aligned crop for CSCS/Ghost).

        Args:
            swapper_model: The name of the active swapper (e.g. ``"Inswapper128"``).
            kps_5:         Detected 5-point keypoints, shape [5, 2].

        Returns:
            Fitted ``skimage.transform.SimilarityTransform``.

        Raises:
            ValueError: If the transform estimation fails (degenerate face geometry).
        """
        # FW-QUAL-10: use GHOSTFACE_MODELS frozenset instead of chained != comparisons
        if swapper_model not in self.GHOSTFACE_MODELS and swapper_model != "CSCS":
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
            # FW-QUAL-10: swapper_model in GHOSTFACE_MODELS
            tform = trans.SimilarityTransform()
            dst = faceutil.get_arcface_template(image_size=512, mode="arcfacemap")
            M, _ = faceutil.estimate_norm_arcface_template(kps_5, src=dst)
            if M is None or np.any(np.isnan(M)) or np.any(np.isinf(M)):
                raise ValueError(
                    "GhostFace transform estimation failed (degenerate face geometry)"
                )
            tform.params[0:2] = M
        return tform

    def get_transformed_and_scaled_faces(
        self, tform, img, interp_mode: str = "bilinear"
    ):
        """
        Applies the similarity transform to extract aligned face crops at four resolutions.
        OPTIMIZED: GPU warping directly from transformation matrix using Kornia.

        Args:
            tform:       Fitted ``SimilarityTransform`` from ``get_face_similarity_tform``.
            img:         Full-frame CHW uint8 tensor.
            interp_mode: Interpolation mode for warp_affine (e.g. "bilinear" or "bicubic").

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

        assert self.t384 is not None, (
            "t384 must be initialized via set_scaling_transforms"
        )
        assert self.t256 is not None, (
            "t256 must be initialized via set_scaling_transforms"
        )
        assert self.t128 is not None, (
            "t128 must be initialized via set_scaling_transforms"
        )
        original_face_384 = self.t384(original_face_512)
        original_face_256 = self.t256(original_face_512)
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

    @staticmethod
    def _apply_likeness(
        source_latent: torch.Tensor, target_latent: torch.Tensor, params: dict
    ) -> torch.Tensor:
        """FW-QUAL-09: Identity Boost (Face Likeness) via SLERP / LERP on ArcFace embeddings.

        Promoted from inner function to @staticmethod so it is reusable and not
        re-created on every call to get_affined_face_dim_and_swapping_latents.
        """
        if not params.get("FaceLikenessEnableToggle", False):
            return source_latent

        factor = float(params.get("FaceLikenessFactorDecimalSlider", 0.0))
        if factor == 0.0:
            return source_latent

        # 1. Capture original energy (Norms are generally constant in ArcFace)
        s_norm = torch.norm(source_latent)
        t_norm = torch.norm(target_latent)

        if s_norm < 1e-6 or t_norm < 1e-6:
            return source_latent

        # 2. Normalize to get directional vectors on the hypersphere
        s_dir = source_latent / s_norm
        t_dir = target_latent / t_norm

        if factor < 0.0:
            # --- INTERPOLATION (SLERP) ---
            # Move naturally towards the target face along the sphere
            t = 1.0 + factor

            cos_theta = torch.sum(s_dir * t_dir)
            cos_theta = torch.clamp(cos_theta, -0.9999, 0.9999)
            theta = torch.acos(cos_theta)
            sin_theta = torch.sin(theta)

            if sin_theta < 1e-3:
                blended_dir = (1.0 - t) * t_dir + t * s_dir
            else:
                weight_t = torch.sin((1.0 - t) * theta) / sin_theta
                weight_s = torch.sin(t * theta) / sin_theta
                blended_dir = weight_t * t_dir + weight_s * s_dir
        else:
            # --- EXTRAPOLATION (LERP) ---
            # Push the vector away from the target to exaggerate the source identity
            difference_vector = s_dir - t_dir
            blended_dir = s_dir + (factor * difference_vector)

        # 3. Always restore original Source Energy to prevent latent space corruption
        blended_dir = blended_dir / torch.norm(blended_dir)
        final_latent = blended_dir * s_norm

        return final_latent

    def get_affined_face_dim_and_swapping_latents(
        self,
        original_faces: tuple,
        swapper_model,
        dfm_model_name,
        s_e,
        t_e,
        parameters,
        cmddebug,
        tform,
    ):
        """
        Selects the correct input face resolution and computes the swapping latent vector
        for the active swapper model.

        Args:
            original_faces: Tuple ``(face_512, face_384, face_256, face_128)`` of CHW tensors.
            swapper_model:  Active swapper name (e.g. ``"Inswapper128"``).
            dfm_model_name: DFM model filename; used only when swapper_model is ``"DeepFaceLive (DFM)"``.
            s_e:            Source ArcFace embedding (numpy array) or ``None`` for DFM.
            t_e:            Target ArcFace embedding (numpy array) or ``None``.
            parameters:     Per-face parameter dict.
            cmddebug:       Whether command-line debug output is enabled.
            tform:          Similarity transform (used by Inswapper auto-resolution).

        Returns:
            Tuple ``(input_face_affined, dfm_model_instance, dim, latent)`` where
            *input_face_affined* is ``None`` on failure, *dim* is the resolution multiplier
            (1=128, 2=256, 3=384, 4=512), and *latent* is the model-specific embedding tensor.
        """
        original_face_512, original_face_384, original_face_256, original_face_128 = (
            original_faces
        )

        dfm_model_instance = None
        input_face_affined = None
        dim = 1
        latent = None

        # FW-QUAL-09: apply_likeness_with_norm_preservation promoted to @staticmethod.
        # Use self._apply_likeness(...) everywhere below.

        # --- Inswapper128 Logic ---
        if swapper_model == "Inswapper128":
            # FS-ROBUST-01: calc_inswapper_latent may return None on emap failure
            _s_latent_np = self.models_processor.calc_inswapper_latent(s_e)
            _t_latent_np = self.models_processor.calc_inswapper_latent(t_e)
            if _s_latent_np is None or _t_latent_np is None:
                print(
                    "[ERROR] calc_inswapper_latent returned None (emap unavailable). Skipping swap."
                )
                return input_face_affined, dfm_model_instance, dim, latent
            latent = (
                torch.from_numpy(_s_latent_np).float().to(self.models_processor.device)
            )
            dst_latent = (
                torch.from_numpy(_t_latent_np).float().to(self.models_processor.device)
            )

            latent = self._apply_likeness(latent, dst_latent, parameters)

            dim = 1
            if parameters["SwapperResAutoSelectEnableToggle"]:
                if tform.scale <= 1.00:
                    dim = 4
                    input_face_affined = original_face_512
                elif tform.scale <= 1.75:
                    dim = 3
                    input_face_affined = original_face_384
                elif tform.scale <= 2:
                    dim = 2
                    input_face_affined = original_face_256
                else:
                    dim = 1
                    input_face_affined = original_face_128
            else:
                if parameters["SwapperResSelection"] == "128":
                    dim = 1
                    input_face_affined = original_face_128
                elif parameters["SwapperResSelection"] == "256":
                    dim = 2
                    input_face_affined = original_face_256
                elif parameters["SwapperResSelection"] == "384":
                    dim = 3
                    input_face_affined = original_face_384
                elif parameters["SwapperResSelection"] == "512":
                    dim = 4
                    input_face_affined = original_face_512

        # --- InStyleSwapper Logic ---
        elif swapper_model in (
            "InStyleSwapper256 Version A",
            "InStyleSwapper256 Version B",
            "InStyleSwapper256 Version C",
        ):
            version = swapper_model[-1]
            latent = (
                torch.from_numpy(
                    self.models_processor.calc_swapper_latent_iss(s_e, version)
                )
                .float()
                .to(self.models_processor.device)
            )
            dst_latent = (
                torch.from_numpy(
                    self.models_processor.calc_swapper_latent_iss(t_e, version)
                )
                .float()
                .to(self.models_processor.device)
            )

            latent = self._apply_likeness(latent, dst_latent, parameters)

            if (
                (
                    parameters["SwapModelSelection"] == "InStyleSwapper256 Version A"
                    and parameters["InStyleResAEnableToggle"]
                )
                or (
                    parameters["SwapModelSelection"] == "InStyleSwapper256 Version B"
                    and parameters["InStyleResBEnableToggle"]
                )
                or (
                    parameters["SwapModelSelection"] == "InStyleSwapper256 Version C"
                    and parameters["InStyleResCEnableToggle"]
                )
            ):
                dim = 4
                input_face_affined = original_face_512
            else:
                dim = 2
                input_face_affined = original_face_256

        # --- SimSwap Logic ---
        elif swapper_model == "SimSwap512":
            latent = (
                torch.from_numpy(
                    self.models_processor.calc_swapper_latent_simswap512(s_e)
                )
                .float()
                .to(self.models_processor.device)
            )
            dst_latent = (
                torch.from_numpy(
                    self.models_processor.calc_swapper_latent_simswap512(t_e)
                )
                .float()
                .to(self.models_processor.device)
            )

            latent = self._apply_likeness(latent, dst_latent, parameters)

            dim = 4
            input_face_affined = original_face_512

        # --- GhostFace Logic ---
        # FW-QUAL-10: use GHOSTFACE_MODELS frozenset
        elif swapper_model in self.GHOSTFACE_MODELS:
            latent = (
                torch.from_numpy(self.models_processor.calc_swapper_latent_ghost(s_e))
                .float()
                .to(self.models_processor.device)
            )
            dst_latent = (
                torch.from_numpy(self.models_processor.calc_swapper_latent_ghost(t_e))
                .float()
                .to(self.models_processor.device)
            )

            latent = self._apply_likeness(latent, dst_latent, parameters)

            dim = 2
            input_face_affined = original_face_256

        # --- CSCS Logic ---
        elif swapper_model == "CSCS":
            latent = (
                torch.from_numpy(self.models_processor.calc_swapper_latent_cscs(s_e))
                .float()
                .to(self.models_processor.device)
            )
            dst_latent = (
                torch.from_numpy(self.models_processor.calc_swapper_latent_cscs(t_e))
                .float()
                .to(self.models_processor.device)
            )

            latent = self._apply_likeness(latent, dst_latent, parameters)

            dim = 2
            input_face_affined = original_face_256

        # --- DFM Logic ---
        elif swapper_model == "DeepFaceLive (DFM)" and dfm_model_name:
            # Explicitly notify FaceSwappers to unload the previous ONNX model
            # This prevents VRAM leaks when switching to heavy DFM models.
            self.models_processor.face_swappers._manage_model("DeepFaceLive (DFM)")
            self.models_processor.face_swappers.current_swapper_model = (
                "DeepFaceLive (DFM)"
            )

            dfm_model_instance = self.models_processor.load_dfm_model(dfm_model_name)

            # FIX: Attach the filename to the instance so we can robustly extract its resolution later
            if dfm_model_instance is not None:
                dfm_model_instance._dfm_filename_fallback = dfm_model_name

            latent = []
            input_face_affined = original_face_512
            dim = 4

        return input_face_affined, dfm_model_instance, dim, latent

    def _fix_drift_and_texture(
        self,
        current_face: torch.Tensor,
        prev_face: torch.Tensor,
        first_face: torch.Tensor,
        blend_texture: bool = True,
    ) -> torch.Tensor:
        """
        Corrects spatial drift via Phase Correlation and restores skin textures
        (high frequency) to prevent the plastic effect from multiple iterations.
        Includes an improved LAB-space Adaptive Instance Normalization (AdaIN)
        to lock color and contrast without destroying white balance.
        Expected tensors in [C, H, W] format as Floats (0.0 - 1.0).
        """
        device = current_face.device

        # --- 1. ANTI-DRIFT (Phase Correlation FFT) ---
        anchor_face = first_face if first_face is not None else prev_face

        gray_curr = current_face.mean(dim=0, keepdim=True)
        gray_anchor = anchor_face.mean(dim=0, keepdim=True)

        H, W = gray_curr.shape[1], gray_curr.shape[2]

        gray_curr_blurred = self._DRIFT_TEXTURE_BLUR(gray_curr)
        gray_anchor_blurred = self._DRIFT_TEXTURE_BLUR(gray_anchor)

        window_y = torch.hann_window(H, device=device).view(H, 1)
        window_x = torch.hann_window(W, device=device).view(1, W)
        window = window_y * window_x

        G_curr = torch.fft.fft2(gray_curr_blurred * window)
        G_anchor = torch.fft.fft2(gray_anchor_blurred * window)

        R = G_anchor * torch.conj(G_curr)
        R = R / (torch.abs(R) + 1e-8)

        r = torch.fft.ifft2(R).real
        r = torch.fft.fftshift(r)

        max_idx = r.argmax()
        dy = (max_idx // W).item() - H // 2
        dx = (max_idx % W).item() - W // 2

        max_drift = max(2, int(H * 0.015))

        if abs(dy) <= 1 and abs(dx) <= 1:
            dy, dx = 0, 0
        elif abs(dy) > max_drift or abs(dx) > max_drift:
            dy, dx = 0, 0

        if dy != 0 or dx != 0:
            current_face = torch.roll(current_face, shifts=(dy, dx), dims=(1, 2))

            if dy > 0:
                current_face[:, :dy, :] = anchor_face[:, :dy, :]
            elif dy < 0:
                current_face[:, dy:, :] = anchor_face[:, dy:, :]
            if dx > 0:
                current_face[:, :, :dx] = anchor_face[:, :, :dx]
            elif dx < 0:
                current_face[:, :, dx:] = anchor_face[:, :, dx:]

        # --- 2. TEXTURE & COLOR PRESERVATION ---
        if blend_texture and first_face is not None:
            # --- Improved Color & Luminance Lock (LAB AdaIN) ---
            # Operating in LAB color space prevents the destruction of white balance
            # and contrast that happens when normalizing RGB channels independently.

            # Convert to LAB (requires [B, C, H, W] shape)
            curr_bchw = current_face.unsqueeze(0).clamp(0.0, 1.0)
            first_bchw = first_face.unsqueeze(0).clamp(0.0, 1.0)

            curr_lab = kc.rgb_to_lab(curr_bchw)
            first_lab = kc.rgb_to_lab(first_bchw)

            # Calculate Mean and Std for each LAB channel independently
            mean_curr = curr_lab.mean(dim=(2, 3), keepdim=True)
            std_curr = curr_lab.std(dim=(2, 3), keepdim=True) + 1e-6

            mean_first = first_lab.mean(dim=(2, 3), keepdim=True)
            std_first = first_lab.std(dim=(2, 3), keepdim=True) + 1e-6

            # Apply Adaptive Instance Normalization
            matched_lab = (curr_lab - mean_curr) * (std_first / std_curr) + mean_first

            # Convert back to RGB and remove batch dimension
            current_face = kc.lab_to_rgb(matched_lab).squeeze(0).clamp(0.0, 1.0)

            # --- Frequency Separation (Micro-details) ---
            k = max(3, (H // 32) * 2 + 1)
            sigma = k * 0.3
            blur_tex = v2.GaussianBlur(kernel_size=k, sigma=sigma)

            low_pass_first = blur_tex(first_face)
            high_pass_first = first_face - low_pass_first

            # Dampen the high frequencies slightly to avoid over-sharpening noise
            high_pass_first = torch.clamp(high_pass_first, -0.3, 0.3) * 0.75

            low_pass_curr = blur_tex(current_face)

            # Reconstruct the image: New structural shape (low freq) + Original skin texture (high freq)
            current_face = low_pass_curr + high_pass_first

        return torch.clamp(current_face, 0.0, 1.0)

    def get_swapped_and_prev_face(
        self,
        output,
        input_face_affined,
        original_face_512,
        latent,
        itex,
        dim,
        swapper_model,
        dfm_model,
        parameters,
    ):
        """
        Runs the swapper model inference and returns the swapped face tensor.

        Applies optional pre-swap sharpness, executes the swapper loop *itex* times
        (strength slider), and delegates to the architecture-specific branch
        (Inswapper, Ghost, SimSwap, InStyle, CSCS, or DFM).

        Args:
            output:             Pre-allocated output tensor (HWC float32, [0..1]).
            input_face_affined: Aligned face CHW tensor at the chosen resolution.
            original_face_512:  Unmodified 512-px face CHW uint8 tensor (used by DFM).
            latent:             Swapping latent computed by ``get_affined_face_dim_and_swapping_latents``.
            itex:               Number of inference iterations (from StrengthAmountSlider).
            dim:                Resolution multiplier (1=128, 2=256, 3=384, 4=512).
            swapper_model:      Active swapper name.
            dfm_model:          Loaded ``DFMModel`` instance, or ``None`` for non-DFM swappers.
            parameters:         Per-face parameter dict.

        Returns:
            Tuple ``(swap_chw_uint8, prev_face_hwc_float)``.
        """
        if parameters["PreSwapSharpnessDecimalSlider"] != 1.0:
            input_face_affined = input_face_affined.permute(2, 0, 1)
            input_face_affined = v2.functional.adjust_sharpness(
                input_face_affined, parameters["PreSwapSharpnessDecimalSlider"]
            )
            input_face_affined = input_face_affined.permute(1, 2, 0)

        # prev_face is updated at the start of each iteration so that after
        # N iterations it holds the N-1 result.  The alpha blend in swap_core
        # then interpolates between pass N-1 and pass N for fractional slider values.
        # Initialized here as a fallback for the DFM branch and itex=0 edge case.
        prev_face = input_face_affined.clone()
        first_pass_face = None

        # Strength mode 2 toggle
        use_mode_2 = parameters.get("StrengthMode2EnableToggle", False)

        if swapper_model == "Inswapper128":
            _use_batched = False

            for k in range(itex):
                prev_face = (
                    input_face_affined.clone()
                )  # save N-1 result before this pass

                if _use_batched:
                    # ------ BATCHED PATH (Custom provider, dim > 1) ------
                    tiles_list = []
                    tile_coords = []
                    for j in range(dim):
                        for i in range(dim):
                            tiles_list.append(
                                input_face_affined[j::dim, i::dim]
                                .permute(2, 0, 1)
                                .contiguous()
                            )
                            tile_coords.append((j, i))
                    batch_input = torch.stack(tiles_list, dim=0)  # [B, 3, 128, 128]
                    batch_output = torch.empty_like(batch_input)

                    self.models_processor.run_inswapper_batched(
                        batch_input, latent, batch_output
                    )

                    if self.models_processor.device == "cuda":
                        torch.cuda.current_stream().synchronize()

                    # Replace any near-zero output tile with the input tile
                    tile_sums = batch_output.abs().sum(dim=(1, 2, 3))
                    zero_mask = tile_sums < 1.0
                    if zero_mask.any():
                        batch_output[zero_mask] = batch_input[zero_mask]

                    # --- MODE 2 ---
                    if use_mode_2:
                        temp_output = input_face_affined.clone()
                        for idx, (j, i) in enumerate(tile_coords):
                            temp_output[j::dim, i::dim] = batch_output[idx].permute(
                                1, 2, 0
                            )

                        curr_chw = temp_output.permute(2, 0, 1)
                        if k == 0:
                            first_pass_face = curr_chw.clone()
                        else:
                            prev_chw = prev_face.permute(2, 0, 1)
                            curr_chw = self._fix_drift_and_texture(
                                curr_chw, prev_chw, first_pass_face
                            )
                            temp_output = curr_chw.permute(1, 2, 0)

                        prev_face = input_face_affined.clone()
                        input_face_affined = temp_output.clone()
                        output = torch.clamp(temp_output * 255.0, 0, 255)

                    # --- NORMAL MODE ---
                    else:
                        for idx, (j, i) in enumerate(tile_coords):
                            output[j::dim, i::dim] = batch_output[idx].permute(1, 2, 0)

                        prev_face = input_face_affined.clone()
                        input_face_affined = output.clone()
                        output = torch.mul(output, 255)
                        output = torch.clamp(output, 0, 255)

                else:
                    # ------ SEQUENTIAL PATH (ORT providers or dim==1) ------
                    # Lists to hold independent memory buffers for this iteration
                    tile_inputs = []
                    tile_outputs = []
                    tile_coords = []

                    for j in range(dim):
                        for i in range(dim):
                            tile = input_face_affined[j::dim, i::dim]
                            t_in = tile.permute(2, 0, 1).contiguous().unsqueeze(0)
                            t_out = torch.empty_like(t_in)

                            tile_inputs.append(t_in)
                            tile_outputs.append(t_out)
                            tile_coords.append((j, i))

                    with torch.no_grad():
                        for idx in range(len(tile_inputs)):
                            self.models_processor.run_inswapper(
                                tile_inputs[idx], latent, tile_outputs[idx]
                            )

                    if self.models_processor.device == "cuda":
                        torch.cuda.current_stream().synchronize()

                    # --- MODE 2 ---
                    if use_mode_2:
                        temp_output = input_face_affined.clone()
                        for idx, (j, i) in enumerate(tile_coords):
                            res = (
                                tile_inputs[idx]
                                if tile_outputs[idx].sum() < 1.0
                                else tile_outputs[idx]
                            )
                            temp_output[j::dim, i::dim] = res.squeeze(0).permute(
                                1, 2, 0
                            )

                        curr_chw = temp_output.permute(2, 0, 1)
                        if k == 0:
                            first_pass_face = curr_chw.clone()
                        else:
                            prev_chw = prev_face.permute(2, 0, 1)
                            curr_chw = self._fix_drift_and_texture(
                                curr_chw, prev_chw, first_pass_face
                            )
                            temp_output = curr_chw.permute(1, 2, 0)

                        prev_face = input_face_affined.clone()
                        input_face_affined = temp_output.clone()
                        output = torch.clamp(temp_output * 255.0, 0, 255)

                    # --- NORMAL MODE ---
                    else:
                        for idx, (j, i) in enumerate(tile_coords):
                            if tile_outputs[idx].sum() < 1.0:
                                res = tile_inputs[idx]
                            else:
                                res = tile_outputs[idx]

                            res_hwc = res.squeeze(0).permute(1, 2, 0)
                            output[j::dim, i::dim] = res_hwc

                        prev_face = input_face_affined.clone()
                        input_face_affined = output.clone()
                        output = torch.mul(output, 255)
                        output = torch.clamp(output, 0, 255)

        elif swapper_model in (
            "InStyleSwapper256 Version A",
            "InStyleSwapper256 Version B",
            "InStyleSwapper256 Version C",
        ):
            version = swapper_model[-1]
            dim_res = dim // 2

            for k in range(
                itex
            ):  # FW-QUAL-06: renamed k -> _ - Fix : Restored k to prevent crash
                prev_face = (
                    input_face_affined.clone()
                )  # save N-1 result before this pass
                tile_inputs = []
                tile_outputs = []
                tile_coords = []

                for j in range(dim_res):
                    for i in range(dim_res):
                        tile = input_face_affined[j::dim_res, i::dim_res]
                        t_in = tile.permute(2, 0, 1).contiguous().unsqueeze(0)
                        t_out = torch.empty_like(t_in)

                        tile_inputs.append(t_in)
                        tile_outputs.append(t_out)
                        tile_coords.append((j, i))

                with torch.no_grad():
                    for idx in range(len(tile_inputs)):
                        self.models_processor.run_iss_swapper(
                            tile_inputs[idx], latent, tile_outputs[idx], version
                        )

                if self.models_processor.device == "cuda":
                    torch.cuda.current_stream().synchronize()

                # --- MODE 2 ---
                if use_mode_2:
                    temp_output = input_face_affined.clone()
                    for idx, (j, i) in enumerate(tile_coords):
                        res = (
                            tile_inputs[idx]
                            if tile_outputs[idx].sum() < 1.0
                            else tile_outputs[idx]
                        )
                        temp_output[j::dim_res, i::dim_res] = res.squeeze(0).permute(
                            1, 2, 0
                        )

                    curr_chw = temp_output.permute(2, 0, 1)
                    if k == 0:
                        first_pass_face = curr_chw.clone()
                    else:
                        prev_chw = prev_face.permute(2, 0, 1)
                        curr_chw = self._fix_drift_and_texture(
                            curr_chw, prev_chw, first_pass_face
                        )
                        temp_output = curr_chw.permute(1, 2, 0)

                    prev_face = input_face_affined.clone()
                    input_face_affined = temp_output.clone()
                    output = torch.clamp(temp_output * 255.0, 0, 255)

                # --- NORMAL MODE ---
                else:
                    for idx, (j, i) in enumerate(tile_coords):
                        if tile_outputs[idx].sum() < 1.0:
                            res = tile_inputs[idx]
                        else:
                            res = tile_outputs[idx]

                        res_hwc = res.squeeze(0).permute(1, 2, 0)
                        output[j::dim_res, i::dim_res] = res_hwc

                    prev_face = input_face_affined.clone()
                    input_face_affined = output.clone()
                    output = torch.mul(output, 255)
                    output = torch.clamp(output, 0, 255)

        elif swapper_model == "SimSwap512":
            for k in range(
                itex
            ):  # FW-QUAL-06: renamed k -> _ - Fix : restored k to prevent crash
                prev_face = (
                    input_face_affined.clone()
                )  # save N-1 result before this pass
                input_face_disc = input_face_affined.permute(2, 0, 1)
                input_face_disc = torch.unsqueeze(input_face_disc, 0).contiguous()
                swapper_output = torch.empty(
                    (1, 3, 512, 512),
                    dtype=torch.float32,
                    device=self.models_processor.device,
                ).contiguous()

                self.models_processor.run_swapper_simswap512(
                    input_face_disc, latent, swapper_output
                )

                if self.models_processor.device == "cuda":
                    torch.cuda.current_stream().synchronize()

                # FW-BUG-08: use abs().max() instead of sum() for zero-face heuristic
                if swapper_output.abs().max() < 1e-4:
                    swapper_output = input_face_disc

                swapper_output = torch.squeeze(swapper_output)

                # --- MODE 2 ---
                if use_mode_2:
                    if k == 0:
                        first_pass_face = swapper_output.clone()
                    else:
                        prev_chw = prev_face.permute(2, 0, 1)
                        swapper_output = self._fix_drift_and_texture(
                            swapper_output, prev_chw, first_pass_face
                        )

                    swapper_output = swapper_output.permute(1, 2, 0)
                    prev_face = input_face_affined.clone()
                    input_face_affined = swapper_output.clone()
                    output = torch.clamp(swapper_output * 255.0, 0, 255)

                # --- NORMAL MODE ---
                else:
                    swapper_output = swapper_output.permute(1, 2, 0)
                    prev_face = input_face_affined.clone()
                    input_face_affined = swapper_output.clone()

                    output = swapper_output.clone()
                    output = torch.mul(output, 255)
                    output = torch.clamp(output, 0, 255)

        # FW-QUAL-10: use GHOSTFACE_MODELS frozenset
        elif swapper_model in self.GHOSTFACE_MODELS:
            for k in range(itex):  # FW-QUAL-06: renamed k -> _
                prev_face = (
                    input_face_affined.clone()
                )  # save N-1 result before this pass
                input_face_disc = torch.mul(input_face_affined, 255.0).permute(2, 0, 1)
                input_face_disc = torch.div(input_face_disc.float(), 127.5)
                input_face_disc = torch.sub(input_face_disc, 1)
                input_face_disc = torch.unsqueeze(input_face_disc, 0).contiguous()
                swapper_output = torch.empty(
                    (1, 3, 256, 256),
                    dtype=torch.float32,
                    device=self.models_processor.device,
                ).contiguous()

                self.models_processor.run_swapper_ghostface(
                    input_face_disc, latent, swapper_output, swapper_model
                )

                if self.models_processor.device == "cuda":
                    torch.cuda.current_stream().synchronize()

                swapper_output = swapper_output[0]
                # FW-BUG-11: use abs().mean() instead of sum() for zero-output heuristic
                if swapper_output.abs().mean() < 0.01:
                    # input_face_affined is [H,W,3] in [0,1]; convert to the
                    # GhostFace [-1,1] CHW range that swapper_output uses.
                    swapper_output = input_face_affined.permute(2, 0, 1) * 2.0 - 1.0

                # --- MODE 2 ---
                if use_mode_2:
                    swapper_output = torch.add(torch.mul(swapper_output, 127.5), 127.5)
                    curr_chw = torch.div(swapper_output, 255.0)

                    if k == 0:
                        first_pass_face = curr_chw.clone()
                    else:
                        prev_chw = prev_face.permute(2, 0, 1)
                        curr_chw = self._fix_drift_and_texture(
                            curr_chw, prev_chw, first_pass_face
                        )

                    temp_output = curr_chw.permute(1, 2, 0)
                    prev_face = input_face_affined.clone()
                    input_face_affined = temp_output.clone()
                    output = torch.clamp(curr_chw.permute(1, 2, 0) * 255.0, 0, 255)

                # --- NORMAL MODE ---
                else:
                    if swapper_output.sum() < 1.0:
                        pass
                    swapper_output = swapper_output.permute(1, 2, 0)
                    swapper_output = torch.mul(swapper_output, 127.5)
                    swapper_output = torch.add(swapper_output, 127.5)

                    prev_face = input_face_affined.clone()
                    input_face_affined = swapper_output.clone()
                    input_face_affined = torch.div(input_face_affined, 255)

                    output = swapper_output.clone()
                    output = torch.clamp(output, 0, 255)

        elif swapper_model == "CSCS":
            for k in range(itex):  # FW-QUAL-06: renamed k -> _
                prev_face = (
                    input_face_affined.clone()
                )  # save N-1 result before this pass
                input_face_disc = input_face_affined.permute(2, 0, 1)
                input_face_disc = v2.functional.normalize(
                    input_face_disc, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=False
                )
                input_face_disc = torch.unsqueeze(input_face_disc, 0).contiguous()
                swapper_output = torch.empty(
                    (1, 3, 256, 256),
                    dtype=torch.float32,
                    device=self.models_processor.device,
                ).contiguous()

                self.models_processor.run_swapper_cscs(
                    input_face_disc, latent, swapper_output
                )

                if self.models_processor.device == "cuda":
                    torch.cuda.current_stream().synchronize()

                swapper_output = torch.squeeze(swapper_output)
                swapper_output = torch.add(torch.mul(swapper_output, 0.5), 0.5)

                # --- MODE 2 ---
                if use_mode_2:
                    if k == 0:
                        first_pass_face = swapper_output.clone()
                    else:
                        prev_chw = prev_face.permute(2, 0, 1)
                        swapper_output = self._fix_drift_and_texture(
                            swapper_output, prev_chw, first_pass_face
                        )

                    temp_output = swapper_output.permute(1, 2, 0)
                    prev_face = input_face_affined.clone()
                    input_face_affined = temp_output.clone()
                    output = torch.clamp(temp_output * 255.0, 0, 255)

                # --- NORMAL MODE ---
                else:
                    swapper_output = swapper_output.permute(1, 2, 0)

                    prev_face = input_face_affined.clone()
                    input_face_affined = swapper_output.clone()

                    output = swapper_output.clone()
                    output = torch.mul(output, 255)
                    output = torch.clamp(output, 0, 255)

        elif swapper_model == "DeepFaceLive (DFM)" and dfm_model:
            # --- 1. DFM Resolution Detection (Deep Introspection) ---
            dfm_res = 256  # Fallback

            from collections import deque

            queue = deque([dfm_model])
            visited = set([id(dfm_model)])
            found_res = None

            while queue and not found_res:
                current = queue.popleft()
                if hasattr(current, "get_inputs") and callable(current.get_inputs):
                    try:
                        shape = current.get_inputs()[0].shape
                        for s in shape:
                            if isinstance(s, int) and s > 32 and s % 16 == 0:
                                found_res = s
                                break
                    except Exception:
                        pass

                if not found_res and hasattr(current, "__dict__"):
                    for k, v in current.__dict__.items():
                        if id(v) not in visited and not k.startswith("__"):
                            visited.add(id(v))
                            queue.append(v)

            if found_res:
                dfm_res = found_res
            elif hasattr(dfm_model, "_dfm_filename_fallback"):
                import re

                match = re.search(
                    r"(128|192|224|256|320|384|448|512)",
                    dfm_model._dfm_filename_fallback,
                )
                if match:
                    dfm_res = int(match.group(1))

            # --- 2. Strict Resize to preserve uint8 (Fixes CUDA White Square) ---
            if dfm_res != 512:
                dfm_input = v2.functional.resize(
                    original_face_512.float(),
                    [dfm_res, dfm_res],
                    interpolation=v2.InterpolationMode.BILINEAR,
                    antialias=True,
                ).to(original_face_512.dtype)
            else:
                dfm_input = original_face_512.clone()

            # --- 3. Synchronized DFM Inference ---
            # We use the new dfm_inference_lock to ensure thread-safety
            with self.models_processor.dfm_inference_lock:
                # PRE-SYNC: Ensure GPU input is ready
                if self.models_processor.device == "cuda":
                    torch.cuda.current_stream().synchronize()

                out_celeb, _, _ = dfm_model.convert(
                    dfm_input,
                    parameters["DFMAmpMorphSlider"] / 100,
                    rct=parameters["DFMRCTColorToggle"],
                )

            # Format security
            if isinstance(out_celeb, np.ndarray):
                out_celeb = torch.from_numpy(out_celeb).to(original_face_512.device)

            if getattr(out_celeb, "ndim", 0) != 3 or out_celeb.shape[2] != 3:
                print(
                    f"[WARN] DFM output shape unexpected: {out_celeb.shape}. Proceeding anyway."
                )

            # --- 4. Dynamic Value Scaling ---
            if out_celeb.max() > 2.0:
                out_celeb_float = out_celeb.float() / 255.0
            else:
                out_celeb_float = out_celeb.float()

            input_face_affined = out_celeb_float.clone()

            prev_face = out_celeb_float.clone()

            output = out_celeb_float * 255.0

        # FW-QUAL-08: warn when all tiles produced zero output (model returned blank).
        # Threshold 30.0 works for the unified [0,255] scale used by all models (incl. DFM after fix above).
        if output.abs().max() < 30.0:
            print(
                "[WARN] Swap model output near-zero for face — possible VRAM pressure"
            )

        output = output.permute(2, 0, 1)
        swap = self.t512(output)
        return swap, prev_face

    def get_border_mask(self, parameters):
        """Creates the border fade mask based on sliders."""
        border_mask = torch.ones(
            (128, 128), dtype=torch.float32, device=self.models_processor.device
        )
        border_mask = torch.unsqueeze(border_mask, 0)

        if not parameters.get("BordermaskEnableToggle", False):
            return border_mask, border_mask.clone()

        top = parameters["BorderTopSlider"]
        left = parameters["BorderLeftSlider"]
        right = 128 - parameters["BorderRightSlider"]
        bottom = 128 - parameters["BorderBottomSlider"]

        # P3-02: clamp border values instead of assert (assert is disabled under -O)
        left = max(0, min(left, 128))
        right = max(left, min(right, 128))
        top = max(0, min(top, 128))
        bottom = max(top, min(bottom, 128))

        border_mask[:, :top, :] = 0
        border_mask[:, bottom:, :] = 0
        border_mask[:, :, :left] = 0
        border_mask[:, :, right:] = 0

        border_mask_calc = border_mask.clone()

        blur_amount = parameters["BorderBlurSlider"]
        blur_kernel_size = blur_amount * 2 + 1
        if blur_kernel_size > 1:
            sigma_val = max(blur_amount * 0.15 + 0.1, 1e-6)
            gauss = v2.GaussianBlur(blur_kernel_size, sigma=sigma_val)
            border_mask = gauss(border_mask)
        return border_mask, border_mask_calc

    def get_dynamic_side_mask(
        self, yaw_deg, pitch_deg, height, width, device, parameters, kps_5, tform
    ):
        """
        Smart Profile Masking:
        Instead of a blind gradient, this uses the projected eye positions to ensure
        we NEVER mask the eyes.
        """
        mask = torch.ones((1, height, width), dtype=torch.float32, device=device)

        if not parameters.get("ProfileAngleMaskEnableToggle", False):
            return mask

        start_angle = parameters.get("ProfileAngleMaskThresholdSlider", 20)
        max_strength = parameters.get("ProfileAngleMaskStrengthSlider", 100) / 100.0

        if tform is not None:
            kps_proj = tform(kps_5)
            le_x = kps_proj[0][0]
            re_x = kps_proj[1][0]
        else:
            le_x = width * 0.35
            re_x = width * 0.65

        le_x_norm = np.clip(le_x / width, 0.0, 1.0)
        re_x_norm = np.clip(re_x / width, 0.0, 1.0)
        eye_safety_margin = 0.05

        abs_yaw = abs(yaw_deg)
        if abs_yaw > start_angle:
            angle_excess = max(0, abs_yaw - start_angle)
            strength_yaw = min(angle_excess / 45.0, 1.0) * max_strength
            linspace_x = torch.linspace(0, 1, width, device=device).view(1, 1, width)

            if yaw_deg > 0:
                # Looking Right -> Mask Left side
                fade_end = max(0.0, le_x_norm - eye_safety_margin)
                if fade_end > 0.05:
                    grad_yaw = torch.clamp(linspace_x / fade_end, 0, 1)
                    grad_yaw = 1.0 - (1.0 - grad_yaw) * strength_yaw
                    mask = mask * grad_yaw
            else:
                # Looking Left -> Mask Right side
                fade_start = min(1.0, re_x_norm + eye_safety_margin)
                if fade_start < 0.95:
                    grad_yaw = torch.clamp(
                        (linspace_x - fade_start) / (1.0 - fade_start), 0, 1
                    )
                    grad_yaw = 1.0 - grad_yaw
                    mask_r = torch.ones_like(linspace_x)
                    mask_r[linspace_x > fade_start] = 1.0 - (
                        (linspace_x[linspace_x > fade_start] - fade_start)
                        / (1.0 - fade_start)
                    )
                    grad_yaw = 1.0 - (1.0 - mask_r) * strength_yaw
                    mask = mask * grad_yaw

        return mask

    def _apply_restorer_with_auto(
        self,
        swap: torch.Tensor,
        swap2: torch.Tensor,
        swap_original2: torch.Tensor,
        original_face_512: torch.Tensor,
        mask_forcalc_512: torch.Tensor,
        parameters: dict,
        tform_scale: float,
        debug: bool,
        debug_info: dict,
        slot_id: int,
    ) -> torch.Tensor:
        """
        FW-QUAL-01/02: Shared helper for Restoration-2 auto-blend logic.

        Applies auto sharpness-driven alpha blending (or Gaussian blur fallback)
        between `swap_original2` (pre-restorer) and `swap2` (post-restorer).
        When auto mode is disabled, applies a simple weighted blend.

        Args:
            swap:             Current swap tensor (not used directly, but returned as output).
            swap2:            Restorer output tensor.
            swap_original2:   Snapshot of swap *before* restorer was applied.
            original_face_512: Original face tensor (used by face_restorer_auto).
            mask_forcalc_512: Mask tensor for sharpness calculation.
            parameters:       Per-face parameters dict.
            tform_scale:      Similarity-transform scale (used as scale_factor).
            debug:            Whether debug mode is active.
            debug_info:       Mutable dict for debug annotations; key is f'Restore2_{slot_id}'.
            slot_id:          Restorer slot identifier (2 for both callers).

        Returns:
            Updated swap tensor after blending.
        """
        debug_key = f"Restore2_{slot_id}"
        if parameters["FaceRestorerAutoEnable2Toggle"]:
            alpha_restorer2 = float(parameters["FaceRestorerBlend2Slider"]) / 100.0
            adjust_sharpness2 = float(parameters["FaceRestorerAutoSharpAdjust2Slider"])
            scale_factor2 = round(tform_scale, 2)
            automasktoggle2 = parameters["FaceRestorerAutoMask2EnableToggle"]
            automaskadjust2 = parameters[
                "FaceRestorerAutoSharpMask2AdjustDecimalSlider"
            ]
            automaskblur2 = 2
            restore_mask = mask_forcalc_512.clone()

            alpha_auto2, blur_value2 = self.face_restorer_auto(
                original_face_512.clone(),
                swap_original2.clone(),
                swap2,
                alpha_restorer2,
                adjust_sharpness2,
                scale_factor2,
                debug,
                restore_mask,
                automasktoggle2,
                automaskadjust2,
                automaskblur2,
            )

            if blur_value2 > 0:
                kernel_size = 2 * blur_value2 + 1
                sigma = blur_value2 * 0.1
                gaussian_blur = v2.GaussianBlur(kernel_size=kernel_size, sigma=sigma)
                swap = gaussian_blur(swap_original2)
                debug_info[debug_key] = f": {-blur_value2:.2f}"
            elif isinstance(alpha_auto2, torch.Tensor):
                swap = swap2 * alpha_auto2 + swap_original2 * (1 - alpha_auto2)
            elif alpha_auto2 != 0:
                swap = swap2 * alpha_auto2 + swap_original2 * (1 - alpha_auto2)
                if debug:
                    debug_info[debug_key] = f": {alpha_auto2 * 100:.2f}"
            else:
                swap = swap_original2
                if debug:
                    debug_info[debug_key] = f": {alpha_auto2 * 100:.2f}"
        else:
            alpha_restorer2 = float(parameters["FaceRestorerBlend2Slider"]) / 100.0
            swap = torch.add(
                torch.mul(swap2, alpha_restorer2),
                torch.mul(swap_original2, 1 - alpha_restorer2),
            )
        return swap

    # ------------------------------------------------------------------
    # Mouth action detection helper
    # ------------------------------------------------------------------
    def _detect_mouth_action_score(
        self,
        face_bbox: "np.ndarray | None" = None,
        vr_crop_chw: "torch.Tensor | np.ndarray | None" = None,
    ) -> "float | None":
        """Run the mouth action detector on a face-region crop.

        For standard mode pass ``face_bbox`` ([x1,y1,x2,y2]) — the function crops
        ``self.frame`` at 2× scale around the bbox so the head and mouth surroundings
        are included without flooding the model with irrelevant background.

        For VR180 mode pass ``vr_crop_chw`` (the 512×512 perspective crop tensor) —
        that image is already a focused face region and is used directly.

        Falls back to the full frame when neither argument is supplied.

        Returns a positive float on detection, or None (no detection) so that the
        MouthOpennessState occlusion-timeout can bridge short missed-frame gaps.
        """
        from app.processors.mouth_action_detector import MouthActionDetector

        detector = MouthActionDetector.get()
        if not detector.available:
            if not getattr(self, "_mouth_action_detector_warned", False):
                err = detector.load_error or "unknown error"
                print(f"[WARN] Mouth action detector unavailable: {err}")
                self._mouth_action_detector_warned = True
            return None

        if vr_crop_chw is not None:
            # VR perspective crop — already CHW, convert to numpy if needed
            if isinstance(vr_crop_chw, torch.Tensor):
                img_chw_np = vr_crop_chw.cpu().numpy().astype(np.uint8)
            else:
                img_chw_np = np.asarray(vr_crop_chw, dtype=np.uint8)
        elif face_bbox is not None and self.frame is not None:
            # Standard mode — crop self.frame (HWC) at 2× scale around the face bbox
            frame_hwc = self.frame  # HWC uint8 RGB numpy
            fh, fw = frame_hwc.shape[:2]
            x1 = float(face_bbox[0])
            y1 = float(face_bbox[1])
            x2 = float(face_bbox[2])
            y2 = float(face_bbox[3])
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            half_w = x2 - x1  # 2× scale → full bbox width as half-extent
            half_h = y2 - y1
            crop_x1 = max(0, int(cx - half_w))
            crop_y1 = max(0, int(cy - half_h))
            crop_x2 = min(fw, int(cx + half_w))
            crop_y2 = min(fh, int(cy + half_h))
            crop_hwc = frame_hwc[crop_y1:crop_y2, crop_x1:crop_x2]
            if crop_hwc.size == 0:
                return None
            img_chw_np = np.transpose(crop_hwc, (2, 0, 1))
        else:
            # Fallback — full frame
            img_chw_np = np.transpose(self.frame, (2, 0, 1))

        raw_score = detector.score(img_chw_np)

        # Return None on no detection so the state machine uses the occlusion grace period
        return raw_score if raw_score > 0.0 else None

    # ------------------------------------------------------------------
    # Auto-Mouth Expression helper
    # ------------------------------------------------------------------
    def _apply_auto_mouth(
        self,
        params: dict,
        target_fb: Any,
        face_bbox: "np.ndarray | None" = None,
        vr_crop_chw: "torch.Tensor | np.ndarray | None" = None,
    ) -> dict:
        """Check auto-mouth state and, if active, return a modified params dict
        using Simple-mode lip transfer (same quality path as 'Restore The Lips').

        Returns *params* unchanged (same object, zero allocation) when the
        feature is disabled or not yet triggered.
        """
        if not params.get("AutoMouthExpressionEnableToggle", False):
            return params

        from app.processors.mouth_openness import MouthOpennessState

        _alpha = params.get("AutoMouthEMAAlphaDecimalSlider", 0.65)
        _threshold = params.get("AutoMouthOpenThresholdDecimalSlider", 0.50)

        # Run detection on the face-region crop for this specific face.
        # None = no detection; triggers occlusion grace period in state machine.
        _ratio: "float | None" = self._detect_mouth_action_score(
            face_bbox=face_bbox, vr_crop_chw=vr_crop_chw
        )

        # Defensive: handle stale button objects that predate this attribute
        _state: "MouthOpennessState | None" = getattr(
            target_fb, "mouth_openness_state", None
        )
        if _state is None:
            target_fb.mouth_openness_state = MouthOpennessState()
            _state = target_fb.mouth_openness_state

        _auto_active, _ema_value = _state.update(_ratio, _alpha, _threshold)

        if _auto_active:
            # Smart gate: respect user's expression restorer when they already have
            # lip transfer active. Only skip auto-mouth when the user is explicitly
            # driving lip motion so we don't override their settings silently.
            _user_mode = params.get("FaceExpressionModeSelection", "Advanced")
            _user_region = str(
                params.get("FaceExpressionAnimationRegionSelection", "all")
            )
            _user_has_lip_transfer = params.get(
                "FaceExpressionEnableBothToggle", False
            ) and (
                (
                    _user_mode == "Advanced"
                    and params.get("FaceExpressionLipsToggle", False)
                )
                or (
                    _user_mode == "Simple"
                    and ("lips" in _user_region or "all" in _user_region)
                )
            )
            if _user_has_lip_transfer:
                return params

            _base_strength = params.get(
                "AutoMouthExpressionStrengthDecimalSlider", 1.00
            )
            _normalize = params.get("AutoMouthNormalizeLipsToggle", True)
            _region = params.get("AutoMouthAnimationRegionSelection", "lips")

            # Proportional strength: ramp from 0 at threshold to full at threshold + ramp_range.
            # This produces a smooth fade-in instead of a binary snap-on.
            _ramp_range = max(_threshold * 0.5, 0.04)
            _proportion = min(1.0, max(0.0, (_ema_value - _threshold) / _ramp_range))
            _strength = _base_strength * _proportion

            # Skip the restorer entirely when strength is negligible — avoids a full
            # landmark-detection + warp-decode cycle with no visible output change.
            if _strength < 0.01:
                return params

            _p = dict(params)
            # Simple mode uses lp_retarget_lip internally via the normalize path —
            # this is the same high-quality path as the manual "Restore The Lips" toggle.
            _p["FaceExpressionEnableBothToggle"] = True
            _p["FaceExpressionModeSelection"] = "Simple"
            _p["FaceExpressionBeforeTypeSelection"] = "Beginning"
            _p["FaceExpressionAnimationRegionSelection"] = _region
            _p["FaceExpressionFriendlyFactorDecimalSlider"] = _strength
            # FaceExpressionNormalizeLipsEnableToggle is the correct Simple-mode key.
            _p["FaceExpressionNormalizeLipsEnableToggle"] = _normalize
            # Pin normalize threshold to ensure lp_retarget_lip always fires when
            # auto-mouth is active, regardless of the user's slider value.
            _p["FaceExpressionNormalizeLipsThresholdDecimalSlider"] = 0.03
            # Pin neutral factor to 1.0 so _strength alone controls intensity.
            # Without this, the user's neutral slider would silently dampen auto-mouth.
            _p["FaceExpressionNeutralDecimalSlider"] = 1.0
            # Explicitly set the Advanced-mode toggle keys that swap_core checks with
            # bare [] access. Plain dict copies may be missing these keys on the first
            # frame for a new face (before ParametersDict has resolved them from defaults).
            _p["FaceExpressionLipsToggle"] = False  # Simple mode; avoid KeyError
            _p["FaceExpressionEyesToggle"] = False
            _p["FaceExpressionBrowsToggle"] = False
            _p["FaceExpressionGeneralToggle"] = False
            # Face-parser mouth/lip override — reads configurable values from the
            # AutoMouth UI section and forces them onto the per-face params, overriding
            # whatever the user has set in the Face Swap tab for these three sliders.
            _mouth_val = int(params.get("AutoMouthMouthParserSlider", 1))
            _upper_val = int(params.get("AutoMouthUpperLipParserSlider", 3))
            _lower_val = int(params.get("AutoMouthLowerLipParserSlider", 17))
            if _mouth_val > 0 or _upper_val > 0 or _lower_val > 0:
                _p["FaceParserEnableToggle"] = True
            _p["MouthParserSlider"] = _mouth_val
            _p["UpperLipParserSlider"] = _upper_val
            _p["LowerLipParserSlider"] = _lower_val
            return _p

        return params

    def swap_core(
        self,
        img: torch.Tensor,
        kps_5: np.ndarray,
        kps: np.ndarray | None = None,  # FW-ROBUST-06: changed default False -> None
        s_e: np.ndarray | None = None,
        t_e: np.ndarray | None = None,
        parameters: dict | None = None,
        control: dict | None = None,
        dfm_model_name: str | None = None,
        is_perspective_crop: bool = False,
        kv_map: Dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """
        Core function for face swapping. Handles:
        1. Alignment and Scaling.
        2. Swapping (Model inference).
        3. Blending and Masking (XSeg, Occluder, Texture Transfer).
        4. Color Correction.
        5. Restoration (GFPGAN/CodeFormer).
        6. Reverse alignment (Untransform).
        """
        valid_s_e = s_e if isinstance(s_e, np.ndarray) else None
        valid_t_e = t_e if isinstance(t_e, np.ndarray) else None
        parameters = parameters if parameters is not None else {}
        control = control if control is not None else {}
        swapper_model = parameters["SwapModelSelection"]
        itex = 1  # FW-BUG-10: default before any branching to prevent NameError

        # FW-PERF-4: set_scaling_transforms is already called in process_frame;
        # calling it again here per-face-per-frame rebuilds 12 transform objects
        # unnecessarily. Removed.

        debug = control.get("CommandLineDebugEnableToggle", False)
        debug_info: dict[str, str] = {}

        tform = self.get_face_similarity_tform(swapper_model, kps_5)

        # OPTIMIZATION: Transform full-frame smoothed keypoints to the 512x512 crop space
        kps_all_crop = None
        if kps is not None and len(kps) == 203:
            raw_kps_crop = tform(kps)
            kps_all_crop = np.array(raw_kps_crop, dtype=np.float32)

        # STATE TRACKER: Suit si le tenseur 'swap' a subi une modification géométrique
        is_geometry_altered = False

        # FW-PERF-5: use promoted instance-attribute transforms (initialized in
        # set_scaling_transforms) instead of constructing new objects each call
        t512_mask = self.t512_mask
        t128_mask = self.t128_mask
        assert t512_mask is not None, (
            "t512_mask must be initialized via set_scaling_transforms before swap_core"
        )
        assert t128_mask is not None, (
            "t128_mask must be initialized via set_scaling_transforms before swap_core"
        )

        _face_interp = (
            "bicubic"
            if parameters.get("FaceAlignmentInterpolation", "Bilinear") == "Bicubic"
            else "bilinear"
        )
        original_face_512, original_face_384, original_face_256, original_face_128 = (
            self.get_transformed_and_scaled_faces(tform, img, interp_mode=_face_interp)
        )
        original_faces = (
            original_face_512,
            original_face_384,
            original_face_256,
            original_face_128,
        )
        swap = original_face_512
        # Initialise prev_face to the normalised original face so that the
        # StrengthEnableToggle blend at the end of swap_core always has a valid
        # tensor, even when get_swapped_and_prev_face is skipped (e.g. DFM
        # selected but no model file chosen, or input_face_affined is None).
        prev_face = torch.div(original_face_512.float(), 255.0).permute(1, 2, 0)

        # --- SWAPPING INFERENCE ---
        if valid_s_e is not None or (
            swapper_model == "DeepFaceLive (DFM)" and dfm_model_name
        ):
            input_face_affined, dfm_model_instance, dim, latent = (
                self.get_affined_face_dim_and_swapping_latents(
                    original_faces,
                    swapper_model,
                    dfm_model_name,
                    valid_s_e,
                    valid_t_e,
                    parameters,
                    debug,
                    tform,
                )
            )

            # FW-BUG-03: guard input_face_affined is None (latent computation failed)
            if input_face_affined is None:
                swap = original_face_512
                # skip to mask section — latent computation failed, use original face
            else:
                # Optional Face Scaling adjustment
                if parameters["FaceAdjEnableToggle"]:
                    input_face_affined = v2.functional.affine(
                        input_face_affined,
                        0,
                        (0, 0),
                        1 + parameters["FaceScaleAmountSlider"] / 100,
                        0,
                        center=(dim * 128 / 2, dim * 128 / 2),
                        interpolation=v2.InterpolationMode.BILINEAR,
                    )

                itex = 1
                if parameters["StrengthEnableToggle"]:
                    itex = ceil(parameters["StrengthAmountSlider"] / 100.0)

                output_size = int(128 * dim)
                output = torch.zeros(
                    (output_size, output_size, 3),
                    dtype=torch.float32,
                    device=self.models_processor.device,
                )
                input_face_affined = input_face_affined.permute(1, 2, 0).contiguous()
                input_face_affined = torch.div(input_face_affined, 255.0)

                swap, prev_face = self.get_swapped_and_prev_face(
                    output,
                    input_face_affined,
                    original_face_512,
                    latent,
                    itex,
                    dim,
                    swapper_model,
                    dfm_model_instance,
                    parameters,
                )

        else:
            swap = original_face_512
            if parameters["StrengthEnableToggle"]:
                itex = ceil(parameters["StrengthAmountSlider"] / 100.0)
                prev_face = torch.div(swap, 255.0)
                prev_face = prev_face.permute(1, 2, 0)

        if parameters["StrengthEnableToggle"]:
            if itex == 0:
                swap = original_face_512.clone()
            else:
                alpha = np.mod(parameters["StrengthAmountSlider"], 100) * 0.01
                if alpha == 0:
                    alpha = 1

                prev_face = torch.mul(prev_face, 255)
                prev_face = torch.clamp(prev_face, 0, 255)
                prev_face = prev_face.permute(2, 0, 1)

                if prev_face.shape[-1] != swap.shape[-1]:
                    prev_face = v2.Resize(
                        (swap.shape[-2], swap.shape[-1]), antialias=True
                    )(prev_face)

                swap = torch.mul(swap, alpha)
                prev_face = torch.mul(prev_face, 1 - alpha)
                swap = torch.add(swap, prev_face)

        # --- DYNAMIC MASKS INITIALIZATION ---
        current_swap_h, current_swap_w = swap.shape[1], swap.shape[2]

        yaw_deg, pitch_deg = faceutil.calc_face_yaw_pitch(kps_5)
        side_mask = self.get_dynamic_side_mask(
            yaw_deg,
            pitch_deg,
            current_swap_h,
            current_swap_w,
            self.models_processor.device,
            parameters,
            kps_5,
            tform,
        )

        # FW-PERF-09: skip get_border_mask entirely when the toggle is off;
        # when disabled it would just return all-ones, so the result equals side_mask.
        if parameters.get("BordermaskEnableToggle", False):
            border_mask, border_mask_calc = self.get_border_mask(parameters)
            if (
                border_mask.shape[1] != current_swap_h
                or border_mask.shape[2] != current_swap_w
            ):
                resizer = v2.Resize((current_swap_h, current_swap_w), antialias=True)
                border_mask = resizer(border_mask)
                border_mask_calc = resizer(border_mask_calc)
            border_mask = border_mask * side_mask
            border_mask_calc = border_mask_calc * side_mask
        else:
            border_mask = side_mask
            border_mask_calc = side_mask

        swap_mask = torch.ones(
            (current_swap_h, current_swap_w),
            dtype=torch.float32,
            device=self.models_processor.device,
        )
        swap_mask = torch.unsqueeze(swap_mask, 0)
        swap_mask_noFP = border_mask.clone()

        # FW-MEM-03: allocate one base ones-tensor and share it for masks that will
        # be unconditionally overwritten before first read; only deep-clone where the
        # initial all-ones value is truly consumed before the mask is reassigned.
        BgExclude = torch.ones(
            (1, 512, 512), dtype=torch.float32, device=self.models_processor.device
        )
        diff_mask = BgExclude  # overwritten inside Differencing block before read
        texture_mask_view = BgExclude  # overwritten inside FaceParser block before read
        # FW-QUAL-04: restore_mask removed here; it is always assigned inside the
        # FaceRestorerAutoEnable2Toggle branch before being read (no early init needed)
        texture_exclude_512 = (
            BgExclude.clone()
        )  # used as default in out.get(); needs own copy

        calc_mask = (
            BgExclude.clone()
        )  # may be read before overwrite in face-parser path
        calc_mask_dill = BgExclude.clone()  # consumed by VGG/masking before overwrite
        mask_forcalc_512 = BgExclude.clone()  # consumed by _apply_restorer_with_auto

        M_ref = cast(np.ndarray, tform.params)[0:2]
        ones_column_ref = np.ones((kps_5.shape[0], 1), dtype=np.float32)
        kps_ref = np.hstack([kps_5, ones_column_ref]) @ M_ref.T

        swap = torch.clamp(swap, 0.0, 255.0)

        # --- FACE EDITING (Beginning) ---
        # Expression Restorer beginning
        if (
            parameters["FaceExpressionEnableBothToggle"]
            and (
                parameters["FaceExpressionLipsToggle"]
                or parameters["FaceExpressionEyesToggle"]
                or parameters["FaceExpressionBrowsToggle"]
                or parameters["FaceExpressionGeneralToggle"]
                or parameters.get("FaceExpressionModeSelection", "Advanced") == "Simple"
            )
            and parameters["FaceExpressionBeforeTypeSelection"] == "Beginning"
        ):
            swap = self.frame_edits.apply_face_expression_restorer(
                original_face_512,
                swap,
                cast(dict, parameters),
                driving_kps=kps_all_crop,
                target_kps=None if is_geometry_altered else kps_all_crop,
            )
            is_geometry_altered = True

        # Face editor beginning
        if (
            parameters["FaceEditorEnableToggle"]
            and self.local_control_state_from_feeder.get(
                "edit_enabled", True
            )  # FW-RACE-02
            and parameters["FaceEditorBeforeTypeSelection"] == "Beginning"
        ):
            editor_mask = swap_mask.clone()
            swap = swap * editor_mask + original_face_512 * (1 - editor_mask)
            swap = self.frame_edits.swap_edit_face_core(
                swap,
                swap,
                parameters,
                control,
                kps_all=None if is_geometry_altered else kps_all_crop,
            )
            is_geometry_altered = True

        # First Denoiser pass - Before Restorers
        if control.get("DenoiserUNetEnableBeforeRestorersToggle", False):
            swap = self._apply_denoiser_pass(swap, control, "Before", kv_map)

        # --- MOUTH ENHANCEMENT & ALIGNMENT (PRE-RESTORER) ---
        paste_after_restorer = parameters.get("MouthParserStretchAfterToggle", False)

        if not paste_after_restorer:
            mouth_overlay_pkg = None
            if hasattr(self.models_processor, "face_masks"):
                mouth_overlay_pkg = self.models_processor.face_masks.get_mouth_overlay(
                    swap, original_face_512, parameters
                )

            if mouth_overlay_pkg is not None:
                overlay_rgb, overlay_mask = mouth_overlay_pkg
                if overlay_rgb is not None and overlay_mask is not None:
                    if overlay_rgb.shape[-1] != swap.shape[-1]:
                        overlay_rgb = v2.Resize(
                            (swap.shape[-2], swap.shape[-1]), antialias=True
                        )(overlay_rgb)
                        overlay_mask = v2.Resize(
                            (swap.shape[-2], swap.shape[-1]), antialias=True
                        )(overlay_mask.unsqueeze(0)).squeeze(0)

                    swap = swap * (1.0 - overlay_mask) + overlay_rgb * overlay_mask

        # --- RESTORATION 1 ---
        # FW-PERF-11: defer clone until we know it is needed (lazy snapshot)
        swap_original = None

        if parameters["FaceRestorerEnableToggle"]:
            # FW-PERF-11: clone only when the restorer will actually run
            swap_original = swap.clone()
            swap_restorecalc = self.models_processor.apply_facerestorer(
                swap,
                parameters["FaceRestorerDetTypeSelection"],
                parameters["FaceRestorerTypeSelection"],
                parameters["FaceRestorerBlendSlider"],
                parameters["FaceFidelityWeightDecimalSlider"],
                control["DetectorScoreSlider"],
                kps_ref,
                slot_id=1,
            )
        else:
            swap_restorecalc = swap.clone()

        # Occluder
        if parameters["OccluderEnableToggle"]:
            mask = self.models_processor.face_masks.apply_occlusion(
                original_face_256,
                parameters["OccluderSizeSlider"],
                parameters=parameters,
                original_face_512=swap_restorecalc,
            )
            if mask.shape[-1] != swap_mask.shape[-1]:
                mask = v2.Resize(
                    (swap_mask.shape[-2], swap_mask.shape[-1]), antialias=True
                )(mask)
            swap_mask = torch.mul(swap_mask, mask)

            gauss = v2.GaussianBlur(
                parameters["OccluderXSegBlurSlider"] * 2 + 1,
                (parameters["OccluderXSegBlurSlider"] + 1) * 0.2,
            )
            swap_mask = gauss(swap_mask)

            if swap_mask_noFP.shape[-1] != swap_mask.shape[-1]:
                swap_mask_noFP = v2.Resize(
                    (swap_mask.shape[-2], swap_mask.shape[-1]), antialias=True
                )(swap_mask_noFP)
            swap_mask_noFP *= swap_mask

        # --- MASKS (Parser / CLIPs / Restore) ---
        need_any_parser = (
            parameters.get("FaceParserEnableToggle", False)
            or (
                parameters.get("DFLXSegEnableToggle", False)
                and (
                    (
                        parameters.get("XSegMouthEnableToggle", False)
                        and parameters.get("DFLXSegSizeSlider", 0)
                        != parameters.get("DFLXSeg2SizeSlider", 0)
                    )
                    or parameters.get("XSegExcludeInnerMouthToggle", False)
                )
            )
            or (
                parameters.get("TransferTextureEnableToggle", False)
                or parameters.get("DifferencingEnableToggle", False)
            )
            and (parameters.get("ExcludeMaskEnableToggle", False))
        )

        FaceParser_mask = None
        mouth_512 = None
        inner_mouth_protection_512 = None

        if need_any_parser:
            out = self.models_processor.process_masks_and_masks(
                swap_restorecalc,
                original_face_512,
                parameters,
                control,
            )
            if not parameters.get("FaceParserEndToggle", False):
                FaceParser_mask = out.get("FaceParser_mask", None)

            texture_exclude_512 = out.get("texture_mask", texture_exclude_512)
            mouth_512 = out.get("mouth", None)
            inner_mouth_protection_512 = out.get("inner_mouth_protection", None)

        if FaceParser_mask is not None:
            if FaceParser_mask.shape[-1] != swap_mask.shape[-1]:
                FaceParser_mask = v2.Resize(
                    (swap_mask.shape[-2], swap_mask.shape[-1]), antialias=True
                )(FaceParser_mask)
            swap_mask = swap_mask * FaceParser_mask

        # CLIPs
        if parameters.get("ClipEnableToggle", False):
            mask_clip = self.models_processor.run_CLIPs(
                original_face_512,
                parameters["ClipText"],
                parameters["ClipAmountSlider"],
            )
            if mask_clip.shape[-1] != swap_mask.shape[-1]:
                mask_clip = v2.Resize(
                    (swap_mask.shape[-2], swap_mask.shape[-1]), antialias=True
                )(mask_clip)
            swap_mask *= mask_clip
            if swap_mask_noFP.shape[-1] != mask_clip.shape[-1]:
                swap_mask_noFP = v2.Resize(
                    (mask_clip.shape[-2], mask_clip.shape[-1]), antialias=True
                )(swap_mask_noFP)
            swap_mask_noFP *= mask_clip

        # Restore Eyes/Mouth
        if parameters.get("RestoreMouthEnableToggle", False) or parameters.get(
            "RestoreEyesEnableToggle", False
        ):
            M = cast(np.ndarray, tform.params)[0:2]
            ones_column = np.ones((kps_5.shape[0], 1), dtype=np.float32)
            dst_kps_5 = np.hstack([kps_5, ones_column]) @ M.T

            img_swap_mask = torch.ones(
                (1, 512, 512), dtype=torch.float32, device=self.models_processor.device
            )
            img_orig_mask = torch.zeros(
                (1, 512, 512), dtype=torch.float32, device=self.models_processor.device
            )

            if parameters.get("RestoreMouthEnableToggle", False):
                img_swap_mask = self.models_processor.restore_mouth(
                    img_orig_mask,
                    img_swap_mask,
                    dst_kps_5,
                    parameters["RestoreMouthBlendAmountSlider"] / 100.0,
                    parameters["RestoreMouthFeatherBlendSlider"],
                    parameters["RestoreMouthSizeFactorSlider"] / 100.0,
                    parameters["RestoreXMouthRadiusFactorDecimalSlider"],
                    parameters["RestoreYMouthRadiusFactorDecimalSlider"],
                    parameters["RestoreXMouthOffsetSlider"],
                    parameters["RestoreYMouthOffsetSlider"],
                ).clamp(0, 1)

            if parameters.get("RestoreEyesEnableToggle", False):
                img_swap_mask = self.models_processor.restore_eyes(
                    img_orig_mask,
                    img_swap_mask,
                    dst_kps_5,
                    parameters["RestoreEyesBlendAmountSlider"] / 100.0,
                    parameters["RestoreEyesFeatherBlendSlider"],
                    parameters["RestoreEyesSizeFactorDecimalSlider"],
                    parameters["RestoreXEyesRadiusFactorDecimalSlider"],
                    parameters["RestoreYEyesRadiusFactorDecimalSlider"],
                    parameters["RestoreXEyesOffsetSlider"],
                    parameters["RestoreYEyesOffsetSlider"],
                    parameters["RestoreEyesSpacingOffsetSlider"],
                ).clamp(0, 1)

            if parameters.get("RestoreEyesMouthBlurSlider", 0) > 0:
                b = parameters["RestoreEyesMouthBlurSlider"]
                gauss = v2.GaussianBlur(b * 2 + 1, (b + 1) * 0.2)
                img_swap_mask = gauss(img_swap_mask)

            if img_swap_mask.shape[-1] != swap_mask.shape[-1]:
                mask_resized = v2.Resize(
                    (swap_mask.shape[-2], swap_mask.shape[-1]), antialias=True
                )(img_swap_mask)
            else:
                mask_resized = img_swap_mask
            swap_mask = swap_mask * mask_resized

        # --- DFL XSeg ---
        # FW-PERF-5: use promoted instance-attribute transform
        t256_near = self.t256_near
        assert t256_near is not None, (
            "t256_near must be initialized via set_scaling_transforms"
        )

        if parameters.get("DFLXSegEnableToggle", False):
            img_xseg_256 = t256_near(original_face_512)
            mouth_256 = None
            inner_mouth_protection_256 = None
            if (
                parameters.get("DFLXSegEnableToggle", False)
                and parameters.get("XSegMouthEnableToggle", False)
                and parameters.get("DFLXSegSizeSlider", 0)
                != parameters.get("DFLXSeg2SizeSlider", 0)
                and mouth_512 is not None
            ):
                mouth_256 = t256_near(mouth_512.unsqueeze(0))

            if (
                parameters.get("XSegExcludeInnerMouthToggle", False)
                and inner_mouth_protection_512 is not None
            ):
                inner_mouth_protection_256 = t256_near(
                    inner_mouth_protection_512.unsqueeze(0)
                ).squeeze(0)

            img_mask_256, mask_forcalc_256, mask_forcalc_dill_256, outpred_noFP_256 = (
                self.models_processor.apply_dfl_xseg(
                    img_xseg_256,
                    -parameters["DFLXSegSizeSlider"],
                    mouth_256 if mouth_256 is not None else 0,
                    parameters,
                    inner_mouth_mask=inner_mouth_protection_256,
                )
            )

            if img_mask_256.shape[-1] != swap_mask.shape[-1]:
                img_mask_res = v2.Resize(
                    (swap_mask.shape[-2], swap_mask.shape[-1]), antialias=True
                )(img_mask_256)
                outpred_noFP_res = v2.Resize(
                    (swap_mask.shape[-2], swap_mask.shape[-1]), antialias=True
                )(outpred_noFP_256)
            else:
                img_mask_res = img_mask_256
                outpred_noFP_res = outpred_noFP_256

            mask_forcalc_512 = t512_mask(mask_forcalc_256)
            mask_forcalc_dill_512 = t512_mask(mask_forcalc_dill_256)

            mask_forcalc_512 = 1 - mask_forcalc_512
            mask_forcalc_dill_512 = 1 - mask_forcalc_dill_512
            calc_mask = mask_forcalc_512
            calc_mask_dill = mask_forcalc_dill_512

            if swap_mask_noFP.shape[-1] != outpred_noFP_res.shape[-1]:
                swap_mask_noFP = v2.Resize(
                    (outpred_noFP_res.shape[-2], outpred_noFP_res.shape[-1]),
                    antialias=True,
                )(swap_mask_noFP)

            swap_mask_noFP = swap_mask_noFP * (1.0 - outpred_noFP_res)
            swap_mask = swap_mask * (1.0 - img_mask_res)
        else:
            calc_mask = t512_mask(swap_mask.clone()).clamp(0, 1)
            calc_mask_dill = calc_mask.clone()
            mask_forcalc_512 = calc_mask.clone()

        mask_autocolor = calc_mask.clone()
        mask_autocolor = mask_autocolor > 0.05

        # Auto Restore (First Pass)
        if (
            parameters["FaceRestorerEnableToggle"]
            and parameters["FaceRestorerAutoEnableToggle"]
        ):
            original_face_512_autorestore = original_face_512.clone()
            assert swap_original is not None, (
                "swap_original must be set when FaceRestorerEnableToggle is active"
            )
            swap_original_autorestore = swap_original.clone()
            alpha_restorer = float(parameters["FaceRestorerBlendSlider"]) / 100.0
            adjust_sharpness = float(parameters["FaceRestorerAutoSharpAdjustSlider"])
            scale_factor = round(tform.scale, 2)
            automasktoggle = parameters["FaceRestorerAutoMaskEnableToggle"]
            automaskadjust = parameters["FaceRestorerAutoSharpMaskAdjustDecimalSlider"]
            automaskblur = 2
            restore_mask = mask_forcalc_512.clone()

            alpha_auto, blur_value = self.face_restorer_auto(
                original_face_512_autorestore,
                swap_original_autorestore,
                swap_restorecalc,
                alpha_restorer,
                adjust_sharpness,
                scale_factor,
                debug,
                restore_mask,
                automasktoggle,
                automaskadjust,
                automaskblur,
            )

            if blur_value > 0:
                kernel_size = 2 * blur_value + 1
                sigma = blur_value * 0.1
                gaussian_blur = v2.GaussianBlur(kernel_size=kernel_size, sigma=sigma)
                swap = gaussian_blur(swap_original)
                debug_info["Restore1"] = f": {-blur_value:.2f}"
            elif isinstance(alpha_auto, torch.Tensor):
                swap = swap_restorecalc * alpha_auto + swap_original * (1 - alpha_auto)
            elif alpha_auto != 0:
                swap = swap_restorecalc * alpha_auto + swap_original * (1 - alpha_auto)
                if debug:
                    debug_info["Restore1"] = f": {alpha_auto * 100:.2f}"
            else:
                swap = swap_original
                if debug:
                    debug_info["Restore1"] = f": {alpha_auto * 100:.2f}"

        elif parameters["FaceRestorerEnableToggle"]:
            alpha_restorer = float(parameters["FaceRestorerBlendSlider"]) / 100.0
            swap = torch.add(
                torch.mul(swap_restorecalc, alpha_restorer),
                torch.mul(swap_original, 1 - alpha_restorer),
            ).contiguous()

        # Expression Restorer (After First)
        if (
            parameters["FaceExpressionEnableBothToggle"]
            and (
                parameters["FaceExpressionLipsToggle"]
                or parameters["FaceExpressionEyesToggle"]
                or parameters["FaceExpressionBrowsToggle"]
                or parameters["FaceExpressionGeneralToggle"]
                or parameters.get("FaceExpressionModeSelection", "Advanced") == "Simple"
            )
            and parameters["FaceExpressionBeforeTypeSelection"]
            == "After First Restorer"
        ):
            swap = self.frame_edits.apply_face_expression_restorer(
                original_face_512,
                swap,
                cast(dict, parameters),
                driving_kps=kps_all_crop,
                target_kps=None if is_geometry_altered else kps_all_crop,
            )
            is_geometry_altered = True

        # Face Editor (After First)
        if (
            parameters["FaceEditorEnableToggle"]
            and self.local_control_state_from_feeder.get(
                "edit_enabled", True
            )  # FW-RACE-02
            and parameters["FaceEditorBeforeTypeSelection"] == "After First Restorer"
        ):
            editor_mask = swap_mask.clone()
            swap = swap * editor_mask + original_face_512 * (1 - editor_mask)
            swap = self.frame_edits.swap_edit_face_core(
                swap,
                swap_restorecalc,
                parameters,
                control,
                kps_all=None if is_geometry_altered else kps_all_crop,
            )
            is_geometry_altered = True
            if swap_mask_noFP.shape[-1] != swap.shape[-1]:
                swap_mask = v2.Resize((swap.shape[-2], swap.shape[-1]), antialias=True)(
                    swap_mask_noFP
                )
            else:
                swap_mask = swap_mask_noFP

        # Second Denoiser pass - After First Restorer
        if control.get("DenoiserAfterFirstRestorerToggle", False):
            swap = self._apply_denoiser_pass(swap, control, "AfterFirst", kv_map)

        # --- RESTORATION 2 ---
        # FW-QUAL-01/02: duplicated ~60-line block extracted to _apply_restorer_with_auto
        if (
            parameters["FaceRestorerEnable2Toggle"]
            and not parameters["FaceRestorerEnable2EndToggle"]
        ):
            swap_original2 = swap.clone()
            swap2 = self.models_processor.apply_facerestorer(
                swap,
                parameters["FaceRestorerDetType2Selection"],
                parameters["FaceRestorerType2Selection"],
                parameters["FaceRestorerBlend2Slider"],
                parameters["FaceFidelityWeight2DecimalSlider"],
                control["DetectorScoreSlider"],
                kps_ref,
                slot_id=2,
            )
            swap = self._apply_restorer_with_auto(
                swap,
                swap2,
                swap_original2,
                original_face_512,
                mask_forcalc_512,
                parameters,
                tform.scale,
                debug,
                debug_info,
                slot_id=2,
            )

        # Expression (After Second)
        if (
            parameters["FaceExpressionEnableBothToggle"]
            and (
                parameters["FaceExpressionLipsToggle"]
                or parameters["FaceExpressionEyesToggle"]
                or parameters["FaceExpressionBrowsToggle"]
                or parameters["FaceExpressionGeneralToggle"]
                or parameters.get("FaceExpressionModeSelection", "Advanced") == "Simple"
            )
            and parameters["FaceExpressionBeforeTypeSelection"]
            == "After Second Restorer"
        ):
            swap = self.frame_edits.apply_face_expression_restorer(
                original_face_512,
                swap,
                cast(dict, parameters),
                driving_kps=kps_all_crop,
                target_kps=None if is_geometry_altered else kps_all_crop,
            )
            is_geometry_altered = True

        # Editor (After Second)
        if (
            parameters["FaceEditorEnableToggle"]
            and self.local_control_state_from_feeder.get(
                "edit_enabled", True
            )  # FW-RACE-02
            and parameters["FaceEditorBeforeTypeSelection"] == "After Second Restorer"
        ):
            editor_mask = t512_mask(swap_mask).clone()
            swap = swap * editor_mask + original_face_512 * (1 - editor_mask)
            swap = self.frame_edits.swap_edit_face_core(
                swap,
                swap,
                parameters,
                control,
                kps_all=None if is_geometry_altered else kps_all_crop,
            )
            is_geometry_altered = True
            if swap_mask_noFP.shape[-1] != swap.shape[-1]:
                swap_mask = v2.Resize((swap.shape[-2], swap.shape[-1]), antialias=True)(
                    swap_mask_noFP
                )
            else:
                swap_mask = swap_mask_noFP

        # --- AUTO COLOR (Mask 512) ---
        # FW-QUAL-12: AutoColorEnableToggle runs here — BEFORE FaceParser mask is applied
        # to the global swap_mask (the EndingColorTransfer runs at the end, AFTER the
        # FaceParser end-pass and the final swap_mask re-calculation, so it operates on a
        # tighter mask that excludes eyes/mouth/hairline etc.).

        # Q-QUAL-03: build a smoothed reference face for AutoColor to reduce per-frame flicker.
        # Key by target embedding bytes so each target face has its own EMA history.
        original_face_for_color = original_face_512
        if parameters.get("AutoColorEnableToggle", False) and valid_t_e is not None:
            _ema_key = valid_t_e.tobytes()
            _face_f = original_face_512.float()
            _curr_mean = _face_f.mean(dim=(1, 2), keepdim=True)
            _curr_std = _face_f.std(dim=(1, 2), keepdim=True) + 1e-6
            if _ema_key in self._color_stats_ema:
                self._color_stats_ema.move_to_end(_ema_key)
                _prev = self._color_stats_ema[_ema_key]
                _ema_mean = (
                    self._COLOR_EMA_ALPHA * _curr_mean
                    + (1.0 - self._COLOR_EMA_ALPHA) * _prev["mean"]
                )
                _ema_std = (
                    self._COLOR_EMA_ALPHA * _curr_std
                    + (1.0 - self._COLOR_EMA_ALPHA) * _prev["std"]
                )
            else:
                _ema_mean, _ema_std = _curr_mean, _curr_std
                # LRU eviction before inserting a new entry
                if len(self._color_stats_ema) >= self._COLOR_STATS_EMA_MAX:
                    self._color_stats_ema.popitem(last=False)
            self._color_stats_ema[_ema_key] = {
                "mean": _ema_mean.detach(),
                "std": _ema_std.detach(),
            }
            # Remap original_face to have smoothed colour statistics
            original_face_for_color = (
                ((_face_f - _curr_mean) / _curr_std * _ema_std + _ema_mean)
                .clamp(0, 255)
                .to(original_face_512.dtype)
            )

        if parameters.get("AutoColorEnableToggle", False):
            if parameters["AutoColorTransferTypeSelection"] == "Test":
                swap = faceutil.histogram_matching(
                    original_face_for_color,
                    swap,
                    parameters["AutoColorBlendAmountSlider"],
                )
            elif parameters["AutoColorTransferTypeSelection"] == "Test_Mask":
                swap = faceutil.histogram_matching_withmask(
                    original_face_for_color,
                    swap,
                    mask_autocolor,
                    parameters["AutoColorBlendAmountSlider"],
                )
            elif parameters["AutoColorTransferTypeSelection"] == "DFL_Test":
                swap = faceutil.histogram_matching_DFL_test(
                    original_face_for_color,
                    swap,
                    parameters["AutoColorBlendAmountSlider"],
                )
            elif parameters["AutoColorTransferTypeSelection"] == "DFL_Orig":
                swap = faceutil.histogram_matching_DFL_Orig(
                    original_face_for_color,
                    swap,
                    mask_autocolor,
                    parameters["AutoColorBlendAmountSlider"],
                )
            elif parameters["AutoColorTransferTypeSelection"] == "AdaIN_Statistical":
                swap = faceutil.apply_adain_color_transfer(
                    swap,
                    original_face_for_color,
                    mask_autocolor,
                    parameters["AutoColorBlendAmountSlider"],
                )

        # --- TRANSFER TEXTURE ---
        if parameters.get("TransferTextureEnableToggle", False):
            # 1. Ensure resolutions match target 512x512
            if swap.shape[-1] != 512:
                swap = t512_mask(swap)
                swap_mask = t512_mask(swap_mask)
                swap_mask_noFP = t512_mask(swap_mask_noFP)

            mask_input_vgg = t128_mask(calc_mask.clone())
            mask_vgg_512 = torch.ones(
                (1, 512, 512), dtype=torch.float32, device=self.models_processor.device
            )

            TextureFeatureLayerTypeSelection = "combo_relu3_3_relu3_1"
            upper_thresh = parameters["TextureUpperLimitSlider"] / 100.0

            # 2. VGG Mask Processing
            if parameters.get("ExcludeOriginalVGGMaskEnableToggle", False):
                # Fetch threshold values from UI
                thr = (
                    parameters["VGGMaskThresholdSlider"]
                    if parameters.get("ExcludeVGGMaskEnableToggle", False)
                    else 0
                )
                soft = 100
                mode = "smooth"

                # Retrieve BOTH the thresholded mask and the raw normalized difference (Size: 128x128)
                mask_vgg_raw, diff_norm_texture_raw = (
                    self.models_processor.apply_vgg_mask_simple(
                        swap,
                        original_face_512,
                        mask_input_vgg,
                        center_pct=thr,
                        softness_pct=soft,
                        feature_layer=TextureFeatureLayerTypeSelection,
                        mode=mode,
                    )
                )

                # Upscale to 512x512 IMMEDIATELY to prevent tensor mismatch
                mask_vgg_512 = t512_mask(mask_vgg_raw).clamp(0.0, 1.0)
                diff_norm_texture_512 = t512_mask(diff_norm_texture_raw).clamp(0.0, 1.0)

                # Fallback to the raw difference texture if manipulation is disabled (Restoring old behavior)
                if not parameters.get("ExcludeVGGMaskEnableToggle", False):
                    mask_vgg_512 = diff_norm_texture_512.clone()

                # Optional VGG specific blur
                if parameters.get("TextureBlendAmountSlider", 0) > 0:
                    b = parameters["TextureBlendAmountSlider"]
                    gauss = v2.GaussianBlur(b * 2 + 1, (b + 1) * 0.2)
                    mask_vgg_512 = gauss(mask_vgg_512.float())

            # 3. Features Exclusion Logic (Eyes, Mouth, etc.)
            if parameters.get("ExcludeMaskEnableToggle", False):
                # texture_exclude_512: 1 means KEEP texture (skin), 0 means REMOVE texture (eyes/mouth)
                feature_mask = texture_exclude_512.clone().float()

                # This creates a smooth gradient transition instead of a harsh binary cut-off.
                if parameters.get("ExcludeOriginalVGGMaskEnableToggle", False):
                    blur_val = parameters.get("FaceParserBlurTextureSlider", 0)
                    if blur_val > 0:
                        kernel_size = int(blur_val * 2 + 1)
                        sigma = max((blur_val + 1) * 0.2, 1e-6)
                        blur_op = v2.GaussianBlur(kernel_size, sigma=sigma)
                        feature_mask = blur_op(feature_mask)

                # Combine VGG mask with the spatial FaceParser mask
                if parameters.get("ExcludeOriginalVGGMaskEnableToggle", False):
                    # Clamp upper limits to protect extreme highlights/differences
                    mask_vgg_512 = torch.where(
                        mask_vgg_512 >= upper_thresh, upper_thresh, mask_vgg_512
                    )

                mask_final_512 = (
                    torch.max(mask_vgg_512 * (1.0 - feature_mask), 1.0 - calc_mask_dill)
                ).clamp(0.0, 1.0)

            elif parameters.get("ExcludeOriginalVGGMaskEnableToggle", False):
                # Clamp upper limits to protect extreme highlights/differences
                mask_vgg_512 = torch.where(
                    mask_vgg_512 >= upper_thresh, upper_thresh, mask_vgg_512
                )
                # Protect background if no spatial exclusion is active
                mask_final_512 = torch.max(mask_vgg_512, 1.0 - calc_mask_dill).clamp(
                    0.0, 1.0
                )

            else:
                # Fallback to raw mask if everything is disabled
                mask_final_512 = (1.0 - mask_forcalc_512).clamp(0.0, 1.0)

            # FW-QUAL-03: dead-code block converted from triple-quoted string to comments.
            # 4. Final Mask Smoothing (Applied only ONCE at the end) — disabled / superseded
            # if parameters.get("FaceParserBlurTextureSlider", 0) > 0:
            #     orig_m = mask_final_512.clone()
            #     b_fp = parameters["FaceParserBlurTextureSlider"]
            #     kernel_size = int(b_fp * 2 + 1)
            #     gauss = v2.GaussianBlur(kernel_size, (b_fp + 1) * 0.2)
            #     mask_final_512 = gauss(mask_final_512.type(torch.float32))
            #     # Restore sharp inner boundaries while softening the gradients
            #     mask_final_512 = torch.max(mask_final_512, orig_m).clamp(0.0, 1.0)
            # 5. AutoColor Backup Logic
            if parameters.get("AutoColorEnableToggle", False):
                swap_texture_backup = swap.clone()
            else:
                swap_texture_backup = faceutil.histogram_matching_DFL_Orig(
                    original_face_512, swap.clone(), mask_autocolor, 100
                )

            # 5. Gradient / Texture Generation Settings
            TransferTextureKernelSizeSlider = 12
            TransferTextureSigmaDecimalSlider = 4.00
            TransferTextureWeightSlider = 1
            TransferTexturePhiDecimalSlider = 9.7
            TransferTextureGammaDecimalSlider = 0.5

            if parameters.get("TransferTextureModeEnableToggle", False):
                TransferTextureLambdSlider = 8
                TransferTextureThetaSlider = 8
            else:
                TransferTextureLambdSlider = 2
                TransferTextureThetaSlider = 1

            clip_limit = (
                parameters["TransferTextureClipLimitDecimalSlider"]
                if parameters.get("TransferTextureClaheEnableToggle", False)
                else 0.0
            )
            alpha_clahe = parameters["TransferTextureAlphaClaheDecimalSlider"]
            grid_size = (4, 4)
            global_gamma = parameters["TransferTexturePreGammaDecimalSlider"]
            global_contrast = parameters["TransferTexturePreContrastDecimalSlider"]

            gradient_texture = self.gradient_magnitude(
                original_face_512,
                calc_mask_dill,
                TransferTextureKernelSizeSlider,
                TransferTextureWeightSlider,
                TransferTextureSigmaDecimalSlider,
                TransferTextureLambdSlider,
                TransferTextureGammaDecimalSlider,
                TransferTexturePhiDecimalSlider,
                TransferTextureThetaSlider,
                clip_limit,
                alpha_clahe,
                grid_size,
                global_gamma,
                global_contrast,
            )

            gradient_texture = faceutil.histogram_matching_DFL_Orig(
                original_face_512, gradient_texture, mask_autocolor, 100
            )

            if parameters["FaceParserBlurTextureSlider"] > 0:
                orig = mask_final_512.clone()
                gauss = v2.GaussianBlur(
                    parameters["FaceParserBlurTextureSlider"] * 2 + 1,
                    (parameters["FaceParserBlurTextureSlider"] + 1) * 0.2,
                )
                mask_final_512 = gauss(mask_final_512.type(torch.float32))
                mask_final_512 = torch.max(mask_final_512, orig).clamp(0.0, 1.0)
            # 6. Final Blending
            # alpha_t modulates the overall strength, w determines the per-pixel application map
            alpha_t = parameters["TransferTextureBlendAmountSlider"] / 100.0
            w = alpha_t * (1.0 - mask_final_512)
            w = w.clamp(0.0, 1.0)

            swap = (swap_texture_backup * (1.0 - w) + gradient_texture * w).clamp(
                0, 255
            )
            texture_mask_view = (1.0 - mask_final_512).clone()

        # --- DIFFERENCING ---
        if parameters.get("DifferencingEnableToggle", False):
            if swap.shape[-1] != 512:
                swap = t512_mask(swap)
                swap_mask = t512_mask(swap_mask)
                swap_mask_noFP = t512_mask(swap_mask_noFP)

            diff_mask_128 = t128_mask(calc_mask.clone())
            swapped_face_resized = swap.clone()
            original_face_resized = original_face_512.clone()
            FeatureLayerTypeSelection = "combo_relu3_3_relu3_1"

            lower_thresh = parameters["DifferencingLowerLimitThreshSlider"] / 100.0
            upper_thresh = parameters["DifferencingUpperLimitThreshSlider"] / 100.0
            middle_value = parameters["DifferencingMiddleLimitValueSlider"] / 100.0
            upper_value = parameters["DifferencingUpperLimitValueSlider"] / 100.0

            mask_diff_128, diff_norm_texture = (
                self.models_processor.apply_perceptual_diff_onnx(
                    swapped_face_resized,
                    original_face_resized,
                    diff_mask_128,
                    lower_thresh,
                    0,
                    upper_thresh,
                    upper_value,
                    middle_value,
                    FeatureLayerTypeSelection,
                    False,
                )
            )

            eps = 1e-6
            inv_lower = 1.0 / max(lower_thresh, eps)
            inv_mid = 1.0 / max((upper_thresh - lower_thresh), eps)
            inv_high = 1.0 / max((1.0 - upper_thresh), eps)

            res_low = diff_norm_texture * inv_lower * middle_value
            res_mid = middle_value + (diff_norm_texture - lower_thresh) * inv_mid * (
                upper_value - middle_value
            )
            res_high = upper_value + (diff_norm_texture - upper_thresh) * inv_high * (
                1.0 - upper_value
            )

            piece = torch.where(
                diff_norm_texture < lower_thresh,
                res_low,
                torch.where(diff_norm_texture > upper_thresh, res_high, res_mid),
            )

            mask512 = t512_mask(piece)
            if parameters.get("DifferencingBlendAmountSlider", 0) > 0:
                b = parameters["DifferencingBlendAmountSlider"]
                gauss = v2.GaussianBlur(b * 2 + 1, (b + 1) * 0.2)
                mask512 = gauss(mask512.float())

            mask512 = torch.max((mask512), 1 - calc_mask_dill)
            mask512 = (mask512).clamp(0, 1)

            swap = (swap * mask512 + original_face_512 * (1.0 - mask512)).clamp(0, 255)
            diff_mask = 1 - mask512.clone()

        # Face Editor (After Texture Transfer)
        if (
            parameters["FaceEditorEnableToggle"]
            and self.local_control_state_from_feeder.get(
                "edit_enabled", True
            )  # FW-RACE-02
            and parameters["FaceEditorBeforeTypeSelection"] == "After Texture Transfer"
        ):
            editor_mask = t512_mask(swap_mask).clone()
            if swap.shape[-1] != 512:
                swap = t512_mask(swap)

            swap = swap * editor_mask + original_face_512 * (1 - editor_mask)
            swap = self.frame_edits.swap_edit_face_core(
                swap,
                swap,
                parameters,
                control,
                kps_all=None if is_geometry_altered else kps_all_crop,
            )
            is_geometry_altered = True

            if swap_mask_noFP.shape[-1] != swap.shape[-1]:
                swap_mask = v2.Resize((swap.shape[-2], swap.shape[-1]), antialias=True)(
                    swap_mask_noFP
                )
            else:
                swap_mask = swap_mask_noFP

        # --- COLOR CORRECTIONS ---
        if parameters["ColorEnableToggle"]:
            swap = torch.unsqueeze(swap, 0).contiguous()
            swap = v2.functional.adjust_gamma(
                swap, parameters["ColorGammaDecimalSlider"], 1.0
            )
            swap = torch.squeeze(swap)
            swap = swap.permute(1, 2, 0).type(torch.float32)

            del_color = torch.tensor(
                [
                    parameters["ColorRedSlider"],
                    parameters["ColorGreenSlider"],
                    parameters["ColorBlueSlider"],
                ],
                device=self.models_processor.device,
            )
            swap += del_color
            swap = torch.clamp(swap, min=0.0, max=255.0)
            swap = swap.permute(2, 0, 1) / 255.0

            swap = v2.functional.adjust_brightness(
                swap, parameters["ColorBrightnessDecimalSlider"]
            )
            swap = v2.functional.adjust_contrast(
                swap, parameters["ColorContrastDecimalSlider"]
            )
            swap = v2.functional.adjust_saturation(
                swap, parameters["ColorSaturationDecimalSlider"]
            )
            swap = v2.functional.adjust_sharpness(
                swap, parameters["ColorSharpnessDecimalSlider"]
            )
            swap = v2.functional.adjust_hue(swap, parameters["ColorHueDecimalSlider"])

            swap = swap * 255.0

        # --- RESTORATION 2 (END) ---
        # FW-QUAL-01/02: duplicated ~60-line block extracted to _apply_restorer_with_auto
        if (
            parameters["FaceRestorerEnable2Toggle"]
            and parameters["FaceRestorerEnable2EndToggle"]
        ):
            swap_original2 = swap.clone()
            swap2 = self.models_processor.apply_facerestorer(
                swap,
                parameters["FaceRestorerDetType2Selection"],
                parameters["FaceRestorerType2Selection"],
                parameters["FaceRestorerBlend2Slider"],
                parameters["FaceFidelityWeight2DecimalSlider"],
                control["DetectorScoreSlider"],
                kps_ref,
                slot_id=2,
            )
            swap = self._apply_restorer_with_auto(
                swap,
                swap2,
                swap_original2,
                original_face_512,
                mask_forcalc_512,
                parameters,
                tform.scale,
                debug,
                debug_info,
                slot_id=2,
            )

        # --- MOUTH ENHANCEMENT & ALIGNMENT (POST-RESTORER) ---
        if parameters.get("MouthParserStretchAfterToggle", False):
            mouth_overlay_pkg = None
            if hasattr(self.models_processor, "face_masks"):
                # 'swap' now contains the fully restored face
                mouth_overlay_pkg = self.models_processor.face_masks.get_mouth_overlay(
                    swap, original_face_512, parameters
                )

            if mouth_overlay_pkg is not None:
                overlay_rgb, overlay_mask = mouth_overlay_pkg
                if overlay_rgb is not None and overlay_mask is not None:
                    if overlay_rgb.shape[-1] != swap.shape[-1]:
                        overlay_rgb = v2.Resize(
                            (swap.shape[-2], swap.shape[-1]), antialias=True
                        )(overlay_rgb)
                        overlay_mask = v2.Resize(
                            (swap.shape[-2], swap.shape[-1]), antialias=True
                        )(overlay_mask.unsqueeze(0)).squeeze(0)

                    swap = swap * (1.0 - overlay_mask) + overlay_rgb * overlay_mask

        # --- FACE PARSER (END) ---
        if parameters.get("FaceParserEnableToggle") and parameters.get(
            "FaceParserEndToggle"
        ):
            out = self.models_processor.process_masks_and_masks(
                swap,
                original_face_512,
                parameters,
                control,
            )

            FaceParser_mask = out.get("FaceParser_mask", None)

            if FaceParser_mask is not None:
                if FaceParser_mask.shape[-1] != swap_mask.shape[-1]:
                    FaceParser_mask = v2.Resize(
                        (swap.shape[-2], swap.shape[-1]), antialias=True
                    )(FaceParser_mask)

                swap_mask = swap_mask * FaceParser_mask

        # Recalculate AutoColor Mask
        calc_mask = t512_mask(swap_mask.clone()).clamp(0, 1)
        mask_autocolor = calc_mask.clone()
        mask_autocolor = mask_autocolor > 0.05

        # AutoColor End (EndingColorTransfer)
        if parameters.get("EndingColorTransferEnableToggle", False):
            if parameters["EndingColorTransferTypeSelection"] == "Test":
                swap = faceutil.histogram_matching(
                    original_face_512, swap, parameters["EndingColorBlendAmountSlider"]
                )
            elif parameters["EndingColorTransferTypeSelection"] == "Test_Mask":
                swap = faceutil.histogram_matching_withmask(
                    original_face_512,
                    swap,
                    mask_autocolor,
                    parameters["EndingColorBlendAmountSlider"],
                )
            elif parameters["EndingColorTransferTypeSelection"] == "DFL_Test":
                swap = faceutil.histogram_matching_DFL_test(
                    original_face_512, swap, parameters["EndingColorBlendAmountSlider"]
                )
            elif parameters["EndingColorTransferTypeSelection"] == "DFL_Orig":
                swap = faceutil.histogram_matching_DFL_Orig(
                    original_face_512,
                    swap,
                    mask_autocolor,
                    parameters["EndingColorBlendAmountSlider"],
                )
            elif parameters["EndingColorTransferTypeSelection"] == "AdaIN_Statistical":
                swap = faceutil.apply_adain_color_transfer(
                    swap,
                    original_face_512,
                    mask_autocolor,
                    parameters["EndingColorBlendAmountSlider"],
                )

        # Third denoiser pass - After all restorations, colour corrections and
        # ending colour transfer.  Placed last so no subsequent colour operation
        # can undo the denoiser's normalisation and cause a visible colour shift.
        if control.get("DenoiserAfterRestorersToggle", False):
            swap = self._apply_denoiser_pass(swap, control, "After", kv_map)

        # Final blending
        if (
            parameters["FinalBlendAdjEnableToggle"]
            and parameters["FinalBlendAmountSlider"] > 0
        ):
            final_blur_strength = parameters["FinalBlendAmountSlider"]
            kernel_size = 2 * final_blur_strength + 1
            sigma = final_blur_strength * 0.1
            gaussian_blur = v2.GaussianBlur(kernel_size=kernel_size, sigma=sigma)
            swap = gaussian_blur(swap)

        # Artefacts: Jpeg
        if parameters["JPEGCompressionEnableToggle"]:
            jpeg_q = int(parameters["JPEGCompressionAmountSlider"])
            if jpeg_q != 100:
                s = float(tform.scale)
                gamma = 0.60
                strength = 0.80
                q_min = 14
                q_max = 100

                jpeg_q_eff = faceutil._map_jpeg_quality(
                    base_q=jpeg_q,
                    face_scale=s,
                    gamma=gamma,
                    strength=strength,
                    q_min=q_min,
                    q_max=q_max,
                )
                if debug:
                    debug_info["JPEG Quality"] = f"{jpeg_q_eff}"

                # FW-BUG-14: renamed swap2 -> swap_jpeg in JPEG block
                swap_jpeg = faceutil.jpegBlur(swap, jpeg_q_eff)
                blend = parameters["JPEGCompressionBlendSlider"] / 100.0
                swap = torch.add(swap_jpeg * blend, swap * (1.0 - blend))

        # Artefacts: BlockShift
        if parameters["BlockShiftEnableToggle"]:
            base_quality = parameters["BlockShiftAmountSlider"]
            max_px = parameters["BlockShiftMaxAmountSlider"]

            # FW-BUG-14: renamed swap2 -> swap_blockshift in BlockShift block
            swap_blockshift = self.apply_block_shift_gpu_jitter(
                swap,
                block_size=base_quality,
                max_amount_pixels=float(max_px),
                seed=1337,
            )

            block_shift_blend = parameters["BlockShiftBlendAmountSlider"] / 100.0
            swap = torch.add(
                torch.mul(swap_blockshift, block_shift_blend),
                torch.mul(swap, 1 - block_shift_blend),
            )

        if parameters["ColorNoiseDecimalSlider"] > 0:
            swap = swap.to(torch.float32)
            noise = (
                (torch.rand_like(swap, dtype=torch.float32) - 0.5)
                * 2
                * parameters["ColorNoiseDecimalSlider"]
            )
            swap = torch.clamp(swap + noise, 0.0, 255.0)

        # FW-PERF-14: only run the FFT/pooling analysis when debug output will be shown
        if debug and control.get("AnalyseImageEnableToggle", False):
            image_analyse_swap = self.analyze_image(swap)
            # FW-QUAL-07: flatten image analysis dict into individual debug keys
            for _k, _v in image_analyse_swap.items():
                debug_info[f"JS:{_k}"] = _v

        if debug and debug_info:
            one_liner = ", ".join(f"{key}={value}" for key, value in debug_info.items())
            print(f"[DEBUG] {one_liner}")

        if is_perspective_crop:
            return t512_mask(swap), t512_mask(swap_mask), None

        # Mask Post-Processing (Final Blend)
        gauss = v2.GaussianBlur(
            parameters["OverallMaskBlendAmountSlider"] * 2 + 1,
            (parameters["OverallMaskBlendAmountSlider"] + 1) * 0.2,
        )
        swap_mask = gauss(swap_mask)

        if border_mask.shape[-1] != swap_mask.shape[-1]:
            border_mask = v2.Resize(
                (swap_mask.shape[-2], swap_mask.shape[-1]), antialias=True
            )(border_mask)

        swap_mask = torch.mul(swap_mask, border_mask)

        if swap.shape[-1] != 512:
            swap = t512_mask(swap)
            swap_mask = t512_mask(swap_mask)

        swap = torch.mul(swap, swap_mask)

        # --- VIEW MODES ---
        original_face_512_clone = None
        if self.is_view_face_compare:
            original_face_512_clone = original_face_512.clone()
            original_face_512_clone = original_face_512_clone.type(torch.uint8)
            original_face_512_clone = original_face_512_clone.permute(1, 2, 0)

        swap_mask_clone = None
        if self.is_view_face_mask:
            mask_show_type = parameters["MaskShowSelection"]
            if mask_show_type == "swap_mask":
                if (
                    parameters["FaceEditorEnableToggle"]
                    and self.local_control_state_from_feeder.get(
                        "edit_enabled", True
                    )  # FW-RACE-02
                ):
                    swap_mask_clone = torch.ones_like(swap_mask).clone()
                else:
                    swap_mask_clone = swap_mask.clone()
            elif mask_show_type == "diff":
                swap_mask_clone = diff_mask.clone()
            elif mask_show_type == "texture":
                swap_mask_clone = texture_mask_view.clone()

            if swap_mask_clone is not None:
                if swap_mask_clone.shape[-1] != 512:
                    swap_mask_clone = t512_mask(swap_mask_clone)
                swap_mask_clone = torch.sub(1, swap_mask_clone)
                swap_mask_clone = torch.cat(
                    (swap_mask_clone, swap_mask_clone, swap_mask_clone), 0
                )
                swap_mask_clone = swap_mask_clone.permute(1, 2, 0)
                swap_mask_clone = torch.mul(swap_mask_clone, 255.0).type(torch.uint8)

        # --- OPTIMIZED UNTRANSFORM (PASTE BACK) USING KORNIA ---
        # Eliminates CPU bound calculations, manual slicing, and memory-heavy paddings.
        # Warps directly to the full frame resolution in one highly optimized GPU pass.

        M_inv = (
            torch.from_numpy(cast(np.ndarray, tform.inverse.params)[0:2])
            .float()
            .unsqueeze(0)
            .to(self.models_processor.device)
        )
        dsize = (img.shape[1], img.shape[2])  # Full frame size (H, W)

        # Warp the 512x512 face and mask directly into the full frame space
        swap_full = kgm.warp_affine(
            swap.unsqueeze(0).float(),
            M_inv,
            dsize=dsize,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        ).squeeze(0)

        swap_mask_full = kgm.warp_affine(
            swap_mask.unsqueeze(0).float(),
            M_inv,
            dsize=dsize,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        ).squeeze(0)

        # Alpha blending on the GPU
        img_float = img.float()
        swap_mask_minus = 1.0 - swap_mask_full

        # We just add the pre-multiplied face to the masked background.
        img_float = swap_full + (img_float * swap_mask_minus)

        img = img_float.clamp_(0, 255).type(torch.uint8)

        return img, original_face_512_clone, swap_mask_clone

    @torch.no_grad()
    def gradient_magnitude(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        kernel_size: int,
        weighting_strength: float,
        sigma: float,
        lambd: float,
        gamma: float,
        psi: float,
        theta_count: int,
        clip_limit: float,
        alpha_clahe: float,
        grid_size: tuple[int, int],
        global_gamma: float,
        global_contrast: float,
    ) -> torch.Tensor:
        """
        Calculates the weighted Gabor magnitude for texture transfer.

        Args:
            image: Tensor [C, H, W] in [0..255]
            mask:  Tensor [C, H, W] (0/1)
        Returns:
            Tensor [C, H, W] – weighted Gabor magnitude
        """

        C, H, W = image.shape
        image = image.float() / 255.0
        mask = mask.bool()

        # 1) Global Gamma & Contrast
        if global_gamma != 1.0:
            image = image.pow(global_gamma)
        if global_contrast != 1.0:
            m_gc = image.mean((1, 2), keepdim=True)
            image = (image - m_gc) * global_contrast + m_gc

        # 2) CLAHE in L-channel (with alpha_clahe blending)
        if clip_limit > 0.0:
            image = image.unsqueeze(0).clamp(0, 1)  # [1,3,H,W]
            mask_b3 = mask.unsqueeze(0)  # [1,3,H,W]

            lab = kc.rgb_to_lab(image)  # [1,3,H,W]
            L = lab[:, 0:1, :, :] / 100.0  # [1,1,H,W]

            mb = mask_b3[:, 0:1, :, :]  # [1,1,H,W]
            area_l = mb.sum((2, 3), keepdim=True).clamp(min=1)
            mean_l = (L * mb).sum((2, 3), keepdim=True) / area_l
            Lf = torch.where(mb, L, mean_l)
            Leq = ke.equalize_clahe(
                Lf,
                clip_limit=clip_limit,
                grid_size=grid_size,
                slow_and_differentiable=False,
            ).clamp(0, 1)
            L_blend = alpha_clahe * Leq + (1 - alpha_clahe) * L
            Lnew = torch.where(mb, L_blend, L)

            lab_eq = torch.cat([Lnew * 100.0, lab[:, 1:, :, :]], dim=1)  # [1,3,H,W]
            x_eq = kc.lab_to_rgb(lab_eq)
            image = x_eq.squeeze(0)

        # 3) Gabor Filter setup
        kernel_size = max(1, 2 * kernel_size - 1)
        if theta_count == 10:
            theta_values = torch.tensor([math.pi / 4], device=image.device)
        else:
            theta_values = torch.linspace(
                0, math.pi, theta_count + 1, device=image.device
            )[:-1]

        # 4) Single Gabor Filter call
        magnitude = self.apply_gabor_filter_torch(
            image, kernel_size, sigma, lambd, gamma, psi, theta_values
        )  # [C, H, W]

        # 5) Invert
        max_mv = magnitude.amax((1, 2), keepdim=True)
        inverted = max_mv - magnitude  # [C, H, W]

        # 6) Weighting
        if weighting_strength > 0:
            img_m = image * mask
            weighted = inverted * (
                (1 - weighting_strength) + weighting_strength * img_m
            )
        else:
            weighted = inverted

        return weighted * 255  # [C, H, W]

    def apply_gabor_filter_torch(
        self, image, kernel_size, sigma, lambd, gamma, psi, theta_values
    ):
        """
        Applies Gabor filter bank to image.

        Args:
            image: Tensor [C, H, W]
            theta_values: Tensor [N]
        Returns:
            Tensor [C, H, W]
        """
        C, H, W = image.shape
        image = image.unsqueeze(0)  # → [1, C, H, W]

        N = theta_values.shape[0]

        kernels = self.get_gabor_kernels(
            kernel_size, sigma, lambd, gamma, psi, theta_values, image.device
        )  # [N, 1, k, k]

        # FW-PERF-06: cache expanded kernels keyed by (shape, C) to avoid repeat_interleave overhead
        if not hasattr(self, "_gabor_kernels_cache"):
            from collections import OrderedDict

            self._gabor_kernels_cache = OrderedDict()
        expand_cache_key = (*kernels.shape, C)
        if expand_cache_key not in self._gabor_kernels_cache:
            # FW-MEM-01: bound the LRU cache size
            MAX_GABOR_CACHE = 16
            if len(self._gabor_kernels_cache) >= MAX_GABOR_CACHE:
                self._gabor_kernels_cache.popitem(last=False)
            self._gabor_kernels_cache[expand_cache_key] = kernels.repeat_interleave(
                C, dim=0
            )
        # expand to all channels:
        weight = self._gabor_kernels_cache[expand_cache_key]  # → [N*C, 1, k, k]
        out = F.conv2d(
            image,  # [1, C, H, W]
            weight,
            padding=kernel_size // 2,
            groups=C,  # each channel group gets N filters
        )  # out: [1, N*C, H, W]

        # reshape to [N, C, H, W]:
        out = out.squeeze(0).view(N, C, H, W)
        magnitudes = out.amax(dim=0)
        return magnitudes

    def get_gabor_kernels(
        self, kernel_size, sigma, lambd, gamma, psi, theta_values, device
    ):
        """
        Returns: Tensor [N, 1, k, k]

        FW-QUAL-3: Kernels are cached by parameter tuple so they are only
        rebuilt when the parameters actually change.
        """
        N = theta_values.shape[0]
        cache_key = (
            int(kernel_size),
            float(sigma),
            float(lambd),
            float(gamma),
            float(psi),
            int(N),
            str(device),
        )
        cached = self._gabor_kernels_cache.get(cache_key)
        if cached is not None:
            return cached

        half = kernel_size // 2
        y, x = torch.meshgrid(
            torch.linspace(-half, half, kernel_size, device=device),
            torch.linspace(-half, half, kernel_size, device=device),
            indexing="ij",
        )

        kernels = []
        for theta in theta_values:
            x_theta = x * torch.cos(theta) + y * torch.sin(theta)
            y_theta = -x * torch.sin(theta) + y * torch.cos(theta)

            gb = torch.exp(-0.5 * (x_theta**2 + (gamma**2) * y_theta**2) / sigma**2)
            gb *= torch.cos(2 * math.pi * x_theta / lambd + psi)
            kernels.append(gb)

        result = torch.stack(kernels).unsqueeze(1)  # → [N, 1, k, k]
        # FW-MEM-01: evict oldest entry if cache exceeds limit
        MAX_GABOR_CACHE = 32
        if len(self._gabor_kernels_cache) >= MAX_GABOR_CACHE:
            self._gabor_kernels_cache.popitem(last=False)
        self._gabor_kernels_cache[cache_key] = result
        return result

    def face_restorer_auto(
        self,
        original_face_512,  # [3,H,W], float in [0..255]
        swap_original,  # [3,H,W]
        swap,  # [3,H,W]
        alpha,  # initial scalar alpha (ignored; we binary search below)
        adjust_sharpness,
        scale_factor,
        debug,
        swap_mask,
        alpha_map_enable: bool = False,
        alpha_map_strength: float = 0.5,
        alpha_map_blur: int = 7,
    ):
        """Auto-Restorer: Blends between restored and original image based on sharpness."""
        # Baseline sharpness of original
        scores_original = self.sharpness_score(original_face_512)
        score_new_original = (
            scores_original["combined"].item() * 100 + adjust_sharpness / 10.0
        )

        # Binary search for scalar alpha
        alpha = 1.0
        max_iterations = 7
        alpha_min, alpha_max = 0.0, 1.0
        tolerance = 0.5
        min_alpha_change = 0.05
        iteration = 0
        prev_alpha = alpha
        iteration_blur = 0

        while iteration < max_iterations:
            swap2 = swap * alpha + swap_original * (1 - alpha)
            swap2_masked = swap2.clone()

            scores_swap = self.sharpness_score(swap2_masked)
            score_new_swap = scores_swap["combined"].item() * 100
            sharpness_diff = score_new_swap - score_new_original

            if abs(sharpness_diff) < tolerance:
                break

            if sharpness_diff < 0:
                if alpha > 0.99:
                    prev_alpha = alpha
                    break
                alpha_min = alpha
                alpha = (alpha + alpha_max) / 2.0
            else:
                alpha_max = alpha
                alpha = (alpha + alpha_min) / 2.0

            # Very small alpha -> blur fallback on base
            if sharpness_diff >= 0 and alpha < 0.07:
                prev_alpha = 0.0
                base = swap_original
                max_blur_strength = 10
                # FW-PERF-10: precompute GaussianBlur objects outside the scoring loop
                blur_kernels_for_auto = [
                    (
                        v2.GaussianBlur(1, 1e-6)
                        if bs == 0
                        else v2.GaussianBlur(2 * bs + 1, max(bs, 1e-6))
                    )
                    for bs in range(0, max_blur_strength + 1)
                ]
                for bs, gaussian_blur in enumerate(blur_kernels_for_auto):
                    swap2_blurred = gaussian_blur(base)
                    scores_swap_b = self.sharpness_score(swap2_blurred)
                    score_new_swap_b = scores_swap_b["combined"].item() * 100.0
                    sharpness_diff_b = score_new_swap_b - score_new_original

                    if sharpness_diff_b < 0:
                        iteration_blur = 0 if bs == 0 else (bs - 1)
                        break
                    if abs(sharpness_diff_b) <= tolerance:
                        iteration_blur = bs
                        break
                    iteration_blur = bs
                break

            if abs(prev_alpha - alpha) < min_alpha_change:
                prev_alpha = (prev_alpha + alpha) / 2.0
                if abs(prev_alpha) <= 0.05:
                    prev_alpha = 0.0
                break

            prev_alpha = alpha
            iteration += 1

        # Per-pixel alpha map, derived from sharpness distribution
        if alpha_map_enable and (prev_alpha > 0.0):
            # Build the *final* composite (for a stable map), then sharpness map of it
            swap_final = swap * prev_alpha + swap_original * (1 - prev_alpha)

            s_map = self.sharpness_map(
                swap_final,
                mask=swap_mask,
                tenengrad_thresh=0.05,
                comb_weight=0.5,
                smooth_kernel=alpha_map_blur
                if (alpha_map_blur and alpha_map_blur % 2 == 1)
                else 0,
            )

            # Mean sharpness inside mask (or global)
            if swap_mask is not None:
                m = (
                    (swap_mask if swap_mask.dim() == 2 else swap_mask.squeeze(0))
                    .float()
                    .to(s_map.device)
                )
                denom = m.sum().clamp_min(1.0)
                mu = (s_map * m).sum() / denom
            else:
                mu = s_map.mean()

            # Deviation map around mean, scale around prev_alpha
            dev = (s_map - mu).clamp(-1.0, 1.0)
            alpha_map = prev_alpha * (1.0 + alpha_map_strength * dev)
            alpha_map = alpha_map.clamp(0.0, 1.0)

            # Keep outside-face area at scalar prev_alpha (if a mask is provided)
            if swap_mask is not None:
                m = (
                    (swap_mask if swap_mask.dim() == 2 else swap_mask.squeeze(0))
                    .float()
                    .to(alpha_map.device)
                )
                alpha_map = alpha_map * m + prev_alpha * (1.0 - m)

            return alpha_map.unsqueeze(0), iteration_blur

        # Fallback: scalar like before
        return prev_alpha, iteration_blur

    def sharpness_score(
        self,
        image: torch.Tensor,
        mask: torch.Tensor = None,
        tenengrad_thresh: float = 0.05,
        comb_weight: float = 0.5,
    ) -> dict:
        """
        Calculates three sharpness metrics on an RGB image:
          1) var_lap: Variance of Laplacian
          2) tten: Thresholded Tenengrad (Proportion of strong edges)
          3) combined: comb_weight*var_lap + (1-comb_weight)*tten

        Args:
            image: Tensor [3, H, W], float in [0..1]
            mask:  optional Tensor [H, W] or [1, H, W] with 1=valid, 0=ignore
            tenengrad_thresh: Threshold for Tenengrad (0..1)
            comb_weight: Weight for var_lap in combination (0..1)

        Returns:
            {
              "var_lap": float Tensor,
              "ttengrad": float Tensor,
              "combined": float Tensor
            }
        """
        image = image / 255.0

        # 1) Grayscale [1,1,H,W]
        gray = image.mean(dim=0, keepdim=True).unsqueeze(0)

        # 2) Optional Mask on [H,W]
        if mask is not None:
            m = mask.float()
            if m.dim() == 3:  # [1,H,W]
                m = m.squeeze(0)
        else:
            m = None

        def valid_count(t):
            return m.sum().clamp(min=1.0) if m is not None else t.numel()

        # --- Variance of Laplacian ---
        # OPTIMIZED: Use pre-allocated kernel from VRAM to avoid CPU micro-allocations
        L = F.conv2d(gray, self.kernel_lap, padding=1).squeeze()  # [H,W]
        L2 = L.pow(2)
        if m is not None:
            L = L * m
            L2 = L2 * m
        cnt = valid_count(L2)
        mean_L2 = L2.sum() / cnt
        mean_L = L.sum() / cnt
        var_lap = (mean_L2 - mean_L.pow(2)).clamp(min=0.0)

        # --- Thresholded Tenengrad ---
        # OPTIMIZED: Use pre-allocated kernels
        Gx = F.conv2d(gray, self.kernel_sobel_x, padding=1).squeeze()  # [H,W]
        Gy = F.conv2d(gray, self.kernel_sobel_y, padding=1).squeeze()
        G = (Gx.pow(2) + Gy.pow(2)).sqrt()
        if m is not None:
            G = G * m
        total = cnt
        strong = (G > tenengrad_thresh).float().sum()
        ttengrad = strong / total

        # --- Combined Score ---
        combined = comb_weight * var_lap + (1 - comb_weight) * ttengrad

        return {"var_lap": var_lap, "ttengrad": ttengrad, "combined": combined}

    def sharpness_map(
        self,
        image: torch.Tensor,  # [3,H,W], float in [0..255]
        mask: torch.Tensor | None = None,
        tenengrad_thresh: float = 0.05,
        comb_weight: float = 0.5,
        smooth_kernel: int = 5,  # odd; 0/1 = no blur
    ) -> torch.Tensor:
        """
        Returns a normalized per-pixel sharpness map in [0..1] with shape [H,W].
        Combines Laplacian energy + gradient magnitude (Tenengrad-like).
        """
        eps = 1e-8
        device = image.device

        # [3,H,W] -> [1,1,H,W] gray, range [0..1]
        gray = (image / 255.0).mean(dim=0, keepdim=True).unsqueeze(0)

        # OPTIMIZED: Convs using pre-allocated VRAM kernels
        lap = F.conv2d(gray, self.kernel_lap, padding=1).squeeze(0).squeeze(0)  # [H,W]
        gx = (
            F.conv2d(gray, self.kernel_sobel_x, padding=1).squeeze(0).squeeze(0)
        )  # [H,W]
        gy = F.conv2d(gray, self.kernel_sobel_y, padding=1).squeeze(0).squeeze(0)
        grad = (gx.pow(2) + gy.pow(2)).sqrt()  # [H,W]

        # Robust normalization via percentiles inside mask (if given)
        def robust_norm(x, msk):
            if msk is not None:
                sel = x[msk > 0]
                if sel.numel() < 16:  # fallback if mask tiny
                    sel = x.reshape(-1)
            else:
                sel = x.reshape(-1)
            p5 = (
                torch.quantile(sel, 0.05)
                if sel.numel() > 0
                else torch.tensor(0.0, device=device)
            )
            p95 = (
                torch.quantile(sel, 0.95)
                if sel.numel() > 0
                else torch.tensor(1.0, device=device)
            )
            y = (x - p5) / (p95 - p5 + eps)
            return y.clamp_(0, 1)

        m = None
        if mask is not None:
            m = (mask if mask.dim() == 2 else mask.squeeze(0)).float().to(device)

        lap_n = robust_norm(lap.abs(), m)
        grad_n = robust_norm(grad, m)

        smap = comb_weight * lap_n + (1.0 - comb_weight) * grad_n  # [H,W]

        # Optional smoothing to avoid noisy alpha
        # FW-ROBUST-09: ensure kernel size is odd and at least 3
        if smooth_kernel:
            if smooth_kernel % 2 == 0:
                smooth_kernel += 1
            smooth_kernel = max(3, smooth_kernel)
        if smooth_kernel and smooth_kernel >= 3:
            k = smooth_kernel
            smap3 = smap.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
            gb = v2.GaussianBlur(kernel_size=k, sigma=max(1, k // 2))
            smap = gb(smap3).squeeze(0).squeeze(0)

        return smap.clamp(0, 1)

    @torch.no_grad()
    def apply_block_shift_gpu_jitter(
        self,
        img: torch.Tensor,
        block_size: int,
        max_amount_pixels: float,
        *,
        seed: int = 1337,
        pad_mode: str = "replicate",
        align_corners: bool = True,
    ) -> torch.Tensor:
        """
        MPEG-like Block Jitter: shifts every BxB block field by a
        deterministic (bx, by)-dependent offset in pixels.

        Args:
            img: Tensor [C, H, W] (BGR/RGB agnostic). CPU or CUDA.
            block_size: Block size B (e.g. 8).
            max_amount_pixels: max |Offset| in pixels (applied to both axes).
            seed: global seed for deterministic offsets (frame-stable).
            pad_mode: Padding mode for border (replicate|reflect|zeros).
            align_corners: as in grid_sample.

        Returns:
            Tensor [C, H, W] – same Device/Dtype as input.
        """
        seed = seed + self.frame_number * 17
        assert img.ndim == 3, "expected [C,H,W]"
        C, H, W = img.shape
        device = img.device
        dtype = img.dtype

        # calculate on float32 for grid_sample if necessary
        work = (
            img
            if img.dtype in (torch.float32, torch.float16, torch.bfloat16)
            else img.float()
        )

        # Pad to multiples of B (bottom/right), crop back later
        # FW-BUG-13: use block_size directly as B; old 2**block_size was exponential
        B = max(1, int(block_size))
        H_pad = (B - (H % B)) % B
        W_pad = (B - (W % B)) % B
        if H_pad or W_pad:
            pad = (0, W_pad, 0, H_pad)  # (left, right, top, bottom)
            mode = {
                "replicate": "replicate",
                "reflect": "reflect",
                "zeros": "constant",
            }[pad_mode]
            work = F.pad(work[None], pad=pad, mode=mode).squeeze(0)
        Hp, Wp = work.shape[-2:]

        # Number of blocks
        nby = Hp // B
        nbx = Wp // B

        # --- deterministic offsets per block in range [-max, +max] ---
        # Build block coordinate fields
        by_grid, bx_grid = torch.meshgrid(
            torch.arange(nby, device=device, dtype=torch.float32),
            torch.arange(nbx, device=device, dtype=torch.float32),
            indexing="ij",
        )
        # simple Hash -> [0,1)
        h = torch.sin((bx_grid * 12.9898 + by_grid * 78.233 + float(seed)) * 43758.5453)
        frac = torch.frac(h * 0.5 + 0.5)

        # derive two independent offsets from hash
        # The / 4 scales the slider range to produce subtle MPEG-like block artifacts
        # rather than extreme pixel shifts (slider values of 40-100 become 10-25 px).
        _scaled_max = float(max_amount_pixels) / 4.0
        dx_base = ((frac) * 2.0 - 1.0) * _scaled_max

        # second "source": just another linear combo
        h2 = torch.sin(
            (bx_grid * 96.233 + by_grid * 15.987 + (float(seed) + 101)) * 12345.6789
        )
        frac2 = torch.frac(h2 * 0.5 + 0.5)
        dy_base = ((frac2) * 2.0 - 1.0) * _scaled_max

        # upsample to pixel grid by tiling each block offset BxB
        dx = torch.repeat_interleave(
            torch.repeat_interleave(dx_base, B, dim=0), B, dim=1
        )  # [Hp,Wp]
        dy = torch.repeat_interleave(
            torch.repeat_interleave(dy_base, B, dim=0), B, dim=1
        )  # [Hp,Wp]

        # --- Build Flow-Field for grid_sample ---
        xs = torch.linspace(-1.0, 1.0, Wp, device=device)
        ys = torch.linspace(-1.0, 1.0, Hp, device=device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")  # [Hp,Wp]
        dx_norm = (2.0 * dx) / max(Wp - 1, 1)
        dy_norm = (2.0 * dy) / max(Hp - 1, 1)

        flow_x = grid_x + dx_norm
        flow_y = grid_y + dy_norm
        flow = torch.stack([flow_x, flow_y], dim=-1)  # [Hp,Wp,2]

        warped = F.grid_sample(
            work[None],
            flow[None],
            mode="bilinear",
            padding_mode="border",
            align_corners=align_corners,
        ).squeeze(0)

        # crop back to original size if padded
        if H_pad or W_pad:
            warped = warped[..., :H, :W]

        if warped.dtype != dtype:
            warped = warped.to(dtype)

        return warped

    def analyze_image(self, image):
        """
        Analyses a CHW uint8 image tensor and returns a dict of quality scores in [0, 1].

        Computed metrics:
          - ``jpeg_artifacts``: High-frequency energy (higher = more ringing/blocking).
          - ``salt_pepper_noise``: Fraction of pixels with sharp local outliers.
          - ``speckle_noise``: Mean local variance (higher = more speckle).
          - ``blur``: Inverted Laplacian edge strength (higher = blurrier).
          - ``low_contrast``: Inverted pixel standard deviation (higher = flatter).
        """
        image = image.float() / 255.0
        C, H, W = image.shape
        grayscale = torch.mean(image, dim=0, keepdim=True)
        analysis = {}
        fft = torch.fft.fft2(grayscale)
        high_freq_energy = torch.mean(torch.abs(fft))
        analysis["jpeg_artifacts"] = min(high_freq_energy.item() / 50, 1.0)
        median_filtered = F.avg_pool2d(grayscale, 3, stride=1, padding=1)
        noise_map = torch.abs(grayscale - median_filtered)
        sp_noise = torch.mean((noise_map > 0.1).float())
        analysis["salt_pepper_noise"] = min(sp_noise.item() * 10, 1.0)
        local_var = F.avg_pool2d(grayscale**2, 5, stride=1, padding=2) - (
            F.avg_pool2d(grayscale, 5, stride=1, padding=2) ** 2
        )
        speckle_noise = torch.mean(local_var)
        analysis["speckle_noise"] = min(speckle_noise.item() * 50, 1.0)

        # OPTIMIZED: Use pre-allocated Laplacian kernel
        laplace_edges = F.conv2d(grayscale.unsqueeze(0), self.kernel_lap, padding=1)
        edge_strength = torch.mean(torch.abs(laplace_edges))
        analysis["blur"] = 1.0 - min(edge_strength.item() * 5, 1.0)
        contrast = grayscale.std()
        analysis["low_contrast"] = 1.0 - min(contrast.item() * 10, 1.0)
        return analysis
