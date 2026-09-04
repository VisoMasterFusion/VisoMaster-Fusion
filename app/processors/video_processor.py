import threading
import queue
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any, Tuple, Optional, List
import time
import subprocess
from pathlib import Path
import os
import gc
from functools import partial
import shutil
import uuid
from datetime import datetime
import cv2
import numpy
import torch
import pyvirtualcam
import copy
from PySide6.QtCore import QObject, QTimer, Signal, Slot

# Internal project imports
from app.processors.workers.frame_worker import FrameWorker
from app.processors.video_utils.sequential_detector import SequentialDetector
from app.processors.video_utils.worker_pool_manager import WorkerPoolManager
from app.processors.video_utils.media_pipeline import (
    MediaPipeline,
    TAIL_TOLERANCE,
    MAX_CONSECUTIVE_ERRORS,
)
from app.ui.widgets.actions import graphics_view_actions
from app.ui.widgets.actions import common_actions as common_widget_actions
from app.ui.widgets.actions import video_control_actions
from app.ui.widgets.actions import layout_actions
from app.ui.widgets.actions import list_view_actions
from app.ui.widgets.actions import save_load_actions
from app.ui.widgets.settings_layout_data import CAMERA_BACKENDS
from app.processors.video_utils.video_encoding import FFmpegEncoder, FFmpegPostProcessor
from app.processors.video_utils.issue_scanner import (
    IssueScanner,
    IssueScanTargetSnapshot,
)
import app.helpers.miscellaneous as misc_helpers
from app.helpers.typing_helper import (
    ControlTypes,
    FacesParametersTypes,
)

if TYPE_CHECKING:
    from app.ui.main_ui import MainWindow


class VideoProcessor(QObject):
    """
    Manages all video, image, and webcam processing pipelines.

    This class handles:
    - Reading frames from media (video, image, webcam).
    - Dispatching frames to worker threads (FrameWorker) for processing.
    - Managing the display metronome (QTimer) for smooth playback/recording.
    - Handling default and multi-segment recording via FFmpeg.
    - Controlling the virtual camera (pyvirtualcam) output.
    - Managing audio playback (ffplay) during preview.

    Thread Safety:
    - Critical: Handles `cuda streams` and TensorRT synchronization.
    - Uses `state_lock` to safeguard parameter updates during playback.
    """

    # --- Signals ---
    # Removed QPixmap to ensure thread safety. GUI thread will handle conversion.
    frame_processed_signal = Signal(int, numpy.ndarray)
    webcam_frame_processed_signal = Signal(numpy.ndarray)
    single_frame_processed_signal = Signal(int, int, numpy.ndarray, object)
    processing_started_signal = Signal()  # Unified signal for any processing start
    processing_stopped_signal = Signal()  # Unified signal for any processing stop
    processing_heartbeat_signal = Signal()  # Emits periodically to show liveness
    fatal_processing_error_signal = Signal(str)

    def __init__(self, main_window: "MainWindow", num_threads=2):
        """
        Initialises the VideoProcessor.

        Sets up all media-state, processing-flag, subprocess, metronome, frame-display,
        and multi-segment recording attributes.  Connects internal worker signals to
        their display/storage slots.

        Args:
            main_window: The application's MainWindow, used to access UI widgets,
                         controls, and the models processor.
            num_threads: Number of persistent FrameWorker pool threads to create for
                         parallel frame processing.
        """
        super().__init__()
        self.main_window = main_window

        # --- Worker Thread Management ---
        self.num_threads = num_threads
        # RAM OPTIMIZATION: Bounded Preroll. We only need 10 frames to ensure smooth UI playback.
        self.preroll_target = 10

        # RAM OPTIMIZATION: Tightened queue bounds. We only need enough queue depth to hold
        # the active threads, the preroll requirement, and a tiny 4-frame slack buffer.
        self.max_display_buffer_size = self.num_threads + self.preroll_target + 4
        self.max_frames_to_display_size = 8  # VP-22: Hard cap on frames_to_display dict

        # Instantiate the decoupled thread and VRAM manager
        self.worker_pool_manager = WorkerPoolManager(self.main_window)
        self.worker_pool_manager.recreate_queue(self.max_display_buffer_size)

        # Instantiate the decoupled Producer/Consumer pipeline
        self.media_pipeline = MediaPipeline(self, self.main_window)

        # --- Media State ---
        self.media_capture: cv2.VideoCapture | None = None
        self.file_type: str | None = None  # "video", "image", or "webcam"
        self.fps = 0.0  # Target FPS for playback or recording
        self.media_path: str | None = None
        self.media_rotation: int = 0
        self.current_frame_number = 0  # The *next* frame to be read/processed
        self.max_frame_number = 0
        self.current_frame: Optional[numpy.ndarray] = (
            None  # The most recently read/processed frame
        )

        # --- Sequential Detection State ---
        # Initialize the decoupled detector state manager
        self.sequential_detector = SequentialDetector(self.main_window)
        # Transition flags: reset tracker only when target-face presence changes,
        # not on every frame — prevents ByteTrack reinitialization per frame (webcam FPS fix).
        self._video_had_targets: bool = False
        self._webcam_had_targets: bool = False

        # --- Processing State Flags ---
        self.processing = False  # MASTER flag: True if playback, recording, or webcam stream is active
        self.recording: bool = False  # True if "default-style" recording is active
        self.is_processing_segments: bool = (
            False  # True if "multi-segment" recording is active
        )
        self.triggered_by_job_manager: bool = False  # For multi-segment job integration
        self._fatal_processing_error_latched: bool = False
        self.last_processing_error: str | None = None
        self.active_output_folder: str = ""

        # --- Subprocesses ---
        self.virtcam: pyvirtualcam.Camera | None = None
        self._virtcam_error_latch: bool = False
        self.encoder = FFmpegEncoder()
        self.ffmpeg_input_sp: subprocess.Popen | None = (
            None  # ffmpeg process that feeds raw frames for recording FPS cap mode
        )
        self.ffmpeg_input_width: int = 0
        self.ffmpeg_input_height: int = 0
        # True when this processing session intends to use an FFmpeg FPS-cap
        # path (even if the subprocess hasn't fully started yet). This flag
        # is used to decide output->source frame mapping and restore logic
        # after finalization without relying on process-stop side effects.
        self._used_ffmpeg_cap: bool = False
        self.ffmpeg_input_prefetched_frame: Optional[numpy.ndarray] = None
        self.tail_pending_stall_start_sec: float = 0.0
        self.tail_force_finalize_due_to_stall: bool = False
        self.recording_source_fps: float = 0.0

        # --- Metronome and Timing ---
        self.processing_start_frame: int = (
            0  # The frame number where processing started
        )

        # --- Performance Timing ---
        self.start_time = 0.0
        self.end_time = 0.0
        self.play_start_time = 0.0  # Used by default style for audio segmenting
        self.play_end_time = 0.0  # Used by default style for audio segmenting

        # Adding Cuda Streams for thread safety
        self.feeder_stream = (
            None  # torch.cuda.Stream() if torch.cuda.is_available() else None
        )

        # --- Default Recording State ---
        self.temp_file: str = ""  # Temporary video file (without audio)
        # Counters for accurate duration calculation
        self.frames_written: int = 0  # Number of frames successfully sent to FFmpeg
        self.last_displayed_frame: int | None = (
            None  # Last frame number that was displayed/written
        )

        # --- Multi-Segment Recording State ---
        self.segments_to_process: List[Tuple[int, int]] = []
        self.current_segment_index: int = -1
        self.temp_segment_files: List[str] = []
        self.current_segment_end_frame: int | None = None
        self.segment_temp_dir: str | None = None

        # --- Utility Timers ---
        self.gpu_memory_update_timer = QTimer()
        self.gpu_memory_update_timer.timeout.connect(
            partial(common_widget_actions.update_gpu_memory_progressbar, main_window)
        )

        # --- Frame Display/Storage ---
        self.next_frame_to_display = 0  # The next frame number the UI should display

        # Fallback frame cached during slider seek preview so process_current_frame()
        # can use it when the near-EOF re-read fails (OpenCV seek unreliability).
        self._seek_cached_frame: Optional[Tuple[int, numpy.ndarray]] = None

        # Note: frames_to_display and webcam_frames_to_display are now managed
        # dynamically via @property decorators routing to self.media_pipeline.

        # Frame cache
        self._last_requested_frame_num: int | None = None
        self._cached_raw_frame_media_path: str | None = None
        self._cached_raw_frame_number: int | None = None
        self._cached_raw_frame_target_height: int | None = None
        self._cached_raw_frame_bgr: numpy.ndarray | None = None
        self._cached_raw_image_path: str | None = None
        self._cached_raw_image_target_height: int | None = None
        self._cached_raw_image_bgr: numpy.ndarray | None = None

        # --- Signal Connections ---
        self.frame_processed_signal.connect(self.store_frame_to_display)
        self.webcam_frame_processed_signal.connect(self.store_webcam_frame_to_display)
        self.single_frame_processed_signal.connect(self.display_current_frame)
        self.single_frame_processed_signal.connect(self.store_single_frame_to_display)
        self.fatal_processing_error_signal.connect(self._handle_fatal_processing_error)

    @property
    def ui_state_is_dirty(self) -> bool:
        return self.media_pipeline.ui_state_is_dirty

    @ui_state_is_dirty.setter
    def ui_state_is_dirty(self, value: bool) -> None:
        self.media_pipeline.ui_state_is_dirty = value

    @property
    def feeder_parameters(self) -> FacesParametersTypes | None:
        return self.media_pipeline.feeder_parameters

    @feeder_parameters.setter
    def feeder_parameters(self, value: FacesParametersTypes | None) -> None:
        self.media_pipeline.feeder_parameters = value

    @property
    def feeder_control(self) -> ControlTypes | None:
        return self.media_pipeline.feeder_control

    @feeder_control.setter
    def feeder_control(self, value: ControlTypes | None) -> None:
        self.media_pipeline.feeder_control = value

    @property
    def feeder_thread(self) -> threading.Thread | None:
        return self.media_pipeline.feeder_thread

    @feeder_thread.setter
    def feeder_thread(self, value: threading.Thread | None) -> None:
        self.media_pipeline.feeder_thread = value

    @property
    def detector_thread(self) -> threading.Thread | None:
        return self.media_pipeline.detector_thread

    @detector_thread.setter
    def detector_thread(self, value: threading.Thread | None) -> None:
        self.media_pipeline.detector_thread = value

    @property
    def state_lock(self) -> threading.Lock:
        return self.media_pipeline.state_lock

    @property
    def preroll_timer(self) -> QTimer:
        return self.media_pipeline.preroll_timer

    @property
    def playback_started(self) -> bool:
        return self.media_pipeline.playback_started

    @playback_started.setter
    def playback_started(self, value: bool) -> None:
        self.media_pipeline.playback_started = value

    @property
    def playback_display_start_time(self) -> float:
        return self.media_pipeline.playback_display_start_time

    @playback_display_start_time.setter
    def playback_display_start_time(self, value: float) -> None:
        self.media_pipeline.playback_display_start_time = value

    @property
    def skipped_frames(self) -> set[int]:
        return self.media_pipeline.skipped_frames

    @property
    def consecutive_read_errors(self) -> int:
        return self.media_pipeline.consecutive_read_errors

    @consecutive_read_errors.setter
    def consecutive_read_errors(self, value: int) -> None:
        self.media_pipeline.consecutive_read_errors = value

    @property
    def max_consecutive_errors(self) -> int:
        return MAX_CONSECUTIVE_ERRORS

    @property
    def total_skipped_frames(self) -> int:
        return self.media_pipeline.total_skipped_frames

    @total_skipped_frames.setter
    def total_skipped_frames(self, value: int) -> None:
        self.media_pipeline.total_skipped_frames = value

    @property
    def stopped_by_error_limit(self) -> bool:
        return self.media_pipeline.stopped_by_error_limit

    @stopped_by_error_limit.setter
    def stopped_by_error_limit(self, value: bool) -> None:
        self.media_pipeline.stopped_by_error_limit = value

    @property
    def manual_dropped_skip_count(self) -> int:
        return self.media_pipeline.manual_dropped_skip_count

    @manual_dropped_skip_count.setter
    def manual_dropped_skip_count(self, value: int) -> None:
        self.media_pipeline.manual_dropped_skip_count = value

    @property
    def read_error_skip_count(self) -> int:
        return self.media_pipeline.read_error_skip_count

    @read_error_skip_count.setter
    def read_error_skip_count(self, value: int) -> None:
        self.media_pipeline.read_error_skip_count = value

    @property
    def frame_queue(self) -> queue.Queue:
        """Facade Property: Dynamically forwards queue operations to the Manager."""
        return self.worker_pool_manager.frame_queue

    @property
    def frames_to_display(self) -> dict:
        """Facade Property: Routes buffer access to the MediaPipeline."""
        return self.media_pipeline.frames_to_display

    @property
    def webcam_frames_to_display(self) -> queue.Queue:
        """Facade Property: Routes webcam buffer access to the MediaPipeline."""
        return self.media_pipeline.webcam_frames_to_display

    def store_frame_to_display(self, frame_number: int, frame: numpy.ndarray) -> None:
        """Facade: Routes finished frames to the MediaPipeline."""
        self.media_pipeline.store_frame_to_display(frame_number, frame)

    def store_webcam_frame_to_display(self, frame: numpy.ndarray) -> None:
        """Facade: Routes finished webcam frames to the MediaPipeline."""
        self.media_pipeline.store_webcam_frame_to_display(frame)

    def stop_live_sound(self) -> None:
        """Facade: Stops audio via MediaPipeline."""
        self.media_pipeline.stop_live_sound()

    @Slot(int, int, numpy.ndarray, object)
    def display_current_frame(
        self,
        generation: int,
        frame_number: int,
        frame: numpy.ndarray,
        preview_cache: object = None,
    ) -> None:
        """
        Slot to display a single, specific frame.
        Used after seeking or loading new media. NOT part of the metronome loop.
        """
        # Validate against WorkerPoolManager's generation state to drop out-of-order ghost frames
        if (
            generation != 0
            and generation
            != self.worker_pool_manager.active_single_frame_request_generation
        ):
            return

        # Reject "ghost" frames from older threads during fast UI scrubbing
        if self.file_type == "video" and frame_number != self.next_frame_to_display:
            del frame
            return

        pixmap = common_widget_actions.get_pixmap_from_frame(self.main_window, frame)

        if getattr(self.main_window, "loading_new_media", False):
            graphics_view_actions.update_graphics_view(
                self.main_window, pixmap, frame_number, reset_fit=True
            )
            self.main_window.loading_new_media = False
        else:
            graphics_view_actions.update_graphics_view(
                self.main_window, pixmap, frame_number
            )

        self.current_frame = frame
        common_widget_actions.update_gpu_memory_progressbar(self.main_window)

        # Check if auto-fit was requested for this generation in the WorkerPoolManager
        if (
            self.worker_pool_manager.fit_on_single_frame_request_generation is not None
            and generation
            == self.worker_pool_manager.fit_on_single_frame_request_generation
        ):
            self.worker_pool_manager.fit_on_single_frame_request_generation = None
            QTimer.singleShot(
                0,
                lambda: layout_actions.fit_image_to_view_onchange(self.main_window),
            )

    @Slot(int, int, numpy.ndarray, object)
    def store_single_frame_to_display(
        self,
        generation: int,
        frame_number: int,
        frame: numpy.ndarray,
        preview_cache: object = None,
    ) -> None:
        """Stores a single preview frame directly to the display buffer, respecting generation order."""
        if (
            generation != 0
            and generation
            != self.worker_pool_manager.active_single_frame_request_generation
        ):
            return
        self.store_frame_to_display(frame_number, frame)

    def _get_target_input_height(self) -> Optional[int]:
        """
        Helper to determine the target input height if global resize is enabled.
        Returns None if resizing is disabled or invalid.
        """
        return self._get_target_input_height_for_control(self.main_window.control)

    @staticmethod
    def _get_target_input_height_for_control(
        control: Mapping[str, Any] | None,
    ) -> Optional[int]:
        resize_enabled = (
            bool(control.get("GlobalInputResizeToggle", False))
            if isinstance(control, Mapping)
            else False
        )

        if not resize_enabled:
            return None

        try:
            # Get the selected resolution string (e.g., "720p")
            size_str = (
                control.get("GlobalInputResizeSizeSelection", "720p")
                if isinstance(control, Mapping)
                else "720p"
            )
            # Extract the number (e.g., 720)
            return int(str(size_str).replace("p", ""))
        except Exception as e:
            print(
                f"[WARN] Could not parse global input resolution, defaulting to original size. Error: {e}"
            )
            return None

    def _get_issue_scanner_instance(self) -> IssueScanner:
        """Helper to instantiate the decoupled IssueScanner with current media state."""
        return IssueScanner(
            main_window=self.main_window,
            sequential_detector=self.sequential_detector,
            media_path=self.media_path,
            max_frame_number=self.max_frame_number,
            media_rotation=self.media_rotation,
        )

    def get_issue_scan_unavailable_reason(
        self,
        control: Mapping[str, Any] | None,
        scan_ranges: Iterable[tuple[int, int]] | None = None,
        markers: Mapping[Any, Any] | None = None,
        fallback_control: Mapping[str, Any] | None = None,
    ) -> str | None:
        return self._get_issue_scanner_instance().get_issue_scan_unavailable_reason(
            control, scan_ranges, markers, fallback_control
        )

    def send_frame_to_virtualcam(self, frame: numpy.ndarray):
        """
        OPTIMIZED: Sends the given frame to the pyvirtualcam device.
        Removed sleep_until_next_frame() to prevent blocking the Main GUI Thread.
        The UI metronome (QTimer) already handles perfect timing and synchronization.
        """
        if self.main_window.control.get("SendVirtCamFramesEnableToggle", False):
            # JIT Initialization: Ensure VirtCam is spun up if toggle is active but uninitialized.
            # We use a Circuit Breaker (_virtcam_error_latch) to prevent infinite loops if initialization completely fails.
            if not self.virtcam and not getattr(self, "_virtcam_error_latch", False):
                self.enable_virtualcam()

            # Need to check again if virtcam was successfully enabled
            if self.virtcam:
                height, width, _ = frame.shape
                if self.virtcam.height != height or self.virtcam.width != width:
                    # Resolution changed (e.g. source swap / restorer output differs).
                    # Avoid hammering OBS with rapid close/reopen cycles
                    print(
                        f"[INFO] VirtCam resolution changed ({self.virtcam.width}x{self.virtcam.height} -> {width}x{height}). Restarting virtual camera."
                    )
                    self.enable_virtualcam()
                    return  # Frame already consumed; next tick will send at the new size.

                if self.virtcam:
                    try:
                        self.virtcam.send(frame)
                    except Exception as e:
                        print(
                            f"[WARN] Catastrophic failure sending frame to virtualcam: {e}"
                        )
                        # If the driver crashes midway, trip the circuit breaker and disable to prevent spam.
                        self._virtcam_error_latch = True
                        self.disable_virtualcam()

    def set_number_of_threads(self, value):
        """Updates the thread count for the *next* worker pool."""
        if not value:
            value = 1
        # Stop processing if it's running, to apply the new count on next start
        if self.processing or self.is_processing_segments:
            print(
                f"[INFO] Setting thread count to {value}. Stopping active processing."
            )
            self.stop_processing()
        else:
            print(f"[INFO] Max Threads set as {value}. Will be applied on next run.")

        self.main_window.models_processor.set_number_of_threads(value)
        self.num_threads = value
        self.preroll_target = 10
        self.max_display_buffer_size = self.num_threads + self.preroll_target + 4

    def process_video(self):
        """
        Start video processing.
        This can be either simple playback OR "default-style" recording.
        """

        # 1. Guards
        if self.processing or self.is_processing_segments:
            print(
                "[INFO] Processing already in progress (play or segment). Ignoring start request."
            )
            # Reset recording flag so a caller that set it before this guard fires
            # does not leave the application in a state where recording=True but
            # nothing is actually recording.
            if self.recording and not self.is_processing_segments:
                self.recording = False
                video_control_actions.reset_media_buttons(self.main_window)
            return

        if self.file_type != "video":
            print("[WARN] Process video: Only applicable for video files.")
            return

        if not (self.media_capture and self.media_capture.isOpened()):
            # Attempt lazy reopen — the capture may have been released during finalization
            # of a previous recording and the OS file handle not yet fully freed.
            if self.file_type == "video" and self.media_path:
                print(
                    "[INFO] media_capture not open on process_video() entry; attempting reopen..."
                )
                current_slider_pos = self.main_window.videoSeekSlider.value()
                if self._reopen_video_capture(current_slider_pos):
                    print("[INFO] media_capture reopened successfully.")
                else:
                    self.media_capture = None

            if not (self.media_capture and self.media_capture.isOpened()):
                print("[ERROR] Unable to open the video source.")
                self.processing = False
                self.recording = False
                self.is_processing_segments = False
                video_control_actions.reset_media_buttons(self.main_window)
                return

        # 2. Determine source/target FPS (after guards so media_capture is confirmed open)
        src_fps = self.media_capture.get(cv2.CAP_PROP_FPS)
        if src_fps <= 0:
            src_fps = 30.0
        self.recording_source_fps = float(src_fps)

        if self.recording:
            # Recording must not be affected by playback custom FPS controls.
            fps_cap_enabled = bool(
                self.main_window.control.get("OutputFpsCapEnableToggle", False)
            )
            fps_cap_value = float(
                self.main_window.control.get("OutputMaxFpsSlider", 30) or 30
            )
            use_ffmpeg_cap = (
                fps_cap_enabled
                and fps_cap_value > 0
                and self.recording_source_fps > fps_cap_value
            )

            self.fps = fps_cap_value if use_ffmpeg_cap else self.recording_source_fps
            self._used_ffmpeg_cap = use_ffmpeg_cap

            # When FPS cap is active, max_frame_number must be in output frame space
            # (i.e. how many frames FFmpeg will actually emit), not source frame count.
            if use_ffmpeg_cap:
                src_frame_count = int(self.media_capture.get(cv2.CAP_PROP_FRAME_COUNT))
                duration_sec = (
                    src_frame_count / self.recording_source_fps
                    if self.recording_source_fps > 0
                    else 0
                )
                if src_frame_count > 0 and duration_sec > 0:
                    output_frames = max(1, int(round(duration_sec * self.fps)))
                    self.max_frame_number = output_frames - 1
                    # Slider stays in source frame space (approach 2); no setMaximum needed.
                else:
                    print(
                        f"[WARN] FPS cap: could not compute output frame count "
                        f"(src_frame_count={src_frame_count}, "
                        f"recording_source_fps={self.recording_source_fps}). "
                        "Disabling FPS-cap input path and falling back to source FPS."
                    )
                    self._used_ffmpeg_cap = False
                    self.fps = self.recording_source_fps
        else:
            self._used_ffmpeg_cap = False
            if self.main_window.control["VideoPlaybackCustomFpsToggle"]:
                self.fps = self.main_window.control["VideoPlaybackCustomFpsSlider"]
            else:
                self.fps = self.recording_source_fps

        mode = "recording (default-style)" if self.recording else "playback"
        print(f"[INFO] Starting video {mode} processing setup...")

        # 3. Set State Flags
        self.processing = True  # General flag ON
        self.is_processing_segments = False
        self.is_playing_segments = False  # Flag for segmented playback via UI toggle
        self.playback_started = False
        self.stopped_by_error_limit = False  # Reset error limit flag for new processing
        self.tail_pending_stall_start_sec = 0.0
        self.tail_force_finalize_due_to_stall = False
        self._fatal_processing_error_latched = False
        self.last_processing_error = None

        # Initialize feeder state with the current UI global state
        with self.state_lock:
            self.feeder_parameters = copy.deepcopy(self.main_window.parameters)
            self.feeder_control = copy.deepcopy(self.main_window.control)

        # Seed global PyTorch/CUDA RNG once per video session from the denoiser seed
        # slider. This ensures reproducible denoiser output for the whole video without
        # resetting the seed on every frame (which would break multi-threaded workers).
        _denoiser_seed = int(
            self.main_window.control.get("DenoiserBaseSeedSlider", 220)
        )
        torch.manual_seed(_denoiser_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(_denoiser_seed)

        # Check if this recording was initiated by the Job Manager
        job_mgr_flag = getattr(self.main_window, "job_manager_initiated_record", False)
        if self.recording and job_mgr_flag:
            self.triggered_by_job_manager = True
            print("[INFO] Detected default-style recording initiated by Job Manager.")
        else:
            self.triggered_by_job_manager = False
        try:
            self.main_window.job_manager_initiated_record = False
        except Exception:
            pass

        # 4. Setup Recording (if applicable)
        if self.recording:
            output_folder = video_control_actions.resolve_output_folder(
                self.main_window, str(self.media_path)
            )
            self.active_output_folder = output_folder
            # Disable UI elements
            if not self.main_window.control["KeepControlsToggle"]:
                layout_actions.disable_all_parameters_and_control_widget(
                    self.main_window
                )

        # 6a. Reset Timers and Containers
        self.start_time = time.perf_counter()
        self.frames_to_display.clear()
        self.frames_written = 0
        self.last_displayed_frame = None

        # 6b. START WORKER POOL
        print(f"[INFO] Starting {self.num_threads} persistent worker thread(s)...")
        # Ensure old workers are cleared (from a previous run).
        self.join_and_clear_threads(clear_module_caches=False)
        self.worker_pool_manager.recreate_queue(self.max_display_buffer_size)
        self.worker_pool_manager.start_persistent_pool(self.num_threads)

        # --- 7. AUDIO/VIDEO SYNC LOGIC ---

        # 7a. Get the target frame (slider is in SOURCE-frame space under Approach 2)
        actual_start_frame = self.main_window.videoSeekSlider.value()

        # --- Segmented Playback Snap Logic ---
        if not self.recording and self.main_window.control.get(
            "VideoPlaybackSegmentsToggle", False
        ):
            raw_markers = getattr(self.main_window, "job_marker_pairs", [])
            valid_pairs = []
            for pair in raw_markers:
                if pair[1] is not None and pair[0] < pair[1]:
                    valid_pairs.append((int(pair[0]), int(pair[1])))

            if valid_pairs:
                self.segments_to_process = sorted(valid_pairs)
                self.is_playing_segments = True

                found_segment = False
                for i, (start_f, end_f) in enumerate(self.segments_to_process):
                    if actual_start_frame < start_f:
                        # Slider is before this segment; snap forward to its start
                        actual_start_frame = start_f
                        self.current_segment_index = i
                        self.current_segment_end_frame = end_f
                        found_segment = True
                        break
                    elif start_f <= actual_start_frame < end_f:
                        # Slider is actively inside this segment
                        self.current_segment_index = i
                        self.current_segment_end_frame = end_f
                        found_segment = True
                        break

                if not found_segment:
                    # Slider is at the exact end of the last segment or past it.
                    # Loop seamlessly back to the first segment so it doesn't instantly EOF.
                    actual_start_frame = self.segments_to_process[0][0]
                    self.current_segment_index = 0
                    self.current_segment_end_frame = self.segments_to_process[0][1]

                print(
                    f"[INFO] Sync: Segment Playback Active. Snapping to frame {actual_start_frame} (Segment {self.current_segment_index + 1}/{len(self.segments_to_process)})."
                )

        elif not self.recording and self.main_window.control.get(
            "VideoPlaybackLoopToggle", False
        ):
            # --- EOF Playback Loop Snap Logic ---
            # If the user clicks play exactly at (or very near) the end of the file, immediately
            # snap to 0. This prevents the initial capture read from failing and locking the state.
            if actual_start_frame >= self.max_frame_number - 1:
                actual_start_frame = 0
                print(
                    "[INFO] Sync: EOF reached with loop enabled. Snapping to frame 0 for playback."
                )

        # Native audio seeking can land at a preceding indexed/keyframe point,
        # whereas OpenCV returns the requested frame.  In the optional accurate
        # sync mode, use that keyframe as the shared preview origin.  Recording
        # deliberately retains exact requested-frame semantics.
        self.media_pipeline.live_sound_seek_time = None
        if (
            self.main_window.liveSoundButton.isChecked()
            and self.main_window.control.get("AccurateAudioVideoSyncToggle", False)
            and not self.recording
        ):
            actual_start_frame, self.media_pipeline.live_sound_seek_time = (
                self.media_pipeline.resolve_live_preview_start(
                    actual_start_frame, src_fps
                )
            )

        print(f"[INFO] Sync: Seeking directly to source-frame {actual_start_frame}...")

        # 7b/7c. Read the first frame (OpenCV path or FFmpeg FPS-cap path).
        target_height = self._get_target_input_height()

        # Compute output-space start frame when using FFmpeg FPS-cap so seeks and
        # internal feeder use the same frame origin (output frame space).
        output_start_frame = actual_start_frame
        if self._used_ffmpeg_cap and self.recording_source_fps > 0 and self.fps > 0:
            output_start_frame = max(0, self.source_to_output_frame(actual_start_frame))

        if self._used_ffmpeg_cap:
            if not self._start_recording_ffmpeg_input_stream(
                start_frame=output_start_frame,
                target_fps=float(self.fps),
                target_height=target_height,
            ):
                print("[ERROR] Failed to start FFmpeg recording input stream.")
                self.stop_processing()
                return

            print(
                "[INFO] Sync: Reading first frame from FFmpeg recording input stream..."
            )
            ret, frame_bgr = self._read_frame_from_ffmpeg_input_stream()
            print(f"[INFO] Sync: Initial FFmpeg stream read complete (Result: {ret}).")

            if not ret or frame_bgr is None:
                print("[ERROR] FFmpeg recording input stream produced no first frame.")
                self.stop_processing()
                return

            # Preserve the prefetched frame so the feeder still processes frame 0.
            self.ffmpeg_input_prefetched_frame = frame_bgr.copy()
        else:
            misc_helpers.seek_frame(self.media_capture, actual_start_frame)

            print(
                f"[INFO] Sync: Reading frame {actual_start_frame} using locked helper (Target Height: {target_height})..."
            )
            ret, frame_bgr = misc_helpers.read_frame(
                self.media_capture,
                self.media_rotation,
                preview_target_height=target_height,
            )
            print(f"[INFO] Sync: Initial read complete (Result: {ret}).")

            if not ret:
                fallback_frame = int(self.media_capture.get(cv2.CAP_PROP_POS_FRAMES))
                fallback_frame_to_try = max(0, fallback_frame - 1)
                print(
                    f"[WARN] Failed initial read for frame {actual_start_frame}. Retrying from frame {fallback_frame_to_try}."
                )
                if fallback_frame_to_try == actual_start_frame:
                    print("[ERROR] Fallback frame is the same. Cannot proceed.")
                    self.stop_processing()
                    return
                self.media_capture.set(cv2.CAP_PROP_POS_FRAMES, fallback_frame_to_try)
                print(
                    f"[INFO] Sync: Retrying read for frame {fallback_frame_to_try} using locked helper..."
                )
                ret, frame_bgr = misc_helpers.read_frame(
                    self.media_capture,
                    self.media_rotation,
                    preview_target_height=target_height,
                )
                print(f"[INFO] Sync: Retry read complete (Result: {ret}).")
                if not ret:
                    print(
                        f"[ERROR] Capture failed definitively near frame {actual_start_frame}."
                    )
                    self.stop_processing()
                    return
                actual_start_frame = fallback_frame_to_try

        # 7d. Frame is valid - Store for potential FFmpeg init
        frame_rgb = numpy.ascontiguousarray(frame_bgr[..., ::-1])  # BGR to RGB
        self.current_frame = frame_rgb  # Store for FFmpeg dimensions

        # DELAYED FFMPEG CREATION
        if self.recording:
            self.temp_file = self._prepare_default_temp_file()
            if os.path.exists(self.temp_file):
                try:
                    os.remove(self.temp_file)
                except OSError:
                    pass

            frame_height, frame_width, _ = self.current_frame.shape

            success = self.encoder.start_process(
                output_filename=self.temp_file,
                frame_width=frame_width,
                frame_height=frame_height,
                fps=self.fps,
                control=self.main_window.control,
                is_segment=False,
                media_path=self.media_path,
            )

            if not success:
                print("[ERROR] Failed to start FFmpeg for default-style recording.")
                self.stop_processing()  # Abort the start
                return

        if not self.ffmpeg_input_sp:
            # !!! CRITICAL: Reset position AGAIN so the feeder reads this frame too !!!
            print(
                f"[INFO] Sync: Resetting position to frame {actual_start_frame} for feeder thread..."
            )
            misc_helpers.seek_frame(self.media_capture, actual_start_frame)
            print("[INFO] Sync: Position reset complete.")

        # 7e. Update counters
        # Internal processing frame numbers live in output frame space when
        # using FFmpeg FPS cap; map accordingly.
        if self._used_ffmpeg_cap:
            self.next_frame_to_display = output_start_frame
            # Keep processing_start_frame in SOURCE-frame space for segment/audio logic
            self.processing_start_frame = actual_start_frame
            self.current_frame_number = output_start_frame
        else:
            self.next_frame_to_display = actual_start_frame
            self.processing_start_frame = actual_start_frame
            self.current_frame_number = actual_start_frame

        # Calculate play_start_time used for audio merging: always in source time.
        if self.recording:
            self.play_start_time = (
                float(actual_start_frame) / float(self.recording_source_fps)
                if self.recording_source_fps > 0
                else 0.0
            )
        else:
            self.play_start_time = (
                float(actual_start_frame) / float(self.fps) if self.fps > 0 else 0.0
            )
        if self.recording:
            print(
                f"[INFO] Recording audio start time set to: {self.play_start_time:.3f}s (Frame: {actual_start_frame})"
            )

        # 7f. Update the slider
        self.main_window.videoSeekSlider.blockSignals(True)
        self.main_window.videoSeekSlider.setValue(actual_start_frame)
        self.main_window.videoSeekSlider.blockSignals(False)

        # --- 8. STARTING THE FEEDER THREAD AND METRONOME VIA MEDIAPIPELINE ---
        # Initialize timing BEFORE starting the metronome to ensure immediate execution.
        self.media_pipeline.last_display_schedule_time_sec = time.perf_counter()

        print(
            f"[INFO] Starting feeder thread via Pipeline (Mode: video, Recording: {self.recording})..."
        )
        self.media_pipeline.start_feeder(mode="video", recording=self.recording)

        if self.recording:
            self.media_pipeline.max_frames_to_display_size = 8
            # Recording: start the display metronome immediately
            print("[INFO] Recording mode: Starting metronome immediately.")
            self.media_pipeline.start_metronome(9999.0, is_first_start=True)
        else:
            if self.main_window.control.get("VideoPlaybackBufferingToggle", False):
                self.media_pipeline.max_frames_to_display_size = (
                    self.preroll_target + 10
                )
                # Playback: start the preroll monitor
                print(
                    f"[INFO] Playback mode: Waiting for preroll buffer (target: {self.preroll_target} frames)..."
                )

                # Ensure the connection is clean
                try:
                    self.media_pipeline.preroll_timer.timeout.disconnect()
                except RuntimeError:
                    pass  # Disconnection failed, which is normal the first time

                self.media_pipeline.preroll_timer.timeout.connect(
                    self.media_pipeline._check_preroll_and_start_playback
                )
                self.media_pipeline.preroll_timer.start(100)
            else:
                self.media_pipeline.max_frames_to_display_size = 8
                print("[INFO] Playback mode. Starting playback.")
                self.media_pipeline._start_synchronized_playback()

    def _cancel_single_frame_preview_state(self):
        """Facade: Forwards single-frame cancellation to the WorkerPoolManager."""
        self.worker_pool_manager.cancel_single_frame_preview_state()

    def _clear_single_frame_preview_caches(self):
        self._last_requested_frame_num = None
        self._cached_raw_frame_media_path = None
        self._cached_raw_frame_number = None
        self._cached_raw_frame_target_height = None
        self._cached_raw_frame_bgr = None
        self._cached_raw_image_path = None
        self._cached_raw_image_target_height = None
        self._cached_raw_image_bgr = None
        self._seek_cached_frame = None

    def start_frame_worker(
        self,
        frame_number,
        frame,
        is_single_frame=False,
        synchronous=False,
        fit_on_complete: bool = False,
    ):
        """Facade: Forwards single-frame UI processing to the WorkerPoolManager."""
        return self.worker_pool_manager.start_single_frame_worker(
            frame_number, frame, is_single_frame, synchronous, fit_on_complete
        )

    def process_current_frame(
        self,
        synchronous: bool = False,
        fit_on_complete: bool = False,
        suppress_raw_preview: bool = False,
    ) -> "FrameWorker | None":
        """
        Process the single, currently selected frame (e.g., after seek or for image).
        This is a one-shot operation, not part of the metronome.

        Args:
            synchronous: If True, blocks until processing is done.
            fit_on_complete: If True, auto-fits the view after generation.
            suppress_raw_preview: If True, skips displaying the unprocessed raw frame
                                  while waiting for the AI worker. Prevents UI flashing.
        """
        # --- WEBCAM LIVE GUARD ---
        # If the webcam is actively streaming, do NOT stop the feed.
        # Simply mark the UI state as dirty so the background pipeline picks up
        # the new parameters (Swap/Edit) on the very next live frame.
        if self.file_type == "webcam" and self.processing:
            self.ui_state_is_dirty = True
            if fit_on_complete:
                from app.ui.widgets.actions import layout_actions

                QTimer.singleShot(
                    0,
                    lambda: layout_actions.fit_image_to_view_onchange(self.main_window),
                )
            return None

        if self.processing or self.is_processing_segments:
            print("[INFO] Stopping active processing to process single frame.")
            if not self.stop_processing():
                print("[WARN] Could not stop active processing cleanly.")

        # Seed global PyTorch/CUDA RNG...
        _denoiser_seed = int(
            self.main_window.control.get("DenoiserBaseSeedSlider", 220)
        )
        torch.manual_seed(_denoiser_seed)
        # PROTECTED: Prevent a 2GB VRAM spike when scrubbing an idle timeline.
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            torch.cuda.manual_seed(_denoiser_seed)

        # Set frame number for processing
        if self.file_type == "video":
            self.current_frame_number = self.main_window.videoSeekSlider.value()
        elif self.file_type == "image" or self.file_type == "webcam":
            self.current_frame_number = 0

        self.next_frame_to_display = self.current_frame_number

        frame_changed = (
            getattr(self, "_last_requested_frame_num", -1) != self.current_frame_number
        )
        self._last_requested_frame_num = self.current_frame_number

        frame_to_process = None
        read_successful = False

        # --- Determine Input Resolution (Global Resize) ---
        target_height = self._get_target_input_height()

        # --- Read the frame based on file type ---
        if self.file_type == "video" and self.media_capture:
            is_cached = (
                self._cached_raw_frame_media_path == self.media_path
                and self._cached_raw_frame_number == self.current_frame_number
                and self._cached_raw_frame_target_height == target_height
                and self._cached_raw_frame_bgr is not None
            )

            if is_cached:
                frame_bgr = self._cached_raw_frame_bgr
                ret = True
            else:
                misc_helpers.seek_frame(self.media_capture, self.current_frame_number)
                ret, frame_bgr = misc_helpers.read_frame(
                    self.media_capture,
                    self.media_rotation,
                    preview_target_height=target_height,
                )
                if ret and frame_bgr is not None:
                    self._cached_raw_frame_media_path = self.media_path
                    self._cached_raw_frame_number = self.current_frame_number
                    self._cached_raw_frame_target_height = target_height
                    self._cached_raw_frame_bgr = frame_bgr.copy()
                    misc_helpers.seek_frame(
                        self.media_capture, self.current_frame_number
                    )

            if ret and frame_bgr is not None:
                # BGR to RGB
                frame_to_process = numpy.ascontiguousarray(frame_bgr[..., ::-1])
                read_successful = True
            else:
                fn = self.current_frame_number
                max_fn = self.max_frame_number
                # Fallback: use the raw frame cached during the last slider seek preview.
                if (
                    self._seek_cached_frame is not None
                    and self._seek_cached_frame[0] == fn
                    and self._seek_cached_frame[1] is not None
                ):
                    cached_frame_bgr = self._seek_cached_frame[1]
                    if (
                        target_height is not None
                        and cached_frame_bgr.shape[0] > target_height
                    ):
                        h, w = cached_frame_bgr.shape[:2]
                        scale = target_height / h
                        cached_frame_bgr = cv2.resize(
                            cached_frame_bgr,
                            (int(w * scale), target_height),
                            interpolation=cv2.INTER_AREA,
                        )
                    frame_to_process = cached_frame_bgr[..., ::-1]  # BGR to RGB
                    read_successful = True
                    misc_helpers.seek_frame(self.media_capture, fn)
                    print(
                        f"[INFO] Using cached slider frame {fn} as fallback for single processing."
                    )
                elif fn >= max_fn - TAIL_TOLERANCE:
                    print(
                        f"[INFO] EOF reached at frame {fn} (reported max={max_fn}), stopping gracefully."
                    )
                    self.current_frame_number = max_fn + 1
                    return None
                else:
                    print(
                        f"[ERROR] Cannot read frame {self.current_frame_number} for single processing!"
                    )
                    self.main_window.last_seek_read_failed = True

        elif self.file_type == "image":
            is_cached = (
                self._cached_raw_image_path == self.media_path
                and self._cached_raw_image_target_height == target_height
                and self._cached_raw_image_bgr is not None
            )

            if is_cached:
                frame_bgr = self._cached_raw_image_bgr
            else:
                frame_bgr = misc_helpers.read_image_file(self.media_path)
                if frame_bgr is not None:
                    self._cached_raw_image_path = self.media_path
                    self._cached_raw_image_target_height = target_height
                    self._cached_raw_image_bgr = frame_bgr.copy()

            if frame_bgr is not None:
                if target_height is not None and frame_bgr.shape[0] > target_height:
                    h, w = frame_bgr.shape[:2]
                    scale = target_height / h
                    new_w = int(w * scale)
                    frame_bgr = cv2.resize(
                        frame_bgr, (new_w, target_height), interpolation=cv2.INTER_AREA
                    )

                frame_to_process = numpy.ascontiguousarray(
                    frame_bgr[..., ::-1]
                )  # BGR to RGB
                read_successful = True
            else:
                print("[ERROR] Unable to read image file for processing.")

        elif self.file_type == "webcam" and self.media_capture:
            ret, frame_bgr = misc_helpers.read_frame(
                self.media_capture, 0, preview_target_height=None
            )
            if ret and frame_bgr is not None:
                frame_to_process = numpy.ascontiguousarray(
                    frame_bgr[..., ::-1]
                )  # BGR to RGB
                read_successful = True
            else:
                print("[ERROR] Unable to read Webcam frame for processing!")

        # --- Process if read was successful ---
        if read_successful and frame_to_process is not None:
            # Check if the UI is currently simulating a navigation step
            is_stepping = getattr(self.main_window, "_is_stepping_media", False)
            is_compare_active = getattr(
                self.main_window, "view_face_compare_enabled", False
            )
            is_mask_active = getattr(self.main_window, "view_face_mask_enabled", False)

            # Block the raw image preview IF explicitly requested (e.g., Stop button)
            # OR IF we are actively stepping through navigation with a special preview mode active
            force_suppression = suppress_raw_preview or (
                is_stepping and (is_compare_active or is_mask_active)
            )

            if frame_changed and not force_suppression:
                frame_bgr_preview = numpy.ascontiguousarray(frame_to_process[..., ::-1])
                self.display_current_frame(
                    generation=0,
                    frame_number=self.current_frame_number,
                    frame=frame_bgr_preview,
                    preview_cache=None,
                )

            return self.start_frame_worker(
                self.current_frame_number,
                frame_to_process,
                is_single_frame=True,
                synchronous=synchronous,
                fit_on_complete=fit_on_complete,
            )

        return None

    @Slot(str)
    def _handle_fatal_processing_error(self, reason: str) -> None:
        if self._fatal_processing_error_latched:
            return
        self._fatal_processing_error_latched = True
        self.last_processing_error = reason
        self.stop_processing()

    def stop_processing(self) -> bool:
        """
        General Stop / Abort Function.
        This is the master function to stop *any* active processing
        (playback, recording, segments, webcam).

        Returns:
            True if any active processing was stopped or a broken capture was recovered.
        """
        # Step 0: Capture current state for return value and cleanup logic
        was_active = self.processing or self.is_processing_segments or self.recording
        was_recording_default_style = self.recording
        was_processing_segments = self.is_processing_segments

        # VP-34: Check if capture is missing/broken while idle. If so, fix it.
        if not was_active:
            self._cancel_single_frame_preview_state()
            self._clear_single_frame_preview_caches()
            self._stop_recording_ffmpeg_input_stream()
            if self.file_type == "video" and self.media_path:
                if not self.media_capture or not self.media_capture.isOpened():
                    print(
                        "[INFO] stop_processing: Capture missing/closed while idle. Recovering..."
                    )
                    self._reopen_video_capture(self.main_window.videoSeekSlider.value())
                    video_control_actions.reset_media_buttons(self.main_window)
                    return True
            video_control_actions.reset_media_buttons(self.main_window)
            return False  # Nothing was active and capture seems OK

        print("[INFO] Aborting active processing...")

        # Purge pending model unloads
        self.main_window.models_processor.execute_all_deferred_unloads()

        # 1. Reset flags FIRST to stop all loops immediately.
        # VP-29: Set recording=False early to prevent further frames from being
        # dispatched to FFmpeg by concurrent worker threads.
        self.processing = False
        self.is_processing_segments = False
        self.recording = False
        self.tail_pending_stall_start_sec = 0.0
        self.tail_force_finalize_due_to_stall = False
        self.triggered_by_job_manager = False
        self.active_output_folder = ""
        self._cancel_single_frame_preview_state()

        # 2. Stop utility timers and audio
        self.gpu_memory_update_timer.stop()
        self.preroll_timer.stop()
        self.stop_live_sound()
        self._stop_recording_ffmpeg_input_stream()

        # Face tracker defaults (use thread-safe reset from new manager)
        self.sequential_detector.reset_state()

        # 3a. Release the capture object to unblock the feeder.
        # The feeder calls read_frame() in a loop; releasing here causes the next read
        # to fail immediately, driving the feeder's EOF branch and exit.
        print("[INFO] Releasing media capture to unblock feeder thread...")
        if self.media_capture:
            misc_helpers.release_capture(self.media_capture)
            self.media_capture = None

        # 3b. Wait for the producer threads to fully exit.
        print("[INFO] Waiting for producer threads to complete...")
        if self.feeder_thread and self.feeder_thread.is_alive():
            self.feeder_thread.join(timeout=3.0)
            if self.feeder_thread.is_alive():
                print("[WARN] Feeder thread did not join gracefully within 3s timeout.")
        self.feeder_thread = None

        if self.detector_thread and self.detector_thread.is_alive():
            self.detector_thread.join(timeout=3.0)
            if self.detector_thread.is_alive():
                print(
                    "[WARN] Detector thread did not join gracefully within 3s timeout."
                )
        self.detector_thread = None
        print("[INFO] Producer threads joined.")

        # 3c. Clear display buffers and join worker threads.
        # VP-24: We clear the queue and then send poison pills to wake workers
        # blocked on queue.get().
        for key in list(self.frames_to_display.keys()):
            arr = self.frames_to_display.pop(key)
            del arr
        self.frames_to_display.clear()
        self._clear_single_frame_preview_caches()
        while not self.webcam_frames_to_display.empty():
            try:
                arr = self.webcam_frames_to_display.get_nowait()
                del arr
            except queue.Empty:
                break
        with self.frame_queue.mutex:
            self.frame_queue.queue.clear()

        # --- Clear the raw frame queue ---
        if hasattr(self, "media_pipeline") and hasattr(
            self.media_pipeline, "raw_frame_queue"
        ):
            with self.media_pipeline.raw_frame_queue.mutex:
                self.media_pipeline.raw_frame_queue.queue.clear()

        print("[INFO] Waiting for worker threads to complete...")
        self.join_and_clear_threads()
        print("[INFO] Worker threads joined.")

        # 5. Stop and cleanup FFmpeg encoder
        if self.encoder.is_running():
            print("[INFO] Closing and waiting for active FFmpeg encoder...")
            self.encoder.close_process()

        # 6. Cleanup temp files based on stopped mode.
        if was_processing_segments:
            print("[INFO] Cleaning up segment temporary directory due to abort.")
            self._cleanup_temp_dir()
        elif was_recording_default_style:
            print("[INFO] Cleaning up default-style temporary file due to abort.")
            if self.temp_file and os.path.exists(self.temp_file):
                try:
                    os.remove(self.temp_file)
                    print(f"[INFO] Removed temporary file: {self.temp_file}")
                except OSError as e:
                    print(
                        f"[WARN] Could not remove temp file {self.temp_file} during abort: {e}"
                    )
            self.temp_file = ""

        # 7. Reset segment state
        self.segments_to_process = []
        self.current_segment_index = -1
        self.temp_segment_files = []
        self.current_segment_end_frame = None
        self.playback_display_start_time = 0.0

        # We ensure state is completely cleared on full abort
        self.sequential_detector.reset_state()

        # 8. RE-OPEN media capture IMMEDIATELY.
        # VP-34: This is critical. By ensuring media_capture is re-opened before
        # returning, we ensure that on_change_video_seek_slider() (which calls
        # stop_processing() first) can still read a frame for the preview.
        if self.file_type == "video" and self.media_path:
            last_processed = self.next_frame_to_display - 1
            start_frame = getattr(self, "processing_start_frame", 0)
            if self._used_ffmpeg_cap and self.fps > 0 and self.recording_source_fps > 0:
                last_processed = self.output_to_source_frame(last_processed)

            # --- Stop Revert Bug ---
            # Do NOT use max(start_frame, last_processed) as it breaks seamless looping.
            # Only revert to start_frame if absolutely no frames were displayed.
            if self.next_frame_to_display == getattr(self, "processing_start_frame", 0):
                current_slider_pos = start_frame
            else:
                current_slider_pos = max(0, last_processed)

            # Slider stays in source frame space (approach 2).
            # If FPS-cap recording was active, map output frame -> source frame before seek.
            src_slider_max = self.main_window.videoSeekSlider.maximum()
            current_slider_pos = min(current_slider_pos, src_slider_max)
            if self._reopen_video_capture(current_slider_pos):
                # Restore max_frame_number/fps to source space after FPS-cap recording.
                if was_recording_default_style:
                    self._restore_source_frame_state_after_capture_reopen()
                self.main_window.videoSeekSlider.blockSignals(True)
                self.main_window.videoSeekSlider.setValue(current_slider_pos)
                self.main_window.videoSeekSlider.blockSignals(False)
                print(
                    f"[INFO] Video capture re-opened and seeked to {current_slider_pos} after stop."
                )
            else:
                print("[WARN] Failed to re-open media capture after active stop.")
        elif self.file_type == "webcam":
            # For webcam, re-opening essentially prepares it for the next 'Play' click.
            try:
                webcam_index = int(
                    self.main_window.control.get("WebcamDeviceSelection", 0)
                )

                backend_name = str(
                    self.main_window.control.get("WebcamBackendSelection", "Default")
                )
                backend_id = CAMERA_BACKENDS.get(backend_name, cv2.CAP_ANY)

                self.media_capture = cv2.VideoCapture(webcam_index, backend_id)

                if self.media_capture.isOpened():
                    try:
                        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                        self.media_capture.set(cv2.CAP_PROP_FOURCC, fourcc)
                    except Exception:
                        pass

                    res_str = str(
                        self.main_window.control.get(
                            "WebcamMaxResSelection", "1280x720"
                        )
                    )
                    target_width, target_height = map(int, res_str.split("x"))
                    self.media_capture.set(cv2.CAP_PROP_FRAME_WIDTH, target_width)
                    self.media_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, target_height)
                else:
                    print("[WARN] Failed to re-open webcam capture after stop.")
                    self.media_capture = None
            except Exception as e:
                print(f"[WARN] Error re-opening webcam capture: {e}")
                self.media_capture = None

        # 9. Final cleanup and UI reset
        layout_actions.enable_all_parameters_and_control_widget(self.main_window)
        video_control_actions.reset_media_buttons(self.main_window)

        print("[INFO] Clearing GPU Cache and running garbage collection.")
        try:
            if torch.cuda.is_available() and torch.cuda.is_initialized():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        except Exception as e:
            print(f"[WARN] Error clearing Torch cache: {e}")
        gc.collect()

        try:
            self.disable_virtualcam()
        except Exception:
            pass

        # compute end metrics using helper
        self.play_end_time, end_frame_for_calc, frames_actually_processed, duration = (
            self._compute_play_end()
        )
        if duration is not None:
            print(
                f"[INFO] Probed temp video duration during abort: {duration:.3f}s (recorded clip length), "
                f"play_end_time set to {self.play_end_time:.3f}s [media time]."
            )
        else:
            print(
                f"[INFO] Calculated recording end time (frame estimate) during abort: {self.play_end_time:.3f}s (based on frame {end_frame_for_calc})"
            )

        # 11. Final Timing and Logging
        self.end_time = time.perf_counter()
        processing_time_sec = self.end_time - self.start_time

        try:
            # We now fetch the absolute frames from the media pipeline
            # to guarantee accurate FPS during looping and segment jumps.
            num_frames_processed = getattr(
                self.media_pipeline, "absolute_frames_processed", 0
            )
        except Exception:
            num_frames_processed = 0

        self._log_processing_summary(processing_time_sec, num_frames_processed)

        # MP-REFRESH: Force a refresh of the current frame to match current UI state.
        # This prevents confusion if parameters were changed but not yet processed
        # by a worker before the manual stop.
        if self.file_type in ["video", "image"] and not (
            was_recording_default_style or was_processing_segments
        ):
            print(
                "[INFO] Stop Processing: Triggering final frame refresh to match UI state (raw preview suppressed)."
            )
            # We call this asynchronously to let the UI finish its current state cleanup first.
            # suppress_raw_preview=True ensures the UI doesn't flash the original image while computing.
            self.process_current_frame(synchronous=False, suppress_raw_preview=True)

        self.processing_stopped_signal.emit()
        self._used_ffmpeg_cap = False

        return True  # Processing was stopped

    def join_and_clear_threads(self, clear_module_caches: bool = True):
        """Facade: Delegates thread synchronization and VRAM cleanup to WorkerPoolManager."""
        self.worker_pool_manager.join_and_clear_threads(clear_module_caches)

    def _log_hevc_thumbnail_hint_once(self) -> None:
        """Print a one-time hint about HEVC thumbnail rendering on Windows 10.

        Default recording codec is HEVC (hevc_nvenc / libx265). Windows 10 does
        not generate File Explorer thumbnails for HEVC files unless the user
        installs the "HEVC Video Extensions" package from the Microsoft Store.
        This hint surfaces the workaround so users don't think VisoMaster broke
        their thumbnails.
        """
        if getattr(self, "_hevc_hint_logged", False):
            return
        self._hevc_hint_logged = True
        if os.name == "nt":
            print(
                "[INFO] Recording finished as HEVC (H.265). "
                "Windows Explorer thumbnails for HEVC require the "
                "'HEVC Video Extensions' from the Microsoft Store."
            )

    def _reopen_video_capture(self, seek_frame: int = 0) -> bool:
        """
        Private helper to robustly re-open the video capture.
        Performs up to 3 attempts with a test read to ensure the capture is
        actually functional (not just 'open' according to OpenCV).
        """
        if not self.media_path:
            return False

        for attempt in range(3):
            try:
                print(f"[INFO] Re-opening video capture (attempt {attempt + 1})...")
                # First ensure any existing capture is released
                if self.media_capture:
                    misc_helpers.release_capture(self.media_capture)
                    self.media_capture = None

                self.media_capture = cv2.VideoCapture(self.media_path)
                # Explicitly enable OpenCV's auto-rotation to let it handle metadata natively
                if hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO"):
                    self.media_capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
                if self.media_capture and self.media_capture.isOpened():
                    # PERFORM TEST READ: essential on Windows to detect silent handle failures
                    misc_helpers.seek_frame(self.media_capture, seek_frame)
                    ret, _ = self.media_capture.read()
                    if ret:
                        # Success! Reset counters and seek back to the target frame.
                        self.current_frame_number = seek_frame
                        self.next_frame_to_display = seek_frame
                        misc_helpers.seek_frame(self.media_capture, seek_frame)
                        print(
                            f"[INFO] Video capture re-opened and verified at frame {seek_frame}."
                        )
                        return True
                    else:
                        print(
                            f"[WARN] Attempt {attempt + 1}: Capture is open but read() failed."
                        )
                        seek_frame = max(0, seek_frame - 1)
                else:
                    print(
                        f"[WARN] Attempt {attempt + 1}: VideoCapture.isOpened() is False."
                    )
            except Exception as e:
                print(f"[WARN] Attempt {attempt + 1}: Exception during re-open: {e}")

            # Cleanup before retry
            if self.media_capture:
                misc_helpers.release_capture(self.media_capture)
                self.media_capture = None
            time.sleep(0.2)

        print("[ERROR] Failed to re-open functional video capture after 3 attempts.")
        return False

    @staticmethod
    def _scaled_dimensions_for_height(
        src_width: int, src_height: int, target_height: int
    ) -> tuple[int, int]:
        """Compute aspect-preserving output dimensions with even alignment."""
        if src_width <= 0 or src_height <= 0 or target_height <= 0:
            return src_width, src_height

        out_height = max(2, int(target_height))
        if out_height % 2 != 0:
            out_height += 1

        out_width = max(2, int(round(src_width * (out_height / float(src_height)))))
        if out_width % 2 != 0:
            out_width += 1

        return out_width, out_height

    def _start_recording_ffmpeg_input_stream(
        self,
        start_frame: int,
        target_fps: float,
        target_height: Optional[int],
    ) -> bool:
        """Start FFmpeg rawvideo stream for recording FPS-cap mode."""
        if not self.media_path:
            print("[ERROR] Cannot start FFmpeg input stream: media path is missing.")
            return False

        if target_fps <= 0:
            print("[ERROR] Cannot start FFmpeg input stream: target FPS is invalid.")
            return False

        self._stop_recording_ffmpeg_input_stream()

        src_w = (
            int(self.media_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            if self.media_capture
            else 0
        )
        src_h = (
            int(self.media_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if self.media_capture
            else 0
        )

        if src_w <= 0 or src_h <= 0:
            print(
                "[ERROR] Cannot start FFmpeg input stream: source dimensions are invalid."
            )
            return False

        out_w, out_h = src_w, src_h
        vf_filters = [f"fps={target_fps:.6f}"]

        if target_height and target_height > 0:
            out_w, out_h = self._scaled_dimensions_for_height(
                src_w, src_h, target_height
            )
            if out_w != src_w or out_h != src_h:
                vf_filters.append(
                    f"scale={out_w}:{out_h}:flags=lanczos+accurate_rnd+full_chroma_int"
                )

        args = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-fflags",
            "+discardcorrupt",
            "-err_detect",
            "ignore_err",
        ]

        if start_frame > 0 and target_fps > 0:
            # start_frame is in output frame space; seek to the equivalent timestamp.
            start_time_sec = float(start_frame) / float(target_fps)
            args.extend(["-ss", f"{start_time_sec:.6f}"])

        args.extend(
            [
                "-i",
                str(self.media_path),
                "-an",
                "-sn",
                "-vf",
                ",".join(vf_filters),
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "pipe:1",
            ]
        )

        try:
            self.ffmpeg_input_sp = subprocess.Popen(
                args,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=10**7,
            )
            self.ffmpeg_input_width = out_w
            self.ffmpeg_input_height = out_h
            # Mark that this processing session is using the FFmpeg FPS-cap path.
            self._used_ffmpeg_cap = True
            self.ffmpeg_input_prefetched_frame = None
            print(
                f"[INFO] Recording input stream enabled via FFmpeg: {out_w}x{out_h} @ {target_fps:.3f}fps"
            )
            return True
        except FileNotFoundError:
            print("[ERROR] FFmpeg not found while starting recording input stream.")
            self.ffmpeg_input_sp = None
            self._used_ffmpeg_cap = False
            return False
        except Exception as e:
            print(f"[ERROR] Failed to start FFmpeg recording input stream: {e}")
            self.ffmpeg_input_sp = None
            self._used_ffmpeg_cap = False
            return False

    def _read_frame_from_ffmpeg_input_stream(
        self,
    ) -> tuple[bool, Optional[numpy.ndarray]]:
        """Read one BGR frame from FFmpeg rawvideo stdout."""
        if self.ffmpeg_input_prefetched_frame is not None:
            frame = self.ffmpeg_input_prefetched_frame
            self.ffmpeg_input_prefetched_frame = None
            return True, frame

        if (
            not self.ffmpeg_input_sp
            or not self.ffmpeg_input_sp.stdout
            or self.ffmpeg_input_width <= 0
            or self.ffmpeg_input_height <= 0
        ):
            return False, None

        frame_size = self.ffmpeg_input_width * self.ffmpeg_input_height * 3
        try:
            raw = self.ffmpeg_input_sp.stdout.read(frame_size)
        except Exception:
            return False, None

        if not raw or len(raw) != frame_size:
            return False, None

        frame = numpy.frombuffer(raw, dtype=numpy.uint8).reshape(
            (self.ffmpeg_input_height, self.ffmpeg_input_width, 3)
        )
        return True, frame.copy()

    def _stop_recording_ffmpeg_input_stream(self) -> None:
        """Stop and cleanup FFmpeg recording input stream process."""
        proc = self.ffmpeg_input_sp
        if not proc:
            self.ffmpeg_input_width = 0
            self.ffmpeg_input_height = 0
            self.ffmpeg_input_prefetched_frame = None
            return

        try:
            if proc.stdout and not proc.stdout.closed:
                proc.stdout.close()
        except Exception:
            pass

        try:
            if proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            try:
                proc.kill()
                proc.wait(timeout=2.0)
            except Exception:
                pass
        except Exception:
            pass

        self.ffmpeg_input_sp = None
        self.ffmpeg_input_width = 0
        self.ffmpeg_input_height = 0
        self.ffmpeg_input_prefetched_frame = None

    def _restore_source_frame_state_after_capture_reopen(self) -> None:
        """After re-opening `media_capture`, refresh `max_frame_number` and `fps`
        from the reopened capture (source-frame space). This centralizes the
        restoration logic so callers don't duplicate the probe/assignment code.
        """
        try:
            if self.media_capture and self.media_capture.isOpened():
                src_count = int(self.media_capture.get(cv2.CAP_PROP_FRAME_COUNT))
                src_fps_val = self.media_capture.get(cv2.CAP_PROP_FPS)
                if src_count > 0 and src_count - 1 != self.max_frame_number:
                    self.max_frame_number = src_count - 1
                if src_fps_val > 0:
                    self.fps = src_fps_val
        except Exception:
            pass

    def source_to_output_frame(
        self,
        source_frame: int,
        src_fps: float | None = None,
        out_fps: float | None = None,
    ) -> int:
        """Map a source-frame index to output-frame index using fps values.

        Falls back to `self.recording_source_fps` and `self.fps` when fps args
        are not provided. Returns an integer rounded value >= 0.
        """
        try:
            sf = int(source_frame)
        except Exception:
            return 0
        src = (
            float(src_fps)
            if src_fps is not None
            else float(self.recording_source_fps or 0)
        )
        out = float(out_fps) if out_fps is not None else float(self.fps or 0)
        if src <= 0 or out <= 0:
            return sf
        return max(0, round(float(sf) * out / src))

    def output_to_source_frame(
        self,
        output_frame: int,
        src_fps: float | None = None,
        out_fps: float | None = None,
    ) -> int:
        """Map an output-frame index back to source-frame index using fps values.

        Falls back to `self.recording_source_fps` and `self.fps` when fps args
        are not provided. Returns an integer rounded value >= 0.
        """
        try:
            of = int(output_frame)
        except Exception:
            return 0
        src = (
            float(src_fps)
            if src_fps is not None
            else float(self.recording_source_fps or 0)
        )
        out = float(out_fps) if out_fps is not None else float(self.fps or 0)
        if src <= 0 or out <= 0:
            return of
        return max(0, round(float(of) * src / out))

    def _safe_unfinished_tasks(self) -> int:
        """Return a safe estimate of unfinished tasks on the frame_queue.

        Uses getattr to avoid attribute errors on non-standard queue types.
        """
        try:
            return int(max(0, getattr(self.frame_queue, "unfinished_tasks", 0)))
        except Exception:
            return 0

    # --- Utility Methods ---

    def _format_duration(self, total_seconds: float) -> str:
        """
        Converts a duration in seconds to a human-readable string (e.g., 1h 15m 30.55s).

        :param total_seconds: The duration in seconds.
        :return: A formatted string.
        """
        try:
            total_seconds = float(total_seconds)

            hours = int(total_seconds // 3600)
            minutes = int((total_seconds % 3600) // 60)
            seconds = total_seconds % 60

            parts = []
            if hours > 0:
                parts.append(f"{hours}h")
            if minutes > 0 or (hours > 0 and seconds == 0):
                parts.append(f"{minutes}m")

            # Always show seconds
            if hours > 0 or minutes > 0:
                # Show 2 decimal places if we also show hours/minutes
                parts.append(f"{seconds:05.2f}s")
            else:
                # Show 3 decimal places if it's only seconds
                parts.append(f"{seconds:.3f}s")

            return " ".join(parts)
        except Exception:
            # Fallback in case of an error
            return f"{total_seconds:.3f} seconds"

    def _apply_job_timestamp_to_output_name(
        self,
        was_triggered_by_job: bool,
        job_name: Optional[str],
        use_job_name: bool,
        output_file_name: Optional[str],
    ) -> tuple[Optional[str], Optional[str]]:
        """Appends the standard output timestamp to job-driven names."""
        if not was_triggered_by_job:
            return job_name, output_file_name

        timestamp = datetime.now().strftime(r"%Y_%m_%d_%H_%M_%S")
        if use_job_name and job_name:
            job_name = f"{job_name}_{timestamp}"
        elif output_file_name:
            output_file_name = f"{output_file_name}_{timestamp}"

        return job_name, output_file_name

    def _log_processing_summary(
        self, processing_time_sec: float, num_frames_processed: int
    ):
        """
        Calculates and prints the final processing time and average FPS.
        Uses the actual display duration for FPS calculation if playback occurred.
        """

        # 1. Print formatted duration (overall processing time)
        formatted_duration = self._format_duration(processing_time_sec)
        print(f"\n[INFO] Processing completed in {formatted_duration}")

        # 2. Calculate and print FPS (based on actual display time)
        display_duration_sec = 0.0
        # Check if playback actually started displaying frames
        if (
            self.playback_display_start_time > 0
            and self.end_time > self.playback_display_start_time
        ):
            display_duration_sec = self.end_time - self.playback_display_start_time
            print(
                f"[INFO] (Actual display duration: {self._format_duration(display_duration_sec)})"
            )
        else:
            # Playback might have stopped during preroll or it was a recording-only task
            # Use the overall time, but mention it includes setup/buffering
            display_duration_sec = processing_time_sec
            if (
                self.start_time != self.playback_display_start_time
            ):  # Check if display never started
                print(
                    "[INFO] (Note: FPS calculation includes initial buffering/setup time)"
                )

        try:
            if (
                display_duration_sec > 0.01 and num_frames_processed > 0
            ):  # Use a small threshold for duration
                avg_fps = num_frames_processed / display_duration_sec
                print(f"[INFO] Average Display FPS: {avg_fps:.2f}\n")
            elif num_frames_processed > 0:
                print(
                    "[WARN] Display duration too short to calculate meaningful FPS.\n"
                )
            else:
                print(
                    "[WARN] No frames were displayed or duration was zero, cannot calculate FPS.\n"
                )
        except Exception as e:
            print(f"[WARN] Could not calculate average FPS: {e}\n")

    def _prepare_default_temp_file(self) -> str:
        """
        Prepares the temporary directory and generates a temp file path for default recording.
        Cleans up orphaned temp files from previous crashed sessions.
        """
        date_and_time = datetime.now().strftime(r"%Y_%m_%d_%H_%M_%S")
        try:
            base_temp_dir = os.path.join(os.getcwd(), "temp_files", "default")
            os.makedirs(base_temp_dir, exist_ok=True)

            try:
                _cutoff = time.time() - 86400  # 24 hours
                for _stale in Path(base_temp_dir).glob("temp_output_*.mp4"):
                    try:
                        if _stale.stat().st_mtime < _cutoff:
                            _stale.unlink()
                            print(f"[INFO] Removed stale temp file: {_stale.name}")
                    except OSError:
                        pass

                _stale_audio_dir = Path(base_temp_dir) / "temp_audio"
                if _stale_audio_dir.is_dir():
                    for _stale_audio_file in _stale_audio_dir.iterdir():
                        try:
                            if _stale_audio_file.stat().st_mtime < _cutoff:
                                if _stale_audio_file.is_dir():
                                    import shutil

                                    shutil.rmtree(_stale_audio_file, ignore_errors=True)
                                else:
                                    _stale_audio_file.unlink()
                        except OSError:
                            pass
            except Exception:
                pass  # Non-critical; never block recording startup

            temp_path = os.path.join(base_temp_dir, f"temp_output_{date_and_time}.mp4")
            print(f"[INFO] Default temp file will be created at: {temp_path}")
            return temp_path
        except Exception as e:
            print(f"[ERROR] Failed to create temporary directory/file path: {e}")
            return f"temp_output_{date_and_time}.mp4"

    def _identify_frame_segments(self, actual_end_frame: int) -> List[Tuple[int, int]]:
        """
        Identify all continuous segments of successfully processed frames.
        Returns a list of (start_frame, end_frame) tuples, using absolute frame
        numbers from the original media.  When recording began partway through
        the source the first segment needs to start at ``self.processing_start_frame``
        rather than zero.

        Args:
            actual_end_frame: The actual last frame that was recorded (0-based,
                              absolute index in source)

        Example: if recording started at frame 100 and frames 150, 175 were
        skipped in a 200-frame video:
        Returns: [(100, 149), (151, 174), (176, 199)]
        """
        # Determine the first frame we actually processed (may be >0 if we
        # sought).  Default to 0 for standard playback/recordings that start
        # at the beginning.
        start_frame = getattr(self, "processing_start_frame", 0) or 0

        if not self.skipped_frames:
            # No skipped frames - single segment from start_frame to end
            return [(start_frame, actual_end_frame)]

        # Sort skipped frames and ignore any that occur before start_frame
        sorted_skipped = [f for f in sorted(self.skipped_frames) if f >= start_frame]
        segments = []
        segment_start = start_frame

        for skipped_frame in sorted_skipped:
            if skipped_frame > segment_start:
                # Frames from segment_start to skipped_frame-1 are successful
                segment_end = skipped_frame - 1
                if segment_start <= segment_end:
                    segments.append((segment_start, segment_end))
            # Next segment starts after the skipped frame
            segment_start = skipped_frame + 1

        # Add final segment if there are frames after the last skipped frame
        if segment_start <= actual_end_frame:
            segments.append((segment_start, actual_end_frame))

        # summary only; detailed segment listings are rarely needed and can
        # clutter the console.  If fuller diagnostics are required the
        # developer can re-enable by inspecting `self.skipped_frames` directly.
        print(f"[INFO] Identified {len(segments)} continuous frame segment(s)")

        return segments

    def _get_issue_scan_ranges(self) -> List[Tuple[int, int]]:
        """Facade: Forwards scan range calculation to the decoupled IssueScanner."""
        return self._get_issue_scanner_instance()._get_issue_scan_ranges()

    def describe_issue_scan_scope(
        self, scan_ranges: Optional[List[Tuple[int, int]]] = None
    ) -> str:
        """Facade: Forwards scope description to the decoupled IssueScanner."""
        return self._get_issue_scanner_instance().describe_issue_scan_scope(scan_ranges)

    @staticmethod
    def _compute_longest_issue_run(issue_frames: list[int]) -> int:
        """Facade: Forwards calculation to the decoupled IssueScanner."""
        return IssueScanner._compute_longest_issue_run(issue_frames)

    def prepare_issue_scan_target_faces_snapshot(
        self,
        scan_ranges: list[tuple[int, int]],
        base_control: ControlTypes,
        base_params: FacesParametersTypes,
        control_defaults_snapshot: Optional[ControlTypes] = None,
    ) -> IssueScanTargetSnapshot:
        return (
            self._get_issue_scanner_instance().prepare_issue_scan_target_faces_snapshot(
                scan_ranges, base_control, base_params, control_defaults_snapshot
            )
        )

    def scan_issue_frames(
        self,
        progress_callback=None,
        issue_found_callback=None,
        is_cancelled=None,
        scan_ranges: Optional[List[Tuple[int, int]]] = None,
        target_height: Optional[int] = None,
        base_control: Optional[dict] = None,
        base_params: Optional[dict] = None,
        target_faces_snapshot: Optional[IssueScanTargetSnapshot] = None,
        control_defaults_snapshot: Optional[dict] = None,
        reset_frame_number: Optional[int] = None,
    ) -> Optional[dict]:
        try:
            return self._get_issue_scanner_instance().scan_issue_frames(
                progress_callback,
                issue_found_callback,
                is_cancelled,
                scan_ranges,
                target_height,
                base_control,
                base_params,
                target_faces_snapshot,
                control_defaults_snapshot,
            )
        finally:
            # The scanner walks the media with its own capture, so restore the
            # live playback position afterwards. Without this the next play or
            # seek resumes from the last scanned frame.
            self.current_frame_number = (
                reset_frame_number
                if reset_frame_number is not None
                else int(self.main_window.videoSeekSlider.value())
            )

    def _probe_video_duration(self, file_path: str) -> float | None:
        """
        Return the duration (in seconds) of the video file at `file_path` using
        ffprobe.  If probing fails for any reason the function returns None.
        """
        if not file_path or not os.path.isfile(file_path):
            return None
        try:
            args = [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                file_path,
            ]
            result = subprocess.run(args, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                return None
            duration_str = result.stdout.strip()
            return float(duration_str) if duration_str else None
        except Exception as e:
            print(f"[WARN] Failed to probe video duration for {file_path}: {e}")
            return None

    def _compute_play_end(self) -> Tuple[float, int, int, float | None]:
        """Compute timing values used when finalizing a recording.

        Returns a tuple of:
          (play_end_time, end_frame_for_calc, frames_actually_processed, duration_probed)

        ``duration_probed`` is the length of the temp video file if probing
        succeeded, otherwise ``None``.  ``play_end_time`` is always an absolute
        timestamp in the original media timeline (i.e. includes ``play_start_time``).
        """
        end_frame = min(self.next_frame_to_display, self.max_frame_number + 1)
        frames_processed = end_frame - self.total_skipped_frames

        duration = None
        if self.temp_file and Path(self.temp_file).is_file():
            duration = self._probe_video_duration(self.temp_file)

        if duration is not None:
            play_end = self.play_start_time + duration
        elif self.frames_written > 0 and self.fps > 0:
            play_end = self.play_start_time + (self.frames_written / float(self.fps))
        else:
            play_end = float(end_frame / float(self.fps)) if self.fps > 0 else 0.0

        return play_end, end_frame, frames_processed, duration

    def _attempt_segment_video_only_fallback(
        self, list_file_path: str, final_file_path: str, failure_message: str
    ) -> bool:
        """Try segment video-only concat fallback and show UI error if it fails."""
        print("[WARN] Attempting segment video-only fallback concatenation...")
        if FFmpegPostProcessor.concatenate_segments_video_only(
            list_file_path, final_file_path
        ):
            return True

        self.main_window.display_messagebox_signal.emit(
            "Recording Error",
            failure_message,
            self.main_window,
        )
        return False

    def _rebuild_segment_audio_if_needed(self, segment_num: int) -> None:
        """Rebuild current segment audio from kept frame ranges when frames were skipped."""
        if not (
            self.total_skipped_frames > 0
            and self.temp_segment_files
            and self.current_segment_index >= 0
            and self.current_segment_index < len(self.segments_to_process)
        ):
            return

        current_segment_path = self.temp_segment_files[-1]
        if not (
            os.path.exists(current_segment_path)
            and os.path.getsize(current_segment_path) > 0
            and self.segment_temp_dir
        ):
            return

        start_frame, end_frame = self.segments_to_process[self.current_segment_index]
        actual_end_frame = (
            self.last_displayed_frame
            if self.last_displayed_frame is not None
            else end_frame
        )

        if actual_end_frame < start_frame:
            print(
                f"[WARN] Segment {segment_num}: invalid frame range for audio correction ({start_frame}..{actual_end_frame})."
            )
            return

        temp_audio_dir = os.path.join(
            self.segment_temp_dir,
            f"segment_audio_{self.current_segment_index:03d}_{uuid.uuid4().hex}",
        )
        os.makedirs(temp_audio_dir, exist_ok=True)

        previous_start_frame = getattr(self, "processing_start_frame", 0)
        try:
            self.processing_start_frame = start_frame
            keep_segments = self._identify_frame_segments(actual_end_frame)
        finally:
            self.processing_start_frame = previous_start_frame

        try:
            print(
                f"[INFO] Segment {segment_num}: rebuilding audio for skipped frames "
                f"(manual dropped={self.manual_dropped_skip_count}, read errors={self.read_error_skip_count})."
            )
            audio_ok, audio_files = FFmpegPostProcessor.extract_audio_segments(
                media_path=str(self.media_path),
                fps=self.recording_source_fps,
                segments=keep_segments,
                temp_audio_dir=temp_audio_dir,
            )
            if not (audio_ok and audio_files):
                print(
                    f"[WARN] Segment {segment_num}: audio extraction failed during skip correction, keeping original segment audio."
                )
                return

            corrected_audio = FFmpegPostProcessor.concatenate_audio_segments(
                audio_files=audio_files, temp_audio_dir=temp_audio_dir
            )
            if not corrected_audio:
                print(
                    f"[WARN] Segment {segment_num}: corrected audio concatenation failed, keeping original segment audio."
                )
                return

            remuxed_segment_path = os.path.join(
                self.segment_temp_dir,
                f"segment_{self.current_segment_index:03d}_synced.mp4",
            )
            args = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                current_segment_path,
                "-i",
                corrected_audio,
                "-c:v",
                "copy",
                "-c:a",
                "copy",
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-shortest",
                "-y",
                remuxed_segment_path,
            ]
            subprocess.run(args, check=True)
            os.replace(remuxed_segment_path, current_segment_path)
            print(
                f"[INFO] Segment {segment_num}: rebuilt audio after skipping {self.total_skipped_frames} frame(s)."
            )
        except Exception as e:
            print(
                f"[WARN] Segment {segment_num}: failed to rebuild synced audio ({e}), keeping original segment audio."
            )
        finally:
            shutil.rmtree(temp_audio_dir, ignore_errors=True)

    def _auto_save_workspace_for_output(self, final_file_path: str) -> None:
        if not final_file_path:
            return

        if self.main_window.control.get("AutoSaveWorkspaceToggle"):
            try:
                save_load_actions.save_current_workspace(
                    self.main_window, f"{final_file_path}.json"
                )
            except Exception as e:
                print(f"[WARN] Failed to auto-save workspace after recording: {e}")

        if self.main_window.control.get("AutoSaveLastWorkspaceToggle"):
            try:
                save_load_actions.save_current_workspace(
                    self.main_window, str(self.main_window.last_workspace_path)
                )
            except Exception as e:
                print(
                    f"[WARN] Failed to auto-save last_workspace.json after recording: {e}"
                )

    def _purge_queues_and_buffers(self) -> None:
        """
        Explicitly destroys heavy numpy arrays in display buffers and queues.
        Relying solely on .clear() can delay Garbage Collection in CPython,
        leading to RAM spikes during segment transitions or video finalization.
        """
        # 1. Explicitly purge the Display Dictionary
        for key in list(self.frames_to_display.keys()):
            arr = self.frames_to_display.pop(key)
            del arr
        self.frames_to_display.clear()

        # 2. Explicitly purge the Worker Frame Queue
        with self.frame_queue.mutex:
            while len(self.frame_queue.queue) > 0:
                item = self.frame_queue.queue.popleft()
                del item
            self.frame_queue.queue.clear()

        # 3. Explicitly purge the Decoder Raw Queue
        if hasattr(self, "media_pipeline") and hasattr(
            self.media_pipeline, "raw_frame_queue"
        ):
            with self.media_pipeline.raw_frame_queue.mutex:
                while len(self.media_pipeline.raw_frame_queue.queue) > 0:
                    item = self.media_pipeline.raw_frame_queue.queue.popleft()
                    del item
                self.media_pipeline.raw_frame_queue.queue.clear()

    def _finalize_default_style_recording(self):
        """Finalizes a successful default-style recording (adds audio, cleans up)."""
        print("[INFO] Finalizing default-style recording...")
        temp_audio_dir: str | None = None
        final_file_path = ""

        # Check if processing stopped due to error limit
        if self.stopped_by_error_limit:
            print(
                f"[WARN] Recording stopped due to excessive consecutive read errors ({self.consecutive_read_errors}). "
                f"Output will be saved with '_incomplete' suffix. Total skipped frames: {self.total_skipped_frames}."
            )

        try:
            self.processing = False  # Stop metronome

            # 1. Stop timers and any residual audio subprocess
            self.gpu_memory_update_timer.stop()
            self.preroll_timer.stop()
            self.stop_live_sound()
            self._stop_recording_ffmpeg_input_stream()

            # 2. Release capture early to unblock the feeder.
            print("[INFO] Releasing media capture to unblock feeder thread...")
            if self.media_capture:
                misc_helpers.release_capture(self.media_capture)
                self.media_capture = None

            # 3. Wait for the producer threads to exit fully.
            print("[INFO] Waiting for producer threads to complete...")
            if self.feeder_thread and self.feeder_thread.is_alive():
                self.feeder_thread.join(timeout=3.0)
                if self.feeder_thread.is_alive():
                    print(
                        "[WARN] Feeder thread did not exit cleanly during finalization."
                    )
            self.feeder_thread = None

            if self.detector_thread and self.detector_thread.is_alive():
                self.detector_thread.join(timeout=3.0)
                if self.detector_thread.is_alive():
                    print(
                        "[WARN] Detector thread did not exit cleanly during finalization."
                    )
            self.detector_thread = None
            print("[INFO] Producer threads joined.")

            # 4. Clear buffers and join worker threads.
            self._purge_queues_and_buffers()

            print("[INFO] Waiting for final worker threads...")
            self.join_and_clear_threads()
            print("[INFO] Worker threads joined.")

            # 5. Finalize FFmpeg (close stdin, wait for file to be written)
            if self.encoder.is_running():
                print("[INFO] Closing FFmpeg encoder...")
                # VP-29: Mark recording stopped early.
                self.recording = False

                # Safely close the pipe and wait for the file to finalize
                self.encoder.close_process()

                # VP-HEVC-INFO: Notify the user about Windows Explorer thumbnail
                # support for HEVC outputs. Default codec is hevc_nvenc / libx265.
                self._log_hevc_thumbnail_hint_once()

            # 6. Calculate audio segment times.
            self.play_end_time, end_frame_for_calc, _, duration_probed = (
                self._compute_play_end()
            )
            actual_frames_processed = max(0, int(self.frames_written))
            # Compute source span duration (in seconds). When FPS-cap (FFmpeg input)
            # is active, the end_frame_for_calc is in output-frame space, so map
            # it back to source-frame space before computing duration using
            # recording_source_fps.
            processing_start_src = getattr(self, "processing_start_frame", 0) or 0
            if self._used_ffmpeg_cap and self.recording_source_fps > 0 and self.fps > 0:
                source_end_frame = self.output_to_source_frame(end_frame_for_calc)
                source_span_duration = max(
                    0.0,
                    float(
                        (source_end_frame - processing_start_src)
                        / float(self.recording_source_fps)
                    ),
                )
            else:
                source_span_duration = (
                    max(
                        0.0,
                        float(
                            (end_frame_for_calc - processing_start_src)
                            / float(self.fps)
                            if self.fps > 0
                            else 0.0
                        ),
                    )
                    if self.fps > 0
                    else 0.0
                )
            encoded_duration = (
                float(actual_frames_processed / float(self.fps))
                if self.fps > 0
                else 0.0
            )
            print(
                f"[INFO] Calculated recording end time: {self.play_end_time:.3f}s "
                f"(Frame {end_frame_for_calc}, skipped {self.total_skipped_frames}, "
                f"actual {actual_frames_processed})"
            )
            print(
                "[INFO] Recording duration diagnostics: "
                f"source_span={source_span_duration:.3f}s, "
                f"encoded_from_frames={encoded_duration:.3f}s, "
                f"temp_video_probe={duration_probed:.3f}s"
                if duration_probed is not None
                else "[INFO] Recording duration diagnostics: "
                f"source_span={source_span_duration:.3f}s, "
                f"encoded_from_frames={encoded_duration:.3f}s, "
                "temp_video_probe=unavailable"
            )

            # 7a. Audio Merging
            if self.play_end_time <= self.play_start_time:
                print("[WARN] Recording produced no frames. Skipping audio merge.")
                common_widget_actions.create_and_show_toast_message(
                    self.main_window,
                    "No Video Created",
                    "Recording produced no frames, so no video file was saved.",
                    style_type="warning",
                )
                if self.temp_file and os.path.exists(self.temp_file):
                    try:
                        os.remove(self.temp_file)
                    except OSError:
                        pass
                self.temp_file = ""
            elif (
                self.temp_file
                and os.path.exists(self.temp_file)
                and os.path.getsize(self.temp_file) > 0
            ):
                # 5a. Determine final output path
                was_triggered_by_job = getattr(self, "triggered_by_job_manager", False)
                job_name = (
                    getattr(self.main_window, "current_job_name", None)
                    if was_triggered_by_job
                    else None
                )
                use_job_name = (
                    getattr(self.main_window, "use_job_name_for_output", False)
                    if was_triggered_by_job
                    else False
                )
                output_file_name = (
                    getattr(self.main_window, "output_file_name", None)
                    if was_triggered_by_job
                    else None
                )

                job_name, output_file_name = self._apply_job_timestamp_to_output_name(
                    was_triggered_by_job,
                    job_name,
                    use_job_name,
                    output_file_name,
                )

                output_folder = (
                    str(getattr(self, "active_output_folder", "") or "").strip()
                    or str(
                        self.main_window.control.get("OutputMediaFolder", "")
                    ).strip()
                )

                final_file_path = misc_helpers.get_output_file_path(
                    self.media_path,
                    output_folder,
                    job_name=job_name,
                    use_job_name_for_output=use_job_name,
                    output_file_name=output_file_name,
                )

                # Add suffix only for real read-error stops (not tail-drain timeout force-finalize).
                has_real_read_errors = (
                    int(self.read_error_skip_count) > 0
                    or int(self.consecutive_read_errors) > 0
                )
                if self.stopped_by_error_limit and has_real_read_errors:
                    path_obj = Path(final_file_path)
                    final_file_path = str(
                        path_obj.parent / f"{path_obj.stem}_incomplete{path_obj.suffix}"
                    )
                    print(
                        f"[WARN] Output marked as incomplete due to excessive read errors: {final_file_path}"
                    )

                output_dir = os.path.dirname(final_file_path)
                if output_dir and not os.path.exists(output_dir):
                    os.makedirs(output_dir, exist_ok=True)

                if Path(final_file_path).is_file():
                    try:
                        os.remove(final_file_path)
                    except OSError:
                        pass

                # 7b. Run FFmpeg audio merge command
                print("[INFO] Adding audio (default-style merge)...")
                try:
                    if self.total_skipped_frames > 0:
                        print(
                            "[INFO] Rebuilding audio because frames were skipped "
                            f"(manual dropped={self.manual_dropped_skip_count}, read errors={self.read_error_skip_count})."
                        )
                        temp_audio_root = os.path.join(
                            os.path.dirname(self.temp_file), "temp_audio"
                        )
                        temp_audio_dir = os.path.join(
                            temp_audio_root,
                            f"{Path(self.temp_file).stem}_{uuid.uuid4().hex}",
                        )
                        os.makedirs(temp_audio_dir, exist_ok=True)

                        # Convert skipped frame map into keep-ranges, then extract and concat audio.
                        start_frame_for_calc = (
                            getattr(self, "processing_start_frame", 0) or 0
                        )
                        actual_end_frame = (
                            self.last_displayed_frame
                            if self.last_displayed_frame is not None
                            else end_frame_for_calc - 1
                        )
                        if actual_end_frame < start_frame_for_calc:
                            raise RuntimeError(
                                f"invalid frame boundaries: start={start_frame_for_calc}, end={actual_end_frame}"
                            )
                        segments = self._identify_frame_segments(actual_end_frame)
                        audio_ok, audio_files = (
                            FFmpegPostProcessor.extract_audio_segments(
                                media_path=str(self.media_path),
                                fps=self.recording_source_fps,
                                segments=segments,
                                temp_audio_dir=temp_audio_dir,
                                frame_origin=start_frame_for_calc,
                                time_offset_sec=self.play_start_time,
                            )
                        )
                        if not audio_ok or not audio_files:
                            raise RuntimeError("failed to extract segmented audio")

                        final_audio_path = (
                            FFmpegPostProcessor.concatenate_audio_segments(
                                audio_files=audio_files, temp_audio_dir=temp_audio_dir
                            )
                        )
                        if not final_audio_path:
                            raise RuntimeError("failed to concatenate segmented audio")

                        args = [
                            "ffmpeg",
                            "-hide_banner",
                            "-loglevel",
                            "error",
                            "-i",
                            self.temp_file,
                            "-i",
                            final_audio_path,
                            "-c:v",
                            "copy",
                            "-c:a",
                            "copy",
                            "-map",
                            "0:v:0",
                            "-map",
                            "1:a:0",
                            "-shortest",
                            final_file_path,
                        ]
                    else:
                        args = [
                            "ffmpeg",
                            "-hide_banner",
                            "-loglevel",
                            "error",
                            "-i",
                            self.temp_file,
                            "-ss",
                            str(self.play_start_time),
                            "-to",
                            str(self.play_end_time),
                            "-i",
                            self.media_path,
                            "-c:v",
                            "copy",
                            "-c:a",
                            "aac",
                            "-map",
                            "0:v:0",
                            "-map",
                            "1:a:0?",
                            "-shortest",
                            # REMOVED: "-af", "aresample=async=1000" (Breaks CFR sync and incompatible with -c:a copy)
                            final_file_path,
                        ]

                    subprocess.run(args, check=True)
                    final_output_duration = self._probe_video_duration(final_file_path)
                    if final_output_duration is not None:
                        print(
                            f"[INFO] Final output duration probe: {final_output_duration:.3f}s"
                        )
                    else:
                        print("[INFO] Final output duration probe: unavailable")

                    print(
                        f"[INFO] --- Successfully created final video: {final_file_path} ---"
                    )
                    common_widget_actions.create_and_show_toast_message(
                        self.main_window,
                        "Video Saved",
                        f"Saved video to file: {final_file_path}",
                    )
                except Exception as e:
                    print(f"[ERROR] Audio merge failed: {e}")
                    if self.temp_file and os.path.exists(self.temp_file):
                        print(
                            "[WARN] Falling back to video-only output for default-style recording."
                        )
                        if FFmpegPostProcessor.write_video_only_output(
                            source_video=self.temp_file, output_video=final_file_path
                        ):
                            print(
                                f"[INFO] --- Video-only fallback succeeded: {final_file_path} ---"
                            )
                            common_widget_actions.create_and_show_toast_message(
                                self.main_window,
                                "Video Saved",
                                f"Saved video to file (without audio): {final_file_path}",
                            )
                        else:
                            self.main_window.display_messagebox_signal.emit(
                                "Recording Error",
                                f"Audio merge failed and video-only fallback also failed:\n{e}",
                                self.main_window,
                            )
                finally:
                    if self.temp_file and os.path.exists(self.temp_file):
                        try:
                            os.remove(self.temp_file)
                        except OSError:
                            pass
                    self.temp_file = ""
                    if temp_audio_dir and os.path.isdir(temp_audio_dir):
                        try:
                            shutil.rmtree(temp_audio_dir, ignore_errors=True)
                        except OSError:
                            pass
                    temp_audio_dir = None

            # 8a. Final Timing and Logging
            self.end_time = time.perf_counter()
            processing_time_sec = self.end_time - self.start_time

            try:
                # Fetch the absolute frames from the media pipeline for accurate FPS math
                num_frames_processed = getattr(
                    self.media_pipeline, "absolute_frames_processed", 0
                )
            except Exception:
                num_frames_processed = 0

            self._log_processing_summary(processing_time_sec, num_frames_processed)

            self._auto_save_workspace_for_output(final_file_path)

            # 8b. Reopen media capture AFTER FFmpeg audio merge.
            if self.file_type == "video" and self.media_path:
                last_processed = self.next_frame_to_display - 1
                start_frame = getattr(self, "processing_start_frame", 0)
                if (
                    self._used_ffmpeg_cap
                    and self.fps > 0
                    and self.recording_source_fps > 0
                ):
                    last_processed = self.output_to_source_frame(last_processed)
                reset_frame = max(start_frame, last_processed)
                # Slider stays in source frame space (approach 2).
                # If FPS-cap recording was active, map output frame -> source frame before seek.
                src_slider_max = self.main_window.videoSeekSlider.maximum()
                reset_frame = min(reset_frame, src_slider_max)

                if self._reopen_video_capture(reset_frame):
                    # Restore max_frame_number/fps to source space after FPS-cap recording.
                    self._restore_source_frame_state_after_capture_reopen()
                    self.main_window.videoSeekSlider.blockSignals(True)
                    self.main_window.videoSeekSlider.setValue(reset_frame)
                    self.main_window.videoSeekSlider.blockSignals(False)
                else:
                    print("[WARN] Failed to re-open media capture after recording.")

        except Exception as e:
            print(f"[ERROR] Exception during _finalize_default_style_recording: {e}")

        finally:
            # 10. Reset State and UI
            self.recording = False
            self.processing = False
            self.is_processing_segments = False
            self._used_ffmpeg_cap = False
            self.tail_pending_stall_start_sec = 0.0
            self.tail_force_finalize_due_to_stall = False

            layout_actions.enable_all_parameters_and_control_widget(self.main_window)
            video_control_actions.reset_media_buttons(self.main_window)

            print("[INFO] Clearing GPU Cache.")
            try:
                if torch.cuda.is_available() and torch.cuda.is_initialized():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            gc.collect()

            try:
                self.disable_virtualcam()
            except Exception:
                pass

            if (
                self.main_window.control.get("OpenOutputToggle")
                and not self.triggered_by_job_manager
            ):
                try:
                    list_view_actions.open_output_media_folder(self.main_window)
                except Exception:
                    pass

            print("[INFO] Default-style recording finalized.")
            self.processing_stopped_signal.emit()

    # --- Virtual Camera Methods ---

    def enable_virtualcam(self, backend=False):
        """Starts the pyvirtualcam device."""

        # Reset the circuit breaker latch when an explicit start is requested
        self._virtcam_error_latch = False

        # Guard: Only run if the user has actually enabled the virtual cam
        if not self.main_window.control.get("SendVirtCamFramesEnableToggle", False):
            # Ensure it's also disabled if the toggle is off
            self.disable_virtualcam()
            return

        if not self.media_capture and not isinstance(self.current_frame, numpy.ndarray):
            print("[WARN] Cannot enable virtual camera without media loaded.")
            return

        frame_height, frame_width = 0, 0
        current_fps = self.fps if self.fps > 0 else 30

        if (
            isinstance(self.current_frame, numpy.ndarray)
            and self.current_frame.ndim == 3
        ):
            frame_height, frame_width, _ = self.current_frame.shape
        elif self.media_capture and self.media_capture.isOpened():
            frame_height = int(self.media_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_width = int(self.media_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            if current_fps == 30:
                current_fps = (
                    self.media_capture.get(cv2.CAP_PROP_FPS)
                    if self.media_capture.get(cv2.CAP_PROP_FPS) > 0
                    else 30
                )

        if frame_width <= 0 or frame_height <= 0:
            print(
                f"[ERROR] Cannot enable virtual camera: Invalid dimensions ({frame_width}x{frame_height})."
            )
            return

        self.disable_virtualcam()  # Close existing cam first

        # OBS Virtual Camera (and some other backends) uses a Windows kernel-mode
        # virtual device.  If a new pyvirtualcam.Camera() is opened immediately
        # after close(), the driver has not yet fully released the handle and the
        # new connection is silently ignored by OBS — producing the symptom where
        # the virtual cam appears to stop and cannot be reactivated without
        # switching to another cam and back.  A short settling delay eliminates
        # this race condition.
        time.sleep(0.15)

        backend_to_use = backend or self.main_window.control["VirtCamBackendSelection"]
        print(
            f"[INFO] Enabling virtual camera: {frame_width}x{frame_height} @ {int(current_fps)}fps, Backend: {backend_to_use}, Format: BGR"
        )

        for attempt in range(2):
            try:
                self.virtcam = pyvirtualcam.Camera(
                    width=frame_width,
                    height=frame_height,
                    fps=int(current_fps),
                    backend=backend_to_use,
                    fmt=pyvirtualcam.PixelFormat.BGR,  # Processed frame is BGR
                )
                print(f"[INFO] Virtual camera '{self.virtcam.device}' started.")
                break  # success — exit retry loop
            except Exception as e:
                if attempt == 0:
                    # First attempt failed (driver may still be releasing the handle).
                    print(
                        f"[WARN] Virtual camera open failed (attempt 1): {e}. Retrying in 500 ms"
                    )
                    time.sleep(0.5)
                else:
                    # Second attempt failed. Trip the circuit breaker to prevent infinite loop.
                    print(f"[ERROR] Failed to enable virtual camera: {e}")
                    self.virtcam = None
                    self._virtcam_error_latch = True

    def disable_virtualcam(self):
        """Stops the pyvirtualcam device."""
        # Also reset the error latch when explicitly turning it off,
        # allowing a fresh try next time the user turns it on.
        self._virtcam_error_latch = False

        if self.virtcam:
            print(f"[INFO] Disabling virtual camera '{self.virtcam.device}'.")
            try:
                self.virtcam.close()
            except Exception as e:
                print(f"[WARN] Error closing virtual camera: {e}")
            self.virtcam = None

    # --- Multi-Segment Recording Methods ---

    def start_multi_segment_recording(
        self, segments: list[tuple[int, int]], triggered_by_job_manager: bool = False
    ):
        """
        Initializes and starts a multi-segment recording job.

        :param segments: A list of (start_frame, end_frame) tuples.
        :param triggered_by_job_manager: Flag for Job Manager integration.
        """

        # 1. Guards
        if self.processing or self.is_processing_segments:
            print(
                "[WARN] Attempted to start segment recording while already processing."
            )
            return

        if self.file_type != "video":
            print("[ERROR] Multi-segment recording only supported for video files.")
            return
        if not segments:
            print("[ERROR] No segments provided for multi-segment recording.")
            return
        if not (self.media_capture and self.media_capture.isOpened()):
            print("[ERROR] Video source not open for multi-segment recording.")
            return

        print("[INFO] --- Initializing multi-segment recording... ---")

        # 2. Set State Flags
        self.is_processing_segments = True
        self.recording = False
        self.processing = True  # Master flag
        self.triggered_by_job_manager = triggered_by_job_manager
        self.stopped_by_error_limit = False  # Reset error limit flag for new processing
        # Ensure all elements in 'segments' are strictly tuples of integers.
        sanitized_segments = []
        for seg in segments:
            try:
                # Convert list to tuple and ensure elements are ints
                sanitized_segments.append((int(seg[0]), int(seg[1])))
            except (IndexError, TypeError, ValueError) as e:
                print(f"[WARN] Ignoring malformed segment {seg}: {e}")

        self.segments_to_process = sorted(sanitized_segments)
        self.current_segment_index = -1
        self.temp_segment_files = []
        self.segment_temp_dir = None
        output_folder = video_control_actions.resolve_output_folder(
            self.main_window, str(self.media_path)
        )
        self.active_output_folder = output_folder

        # 3. Disable UI
        if not self.main_window.control["KeepControlsToggle"]:
            layout_actions.disable_all_parameters_and_control_widget(self.main_window)

        # 4. Create Temp Directory
        try:
            base_temp_dir = os.path.join(os.getcwd(), "temp_files", "segments")
            os.makedirs(base_temp_dir, exist_ok=True)
            unique_id = uuid.uuid4()
            self.segment_temp_dir = os.path.join(base_temp_dir, f"run_{unique_id}")
            os.makedirs(self.segment_temp_dir, exist_ok=True)
            print(
                f"[INFO] Created temporary directory for segments: {self.segment_temp_dir}"
            )
        except Exception as e:
            print(f"[ERROR] Failed to create temporary directory: {e}")
            self.main_window.display_messagebox_signal.emit(
                "File System Error",
                f"Failed to create temporary directory:\n{e}",
                self.main_window,
            )
            self.stop_processing()
            return

        # 5. Start Process
        self.start_time = time.perf_counter()

        # 6. Start the first segment
        self.process_next_segment()

    def process_next_segment(self):
        """
        Sets up and starts processing for the *next* segment in the list.
        This function is called iteratively by stop_current_segment.
        """

        # 1. Increment segment index
        self.current_segment_index += 1
        segment_num = self.current_segment_index + 1

        # 2. Check if all segments are done
        if self.current_segment_index >= len(self.segments_to_process):
            print("[INFO] All segments processed.")
            self.finalize_segment_concatenation()
            return

        # 3. Get segment details
        start_frame, end_frame = self.segments_to_process[self.current_segment_index]
        print(
            f"[INFO] --- Starting Segment {segment_num}/{len(self.segments_to_process)} (Frames: {start_frame} - {end_frame}) ---"
        )
        self.current_segment_end_frame = end_frame

        if not self.media_capture or not self.media_capture.isOpened():
            print(
                f"[ERROR] Media capture not available for seeking to segment {segment_num}."
            )
            self.stop_processing()
            return

        # 4. Seek to the start frame of the segment
        print(f"[INFO] Seeking to start frame {start_frame}...")
        misc_helpers.seek_frame(self.media_capture, start_frame)

        # --- Apply Global Resize here too ---
        target_height = self._get_target_input_height()
        # -----------------------------------------------------

        ret, frame_bgr = misc_helpers.read_frame(
            self.media_capture,
            self.media_rotation,
            preview_target_height=target_height,
        )
        if ret:
            self.current_frame = numpy.ascontiguousarray(
                frame_bgr[..., ::-1]
            )  # BGR to RGB
            # Must re-set position, as read() advances it
            misc_helpers.seek_frame(self.media_capture, start_frame)
            self.current_frame_number = start_frame
            self.next_frame_to_display = start_frame
            # Update slider for visual feedback
            self.main_window.videoSeekSlider.blockSignals(True)
            self.main_window.videoSeekSlider.setValue(start_frame)
            self.main_window.videoSeekSlider.blockSignals(False)
        else:
            print(
                f"[ERROR] Could not read frame {start_frame} at start of segment {segment_num}. Aborting."
            )
            self.stop_processing()
            return

        # 5. Clear containers AND START WORKER POOL for the new segment
        self.frames_to_display.clear()
        print(
            f"[INFO] Starting {self.num_threads} persistent worker thread(s) for segment..."
        )
        # Ensure old workers are cleaned up (if present)
        self.join_and_clear_threads()
        self.worker_pool_manager.recreate_queue(self.max_display_buffer_size)
        self.worker_pool_manager.start_persistent_pool(self.num_threads)

        # 6. Setup FFmpeg subprocess for this segment
        temp_segment_filename = f"segment_{self.current_segment_index:03d}.mp4"
        temp_segment_path = os.path.join(self.segment_temp_dir, temp_segment_filename)
        self.temp_segment_files.append(temp_segment_path)

        frame_height, frame_width, _ = self.current_frame.shape
        start_frame, end_frame = self.segments_to_process[self.current_segment_index]

        # Calculate time boundaries for audio extraction mapping
        start_time_sec = start_frame / self.fps if self.fps > 0 else 0.0
        end_time_sec = end_frame / self.fps if self.fps > 0 else 0.0

        success = self.encoder.start_process(
            output_filename=temp_segment_path,
            frame_width=frame_width,
            frame_height=frame_height,
            fps=self.fps,
            control=self.main_window.control,
            is_segment=True,
            media_path=self.media_path,
            start_time_sec=start_time_sec,
            end_time_sec=end_time_sec,
        )

        if not success:
            print(
                f"[ERROR] Failed to create ffmpeg subprocess for segment {segment_num}. Aborting."
            )
            self.stop_processing()
            return

        # 7. Synchronously process the first frame of the segment
        # VP-15: Use synchronous=True so the first frame is fully processed and the
        # single_frame_processed_signal has fired before the metronome starts.
        # This prevents the metronome from ticking before any frame is in frames_to_display.
        current_start_frame = self.current_frame_number
        print(
            f"[INFO] Sync: Synchronously processing first frame {current_start_frame} of segment..."
        )
        with self.frame_queue.mutex:
            self.frame_queue.queue.clear()

        self.start_frame_worker(
            current_start_frame,
            self.current_frame,
            is_single_frame=True,
            synchronous=True,
        )

        # 8. Update counters
        # self.current_frame_number was set to start_frame (e.g., 100)
        # We must increment it so the *next* read is correct (e.g., 101)
        self.current_frame_number += 1

        # 9. Start Metronome ET Feeder VIA MEDIAPIPELINE
        target_fps = 9999.0  # Always max speed for segments
        is_first = self.current_segment_index == 0

        # Push the UI state into the pipeline before starting the thread
        with self.state_lock:
            self.feeder_parameters = copy.deepcopy(self.main_window.parameters)
            self.feeder_control = copy.deepcopy(self.main_window.control)

        print(
            f"[INFO] Starting feeder thread via Pipeline (Mode: segment {self.current_segment_index})..."
        )
        self.media_pipeline.start_feeder(
            mode=f"segment {self.current_segment_index}", recording=True
        )

        # Start the display metronome
        self.media_pipeline.start_metronome(target_fps, is_first_start=is_first)

    def stop_current_segment(self):
        """
        Stops processing the *current* segment, finalizes its file,
        and triggers the next segment or final concatenation.
        """
        if not self.is_processing_segments:
            print("[WARN] stop_current_segment called but not processing segments.")
            return

        segment_num = self.current_segment_index + 1
        print(f"[INFO] --- Stopping Segment {segment_num} --- ")

        # 1. Stop timers
        self.gpu_memory_update_timer.stop()

        # 2a. Wait for the producer threads
        print(f"[INFO] Waiting for producer threads from segment {segment_num}...")
        if self.feeder_thread and self.feeder_thread.is_alive():
            self.feeder_thread.join(timeout=2.0)

            # VP-26: If the join timed out, abort rather than proceeding with two live feeders.
            if self.feeder_thread.is_alive():
                print(
                    f"[ERROR] Feeder thread from segment {segment_num} did not join within timeout. Aborting segment processing."
                )
                self.feeder_thread = None
                self.stop_processing()
                return

        if self.detector_thread and self.detector_thread.is_alive():
            self.detector_thread.join(timeout=2.0)
            if self.detector_thread.is_alive():
                print(
                    f"[ERROR] Detector thread from segment {segment_num} did not join within timeout. Aborting segment processing."
                )
                self.detector_thread = None
                self.stop_processing()
                return

        print("[INFO] Producer threads joined.")
        self.feeder_thread = None
        self.detector_thread = None

        # 2b. Wait for workers
        print(f"[INFO] Waiting for workers from segment {segment_num}...")
        self.join_and_clear_threads()
        print("[INFO] Workers joined.")

        # --- Clear raw frame queue ---
        self._purge_queues_and_buffers()

        # 3. Finalize FFmpeg for this segment
        if self.encoder.is_running():
            print(
                f"[INFO] Closing and waiting for active FFmpeg encoder (segment {segment_num})..."
            )
            self.encoder.close_process()
        else:
            print(
                f"[WARN] No active FFmpeg encoder found when stopping segment {segment_num}."
            )

        if self.temp_segment_files and not os.path.exists(self.temp_segment_files[-1]):
            print(
                f"[ERROR] Segment file '{self.temp_segment_files[-1]}' not found after processing segment {segment_num}."
            )

        # If frames were skipped in this segment, rebuild segment audio
        # from valid frame ranges so concatenated output stays in sync.
        self._rebuild_segment_audio_if_needed(segment_num)

        # 4. Process the *next* segment
        self.process_next_segment()

    def finalize_segment_concatenation(self):
        """Concatenates all valid temporary segment files into the final output file."""
        print("[INFO] --- Finalizing concatenation of segments... ---")

        # Check if processing stopped due to error limit
        if self.stopped_by_error_limit:
            print(
                f"[WARN] Segment recording stopped due to excessive consecutive read errors ({self.consecutive_read_errors}). "
                f"Output will be saved with '_incomplete' suffix. Total skipped frames: {self.total_skipped_frames}."
            )

        # Failsafe: If this is called while an ffmpeg process is still running
        if self.encoder.is_running():
            segment_num = self.current_segment_index + 1
            print(
                f"[INFO] Finalizing: Stopping active FFmpeg process for segment {segment_num}..."
            )
            self.encoder.close_process()

        was_triggered_by_job = self.triggered_by_job_manager

        # 1. Reset state flags
        self.processing = False
        self.is_processing_segments = False
        self.recording = False

        # 2. Find all valid (non-empty) segment files
        valid_segment_files = [
            f
            for f in self.temp_segment_files
            if f and os.path.exists(f) and os.path.getsize(f) > 0
        ]

        if not valid_segment_files:
            print("[WARN] No valid temporary segment files found to concatenate.")
            common_widget_actions.create_and_show_toast_message(
                self.main_window,
                "No Video Created",
                "No valid recorded segments were found, so no video file was saved.",
                style_type="warning",
            )
            self._cleanup_temp_dir()
            layout_actions.enable_all_parameters_and_control_widget(self.main_window)
            video_control_actions.reset_media_buttons(self.main_window)
            self.segments_to_process = []
            self.current_segment_index = -1
            self.temp_segment_files = []
            self.triggered_by_job_manager = False
            self.active_output_folder = ""
            return

        # 3. Determine final output path
        job_name = (
            getattr(self.main_window, "current_job_name", None)
            if was_triggered_by_job
            else None
        )
        use_job_name = (
            getattr(self.main_window, "use_job_name_for_output", False)
            if was_triggered_by_job
            else False
        )
        output_file_name = (
            getattr(self.main_window, "output_file_name", None)
            if was_triggered_by_job
            else None
        )
        output_folder = self.active_output_folder

        job_name, output_file_name = self._apply_job_timestamp_to_output_name(
            was_triggered_by_job,
            job_name,
            use_job_name,
            output_file_name,
        )

        final_file_path = misc_helpers.get_output_file_path(
            self.media_path,
            output_folder,
            job_name=job_name,
            use_job_name_for_output=use_job_name,
            output_file_name=output_file_name,
        )

        # Add suffix only for real read-error stops.
        has_real_read_errors = (
            int(self.read_error_skip_count) > 0 or int(self.consecutive_read_errors) > 0
        )
        if self.stopped_by_error_limit and has_real_read_errors:
            path_obj = Path(final_file_path)
            final_file_path = str(
                path_obj.parent / f"{path_obj.stem}_incomplete{path_obj.suffix}"
            )
            print(
                f"[WARN] Output marked as incomplete due to excessive read errors: {final_file_path}"
            )

        output_dir = os.path.dirname(final_file_path)

        # Check if output_dir is not an empty string before creating it
        if output_dir and not os.path.exists(output_dir):
            try:
                # Added exist_ok=True for thread-safety
                os.makedirs(output_dir, exist_ok=True)
                print(f"[INFO] Created output directory: {output_dir}")
            except OSError as e:
                print(f"[ERROR] Failed to create output directory {output_dir}: {e}")
                self.main_window.display_messagebox_signal.emit(
                    "File Error",
                    f"Could not create output directory:\n{output_dir}\n\n{e}",
                    self.main_window,
                )
                self._cleanup_temp_dir()
                layout_actions.enable_all_parameters_and_control_widget(
                    self.main_window
                )
                video_control_actions.reset_media_buttons(self.main_window)
                self.active_output_folder = ""
                return

        if Path(final_file_path).is_file():
            print(f"[INFO] Removing existing final file: {final_file_path}")
            try:
                os.remove(final_file_path)
            except OSError as e:
                print(f"[ERROR] Failed to remove existing file {final_file_path}: {e}")
                self.main_window.display_messagebox_signal.emit(
                    "File Error",
                    f"Could not delete existing file:\n{final_file_path}\n\n{e}",
                    self.main_window,
                )
                self._cleanup_temp_dir()
                layout_actions.enable_all_parameters_and_control_widget(
                    self.main_window
                )
                video_control_actions.reset_media_buttons(self.main_window)
                self.active_output_folder = ""
                return

        # 4. Create FFmpeg list file
        list_file_path = os.path.join(self.segment_temp_dir, "mylist.txt")
        concatenation_successful = False
        concat_args = []  # VP-33: initialise before try so except blocks can reference it safely
        try:
            print(f"[INFO] Creating ffmpeg list file: {list_file_path}")
            with open(list_file_path, "w", encoding="utf-8") as f_list:
                for segment_path in valid_segment_files:
                    abs_path = os.path.abspath(segment_path)
                    # FFmpeg concat requires forward slashes, even on Windows
                    formatted_path = abs_path.replace("\\", "/")
                    f_list.write(f"file '{formatted_path}'" + os.linesep)

            # 5. Run final concatenation command
            print(
                f"[INFO] Concatenating {len(valid_segment_files)} valid segments into {final_file_path}..."
            )
            concat_args = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                list_file_path,
                "-c:v",
                "copy",
                "-c:a",
                "copy",
                # REMOVED: "-af", "aresample=async=1000" (Breaks CFR sync and incompatible with -c:a copy)
                final_file_path,
            ]
            subprocess.run(concat_args, check=True)
            concatenation_successful = True
            log_prefix = "Job Manager: " if was_triggered_by_job else ""
            print(
                f"[INFO] --- {log_prefix}Successfully created final video: {final_file_path} ---"
            )

        except subprocess.CalledProcessError as e:
            print(f"[ERROR] FFmpeg command failed during final concatenation: {e}")
            print(f"FFmpeg arguments: {' '.join(concat_args)}")
            if self._attempt_segment_video_only_fallback(
                list_file_path,
                final_file_path,
                f"FFmpeg command failed during concatenation:\n{e}\nCould not create final video.",
            ):
                concatenation_successful = True
        except FileNotFoundError:
            print("[ERROR] FFmpeg not found. Ensure it's in your system PATH.")
            self.main_window.display_messagebox_signal.emit(
                "Recording Error", "FFmpeg not found.", self.main_window
            )
        except Exception as e:
            print(f"[ERROR] An unexpected error occurred during finalization: {e}")
            if self._attempt_segment_video_only_fallback(
                list_file_path,
                final_file_path,
                f"An unexpected error occurred:\n{e}",
            ):
                concatenation_successful = True

        finally:
            # 6. Cleanup
            self._cleanup_temp_dir()

            if concatenation_successful:
                self._auto_save_workspace_for_output(final_file_path)
                common_widget_actions.create_and_show_toast_message(
                    self.main_window,
                    "Video Saved",
                    f"Saved video to file: {final_file_path}",
                )

            # 7. Reset state
            self.segments_to_process = []
            self.current_segment_index = -1
            self.temp_segment_files = []
            self.current_segment_end_frame = None
            self.triggered_by_job_manager = False
            self.active_output_folder = ""
            print("[INFO] Purging residual frames and pills from queues...")

            # --- Clear raw frame queue ---
            self._purge_queues_and_buffers()

            # 8. Final timing
            self.end_time = time.perf_counter()
            processing_time_sec = self.end_time - self.start_time
            formatted_duration = self._format_duration(
                processing_time_sec
            )  # Use the new helper

            if concatenation_successful:
                print(
                    f"[INFO] Total segment processing and concatenation finished in {formatted_duration}"
                )
            else:
                print(
                    f"[WARN] Segment processing/concatenation failed after {formatted_duration}."
                )

            # --- Inject the absolute FPS summary for multi-segment jobs ---
            try:
                num_frames_processed = getattr(
                    self.media_pipeline, "absolute_frames_processed", 0
                )
            except Exception:
                num_frames_processed = 0

            self._log_processing_summary(processing_time_sec, num_frames_processed)

            # 9. Final cleanup and UI reset
            print(
                "[INFO] Clearing GPU Cache and running garbage collection post-concatenation."
            )
            try:
                if torch.cuda.is_available() and torch.cuda.is_initialized():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            except Exception as e:
                print(f"[WARN] Error clearing Torch cache: {e}")
            gc.collect()

            # Reset media capture
            if self.file_type == "video" and self.media_path:
                current_slider_pos = self.main_window.videoSeekSlider.value()
                if self._reopen_video_capture(current_slider_pos):
                    print("[INFO] Video capture re-opened and seeked.")
                else:
                    print("[WARN] Failed to re-open media capture after segments.")
            elif self.file_type == "video":
                print("[WARN] media_path not set, cannot re-open video capture.")

            layout_actions.enable_all_parameters_and_control_widget(self.main_window)
            video_control_actions.reset_media_buttons(self.main_window)
            print("[INFO] Multi-segment processing flow finished.")

            if self.main_window.control["OpenOutputToggle"]:
                try:
                    list_view_actions.open_output_media_folder(
                        self.main_window, output_dir
                    )
                except Exception:
                    pass

            # Emit signal to notify JobProcessor that processing has finished SUCCESSFULLY
            print("[INFO] Emitting processing_stopped_signal (multi-segment success).")
            self.processing_stopped_signal.emit()

    def _cleanup_temp_dir(self):
        """Safely removes the temporary directory used for segments."""
        if self.segment_temp_dir and os.path.exists(self.segment_temp_dir):
            try:
                print(
                    f"[INFO] Cleaning up temporary segment directory: {self.segment_temp_dir}"
                )
                shutil.rmtree(self.segment_temp_dir, ignore_errors=True)
            except Exception as e:
                print(
                    f"[WARN] Failed to delete temporary directory {self.segment_temp_dir}: {e}"
                )
        self.segment_temp_dir = None

    # --- Webcam Methods ---

    def process_webcam(self):
        """Starts the webcam stream using the unified metronome and User Settings."""
        if self.processing:
            print("[WARN] Processing already active, cannot start webcam.")
            return
        if self.file_type != "webcam":
            print("[WARN] Process_webcam: Only applicable for webcam input.")
            return

        # 1. Retrieve User Settings from the UI Control Dictionary
        try:
            # Device Index
            webcam_index = int(self.main_window.control.get("WebcamDeviceSelection", 0))

            # Resolution (String like "1920x1080")
            res_str = self.main_window.control.get("WebcamMaxResSelection", "1280x720")
            target_width, target_height = map(int, res_str.split("x"))

            # Backend (String like "DirectShow") -> Mapped to cv2 Constant
            backend_name = self.main_window.control.get(
                "WebcamBackendSelection", "Default"
            )
            backend_id = CAMERA_BACKENDS.get(backend_name, cv2.CAP_ANY)

            # FPS (String like "30")
            target_fps = int(self.main_window.control.get("WebCamMaxFPSSelection", 30))

        except Exception as e:
            print(
                f"[ERROR] Error parsing webcam settings: {e}. Falling back to defaults."
            )
            webcam_index = 0
            target_width, target_height = 1280, 720
            backend_id = cv2.CAP_ANY
            target_fps = 30

        print(
            f"[INFO] Init Webcam: Device={webcam_index}, Backend={backend_name}, Target={target_width}x{target_height} @ {target_fps}fps"
        )

        # 2. Initialize VideoCapture with the selected Backend (Prevent Race Condition)
        reinitialize_needed = True

        # Determine if we can safely reuse the existing capture
        if self.media_capture and self.media_capture.isOpened():
            selected_btn = getattr(self.main_window, "selected_video_button", None)
            from app.ui.widgets import widget_components

            if isinstance(
                selected_btn, widget_components.TargetMediaCardButton
            ) and getattr(selected_btn, "is_webcam", False):
                if (
                    selected_btn.webcam_index == webcam_index
                    and selected_btn.webcam_backend == backend_id
                ):
                    reinitialize_needed = False
                    print(
                        "[INFO] Reusing existing webcam capture to prevent hardware lock issues."
                    )

        if reinitialize_needed:
            if self.media_capture:
                misc_helpers.release_capture(self.media_capture)
                self.media_capture = None
                # CRITICAL: Wait for OS driver hardware lock to fully release
                time.sleep(0.5)

            try:
                self.media_capture = cv2.VideoCapture(webcam_index, backend_id)
            except Exception as e:
                print(f"[ERROR] Failed to init webcam with backend {backend_name}: {e}")
                self.media_capture = cv2.VideoCapture(webcam_index)

        if not (self.media_capture and self.media_capture.isOpened()):
            print("[ERROR] Unable to open webcam source.")
            video_control_actions.reset_media_buttons(self.main_window)
            return

        # 3. Apply Configuration
        try:
            # Force MJPG to allow high framerate at high res (saves USB bandwidth)
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            self.media_capture.set(cv2.CAP_PROP_FOURCC, fourcc)
        except Exception:
            pass

        self.media_capture.set(cv2.CAP_PROP_FRAME_WIDTH, target_width)
        self.media_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, target_height)
        self.media_capture.set(cv2.CAP_PROP_FPS, target_fps)

        # 4. Verify actual resolution obtained
        actual_w = self.media_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_h = self.media_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(
            f"[INFO] Webcam initialized at: {int(actual_w)}x{int(actual_h)} (Requested: {target_width}x{target_height})"
        )

        # Warn if the camera refused the resolution
        if int(actual_w) != target_width or int(actual_h) != target_height:
            print(
                f"[WARN] Camera did not accept requested resolution. Using {int(actual_w)}x{int(actual_h)}."
            )
            if int(actual_w) == 640 and backend_name != "DirectShow":
                print(
                    "[TIP] Try changing 'Webcam Backend' to 'DirectShow' in Settings to unlock HD."
                )

        print("[INFO] Starting webcam processing setup...")

        # 5. Set State Flags
        self.processing = True
        self.is_processing_segments = False
        self.recording = False
        self.start_time = time.perf_counter()

        # 6. Clear Containers
        self.frames_to_display.clear()
        self.webcam_frames_to_display.queue.clear()
        with self.frame_queue.mutex:
            self.frame_queue.queue.clear()

        # --- Clear raw frame queue ---
        if hasattr(self, "media_pipeline") and hasattr(
            self.media_pipeline, "raw_frame_queue"
        ):
            with self.media_pipeline.raw_frame_queue.mutex:
                self.media_pipeline.raw_frame_queue.queue.clear()

        # 7. Start Metronome ET Feeder VIA MEDIAPIPELINE
        fps = self.media_capture.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30
        self.fps = fps

        print(f"[INFO] Webcam target FPS: {self.fps}")

        self.join_and_clear_threads()
        self.worker_pool_manager.recreate_queue(self.max_display_buffer_size)
        self.worker_pool_manager.start_persistent_pool(self.num_threads)

        # Start the feeder thread via pipeline
        print("[INFO] Starting feeder thread via Pipeline (Mode: webcam)...")
        self.media_pipeline.start_feeder(mode="webcam", recording=False)

        # Start the display metronome
        self.media_pipeline.start_metronome(self.fps, is_first_start=True)
