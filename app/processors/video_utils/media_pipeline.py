import os
import copy
import time
import threading
import queue
import subprocess
import cv2
import psutil
import numpy
import torch
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, cast

# --- QT & FFmpeg Logger Suppression ---
# Silences the internal C++ demuxer probe data (Input #0...) from bleeding into the python console.
os.environ["QT_LOGGING_RULES"] = "qt.multimedia.*=false"
os.environ["FFMPEG_LOG_LEVEL"] = "quiet"
os.environ["AV_LOG_LEVEL"] = "quiet"

from PySide6.QtCore import QObject, QTimer, Qt, Slot, QUrl
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput

from app.helpers.typing_helper import ControlTypes, FacesParametersTypes
import app.helpers.miscellaneous as misc_helpers
from app.ui.widgets.actions import (
    graphics_view_actions,
    video_control_actions,
    common_actions as common_widget_actions,
)

if TYPE_CHECKING:
    from app.ui.main_ui import MainWindow
    from app.processors.video_processor import VideoProcessor

# Constants used for handling media tails and read errors
TAIL_TOLERANCE = 30
MAX_CONSECUTIVE_ERRORS = 300
TAIL_PENDING_STALL_TIMEOUT_SEC = 8.0


def fast_state_copy(obj: Any) -> Any:
    """
    Custom fast deepcopy for HPC video pipelines.
    Isolates dictionaries and lists to guarantee temporal independence for each frame worker,
    but strictly passes heavy arrays (PyTorch Tensors, NumPy arrays) by reference
    to prevent RAM and VRAM memory leaks.
    """
    if isinstance(obj, dict):
        new_dict = type(obj)()
        for k, v in obj.items():
            new_dict[k] = fast_state_copy(v)
        return new_dict
    elif isinstance(obj, list):
        return [fast_state_copy(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(fast_state_copy(v) for v in obj)
    elif isinstance(obj, set):
        return {fast_state_copy(v) for v in obj}
    elif isinstance(obj, (numpy.ndarray, torch.Tensor)):
        # STRIKE WARNING: Do NOT duplicate heavy machine learning tensors. Pass by reference.
        return obj
    else:
        # Pass immutable basic types as-is
        if isinstance(obj, (int, float, str, bool, bytes, type(None))):
            return obj
        import copy

        return copy.copy(obj)


class MediaPipeline(QObject):
    """
    Manages the Producer/Consumer flow of the Video Processing Engine.

    Responsibilities:
    - Producer: Reads raw frames (cv2/ffmpeg), isolates UI state, runs Sequential Detection, and feeds the WorkerPool.
    - Consumer: The Metronome (QTimer) pulls finished frames, syncs audio (ffplay), and updates the UI/FFmpeg outputs.
    - Safety: Strictly guards the UI state using Locks and `fast_state_copy` to prevent Thread Bleed.
    """

    def __init__(self, vp: "VideoProcessor", main_window: "MainWindow"):
        super().__init__()
        self.vp = vp
        self.main_window = main_window

        # --- Pipeline State ---
        self.feeder_thread: Optional[threading.Thread] = None
        self.detector_thread: Optional[threading.Thread] = None

        # --- Audio State (Native Qt) ---
        self.audio_output: Optional[QAudioOutput] = None
        self.media_player: Optional[QMediaPlayer] = None
        # Set only when Accurate Audio-Video Sync is enabled for a preview.
        self.live_sound_seek_time: Optional[float] = None

        # --- Timers (Consumer) ---
        self.preroll_timer = QTimer(self)
        self.precise_metronome = QTimer(self)
        self.precise_metronome.setTimerType(Qt.TimerType.PreciseTimer)
        self.precise_metronome.setSingleShot(True)
        self.precise_metronome.timeout.connect(self.display_next_frame)

        # --- State Lock & Feeder dictionaries (Producer) ---
        self.state_lock = threading.Lock()
        self.feeder_parameters: Optional[FacesParametersTypes] = None
        self.feeder_control: Optional[ControlTypes] = None
        self.ui_state_is_dirty: bool = True

        # --- Buffers ---
        # Changed to store ONLY numpy arrays to prevent VRAM memory bloat
        self.frames_to_display: Dict[int, numpy.ndarray] = {}
        self.webcam_frames_to_display: queue.Queue[numpy.ndarray] = queue.Queue()
        self.max_frames_to_display_size: int = 8
        self._wrap_frame_target: int = -1  # Used to synchronize seamless looping

        # --- Intermediate Stage 1 to Stage 2 Buffer (Decoder to Detector) ---
        dynamic_raw_queue_size: int = max(5, self.vp.num_threads // 2)
        self.raw_frame_queue: queue.Queue[Optional[Tuple[int, numpy.ndarray]]] = (
            queue.Queue(maxsize=dynamic_raw_queue_size)
        )

        # --- Segment Playback State ---
        self.segment_jumps: Dict[int, int] = {}

        # --- Timing & Metrics ---
        self.last_display_schedule_time_sec: float = 0.0
        self.target_delay_sec: float = 1.0 / 30.0
        self.playback_started: bool = False
        self.playback_display_start_time: float = 0.0
        self.heartbeat_frame_counter: int = 0
        self._fps_window_start_sec: float = 0.0
        self._fps_window_frames: int = 0
        self._display_fps_ema: float = 0.0

        # --- Error & Skip Tracking ---
        self.skipped_frames: set[int] = set()
        self.consecutive_read_errors: int = 0
        self.total_skipped_frames: int = 0
        self.manual_dropped_skip_count: int = 0
        self.read_error_skip_count: int = 0
        self.stopped_by_error_limit: bool = False

    # --- BUFFER MANAGEMENT ---

    def _get_dynamic_max_buffer_size(
        self, frame_shape: Optional[Tuple[int, ...]] = None
    ) -> int:
        """Calculates an adaptive maximum buffer limit based on active system RAM and frame dimensions.

        Prevents high-resolution media (4K, 8K, VR180) from consuming excessive memory while ensuring
        worker threads are never starved of tasks.
        """
        num_threads: int = self.vp.num_threads
        # Minimum required buffer depth to prevent 5-frame worker thread starvation
        min_required_buffer: int = max(num_threads * 2, 8)
        default_max_buffer: int = self.vp.max_display_buffer_size

        if not frame_shape or len(frame_shape) < 2:
            return default_max_buffer

        height: int = frame_shape[0]
        width: int = frame_shape[1]
        channels: int = frame_shape[2] if len(frame_shape) > 2 else 3
        frame_bytes: int = height * width * channels

        if frame_bytes <= 0:
            return default_max_buffer

        # Budget 2.5 GB specifically for the uncompressed frame buffer queue
        BUFFER_RAM_BUDGET_BYTES: int = 2500 * 1024 * 1024  # 2.5 GB

        # Compute max capacity based on resolution
        dynamic_cap: int = int(BUFFER_RAM_BUDGET_BYTES / frame_bytes)

        # Clamp between min_required_buffer (prevents stutter) and default_max_buffer (prevents RAM bloat)
        return max(min_required_buffer, min(dynamic_cap, default_max_buffer))

    def store_frame_to_display(self, frame_number: int, frame: numpy.ndarray) -> None:
        """Stores a processed video/image frame from a worker thread into the consumer dictionary."""
        if not self.vp.processing and not self.vp.is_processing_segments:
            del frame
            return

        # Intercept wrongly arriving frames from the webcam feed
        if self.vp.file_type == "webcam":
            self.store_webcam_frame_to_display(frame)
            return

        draining_tail: bool = self.is_draining_tail()

        # --- PREDICT VALID FUTURE PATH (Lookahead 60 frames) ---
        # Prevents aggressive buffer eviction during segment loops by explicitly
        # mapping the playhead's true future path across jumps and seamless loops.
        valid_future_path: set[int] = set()

        if self.vp.file_type == "video" and not draining_tail:
            with self.state_lock:
                curr: int = self.vp.next_frame_to_display
                wrap_target: int = self._wrap_frame_target
                jumps: dict[int, int] = self.segment_jumps.copy()

            # Cache evaluation: only recompute the 60-frame lookahead if the state shifted
            if (
                not hasattr(self, "_cached_valid_future_path")
                or getattr(self, "_cached_path_next_frame", -1) != curr
                or getattr(self, "_cached_path_wrap", -1) != wrap_target
                or getattr(self, "_cached_path_jumps", None) != jumps
            ):
                new_path: set[int] = set()
                sim_curr: int = curr
                for _ in range(60):
                    new_path.add(sim_curr)
                    if sim_curr in jumps:
                        sim_curr = jumps[sim_curr]
                    elif wrap_target != -1 and sim_curr == wrap_target:
                        sim_curr = 0
                    else:
                        sim_curr += 1

                self._cached_valid_future_path = new_path
                self._cached_path_next_frame = curr
                self._cached_path_wrap = wrap_target
                self._cached_path_jumps = jumps

            valid_future_path = self._cached_valid_future_path

            # Early discard of stale frames completely outside the upcoming path
            if frame_number not in valid_future_path and frame_number < curr:
                del frame
                return

        self.frames_to_display[frame_number] = frame

        # Evict stale frames when the buffer exceeds the soft cap.
        if (
            len(self.frames_to_display) > self.max_frames_to_display_size
            and not draining_tail
        ):
            # Phase 4c: Compute stale keys ONCE outside the eviction loop to achieve O(N) removal
            stale_keys: list[int] = []
            if self.vp.file_type == "video":
                stale_keys = [
                    k
                    for k in self.frames_to_display.keys()
                    if k not in valid_future_path
                ]
            else:
                stale_keys = [
                    k
                    for k in self.frames_to_display.keys()
                    if k < self.vp.next_frame_to_display
                ]

            while (
                len(self.frames_to_display) > self.max_frames_to_display_size
                and stale_keys
            ):
                oldest: int = min(stale_keys)
                stale_keys.remove(oldest)
                if oldest in self.frames_to_display:
                    arr: numpy.ndarray = self.frames_to_display.pop(oldest)
                    del arr

    def store_webcam_frame_to_display(self, frame: numpy.ndarray) -> None:
        """Stores a processed webcam frame. Overwrites stale frames so the UI is always real-time."""
        while not self.webcam_frames_to_display.empty():
            try:
                stale_frame = self.webcam_frames_to_display.get_nowait()
                del stale_frame
            except queue.Empty:
                break

        self.webcam_frames_to_display.put(frame)

    # --- PRODUCER (FEEDER THREAD) ---

    def start_feeder(self, mode: str, recording: bool) -> None:
        """Initializes and starts the feeder background thread(s)."""
        print(
            f"[INFO] Starting feeder pipeline (Mode: {mode}, Recording: {recording})..."
        )

        if self.vp.file_type == "webcam":
            # Single stage for real-time webcam (I/O bound to sensor framerate)
            self.feeder_thread = threading.Thread(target=self._feed_webcam, daemon=True)
            self.feeder_thread.start()
        else:
            # Decoupled 2-stage pipeline for video/segments

            # --- DYNAMIC RAW QUEUE SIZING (RAM OPTIMIZED) ---
            dynamic_raw_queue_size: int = max(5, self.vp.num_threads // 2)

            # Safely reset the intermediate queue AND update its maxsize in-place
            # This prevents reference-bleeding because we never destroy the original queue object.
            with self.raw_frame_queue.mutex:
                self.raw_frame_queue.queue.clear()
                self.raw_frame_queue.maxsize = dynamic_raw_queue_size

                # Unblock any stale threads that might have been waiting during a previous run
                self.raw_frame_queue.not_empty.notify_all()
                self.raw_frame_queue.not_full.notify_all()
                self.raw_frame_queue.all_tasks_done.notify_all()

            self.detector_thread = threading.Thread(
                target=self._detector_loop, daemon=True
            )
            self.detector_thread.start()

            self.feeder_thread = threading.Thread(target=self._decode_loop, daemon=True)
            self.feeder_thread.start()

    def _decode_loop(self) -> None:
        """
        STAGE 1: Dedicated Decoder Thread.
        Reads raw frames from disk/FFmpeg as fast as possible and pushes them to the raw_frame_queue.
        Completely decoupled from the heavy AI inference workers and slider thread limits.
        """
        is_segment_mode = self.vp.is_processing_segments
        is_playing_segments = getattr(self.vp, "is_playing_segments", False)

        def stop_flag_check():
            return (
                self.vp.is_processing_segments
                if is_segment_mode
                else self.vp.processing
            )

        print(
            f"[INFO] Decoder Thread: Starting video read loop (Mode: {'Segment' if is_segment_mode else 'Standard'})."
        )

        self.consecutive_read_errors = 0
        self.skipped_frames.clear()
        self.total_skipped_frames = 0
        self.manual_dropped_skip_count = 0
        self.read_error_skip_count = 0

        cached_resize_toggle = self.main_window.control.get(
            "GlobalInputResizeToggle", False
        )
        cached_target_height = self.vp._get_target_input_height()

        while stop_flag_check():
            try:
                # 1. Mode-specific stop logic
                if is_segment_mode:
                    if self.vp.current_segment_end_frame is None:
                        time.sleep(0.01)
                        continue
                    if self.vp.current_frame_number > self.vp.current_segment_end_frame:
                        print(
                            f"[INFO] Decoder Thread: Reached end of segment {self.vp.current_segment_index + 1}. Stopping feed."
                        )
                        break
                elif is_playing_segments:
                    # --- Segment Playback Jump Logic ---
                    if self.vp.current_segment_end_frame is None:
                        time.sleep(0.01)
                        continue
                    if self.vp.current_frame_number > self.vp.current_segment_end_frame:
                        next_idx = self.vp.current_segment_index + 1
                        is_playback_loop_enabled = self.main_window.control.get(
                            "VideoPlaybackLoopToggle", False
                        )

                        if next_idx < len(self.vp.segments_to_process):
                            next_start, next_end = self.vp.segments_to_process[next_idx]
                        elif is_playback_loop_enabled:
                            next_idx = 0
                            next_start, next_end = self.vp.segments_to_process[next_idx]
                        else:
                            print(
                                "[INFO] Decoder Thread: Reached end of final playback segment. Stopping feed."
                            )
                            break

                        jump_from = self.vp.current_segment_end_frame
                        jump_to = next_start

                        with self.state_lock:
                            self.segment_jumps[jump_from] = jump_to

                            # Reuse wrap logic to protect frames from buffer eviction during backward jumps
                            if jump_to < jump_from:
                                self._wrap_frame_target = jump_from

                            self.vp.current_segment_index = next_idx
                            self.vp.current_segment_end_frame = next_end
                            self.vp.current_frame_number = jump_to

                        print(
                            f"[INFO] Decoder Thread: Segment jump registered ({jump_from} -> {jump_to})."
                        )
                        misc_helpers.seek_frame(self.vp.media_capture, jump_to)
                        self.consecutive_read_errors = 0
                        continue
                else:
                    if self.vp.current_frame_number > self.vp.max_frame_number:
                        is_playback_loop_enabled = self.main_window.control.get(
                            "VideoPlaybackLoopToggle", False
                        )
                        if is_playback_loop_enabled and not self.vp.recording:
                            print(
                                "[INFO] Decoder Thread: End of media, flowing seamlessly back to start."
                            )
                            with self.state_lock:
                                self._wrap_frame_target = (
                                    self.vp.current_frame_number - 1
                                )
                            self.vp.current_frame_number = 0
                            misc_helpers.seek_frame(self.vp.media_capture, 0)
                            continue
                        else:
                            break

                # 2. Handle Manual Frame Drops
                if (
                    (is_segment_mode or self.vp.recording)
                    and not self.vp.ffmpeg_input_sp
                    and self.vp.current_frame_number in self.main_window.dropped_frames
                ):
                    self._mark_skipped_frame(
                        self.vp.current_frame_number, "manual_drop"
                    )
                    self.vp.current_frame_number += 1
                    misc_helpers.seek_frame(
                        self.vp.media_capture, self.vp.current_frame_number
                    )
                    continue

                # 3. Determine Input Resolution
                current_resize_toggle = self.main_window.control.get(
                    "GlobalInputResizeToggle", False
                )
                if current_resize_toggle != cached_resize_toggle:
                    cached_resize_toggle = current_resize_toggle
                    cached_target_height = self.vp._get_target_input_height()
                target_height = cached_target_height

                # 4. Read Frame
                if self.vp.ffmpeg_input_sp:
                    ret, frame_bgr = self.vp._read_frame_from_ffmpeg_input_stream()
                else:
                    ret, frame_bgr = misc_helpers.read_frame(
                        self.vp.media_capture,
                        self.vp.media_rotation,
                        preview_target_height=target_height,
                    )

                # 5. Handle Read Errors and EOF
                if not ret:
                    if self.vp.ffmpeg_input_sp:
                        remaining_frames = (
                            self.vp.max_frame_number - self.vp.current_frame_number
                        )
                        eof_like = (
                            self.vp.current_frame_number
                            >= self.vp.max_frame_number - TAIL_TOLERANCE
                            or remaining_frames <= MAX_CONSECUTIVE_ERRORS
                        )
                        if eof_like:
                            print(
                                "[INFO] Decoder Thread: FFmpeg input stream EOF reached."
                            )
                        else:
                            self.consecutive_read_errors += 1
                            self._mark_skipped_frame(
                                self.vp.current_frame_number, "read_error"
                            )
                            self.stopped_by_error_limit = True
                            print(
                                f"[WARN] Decoder Thread: FFmpeg input stream terminated early at {self.vp.current_frame_number}. Treating as corrupted input."
                            )
                        with self.state_lock:
                            self.vp.next_frame_to_display = self.vp.max_frame_number + 1
                        break

                    fn = self.vp.current_frame_number
                    if (
                        is_segment_mode or is_playing_segments
                    ) and self.vp.current_segment_end_frame is not None:
                        if fn >= self.vp.current_segment_end_frame - TAIL_TOLERANCE:
                            if is_playing_segments:
                                print(
                                    f"[INFO] Decoder Thread: Read failure near playback segment tail (frame={fn}). Forcing segment jump."
                                )
                                with self.state_lock:
                                    self.vp.current_frame_number = (
                                        self.vp.current_segment_end_frame + 1
                                    )
                                continue
                            else:
                                with self.state_lock:
                                    self.vp.next_frame_to_display = (
                                        self.vp.current_segment_end_frame + 1
                                    )
                                    self.vp.current_frame_number = (
                                        self.vp.current_segment_end_frame + 1
                                    )
                                print(
                                    f"[INFO] Decoder Thread: Treat read failure near segment tail as EOF (frame={fn})."
                                )
                                break

                    if (
                        not is_segment_mode
                        and fn >= self.vp.max_frame_number - TAIL_TOLERANCE
                    ):
                        is_playback_loop_enabled = self.main_window.control.get(
                            "VideoPlaybackLoopToggle", False
                        )
                        if is_playback_loop_enabled and not self.vp.recording:
                            print(
                                f"[INFO] Decoder Thread: Read failure near file end (frame={fn}/{self.vp.max_frame_number}), flowing seamlessly to start."
                            )
                            with self.state_lock:
                                self._wrap_frame_target = fn - 1
                            self.vp.current_frame_number = 0
                            misc_helpers.seek_frame(self.vp.media_capture, 0)
                            self.consecutive_read_errors = 0
                            continue
                        else:
                            print(
                                f"[INFO] Decoder Thread: Read failure near file end (frame={fn}/{self.vp.max_frame_number}), treating as EOF."
                            )
                            with self.state_lock:
                                self.vp.next_frame_to_display = (
                                    self.vp.max_frame_number + 1
                                )
                            break

                    self.consecutive_read_errors += 1
                    self._mark_skipped_frame(self.vp.current_frame_number, "read_error")

                    if self.consecutive_read_errors > MAX_CONSECUTIVE_ERRORS:
                        print(
                            f"[INFO] Decoder Thread: Too many consecutive read errors ({self.consecutive_read_errors}). Stopping."
                        )
                        try:
                            near_eof = fn >= self.vp.max_frame_number - TAIL_TOLERANCE
                        except Exception:
                            near_eof = False

                        if not near_eof:
                            self.stopped_by_error_limit = True

                        with self.state_lock:
                            self.vp.next_frame_to_display = self.vp.max_frame_number + 1

                        if is_segment_mode:
                            self.vp.is_processing_segments = False
                        break

                    print(
                        f"[WARN] Decoder Thread: Skipping unreadable frame {self.vp.current_frame_number}."
                    )
                    self.vp.current_frame_number += 1
                    misc_helpers.seek_frame(
                        self.vp.media_capture, self.vp.current_frame_number
                    )
                    continue

                if frame_bgr is None:
                    continue

                self.consecutive_read_errors = 0
                frame_num_to_process = self.vp.current_frame_number

                # 6. Push to Intermediate Queue
                # We use a loop with timeout so the thread can gracefully exit if stop_flag_check() becomes False
                while stop_flag_check():
                    try:
                        self.raw_frame_queue.put(
                            (frame_num_to_process, frame_bgr), timeout=0.5
                        )
                        self.vp.current_frame_number += 1
                        break
                    except queue.Full:
                        continue

            except Exception as e:
                print(
                    f"[ERROR] Error in _decode_loop (Mode: {'Segment' if is_segment_mode else 'Standard'}): {e}"
                )
                if is_segment_mode:
                    self.vp.is_processing_segments = False
                else:
                    self.vp.processing = False
                break

        # --- SHUTDOWN & CLEANUP ---
        print("[INFO] Decoder Thread finished.")

        if self.total_skipped_frames > 0:
            print(
                f"[INFO] Decoder Thread skipped a total of {self.total_skipped_frames} frames."
            )

        # Emit Poison Pill to safely shut down the Detector Thread
        while True:
            # If the user pressed stop, immediately purge the queue to push the poison pill instantly,
            # bypassing the natural backpressure mechanism.
            if not stop_flag_check():
                with self.raw_frame_queue.mutex:
                    self.raw_frame_queue.queue.clear()

            try:
                self.raw_frame_queue.put(None, timeout=0.5)
                break
            except queue.Full:
                continue

    def _mark_skipped_frame(self, frame_number: int, reason: str) -> None:
        """Tracks corrupted/skipped frames to ensure the PostProcessor can perfectly sync the audio later."""
        self.skipped_frames.add(frame_number)
        self.total_skipped_frames += 1

        if reason == "manual_drop":
            self.manual_dropped_skip_count += 1
        elif reason == "read_error":
            self.read_error_skip_count += 1

    def _detector_loop(self) -> None:
        """
        STAGE 2: Dedicated Detector Thread.
        Pulls raw frames from the raw_frame_queue, checks RAM/VRAM safety limits,
        isolates UI state, runs Sequential Detection, and feeds the WorkerPool.
        This isolates GPU-bound detection from I/O-bound decoding.
        """
        is_segment_mode = self.vp.is_processing_segments

        def stop_flag_check():
            return (
                self.vp.is_processing_segments
                if is_segment_mode
                else self.vp.processing
            )

        print(
            f"[INFO] Detector Thread: Starting detection loop (Mode: {'Segment' if is_segment_mode else 'Standard'})."
        )

        # --- ISOLATED CUDA STREAM ---
        if torch.cuda.is_available():
            detector_stream = torch.cuda.Stream()
            torch.cuda.set_stream(detector_stream)

        last_marker_data = None
        self.ui_state_is_dirty = True
        current_frame_shape: Optional[Tuple[int, ...]] = None

        while stop_flag_check():
            try:
                # 0. Guard: feeder_parameters must be initialised
                if self.feeder_parameters is None:
                    time.sleep(0.005)
                    continue

                # 1. Pull from Stage 1 (Decoder)
                try:
                    raw_item = self.raw_frame_queue.get(timeout=0.2)
                except queue.Empty:
                    continue

                # Poison Pill from Decoder indicates EOF or Stop
                if raw_item is None:
                    print("[INFO] Detector Thread: Received poison pill. Exiting loop.")
                    break

                frame_num_to_process, frame_bgr = raw_item
                current_frame_shape = frame_bgr.shape

                # 2. Buffer control (Adaptive VRAM & RAM Safety Net)
                # Hoist loop-invariant calculations out of the polling loop
                dynamic_buffer_limit: int = self._get_dynamic_max_buffer_size(
                    current_frame_shape
                )
                min_safe_buffer: int = max(self.vp.num_threads * 2, 6)
                MIN_FREE_RAM_BYTES: int = 2500 * 1024 * 1024  # 2.5 GB

                last_mem_check_time: float = 0.0
                is_ram_safe: bool = True
                is_vram_safe: bool = True

                while stop_flag_check():
                    in_flight_frames: int = (
                        len(self.frames_to_display) + self.vp.frame_queue.qsize()
                    )

                    # Dynamic Enqueue limit (Lightweight integer comparison, checked every 20ms)
                    if in_flight_frames >= dynamic_buffer_limit:
                        time.sleep(0.02)
                        continue

                    # Memory Emergency Guards (Heavyweight OS/Driver calls, throttled to 2Hz)
                    if in_flight_frames > min_safe_buffer:
                        current_time = time.perf_counter()

                        # Only query psutil and CUDA driver every 0.5 seconds
                        if current_time - last_mem_check_time > 0.5:
                            is_ram_safe = (
                                psutil.virtual_memory().available >= MIN_FREE_RAM_BYTES
                            )

                            if (
                                torch.cuda.is_available()
                                and torch.cuda.is_initialized()
                            ):
                                free_vram, _ = torch.cuda.mem_get_info()
                                is_vram_safe = free_vram >= 1073741824

                            last_mem_check_time = current_time

                        if not is_ram_safe or not is_vram_safe:
                            time.sleep(
                                0.05
                            )  # Back off slightly longer (50ms) under memory pressure
                            continue

                    break  # Safe to proceed

                if not stop_flag_check():
                    break

                local_params_for_worker: FacesParametersTypes
                local_control_for_worker: ControlTypes

                # 3. State Isolation (Preventing Thread Bleed)
                with self.state_lock:
                    marker_data = self.main_window.markers.get(frame_num_to_process)
                    if marker_data and marker_data != last_marker_data:
                        print(
                            f"[INFO] Frame {frame_num_to_process} is a marker. Updating feeder state."
                        )
                        self.feeder_parameters = copy.deepcopy(
                            marker_data["parameters"]
                        )
                        new_control = cast(ControlTypes, {})
                        for (
                            widget_name,
                            widget,
                        ) in self.main_window.parameter_widgets.items():
                            if widget_name in self.main_window.control:
                                new_control[widget_name] = widget.default_value

                        if "control" in marker_data and isinstance(
                            marker_data["control"], dict
                        ):
                            new_control.update(
                                cast(ControlTypes, marker_data["control"]).copy()
                            )
                        self.feeder_control = new_control
                        last_marker_data = marker_data
                        self.ui_state_is_dirty = True

                    if getattr(self, "ui_state_is_dirty", True) or not hasattr(
                        self, "_cached_params"
                    ):
                        self._cached_params = fast_state_copy(self.feeder_parameters)
                        self._cached_control = fast_state_copy(self.feeder_control)
                        self.ui_state_is_dirty = False
                        print("[INFO] Global State changed : Dirty flag cleared")

                    local_params_for_worker = fast_state_copy(self._cached_params)
                    local_control_for_worker = fast_state_copy(self._cached_control)
                    local_params_for_worker = cast(FacesParametersTypes, {})

                    if self._cached_params is not None:
                        for face_id, face_data in self._cached_params.items():
                            if isinstance(face_data, dict):
                                local_params_for_worker[face_id] = cast(
                                    Any, face_data.copy()
                                )
                            else:
                                local_params_for_worker[face_id] = face_data

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                # 4. Sequential Detection Check
                is_swap_active = self.main_window.swapfacesButton.isChecked()
                is_edit_active = self.main_window.editFacesButton.isChecked()

                if len(self.main_window.target_faces) > 0 and (
                    is_swap_active or is_edit_active
                ):
                    self.vp._video_had_targets = True
                    is_master_edit_active = self.main_window.editFacesButton.isChecked()
                    bboxes, kpss_5, kpss, kpss_203 = self.vp.sequential_detector.run(
                        frame_rgb=frame_rgb,
                        local_control_for_worker=local_control_for_worker,
                        local_params_for_worker=local_params_for_worker,
                        is_master_edit_active=is_master_edit_active,
                        frame_number=frame_num_to_process,
                    )
                else:
                    bboxes = numpy.empty((0, 4), dtype=numpy.float32)
                    kpss_5 = numpy.empty((0, 5, 2), dtype=numpy.float32)
                    kpss = numpy.empty((0, 68, 2), dtype=numpy.float32)
                    kpss_203 = numpy.empty((0, 203, 2), dtype=numpy.float32)

                    if self.vp._video_had_targets:
                        self.vp.sequential_detector.reset_state()
                        self.vp._video_had_targets = False

                task = (
                    frame_num_to_process,
                    frame_rgb,
                    local_params_for_worker,
                    local_control_for_worker,
                    bboxes,
                    kpss_5,
                    kpss,
                    kpss_203,
                )

                # 5. Feed WorkerPool
                # Wrapped in a timeout loop to allow thread shutdown during queue backpressure
                while stop_flag_check():
                    try:
                        self.vp.frame_queue.put(task, timeout=0.5)
                        break
                    except queue.Full:
                        continue

            except Exception as e:
                print(
                    f"[ERROR] Error in _detector_loop (Mode: {'Segment' if is_segment_mode else 'Standard'}): {e}"
                )
                if is_segment_mode:
                    self.vp.is_processing_segments = False
                else:
                    self.vp.processing = False
                break

        # --- Reliable End-of-Loop Cleanup (Unblock workers) ---
        for _ in self.vp.worker_pool_manager.worker_threads:
            while True:
                # If forcefully stopped, purge the queue so pills go through instantly,
                # overriding the backpressure bottleneck.
                if not stop_flag_check():
                    with self.vp.frame_queue.mutex:
                        self.vp.frame_queue.queue.clear()

                try:
                    self.vp.frame_queue.put(None, timeout=0.5)
                    break
                except queue.Full:
                    # If natural EOF, wait patiently for workers to drain valid frames
                    # before pushing the poison pill, preventing dropped signals.
                    continue

        # --- Release the Custom CUDA Stream ---
        if torch.cuda.is_available() and "detector_stream" in locals():
            detector_stream.synchronize()
            torch.cuda.set_stream(torch.cuda.default_stream())
            del detector_stream

        print("[INFO] Detector Thread finished.")

    def _feed_webcam(self) -> None:
        """Continuous extraction loop for Live Webcam hardware."""
        self.ui_state_is_dirty = True
        while self.vp.processing:
            try:
                in_flight_frames = (
                    len(self.webcam_frames_to_display.queue)
                    + self.vp.frame_queue.qsize()
                )
                if in_flight_frames >= self.vp.max_display_buffer_size:
                    time.sleep(0.02)  # Increased to 20ms to drop CPU spin usage
                    continue

                ret, frame_bgr = misc_helpers.read_frame(
                    self.vp.media_capture, 0, preview_target_height=None
                )

                # Explicit Mypy guard for frame_bgr
                if not ret or frame_bgr is None:
                    print("[WARN] Feeder: Failed to read webcam frame.")
                    continue

                # --- ZERO-COPY PINNED RING BUFFER (BGR -> RGB) ---
                current_shape = frame_bgr.shape
                if (
                    not hasattr(self, "_webcam_pinned_ring_buffer")
                    or getattr(self, "_webcam_pinned_ring_shape", None) != current_shape
                ):
                    self._webcam_pinned_ring_shape = current_shape
                    buffer_count = self.vp.max_display_buffer_size + 5
                    self._webcam_pinned_ring_buffer = []
                    is_cuda = torch.cuda.is_available()
                    for _ in range(buffer_count):
                        if is_cuda:
                            t = torch.empty(
                                current_shape, dtype=torch.uint8
                            ).pin_memory()
                            self._webcam_pinned_ring_buffer.append(t.numpy())
                        else:
                            self._webcam_pinned_ring_buffer.append(
                                numpy.empty(current_shape, dtype=numpy.uint8)
                            )
                    self._webcam_pinned_ring_idx = 0
                    print(
                        f"[INFO] Webcam: Initialized Zero-Copy RGB Ring Buffer for shape {current_shape} ({buffer_count} frames)"
                    )

                pinned_dst = self._webcam_pinned_ring_buffer[
                    self._webcam_pinned_ring_idx
                ]
                self._webcam_pinned_ring_idx = (self._webcam_pinned_ring_idx + 1) % len(
                    self._webcam_pinned_ring_buffer
                )

                cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB, dst=pinned_dst)
                frame_rgb = pinned_dst

                with self.main_window.models_processor.model_lock:
                    if getattr(self, "ui_state_is_dirty", True) or not hasattr(
                        self, "_webcam_cached_params"
                    ):
                        self._webcam_cached_params = fast_state_copy(
                            self.main_window.parameters
                        )
                        self._webcam_cached_control = fast_state_copy(
                            self.main_window.control
                        )
                        self.ui_state_is_dirty = False
                        print("[INFO] Global State changed : Dirty flag cleared")

                    local_params_for_worker = fast_state_copy(
                        self._webcam_cached_params
                    )
                    local_control_for_worker = fast_state_copy(
                        self._webcam_cached_control
                    )

                # 5. Sequential Detection Check
                is_swap_active = self.main_window.swapfacesButton.isChecked()
                is_edit_active = self.main_window.editFacesButton.isChecked()

                if len(self.main_window.target_faces) > 0 and (
                    is_swap_active or is_edit_active
                ):
                    self.vp._webcam_had_targets = True
                    is_master_edit_active = self.main_window.editFacesButton.isChecked()
                    bboxes, kpss_5, kpss, kpss_203 = self.vp.sequential_detector.run(
                        frame_rgb=frame_rgb,
                        local_control_for_worker=local_control_for_worker,
                        local_params_for_worker=local_params_for_worker,
                        is_master_edit_active=is_master_edit_active,
                        frame_number=0,
                    )
                else:
                    bboxes = numpy.empty((0, 4), dtype=numpy.float32)
                    kpss_5 = numpy.empty((0, 5, 2), dtype=numpy.float32)
                    kpss = numpy.empty((0, 68, 2), dtype=numpy.float32)
                    kpss_203 = numpy.empty((0, 203, 2), dtype=numpy.float32)

                    if self.vp._webcam_had_targets:
                        self.vp.sequential_detector.reset_state()
                        self.vp._webcam_had_targets = False

                task = (
                    0,
                    frame_rgb,
                    local_params_for_worker,
                    local_control_for_worker,
                    bboxes,
                    kpss_5,
                    kpss,
                    kpss_203,
                )
                self.vp.frame_queue.put(task)

            except Exception as e:
                print(f"[ERROR] Error in _feed_webcam loop: {e}")
                self.vp.processing = False

    # --- CONSUMER (METRONOME) ---

    def _source_video_fps(self) -> float:
        source_fps = float(getattr(self.vp, "recording_source_fps", 0.0) or 0.0)
        if source_fps > 0.0:
            return source_fps
        return float(getattr(self.vp, "fps", 0.0) or 0.0)

    def _set_playback_fps_text(self, playback_fps: float) -> None:
        fps_widget = getattr(self.main_window, "videoFpsLineEdit", None)
        if fps_widget is None:
            return

        source_fps = max(0, round(self._source_video_fps()))
        measured_fps = max(0, round(playback_fps))
        fps_widget.setText(f"{source_fps}/{measured_fps}")

    def _reset_playback_fps_display(self, now_sec: float | None = None) -> None:
        self._fps_window_start_sec = (
            time.perf_counter() if now_sec is None else float(now_sec)
        )
        self._fps_window_frames = 0
        self._display_fps_ema = 0.0
        self._set_playback_fps_text(0.0)

    def _update_playback_fps_display(self, now_sec: float | None = None) -> None:
        """Counts displayed frames and refreshes the readout ~1.3x per second.

        Measuring over a window (rather than per-frame deltas) keeps the number
        readable while a single slow frame is being swapped, and the EMA stops it
        from flickering between adjacent integers.
        """
        now = time.perf_counter() if now_sec is None else float(now_sec)
        if self._fps_window_start_sec <= 0.0:
            self._fps_window_start_sec = now

        self._fps_window_frames += 1
        elapsed = now - self._fps_window_start_sec
        if elapsed < 0.75:
            return

        measured_fps = self._fps_window_frames / max(elapsed, 1e-6)
        if self._display_fps_ema > 0.0:
            self._display_fps_ema = 0.65 * self._display_fps_ema + 0.35 * measured_fps
        else:
            self._display_fps_ema = measured_fps

        self._set_playback_fps_text(self._display_fps_ema)
        self._fps_window_start_sec = now
        self._fps_window_frames = 0

    def start_metronome(self, target_fps: float, is_first_start: bool = True) -> None:
        """Starts the hyper-precise Qt Timer that pulls completed frames and updates the UI."""
        if target_fps <= 0:
            target_fps = 30.0
        if target_fps > 9000:
            self.target_delay_sec = 0.005
        else:
            self.target_delay_sec = 1.0 / target_fps

        self._reset_playback_fps_display()

        self.vp.gpu_memory_update_timer.start(5000)

        if is_first_start:
            self.vp.processing_started_signal.emit()
            self.playback_display_start_time = time.perf_counter()
            # NEW: Initialize the absolute frame counter for accurate FPS tracking
            self.absolute_frames_processed = 0

        self.last_display_schedule_time_sec = time.perf_counter()
        self.heartbeat_frame_counter = 0
        self.display_next_frame()

    def _check_preroll_and_start_playback(self) -> None:
        """Checks if the internal producer buffer has enough frames to start smooth metronome playback."""
        if not self.vp.processing:
            self.preroll_timer.stop()
            return
        if self.playback_started:
            self.preroll_timer.stop()
            return

        is_feeder_done = (
            not self.feeder_thread.is_alive() if self.feeder_thread else True
        ) and (not self.detector_thread.is_alive() if self.detector_thread else True)

        # Query the dynamic target directly from the VideoProcessor
        if len(self.frames_to_display) >= self.vp.preroll_target or is_feeder_done:
            self.preroll_timer.stop()
            self.playback_started = True
            print(
                f"[INFO] Preroll buffer ready ({len(self.frames_to_display)} frames). Starting playback components..."
            )
            self._start_synchronized_playback()
        else:
            print(
                f"[INFO] Buffering... {len(self.frames_to_display)} / {self.vp.preroll_target}"
            )

    def is_draining_tail(self) -> bool:
        """Checks if the file is finished reading and we are just waiting for the final worker threads to finish encoding."""
        if not (self.vp.recording or self.vp.is_processing_segments):
            return False

        # The pipeline is draining if the producers have completely shut down,
        # meaning no new frames will arrive, and we must flush whatever is left.
        return (
            (self.feeder_thread is not None)
            and (not self.feeder_thread.is_alive())
            and (not self.detector_thread.is_alive() if self.detector_thread else True)
        )

    def _handle_tail_drain_wait(self, frame_number_to_display: int) -> bool:
        """Safeguard to prevent the encoding from hanging infinitely if a worker thread dies at the end of a video."""
        if frame_number_to_display in self.frames_to_display:
            self.vp.tail_pending_stall_start_sec = 0.0
            return False

        pending_tasks = self.vp._safe_unfinished_tasks()
        if pending_tasks == 0:
            self.vp.tail_pending_stall_start_sec = 0.0
            return True

        now_sec = time.perf_counter()
        if self.vp.tail_pending_stall_start_sec <= 0.0:
            self.vp.tail_pending_stall_start_sec = now_sec
            return True

        if (
            now_sec - self.vp.tail_pending_stall_start_sec
            >= TAIL_PENDING_STALL_TIMEOUT_SEC
        ):
            self.vp.tail_force_finalize_due_to_stall = True
            self.vp.tail_pending_stall_start_sec = 0.0
            print(
                f"[WARN] Tail-drain stalled for too long ({TAIL_PENDING_STALL_TIMEOUT_SEC:.1f}s). Forcing finalization."
            )
            return True

        return True

    def _abort_if_pipeline_cannot_produce_frame(
        self, frame_number_to_display: int
    ) -> bool:
        if not (self.vp.recording or self.vp.is_processing_segments):
            return False

        feeder_alive = bool(self.feeder_thread and self.feeder_thread.is_alive())
        detector_alive = bool(self.detector_thread and self.detector_thread.is_alive())

        if feeder_alive or detector_alive or self.vp.frame_queue.qsize() > 0:
            return False

        workers = list(self.vp.worker_pool_manager.worker_threads)
        if workers and any(worker.is_alive() for worker in workers):
            return False

        pending_tasks = self.vp._safe_unfinished_tasks()
        error_msg = (
            "Processing pipeline stopped before producing required frame "
            f"{frame_number_to_display}; pending_tasks={pending_tasks}."
        )
        self.vp.last_processing_error = error_msg
        self.vp.stop_processing()
        return True

    def display_next_frame(self) -> None:
        """
        The Core Consumer Loop.
        Pulls a finished frame, updates PySide6 GUI, and writes to FFmpeg.
        Audio plays independently via QMediaPlayer to prevent stuttering.
        """
        is_playback_loop_enabled = self.main_window.control.get(
            "VideoPlaybackLoopToggle", False
        )
        should_stop_playback = False
        should_finalize_default_recording = False
        is_playing_segments = getattr(self.vp, "is_playing_segments", False)

        # 0. Check End-of-Media First
        if self.vp.file_type == "video":
            hit_eof = False
            with self.state_lock:
                if self.vp.is_processing_segments:
                    if (
                        self.vp.current_segment_end_frame is not None
                        and self.vp.next_frame_to_display
                        > self.vp.current_segment_end_frame
                    ):
                        hit_eof = True
                else:
                    if (
                        self._wrap_frame_target != -1
                        and self.vp.next_frame_to_display > self._wrap_frame_target
                    ):
                        hit_eof = True
                    elif self.vp.next_frame_to_display > self.vp.max_frame_number:
                        hit_eof = True
                    elif is_playing_segments and self.vp.segments_to_process:
                        final_segment_end = self.vp.segments_to_process[-1][1]
                        if self.vp.next_frame_to_display > final_segment_end:
                            hit_eof = True

            if hit_eof:
                # Check worker pool state to prevent premature truncation
                pending_tasks = self.vp._safe_unfinished_tasks()
                is_drain_complete = not self.frames_to_display and (
                    pending_tasks == 0 or self.vp.tail_force_finalize_due_to_stall
                )

                if self.vp.is_processing_segments:
                    if is_drain_complete:
                        print(
                            f"[INFO] Segment {self.vp.current_segment_index + 1} end frame ({self.vp.current_segment_end_frame}) reached."
                        )
                        self.vp.stop_current_segment()
                        return
                    # If drain is not complete, the loop bypasses this block and safely continues to pull min(self.frames_to_display) below.
                else:
                    if self.vp.recording:
                        if is_drain_complete:
                            print("[INFO] End of media reached.")
                            should_finalize_default_recording = True
                    elif is_playback_loop_enabled:
                        print(
                            "[INFO] Metronome: End of media reached. Seamless loop wrapping active."
                        )
                        with self.state_lock:
                            if is_playing_segments and self.vp.segments_to_process:
                                self.vp.next_frame_to_display = (
                                    self.vp.segments_to_process[0][0]
                                )
                                self.vp.current_segment_index = 0
                                self.vp.current_segment_end_frame = (
                                    self.vp.segments_to_process[0][1]
                                )
                            else:
                                self.vp.next_frame_to_display = 0
                            self._wrap_frame_target = -1

                        self.main_window.videoSeekSlider.blockSignals(True)
                        self.main_window.videoSeekSlider.setValue(
                            self.vp.next_frame_to_display
                        )
                        self.main_window.videoSeekSlider.blockSignals(False)

                        self.stop_live_sound()
                        # The loop/segment wrap continues video at its exact
                        # target frame, so it must not reuse the initial
                        # keyframe origin selected for preview startup.
                        self.live_sound_seek_time = None
                        if self.main_window.liveSoundButton.isChecked():
                            self.start_live_sound()
                    else:
                        print("[INFO] End of media reached.")
                        should_stop_playback = True

            if should_finalize_default_recording:
                self.vp._finalize_default_style_recording()
                return
            elif should_stop_playback:
                self.vp.stop_processing()
                return

        if not self.vp.processing:
            return

        # 1. Metronome Timing Math
        now_sec = time.perf_counter()
        self.last_display_schedule_time_sec += self.target_delay_sec
        if self.last_display_schedule_time_sec < now_sec:
            self.last_display_schedule_time_sec = now_sec + 0.001

        wait_time_sec = self.last_display_schedule_time_sec - now_sec
        # Recording and segment export run the metronome at 9999 fps, so
        # target_delay_sec collapses to 5ms and this floor decides the real tick rate.
        # At 1ms a Qt PreciseTimer raises the global Windows timer resolution and wakes
        # the GUI thread up to 1000x/s for the whole job. Nothing is being watched at
        # that rate - the metronome only drains finished frames to the encoder, and
        # nobody is looking at the preview - so back it off. Playback keeps the 1ms
        # floor, where the tick rate is the frame rate and smoothness is visible.
        is_export = bool(self.vp.recording or self.vp.is_processing_segments)
        wait_ms = max(4 if is_export else 1, int(wait_time_sec * 1000))

        if self.vp.processing:
            self.precise_metronome.start(wait_ms)

        # 2. Extract Frame from Buffer
        frame = None
        frame_number_to_display = 0

        if self.vp.file_type == "webcam":
            if self.webcam_frames_to_display.empty():
                return
            frame = self.webcam_frames_to_display.get()
        else:
            draining_tail = self.is_draining_tail()
            if draining_tail and self.frames_to_display:
                frame_number_to_display = min(self.frames_to_display)
            else:
                frame_number_to_display = self.vp.next_frame_to_display

            original_frame = frame_number_to_display
            while (
                frame_number_to_display in self.skipped_frames
                and frame_number_to_display <= self.vp.max_frame_number
            ):
                frame_number_to_display += 1

            if frame_number_to_display > original_frame:
                skipped_count = frame_number_to_display - original_frame
                print(
                    f"[INFO] Display: Advancing past {skipped_count} skipped frame(s), jumping to frame {frame_number_to_display}"
                )
                self.vp.next_frame_to_display = frame_number_to_display

            if frame_number_to_display not in self.frames_to_display:
                if draining_tail:
                    if self._handle_tail_drain_wait(frame_number_to_display):
                        return
                elif self._abort_if_pipeline_cannot_produce_frame(
                    frame_number_to_display
                ):
                    return
                else:
                    return

            frame = self.frames_to_display.pop(frame_number_to_display)
            self.vp.tail_pending_stall_start_sec = 0.0

        # 3. Output to System (UI, Encoder, VirtualCam)
        self.vp.current_frame = frame

        # Increment absolute counter strictly upon frame display
        if hasattr(self, "absolute_frames_processed"):
            self.absolute_frames_processed += 1

        if self.vp.file_type != "webcam":
            self.heartbeat_frame_counter += 1
            if self.heartbeat_frame_counter >= 500:
                self.heartbeat_frame_counter = 0
                self.vp.processing_heartbeat_signal.emit()

        self.vp.send_frame_to_virtualcam(frame)

        if self.vp.is_processing_segments or self.vp.recording:
            if self.vp.encoder.is_running():
                if self.vp.encoder.write_frame(frame):
                    self.vp.frames_written += 1
                    self.vp.last_displayed_frame = frame_number_to_display
                else:
                    log_prefix = (
                        f"segment {self.vp.current_segment_index + 1}"
                        if self.vp.is_processing_segments
                        else "recording"
                    )
                    print(
                        f"[WARN] Error writing frame {frame_number_to_display} to FFmpeg encoder during {log_prefix}."
                    )
            else:
                log_prefix = (
                    f"segment {self.vp.current_segment_index + 1}"
                    if self.vp.is_processing_segments
                    else "recording"
                )
                print(
                    f"[WARN] FFmpeg encoder not available for {log_prefix} when trying to write frame {frame_number_to_display}."
                )

        if self.vp.file_type != "webcam":
            if frame_number_to_display in self.main_window.markers:
                with self.main_window.models_processor.model_lock:
                    video_control_actions.update_parameters_and_control_from_marker(
                        self.main_window, frame_number_to_display
                    )
                    video_control_actions.update_widget_values_from_markers(
                        self.main_window, frame_number_to_display
                    )

        # CREATE QPIXMAP JUST-IN-TIME (GUI Thread safety)
        pixmap = common_widget_actions.get_pixmap_from_frame(self.main_window, frame)
        slider_display_frame = frame_number_to_display
        if (
            self.vp._used_ffmpeg_cap
            and self.vp.fps > 0
            and self.vp.recording_source_fps > 0
        ):
            src_slider_max = self.main_window.videoSeekSlider.maximum()
            slider_display_frame = min(
                self.vp.output_to_source_frame(frame_number_to_display), src_slider_max
            )

        graphics_view_actions.update_graphics_view(
            self.main_window, pixmap, slider_display_frame
        )
        self._update_playback_fps_display()

        self.main_window.models_processor.check_deferred_unloads(
            frame_number_to_display
        )
        if self.vp.file_type != "webcam":
            jump_occurred = False
            if (
                hasattr(self, "segment_jumps")
                and frame_number_to_display in self.segment_jumps
            ):
                next_f = self.segment_jumps.pop(frame_number_to_display)
                with self.state_lock:
                    self.vp.next_frame_to_display = next_f
                    if self._wrap_frame_target == frame_number_to_display:
                        self._wrap_frame_target = -1
                jump_occurred = True

                print(f"[INFO] Metronome: Executing segment jump to frame {next_f}.")
                self.main_window.videoSeekSlider.blockSignals(True)
                self.main_window.videoSeekSlider.setValue(next_f)
                self.main_window.videoSeekSlider.blockSignals(False)

                self.stop_live_sound()
                # The feeder continues from the exact segment boundary; do
                # not reuse an earlier keyframe selected for initial playback.
                self.live_sound_seek_time = None
                if self.main_window.liveSoundButton.isChecked():
                    self.start_live_sound()

            if not jump_occurred:
                self.vp.next_frame_to_display += 1

    # --- AUDIO SYNCHRONIZATION ---
    def resolve_live_preview_start(
        self, requested_frame: int, source_fps: float
    ) -> tuple[int, float]:
        """Return the preceding video keyframe used as the preview origin."""
        requested_frame = max(0, int(requested_frame))
        if not self.vp.media_path or source_fps <= 0:
            return requested_frame, requested_frame / max(source_fps, 1.0)

        requested_time = requested_frame / float(source_fps)
        args = [
            "ffprobe",
            "-v",
            "error",
            "-read_intervals",
            f"{requested_time:.6f}%+0.25",
            "-select_streams",
            "v:0",
            "-skip_frame",
            "nokey",
            "-show_entries",
            "frame=best_effort_timestamp_time",
            "-of",
            "csv=p=0",
            self.vp.media_path,
        ]
        try:
            result = subprocess.run(
                args,
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
            if result.returncode != 0:
                raise RuntimeError(result.stderr.strip() or "ffprobe failed")

            keyframe_times: list[float] = []
            for line in result.stdout.splitlines():
                try:
                    timestamp = float(line.split(",", 1)[0].strip())
                except ValueError:
                    continue
                if 0 <= timestamp <= requested_time + (1.0 / source_fps):
                    keyframe_times.append(timestamp)
            if not keyframe_times:
                raise RuntimeError("no preceding video keyframe returned")

            keyframe_time = max(keyframe_times)
            keyframe_frame = min(
                requested_frame,
                max(0, int(round(keyframe_time * source_fps))),
            )
            print(
                "[INFO] Live preview keyframe alignment: "
                f"requested frame {requested_frame} ({requested_time:.3f}s) -> "
                f"frame {keyframe_frame} ({keyframe_time:.3f}s)."
            )
            return keyframe_frame, keyframe_time
        except Exception as e:
            print(
                "[WARN] Could not resolve a live-preview keyframe; "
                f"using requested frame {requested_frame}: {e}"
            )
            return requested_frame, requested_time

    def start_live_sound(self) -> None:
        """Starts QMediaPlayer audio synced exactly to the current metronome frame."""
        if not self.vp.media_capture:
            print("[WARN] start_live_sound: media_capture is None, cannot start audio.")
            return

        fps = self.vp.media_capture.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0

        # Default mode keeps the requested-frame seek. Accurate-sync mode uses
        # the keyframe timestamp resolved before the feeder starts.
        seek_time = self.live_sound_seek_time
        if seek_time is None:
            seek_time = self.vp.next_frame_to_display / fps
        seek_time_ms = int(seek_time * 1000)

        playback_rate = 1.0
        if (
            self.main_window.control.get("VideoPlaybackCustomFpsToggle", False)
            and not self.vp.recording
        ):
            fpscust = self.main_window.control.get("VideoPlaybackCustomFpsSlider", fps)
            if fps > 0 and fpscust > 0:
                playback_rate = fpscust / fps

        # SAFETY: Qt does not support negative rates or exactly 0.0.
        if playback_rate < 0.01:
            playback_rate = 0.01

        # 0. Lazy load QtMultimedia to save VRAM on idle workspaces
        if self.media_player is None:
            self.audio_output = QAudioOutput(self)
            self.media_player = QMediaPlayer(self)
            self.media_player.setAudioOutput(self.audio_output)

        # 1. Prevent reloading the same source to keep the media in memory
        media_url = QUrl.fromLocalFile(str(self.vp.media_path))
        is_new_source = self.media_player.source() != media_url

        if is_new_source:
            self.media_player.setSource(media_url)

        # Set volume (Qt handles volume linearly from 0.0 to 1.0)
        volume = float(
            self.main_window.control.get("LiveSoundVolumeDecimalSlider", 1.0)
        )
        if self.audio_output:
            self.audio_output.setVolume(volume)

        # Set speed
        self.media_player.setPlaybackRate(playback_rate)

        # 2. Start the media player engine to transition its state
        self.media_player.play()

        # 3. Attempt immediate seek
        self.media_player.setPosition(seek_time_ms)

        # 4. ASYNC SAFETY NET FOR FIRST LOAD
        if is_new_source:
            QTimer.singleShot(40, lambda: self.media_player.setPosition(seek_time_ms))

        print(
            f"[INFO] Native audio started at {seek_time_ms}ms with {playback_rate:.2f}x speed."
        )

    def _start_synchronized_playback(self) -> None:
        """Triggered when the preroll buffer is full. Starts audio and video simultaneously."""
        if self.main_window.liveSoundButton.isChecked() and not self.vp.recording:
            print("[INFO] Starting native audio playback (QMediaPlayer)...")
            self.start_live_sound()

            # A 50ms micro-delay is plenty for the Qt audio buffer to initialize.
            AUDIO_STARTUP_LATENCY_MS = 50
            print(
                f"[INFO] Waiting {AUDIO_STARTUP_LATENCY_MS}ms for audio buffer to initialize..."
            )

            QTimer.singleShot(
                int(AUDIO_STARTUP_LATENCY_MS),
                self._start_video_metronome_after_audio_delay,
            )
        else:
            print("[INFO] No audio. Starting video metronome immediately.")
            self.start_metronome(self.vp.fps, is_first_start=True)

    @Slot()
    def _start_video_metronome_after_audio_delay(self) -> None:
        """Slot invoked by the QTimer to start the metronome after the audio delay."""
        if not self.vp.processing:
            return
        print("[INFO] Audio startup delay complete. Starting video metronome.")
        self.start_metronome(self.vp.fps, is_first_start=True)

    def stop_live_sound(self) -> None:
        """Stops the native Qt audio playback cleanly without zombie processes."""
        if (
            self.media_player
            and self.media_player.playbackState()
            != QMediaPlayer.PlaybackState.StoppedState
        ):
            self.media_player.stop()
            print("[INFO] Native audio playback stopped cleanly.")
