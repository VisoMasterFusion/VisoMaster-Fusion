import threading
import queue
from typing import TYPE_CHECKING, Dict, Tuple
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
import psutil
import numpy
import torch
import pyvirtualcam
import math
from PySide6.QtCore import QObject, QTimer, Signal, Slot
from PySide6.QtGui import QPixmap
from app.processors.workers.frame_worker import FrameWorker
from app.ui.widgets.actions import graphics_view_actions
from app.ui.widgets.actions import common_actions as common_widget_actions
from app.ui.widgets.actions import video_control_actions
from app.ui.widgets.actions import layout_actions
from app.ui.widgets.actions import save_load_actions
from app.ui.widgets.actions import list_view_actions
import app.helpers.miscellaneous as misc_helpers
import warnings

if TYPE_CHECKING:
    from app.ui.main_ui import MainWindow


class VideoProcessor(QObject):
    """
    Manages all video, image, and webcam processing.
    This class handles:
    - Reading frames from media (video, image, webcam).
    - Dispatching frames to worker threads (FrameWorker) for processing.
    - Managing the display metronome (QTimer) for smooth playback/recording.
    - Handling default and multi-segment recording via FFmpeg.
    - Controlling the virtual camera (pyvirtualcam) output.
    - Managing audio playback (ffplay) during preview.
    """

    # --- Signals ---
    frame_processed_signal = Signal(int, QPixmap, numpy.ndarray)
    webcam_frame_processed_signal = Signal(QPixmap, numpy.ndarray)
    single_frame_processed_signal = Signal(int, QPixmap, numpy.ndarray)
    processing_started_signal = Signal()  # Unified signal for any processing start
    processing_stopped_signal = Signal() # Unified signal for any processing stop
    processing_heartbeat_signal = Signal() # Emits periodically to show liveness

    def __init__(self, main_window: "MainWindow", num_threads=2):
        super().__init__()
        self.main_window = main_window

        # --- Worker Thread Management ---
        self.num_threads = num_threads
        self.preroll_target = max(20, self.num_threads * 2) # Target number of frames before playback starts
        self.max_display_buffer_size = self.preroll_target * 4 # Max frames allowed "in flight" (queued + being displayed)
        self.frame_queue = queue.Queue(maxsize=self.max_display_buffer_size)  # Holds frame numbers for workers
        self.threads: Dict[int, threading.Thread] = {} # Active worker threads, keyed by frame number

        # --- Media State ---
        self.media_capture: cv2.VideoCapture | None = None # The OpenCV capture object
        self.file_type = None # "video", "image", or "webcam"
        self.fps = 0.0 # Target FPS for playback or recording
        self.media_path = None
        self.current_frame_number = 0 # The *next* frame to be read/processed
        self.max_frame_number = 0
        self.current_frame: numpy.ndarray = [] # The most recently read/processed frame

        # --- Processing State Flags ---
        self.processing = (
            False  # MASTER flag: True if playback, recording, or webcam stream is active
        )
        self.recording: bool = False  # True if "default-style" recording is active
        self.is_processing_segments: bool = False  # True if "multi-segment" recording is active
        self.triggered_by_job_manager: bool = False  # For multi-segment job integration

        # --- Subprocesses ---
        self.virtcam: pyvirtualcam.Camera | None = None
        self.recording_sp: subprocess.Popen | None = (
            None  # FFmpeg process for both recording styles
        )
        self.ffplay_sound_sp = None # ffplay process for live audio

        # --- Metronome and Timing ---
        self.processing_start_frame: int = 0 # The frame number where processing started
        self.last_display_schedule_time_sec: float = 0.0 # Used by metronome to prevent drift
        self.target_delay_sec: float = 1.0 / 30.0 # Time between frames for metronome
        self.preroll_timer = QTimer(self)
        self.feeder_thread: threading.Thread | None = None  # The dedicated thread that reads frames and "feeds" the workers
        self.playback_started: bool = False
        self.heartbeat_frame_counter: int = 0 # Counter for heartbeat signal

        # --- Performance Timing ---
        self.start_time = 0.0
        self.end_time = 0.0
        self.playback_display_start_time = 0.0 # Time when frames *actually* started displaying
        self.play_start_time = 0.0  # Used by default style for audio segmenting
        self.play_end_time = 0.0  # Used by default style for audio segmenting

        # --- Default Recording State ---
        self.temp_file: str = ""  # Temporary video file (without audio)

        # --- Multi-Segment Recording State ---
        self.segments_to_process: list[tuple[int, int]] = []
        self.current_segment_index: int = -1
        self.temp_segment_files: list[str] = []
        self.current_segment_end_frame: int | None = None
        self.segment_temp_dir: str | None = None

        # --- Utility Timers ---
        self.gpu_memory_update_timer = QTimer()
        self.gpu_memory_update_timer.timeout.connect(
            partial(common_widget_actions.update_gpu_memory_progressbar, main_window)
        )

        # --- Frame Display/Storage ---
        self.next_frame_to_display = 0 # The next frame number the UI should display
        self.frames_to_display: Dict[int, Tuple[QPixmap, numpy.ndarray]] = {} # Processed video frames
        self.webcam_frames_to_display = queue.Queue() # Processed webcam frames

        # --- Signal Connections ---
        self.frame_processed_signal.connect(self.store_frame_to_display)
        self.webcam_frame_processed_signal.connect(self.store_webcam_frame_to_display)
        self.single_frame_processed_signal.connect(self.display_current_frame)
        self.single_frame_processed_signal.connect(self.store_frame_to_display)

    @Slot(int, QPixmap, numpy.ndarray)
    def store_frame_to_display(self, frame_number, pixmap, frame):
        """Slot to store a processed video/image frame from a worker."""
        self.frames_to_display[frame_number] = (pixmap, frame)

    @Slot(QPixmap, numpy.ndarray)
    def store_webcam_frame_to_display(self, pixmap, frame):
        """
        Slot to store a processed webcam frame from a worker.
        For live webcam, we only want the *latest* frame.
        We must clear any old frames waiting in the queue before adding the new one.
        This changes the queue from a buffer into a "mailbox" (latest item only).
        """
        # Clear all pending (old) frames from the queue
        while not self.webcam_frames_to_display.empty():
            try:
                self.webcam_frames_to_display.get_nowait()
            except queue.Empty:
                break
        
        # Put the new, latest frame in the now-empty queue
        self.webcam_frames_to_display.put((pixmap, frame))

    @Slot(int, QPixmap, numpy.ndarray)
    def display_current_frame(self, frame_number, pixmap, frame):
        """
        Slot to display a single, specific frame.
        Used after seeking or loading new media. NOT part of the metronome loop.
        """
        if self.main_window.loading_new_media:
            graphics_view_actions.update_graphics_view(
                self.main_window, pixmap, frame_number, reset_fit=True
            )
            self.main_window.loading_new_media = False
        else:
            graphics_view_actions.update_graphics_view(
                self.main_window, pixmap, frame_number
            )
        self.current_frame = frame
        torch.cuda.empty_cache()
        common_widget_actions.update_gpu_memory_progressbar(self.main_window)

    def _start_metronome(self, target_fps: float, is_first_start: bool = True):
        """
        Unified metronome starter.
        This function configures and starts the metronome loop for all processing types.
        
        :param target_fps: The target FPS. Use > 9000 for max speed (recording).
        :param is_first_start: True if this is the very first start (e.g., not a new segment).
        """

        # 2. Determine timer interval
        if target_fps <= 0:
            target_fps = 30.0 # Fallback
        
        if target_fps > 9000: # Convention for "max speed"
            log_fps = f"MAX SPEED (for {self.fps:.2f} FPS recording)"
            self.target_delay_sec = 0.005
        else:
            log_fps = f"{target_fps:.2f} FPS"
            self.target_delay_sec = 1.0 / target_fps
            
        print(f"Starting unified metronome (Target: {log_fps}).")
        
        # 3. Start utility timers and emit signal
        self.gpu_memory_update_timer.start(5000)
        
        if is_first_start:
             self.processing_started_signal.emit() # EMIT UNIFIED SIGNAL
             # Record the time when the display *actually* starts
             self.playback_display_start_time = time.perf_counter()
             print(f"Metronome: Display loop initiated at {self.playback_display_start_time:.3f}s")

        # 4. Start the metronome loop
        self.last_display_schedule_time_sec = time.perf_counter()
        self.heartbeat_frame_counter = 0 # Reset heartbeat counter
        self.display_next_frame() # Start the loop

    def _check_preroll_and_start_playback(self):
        """
        Called by preroll_timer.
        Checks if the display buffer is full enough to start playback.
        """
        if not self.processing:
            self.preroll_timer.stop()
            return
        
        # If playback has already started, stop this timer and exit.
        if self.playback_started:
            self.preroll_timer.stop()
            return
            
        # Check if the buffer is filled
        if len(self.frames_to_display) >= self.preroll_target:
            self.preroll_timer.stop()
            self.playback_started = True
            print(f"Preroll buffer filled ({len(self.frames_to_display)} frames). Starting playback components...")
            
            # Call the dedicated playback start function
            self._start_synchronized_playback()
            
        else:
            # Not ready yet, keep waiting
            print(f"Buffering... {len(self.frames_to_display)} / {self.preroll_target}")

    def _feeder_loop(self):
        """
        This function runs in a separate thread (self.feeder_thread).
        Its only job is to read frames from the source and send them to the workers,
        based on the current processing mode.
        """
        print(f"Feeder thread started (Mode: {self.file_type}, Segments: {self.is_processing_segments}).")
        
        # Determine which feed logic to use
        try:
            if self.file_type == "webcam":
                self._feed_webcam()
            elif self.file_type == "video": # Handles both standard video and segment video
                self._feed_video_loop()
            else:
                print(f"[ERROR] Feeder thread: Unknown mode (file_type: {self.file_type}).")
                
        except Exception as e:
            print(f"[ERROR] Unhandled exception in feeder thread: {e}")
            
        print("Feeder thread finished.")

    def _feed_video_loop(self):
        """
        Unified feeder logic for standard video playback AND segment recording.
        Reads frames as long as processing is active and within the limits
        (full video or current segment).
        """
        
        # Determine the mode at startup
        is_segment_mode = self.is_processing_segments
        
        # Determine the stop condition (control variable)
        # In segment mode, the loop is controlled by 'is_processing_segments'
        # In video mode, the loop is controlled by 'processing'
        stop_flag_check = lambda: self.is_processing_segments if is_segment_mode else self.processing

        print(f"Feeder: Starting video loop (Mode: {'Segment' if is_segment_mode else 'Standard'}).")

        while stop_flag_check():
            try:
                # 1. Mode-specific stop logic
                if is_segment_mode:
                    if self.current_segment_end_frame is None:
                        time.sleep(0.01) # Wait for the segment to be configured
                        continue
                    if self.current_frame_number > self.current_segment_end_frame:
                        # This segment is finished, the feeder's job is done.
                        # We break the loop to allow the thread to terminate cleanly.
                        # Previously, this was a time.sleep(0.01) loop, which
                        # caused the thread to become an orphan.
                        print(f"Feeder: Reached end of segment {self.current_segment_index + 1}. Stopping feed.")
                        break 
                else: # Standard mode
                    if self.current_frame_number > self.max_frame_number:
                        break  # End of video

                # 2. Buffer control (identical for both modes)
                in_flight_frames = len(self.frames_to_display) + self.frame_queue.qsize()
                if in_flight_frames >= self.max_display_buffer_size:
                    time.sleep(0.005) # Wait 5ms (buffer full)
                    continue 

                # 3. Frame reading (identical)
                # In segment mode, self.recording is False, so preview_mode=False
                # In standard mode, self.recording determines the preview_mode
                ret, frame_bgr = misc_helpers.read_frame(
                    self.media_capture, preview_mode=not self.recording and not is_segment_mode
                )
                if not ret:
                    print(f"[ERROR] Feeder: Could not read frame {self.current_frame_number} (Mode: {'Segment' if is_segment_mode else 'Standard'})!")
                    break  # Stop reading

                # 4. Send to worker (identical)
                frame_rgb = frame_bgr[..., ::-1]
                frame_num_to_process = self.current_frame_number
                
                self.frame_queue.put(frame_num_to_process)
                self.start_frame_worker(frame_num_to_process, frame_rgb)
                self.current_frame_number += 1
                
            except Exception as e:
                print(f"[ERROR] Error in _feed_video_loop (Mode: {'Segment' if is_segment_mode else 'Standard'}): {e}")
                if is_segment_mode:
                    self.is_processing_segments = False
                else:
                    self.processing = False # Stop the loop

    def _feed_webcam(self):
        """Feeder logic for webcam streaming."""
        while self.processing:
            try:
                in_flight_frames = len(self.webcam_frames_to_display.queue) + self.frame_queue.qsize()
                if in_flight_frames >= self.max_display_buffer_size:
                    time.sleep(0.005) # Wait 5ms (buffer full)
                    continue

                ret, frame_bgr = misc_helpers.read_frame(
                    self.media_capture, preview_mode=False
                )
                if not ret:
                    print("[WARN] Feeder: Failed to read webcam frame.")
                    continue  # Try again

                frame_rgb = frame_bgr[..., ::-1]
                self.frame_queue.put(0)  # Frame number is not relevant
                self.start_frame_worker(0, frame_rgb, is_single_frame=False)
                
            except Exception as e:
                print(f"[ERROR] Error in _feed_webcam loop: {e}")
                self.processing = False
            
    def display_next_frame(self):
        """
        The core metronome loop.
        This function is called repeatedly via QTimer.singleShot. It handles:
        1. Precise timing to schedule the *next* call.
        2. Popping the *current* processed frame from the display buffer.
        3. Displaying the frame, sending it to virtualcam, and writing it to FFmpeg.
        """
        
        # 1. Stop check
        if not self.processing:  # General check (if stop_processing was called)
            self.stop_processing()  # Final cleanup
            return

        # 2. End-of-media / End-of-segment logic
        should_stop_playback = False
        should_finalize_default_recording = False
        if self.file_type == "video":
            if self.is_processing_segments:
                # --- Segment Recording Stop Logic ---
                if (
                    self.current_segment_end_frame is not None
                    and self.next_frame_to_display > self.current_segment_end_frame
                ):
                    print(
                        f"Segment {self.current_segment_index + 1} end frame ({self.current_segment_end_frame}) reached."
                    )
                    self.stop_current_segment()  # Segment logic handles its own stop
                    return
            elif self.next_frame_to_display > self.max_frame_number:
                # --- Default Playback/Recording Stop Logic ---
                print("End of media reached.")
                if self.recording:
                    should_finalize_default_recording = True
                else:
                    should_stop_playback = True

            if should_finalize_default_recording:
                self._finalize_default_style_recording()
                return
            elif should_stop_playback:
                self.stop_processing()
                return

        # --- 3. METRONOME TIMING LOGIC ---
        # This logic ensures precise timing and prevents clock drift.
        # We schedule the next call based on the *last scheduled time*,
        # not the *current time*.

        now_sec = time.perf_counter()
        
        # Calculate next tick time (based on *last* scheduled time to prevent drift)
        self.last_display_schedule_time_sec += self.target_delay_sec

        # Catch up if we are late
        # If processing took too long, the next scheduled time might be in the past.
        if self.last_display_schedule_time_sec < now_sec:
            # We are late. Schedule the next tick "now" (or 1ms in the future)
            self.last_display_schedule_time_sec = now_sec + 0.001 

        # Calculate actual wait time
        wait_time_sec = self.last_display_schedule_time_sec - now_sec
        wait_ms = int(wait_time_sec * 1000)
        
        if wait_ms <= 0:
            wait_ms = 1  # Just in case, wait at least 1ms

        # --- 4. Schedule the *next* call IMMEDIATELY ---
        # We use singleShot for precision.
        if self.processing:
            QTimer.singleShot(wait_ms, self.display_next_frame)
            
        # --- 6. Get the frame to display (if ready) ---
        pixmap = None
        frame = None
        frame_number_to_display = 0 # Used for UI update

        if self.file_type == "webcam":
            # --- Webcam Logic (Queue) ---
            if self.webcam_frames_to_display.empty():
                return # Frame not ready, skip display
            pixmap, frame = self.webcam_frames_to_display.get()
            frame_number_to_display = 0 # Not relevant for webcam
        
        else:
            # --- Video/Image Logic (Dictionary) ---
            frame_number_to_display = self.next_frame_to_display
            if frame_number_to_display not in self.frames_to_display:
                # Frame not ready.
                # If the buffer is not full, video will stutter.
                # print(f"[DEBUG] Frame {frame_number_to_display} not in buffer. Display may stutter.")
                return
            pixmap, frame = self.frames_to_display.pop(frame_number_to_display)

        
        # --- 7. Frame is ready: Process and Display ---
        self.current_frame = frame  # Update current frame state

        # Emit a signal every 500 frames to notify JobProcessor we are still alive
        if self.file_type != "webcam": # Don't spam on webcam
            self.heartbeat_frame_counter += 1
            if self.heartbeat_frame_counter >= 500:
                self.heartbeat_frame_counter = 0
                self.processing_heartbeat_signal.emit()

        # Send to Virtual Cam
        self.send_frame_to_virtualcam(frame)

        # Write to FFmpeg
        if self.is_processing_segments or self.recording:
            if (
                self.recording_sp
                and self.recording_sp.stdin
                and not self.recording_sp.stdin.closed
            ):
                try:
                    self.recording_sp.stdin.write(frame.tobytes())
                except OSError as e:
                    log_prefix = (
                        f"segment {self.current_segment_index + 1}"
                        if self.is_processing_segments
                        else "recording"
                    )
                    print(
                        f"[WARN] Error writing frame {frame_number_to_display} to FFmpeg stdin during {log_prefix}: {e}"
                    )
            else:
                log_prefix = (
                    f"segment {self.current_segment_index + 1}"
                    if self.is_processing_segments
                    else "recording"
                )
                print(
                    f"[WARN] FFmpeg stdin not available for {log_prefix} when trying to write frame {frame_number_to_display}."
                )

        # Update UI
        if not self.is_processing_segments and self.file_type != "webcam":
            video_control_actions.update_widget_values_from_markers(
                self.main_window, frame_number_to_display
            )

        graphics_view_actions.update_graphics_view(
            self.main_window, pixmap, frame_number_to_display
        )

        # --- 8. Clean up and Increment ---
        if self.file_type != "webcam":
            # Clean up thread
            if frame_number_to_display in self.threads:
                self.threads.pop(frame_number_to_display)

            # Increment for next frame
            self.next_frame_to_display += 1

    def send_frame_to_virtualcam(self, frame: numpy.ndarray):
        """Sends the given frame to the pyvirtualcam device, if enabled."""
        if self.main_window.control["SendVirtCamFramesEnableToggle"] and self.virtcam:
            height, width, _ = frame.shape
            if self.virtcam.height != height or self.virtcam.width != width:
                self.enable_virtualcam()  # Re-enable with new dimensions

            # Need to check again if virtcam was successfully re-enabled
            if self.virtcam:
                try:
                    self.virtcam.send(frame)
                    self.virtcam.sleep_until_next_frame()
                except Exception as e:
                    print(f"[WARN] Failed sending frame to virtualcam: {e}")

    def set_number_of_threads(self, value):
        """Stops processing and updates the thread count for workers."""
        self.stop_processing()  # Stop any active processing before changing threads
        self.main_window.models_processor.set_number_of_threads(value)
        self.num_threads = value
        self.frame_queue = queue.Queue(maxsize=self.num_threads)
        print(f"Max Threads set as {value} ")

    def process_video(self):
        """
        Start video processing.
        This can be either simple playback OR "default-style" recording.
        """
        
        # 1. Determine target FPS
        if self.main_window.control["VideoPlaybackCustomFpsToggle"]:
            # Custom FPS mode is enabled
            self.fps = self.main_window.control["VideoPlaybackCustomFpsSlider"]
        else:
            # Custom FPS mode is DISABLED, use original
            self.fps = self.media_capture.get(cv2.CAP_PROP_FPS)
            if self.fps <= 0:
                self.fps = 30 # Fallback
        
        # 2. Guards
        if self.processing or self.is_processing_segments:  # Check both flags
            print(
                "Processing already in progress (play or segment). Ignoring start request."
            )
            return

        if self.file_type != "video":
            print("process_video: Only applicable for video files.")
            return

        if not (self.media_capture and self.media_capture.isOpened()):
            print("Error: Unable to open the video source.")
            self.processing = False
            self.recording = False
            self.is_processing_segments = False
            video_control_actions.reset_media_buttons(self.main_window)
            return

        mode = "recording (default-style)" if self.recording else "playback"
        print(f"Starting video {mode} processing setup...")

        # 3. Set State Flags
        self.processing = True  # General flag ON
        self.is_processing_segments = False
        self.playback_started = False

        # Check if this recording was initiated by the Job Manager
        job_mgr_flag = getattr(self.main_window, "job_manager_initiated_record", False)
        if self.recording and job_mgr_flag:
            self.triggered_by_job_manager = True
            print("Detected default-style recording initiated by Job Manager.")
        else:
            self.triggered_by_job_manager = False
        try:
            self.main_window.job_manager_initiated_record = False
        except Exception:
            pass

        # 4. Setup Recording (if applicable)
        if self.recording:
            # Disable UI elements
            if not self.main_window.control["KeepControlsToggle"]:
                layout_actions.disable_all_parameters_and_control_widget(
                    self.main_window
                )
            # Create the ffmpeg subprocess
            if not self.create_ffmpeg_subprocess(output_filename=None):
                print("[ERROR] Failed to start FFmpeg for default-style recording.")
                self.stop_processing()  # Abort the start
                return

        # 5. Setup Audio (if applicable)
        # Note: Audio is not started here. It's started by
        # _start_synchronized_playback after the preroll buffer is full.

        # 6. Reset Timers and Containers
        self.start_time = time.perf_counter()
        self.frames_to_display.clear()
        self.threads.clear()

       # --- 7. AUDIO/VIDEO SYNC LOGIC (MODIFIED) ---

        # 7a. Get the target frame
        actual_start_frame = self.main_window.videoSeekSlider.value()
        print(f"Sync: Seeking directly to frame {actual_start_frame}...")

        # 7b. Set the capture position
        self.media_capture.set(cv2.CAP_PROP_POS_FRAMES, actual_start_frame)

        # 7c. Read the frame using the LOCKED helper function ONCE for dimensions.
        print(f"Sync: Reading frame {actual_start_frame} (for dimensions/state) using locked helper...")
        ret, frame_bgr = misc_helpers.read_frame(
            self.media_capture, preview_mode=False # Always read full frame for consistency
        )
        print(f"Sync: Initial read complete (Result: {ret}).")

        if not ret:
            # Fallback logic... (Keep your existing fallback logic here)
            # ... [Fallback logic - attempts to read fallback_frame_to_try] ...
            fallback_frame = int(self.media_capture.get(cv2.CAP_PROP_POS_FRAMES))
            fallback_frame_to_try = max(0, fallback_frame - 1)
            print(f"[WARN] Failed initial read for frame {actual_start_frame}. Retrying from frame {fallback_frame_to_try}.")
            if fallback_frame_to_try == actual_start_frame:
                 print(f"[ERROR] Fallback frame is the same. Cannot proceed.")
                 self.stop_processing()
                 return
            self.media_capture.set(cv2.CAP_PROP_POS_FRAMES, fallback_frame_to_try)
            print(f"Sync: Retrying read for frame {fallback_frame_to_try} using locked helper...")
            ret, frame_bgr = misc_helpers.read_frame(
                self.media_capture, preview_mode=False
            )
            print(f"Sync: Retry read complete (Result: {ret}).")
            if not ret:
                print(f"[ERROR] Capture failed definitively near frame {actual_start_frame}.")
                self.stop_processing()
                return
            actual_start_frame = fallback_frame_to_try # Use the frame we successfully read

        # 7d. Frame is valid - Store for potential FFmpeg init, DO NOT PROCESS SYNC
        frame_rgb = numpy.ascontiguousarray(
            frame_bgr[..., ::-1]
        )  # BGR to RGB
        self.current_frame = frame_rgb # Store for FFmpeg dimensions

        # !!! CRITICAL: Reset position AGAIN so the feeder reads this frame too !!!
        print(f"Sync: Resetting position to frame {actual_start_frame} for feeder thread...")
        self.media_capture.set(cv2.CAP_PROP_POS_FRAMES, actual_start_frame)
        print(f"Sync: Position reset complete.")

        # 7e. REMOVE SYNCHRONOUS PROCESSING STEP
        # The feeder thread will now handle queueing frame 'actual_start_frame'

        # 7f. Update counters - Feeder will start reading FROM actual_start_frame
        self.next_frame_to_display = actual_start_frame # Display starts here once buffered
        self.processing_start_frame = actual_start_frame
        self.current_frame_number = actual_start_frame # Feeder reads this frame first when it starts

        # Calculate play_start_time based on the confirmed actual_start_frame
        self.play_start_time = (
            float(actual_start_frame / float(self.fps)) if self.fps > 0 else 0.0
        )
        print(f"Recording audio start time set to: {self.play_start_time:.3f}s (Frame: {actual_start_frame})")

        # 7g. Update the slider (if needed, ensure signals blocked/unblocked correctly)
        self.main_window.videoSeekSlider.blockSignals(True)
        self.main_window.videoSeekSlider.setValue(actual_start_frame)
        self.main_window.videoSeekSlider.blockSignals(False)

        # --- 8. STARTING THE FEEDER THREAD AND METRONOME ---
        print(f"Starting feeder thread (Mode: video, Recording: {self.recording})...")
        self.feeder_thread = threading.Thread(target=self._feeder_loop, daemon=True)
        self.feeder_thread.start()

        if self.recording:
            # Recording: start the display metronome immediately
            print("Recording mode: Starting metronome immediately.")
            self._start_metronome(9999.0, is_first_start=True)
        else:
            if self.main_window.control.get("VideoPlaybackBufferingToggle", False):
                # Playback: start the preroll monitor
                print(f"Playback mode: Waiting for preroll buffer (target: {self.preroll_target} frames)...")
                
                # Ensure the connection is clean (avoids multiple connections)
                try:
                    self.preroll_timer.timeout.disconnect(self._check_preroll_and_start_playback)
                except RuntimeError:
                    pass # Disconnection failed, which is normal the first time
                
                self.preroll_timer.timeout.connect(self._check_preroll_and_start_playback)
                self.preroll_timer.start(100)
            else:
                # Recording: start the display metronome immediately
                print("Playback mode.")
                self._start_synchronized_playback()

    def start_frame_worker(self, frame_number, frame, is_single_frame=False):
        """Starts a FrameWorker to process the given frame."""
        worker = FrameWorker(
            frame, self.main_window, frame_number, self.frame_queue, is_single_frame
        )
        self.threads[frame_number] = worker
        if is_single_frame:
            worker.run()  # Process synchronously (blocks until done)
        else:
            worker.start()  # Process asynchronously (in a new thread)

    def process_current_frame(self):
        """
        Process the single, currently selected frame (e.g., after seek or for image).
        This is a one-shot operation, not part of the metronome.
        """
        if self.processing or self.is_processing_segments:
            print("[INFO] Stopping active processing to process single frame.")
            if not self.stop_processing():
                print(
                    "[WARN] Could not stop active processing cleanly."
                )

        # Set frame number for processing
        if self.file_type == "video":
            self.current_frame_number = self.main_window.videoSeekSlider.value()
        elif self.file_type == "image" or self.file_type == "webcam":
            self.current_frame_number = 0

        self.next_frame_to_display = self.current_frame_number

        frame_to_process = None
        read_successful = False

        # --- Read the frame based on file type ---
        if self.file_type == "video" and self.media_capture:
            self.media_capture.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_number)
            ret, frame_bgr = misc_helpers.read_frame(
                self.media_capture, preview_mode=False
            )
            if ret:
                frame_to_process = frame_bgr[..., ::-1]  # BGR to RGB
                read_successful = True
                self.media_capture.set(
                    cv2.CAP_PROP_POS_FRAMES, self.current_frame_number
                )
            else:
                print(
                    f"[ERROR] Cannot read frame {self.current_frame_number} for single processing!"
                )
                self.main_window.last_seek_read_failed = True
                self.main_window.display_messagebox_signal.emit(
                    "Error Reading Frame",
                    f"Error Reading Frame {self.current_frame_number}.",
                    self.main_window,
                )

        elif self.file_type == "image":
            frame_bgr = misc_helpers.read_image_file(self.media_path)
            if frame_bgr is not None:
                frame_to_process = frame_bgr[..., ::-1]  # BGR to RGB
                read_successful = True
            else:
                print("[ERROR] Unable to read image file for processing.")

        elif self.file_type == "webcam" and self.media_capture:
            ret, frame_bgr = misc_helpers.read_frame(
                self.media_capture, preview_mode=False
            )
            if ret:
                frame_to_process = frame_bgr[..., ::-1]  # BGR to RGB
                read_successful = True
            else:
                print("[ERROR] Unable to read Webcam frame for processing!")

        # --- Process if read was successful ---
        if read_successful and frame_to_process is not None:
            # Clear any pending workers
            with self.frame_queue.mutex:
                self.frame_queue.queue.clear()
            self.threads.clear()

            # Process this frame synchronously
            self.frame_queue.put(self.current_frame_number)
            self.start_frame_worker(
                self.current_frame_number, frame_to_process, is_single_frame=True
            )

    def stop_processing(self):
        """
        General Stop / Abort Function.
        This is the master function to stop *any* active processing
        (playback, recording, segments, webcam).
        It stops timers, threads, ffmpeg, and cleans up temp files/dirs.
        """
        if not self.processing and not self.is_processing_segments:
            video_control_actions.reset_media_buttons(self.main_window)
            return False  # Nothing was stopped

        print("stop_processing called: Aborting active processing...")
        was_processing_segments = self.is_processing_segments
        was_recording_default_style = self.recording

        # 1. Reset flags FIRST to stop all loops
        self.processing = False
        self.is_processing_segments = False
        self.recording = False
        self.triggered_by_job_manager = False

        # 2. Stop utility timers and audio
        self.gpu_memory_update_timer.stop()
        self.preroll_timer.stop()
        self.stop_live_sound()

        # 3a. Wait for the feeder thread (ADDED)
        print("Waiting for feeder thread to complete...")
        if self.feeder_thread and self.feeder_thread.is_alive():
            self.feeder_thread.join(timeout=2.0) # Wait 2 seconds
            if self.feeder_thread.is_alive():
                print("[WARN] Feeder thread did not join gracefully.")
        self.feeder_thread = None
        print("Feeder thread joined.")
        
        # 3b. Wait for worker threads
        print("Waiting for worker threads to complete...")
        self.join_and_clear_threads()
        print("Worker threads joined.")

        # 4. Clear frame storage
        self.frames_to_display.clear()
        self.webcam_frames_to_display.queue.clear()
        with self.frame_queue.mutex:
            self.frame_queue.queue.clear()

        # 5. Stop and cleanup ffmpeg
        if self.recording_sp:
            print("Closing and waiting for active FFmpeg subprocess...")
            if self.recording_sp.stdin and not self.recording_sp.stdin.closed:
                try:
                    self.recording_sp.stdin.close()
                except OSError as e:
                    print(f"[WARN] Error closing ffmpeg stdin during abort: {e}")
            try:
                self.recording_sp.wait(timeout=5)
                print("FFmpeg subprocess terminated.")
            except subprocess.TimeoutExpired:
                print("[WARN] FFmpeg subprocess did not terminate gracefully, killing.")
                self.recording_sp.kill()
                self.recording_sp.wait()
            except Exception as e:
                print(f"[ERROR] Error waiting for FFmpeg subprocess: {e}")
            self.recording_sp = None

        # 6. Cleanup temp files/dirs based on what was running
        if was_processing_segments:
            print("Cleaning up segment temporary directory due to abort.")
            self._cleanup_temp_dir()
        elif was_recording_default_style:
            print("Cleaning up default-style temporary file due to abort.")
            if self.temp_file and os.path.exists(self.temp_file):
                try:
                    os.remove(self.temp_file)
                    print(f"Removed temporary file: {self.temp_file}")
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
        self.playback_display_start_time = 0.0 # Reset display start time

        # 8. Reset capture position
        if self.file_type == "video" and self.media_capture:
            try:
                current_slider_pos = self.main_window.videoSeekSlider.value()
                self.current_frame_number = current_slider_pos
                self.next_frame_to_display = current_slider_pos
                if self.media_capture.isOpened():
                    self.media_capture.set(cv2.CAP_PROP_POS_FRAMES, current_slider_pos)
            except Exception as e:
                print(f"[WARN] Could not reset video capture position: {e}")

        # 9. Re-enable UI
        if was_processing_segments or was_recording_default_style:
            layout_actions.enable_all_parameters_and_control_widget(self.main_window)

        # 10. Final cleanup
        print("Clearing GPU Cache and running garbage collection.")
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        except Exception as e:
            print(f"[WARN] Error clearing Torch cache: {e}")
        gc.collect()

        video_control_actions.reset_media_buttons(self.main_window)
        try:
            self.disable_virtualcam()
        except Exception:
            pass
        print("Processing aborted and cleaned up.")

        end_frame_for_calc = min(self.next_frame_to_display, self.max_frame_number + 1)
        self.play_end_time = (
            float(end_frame_for_calc / float(self.fps)) if self.fps > 0 else 0.0
        )
        print(
            f"Calculated recording end time: {self.play_end_time:.3f}s (based on frame {end_frame_for_calc})"
        )

        # 11. Final Timing and Logging
        self.end_time = time.perf_counter()
        processing_time_sec = self.end_time - self.start_time
        
        try:
            # Calculate processed frames
            start_frame_num = getattr(self, 'processing_start_frame', end_frame_for_calc)
            num_frames_processed = end_frame_for_calc - start_frame_num
            if num_frames_processed < 0:
                num_frames_processed = 0
        except Exception:
            num_frames_processed = 0 # Safety fallback
        
        # Log the summary
        self._log_processing_summary(processing_time_sec, num_frames_processed)

        # Emit signal to notify other components (like JobProcessor) that processing has ended
        self.processing_stopped_signal.emit()
        
        return True  # Processing was stopped

    def join_and_clear_threads(self):
        """Waits for all active worker threads to finish and clears the list."""
        active_threads = list(self.threads.values())
        for thread in active_threads:
            try:
                if thread.is_alive():
                    thread.join(timeout=1.0)
                    if thread.is_alive():
                        print(f"[WARN] Thread {thread.name} did not join gracefully.")
            except Exception as e:
                print(f"[WARN] Error joining thread {thread.name}: {e}")
        self.threads.clear()

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

    def _log_processing_summary(self, processing_time_sec: float, num_frames_processed: int):
        """
        Calculates and prints the final processing time and average FPS.
        Uses the actual display duration for FPS calculation if playback occurred.

        :param processing_time_sec: Total duration of the processing (start to end).
        :param num_frames_processed: Total number of frames displayed or intended for display.
        """

        # 1. Print formatted duration (overall processing time)
        formatted_duration = self._format_duration(processing_time_sec)
        print(f"\nProcessing completed in {formatted_duration}")

        # 2. Calculate and print FPS (based on actual display time)
        display_duration_sec = 0.0
        # Check if playback actually started displaying frames
        if self.playback_display_start_time > 0 and self.end_time > self.playback_display_start_time:
            display_duration_sec = self.end_time - self.playback_display_start_time
            print(f"(Actual display duration: {self._format_duration(display_duration_sec)})")
        else:
            # Playback might have stopped during preroll or it was a recording-only task
            # Use the overall time, but mention it includes setup/buffering
            display_duration_sec = processing_time_sec
            if self.start_time != self.playback_display_start_time : # Check if display never started
                 print("(Note: FPS calculation includes initial buffering/setup time)")


        try:
            if display_duration_sec > 0.01 and num_frames_processed > 0: # Use a small threshold for duration
                avg_fps = num_frames_processed / display_duration_sec
                print(f"Average Display FPS: {avg_fps:.2f}\n")
            elif num_frames_processed > 0:
                print("Display duration too short to calculate meaningful FPS.\n")
            else:
                 print("No frames were displayed or duration was zero, cannot calculate FPS.\n")
        except Exception as e:
            print(f"[WARN] Could not calculate average FPS: {e}\n")

    # --- FFmpeg and Finalization ---

    def create_ffmpeg_subprocess(self, output_filename: str):
        """
        Creates the FFmpeg subprocess for recording.
        This is a merged function used by both default-style and multi-segment recording.
        
        :param output_filename: The direct output path. If None, it's default-style
                                recording and a temp file will be generated.
        """
        control = self.main_window.control.copy()
        is_segment = (output_filename is not None)

        # 1. Guards
        if (
            not isinstance(self.current_frame, numpy.ndarray)
            or self.current_frame.size == 0
        ):
            print("[ERROR] Current frame invalid. Cannot get dimensions.")
            return False
        if not self.media_path or not Path(self.media_path).is_file():
            print("[ERROR] Original media path invalid.")
            return False
        if self.fps <= 0:
            print("[ERROR] Invalid FPS.")
            return False
            
        start_time_sec = 0.0
        end_time_sec = 0.0
        
        if is_segment:
            if self.current_segment_index < 0 or self.current_segment_index >= len(
                self.segments_to_process
            ):
                print(f"[ERROR] Invalid segment index {self.current_segment_index}.")
                return False
            start_frame, end_frame = self.segments_to_process[
                self.current_segment_index
            ]
            start_time_sec = start_frame / self.fps
            end_time_sec = end_frame / self.fps

        # 2. Frame Dimensions
        frame_height, frame_width, _ = self.current_frame.shape
        if is_segment:
            # Adjust dimensions based on frame enhancer
            # Note: Frame enhancer scaling is only applied to segments here, not default-style.
            if control["FrameEnhancerEnableToggle"]:
                if control["FrameEnhancerTypeSelection"] in (
                    "RealEsrgan-x2-Plus",
                    "BSRGan-x2",
                ):
                    frame_height = frame_height * 2
                    frame_width = frame_width * 2
                elif control["FrameEnhancerTypeSelection"] in (
                    "RealEsrgan-x4-Plus",
                    "BSRGan-x4",
                    "UltraSharp-x4",
                    "UltraMix-x4",
                    "RealEsr-General-x4v3",
                ):
                    frame_height = frame_height * 4
                    frame_width = frame_width * 4
        
        # Calculate downscale dimensions
        frame_height_down = frame_height
        frame_width_down = frame_width
        if control["FrameEnhancerDownToggle"]:
            if frame_width != 1920 or frame_height != 1080:
                frame_width_down_mult = frame_width / 1920
                frame_height_down = math.ceil(frame_height / frame_width_down_mult)
                frame_width_down = 1920
            else:
                print("Already 1920*1080")

        # 3. Output File Path and Logging
        if is_segment:
            segment_num = self.current_segment_index + 1
            print(
                f"Creating FFmpeg (Segment {segment_num}): Video Dim={frame_width}x{frame_height}, FPS={self.fps}, Output='{output_filename}'"
            )
            print(
                f"  Audio Segment: Start={start_time_sec:.3f}s, End={end_time_sec:.3f}s (Frames {start_frame}-{end_frame})"
            )

            if Path(output_filename).is_file():
                try:
                    os.remove(output_filename)
                except OSError as e:
                    print(
                        f"[WARN] Could not remove existing segment file {output_filename}: {e}"
                    )
        else:
            # Default-style: create a unique temp file
            date_and_time = datetime.now().strftime(r"%Y_%m_%d_%H_%M_%S")
            try:
                base_temp_dir = os.path.join(os.getcwd(), "temp_files", "default")
                os.makedirs(base_temp_dir, exist_ok=True)
                self.temp_file = os.path.join(
                    base_temp_dir, f"temp_output_{date_and_time}.mp4"
                )
                print(f"Default temp file will be created at: {self.temp_file}")
            except Exception as e:
                print(f"[ERROR] Failed to create temporary directory/file path: {e}")
                self.temp_file = f"temp_output_{date_and_time}.mp4"
                print(
                    f"[WARN] Falling back to local directory for temp file: {self.temp_file}"
                )

            print(
                f"Creating FFmpeg : Video Dim={frame_width}x{frame_height}, FPS={self.fps}, Temp Output='{self.temp_file}'"
            )

            if Path(self.temp_file).is_file():
                try:
                    os.remove(self.temp_file)
                except OSError as e:
                    print(
                        f"[WARN] Could not remove existing temp file {self.temp_file}: {e}"
                    )

        # 4. Build FFmpeg Arguments
        hdrpreset = control["FFPresetsHDRSelection"]
        sdrpreset = control["FFPresetsSDRSelection"]
        ffquality = control["FFQualitySlider"]
        ffspatial = int(control["FFSpatialAQToggle"])
        fftemporal = int(control["FFTemporalAQToggle"])
        
        # Base args: read raw video from stdin
        args = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24", # The processed frame from FrameWorker is BGR
            "-s",
            f"{frame_width}x{frame_height}",
            "-r",
            str(self.fps),
            "-i",
            "pipe:0",  # Read from stdin
        ]
        
        if is_segment:
            # For segments, add the audio source and time limits
            args.extend(
                [
                    "-ss",
                    str(start_time_sec),
                    "-to",
                    str(end_time_sec),
                    "-i",
                    self.media_path,
                    "-map",
                    "0:v:0", # Map video from stdin
                    "-map",
                    "1:a:0?", # Map audio from media_path (if exists)
                    "-c:a",
                    "copy",
                    "-shortest",
                ]
            )
        
        # Video codec args
        if control["HDREncodeToggle"]:
            # HDR uses X265
            args.extend(
                [
                    "-c:v",
                    "libx265",
                    "-profile:v",
                    "main10",
                    "-preset",
                    str(hdrpreset),
                    "-pix_fmt",
                    "yuv420p10le",
                    "-x265-params",
                    f"crf={ffquality}:vbv-bufsize=10000:vbv-maxrate=10000:selective-sao=0:no-sao=1:strong-intra-smoothing=0:rect=0:aq-mode={ffspatial}:t-aq={fftemporal}:hdr-opt=1:repeat-headers=1:colorprim=bt2020:range=limited:transfer=smpte2084:colormatrix=bt2020nc:range=limited:master-display='G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,1)':max-cll=1000,400",
                ]
            )
        else:
            # NVENC for SDR
            args.extend(
                [
                    "-c:v",
                    "hevc_nvenc",
                    "-preset",
                    str(sdrpreset),
                    "-profile:v",
                    "main10",
                    "-cq",
                    str(ffquality),
                    "-pix_fmt",
                    "yuv420p10le",
                    "-colorspace",
                    "rgb",
                    "-color_primaries",
                    "bt709",
                    "-color_trc",
                    "bt709",
                    "-spatial-aq",
                    str(ffspatial),
                    "-temporal-aq",
                    str(fftemporal),
                    "-tier",
                    "high",
                    "-tag:v",
                    "hvc1",
                ]
            )
            
        # Downscale filter
        if control["FrameEnhancerDownToggle"]:
            args.extend(
                [
                    "-vf",
                    f"scale={frame_width_down}x{frame_height_down}:flags=lanczos+accurate_rnd+full_chroma_int",
                ]
            )
        
        # Output file
        if is_segment:
            args.extend([output_filename])
        else:
            args.extend([self.temp_file])
            
        # 5. Start Subprocess
        try:
            self.recording_sp = subprocess.Popen(
                args, stdin=subprocess.PIPE, bufsize=-1
            )
            return True
        except FileNotFoundError:
            print(
                "[ERROR] FFmpeg command not found. Ensure FFmpeg is installed and in system PATH."
            )
            self.main_window.display_messagebox_signal.emit(
                "FFmpeg Error", "FFmpeg command not found.", self.main_window
            )
            return False
        except Exception as e:
            print(f"[ERROR] Failed to start FFmpeg subprocess : {e}")
            if is_segment:
                self.main_window.display_messagebox_signal.emit(
                    "FFmpeg Error",
                    f"Failed to start FFmpeg for segment {segment_num}:\n{e}",
                    self.main_window,
                )
            else:
                self.main_window.display_messagebox_signal.emit(
                    "FFmpeg Error", f"Failed to start FFmpeg:\n{e}", self.main_window
                )
            return False

    def _finalize_default_style_recording(self):
        """Finalizes a successful default-style recording (adds audio, cleans up)."""
        print("Finalizing default-style recording...")
        self.processing = False # Stop metronome

        # 1. Stop timers
        self.gpu_memory_update_timer.stop()

        # 2. Wait for final frames
        print("Waiting for final worker threads...")
        self.join_and_clear_threads()
        self.frames_to_display.clear()

        # 3. Finalize ffmpeg (close stdin, wait for file to be written)
        if self.recording_sp:
            if self.recording_sp.stdin and not self.recording_sp.stdin.closed:
                try:
                    print("Closing FFmpeg stdin...")
                    self.recording_sp.stdin.close()
                except OSError as e:
                    print(f"[WARN] Error closing FFmpeg stdin during finalization: {e}")
            print("Waiting for FFmpeg subprocess to finish writing...")
            try:
                self.recording_sp.wait(timeout=10)
                print("FFmpeg subprocess finished.")
            except subprocess.TimeoutExpired:
                print(
                    "[WARN] FFmpeg subprocess timed out during finalization, killing."
                )
                self.recording_sp.kill()
                self.recording_sp.wait()
            except Exception as e:
                print(
                    f"[ERROR] Error waiting for FFmpeg subprocess during finalization: {e}"
                )
            self.recording_sp = None
        else:
            print(
                "[WARN] No recording subprocess found during finalization."
            )

        # 4. Calculate audio segment times
        end_frame_for_calc = min(self.next_frame_to_display, self.max_frame_number + 1)
        self.play_end_time = (
            float(end_frame_for_calc / float(self.fps)) if self.fps > 0 else 0.0
        )
        print(
            f"Calculated recording end time: {self.play_end_time:.3f}s (based on frame {end_frame_for_calc})"
        )

        # 5. Audio Merging
        if (
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

            final_file_path = misc_helpers.get_output_file_path(
                self.media_path,
                self.main_window.control["OutputMediaFolder"],
                job_name=job_name,
                use_job_name_for_output=use_job_name,
                output_file_name=output_file_name,
            )

            output_dir = os.path.dirname(final_file_path)
            
            # Check if output_dir is not an empty string before creating it
            # This prevents [WinError 3] if the path is just a filename.
            if output_dir and not os.path.exists(output_dir):
                try:
                    # Added exist_ok=True for thread-safety
                    os.makedirs(output_dir, exist_ok=True)
                    print(f"Created output directory: {output_dir}")
                except OSError as e:
                    print(
                        f"[ERROR] Failed to create output directory {output_dir}: {e}"
                    )
                    self.main_window.display_messagebox_signal.emit(
                        "File Error",
                        f"Could not create output directory:\n{output_dir}\n\n{e}",
                        self.main_window,
                    )
                    try:
                        os.remove(self.temp_file)
                    except OSError:
                        pass
                    self.temp_file = ""
                    layout_actions.enable_all_parameters_and_control_widget(
                        self.main_window
                    )
                    video_control_actions.reset_media_buttons(self.main_window)
                    self.recording = False
                    return

            if Path(final_file_path).is_file():
                print(f"Removing existing final file: {final_file_path}")
                try:
                    os.remove(final_file_path)
                except OSError as e:
                    print(
                        f"[WARN] Failed to remove existing final file {final_file_path}: {e}"
                    )

            # 5b. Run FFmpeg audio merge command
            print("Adding audio (default-style merge)...")
            args = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                self.temp_file, # Input 0: temp video (no audio)
                "-ss",
                str(self.play_start_time), # Start time for audio
                "-to",
                str(self.play_end_time), # End time for audio
                "-i",
                self.media_path, # Input 1: original media (for audio)
                "-c:v",
                "copy",
                "-map",
                "0:v:0", # Map video from input 0
                "-map",
                "1:a:0?", # Map audio from input 1 (if exists)
                "-shortest",
                "-af",
                "aresample=async=1000",
                final_file_path,
            ]
            try:
                subprocess.run(
                    args, check=True
                )
                print(
                    f"--- Successfully created final video (default-style): {final_file_path} ---"
                )
            except subprocess.CalledProcessError as e:
                print(
                    f"[ERROR] FFmpeg command failed during default-style audio merge: {e}"
                )
                print(f"FFmpeg arguments: {' '.join(args)}")
                self.main_window.display_messagebox_signal.emit(
                    "Recording Error",
                    f"FFmpeg command failed during audio merge:\n{e}\nCheck console for command.",
                    self.main_window,
                )
            except FileNotFoundError:
                print("[ERROR] FFmpeg not found. Cannot merge audio.")
                self.main_window.display_messagebox_signal.emit(
                    "Recording Error", "FFmpeg not found.", self.main_window
                )
            finally:
                # 5c. Clean up temp file
                print(f"Removing temporary file: {self.temp_file}")
                try:
                    os.remove(self.temp_file)
                except OSError as e:
                    print(f"[WARN] Failed to remove temp file {self.temp_file}: {e}")
                self.temp_file = ""
        else:
            if not self.temp_file:
                print("[WARN] No temporary file name recorded. Cannot merge audio.")
            elif not os.path.exists(self.temp_file):
                print(
                    f"[WARN] Temporary video file missing: {self.temp_file}. Cannot merge audio."
                )
            else:
                print(
                    f"[WARN] Temporary video file empty: {self.temp_file}. Cannot merge audio."
                )
                try:
                    os.remove(self.temp_file)
                except OSError:
                    pass
                self.temp_file = ""

        # 6. Final Timing and Logging
        self.end_time = time.perf_counter()
        processing_time_sec = self.end_time - self.start_time
        
        try:
            # Calculate processed frames
            start_frame_num = getattr(self, 'processing_start_frame', end_frame_for_calc)
            num_frames_processed = end_frame_for_calc - start_frame_num
            if num_frames_processed < 0:
                num_frames_processed = 0
        except Exception:
            num_frames_processed = 0 # Safety fallback

        # Log the summary
        self._log_processing_summary(processing_time_sec, num_frames_processed)

        # 7. Reset State and UI
        self.recording = False

        if self.main_window.control["AutoSaveWorkspaceToggle"]:
            json_file_path = misc_helpers.get_output_file_path(
                self.media_path, self.main_window.control["OutputMediaFolder"]
            )
            json_file_path += ".json"
            save_load_actions.save_current_workspace(self.main_window, json_file_path)

        layout_actions.enable_all_parameters_and_control_widget(self.main_window)
        video_control_actions.reset_media_buttons(self.main_window)

        # 8. Final Cleanup
        print("Clearing GPU Cache and running garbage collection post-recording.")
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        except Exception as e:
            print(f"[WARN] Error clearing Torch cache: {e}")
        gc.collect()

        video_control_actions.reset_media_buttons(self.main_window)
        try:
            self.disable_virtualcam()
        except Exception:
            pass
        print("default-style recording finalized.")

        if self.main_window.control["OpenOutputToggle"]:
            try:
                list_view_actions.open_output_media_folder(self.main_window)
            except Exception:
                pass

        # Emit signal to notify JobProcessor that processing has finished SUCCESSFULLY
        print("Emitting processing_stopped_signal (default-style success).")
        self.processing_stopped_signal.emit()

    # --- Virtual Camera Methods ---

    def enable_virtualcam(self, backend=False):
        """Starts the pyvirtualcam device."""

        # Guard: Only run if the user has actually enabled the virtual cam
        if not self.main_window.control.get("SendVirtCamFramesEnableToggle", False):
            # Ensure it's also disabled if the toggle is off
            self.disable_virtualcam()
            return

        if not self.media_capture and not isinstance(self.current_frame, numpy.ndarray):
            print(
                "[WARN] Cannot enable virtual camera without media loaded."
            )
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
            frame_width = int(self.media_capture.get(cv2.CAP_PROP_WIDTH))
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
        try:
            backend_to_use = (
                backend or self.main_window.control["VirtCamBackendSelection"]
            )
            print(
                f"Enabling virtual camera: {frame_width}x{frame_height} @ {int(current_fps)}fps, Backend: {backend_to_use}, Format: BGR"
            )
            self.virtcam = pyvirtualcam.Camera(
                width=frame_width,
                height=frame_height,
                fps=int(current_fps),
                backend=backend_to_use,
                fmt=pyvirtualcam.PixelFormat.BGR, # Processed frame is BGR
            )
            print(f"Virtual camera '{self.virtcam.device}' started.")
        except Exception as e:
            print(f"[ERROR] Failed to enable virtual camera: {e}")
            self.virtcam = None

    def disable_virtualcam(self):
        """Stops the pyvirtualcam device."""
        if self.virtcam:
            print(f"Disabling virtual camera '{self.virtcam.device}'.")
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

        print("--- Initializing multi-segment recording... ---")
        
        # 2. Set State Flags
        self.is_processing_segments = True
        self.recording = False
        self.processing = True # Master flag
        self.triggered_by_job_manager = triggered_by_job_manager
        self.segments_to_process = sorted(
            segments
        )
        self.current_segment_index = -1
        self.temp_segment_files = []
        self.segment_temp_dir = None

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
            print(f"Created temporary directory for segments: {self.segment_temp_dir}")
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
            print("All segments processed.")
            self.finalize_segment_concatenation()
            return

        # 3. Get segment details
        start_frame, end_frame = self.segments_to_process[self.current_segment_index]
        print(
            f"--- Starting Segment {segment_num}/{len(self.segments_to_process)} (Frames: {start_frame} - {end_frame}) ---"
        )
        self.current_segment_end_frame = end_frame

        if not self.media_capture or not self.media_capture.isOpened():
            print(
                f"[ERROR] Media capture not available for seeking to segment {segment_num}."
            )
            self.stop_processing()
            return

        # 4. Seek to the start frame of the segment
        print(f"Seeking to start frame {start_frame}...")
        self.media_capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        ret, frame_bgr = misc_helpers.read_frame(self.media_capture, preview_mode=False)
        if ret:
            self.current_frame = numpy.ascontiguousarray(
                frame_bgr[..., ::-1]
            )  # BGR to RGB
            # Must re-set position, as read() advances it
            self.media_capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame) 
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

        # 5. Clear containers for the new segment
        self.frames_to_display.clear()
        with self.frame_queue.mutex:
            self.frame_queue.queue.clear()
        self.threads.clear()

        # 6. Setup FFmpeg subprocess for this segment
        temp_segment_filename = (
            f"segment_{self.current_segment_index:03d}.mp4"
        )
        temp_segment_path = os.path.join(self.segment_temp_dir, temp_segment_filename)
        self.temp_segment_files.append(temp_segment_path)

        if not self.create_ffmpeg_subprocess(output_filename=temp_segment_path):
            print(
                f"[ERROR] Failed to create ffmpeg subprocess for segment {segment_num}. Aborting."
            )
            self.stop_processing()
            return

        # 7. Synchronously process the first frame of the segment
        #    (This mirrors the logic in process_video for a smooth start)
        current_start_frame = self.current_frame_number
        print(f"Sync: Synchronously processing first frame {current_start_frame} of segment...")
        with self.frame_queue.mutex:
            self.frame_queue.queue.clear()
        self.frame_queue.put(current_start_frame)
        self.start_frame_worker(current_start_frame, self.current_frame, is_single_frame=True)
        # Now, self.frames_to_display[current_start_frame] is ready.
        
        # 8. Update counters
        # self.current_frame_number was set to start_frame (e.g., 100)
        # We must increment it so the *next* read is correct (e.g., 101)
        self.current_frame_number += 1 

        # 9. Start Metronome ET Feeder
        target_fps = 9999.0 # Always max speed for segments
        is_first = (self.current_segment_index == 0) 

        # Start the feeder thread
        print(f"Starting feeder thread (Mode: segment {self.current_segment_index})...")
        self.feeder_thread = threading.Thread(target=self._feeder_loop, daemon=True)
        self.feeder_thread.start()
        
        # Start the display metronome
        self._start_metronome(target_fps, is_first_start=is_first)

    def stop_current_segment(self):
        """
        Stops processing the *current* segment, finalizes its file,
        and triggers the next segment or final concatenation.
        """
        if not self.is_processing_segments:
            print("[WARN] stop_current_segment called but not processing segments.")
            return

        segment_num = self.current_segment_index + 1
        print(f"--- Stopping Segment {segment_num} --- ")
        
        # 1. Stop timers
        self.gpu_memory_update_timer.stop()

        # 2a. Wait for the feeder thread
        print(f"Waiting for feeder thread from segment {segment_num}...")
        if self.feeder_thread and self.feeder_thread.is_alive():
            self.feeder_thread.join(timeout=2.0)
            
            # Add a check to see if the join timed out
            if self.feeder_thread.is_alive():
                print(f"[WARN] Feeder thread from segment {segment_num} did not join gracefully.")
            else:
                print("Feeder thread joined.")
            
        else:
            # This case is normal if the feeder finished its work very quickly
            print("Feeder thread was already finished.")

        self.feeder_thread = None
        
        # 2b. Wait for workers
        print(f"Waiting for workers from segment {segment_num}...")
        self.join_and_clear_threads()
        print("Workers joined.")
        self.frames_to_display.clear()

        # 3. Finalize FFmpeg for this segment
        if self.recording_sp:
            if self.recording_sp.stdin and not self.recording_sp.stdin.closed:
                try:
                    print(f"Closing FFmpeg stdin for segment {segment_num}...")
                    self.recording_sp.stdin.close()
                except OSError as e:
                    print(
                        f"[WARN] Error closing FFmpeg stdin for segment {segment_num}: {e}"
                    )
            print(
                f"Waiting for FFmpeg subprocess (segment {segment_num}) to finish writing..."
            )
            try:
                self.recording_sp.wait(timeout=10)
                print(f"FFmpeg subprocess (segment {segment_num}) finished.")
            except subprocess.TimeoutExpired:
                print(
                    f"[WARN] FFmpeg subprocess (segment {segment_num}) timed out, killing."
                )
                self.recording_sp.kill()
                self.recording_sp.wait()
            except Exception as e:
                print(
                    f"[ERROR] Error waiting for FFmpeg subprocess (segment {segment_num}): {e}"
                )
            self.recording_sp = None
        else:
            print(
                f"[WARN] No active FFmpeg subprocess found when stopping segment {segment_num}."
            )

        if self.temp_segment_files and not os.path.exists(self.temp_segment_files[-1]):
            print(
                f"[ERROR] Segment file '{self.temp_segment_files[-1]}' not found after processing segment {segment_num}."
            )

        # 4. Process the *next* segment
        self.process_next_segment()

    def finalize_segment_concatenation(self):
        """Concatenates all valid temporary segment files into the final output file."""
        print("--- Finalizing concatenation of segments... ---")

        # Failsafe: If this is called while an ffmpeg process is still running
        if self.recording_sp:
            segment_num = self.current_segment_index + 1
            print(
                f"Finalizing: Stopping active FFmpeg process for segment {segment_num}..."
            )
            if self.recording_sp.stdin and not self.recording_sp.stdin.closed:
                try:
                    self.recording_sp.stdin.close()
                except OSError as e:
                    print(
                        f"[WARN] Error closing FFmpeg stdin during early finalization: {e}"
                    )
            try:
                self.recording_sp.wait(timeout=10)
                print(f"FFmpeg subprocess (segment {segment_num}) finished writing.")
            except subprocess.TimeoutExpired:
                print(
                    f"[WARN] FFmpeg subprocess (segment {segment_num}) timed out, killing."
                )
                self.recording_sp.kill()
                self.recording_sp.wait()
            except Exception as e:
                print(
                    f"[ERROR] Error waiting for FFmpeg subprocess: {e}"
                )
            self.recording_sp = None

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
            print(
                "[WARN] No valid temporary segment files found to concatenate."
            )
            self._cleanup_temp_dir()
            layout_actions.enable_all_parameters_and_control_widget(self.main_window)
            video_control_actions.reset_media_buttons(self.main_window)
            self.segments_to_process = []
            self.current_segment_index = -1
            self.temp_segment_files = []
            self.triggered_by_job_manager = False
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

        final_file_path = misc_helpers.get_output_file_path(
            self.media_path,
            self.main_window.control["OutputMediaFolder"],
            job_name=job_name,
            use_job_name_for_output=use_job_name,
            output_file_name=output_file_name,
        )

        output_dir = os.path.dirname(final_file_path)

        # Check if output_dir is not an empty string before creating it
        # This prevents [WinError 3] if the path is just a filename.
        if output_dir and not os.path.exists(output_dir):
            try:
                # Added exist_ok=True for thread-safety
                os.makedirs(output_dir, exist_ok=True)
                print(f"Created output directory: {output_dir}")
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
                return

        if Path(final_file_path).is_file():
            print(f"Removing existing final file: {final_file_path}")
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
                return

        # 4. Create FFmpeg list file
        list_file_path = os.path.join(self.segment_temp_dir, "mylist.txt")
        concatenation_successful = False
        try:
            print(f"Creating ffmpeg list file: {list_file_path}")
            with open(
                list_file_path, "w", encoding="utf-8"
            ) as f_list:
                for segment_path in valid_segment_files:
                    abs_path = os.path.abspath(segment_path)
                    # FFmpeg concat requires forward slashes, even on Windows
                    formatted_path = abs_path.replace("\\", "/")
                    f_list.write(f"file '{formatted_path}'" + os.linesep)

            # 5. Run final concatenation command
            print(
                f"Concatenating {len(valid_segment_files)} valid segments into {final_file_path}..."
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
                "-af",
                "aresample=async=1000",
                final_file_path,
            ]
            subprocess.run(
                concat_args, check=True
            )
            concatenation_successful = True
            log_prefix = "Job Manager: " if was_triggered_by_job else ""
            print(
                f"--- {log_prefix}Successfully created final video: {final_file_path} ---"
            )

        except subprocess.CalledProcessError as e:
            print(f"[ERROR] FFmpeg command failed during final concatenation: {e}")
            print(f"FFmpeg arguments: {' '.join(concat_args)}")
            self.main_window.display_messagebox_signal.emit(
                "Recording Error",
                f"FFmpeg command failed during concatenation:\n{e}\nCould not create final video.",
                self.main_window,
            )
        except FileNotFoundError:
            print("[ERROR] FFmpeg not found. Ensure it's in your system PATH.")
            self.main_window.display_messagebox_signal.emit(
                "Recording Error", "FFmpeg not found.", self.main_window
            )
        except Exception as e:
            print(f"[ERROR] An unexpected error occurred during finalization: {e}")
            self.main_window.display_messagebox_signal.emit(
                "Recording Error",
                f"An unexpected error occurred:\n{e}",
                self.main_window,
            )

        finally:
            # 6. Cleanup
            self._cleanup_temp_dir()

            # 7. Reset state
            self.segments_to_process = []
            self.current_segment_index = -1
            self.temp_segment_files = []
            self.current_segment_end_frame = None
            self.triggered_by_job_manager = False

            # 8. Final timing
            self.end_time = time.perf_counter()
            processing_time_sec = self.end_time - self.start_time
            formatted_duration = self._format_duration(processing_time_sec) # Use the new helper

            if concatenation_successful:
                print(
                    f"Total segment processing and concatenation finished in {formatted_duration}"
                )
            else:
                print(
                    f"Segment processing/concatenation failed after {formatted_duration}."
                )

            # 9. Final cleanup and UI reset
            print(
                "Clearing GPU Cache and running garbage collection post-concatenation."
            )
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            except Exception as e:
                print(f"[WARN] Error clearing Torch cache: {e}")
            gc.collect()

            layout_actions.enable_all_parameters_and_control_widget(self.main_window)
            video_control_actions.reset_media_buttons(self.main_window)
            print("Multi-segment processing flow finished.")

            if self.main_window.control["OpenOutputToggle"]:
                try:
                    list_view_actions.open_output_media_folder(self.main_window)
                except Exception:
                    pass
                    
            # Emit signal to notify JobProcessor that processing has finished SUCCESSFULLY
            print("Emitting processing_stopped_signal (multi-segment success).")
            self.processing_stopped_signal.emit()

    def _cleanup_temp_dir(self):
        """Safely removes the temporary directory used for segments."""
        if self.segment_temp_dir and os.path.exists(self.segment_temp_dir):
            try:
                print(
                    f"Cleaning up temporary segment directory: {self.segment_temp_dir}"
                )
                shutil.rmtree(self.segment_temp_dir, ignore_errors=True)
            except Exception as e:
                print(
                    f"[WARN] Failed to delete temporary directory {self.segment_temp_dir}: {e}"
                )
        self.segment_temp_dir = None

    # --- Audio Methods ---

    def start_live_sound(self):
        """Starts ffplay subprocess to play audio synced to the current frame."""
        # Calculate seek time based on the *next* frame to be displayed
        seek_time = (self.next_frame_to_display) / self.media_capture.get(cv2.CAP_PROP_FPS)
        
        # Adjust audio speed if custom FPS is used
        fpsdiv = 1.0
        if (
            self.main_window.control["VideoPlaybackCustomFpsToggle"]
            and not self.recording
        ):
            fpsorig = self.media_capture.get(cv2.CAP_PROP_FPS)
            fpscust = self.main_window.control["VideoPlaybackCustomFpsSlider"]
            if fpsorig > 0 and fpscust > 0:
                fpsdiv = fpscust / fpsorig
        if fpsdiv < 0.5:
            fpsdiv = 0.5 # Don't allow less than 0.5x speed
  
        args = [
            "ffplay",
            "-vn", # No video
            "-nodisp",
            "-stats",
            "-loglevel",
            "quiet",
            "-sync",
            "audio",
            "-af",
            f"volume={self.main_window.control['LiveSoundVolumeDecimalSlider']}, atempo={fpsdiv}",
            "-i", # Specify the input...
            self.media_path,
            "-ss", # ... THEN specify the seek time for a precise seek
            str(seek_time),
        ]

        self.ffplay_sound_sp = subprocess.Popen(
            args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )

    def _start_synchronized_playback(self):
        """
        Starts the playback components (audio and video) in a synchronized manner.
        Called once the preroll buffer is filled.
        """
        # 1. Start audio (ffplay) *first*
        if self.main_window.liveSoundButton.isChecked() and not self.recording:
            print("Starting audio subprocess (ffplay)...")
            self.start_live_sound()
            
            # 2. Start video (metronome) AFTER a delay
            # This is to allow ffplay time to initialize.
            AUDIO_STARTUP_LATENCY_MS = self.main_window.control.get("LiveSoundDelaySlider")
            print(f"Waiting {AUDIO_STARTUP_LATENCY_MS}ms for audio to initialize...")
            
            # Use the function with the clarified name
            QTimer.singleShot(AUDIO_STARTUP_LATENCY_MS, self._start_video_metronome_after_audio_delay)
        
        else:
            # No audio, start video immediately
            print("No audio. Starting video metronome immediately.")
            self._start_metronome(self.fps, is_first_start=True)

    def _start_video_metronome_after_audio_delay(self):
        """
        Slot for QTimer.singleShot.
        Starts the video metronome *after* the audio initialization delay has passed.
        """
        if not self.processing: # Check in case the user stopped processing
            return
        print("Audio startup delay complete. Starting video metronome.")
        self._start_metronome(self.fps, is_first_start=True)

    def stop_live_sound(self):
        """Stops the ffplay audio subprocess."""
        if self.ffplay_sound_sp:
            parent_pid = self.ffplay_sound_sp.pid
            try:
                # Kill parent and any child processes
                try:
                    parent_proc = psutil.Process(parent_pid)
                    children = parent_proc.children(recursive=True)
                    for child in children:
                        try:
                            child.kill()
                        except psutil.NoSuchProcess:
                            pass
                except psutil.NoSuchProcess:
                    pass

                self.ffplay_sound_sp.terminate()
                try:
                    self.ffplay_sound_sp.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.ffplay_sound_sp.kill()
            except psutil.NoSuchProcess:
                pass
            except Exception as e:
                print(f"[WARN] Error stopping live sound: {e}")

            self.ffplay_sound_sp = None

    # --- Webcam Methods ---

    def process_webcam(self):
        """Starts the webcam stream using the unified metronome."""
        if self.processing:
            print("[WARN] Processing already active, cannot start webcam.")
            return
        if self.file_type != "webcam":
            print("process_webcam: Only applicable for webcam input.")
            return
        if not (self.media_capture and self.media_capture.isOpened()):
            print("Error: Unable to open webcam source.")
            video_control_actions.reset_media_buttons(self.main_window)
            return
            
        print("Starting webcam processing setup...")

        # 1. Set State Flags
        self.processing = True
        self.is_processing_segments = False
        self.recording = False
        
        # 2. Clear Containers
        self.frames_to_display.clear()
        self.webcam_frames_to_display.queue.clear()
        with self.frame_queue.mutex:
            self.frame_queue.queue.clear()
        
        # 3. Start Metronome ET Feeder
        fps = self.media_capture.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30
        self.fps = fps
        
        print(f"Webcam target FPS: {self.fps}")

        # Start the feeder thread
        print("Starting feeder thread (Mode: webcam)...")
        self.feeder_thread = threading.Thread(target=self._feeder_loop, daemon=True)
        self.feeder_thread.start()
        
        # Start the display metronome
        self._start_metronome(self.fps, is_first_start=True)
