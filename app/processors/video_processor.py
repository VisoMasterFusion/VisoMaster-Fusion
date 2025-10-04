import threading
import queue
from typing import TYPE_CHECKING, Dict, Tuple
import time
import subprocess
from pathlib import Path
import os
import gc
from functools import partial
import tempfile
import shutil
import uuid
from datetime import datetime # Added from default version for temp file naming
import cv2
import psutil
import numpy
import torch
import pyvirtualcam
import math
from PySide6.QtCore import QObject, QTimer, Signal, Slot, QEventLoop
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
    frame_processed_signal = Signal(int, QPixmap, numpy.ndarray)
    webcam_frame_processed_signal = Signal(QPixmap, numpy.ndarray)
    single_frame_processed_signal = Signal(int, QPixmap, numpy.ndarray)
    start_segment_timers_signal = Signal(int) # For multi-segment
    processing_started_signal = Signal() # Unified signal for any processing start

    def __init__(self, main_window: 'MainWindow', num_threads=2):
        super().__init__()
        self.main_window = main_window
        self.frame_queue = queue.Queue(maxsize=num_threads)
        self.media_capture: cv2.VideoCapture|None = None
        self.file_type = None
        self.fps = 0
        self.processing = False # General flag: True if playing OR recording (either style)
        self.current_frame_number = 0
        self.max_frame_number = 0
        self.media_path = None
        self.num_threads = num_threads
        self.threads: Dict[int, threading.Thread] = {}
        self.current_frame: numpy.ndarray = []
        self.virtcam: pyvirtualcam.Camera|None = None
        self.recording_sp: subprocess.Popen|None = None # Used by both recording styles
        
        self.ffplay_sound_sp = None

        # --- Flags and State for Recording Styles ---
        self.recording: bool = False # default style recording flag
        self.is_processing_segments: bool = False # Your multi-segment recording flag
        self.temp_file: str = '' # default style temporary video file (without audio)
        self.triggered_by_job_manager: bool = False # For multi-segment job integration

        # --- Multi-Segment Recording State ---
        self.segments_to_process: list[tuple[int, int]] = []
        self.current_segment_index: int = -1
        self.temp_segment_files: list[str] = []
        self.current_segment_end_frame: int | None = None
        self.segment_temp_dir: str | None = None
        # --- End Multi-Segment State ---

        # --- Timing ---
        self.start_time = 0.0 # Used by both for performance timing
        self.end_time = 0.0
        self.play_start_time = 0.0 # Used by default style for audio segmenting
        self.play_end_time = 0.0   # Used by default style for audio segmenting

        # --- Timers ---
        self.frame_read_timer = QTimer()
        # frame_read_timer.timeout connected dynamically
        self.frame_display_timer = QTimer()
        # frame_display_timer.timeout connected dynamically
        self.gpu_memory_update_timer = QTimer()
        self.gpu_memory_update_timer.timeout.connect(partial(common_widget_actions.update_gpu_memory_progressbar, main_window))

        # --- Frame Handling ---
        self.next_frame_to_display = 0
        self.frames_to_display: Dict[int, Tuple[QPixmap, numpy.ndarray]] = {}
        self.webcam_frames_to_display = queue.Queue()

        # --- Signal Connections ---
        self.frame_processed_signal.connect(self.store_frame_to_display)
        self.webcam_frame_processed_signal.connect(self.store_webcam_frame_to_display)
        self.single_frame_processed_signal.connect(self.display_current_frame)
        self.start_segment_timers_signal.connect(self._start_timers_from_signal) # For multi-segment

    @Slot(int, QPixmap, numpy.ndarray)
    def store_frame_to_display(self, frame_number, pixmap, frame):
        self.frames_to_display[frame_number] = (pixmap, frame)

    @Slot(QPixmap, numpy.ndarray)
    def store_webcam_frame_to_display(self, pixmap, frame):
        self.webcam_frames_to_display.put((pixmap, frame))

    @Slot(int, QPixmap, numpy.ndarray)
    def display_current_frame(self, frame_number, pixmap, frame):
        # This handles single frame updates (e.g., after seeking)
        if self.main_window.loading_new_media:
            graphics_view_actions.update_graphics_view(self.main_window, pixmap, frame_number, reset_fit=True)
            self.main_window.loading_new_media = False
        else:
            graphics_view_actions.update_graphics_view(self.main_window, pixmap, frame_number)
        self.current_frame = frame
        torch.cuda.empty_cache()
        common_widget_actions.update_gpu_memory_progressbar(self.main_window)

    def display_next_frame(self):
        # Handles displaying frames during playback or recording modes
        should_stop_playback = False
        should_finalize_default_recording = False

        if not self.processing: # General check first (covers cases where stop initiated elsewhere)
            # If processing was already false, ensure timers are stopped and exit.
            # This might happen if stop_processing was called between timer ticks.
            if self.frame_read_timer.isActive() or self.frame_display_timer.isActive():
                 self.stop_processing() # Ensure cleanup if somehow still active
            return

        # --- Segment Processing Logic ---
        if self.is_processing_segments:
            if self.current_segment_end_frame is not None and self.next_frame_to_display > self.current_segment_end_frame:
                print(f"Segment {self.current_segment_index + 1} end frame ({self.current_segment_end_frame}) reached.")
                self.stop_current_segment() # Segment logic handles its own stop/transition
                return # Don't proceed further in this tick

        # --- End of Media Check (Playback or default Recording) ---
        elif self.next_frame_to_display > self.max_frame_number:
            print("End of media reached.")
            if self.recording: # If default style recording was active, finalize it
                should_finalize_default_recording = True
            else: # Just normal playback ending
                should_stop_playback = True

        # --- Perform Stop/Finalize Actions ---
        if should_finalize_default_recording:
            self._finalize_default_style_recording()
            return # Finalization handles everything from here
        elif should_stop_playback:
            self.stop_processing() # Call the general stop/abort for playback
            return

        # --- Display Frame Logic ---
        if self.next_frame_to_display not in self.frames_to_display:
            # Frame not ready yet, wait for next timer tick
            return
        else:
            pixmap, frame = self.frames_to_display.pop(self.next_frame_to_display)
            self.current_frame = frame # Update current frame state

            # Send to Virtual Cam if enabled
            self.send_frame_to_virtualcam(frame)

            # Write to FFmpeg pipe if recording (either style)
            if self.is_processing_segments or self.recording:
                if self.recording_sp and self.recording_sp.stdin and not self.recording_sp.stdin.closed:
                    try:
                        self.recording_sp.stdin.write(frame.tobytes())
                    except OSError as e:
                        # Log appropriately based on mode
                        log_prefix = f"segment {self.current_segment_index + 1}" if self.is_processing_segments else "recording"
                        print(f"[WARN] Error writing frame {self.next_frame_to_display} to FFmpeg stdin during {log_prefix}: {e}")
                else:
                    # Log appropriately based on mode
                    log_prefix = f"segment {self.current_segment_index + 1}" if self.is_processing_segments else "recording"
                    print(f"[WARN] FFmpeg stdin not available for {log_prefix} when trying to write frame {self.next_frame_to_display}.")

            # Update UI (slider/time only during playback/default recording, not multi-segment)
            if not self.is_processing_segments:
                 video_control_actions.update_widget_values_from_markers(self.main_window, self.next_frame_to_display)

            graphics_view_actions.update_graphics_view(self.main_window, pixmap, self.next_frame_to_display)

            # Clean up thread entry
            if self.next_frame_to_display in self.threads:
                    self.threads.pop(self.next_frame_to_display)

            # Increment for next frame
            self.next_frame_to_display += 1

    def display_next_webcam_frame(self):
        # Handles webcam stream display (no recording logic here)
        if not self.processing:
            self.stop_processing() # Should already be stopped, but safe check
            return

        if self.webcam_frames_to_display.empty():
            return
        else:
            pixmap, frame = self.webcam_frames_to_display.get()
            self.current_frame = frame
            self.send_frame_to_virtualcam(frame)
            graphics_view_actions.update_graphics_view(self.main_window, pixmap, 0) # Frame number is irrelevant for webcam

    def send_frame_to_virtualcam(self, frame: numpy.ndarray):
        if self.main_window.control['SendVirtCamFramesEnableToggle'] and self.virtcam:
            height, width, _ = frame.shape
            if self.virtcam.height != height or self.virtcam.width != width:
                self.enable_virtualcam() # Re-enable with new dimensions

            # Need to check again if virtcam was successfully re-enabled
            if self.virtcam:
                try:
                    self.virtcam.send(frame)
                    self.virtcam.sleep_until_next_frame()
                except Exception as e:
                    print(f"[WARN] Failed sending frame to virtualcam: {e}")
                    # Optionally disable virtcam feature temporarily or permanently after errors
                    # self.disable_virtualcam()
                    # self.main_window.control['SendVirtCamFramesEnableToggle'] = False

    def set_number_of_threads(self, value):
        self.stop_processing() # Stop any active processing before changing threads
        self.main_window.models_processor.set_number_of_threads(value)
        self.num_threads = value
        self.frame_queue = queue.Queue(maxsize=self.num_threads)
        print(f"Max Threads set as {value} ")

    def process_video(self):
        """
        Start video processing. Can be playback OR default style recording.
        The self.recording flag should be set externally (e.g., by UI button press)
        BEFORE calling this method if default-style recording is desired.
        """
        if self.processing or self.is_processing_segments: # Check both flags
            print("Processing already in progress (play or segment). Ignoring start request.")
            return

        if self.file_type != 'video':
            print("process_video: Only applicable for video files.")
            return

        if not (self.media_capture and self.media_capture.isOpened()):
            print("Error: Unable to open the video source.")
            self.processing = False
            self.recording = False # Ensure flags are false
            self.is_processing_segments = False
            video_control_actions.reset_media_buttons(self.main_window)
            return

        mode = "recording (default-style)" if self.recording else "playback"
        print(f"Starting video {mode} processing setup...")

        self.processing = True # General flag ON
        # Ensure multi-segment flag is OFF for this mode
        self.is_processing_segments = False

        # Determine if this default-style recording was initiated by the Job Manager
        job_mgr_flag = getattr(self.main_window, 'job_manager_initiated_record', False)
        if self.recording and job_mgr_flag:
            self.triggered_by_job_manager = True
            print("Detected default-style recording initiated by Job Manager.")
        else:
            self.triggered_by_job_manager = False

        # Clear the flag so manual recordings won't be affected
        try:
            self.main_window.job_manager_initiated_record = False
        except Exception:
            pass

        if self.recording:
            # Disable UI elements during default style recording
            if not self.main_window.control['KeepControlsToggle']:
                layout_actions.disable_all_parameters_and_control_widget(self.main_window)
                
            # Create the ffmpeg subprocess for default style
            if not self.create_ffmpeg_subprocess_default_style():
                print("[ERROR] Failed to start FFmpeg for default-style recording.")
                self.stop_processing() # Abort the start
                return
        
        if self.main_window.liveSoundButton.isChecked():
            self.start_live_sound()

        self.start_time = time.perf_counter()
        self.frames_to_display.clear()
        self.threads.clear()

        # Calculate start time based on current slider position (for default audio merge)
        self.current_frame_number = self.main_window.videoSeekSlider.value()
        self.media_capture.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_number)
        self.play_start_time = float(self.current_frame_number / float(self.fps)) if self.fps > 0 else 0.0
        self.next_frame_to_display = self.current_frame_number

        # Read the very first frame to ensure self.current_frame is populated before ffmpeg might need it
        # (though default_style ffmpeg uses it before starting pipe, segment style needs it)
        ret, frame_bgr = misc_helpers.read_frame(self.media_capture, preview_mode=False)
        if ret:
             self.current_frame = numpy.ascontiguousarray(frame_bgr[..., ::-1]) # BGR to RGB
        else:
             print(f"[ERROR] Could not read first frame {self.current_frame_number} for {mode}.")
             self.stop_processing()
             return
        # IMPORTANT: Reset capture position back after the read
        self.media_capture.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_number)


        # --- Connect and Start Timers ---
        # Attempt to disconnect previous timer connections, ignoring runtime warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            try:
                self.frame_display_timer.timeout.disconnect(self.display_next_frame)
                self.frame_read_timer.timeout.disconnect(self.process_next_frame)
            except (TypeError, RuntimeError):
                pass
        self.frame_display_timer.timeout.connect(self.display_next_frame)
        self.frame_read_timer.timeout.connect(self.process_next_frame)

        # Determine FPS for timer interval
        if self.main_window.control['VideoPlaybackCustomFpsToggle'] and not self.recording: # Use custom FPS only for playback
            fps = self.main_window.control['VideoPlaybackCustomFpsSlider']
        else:
            fps = self.media_capture.get(cv2.CAP_PROP_FPS)
            if fps <= 0: # Fallback FPS if capture doesn't report it
                 print("[WARN] Video source reported invalid FPS, using fallback 30.")
                 fps = 30
            self.fps = fps # Store the determined FPS

        interval = 1000 / fps if fps > 0 else 33 # Default to ~30fps if calculation fails
        # Apply default 80% interval logic for potentially smoother playback/recording frame feeding
        interval = int(interval * 0.8)
        if interval <= 0: interval = 1 # Ensure interval is positive

        print(f"Starting {mode} timers with interval {interval} ms (Target FPS: {fps:.2f}).")
        self.frame_read_timer.start(interval)
        self.frame_display_timer.start(interval) # Start display timer with same interval
        self.gpu_memory_update_timer.start(5000)
        self.processing_started_signal.emit()  # EMIT UNIFIED SIGNAL HERE

    def process_next_frame(self):
        """Read the next frame and enqueue for processing (used by playback and default recording)."""
        if not self.processing: # Check if processing stopped externally
            self.frame_read_timer.stop()
            return

        # --- Segment logic IS NOT handled here, it's in its own flow ---
        # if self.is_processing_segments: ... -> This path should not be taken when this timer connection is active

        # --- End of video check (for playback or default recording) ---
        if self.current_frame_number > self.max_frame_number:
            print("All frames read. Stopping frame reading timer.")
            self.frame_read_timer.stop()
            # The display_next_frame logic will handle the final stop/finalization
            return

        # --- Queue Check ---
        if self.frame_queue.qsize() >= self.num_threads:
            # Queue is full, skip reading this cycle to allow workers to catch up
            return

        # --- Read Frame ---
        # Use preview_mode=True only for pure playback, not recording
        ret, frame_bgr = misc_helpers.read_frame(self.media_capture, preview_mode = not self.recording)

        if ret:
            frame_rgb = frame_bgr[..., ::-1] # Convert BGR to RGB
            self.frame_queue.put(self.current_frame_number)
            self.start_frame_worker(self.current_frame_number, frame_rgb) # is_single_frame defaults to False
            self.current_frame_number += 1
        else:
            # Frame read failed!
            failed_frame_num = self.current_frame_number
            print(f"[ERROR] Cannot read frame {failed_frame_num}! Video source may be corrupted or end unexpectedly.")
            self.frame_read_timer.stop() # Stop trying to read more frames immediately

            # If default style recording was active, attempt to finalize with frames read so far
            if self.recording:
                print(f"Attempting to finalize default-style recording due to read error at frame {failed_frame_num}.")
                # Set next_frame_to_display to the failed frame to trigger finalization in display loop?
                # Or call finalize directly? Calling directly is safer.
                self.next_frame_to_display = failed_frame_num # Ensure display loop knows where we stopped
                self._finalize_default_style_recording()
            else:
                # For normal playback, just stop everything via the general abort
                self.stop_processing()
            # Emit message box signal? (As in default version)
            self.main_window.display_messagebox_signal.emit('Error Reading Frame', f'Error Reading Frame {failed_frame_num}.\n Stopped Processing!', self.main_window)


    def start_frame_worker(self, frame_number, frame, is_single_frame=False):
        """Start a FrameWorker to process the given frame."""
        # Pass the correct recording flag based on the current mode
        worker = FrameWorker(frame, self.main_window, frame_number, self.frame_queue, is_single_frame)
        self.threads[frame_number] = worker
        if is_single_frame:
            worker.run() # Process synchronously for single frames
        else:
            worker.start() # Process asynchronously in a thread

    def process_current_frame(self):
        """Process the single, currently selected frame (e.g., after seek or for image)."""
        # Stop any ongoing playback/recording first
        if self.processing or self.is_processing_segments:
             print("[INFO] Stopping active processing to process single frame.")
             if not self.stop_processing(): # Try to stop gracefully
                  print("[WARN] Could not stop active processing cleanly, proceeding with single frame anyway.")

        # Set frame number for processing
        if self.file_type == 'video':
             self.current_frame_number = self.main_window.videoSeekSlider.value()
        elif self.file_type == 'image' or self.file_type == 'webcam':
             self.current_frame_number = 0 # Simple index for non-video

        self.next_frame_to_display = self.current_frame_number # Target this frame for display

        frame_to_process = None
        read_successful = False

        # --- Read the frame based on file type ---
        if self.file_type == 'video' and self.media_capture:
            self.media_capture.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_number)
            ret, frame_bgr = misc_helpers.read_frame(self.media_capture, preview_mode=False)
            if ret:
                frame_to_process = frame_bgr[..., ::-1] # BGR to RGB
                read_successful = True
                # Set capture back in case user immediately hits play (might be redundant)
                self.media_capture.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_number)
            else:
                 print(f"[ERROR] Cannot read frame {self.current_frame_number} for single processing!")
                 self.main_window.last_seek_read_failed = True # Use existing flag if available
                 self.main_window.display_messagebox_signal.emit('Error Reading Frame', f'Error Reading Frame {self.current_frame_number}.', self.main_window)

        elif self.file_type == 'image':
            frame_bgr = misc_helpers.read_image_file(self.media_path)
            if frame_bgr is not None:
                frame_to_process = frame_bgr[..., ::-1] # BGR to RGB
                read_successful = True
            else:
                print("[ERROR] Unable to read image file for processing.")

        elif self.file_type == 'webcam' and self.media_capture:
            ret, frame_bgr = misc_helpers.read_frame(self.media_capture, preview_mode=False)
            if ret:
                frame_to_process = frame_bgr[..., ::-1] # BGR to RGB
                read_successful = True
            else:
                print("[ERROR] Unable to read Webcam frame for processing!")

        # --- Process if read was successful ---
        if read_successful and frame_to_process is not None:
             # Ensure queue is empty and ready for single frame
             with self.frame_queue.mutex: self.frame_queue.queue.clear()
             self.threads.clear() # Clear any old threads

             self.frame_queue.put(self.current_frame_number)
             # Start worker for single frame, synchronously
             self.start_frame_worker(self.current_frame_number, frame_to_process, is_single_frame=True)
             # Note: Worker now calls display_current_frame via signal

        # No need to join threads here as single frame worker runs synchronously


    def process_next_webcam_frame(self):
         """ Read and enqueue next webcam frame. """
         if not self.processing: # Check if processing stopped
              self.frame_read_timer.stop()
              return

         if self.frame_queue.qsize() >= self.num_threads:
              # Queue full, wait
              return

         if self.file_type == 'webcam' and self.media_capture:
              ret, frame_bgr = misc_helpers.read_frame(self.media_capture, preview_mode=False)
              if ret:
                   frame_rgb = frame_bgr[..., ::-1] # BGR to RGB
                   # Use a placeholder frame number (0) for webcam frames in queue/workers
                   self.frame_queue.put(0)
                   # Start worker asynchronously
                   self.start_frame_worker(0, frame_rgb, is_single_frame=False)
              else:
                   print("[WARN] Failed to read webcam frame during stream.")
                   # Optionally stop processing or just log and continue
                   # self.stop_processing()


    def stop_processing(self):
        """
        General Stop / Abort Function.
        Stops timers, threads, ffmpeg (if running), cleans up temporary files/dirs
        associated with the mode that was active (default recording OR segment recording).
        Does NOT finalize recordings - that happens in dedicated finalize methods.
        """
        if not self.processing and not self.is_processing_segments:
            # print("No processing active to stop.")
            video_control_actions.reset_media_buttons(self.main_window)
            return False # Nothing was stopped

        print("stop_processing called: Aborting active processing...")
        was_processing_segments = self.is_processing_segments
        was_recording_default_style = self.recording # Capture state before reset

        # Reset flags FIRST
        self.processing = False
        self.is_processing_segments = False
        self.recording = False # Reset default style recording flag too
        self.triggered_by_job_manager = False

        # Stop timers immediately
        self.frame_read_timer.stop()
        self.frame_display_timer.stop()
        self.gpu_memory_update_timer.stop()
        self.stop_live_sound()

        # Disconnect timer signals to prevent future triggers
        # Need to handle potential TypeErrors if never connected
        try: self.frame_read_timer.timeout.disconnect()
        except (TypeError, RuntimeError): pass
        try: self.frame_display_timer.timeout.disconnect()
        except (TypeError, RuntimeError): pass


        # Wait for worker threads to finish processing queued items
        print("Waiting for worker threads to complete...")
        self.join_and_clear_threads()
        print("Worker threads joined.")

        # Clear display queues
        self.frames_to_display.clear()
        self.webcam_frames_to_display.queue.clear()
        # Clear the processing queue
        with self.frame_queue.mutex:
            self.frame_queue.queue.clear()

        # Stop and cleanup ffmpeg subprocess if it was running
        if self.recording_sp:
            print("Closing and waiting for active FFmpeg subprocess...")
            if self.recording_sp.stdin and not self.recording_sp.stdin.closed:
                try:
                    self.recording_sp.stdin.close()
                except OSError as e:
                     print(f"[WARN] Error closing ffmpeg stdin during abort: {e}")
            # Wait for process to terminate, with a timeout?
            try:
                self.recording_sp.wait(timeout=5) # Wait max 5 seconds
                print("FFmpeg subprocess terminated.")
            except subprocess.TimeoutExpired:
                print("[WARN] FFmpeg subprocess did not terminate gracefully, killing.")
                self.recording_sp.kill()
                self.recording_sp.wait() # Wait again after kill
            except Exception as e:
                 print(f"[ERROR] Error waiting for FFmpeg subprocess: {e}")
            self.recording_sp = None


        # Cleanup temporary files/dirs based on the mode that was aborted
        if was_processing_segments:
            print("Cleaning up segment temporary directory due to abort.")
            self._cleanup_temp_dir() # Cleanup segment dir
        elif was_recording_default_style:
             print("Cleaning up default-style temporary file due to abort.")
             if self.temp_file and os.path.exists(self.temp_file):
                 try:
                     os.remove(self.temp_file)
                     print(f"Removed temporary file: {self.temp_file}")
                 except OSError as e:
                     print(f"[WARN] Could not remove temp file {self.temp_file} during abort: {e}")
             self.temp_file = '' # Reset temp file name

        # Reset segment state variables (always safe to do on abort)
        self.segments_to_process = []
        self.current_segment_index = -1
        self.temp_segment_files = []
        self.current_segment_end_frame = None

        # Reset capture position to slider value if applicable
        if self.file_type == 'video' and self.media_capture:
            try:
                current_slider_pos = self.main_window.videoSeekSlider.value()
                self.current_frame_number = current_slider_pos
                self.next_frame_to_display = current_slider_pos
                # Ensure capture is still open before setting position
                if self.media_capture.isOpened():
                    self.media_capture.set(cv2.CAP_PROP_POS_FRAMES, current_slider_pos)
            except Exception as e:
                print(f"[WARN] Could not reset video capture position: {e}")

        # Re-enable UI elements if they were disabled by either recording mode
        if was_processing_segments or was_recording_default_style:
             layout_actions.enable_all_parameters_and_control_widget(self.main_window)

        # Final cleanup (cache, gc, buttons)
        print("Clearing GPU Cache and running garbage collection.")
        try:
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        except ImportError: pass
        except Exception as e: # Catch other potential torch errors
             print(f"[WARN] Error clearing Torch cache: {e}")
        gc.collect()

        video_control_actions.reset_media_buttons(self.main_window)
        # Disable virtual camera if active when stopping without ongoing processing
        try: self.disable_virtualcam()
        except Exception: pass
        print("Processing aborted and cleaned up.")

        # Determine the end time based on the frame counter
        # The last frame successfully *processed* (and intended for display) was self.next_frame_to_display - 1
        # The range for audio extraction should go up to the *start* of the frame *after* the last processed one.
        end_frame_for_calc = min(self.next_frame_to_display, self.max_frame_number + 1)
        self.play_end_time = float(end_frame_for_calc / float(self.fps)) if self.fps > 0 else 0.0
        print(f"Calculated default-style recording end time: {self.play_end_time:.3f}s (based on frame {end_frame_for_calc})")
        
        # --- Final Timing and Logging (default style) ---
        self.end_time = time.perf_counter()
        processing_time = self.end_time - self.start_time
        print(f"\nProcessing completed in {processing_time:.2f} seconds")
        try:
            duration = self.play_end_time - self.play_start_time
            if duration > 0 or processing_time > 0:
                 processed_frames = duration * self.fps
                 avg_fps = processed_frames / processing_time
                 print(f'Average Processing FPS: {avg_fps:.2f}\n')
            else:
                 print("Could not calculate average FPS (duration or processing time is zero).\n")
        except Exception as e:
            print(f"[WARN] Could not calculate average FPS: {e}\n")

        return True # Processing was stopped

    def join_and_clear_threads(self):
        # print("Joining worker threads...")
        active_threads = list(self.threads.values()) # Copy list as dict may change
        for thread in active_threads:
            try:
                if thread.is_alive():
                    thread.join(timeout=1.0) # Add a small timeout
                    if thread.is_alive():
                         print(f"[WARN] Thread {thread.name} did not join gracefully.")
            except Exception as e:
                 print(f"[WARN] Error joining thread {thread.name}: {e}")
        # print('Clearing thread dictionary.')
        self.threads.clear()

    # --- default Style FFmpeg and Finalization ---

    def create_ffmpeg_subprocess_default_style(self):
        control = self.main_window.control.copy()
        """Creates the FFmpeg subprocess for simple (default') style recording."""
        if not isinstance(self.current_frame, numpy.ndarray) or self.current_frame.size == 0:
             print("[ERROR] create_ffmpeg_subprocess_default_style: Current frame invalid. Cannot get dimensions.")
             return False
        if self.fps <= 0:
             print("[ERROR] create_ffmpeg_subprocess_default_style: Invalid FPS.")
             return False

        frame_height, frame_width, _ = self.current_frame.shape
        # Added option to resize video frame to 1920*1080
        frame_height_down = frame_height
        frame_width_down = frame_width
        #frame_height_down_mult = frame_height / 1080
        frame_width_down_mult = frame_width / 1920
          
        if control['FrameEnhancerDownToggle']:
            if (frame_width != 1920 or frame_height != 1080):
                frame_height_down = math.ceil(frame_height / frame_width_down_mult)
                #frame_width_down = math.ceil(frame_width / frame_width_down_mult)
                frame_width_down = 1920
            else:
                print("Already 1920*1080")
                
        date_and_time = datetime.now().strftime(r'%Y_%m_%d_%H_%M_%S')
        # Use a temporary directory within the project root
        try:
            # Define the base temp directory within the project
            base_temp_dir = os.path.join(os.getcwd(), "temp_files", "default")
            os.makedirs(base_temp_dir, exist_ok=True) # Ensure base dir exists
            self.temp_file = os.path.join(base_temp_dir, f'temp_output_{date_and_time}.mp4')
            print(f"Default temp file will be created at: {self.temp_file}")
        except Exception as e:
            print(f"[ERROR] Failed to create temporary directory/file path: {e}")
            # Fallback to local dir relative to cwd if creation fails?
            self.temp_file = f'temp_output_{date_and_time}.mp4'
            print(f"[WARN] Falling back to local directory for temp file: {self.temp_file}")

        print(f"Creating FFmpeg (default-style): Video Dim={frame_width}x{frame_height}, FPS={self.fps}, Temp Output='{self.temp_file}'")

        if Path(self.temp_file).is_file():
            try:
                os.remove(self.temp_file)
            except OSError as e:
                print(f"[WARN] Could not remove existing temp file {self.temp_file}: {e}")

        # Change color options to keep original video colors
        output_pix_fmt = 'yuvj420p' # Standard 8-bit SDR
        output_color_trc = 'bt2020-10'
        output_colorspace = 'rgb'
        output_color_primaries = 'bt2020'
        
        args = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{frame_width}x{frame_height}",
            "-r", str(self.fps),
            "-i", "pipe:0", # Read from stdin (pipe:0 is more explicit)
        ]
        # Merged settings from experimental and vr180 patches
        if control['HDREncodeToggle']:
            if control['FrameEnhancerDownToggle']:
                args.extend([
                    # Video Codec: Use NVIDIA HEVC encoder
                    "-c:v", "libx265",
                    "-vf", f"scale={frame_width_down}x{frame_height_down}:flags=lanczos+accurate_rnd+full_chroma_int",
                    "-profile:v", "main10",
                    "-preset", "slow",
                    "-pix_fmt", "yuv420p10le",
                    "-x265-params", f"crf=18:vbv-bufsize=10000:vbv-maxrate=8000:selective-sao=0:no-sao=1:strong-intra-smoothing=0:rect=0:aq-mode=1:hdr-opt=1:repeat-headers=1:colorprim=bt2020:range=limited:transfer=smpte2084:colormatrix=bt2020nc:range=limited:master-display='G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,1)':max-cll=1000,400",
                    # Output File
                    self.temp_file
                ])
            else:
                args.extend([
                    # Video Codec: Use NVIDIA HEVC encoder
                    "-c:v", "libx265",
                    "-profile:v", "main10",
                    "-preset", "slow",
                    "-pix_fmt", "yuv420p10le",
                    "-x265-params", "crf=18:vbv-bufsize=10000:vbv-maxrate=8000:selective-sao=0:no-sao=1:strong-intra-smoothing=0:rect=0:aq-mode=1:hdr-opt=1:repeat-headers=1:colorprim=bt2020:range=limited:transfer=smpte2084:colormatrix=bt2020nc:range=limited:master-display='G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,1)':max-cll=1000,400",
                    #Output File
                    self.temp_file
                ])
        else:
            if control['FrameEnhancerDownToggle']:
                args.extend([
                    "-c:v", "hevc_nvenc",
                    "-preset", "p5",
                    "-cq", "18", # Higher quality setting from experimental patch
                    "-vf", f"scale={frame_width_down}x{frame_height_down}:flags=lanczos+accurate_rnd+full_chroma_int",
                    "-pix_fmt", output_pix_fmt,
                    "-colorspace", output_colorspace,
                    "-color_primaries", output_color_primaries,
                    "-color_trc", output_color_trc,
                    "-tag:v", "hvc1",
                    self.temp_file
                ])
            else:
                
                args.extend([
                    "-c:v", "hevc_nvenc",
                    "-preset", "p5",
                    "-cq", "18", # Higher quality setting from experimental patch
                    "-pix_fmt", output_pix_fmt,
                    "-colorspace", output_colorspace,
                    "-color_primaries", output_color_primaries,
                    "-color_trc", output_color_trc,
                    "-tag:v", "hvc1",
                    self.temp_file
                ])
        try:
            # bufsize=-1 uses system default, often buffered which might be better for video
            self.recording_sp = subprocess.Popen(args, stdin=subprocess.PIPE, bufsize=-1)
            return True
        except FileNotFoundError:
            print("[ERROR] FFmpeg command not found. Ensure FFmpeg is installed and in system PATH.")
            self.main_window.display_messagebox_signal.emit('FFmpeg Error', 'FFmpeg command not found.', self.main_window)
            return False
        except Exception as e:
            print(f"[ERROR] Failed to start FFmpeg subprocess (default-style): {e}")
            self.main_window.display_messagebox_signal.emit('FFmpeg Error', f'Failed to start FFmpeg:\n{e}', self.main_window)
            return False

    def _finalize_default_style_recording(self):
        """Finalizes a successful default-style recording (adds audio, cleans up)."""
        print("Finalizing default-style recording...")
        self.processing = False # Stop processing flag

        # Stop timers (might already be stopped, but safe)
        self.frame_read_timer.stop()
        self.frame_display_timer.stop()
        self.gpu_memory_update_timer.stop()

        # Disconnect timer signals
        try: self.frame_read_timer.timeout.disconnect()
        except (TypeError, RuntimeError): pass
        try: self.frame_display_timer.timeout.disconnect()
        except (TypeError, RuntimeError): pass

        # Ensure worker threads finish up (display might have last frame)
        print("Waiting for final worker threads...")
        self.join_and_clear_threads()
        self.frames_to_display.clear() # Clear display queue

        # Finalize ffmpeg subprocess (close pipe, wait)
        if self.recording_sp:
            if self.recording_sp.stdin and not self.recording_sp.stdin.closed:
                try:
                    print("Closing FFmpeg stdin...")
                    self.recording_sp.stdin.close()
                except OSError as e:
                    print(f"[WARN] Error closing FFmpeg stdin during finalization: {e}")
            print("Waiting for FFmpeg subprocess to finish writing...")
            try:
                 self.recording_sp.wait(timeout=10) # Wait up to 10 seconds
                 print("FFmpeg subprocess finished.")
            except subprocess.TimeoutExpired:
                 print("[WARN] FFmpeg subprocess timed out during finalization, killing.")
                 self.recording_sp.kill()
                 self.recording_sp.wait()
            except Exception as e:
                 print(f"[ERROR] Error waiting for FFmpeg subprocess during finalization: {e}")
            self.recording_sp = None
        else:
            print("[WARN] No recording subprocess found during default-style finalization.")
            # If no subprocess, likely means temp file wasn't created or error occurred early.

        # Determine the end time based on the frame counter
        # The last frame successfully *processed* (and intended for display) was self.next_frame_to_display - 1
        # The range for audio extraction should go up to the *start* of the frame *after* the last processed one.
        end_frame_for_calc = min(self.next_frame_to_display, self.max_frame_number + 1)
        self.play_end_time = float(end_frame_for_calc / float(self.fps)) if self.fps > 0 else 0.0
        print(f"Calculated default-style recording end time: {self.play_end_time:.3f}s (based on frame {end_frame_for_calc})")


        # --- Audio Merging (default logic) ---
        if self.temp_file and os.path.exists(self.temp_file) and os.path.getsize(self.temp_file) > 0:
            # --- Determine Final Output Path (incorporating job manager settings) ---
            was_triggered_by_job = getattr(self, 'triggered_by_job_manager', False)
            job_name = getattr(self.main_window, 'current_job_name', None) if was_triggered_by_job else None
            use_job_name = getattr(self.main_window, 'use_job_name_for_output', False) if was_triggered_by_job else False
            output_file_name = getattr(self.main_window, 'output_file_name', None) if was_triggered_by_job else None
            # DEBUG: inspect filename determination flags
            #print(f"[DEBUG] _finalize_default_style_recording: triggered_by_job={was_triggered_by_job}, job_name={job_name}, use_job_name_for_output={use_job_name}, output_file_name={output_file_name}")

            final_file_path = misc_helpers.get_output_file_path(
                self.media_path,
                self.main_window.control['OutputMediaFolder'],
                job_name=job_name,
                use_job_name_for_output=use_job_name,
                output_file_name=output_file_name
            )

            # Ensure output directory exists
            output_dir = os.path.dirname(final_file_path)
            if not os.path.exists(output_dir):
                try: os.makedirs(output_dir); print(f"Created output directory: {output_dir}")
                except OSError as e:
                     print(f"[ERROR] Failed to create output directory {output_dir}: {e}")
                     self.main_window.display_messagebox_signal.emit('File Error', f'Could not create output directory:\n{output_dir}\n\n{e}', self.main_window)
                     # Cleanup temp file even if output dir fails
                     try: os.remove(self.temp_file)
                     except OSError: pass
                     self.temp_file = ''
                     # Reset UI and return
                     layout_actions.enable_all_parameters_and_control_widget(self.main_window)
                     video_control_actions.reset_media_buttons(self.main_window)
                     self.recording = False
                     return

            if Path(final_file_path).is_file():
                print(f"Removing existing final file: {final_file_path}")
                try:
                    os.remove(final_file_path)
                except OSError as e:
                    print(f"[WARN] Failed to remove existing final file {final_file_path}: {e}") # Log but attempt merge anyway

            print("Adding audio (default-style merge)...")
            # Arguments exactly like default stop_processing merge step
            args = ["ffmpeg",
                    '-hide_banner',
                    '-loglevel', 'error',
                    "-i", self.temp_file, # Input 0: Temp video
                     # Input 1: Original audio, segmented using -ss/-to
                    "-ss", str(self.play_start_time),
                    "-to", str(self.play_end_time),
                    "-i", self.media_path,
                    "-c:v", "copy",       # Copy both video (already encoded) and audio
                    "-map", "0:v:0",    # Map video from input 0
                    "-map", "1:a:0?",   # Map audio from input 1 (optional)
                    "-shortest",        # Finish when shortest input ends (should be audio segment)
                    "-af", "aresample=async=1000", #Audio resample for sync
                    final_file_path]
            try:
                subprocess.run(args, check=True) # Use check=True to raise error on failure
                print(f"--- Successfully created final video (default-style): {final_file_path} ---")
            except subprocess.CalledProcessError as e:
                print(f"[ERROR] FFmpeg command failed during default-style audio merge: {e}")
                print(f"FFmpeg arguments: {' '.join(args)}") # Log the command
                self.main_window.display_messagebox_signal.emit('Recording Error', f'FFmpeg command failed during audio merge:\n{e}\nCheck console for command.', self.main_window)
            except FileNotFoundError:
                 print("[ERROR] FFmpeg not found. Cannot merge audio.")
                 self.main_window.display_messagebox_signal.emit('Recording Error', 'FFmpeg not found.', self.main_window)
            finally:
                 # Clean up temp file regardless of audio merge success
                 print(f"Removing temporary file: {self.temp_file}")
                 try:
                     os.remove(self.temp_file)
                 except OSError as e:
                     print(f"[WARN] Failed to remove temp file {self.temp_file}: {e}")
                 self.temp_file = ''
        else:
            if not self.temp_file:
                 print("[WARN] No temporary file name recorded. Cannot merge audio.")
            elif not os.path.exists(self.temp_file):
                 print(f"[WARN] Temporary video file missing: {self.temp_file}. Cannot merge audio.")
            else:
                 print(f"[WARN] Temporary video file empty: {self.temp_file}. Cannot merge audio.")
                 # Clean up the empty file
                 try: os.remove(self.temp_file)
                 except OSError: pass
                 self.temp_file = ''


        # --- Final Timing and Logging (default style) ---
        self.end_time = time.perf_counter()
        processing_time = self.end_time - self.start_time
        print(f"\nProcessing completed in {processing_time:.2f} seconds")
        try:
            duration = self.play_end_time - self.play_start_time
            if duration > 0 or processing_time > 0:
                 processed_frames = duration * self.fps
                 avg_fps = processed_frames / processing_time
                 print(f'Average Processing FPS: {avg_fps:.2f}\n')
            else:
                 print("Could not calculate average FPS (duration or processing time is zero).\n")
        except Exception as e:
            print(f"[WARN] Could not calculate average FPS: {e}\n")


        # --- Reset State and UI ---
        self.recording = False # Ensure recording flag is off
        
        if self.main_window.control['AutoSaveWorkspaceToggle']:
            json_file_path = misc_helpers.get_output_file_path(self.media_path, self.main_window.control['OutputMediaFolder'])      
            json_file_path += ".json"
            save_load_actions.save_current_workspace(self.main_window, json_file_path)
            
        layout_actions.enable_all_parameters_and_control_widget(self.main_window)
        video_control_actions.reset_media_buttons(self.main_window)

        # Final cleanup
        print("Clearing GPU Cache and running garbage collection post-recording.")
        try:
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        except ImportError: pass
        except Exception as e: print(f"[WARN] Error clearing Torch cache: {e}")
        gc.collect()

        video_control_actions.reset_media_buttons(self.main_window)
        # Ensure virtual camera is disabled after abort
        try: self.disable_virtualcam()
        except Exception: pass
        print("default-style recording finalized.")
        
        if self.main_window.control['OpenOutputToggle']:
            try: list_view_actions.open_output_media_folder(self.main_window)
            except Exception: pass

    # --- Virtual Camera Methods ---

    def enable_virtualcam(self, backend=False):
        if not self.media_capture and not isinstance(self.current_frame, numpy.ndarray):
             print("[WARN] Cannot enable virtual camera without media loaded or a current frame.")
             return

        frame_height, frame_width = 0, 0
        current_fps = self.fps if self.fps > 0 else 30 # Use stored FPS or default

        # Try getting dimensions from current_frame first
        if isinstance(self.current_frame, numpy.ndarray) and self.current_frame.ndim == 3:
            frame_height, frame_width, _ = self.current_frame.shape
        # Fallback to media_capture if current_frame is invalid
        elif self.media_capture and self.media_capture.isOpened():
            frame_height = int(self.media_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_width = int(self.media_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            # Use capture FPS if self.fps wasn't set
            if current_fps == 30: current_fps = self.media_capture.get(cv2.CAP_PROP_FPS) if self.media_capture.get(cv2.CAP_PROP_FPS) > 0 else 30


        if frame_width <= 0 or frame_height <= 0:
             print(f"[ERROR] Cannot enable virtual camera: Invalid dimensions ({frame_width}x{frame_height}).")
             return

        self.disable_virtualcam() # Close existing cam first
        try:
            backend_to_use = backend or self.main_window.control['VirtCamBackendSelection']
            print(f"Enabling virtual camera: {frame_width}x{frame_height} @ {int(current_fps)}fps, Backend: {backend_to_use}, Format: BGR")
            # Using BGR format as per default version and send_frame logic
            self.virtcam = pyvirtualcam.Camera(width=frame_width, height=frame_height, fps=int(current_fps), backend=backend_to_use, fmt=pyvirtualcam.PixelFormat.BGR)
            print(f"Virtual camera '{self.virtcam.device}' started.")
        except Exception as e:
            print(f"[ERROR] Failed to enable virtual camera: {e}")
            self.virtcam = None # Ensure virtcam is None on failure
            # Optionally notify user via messagebox
            #self.main_window.display_messagebox_signal.emit('Virtual Camera Error', f'Failed to start virtual camera:\n{e}', self.main_window) 
            # Deactivated messagebox option on error else it stops job manager processes

    def disable_virtualcam(self):
        if self.virtcam:
            print(f"Disabling virtual camera '{self.virtcam.device}'.")
            try:
                self.virtcam.close()
            except Exception as e:
                 print(f"[WARN] Error closing virtual camera: {e}")
            self.virtcam = None


    # --- Multi-Segment Recording Methods (Your Original Logic - Minor Adaptations) ---

    def create_ffmpeg_subprocess_segment(self, output_filename: str):
        control = self.main_window.control.copy()
        """Creates the FFmpeg subprocess for a single segment during multi-segment recording."""
        # This is your original ffmpeg logic, slightly adapted checks
        if not isinstance(self.current_frame, numpy.ndarray) or self.current_frame.size == 0:
            print("[ERROR] create_ffmpeg_subprocess_segment: Invalid current_frame.")
            return False
        if not self.media_path or not Path(self.media_path).is_file():
            print("[ERROR] create_ffmpeg_subprocess_segment: Original media path invalid.")
            return False
        if self.fps <= 0:
            print("[ERROR] create_ffmpeg_subprocess_segment: Invalid FPS.")
            return False
        if self.current_segment_index < 0 or self.current_segment_index >= len(self.segments_to_process):
            print(f"[ERROR] create_ffmpeg_subprocess_segment: Invalid segment index {self.current_segment_index}.")
            return False

        start_frame, end_frame = self.segments_to_process[self.current_segment_index]
        start_time_sec = start_frame / self.fps
        # Use end_frame + 1 for duration calc if using -t, or end_frame / fps for -to
        # Let's stick to -ss and -to for consistency with default style where applicable
        # end_time_sec = (end_frame + 1) / self.fps # End time is exclusive in -to? No, ffmpeg docs say -to is inclusive. Use end_frame.
        end_time_sec = end_frame / self.fps # Let's try end_frame directly for -to

        frame_height, frame_width, _ = self.current_frame.shape
        # Fix for frame enhancer activated while doing segments
        if control['FrameEnhancerEnableToggle']:
            if control['FrameEnhancerTypeSelection'] in ('RealEsrgan-x2-Plus', 'BSRGan-x2'):
                frame_height = frame_height * 2
                frame_width = frame_width * 2
            elif control['FrameEnhancerTypeSelection'] in ('RealEsrgan-x4-Plus', 'BSRGan-x4', 'UltraSharp-x4', 'UltraMix-x4', 'RealEsr-General-x4v3'):
                frame_height = frame_height * 4
                frame_width = frame_width * 4
        # Added option to resize video frame to 1920*1080
        frame_height_down = frame_height
        frame_width_down = frame_width
        #frame_height_down_mult = frame_height / 1080
        frame_width_down_mult = frame_width / 1920
          
        if control['FrameEnhancerDownToggle']:
            if (frame_width != 1920 or frame_height != 1080):
                frame_height_down = math.ceil(frame_height / frame_width_down_mult)
                #frame_width_down = math.ceil(frame_width / frame_width_down_mult)
                frame_width_down = 1920
            else:
                print("Already 1920*1080")
                
        segment_num = self.current_segment_index + 1
        print(f"Creating FFmpeg (Segment {segment_num}): Video Dim={frame_width}x{frame_height}, FPS={self.fps}, Output='{output_filename}'")
        print(f"  Audio Segment: Start={start_time_sec:.3f}s, End={end_time_sec:.3f}s (Frames {start_frame}-{end_frame})")

        if Path(output_filename).is_file():
            try:
                os.remove(output_filename)
            except OSError as e:
                print(f"[WARN] Could not remove existing segment file {output_filename}: {e}")

        output_pix_fmt = 'yuvj420p' # Standard 8-bit SDR
        output_color_trc = 'bt2020-10'
        output_colorspace = 'rgb'
        output_color_primaries = 'bt2020'
        
        args = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",
            # Input 0: Processed Video from Pipe (RGB)
            "-f", "rawvideo",
            "-pix_fmt", "bgr24", # Sending RGB frames
            "-s", f"{frame_width}x{frame_height}",
            "-r", str(self.fps),
            "-i", "pipe:0", # Input 0 from pipe
            # Input 1: Original Audio from File (Segmented)
            "-ss", str(start_time_sec),
            "-to", str(end_time_sec),
            "-i", self.media_path, # Input 1 is original file
            # Mapping
            "-map", "0:v:0", # Video from pipe
            "-map", "1:a:0?", # Audio from file (optional)
            # Audio Codec
            "-c:a", "copy", # Copy original audio stream segment
            # Options
            "-shortest", # Stop when shortest input (audio segment) ends
        ]
        # Video Codec & Options (Pad and format for compatibility)
        if control['HDREncodeToggle']:
            if control['FrameEnhancerDownToggle']:
                args.extend([
                    # Video Codec: Use NVIDIA HEVC encoder
                    "-c:v", "libx265",
                    "-vf", f"scale={frame_width_down}x{frame_height_down}:flags=lanczos+accurate_rnd+full_chroma_int",
                    "-profile:v", "main10",
                    "-preset", "slow",
                    "-pix_fmt", "yuv420p10le",
                    "-x265-params", f"crf=18:vbv-bufsize=10000:vbv-maxrate=8000:selective-sao=0:no-sao=1:strong-intra-smoothing=0:rect=0:aq-mode=1:hdr-opt=1:repeat-headers=1:colorprim=bt2020:range=limited:transfer=smpte2084:colormatrix=bt2020nc:range=limited:master-display='G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,1)':max-cll=1000,400",
                    # Output File
                    output_filename
                ])
            else:
                args.extend([
                    # Video Codec: Use NVIDIA HEVC encoder
                    "-c:v", "libx265",
                    "-profile:v", "main10",
                    "-preset", "slow",
                    "-pix_fmt", "yuv420p10le",
                    "-x265-params", "crf=18:vbv-bufsize=10000:vbv-maxrate=8000:selective-sao=0:no-sao=1:strong-intra-smoothing=0:rect=0:aq-mode=1:hdr-opt=1:repeat-headers=1:colorprim=bt2020:range=limited:transfer=smpte2084:colormatrix=bt2020nc:range=limited:master-display='G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,1)':max-cll=1000,400",
                    #Output File
                    output_filename
                ])
        else:
            if control['FrameEnhancerDownToggle']:
                args.extend([
                    # Video Codec: Use NVIDIA HEVC encoder
                    "-c:v", "hevc_nvenc",
                    # Quality Setting: NVENC uses -cq (Constant Quality scale, lower=better, ~18-28 is common range)
                    # Or use -qp (Constant Quantization Parameter)
                    # Or target bitrate: -b:v 60M
                    "-cq", "18", # Experiment with this value
                    # Preset: Controls speed vs quality trade-off for NVENC (e.g., p1-p7, default is p4/p5)
                    # p5=medium, p6=slow, p7=slower (higher quality)
                    "-preset", "p5",
                    # Set color properties (NVENC should respect these)
                    "-vf", f"scale={frame_width_down}x{frame_height_down}:flags=lanczos+accurate_rnd+full_chroma_int",
                    "-pix_fmt", output_pix_fmt,
                    "-colorspace", output_colorspace,
                    "-color_primaries", output_color_primaries,
                    "-color_trc", output_color_trc,
                    "-tag:v", "hvc1",
                    # Output File
                    output_filename
                ])
            else:
                args.extend([
                    # Video Codec: Use NVIDIA HEVC encoder
                    "-c:v", "hevc_nvenc",
                    # Quality Setting: NVENC uses -cq (Constant Quality scale, lower=better, ~18-28 is common range)
                    # Or use -qp (Constant Quantization Parameter)
                    # Or target bitrate: -b:v 60M
                    "-cq", "18", # Experiment with this value
                    # Preset: Controls speed vs quality trade-off for NVENC (e.g., p1-p7, default is p4/p5)
                    # p5=medium, p6=slow, p7=slower (higher quality)
                    "-preset", "p5",
                    # Set color properties (NVENC should respect these)
                    "-pix_fmt", output_pix_fmt,
                    "-colorspace", output_colorspace,
                    "-color_primaries", output_color_primaries,
                    "-color_trc", output_color_trc,
                    "-tag:v", "hvc1",
                    # Output File
                    output_filename
                ])
        
        try:
            self.recording_sp = subprocess.Popen(args, stdin=subprocess.PIPE, bufsize=-1)
            return True
        except FileNotFoundError:
            print("[ERROR] FFmpeg command not found. Ensure FFmpeg is installed and in system PATH.")
            self.main_window.display_messagebox_signal.emit('FFmpeg Error', 'FFmpeg command not found.', self.main_window)
            return False
        except Exception as e:
            print(f"[ERROR] Failed to start FFmpeg subprocess for segment {segment_num}: {e}")
            self.main_window.display_messagebox_signal.emit('FFmpeg Error', f'Failed to start FFmpeg for segment {segment_num}:\n{e}', self.main_window)
            return False


    def start_multi_segment_recording(self, segments: list[tuple[int, int]], triggered_by_job_manager: bool = False):
        if self.processing or self.is_processing_segments:
            print("[WARN] Attempted to start segment recording while already processing.")
            # Optionally stop existing process? Or just return? Returning is safer.
            # self.stop_processing()
            return

        if self.file_type != 'video':
            print("[ERROR] Multi-segment recording only supported for video files.")
            return
        if not segments:
             print("[ERROR] No segments provided for multi-segment recording.")
             return
        if not (self.media_capture and self.media_capture.isOpened()):
             print("[ERROR] Video source not open for multi-segment recording.")
             return

        print("--- Initializing multi-segment recording... ---")
        # Set flags for this mode
        self.is_processing_segments = True
        self.recording = False # Ensure default flag is off
        self.processing = True # General flag ON

        self.triggered_by_job_manager = triggered_by_job_manager
        self.segments_to_process = sorted(segments) # Ensure segments are processed in order
        self.current_segment_index = -1
        self.temp_segment_files = []
        self.segment_temp_dir = None # Reset just in case

        # Disable UI elements
        if not self.main_window.control['KeepControlsToggle']:
            layout_actions.disable_all_parameters_and_control_widget(self.main_window)

        # Create temporary directory
        try:
            # Use system temp dir for better cleanup potential - CHANGED
            # base_temp_dir = os.path.join(tempfile.gettempdir(), "VisoMasterSegments")
            # Define the base temp directory within the project
            base_temp_dir = os.path.join(os.getcwd(), "temp_files", "segments")
            os.makedirs(base_temp_dir, exist_ok=True) # Ensure base dir exists
            # Unique subdir for this run
            unique_id = uuid.uuid4()
            self.segment_temp_dir = os.path.join(base_temp_dir, f"run_{unique_id}")
            os.makedirs(self.segment_temp_dir, exist_ok=True)
            print(f"Created temporary directory for segments: {self.segment_temp_dir}")
        except Exception as e:
            print(f"[ERROR] Failed to create temporary directory: {e}")
            self.main_window.display_messagebox_signal.emit('File System Error', f'Failed to create temporary directory:\n{e}', self.main_window)
            self.stop_processing() # Abort start
            return

        self.start_time = time.perf_counter() # Start overall timer
        # Disconnect playback/default timers if somehow connected
        try: self.frame_read_timer.timeout.disconnect(self.process_next_frame)
        except (TypeError, RuntimeError): pass
        try: self.frame_display_timer.timeout.disconnect(self.display_next_frame)
        except (TypeError, RuntimeError): pass
        # Connect segment timer signal
        try: self.start_segment_timers_signal.disconnect(self._start_timers_from_signal)
        except (TypeError, RuntimeError): pass
        self.start_segment_timers_signal.connect(self._start_timers_from_signal)

        # Start processing the first segment
        self.process_next_segment()

    @Slot(int)
    def _start_timers_from_signal(self, interval: int):
        """Slot to start frame read and display timers from the main thread for segments."""
        if not self.is_processing_segments:
            print("[WARN] _start_timers_from_signal called but not in segment processing mode.")
            return

        print(f"Starting segment processing timers (from signal) with interval {interval} ms.")

        # Ensure correct timer connections for segment mode
        try: self.frame_read_timer.timeout.disconnect() # Disconnect any previous
        except (TypeError, RuntimeError): pass
        try: self.frame_display_timer.timeout.disconnect()
        except (TypeError, RuntimeError): pass

        self.frame_read_timer.timeout.connect(self.process_next_segment_frame) # Use dedicated frame reader
        self.frame_display_timer.timeout.connect(self.display_next_frame) # Display logic is shared

        if interval <= 0: interval = 1 # Ensure positive interval
        self.frame_read_timer.start(interval)
        self.frame_display_timer.start(interval) # Match intervals
        self.gpu_memory_update_timer.start(5000)

        self.processing_started_signal.emit() # EMIT UNIFIED SIGNAL HERE

    def process_next_segment(self):
        """Sets up and starts processing for the next segment in the list."""
        self.current_segment_index += 1
        segment_num = self.current_segment_index + 1

        if self.current_segment_index >= len(self.segments_to_process):
            print("All segments processed.")
            self.finalize_segment_concatenation()
            return

        start_frame, end_frame = self.segments_to_process[self.current_segment_index]
        print(f"--- Starting Segment {segment_num}/{len(self.segments_to_process)} (Frames: {start_frame} - {end_frame}) ---")
        self.current_segment_end_frame = end_frame

        if not self.media_capture or not self.media_capture.isOpened():
            print(f"[ERROR] Media capture not available for seeking to segment {segment_num}.")
            self.stop_processing() # Abort
            return

        # --- Seek and Prepare ---
        print(f"Seeking to start frame {start_frame}...")
        self.media_capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        # Read frame after seek to verify and populate self.current_frame
        ret, frame_bgr = misc_helpers.read_frame(self.media_capture, preview_mode=False)
        if ret:
            self.current_frame = numpy.ascontiguousarray(frame_bgr[..., ::-1]) # BGR to RGB
            # Reset position again to ensure reading starts exactly at start_frame
            self.media_capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            self.current_frame_number = start_frame
            self.next_frame_to_display = start_frame
            # Update UI slider (blocked)
            self.main_window.videoSeekSlider.blockSignals(True)
            self.main_window.videoSeekSlider.setValue(start_frame)
            self.main_window.videoSeekSlider.blockSignals(False)
        else:
            print(f"[ERROR] Could not read frame {start_frame} at start of segment {segment_num}. Aborting.")
            self.stop_processing() # Abort
            return

        # Clear queues and thread dict for the new segment
        self.frames_to_display.clear()
        with self.frame_queue.mutex: self.frame_queue.queue.clear()
        self.threads.clear()


        # --- Create FFmpeg Subprocess for this Segment ---
        temp_segment_filename = f"segment_{self.current_segment_index:03d}.mp4" # Padded index
        temp_segment_path = os.path.join(self.segment_temp_dir, temp_segment_filename)
        self.temp_segment_files.append(temp_segment_path) # Store path for concatenation

        if not self.create_ffmpeg_subprocess_segment(output_filename=temp_segment_path):
            print(f"[ERROR] Failed to create ffmpeg subprocess for segment {segment_num}. Aborting.")
            self.stop_processing() # Abort
            return

        # --- Start Timers via Signal ---
        # Calculate interval based on actual video FPS
        fps = self.media_capture.get(cv2.CAP_PROP_FPS)
        if fps <= 0: fps = self.fps if self.fps > 0 else 30 # Use stored or default
        interval = 1000 / fps if fps > 0 else 33
        interval = int(interval) # Use integer interval for timers
        if interval <=0 : interval = 1

        # Emit signal to start timers from main thread
        self.start_segment_timers_signal.emit(interval)


    def process_next_segment_frame(self):
        """Reads the next frame specifically for multi-segment processing."""
        if not self.is_processing_segments:
             # Should not happen if timers are managed correctly, but good check
             print("[WARN] process_next_segment_frame called unexpectedly.")
             self.frame_read_timer.stop()
             return

        # Check if current segment is finished based on frame number
        if self.current_segment_end_frame is not None and self.current_frame_number > self.current_segment_end_frame:
            # We have read past the end frame for this segment. Stop reading.
            # The display loop will trigger stop_current_segment when it displays the last frame.
            print(f"Segment {self.current_segment_index + 1} read limit ({self.current_segment_end_frame}) reached. Stopping frame read.")
            self.frame_read_timer.stop()
            return

        # Check queue size
        if self.frame_queue.qsize() >= self.num_threads:
            return # Wait for queue to drain

        # Read frame (use preview_mode=False for recording)
        ret, frame_bgr = misc_helpers.read_frame(self.media_capture, preview_mode=False)

        if ret:
            frame_rgb = frame_bgr[..., ::-1] # BGR to RGB
            self.frame_queue.put(self.current_frame_number)
            # Start worker asynchronously, indicating it's part of recording
            self.start_frame_worker(self.current_frame_number, frame_rgb, is_single_frame=False) # is_recording_mode handled internally
            self.current_frame_number += 1
        else:
            # Frame read failed during segment processing!
            failed_frame_num = self.current_frame_number
            print(f"[ERROR] Cannot read frame {failed_frame_num} during segment {self.current_segment_index + 1}! Video source may be corrupted.")
            self.frame_read_timer.stop() # Stop trying to read

            # Attempt to finalize the current segment and then proceed to concatenation
            print(f"Attempting to finalize segment {self.current_segment_index + 1} due to read error.")
            self.stop_current_segment() # This will try to close ffmpeg and move to next/finalize


    def stop_current_segment(self):
        """Stops processing the current segment and triggers the next one or finalization."""
        if not self.is_processing_segments:
            print("[WARN] stop_current_segment called but not processing segments.")
            return

        segment_num = self.current_segment_index + 1
        print(f"--- Stopping Segment {segment_num} --- ")

        # Stop timers for this segment
        self.frame_read_timer.stop()
        self.frame_display_timer.stop()
        self.gpu_memory_update_timer.stop() # Stop GPU monitor too between segments

        # Disconnect timer signals for safety before next segment connects
        try: self.frame_read_timer.timeout.disconnect()
        except (TypeError, RuntimeError): pass
        try: self.frame_display_timer.timeout.disconnect()
        except (TypeError, RuntimeError): pass

        # Wait for threads from this segment to finish
        print(f"Waiting for workers from segment {segment_num}...")
        self.join_and_clear_threads()
        print("Workers joined.")
        self.frames_to_display.clear() # Clear display queue

        # Close and wait for the segment's ffmpeg subprocess
        if self.recording_sp:
            if self.recording_sp.stdin and not self.recording_sp.stdin.closed:
                try:
                    print(f"Closing FFmpeg stdin for segment {segment_num}...")
                    self.recording_sp.stdin.close()
                except OSError as e:
                    print(f"[WARN] Error closing FFmpeg stdin for segment {segment_num}: {e}")
            print(f"Waiting for FFmpeg subprocess (segment {segment_num}) to finish writing...")
            try:
                self.recording_sp.wait(timeout=10) # Wait with timeout
                print(f"FFmpeg subprocess (segment {segment_num}) finished.")
            except subprocess.TimeoutExpired:
                print(f"[WARN] FFmpeg subprocess (segment {segment_num}) timed out, killing.")
                self.recording_sp.kill()
                self.recording_sp.wait()
            except Exception as e:
                 print(f"[ERROR] Error waiting for FFmpeg subprocess (segment {segment_num}): {e}")
            self.recording_sp = None
        else:
            print(f"[WARN] No active FFmpeg subprocess found when stopping segment {segment_num}.")

        # Check if the segment file was actually created
        if self.temp_segment_files and not os.path.exists(self.temp_segment_files[-1]):
            print(f"[ERROR] Segment file '{self.temp_segment_files[-1]}' not found after processing segment {segment_num}. It might be empty or FFmpeg failed.")
            # Decide whether to continue or abort? For now, log and continue to concatenation attempt.

        # Move to the next segment
        self.process_next_segment()

    def finalize_segment_concatenation(self):
        """Concatenates all valid temporary segment files into the final output file."""
        print("--- Finalizing concatenation of segments... ---")

        # --- Gracefully stop current FFmpeg process if active (for early stop) ---
        if self.recording_sp:
            segment_num = self.current_segment_index + 1
            print(f"Finalizing: Stopping active FFmpeg process for segment {segment_num}...")
            if self.recording_sp.stdin and not self.recording_sp.stdin.closed:
                try:
                    self.recording_sp.stdin.close()
                except OSError as e:
                    print(f"[WARN] Error closing FFmpeg stdin during early finalization: {e}")
            try:
                self.recording_sp.wait(timeout=10) # Wait up to 10 seconds for FFmpeg to finish
                print(f"FFmpeg subprocess (segment {segment_num}) finished writing.")
            except subprocess.TimeoutExpired:
                print(f"[WARN] FFmpeg subprocess (segment {segment_num}) timed out during early finalization, killing.")
                self.recording_sp.kill()
                self.recording_sp.wait()
            except Exception as e:
                print(f"[ERROR] Error waiting for FFmpeg subprocess during early finalization: {e}")
            self.recording_sp = None # Ensure it's cleared
        # --- End graceful stop ---

        was_triggered_by_job = self.triggered_by_job_manager # Store flag before reset

        # Ensure processing flags are off before finalization
        self.processing = False
        self.is_processing_segments = False
        self.recording = False # Ensure all flags off

        # Check if there are any valid segment files to process
        valid_segment_files = [
            f for f in self.temp_segment_files
            if f and os.path.exists(f) and os.path.getsize(f) > 0
        ]

        if not valid_segment_files:
            print("[WARN] No valid temporary segment files found to concatenate. Aborting finalization.")
            self._cleanup_temp_dir()
            # Reset UI fully
            layout_actions.enable_all_parameters_and_control_widget(self.main_window)
            video_control_actions.reset_media_buttons(self.main_window)
            # Reset state vars (already done by _cleanup, but belt & suspenders)
            self.segments_to_process = []
            self.current_segment_index = -1
            self.temp_segment_files = []
            self.triggered_by_job_manager = False
            return # Nothing to concatenate

        # --- Determine Final Output Path ---
        job_name = getattr(self.main_window, 'current_job_name', None) if was_triggered_by_job else None
        use_job_name = getattr(self.main_window, 'use_job_name_for_output', False) if was_triggered_by_job else False
        output_file_name = getattr(self.main_window, 'output_file_name', None) if was_triggered_by_job else None

        final_file_path = misc_helpers.get_output_file_path(
            self.media_path,
            self.main_window.control['OutputMediaFolder'],
            job_name=job_name,
            use_job_name_for_output=use_job_name,
            output_file_name=output_file_name
        )
        #print(f"[DEBUG] _finalize_default_style_recording: Computed final_file_path={final_file_path}")

        # --- Ensure Output Directory Exists ---
        output_dir = os.path.dirname(final_file_path)
        if not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
                print(f"Created output directory: {output_dir}")
            except OSError as e:
                print(f"[ERROR] Failed to create output directory {output_dir}: {e}")
                self.main_window.display_messagebox_signal.emit('File Error', f'Could not create output directory:\n{output_dir}\n\n{e}', self.main_window)
                self._cleanup_temp_dir()
                layout_actions.enable_all_parameters_and_control_widget(self.main_window)
                video_control_actions.reset_media_buttons(self.main_window)
                return

        # --- Remove Existing Final File ---
        if Path(final_file_path).is_file():
            print(f"Removing existing final file: {final_file_path}")
            try:
                os.remove(final_file_path)
            except OSError as e:
                print(f"[ERROR] Failed to remove existing file {final_file_path}: {e}")
                self.main_window.display_messagebox_signal.emit('File Error', f'Could not delete existing file:\n{final_file_path}\n\n{e}', self.main_window)
                self._cleanup_temp_dir()
                layout_actions.enable_all_parameters_and_control_widget(self.main_window)
                video_control_actions.reset_media_buttons(self.main_window)
                return

        # --- Create FFmpeg Concat List File ---
        list_file_path = os.path.join(self.segment_temp_dir, "mylist.txt")
        concatenation_successful = False
        try:
            print(f"Creating ffmpeg list file: {list_file_path}")
            with open(list_file_path, 'w', encoding='utf-8') as f_list: # Specify encoding
                for segment_path in valid_segment_files:
                     # Use absolute path, sanitize for ffmpeg concat demuxer
                     abs_path = os.path.abspath(segment_path)
                     # FFmpeg concat requires forward slashes, even on Windows
                     formatted_path = abs_path.replace('\\', '/') # Correctly escape backslash for replacement
                     # Paths with spaces or special chars need single quotes
                     # Ensure a proper newline character is written
                     f_list.write(f"file '{formatted_path}'" + os.linesep)

            print(f"Concatenating {len(valid_segment_files)} valid segments into {final_file_path}...")
            concat_args = [
                "ffmpeg", '-hide_banner', '-loglevel', 'error',
                "-f", "concat",
                "-safe", "0", # Allow unsafe paths (though we used absolute)
                "-i", list_file_path,
                "-c:v", "copy", # Copy streams directly without re-encoding
                "-af", "aresample=async=1000", # Audio resample for sync
                final_file_path
            ]
            subprocess.run(concat_args, check=True) # check=True raises error on failure
            concatenation_successful = True
            log_prefix = "Job Manager: " if was_triggered_by_job else ""
            print(f"--- {log_prefix}Successfully created final video: {final_file_path} ---")

        except subprocess.CalledProcessError as e:
            print(f"[ERROR] FFmpeg command failed during final concatenation: {e}")
            print(f"FFmpeg arguments: {' '.join(concat_args)}")
            self.main_window.display_messagebox_signal.emit('Recording Error', f'FFmpeg command failed during concatenation:\n{e}\nCould not create final video.', self.main_window)
        except FileNotFoundError:
            print("[ERROR] FFmpeg not found. Ensure it's in your system PATH.")
            self.main_window.display_messagebox_signal.emit('Recording Error', 'FFmpeg not found.', self.main_window)
        except Exception as e:
            print(f"[ERROR] An unexpected error occurred during finalization: {e}")
            self.main_window.display_messagebox_signal.emit('Recording Error', f'An unexpected error occurred:\n{e}', self.main_window)

        finally:
            # --- Cleanup and Reset ---
            self._cleanup_temp_dir() # Always cleanup temp dir

            # Reset segment state regardless of success/failure
            self.segments_to_process = []
            self.current_segment_index = -1
            self.temp_segment_files = []
            self.current_segment_end_frame = None
            self.triggered_by_job_manager = False

            self.end_time = time.perf_counter()
            processing_time = self.end_time - self.start_time

            if concatenation_successful:
                print(f"Total segment processing and concatenation finished in {processing_time:.2f} seconds")
            else:
                print(f"Segment processing/concatenation failed or aborted after {processing_time:.2f} seconds.")

            # Final GPU clear and GC
            print("Clearing GPU Cache and running garbage collection post-concatenation.")
            try:
                if torch.cuda.is_available(): torch.cuda.empty_cache()
            except ImportError: pass
            except Exception as e: print(f"[WARN] Error clearing Torch cache: {e}")
            gc.collect()

            # Always re-enable UI and reset buttons
            layout_actions.enable_all_parameters_and_control_widget(self.main_window)
            video_control_actions.reset_media_buttons(self.main_window)
            print("Multi-segment processing flow finished.")
            
            if self.main_window.control['OpenOutputToggle']:
                try: list_view_actions.open_output_media_folder(self.main_window)
                except Exception: pass

    def _cleanup_temp_dir(self):
        """Safely removes the temporary directory used for segments."""
        if self.segment_temp_dir and os.path.exists(self.segment_temp_dir):
            try:
                print(f"Cleaning up temporary segment directory: {self.segment_temp_dir}")
                # Force removal even if files are somehow locked (use with caution)
                shutil.rmtree(self.segment_temp_dir, ignore_errors=True)
            except Exception as e: # Catch broader exceptions just in case
                print(f"[WARN] Failed to delete temporary directory {self.segment_temp_dir}: {e}")
        self.segment_temp_dir = None # Reset variable

    def start_live_sound(self):
        # Start up audio if requested
        seek_time = (self.next_frame_to_display)/self.fps
        # Calculate custom fps from slider to evaluate audio speed playback
        # Change audio speed slider too volume slider
        
        fpsdiv = 1
        if self.main_window.control['VideoPlaybackCustomFpsToggle'] and not self.recording: # Use custom FPS only for playback
            fpsorig = self.media_capture.get(cv2.CAP_PROP_FPS)
            fpscust = self.main_window.control['VideoPlaybackCustomFpsSlider']
            fpsdiv = fpscust / fpsorig
        if fpsdiv < 0.5: fpsdiv = 0.5
        args =  ["ffplay",
                '-vn',
                '-ss', str(seek_time),
                '-nodisp',
                '-stats',
                '-loglevel',  'quiet',
                '-sync',  'audio',
                '-af', f'volume={self.main_window.control["LiveSoundVolumeDecimalSlider"]}, atempo={fpsdiv}', # Audio speed and volume
                self.media_path]

        self.ffplay_sound_sp = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    def stop_live_sound(self):
        if self.ffplay_sound_sp:
            parent_pid = self.ffplay_sound_sp.pid

            try:
                # Terminate any child processes spawned by ffplay
                try:
                    parent_proc = psutil.Process(parent_pid)
                    children = parent_proc.children(recursive=True)
                    for child in children:
                        try:
                            child.kill()
                        except psutil.NoSuchProcess:
                            pass  # The child process has already terminated
                except psutil.NoSuchProcess:
                    pass  # The parent process has already terminated

                # Terminate the parent process
                self.ffplay_sound_sp.terminate()
                try:
                    self.ffplay_sound_sp.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.ffplay_sound_sp.kill()

            except psutil.NoSuchProcess:
                pass  # The process no longer exists

            self.ffplay_sound_sp = None

    # --- End Multi-Segment Methods ---

    # Add method to start webcam streaming
    def process_webcam(self):
        """Start webcam streaming: read and display frames from webcam."""
        if self.processing:
            return
        if self.file_type != 'webcam':
            print("process_webcam: Only applicable for webcam input.")
            return
        if not (self.media_capture and self.media_capture.isOpened()):
            print("Error: Unable to open webcam source.")
            video_control_actions.reset_media_buttons(self.main_window)
            return
        self.processing = True
        self.is_processing_segments = False
        self.recording = False
        self.frames_to_display.clear()
        self.webcam_frames_to_display.queue.clear()
        with self.frame_queue.mutex:
            self.frame_queue.queue.clear()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            try:
                self.frame_read_timer.timeout.disconnect()
                self.frame_display_timer.timeout.disconnect()
            except (TypeError, RuntimeError):
                pass
        self.frame_read_timer.timeout.connect(self.process_next_webcam_frame)
        self.frame_display_timer.timeout.connect(self.display_next_webcam_frame)
        fps = self.media_capture.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30
        self.fps = fps
        interval = int(1000 / fps) if fps > 0 else 33
        if interval <= 0:
            interval = 1
        self.frame_read_timer.start(interval)
        self.frame_display_timer.start(interval)
        self.gpu_memory_update_timer.start(5000)
        self.processing_started_signal.emit()