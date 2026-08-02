import queue
import threading
import gc
from typing import TYPE_CHECKING, List, Optional, Tuple, Dict, Any

import numpy
import torch
from PySide6.QtCore import QObject, QTimer

from app.processors.workers.frame_worker import FrameWorker
from app.helpers.typing_helper import ControlTypes, FacesParametersTypes

if TYPE_CHECKING:
    from app.ui.main_ui import MainWindow


class WorkerPoolManager(QObject):
    """
    Encapsulates all Threading, Queue, and VRAM lifecycle management.
    
    Responsibilities:
    - Maintains the persistent pool of FrameWorkers for continuous video processing.
    - Manages the single-frame asynchronous worker used for timeline scrubbing.
    - Debounces rapid UI scrubbing requests via QTimer to prevent TensorRT context crashes.
    - Strictly controls VRAM garbage collection (torch.cuda.empty_cache) upon thread termination.
    """

    def __init__(self, main_window: "MainWindow"):
        super().__init__()
        self.main_window = main_window

        # --- Persistent Pool State (Video / Webcam) ---
        # The Queue holds tasks: (frame_num, frame_rgb, params, control, bboxes, kpss_5, kpss, kpss_203)
        self.frame_queue: queue.Queue[Optional[Tuple[int, numpy.ndarray, FacesParametersTypes, ControlTypes, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]]] = queue.Queue()
        self.worker_threads: List[FrameWorker] = []

        # --- Single-Frame State (Scrubbing / Image processing) ---
        self.current_single_frame_worker: Optional[FrameWorker] = None
        
        # Generation tracking prevents out-of-order frames from updating the UI during fast scrubs
        self.single_frame_request_generation: int = 0
        self.active_single_frame_request_generation: int = 0
        self.fit_on_single_frame_request_generation: Optional[int] = None
        
        self.pending_single_frame_request: Optional[Dict[str, Any]] = None
        
        # QTimer for debouncing fast slider movements
        self.single_frame_handoff_timer = QTimer(self)
        self.single_frame_handoff_timer.setInterval(15)
        self.single_frame_handoff_timer.timeout.connect(self._try_start_pending_single_frame_worker)

    def recreate_queue(self, maxsize: int) -> None:
        """
        Safely recreates the frame queue with a new maximum size limit.
        Used to bound RAM usage dynamically based on the number of threads.
        """
        with self.frame_queue.mutex:
            self.frame_queue.queue.clear()
            self.frame_queue.all_tasks_done.notify_all()
            self.frame_queue.not_full.notify_all()
        self.frame_queue = queue.Queue(maxsize=maxsize)

    def start_persistent_pool(self, num_threads: int) -> None:
        """Starts the persistent FrameWorker pool for continuous processing."""
        print(f"[INFO] WorkerPoolManager: Starting {num_threads} persistent worker thread(s)...")
        self.worker_threads = []
        for i in range(num_threads):
            worker = FrameWorker(
                frame_queue=self.frame_queue,
                main_window=self.main_window,
                worker_id=i,
            )
            worker.start()
            self.worker_threads.append(worker)

    def join_and_clear_threads(self, clear_module_caches: bool = True) -> None:
        """
        Stops and waits for all pool worker threads to finish.
        Sends poison pills to wake blocked workers and enforces strict VRAM cleanup.

        Args:
            clear_module_caches: If True, clears module-level VR caches. Set to False 
                                 during mid-job pool restarts to keep caches warm.
        """
        active_threads = self.worker_threads
        if not active_threads:
            return  # Nothing to do

        print(f"[INFO] WorkerPoolManager: Signaling {len(active_threads)} active worker(s) to stop...")

        # 1. Set stop event for all workers in the pool
        for thread in active_threads:
            if hasattr(thread, "stop_event") and not thread.stop_event.is_set():
                try:
                    thread.stop_event.set()
                except Exception as e:
                    print(f"[WARN] Error setting stop_event on thread {thread.name}: {e}")

        # 2. Wake up any workers blocked on queue.get() by sending a "poison pill" (None).
        # Clear the queue first so pills are never lost when the queue is full.
        with self.frame_queue.mutex:
            self.frame_queue.queue.clear()
            
        for _ in active_threads:
            try:
                self.frame_queue.put(None, block=False)
            except queue.Full:
                pass
            except Exception as e:
                print(f"[WARN] Error putting poison pill in queue: {e}")

        # 3. Join all threads safely
        for thread in active_threads:
            try:
                if thread.is_alive():
                    thread.join(timeout=2.0)
                    if thread.is_alive():
                        print(f"[WARN] Thread {thread.name} did not join gracefully.")
            except Exception as e:
                print(f"[WARN] Error joining thread {thread.name}: {e}")

        # 4. Clear the worker list
        self.worker_threads.clear()

        # 5. Strict VRAM Garbage Collection
        # Release GPU memory held by the now-dead workers (kernel tensors, etc.).
        gc.collect()
        # PROTECTED: Only empty cache if CUDA is already awake. 
        # Calling empty_cache() on an asleep GPU forces a 2GB context initialization.
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            torch.cuda.empty_cache()

        # 6. Release module-level VR caches (Prevents RAM memory leak across jobs)
        if clear_module_caches:
            try:
                from app.processors.external.Equirec2Perspec_vr import clear_persp_cache
                from app.helpers.vr_utils import clear_feathered_mask_cache

                clear_persp_cache()
                clear_feathered_mask_cache()
            except Exception:
                pass 

    def _launch_async_single_frame_worker(self, frame_number: int, frame: numpy.ndarray, generation: int) -> FrameWorker:
        """Launches a one-shot worker explicitly for UI single-frame processing."""
        worker = FrameWorker(
            frame=frame,
            main_window=self.main_window,
            frame_number=frame_number,
            frame_queue=None,
            is_single_frame=True,
            worker_id=-1,
        )
        worker.preview_generation = generation
        self.current_single_frame_worker = worker
        worker.start()
        return worker

    def _try_start_pending_single_frame_worker(self) -> None:
        """Timer callback: Launches the pending frame if the previous worker has finished."""
        if self.pending_single_frame_request is None:
            self.single_frame_handoff_timer.stop()
            return

        current_worker = self.current_single_frame_worker
        if current_worker is not None and current_worker.is_alive():
            return  # Still busy, wait for the next timer tick

        request = self.pending_single_frame_request
        self.pending_single_frame_request = None
        self.single_frame_handoff_timer.stop()
        self.current_single_frame_worker = None
        
        self._launch_async_single_frame_worker(
            request["frame_number"],
            request["frame"],
            request["generation"],
        )

    def cancel_single_frame_preview_state(self) -> None:
        """Immediately aborts any active or pending UI single-frame scrubs."""
        self.single_frame_request_generation += 1
        self.active_single_frame_request_generation = self.single_frame_request_generation
        self.pending_single_frame_request = None
        self.single_frame_handoff_timer.stop()
        self.fit_on_single_frame_request_generation = None

        worker = self.current_single_frame_worker
        if worker is not None and worker.is_alive():
            worker.stop_event.set()
            worker.join(timeout=2.0)
            if worker.is_alive():
                print("[WARN] Single-frame preview worker did not join gracefully.")
                self.current_single_frame_worker = None
                return

        self.current_single_frame_worker = None

    def start_single_frame_worker(self, frame_number: int, frame: numpy.ndarray, is_single_frame: bool = False, synchronous: bool = False, fit_on_complete: bool = False) -> Optional[FrameWorker]:
        """
        Manages the execution of a single frame outside the main video pool.
        Crucial for thread safety during fast UI scrubbing (debouncing).
        """
        # Stop any previous single-frame worker before starting a new one to prevent 
        # concurrent TensorRT inference crashes.
        prev = self.current_single_frame_worker

        if synchronous:
            self.pending_single_frame_request = None
            self.single_frame_handoff_timer.stop()
            if prev is not None and prev.is_alive():
                prev.stop_event.set()
                prev.join()
            self.current_single_frame_worker = None
            
            worker = FrameWorker(
                frame=frame, 
                main_window=self.main_window,
                frame_number=frame_number,
                frame_queue=None, 
                is_single_frame=is_single_frame,
                worker_id=-1,
            )
            
            self.fit_on_single_frame_request_generation = 0 if fit_on_complete else None
            worker.preview_generation = 0
            worker.run() # Blocking execution
            return worker
            
        else:
            # Asynchronous Execution (Debounced)
            self.single_frame_request_generation += 1
            self.active_single_frame_request_generation = self.single_frame_request_generation
            
            self.fit_on_single_frame_request_generation = self.single_frame_request_generation if fit_on_complete else None
                
            request = {
                "frame_number": frame_number,
                "frame": frame,
                "generation": self.single_frame_request_generation,
            }
            
            if prev is not None and prev.is_alive():
                prev.stop_event.set()

            self.pending_single_frame_request = request
            
            # Dynamically fetch user-defined delay
            frameworker_delay = max(
                int(self.main_window.control.get("FrameWorkerDelayDecimalSlider", 0.3) * 1000),
                15,
            )
            self.single_frame_handoff_timer.setInterval(frameworker_delay)
            self.single_frame_handoff_timer.start()

            return prev