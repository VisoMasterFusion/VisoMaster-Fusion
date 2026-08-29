import uuid
from functools import partial
from typing import TYPE_CHECKING, Dict, List, Tuple, Optional
import traceback
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import torch
import numpy
from PySide6 import QtCore as qtc
from PySide6.QtGui import QImage

from app.helpers import miscellaneous as misc_helpers
from app.ui.widgets.actions import common_actions as common_widget_actions
from app.ui.widgets.actions import filter_actions
from app.ui.widgets.actions import target_videos_list_actions
from app.ui.widgets.settings_layout_data import CAMERA_BACKENDS
from app.processors.video_utils.issue_scanner import IssueScanner

if TYPE_CHECKING:
    from app.ui.main_ui import MainWindow


class TargetMediaLoaderWorker(qtc.QThread):
    # Define signals to emit when loading is done or if there are updates - changed to QImage
    thumbnail_ready = qtc.Signal(
        str, QImage, str, str, object
    )  # media path, QImage, file_type, media_id, MediaMetadata|None
    webcam_thumbnail_ready = qtc.Signal(str, QImage, str, str, int, int)
    finished = qtc.Signal()  # Signal to indicate completion

    def __init__(
        self,
        main_window: "MainWindow",
        folder_name=False,
        files_list=None,
        media_ids=None,
        sort_files_list_by_name=True,
        metadata_enabled=None,
        webcam_mode=False,
        parent=None,
    ):
        super().__init__(parent)
        self.main_window = main_window
        self.folder_name = folder_name
        self.files_list = files_list or []
        self.media_ids = media_ids or []
        self.sort_files_list_by_name = sort_files_list_by_name

        # Safely fetch UI states once to avoid duplication
        filter_button = getattr(main_window, "targetVideosFilterMenuButton", None)
        filter_panel_open = (
            filter_button.isChecked() if filter_button is not None else False
        )

        # Check if the search text box contains active text
        search_box = getattr(main_window, "targetVideosSearchBox", None)
        search_text_active = bool(search_box.text().strip()) if search_box else False

        sort_needs_metadata = target_videos_list_actions.current_sort_needs_metadata(
            main_window
        )

        # Evaluate metadata requirement including the search text state
        if metadata_enabled is None:
            metadata_enabled = (
                filter_panel_open or search_text_active or sort_needs_metadata
            )

        self.metadata_enabled = bool(metadata_enabled)

        print(
            f"[INFO] TargetMediaLoaderWorker initialized. Metadata extraction: "
            f"{self.metadata_enabled} (Filter panel open: {filter_panel_open}, "
            f"Search active: {search_text_active}, "
            f"Sorting requires metadata: {sort_needs_metadata})"
        )

        self.webcam_mode = webcam_mode
        self._running = True  # Flag to control the running state
        self.control_snapshot = (
            main_window.control.copy() if getattr(main_window, "control", None) else {}
        )

    def run(self):
        if self.folder_name:
            self.load_videos_and_images_from_folder(self.folder_name)
        if self.files_list:
            self.load_videos_and_images_from_files_list(self.files_list)
        if self.webcam_mode:
            self.load_webcams()
        self.finished.emit()

    def _iter_sorted_recursive_media_files(self, folder_name: str):
        for dirpath, dirnames, filenames in os.walk(folder_name, topdown=True):
            dirnames.sort(key=str.lower)
            for filename in sorted(filenames, key=str.lower):
                media_file_path = os.path.abspath(os.path.join(dirpath, filename))
                if misc_helpers.get_file_type(media_file_path):
                    yield media_file_path

    def load_videos_and_images_from_folder(self, folder_name: str) -> None:
        # Initially hide the placeholder text
        self.main_window.placeholder_update_signal.emit(
            self.main_window.targetVideosList, True
        )
        recursive_toggle = self.control_snapshot.get(
            "TargetMediaFolderRecursiveToggle", False
        )

        if recursive_toggle:
            media_files = list(self._iter_sorted_recursive_media_files(folder_name))
            media_count = len(media_files)
        else:
            video_files = misc_helpers.get_video_files(folder_name, recursive_toggle)
            image_files = misc_helpers.get_image_files(folder_name, recursive_toggle)
            media_files = video_files + image_files
            # Sorting the list
            media_files.sort(key=lambda x: os.path.basename(str(x)).lower())
            media_count = len(media_files)

        print(
            f"[INFO] TargetMediaLoaderWorker: Preparing thumbnails from folder '{folder_name}'. "
            f"Total media count: {media_count}, Metadata extraction enabled: {self.metadata_enabled}."
        )

        paired_files_ids = [
            (
                os.path.join(folder_name, f),
                self.media_ids[i] if self.media_ids else str(uuid.uuid1().int),
            )
            for i, f in enumerate(media_files)
        ]

        self._process_media_concurrently(paired_files_ids)
        # Show/Hide the placeholder text based on the number of items in ListWidget
        self.main_window.placeholder_update_signal.emit(
            self.main_window.targetVideosList, False
        )

    def load_videos_and_images_from_files_list(self, files_list: List[str]) -> None:
        self.main_window.placeholder_update_signal.emit(
            self.main_window.targetVideosList, True
        )

        # Associate ID and Paths before sorting
        paired_files_ids = []
        for idx, path in enumerate(files_list):
            m_id = self.media_ids[idx] if self.media_ids else str(uuid.uuid1().int)
            paired_files_ids.append((path, m_id))

        # Keep existing behavior by default; allow callers to preserve original order.
        if self.sort_files_list_by_name:
            paired_files_ids.sort(key=lambda x: os.path.basename(str(x[0])).lower())

        self._process_media_concurrently(paired_files_ids)

        self.main_window.placeholder_update_signal.emit(
            self.main_window.targetVideosList, False
        )

    def _process_media_concurrently(
        self, paired_files_ids: List[Tuple[str, str]]
    ) -> None:
        """
        Executes media extraction concurrently to maximize I/O throughput.
        Maintains order by using executor.map.
        """

        def extract_task(
            media_file_path: str, media_id: str
        ) -> Tuple[str, QImage, str, str, Optional[object]]:
            # Guarantee a return tuple even on failure to maintain 1:1 progress counts
            if not os.path.exists(media_file_path):
                return (media_file_path, QImage(), "error", media_id, None)

            file_type = misc_helpers.get_file_type(media_file_path)
            if not file_type:
                return (media_file_path, QImage(), "error", media_id, None)

            thumbnail_result = common_widget_actions.extract_frame_as_image(
                self.main_window,
                media_file_path,
                file_type,
                cache_thumbnail=True,
                return_metadata=self.metadata_enabled,
            )

            if self.metadata_enabled:
                q_image, metadata = thumbnail_result
            else:
                q_image, metadata = thumbnail_result, None

            if q_image:
                return (media_file_path, q_image, file_type, media_id, metadata)

            # Fallback for extraction failure to prevent UI progress bar hanging
            return (media_file_path, QImage(), "error", media_id, None)

        # Determine thread count safely, capping at 16 to avoid OS resource exhaustion
        max_workers = min(16, (os.cpu_count() or 1) + 4)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for result in executor.map(lambda p: extract_task(*p), paired_files_ids):
                if not self._running:
                    executor.shutdown(wait=False, cancel_futures=True)
                    break
                if result:
                    # Thread-safe signal emission to the main GUI thread
                    self.thumbnail_ready.emit(*result)

    def load_webcams(self):
        self.main_window.placeholder_update_signal.emit(
            self.main_window.targetVideosList, True
        )
        camera_backend = CAMERA_BACKENDS[
            self.control_snapshot.get("WebcamBackendSelection", "DirectShow")
        ]
        max_no = int(self.control_snapshot.get("WebcamMaxNoSelection", 1))

        for i in range(max_no):
            try:
                q_image = common_widget_actions.extract_frame_as_image(
                    self.main_window,
                    media_file_path=f"Webcam {i}",
                    file_type="webcam",
                    webcam_index=i,
                    webcam_backend=camera_backend,
                )
                media_id = str(uuid.uuid1().int)

                if q_image:
                    # Emit the signal to update GUI
                    self.webcam_thumbnail_ready.emit(
                        f"Webcam {i}", q_image, "webcam", media_id, i, camera_backend
                    )
            except Exception:
                traceback.print_exc()

        self.main_window.placeholder_update_signal.emit(
            self.main_window.targetVideosList, False
        )

    def stop(self):
        # Stop the thread by setting the running flag to False.
        self._running = False
        self.quit()
        self.wait(1000)
        if self.isRunning():
            self.terminate()


class IssueScanWorker(qtc.QThread):
    progress = qtc.Signal(int, int, int, float)
    completed = qtc.Signal(object, int, int, str, float, bool)
    issue_found = qtc.Signal(str, int)
    cancelled = qtc.Signal()
    failed = qtc.Signal(str)

    def __init__(self, main_window: "MainWindow", parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self._cancel_event = threading.Event()
        self._scan_ranges = main_window.video_processor._get_issue_scan_ranges()
        self._scan_scope_text = main_window.video_processor.describe_issue_scan_scope(
            self._scan_ranges
        )
        self._base_control = IssueScanner._filter_scan_control(
            main_window.control.copy()
        )
        self._base_params = IssueScanner._filter_scan_face_params(
            {
                face_id: params.copy()
                for face_id, params in main_window.parameters.items()
            },
            getattr(main_window, "target_faces", {}).keys(),
        )
        self._control_defaults_snapshot = IssueScanner._filter_scan_control(
            {
                widget_name: widget.default_value
                for widget_name, widget in main_window.parameter_widgets.items()
                if widget_name in main_window.control
            }
        )
        self._target_faces_snapshot = (
            main_window.video_processor.prepare_issue_scan_target_faces_snapshot(
                self._scan_ranges,
                self._base_control,
                self._base_params,
                self._control_defaults_snapshot,
            )
        )
        self._reset_frame_number = int(main_window.videoSeekSlider.value())

    def cancel(self):
        self._cancel_event.set()

    def run(self):
        try:
            if self._cancel_event.is_set():
                self.cancelled.emit()
                return

            if self._cancel_event.is_set():
                self.cancelled.emit()
                return

            start_time = time.monotonic()

            def progress_with_fps(
                processed: int, total: int, frame_number: int
            ) -> None:
                elapsed = time.monotonic() - start_time
                scan_fps = (processed / elapsed) if elapsed > 0 else 0.0
                self.progress.emit(processed, total, frame_number, scan_fps)

            def issue_found_callback(face_id: str, frame_number: int) -> None:
                self.issue_found.emit(str(face_id), int(frame_number))

            result = self.main_window.video_processor.scan_issue_frames(
                progress_callback=progress_with_fps,
                issue_found_callback=issue_found_callback,
                is_cancelled=self._cancel_event.is_set,
                scan_ranges=self._scan_ranges,
                base_control=self._base_control,
                base_params=self._base_params,
                target_faces_snapshot=self._target_faces_snapshot,
                control_defaults_snapshot=self._control_defaults_snapshot,
                reset_frame_number=self._reset_frame_number,
            )
            if result is None:
                self.cancelled.emit()
                return
            elapsed_seconds = time.monotonic() - start_time
            self.completed.emit(
                result["issue_frames_by_face"],
                result["frames_scanned"],
                result["faces_with_issues"],
                self._scan_scope_text,
                elapsed_seconds,
                bool(result.get("cancelled", False)),
            )
        except Exception as exc:
            print(f"[ERROR] IssueScanWorker Failed to run: {exc}")
            traceback.print_exc()
            self.failed.emit(str(exc))


class InputFacesLoaderWorker(qtc.QThread):
    # Define signals to emit when loading is done or if there are updates - Changed to QImage
    thumbnail_ready = qtc.Signal(str, numpy.ndarray, object, QImage, str)
    finished = qtc.Signal()  # Signal to indicate completion

    def __init__(
        self,
        main_window: "MainWindow",
        media_path=False,
        folder_name=False,
        files_list=None,
        face_ids=None,
        parent=None,
    ):
        super().__init__(parent)
        self.main_window = main_window
        self.folder_name = folder_name
        self.files_list = files_list or []
        self.face_ids = face_ids or []
        self._running = True  # Flag to control the running state

        # SNAPSHOT : get parameters in main thread before run()
        self.control_snapshot = (
            main_window.control.copy() if getattr(main_window, "control", None) else {}
        )

    def run(self):
        """
        Main worker thread execution. Loads models first, then processes files.
        """
        try:
            # Proceed with file processing now that models are ready.
            if self.folder_name or self.files_list:
                self.main_window.placeholder_update_signal.emit(
                    self.main_window.inputFacesList, True
                )
                self.load_faces(self.folder_name, self.files_list)
                self.main_window.placeholder_update_signal.emit(
                    self.main_window.inputFacesList, False
                )
        except Exception as e:
            print(f"[ERROR] Error in InputFacesLoaderWorker: {e}")
            traceback.print_exc()
        finally:
            self.finished.emit()

    def load_faces(
        self, folder_name: bool | str = False, files_list: Optional[List[str]] = None
    ) -> None:
        # Use the snapshot - thread-safe
        control = self.control_snapshot
        files_list = files_list or []

        # OPTIMIZED: Pair the file paths with their correct IDs before any processing
        # This prevents ID shifting if an image fails, and avoids destructive sorting.
        paired_files_ids: List[Tuple[str, str]] = []

        if folder_name and isinstance(folder_name, str):
            image_files = misc_helpers.get_image_files(
                folder_name,
                bool(control.get("InputFacesFolderRecursiveToggle", False)),
            )
            image_files.sort()  # Safe to sort here, IDs are generated fresh
            for path in image_files:
                paired_files_ids.append(
                    (os.path.join(folder_name, path), str(uuid.uuid1().int))
                )
        elif files_list:
            # DO NOT SORT if loading from a workspace, keep original saved order
            for idx, path in enumerate(files_list):
                f_id = self.face_ids[idx] if self.face_ids else str(uuid.uuid1().int)
                paired_files_ids.append((path, f_id))

        def load_image_task(
            image_path: str, f_id: str
        ) -> Tuple[str, str, Optional[numpy.ndarray]]:
            """Background CPU task to read and prepare image bytes."""
            if not misc_helpers.is_image_file(image_path):
                return image_path, f_id, None
            frame = misc_helpers.read_image_file(image_path)
            if frame is None:
                return image_path, f_id, None
            # Swap channels from BGR to RGB concurrently
            return image_path, f_id, frame[..., ::-1]

        max_workers = min(16, (os.cpu_count() or 1) + 4)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for image_file_path, face_id, frame in executor.map(
                lambda p: load_image_task(*p), paired_files_ids
            ):
                if not self._running:  # Check if the thread is still running
                    executor.shutdown(wait=False, cancel_futures=True)
                    break

                if frame is None:
                    continue

                # WORKER SAFETY: Wrap the entire GPU image processing in a try/except block.
                # All CUDA tensor operations MUST execute here synchronously to prevent
                # Context errors or GPU race conditions.
                try:
                    img = torch.from_numpy(frame.astype("uint8")).to(
                        self.main_window.models_processor.device
                    )
                    img = img.permute(2, 0, 1)

                    _, kpss_5, _ = self.main_window.function_worker.run_detect(
                        img,
                        str(control.get("DetectorModelSelection", "RetinaFace")),
                        max_num=1,
                        score=float(control.get("DetectorScoreSlider", 50)) / 100.0,
                        input_size=(512, 512),
                        use_landmark_detection=bool(
                            control.get("LandmarkDetectToggle", False)
                        ),
                        landmark_detect_mode=str(
                            control.get("LandmarkDetectModelSelection", "203")
                        ),
                        landmark_score=float(
                            control.get("LandmarkDetectScoreSlider", 50)
                        )
                        / 100.0,
                        from_points=bool(control.get("DetectFromPointsToggle", False)),
                        rotation_angles=[0]
                        if not bool(control.get("AutoRotationToggle", False))
                        else [0, 90, 180, 270],
                    )

                    if kpss_5 is None or len(kpss_5) == 0:
                        continue

                    face_kps = kpss_5[0]
                    if face_kps.any():
                        # Calculate embedding ONLY for the selected recognition model
                        selected_recognition_model = str(
                            control.get(
                                "RecognitionModelSelection", "Inswapper128ArcFace"
                            )
                        )
                        similarity_type = str("Auto")
                        face_emb, cropped_img = (
                            self.main_window.function_worker.run_recognize_direct(
                                img,
                                face_kps,
                                similarity_type,
                                selected_recognition_model,  # Use selected model
                            )
                        )

                        if face_emb is None:  # Check if recognition failed
                            continue

                        cropped_img_np = cropped_img.cpu().numpy()
                        # Swap channels from RGB to BGR for pixmap creation
                        face_img = numpy.ascontiguousarray(cropped_img_np[..., ::-1])

                        # QIMAGE THREAD-SAFE
                        height, width, channel = face_img.shape
                        bytes_per_line = 3 * width
                        q_image = QImage(
                            face_img.data,
                            width,
                            height,
                            bytes_per_line,
                            QImage.Format_BGR888,
                        ).copy()

                        embedding_store: Dict[str, numpy.ndarray] = {
                            selected_recognition_model: face_emb,
                            "kps_5": face_kps,
                        }

                        self.thumbnail_ready.emit(
                            image_file_path, face_img, embedding_store, q_image, face_id
                        )

                except Exception as e:
                    print(
                        f"[ERROR] InputFacesLoaderWorker: Failed to process {image_file_path}. Reason: {e}"
                    )
                    continue  # Skip this specific corrupt image and continue the loop

    def stop(self):
        # Stop the thread by setting the running flag to False.
        self._running = False
        self.quit()
        self.wait(1000)
        if self.isRunning():
            self.terminate()


class FilterWorker(qtc.QThread):
    filtered_results = qtc.Signal(list, int)  # (visible_indices, snapshot_size)

    def __init__(
        self, main_window: "MainWindow", search_text="", filter_list="target_videos"
    ):
        super().__init__()
        self.main_window = main_window
        self.search_text = search_text
        self.filter_list = filter_list
        # Snapshot attributes set by filter_actions before start() is called.
        # Initialised to safe empty defaults so the worker never accesses Qt widgets.
        self.items_snapshot: list = []
        self.include_file_types: list = []
        self.min_image_size: tuple[int, int] = (0, 0)
        self.filter_list_widget = self.get_list_widget()
        self.filtered_results.connect(
            partial(
                filter_actions.update_filtered_list,
                main_window,
                self.filter_list_widget,
            )
        )

    def get_list_widget(self):
        list_widget = False
        if self.filter_list == "target_videos":
            list_widget = self.main_window.targetVideosList
        elif self.filter_list == "input_faces":
            list_widget = self.main_window.inputFacesList
        elif self.filter_list == "merged_embeddings":
            list_widget = self.main_window.inputEmbeddingsList
        return list_widget

    def run(self):
        if self.filter_list == "target_videos":
            self.filter_target_videos()
        elif self.filter_list == "input_faces":
            self.filter_input_faces()
        elif self.filter_list == "merged_embeddings":
            self.filter_merged_embeddings()

    def filter_target_videos(self):
        # Operates only on pre-captured plain Python data — no Qt widget access.
        search_text = self.search_text
        include_file_types = self.include_file_types

        min_width, min_height = self.min_image_size

        visible_indices = []
        for index, media_path, file_type, width, height in self.items_snapshot:
            if search_text and search_text not in media_path.lower():
                continue
            if file_type not in include_file_types:
                continue
            # Items whose dimensions are unknown (webcams) are never filtered out
            # by the size sliders.
            if width and height and (width < min_width or height < min_height):
                continue
            visible_indices.append(index)

        self.filtered_results.emit(visible_indices, len(self.items_snapshot))

    def filter_input_faces(self):
        # Operates only on pre-captured plain Python data — no Qt widget access.
        search_text = self.search_text

        visible_indices = []
        for index, media_path in self.items_snapshot:
            if not search_text or search_text in media_path.lower():
                visible_indices.append(index)

        self.filtered_results.emit(visible_indices, len(self.items_snapshot))

    def filter_merged_embeddings(self):
        # Operates only on pre-captured plain Python data — no Qt widget access.
        search_text = self.search_text

        visible_indices = []
        for index, embedding_name in self.items_snapshot:
            if not search_text or search_text in embedding_name.lower():
                visible_indices.append(index)

        self.filtered_results.emit(visible_indices, len(self.items_snapshot))

    def stop_thread(self):
        self.quit()
        self.wait()
