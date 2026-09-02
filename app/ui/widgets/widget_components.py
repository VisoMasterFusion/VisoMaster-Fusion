# pylint: disable=keyword-arg-before-vararg
import os
import traceback
from functools import partial
import uuid
from typing import TYPE_CHECKING, Any, Dict, List
from send2trash import send2trash
import subprocess
import sys
import time

from PySide6 import QtWidgets, QtGui, QtCore
from PySide6.QtWidgets import QPushButton
import cv2
import numpy as np
import torch
import gc

import app.ui.widgets.actions.common_actions as common_widget_actions
from app.ui.widgets.actions import video_control_actions
from app.ui.widgets.actions import graphics_view_actions
from app.ui.widgets.actions import card_actions
from app.ui.widgets.actions import save_load_actions
import app.helpers.miscellaneous as misc_helpers
from app.helpers.miscellaneous import get_video_rotation

if TYPE_CHECKING:
    from app.ui.main_ui import MainWindow


class TwoLineElidedLabel(QtWidgets.QLabel):
    def __init__(self, text: str = "", parent=None):
        super().__init__(parent)
        self._full_text = ""
        self.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignHCenter | QtCore.Qt.AlignmentFlag.AlignTop
        )
        self.setWordWrap(False)
        self.setText(text)

    def setText(self, text: str) -> None:
        self._full_text = text
        self._update_display_text()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._update_display_text()

    def _fit_text_prefix(self, text: str, max_width: int) -> str:
        if not text or max_width <= 0:
            return ""

        font_metrics = self.fontMetrics()
        if font_metrics.horizontalAdvance(text) <= max_width:
            return text

        fitted_text = ""
        last_space_index = -1
        for index, char in enumerate(text):
            candidate = fitted_text + char
            if font_metrics.horizontalAdvance(candidate) > max_width:
                break
            fitted_text = candidate
            if char.isspace():
                last_space_index = index

        if last_space_index > 0:
            return text[:last_space_index].rstrip()
        return fitted_text.rstrip()

    def _update_display_text(self) -> None:
        if not self._full_text:
            super().setText("")
            return

        max_width = self.contentsRect().width()
        if max_width <= 0:
            super().setText(self._full_text)
            return

        font_metrics = self.fontMetrics()
        first_line = self._fit_text_prefix(self._full_text, max_width)

        if not first_line or first_line == self._full_text:
            display_text = font_metrics.elidedText(
                self._full_text, QtCore.Qt.TextElideMode.ElideRight, max_width * 2
            )
            super().setText(display_text)
            return

        remaining_text = self._full_text[len(first_line) :].lstrip()
        second_line = font_metrics.elidedText(
            remaining_text, QtCore.Qt.TextElideMode.ElideRight, max_width
        )
        super().setText(f"{first_line}\n{second_line}")


class CardButton(QPushButton):
    def __init__(self, *args, **kwargs):
        self.main_window = kwargs.pop("main_window", None)
        super().__init__(*args, **kwargs)
        self.setFocusPolicy(QtCore.Qt.NoFocus)
        self.setAutoDefault(False)
        self.setDefault(False)
        self.list_item = None
        self.list_widget = None
        self.popMenu = None

    def _release_context_menu(self, *action_attrs: str) -> None:
        for attr in action_attrs:
            action = getattr(self, attr, None)
            if action is not None:
                try:
                    action.deleteLater()
                except RuntimeError:
                    pass
                setattr(self, attr, None)
        if self.popMenu is not None:
            try:
                self.popMenu.deleteLater()
            except RuntimeError:
                pass
            self.popMenu = None

    def _exec_context_menu(self, point) -> None:
        if self.popMenu is not None:
            self.popMenu.exec_(self.mapToGlobal(point))

    def _restore_pre_click_checked_state(self):
        self.blockSignals(True)
        self.setChecked(not self.isChecked())
        self.blockSignals(False)

    def get_item_position(self):
        if self.list_widget is None or self.list_item is None:
            return None
        # row() is O(1) and stays correct after the list is re-sorted, unlike
        # scanning for the item whose widget is self.
        row = self.list_widget.row(self.list_item)
        return row if row >= 0 else None

    # To find the index of second last selected button by traversing the list
    # Mainly used as a helper for Shift Selection of CardButtons
    def get_index_of_second_last_selected_item(self):
        total_items_count = self.list_widget.count()
        if total_items_count < 2:
            return None
        selected_count = 0
        for i in range(self.list_widget.count() - 1, -1, -1):
            list_item = self.list_widget.item(i)
            card_button: CardButton = list_item.listWidget().itemWidget(list_item)
            if card_button.isChecked():
                selected_count += 1
                if selected_count == 2:
                    return i
        return None

    # To find all the selected buttons behind 'item_index' (Only those which are sequentially selected)
    # Mainly used as a helper for Shift Selection of CardButtons
    def get_sequential_trailing_selected_items(
        self, item_index
    ) -> list[tuple[int, QPushButton]]:
        selected_items = []
        for i in range(item_index - 1, -1, -1):
            list_item = self.list_widget.item(i)
            card_button: CardButton = list_item.listWidget().itemWidget(list_item)
            if card_button.isChecked():
                selected_items.append((i, card_button))
            else:
                break
        return selected_items

    def deselect_all_trailing_items(self, item_index):
        for i in range(item_index - 1, -1, -1):
            list_item = self.list_widget.item(i)
            card_button: CardButton = list_item.listWidget().itemWidget(list_item)
            card_button.blockSignals(True)
            card_button.setChecked(False)
            card_button.blockSignals(False)

    def select_all_items_between_range(
        self, lower_range, upper_range
    ) -> list[QPushButton]:
        card_buttons = []
        # Include items in the lower_range and upper_range indexes too
        for i in range(lower_range, upper_range + 1):
            list_item = self.list_widget.item(i)
            card_button: CardButton = list_item.listWidget().itemWidget(list_item)
            card_button.blockSignals(True)
            card_button.setChecked(True)
            card_button.blockSignals(False)
            card_buttons.append(card_button)
        return card_buttons


class TargetMediaCardButton(CardButton):
    def __init__(
        self,
        media_path: str,
        file_type: str,
        media_id: str,
        is_webcam=False,
        webcam_index=-1,
        webcam_backend=-1,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.media_id = media_id
        self.file_type = file_type
        self.media_path = os.path.normpath(media_path)
        self.is_webcam = is_webcam
        self.webcam_index = webcam_index
        self.webcam_backend = webcam_backend
        self.media_capture: cv2.VideoCapture | bool = False
        self._thumbnail_pixmap = QtGui.QPixmap()
        self.setCheckable(True)
        self.setText("")
        self.setToolTip(media_path)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(3, 3, 3, 2)
        layout.setSpacing(0)

        self.thumbnail_label = QtWidgets.QLabel(self)
        self.thumbnail_label.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignHCenter | QtCore.Qt.AlignmentFlag.AlignBottom
        )
        self.thumbnail_label.setAttribute(
            QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, True
        )
        layout.addWidget(self.thumbnail_label, 1)

        filename = os.path.basename(media_path)
        self.text_label = TwoLineElidedLabel(filename, self)
        title_font = self.text_label.font()
        title_font.setPointSize(9)
        self.text_label.setFont(title_font)
        line_spacing = QtGui.QFontMetrics(title_font).lineSpacing()
        self.text_label.setFixedHeight((line_spacing * 2) + 4)
        self.text_label.setAttribute(
            QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, True
        )
        layout.addWidget(self.text_label, 0)

        self.clicked.connect(self.load_media)

        # Set the context menu policy to trigger the custom context menu on right-click
        self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        # Connect the custom context menu request signal to the custom slot
        self.customContextMenuRequested.connect(self.on_context_menu)

    def set_thumbnail_pixmap(self, pixmap: QtGui.QPixmap) -> None:
        self._thumbnail_pixmap = pixmap
        self._update_thumbnail_pixmap()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._update_thumbnail_pixmap()

    def _update_thumbnail_pixmap(self) -> None:
        if self._thumbnail_pixmap.isNull():
            self.thumbnail_label.clear()
            return

        target_size = self.thumbnail_label.contentsRect().size()
        if not target_size.isValid():
            return

        scaled_pixmap = self._thumbnail_pixmap.scaled(
            target_size,
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )
        self.thumbnail_label.setPixmap(scaled_pixmap)

    def reset_media_state(self):
        main_window = self.main_window
        # Deselect the currently selected video
        if main_window.selected_video_button:
            main_window.selected_video_button.toggle()  # Deselect the previous video
            main_window.selected_video_button = False

        # Stop the current video processing
        main_window.video_processor.stop_processing()

    def reset_related_widgets_and_values(self):
        main_window = self.main_window

        # Set up videoSeekLineEdit
        video_control_actions.set_up_video_seek_line_edit(main_window)
        # Clear current target faces
        card_actions.clear_target_faces(main_window, refresh_frame=False)
        # Check if the user wants to keep input faces/embeddings selected
        # Keep Inputs checked if KeepInput, AutoSwap or Batch is active
        if not (
            main_window.control.get("KeepInputToggle", False)
            or getattr(main_window, "is_batch_processing", False)
            or main_window.control.get("AutoSwapToggle", False)
        ):
            # Default behavior: Uncheck input faces
            card_actions.uncheck_all_input_faces(main_window)
            # Default behavior: Uncheck merged embeddings
            card_actions.uncheck_all_merged_embeddings(main_window)
        # Remove all markers
        video_control_actions.remove_all_markers(main_window)

        main_window.cur_selected_target_face_button = False

        # Reset buttons and slider
        video_control_actions.reset_media_buttons(main_window)

    def _toggle_timeline_visibility(
        self, main_window: "MainWindow", is_visible: bool
    ) -> None:
        """Helper to show/hide timeline specific controls based on media type."""
        if (
            hasattr(main_window, "timelineScrollArea")
            and main_window.timelineScrollArea
        ):
            main_window.timelineScrollArea.setVisible(is_visible)
        if hasattr(main_window, "videoSeekLineEdit") and main_window.videoSeekLineEdit:
            main_window.videoSeekLineEdit.setVisible(is_visible)
        if hasattr(main_window, "videoTimeLineEdit") and main_window.videoTimeLineEdit:
            main_window.videoTimeLineEdit.setVisible(is_visible)
        if hasattr(main_window, "videoFpsLineEdit") and main_window.videoFpsLineEdit:
            main_window.videoFpsLineEdit.setVisible(is_visible)
        if hasattr(main_window, "zoomLabel") and main_window.zoomLabel:
            main_window.zoomLabel.setVisible(is_visible)
        if (
            hasattr(main_window, "timelineZoomSlider")
            and main_window.timelineZoomSlider
        ):
            main_window.timelineZoomSlider.setVisible(is_visible)

    def load_media(self):
        main_window = self.main_window
        if video_control_actions.block_if_issue_scan_active(
            main_window, "change target media"
        ):
            self._restore_pre_click_checked_state()
            return

        # Deselect the currently selected video
        if main_window.selected_video_button:
            main_window.selected_video_button.toggle()  # Deselect the previous video
            main_window.selected_video_button = False

        # Stop the current video processing
        main_window.video_processor.stop_processing()
        main_window.video_processor._clear_single_frame_preview_caches()

        if main_window.selected_target_face_id:
            main_window.current_widget_parameters = main_window.parameters[
                main_window.selected_target_face_id
            ].copy()

        # Reset the frame counter
        main_window.video_processor.current_frame_number = 0
        main_window.video_processor.media_path = self.media_path
        main_window.parameters = {}
        main_window.selected_target_face_id = None
        main_window.video_processor.current_frame = []

        # Release the previous media_capture if it exists
        if main_window.video_processor.media_capture:
            main_window.video_processor.media_capture.release()

        frame = None
        max_frames_number = 0  # Initialize max_frames_number for either video or image
        rotation_angle = 0  # MODIFICATION: Added rotation variable

        if self.file_type == "video":
            # Get video rotation metadata before loading
            rotation_angle = get_video_rotation(self.media_path)
            # Check for Variable Frame Rate (VFR) and warn the user
            vfr_start = time.perf_counter()
            misc_helpers.check_and_warn_vfr(self.media_path)
            print(
                f"[INFO] Media Load [1/3]: Variable Frame Rate (VFR) validation completed in "
                f"{time.perf_counter() - vfr_start:.3f}s for '{self.media_path}'."
            )
            main_window.video_processor.media_rotation = rotation_angle
            open_start = time.perf_counter()
            media_capture = cv2.VideoCapture(self.media_path)
            print(
                f"[INFO] Media Load [2/3]: OpenCV VideoCapture initialization completed in "
                f"{time.perf_counter() - open_start:.3f}s for '{self.media_path}'."
            )
            # Explicitly enable OpenCV's auto-rotation to let it handle metadata natively
            if hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO"):
                media_capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
            if not media_capture.isOpened():
                print(f"[ERROR] Error opening video {self.media_path}")
                return  # If the video cannot be opened, exit the function

            media_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            max_frames_number = int(media_capture.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
            read_start = time.perf_counter()
            _, frame = misc_helpers.read_frame(media_capture, rotation_angle)
            print(
                f"[INFO] Media Load [3/3]: First frame decoded and read into memory in "
                f"{time.perf_counter() - read_start:.3f}s for '{self.media_path}'."
            )
            main_window.video_processor.media_capture = media_capture
            self.media_capture = media_capture
            main_window.video_processor.fps = media_capture.get(cv2.CAP_PROP_FPS)
            main_window.video_processor.max_frame_number = max_frames_number
            main_window.video_processor.current_frame_number = 0
            main_window.video_processor.next_frame_to_display = 0

        elif self.file_type == "image":
            frame = misc_helpers.read_image_file(self.media_path)
            max_frames_number = 0  # For an image, there is only one "frame"
            main_window.video_processor.max_frame_number = max_frames_number

        elif self.file_type == "webcam":
            # MODIFICATION: Set rotation to 0 for webcam
            main_window.video_processor.media_rotation = 0
            res_width, res_height = self.main_window.control[
                "WebcamMaxResSelection"
            ].split("x")

            media_capture = cv2.VideoCapture(self.webcam_index, self.webcam_backend)

            try:
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                media_capture.set(cv2.CAP_PROP_FOURCC, fourcc)
            except Exception:
                pass

            media_capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(res_width))
            media_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(res_height))
            max_frames_number = 999999
            _, frame = misc_helpers.read_frame(media_capture, 0)  # 0 for webcam
            main_window.video_processor.media_capture = media_capture
            self.media_capture = media_capture
            main_window.video_processor.fps = media_capture.get(cv2.CAP_PROP_FPS)
            main_window.video_processor.max_frame_number = max_frames_number

        if frame is not None:
            main_window.scene.clear()
            if self.file_type == "video":
                # restore initial video position after reading. == 0
                media_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

            main_window.video_processor.current_frame = frame
            pixmap = common_widget_actions.get_pixmap_from_frame(main_window, frame)
            graphics_view_actions.update_graphics_view(
                main_window, pixmap, 0, reset_fit=True
            )

        self.reset_related_widgets_and_values()

        main_window.video_processor.file_type = self.file_type
        main_window.videoSeekSlider.blockSignals(
            True
        )  # Block signals to prevent unnecessary updates
        main_window.videoSeekSlider.setMaximum(max_frames_number)
        main_window.videoSeekSlider.setValue(0)  # Set the slider to 0 for the new video
        main_window.videoSeekSlider.blockSignals(False)  # Unblock signals

        # --- FORCE TIMELINE & THUMBNAILS TO REFRESH ---
        main_window.videoSeekSlider.rangeChanged.emit(0, max_frames_number)
        if hasattr(main_window, "timelineContainer"):
            main_window.timelineContainer.thumbnail_track.request_thumbnails()

        # Toggle UI elements visibility based on file type
        self._toggle_timeline_visibility(
            main_window, self.file_type in ["video", "webcam"]
        )

        # Append the selected video button to the list
        main_window.selected_video_button = self

        # Update the graphics frame after the reset
        main_window.graphicsViewFrame.update()

        # Set Parameter widget values to default
        common_widget_actions.set_widgets_values_using_face_id_parameters(
            main_window=main_window, face_id=None
        )

        main_window.loading_new_media = True
        common_widget_actions.refresh_frame(main_window, synchronous=True)

        if main_window.control.get("AutoSwapToggle"):
            # Run detect on 0 frame or image
            card_actions.find_target_faces(main_window)
            if main_window.target_faces and not main_window.selected_target_face_id:
                list(main_window.target_faces.values())[0].click()
            common_widget_actions.refresh_frame(main_window)

            from app.ui.widgets.actions import layout_actions

            layout_actions.fit_image_to_view_onchange(main_window)

        if (
            main_window.control["SendVirtCamFramesEnableToggle"]
            and self.file_type != "image"
        ):
            # Re-initialize virtualcam to reset its dimensions with that of the new video
            main_window.video_processor.enable_virtualcam()

        # --- AUTO-START WEBCAM FEED ---
        # A webcam is a live sensor, so it should stream immediately without waiting for "Play".
        if self.file_type == "webcam":
            main_window.buttonMediaPlay.blockSignals(True)
            main_window.buttonMediaPlay.setChecked(True)
            main_window.buttonMediaPlay.blockSignals(False)
            video_control_actions.set_play_button_icon_to_stop(main_window)

            # Start the live stream immediately
            main_window.video_processor.process_webcam()

        # list_view_actions.find_target_faces(main_window)

    def deselect_currently_selected_video(self, main_window):
        if video_control_actions.block_if_issue_scan_active(
            main_window, "deselect the current media"
        ):
            return

        # Deselect the currently selected video
        if main_window.selected_video_button == self:
            self.reset_media_state()

            # Reset the frame counter / media state
            main_window.video_processor.current_frame_number = 0
            main_window.video_processor.media_path = None
            main_window.parameters = {}
            main_window.selected_target_face_id = None

            if main_window.video_processor.media_capture:
                try:
                    main_window.video_processor.media_capture.release()
                except Exception:
                    pass
            main_window.video_processor.media_capture = None
            main_window.video_processor.current_frame = None
            main_window.video_processor.fps = 0
            main_window.video_processor.max_frame_number = 0
            main_window.video_processor.file_type = None
            main_window.video_processor._clear_single_frame_preview_caches()

            self.main_window.scene.clear()
            self.reset_related_widgets_and_values()

            main_window.videoSeekSlider.blockSignals(True)
            main_window.videoSeekSlider.setMaximum(1)
            main_window.videoSeekSlider.setValue(0)
            main_window.videoSeekSlider.blockSignals(False)

            main_window.videoSeekSlider.rangeChanged.emit(0, 1)
            if hasattr(main_window, "timelineContainer"):
                main_window.timelineContainer.thumbnail_track.request_thumbnails()

            self._toggle_timeline_visibility(main_window, False)

            main_window.selected_video_button = None
            main_window.graphicsViewFrame.update()

            if self.media_capture:
                try:
                    self.media_capture.release()
                except Exception:
                    pass
                self.media_capture = None

            # Free media-related GPU memory only (keep models loaded)
            if not getattr(main_window, "is_batch_processing", False):
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        i = self.get_item_position()
        if i is not None:
            main_window.targetVideosList.takeItem(i)
        main_window.target_videos.pop(self.media_id, None)

        if not main_window.target_videos:
            main_window.placeholder_update_signal.emit(
                self.main_window.targetVideosList, False
            )

    def remove_target_media_from_list(self):
        main_window = self.main_window
        if video_control_actions.block_if_issue_scan_active(
            main_window, "remove target media"
        ):
            return
        self.deselect_currently_selected_video(main_window)
        self.deleteLater()

    def delete_target_media_to_trash(self):
        main_window = self.main_window
        if video_control_actions.block_if_issue_scan_active(
            main_window, "delete target media"
        ):
            return
        self.deselect_currently_selected_video(main_window)

        # Send the file to the trash
        if os.path.exists(self.media_path):
            send2trash(self.media_path)
            print(f"[INFO] {self.media_path} has been sent to the trash.")
        else:
            print(f"[ERROR] {self.media_path} does not exist.")

        self.deleteLater()

    def open_target_path_by_explorer(self):
        if os.path.exists(self.media_path):
            # Normalize path
            normalized_path = os.path.normpath(os.path.abspath(self.media_path))
            print(normalized_path)
            if sys.platform == "win32":
                # Windows - use full path to explorer.exe to avoid PATH issues
                try:
                    # Method 1: Using subprocess without shell (more secure and reliable)
                    subprocess.Popen(f'explorer /select,"{normalized_path}"')
                except FileNotFoundError:
                    # Fallback: Use full path to explorer.exe
                    subprocess.Popen(
                        f'C:\\Windows\\explorer.exe /select,"{normalized_path}"'
                    )
            elif sys.platform == "darwin":
                # macOS
                subprocess.run(["open", "-R", self.media_path])
            else:
                # Linux
                directory = os.path.dirname(os.path.abspath(self.media_path))
                subprocess.run(["xdg-open", directory])

    def create_context_menu(self):
        from app.ui.widgets.actions import list_view_actions

        self.popMenu = QtWidgets.QMenu(self)
        self.remove_action = QtGui.QAction("Remove from list", self)
        self.remove_action.triggered.connect(self.remove_target_media_from_list)
        self.popMenu.addAction(self.remove_action)

        self.delete_action = QtGui.QAction("Delete file to recycle bin", self)
        self.delete_action.triggered.connect(self.delete_target_media_to_trash)
        self.popMenu.addAction(self.delete_action)

        self.open_path_action = QtGui.QAction("Open file location", self)
        self.open_path_action.triggered.connect(self.open_target_path_by_explorer)
        self.popMenu.addAction(self.open_path_action)
        self.popMenu.addSeparator()

        self.clear_all_media_action = QtGui.QAction("Clear All Media", self)
        self.clear_all_media_action.triggered.connect(
            partial(list_view_actions.clear_all_target_media, self.main_window)
        )
        self.popMenu.addAction(self.clear_all_media_action)

        self.delete_all_files_action = QtGui.QAction("⚠ Delete all files to recycle bin", self)
        self.delete_all_files_action.triggered.connect(
            partial(list_view_actions.delete_all_target_media_to_trash, self.main_window)
        )
        self.popMenu.addAction(self.delete_all_files_action)

    def on_context_menu(self, point):
        self.create_context_menu()
        try:
            scan_active = video_control_actions.is_issue_scan_active(self.main_window)
            self.remove_action.setEnabled(not scan_active)
            self.delete_action.setEnabled(not scan_active)
            self.clear_all_media_action.setEnabled(
                bool(self.main_window.target_videos) and not scan_active
            )
            has_files = any(
                not getattr(b, "is_webcam", False)
                and getattr(b, "media_path", None)
                and os.path.exists(b.media_path)
                for b in (self.main_window.target_videos or {}).values()
            )
            self.delete_all_files_action.setEnabled(has_files and not scan_active)
            self._exec_context_menu(point)
        finally:
            self._release_context_menu(
                "remove_action",
                "delete_action",
                "open_path_action",
                "clear_all_media_action",
                "delete_all_files_action",
            )


class TargetFaceCardButton(CardButton):
    def __init__(
        self,
        media_path,
        cropped_face,
        embedding_store: Dict[str, np.ndarray],
        face_id: str,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # if self.main_window.target_faces:
        #     self.face_id = max([target_face.face_id for target_face in self.main_window.target_faces]) + 1
        # else:
        #     self.face_id = 0
        self.face_id = face_id
        self.media_path = media_path
        self.cropped_face = cropped_face

        self.embedding_store = (
            embedding_store  # Key: embedding_swap_model, Value: embedding
        )

        self.assigned_input_faces: Dict[
            str, Dict[str, np.ndarray]
        ] = {}  # Inside Dict (key - input face_id): {Key: embedding_swap_model, Value: InputFaceCardButton.embedding_store}
        self.assigned_merged_embeddings: Dict[
            str, Dict[str, np.ndarray]
        ] = {}  # Key: embedding_swap_model, Value: EmbeddingCardButton.embedding_store
        self.assigned_input_embedding: Dict[
            str, np.ndarray
        ] = {}  # Key: embedding_swap_model, Value: np.ndarray
        self.assigned_kv_map: Dict | None = None

        # Face re-aging: aged versions of embedding/KV map (populated by Apply button)
        self.aged_input_embedding: Dict[str, np.ndarray] = {}
        self.aged_kv_map: Dict | None = None

        # Auto-mouth expression: per-face EMA state
        from app.processors.mouth_openness import MouthOpennessState

        self.mouth_openness_state: MouthOpennessState = MouthOpennessState()

        self.setCheckable(True)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(1)

        self.display_label = QtWidgets.QLabel("", self)
        self.display_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignBottom)
        title_font = self.display_label.font()
        title_font.setPointSize(9)
        title_font.setBold(True)
        self.display_label.setFont(title_font)
        self.display_label.setAttribute(
            QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, True
        )
        layout.addStretch(1)
        layout.addWidget(self.display_label)
        self.clicked.connect(self.load_target_face)

        # Set the context menu policy to trigger the custom context menu on right-click
        self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        # Connect the custom context menu request signal to the custom slot
        self.customContextMenuRequested.connect(self.on_context_menu)

        # Create parameter dict for the target
        if not self.main_window.parameters.get(self.face_id):
            common_widget_actions.create_parameter_dict_for_face_id(
                self.main_window, self.face_id
            )

    def set_embedding(self, embedding_swap_model: str, embedding: np.ndarray):
        self.embedding_store[embedding_swap_model] = embedding

    def get_embedding(self, embedding_swap_model: str) -> np.ndarray:
        """
        Retrieves the embedding for the specified model.
        If the embedding for the requested model is not already stored,
        it calculates, stores, and returns it on the fly.
        """
        # Check if the embedding for the requested model is already in the store
        stored_embedding = self.embedding_store.get(embedding_swap_model)
        if stored_embedding is not None and stored_embedding.size > 0:
            return stored_embedding
        else:
            # Embedding not found or empty, calculate it now
            print(
                f"[INFO] TargetFaceCardButton {self.face_id}: Calculating missing embedding for model '{embedding_swap_model}'..."
            )
            if self.cropped_face is None or self.cropped_face.size == 0:
                print(
                    f"[ERROR] TargetFaceCardButton {self.face_id}: Cannot calculate embedding, cropped_face is missing or empty."
                )
                return np.array([])  # Return empty array on error

            try:
                # Prepare the cropped face image for run_recognize_direct
                # cropped_face is stored as numpy BGR uint8
                # run_recognize_direct expects tensor CHW RGB (uint8 or float)

                # 1. Convert BGR numpy to RGB numpy
                face_img_rgb_np = self.cropped_face[..., ::-1]

                # 2. Convert RGB numpy to RGB tensor CHW
                face_img_rgb_tensor = torch.from_numpy(
                    np.ascontiguousarray(face_img_rgb_np)
                ).to(self.main_window.models_processor.device)
                face_img_rgb_tensor_chw = face_img_rgb_tensor.permute(2, 0, 1)

                # 3. Need keypoints (kps_5) on the cropped image (112x112 or similar)
                #    We don't have the original kps_5 relative to the *full* image here.
                #    We need to re-detect landmarks *on the cropped face* to get kps_5 for run_recognize_direct.
                #    However, run_recognize_direct itself *also* performs alignment based on kps.
                #    Let's try passing None for kps initially, as run_recognize_direct might handle it internally
                #    by detecting on the aligned 112x112 image it creates.
                #    If that fails, we'd need to run a landmark detector here first.

                # Alternative: Let's use the internal alignment of run_recognize_direct by providing dummy kps
                # that represent the corners/center, assuming the cropped_face is already roughly aligned.
                # We can estimate kps based on the standard template relative to the crop size (e.g., 112x112)
                # For simplicity and given cropped_face might not be exactly 112x112 initially,
                # let's try calling run_recognize directly. It internally aligns to 112x112.
                # We need *some* kps to pass, even if approximate. We'll derive them from the image size.

                h, w, _ = self.cropped_face.shape  # Use the actual shape
                # Approximate 5 keypoints based on typical face proportions within the crop
                # These are rough estimates assuming centered face in the crop
                approx_kps_5 = np.array(
                    [
                        [w * 0.3, h * 0.4],  # Left eye
                        [w * 0.7, h * 0.4],  # Right eye
                        [w * 0.5, h * 0.55],  # Nose
                        [w * 0.35, h * 0.7],  # Left mouth corner
                        [w * 0.65, h * 0.7],  # Right mouth corner
                    ],
                    dtype=np.float32,
                )

                # Get the similarity type from global controls
                similarity_type = str("Auto")

                # Call run_recognize_direct (which expects CHW tensor)
                new_embedding, _ = (
                    self.main_window.function_worker.run_recognize_direct(
                        face_img_rgb_tensor_chw,  # Pass the CHW tensor
                        approx_kps_5,  # Pass the estimated keypoints on the crop
                        similarity_type,
                        embedding_swap_model,  # Use the requested model
                    )
                )

                if new_embedding is not None and new_embedding.size > 0:
                    # Store the newly calculated embedding
                    self.embedding_store[embedding_swap_model] = new_embedding
                    print(
                        f"[INFO] TargetFaceCardButton {self.face_id}: Stored new embedding for '{embedding_swap_model}'."
                    )
                    return new_embedding
                else:
                    print(
                        f"[ERROR] TargetFaceCardButton {self.face_id}: Failed to calculate embedding for '{embedding_swap_model}'."
                    )
                    # Store an empty array to prevent repeated calculation attempts within the same run
                    self.embedding_store[embedding_swap_model] = np.array([])
                    return np.array([])

            except Exception as e:
                print(
                    f"[ERROR] TargetFaceCardButton {self.face_id}: Exception during on-the-fly embedding calculation for '{embedding_swap_model}': {e}"
                )
                traceback.print_exc()
                # Store an empty array to prevent repeated calculation attempts
                self.embedding_store[embedding_swap_model] = np.array([])
                return np.array([])

    def load_target_face(self):
        main_window = self.main_window
        main_window.cur_selected_target_face_button = self
        self.setChecked(True)
        for _, target_face_button in main_window.target_faces.items():
            # Uncheck all other target faces
            if target_face_button != self:
                target_face_button.setChecked(False)

        # Check if KeepInput toggle, or Autoswap or Batch
        if (
            main_window.control.get("KeepInputToggle", False)
            or getattr(main_window, "is_batch_processing", False)
            or main_window.control.get("AutoSwapToggle", False)
        ):
            # KeepInputToggle/Batch/AutoSwap are ON
            # 1. Update assigned faces to correspond on global selection
            self.assigned_input_faces.clear()
            self.assigned_merged_embeddings.clear()

            for input_face_id, input_face_button in main_window.input_faces.items():
                if input_face_button.isChecked():
                    self.assigned_input_faces[input_face_id] = (
                        input_face_button.embedding_store
                    )

            for embedding_id, embed_button in main_window.merged_embeddings.items():
                if embed_button.isChecked():
                    self.assigned_merged_embeddings[embedding_id] = (
                        embed_button.embedding_store
                    )

            # 2. Recalculate assigned embedding (Inputs might have changed)
            self.calculate_assigned_input_embedding()

        else:
            # KeepInputToggle/Batch/AutoSwap are OFF (Default run)
            # 1. Uncheck all inputs/embeddings
            card_actions.uncheck_all_input_faces(main_window)
            card_actions.uncheck_all_merged_embeddings(main_window)

            # 2. Check only inputs/embeddings assigned fpr this face
            for input_face_id in self.assigned_input_faces.keys():
                if main_window.input_faces.get(input_face_id):
                    main_window.input_faces[input_face_id].setChecked(True)
            for embedding_id in self.assigned_merged_embeddings.keys():
                if main_window.merged_embeddings.get(embedding_id):
                    main_window.merged_embeddings[embedding_id].setChecked(True)

        main_window.selected_target_face_id = self.face_id
        main_window.current_kv_tensors_map = self.assigned_kv_map
        video_control_actions.refresh_issue_frames_for_selected_face(main_window)
        video_control_actions.update_scan_review_button_states(main_window)

        common_widget_actions.set_widgets_values_using_face_id_parameters(
            main_window=main_window, face_id=self.face_id
        )

        main_window.current_widget_parameters = main_window.parameters[
            self.face_id
        ].copy()

    def calculate_assigned_input_embedding(self) -> None:
        """
        Gathers and processes selected embeddings and K/V Maps dynamically based on UI state.
        Defers tensor concatenation/averaging to execution time to respect real-time UI toggles.
        """
        control = self.main_window.control.copy()

        all_input_embeddings: list[dict[str, np.ndarray]] = []
        all_embedding_swap_models: set[str] = set()

        # Itera su `assigned_input_faces` e raccogli gli embedding e i modelli
        for _, embedding_store in self.assigned_input_faces.items():
            if embedding_store:  # Verifica se l'embedding_store non è vuoto
                all_embedding_swap_models.update(embedding_store.keys())
                all_input_embeddings.append(embedding_store)  # Aggiungi l'intero store

        # Itera su `assigned_merged_embeddings` e raccogli gli embedding e i modelli
        for _, embedding_store in self.assigned_merged_embeddings.items():
            if embedding_store:  # Verifica se l'embedding_store non è vuoto
                all_embedding_swap_models.update(embedding_store.keys())
                all_input_embeddings.append(embedding_store)  # Aggiungi l'intero store

        # Calcolo degli embedding se presenti
        if len(all_input_embeddings) > 0:
            self.assigned_input_embedding = {}
            for model in all_embedding_swap_models:
                # Gather all embeddings for the current swap model
                embeddings_to_merge = [
                    store[model] for store in all_input_embeddings if model in store
                ]

                # 1. Apply Mean or Median
                if control.get("EmbMergeMethodSelection", "Mean") == "Mean":
                    merged_emb = np.mean(embeddings_to_merge, axis=0)
                elif control.get("EmbMergeMethodSelection", "Mean") == "Median":
                    merged_emb = np.median(embeddings_to_merge, axis=0)
                else:
                    merged_emb = np.mean(embeddings_to_merge, axis=0)  # Fallback

                # 2. Apply L2 Normalization
                norm = np.linalg.norm(merged_emb)
                if norm > 0:
                    merged_emb = merged_emb / norm

                self.assigned_input_embedding[model] = merged_emb

        else:
            self.assigned_input_embedding = {}

        # --- New KV Map Logic ---
        main_window = self.main_window
        control = main_window.control
        denoiser_on = (
            control.get("DenoiserUNetEnableBeforeRestorersToggle", False)
            or control.get("DenoiserAfterFirstRestorerToggle", False)
            or control.get("DenoiserAfterRestorersToggle", False)
        )

        # 1. READ THE NEW TOGGLE
        average_kv_maps = control.get("AverageKVToggle", True)

        self.assigned_kv_map = None
        self.kv_data_color_transferred = False

        if denoiser_on and (
            self.assigned_input_faces or self.assigned_merged_embeddings
        ):
            all_kv_maps = []

            # 1. Embeddings priority
            for embedding_id in self.assigned_merged_embeddings.keys():
                embed_button = main_window.merged_embeddings.get(embedding_id)
                if not embed_button:
                    continue

                # NEW LOGIC: Use `.extend()` to inject the unbaked list of raw tensors
                if hasattr(embed_button, "kv_map_list") and embed_button.kv_map_list:
                    all_kv_maps.extend(embed_button.kv_map_list)

                # BACKWARD COMPATIBILITY: Support old saved embeddings that have a baked map
                elif (
                    hasattr(embed_button, "kv_map")
                    and embed_button.kv_map is not None
                    and len(embed_button.kv_map) > 0
                ):
                    all_kv_maps.append(embed_button.kv_map)

            # 2. Fallback to Input Faces
            if len(all_kv_maps) == 0:
                extracted_new_kv = False
                for input_face_id in self.assigned_input_faces.keys():
                    input_face_button = main_window.input_faces.get(input_face_id)
                    if not input_face_button:
                        continue

                    with main_window.function_worker.denoiser_kv_extraction_lock:
                        if (
                            hasattr(input_face_button, "kv_map")
                            and input_face_button.kv_map is not None
                            and len(input_face_button.kv_map) > 0
                        ):
                            all_kv_maps.append(input_face_button.kv_map)
                        else:
                            print(
                                f"[INFO] Generating K/V map for input face: {input_face_button.media_path}"
                            )
                            try:
                                from PIL import Image

                                cropped_face_np = input_face_button.cropped_face
                                pil_img = Image.fromarray(cropped_face_np[..., ::-1])

                                if pil_img.size != (512, 512):
                                    pil_img = pil_img.resize(
                                        (512, 512), Image.Resampling.LANCZOS
                                    )

                                # Batch processing: Keep the extractor loaded for the duration of the loop
                                kv_map = (
                                    main_window.function_worker.get_kv_map_for_face(
                                        pil_img,
                                        unload_after=False,
                                    )
                                )

                                if kv_map:
                                    input_face_button.kv_map = kv_map
                                    all_kv_maps.append(kv_map)
                                    extracted_new_kv = True
                                    print("[INFO] Generated and cached K/V map.")
                                else:
                                    input_face_button.kv_map = {}
                            except Exception as e:
                                print(f"[ERROR] Error generating K/V map: {e}")
                                import traceback

                                traceback.print_exc()
                                input_face_button.kv_map = {}

                # Cleanup: Unload the extractor once the batch is fully processed
                if extracted_new_kv:
                    main_window.function_worker.unload_denoiser_kv_extractor()

            # 3. Merge all collected KV Maps and strictly enforce VRAM localization
            if all_kv_maps:
                import torch

                target_device = main_window.models_processor.device

                if len(all_kv_maps) == 1:
                    # Make sure even single cached maps are pushed to the GPU
                    self.assigned_kv_map = {}
                    for layer_key, layer_dict in all_kv_maps[0].items():
                        self.assigned_kv_map[layer_key] = {
                            k: v.to(target_device) for k, v in layer_dict.items()
                        }
                else:
                    merge_mode_str = "Averaging" if average_kv_maps else "Concatenating"
                    print(
                        f"[INFO] {merge_mode_str} K/V maps across {len(all_kv_maps)} prioritized sources..."
                    )
                    merged_kv_map: dict[str, dict[str, torch.Tensor]] = {}
                    first_map = all_kv_maps[0]

                    for layer_key, layer_dict in first_map.items():
                        merged_kv_map[layer_key] = {}
                        for kv_key in layer_dict.keys():
                            tensors_to_merge = []
                            for m in all_kv_maps:
                                if layer_key in m and kv_key in m[layer_key]:
                                    tensors_to_merge.append(m[layer_key][kv_key])

                            if tensors_to_merge:
                                # Force all tensors onto the target VRAM device before operations
                                tensors_to_merge = [
                                    t.to(target_device) for t in tensors_to_merge
                                ]

                                # 3. DYNAMICALLY MERGE BASED ON THE TOGGLE
                                if average_kv_maps:
                                    stacked = torch.stack(tensors_to_merge, dim=0)
                                    merged_tensor = torch.mean(stacked, dim=0)
                                else:
                                    # Concatenate along the sequence dimension (-1)
                                    # to expand the attention window
                                    merged_tensor = torch.cat(tensors_to_merge, dim=-1)

                                merged_kv_map[layer_key][kv_key] = merged_tensor

                    self.assigned_kv_map = merged_kv_map
            else:
                self.assigned_kv_map = None

        if getattr(main_window, "selected_target_face_id", None) == getattr(
            self, "face_id", None
        ):
            main_window.current_kv_tensors_map = self.assigned_kv_map

        # Dirty Flag
        if (
            hasattr(self.main_window, "video_processor")
            and self.main_window.video_processor
        ):
            self.main_window.video_processor.ui_state_is_dirty = True

    def create_context_menu(self):
        # create context menu
        from app.ui.widgets.actions import list_view_actions

        self.popMenu = QtWidgets.QMenu(self)
        self.face_header_action = QtGui.QAction(self.get_display_label(), self)
        header_font = self.popMenu.font()
        header_font.setBold(True)
        self.face_header_action.setFont(header_font)
        self.face_header_action.setEnabled(False)
        self.popMenu.addAction(self.face_header_action)
        self.popMenu.addSeparator()
        self.parameters_copy_action = QtGui.QAction("Copy Parameters", self)
        self.parameters_copy_action.triggered.connect(self.copy_parameters)
        self.parameters_paste_action = QtGui.QAction("Paste Parameters", self)
        self.parameters_paste_action.triggered.connect(self.paste_and_apply_parameters)
        self.save_parameters_action = QtGui.QAction(
            "Save Parameters and Settings", self
        )
        self.save_parameters_action.triggered.connect(
            partial(
                save_load_actions.save_current_parameters_and_control,
                self.main_window,
                self.face_id,
            )
        )
        self.load_parameters_action = QtGui.QAction("Load Parameters Only", self)
        self.load_parameters_action.triggered.connect(
            partial(
                save_load_actions.load_parameters_and_settings,
                self.main_window,
                self.face_id,
            )
        )
        self.load_parameters_and_settings_action = QtGui.QAction(
            "Load Parameters and Settings", self
        )
        self.load_parameters_and_settings_action.triggered.connect(
            partial(
                save_load_actions.load_parameters_and_settings,
                self.main_window,
                self.face_id,
                True,
            )
        )
        current_face_size = getattr(
            self.main_window, "face_thumbnail_button_size", None
        )
        self.thumbnail_size_action_group = QtGui.QActionGroup(self.popMenu)
        self.thumbnail_size_action_group.setExclusive(True)

        self.small_thumbnails_action = QtGui.QAction("Small Thumbnails", self)
        self.small_thumbnails_action.setCheckable(True)
        self.small_thumbnails_action.setChecked(current_face_size == (70, 70))
        self.thumbnail_size_action_group.addAction(self.small_thumbnails_action)
        self.small_thumbnails_action.triggered.connect(
            partial(
                list_view_actions.apply_face_thumbnail_size, self.main_window, (70, 70)
            )
        )
        self.large_thumbnails_action = QtGui.QAction("Large Thumbnails", self)
        self.large_thumbnails_action.setCheckable(True)
        self.large_thumbnails_action.setChecked(current_face_size == (96, 96))
        self.thumbnail_size_action_group.addAction(self.large_thumbnails_action)
        self.large_thumbnails_action.triggered.connect(
            partial(
                list_view_actions.apply_face_thumbnail_size, self.main_window, (96, 96)
            )
        )
        self.remove_action = QtGui.QAction("Remove from List", self)
        self.remove_action.triggered.connect(self.remove_target_face_from_list)
        self.popMenu.addAction(self.parameters_copy_action)
        self.popMenu.addAction(self.parameters_paste_action)
        self.popMenu.addAction(self.save_parameters_action)
        self.popMenu.addAction(self.load_parameters_action)
        self.popMenu.addAction(self.load_parameters_and_settings_action)
        self.popMenu.addSeparator()
        self.popMenu.addAction(self.small_thumbnails_action)
        self.popMenu.addAction(self.large_thumbnails_action)
        self.popMenu.addSeparator()
        self.popMenu.addAction(self.remove_action)

    def on_context_menu(self, point):
        # show context menu
        self.create_context_menu()
        try:
            scan_active = video_control_actions.is_issue_scan_active(self.main_window)
            current_face_size = getattr(
                self.main_window, "face_thumbnail_button_size", None
            )
            self.face_header_action.setText(self.get_display_label())
            self.parameters_paste_action.setEnabled(not scan_active)
            self.load_parameters_action.setEnabled(not scan_active)
            self.load_parameters_and_settings_action.setEnabled(not scan_active)
            self.remove_action.setEnabled(not scan_active)
            self.small_thumbnails_action.setChecked(current_face_size == (70, 70))
            self.large_thumbnails_action.setChecked(current_face_size == (96, 96))
            self._exec_context_menu(point)
        finally:
            self._release_context_menu(
                "face_header_action",
                "parameters_copy_action",
                "parameters_paste_action",
                "save_parameters_action",
                "load_parameters_action",
                "load_parameters_and_settings_action",
                "thumbnail_size_action_group",
                "small_thumbnails_action",
                "large_thumbnails_action",
                "remove_action",
            )

    def remove_target_face_from_list(self):
        main_window = self.main_window
        if video_control_actions.block_if_issue_scan_active(
            main_window, "remove a target face"
        ):
            return

        if main_window.video_processor.processing:
            # --- WEBCAM GUARD ---
            # Do not kill the live hardware sensor. Just flag the UI state as dirty
            # so the background threads adapt to the removed face seamlessly.
            if main_window.video_processor.file_type == "webcam":
                main_window.video_processor.ui_state_is_dirty = True
            else:
                main_window.video_processor.stop_processing()

        i = self.get_item_position()
        main_window.targetFacesList.takeItem(i)
        main_window.target_faces.pop(self.face_id)
        from app.ui.widgets.actions import list_view_actions

        list_view_actions.refresh_target_face_display_labels(main_window)
        # Pop parameters using the target's face_id
        main_window.parameters.pop(self.face_id)
        if hasattr(main_window, "issue_frames_by_face"):
            main_window.issue_frames_by_face.pop(str(self.face_id), None)
        # Click and Select the first target face if target_faces are not empty
        if main_window.target_faces:
            list(main_window.target_faces.values())[0].click()

        # Otherwise reset parameter widgets value to the default
        else:
            common_widget_actions.set_widgets_values_using_face_id_parameters(
                main_window, face_id=None
            )
            main_window.selected_target_face_id = None
            video_control_actions.refresh_issue_frames_for_selected_face(main_window)
        video_control_actions.update_scan_review_button_states(main_window)
        video_control_actions.remove_face_parameters_and_control_from_markers(
            main_window, self.face_id
        )  # Remove parameters for the face from all markers
        common_widget_actions.refresh_frame(self.main_window)

        # Explicitly release large data before Qt schedules widget destruction.
        # KV maps can be 10-100 MB; embeddings are smaller but numpy arrays that
        # benefit from prompt deallocation. deleteLater() only schedules the C++
        # widget object; Python-side attributes survive until GC runs otherwise.
        self.assigned_kv_map = None
        self.aged_kv_map = None
        self.assigned_input_embedding.clear()
        self.aged_input_embedding.clear()
        self.embedding_store.clear()
        self.assigned_input_faces.clear()
        self.assigned_merged_embeddings.clear()

        self.deleteLater()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_display_label(self) -> str:
        item_position = self.get_item_position()
        if item_position is None:
            return "Face"
        return f"Face {item_position + 1}"

    def refresh_display_label(self):
        display_label = self.get_display_label()
        self.display_label.setText(display_label)
        face_header_action = getattr(self, "face_header_action", None)
        if face_header_action is not None:
            face_header_action.setText(display_label)

    def remove_assigned_input_face(self, input_face_id):
        if self.assigned_input_faces.get(input_face_id):
            self.assigned_input_faces.pop(input_face_id)
            self.calculate_assigned_input_embedding()

    def remove_assigned_merged_embedding(self, embedding_id):
        if self.assigned_merged_embeddings.get(embedding_id):
            self.assigned_merged_embeddings.pop(embedding_id)
            self.calculate_assigned_input_embedding()

    def copy_parameters(self):
        common_widget_actions.copy_selected_face_parameters(
            self.main_window, self.face_id
        )

    def paste_and_apply_parameters(self):
        common_widget_actions.paste_selected_face_parameters(
            self.main_window, self.face_id
        )


class InputFaceCardButton(CardButton):
    def __init__(
        self,
        media_path,
        cropped_face,
        embedding_store: Dict[str, np.ndarray],
        face_id: str,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.face_id = face_id
        self.cropped_face = cropped_face
        self.embedding_store = (
            embedding_store  # Key: embedding_swap_model, Value: embedding
        )
        self.media_path = media_path
        self.kv_map: Dict | None = None

        self.setCheckable(True)
        self.setToolTip(media_path)
        self.clicked.connect(self.load_input_face)

        # Set the context menu policy to trigger the custom context menu on right-click
        self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        # Connect the custom context menu request signal to the custom slot
        self.customContextMenuRequested.connect(self.on_context_menu)

    def set_embedding(self, embedding_swap_model: str, embedding: np.ndarray):
        self.embedding_store[embedding_swap_model] = embedding

    def get_embedding(self, embedding_swap_model: str) -> np.ndarray:
        return self.embedding_store.get(embedding_swap_model, np.array([]))

    def load_input_face(self):
        main_window = self.main_window
        if video_control_actions.block_if_issue_scan_active(
            main_window, "change input-face assignments"
        ):
            self._restore_pre_click_checked_state()
            return

        if main_window.cur_selected_target_face_button:
            cur_selected_target_face_button = (
                main_window.cur_selected_target_face_button
            )

            if QtWidgets.QApplication.keyboardModifiers() == QtCore.Qt.ShiftModifier:
                # Step 1: Find the index of the last selected item before selecting the 'current_item_position' item. If this is None, then shift select shouldn't work
                # Step 2: Find and store the details of all sequentially selected items behind 'second_last_item_position'
                # Step 3: If there are trailing items, then deselect all checked items behind the last sequentially trailing item (This is to make sure all unsequentially selected items are deselected)
                # Step 4: Now select all the items between second_last_item_position (or last trailed item, if there was trailing selected items) and the current_item_position, to complete the Shift Selection
                current_item_position = self.get_item_position()
                second_last_item_position = (
                    self.get_index_of_second_last_selected_item()
                )
                if second_last_item_position is not None:
                    selected_input_faces = []
                    if current_item_position >= second_last_item_position:
                        trailing_selected_items = (
                            self.get_sequential_trailing_selected_items(
                                second_last_item_position
                            )
                        )
                        if trailing_selected_items:
                            self.deselect_all_trailing_items(
                                trailing_selected_items[-1][0]
                            )

                            selected_input_faces = self.select_all_items_between_range(
                                trailing_selected_items[-1][0], current_item_position
                            )
                        else:
                            selected_input_faces = self.select_all_items_between_range(
                                second_last_item_position, current_item_position
                            )

                    else:
                        for input_face_id in (
                            cur_selected_target_face_button.assigned_input_faces.keys()
                        ):
                            input_face_button = main_window.input_faces[input_face_id]
                            if input_face_button != self:
                                input_face_button.setChecked(False)

                    cur_selected_target_face_button.assigned_input_faces = {}
                    for input_face in selected_input_faces:
                        cur_selected_target_face_button.assigned_input_faces[
                            input_face.face_id
                        ] = input_face.embedding_store

            elif (
                not QtWidgets.QApplication.keyboardModifiers()
                == QtCore.Qt.ControlModifier
            ):
                for (
                    input_face_id
                ) in cur_selected_target_face_button.assigned_input_faces.keys():
                    input_face_button = main_window.input_faces[input_face_id]
                    if input_face_button != self:
                        input_face_button.setChecked(False)
                cur_selected_target_face_button.assigned_input_faces = {}

            cur_selected_target_face_button.assigned_input_faces[self.face_id] = (
                self.embedding_store
            )

            if not self.isChecked():
                cur_selected_target_face_button.assigned_input_faces.pop(self.face_id)
            cur_selected_target_face_button.calculate_assigned_input_embedding()
            if not self.isChecked():
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        else:
            if (
                not QtWidgets.QApplication.keyboardModifiers()
                == QtCore.Qt.ControlModifier
            ):
                # If there is no target face selected, uncheck all other input faces
                for _, input_face_button in main_window.input_faces.items():
                    if input_face_button != self:
                        input_face_button.setChecked(False)

        common_widget_actions.refresh_frame(main_window)

    def remove_kv_data_file(self):
        if isinstance(self.kv_map, str) and self.kv_map.endswith(".pt"):
            try:
                if os.path.exists(self.kv_map):
                    os.remove(self.kv_map)
                    print(f"[INFO] Removed K/V data file: {self.kv_map}")
            except Exception as e:
                print(f"[ERROR] Error removing K/V data file {self.kv_map}: {e}")

        if isinstance(self.kv_map, str) and os.path.exists(self.kv_map):
            try:
                os.remove(self.kv_map)
                print(f"[INFO] Removed K/V data file: {self.kv_map}")
            except Exception as e:
                print(f"[ERROR] Error removing K/V data file {self.kv_map}: {e}")

    def _remove_face_from_lists(self):
        main_window = self.main_window
        i = self.get_item_position()
        if i is not None:
            main_window.inputFacesList.takeItem(i)
            main_window.input_faces.pop(self.face_id)
            for target_face_id in main_window.target_faces:
                main_window.target_faces[target_face_id].remove_assigned_input_face(
                    self.face_id
                )
            self.deleteLater()
            return True
        return False

    def deselect_currently_selected_face(self, main_window):
        self.remove_kv_data_file()
        self._remove_face_from_lists()

        common_widget_actions.refresh_frame(self.main_window)

        if not main_window.input_faces:
            main_window.placeholder_update_signal.emit(
                self.main_window.inputFacesList, False
            )

    def remove_input_face_from_list(self):
        main_window = self.main_window
        if video_control_actions.block_if_issue_scan_active(
            main_window, "remove input faces"
        ):
            return
        faces_to_remove = [
            face_button
            for _, face_button in main_window.input_faces.items()
            if face_button.isChecked()
        ]

        if not faces_to_remove:
            faces_to_remove = [self]

        was_removed = False

        for face_to_remove in faces_to_remove:
            face_to_remove.remove_kv_data_file()
            if face_to_remove._remove_face_from_lists():
                was_removed = True

        if was_removed:
            common_widget_actions.refresh_frame(main_window)
            if not main_window.input_faces:
                main_window.placeholder_update_signal.emit(
                    main_window.inputFacesList, False
                )

    def delete_input_face_to_trash(self):
        main_window = self.main_window
        if video_control_actions.block_if_issue_scan_active(
            main_window, "delete input faces"
        ):
            return
        self.remove_kv_data_file()
        self._remove_face_from_lists()

        # Send the file to the trash
        if os.path.exists(self.media_path):
            send2trash(self.media_path)
            print(f"[INFO] {self.media_path} has been sent to the trash.")
        else:
            print(f"[ERROR] {self.media_path} does not exist.")

        common_widget_actions.refresh_frame(main_window)
        if not main_window.input_faces:
            main_window.placeholder_update_signal.emit(
                main_window.inputFacesList, False
            )

    def open_target_path_by_explorer(self):
        if os.path.exists(self.media_path):
            # Normalize path
            normalized_path = os.path.normpath(os.path.abspath(self.media_path))

            if sys.platform == "win32":
                # Windows - use full path to explorer.exe to avoid PATH issues
                try:
                    # Method 1: Using subprocess without shell (more secure and reliable)
                    subprocess.Popen(f'explorer /select,"{normalized_path}"')
                except FileNotFoundError:
                    # Fallback: Use full path to explorer.exe
                    subprocess.Popen(
                        f'C:\\Windows\\explorer.exe /select,"{normalized_path}"'
                    )
            elif sys.platform == "darwin":
                # macOS
                subprocess.run(["open", "-R", self.media_path])
            else:
                # Linux
                directory = os.path.dirname(os.path.abspath(self.media_path))
                subprocess.run(["xdg-open", directory])

    def create_context_menu(self):
        # create context menu
        from app.ui.widgets.actions import list_view_actions

        self.popMenu = QtWidgets.QMenu(self)
        self.create_embed_action = QtGui.QAction(
            "Create embedding from selected faces", self
        )
        self.create_embed_action.triggered.connect(
            self.create_embedding_from_selected_faces
        )
        self.popMenu.addAction(self.create_embed_action)

        self.remove_action = QtGui.QAction("Remove from list", self)
        self.remove_action.triggered.connect(self.remove_input_face_from_list)
        self.popMenu.addAction(self.remove_action)

        self.delete_action = QtGui.QAction("Delete file to recycle bin", self)
        self.delete_action.triggered.connect(self.delete_input_face_to_trash)
        self.popMenu.addAction(self.delete_action)

        self.open_path_action = QtGui.QAction("Open file location", self)
        self.open_path_action.triggered.connect(self.open_target_path_by_explorer)
        self.popMenu.addAction(self.open_path_action)
        self.popMenu.addSeparator()

        current_face_size = getattr(
            self.main_window, "face_thumbnail_button_size", None
        )
        self.thumbnail_size_action_group = QtGui.QActionGroup(self.popMenu)
        self.thumbnail_size_action_group.setExclusive(True)

        self.small_thumbnails_action = QtGui.QAction("Small Thumbnails", self)
        self.small_thumbnails_action.setCheckable(True)
        self.small_thumbnails_action.setChecked(current_face_size == (70, 70))
        self.thumbnail_size_action_group.addAction(self.small_thumbnails_action)
        self.small_thumbnails_action.triggered.connect(
            partial(
                list_view_actions.apply_face_thumbnail_size, self.main_window, (70, 70)
            )
        )
        self.popMenu.addAction(self.small_thumbnails_action)

        self.large_thumbnails_action = QtGui.QAction("Large Thumbnails", self)
        self.large_thumbnails_action.setCheckable(True)
        self.large_thumbnails_action.setChecked(current_face_size == (96, 96))
        self.thumbnail_size_action_group.addAction(self.large_thumbnails_action)
        self.large_thumbnails_action.triggered.connect(
            partial(
                list_view_actions.apply_face_thumbnail_size, self.main_window, (96, 96)
            )
        )
        self.popMenu.addAction(self.large_thumbnails_action)
        self.popMenu.addSeparator()

        self.clear_all_faces_action = QtGui.QAction("Clear All Faces", self)
        self.clear_all_faces_action.triggered.connect(
            partial(list_view_actions.clear_all_input_faces, self.main_window)
        )
        self.popMenu.addAction(self.clear_all_faces_action)

    def on_context_menu(self, point):
        # show context menu
        self.create_context_menu()
        try:
            scan_active = video_control_actions.is_issue_scan_active(self.main_window)
            current_face_size = getattr(
                self.main_window, "face_thumbnail_button_size", None
            )
            self.create_embed_action.setEnabled(not scan_active)
            self.remove_action.setEnabled(not scan_active)
            self.delete_action.setEnabled(not scan_active)
            self.small_thumbnails_action.setChecked(current_face_size == (70, 70))
            self.large_thumbnails_action.setChecked(current_face_size == (96, 96))
            self.clear_all_faces_action.setEnabled(
                bool(self.main_window.input_faces) and not scan_active
            )
            self._exec_context_menu(point)
        finally:
            self._release_context_menu(
                "create_embed_action",
                "remove_action",
                "delete_action",
                "open_path_action",
                "thumbnail_size_action_group",
                "small_thumbnails_action",
                "large_thumbnails_action",
                "clear_all_faces_action",
            )

    def create_embedding_from_selected_faces(self):
        if video_control_actions.block_if_issue_scan_active(
            self.main_window, "create embeddings"
        ):
            return

        # Raccogli i bottoni (oggetti) invece che solo gli store.
        # Abbiamo bisogno del 'cropped_face' per estrarre le KV map.
        selected_faces = [
            input_face
            for _, input_face in self.main_window.input_faces.items()
            if input_face.isChecked()
        ]

        # Controlla se ci sono facce selezionate
        if len(selected_faces) == 0:
            common_widget_actions.create_and_show_messagebox(
                self.main_window,
                "No Faces Selected!",
                "You need to select at least one face to create a merged embedding!",
                self,
            )
        else:
            # Passa i bottoni completi al dialogo
            embed_create_dialog = CreateEmbeddingDialog(
                self.main_window, selected_faces
            )
            embed_create_dialog.exec_()


class EmbeddingCardButton(CardButton):
    def __init__(
        self,
        embedding_name: str,
        embedding_store: Dict[str, np.ndarray],
        embedding_id: str,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.embedding_id = embedding_id
        self.embedding_store = (
            embedding_store  # Key: embedding_swap_model, Value: embedding
        )
        self.embedding_name = embedding_name

        # Support both legacy dictionary maps and new batched list maps
        self._kv_map: Dict | None = None
        self._kv_map_list: List[Dict] | None = None

        self.setCheckable(True)
        self.setText(embedding_name)
        self.setToolTip(embedding_name)
        self.clicked.connect(self.load_embedding)

        # Set the context menu policy to trigger the custom context menu on right-click
        self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        # Connect the custom context menu request signal to the custom slot
        self.customContextMenuRequested.connect(self.on_context_menu)

    # --- Property definitions to intercept when a K/V map is assigned ---

    @property
    def kv_map(self) -> Dict | None:
        """Getter for the legacy single K/V map."""
        return self._kv_map

    @kv_map.setter
    def kv_map(self, value: Dict | None) -> None:
        """Setter for legacy K/V maps. Updates the UI color automatically."""
        self._kv_map = value
        self._update_ui_state()

    @property
    def kv_map_list(self) -> List[Dict] | None:
        """Getter for the new raw K/V map lists."""
        return self._kv_map_list

    @kv_map_list.setter
    def kv_map_list(self, value: List[Dict] | None) -> None:
        """Setter for the dynamic unbaked K/V map lists. Updates UI color automatically."""
        self._kv_map_list = value
        self._update_ui_state()

    def _update_ui_state(self) -> None:
        """
        Consolidated UI updater.
        Checks if either a legacy baked map or a new raw map list is present and valid.
        """
        has_legacy_map = self._kv_map is not None and len(self._kv_map) > 0
        has_new_map_list = self._kv_map_list is not None and len(self._kv_map_list) > 0

        # If valid tensors are loaded, change the text color to the accent and update tooltip
        if has_legacy_map or has_new_map_list:
            # Using the UI's native accent color (#4090a3) for consistency
            self.setStyleSheet("color: #4090a3;")
            self.setToolTip(f"{self.embedding_name} (Includes K/V Maps)")
        else:
            # Reset to default UI style
            self.setStyleSheet("")
            self.setToolTip(self.embedding_name)

    def set_embedding(self, embedding_swap_model: str, embedding: np.ndarray) -> None:
        self.embedding_store[embedding_swap_model] = embedding

    def get_embedding(self, embedding_swap_model: str) -> np.ndarray | None:
        """Restituisce l'embedding associato a un embedding_swap_model, se esiste."""
        return self.embedding_store.get(embedding_swap_model, None)

    def load_embedding(self) -> None:
        main_window = self.main_window
        if video_control_actions.block_if_issue_scan_active(
            main_window, "change merged-embedding assignments"
        ):
            self._restore_pre_click_checked_state()
            return

        if main_window.cur_selected_target_face_button:
            cur_selected_target_face_button = (
                main_window.cur_selected_target_face_button
            )
            if (
                not QtWidgets.QApplication.keyboardModifiers()
                == QtCore.Qt.ControlModifier
            ):
                for (
                    embedding_id
                ) in cur_selected_target_face_button.assigned_merged_embeddings.keys():
                    embed_button = main_window.merged_embeddings[embedding_id]
                    if embed_button != self:
                        embed_button.setChecked(False)
                cur_selected_target_face_button.assigned_merged_embeddings = {}

            cur_selected_target_face_button.assigned_merged_embeddings[
                self.embedding_id
            ] = self.embedding_store

            if not self.isChecked():
                cur_selected_target_face_button.assigned_merged_embeddings.pop(
                    self.embedding_id
                )
            cur_selected_target_face_button.calculate_assigned_input_embedding()
            if not self.isChecked():
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        else:
            if (
                not QtWidgets.QApplication.keyboardModifiers()
                == QtCore.Qt.ControlModifier
            ):
                # If there is no target face selected, uncheck all other input faces
                for embedding_id, embed_button in main_window.merged_embeddings.items():
                    if embed_button != self:
                        embed_button.setChecked(False)

        common_widget_actions.refresh_frame(main_window)

    def create_context_menu(self) -> None:
        # create context menu
        from app.ui.widgets.actions import list_view_actions

        self.popMenu = QtWidgets.QMenu(self)
        self.remove_action = QtGui.QAction("Remove Embedding", self)
        self.remove_action.triggered.connect(self.remove_embedding_from_list)
        self.popMenu.addAction(self.remove_action)
        self.popMenu.addSeparator()

        self.clear_all_embeddings_action = QtGui.QAction("Clear All Embeddings", self)
        self.clear_all_embeddings_action.triggered.connect(
            partial(list_view_actions.clear_all_embeddings, self.main_window)
        )
        self.popMenu.addAction(self.clear_all_embeddings_action)

    def on_context_menu(self, point: QtCore.QPoint) -> None:
        # show context menu
        self.create_context_menu()
        try:
            scan_active = video_control_actions.is_issue_scan_active(self.main_window)
            self.remove_action.setEnabled(not scan_active)
            self.clear_all_embeddings_action.setEnabled(
                bool(self.main_window.merged_embeddings) and not scan_active
            )
            self._exec_context_menu(point)
        finally:
            self._release_context_menu(
                "remove_action",
                "clear_all_embeddings_action",
            )

    def remove_embedding_from_list(self) -> None:
        if video_control_actions.block_if_issue_scan_active(
            self.main_window, "remove embeddings"
        ):
            return

        main_window = self.main_window
        for i in range(main_window.inputEmbeddingsList.count() - 1, -1, -1):
            list_item = main_window.inputEmbeddingsList.item(i)
            if list_item.listWidget().itemWidget(list_item) == self:
                main_window.inputEmbeddingsList.takeItem(i)
                main_window.merged_embeddings.pop(self.embedding_id)
                for target_face_id in main_window.target_faces:
                    main_window.target_faces[
                        target_face_id
                    ].remove_assigned_merged_embedding(self.embedding_id)
        common_widget_actions.refresh_frame(self.main_window)
        self.deleteLater()


class CreateEmbeddingDialog(QtWidgets.QDialog):
    def __init__(self, main_window: "MainWindow", selected_faces: list | None = None):
        super().__init__()
        # InputFaceCardButton for acces to .cropped_face and .embedding_store
        self.selected_faces = selected_faces or []
        self.main_window = main_window
        self.embedding_name = ""
        self.merge_type = ""
        self.setWindowTitle("Create Embedding")
        self.setWindowIcon(QtGui.QIcon(":/media/media/visomaster_small.png"))

        # Create widgets
        self.embed_name_edit = QtWidgets.QLineEdit(self)
        self.embed_name_edit.setPlaceholderText("Enter embedding name")

        self.merge_type_selection = QtWidgets.QComboBox(self)
        self.merge_type_selection.addItems(["Mean", "Median"])
        self.merge_type_selection.setCurrentText(
            main_window.control["EmbMergeMethodSelection"]
        )

        # Checkbox to optionally generate and include K/V maps
        self.include_kv_checkbox = QtWidgets.QCheckBox(
            "Generate and include K/V Maps (For Denoiser)", self
        )
        self.include_kv_checkbox.setChecked(
            False
        )  # Optional feature, default to false to save time
        self.include_kv_checkbox.setToolTip(
            "Will extract K/V Maps from selected faces and merge them. This might take a moment."
        )

        # Create button box
        QBtn = QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        self.buttonBox = QtWidgets.QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.create_embedding)
        self.buttonBox.rejected.connect(self.reject)

        # Create layout and add widgets
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(QtWidgets.QLabel("Embedding Name:"))
        layout.addWidget(self.embed_name_edit)
        layout.addWidget(QtWidgets.QLabel("Merge Type:"))
        layout.addWidget(self.merge_type_selection)
        layout.addWidget(self.include_kv_checkbox)
        layout.addWidget(self.buttonBox)

        # Set dialog layout
        self.setLayout(layout)

    def create_embedding(self) -> None:
        self.embedding_name = self.embed_name_edit.text().strip()
        self.merge_type = self.merge_type_selection.currentText()

        if self.embedding_name == "":
            common_widget_actions.create_and_show_messagebox(
                self.main_window,
                "Empty Embedding Name!",
                "Embedding Name cannot be empty!",
                self,
            )
            return

        # 1. Classic embedding merge and KPS separation
        merged_embedding_store: dict[str, list[np.ndarray]] = {}
        kps_5_list: list[np.ndarray] = []  # List to safely collect spatial keypoints

        for input_face in self.selected_faces:
            for embedding_swap_model, embedding in input_face.embedding_store.items():
                # Isolate keypoints to prevent L2 Normalization (which destroys spatial pixel coordinates)
                if embedding_swap_model == "kps_5":
                    kps_5_list.append(embedding)
                    continue

                if embedding_swap_model not in merged_embedding_store:
                    merged_embedding_store[embedding_swap_model] = []
                merged_embedding_store[embedding_swap_model].append(embedding)

        # Calculate the merged embedding for each arcface model
        final_embedding_store = {}
        for swap_model, embeddings in merged_embedding_store.items():
            if self.merge_type == "Mean":
                merged_emb = np.mean(embeddings, axis=0)
            elif self.merge_type == "Median":
                merged_emb = np.median(embeddings, axis=0)
            else:
                merged_emb = np.mean(embeddings, axis=0)  # Fallback

            # Apply L2 Normalization ONLY to standard latent embeddings
            norm = np.linalg.norm(merged_emb)
            if norm > 0:
                merged_emb = merged_emb / norm

            final_embedding_store[swap_model] = merged_emb

        # Process kps_5 spatial averaging (Always use Mean, never L2 Normalize)
        if kps_5_list:
            final_embedding_store["kps_5"] = np.mean(kps_5_list, axis=0)

        # 2. Extract and Collect RAW K/V Maps (Deferred Merging)
        final_kv_map_list = []
        if self.include_kv_checkbox.isChecked():
            all_kv_maps = []
            import traceback
            from PIL import Image

            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)

            try:
                extracted_new_kv = False
                for input_face in self.selected_faces:
                    with self.main_window.function_worker.denoiser_kv_extraction_lock:
                        # Check Cache first
                        if (
                            hasattr(input_face, "kv_map")
                            and input_face.kv_map is not None
                            and len(input_face.kv_map) > 0
                        ):
                            all_kv_maps.append(input_face.kv_map)
                        else:
                            # Generate Cache
                            print(
                                f"[INFO] Dialog: Generating K/V map for {input_face.media_path}"
                            )
                            try:
                                cropped_face_np = input_face.cropped_face
                                pil_img = Image.fromarray(cropped_face_np[..., ::-1])
                                if pil_img.size != (512, 512):
                                    pil_img = pil_img.resize(
                                        (512, 512), Image.Resampling.LANCZOS
                                    )

                                # Batch processing: Keep the extractor loaded for the duration of the loop
                                # average_refs has no mathematical impact here because we process 1 image at a time
                                kv_map = self.main_window.function_worker.get_kv_map_for_face(
                                    pil_img,
                                    unload_after=False,
                                )

                                if kv_map:
                                    input_face.kv_map = kv_map
                                    all_kv_maps.append(kv_map)
                                    extracted_new_kv = True
                            except Exception as e:
                                print(f"[ERROR] Error generating K/V map: {e}")
                                traceback.print_exc()

                # Cleanup: Unload the extractor once the batch is fully processed
                if extracted_new_kv:
                    self.main_window.function_worker.unload_denoiser_kv_extractor()

                # Enforce VRAM localization for created embeddings without merging them
                if all_kv_maps:
                    target_device = self.main_window.models_processor.device
                    for m in all_kv_maps:
                        for layer_key, layer_dict in m.items():
                            for k, v in layer_dict.items():
                                layer_dict[k] = v.to(target_device)

                    final_kv_map_list = all_kv_maps

            finally:
                QtWidgets.QApplication.restoreOverrideCursor()

        # 3. Button Creation and Injection
        from app.ui.widgets.actions import list_view_actions

        embedding_id = str(uuid.uuid1().int)

        list_view_actions.create_and_add_embed_button_to_list(
            main_window=self.main_window,
            embedding_name=self.embedding_name,
            embedding_store=final_embedding_store,
            embedding_id=embedding_id,
        )

        # 4. Assign the RAW K/V Map list to the new button
        if final_kv_map_list:
            if embedding_id in self.main_window.merged_embeddings:
                # Store the unmerged list so calculate_assigned_input_embedding can merge it dynamically
                self.main_window.merged_embeddings[
                    embedding_id
                ].kv_map_list = final_kv_map_list
                print(
                    f"[INFO] Successfully linked {len(final_kv_map_list)} raw K/V maps to embedding '{self.embedding_name}'."
                )

        self.accept()


class LoadingDialog(QtWidgets.QDialog):
    def __init__(
        self, message="Loading Models, please wait...\nDon't panic if it looks stuck!"
    ):
        super().__init__()
        self.setWindowTitle("Loading Models")
        self.setWindowIcon(QtGui.QIcon(":/media/media/visomaster_small.png"))
        self.setWindowFlag(QtCore.Qt.WindowCloseButtonHint, False)
        self.setModal(True)  # Block interaction with other windows
        self.setFixedSize(225, 125)  # Increased size for better layout

        # Create main layout
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(4, 4, 4, 4)  # Add some padding
        layout.setSpacing(8)  # Add spacing between elements

        # Icon Label
        self.icon_label = QtWidgets.QLabel()
        self.icon_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.icon_label.setPixmap(
            QtGui.QPixmap(":/media/media/repeat.png").scaled(
                30,
                30,
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation,
            )
        )

        # Message Label
        self.label = QtWidgets.QLabel(message)
        self.label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label.setWordWrap(True)  # Allow text to wrap within the dialog
        self.label.setStyleSheet("""
            font-size: 12px;  /* Set font size */
            font-weight: bold;  /* Make the text bold */
        """)

        # Add widgets to layout
        layout.addWidget(self.icon_label)
        layout.addWidget(self.label)
        self.setLayout(layout)


# Custom progress dialog
class ProgressDialog(QtWidgets.QProgressDialog):
    """
    QProgressDialog with confirmation-before-cancel behavior that works with PySide6.

    IMPORTANT:
    - Do NOT rely on overriding cancel()/wasCanceled(); QProgressDialog's cancel is not virtual.
    - Use the `canceled` signal to intercept cancellation.
    - Batch code must check confirmedCanceled() instead of wasCanceled().
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._confirmed_cancelled = False
        self._confirm_dialog_open = False
        self._confirmation_disabled = False

        # Prevent Qt from auto-closing/resetting the dialog unexpectedly
        try:
            self.setAutoClose(False)
        except Exception:
            pass
        try:
            self.setAutoReset(False)
        except Exception:
            pass

        # Ensure cancel text exists
        try:
            self.setCancelButtonText("Cancel")
        except Exception:
            pass

        # Intercept Qt's cancel flow via signal (this is reliable in PySide6)
        self.canceled.connect(self._on_canceled)

    def confirmedCanceled(self) -> bool:
        """Return True only if the user confirmed stopping."""
        return self._confirmed_cancelled

    def close_without_confirmation(self):
        """
        Close the dialog for normal teardown without treating it as a user cancel.
        """
        self._confirmation_disabled = True
        blocker = None
        try:
            blocker = QtCore.QSignalBlocker(self)
        except Exception:
            blocker = None

        try:
            try:
                self.reset()
            except Exception:
                pass

            try:
                self.close()
            except Exception:
                pass
        finally:
            del blocker

    def _on_canceled(self):
        """
        Qt has already marked the dialog as canceled and may hide it.
        We show confirmation ASAP (queued to the event loop) and then either:
        - confirm: keep _confirmed_cancelled=True (batch loop will stop)
        - decline: reset & re-show dialog, and keep _confirmed_cancelled=False (batch continues)
        """
        if self._confirmation_disabled:
            return
        if self._confirmed_cancelled:
            return
        if self._confirm_dialog_open:
            return

        # Defer confirmation to next event loop turn to avoid showing behind/after close
        QtCore.QTimer.singleShot(0, self._show_confirm_and_apply)

    def _show_confirm_and_apply(self):
        if self._confirmation_disabled:
            return
        if self._confirmed_cancelled:
            return
        if self._confirm_dialog_open:
            return

        self._confirm_dialog_open = True
        try:
            parent = self.parent() or self

            box = QtWidgets.QMessageBox(parent)
            box.setIcon(QtWidgets.QMessageBox.Warning)
            box.setWindowTitle("Confirm stop")
            box.setText("Stop the current task?")
            box.setInformativeText(
                "Processing will stop immediately.\nOutputs may be incomplete."
            )
            box.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
            box.setDefaultButton(QtWidgets.QMessageBox.No)

            # Force on-top to avoid “dialog appears only after main window closes”
            try:
                box.setWindowFlag(QtCore.Qt.WindowStaysOnTopHint, True)
            except Exception:
                pass

            ret = box.exec()

            if ret == QtWidgets.QMessageBox.Yes:
                self._confirmed_cancelled = True
                # leave as-is; batch loop will see confirmedCanceled()==True and stop
                return

            # User declined: undo the cancel state and re-show progress dialog
            self._confirmed_cancelled = False

            # reset() clears internal canceled/hidden state; safe even if already hidden
            try:
                self.reset()
            except Exception:
                pass

            try:
                self.show()
                self.raise_()
                self.activateWindow()
            except Exception:
                pass

        finally:
            self._confirm_dialog_open = False


class LoadLastWorkspaceDialog(QtWidgets.QDialog):
    def __init__(
        self,
        main_window: "MainWindow",
    ):
        super().__init__()
        self.main_window = main_window
        self.setWindowTitle("Load Last Workspace")
        self.setWindowIcon(QtGui.QIcon(":/media/media/visomaster_small.png"))

        # Create button box
        QBtn = QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        self.buttonBox = QtWidgets.QDialogButtonBox(QBtn)
        self.buttonBox.setCenterButtons(True)  # <-- ADD THIS LINE
        self.buttonBox.accepted.connect(self.load_workspace)
        self.buttonBox.rejected.connect(self.reject)

        # Create layout and add widgets
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(QtWidgets.QLabel("Do you want to load your last workspace?"))
        layout.addWidget(self.buttonBox)

        # Set dialog layout
        self.setLayout(layout)

    def load_workspace(self):
        self.accept()
        save_load_actions.load_saved_workspace(
            self.main_window, str(self.main_window.last_workspace_path)
        )


class JobLoadingDialog(QtWidgets.QDialog):
    def __init__(self, total_steps, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Loading Job Data...")
        self.setWindowIcon(QtGui.QIcon(":/media/media/visomaster_small.png"))
        self.setWindowFlag(QtCore.Qt.WindowCloseButtonHint, False)
        self.setModal(True)
        self.setFixedSize(300, 120)

        self.layout = QtWidgets.QVBoxLayout()
        self.label = QtWidgets.QLabel("Loading job data...")
        self.label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, total_steps)
        self.progress_bar.setValue(0)
        self.step_label = QtWidgets.QLabel("")
        self.step_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        self.layout.addWidget(self.label)
        self.layout.addWidget(self.progress_bar)
        self.layout.addWidget(self.step_label)
        self.setLayout(self.layout)

    def update_progress(self, current, total, step_name):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        self.step_label.setText(f"{step_name} ({current}/{total})")
        QtWidgets.QApplication.processEvents()


class SaveJobDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, input_filename=""):
        super().__init__(parent)
        self.setWindowTitle("Save Job")
        self.setWindowIcon(QtGui.QIcon(":/media/media/visomaster_small.png"))

        # Widgets
        self.job_name_label = QtWidgets.QLabel("Job Name:")
        self.job_name_edit = QtWidgets.QLineEdit(self)
        # self.job_name_edit.setPlaceholderText("Enter job name")
        self.job_name_edit.setText(input_filename)

        self.set_output_name_checkbox = QtWidgets.QCheckBox(
            "Use job name for output file name", self
        )
        self.set_output_name_checkbox.setChecked(True)

        self.output_name_label = QtWidgets.QLabel("Output File Name:")
        self.output_name_edit = QtWidgets.QLineEdit(self)
        # self.output_name_edit.setPlaceholderText("Leave blank for default")
        self.output_name_edit.setText(input_filename)

        # Button box
        QBtn = QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        self.buttonBox = QtWidgets.QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        # Layout
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.job_name_label)
        layout.addWidget(self.job_name_edit)
        layout.addWidget(self.set_output_name_checkbox)
        layout.addWidget(self.output_name_label)
        layout.addWidget(self.output_name_edit)
        layout.addWidget(self.buttonBox)
        self.setLayout(layout)

        # Connect checkbox signal to slot
        self.set_output_name_checkbox.toggled.connect(self._toggle_output_name_field)

        # Initial state
        self._toggle_output_name_field(self.set_output_name_checkbox.isChecked())

    def _toggle_output_name_field(self, checked):
        """Show/hide the output file name field based on checkbox state."""
        self.output_name_label.setVisible(not checked)
        self.output_name_edit.setVisible(not checked)
        # Adjust dialog size hint based on visibility
        self.adjustSize()

    @property
    def job_name(self):
        return self.job_name_edit.text().strip()

    @property
    def use_job_name_for_output(self):
        return self.set_output_name_checkbox.isChecked()

    @property
    def output_file_name(self):
        # Return the output file name only if the checkbox is unchecked and the field is not empty
        if not self.use_job_name_for_output:
            name = self.output_name_edit.text().strip()
            return name if name else None  # Return None if empty, job_name will be used
        return None  # Return None if checkbox is checked


class ParametersWidget:
    def __init__(self, *args, **kwargs):
        self.default_value = kwargs.get("default_value", False)
        self.min_value = kwargs.get("min_value", False)
        self.max_value = kwargs.get("max_value", False)
        self.group_layout_data: Dict[str, Dict[str, Any]] = kwargs.get(
            "group_layout_data", {}
        )
        self.widget_name = kwargs.get("widget_name", False)
        self.label_widget: QtWidgets.QLabel = kwargs.get("label_widget", False)
        self.group_widget: QtWidgets.QGroupBox = kwargs.get("group_widget", False)
        self.main_window: "MainWindow" = kwargs.get("main_window", False)
        self.line_edit: ParameterLineEdit | ParameterLineDecimalEdit = (
            False  # Only sliders have textbox currently
        )
        self.reset_default_button: QPushButton = False
        self.enable_refresh_frame = True  # This flag can be used to temporarily disable refreshing the frame when the widget value is changed


class SelectionBox(QtWidgets.QComboBox, ParametersWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        ParametersWidget.__init__(self, *args, **kwargs)
        self.selection_values = kwargs.get("selection_values", [])
        self.currentTextChanged.connect(
            partial(
                common_widget_actions.show_hide_related_widgets,
                self.main_window,
                self,
                self.widget_name,
            )
        )

    def reset_to_default_value(self):
        # Check if selection values are dynamically retrieved
        if callable(self.selection_values) and callable(self.default_value):
            self.clear()
            self.addItems(self.selection_values())
            self.set_value(self.default_value())
        else:
            self.set_value(self.default_value)

    def set_value(self, value):
        resolved_value = value() if callable(value) else value
        data_index = self.findData(resolved_value)
        if data_index != -1:
            self.setCurrentIndex(data_index)
        else:
            self.setCurrentText(resolved_value)

    def showPopup(self):
        view = self.view()
        if view and self.count() > 0:
            view.setUniformItemSizes(True)
            view.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
            row_height = max(view.sizeHintForRow(0), self.sizeHint().height())
            frame = view.frameWidth() * 2
            desired_height = (row_height * self.count()) + frame
            popup_origin = self.mapToGlobal(QtCore.QPoint(0, 0))
            app = QtWidgets.QApplication.instance()
            screen = app.screenAt(popup_origin) if app else None
            if screen is None:
                screen = self.screen()
            available = (
                screen.availableGeometry() if screen else QtCore.QRect(0, 0, 800, 600)
            )
            space_below = available.bottom() - (popup_origin.y() + self.height()) - 8
            space_above = popup_origin.y() - available.top() - 8
            max_space = max(space_below, space_above, 0)

            if desired_height <= max(space_below, 0):
                popup_height = desired_height
            elif desired_height <= max(space_above, 0):
                popup_height = desired_height
            else:
                popup_height = max(min(desired_height, max_space), row_height + frame)

            view.setMinimumHeight(popup_height)
            view.setMaximumHeight(popup_height)
        super().showPopup()

    def wheelEvent(self, event: QtGui.QWheelEvent):
        wheel_control_enabled = bool(
            self.main_window.control.get("SliderMouseWheelControlToggle", False)
        )
        ctrl_pressed = bool(
            QtWidgets.QApplication.keyboardModifiers()
            & QtCore.Qt.KeyboardModifier.ControlModifier
        )
        if not wheel_control_enabled and not ctrl_pressed:
            event.ignore()
            return
        super().wheelEvent(event)


class ToggleButton(QtWidgets.QPushButton, ParametersWidget):
    _circle_position = None

    def __init__(
        self,
        bg_color="#000000",
        circle_color="#ffffff",
        active_color="#4facc9",
        default_value=False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        ParametersWidget.__init__(self, *args, **kwargs)

        self.setFixedSize(30, 15)
        self.setCursor(QtCore.Qt.PointingHandCursor)
        self.setCheckable(True)

        self._bg_color = bg_color
        self._circle_color = circle_color
        self._active_color = active_color
        self.default_value = bool(default_value)
        self._circle_position = 1  # Start position of the circle
        self.animation_curve = QtCore.QEasingCurve.OutCubic

        # Animation
        self.animation = QtCore.QPropertyAnimation(self, b"circle_position", self)
        self.animation.setDuration(300)  # Animation duration in milliseconds
        self.animation.setEasingCurve(self.animation_curve)

        self.toggled.connect(
            partial(
                common_widget_actions.show_hide_related_widgets,
                self.main_window,
                self,
                self.widget_name,
                None,
            )
        )

        # Check Denoiser Button
        if self.widget_name and "Denoiser" in self.widget_name:
            self.toggled.connect(self._trigger_kv_recalc)

    def _trigger_kv_recalc(self, checked):
        """
        Forces the update of K/V Maps and refreshes the image.
        Uses a QTimer to delay execution and ensure that
        main_window.control has properly registered the 'True' state of the button.
        """

        def delayed_recalc():
            # 1. Recalculate the K/V map for all target faces
            if hasattr(self.main_window, "target_faces"):
                for face in self.main_window.target_faces.values():
                    face.calculate_assigned_input_embedding()

            # 2. Force image refresh with the new data
            import app.ui.widgets.actions.common_actions as common_widget_actions

            common_widget_actions.refresh_frame(self.main_window)

        # 50 ms delay for PySide6
        QtCore.QTimer.singleShot(50, delayed_recalc)

    # Property for animation
    def _get_circle_position(self):
        return self._circle_position

    def _set_circle_position(self, pos: int):
        self._circle_position = pos
        self.update()  # Update the widget to trigger paintEvent

    circle_position = QtCore.Property(
        int, fget=_get_circle_position, fset=_set_circle_position
    )

    def start_animation(self):
        # Animate circle position when toggled
        start_pos = 1 if self.isChecked() else 15
        end_pos = 15 if self.isChecked() else 1

        self.animation.setStartValue(start_pos)
        self.animation.setEndValue(end_pos)
        self.animation.start()

    def paintEvent(self, e):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.setPen(QtCore.Qt.NoPen)

        rect = QtCore.QRect(0, 0, self.width(), self.height())

        if self.isChecked():
            p.setBrush(QtGui.QColor(self._active_color))
            p.drawRoundedRect(
                0, 0, rect.width(), self.height(), self.height() / 2, self.height() / 2
            )
        else:
            p.setBrush(QtGui.QColor(self._bg_color))
            p.drawRoundedRect(
                0, 0, rect.width(), self.height(), self.height() / 2, self.height() / 2
            )

        # Draw the circle at the animated position
        p.setBrush(QtGui.QColor(self._circle_color))
        p.drawEllipse(self._circle_position, 1, 13, 13)

        p.end()

    def reset_to_default_value(self):
        self.setChecked(bool(self.default_value))

    # Custom method in all parameter widgets to set value
    def set_value(self, value):
        self.setChecked(value)


class ParameterSlider(QtWidgets.QSlider, ParametersWidget):
    def __init__(
        self,
        min_value=0,
        max_value=0,
        default_value=0,
        step_size=1,
        fixed_width=130,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        ParametersWidget.__init__(self, *args, **kwargs)
        self.min_value = int(min_value)
        self.max_value = int(max_value)
        self.step_size = int(step_size)
        self.default_value = int(default_value)

        # Debounce timer for handle_slider_moved
        self.debounce_timer = QtCore.QTimer()
        self.debounce_timer.setSingleShot(
            True
        )  # Assicura che il timer scatti una sola volta
        self.debounce_timer.timeout.connect(
            self.handle_slider_moved
        )  # Collega il timeout al metodo

        self.setMinimum(int(min_value))
        self.setMaximum(int(max_value))
        self.setValue(self.default_value)
        self.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum
        )
        # Set a fixed width for the slider
        self.setFixedWidth(fixed_width)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_Hover)

        # Connect sliderMoved with debounce
        self.sliderMoved.connect(self.start_debounce)

    def start_debounce(self):
        """Start debounce timer for slider movements."""
        self.debounce_timer.start(300)  # Attendi 300ms dopo lo spostamento dello slider

    def handle_slider_moved(self):
        """Handle the slider movement after debounce."""
        position = self.sliderPosition()  # Ottieni la posizione attuale dello slider
        # """Handle the slider movement (dragging) and set the correct value."""
        new_value = round(position / self.step_size) * self.step_size

        # Set the scaled value
        self.setValue(new_value)

        # print(f"Slider moved to: {new_value}")  # Debugging: log the final value

    def reset_to_default_value(self):
        self.setValue(int(self.default_value))

    # def value(self):
    #     # """Return the slider value as a float, scaled by the decimals."""
    #     return super().value()

    def setValue(self, value):
        """Set the slider value, scaling it from a float to the internal integer."""
        super().setValue(int(value))
        if self.line_edit:
            self.line_edit.set_value(
                int(value)
            )  # Aggiorna immediatamente il valore nel line edit

    def wheelEvent(self, event):
        """Override wheel event to define custom increments/decrements with the mouse wheel."""
        wheel_control_enabled = bool(
            self.main_window.control.get("SliderMouseWheelControlToggle", False)
        )
        ctrl_pressed = bool(
            QtWidgets.QApplication.keyboardModifiers()
            & QtCore.Qt.KeyboardModifier.ControlModifier
        )
        if not wheel_control_enabled and not ctrl_pressed:
            event.ignore()
            return

        num_steps = event.angleDelta().y() / 120  # 120 is one step of the wheel

        # Adjust the current value based on the number of steps
        current_value = self.value()

        # Calculate the new value based on the step size and num_steps
        new_value = current_value + (self.step_size * num_steps)

        # Ensure the new value is within the valid range
        new_value = min(max(new_value, self.min_value), self.max_value)

        # Update the slider's internal value (ensuring precision)
        self.setValue(new_value)

        # Accept the event
        event.accept()

    def keyPressEvent(self, event: QtGui.QKeyEvent):
        """Override key press event to handle arrow key increments/decrements."""
        # Get the current value of the slider
        current_value = self.value()

        # Check which key is pressed
        if event.key() == QtCore.Qt.Key_Right:
            # Increment value by step_size when right arrow is pressed
            new_value = current_value + self.step_size
        elif event.key() == QtCore.Qt.Key_Left:
            # Decrement value by step_size when left arrow is pressed
            new_value = current_value - self.step_size
        else:
            # Pass the event to the base class if it's not an arrow key
            super().keyPressEvent(event)
            return

        # Ensure the new value is within the valid range
        new_value = min(max(new_value, self.min_value), self.max_value)

        # Set the new value to the slider
        self.setValue(new_value)

        # Accept the event
        event.accept()

    def mousePressEvent(self, event):
        """Handle the mouse press event to update the slider value immediately."""
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.setValue(self.pos_to_value(event.pos().x()))

        # Chiama il metodo della classe base per gestire il resto dell'evento
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        new_value = self.pos_to_value(event.pos().x())
        QtWidgets.QToolTip.showText(QtGui.QCursor.pos(), f"{new_value}")
        super().mouseMoveEvent(event)

    def set_value(self, value):
        self.setValue(value)

    def pos_to_value(self, x) -> float:
        # Calcola la posizione cliccata lungo la barra dello slider
        new_position = QtWidgets.QStyle.sliderValueFromPosition(
            self.minimum(), self.maximum(), x, self.width()
        )
        # Applica lo step size, arrotondando il valore allo step più vicino
        return round(new_position / self.step_size) * self.step_size


class ParameterDecimalSlider(QtWidgets.QSlider, ParametersWidget):
    def __init__(
        self,
        min_value=0.0,
        max_value=1.0,
        default_value=0.00,
        decimals=2,
        step_size=0.01,
        fixed_width=130,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        ParametersWidget.__init__(self, *args, **kwargs)

        # Ensure min, max, and default are floats
        min_value = float(min_value)
        max_value = float(max_value)
        default_value = float(default_value)

        # Store step size and decimal precision
        self.step_size = step_size
        self.decimals = decimals

        # Debounce timer for handle_slider_moved
        self.debounce_timer = QtCore.QTimer()
        self.debounce_timer.setSingleShot(
            True
        )  # Assicura che il timer scatti una sola volta
        self.debounce_timer.timeout.connect(
            self.handle_slider_moved
        )  # Collega il timeout al metodo

        # Scale values for internal handling (to manage decimals)
        self.scale_factor = 10**self.decimals
        self.min_value = int(min_value * self.scale_factor)
        self.max_value = int(max_value * self.scale_factor)
        self.default_value = int(default_value * self.scale_factor)

        # Set slider properties
        self.setMinimum(self.min_value)
        self.setMaximum(self.max_value)
        self.setValue(float(self.default_value) / self.scale_factor)
        self.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum
        )
        self.setFixedWidth(fixed_width)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_Hover)

        # Connect sliderMoved with debounce
        self.sliderMoved.connect(self.start_debounce)

    def start_debounce(self):
        """Start debounce timer for slider movements."""
        self.debounce_timer.start(300)  # Attendi 300ms dopo lo spostamento dello slider

    def handle_slider_moved(self):
        """Handle the slider movement after debounce."""
        position = self.sliderPosition()  # Ottieni la posizione attuale dello slider
        new_value = position / self.scale_factor
        new_value = round(new_value / self.step_size) * self.step_size

        # Imposta il nuovo valore
        self.setValue(new_value)

        # print(f"Slider moved to: {new_value}")  # Debugging: log the final value

    def reset_to_default_value(self):
        """Reset the slider to its default value."""
        self.setValue(float(self.default_value) / self.scale_factor)

    def value(self):
        """Return the slider value as a float, scaled by the decimals."""
        return super().value() / self.scale_factor

    def setValue(self, value):
        """Set the slider value, scaling it from a float to the internal integer."""
        # Arrotonda il valore a 2 decimali, come specificato in decimals
        value = round(value, self.decimals)

        # Moltiplica per il fattore di scala e arrotonda prima di convertirlo in intero
        scaled_value = int(round(float(value) * float(self.scale_factor)))

        super().setValue(scaled_value)
        if self.line_edit:
            self.line_edit.set_value(float(value))

    def wheelEvent(self, event):
        """Override wheel event to define custom increments/decrements with the mouse wheel."""
        wheel_control_enabled = bool(
            self.main_window.control.get("SliderMouseWheelControlToggle", False)
        )
        ctrl_pressed = bool(
            QtWidgets.QApplication.keyboardModifiers()
            & QtCore.Qt.KeyboardModifier.ControlModifier
        )
        if not wheel_control_enabled and not ctrl_pressed:
            event.ignore()
            return

        num_steps = event.angleDelta().y() / 120  # 120 is one step of the wheel

        # Adjust the current value based on the number of steps
        current_value = self.value()

        # Calculate the new value based on the step size and num_steps
        new_value = current_value + (self.step_size * num_steps)

        # Ensure the new value is within the valid range
        new_value = min(
            max(round(new_value, self.decimals), self.min_value / self.scale_factor),
            self.max_value / self.scale_factor,
        )

        # Update the slider's internal value (ensuring precision)
        self.setValue(new_value)

        # Accept the event
        event.accept()

    def keyPressEvent(self, event):
        """Override key press event to handle arrow key increments/decrements."""
        # Get the current value of the slider
        current_value = self.value()

        # Check which key is pressed
        if event.key() == QtCore.Qt.Key_Right:
            # Increment value by step_size when right arrow is pressed
            new_value = current_value + self.step_size
        elif event.key() == QtCore.Qt.Key_Left:
            # Decrement value by step_size when left arrow is pressed
            new_value = current_value - self.step_size
        else:
            # Pass the event to the base class if it's not an arrow key
            super().keyPressEvent(event)
            return

        # Ensure the new value is within the valid range
        new_value = min(
            max(round(new_value, self.decimals), self.min_value / self.scale_factor),
            self.max_value / self.scale_factor,
        )

        # Set the new value to the slider
        self.setValue(new_value)

        # Accept the event
        event.accept()

    def mousePressEvent(self, event):
        """Handle the mouse press event to update the slider value immediately."""
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.setValue(self.pos_to_value(event.pos().x()))

        # Chiama il metodo della classe base per gestire il resto dell'evento
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        new_value = self.pos_to_value(event.pos().x())
        QtWidgets.QToolTip.showText(QtGui.QCursor.pos(), f"{new_value}")
        super().mouseMoveEvent(event)

    def set_value(self, value):
        self.setValue(value)

    def pos_to_value(self, x) -> float:
        new_position = QtWidgets.QStyle.sliderValueFromPosition(
            self.minimum(), self.maximum(), x, self.width()
        )

        # Converti la nuova posizione nello spazio decimale
        new_value = new_position / self.scale_factor

        # Applica lo step size, arrotondando il valore allo step più vicino
        new_value = round(new_value / self.step_size) * self.step_size

        # Imposta il nuovo valore con la precisione corretta
        return round(new_value, self.decimals)


class ParameterLineEdit(QtWidgets.QLineEdit):
    def __init__(
        self,
        min_value: int,
        max_value: int,
        default_value: str,
        fixed_width: int = 38,
        max_length: int = 3,
        alignment: int = 1,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.setFixedWidth(fixed_width)  # Make the line edit narrower
        self.setMaxLength(max_length)
        self.setValidator(
            QtGui.QIntValidator(min_value, max_value)
        )  # Restrict input to numbers

        # Optional: Align text to the right for better readability
        if alignment == 0:
            self.setAlignment(QtGui.Qt.AlignLeft)
        elif alignment == 1:
            self.setAlignment(QtGui.Qt.AlignCenter)
        else:
            self.setAlignment(QtGui.Qt.AlignRight)

        self.setText(default_value)

    def set_value(self, value: int):
        """Set the line edit's value."""
        self.setText(str(value))


class ParameterLineDecimalEdit(QtWidgets.QLineEdit):
    def __init__(
        self,
        min_value: float,
        max_value: float,
        default_value: str,
        decimals: int = 2,
        step_size=0.01,
        fixed_width: int = 38,
        max_length: int = 5,
        alignment: int = 1,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.setFixedWidth(fixed_width)  # Adjust the width for decimal numbers
        self.decimals = decimals
        self.step_size = step_size
        self.min_value = min_value
        self.max_value = max_value
        float_default_value = float(default_value)
        self.setMaxLength(max_length)
        self.setValidator(QtGui.QDoubleValidator(min_value, max_value, decimals))
        # Optional: Align text to the right for better readability
        if alignment == 0:
            self.setAlignment(QtGui.Qt.AlignLeft)
        elif alignment == 1:
            self.setAlignment(QtGui.Qt.AlignCenter)
        else:
            self.setAlignment(QtGui.Qt.AlignRight)
        self.setText(f"{float_default_value:.{self.decimals}f}")

    def set_value(self, value: float):
        """Set the line edit's value with proper handling for step size and rounding."""
        # Clamp the value to ensure it's within min and max range
        new_value = max(min(value, self.max_value), self.min_value)

        # Round the value to the nearest step size
        rounded_value = round(new_value / self.step_size) * self.step_size

        # Ensure the value is rounded to the specified number of decimals
        rounded_value = round(rounded_value, self.decimals)

        # Ensure the formatted value has exactly 'self.decimals' decimal places, even for negative numbers
        format_string = f"{{:.{self.decimals}f}}"

        formatted_value = format_string.format(rounded_value)

        # Set the text with the correct number of decimal places
        self.setText(formatted_value)

    def get_value(self) -> float:
        """Get the current value from the line edit."""
        return float(self.text())


class ParameterText(QtWidgets.QLineEdit, ParametersWidget):
    def __init__(
        self,
        default_value: str,
        fixed_width: int = 130,
        max_length: int = 500,
        alignment: int = 0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        ParametersWidget.__init__(self, *args, **kwargs)
        self.data_type = kwargs.get("data_type")
        self.exec_function = kwargs.get("exec_function")
        self.exec_function_args = kwargs.get("exec_function_args", [])

        self.setFixedWidth(fixed_width)  # Make the line edit narrower
        self.setMaxLength(max_length)
        self.default_value = default_value

        # Optional: Align text to the right for better readability
        if alignment == 0:
            self.setAlignment(QtGui.Qt.AlignLeft)
        elif alignment == 1:
            self.setAlignment(QtGui.Qt.AlignCenter)
        else:
            self.setAlignment(QtGui.Qt.AlignRight)

        # Set the initial text to the default value
        self.setText(self.default_value)

    def reset_to_default_value(self):
        """Reset the line edit to its default value."""
        self.setText(self.default_value)
        if self.data_type == "parameter":
            common_widget_actions.update_parameter(
                self.main_window,
                self.widget_name,
                self.text(),
                enable_refresh_frame=self.enable_refresh_frame,
            )
        else:
            common_widget_actions.update_control(
                self.main_window,
                self.widget_name,
                self.text(),
                exec_function=self.exec_function,
                exec_function_args=self.exec_function_args,
            )

    def focusOutEvent(self, event):
        """Handle the focus out event (when the QLineEdit loses focus)."""
        if self.data_type == "parameter":
            common_widget_actions.update_parameter(
                self.main_window,
                self.widget_name,
                self.text(),
                enable_refresh_frame=self.enable_refresh_frame,
            )
        else:
            common_widget_actions.update_control(
                self.main_window,
                self.widget_name,
                self.text(),
                exec_function=self.exec_function,
                exec_function_args=self.exec_function_args,
            )

        # Call the base class method to ensure normal behavior
        super().focusOutEvent(event)

    def set_value(self, value):
        self.setText(value)


class ParameterResetDefaultButton(QtWidgets.QPushButton):
    def __init__(
        self,
        related_widget: ParameterSlider | ParameterDecimalSlider | SelectionBox,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.related_widget = related_widget
        button_icon = QtGui.QIcon(QtGui.QPixmap(":/media/media/reset_default.png"))
        self.setIcon(button_icon)
        self.setFixedWidth(30)  # Make the line edit narrower
        self.setCursor(QtCore.Qt.PointingHandCursor)
        self.setToolTip("Reset to default value")

        self.clicked.connect(related_widget.reset_to_default_value)


class FormGroupBox(QtWidgets.QGroupBox):
    def __init__(
        self,
        main_window: "MainWindow",
        title="Form Group",
        parent=None,
    ):
        super().__init__(title, parent)
        self.main_window = main_window
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred
        )
        self.setFlat(True)


class SectionHeaderButton(QtWidgets.QPushButton):
    def __init__(self, title: str, expanded: bool = True, parent=None):
        super().__init__(title, parent)
        self._indicator_angle: float = 90.0 if expanded else 0.0
        self._fixed_height = 24
        self.setCursor(QtCore.Qt.PointingHandCursor)
        self.setFlat(True)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        self.setFixedHeight(self._fixed_height)
        self.setText("")
        self.setStyleSheet("QPushButton {border: none;background: transparent;}")
        self._title = title

    def _get_indicator_angle(self) -> float:
        return self._indicator_angle

    def _set_indicator_angle(self, angle: float):
        self._indicator_angle = angle
        self.update()

    indicator_angle = QtCore.Property(
        float,
        fget=_get_indicator_angle,
        fset=_set_indicator_angle,
    )

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        background_rect = self.rect().adjusted(0, 0, -1, -1)
        background_color = self.palette().buttonText().color()
        background_color.setAlpha(20 if self.underMouse() else 10)
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(background_color)
        painter.drawRoundedRect(background_rect, 4, 4)

        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(self.palette().buttonText())

        triangle = QtGui.QPolygonF(
            [
                QtCore.QPointF(-3.5, -4.5),
                QtCore.QPointF(-3.5, 4.5),
                QtCore.QPointF(4.5, 0),
            ]
        )
        center = QtCore.QPointF(11, self.height() / 2)
        painter.translate(center)
        painter.rotate(self._indicator_angle)
        painter.drawPolygon(triangle)
        painter.resetTransform()

        text_rect = self.rect().adjusted(24, 0, -8, 0)
        painter.setPen(self.palette().buttonText().color())
        font = painter.font()
        font.setWeight(QtGui.QFont.Weight.Bold)
        painter.setFont(font)
        painter.drawText(
            text_rect,
            QtCore.Qt.AlignmentFlag.AlignVCenter | QtCore.Qt.AlignmentFlag.AlignLeft,
            self._title,
        )
        painter.end()


class CollapsibleSection(QtWidgets.QWidget):
    _expanded_layout_spacing = 2
    _collapsed_layout_spacing = 0
    _expanded_bottom_margin = 2
    _collapsed_bottom_margin = 2

    def __init__(
        self,
        main_window: "MainWindow",
        title: str,
        section_id: str,
        expanded: bool = True,
        parent=None,
    ):
        super().__init__(parent)
        self.main_window = main_window
        self.section_id = section_id
        self._expanded = expanded

        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Maximum,
        )

        self.header_button = SectionHeaderButton(title, expanded=expanded, parent=self)
        self.header_button.clicked.connect(
            lambda _checked=False: self.toggle_expanded()
        )

        self.content_widget = QtWidgets.QWidget(self)
        self.content_widget.setVisible(expanded)
        self.content_widget.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Maximum,
        )

        self.indicator_animation = QtCore.QPropertyAnimation(
            self.header_button, b"indicator_angle", self
        )
        self.indicator_animation.setDuration(160)
        self.indicator_animation.setEasingCurve(QtCore.QEasingCurve.Type.OutCubic)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, self._expanded_bottom_margin)
        layout.setSpacing(self._expanded_layout_spacing)
        layout.addWidget(self.header_button)
        layout.addWidget(self.content_widget)
        self._apply_layout_mode(expanded)

    def content_layout(self) -> QtWidgets.QVBoxLayout:
        existing_layout = self.content_widget.layout()
        if existing_layout is None:
            content_layout = QtWidgets.QVBoxLayout(self.content_widget)
            content_layout.setContentsMargins(0, 0, 0, 0)
            content_layout.setSpacing(0)
            return content_layout
        return existing_layout

    def is_expanded(self) -> bool:
        return self._expanded

    def _collapsed_height(self) -> int:
        margins = self.layout().contentsMargins()
        return margins.top() + self.header_button.height() + margins.bottom()

    def _collapsed_size(self) -> QtCore.QSize:
        return QtCore.QSize(0, self._collapsed_height())

    def _apply_layout_mode(self, expanded: bool) -> None:
        layout = self.layout()
        if expanded:
            layout.setContentsMargins(0, 0, 0, self._expanded_bottom_margin)
            layout.setSpacing(self._expanded_layout_spacing)
            self.content_widget.setVisible(True)
            self.content_widget.setMinimumHeight(0)
            self.content_widget.setMaximumHeight(16777215)
        else:
            layout.setContentsMargins(0, 0, 0, self._collapsed_bottom_margin)
            layout.setSpacing(self._collapsed_layout_spacing)
            self.content_widget.setVisible(False)
            self.content_widget.setMinimumHeight(0)
            self.content_widget.setMaximumHeight(0)
        self.content_widget.updateGeometry()
        layout.invalidate()
        layout.activate()

    def minimumSizeHint(self) -> QtCore.QSize:
        if not self._expanded:
            return self._collapsed_size()
        return super().minimumSizeHint()

    def sizeHint(self) -> QtCore.QSize:
        if not self._expanded:
            return self._collapsed_size()
        return super().sizeHint()

    def set_expanded(
        self,
        expanded: bool,
        animate: bool = True,
        update_state: bool = True,
    ):
        expanded = bool(expanded)
        previous_state = self._expanded
        self._expanded = expanded

        self._apply_layout_mode(expanded)
        self.adjustSize()
        self.updateGeometry()
        parent_layout = self.parentWidget().layout() if self.parentWidget() else None
        if parent_layout is not None:
            parent_layout.invalidate()
            parent_layout.activate()

        start_angle = 90 if previous_state else 0
        end_angle = 90 if expanded else 0
        if animate:
            self.indicator_animation.stop()
            self.indicator_animation.setStartValue(start_angle)
            self.indicator_animation.setEndValue(end_angle)
            self.indicator_animation.start()
        else:
            self.header_button.indicator_angle = end_angle

        if update_state:
            self.main_window.parameter_section_states[self.section_id] = expanded

    def toggle_expanded(self):
        self.set_expanded(not self._expanded)
