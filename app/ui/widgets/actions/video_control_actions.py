from typing import TYPE_CHECKING, cast

import copy
from functools import partial
import os
import traceback

from PySide6.QtCore import QPoint
from PySide6.QtWidgets import QMenu
import cv2
import numpy
from PIL import Image
from PySide6 import QtGui, QtWidgets, QtCore

from app.helpers.typing_helper import ControlTypes, FacesParametersTypes

if TYPE_CHECKING:
    from app.ui.main_ui import MainWindow
import app.helpers.miscellaneous as misc_helpers
from app.ui.widgets.actions import common_actions as common_widget_actions
from app.ui.widgets.actions import graphics_view_actions
import app.ui.widgets.actions.layout_actions as layout_actions
from app.ui.widgets.actions import card_actions
from app.ui.widgets import widget_components


def set_up_video_seek_line_edit(main_window: "MainWindow"):
    video_processor = main_window.video_processor
    videoSeekLineEdit = main_window.videoSeekLineEdit
    videoSeekLineEdit.setAlignment(QtCore.Qt.AlignCenter)
    videoSeekLineEdit.setText("0")
    videoSeekLineEdit.setValidator(
        QtGui.QIntValidator(0, video_processor.max_frame_number)
    )  # Restrict input to numbers


def set_up_video_seek_slider(main_window: "MainWindow"):
    main_window.videoSeekSlider.markers = set()  # Store unique tick positions
    main_window.videoSeekSlider.setTickPosition(
        QtWidgets.QSlider.TickPosition.TicksBelow
    )  # Default position for tick marks

    def add_marker_and_paint(self: QtWidgets.QSlider, value=None):
        """Add a tick mark at a specific slider value."""
        if value is None or isinstance(value, bool):  # Default to current slider value
            value = self.value()
        if self.minimum() <= value <= self.maximum() and value not in self.markers:
            self.markers.add(value)
            self.update()

    def remove_marker_and_paint(self: QtWidgets.QSlider, value=None):
        """Remove a tick mark."""
        if value is None or isinstance(value, bool):  # Default to current slider value
            value = self.value()
        if value in self.markers:
            self.markers.remove(value)
            self.update()

    def paintEvent(self: QtWidgets.QSlider, event: QtGui.QPaintEvent):
        # Do not draw the slider if the current media is a single image
        if main_window.video_processor.file_type == "image":
            return super(QtWidgets.QSlider, self).paintEvent(event)
        # Set up the painter and style option
        painter = QtWidgets.QStylePainter(self)
        opt = QtWidgets.QStyleOptionSlider()
        self.initStyleOption(opt)
        style = self.style()

        # Get groove and handle geometry
        groove_rect = style.subControlRect(
            QtWidgets.QStyle.ComplexControl.CC_Slider,
            opt,
            QtWidgets.QStyle.SubControl.SC_SliderGroove,
        )
        groove_y = (
            groove_rect.top() + groove_rect.bottom()
        ) // 2  # Groove's vertical center
        groove_start = groove_rect.left()
        groove_end = groove_rect.right()
        groove_width = groove_end - groove_start

        # Calculate handle position based on the current slider value
        normalized_value = (self.value() - self.minimum()) / (
            self.maximum() - self.minimum()
        )
        handle_center_x = groove_start + normalized_value * groove_width

        # Make the handle thinner
        handle_width = 5  # Fixed width for thin handle
        handle_height = groove_rect.height()  # Slightly shorter than groove height
        handle_left_x = handle_center_x - (handle_width // 2)
        handle_top_y = groove_y - (handle_height // 2)

        # Define the handle rectangle
        handle_rect = QtCore.QRect(
            handle_left_x, handle_top_y, handle_width, handle_height
        )

        # Draw the groove
        painter.setPen(
            QtGui.QPen(QtGui.QColor("gray"), 3)
        )  # Groove color and thickness
        painter.drawLine(groove_start, groove_y, groove_end, groove_y)

        # Draw the thin handle
        painter.setPen(QtGui.QPen(QtGui.QColor("white"), 1))  # Handle border color
        painter.setBrush(QtGui.QBrush(QtGui.QColor("white")))  # Handle fill color
        painter.drawRect(handle_rect)

        # Draw markers (if any)
        if self.markers:
            painter.setPen(
                QtGui.QPen(QtGui.QColor("#4090a3"), 3)
            )  # Marker color and thickness
            for value in sorted(self.markers):
                # Calculate marker position
                marker_normalized_value = (value - self.minimum()) / (
                    self.maximum() - self.minimum()
                )
                marker_x = groove_start + marker_normalized_value * groove_width
                painter.drawLine(
                    marker_x, groove_rect.top(), marker_x, groove_rect.bottom()
                )
        # Draw Job Start/End Brackets on the groove line
        painter.setFont(
            QtGui.QFont("Arial", 16, QtGui.QFont.Bold)
        )  # Increased font size from 12 to 16
        font_metrics = painter.fontMetrics()
        bracket_height = font_metrics.height()
        bracket_y_pos = groove_y + (bracket_height // 4)

        # Iterate through all defined job marker pairs
        for start_frame, end_frame in main_window.job_marker_pairs:
            if start_frame is not None:
                start_normalized_value = (start_frame - self.minimum()) / (
                    self.maximum() - self.minimum()
                )
                start_x = groove_start + start_normalized_value * groove_width
                # Draw the green start bracket
                painter.setPen(
                    QtGui.QPen(QtGui.QColor("#4CAF50"), 1)
                )  # Green for start bracket
                painter.drawText(
                    int(start_x - 4), int(bracket_y_pos), "["
                )  # Adjusted X offset slightly

            if end_frame is not None:
                end_normalized_value = (end_frame - self.minimum()) / (
                    self.maximum() - self.minimum()
                )
                end_x = groove_start + end_normalized_value * groove_width
                # Draw the red end bracket
                painter.setPen(
                    QtGui.QPen(QtGui.QColor("#e8483c"), 1)
                )  # Red for end bracket
                painter.drawText(
                    int(end_x - 4), int(bracket_y_pos), "]"
                )  # Adjusted X offset slightly

    main_window.videoSeekSlider.add_marker_and_paint = partial(
        add_marker_and_paint, main_window.videoSeekSlider
    )
    main_window.videoSeekSlider.remove_marker_and_paint = partial(
        remove_marker_and_paint, main_window.videoSeekSlider
    )
    main_window.videoSeekSlider.paintEvent = partial(
        paintEvent, main_window.videoSeekSlider
    )


def add_video_slider_marker(main_window: "MainWindow"):
    if (
        not isinstance(
            main_window.selected_video_button, widget_components.TargetMediaCardButton
        )
        or main_window.selected_video_button.file_type != "video"
    ):
        common_widget_actions.create_and_show_messagebox(
            main_window,
            "Markers Not Available",
            "Markers can only be used for videos!",
            main_window.videoSeekSlider,
        )
        return
    current_position = int(main_window.videoSeekSlider.value())
    # print("current_position", current_position)
    if not main_window.target_faces:
        common_widget_actions.create_and_show_messagebox(
            main_window,
            "No Target Face Found",
            "You need to have at least one target face to create a marker",
            main_window.videoSeekSlider,
        )
    elif main_window.markers.get(current_position):
        common_widget_actions.create_and_show_messagebox(
            main_window,
            "Marker Already Exists!",
            "A Marker already exists for this position!",
            main_window.videoSeekSlider,
        )
    else:
        add_marker(
            main_window,
            copy.deepcopy(main_window.parameters),
            main_window.control.copy(),
            current_position,
        )


def show_add_marker_menu(main_window: "MainWindow"):
    """Shows a context menu for adding different types of markers."""
    if (
        not isinstance(
            main_window.selected_video_button, widget_components.TargetMediaCardButton
        )
        or main_window.selected_video_button.file_type != "video"
    ):
        common_widget_actions.create_and_show_messagebox(
            main_window,
            "Markers Not Available",
            "Markers can only be used for videos!",
            main_window.videoSeekSlider,
        )
        return

    button = main_window.addMarkerButton
    menu = QMenu(main_window)

    # Action for standard marker
    add_standard_action = menu.addAction("Add Standard Marker")
    add_standard_action.triggered.connect(lambda: add_video_slider_marker(main_window))

    menu.addSeparator()

    # Determine if the next action should be adding a start or an end marker
    can_add_start = True
    can_add_end = False
    if main_window.job_marker_pairs:
        last_pair = main_window.job_marker_pairs[-1]
        if last_pair[1] is None:  # Last pair is incomplete (start set, end not set)
            can_add_start = False
            can_add_end = True

    # Action for job start marker
    set_start_action = menu.addAction("Add Record Start Marker")
    set_start_action.triggered.connect(lambda: set_job_start_frame(main_window))
    set_start_action.setEnabled(can_add_start)

    # Action for job end marker
    set_end_action = menu.addAction("Add Record End Marker")
    set_end_action.triggered.connect(lambda: set_job_end_frame(main_window))
    set_end_action.setEnabled(can_add_end)

    # Show the menu below the button
    menu.exec(button.mapToGlobal(QPoint(0, button.height())))


def set_job_start_frame(main_window: "MainWindow"):
    """Adds a new job marker pair starting at the current slider position."""
    current_pos = int(main_window.videoSeekSlider.value())

    # Basic validation: Ensure we are not adding a start if the last pair is incomplete
    if main_window.job_marker_pairs and main_window.job_marker_pairs[-1][1] is None:
        QtWidgets.QMessageBox.warning(
            main_window,
            "Invalid Action",
            "Cannot add a new Start marker before completing the previous End marker.",
        )
        return

    # Add the new start marker (end frame is initially None)
    main_window.job_marker_pairs.append((current_pos, None))
    main_window.videoSeekSlider.update()  # Trigger repaint to show the new marker
    print(
        f"Job Start Marker added for pair {len(main_window.job_marker_pairs)} at Frame: {current_pos}"
    )


def set_job_end_frame(main_window: "MainWindow"):
    """Sets the job end frame marker for the last incomplete pair."""
    current_pos = int(main_window.videoSeekSlider.value())

    # Validation: Check if there's an incomplete pair to add an end to
    if (
        not main_window.job_marker_pairs
        or main_window.job_marker_pairs[-1][1] is not None
    ):
        QtWidgets.QMessageBox.critical(
            main_window,
            "Error",
            "Cannot set End marker without a preceding Start marker.",
        )
        return

    last_pair_index = len(main_window.job_marker_pairs) - 1
    start_frame = main_window.job_marker_pairs[last_pair_index][0]

    # Validation: Check end frame is after start frame
    if current_pos <= start_frame:
        QtWidgets.QMessageBox.warning(
            main_window,
            "Invalid Position",
            "Job end frame must be after the job start frame.",
        )
        return

    # Update the last pair with the end frame
    main_window.job_marker_pairs[last_pair_index] = (start_frame, current_pos)
    main_window.videoSeekSlider.update()  # Trigger repaint to show the new marker
    print(
        f"Job End Marker added for pair {last_pair_index + 1} at Frame: {current_pos}"
    )


def remove_video_slider_marker(main_window: "MainWindow"):
    if (
        not isinstance(
            main_window.selected_video_button, widget_components.TargetMediaCardButton
        )
        or main_window.selected_video_button.file_type != "video"
    ):
        common_widget_actions.create_and_show_messagebox(
            main_window,
            "Markers Not Available",
            "Markers can only be used for videos!",
            main_window.videoSeekSlider,
        )
        return

    current_position = int(main_window.videoSeekSlider.value())
    pair_removed = False

    removed_pair_indices = []
    for i, (start_frame, end_frame) in enumerate(main_window.job_marker_pairs):
        if start_frame == current_position or end_frame == current_position:
            print(
                f"Removing Job Marker Pair {i + 1} ({start_frame}, {end_frame}) because marker found at position: {current_position}"
            )
            removed_pair_indices.append(i)
            pair_removed = True

    main_window.job_marker_pairs = [
        pair
        for i, pair in enumerate(main_window.job_marker_pairs)
        if i not in removed_pair_indices
    ]

    if pair_removed:
        main_window.videoSeekSlider.update()
        return

    if main_window.markers.get(current_position):
        remove_marker(main_window, current_position)
    else:
        common_widget_actions.create_and_show_messagebox(
            main_window,
            "No Marker Found!",
            "No Marker Found for this position!",
            main_window.videoSeekSlider,
        )


def add_marker(
    main_window: "MainWindow",
    parameters,
    control,
    position,
):
    main_window.videoSeekSlider.add_marker_and_paint(position)
    main_window.markers[position] = {"parameters": parameters, "control": control}
    print(f"Marker Added for Frame: {position}")


def remove_marker(main_window: "MainWindow", position):
    if main_window.markers.get(position):
        main_window.videoSeekSlider.remove_marker_and_paint(position)
        main_window.markers.pop(position)
        print(f"Marker Removed from position: {position}")


def move_slider_to_nearest_marker(main_window: "MainWindow", direction: str):
    """
    Move the slider to the nearest marker in the specified direction.

    :param direction: 'next' to move to the next marker, 'previous' to move to the previous marker.
    """
    new_position = None
    current_position = int(main_window.videoSeekSlider.value())

    # Combine standard markers with all job start/end markers from pairs
    all_markers = set(main_window.markers.keys())
    for start_frame, end_frame in main_window.job_marker_pairs:
        if start_frame is not None:
            all_markers.add(start_frame)
        if end_frame is not None:
            all_markers.add(end_frame)

    if not all_markers:
        return  # No markers to navigate to

    sorted_markers = sorted(list(all_markers))

    if direction == "next":
        filtered_markers = [
            marker for marker in sorted_markers if marker > current_position
        ]
        new_position = filtered_markers[0] if filtered_markers else None
    elif direction == "previous":
        filtered_markers = [
            marker for marker in sorted_markers if marker < current_position
        ]
        new_position = filtered_markers[-1] if filtered_markers else None

    if new_position is not None:
        main_window.videoSeekSlider.setValue(new_position)
        main_window.video_processor.process_current_frame()


# Wrappers for specific directions
def move_slider_to_next_nearest_marker(main_window: "MainWindow"):
    move_slider_to_nearest_marker(main_window, "next")


def move_slider_to_previous_nearest_marker(main_window: "MainWindow"):
    move_slider_to_nearest_marker(main_window, "previous")


def remove_face_parameters_and_control_from_markers(main_window: "MainWindow", face_id):
    for _, marker_data in main_window.markers.items():
        marker_data["parameters"].pop(
            face_id, None
        )  # Use .pop with default to avoid KeyError
        # If the parameters is empty, then there is no longer any marker to be set for any target face
        if not marker_data["parameters"]:
            delete_all_markers(main_window)
            break


def remove_all_markers(main_window: "MainWindow"):
    standard_markers_positions = list(main_window.markers.keys())
    for marker_position in standard_markers_positions:
        remove_marker(main_window, marker_position)
    main_window.markers.clear()
    if main_window.job_marker_pairs:
        print("Clearing job marker pairs.")
        main_window.job_marker_pairs.clear()


def advance_video_slider_by_n_frames(main_window: "MainWindow", n=30):
    video_processor = main_window.video_processor
    if video_processor.media_capture:
        current_position = int(main_window.videoSeekSlider.value())
        new_position = current_position + n
        if new_position > video_processor.max_frame_number:
            new_position = video_processor.max_frame_number
        main_window.videoSeekSlider.setValue(new_position)

        # Execute post seek (Markers, Autoswap)
        run_post_seek_actions(main_window, new_position)

        main_window.video_processor.process_current_frame()


def rewind_video_slider_by_n_frames(main_window: "MainWindow", n=30):
    video_processor = main_window.video_processor
    if video_processor.media_capture:
        current_position = int(main_window.videoSeekSlider.value())
        new_position = current_position - n
        if new_position < 0:
            new_position = 0
        main_window.videoSeekSlider.setValue(new_position)

        # Execute post seek (Markers, Autoswap)
        run_post_seek_actions(main_window, new_position)

        main_window.video_processor.process_current_frame()


def delete_all_markers(main_window: "MainWindow"):
    main_window.videoSeekSlider.markers = set()
    main_window.videoSeekSlider.update()
    main_window.markers.clear()


def view_fullscreen(main_window: "MainWindow"):
    if main_window.is_full_screen:
        main_window.showNormal()  # Exit full-screen mode
        main_window.menuBar().show()
    else:
        main_window.showFullScreen()  # Enter full-screen mode
        main_window.menuBar().hide()

    main_window.is_full_screen = not main_window.is_full_screen


def enable_zoom_and_pan(view: QtWidgets.QGraphicsView):
    SCALE_FACTOR = 1.1
    view.zoom_value = 0  # Track zoom level
    view.last_scale_factor = 1.0  # Track the last scale factor (1.0 = no scaling)
    view.is_panning = False  # Track whether panning is active
    view.pan_start_pos = QtCore.QPoint()  # Store the initial mouse position for panning

    def zoom(self: QtWidgets.QGraphicsView, step=False):
        """Zoom in or out by a step."""
        if not step:
            factor = self.last_scale_factor
        else:
            self.zoom_value += step
            factor = SCALE_FACTOR**step
            self.last_scale_factor *= factor  # Update the last scale factor
        if factor > 0:
            self.scale(factor, factor)

    def wheelEvent(self: QtWidgets.QGraphicsView, event: QtGui.QWheelEvent):
        """Handle mouse wheel event for zooming."""
        delta = event.angleDelta().y()
        if delta != 0:
            zoom(self, delta // abs(delta))

    def reset_zoom(self: QtWidgets.QGraphicsView):
        # print("Called reset_zoom()")
        # Reset zoom level to fit the view.
        self.zoom_value = 0
        if not self.scene():
            return
        items = self.scene().items()
        if not items:
            return
        rect = self.scene().itemsBoundingRect()
        self.setSceneRect(rect)
        unity = self.transform().mapRect(QtCore.QRectF(0, 0, 1, 1))
        self.scale(1 / unity.width(), 1 / unity.height())
        view_rect = self.viewport().rect()
        scene_rect = self.transform().mapRect(rect)
        factor = min(
            view_rect.width() / scene_rect.width(),
            view_rect.height() / scene_rect.height(),
        )
        self.scale(factor, factor)

    def mousePressEvent(self: QtWidgets.QGraphicsView, event: QtGui.QMouseEvent):
        """Handle mouse press event for panning."""
        if event.button() == QtCore.Qt.MouseButton.RightButton:
            self.is_panning = True
            self.pan_start_pos = event.pos()  # Store the initial mouse position
            self.setCursor(
                QtCore.Qt.ClosedHandCursor
            )  # Change cursor to indicate panning
        else:
            # Explicitly call the base class implementation
            QtWidgets.QGraphicsView.mousePressEvent(self, event)

    def mouseMoveEvent(self: QtWidgets.QGraphicsView, event: QtGui.QMouseEvent):
        """Handle mouse move event for panning."""
        if self.is_panning:
            # Calculate the distance moved
            delta = event.pos() - self.pan_start_pos
            self.pan_start_pos = event.pos()  # Update the start position
            # Translate the view
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta.x()
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - delta.y()
            )
        else:
            # Explicitly call the base class implementation
            QtWidgets.QGraphicsView.mouseMoveEvent(self, event)

    def mouseReleaseEvent(self: QtWidgets.QGraphicsView, event: QtGui.QMouseEvent):
        """Handle mouse release event for panning."""
        if event.button() == QtCore.Qt.MouseButton.RightButton:
            self.is_panning = False
            self.setCursor(QtCore.Qt.ArrowCursor)  # Reset the cursor
        else:
            # Explicitly call the base class implementation
            QtWidgets.QGraphicsView.mouseReleaseEvent(self, event)

    # Attach methods to the view
    view.zoom = partial(zoom, view)
    view.reset_zoom = partial(reset_zoom, view)
    view.wheelEvent = partial(wheelEvent, view)
    view.mousePressEvent = partial(mousePressEvent, view)
    view.mouseMoveEvent = partial(mouseMoveEvent, view)
    view.mouseReleaseEvent = partial(mouseReleaseEvent, view)

    # view.zoom = zoom.__get__(view)
    # view.reset_zoom = reset_zoom.__get__(view)
    # view.wheelEvent = wheelEvent.__get__(view)

    # Set anchors for better interaction
    view.setTransformationAnchor(
        QtWidgets.QGraphicsView.ViewportAnchor.AnchorUnderMouse
    )
    view.setResizeAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorUnderMouse)


def play_video(main_window: "MainWindow", checked: bool):
    video_processor = main_window.video_processor
    if checked and video_processor.file_type == "webcam":
        if video_processor.processing:
            print(
                "play_video: Webcam already streaming. Stopping the stream before restarting."
            )
            video_processor.stop_processing()
        print("play_video: Starting webcam stream processing.")
        set_play_button_icon_to_stop(main_window)
        video_processor.process_webcam()
        return
    if checked:
        if (
            video_processor.processing
            or video_processor.current_frame_number == video_processor.max_frame_number
        ):
            print(
                "play_video: Video already playing. Stopping the current video before starting a new one."
            )
            video_processor.stop_processing()
            return
        print("play_video: Starting video processing.")
        set_play_button_icon_to_stop(main_window)
        video_processor.process_video()
    else:
        video_processor = main_window.video_processor
        # print("play_video: Stopping video processing.")
        set_play_button_icon_to_play(main_window)
        video_processor.stop_processing()
        main_window.buttonMediaRecord.blockSignals(True)
        main_window.buttonMediaRecord.setChecked(False)
        main_window.buttonMediaRecord.blockSignals(False)
        set_record_button_icon_to_play(main_window)


def record_video(main_window: "MainWindow", checked: bool):
    video_processor = main_window.video_processor
    # Determine if this record action was initiated by the Job Manager
    job_mgr_flag = getattr(main_window, "job_manager_initiated_record", False)

    # Check if Batch processing
    is_batch_processing = getattr(main_window, "is_batch_processing", False)

    if video_processor.file_type not in ["video", "image"]:
        main_window.buttonMediaRecord.blockSignals(True)
        main_window.buttonMediaRecord.setChecked(False)
        main_window.buttonMediaRecord.blockSignals(False)
        if video_processor.file_type == "webcam":
            common_widget_actions.create_and_show_messagebox(
                main_window,
                "Recording Not Supported",
                "Recording webcam stream is not supported yet.",
                main_window,
            )
        return

    if checked:
        if video_processor.processing or video_processor.is_processing_segments:
            print("record_video: Processing already active. Request ignored.")
            main_window.buttonMediaRecord.blockSignals(True)
            main_window.buttonMediaRecord.setChecked(True)
            main_window.buttonMediaRecord.blockSignals(False)
            set_record_button_icon_to_stop(main_window)
            return
        if not str(main_window.control.get("OutputMediaFolder", "")).strip():
            common_widget_actions.create_and_show_messagebox(
                main_window,
                "No Output Folder Selected",
                "Please select an Output folder before recording!",
                main_window,
            )
            main_window.buttonMediaRecord.setChecked(False)  # Uncheck the button
            return
        if not misc_helpers.is_ffmpeg_in_path():
            common_widget_actions.create_and_show_messagebox(
                main_window,
                "FFMPEG Not Found",
                "FFMPEG was not found in your system. Check installation!",
                main_window,
            )
            main_window.buttonMediaRecord.setChecked(False)  # Uncheck the button
            return

        marker_pairs = main_window.job_marker_pairs

        # If no Markers OR Batch processing, use default style recording
        if not marker_pairs or (
            is_batch_processing and video_processor.file_type == "video"
        ):
            # --- Default Recording Style ---

            # If not Batch processing, use slider position
            # If Batch processing, use frame position 0
            current_frame = 0
            if not is_batch_processing:
                current_frame = main_window.videoSeekSlider.value()

            max_frame = video_processor.max_frame_number
            if max_frame is None or max_frame <= 0:
                common_widget_actions.create_and_show_messagebox(
                    main_window, "Error", "Cannot determine video length.", main_window
                )
                main_window.buttonMediaRecord.setChecked(False)
                return

            # Check start position (0 for batch, slider for manual)
            if current_frame >= max_frame:
                common_widget_actions.create_and_show_messagebox(
                    main_window,
                    "Recording Error",
                    f"Cannot start recording from frame {current_frame}. Scrubber is at or past the end of the video ({max_frame}).",
                    main_window,
                )
                main_window.buttonMediaRecord.setChecked(False)
                return
            # --- Proceed with Default Recording ---
            print(
                "Record button pressed: Starting default recording (full video or from slider)."
            )
            set_record_button_icon_to_stop(main_window)
            # Disable play button during recording
            main_window.buttonMediaPlay.setEnabled(False)
            video_processor.recording = True  # SET THE FLAG FOR DEFAULT RECORDING

            # If batch process force slider to 0 so that process_video() begins at start.
            if is_batch_processing:
                main_window.videoSeekSlider.blockSignals(True)
                main_window.videoSeekSlider.setValue(0)
                main_window.videoSeekSlider.blockSignals(False)
                print("Batch processing: Forcing video record to start from frame 0.")

            video_processor.process_video()  # CALL THE DEFAULT PROCESSOR

        else:  # MARKERS ARE SET (and not in batch) -> Multi-Segment Recording Style
            # --- Validate Marker Pairs ---
            valid_pairs = []
            for i, pair in enumerate(marker_pairs):
                if pair[1] is None:
                    common_widget_actions.create_and_show_messagebox(
                        main_window,
                        "Incomplete Segment",
                        f"Marker pair {i + 1} ({pair[0]}, None) is incomplete. Please set an End marker.",
                        main_window,
                    )
                    main_window.buttonMediaRecord.setChecked(False)
                    return  # Stop if invalid
                elif pair[0] >= pair[1]:
                    common_widget_actions.create_and_show_messagebox(
                        main_window,
                        "Invalid Segment",
                        f"Marker pair {i + 1} ({pair[0]}, {pair[1]}) is invalid. Start must be before End.",
                        main_window,
                    )
                    main_window.buttonMediaRecord.setChecked(False)
                    return  # Stop if invalid
                else:
                    valid_pairs.append(pair)
            # --- End Validation ---

            # Proceed if we have valid marker pairs
            if valid_pairs:
                print(
                    f"Record button pressed: Starting multi-segment recording for {len(valid_pairs)} segment(s)."
                )
                set_record_button_icon_to_stop(main_window)
                # Disable play button during segment recording
                main_window.buttonMediaPlay.setEnabled(False)
                is_job_context = job_mgr_flag
                print(f"[DEBUG] record_video: job_mgr_flag = {is_job_context}")
                video_processor.start_multi_segment_recording(
                    valid_pairs, triggered_by_job_manager=is_job_context
                )
                try:
                    main_window.job_manager_initiated_record = False
                except Exception:
                    pass
            else:
                print(
                    "[WARN] Recording not started due to invalid marker configuration."
                )

    else:
        if video_processor.is_processing_segments:
            print(
                "Record button released: User requested stop during segment processing. Finalizing..."
            )
            # Finalize segment concatenation with segments processed so far
            video_processor.finalize_segment_concatenation()
        elif video_processor.recording:  # Check if default style recording was active
            print(
                "Record button released: User requested stop during default recording. Finalizing..."
            )
            # Finalize the default style recording
            video_processor._finalize_default_style_recording()
        else:
            # No recording was active (maybe an immediate click-off or already stopped)
            print("Record button released: No active recording found.")
            set_record_button_icon_to_play(main_window)
            main_window.buttonMediaPlay.setEnabled(True)
            reset_media_buttons(main_window)


def set_record_button_icon_to_play(main_window: "MainWindow"):
    main_window.buttonMediaRecord.setIcon(QtGui.QIcon(":/media/media/rec_off.png"))
    main_window.buttonMediaRecord.setToolTip("Start Recording")


def set_record_button_icon_to_stop(main_window: "MainWindow"):
    main_window.buttonMediaRecord.setIcon(QtGui.QIcon(":/media/media/rec_on.png"))
    main_window.buttonMediaRecord.setToolTip("Stop Recording")


def set_play_button_icon_to_play(main_window: "MainWindow"):
    main_window.buttonMediaPlay.setIcon(QtGui.QIcon(":/media/media/play_off.png"))
    main_window.buttonMediaPlay.setToolTip("Play")


def set_play_button_icon_to_stop(main_window: "MainWindow"):
    main_window.buttonMediaPlay.setIcon(QtGui.QIcon(":/media/media/play_on.png"))
    main_window.buttonMediaPlay.setToolTip("Stop")


def reset_media_buttons(main_window: "MainWindow"):
    # Rest the state and icons of the buttons without triggering Onchange methods
    main_window.buttonMediaPlay.blockSignals(True)
    main_window.buttonMediaPlay.setChecked(False)
    main_window.buttonMediaPlay.setEnabled(True)  # Re-enable the button
    main_window.buttonMediaPlay.blockSignals(False)
    main_window.buttonMediaRecord.blockSignals(True)
    main_window.buttonMediaRecord.setChecked(False)
    main_window.buttonMediaRecord.blockSignals(False)
    set_play_button_icon(main_window)
    set_record_button_icon(main_window)


def set_play_button_icon(main_window: "MainWindow"):
    if main_window.buttonMediaPlay.isChecked():
        main_window.buttonMediaPlay.setIcon(QtGui.QIcon(":/media/media/play_on.png"))
        main_window.buttonMediaPlay.setToolTip("Stop")
    else:
        main_window.buttonMediaPlay.setIcon(QtGui.QIcon(":/media/media/play_off.png"))
        main_window.buttonMediaPlay.setToolTip("Play")


def set_record_button_icon(main_window: "MainWindow"):
    if main_window.buttonMediaRecord.isChecked():
        main_window.buttonMediaRecord.setIcon(QtGui.QIcon(":/media/media/rec_on.png"))
        main_window.buttonMediaRecord.setToolTip("Stop Recording")
    else:
        main_window.buttonMediaRecord.setIcon(QtGui.QIcon(":/media/media/rec_off.png"))
        main_window.buttonMediaRecord.setToolTip("Start Recording")


# @misc_helpers.benchmark
@QtCore.Slot(int)
def on_change_video_seek_slider(main_window: "MainWindow", new_position=0):
    # print("Called on_change_video_seek_slider()")
    video_processor = main_window.video_processor

    was_processing = video_processor.stop_processing()
    if was_processing:
        print(
            "on_change_video_seek_slider: Processing in progress. Stopping current processing."
        )

    video_processor.current_frame_number = new_position
    video_processor.next_frame_to_display = new_position
    if video_processor.media_capture:
        misc_helpers.seek_frame(video_processor.media_capture, new_position)

        # Read the raw frame without triggering the full pipeline.
        ret, frame = misc_helpers.read_frame(video_processor.media_capture)
        if ret:
            # For preview, show the raw frame immediately.
            # The processed frame will be shown when the slider is released.
            pixmap = common_widget_actions.get_pixmap_from_frame(main_window, frame)
            graphics_view_actions.update_graphics_view(
                main_window, pixmap, new_position
            )

        else:
            main_window.last_seek_read_failed = True
    # Only update parameters and widgets if the slider is NOT being actively dragged.
    # This ensures playback, clicks, and button presses update the UI,
    # but fast scrubbing does not cause lag or skip marker updates.
    if not main_window.videoSeekSlider.isSliderDown():
        run_post_seek_actions(main_window, new_position)
    # Do not automatically restart the video, let the user press Play to resume
    # print("on_change_video_seek_slider: Video stopped after slider movement.")


def _get_marker_data_for_position(
    main_window: "MainWindow", new_position: int
) -> dict | None:
    """
    Finds the marker data that should be active at a given frame position.
    It looks for the marker at the exact position, or the nearest one *before* it.
    """
    if not main_window.markers:
        return None

    # 1. Check for an exact match first (most common case for playback/buttons)
    if new_position in main_window.markers:
        return main_window.markers.get(new_position)

    # 2. If no exact match, find the last marker *before* this position
    # Get all marker keys that are less than or equal to the current position
    relevant_marker_keys = [k for k in main_window.markers.keys() if k <= new_position]

    if not relevant_marker_keys:
        # No markers at or before this position
        return None

    # 3. Get the most recent (largest) key from that list
    last_marker_key = max(relevant_marker_keys)
    return main_window.markers.get(last_marker_key)


def update_parameters_and_control_from_marker(
    main_window: "MainWindow", new_position: int
):
    # Find marker only at the *exact* new position
    marker_data = _get_marker_data_for_position(main_window, new_position)  # New logic
    # Save the Global Marker Track toggle
    current_track_markers_value = main_window.control.get("TrackMarkersToggle", False)

    if marker_data:
        # --- A marker was found, load its parameters AND controls ---

        # Load Parameters (Full Replacement)
        loaded_marker_params: FacesParametersTypes = copy.deepcopy(
            marker_data["parameters"]
        )
        main_window.parameters = loaded_marker_params

        # Ensure parameter dicts exist for all *current* faces
        active_target_face_ids = list(main_window.target_faces.keys())
        for face_id_key in active_target_face_ids:
            if str(face_id_key) not in main_window.parameters:
                common_widget_actions.create_parameter_dict_for_face_id(
                    main_window, str(face_id_key)
                )

        # Load Controls (Full Replacement)
        if "control" in marker_data:
            control_data = marker_data["control"]
            if isinstance(control_data, dict):
                # We must do a full replacement, not just an update.
                # First, reset all controls to their default values.
                for widget_name, widget in main_window.parameter_widgets.items():
                    if widget_name in main_window.control:  # It's a control widget
                        main_window.control[widget_name] = widget.default_value

                # Now, apply the marker's specific controls
                main_window.control.update(cast(ControlTypes, control_data).copy())

    # If no marker_data is found, DO NOTHING.
    # This preserves the user's current settings (manual or from a previous marker).
    # Re-apply the saved Global Marker Track toggle
    main_window.control["TrackMarkersToggle"] = current_track_markers_value


def update_widget_values_from_markers(main_window: "MainWindow", new_position: int):
    # 1. Update Parameter-based widgets (Face Swap, Editor, Restorers)
    if main_window.selected_target_face_id is not None:
        common_widget_actions.set_widgets_values_using_face_id_parameters(
            main_window, main_window.selected_target_face_id
        )
    else:
        # If no face is selected, update widgets to the "current" state
        # (which might be default or from the last marker).
        common_widget_actions.set_widgets_values_using_face_id_parameters(
            main_window, False
        )

    # 2. Update Control-based widgets (Settings, Denoiser)
    common_widget_actions.set_control_widgets_values(
        main_window, enable_exec_func=False
    )


def on_slider_moved(main_window: "MainWindow"):
    # print("Called on_slider_moved()")
    main_window.videoSeekSlider.value()
    # print(f"\nSlider Moved. position: {position}\n")


def on_slider_pressed(main_window: "MainWindow"):
    main_window.videoSeekSlider.value()
    # print(f"\nSlider Pressed. position: {position}\n")


def run_post_seek_actions(main_window: "MainWindow", new_position: int):
    """
    Executes heavy operations (markers, AutoSwap) after a seek.
    This function is called after a slider release or a jump via button/shortcut.
    """

    # Check if the user wants to update the UI based on markers when seeking
    track_markers_enabled = main_window.control.get("TrackMarkersToggle", False)

    if track_markers_enabled:
        # 1. Update parameters if the slider lands on a marker
        # Acquire lock to safely modify parameters read by worker threads
        with main_window.models_processor.model_lock:
            update_parameters_and_control_from_marker(main_window, new_position)
            update_widget_values_from_markers(main_window, new_position)
    # If tracking is disabled, we do nothing, preserving the user's manual changes.

    # 2. If AutoSwap is enabled, run face detection/matching NOW.
    # This is independent of marker tracking.
    if main_window.control.get("AutoSwapToggle", False):
        # Find new faces and add them to the target list.
        card_actions.find_target_faces(main_window)

        # This block is necessary to auto-select the first face and assign inputs.
        if main_window.target_faces and not main_window.selected_target_face_id:
            list(main_window.target_faces.values())[0].click()


# @misc_helpers.benchmark
def on_slider_released(main_window: "MainWindow"):
    """
    This function is connected to the sliderReleased signal.
    It triggers
    the full processing pipeline ONLY AFTER the user has finished dragging.
    """
    # print("Called on_slider_released()")

    new_position = main_window.videoSeekSlider.value()  # Get the final position
    # print(f"\nSlider released. New position: {new_position}\n")

    video_processor = main_window.video_processor
    if video_processor.media_capture:
        # Execute post seek (Markers, Autoswap)
        # Run post-seek actions ONCE on slider release to apply
        # the parameters for the final frame position.
        run_post_seek_actions(main_window, new_position)

        # This is the heavy processing call that runs the AI models (swap, etc.)
        # It will now use the correct faces and parameters from the functions above.
        video_processor.process_current_frame()


def process_swap_faces(main_window: "MainWindow"):
    video_processor = main_window.video_processor
    video_processor.process_current_frame()


def process_edit_faces(main_window: "MainWindow"):
    video_processor = main_window.video_processor
    video_processor.process_current_frame()


def process_compare_checkboxes(main_window: "MainWindow"):
    main_window.video_processor.process_current_frame()
    layout_actions.fit_image_to_view_onchange(main_window)


def save_current_frame_to_file(main_window: "MainWindow"):
    if not main_window.outputFolderLineEdit.text():
        common_widget_actions.create_and_show_messagebox(
            main_window,
            "No Output Folder Selected",
            "Please select an Output folder to save the Images/Videos before Saving/Recording!",
            main_window,
        )
        return
    frame = main_window.video_processor.current_frame.copy()
    image_format = "image"
    if main_window.control["ImageFormatToggle"]:
        image_format = "jpegimage"

    if isinstance(frame, numpy.ndarray):
        save_filename = misc_helpers.get_output_file_path(
            main_window.video_processor.media_path,
            str(main_window.control["OutputMediaFolder"]),
            media_type=image_format,
        )
        if save_filename:
            # frame is main_window.video_processor.current_frame, which is already RGB.
            frame = frame[..., ::-1]
            pil_image = Image.fromarray(
                frame
            )  # Correct: Pass RGB frame directly to Pillow.
            if main_window.control["ImageFormatToggle"]:
                pil_image.save(save_filename, "JPEG", quality=95)
            else:
                pil_image.save(save_filename, "PNG")
            common_widget_actions.create_and_show_toast_message(
                main_window,
                "Image Saved",
                f"Saved Current Image to file: {save_filename}",
            )

    else:
        common_widget_actions.create_and_show_messagebox(
            main_window,
            "Invalid Frame",
            "Cannot save the current frame!",
            parent_widget=main_window.saveImageButton,
        )


def process_batch_images(main_window: "MainWindow", process_all_faces: bool):
    """
    Processes a batch of images and/or videos from the target list.

    If 'process_all_faces' is True (Batch Process All Faces):
    - Processes ONLY images.
    - Finds all faces in each image and applies current settings.

    If 'process_all_faces' is False (Batch Process Selected Face):
    - Processes BOTH images and videos.
    - Images: Applies current UI settings (target faces, inputs) to the image.
    - Videos: Applies current UI settings (markers, inputs, etc.) by running a full 'record' operation.
    """
    # 1. Check if output folder is set
    if not main_window.outputFolderLineEdit.text():
        common_widget_actions.create_and_show_messagebox(
            main_window,
            "No Output Folder Selected",
            "Please select an Output folder to save the Images/Videos before Saving/Recording!",
            main_window,
        )
        return

    # 2. Collect all media paths from the target list
    media_files_to_process = []  # List of tuples: (media_path, file_type)
    num_images = 0
    num_videos = 0

    # Iterate through the TargetVideosList to find items
    for i in range(main_window.targetVideosList.count()):
        item = main_window.targetVideosList.item(i)
        widget = main_window.targetVideosList.itemWidget(item)
        if not widget:
            continue

        file_type = widget.file_type
        media_path = widget.media_path

        if process_all_faces:
            # "All Faces" mode only processes images
            if file_type == "image":
                media_files_to_process.append((media_path, file_type))
                num_images += 1
        else:
            # "Selected Face" mode processes images AND videos
            if file_type == "image":
                media_files_to_process.append((media_path, file_type))
                num_images += 1
            elif file_type == "video":
                media_files_to_process.append((media_path, file_type))
                num_videos += 1

    # 3. Check if any media files were found
    if not media_files_to_process:
        common_widget_actions.create_and_show_messagebox(
            main_window,
            "No Media Found",
            "No compatible images (or videos) found in the target list to process.",
            main_window,
        )
        return

    # We need a copy of the current UI parameters (for 'all faces' mode)
    saved_current_parameters = None
    if process_all_faces:
        if main_window.current_widget_parameters is not None:
            saved_current_parameters = main_window.current_widget_parameters.copy()

    # Get the currently selected source faces and embeddings (for both modes)
    saved_input_faces = [
        face for face in main_window.input_faces.values() if face.isChecked()
    ]
    saved_embeddings = [
        embed for embed in main_window.merged_embeddings.values() if embed.isChecked()
    ]

    # Check if at least one source is selected (for both modes)
    if not saved_input_faces and not saved_embeddings:
        common_widget_actions.create_and_show_messagebox(
            main_window,
            "No Input Faces",
            "Please select at least one Input Face or Embedding to use for swapping.",
            main_window,
        )
        return

    # 4. Confirmation Dialog (MODIFIED)
    if process_all_faces:
        confirm_title = "Confirm Batch Swap (All Faces)"
        confirm_msg = (
            f"Found {num_images} images in the target list.\n\n"
            "This will find ALL faces in each image and process them using the "
            "currently selected input faces and parameters.\n\n"
            "Proceed with batch swap?"
        )
    else:
        confirm_title = "Confirm Batch Swap (Current Config)"
        confirm_msg = (
            f"Found {num_images} images and {num_videos} videos in the target list.\n\n"
            "This will process each item using the current UI configuration "
            "(Using the selected inputs faces and parameters on the selected Target Face).\n\n"
            "Proceed with batch swap?"
        )

    reply = QtWidgets.QMessageBox.question(
        main_window,
        confirm_title,
        confirm_msg,
        QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
        QtWidgets.QMessageBox.No,
    )

    if reply == QtWidgets.QMessageBox.No:
        return

    # 5. Store original state to restore it later
    original_media_path = main_window.video_processor.media_path
    original_file_type = main_window.video_processor.file_type
    original_frame_num = main_window.video_processor.current_frame_number

    # Store target faces only if NOT in 'all faces' mode, otherwise they get cleared
    original_target_faces = {}
    if not process_all_faces:
        original_target_faces = main_window.target_faces.copy()

    # 6. Setup Progress Dialog
    progress_dialog = QtWidgets.QProgressDialog(
        "Starting batch processing...",
        "Cancel",
        0,
        len(media_files_to_process),
        main_window,
    )
    progress_dialog.setWindowModality(QtCore.Qt.WindowModal)
    progress_dialog.setWindowTitle("Batch Processing Media")
    progress_dialog.setValue(0)
    progress_dialog.show()

    processed_count = 0
    failed_count = 0
    main_window.is_batch_processing = True

    try:
        # 7. Processing Loop
        for i, (media_path, file_type) in enumerate(media_files_to_process):
            # Update progress and check for cancellation
            progress_dialog.setValue(i)
            progress_dialog.setLabelText(f"Processing: {os.path.basename(media_path)}")
            QtWidgets.QApplication.processEvents()  # Keep UI responsive

            if progress_dialog.wasCanceled():
                break

            try:
                # --- PROCESSING LOGIC ---

                # 7a.
                # Set the video_processor state to the new media
                main_window.video_processor.media_path = media_path
                main_window.video_processor.file_type = file_type

                # --- Minimal Media Load ---
                # Release previous capture if it exists
                if main_window.video_processor.media_capture:
                    misc_helpers.release_capture(
                        main_window.video_processor.media_capture
                    )
                    main_window.video_processor.media_capture = None

                frame_bgr = None
                if file_type == "image":
                    frame_bgr = misc_helpers.read_image_file(media_path)
                    main_window.video_processor.max_frame_number = 0
                    main_window.video_processor.fps = 0
                    # [FIX 1] Update the slider for images
                    main_window.videoSeekSlider.setMaximum(0)

                elif file_type == "video":
                    media_capture = cv2.VideoCapture(media_path)
                    if not media_capture.isOpened():
                        raise Exception(f"Could not open video file: {media_path}")

                    main_window.video_processor.media_capture = media_capture
                    max_frames = int(media_capture.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
                    main_window.video_processor.max_frame_number = max_frames
                    main_window.video_processor.fps = media_capture.get(
                        cv2.CAP_PROP_FPS
                    )

                    # [FIX 1] Update the slider for this video
                    main_window.videoSeekSlider.blockSignals(True)
                    main_window.videoSeekSlider.setMaximum(max_frames)
                    main_window.videoSeekSlider.blockSignals(False)

                    ret, frame_bgr = misc_helpers.read_frame(media_capture)
                    if ret:
                        # Reset capture to frame 0 for processing
                        misc_helpers.seek_frame(media_capture, 0)

                if frame_bgr is None:
                    raise Exception(f"Could not read first frame from: {media_path}")

                # Set current_frame for processor use (e.g., FFmpeg dimensions)
                main_window.video_processor.current_frame = frame_bgr[
                    ..., ::-1
                ]  # BGR to RGB
                main_window.video_processor.current_frame_number = 0
                # --- End Minimal Media Load ---

                if file_type == "image":
                    # --- IMAGE PROCESSING ---
                    if process_all_faces:
                        # --- "ALL FACES" LOGIC ---
                        card_actions.clear_target_faces(
                            main_window, refresh_frame=False
                        )
                        card_actions.find_target_faces(main_window)
                        for target_face in main_window.target_faces.values():
                            if saved_current_parameters:
                                main_window.parameters[target_face.face_id] = (
                                    saved_current_parameters.copy()
                                )
                            for input_face in saved_input_faces:
                                target_face.assigned_input_faces[input_face.face_id] = (
                                    input_face.embedding_store
                                )
                            for embed in saved_embeddings:
                                target_face.assigned_merged_embeddings[
                                    embed.embedding_id
                                ] = embed.embedding_store
                            target_face.calculate_assigned_input_embedding()

                        main_window.video_processor.process_current_frame(
                            synchronous=True
                        )

                    else:
                        # --- "CURRENT CONFIG" LOGIC (Image) ---
                        main_window.video_processor.process_current_frame(
                            synchronous=True
                        )

                    # --- Get and save the processed image ---
                    # [MODIFICATION] - Le 'current_frame' du processeur est en BGR
                    frame = main_window.video_processor.current_frame
                    if not isinstance(frame, numpy.ndarray) or frame.size == 0:
                        frame_bgr_fallback = misc_helpers.read_image_file(media_path)
                        if frame_bgr_fallback is not None:
                            print(
                                f"[WARN] Processing returned an empty frame for {media_path}. Saving original image instead."
                            )
                            frame = frame_bgr_fallback  # Use BGR
                        else:
                            raise Exception(
                                "Processing returned an invalid frame and original could not be read."
                            )

                    image_format = "image"
                    if main_window.control["ImageFormatToggle"]:
                        image_format = "jpegimage"
                    save_filename = misc_helpers.get_output_file_path(
                        media_path,
                        str(main_window.control["OutputMediaFolder"]),
                        media_type=image_format,
                    )

                    if save_filename:
                        # 'frame' is BGR (from processor or fallback read)
                        # PIL needs RGB
                        pil_image = Image.fromarray(
                            frame[..., ::-1]
                        )  # Convert BGR -> RGB for PIL
                        if main_window.control["ImageFormatToggle"]:
                            pil_image.save(save_filename, "JPEG", quality=95)
                        else:
                            pil_image.save(save_filename, "PNG")
                        processed_count += 1
                    else:
                        raise Exception("Could not generate output filename.")

                elif file_type == "video":
                    # --- VIDEO PROCESSING (Selected Face Mode Only) ---
                    # This will use the markers, inputs, etc., currently set in the UI
                    # and will block until the video is fully processed and saved.

                    # 1. Trigger the recording. This will start the async process.
                    record_video(main_window, True)

                    # 2. Wait for the processing to finish.
                    # This loop now checks for cancellation
                    while (
                        main_window.video_processor.processing
                        or main_window.video_processor.is_processing_segments
                    ):
                        QtWidgets.QApplication.processEvents()  # Process UI events (like cancel button)

                        # Check for cancellation *inside* the video wait loop
                        if progress_dialog.wasCanceled():
                            print(
                                f"Cancel detected during video processing: {media_path}. Aborting..."
                            )
                            # Call the 'abort' function
                            main_window.video_processor.stop_processing()
                            # 'stop_processing' sets .processing to False,
                            # so the loop will exit.
                            break

                        QtCore.QThread.msleep(1)  # 2ms sleep

                    # 3. At this point, record_video has completed (or been aborted)
                    # We must check *again* if the loop was exited due to cancellation
                    # to avoid incorrectly incrementing the 'processed_count'.
                    if not progress_dialog.wasCanceled():
                        print(f"Finished processing video: {media_path}")
                        processed_count += 1
                    else:
                        print(f"Video processing was cancelled for: {media_path}")

            except Exception as e:
                # Log the error for this specific file and continue
                print(f"Failed to process {media_path}: {e}")
                traceback.print_exc()
                failed_count += 1
                # Ensure processing is stopped if an error occurred
                if (
                    main_window.video_processor.processing
                    or main_window.video_processor.is_processing_segments
                ):
                    main_window.video_processor.stop_processing()

    finally:
        main_window.is_batch_processing = False
        # 8. Close the progress dialog
        progress_dialog.close()

        # 9. Show completion message
        if progress_dialog.wasCanceled():
            result_msg = (
                f"Batch processing cancelled.\n\n"
                f"Processed: {processed_count}\n"
                f"Failed: {failed_count}"
            )
        else:
            result_msg = (
                f"Batch processing complete.\n\n"
                f"Successfully processed: {processed_count}\n"
                f"Failed to process: {failed_count}"
            )

        common_widget_actions.create_and_show_messagebox(
            main_window, "Batch Complete", result_msg, main_window
        )

        # 10. Restore original state

        # Clear faces from the last processed image IF we were in 'all faces' mode
        if process_all_faces:
            card_actions.clear_target_faces(main_window, refresh_frame=False)
        else:
            # Otherwise, restore the original target faces
            main_window.target_faces = original_target_faces

        # Release the last media capture from the batch
        if main_window.video_processor.media_capture:
            misc_helpers.release_capture(main_window.video_processor.media_capture)

        main_window.video_processor.media_path = original_media_path
        main_window.video_processor.file_type = original_file_type
        main_window.video_processor.current_frame_number = original_frame_num

        # [FIX 3] Do not restore the old capture object
        main_window.video_processor.media_capture = None  # original_media_capture

        # Restore the view to its original state
        if main_window.video_processor.media_path:
            # Reload and re-process the frame that was active before the batch
            if main_window.video_processor.file_type == "video":
                # --- MODIFICATION: Re-open the original video capture ---
                print(f"Restoring original video capture: {original_media_path}")
                new_capture = cv2.VideoCapture(original_media_path)
                if new_capture and new_capture.isOpened():
                    main_window.video_processor.media_capture = new_capture
                    # Set the slider max back to the original video's max
                    original_max_frames = (
                        int(new_capture.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
                    )
                    main_window.videoSeekSlider.blockSignals(True)
                    main_window.videoSeekSlider.setMaximum(original_max_frames)
                    main_window.videoSeekSlider.setValue(original_frame_num)
                    main_window.videoSeekSlider.blockSignals(False)
                    main_window.video_processor.max_frame_number = original_max_frames
                else:
                    print(
                        f"[ERROR] Failed to re-open original media capture: {original_media_path}"
                    )
                # --- FIN MODIFICATION ---
                main_window.video_processor.process_current_frame(synchronous=True)

            elif main_window.video_processor.file_type == "image":
                # [MODIFICATION] Correctly restore the slider for the image
                main_window.videoSeekSlider.blockSignals(True)
                main_window.videoSeekSlider.setMaximum(0)
                main_window.videoSeekSlider.setValue(0)
                main_window.videoSeekSlider.blockSignals(False)
                main_window.video_processor.max_frame_number = 0
                main_window.video_processor.process_current_frame(synchronous=True)
        else:
            # If no media was loaded, clear the scene
            main_window.scene.clear()
            # Manually update graphics view to show nothing
            graphics_view_actions.update_graphics_view(main_window, QtGui.QPixmap(), 0)
            # [MODIFICATION] Reset the slider
            main_window.videoSeekSlider.blockSignals(True)
            main_window.videoSeekSlider.setMaximum(0)
            main_window.videoSeekSlider.setValue(0)
            main_window.videoSeekSlider.blockSignals(False)
            main_window.video_processor.max_frame_number = 0


def toggle_live_sound(main_window: "MainWindow", toggle_value: bool):
    video_processor = main_window.video_processor
    was_processing = video_processor.processing

    # If the video was playing, then stop and start it again to enable the audio
    # Otherwise, just the toggle value so that the next time the play button is hit, it would automatically enable/disable the audio
    # The play button is clicked twice in the below block to simulate the above mentioned behaviour. It should be changed into a set up in the next refactor
    if was_processing:
        main_window.buttonMediaPlay.click()
        main_window.buttonMediaPlay.click()
