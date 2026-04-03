from typing import TYPE_CHECKING, cast

import copy
from functools import partial
import os
from pathlib import Path
import traceback

from PySide6.QtCore import QPoint
from PySide6.QtWidgets import QMenu
import cv2
import numpy
from PIL import Image
from PySide6 import QtGui, QtWidgets, QtCore

from app.helpers.typing_helper import ControlTypes, FacesParametersTypes, MarkerData
from app.helpers.miscellaneous import get_video_rotation

if TYPE_CHECKING:
    from app.ui.main_ui import MainWindow
import app.helpers.miscellaneous as misc_helpers
from app.ui.widgets.actions import common_actions as common_widget_actions
from app.ui.widgets.actions import graphics_view_actions
import app.ui.widgets.actions.layout_actions as layout_actions
from app.ui.widgets.actions import card_actions
from app.ui.widgets import widget_components
from app.ui.widgets import ui_workers


def set_up_video_seek_line_edit(main_window: "MainWindow"):
    """Configures the video seek line-edit widget: centres text and restricts input to valid frame numbers."""
    video_processor = main_window.video_processor
    videoSeekLineEdit = main_window.videoSeekLineEdit
    videoSeekLineEdit.setAlignment(QtCore.Qt.AlignCenter)
    videoSeekLineEdit.setText("0")
    videoSeekLineEdit.setValidator(
        QtGui.QIntValidator(0, video_processor.max_frame_number)
    )  # Restrict input to numbers


def set_up_video_seek_slider(main_window: "MainWindow"):
    """
    Configures the video seek slider with custom painting, marker management, and
    job-bracket rendering.  Attaches add_marker_and_paint, remove_marker_and_paint,
    and a custom paintEvent directly to the slider instance.
    """
    main_window.videoSeekSlider.markers = set()  # Store unique tick positions
    main_window.videoSeekSlider.markers_sorted = []  # Sorted list for iteration in paintEvent
    main_window.videoSeekSlider.issue_markers = set()
    main_window.videoSeekSlider.issue_markers_sorted = []
    main_window.videoSeekSlider.dropped_markers = set()
    main_window.videoSeekSlider.dropped_markers_sorted = []
    main_window.videoSeekSlider.setTickPosition(
        QtWidgets.QSlider.TickPosition.TicksBelow
    )

    def add_marker_and_paint(self: QtWidgets.QSlider, value=None):
        """Add a tick mark at a specific slider value."""
        if value is None or isinstance(value, bool):  # Default to current slider value
            value = self.value()
        if self.minimum() <= value <= self.maximum() and value not in self.markers:
            self.markers.add(value)
            if value not in self.markers_sorted:
                self.markers_sorted.append(value)
                self.markers_sorted.sort()
            self.update()

    def remove_marker_and_paint(self: QtWidgets.QSlider, value=None):
        """Remove a tick mark."""
        if value is None or isinstance(value, bool):  # Default to current slider value
            value = self.value()
        if value in self.markers:
            self.markers.remove(value)
            if value in self.markers_sorted:
                self.markers_sorted.remove(value)
            self.update()

    def _add_sorted_marker(
        marker_set: set[int], marker_list: list[int], value: int
    ) -> bool:
        if value not in marker_set:
            marker_set.add(value)
            marker_list.append(value)
            marker_list.sort()
            return True
        return False

    def _remove_sorted_marker(
        marker_set: set[int], marker_list: list[int], value: int
    ) -> bool:
        if value in marker_set:
            marker_set.remove(value)
            if value in marker_list:
                marker_list.remove(value)
            return True
        return False

    def add_issue_marker_and_paint(self: QtWidgets.QSlider, value=None):
        if value is None or isinstance(value, bool):
            value = self.value()
        if self.minimum() <= value <= self.maximum() and _add_sorted_marker(
            self.issue_markers, self.issue_markers_sorted, value
        ):
            self.update()

    def remove_issue_marker_and_paint(self: QtWidgets.QSlider, value=None):
        if value is None or isinstance(value, bool):
            value = self.value()
        if _remove_sorted_marker(self.issue_markers, self.issue_markers_sorted, value):
            self.update()

    def add_dropped_marker_and_paint(self: QtWidgets.QSlider, value=None):
        if value is None or isinstance(value, bool):
            value = self.value()
        if self.minimum() <= value <= self.maximum() and _add_sorted_marker(
            self.dropped_markers, self.dropped_markers_sorted, value
        ):
            self.update()

    def remove_dropped_marker_and_paint(self: QtWidgets.QSlider, value=None):
        if value is None or isinstance(value, bool):
            value = self.value()
        if _remove_sorted_marker(
            self.dropped_markers, self.dropped_markers_sorted, value
        ):
            self.update()

    def paintEvent(self: QtWidgets.QSlider, event: QtGui.QPaintEvent):
        """Custom paint: draws the groove, a thin white handle, coloured marker ticks, and job-bracket characters."""
        if self.maximum() == self.minimum():
            return super(QtWidgets.QSlider, self).paintEvent(event)
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

        def marker_x_for_value(value: int) -> float:
            marker_normalized_value = (value - self.minimum()) / (
                self.maximum() - self.minimum()
            )
            return groove_start + marker_normalized_value * groove_width

        # Draw issue markers underneath saved markers.
        if self.issue_markers:
            issue_pen = QtGui.QPen(QtGui.QColor("#ff9800"), 3)
            issue_pen.setCapStyle(QtCore.Qt.PenCapStyle.SquareCap)
            painter.setPen(issue_pen)
            issue_top = groove_y - 2
            issue_bottom = groove_y + 2
            for value in self.issue_markers_sorted:
                if value in self.dropped_markers:
                    continue
                marker_x = marker_x_for_value(value)
                painter.drawLine(marker_x, issue_top, marker_x, issue_bottom)

        # Draw standard markers (if any)
        if self.markers:
            painter.setPen(
                QtGui.QPen(QtGui.QColor("#4090a3"), 3)
            )  # Marker color and thickness
            for value in self.markers_sorted:
                marker_x = marker_x_for_value(value)
                painter.drawLine(
                    marker_x, groove_rect.top(), marker_x, groove_rect.bottom()
                )

        # Draw dropped markers above all frame markers.
        if self.dropped_markers:
            painter.setPen(QtGui.QPen(QtGui.QColor("#e8483c"), 3))
            for value in self.dropped_markers_sorted:
                marker_x = marker_x_for_value(value)
                painter.drawLine(
                    marker_x, groove_rect.top(), marker_x, groove_rect.bottom()
                )

        # Draw Job Start/End Brackets on the groove line
        painter.setFont(QtGui.QFont("Arial", 16, QtGui.QFont.Bold))
        font_metrics = painter.fontMetrics()
        bracket_height = font_metrics.height()
        bracket_y_pos = groove_y + (bracket_height // 4)

        for start_frame, end_frame in main_window.job_marker_pairs:
            if start_frame is not None:
                start_x = marker_x_for_value(int(start_frame))
                painter.setPen(QtGui.QPen(QtGui.QColor("#4CAF50"), 1))
                painter.drawText(int(start_x - 4), int(bracket_y_pos), "[")

            if end_frame is not None:
                end_x = marker_x_for_value(int(end_frame))
                painter.setPen(QtGui.QPen(QtGui.QColor("#e8483c"), 1))
                painter.drawText(int(end_x - 4), int(bracket_y_pos), "]")

    main_window.videoSeekSlider.add_marker_and_paint = partial(
        add_marker_and_paint, main_window.videoSeekSlider
    )
    main_window.videoSeekSlider.remove_marker_and_paint = partial(
        remove_marker_and_paint, main_window.videoSeekSlider
    )
    main_window.videoSeekSlider.add_issue_marker_and_paint = partial(
        add_issue_marker_and_paint, main_window.videoSeekSlider
    )
    main_window.videoSeekSlider.remove_issue_marker_and_paint = partial(
        remove_issue_marker_and_paint, main_window.videoSeekSlider
    )
    main_window.videoSeekSlider.add_dropped_marker_and_paint = partial(
        add_dropped_marker_and_paint, main_window.videoSeekSlider
    )
    main_window.videoSeekSlider.remove_dropped_marker_and_paint = partial(
        remove_dropped_marker_and_paint, main_window.videoSeekSlider
    )
    main_window.videoSeekSlider.paintEvent = partial(
        paintEvent, main_window.videoSeekSlider
    )


def add_video_slider_marker(main_window: "MainWindow"):
    """
    Adds a standard parameter marker at the current slider position.

    Requires a video to be loaded and at least one target face to be present.
    Shows an error message box if either precondition is not met, or if a marker
    already exists at that position.
    """
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
        f"[INFO] Job Start Marker added for pair {len(main_window.job_marker_pairs)} at Frame: {current_pos}"
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
        f"[INFO] Job End Marker added for pair {last_pair_index + 1} at Frame: {current_pos}"
    )


def remove_video_slider_marker(main_window: "MainWindow"):
    """
    Removes the marker (standard or job-bracket) at the current slider position.

    If the position belongs to a job marker pair, the entire pair is removed.
    If no marker exists at the position, an error message box is shown.
    """
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
                f"[INFO] Removing Job Marker Pair {i + 1} ({start_frame}, {end_frame}) because marker found at position: {current_position}"
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
    """Stores a snapshot of the current parameters and control state at the given frame position."""
    main_window.videoSeekSlider.add_marker_and_paint(position)
    main_window.markers[position] = {"parameters": parameters, "control": control}
    print(f"[INFO] Marker Added for Frame: {position}")


def remove_marker(main_window: "MainWindow", position):
    """Removes the marker at the specified frame position, if one exists."""
    if main_window.markers.get(position):
        main_window.videoSeekSlider.remove_marker_and_paint(position)
        main_window.markers.pop(position)
        print(f"[INFO] Marker Removed from position: {position}")


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
    """Moves the slider to the nearest marker that is after the current position."""
    move_slider_to_nearest_marker(main_window, "next")


def move_slider_to_previous_nearest_marker(main_window: "MainWindow"):
    """Moves the slider to the nearest marker that is before the current position."""
    move_slider_to_nearest_marker(main_window, "previous")


def add_scan_review_controls(main_window: "MainWindow"):
    """Creates a collapsible second row of scan/review controls."""
    if hasattr(main_window, "scanToolsToggleButton"):
        return

    def create_divider(parent, height: int = 16, margin: int = 12):
        divider_container = QtWidgets.QWidget(parent)
        divider_layout = QtWidgets.QHBoxLayout(divider_container)
        divider_layout.setContentsMargins(margin, 0, margin, 0)
        divider_layout.setSpacing(0)
        divider = QtWidgets.QFrame(divider_container)
        divider.setFrameShape(QtWidgets.QFrame.Shape.VLine)
        divider.setFrameShadow(QtWidgets.QFrame.Shadow.Plain)
        divider.setLineWidth(1)
        divider.setMidLineWidth(0)
        divider.setFixedHeight(height)
        divider.setStyleSheet("color: rgba(180, 180, 180, 110);")
        divider_layout.addWidget(divider)
        return divider_container

    toggle_button = QtWidgets.QPushButton("Scan Tools")
    toggle_button.setCheckable(True)
    toggle_button.setChecked(False)
    toggle_button.setFlat(True)
    toggle_button.setToolTip("Show or hide the scan tools.")
    toggle_button.clicked.connect(
        lambda checked: set_scan_tools_expanded(main_window, checked)
    )
    main_window.scanToolsToggleButton = toggle_button
    media_layout = getattr(
        main_window,
        "mediaControlsTransportLayout",
        main_window.horizontalLayoutMediaButtons,
    )
    media_layout.addWidget(toggle_button)
    if hasattr(main_window, "_sync_media_controls_balance"):
        main_window._sync_media_controls_balance()

    section = QtWidgets.QWidget(main_window)
    section_layout = QtWidgets.QVBoxLayout(section)
    section_layout.setContentsMargins(0, 0, 0, 0)
    section_layout.setSpacing(4)

    container = QtWidgets.QWidget(section)
    container_layout = QtWidgets.QHBoxLayout(container)
    container_layout.setContentsMargins(0, 0, 0, 0)
    container_layout.setSpacing(6)
    main_window.scanControlsLayout = container_layout
    main_window.scanControlsContainer = container
    left_group = QtWidgets.QWidget(container)
    left_layout = QtWidgets.QHBoxLayout(left_group)
    left_layout.setContentsMargins(0, 0, 0, 0)
    left_layout.setSpacing(6)
    navigation_group = QtWidgets.QWidget(container)
    navigation_layout = QtWidgets.QHBoxLayout(navigation_group)
    navigation_layout.setContentsMargins(0, 0, 0, 0)
    navigation_layout.setSpacing(6)
    frame_group = QtWidgets.QWidget(container)
    frame_layout = QtWidgets.QHBoxLayout(frame_group)
    frame_layout.setContentsMargins(0, 0, 0, 0)
    frame_layout.setSpacing(6)
    cleanup_group = QtWidgets.QWidget(container)
    cleanup_layout = QtWidgets.QHBoxLayout(cleanup_group)
    cleanup_layout.setContentsMargins(0, 0, 0, 0)
    cleanup_layout.setSpacing(6)
    main_window.scanControlsLeftGroup = left_group
    main_window.scanControlsNavigationGroup = navigation_group
    main_window.scanControlsFrameGroup = frame_group
    main_window.scanControlsCleanupGroup = cleanup_group

    run_scan_button = QtWidgets.QPushButton("Scan for Issues")
    run_scan_button.setToolTip(
        "Scans the current render range using your current settings.\n"
        "If record start/end markers exist, only those ranges are scanned.\n"
        "Saved settings markers are applied during the scan.\n"
        "Flags detection or similarity misses for the selected target face.\n"
        "Single-frame preview may differ from playback on borderline frames."
    )
    run_scan_button.setSizePolicy(
        QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed
    )
    run_scan_button.setFlat(True)
    run_scan_button.clicked.connect(lambda: toggle_issue_scan(main_window))
    main_window.runScanButton = run_scan_button

    button_specs = [
        (
            "prevIssueButton",
            "Prev Issue",
            "Move to the previous issue frame for the selected target face.",
            lambda: move_slider_to_previous_issue(main_window),
            "navigation",
        ),
        (
            "nextIssueButton",
            "Next Issue",
            "Move to the next issue frame for the selected target face.",
            lambda: move_slider_to_next_issue(main_window),
            "navigation",
        ),
        (
            "dropFrameButton",
            "Drop Frame",
            "Drops the current frame from render output.",
            lambda: toggle_drop_frame(main_window),
            "frame",
        ),
        (
            "dropAllIssueFramesButton",
            "Drop Issue Frames",
            "Marks the selected target face's issue frames as dropped.\n"
            "Dropped frames are excluded from render output.",
            lambda: drop_all_issue_frames(main_window),
            "frame",
        ),
        (
            "clearScanResultsButton",
            "Clear Issues",
            "Removes the current scan issue markers.\nDoes not affect dropped frames.",
            lambda: clear_scan_results(main_window),
            "cleanup",
        ),
        (
            "clearDroppedFramesButton",
            "Restore Dropped",
            "Restores all dropped frames to render output.",
            lambda: clear_dropped_frames(main_window),
            "cleanup",
        ),
    ]

    left_layout.addWidget(run_scan_button)

    for attr_name, text, tooltip, handler, group_name in button_specs:
        button = QtWidgets.QPushButton(text)
        button.setToolTip(tooltip)
        button.setFlat(True)
        button.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed
        )
        button.clicked.connect(handler)
        setattr(main_window, attr_name, button)
        if group_name == "left":
            target_layout = left_layout
        elif group_name == "navigation":
            target_layout = navigation_layout
        elif group_name == "frame":
            target_layout = frame_layout
        else:
            target_layout = cleanup_layout
        target_layout.addWidget(button)

    container_layout.addStretch(1)
    container_layout.addWidget(left_group, 0, QtCore.Qt.AlignmentFlag.AlignLeft)
    container_layout.addWidget(create_divider(container))
    container_layout.addWidget(
        navigation_group, 0, QtCore.Qt.AlignmentFlag.AlignHCenter
    )
    container_layout.addWidget(create_divider(container))
    container_layout.addWidget(frame_group, 0, QtCore.Qt.AlignmentFlag.AlignHCenter)
    container_layout.addWidget(create_divider(container))
    container_layout.addWidget(cleanup_group, 0, QtCore.Qt.AlignmentFlag.AlignRight)
    container_layout.addStretch(1)
    section_layout.addWidget(container)
    main_window.scanToolsSection = section
    main_window.verticalLayoutMediaControls.addWidget(section)
    set_scan_tools_expanded(
        main_window, getattr(main_window, "scan_tools_expanded", False)
    )

    update_drop_frame_button_label(main_window)
    update_scan_review_button_states(main_window)


# --- Scan / Issue / Drop UI state helpers ---
def set_scan_tools_expanded(main_window: "MainWindow", expanded: bool):
    """Shows or hides the scan-tools row and updates the toggle label."""
    main_window.scan_tools_expanded = expanded
    toggle_button = getattr(main_window, "scanToolsToggleButton", None)
    container = getattr(main_window, "scanControlsContainer", None)
    if toggle_button is not None:
        toggle_button.blockSignals(True)
        toggle_button.setChecked(expanded)
        toggle_button.blockSignals(False)
    if container is not None:
        container.setVisible(expanded)


def update_drop_frame_button_label(main_window: "MainWindow"):
    """Updates the drop-frame button text to reflect the current frame state."""
    button = getattr(main_window, "dropFrameButton", None)
    if button is None:
        return
    current_frame = int(main_window.videoSeekSlider.value())
    if current_frame in main_window.dropped_frames:
        button.setText("Restore Frame")
        button.setToolTip("Restore this frame so it is included in render output.")
    else:
        button.setText("Drop Frame")
        button.setToolTip("Drop this frame from render output.")


def update_scan_review_button_states(main_window: "MainWindow"):
    """Enable scan/review actions based on available target-face context."""
    has_target_faces = bool(getattr(main_window, "target_faces", {}))
    has_selected_face = (
        getattr(main_window, "selected_target_face_id", None) is not None
    )

    scan_button = getattr(main_window, "runScanButton", None)
    if scan_button is not None:
        scan_button.setEnabled(has_target_faces)

    for button_name in (
        "runScanButton",
        "prevIssueButton",
        "nextIssueButton",
        "dropAllIssueFramesButton",
    ):
        button = getattr(main_window, button_name, None)
        if button is not None:
            if button_name == "runScanButton":
                continue
            button.setEnabled(has_selected_face)


def _set_slider_marker_values(
    slider: QtWidgets.QSlider,
    attr_set_name: str,
    attr_sorted_name: str,
    values: list[int],
):
    slider_set = set(values)
    setattr(slider, attr_set_name, slider_set)
    setattr(slider, attr_sorted_name, sorted(slider_set))


def get_selected_face_issue_frames(main_window: "MainWindow") -> set[int]:
    selected_face_id = getattr(main_window, "selected_target_face_id", None)
    if selected_face_id is None:
        return set()
    return set(main_window.issue_frames_by_face.get(str(selected_face_id), set()))


def refresh_issue_frames_for_selected_face(main_window: "MainWindow"):
    """Refresh visible issue markers to match the currently selected target face."""
    visible_issue_frames = get_selected_face_issue_frames(main_window)
    main_window.issue_frames = visible_issue_frames
    _set_slider_marker_values(
        main_window.videoSeekSlider,
        "issue_markers",
        "issue_markers_sorted",
        list(visible_issue_frames),
    )
    main_window.videoSeekSlider.update()
    update_scan_review_button_states(main_window)


def set_issue_frames_for_face(main_window: "MainWindow", face_id, frames):
    """Stores issue frames for a specific target face and refreshes visible markers if selected."""
    if face_id is None:
        return
    normalized = {int(frame) for frame in frames}
    main_window.issue_frames_by_face[str(face_id)] = normalized
    if str(face_id) == str(getattr(main_window, "selected_target_face_id", None)):
        refresh_issue_frames_for_selected_face(main_window)


def add_issue_frame_for_face(main_window: "MainWindow", face_id, frame_number: int):
    """Merge a single issue frame into the stored per-face mapping."""
    if face_id is None:
        return
    normalized_face_id = str(face_id)
    face_frames = main_window.issue_frames_by_face.setdefault(normalized_face_id, set())
    normalized_frame = int(frame_number)
    if normalized_frame in face_frames:
        return
    face_frames.add(normalized_frame)
    if normalized_face_id == str(getattr(main_window, "selected_target_face_id", None)):
        refresh_issue_frames_for_selected_face(main_window)


def set_issue_frames_by_face(main_window: "MainWindow", frames_by_face):
    """Replaces all stored issue results and refreshes visible markers for the selected face."""
    normalized_mapping: dict[str, set[int]] = {}
    for face_id, frames in (frames_by_face or {}).items():
        normalized_mapping[str(face_id)] = {int(frame) for frame in frames}
    main_window.issue_frames_by_face = normalized_mapping
    refresh_issue_frames_for_selected_face(main_window)


def set_dropped_frames(main_window: "MainWindow", frames):
    """Replaces the current dropped-frame set and refreshes slider visuals."""
    normalized = {int(frame) for frame in frames}
    main_window.dropped_frames = normalized
    _set_slider_marker_values(
        main_window.videoSeekSlider,
        "dropped_markers",
        "dropped_markers_sorted",
        list(normalized),
    )
    main_window.videoSeekSlider.update()
    update_drop_frame_button_label(main_window)


def clear_scan_results(main_window: "MainWindow"):
    selected_face_id = getattr(main_window, "selected_target_face_id", None)
    if selected_face_id is None:
        main_window.issue_frames_by_face.clear()
    else:
        main_window.issue_frames_by_face[str(selected_face_id)] = set()
    refresh_issue_frames_for_selected_face(main_window)


def clear_dropped_frames(main_window: "MainWindow"):
    set_dropped_frames(main_window, [])


def toggle_drop_frame(main_window: "MainWindow"):
    current_position = int(main_window.videoSeekSlider.value())
    if current_position in main_window.dropped_frames:
        main_window.dropped_frames.remove(current_position)
        main_window.videoSeekSlider.remove_dropped_marker_and_paint(current_position)
    else:
        main_window.dropped_frames.add(current_position)
        main_window.videoSeekSlider.add_dropped_marker_and_paint(current_position)
    update_drop_frame_button_label(main_window)


def drop_all_issue_frames(main_window: "MainWindow"):
    selected_issue_frames = get_selected_face_issue_frames(main_window)
    if not selected_issue_frames:
        return
    reply = QtWidgets.QMessageBox.question(
        main_window,
        "Drop Issue Frames",
        "Mark all current issue frames as dropped for render output?",
        QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
        QtWidgets.QMessageBox.No,
    )
    if reply != QtWidgets.QMessageBox.Yes:
        return
    merged = set(main_window.dropped_frames)
    merged.update(selected_issue_frames)
    set_dropped_frames(main_window, merged)


# --- Issue scan execution / progress helpers ---
def _move_slider_to_nearest_issue(main_window: "MainWindow", direction: str):
    current_position = int(main_window.videoSeekSlider.value())
    review_frames = sorted(
        get_selected_face_issue_frames(main_window) | set(main_window.dropped_frames)
    )
    if not review_frames:
        return

    new_position = None
    if direction == "next":
        filtered = [frame for frame in review_frames if frame > current_position]
        new_position = filtered[0] if filtered else None
    else:
        filtered = [frame for frame in review_frames if frame < current_position]
        new_position = filtered[-1] if filtered else None

    if new_position is not None:
        main_window.videoSeekSlider.setValue(new_position)
        main_window.video_processor.process_current_frame()


def move_slider_to_next_issue(main_window: "MainWindow"):
    _move_slider_to_nearest_issue(main_window, "next")


def move_slider_to_previous_issue(main_window: "MainWindow"):
    _move_slider_to_nearest_issue(main_window, "previous")


def _set_slider_frame_without_side_effects(
    main_window: "MainWindow", frame_number: int
) -> None:
    slider = main_window.videoSeekSlider
    slider.blockSignals(True)
    slider.setValue(int(frame_number))
    slider.blockSignals(False)
    slider.update()


def _get_issue_scan_ui_state(main_window: "MainWindow") -> dict:
    state = getattr(main_window, "scan_issue_ui_state", None)
    if state is None:
        state = {}
        main_window.scan_issue_ui_state = state
    return state


def _restore_issue_scan_display(main_window: "MainWindow") -> None:
    video_processor = getattr(main_window, "video_processor", None)
    process_current_frame = getattr(video_processor, "process_current_frame", None)
    if callable(process_current_frame):
        process_current_frame()


def _set_issue_scan_tool_button_state(
    main_window: "MainWindow", scan_active: bool
) -> None:
    run_button = getattr(main_window, "runScanButton", None)
    if run_button is not None:
        run_button.setEnabled(
            True if scan_active else bool(getattr(main_window, "target_faces", {}))
        )

    for button_name in (
        "prevIssueButton",
        "nextIssueButton",
        "dropFrameButton",
        "dropAllIssueFramesButton",
        "clearScanResultsButton",
        "clearDroppedFramesButton",
    ):
        button = getattr(main_window, button_name, None)
        if button is not None:
            button.setEnabled(not scan_active)


def _start_issue_scan_ui(main_window: "MainWindow", worker, scope_text: str) -> None:
    state = _get_issue_scan_ui_state(main_window)
    state.clear()
    state.update(
        {
            "active": True,
            "start_frame": int(main_window.videoSeekSlider.value()),
            "scope_text": scope_text,
            "keep_controls": bool(main_window.control.get("KeepControlsToggle", False)),
        }
    )

    if not state["keep_controls"]:
        layout_actions.disable_all_parameters_and_control_widget(main_window)

    _set_issue_scan_tool_button_state(main_window, True)

    run_button = getattr(main_window, "runScanButton", None)
    if run_button is not None:
        run_button.setText("Abort Scan")
        run_button.setToolTip(
            f"{scope_text}\nAbort the active issue scan and keep the issue frames found so far."
        )

    play_button = getattr(main_window, "buttonMediaPlay", None)
    if play_button is not None:
        play_button.setEnabled(False)

    record_button = getattr(main_window, "buttonMediaRecord", None)
    if record_button is not None:
        record_button.setEnabled(False)

    toggle_button = getattr(main_window, "scanToolsToggleButton", None)
    if toggle_button is not None:
        toggle_button.setEnabled(False)

    QtCore.QCoreApplication.processEvents()


def _restore_issue_scan_ui(main_window: "MainWindow") -> None:
    state = _get_issue_scan_ui_state(main_window)
    if not state.get("active"):
        return

    if not state.get("keep_controls", False):
        layout_actions.enable_all_parameters_and_control_widget(main_window)

    start_frame = int(state.get("start_frame", main_window.videoSeekSlider.value()))
    _set_slider_frame_without_side_effects(main_window, start_frame)
    main_window.video_processor.current_frame_number = start_frame

    run_button = getattr(main_window, "runScanButton", None)
    if run_button is not None:
        run_button.setText("Scan for Issues")
        run_button.setToolTip(
            "Scans the current render range using your current settings.\n"
            "If record start/end markers exist, only those ranges are scanned.\n"
            "Saved settings markers are applied during the scan.\n"
            "Flags detection or similarity misses for the selected target face.\n"
            "Single-frame preview may differ from playback on borderline frames."
        )

    play_button = getattr(main_window, "buttonMediaPlay", None)
    if play_button is not None:
        play_button.setEnabled(True)

    record_button = getattr(main_window, "buttonMediaRecord", None)
    if record_button is not None:
        record_button.setEnabled(True)

    toggle_button = getattr(main_window, "scanToolsToggleButton", None)
    if toggle_button is not None:
        toggle_button.setEnabled(True)

    state.clear()
    _set_issue_scan_tool_button_state(main_window, False)
    update_scan_review_button_states(main_window)
    update_drop_frame_button_label(main_window)
    _restore_issue_scan_display(main_window)
    QtCore.QCoreApplication.processEvents()


def toggle_issue_scan(main_window: "MainWindow") -> None:
    worker = getattr(main_window, "scan_issue_worker", None)
    if worker is not None:
        worker.cancel()
        return
    run_issue_scan(main_window)


def _handle_issue_scan_progress(
    main_window: "MainWindow",
    scope_text: str,
    processed: int,
    total: int,
    frame_number: int,
    scan_fps: float,
):
    _set_slider_frame_without_side_effects(main_window, frame_number)
    run_button = getattr(main_window, "runScanButton", None)
    if run_button is not None:
        run_button.setText(f"Abort Scan ({processed}/{max(total, 1)})")
    QtCore.QCoreApplication.processEvents()


def _handle_issue_scan_issue_found(
    main_window: "MainWindow", face_id: str, frame_number: int
) -> None:
    add_issue_frame_for_face(main_window, face_id, frame_number)


def _cleanup_issue_scan_worker(main_window: "MainWindow"):
    worker = getattr(main_window, "scan_issue_worker", None)
    if worker is not None:
        worker.deleteLater()
        main_window.scan_issue_worker = None


def _handle_issue_scan_completed(
    main_window: "MainWindow",
    issue_frames_by_face: dict,
    frames_scanned: int,
    faces_with_issues: int,
    scope_text: str,
    elapsed_seconds: float,
    cancelled: bool = False,
):
    set_issue_frames_by_face(main_window, issue_frames_by_face)
    _restore_issue_scan_ui(main_window)
    _cleanup_issue_scan_worker(main_window)
    total_issue_frames = sum(
        len(set(frames)) for frames in issue_frames_by_face.values()
    )
    scan_fps = (frames_scanned / elapsed_seconds) if elapsed_seconds > 0 else 0.0
    print(
        f"[INFO] Scan: Scope: {scope_text.removeprefix('Scanning ')} | Frames: {frames_scanned} | Time: {elapsed_seconds:.1f}s | FPS: {scan_fps:.1f} | Issues: {total_issue_frames} | Faces with issues: {faces_with_issues} | Cancelled: {cancelled}"
    )
    if cancelled:
        common_widget_actions.create_and_show_toast_message(
            main_window,
            "Scan Aborted",
            f"Stopped after {frames_scanned} scanned frames. Kept {total_issue_frames} issue frames found so far.",
            style_type="warning",
        )
    elif total_issue_frames:
        common_widget_actions.create_and_show_toast_message(
            main_window,
            "Scan Complete",
            f"Scanned {frames_scanned} frames in {elapsed_seconds:.1f}s ({scan_fps:.1f} FPS). Found {total_issue_frames} issues across {faces_with_issues} faces.",
        )
    else:
        common_widget_actions.create_and_show_toast_message(
            main_window,
            "Scan Complete",
            f"Scanned {frames_scanned} frames in {elapsed_seconds:.1f}s ({scan_fps:.1f} FPS). No issue frames found.",
        )


def _handle_issue_scan_cancelled(main_window: "MainWindow"):
    _restore_issue_scan_ui(main_window)
    _cleanup_issue_scan_worker(main_window)
    common_widget_actions.create_and_show_toast_message(
        main_window,
        "Scan Cancelled",
        "Issue scan aborted before finalizing. Kept any issue frames found so far.",
        style_type="warning",
    )


def _handle_issue_scan_failed(main_window: "MainWindow", error_message: str):
    _restore_issue_scan_ui(main_window)
    _cleanup_issue_scan_worker(main_window)
    common_widget_actions.create_and_show_messagebox(
        main_window,
        "Scan Failed",
        error_message,
        main_window.runScanButton,
    )


def run_issue_scan(main_window: "MainWindow"):
    video_processor = main_window.video_processor
    if not getattr(main_window, "target_faces", {}):
        common_widget_actions.create_and_show_messagebox(
            main_window,
            "Scan Not Available",
            "No target faces found. Use Find Faces before running a scan.",
            main_window.videoSeekSlider,
        )
        return
    if video_processor.file_type != "video" or not video_processor.media_path:
        common_widget_actions.create_and_show_messagebox(
            main_window,
            "Scan Not Available",
            "Issue scans are only available when a video is loaded.",
            main_window.videoSeekSlider,
        )
        return
    if not Path(video_processor.media_path).is_file():
        common_widget_actions.create_and_show_messagebox(
            main_window,
            "Scan Not Available",
            "The selected video could not be found on disk.",
            main_window.videoSeekSlider,
        )
        return
    if getattr(main_window, "scan_issue_worker", None) is not None:
        return

    was_processing = video_processor.stop_processing()
    if was_processing:
        print("[INFO] Stopped active processing before running issue scan.")

    worker = ui_workers.IssueScanWorker(main_window)
    main_window.scan_issue_worker = worker
    scope_text = worker._scan_scope_text
    set_issue_frames_by_face(main_window, {})
    _start_issue_scan_ui(main_window, worker, scope_text)
    worker.progress.connect(
        lambda processed, total, frame_number, scan_fps: _handle_issue_scan_progress(
            main_window, scope_text, processed, total, frame_number, scan_fps
        )
    )
    worker.issue_found.connect(
        lambda face_id, frame_number: _handle_issue_scan_issue_found(
            main_window, face_id, frame_number
        )
    )
    worker.completed.connect(
        lambda issue_frames_by_face, frames_scanned, faces_with_issues, completed_scope_text, elapsed_seconds, cancelled: (
            _handle_issue_scan_completed(
                main_window,
                issue_frames_by_face,
                frames_scanned,
                faces_with_issues,
                completed_scope_text,
                elapsed_seconds,
                cancelled,
            )
        )
    )
    worker.cancelled.connect(lambda: _handle_issue_scan_cancelled(main_window))
    worker.failed.connect(
        lambda error_message: _handle_issue_scan_failed(main_window, error_message)
    )
    worker.start()


def remove_face_parameters_and_control_from_markers(main_window: "MainWindow", face_id):
    """
    Removes all stored parameter entries for *face_id* from every marker.

    If any marker's parameter dict becomes empty after the removal, all markers are
    deleted because there is no longer any face data to track.
    """
    for _, marker_data in main_window.markers.items():
        marker_data["parameters"].pop(
            face_id, None
        )  # Use .pop with default to avoid KeyError
        # If the parameters is empty, then there is no longer any marker to be set for any target face
        if not marker_data["parameters"]:
            delete_all_markers(main_window)
            break


def remove_all_markers(main_window: "MainWindow"):
    """Removes all standard, issue, and dropped-frame markers plus job pairs."""
    standard_markers_positions = list(main_window.markers.keys())
    for marker_position in standard_markers_positions:
        remove_marker(main_window, marker_position)
    main_window.markers.clear()
    set_issue_frames_by_face(main_window, {})
    set_dropped_frames(main_window, [])
    if main_window.job_marker_pairs:
        print("[INFO] Clearing job marker pairs.")
        main_window.job_marker_pairs.clear()


def advance_video_slider_by_n_frames(main_window: "MainWindow", n=None):
    """
    Advances the seek slider forward by *n* frames (clamped to the last frame).

    Relies on the slider's valueChanged signal to handle raw frame reading and
    post-seek actions natively, avoiding duplicated heavy processing.
    """
    video_processor = main_window.video_processor
    if video_processor.media_capture:
        if n is None:
            n = int(main_window.control.get("FrameSkipStepSlider", 30))

        current_position = int(main_window.videoSeekSlider.value())
        new_position = current_position + n
        if new_position > video_processor.max_frame_number:
            new_position = video_processor.max_frame_number

        # 1. Setting the value triggers 'on_change_video_seek_slider' automatically.
        # Since the slider is not being dragged (isSliderDown() == False),
        # that slot will naturally execute 'run_post_seek_actions' ONCE.
        main_window.videoSeekSlider.setValue(new_position)

        # 2. Check if this is a single frame step (like 'V' key)
        is_single_frame_step = n == 1

        # 3. Run AI models. Runs synchronously only for single steps to prevent "flash".
        main_window.video_processor.process_current_frame(
            synchronous=is_single_frame_step
        )


def rewind_video_slider_by_n_frames(main_window: "MainWindow", n=None):
    """
    Rewinds the seek slider backward by *n* frames (clamped to frame 0).

    Relies on the slider's valueChanged signal to handle raw frame reading and
    post-seek actions natively, avoiding duplicated heavy processing.
    """
    video_processor = main_window.video_processor
    if video_processor.media_capture:
        if n is None:
            n = int(main_window.control.get("FrameSkipStepSlider", 30))

        current_position = int(main_window.videoSeekSlider.value())
        new_position = current_position - n
        if new_position < 0:
            new_position = 0

        # 1. Setting the value triggers 'on_change_video_seek_slider' automatically.
        # Prevents double execution of heavy Face Detection.
        main_window.videoSeekSlider.setValue(new_position)

        # 2. Check if this is a single frame step (like 'C' key)
        is_single_frame_step = n == 1

        # 3. Run AI models. Runs synchronously only for single steps to prevent "flash".
        main_window.video_processor.process_current_frame(
            synchronous=is_single_frame_step
        )


def delete_all_markers(main_window: "MainWindow"):
    """Clears all slider marker overlays and marker dictionaries without removing job pairs."""
    main_window.videoSeekSlider.markers = set()
    main_window.videoSeekSlider.markers_sorted = []
    main_window.videoSeekSlider.issue_markers = set()
    main_window.videoSeekSlider.issue_markers_sorted = []
    main_window.videoSeekSlider.dropped_markers = set()
    main_window.videoSeekSlider.dropped_markers_sorted = []
    main_window.videoSeekSlider.update()
    main_window.markers.clear()
    main_window.issue_frames_by_face.clear()
    main_window.issue_frames.clear()
    main_window.dropped_frames.clear()
    update_drop_frame_button_label(main_window)


def _restore_window_base_mode(
    main_window: "MainWindow",
    *,
    restore_to_fullscreen: bool,
    restore_to_maximized: bool,
    restore_geometry=None,
):
    if restore_to_fullscreen:
        try:
            main_window.setWindowState(QtCore.Qt.WindowState.WindowFullScreen)
        except Exception:
            main_window.showFullScreen()
        main_window.is_full_screen = True
        return

    if restore_to_maximized:
        main_window.showMaximized()
    else:
        main_window.showNormal()
        if restore_geometry is not None:
            main_window.setGeometry(restore_geometry)
    main_window.is_full_screen = False


def view_fullscreen(main_window: "MainWindow"):
    """Toggle fullscreen without changing theatre mode."""

    if main_window.isFullScreen():
        restore_to_maximized = bool(
            getattr(main_window, "_fullscreen_restore_was_maximized", False)
        )
        restore_geometry = getattr(main_window, "_fullscreen_restore_geometry", None)
        _restore_window_base_mode(
            main_window,
            restore_to_fullscreen=False,
            restore_to_maximized=restore_to_maximized,
            restore_geometry=restore_geometry,
        )
        main_window._fullscreen_restore_was_maximized = False
        main_window._fullscreen_restore_geometry = None
    else:
        was_maximized = main_window.isMaximized()
        main_window._fullscreen_restore_was_maximized = was_maximized
        main_window._fullscreen_restore_geometry = (
            main_window.normalGeometry() if was_maximized else main_window.geometry()
        )
        main_window.showFullScreen()  # Enter full-screen mode
        main_window.is_full_screen = True

    sync_actions = getattr(main_window, "_sync_viewer_menu_actions", None)

    if callable(sync_actions):
        sync_actions()


def fit_view_to_current_image(main_window: "MainWindow"):
    layout_actions.fit_image_to_view_onchange(main_window)


def zoom_current_image_100(main_window: "MainWindow"):
    view = main_window.graphicsViewFrame
    if not view.scene():
        return
    items = view.scene().items()
    pixmap_item = next(
        (item for item in items if isinstance(item, QtWidgets.QGraphicsPixmapItem)),
        None,
    )
    if pixmap_item is None:
        return

    view.resetTransform()
    view.setSceneRect(pixmap_item.boundingRect())
    view.centerOn(pixmap_item)
    view.zoom_value = 0
    view.last_scale_factor = 1.0


def show_graphics_view_context_menu(
    main_window: "MainWindow", global_pos: QtCore.QPoint
):
    menu = QtWidgets.QMenu(main_window.graphicsViewFrame)
    viewer_mode_actions_enabled = bool(
        getattr(main_window, "viewer_mode_actions_enabled", True)
    )
    fit_action = menu.addAction("Fit to View")
    zoom_100_action = menu.addAction("100% Zoom")
    menu.addSeparator()
    fullscreen_action = menu.addAction("Fullscreen")
    fullscreen_action.setCheckable(True)
    fullscreen_action.setChecked(bool(main_window._is_fullscreen_menu_active()))
    theatre_action = menu.addAction("Theatre Mode")
    theatre_action.setCheckable(True)
    theatre_action.setChecked(bool(getattr(main_window, "is_theatre_mode", False)))
    menu.addSeparator()
    face_compare_action = menu.addAction("Face Compare")
    face_compare_action.setCheckable(True)
    face_compare_action.setChecked(bool(main_window.view_face_compare_enabled))
    face_compare_action.setEnabled(viewer_mode_actions_enabled)
    menu.addSeparator()
    current_mask_value = main_window._get_current_mask_show_value()
    mask_actions = {}
    for option in main_window._get_mask_show_options():
        action = menu.addAction(main_window._mask_show_context_menu_label(option))
        action.setCheckable(True)
        action.setChecked(
            bool(main_window.view_face_mask_enabled) and option == current_mask_value
        )
        action.setEnabled(viewer_mode_actions_enabled)
        mask_actions[action] = option
    menu.addSeparator()
    save_action = menu.addAction("Save Image")
    selected_action = menu.exec(global_pos)
    if selected_action is fit_action:
        fit_view_to_current_image(main_window)
    elif selected_action is zoom_100_action:
        zoom_current_image_100(main_window)
    elif selected_action is fullscreen_action:
        view_fullscreen(main_window)
        main_window._sync_viewer_menu_actions()
    elif selected_action is theatre_action:
        QtCore.QTimer.singleShot(
            0,
            lambda: _run_context_menu_theatre_toggle(main_window),
        )
    elif selected_action is face_compare_action:
        main_window._set_compare_mode("compare", face_compare_action.isChecked())
        main_window._sync_viewer_menu_actions()
    elif selected_action is save_action:
        save_current_frame_to_file(main_window)
    elif selected_action in mask_actions:
        selected_value = mask_actions[selected_action]
        main_window._handle_viewer_mask_action(
            selected_value, selected_action.isChecked()
        )


def enable_zoom_and_pan(main_window: "MainWindow", view: QtWidgets.QGraphicsView):
    """
    Attaches mouse-wheel zoom, middle-click pan, and context-menu behaviour to a
    QGraphicsView instance.

    Monkey-patches zoom, reset_zoom, wheelEvent, mousePressEvent, mouseMoveEvent,
    mouseReleaseEvent, and contextMenuEvent directly onto the view object so no
    subclass is required.
    """
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
        """Resets the view transform so the scene content fits the viewport exactly."""
        fit_view_to_current_image(main_window)

    def mousePressEvent(self: QtWidgets.QGraphicsView, event: QtGui.QMouseEvent):
        """Handle mouse press event for panning."""
        if event.button() == QtCore.Qt.MouseButton.MiddleButton:
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
        if event.button() == QtCore.Qt.MouseButton.MiddleButton:
            self.is_panning = False
            self.setCursor(QtCore.Qt.ArrowCursor)  # Reset the cursor
        else:
            # Explicitly call the base class implementation
            QtWidgets.QGraphicsView.mouseReleaseEvent(self, event)

    def contextMenuEvent(self: QtWidgets.QGraphicsView, event: QtGui.QContextMenuEvent):
        show_graphics_view_context_menu(main_window, event.globalPos())
        event.accept()

    # Attach methods to the view
    view.zoom = partial(zoom, view)
    view.reset_zoom = partial(reset_zoom, view)
    view.wheelEvent = partial(wheelEvent, view)
    view.mousePressEvent = partial(mousePressEvent, view)
    view.mouseMoveEvent = partial(mouseMoveEvent, view)
    view.mouseReleaseEvent = partial(mouseReleaseEvent, view)
    view.contextMenuEvent = partial(contextMenuEvent, view)

    # view.zoom = zoom.__get__(view)
    # view.reset_zoom = reset_zoom.__get__(view)
    # view.wheelEvent = wheelEvent.__get__(view)

    # Set anchors for better interaction
    view.setTransformationAnchor(
        QtWidgets.QGraphicsView.ViewportAnchor.AnchorUnderMouse
    )
    view.setResizeAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorUnderMouse)


def play_video(main_window: "MainWindow", checked: bool):
    """
    Starts or stops video/webcam playback in response to the Play button toggle.

    When *checked* is True: starts playback (or webcam stream) if not already running
    and the slider is not at the end of the video.
    When *checked* is False: stops any active processing and resets button states.
    """
    video_processor = main_window.video_processor
    if checked and video_processor.file_type == "webcam":
        if video_processor.processing:
            print(
                "[WARN] Webcam already streaming. Stopping the stream before restarting."
            )
            video_processor.stop_processing()
        print("[INFO] Starting webcam stream processing.")
        set_play_button_icon_to_stop(main_window)
        video_processor.process_webcam()
        return
    if checked:
        if (
            video_processor.processing
            or video_processor.current_frame_number == video_processor.max_frame_number
        ):
            print(
                "[WARN] Video already playing. Stopping the current video before starting a new one."
            )
            video_processor.stop_processing()
            return
        print("[INFO] Starting video processing.")
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
    """
    Starts or stops recording in response to the Record button toggle.

    Supports three recording modes:
      - Default style (no job markers): records from the current slider position.
      - Multi-segment style (job markers set): records each marker-pair segment.
      - Batch processing: forces recording from frame 0.

    When *checked* is False: prompts the user to confirm stopping (unless triggered
    programmatically by the Job Manager) and finalises the current recording.
    """
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
            print("[WARN] Processing already active. Request ignored.")
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
                "[INFO] Record button pressed: Starting default recording (full video or from slider)."
            )
            _disable_compare_preview_modes_for_recording(main_window)
            set_record_button_icon_to_stop(main_window)
            # Disable play button during recording
            main_window.buttonMediaPlay.setEnabled(False)
            video_processor.recording = True  # SET THE FLAG FOR DEFAULT RECORDING

            # If batch process force slider to 0 so that process_video() begins at start.
            if is_batch_processing:
                main_window.videoSeekSlider.blockSignals(True)
                main_window.videoSeekSlider.setValue(0)
                main_window.videoSeekSlider.blockSignals(False)
                print(
                    "[INFO] Batch processing: Forcing video record to start from frame 0."
                )

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
                    # Force cast to tuple of ints.
                    valid_pairs.append((int(pair[0]), int(pair[1])))

            # Proceed if we have valid marker pairs
            if valid_pairs:
                print(
                    f"[INFO] Record button pressed: Starting multi-segment recording for {len(valid_pairs)} segment(s)."
                )
                _disable_compare_preview_modes_for_recording(main_window)
                set_record_button_icon_to_stop(main_window)
                # Disable play button during segment recording
                main_window.buttonMediaPlay.setEnabled(False)
                is_job_context = job_mgr_flag
                print(f"[INFO] Is job manager flag = {is_job_context}")
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
        # --- Stop confirmation (manual UI only) ---
        # The record button is toggle-based, and many call sites do not check return
        # values. Therefore, cancellation must be handled here without relying on
        # callers.
        #
        # Do NOT prompt when this stop was initiated programmatically by Job Manager.
        if (
            video_processor.is_processing_segments or video_processor.recording
        ) and not job_mgr_flag:
            try:
                box = QtWidgets.QMessageBox(main_window)
                box.setIcon(QtWidgets.QMessageBox.Warning)
                box.setWindowTitle("Confirm stop")
                box.setText("Stop multi-segment recording?")
                box.setInformativeText(
                    "Segment recording will stop immediately. Output may be incomplete."
                )
                box.setStandardButtons(
                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
                )
                box.setDefaultButton(QtWidgets.QMessageBox.No)

                if box.exec() != QtWidgets.QMessageBox.Yes:
                    # User declined. Re-arm the toggle to the ON state.
                    main_window.buttonMediaRecord.blockSignals(True)
                    main_window.buttonMediaRecord.setChecked(True)
                    main_window.buttonMediaRecord.blockSignals(False)
                    set_record_button_icon_to_stop(main_window)
                    return
            except Exception:
                # If anything goes wrong with the dialog, fail safe by NOT stopping.
                main_window.buttonMediaRecord.blockSignals(True)
                main_window.buttonMediaRecord.setChecked(True)
                main_window.buttonMediaRecord.blockSignals(False)
                set_record_button_icon_to_stop(main_window)
                return

        if video_processor.is_processing_segments:
            print(
                "[INFO] Record button released: User requested stop during segment processing. Finalizing..."
            )
            # Finalize segment concatenation with segments processed so far
            video_processor.finalize_segment_concatenation()
        elif video_processor.recording:  # Check if default style recording was active
            print(
                "[INFO] Record button released: User requested stop during default recording. Finalizing..."
            )
            # Finalize the default style recording
            video_processor._finalize_default_style_recording()
        else:
            # No recording was active (maybe an immediate click-off or already stopped)
            print("[WARN] Record button released: No active recording found.")
            set_record_button_icon_to_play(main_window)
            main_window.buttonMediaPlay.setEnabled(True)
            reset_media_buttons(main_window)


def set_record_button_icon_to_play(main_window: "MainWindow"):
    """Sets the Record button icon and tooltip to the 'ready-to-record' (stopped) state."""
    main_window.buttonMediaRecord.setIcon(QtGui.QIcon(":/media/media/rec_hover.png"))
    main_window.buttonMediaRecord.setToolTip("Start Recording")


def set_record_button_icon_to_stop(main_window: "MainWindow"):
    """Sets the Record button icon and tooltip to the 'recording active' (stop) state."""
    main_window.buttonMediaRecord.setIcon(QtGui.QIcon(":/media/media/rec_on.png"))
    main_window.buttonMediaRecord.setToolTip("Stop Recording")


def _disable_compare_preview_modes_for_recording(main_window: "MainWindow") -> None:
    disabled_modes = []
    if getattr(main_window, "view_face_compare_enabled", False):
        main_window._set_compare_mode("compare", False)
        disabled_modes.append("Face Compare")
    if getattr(main_window, "view_face_mask_enabled", False):
        main_window._set_compare_mode("mask", False)
        disabled_modes.append("Face Mask")

    if not disabled_modes:
        return

    disabled_modes_text = " and ".join(disabled_modes)
    print(
        f"[INFO] Disabled {disabled_modes_text} preview for recording to keep frame size stable."
    )
    common_widget_actions.create_and_show_toast_message(
        main_window,
        "Preview Disabled for Recording",
        f"Disabled {disabled_modes_text} preview before recording.",
        style_type="warning",
    )


def initialize_media_button_icons(main_window: "MainWindow"):
    """Assign default icons for the visible centered media-control strip."""
    icon_map = {
        "frameRewindButton": ":/media/media/tl_left_hover.png",
        "frameAdvanceButton": ":/media/media/tl_right_hover.png",
        "addMarkerButton": ":/media/media/add_marker_hover.png",
        "removeMarkerButton": ":/media/media/remove_marker_hover.png",
        "previousMarkerButton": ":/media/media/previous_marker_hover.png",
        "nextMarkerButton": ":/media/media/next_marker_hover.png",
        "liveSoundButton": ":/media/media/audio_toggle.png",
        "viewFullScreenButton": ":/media/media/fullscreen_v2.png",
        "theatreModeButton": ":/media/media/theatre_mode.png",
    }
    for button_name, icon_path in icon_map.items():
        button = getattr(main_window, button_name, None)
        if button is not None:
            button.setIcon(QtGui.QIcon(icon_path))

    set_play_button_icon(main_window)
    set_record_button_icon(main_window)


def set_play_button_icon_to_play(main_window: "MainWindow"):
    """Sets the Play button icon and tooltip to the 'ready-to-play' (stopped) state."""
    main_window.buttonMediaPlay.setIcon(QtGui.QIcon(":/media/media/play_hover.png"))
    main_window.buttonMediaPlay.setToolTip("Play")


def set_play_button_icon_to_stop(main_window: "MainWindow"):
    """Sets the Play button icon and tooltip to the 'playing active' (stop) state."""
    main_window.buttonMediaPlay.setIcon(QtGui.QIcon(":/media/media/play_on.png"))
    main_window.buttonMediaPlay.setToolTip("Stop")


def reset_media_buttons(main_window: "MainWindow"):
    """
    Resets the Play and Record buttons to their unchecked, enabled state without
    triggering their toggled/clicked signals.  Updates icons to match the new state.
    """
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
    """Updates the Play button icon and tooltip to reflect its current checked state."""
    if main_window.buttonMediaPlay.isChecked():
        main_window.buttonMediaPlay.setIcon(QtGui.QIcon(":/media/media/play_on.png"))
        main_window.buttonMediaPlay.setToolTip("Stop")
    else:
        main_window.buttonMediaPlay.setIcon(QtGui.QIcon(":/media/media/play_hover.png"))
        main_window.buttonMediaPlay.setToolTip("Play")


def set_record_button_icon(main_window: "MainWindow"):
    """Updates the Record button icon and tooltip to reflect its current checked state."""
    if main_window.buttonMediaRecord.isChecked():
        main_window.buttonMediaRecord.setIcon(QtGui.QIcon(":/media/media/rec_on.png"))
        main_window.buttonMediaRecord.setToolTip("Stop Recording")
    else:
        main_window.buttonMediaRecord.setIcon(
            QtGui.QIcon(":/media/media/rec_hover.png")
        )
        main_window.buttonMediaRecord.setToolTip("Start Recording")


# @misc_helpers.benchmark
@QtCore.Slot(int)
def on_change_video_seek_slider(main_window: "MainWindow", new_position=0):
    """
    Slot connected to the slider's valueChanged signal.

    Stops any active processing, seeks the capture to the new position, reads the
    raw frame for immediate preview, and defers heavy post-seek work (marker
    application, AutoSwap) until the slider is released.
    """
    # print("Called on_change_video_seek_slider()")
    video_processor = main_window.video_processor

    was_processing = video_processor.stop_processing()
    if was_processing:
        print("[WARN] Processing in progress. Stopping current processing.")

    video_processor.current_frame_number = new_position
    video_processor.next_frame_to_display = new_position
    update_drop_frame_button_label(main_window)
    if video_processor.media_capture:
        misc_helpers.seek_frame(video_processor.media_capture, new_position)

        # Read the raw frame without triggering the full pipeline.
        ret, frame = misc_helpers.read_frame(
            video_processor.media_capture, video_processor.media_rotation
        )
        if ret:
            # Cache the raw frame so process_current_frame() can use it as a
            # fallback when the near-EOF re-read fails (OpenCV reliability issue).
            video_processor._seek_cached_frame = (new_position, frame)
            # For preview, show the raw frame immediately.
            # The processed frame will be shown when the slider is released.
            pixmap = common_widget_actions.get_pixmap_from_frame(main_window, frame)
            graphics_view_actions.update_graphics_view(
                main_window,
                pixmap,
                new_position,
                size_mode="native_pixmap_size",
            )

        else:
            # VP-34: Read failed. Trigger a stop/reopen cycle to recover from silent handle failures.
            print(
                f"[WARN] on_change_video_seek_slider: Read failed at frame {new_position}. Attempting recovery..."
            )
            video_processor._seek_cached_frame = None
            main_window.last_seek_read_failed = True
            video_processor.stop_processing()
    # Only update parameters and widgets if the slider is NOT being actively dragged.
    # This ensures playback, clicks, and button presses update the UI,
    # but fast scrubbing does not cause lag or skip marker updates.
    if not main_window.videoSeekSlider.isSliderDown():
        run_post_seek_actions(main_window, new_position)
    # Do not automatically restart the video, let the user press Play to resume
    # print("on_change_video_seek_slider: Video stopped after slider movement.")


def _get_marker_data_for_position(
    main_window: "MainWindow", new_position: int
) -> MarkerData | None:
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
    """
    Loads the parameters and control state stored in the nearest marker at or before
    *new_position* into the live main_window state.

    If no marker is found, the current state is left unchanged.  The global
    TrackMarkersToggle value is always preserved after the load.
    """
    # Find marker only at the *exact* new position
    marker_data = _get_marker_data_for_position(main_window, new_position)
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
    """
    Refreshes all UI widgets to reflect the parameter and control state that was
    loaded by update_parameters_and_control_from_marker for *new_position*.
    """
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
    """Slot connected to sliderMoved; currently a no-op placeholder for future drag-time logic."""
    # print("Called on_slider_moved()")
    main_window.videoSeekSlider.value()
    # print(f"\nSlider Moved. position: {position}\n")


def on_slider_pressed(main_window: "MainWindow"):
    """Slot connected to sliderPressed; currently a no-op placeholder for future press-time logic."""
    main_window.videoSeekSlider.value()
    # print(f"\nSlider Pressed. position: {position}\n")


def run_post_seek_actions(main_window: "MainWindow", new_position: int):
    """
    Executes heavy operations (markers, AutoSwap) after a seek.
    This function is called after a slider release or a jump via button/shortcut.
    """

    # Reset ByteTracker state on every user-initiated seek so stale Kalman predictions
    # from the previous position do not corrupt detection on the new position.
    main_window.models_processor.face_detectors.reset_tracker()

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
    """Triggers a single-frame re-process after the Swap Faces button state changes.

    Runs synchronously so the processed result (including any required model loading
    or first-time CUDA graph builds) is applied to the currently displayed frame
    before control returns to the UI, matching the behaviour of the single-frame-step
    advance button.  For the Custom provider, build-progress dialogs are shown via
    show_build_dialog signals emitted before each CUDA graph capture; those signals
    call processEvents() so the dialog paints even while the main thread is occupied.
    """
    video_processor = main_window.video_processor
    video_processor.process_current_frame(synchronous=True)


def process_edit_faces(main_window: "MainWindow"):
    """Triggers a single-frame re-process after the Edit Faces button state changes.

    Runs synchronously for the same reason as process_swap_faces.
    """
    video_processor = main_window.video_processor
    video_processor.process_current_frame(synchronous=True)


def process_compare_checkboxes(main_window: "MainWindow"):
    """Triggers a single-frame re-process and view resize after a compare/mask checkbox changes."""
    main_window.video_processor.process_current_frame(fit_on_complete=True)


def save_current_frame_to_file(main_window: "MainWindow"):
    """
    Saves the currently displayed (processed) frame to an image file in the output folder.

    The format (PNG or JPEG) is determined by the ImageFormatToggle control.
    Shows an error message if no output folder is configured or no valid frame is available.
    """
    if not main_window.outputFolderLineEdit.text():
        common_widget_actions.create_and_show_messagebox(
            main_window,
            "No Output Folder Selected",
            "Please select an Output folder to save the Images/Videos before Saving/Recording!",
            main_window,
        )
        return
    output_folder = str(main_window.control.get("OutputMediaFolder", "")).strip()
    if main_window.control.get("OutputToTargetLocationToggle", False):
        output_folder = os.path.dirname(str(main_window.video_processor.media_path))
    if main_window.control.get("ClusterOutputBySourceToggle", False):
        target_face_button = getattr(
            main_window, "cur_selected_target_face_button", None
        )
        assigned_embeddings = (
            getattr(target_face_button, "assigned_merged_embeddings", None)
            if target_face_button
            else None
        )
        embedding_id = (
            next(iter(assigned_embeddings), None) if assigned_embeddings else None
        )
        embedding_button = (
            main_window.merged_embeddings.get(embedding_id)
            if embedding_id is not None
            else None
        )
        embedding_name = (
            str(getattr(embedding_button, "embedding_name", "")).strip()
            if embedding_button is not None
            else ""
        )
        if embedding_name:
            output_folder = os.path.join(output_folder, embedding_name)
    frame = main_window.video_processor.current_frame.copy()
    image_format = "image"
    if main_window.control["ImageFormatToggle"]:
        image_format = "jpegimage"

    if isinstance(frame, numpy.ndarray):
        os.makedirs(output_folder, exist_ok=True)
        save_filename = misc_helpers.get_output_file_path(
            main_window.video_processor.media_path,
            output_folder,
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

    # 4. Confirmation Dialog
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
    progress_dialog = widget_components.ProgressDialog(
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

            if progress_dialog.confirmedCanceled():
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
                    # Update the slider for images
                    main_window.videoSeekSlider.setMaximum(0)

                elif file_type == "video":
                    # Get rotation for batch processing
                    rotation_angle = get_video_rotation(media_path)
                    main_window.video_processor.media_rotation = rotation_angle
                    media_capture = cv2.VideoCapture(media_path)
                    # Explicitly enable OpenCV's auto-rotation ---
                    if hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO"):
                        media_capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
                    if not media_capture.isOpened():
                        raise Exception(f"Could not open video file: {media_path}")

                    main_window.video_processor.media_capture = media_capture
                    max_frames = int(media_capture.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
                    main_window.video_processor.max_frame_number = max_frames
                    main_window.video_processor.fps = media_capture.get(
                        cv2.CAP_PROP_FPS
                    )

                    # Update the slider for this video
                    main_window.videoSeekSlider.blockSignals(True)
                    main_window.videoSeekSlider.setMaximum(max_frames)
                    main_window.videoSeekSlider.blockSignals(False)

                    ret, frame_bgr = misc_helpers.read_frame(
                        media_capture, rotation_angle
                    )
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
                    output_folder = str(
                        main_window.control.get("OutputMediaFolder", "")
                    ).strip()
                    if main_window.control.get("OutputToTargetLocationToggle", False):
                        output_folder = os.path.dirname(str(media_path))
                    if main_window.control.get("ClusterOutputBySourceToggle", False):
                        target_face_button = getattr(
                            main_window, "cur_selected_target_face_button", None
                        )
                        assigned_embeddings = (
                            getattr(
                                target_face_button, "assigned_merged_embeddings", None
                            )
                            if target_face_button
                            else None
                        )
                        embedding_id = (
                            next(iter(assigned_embeddings), None)
                            if assigned_embeddings
                            else None
                        )
                        embedding_button = (
                            main_window.merged_embeddings.get(embedding_id)
                            if embedding_id is not None
                            else None
                        )
                        embedding_name = (
                            str(getattr(embedding_button, "embedding_name", "")).strip()
                            if embedding_button is not None
                            else ""
                        )
                        if embedding_name:
                            output_folder = os.path.join(output_folder, embedding_name)
                    os.makedirs(output_folder, exist_ok=True)
                    save_filename = misc_helpers.get_output_file_path(
                        media_path,
                        output_folder,
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
                        if progress_dialog.confirmedCanceled():
                            print(
                                f"[WARN] Cancel detected during video processing: {media_path}. Aborting..."
                            )
                            # Call the 'abort' function
                            main_window.video_processor.stop_processing()
                            # 'stop_processing' sets .processing to False,
                            # so the loop will exit.
                            break

                        QtCore.QThread.msleep(1)  # 1ms sleep

                    # 3. At this point, record_video has completed (or been aborted)
                    # We must check *again* if the loop was exited due to cancellation
                    # to avoid incorrectly incrementing the 'processed_count'.
                    if not progress_dialog.confirmedCanceled():
                        print(f"[INFO] Finished processing video: {media_path}")
                        processed_count += 1
                    else:
                        print(
                            f"[WARN] Video processing was cancelled for: {media_path}"
                        )

            except Exception as e:
                # Log the error for this specific file and continue
                print(f"[ERROR] Failed to process {media_path}: {e}")
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
        if progress_dialog.confirmedCanceled():
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

        # Do not restore the old capture object
        main_window.video_processor.media_capture = None  # original_media_capture

        # Restore the view to its original state
        if main_window.video_processor.media_path:
            # Reload and re-process the frame that was active before the batch
            if main_window.video_processor.file_type == "video":
                # --- Re-open the original video capture ---
                print(f"[INFO] Restoring original video capture: {original_media_path}")
                new_capture = cv2.VideoCapture(original_media_path)
                # Explicitly enable OpenCV's auto-rotation ---
                if hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO"):
                    new_capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
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

                main_window.video_processor.process_current_frame(synchronous=True)

            elif main_window.video_processor.file_type == "image":
                # Correctly restore the slider for the image
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
            # Reset the slider
            main_window.videoSeekSlider.blockSignals(True)
            main_window.videoSeekSlider.setMaximum(0)
            main_window.videoSeekSlider.setValue(0)
            main_window.videoSeekSlider.blockSignals(False)
            main_window.video_processor.max_frame_number = 0


def toggle_live_sound(main_window: "MainWindow", toggle_value: bool):
    """
    Enables or disables live audio playback during video preview.

    If video was already playing, it is stopped and restarted so the audio
    subprocess is created (or destroyed) with the new setting in effect.
    """
    video_processor = main_window.video_processor
    was_processing = video_processor.processing

    # If the video was playing, then stop and start it again to enable the audio
    # Otherwise, just the toggle value so that the next time the play button is hit, it would automatically enable/disable the audio
    # The play button is clicked twice in the below block to simulate the above mentioned behaviour. It should be changed into a set up in the next refactor
    if was_processing:
        main_window.buttonMediaPlay.click()
        main_window.buttonMediaPlay.click()


def _set_media_controls_visible(main_window: "MainWindow", visible: bool):
    """
    Role: Recursively shows/hides media control widgets and safely manages layout spacers.
    Impact: Avoids blank spaces (cadres) by cleanly detaching spacers using takeAt()
            instead of forcing their sizes to 0.
    """
    if not hasattr(main_window, "_media_controls_currently_visible"):
        main_window._media_controls_currently_visible = True

    if main_window._media_controls_currently_visible == visible:
        return  # State unchanged

    main_window._media_controls_currently_visible = visible

    if not visible:
        main_window._media_spacers_storage = []

        def hide_and_remove(layout):
            # Iterate backwards to safely use takeAt() without breaking layout indices
            for i in reversed(range(layout.count())):
                item = layout.itemAt(i)
                if item.widget():
                    item.widget().hide()
                elif item.spacerItem():
                    spacer = layout.takeAt(i)
                    main_window._media_spacers_storage.append((layout, i, spacer))
                elif item.layout():
                    hide_and_remove(item.layout())

        hide_and_remove(main_window.verticalLayoutMediaControls)
    else:
        # Restore spacers safely in the exact original order
        storage = getattr(main_window, "_media_spacers_storage", [])
        for layout, i, spacer in reversed(storage):
            layout.insertItem(i, spacer)
        main_window._media_spacers_storage = []

        def show_widgets(layout):
            for i in range(layout.count()):
                item = layout.itemAt(i)
                if item.widget():
                    item.widget().show()
                elif item.layout():
                    show_widgets(item.layout())

        show_widgets(main_window.verticalLayoutMediaControls)

    main_window.verticalLayout.invalidate()


def toggle_theatre_mode(main_window: "MainWindow"):
    """
    Role: Activates Theatre Mode by safely detaching UI elements and preventing layout crushing.
    Impact: Solves the "squashed text" bug and preserves exact QDockWidget sizes using saveState/restoreState.
    """

    is_theatre = getattr(main_window, "is_theatre_mode", False)
    if not is_theatre:
        # --- ENTER THEATRE MODE ---
        main_window.is_theatre_mode = True

        # 0. Save the exact state of all docks and toolbars (sizes, proportions, splitters)
        main_window._saved_window_state = main_window.saveState()
        main_window._was_maximized = main_window.isMaximized()
        main_window._was_custom_fullscreen = main_window.isFullScreen()
        if main_window.isMaximized() or main_window.isFullScreen():
            main_window._was_normal_geometry = main_window.normalGeometry()
        else:
            main_window._was_normal_geometry = main_window.geometry()

        # 1. Save state and hide Docks, MenuBar
        main_window._saved_dock_states = {
            "input_Target_DockWidget": main_window.input_Target_DockWidget.isVisible(),
            "input_Faces_DockWidget": main_window.input_Faces_DockWidget.isVisible(),
            "jobManagerDockWidget": main_window.jobManagerDockWidget.isVisible(),
            "controlOptionsDockWidget": main_window.controlOptionsDockWidget.isVisible(),
            "menuBar": main_window.menuBar().isVisible(),
            "facesPanelGroupBox": main_window.facesPanelGroupBox.isVisible(),
        }

        main_window.input_Target_DockWidget.hide()
        main_window.input_Faces_DockWidget.hide()
        main_window.jobManagerDockWidget.hide()
        main_window.controlOptionsDockWidget.hide()
        main_window.menuBar().hide()
        main_window.facesPanelGroupBox.hide()

        # 2. Save and remove layout margins/spacings (Removes borders)
        main_window._saved_layout_props = {
            "h_margin": main_window.horizontalLayout.contentsMargins(),
            "v_margin": main_window.verticalLayout.contentsMargins(),
            "h_spacing": main_window.horizontalLayout.spacing(),
            "v_spacing": main_window.verticalLayout.spacing(),
            "frame_shape": main_window.graphicsViewFrame.frameShape(),
            "media_controls_margin": main_window.verticalLayoutMediaControls.contentsMargins(),
        }

        main_window.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        main_window.verticalLayout.setContentsMargins(0, 0, 0, 0)
        main_window.horizontalLayout.setSpacing(0)
        main_window.verticalLayout.setSpacing(0)
        main_window.verticalLayoutMediaControls.setContentsMargins(0, 0, 0, 16)

        # 3. Safely detach spacers and hide Top Bar widgets
        main_window._top_bar_spacers = []
        main_window._top_bar_widgets_state = {}

        for i in reversed(range(main_window.panelVisibilityCheckBoxLayout.count())):
            item = main_window.panelVisibilityCheckBoxLayout.itemAt(i)
            if item.widget():
                main_window._top_bar_widgets_state[item.widget()] = (
                    item.widget().isVisible()
                )
                item.widget().hide()
            elif item.spacerItem():
                spacer = main_window.panelVisibilityCheckBoxLayout.takeAt(i)
                main_window._top_bar_spacers.append(
                    (main_window.panelVisibilityCheckBoxLayout, i, spacer)
                )

        # Detach main layout stray spacers (top/bottom empty spaces)
        main_window._main_v_spacers = []
        for i in reversed(range(main_window.verticalLayout.count())):
            item = main_window.verticalLayout.itemAt(i)
            if item.spacerItem():
                spacer = main_window.verticalLayout.takeAt(i)
                main_window._main_v_spacers.append(
                    (main_window.verticalLayout, i, spacer)
                )

        # 4. Hide media controls safely
        main_window._media_controls_currently_visible = True
        _set_media_controls_visible(main_window, False)

        # 5. Clean GraphicsView Frame for edge-to-edge video
        main_window.graphicsViewFrame.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        main_window.graphicsViewFrame.setStyleSheet("background-color: black;")
        main_window.graphicsViewFrame.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        main_window.graphicsViewFrame.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )

        # 6. Preserve the base window mode; only remain fullscreen if already fullscreen.
        if main_window._was_custom_fullscreen:
            main_window.setWindowState(QtCore.Qt.WindowState.WindowFullScreen)
            main_window.showFullScreen()
            QtWidgets.QApplication.processEvents()
            main_window.is_full_screen = True
        else:
            main_window.is_full_screen = False

        # 7. Activate Hover Timer for media controls overlay
        if not hasattr(main_window, "theatre_hover_timer"):
            main_window.theatre_hover_timer = QtCore.QTimer(main_window)
            main_window.theatre_hover_timer.setInterval(100)

            def check_mouse_pos():
                if not getattr(main_window, "is_theatre_mode", False):
                    return
                mouse_pos = main_window.mapFromGlobal(QtGui.QCursor.pos())
                if mouse_pos.y() > main_window.height() * 0.85:
                    _set_media_controls_visible(main_window, True)
                else:
                    _set_media_controls_visible(main_window, False)

            main_window.theatre_hover_timer.timeout.connect(check_mouse_pos)

        main_window.theatre_hover_timer.start()

    else:
        # --- EXIT THEATRE MODE ---
        main_window.is_theatre_mode = False
        main_window.setUpdatesEnabled(False)

        if hasattr(main_window, "theatre_hover_timer"):
            main_window.theatre_hover_timer.stop()

        # 1. Force media controls to become visible and restore their spacers
        _set_media_controls_visible(main_window, True)

        # 2. Re-attach main layout stray spacers
        for layout, i, spacer in reversed(getattr(main_window, "_main_v_spacers", [])):
            layout.insertItem(i, spacer)
        main_window._main_v_spacers = []

        # 3. Re-attach top bar spacers and restore widget visibility
        for layout, i, spacer in reversed(getattr(main_window, "_top_bar_spacers", [])):
            layout.insertItem(i, spacer)
        main_window._top_bar_spacers = []

        for i in range(main_window.panelVisibilityCheckBoxLayout.count()):
            item = main_window.panelVisibilityCheckBoxLayout.itemAt(i)
            if item.widget():
                was_visible = getattr(main_window, "_top_bar_widgets_state", {}).get(
                    item.widget(), True
                )
                item.widget().setVisible(was_visible)

        # 4. Restore Main Layout Props (restores spaces and margins)
        props = getattr(main_window, "_saved_layout_props", {})
        if "h_margin" in props:
            main_window.horizontalLayout.setContentsMargins(props["h_margin"])
        if "v_margin" in props:
            main_window.verticalLayout.setContentsMargins(props["v_margin"])
        if "h_spacing" in props:
            main_window.horizontalLayout.setSpacing(props["h_spacing"])
        if "v_spacing" in props:
            main_window.verticalLayout.setSpacing(props["v_spacing"])
        if "frame_shape" in props:
            main_window.graphicsViewFrame.setFrameShape(props["frame_shape"])
        if "media_controls_margin" in props:
            main_window.verticalLayoutMediaControls.setContentsMargins(
                props["media_controls_margin"]
            )

        main_window.graphicsViewFrame.setStyleSheet("")
        main_window.graphicsViewFrame.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        main_window.graphicsViewFrame.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )

        # 5. Restore Docks & Panels
        states = getattr(main_window, "_saved_dock_states", {})
        if states.get("input_Target_DockWidget"):
            main_window.input_Target_DockWidget.show()
        if states.get("input_Faces_DockWidget"):
            main_window.input_Faces_DockWidget.show()
        if states.get("jobManagerDockWidget"):
            main_window.jobManagerDockWidget.show()
        if states.get("controlOptionsDockWidget"):
            main_window.controlOptionsDockWidget.show()
        if states.get("menuBar"):
            main_window.menuBar().show()
        if states.get("facesPanelGroupBox"):
            main_window.facesPanelGroupBox.show()
            hint_height = main_window.facesPanelGroupBox.sizeHint().height()
            main_window.facesPanelGroupBox.setMinimumHeight(hint_height)

        # Exit Fullscreen mode
        was_custom_fullscreen = getattr(main_window, "_was_custom_fullscreen", False)
        was_maximized = getattr(main_window, "_was_maximized", False)
        saved_normal_geometry = getattr(main_window, "_was_normal_geometry", None)

        _restore_window_base_mode(
            main_window,
            restore_to_fullscreen=was_custom_fullscreen,
            restore_to_maximized=was_maximized and not was_custom_fullscreen,
            restore_geometry=saved_normal_geometry,
        )

        # Restores the exact layout state of docks (fixes the Input Faces / Target Video sizing)
        if hasattr(main_window, "_saved_window_state"):
            main_window.restoreState(main_window._saved_window_state)

        # Force the application layout engine to process the new geometry immediately
        QtWidgets.QApplication.processEvents()
        main_window.verticalLayout.invalidate()

        # Release the minimum height lock so the UI is dynamically responsive again
        if states.get("facesPanelGroupBox"):
            main_window.facesPanelGroupBox.setMinimumHeight(0)

        main_window.setUpdatesEnabled(True)

    layout_actions.fit_image_to_view_onchange(main_window)


def _run_context_menu_theatre_toggle(main_window: "MainWindow") -> None:
    toggle_theatre_mode(main_window)
    graphics_view = main_window.graphicsViewFrame
    graphics_view.update()
    graphics_view.viewport().update()
    graphics_view.repaint()
    graphics_view.viewport().repaint()
    main_window.update()
    main_window._sync_viewer_menu_actions()
