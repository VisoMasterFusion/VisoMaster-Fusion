from typing import TYPE_CHECKING
import copy
import os
from functools import partial

from PySide6.QtCore import QPoint
from PySide6.QtWidgets import QMenu
import cv2
import numpy
from PIL import Image
from PySide6 import QtGui,QtWidgets,QtCore

if TYPE_CHECKING:
    from app.ui.main_ui import MainWindow
import app.helpers.miscellaneous as misc_helpers
from app.ui.widgets.actions import common_actions as common_widget_actions
from app.ui.widgets.actions import graphics_view_actions
import app.ui.widgets.actions.layout_actions as layout_actions

def set_up_video_seek_line_edit(main_window: 'MainWindow'):
    video_processor = main_window.video_processor
    videoSeekLineEdit = main_window.videoSeekLineEdit
    videoSeekLineEdit.setAlignment(QtCore.Qt.AlignCenter)
    videoSeekLineEdit.setText('0')
    videoSeekLineEdit.setValidator(QtGui.QIntValidator(0, video_processor.max_frame_number))  # Restrict input to numbers 

def set_up_video_seek_slider(main_window: 'MainWindow'):
    main_window.videoSeekSlider.markers = set()  # Store unique tick positions
    main_window.videoSeekSlider.setTickPosition(QtWidgets.QSlider.TickPosition.TicksBelow)  # Default position for tick marks

    def add_marker_and_paint(self: QtWidgets.QSlider, value=None):
        """Add a tick mark at a specific slider value."""
        if value is None or isinstance(value, bool):  # Default to current slider value
            value = self.value()
        if self.minimum() <= value <= self.maximum() and value not in self.markers:
            self.markers.add(value)
            self.update()

    def remove_marker_and_paint(self:QtWidgets.QSlider, value=None):
        """Remove a tick mark."""
        if value is None or isinstance(value, bool):  # Default to current slider value
            value = self.value()
        if value in self.markers:
            self.markers.remove(value)
            self.update()

    def paintEvent(self: QtWidgets.QSlider, event:QtGui.QPaintEvent):
        # Dont need a seek slider if the current selected file is an image
        if main_window.video_processor.file_type=='image':
            return super(QtWidgets.QSlider, self).paintEvent(event)
        # Set up the painter and style option
        painter = QtWidgets.QStylePainter(self)
        opt = QtWidgets.QStyleOptionSlider()
        self.initStyleOption(opt)
        style = self.style()

        # Get groove and handle geometry
        groove_rect = style.subControlRect(
            QtWidgets.QStyle.ComplexControl.CC_Slider, opt, QtWidgets.QStyle.SubControl.SC_SliderGroove
        )
        groove_y = (groove_rect.top() + groove_rect.bottom()) // 2  # Groove's vertical center
        groove_start = groove_rect.left()
        groove_end = groove_rect.right()
        groove_width = groove_end - groove_start

        # Calculate handle position based on the current slider value
        normalized_value = (self.value() - self.minimum()) / (self.maximum() - self.minimum())
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
        painter.setPen(QtGui.QPen(QtGui.QColor("gray"), 3))  # Groove color and thickness
        painter.drawLine(groove_start, groove_y, groove_end, groove_y)

        # Draw the thin handle
        painter.setPen(QtGui.QPen(QtGui.QColor("white"), 1))  # Handle border color
        painter.setBrush(QtGui.QBrush(QtGui.QColor("white")))  # Handle fill color
        painter.drawRect(handle_rect)

        # Draw markers (if any)
        if self.markers:
            painter.setPen(QtGui.QPen(QtGui.QColor("#4090a3"), 3))  # Marker color and thickness
            for value in sorted(self.markers):
                # Calculate marker position
                marker_normalized_value = (value - self.minimum()) / (self.maximum() - self.minimum())
                marker_x = groove_start + marker_normalized_value * groove_width
                painter.drawLine(marker_x, groove_rect.top(), marker_x, groove_rect.bottom())
        # Draw Job Start/End Brackets on the groove line
        painter.setFont(QtGui.QFont('Arial', 16, QtGui.QFont.Bold)) # Increased font size from 12 to 16
        font_metrics = painter.fontMetrics()
        bracket_height = font_metrics.height()
        bracket_y_pos = groove_y + (bracket_height // 4)

        # Iterate through all defined job marker pairs
        for start_frame, end_frame in main_window.job_marker_pairs:
            if start_frame is not None:
                start_normalized_value = (start_frame - self.minimum()) / (self.maximum() - self.minimum())
                start_x = groove_start + start_normalized_value * groove_width
                # Draw the green start bracket
                painter.setPen(QtGui.QPen(QtGui.QColor("#4CAF50"), 1)) # Green for start bracket
                painter.drawText(int(start_x - 4), int(bracket_y_pos), '[') # Adjusted X offset slightly

            if end_frame is not None:
                end_normalized_value = (end_frame - self.minimum()) / (self.maximum() - self.minimum())
                end_x = groove_start + end_normalized_value * groove_width
                # Draw the red end bracket
                painter.setPen(QtGui.QPen(QtGui.QColor("#e8483c"), 1)) # Red for end bracket
                painter.drawText(int(end_x - 4), int(bracket_y_pos), ']') # Adjusted X offset slightly

    main_window.videoSeekSlider.add_marker_and_paint = partial(add_marker_and_paint, main_window.videoSeekSlider)
    main_window.videoSeekSlider.remove_marker_and_paint = partial(remove_marker_and_paint, main_window.videoSeekSlider)
    main_window.videoSeekSlider.paintEvent = partial(paintEvent, main_window.videoSeekSlider)

def add_video_slider_marker(main_window: 'MainWindow'):
    if main_window.selected_video_button.file_type!='video':
        common_widget_actions.create_and_show_messagebox(main_window, 'Markers Not Available', 'Markers can only be used for videos!', main_window.videoSeekSlider)
        return
    current_position = int(main_window.videoSeekSlider.value())
    # print("current_position", current_position)
    if not main_window.target_faces:
        common_widget_actions.create_and_show_messagebox(main_window, 'No Target Face Found', 'You need to have at least one target face to create a marker', main_window.videoSeekSlider)
    elif main_window.markers.get(current_position):
        common_widget_actions.create_and_show_messagebox(main_window, 'Marker Already Exists!', 'A Marker already exists for this position!', main_window.videoSeekSlider)
    else:
        add_marker(main_window, copy.deepcopy(main_window.parameters), main_window.control.copy(), current_position)

def show_add_marker_menu(main_window: 'MainWindow'):
    """Shows a context menu for adding different types of markers."""
    if not main_window.selected_video_button or main_window.selected_video_button.file_type != 'video':
        common_widget_actions.create_and_show_messagebox(main_window, 'Markers Not Available', 'Markers can only be used for videos!', main_window.videoSeekSlider)
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
        if last_pair[1] is None: # Last pair is incomplete (start set, end not set)
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

def set_job_start_frame(main_window: 'MainWindow'):
    """Adds a new job marker pair starting at the current slider position."""
    current_pos = int(main_window.videoSeekSlider.value())

    # Basic validation: Ensure we are not adding a start if the last pair is incomplete
    if main_window.job_marker_pairs and main_window.job_marker_pairs[-1][1] is None:
         QtWidgets.QMessageBox.warning(main_window, "Invalid Action",
                                     "Cannot add a new Start marker before completing the previous End marker.")
         return

    # Add the new start marker (end frame is initially None)
    main_window.job_marker_pairs.append((current_pos, None))
    main_window.videoSeekSlider.update() # Trigger repaint to show the new marker
    print(f"Job Start Marker added for pair {len(main_window.job_marker_pairs)} at Frame: {current_pos}")

def set_job_end_frame(main_window: 'MainWindow'):
    """Sets the job end frame marker for the last incomplete pair."""
    current_pos = int(main_window.videoSeekSlider.value())

    # Validation: Check if there's an incomplete pair to add an end to
    if not main_window.job_marker_pairs or main_window.job_marker_pairs[-1][1] is not None:
        QtWidgets.QMessageBox.critical(main_window, "Error",
                                     "Cannot set End marker without a preceding Start marker.")
        return

    last_pair_index = len(main_window.job_marker_pairs) - 1
    start_frame = main_window.job_marker_pairs[last_pair_index][0]

    # Validation: Check end frame is after start frame
    if current_pos <= start_frame:
        QtWidgets.QMessageBox.warning(main_window, "Invalid Position",
                                    "Job end frame must be after the job start frame.")
        return

    # Update the last pair with the end frame
    main_window.job_marker_pairs[last_pair_index] = (start_frame, current_pos)
    main_window.videoSeekSlider.update() # Trigger repaint to show the new marker
    print(f"Job End Marker added for pair {last_pair_index + 1} at Frame: {current_pos}")

def remove_video_slider_marker(main_window: 'MainWindow'):
    if not main_window.selected_video_button or main_window.selected_video_button.file_type != 'video':
        common_widget_actions.create_and_show_messagebox(main_window, 'Markers Not Available', 'Markers can only be used for videos!', main_window.videoSeekSlider)
        return

    current_position = int(main_window.videoSeekSlider.value())
    pair_removed = False

    new_marker_pairs = []
    removed_pair_indices = []
    for i, (start_frame, end_frame) in enumerate(main_window.job_marker_pairs):
        if start_frame == current_position or end_frame == current_position:
            print(f"Removing Job Marker Pair {i+1} ({start_frame}, {end_frame}) because marker found at position: {current_position}")
            removed_pair_indices.append(i)
            pair_removed = True

    main_window.job_marker_pairs = [
        pair for i, pair in enumerate(main_window.job_marker_pairs) 
        if i not in removed_pair_indices
    ]

    if pair_removed:
        main_window.videoSeekSlider.update()
        return

    if main_window.markers.get(current_position):
        remove_marker(main_window, current_position)
    else:
        common_widget_actions.create_and_show_messagebox(main_window, 'No Marker Found!', 'No Marker Found for this position!', main_window.videoSeekSlider)

def add_marker(main_window: 'MainWindow', parameters, control, position,):
    main_window.videoSeekSlider.add_marker_and_paint(position)
    main_window.markers[position] = {'parameters': parameters, 'control': control}
    print(f"Marker Added for Frame: {position}")

def remove_marker(main_window: 'MainWindow', position):
    if main_window.markers.get(position):
        main_window.videoSeekSlider.remove_marker_and_paint(position)
        main_window.markers.pop(position)
        print(f"Marker Removed from position: {position}")

def move_slider_to_nearest_marker(main_window: 'MainWindow', direction: str):
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
        return # No markers to navigate to

    sorted_markers = sorted(list(all_markers))

    if direction == "next":
        filtered_markers = [marker for marker in sorted_markers if marker > current_position]
        new_position = filtered_markers[0] if filtered_markers else None
    elif direction == "previous":
        filtered_markers = [marker for marker in sorted_markers if marker < current_position]
        new_position = filtered_markers[-1] if filtered_markers else None

    if new_position is not None:
        main_window.videoSeekSlider.setValue(new_position)
        main_window.video_processor.process_current_frame()

# Wrappers for specific directions
def move_slider_to_next_nearest_marker(main_window: 'MainWindow'):
    move_slider_to_nearest_marker(main_window, "next")

def move_slider_to_previous_nearest_marker(main_window: 'MainWindow'):
    move_slider_to_nearest_marker(main_window, "previous")

def remove_face_parameters_and_control_from_markers(main_window: 'MainWindow', face_id):
    for _, marker_data in main_window.markers.items():
        marker_data['parameters'].pop(face_id, None) # Use .pop with default to avoid KeyError
        # If the parameters is empty, then there is no longer any marker to be set for any target face
        if not marker_data['parameters']:
            delete_all_markers(main_window)
            break

def remove_all_markers(main_window: 'MainWindow'):
    standard_markers_positions = list(main_window.markers.keys())
    for marker_position in standard_markers_positions: remove_marker(main_window, marker_position)
    if main_window.job_marker_pairs:
        print("Clearing job marker pairs.")
        main_window.job_marker_pairs.clear(); main_window.videoSeekSlider.update()

def advance_video_slider_by_n_frames(main_window: 'MainWindow', n=30):
    video_processor = main_window.video_processor
    if video_processor.media_capture:
        current_position = int(main_window.videoSeekSlider.value())
        new_position = current_position + n
        if new_position > video_processor.max_frame_number:
            new_position = video_processor.max_frame_number
        main_window.videoSeekSlider.setValue(new_position)
        main_window.video_processor.process_current_frame()

def rewind_video_slider_by_n_frames(main_window: 'MainWindow', n=30):
    video_processor = main_window.video_processor
    if video_processor.media_capture:
        current_position = int(main_window.videoSeekSlider.value())
        new_position = current_position - n
        if new_position < 0:
            new_position = 0
        main_window.videoSeekSlider.setValue(new_position)
        main_window.video_processor.process_current_frame()

def delete_all_markers(main_window: 'MainWindow'):
    main_window.videoSeekSlider.markers = set()
    main_window.videoSeekSlider.update()
    main_window.markers = {}

def view_fullscreen(main_window: 'MainWindow'):

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

    def zoom(self:QtWidgets.QGraphicsView, step=False):
        """Zoom in or out by a step."""
        if not step:
            factor = self.last_scale_factor
        else:
            self.zoom_value += step
            factor = SCALE_FACTOR ** step
            self.last_scale_factor *= factor  # Update the last scale factor
        if factor > 0:
            self.scale(factor, factor)

    def wheelEvent(self:QtWidgets.QGraphicsView, event:QtGui.QWheelEvent):
        """Handle mouse wheel event for zooming."""
        delta = event.angleDelta().y()
        if delta != 0:
            zoom(self, delta // abs(delta))
    
    def reset_zoom(self:QtWidgets.QGraphicsView):
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
        factor = min(view_rect.width() / scene_rect.width(),
                    view_rect.height() / scene_rect.height())
        self.scale(factor, factor)

    def mousePressEvent(self: QtWidgets.QGraphicsView, event: QtGui.QMouseEvent):
        """Handle mouse press event for panning."""
        if event.button() == QtCore.Qt.MouseButton.RightButton:
            self.is_panning = True
            self.pan_start_pos = event.pos()  # Store the initial mouse position
            self.setCursor(QtCore.Qt.ClosedHandCursor)  # Change cursor to indicate panning
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
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
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
    view.setTransformationAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorUnderMouse)
    view.setResizeAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorUnderMouse)


def play_video(main_window: 'MainWindow', checked: bool):
    video_processor = main_window.video_processor
    if checked:
        if video_processor.processing or video_processor.current_frame_number==video_processor.max_frame_number:
            print("play_video: Video already playing. Stopping the current video before starting a new one.")
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


def record_video(main_window: 'MainWindow', checked: bool):
    video_processor = main_window.video_processor
    # Determine if this record action was initiated by the Job Manager
    job_mgr_flag = getattr(main_window, 'job_manager_initiated_record', False)
    if video_processor.file_type not in ['video', 'image']:
        main_window.buttonMediaRecord.blockSignals(True)
        main_window.buttonMediaRecord.setChecked(False)
        main_window.buttonMediaRecord.blockSignals(False)
        if video_processor.file_type == 'webcam':
            common_widget_actions.create_and_show_messagebox(main_window, 'Recording Not Supported', 'Recording webcam stream is not supported yet.', main_window)
        return
    
    if checked:
        if video_processor.processing or video_processor.is_processing_segments:
            print("record_video: Processing already active. Request ignored.")
            main_window.buttonMediaRecord.blockSignals(True)
            main_window.buttonMediaRecord.setChecked(True)
            main_window.buttonMediaRecord.blockSignals(False)
            set_record_button_icon_to_stop(main_window)
            return
        if not main_window.control.get('OutputMediaFolder','').strip():
            common_widget_actions.create_and_show_messagebox(main_window, 'No Output Folder Selected','Please select an Output folder before recording!', main_window)
            main_window.buttonMediaRecord.setChecked(False) # Uncheck the button
            return
        if not misc_helpers.is_ffmpeg_in_path():
            common_widget_actions.create_and_show_messagebox(main_window, 'FFMPEG Not Found','FFMPEG was not found in your system. Check installation!', main_window)
            main_window.buttonMediaRecord.setChecked(False) # Uncheck the button
            return
        
        marker_pairs = main_window.job_marker_pairs
        if not marker_pairs: # NO MARKERS SET -> Default Recording Style
            # --- Validate start position for default recording ---
            current_frame = main_window.videoSeekSlider.value()
            max_frame = video_processor.max_frame_number
            if max_frame is None or max_frame <= 0:
                 common_widget_actions.create_and_show_messagebox(main_window, 'Error', 'Cannot determine video length.', main_window)
                 main_window.buttonMediaRecord.setChecked(False)
                 return
            if current_frame >= max_frame:
                common_widget_actions.create_and_show_messagebox(
                    main_window, 'Recording Error',
                    f'Cannot start recording from frame {current_frame}. Scrubber is at or past the end of the video ({max_frame}).',
                    main_window
                )
                main_window.buttonMediaRecord.setChecked(False)
                return
            # --- Proceed with Default Recording ---
            print("Record button pressed: Starting default recording from current position.")
            set_record_button_icon_to_stop(main_window)
            # Disable play button during recording
            main_window.buttonMediaPlay.setEnabled(False)
            video_processor.recording = True # SET THE FLAG FOR DEFAULT RECORDING
            video_processor.process_video()  # CALL THE DEFAULT PROCESSOR

        else: # MARKERS ARE SET -> Multi-Segment Recording Style
            # --- Validate Marker Pairs ---
            valid_pairs = []
            for i, pair in enumerate(marker_pairs):
                if pair[1] is None:
                    common_widget_actions.create_and_show_messagebox(main_window, 'Incomplete Segment', f'Marker pair {i+1} ({pair[0]}, None) is incomplete. Please set an End marker.', main_window)
                    main_window.buttonMediaRecord.setChecked(False)
                    return # Stop if invalid
                elif pair[0] >= pair[1]:
                    common_widget_actions.create_and_show_messagebox(main_window, 'Invalid Segment', f'Marker pair {i+1} ({pair[0]}, {pair[1]}) is invalid. Start must be before End.', main_window)
                    main_window.buttonMediaRecord.setChecked(False)
                    return # Stop if invalid
                else:
                    valid_pairs.append(pair)
            # --- End Validation ---

            # Proceed if we have valid marker pairs
            if valid_pairs:
                print(f"Record button pressed: Starting multi-segment recording for {len(valid_pairs)} segment(s).")
                set_record_button_icon_to_stop(main_window)
                # Disable play button during segment recording
                main_window.buttonMediaPlay.setEnabled(False)
                is_job_context = job_mgr_flag
                print(f"[DEBUG] record_video: job_mgr_flag = {is_job_context}")
                video_processor.start_multi_segment_recording(
                    valid_pairs,
                    triggered_by_job_manager=is_job_context
                )
                try:
                    main_window.job_manager_initiated_record = False
                except Exception:
                    pass
            else:
                print("[WARN] Recording not started due to invalid marker configuration.")

    else:
        if video_processor.is_processing_segments:
            print("Record button released: User requested stop during segment processing. Finalizing...")
            # Finalize segment concatenation with segments processed so far
            video_processor.finalize_segment_concatenation()
        elif video_processor.recording: # Check if default style recording was active
            print("Record button released: User requested stop during default recording. Finalizing...")
            # Finalize the default style recording
            video_processor._finalize_default_style_recording()
        else:
            # No recording was active (maybe an immediate click-off or already stopped)
            print("Record button released: No active recording found.")
            set_record_button_icon_to_play(main_window)
            main_window.buttonMediaPlay.setEnabled(True)
            reset_media_buttons(main_window)

def set_record_button_icon_to_play(main_window: 'MainWindow'):
    main_window.buttonMediaRecord.setIcon(QtGui.QIcon(":/media/media/rec_off.png"))
    main_window.buttonMediaRecord.setToolTip("Start Recording")
def set_record_button_icon_to_stop(main_window: 'MainWindow'):
    main_window.buttonMediaRecord.setIcon(QtGui.QIcon(":/media/media/rec_on.png"))
    main_window.buttonMediaRecord.setToolTip("Stop Recording")

def set_play_button_icon_to_play(main_window: 'MainWindow'):
    main_window.buttonMediaPlay.setIcon(QtGui.QIcon(":/media/media/play_off.png"))
    main_window.buttonMediaPlay.setToolTip("Play")

def set_play_button_icon_to_stop(main_window: 'MainWindow'):
    main_window.buttonMediaPlay.setIcon(QtGui.QIcon(":/media/media/play_on.png"))
    main_window.buttonMediaPlay.setToolTip("Stop")

def reset_media_buttons(main_window: 'MainWindow'):
    # Rest the state and icons of the buttons without triggering Onchange methods
    main_window.buttonMediaPlay.blockSignals(True)
    main_window.buttonMediaPlay.setChecked(False)
    main_window.buttonMediaPlay.setEnabled(True) # Re-enable the button
    main_window.buttonMediaPlay.blockSignals(False)
    main_window.buttonMediaRecord.blockSignals(True)
    main_window.buttonMediaRecord.setChecked(False)
    main_window.buttonMediaRecord.blockSignals(False)
    set_play_button_icon(main_window)
    set_record_button_icon(main_window)

def set_play_button_icon(main_window: 'MainWindow'):
    if main_window.buttonMediaPlay.isChecked(): 
        main_window.buttonMediaPlay.setIcon(QtGui.QIcon(":/media/media/play_on.png"))
        main_window.buttonMediaPlay.setToolTip("Stop")
    else:
        main_window.buttonMediaPlay.setIcon(QtGui.QIcon(":/media/media/play_off.png"))
        main_window.buttonMediaPlay.setToolTip("Play")

def set_record_button_icon(main_window: 'MainWindow'):
    if main_window.buttonMediaRecord.isChecked(): 
        main_window.buttonMediaRecord.setIcon(QtGui.QIcon(":/media/media/rec_on.png"))
        main_window.buttonMediaRecord.setToolTip("Stop Recording")
    else:
        main_window.buttonMediaRecord.setIcon(QtGui.QIcon(":/media/media/rec_off.png"))
        main_window.buttonMediaRecord.setToolTip("Start Recording")

# @misc_helpers.benchmark
@QtCore.Slot(int)
def on_change_video_seek_slider(main_window: 'MainWindow', new_position=0):
    # print("Called on_change_video_seek_slider()")
    video_processor = main_window.video_processor

    was_processing = video_processor.stop_processing()
    if was_processing:
        print("on_change_video_seek_slider: Processing in progress. Stopping current processing.")

    video_processor.current_frame_number = new_position
    video_processor.next_frame_to_display = new_position
    if video_processor.media_capture:
        video_processor.media_capture.set(cv2.CAP_PROP_POS_FRAMES, new_position)
        ret, frame = misc_helpers.read_frame(video_processor.media_capture)
        if ret:
            pixmap = common_widget_actions.get_pixmap_from_frame(main_window, frame)
            graphics_view_actions.update_graphics_view(main_window, pixmap, new_position)
            # if video_processor.current_frame_number == video_processor.max_frame_number:
            #     video_processor.media_capture.set(cv2.CAP_PROP_POS_FRAMES, new_position)
            update_parameters_and_control_from_marker(main_window, new_position)
            update_widget_values_from_markers(main_window, new_position)
        else:
            main_window.last_seek_read_failed = True
            
    # Do not automatically restart the video, let the user press Play to resume
    # print("on_change_video_seek_slider: Video stopped after slider movement.")

def update_parameters_and_control_from_marker(main_window: 'MainWindow', new_position: int):
    marker_data = main_window.markers.get(new_position)
    if marker_data:
        # Load parameters from the marker as a base
        loaded_marker_params = copy.deepcopy(marker_data['parameters'])
        main_window.parameters = loaded_marker_params
        active_target_face_ids = list(main_window.target_faces.keys()) # Get current face IDs
        for face_id_key in active_target_face_ids:
            # common_actions.create_parameter_dict_for_face_id handles if face_id already exists
            common_widget_actions.create_parameter_dict_for_face_id(main_window, face_id_key)
        # Update control settings
        main_window.control.update(marker_data['control'].copy())

def update_widget_values_from_markers(main_window: 'MainWindow', new_position: int):
    if main_window.markers.get(new_position):
        if main_window.selected_target_face_id is not None: common_widget_actions.set_widgets_values_using_face_id_parameters(main_window, main_window.selected_target_face_id)
        common_widget_actions.set_control_widgets_values(main_window, enable_exec_func=False)

def on_slider_moved(main_window: 'MainWindow'):
    # print("Called on_slider_moved()")
    position = main_window.videoSeekSlider.value()
    # print(f"\nSlider Moved. position: {position}\n")

def on_slider_pressed(main_window: 'MainWindow'):

    position = main_window.videoSeekSlider.value()
    # print(f"\nSlider Pressed. position: {position}\n")

# @misc_helpers.benchmark
def on_slider_released(main_window: 'MainWindow'):
    # print("Called on_slider_released()")

    new_position = main_window.videoSeekSlider.value()
    # print(f"\nSlider released. New position: {new_position}\n")
    # Perform the update to the new frame
    video_processor = main_window.video_processor
    if video_processor.media_capture:
        video_processor.process_current_frame()  # Process the current frame

def process_swap_faces(main_window: 'MainWindow'):
    video_processor = main_window.video_processor
    video_processor.process_current_frame()

def process_edit_faces(main_window: 'MainWindow'):
    video_processor = main_window.video_processor
    video_processor.process_current_frame()

def process_compare_checkboxes(main_window: 'MainWindow'):
    main_window.video_processor.process_current_frame()
    layout_actions.fit_image_to_view_onchange(main_window)

def save_current_frame_to_file(main_window: 'MainWindow'):
    if not main_window.outputFolderLineEdit.text():
        common_widget_actions.create_and_show_messagebox(main_window, 'No Output Folder Selected','Please select an Output folder to save the Images/Videos before Saving/Recording!', main_window)
        return
    frame = main_window.video_processor.current_frame.copy()
    if isinstance(frame, numpy.ndarray):
        save_filename = misc_helpers.get_output_file_path(main_window.video_processor.media_path, main_window.control['OutputMediaFolder'], media_type='image')
        if save_filename:
            # frame is main_window.video_processor.current_frame, which is already RGB.
            frame = frame[..., ::-1]
            pil_image = Image.fromarray(frame) # Correct: Pass RGB frame directly to Pillow.
            pil_image.save(save_filename, 'PNG')
            common_widget_actions.create_and_show_toast_message(main_window, 'Image Saved', f'Saved Current Image to file: {save_filename}')

    else:
        common_widget_actions.create_and_show_messagebox(main_window, 'Invalid Frame', 'Cannot save the current frame!', parent_widget=main_window.saveImageButton)

def toggle_live_sound(main_window: 'MainWindow', toggle_value: bool):
    video_processor = main_window.video_processor
    was_processing = video_processor.processing

    # If the video was playing, then stop and start it again to enable the audio
    # Otherwise, just the toggle value so that the next time the play button is hit, it would automatically enable/disable the audio
    # The play button is clicked twice in the below block to simulate the above mentioned behaviour. It should be changed into a set up in the next refactor
    if was_processing:
        main_window.buttonMediaPlay.click()
        main_window.buttonMediaPlay.click()
