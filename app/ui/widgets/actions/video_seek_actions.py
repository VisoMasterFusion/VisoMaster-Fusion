import cv2
import numpy as np
from typing import TYPE_CHECKING
from PySide6 import QtWidgets, QtCore, QtGui
import app.helpers.miscellaneous as misc_helpers
from app.helpers.miscellaneous import get_video_rotation
from app.ui.widgets.actions.video_control_actions import _get_marker_data_for_position

if TYPE_CHECKING:
    from app.ui.main_ui import MainWindow

class TimelineSeekSlider(QtWidgets.QSlider):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window: "MainWindow" = None
        
        # 1. Initialize all state markers natively
        self.markers = set()
        self.markers_sorted = []
        self.issue_markers = set()
        self.issue_markers_sorted = []
        self.dropped_markers = set()
        self.dropped_markers_sorted = []
        
        self.setTickPosition(QtWidgets.QSlider.TickPosition.TicksBelow)

        # 2. Provisional State Tracking
        self._last_provisional_state = False
        self._last_polled_position = -1
        self._cached_baseline_params = {}
        self._cached_baseline_ctrl = {}
        self._connected_widgets = set()

    def setup(self, main_window: "MainWindow"):
        """Injects the main window dependency after UI creation."""
        self.main_window = main_window
        self.valueChanged.connect(self.trigger_check)
        self.valueChanged.connect(self._auto_scroll_viewport)
        QtCore.QTimer.singleShot(500, self.bind_provisional_events)

    def setValue(self, value: int):
        """Override native setValue to catch programmatic updates even when signals are blocked."""
        super().setValue(value)
        # Manually trigger the auto-scroll check
        self._auto_scroll_viewport(value)

    def _auto_scroll_viewport(self, value: int):
        """Automatically page-scrolls the timeline if the playhead exits the viewport."""
        if not getattr(self, "main_window", None):
            return
            
        scroll_area = getattr(self.main_window, "timelineScrollArea", None)
        if not scroll_area:
            return

        # Only auto-scroll if we are actually zoomed in
        if self.width() <= scroll_area.viewport().width():
            return

        max_val = max(1, self.maximum())
        # Calculate the exact pixel coordinate of the playhead
        playhead_pixel_x = int(self.width() * (value / max_val))

        scrollbar = scroll_area.horizontalScrollBar()
        scroll_x = scrollbar.value()
        viewport_width = scroll_area.viewport().width()

        # Add a 10-pixel padding so it doesn't hug the absolute mathematical edge
        margin = 10 

        if playhead_pixel_x > scroll_x + viewport_width:
            # Playback passed the right edge. Jump the scrollbar so the playhead is at the left edge.
            scrollbar.setValue(max(0, playhead_pixel_x - margin))
            
        elif playhead_pixel_x < scroll_x:
            # User scrubbed backwards past the left edge. Jump the scrollbar so the playhead is at the right edge.
            scrollbar.setValue(max(0, playhead_pixel_x - viewport_width + margin))

    # --- Marker Management Methods ---
    def add_marker_and_paint(self, value=None):
        if value is None or isinstance(value, bool):
            value = self.value()
        if self.minimum() <= value <= self.maximum() and value not in self.markers:
            self.markers.add(value)
            if value not in self.markers_sorted:
                self.markers_sorted.append(value)
                self.markers_sorted.sort()
            self._last_polled_position = -1
            self.update()
            self.trigger_check()

    def remove_marker_and_paint(self, value=None):
        if value is None or isinstance(value, bool):
            value = self.value()
        if value in self.markers:
            self.markers.remove(value)
            if value in self.markers_sorted:
                self.markers_sorted.remove(value)
            self._last_polled_position = -1
            self.update()
            self.trigger_check()

    def _add_sorted_marker(self, marker_set: set[int], marker_list: list[int], value: int) -> bool:
        if value not in marker_set:
            marker_set.add(value)
            marker_list.append(value)
            marker_list.sort()
            return True
        return False

    def _remove_sorted_marker(self, marker_set: set[int], marker_list: list[int], value: int) -> bool:
        if value in marker_set:
            marker_set.remove(value)
            if value in marker_list:
                marker_list.remove(value)
            return True
        return False

    def add_issue_marker_and_paint(self, value=None):
        if value is None or isinstance(value, bool):
            value = self.value()
        if self.minimum() <= value <= self.maximum() and self._add_sorted_marker(self.issue_markers, self.issue_markers_sorted, value):
            self.update()

    def remove_issue_marker_and_paint(self, value=None):
        if value is None or isinstance(value, bool):
            value = self.value()
        if self._remove_sorted_marker(self.issue_markers, self.issue_markers_sorted, value):
            self.update()

    def add_dropped_marker_and_paint(self, value=None):
        if value is None or isinstance(value, bool):
            value = self.value()
        if self.minimum() <= value <= self.maximum() and self._add_sorted_marker(self.dropped_markers, self.dropped_markers_sorted, value):
            self.update()

    def remove_dropped_marker_and_paint(self, value=None):
        if value is None or isinstance(value, bool):
            value = self.value()
        if self._remove_sorted_marker(self.dropped_markers, self.dropped_markers_sorted, value):
            self.update()

    # --- Provisional State Logic ---
    def is_state_provisional(self) -> bool:
        if not self.main_window: return False
        current_position = self.value()

        if not self.markers_sorted or current_position < self.markers_sorted[0]:
            return True

        if current_position != self._last_polled_position:
            self._last_polled_position = current_position
            marker_data = _get_marker_data_for_position(self.main_window, current_position)
            
            if not marker_data:
                return True
                
            self._cached_baseline_params = marker_data.get("parameters", {})
            self._cached_baseline_ctrl = marker_data.get("control", {})

        baseline_params = self._cached_baseline_params
        baseline_ctrl = self._cached_baseline_ctrl

        if self.main_window.parameters != baseline_params:
            return True

        protected_keys = {
            "TrackMarkersToggle", "OutputMediaFolder", "OutputToTargetLocationToggle",
            "PreserveOutputDirectoryStructureToggle", "ClusterOutputBySourceToggle",
        }

        curr_ctrl = self.main_window.control
        for k, v in curr_ctrl.items():
            if k in protected_keys: continue
            if k not in baseline_ctrl or baseline_ctrl[k] != v: return True
        for k in baseline_ctrl:
            if k in protected_keys: continue
            if k not in curr_ctrl: return True

        return False

    def check_provisional_changes(self):
        current_state = self.is_state_provisional()
        if current_state != self._last_provisional_state:
            self._last_provisional_state = current_state
            self.update()

    def trigger_check(self, *args, **kwargs):
        QtCore.QTimer.singleShot(150, self.check_provisional_changes)

    def bind_provisional_events(self):
        if hasattr(self.main_window, "parameter_widgets"):
            for name, widget in self.main_window.parameter_widgets.items():
                if name in self._connected_widgets:
                    continue
                try:
                    if isinstance(widget, QtWidgets.QSlider):
                        widget.valueChanged.connect(self.trigger_check)
                    elif isinstance(widget, QtWidgets.QComboBox):
                        widget.currentIndexChanged.connect(self.trigger_check)
                    elif isinstance(widget, QtWidgets.QAbstractButton):
                        widget.toggled.connect(self.trigger_check)
                    elif isinstance(widget, QtWidgets.QLineEdit):
                        widget.textChanged.connect(self.trigger_check)
                    self._connected_widgets.add(name)
                except Exception:
                    pass

    # --- Paint Event ---
    def paintEvent(self, event: QtGui.QPaintEvent):
        """Native paint event overriding QSlider's default."""
        if not self.main_window:
            return super().paintEvent(event)
            
        if self.maximum() == self.minimum() or self.main_window.video_processor.file_type == "image":
            return super().paintEvent(event)
        
        painter = QtWidgets.QStylePainter(self)
        opt = QtWidgets.QStyleOptionSlider()
        self.initStyleOption(opt)
        style = self.style()

        groove_rect = style.subControlRect(
            QtWidgets.QStyle.ComplexControl.CC_Slider, opt, QtWidgets.QStyle.SubControl.SC_SliderGroove
        )
        groove_y = (groove_rect.top() + groove_rect.bottom()) // 2
        groove_start = groove_rect.left()
        groove_end = groove_rect.right()
        groove_width = groove_end - groove_start

        def marker_x_for_value(value: int) -> float:
            marker_normalized_value = (value - self.minimum()) / max(1, (self.maximum() - self.minimum()))
            return groove_start + marker_normalized_value * groove_width

        normalized_value = (self.value() - self.minimum()) / max(1, (self.maximum() - self.minimum()))
        handle_center_x = groove_start + normalized_value * groove_width

        handle_width = 5
        handle_height = groove_rect.height()
        handle_left_x = int(handle_center_x - (handle_width // 2))
        handle_top_y = groove_y - (handle_height // 2)

        handle_rect = QtCore.QRect(handle_left_x, handle_top_y, handle_width, handle_height)

        has_provisional_changes = self._last_provisional_state

        # 1. Base Alternating Segments
        painter.setPen(QtGui.QPen(QtGui.QColor("gray"), 3))
        if not self.markers_sorted:
            painter.drawLine(groove_start, groove_y, groove_end, groove_y)
        else:
            first_marker_x = marker_x_for_value(self.markers_sorted[0])
            painter.drawLine(groove_start, groove_y, int(first_marker_x), groove_y)

            color_a = QtGui.QColor("#7e57c2")
            color_b = QtGui.QColor("#42a5f5")

            for i in range(len(self.markers_sorted)):
                start_val = self.markers_sorted[i]
                end_val = self.markers_sorted[i + 1] if i + 1 < len(self.markers_sorted) else self.maximum()
                start_x = marker_x_for_value(start_val)
                end_x = marker_x_for_value(end_val)

                current_color = color_a if i % 2 == 0 else color_b
                painter.setPen(QtGui.QPen(current_color, 3))
                painter.drawLine(int(start_x), groove_y, int(end_x), groove_y)
                painter.drawLine(int(start_x), groove_rect.top(), int(start_x), groove_rect.bottom())

        # 2. Provisional Changes Overlay
        if has_provisional_changes:
            provisional_start_val = self.value()
            provisional_end_val = self.maximum()

            for m in self.markers_sorted:
                if m > provisional_start_val:
                    provisional_end_val = m
                    break

            start_x = marker_x_for_value(provisional_start_val)
            end_x = marker_x_for_value(provisional_end_val)
            painter.setPen(QtGui.QPen(QtGui.QColor("#e5c07b"), 3))
            painter.drawLine(int(start_x), groove_y, int(end_x), groove_y)

        # 3. Issue markers
        if self.issue_markers:
            issue_pen = QtGui.QPen(QtGui.QColor("#ff9800"), 3)
            issue_pen.setCapStyle(QtCore.Qt.PenCapStyle.SquareCap)
            painter.setPen(issue_pen)
            issue_top = groove_y - 2
            issue_bottom = groove_y + 2
            for value in self.issue_markers_sorted:
                if value in self.dropped_markers: continue
                marker_x = marker_x_for_value(value)
                painter.drawLine(int(marker_x), issue_top, int(marker_x), issue_bottom)

        # 4. Dropped markers
        if self.dropped_markers:
            painter.setPen(QtGui.QPen(QtGui.QColor("#e8483c"), 3))
            for value in self.dropped_markers_sorted:
                marker_x = marker_x_for_value(value)
                painter.drawLine(int(marker_x), groove_rect.top(), int(marker_x), groove_rect.bottom())

        # 5. Playhead Handle
        handle_color = QtGui.QColor("#e5c07b") if has_provisional_changes else QtGui.QColor("white")
        painter.setPen(QtGui.QPen(handle_color, 1))
        painter.setBrush(QtGui.QBrush(handle_color))
        painter.drawRect(handle_rect)

        # 6. Job Start/End Brackets
        painter.setFont(QtGui.QFont("Arial", 16, QtGui.QFont.Bold))
        font_metrics = painter.fontMetrics()
        
        # Calculate exactly how much space the bracket needs
        bracket_width = font_metrics.horizontalAdvance("[") + 4
        bracket_height = font_metrics.height()

        # Center the bounding box vertically around the groove
        bracket_top_y = int(groove_y - (bracket_height / 2))

        for start_frame, end_frame in self.main_window.job_marker_pairs:
            if start_frame is not None:
                start_x = marker_x_for_value(int(start_frame))
                # Create a strict bounding box to draw inside
                rect = QtCore.QRect(int(start_x - 6), bracket_top_y, bracket_width, bracket_height)
                painter.setPen(QtGui.QPen(QtGui.QColor("#4CAF50"), 1))
                painter.drawText(rect, QtCore.Qt.AlignmentFlag.AlignCenter, "[")

            if end_frame is not None:
                end_x = marker_x_for_value(int(end_frame))
                # Create a strict bounding box to draw inside
                rect = QtCore.QRect(int(end_x - 6), bracket_top_y, bracket_width, bracket_height)
                painter.setPen(QtGui.QPen(QtGui.QColor("#e8483c"), 1))
                painter.drawText(rect, QtCore.Qt.AlignmentFlag.AlignCenter, "]")

class ThumbnailExtractionWorker(QtCore.QThread):
    # Sends (frame_number, QImage) back to the main thread
    thumbnail_ready = QtCore.Signal(int, QtGui.QImage)
    
    def __init__(self, media_path, frame_intervals, target_height, parent=None):
        super().__init__(parent)
        self.media_path = media_path
        self.frame_intervals = frame_intervals
        self.target_height = target_height
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True

    def run(self):
        if not self.media_path or not self.frame_intervals:
            return

        rotation_angle = get_video_rotation(self.media_path)
        cap = cv2.VideoCapture(self.media_path)
        
        if hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO"):
            cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)

        if not cap.isOpened():
            return

        for frame_num in self.frame_intervals:
            if self._is_cancelled:
                break
                
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = misc_helpers.read_frame(cap, rotation_angle)
            
            if ret and frame is not None and not self._is_cancelled:
                # Convert BGR to RGB
                frame_rgb = frame[..., ::-1]
                
                # Calculate new width maintaining aspect ratio
                h, w, _ = frame_rgb.shape
                aspect_ratio = w / max(1, h)
                target_width = int(self.target_height * aspect_ratio)
                
                # Resize using OpenCV (faster in background thread than Qt)
                resized = cv2.resize(frame_rgb, (target_width, self.target_height), interpolation=cv2.INTER_AREA)
                
                # Convert to Thread-Safe QImage
                bytes_per_line = 3 * target_width
                q_img = QtGui.QImage(
                    resized.data, target_width, self.target_height, bytes_per_line, QtGui.QImage.Format_RGB888
                ).copy() # Crucial: .copy() detaches memory from the numpy array
                
                self.thumbnail_ready.emit(frame_num, q_img)
                
        cap.release()

class ThumbnailTrackWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(40)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        
        self.main_window = None
        self.worker = None
        self.thumbnail_cache = {} 
        self.expected_intervals = []
        self.thumbnail_width = 71 
        
        self.resize_timer = QtCore.QTimer(self)
        self.resize_timer.setSingleShot(True)
        self.resize_timer.setInterval(400)
        self.resize_timer.timeout.connect(self.request_thumbnails)

    def setup(self, main_window):
        self.main_window = main_window
        # Connect the scrollbar to our debounce timer so scrolling loads new thumbnails
        scroll_area = getattr(main_window, "timelineScrollArea", None)
        if scroll_area:
            scroll_area.horizontalScrollBar().valueChanged.connect(self.on_scroll)

    def on_scroll(self, value):
        if self.main_window:
            self.resize_timer.start()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Restart the timer every time the widget resizes (zooms)
        if self.main_window:
            self.resize_timer.start()

    def request_thumbnails(self):
        if not self.main_window or not self.main_window.video_processor:
            return
            
        vp = self.main_window.video_processor
        if vp.file_type != "video" or not vp.media_path or vp.max_frame_number <= 0:
            self.thumbnail_cache.clear()
            self.update()
            return

        import math
        self.thumbnail_width = int(self.height() * 1.777)
        total_thumbnails = math.ceil(self.width() / max(1, self.thumbnail_width))
        
        # 1. Calculate ALL global intervals for the current zoom level
        # We need this to know where to draw placeholders across the entire widget
        all_intervals = []
        for i in range(total_thumbnails):
            x_pos = i * self.thumbnail_width
            frame_num = int((x_pos / max(1, self.width())) * vp.max_frame_number)
            all_intervals.append(min(frame_num, vp.max_frame_number))
            
        self.expected_intervals = all_intervals
        
        # 2. Determine which thumbnails are currently VISIBLE in the scroll area
        scroll_area = getattr(self.main_window, "timelineScrollArea", None)
        if scroll_area:
            scroll_x = scroll_area.horizontalScrollBar().value()
            viewport_w = scroll_area.viewport().width()
            
            # Calculate indices based on scroll position (Add a 1-thumbnail buffer on each side for smooth scrolling)
            start_idx = max(0, int(scroll_x // self.thumbnail_width) - 1)
            end_idx = min(total_thumbnails, int((scroll_x + viewport_w) // self.thumbnail_width) + 2)
            
            visible_intervals = all_intervals[start_idx:end_idx]
        else:
            visible_intervals = all_intervals

        # 3. Clean up cache
        # We ONLY remove thumbnails from old zoom levels to save RAM.
        # We keep the scrolled ones cached so scrubbing back and forth is instantly responsive
        keys_to_remove = [k for k in self.thumbnail_cache.keys() if k not in all_intervals]
        for k in keys_to_remove:
            del self.thumbnail_cache[k]

        # 4. Only request extraction for VISIBLE thumbnails we don't have yet
        missing_visible = [f for f in visible_intervals if f not in self.thumbnail_cache]

        if missing_visible:
            # Cancel any active worker
            if self.worker and self.worker.isRunning():
                self.worker.cancel()
                self.worker.wait()
                
            self.worker = ThumbnailExtractionWorker(vp.media_path, missing_visible, self.height(), self)
            self.worker.thumbnail_ready.connect(self.on_thumbnail_ready)
            self.worker.start()
            
        self.update() # Force repaint of visible area

    @QtCore.Slot(int, QtGui.QImage)
    def on_thumbnail_ready(self, frame_num, q_image):
        # Convert the thread-safe QImage back to a GUI-friendly QPixmap
        self.thumbnail_cache[frame_num] = QtGui.QPixmap.fromImage(q_image)
        # We received new data, request a repaint
        self.update()

    def mousePressEvent(self, event: QtGui.QMouseEvent):
        """Handle clicks on the thumbnail track."""
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            # Left-click: Jump to this position and start scrubbing
            slider = getattr(self.main_window, "videoSeekSlider", None)
            if slider:
                slider.setSliderDown(True)  # Tell the app we are "dragging"
            self._seek_to_mouse_pos(event.pos().x())
            
        elif event.button() == QtCore.Qt.MouseButton.RightButton:
            # Right-click: Center the viewport on the current playhead
            if not self.main_window:
                return

            slider = getattr(self.main_window, "videoSeekSlider", None)
            scroll_area = getattr(self.main_window, "timelineScrollArea", None)
            
            if slider and scroll_area:
                current_val = slider.value()
                max_val = max(1, slider.maximum())
                
                # Calculate the exact pixel coordinate of the playhead
                relative_pos = current_val / max_val
                playhead_pixel_x = self.width() * relative_pos
                
                # Calculate where the scrollbar needs to be to center that pixel
                viewport_width = scroll_area.viewport().width()
                target_scroll_pos = int(playhead_pixel_x - (viewport_width / 2))
                
                # Snap the scrollbar
                scroll_area.horizontalScrollBar().setValue(max(0, target_scroll_pos))

    def mouseMoveEvent(self, event: QtGui.QMouseEvent):
        """Allow the user to click and drag across the thumbnails to scrub."""
        if event.buttons() & QtCore.Qt.MouseButton.LeftButton:
            self._seek_to_mouse_pos(event.pos().x())

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent):
        """Trigger final processing when the user lets go of the mouse."""
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            slider = getattr(self.main_window, "videoSeekSlider", None)
            if slider:
                slider.setSliderDown(False)  # We stopped dragging
                # Emit the released signal so the final AI processing runs
                slider.sliderReleased.emit()

    def _seek_to_mouse_pos(self, x_pos: int):
        """Convert the mouse X coordinate into a video frame and update the slider."""
        if not self.main_window or not self.main_window.video_processor:
            return
            
        vp = self.main_window.video_processor
        if vp.max_frame_number <= 0:
            return
            
        # Clamp the X position to the widget boundaries so dragging outside doesn't crash it
        x_pos = max(0, min(x_pos, self.width()))
        
        # Use the exact same math we used to draw the thumbnails to find the frame
        frame_num = int((x_pos / max(1, self.width())) * vp.max_frame_number)
        frame_num = min(frame_num, vp.max_frame_number)
        
        # Update the slider! This instantly triggers on_change_video_seek_slider
        # which safely handles OpenCV extraction, UI updates, and previewing.
        slider = getattr(self.main_window, "videoSeekSlider", None)
        if slider:
            slider.setValue(frame_num)

    def paintEvent(self, event: QtGui.QPaintEvent):
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), QtGui.QColor("#1e1e1e"))
        
        if not self.expected_intervals:
            return

        # Safely get the video's FPS to calculate timestamps
        vp = self.main_window.video_processor if self.main_window else None
        fps = float(getattr(vp, "fps", 0.0) or 0.0) if vp else 0.0

        # Setup a small, legible font for the timestamps
        font = painter.font()
        font.setPointSize(8)
        font.setBold(True)
        painter.setFont(font)

        # Draw the thumbnails along the track
        for i, frame_num in enumerate(self.expected_intervals):
            x_pos = i * self.thumbnail_width
            
            if frame_num in self.thumbnail_cache:
                pixmap = self.thumbnail_cache[frame_num]
                # In case the actual video aspect ratio is wider/narrower, we center it
                actual_width = pixmap.width()
                offset = max(0, (self.thumbnail_width - actual_width) // 2)
                painter.drawPixmap(x_pos + offset, 0, pixmap)
                
                # --- TIMESTAMP OVERLAY ---
                if fps > 0.0:
                    total_seconds = frame_num / fps
                    hours = int(total_seconds // 3600)
                    minutes = int((total_seconds % 3600) // 60)
                    seconds = int(total_seconds % 60)
                    
                    if hours > 0:
                        time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                    else:
                        time_str = f"{minutes:02d}:{seconds:02d}"

                    # Create a bounding box at the bottom of the thumbnail space (14px high)
                    text_rect = QtCore.QRect(x_pos, self.height() - 14, self.thumbnail_width, 14)
                    
                    # Draw a semi-transparent black background strip so white text always pops
                    painter.fillRect(text_rect, QtGui.QColor(0, 0, 0, 160))
                    
                    # Draw the white text centered in the strip
                    painter.setPen(QtGui.QColor("white"))
                    painter.drawText(text_rect, QtCore.Qt.AlignmentFlag.AlignCenter, time_str)
                    
            else:
                # Draw a placeholder while loading
                painter.setPen(QtGui.QColor("#444444"))
                painter.drawRect(x_pos, 0, self.thumbnail_width - 1, self.height() - 1)


class CompositeTimelineWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(2)
        
        self.thumbnail_track = ThumbnailTrackWidget(self)
        self.slider = TimelineSeekSlider(self)
        
        self.layout.addWidget(self.thumbnail_track)
        self.layout.addWidget(self.slider)

    def setup(self, main_window):
        """Passes the main window dependency down to both children."""
        self.slider.setup(main_window)
        self.thumbnail_track.setup(main_window)