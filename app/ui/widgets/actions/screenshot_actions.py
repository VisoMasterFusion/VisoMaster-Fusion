"""Session screenshot capture, comparison, and save controls."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6 import QtCore, QtGui, QtWidgets

from app.ui.widgets.actions import common_actions as common_widget_actions
from app.ui.widgets.actions import graphics_view_actions
from app.ui.widgets.actions import video_control_actions

if TYPE_CHECKING:
    from app.ui.main_ui import MainWindow


def build_screenshot_path(
    output_folder: str,
    source_media_path: str,
    frame_number: int,
    now: datetime | None = None,
) -> Path:
    """Build a collision-free PNG path using the captured frame's metadata."""
    source_stem = Path(source_media_path).stem if source_media_path else "webcam"
    source_stem = source_stem or "screenshot"
    timestamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    base_name = f"{source_stem}_frame_{frame_number:06d}_{timestamp}"
    output_dir = Path(output_folder)
    candidate = output_dir / f"{base_name}.png"
    suffix = 1
    while candidate.exists():
        candidate = output_dir / f"{base_name}_{suffix:02d}.png"
        suffix += 1
    return candidate


def _has_session_screenshot(main_window: "MainWindow") -> bool:
    pixmap = getattr(main_window, "session_screenshot", None)
    return pixmap is not None and not pixmap.isNull()


def _make_separator(parent: QtWidgets.QWidget) -> QtWidgets.QWidget:
    container = QtWidgets.QWidget(parent)
    layout = QtWidgets.QHBoxLayout(container)
    layout.setContentsMargins(6, 0, 6, 0)
    layout.setSpacing(0)
    separator = QtWidgets.QFrame(container)
    separator.setFrameShape(QtWidgets.QFrame.Shape.VLine)
    separator.setFrameShadow(QtWidgets.QFrame.Shadow.Plain)
    separator.setFixedHeight(18)
    separator.setStyleSheet("color: rgba(180, 180, 180, 110);")
    layout.addWidget(separator)
    return container


def add_screenshot_controls(main_window: "MainWindow") -> None:
    """Add the three-button session screenshot workflow to the media controls."""
    if hasattr(main_window, "takeScreenshotButton"):
        return

    # In-memory only: one reusable screenshot slot for this application session.
    main_window.session_screenshot = None
    main_window.session_screenshot_source_path = ""
    main_window.session_screenshot_frame_number = 0
    main_window.session_screenshot_saved_path = ""
    main_window._base_pixmap_item = None
    main_window._screenshot_overlay_item = None

    layout = main_window.mediaControlsMainLayout
    parent = main_window.mediaControlsCenterWidget

    scan_button = getattr(main_window, "scanToolsToggleButton", None)
    insertion_index = layout.indexOf(scan_button) if scan_button is not None else -1
    if insertion_index < 0:
        insertion_index = layout.count()

    separator = _make_separator(parent)
    layout.insertWidget(insertion_index, separator)
    insertion_index += 1
    main_window.screenshotControlsSeparator = separator

    # Create hidden buttons to maintain compatibility with existing signal logic
    main_window.takeScreenshotButton = QtWidgets.QPushButton("Take SS", parent)
    main_window.takeScreenshotButton.setVisible(False)
    main_window.takeScreenshotButton.clicked.connect(lambda: take_screenshot(main_window))

    main_window.showScreenshotButton = QtWidgets.QPushButton("Show SS", parent)
    main_window.showScreenshotButton.setCheckable(True)
    main_window.showScreenshotButton.setVisible(False)
    main_window.showScreenshotButton.toggled.connect(lambda checked: show_screenshot(main_window, checked))

    main_window.saveScreenshotButton = QtWidgets.QPushButton("Save SS", parent)
    main_window.saveScreenshotButton.setVisible(False)
    main_window.saveScreenshotButton.clicked.connect(lambda: save_screenshot(main_window))

    # Create the new unified menu button
    main_window.screenshotMenuButton = QtWidgets.QToolButton(parent)
    main_window.screenshotMenuButton.setObjectName("screenshotMenuButton")
    main_window.screenshotMenuButton.setText("Screenshot")
    main_window.screenshotMenuButton.setIcon(QtGui.QIcon(":/media/media/image.png"))
    main_window.screenshotMenuButton.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
    main_window.screenshotMenuButton.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
    main_window.screenshotMenuButton.setAutoRaise(True)
    
    menu = QtWidgets.QMenu(main_window.screenshotMenuButton)

    take_action = menu.addAction("Take Screenshot")
    take_action.triggered.connect(main_window.takeScreenshotButton.click)

    main_window.showScreenshotAction = menu.addAction("Show Screenshot")
    main_window.showScreenshotAction.setCheckable(True)
    main_window.showScreenshotAction.triggered.connect(main_window.showScreenshotButton.setChecked)
    main_window.showScreenshotButton.toggled.connect(main_window.showScreenshotAction.setChecked)

    main_window.saveScreenshotAction = menu.addAction("Save Screenshot")
    main_window.saveScreenshotAction.triggered.connect(main_window.saveScreenshotButton.click)

    main_window.screenshotMenuButton.setMenu(menu)

    main_window.screenshotStatusIndicator = QtWidgets.QLabel("●", parent)
    main_window.screenshotStatusIndicator.setObjectName("screenshotStatusIndicator")
    main_window.screenshotStatusIndicator.setFixedWidth(16)
    main_window.screenshotStatusIndicator.setAlignment(
        QtCore.Qt.AlignmentFlag.AlignCenter
    )

    # Insert only the menu button and status indicator
    for widget in (
        main_window.screenshotMenuButton,
        main_window.screenshotStatusIndicator,
    ):
        layout.insertWidget(insertion_index, widget)
        insertion_index += 1

    play_button = getattr(main_window, "buttonMediaPlay", None)
    if play_button is not None:
        target_height = play_button.sizeHint().height()
        main_window.screenshotMenuButton.setMinimumHeight(target_height)
        main_window.screenshotMenuButton.setMaximumHeight(target_height)

        # Format width for the menu button
        text_width = QtGui.QFontMetrics(main_window.screenshotMenuButton.font()).horizontalAdvance(main_window.screenshotMenuButton.text()) + 40
        main_window.screenshotMenuButton.setMinimumWidth(text_width)
        main_window.screenshotMenuButton.setMaximumWidth(120)

        size_policy = main_window.screenshotMenuButton.sizePolicy()
        size_policy.setHorizontalPolicy(QtWidgets.QSizePolicy.Policy.Minimum)
        size_policy.setVerticalPolicy(QtWidgets.QSizePolicy.Policy.Fixed)
        main_window.screenshotMenuButton.setSizePolicy(size_policy)

    update_screenshot_indicator(main_window)
    if hasattr(main_window, "_queue_media_controls_balance"):
        main_window._queue_media_controls_balance()


def update_screenshot_indicator(main_window: "MainWindow") -> None:
    has_screenshot = _has_session_screenshot(main_window)
    indicator = getattr(main_window, "screenshotStatusIndicator", None)
    if indicator is not None:
        color = "#55c978" if has_screenshot else "#777777"
        indicator.setStyleSheet(f"color: {color}; font-size: 16px;")
        if has_screenshot:
            tooltip = (
                f"Session screenshot ready — frame "
                f"{main_window.session_screenshot_frame_number}."
            )
            saved_path = getattr(main_window, "session_screenshot_saved_path", "")
            if saved_path:
                tooltip += f"\nLast saved: {saved_path}"
            indicator.setToolTip(tooltip)
        else:
            indicator.setToolTip("No session screenshot captured.")

    # Updated for the new menu actions
    show_action = getattr(main_window, "showScreenshotAction", None)
    save_action = getattr(main_window, "saveScreenshotAction", None)
    if show_action is not None:
        show_action.setEnabled(has_screenshot)
    if save_action is not None:
        save_action.setEnabled(has_screenshot)


def take_screenshot(main_window: "MainWindow") -> bool:
    """Capture the live processed frame without including viewport scaling or HUD."""
    pixmap_item = graphics_view_actions.get_base_pixmap_item(main_window)
    if pixmap_item is None or pixmap_item.pixmap().isNull():
        common_widget_actions.create_and_show_toast_message(
            main_window,
            "No Frame Available",
            "Load and display a frame before taking a screenshot.",
            style_type="warning",
        )
        return False

    main_window.session_screenshot = pixmap_item.pixmap().copy()
    main_window.session_screenshot_source_path = str(
        getattr(main_window.video_processor, "media_path", "") or ""
    )
    main_window.session_screenshot_frame_number = int(
        main_window.videoSeekSlider.value()
    )
    main_window.session_screenshot_saved_path = ""

    show_button = getattr(main_window, "showScreenshotButton", None)
    if show_button is not None and show_button.isChecked():
        show_screenshot(main_window, True)
    update_screenshot_indicator(main_window)
    return True


def show_screenshot(main_window: "MainWindow", visible: bool) -> bool:
    """Toggle the saved screenshot above the continuously updated live frame."""
    if visible and not _has_session_screenshot(main_window):
        show_button = getattr(main_window, "showScreenshotButton", None)
        if show_button is not None:
            show_button.blockSignals(True)
            show_button.setChecked(False)
            show_button.blockSignals(False)
        return False

    if visible:
        display_item = graphics_view_actions.ensure_screenshot_overlay_item(main_window)
        display_item.setPixmap(main_window.session_screenshot)
        display_item.show()
        
        show_action = getattr(main_window, "showScreenshotAction", None)
        if show_action is not None:
            show_action.blockSignals(True)
            show_action.setText("Hide Screenshot")
            show_action.setChecked(True)
            show_action.blockSignals(False)
    else:
        overlay_item = graphics_view_actions.get_screenshot_overlay_item(main_window)
        if overlay_item is not None:
            overlay_item.hide()
        display_item = graphics_view_actions.get_base_pixmap_item(main_window)
        
        show_action = getattr(main_window, "showScreenshotAction", None)
        if show_action is not None:
            show_action.blockSignals(True)
            show_action.setText("Show Screenshot")
            show_action.setChecked(False)
            show_action.blockSignals(False)

    if display_item is not None:
        scene_rect = display_item.boundingRect()
        graphics_view_actions.fit_image_to_view(
            main_window, display_item, scene_rect
        )
    return True


def save_screenshot(main_window: "MainWindow") -> Path | None:
    """Save the captured frame, retaining its original source and frame metadata."""
    if not _has_session_screenshot(main_window):
        return None

    source_path = main_window.session_screenshot_source_path
    output_folder = video_control_actions.resolve_output_folder(
        main_window, source_path
    )
    if not output_folder:
        common_widget_actions.create_and_show_toast_message(
            main_window,
            "Output Folder Required",
            "Select an Output Folder in Settings before saving a screenshot.",
            style_type="warning",
        )
        return None

    try:
        output_dir = Path(output_folder)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = build_screenshot_path(
            str(output_dir),
            source_path,
            main_window.session_screenshot_frame_number,
        )
        image = main_window.session_screenshot.toImage()
        if not image.save(str(output_path), "PNG"):
            raise OSError("Qt could not encode the screenshot as PNG.")
    except (OSError, ValueError) as exc:
        common_widget_actions.create_and_show_toast_message(
            main_window,
            "Screenshot Save Failed",
            str(exc),
            style_type="error",
        )
        return None

    main_window.session_screenshot_saved_path = str(output_path)
    update_screenshot_indicator(main_window)
    common_widget_actions.create_and_show_toast_message(
        main_window,
        "Screenshot Saved",
        str(output_path),
        style_type="success",
    )
    return output_path
