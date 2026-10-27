from PySide6 import QtWidgets, QtGui, QtCore
from typing import TYPE_CHECKING
from app.ui.widgets.actions import video_control_actions

if TYPE_CHECKING:
    from app.ui.main_ui import MainWindow


def _item_belongs_to_scene(item, scene) -> bool:
    try:
        return item is not None and item.scene() is scene
    except RuntimeError:
        return False


def get_base_pixmap_item(main_window: "MainWindow"):
    """Return the live frame item, excluding the screenshot comparison overlay."""
    scene = main_window.graphicsViewFrame.scene()
    pixmap_item = getattr(main_window, "_base_pixmap_item", None)
    if _item_belongs_to_scene(pixmap_item, scene):
        return pixmap_item

    overlay_item = getattr(main_window, "_screenshot_overlay_item", None)
    for item in scene.items():
        if isinstance(item, QtWidgets.QGraphicsPixmapItem) and item is not overlay_item:
            main_window._base_pixmap_item = item
            return item
    return None


def get_screenshot_overlay_item(main_window: "MainWindow"):
    scene = main_window.graphicsViewFrame.scene()
    overlay_item = getattr(main_window, "_screenshot_overlay_item", None)
    return overlay_item if _item_belongs_to_scene(overlay_item, scene) else None


def ensure_screenshot_overlay_item(main_window: "MainWindow"):
    """Create or reuse the top-level session screenshot item."""
    overlay_item = get_screenshot_overlay_item(main_window)
    if overlay_item is None:
        overlay_item = QtWidgets.QGraphicsPixmapItem()
        overlay_item.setTransformationMode(
            QtCore.Qt.TransformationMode.SmoothTransformation
        )
        overlay_item.setZValue(10000.0)
        main_window.graphicsViewFrame.scene().addItem(overlay_item)
        main_window._screenshot_overlay_item = overlay_item
    return overlay_item


def get_display_pixmap_item(main_window: "MainWindow"):
    overlay_item = get_screenshot_overlay_item(main_window)
    if overlay_item is not None and overlay_item.isVisible():
        return overlay_item
    return get_base_pixmap_item(main_window)


# @misc_helpers.benchmark  (Keep this decorator if you have it)
def update_graphics_view(
    main_window: "MainWindow",
    pixmap: QtGui.QPixmap,
    current_frame_number: int,
    reset_fit: bool = False,
    size_mode: str = "preserve_previous_pixmap_size",
):
    # Update the video seek slider and line edit safely to avoid recursive signal firing
    if main_window.videoSeekSlider.value() != current_frame_number:
        main_window.videoSeekSlider.blockSignals(True)
        main_window.videoSeekSlider.setValue(current_frame_number)
        main_window.videoSeekSlider.blockSignals(False)

    current_text = main_window.videoSeekLineEdit.text()
    if current_text != str(current_frame_number):
        main_window.videoSeekLineEdit.setText(str(current_frame_number))
    video_control_actions.update_video_time_line_edit(main_window, current_frame_number)

    scene = main_window.graphicsViewFrame.scene()
    pixmap_item = get_base_pixmap_item(main_window)

    # Resize the pixmap if necessary (e.g., face compare or mask compare mode)
    if pixmap_item and size_mode == "preserve_previous_pixmap_size":
        bounding_rect = pixmap_item.boundingRect()
        b_width = int(bounding_rect.width())
        b_height = int(bounding_rect.height())

        if b_width > pixmap.width() and b_height > pixmap.height():
            pixmap = pixmap.scaled(
                b_width,
                b_height,
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation,
            )

    if pixmap_item:
        pixmap_item.setPixmap(pixmap)
    else:
        pixmap_item = QtWidgets.QGraphicsPixmapItem(pixmap)
        pixmap_item.setTransformationMode(
            QtCore.Qt.TransformationMode.SmoothTransformation
        )
        scene.addItem(pixmap_item)

    main_window._base_pixmap_item = pixmap_item
    take_button = getattr(main_window, "takeScreenshotButton", None)
    if take_button is not None:
        take_button.setEnabled(not pixmap.isNull())

    # Keep a visible session screenshot on top while the live frame updates below it.
    show_button = getattr(main_window, "showScreenshotButton", None)
    session_screenshot = getattr(main_window, "session_screenshot", None)
    if (
        show_button is not None
        and show_button.isChecked()
        and session_screenshot is not None
        and not session_screenshot.isNull()
    ):
        overlay_item = ensure_screenshot_overlay_item(main_window)
        overlay_item.setPixmap(session_screenshot)
        overlay_item.show()

    display_item = get_display_pixmap_item(main_window) or pixmap_item
    scene_rect = display_item.boundingRect()
    main_window.graphicsViewFrame.setSceneRect(scene_rect)

    if reset_fit:
        fit_image_to_view(main_window, display_item, scene_rect)


def zoom_andfit_image_to_view_onchange(main_window: "MainWindow", new_transform):
    """Restore the previous transform (zoom and pan state) and update the view."""
    main_window.graphicsViewFrame.setTransform(new_transform, combine=False)


def fit_image_to_view(
    main_window: "MainWindow", pixmap_item: QtWidgets.QGraphicsPixmapItem, scene_rect
):
    """Reset the view and fit the image to the view, keeping the aspect ratio."""
    graphicsViewFrame = main_window.graphicsViewFrame
    graphicsViewFrame.resetTransform()
    graphicsViewFrame.setSceneRect(scene_rect)
    graphicsViewFrame.fitInView(pixmap_item, QtCore.Qt.AspectRatioMode.KeepAspectRatio)
