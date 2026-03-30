from PySide6 import QtWidgets, QtGui, QtCore
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.ui.main_ui import MainWindow


# @misc_helpers.benchmark  (Keep this decorator if you have it)
def update_graphics_view(
    main_window: "MainWindow",
    pixmap: QtGui.QPixmap,
    current_frame_number: int,
    reset_fit: bool = False,
):
    # print('(update_graphics_view) current_frame_number', current_frame_number)

    # Update the video seek slider and line edit safely to avoid recursive signal firing
    if main_window.videoSeekSlider.value() != current_frame_number:
        main_window.videoSeekSlider.blockSignals(True)
        main_window.videoSeekSlider.setValue(current_frame_number)
        main_window.videoSeekSlider.blockSignals(False)

    current_text = main_window.videoSeekLineEdit.text()
    if current_text != str(current_frame_number):
        main_window.videoSeekLineEdit.setText(str(current_frame_number))

    # Safely find the QGraphicsPixmapItem in the scene, ignoring other overlays (rectangles, text, etc.)
    scene = main_window.graphicsViewFrame.scene()
    pixmap_item = None
    for item in scene.items():
        if isinstance(item, QtWidgets.QGraphicsPixmapItem):
            pixmap_item = item
            break

    # Resize the pixmap if necessary (e.g., face compare or mask compare mode)
    if pixmap_item:
        bounding_rect = pixmap_item.boundingRect()
        b_width = int(bounding_rect.width())  # Explicit cast to int for PySide6 safety
        b_height = int(
            bounding_rect.height()
        )  # Explicit cast to int for PySide6 safety

        # If the old pixmap bounding rect is larger than the new pixmap, scale the new one
        if b_width > pixmap.width() and b_height > pixmap.height():
            pixmap = pixmap.scaled(
                b_width,
                b_height,
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation,  # Added smooth filter
            )

    # Update or create pixmap item
    if pixmap_item:
        pixmap_item.setPixmap(pixmap)  # Update the pixmap of the existing item
    else:
        pixmap_item = QtWidgets.QGraphicsPixmapItem(pixmap)
        pixmap_item.setTransformationMode(
            QtCore.Qt.TransformationMode.SmoothTransformation
        )
        scene.addItem(pixmap_item)

    # Set the scene rectangle to the bounding rectangle of the pixmap
    scene_rect = pixmap_item.boundingRect()
    main_window.graphicsViewFrame.setSceneRect(scene_rect)

    # Reset the view or restore the previous transform
    if reset_fit:
        fit_image_to_view(main_window, pixmap_item, scene_rect)


def zoom_andfit_image_to_view_onchange(main_window: "MainWindow", new_transform):
    """Restore the previous transform (zoom and pan state) and update the view."""
    main_window.graphicsViewFrame.setTransform(new_transform, combine=False)


def fit_image_to_view(
    main_window: "MainWindow", pixmap_item: QtWidgets.QGraphicsPixmapItem, scene_rect
):
    """Reset the view and fit the image to the view, keeping the aspect ratio."""
    graphicsViewFrame = main_window.graphicsViewFrame
    # Reset the transform and set the scene rectangle
    graphicsViewFrame.resetTransform()
    graphicsViewFrame.setSceneRect(scene_rect)
    # Fit the image to the view, keeping the aspect ratio
    graphicsViewFrame.fitInView(pixmap_item, QtCore.Qt.AspectRatioMode.KeepAspectRatio)
