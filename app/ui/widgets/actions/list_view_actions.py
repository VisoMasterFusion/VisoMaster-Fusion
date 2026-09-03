from collections import deque
from functools import partial
from typing import TYPE_CHECKING, Dict, Type
from pathlib import Path
import sys
import os
import subprocess
import time

from PySide6 import QtWidgets, QtGui, QtCore

from app.helpers.app_metadata import AppDisplayMetadata, get_app_display_metadata
from app.ui.widgets.actions import common_actions as common_widget_actions
from app.ui.widgets.actions import card_actions
from app.ui.widgets.actions import filter_actions
from app.ui.widgets import widget_components
import app.helpers.miscellaneous as misc_helpers
from app.ui.widgets import ui_workers
from app.ui.widgets import sortable_widgets

if TYPE_CHECKING:
    from app.ui.main_ui import MainWindow

_WORKER_STOP_TIMEOUT_MS = 1000
_TARGET_BUTTON_SIZE = (96, 96)
_SMALL_FACE_BUTTON_SIZE = (70, 70)
_LARGE_FACE_BUTTON_SIZE = (96, 96)
_FACE_BUTTON_SIZE = _SMALL_FACE_BUTTON_SIZE
_EMBED_BUTTON_SIZE = (120, 25)
_EMBED_LIST_HEIGHT = 140
_TARGET_MEDIA_BATCH_SIZE = 24
_TARGET_MEDIA_BATCH_INTERVAL_MS = 1


def _get_target_media_batch_size(pending_count: int) -> int:
    # Adaptive batch sizing: small queue keeps UI very responsive, large queue
    # increases throughput to reduce total drain time.
    if pending_count >= 1500:
        return 64
    if pending_count >= 900:
        return 48
    if pending_count >= 400:
        return 36
    return _TARGET_MEDIA_BATCH_SIZE


def _ensure_target_media_batch_timer(main_window: "MainWindow") -> QtCore.QTimer:
    timer = getattr(main_window, "_target_media_batch_timer", None)
    if timer is None:
        timer = QtCore.QTimer(main_window)
        timer.setSingleShot(True)
        timer.timeout.connect(partial(_flush_target_media_thumbnail_batch, main_window))
        main_window._target_media_batch_timer = timer
    return timer


def _flush_target_media_thumbnail_batch(main_window: "MainWindow") -> None:
    pending_items = getattr(main_window, "_pending_target_media_thumbnails", None)
    if not pending_items:
        return

    list_widget = main_window.targetVideosList
    pending_before = len(pending_items)
    adaptive_batch_size = _get_target_media_batch_size(pending_before)
    batch_size = min(adaptive_batch_size, pending_before)
    list_widget.setUpdatesEnabled(False)
    try:
        for _ in range(batch_size):
            media_path, q_image, file_type, media_id, metadata = pending_items.popleft()
            add_media_thumbnail_button(
                main_window,
                widget_components.TargetMediaCardButton,
                list_widget,
                main_window.target_videos,
                q_image,
                media_path=media_path,
                file_type=file_type,
                media_id=media_id,
                metadata=metadata,
            )
    finally:
        list_widget.setUpdatesEnabled(True)
        list_widget.viewport().update()
        _advance_target_media_progress(main_window, batch_size)

    if pending_items:
        _ensure_target_media_batch_timer(main_window).start(
            _TARGET_MEDIA_BATCH_INTERVAL_MS
        )


def _queue_target_media_thumbnail(
    main_window: "MainWindow", media_path, q_image, file_type, media_id, metadata=None
) -> None:
    pending_items = getattr(main_window, "_pending_target_media_thumbnails", None)
    if pending_items is None:
        pending_items = deque()
        main_window._pending_target_media_thumbnails = pending_items

    pending_items.append((media_path, q_image, file_type, media_id, metadata))
    _add_target_media_progress_total(main_window, 1)
    timer = _ensure_target_media_batch_timer(main_window)
    if not timer.isActive():
        timer.start(_TARGET_MEDIA_BATCH_INTERVAL_MS)


def _add_target_media_progress_total(main_window: "MainWindow", n: int) -> None:
    """Grow the loading progress bar's total.  GUI thread only."""
    bar = getattr(main_window, "targetVideosListProgressBar", None)
    if bar is None or n <= 0:
        return
    bar.setMaximum(bar.maximum() + n)
    bar.setVisible(True)


def _advance_target_media_progress(main_window: "MainWindow", n: int) -> None:
    """Advance the loading progress bar, hiding it once the queue has drained."""
    bar = getattr(main_window, "targetVideosListProgressBar", None)
    if bar is None:
        return

    # Qt's QProgressBar.value() returns -1 when in an indeterminate state
    # (which happens right after bar.reset() is called).
    # We must clamp current_value to 0, otherwise the first advancement eats +1,
    # causing the permanent off-by-one hang (e.g., 1/2 or 20/21).
    current_value = max(0, bar.value())

    bar.setValue(min(current_value + n, bar.maximum()))

    if bar.value() >= bar.maximum():
        _reset_target_media_progress(main_window)


def _reset_target_media_progress(main_window: "MainWindow") -> None:
    """Hide and zero the loading progress bar (also used when loading is cancelled)."""
    bar = getattr(main_window, "targetVideosListProgressBar", None)
    if bar is None:
        return
    bar.setVisible(False)
    bar.setMaximum(0)
    bar.reset()


def _has_pending_target_media_thumbnail_work(main_window: "MainWindow") -> bool:
    timer = getattr(main_window, "_target_media_batch_timer", None)
    pending_items = getattr(main_window, "_pending_target_media_thumbnails", None)
    return bool(pending_items) or bool(timer and timer.isActive())


# INPUT FACES BATCHING SYSTEM
def _ensure_input_face_batch_timer(main_window: "MainWindow") -> QtCore.QTimer:
    """
    Ensures a QTimer exists for batching input face thumbnails.
    This timer regulates the flow of UI updates to prevent freezing.
    """
    timer = getattr(main_window, "_input_face_batch_timer", None)
    if timer is None:
        timer = QtCore.QTimer(main_window)
        timer.setSingleShot(True)
        timer.timeout.connect(partial(_flush_input_face_thumbnail_batch, main_window))
        main_window._input_face_batch_timer = timer
    return timer


def _flush_input_face_thumbnail_batch(main_window: "MainWindow") -> None:
    """
    Processes a batch of pending input face thumbnails.
    Disables UI updates during processing to eliminate flickering and reduce main thread load.
    """
    pending_items = getattr(main_window, "_pending_input_face_thumbnails", None)
    if not pending_items:
        return

    list_widget = main_window.inputFacesList
    pending_before = len(pending_items)

    # Block rendering to prevent UI flickering (Critical for performance)
    list_widget.setUpdatesEnabled(False)
    try:
        # Conservative batch size (24) to avoid saturating the main thread
        # with heavy PyTorch Tensors and K/V dictionary injections.
        batch_size = min(24, pending_before)
        for _ in range(batch_size):
            media_path, cropped_face, embedding_store, q_image, face_id = (
                pending_items.popleft()
            )
            add_media_thumbnail_button(
                main_window,
                widget_components.InputFaceCardButton,
                list_widget,
                main_window.input_faces,
                q_image,
                media_path=media_path,
                cropped_face=cropped_face,
                embedding_store=embedding_store,
                face_id=face_id,
            )
    finally:
        # Re-enable and force a single paint event
        list_widget.setUpdatesEnabled(True)
        list_widget.viewport().update()

    # If items remain in the queue, restart the timer for the next batch
    if pending_items:
        _ensure_input_face_batch_timer(main_window).start(
            _TARGET_MEDIA_BATCH_INTERVAL_MS
        )


def _queue_input_face_thumbnail(
    main_window: "MainWindow",
    media_path,
    cropped_face,
    embedding_store,
    q_image,
    face_id,
) -> None:
    """
    Appends a loaded input face to the queue and starts the batching timer if not already running.
    """
    pending_items = getattr(main_window, "_pending_input_face_thumbnails", None)
    if pending_items is None:
        pending_items = deque()
        main_window._pending_input_face_thumbnails = pending_items

    pending_items.append((media_path, cropped_face, embedding_store, q_image, face_id))
    timer = _ensure_input_face_batch_timer(main_window)
    if not timer.isActive():
        timer.start(_TARGET_MEDIA_BATCH_INTERVAL_MS)


# Functions to add Buttons with thumbnail for selecting videos/images and faces
@QtCore.Slot(str, QtGui.QImage, str, str, object)
def add_media_thumbnail_to_target_videos_list(
    main_window: "MainWindow", media_path, q_image, file_type, media_id, metadata=None
):
    _queue_target_media_thumbnail(
        main_window, media_path, q_image, file_type, media_id, metadata
    )


# Functions to add Buttons with thumbnail for selecting videos/images and faces
@QtCore.Slot(str, QtGui.QImage, str, str, int, int)
def add_webcam_thumbnail_to_target_videos_list(
    main_window: "MainWindow",
    media_path,
    q_image,
    file_type,
    media_id,
    webcam_index,
    webcam_backend,
):
    add_media_thumbnail_button(
        main_window,
        widget_components.TargetMediaCardButton,
        main_window.targetVideosList,
        main_window.target_videos,
        q_image,
        media_path=media_path,
        file_type=file_type,
        media_id=media_id,
        is_webcam=True,
        webcam_index=webcam_index,
        webcam_backend=webcam_backend,
    )


@QtCore.Slot()
def add_media_thumbnail_to_target_faces_list(
    main_window: "MainWindow", cropped_face, embedding_store, image_data, face_id
):
    add_media_thumbnail_button(
        main_window,
        widget_components.TargetFaceCardButton,
        main_window.targetFacesList,
        main_window.target_faces,
        image_data,
        cropped_face=cropped_face,
        embedding_store=embedding_store,
        face_id=face_id,
    )


@QtCore.Slot(str, object, object, QtGui.QImage, str)
def add_media_thumbnail_to_source_faces_list(
    main_window: "MainWindow",
    media_path,
    cropped_face,
    embedding_store,
    q_image,
    face_id,
):
    # Route the incoming data to the queue instead of instantiating the UI widget immediately.
    # This delegates the heavy lifting to the batching system, preventing UI freezing and flickering.
    _queue_input_face_thumbnail(
        main_window, media_path, cropped_face, embedding_store, q_image, face_id
    )


def add_media_thumbnail_button(
    main_window: "MainWindow",
    buttonClass: "Type[widget_components.CardButton]",
    listWidget: QtWidgets.QListWidget,
    buttons_list: Dict,
    image_data,  # Accepts QImage (from workers) or QPixmap (from main thread)
    **kwargs,
):
    if buttonClass == widget_components.TargetMediaCardButton:
        constructor_args = [
            kwargs.get("media_path"),
            kwargs.get("file_type"),
            kwargs.get("media_id"),
        ]
        if kwargs.get("is_webcam"):
            constructor_args.extend(
                [
                    kwargs.get("is_webcam"),
                    kwargs.get("webcam_index"),
                    kwargs.get("webcam_backend"),
                ]
            )
    elif buttonClass in (
        widget_components.TargetFaceCardButton,
        widget_components.InputFaceCardButton,
    ):
        constructor_args = [
            kwargs.get("media_path", ""),
            kwargs.get("cropped_face"),
            kwargs.get("embedding_store"),
            kwargs.get("face_id"),
        ]

    if buttonClass == widget_components.TargetMediaCardButton:
        button_size = QtCore.QSize(*_TARGET_BUTTON_SIZE)
    else:
        button_size = QtCore.QSize(*_get_face_button_size(main_window))

    button: widget_components.CardButton = buttonClass(
        *constructor_args, main_window=main_window
    )

    # --- Main thread conversion ---
    if isinstance(image_data, QtGui.QImage):
        pixmap = QtGui.QPixmap.fromImage(image_data)
    else:
        pixmap = image_data

    button.setFixedSize(button_size)
    button.setCheckable(True)

    if buttonClass == widget_components.TargetMediaCardButton:
        button.set_thumbnail_pixmap(pixmap)
    else:
        button.setIcon(QtGui.QIcon(pixmap))
        button.setIconSize(
            button_size - QtCore.QSize(8, 8)
        )  # Slightly smaller than the button size to add some margin

    if buttonClass in [
        widget_components.TargetFaceCardButton,
        widget_components.InputFaceCardButton,
    ]:
        buttons_list[button.face_id] = button
    elif buttonClass == widget_components.TargetMediaCardButton:
        buttons_list[button.media_id] = button
    elif buttonClass == widget_components.EmbeddingCardButton:
        buttons_list[button.embedding_id] = button

    # Create a QListWidgetItem and set the button as its widget
    list_item = sortable_widgets.SortableListWidgetItem()
    list_item.setSizeHint(button_size)
    # Align the item to center
    list_item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

    if buttonClass == widget_components.TargetMediaCardButton:
        # Populate the sort/filter keys before inserting: the target media list has
        # sorting enabled, so the item is placed using this data on addItem().
        _apply_target_media_item_data(list_item, button, kwargs.get("metadata"))

    listWidget.addItem(list_item)
    button.list_item = list_item
    button.list_widget = listWidget
    list_item.setCardButton(button)
    listWidget.setItemWidget(list_item, button)

    if buttonClass == widget_components.TargetFaceCardButton:
        refresh_target_face_display_labels(main_window)


def _apply_target_media_item_data(
    list_item: "sortable_widgets.SortableListWidgetItem",
    button: "widget_components.TargetMediaCardButton",
    metadata,
) -> None:
    """Attach the sort/filter keys for a target media item.

    Everything here comes from data the loader thread already gathered, so no
    media file is opened on the GUI thread.
    """
    list_item.setFileType(button.file_type or "")

    media_path = button.media_path
    # Webcams have no file on disk, so they only carry their file type.
    if media_path and not button.is_webcam:
        list_item.setFileInfo(QtCore.QFileInfo(str(media_path)))

    if metadata is not None:
        list_item.setMediaMetadata(metadata)
        imgdim = sortable_widgets.SortableImageDimension.from_metadata(metadata)
        if imgdim is not None:
            list_item.setImageDimension(imgdim)

    tooltip = list_item.to_tooltip()
    if tooltip:
        button.setToolTip(tooltip)


def refresh_target_face_display_labels(main_window: "MainWindow"):
    target_faces_list = getattr(main_window, "targetFacesList", None)
    if target_faces_list is None:
        return

    for i in range(target_faces_list.count()):
        list_item = target_faces_list.item(i)
        target_face_button = target_faces_list.itemWidget(list_item)
        if isinstance(target_face_button, widget_components.TargetFaceCardButton):
            target_face_button.refresh_display_label()


def _get_face_button_size(main_window: "MainWindow") -> tuple[int, int]:
    return getattr(main_window, "face_thumbnail_button_size", _FACE_BUTTON_SIZE)


def apply_face_thumbnail_size(
    main_window: "MainWindow", button_size_tuple: tuple[int, int]
) -> None:
    main_window.face_thumbnail_button_size = button_size_tuple
    button_size = QtCore.QSize(*button_size_tuple)
    grid_size_with_padding = button_size + QtCore.QSize(4, 4)

    for listWidget in (main_window.targetFacesList, main_window.inputFacesList):
        listWidget.setGridSize(grid_size_with_padding)
        for i in range(listWidget.count()):
            list_item = listWidget.item(i)
            button = listWidget.itemWidget(list_item)
            if button is None:
                continue
            button.setFixedSize(button_size)
            button.setIconSize(
                button_size - QtCore.QSize(8, 8)
            )  # Slightly smaller than the button size to add some margin
            list_item.setSizeHint(button_size)
        listWidget.doItemsLayout()
        listWidget.viewport().update()


def initialize_media_list_widgets(main_window: "MainWindow"):
    """One-time configuration for target/input media and face list widgets."""
    if not hasattr(main_window, "face_thumbnail_button_size"):
        main_window.face_thumbnail_button_size = _FACE_BUTTON_SIZE

    for listWidget, button_size_tuple in [
        (main_window.targetVideosList, _TARGET_BUTTON_SIZE),
        (main_window.targetFacesList, _get_face_button_size(main_window)),
        (main_window.inputFacesList, _get_face_button_size(main_window)),
    ]:
        button_size = QtCore.QSize(*button_size_tuple)
        grid_size_with_padding = button_size + QtCore.QSize(4, 4)
        listWidget.setGridSize(grid_size_with_padding)
        listWidget.setWrapping(True)
        listWidget.setFlow(QtWidgets.QListView.LeftToRight)
        listWidget.setViewMode(QtWidgets.QListView.IconMode)
        listWidget.setMovement(QtWidgets.QListView.Static)
        listWidget.setResizeMode(QtWidgets.QListView.Adjust)
        listWidget.setUniformItemSizes(True)
        # MODIFIED: Both target and input media now use explicit Python queue batchers.
        if listWidget in (main_window.targetVideosList, main_window.inputFacesList):
            listWidget.setLayoutMode(QtWidgets.QListView.SinglePass)
        else:
            listWidget.setLayoutMode(QtWidgets.QListView.Batched)
            listWidget.setBatchSize(_TARGET_MEDIA_BATCH_SIZE)
        listWidget.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
        listWidget.setHorizontalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)

    _set_up_panel_context_menu(
        main_window, main_window.targetVideosList, "target_media"
    )
    _set_up_panel_context_menu(main_window, main_window.inputFacesList, "input_faces")


def initialize_embeddings_list_widget(main_window: "MainWindow"):
    """One-time configuration for the inputEmbeddingsList widget."""
    inputEmbeddingsList = main_window.inputEmbeddingsList

    inputEmbeddingsList.setWrapping(True)
    inputEmbeddingsList.setFlow(QtWidgets.QListView.TopToBottom)

    # Dynamic resizing, more spacing, and disabling uniform sizes
    inputEmbeddingsList.setResizeMode(QtWidgets.QListView.Adjust)
    inputEmbeddingsList.setSpacing(4)
    inputEmbeddingsList.setUniformItemSizes(False)

    inputEmbeddingsList.setViewMode(QtWidgets.QListView.IconMode)
    inputEmbeddingsList.setMovement(QtWidgets.QListView.Static)

    inputEmbeddingsList.setFixedHeight(_EMBED_LIST_HEIGHT)

    inputEmbeddingsList.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
    inputEmbeddingsList.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
    inputEmbeddingsList.setVerticalScrollMode(
        QtWidgets.QAbstractItemView.ScrollPerPixel
    )
    
    # Smooth / slower wheel scrolling so embedding cards are easier to hit
    class _SmoothWheelFilter(QtCore.QObject):
        def __init__(self, list_widget: QtWidgets.QListWidget):
            super().__init__(list_widget)
            self._list = list_widget
            self._step_px = 24  # pixels per wheel notch (lower = slower)

        def eventFilter(self, obj, event):
            if event.type() == QtCore.QEvent.Type.Wheel:
                delta = event.angleDelta().y()
                if delta == 0:
                    delta = event.angleDelta().x()
                # Prefer horizontal bar when content is laid out left-to-right
                hbar = self._list.horizontalScrollBar()
                vbar = self._list.verticalScrollBar()
                steps = -1 if delta > 0 else 1
                if hbar.maximum() > 0:
                    hbar.setValue(hbar.value() + steps * self._step_px)
                elif vbar.maximum() > 0:
                    vbar.setValue(vbar.value() + steps * self._step_px)
                return True
            return super().eventFilter(obj, event)

    if not getattr(inputEmbeddingsList, "_smooth_wheel_filter", None):
        filt = _SmoothWheelFilter(inputEmbeddingsList)
        inputEmbeddingsList.viewport().installEventFilter(filt)
        inputEmbeddingsList._smooth_wheel_filter = filt
    
    inputEmbeddingsList.setHorizontalScrollMode(
        QtWidgets.QAbstractItemView.ScrollPerPixel
    )
    inputEmbeddingsList.setLayoutDirection(QtCore.Qt.LeftToRight)

    # SinglePass layout mode for better dynamic item sizing
    inputEmbeddingsList.setLayoutMode(QtWidgets.QListView.SinglePass)
    _set_up_panel_context_menu(main_window, inputEmbeddingsList, "embeddings")


def sort_embeddings_list_az(main_window: "MainWindow") -> None:
    """Reorder embeddings A-Z via Qt sort (safe with item widgets)."""
    if not main_window.control.get("SortEmbeddingsAZToggle", False):
        return

    list_widget = getattr(main_window, "inputEmbeddingsList", None)
    if list_widget is None or list_widget.count() <= 1:
        return

    for i in range(list_widget.count()):
        item = list_widget.item(i)
        button = list_widget.itemWidget(item)
        name = getattr(button, "embedding_name", "") if button else ""
        item.setText(name or "")

    list_widget.sortItems(QtCore.Qt.AscendingOrder)

    for i in range(list_widget.count()):
        item = list_widget.item(i)
        item.setText("")
        button = list_widget.itemWidget(item)
        if button is not None:
            button.list_item = item


def create_and_add_embed_button_to_list(
    main_window: "MainWindow",
    embedding_name: str,
    embedding_store: dict,
    embedding_id: str,
) -> None:
    inputEmbeddingsList = main_window.inputEmbeddingsList
    embed_button = widget_components.EmbeddingCardButton(
        main_window=main_window,
        embedding_name=embedding_name,
        embedding_store=embedding_store,
        embedding_id=embedding_id,
    )
    embed_button.setStyleSheet("QPushButton { padding: 2px 8px; }")

    size = embed_button.sizeHint()
    size.setHeight(24)
    size.setWidth(max(size.width(), 60))
    embed_button.setFixedSize(size)

    list_item = QtWidgets.QListWidgetItem(inputEmbeddingsList)
    list_item.setSizeHint(size)
    list_item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
    embed_button.list_item = list_item
    inputEmbeddingsList.setItemWidget(list_item, embed_button)
    main_window.merged_embeddings[embed_button.embedding_id] = embed_button
    sort_embeddings_list_az(main_window)


def clear_stop_loading_target_media(main_window: "MainWindow", clear_list: bool = True):
    batch_timer = getattr(main_window, "_target_media_batch_timer", None)
    if batch_timer is not None:
        batch_timer.stop()
    main_window._pending_target_media_thumbnails = deque()
    _reset_target_media_progress(main_window)

    if main_window.video_loader_worker is not None:
        worker = main_window.video_loader_worker
        worker.blockSignals(True)
        worker._running = False
        worker.quit()
        if not worker.wait(_WORKER_STOP_TIMEOUT_MS):
            worker.terminate()
            worker.wait()
        main_window.video_loader_worker = None
        if clear_list:
            main_window.targetVideosList.clear()


@QtCore.Slot()
def select_target_medias(
    main_window: "MainWindow", source_type="folder", folder_name=False, files_list=None
):
    from app.ui.widgets.actions import video_control_actions

    if video_control_actions.block_if_issue_scan_active(
        main_window, "change target media"
    ):
        return

    files_list = files_list or []
    if source_type == "folder":
        folder_name = QtWidgets.QFileDialog.getExistingDirectory(
            dir=main_window.last_target_media_folder_path
        )
        if not folder_name:
            return
        main_window.targetVideosPathLineEdit.setText(folder_name)
        main_window.targetVideosPathLineEdit.setToolTip(folder_name)
        main_window.last_target_media_folder_path = folder_name

    elif source_type == "files":
        files_list = QtWidgets.QFileDialog.getOpenFileNames()[0]
        if not files_list:
            return
        # Get Folder name from the first file
        file_dir = misc_helpers.get_dir_of_file(files_list[0])
        main_window.targetVideosPathLineEdit.setText(file_dir)
        main_window.targetVideosPathLineEdit.setToolTip(file_dir)
        main_window.last_target_media_folder_path = file_dir

    clear_stop_loading_target_media(main_window)
    card_actions.clear_target_faces(main_window)

    main_window.selected_video_button = None
    main_window.target_videos = {}

    main_window.video_loader_worker = ui_workers.TargetMediaLoaderWorker(
        main_window=main_window, folder_name=folder_name, files_list=files_list
    )
    main_window.video_loader_worker.thumbnail_ready.connect(
        partial(add_media_thumbnail_to_target_videos_list, main_window)
    )
    main_window.video_loader_worker.finished.connect(
        partial(filter_target_videos, main_window)
    )
    main_window.video_loader_worker.start()

    # Re-bind auto-watch to the newly selected folder
    set_target_folder_auto_watch(
        main_window,
        bool(main_window.control.get("AutoLoadTargetFolderToggle", False)),
    )


@QtCore.Slot()
def filter_target_videos(main_window):
    from app.ui.widgets.actions import video_control_actions

    if _has_pending_target_media_thumbnail_work(main_window):
        QtCore.QTimer.singleShot(0, partial(filter_target_videos, main_window))
        return

    if video_control_actions.is_issue_scan_active(main_window):
        video_control_actions._mark_pending_target_media_refresh(main_window)
        return
    filter_actions.filter_target_videos(main_window)
    load_target_webcams(main_window)


@QtCore.Slot()
def load_target_webcams(main_window: "MainWindow", *args, **kwargs):
    from app.ui.widgets.actions import video_control_actions

    if video_control_actions.is_issue_scan_active(main_window):
        video_control_actions._mark_pending_target_media_refresh(main_window)
        return
    if main_window.targetVideosFilterWebcamsCheckBox.isChecked():
        main_window.video_loader_worker = ui_workers.TargetMediaLoaderWorker(
            main_window=main_window, webcam_mode=True
        )
        main_window.video_loader_worker.webcam_thumbnail_ready.connect(
            partial(add_webcam_thumbnail_to_target_videos_list, main_window)
        )
        main_window.video_loader_worker.start()
    else:
        main_window.placeholder_update_signal.emit(main_window.targetVideosList, True)
        for (
            _,
            target_video,
        ) in main_window.target_videos.copy().items():  # Use a copy of the dict to prevent Dictionary changed during iteration exceptions
            if target_video.file_type == "webcam":
                target_video.remove_target_media_from_list()
                if target_video == main_window.selected_video_button:
                    main_window.selected_video_button = None
        main_window.placeholder_update_signal.emit(main_window.targetVideosList, False)


def clear_stop_loading_input_media(main_window: "MainWindow", clear_list: bool = True):
    # ADDED: Safely stop the batching timer and clear the queue.
    batch_timer = getattr(main_window, "_input_face_batch_timer", None)
    if batch_timer is not None:
        batch_timer.stop()
    main_window._pending_input_face_thumbnails = deque()
    if main_window.input_faces_loader_worker is not None:
        worker = main_window.input_faces_loader_worker
        worker.blockSignals(True)
        worker._running = False
        worker.quit()
        if not worker.wait(_WORKER_STOP_TIMEOUT_MS):
            worker.terminate()
            worker.wait()
        main_window.input_faces_loader_worker = None
        if clear_list:
            main_window.inputFacesList.clear()


def _set_path_line_edit_value(line_edit: QtWidgets.QLineEdit, path: str) -> None:
    line_edit.setText(path)
    line_edit.setToolTip(path)


def _confirm_panel_clear(main_window: "MainWindow", title: str, message: str) -> bool:
    reply = QtWidgets.QMessageBox.question(
        main_window,
        title,
        message,
        QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
        QtWidgets.QMessageBox.No,
    )
    return reply == QtWidgets.QMessageBox.Yes


def clear_all_target_media(main_window: "MainWindow") -> bool:
    from app.ui.widgets.actions import video_control_actions
    import gc
    import torch

    if video_control_actions.block_if_issue_scan_active(main_window, "clear all media"):
        return False

    if not main_window.target_videos:
        return False

    if not main_window.control.get("SkipClearConfirmationToggle", False):
        confirmed = _confirm_panel_clear(
            main_window,
            "Clear All Media",
            "This will remove all target media, including webcams, and reset the "
            "Target Media panel.\n\nFiles on disk will not be deleted.",
        )
        if not confirmed:
            return False

    clear_stop_loading_target_media(main_window, clear_list=False)

    # Stop processor first so nothing tries to read a cleared path
    vp = main_window.video_processor
    vp.stop_processing()
    vp.media_path = None
    vp.file_type = None
    vp.current_frame = None
    if vp.media_capture:
        try:
            vp.media_capture.release()
        except Exception:
            pass
        vp.media_capture = None
    vp._clear_single_frame_preview_caches()

    # Remove items WHILE selected_video_button is still set so deselect
    # clears the preview for the active item
    for target_media_button in list(main_window.target_videos.values()):
        target_media_button.remove_target_media_from_list()

    if main_window.target_faces:
        card_actions.clear_target_faces(main_window, refresh_frame=False)

    main_window.target_videos.clear()
    main_window.selected_video_button = None

    # Ensure preview is cleared even if nothing was selected
    main_window.scene.clear()
    main_window.graphicsViewFrame.update()
    if hasattr(main_window, "timelineContainer"):
        main_window.timelineContainer.thumbnail_track.request_thumbnails()
    main_window.videoSeekSlider.blockSignals(True)
    main_window.videoSeekSlider.setMaximum(1)
    main_window.videoSeekSlider.setValue(0)
    main_window.videoSeekSlider.blockSignals(False)

    _set_path_line_edit_value(main_window.targetVideosPathLineEdit, "")
    main_window.last_target_media_folder_path = ""
    main_window.placeholder_update_signal.emit(main_window.targetVideosList, False)

    # Free media-related GPU memory only (keep models loaded)
    if not getattr(main_window, "is_batch_processing", False):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return True

def clear_all_input_faces(main_window: "MainWindow") -> bool:
    from app.ui.widgets.actions import video_control_actions

    if video_control_actions.block_if_issue_scan_active(main_window, "clear all faces"):
        return False

    if not main_window.input_faces:
        return False

    if not main_window.control.get("SkipClearConfirmationToggle", False):
        confirmed = _confirm_panel_clear(
            main_window,
            "Clear All Faces",
            "This will remove all input faces and reset the Input Faces panel.\n\n"
            "Files on disk will not be deleted.",
        )
        if not confirmed:
            return False

    clear_stop_loading_input_media(main_window, clear_list=False)

    for input_face_button in list(main_window.input_faces.values()):
        input_face_button.remove_kv_data_file()
        input_face_button._remove_face_from_lists()
        input_face_button.deleteLater()

    common_widget_actions.refresh_frame(main_window)
    _set_path_line_edit_value(main_window.inputFacesPathLineEdit, "")
    main_window.last_input_media_folder_path = ""
    main_window.placeholder_update_signal.emit(main_window.inputFacesList, False)
    return True


def clear_all_embeddings(main_window: "MainWindow") -> bool:
    from app.ui.widgets.actions import video_control_actions

    if video_control_actions.block_if_issue_scan_active(
        main_window, "clear all embeddings"
    ):
        return False

    if not main_window.merged_embeddings:
        return False

    if not main_window.control.get("SkipClearConfirmationToggle", False):
        confirmed = _confirm_panel_clear(
            main_window,
            "Clear All Embeddings",
            "This will remove all embeddings and reset the Embeddings panel.\n\n"
            "Files on disk will not be deleted.",
        )
        if not confirmed:
            return False

    card_actions.clear_merged_embeddings(main_window)
    return True

def delete_all_target_media_to_trash(main_window: "MainWindow") -> bool:
    from app.ui.widgets.actions import video_control_actions
    from send2trash import send2trash
    import gc
    import torch

    if video_control_actions.block_if_issue_scan_active(
        main_window, "delete all target media"
    ):
        return False

    # Only real files currently in the target pane (skip webcams / missing paths)
    buttons = [
        b
        for b in (main_window.target_videos or {}).values()
        if not getattr(b, "is_webcam", False)
        and getattr(b, "media_path", None)
        and os.path.exists(b.media_path)
    ]
    if not buttons:
        return False

    if not main_window.control.get("SkipClearConfirmationToggle", False):
        confirmed = _confirm_panel_clear(
            main_window,
            "Delete All Target Files",
            f"Send {len(buttons)} file(s) from the Target Media pane to the recycle bin?\n\n"
            "This cannot be undone from the app (files go to the system trash).",
        )
        if not confirmed:
            return False

    clear_stop_loading_target_media(main_window, clear_list=False)

    vp = main_window.video_processor
    vp.stop_processing()
    vp.media_path = None
    vp.file_type = None
    vp.current_frame = None
    if vp.media_capture:
        try:
            vp.media_capture.release()
        except Exception:
            pass
        vp.media_capture = None
    vp._clear_single_frame_preview_caches()

    for button in list(main_window.target_videos.values()):
        path = getattr(button, "media_path", None)
        is_webcam = getattr(button, "is_webcam", False)
        if path and not is_webcam and os.path.exists(path):
            try:
                send2trash(path)
                print(f"[INFO] {path} has been sent to the trash.")
            except Exception as e:
                print(f"[ERROR] Failed to trash {path}: {e}")
        button.remove_target_media_from_list()

    if main_window.target_faces:
        card_actions.clear_target_faces(main_window, refresh_frame=False)

    main_window.target_videos.clear()
    main_window.selected_video_button = None
    main_window.scene.clear()
    main_window.graphicsViewFrame.update()
    main_window.placeholder_update_signal.emit(main_window.targetVideosList, False)

    if not getattr(main_window, "is_batch_processing", False):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        set_target_folder_auto_watch(
        main_window,
        bool(main_window.control.get("AutoLoadTargetFolderToggle", False)),
    )

    return True

def _build_panel_context_menu(
    main_window: "MainWindow",
    list_widget: QtWidgets.QListWidget,
    panel_type: str,
) -> QtWidgets.QMenu:
    from app.ui.widgets.actions import video_control_actions

    scan_active = video_control_actions.is_issue_scan_active(main_window)
    menu = QtWidgets.QMenu(list_widget)

    if panel_type == "target_media":
        clear_action = QtGui.QAction("Clear All Media", menu)
        clear_action.setEnabled(bool(main_window.target_videos) and not scan_active)
        clear_action.triggered.connect(partial(clear_all_target_media, main_window))
        menu.addAction(clear_action)

        delete_action = QtGui.QAction("⚠ Delete all files to recycle bin", menu)
        has_files = any(
            not getattr(b, "is_webcam", False)
            and getattr(b, "media_path", None)
            and os.path.exists(b.media_path)
            for b in (main_window.target_videos or {}).values()
        )
        delete_action.setEnabled(has_files and not scan_active)
        delete_action.triggered.connect(
            partial(delete_all_target_media_to_trash, main_window)
        )
        menu.addAction(delete_action)
    elif panel_type == "input_faces":
        clear_action = QtGui.QAction("Clear All Faces", menu)
        clear_action.setEnabled(bool(main_window.input_faces) and not scan_active)
        clear_action.triggered.connect(partial(clear_all_input_faces, main_window))
        menu.addAction(clear_action)
    else:
        clear_action = QtGui.QAction("Clear All Embeddings", menu)
        clear_action.setEnabled(bool(main_window.merged_embeddings) and not scan_active)
        clear_action.triggered.connect(partial(clear_all_embeddings, main_window))
        menu.addAction(clear_action)

    return menu


def _show_panel_context_menu(
    main_window: "MainWindow",
    list_widget: QtWidgets.QListWidget,
    panel_type: str,
    position: QtCore.QPoint,
) -> None:
    if list_widget.itemAt(position) is not None:
        return

    menu = _build_panel_context_menu(main_window, list_widget, panel_type)
    menu.exec(list_widget.viewport().mapToGlobal(position))


def _set_up_panel_context_menu(
    main_window: "MainWindow",
    list_widget: QtWidgets.QListWidget,
    panel_type: str,
) -> None:
    list_widget.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
    list_widget.customContextMenuRequested.connect(
        partial(_show_panel_context_menu, main_window, list_widget, panel_type)
    )


@QtCore.Slot()
def select_input_face_images(
    main_window: "MainWindow", source_type="folder", folder_name=False, files_list=None
):
    from app.ui.widgets.actions import video_control_actions

    if video_control_actions.block_if_issue_scan_active(
        main_window, "load input faces"
    ):
        return

    files_list = files_list or []
    if source_type == "folder":
        folder_name = QtWidgets.QFileDialog.getExistingDirectory(
            dir=main_window.last_input_media_folder_path
        )
        if not folder_name:
            return
        main_window.inputFacesPathLineEdit.setText(folder_name)
        main_window.inputFacesPathLineEdit.setToolTip(folder_name)
        main_window.last_input_media_folder_path = folder_name

    elif source_type == "files":
        files_list = QtWidgets.QFileDialog.getOpenFileNames()[0]
        if not files_list:
            return
        file_dir = misc_helpers.get_dir_of_file(files_list[0])
        main_window.inputFacesPathLineEdit.setText(file_dir)
        main_window.inputFacesPathLineEdit.setToolTip(file_dir)
        main_window.last_input_media_folder_path = file_dir

    clear_stop_loading_input_media(main_window)
    card_actions.clear_input_faces(main_window)
    main_window.input_faces_loader_worker = ui_workers.InputFacesLoaderWorker(
        main_window=main_window, folder_name=folder_name, files_list=files_list
    )
    main_window.input_faces_loader_worker.thumbnail_ready.connect(
        partial(add_media_thumbnail_to_source_faces_list, main_window)
    )

    main_window.input_faces_loader_worker.start()


def set_up_list_widget_placeholder(
    main_window: "MainWindow", list_widget: QtWidgets.QListWidget
):
    # Placeholder label
    placeholder_label = QtWidgets.QLabel(list_widget)
    placeholder_label.setText(
        "<html><body style='text-align:center;'>"
        "<p>Drop Files</p>"
        "<p><b>or</b></p>"
        "<p>Click here to Select a Folder</p>"
        "</body></html>"
    )
    # placeholder_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
    placeholder_label.setStyleSheet("color: gray; font-size: 15px; font-weight: bold;")

    # Center the label inside the QListWidget
    # placeholder_label.setGeometry(list_widget.rect())  # Match QListWidget's size
    placeholder_label.setAttribute(
        QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents
    )  # Allow interactions to pass through
    placeholder_label.setVisible(not list_widget.count())  # Show if the list is empty

    # Use a QVBoxLayout to center the placeholder label
    layout = QtWidgets.QVBoxLayout(list_widget)
    layout.addWidget(placeholder_label)
    layout.setAlignment(
        QtCore.Qt.AlignmentFlag.AlignCenter
    )  # Center the label vertically and horizontally
    layout.setContentsMargins(0, 0, 0, 0)  # Remove margins to ensure full coverage

    # Keep a reference for toggling visibility later
    list_widget.placeholder_label = placeholder_label
    # Set default cursor as PointingHand
    list_widget.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)


def select_output_media_folder(main_window: "MainWindow"):
    folder_name = QtWidgets.QFileDialog.getExistingDirectory(main_window)
    if folder_name:
        main_window.outputFolderLineEdit.setText(folder_name)
        common_widget_actions.create_control(
            main_window, "OutputMediaFolder", folder_name
        )


def _show_missing_folder_message(
    main_window: "MainWindow", folder_label: str, folder_name: str | None = None
) -> None:
    if folder_name:
        message = f"Could not find:\n{folder_name}"
    else:
        message = f"No {folder_label.lower()} is currently set."
    common_widget_actions.create_and_show_messagebox(
        main_window,
        f"{folder_label} Unavailable",
        message,
        parent_widget=main_window,
    )


def _open_folder_in_file_manager(main_window: "MainWindow", folder_name: str) -> None:
    normalized_path = os.path.normpath(os.path.abspath(folder_name))

    if sys.platform == "win32":
        try:
            subprocess.Popen(["explorer", normalized_path])
        except FileNotFoundError:
            subprocess.Popen([r"C:\Windows\explorer.exe", normalized_path])
    elif sys.platform == "darwin":
        subprocess.run(["open", normalized_path])
    else:
        subprocess.run(["xdg-open", normalized_path])


def open_output_media_folder(main_window: "MainWindow", folder_name: str | None = None):
    if not folder_name:
        configured_folder = main_window.control.get("OutputMediaFolder")
        folder_name = configured_folder if isinstance(configured_folder, str) else None
    if not isinstance(folder_name, str) or not folder_name.strip():
        _show_missing_folder_message(main_window, "Output Folder")
        return
    if not os.path.isdir(folder_name):
        _show_missing_folder_message(main_window, "Output Folder", folder_name)
        return
    _open_folder_in_file_manager(main_window, folder_name)


def open_target_media_folder(main_window: "MainWindow"):
    folder_name = main_window.targetVideosPathLineEdit.text().strip()
    if not folder_name:
        folder_name = getattr(main_window, "last_target_media_folder_path", "").strip()
    if not folder_name:
        _show_missing_folder_message(main_window, "Target Media Folder")
        return
    if not os.path.isdir(folder_name):
        _show_missing_folder_message(main_window, "Target Media Folder", folder_name)
        return
    _open_folder_in_file_manager(main_window, folder_name)


def open_input_faces_folder(main_window: "MainWindow"):
    folder_name = main_window.inputFacesPathLineEdit.text().strip()
    if not folder_name:
        folder_name = getattr(main_window, "last_input_media_folder_path", "").strip()
    if not folder_name:
        _show_missing_folder_message(main_window, "Input Faces Folder")
        return
    if not os.path.isdir(folder_name):
        _show_missing_folder_message(main_window, "Input Faces Folder", folder_name)
        return
    _open_folder_in_file_manager(main_window, folder_name)


def show_shortcuts(main_window: "MainWindow"):
    # HTML formating
    shortcuts_text = (
        "<b><u>Actions:</u></b><br>"
        "<b>F11</b> : Fullscreen<br>"
        "<b>T</b> : Theatre Mode<br>"
        "<b>Space</b> : Play/Stop<br>"
        "<b>R</b> : Record start/stop<br>"
        "<b>S</b> : Swap face<br>"
        "<br>"
        "<b><u>Seeking:</u></b><br>"
        "<b>V</b> : Advance 1 frame<br>"
        "<b>C</b> : Rewind 1 frame<br>"
        "<b>D</b> : Advance frames by slider value<br>"
        "<b>A</b> : Rewind frames by slider value<br>"
        "<b>Z</b> : Seek to start<br>"
        "<br>"
        "<b><u>Markers:</u></b><br>"
        "<b>F</b> : Add video marker<br>"
        "<b>ALT+F</b> : Remove video marker<br>"
        "<b>W</b> : Move to next marker<br>"
        "<b>Q</b> : Move to previous marker<br>"
        "<br>"
        "<b><u>Viewport:</u></b><br>"
        "<b>Ctrl+0</b> : Fit to View<br>"
        "<b>Ctrl+1</b> : 100% Zoom<br>"
        "<b>Middle Mouse Drag</b> : Pan view<br>"
        "<b>X</b> : Face Compare<br>"
        "<b>Right Click</b> : Viewport menu (Fit to View, 100% Zoom, Save Image)<br>"
        "<br>"
    )

    main_window.display_messagebox_signal.emit(
        "Shortcuts",
        shortcuts_text,
        main_window,
    )


def show_presets(main_window: "MainWindow"):
    # HTML formating
    presets_text = (
        "<b><u>What are Presets?</u></b><br>"
        "Presets are a functionality that allows saving and applying parameters on swapped faces.<br>"
        "Saved options come from the: 'Face Swap', 'Face Editor', 'Restorers', 'Denoiser', and 'Settings' tabs."
        "<br><br>"
        "<b><u>Option Categories</u></b><br>"
        "There are two distinct categories:"
        "<br><br>"
        "<b>1. Parameters (Applied <u>per face</u>)</b><br>"
        "Includes all options from:<br>"
        "&nbsp;&nbsp;&bull; 'Face Swap'<br>"
        "&nbsp;&nbsp;&bull; 'Face Editor'<br>"
        "&nbsp;&nbsp;&bull; 'Restorers'"
        "<br><br>"
        "<b>2. Controls (Applied <u>globally</u>)</b><br>"
        "Includes all options from:<br>"
        "&nbsp;&nbsp;&bull; 'Denoiser'<br>"
        "&nbsp;&nbsp;&bull; 'Settings'"
        "<br><br>"
        # Une couleur (ex: #FFCC00 pour jaune/orange) aide à attirer l'œil
        "<b><u><font color='#FFCC00'>IMPORTANT</font></u></b><br>"
        "To apply the <b>Controls</b> options (Denoiser/Settings), the "
        "<b>'Apply Settings'</b> button <u>must be checked</u> (it is OFF by default)."
    )

    main_window.display_messagebox_signal.emit(
        "Presets",
        presets_text,
        main_window,
    )


def _get_app_display_metadata(main_window: "MainWindow") -> AppDisplayMetadata:
    metadata = getattr(main_window, "app_display_metadata", None)
    if metadata is not None:
        return metadata

    base_title = getattr(main_window, "_base_window_title", main_window.windowTitle())
    return get_app_display_metadata(main_window.project_root_path, base_title)


def _open_about_link(main_window: "MainWindow", link_type: str):
    project_root = Path(main_window.project_root_path)
    local_links = {
        "quickstart": project_root / "docs" / "quickstart.md",
        "manual": project_root / "docs" / "user_manual.md",
    }
    remote_links = {
        "github": "https://github.com/VisoMasterFusion/VisoMaster-Fusion",
        "discord": "https://discord.gg/5rx4SQuDbp",
    }

    if link_type in local_links:
        target_path = local_links[link_type]
        if target_path.is_file():
            QtGui.QDesktopServices.openUrl(
                QtCore.QUrl.fromLocalFile(str(target_path.resolve()))
            )
        else:
            common_widget_actions.create_and_show_messagebox(
                main_window,
                "Document Not Found",
                f"Could not find:\n{target_path}",
                parent_widget=main_window,
            )
        return

    target_url = remote_links.get(link_type)
    if target_url:
        QtGui.QDesktopServices.openUrl(QtCore.QUrl(target_url))


def show_about(main_window: "MainWindow"):
    dialog = QtWidgets.QDialog(main_window)
    dialog.setWindowTitle("About")
    dialog.setModal(True)
    dialog.setMinimumWidth(420)

    layout = QtWidgets.QVBoxLayout(dialog)
    layout.setContentsMargins(18, 18, 18, 18)
    layout.setSpacing(12)

    title_label = QtWidgets.QLabel("VisoMaster Fusion", dialog)
    title_font = title_label.font()
    title_font.setPointSize(title_font.pointSize() + 2)
    title_font.setBold(True)
    title_label.setFont(title_font)

    version_label = QtWidgets.QLabel(
        _get_app_display_metadata(main_window).about_version_text, dialog
    )
    description_label = QtWidgets.QLabel(
        "Advanced image and video editing toolkit.\n"
        "See the User Manual for setup and usage guidance.",
        dialog,
    )
    description_label.setWordWrap(True)

    links_group = QtWidgets.QGroupBox("Quick Links", dialog)
    links_layout = QtWidgets.QVBoxLayout(links_group)
    links_layout.setContentsMargins(12, 12, 12, 12)
    links_layout.setSpacing(6)

    links_label = QtWidgets.QLabel(links_group)
    links_label.setTextFormat(QtCore.Qt.TextFormat.RichText)
    links_label.setTextInteractionFlags(
        QtCore.Qt.TextInteractionFlag.TextBrowserInteraction
    )
    links_label.setOpenExternalLinks(False)
    links_label.setWordWrap(True)
    links_label.setText(
        '<a href="quickstart">Quick Start Guide</a><br>'
        '<a href="manual">User Manual</a><br>'
        '<a href="discord">Discord</a><br>'
        '<a href="github">GitHub</a>'
    )
    links_label.linkActivated.connect(
        lambda link_type: _open_about_link(main_window, link_type)
    )
    links_layout.addWidget(links_label)

    close_button = QtWidgets.QPushButton("Close", dialog)
    close_button.clicked.connect(dialog.accept)

    layout.addWidget(title_label)
    layout.addWidget(version_label)
    layout.addWidget(description_label)
    layout.addWidget(links_group)
    layout.addWidget(close_button, alignment=QtCore.Qt.AlignmentFlag.AlignRight)

    dialog.exec()

def _get_target_folder_path(main_window: "MainWindow") -> str:
    path = ""
    line = getattr(main_window, "targetVideosPathLineEdit", None)
    if line is not None:
        path = (line.text() or "").strip()
    if not path:
        path = (getattr(main_window, "last_target_media_folder_path", "") or "").strip()
    return path if path and os.path.isdir(path) else ""


def _existing_target_media_paths(main_window: "MainWindow") -> set[str]:
    paths = set()
    for button in (main_window.target_videos or {}).values():
        media_path = getattr(button, "media_path", None)
        if media_path:
            paths.add(os.path.abspath(media_path))
    return paths


def _collect_watch_dirs(folder: str, recursive: bool) -> list[str]:
    dirs = [folder]
    if recursive:
        for dirpath, dirnames, _ in os.walk(folder):
            for d in dirnames:
                dirs.append(os.path.join(dirpath, d))
    return dirs


def _is_target_media_file_stable(
    main_window: "MainWindow", path: str
) -> bool:
    """True once ``path`` has been observed at the same non-zero size across
    two consecutive scans.

    Files inside a watched folder (e.g. the destination of an in-progress
    browser download) can appear in a ``directoryChanged`` event before all
    their bytes are written to disk. Reading such a file yields a
    truncated/empty buffer, which surfaces downstream as
    ``cv2.imdecode`` raising "(-215:Assertion failed) !buf.empty()".
    Requiring the size to be stable across two scan passes (roughly one
    debounce interval apart) avoids handing a still-writing file to the
    loader.
    """
    size_cache = getattr(main_window, "_target_folder_pending_sizes", None)
    if size_cache is None:
        size_cache = {}
        main_window._target_folder_pending_sizes = size_cache

    try:
        size = os.path.getsize(path)
    except OSError:
        # File briefly inaccessible (still being created/locked) — not stable yet.
        return False

    if size == 0:
        size_cache[path] = (size, size_cache.get(path, (0, time.monotonic()))[1])
        return False

    prev = size_cache.get(path)
    if prev is not None and prev[0] == size:
        size_cache.pop(path, None)
        return True

    # First time seeing this file, or its size changed since last scan —
    # remember the size (and first-seen time) and check again next cycle.
    first_seen = prev[1] if prev is not None else time.monotonic()
    size_cache[path] = (size, first_seen)
    return False


# Files pending longer than this are loaded anyway on the next scan rather
# than polled forever (e.g. a stalled/failed download that will never
# finish) — the existing read_image_file error handling covers a genuine
# failure at that point, same as before this stability check existed.
_TARGET_MEDIA_STABILITY_TIMEOUT_SECONDS = 30.0


def scan_and_append_new_target_media(main_window: "MainWindow"):
    """Append only new media files from the configured target folder."""
    from app.ui.widgets.actions import video_control_actions
    import app.helpers.miscellaneous as misc_helpers
    from app.ui.widgets import ui_workers

    if video_control_actions.block_if_issue_scan_active(
        main_window, "auto-load target media"
    ):
        return

    folder = _get_target_folder_path(main_window)
    if not folder:
        return

    recursive = bool(
        main_window.control.get("AutoLoadTargetFolderRecursiveToggle", False)
    )
    if recursive:
        media_files = []
        for dirpath, _, filenames in os.walk(folder):
            for filename in filenames:
                full = os.path.abspath(os.path.join(dirpath, filename))
                if misc_helpers.get_file_type(full):
                    media_files.append(full)
    else:
        media_files = [
            os.path.abspath(p)
            for p in (
                misc_helpers.get_video_files(folder, False)
                + misc_helpers.get_image_files(folder, False)
            )
        ]

    existing = _existing_target_media_paths(main_window)
    new_files = [p for p in media_files if p not in existing]
    if not new_files:
        return

    ready_files = []
    pending = False
    for p in new_files:
        if _is_target_media_file_stable(main_window, p):
            ready_files.append(p)
            continue

        # Still changing size (or briefly unreadable). Give up waiting and
        # load it anyway once it's been pending too long.
        size_cache = getattr(main_window, "_target_folder_pending_sizes", {})
        first_seen = size_cache.get(p, (None, time.monotonic()))[1]
        if time.monotonic() - first_seen >= _TARGET_MEDIA_STABILITY_TIMEOUT_SECONDS:
            print(
                f"[WARN] '{p}' has not finished writing after "
                f"{_TARGET_MEDIA_STABILITY_TIMEOUT_SECONDS:.0f}s — loading it anyway."
            )
            ready_files.append(p)
            size_cache.pop(p, None)
        else:
            pending = True

    if pending:
        # Re-check the still-writing file(s) on the next debounce cycle
        # instead of failing to load them now.
        timer = getattr(main_window, "_target_folder_watch_timer", None)
        if timer is not None:
            timer.start()

    if not ready_files:
        return

    if main_window.video_loader_worker is not None and main_window.video_loader_worker.isRunning():
        return

    main_window.video_loader_worker = ui_workers.TargetMediaLoaderWorker(
        main_window=main_window,
        files_list=ready_files,
        sort_files_list_by_name=True,
    )
    main_window.video_loader_worker.thumbnail_ready.connect(
        partial(add_media_thumbnail_to_target_videos_list, main_window)
    )
    main_window.video_loader_worker.finished.connect(
        partial(filter_target_videos, main_window)
    )
    main_window.video_loader_worker.start()


def set_target_folder_auto_watch(main_window: "MainWindow", enabled: bool):
    watcher = getattr(main_window, "_target_folder_watcher", None)
    timer = getattr(main_window, "_target_folder_watch_timer", None)

    if watcher is None:
        watcher = QtCore.QFileSystemWatcher(main_window)
        main_window._target_folder_watcher = watcher

        def _on_dir_changed(path: str):
            t = getattr(main_window, "_target_folder_watch_timer", None)
            if t is not None:
                t.start()

        watcher.directoryChanged.connect(_on_dir_changed)

    if timer is None:
        timer = QtCore.QTimer(main_window)
        timer.setSingleShot(True)
        timer.setInterval(800)
        timer.timeout.connect(partial(scan_and_append_new_target_media, main_window))
        main_window._target_folder_watch_timer = timer

    for d in list(watcher.directories()):
        watcher.removePath(d)

    if not enabled:
        timer.stop()
        return

    folder = _get_target_folder_path(main_window)
    if not folder:
        return

    recursive = bool(
        main_window.control.get("AutoLoadTargetFolderRecursiveToggle", False)
    )
    for d in _collect_watch_dirs(folder, recursive):
        watcher.addPath(d)

    scan_and_append_new_target_media(main_window)
