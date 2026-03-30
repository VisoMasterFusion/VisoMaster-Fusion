from typing import TYPE_CHECKING

from PySide6 import QtWidgets

if TYPE_CHECKING:
    from app.ui.main_ui import MainWindow


def filter_target_videos(main_window: "MainWindow", *args):
    main_window.target_videos_filter_worker.stop_thread()

    # Capture all Qt widget data in the main thread before starting the worker
    search_text = main_window.targetVideosSearchBox.text().lower()

    include_file_types = []
    if main_window.filterImagesCheckBox.isChecked():
        include_file_types.append("image")
    if main_window.filterVideosCheckBox.isChecked():
        include_file_types.append("video")
    if main_window.filterWebcamsCheckBox.isChecked():
        include_file_types.append("webcam")

    items_snapshot = []
    for i in range(main_window.targetVideosList.count()):
        item = main_window.targetVideosList.item(i)
        item_widget = main_window.targetVideosList.itemWidget(item)
        if item_widget is not None:
            items_snapshot.append((i, item_widget.media_path, item_widget.file_type))

    worker = main_window.target_videos_filter_worker
    worker.search_text = search_text
    worker.include_file_types = include_file_types
    worker.items_snapshot = items_snapshot
    worker.start()


def filter_input_faces(main_window: "MainWindow", *args):
    main_window.input_faces_filter_worker.stop_thread()

    # Capture all Qt widget data in the main thread before starting the worker
    search_text = main_window.inputFacesSearchBox.text().lower()

    items_snapshot = []
    for i in range(main_window.inputFacesList.count()):
        item = main_window.inputFacesList.item(i)
        item_widget = main_window.inputFacesList.itemWidget(item)
        if item_widget is not None:
            items_snapshot.append((i, item_widget.media_path))

    worker = main_window.input_faces_filter_worker
    worker.search_text = search_text
    worker.items_snapshot = items_snapshot
    worker.start()


def filter_merged_embeddings(main_window: "MainWindow", *args):
    main_window.merged_embeddings_filter_worker.stop_thread()

    # Capture all Qt widget data in the main thread before starting the worker
    search_text = main_window.inputEmbeddingsSearchBox.text().lower()

    items_snapshot = []
    for i in range(main_window.inputEmbeddingsList.count()):
        item = main_window.inputEmbeddingsList.item(i)
        item_widget = main_window.inputEmbeddingsList.itemWidget(item)
        if item_widget is not None:
            items_snapshot.append((i, item_widget.embedding_name))

    worker = main_window.merged_embeddings_filter_worker
    worker.search_text = search_text
    worker.items_snapshot = items_snapshot
    worker.start()


def update_filtered_list(
    main_window: "MainWindow",
    filter_list_widget: QtWidgets.QListWidget,
    visible_indices: list,
    snapshot_size: int = 0,
):
    filter_list_widget.setUpdatesEnabled(False)

    try:
        # Only manage items that existed at snapshot time; items added after the
        # snapshot was captured (index >= snapshot_size) are left visible so they
        # are not accidentally hidden by a stale filter result.
        limit = snapshot_size if snapshot_size > 0 else filter_list_widget.count()
        for i in range(min(limit, filter_list_widget.count())):
            filter_list_widget.item(i).setHidden(True)

        # Show only the items in the visible_indices list
        for i in visible_indices:
            if i < filter_list_widget.count():
                filter_list_widget.item(i).setHidden(False)
    finally:
        filter_list_widget.setUpdatesEnabled(True)
