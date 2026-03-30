from functools import partial
from typing import TYPE_CHECKING, Dict, Type
import subprocess
import sys
import os

from PySide6 import QtWidgets, QtGui, QtCore

from app.ui.widgets.actions import common_actions as common_widget_actions
from app.ui.widgets.actions import card_actions
from app.ui.widgets.actions import filter_actions
from app.ui.widgets import widget_components
import app.helpers.miscellaneous as misc_helpers
from app.ui.widgets import ui_workers

if TYPE_CHECKING:
    from app.ui.main_ui import MainWindow

_WORKER_STOP_TIMEOUT_MS = 1000
_TARGET_BUTTON_SIZE = (90, 90)
_FACE_BUTTON_SIZE = (70, 70)
_EMBED_BUTTON_SIZE = (120, 25)
_EMBED_LIST_HEIGHT = 140


# Functions to add Buttons with thumbnail for selecting videos/images and faces
@QtCore.Slot(str, QtGui.QImage, str, str)
def add_media_thumbnail_to_target_videos_list(
    main_window: "MainWindow", media_path, q_image, file_type, media_id
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
    add_media_thumbnail_button(
        main_window,
        widget_components.InputFaceCardButton,
        main_window.inputFacesList,
        main_window.input_faces,
        q_image,
        media_path=media_path,
        cropped_face=cropped_face,
        embedding_store=embedding_store,
        face_id=face_id,
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
        button_size = QtCore.QSize(*_FACE_BUTTON_SIZE)

    button: widget_components.CardButton = buttonClass(
        *constructor_args, main_window=main_window
    )

    # --- Main thread conversion ---
    if isinstance(image_data, QtGui.QImage):
        pixmap = QtGui.QPixmap.fromImage(image_data)
    else:
        pixmap = image_data

    button.setIcon(QtGui.QIcon(pixmap))
    button.setIconSize(
        button_size - QtCore.QSize(8, 8)
    )  # Slightly smaller than the button size to add some margin
    button.setFixedSize(button_size)
    button.setCheckable(True)

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
    list_item = QtWidgets.QListWidgetItem(listWidget)
    list_item.setSizeHint(button_size)
    button.list_item = list_item
    button.list_widget = listWidget
    # Align the item to center
    list_item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
    listWidget.setItemWidget(list_item, button)


def initialize_media_list_widgets(main_window: "MainWindow"):
    """One-time configuration for target/input media and face list widgets."""
    for listWidget, button_size_tuple in [
        (main_window.targetVideosList, _TARGET_BUTTON_SIZE),
        (main_window.targetFacesList, _FACE_BUTTON_SIZE),
        (main_window.inputFacesList, _FACE_BUTTON_SIZE),
    ]:
        button_size = QtCore.QSize(*button_size_tuple)
        grid_size_with_padding = button_size + QtCore.QSize(4, 4)
        listWidget.setGridSize(grid_size_with_padding)
        listWidget.setWrapping(True)
        listWidget.setFlow(QtWidgets.QListView.LeftToRight)
        listWidget.setResizeMode(QtWidgets.QListView.Adjust)


def initialize_embeddings_list_widget(main_window: "MainWindow"):
    """One-time configuration for the inputEmbeddingsList widget."""
    inputEmbeddingsList = main_window.inputEmbeddingsList
    button_size = QtCore.QSize(*_EMBED_BUTTON_SIZE)
    grid_size_with_padding = button_size + QtCore.QSize(4, 4)

    inputEmbeddingsList.setGridSize(grid_size_with_padding)
    inputEmbeddingsList.setWrapping(True)
    inputEmbeddingsList.setFlow(QtWidgets.QListView.TopToBottom)
    inputEmbeddingsList.setResizeMode(QtWidgets.QListView.Fixed)
    inputEmbeddingsList.setSpacing(2)
    inputEmbeddingsList.setUniformItemSizes(True)
    inputEmbeddingsList.setViewMode(QtWidgets.QListView.IconMode)
    inputEmbeddingsList.setMovement(QtWidgets.QListView.Static)

    inputEmbeddingsList.setFixedHeight(_EMBED_LIST_HEIGHT)

    col_width = grid_size_with_padding.width()
    min_width = (3 * col_width) + 16
    inputEmbeddingsList.setMinimumWidth(min_width)

    inputEmbeddingsList.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
    inputEmbeddingsList.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
    inputEmbeddingsList.setVerticalScrollMode(
        QtWidgets.QAbstractItemView.ScrollPerPixel
    )
    inputEmbeddingsList.setHorizontalScrollMode(
        QtWidgets.QAbstractItemView.ScrollPerPixel
    )

    inputEmbeddingsList.setLayoutDirection(QtCore.Qt.LeftToRight)
    inputEmbeddingsList.setLayoutMode(QtWidgets.QListView.Batched)


def create_and_add_embed_button_to_list(
    main_window: "MainWindow", embedding_name, embedding_store, embedding_id
):
    inputEmbeddingsList = main_window.inputEmbeddingsList
    embed_button = widget_components.EmbeddingCardButton(
        main_window=main_window,
        embedding_name=embedding_name,
        embedding_store=embedding_store,
        embedding_id=embedding_id,
    )

    button_size = QtCore.QSize(*_EMBED_BUTTON_SIZE)
    embed_button.setFixedSize(button_size)

    list_item = QtWidgets.QListWidgetItem(inputEmbeddingsList)
    list_item.setSizeHint(button_size)
    embed_button.list_item = list_item
    list_item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

    inputEmbeddingsList.setItemWidget(list_item, embed_button)

    main_window.merged_embeddings[embed_button.embedding_id] = embed_button


def clear_stop_loading_target_media(main_window: "MainWindow"):
    if main_window.video_loader_worker is not None:
        worker = main_window.video_loader_worker
        worker._running = False
        worker.quit()
        if not worker.wait(_WORKER_STOP_TIMEOUT_MS):
            worker.terminate()
            worker.wait()
        main_window.video_loader_worker = None
        main_window.targetVideosList.clear()


@QtCore.Slot()
def select_target_medias(
    main_window: "MainWindow", source_type="folder", folder_name=False, files_list=None
):
    files_list = files_list or []
    if source_type == "folder":
        folder_name = QtWidgets.QFileDialog.getExistingDirectory(
            dir=main_window.last_target_media_folder_path
        )
        if not folder_name:
            return
        main_window.labelTargetVideosPath.setText(
            misc_helpers.truncate_text(folder_name)
        )
        main_window.labelTargetVideosPath.setToolTip(folder_name)
        main_window.last_target_media_folder_path = folder_name

    elif source_type == "files":
        files_list = QtWidgets.QFileDialog.getOpenFileNames()[0]
        if not files_list:
            return
        # Get Folder name from the first file
        file_dir = misc_helpers.get_dir_of_file(files_list[0])
        main_window.labelTargetVideosPath.setText(
            file_dir
        )  # Just a temp text until i think of something better
        main_window.labelTargetVideosPath.setToolTip(file_dir)
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


@QtCore.Slot()
def filter_target_videos(main_window):
    filter_actions.filter_target_videos(main_window)
    load_target_webcams(main_window)


@QtCore.Slot()
def load_target_webcams(
    main_window: "MainWindow",
):
    if main_window.filterWebcamsCheckBox.isChecked():
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


def clear_stop_loading_input_media(main_window: "MainWindow"):
    if main_window.input_faces_loader_worker is not None:
        worker = main_window.input_faces_loader_worker
        worker._running = False
        worker.quit()
        if not worker.wait(_WORKER_STOP_TIMEOUT_MS):
            worker.terminate()
            worker.wait()
        main_window.input_faces_loader_worker = None
        main_window.inputFacesList.clear()


@QtCore.Slot()
def select_input_face_images(
    main_window: "MainWindow", source_type="folder", folder_name=False, files_list=None
):
    files_list = files_list or []
    if source_type == "folder":
        folder_name = QtWidgets.QFileDialog.getExistingDirectory(
            dir=main_window.last_input_media_folder_path
        )
        if not folder_name:
            return
        main_window.labelInputFacesPath.setText(misc_helpers.truncate_text(folder_name))
        main_window.labelInputFacesPath.setToolTip(folder_name)
        main_window.last_input_media_folder_path = folder_name

    elif source_type == "files":
        files_list = QtWidgets.QFileDialog.getOpenFileNames()[0]
        if not files_list:
            return
        file_dir = misc_helpers.get_dir_of_file(files_list[0])
        main_window.labelInputFacesPath.setText(
            file_dir
        )  # Just a temp text until i think of something better
        main_window.labelInputFacesPath.setToolTip(file_dir)
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


def open_output_media_folder(main_window: "MainWindow", folder_name: str | None = None):
    if not folder_name:
        folder_name = getattr(main_window.video_processor, "last_output_folder", "")
    if not folder_name:
        folder_name = main_window.control.get("OutputMediaFolder")
    if isinstance(folder_name, str) and folder_name:
        if os.path.exists(folder_name):
            # Normalize path
            normalized_path = os.path.normpath(os.path.abspath(folder_name))

            if sys.platform == "win32":
                # Windows - use full path to explorer.exe to avoid PATH issues
                try:
                    # Method 1: Using subprocess without shell (more secure and reliable)
                    subprocess.Popen(["explorer", normalized_path])
                except FileNotFoundError:
                    # Fallback: Use full path to explorer.exe
                    subprocess.Popen([r"C:\Windows\explorer.exe", normalized_path])
            elif sys.platform == "darwin":
                # macOS
                subprocess.run(["open", "-R", folder_name])
            else:
                # Linux
                directory = os.path.dirname(os.path.abspath(folder_name))
                subprocess.run(["xdg-open", directory])


def show_shortcuts(main_window: "MainWindow"):
    # HTML formating
    shortcuts_text = (
        "<b><u>Actions:</u></b><br>"
        "<b>F11</b> : View fullscreen<br>"
        "<b>T</b> : Theatre Mode<br>"
        "<b>Space</b> : Play/Stop<br>"
        "<b>R</b> : Record start/stop<br>"
        "<b>S</b> : Swap face"
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
