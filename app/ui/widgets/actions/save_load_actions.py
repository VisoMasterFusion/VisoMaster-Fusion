import os
import json
from pathlib import Path
import uuid
import copy
from functools import partial
from typing import TYPE_CHECKING, Dict, Union, cast

from PySide6 import QtWidgets, QtCore
import numpy as np
import torch

from app.ui.widgets import widget_components
from app.ui.widgets.actions import common_actions as common_widget_actions
from app.ui.widgets.actions import card_actions
from app.ui.widgets.actions import list_view_actions
from app.ui.widgets.actions import video_control_actions
from app.ui.widgets.actions import layout_actions
from app.ui.widgets.actions import filter_actions
from app.ui.widgets import ui_workers
from app.helpers.typing_helper import ParametersTypes, MarkerTypes
import app.helpers.miscellaneous as misc_helpers

if TYPE_CHECKING:
    from app.ui.main_ui import MainWindow


def open_embeddings_from_file(main_window: "MainWindow"):
    embedding_filename, _ = QtWidgets.QFileDialog.getOpenFileName(
        main_window,
        filter="JSON (*.json)",
        dir=misc_helpers.get_dir_of_file(main_window.loaded_embedding_filename),
    )
    if embedding_filename:
        try:
            with open(embedding_filename, "r") as embed_file:  # pylint: disable=unspecified-encoding
                embeddings_list = json.load(embed_file)
                card_actions.clear_merged_embeddings(main_window)

                # Reset for each target face
                for _, target_face in main_window.target_faces.items():
                    target_face.assigned_merged_embeddings = {}
                    target_face.assigned_input_embedding = {}

                # Load embeddings from file and build the embedding_store dictionary
                for embed_data in embeddings_list:
                    embedding_store = embed_data.get("embedding_store", {})
                    # Convert each embedding to a numpy array
                    for recogn_model, embed in embedding_store.items():
                        embedding_store[recogn_model] = np.array(embed)

                    # Pass the entire embedding_store to the function
                    list_view_actions.create_and_add_embed_button_to_list(
                        main_window,
                        embed_data["name"],
                        embedding_store,
                        embedding_id=str(uuid.uuid1().int),
                    )
        except (json.JSONDecodeError, KeyError, TypeError, Exception) as e:
            QtWidgets.QMessageBox.critical(
                main_window, "Error", f"Failed to load embeddings: {e}"
            )
            return

    main_window.loaded_embedding_filename = (
        embedding_filename or main_window.loaded_embedding_filename
    )


def save_embeddings_to_file(main_window: "MainWindow", save_as=False):
    if not main_window.merged_embeddings:
        common_widget_actions.create_and_show_messagebox(
            main_window,
            "Embeddings List Empty!",
            "No Embeddings available to save",
            parent_widget=main_window,
        )
        return

    # Define the save filename
    embedding_filename = main_window.loaded_embedding_filename
    if (
        not embedding_filename
        or not misc_helpers.is_file_exists(embedding_filename)
        or save_as
    ):
        embedding_filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            main_window, filter="JSON (*.json)"
        )

    # Build a list of dicts, each containing the embedding name and its embedding_store
    embeddings_list = [
        {
            "name": embed_button.embedding_name,
            "embedding_store": {
                k: v.tolist() for k, v in embed_button.embedding_store.items()
            },  # Convert embeddings to lists
        }
        for embedding_id, embed_button in main_window.merged_embeddings.items()
    ]

    # Save to file
    if embedding_filename:
        with open(embedding_filename, "w") as embed_file:  # pylint: disable=unspecified-encoding
            embeddings_as_json = json.dumps(
                embeddings_list, indent=4
            )  # Save with indentation for readability
            embed_file.write(embeddings_as_json)

            # Show a confirmation message
            common_widget_actions.create_and_show_toast_message(
                main_window,
                "Embeddings Saved",
                f"Saved Embeddings to file: {embedding_filename}",
            )

        main_window.loaded_embedding_filename = embedding_filename


# This method is used to convert the data type of Parameters Dict
# Parameters are converted to dict when serializing to JSON
# Parameters are converted to ParametersDict when reading from JSON
def convert_parameters_to_supported_type(
    main_window: "MainWindow",
    parameters: Union[dict, ParametersTypes],
    convert_type: type,
):
    if convert_type is dict:
        if isinstance(parameters, misc_helpers.ParametersDict):
            return parameters.data
    elif convert_type is misc_helpers.ParametersDict:
        if isinstance(parameters, dict):
            return misc_helpers.ParametersDict(
                parameters,
                cast(misc_helpers.ParametersDict, main_window.default_parameters).data,
            )
    return parameters


def convert_markers_to_supported_type(
    main_window: "MainWindow",
    markers: MarkerTypes,
    convert_type: type,
):
    # Convert Parameters inside the markers from ParametersDict to dict
    for _, marker_data in markers.items():
        if "parameters" in marker_data:
            for target_face_id, target_parameters in marker_data["parameters"].items():
                marker_data["parameters"][target_face_id] = (
                    convert_parameters_to_supported_type(
                        main_window, target_parameters, convert_type
                    )
                )
    return markers


def save_current_parameters_and_control(main_window: "MainWindow", face_id):
    data_filename, _ = QtWidgets.QFileDialog.getSaveFileName(
        main_window, filter="JSON (*.json)"
    )
    data = {
        "parameters": convert_parameters_to_supported_type(
            main_window, main_window.parameters[face_id], dict
        ),
        "control": main_window.control.copy(),
    }

    if data_filename:
        with open(data_filename, "w") as data_file:  # pylint: disable=unspecified-encoding
            data_as_json = json.dumps(
                data, indent=4
            )  # Save with indentation for readability
            data_file.write(data_as_json)


def load_parameters_and_settings(
    main_window: "MainWindow", face_id, load_settings=False
):
    data_filename, _ = QtWidgets.QFileDialog.getOpenFileName(
        main_window, filter="JSON (*.json)"
    )
    if data_filename:
        with open(data_filename, "r") as data_file:  # pylint: disable=unspecified-encoding
            data = json.load(data_file)
            main_window.parameters[face_id] = convert_parameters_to_supported_type(
                main_window, data["parameters"].copy(), misc_helpers.ParametersDict
            )
            if main_window.selected_target_face_id == face_id:
                common_widget_actions.set_widgets_values_using_face_id_parameters(
                    main_window, face_id
                )
            if load_settings:
                main_window.control.update(data["control"])
                common_widget_actions.set_control_widgets_values(main_window)
            common_widget_actions.refresh_frame(main_window)


def get_auto_load_workspace_toggle(
    main_window: "MainWindow", data_filename: str | bool = False
):
    if not data_filename:
        data_filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            main_window, filter="JSON (*.json)"
        )
    # Check if File exists (In cases when filename is passed as function argument instead of from the file picker)
    if isinstance(data_filename, str) and not Path(data_filename).is_file():
        data_filename = False
    if data_filename:
        with open(data_filename, "r") as data_file:  # pylint: disable=unspecified-encoding
            try:
                data = json.load(data_file)
            except json.JSONDecodeError:
                return False
            control = data["control"]
            return control.get("AutoLoadWorkspaceToggle", False)


def load_saved_workspace(
    main_window: "MainWindow", data_filename: Union[str, bool] = False
):
    if not data_filename:
        data_filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            main_window, filter="JSON (*.json)"
        )
    # Check if File exists (In cases when filename is passed as function argument instead of from the file picker)
    if not (isinstance(data_filename, str) and Path(data_filename).is_file()):
        data_filename = False
    if data_filename:
        with open(data_filename, "r") as data_file:  # pylint: disable=unspecified-encoding
            data = json.load(data_file)
        try:
            list_view_actions.clear_stop_loading_input_media(main_window)
            list_view_actions.clear_stop_loading_target_media(main_window)
            main_window.target_videos = {}
            card_actions.clear_input_faces(main_window)
            card_actions.clear_target_faces(main_window)
            card_actions.clear_merged_embeddings(main_window)

            # Load control (settings)
            control = data.get("control", {})
            for control_name, control_value in control.items():
                main_window.control[control_name] = control_value

            # Add target medias
            target_medias_data = data.get("target_medias_data", [])
            target_medias_files_list = []
            target_media_ids = []
            for media_data in target_medias_data:
                target_medias_files_list.append(media_data["media_path"])
                target_media_ids.append(media_data["media_id"])

            main_window.video_loader_worker = ui_workers.TargetMediaLoaderWorker(
                main_window=main_window,
                folder_name=False,
                files_list=target_medias_files_list,
                media_ids=target_media_ids,
            )
            main_window.video_loader_worker.thumbnail_ready.connect(
                partial(
                    list_view_actions.add_media_thumbnail_to_target_videos_list,
                    main_window,
                )
            )
            main_window.video_loader_worker.run()

            # OPTIMIZED: Force PySide6 to process the pending 'thumbnail_ready' signals
            # before continuing, ensuring UI elements are fully instantiated.
            QtWidgets.QApplication.processEvents()

            # Select target media (Secured with .get to prevent KeyError on older workspaces)
            selected_media_id = data.get("selected_media_id", False)
            if selected_media_id and main_window.target_videos.get(selected_media_id):
                main_window.target_videos[selected_media_id].click()

            # Add input faces (imgs)
            input_media_paths, input_face_ids = [], []
            for face_id, input_face_data in data.get("input_faces_data", {}).items():
                input_media_paths.append(input_face_data["media_path"])
                input_face_ids.append(face_id)
            main_window.input_faces_loader_worker = ui_workers.InputFacesLoaderWorker(
                main_window=main_window,
                folder_name=False,
                files_list=input_media_paths,
                face_ids=input_face_ids,
            )
            main_window.input_faces_loader_worker.thumbnail_ready.connect(
                partial(
                    list_view_actions.add_media_thumbnail_to_source_faces_list,
                    main_window,
                )
            )
            main_window.input_faces_loader_worker.finished.connect(
                partial(common_widget_actions.refresh_frame, main_window)
            )
            # Use run() instead of start(), as we dont want it running in a different thread as it could create synchronisation issues in the steps below
            main_window.input_faces_loader_worker.run()

            # Force PySide6 event loop to flush the queue.
            # This instantly populates `main_window.input_faces` before the next loop tries to access them.
            QtWidgets.QApplication.processEvents()

            for face_id, input_face_data in data.get("input_faces_data", {}).items():
                if face_id in main_window.input_faces:
                    input_face_button = main_window.input_faces[face_id]
                    kv_map_path = input_face_data.get("kv_map")
                    if kv_map_path and os.path.exists(kv_map_path):
                        try:
                            payload = torch.load(kv_map_path, map_location="cpu")
                            if isinstance(payload, dict):
                                input_face_button.kv_map = payload.get("kv_map")
                            else:  # Backwards compatibility
                                input_face_button.kv_map = payload
                        except Exception as e:
                            print(
                                f"[ERROR] Error loading K/V map from {kv_map_path}: {e}"
                            )

            # Add embeddings
            embeddings_data = data.get("embeddings_data", {})
            for embedding_id, embedding_data in embeddings_data.items():
                embedding_store_loaded = {
                    embed_model: np.array(embedding)
                    for embed_model, embedding in embedding_data[
                        "embedding_store"
                    ].items()
                }
                embedding_name = embedding_data["embedding_name"]
                list_view_actions.create_and_add_embed_button_to_list(
                    main_window,
                    embedding_name,
                    embedding_store_loaded,
                    embedding_id=embedding_id,
                )

            # Add target_faces
            for face_id, target_face_data in data.get("target_faces_data", {}).items():
                cropped_face = np.array(target_face_data["cropped_face"]).astype(
                    "uint8"
                )
                pixmap = common_widget_actions.get_pixmap_from_frame(
                    main_window, cropped_face
                )
                embedding_store: Dict[str, np.ndarray] = {
                    embed_model: np.array(embedding)
                    for embed_model, embedding in target_face_data[
                        "embedding_store"
                    ].items()
                }
                list_view_actions.add_media_thumbnail_to_target_faces_list(
                    main_window, cropped_face, embedding_store, pixmap, face_id
                )
                main_window.parameters[face_id] = convert_parameters_to_supported_type(
                    main_window,
                    data["target_faces_data"][face_id]["parameters"],
                    misc_helpers.ParametersDict,
                )

                # Set assigned embeddinng buttons
                embed_buttons = main_window.merged_embeddings
                assigned_merged_embeddings: list = target_face_data[
                    "assigned_merged_embeddings"
                ]
                for assigned_merged_embedding_id in assigned_merged_embeddings:
                    main_window.target_faces[face_id].assigned_merged_embeddings[
                        assigned_merged_embedding_id
                    ] = embed_buttons[assigned_merged_embedding_id].embedding_store

                # Set assigned input face buttons
                assigned_input_faces: list = target_face_data["assigned_input_faces"]
                for assigned_input_face_id in assigned_input_faces:
                    if assigned_input_face_id in main_window.input_faces:
                        main_window.target_faces[face_id].assigned_input_faces[
                            assigned_input_face_id
                        ] = main_window.input_faces[
                            assigned_input_face_id
                        ].embedding_store
                    else:
                        print(
                            f"[WARN] Input face {assigned_input_face_id} missing from session. Skipping assignment."
                        )

                # Set assigned input embedding (Input face + merged embeddings)
                assigned_input_embedding = {
                    embed_model: np.array(embedding)
                    for embed_model, embedding in target_face_data[
                        "assigned_input_embedding"
                    ].items()
                }
                main_window.target_faces[
                    face_id
                ].assigned_input_embedding = assigned_input_embedding

            # Add markers
            video_control_actions.remove_all_markers(main_window)

            # Load job marker pairs (New format)
            main_window.job_marker_pairs = data.get("job_marker_pairs", [])
            # Fallback for old format (job_start_frame, job_end_frame)
            if (
                not main_window.job_marker_pairs
            ):  # Only try fallback if new format wasn't found
                job_start_frame = data.get("job_start_frame", None)
                job_end_frame = data.get("job_end_frame", None)
                if job_start_frame is not None:
                    main_window.job_marker_pairs.append(
                        (job_start_frame, job_end_frame)
                    )

            # Convert params to ParametersDict
            data["markers"] = convert_markers_to_supported_type(
                main_window, data.get("markers", {}), misc_helpers.ParametersDict
            )

            for marker_position, marker_data in data["markers"].items():
                video_control_actions.add_marker(
                    main_window,
                    marker_data["parameters"],
                    marker_data["control"],
                    int(marker_position),
                )
            loaded_issue_frames_by_face = data.get("issue_frames_by_face")
            if loaded_issue_frames_by_face is not None:
                video_control_actions.set_issue_frames_by_face(
                    main_window, loaded_issue_frames_by_face
                )
            else:
                selected_face_id = getattr(main_window, "selected_target_face_id", None)
                if selected_face_id is None and getattr(
                    main_window, "target_faces", {}
                ):
                    selected_face_id = str(next(iter(main_window.target_faces.keys())))
                if selected_face_id is not None:
                    video_control_actions.set_issue_frames_for_face(
                        main_window, selected_face_id, data.get("issue_frames", [])
                    )
                else:
                    video_control_actions.set_issue_frames_by_face(main_window, {})
            video_control_actions.set_dropped_frames(
                main_window, data.get("dropped_frames", [])
            )
            # main_window.videoSeekSlider.setValue(0)
            # video_control_actions.update_widget_values_from_markers(main_window, 0)

            # Update slider visuals after loading markers
            main_window.videoSeekSlider.update()
            video_control_actions.update_drop_frame_button_label(main_window)

            # Set target media and input faces folder names
            main_window.last_target_media_folder_path = data.get(
                "last_target_media_folder_path", ""
            )
            main_window.last_input_media_folder_path = data.get(
                "last_input_media_folder_path", ""
            )
            main_window.loaded_embedding_filename = data.get(
                "loaded_embedding_filename", ""
            )
            common_widget_actions.set_control_widgets_values(main_window)
            # Set output folder using .get() with a default empty string
            output_folder = control.get("OutputMediaFolder", "")
            common_widget_actions.create_control(
                main_window, "OutputMediaFolder", output_folder
            )
            # Also use .get() when setting the line edit text
            main_window.outputFolderLineEdit.setText(output_folder)

            # Recalculate assigned embeddings and K/V maps for all target faces
            for target_face_button in main_window.target_faces.values():
                target_face_button.calculate_assigned_input_embedding()

            # Restore tab order if present in the saved data
            if "tab_state" in data:
                tab_state = data["tab_state"]

                # Create a mapping of tab text to tab indices
                tab_texts = {}
                for i in range(main_window.tabWidget.count()):
                    tab_texts[main_window.tabWidget.tabText(i)] = i

                # Reorder the tabs based on the saved order
                for i, tab_info in enumerate(tab_state["tab_order"]):
                    tab_text = tab_info["text"]
                    if tab_text in tab_texts:
                        current_index = tab_texts[tab_text]
                        # Only move if not already in the right position
                        if current_index != i:
                            main_window.tabWidget.tabBar().moveTab(current_index, i)
                            # Update the mapping after moving the tab
                            tab_texts = {}
                            for j in range(main_window.tabWidget.count()):
                                tab_texts[main_window.tabWidget.tabText(j)] = j

                # Set the active tab index
                if "current_tab_index" in tab_state:
                    main_window.tabWidget.setCurrentIndex(
                        tab_state["current_tab_index"]
                    )

            layout_actions.fit_image_to_view_onchange(main_window)

            if main_window.target_faces:
                list(main_window.target_faces.values())[0].click()
            else:
                main_window.current_widget_parameters = data.get(
                    "current_widget_parameters", main_window.default_parameters.copy()
                )
                main_window.current_widget_parameters = cast(
                    ParametersTypes,
                    misc_helpers.ParametersDict(
                        main_window.current_widget_parameters,
                        cast(
                            misc_helpers.ParametersDict, main_window.default_parameters
                        ).data,
                    ),
                )
                common_widget_actions.set_widgets_values_using_face_id_parameters(
                    main_window, face_id=None
                )

            # Restore Window State
            window_state = data.get("window_state_data", {})
            is_maximized = window_state.get("isMaximized", False)
            is_fullScreen = window_state.get("isFullScreen", False)

            if is_maximized:
                main_window.resize(main_window.sizeHint())
                main_window.showMaximized()
            elif is_fullScreen:
                main_window.resize(main_window.sizeHint())
                main_window.showFullScreen()
                main_window.menuBar().hide()
                main_window.is_full_screen = True
            else:
                main_window.setGeometry(
                    window_state.get("x", main_window.x()),
                    window_state.get("y", main_window.y()),
                    window_state.get("width", main_window.width()),
                    window_state.get("height", main_window.height()),
                )
            main_window.TargetMediaCheckBox.setChecked(
                window_state.get("TargetMediaCheckBox", True)
            )
            main_window.InputFacesCheckBox.setChecked(
                window_state.get("InputFacesCheckBox", True)
            )
            main_window.JobsCheckBox.setChecked(window_state.get("JobsCheckBox", True))
            main_window.facesPanelCheckBox.setChecked(
                window_state.get("facesPanelCheckBox", True)
            )
            main_window.parametersPanelCheckBox.setChecked(
                window_state.get("parametersPanelCheckBox", True)
            )
            main_window.filterImagesCheckBox.setChecked(
                window_state.get("filterImagesCheckBox", True)
            )
            main_window.filterVideosCheckBox.setChecked(
                window_state.get("filterVideosCheckBox", True)
            )
            main_window.filterWebcamsCheckBox.setChecked(
                window_state.get("filterWebcamsCheckBox", False)
            )
            if hasattr(main_window, "scanToolsToggleButton"):
                video_control_actions.set_scan_tools_expanded(
                    main_window, window_state.get("scan_tools_expanded", False)
                )
            filter_actions.filter_target_videos(main_window)
            list_view_actions.load_target_webcams(main_window)

            # restore dock layout if it was saved
            dock_state_str = window_state.get("dock_state", data.get("dock_state", ""))
            if dock_state_str:
                try:
                    ba = QtCore.QByteArray.fromBase64(dock_state_str.encode("utf-8"))
                    main_window.restoreState(ba)
                except Exception as e:
                    print(f"[WARN] Failed to restore dock layout: {e}")
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            QtWidgets.QMessageBox.critical(
                main_window, "Error", f"Failed to load workspace: {e}"
            )
            return


def save_current_workspace(
    main_window: "MainWindow", data_filename: str | bool = False
):
    target_faces_data = {}
    embeddings_data = {}
    input_faces_data = {}
    target_medias_data = []

    # --- Save Window State ---
    # --- Save dock layout / panel sizes ---
    try:
        # saveState returns QByteArray; convert to base64 string for json compatibility
        dock_state_data = main_window.saveState().toBase64().data().decode("utf-8")
    except Exception:
        dock_state_data = ""

    window_state_data = {
        "x": main_window.x(),
        "y": main_window.y(),
        "height": main_window.height(),
        "width": main_window.width(),
        "isMaximized": main_window.isMaximized(),
        "isFullScreen": main_window.is_full_screen,
        "TargetMediaCheckBox": main_window.TargetMediaCheckBox.isChecked(),
        "InputFacesCheckBox": main_window.InputFacesCheckBox.isChecked(),
        "JobsCheckBox": main_window.JobsCheckBox.isChecked(),
        "facesPanelCheckBox": main_window.facesPanelCheckBox.isChecked(),
        "parametersPanelCheckBox": main_window.parametersPanelCheckBox.isChecked(),
        "filterImagesCheckBox": main_window.filterImagesCheckBox.isChecked(),
        "filterVideosCheckBox": main_window.filterVideosCheckBox.isChecked(),
        "filterWebcamsCheckBox": main_window.filterWebcamsCheckBox.isChecked(),
        "scan_tools_expanded": getattr(main_window, "scan_tools_expanded", False),
        "dock_state": dock_state_data,
    }

    # --- Check if Denoiser is enabled ---
    control = main_window.control
    is_denoiser_enabled = (
        control.get("DenoiserUNetEnableBeforeRestorersToggle", False)
        or control.get("DenoiserAfterFirstRestorerToggle", False)
        or control.get("DenoiserAfterRestorersToggle", False)
    )

    # --- Serialize Target Medias ---
    for media_id, target_media in main_window.target_videos.items():
        target_medias_data.append(
            {
                "media_path": target_media.media_path,
                "file_type": target_media.file_type,
                "media_id": media_id,
                "is_webcam": target_media.is_webcam,
                "webcam_index": target_media.webcam_index,
                "webcam_backend": target_media.webcam_backend,
            }
        )

    # --- Serialize Input Faces ---
    for face_id, input_face in main_window.input_faces.items():
        kv_map_path = None
        if is_denoiser_enabled and getattr(input_face, "kv_map", None) is not None:
            # Use Pathlib
            kv_data_dir = (
                main_window.project_root_path / "model_assets" / "reference_kv_data"
            )
            kv_data_dir.mkdir(parents=True, exist_ok=True)
            kv_map_file = kv_data_dir / f"input_{input_face.face_id}.pt"
            try:
                payload = {"kv_map": input_face.kv_map}
                torch.save(payload, str(kv_map_file))
                kv_map_path = str(kv_map_file)
            except Exception as e:
                print(
                    f"[ERROR] Error saving K/V map for input face {input_face.face_id} to {kv_map_file}: {e}"
                )
        input_faces_data[face_id] = {
            "media_path": input_face.media_path,
            "kv_map": kv_map_path,
        }

    # --- Serialize Target Faces & Parameters ---
    for face_id, target_face in main_window.target_faces.items():
        assigned_kv_map_serializable = None
        target_faces_data[face_id] = {
            "cropped_face": target_face.cropped_face.tolist(),
            "embedding_store": {
                embed_model: embedding.tolist()
                for embed_model, embedding in target_face.embedding_store.items()
            },
            "parameters": main_window.parameters.get(
                str(face_id), main_window.default_parameters
            ).data.copy(),  # Use .get with default, ensure it's dict
            "assigned_input_faces": list(target_face.assigned_input_faces.keys()),
            "assigned_merged_embeddings": list(
                target_face.assigned_merged_embeddings.keys()
            ),
            "assigned_input_embedding": {
                model: emb.tolist()
                for model, emb in target_face.assigned_input_embedding.items()
            },
            "assigned_kv_map": assigned_kv_map_serializable,
        }

    # --- Serialize Embeddings ---
    for embedding_id, embedding_button in main_window.merged_embeddings.items():
        embeddings_data[embedding_id] = {
            "embedding_name": embedding_button.embedding_name,
            "embedding_store": {
                model: emb.tolist()
                for model, emb in embedding_button.embedding_store.items()
            },
        }
    # --- Serialize Markers ---
    # Convert Parameters inside the markers from ParametersDict to dict before saving
    markers_to_save = convert_markers_to_supported_type(
        main_window, copy.deepcopy(main_window.markers), dict
    )

    # Save tab order - store the current tab index and the tab order
    tab_state = {
        "current_tab_index": main_window.tabWidget.currentIndex(),
        "tab_order": [],
    }

    # Store the tab order by getting the tab text for each position
    for i in range(main_window.tabWidget.count()):
        tab_state["tab_order"].append(
            {"text": main_window.tabWidget.tabText(i), "original_index": i}
        )

    # --- Prepare Workspace Data ---
    current_params_to_save = {}
    if isinstance(main_window.current_widget_parameters, misc_helpers.ParametersDict):
        # If it's the expected custom class, get its underlying data dictionary
        current_params_to_save = main_window.current_widget_parameters.data.copy()
    elif isinstance(main_window.current_widget_parameters, dict):
        # If it's already a dictionary (the unexpected case), just copy it
        current_params_to_save = main_window.current_widget_parameters.copy()
    else:
        # Fallback for safety, log a warning
        print(
            f"[WARN] Unexpected type for current widget parameters: {type(main_window.current_widget_parameters)}. Saving empty dict."
        )

    data = {
        "control": main_window.control.copy(),
        "target_medias_data": target_medias_data,
        "selected_media_id": main_window.selected_video_button.media_id
        if isinstance(
            main_window.selected_video_button, widget_components.TargetMediaCardButton
        )
        else False,
        "input_faces_data": input_faces_data,
        "target_faces_data": target_faces_data,
        "embeddings_data": embeddings_data,
        "markers": markers_to_save,
        "issue_frames_by_face": {
            str(face_id): sorted(frames)
            for face_id, frames in main_window.issue_frames_by_face.items()
        },
        "dropped_frames": sorted(main_window.dropped_frames),
        "job_marker_pairs": main_window.job_marker_pairs,  # Save the list of tuples
        "last_target_media_folder_path": main_window.last_target_media_folder_path,
        "last_input_media_folder_path": main_window.last_input_media_folder_path,
        "loaded_embedding_filename": main_window.loaded_embedding_filename,
        "current_widget_parameters": current_params_to_save,  # Use the safely prepared dict
        "tab_state": tab_state,  # Add the tab state to the saved data
        "window_state_data": window_state_data,
    }
    if data_filename is False:
        data_filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            main_window, filter="JSON (*.json)"
        )

    if data_filename:
        try:
            with open(data_filename, "w") as data_file:  # pylint: disable=unspecified-encoding
                data_as_json = json.dumps(
                    data, indent=4
                )  # Save with indentation for readability
                data_file.write(data_as_json)
            if isinstance(data_filename, str) and data_filename.endswith(
                "last_workspace.json"
            ):
                print(f"[INFO] Last workspace saved to: {data_filename}")
            else:
                common_widget_actions.create_and_show_toast_message(
                    main_window,
                    "Workspace Saved",
                    f"Saved Workspace to file: {data_filename}",
                )
        except Exception as e:
            print(f"[ERROR] Failed to save workspace {data_filename}: {e}")
            if not (
                isinstance(data_filename, str)
                and data_filename.endswith("last_workspace.json")
            ):  # Don't show error for auto-save
                common_widget_actions.create_and_show_messagebox(
                    main_window,
                    "Save Error",
                    f"Failed to save workspace:\\n{e}",
                    main_window,
                )


def save_current_job(main_window: "MainWindow"):
    # Check for necessary conditions
    if not main_window.selected_video_button:
        common_widget_actions.create_and_show_messagebox(
            main_window, "Error", "No target video selected.", main_window
        )
        return
    if not main_window.target_faces:
        common_widget_actions.create_and_show_messagebox(
            main_window, "Error", "No target faces detected or assigned.", main_window
        )
        return

    # Check on Dict to prevent crash
    if not any(
        len(tf.assigned_input_faces) > 0 for tf in main_window.target_faces.values()
    ):
        common_widget_actions.create_and_show_messagebox(
            main_window,
            "Error",
            "No input faces assigned to any target face.",
            main_window,
        )
        return

    # Show dialog to get job name and output options
    dialog = widget_components.SaveJobDialog(main_window)
    if dialog.exec() == QtWidgets.QDialog.Accepted:
        job_name = dialog.job_name
        use_job_name = dialog.use_job_name_for_output
        output_filename = dialog.output_file_name
        if not job_name:
            common_widget_actions.create_and_show_messagebox(
                main_window, "Error", "Job name cannot be empty.", main_window
            )
            return
    else:
        return  # User cancelled

    # --- Check if Denoiser is enabled ---
    control = main_window.control
    is_denoiser_enabled = (
        control.get("DenoiserUNetEnableBeforeRestorersToggle", False)
        or control.get("DenoiserAfterFirstRestorerToggle", False)
        or control.get("DenoiserAfterRestorersToggle", False)
    )

    # --- Serialize Input Faces ---
    input_faces_data = {}
    for face_id, input_face in main_window.input_faces.items():
        kv_map_path = None
        if is_denoiser_enabled and getattr(input_face, "kv_map", None) is not None:
            # Use Pathlib
            kv_data_dir = (
                main_window.project_root_path / "model_assets" / "reference_kv_data"
            )
            kv_data_dir.mkdir(parents=True, exist_ok=True)
            kv_map_file = kv_data_dir / f"input_{input_face.face_id}.pt"
            try:
                payload = {"kv_map": input_face.kv_map}
                torch.save(payload, str(kv_map_file))
                kv_map_path = str(kv_map_file)
            except Exception as e:
                print(
                    f"[ERROR] Error saving K/V map for input face {input_face.face_id} to {kv_map_file}: {e}"
                )
        input_faces_data[face_id] = {
            "media_path": input_face.media_path,
            "kv_map": kv_map_path,
        }

    # Prepare job data
    job_data = {
        "job_name": job_name,
        "use_job_name_for_output": use_job_name,
        "output_file_name": output_filename,
        "target_media_path": main_window.selected_video_button.media_path
        if isinstance(
            main_window.selected_video_button, widget_components.TargetMediaCardButton
        )
        else None,
        "target_media_id": main_window.selected_video_button.media_id
        if isinstance(
            main_window.selected_video_button, widget_components.TargetMediaCardButton
        )
        else None,
        "target_media_type": main_window.selected_video_button.file_type
        if isinstance(
            main_window.selected_video_button, widget_components.TargetMediaCardButton
        )
        else None,
        "input_faces_data": input_faces_data,
        "target_faces_data": {},
        "embeddings_data": {
            eid: {
                "name": emb.embedding_name,
                "store": {m: e.tolist() for m, e in emb.embedding_store.items()},
            }
            for eid, emb in main_window.merged_embeddings.items()
        },
        "markers": convert_markers_to_supported_type(
            main_window, copy.deepcopy(main_window.markers), dict
        ),
        "issue_frames_by_face": {
            str(face_id): sorted(frames)
            for face_id, frames in main_window.issue_frames_by_face.items()
        },
        "dropped_frames": sorted(main_window.dropped_frames),
        "control": main_window.control.copy(),
        "job_marker_pairs": main_window.job_marker_pairs,
        "scan_tools_expanded": getattr(main_window, "scan_tools_expanded", False),
        "current_widget_parameters": main_window.current_widget_parameters.data.copy()
        if isinstance(
            main_window.current_widget_parameters, misc_helpers.ParametersDict
        )
        else main_window.current_widget_parameters.copy(),
        "last_target_media_folder_path": main_window.last_target_media_folder_path,
        "last_input_media_folder_path": main_window.last_input_media_folder_path,
    }

    # Serialize target face specifics for the job
    for face_id, target_face in main_window.target_faces.items():
        # Security to handle either custom ParametersDict or native dict
        params_source = main_window.parameters.get(
            str(face_id), main_window.default_parameters
        )
        params_to_save = (
            params_source.data.copy()
            if isinstance(params_source, misc_helpers.ParametersDict)
            else params_source.copy()
        )

        job_data["target_faces_data"][face_id] = {
            "cropped_face": target_face.cropped_face.tolist(),
            "embedding_store": {
                m: e.tolist() for m, e in target_face.embedding_store.items()
            },
            "parameters": params_to_save,
            "assigned_input_faces": list(target_face.assigned_input_faces.keys()),
            "assigned_merged_embeddings": list(
                target_face.assigned_merged_embeddings.keys()
            ),
        }

    # Use pathlib
    jobs_dir = main_window.project_root_path / ".jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    save_path = jobs_dir / f"{job_name}.json"

    # Save the job file
    try:
        with open(save_path, "w") as f:
            json.dump(job_data, f, indent=4)
        common_widget_actions.create_and_show_toast_message(
            main_window, "Job Saved", f"Job '{job_name}' saved successfully."
        )
        # Refresh the Job Manager list if it's visible
        if hasattr(main_window, "jobManagerList"):
            main_window.jobManagerList.refresh_jobs()
    except Exception as e:
        print(f"[ERROR] Failed to save job '{job_name}': {e}")
        common_widget_actions.create_and_show_messagebox(
            main_window, "Save Job Error", f"Failed to save job:\n{e}", main_window
        )
