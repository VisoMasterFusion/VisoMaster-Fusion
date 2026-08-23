import os
import json
import hashlib
import io
import threading
from pathlib import Path
import uuid
import copy
from functools import partial
from typing import TYPE_CHECKING, Dict, Union, cast, Optional, List

from PySide6 import QtWidgets, QtCore
import numpy as np
import torch

from app.ui.widgets import widget_components
from app.ui.widgets.actions import common_actions as common_widget_actions
from app.ui.widgets.actions import card_actions
from app.ui.widgets.actions import list_view_actions
from app.ui.widgets.actions import video_control_actions
from app.ui.widgets.actions import control_actions
from app.ui.widgets.actions import layout_actions
from app.ui.widgets.actions import filter_actions
from app.ui.widgets import ui_workers
from app.helpers.typing_helper import ParametersTypes, MarkerTypes
import app.helpers.miscellaneous as misc_helpers

if TYPE_CHECKING:
    from app.ui.main_ui import MainWindow

# Global lock for registry I/O to ensure thread safety across concurrent saves
_KV_REGISTRY_LOCK = threading.Lock()


def _load_registry(registry_path: Path) -> Dict[str, List[str]]:
    """
    Loads the KV map registry safely.
    Returns an empty dictionary if the file is missing or corrupted.
    """
    if not registry_path.exists():
        return {}
    try:
        with open(registry_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"[WARN] Failed to read KV registry (might be corrupted), resetting: {e}")
        return {}


def _update_registry(
    main_window: "MainWindow",
    kv_filename: str,
    parent_file_paths: Union[str, List[str]],
    sub_folder: str = "reference_kv_data",
) -> None:
    """
    Thread-safe function to update the central KV registry.
    Adds one or more parent file paths (workspaces, jobs, or source image paths)
    to the list of dependents for the specified KV map.
    """
    if not parent_file_paths:
        return

    if isinstance(parent_file_paths, str):
        parent_file_paths = [parent_file_paths]

    registry_path = (
        main_window.project_root_path / "model_assets" / sub_folder / "kv_registry.json"
    )

    with _KV_REGISTRY_LOCK:
        registry = _load_registry(registry_path)

        # Initialize the list for this KV map if it doesn't exist
        if kv_filename not in registry:
            registry[kv_filename] = []

        registry_modified = False
        for path in parent_file_paths:
            if not path:
                continue
            # Normalize path separators for consistency across OS (e.g., resolving \ vs /)
            normalized_path = str(Path(path).resolve())
            if normalized_path not in registry[kv_filename]:
                registry[kv_filename].append(normalized_path)
                registry_modified = True

        # Write back to disk only if changes were made
        if registry_modified:
            try:
                with open(registry_path, "w", encoding="utf-8") as f:
                    json.dump(registry, f, indent=4)
            except IOError as e:
                print(f"[ERROR] Failed to write KV registry to {registry_path}: {e}")


def _save_hashed_kv_payload(
    main_window: "MainWindow",
    payload: dict,
    sub_folder: str = "reference_kv_data",
    parent_file_paths: Optional[Union[str, List[str]]] = None,
) -> Optional[str]:
    """
    Serializes a K/V map payload, computes its SHA-256 hash, and saves it to disk.
    Prevents duplicates by only saving if the hash-based file doesn't exist.
    Registers multiple dependencies (e.g., workspace JSON and source image) in a central JSON manifest.
    """
    try:
        # 1. Serialize payload to memory to compute the hash safely without disk I/O
        buffer = io.BytesIO()
        torch.save(payload, buffer)
        buffer_bytes = buffer.getvalue()

        # 2. Compute SHA-256 hash
        payload_hash = hashlib.sha256(buffer_bytes).hexdigest()

        # 3. Define directory and file path
        kv_data_dir = main_window.project_root_path / "model_assets" / sub_folder
        kv_data_dir.mkdir(parents=True, exist_ok=True)

        kv_map_file = kv_data_dir / f"kv_{payload_hash}.pt"

        # 4. Only write to disk if the file doesn't already exist (Deduplication)
        if not kv_map_file.exists():
            with open(kv_map_file, "wb") as f:
                f.write(buffer_bytes)
            print(f"[INFO] Saved new K/V map: {kv_map_file.name}")
        else:
            print(
                f"[INFO] K/V map {kv_map_file.name} already exists. Skipping disk write."
            )

        # 5. Update the central registry (Reference Counting)
        if parent_file_paths:
            _update_registry(
                main_window, kv_map_file.name, parent_file_paths, sub_folder
            )

        return str(kv_map_file)

    except Exception as e:
        print(f"[ERROR] Failed to hash and save K/V map payload: {e}")
        return None


def sanitize_markers_dictionary(
    main_window: "MainWindow", markers: dict | None
) -> dict:
    """
    Deep-sanitizes the markers dictionary, ensuring both 'control' and 'parameters'
    for every marker conform strictly to the current active schemas.
    """
    if not markers:
        return {}

    sanitized_markers = {}
    default_control = main_window.control
    default_params = cast(
        misc_helpers.ParametersDict, main_window.default_parameters
    ).data

    for marker_position, marker_data in markers.items():
        marker_payload = copy.deepcopy(marker_data)

        # 1. Sanitize Control Settings
        marker_payload["control"] = sanitize_state_dictionary(
            marker_payload.get("control", {}), default_control
        )

        # 2. Sanitize Parameters (mapped by face_id)
        if "parameters" in marker_payload:
            for face_id, face_params in marker_payload["parameters"].items():
                # Extract dictionary if it's wrapped in ParametersDict
                face_params_dict = (
                    face_params.data if hasattr(face_params, "data") else face_params
                )
                marker_payload["parameters"][face_id] = sanitize_state_dictionary(
                    face_params_dict, default_params
                )

        sanitized_markers[marker_position] = marker_payload

    return sanitized_markers


def sanitize_state_dictionary(loaded_dict: dict, reference_dict: dict) -> dict:
    """
    Sanitizes a loaded dictionary (parameters or controls) against a reference dictionary.
    Removes deprecated keys and initializes missing keys with default values.
    Ensures safe injection into PySide6 UI and CUDA/TensorRT worker pipelines.
    """
    sanitized_dict = {}
    if not isinstance(loaded_dict, dict):
        return copy.deepcopy(reference_dict)

    # 1. Inject defaults for missing keys and retain valid loaded keys
    for key, default_value in reference_dict.items():
        if key in loaded_dict:
            sanitized_dict[key] = loaded_dict[key]
        else:
            sanitized_dict[key] = default_value

    # 2. Identify and log keys that exist in the loaded file but are no longer in the app
    obsolete_keys = set(loaded_dict.keys()) - set(reference_dict.keys())
    if obsolete_keys:
        print(
            f"[INFO] Sanitization: Removed obsolete keys during load: {obsolete_keys}"
        )

    return sanitized_dict


def _get_clamped_window_geometry(
    main_window: "MainWindow", x: int, y: int, width: int, height: int
) -> QtCore.QRect:
    app = QtWidgets.QApplication.instance()
    screens = app.screens() if app else []
    if not screens:
        return QtCore.QRect(x, y, width, height)

    saved_rect = QtCore.QRect(x, y, width, height)
    saved_center = saved_rect.center()

    target_screen = None
    for screen in screens:
        if screen.availableGeometry().contains(saved_center):
            target_screen = screen
            break

    if target_screen is None:
        target_screen = app.primaryScreen() or screens[0]

    available = target_screen.availableGeometry()
    clamped_width = min(max(1, width), available.width())
    clamped_height = min(max(1, height), available.height())
    max_x = available.x() + available.width() - clamped_width
    max_y = available.y() + available.height() - clamped_height
    clamped_x = min(max(x, available.x()), max_x)
    clamped_y = min(max(y, available.y()), max_y)

    return QtCore.QRect(clamped_x, clamped_y, clamped_width, clamped_height)


def _get_target_screen_for_rect(rect: QtCore.QRect):
    app = QtWidgets.QApplication.instance()
    screens = app.screens() if app else []
    if not screens:
        return None

    rect_center = rect.center()
    for screen in screens:
        if screen.availableGeometry().contains(rect_center):
            return screen

    return app.primaryScreen() or screens[0]


def _clamp_window_frame_to_available_geometry(main_window: "MainWindow"):
    frame_rect = main_window.frameGeometry()
    if not frame_rect.isValid():
        return

    target_screen = _get_target_screen_for_rect(frame_rect)
    if target_screen is None:
        return

    available = target_screen.availableGeometry()
    if (
        frame_rect.width() > available.width()
        or frame_rect.height() > available.height()
    ):
        excess_width = max(frame_rect.width() - available.width(), 0)
        excess_height = max(frame_rect.height() - available.height(), 0)
        new_width = max(1, main_window.width() - excess_width)
        new_height = max(1, main_window.height() - excess_height)
        if new_width != main_window.width() or new_height != main_window.height():
            main_window.resize(new_width, new_height)
            frame_rect = main_window.frameGeometry()

    max_x = available.x() + available.width() - frame_rect.width()
    max_y = available.y() + available.height() - frame_rect.height()
    clamped_x = min(max(frame_rect.x(), available.x()), max_x)
    clamped_y = min(max(frame_rect.y(), available.y()), max_y)

    if clamped_x != frame_rect.x() or clamped_y != frame_rect.y():
        main_window.move(clamped_x, clamped_y)


def _apply_workspace_window_state(
    main_window: "MainWindow", window_state: dict
) -> bool:
    is_maximized = window_state.get("isMaximized", False)
    is_fullscreen = window_state.get("isFullScreen", False)

    main_window._fullscreen_restore_was_maximized = False
    main_window._fullscreen_restore_geometry = None

    if is_maximized:
        main_window.resize(main_window.sizeHint())
        main_window.showMaximized()
        main_window.menuBar().show()
        main_window.is_full_screen = False
        return False

    restored_rect = _get_clamped_window_geometry(
        main_window,
        window_state.get("x", main_window.x()),
        window_state.get("y", main_window.y()),
        window_state.get("width", main_window.width()),
        window_state.get("height", main_window.height()),
    )

    if is_fullscreen:
        main_window._fullscreen_restore_was_maximized = False
        main_window._fullscreen_restore_geometry = restored_rect
        main_window.resize(main_window.sizeHint())
        main_window.showFullScreen()
        main_window.is_full_screen = True
        return False

    main_window.setGeometry(restored_rect)
    main_window.menuBar().show()
    main_window.is_full_screen = False
    return True


def open_embeddings_from_file(main_window: "MainWindow"):
    if video_control_actions.block_if_issue_scan_active(main_window, "load embeddings"):
        return

    embedding_filename, _ = QtWidgets.QFileDialog.getOpenFileName(
        main_window,
        filter="JSON (*.json)",
        dir=misc_helpers.get_dir_of_file(main_window.loaded_embedding_filename),
    )
    if embedding_filename:
        try:
            with open(embedding_filename, "r", encoding="utf-8") as embed_file:
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

                    embedding_id = str(uuid.uuid1().int)

                    # Pass the entire embedding_store to the function
                    list_view_actions.create_and_add_embed_button_to_list(
                        main_window,
                        embed_data["name"],
                        embedding_store,
                        embedding_id=embedding_id,
                    )

                    # Restore KV map if it exists
                    if embedding_id in main_window.merged_embeddings:
                        embed_button = main_window.merged_embeddings[embedding_id]
                        kv_map_path = embed_data.get("kv_map")
                        if kv_map_path and os.path.exists(kv_map_path):
                            try:
                                payload = torch.load(
                                    kv_map_path, map_location="cpu", weights_only=False
                                )
                                if isinstance(payload, dict):
                                    # Check for new list format first, fallback to legacy dict format
                                    if "kv_map_list" in payload:
                                        embed_button.kv_map_list = payload[
                                            "kv_map_list"
                                        ]
                                    else:
                                        embed_button.kv_map = payload.get("kv_map")
                                else:
                                    embed_button.kv_map = payload
                                print(
                                    f"[INFO] Restored standalone K/V map for imported embedding: {embed_data['name']}"
                                )
                                # SELF-HEALING REGISTRY: Log this dependency on load
                                _update_registry(
                                    main_window,
                                    Path(kv_map_path).name,
                                    str(embedding_filename),
                                    "reference_kv_data",
                                )
                            except Exception as e:
                                print(
                                    f"[ERROR] Error loading K/V map for imported embedding from {kv_map_path}: {e}"
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

    # Define the save filename at the start so we can register dependencies
    embedding_filename = main_window.loaded_embedding_filename
    if (
        not embedding_filename
        or not misc_helpers.is_file_exists(embedding_filename)
        or save_as
    ):
        embedding_filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            main_window, filter="JSON (*.json)"
        )
        if not embedding_filename:
            return  # User cancelled the save
    elif (
        QtWidgets.QMessageBox.question(
            main_window,
            "Confirm Save",
            (
                "Save all embeddings to the current file?\n\n"
                f"This will overwrite:\n{embedding_filename}"
            ),
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        != QtWidgets.QMessageBox.Yes
    ):
        return

    # Build a list of dicts, each containing the embedding name, embedding_store, and kv_map path
    embeddings_list = []
    for embedding_id, embed_button in main_window.merged_embeddings.items():
        kv_map_path = None
        # Check for either the new list format or the legacy map format
        kv_payload_to_save = getattr(embed_button, "kv_map_list", None) or getattr(
            embed_button, "kv_map", None
        )

        if kv_payload_to_save is not None:
            try:
                # Save whatever payload we found (either a List[Dict] or a Dict)
                payload = (
                    {"kv_map_list": kv_payload_to_save}
                    if getattr(embed_button, "kv_map_list", None)
                    else {"kv_map": kv_payload_to_save}
                )
                # Optimize: Route through hashing helper to deduplicate and register
                kv_map_path = _save_hashed_kv_payload(
                    main_window,
                    payload,
                    "reference_kv_data",
                    parent_file_paths=str(embedding_filename),
                )
            except Exception as e:
                print(
                    f"[ERROR] Error saving hashed K/V map for embedding {embedding_id}: {e}"
                )

        embeddings_list.append(
            {
                "name": embed_button.embedding_name,
                "embedding_store": {
                    k: v.tolist() for k, v in embed_button.embedding_store.items()
                },  # Convert embeddings to lists
                "kv_map": kv_map_path,  # Save the path to JSON file
            }
        )

    # Save to file
    if embedding_filename:
        with open(embedding_filename, "w", encoding="utf-8") as embed_file:
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

    default_params = cast(
        misc_helpers.ParametersDict, main_window.default_parameters
    ).data
    raw_params = convert_parameters_to_supported_type(
        main_window, main_window.parameters[face_id], dict
    )

    data = {
        "parameters": sanitize_state_dictionary(raw_params, default_params),
        "control": sanitize_state_dictionary(
            main_window.control.copy(), main_window.control
        ),
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
    if video_control_actions.block_if_issue_scan_active(
        main_window,
        "load parameters and settings" if load_settings else "load parameters",
    ):
        return

    data_filename, _ = QtWidgets.QFileDialog.getOpenFileName(
        main_window, filter="JSON (*.json)"
    )
    if data_filename:
        with open(data_filename, "r") as data_file:  # pylint: disable=unspecified-encoding
            data = json.load(data_file)

            # --- Sanitize Parameters ---
            default_params = cast(
                misc_helpers.ParametersDict, main_window.default_parameters
            ).data
            sanitized_params = sanitize_state_dictionary(
                data.get("parameters", {}), default_params
            )

            main_window.parameters[face_id] = convert_parameters_to_supported_type(
                main_window, sanitized_params, misc_helpers.ParametersDict
            )

            if main_window.selected_target_face_id == face_id:
                common_widget_actions.set_widgets_values_using_face_id_parameters(
                    main_window, face_id
                )
            if load_settings:
                # --- Sanitize Controls ---
                sanitized_control = sanitize_state_dictionary(
                    data.get("control", {}), main_window.control
                )

                main_window.control.update(sanitized_control)
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
    if video_control_actions.block_if_issue_scan_active(
        main_window, "load a workspace"
    ):
        return

    if not data_filename:
        data_filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            main_window, filter="JSON (*.json)"
        )
    # Check if File exists (In cases when filename is passed as function argument instead of from the file picker)
    if not (isinstance(data_filename, str) and Path(data_filename).is_file()):
        data_filename = False
    if data_filename:
        with open(data_filename, "r", encoding="utf-8") as data_file:
            data = json.load(data_file)
        try:
            window_state = data.get("window_state_data", {})
            list_view_actions.clear_stop_loading_input_media(main_window)
            list_view_actions.clear_stop_loading_target_media(main_window)
            main_window.target_videos = {}
            card_actions.clear_input_faces(main_window)
            card_actions.clear_target_faces(main_window)
            card_actions.clear_merged_embeddings(main_window)

            # Load control (settings)
            control = sanitize_state_dictionary(
                data.get("control", {}), main_window.control
            )

            for control_name, control_value in control.items():
                main_window.control[control_name] = control_value

            # Add target medias
            target_medias_data = data.get("target_medias_data", [])
            target_medias_files_list = []
            target_media_ids = []
            for media_data in target_medias_data:
                target_medias_files_list.append(media_data["media_path"])
                target_media_ids.append(media_data["media_id"])

            from app.ui.widgets.actions import target_videos_list_actions

            main_window.video_loader_worker = ui_workers.TargetMediaLoaderWorker(
                main_window=main_window,
                folder_name=False,
                files_list=target_medias_files_list,
                media_ids=target_media_ids,
                sort_files_list_by_name=False,
                metadata_enabled=bool(
                    window_state.get("targetVideosFilterMenuExpanded", False)
                    or target_videos_list_actions.current_sort_needs_metadata(
                        main_window
                    )
                ),
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
            while list_view_actions._has_pending_target_media_thumbnail_work(
                main_window
            ):
                list_view_actions._flush_target_media_thumbnail_batch(main_window)
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
            main_window.input_faces_loader_worker.run()

            # Force PySide6 event loop to flush the queue.
            # This instantly populates `main_window.input_faces` before the next loop tries to access them.
            QtWidgets.QApplication.processEvents()

            # Drain remaining input-face batches, then hide placeholder if faces exist
            while getattr(main_window, "_pending_input_face_thumbnails", None):
                list_view_actions._flush_input_face_thumbnail_batch(main_window)
                QtWidgets.QApplication.processEvents()
            main_window.placeholder_update_signal.emit(
                main_window.inputFacesList, False
            )

            for face_id, input_face_data in data.get("input_faces_data", {}).items():
                if face_id in main_window.input_faces:
                    input_face_button = main_window.input_faces[face_id]
                    kv_map_path = input_face_data.get("kv_map")
                    media_path = input_face_data.get(
                        "media_path"
                    )  # Extract the image path

                    if kv_map_path and os.path.exists(kv_map_path):
                        try:
                            payload = torch.load(
                                kv_map_path, map_location="cpu", weights_only=False
                            )
                            if isinstance(payload, dict):
                                input_face_button.kv_map = payload.get("kv_map")
                            else:  # Backwards compatibility
                                input_face_button.kv_map = payload

                            # SELF-HEALING REGISTRY: Log dual dependencies on load (Workspace JSON + Image File)
                            parents_to_register = [str(data_filename)]
                            if media_path:
                                parents_to_register.append(str(media_path))

                            _update_registry(
                                main_window,
                                Path(kv_map_path).name,
                                parents_to_register,
                                "reference_kv_data",
                            )
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
                # Ancient job compatibility
                embedding_name = embedding_data.get(
                    "embedding_name", embedding_data.get("name", "Unknown")
                )

                # Embedding button creation
                list_view_actions.create_and_add_embed_button_to_list(
                    main_window,
                    embedding_name,
                    embedding_store_loaded,
                    embedding_id=embedding_id,
                )

                if embedding_id in main_window.merged_embeddings:
                    embed_button = main_window.merged_embeddings[embedding_id]
                    kv_map_path = embedding_data.get("kv_map")
                    if kv_map_path and os.path.exists(kv_map_path):
                        try:
                            # R-03: Safe standard weight loading, mapped to CPU initially to prevent VRAM fragmentation
                            payload = torch.load(
                                kv_map_path, map_location="cpu", weights_only=False
                            )

                            if isinstance(payload, dict):
                                # Check for the unbaked tensor list first
                                if "kv_map_list" in payload:
                                    embed_button.kv_map_list = payload["kv_map_list"]
                                else:
                                    # BACKWARD COMPATIBILITY: Fallback for older projects using a single baked dict
                                    embed_button.kv_map = payload.get("kv_map")
                            else:
                                # ANCIENT COMPATIBILITY: Direct payload fallback
                                embed_button.kv_map = payload

                            print(
                                f"[INFO] Restored K/V map data for embedding: {embedding_name}"
                            )
                            # SELF-HEALING REGISTRY: Log this dependency on load
                            _update_registry(
                                main_window,
                                Path(kv_map_path).name,
                                str(data_filename),
                                "reference_kv_data",
                            )
                        except Exception as e:
                            print(
                                f"[ERROR] Error loading K/V map for embedding from {kv_map_path}: {e}"
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

                # --- Sanitize Target Face Parameters ---
                raw_params = data["target_faces_data"][face_id].get("parameters", {})
                default_params = cast(
                    misc_helpers.ParametersDict, main_window.default_parameters
                ).data
                sanitized_params = sanitize_state_dictionary(raw_params, default_params)

                main_window.parameters[face_id] = convert_parameters_to_supported_type(
                    main_window,
                    sanitized_params,
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

            # --- Sanitize and Convert Markers ---
            raw_markers = data.get("markers", {})
            sanitized_markers_dict = sanitize_markers_dictionary(
                main_window, raw_markers
            )
            data["markers"] = convert_markers_to_supported_type(
                main_window,
                cast(MarkerTypes, sanitized_markers_dict),
                misc_helpers.ParametersDict,
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
            main_window.targetVideosPathLineEdit.setText(
                main_window.last_target_media_folder_path
            )
            main_window.targetVideosPathLineEdit.setToolTip(
                main_window.last_target_media_folder_path
            )
            main_window.inputFacesPathLineEdit.setText(
                main_window.last_input_media_folder_path
            )
            main_window.inputFacesPathLineEdit.setToolTip(
                main_window.last_input_media_folder_path
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
                saved_face_id = data.get("selected_target_face_id")
                first_face_id = (
                    saved_face_id
                    if saved_face_id in main_window.target_faces
                    else list(main_window.target_faces.keys())[0]
                )

                main_window.selected_target_face_id = first_face_id
                first_face_button = main_window.target_faces.get(first_face_id)

                if first_face_button:
                    first_face_button.setChecked(True)
                    main_window.cur_selected_target_face_button = first_face_button

                    for (
                        target_face_id,
                        target_face_button,
                    ) in main_window.target_faces.items():
                        if target_face_id != first_face_id:
                            target_face_button.setChecked(False)

                    card_actions.uncheck_all_input_faces(main_window)
                    card_actions.uncheck_all_merged_embeddings(main_window)

                    for input_face_id in first_face_button.assigned_input_faces.keys():
                        assigned_input_btn = main_window.input_faces.get(input_face_id)
                        if assigned_input_btn:
                            assigned_input_btn.setChecked(True)

                    for (
                        embedding_id
                    ) in first_face_button.assigned_merged_embeddings.keys():
                        assigned_embed_btn = main_window.merged_embeddings.get(
                            embedding_id
                        )
                        if assigned_embed_btn:
                            assigned_embed_btn.setChecked(True)

                    main_window.current_kv_tensors_map = getattr(
                        first_face_button, "assigned_kv_map", None
                    )

                video_control_actions.refresh_issue_frames_for_selected_face(
                    main_window
                )
                video_control_actions.update_scan_review_button_states(main_window)

                common_widget_actions.set_widgets_values_using_face_id_parameters(
                    main_window, face_id=first_face_id
                )
                main_window.current_widget_parameters = main_window.parameters[
                    first_face_id
                ].copy()
            else:
                raw_current_params = data.get(
                    "current_widget_parameters",
                    cast(
                        misc_helpers.ParametersDict, main_window.default_parameters
                    ).data.copy(),
                )
                default_params = cast(
                    misc_helpers.ParametersDict, main_window.default_parameters
                ).data
                sanitized_current_params = sanitize_state_dictionary(
                    raw_current_params, default_params
                )

                main_window.current_widget_parameters = cast(
                    ParametersTypes,
                    misc_helpers.ParametersDict(
                        sanitized_current_params,
                        default_params,
                    ),
                )
                common_widget_actions.set_widgets_values_using_face_id_parameters(
                    main_window, face_id=None
                )

            swap_faces_state = data.get("swap_faces_enabled", False)
            main_window.swapfacesButton.setChecked(swap_faces_state)

            edit_faces_state = data.get("edit_faces_enabled", False)
            main_window.editFacesButton.setChecked(edit_faces_state)
            control_actions.handle_face_editor_button_click(main_window)

            # Restore Window State
            needs_post_restore_frame_clamp = _apply_workspace_window_state(
                main_window, window_state
            )
            panel_state_map = {
                "target_media": window_state.get(
                    "target_media",
                    window_state.get("TargetMediaCheckBox", True),
                ),
                "input_faces": window_state.get(
                    "input_faces",
                    window_state.get("InputFacesCheckBox", True),
                ),
                "jobs": window_state.get(
                    "jobs",
                    window_state.get("JobsCheckBox", True),
                ),
                "faces": window_state.get(
                    "faces",
                    window_state.get("facesPanelCheckBox", True),
                ),
                "parameters": window_state.get(
                    "parameters",
                    window_state.get("parametersPanelCheckBox", True),
                ),
            }
            for panel_key, visible in panel_state_map.items():
                main_window._set_panel_visibility(panel_key, visible)

            def restore_checkbox_without_emitting_signals(checkbox, checked: bool):
                if hasattr(checkbox, "blockSignals"):
                    previous_state = checkbox.blockSignals(True)
                    try:
                        checkbox.setChecked(checked)
                    finally:
                        checkbox.blockSignals(previous_state)
                else:
                    checkbox.setChecked(checked)

            restore_checkbox_without_emitting_signals(
                main_window.targetVideosFilterImagesCheckBox,
                window_state.get("filterImagesCheckBox", True),
            )
            restore_checkbox_without_emitting_signals(
                main_window.targetVideosFilterVideosCheckBox,
                window_state.get("filterVideosCheckBox", True),
            )
            restore_checkbox_without_emitting_signals(
                main_window.targetVideosFilterWebcamsCheckBox,
                window_state.get("filterWebcamsCheckBox", False),
            )
            restore_checkbox_without_emitting_signals(
                main_window.targetVideosFilterMenuButton,
                window_state.get("targetVideosFilterMenuExpanded", False),
            )
            target_videos_list_actions.toggle_target_video_filters_sorting(main_window)
            saved_face_thumbnail_size = window_state.get("face_thumbnail_size")
            if saved_face_thumbnail_size == "small":
                list_view_actions.apply_face_thumbnail_size(
                    main_window, list_view_actions._SMALL_FACE_BUTTON_SIZE
                )
            elif saved_face_thumbnail_size == "large":
                list_view_actions.apply_face_thumbnail_size(
                    main_window, list_view_actions._LARGE_FACE_BUTTON_SIZE
                )
            else:
                list_view_actions.apply_face_thumbnail_size(
                    main_window, list_view_actions._FACE_BUTTON_SIZE
                )
            if hasattr(main_window, "scanToolsToggleButton"):
                video_control_actions.set_scan_tools_expanded(
                    main_window, window_state.get("scan_tools_expanded", False)
                )
            parameter_section_states = None
            if "parameter_section_states" in window_state:
                parameter_section_states = window_state["parameter_section_states"]
            elif "parameter_section_states" in data:
                parameter_section_states = data["parameter_section_states"]
            main_window.apply_parameter_section_states(parameter_section_states)
            filter_actions.filter_target_videos(main_window)
            list_view_actions.load_target_webcams(main_window)

            # restore dock layout if it was saved
            dock_state_str = window_state.get("dock_state", data.get("dock_state", ""))
            if dock_state_str:
                try:
                    ba = QtCore.QByteArray.fromBase64(dock_state_str.encode("utf-8"))
                    main_window.restoreState(ba)
                    main_window._refresh_panel_visibility_state_from_widgets()
                except Exception as e:
                    print(f"[WARN] Failed to restore dock layout: {e}")

            if needs_post_restore_frame_clamp:
                QtCore.QTimer.singleShot(
                    0,
                    partial(_clamp_window_frame_to_available_geometry, main_window),
                )
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            QtWidgets.QMessageBox.critical(
                main_window, "Error", f"Failed to load workspace: {e}"
            )
            return


def save_current_workspace(
    main_window: "MainWindow", data_filename: str | Path | bool = False
):
    # Prompt for filename FIRST so we can use it for the registry and avoid useless work if cancelled
    if data_filename is False:
        dialog_filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            main_window, filter="JSON (*.json)"
        )
        if not dialog_filename:
            return  # User cancelled the save operation
        data_filename = dialog_filename

    resolved_parent_path = str(data_filename)

    target_faces_data = {}
    embeddings_data = {}
    input_faces_data = {}
    target_medias_data = []

    # --- Save Window State ---
    # --- Save dock layout / panel sizes ---
    is_theatre_mode = bool(getattr(main_window, "is_theatre_mode", False))
    saved_is_fullscreen = bool(main_window.is_full_screen)
    saved_is_maximized = (
        bool(main_window.isMaximized()) if not saved_is_fullscreen else False
    )
    saved_geometry = main_window.geometry()
    dock_state_source = None
    if is_theatre_mode:
        saved_is_fullscreen = bool(
            getattr(main_window, "_was_custom_fullscreen", main_window.is_full_screen)
        )
        saved_is_maximized = (
            bool(getattr(main_window, "_was_maximized", main_window.isMaximized()))
            if not saved_is_fullscreen
            else False
        )
        saved_geometry = getattr(main_window, "_was_normal_geometry", saved_geometry)
        dock_state_source = getattr(main_window, "_saved_window_state", None)
    elif saved_is_fullscreen:
        restore_geometry = getattr(main_window, "_fullscreen_restore_geometry", None)
        if restore_geometry is not None:
            saved_geometry = restore_geometry

    try:
        # saveState returns QByteArray; convert to base64 string for json compatibility
        if dock_state_source is None:
            dock_state_source = main_window.saveState()
        dock_state_data = dock_state_source.toBase64().data().decode("utf-8")
    except Exception:
        dock_state_data = ""

    window_state_data = {
        "x": saved_geometry.x(),
        "y": saved_geometry.y(),
        "height": saved_geometry.height(),
        "width": saved_geometry.width(),
        "isMaximized": saved_is_maximized,
        "isFullScreen": saved_is_fullscreen,
        "target_media": main_window.panel_visibility_state.get("target_media", True),
        "input_faces": main_window.panel_visibility_state.get("input_faces", True),
        "jobs": main_window.panel_visibility_state.get("jobs", True),
        "faces": main_window.panel_visibility_state.get("faces", True),
        "parameters": main_window.panel_visibility_state.get("parameters", True),
        "filterImagesCheckBox": main_window.targetVideosFilterImagesCheckBox.isChecked(),
        "filterVideosCheckBox": main_window.targetVideosFilterVideosCheckBox.isChecked(),
        "filterWebcamsCheckBox": main_window.targetVideosFilterWebcamsCheckBox.isChecked(),
        "targetVideosFilterMenuExpanded": (
            main_window.targetVideosFilterMenuButton.isChecked()
        ),
        "face_thumbnail_size": (
            "small"
            if getattr(
                main_window,
                "face_thumbnail_button_size",
                list_view_actions._FACE_BUTTON_SIZE,
            )
            == list_view_actions._SMALL_FACE_BUTTON_SIZE
            else "large"
        ),
        "scan_tools_expanded": getattr(main_window, "scan_tools_expanded", False),
        "parameter_section_states": {
            section_id: bool(expanded)
            for section_id, expanded in getattr(
                main_window, "parameter_section_states", {}
            ).items()
        },
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
            try:
                payload = {"kv_map": input_face.kv_map}
                # Optimize: Route through hashing helper to deduplicate and register BOTH paths
                kv_map_path = _save_hashed_kv_payload(
                    main_window,
                    payload,
                    "reference_kv_data",
                    parent_file_paths=[
                        resolved_parent_path,
                        str(input_face.media_path),
                    ],
                )
            except Exception as e:
                print(
                    f"[ERROR] Error saving hashed K/V map for input face {input_face.face_id}: {e}"
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
        kv_map_path = None
        # Check for either the new list format or the legacy map format
        kv_payload_to_save = getattr(embedding_button, "kv_map_list", None) or getattr(
            embedding_button, "kv_map", None
        )

        if kv_payload_to_save is not None:
            try:
                # Save whatever payload we found
                payload = (
                    {"kv_map_list": kv_payload_to_save}
                    if getattr(embedding_button, "kv_map_list", None)
                    else {"kv_map": kv_payload_to_save}
                )
                # Optimize: Route through hashing helper to deduplicate and register
                kv_map_path = _save_hashed_kv_payload(
                    main_window,
                    payload,
                    "reference_kv_data",
                    parent_file_paths=resolved_parent_path,
                )
            except Exception as e:
                print(
                    f"[ERROR] Error saving hashed K/V map for embedding {embedding_id}: {e}"
                )

        embeddings_data[embedding_id] = {
            "embedding_name": embedding_button.embedding_name,
            "embedding_store": {
                model: emb.tolist()
                for model, emb in embedding_button.embedding_store.items()
            },
            "kv_map": kv_map_path,
        }

    # --- Serialize Markers ---
    # Convert Parameters inside the markers from ParametersDict to dict before saving
    markers_to_save = sanitize_markers_dictionary(
        main_window,
        convert_markers_to_supported_type(
            main_window, cast(MarkerTypes, copy.deepcopy(main_window.markers)), dict
        ),
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
        "control": sanitize_state_dictionary(
            main_window.control.copy(), main_window.control
        ),
        "target_medias_data": target_medias_data,
        "selected_media_id": main_window.selected_video_button.media_id
        if isinstance(
            main_window.selected_video_button, widget_components.TargetMediaCardButton
        )
        else False,
        "selected_target_face_id": getattr(
            main_window, "selected_target_face_id", None
        ),
        "swap_faces_enabled": main_window.swapfacesButton.isChecked(),
        "edit_faces_enabled": main_window.editFacesButton.isChecked(),
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

    if data_filename:
        try:
            with open(data_filename, "w", encoding="utf-8") as data_file:
                data_as_json = json.dumps(
                    data, indent=4
                )  # Save with indentation for readability
                data_file.write(data_as_json)
            if str(data_filename).endswith("last_workspace.json"):
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
                    f"Failed to save workspace:\n{e}",
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

    # Construct the intended job file path early for registry registration
    jobs_dir = main_window.project_root_path / ".jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    save_path = jobs_dir / f"{job_name}.json"
    resolved_parent_path = str(save_path)

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
            try:
                payload = {"kv_map": input_face.kv_map}
                # Optimize: Route through hashing helper to deduplicate and register BOTH paths
                kv_map_path = _save_hashed_kv_payload(
                    main_window,
                    payload,
                    "reference_kv_data",
                    parent_file_paths=[
                        resolved_parent_path,
                        str(input_face.media_path),
                    ],
                )
            except Exception as e:
                print(
                    f"[ERROR] Error saving hashed K/V map for input face {input_face.face_id}: {e}"
                )

        input_faces_data[face_id] = {
            "media_path": input_face.media_path,
            "kv_map": kv_map_path,
        }

    # --- Serialize Embeddings for Job ---
    embeddings_data = {}
    for eid, emb in main_window.merged_embeddings.items():
        kv_map_path = None
        # Check for either the new list format or the legacy map format
        kv_payload_to_save = getattr(emb, "kv_map_list", None) or getattr(
            emb, "kv_map", None
        )

        if kv_payload_to_save is not None:
            try:
                payload = (
                    {"kv_map_list": kv_payload_to_save}
                    if getattr(emb, "kv_map_list", None)
                    else {"kv_map": kv_payload_to_save}
                )
                # Optimize: Route through hashing helper to deduplicate and register
                kv_map_path = _save_hashed_kv_payload(
                    main_window,
                    payload,
                    "reference_kv_data",
                    parent_file_paths=resolved_parent_path,
                )
            except Exception as e:
                print(f"[ERROR] Error saving hashed K/V map for embedding {eid}: {e}")

        embeddings_data[eid] = {
            "name": emb.embedding_name,
            "store": {m: e.tolist() for m, e in emb.embedding_store.items()},
            "kv_map": kv_map_path,
        }

    # Prepare job data
    default_params = cast(
        misc_helpers.ParametersDict, main_window.default_parameters
    ).data
    raw_current_params = (
        main_window.current_widget_parameters.data.copy()
        if isinstance(
            main_window.current_widget_parameters, misc_helpers.ParametersDict
        )
        else main_window.current_widget_parameters.copy()
    )

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
        # --- Explicitly save button states for jobs ---
        "swap_faces_enabled": main_window.swapfacesButton.isChecked(),
        "edit_faces_enabled": main_window.editFacesButton.isChecked(),
        "input_faces_data": input_faces_data,
        "target_faces_data": {},
        "embeddings_data": embeddings_data,
        "markers": sanitize_markers_dictionary(
            main_window,
            convert_markers_to_supported_type(
                main_window, cast(MarkerTypes, copy.deepcopy(main_window.markers)), dict
            ),
        ),
        "issue_frames_by_face": {
            str(face_id): sorted(frames)
            for face_id, frames in main_window.issue_frames_by_face.items()
        },
        "dropped_frames": sorted(main_window.dropped_frames),
        "control": sanitize_state_dictionary(
            main_window.control.copy(), main_window.control
        ),
        "job_marker_pairs": main_window.job_marker_pairs,
        "scan_tools_expanded": getattr(main_window, "scan_tools_expanded", False),
        "current_widget_parameters": sanitize_state_dictionary(
            raw_current_params, default_params
        ),
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

        sanitized_params_to_save = sanitize_state_dictionary(
            params_to_save, default_params
        )

        job_data["target_faces_data"][face_id] = {
            "cropped_face": target_face.cropped_face.tolist(),
            "embedding_store": {
                m: e.tolist() for m, e in target_face.embedding_store.items()
            },
            "parameters": sanitized_params_to_save,
            "assigned_input_faces": list(target_face.assigned_input_faces.keys()),
            "assigned_merged_embeddings": list(
                target_face.assigned_merged_embeddings.keys()
            ),
        }

    # Save the job file
    try:
        with open(save_path, "w", encoding="utf-8") as f:
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


def purge_unused_kv_maps(main_window: "MainWindow"):
    """
    Scans the KV registry for missing parent files and orphaned .pt hashes.
    Prompts the user for verification before safely deleting unused tensors.
    """
    registry_path = (
        main_window.project_root_path
        / "model_assets"
        / "reference_kv_data"
        / "kv_registry.json"
    )
    kv_data_dir = main_window.project_root_path / "model_assets" / "reference_kv_data"

    # 1. Load registry safely, releasing lock immediately for UI interaction
    with _KV_REGISTRY_LOCK:
        registry = _load_registry(registry_path)

    # 2. Extract unique parent file paths
    unique_parent_paths = set()
    for paths in registry.values():
        unique_parent_paths.update(paths)

    # 3. Check for missing parent files
    missing_paths = [p for p in unique_parent_paths if not Path(p).exists()]
    paths_to_remove = set()

    for missing_path in missing_paths:
        reply = QtWidgets.QMessageBox.question(
            main_window,
            "Missing File Detected",
            f"The saved workspace or job file cannot be found:\n{missing_path}\n\n"
            "Have you deleted or moved this file?\n\n"
            "Click 'Yes' if it's gone and you want to clean up its unused data.\n"
            "Click 'No' to keep its K/V maps safely.",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        if reply == QtWidgets.QMessageBox.Yes:
            paths_to_remove.add(missing_path)

    # 4. Process Registry Removals and Registered File Deletions
    files_deleted_count = 0
    registry_modified = False
    keys_to_delete = []

    for kv_filename, paths in registry.items():
        original_len = len(paths)
        # Keep only paths that the user did NOT mark for removal
        registry[kv_filename] = [p for p in paths if p not in paths_to_remove]

        if len(registry[kv_filename]) != original_len:
            registry_modified = True

        # If the K/V map is no longer referenced by ANY valid workspace/job
        if not registry[kv_filename]:
            keys_to_delete.append(kv_filename)

    # Physically delete unreferenced hashed files
    for kv_filename in keys_to_delete:
        pt_path = kv_data_dir / kv_filename
        try:
            if pt_path.exists():
                pt_path.unlink()
                print(f"[INFO] Purged unused K/V map: {kv_filename}")
                files_deleted_count += 1
        except Exception as e:
            print(f"[ERROR] Failed to delete {kv_filename}: {e}")

        # Remove from registry map
        del registry[kv_filename]
        registry_modified = True

    # 5. Check for Legacy/Unregistered Files
    # This cleans up the hundreds of old files you generated before we added hashing
    unregistered_pt_files = []
    if kv_data_dir.exists():
        for pt_file in kv_data_dir.glob("*.pt"):
            if pt_file.name not in registry:
                unregistered_pt_files.append(pt_file)

    legacy_files_deleted = 0
    if unregistered_pt_files:
        reply = QtWidgets.QMessageBox.question(
            main_window,
            "Legacy K/V Maps Detected",
            f"Found {len(unregistered_pt_files)} old K/V map files that are not tracked in the current registry.\n\n"
            "Do you want to permanently delete them to free up disk space?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        if reply == QtWidgets.QMessageBox.Yes:
            for pt_file in unregistered_pt_files:
                try:
                    pt_file.unlink()
                    legacy_files_deleted += 1
                    print(f"[INFO] Purged legacy K/V map: {pt_file.name}")
                except Exception as e:
                    print(f"[ERROR] Failed to delete {pt_file.name}: {e}")

    # 6. Save Updated Registry Safely
    if registry_modified:
        with _KV_REGISTRY_LOCK:
            try:
                with open(registry_path, "w", encoding="utf-8") as f:
                    json.dump(registry, f, indent=4)
            except IOError as e:
                print(f"[ERROR] Failed to write updated KV registry: {e}")

    # 7. Final Notification
    total_deleted = files_deleted_count + legacy_files_deleted
    if total_deleted > 0:
        common_widget_actions.create_and_show_messagebox(
            main_window,
            "Storage Purge Complete",
            f"Successfully purged {total_deleted} unused K/V map(s).",
            main_window,
        )
    elif not missing_paths and not unregistered_pt_files:
        common_widget_actions.create_and_show_messagebox(
            main_window,
            "Storage Purge",
            "All tracked K/V maps are currently in use. Nothing to purge.",
            main_window,
        )
