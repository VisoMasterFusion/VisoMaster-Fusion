import json
from pathlib import Path
import copy
from functools import partial
from typing import TYPE_CHECKING
import os
import shutil
import time
from PySide6.QtCore import QThread, Signal, Slot, QMetaObject, Qt
from PySide6 import QtWidgets
from PySide6.QtWidgets import QMessageBox
import numpy as np
import threading
import re
import traceback  # Import traceback for error logging
from send2trash import send2trash

from app.ui.widgets.actions import common_actions as common_widget_actions
from app.ui.widgets.actions import card_actions
from app.ui.widgets.actions import list_view_actions
from app.ui.widgets.actions import video_control_actions
from app.ui.widgets.actions import layout_actions
from app.ui.widgets import ui_workers
from app.helpers.typing_helper import ParametersTypes, MarkerTypes
import app.helpers.miscellaneous as misc_helpers
from app.ui.widgets import widget_components

if TYPE_CHECKING:
    from app.ui.main_ui import MainWindow


# --- Constants ---

jobs_dir = os.path.join(os.getcwd(), "jobs")
os.makedirs(jobs_dir, exist_ok=True)  # Ensure the directory exists

# --- Parameter Conversion Helpers ---


def convert_parameters_to_job_type(
    main_window: "MainWindow", parameters: dict | ParametersTypes, convert_type: type
) -> dict | ParametersTypes:
    """
    Converts a parameter object to the specified type (dict or ParametersDict).
    Useful for JSON serialization (to dict) and deserialization (to ParametersDict).
    """
    if convert_type == dict:
        if isinstance(parameters, misc_helpers.ParametersDict):
            # Convert ParametersDict to a plain dict for JSON saving
            return parameters.data.copy()
        elif isinstance(parameters, dict):
            # Already a dict, return a copy
            return parameters.copy()
        else:
            print(
                f"[WARN] Unexpected type {type(parameters)} encountered when converting to dict."
            )
            return parameters
    elif convert_type == misc_helpers.ParametersDict:
        if not isinstance(parameters, misc_helpers.ParametersDict):
            if isinstance(parameters, dict):
                # Convert a plain dict back to ParametersDict after loading
                return misc_helpers.ParametersDict(
                    parameters, main_window.default_parameters
                )
            else:
                print(
                    f"[WARN] Unexpected type {type(parameters)} encountered when converting to ParametersDict."
                )
                return parameters
        else:
            # Already a ParametersDict, return it
            return parameters
    else:
        print(
            f"[WARN] Invalid convert_type {convert_type} specified in convert_parameters_to_job_type."
        )
        return parameters


def convert_markers_to_job_type(
    main_window: "MainWindow",
    markers: MarkerTypes,
    convert_type: dict | misc_helpers.ParametersDict,
) -> MarkerTypes:
    """
    Recursively converts parameter dictionaries within a markers object
    to the specified type (dict or ParametersDict).
    """
    for _, marker_data in markers.items():
        # Convert parameters for each face_id within the marker
        if "parameters" in marker_data and isinstance(marker_data["parameters"], dict):
            for target_face_id, target_parameters in marker_data["parameters"].items():
                marker_data["parameters"][target_face_id] = (
                    convert_parameters_to_job_type(
                        main_window, target_parameters, convert_type
                    )
                )

        # Also convert the control dict within the marker
        if "control" in marker_data:
            marker_data["control"] = convert_parameters_to_job_type(
                main_window, marker_data["control"], convert_type
            )
    return markers


# --- Job File Management ---


def save_job(
    main_window: "MainWindow",
    job_name: str,
    use_job_name_for_output: bool = True,
    output_file_name: str = None,
):
    """
    Saves the current workspace as a job JSON file in the 'jobs' directory.
    This is a wrapper for save_job_workspace.
    """
    try:
        data_filename = os.path.join(jobs_dir, f"{job_name}")
        save_job_workspace(
            main_window, data_filename, use_job_name_for_output, output_file_name
        )
        print(f"[DEBUG] Job saved: {data_filename}")
        common_widget_actions.create_and_show_toast_message(
            main_window, "Job Saved", f"Job '{job_name}' has been saved."
        )
    except Exception as e:
        print(f"[ERROR] Failed to save job: {e}")
        QMessageBox.critical(
            main_window, "Save Error", f"Failed to save job '{job_name}':\n{e}"
        )


def list_jobs() -> list[str]:
    """Lists all saved jobs (JSON files) from the 'jobs' directory."""
    if not os.path.exists(jobs_dir):
        return []
    # Return job names without the .json extension
    return [f.replace(".json", "") for f in os.listdir(jobs_dir) if f.endswith(".json")]


def delete_job(main_window: "MainWindow") -> bool:
    """
    Moves the selected job(s) from the 'jobs' directory to the system trash
    after user confirmation.
    """
    selected_jobs = get_selected_jobs(main_window)
    if not selected_jobs:
        QMessageBox.warning(
            main_window, "No Job Selected", "Please select one or more jobs to delete."
        )
        return False

    # Confirm deletion with the user
    confirm = QMessageBox.question(
        main_window,
        "Confirm Deletion",
        f"Are you sure you want to delete the selected job{'s' if len(selected_jobs) > 1 else ''}?\n\n"
        + ", ".join(selected_jobs)
        + "\n\n(Files will be moved to the Recycle Bin)",
        QMessageBox.Yes | QMessageBox.No,
    )
    if confirm != QMessageBox.Yes:
        return False

    deleted_any = False
    for job_name in selected_jobs:
        job_file = os.path.join(jobs_dir, f"{job_name}.json")
        if os.path.exists(job_file):
            try:
                send2trash(job_file)  # Use send2trash for safety
                print(f"[DEBUG] Job moved to trash: {job_file}")
                deleted_any = True
            except Exception as e:
                print(f"[ERROR] Failed to move job to trash: {e}")
                QMessageBox.warning(
                    main_window,
                    "Delete Error",
                    f"Could not move job '{job_name}' to trash:\n{e}",
                )
        else:
            print(f"[DEBUG] Job file not found for deletion: {job_file}")

    if deleted_any:
        refresh_job_list(main_window)  # Update the UI list
        common_widget_actions.create_and_show_toast_message(
            main_window, "Job(s) Deleted", "Selected jobs moved to Recycle Bin."
        )
        return True
    else:
        QMessageBox.warning(
            main_window, "Job(s) Not Found", "None of the selected jobs exist."
        )
        return False


def load_job(main_window: "MainWindow"):
    """
    Loads the currently selected job from the job list widget.
    Shows a warning if no job or multiple jobs are selected.

    This performs a FULL, HEAVY load of the workspace.
    """
    selected_jobs = get_selected_jobs(main_window)
    if not selected_jobs:
        QMessageBox.warning(
            main_window, "No Job Selected", "Please select a job from the list."
        )
        return
    if len(selected_jobs) > 1:
        QMessageBox.warning(
            main_window,
            "Multiple Jobs Selected",
            "You can only load one job at a time. Please select a single job to load.",
        )
        return

    job_name = selected_jobs[0]

    # Use the original, heavy load_job_workspace function
    print(f"[DEBUG] Performing full workspace load for job: {job_name}")
    load_job_workspace(main_window, job_name)


# --- Workspace (De)serialization ---


def _clear_main_window_state(main_window: "MainWindow"):
    """Resets the main window UI and data to a clean state before loading."""
    main_window.selected_video_button = None
    main_window.control["AutoSwapToggle"] = False

    list_view_actions.clear_stop_loading_input_media(main_window)
    list_view_actions.clear_stop_loading_target_media(main_window)
    main_window.target_videos = {}
    card_actions.clear_input_faces(main_window)
    card_actions.clear_target_faces(main_window)  # Also clears parameters
    card_actions.clear_merged_embeddings(main_window)
    video_control_actions.remove_all_markers(
        main_window
    )  # Clear markers from slider and data
    main_window.job_marker_pairs.clear()  # Clear job segments

    if hasattr(main_window, "selected_video_button"):
        btn = main_window.selected_video_button
        if btn and (
            not hasattr(btn, "media_id")
            or btn.media_id not in main_window.target_videos
        ):
            main_window.selected_video_button = None


def _load_job_target_media(main_window: "MainWindow", data: dict):
    """Loads target media files from the job data."""
    target_medias_data = data.get("target_medias_data", [])

    # Validate paths before loading
    valid_target_medias_data = []
    for m in target_medias_data:
        if "media_path" in m and os.path.exists(m["media_path"]):
            valid_target_medias_data.append(m)
        else:
            print(
                f"[WARN] Target media path not found, skipping: {m.get('media_path')}"
            )

    target_medias_files_list, target_media_ids = (
        zip(*[(m["media_path"], m["media_id"]) for m in valid_target_medias_data])
        if valid_target_medias_data
        else ([], [])
    )

    main_window.video_loader_worker = ui_workers.TargetMediaLoaderWorker(
        main_window=main_window,
        folder_name=False,
        files_list=target_medias_files_list,
        media_ids=target_media_ids,
    )
    main_window.video_loader_worker.thumbnail_ready.connect(
        partial(
            list_view_actions.add_media_thumbnail_to_target_videos_list, main_window
        )
    )
    # .run() is synchronous, ensuring media is loaded before proceeding.
    main_window.video_loader_worker.run()

    # Select the previously active media
    selected_media_id = data.get("selected_media_id", False)
    if selected_media_id and main_window.target_videos.get(selected_media_id):
        main_window.target_videos[selected_media_id].click()


def _load_job_input_faces(main_window: "MainWindow", data: dict):
    """Loads input face images from the job data."""
    input_faces_data = data.get("input_faces_data", {})

    valid_input_faces_data = {}
    for face_id, f in input_faces_data.items():
        if "media_path" in f and os.path.exists(f["media_path"]):
            valid_input_faces_data[face_id] = f
        else:
            print(
                f"[WARN] Input face media path not found, skipping: {f.get('media_path')}"
            )

    input_media_paths, input_face_ids = (
        zip(
            *[
                (f["media_path"], face_id)
                for face_id, f in valid_input_faces_data.items()
            ]
        )
        if valid_input_faces_data
        else ([], [])
    )

    # Create the worker, but DO NOT start it yet.
    main_window.input_faces_loader_worker = ui_workers.InputFacesLoaderWorker(
        main_window=main_window,
        folder_name=False,
        files_list=input_media_paths,
        face_ids=input_face_ids,
    )
    main_window.input_faces_loader_worker.thumbnail_ready.connect(
        partial(list_view_actions.add_media_thumbnail_to_source_faces_list, main_window)
    )
    # main_window.input_faces_loader_worker.finished.connect(
    #     partial(common_widget_actions.refresh_frame, main_window)
    # )
    # main_window.input_faces_loader_worker.files_list = list(
    #     main_window.input_faces_loader_worker.files_list
    # )
    # # .run() is synchronous, ensuring faces are loaded before proceeding.
    # main_window.input_faces_loader_worker.run()


def _load_job_embeddings(main_window: "MainWindow", data: dict):
    """Loads saved embeddings from the job data."""
    for embedding_id, embedding_data in data.get("embeddings_data", {}).items():
        # Convert list back to numpy array
        embedding_store = {
            embed_model: np.array(embed)
            for embed_model, embed in embedding_data["embedding_store"].items()
        }
        list_view_actions.create_and_add_embed_button_to_list(
            main_window,
            embedding_data["embedding_name"],
            embedding_store,
            embedding_id=embedding_id,
        )


def _load_job_target_faces_and_params(main_window: "MainWindow", data: dict):
    """Loads detected target faces, their parameters, and assignments."""
    loaded_target_faces_data = data.get("target_faces_data", {})
    for face_id_str, target_face_data in loaded_target_faces_data.items():
        face_id = int(face_id_str)
        # Convert list back to numpy array
        cropped_face = np.array(target_face_data["cropped_face"]).astype("uint8")
        pixmap = common_widget_actions.get_pixmap_from_frame(main_window, cropped_face)
        embedding_store = {
            embed_model: np.array(embed)
            for embed_model, embed in target_face_data["embedding_store"].items()
        }

        # Create the button and add it to the list/main_window dict
        list_view_actions.add_media_thumbnail_to_target_faces_list(
            main_window, cropped_face, embedding_store, pixmap, face_id
        )

        # Convert the loaded parameters dict into a ParametersDict object
        main_window.parameters[face_id] = convert_parameters_to_job_type(
            main_window,
            target_face_data.get("parameters", {}),
            misc_helpers.ParametersDict,
        )

        # Load assigned faces/embeddings into the created target_face object
        if face_id in main_window.target_faces:
            target_face_obj = main_window.target_faces[face_id]

            # Load assigned merged embeddings
            target_face_obj.assigned_merged_embeddings.clear()
            for assigned_id in target_face_data.get("assigned_merged_embeddings", []):
                if assigned_id in main_window.merged_embeddings:
                    target_face_obj.assigned_merged_embeddings[assigned_id] = (
                        main_window.merged_embeddings[assigned_id].embedding_store
                    )

            # Load assigned input faces
            target_face_obj.assigned_input_faces.clear()
            for assigned_id in target_face_data.get("assigned_input_faces", []):
                if assigned_id in main_window.input_faces:
                    target_face_obj.assigned_input_faces[assigned_id] = (
                        main_window.input_faces[assigned_id].embedding_store
                    )

            # Load pre-calculated assigned input embedding
            target_face_obj.assigned_input_embedding = {
                embed_model: np.array(embed)
                for embed_model, embed in target_face_data.get(
                    "assigned_input_embedding", {}
                ).items()
            }
        else:
            print(
                f"[WARN] Target face object with id {face_id} not found after creation."
            )


def _load_job_controls_and_state(
    main_window: "MainWindow", data: dict, is_batch_load: bool = False
):
    """Loads global control settings and misc UI state."""
    for control_name, control_value in data.get("control", {}).items():
        main_window.control[control_name] = control_value
    # Ensure AutoSwap is off after loading a job
    main_window.control["AutoSwapToggle"] = False

    # Restore swap faces button state
    swap_faces_state = data.get("swap_faces_enabled", True)
    main_window.swapfacesButton.setChecked(swap_faces_state)
    # On a batch load, this is harmful and breaks the logic.
    if swap_faces_state and not is_batch_load:
        # This will trigger a frame refresh via its own logic
        video_control_actions.process_swap_faces(main_window)
    print(f"[DEBUG] Swap Faces button state restored: {swap_faces_state}")

    # Restore misc paths and settings
    main_window.last_target_media_folder_path = data.get(
        "last_target_media_folder_path", ""
    )
    main_window.last_input_media_folder_path = data.get(
        "last_input_media_folder_path", ""
    )
    main_window.loaded_embedding_filename = data.get("loaded_embedding_filename", "")

    # Update all control widgets in the "Settings" tab
    common_widget_actions.set_control_widgets_values(main_window)

    # Ensure output folder is set correctly
    output_folder = data.get("control", {}).get("OutputMediaFolder", "")
    common_widget_actions.create_control(
        main_window, "OutputMediaFolder", output_folder
    )
    main_window.outputFolderLineEdit.setText(output_folder)

    # Update parameter widgets to default (or first face's)
    common_widget_actions.set_widgets_values_using_face_id_parameters(
        main_window, face_id=False
    )

    if not is_batch_load:
        layout_actions.fit_image_to_view_onchange(main_window)


def _load_job_markers(main_window: "MainWindow", data: dict):
    """Loads standard markers and job segment markers."""
    # Load standard markers
    loaded_markers = data.get("markers", {})
    # Convert marker parameters from dict to ParametersDict
    loaded_markers_converted = convert_markers_to_job_type(
        main_window, copy.deepcopy(loaded_markers), misc_helpers.ParametersDict
    )

    for marker_position, marker_data in loaded_markers_converted.items():
        video_control_actions.add_marker(
            main_window,
            marker_data.get("parameters", {}),
            marker_data.get("control", {}),
            int(marker_position),
        )

    # Load job marker pairs (segments)
    main_window.job_marker_pairs = data.get("job_marker_pairs", [])

    # Update slider visuals to show markers
    main_window.videoSeekSlider.update()


def _validate_job_files_exist(data: dict) -> tuple[bool, str | None]:
    """
    (NEW) Performs a pre-flight check on job data *before* loading.
    Validates that all required media files exist on disk.
    This is the single source of truth for job file validation.
    Returns (True, None) on success, or (False, error_message) on failure.
    """
    is_job_valid = True
    skip_reason = ""

    # --- 1. Validate the SINGLE required target media ---
    job_selected_media_id = data.get("selected_media_id")
    media_path_to_check = None

    if not job_selected_media_id:
        is_job_valid = False
        skip_reason = "No target media selected in job."
    else:
        found_media = False
        for media in data.get("target_medias_data", []):
            if media.get("media_id") == job_selected_media_id:
                media_path_to_check = media.get("media_path")
                if not media_path_to_check:
                    is_job_valid = False
                    skip_reason = (
                        f"Selected media ID {job_selected_media_id} has no media path."
                    )
                elif not os.path.exists(media_path_to_check):
                    is_job_valid = False
                    skip_reason = f"Target media file not found: {media_path_to_check}"
                found_media = True
                break
        if not found_media and is_job_valid:
            is_job_valid = False
            skip_reason = f"Selected media ID {job_selected_media_id} not found in job's media list."

    # --- 2. Validate all REQUIRED input faces and embeddings ---
    if is_job_valid:
        required_face_ids = set()
        required_embed_ids = set()

        for target_face in data.get("target_faces_data", {}).values():
            required_face_ids.update(target_face.get("assigned_input_faces", []))
            required_embed_ids.update(target_face.get("assigned_merged_embeddings", []))

        # Check face files
        all_input_faces_in_job = data.get("input_faces_data", {})
        for face_id in required_face_ids:
            if face_id not in all_input_faces_in_job:
                is_job_valid = False
                skip_reason = f"Required input face ID {face_id} not found in job data."
                break

            face_path = all_input_faces_in_job[face_id].get("media_path")
            if not face_path:
                is_job_valid = False
                skip_reason = f"Input face ID {face_id} has no media path."
                break
            if not os.path.exists(face_path):
                is_job_valid = False
                skip_reason = f"Input face file not found: {face_path}"
                break

        # Check embeddings (just need to exist in the job data)
        if is_job_valid:
            all_embeddings_in_job = data.get("embeddings_data", {})
            for embed_id in required_embed_ids:
                if embed_id not in all_embeddings_in_job:
                    is_job_valid = False
                    skip_reason = (
                        f"Required embedding ID {embed_id} not found in job data."
                    )
                    break

    # --- 3. Final decision ---
    if is_job_valid:
        return (True, None)
    else:
        return (False, skip_reason)


def _validate_job_data_for_loading(data: dict) -> tuple[bool, str | None]:
    """
    Performs a pre-flight check on job data *before* loading.
    This is a wrapper for _validate_job_files_exist, used for single job loads.
    Returns (True, None) on success, or (False, error_message) on failure.
    """
    # This function is now a simple wrapper.
    # The _analyze_job_batch method contains a more complex
    # "validate-and-collect" logic.
    return _validate_job_files_exist(data)


def load_job_workspace(main_window: "MainWindow", job_name: str):
    """
    Main function to load a job workspace. (HEAVY LOAD)
    Orchestrates clearing the UI and loading all components from the job file.
    This is used by the 'Load Job' button.
    """

    print("[DEBUG] Loading job workspace...")
    data_filename = os.path.join(jobs_dir, f"{job_name}.json")
    if not Path(data_filename).is_file():
        print(f"[DEBUG] No valid file found for job: {job_name}.")
        QMessageBox.critical(
            main_window, "Load Error", f"Job file not found:\n{data_filename}"
        )
        return

    try:
        with open(data_filename, "r") as data_file:
            data = json.load(data_file)
    except Exception as e:
        print(f"[ERROR] Failed to read or parse job file {data_filename}: {e}")
        QMessageBox.critical(
            main_window, "Load Error", f"Failed to load job '{job_name}':\n{e}"
        )
        return

    # --- Validate job data BEFORE clearing workspace ---
    is_valid, error_msg = _validate_job_data_for_loading(data)
    if not is_valid:
        print(f"[ERROR] Cannot load job '{job_name}'. Reason: {error_msg}")
        QMessageBox.critical(
            main_window,
            "Load Error",
            f"Failed to load job '{job_name}':\n\n{error_msg}",
        )
        return  # Abort loading

    # --- Show Progress Dialog ---
    steps = [
        "Clearing State",
        "Loading Target Media",
        "Loading Input Faces",
        "Loading Embeddings",
        "Loading Target Faces",
        "Loading Settings",
        "Loading Markers",
        "Finalizing",
    ]
    total_steps = len(steps)
    progress_dialog = widget_components.JobLoadingDialog(
        total_steps, parent=main_window
    )
    progress_dialog.show()
    QtWidgets.QApplication.processEvents()

    # --- Execute Loading Steps ---
    try:
        progress_dialog.update_progress(1, total_steps, steps[0])
        _clear_main_window_state(main_window)

        # Store job name context for processing
        main_window.current_job_name = job_name
        main_window.use_job_name_for_output = data.get("use_job_name_for_output", False)
        main_window.output_file_name = data.get("output_file_name", None)

        progress_dialog.update_progress(2, total_steps, steps[1])
        _load_job_target_media(main_window, data)

        progress_dialog.update_progress(3, total_steps, steps[2])
        _load_job_input_faces(main_window, data)

        progress_dialog.update_progress(4, total_steps, steps[3])
        _load_job_embeddings(main_window, data)

        progress_dialog.update_progress(5, total_steps, steps[4])
        _load_job_target_faces_and_params(main_window, data)

        progress_dialog.update_progress(6, total_steps, steps[5])
        _load_job_controls_and_state(main_window, data, is_batch_load=False)

        progress_dialog.update_progress(7, total_steps, steps[6])
        _load_job_markers(main_window, data)

        progress_dialog.update_progress(8, total_steps, steps[7])
        print(f"[DEBUG] Loaded workspace from: {data_filename}")

        # After loading, check if any target faces were loaded
        if main_window.target_faces:
            # Get the ID of the first loaded target face
            first_face_id = list(main_window.target_faces.keys())[0]

            # Ensure this face is marked as selected internally
            main_window.selected_target_face_id = first_face_id

            # Get the actual button instance
            first_face_button = main_window.target_faces.get(first_face_id)

            if first_face_button:
                first_face_button.setChecked(True)  # Visually select it
                main_window.cur_selected_target_face_button = first_face_button
                print(
                    f"[DEBUG] Set cur_selected_target_face_button to face_id: {first_face_id}"
                )

                # *** Visually check assigned embeddings for this face ***
                print(
                    f"[DEBUG] Checking assigned embeddings for face_id: {first_face_id}"
                )
                assigned_embedding_ids = (
                    first_face_button.assigned_merged_embeddings.keys()
                )
                for embed_id in assigned_embedding_ids:
                    embed_button = main_window.merged_embeddings.get(embed_id)
                    if embed_button:
                        embed_button.setChecked(True)
                        print(
                            f"  - Checked embedding: {embed_button.embedding_name} (ID: {embed_id})"
                        )

                # *** Visually check assigned input faces for this face ***
                print(
                    f"[DEBUG] Checking assigned input faces for face_id: {first_face_id}"
                )
                assigned_input_face_ids = first_face_button.assigned_input_faces.keys()
                for input_face_id in assigned_input_face_ids:
                    input_face_button = main_window.input_faces.get(input_face_id)
                    if input_face_button:
                        input_face_button.setChecked(True)
                        print(
                            f"  - Checked input face: {input_face_button.media_path} (ID: {input_face_id})"
                        )

            print(
                f"[DEBUG] Setting parameter widgets for loaded face_id: {first_face_id}"
            )
            # Now call the update function with the specific face_id
            common_widget_actions.set_widgets_values_using_face_id_parameters(
                main_window, face_id=first_face_id
            )
            # Store these parameters as the 'current' ones used by UI until another face is selected
            main_window.current_widget_parameters = main_window.parameters[
                first_face_id
            ].copy()

        else:
            # If no faces were loaded, ensure selection is cleared
            main_window.selected_target_face_id = False
            main_window.cur_selected_target_face_button = None
            main_window.current_widget_parameters = (
                main_window.default_parameters.copy()
            )
            print(
                "[DEBUG] No target faces loaded in job, parameter widgets retain default values."
            )

        # Update slider visuals after loading everything
        main_window.videoSeekSlider.update()

        # Final refresh ensures graphics view is up-to-date after potential parameter changes
        common_widget_actions.refresh_frame(main_window)

        # --- Re-enable all UI controls after loading ---
        layout_actions.enable_all_parameters_and_control_widget(main_window)
        print("[DEBUG] All UI controls re-enabled after job load.")

        print(f"[DEBUG] Successfully loaded workspace from: {data_filename}")

    except Exception as e:
        print(f"[ERROR] Error during job workspace loading: {e}")
        traceback.print_exc()
        QMessageBox.critical(
            main_window,
            "Load Error",
            f"An error occurred while loading job '{job_name}':\n{e}",
        )
    finally:
        progress_dialog.close()


def _restore_workspace_from_snapshot(main_window: "MainWindow", data: dict):
    """
    Restores the entire UI workspace from a saved snapshot dictionary.
    This is used after a job batch finishes to return to the pre-batch state.
    """

    print("[DEBUG] Restoring workspace from snapshot...")
    if not data:
        print("[ERROR] Workspace snapshot data is empty. Cannot restore.")
        QMessageBox.critical(
            main_window,
            "Restore Error",
            "Failed to restore workspace: Snapshot data was empty.",
        )
        return

    # --- Show Progress Dialog ---
    steps = [
        "Clearing State",
        "Restoring Target Media",
        "Restoring Input Faces",
        "Restoring Embeddings",
        "Restoring Target Faces",
        "Restoring Settings",
        "Restoring Markers",
        "Finalizing",
    ]
    total_steps = len(steps)
    progress_dialog = widget_components.JobLoadingDialog(
        total_steps, parent=main_window
    )
    progress_dialog.setWindowTitle("Restoring Workspace")
    progress_dialog.update_progress(0, total_steps, "Initializing restore...")
    progress_dialog.show()
    QtWidgets.QApplication.processEvents()

    # --- Execute Loading Steps ---
    try:
        progress_dialog.update_progress(1, total_steps, steps[0])
        _clear_main_window_state(main_window)

        # Clear job name context (we are restoring, not loading a job)
        main_window.current_job_name = None
        main_window.use_job_name_for_output = False
        main_window.output_file_name = None

        progress_dialog.update_progress(2, total_steps, steps[1])
        _load_job_target_media(main_window, data)

        progress_dialog.update_progress(3, total_steps, steps[2])
        _load_job_input_faces(main_window, data)

        progress_dialog.update_progress(4, total_steps, steps[3])
        _load_job_embeddings(main_window, data)

        progress_dialog.update_progress(5, total_steps, steps[4])
        _load_job_target_faces_and_params(main_window, data)

        progress_dialog.update_progress(6, total_steps, steps[5])
        # Note: is_batch_load=False here ensures swap faces button logic runs if needed during restore
        _load_job_controls_and_state(main_window, data, is_batch_load=False)

        progress_dialog.update_progress(7, total_steps, steps[6])
        _load_job_markers(main_window, data)

        progress_dialog.update_progress(8, total_steps, steps[7])

        # After restoring, check if any target faces were restored
        if main_window.target_faces:
            # Get the ID of the first restored target face
            first_face_id = list(main_window.target_faces.keys())[0]

            # Ensure this face is marked as selected internally
            main_window.selected_target_face_id = first_face_id

            # Get the actual button instance
            first_face_button = main_window.target_faces.get(first_face_id)

            if first_face_button:
                first_face_button.setChecked(True)  # Visually select it
                main_window.cur_selected_target_face_button = first_face_button
                print(
                    f"[DEBUG] Set cur_selected_target_face_button to face_id: {first_face_id}"
                )

                # *** Visually check assigned embeddings for this face ***
                print(
                    f"[DEBUG] Checking assigned embeddings for restored face_id: {first_face_id}"
                )
                assigned_embedding_ids = (
                    first_face_button.assigned_merged_embeddings.keys()
                )
                for embed_id in assigned_embedding_ids:
                    embed_button = main_window.merged_embeddings.get(embed_id)
                    if embed_button:
                        embed_button.setChecked(True)
                        print(
                            f"  - Checked embedding: {embed_button.embedding_name} (ID: {embed_id})"
                        )

                # *** Visually check assigned input faces for this face ***
                print(
                    f"[DEBUG] Checking assigned input faces for restored face_id: {first_face_id}"
                )
                assigned_input_face_ids = first_face_button.assigned_input_faces.keys()
                for input_face_id in assigned_input_face_ids:
                    input_face_button = main_window.input_faces.get(input_face_id)
                    if input_face_button:
                        input_face_button.setChecked(True)
                        print(
                            f"  - Checked input face: {input_face_button.media_path} (ID: {input_face_id})"
                        )

            print(
                f"[DEBUG] Setting parameter widgets for restored face_id: {first_face_id}"
            )
            # Now call the update function with the specific face_id
            common_widget_actions.set_widgets_values_using_face_id_parameters(
                main_window, face_id=first_face_id
            )
            # Store these parameters as the 'current' ones used by UI
            main_window.current_widget_parameters = main_window.parameters[
                first_face_id
            ].copy()
        else:
            # If no faces were restored
            main_window.selected_target_face_id = False
            main_window.cur_selected_target_face_button = None
            main_window.current_widget_parameters = (
                main_window.default_parameters.copy()
            )
            print(
                "[DEBUG] No target faces restored in snapshot, parameter widgets retain default values."
            )

        # Update slider visuals after restoring everything
        main_window.videoSeekSlider.update()

        # Final refresh ensures graphics view is up-to-date
        common_widget_actions.refresh_frame(main_window)

        # --- Re-enable all UI controls after restoring ---
        layout_actions.enable_all_parameters_and_control_widget(main_window)
        print("[DEBUG] All UI controls re-enabled after workspace restore.")

        print("[DEBUG] Workspace restored successfully from snapshot.")

    except Exception as e:
        print(f"[ERROR] Error during workspace snapshot restoration: {e}")
        traceback.print_exc()
        QMessageBox.critical(
            main_window,
            "Restore Error",
            f"An error occurred while restoring the workspace:\n{e}",
        )
    finally:
        progress_dialog.close()


def _serialize_job_data(main_window: "MainWindow") -> dict:
    """Serializes all necessary workspace data into a dictionary for saving."""

    target_faces_data = {}
    embeddings_data = {}
    input_faces_data = {}

    # Serialize Input Faces
    for face_id, input_face in main_window.input_faces.items():
        input_faces_data[face_id] = {"media_path": input_face.media_path}

    # Serialize Target Faces and their parameters
    for face_id, target_face in main_window.target_faces.items():
        params_obj = main_window.parameters.get(face_id)
        parameters_to_save = convert_parameters_to_job_type(
            main_window, params_obj, dict
        )

        target_faces_data[face_id] = {
            "cropped_face": target_face.cropped_face.tolist(),
            "embedding_store": {
                embed_model: embedding.tolist()
                for embed_model, embedding in target_face.embedding_store.items()
            },
            "parameters": parameters_to_save,
            "assigned_input_faces": list(target_face.assigned_input_faces.keys()),
            "assigned_merged_embeddings": list(
                target_face.assigned_merged_embeddings.keys()
            ),
            "assigned_input_embedding": {
                embed_model: embedding.tolist()
                for embed_model, embedding in target_face.assigned_input_embedding.items()
            },
        }

    # Serialize Merged Embeddings
    for embedding_id, embed_button in main_window.merged_embeddings.items():
        embeddings_data[embedding_id] = {
            "embedding_store": {
                embed_model: embedding.tolist()
                for embed_model, embedding in embed_button.embedding_store.items()
            },
            "embedding_name": embed_button.embedding_name,
        }

    # Serialize Target Media (excluding webcams)
    target_medias_data = [
        {"media_id": media_id, "media_path": target_media.media_path}
        for media_id, target_media in main_window.target_videos.items()
        if not target_media.is_webcam
    ]

    # Get selected media ID
    selected_media_id = (
        main_window.selected_video_button.media_id
        if main_window.selected_video_button
        else None
    )

    # Convert markers and controls to plain dicts for JSON
    markers_to_save = convert_markers_to_job_type(
        main_window, copy.deepcopy(main_window.markers), dict
    )
    control_to_save = convert_parameters_to_job_type(
        main_window, main_window.control, dict
    )

    # Assemble the final data dictionary
    workspace_data = {
        "target_medias_data": target_medias_data,
        "input_faces_data": input_faces_data,
        "embeddings_data": embeddings_data,
        "target_faces_data": target_faces_data,
        "control": control_to_save,
        "markers": markers_to_save,
        "job_marker_pairs": copy.deepcopy(main_window.job_marker_pairs),
        "selected_media_id": selected_media_id,
        "swap_faces_enabled": main_window.swapfacesButton.isChecked(),
        "last_target_media_folder_path": main_window.last_target_media_folder_path,
        "last_input_media_folder_path": main_window.last_input_media_folder_path,
        "loaded_embedding_filename": main_window.loaded_embedding_filename,
    }
    return workspace_data


def save_job_workspace(
    main_window: "MainWindow",
    job_name_path: str,
    use_job_name_for_output: bool = True,
    output_file_name: str = None,
):
    """
    Main function to save the current workspace to a job file.
    Note: 'job_name_path' is the full path *without* the .json extension.
    """
    print("[DEBUG] Saving job workspace...")
    data_filename = f"{job_name_path}.json"

    # Get all workspace data
    workspace_data = _serialize_job_data(main_window)

    # Add job-specific output settings
    workspace_data["use_job_name_for_output"] = use_job_name_for_output
    workspace_data["output_file_name"] = (
        output_file_name if not use_job_name_for_output else None
    )

    # Write data to JSON file
    with open(data_filename, "w") as data_file:
        json.dump(workspace_data, data_file, indent=4)

    print(f"[DEBUG] Job successfully saved to: {data_filename}")


# --- UI Setup and Signal Connections ---


def update_job_manager_buttons(main_window: "MainWindow"):
    """Enable/disable job manager buttons based on selection and job list state."""
    job_list = main_window.jobQueueList
    selected_count = len(job_list.selectedItems()) if job_list else 0
    job_count = job_list.count() if job_list else 0

    enable_on_selection = selected_count > 0
    enable_on_multi_selection = selected_count > 0
    enable_on_single_selection = selected_count == 1

    if hasattr(main_window, "buttonProcessSelected"):
        main_window.buttonProcessSelected.setEnabled(enable_on_multi_selection)
    if hasattr(main_window, "loadJobButton"):
        main_window.loadJobButton.setEnabled(
            enable_on_single_selection
        )  # Only allow loading one job
    if hasattr(main_window, "deleteJobButton"):
        main_window.deleteJobButton.setEnabled(enable_on_multi_selection)

    if hasattr(main_window, "buttonProcessAll"):
        main_window.buttonProcessAll.setEnabled(job_count > 0)


def setup_job_manager_ui(main_window: "MainWindow"):
    """Initialize UI widgets, connect signals, and refresh the job list."""
    # Find all child widgets related to the Job Manager dock
    main_window.addJobButton = main_window.findChild(
        QtWidgets.QPushButton, "addJobButton"
    )
    main_window.deleteJobButton = main_window.findChild(
        QtWidgets.QPushButton, "deleteJobButton"
    )
    main_window.jobQueueList = main_window.findChild(
        QtWidgets.QListWidget, "jobQueueList"
    )
    main_window.buttonProcessSelected = main_window.findChild(
        QtWidgets.QPushButton, "buttonProcessSelected"
    )
    main_window.buttonProcessAll = main_window.findChild(
        QtWidgets.QPushButton, "buttonProcessAll"
    )
    main_window.loadJobButton = main_window.findChild(
        QtWidgets.QPushButton, "loadJobButton"
    )
    main_window.refreshJobListButton = main_window.findChild(
        QtWidgets.QPushButton, "refreshJobListButton"
    )

    # Enable multi-selection for the job list
    if main_window.jobQueueList:
        main_window.jobQueueList.setSelectionMode(
            QtWidgets.QAbstractItemView.ExtendedSelection
        )

    # Connect signals to slots
    connect_job_manager_signals(main_window)

    # Initial population and state update
    refresh_job_list(main_window)
    update_job_manager_buttons(main_window)
    main_window.job_processor = None


def prompt_job_name(main_window: "MainWindow"):
    """
    Prompt user to enter a job name before saving.
    Includes validation checks for workspace readiness.
    """

    # --- Validation Checks ---
    has_target_face = bool(getattr(main_window, "target_faces", {}))
    if not has_target_face:
        reply = QMessageBox.warning(
            main_window,
            "Confirm Save",
            "No target faces found!\nNo face swaps will happen for this job. Proceed anyway?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply == QMessageBox.No:
            return

    output_folder = main_window.control.get("OutputMediaFolder", "").strip()
    if not output_folder:
        QMessageBox.warning(
            main_window,
            "Workspace Not Ready",
            "An Output Folder must be set in the 'Settings' tab before saving a job.",
        )
        return

    at_least_one_target_has_input = False
    if main_window.target_faces:
        for target_face in main_window.target_faces.values():
            # Check if this target face has any input faces OR merged embeddings assigned
            if (
                len(target_face.assigned_input_faces)
                + len(target_face.assigned_merged_embeddings)
            ) > 0:
                at_least_one_target_has_input = True
                break

    if not at_least_one_target_has_input:
        reply = QMessageBox.warning(
            main_window,
            "Confirm Save",
            "No input faces or embeddings are assigned to ANY target face!\n"
            "No face swaps will happen for this job. Proceed anyway?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply == QMessageBox.No:
            return
    # --- End Validation ---

    dialog = widget_components.SaveJobDialog(main_window)
    if dialog.exec() == QtWidgets.QDialog.Accepted:
        job_name = dialog.job_name
        use_job_name_for_output = dialog.use_job_name_for_output
        output_file_name = dialog.output_file_name

        # Validate job name
        if not job_name:
            QMessageBox.warning(
                main_window, "Invalid Job Name", "Job name cannot be empty."
            )
            return
        if not re.match(r"^[\w\- ]+$", job_name):
            QMessageBox.warning(
                main_window,
                "Invalid Job Name",
                "Job name contains invalid characters. Only letters, numbers, spaces, dashes, and underscores are allowed.",
            )
            return

        # Validate output file name if provided
        if not use_job_name_for_output and output_file_name:
            if not re.match(r"^[\w\- ]+$", output_file_name):
                QMessageBox.warning(
                    main_window,
                    "Invalid Output File Name",
                    "Output file name contains invalid characters.\n"
                    "Only letters, numbers, spaces, dashes, and underscores are allowed.",
                )
                return

        # Save the job and refresh the list
        save_job(main_window, job_name, use_job_name_for_output, output_file_name)
        refresh_job_list(main_window)


def connect_job_manager_signals(main_window: "MainWindow"):
    """Connect Job Manager UI buttons to their respective functions."""
    if main_window.addJobButton:
        main_window.addJobButton.clicked.connect(lambda: prompt_job_name(main_window))
    if main_window.deleteJobButton:
        main_window.deleteJobButton.clicked.connect(lambda: delete_job(main_window))
    if main_window.loadJobButton:
        main_window.loadJobButton.clicked.connect(
            lambda: load_job(main_window)
        )  # Uses heavy load
    if main_window.buttonProcessAll:
        main_window.buttonProcessAll.clicked.connect(
            lambda: start_processing_all_jobs(main_window)
        )
    if main_window.buttonProcessSelected:
        main_window.buttonProcessSelected.clicked.connect(
            lambda: process_selected_job(main_window)
        )
    if main_window.refreshJobListButton:
        main_window.refreshJobListButton.clicked.connect(
            lambda: refresh_job_list(main_window)
        )
    if main_window.jobQueueList:
        main_window.jobQueueList.itemSelectionChanged.connect(
            lambda: update_job_manager_buttons(main_window)
        )


def refresh_job_list(main_window: "MainWindow"):
    """Updates the job queue list widget with the latest job files."""
    if main_window.jobQueueList:
        main_window.jobQueueList.clear()
        job_names = list_jobs()
        main_window.jobQueueList.addItems(job_names)
        update_job_manager_buttons(main_window)


def get_selected_jobs(main_window: "MainWindow") -> list[str]:
    """Returns a list of selected job names from the job list widget."""
    selected_items = main_window.jobQueueList.selectedItems()
    return [item.text() for item in selected_items] if selected_items else []


# --- Job Processing Slots ---
# These functions are called by signals from the JobProcessor thread
# and run in the MAIN THREAD to safely modify the UI.


@Slot(dict)
def load_master_assets(main_window: "MainWindow", master_data: dict):
    """
    (SLOT) Loads all unique assets for the entire job batch.
    This is a HEAVY load, but only runs ONCE per batch.
    """
    print("[DEBUG] (Main Thread) load_master_assets called.")

    # --- Show Progress Dialog ---
    steps = [
        "Clearing State",
        "Loading All Target Media",
        "Loading All Input Faces",
        "Loading All Embeddings",
    ]
    total_steps = len(steps)

    # Create the dialog with only the accepted arguments
    progress_dialog = widget_components.JobLoadingDialog(
        total_steps, parent=main_window
    )

    # Set the title and label text *after* creation
    progress_dialog.setWindowTitle("Preparing Job Batch")
    # Use the dialog's own update method to set the initial text
    progress_dialog.update_progress(0, total_steps, "Initializing batch...")

    progress_dialog.show()
    QtWidgets.QApplication.processEvents()

    try:
        progress_dialog.update_progress(1, total_steps, steps[0])
        _clear_main_window_state(main_window)

        progress_dialog.update_progress(2, total_steps, steps[1])
        _load_job_target_media(main_window, master_data)  # Loads all unique media

        progress_dialog.update_progress(3, total_steps, steps[2])
        _load_job_input_faces(main_window, master_data)  # Loads all unique faces

        progress_dialog.update_progress(4, total_steps, steps[3])
        _load_job_embeddings(main_window, master_data)  # Loads all unique embeddings

        print("[DEBUG] (Main Thread) Master assets loaded.")

    except Exception as e:
        print(f"[ERROR] (Main Thread) Error during master asset loading: {e}")
        traceback.print_exc()
        QMessageBox.critical(
            main_window,
            "Load Error",
            f"An error occurred while loading batch assets:\n{e}",
        )
    finally:
        progress_dialog.close()
        # Use the instance event from the job_processor
        if main_window.job_processor:
            main_window.job_processor.master_assets_loaded_event.set()


@Slot(dict)
def load_job_settings(main_window: "MainWindow", job_data: dict):
    """
    (SLOT) Loads the lightweight settings for a single job from the batch.
    Assumes master assets are already loaded.
    """
    print("[DEBUG] (Main Thread) load_job_settings called.")
    try:
        # Store job name context for processing
        main_window.current_job_name = job_data.get("job_name_internal", "Unknown Job")
        main_window.use_job_name_for_output = job_data.get(
            "use_job_name_for_output", False
        )
        main_window.output_file_name = job_data.get("output_file_name", None)

        # --- Re-ordered loading logic ---

        # 1. Select the media FIRST. This triggers the (asynchronous) loading
        # of the first frame via process_current_frame.
        selected_media_id = job_data.get("selected_media_id", False)
        if selected_media_id and main_window.target_videos.get(selected_media_id):
            print(f"[DEBUG] (Main Thread) Clicking target media: {selected_media_id}")
            main_window.target_videos[selected_media_id].click()
        else:
            print(
                f"[WARN] (Main Thread) Could not select media_id {selected_media_id} for job."
            )
            # Try to select the first available media
            if main_window.target_videos:
                first_media = next(iter(main_window.target_videos.values()))
                print(
                    f"[WARN] (Main Thread) Selecting first available media instead: {first_media.media_id}"
                )
                first_media.click()
            else:
                print("[ERROR] (Main Thread) No target media loaded, cannot proceed.")
                # This job will likely fail, but we must continue

        # 2. Load target faces and parameters. This is safe.
        _load_job_target_faces_and_params(main_window, job_data)

        # 3. Load controls. This is now safe because the swap_button logic
        #    is fixed (is_batch_load=True) and won't trigger a bad refresh.
        _load_job_controls_and_state(main_window, job_data, is_batch_load=True)

        # 4. Load markers. This is safe.
        _load_job_markers(main_window, job_data)

        print(
            f"[DEBUG] (Main Thread) Lightweight settings loaded for job: {main_window.current_job_name}"
        )

    except Exception as e:
        print(f"[ERROR] (Main Thread) Error during job settings loading: {e}")
        traceback.print_exc()
        QMessageBox.critical(
            main_window,
            "Load Error",
            f"An error occurred while loading settings for job:\n{e}",
        )
    finally:
        # Use the instance event from the job_processor
        if main_window.job_processor:
            main_window.job_processor.job_settings_loaded_event.set()


@Slot()
def clear_job_settings(main_window: "MainWindow"):
    """
    (SLOT) Clears only the settings related to a single job,
    leaving the master assets (media, input faces, embeddings) loaded.
    """
    print("[DEBUG] (Main Thread) clear_job_settings called.")
    try:
        card_actions.clear_target_faces(
            main_window
        )  # Clears target faces and parameters
        video_control_actions.remove_all_markers(
            main_window
        )  # Clear markers from slider and data
        main_window.job_marker_pairs.clear()  # Clear job segments
        print("[DEBUG] (Main Thread) Job-specific settings cleared.")
    except Exception as e:
        print(f"[ERROR] (Main Thread) Error clearing job settings: {e}")
        traceback.print_exc()
    finally:
        # Use the instance event from the job_processor
        if main_window.job_processor:
            main_window.job_processor.job_settings_cleared_event.set()


@Slot()
def handle_batch_completion(main_window: "MainWindow"):
    """
    (SLOT) Called when the JobProcessor finishes its batch.
    Restores the workspace to its pre-batch state using a snapshot.
    Shows confirmation and skipped job reports after restoration.
    """
    if not main_window.job_processor:
        return

    batch_succeeded = main_window.job_processor.batch_succeeded
    skipped_jobs = main_window.job_processor.skipped_jobs

    print(f"[DEBUG] (Main Thread) Batch finished (Success: {batch_succeeded}).")

    # --- Restore Workspace from Snapshot ---
    snapshot = getattr(main_window, "workspace_snapshot_before_batch", None)
    if snapshot:
        print("[DEBUG] (Main Thread) Found workspace snapshot. Restoring...")
        try:
            _restore_workspace_from_snapshot(main_window, snapshot)
        except Exception as e:
            print(f"[ERROR] (Main Thread) Critical error during workspace restore: {e}")
            traceback.print_exc()
            QMessageBox.critical(
                main_window,
                "Restore Error",
                f"Failed to restore workspace after batch: {e}\n"
                "The UI may be in an unstable state. Please restart.",
            )
        finally:
            main_window.workspace_snapshot_before_batch = None  # Clear snapshot
            print("[DEBUG] (Main Thread) Workspace snapshot cleared.")
    else:
        print(
            "[WARN] (Main Thread) No workspace snapshot found. UI will remain in its last state."
        )
        # (Fallback) Just refresh VRAM, as we can't restore or reset
        common_widget_actions.update_gpu_memory_progressbar(main_window)

    # --- Show Batch Reports (AFTER restoration) ---

    # Report skipped jobs (if any)
    if skipped_jobs:
        skipped_message = "The following jobs were skipped due to errors:\n\n"
        skipped_message += "\n".join(f"- {job_error}" for job_error in skipped_jobs)
        QMessageBox.warning(main_window, "Skipped Jobs", skipped_message)

    # Report final status
    if batch_succeeded:
        QMessageBox.information(
            main_window,
            "Job Processing Complete",
            "All valid jobs have finished processing.",
        )
    else:
        # Batch failed
        QMessageBox.warning(
            main_window,
            "Job Processing Failed",
            "The job batch finished with errors. Please check the log for details. "
            "Any jobs that failed were not moved to 'completed'.",
        )


# --- Job Processing Thread ---


def process_selected_job(main_window: "MainWindow"):
    """Starts a JobProcessor thread for only the selected jobs."""
    selected_jobs = get_selected_jobs(main_window)
    if not selected_jobs:
        QMessageBox.warning(
            main_window, "No Job Selected", "Please select one or more jobs to process."
        )
        return

    print(f"[DEBUG] Processing selected jobs: {selected_jobs}")
    start_job_processor(main_window, jobs_to_process=selected_jobs)


def start_processing_all_jobs(main_window: "MainWindow"):
    """Starts a JobProcessor thread for all jobs in the list."""
    print("[DEBUG] Processing all jobs...")
    start_job_processor(main_window, jobs_to_process=None)  # None means all jobs


def start_job_processor(main_window: "MainWindow", jobs_to_process: list[str] | None):
    """Helper function to create, connect, and start the JobProcessor thread."""

    # Ensure no other processor is running
    if main_window.job_processor and main_window.job_processor.isRunning():
        QMessageBox.warning(
            main_window, "Already Processing", "A job processor is already running."
        )
        return

    # Save the current workspace state before starting the batch
    print("[DEBUG] Saving workspace snapshot before starting job processor...")
    main_window.workspace_snapshot_before_batch = _serialize_job_data(main_window)
    print("[DEBUG] Workspace snapshot saved.")

    main_window.job_processor = JobProcessor(
        main_window, jobs_to_process=jobs_to_process
    )

    # Connect signals from the worker thread to slots in the main thread
    # Connect signals for new batch-loading process
    main_window.job_processor.load_master_assets_signal.connect(
        lambda assets: load_master_assets(main_window, assets)
    )
    main_window.job_processor.load_job_settings_signal.connect(
        lambda settings: load_job_settings(main_window, settings)
    )
    main_window.job_processor.clear_job_settings_signal.connect(
        lambda: clear_job_settings(main_window)
    )

    # Connect signals for job completion
    main_window.job_processor.job_completed_signal.connect(
        lambda job_name: refresh_job_list(
            main_window
        )  # The signal sends job_name, refresh_job_list is called
    )
    main_window.job_processor.all_jobs_done_signal.connect(
        lambda: handle_batch_completion(main_window)
    )
    main_window.job_processor.job_failed_signal.connect(
        lambda job_name, error_msg: QMessageBox.critical(
            main_window, "Job Failed", f"Job '{job_name}' failed:\n{error_msg}"
        )
    )

    print("[DEBUG] Starting job_processor thread...")
    main_window.job_processor.start()


class JobProcessor(QThread):
    """
    A QThread worker responsible for processing a queue of jobs sequentially.

    Processing Logic:
    1. Analyzes all jobs to find unique assets.
    2. Emits signal to load all assets ONCE.
    3. Loops through jobs, emitting signals to:
       a. Load lightweight settings for a job.
       b. Trigger processing.
       c. Clear lightweight settings.
    """

    # --- Signals for batch processing ---
    # Signal main thread to load all heavy assets for the *entire batch*
    load_master_assets_signal = Signal(dict)
    # Signal main thread to load *only* the settings for the *next job*
    load_job_settings_signal = Signal(dict)
    # Signal main thread to *clear* settings from the *previous job*
    clear_job_settings_signal = Signal()

    # --- Signals for reporting ---
    job_completed_signal = Signal(str)
    all_jobs_done_signal = Signal()
    job_failed_signal = Signal(str, str)

    # --- JobProcessor Timeouts (in seconds) ---
    JOB_START_TIMEOUT = 30  # Max time to wait for the video processor to start
    JOB_HEARTBEAT_WATCHDOG_TIMEOUT = 900  # (15 min - Heartbeat is every 500 frames) Max time between heartbeats before job is considered frozen
    MASTER_ASSETS_LOAD_TIMEOUT = 600  # Max time to load all batch assets (10 minutes)
    JOB_SETTINGS_LOAD_TIMEOUT = (
        180  # Max time to load a single job's settings (3 minutes)
    )
    JOB_SETTINGS_CLEAR_TIMEOUT = 30  # Max time to clear a single job's settings

    def __init__(
        self, main_window: "MainWindow", jobs_to_process: list[str] | None = None
    ):
        """
        Initializes the processor.
        :param main_window: Reference to the main UI.
        :param jobs_to_process: A list of job names to process. If None, processes all jobs.
        """
        super().__init__()
        self.main_window = main_window
        self.jobs_dir = os.path.join(os.getcwd(), "jobs")
        self.completed_dir = os.path.join(self.jobs_dir, "completed")

        if jobs_to_process is not None:
            self.jobs = jobs_to_process
        else:
            self.jobs = list_jobs()  # Get all current jobs

        self.current_job_name = None
        self.batch_succeeded = (
            False  # Flag to track if the batch finished without errors
        )
        self.skipped_jobs = []  # Store jobs that fail pre-flight checks

        if not os.path.exists(self.completed_dir):
            os.makedirs(self.completed_dir)

        # --- Encapsulated Threading Events ---
        self.master_assets_loaded_event = threading.Event()
        self.job_settings_loaded_event = threading.Event()
        self.job_settings_cleared_event = threading.Event()
        self.processing_started_event = threading.Event()
        self.processing_stopped_event = threading.Event()
        self.processing_heartbeat_event = threading.Event()  # For watchdog

        # Connect to the video processor's signals.
        # We MUST use DirectConnection:
        # The VideoProcessor lives in the MainThread (A).
        # This JobProcessor lives in the WorkerThread (B).
        # When this thread (B) is blocked waiting on an event (e.g., self.processing_heartbeat_event.wait()),
        # a default QueuedConnection would mean the slot (e.g., self.handle_processing_heartbeat)
        # would wait in Thread B's event queue... which is blocked. Deadlock.
        # DirectConnection forces the slot to run in the *emitter's* thread (MainThread A),
        # which is not blocked and can safely set the event, unblocking this thread (B).

        self.main_window.video_processor.processing_started_signal.connect(
            self.handle_processing_started, Qt.DirectConnection
        )
        self.main_window.video_processor.processing_stopped_signal.connect(
            self.handle_processing_stopped, Qt.DirectConnection
        )
        self.main_window.video_processor.processing_heartbeat_signal.connect(
            self.handle_processing_heartbeat, Qt.DirectConnection
        )

    @Slot()
    def handle_processing_started(self):
        """Slot to receive signal from VideoProcessor when recording starts."""
        print("[DEBUG] JobProcessor received processing_started_signal.")
        self.processing_started_event.set()

    @Slot()
    def handle_processing_stopped(self):
        """Slot to receive signal from VideoProcessor when recording/processing stops."""
        print("[DEBUG] JobProcessor received processing_stopped_signal.")
        self.processing_stopped_event.set()

    @Slot()
    def handle_processing_heartbeat(self):
        """Slot to receive heartbeat from VideoProcessor. Runs in Main Thread."""
        # This function runs in the Main Thread (due to DirectConnection)
        # and sets the event that the Worker Thread is waiting on.
        self.processing_heartbeat_event.set()

    def _read_job_file(self, job_name: str) -> dict | None:
        """Reads and parses a job's JSON file."""
        data_filename = os.path.join(self.jobs_dir, f"{job_name}.json")
        if not Path(data_filename).is_file():
            print(f"[DEBUG] No valid file found for job: {job_name}.")
            self.job_failed_signal.emit(
                job_name, f"Job file not found: {data_filename}"
            )
            return None
        try:
            with open(data_filename, "r") as data_file:
                data = json.load(data_file)
            data["job_name_internal"] = job_name  # Add job name for reference
            return data
        except Exception as e:
            print(f"[ERROR] Failed to read or parse job file {data_filename}: {e}")
            self.job_failed_signal.emit(job_name, f"Failed to load job file: {e}")
            return None

    def _analyze_job_batch(self) -> dict | None:
        """
        (SMART ANALYSIS & VALIDATION) Reads all job JSONs in the batch.
        1. Validates each job's files using the central validation function.
        2. Aggregates a master list of *unique* required assets from *valid* jobs.
        3. Populates self.job_data_list (valid jobs) and self.skipped_jobs (invalid jobs).
        """
        print(
            "[DEBUG] (JobProcessor) Performing SMART analysis and VALIDATION of job batch..."
        )
        master_data = {
            "target_medias_data": [],
            "input_faces_data": {},
            "embeddings_data": {},
        }
        # Use sets to track what we've already added to the master list
        seen_media_ids = set()
        seen_face_ids = set()
        seen_embed_ids = set()

        valid_job_data_list = []  # List for jobs that pass validation
        self.skipped_jobs.clear()  # Clear skipped list for this batch

        for job_name in self.jobs:
            data = self._read_job_file(job_name)
            if data is None:
                # _read_job_file already emitted a failure signal, but we also
                # add it to the skipped list for the final report.
                self.skipped_jobs.append(
                    f"{job_name}: Job file could not be read or parsed."
                )
                continue  # Skip to the next job

            # --- 1. Validate the entire job FIRST ---
            # Use the new, centralized validation function
            is_job_valid, skip_reason = _validate_job_files_exist(data)

            if not is_job_valid:
                print(
                    f"[WARN] (JobProcessor) Skipping job '{job_name}'. Reason: {skip_reason}"
                )
                self.skipped_jobs.append(f"{job_name}: {skip_reason}")
                continue  # Skip this job

            # --- 2. Job is valid, now COLLECT unique assets ---
            # (No need to check os.path.exists here, validation already passed)

            # Collect target media
            job_selected_media_id = data.get("selected_media_id")
            if job_selected_media_id not in seen_media_ids:
                for media in data.get("target_medias_data", []):
                    if media.get("media_id") == job_selected_media_id:
                        master_data["target_medias_data"].append(media)
                        seen_media_ids.add(job_selected_media_id)
                        break

            # Get required faces and embeddings for this job
            required_face_ids = set()
            required_embed_ids = set()
            for target_face in data.get("target_faces_data", {}).values():
                required_face_ids.update(target_face.get("assigned_input_faces", []))
                required_embed_ids.update(
                    target_face.get("assigned_merged_embeddings", [])
                )

            # Collect input faces
            all_input_faces_in_job = data.get("input_faces_data", {})
            for face_id in required_face_ids:
                if face_id not in seen_face_ids:
                    # We know face_id exists in all_input_faces_in_job from validation
                    master_data["input_faces_data"][face_id] = all_input_faces_in_job[
                        face_id
                    ]
                    seen_face_ids.add(face_id)

            # Collect embeddings
            all_embeddings_in_job = data.get("embeddings_data", {})
            for embed_id in required_embed_ids:
                if embed_id not in seen_embed_ids:
                    # We know embed_id exists in all_embeddings_in_job from validation
                    master_data["embeddings_data"][embed_id] = all_embeddings_in_job[
                        embed_id
                    ]
                    seen_embed_ids.add(embed_id)

            # --- 3. Add valid job to processing list ---
            valid_job_data_list.append(data)

        print("[DEBUG] (JobProcessor) Smart Analysis complete:")
        print(f"  - {len(valid_job_data_list)} jobs are valid and will be processed.")
        print(f"  - {len(self.skipped_jobs)} jobs will be skipped due to errors.")
        print(
            f"  - {len(master_data['target_medias_data'])} unique target media required."
        )
        print(
            f"  - {len(master_data['input_faces_data'])} unique input faces required."
        )
        print(f"  - {len(master_data['embeddings_data'])} unique embeddings required.")

        self.job_data_list = valid_job_data_list  # Store *only* the valid jobs
        return master_data

    def _trigger_and_wait_for_processing(self, job_name: str) -> bool:
        """
        Triggers the record button and waits for processing to start and stop.
        Uses a heartbeat event to ensure the job has not frozen.
        Returns True on success, False on failure.
        """
        print(f"[DEBUG] Toggling record button for job '{job_name}'...")
        self.processing_started_event.clear()
        self.processing_stopped_event.clear()

        # Mark that this recording was initiated by the Job Manager
        self.main_window.job_manager_initiated_record = True

        # Toggle the button (thread-safe UI call)
        QMetaObject.invokeMethod(
            self.main_window.buttonMediaRecord, "toggle", Qt.QueuedConnection
        )

        # --- Wait for Processing to Start ---
        if not self.processing_started_event.wait(timeout=self.JOB_START_TIMEOUT):
            error_msg = "Timeout waiting for processing to start signal."
            print(f"[ERROR] {error_msg}")
            # Attempt to toggle off the record button if it got stuck
            if self.main_window.buttonMediaRecord.isChecked():
                print("[WARN] Attempting to toggle record button off due to timeout.")
                QMetaObject.invokeMethod(
                    self.main_window.buttonMediaRecord, "toggle", Qt.QueuedConnection
                )
            self.main_window.video_processor.stop_processing()
            self.job_failed_signal.emit(job_name, error_msg)
            return False  # Failure

        # --- Wait for Processing to Complete (with Heartbeat Watchdog) ---
        print(
            "[DEBUG] JobProcessor detected processing started. Waiting for completion with heartbeat..."
        )

        # (NEW) Polling loop logic
        watchdog_timer_start = time.perf_counter()

        while True:
            # Check if the stop signal was set (e.g., job finished)
            # We must check this *before* waiting.
            if self.processing_stopped_event.is_set():
                break  # Exit the while loop (SUCCESS)

            # Wait for a heartbeat, but only for a short time (1 second)
            # This makes the loop responsive to the stop signal.
            try:
                heartbeat_received = self.processing_heartbeat_event.wait(timeout=1.0)
            except Exception as e:
                error_msg = f"Heartbeat wait error: {e}"
                print(f"[ERROR] {error_msg}")
                self.main_window.video_processor.stop_processing()  # Force stop
                self.job_failed_signal.emit(job_name, error_msg)
                return False  # Failure

            # --- Check our 3 conditions ---

            # 1. Did we get a heartbeat?
            if heartbeat_received:
                # Yes. Reset the watchdog and clear the event for the next wait.
                watchdog_timer_start = time.perf_counter()
                self.processing_heartbeat_event.clear()
                continue  # Go to next loop iteration

            # 2. Did the job stop while we were in the 1-second wait?
            if self.processing_stopped_event.is_set():
                break  # Exit the while loop (SUCCESS)

            # 3. No heartbeat AND job not stopped. Check watchdog.
            now = time.perf_counter()
            if (now - watchdog_timer_start) > self.JOB_HEARTBEAT_WATCHDOG_TIMEOUT:
                # Watchdog timed out!
                error_msg = f"Job frozen. No heartbeat received in {self.JOB_HEARTBEAT_WATCHDOG_TIMEOUT} seconds."
                print(f"[ERROR] {error_msg}")
                self.main_window.video_processor.stop_processing()  # Force stop
                self.job_failed_signal.emit(job_name, error_msg)
                return False  # Failure

            # No heartbeat, not stopped, watchdog OK. Loop again.
            pass

        # If the while loop exits, it means self.processing_stopped_event was set.
        print("[DEBUG] JobProcessor received stop signal.")
        return True  # Success

    def run(self):
        """
        Main thread loop: Performs batch analysis, loads assets, and processes jobs.
        """
        print("[DEBUG] Entering JobProcessor.run()...")
        self.batch_succeeded = False  # Default to False

        if not self.jobs:
            print("[DEBUG] No jobs to process. Exiting run().")
            self.all_jobs_done_signal.emit()
            return

        # --- 1. Analyze all jobs in the batch ---
        master_data = self._analyze_job_batch()
        if master_data is None:
            print(
                "[ERROR] Failed to analyze job batch (master_data is None). Aborting."
            )
            # _analyze_job_batch or _read_job_file should have emitted signals
            self.all_jobs_done_signal.emit()
            return

        # Check if any valid jobs remain after analysis
        if not self.job_data_list:
            print(
                "[DEBUG] No valid jobs found in batch after analysis. Skipping processing."
            )
            self.batch_succeeded = True  # No processing failed
            self.all_jobs_done_signal.emit()
            return

        # --- 2. Load all master assets ONCE ---
        print("[DEBUG] Emitting load_master_assets_signal...")
        self.master_assets_loaded_event.clear()  # Use instance event
        self.load_master_assets_signal.emit(master_data)

        # Wait for the main thread to finish loading all assets
        if not self.master_assets_loaded_event.wait(
            timeout=self.MASTER_ASSETS_LOAD_TIMEOUT
        ):  # Use instance event
            error_msg = "Timeout waiting for master assets to load."
            print(f"[ERROR] {error_msg}")
            self.job_failed_signal.emit("Batch Load", error_msg)
            return  # <-- Fails, batch_succeeded remains False

        print("[DEBUG] master_assets_loaded_event received, load complete.")

        # --- 3. Process each job with lightweight loading ---
        for job_data in self.job_data_list:
            job_name = job_data.get("job_name_internal", "Unknown")
            self.current_job_name = job_name
            print(f"[DEBUG] Beginning processing on job: {job_name}")

            # --- 3a. Load lightweight settings for this job ---
            print(f"[DEBUG] Emitting load_job_settings_signal for '{job_name}'")
            self.job_settings_loaded_event.clear()  # Use instance event
            self.load_job_settings_signal.emit(job_data)

            if not self.job_settings_loaded_event.wait(
                timeout=self.JOB_SETTINGS_LOAD_TIMEOUT
            ):  # Use instance event
                error_msg = f"Timeout waiting for job settings '{job_name}' to load."
                print(f"[ERROR] {error_msg}")
                self.job_failed_signal.emit(job_name, error_msg)
                break  # Abort batch, batch_succeeded remains False

            # --- 3b. Trigger video processing and wait ---
            if not self._trigger_and_wait_for_processing(job_name):
                # Job failed (timeout or other error)
                print(
                    f"[ERROR] Job '{job_name}' failed during processing. Aborting batch."
                )
                # job_failed_signal was already emitted in the helper function
                break  # Abort batch, batch_succeeded remains False

            print(f"[DEBUG] Processing finished for job: {self.current_job_name}")

            # --- 3c. Move job file to 'completed' ---
            job_path = os.path.join(self.jobs_dir, f"{job_name}.json")
            completed_path = os.path.join(self.completed_dir, f"{job_name}.json")
            if os.path.exists(job_path):
                try:
                    shutil.move(job_path, completed_path)
                    print(f"[DEBUG] Moved job '{job_name}' to completed folder.")
                    self.job_completed_signal.emit(job_name)
                except Exception as e:
                    error_msg = f"Failed to move job {job_name} to completed: {e}"
                    print(f"[ERROR] {error_msg}")
                    self.job_failed_signal.emit(job_name, error_msg)
                    break  # Abort batch, batch_succeeded remains False
            else:
                print(
                    f"[WARN] Job file not found after processing: {job_path}. Skipping move."
                )

            # --- 3d. Clear lightweight settings ---
            print(f"[DEBUG] Emitting clear_job_settings_signal for '{job_name}'")
            self.job_settings_cleared_event.clear()  # Use instance event
            self.clear_job_settings_signal.emit()
            if not self.job_settings_cleared_event.wait(
                timeout=self.JOB_SETTINGS_CLEAR_TIMEOUT
            ):  # Use instance event
                error_msg = f"Timeout waiting for job settings '{job_name}' to clear."
                print(f"[ERROR] {error_msg}")
                self.job_failed_signal.emit(job_name, error_msg)
                break  # Abort batch, batch_succeeded remains False

            print(f"[DEBUG] Job '{job_name}' fully completed. Moving to next.")
            self.msleep(1000)  # Small delay between jobs
        else:
            # This 'else' block executes ONLY if the 'for' loop
            # completes without a 'break' statement.
            print("[DEBUG] JobProcessor loop completed without errors.")
            self.batch_succeeded = True

        # --- 4. Finished ---
        print("[DEBUG] Finished processing all jobs loop.")
        self.all_jobs_done_signal.emit()
