import json
from pathlib import Path
import uuid
import copy
from functools import partial
from typing import TYPE_CHECKING, Dict
import os
import shutil
from PySide6.QtCore import QThread, Signal, QCoreApplication, Slot
from PySide6 import QtWidgets
import numpy as np
from PySide6.QtWidgets import QInputDialog, QMessageBox
import threading
import re

from app.ui.widgets.actions import common_actions as common_widget_actions
from app.ui.widgets.actions import card_actions
from app.ui.widgets.actions import list_view_actions
from app.ui.widgets.actions import video_control_actions
from app.ui.widgets.actions import layout_actions
from app.ui.widgets import ui_workers
from app.helpers.typing_helper import ParametersTypes, MarkerTypes
import app.helpers.miscellaneous as misc_helpers
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from app.ui.main_ui import MainWindow


jobs_dir = os.path.join(os.getcwd(), "jobs")
os.makedirs(jobs_dir, exist_ok=True)  # Ensure the directory exists

# Add a global event for job loading
job_loaded_event = threading.Event()

def convert_parameters_to_job_type(main_window: "MainWindow", parameters: dict|ParametersTypes, convert_type: type):
    if convert_type == dict:
        if isinstance(parameters, misc_helpers.ParametersDict):
            # Explicitly get the underlying dictionary data
            return parameters.data.copy()
        elif isinstance(parameters, dict):
            # If it's already a dict, return a copy
            return parameters.copy()
        else:
            # Handle unexpected types if necessary, log a warning
            print(f"[WARN] Unexpected type {type(parameters)} encountered when converting to dict.")
            # Attempt to return as is, or raise an error if appropriate
            return parameters
    elif convert_type == misc_helpers.ParametersDict:
        if not isinstance(parameters, misc_helpers.ParametersDict):
            if isinstance(parameters, dict):
                # Convert a standard dict to ParametersDict
                return misc_helpers.ParametersDict(parameters, main_window.default_parameters)
            else:
                # Handle unexpected types
                print(f"[WARN] Unexpected type {type(parameters)} encountered when converting to ParametersDict.")
                return parameters
        else:
            # It's already a ParametersDict, return it
            return parameters
    else:
        # Invalid convert_type specified
        print(f"[WARN] Invalid convert_type {convert_type} specified in convert_parameters_to_job_type.")
        return parameters

def convert_markers_to_job_type(main_window: "MainWindow", markers: MarkerTypes, convert_type: dict|misc_helpers.ParametersDict):
    # Convert Parameters inside the markers from ParametersDict to dict or vice-versa
    for _, marker_data in markers.items():
        # Convert parameters for each face_id within the marker
        if 'parameters' in marker_data and isinstance(marker_data['parameters'], dict):
            for target_face_id, target_parameters in marker_data['parameters'].items():
                marker_data['parameters'][target_face_id] = convert_parameters_to_job_type(main_window, target_parameters, convert_type)
        
        # Also convert the control dict within the marker
        if 'control' in marker_data:
            marker_data['control'] = convert_parameters_to_job_type(main_window, marker_data['control'], convert_type)
            
    return markers

def save_job(main_window, job_name: str, use_job_name_for_output: bool = True, output_file_name: str = None):
    """Saves the current workspace as a job in the 'jobs' directory."""
    data_filename = os.path.join(jobs_dir, f"{job_name}")
    save_job_workspace(main_window, data_filename, use_job_name_for_output, output_file_name)
    print(f"[DEBUG] Job saved: {data_filename}")

def list_jobs():
    """Lists all saved jobs from the 'jobs' directory."""
    if not os.path.exists(jobs_dir):
        return []
    return [f.replace('.json', '') for f in os.listdir(jobs_dir) if f.endswith('.json')]

def delete_job(main_window: "MainWindow"):
    """Deletes the selected job(s) from the 'jobs' directory after confirmation."""
    selected_jobs = get_selected_jobs(main_window)
    if not selected_jobs:
        QtWidgets.QMessageBox.warning(main_window, "No Job Selected", "Please select one or more jobs to delete.")
        return False

    confirm = QtWidgets.QMessageBox.question(
        main_window,
        "Confirm Deletion",
        f"Are you sure you want to delete the selected job{'s' if len(selected_jobs) > 1 else ''}?\n\n" + ", ".join(selected_jobs),
        QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
    )
    if confirm != QtWidgets.QMessageBox.Yes:
        return False

    deleted_any = False
    for job_name in selected_jobs:
        job_file = os.path.join(jobs_dir, f"{job_name}.json")
        if os.path.exists(job_file):
            os.remove(job_file)
            print(f"[DEBUG] Job deleted: {job_file}")
            deleted_any = True
        else:
            print(f"[DEBUG] Job file not found for deletion: {job_file}")
    if deleted_any:
        refresh_job_list(main_window)
        return True
    else:
        QtWidgets.QMessageBox.warning(main_window, "Job(s) Not Found", "None of the selected jobs exist.")
        return False

def load_job(main_window):
    """Loads whichever job is currently selected in the ListWidget."""
    selected_jobs = get_selected_jobs(main_window)
    if not selected_jobs:
        QMessageBox.warning(main_window, "No Job Selected", "Please select a job from the list.")
        return
    if len(selected_jobs) > 1:
        QMessageBox.warning(main_window, "Multiple Jobs Selected", "You can only load one job at a time. Please select a single job to load.")
        return
    job_name = selected_jobs[0]
    load_job_by_name(main_window, job_name)

def load_job_workspace(main_window: "MainWindow", job_name: str):
    from app.ui.widgets import widget_components
    print("[DEBUG] Loading job workspace...")
    jobs_dir = os.path.join(os.getcwd(), "jobs"); os.makedirs(jobs_dir, exist_ok=True)
    data_filename = os.path.join(jobs_dir, f"{job_name}.json")
    if not Path(data_filename).is_file():
        print(f"[DEBUG] No valid file found for job: {job_name}."); return
    with open(data_filename, 'r') as data_file: data = json.load(data_file)

    # Define steps for progress
    steps = [
        "Target Videos", "Input Faces", "Embeddings", "Target Faces", "Controls", "Swap Faces", "Markers", "Misc Fields", "Finalizing"
    ]
    total_steps = len(steps)
    progress_dialog = widget_components.JobLoadingDialog(total_steps, parent=main_window)
    progress_dialog.show()
    QtWidgets.QApplication.processEvents()
    step_idx = 0

    # --- Clear previous state --- 
    main_window.selected_video_button = None 
    main_window.control['AutoSwapToggle'] = False

    # Clear job name and output flag on main_window for later use
    main_window.current_job_name = job_name
    main_window.use_job_name_for_output = data.get('use_job_name_for_output', False)
    main_window.output_file_name = data.get('output_file_name', None) # Load output file name
    list_view_actions.clear_stop_loading_input_media(main_window)
    list_view_actions.clear_stop_loading_target_media(main_window)
    main_window.target_videos = {}
    card_actions.clear_input_faces(main_window)
    card_actions.clear_target_faces(main_window)
    card_actions.clear_merged_embeddings(main_window)
    if hasattr(main_window, 'selected_video_button'):
        btn = main_window.selected_video_button
        if btn and (not hasattr(btn, 'media_id') or btn.media_id not in main_window.target_videos):
            main_window.selected_video_button = None
    # Step 1: Target Videos
    step_idx += 1
    progress_dialog.update_progress(step_idx, total_steps, steps[step_idx-1])
    target_medias_data = data.get('target_medias_data', [])
    target_medias_files_list, target_media_ids = zip(*[(m['media_path'], m['media_id']) for m in target_medias_data]) if target_medias_data else ([], [])
    main_window.video_loader_worker = ui_workers.TargetMediaLoaderWorker(main_window=main_window, folder_name=False, files_list=target_medias_files_list, media_ids=target_media_ids)
    main_window.video_loader_worker.thumbnail_ready.connect(partial(list_view_actions.add_media_thumbnail_to_target_videos_list, main_window))
    main_window.video_loader_worker.run()
    selected_media_id = data.get('selected_media_id', False)
    if selected_media_id and main_window.target_videos.get(selected_media_id): main_window.target_videos[selected_media_id].click()

    # Step 2: Input Faces
    step_idx += 1
    progress_dialog.update_progress(step_idx, total_steps, steps[step_idx-1])
    input_faces_data = data.get('input_faces_data', {})
    input_media_paths, input_face_ids = zip(*[(f['media_path'], face_id) for face_id, f in input_faces_data.items()]) if input_faces_data else ([], [])
    main_window.input_faces_loader_worker = ui_workers.InputFacesLoaderWorker(main_window=main_window, folder_name=False, files_list=input_media_paths, face_ids=input_face_ids)
    main_window.input_faces_loader_worker.thumbnail_ready.connect(partial(list_view_actions.add_media_thumbnail_to_source_faces_list, main_window))
    main_window.input_faces_loader_worker.finished.connect(partial(common_widget_actions.refresh_frame, main_window))
    main_window.input_faces_loader_worker.files_list = list(main_window.input_faces_loader_worker.files_list)
    main_window.input_faces_loader_worker.run()

    # Step 3: Embeddings
    step_idx += 1
    progress_dialog.update_progress(step_idx, total_steps, steps[step_idx-1])
    for embedding_id, embedding_data in data.get('embeddings_data', {}).items():
        embedding_store = {embed_model: np.array(embed) for embed_model, embed in embedding_data['embedding_store'].items()}
        list_view_actions.create_and_add_embed_button_to_list(main_window, embedding_data['embedding_name'], embedding_store, embedding_id=embedding_id)

    # Step 4: Target Faces
    step_idx += 1
    progress_dialog.update_progress(step_idx, total_steps, steps[step_idx-1])
    loaded_target_faces_data = data.get('target_faces_data', {})
    for face_id_str, target_face_data in loaded_target_faces_data.items():
        face_id = int(face_id_str)
        cropped_face = np.array(target_face_data['cropped_face']).astype('uint8')
        pixmap = common_widget_actions.get_pixmap_from_frame(main_window, cropped_face)
        embedding_store = {embed_model: np.array(embed) for embed_model, embed in target_face_data['embedding_store'].items()}
        # Create the button and add it to the list/main_window dict
        list_view_actions.add_media_thumbnail_to_target_faces_list(main_window, cropped_face, embedding_store, pixmap, face_id)
        
        # Convert the loaded parameters dict into a ParametersDict object for the main_window, using integer key
        main_window.parameters[face_id] = convert_parameters_to_job_type(
            main_window, 
            target_face_data.get('parameters', {}), # Load parameters from target_face_data
            misc_helpers.ParametersDict
        )
        
        # Load assigned faces/embeddings into the created target_face object
        if face_id in main_window.target_faces:
            target_face_obj = main_window.target_faces[face_id]
            
            # Load assigned merged embeddings
            target_face_obj.assigned_merged_embeddings.clear() # Clear first
            for assigned_id in target_face_data.get('assigned_merged_embeddings', []):
                if assigned_id in main_window.merged_embeddings: 
                    target_face_obj.assigned_merged_embeddings[assigned_id] = main_window.merged_embeddings[assigned_id].embedding_store
            
            # Load assigned input faces
            target_face_obj.assigned_input_faces.clear() # Clear first
            for assigned_id in target_face_data.get('assigned_input_faces', []):
                if assigned_id in main_window.input_faces: 
                    target_face_obj.assigned_input_faces[assigned_id] = main_window.input_faces[assigned_id].embedding_store
            
            # Load pre-calculated assigned input embedding (if saved)
            target_face_obj.assigned_input_embedding = {
                embed_model: np.array(embed) 
                for embed_model, embed in target_face_data.get('assigned_input_embedding', {}).items()
            }
        else:
             print(f"[WARN] Target face object with id {face_id} not found after creation in load_job_workspace.")

    # Step 5: Controls
    step_idx += 1
    progress_dialog.update_progress(step_idx, total_steps, steps[step_idx-1])
    for control_name, control_value in data.get('control', {}).items():
        main_window.control[control_name] = control_value
    main_window.control['AutoSwapToggle'] = False #Re-disable this. Probably not needed, doesn't hurt.

    # Step 6: Swap Faces
    step_idx += 1
    progress_dialog.update_progress(step_idx, total_steps, steps[step_idx-1])
    swap_faces_state = data.get('swap_faces_enabled', True)
    main_window.swapfacesButton.setChecked(swap_faces_state)
    if swap_faces_state:
        video_control_actions.process_swap_faces(main_window)
    print(f"[DEBUG] Swap Faces button state restored: {swap_faces_state}")

    # Step 7: Markers
    step_idx += 1
    progress_dialog.update_progress(step_idx, total_steps, steps[step_idx-1])
    # Standard markers were already cleared, now load new ones
    loaded_markers = data.get('markers', {})
    loaded_markers_converted = convert_markers_to_job_type(main_window, copy.deepcopy(loaded_markers), dict)
    
    for marker_position, marker_data in loaded_markers_converted.items():
        loaded_params_dict = marker_data.get('parameters', {})
        reconstructed_parameters = {}
        for face_id_str, params in loaded_params_dict.items():
            face_id = int(face_id_str)
            reconstructed_parameters[face_id] = misc_helpers.ParametersDict(params, main_window.default_parameters)
        
        control_dict = marker_data.get('control', {})
        video_control_actions.add_marker(
            main_window, 
            reconstructed_parameters, # Now Dict[int, ParametersDict]
            control_dict, 
            int(marker_position)
        )

    # Load job marker pairs
    main_window.job_marker_pairs = data.get('job_marker_pairs', []) # Load the list directly
    # Remove loading of obsolete keys
    # main_window.job_start_frame = data.get('job_start_frame', None)
    # main_window.job_end_frame = data.get('job_end_frame', None)

    # Step 8: Misc Fields
    step_idx += 1
    progress_dialog.update_progress(step_idx, total_steps, steps[step_idx-1])
    main_window.last_target_media_folder_path = data.get('last_target_media_folder_path', '')
    main_window.last_input_media_folder_path = data.get('last_input_media_folder_path', '')
    main_window.loaded_embedding_filename = data.get('loaded_embedding_filename', '')
    common_widget_actions.set_control_widgets_values(main_window)
    output_folder = data.get('control', {}).get('OutputMediaFolder', '')
    common_widget_actions.create_control(main_window, 'OutputMediaFolder', output_folder)
    main_window.outputFolderLineEdit.setText(output_folder)
    layout_actions.fit_image_to_view_onchange(main_window)
    common_widget_actions.set_widgets_values_using_face_id_parameters(main_window, face_id=False)
    print(f"[DEBUG] Loaded workspace from: {data_filename}")
    progress_dialog.close()
    # Update slider visuals after loading everything
    main_window.videoSeekSlider.update()
    job_loaded_event.set()

def save_job_workspace(main_window: "MainWindow", job_name: str, use_job_name_for_output: bool = True, output_file_name: str = None):
    print("[DEBUG] Saving job workspace...")
    jobs_dir = os.path.join(os.getcwd(), "jobs"); os.makedirs(jobs_dir, exist_ok=True)
    # Note: job_name here is actually the full path constructed in save_job
    # Let's keep data_filename consistent with that for clarity
    data_filename = f"{job_name}.json" # This assumes job_name passed doesn't have .json yet
    
    target_faces_data = {}; embeddings_data = {}; input_faces_data = {}
    for face_id, input_face in main_window.input_faces.items():
        input_faces_data[face_id] = {'media_path': input_face.media_path}
    for face_id, target_face in main_window.target_faces.items():
        # Get parameters, checking if it's ParametersDict or just dict
        params_obj = main_window.parameters.get(face_id)
        if isinstance(params_obj, misc_helpers.ParametersDict):
            parameters_to_save = params_obj.data.copy()
        elif isinstance(params_obj, dict):
            parameters_to_save = params_obj.copy()
        else:
            # Fallback or error handling if it's neither
            print(f"[WARN] Unexpected type for parameters[{face_id}]: {type(params_obj)}. Saving empty dict.")
            parameters_to_save = {}

        target_faces_data[face_id] = {
            'cropped_face': target_face.cropped_face.tolist(),
            'embedding_store': {embed_model: embedding.tolist() for embed_model, embedding in target_face.embedding_store.items()},
            'parameters': parameters_to_save, # Use the prepared dict
            # Convert the control dict to a plain dict before saving
            'control': convert_parameters_to_job_type(main_window, main_window.control, dict), 
            'assigned_input_faces': [input_face_id for input_face_id in target_face.assigned_input_faces.keys()],
            'assigned_merged_embeddings': [embedding_id for embedding_id in target_face.assigned_merged_embeddings.keys()],
            'assigned_input_embedding': {embed_model: embedding.tolist() for embed_model, embedding in target_face.assigned_input_embedding.items()}
        }
    for embedding_id, embed_button in main_window.merged_embeddings.items():
        embeddings_data[embedding_id] = {
            'embedding_store': {embed_model: embedding.tolist() for embed_model, embedding in embed_button.embedding_store.items()},
            'embedding_name': embed_button.embedding_name
        }
    target_medias_data = [{'media_id': media_id, 'media_path': target_media.media_path}  
                          for media_id, target_media in main_window.target_videos.items() if not target_media.is_webcam]
    selected_media_id = main_window.selected_video_button.media_id if main_window.selected_video_button else False
    # Ensure markers and controls are converted to plain dicts for JSON
    markers_to_save = convert_markers_to_job_type(main_window, copy.deepcopy(main_window.markers), dict)
    control_to_save = convert_parameters_to_job_type(main_window, main_window.control, dict)
    
    # swap_faces_state = True # Old logic - use current state
    # print(f"[DEBUG] Swap Faces button state saved: {swap_faces_state}") # Use current state
    workspace_data = {
        'target_medias_data': target_medias_data,
        'input_faces_data': input_faces_data,
        'embeddings_data': embeddings_data,
        'target_faces_data': target_faces_data,
        'control': control_to_save, # Save converted control dict
        'markers': markers_to_save, # Save converted standard markers
        'job_marker_pairs': copy.deepcopy(main_window.job_marker_pairs), # Save the job marker pairs list
        'selected_media_id': getattr(main_window.selected_video_button, 'media_id', None),
        'swap_faces_enabled': main_window.swapfacesButton.isChecked(),
        # Remove obsolete keys
        # 'job_start_frame': main_window.job_start_frame,
        # 'job_end_frame': main_window.job_end_frame,
        'last_target_media_folder_path': main_window.last_target_media_folder_path,
        'last_input_media_folder_path': main_window.last_input_media_folder_path,
        'loaded_embedding_filename': main_window.loaded_embedding_filename,
        'use_job_name_for_output': use_job_name_for_output,
        'output_file_name': output_file_name if not use_job_name_for_output else None # Store output_file_name
    }
    with open(data_filename, 'w') as data_file:
        json.dump(workspace_data, data_file, indent=4)
    print(f"[DEBUG] Job successfully saved to: {data_filename}")

def update_job_manager_buttons(main_window):
    """Enable/disable job manager buttons based on selection and job list state."""
    job_list = main_window.jobQueueList
    selected_count = len(job_list.selectedItems()) if job_list else 0
    job_count = job_list.count() if job_list else 0

    # Enable/disable based on selection
    enable_on_selection = selected_count > 0
    if hasattr(main_window, 'buttonProcessSelected') and main_window.buttonProcessSelected:
        main_window.buttonProcessSelected.setEnabled(enable_on_selection)
    if hasattr(main_window, 'loadJobButton') and main_window.loadJobButton:
        main_window.loadJobButton.setEnabled(enable_on_selection)
    if hasattr(main_window, 'deleteJobButton') and main_window.deleteJobButton:
        main_window.deleteJobButton.setEnabled(enable_on_selection)

    # Enable/disable 'All' based on job list
    if hasattr(main_window, 'buttonProcessAll') and main_window.buttonProcessAll:
        main_window.buttonProcessAll.setEnabled(job_count > 0)

def setup_job_manager_ui(main_window):
    """Initialize UI widgets, connect signals, and refresh the job list for the job manager."""
    main_window.addJobButton = main_window.findChild(QtWidgets.QPushButton, "addJobButton")
    main_window.deleteJobButton = main_window.findChild(QtWidgets.QPushButton, "deleteJobButton")
    main_window.jobQueueList = main_window.findChild(QtWidgets.QListWidget, "jobQueueList")
    main_window.buttonProcessSelected = main_window.findChild(QtWidgets.QPushButton, "buttonProcessSelected")
    main_window.buttonProcessAll = main_window.findChild(QtWidgets.QPushButton, "buttonProcessAll")
    main_window.loadJobButton = main_window.findChild(QtWidgets.QPushButton, "loadJobButton")
    main_window.refreshJobListButton = main_window.findChild(QtWidgets.QPushButton, "refreshJobListButton")

    # Enable multi-selection for the job list
    if main_window.jobQueueList:
        main_window.jobQueueList.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)

    # Connect buttons
    if main_window.buttonProcessAll:
        main_window.buttonProcessAll.clicked.connect(lambda: start_processing_all_jobs(main_window))
    if main_window.buttonProcessSelected:
        main_window.buttonProcessSelected.clicked.connect(lambda: process_selected_job(main_window))
    if main_window.addJobButton and main_window.deleteJobButton:
        connect_job_manager_signals(main_window)
    if main_window.refreshJobListButton:
        main_window.refreshJobListButton.clicked.connect(lambda: refresh_job_list(main_window))
    main_window.jobQueueList.itemSelectionChanged.connect(lambda: update_job_manager_buttons(main_window))
    refresh_job_list(main_window)
    update_job_manager_buttons(main_window)
    main_window.job_processor = None

def prompt_job_name(main_window):
    """Prompt user to enter a job name before saving, with option to set output file name."""
    from app.ui.widgets import widget_components
    # Check for workspace readiness: at least one source/target face or embedding selected
    has_source_face = bool(getattr(main_window, 'input_faces', {})) and any(getattr(main_window, 'input_faces', {}))
    has_target_face = bool(getattr(main_window, 'target_faces', {})) and any(getattr(main_window, 'target_faces', {}))
    has_embedding = bool(getattr(main_window, 'merged_embeddings', {})) and any(getattr(main_window, 'merged_embeddings', {}))

    # Check for at least one target face
    if not has_target_face:
        reply = QMessageBox.warning(main_window, "Confirm Save", 
                                    "No target face selected! No face swaps will happen for this job. Proceed anyway?",
                                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.No:
            return

    # Check OutputMediaFolder is not empty
    output_folder = main_window.control.get('OutputMediaFolder', '').strip() if hasattr(main_window, 'control') else ''
    if not output_folder:
        QMessageBox.warning(main_window, "Workspace Not Ready", "Select an Output Folder. Your workspace must be fully ready to record before saving a job.")
        return

    # Check if ANY target face has input faces or embeddings assigned
    at_least_one_target_has_input = False
    if main_window.target_faces:
        for face_id, target_face in main_window.target_faces.items():
            has_input_faces = bool(getattr(target_face, 'assigned_input_faces', {})) and any(getattr(target_face, 'assigned_input_faces', {}))
            has_merged_embeddings = bool(getattr(target_face, 'assigned_merged_embeddings', {})) and any(getattr(target_face, 'assigned_merged_embeddings', {}))
            # assigned_input_embedding is derived so we check the sources
            if has_input_faces or has_merged_embeddings:
                at_least_one_target_has_input = True
                break

    if not at_least_one_target_has_input:
        reply = QMessageBox.warning(main_window, "Confirm Save", 
                                    "No input faces or merged embedding assigned to ANY target face! No face swaps will happen for this job. Proceed anyway?",
                                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.No:
            return

    dialog = widget_components.SaveJobDialog(main_window)
    if dialog.exec() == QtWidgets.QDialog.Accepted:
        job_name = dialog.job_name
        use_job_name_for_output = dialog.use_job_name_for_output
        output_file_name = dialog.output_file_name

        # Validate job name
        if not job_name:
            QMessageBox.warning(main_window, "Invalid Job Name", "Job name cannot be empty.")
            return
        if not re.match(r'^[\w\- ]+$', job_name):
            QMessageBox.warning(main_window, "Invalid Job Name", "Job name contains invalid characters. Only letters, numbers, spaces, dashes, and underscores are allowed.")
            return

        # Validate output file name if provided
        if not use_job_name_for_output and output_file_name:
            if not re.match(r'^[\w\- ]+$', output_file_name):
                QMessageBox.warning(main_window, "Invalid Output File Name", "Output file name contains invalid characters. Only letters, numbers, spaces, dashes, and underscores are allowed.")
                return
        elif not use_job_name_for_output and not output_file_name:
             # If checkbox is unchecked but output name is empty, maybe warn or default to job name?
             # For now, let's default to using the job name in save_job if output_file_name is None/empty.
             pass 

        # Pass the output_file_name to save_job
        save_job(main_window, job_name, use_job_name_for_output, output_file_name)
        refresh_job_list(main_window)

def load_job_by_name(main_window: "MainWindow", job_name):
    """Loads a specific job workspace by name using load_job_workspace."""
    print(f"[DEBUG] load_job_by_name('{job_name}') called.")
    try:
        # Call the specific job loading function
        load_job_workspace(main_window, job_name)
        # The rest of the logic (setting attributes) is handled within load_job_workspace
        print(f"[DEBUG] Finished loading job '{job_name}' via load_job_workspace. Setting job_loaded_event.")
    except Exception as e:
        print(f"[ERROR] Failed to load job '{job_name}': {e}")
        # Optionally show message box to user
    finally:
        # Signal that loading is complete (or failed)
        job_loaded_event.set()

def connect_job_manager_signals(main_window):
    """Connect Job Manager UI buttons to job actions."""
    main_window.addJobButton.clicked.connect(lambda: prompt_job_name(main_window))
    main_window.deleteJobButton.clicked.connect(lambda: delete_job(main_window))
    if main_window.loadJobButton:
        main_window.loadJobButton.clicked.connect(lambda: load_job(main_window))

def refresh_job_list(main_window):
    """Updates the job queue list with the latest job files."""
    main_window.jobQueueList.clear()
    job_names = list_jobs()
    main_window.jobQueueList.addItems(job_names)
    update_job_manager_buttons(main_window)

def get_selected_job(main_window):
    """Gets the currently selected job from the job list."""
    selected_item = main_window.jobQueueList.currentItem()
    return selected_item.text() if selected_item else None

def get_selected_jobs(main_window):
    """Returns a list of selected job names from the job list widget."""
    selected_items = main_window.jobQueueList.selectedItems()
    return [item.text() for item in selected_items] if selected_items else []

def process_selected_job(main_window: "MainWindow"):
    """Process only the selected jobs in the job list."""
    selected_jobs = get_selected_jobs(main_window)
    if not selected_jobs:
        QtWidgets.QMessageBox.warning(main_window, "No Job Selected", "Please select one or more jobs to process.")
        return
    print(f"[DEBUG] Processing selected jobs: {selected_jobs}")
    main_window.job_processor = JobProcessor(main_window, jobs_to_process=selected_jobs)
    main_window.job_processor.load_job_signal.connect(lambda job_name: load_job_by_name(main_window, job_name))
    main_window.job_processor.job_completed_signal.connect(lambda: refresh_job_list(main_window))
    main_window.job_processor.all_jobs_done_signal.connect(
        lambda:
            QtWidgets.QMessageBox.information(
                main_window, "Job Processing Complete", "Selected jobs finished processing."
            )
    )
    main_window.job_processor.start()

class JobProcessor(QThread):
    job_completed_signal = Signal(str)
    all_jobs_done_signal = Signal()
    load_job_signal = Signal(str)

    def __init__(self, main_window: "MainWindow", jobs_to_process=None):
        super().__init__()
        self.main_window = main_window
        self.jobs_dir = os.path.join(os.getcwd(), "jobs")
        self.completed_dir = os.path.join(self.jobs_dir, "completed")
        if jobs_to_process is not None:
            self.jobs = jobs_to_process
        else:
            self.jobs = list_jobs()
        self.current_job = None
        self.processing_started_event = threading.Event() # Event to wait for unified processing start

        if not os.path.exists(self.completed_dir):
            os.makedirs(self.completed_dir)
        
        # Connect to the video processor's new unified signal
        self.main_window.video_processor.processing_started_signal.connect(self.handle_processing_started)

    @Slot()
    def handle_processing_started(self):
        print("[DEBUG] JobProcessor received processing_started_signal.")
        self.processing_started_event.set()

    def run(self):
        print("[DEBUG] Entering JobProcessor.run()...")

        if not self.jobs:
            print("[DEBUG] No jobs to process. Exiting run().")
            self.all_jobs_done_signal.emit()
            return

        job_failed = False
        for job_name in self.jobs:
            if job_failed:
                 print(f"[DEBUG] Skipping job {job_name} due to previous failure.")
                 continue
                 
            self.current_job = job_name
            print(f"[DEBUG] Beginning processing on job: {job_name}")

            print(f"[DEBUG] Emitting load_job_signal('{job_name}')")
            job_loaded_event.clear()
            self.load_job_signal.emit(job_name)
            if not job_loaded_event.wait(timeout=180):
                 print(f"[ERROR] Timeout waiting for job '{job_name}' to load. Aborting.")
                 job_failed = True
                 continue
            print("[DEBUG] job_loaded_event received!")

            print(f"[DEBUG] Toggling record button for job '{job_name}'...")
            # Mark that this recording was initiated by the Job Manager
            self.main_window.job_manager_initiated_record = True
            self.main_window.buttonMediaRecord.toggle()

            # Wait for processing (either style) to actually start before checking for completion
            self.processing_started_event.clear()
            if not self.processing_started_event.wait(timeout=20): # Wait up to 20 seconds
                print("[ERROR] Timeout waiting for processing to start signal. Aborting job.")
                # Attempt to toggle off the record button if it got stuck toggled on
                if self.main_window.buttonMediaRecord.isChecked():
                    print("[WARN] Attempting to toggle record button off due to timeout.")
                    self.main_window.buttonMediaRecord.toggle()
                # Also attempt to stop any potentially stuck processing
                print("[WARN] Attempting to stop video processor due to timeout.")
                self.main_window.video_processor.stop_processing()
                job_failed = True
                continue # Skip to next job
                
            print("[DEBUG] JobProcessor detected processing started. Proceeding to wait for completion.")
            self.wait_for_processing_to_complete()

            if not job_failed:
                job_path = os.path.join(self.jobs_dir, f"{job_name}.json")
                completed_path = os.path.join(self.completed_dir, f"{job_name}.json")
                if os.path.exists(job_path):
                    try:
                        shutil.move(job_path, completed_path)
                        print(f"[DEBUG] Moved job '{job_name}' to completed folder.")
                        self.job_completed_signal.emit(job_name)
                    except Exception as e:
                        print(f"[ERROR] Failed to move job {job_name} to completed: {e}")
                else:
                    print(f"[WARN] Job file not found after processing: {job_path}. Skipping move.")
            else:
                 print(f"[DEBUG] Job {job_name} failed or was skipped, not moving to completed.")

        print("[DEBUG] Finished processing all jobs loop.")
        if job_failed:
             print("[DEBUG] One or more jobs failed or were skipped.")
        self.all_jobs_done_signal.emit()

    def wait_for_processing_to_complete(self):
        """Waits until video processing (either style) has stopped by monitoring processor flags."""
        print(f"[DEBUG] wait_for_processing_to_complete() waiting for processing to finish...")
        while self.main_window.video_processor.processing:
            # Check both flags relevant to the two recording modes
            if not self.main_window.video_processor.is_processing_segments and not self.main_window.video_processor.recording:
                # If neither recording flag is active, but processing is still True, 
                # it might be in a cleanup phase. Wait for processing flag itself.
                pass 
            self.msleep(500) # Check every 500ms
        print(f"[DEBUG] Processing finished (processing flag is False) for job: {self.current_job}")

def start_processing_all_jobs(main_window: "MainWindow"):
    """Starts processing all jobs in sequence."""
    print("[DEBUG] Entered start_processing_all_jobs...")
    main_window.job_processor = JobProcessor(main_window)
    print("[DEBUG] Connecting signals in start_processing_all_jobs...")
    main_window.job_processor.load_job_signal.connect(lambda job_name: load_job_by_name(main_window, job_name))
    main_window.job_processor.job_completed_signal.connect(lambda: refresh_job_list(main_window))
    main_window.job_processor.all_jobs_done_signal.connect(
        lambda:
            QtWidgets.QMessageBox.information(
                main_window, "Job Processing Complete", "All jobs finished processing."
            )
    )
    print("[DEBUG] About to start job_processor thread...")
    main_window.job_processor.start()
    print("[DEBUG] Exiting start_processing_all_jobs...")