from typing import TYPE_CHECKING, Any
from pathlib import Path
import traceback

import numpy as np
import torch
from torchvision.transforms import v2
from PIL import Image

from PySide6 import QtWidgets
import qdarktheme

if TYPE_CHECKING:
    from app.ui.main_ui import MainWindow
from app.processors.utils import platform_support
from app.ui.widgets.actions import common_actions as common_widget_actions

#'''
#    Define functions here that has to be executed when value of a control widget (In the settings tab) is changed.
#    The first two parameters should be the MainWindow object and the new value of the control
#'''


def handle_face_detector_tracking_reset(main_window: "MainWindow", value: bool) -> None:
    """Resets the tracker instance when tracking is toggled or media changes."""
    main_window.function_worker.reset_face_tracker()
    # When ByteTrack is disabled, reset its child toggle so it doesn't stay True
    # while hidden (parentToggle mechanism only hides the widget, it doesn't reset the value).
    if not value:
        main_window.control["ShowByteTrackBBoxToggle"] = False
        widget = main_window.parameter_widgets.get("ShowByteTrackBBoxToggle")
        if widget is not None:
            widget.blockSignals(True)
            widget.setChecked(False)
            widget.blockSignals(False)
    common_widget_actions.refresh_frame(main_window)


def clear_gpu_memory(main_window: "MainWindow"):
    """
    Centralized VRAM management. Stops processing, unloads all backend models,
    resets UI toggles, and safely reconstructs Denoiser KV maps if required.
    """
    main_window.video_processor.stop_processing()
    main_window.function_worker.clear_gpu_memory()
    main_window.swapfacesButton.setChecked(False)
    main_window.editFacesButton.setChecked(False)

    # --- FW-VRAM-RECOVERY: Automatically restore KV maps safely ---
    # Centralized recovery: restores KV maps if Denoiser is active so the
    # next frame swap doesn't fail due to missing PyTorch tensor references.
    denoiser_on = (
        main_window.control.get("DenoiserUNetEnableBeforeRestorersToggle", False)
        or main_window.control.get("DenoiserAfterFirstRestorerToggle", False)
        or main_window.control.get("DenoiserAfterRestorersToggle", False)
    )

    if (
        denoiser_on
        and hasattr(main_window, "target_faces")
        and main_window.target_faces
    ):
        print("[INFO] Denoiser is active. Restoring KV maps after VRAM clear...")
        for target_face in main_window.target_faces.values():
            if hasattr(target_face, "calculate_assigned_input_embedding"):
                target_face.calculate_assigned_input_embedding()

        # Refresh context pointers for the active face so the UI tracks the newly allocated tensors
        selected_id = getattr(main_window, "selected_target_face_id", None)
        if selected_id:
            active_face = main_window.target_faces.get(selected_id)
            if active_face and hasattr(active_face, "assigned_kv_map"):
                main_window.current_kv_tensors_map = active_face.assigned_kv_map

    common_widget_actions.update_gpu_memory_progressbar(main_window)

    # main_window.videoSeekSlider.markers = set() # Comment this to keep markers visible after vram clear
    main_window.videoSeekSlider.update()

    # Push a fresh frame to ensure the UI updates and models are JIT-loaded securely
    common_widget_actions.refresh_frame(main_window)


def change_execution_provider(
    main_window: "MainWindow", new_provider: str | None = None
) -> None:
    """
    Changes the global execution provider.
    If new_provider is omitted (e.g., during startup initialization), it safely
    falls back to reading the current state from the main_window's control dictionary.

    A provider the current machine cannot supply is downgraded to the best local
    one rather than raising. Workspaces are portable, so a file saved on an
    NVIDIA box will ask a Mac for "TensorRT" or "CUDA"; that should not be fatal.
    """
    supported = platform_support.available_execution_providers()
    fallback = platform_support.default_execution_provider()

    if new_provider is None:
        new_provider = str(
            main_window.control.get("ProvidersPrioritySelection", fallback)
        )

    if new_provider not in supported:
        print(
            f"[WARN] Execution provider '{new_provider}' is not available on this "
            f"machine. Falling back to '{fallback}'."
        )
        new_provider = fallback
        main_window.control["ProvidersPrioritySelection"] = new_provider
        provider_widget = main_window.parameter_widgets.get(
            "ProvidersPrioritySelection"
        )
        if provider_widget and hasattr(provider_widget, "setCurrentText"):
            provider_widget.setCurrentText(new_provider)

    main_window.video_processor.stop_processing()
    main_window.function_worker.switch_providers_priority(new_provider)

    # Route directly through the centralized UI memory cleaner
    # to guarantee VRAM is wiped and all dependent KV maps are safely restored.
    clear_gpu_memory(main_window)


def change_threads_number(main_window: "MainWindow", new_threads_number: int) -> None:
    main_window.video_processor.set_number_of_threads(new_threads_number)
    platform_support.empty_cache()
    common_widget_actions.update_gpu_memory_progressbar(main_window)


def change_theme(main_window: "MainWindow", new_theme):
    def get_style_data(filename, theme="dark", custom_colors=None):
        custom_colors = custom_colors or {"primary": "#4090a3"}
        styles_dir = Path(__file__).resolve().parent.parent.parent / "styles"
        with open(styles_dir / filename, "r") as f:  # pylint: disable=unspecified-encoding
            _style = f.read()
            _style = (
                qdarktheme.load_stylesheet(theme=theme, custom_colors=custom_colors)
                + "\n"
                + _style
            )
        return _style

    app = QtWidgets.QApplication.instance()

    _style = ""
    if new_theme == "Dark":
        _style = get_style_data(
            "dark_styles.qss",
            "dark",
        )
    elif new_theme == "Light":
        _style = get_style_data(
            "light_styles.qss",
            "light",
        )
    elif new_theme == "Dark-Blue":
        _style = get_style_data(
            "dark_blue.qss",
            "dark",
        )
    elif new_theme == "True-Dark":
        _style = get_style_data("true_dark.qss", "dark")
    elif new_theme == "OLED-Black":
        _style = get_style_data("oled_black.qss", "dark")
    elif new_theme == "Windows11-Dark":
        _style = get_style_data(
            "windows11_dark.qss",
            "dark",
            {"primary": "#4cc2ff"},
        )
    elif new_theme == "Solarized-Dark":
        _style = get_style_data("solarized_dark.qss", "dark")
    elif new_theme == "Solarized-Light":
        _style = get_style_data("solarized_light.qss", "light")
    elif new_theme == "Dracula":
        _style = get_style_data("dracula.qss", "dark")
    elif new_theme == "Nord":
        _style = get_style_data("nord.qss", "dark")
    elif new_theme == "Gruvbox":
        _style = get_style_data("gruvbox.qss", "dark")
    elif new_theme == "Monokai":
        _style = get_style_data("monokai.qss", "dark")

    app.setStyleSheet(_style)
    main_window._vram_high_style_active = None
    common_widget_actions.update_gpu_memory_progressbar(main_window)
    main_window.update()


def set_video_playback_fps(main_window: "MainWindow", set_video_fps=False):
    # print("Called set_video_playback_fps()")
    if set_video_fps and main_window.video_processor.media_capture:
        main_window.parameter_widgets["VideoPlaybackCustomFpsSlider"].set_value(
            main_window.video_processor.fps
        )


def toggle_virtualcam(main_window: "MainWindow", toggle_value=False):
    video_processor = main_window.video_processor
    if toggle_value:
        video_processor.enable_virtualcam()
    else:
        video_processor.disable_virtualcam()


def enable_virtualcam(main_window: "MainWindow", backend):
    # Only attempt to enable if the main toggle is actually checked
    if main_window.control.get("SendVirtCamFramesEnableToggle", False):
        print("[INFO] Backend: ", backend)
        main_window.video_processor.enable_virtualcam(backend=backend)


def handle_denoiser_state_change(
    main_window: "MainWindow",
    new_value_of_toggle_that_just_changed: bool,
    control_name_that_changed: str,
) -> None:
    """
    Manages unloading of denoiser models (UNet, VAEs, KV Extractor) based on the
    overall state of all denoiser UI toggles. Models are lazy-loaded by the inference
    pipeline and unloaded here when ALL passes are disabled.
    """

    # 1. Get the current state of all relevant toggles from the UI's control dictionary.
    old_before_enabled = main_window.control.get(
        "DenoiserUNetEnableBeforeRestorersToggle", False
    )
    old_after_first_enabled = main_window.control.get(
        "DenoiserAfterFirstRestorerToggle", False
    )
    old_after_enabled = main_window.control.get("DenoiserAfterRestorersToggle", False)
    old_exclusive_path_enabled = main_window.control.get(
        "UseReferenceExclusivePathToggle", False
    )

    # 2. Determine the *new* state of all toggles by applying the incoming change.
    is_now_before_enabled = (
        new_value_of_toggle_that_just_changed
        if control_name_that_changed == "DenoiserUNetEnableBeforeRestorersToggle"
        else old_before_enabled
    )
    is_now_after_first_enabled = (
        new_value_of_toggle_that_just_changed
        if control_name_that_changed == "DenoiserAfterFirstRestorerToggle"
        else old_after_first_enabled
    )
    is_now_after_enabled = (
        new_value_of_toggle_that_just_changed
        if control_name_that_changed == "DenoiserAfterRestorersToggle"
        else old_after_enabled
    )

    # state of the exclusive path toggle is now determined
    is_now_exclusive_path_enabled = (
        new_value_of_toggle_that_just_changed
        if control_name_that_changed == "UseReferenceExclusivePathToggle"
        else old_exclusive_path_enabled
    )

    # 3. Determine if ANY denoiser pass will be active after this change.
    any_denoiser_will_be_active = (
        is_now_before_enabled or is_now_after_first_enabled or is_now_after_enabled
    )

    # 4. Handle Unloads based on the correct final state.
    if any_denoiser_will_be_active:
        print(
            "[INFO] At least one denoiser pass is active. Models will be lazy-loaded on demand."
        )
        # We do NOT eagerly load the UNet/VAEs here.
        # FaceDenoiser.apply_denoiser_unet() handles JIT loading securely.

        if not is_now_exclusive_path_enabled:
            print(
                "[INFO] Exclusive path is inactive. Ensuring KV Extractor is unloaded."
            )
            main_window.function_worker.unload_denoiser_kv_extractor()
    else:
        # If NO denoiser pass will be active, aggressively free VRAM.
        print(
            "[INFO] All denoiser passes are inactive. Unloading all denoiser-related models."
        )
        main_window.function_worker.unload_denoiser_models()
        main_window.function_worker.unload_denoiser_kv_extractor()

    # 5. Update UI visibility for the specific pass that was just toggled.
    pass_suffix_to_update = None
    if control_name_that_changed == "DenoiserUNetEnableBeforeRestorersToggle":
        pass_suffix_to_update = "Before"
    elif control_name_that_changed == "DenoiserAfterFirstRestorerToggle":
        pass_suffix_to_update = "AfterFirst"
    elif control_name_that_changed == "DenoiserAfterRestorersToggle":
        pass_suffix_to_update = "After"

    if pass_suffix_to_update:
        mode_combo_name = f"DenoiserModeSelection{pass_suffix_to_update}"
        mode_combo_widget = main_window.parameter_widgets.get(mode_combo_name)
        if mode_combo_widget:
            current_mode_text = mode_combo_widget.currentText()
            main_window.update_denoiser_controls_visibility_for_pass(
                pass_suffix_to_update, current_mode_text
            )

    # Frame refresh is handled by common_actions.update_control after this function returns.


def handle_average_kv_toggle_change(
    main_window: "MainWindow", new_value: bool, control_name: str
) -> None:
    """
    Dynamically handles the Average K/V toggle state change.

    Why: Synchronizes the main UI control dictionary before triggering VRAM tensor recalculations.
    If we do not explicitly set the control value here, the downstream `calculate_assigned_input_embedding`
    will read the out-of-sync previous state, causing inverted logic (ON triggers Concatenation, OFF triggers Averaging).
    """
    # 1. FORCE UI STATE SYNCHRONIZATION
    # Ensure the control dictionary knows the new value before calculation methods read it.
    main_window.control[control_name] = new_value

    # 2. RECALCULATE TENSOR MERGING FOR ALL TARGET FACES
    if hasattr(main_window, "target_faces") and main_window.target_faces:
        for target_face in main_window.target_faces.values():
            # Clear the downstream combined VRAM maps
            if hasattr(target_face, "assigned_kv_map"):
                target_face.assigned_kv_map = None
            if hasattr(target_face, "aged_kv_map"):
                target_face.aged_kv_map = None

            # Re-run the lightweight tensor merge (mean/cat) using the fresh toggle state
            if hasattr(target_face, "calculate_assigned_input_embedding"):
                target_face.calculate_assigned_input_embedding()

    # 3. UPDATE ACTIVE CONTEXT POINTER
    selected_id = getattr(main_window, "selected_target_face_id", None)
    if selected_id and hasattr(main_window, "target_faces"):
        active_face = main_window.target_faces.get(selected_id)
        if active_face and hasattr(active_face, "assigned_kv_map"):
            main_window.current_kv_tensors_map = active_face.assigned_kv_map

    # 4. DISPATCH UI THREAD RENDER
    common_widget_actions.refresh_frame(main_window)


def handle_face_mask_state_change(
    main_window: "MainWindow", new_value: bool, control_name: str
):
    """Loads or unloads a specific face mask model based on its toggle state."""
    model_map = {
        "OccluderEnableToggle": "Occluder",
        "DFLXSegEnableToggle": "XSeg",
        "FaceParserEnableToggle": "FaceParser",
    }
    model_to_change = model_map.get(control_name)
    if not model_to_change:
        return

    if new_value:
        main_window.models_processor.load_model(model_to_change)
    else:
        main_window.models_processor.unload_model(model_to_change)


def handle_restorer_state_change(
    main_window: "MainWindow", new_value: bool, control_name: str
) -> None:
    """Loads or unloads a specific face restorer model based on its toggle state."""
    params = main_window.current_widget_parameters
    model_map = main_window.function_worker.face_restorers.model_map
    face_restorers_manager = main_window.function_worker.face_restorers

    model_type_key = None
    active_model_attr = None
    # Identify which slot is being changed and which is the "other" slot
    other_active_model_attr = None

    if control_name == "FaceRestorerEnableToggle":
        model_type_key = "FaceRestorerTypeSelection"
        active_model_attr = "active_model_slot1"
        other_active_model_attr = "active_model_slot2"
    elif control_name == "FaceRestorerEnable2Toggle":
        model_type_key = "FaceRestorerType2Selection"
        active_model_attr = "active_model_slot2"
        other_active_model_attr = "active_model_slot1"

    if not model_type_key:
        return

    model_type = params.get(model_type_key)
    model_to_change = model_map.get(model_type)

    if model_to_change:
        if new_value:
            # Check if the other slot is already using this model
            other_model = (
                getattr(face_restorers_manager, other_active_model_attr, None)
                if other_active_model_attr
                else None
            )
            if model_to_change == other_model:
                print(
                    f"[WARN] Model '{model_to_change}' is already loaded by the other restorer slot. Skipping redundant load."
                )
            else:
                main_window.models_processor.load_model(model_to_change)

            if active_model_attr:
                setattr(
                    face_restorers_manager,
                    active_model_attr,
                    model_to_change,
                )
        else:
            # Check if the other slot is using this model before unloading
            other_model = (
                getattr(face_restorers_manager, other_active_model_attr, None)
                if other_active_model_attr
                else None
            )
            if model_to_change != other_model:
                main_window.models_processor.unload_model(model_to_change)
            else:
                print(
                    f"[WARN] Model '{model_to_change}' is still in use by the other restorer slot. Skipping unload."
                )
            if active_model_attr:
                setattr(face_restorers_manager, active_model_attr, None)


def handle_model_selection_change(
    main_window: "MainWindow", new_model_type: str, control_name: str
) -> None:
    """Unloads the old model and loads the new one when a selection dropdown changes."""
    params = main_window.current_widget_parameters
    model_map = main_window.function_worker.face_restorers.model_map
    face_restorers_manager = main_window.function_worker.face_restorers

    is_enabled = False
    active_model_attr = None
    old_model_name = None
    other_active_model_attr = None

    if control_name == "FaceRestorerTypeSelection":
        is_enabled = params.get("FaceRestorerEnableToggle", False)
        active_model_attr = "active_model_slot1"
        old_model_name = face_restorers_manager.active_model_slot1
        other_active_model_attr = "active_model_slot2"
    elif control_name == "FaceRestorerType2Selection":
        is_enabled = params.get("FaceRestorerEnable2Toggle", False)
        active_model_attr = "active_model_slot2"
        old_model_name = face_restorers_manager.active_model_slot2
        other_active_model_attr = "active_model_slot1"

    new_model_name = model_map.get(new_model_type)

    # Get the model currently used by the other slot
    other_model = (
        getattr(face_restorers_manager, other_active_model_attr, None)
        if other_active_model_attr
        else None
    )

    # Unload the old model only if it's different from the new one AND not in use by the other slot.
    if (
        old_model_name
        and old_model_name != new_model_name
        and old_model_name != other_model
    ):
        main_window.models_processor.unload_model(old_model_name)

    # If the enhancer is enabled, load the new model, but only if it's not already loaded by the other slot.
    if is_enabled and new_model_name:
        if new_model_name != other_model:
            main_window.models_processor.load_model(new_model_name)
        else:
            print(
                f"[WARN] Model '{new_model_name}' is already loaded by the other restorer slot. Skipping redundant load."
            )

        if active_model_attr:
            setattr(
                face_restorers_manager,
                active_model_attr,
                new_model_name,
            )
    elif active_model_attr:
        setattr(face_restorers_manager, active_model_attr, None)


def handle_landmark_state_change(
    main_window: "MainWindow", new_value: bool, control_name: str
) -> None:
    """Loads/Unloads landmark models when the main toggle is changed."""
    models_processor = main_window.models_processor
    function_worker = main_window.function_worker
    landmark_detectors = function_worker.face_landmark_detectors

    if not new_value:
        # Toggle is OFF: Unload all landmark models EXCEPT essential ones (like 203)
        print(
            "[INFO] Landmark detection disabled. Unloading non-essential landmark models."
        )
        landmark_detectors.unload_models(keep_essential=True)

        # Clear the state variable, *unless* the current model is the essential one
        MODEL_203_NAME = "FaceLandmark203"
        if landmark_detectors.current_landmark_model_name != MODEL_203_NAME:
            landmark_detectors.current_landmark_model_name = None
    else:
        # Toggle is ON: Load the currently selected model from the dropdown
        from app.processors.models_data import landmark_model_mapping

        current_selection = main_window.control.get(
            "LandmarkDetectModelSelection", "203"
        )
        model_to_load = landmark_model_mapping.get(str(current_selection))

        if model_to_load:
            print(
                f"[INFO] Landmark detection enabled. Loading selected model: {model_to_load}"
            )
            models_processor.load_model(model_to_load)
            landmark_detectors.active_landmark_models.add(model_to_load)
            landmark_detectors.current_landmark_model_name = model_to_load


def handle_landmark_model_selection_change(
    main_window: "MainWindow", new_detect_mode: str, control_name: str
) -> None:
    """Unloads the old landmark model and loads the new one."""
    from app.processors.models_data import landmark_model_mapping

    is_enabled = main_window.control.get("LandmarkDetectToggle", False)
    new_model_name = landmark_model_mapping.get(new_detect_mode)

    if not new_model_name:
        return  # Invalid selection

    models_processor = main_window.models_processor
    function_worker = main_window.function_worker
    landmark_detectors = function_worker.face_landmark_detectors

    old_model_name = landmark_detectors.current_landmark_model_name

    # Special case: Model 203 is used by Face Editor/Expression Restorer
    MODEL_203_NAME = "FaceLandmark203"

    # Unload the old model, IF it's different, AND it's not model 203
    if (
        old_model_name
        and old_model_name != new_model_name
        and old_model_name != MODEL_203_NAME
    ):
        print(f"[INFO] Unloading previously selected landmark model: {old_model_name}")
        models_processor.unload_model(old_model_name)
        # We also need to remove it from the active_landmark_models set
        if old_model_name in landmark_detectors.active_landmark_models:
            landmark_detectors.active_landmark_models.remove(old_model_name)

    # If the main toggle is enabled, load the new model
    if is_enabled:
        print(f"[INFO] Loading selected landmark model: {new_model_name}")
        models_processor.load_model(new_model_name)
        landmark_detectors.active_landmark_models.add(new_model_name)

    # Update the state variable to remember the new model
    landmark_detectors.current_landmark_model_name = new_model_name


def handle_frame_enhancer_state_change(
    main_window: "MainWindow", new_value: bool, control_name: str
) -> None:
    """Loads or unloads the currently selected frame enhancer model."""
    frame_enhancers = main_window.function_worker.frame_enhancers

    if new_value:
        # Get the currently selected enhancer type from the UI controls
        enhancer_type = main_window.control.get("FrameEnhancerTypeSelection")
        if enhancer_type:
            model_to_load = frame_enhancers.model_map.get(enhancer_type)
            if model_to_load:
                # Load only the selected model
                main_window.models_processor.load_model(model_to_load)
                frame_enhancers.current_enhancer_model = model_to_load
    else:
        # Unload the currently active model
        frame_enhancers.unload_models()


def handle_enhancer_model_selection_change(
    main_window: "MainWindow", new_enhancer_type: str, control_name: str
) -> None:
    """Unloads the old enhancer model and loads the new one when the selection changes."""
    frame_enhancers = main_window.function_worker.frame_enhancers
    is_enabled = main_window.control.get("FrameEnhancerEnableToggle", False)

    # Get the actual ONNX model name from the user-friendly type
    new_model_name = frame_enhancers.model_map.get(new_enhancer_type)
    old_model_name = frame_enhancers.current_enhancer_model

    # Unload the old model if it's different from the new one
    if old_model_name and old_model_name != new_model_name:
        main_window.models_processor.unload_model(old_model_name)

    # If the enhancer is enabled, load the new model
    if is_enabled and new_model_name:
        main_window.models_processor.load_model(new_model_name)
        frame_enhancers.current_enhancer_model = new_model_name
    else:
        # If disabled, just ensure the current model is cleared
        frame_enhancers.current_enhancer_model = new_model_name


def _check_and_manage_face_editor_models(main_window: "MainWindow") -> None:
    """
    Central function to load/unload FaceEditor (LivePortrait) models
    based on the state of BOTH UI controls.
    """
    models_processor = main_window.models_processor
    function_worker = main_window.function_worker

    # 1. Check if the main 'Edit Face' button (outside the tab) is checked
    is_edit_face_active = main_window.editFacesButton.isChecked()

    # 2. Check if the 'Enable Face Pose/Expression Editor' parameter toggle (inside the tab) is active
    # We read from 'current_widget_parameters' to get the most up-to-date UI state
    is_face_editor_param_active = main_window.current_widget_parameters.get(
        "FaceEditorEnableToggle", False
    )

    # 3. Check if the 'Enable Face Expression Restorer' parameter toggle is active
    is_expr_restore_active = main_window.current_widget_parameters.get(
        "FaceExpressionEnableBothToggle", False
    )

    # The 'Edit Face' feature is only *truly* active if BOTH its buttons are on.
    true_edit_active = is_edit_face_active and is_face_editor_param_active

    # Any LivePortrait feature is active if (Edit Face is fully on) OR (Expression Restore is on)
    any_editor_feature_active = true_edit_active or is_expr_restore_active

    # Check the *actual* loaded state from the face_editors module. The
    # PerformRecast ("Recast" mode) models are tracked separately because they
    # are not part of the LivePortrait face-editor group.
    recast_loaded = any(
        models_processor.models.get(m) is not None
        for m in function_worker.perform_recast.model_group
    )
    models_are_currently_loaded = (
        function_worker.face_editors.current_face_editor_type is not None
        or recast_loaded
    )

    if any_editor_feature_active and not models_are_currently_loaded:
        # A feature is ON, but models are OFF.
        # We don't need to do anything here. The lazy-loader in
        # FrameWorker/FaceEditors will load them on first use.
        print(
            "[INFO] Face Editor/Expression Restorer is active. Models will be lazy-loaded on use."
        )
        pass
    elif not any_editor_feature_active and models_are_currently_loaded:
        # NO feature is ON, but models *are* loaded. Unload them.
        print(
            "[INFO] Face Editor and Expression Restorer are inactive. Unloading LivePortrait models."
        )
        function_worker.unload_face_editor_models()
        if recast_loaded:
            function_worker.unload_perform_recast_models()


def handle_face_editor_button_click(main_window: "MainWindow"):
    """Called when the 'Edit Faces' button is clicked."""
    # This function is called by the button click signal.
    # We just need to check the overall state.
    _check_and_manage_face_editor_models(main_window)


def handle_face_expression_toggle_change(
    main_window: "MainWindow", new_value: bool, control_name: str
):
    """Called when the 'FaceExpressionEnableBothToggle' parameter changes."""
    # This function is called by the parameter change.
    # We just need to check the overall state.
    _check_and_manage_face_editor_models(main_window)


def _set_parameter_toggle_state(
    main_window: "MainWindow", parameter_name: str, value: bool
) -> bool:
    widget = main_window.parameter_widgets.get(parameter_name)
    if widget is None or widget.isChecked() == value:
        return False

    widget.blockSignals(True)
    widget.setChecked(value)
    widget.blockSignals(False)

    common_widget_actions.update_parameter(
        main_window,
        parameter_name,
        value,
        enable_refresh_frame=False,
    )
    common_widget_actions.show_hide_related_widgets(
        main_window,
        widget,
        parameter_name,
    )
    return True


def handle_face_expression_eye_blend_toggle(
    main_window: "MainWindow", new_value: bool, control_name: str
) -> None:
    """
    Keeps the Relative Lids + Retargeted Gaze eye mode in a valid state.

    - Enabling the combined eye mode auto-enables Relative Position and
      Retargeting Eyes.
    - Disabling either dependency auto-disables the combined eye mode.
    """
    stable_toggle = "FaceExpressionStableGazeEyesToggle"
    relative_toggle = "FaceExpressionRelativeEyesToggle"
    retarget_toggle = "FaceExpressionRetargetingEyesBothEnableToggle"
    changed_dependency_state = False

    if control_name == stable_toggle and new_value:
        changed_dependency_state |= _set_parameter_toggle_state(
            main_window, relative_toggle, True
        )
        changed_dependency_state |= _set_parameter_toggle_state(
            main_window, retarget_toggle, True
        )
    elif control_name in {relative_toggle, retarget_toggle} and not new_value:
        changed_dependency_state |= _set_parameter_toggle_state(
            main_window, stable_toggle, False
        )

    if changed_dependency_state:
        common_widget_actions.refresh_frame(main_window)


def apply_face_reaging(main_window: "MainWindow", *_args) -> None:
    """Apply age transformation to the assigned input face for the selected target face.

    Re-computes ArcFace embeddings (and optionally the denoiser KV map) from the
    age-transformed face image, storing the results on the target face button so
    frame_worker can use them when FaceReagingEnableToggle is active.
    """
    # Guard: do nothing if Swap Faces is not active
    if not main_window.swapfacesButton.isChecked():
        return

    target_face = main_window.cur_selected_target_face_button
    if target_face is None:
        return

    face_id = target_face.face_id
    params: Any = main_window.parameters.get(face_id, {})

    # Guard: do nothing if the toggle is currently disabled
    if not params.get("FaceReagingEnableToggle", False):
        return

    if not target_face.assigned_input_faces:
        print(
            "[WARN] apply_face_reaging: No input face assigned to the selected target face."
        )
        return
    source_age = int(params.get("FaceReagingSourceAgeSlider", 25))
    target_age_val = int(params.get("FaceReagingTargetAgeSlider", 70))

    first_input_id = list(target_face.assigned_input_faces.keys())[0]
    input_face_button = main_window.input_faces.get(first_input_id)
    if input_face_button is None:
        print("[WARN] apply_face_reaging: Input face button not found.")
        return

    cropped_face_bgr = input_face_button.cropped_face
    if cropped_face_bgr is None or cropped_face_bgr.size == 0:
        print("[WARN] apply_face_reaging: Input face has no cropped image.")
        return

    try:
        models_processor = main_window.models_processor
        function_worker = main_window.function_worker

        # BGR numpy → RGB CHW uint8 tensor
        face_rgb_np = np.ascontiguousarray(cropped_face_bgr[..., ::-1])
        face_chw = torch.from_numpy(face_rgb_np).permute(2, 0, 1)  # CHW uint8

        # Ensure 512 × 512
        if face_chw.shape[1] != 512 or face_chw.shape[2] != 512:
            face_chw = v2.Resize((512, 512), antialias=False)(face_chw)

        # Run re-aging
        aged_chw = function_worker.apply_reaging(
            face_chw, source_age, target_age_val
        )  # CHW uint8 RGB

        # Move to the device for recognition
        aged_chw_dev = aged_chw.to(models_processor.device)

        # Approximate 5-point keypoints for the 512 × 512 face crop
        h, w = aged_chw.shape[1], aged_chw.shape[2]
        approx_kps_5 = np.array(
            [
                [w * 0.3, h * 0.40],  # left eye
                [w * 0.7, h * 0.40],  # right eye
                [w * 0.5, h * 0.55],  # nose
                [w * 0.35, h * 0.70],  # left mouth corner
                [w * 0.65, h * 0.70],  # right mouth corner
            ],
            dtype=np.float32,
        )

        similarity_type = str("Auto")

        # Determine which arcface models to recompute. Filter strictly to known
        # arcface model names so that non-model keys (e.g. "kps_5") that may also
        # live inside assigned_input_embedding are never passed to run_recognize_direct.
        # Local import kept to prevent circular dependency with models_data
        from app.processors.models_data import arcface_mapping_model_dict

        _valid_arcface_models: set = set(arcface_mapping_model_dict.values())

        # Guard against None / empty assigned_input_embedding
        _assigned = target_face.assigned_input_embedding
        models_to_compute = (
            (set(_assigned.keys()) & _valid_arcface_models) if _assigned else set()
        )
        if not models_to_compute:
            models_to_compute = _valid_arcface_models

        aged_embeddings = {}
        for arcface_model in models_to_compute:
            try:
                embedding, _ = function_worker.run_recognize_direct(
                    aged_chw_dev, approx_kps_5, similarity_type, arcface_model
                )
                if embedding is not None and embedding.size > 0:
                    aged_embeddings[arcface_model] = embedding
            except Exception as e_emb:
                print(
                    f"[WARN] apply_face_reaging: embedding for '{arcface_model}' failed: {e_emb}"
                )

        target_face.aged_input_embedding = aged_embeddings

        # Recompute KV map if denoiser is active
        control = main_window.control
        denoiser_on = (
            control.get("DenoiserUNetEnableBeforeRestorersToggle", False)
            or control.get("DenoiserAfterFirstRestorerToggle", False)
            or control.get("DenoiserAfterRestorersToggle", False)
        )

        if denoiser_on:
            try:
                aged_hwc = aged_chw.permute(1, 2, 0).cpu().numpy()
                pil_img = Image.fromarray(aged_hwc)
                with function_worker.denoiser_kv_extraction_lock:
                    kv_map = function_worker.get_kv_map_for_face(pil_img)
                target_face.aged_kv_map = kv_map
            except Exception as e_kv:
                print(f"[ERROR] apply_face_reaging: KV map extraction failed: {e_kv}")
                target_face.aged_kv_map = None
        else:
            target_face.aged_kv_map = None

        print(
            f"[INFO] apply_face_reaging: Applied (source={source_age}, target={target_age_val}) "
            f"for face {face_id}. Embeddings computed for: {list(aged_embeddings.keys())}"
        )

        common_widget_actions.refresh_frame(main_window)

    except Exception as exc:
        print(f"[ERROR] apply_face_reaging: {exc}")
        traceback.print_exc()


def update_video_slider_step(main_window: "MainWindow", new_step) -> None:
    """Updates the video slider step size dynamically from the settings slider."""
    if hasattr(main_window, "videoSeekSlider"):
        step_val = int(new_step)
        main_window.videoSeekSlider.setSingleStep(step_val)
        main_window.videoSeekSlider.setPageStep(step_val)


def handle_face_reaging_toggle_change(
    main_window: "MainWindow", new_value: bool
) -> None:
    """Clear aged embedding/KV map when Face Re-Aging toggle is disabled.

    When the toggle is turned off the original (non-aged) embedding and KV map
    should be used immediately, so we wipe the cached aged data and refresh.
    """
    if new_value:
        # Toggle just enabled — user still needs to press Apply, nothing to clear.
        return
    target_face = main_window.cur_selected_target_face_button
    if target_face is None:
        return
    target_face.aged_input_embedding = {}
    target_face.aged_kv_map = None
    common_widget_actions.refresh_frame(main_window)


def handle_auto_mouth_toggle(main_window: "MainWindow", new_value: bool) -> None:
    """Called when AutoMouthExpressionEnableToggle changes.

    When enabled, prints a one-time status message.  The actual detection model
    is loaded lazily on the first processed frame.
    """
    if not new_value:
        return

    from app.processors.mouth_action_detector import MouthActionDetector

    detector = MouthActionDetector.get()
    if not detector.available:
        err = detector.load_error or "unknown error"
        print(f"[WARN] Auto Mouth Expression: detector unavailable — {err}")
    else:
        print("[INFO] Auto Mouth Expression enabled. Mouth action detector ready.")


def handle_ff_auto_quality_toggle(main_window: "MainWindow", new_value: bool) -> None:
    """Disable manual FFQualitySlider while auto source-quality matching is enabled."""
    quality_widget = main_window.parameter_widgets.get("FFQualitySlider")
    if quality_widget is None:
        return

    manual_enabled = not bool(new_value)

    quality_widget.setEnabled(manual_enabled)
    if hasattr(quality_widget, "row_widget") and quality_widget.row_widget:
        quality_widget.row_widget.setEnabled(manual_enabled)
    if hasattr(quality_widget, "label_widget") and quality_widget.label_widget:
        quality_widget.label_widget.setEnabled(manual_enabled)
    if (
        hasattr(quality_widget, "reset_default_button")
        and quality_widget.reset_default_button
    ):
        quality_widget.reset_default_button.setEnabled(manual_enabled)
    if hasattr(quality_widget, "line_edit") and quality_widget.line_edit:
        quality_widget.line_edit.setEnabled(manual_enabled)


def handle_sort_embeddings_az_toggle(
    main_window: "MainWindow", new_value: bool, *args
) -> None:
    if not new_value:
        return
    try:
        from app.ui.widgets.actions import list_view_actions

        list_view_actions.sort_embeddings_list_az(main_window)
    except Exception as e:
        print(f"[ERROR] sort embeddings: {e}")
        import traceback

        traceback.print_exc()


def handle_seek_bar_thumbnails_toggle(main_window: "MainWindow", enabled: bool):
    thumb = None
    composite = None

    for attr in ("compositeTimeline", "composite_timeline", "timelineWidget"):
        composite = getattr(main_window, attr, None)
        if composite is not None and hasattr(composite, "thumbnail_track"):
            thumb = composite.thumbnail_track
            break

    if thumb is None:
        slider = getattr(main_window, "videoSeekSlider", None)
        if slider is not None:
            parent = slider.parent()
            if parent is not None and hasattr(parent, "thumbnail_track"):
                composite = parent
                thumb = parent.thumbnail_track

    if thumb is None:
        return

    layout = getattr(composite, "layout", None) if composite is not None else None
    if layout is not None and not hasattr(layout, "setSpacing"):
        layout = None

    scroll_area = getattr(main_window, "timelineScrollArea", None)

    if enabled:
        if layout is not None:
            layout.setSpacing(2)
            layout.setContentsMargins(0, 0, 0, 0)
        thumb.setFixedHeight(40)
        thumb.setMaximumHeight(40)
        thumb.setVisible(True)
        if composite is not None:
            composite.setMinimumHeight(0)
            composite.setMaximumHeight(16777215)
        if scroll_area is not None:
            scroll_area.setMinimumHeight(0)
            scroll_area.setMaximumHeight(16777215)
        thumb.request_thumbnails()
    else:
        if getattr(thumb, "worker", None) and thumb.worker.isRunning():
            thumb.worker.cancel()
            thumb.worker.wait()
        thumb.thumbnail_cache.clear()
        thumb.expected_intervals.clear()
        thumb.setVisible(False)
        thumb.setFixedHeight(0)
        thumb.setMaximumHeight(0)
        if layout is not None:
            layout.setSpacing(0)
            layout.setContentsMargins(0, 0, 0, 0)
        if composite is not None:
            # Keep only the slider height (~20-28px typical)
            slider = getattr(composite, "slider", None) or getattr(
                main_window, "videoSeekSlider", None
            )
            h = slider.height() if slider is not None else 24
            composite.setFixedHeight(h)
            composite.setMaximumHeight(h)
            composite.adjustSize()
            composite.updateGeometry()
        if scroll_area is not None:
            h = composite.height() if composite is not None else 24
            scroll_area.setFixedHeight(h)
            scroll_area.setMaximumHeight(h)
            scroll_area.adjustSize()
            scroll_area.updateGeometry()
        thumb.update()

def reset_face_editor_expression_params(main_window: "MainWindow"):
    from app.ui.widgets.actions import common_actions as common_widget_actions

    defaults = {
        "EyesOpenRatioDecimalSlider": 0.00,
        "LipsOpenRatioDecimalSlider": 0.00,
        "HeadPitchSlider": 0,
        "HeadYawSlider": 0,
        "HeadRollSlider": 0,
        "XAxisMovementDecimalSlider": 0.00,
        "YAxisMovementDecimalSlider": 0.00,
        "ZAxisMovementDecimalSlider": 1.00,
        "MouthPoutingDecimalSlider": 0.00,
        "MouthPursingDecimalSlider": 0.00,
        "MouthGrinDecimalSlider": 0.00,
        "LipsCloseOpenSlider": 0,
        "MouthSmileDecimalSlider": 0.00,
        "EyeWinkDecimalSlider": 0.00,
        "EyeBrowsDirectionDecimalSlider": 0.00,
        "EyeGazeHorizontalDecimalSlider": 0.00,
        "EyeGazeVerticalDecimalSlider": 0.00,
    }

    previous_batch = getattr(main_window, "_batch_update_in_progress", False)
    main_window._batch_update_in_progress = True
    try:
        for name, value in defaults.items():
            common_widget_actions.update_parameter(
                main_window, name, value, enable_refresh_frame=False
            )
            widget = main_window.parameter_widgets.get(name)
            if widget:
                widget.enable_refresh_frame = False
                common_widget_actions._set_single_widget_value(widget, value)
                widget.enable_refresh_frame = True
    finally:
        main_window._batch_update_in_progress = previous_batch
        if not previous_batch:
            common_widget_actions.refresh_frame(main_window)
