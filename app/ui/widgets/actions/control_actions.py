from typing import TYPE_CHECKING
import torch
import qdarkstyle
from PySide6 import QtWidgets
import qdarktheme

if TYPE_CHECKING:
    from app.ui.main_ui import MainWindow
from app.ui.widgets.actions import common_actions as common_widget_actions

#'''
#    Define functions here that has to be executed when value of a control widget (In the settings tab) is changed.
#    The first two parameters should be the MainWindow object and the new value of the control
#'''


def change_execution_provider(main_window: "MainWindow", new_provider):
    main_window.video_processor.stop_processing()
    main_window.models_processor.switch_providers_priority(new_provider)
    main_window.models_processor.clear_gpu_memory()
    common_widget_actions.update_gpu_memory_progressbar(main_window)


def change_threads_number(main_window: "MainWindow", new_threads_number):
    main_window.video_processor.set_number_of_threads(new_threads_number)
    torch.cuda.empty_cache()
    common_widget_actions.update_gpu_memory_progressbar(main_window)


def change_theme(main_window: "MainWindow", new_theme):
    def get_style_data(filename, theme="dark", custom_colors=None):
        custom_colors = custom_colors or {"primary": "#4090a3"}
        with open(f"app/ui/styles/{filename}", "r") as f:  # pylint: disable=unspecified-encoding
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
        _style = (
            get_style_data(
                "dark_styles.qss",
                "dark",
            )
            + qdarkstyle.load_stylesheet()
        )
    elif new_theme == "True-Dark":
        _style = get_style_data("true_dark.qss", "dark")
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

    app.setStyleSheet(_style)
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
        print("Backend: ", backend)
        main_window.video_processor.enable_virtualcam(backend=backend)


def handle_denoiser_state_change(
    main_window: "MainWindow",
    new_value_of_toggle_that_just_changed: bool,
    control_name_that_changed: str,
):
    """
    Manages loading/unloading of denoiser models (UNet, VAEs) based on UI toggle states.
    The actual frame refresh is handled by the `update_control` function after this.
    """
    # Determine the state of denoisers *as they were* before this change
    # main_window.control still holds the old values for all controls at this point within exec_function
    old_before_enabled = main_window.control.get(
        "DenoiserUNetEnableBeforeRestorersToggle", False
    )
    old_after_first_enabled = main_window.control.get(
        "DenoiserAfterFirstRestorerToggle", False
    )
    old_after_enabled = main_window.control.get("DenoiserAfterRestorersToggle", False)
    denoiser_was_active = (
        old_before_enabled or old_after_first_enabled or old_after_enabled
    )

    # Determine the state of denoisers *as they will be* after this change
    is_now_before_enabled = old_before_enabled  # Default to old state
    is_now_after_enabled = old_after_enabled  # Default to old state
    is_now_after_first_enabled = old_after_first_enabled  # Default to old state

    if control_name_that_changed == "DenoiserUNetEnableBeforeRestorersToggle":
        is_now_before_enabled = new_value_of_toggle_that_just_changed
    elif control_name_that_changed == "DenoiserAfterFirstRestorerToggle":
        is_now_after_first_enabled = new_value_of_toggle_that_just_changed
    elif control_name_that_changed == "DenoiserAfterRestorersToggle":
        is_now_after_enabled = new_value_of_toggle_that_just_changed

    any_denoiser_will_be_active = (
        is_now_before_enabled or is_now_after_first_enabled or is_now_after_enabled
    )

    if any_denoiser_will_be_active:
        main_window.models_processor.ensure_kv_extractor_loaded()
        main_window.models_processor.ensure_denoiser_models_loaded()
        # If a denoiser section was just activated, update its control visibility
        pass_suffix_to_update = None
        if (
            control_name_that_changed == "DenoiserUNetEnableBeforeRestorersToggle"
            and new_value_of_toggle_that_just_changed
        ):
            pass_suffix_to_update = "Before"
        elif (
            control_name_that_changed == "DenoiserAfterFirstRestorerToggle"
            and new_value_of_toggle_that_just_changed
        ):
            pass_suffix_to_update = "AfterFirst"
        elif (
            control_name_that_changed == "DenoiserAfterRestorersToggle"
            and new_value_of_toggle_that_just_changed
        ):
            pass_suffix_to_update = "After"

        if pass_suffix_to_update:
            mode_combo_name = f"DenoiserModeSelection{pass_suffix_to_update}"
            mode_combo_widget = main_window.parameter_widgets.get(mode_combo_name)
            if mode_combo_widget:
                current_mode_text = mode_combo_widget.currentText()
                main_window.update_denoiser_controls_visibility_for_pass(
                    pass_suffix_to_update, current_mode_text
                )

    else:  # No denoiser will be active
        if denoiser_was_active:  # Was on, now off
            main_window.models_processor.unload_denoiser_models()
            main_window.models_processor.unload_kv_extractor()

    # Frame refresh is handled by common_actions.update_control after this function returns.


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
):
    """Loads or unloads a specific face restorer model based on its toggle state."""
    params = main_window.current_widget_parameters
    model_map = main_window.models_processor.face_restorers.model_map

    slot_id = 0
    model_type_key = None
    active_model_attr = None

    if control_name == "FaceRestorerEnableToggle":
        model_type_key = "FaceRestorerTypeSelection"
        slot_id = 1
        active_model_attr = "active_model_slot1"
    elif control_name == "FaceRestorerEnable2Toggle":
        model_type_key = "FaceRestorerType2Selection"
        slot_id = 2
        active_model_attr = "active_model_slot2"

    if not model_type_key:
        return

    model_type = params.get(model_type_key)
    model_to_change = model_map.get(model_type)

    if model_to_change:
        if new_value:
            main_window.models_processor.load_model(model_to_change)
            setattr(
                main_window.models_processor.face_restorers,
                active_model_attr,
                model_to_change,
            )
        else:
            main_window.models_processor.unload_model(model_to_change)
            setattr(
                main_window.models_processor.face_restorers, active_model_attr, None
            )


def handle_model_selection_change(
    main_window: "MainWindow", new_model_type: str, control_name: str
):
    """Unloads the old model and loads the new one when a selection dropdown changes."""
    params = main_window.current_widget_parameters
    model_map = main_window.models_processor.face_restorers.model_map

    is_enabled = False
    active_model_attr = None
    old_model_name = None

    if control_name == "FaceRestorerTypeSelection":
        is_enabled = params.get("FaceRestorerEnableToggle", False)
        active_model_attr = "active_model_slot1"
        old_model_name = main_window.models_processor.face_restorers.active_model_slot1
    elif control_name == "FaceRestorerType2Selection":
        is_enabled = params.get("FaceRestorerEnable2Toggle", False)
        active_model_attr = "active_model_slot2"
        old_model_name = main_window.models_processor.face_restorers.active_model_slot2

    new_model_name = model_map.get(new_model_type)

    if old_model_name and old_model_name != new_model_name:
        main_window.models_processor.unload_model(old_model_name)

    if is_enabled and new_model_name:
        main_window.models_processor.load_model(new_model_name)
        setattr(
            main_window.models_processor.face_restorers,
            active_model_attr,
            new_model_name,
        )
    else:
        setattr(main_window.models_processor.face_restorers, active_model_attr, None)


def handle_landmark_state_change(
    main_window: "MainWindow", new_value: bool, control_name: str
):
    """Unloads landmark models if the main landmark toggle is disabled."""
    if not new_value:
        main_window.models_processor.face_landmark_detectors.unload_models()


def handle_landmark_model_selection_change(
    main_window: "MainWindow", new_detect_mode: str, control_name: str
):
    """Unloads the old landmark model and loads the new one."""
    from app.processors.models_data import landmark_model_mapping

    is_enabled = main_window.control.get("LandmarkDetectToggle", False)
    new_model_name = landmark_model_mapping.get(new_detect_mode)

    # If landmark detection is active, ensure the newly selected model is loaded.
    # It will not unload any other models, preventing the reloading issue.
    if is_enabled and new_model_name:
        main_window.models_processor.load_model(new_model_name)


def handle_frame_enhancer_state_change(
    main_window: "MainWindow", new_value: bool, control_name: str
):
    """Loads or unloads frame enhancer models."""
    if new_value:
        main_window.models_processor.frame_enhancers.ensure_models_loaded()
    else:
        main_window.models_processor.frame_enhancers.unload_models()
