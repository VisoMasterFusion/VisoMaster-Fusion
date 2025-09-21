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

def change_execution_provider(main_window: 'MainWindow', new_provider):
    main_window.video_processor.stop_processing()
    main_window.models_processor.switch_providers_priority(new_provider)
    main_window.models_processor.clear_gpu_memory()
    common_widget_actions.update_gpu_memory_progressbar(main_window)

def change_threads_number(main_window: 'MainWindow', new_threads_number):
    main_window.video_processor.set_number_of_threads(new_threads_number)
    torch.cuda.empty_cache()
    common_widget_actions.update_gpu_memory_progressbar(main_window)

def change_theme(main_window: 'MainWindow', new_theme):

    def get_style_data(filename, theme='dark', custom_colors=None):
        custom_colors = custom_colors or {"primary": "#4090a3"}
        with open(f"app/ui/styles/{filename}", "r") as f: # pylint: disable=unspecified-encoding
            _style = f.read()
            _style = qdarktheme.load_stylesheet(theme=theme, custom_colors=custom_colors)+'\n'+_style
        return _style
    app = QtWidgets.QApplication.instance()

    _style = ''
    if new_theme == "Dark":
        _style = get_style_data('dark_styles.qss', 'dark',)
    elif new_theme == "Light":
        _style = get_style_data('light_styles.qss', 'light',)
    elif new_theme == "Dark-Blue":
        _style = get_style_data('dark_styles.qss', 'dark',) + qdarkstyle.load_stylesheet() 
    elif new_theme == "True-Dark":
        _style = get_style_data('true_dark.qss', 'dark')
    elif new_theme == "Solarized-Dark":
        _style = get_style_data('solarized_dark.qss', 'dark')
    elif new_theme == "Solarized-Light":
        _style = get_style_data('solarized_light.qss', 'light')
    elif new_theme == "Dracula":
        _style = get_style_data('dracula.qss', 'dark')
    elif new_theme == "Nord":
        _style = get_style_data('nord.qss', 'dark')
    elif new_theme == "Gruvbox":
        _style = get_style_data('gruvbox.qss', 'dark')

    app.setStyleSheet(_style)
    main_window.update()

def set_video_playback_fps(main_window: 'MainWindow', set_video_fps=False):
    # print("Called set_video_playback_fps()")
    if set_video_fps and main_window.video_processor.media_capture:
        main_window.parameter_widgets['VideoPlaybackCustomFpsSlider'].set_value(main_window.video_processor.fps)

def toggle_virtualcam(main_window: 'MainWindow', toggle_value=False):
    video_processor = main_window.video_processor
    if toggle_value:
        video_processor.enable_virtualcam()
    else:
        video_processor.disable_virtualcam()

def enable_virtualcam(main_window: 'MainWindow', backend):
    print('backend', backend)
    main_window.video_processor.enable_virtualcam(backend=backend)

def handle_denoiser_state_change(main_window: 'MainWindow', new_value_of_toggle_that_just_changed: bool, control_name_that_changed: str):
    """
    Manages loading/unloading of denoiser models (UNet, VAEs) based on UI toggle states.
    The actual frame refresh is handled by the `update_control` function after this.
    """
    # Determine the state of denoisers *as they were* before this change
    # main_window.control still holds the old values for all controls at this point within exec_function
    old_before_enabled = main_window.control.get('DenoiserUNetEnableBeforeRestorersToggle', False)
    old_after_first_enabled = main_window.control.get('DenoiserAfterFirstRestorerToggle', False)
    old_after_enabled = main_window.control.get('DenoiserAfterRestorersToggle', False)
    denoiser_was_active = old_before_enabled or old_after_first_enabled or old_after_enabled

    # Determine the state of denoisers *as they will be* after this change
    is_now_before_enabled = old_before_enabled # Default to old state
    is_now_after_enabled = old_after_enabled   # Default to old state
    is_now_after_first_enabled = old_after_first_enabled  # Default to old state

    if control_name_that_changed == 'DenoiserUNetEnableBeforeRestorersToggle':
        is_now_before_enabled = new_value_of_toggle_that_just_changed
    elif control_name_that_changed == 'DenoiserAfterFirstRestorerToggle':
        is_now_after_first_enabled = new_value_of_toggle_that_just_changed
    elif control_name_that_changed == 'DenoiserAfterRestorersToggle':
        is_now_after_enabled = new_value_of_toggle_that_just_changed
    
    any_denoiser_will_be_active = is_now_before_enabled or is_now_after_first_enabled or is_now_after_enabled
    
    if any_denoiser_will_be_active:
        main_window.models_processor.ensure_kv_extractor_loaded()
        main_window.models_processor.ensure_denoiser_models_loaded()
        # If a denoiser section was just activated, update its control visibility
        pass_suffix_to_update = None
        if control_name_that_changed == 'DenoiserUNetEnableBeforeRestorersToggle' and new_value_of_toggle_that_just_changed:
            pass_suffix_to_update = "Before"
        elif control_name_that_changed == 'DenoiserAfterFirstRestorerToggle' and new_value_of_toggle_that_just_changed:
            pass_suffix_to_update = "AfterFirst"
        elif control_name_that_changed == 'DenoiserAfterRestorersToggle' and new_value_of_toggle_that_just_changed:
            pass_suffix_to_update = "After"

        if pass_suffix_to_update:
            mode_combo_name = f'DenoiserModeSelection{pass_suffix_to_update}'
            mode_combo_widget = main_window.parameter_widgets.get(mode_combo_name)
            if mode_combo_widget:
                current_mode_text = mode_combo_widget.currentText()
                main_window.update_denoiser_controls_visibility_for_pass(pass_suffix_to_update, current_mode_text)

    else: # No denoiser will be active
        if denoiser_was_active: # Was on, now off
            main_window.models_processor.unload_denoiser_models()
            main_window.models_processor.unload_kv_extractor()
    
    # Frame refresh is handled by common_actions.update_control after this function returns.