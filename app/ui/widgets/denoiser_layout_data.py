from app.helpers.typing_helper import LayoutDictTypes
import app.ui.widgets.actions.layout_actions as layout_actions
import app.ui.widgets.actions.control_actions as control_actions

DENOISER_LAYOUT_DATA: LayoutDictTypes = {
    'ReF-LDM Denoiser': {
        'UseReferenceExclusivePathToggle': { # New ToggleButton
            'level': 1,
            'widget_type': 'ToggleButton',
            'label': 'Exclusive Reference Path',
            'control_name': 'UseReferenceExclusivePathToggle',
            'default': True,
            'help': 'If enabled, forces the UNet to use only reference K/V for attention, maximizing focus on the reference features.'
        },
        'DenoiserBaseSeedSlider': {
            'level': 1,
            'widget_type': 'ParameterSlider',
            'label': 'Base Seed',
            'control_name': 'DenoiserBaseSeedSlider',
            'min_value': '1', 'max_value': '999', 'default': '220', 'step': 1,
            'help': 'Set a fixed base seed for the denoiser. This seed will be used for all frames and both denoiser passes (if applicable) to ensure consistent noise patterns.'
        },
        'DenoiserUNetEnableBeforeRestorersToggle': {
            'level': 1,
            'widget_type': 'ToggleButton',
            'label': 'Enable Denoiser before Restorers',
            'control_name': 'DenoiserUNetEnableBeforeRestorersToggle',
            'default': False,
            'help': 'Enable UNet-based image denoising. This is applied to the 512x512 aligned/swapped face before other restorers.',
            'exec_function': control_actions.handle_denoiser_state_change,
            'exec_function_args': ['DenoiserUNetEnableBeforeRestorersToggle'],
        },
        'DenoiserModeSelectionBefore': {
            'level': 2,
            'widget_type': 'SelectionBox',
            'label': 'Denoiser Mode (Before)',
            'control_name': 'DenoiserModeSelectionBefore',
            'options': ["Single Step (Fast)", "Full Restore (DDIM)"],
            'default': "Single Step (Fast)",
            'parentToggle': 'DenoiserUNetEnableBeforeRestorersToggle',
            'requiredToggleValue': True,
            'help': 'Denoising mode for the pass before restorers. Single Step is generally faster.'
        },
        'DenoiserSingleStepTimestepSliderBefore': {
            'level': 3,
            'widget_type': 'ParameterSlider',
            'label': 'Single Step Timestep (t) (Before)',
            'control_name': 'DenoiserSingleStepTimestepSliderBefore',
            'min_value': '1', 'max_value': '500', 'default': '1', 'step': 1,
            'parentToggle': 'DenoiserUNetEnableBeforeRestorersToggle',
            'requiredToggleValue': True,
            'parentSelection': 'DenoiserModeSelectionBefore',
            'requiredSelectionValue': "Single Step (Fast)",
            'help': 'Timestep for single-step denoising (Before Restorers). Lower values mean less noise added/removed.'
        },
        'DenoiserDDIMStepsSliderBefore': {
            'level': 3,
            'widget_type': 'ParameterSlider',
            'label': 'DDIM Steps (Before)',
            'control_name': 'DenoiserDDIMStepsSliderBefore',
            'min_value': '1', 'max_value': '300', 'default': '10', 'step': 1,
            'parentToggle': 'DenoiserUNetEnableBeforeRestorersToggle',
            'requiredToggleValue': True,
            'parentSelection': 'DenoiserModeSelectionBefore',
            'requiredSelectionValue': "Full Restore (DDIM)",
            'help': "Number of DDIM steps for full restoration (Before Restorers). Higher = more detail, slower."
        },
        'DenoiserCFGScaleDecimalSliderBefore': {
            'level': 3,
            'widget_type': 'ParameterDecimalSlider',
            'label': 'CFG Scale (Before)',
            'control_name': 'DenoiserCFGScaleDecimalSliderBefore',
            'min_value': '0.0', 'max_value': '10.0', 'default': '1.0', 'step': 0.1, 'decimals': 1,
            'parentToggle': 'DenoiserUNetEnableBeforeRestorersToggle',
            'requiredToggleValue': True,
            'parentSelection': 'DenoiserModeSelectionBefore',
            'requiredSelectionValue': "Full Restore (DDIM)",
            'help': "Classifier-Free Guidance scale for DDIM (Before Restorers). Higher = stronger adherence to K/V."
        },
        'DenoiserAfterFirstRestorerToggle': {
            'level': 1,
            'widget_type': 'ToggleButton',
            'label': 'Enable Denoiser After first Restorer',
            'control_name': 'DenoiserAfterFirstRestorerToggle',
            'default': False,
            'help': 'Apply the UNet Denoiser again after first restorer have been applied. Uses the same UNet model and step settings.',
            'exec_function': control_actions.handle_denoiser_state_change,
            'exec_function_args': ['DenoiserAfterFirstRestorerToggle'],
        },
        'DenoiserModeSelectionAfterFirst': {
            'level': 2,
            'widget_type': 'SelectionBox',
            'label': 'Denoiser Mode (After)',
            'control_name': 'DenoiserModeSelectionAfterFirst',
            'options': ["Single Step (Fast)", "Full Restore (DDIM)"],
            'default': "Single Step (Fast)",
            'parentToggle': 'DenoiserAfterFirstRestorerToggle',
            'requiredToggleValue': True,
            'help': 'Denoising mode for the pass after first restorer. Single Step is generally faster.'
        },
        'DenoiserSingleStepTimestepSliderAfterFirst': {
            'level': 3,
            'widget_type': 'ParameterSlider',
            'label': 'Single Step Timestep (t) (After)',
            'control_name': 'DenoiserSingleStepTimestepSliderAfterFirst',
            'min_value': '1', 'max_value': '500', 'default': '1', 'step': 1, # Max value was 200
            'parentToggle': 'DenoiserAfterFirstRestorerToggle',
            'requiredToggleValue': True,
            'parentSelection': 'DenoiserModeSelectionAfterFirst',
            'requiredSelectionValue': "Single Step (Fast)",
            'help': 'Timestep for single-step denoising (After first Restorer). Lower values mean less noise added/removed.'
        },
        'DenoiserDDIMStepsSliderAfterFirst': {
            'level': 3,
            'widget_type': 'ParameterSlider',
            'label': 'DDIM Steps (After First)',
            'control_name': 'DenoiserDDIMStepsSliderAfterFirst',
            'min_value': '1', 'max_value': '300', 'default': '10', 'step': 1,
            'parentToggle': 'DenoiserAfterFirstRestorerToggle',
            'requiredToggleValue': True,
            'parentSelection': 'DenoiserModeSelectionAfterFirst',
            'requiredSelectionValue': "Full Restore (DDIM)",
            'help': "Number of DDIM steps for full restoration (After First Restorer). Higher = more detail, slower."
        },
        'DenoiserCFGScaleDecimalSliderAfterFirst': {
            'level': 3,
            'widget_type': 'ParameterDecimalSlider',
            'label': 'CFG Scale (After First)',
            'control_name': 'DenoiserCFGScaleDecimalSliderAfterFirst',
            'min_value': '0.0', 'max_value': '10.0', 'default': '1.0', 'step': 0.1, 'decimals': 1,
            'parentToggle': 'DenoiserAfterFirstRestorerToggle',
            'requiredToggleValue': True,
            'parentSelection': 'DenoiserModeSelectionAfterFirst',
            'requiredSelectionValue': "Full Restore (DDIM)",
            'help': "Classifier-Free Guidance scale for DDIM (After First Restorer). Higher = stronger adherence to K/V."
        },
        'DenoiserAfterRestorersToggle': {
            'level': 1,
            'widget_type': 'ToggleButton',
            'label': 'Enable Denoiser After Restorers',
            'control_name': 'DenoiserAfterRestorersToggle',
            'default': False,
            'help': 'Apply the UNet Denoiser again after face restorers have been applied. Uses the same UNet model and step settings.',
            'exec_function': control_actions.handle_denoiser_state_change,
            'exec_function_args': ['DenoiserAfterRestorersToggle'],
        },
        'DenoiserModeSelectionAfter': {
            'level': 2,
            'widget_type': 'SelectionBox',
            'label': 'Denoiser Mode (After)',
            'control_name': 'DenoiserModeSelectionAfter',
            'options': ["Single Step (Fast)", "Full Restore (DDIM)"],
            'default': "Single Step (Fast)",
            'parentToggle': 'DenoiserAfterRestorersToggle',
            'requiredToggleValue': True,
            'help': 'Denoising mode for the pass after restorers. Single Step is generally faster.'
        },
        'DenoiserSingleStepTimestepSliderAfter': {
            'level': 3,
            'widget_type': 'ParameterSlider',
            'label': 'Single Step Timestep (t) (After)',
            'control_name': 'DenoiserSingleStepTimestepSliderAfter',
            'min_value': '1', 'max_value': '500', 'default': '1', 'step': 1, # Max value was 200
            'parentToggle': 'DenoiserAfterRestorersToggle',
            'requiredToggleValue': True,
            'parentSelection': 'DenoiserModeSelectionAfter',
            'requiredSelectionValue': "Single Step (Fast)",
            'help': 'Timestep for single-step denoising (After Restorers). Lower values mean less noise added/removed.'
        },
        'DenoiserDDIMStepsSliderAfter': {
            'level': 3,
            'widget_type': 'ParameterSlider',
            'label': 'DDIM Steps (After)',
            'control_name': 'DenoiserDDIMStepsSliderAfter',
            'min_value': '1', 'max_value': '300', 'default': '10', 'step': 1,
            'parentToggle': 'DenoiserAfterRestorersToggle',
            'requiredToggleValue': True,
            'parentSelection': 'DenoiserModeSelectionAfter',
            'requiredSelectionValue': "Full Restore (DDIM)",
            'help': "Number of DDIM steps for full restoration (After Restorers). Higher = more detail, slower."
        },
        'DenoiserCFGScaleDecimalSliderAfter': {
            'level': 3,
            'widget_type': 'ParameterDecimalSlider',
            'label': 'CFG Scale (After)',
            'control_name': 'DenoiserCFGScaleDecimalSliderAfter',
            'min_value': '0.0', 'max_value': '10.0', 'default': '1.0', 'step': 0.1, 'decimals': 1,
            'parentToggle': 'DenoiserAfterRestorersToggle',
            'requiredToggleValue': True,
            'parentSelection': 'DenoiserModeSelectionAfter',
            'requiredSelectionValue': "Full Restore (DDIM)",
            'help': "Classifier-Free Guidance scale for DDIM (After Restorers). Higher = stronger adherence to K/V."
        }
    }
}