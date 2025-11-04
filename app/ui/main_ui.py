from typing import Dict, Optional
from pathlib import Path
import os
from functools import partial
import copy

from PySide6 import QtWidgets, QtGui
from PySide6 import QtCore
import torch

from app.ui.core.main_window import Ui_MainWindow
import app.ui.widgets.actions.common_actions as common_widget_actions
from app.ui.widgets.actions import card_actions
from app.ui.widgets.actions import layout_actions
from app.ui.widgets.actions import video_control_actions
from app.ui.widgets.actions import filter_actions
from app.ui.widgets.actions import save_load_actions
from app.ui.widgets.actions import list_view_actions
from app.ui.widgets.actions import graphics_view_actions
from app.ui.widgets.actions import job_manager_actions
from app.ui.widgets.actions import preset_actions
from app.ui.widgets.advanced_embedding_editor import EmbeddingGUI

from app.processors.video_processor import VideoProcessor
from app.processors.models_processor import ModelsProcessor
from app.ui.widgets import widget_components
from app.ui.widgets.event_filters import (
    GraphicsViewEventFilter,
    VideoSeekSliderEventFilter,
    videoSeekSliderLineEditEventFilter,
    ListWidgetEventFilter,
)
from app.ui.widgets import ui_workers
from app.ui.widgets.common_layout_data import COMMON_LAYOUT_DATA
from app.ui.widgets.denoiser_layout_data import DENOISER_LAYOUT_DATA
from app.ui.widgets.swapper_layout_data import SWAPPER_LAYOUT_DATA
from app.ui.widgets.settings_layout_data import SETTINGS_LAYOUT_DATA
from app.ui.widgets.face_editor_layout_data import FACE_EDITOR_LAYOUT_DATA
from app.helpers.miscellaneous import DFMModelManager, ParametersDict, ThumbnailManager
from app.helpers.typing_helper import (
    FacesParametersTypes,
    ParametersTypes,
    ControlTypes,
    MarkerTypes,
)
from app.processors.models_data import (
    models_dir as global_models_dir,
)  # For UNet model discovery


ParametersWidgetTypes = Dict[
    str,
    widget_components.ToggleButton
    | widget_components.SelectionBox
    | widget_components.ParameterDecimalSlider
    | widget_components.ParameterSlider
    | widget_components.ParameterLineEdit,
]


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    placeholder_update_signal = QtCore.Signal(QtWidgets.QListWidget, bool)
    gpu_memory_update_signal = QtCore.Signal(int, int)
    model_loading_signal = QtCore.Signal()
    model_loaded_signal = QtCore.Signal()
    display_messagebox_signal = QtCore.Signal(str, str, QtWidgets.QWidget)

    def initialize_variables(self):
        self.video_loader_worker: ui_workers.TargetMediaLoaderWorker | bool = False
        self.input_faces_loader_worker: ui_workers.InputFacesLoaderWorker | bool = False
        self.target_videos_filter_worker = ui_workers.FilterWorker(
            main_window=self, search_text="", filter_list="target_videos"
        )
        self.input_faces_filter_worker = ui_workers.FilterWorker(
            main_window=self, search_text="", filter_list="input_faces"
        )
        self.merged_embeddings_filter_worker = ui_workers.FilterWorker(
            main_window=self, search_text="", filter_list="merged_embeddings"
        )
        self.video_processor = VideoProcessor(self)
        self.models_processor = ModelsProcessor(self)
        # Connect the signals from the worker thread to our new slots
        self.models_processor.show_build_dialog.connect(self.show_build_dialog)
        self.models_processor.hide_build_dialog.connect(self.hide_build_dialog)

        self.target_videos: Dict[
            str, widget_components.TargetMediaCardButton
        ] = {}  # Contains button objects of target videos (Set as list instead of single video to support batch processing in future)
        self.target_faces: Dict[
            str, widget_components.TargetFaceCardButton
        ] = {}  # Contains button objects of target faces
        self.input_faces: Dict[
            str, widget_components.InputFaceCardButton
        ] = {}  # Contains button objects of source faces (images)
        self.merged_embeddings: Dict[str, widget_components.EmbeddingCardButton] = {}
        self.cur_selected_target_face_button: Optional[
            widget_components.TargetFaceCardButton
        ] = None
        self.selected_video_button: widget_components.TargetMediaCardButton | None = (
            None
        )
        self.selected_target_face_id = False
        self._rightFacesStrip = None  # Container-Widget
        self._rightFacesButtonsRow = None  # HLayout für Buttons
        self._faceButtonsOriginalTexts = {}  # zum Wiederherstellen
        self._rightFacesStripVisible = False

        # --- Initialize Managers ---
        self.thumbnail_manager = ThumbnailManager()
        self.dfm_model_manager = DFMModelManager()

        self.parameters: FacesParametersTypes = {}
        self.default_parameters: ParametersTypes = ParametersDict({}, {})
        self.copied_parameters: ParametersTypes = {}
        self.current_widget_parameters: ParametersTypes = {}

        self.markers: MarkerTypes = {}  # Video Markers (Contains parameters for each face)
        self.parameters_list = {}
        self.control: ControlTypes = {}
        self.parameter_widgets: ParametersWidgetTypes = {}

        # UNet related
        self.previous_kv_file_selection = ""
        self.current_kv_tensors_map: Dict[str, torch.Tensor] | None = None
        self.fixed_unet_model_name = "RefLDM_UNET_EXTERNAL_KV"

        self.loaded_embedding_filename: str = ""

        # List of (start_frame, end_frame) tuples for job segments
        # end_frame can be None if a start marker is set but the end is not yet set.
        self.job_marker_pairs: list[tuple[int, int | None]] = []

        self.last_target_media_folder_path = ""
        self.last_input_media_folder_path = ""

        self.is_full_screen = False

        # This flag is used to make sure new loaded media is properly fit into the graphics frame on the first load
        self.project_root_path = Path(__file__).resolve().parent.parent.parent
        self.actual_models_dir_path = self.project_root_path / global_models_dir
        self.loading_new_media = False

        self.gpu_memory_update_signal.connect(
            partial(common_widget_actions.set_gpu_memory_progressbar_value, self)
        )
        self.placeholder_update_signal.connect(
            partial(common_widget_actions.update_placeholder_visibility, self)
        )
        self.model_loading_signal.connect(
            partial(common_widget_actions.show_model_loading_dialog, self)
        )
        self.model_loaded_signal.connect(
            partial(common_widget_actions.hide_model_loading_dialog, self)
        )
        self.display_messagebox_signal.connect(
            partial(common_widget_actions.create_and_show_messagebox, self)
        )
        self.last_seek_read_failed = False
        self.embedding_editor_window = None

    def initialize_widgets(self):
        # Initialize QListWidget for target media
        self.targetVideosList.setFlow(QtWidgets.QListWidget.LeftToRight)
        self.targetVideosList.setWrapping(True)
        self.targetVideosList.setResizeMode(QtWidgets.QListWidget.Adjust)

        # Initialize QListWidget for face images
        self.inputFacesList.setFlow(QtWidgets.QListWidget.LeftToRight)
        self.inputFacesList.setWrapping(True)
        self.inputFacesList.setResizeMode(QtWidgets.QListWidget.Adjust)

        # Set up Menu Actions
        layout_actions.set_up_menu_actions(self)

        # Set up placeholder texts in ListWidgets (Target Videos and Input Faces)
        list_view_actions.set_up_list_widget_placeholder(self, self.targetVideosList)
        list_view_actions.set_up_list_widget_placeholder(self, self.inputFacesList)

        # Set up click to select and drop action on ListWidgets
        self.targetVideosList.setAcceptDrops(True)
        self.targetVideosList.viewport().setAcceptDrops(False)
        self.inputFacesList.setAcceptDrops(True)
        self.inputFacesList.viewport().setAcceptDrops(False)
        list_widget_event_filter = ListWidgetEventFilter(self, self)
        self.targetVideosList.installEventFilter(list_widget_event_filter)
        self.targetVideosList.viewport().installEventFilter(list_widget_event_filter)
        self.inputFacesList.installEventFilter(list_widget_event_filter)
        self.inputFacesList.viewport().installEventFilter(list_widget_event_filter)

        # Set up folder open buttons for Target and Input
        self.buttonTargetVideosPath.clicked.connect(
            partial(list_view_actions.select_target_medias, self, "folder")
        )
        self.buttonInputFacesPath.clicked.connect(
            partial(list_view_actions.select_input_face_images, self, "folder")
        )

        # Initialize graphics frame to view frames
        self.scene = QtWidgets.QGraphicsScene()
        self.graphicsViewFrame.setScene(self.scene)
        # Event filter to start playing when clicking on frame
        graphics_event_filter = GraphicsViewEventFilter(
            self,
            self.graphicsViewFrame,
        )
        self.graphicsViewFrame.installEventFilter(graphics_event_filter)

        video_control_actions.enable_zoom_and_pan(self.graphicsViewFrame)

        video_slider_event_filter = VideoSeekSliderEventFilter(
            self, self.videoSeekSlider
        )
        self.videoSeekSlider.installEventFilter(video_slider_event_filter)
        self.videoSeekSlider.valueChanged.connect(
            partial(video_control_actions.on_change_video_seek_slider, self)
        )
        self.videoSeekSlider.sliderPressed.connect(
            partial(video_control_actions.on_slider_pressed, self)
        )
        self.videoSeekSlider.sliderReleased.connect(
            partial(video_control_actions.on_slider_released, self)
        )
        video_control_actions.set_up_video_seek_slider(self)
        self.frameAdvanceButton.clicked.connect(
            partial(video_control_actions.advance_video_slider_by_n_frames, self)
        )
        self.frameRewindButton.clicked.connect(
            partial(video_control_actions.rewind_video_slider_by_n_frames, self)
        )

        # JOB MANAGER changes addMarkerButton connection
        self.addMarkerButton.clicked.connect(
            partial(video_control_actions.show_add_marker_menu, self)
        )
        self.removeMarkerButton.clicked.connect(
            partial(video_control_actions.remove_video_slider_marker, self)
        )
        self.nextMarkerButton.clicked.connect(
            partial(video_control_actions.move_slider_to_next_nearest_marker, self)
        )
        self.previousMarkerButton.clicked.connect(
            partial(video_control_actions.move_slider_to_previous_nearest_marker, self)
        )

        self.viewFullScreenButton.clicked.connect(
            partial(video_control_actions.view_fullscreen, self)
        )
        # Set up videoSeekLineEdit and add the event filter to handle changes
        video_control_actions.set_up_video_seek_line_edit(self)
        video_seek_line_edit_event_filter = videoSeekSliderLineEditEventFilter(
            self, self.videoSeekLineEdit
        )
        self.videoSeekLineEdit.installEventFilter(video_seek_line_edit_event_filter)

        # Audio toggle
        self.liveSoundButton.toggled.connect(
            partial(video_control_actions.toggle_live_sound, self)
        )

        # Connect the Play/Stop button to the play_video method
        self.buttonMediaPlay.toggled.connect(
            partial(video_control_actions.play_video, self)
        )
        self.buttonMediaRecord.toggled.connect(
            partial(video_control_actions.record_video, self)
        )
        # self.buttonMediaStop.clicked.connect(partial(self.video_processor.stop_processing))
        self.findTargetFacesButton.clicked.connect(
            partial(card_actions.find_target_faces, self)
        )
        self.clearTargetFacesButton.clicked.connect(
            partial(card_actions.clear_target_faces, self)
        )
        self.targetVideosSearchBox.textChanged.connect(
            partial(filter_actions.filter_target_videos, self)
        )
        self.filterImagesCheckBox.clicked.connect(
            partial(filter_actions.filter_target_videos, self)
        )
        self.filterVideosCheckBox.clicked.connect(
            partial(filter_actions.filter_target_videos, self)
        )
        self.filterWebcamsCheckBox.clicked.connect(
            partial(filter_actions.filter_target_videos, self)
        )
        self.filterWebcamsCheckBox.clicked.connect(
            partial(list_view_actions.load_target_webcams, self)
        )

        self.inputFacesSearchBox.textChanged.connect(
            partial(filter_actions.filter_input_faces, self)
        )
        self.inputEmbeddingsSearchBox.textChanged.connect(
            partial(filter_actions.filter_merged_embeddings, self)
        )
        self.openEditorButton.clicked.connect(self.open_embedding_editor)
        self.openEmbeddingButton.clicked.connect(
            partial(save_load_actions.open_embeddings_from_file, self)
        )
        self.saveEmbeddingButton.clicked.connect(
            partial(save_load_actions.save_embeddings_to_file, self)
        )
        self.saveEmbeddingAsButton.clicked.connect(
            partial(save_load_actions.save_embeddings_to_file, self, True)
        )

        self.swapfacesButton.clicked.connect(
            partial(video_control_actions.process_swap_faces, self)
        )
        self.editFacesButton.clicked.connect(
            partial(video_control_actions.process_edit_faces, self)
        )

        self.saveImageButton.clicked.connect(
            partial(video_control_actions.save_current_frame_to_file, self)
        )
        self.batchImageButton.clicked.connect(
            partial(video_control_actions.process_batch_images, self, False)
        )
        self.batchallImageButton.clicked.connect(
            partial(video_control_actions.process_batch_images, self, True)
        )
        self.clearMemoryButton.clicked.connect(
            partial(common_widget_actions.clear_gpu_memory, self)
        )

        self.parametersPanelCheckBox.toggled.connect(
            partial(layout_actions.show_hide_parameters_panel, self)
        )
        self.facesPanelCheckBox.toggled.connect(
            partial(layout_actions.show_hide_faces_panel, self)
        )
        self.facesPanelCheckBox.toggled.connect(self._on_faces_panel_toggled)
        self.TargetMediaCheckBox.toggled.connect(
            partial(layout_actions.show_hide_input_target_media_panel, self)
        )
        self.InputFacesCheckBox.toggled.connect(
            partial(layout_actions.show_hide_input_faces_panel, self)
        )
        self.JobsCheckBox.toggled.connect(
            partial(layout_actions.show_hide_input_jobs_panel, self)
        )

        self.faceMaskCheckBox.clicked.connect(
            partial(video_control_actions.process_compare_checkboxes, self)
        )
        self.faceCompareCheckBox.clicked.connect(
            partial(video_control_actions.process_compare_checkboxes, self)
        )

        layout_actions.add_widgets_to_tab_layout(
            self,
            LAYOUT_DATA=COMMON_LAYOUT_DATA,
            layoutWidget=self.commonWidgetsLayout,
            data_type="parameter",
        )
        layout_actions.add_widgets_to_tab_layout(
            self,
            LAYOUT_DATA=DENOISER_LAYOUT_DATA,
            layoutWidget=self.denoiserWidgetsLayout,
            data_type="control",
        )
        layout_actions.add_widgets_to_tab_layout(
            self,
            LAYOUT_DATA=SWAPPER_LAYOUT_DATA,
            layoutWidget=self.swapWidgetsLayout,
            data_type="parameter",
        )
        layout_actions.add_widgets_to_tab_layout(
            self,
            LAYOUT_DATA=SETTINGS_LAYOUT_DATA,
            layoutWidget=self.settingsWidgetsLayout,
            data_type="control",
        )
        layout_actions.add_widgets_to_tab_layout(
            self,
            LAYOUT_DATA=FACE_EDITOR_LAYOUT_DATA,
            layoutWidget=self.faceEditorWidgetsLayout,
            data_type="parameter",
        )

        # Set up output folder select button (It is inside the settings tab Widget)
        self.outputFolderButton.clicked.connect(
            partial(list_view_actions.select_output_media_folder, self)
        )
        common_widget_actions.create_control(self, "OutputMediaFolder", "")
        self.outputOpenButton.clicked.connect(
            partial(list_view_actions.open_output_media_folder, self)
        )

        # Initialize presets list and buttons
        preset_actions.refresh_presets_list(self)
        preset_actions.setup_preset_list_context_menu(self)
        self.applyPresetButton.clicked.connect(
            partial(preset_actions.apply_selected_preset, self)
        )
        self.savePresetButton.clicked.connect(
            partial(preset_actions.save_current_as_preset, self)
        )
        self.controlPresetButton.clicked.connect(
            partial(preset_actions.control_preset_toggle, self)
        )
        self.presetsList.itemDoubleClicked.connect(
            partial(preset_actions.handle_preset_double_click, self)
        )
        self.presetsList.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)

        # Initialize current_widget_parameters with default values
        self.current_widget_parameters = ParametersDict(
            copy.deepcopy(self.default_parameters), self.default_parameters
        )
        self._populate_denoiser_unet_models()
        self._populate_reference_kv_tensors()

        # Initialize the button states
        video_control_actions.reset_media_buttons(self)

        # Set GPU Memory Progressbar
        font = self.vramProgressBar.font()
        font.setBold(True)
        self.vramProgressBar.setFont(font)
        common_widget_actions.update_gpu_memory_progressbar(self)
        # Set face_swap_tab as the default focused tab
        self.tabWidget.setCurrentIndex(0)
        # widget_actions.add_groupbox_and_widgets_from_layout_map(self)

        job_manager_actions.setup_job_manager_ui(self)

        # Connect Denoiser Mode SelectionBox signals to update visibility
        denoiser_mode_before_combo = self.parameter_widgets.get(
            "DenoiserModeSelectionBefore"
        )
        if denoiser_mode_before_combo:
            # Pass the new text (current_mode_text) from the signal to the handler
            denoiser_mode_before_combo.currentTextChanged.connect(
                lambda text,
                ps="Before": self.update_denoiser_controls_visibility_for_pass(ps, text)
            )
            # Initial call using the value from self.control, which should be the default
            initial_mode_before = self.control.get(
                "DenoiserModeSelectionBefore", "Single Step (Fast)"
            )
            self.update_denoiser_controls_visibility_for_pass(
                "Before", initial_mode_before
            )

        denoiser_mode_after_first_combo = self.parameter_widgets.get(
            "DenoiserModeSelectionAfterFirst"
        )
        if denoiser_mode_after_first_combo:
            denoiser_mode_after_first_combo.currentTextChanged.connect(
                lambda text,
                ps="AfterFirst": self.update_denoiser_controls_visibility_for_pass(
                    ps, text
                )
            )
            initial_mode_after_first = self.control.get(
                "DenoiserModeSelectionAfterFirst", "Single Step (Fast)"
            )
            self.update_denoiser_controls_visibility_for_pass(
                "AfterFirst", initial_mode_after_first
            )

        denoiser_mode_after_combo = self.parameter_widgets.get(
            "DenoiserModeSelectionAfter"
        )
        if denoiser_mode_after_combo:
            denoiser_mode_after_combo.currentTextChanged.connect(
                lambda text,
                ps="After": self.update_denoiser_controls_visibility_for_pass(ps, text)
            )
            initial_mode_after = self.control.get(
                "DenoiserModeSelectionAfter", "Single Step (Fast)"
            )
            self.update_denoiser_controls_visibility_for_pass(
                "After", initial_mode_after
            )

        # Initialize the dedicated progress dialog for TensorRT builds.
        # This object is created here, in the main GUI thread.
        self.build_progress_dialog = QtWidgets.QProgressDialog(self)
        self.build_progress_dialog.setCancelButton(None)
        self.build_progress_dialog.setWindowModality(
            QtCore.Qt.WindowModality.WindowModal
        )
        self.build_progress_dialog.setRange(0, 0)  # Indeterminate (busy) mode
        self.build_progress_dialog.close()  # Ensure it's hidden on startup

    def update_denoiser_controls_visibility_for_pass(
        self, pass_suffix: str, current_mode_text: str
    ):
        """
        Updates visibility of denoiser controls for a specific pass (Before, AfterFirst, After)
        based on the provided current_mode_text.
        """
        current_mode = current_mode_text  # Use the passed text directly

        # Define widget names based on the pass_suffix
        single_step_slider_name = f"DenoiserSingleStepTimestepSlider{pass_suffix}"
        ddim_steps_slider_name = f"DenoiserDDIMStepsSlider{pass_suffix}"
        cfg_scale_slider_name = f"DenoiserCFGScaleDecimalSlider{pass_suffix}"

        # Get widget instances from self.parameter_widgets
        single_step_widget = self.parameter_widgets.get(single_step_slider_name)
        ddim_steps_widget = self.parameter_widgets.get(ddim_steps_slider_name)
        cfg_scale_widget = self.parameter_widgets.get(cfg_scale_slider_name)

        # Helper to set visibility for a widget and its associated label and reset button
        def set_widget_visibility(widget_instance, is_visible):
            if widget_instance:
                widget_instance.setVisible(is_visible)
                if (
                    hasattr(widget_instance, "label_widget")
                    and widget_instance.label_widget
                ):
                    widget_instance.label_widget.setVisible(is_visible)
                if (
                    hasattr(widget_instance, "reset_default_button")
                    and widget_instance.reset_default_button
                ):
                    widget_instance.reset_default_button.setVisible(is_visible)
                if hasattr(widget_instance, "line_edit") and widget_instance.line_edit:
                    widget_instance.line_edit.setVisible(is_visible)

        # Set visibility for Single Step controls
        is_single_step_mode = current_mode == "Single Step (Fast)"
        set_widget_visibility(single_step_widget, is_single_step_mode)

        # Set visibility for Full Restore (DDIM) controls
        is_full_restore_mode = current_mode == "Full Restore (DDIM)"
        set_widget_visibility(ddim_steps_widget, is_full_restore_mode)
        set_widget_visibility(cfg_scale_widget, is_full_restore_mode)

    def _populate_denoiser_unet_models(self):
        unet_model_files = []
        # default_unet_model = "ref_ldm_unet_real_refs_n1.onnx" # Prioritize based on existence and sorting later

        if os.path.exists(global_models_dir):
            for f_name in os.listdir(global_models_dir):
                if f_name.startswith("ref_ldm_unet_") and f_name.endswith(".onnx"):
                    unet_model_files.append(f_name)

        # Ensure the default model is in the list if it exists, and prioritize it
        unet_model_files.sort()  # Sort alphabetically for consistent order

        denoiser_model_widget = self.parameter_widgets.get("DenoiserUNetModelSelection")
        if denoiser_model_widget and isinstance(
            denoiser_model_widget, widget_components.SelectionBox
        ):
            current_selection_in_control = self.control.get(
                "DenoiserUNetModelSelection"
            )
            denoiser_model_widget.clear()

            if unet_model_files:
                denoiser_model_widget.addItems(unet_model_files)

                # If a previous selection exists and is still valid, keep it. Otherwise, pick the first.
                if (
                    not current_selection_in_control
                    or current_selection_in_control not in unet_model_files
                ):
                    new_selection = unet_model_files[0]
                    self.control["DenoiserUNetModelSelection"] = new_selection
                    denoiser_model_widget.setCurrentText(new_selection)
                else:
                    denoiser_model_widget.setCurrentText(current_selection_in_control)
            else:
                denoiser_model_widget.addItem("No UNet models found")
                self.control["DenoiserUNetModelSelection"] = ""  # No model selected
                denoiser_model_widget.setCurrentText("No UNet models found")

    def _populate_reference_kv_tensors(self):
        kv_tensor_files = []
        kv_tensors_dir = os.path.join(global_models_dir, "reference_kv_data")

        if os.path.exists(kv_tensors_dir):
            for f_name in os.listdir(kv_tensors_dir):
                if f_name.endswith(".pt"):
                    kv_tensor_files.append(f_name)

        kv_tensor_files.sort()

        kv_tensor_widget = self.parameter_widgets.get("ReferenceKVTensorsSelection")
        if kv_tensor_widget and isinstance(
            kv_tensor_widget, widget_components.SelectionBox
        ):
            current_selection_in_control = self.control.get(
                "ReferenceKVTensorsSelection"
            )
            kv_tensor_widget.clear()

            if kv_tensor_files:
                kv_tensor_widget.addItems(kv_tensor_files)

                if (
                    not current_selection_in_control
                    or current_selection_in_control not in kv_tensor_files
                ):
                    new_selection = kv_tensor_files[0]
                    self.control["ReferenceKVTensorsSelection"] = new_selection
                    kv_tensor_widget.setCurrentText(new_selection)
                else:
                    kv_tensor_widget.setCurrentText(current_selection_in_control)
            else:
                kv_tensor_widget.addItem("No K/V Tensors found")
                self.control["ReferenceKVTensorsSelection"] = ""
                kv_tensor_widget.setCurrentText("No K/V Tensors found")

    def handle_reference_kv_file_change(self, new_kv_file_name: str):
        with self.models_processor.model_lock:
            self.current_kv_tensors_map = None
            self.control["ReferenceKVTensorsSelection"] = new_kv_file_name
            self.previous_kv_file_selection = new_kv_file_name
            if new_kv_file_name and new_kv_file_name != "No K/V tensor files found":
                kv_file_path = (
                    self.actual_models_dir_path / "reference_kv_data" / new_kv_file_name
                )
                if kv_file_path.exists():
                    try:
                        self.model_loading_signal.emit()
                        kv_payload = torch.load(
                            kv_file_path, map_location="cpu", weights_only=True
                        )
                        self.current_kv_tensors_map = kv_payload.get("kv_map")
                        if self.current_kv_tensors_map:
                            print(
                                f"Successfully loaded K/V map from {new_kv_file_name} for {len(self.current_kv_tensors_map)} layers."
                            )
                        else:
                            print(f"Warning: 'kv_map' not found in {new_kv_file_name}.")
                            self.current_kv_tensors_map = None
                        self.model_loaded_signal.emit()
                    except Exception as e:
                        print(f"Error loading K/V tensor file {kv_file_path}: {e}")
                        self.current_kv_tensors_map = None
                        self.model_loaded_signal.emit()
                else:
                    print(f"K/V tensor file not found: {kv_file_path}")
                    self.current_kv_tensors_map = None
            else:
                self.current_kv_tensors_map = None

        denoiser_enabled_before = self.control.get(
            "DenoiserUNetEnableBeforeRestorersToggle", False
        )
        denoiser_enabled_after_first = self.control.get(
            "DenoiserAfterFirstRestorerToggle", False
        )
        denoiser_enabled_after = self.control.get("DenoiserAfterRestorersToggle", False)

        if (
            denoiser_enabled_before
            or denoiser_enabled_after_first
            or denoiser_enabled_after
        ):
            if new_kv_file_name:
                common_widget_actions.refresh_frame(self)

    @QtCore.Slot(str, str)
    def show_build_dialog(self, title, text):
        """
        Slot to show or update the TensorRT build progress dialog.
        This function is guaranteed to run in the main GUI thread.
        """
        if self.build_progress_dialog:
            self.build_progress_dialog.setWindowTitle(title)
            self.build_progress_dialog.setLabelText(text)
            self.build_progress_dialog.show()
            # Force the GUI to update immediately
            QtCore.QCoreApplication.processEvents()

    @QtCore.Slot()
    def hide_build_dialog(self):
        """
        Slot to hide the TensorRT build progress dialog.
        This function is guaranteed to run in the main GUI thread.
        """
        if self.build_progress_dialog:
            self.build_progress_dialog.close()

    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        self.initialize_variables()
        self.initialize_widgets()
        self.load_last_workspace()

    @QtCore.Slot(list)
    def handle_unload_request(self, model_names: list):
        """Unloads models requested by a worker thread, keeping essential ones."""
        current_swapper = self.control.get("FaceSwapperTypeSelection", "Inswapper128")
        active_arcface_model = self.models_processor.get_arcface_model(current_swapper)

        print(f"Unload request for: {model_names}")
        print(f"Keeping active model: {active_arcface_model}")

        for model_name in model_names:
            # Do not unload the recognition model for the currently selected swapper
            if model_name == active_arcface_model:
                continue
            # Special case: CSCS uses two models, keep both if it's active
            if model_name == "CSCSIDArcFace" and active_arcface_model == "CSCSArcFace":
                continue

            self.models_processor.unload_model(model_name)

        # After unloading, refresh the VRAM display
        common_widget_actions.update_gpu_memory_progressbar(self)

    def resizeEvent(self, event: QtGui.QResizeEvent):
        # print("Called resizeEvent()")
        super().resizeEvent(event)
        # Call the method to fit the image to the view whenever the window resizes
        if self.scene.items():
            pixmap_item = self.scene.items()[0]
            # Set the scene rectangle to the bounding rectangle of the pixmap
            scene_rect = pixmap_item.boundingRect()
            self.graphicsViewFrame.setSceneRect(scene_rect)
            graphics_view_actions.fit_image_to_view(self, pixmap_item, scene_rect)

    def keyPressEvent(self, event):
        match event.key():
            case QtCore.Qt.Key_F11:
                video_control_actions.view_fullscreen(self)
            case QtCore.Qt.Key_V:
                video_control_actions.advance_video_slider_by_n_frames(self, n=1)
            case QtCore.Qt.Key_C:
                video_control_actions.rewind_video_slider_by_n_frames(self, n=1)
            case QtCore.Qt.Key_D:
                video_control_actions.advance_video_slider_by_n_frames(self, n=30)
            case QtCore.Qt.Key_A:
                video_control_actions.rewind_video_slider_by_n_frames(self, n=30)
            case QtCore.Qt.Key_Z:
                self.videoSeekSlider.setValue(0)
            case QtCore.Qt.Key_Space:
                self.buttonMediaPlay.click()
            case QtCore.Qt.Key_R:
                self.buttonMediaRecord.click()
            case QtCore.Qt.Key_F:
                if event.modifiers() & QtCore.Qt.KeyboardModifier.AltModifier:
                    video_control_actions.remove_video_slider_marker(self)
                else:
                    video_control_actions.add_video_slider_marker(self)
            case QtCore.Qt.Key_W:
                video_control_actions.move_slider_to_nearest_marker(self, "next")
            case QtCore.Qt.Key_Q:
                video_control_actions.move_slider_to_nearest_marker(self, "previous")
            case QtCore.Qt.Key_S:
                self.swapfacesButton.click()

    def closeEvent(self, event):
        print("MainWindow: closeEvent called.")

        self.video_processor.stop_processing()
        list_view_actions.clear_stop_loading_input_media(self)
        list_view_actions.clear_stop_loading_target_media(self)

        save_load_actions.save_current_workspace(self, "last_workspace.json")
        # Optionally handle the event if needed
        event.accept()

    def load_last_workspace(self):
        # Show the load workspace dialog if the file exists
        if Path("last_workspace.json").is_file():
            auto_load_workspace_toggle = (
                save_load_actions.get_auto_load_workspace_toggle(
                    self, "last_workspace.json"
                )
            )
            if auto_load_workspace_toggle:
                save_load_actions.load_saved_workspace(self, "last_workspace.json")
            else:
                load_dialog = widget_components.LoadLastWorkspaceDialog(self)
                load_dialog.exec_()

            # Re-populate and set current selection for dynamic widgets like DenoiserUNetModelSelection
            self._populate_denoiser_unet_models()
            self._populate_reference_kv_tensors()

    def save_last_workspace(self):
        pass

    @QtCore.Slot(bool)
    def _on_faces_panel_toggled(self, checked: bool):
        # checked=True  -> Faces-Panel sichtbar  -> alles zurück ins Panel
        # checked=False -> Faces-Panel aus       -> Liste + Buttons nach rechts
        if checked:
            # zurück ins Panel
            if getattr(self, "_rightFacesStrip", None):
                self._restore_faces_strip_to_panel()
        else:
            # nach rechts ziehen
            self._ensure_right_faces_strip()
            self._move_faces_strip_to_right()

    def _ensure_right_faces_strip(self):
        if getattr(self, "_rightFacesStrip", None) is not None:
            return

        self._rightFacesStrip = QtWidgets.QWidget(self.dockWidgetContents_2)
        stripLayout = QtWidgets.QVBoxLayout(self._rightFacesStrip)
        stripLayout.setContentsMargins(0, 0, 0, 0)
        stripLayout.setSpacing(6)

        # Faces oben
        self._rightFacesFacesHolder = QtWidgets.QVBoxLayout()
        self._rightFacesFacesHolder.setContentsMargins(0, 0, 0, 0)
        self._rightFacesFacesHolder.setSpacing(0)
        stripLayout.addLayout(self._rightFacesFacesHolder)

        # Buttons unten nebeneinander
        self._rightFacesButtonsRow = QtWidgets.QHBoxLayout()
        self._rightFacesButtonsRow.setContentsMargins(0, 0, 0, 0)
        self._rightFacesButtonsRow.setSpacing(8)
        stripLayout.addLayout(self._rightFacesButtonsRow)

        # unter den Tabs im rechten Dock
        self.gridLayout_5.addWidget(self._rightFacesStrip, 2, 0, 1, 1)

        # ---- Höhe & Policies für den Strip kompakt setzen ----
        sp_strip = self._rightFacesStrip.sizePolicy()
        sp_strip.setVerticalPolicy(QtWidgets.QSizePolicy.Fixed)
        self._rightFacesStrip.setSizePolicy(sp_strip)

        # Max-Höhen: Liste + Buttons zusammen ~170px
        # -> Liste (Thumbnails)
        try:
            self.targetFacesList.setMaximumHeight(80)  # kompakte Thumbnail-Zeile
        except Exception:
            pass

        # Buttons-Reihe: optional per Container fixieren
        self._rightFacesButtonsRowContainer = getattr(
            self, "_rightFacesButtonsRowContainer", None
        )
        if self._rightFacesButtonsRowContainer is None:
            self._rightFacesButtonsRowContainer = QtWidgets.QWidget(
                self._rightFacesStrip
            )
            lay = QtWidgets.QHBoxLayout(self._rightFacesButtonsRowContainer)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.setSpacing(8)
            # alten Row-Layout-Inhalt in den Container umziehen
            while self._rightFacesButtonsRow.count():
                item = self._rightFacesButtonsRow.takeAt(0)
                w = item.widget()
                if w:
                    lay.addWidget(w)
            # alten Row durch Container ersetzen
            parent_v = self._rightFacesStrip.layout()
            parent_v.removeItem(self._rightFacesButtonsRow)
            parent_v.addWidget(self._rightFacesButtonsRowContainer)
            self._rightFacesButtonsRow = lay

        self._rightFacesButtonsRowContainer.setFixedHeight(32)

        # Gesamthöhe des Strips hart deckeln
        self._rightFacesStrip.setMaximumHeight(120)

        # ---- Layout-Gewichte: Tabs (Zeile 1) kriegen alles, Strip (Zeile 2) nichts ----
        self.gridLayout_5.setRowStretch(1, 1)  # Tabs
        self.gridLayout_5.setRowStretch(2, 0)  # Faces-Strip

    def _move_faces_strip_to_right(self):
        # targetFacesList aus Faces-Panel lösen
        try:
            self.gridLayout_2.removeWidget(self.targetFacesList)
        except Exception:
            pass

        self.targetFacesList.setParent(self._rightFacesStrip)
        sp = self.targetFacesList.sizePolicy()
        sp.setVerticalStretch(1)  # Liste füllt den oberen Bereich
        self.targetFacesList.setSizePolicy(sp)
        self._rightFacesFacesHolder.addWidget(self.targetFacesList)

        # Buttons aus der linken Column nehmen
        btns = [
            self.findTargetFacesButton,
            self.clearTargetFacesButton,
            self.swapfacesButton,
            self.editFacesButton,
        ]

        # Originaltexte einmalig merken
        if not hasattr(self, "_faceButtonsOriginalTexts"):
            self._faceButtonsOriginalTexts = {}
        if not self._faceButtonsOriginalTexts:
            for b in btns:
                self._faceButtonsOriginalTexts[b.objectName()] = b.text()

        try:
            for b in btns:
                self.controlButtonsLayout.removeWidget(b)
        except Exception:
            pass

        # Kurzlabels setzen und unten nebeneinander einsetzen
        short_map = {
            "findTargetFacesButton": "Find",
            "clearTargetFacesButton": "Clear",
            "swapfacesButton": "Swap",
            "editFacesButton": "Edit",
        }
        for b in btns:
            b.setParent(self._rightFacesStrip)
            b.setText(short_map.get(b.objectName(), b.text()))
            b.setFlat(True)
            self._rightFacesButtonsRow.addWidget(b)

        self._rightFacesButtonsRow.addStretch(1)
        # Kompakte Darstellung erzwingen
        self.targetFacesList.setViewMode(QtWidgets.QListView.IconMode)
        self.targetFacesList.setWrapping(True)
        self.targetFacesList.setSpacing(4)
        self.targetFacesList.setMinimumHeight(60)
        self.targetFacesList.setMaximumHeight(80)  # wichtig

    def _restore_faces_strip_to_panel(self):
        # Liste zurück ins Faces-Panel (Originalzelle 1,1,1,1 – wie in deiner .ui)
        try:
            self._rightFacesFacesHolder.removeWidget(self.targetFacesList)
        except Exception:
            pass

        self.targetFacesList.setParent(self.facesPanelGroupBox)
        self.gridLayout_2.addWidget(self.targetFacesList, 1, 1, 1, 1)

        # Buttons zurück in die linke, vertikale Spalte (untereinander)
        btns = [
            self.findTargetFacesButton,
            self.clearTargetFacesButton,
            self.swapfacesButton,
            self.editFacesButton,
        ]

        try:
            for b in btns:
                self._rightFacesButtonsRow.removeWidget(b)
        except Exception:
            pass

        for b in btns:
            b.setParent(self.facesButtonsWidget)
            # Originaltext wiederherstellen
            if hasattr(self, "_faceButtonsOriginalTexts"):
                orig = self._faceButtonsOriginalTexts.get(b.objectName())
                if orig:
                    b.setText(orig)
            b.setFlat(True)
            self.controlButtonsLayout.addWidget(b)

        # Limits zurücksetzen
        self.targetFacesList.setMaximumHeight(16777215)
        if hasattr(self, "_rightFacesStrip"):
            self._rightFacesStrip.setMaximumHeight(16777215)
        if (
            hasattr(self, "_rightFacesButtonsRowContainer")
            and self._rightFacesButtonsRowContainer
        ):
            self._rightFacesButtonsRowContainer.setMaximumHeight(16777215)

        # Tabs/Strip-Stretches optional neutralisieren (nicht zwingend)
        self.gridLayout_5.setRowStretch(1, 0)
        self.gridLayout_5.setRowStretch(2, 0)

    def open_embedding_editor(self):
        if self.embedding_editor_window is None:
            self.embedding_editor_window = EmbeddingGUI()
        self.embedding_editor_window.show()
