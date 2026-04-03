from typing import Dict, Optional
from pathlib import Path
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
import app.ui.widgets.actions.control_actions as control_actions
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
from app.helpers.app_metadata import get_app_display_metadata
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

# --- Constants ---
DENOISER_MODE_SINGLE_STEP = "Single Step (Fast)"
DENOISER_MODE_FULL_RESTORE = "Full Restore (DDIM)"

ParametersWidgetTypes = Dict[
    str,
    widget_components.ToggleButton
    | widget_components.SelectionBox
    | widget_components.ParameterDecimalSlider
    | widget_components.ParameterSlider
    | widget_components.ParameterLineEdit,
]

_FACE_STRIP_MAX_HEIGHT = 120
_FACE_STRIP_LIST_HEIGHT = 80
_FACE_STRIP_BUTTONS_HEIGHT = 32
_FACES_PANEL_ROW_HEIGHT = 144


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    placeholder_update_signal = QtCore.Signal(QtWidgets.QListWidget, bool)
    gpu_memory_update_signal = QtCore.Signal(int, int)
    model_loading_signal = QtCore.Signal()
    model_loaded_signal = QtCore.Signal()
    display_messagebox_signal = QtCore.Signal(str, str, QtWidgets.QWidget)

    def initialize_variables(self):
        self.video_loader_worker: ui_workers.TargetMediaLoaderWorker | None = None
        self.input_faces_loader_worker: ui_workers.InputFacesLoaderWorker | None = None
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
        self.selected_target_face_id = None
        self._rightFacesStrip = None  # Container-Widget
        self._rightFacesButtonsRow = None  # HLayout für Buttons
        self._faceButtonsOriginalTexts = {}  # zum Wiederherstellen
        self._rightFacesStripVisible = False
        self._theatre_normal_panel_states: dict[str, bool] | None = None
        self._theatre_mode_panel_states: dict[str, bool] | None = None
        self._fullscreen_restore_was_maximized = False
        self._fullscreen_restore_geometry = None
        self.panel_visibility_state: dict[str, bool] = {
            "target_media": True,
            "input_faces": True,
            "jobs": True,
            "faces": True,
            "parameters": True,
        }
        self.view_face_compare_enabled = False
        self.view_face_mask_enabled = False
        self.viewer_mode_actions_enabled = True

        # --- Initialize Managers ---
        self.thumbnail_manager = ThumbnailManager()
        self.dfm_model_manager = DFMModelManager()

        self.parameters: FacesParametersTypes = {}
        self.default_parameters: ParametersTypes = ParametersDict({}, {})
        self.copied_parameters: ParametersTypes = {}
        self.current_widget_parameters: ParametersTypes = {}

        self.markers: MarkerTypes = {}  # Video Markers (Contains parameters for each face)
        self.issue_frames_by_face: dict[str, set[int]] = {}
        self.issue_frames: set[int] = set()
        self.dropped_frames: set[int] = set()
        self.scan_tools_expanded = False
        self.scan_issue_worker = None
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
        self.is_batch_processing = False

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

        # Initialize list widgets with consistent sizing and layout configuration
        list_view_actions.initialize_media_list_widgets(self)
        list_view_actions.initialize_embeddings_list_widget(self)

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

        video_control_actions.enable_zoom_and_pan(self, self.graphicsViewFrame)

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
        video_control_actions.add_scan_review_controls(self)
        video_control_actions.initialize_media_button_icons(self)

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
        # Connect the button click to our new model management function
        self.editFacesButton.clicked.connect(
            partial(control_actions.handle_face_editor_button_click, self)
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

        QtCore.QTimer.singleShot(0, self._configure_faces_panel_button_column)
        self.theatreModeButton.clicked.connect(
            partial(video_control_actions.toggle_theatre_mode, self)
        )
        self._install_view_panel_toggle_actions()
        self._install_view_navigation_actions()
        self._install_media_controls_layout()
        self._install_compare_mask_toggle_buttons()
        self._install_media_controls_separator()
        self.verticalSpacer.changeSize(
            20,
            2,
            QtWidgets.QSizePolicy.Policy.Minimum,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        self.gridLayout_2.setContentsMargins(9, 4, 9, 9)
        self.verticalLayout.invalidate()
        QtCore.QTimer.singleShot(0, self._normalize_media_control_button_sizes)

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
        self._connect_mask_show_selection_sync()
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
            denoiser_mode_before_combo.currentTextChanged.connect(
                partial(self.update_denoiser_controls_visibility_for_pass, "Before")
            )
            initial_mode_before = self.control.get(
                "DenoiserModeSelectionBefore", DENOISER_MODE_SINGLE_STEP
            )
            self.update_denoiser_controls_visibility_for_pass(
                "Before", initial_mode_before
            )

        denoiser_mode_after_first_combo = self.parameter_widgets.get(
            "DenoiserModeSelectionAfterFirst"
        )
        if denoiser_mode_after_first_combo:
            denoiser_mode_after_first_combo.currentTextChanged.connect(
                partial(self.update_denoiser_controls_visibility_for_pass, "AfterFirst")
            )
            initial_mode_after_first = self.control.get(
                "DenoiserModeSelectionAfterFirst", DENOISER_MODE_SINGLE_STEP
            )
            self.update_denoiser_controls_visibility_for_pass(
                "AfterFirst", initial_mode_after_first
            )

        denoiser_mode_after_combo = self.parameter_widgets.get(
            "DenoiserModeSelectionAfter"
        )
        if denoiser_mode_after_combo:
            denoiser_mode_after_combo.currentTextChanged.connect(
                partial(self.update_denoiser_controls_visibility_for_pass, "After")
            )
            initial_mode_after = self.control.get(
                "DenoiserModeSelectionAfter", DENOISER_MODE_SINGLE_STEP
            )
            self.update_denoiser_controls_visibility_for_pass(
                "After", initial_mode_after
            )

        # Initialize the dedicated progress dialog for TensorRT builds.
        # This object is created here, in the main GUI thread.
        self.build_progress_dialog = QtWidgets.QProgressDialog(self)
        flags = self.build_progress_dialog.windowFlags()
        flags &= ~QtCore.Qt.WindowCloseButtonHint
        self.build_progress_dialog.setWindowFlags(flags)
        self.build_progress_dialog.setCancelButton(None)
        self.build_progress_dialog.setWindowModality(
            QtCore.Qt.WindowModality.WindowModal
        )
        self.build_progress_dialog.setRange(0, 0)  # Indeterminate (busy) mode
        self.build_progress_dialog.close()  # Ensure it's hidden on startup

        self.scan_progress_dialog = QtWidgets.QProgressDialog(self)
        scan_flags = self.scan_progress_dialog.windowFlags()
        scan_flags &= ~QtCore.Qt.WindowCloseButtonHint
        self.scan_progress_dialog.setWindowFlags(scan_flags)
        self.scan_progress_dialog.setWindowModality(
            QtCore.Qt.WindowModality.WindowModal
        )
        self.scan_progress_dialog.setCancelButtonText("Abort")
        self.scan_progress_dialog.setAutoClose(False)
        self.scan_progress_dialog.setAutoReset(False)
        self.scan_progress_dialog.close()

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
        is_single_step_mode = current_mode == DENOISER_MODE_SINGLE_STEP
        set_widget_visibility(single_step_widget, is_single_step_mode)

        # Set visibility for Full Restore (DDIM) controls
        is_full_restore_mode = current_mode == DENOISER_MODE_FULL_RESTORE
        set_widget_visibility(ddim_steps_widget, is_full_restore_mode)
        set_widget_visibility(cfg_scale_widget, is_full_restore_mode)

    def _populate_model_file_selection_widget(
        self, widget_name: str, scan_dir: Path, prefix: str, ext: str
    ):
        """Helper that scans scan_dir for files matching prefix+ext, populates a SelectionBox widget."""
        model_files = []
        if scan_dir.exists() and scan_dir.is_dir():
            for f_path in scan_dir.iterdir():
                if f_path.is_file():
                    f_name = f_path.name
                    if (not prefix or f_name.startswith(prefix)) and f_name.endswith(
                        ext
                    ):
                        model_files.append(f_name)
        model_files.sort()

        selection_widget = self.parameter_widgets.get(widget_name)
        if selection_widget and isinstance(
            selection_widget, widget_components.SelectionBox
        ):
            current_selection = self.control.get(widget_name)
            selection_widget.clear()
            if model_files:
                selection_widget.addItems(model_files)
                if not current_selection or current_selection not in model_files:
                    new_selection = model_files[0]
                    self.control[widget_name] = new_selection
                    selection_widget.setCurrentText(new_selection)
                else:
                    selection_widget.setCurrentText(current_selection)
            else:
                placeholder = f"No {ext.lstrip('.')} models found"
                selection_widget.addItem(placeholder)
                self.control[widget_name] = ""
                selection_widget.setCurrentText(placeholder)

    def _populate_denoiser_unet_models(self):
        self._populate_model_file_selection_widget(
            widget_name="DenoiserUNetModelSelection",
            scan_dir=self.actual_models_dir_path,
            prefix="ref_ldm_unet_",
            ext=".onnx",
        )

    def _populate_reference_kv_tensors(self):
        kv_tensors_dir = self.actual_models_dir_path / "reference_kv_data"
        self._populate_model_file_selection_widget(
            widget_name="ReferenceKVTensorsSelection",
            scan_dir=kv_tensors_dir,
            prefix="",
            ext=".pt",
        )

    def handle_reference_kv_file_change(self, new_kv_file_name: str):
        self.control["ReferenceKVTensorsSelection"] = new_kv_file_name
        self.previous_kv_file_selection = new_kv_file_name
        new_kv_tensors_map = None
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
                    new_kv_tensors_map = kv_payload.get("kv_map")
                    if new_kv_tensors_map:
                        print(
                            f"[INFO] Successfully loaded K/V map from {new_kv_file_name} for {len(new_kv_tensors_map)} layers."
                        )
                    else:
                        print(f"[WARN] 'kv_map' not found in {new_kv_file_name}.")
                        new_kv_tensors_map = None
                    self.model_loaded_signal.emit()
                except Exception as e:
                    print(f"[ERROR] Error loading K/V tensor file {kv_file_path}: {e}")
                    new_kv_tensors_map = None
                    self.model_loaded_signal.emit()
            else:
                print(f"[ERROR] K/V tensor file not found: {kv_file_path}")
        with self.models_processor.model_lock:
            self.current_kv_tensors_map = new_kv_tensors_map

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
        self._base_window_title = self.windowTitle()
        self.initialize_variables()
        self._apply_runtime_window_title()
        self.initialize_widgets()
        self.load_last_workspace()

    @QtCore.Slot(list)
    def handle_unload_request(self, model_names: list):
        """Unloads models requested by a worker thread, keeping essential ones."""
        current_swapper = self.control.get("FaceSwapperTypeSelection", "Inswapper128")
        active_arcface_model = self.models_processor.get_arcface_model(current_swapper)

        print(f"[INFO] Unload request for: {model_names}")
        print(f"[INFO] Keeping active model: {active_arcface_model}")

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
        # print("[INFO] Called resizeEvent()")
        super().resizeEvent(event)
        self._queue_media_controls_balance()
        # Call the method to fit the image to the view whenever the window resizes
        items = self.scene.items()
        pixmap_item = next(
            (item for item in items if isinstance(item, QtWidgets.QGraphicsPixmapItem)),
            None,
        )
        if pixmap_item:
            # Set the scene rectangle to the bounding rectangle of the pixmap
            scene_rect = pixmap_item.boundingRect()
            self.graphicsViewFrame.setSceneRect(scene_rect)
            graphics_view_actions.fit_image_to_view(self, pixmap_item, scene_rect)

    def changeEvent(self, event: QtCore.QEvent):
        super().changeEvent(event)
        if event.type() == QtCore.QEvent.Type.WindowStateChange:
            self.is_full_screen = self.isFullScreen()

    def eventFilter(self, watched, event):
        viewport_widget = getattr(self, "mediaControlsViewportWidget", None)
        if (
            viewport_widget is not None
            and watched is viewport_widget
            and event.type() == QtCore.QEvent.Type.Resize
        ):
            self._queue_media_controls_balance()
        return super().eventFilter(watched, event)

    def _queue_media_controls_balance(self):
        if getattr(self, "_media_controls_balance_queued", False):
            return
        self._media_controls_balance_queued = True

        def _run_balance():
            self._media_controls_balance_queued = False
            self._sync_media_controls_balance()

        QtCore.QTimer.singleShot(0, _run_balance)

    def _apply_runtime_window_title(self):
        base_title = getattr(self, "_base_window_title", self.windowTitle())
        self.app_display_metadata = get_app_display_metadata(
            self.project_root_path, base_title
        )
        self.setWindowTitle(self.app_display_metadata.window_title)

    def keyPressEvent(self, event):
        match event.key():
            case QtCore.Qt.Key_Escape:
                if self.isFullScreen():
                    video_control_actions.view_fullscreen(self)
                elif getattr(self, "is_theatre_mode", False):
                    video_control_actions.toggle_theatre_mode(self)
            case QtCore.Qt.Key_F11:
                video_control_actions.view_fullscreen(self)
            case QtCore.Qt.Key_T:
                video_control_actions.toggle_theatre_mode(self)
            case QtCore.Qt.Key_V:
                video_control_actions.advance_video_slider_by_n_frames(self, n=1)
            case QtCore.Qt.Key_C:
                video_control_actions.rewind_video_slider_by_n_frames(self, n=1)
            case QtCore.Qt.Key_D:
                video_control_actions.advance_video_slider_by_n_frames(self)
            case QtCore.Qt.Key_A:
                video_control_actions.rewind_video_slider_by_n_frames(self)
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
        print("[INFO] MainWindow: closeEvent called.")

        self.video_processor.stop_processing()
        list_view_actions.clear_stop_loading_input_media(self)
        list_view_actions.clear_stop_loading_target_media(self)

        save_load_actions.save_current_workspace(
            self, str(self.project_root_path / "last_workspace.json")
        )
        self.video_processor.join_and_clear_threads()
        # Optionally handle the event if needed
        event.accept()

    def load_last_workspace(self):
        # Show the load workspace dialog if the file exists
        last_workspace_path = self.project_root_path / "last_workspace.json"
        if last_workspace_path.is_file():
            auto_load_workspace_toggle = (
                save_load_actions.get_auto_load_workspace_toggle(
                    self, str(last_workspace_path)
                )
            )
            if auto_load_workspace_toggle:
                save_load_actions.load_saved_workspace(self, str(last_workspace_path))
            else:
                load_dialog = widget_components.LoadLastWorkspaceDialog(self)
                load_dialog.exec_()

            # Re-populate and set current selection for dynamic widgets like DenoiserUNetModelSelection
            self._populate_denoiser_unet_models()
            self._populate_reference_kv_tensors()

    @QtCore.Slot(bool)
    def _on_faces_panel_toggled(self, checked: bool):
        """
        Handles the visibility toggle of the Faces Panel.
        Moves the list and buttons safely using Qt's automatic layout reparenting.
        """
        if checked:
            if getattr(self, "_rightFacesStrip", None):
                self._restore_faces_strip_to_panel()
        else:
            self._ensure_right_faces_strip()
            self._move_faces_strip_to_right()

    def _normalize_media_control_button_sizes(self):
        """Apply a consistent preferred size to the centered media controls."""
        play_size = self.buttonMediaPlay.sizeHint()
        if not play_size.isValid():
            return

        target_height = play_size.height()
        scan_tools_button = getattr(self, "scanToolsToggleButton", None)
        scan_tools_text_width = 0
        if scan_tools_button is not None:
            scan_tools_font_metrics = QtGui.QFontMetrics(scan_tools_button.font())
            scan_tools_text_width = (
                scan_tools_font_metrics.horizontalAdvance(scan_tools_button.text()) + 28
            )

        preferred_width = max(play_size.width() * 2 + 10, scan_tools_text_width + 8, 96)
        target_size = QtCore.QSize(preferred_width, target_height)
        transport_icon_size = QtCore.QSize(22, 22)
        marker_icon_size = QtCore.QSize(20, 20)
        utility_icon_size = QtCore.QSize(22, 22)

        transport_buttons = [
            self.frameRewindButton,
            self.buttonMediaRecord,
            self.buttonMediaPlay,
            self.frameAdvanceButton,
        ]
        marker_buttons = [
            self.addMarkerButton,
            self.removeMarkerButton,
            self.previousMarkerButton,
            self.nextMarkerButton,
        ]
        icon_utility_buttons = [
            self.liveSoundButton,
            self.viewFullScreenButton,
            self.theatreModeButton,
        ]
        text_utility_buttons = [
            getattr(self, "scanToolsToggleButton", None),
        ]

        for button in transport_buttons:
            min_width = max(32, transport_icon_size.width() + 18)
            button.setMinimumHeight(target_size.height())
            button.setMaximumHeight(target_size.height())
            button.setMinimumWidth(min_width)
            button.setMaximumWidth(target_size.width())
            size_policy = button.sizePolicy()
            size_policy.setHorizontalPolicy(QtWidgets.QSizePolicy.Minimum)
            size_policy.setVerticalPolicy(QtWidgets.QSizePolicy.Fixed)
            button.setSizePolicy(size_policy)
            button.setIconSize(transport_icon_size)

        for button in marker_buttons:
            min_width = max(30, marker_icon_size.width() + 16)
            button.setMinimumHeight(target_size.height())
            button.setMaximumHeight(target_size.height())
            button.setMinimumWidth(min_width)
            button.setMaximumWidth(target_size.width())
            size_policy = button.sizePolicy()
            size_policy.setHorizontalPolicy(QtWidgets.QSizePolicy.Minimum)
            size_policy.setVerticalPolicy(QtWidgets.QSizePolicy.Fixed)
            button.setSizePolicy(size_policy)
            button.setIconSize(marker_icon_size)

        for button in icon_utility_buttons:
            min_width = max(30, utility_icon_size.width() + 16)
            button.setMinimumHeight(target_size.height())
            button.setMaximumHeight(target_size.height())
            button.setMinimumWidth(min_width)
            button.setMaximumWidth(target_size.width())
            size_policy = button.sizePolicy()
            size_policy.setHorizontalPolicy(QtWidgets.QSizePolicy.Minimum)
            size_policy.setVerticalPolicy(QtWidgets.QSizePolicy.Fixed)
            button.setSizePolicy(size_policy)
            button.setIconSize(utility_icon_size)

        self.liveSoundButton.setIconSize(QtCore.QSize(24, 24))
        self.theatreModeButton.setIconSize(QtCore.QSize(28, 28))

        for button in text_utility_buttons:
            if button is None:
                continue
            font_metrics = QtGui.QFontMetrics(button.font())
            text_width = font_metrics.horizontalAdvance(button.text()) + 28
            minimum_width = text_width
            button.setMinimumHeight(target_size.height())
            button.setMaximumHeight(target_size.height())
            button.setMinimumWidth(minimum_width)
            button.setMaximumWidth(target_size.width())
            size_policy = button.sizePolicy()
            size_policy.setHorizontalPolicy(QtWidgets.QSizePolicy.Minimum)
            size_policy.setVerticalPolicy(QtWidgets.QSizePolicy.Fixed)
            button.setSizePolicy(size_policy)

        if hasattr(self, "mediaControlsMainLayout"):
            self.mediaControlsMainLayout.setSpacing(12)
        self._sync_media_controls_balance()

    def _install_media_controls_layout(self):
        """Center the media controls in a single simple row."""
        if getattr(self, "_media_controls_layout_installed", False):
            return

        top_layout = self.horizontalLayoutMediaButtons

        while top_layout.count():
            item = top_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.hide()

        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(0)

        self.mediaControlsViewportWidget = QtWidgets.QWidget(self.mediaLayout)
        self.mediaControlsViewportWidget.setObjectName("mediaControlsViewportWidget")
        self.mediaControlsViewportWidget.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed
        )
        self.mediaControlsViewportWidget.installEventFilter(self)
        self.mediaControlsViewportLayout = QtWidgets.QGridLayout(
            self.mediaControlsViewportWidget
        )
        self.mediaControlsViewportLayout.setContentsMargins(0, 0, 0, 0)
        self.mediaControlsViewportLayout.setSpacing(0)

        self.mediaControlsCenterWidget = QtWidgets.QWidget(
            self.mediaControlsViewportWidget
        )
        self.mediaControlsCenterWidget.setObjectName("mediaControlsCenterWidget")
        self.mediaControlsCenterWidget.setSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        self.mediaControlsCenterLayout = QtWidgets.QHBoxLayout(
            self.mediaControlsCenterWidget
        )
        self.mediaControlsCenterLayout.setContentsMargins(0, 0, 0, 0)
        self.mediaControlsCenterLayout.setSpacing(12)
        self.mediaControlsCenterLayout.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.mediaControlsViewportLayout.addItem(
            QtWidgets.QSpacerItem(
                0,
                0,
                QtWidgets.QSizePolicy.Policy.Expanding,
                QtWidgets.QSizePolicy.Policy.Minimum,
            ),
            0,
            0,
        )
        self.mediaControlsViewportLayout.addWidget(self.mediaControlsCenterWidget, 0, 1)
        self.mediaControlsViewportLayout.addItem(
            QtWidgets.QSpacerItem(
                0,
                0,
                QtWidgets.QSizePolicy.Policy.Expanding,
                QtWidgets.QSizePolicy.Policy.Minimum,
            ),
            0,
            2,
        )
        self.mediaControlsViewportLayout.setColumnStretch(0, 1)
        self.mediaControlsViewportLayout.setColumnStretch(2, 1)

        top_layout.addWidget(self.mediaControlsViewportWidget, 1)

        self.mediaControlsMainLayout = self.mediaControlsCenterLayout
        self.mediaControlsTransportLayout = self.mediaControlsMainLayout
        self.mediaControlsUtilityLayout = self.mediaControlsMainLayout

        for button in [
            self.addMarkerButton,
            self.removeMarkerButton,
            self.previousMarkerButton,
            self.nextMarkerButton,
            self.frameRewindButton,
            self.buttonMediaRecord,
            self.buttonMediaPlay,
            self.frameAdvanceButton,
            self.liveSoundButton,
            self.viewFullScreenButton,
            self.theatreModeButton,
        ]:
            button.show()
            self.mediaControlsMainLayout.addWidget(button)

        scan_tools_button = getattr(self, "scanToolsToggleButton", None)
        if scan_tools_button is not None:
            scan_tools_button.show()
            self.mediaControlsMainLayout.addWidget(scan_tools_button)

        self._media_controls_layout_installed = True
        self._queue_media_controls_balance()

    def _install_view_panel_toggle_actions(self):
        """Move the top panel toggle checkboxes into the View menu."""
        if getattr(self, "_view_panel_toggle_actions_installed", False):
            return

        panel_toggle_specs = [
            ("target_media", "TargetMediaCheckBox", "Target Videos/Images"),
            ("input_faces", "InputFacesCheckBox", "Input Faces"),
            ("jobs", "JobsCheckBox", "Job Manager"),
            ("faces", "facesPanelCheckBox", "Faces / Embeddings"),
            ("parameters", "parametersPanelCheckBox", "Parameters"),
        ]

        for panel_key, checkbox_attr, label in panel_toggle_specs:
            checkbox = getattr(self, checkbox_attr, None)
            if checkbox is None:
                continue

            checkbox.hide()
            self.panel_visibility_state[panel_key] = self._current_panel_visibility(
                panel_key
            )

            action = QtGui.QAction(label, self.menuView)
            action.setCheckable(True)
            action.setChecked(self.panel_visibility_state[panel_key])
            action.toggled.connect(
                lambda checked, key=panel_key: self._set_panel_visibility(key, checked)
            )

            setattr(self, f"actionViewToggle_{panel_key}", action)
            self.menuView.insertAction(self.actionView_Fullscreen_F11, action)

        self.menuView.insertSeparator(self.actionView_Fullscreen_F11)
        self.panelVisibilityCheckBoxLayout.setContentsMargins(0, 0, 0, 0)
        self.panelVisibilityCheckBoxLayout.setSpacing(0)
        self.panelVisibilityCheckBoxLayout.setSizeConstraint(
            QtWidgets.QLayout.SizeConstraint.SetMinimumSize
        )
        self.verticalLayout.removeItem(self.panelVisibilityCheckBoxLayout)
        self._install_panel_visibility_sync()
        self._view_panel_toggle_actions_installed = True

    def _install_view_navigation_actions(self):
        """Add viewport navigation and presentation actions to the View menu."""
        if getattr(self, "_view_navigation_actions_installed", False):
            return

        self.actionView_Fullscreen_F11.setText("Fullscreen\tF11")
        self.actionView_Fullscreen_F11.setCheckable(True)

        fit_action = QtGui.QAction("Fit to View", self.menuView)
        fit_action.setShortcut(QtGui.QKeySequence("Ctrl+0"))
        fit_action.setShortcutContext(QtCore.Qt.ShortcutContext.WindowShortcut)
        fit_action.triggered.connect(
            lambda: video_control_actions.fit_view_to_current_image(self)
        )

        zoom_100_action = QtGui.QAction("100% Zoom", self.menuView)
        zoom_100_action.setShortcut(QtGui.QKeySequence("Ctrl+1"))
        zoom_100_action.setShortcutContext(QtCore.Qt.ShortcutContext.WindowShortcut)
        zoom_100_action.triggered.connect(
            lambda: video_control_actions.zoom_current_image_100(self)
        )

        theatre_action = QtGui.QAction("Theatre Mode\tT", self.menuView)
        theatre_action.setCheckable(True)
        theatre_action.triggered.connect(
            lambda: video_control_actions.toggle_theatre_mode(self)
        )

        face_compare_action = QtGui.QAction("Face Compare", self.menuView)
        face_compare_action.setCheckable(True)
        face_compare_action.triggered.connect(
            lambda checked: (
                self._set_compare_mode("compare", checked),
                self._sync_viewer_menu_actions(),
            )
        )

        mask_actions = {}
        for option in self._get_mask_show_options():
            action = QtGui.QAction(
                self._mask_show_context_menu_label(option), self.menuView
            )
            action.setCheckable(True)
            action.triggered.connect(
                lambda checked, value=option: self._handle_viewer_mask_action(
                    value, checked
                )
            )
            mask_actions[option] = action

        self.actionView_FitToView = fit_action
        self.actionView_100Zoom = zoom_100_action
        self.actionView_TheatreMode = theatre_action
        self.actionView_FaceCompare = face_compare_action
        self._view_mask_actions = mask_actions

        self.menuView.removeAction(self.actionView_Fullscreen_F11)
        self.menuView.addAction(self.actionView_FitToView)
        self.menuView.addAction(self.actionView_100Zoom)
        self.menuView.addSeparator()
        self.menuView.addAction(self.actionView_Fullscreen_F11)
        self.menuView.addAction(self.actionView_TheatreMode)
        self.menuView.addSeparator()
        self.menuView.addAction(self.actionView_FaceCompare)
        self.menuView.addSeparator()
        for option in self._get_mask_show_options():
            self.menuView.addAction(self._view_mask_actions[option])

        self.addAction(self.actionView_FitToView)
        self.addAction(self.actionView_100Zoom)
        self.addAction(self.actionView_TheatreMode)
        self.menuView.aboutToShow.connect(self._sync_viewer_menu_actions)
        self._sync_viewer_menu_actions()
        self._view_navigation_actions_installed = True

    def _install_compare_mask_toggle_buttons(self):
        """Initialize hidden compare/mask checkbox state without visible media-bar buttons."""
        if getattr(self, "_compare_mask_toggle_buttons_installed", False):
            return
        compare_checkbox = getattr(self, "faceCompareCheckBox", None)
        if compare_checkbox is not None:
            self.view_face_compare_enabled = compare_checkbox.isChecked()
            compare_checkbox.hide()

        mask_checkbox = getattr(self, "faceMaskCheckBox", None)
        if mask_checkbox is not None:
            self.view_face_mask_enabled = mask_checkbox.isChecked()
            mask_checkbox.hide()

        self._compare_mask_toggle_buttons_installed = True

    def _set_panel_visibility(self, panel_key: str, checked: bool):
        """Apply panel visibility from the visible View menu actions."""
        self.panel_visibility_state[panel_key] = checked

        panel_handlers = {
            "target_media": layout_actions.show_hide_input_target_media_panel,
            "input_faces": layout_actions.show_hide_input_faces_panel,
            "jobs": layout_actions.show_hide_input_jobs_panel,
            "faces": layout_actions.show_hide_faces_panel,
            "parameters": layout_actions.show_hide_parameters_panel,
        }
        panel_handlers[panel_key](self, checked)
        if panel_key == "faces":
            self._on_faces_panel_toggled(checked)

        action = getattr(self, f"actionViewToggle_{panel_key}", None)
        if action is not None and action.isChecked() != checked:
            action.blockSignals(True)
            action.setChecked(checked)
            action.blockSignals(False)

    def _panel_widget_for_key(self, panel_key: str):
        panel_widgets = {
            "target_media": self.input_Target_DockWidget,
            "input_faces": self.input_Faces_DockWidget,
            "jobs": self.jobManagerDockWidget,
            "faces": self.facesPanelGroupBox,
            "parameters": self.controlOptionsDockWidget,
        }
        return panel_widgets[panel_key]

    def _current_panel_visibility(self, panel_key: str) -> bool:
        return not self._panel_widget_for_key(panel_key).isHidden()

    def _sync_panel_visibility_action(self, panel_key: str, checked: bool):
        self.panel_visibility_state[panel_key] = checked
        action = getattr(self, f"actionViewToggle_{panel_key}", None)
        if action is not None and action.isChecked() != checked:
            action.blockSignals(True)
            action.setChecked(checked)
            action.blockSignals(False)

    def _refresh_panel_visibility_state_from_widgets(self):
        for panel_key in self.panel_visibility_state:
            self._sync_panel_visibility_action(
                panel_key, self._current_panel_visibility(panel_key)
            )

    def _install_panel_visibility_sync(self):
        if getattr(self, "_panel_visibility_sync_installed", False):
            return

        dock_sync_specs = [
            ("target_media", self.input_Target_DockWidget),
            ("input_faces", self.input_Faces_DockWidget),
            ("jobs", self.jobManagerDockWidget),
            ("parameters", self.controlOptionsDockWidget),
        ]
        for panel_key, dock_widget in dock_sync_specs:
            dock_widget.visibilityChanged.connect(
                lambda visible, key=panel_key: self._sync_panel_visibility_action(
                    key, visible
                )
            )

        self._panel_visibility_sync_installed = True

    def _set_compare_mode(self, mode_key: str, checked: bool):
        """Apply compare/mask preview mode from shared hidden checkbox state."""
        if mode_key == "compare":
            self.view_face_compare_enabled = checked
            checkbox = getattr(self, "faceCompareCheckBox", None)
        else:
            self.view_face_mask_enabled = checked
            checkbox = getattr(self, "faceMaskCheckBox", None)

        if checkbox is not None and checkbox.isChecked() != checked:
            checkbox.blockSignals(True)
            checkbox.setChecked(checked)
            checkbox.blockSignals(False)
        video_control_actions.process_compare_checkboxes(self)
        self._sync_viewer_menu_actions()

    def _get_mask_show_selection_widget(self):
        return self.parameter_widgets.get("MaskShowSelection")

    def _get_mask_show_label_map(self) -> dict[str, str]:
        return {
            "swap_mask": "Swap",
            "diff": "Differencing",
            "texture": "Texture Transfer",
        }

    def _get_mask_show_options(self) -> list[str]:
        widget = self._get_mask_show_selection_widget()
        if isinstance(widget, widget_components.SelectionBox):
            values = []
            for index in range(widget.count()):
                item_value = widget.itemData(index)
                values.append(
                    item_value if item_value is not None else widget.itemText(index)
                )
            return values
        return list(SWAPPER_LAYOUT_DATA["Masks"]["MaskShowSelection"]["options"])

    def _get_current_mask_show_value(self) -> str:
        widget = self._get_mask_show_selection_widget()
        if isinstance(widget, widget_components.SelectionBox):
            current_value = widget.currentData()
            if current_value:
                return str(current_value)
            current_value = widget.currentText()
            if current_value:
                return current_value
        return str(
            self.current_widget_parameters.get(
                "MaskShowSelection",
                SWAPPER_LAYOUT_DATA["Masks"]["MaskShowSelection"]["default"],
            )
        )

    def _mask_show_option_label(self, value: str) -> str:
        return self._get_mask_show_label_map().get(
            value, value.replace("_", " ").title()
        )

    def _mask_show_context_menu_label(self, value: str) -> str:
        context_labels = {
            "swap_mask": "Swap Mask",
            "diff": "Differencing Mask",
            "texture": "Texture Transfer Mask",
        }
        return context_labels.get(value, self._mask_show_option_label(value))

    def _is_fullscreen_menu_active(self) -> bool:
        return self.isFullScreen()

    def _handle_viewer_mask_action(self, value: str, checked: bool):
        current_value = self._get_current_mask_show_value()
        if self.view_face_mask_enabled and value == current_value and not checked:
            self._set_compare_mode("mask", False)
        elif checked or not self.view_face_mask_enabled or value != current_value:
            self._select_mask_show_option(value)
        self._sync_viewer_menu_actions()

    def _select_mask_show_option(self, value: str):
        widget = self._get_mask_show_selection_widget()
        if isinstance(widget, widget_components.SelectionBox):
            current_value = widget.currentData()
            if current_value != value:
                widget.set_value(value)
        else:
            common_widget_actions.update_parameter(self, "MaskShowSelection", value)
        if not self.view_face_mask_enabled:
            self._set_compare_mode("mask", True)
        else:
            self._sync_viewer_menu_actions()

    def _sync_face_mask_button_presentation(self):
        return

    def _connect_mask_show_selection_sync(self):
        if getattr(self, "_mask_show_selection_sync_installed", False):
            return
        widget = self._get_mask_show_selection_widget()
        if isinstance(widget, widget_components.SelectionBox):
            current_value = self._get_current_mask_show_value()
            label_map = self._get_mask_show_label_map()
            widget.blockSignals(True)
            widget.clear()
            for option in SWAPPER_LAYOUT_DATA["Masks"]["MaskShowSelection"]["options"]:
                widget.addItem(label_map.get(option, option), option)
            widget.set_value(current_value)
            widget.blockSignals(False)
            self._mask_show_selection_sync_installed = True

    def _sync_viewer_menu_actions(self):
        fullscreen_action = getattr(self, "actionView_Fullscreen_F11", None)
        if fullscreen_action is not None:
            fullscreen_action.blockSignals(True)
            fullscreen_action.setCheckable(True)
            fullscreen_action.setChecked(self._is_fullscreen_menu_active())
            fullscreen_action.setEnabled(True)
            fullscreen_action.blockSignals(False)

        theatre_action = getattr(self, "actionView_TheatreMode", None)
        if theatre_action is not None:
            theatre_action.blockSignals(True)
            theatre_action.setCheckable(True)
            theatre_action.setChecked(bool(getattr(self, "is_theatre_mode", False)))
            theatre_action.blockSignals(False)

        compare_action = getattr(self, "actionView_FaceCompare", None)
        if compare_action is not None:
            compare_action.blockSignals(True)
            compare_action.setChecked(bool(self.view_face_compare_enabled))
            compare_action.setEnabled(
                bool(getattr(self, "viewer_mode_actions_enabled", True))
            )
            compare_action.blockSignals(False)

        current_mask_value = self._get_current_mask_show_value()
        viewer_mode_actions_enabled = bool(
            getattr(self, "viewer_mode_actions_enabled", True)
        )
        for option, action in getattr(self, "_view_mask_actions", {}).items():
            action.blockSignals(True)
            action.setChecked(
                bool(self.view_face_mask_enabled) and option == current_mask_value
            )
            action.setEnabled(viewer_mode_actions_enabled)
            action.blockSignals(False)

    def _install_media_controls_separator(self):
        """Insert visual dividers between control groups in the media button row."""
        if getattr(self, "_media_controls_separator", None) is not None:
            return

        def _make_separator(
            name: str, height: int = 16, margin: int = 12
        ) -> QtWidgets.QWidget:
            separator_container = QtWidgets.QWidget(self.mediaLayout)
            separator_container.setObjectName(f"{name}Container")
            separator_layout = QtWidgets.QHBoxLayout(separator_container)
            separator_layout.setContentsMargins(margin, 0, margin, 0)
            separator_layout.setSpacing(0)
            separator = QtWidgets.QFrame(separator_container)
            separator.setObjectName(name)
            separator.setFrameShape(QtWidgets.QFrame.Shape.VLine)
            separator.setFrameShadow(QtWidgets.QFrame.Shadow.Plain)
            separator.setLineWidth(1)
            separator.setMidLineWidth(0)
            separator.setFixedHeight(height)
            separator.setStyleSheet("color: rgba(180, 180, 180, 110);")
            separator_layout.addWidget(separator)
            separator_container.setProperty("defaultMargin", margin)
            return separator_container

        transport_layout = getattr(self, "mediaControlsMainLayout", None)
        if transport_layout is not None:
            playback_marker_separator = _make_separator("mediaControlsSeparator")
            insert_index = transport_layout.indexOf(self.frameRewindButton)
            if insert_index != -1:
                transport_layout.insertWidget(insert_index, playback_marker_separator)
                self._media_controls_separator = playback_marker_separator
            marker_view_separator = _make_separator("mediaControlsViewSeparator")
            insert_index = transport_layout.indexOf(self.liveSoundButton)
            if insert_index != -1:
                transport_layout.insertWidget(insert_index, marker_view_separator)
                self._media_controls_view_separator = marker_view_separator

        self._sync_media_controls_balance()

    def _sync_media_controls_balance(self):
        """Keep the centered media controls minimum width in sync with their content."""
        viewport_widget = getattr(self, "mediaControlsViewportWidget", None)
        center_widget = getattr(self, "mediaControlsCenterWidget", None)
        if viewport_widget is None or center_widget is None:
            return

        main_layout = getattr(self, "mediaControlsMainLayout", None)
        if main_layout is None:
            return

        main_spacing = 12
        separator_margin = 12

        main_layout.setSpacing(main_spacing)
        for separator_name in (
            "_media_controls_separator",
            "_media_controls_view_separator",
        ):
            separator_container = getattr(self, separator_name, None)
            if separator_container is None or separator_container.layout() is None:
                continue
            separator_container.layout().setContentsMargins(
                separator_margin, 0, separator_margin, 0
            )

        center_width = max(
            center_widget.sizeHint().width(), center_widget.minimumSizeHint().width()
        )
        viewport_widget.setMinimumWidth(center_width)

    def _configure_faces_panel_button_column(self):
        """Keep the left-side face buttons matched to the visible target-faces box height."""
        buttons = [
            self.findTargetFacesButton,
            self.clearTargetFacesButton,
            self.swapfacesButton,
            self.editFacesButton,
        ]

        self.gridLayout_2.setRowStretch(1, 0)
        self.controlButtonsLayout.setSpacing(4)

        self.targetFacesList.setMinimumHeight(_FACES_PANEL_ROW_HEIGHT)
        self.targetFacesList.setMaximumHeight(_FACES_PANEL_ROW_HEIGHT)
        self.inputEmbeddingsList.setMinimumHeight(_FACES_PANEL_ROW_HEIGHT)
        self.inputEmbeddingsList.setMaximumHeight(_FACES_PANEL_ROW_HEIGHT)

        margins = self.controlButtonsLayout.contentsMargins()
        spacing_total = self.controlButtonsLayout.spacing() * (len(buttons) - 1)
        margins_total = margins.top() + margins.bottom()
        button_height = max(
            1, (_FACES_PANEL_ROW_HEIGHT - spacing_total - margins_total) // len(buttons)
        )

        for i, button in enumerate(buttons):
            button.setMinimumHeight(button_height)
            button.setMaximumHeight(button_height)
            size_policy = button.sizePolicy()
            size_policy.setVerticalPolicy(QtWidgets.QSizePolicy.Fixed)
            button.setSizePolicy(size_policy)
            self.controlButtonsLayout.setStretch(i, 1)

        self.facesButtonsWidget.setMinimumHeight(_FACES_PANEL_ROW_HEIGHT)
        self.facesButtonsWidget.setMaximumHeight(_FACES_PANEL_ROW_HEIGHT)
        faces_widget_policy = self.facesButtonsWidget.sizePolicy()
        faces_widget_policy.setVerticalPolicy(QtWidgets.QSizePolicy.Fixed)
        self.facesButtonsWidget.setSizePolicy(faces_widget_policy)

        self.verticalWidget.setMinimumHeight(_FACES_PANEL_ROW_HEIGHT)
        self.verticalWidget.setMaximumHeight(_FACES_PANEL_ROW_HEIGHT)
        vertical_widget_policy = self.verticalWidget.sizePolicy()
        vertical_widget_policy.setVerticalPolicy(QtWidgets.QSizePolicy.Fixed)
        self.verticalWidget.setSizePolicy(vertical_widget_policy)

        target_width = self.saveImageButton.sizeHint().width()
        self.facesButtonsWidget.setMinimumWidth(target_width)
        self.facesButtonsWidget.setMaximumWidth(target_width)
        self.verticalWidget.setMinimumWidth(target_width)
        self.verticalWidget.setMaximumWidth(target_width)

        for button in buttons:
            button.setMinimumWidth(target_width)
            button.setMaximumWidth(target_width)

        panel_policy = self.facesPanelGroupBox.sizePolicy()
        panel_policy.setVerticalPolicy(QtWidgets.QSizePolicy.Maximum)
        self.facesPanelGroupBox.setSizePolicy(panel_policy)
        self.facesPanelGroupBox.setMinimumHeight(0)

    def _ensure_right_faces_strip(self):
        """Initializes the right faces strip container once."""
        if getattr(self, "_rightFacesStrip", None) is not None:
            return

        self._rightFacesStrip = QtWidgets.QWidget(self.dockWidgetContents_2)
        stripLayout = QtWidgets.QVBoxLayout(self._rightFacesStrip)
        stripLayout.setContentsMargins(0, 0, 0, 0)
        stripLayout.setSpacing(6)

        self._rightFacesFacesHolder = QtWidgets.QVBoxLayout()
        self._rightFacesFacesHolder.setContentsMargins(0, 0, 0, 0)
        self._rightFacesFacesHolder.setSpacing(0)
        stripLayout.addLayout(self._rightFacesFacesHolder)

        self._rightFacesButtonsRowContainer = QtWidgets.QWidget(self._rightFacesStrip)
        self._rightFacesButtonsRow = QtWidgets.QHBoxLayout(
            self._rightFacesButtonsRowContainer
        )
        self._rightFacesButtonsRow.setContentsMargins(0, 0, 0, 0)
        self._rightFacesButtonsRow.setSpacing(8)

        # Add a permanent stretch to push buttons to the left safely without memory leaks
        self._rightFacesButtonsRow.addStretch(1)

        stripLayout.addWidget(self._rightFacesButtonsRowContainer)
        self.gridLayout_5.addWidget(self._rightFacesStrip, 2, 0, 1, 1)

        sp_strip = self._rightFacesStrip.sizePolicy()
        sp_strip.setVerticalPolicy(QtWidgets.QSizePolicy.Fixed)
        self._rightFacesStrip.setSizePolicy(sp_strip)

        self.targetFacesList.setMaximumHeight(_FACE_STRIP_LIST_HEIGHT)
        self._rightFacesButtonsRowContainer.setFixedHeight(_FACE_STRIP_BUTTONS_HEIGHT)
        self._rightFacesStrip.setMaximumHeight(_FACE_STRIP_MAX_HEIGHT)

        self.gridLayout_5.setRowStretch(1, 1)
        self.gridLayout_5.setRowStretch(2, 0)

    def _move_faces_strip_to_right(self):
        """
        Moves widgets to the right strip.
        Adding a widget to a new layout automatically removes it from the old one in Qt.
        """
        self._rightFacesFacesHolder.addWidget(self.targetFacesList)

        sp = self.targetFacesList.sizePolicy()
        sp.setVerticalStretch(1)
        self.targetFacesList.setSizePolicy(sp)

        btns = [
            (self.findTargetFacesButton, "Find"),
            (self.clearTargetFacesButton, "Clear"),
            (self.swapfacesButton, "Swap"),
            (self.editFacesButton, "Edit"),
        ]

        if not self._faceButtonsOriginalTexts:
            self._faceButtonsOriginalTexts = {
                btn.objectName(): btn.text() for btn, _ in btns
            }

        for i, (btn, short_text) in enumerate(btns):
            # Insert before the stretch item (which is at the end)
            self._rightFacesButtonsRow.insertWidget(i, btn)
            btn.setText(short_text)
            btn.setFlat(True)

        self.targetFacesList.setViewMode(QtWidgets.QListView.IconMode)
        self.targetFacesList.setWrapping(True)
        self.targetFacesList.setSpacing(4)
        self.targetFacesList.setMinimumHeight(60)
        self.targetFacesList.setMaximumHeight(_FACE_STRIP_LIST_HEIGHT)

    def _restore_faces_strip_to_panel(self):
        """Restores widgets to their original left panel seamlessly."""
        self.gridLayout_2.addWidget(self.targetFacesList, 1, 1, 1, 1)

        btns = [
            self.findTargetFacesButton,
            self.clearTargetFacesButton,
            self.swapfacesButton,
            self.editFacesButton,
        ]

        for btn in btns:
            self.controlButtonsLayout.addWidget(btn)
            if hasattr(self, "_faceButtonsOriginalTexts"):
                orig = self._faceButtonsOriginalTexts.get(btn.objectName())
                if orig:
                    btn.setText(orig)
            btn.setFlat(True)

        self.targetFacesList.setMaximumHeight(16777215)
        if hasattr(self, "_rightFacesStrip"):
            self._rightFacesStrip.setMaximumHeight(16777215)
        if getattr(self, "_rightFacesButtonsRowContainer", None):
            self._rightFacesButtonsRowContainer.setMaximumHeight(16777215)

        # VERY IMPORTANT: Restore the list mode so it looks normal again on the left
        self.targetFacesList.setViewMode(QtWidgets.QListView.ListMode)
        QtCore.QTimer.singleShot(0, self._configure_faces_panel_button_column)

    def open_embedding_editor(self):
        if self.embedding_editor_window is None:
            self.embedding_editor_window = EmbeddingGUI()
        self.embedding_editor_window.show()
