from typing import TYPE_CHECKING, cast, Union, List, Callable
from functools import partial
import re

from PySide6 import QtWidgets, QtCore, QtGui

if TYPE_CHECKING:
    from app.ui.main_ui import MainWindow
from app.ui.widgets.actions import common_actions as common_widget_actions
from app.ui.widgets.actions import graphics_view_actions
from app.ui.widgets.actions import list_view_actions
from app.ui.widgets.actions import save_load_actions
from app.ui.widgets.actions import video_control_actions
from app.ui.widgets import widget_components

# from app.UI.Widgets.WidgetComponents import *
from app.helpers.typing_helper import LayoutDictTypes


def add_widgets_to_tab_layout(
    main_window: "MainWindow",
    LAYOUT_DATA: LayoutDictTypes,
    layoutWidget: QtWidgets.QVBoxLayout,
    data_type="parameter",
    section_namespace: str = "default",
):
    layout = QtWidgets.QVBoxLayout()
    layout.setContentsMargins(0, 0, 10, 0)
    scroll_area = QtWidgets.QScrollArea()
    scroll_area.setWidgetResizable(True)
    scroll_area.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    scroll_content = QtWidgets.QWidget()
    scroll_content.setSizePolicy(
        QtWidgets.QSizePolicy.Policy.Expanding,
        QtWidgets.QSizePolicy.Policy.Preferred,
    )
    scroll_content.setLayout(layout)
    scroll_area.setWidget(scroll_content)
    scroll_area.setFrameShape(QtWidgets.QFrame.NoFrame)

    def add_horizontal_layout_to_category(
        category_layout: QtWidgets.QFormLayout, *widgets
    ):
        row_widget = QtWidgets.QWidget()
        row_widget.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Maximum,
        )
        horizontal_layout = QtWidgets.QHBoxLayout(row_widget)

        for widget in widgets:
            horizontal_layout.addWidget(widget)
        category_layout.addRow(row_widget)
        return row_widget, horizontal_layout

    def create_layout_action_button(button_data: dict):
        action_button = QtWidgets.QPushButton(cast(str, button_data["label"]))
        action_button.setToolTip(cast(str, button_data.get("help", "")))
        if "fixed_width" in button_data:
            action_button.setFixedWidth(cast(int, button_data["fixed_width"]))
        else:
            action_button.adjustSize()
            action_button.setMinimumWidth(action_button.sizeHint().width() + 16)
        if "exec_function" in button_data:
            action_button.clicked.connect(
                partial(cast(Callable, button_data["exec_function"]), main_window)
            )
        return action_button

    def build_section_id(category_name: str) -> str:
        normalized_name = re.sub(r"[^a-z0-9]+", "_", category_name.lower()).strip("_")
        return f"{section_namespace}:{normalized_name}"

    for category, widgets in LAYOUT_DATA.items():
        section_id = build_section_id(category)
        display_title = (
            "Face Editor"
            if section_namespace == "face_editor" and not category
            else category
        )
        group_box = widget_components.CollapsibleSection(
            main_window,
            title=display_title,
            section_id=section_id,
            expanded=main_window.parameter_section_states.get(section_id, True),
        )
        category_layout = QtWidgets.QFormLayout()
        group_box.content_widget.setLayout(category_layout)
        main_window.register_parameter_section(section_id, group_box)

        for widget_name, widget_data in widgets.items():
            spacing_level = cast(int, widget_data["level"])
            label = QtWidgets.QLabel(cast(str, widget_data["label"]))
            label.setToolTip(cast(str, widget_data["help"]))

            if "Toggle" in widget_name:
                widget = widget_components.ToggleButton(
                    label=cast(str, widget_data["label"]),
                    widget_name=widget_name,
                    group_layout_data=widgets,
                    label_widget=label,
                    main_window=main_window,
                )
                widget.setChecked(cast(bool, widget_data["default"]))
                widget.reset_default_button = (
                    widget_components.ParameterResetDefaultButton(related_widget=widget)
                )

                row_widget, horizontal_layout = add_horizontal_layout_to_category(
                    category_layout, widget, label, widget.reset_default_button
                )

                if data_type == "parameter":
                    common_widget_actions.create_default_parameter(
                        main_window, widget_name, cast(bool, widget_data["default"])
                    )
                else:
                    common_widget_actions.create_control(
                        main_window, widget_name, cast(bool, widget_data["default"])
                    )

                def onchange_toggle(
                    toggle_widget: widget_components.ToggleButton,
                    toggle_widget_name,
                    widget_data: dict,
                    *args,
                ):
                    toggle_state = toggle_widget.isChecked()
                    if data_type == "parameter":
                        common_widget_actions.update_parameter(
                            main_window,
                            toggle_widget_name,
                            toggle_state,
                            enable_refresh_frame=toggle_widget.enable_refresh_frame,
                            exec_function=widget_data.get("exec_function"),
                            exec_function_args=cast(
                                list, widget_data.get("exec_function_args", [])
                            ),
                        )
                    elif data_type == "control":
                        common_widget_actions.update_control(
                            main_window,
                            toggle_widget_name,
                            toggle_state,
                            exec_function=widget_data.get("exec_function"),
                            exec_function_args=cast(
                                list, widget_data.get("exec_function_args", [])
                            ),
                        )

                widget.toggled.connect(
                    partial(onchange_toggle, widget, widget_name, widget_data)
                )

            elif "Selection" in widget_name:
                options = widget_data["options"]
                default = widget_data["default"]

                if callable(options):
                    options = options(main_window.dfm_model_manager)
                if callable(default):
                    default = default(main_window.dfm_model_manager)

                widget = widget_components.SelectionBox(
                    label=cast(str, widget_data["label"]),
                    widget_name=widget_name,
                    group_layout_data=widgets,
                    label_widget=label,
                    main_window=main_window,
                    default_value=default,
                    selection_values=options,
                )

                widget.addItems(cast(List[str], options))
                widget.setCurrentText(cast(str, default))

                widget.reset_default_button = (
                    widget_components.ParameterResetDefaultButton(related_widget=widget)
                )
                row_widget, horizontal_layout = add_horizontal_layout_to_category(
                    category_layout, label, widget, widget.reset_default_button
                )

                if data_type == "parameter":
                    common_widget_actions.create_default_parameter(
                        main_window, widget_name, default
                    )
                else:
                    common_widget_actions.create_control(
                        main_window, widget_name, default
                    )

                def onchange_selection(
                    selection_widget: widget_components.SelectionBox,
                    selection_widget_name,
                    widget_data: dict,
                    selected_value=False,
                ):
                    actual_value = selection_widget.currentData()
                    if actual_value is None:
                        actual_value = selected_value
                    if data_type == "parameter":
                        common_widget_actions.update_parameter(
                            main_window,
                            selection_widget_name,
                            actual_value,
                            enable_refresh_frame=selection_widget.enable_refresh_frame,
                            exec_function=widget_data.get("exec_function"),
                            exec_function_args=cast(
                                list, widget_data.get("exec_function_args", [])
                            ),
                        )
                    elif data_type == "control":
                        common_widget_actions.update_control(
                            main_window,
                            selection_widget_name,
                            actual_value,
                            exec_function=widget_data.get("exec_function"),
                            exec_function_args=cast(
                                list, widget_data.get("exec_function_args", [])
                            ),
                        )

                widget.currentTextChanged.connect(
                    partial(onchange_selection, widget, widget_name, widget_data)
                )

            elif "DecimalSlider" in widget_name:
                widget = widget_components.ParameterDecimalSlider(
                    label=cast(str, widget_data["label"]),
                    widget_name=widget_name,
                    group_layout_data=widgets,
                    label_widget=label,
                    min_value=float(
                        cast(Union[int, float, str], widget_data["min_value"])
                    ),
                    max_value=float(
                        cast(Union[int, float, str], widget_data["max_value"])
                    ),
                    default_value=float(
                        cast(Union[int, float, str], widget_data["default"])
                    ),
                    decimals=int(cast(Union[int, float, str], widget_data["decimals"])),
                    step_size=float(cast(Union[int, float, str], widget_data["step"])),
                    main_window=main_window,
                )
                widget.line_edit = widget_components.ParameterLineDecimalEdit(
                    min_value=float(
                        cast(Union[int, float, str], widget_data["min_value"])
                    ),
                    max_value=float(
                        cast(Union[int, float, str], widget_data["max_value"])
                    ),
                    default_value=str(widget_data["default"]),
                    decimals=int(cast(Union[int, float, str], widget_data["decimals"])),
                    step_size=float(cast(Union[int, float, str], widget_data["step"])),
                    fixed_width=48,
                    max_length=7
                    if int(cast(Union[int, float, str], widget_data["decimals"])) > 1
                    else 5,
                )
                widget.reset_default_button = (
                    widget_components.ParameterResetDefaultButton(related_widget=widget)
                )
                row_widget, horizontal_layout = add_horizontal_layout_to_category(
                    category_layout,
                    label,
                    widget,
                    widget.line_edit,
                    widget.reset_default_button,
                )

                if data_type == "parameter":
                    common_widget_actions.create_default_parameter(
                        main_window,
                        widget_name,
                        float(cast(Union[int, float, str], widget_data["default"])),
                    )
                else:
                    common_widget_actions.create_control(
                        main_window,
                        widget_name,
                        float(cast(Union[int, float, str], widget_data["default"])),
                    )

                def onchange_decimal_slider(
                    slider_widget: widget_components.ParameterDecimalSlider,
                    slider_widget_name,
                    widget_data: dict,
                    new_value=False,
                ):
                    actual_value = slider_widget.value()
                    if data_type == "parameter":
                        common_widget_actions.update_parameter(
                            main_window,
                            slider_widget_name,
                            actual_value,
                            enable_refresh_frame=slider_widget.enable_refresh_frame,
                        )
                    elif data_type == "control":
                        common_widget_actions.update_control(
                            main_window,
                            slider_widget_name,
                            actual_value,
                            exec_function=widget_data.get("exec_function"),
                            exec_function_args=cast(
                                list, widget_data.get("exec_function_args", [])
                            ),
                        )
                    slider_widget.line_edit.set_value(actual_value)

                widget.debounce_timer.timeout.connect(
                    partial(onchange_decimal_slider, widget, widget_name, widget_data)
                )

                def onchange_decimal_line_edit(
                    slider_widget: widget_components.ParameterDecimalSlider,
                    slider_widget_name: str,
                    widget_data: dict,
                    new_value=False,
                ):
                    if not new_value:
                        new_value = 0.0
                    try:
                        new_value = float(new_value)
                    except ValueError:
                        new_value = slider_widget.value()
                    if new_value > (
                        slider_widget.max_value / slider_widget.scale_factor
                    ):
                        new_value = slider_widget.max_value / slider_widget.scale_factor
                    elif new_value < (
                        slider_widget.min_value / slider_widget.scale_factor
                    ):
                        new_value = slider_widget.min_value / slider_widget.scale_factor
                    slider_widget.setValue(new_value)
                    slider_widget.line_edit.set_value(new_value)
                    if data_type == "parameter":
                        common_widget_actions.update_parameter(
                            main_window,
                            slider_widget_name,
                            new_value,
                            enable_refresh_frame=slider_widget.enable_refresh_frame,
                        )
                    elif data_type == "control":
                        common_widget_actions.update_control(
                            main_window,
                            slider_widget_name,
                            new_value,
                            exec_function=widget_data.get("exec_function"),
                            exec_function_args=cast(
                                list, widget_data.get("exec_function_args", [])
                            ),
                        )

                widget.line_edit.textChanged.connect(
                    partial(
                        onchange_decimal_line_edit, widget, widget_name, widget_data
                    )
                )

            elif "Slider" in widget_name:
                widget = widget_components.ParameterSlider(
                    label=cast(str, widget_data["label"]),
                    widget_name=widget_name,
                    group_layout_data=widgets,
                    label_widget=label,
                    min_value=widget_data["min_value"],
                    max_value=widget_data["max_value"],
                    default_value=widget_data["default"],
                    step_size=widget_data["step"],
                    main_window=main_window,
                )
                if widget_data.get("enable_refresh_frame") is False:
                    widget.enable_refresh_frame = False
                widget.line_edit = widget_components.ParameterLineEdit(
                    min_value=int(
                        cast(Union[int, float, str], widget_data["min_value"])
                    ),
                    max_value=int(
                        cast(Union[int, float, str], widget_data["max_value"])
                    ),
                    default_value=str(widget_data["default"]),
                )
                widget.reset_default_button = (
                    widget_components.ParameterResetDefaultButton(related_widget=widget)
                )
                _slider_row_widgets: list = [
                    label,
                    widget,
                    widget.line_edit,
                    widget.reset_default_button,
                ]
                if "action_button" in widget_data:
                    _ab_data: dict = cast(dict, widget_data["action_button"])
                    _action_btn = create_layout_action_button(_ab_data)
                    _slider_row_widgets.append(_action_btn)
                row_widget, horizontal_layout = add_horizontal_layout_to_category(
                    category_layout,
                    *_slider_row_widgets,
                )
                if "below_row_button" in widget_data:
                    _below_ab_data: dict = cast(dict, widget_data["below_row_button"])
                    _below_action_btn = create_layout_action_button(_below_ab_data)
                    _below_spacer = QtWidgets.QWidget()
                    _below_spacer.setSizePolicy(
                        QtWidgets.QSizePolicy.Policy.Expanding,
                        QtWidgets.QSizePolicy.Policy.Maximum,
                    )
                    _below_row_widget, _below_horizontal_layout = (
                        add_horizontal_layout_to_category(
                            category_layout,
                            _below_action_btn,
                            _below_spacer,
                        )
                    )
                    widget.below_row_widget = _below_row_widget

                if data_type == "parameter":
                    common_widget_actions.create_default_parameter(
                        main_window,
                        widget_name,
                        int(cast(Union[int, float, str], widget_data["default"])),
                    )
                else:
                    common_widget_actions.create_control(
                        main_window,
                        widget_name,
                        int(cast(Union[int, float, str], widget_data["default"])),
                    )

                def onchange_int_slider(
                    slider_widget: widget_components.ParameterSlider,
                    slider_widget_name,
                    widget_data: dict,
                    new_value=False,
                ):
                    if data_type == "parameter":
                        common_widget_actions.update_parameter(
                            main_window,
                            slider_widget_name,
                            new_value,
                            enable_refresh_frame=slider_widget.enable_refresh_frame,
                        )
                    elif data_type == "control":
                        common_widget_actions.update_control(
                            main_window,
                            slider_widget_name,
                            new_value,
                            exec_function=widget_data.get("exec_function"),
                            exec_function_args=cast(
                                list, widget_data.get("exec_function_args", [])
                            ),
                        )
                    slider_widget.line_edit.setText(str(new_value))

                widget.debounce_timer.timeout.connect(
                    partial(onchange_int_slider, widget, widget_name, widget_data)
                )

                def onchange_int_line_edit(
                    slider_widget: widget_components.ParameterSlider,
                    slider_widget_name,
                    widget_data,
                    new_value=False,
                ):
                    if not new_value:
                        new_value = 0
                    try:
                        new_value = int(new_value)
                    except ValueError:
                        new_value = slider_widget.value()
                    if new_value > slider_widget.max_value:
                        new_value = slider_widget.max_value
                    elif new_value < slider_widget.min_value:
                        new_value = slider_widget.min_value
                    slider_widget.line_edit.set_value(new_value)
                    slider_widget.setValue(int(new_value))
                    if data_type == "parameter":
                        common_widget_actions.update_parameter(
                            main_window,
                            slider_widget_name,
                            new_value,
                            enable_refresh_frame=slider_widget.enable_refresh_frame,
                        )
                    elif data_type == "control":
                        common_widget_actions.update_control(
                            main_window,
                            slider_widget_name,
                            new_value,
                            exec_function=widget_data.get("exec_function"),
                            exec_function_args=cast(
                                list, widget_data.get("exec_function_args", [])
                            ),
                        )

                widget.line_edit.textChanged.connect(
                    partial(onchange_int_line_edit, widget, widget_name, widget_data)
                )

            elif "Text" in widget_name:

                def on_enter_pressed(
                    text_widget: widget_components.ParameterText,
                    text_widget_name,
                    widget_data,
                ):
                    new_value = text_widget.text()
                    if data_type == "parameter":
                        common_widget_actions.update_parameter(
                            main_window,
                            text_widget_name,
                            new_value,
                            enable_refresh_frame=text_widget.enable_refresh_frame,
                        )
                    else:
                        common_widget_actions.update_control(
                            main_window,
                            text_widget_name,
                            new_value,
                            exec_function=cast(
                                Callable, widget_data.get("exec_function")
                            ),
                            exec_function_args=cast(
                                list, widget_data.get("exec_function_args", [])
                            ),
                        )

                widget = widget_components.ParameterText(
                    default_value=cast(str, widget_data["default"]),
                    fixed_width=cast(int, widget_data["width"]),
                    max_length=256,
                )
                widget.returnPressed.connect(
                    partial(on_enter_pressed, widget, widget_name, widget_data)
                )
                widget.label_widget = label
                widget.widget_name = widget_name
                widget.group_layout_data = widgets
                widget.main_window = main_window
                widget.enable_refresh_frame = True

                widget.reset_default_button = (
                    widget_components.ParameterResetDefaultButton(related_widget=widget)
                )
                row_widget, horizontal_layout = add_horizontal_layout_to_category(
                    category_layout, label, widget, widget.reset_default_button
                )

                if data_type == "parameter":
                    common_widget_actions.create_default_parameter(
                        main_window, widget_name, cast(str, widget_data["default"])
                    )
                else:
                    common_widget_actions.create_control(
                        main_window, widget_name, cast(str, widget_data["default"])
                    )

                # The exec_function is now handled within the ParameterLineEdit itself
                # widget.returnPressed.connect(partial(on_enter_pressed, widget, widget_name))

            horizontal_layout.setContentsMargins(spacing_level * 10, 0, 0, 0)
            widget.row_widget = row_widget
            main_window.parameter_widgets[widget_name] = widget

        category_layout.setVerticalSpacing(2)
        category_layout.setHorizontalSpacing(2)
        layout.addWidget(group_box)

    layout.addStretch(1)

    layoutWidget.addWidget(scroll_area)

    # Default show/hide widgets
    for category, widgets in LAYOUT_DATA.items():
        for widget_name, widget_data in widgets.items():
            widget = main_window.parameter_widgets[widget_name]
            common_widget_actions.show_hide_related_widgets(
                main_window, widget, widget_name
            )


def show_hide_faces_panel(main_window: "MainWindow", checked):
    if checked:
        main_window.facesPanelGroupBox.show()
    else:
        main_window.facesPanelGroupBox.hide()
    fit_image_to_view_onchange(main_window)


def show_hide_input_target_media_panel(main_window: "MainWindow", checked):
    if checked:
        main_window.input_Target_DockWidget.show()
    else:
        main_window.input_Target_DockWidget.hide()
    fit_image_to_view_onchange(main_window)


def show_hide_input_faces_panel(main_window: "MainWindow", checked):
    if checked:
        main_window.input_Faces_DockWidget.show()
    else:
        main_window.input_Faces_DockWidget.hide()
    fit_image_to_view_onchange(main_window)


def show_hide_input_jobs_panel(main_window: "MainWindow", checked):
    if checked:
        main_window.jobManagerDockWidget.show()
    else:
        main_window.jobManagerDockWidget.hide()
    fit_image_to_view_onchange(main_window)


def show_hide_parameters_panel(main_window: "MainWindow", checked):
    if checked:
        main_window.controlOptionsDockWidget.show()
    else:
        main_window.controlOptionsDockWidget.hide()
    fit_image_to_view_onchange(main_window)


def show_hide_theatre_mode_panels(main_window: "MainWindow", checked):
    # Collects the current state of all visible panel toggle actions/state
    def collect_states():
        return {
            "target_media": main_window.panel_visibility_state.get(
                "target_media", True
            ),
            "faces": main_window.panel_visibility_state.get("faces", True),
            "parameters": main_window.panel_visibility_state.get("parameters", True),
            "input_faces": main_window.panel_visibility_state.get("input_faces", True),
            "jobs": main_window.panel_visibility_state.get("jobs", True),
        }

    # Applies the saved states to the visible View-menu actions/state
    def apply_states(states):
        main_window._set_panel_visibility(
            "target_media", states.get("target_media", True)
        )
        main_window._set_panel_visibility("faces", states.get("faces", True))
        main_window._set_panel_visibility("parameters", states.get("parameters", True))
        main_window._set_panel_visibility(
            "input_faces", states.get("input_faces", True)
        )
        main_window._set_panel_visibility("jobs", states.get("jobs", True))

    if checked:
        # Entering Theatre Mode: Save normal states and apply theatre states (default all False/Hidden)
        main_window._theatre_normal_panel_states = collect_states()
        apply_states(
            main_window._theatre_mode_panel_states
            or {
                "target_media": False,
                "faces": False,
                "parameters": False,
                "input_faces": False,
                "jobs": False,
            }
        )
    else:
        # Exiting Theatre Mode: Save theatre states and restore normal states (default all True/Visible)
        main_window._theatre_mode_panel_states = collect_states()
        apply_states(
            main_window._theatre_normal_panel_states
            or {
                "target_media": True,
                "faces": True,
                "parameters": True,
                "input_faces": True,
                "jobs": True,
            }
        )
        main_window._theatre_normal_panel_states = None

    fit_image_to_view_onchange(main_window)


def fit_image_to_view_onchange(main_window: "MainWindow", *args):
    pixmap_item = None
    # Secure image search
    for item in main_window.scene.items():
        if isinstance(item, QtWidgets.QGraphicsPixmapItem):
            pixmap_item = item
            break

    if pixmap_item:
        scene_rect = pixmap_item.boundingRect()
        QtCore.QTimer.singleShot(
            0,
            partial(
                graphics_view_actions.fit_image_to_view,
                main_window,
                pixmap_item,
                scene_rect,
            ),
        )


def set_up_menu_actions(main_window: "MainWindow"):
    if getattr(main_window, "_menu_actions_setup_installed", False):
        return

    if not hasattr(main_window, "actionEdit_CopyParameters"):
        main_window.actionEdit_CopyParameters = QtGui.QAction(
            "Copy Parameters from Selected Face", main_window
        )
        main_window.actionEdit_PasteParameters = QtGui.QAction(
            "Paste Parameters to Selected Face", main_window
        )
        main_window.actionEdit_ResetParameters = QtGui.QAction(
            "Reset Selected Face Parameters", main_window
        )
        main_window.menuEdit.clear()
        main_window.menuEdit.addAction(main_window.actionEdit_CopyParameters)
        main_window.menuEdit.addAction(main_window.actionEdit_PasteParameters)
        main_window.menuEdit.addSeparator()
        main_window.menuEdit.addAction(main_window.actionEdit_ResetParameters)

    if not hasattr(main_window, "actionHelp_QuickStartGuide"):
        main_window.actionHelp_QuickStartGuide = QtGui.QAction(
            "Quick Start Guide", main_window
        )
        main_window.actionHelp_UserManual = QtGui.QAction("User Manual", main_window)
        main_window.menuHelp.insertAction(
            main_window.actionView_Help_Shortcuts,
            main_window.actionHelp_QuickStartGuide,
        )
        main_window.menuHelp.insertAction(
            main_window.actionView_Help_Shortcuts,
            main_window.actionHelp_UserManual,
        )
        main_window.menuHelp.insertSeparator(main_window.actionView_Help_Shortcuts)

    if not hasattr(main_window, "actionHelp_About"):
        main_window.actionHelp_About = QtGui.QAction("About", main_window)
        main_window.menuHelp.addSeparator()
        main_window.menuHelp.addAction(main_window.actionHelp_About)

    main_window.actionLoad_SavedWorkspace.triggered.connect(
        partial(
            save_load_actions.load_saved_workspace,
            main_window,
        )
    )
    main_window.actionReset_to_LastWorkspace.triggered.connect(
        partial(
            save_load_actions.load_saved_workspace,
            main_window,
            str(main_window.last_workspace_path),
        )
    )
    main_window.actionSave_LastWorkspace.triggered.connect(
        partial(
            save_load_actions.save_current_workspace,
            main_window,
            str(main_window.last_workspace_path),
        )
    )
    main_window.actionSave_CurrentWorkspace.triggered.connect(
        partial(
            save_load_actions.save_current_workspace,
            main_window,
        )
    )

    def quit_without_saving(_checked=False):
        # close() rather than QCoreApplication.quit() so closeEvent still runs and
        # the processing threads are shut down cleanly.
        main_window.quit_without_saving = True
        main_window.close()

    main_window.actionQuit_WithoutSaving.triggered.connect(quit_without_saving)
    main_window.actionQuit.triggered.connect(lambda _checked=False: main_window.close())

    main_window.actionOpen_Videos_Folder.triggered.connect(
        partial(list_view_actions.select_target_medias, main_window, "folder")
    )
    main_window.actionOpen_Video_Files.triggered.connect(
        partial(list_view_actions.select_target_medias, main_window, "files")
    )
    main_window.actionLoad_Source_Image_Files.triggered.connect(
        partial(list_view_actions.select_input_face_images, main_window, "files")
    )
    main_window.actionLoad_Source_Images_Folder.triggered.connect(
        partial(list_view_actions.select_input_face_images, main_window, "folder")
    )
    main_window.actionOpen_Target_Media_Folder.triggered.connect(
        partial(list_view_actions.open_target_media_folder, main_window)
    )
    main_window.actionOpen_Input_Faces_Folder.triggered.connect(
        partial(list_view_actions.open_input_faces_folder, main_window)
    )
    main_window.actionOpen_Output_Folder.triggered.connect(
        partial(list_view_actions.open_output_media_folder, main_window)
    )
    main_window.actionLoad_Embeddings.triggered.connect(
        partial(save_load_actions.open_embeddings_from_file, main_window)
    )
    main_window.actionSave_Embeddings.triggered.connect(
        partial(save_load_actions.save_embeddings_to_file, main_window)
    )
    main_window.actionSave_Embeddings_As.triggered.connect(
        partial(save_load_actions.save_embeddings_to_file, main_window, True)
    )
    main_window.actionView_Fullscreen_F11.triggered.connect(
        partial(video_control_actions.view_fullscreen, main_window)
    )
    main_window.actionEdit_CopyParameters.triggered.connect(
        partial(common_widget_actions.copy_selected_face_parameters, main_window)
    )
    main_window.actionEdit_PasteParameters.triggered.connect(
        partial(common_widget_actions.paste_selected_face_parameters, main_window)
    )
    main_window.actionEdit_ResetParameters.triggered.connect(
        partial(common_widget_actions.reset_selected_face_parameters, main_window)
    )
    main_window.actionView_Help_Shortcuts.triggered.connect(
        partial(list_view_actions.show_shortcuts, main_window)
    )
    main_window.actionView_Help_Presets.triggered.connect(
        partial(list_view_actions.show_presets, main_window)
    )
    main_window.actionHelp_QuickStartGuide.triggered.connect(
        partial(list_view_actions._open_about_link, main_window, "quickstart")
    )
    main_window.actionHelp_UserManual.triggered.connect(
        partial(list_view_actions._open_about_link, main_window, "manual")
    )
    main_window.actionHelp_About.triggered.connect(
        partial(list_view_actions.show_about, main_window)
    )
    main_window._menu_actions_setup_installed = True


def set_all_parameters_and_control_widgets_enabled(
    main_window: "MainWindow", enabled: bool
):
    disabled = not enabled
    main_window.viewer_mode_actions_enabled = enabled

    # Bottom buttons
    main_window.saveImageButton.setDisabled(disabled)
    main_window.batchImageButton.setDisabled(disabled)
    main_window.batchallImageButton.setDisabled(disabled)
    main_window.findTargetFacesButton.setDisabled(disabled)
    main_window.clearTargetFacesButton.setDisabled(disabled)
    main_window.swapfacesButton.setDisabled(disabled)
    main_window.editFacesButton.setDisabled(disabled)
    main_window.openEmbeddingButton.setDisabled(disabled)
    main_window.saveEmbeddingButton.setDisabled(disabled)
    main_window.saveEmbeddingAsButton.setDisabled(disabled)

    # Video control buttons
    main_window.videoSeekSlider.setDisabled(disabled)
    main_window.addMarkerButton.setDisabled(disabled)
    main_window.removeMarkerButton.setDisabled(disabled)
    main_window.nextMarkerButton.setDisabled(disabled)
    main_window.previousMarkerButton.setDisabled(disabled)
    main_window.frameAdvanceButton.setDisabled(disabled)
    main_window.frameRewindButton.setDisabled(disabled)
    for attr_name in (
        "scanToolsToggleButton",
        "runScanButton",
        "clearScanResultsButton",
        "prevIssueButton",
        "nextIssueButton",
        "dropFrameButton",
        "dropAllIssueFramesButton",
        "clearDroppedFramesButton",
    ):
        if hasattr(main_window, attr_name):
            getattr(main_window, attr_name).setDisabled(disabled)

    # Compare/mask toolbar toggles
    if hasattr(main_window, "faceCompareToggleButton"):
        main_window.faceCompareToggleButton.setDisabled(disabled)
    if hasattr(main_window, "faceMaskToggleButton"):
        main_window.faceMaskToggleButton.setDisabled(disabled)

    # List items
    for _, embed_button in main_window.merged_embeddings.items():
        embed_button.setDisabled(disabled)
    for _, target_media_button in main_window.target_videos.items():
        target_media_button.setDisabled(disabled)
    for _, input_face_button in main_window.input_faces.items():
        input_face_button.setDisabled(disabled)
    for _, target_face_button in main_window.target_faces.items():
        target_face_button.setDisabled(disabled)

    # Parameters and controls dict widgets - SECURED
    for _, widget in main_window.parameter_widgets.items():
        if not widget:
            continue

        widget.setDisabled(disabled)

        # Check safely if the attributes exist before disabling them
        reset_btn = getattr(widget, "reset_default_button", None)
        if reset_btn:
            reset_btn.setDisabled(disabled)

        label_w = getattr(widget, "label_widget", None)
        if label_w:
            label_w.setDisabled(disabled)

        line_e = getattr(widget, "line_edit", None)
        if line_e:
            line_e.setDisabled(disabled)


def disable_all_parameters_and_control_widget(main_window: "MainWindow"):
    set_all_parameters_and_control_widgets_enabled(main_window, False)


def enable_all_parameters_and_control_widget(main_window: "MainWindow"):
    set_all_parameters_and_control_widgets_enabled(main_window, True)
