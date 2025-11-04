import json
from pathlib import Path
from PySide6 import QtWidgets, QtCore
from typing import TYPE_CHECKING
from functools import partial
from send2trash import send2trash

from app.ui.widgets.actions import common_actions as common_widget_actions
from app.helpers.miscellaneous import ParametersDict

if TYPE_CHECKING:
    from app.ui.main_ui import MainWindow
    from PySide6.QtWidgets import QListWidgetItem


def control_preset_toggle(main_window: "MainWindow"):
    return


def handle_preset_double_click(main_window: "MainWindow", item: "QListWidgetItem"):
    """Handle double click on preset item by showing confirmation dialog"""
    result = QtWidgets.QMessageBox.question(
        main_window,
        "Apply Preset",
        f"Do you want to apply preset: {item.text()}?",
        QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
    )

    if result == QtWidgets.QMessageBox.Yes:
        apply_selected_preset(main_window)


def rename_preset(main_window: "MainWindow", item: "QListWidgetItem"):
    """Rename the selected preset"""
    old_name = item.text()
    new_name, ok = QtWidgets.QInputDialog.getText(
        main_window, "Rename Preset", "Enter new name:", text=old_name
    )

    if ok and new_name and new_name != old_name:
        old_path = Path("presets") / f"{old_name}.json"
        new_path = Path("presets") / f"{new_name}.json"
        old_path_ctl = Path("presets") / f"{old_name}_ctl.json"
        new_path_ctl = Path("presets") / f"{new_name}_ctl.json"

        if new_path.exists():
            QtWidgets.QMessageBox.warning(
                main_window,
                "Name Exists",
                f"A preset named '{new_name}' already exists.",
                QtWidgets.QMessageBox.Ok,
            )
            return

        try:
            old_path.rename(new_path)
            old_path_ctl.rename(new_path_ctl)
            refresh_presets_list(main_window)
            common_widget_actions.create_and_show_toast_message(
                main_window,
                "Preset Renamed",
                f"Renamed preset: {old_name} to {new_name}",
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                main_window,
                "Error",
                f"Failed to rename preset: {str(e)}",
                QtWidgets.QMessageBox.Ok,
            )


def delete_preset(main_window: "MainWindow", item: "QListWidgetItem"):
    """Rename the selected preset"""
    delete_preset = item.text()
    delete_path = Path("presets") / f"{delete_preset}.json"
    delete_path_ctl = Path("presets") / f"{delete_preset}_ctl.json"

    try:
        send2trash(delete_path)
        send2trash(delete_path_ctl)
        print(f"Preset: {delete_preset} has been sent to the trash.")
        refresh_presets_list(main_window)

    except Exception as e:
        QtWidgets.QMessageBox.critical(
            main_window,
            "Error",
            f"Failed to delete preset: {str(e)}",
            QtWidgets.QMessageBox.Ok,
        )


def show_preset_context_menu(main_window: "MainWindow", position):
    """Show context menu for preset list items"""
    item = main_window.presetsList.itemAt(position)
    if item:
        menu = QtWidgets.QMenu()
        rename_action = menu.addAction("Rename")
        delete_action = menu.addAction("Delete")
        action = menu.exec_(main_window.presetsList.viewport().mapToGlobal(position))

        if action == rename_action:
            rename_preset(main_window, item)

        if action == delete_action:
            delete_preset(main_window, item)


def setup_preset_list_context_menu(main_window: "MainWindow"):
    """Set up the context menu for the presets list"""
    main_window.presetsList.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
    main_window.presetsList.customContextMenuRequested.connect(
        partial(show_preset_context_menu, main_window)
    )


def refresh_presets_list(main_window: "MainWindow"):
    """Refresh the presets list with all JSON files in the presets directory"""
    main_window.presetsList.clear()
    presets_dir = Path("presets")
    if not presets_dir.exists():
        presets_dir.mkdir(exist_ok=True)

    for json_file in presets_dir.glob("*.json"):
        if not json_file.name.endswith("_ctl.json"):
            main_window.presetsList.addItem(json_file.stem)


def save_current_as_preset(main_window: "MainWindow"):
    """Save current parameters as a preset JSON file"""
    preset_list = main_window.presetsList
    selected_count = len(preset_list.selectedItems()) if preset_list else 0
    preset_name_list = []
    for i in range(preset_list.count()):
        item = preset_list.item(i)
        preset_name_list.append(item.text())
    if selected_count > 1:
        QtWidgets.QMessageBox.warning(
            main_window,
            "Selection error.",
            "Select only one preset to overwrite or deselect all to create a new preset.",
            QtWidgets.QMessageBox.Ok,
        )
        return

    if selected_count == 0:
        while True:
            name, ok = QtWidgets.QInputDialog.getText(
                main_window, "Save Preset", "Enter preset name:"
            )
            if not ok or not name:
                return
            if name in preset_name_list:
                ok = QtWidgets.QMessageBox.question(
                    main_window,
                    "Confirm Overwrite",
                    f"Preset: {name} already exists, overwrite ?",
                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                )
                if ok == QtWidgets.QMessageBox.Yes:
                    ok = True
                    break
                if ok == QtWidgets.QMessageBox.No:
                    continue
            else:
                break
    else:
        current_item = main_window.presetsList.currentItem()
        name = current_item.text()
        ok = QtWidgets.QMessageBox.question(
            main_window,
            "Confirm Overwrite",
            f"Are you sure you want to overwrite preset: {name}?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
        )
        if ok == QtWidgets.QMessageBox.Yes:
            ok = True
        if ok == QtWidgets.QMessageBox.No:
            return

    if ok and name:
        preset_path = Path("presets") / f"{name}.json"
        preset_path.parent.mkdir(exist_ok=True)
        preset_path_ctl = Path("presets") / f"{name}_ctl.json"
        preset_path_ctl.parent.mkdir(exist_ok=True)

        # Get current parameters but exclude input/output paths
        current_params = {}
        if main_window.selected_target_face_id:
            params = main_window.parameters[
                main_window.selected_target_face_id
            ].data.copy()
            # Remove any input/output specific settings
            params.pop("InputFolder", None)
            params.pop("OutputFolder", None)
            current_params = params
        else:
            params = main_window.current_widget_parameters.data.copy()
            params.pop("InputFolder", None)
            params.pop("OutputFolder", None)
            current_params = params
        current_ctl = {}
        control = main_window.control.copy()
        current_ctl = control
        # Save to file
        with open(preset_path, "w") as f:
            json.dump(current_params, f, indent=4)
        with open(preset_path_ctl, "w") as c:
            json.dump(current_ctl, c, indent=4)
        refresh_presets_list(main_window)
        common_widget_actions.create_and_show_toast_message(
            main_window, "Preset Saved", f"Saved preset: {name}"
        )


def apply_selected_preset(main_window: "MainWindow"):
    """Apply the selected preset while preserving input/output directories"""
    current_item = main_window.presetsList.currentItem()
    if not current_item:
        return

    preset_path = Path("presets") / f"{current_item.text()}.json"
    if not preset_path.exists():
        return

    with open(preset_path, "r") as f:
        preset_params = json.load(f)

    preset_path_ctl = Path("presets") / f"{current_item.text()}_ctl.json"
    if not preset_path_ctl.exists():
        return

    with open(preset_path_ctl, "r") as c:
        preset_ctl = json.load(c)
    # Preserve current input/output directories
    if main_window.selected_target_face_id:
        current_params = main_window.parameters[main_window.selected_target_face_id]

        input_folder = current_params.get("InputFolder")
        output_folder = current_params.get("OutputFolder")

        # Update parameters with preset while preserving paths
        new_params = preset_params.copy()
        if input_folder:
            new_params["InputFolder"] = input_folder
        if output_folder:
            new_params["OutputFolder"] = output_folder

        main_window.parameters[main_window.selected_target_face_id] = ParametersDict(
            new_params, main_window.default_parameters.data
        )  # type: ignore
        if main_window.selected_target_face_id == main_window.selected_target_face_id:
            common_widget_actions.set_widgets_values_using_face_id_parameters(
                main_window, main_window.selected_target_face_id
            )
        else:
            # Handle case when no face is selected
            current_input = main_window.current_widget_parameters.get("InputFolder", "")
            current_output = main_window.current_widget_parameters.get(
                "OutputFolder", ""
            )

            new_params = preset_params.copy()
            if current_input:
                new_params["InputFolder"] = current_input
            if current_output:
                new_params["OutputFolder"] = current_output

        main_window.current_widget_parameters = ParametersDict(
            new_params, main_window.default_parameters.data
        )  # type: ignore
        common_widget_actions.set_widgets_values_using_face_id_parameters(
            main_window, False
        )

    if main_window.controlPresetButton.isChecked():
        new_ctl = preset_ctl.copy()
        main_window.control.update(new_ctl)
        common_widget_actions.set_control_widgets_values(main_window)

    # Refresh the frame to show changes
    common_widget_actions.refresh_frame(main_window)
    common_widget_actions.create_and_show_toast_message(
        main_window, "Preset Applied", f"Applied preset: {current_item.text()}"
    )
