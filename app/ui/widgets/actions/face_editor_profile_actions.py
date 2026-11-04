"""Save and apply face-editor profiles."""

import json
import math
import os
import re
import tempfile
from pathlib import Path

from PySide6 import QtCore, QtWidgets

PROFILES_DIR = Path(__file__).resolve().parents[4] / "profiles" / "face_editor"
DEFAULT_PROFILE_NAME = "none"

FACE_EDITOR_SETTINGS_KEYS = [
    "FaceEditorBeforeTypeSelection",
    "FaceEditorCropScaleDecimalSlider",
    "FaceEditorVYRatioDecimalSlider",
    "FaceEditorBlurAmountSlider",
    "FaceEditorTypeSelection",
    "EyesOpenRatioDecimalSlider",
    "LipsOpenRatioDecimalSlider",
    "HeadPitchSlider",
    "HeadYawSlider",
    "HeadRollSlider",
    "XAxisMovementDecimalSlider",
    "YAxisMovementDecimalSlider",
    "ZAxisMovementDecimalSlider",
    "MouthPoutingDecimalSlider",
    "MouthPursingDecimalSlider",
    "MouthGrinDecimalSlider",
    "LipsCloseOpenSlider",
    "MouthSmileDecimalSlider",
    "EyeWinkDecimalSlider",
    "EyeBrowsDirectionDecimalSlider",
    "EyeGazeHorizontalDecimalSlider",
    "EyeGazeVerticalDecimalSlider",
    "FaceMakeupEnableToggle",
    "FaceMakeupRedSlider",
    "FaceMakeupGreenSlider",
    "FaceMakeupBlueSlider",
    "FaceMakeupBlendAmountDecimalSlider",
    "HairMakeupEnableToggle",
    "HairMakeupRedSlider",
    "HairMakeupGreenSlider",
    "HairMakeupBlueSlider",
    "HairMakeupBlendAmountDecimalSlider",
    "EyeBrowsMakeupEnableToggle",
    "EyeBrowsMakeupRedSlider",
    "EyeBrowsMakeupGreenSlider",
    "EyeBrowsMakeupBlueSlider",
    "EyeBrowsMakeupBlendAmountDecimalSlider",
    "LipsMakeupEnableToggle",
    "LipsMakeupRedSlider",
    "LipsMakeupGreenSlider",
    "LipsMakeupBlueSlider",
    "LipsMakeupBlendAmountDecimalSlider",
    "FaceShapingEnableToggle",
    "FaceShapingPreSwapToggle",
    "FaceShapingFaceSlimSlider",
    "FaceShapingChinSlimSlider",
    "FaceShapingForeheadSlider",
    "FaceShapingChinLengthSlider",
    "FaceShapingFaceLengthSlider",
]


def _profile_path(name):
    """Resolve a profile name without allowing traversal or unsafe Windows names."""
    if (
        not isinstance(name, str)
        or not name
        or name != name.strip()
        or name in (".", "..")
        or name.endswith(".")
        or re.search(r'[\\/:*?"<>|\x00-\x1f]', name)
        or re.fullmatch(r"(?i:con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\..*)?", name)
        or name.casefold() == DEFAULT_PROFILE_NAME
    ):
        raise ValueError("Choose a valid profile name other than 'none'.")
    root = PROFILES_DIR.resolve()
    path = root / f"{name}.json"
    if path.resolve().parent != root:
        raise ValueError("Profile must be inside the profiles folder.")
    return path


def get_profile_names():
    names = [DEFAULT_PROFILE_NAME]
    if PROFILES_DIR.is_dir():
        names.extend(
            p.stem
            for p in sorted(PROFILES_DIR.glob("*.json"))
            if p.is_file() and p.stem.casefold() != DEFAULT_PROFILE_NAME
        )
    return names


def _validate_settings(data):
    from app.ui.widgets.face_editor_layout_data import FACE_EDITOR_LAYOUT_DATA

    if not isinstance(data, dict):
        raise TypeError("The profile must contain a JSON object.")
    layout = FACE_EDITOR_LAYOUT_DATA[""]
    validated = {}
    for key in FACE_EDITOR_SETTINGS_KEYS:
        if key not in data:
            continue
        value = data[key]
        spec = layout[key]
        if "Toggle" in key:
            if type(value) is not bool:
                raise ValueError(f"{key} must be true or false.")
        elif "Selection" in key:
            if not isinstance(value, str) or value not in spec["options"]:
                raise ValueError(f"{key} has an invalid selection.")
        elif "Slider" in key:
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"{key} must be a finite number.")
            if not float(spec["min_value"]) <= value <= float(spec["max_value"]):
                raise ValueError(f"{key} is outside its allowed range.")
            if "DecimalSlider" not in key:
                if int(value) != value:
                    raise ValueError(f"{key} must be an integer.")
                value = int(value)
        validated[key] = value
    return validated


def _load_profile(name):
    with _profile_path(name).open("r", encoding="utf-8") as handle:
        return _validate_settings(json.load(handle))


def _save_profile(name, data):
    path = _profile_path(name)
    data = _validate_settings(data)
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
            json.dump(data, handle, indent=2)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _collect_settings(main_window):
    result = {}
    for key in FACE_EDITOR_SETTINGS_KEYS:
        widget = main_window.parameter_widgets.get(key)
        if widget is None:
            if key in main_window.current_widget_parameters:
                result[key] = main_window.current_widget_parameters[key]
        elif hasattr(widget, "isChecked"):
            result[key] = bool(widget.isChecked())
        elif hasattr(widget, "value"):
            result[key] = widget.value()
        elif hasattr(widget, "currentText"):
            result[key] = widget.currentText()
    return _validate_settings(result)


def _apply_settings(main_window, data):
    from app.ui.widgets.actions import common_actions

    data = _validate_settings(data)
    previous_batch = getattr(main_window, "_batch_update_in_progress", False)
    main_window._batch_update_in_progress = True
    try:
        for key, value in data.items():
            common_actions.update_parameter(
                main_window, key, value, enable_refresh_frame=False
            )
            widget = main_window.parameter_widgets.get(key)
            if widget is None:
                continue
            timer = getattr(widget, "debounce_timer", None)
            if timer is not None:
                timer.stop()
            blockers = [QtCore.QSignalBlocker(widget)]
            line_edit = getattr(widget, "line_edit", None)
            if isinstance(line_edit, QtCore.QObject):
                blockers.append(QtCore.QSignalBlocker(line_edit))
            try:
                common_actions._set_single_widget_value(widget, value)
                if isinstance(line_edit, QtCore.QObject):
                    line_edit.set_value(value)
            finally:
                for blocker in blockers:
                    blocker.unblock()
            if "Toggle" in key or "Selection" in key:
                common_actions.show_hide_related_widgets(main_window, widget, key)
    finally:
        main_window._batch_update_in_progress = previous_batch
    if not previous_batch and data:
        common_actions.refresh_frame(main_window)


def _show_error(main_window, action, error):
    QtWidgets.QMessageBox.warning(
        main_window, "Face editor profiles", f"{action}: {error}"
    )


def on_profile_changed(main_window, value, *args):
    if not value or getattr(main_window, "_batch_update_in_progress", False):
        return
    try:
        if value == DEFAULT_PROFILE_NAME:
            data = {
                key: main_window.default_parameters[key]
                for key in FACE_EDITOR_SETTINGS_KEYS
                if key in main_window.default_parameters
            }
        else:
            data = _load_profile(value)
        _apply_settings(main_window, data)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        _show_error(main_window, "Could not load profile", error)


def _confirm_overwrite(main_window, name):
    return (
        QtWidgets.QMessageBox.question(
            main_window,
            "Overwrite profile",
            f"Overwrite profile '{name}'?",
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        )
        == QtWidgets.QMessageBox.StandardButton.Yes
    )


def on_save_clicked(main_window, *args):
    selection = main_window.parameter_widgets.get("FaceEditorProfileSelection")
    if selection is None or selection.currentText() == DEFAULT_PROFILE_NAME:
        return
    name = selection.currentText()
    try:
        if _profile_path(name).exists() and not _confirm_overwrite(main_window, name):
            return
        _save_profile(name, _collect_settings(main_window))
    except (OSError, ValueError, TypeError) as error:
        _show_error(main_window, "Could not save profile", error)


def on_save_as_clicked(main_window, *args):
    selection = main_window.parameter_widgets.get("FaceEditorProfileSelection")
    name, ok = QtWidgets.QInputDialog.getText(
        main_window, "Save Profile As", "Profile name:"
    )
    if not ok:
        return
    name = name.strip()
    try:
        path = _profile_path(name)
        if path.exists() and not _confirm_overwrite(main_window, name):
            return
        _save_profile(name, _collect_settings(main_window))
    except (OSError, ValueError, TypeError) as error:
        _show_error(main_window, "Could not save profile", error)
        return
    if selection is not None:
        if selection.findText(name) < 0:
            selection.addItem(name)
        # Saving the current state must not reapply it.
        blocker = QtCore.QSignalBlocker(selection)
        selection.setCurrentText(name)
        blocker.unblock()
        from app.ui.widgets.actions import common_actions

        common_actions.update_parameter(
            main_window, "FaceEditorProfileSelection", name, enable_refresh_frame=False
        )


def on_delete_clicked(main_window, *args):
    selection = main_window.parameter_widgets.get("FaceEditorProfileSelection")
    if selection is None:
        return
    name = selection.currentText()
    if not name or name == DEFAULT_PROFILE_NAME:
        return
    reply = QtWidgets.QMessageBox.question(
        main_window,
        "Delete Profile",
        f"Delete profile '{name}'?",
        QtWidgets.QMessageBox.StandardButton.Yes
        | QtWidgets.QMessageBox.StandardButton.No,
        QtWidgets.QMessageBox.StandardButton.No,
    )
    if reply != QtWidgets.QMessageBox.StandardButton.Yes:
        return
    try:
        _profile_path(name).unlink()
    except (OSError, ValueError) as error:
        _show_error(main_window, "Could not delete profile", error)
        return
    blocker = QtCore.QSignalBlocker(selection)
    index = selection.findText(name)
    if index >= 0:
        selection.removeItem(index)
    selection.setCurrentText(DEFAULT_PROFILE_NAME)
    blocker.unblock()
    from app.ui.widgets.actions import common_actions

    common_actions.update_parameter(
        main_window,
        "FaceEditorProfileSelection",
        DEFAULT_PROFILE_NAME,
        enable_refresh_frame=False,
    )
    on_profile_changed(main_window, DEFAULT_PROFILE_NAME)
