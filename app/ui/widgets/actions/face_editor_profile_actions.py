import json
from pathlib import Path
from typing import Any, Dict, List

from PySide6 import QtWidgets

PROFILES_DIR = Path(__file__).resolve().parents[4] / "profiles" / "face_editor"
PROFILES_DIR.mkdir(parents=True, exist_ok=True)
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

_FACE_EDITOR_DEFAULTS: Dict[str, Any] = {
    "FaceEditorBeforeTypeSelection": "Beginning",
    "FaceEditorCropScaleDecimalSlider": 2.50,
    "FaceEditorVYRatioDecimalSlider": -0.125,
    "FaceEditorBlurAmountSlider": 5,
    "FaceEditorTypeSelection": "Human-Face",
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
    "FaceMakeupEnableToggle": False,
    "FaceMakeupRedSlider": 0,
    "FaceMakeupGreenSlider": 0,
    "FaceMakeupBlueSlider": 0,
    "FaceMakeupBlendAmountDecimalSlider": 0.05,
    "HairMakeupEnableToggle": False,
    "HairMakeupRedSlider": 0,
    "HairMakeupGreenSlider": 0,
    "HairMakeupBlueSlider": 0,
    "HairMakeupBlendAmountDecimalSlider": 0.05,
    "EyeBrowsMakeupEnableToggle": False,
    "EyeBrowsMakeupRedSlider": 0,
    "EyeBrowsMakeupGreenSlider": 0,
    "EyeBrowsMakeupBlueSlider": 0,
    "EyeBrowsMakeupBlendAmountDecimalSlider": 0.05,
    "LipsMakeupEnableToggle": False,
    "LipsMakeupRedSlider": 0,
    "LipsMakeupGreenSlider": 0,
    "LipsMakeupBlueSlider": 0,
    "LipsMakeupBlendAmountDecimalSlider": 0.05,
    "FaceShapingEnableToggle": False,
    "FaceShapingPreSwapToggle": False,
    "FaceShapingFaceSlimSlider": 50,
    "FaceShapingChinSlimSlider": 50,
    "FaceShapingForeheadSlider": 50,
    "FaceShapingChinLengthSlider": 50,
    "FaceShapingFaceLengthSlider": 50,
}


def _profile_path(name: str) -> Path:
    return PROFILES_DIR / f"{name}.json"


def get_profile_names() -> List[str]:
    names = [DEFAULT_PROFILE_NAME]
    if PROFILES_DIR.exists():
        for p in sorted(PROFILES_DIR.glob("*.json")):
            if p.stem != DEFAULT_PROFILE_NAME and p.stem not in names:
                names.append(p.stem)
    return names


def _load_profile(name: str) -> Dict[str, Any]:
    if name == DEFAULT_PROFILE_NAME:
        return dict(_FACE_EDITOR_DEFAULTS)
    p = _profile_path(name)
    if not p.exists():
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_profile(name: str, data: Dict[str, Any]) -> bool:
    if not name or name == DEFAULT_PROFILE_NAME:
        return False
    try:
        with open(_profile_path(name), "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return True
    except Exception:
        return False


def _delete_profile(name: str) -> bool:
    if not name or name == DEFAULT_PROFILE_NAME:
        return False
    p = _profile_path(name)
    if p.exists():
        try:
            p.unlink()
            return True
        except Exception:
            return False
    return False


def _collect_settings(main_window) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key in FACE_EDITOR_SETTINGS_KEYS:
        w = main_window.parameter_widgets.get(key)
        if w is None:
            continue
        if hasattr(w, "isChecked"):
            out[key] = bool(w.isChecked())
        elif hasattr(w, "value"):
            out[key] = w.value()
        elif hasattr(w, "currentText"):
            out[key] = w.currentText()
    return out


def _set_widget_visual(widget, value):
    line = getattr(widget, "line_edit", None)
    if line is not None:
        try:
            if hasattr(line, "set_value"):
                line.set_value(value)
            elif hasattr(line, "setText"):
                line.setText(str(value))
        except Exception:
            pass
    try:
        if hasattr(widget, "setValue") and not isinstance(value, (str, bool)):
            widget.setValue(value)
    except Exception:
        pass
    try:
        if isinstance(value, bool) and hasattr(widget, "setChecked"):
            widget.setChecked(value)
    except Exception:
        pass
    try:
        if isinstance(value, str) and hasattr(widget, "setCurrentText"):
            widget.setCurrentText(value)
    except Exception:
        pass


def _apply_settings(main_window, data: Dict[str, Any]):
    from app.ui.widgets.actions import common_actions as common_widget_actions

    for key in FACE_EDITOR_SETTINGS_KEYS:
        if key not in data:
            continue
        value = data[key]
        try:
            common_widget_actions.update_parameter(main_window, key, value)
        except Exception:
            pass
        w = main_window.parameter_widgets.get(key)
        if w is not None:
            _set_widget_visual(w, value)


def on_profile_changed(main_window, value, *args):
    if not value:
        return
    data = _load_profile(value)
    _apply_settings(main_window, data)


def on_save_clicked(main_window, *args):
    sel = main_window.parameter_widgets.get("FaceEditorProfileSelection")
    if sel is None:
        return
    name = sel.currentText()
    if not name or name == DEFAULT_PROFILE_NAME:
        return
    _save_profile(name, _collect_settings(main_window))


def on_save_as_clicked(main_window, *args):
    sel = main_window.parameter_widgets.get("FaceEditorProfileSelection")
    name, ok = QtWidgets.QInputDialog.getText(
        main_window, "Save Profile As", "Profile name:"
    )
    if not ok:
        return
    name = name.strip() if name else ""
    if not name or name == DEFAULT_PROFILE_NAME:
        return
    if not _save_profile(name, _collect_settings(main_window)):
        return
    if sel is not None:
        existing = [sel.itemText(i) for i in range(sel.count())]
        if name not in existing:
            sel.addItem(name)
        sel.setCurrentText(name)


def on_delete_clicked(main_window, *args):
    sel = main_window.parameter_widgets.get("FaceEditorProfileSelection")
    if sel is None:
        return
    name = sel.currentText()
    if not name or name == DEFAULT_PROFILE_NAME:
        return
    reply = QtWidgets.QMessageBox.question(
        main_window,
        "Delete Profile",
        f"Delete profile '{name}'?",
        QtWidgets.QMessageBox.StandardButton.Yes
        | QtWidgets.QMessageBox.StandardButton.No,
    )
    if reply != QtWidgets.QMessageBox.StandardButton.Yes:
        return
    if not _delete_profile(name):
        return
    idx = sel.findText(name)
    if idx >= 0:
        sel.removeItem(idx)
    sel.setCurrentText(DEFAULT_PROFILE_NAME)
    _apply_settings(main_window, _load_profile(DEFAULT_PROFILE_NAME))