from __future__ import annotations

import json
from threading import RLock
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PySide6 import QtCore, QtWidgets

from app.ui.widgets.actions import (
    common_actions,
    layout_actions,
)
from app.ui.widgets.actions import (
    face_editor_profile_actions as profiles,
)
from app.ui.widgets.face_editor_layout_data import FACE_EDITOR_LAYOUT_DATA


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def editor(qapp, monkeypatch, tmp_path):
    monkeypatch.setattr(profiles, "PROFILES_DIR", tmp_path / "profiles" / "face_editor")
    window = QtWidgets.QWidget()
    window.parameter_widgets = {}
    window.parameter_section_states = {}
    window.register_parameter_section = lambda *_args: None
    window.dfm_model_manager = None
    window.default_parameters = {}
    window.current_widget_parameters = {}
    window.parameters = {}
    window.target_faces = {}
    window.selected_target_face_id = None
    window.markers = {}
    window.videoSeekSlider = SimpleNamespace(value=lambda: 0)
    window.video_processor = SimpleNamespace(
        processing=False,
        process_current_frame=Mock(),
        state_lock=RLock(),
        feeder_parameters={},
        ui_state_is_dirty=False,
    )
    layout_actions.add_widgets_to_tab_layout(
        window,
        FACE_EDITOR_LAYOUT_DATA,
        QtWidgets.QVBoxLayout(window),
        section_namespace="face_editor",
    )
    window.current_widget_parameters = window.default_parameters.copy()
    return window


def test_storage_round_trip_and_reject_invalid_names(editor):
    data = {"HeadPitchSlider": 3, "FaceMakeupEnableToggle": True}
    assert not profiles.PROFILES_DIR.exists()
    profiles._save_profile("Smile", data)
    assert profiles._load_profile("Smile") == data
    assert profiles.get_profile_names() == ["none", "Smile"]
    for name in ("../escape", "C:\\escape", "none", "CON", "bad.", ""):
        with pytest.raises(ValueError):
            profiles._save_profile(name, data)
    assert not (profiles.PROFILES_DIR.parent / "escape.json").exists()


@pytest.mark.parametrize(
    "data",
    [
        [],
        {"HeadPitchSlider": "bad"},
        {"HeadPitchSlider": 1000},
        {"FaceMakeupEnableToggle": 1},
    ],
)
def test_invalid_profile_does_not_change_parameters(editor, data):
    profiles.PROFILES_DIR.mkdir(parents=True)
    (profiles.PROFILES_DIR / "broken.json").write_text(json.dumps(data))
    before = editor.current_widget_parameters.copy()
    with pytest.raises((TypeError, ValueError)):
        profiles._load_profile("broken")
    assert editor.current_widget_parameters == before


def test_apply_updates_face_state_and_widgets_with_one_refresh(editor):
    editor.selected_target_face_id = "face"
    editor.target_faces = {"face": object()}
    editor.parameters["face"] = editor.current_widget_parameters.copy()
    profiles._apply_settings(
        editor,
        {
            "HeadPitchSlider": 4,
            "FaceEditorVYRatioDecimalSlider": -0.123,
            "FaceMakeupEnableToggle": True,
            "FaceMakeupRedSlider": 120,
        },
    )
    assert editor.parameters["face"]["HeadPitchSlider"] == 4
    assert editor.current_widget_parameters["FaceEditorVYRatioDecimalSlider"] == -0.123
    assert editor.parameter_widgets["HeadPitchSlider"].value() == 4
    assert editor.parameter_widgets["HeadPitchSlider"].line_edit.text() == "4"
    assert editor.parameter_widgets["FaceEditorVYRatioDecimalSlider"].value() == -0.123
    assert editor.parameter_widgets["FaceMakeupEnableToggle"].isChecked()
    assert editor.video_processor.process_current_frame.call_count == 1


def test_selector_restoration_does_not_apply_profile(editor, monkeypatch):
    selection = editor.parameter_widgets["FaceEditorProfileSelection"]
    selection.addItem("Smile")
    profiles._save_profile("Smile", {"HeadPitchSlider": 10})
    editor.current_widget_parameters["FaceEditorProfileSelection"] = "Smile"
    editor.current_widget_parameters["HeadPitchSlider"] = 2
    common_actions.set_widgets_values_using_face_id_parameters(editor)
    assert selection.currentText() == "Smile"
    assert editor.current_widget_parameters["HeadPitchSlider"] == 2
    assert editor.parameter_widgets["HeadPitchSlider"].value() == 2


def test_switching_faces_does_not_reapply_profile(editor):
    selection = editor.parameter_widgets["FaceEditorProfileSelection"]
    selection.addItem("Smile")
    profiles._save_profile("Smile", {"HeadPitchSlider": 10})
    first = editor.current_widget_parameters.copy()
    second = first.copy()
    first.update(FaceEditorProfileSelection="Smile", HeadPitchSlider=3)
    second.update(FaceEditorProfileSelection="none", HeadPitchSlider=7)
    editor.parameters = {"one": first, "two": second}
    editor.target_faces = {"one": object(), "two": object()}
    editor.selected_target_face_id = "one"
    common_actions.set_widgets_values_using_face_id_parameters(editor, "one")
    assert first["HeadPitchSlider"] == 3
    editor.selected_target_face_id = "two"
    common_actions.set_widgets_values_using_face_id_parameters(editor, "two")
    assert second["HeadPitchSlider"] == 7
    assert editor.parameter_widgets["HeadPitchSlider"].value() == 7


def test_restoring_marker_does_not_replace_saved_values(editor):
    from app.ui.widgets.actions import video_control_actions

    selection = editor.parameter_widgets["FaceEditorProfileSelection"]
    selection.addItem("Smile")
    profiles._save_profile("Smile", {"HeadPitchSlider": 10})
    params = editor.current_widget_parameters.copy()
    params.update(FaceEditorProfileSelection="Smile", HeadPitchSlider=4)
    editor.selected_target_face_id = "face"
    editor.target_faces = {"face": object()}
    editor.parameters = {"face": params.copy()}
    editor.markers = {0: {"parameters": {"face": params.copy()}}}
    editor.control = {}
    video_control_actions.update_parameters_and_control_from_marker(editor, 0)
    video_control_actions.update_widget_values_from_markers(editor, 0)
    assert editor.markers[0]["parameters"]["face"]["HeadPitchSlider"] == 4
    assert editor.parameters["face"]["HeadPitchSlider"] == 4


def test_invalid_file_reports_error_without_applying(editor, monkeypatch):
    profiles.PROFILES_DIR.mkdir(parents=True)
    (profiles.PROFILES_DIR / "broken.json").write_text('{"HeadPitchSlider": "invalid"}')
    warning = Mock()
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", warning)
    before = editor.current_widget_parameters.copy()
    profiles.on_profile_changed(editor, "broken")
    warning.assert_called_once()
    assert editor.current_widget_parameters == before
    editor.video_processor.process_current_frame.assert_not_called()


def test_hidden_type_and_profile_buttons_follow_visibility(editor):
    selection = editor.parameter_widgets["FaceEditorProfileSelection"]
    toggle = editor.parameter_widgets["FaceEditorEnableToggle"]
    editor_type = editor.parameter_widgets["FaceEditorTypeSelection"]
    assert selection.row_widget.isHidden()
    assert selection.below_row_widget.isHidden()
    assert editor_type.row_widget.isHidden()
    blocker = QtCore.QSignalBlocker(toggle)
    toggle.setChecked(True)
    blocker.unblock()
    common_actions.show_hide_related_widgets(editor, toggle, "FaceEditorEnableToggle")
    assert not selection.row_widget.isHidden()
    assert not selection.below_row_widget.isHidden()
    assert editor_type.row_widget.isHidden()
    blocker = QtCore.QSignalBlocker(toggle)
    toggle.setChecked(False)
    blocker.unblock()
    common_actions.show_hide_related_widgets(editor, toggle, "FaceEditorEnableToggle")
    assert selection.below_row_widget.isHidden()


def test_save_as_requires_confirmation_before_overwrite(editor, monkeypatch):
    profiles._save_profile("Smile", {"HeadPitchSlider": 5})
    monkeypatch.setattr(
        QtWidgets.QInputDialog, "getText", lambda *_args: ("Smile", True)
    )
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "question",
        lambda *_args: QtWidgets.QMessageBox.StandardButton.No,
    )
    profiles.on_save_as_clicked(editor)
    assert profiles._load_profile("Smile")["HeadPitchSlider"] == 5


def test_save_as_creates_and_selects_profile_without_refresh(editor, monkeypatch):
    slider = editor.parameter_widgets["HeadPitchSlider"]
    slider.set_value(6)
    editor.video_processor.process_current_frame.reset_mock()
    monkeypatch.setattr(
        QtWidgets.QInputDialog, "getText", lambda *_args: ("Smile", True)
    )
    profiles.on_save_as_clicked(editor)
    assert profiles._load_profile("Smile")["HeadPitchSlider"] == 6
    assert (
        editor.parameter_widgets["FaceEditorProfileSelection"].currentText() == "Smile"
    )
    assert editor.current_widget_parameters["FaceEditorProfileSelection"] == "Smile"
    editor.video_processor.process_current_frame.assert_not_called()


def test_delete_resets_to_defaults_once(editor, monkeypatch):
    selection = editor.parameter_widgets["FaceEditorProfileSelection"]
    profiles._save_profile("Smile", {"HeadPitchSlider": 5})
    selection.addItem("Smile")
    selection.setCurrentText("Smile")
    assert editor.current_widget_parameters["HeadPitchSlider"] == 5
    editor.video_processor.process_current_frame.reset_mock()
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "question",
        lambda *_args: QtWidgets.QMessageBox.StandardButton.Yes,
    )
    profiles.on_delete_clicked(editor)
    assert not (profiles.PROFILES_DIR / "Smile.json").exists()
    assert editor.current_widget_parameters["HeadPitchSlider"] == 0
    assert editor.video_processor.process_current_frame.call_count == 1
