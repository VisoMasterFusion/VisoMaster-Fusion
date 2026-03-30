"""
Tests for widget visibility logic and DFMModelManager.

Targets:
  - show_hide_related_widgets() — the toggle-case decision logic
      (Case 2 in common_actions.py: "Toggle" in parent_widget_name)
  - DFMModelManager              — filesystem scanning (app/helpers/miscellaneous.py)

show_hide_related_widgets is tested by constructing minimal mock objects that
match the interface the function reads from, and verifying that it calls
.show() / .hide() on the child widget correctly.
No Qt display is needed — we mock at the object level.
"""

from __future__ import annotations

import sys
import os
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Stub PySide6 and all Qt-dependent imports so common_actions can be imported
# ---------------------------------------------------------------------------


def _stub(name: str) -> MagicMock:
    m = MagicMock()
    m.__name__ = name
    m.__spec__ = None
    return m


_STUBS = [
    "PySide6",
    "PySide6.QtWidgets",
    "PySide6.QtCore",
    "PySide6.QtGui",
    # Not installed in test venv — must be set before common_actions is imported
    "pyqttoast",
    "qdarkstyle",
    "qdarktheme",
    "app.ui.widgets.widget_components",
    # control_actions must be stubbed BEFORE settings_layout_data is loaded
    # (settings_layout_data.py does `from app.ui.widgets.actions import control_actions`
    # at module level, and control_actions imports qdarkstyle / qdarktheme).
    "app.ui.widgets.actions.control_actions",
    "app.ui.widgets.actions.card_actions",
    "app.ui.widgets.actions.list_view_actions",
    "app.ui.widgets.actions.video_control_actions",
    "app.ui.widgets.actions.layout_actions",
    "app.ui.widgets.actions.filter_actions",
    "app.ui.widgets.ui_workers",
]
for _s in _STUBS:
    if _s not in sys.modules:
        sys.modules[_s] = _stub(_s)

# Force-clear any stale common_actions entry that a sibling test file may have
# injected as a MagicMock stub, so we import the real module here.
sys.modules.pop("app.ui.widgets.actions.common_actions", None)
# Also clear settings_layout_data so it re-imports cleanly with our control_actions stub.
sys.modules.pop("app.ui.widgets.settings_layout_data", None)

from app.ui.widgets.actions.common_actions import show_hide_related_widgets  # noqa: E402
from app.helpers.miscellaneous import DFMModelManager  # noqa: E402


# =============================================================================
# show_hide_related_widgets — Toggle-parent logic (Case 2)
# =============================================================================
#
# The function signature:
#   show_hide_related_widgets(main_window, parent_widget, parent_widget_name,
#                             value1=False, value2=False)
#
# For the Toggle case ("Toggle" in parent_widget_name):
#   - For each child widget whose layout_info["parentToggle"] contains
#     parent_widget_name, the function evaluates the toggle(s) and calls
#     child.show() or child.hide().
#
# We build a thin mock infrastructure that matches what the function reads.
# =============================================================================


def _make_toggle_setup(
    parent_name: str,
    child_name: str,
    required_toggle_value: bool,
    parent_is_checked: bool,
    has_selection_dependency: bool = False,
):
    """
    Build minimal mocks for a single parent-toggle + one dependent child widget.

    Returns (main_window, parent_widget, child_widget).
    """
    child_widget = MagicMock()

    # Layout info for the child widget
    child_layout_info = {
        "parentToggle": parent_name,
        "requiredToggleValue": required_toggle_value,
    }
    if has_selection_dependency:
        child_layout_info["parentSelection"] = ""  # no selection dep for simplicity

    # group_layout_data: the full layout group dict the parent belongs to
    group_layout_data = {
        child_name: child_layout_info,
    }

    # The parent toggle widget
    parent_widget = MagicMock()
    parent_widget.isChecked.return_value = parent_is_checked
    parent_widget.group_layout_data = group_layout_data

    # main_window.parameter_widgets maps names to widget mocks
    parent_widget_mock = MagicMock()
    parent_widget_mock.isChecked.return_value = parent_is_checked

    main_window = MagicMock()
    main_window.parameter_widgets = {
        parent_name: parent_widget_mock,
        child_name: child_widget,
    }

    return main_window, parent_widget, child_widget


# ---------------------------------------------------------------------------
# WC-04: parent toggle ON, requiredToggleValue=True  → child shown
# ---------------------------------------------------------------------------


def test_toggle_on_required_true_shows_child():
    mw, parent, child = _make_toggle_setup(
        parent_name="VR180ModeEnableToggle",
        child_name="VR180EyeModeSelection",
        required_toggle_value=True,
        parent_is_checked=True,
    )
    show_hide_related_widgets(mw, parent, "VR180ModeEnableToggle")
    child.show.assert_called_once()
    child.hide.assert_not_called()


# ---------------------------------------------------------------------------
# WC-05: parent toggle OFF, requiredToggleValue=True  → child hidden
# ---------------------------------------------------------------------------


def test_toggle_off_required_true_hides_child():
    mw, parent, child = _make_toggle_setup(
        parent_name="VR180ModeEnableToggle",
        child_name="VR180EyeModeSelection",
        required_toggle_value=True,
        parent_is_checked=False,
    )
    show_hide_related_widgets(mw, parent, "VR180ModeEnableToggle")
    child.hide.assert_called_once()
    child.show.assert_not_called()


# ---------------------------------------------------------------------------
# Parent toggle OFF, requiredToggleValue=False  → child shown
# (e.g. a widget that should appear only when the parent is disabled)
# ---------------------------------------------------------------------------


def test_toggle_off_required_false_shows_child():
    mw, parent, child = _make_toggle_setup(
        parent_name="SomeFeatureToggle",
        child_name="DependentWidget",
        required_toggle_value=False,
        parent_is_checked=False,
    )
    show_hide_related_widgets(mw, parent, "SomeFeatureToggle")
    child.show.assert_called_once()
    child.hide.assert_not_called()


# ---------------------------------------------------------------------------
# Parent toggle ON, requiredToggleValue=False  → child hidden
# ---------------------------------------------------------------------------


def test_toggle_on_required_false_hides_child():
    mw, parent, child = _make_toggle_setup(
        parent_name="SomeFeatureToggle",
        child_name="DependentWidget",
        required_toggle_value=False,
        parent_is_checked=True,
    )
    show_hide_related_widgets(mw, parent, "SomeFeatureToggle")
    child.hide.assert_called_once()
    child.show.assert_not_called()


# ---------------------------------------------------------------------------
# Widget not in parameter_widgets → no show/hide call (graceful skip)
# ---------------------------------------------------------------------------


def test_child_not_in_parameter_widgets_is_skipped():
    parent_widget = MagicMock()
    parent_widget.isChecked.return_value = True
    parent_widget.group_layout_data = {
        "MissingWidget": {
            "parentToggle": "SomeToggle",
            "requiredToggleValue": True,
        }
    }
    mw = MagicMock()
    mw.parameter_widgets = {}  # empty — child not present
    # Should not raise
    show_hide_related_widgets(mw, parent_widget, "SomeToggle")


# ---------------------------------------------------------------------------
# parameter_widgets is falsy (None / {}) → function returns early
# ---------------------------------------------------------------------------


def test_no_parameter_widgets_returns_early():
    parent = MagicMock()
    mw = MagicMock()
    mw.parameter_widgets = {}
    # Should not raise, should not call show/hide on anything
    show_hide_related_widgets(mw, parent, "SomeToggle")


# ---------------------------------------------------------------------------
# VR180 real-world scenario: VR180ModeEnableToggle ON → EyeModeSelection visible
# ---------------------------------------------------------------------------


def test_vr180_real_scenario_toggle_on():
    """
    Simulates the actual VR180EyeModeSelection / VR180ModeEnableToggle pair
    from settings_layout_data.
    """
    eye_mode_widget = MagicMock()
    toggle_mock = MagicMock()
    toggle_mock.isChecked.return_value = True  # VR180 is ON

    parent_widget = MagicMock()
    parent_widget.isChecked.return_value = True
    parent_widget.group_layout_data = {
        "VR180EyeModeSelection": {
            "parentToggle": "VR180ModeEnableToggle",
            "requiredToggleValue": True,
        },
        # Other entries that should NOT be affected
        "AutoSwapToggle": {
            "parentToggle": "OtherToggle",
            "requiredToggleValue": True,
        },
    }

    other_widget = MagicMock()
    mw = MagicMock()
    mw.parameter_widgets = {
        "VR180ModeEnableToggle": toggle_mock,
        "VR180EyeModeSelection": eye_mode_widget,
        "AutoSwapToggle": other_widget,
    }

    show_hide_related_widgets(mw, parent_widget, "VR180ModeEnableToggle")

    eye_mode_widget.show.assert_called_once()
    eye_mode_widget.hide.assert_not_called()
    # The unrelated widget should not be shown (its parentToggle is "OtherToggle")
    other_widget.show.assert_not_called()


def test_vr180_real_scenario_toggle_off():
    eye_mode_widget = MagicMock()
    toggle_mock = MagicMock()
    toggle_mock.isChecked.return_value = False  # VR180 is OFF

    parent_widget = MagicMock()
    parent_widget.isChecked.return_value = False
    parent_widget.group_layout_data = {
        "VR180EyeModeSelection": {
            "parentToggle": "VR180ModeEnableToggle",
            "requiredToggleValue": True,
        },
    }

    mw = MagicMock()
    mw.parameter_widgets = {
        "VR180ModeEnableToggle": toggle_mock,
        "VR180EyeModeSelection": eye_mode_widget,
    }

    show_hide_related_widgets(mw, parent_widget, "VR180ModeEnableToggle")

    eye_mode_widget.hide.assert_called_once()
    eye_mode_widget.show.assert_not_called()


# =============================================================================
# DFMModelManager — filesystem scanning
# =============================================================================


class TestDFMModelManager:
    def test_empty_directory_gives_empty_data(self, tmp_path):
        mgr = DFMModelManager(models_path=str(tmp_path))
        assert mgr.get_models_data() == {}
        assert mgr.get_selection_values() == []

    def test_discovers_dfm_files(self, tmp_path):
        (tmp_path / "model_a.dfm").write_bytes(b"")
        (tmp_path / "model_b.dfm").write_bytes(b"")
        mgr = DFMModelManager(models_path=str(tmp_path))
        assert "model_a.dfm" in mgr.get_models_data()
        assert "model_b.dfm" in mgr.get_models_data()

    def test_discovers_onnx_files(self, tmp_path):
        (tmp_path / "swapper.onnx").write_bytes(b"")
        mgr = DFMModelManager(models_path=str(tmp_path))
        assert "swapper.onnx" in mgr.get_models_data()

    def test_ignores_non_model_files(self, tmp_path):
        (tmp_path / "readme.txt").write_text("docs")
        (tmp_path / "config.json").write_text("{}")
        (tmp_path / "model.dfm").write_bytes(b"")
        mgr = DFMModelManager(models_path=str(tmp_path))
        assert "readme.txt" not in mgr.get_models_data()
        assert "config.json" not in mgr.get_models_data()
        assert "model.dfm" in mgr.get_models_data()

    def test_paths_are_absolute_and_correct(self, tmp_path):
        model_file = tmp_path / "mymodel.dfm"
        model_file.write_bytes(b"")
        mgr = DFMModelManager(models_path=str(tmp_path))
        stored_path = mgr.get_models_data()["mymodel.dfm"]
        assert os.path.isabs(stored_path)
        assert stored_path == str(model_file)

    def test_nonexistent_directory_gives_empty_data(self, tmp_path):
        mgr = DFMModelManager(models_path=str(tmp_path / "nonexistent"))
        assert mgr.get_models_data() == {}

    def test_get_selection_values_returns_list_of_names(self, tmp_path):
        (tmp_path / "a.dfm").write_bytes(b"")
        (tmp_path / "b.onnx").write_bytes(b"")
        mgr = DFMModelManager(models_path=str(tmp_path))
        vals = mgr.get_selection_values()
        assert isinstance(vals, list)
        assert set(vals) == {"a.dfm", "b.onnx"}

    def test_get_default_value_returns_first_model(self, tmp_path):
        (tmp_path / "only_model.dfm").write_bytes(b"")
        mgr = DFMModelManager(models_path=str(tmp_path))
        assert mgr.get_default_value() == "only_model.dfm"

    def test_get_default_value_empty_dir_returns_empty_string(self, tmp_path):
        mgr = DFMModelManager(models_path=str(tmp_path))
        assert mgr.get_default_value() == ""

    def test_refresh_models_clears_and_rescans(self, tmp_path):
        (tmp_path / "old.dfm").write_bytes(b"")
        mgr = DFMModelManager(models_path=str(tmp_path))
        assert "old.dfm" in mgr.get_models_data()

        # Add a new file and refresh
        (tmp_path / "new.dfm").write_bytes(b"")
        mgr.refresh_models()
        assert "new.dfm" in mgr.get_models_data()

    def test_refresh_models_removes_deleted_file(self, tmp_path):
        model_file = tmp_path / "temp.dfm"
        model_file.write_bytes(b"")
        mgr = DFMModelManager(models_path=str(tmp_path))
        assert "temp.dfm" in mgr.get_models_data()

        model_file.unlink()
        mgr.refresh_models()
        assert "temp.dfm" not in mgr.get_models_data()
