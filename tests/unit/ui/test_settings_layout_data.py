"""
SLD-* tests for app.ui.widgets.settings_layout_data

Pure data-validation tests — no Qt, no GPU, no mocks required.
These catch schema regressions when settings are added or changed.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Stub out PySide6 and action modules before importing settings_layout_data
# so this test file works without Qt installed.
# ---------------------------------------------------------------------------


def _stub_module(name: str) -> MagicMock:
    # Do NOT pass spec= — we need free attribute access so that e.g.
    # control_actions.change_theme resolves to a callable MagicMock.
    mod = MagicMock()
    mod.__name__ = name
    mod.__spec__ = None
    return mod


_QT_MODULES = [
    "PySide6",
    "PySide6.QtWidgets",
    "PySide6.QtCore",
    "PySide6.QtGui",
]
_ACTION_MODULES = [
    # Do NOT stub the parent package — it's a namespace package and stubbing it
    # prevents sibling test files from importing real submodules in the same session.
    "app.ui.widgets.actions.control_actions",
    "app.ui.widgets.actions.video_control_actions",
]

for _mod_name in _QT_MODULES + _ACTION_MODULES:
    if _mod_name not in sys.modules:
        sys.modules[_mod_name] = _stub_module(_mod_name)

# Also stub cv2 if not available (may not be in minimal env)
try:
    import cv2  # noqa: F401
except ImportError:
    sys.modules["cv2"] = MagicMock()

# Now import the real module
from app.ui.widgets.settings_layout_data import SETTINGS_LAYOUT_DATA  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _all_widget_entries():
    """Yield (category, widget_name, widget_dict) for every entry."""
    for category, widgets in SETTINGS_LAYOUT_DATA.items():
        for widget_name, widget_dict in widgets.items():
            yield category, widget_name, widget_dict


# ---------------------------------------------------------------------------
# SLD-01: all top-level keys are strings
# ---------------------------------------------------------------------------


def test_all_category_keys_are_strings():
    for key in SETTINGS_LAYOUT_DATA.keys():
        assert isinstance(key, str), f"Category key {key!r} is not a string"


# ---------------------------------------------------------------------------
# SLD-02: every entry has 'label' and 'level'
# ---------------------------------------------------------------------------


def test_every_entry_has_label_and_level():
    for cat, name, entry in _all_widget_entries():
        assert "label" in entry, f"{cat}/{name} missing 'label'"
        assert "level" in entry, f"{cat}/{name} missing 'level'"


# ---------------------------------------------------------------------------
# SLD-03: every 'options' list is non-empty
# ---------------------------------------------------------------------------


def test_options_lists_are_non_empty():
    for cat, name, entry in _all_widget_entries():
        if "options" in entry:
            assert len(entry["options"]) > 0, f"{cat}/{name} has empty 'options'"


# ---------------------------------------------------------------------------
# SLD-04: default value is in options list when both present
# ---------------------------------------------------------------------------


def test_default_in_options_when_present():
    for cat, name, entry in _all_widget_entries():
        if "options" in entry and "default" in entry:
            assert entry["default"] in entry["options"], (
                f"{cat}/{name}: default={entry['default']!r} not in options={entry['options']}"
            )


# ---------------------------------------------------------------------------
# SLD-05: VR180EyeModeSelection has correct options
# ---------------------------------------------------------------------------


def test_vr180_eye_mode_options():
    entry = SETTINGS_LAYOUT_DATA["Swap settings"]["VR180EyeModeSelection"]
    assert entry["options"] == ["Both Eyes", "Single Eye"]


# ---------------------------------------------------------------------------
# SLD-06: VR180EyeModeSelection has correct parentToggle
# ---------------------------------------------------------------------------


def test_vr180_eye_mode_parent_toggle():
    entry = SETTINGS_LAYOUT_DATA["Swap settings"]["VR180EyeModeSelection"]
    assert entry.get("parentToggle") == "VR180ModeEnableToggle"


# ---------------------------------------------------------------------------
# SLD-07: VR180EyeModeSelection has requiredToggleValue=True
# ---------------------------------------------------------------------------


def test_vr180_eye_mode_required_toggle_value():
    entry = SETTINGS_LAYOUT_DATA["Swap settings"]["VR180EyeModeSelection"]
    assert entry.get("requiredToggleValue") is True


# ---------------------------------------------------------------------------
# SLD-08: exec_function values are callable when present
# ---------------------------------------------------------------------------


def test_exec_functions_are_callable():
    for cat, name, entry in _all_widget_entries():
        if "exec_function" in entry:
            fn = entry["exec_function"]
            assert callable(fn), f"{cat}/{name}.exec_function is not callable: {fn!r}"


# ---------------------------------------------------------------------------
# SLD-09: no duplicate widget keys across all categories
# ---------------------------------------------------------------------------


def test_no_duplicate_widget_keys():
    seen: dict[str, str] = {}
    for cat, name, _ in _all_widget_entries():
        assert name not in seen, (
            f"Duplicate widget key '{name}' found in both '{seen[name]}' and '{cat}'"
        )
        seen[name] = cat


# ---------------------------------------------------------------------------
# SLD-10: VR180ModeEnableToggle exists and precedes VR180EyeModeSelection
# ---------------------------------------------------------------------------


def test_vr180_mode_toggle_exists():
    swap = SETTINGS_LAYOUT_DATA["Swap settings"]
    assert "VR180ModeEnableToggle" in swap
    assert "VR180EyeModeSelection" in swap


def test_vr180_mode_toggle_before_eye_selection():
    keys = list(SETTINGS_LAYOUT_DATA["Swap settings"].keys())
    toggle_idx = keys.index("VR180ModeEnableToggle")
    eye_idx = keys.index("VR180EyeModeSelection")
    assert toggle_idx < eye_idx, (
        "VR180ModeEnableToggle should appear before VR180EyeModeSelection"
    )


# ---------------------------------------------------------------------------
# SLD-11: parentToggle references point to existing keys
# ---------------------------------------------------------------------------


def test_parent_toggle_references_exist():
    all_keys = set(name for _, name, _ in _all_widget_entries())
    for cat, name, entry in _all_widget_entries():
        if "parentToggle" in entry:
            parent = entry["parentToggle"]
            assert parent in all_keys, (
                f"{cat}/{name}.parentToggle='{parent}' does not exist as a widget key"
            )
