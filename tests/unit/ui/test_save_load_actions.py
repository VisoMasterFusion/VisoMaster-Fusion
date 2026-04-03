"""
Tests for the pure serialization/conversion logic in save_load_actions.py.

Targets:
  - convert_parameters_to_supported_type()  — ParametersDict ↔ dict
  - convert_markers_to_supported_type()     — nested marker type conversion
  - Embedding numpy↔list round-trip         — simulated as used in save/load

All PySide6, widget, and UI imports are stubbed so this runs without Qt.
"""

from __future__ import annotations

import sys
import json
import copy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Stub every heavy import before the module is loaded
# ---------------------------------------------------------------------------


def _stub(name: str) -> MagicMock:
    m = MagicMock()
    m.__name__ = name
    m.__spec__ = None
    return m


_STUBS = [
    # Qt — not installed in test env
    "PySide6",
    "PySide6.QtWidgets",
    "PySide6.QtCore",
    "PySide6.QtGui",
    # Widget components have heavy Qt deps — stub the leaf, not the parent package
    "app.ui.widgets.widget_components",
    "app.ui.widgets.ui_workers",
    # Stub each leaf action module individually so Python can still resolve
    # the real `app.ui.widgets.actions` package and find sibling submodules.
    "app.ui.widgets.actions.common_actions",
    "app.ui.widgets.actions.card_actions",
    "app.ui.widgets.actions.list_view_actions",
    "app.ui.widgets.actions.video_control_actions",
    "app.ui.widgets.actions.layout_actions",
    "app.ui.widgets.actions.filter_actions",
]
for _s in _STUBS:
    if _s not in sys.modules:
        sys.modules[_s] = _stub(_s)

widget_components_stub = sys.modules["app.ui.widgets.widget_components"]
setattr(
    widget_components_stub,
    "TargetMediaCardButton",
    type("TargetMediaCardButton", (), {}),
)

# Provide the real ParametersDict through misc_helpers
from app.helpers.miscellaneous import ParametersDict  # noqa: E402

# Now import the module under test
from app.ui.widgets.actions.save_load_actions import (  # noqa: E402
    convert_parameters_to_supported_type,
    convert_markers_to_supported_type,
    save_current_workspace,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def default_params_data() -> dict:
    return {"brightness": 1.0, "contrast": 0.8, "sharpness": 0.5}


@pytest.fixture
def mock_main_window(default_params_data):
    mw = MagicMock()
    mw.default_parameters = ParametersDict(default_params_data, default_params_data)
    return mw


@pytest.fixture
def sample_params_dict(default_params_data) -> ParametersDict:
    return ParametersDict({"brightness": 1.5, "contrast": 0.9}, default_params_data)


@pytest.fixture
def sample_plain_dict() -> dict:
    return {"brightness": 1.5, "contrast": 0.9}


# ---------------------------------------------------------------------------
# convert_parameters_to_supported_type — ParametersDict → dict
# ---------------------------------------------------------------------------


def test_convert_parameters_dict_to_dict(mock_main_window, sample_params_dict):
    result = convert_parameters_to_supported_type(
        mock_main_window, sample_params_dict, dict
    )
    assert isinstance(result, dict)
    assert not isinstance(result, ParametersDict)
    assert result["brightness"] == 1.5


def test_convert_parameters_dict_to_dict_returns_underlying_data(
    mock_main_window, sample_params_dict
):
    """Returned dict should contain the values stored in .data, not the defaults."""
    result = convert_parameters_to_supported_type(
        mock_main_window, sample_params_dict, dict
    )
    # Only keys explicitly set in sample_params_dict — not the full defaults
    assert set(result.keys()) == {"brightness", "contrast"}


def test_convert_plain_dict_to_dict_passthrough(mock_main_window, sample_plain_dict):
    """A plain dict passed with convert_type=dict is returned as-is."""
    result = convert_parameters_to_supported_type(
        mock_main_window, sample_plain_dict, dict
    )
    assert isinstance(result, dict)
    assert result is sample_plain_dict  # exact same object


# ---------------------------------------------------------------------------
# convert_parameters_to_supported_type — dict → ParametersDict
# ---------------------------------------------------------------------------


def test_convert_dict_to_parameters_dict(
    mock_main_window, sample_plain_dict, default_params_data
):
    result = convert_parameters_to_supported_type(
        mock_main_window, sample_plain_dict, ParametersDict
    )
    assert isinstance(result, ParametersDict)
    assert result["brightness"] == 1.5


def test_convert_dict_to_parameters_dict_uses_defaults(
    mock_main_window, default_params_data
):
    """Missing keys should fall back to default_parameters."""
    result = convert_parameters_to_supported_type(mock_main_window, {}, ParametersDict)
    assert isinstance(result, ParametersDict)
    assert result["brightness"] == default_params_data["brightness"]


def test_convert_parameters_dict_to_parameters_dict_passthrough(
    mock_main_window, sample_params_dict
):
    """A ParametersDict passed with convert_type=ParametersDict is returned unchanged."""
    result = convert_parameters_to_supported_type(
        mock_main_window, sample_params_dict, ParametersDict
    )
    assert isinstance(result, ParametersDict)
    assert result is sample_params_dict


# ---------------------------------------------------------------------------
# Round-trip: ParametersDict → dict → ParametersDict
# ---------------------------------------------------------------------------


def test_round_trip_parameters_dict(
    mock_main_window, sample_params_dict, default_params_data
):
    as_dict = convert_parameters_to_supported_type(
        mock_main_window, sample_params_dict, dict
    )
    restored = convert_parameters_to_supported_type(
        mock_main_window, as_dict, ParametersDict
    )
    assert isinstance(restored, ParametersDict)
    assert restored["brightness"] == sample_params_dict["brightness"]
    assert restored["contrast"] == sample_params_dict["contrast"]


def test_round_trip_is_json_serializable(mock_main_window, sample_params_dict):
    """The dict form must be JSON-serializable (no custom objects)."""
    as_dict = convert_parameters_to_supported_type(
        mock_main_window, sample_params_dict, dict
    )
    json_str = json.dumps(as_dict)
    recovered = json.loads(json_str)
    assert recovered["brightness"] == sample_params_dict["brightness"]


# ---------------------------------------------------------------------------
# convert_markers_to_supported_type — nested conversion
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_markers(sample_params_dict, default_params_data):
    """Markers dict mimicking the real structure: {frame: {parameters: {face_id: PD}, control: {}}}"""
    return {
        100: {
            "parameters": {
                "face_1": ParametersDict({"brightness": 2.0}, default_params_data),
                "face_2": ParametersDict({"contrast": 0.3}, default_params_data),
            },
            "control": {"VR180ModeEnableToggle": False},
        },
        200: {
            "parameters": {
                "face_1": ParametersDict({"sharpness": 0.7}, default_params_data),
            },
            "control": {},
        },
    }


def test_convert_markers_to_dict(mock_main_window, sample_markers):
    result = convert_markers_to_supported_type(mock_main_window, sample_markers, dict)
    for frame_id, marker_data in result.items():
        for face_id, params in marker_data["parameters"].items():
            assert isinstance(params, dict), (
                f"Frame {frame_id}, face {face_id} should be dict"
            )
            assert not isinstance(params, ParametersDict)


def test_convert_markers_to_parameters_dict(mock_main_window, sample_markers):
    # First convert to dict form, then back to ParametersDict
    as_dict_form = convert_markers_to_supported_type(
        mock_main_window, copy.deepcopy(sample_markers), dict
    )
    # Replace ParametersDict values with plain dicts (simulate loaded JSON)
    result = convert_markers_to_supported_type(
        mock_main_window, as_dict_form, ParametersDict
    )
    for frame_id, marker_data in result.items():
        for face_id, params in marker_data["parameters"].items():
            assert isinstance(params, ParametersDict), (
                f"Frame {frame_id}, face {face_id} should be ParametersDict"
            )


def test_convert_markers_mutates_in_place(mock_main_window, sample_markers):
    """convert_markers_to_supported_type converts in-place (no deep copy).
    The caller is responsible for passing a copy if the original must be preserved."""
    original_type = type(sample_markers[100]["parameters"]["face_1"])
    assert original_type is ParametersDict  # precondition
    convert_markers_to_supported_type(mock_main_window, sample_markers, dict)
    # After conversion the nested value is now a plain dict
    assert type(sample_markers[100]["parameters"]["face_1"]) is dict


def test_convert_markers_preserves_control_dict(mock_main_window, sample_markers):
    """The 'control' sub-dict inside each marker must be preserved intact."""
    result = convert_markers_to_supported_type(
        mock_main_window, copy.deepcopy(sample_markers), dict
    )
    assert result[100]["control"]["VR180ModeEnableToggle"] is False


# ---------------------------------------------------------------------------
# Embedding numpy ↔ list round-trip (pattern used in save/load)
# ---------------------------------------------------------------------------


def test_embedding_numpy_to_list_and_back():
    """numpy arrays must survive JSON serialization via .tolist() / np.array()."""
    original = np.random.randn(512).astype(np.float32)
    as_list = original.tolist()

    json_str = json.dumps({"embedding": as_list})
    recovered_list = json.loads(json_str)["embedding"]
    recovered_array = np.array(recovered_list, dtype=np.float32)

    assert np.allclose(original, recovered_array, atol=1e-6)


def test_embedding_store_round_trip():
    """A full embedding_store dict (model→array) survives a JSON round-trip."""
    store = {
        "arcface_w600k_r50": np.random.randn(512).astype(np.float32),
        "arcface_simswap": np.random.randn(512).astype(np.float32),
    }
    serialized = {model: emb.tolist() for model, emb in store.items()}
    json_str = json.dumps(serialized)
    restored = {model: np.array(v) for model, v in json.loads(json_str).items()}

    for model, original_emb in store.items():
        assert np.allclose(original_emb, restored[model], atol=1e-6)


def test_embedding_preserves_shape():
    original = np.random.randn(4, 128).astype(np.float32)
    restored = np.array(json.loads(json.dumps(original.tolist())))
    assert restored.shape == original.shape


class _FakeGeometry:
    def __init__(self, x: int, y: int, width: int, height: int):
        self._x = x
        self._y = y
        self._width = width
        self._height = height

    def x(self):
        return self._x

    def y(self):
        return self._y

    def width(self):
        return self._width

    def height(self):
        return self._height


class _FakeByteArray:
    def __init__(self, value: str):
        self._value = value

    def toBase64(self):
        return self

    def data(self):
        return self._value.encode("utf-8")


class _FakeTabWidget:
    def __init__(self):
        self._tabs = ["Face Swap", "Settings"]

    def currentIndex(self):
        return 0

    def count(self):
        return len(self._tabs)

    def tabText(self, index: int):
        return self._tabs[index]


def _make_workspace_main_window(
    tmp_path: Path,
    *,
    is_theatre_mode: bool,
    is_full_screen: bool,
    is_maximized: bool,
    geometry: _FakeGeometry | None = None,
    normal_geometry: _FakeGeometry | None = None,
    saved_window_state: str = "live-window-state",
    was_custom_fullscreen: bool = False,
    was_maximized: bool = False,
):
    default_params_data = {"brightness": 1.0, "contrast": 0.8}
    geometry = geometry or _FakeGeometry(10, 20, 1280, 720)
    normal_geometry = normal_geometry or _FakeGeometry(100, 200, 900, 600)

    mw = SimpleNamespace()
    mw.default_parameters = ParametersDict(default_params_data, default_params_data)
    mw.is_theatre_mode = is_theatre_mode
    mw.is_full_screen = is_full_screen
    mw._was_custom_fullscreen = was_custom_fullscreen
    mw._was_maximized = was_maximized
    mw._was_normal_geometry = normal_geometry
    mw._saved_window_state = _FakeByteArray(saved_window_state)
    mw.control = {}
    mw.target_videos = {}
    mw.input_faces = {}
    mw.target_faces = {}
    mw.merged_embeddings = {}
    mw.markers = {}
    mw.issue_frames_by_face = {}
    mw.dropped_frames = set()
    mw.job_marker_pairs = []
    mw.last_target_media_folder_path = ""
    mw.last_input_media_folder_path = ""
    mw.loaded_embedding_filename = ""
    mw.current_widget_parameters = ParametersDict({}, default_params_data)
    mw.tabWidget = _FakeTabWidget()
    mw.selected_video_button = False
    mw.panel_visibility_state = {
        "target_media": True,
        "input_faces": True,
        "jobs": True,
        "faces": True,
        "parameters": True,
    }
    mw.filterImagesCheckBox = SimpleNamespace(isChecked=lambda: True)
    mw.filterVideosCheckBox = SimpleNamespace(isChecked=lambda: True)
    mw.filterWebcamsCheckBox = SimpleNamespace(isChecked=lambda: False)
    mw.scan_tools_expanded = False
    mw.project_root_path = tmp_path
    mw.geometry = lambda: geometry
    mw.isMaximized = lambda: is_maximized
    mw.saveState = lambda: _FakeByteArray("live-window-state")
    return mw


def _read_saved_workspace(path: Path) -> dict:
    return json.loads(path.read_text())


def test_save_workspace_non_theatre_uses_live_window_state(tmp_path):
    save_path = tmp_path / "workspace.json"
    geometry = _FakeGeometry(5, 6, 700, 500)
    main_window = _make_workspace_main_window(
        tmp_path,
        is_theatre_mode=False,
        is_full_screen=True,
        is_maximized=False,
        geometry=geometry,
    )

    save_current_workspace(main_window, str(save_path))

    saved = _read_saved_workspace(save_path)["window_state_data"]
    assert saved["isFullScreen"] is True
    assert saved["isMaximized"] is False
    assert saved["dock_state"] == "live-window-state"
    assert (saved["x"], saved["y"], saved["width"], saved["height"]) == (5, 6, 700, 500)


def test_save_workspace_theatre_from_fullscreen_uses_live_window_state(tmp_path):
    save_path = tmp_path / "workspace.json"
    base_geometry = _FakeGeometry(100, 200, 900, 600)
    main_window = _make_workspace_main_window(
        tmp_path,
        is_theatre_mode=True,
        is_full_screen=True,
        is_maximized=False,
        geometry=_FakeGeometry(0, 0, 1920, 1080),
        normal_geometry=base_geometry,
        saved_window_state="pre-theatre-layout",
        was_custom_fullscreen=True,
        was_maximized=False,
    )

    save_current_workspace(main_window, str(save_path))

    saved = _read_saved_workspace(save_path)["window_state_data"]
    assert saved["isFullScreen"] is True
    assert saved["isMaximized"] is False
    assert saved["dock_state"] == "pre-theatre-layout"
    assert (saved["x"], saved["y"], saved["width"], saved["height"]) == (
        100,
        200,
        900,
        600,
    )


def test_save_workspace_theatre_from_maximized_uses_live_window_state(tmp_path):
    save_path = tmp_path / "workspace.json"
    base_geometry = _FakeGeometry(111, 222, 1000, 700)
    main_window = _make_workspace_main_window(
        tmp_path,
        is_theatre_mode=True,
        is_full_screen=False,
        is_maximized=True,
        geometry=_FakeGeometry(0, 0, 1920, 1080),
        normal_geometry=base_geometry,
        saved_window_state="pre-theatre-layout",
        was_custom_fullscreen=False,
        was_maximized=True,
    )

    save_current_workspace(main_window, str(save_path))

    saved = _read_saved_workspace(save_path)["window_state_data"]
    assert saved["isFullScreen"] is False
    assert saved["isMaximized"] is True
    assert saved["dock_state"] == "pre-theatre-layout"
    assert (saved["x"], saved["y"], saved["width"], saved["height"]) == (
        111,
        222,
        1000,
        700,
    )


def test_save_workspace_theatre_from_normal_uses_live_window_state(tmp_path):
    save_path = tmp_path / "workspace.json"
    base_geometry = _FakeGeometry(123, 234, 1010, 710)
    main_window = _make_workspace_main_window(
        tmp_path,
        is_theatre_mode=True,
        is_full_screen=False,
        is_maximized=False,
        geometry=_FakeGeometry(0, 0, 1920, 1080),
        normal_geometry=base_geometry,
        saved_window_state="pre-theatre-layout",
        was_custom_fullscreen=False,
        was_maximized=False,
    )

    save_current_workspace(main_window, str(save_path))

    saved = _read_saved_workspace(save_path)["window_state_data"]
    assert saved["isFullScreen"] is False
    assert saved["isMaximized"] is False
    assert saved["dock_state"] == "pre-theatre-layout"
    assert (saved["x"], saved["y"], saved["width"], saved["height"]) == (
        123,
        234,
        1010,
        710,
    )


def test_save_workspace_theatre_uses_latest_fullscreen_base_toggle(tmp_path):
    save_path = tmp_path / "workspace.json"
    main_window = _make_workspace_main_window(
        tmp_path,
        is_theatre_mode=True,
        is_full_screen=True,
        is_maximized=False,
        saved_window_state="pre-theatre-layout",
        was_custom_fullscreen=False,
        was_maximized=True,
    )

    save_current_workspace(main_window, str(save_path))
    saved = _read_saved_workspace(save_path)["window_state_data"]
    assert saved["isFullScreen"] is True
    assert saved["isMaximized"] is False

    main_window.is_full_screen = False
    main_window.isMaximized = lambda: True
    main_window._was_custom_fullscreen = True
    save_current_workspace(main_window, str(save_path))
    saved = _read_saved_workspace(save_path)["window_state_data"]
    assert saved["isFullScreen"] is False
    assert saved["isMaximized"] is True


def test_save_workspace_theatre_does_not_serialize_theatre_active_flag(tmp_path):
    save_path = tmp_path / "workspace.json"
    main_window = _make_workspace_main_window(
        tmp_path,
        is_theatre_mode=True,
        is_full_screen=False,
        is_maximized=False,
        saved_window_state="pre-theatre-layout",
    )

    save_current_workspace(main_window, str(save_path))

    saved = _read_saved_workspace(save_path)
    assert "is_theatre_mode" not in saved
    assert "is_theatre_mode" not in saved["window_state_data"]
