"""
Tests for the pure serialization/conversion logic in save_load_actions.py.

Targets:
  - convert_parameters_to_supported_type()  — ParametersDict ↔ dict
  - convert_markers_to_supported_type()     — nested marker type conversion
  - Embedding numpy↔list round-trip         — simulated as used in save/load

All PySide6, widget, and UI imports are stubbed so this runs without Qt.
"""

from __future__ import annotations

import importlib
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


def _load_save_load_actions_module():
    original_modules = {stub_name: sys.modules.get(stub_name) for stub_name in _STUBS}
    original_save_load_actions = sys.modules.pop(
        "app.ui.widgets.actions.save_load_actions", None
    )
    try:
        for stub_name in _STUBS:
            sys.modules[stub_name] = _stub(stub_name)

        widget_components_stub = sys.modules["app.ui.widgets.widget_components"]
        setattr(
            widget_components_stub,
            "TargetMediaCardButton",
            type("TargetMediaCardButton", (), {}),
        )

        module = importlib.import_module("app.ui.widgets.actions.save_load_actions")
    finally:
        for stub_name, original_module in original_modules.items():
            if original_module is None:
                sys.modules.pop(stub_name, None)
            else:
                sys.modules[stub_name] = original_module

        if original_save_load_actions is not None:
            sys.modules["app.ui.widgets.actions.save_load_actions"] = (
                original_save_load_actions
            )
        else:
            sys.modules.pop("app.ui.widgets.actions.save_load_actions", None)

    # These tests validate serialization/window-state behavior, not Qt UI display.
    module.common_widget_actions.create_and_show_toast_message = MagicMock()
    module.common_widget_actions.create_and_show_messagebox = MagicMock()
    return module


# Provide the real ParametersDict through misc_helpers
from app.helpers.miscellaneous import ParametersDict  # noqa: E402

# Now import the module under test
save_load_actions = _load_save_load_actions_module()
_apply_workspace_window_state = save_load_actions._apply_workspace_window_state
convert_parameters_to_supported_type = (
    save_load_actions.convert_parameters_to_supported_type
)
convert_markers_to_supported_type = save_load_actions.convert_markers_to_supported_type
save_current_workspace = save_load_actions.save_current_workspace


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


def test_load_save_actions_module_restores_stubbed_sys_modules():
    sentinel = object()
    original_common_actions = sys.modules.get("app.ui.widgets.actions.common_actions")
    sys.modules["app.ui.widgets.actions.common_actions"] = sentinel
    try:
        _load_save_load_actions_module()
        assert sys.modules["app.ui.widgets.actions.common_actions"] is sentinel
    finally:
        if original_common_actions is None:
            sys.modules.pop("app.ui.widgets.actions.common_actions", None)
        else:
            sys.modules["app.ui.widgets.actions.common_actions"] = (
                original_common_actions
            )


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


def _make_embedding_main_window(tmp_path: Path):
    mw = SimpleNamespace()
    mw.merged_embeddings = {}
    mw.loaded_embedding_filename = ""
    mw.project_root_path = tmp_path
    return mw


def _make_embedding_button(*, name="Embedding 1", kv_map=None):
    return SimpleNamespace(
        embedding_name=name,
        embedding_store={"arcface": np.array([1.0, 2.0], dtype=np.float32)},
        kv_map=kv_map,
    )


def test_save_embeddings_to_file_cancelled_confirmation_skips_writes(
    tmp_path, monkeypatch
):
    target_file = tmp_path / "embeddings.json"
    target_file.write_text("original")
    main_window = _make_embedding_main_window(tmp_path)
    main_window.loaded_embedding_filename = str(target_file)
    main_window.merged_embeddings = {
        "embed_1": _make_embedding_button(kv_map={"cache": "value"})
    }

    save_load_actions.common_widget_actions.create_and_show_toast_message.reset_mock()
    save_load_actions.common_widget_actions.create_and_show_messagebox.reset_mock()

    question_mock = MagicMock(return_value=0)
    monkeypatch.setattr(
        save_load_actions.QtWidgets,
        "QMessageBox",
        SimpleNamespace(Yes=1, No=0, question=question_mock),
    )
    get_save_file_name = MagicMock(return_value=(str(tmp_path / "unused.json"), ""))
    monkeypatch.setattr(
        save_load_actions.QtWidgets,
        "QFileDialog",
        SimpleNamespace(getSaveFileName=get_save_file_name),
    )
    torch_save = MagicMock()
    monkeypatch.setattr(save_load_actions.torch, "save", torch_save)

    save_load_actions.save_embeddings_to_file(main_window)

    question_mock.assert_called_once()
    get_save_file_name.assert_not_called()
    torch_save.assert_not_called()
    assert target_file.read_text() == "original"
    assert main_window.loaded_embedding_filename == str(target_file)
    save_load_actions.common_widget_actions.create_and_show_toast_message.assert_not_called()
    save_load_actions.common_widget_actions.create_and_show_messagebox.assert_not_called()


def test_save_embeddings_to_file_confirmed_confirmation_writes_file(
    tmp_path, monkeypatch
):
    target_file = tmp_path / "embeddings.json"
    target_file.write_text("original")
    main_window = _make_embedding_main_window(tmp_path)
    main_window.loaded_embedding_filename = str(target_file)
    main_window.merged_embeddings = {"embed_1": _make_embedding_button()}

    save_load_actions.common_widget_actions.create_and_show_toast_message.reset_mock()
    save_load_actions.common_widget_actions.create_and_show_messagebox.reset_mock()

    question_mock = MagicMock(return_value=1)
    monkeypatch.setattr(
        save_load_actions.QtWidgets,
        "QMessageBox",
        SimpleNamespace(Yes=1, No=0, question=question_mock),
    )
    get_save_file_name = MagicMock(return_value=(str(tmp_path / "unused.json"), ""))
    monkeypatch.setattr(
        save_load_actions.QtWidgets,
        "QFileDialog",
        SimpleNamespace(getSaveFileName=get_save_file_name),
    )
    torch_save = MagicMock()
    monkeypatch.setattr(save_load_actions.torch, "save", torch_save)

    save_load_actions.save_embeddings_to_file(main_window)

    question_mock.assert_called_once()
    get_save_file_name.assert_not_called()
    torch_save.assert_not_called()
    written = json.loads(target_file.read_text())
    assert written == [
        {
            "name": "Embedding 1",
            "embedding_store": {"arcface": [1.0, 2.0]},
            "kv_map": None,
        }
    ]
    assert main_window.loaded_embedding_filename == str(target_file)
    save_load_actions.common_widget_actions.create_and_show_toast_message.assert_called_once()
    save_load_actions.common_widget_actions.create_and_show_messagebox.assert_not_called()


def test_save_embeddings_to_file_save_as_skips_confirmation(tmp_path, monkeypatch):
    existing_file = tmp_path / "embeddings.json"
    existing_file.write_text("original")
    save_as_file = tmp_path / "save_as.json"
    main_window = _make_embedding_main_window(tmp_path)
    main_window.loaded_embedding_filename = str(existing_file)
    main_window.merged_embeddings = {"embed_1": _make_embedding_button()}

    save_load_actions.common_widget_actions.create_and_show_toast_message.reset_mock()
    save_load_actions.common_widget_actions.create_and_show_messagebox.reset_mock()

    question_mock = MagicMock(return_value=1)
    monkeypatch.setattr(
        save_load_actions.QtWidgets,
        "QMessageBox",
        SimpleNamespace(Yes=1, No=0, question=question_mock),
    )
    get_save_file_name = MagicMock(return_value=(str(save_as_file), ""))
    monkeypatch.setattr(
        save_load_actions.QtWidgets,
        "QFileDialog",
        SimpleNamespace(getSaveFileName=get_save_file_name),
    )
    torch_save = MagicMock()
    monkeypatch.setattr(save_load_actions.torch, "save", torch_save)

    save_load_actions.save_embeddings_to_file(main_window, save_as=True)

    question_mock.assert_not_called()
    get_save_file_name.assert_called_once()
    assert json.loads(save_as_file.read_text()) == [
        {
            "name": "Embedding 1",
            "embedding_store": {"arcface": [1.0, 2.0]},
            "kv_map": None,
        }
    ]
    assert main_window.loaded_embedding_filename == str(save_as_file)
    save_load_actions.common_widget_actions.create_and_show_toast_message.assert_called_once()
    save_load_actions.common_widget_actions.create_and_show_messagebox.assert_not_called()


def test_save_embeddings_to_file_with_no_embeddings_shows_existing_messagebox(
    tmp_path, monkeypatch
):
    main_window = _make_embedding_main_window(tmp_path)

    save_load_actions.common_widget_actions.create_and_show_toast_message.reset_mock()
    save_load_actions.common_widget_actions.create_and_show_messagebox.reset_mock()

    question_mock = MagicMock(return_value=1)
    monkeypatch.setattr(
        save_load_actions.QtWidgets,
        "QMessageBox",
        SimpleNamespace(Yes=1, No=0, question=question_mock),
    )
    get_save_file_name = MagicMock(return_value=(str(tmp_path / "unused.json"), ""))
    monkeypatch.setattr(
        save_load_actions.QtWidgets,
        "QFileDialog",
        SimpleNamespace(getSaveFileName=get_save_file_name),
    )

    save_load_actions.save_embeddings_to_file(main_window)

    save_load_actions.common_widget_actions.create_and_show_messagebox.assert_called_once()
    save_load_actions.common_widget_actions.create_and_show_toast_message.assert_not_called()
    question_mock.assert_not_called()
    get_save_file_name.assert_not_called()


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


class _SignalAwareCheckBox:
    def __init__(self, checked: bool = False, callbacks=None):
        self._checked = checked
        self._callbacks = callbacks or []
        self._signals_blocked = False

    def blockSignals(self, blocked: bool):
        previous_state = self._signals_blocked
        self._signals_blocked = blocked
        return previous_state

    def setChecked(self, checked: bool):
        changed = self._checked != checked
        self._checked = checked
        if changed and not self._signals_blocked:
            for callback in self._callbacks:
                callback()

    def isChecked(self):
        return self._checked


class _NoOpSignal:
    def connect(self, *_args, **_kwargs):
        return None


class _NoOpLoaderWorker:
    def __init__(self, *args, **kwargs):
        self.thumbnail_ready = _NoOpSignal()
        self.finished = _NoOpSignal()

    def run(self):
        return None


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
    fullscreen_restore_geometry: _FakeGeometry | None = None,
    theatre_forced_fullscreen: bool = False,
):
    default_params_data = {"brightness": 1.0, "contrast": 0.8}
    geometry = geometry or _FakeGeometry(10, 20, 1280, 720)
    normal_geometry = normal_geometry or _FakeGeometry(100, 200, 900, 600)

    mw = SimpleNamespace()
    mw.default_parameters = ParametersDict(default_params_data, default_params_data)
    mw.is_theatre_mode = is_theatre_mode
    mw.is_full_screen = is_full_screen
    mw._was_custom_fullscreen = was_custom_fullscreen
    mw._theatre_forced_fullscreen = theatre_forced_fullscreen
    mw._was_maximized = was_maximized
    mw._was_normal_geometry = normal_geometry
    mw._fullscreen_restore_was_maximized = False
    mw._fullscreen_restore_geometry = fullscreen_restore_geometry
    mw._saved_window_state = _FakeByteArray(saved_window_state)
    mw.control = {
        "TheatreModeUsesFullscreenToggle": False,
        "ConfirmBeforeStoppingRecordingToggle": True,
    }
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
    mw.targetVideosFilterImagesCheckBox = SimpleNamespace(isChecked=lambda: True)
    mw.targetVideosFilterVideosCheckBox = SimpleNamespace(isChecked=lambda: True)
    mw.targetVideosFilterWebcamsCheckBox = SimpleNamespace(isChecked=lambda: False)
    mw.targetVideosFilterMenuButton = SimpleNamespace(isChecked=lambda: True)
    mw.swapfacesButton = SimpleNamespace(isChecked=lambda: True)
    mw.editFacesButton = SimpleNamespace(isChecked=lambda: False)
    mw.selected_target_face_id = None
    mw.scan_tools_expanded = False
    mw.project_root_path = tmp_path
    mw.geometry = lambda: geometry
    mw.normalGeometry = lambda: normal_geometry
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


def test_save_workspace_non_theatre_fullscreen_uses_restore_geometry(tmp_path):
    save_path = tmp_path / "workspace.json"
    geometry = _FakeGeometry(0, 0, 1920, 1080)
    restore_geometry = _FakeGeometry(50, 60, 800, 500)
    main_window = _make_workspace_main_window(
        tmp_path,
        is_theatre_mode=False,
        is_full_screen=True,
        is_maximized=False,
        geometry=geometry,
        normal_geometry=_FakeGeometry(5, 6, 700, 500),
        fullscreen_restore_geometry=restore_geometry,
    )

    save_current_workspace(main_window, str(save_path))

    saved = _read_saved_workspace(save_path)["window_state_data"]
    assert saved["isFullScreen"] is True
    assert saved["isMaximized"] is False
    assert (saved["x"], saved["y"], saved["width"], saved["height"]) == (
        50,
        60,
        800,
        500,
    )


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


def test_save_workspace_theatre_forced_fullscreen_uses_pre_theatre_window_state(
    tmp_path,
):
    save_path = tmp_path / "workspace.json"
    base_geometry = _FakeGeometry(444, 555, 1200, 800)
    main_window = _make_workspace_main_window(
        tmp_path,
        is_theatre_mode=True,
        is_full_screen=True,
        is_maximized=False,
        geometry=_FakeGeometry(0, 0, 1920, 1080),
        normal_geometry=base_geometry,
        saved_window_state="pre-theatre-layout",
        was_custom_fullscreen=False,
        was_maximized=False,
        theatre_forced_fullscreen=True,
    )

    save_current_workspace(main_window, str(save_path))

    saved = _read_saved_workspace(save_path)
    assert saved["control"]["TheatreModeUsesFullscreenToggle"] is False
    saved_window = saved["window_state_data"]
    assert saved_window["isFullScreen"] is False
    assert saved_window["isMaximized"] is False
    assert saved_window["dock_state"] == "pre-theatre-layout"
    assert (
        saved_window["x"],
        saved_window["y"],
        saved_window["width"],
        saved_window["height"],
    ) == (444, 555, 1200, 800)


def test_save_workspace_theatre_uses_latest_fullscreen_base_toggle(tmp_path):
    save_path = tmp_path / "workspace.json"
    main_window = _make_workspace_main_window(
        tmp_path,
        is_theatre_mode=True,
        is_full_screen=True,
        is_maximized=False,
        normal_geometry=_FakeGeometry(150, 250, 950, 650),
        saved_window_state="pre-theatre-layout",
        was_custom_fullscreen=True,
        was_maximized=False,
    )

    save_current_workspace(main_window, str(save_path))
    saved = _read_saved_workspace(save_path)["window_state_data"]
    assert saved["isFullScreen"] is True
    assert saved["isMaximized"] is False
    assert (saved["x"], saved["y"], saved["width"], saved["height"]) == (
        150,
        250,
        950,
        650,
    )

    main_window.is_full_screen = False
    main_window.isMaximized = lambda: False
    main_window._was_custom_fullscreen = False
    main_window._was_maximized = False
    main_window._was_normal_geometry = _FakeGeometry(220, 330, 1110, 720)
    save_current_workspace(main_window, str(save_path))
    saved = _read_saved_workspace(save_path)["window_state_data"]
    assert saved["isFullScreen"] is False
    assert saved["isMaximized"] is False
    assert (saved["x"], saved["y"], saved["width"], saved["height"]) == (
        220,
        330,
        1110,
        720,
    )


def test_save_workspace_persists_theatre_fullscreen_setting_in_control(tmp_path):
    save_path = tmp_path / "workspace.json"
    main_window = _make_workspace_main_window(
        tmp_path,
        is_theatre_mode=False,
        is_full_screen=False,
        is_maximized=False,
    )
    main_window.control["TheatreModeUsesFullscreenToggle"] = True

    save_current_workspace(main_window, str(save_path))

    saved = _read_saved_workspace(save_path)
    assert saved["control"]["TheatreModeUsesFullscreenToggle"] is True


def test_save_workspace_persists_stop_recording_confirmation_setting(tmp_path):
    save_path = tmp_path / "workspace.json"
    main_window = _make_workspace_main_window(
        tmp_path,
        is_theatre_mode=False,
        is_full_screen=False,
        is_maximized=False,
    )
    main_window.control["ConfirmBeforeStoppingRecordingToggle"] = False

    save_current_workspace(main_window, str(save_path))

    saved = _read_saved_workspace(save_path)
    assert saved["control"]["ConfirmBeforeStoppingRecordingToggle"] is False


def test_apply_workspace_window_state_fullscreen_seeds_restore_geometry(monkeypatch):
    restored_rect = _FakeGeometry(140, 150, 910, 610)
    main_window = SimpleNamespace(
        _fullscreen_restore_was_maximized=True,
        _fullscreen_restore_geometry="stale-geometry",
        resize=MagicMock(),
        sizeHint=lambda: "size-hint",
        showFullScreen=MagicMock(),
        showMaximized=MagicMock(),
        setGeometry=MagicMock(),
        menuBar=lambda: SimpleNamespace(show=MagicMock()),
        is_full_screen=False,
        x=lambda: 0,
        y=lambda: 0,
        width=lambda: 1920,
        height=lambda: 1080,
    )

    monkeypatch.setattr(
        save_load_actions,
        "_get_clamped_window_geometry",
        lambda *_args, **_kwargs: restored_rect,
    )

    needs_clamp = _apply_workspace_window_state(
        main_window,
        {
            "isMaximized": False,
            "isFullScreen": True,
            "x": 1,
            "y": 2,
            "width": 3,
            "height": 4,
        },
    )

    assert needs_clamp is False
    main_window.resize.assert_called_once_with("size-hint")
    main_window.showFullScreen.assert_called_once()
    main_window.showMaximized.assert_not_called()
    main_window.setGeometry.assert_not_called()
    assert main_window.is_full_screen is True
    assert main_window._fullscreen_restore_was_maximized is False
    assert main_window._fullscreen_restore_geometry is restored_rect


def test_apply_workspace_window_state_normal_clears_fullscreen_restore_state(
    monkeypatch,
):
    restored_rect = _FakeGeometry(240, 250, 920, 620)
    menu_bar = SimpleNamespace(show=MagicMock())
    main_window = SimpleNamespace(
        _fullscreen_restore_was_maximized=True,
        _fullscreen_restore_geometry="stale-geometry",
        resize=MagicMock(),
        sizeHint=lambda: "size-hint",
        showFullScreen=MagicMock(),
        showMaximized=MagicMock(),
        setGeometry=MagicMock(),
        menuBar=lambda: menu_bar,
        is_full_screen=True,
        x=lambda: 0,
        y=lambda: 0,
        width=lambda: 1920,
        height=lambda: 1080,
    )

    monkeypatch.setattr(
        save_load_actions,
        "_get_clamped_window_geometry",
        lambda *_args, **_kwargs: restored_rect,
    )

    needs_clamp = _apply_workspace_window_state(
        main_window,
        {
            "isMaximized": False,
            "isFullScreen": False,
            "x": 1,
            "y": 2,
            "width": 3,
            "height": 4,
        },
    )

    assert needs_clamp is True
    main_window.setGeometry.assert_called_once_with(restored_rect)
    menu_bar.show.assert_called_once()
    main_window.showFullScreen.assert_not_called()
    assert main_window.is_full_screen is False
    assert main_window._fullscreen_restore_was_maximized is False
    assert main_window._fullscreen_restore_geometry is None


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


def test_load_workspace_restores_target_media_filters_without_double_loading_webcams(
    tmp_path, monkeypatch
):
    workspace_path = tmp_path / "workspace.json"
    workspace_path.write_text(
        json.dumps(
            {
                "control": {},
                "target_medias_data": [],
                "input_faces_data": {},
                "embeddings_data": {},
                "target_faces_data": {},
                "markers": {},
                "window_state_data": {
                    "filterImagesCheckBox": True,
                    "filterVideosCheckBox": True,
                    "filterWebcamsCheckBox": True,
                },
            }
        )
    )

    filter_calls = []
    webcam_calls = []

    main_window = SimpleNamespace(
        control={},
        parameters={},
        default_parameters=ParametersDict({}, {}),
        current_widget_parameters=ParametersDict({}, {}),
        target_videos={},
        input_faces={},
        target_faces={},
        merged_embeddings={},
        markers={},
        issue_frames_by_face={},
        dropped_frames=set(),
        job_marker_pairs=[],
        selected_target_face_id=None,
        videoSeekSlider=SimpleNamespace(update=MagicMock()),
        placeholder_update_signal=SimpleNamespace(emit=MagicMock()),
        swapfacesButton=MagicMock(),
        editFacesButton=MagicMock(),
        targetVideosList=MagicMock(),
        inputFacesList=MagicMock(),
        inputEmbeddingsList=MagicMock(),
        targetVideosSortComboBox=SimpleNamespace(currentData=lambda: None),
        targetVideosPathLineEdit=SimpleNamespace(
            setText=MagicMock(), setToolTip=MagicMock()
        ),
        inputFacesPathLineEdit=SimpleNamespace(
            setText=MagicMock(), setToolTip=MagicMock()
        ),
        outputFolderLineEdit=SimpleNamespace(setText=MagicMock()),
        _set_panel_visibility=MagicMock(),
        apply_parameter_section_states=MagicMock(),
        _refresh_panel_visibility_state_from_widgets=MagicMock(),
        restoreState=MagicMock(),
        tabWidget=SimpleNamespace(count=lambda: 0),
        video_loader_worker=None,
        input_faces_loader_worker=None,
    )

    main_window.targetVideosFilterImagesCheckBox = _SignalAwareCheckBox(checked=True)
    main_window.targetVideosFilterVideosCheckBox = _SignalAwareCheckBox(checked=True)
    main_window.targetVideosFilterWebcamsCheckBox = _SignalAwareCheckBox(
        checked=False,
        callbacks=[
            lambda: filter_calls.append("signal-filter"),
            lambda: webcam_calls.append("signal-webcam"),
        ],
    )
    main_window.targetVideosFilterMenuButton = _SignalAwareCheckBox(checked=False)

    # The filter/sort menu toggle drives real Qt layouts; the media-filter
    # behaviour under test is unaffected by it.
    from app.ui.widgets.actions import target_videos_list_actions

    monkeypatch.setattr(
        target_videos_list_actions,
        "toggle_target_video_filters_sorting",
        lambda *_args, **_kwargs: None,
    )

    monkeypatch.setattr(
        save_load_actions,
        "_apply_workspace_window_state",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(
        save_load_actions.video_control_actions,
        "block_if_issue_scan_active",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(
        save_load_actions.list_view_actions,
        "clear_stop_loading_input_media",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        save_load_actions.list_view_actions,
        "clear_stop_loading_target_media",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        save_load_actions.card_actions,
        "clear_input_faces",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        save_load_actions.card_actions,
        "clear_target_faces",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        save_load_actions.card_actions,
        "clear_merged_embeddings",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        save_load_actions.ui_workers,
        "TargetMediaLoaderWorker",
        _NoOpLoaderWorker,
    )
    monkeypatch.setattr(
        save_load_actions.ui_workers,
        "InputFacesLoaderWorker",
        _NoOpLoaderWorker,
    )
    monkeypatch.setattr(
        save_load_actions.filter_actions,
        "filter_target_videos",
        lambda *_args, **_kwargs: filter_calls.append("final-filter"),
    )
    monkeypatch.setattr(
        save_load_actions.list_view_actions,
        "load_target_webcams",
        lambda *_args, **_kwargs: webcam_calls.append("final-webcam"),
    )
    monkeypatch.setattr(
        save_load_actions.list_view_actions,
        "apply_face_thumbnail_size",
        lambda *_args, **_kwargs: None,
    )
    # The loader now drains the batched thumbnail queue before continuing. The
    # stubbed list_view_actions would report pending work forever, so model a
    # single outstanding batch.
    pending_batches = {"remaining": 1}
    flushed_batches = []

    def flush_target_media_batch(*_args, **_kwargs):
        pending_batches["remaining"] -= 1
        flushed_batches.append("flush")

    monkeypatch.setattr(
        save_load_actions.list_view_actions,
        "_has_pending_target_media_thumbnail_work",
        lambda *_args, **_kwargs: pending_batches["remaining"] > 0,
    )
    monkeypatch.setattr(
        save_load_actions.list_view_actions,
        "_flush_target_media_thumbnail_batch",
        flush_target_media_batch,
    )
    monkeypatch.setattr(
        save_load_actions.common_widget_actions,
        "set_control_widgets_values",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        save_load_actions.common_widget_actions,
        "create_control",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        save_load_actions.common_widget_actions,
        "set_widgets_values_using_face_id_parameters",
        lambda *_args, **_kwargs: None,
    )
    # Restoring the Edit-Faces button now reaches into the model manager, which
    # is out of scope for this media-filter test.
    monkeypatch.setattr(
        save_load_actions.control_actions,
        "handle_face_editor_button_click",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        save_load_actions.layout_actions,
        "fit_image_to_view_onchange",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        save_load_actions.video_control_actions,
        "remove_all_markers",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        save_load_actions.video_control_actions,
        "set_issue_frames_by_face",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        save_load_actions.video_control_actions,
        "set_issue_frames_for_face",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        save_load_actions.video_control_actions,
        "set_dropped_frames",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        save_load_actions.video_control_actions,
        "update_drop_frame_button_label",
        lambda *_args, **_kwargs: None,
    )

    save_load_actions.load_saved_workspace(main_window, str(workspace_path))

    assert filter_calls == ["final-filter"]
    assert webcam_calls == ["final-webcam"]
    assert flushed_batches == ["flush"]
