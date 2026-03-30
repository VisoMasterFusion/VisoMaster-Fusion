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

# Provide the real ParametersDict through misc_helpers
from app.helpers.miscellaneous import ParametersDict  # noqa: E402

# Now import the module under test
from app.ui.widgets.actions.save_load_actions import (  # noqa: E402
    convert_parameters_to_supported_type,
    convert_markers_to_supported_type,
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
