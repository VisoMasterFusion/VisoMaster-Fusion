"""
Tests for the pure logic in job_manager_actions.py.

Targets:
  - convert_parameters_to_job_type()   — dict/ParametersDict conversion (richer than save_load version)
  - convert_markers_to_job_type()      — deep-copy + nested conversion
  - _validate_job_files_exist()        — pre-flight file-existence check
  - _validate_job_data_for_loading()   — wrapper around the above
  - list_jobs()                        — directory scanning

All PySide6, Qt, send2trash, and UI imports are stubbed.
"""

from __future__ import annotations

import sys
import copy
from unittest.mock import MagicMock
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Stub heavy imports
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
    "PySide6.QtWidgets.QMessageBox",
    "send2trash",
    "app.ui.widgets.widget_components",
    # Stub only the leaf action modules that job_manager_actions.py actually imports.
    # Do NOT stub the parent package ("app.ui.widgets.actions") — it's a namespace
    # package and stubbing it prevents sibling test files from resolving real submodules.
    # Do NOT stub save_load_actions or filter_actions — job_manager_actions.py doesn't
    # import them, and stubbing them here corrupts later test files that DO need them.
    "app.ui.widgets.actions.common_actions",
    "app.ui.widgets.actions.card_actions",
    "app.ui.widgets.actions.list_view_actions",
    "app.ui.widgets.actions.video_control_actions",
    "app.ui.widgets.actions.layout_actions",
    "app.ui.widgets.ui_workers",
    "app.helpers.typing_helper",
]
for _s in _STUBS:
    if _s not in sys.modules:
        sys.modules[_s] = _stub(_s)

from app.helpers.miscellaneous import ParametersDict  # noqa: E402

# Import must happen AFTER stubs so the module-level `jobs_dir = ...` line
# runs with a mocked environment.  We patch it to a temp location below.


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def default_params_data() -> dict:
    return {"brightness": 1.0, "contrast": 0.8}


@pytest.fixture
def mock_main_window(default_params_data):
    mw = MagicMock()
    mw.default_parameters = ParametersDict(default_params_data, default_params_data)
    return mw


@pytest.fixture
def pd(default_params_data) -> ParametersDict:
    return ParametersDict({"brightness": 2.0, "contrast": 0.5}, default_params_data)


# ---------------------------------------------------------------------------
# convert_parameters_to_job_type — ParametersDict → dict
# ---------------------------------------------------------------------------


def test_job_convert_parameters_dict_to_dict(mock_main_window, pd):
    from app.ui.widgets.actions.job_manager_actions import (
        convert_parameters_to_job_type,
    )

    result = convert_parameters_to_job_type(mock_main_window, pd, dict)
    assert isinstance(result, dict)
    assert not isinstance(result, ParametersDict)
    assert result["brightness"] == 2.0


def test_job_convert_parameters_dict_to_dict_returns_copy(mock_main_window, pd):
    """The returned dict must be a copy — mutating it must not affect the original PD."""
    from app.ui.widgets.actions.job_manager_actions import (
        convert_parameters_to_job_type,
    )

    result = convert_parameters_to_job_type(mock_main_window, pd, dict)
    result["brightness"] = 999.0
    assert pd["brightness"] == 2.0  # original unchanged


def test_job_convert_plain_dict_to_dict_returns_copy(mock_main_window):
    from app.ui.widgets.actions.job_manager_actions import (
        convert_parameters_to_job_type,
    )

    original = {"brightness": 3.0}
    result = convert_parameters_to_job_type(mock_main_window, original, dict)
    assert result is not original  # different object
    assert result == original


# ---------------------------------------------------------------------------
# convert_parameters_to_job_type — dict → ParametersDict
# ---------------------------------------------------------------------------


def test_job_convert_dict_to_parameters_dict(mock_main_window, default_params_data):
    from app.ui.widgets.actions.job_manager_actions import (
        convert_parameters_to_job_type,
    )

    plain = {"brightness": 1.7}
    result = convert_parameters_to_job_type(mock_main_window, plain, ParametersDict)
    assert isinstance(result, ParametersDict)
    assert result["brightness"] == 1.7


def test_job_convert_parameters_dict_passthrough(mock_main_window, pd):
    from app.ui.widgets.actions.job_manager_actions import (
        convert_parameters_to_job_type,
    )

    result = convert_parameters_to_job_type(mock_main_window, pd, ParametersDict)
    assert result is pd


# ---------------------------------------------------------------------------
# convert_markers_to_job_type — deep copy + nested conversion
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_markers(default_params_data):
    return {
        50: {
            "parameters": {
                "face_1": ParametersDict({"brightness": 1.2}, default_params_data),
            },
            "control": {"VR180ModeEnableToggle": True},
        },
    }


def test_job_convert_markers_to_dict(mock_main_window, sample_markers):
    from app.ui.widgets.actions.job_manager_actions import convert_markers_to_job_type

    result = convert_markers_to_job_type(mock_main_window, sample_markers, dict)
    params = result[50]["parameters"]["face_1"]
    assert isinstance(params, dict)
    assert not isinstance(params, ParametersDict)


def test_job_convert_markers_does_not_mutate_original(mock_main_window, sample_markers):
    """Deep copy must protect the original from mutation."""
    from app.ui.widgets.actions.job_manager_actions import convert_markers_to_job_type

    original_type = type(sample_markers[50]["parameters"]["face_1"])
    convert_markers_to_job_type(mock_main_window, sample_markers, dict)
    assert type(sample_markers[50]["parameters"]["face_1"]) is original_type


def test_job_convert_markers_control_also_converted(mock_main_window, sample_markers):
    """The 'control' dict within each marker is also processed by convert_parameters_to_job_type."""
    from app.ui.widgets.actions.job_manager_actions import convert_markers_to_job_type

    result = convert_markers_to_job_type(mock_main_window, sample_markers, dict)
    # control should still be accessible after conversion
    assert result[50]["control"]["VR180ModeEnableToggle"] is True


def test_job_convert_markers_round_trip(mock_main_window, sample_markers):
    """Markers converted to dict and back to ParametersDict should restore correctly."""
    from app.ui.widgets.actions.job_manager_actions import convert_markers_to_job_type

    as_dict = convert_markers_to_job_type(
        mock_main_window, copy.deepcopy(sample_markers), dict
    )
    restored = convert_markers_to_job_type(mock_main_window, as_dict, ParametersDict)
    assert isinstance(restored[50]["parameters"]["face_1"], ParametersDict)
    assert restored[50]["parameters"]["face_1"]["brightness"] == 1.2


# ---------------------------------------------------------------------------
# _validate_job_files_exist — pre-flight check
# ---------------------------------------------------------------------------


def _make_valid_job_data(tmp_path: Path) -> dict:
    """Build a minimal valid job data dict with real files on disk."""
    media_file = tmp_path / "video.mp4"
    media_file.write_bytes(b"fake")
    face_file = tmp_path / "face.png"
    face_file.write_bytes(b"fake")

    return {
        "selected_media_id": "media_001",
        "target_medias_data": [
            {"media_id": "media_001", "media_path": str(media_file)},
        ],
        "target_faces_data": {
            "face_1": {
                "assigned_input_faces": ["input_face_1"],
                "assigned_merged_embeddings": [],
            }
        },
        "input_faces_data": {
            "input_face_1": {"media_path": str(face_file)},
        },
        "embeddings_data": {},
    }


def test_validate_valid_job_passes(tmp_path):
    from app.ui.widgets.actions.job_manager_actions import _validate_job_files_exist

    data = _make_valid_job_data(tmp_path)
    is_valid, reason = _validate_job_files_exist(data)
    assert is_valid is True
    assert reason is None


def test_validate_no_selected_media_id_fails():
    from app.ui.widgets.actions.job_manager_actions import _validate_job_files_exist

    data = {
        "selected_media_id": None,
        "target_medias_data": [],
        "target_faces_data": {},
    }
    is_valid, reason = _validate_job_files_exist(data)
    assert is_valid is False
    assert reason is not None


def test_validate_media_file_missing_fails(tmp_path):
    from app.ui.widgets.actions.job_manager_actions import _validate_job_files_exist

    data = {
        "selected_media_id": "media_001",
        "target_medias_data": [
            {"media_id": "media_001", "media_path": str(tmp_path / "nonexistent.mp4")},
        ],
        "target_faces_data": {},
        "input_faces_data": {},
        "embeddings_data": {},
    }
    is_valid, reason = _validate_job_files_exist(data)
    assert is_valid is False
    assert "not found" in reason.lower() or "nonexistent" in reason


def test_validate_selected_media_id_not_in_list_fails():
    from app.ui.widgets.actions.job_manager_actions import _validate_job_files_exist

    data = {
        "selected_media_id": "media_999",
        "target_medias_data": [
            {"media_id": "media_001", "media_path": "/some/path.mp4"}
        ],
        "target_faces_data": {},
        "input_faces_data": {},
        "embeddings_data": {},
    }
    is_valid, reason = _validate_job_files_exist(data)
    assert is_valid is False


def test_validate_required_input_face_missing_from_data(tmp_path):
    from app.ui.widgets.actions.job_manager_actions import _validate_job_files_exist

    media_file = tmp_path / "video.mp4"
    media_file.write_bytes(b"fake")
    data = {
        "selected_media_id": "m1",
        "target_medias_data": [{"media_id": "m1", "media_path": str(media_file)}],
        "target_faces_data": {
            "face_1": {
                "assigned_input_faces": ["missing_face_id"],
                "assigned_merged_embeddings": [],
            }
        },
        "input_faces_data": {},  # "missing_face_id" not present
        "embeddings_data": {},
    }
    is_valid, reason = _validate_job_files_exist(data)
    assert is_valid is False
    assert "missing_face_id" in reason


def test_validate_required_input_face_file_not_on_disk(tmp_path):
    from app.ui.widgets.actions.job_manager_actions import _validate_job_files_exist

    media_file = tmp_path / "video.mp4"
    media_file.write_bytes(b"fake")
    data = {
        "selected_media_id": "m1",
        "target_medias_data": [{"media_id": "m1", "media_path": str(media_file)}],
        "target_faces_data": {
            "face_1": {
                "assigned_input_faces": ["f1"],
                "assigned_merged_embeddings": [],
            }
        },
        "input_faces_data": {"f1": {"media_path": str(tmp_path / "gone.png")}},
        "embeddings_data": {},
    }
    is_valid, reason = _validate_job_files_exist(data)
    assert is_valid is False


def test_validate_required_embedding_missing_from_data(tmp_path):
    from app.ui.widgets.actions.job_manager_actions import _validate_job_files_exist

    media_file = tmp_path / "video.mp4"
    media_file.write_bytes(b"fake")
    data = {
        "selected_media_id": "m1",
        "target_medias_data": [{"media_id": "m1", "media_path": str(media_file)}],
        "target_faces_data": {
            "face_1": {
                "assigned_input_faces": [],
                "assigned_merged_embeddings": ["embed_999"],  # not in embeddings_data
            }
        },
        "input_faces_data": {},
        "embeddings_data": {},  # embed_999 absent
    }
    is_valid, reason = _validate_job_files_exist(data)
    assert is_valid is False
    assert "embed_999" in reason


# ---------------------------------------------------------------------------
# _validate_job_data_for_loading — wrapper (delegates to _validate_job_files_exist)
# ---------------------------------------------------------------------------


def test_validate_job_data_for_loading_is_wrapper(tmp_path):
    from app.ui.widgets.actions.job_manager_actions import (
        _validate_job_data_for_loading,
    )

    data = _make_valid_job_data(tmp_path)
    is_valid, reason = _validate_job_data_for_loading(data)
    assert is_valid is True
    assert reason is None


def test_validate_job_data_for_loading_invalid(tmp_path):
    from app.ui.widgets.actions.job_manager_actions import (
        _validate_job_data_for_loading,
    )

    data = {
        "selected_media_id": None,
        "target_medias_data": [],
        "target_faces_data": {},
    }
    is_valid, _ = _validate_job_data_for_loading(data)
    assert is_valid is False


# ---------------------------------------------------------------------------
# list_jobs()
# ---------------------------------------------------------------------------


def _mock_mw(project_root: Path) -> MagicMock:
    """Minimal main_window mock with project_root_path set."""
    mw = MagicMock()
    mw.project_root_path = project_root
    return mw


def test_list_jobs_empty_dir(tmp_path):
    from app.ui.widgets.actions.job_manager_actions import list_jobs

    result = list_jobs(_mock_mw(tmp_path))
    assert result == []


def test_list_jobs_returns_names_without_extension(tmp_path):
    from app.ui.widgets.actions.job_manager_actions import list_jobs

    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir()
    (jobs_dir / "my_job.json").write_text("{}")
    (jobs_dir / "second.json").write_text("{}")
    result = list_jobs(_mock_mw(tmp_path))
    assert "my_job" in result
    assert "second" in result
    assert len(result) == 2


def test_list_jobs_ignores_non_json_files(tmp_path):
    from app.ui.widgets.actions.job_manager_actions import list_jobs

    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir()
    (jobs_dir / "job1.json").write_text("{}")
    (jobs_dir / "readme.txt").write_text("not a job")
    (jobs_dir / "data.csv").write_text("a,b")
    result = list_jobs(_mock_mw(tmp_path))
    assert result == ["job1"]


def test_list_jobs_nonexistent_dir(tmp_path):
    from app.ui.widgets.actions.job_manager_actions import list_jobs

    # No "jobs" subdir created — get_jobs_dir will create it (empty) → returns []
    result = list_jobs(_mock_mw(tmp_path))
    assert result == []
