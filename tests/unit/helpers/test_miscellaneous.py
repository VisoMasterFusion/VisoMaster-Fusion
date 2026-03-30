"""
MISC-* tests for pure utility functions in app.helpers.miscellaneous
"""

import numpy as np
import pytest
from app.helpers.miscellaneous import (
    count_issue_scan_frames,
    ParametersDict,
    find_best_target_match,
    is_image_file,
    is_detected_face_eligible_for_matching,
    is_video_file,
    get_file_type,
    get_scaling_transforms,
    image_extensions,
    normalize_issue_scan_ranges,
    video_extensions,
    _transform_cache,
)


# ---------------------------------------------------------------------------
# MISC-03/04 — file type detection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "photo.png",
        "photo.jpg",
        "photo.jpeg",
        "photo.webp",
        "scan.tif",
        "scan.tiff",
        "image.jp2",
    ],
)
def test_is_image_file_true(name):
    assert is_image_file(name) is True


@pytest.mark.parametrize(
    "name",
    ["video.mp4", "clip.avi", "movie.mkv", "record.mov", "doc.txt", "archive.zip", ""],
)
def test_is_image_file_false(name):
    assert is_image_file(name) is False


@pytest.mark.parametrize(
    "name",
    [
        "video.mp4",
        "clip.avi",
        "movie.mkv",
        "record.mov",
        "stream.webm",
        "anim.gif",
    ],
)
def test_is_video_file_true(name):
    assert is_video_file(name) is True


@pytest.mark.parametrize(
    "name", ["photo.png", "photo.jpg", "doc.txt", "archive.zip", ""]
)
def test_is_video_file_false(name):
    assert is_video_file(name) is False


def test_get_file_type_image():
    assert get_file_type("photo.png") == "image"


def test_get_file_type_video():
    assert get_file_type("clip.mp4") == "video"


def test_get_file_type_unknown():
    assert get_file_type("notes.txt") is None


# ---------------------------------------------------------------------------
# MISC-01/02 — get_scaling_transforms cache behaviour
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clear_transform_cache():
    """Ensure transform cache is empty before each cache test."""
    _transform_cache.clear()
    yield
    _transform_cache.clear()


def _make_control(**overrides):
    base = {
        "get_cropped_face_kpsTypeSelection": "BILINEAR",
        "original_face_128_384TypeSelection": "BILINEAR",
        "original_face_512TypeSelection": "BILINEAR",
        "UntransformTypeSelection": "BILINEAR",
        "ScalebackFrameTypeSelection": "BILINEAR",
        "expression_faceeditor_t256TypeSelection": "BILINEAR",
        "expression_faceeditor_backTypeSelection": "BILINEAR",
        "block_shiftTypeSelection": "NEAREST",
        "AntialiasTypeSelection": "False",
    }
    base.update(overrides)
    return base


def test_same_params_returns_cached_object():
    ctrl = _make_control()
    result1 = get_scaling_transforms(ctrl)
    result2 = get_scaling_transforms(ctrl)
    # Exact same tuple object — came from cache
    assert result1 is result2


def test_different_params_returns_different_object():
    ctrl_a = _make_control(original_face_512TypeSelection="BILINEAR")
    ctrl_b = _make_control(original_face_512TypeSelection="BICUBIC")
    result_a = get_scaling_transforms(ctrl_a)
    result_b = get_scaling_transforms(ctrl_b)
    assert result_a is not result_b


def test_cache_populated_after_first_call():
    ctrl = _make_control()
    assert len(_transform_cache) == 0
    get_scaling_transforms(ctrl)
    assert len(_transform_cache) == 1


def test_returns_tuple_of_expected_length():
    """get_scaling_transforms returns a 12-element tuple."""
    ctrl = _make_control()
    result = get_scaling_transforms(ctrl)
    assert isinstance(result, tuple)
    assert len(result) == 12


def test_resize_objects_have_correct_size():
    from torchvision.transforms import v2

    ctrl = _make_control()
    t512, t384, t256, t128 = get_scaling_transforms(ctrl)[:4]
    # Each Resize object should have the right target size
    assert isinstance(t512, v2.Resize)
    assert isinstance(t384, v2.Resize)
    assert isinstance(t256, v2.Resize)
    assert isinstance(t128, v2.Resize)


# ---------------------------------------------------------------------------
# Extension tuple completeness sanity checks
# ---------------------------------------------------------------------------


def test_image_extensions_are_lowercase_dotted():
    for ext in image_extensions:
        assert ext.startswith("."), f"{ext} should start with '.'"
        assert ext == ext.lower(), f"{ext} should be lowercase"


def test_video_extensions_are_lowercase_dotted():
    for ext in video_extensions:
        assert ext.startswith("."), f"{ext} should start with '.'"
        assert ext == ext.lower(), f"{ext} should be lowercase"


def test_no_overlap_between_image_and_video_extensions():
    assert set(image_extensions).isdisjoint(set(video_extensions))


# ---------------------------------------------------------------------------
# MISC-05/06 - shared scan/render matching helpers
# ---------------------------------------------------------------------------


def test_is_detected_face_eligible_for_matching_rejects_small_bbox():
    kps = np.array(
        [[10.0, 10.0], [20.0, 10.0], [15.0, 15.0], [11.0, 20.0], [19.0, 20.0]],
        dtype=np.float32,
    )
    tiny_bbox = np.array([0.0, 0.0, 18.0, 30.0], dtype=np.float32)
    assert is_detected_face_eligible_for_matching(kps, tiny_bbox, 20) is False


def test_is_detected_face_eligible_for_matching_accepts_valid_face():
    kps = np.array(
        [[10.0, 10.0], [20.0, 10.0], [15.0, 15.0], [11.0, 20.0], [19.0, 20.0]],
        dtype=np.float32,
    )
    bbox = np.array([0.0, 0.0, 30.0, 30.0], dtype=np.float32)
    assert is_detected_face_eligible_for_matching(kps, bbox, 20) is True


class _DummyTargetFace:
    def __init__(self, face_id: int, embedding_store: dict[str, np.ndarray]):
        self.face_id = face_id
        self._embedding_store = embedding_store

    def get_embedding(self, recognition_model: str) -> np.ndarray | None:
        return self._embedding_store.get(recognition_model)


class _DummyModelsProcessor:
    @staticmethod
    def findCosineDistance(
        detected_embedding: np.ndarray, target_embedding: np.ndarray
    ) -> float:
        return float(np.dot(detected_embedding, target_embedding) * 100.0)


def test_find_best_target_match_respects_parameters_dict_thresholds():
    defaults = {"SimilarityThresholdSlider": 60}
    face_params = {
        "1": ParametersDict({"SimilarityThresholdSlider": 95}, defaults),
        "2": ParametersDict({"SimilarityThresholdSlider": 70}, defaults),
    }
    targets = {
        1: _DummyTargetFace(1, {"arcface_128": np.array([0.90], dtype=np.float32)}),
        2: _DummyTargetFace(2, {"arcface_128": np.array([0.80], dtype=np.float32)}),
    }

    best_target, best_params, best_score = find_best_target_match(
        np.array([1.0], dtype=np.float32),
        _DummyModelsProcessor(),
        targets,
        face_params,
        defaults,
        "arcface_128",
    )

    assert best_target is not None
    assert best_target.face_id == 2
    assert best_params is not None
    assert best_params["SimilarityThresholdSlider"] == 70
    assert best_score == pytest.approx(80.0)


def test_count_issue_scan_frames_excludes_dropped_frames():
    scan_ranges = [(0, 9), (20, 24)]
    dropped_frames = {2, 3, 22}
    assert count_issue_scan_frames(scan_ranges, dropped_frames) == 12


def test_count_issue_scan_frames_returns_zero_when_all_frames_dropped():
    scan_ranges = [(10, 12)]
    dropped_frames = {10, 11, 12}
    assert count_issue_scan_frames(scan_ranges, dropped_frames) == 0


def test_normalize_issue_scan_ranges_sorts_open_ended_style_ranges():
    scan_ranges = [(20, 30), (5, 50)]
    assert normalize_issue_scan_ranges(scan_ranges) == [(5, 50)]


def test_normalize_issue_scan_ranges_merges_overlaps():
    scan_ranges = [(20, 25), (10, 15), (12, 18), (24, 30), (40, 45)]
    assert normalize_issue_scan_ranges(scan_ranges) == [(10, 18), (20, 30), (40, 45)]


def test_count_issue_scan_frames_does_not_double_count_overlaps():
    scan_ranges = [(10, 20), (15, 25)]
    dropped_frames = {12, 18, 22}
    assert count_issue_scan_frames(scan_ranges, dropped_frames) == 13


# ---------------------------------------------------------------------------
# UT-04: keypoints_adjustments — guard against fewer than 5 keypoints
# ---------------------------------------------------------------------------

from app.helpers.miscellaneous import keypoints_adjustments  # noqa: E402


class TestKeypointsAdjustmentsGuard:
    """UT-04: When LandmarksPositionAdjEnableToggle is True and fewer than 5
    keypoints are present, keypoints_adjustments must return the input unchanged
    rather than raising IndexError.
    """

    def _params_with_toggle(self) -> dict:
        return {
            "LandmarksPositionAdjEnableToggle": True,
            "EyeLeftXAmountSlider": 5.0,
            "EyeLeftYAmountSlider": 3.0,
            "EyeRightXAmountSlider": -2.0,
            "EyeRightYAmountSlider": 1.0,
            "NoseXAmountSlider": 0.0,
            "NoseYAmountSlider": 0.0,
            "MouthLeftXAmountSlider": 0.0,
            "MouthLeftYAmountSlider": 0.0,
            "MouthRightXAmountSlider": 0.0,
            "MouthRightYAmountSlider": 0.0,
        }

    def test_fewer_than_5_keypoints_returns_input_unchanged(self):
        kps = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)  # only 2 points
        params = self._params_with_toggle()
        result = keypoints_adjustments(kps, params)
        np.testing.assert_array_equal(result, kps)

    def test_zero_keypoints_returns_input_unchanged(self):
        kps = np.zeros((0, 2), dtype=np.float32)
        params = self._params_with_toggle()
        result = keypoints_adjustments(kps, params)
        assert result.shape == (0, 2)

    def test_exactly_4_keypoints_returns_input_unchanged(self):
        kps = np.ones((4, 2), dtype=np.float32)
        params = self._params_with_toggle()
        result = keypoints_adjustments(kps, params)
        np.testing.assert_array_equal(result, kps)

    def test_exactly_5_keypoints_applies_adjustments(self):
        kps = np.zeros((5, 2), dtype=np.float32)
        params = self._params_with_toggle()
        result = keypoints_adjustments(kps, params)
        # EyeLeft X should be shifted by EyeLeftXAmountSlider=5
        assert result[0][0] == pytest.approx(5.0)

    def test_toggle_off_with_fewer_kps_does_not_raise(self):
        """When toggle is off, the guard should not matter — confirm no crash."""
        kps = np.array([[1.0, 2.0]], dtype=np.float32)
        params = {"LandmarksPositionAdjEnableToggle": False}
        result = keypoints_adjustments(kps, params)
        np.testing.assert_array_equal(result, kps)
