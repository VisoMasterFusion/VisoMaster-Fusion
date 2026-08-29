"""Unit tests for the offline issue scan.

The scan was extracted from VideoProcessor into
``app.processors.video_utils.issue_scanner.IssueScanner``, which owns the state
snapshots, the marker-segment resolution and the detection loop. These tests
drive the scanner directly; the VideoProcessor facade is only covered where it
still adds behaviour of its own (restoring the live playback position).
"""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from app.processors.video_processor import VideoProcessor
from app.processors.video_utils.issue_scanner import IssueScanner
from app.processors.video_utils.sequential_detector import SequentialDetector

MARKER_DATA_PATH = (
    "app.processors.video_utils.issue_scanner"
    ".video_control_actions._get_marker_data_for_position"
)
VIDEO_CAPTURE_PATH = "app.processors.video_utils.issue_scanner.cv2.VideoCapture"
READ_FRAME_PATH = "app.processors.video_utils.issue_scanner.misc_helpers.read_frame"
SEEK_FRAME_PATH = "app.processors.video_utils.issue_scanner.misc_helpers.seek_frame"
RELEASE_CAPTURE_PATH = (
    "app.processors.video_utils.issue_scanner.misc_helpers.release_capture"
)


class _DummyCapture:
    def isOpened(self):
        return True

    def set(self, *_args, **_kwargs):
        return True


class _RecordingFunctionWorker:
    """Minimal function_worker stand-in for the scanner.

    Tracker resets, recognition and cosine matching all go through the
    function_worker now, so the scan tests observe them here.
    """

    def __init__(self, embedding=None, cosine_similarity=1.0):
        self.reset_calls: list[str] = []
        self.recognize_calls: list[tuple[str, str]] = []
        self.embedding = (
            embedding if embedding is not None else np.array([3.0], dtype=np.float32)
        )
        self.cosine_similarity = cosine_similarity

    def reset_face_tracker(self):
        self.reset_calls.append("reset")

    def run_recognize_direct(self, _image, _kps, similarity_type, recognition_model):
        self.recognize_calls.append((recognition_model, similarity_type))
        return self.embedding, None

    def findCosineDistance(self, _detected, _target):
        return self.cosine_similarity


def _make_main_window(**overrides):
    defaults = dict(
        control={},
        parameters={},
        target_faces={},
        dropped_frames=set(),
        markers={},
        job_marker_pairs=[],
        videoSeekSlider=SimpleNamespace(value=lambda: 0),
        editFacesButton=SimpleNamespace(isChecked=lambda: False),
        default_parameters=SimpleNamespace(data={"SimilarityThresholdSlider": 50}),
        models_processor=SimpleNamespace(device="cpu"),
        function_worker=_RecordingFunctionWorker(),
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _make_scanner(
    *,
    media_path="dummy.mp4",
    max_frame_number=100,
    media_rotation=0,
    **main_window_overrides,
) -> IssueScanner:
    main_window = _make_main_window(**main_window_overrides)
    return IssueScanner(
        main_window=main_window,
        sequential_detector=SequentialDetector(main_window),
        media_path=media_path,
        max_frame_number=max_frame_number,
        media_rotation=media_rotation,
    )


class _DummyTargetFace:
    def __init__(self, face_id, embeddings):
        self.face_id = face_id
        self._embeddings = embeddings

    def get_embedding(self, recognition_model):
        return self._embeddings.get(recognition_model)


def _empty_scan_detection_result():
    return (
        np.empty((0, 4), dtype=np.float32),
        np.empty((0, 5, 2), dtype=np.float32),
        np.empty((0, 68, 2), dtype=np.float32),
        np.empty((0, 203, 2), dtype=np.float32),
    )


def _make_target_snapshot(face_id, embeddings_by_model=None):
    return {
        str(face_id): {
            "face_id": str(face_id),
            "embeddings_by_model": embeddings_by_model or {},
        }
    }


def _scan_patches(scanner, *, read_frame=None, seek_frame=None):
    """Patch the capture/IO helpers the scan loop reaches for."""
    read_kwargs = (
        {"side_effect": read_frame}
        if callable(read_frame)
        else {"return_value": (True, np.zeros((4, 4, 3), dtype=np.uint8))}
    )
    seek_kwargs = {"side_effect": seek_frame} if callable(seek_frame) else {}
    return (
        patch(VIDEO_CAPTURE_PATH, return_value=_DummyCapture()),
        patch(READ_FRAME_PATH, **read_kwargs),
        patch(SEEK_FRAME_PATH, **seek_kwargs),
        patch(RELEASE_CAPTURE_PATH),
    )


def test_filter_scan_control_keeps_only_allowlisted_keys():
    filtered = IssueScanner._filter_scan_control(
        {
            "DetectorScoreSlider": 42,
            "FaceTrackingEnableToggle": True,
            "SimilarityTypeSelection": "Pearl",
            "IgnoredControl": "skip",
        }
    )

    assert filtered == {
        "DetectorScoreSlider": 42,
        "FaceTrackingEnableToggle": True,
    }


def test_filter_scan_face_params_keeps_only_threshold_for_target_faces():
    filtered = IssueScanner._filter_scan_face_params(
        {
            "face_1": {
                "SimilarityThresholdSlider": 61,
                "FaceExpressionEnableBothToggle": True,
            },
            "face_2": {"SimilarityThresholdSlider": 75},
        },
        ["face_1"],
    )

    assert filtered == {
        "face_1": {
            "SimilarityThresholdSlider": 61,
        }
    }


def test_get_issue_scan_unavailable_reason_rejects_marker_enabled_vr180_within_range():
    reason = _make_scanner().get_issue_scan_unavailable_reason(
        {"VR180ModeEnableToggle": False},
        scan_ranges=[(10, 20)],
        markers={
            15: {
                "control": {
                    "VR180ModeEnableToggle": True,
                }
            }
        },
    )

    assert reason == "Issue scans are not supported while VR180 mode is enabled."


def test_get_issue_scan_unavailable_reason_allows_marker_enabled_vr180_outside_range():
    reason = _make_scanner().get_issue_scan_unavailable_reason(
        {"VR180ModeEnableToggle": False},
        scan_ranges=[(10, 20)],
        markers={
            25: {
                "control": {
                    "VR180ModeEnableToggle": True,
                }
            }
        },
    )

    assert reason is None


def test_get_issue_scan_unavailable_reason_rejects_mixed_ranges_when_one_uses_vr180():
    reason = _make_scanner().get_issue_scan_unavailable_reason(
        {"VR180ModeEnableToggle": False},
        scan_ranges=[(0, 5), (10, 20)],
        markers={
            12: {
                "control": {
                    "VR180ModeEnableToggle": True,
                }
            }
        },
    )

    assert reason == "Issue scans are not supported while VR180 mode is enabled."


def test_get_issue_scan_unavailable_reason_allows_range_when_start_marker_turns_vr180_off():
    reason = _make_scanner().get_issue_scan_unavailable_reason(
        {"VR180ModeEnableToggle": True},
        scan_ranges=[(10, 20)],
        markers={
            10: {
                "control": {
                    "VR180ModeEnableToggle": False,
                }
            },
            30: {
                "control": {
                    "VR180ModeEnableToggle": True,
                }
            },
        },
        fallback_control={"VR180ModeEnableToggle": True},
    )

    assert reason is None


def test_get_issue_scan_unavailable_reason_rejects_live_vr180_when_no_start_marker_override():
    reason = _make_scanner().get_issue_scan_unavailable_reason(
        {"VR180ModeEnableToggle": True},
        scan_ranges=[(10, 20)],
        markers={
            30: {
                "control": {
                    "VR180ModeEnableToggle": False,
                }
            }
        },
        fallback_control={"VR180ModeEnableToggle": True},
    )

    assert reason == "Issue scans are not supported while VR180 mode is enabled."


def test_scan_issue_frames_restores_the_live_sequential_detector_state():
    scanner = _make_scanner()
    detector = scanner.sequential_detector
    detector.last_detected_faces = [{"id": 1}]
    detector._smoothed_kps = {1: np.array([[1.0, 2.0]], dtype=np.float32)}
    detector._smoothed_dense_kps = {1: np.array([[3.0, 4.0]], dtype=np.float32)}
    detector._smoothed_dense_kps_203 = {1: np.array([[5.0, 6.0]], dtype=np.float32)}

    with (
        patch(VIDEO_CAPTURE_PATH, return_value=_DummyCapture()),
        patch(RELEASE_CAPTURE_PATH),
    ):
        result = scanner.scan_issue_frames(
            scan_ranges=[(1, 0)],
            base_control={},
            base_params={},
            target_faces_snapshot={},
        )

    assert result == {
        "issue_frames_by_face": {},
        "frames_scanned": 0,
        "faces_with_issues": 0,
        "cancelled": False,
    }
    np.testing.assert_array_equal(
        detector._smoothed_dense_kps[1], np.array([[3.0, 4.0]], dtype=np.float32)
    )
    np.testing.assert_array_equal(
        detector._smoothed_kps[1], np.array([[1.0, 2.0]], dtype=np.float32)
    )
    np.testing.assert_array_equal(
        detector._smoothed_dense_kps_203[1],
        np.array([[5.0, 6.0]], dtype=np.float32),
    )
    assert detector.last_detected_faces == [{"id": 1}]


def test_scan_issue_frames_rejects_when_marker_enables_vr180():
    scanner = _make_scanner(
        control={"VR180ModeEnableToggle": False},
        markers={10: {"control": {"VR180ModeEnableToggle": True}}},
    )

    with pytest.raises(RuntimeError, match="Issue scans are not supported"):
        scanner.scan_issue_frames(
            scan_ranges=[(0, 15)],
            base_control={},
            base_params={},
            target_faces_snapshot={},
        )


def test_video_processor_restores_the_playback_frame_after_a_scan():
    """The scanner walks the media with its own capture, so the facade has to
    put the live playback position back where the user left it."""
    scan_calls = []
    scanner = SimpleNamespace(
        scan_issue_frames=lambda *args: scan_calls.append(args) or {"frames_scanned": 0}
    )
    dummy = SimpleNamespace(
        current_frame_number=99,
        main_window=SimpleNamespace(videoSeekSlider=SimpleNamespace(value=lambda: 7)),
        _get_issue_scanner_instance=lambda: scanner,
    )

    result = VideoProcessor.scan_issue_frames(
        dummy, scan_ranges=[(0, 1)], reset_frame_number=3
    )

    assert result == {"frames_scanned": 0}
    assert dummy.current_frame_number == 3
    # The reset frame stays in the facade; the scanner never receives it.
    assert scan_calls[0][3] == [(0, 1)]
    assert len(scan_calls[0]) == 9


def test_video_processor_falls_back_to_the_seek_slider_after_a_scan():
    dummy = SimpleNamespace(
        current_frame_number=99,
        main_window=SimpleNamespace(videoSeekSlider=SimpleNamespace(value=lambda: 7)),
        _get_issue_scanner_instance=lambda: SimpleNamespace(
            scan_issue_frames=lambda *_args: None
        ),
    )

    assert VideoProcessor.scan_issue_frames(dummy) is None
    assert dummy.current_frame_number == 7


def test_describe_issue_scan_scope_uses_normalized_effective_ranges():
    scanner = _make_scanner(job_marker_pairs=[(20, 30), (10, 25), (40, None)])

    scope_text = scanner.describe_issue_scan_scope([(10, 30), (40, 100)])

    assert scope_text == "Scanning 1 marked range and record start frame 40 to end"


def test_describe_issue_scan_scope_uses_raw_open_start_when_ranges_merge():
    scanner = _make_scanner(job_marker_pairs=[(10, 30), (20, None)])

    scope_text = scanner.describe_issue_scan_scope([(10, 100)])

    assert scope_text == "Scanning 1 marked range and record start frame 20 to end"


def test_build_issue_scan_state_segments_switches_only_at_marker_boundaries():
    scanner = _make_scanner(markers={5: {"id": 5}, 8: {"id": 8}})
    resolved_frames = []

    def fake_resolve(frame_number, *_args, **_kwargs):
        resolved_frames.append(frame_number)
        return {"frame": frame_number}, {"params": frame_number}

    scanner._resolve_scan_state_for_frame = fake_resolve

    segments = scanner._build_issue_scan_state_segments(
        [(3, 10)],
        {},
        {},
        {},
    )

    assert resolved_frames == [3, 5, 8]
    assert segments == [
        (3, 4, {"frame": 3}, {"params": 3}),
        (5, 7, {"frame": 5}, {"params": 5}),
        (8, 10, {"frame": 8}, {"params": 8}),
    ]


def test_resolve_scan_state_uses_control_defaults_snapshot_not_live_widgets():
    class _FailingWidgets:
        def items(self):
            raise AssertionError(
                "parameter_widgets should not be read in scan state resolution"
            )

    scanner = _make_scanner(
        parameter_widgets=_FailingWidgets(),
        control={"DetectorModelSelection": "live"},
    )

    with patch(
        MARKER_DATA_PATH,
        return_value={
            "parameters": {},
            "control": {
                "DetectorModelSelection": "marker",
                "IgnoredControl": "marker-only",
            },
        },
    ):
        local_control, local_params = scanner._resolve_scan_state_for_frame(
            10,
            {"DetectorModelSelection": "base"},
            {},
            {},
            {
                "DetectorModelSelection": "default",
                "DetectorScoreSlider": 33,
                "IgnoredDefault": "skip",
            },
        )

    assert local_control == {
        "DetectorModelSelection": "marker",
        "DetectorScoreSlider": 33,
    }
    assert local_params == {}


def test_resolve_scan_state_respects_explicitly_empty_target_faces_snapshot():
    scanner = _make_scanner(target_faces={"live-face": object()})

    with patch(MARKER_DATA_PATH, return_value={"parameters": {}, "control": {}}):
        _local_control, local_params = scanner._resolve_scan_state_for_frame(
            10,
            {},
            {},
            {},
            {},
        )

    assert local_params == {}


def test_resolve_scan_state_filters_non_scan_face_params_and_fills_defaults():
    scanner = _make_scanner(
        target_faces={"live-face": object()},
        default_parameters=SimpleNamespace(
            data={
                "SimilarityThresholdSlider": 50,
                "FaceExpressionEnableBothToggle": True,
            }
        ),
    )

    with patch(
        MARKER_DATA_PATH,
        return_value={
            "parameters": {
                "face_1": {
                    "SimilarityThresholdSlider": 72,
                    "FaceExpressionEnableBothToggle": True,
                }
            },
            "control": {"IgnoredControl": "skip"},
        },
    ):
        local_control, local_params = scanner._resolve_scan_state_for_frame(
            10,
            {"DetectorScoreSlider": 41, "IgnoredBase": "skip"},
            {"face_2": {"SimilarityThresholdSlider": 63}},
            {"face_1": {}, "face_2": {}},
            {"DetectorModelSelection": "SCRFD", "IgnoredDefault": "skip"},
        )

    assert local_control == {"DetectorModelSelection": "SCRFD"}
    assert local_params == {
        "face_1": {"SimilarityThresholdSlider": 72},
        "face_2": {"SimilarityThresholdSlider": 50},
    }


def test_prepare_issue_scan_match_context_uses_auto_snapshot_embeddings():
    scanner = _make_scanner()
    target_faces_snapshot = _make_target_snapshot(
        "face_1",
        {
            "arcface_128": {
                "Auto": np.array([3.0], dtype=np.float32),
                "Opal": np.array([1.0], dtype=np.float32),
                "Pearl": np.array([2.0], dtype=np.float32),
            }
        },
    )

    match_context = scanner._prepare_issue_scan_match_context(
        {
            "RecognitionModelSelection": "arcface_128",
            "SimilarityTypeSelection": "Pearl",
        },
        {"face_1": {"SimilarityThresholdSlider": 65}},
        target_faces_snapshot,
    )

    assert match_context["recognition_model"] == "arcface_128"
    assert match_context["similarity_type"] == "Auto"
    prepared_targets = match_context["prepared_targets"]
    assert len(prepared_targets) == 1
    assert prepared_targets[0][0] == "face_1"
    assert prepared_targets[0][1] == 65.0
    np.testing.assert_array_equal(
        prepared_targets[0][2],
        np.array([3.0], dtype=np.float32),
    )


def test_prepare_issue_scan_target_faces_snapshot_uses_auto_similarity_mode():
    class _TargetFaceWithoutEmbeddingAccess:
        def __init__(self):
            self.face_id = "face_1"
            self.cropped_face = np.zeros((8, 8, 3), dtype=np.uint8)

        def get_embedding(self, _recognition_model):
            raise AssertionError(
                "scan target snapshot should not call widget get_embedding"
            )

    function_worker = _RecordingFunctionWorker()
    scanner = _make_scanner(
        target_faces={"face_1": _TargetFaceWithoutEmbeddingAccess()},
        function_worker=function_worker,
    )

    with patch.object(
        scanner,
        "_build_issue_scan_state_segments",
        return_value=[
            (
                0,
                0,
                {
                    "RecognitionModelSelection": "arcface_128",
                    "SimilarityTypeSelection": "Opal",
                },
                {},
            ),
            (
                1,
                1,
                {
                    "RecognitionModelSelection": "arcface_128",
                    "SimilarityTypeSelection": "Pearl",
                },
                {},
            ),
        ],
    ):
        snapshot = scanner.prepare_issue_scan_target_faces_snapshot(
            [(0, 1)],
            {},
            {},
            {},
        )

    assert function_worker.recognize_calls == [("arcface_128", "Auto")]
    np.testing.assert_array_equal(
        snapshot["face_1"]["embeddings_by_model"]["arcface_128"]["Auto"],
        np.array([3.0], dtype=np.float32),
    )


def test_scan_issue_frames_filters_scan_state_before_detection():
    scanner = _make_scanner()
    captured = {}

    def fake_run(
        frame_rgb=None,
        local_control_for_worker=None,
        local_params_for_worker=None,
        is_master_edit_active=False,
        frame_tensor=None,
        detector_control_override=None,
        frame_number=-1,
    ):
        captured["local_control"] = local_control_for_worker
        captured["local_params"] = local_params_for_worker
        captured["detector_control_override"] = detector_control_override
        return _empty_scan_detection_result()

    capture_patch, read_patch, seek_patch, release_patch = _scan_patches(scanner)
    with (
        capture_patch,
        read_patch,
        seek_patch,
        release_patch,
        patch.object(scanner.sequential_detector, "run", side_effect=fake_run),
    ):
        result = scanner.scan_issue_frames(
            scan_ranges=[(0, 0)],
            base_control={
                "DetectorScoreSlider": 42,
                "KPSSmoothingEnableToggle": False,
                "FaceEditorEnableToggle": True,
            },
            base_params={
                "face_1": {
                    "SimilarityThresholdSlider": 77,
                    "FaceExpressionEnableBothToggle": True,
                }
            },
            target_faces_snapshot={},
        )

    assert result == {
        "issue_frames_by_face": {},
        "frames_scanned": 1,
        "faces_with_issues": 0,
        "cancelled": False,
    }
    assert captured["local_control"] == {
        "DetectorScoreSlider": 42,
        "KPSSmoothingEnableToggle": False,
    }
    assert captured["local_params"] == {"face_1": {"SimilarityThresholdSlider": 77}}
    assert captured["detector_control_override"] == captured["local_control"]


def test_scan_issue_frames_uses_marker_resolved_resize_per_segment():
    scanner = _make_scanner()
    preview_heights = []

    def fake_read_frame(_capture, _rotation, preview_target_height=None):
        preview_heights.append(preview_target_height)
        return True, np.zeros((4, 4, 3), dtype=np.uint8)

    capture_patch, read_patch, seek_patch, release_patch = _scan_patches(
        scanner, read_frame=fake_read_frame
    )
    with (
        capture_patch,
        read_patch,
        seek_patch,
        release_patch,
        patch.object(
            scanner,
            "_build_issue_scan_state_segments",
            return_value=[
                (0, 0, {"DetectorScoreSlider": 40}, {}),
                (
                    1,
                    1,
                    {
                        "GlobalInputResizeToggle": True,
                        "GlobalInputResizeSizeSelection": "720p",
                    },
                    {},
                ),
                (
                    2,
                    2,
                    {
                        "GlobalInputResizeToggle": True,
                        "GlobalInputResizeSizeSelection": "1080p",
                    },
                    {},
                ),
                (
                    3,
                    3,
                    {
                        "GlobalInputResizeToggle": False,
                        "GlobalInputResizeSizeSelection": "720p",
                    },
                    {},
                ),
            ],
        ),
        patch.object(
            scanner.sequential_detector,
            "run",
            return_value=_empty_scan_detection_result(),
        ),
    ):
        result = scanner.scan_issue_frames(
            scan_ranges=[(0, 3)],
            base_control={},
            base_params={},
            target_faces_snapshot={},
        )

    assert result == {
        "issue_frames_by_face": {},
        "frames_scanned": 4,
        "faces_with_issues": 0,
        "cancelled": False,
    }
    assert preview_heights == [None, 720, 1080, None]


def test_scan_issue_frames_uses_explicit_target_height_when_segment_has_no_resize_state():
    scanner = _make_scanner()
    preview_heights = []

    def fake_read_frame(_capture, _rotation, preview_target_height=None):
        preview_heights.append(preview_target_height)
        return True, np.zeros((4, 4, 3), dtype=np.uint8)

    capture_patch, read_patch, seek_patch, release_patch = _scan_patches(
        scanner, read_frame=fake_read_frame
    )
    with (
        capture_patch,
        read_patch,
        seek_patch,
        release_patch,
        patch.object(
            scanner,
            "_build_issue_scan_state_segments",
            return_value=[(0, 0, {"DetectorScoreSlider": 40}, {})],
        ),
        patch.object(
            scanner.sequential_detector,
            "run",
            return_value=_empty_scan_detection_result(),
        ),
    ):
        scanner.scan_issue_frames(
            scan_ranges=[(0, 0)],
            target_height=256,
            base_control={},
            base_params={},
            target_faces_snapshot={},
        )

    assert preview_heights == [256]


def test_scan_issue_frames_reports_progress_per_frame_and_skips_dropped_runs():
    scanner = _make_scanner(
        target_faces={
            "1": _DummyTargetFace(
                "1", {"arcface_128": np.array([1.0], dtype=np.float32)}
            )
        },
        dropped_frames={2, 3, 4, 11},
    )
    progress_updates = []
    seek_calls = []

    capture_patch, read_patch, seek_patch, release_patch = _scan_patches(
        scanner,
        seek_frame=lambda _capture, frame_number: seek_calls.append(frame_number),
    )
    with (
        capture_patch,
        read_patch,
        seek_patch,
        release_patch,
        patch.object(
            scanner,
            "_build_issue_scan_state_segments",
            return_value=[(0, 24, {}, {})],
        ),
        patch.object(
            scanner,
            "_prepare_issue_scan_match_context",
            return_value={
                "recognition_model": "arcface_128",
                "similarity_type": "Auto",
                "prepared_targets": [],
            },
        ),
        patch.object(
            scanner.sequential_detector,
            "run",
            return_value=_empty_scan_detection_result(),
        ),
    ):
        result = scanner.scan_issue_frames(
            scan_ranges=[(0, 24)],
            base_control={},
            base_params={},
            target_faces_snapshot=_make_target_snapshot("1"),
            progress_callback=lambda processed, total, frame_number: (
                progress_updates.append((processed, total, frame_number))
            ),
        )

    assert result == {
        "issue_frames_by_face": {
            "1": list(range(0, 2)) + list(range(5, 11)) + list(range(12, 25))
        },
        "frames_scanned": 21,
        "faces_with_issues": 1,
        "cancelled": False,
    }
    assert progress_updates == [
        (1, 21, 0),
        (2, 21, 1),
        (3, 21, 5),
        (4, 21, 6),
        (5, 21, 7),
        (6, 21, 8),
        (7, 21, 9),
        (8, 21, 10),
        (9, 21, 12),
        (10, 21, 13),
        (11, 21, 14),
        (12, 21, 15),
        (13, 21, 16),
        (14, 21, 17),
        (15, 21, 18),
        (16, 21, 19),
        (17, 21, 20),
        (18, 21, 21),
        (19, 21, 22),
        (20, 21, 23),
        (21, 21, 24),
    ]
    assert seek_calls == [0, 5, 12]


def test_scan_issue_frames_emits_incremental_issue_callback():
    scanner = _make_scanner()
    issue_updates = []

    capture_patch, read_patch, seek_patch, release_patch = _scan_patches(scanner)
    with (
        capture_patch,
        read_patch,
        seek_patch,
        release_patch,
        patch.object(
            scanner,
            "_build_issue_scan_state_segments",
            return_value=[(0, 0, {}, {})],
        ),
        patch.object(
            scanner.sequential_detector,
            "run",
            return_value=_empty_scan_detection_result(),
        ),
    ):
        result = scanner.scan_issue_frames(
            scan_ranges=[(0, 0)],
            base_control={},
            base_params={},
            target_faces_snapshot=_make_target_snapshot("1"),
            issue_found_callback=lambda face_id, frame_number: issue_updates.append(
                (face_id, frame_number)
            ),
        )

    assert result == {
        "issue_frames_by_face": {"1": [0]},
        "frames_scanned": 1,
        "faces_with_issues": 1,
        "cancelled": False,
    }
    assert issue_updates == [("1", 0)]


def test_scan_issue_frames_returns_partial_results_on_cancel():
    scanner = _make_scanner()
    issue_updates = []
    cancel_state = {"count": 0}

    def should_cancel():
        cancel_state["count"] += 1
        return cancel_state["count"] > 1

    capture_patch, read_patch, seek_patch, release_patch = _scan_patches(scanner)
    with (
        capture_patch,
        read_patch,
        seek_patch,
        release_patch,
        patch.object(
            scanner,
            "_build_issue_scan_state_segments",
            return_value=[(0, 1, {}, {})],
        ),
        patch.object(
            scanner.sequential_detector,
            "run",
            return_value=_empty_scan_detection_result(),
        ),
    ):
        result = scanner.scan_issue_frames(
            scan_ranges=[(0, 1)],
            base_control={},
            base_params={},
            target_faces_snapshot=_make_target_snapshot("1"),
            issue_found_callback=lambda face_id, frame_number: issue_updates.append(
                (face_id, frame_number)
            ),
            is_cancelled=should_cancel,
        )

    assert result == {
        "issue_frames_by_face": {"1": [0]},
        "frames_scanned": 1,
        "faces_with_issues": 1,
        "cancelled": True,
    }
    assert issue_updates == [("1", 0)]


# --- Tracker lifecycle -------------------------------------------------------
# Every reset goes through function_worker.reset_face_tracker: once directly and
# once via SequentialDetector.reset_state(). A scan with tracking enabled
# therefore resets three times (initial state reset, pre-scan tracker reset and
# the post-scan restore), plus two more for every tracking (re)configuration.


def test_scan_issue_frames_resets_tracker_before_and_after_tracking_scan():
    function_worker = _RecordingFunctionWorker()
    control = {
        "FaceTrackingEnableToggle": True,
        "DetectorModelSelection": "RetinaFace",
        "MaxFacesToDetectSlider": 1,
        "DetectorScoreSlider": 50,
        "LandmarkDetectToggle": False,
        "LandmarkDetectModelSelection": "203",
        "LandmarkDetectScoreSlider": 50,
        "DetectFromPointsToggle": False,
        "AutoRotationToggle": False,
        "LandmarkMeanEyesToggle": False,
        "KPSSmoothingEnableToggle": False,
        "RecognitionModelSelection": "arcface_128",
    }
    scanner = _make_scanner(control=control, function_worker=function_worker)

    capture_patch, read_patch, seek_patch, release_patch = _scan_patches(scanner)
    with (
        capture_patch,
        read_patch,
        seek_patch,
        release_patch,
        patch.object(
            scanner.sequential_detector,
            "run",
            return_value=_empty_scan_detection_result(),
        ),
    ):
        result = scanner.scan_issue_frames(
            scan_ranges=[(0, 0)],
            base_control=control,
            base_params={},
            target_faces_snapshot={},
        )

    assert result == {
        "issue_frames_by_face": {},
        "frames_scanned": 1,
        "faces_with_issues": 0,
        "cancelled": False,
    }
    assert function_worker.reset_calls == ["reset"] * 3


def _run_tracking_segment_scan(segments, scan_ranges):
    """Run a scan over pre-resolved segments and report the tracker resets."""
    function_worker = _RecordingFunctionWorker()
    scanner = _make_scanner(
        control={"FaceTrackingEnableToggle": False}, function_worker=function_worker
    )
    local_controls_seen = []

    def fake_run(
        frame_rgb=None,
        local_control_for_worker=None,
        detector_control_override=None,
        **_kwargs,
    ):
        local_controls_seen.append(
            (dict(local_control_for_worker), dict(detector_control_override or {}))
        )
        return _empty_scan_detection_result()

    capture_patch, read_patch, seek_patch, release_patch = _scan_patches(scanner)
    with (
        capture_patch,
        read_patch,
        seek_patch,
        release_patch,
        patch.object(
            scanner, "_build_issue_scan_state_segments", return_value=segments
        ),
        patch.object(scanner.sequential_detector, "run", side_effect=fake_run),
    ):
        result = scanner.scan_issue_frames(
            scan_ranges=scan_ranges,
            base_control={"FaceTrackingEnableToggle": False},
            base_params={},
            target_faces_snapshot={},
        )

    return result, function_worker.reset_calls, local_controls_seen


def test_scan_issue_frames_resets_tracker_when_marker_segment_enables_tracking():
    result, reset_calls, _ = _run_tracking_segment_scan(
        [
            (0, 0, {"FaceTrackingEnableToggle": False}, {}),
            (1, 1, {"FaceTrackingEnableToggle": True}, {}),
        ],
        [(0, 1)],
    )

    assert result["frames_scanned"] == 2
    assert reset_calls == ["reset"] * 5


@pytest.mark.parametrize(
    ("changed_key", "second_value"),
    [
        ("ByteTrackTrackThreshSlider", 55),
        ("ByteTrackMatchThreshSlider", 65),
        ("ByteTrackTrackBufferSlider", 45),
    ],
)
def test_scan_issue_frames_resets_tracker_when_bytetrack_config_changes_between_tracking_segments(
    changed_key, second_value
):
    first_control = {
        "FaceTrackingEnableToggle": True,
        "ByteTrackTrackThreshSlider": 40,
        "ByteTrackMatchThreshSlider": 80,
        "ByteTrackTrackBufferSlider": 30,
    }
    second_control = dict(first_control)
    second_control[changed_key] = second_value

    result, reset_calls, local_controls_seen = _run_tracking_segment_scan(
        [(0, 0, first_control, {}), (1, 1, second_control, {})],
        [(0, 1)],
    )

    assert result["frames_scanned"] == 2
    assert reset_calls == ["reset"] * 5
    assert local_controls_seen == [
        (first_control, first_control),
        (second_control, second_control),
    ]


def test_scan_issue_frames_keeps_tracker_when_bytetrack_config_is_unchanged_between_tracking_segments():
    shared_control = {
        "FaceTrackingEnableToggle": True,
        "ByteTrackTrackThreshSlider": 40,
        "ByteTrackMatchThreshSlider": 80,
        "ByteTrackTrackBufferSlider": 30,
    }

    result, reset_calls, _ = _run_tracking_segment_scan(
        [(0, 0, shared_control, {}), (1, 1, dict(shared_control), {})],
        [(0, 1)],
    )

    assert result["frames_scanned"] == 2
    # No mid-scan reconfiguration, so only the three lifecycle resets happen.
    assert reset_calls == ["reset"] * 3


def test_scan_issue_frames_resets_tracker_when_tracking_re_enters_after_disabled_segment():
    result, reset_calls, _ = _run_tracking_segment_scan(
        [
            (0, 0, {"FaceTrackingEnableToggle": True}, {}),
            (1, 1, {"FaceTrackingEnableToggle": False}, {}),
            (2, 2, {"FaceTrackingEnableToggle": True}, {}),
        ],
        [(0, 2)],
    )

    assert result["frames_scanned"] == 3
    assert reset_calls == ["reset"] * 5


def test_scan_issue_frames_clears_sequential_state_when_tracking_re_enters():
    scanner = _make_scanner(control={"FaceTrackingEnableToggle": False})
    detector = scanner.sequential_detector
    detector.last_detected_faces = [{"persisted": True}]
    detector._smoothed_kps = {1: np.array([[1.0, 2.0]], dtype=np.float32)}
    detector._smoothed_dense_kps = {1: np.array([[3.0, 4.0]], dtype=np.float32)}
    detector._smoothed_dense_kps_203 = {1: np.array([[5.0, 6.0]], dtype=np.float32)}
    state_snapshots = []
    frame_counter = {"value": 0}

    def fake_run(**_kwargs):
        state_snapshots.append(
            (
                list(detector.last_detected_faces),
                dict(detector._smoothed_kps),
                dict(detector._smoothed_dense_kps),
                dict(detector._smoothed_dense_kps_203),
            )
        )
        marker = frame_counter["value"]
        frame_counter["value"] += 1
        detector.last_detected_faces = [{"from_segment": marker}]
        detector._smoothed_kps = {marker: np.array([[9.0, 9.0]], dtype=np.float32)}
        detector._smoothed_dense_kps = {
            marker: np.array([[8.0, 8.0]], dtype=np.float32)
        }
        detector._smoothed_dense_kps_203 = {
            marker: np.array([[7.0, 7.0]], dtype=np.float32)
        }
        return _empty_scan_detection_result()

    capture_patch, read_patch, seek_patch, release_patch = _scan_patches(scanner)
    with (
        capture_patch,
        read_patch,
        seek_patch,
        release_patch,
        patch.object(
            scanner,
            "_build_issue_scan_state_segments",
            return_value=[
                (0, 0, {"FaceTrackingEnableToggle": True}, {}),
                (1, 1, {"FaceTrackingEnableToggle": False}, {}),
                (2, 2, {"FaceTrackingEnableToggle": True}, {}),
            ],
        ),
        patch.object(scanner.sequential_detector, "run", side_effect=fake_run),
    ):
        result = scanner.scan_issue_frames(
            scan_ranges=[(0, 2)],
            base_control={"FaceTrackingEnableToggle": False},
            base_params={},
            target_faces_snapshot={},
        )

    assert result["frames_scanned"] == 3
    # Cleared once up front and again when tracking re-enters.
    assert state_snapshots[0] == ([], {}, {}, {})
    assert state_snapshots[2] == ([], {}, {}, {})
    # The live state the user left behind survives the scan untouched.
    np.testing.assert_array_equal(
        detector._smoothed_dense_kps_203[1],
        np.array([[5.0, 6.0]], dtype=np.float32),
    )
    assert detector.last_detected_faces == [{"persisted": True}]
