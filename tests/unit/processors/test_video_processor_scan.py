from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from app.processors.video_processor import VideoProcessor


class _DummyCapture:
    def isOpened(self):
        return True

    def set(self, *_args, **_kwargs):
        return True


class _DummyTargetFace:
    def __init__(self, face_id, embeddings):
        self.face_id = face_id
        self._embeddings = embeddings

    def get_embedding(self, recognition_model):
        return self._embeddings.get(recognition_model)


def _make_target_snapshot(face_id, embeddings_by_model=None):
    return {
        str(face_id): {
            "face_id": str(face_id),
            "embeddings_by_model": embeddings_by_model or {},
        }
    }


def test_scan_issue_frames_restores_dense_smoothing_state():
    processor = VideoProcessor.__new__(VideoProcessor)
    processor.media_path = "dummy.mp4"
    processor.media_rotation = 0
    processor.fps = 30.0
    processor.current_frame_number = 0
    processor.last_detected_faces = [{"id": 1}]
    processor._smoothed_kps = {1: np.array([[1.0, 2.0]], dtype=np.float32)}
    processor._smoothed_dense_kps = {1: np.array([[3.0, 4.0]], dtype=np.float32)}
    processor.last_detected_faces = [{"id": 1}]
    processor.main_window = SimpleNamespace(
        control={},
        parameters={},
        target_faces={},
        dropped_frames=set(),
        markers={},
        videoSeekSlider=SimpleNamespace(value=lambda: 7),
    )
    processor._get_target_input_height = lambda: 256

    with (
        patch(
            "app.processors.video_processor.cv2.VideoCapture",
            return_value=_DummyCapture(),
        ),
        patch("app.processors.video_processor.misc_helpers.release_capture"),
    ):
        result = processor.scan_issue_frames(
            scan_ranges=[(1, 0)],
            base_control={},
            base_params={},
            target_faces_snapshot={},
            reset_frame_number=3,
        )

    assert result == {
        "issue_frames_by_face": {},
        "frames_scanned": 0,
        "faces_with_issues": 0,
        "cancelled": False,
    }
    np.testing.assert_array_equal(
        processor._smoothed_dense_kps[1], np.array([[3.0, 4.0]], dtype=np.float32)
    )
    np.testing.assert_array_equal(
        processor._smoothed_kps[1], np.array([[1.0, 2.0]], dtype=np.float32)
    )
    assert processor.last_detected_faces == [{"id": 1}]
    assert processor.current_frame_number == 3


def test_describe_issue_scan_scope_uses_normalized_effective_ranges():
    processor = VideoProcessor.__new__(VideoProcessor)
    processor.max_frame_number = 100
    processor.main_window = SimpleNamespace(
        job_marker_pairs=[(20, 30), (10, 25), (40, None)]
    )

    scope_text = processor.describe_issue_scan_scope([(10, 30), (40, 100)])

    assert scope_text == "Scanning 1 marked range and record start frame 40 to end"


def test_describe_issue_scan_scope_uses_raw_open_start_when_ranges_merge():
    processor = VideoProcessor.__new__(VideoProcessor)
    processor.max_frame_number = 100
    processor.main_window = SimpleNamespace(job_marker_pairs=[(10, 30), (20, None)])

    scope_text = processor.describe_issue_scan_scope([(10, 100)])

    assert scope_text == "Scanning 1 marked range and record start frame 20 to end"


def test_build_issue_scan_state_segments_switches_only_at_marker_boundaries():
    processor = VideoProcessor.__new__(VideoProcessor)
    processor.main_window = SimpleNamespace(markers={5: {"id": 5}, 8: {"id": 8}})
    resolved_frames = []

    def fake_resolve(frame_number, *_args, **_kwargs):
        resolved_frames.append(frame_number)
        return {"frame": frame_number}, {"params": frame_number}

    processor._resolve_scan_state_for_frame = fake_resolve

    segments = processor._build_issue_scan_state_segments(
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
    processor = VideoProcessor.__new__(VideoProcessor)

    class _FailingWidgets:
        def items(self):
            raise AssertionError(
                "parameter_widgets should not be read in scan state resolution"
            )

    processor.main_window = SimpleNamespace(
        markers={},
        parameter_widgets=_FailingWidgets(),
        control={"ControlA": "live"},
        default_parameters=SimpleNamespace(data={"SimilarityThresholdSlider": 50}),
        target_faces={},
    )

    with patch(
        "app.processors.video_processor.video_control_actions._get_marker_data_for_position",
        return_value={
            "parameters": {},
            "control": {"ControlA": "marker", "ControlB": "marker-only"},
        },
    ):
        local_control, local_params = processor._resolve_scan_state_for_frame(
            10,
            {"ControlA": "base"},
            {},
            {},
            {"ControlA": "default", "ControlC": "default-only"},
        )

    assert local_control == {
        "ControlA": "marker",
        "ControlB": "marker-only",
        "ControlC": "default-only",
    }
    assert local_params == {}


def test_resolve_scan_state_respects_explicitly_empty_target_faces_snapshot():
    processor = VideoProcessor.__new__(VideoProcessor)

    processor.main_window = SimpleNamespace(
        markers={},
        target_faces={"live-face": object()},
        default_parameters=SimpleNamespace(data={"SimilarityThresholdSlider": 50}),
    )

    with patch(
        "app.processors.video_processor.video_control_actions._get_marker_data_for_position",
        return_value={"parameters": {}, "control": {}},
    ):
        _local_control, local_params = processor._resolve_scan_state_for_frame(
            10,
            {},
            {},
            {},
            {},
        )

    assert local_params == {}


def test_prepare_issue_scan_match_context_uses_snapshot_embeddings_for_segment_settings():
    processor = VideoProcessor.__new__(VideoProcessor)
    processor.main_window = SimpleNamespace(
        default_parameters=SimpleNamespace(data={"SimilarityThresholdSlider": 50}),
    )
    target_faces_snapshot = _make_target_snapshot(
        "face_1",
        {
            "arcface_128": {
                "Opal": np.array([1.0], dtype=np.float32),
                "Pearl": np.array([2.0], dtype=np.float32),
            }
        },
    )

    match_context = processor._prepare_issue_scan_match_context(
        {
            "RecognitionModelSelection": "arcface_128",
            "SimilarityTypeSelection": "Pearl",
        },
        {"face_1": {"SimilarityThresholdSlider": 65}},
        target_faces_snapshot,
    )

    prepared_targets = match_context["prepared_targets"]
    assert len(prepared_targets) == 1
    assert prepared_targets[0][0] == "face_1"
    assert prepared_targets[0][1] == 65.0
    np.testing.assert_array_equal(
        prepared_targets[0][2],
        np.array([2.0], dtype=np.float32),
    )


def test_prepare_issue_scan_target_faces_snapshot_uses_segment_recognition_settings():
    processor = VideoProcessor.__new__(VideoProcessor)
    run_recognize_calls = []

    class _TargetFaceWithoutEmbeddingAccess:
        def __init__(self):
            self.face_id = "face_1"
            self.cropped_face = np.zeros((8, 8, 3), dtype=np.uint8)

        def get_embedding(self, _recognition_model):
            raise AssertionError(
                "scan target snapshot should not call widget get_embedding"
            )

    def fake_run_recognize_direct(_img, _kps, similarity_type, recognition_model):
        run_recognize_calls.append((recognition_model, similarity_type))
        if similarity_type == "Pearl":
            return np.array([2.0], dtype=np.float32), None
        return np.array([1.0], dtype=np.float32), None

    processor.main_window = SimpleNamespace(
        target_faces={"face_1": _TargetFaceWithoutEmbeddingAccess()},
        default_parameters=SimpleNamespace(data={"SimilarityThresholdSlider": 50}),
        models_processor=SimpleNamespace(
            device="cpu",
            run_recognize_direct=fake_run_recognize_direct,
        ),
    )

    with patch.object(
        processor,
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
        snapshot = processor.prepare_issue_scan_target_faces_snapshot(
            [(0, 1)],
            {},
            {},
            {},
        )

    assert run_recognize_calls == [
        ("arcface_128", "Opal"),
        ("arcface_128", "Pearl"),
    ]
    np.testing.assert_array_equal(
        snapshot["face_1"]["embeddings_by_model"]["arcface_128"]["Opal"],
        np.array([1.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        snapshot["face_1"]["embeddings_by_model"]["arcface_128"]["Pearl"],
        np.array([2.0], dtype=np.float32),
    )


def test_scan_issue_frames_reports_progress_per_frame_and_skips_dropped_runs():
    processor = VideoProcessor.__new__(VideoProcessor)
    processor.media_path = "dummy.mp4"
    processor.media_rotation = 0
    processor.fps = 30.0
    processor.current_frame_number = 0
    processor.last_detected_faces = []
    processor._smoothed_kps = {}
    processor._smoothed_dense_kps = {}
    processor.main_window = SimpleNamespace(
        control={},
        parameters={},
        target_faces={
            "1": _DummyTargetFace(
                "1", {"arcface_128": np.array([1.0], dtype=np.float32)}
            )
        },
        dropped_frames={2, 3, 4, 11},
        markers={},
        videoSeekSlider=SimpleNamespace(value=lambda: 0),
        default_parameters=SimpleNamespace(data={"SimilarityThresholdSlider": 50}),
        models_processor=SimpleNamespace(device="cpu"),
    )
    processor._get_target_input_height = lambda: 256
    progress_updates = []
    seek_calls = []
    frame = np.zeros((4, 4, 3), dtype=np.uint8)

    def fake_read_frame(*_args, **_kwargs):
        return True, frame.copy()

    with (
        patch(
            "app.processors.video_processor.cv2.VideoCapture",
            return_value=_DummyCapture(),
        ),
        patch(
            "app.processors.video_processor.misc_helpers.read_frame",
            side_effect=fake_read_frame,
        ),
        patch(
            "app.processors.video_processor.misc_helpers.seek_frame",
            side_effect=lambda _capture, frame_number: seek_calls.append(frame_number),
        ),
        patch("app.processors.video_processor.misc_helpers.release_capture"),
        patch.object(
            processor,
            "_build_issue_scan_state_segments",
            return_value=[(0, 24, {}, {})],
        ),
        patch.object(
            processor,
            "_prepare_issue_scan_match_context",
            return_value={
                "recognition_model": "arcface_128",
                "similarity_type": "Opal",
                "prepared_targets": [],
            },
        ),
        patch.object(
            processor,
            "_run_sequential_detection",
            return_value=(
                np.empty((0, 4), dtype=np.float32),
                np.empty((0, 5, 2), dtype=np.float32),
                np.empty((0, 68, 2), dtype=np.float32),
            ),
        ),
    ):
        result = processor.scan_issue_frames(
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
    processor = VideoProcessor.__new__(VideoProcessor)
    processor.media_path = "dummy.mp4"
    processor.media_rotation = 0
    processor.fps = 30.0
    processor.current_frame_number = 0
    processor.last_detected_faces = []
    processor._smoothed_kps = {}
    processor._smoothed_dense_kps = {}
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    issue_updates = []

    processor.main_window = SimpleNamespace(
        control={},
        parameters={},
        target_faces={},
        dropped_frames=set(),
        markers={},
        videoSeekSlider=SimpleNamespace(value=lambda: 0),
        default_parameters=SimpleNamespace(data={"SimilarityThresholdSlider": 50}),
        models_processor=SimpleNamespace(device="cpu"),
    )
    processor._get_target_input_height = lambda: 256

    with (
        patch(
            "app.processors.video_processor.cv2.VideoCapture",
            return_value=_DummyCapture(),
        ),
        patch(
            "app.processors.video_processor.misc_helpers.read_frame",
            return_value=(True, frame.copy()),
        ),
        patch("app.processors.video_processor.misc_helpers.seek_frame"),
        patch("app.processors.video_processor.misc_helpers.release_capture"),
        patch.object(
            processor,
            "_build_issue_scan_state_segments",
            return_value=[(0, 0, {}, {})],
        ),
        patch.object(
            processor,
            "_prepare_issue_scan_match_context",
            return_value={
                "recognition_model": "arcface_128",
                "similarity_type": "Opal",
                "prepared_targets": [],
            },
        ),
        patch.object(
            processor,
            "_run_sequential_detection",
            return_value=(
                np.empty((0, 4), dtype=np.float32),
                np.empty((0, 5, 2), dtype=np.float32),
                np.empty((0, 68, 2), dtype=np.float32),
            ),
        ),
    ):
        result = processor.scan_issue_frames(
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
    processor = VideoProcessor.__new__(VideoProcessor)
    processor.media_path = "dummy.mp4"
    processor.media_rotation = 0
    processor.fps = 30.0
    processor.current_frame_number = 0
    processor.last_detected_faces = []
    processor._smoothed_kps = {}
    processor._smoothed_dense_kps = {}
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    issue_updates = []
    cancel_state = {"count": 0}

    processor.main_window = SimpleNamespace(
        control={},
        parameters={},
        target_faces={},
        dropped_frames=set(),
        markers={},
        videoSeekSlider=SimpleNamespace(value=lambda: 0),
        default_parameters=SimpleNamespace(data={"SimilarityThresholdSlider": 50}),
        models_processor=SimpleNamespace(device="cpu"),
    )
    processor._get_target_input_height = lambda: 256

    def should_cancel():
        cancel_state["count"] += 1
        return cancel_state["count"] > 1

    with (
        patch(
            "app.processors.video_processor.cv2.VideoCapture",
            return_value=_DummyCapture(),
        ),
        patch(
            "app.processors.video_processor.misc_helpers.read_frame",
            return_value=(True, frame.copy()),
        ),
        patch("app.processors.video_processor.misc_helpers.seek_frame"),
        patch("app.processors.video_processor.misc_helpers.release_capture"),
        patch.object(
            processor,
            "_build_issue_scan_state_segments",
            return_value=[(0, 1, {}, {})],
        ),
        patch.object(
            processor,
            "_prepare_issue_scan_match_context",
            return_value={
                "recognition_model": "arcface_128",
                "similarity_type": "Opal",
                "prepared_targets": [],
            },
        ),
        patch.object(
            processor,
            "_run_sequential_detection",
            return_value=(
                np.empty((0, 4), dtype=np.float32),
                np.empty((0, 5, 2), dtype=np.float32),
                np.empty((0, 68, 2), dtype=np.float32),
            ),
        ),
    ):
        result = processor.scan_issue_frames(
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


def test_scan_issue_frames_resets_tracker_before_and_after_tracking_scan():
    processor = VideoProcessor.__new__(VideoProcessor)
    processor.media_path = "dummy.mp4"
    processor.media_rotation = 0
    processor.fps = 30.0
    processor.current_frame_number = 0
    processor.last_detected_faces = []
    processor._smoothed_kps = {}
    processor._smoothed_dense_kps = {}
    reset_calls = []
    frame = np.zeros((4, 4, 3), dtype=np.uint8)

    def fake_run_detect(*_args, **_kwargs):
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0, 5, 2), dtype=np.float32),
            np.empty((0, 68, 2), dtype=np.float32),
        )

    processor.main_window = SimpleNamespace(
        control={
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
            "SimilarityTypeSelection": "Opal",
        },
        parameters={},
        target_faces={},
        dropped_frames=set(),
        markers={},
        videoSeekSlider=SimpleNamespace(value=lambda: 0),
        default_parameters=SimpleNamespace(data={"SimilarityThresholdSlider": 50}),
        models_processor=SimpleNamespace(
            device="cpu",
            run_detect=fake_run_detect,
            face_detectors=SimpleNamespace(
                reset_tracker=lambda: reset_calls.append("reset")
            ),
        ),
    )
    processor._get_target_input_height = lambda: 256

    with (
        patch(
            "app.processors.video_processor.cv2.VideoCapture",
            return_value=_DummyCapture(),
        ),
        patch(
            "app.processors.video_processor.misc_helpers.read_frame",
            return_value=(True, frame.copy()),
        ),
        patch("app.processors.video_processor.misc_helpers.seek_frame"),
        patch("app.processors.video_processor.misc_helpers.release_capture"),
    ):
        result = processor.scan_issue_frames(
            scan_ranges=[(0, 0)],
            base_control=processor.main_window.control,
            base_params={},
            target_faces_snapshot={},
            reset_frame_number=0,
        )

    assert result == {
        "issue_frames_by_face": {},
        "frames_scanned": 1,
        "faces_with_issues": 0,
        "cancelled": False,
    }
    assert reset_calls == ["reset", "reset"]


def test_scan_issue_frames_resets_tracker_when_marker_segment_enables_tracking():
    processor = VideoProcessor.__new__(VideoProcessor)
    processor.media_path = "dummy.mp4"
    processor.media_rotation = 0
    processor.fps = 30.0
    processor.current_frame_number = 0
    processor.last_detected_faces = []
    processor._smoothed_kps = {}
    processor._smoothed_dense_kps = {}
    reset_calls = []
    frame = np.zeros((4, 4, 3), dtype=np.uint8)

    processor.main_window = SimpleNamespace(
        control={"FaceTrackingEnableToggle": False},
        parameters={},
        target_faces={},
        dropped_frames=set(),
        markers={},
        videoSeekSlider=SimpleNamespace(value=lambda: 0),
        default_parameters=SimpleNamespace(data={"SimilarityThresholdSlider": 50}),
        models_processor=SimpleNamespace(
            device="cpu",
            face_detectors=SimpleNamespace(
                reset_tracker=lambda: reset_calls.append("reset")
            ),
        ),
    )
    processor._get_target_input_height = lambda: 256

    with (
        patch(
            "app.processors.video_processor.cv2.VideoCapture",
            return_value=_DummyCapture(),
        ),
        patch(
            "app.processors.video_processor.misc_helpers.read_frame",
            return_value=(True, frame.copy()),
        ),
        patch("app.processors.video_processor.misc_helpers.seek_frame"),
        patch("app.processors.video_processor.misc_helpers.release_capture"),
        patch.object(
            processor,
            "_build_issue_scan_state_segments",
            return_value=[
                (0, 0, {"FaceTrackingEnableToggle": False}, {}),
                (1, 1, {"FaceTrackingEnableToggle": True}, {}),
            ],
        ),
        patch.object(
            processor,
            "_prepare_issue_scan_match_context",
            return_value={
                "recognition_model": "arcface_128",
                "similarity_type": "Opal",
                "prepared_targets": [],
            },
        ),
        patch.object(
            processor,
            "_run_sequential_detection",
            return_value=(
                np.empty((0, 4), dtype=np.float32),
                np.empty((0, 5, 2), dtype=np.float32),
                np.empty((0, 68, 2), dtype=np.float32),
            ),
        ),
    ):
        result = processor.scan_issue_frames(
            scan_ranges=[(0, 1)],
            base_control={"FaceTrackingEnableToggle": False},
            base_params={},
            target_faces_snapshot={},
            reset_frame_number=0,
        )

    assert result == {
        "issue_frames_by_face": {},
        "frames_scanned": 2,
        "faces_with_issues": 0,
        "cancelled": False,
    }
    assert reset_calls == ["reset", "reset", "reset"]


def test_scan_issue_frames_resets_tracker_when_tracking_re_enters_after_disabled_segment():
    processor = VideoProcessor.__new__(VideoProcessor)
    processor.media_path = "dummy.mp4"
    processor.media_rotation = 0
    processor.fps = 30.0
    processor.current_frame_number = 0
    processor.last_detected_faces = []
    processor._smoothed_kps = {}
    processor._smoothed_dense_kps = {}
    reset_calls = []
    frame = np.zeros((4, 4, 3), dtype=np.uint8)

    processor.main_window = SimpleNamespace(
        control={"FaceTrackingEnableToggle": False},
        parameters={},
        target_faces={},
        dropped_frames=set(),
        markers={},
        videoSeekSlider=SimpleNamespace(value=lambda: 0),
        default_parameters=SimpleNamespace(data={"SimilarityThresholdSlider": 50}),
        models_processor=SimpleNamespace(
            device="cpu",
            face_detectors=SimpleNamespace(
                reset_tracker=lambda: reset_calls.append("reset")
            ),
        ),
    )
    processor._get_target_input_height = lambda: 256

    with (
        patch(
            "app.processors.video_processor.cv2.VideoCapture",
            return_value=_DummyCapture(),
        ),
        patch(
            "app.processors.video_processor.misc_helpers.read_frame",
            return_value=(True, frame.copy()),
        ),
        patch("app.processors.video_processor.misc_helpers.seek_frame"),
        patch("app.processors.video_processor.misc_helpers.release_capture"),
        patch.object(
            processor,
            "_build_issue_scan_state_segments",
            return_value=[
                (0, 0, {"FaceTrackingEnableToggle": True}, {}),
                (1, 1, {"FaceTrackingEnableToggle": False}, {}),
                (2, 2, {"FaceTrackingEnableToggle": True}, {}),
            ],
        ),
        patch.object(
            processor,
            "_prepare_issue_scan_match_context",
            return_value={
                "recognition_model": "arcface_128",
                "similarity_type": "Opal",
                "prepared_targets": [],
            },
        ),
        patch.object(
            processor,
            "_run_sequential_detection",
            return_value=(
                np.empty((0, 4), dtype=np.float32),
                np.empty((0, 5, 2), dtype=np.float32),
                np.empty((0, 68, 2), dtype=np.float32),
            ),
        ),
    ):
        result = processor.scan_issue_frames(
            scan_ranges=[(0, 2)],
            base_control={"FaceTrackingEnableToggle": False},
            base_params={},
            target_faces_snapshot={},
            reset_frame_number=0,
        )

    assert result == {
        "issue_frames_by_face": {},
        "frames_scanned": 3,
        "faces_with_issues": 0,
        "cancelled": False,
    }
    assert reset_calls == ["reset", "reset", "reset"]


def test_scan_issue_frames_clears_sequential_state_when_tracking_re_enters():
    processor = VideoProcessor.__new__(VideoProcessor)
    processor.media_path = "dummy.mp4"
    processor.media_rotation = 0
    processor.fps = 30.0
    processor.current_frame_number = 0
    processor.last_detected_faces = [{"persisted": True}]
    processor._smoothed_kps = {1: np.array([[1.0, 2.0]], dtype=np.float32)}
    processor._smoothed_dense_kps = {1: np.array([[3.0, 4.0]], dtype=np.float32)}
    reset_state_snapshots = []
    frame = np.zeros((4, 4, 3), dtype=np.uint8)

    processor.main_window = SimpleNamespace(
        control={"FaceTrackingEnableToggle": False},
        parameters={},
        target_faces={},
        dropped_frames=set(),
        markers={},
        videoSeekSlider=SimpleNamespace(value=lambda: 0),
        default_parameters=SimpleNamespace(data={"SimilarityThresholdSlider": 50}),
        models_processor=SimpleNamespace(
            device="cpu",
            face_detectors=SimpleNamespace(reset_tracker=lambda: None),
        ),
    )
    processor._get_target_input_height = lambda: 256

    def fake_prepare_issue_scan_match_context(*_args, **_kwargs):
        return {
            "recognition_model": "arcface_128",
            "similarity_type": "Opal",
            "prepared_targets": [],
        }

    def fake_run_sequential_detection(*_args, **_kwargs):
        reset_state_snapshots.append(
            (
                list(processor.last_detected_faces),
                dict(processor._smoothed_kps),
                dict(processor._smoothed_dense_kps),
            )
        )
        processor.last_detected_faces = [
            {"from_segment": processor.current_frame_number}
        ]
        processor._smoothed_kps = {
            processor.current_frame_number: np.array([[9.0, 9.0]], dtype=np.float32)
        }
        processor._smoothed_dense_kps = {
            processor.current_frame_number: np.array([[8.0, 8.0]], dtype=np.float32)
        }
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0, 5, 2), dtype=np.float32),
            np.empty((0, 68, 2), dtype=np.float32),
        )

    with (
        patch(
            "app.processors.video_processor.cv2.VideoCapture",
            return_value=_DummyCapture(),
        ),
        patch(
            "app.processors.video_processor.misc_helpers.read_frame",
            return_value=(True, frame.copy()),
        ),
        patch("app.processors.video_processor.misc_helpers.seek_frame"),
        patch("app.processors.video_processor.misc_helpers.release_capture"),
        patch.object(
            processor,
            "_build_issue_scan_state_segments",
            return_value=[
                (0, 0, {"FaceTrackingEnableToggle": True}, {}),
                (1, 1, {"FaceTrackingEnableToggle": False}, {}),
                (2, 2, {"FaceTrackingEnableToggle": True}, {}),
            ],
        ),
        patch.object(
            processor,
            "_prepare_issue_scan_match_context",
            side_effect=fake_prepare_issue_scan_match_context,
        ),
        patch.object(
            processor,
            "_run_sequential_detection",
            side_effect=fake_run_sequential_detection,
        ),
    ):
        result = processor.scan_issue_frames(
            scan_ranges=[(0, 2)],
            base_control={"FaceTrackingEnableToggle": False},
            base_params={},
            target_faces_snapshot={},
            reset_frame_number=0,
        )

    assert result == {
        "issue_frames_by_face": {},
        "frames_scanned": 3,
        "faces_with_issues": 0,
        "cancelled": False,
    }
    assert reset_state_snapshots[0] == ([], {}, {})
    assert reset_state_snapshots[2] == ([], {}, {})
