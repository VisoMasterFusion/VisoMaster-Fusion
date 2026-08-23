"""Unit tests for app.processors.video_utils.sequential_detector.SequentialDetector.

The per-frame detection pass (formerly VideoProcessor._run_sequential_detection)
now lives in this class, so the detector contract is exercised here directly:
the detector-control override that the issue scanner relies on, the no-target
early exit, and the temporal-smoothing pass over matched faces.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from app.processors.video_utils.sequential_detector import SequentialDetector

FIND_BEST_MATCH_PATH = (
    "app.processors.video_utils.sequential_detector.find_best_target_match"
)


def _empty_detection():
    return (
        np.empty((0, 4), dtype=np.float32),
        np.empty((0, 5, 2), dtype=np.float32),
        np.empty((0, 68, 2), dtype=np.float32),
    )


def _make_detector(function_worker, target_faces) -> SequentialDetector:
    main_window = SimpleNamespace(
        target_faces=target_faces,
        default_parameters=SimpleNamespace(data={"SimilarityThresholdSlider": 50}),
        models_processor=SimpleNamespace(device="cpu"),
        function_worker=function_worker,
    )
    return SequentialDetector(main_window)


def _base_control(**overrides):
    control = {
        "DetectorModelSelection": "RetinaFace",
        "MaxFacesToDetectSlider": 2,
        "DetectorScoreSlider": 50,
        "LandmarkDetectToggle": False,
        "LandmarkDetectModelSelection": "68",
        "LandmarkDetectScoreSlider": 50,
        "DetectFromPointsToggle": False,
        "AutoRotationToggle": False,
        "LandmarkMeanEyesToggle": False,
        "KPSSmoothingEnableToggle": False,
    }
    control.update(overrides)
    return control


def test_run_passes_the_detector_control_override_to_the_detector():
    captured = {}

    def fake_run_detect(*_args, **kwargs):
        captured["control_override"] = kwargs.get("control_override")
        return _empty_detection()

    detector = _make_detector(
        SimpleNamespace(run_detect=fake_run_detect), {"face_1": object()}
    )
    override = {"FaceTrackingEnableToggle": True, "DetectorScoreSlider": 35}

    bboxes, kpss_5, kpss, kpss_203 = detector.run(
        np.zeros((8, 8, 3), dtype=np.uint8),
        _base_control(),
        {},
        detector_control_override=override,
    )

    assert bboxes.shape == (0, 4)
    assert kpss_5.shape == (0, 5, 2)
    assert kpss.shape == (0, 68, 2)
    assert kpss_203.shape == (0, 203, 2)
    assert captured["control_override"] == override


def test_run_skips_detection_entirely_when_there_are_no_target_faces():
    def fail_run_detect(*_args, **_kwargs):
        raise AssertionError("detection must not run without target faces")

    detector = _make_detector(SimpleNamespace(run_detect=fail_run_detect), {})
    detector._smoothed_kps = {1: np.zeros((5, 2), dtype=np.float32)}

    bboxes, kpss_5, kpss, kpss_203 = detector.run(
        np.zeros((8, 8, 3), dtype=np.uint8), _base_control(), {}
    )

    assert bboxes.shape == (0, 4)
    assert kpss_5.shape == (0, 5, 2)
    assert kpss.shape == (0, 68, 2)
    assert kpss_203.shape == (0, 203, 2)
    # The stale smoothing state is dropped along with the lost target.
    assert detector._smoothed_kps == {}


def test_run_smooths_matched_faces_and_keeps_the_dense_counts_aligned():
    bboxes = np.array(
        [[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]], dtype=np.float32
    )
    kpss_5 = np.array(
        [
            [[1, 1], [2, 1], [1.5, 2], [1, 3], [2, 3]],
            [[21, 21], [22, 21], [21.5, 22], [21, 23], [22, 23]],
        ],
        dtype=np.float32,
    )
    dense_68 = np.full((68, 2), 4.0, dtype=np.float32)

    function_worker = SimpleNamespace(
        run_detect=lambda *_args, **_kwargs: (bboxes.copy(), kpss_5.copy(), None),
        run_recognize_direct=lambda *_args, **_kwargs: (
            np.ones(4, dtype=np.float32),
            None,
        ),
        run_detect_landmark=lambda *_args, **_kwargs: (
            np.empty((0, 2), dtype=np.float32),
            dense_68.copy(),
            None,
        ),
        findCosineDistance=lambda _a, _b: 1.0,
    )
    detector = _make_detector(function_worker, {"face_1": SimpleNamespace()})

    printed: list[str] = []
    with (
        patch(FIND_BEST_MATCH_PATH, return_value=(object(), None, 0.9)),
        patch(
            "builtins.print",
            side_effect=lambda *args: printed.append(" ".join(map(str, args))),
        ),
    ):
        out_bboxes, out_kpss_5, out_kpss, out_kpss_203 = detector.run(
            np.zeros((64, 64, 3), dtype=np.uint8),
            _base_control(
                LandmarkDetectToggle=True,
                KPSSmoothingEnableToggle=True,
                KPSEmaAlphaSlider=35,
            ),
            {},
            frame_number=34,
        )

    assert out_bboxes.shape == (2, 4)
    assert out_kpss_5.shape == (2, 5, 2)
    # One dense entry is produced per matched face, so the smoothing pass never
    # hits the count-mismatch guard.
    assert out_kpss.shape == (2, 68, 2)
    assert out_kpss_203.shape == (0, 203, 2)
    assert set(detector._smoothed_kps) == {0, 1}
    assert not [line for line in printed if "Dense KPS" in line]
