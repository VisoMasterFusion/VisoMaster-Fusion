from __future__ import annotations

import numpy as np

from app.processors.mouth_action_detector import MouthActionDetector


def teardown_function() -> None:
    MouthActionDetector.unload()


def test_mouth_action_detector_loads_onnx_model(monkeypatch) -> None:
    monkeypatch.setenv("VISOMASTER_MOUTH_ACTION_PROVIDER", "CPU")
    detector = MouthActionDetector.get()

    assert detector.available, detector.load_error
    assert detector._input_name == "image_tensor:0"
    assert detector._boxes_name == "detected_boxes:0"
    assert detector._scores_name == "detected_scores:0"
    assert detector._classes_name == "detected_classes:0"


def test_mouth_action_detector_score_returns_probability(monkeypatch) -> None:
    monkeypatch.setenv("VISOMASTER_MOUTH_ACTION_PROVIDER", "CPU")
    detector = MouthActionDetector.get()
    frame: np.ndarray = np.zeros((3, 320, 320), dtype=np.uint8)

    score = detector.score(frame)

    assert 0.0 <= score <= 1.0


def test_mouth_action_detector_unload_clears_singleton(monkeypatch) -> None:
    monkeypatch.setenv("VISOMASTER_MOUTH_ACTION_PROVIDER", "CPU")
    first = MouthActionDetector.get()
    assert first.available

    MouthActionDetector.unload()

    second = MouthActionDetector.get()
    assert second is not first
    assert second.available
