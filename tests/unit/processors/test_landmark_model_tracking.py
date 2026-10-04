"""Regression coverage for model tracking during concurrent unloads."""

import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from app.processors.face_landmark_detectors import FaceLandmarkDetectors


class LockedSet(set):
    def __init__(self, lock, values=()):
        super().__init__(values)
        self.lock = lock

    def __iter__(self):
        assert self.lock.locked()
        return super().__iter__()

    def add(self, value):
        assert self.lock.locked()
        super().add(value)

    def discard(self, value):
        assert self.lock.locked()
        super().discard(value)


@pytest.mark.parametrize("model", ["YawNet", "DEIMv2Wholebody49Head"])
def test_dependency_registration_uses_cache_lock(model):
    mp = MagicMock()
    mp.device = "cpu"
    session = MagicMock()
    session.get_inputs.return_value = [
        SimpleNamespace(name="image", shape=[1, 3, 128, 128])
    ]
    mp.models = {model: session}
    detector = FaceLandmarkDetectors(mp, MagicMock())
    detector.active_landmark_models = LockedSet(detector._cache_lock)
    detector._run_onnx_binding = MagicMock(return_value=[])
    frame = torch.zeros(3, 128, 128)
    if model == "YawNet":
        detector.estimate_head_yaw_yawnet(
            frame, np.array([16, 16, 100, 100]), head_bboxes=[]
        )
    else:
        detector.detect_head_bboxes_wholebody49(frame)
    assert model in detector.active_landmark_models


@pytest.mark.parametrize("keep_essential", [False, True])
def test_overlapping_unloads_preserve_new_registrations(keep_essential):
    detector = FaceLandmarkDetectors(MagicMock(), MagicMock())
    detector.active_landmark_models = LockedSet(
        detector._cache_lock, ["YawNet", "FaceLandmark203"]
    )
    entered = threading.Event()
    resume = threading.Event()

    def unload(name):
        # Model-manager calls must allow another thread to acquire the cache lock.
        assert detector._cache_lock.acquire(timeout=2)
        detector._cache_lock.release()
        if name == "YawNet" and threading.current_thread().name.startswith("tracking"):
            entered.set()
            assert resume.wait(5)

    detector.models_processor.unload_model.side_effect = unload
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="tracking") as pool:
        pending = pool.submit(detector.unload_models, keep_essential)
        try:
            assert entered.wait(5)
            detector.unload_models(keep_essential)
            with detector._cache_lock:
                detector.active_landmark_models.add("DEIMv2Wholebody49Head")
        finally:
            resume.set()
        pending.result(timeout=5)
    assert "YawNet" not in detector.active_landmark_models
    assert "DEIMv2Wholebody49Head" in detector.active_landmark_models
    assert ("FaceLandmark203" in detector.active_landmark_models) == keep_essential
