"""
YawNet head-yaw tests: ring-angle conventions and the degradation-guard handoff.
The inference test is skipped unless the ONNX weights are present locally.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from app.processors.face_landmark_detectors import FaceLandmarkDetectors

MODEL = "model_assets/yawnet_distill_128_unified_v6u_kappa_1x3x128x128.onnx"


@pytest.mark.parametrize("dtype", [torch.uint8, torch.float32])
def test_yawnet_normalization_preserves_source_frame(dtype):
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    mp = MagicMock()
    mp.device = torch.device("cpu")
    session = MagicMock()
    session.get_inputs.return_value = [
        SimpleNamespace(name="image", shape=[1, 3, 128, 128])
    ]
    mp.models = {"YawNet": session}
    det = FaceLandmarkDetectors(mp, MagicMock())
    frame = torch.full((3, 256, 256), 127, dtype=dtype)
    original = frame.clone()
    heads = np.array([[32, 32, 160, 160, 0.9]], dtype=np.float32)

    def infer(name, feed, outputs):
        crop = feed["image"]
        assert crop.shape == (1, 3, 128, 128)
        assert crop.dtype == torch.float32
        assert crop.device == mp.device
        assert crop.is_contiguous()
        torch.testing.assert_close(crop, torch.full_like(crop, 127 / 127.5 - 1))
        return [np.array([[0.0, 1.0]]), np.array([2.0])]

    det._run_onnx_binding = infer
    result = det.estimate_head_yaw_yawnet(
        frame, np.array([60, 60, 120, 140], dtype=np.float32), head_bboxes=heads
    )
    assert result == pytest.approx((270.0, 2.0))
    torch.testing.assert_close(frame, original)


@pytest.mark.parametrize("face_count", [0, 1, 2])
@pytest.mark.parametrize("angle", [0, 90])
@pytest.mark.parametrize("landmarks_enabled", [True, False, None])
def test_standard_frame_yaw_uses_working_coordinates(
    monkeypatch, face_count, angle, landmarks_enabled
):
    from collections import OrderedDict, defaultdict
    import threading
    from unittest.mock import MagicMock

    from torchvision.transforms import v2
    from app.processors.workers import frame_worker_standard as standard

    worker = MagicMock()
    worker.lock = threading.RLock()
    worker.main_window.target_faces = {}
    worker.main_window.swapfacesButton.isChecked.return_value = False
    worker.main_window.editFacesButton.isChecked.return_value = False
    worker.is_single_frame = True
    worker.precomputed_bboxes = None
    worker._resize_cache = OrderedDict()
    worker._RESIZE_CACHE_MAX = 16
    worker._MIN_FACE_PIXELS = 20
    worker.interpolation_scaleback = v2.InterpolationMode.BILINEAR
    worker.is_view_face_mask = worker.is_view_face_compare = False
    worker._find_best_target_match.return_value = (None, {}, 0.0)
    fw = worker.function_worker
    boxes = np.array([[100, 120, 220, 260]] * face_count, dtype=np.float32).reshape(
        -1, 4
    )
    points = np.array(
        [[[130, 150], [190, 150], [160, 185], [140, 225], [180, 225]]] * face_count,
        dtype=np.float32,
    ).reshape(-1, 5, 2)
    fw.run_detect.return_value = (
        boxes,
        points,
        np.zeros((face_count, 68, 2), dtype=np.float32),
    )
    fw.run_recognize_direct.return_value = (np.ones(512), None)
    heads = np.array([[80, 80, 240, 280, 0.9]], dtype=np.float32)
    fw.run_detect_head_bboxes.return_value = heads
    working_shape = (3, 768, 512) if angle else (3, 512, 768)

    def estimate(img, bbox, head_bboxes, min_kappa):
        assert tuple(img.shape) == working_shape
        np.testing.assert_array_equal(bbox, [100, 120, 220, 260])
        assert head_bboxes is heads
        assert min_kappa == 1.5
        return (90.0, 2.0)

    fw.estimate_head_yaw.side_effect = estimate
    overlays = []

    def draw(img, faces, **kwargs):
        assert tuple(img.shape) == (3, 256, 384)
        for face in faces:
            assert face["yawnet_deg"] == 90.0
            assert not np.array_equal(face["bbox"], boxes[0])
        overlays.append(faces)
        return img

    monkeypatch.setattr(
        standard, "draw_bounding_boxes_on_detected_faces", lambda img, *a, **kw: img
    )
    monkeypatch.setattr(standard, "draw_head_yaw_ring_on_faces", draw)
    control = defaultdict(
        bool,
        {
            "ShowAllDetectedFacesBBoxToggle": True,
            "YawNetEnableToggle": True,
            "ShowYawNetRingToggle": True,
            "YawNetMinKappaDecimalSlider": 1.5,
            "ManualRotationEnableToggle": bool(angle),
            "ManualRotationAngleSlider": angle,
        },
    )
    if landmarks_enabled is not None:
        control["LandmarkDetectToggle"] = landmarks_enabled
    frame = torch.zeros((3, 256, 384), dtype=torch.uint8)
    out = standard.StandardProcessor(worker).process_standard_frame(
        frame, control, threading.Event()
    )
    assert out.shape == frame.shape
    expected_faces = face_count if landmarks_enabled is not False else 0
    assert fw.run_detect_head_bboxes.call_count == bool(expected_faces)
    assert fw.estimate_head_yaw.call_count == expected_faces
    assert len(overlays) == bool(expected_faces)


# ---------------------------------------------------------------------------
# Ring-angle folding
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ring, expected",
    [
        (0.0, 0.0),  # facing camera
        (10.0, 10.0),
        (90.0, 90.0),  # full profile, subject's right
        (180.0, 180.0),  # facing away
        (270.0, 90.0),  # full profile, left -> same distance from frontal
        (350.0, 10.0),  # near-frontal from the other side, NOT 350
        (359.5, 0.5),
    ],
)
def test_yaw_from_frontal_folds_to_0_180(ring, expected):
    assert FaceLandmarkDetectors.yaw_from_frontal(ring) == pytest.approx(expected)


def test_yaw_from_frontal_is_side_symmetric():
    """Mirrored angles must fold to the same distance from frontal."""
    for d in (5.0, 30.0, 75.0, 120.0, 179.0):
        assert FaceLandmarkDetectors.yaw_from_frontal(d) == pytest.approx(
            FaceLandmarkDetectors.yaw_from_frontal(360.0 - d)
        )


def test_yaw_from_frontal_never_exceeds_180():
    for ring in np.arange(-720.0, 720.0, 7.0):
        v = FaceLandmarkDetectors.yaw_from_frontal(float(ring))
        assert 0.0 <= v <= 180.0


def test_folding_prevents_wraparound_false_negative():
    """
    The bug this folding exists to prevent: comparing a RAW ring angle against a
    threshold. 350 deg is 10 deg off frontal, but a naive `raw >= 40` reads it as
    extreme and would bypass the occluder on a nearly frontal face.
    """
    raw = 350.0
    assert raw >= 40.0  # naive check fires
    assert FaceLandmarkDetectors.yaw_from_frontal(raw) < 40.0  # folded check does not


# ---------------------------------------------------------------------------
# Biternion -> ring conversion, including the mirror step
# ---------------------------------------------------------------------------


def _ring_from_cos_sin(cos_v, sin_v):
    deg = (math.degrees(math.atan2(sin_v, cos_v)) + 360.0) % 360.0
    return (360.0 - deg) % 360.0


def test_mirror_step_is_present_and_matters():
    """
    YawNet emits the yawpose convention; the ring convention is its MIRROR. Dropping
    the (360 - deg) step silently swaps left and right, so assert the two differ for
    an off-axis angle.
    """
    cos_v, sin_v = math.cos(math.radians(60.0)), math.sin(math.radians(60.0))
    unmirrored = (math.degrees(math.atan2(sin_v, cos_v)) + 360.0) % 360.0
    assert unmirrored == pytest.approx(60.0)
    assert _ring_from_cos_sin(cos_v, sin_v) == pytest.approx(300.0)


def test_ring_conversion_anchors():
    assert _ring_from_cos_sin(1.0, 0.0) == pytest.approx(0.0)  # frontal
    assert _ring_from_cos_sin(-1.0, 0.0) == pytest.approx(180.0)  # facing away
    # +-90 map to the two profiles and fold to the same distance from frontal
    right = _ring_from_cos_sin(0.0, -1.0)
    left = _ring_from_cos_sin(0.0, 1.0)
    assert sorted([right, left]) == pytest.approx([90.0, 270.0])
    assert FaceLandmarkDetectors.yaw_from_frontal(right) == pytest.approx(
        FaceLandmarkDetectors.yaw_from_frontal(left)
    )


# ---------------------------------------------------------------------------
# Guard handoff: YawNet degrees must reach the occluder/XSeg bypass
# ---------------------------------------------------------------------------


def test_guard_threshold_is_documented_as_uncalibrated():
    """
    The threshold is a plausible value, not a measured one. If someone tunes it they
    should also update the note, so keep the two coupled.
    """
    import inspect

    from app.processors import face_masks

    src = inspect.getsource(face_masks)
    assert "_SEG_GUARD_MIN_ABS_YAW = 40.0" in src
    assert "NOT CALIBRATED" in src


def test_kappa_gate_defaults_to_off():
    """
    2.0 is YawNet's neutral prior (softplus(1.85)), not a confidence floor, so the
    gate must not default to anything near it.
    """
    assert FaceLandmarkDetectors.YAWNET_MIN_KAPPA == 0.0


# ---------------------------------------------------------------------------
# Inference (requires the downloaded weights)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not __import__("pathlib").Path(MODEL).exists(),
    reason="YawNet weights not downloaded",
)
def test_estimate_head_yaw_end_to_end():
    import onnxruntime as ort
    from unittest.mock import MagicMock

    sess = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
    mp = MagicMock()
    mp.models = {"YawNet": sess}
    mp.load_model.return_value = sess
    mp.device = torch.device("cpu")

    det = FaceLandmarkDetectors(mp, MagicMock())
    # Exercise our crop/normalise/convert code against the real graph.
    det._run_onnx_binding = lambda name, feed, outs: sess.run(
        outs, {k: v.cpu().numpy() for k, v in feed.items()}
    )

    frame = torch.randint(0, 256, (3, 480, 640), dtype=torch.uint8)
    face = np.array([300.0, 200.0, 380.0, 300.0], dtype=np.float32)

    out = det.estimate_head_yaw_yawnet(frame, face)
    assert out is not None
    deg, kappa = out
    assert 0.0 <= deg < 360.0
    assert 1e-3 <= kappa <= 100.0  # softplus().clamp(1e-3, 100) upstream

    # A head box that survives clipping to the frame edge must not raise.
    assert (
        det.estimate_head_yaw_yawnet(
            frame, np.array([-40.0, -40.0, 30.0, 30.0], dtype=np.float32)
        )
        is not None
    )

    # A sub-2px box has no usable crop -> None, not an exception.
    assert (
        det.estimate_head_yaw_yawnet(
            frame,
            np.array([10.0, 10.0, 10.4, 10.4], dtype=np.float32),
            head_bboxes=np.array([[10.0, 10.0, 11.0, 11.0, 0.9]], dtype=np.float32),
        )
        is None
    )


# ---------------------------------------------------------------------------
# Preview ring overlay
# ---------------------------------------------------------------------------


def _draw(deg, bbox=(150.0, 200.0, 250.0, 320.0), size=(400, 400)):
    from app.helpers.miscellaneous import draw_head_yaw_ring_on_faces

    img = torch.zeros((3, size[0], size[1]), dtype=torch.uint8)
    face = {"bbox": np.array(bbox, dtype=np.float32)}
    if deg is not None:
        face["yawnet_deg"] = deg
    out = draw_head_yaw_ring_on_faces(img, [face])
    return out.sum(0) > 0


def test_ring_not_drawn_without_an_angle():
    """A face the estimator skipped must draw nothing rather than a default angle."""
    assert _draw(None).sum().item() == 0


def test_ring_is_drawn_with_an_angle():
    assert _draw(90.0).sum().item() > 0


@pytest.mark.parametrize(
    "deg, axis, sign",
    [
        (0.0, "y", +1),  # facing camera -> needle DOWN
        (90.0, "x", +1),  # right
        (180.0, "y", -1),  # facing away -> needle UP
        (270.0, "x", -1),  # left
    ],
)
def test_needle_points_the_right_way(deg, axis, sign):
    """
    Guards the dx = sin, dy = cos mapping. Getting this wrong would draw a head
    turned left as turned right, which is the kind of error nobody notices in code
    review but everybody notices on screen.
    """
    painted = _draw(deg)
    ys, xs = torch.nonzero(painted, as_tuple=True)
    cy = (ys.min() + ys.max()).item() / 2.0
    cx = (xs.min() + xs.max()).item() / 2.0
    # The needle biases the painted centroid away from the ring's geometric centre.
    bias_y = ys.float().mean().item() - cy
    bias_x = xs.float().mean().item() - cx
    moved, still = (bias_y, bias_x) if axis == "y" else (bias_x, bias_y)
    assert sign * moved > 0.3, f"needle bias {moved:+.2f} on {axis} for {deg} deg"
    assert abs(still) < 0.3, f"unexpected bias {still:+.2f} on the other axis"


def test_ring_stays_in_bounds_for_a_face_at_the_frame_edge():
    """A face flush against the top edge must not raise or write out of bounds."""
    painted = _draw(45.0, bbox=(5.0, 2.0, 60.0, 60.0))
    assert painted.shape == (400, 400)


def test_ring_handles_tiny_and_huge_faces():
    from app.helpers.miscellaneous import draw_head_yaw_ring_on_faces

    for bbox in [(200.0, 200.0, 203.0, 203.0), (0.0, 0.0, 399.0, 399.0)]:
        img = torch.zeros((3, 400, 400), dtype=torch.uint8)
        draw_head_yaw_ring_on_faces(
            img, [{"bbox": np.array(bbox, dtype=np.float32), "yawnet_deg": 120.0}]
        )  # must not raise


def test_yawnet_kappa_gate_is_exposed_in_the_ui():
    """
    kappa has no established threshold, so the control must stay reachable for users
    to calibrate against their own footage.
    """
    import ast
    import io

    src = io.open("app/ui/widgets/settings_layout_data.py", encoding="utf-8").read()
    keys = {
        k.value
        for node in ast.walk(ast.parse(src))
        if isinstance(node, ast.Dict)
        for k in node.keys
        if isinstance(k, ast.Constant) and isinstance(k.value, str)
    }
    assert "YawNetMinKappaDecimalSlider" in keys


@pytest.mark.skipif(
    not __import__("pathlib").Path(MODEL).exists(),
    reason="YawNet weights not downloaded",
)
def test_kappa_gate_honours_the_ui_override():
    import onnxruntime as ort
    from unittest.mock import MagicMock

    sess = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
    mp = MagicMock()
    mp.models = {"YawNet": sess}
    mp.load_model.return_value = sess
    mp.device = torch.device("cpu")
    det = FaceLandmarkDetectors(mp, MagicMock())
    det._run_onnx_binding = lambda name, feed, outs: sess.run(
        outs, {k: v.cpu().numpy() for k, v in feed.items()}
    )

    frame = torch.randint(0, 256, (3, 480, 640), dtype=torch.uint8)
    face = np.array([300.0, 200.0, 380.0, 300.0], dtype=np.float32)

    base = det.estimate_head_yaw_yawnet(frame, face)
    assert base is not None, "default gate (0) must accept every reading"
    _, kappa = base

    # A threshold above the reading rejects it; one below accepts it.
    assert det.estimate_head_yaw_yawnet(frame, face, min_kappa=kappa + 1.0) is None
    assert (
        det.estimate_head_yaw_yawnet(frame, face, min_kappa=max(kappa - 0.1, 1e-3))
        is not None
    )
