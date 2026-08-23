"""End-to-end GPU checks for the TUFA / ORFormer landmark detectors.

These run the REAL ONNX graphs through the app's own call path --
run_detect_landmark -> _prepare_crop -> _run_onnx_binding (zero-copy IOBinding) --
rather than stubbing it. That is the only way to catch a wrong input name, a bad
crop scale, or a broken coordinate round-trip.

Marked gpu, so skipped by default. Run with:
    pytest tests/unit/processors/test_landmark_tufa_orformer_gpu.py -m gpu
Requires the two ONNX files in model_assets/ (download_models.py fetches them).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from app.processors.face_landmark_detectors import FaceLandmarkDetectors
from app.processors.models_data import landmark_model_mapping, models_list

pytestmark = pytest.mark.gpu

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODES = ("tufa98", "orformer98")

# WFLW indices, used to assert the points are anatomically arranged rather than
# merely present. 0..32 = face contour, 96/97 = pupils, 54 = nose tip,
# 76/82 = mouth corners.
CONTOUR = slice(0, 33)
CHIN = 16
L_PUPIL, R_PUPIL = 96, 97
NOSE_TIP = 54
MOUTH_L, MOUTH_R = 76, 82


def _model_path(mode: str) -> Path:
    name = landmark_model_mapping[mode]
    entry = next(item for item in models_list if item["model_name"] == name)
    return Path(str(entry["local_path"]))


def _harness(mode: str) -> FaceLandmarkDetectors:
    """FaceLandmarkDetectors backed by a real ORT session on CUDA."""
    import onnxruntime as ort

    path = _model_path(mode)
    if not path.is_file():
        pytest.skip(f"{path.name} not in model_assets; run download_models.py")

    session = ort.InferenceSession(
        str(path), providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )

    inst = FaceLandmarkDetectors.__new__(FaceLandmarkDetectors)
    inst.models_processor = SimpleNamespace(  # type: ignore[assignment]
        models={landmark_model_mapping[mode]: session},
        load_model=lambda name: session,
        device="cuda",
        device_type="cuda",
        binding_device_id=0,
        check_and_clear_pending_build=lambda name: False,
    )
    inst.function_worker = SimpleNamespace(  # type: ignore[assignment]
        run_ort_with_iobinding=lambda sess, binding: sess.run_with_iobinding(binding)
    )
    inst.active_landmark_models = set()
    inst.detector_map = {
        "tufa98": {
            "model_name": "FaceLandmarkTUFA98",
            "function": inst.detect_face_landmark_tufa98,
        },
        "orformer98": {
            "model_name": "FaceLandmarkORFormer98",
            "function": inst.detect_face_landmark_orformer98,
        },
    }
    return inst


@pytest.fixture(scope="module")
def real_face() -> tuple[np.ndarray, np.ndarray]:
    """A real face frame plus its bbox, taken from TUFA's own demo GIF.

    The frame carries the authors' 98-point overlay drawn in pure green, which is
    thresholded to recover the landmark extent -- a far better reference box than a
    hand-guessed one, and it means no labelled dataset is needed here.
    """
    import cv2
    from PIL import Image

    gif = PROJECT_ROOT.parent / "TUFA" / "Figures" / "happy_98.gif"
    if not gif.is_file():
        pytest.skip(f"reference clip not available: {gif}")

    im = Image.open(gif)
    im.seek(0)
    bgr = cv2.cvtColor(np.array(im.convert("RGB")), cv2.COLOR_RGB2BGR)

    b, g, r = (bgr[:, :, i].astype(int) for i in range(3))
    mask = (g > 150) & (b < 110) & (r < 110) & (g - np.maximum(b, r) > 60)
    ys, xs = np.where(mask)
    assert len(xs) > 20, "green overlay not found in the reference frame"
    bbox = np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb, bbox


@pytest.mark.parametrize("mode", MODES)
def test_detector_returns_98_anatomically_arranged_points(mode, real_face):
    rgb, bbox = real_face
    img = torch.from_numpy(rgb).to("cuda").permute(2, 0, 1)

    inst = _harness(mode)
    kpss_5, kpss, _scores = inst.run_detect_landmark(
        img, bbox, None, detect_mode=mode, score=0.5
    )

    assert len(kpss) == 98, f"{mode} returned {len(kpss)} points"
    assert len(kpss_5) == 5
    assert np.all(np.isfinite(kpss))

    # Points must land on the face, not somewhere in the frame at large. Allow a
    # generous margin: the crop is 1.15-1.20x the box, so a contour point can sit
    # slightly outside it.
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    margin = 0.35 * max(w, h)
    assert kpss[:, 0].min() > bbox[0] - margin
    assert kpss[:, 0].max() < bbox[2] + margin
    assert kpss[:, 1].min() > bbox[1] - margin
    assert kpss[:, 1].max() < bbox[3] + margin

    # Anatomy: pupils above the nose tip, nose tip above the mouth corners, chin
    # below everything. y grows downward.
    eye_y = (kpss[L_PUPIL, 1] + kpss[R_PUPIL, 1]) / 2.0
    mouth_y = (kpss[MOUTH_L, 1] + kpss[MOUTH_R, 1]) / 2.0
    assert eye_y < kpss[NOSE_TIP, 1] < mouth_y, f"{mode}: eyes/nose/mouth misordered"
    assert kpss[CHIN, 1] > mouth_y, f"{mode}: chin is not below the mouth"
    # Left pupil left of right pupil, and both inside the contour's x-span.
    assert kpss[L_PUPIL, 0] < kpss[R_PUPIL, 0]
    assert kpss[CONTOUR, 0].min() <= kpss[L_PUPIL, 0]
    assert kpss[CONTOUR, 0].max() >= kpss[R_PUPIL, 0]

    # kps_5 must be the WFLW slice, in the order estimate_norm expects.
    np.testing.assert_allclose(kpss_5[0], kpss[L_PUPIL], atol=1e-4)
    np.testing.assert_allclose(kpss_5[1], kpss[R_PUPIL], atol=1e-4)
    np.testing.assert_allclose(kpss_5[2], kpss[NOSE_TIP], atol=1e-4)
    np.testing.assert_allclose(kpss_5[3], kpss[MOUTH_L], atol=1e-4)
    np.testing.assert_allclose(kpss_5[4], kpss[MOUTH_R], atol=1e-4)


def test_the_two_models_agree_on_a_real_face(real_face):
    """Both predict the same 98-point topology, so they should land in the same
    places. Disagreement well above published NME would mean one of the two crop
    conventions is wrong."""
    rgb, bbox = real_face
    img = torch.from_numpy(rgb).to("cuda").permute(2, 0, 1)

    preds = {}
    for mode in MODES:
        _kpss_5, kpss, _ = _harness(mode).run_detect_landmark(
            img, bbox, None, detect_mode=mode, score=0.5
        )
        preds[mode] = np.asarray(kpss, dtype=np.float64)

    interocular = np.linalg.norm(preds["tufa98"][60] - preds["tufa98"][72])
    nme = (
        np.mean(np.linalg.norm(preds["tufa98"] - preds["orformer98"], axis=1))
        / interocular
    )
    assert nme < 0.06, f"models disagree by {nme * 100:.2f}% of interocular distance"


def test_orformer_visibility_is_per_point_and_in_range(real_face):
    rgb, bbox = real_face
    img = torch.from_numpy(rgb).to("cuda").permute(2, 0, 1)

    _kpss_5, _kpss, scores = _harness("orformer98").run_detect_landmark(
        img, bbox, None, detect_mode="orformer98", score=0.5
    )

    scores = np.asarray(scores)
    assert scores.shape == (98,)
    assert scores.min() >= 0.0 and scores.max() <= 1.0
    # Not a constant map -- the signal has to actually vary across the face to be
    # worth anything downstream.
    assert scores.std() > 1e-3


def test_tufa_reports_no_scores(real_face):
    """TUFA has no confidence head; an accidental non-empty scores array would start
    being thresholded by run_detect_landmark."""
    rgb, bbox = real_face
    img = torch.from_numpy(rgb).to("cuda").permute(2, 0, 1)

    _kpss_5, _kpss, scores = _harness("tufa98").run_detect_landmark(
        img, bbox, None, detect_mode="tufa98", score=0.99
    )

    assert len(scores) == 0
