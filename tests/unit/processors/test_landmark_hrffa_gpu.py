"""End-to-end GPU checks for the HRFFA landmark detector and its DEIMv2 head detector.

These run the REAL ONNX graphs through the app's own call path --
detect_head_bboxes_wholebody49 and run_detect_landmark -> _prepare_crop ->
_run_onnx_binding (zero-copy IOBinding) -- rather than stubbing it. That is the only
way to catch a wrong input name, a squashed-vs-letterboxed preprocess, a bad crop
scale, or a broken coordinate round-trip.

Marked gpu, so skipped by default. Run with:
    pytest tests/unit/processors/test_landmark_hrffa_gpu.py -m gpu
Requires both ONNX files in model_assets/ (download_models.py fetches them).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from app.processors.face_landmark_detectors import FaceLandmarkDetectors
from app.processors.models_data import models_list

pytestmark = pytest.mark.gpu

PROJECT_ROOT = Path(__file__).resolve().parents[3]
FIXTURE = PROJECT_ROOT / "tests" / "fixtures" / "images" / "face_512.png"

# ibug68 semantic indices.
JAW = slice(0, 17)
CHIN = 8
L_EYE, R_EYE = slice(36, 42), slice(42, 48)
NOSE_TIP = 30
MOUTH_L, MOUTH_R = 48, 54


def _model_path(model_name: str) -> Path:
    entry = next(item for item in models_list if item["model_name"] == model_name)
    return Path(str(entry["local_path"]))


def _harness() -> FaceLandmarkDetectors:
    """FaceLandmarkDetectors backed by real ORT sessions for both graphs on CUDA."""
    import onnxruntime as ort

    sessions = {}
    for model_name in ("FaceLandmarkHRFFA", "DEIMv2Wholebody49Head"):
        path = _model_path(model_name)
        if not path.is_file():
            pytest.skip(f"{path.name} not in model_assets; run download_models.py")
        so = ort.SessionOptions()
        so.log_severity_level = 3
        sessions[model_name] = ort.InferenceSession(
            str(path),
            sess_options=so,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )

    inst = FaceLandmarkDetectors.__new__(FaceLandmarkDetectors)
    inst.models_processor = SimpleNamespace(  # type: ignore[assignment]
        models=sessions,
        load_model=lambda name: sessions[name],
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
        "hrffa": {
            "model_name": "FaceLandmarkHRFFA",
            "function": inst.detect_face_landmark_hrffa,
        },
    }
    return inst


@pytest.fixture(scope="module")
def frame() -> torch.Tensor:
    """The repo's own face fixture as a CHW uint8 RGB CUDA tensor, as the app
    passes it.

    It is a SYNTHETIC face (an ellipse with eyes and a mouth), so it is a wiring
    check, not an accuracy benchmark: it proves the crop geometry, the coordinate
    round-trip and the anatomical ordering, and nothing about landmark error.
    The accuracy cross-check against tufa98 needs a real face and lives at the
    bottom of this file.
    """
    import cv2

    if not FIXTURE.is_file():
        pytest.skip(f"fixture not available: {FIXTURE}")
    bgr = cv2.imread(str(FIXTURE))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb).to("cuda").permute(2, 0, 1).contiguous()


def test_the_head_detector_finds_the_head_and_it_contains_the_face(frame):
    """The fixture is a 512x512 face filling the frame, so a correct head box covers
    essentially all of it. This is what catches a squashed-vs-letterboxed
    preprocess: with the wrong one the box lands in the wrong part of the frame.
    """
    inst = _harness()

    heads = inst.detect_head_bboxes_wholebody49(frame)

    assert heads.shape[1] == 5
    assert len(heads) >= 1, "no head found in the reference frame"
    # Sorted by descending score.
    assert np.all(np.diff(heads[:, 4]) <= 0)

    x1, y1, x2, y2, score = heads[0]
    assert 0.0 <= x1 < x2 <= 511.0
    assert 0.0 <= y1 < y2 <= 511.0
    assert score >= inst.HRFFA_HEAD_SCORE_THRESHOLD
    # A head filling the frame: at least 70% of each axis.
    assert (x2 - x1) > 0.7 * 512
    assert (y2 - y1) > 0.7 * 512
    assert "DEIMv2Wholebody49Head" in inst.active_landmark_models


def test_hrffa_returns_68_points_anatomically_arranged(frame):
    inst = _harness()
    # A plausible face box inside the 512px fixture; HRFFA replaces it with the head
    # box its own detector finds.
    bbox = np.array([140.0, 150.0, 380.0, 440.0], dtype=np.float32)

    kpss_5, kpss, scores = inst.run_detect_landmark(
        frame, bbox, None, detect_mode="hrffa", score=0.5
    )

    assert len(kpss) == 68
    assert len(kpss_5) == 5
    # No per-point confidence, so nothing to threshold.
    assert len(scores) == 0
    assert np.all(np.isfinite(kpss))

    # Points land on the face, not somewhere in the frame at large.
    assert kpss[:, 0].min() > 0 and kpss[:, 0].max() < 512
    assert kpss[:, 1].min() > 0 and kpss[:, 1].max() < 512

    # Anatomy: eyes above nose tip above mouth, chin below the mouth. y grows down.
    eye_y = (kpss[L_EYE, 1].mean() + kpss[R_EYE, 1].mean()) / 2.0
    mouth_y = (kpss[MOUTH_L, 1] + kpss[MOUTH_R, 1]) / 2.0
    assert eye_y < kpss[NOSE_TIP, 1] < mouth_y, "eyes/nose/mouth misordered"
    assert kpss[CHIN, 1] > mouth_y, "chin is not below the mouth"
    # Subject's right eye (image left) left of the other, both inside the jaw x-span.
    assert kpss[L_EYE, 0].mean() < kpss[R_EYE, 0].mean()
    assert kpss[JAW, 0].min() <= kpss[L_EYE, 0].mean()
    assert kpss[JAW, 0].max() >= kpss[R_EYE, 0].mean()

    # kps_5 must be the ibug slice convert_face_landmark_68_to_5 promises, in the
    # order estimate_norm expects: eye centres, nose tip, mouth corners.
    np.testing.assert_allclose(kpss_5[2], kpss[NOSE_TIP], atol=1e-4)
    np.testing.assert_allclose(kpss_5[3], kpss[MOUTH_L], atol=1e-4)
    np.testing.assert_allclose(kpss_5[4], kpss[MOUTH_R], atol=1e-4)


def test_precomputed_head_boxes_give_the_same_result_as_self_service(frame):
    """The per-frame optimisation in sequential_detector / frame_worker_standard hands
    the boxes in via head_bboxes=. That must be exactly equivalent to letting the
    landmark function run the head detector itself, or the video path and the
    single-frame path would disagree."""
    inst = _harness()
    bbox = np.array([140.0, 150.0, 380.0, 440.0], dtype=np.float32)

    _, self_service, _ = inst.detect_face_landmark_hrffa(frame, bbox, None)
    heads = inst.detect_head_bboxes_wholebody49(frame)
    _, precomputed, _ = inst.detect_face_landmark_hrffa(
        frame, bbox, None, head_bboxes=heads
    )

    np.testing.assert_allclose(self_service, precomputed, atol=1e-3)


def test_the_fallback_head_box_still_produces_usable_landmarks(frame):
    """What the user gets when the head detector is missing or failed its TRT build.
    Worse than a real head box, but it must still land on the face with the right
    anatomy -- otherwise the degraded path is not worth having."""
    inst = _harness()
    bbox = np.array([140.0, 150.0, 380.0, 440.0], dtype=np.float32)

    _kpss_5, kpss, _scores = inst.detect_face_landmark_hrffa(
        frame, bbox, None, head_bboxes=np.empty((0, 5), dtype=np.float32)
    )

    assert len(kpss) == 68
    assert np.all(np.isfinite(kpss))
    eye_y = (kpss[L_EYE, 1].mean() + kpss[R_EYE, 1].mean()) / 2.0
    mouth_y = (kpss[MOUTH_L, 1] + kpss[MOUTH_R, 1]) / 2.0
    assert eye_y < kpss[NOSE_TIP, 1] < mouth_y
    assert kpss[CHIN, 1] > mouth_y

    # And it should agree with the real-head-box result to within a fraction of the
    # interocular distance -- an approximate crop, not a different face.
    _, reference, _ = inst.detect_face_landmark_hrffa(frame, bbox, None)
    interocular = float(
        np.linalg.norm(reference[R_EYE].mean(0) - reference[L_EYE].mean(0))
    )
    nme = float(np.linalg.norm(kpss - reference, axis=1).mean()) / interocular
    assert nme < 0.15, (
        f"fallback crop drifts too far from the head crop (NME {nme:.3f})"
    )


# ------------------------------------------------------------------------------
# Real face: the accuracy cross-check
# ------------------------------------------------------------------------------
TUFA_GIF = PROJECT_ROOT.parent / "TUFA" / "Figures" / "happy_98.gif"


@pytest.fixture(scope="module")
def real_face() -> tuple[torch.Tensor, np.ndarray]:
    """A real face frame plus its bbox, from TUFA's own demo GIF.

    The frame carries the authors' 98-point overlay in pure green, which is
    thresholded to recover the landmark extent -- a far better reference box than a
    hand-guessed one, and it means no labelled dataset is needed here. Same source
    as test_landmark_tufa_orformer_gpu.py.
    """
    import cv2
    from PIL import Image

    if not TUFA_GIF.is_file():
        pytest.skip(f"reference clip not available: {TUFA_GIF}")

    im = Image.open(str(TUFA_GIF))
    im.seek(0)
    bgr = cv2.cvtColor(np.array(im.convert("RGB")), cv2.COLOR_RGB2BGR)

    b, g, r = (bgr[:, :, i].astype(int) for i in range(3))
    mask = (g > 150) & (b < 110) & (r < 110) & (g - np.maximum(b, r) > 60)
    ys, xs = np.where(mask)
    assert len(xs) > 20, "green overlay not found in the reference frame"
    bbox = np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    frame = torch.from_numpy(rgb).to("cuda").permute(2, 0, 1).contiguous()
    return frame, bbox


def test_the_detected_head_box_encloses_the_face_on_a_real_frame(real_face):
    """Unlike the synthetic fixture the face here occupies part of a wider frame, so
    this is the check that the head box is genuinely localised rather than just
    covering everything."""
    frame, face = real_face
    inst = _harness()

    heads = inst.detect_head_bboxes_wholebody49(frame)

    assert len(heads) >= 1, "no head found in the reference frame"
    hx1, hy1, hx2, hy2, score = heads[0]
    assert score > 0.8, f"weak head detection on a clean frontal face: {score:.2f}"
    # A head box contains the face box and is larger than it, but not by more than
    # about 2x per axis -- that upper bound is what fails if the box is the whole
    # frame or a torso.
    assert hx1 <= face[0] and hy1 <= face[1]
    assert hx2 >= face[2] and hy2 >= face[3]
    assert (hx2 - hx1) < 2.5 * (face[2] - face[0])
    assert (hy2 - hy1) < 2.5 * (face[3] - face[1])


def test_hrffa_agrees_with_tufa98_on_the_five_swap_points(real_face):
    """The strongest correctness signal available without a labelled set.

    hrffa and tufa98 share no topology, no crop and no training data, and hrffa's
    crop is driven by a head box from a third model. If the crop geometry, the
    inverse transform or the 68-to-5 conversion were wrong, the five points that
    actually drive the swap would not line up.
    """
    frame, face = real_face
    path = _model_path("FaceLandmarkTUFA98")
    if not path.is_file():
        pytest.skip(f"{path.name} not in model_assets; run download_models.py")

    import onnxruntime as ort

    inst = _harness()
    so = ort.SessionOptions()
    so.log_severity_level = 3
    inst.models_processor.models["FaceLandmarkTUFA98"] = ort.InferenceSession(
        str(path),
        sess_options=so,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    kp5_hrffa, pts_hrffa, _ = inst.detect_face_landmark_hrffa(frame, face, None)
    kp5_tufa, _pts_tufa, _ = inst.detect_face_landmark_tufa98(frame, face, None)

    interocular = float(
        np.linalg.norm(pts_hrffa[R_EYE].mean(0) - pts_hrffa[L_EYE].mean(0))
    )
    err = np.linalg.norm(
        np.asarray(kp5_hrffa, dtype=np.float64)
        - np.asarray(kp5_tufa, dtype=np.float64),
        axis=1,
    )
    nme = err.mean() / interocular
    assert nme < 0.10, (
        f"hrffa and tufa98 disagree by {nme * 100:.1f}% of interocular distance "
        f"({err.mean():.2f} px); per-point {np.round(err, 2)}"
    )
