"""Focused tests for the TUFA (98- and 314-point) and ORFormer landmark detectors."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from app.processors.face_detectors import FaceDetectors
from app.processors.face_landmark_detectors import FaceLandmarkDetectors
from app.processors.models_data import (
    fp16_safe_models_list,
    landmark_model_mapping,
    landmark_point_counts,
    models_list,
)
from app.processors.utils import faceutil

IDENTITY_IM = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)


def _detectors() -> FaceLandmarkDetectors:
    """A FaceLandmarkDetectors with only what these detectors touch."""
    inst = FaceLandmarkDetectors.__new__(FaceLandmarkDetectors)
    inst.models_processor = SimpleNamespace(  # type: ignore[assignment]
        models={}, load_model=lambda name: object(), device="cpu"
    )
    inst.function_worker = SimpleNamespace()  # type: ignore[assignment]
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
        "tufa314": {
            "model_name": "FaceLandmarkTUFA314",
            "function": inst.detect_face_landmark_tufa314,
        },
    }
    return inst


def _stub_crop(inst: FaceLandmarkDetectors, calls: list) -> None:
    """Replace _prepare_crop so no real warping happens, and record its arguments."""

    def fake(img, bbox, det_kpss, from_points, target_size, **kwargs):
        calls.append(
            {
                "from_points": from_points,
                "target_size": target_size,
                "scale": kwargs.get("scale"),
                "warp_mode": kwargs.get("warp_mode"),
            }
        )
        return torch.zeros(3, target_size, target_size), None, IDENTITY_IM

    inst._prepare_crop = fake  # type: ignore[assignment,method-assign]


# --------------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("model_name", "filename"),
    [
        ("FaceLandmarkTUFA98", "tufa_vits8_256_98pt.onnx"),
        ("FaceLandmarkTUFA314", "tufa_vits8_256_314pt.onnx"),
        ("FaceLandmarkORFormer98", "orformer_hgnet_wflw_98pt_256.onnx"),
    ],
)
def test_each_model_is_registered_once_with_a_matching_url(model_name, filename):
    entries = [item for item in models_list if item["model_name"] == model_name]

    assert len(entries) == 1
    assert entries[0]["local_path"].endswith(filename)
    assert entries[0]["url"].endswith(f"/{filename}")
    # download_models.py indexes ["hash"] unconditionally.
    assert len(entries[0]["hash"]) == 64


@pytest.mark.parametrize(
    "model_name",
    ["FaceLandmarkTUFA98", "FaceLandmarkTUFA314", "FaceLandmarkORFormer98"],
)
def test_neither_model_is_marked_fp16_safe(model_name):
    """All are broken under trt_fp16_enable and must never be added to that list.

    TUFA builds an fp16 engine that runs fine and emits ~69 px of error on a 256 px
    crop -- a silent failure; the 314-point export is the same graph, so it inherits
    that. ORFormer's fp16 build crashes outright (access violation on all three
    isolated probe attempts) and produces no engine. Measurements are in
    onnx-export-notes.md.
    """
    assert model_name not in fp16_safe_models_list


def test_all_three_modes_map_to_their_models():
    """landmark_model_mapping is what control_actions uses to load/unload on a
    dropdown change, so a missing entry silently disables the selection."""
    assert landmark_model_mapping["tufa98"] == "FaceLandmarkTUFA98"
    assert landmark_model_mapping["tufa314"] == "FaceLandmarkTUFA314"
    assert landmark_model_mapping["orformer98"] == "FaceLandmarkORFormer98"


def test_every_mapped_landmark_model_exists_in_models_list():
    known = {item["model_name"] for item in models_list}
    for mode, model_name in landmark_model_mapping.items():
        assert model_name in known, f"mode {mode!r} maps to unregistered {model_name!r}"


def test_every_mode_has_a_point_count():
    """frame_worker_standard builds a zero placeholder of this size when landmark
    detection comes back empty; a missing entry would silently produce a 5-point
    array where the UI expects a dense one."""
    assert set(landmark_point_counts) == set(landmark_model_mapping)
    assert landmark_point_counts["tufa98"] == 98
    assert landmark_point_counts["tufa314"] == 314
    assert landmark_point_counts["orformer98"] == 98


# --------------------------------------------------------------------------------
# TUFA
# --------------------------------------------------------------------------------
def test_tufa_scales_its_normalised_output_to_crop_pixels():
    inst = _detectors()
    calls: list = []
    _stub_crop(inst, calls)

    # Graph emits normalised coords; 0.25 -> 64 px, 0.5 -> 128 px at 256.
    normalised = np.full((1, 98, 2), 0.25, dtype=np.float32)
    normalised[0, 54] = (0.5, 0.5)
    inst._run_onnx_binding = lambda name, inputs, outputs: [normalised]  # type: ignore[method-assign]

    kpss_5, kpss, scores = inst.detect_face_landmark_tufa98(
        torch.zeros(3, 512, 512), np.array([0, 0, 100, 100]), None
    )

    assert kpss.shape == (98, 2)
    np.testing.assert_allclose(kpss[0], (64.0, 64.0), atol=1e-4)
    np.testing.assert_allclose(kpss[54], (128.0, 128.0), atol=1e-4)
    # WFLW nose index feeds slot 2 of the 5-point set.
    np.testing.assert_allclose(kpss_5[2], (128.0, 128.0), atol=1e-4)
    # TUFA has no per-point confidence, so it must pass through unthresholded.
    assert len(scores) == 0


def test_tufa_uses_a_1_15_upright_crop_and_ignores_from_points():
    inst = _detectors()
    calls: list = []
    _stub_crop(inst, calls)
    inst._run_onnx_binding = lambda *a, **k: [np.zeros((1, 98, 2), dtype=np.float32)]  # type: ignore[method-assign]

    inst.detect_face_landmark_tufa98(
        torch.zeros(3, 512, 512), np.array([0, 0, 100, 100]), None, from_points=True
    )

    assert calls == [
        {"from_points": False, "target_size": 256, "scale": 1.15, "warp_mode": None}
    ]


# --------------------------------------------------------------------------------
# TUFA 314
# --------------------------------------------------------------------------------
def test_tufa314_scales_its_normalised_output_to_crop_pixels():
    inst = _detectors()
    calls: list = []
    _stub_crop(inst, calls)

    normalised = np.full((1, 314, 2), 0.25, dtype=np.float32)
    normalised[0, faceutil.LANDMARK_314_NOSE_TIP] = (0.5, 0.5)
    inst._run_onnx_binding = lambda name, inputs, outputs: [normalised]  # type: ignore[method-assign]

    kpss_5, kpss, scores = inst.detect_face_landmark_tufa314(
        torch.zeros(3, 512, 512), np.array([0, 0, 100, 100]), None
    )

    assert kpss.shape == (314, 2)
    np.testing.assert_allclose(kpss[0], (64.0, 64.0), atol=1e-4)
    np.testing.assert_allclose(kpss_5[2], (128.0, 128.0), atol=1e-4)
    # Same as tufa98: no confidence head, so nothing to threshold.
    assert len(scores) == 0


def test_tufa314_uses_the_same_1_15_upright_crop_as_tufa98():
    inst = _detectors()
    calls: list = []
    _stub_crop(inst, calls)
    inst._run_onnx_binding = lambda *a, **k: [np.zeros((1, 314, 2), dtype=np.float32)]  # type: ignore[method-assign]

    inst.detect_face_landmark_tufa314(
        torch.zeros(3, 512, 512), np.array([0, 0, 100, 100]), None, from_points=True
    )

    assert calls == [
        {"from_points": False, "target_size": 256, "scale": 1.15, "warp_mode": None}
    ]


def test_tufa314_requests_its_own_graph():
    """A copy-paste of the tufa98 body would still pass every geometric assertion
    above while quietly running the 98-point weights."""
    inst = _detectors()
    calls: list = []
    _stub_crop(inst, calls)
    seen: list = []

    def fake_binding(name, inputs, outputs):
        seen.append((name, sorted(inputs), list(outputs)))
        return [np.zeros((1, 314, 2), dtype=np.float32)]

    inst._run_onnx_binding = fake_binding  # type: ignore[method-assign]
    inst.detect_face_landmark_tufa314(
        torch.zeros(3, 512, 512), np.array([0, 0, 100, 100]), None
    )

    assert seen == [("FaceLandmarkTUFA314", ["image"], ["landmarks"])]


def test_tufa314_five_point_slice_uses_the_shared_tufa_anchors():
    """The five indices are the anchors the 314-point prompt shares exactly with the
    98-point prompt. estimate_norm needs them in eye-L, eye-R, nose, lip-L, lip-R
    order, so a reordering here misaligns every swap."""
    pts = np.arange(314 * 2, dtype=np.float32).reshape(314, 2)

    landmark_5, _score = faceutil.convert_face_landmark_314_to_5(pts, np.zeros(314))

    np.testing.assert_allclose(landmark_5[0], pts[312])  # eye left (pupil)
    np.testing.assert_allclose(landmark_5[1], pts[313])  # eye right (pupil)
    np.testing.assert_allclose(landmark_5[2], pts[154])  # nose tip
    np.testing.assert_allclose(landmark_5[3], pts[51])  # lip left
    np.testing.assert_allclose(landmark_5[4], pts[258])  # lip right


def test_convert_face_landmark_x_to_5_dispatches_314():
    pts = np.arange(314 * 2, dtype=np.float32).reshape(314, 2)

    landmark_5, _score = faceutil.convert_face_landmark_x_to_5(
        pts, pts_score=np.zeros(314)
    )

    np.testing.assert_allclose(
        landmark_5, faceutil.convert_face_landmark_314_to_5(pts, np.zeros(314))[0]
    )


def test_parse_pt2_from_pt_x_does_not_fall_into_the_101_point_branch_for_314():
    """Without a 314 branch, parse_pt2_from_pt_x hits its ``shape[0] > 101`` case and
    reads pts[:101] as the 101-point topology -- which for this x-sorted set is the
    left third of the face, so the eye centre lands off the face entirely."""
    pts = np.zeros((314, 2), dtype=np.float32)
    pts[:] = (0.0, 0.0)
    pts[faceutil.LANDMARK_314_EYE_LEFT] = (90.0, 100.0)
    pts[faceutil.LANDMARK_314_EYE_RIGHT] = (170.0, 100.0)
    pts[faceutil.LANDMARK_314_LIP_LEFT] = (100.0, 170.0)
    pts[faceutil.LANDMARK_314_LIP_RIGHT] = (160.0, 170.0)

    pt2 = faceutil.parse_pt2_from_pt_x(pts, use_lip=True)

    np.testing.assert_allclose(pt2[0], (130.0, 100.0))  # eye centre
    np.testing.assert_allclose(pt2[1], (130.0, 170.0))  # lip centre


def test_parse_pt2_from_pt314_mean_eyes_uses_the_eye_corners():
    pts = np.zeros((314, 2), dtype=np.float32)
    pts[faceutil.LANDMARK_314_EYE_LEFT] = (1000.0, 1000.0)  # pupil, must be ignored
    pts[faceutil.LANDMARK_314_EYE_RIGHT] = (1000.0, 1000.0)
    pts[faceutil.LANDMARK_314_EYE_LEFT_CORNERS] = [(70.0, 100.0), (110.0, 100.0)]
    pts[faceutil.LANDMARK_314_EYE_RIGHT_CORNERS] = [(150.0, 100.0), (190.0, 100.0)]

    pt2 = faceutil.parse_pt2_from_pt314(pts, use_lip=False, use_mean_eyes=True)

    np.testing.assert_allclose(pt2[0], (90.0, 100.0))
    np.testing.assert_allclose(pt2[1], (170.0, 100.0))


# --------------------------------------------------------------------------------
# ORFormer
# --------------------------------------------------------------------------------
def test_orformer_output_is_already_in_crop_pixels():
    inst = _detectors()
    calls: list = []
    _stub_crop(inst, calls)

    pixels = np.full((1, 98, 2), 40.0, dtype=np.float32)
    occlusion = np.zeros((1, 1, 16, 16), dtype=np.float32)
    inst._run_onnx_binding = lambda *a, **k: [pixels, occlusion]  # type: ignore[method-assign]

    _kpss_5, kpss, _scores = inst.detect_face_landmark_orformer98(
        torch.zeros(3, 512, 512), np.array([0, 0, 100, 100]), None
    )

    # No * 256.0 here -- the scaling is baked into the graph.
    np.testing.assert_allclose(kpss[0], (40.0, 40.0), atol=1e-4)


def test_orformer_visibility_is_one_minus_occlusion_at_the_landmark_cell():
    inst = _detectors()
    calls: list = []
    _stub_crop(inst, calls)

    # 16x16 grid over a 256 px crop => one cell per 16 px.
    occlusion = np.zeros((1, 1, 16, 16), dtype=np.float32)
    occlusion[0, 0, 2, 3] = 0.75  # row 2 (y 32..47), col 3 (x 48..63)
    pixels = np.zeros((1, 98, 2), dtype=np.float32)
    pixels[0, 0] = (50.0, 40.0)  # lands in that cell
    pixels[0, 1] = (10.0, 10.0)  # lands in a zero-occlusion cell
    inst._run_onnx_binding = lambda *a, **k: [pixels, occlusion]  # type: ignore[method-assign]

    _kpss_5, _kpss, scores = inst.detect_face_landmark_orformer98(
        torch.zeros(3, 512, 512), np.array([0, 0, 100, 100]), None
    )

    assert scores.shape == (98,)
    assert scores[0] == pytest.approx(0.25)
    assert scores[1] == pytest.approx(1.0)


def test_orformer_visibility_sampling_clamps_out_of_range_points():
    """A landmark predicted outside the crop must not index past the 16x16 grid."""
    inst = _detectors()
    calls: list = []
    _stub_crop(inst, calls)

    occlusion = np.zeros((1, 1, 16, 16), dtype=np.float32)
    pixels = np.zeros((1, 98, 2), dtype=np.float32)
    pixels[0, 0] = (10_000.0, 10_000.0)
    pixels[0, 1] = (-500.0, -500.0)
    inst._run_onnx_binding = lambda *a, **k: [pixels, occlusion]  # type: ignore[method-assign]

    _kpss_5, _kpss, scores = inst.detect_face_landmark_orformer98(
        torch.zeros(3, 512, 512), np.array([0, 0, 100, 100]), None
    )

    assert np.all(np.isfinite(scores))


def test_orformer_uses_a_1_20_upright_crop_and_ignores_from_points():
    inst = _detectors()
    calls: list = []
    _stub_crop(inst, calls)
    inst._run_onnx_binding = lambda *a, **k: [  # type: ignore[method-assign]
        np.zeros((1, 98, 2), dtype=np.float32),
        np.zeros((1, 1, 16, 16), dtype=np.float32),
    ]

    inst.detect_face_landmark_orformer98(
        torch.zeros(3, 512, 512), np.array([0, 0, 100, 100]), None, from_points=True
    )

    assert calls == [
        {"from_points": False, "target_size": 256, "scale": 1.20, "warp_mode": None}
    ]


def test_orformer_returns_nothing_when_the_graph_omits_the_occlusion_output():
    inst = _detectors()
    calls: list = []
    _stub_crop(inst, calls)
    inst._run_onnx_binding = lambda *a, **k: [np.zeros((1, 98, 2), dtype=np.float32)]  # type: ignore[method-assign]

    result = inst.detect_face_landmark_orformer98(
        torch.zeros(3, 512, 512), np.array([0, 0, 100, 100]), None
    )

    assert result == ([], [], [])


# --------------------------------------------------------------------------------
# The two places ORFormer's visibility must NOT be read as a confidence
# --------------------------------------------------------------------------------
def test_run_detect_landmark_does_not_threshold_orformer_visibility():
    """ORFormer's score sits near 0.5 on a clean face, so thresholding it would
    reject good faces at any slider above ~50. It must pass through like '478'."""
    inst = _detectors()
    calls: list = []
    _stub_crop(inst, calls)

    occlusion = np.full((1, 1, 16, 16), 0.6, dtype=np.float32)  # visibility 0.4
    inst._run_onnx_binding = lambda *a, **k: [  # type: ignore[method-assign]
        np.full((1, 98, 2), 40.0, dtype=np.float32),
        occlusion,
    ]

    _kpss_5, kpss, scores = inst.run_detect_landmark(
        torch.zeros(3, 512, 512),
        np.array([0, 0, 100, 100]),
        None,
        detect_mode="orformer98",
        score=0.9,  # far above the 0.4 mean visibility
    )

    assert len(kpss) == 98, "orformer98 was filtered out by the score threshold"
    assert scores.mean() == pytest.approx(0.4)


def test_run_detect_landmark_still_thresholds_a_real_confidence_model():
    """Guard the exclusion above: it must not have disabled the filter generally."""
    inst = _detectors()
    calls: list = []
    _stub_crop(inst, calls)
    inst.detector_map["98"] = {
        "model_name": "FaceLandmark98",
        "function": lambda *a, **k: (
            np.zeros((5, 2), dtype=np.float32),
            np.zeros((98, 2), dtype=np.float32),
            np.full(98, 0.4, dtype=np.float32),
        ),
    }

    kpss_5, kpss, scores = inst.run_detect_landmark(
        torch.zeros(3, 512, 512),
        np.array([0, 0, 100, 100]),
        None,
        detect_mode="98",
        score=0.9,
    )

    assert (kpss_5, kpss, scores) == ([], [], [])


def test_refine_landmarks_keeps_orformer_keypoints_despite_a_low_score():
    """_refine_landmarks normally drops refined keypoints whose mean score is below the
    detector's confidence. ORFormer's visibility is not a confidence, so it is exempt
    -- otherwise its 5-point output would be discarded on every high-confidence face."""
    inst = FaceDetectors.__new__(FaceDetectors)
    refined_5 = np.array([[1.0, 2.0]] * 5, dtype=np.float32)
    inst.function_worker = SimpleNamespace(
        run_detect_landmark=lambda *a, **k: (
            refined_5,
            np.zeros((98, 2), dtype=np.float32),
            np.full(98, 0.4, dtype=np.float32),  # below the 0.99 detector score
        )
    )

    _det, kpss_5, _kpss, _scores = inst._refine_landmarks(
        img_landmark=torch.zeros(3, 512, 512),
        det=np.array([[0, 0, 100, 100]], dtype=np.float32),
        kpss=np.zeros((1, 5, 2), dtype=np.float32),
        score_values=np.array([0.99], dtype=np.float32),
        use_landmark_detection=True,
        landmark_detect_mode="orformer98",
        landmark_score=0.5,
        from_points=False,
    )

    np.testing.assert_allclose(kpss_5[0], refined_5)
