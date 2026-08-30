"""Focused tests for the HRFFA landmark detector and its DEIMv2 head detector.

HRFFA is the only landmark model here that predicts on a whole-head crop rather than a
face crop, so it drives two ONNX graphs: DEIMv2-Wholebody49 for the head box and HRFFA
for the 68 points. These tests pin the parts that are easy to get silently wrong -- the
class-7 filter, the normalised-vs-absolute box decode, the head-to-face matching, the
1.1x crop geometry and the * 256 denormalisation.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from app.processors.face_landmark_detectors import FaceLandmarkDetectors
from app.processors.models_data import (
    fp16_safe_models_list,
    landmark_model_mapping,
    landmark_point_counts,
    models_list,
)

IDENTITY_IM = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)

HEAD_CLASS_ID = 7


def _detectors(models: dict | None = None) -> FaceLandmarkDetectors:
    """A FaceLandmarkDetectors with only what this detector touches."""
    inst = FaceLandmarkDetectors.__new__(FaceLandmarkDetectors)
    inst.models_processor = SimpleNamespace(  # type: ignore[assignment]
        models=models if models is not None else {},
        load_model=lambda name: object(),
        device="cpu",
    )
    inst.function_worker = SimpleNamespace()  # type: ignore[assignment]
    inst.active_landmark_models = set()
    inst.detector_map = {
        "hrffa": {
            "model_name": "FaceLandmarkHRFFA",
            "function": inst.detect_face_landmark_hrffa,
        },
    }
    return inst


def _stub_crop(inst: FaceLandmarkDetectors, calls: list) -> None:
    """Replace _prepare_crop so no real warping happens, and record its arguments."""

    def fake(img, bbox, det_kpss, from_points, target_size, **kwargs):
        calls.append(
            {
                "bbox": np.asarray(bbox, dtype=np.float32).copy(),
                "det_kpss": det_kpss,
                "from_points": from_points,
                "target_size": target_size,
                "scale": kwargs.get("scale"),
                "warp_mode": kwargs.get("warp_mode"),
            }
        )
        return torch.zeros(3, target_size, target_size), None, IDENTITY_IM

    inst._prepare_crop = fake  # type: ignore[assignment,method-assign]


def _fake_session(
    input_shape=(1, 3, 640, 640),
    input_name: str = "images",
    extra_inputs: tuple[str, ...] = (),
) -> SimpleNamespace:
    inputs = [SimpleNamespace(name=input_name, shape=list(input_shape))]
    inputs += [SimpleNamespace(name=n, shape=[1, 2]) for n in extra_inputs]
    return SimpleNamespace(get_inputs=lambda: inputs)


def _label_xyxy_score(rows: list[tuple[int, float, float, float, float, float]]):
    return np.array([rows], dtype=np.float32)


# --------------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("model_name", "filename"),
    [
        ("FaceLandmarkHRFFA", "hrffa_vitt_ibug68_1x3x256x256.onnx"),
        (
            "DEIMv2Wholebody49Head",
            "deimv2_hgnetv2_n_wholebody49_boxes_only_webgpu.onnx",
        ),
    ],
)
def test_each_model_is_registered_once_with_a_matching_url(model_name, filename):
    entries = [item for item in models_list if item["model_name"] == model_name]

    assert len(entries) == 1
    assert entries[0]["local_path"].endswith(filename)
    assert entries[0]["url"].endswith(f"/{filename}")
    # download_models.py indexes ["hash"] unconditionally.
    assert len(entries[0]["hash"]) == 64


@pytest.mark.parametrize("model_name", ["FaceLandmarkHRFFA", "DEIMv2Wholebody49Head"])
def test_neither_model_is_marked_fp16_safe(model_name):
    """Both are the graph shapes that already failed fp16 here and neither has been
    measured: HRFFA-vitt is a ViT regression head (TUFA's silent ~69 px failure mode)
    and DEIMv2 is a DETR-style decoder (ORFormer's access-violation failure mode)."""
    assert model_name not in fp16_safe_models_list


def test_hrffa_mode_maps_to_its_model():
    """landmark_model_mapping is what control_actions uses to load/unload on a
    dropdown change, so a missing entry silently disables the selection."""
    assert landmark_model_mapping["hrffa"] == "FaceLandmarkHRFFA"


def test_the_head_detector_is_not_a_landmark_mode():
    """DEIMv2Wholebody49Head is a dependency, not a selectable landmark model. If it
    leaked into landmark_model_mapping, control_actions would try to load it as the
    mode's own model and unload the actual HRFFA graph."""
    assert "DEIMv2Wholebody49Head" not in landmark_model_mapping.values()


def test_hrffa_has_a_point_count_of_68():
    """frame_worker_standard builds a zero placeholder of this size when landmark
    detection comes back empty."""
    assert set(landmark_point_counts) == set(landmark_model_mapping)
    assert landmark_point_counts["hrffa"] == 68


# --------------------------------------------------------------------------------
# DEIMv2 head detection
# --------------------------------------------------------------------------------
def test_head_detection_keeps_only_class_7_above_the_threshold():
    inst = _detectors({"DEIMv2Wholebody49Head": _fake_session()})
    # class 0 = body, class 7 = head. Only heads over the threshold survive.
    pred = _label_xyxy_score(
        [
            (0, 0.10, 0.10, 0.90, 0.90, 0.99),  # wrong class, high score
            (HEAD_CLASS_ID, 0.20, 0.10, 0.40, 0.30, 0.90),
            (HEAD_CLASS_ID, 0.60, 0.10, 0.80, 0.30, 0.10),  # below threshold
        ]
    )
    inst._run_onnx_binding = lambda *a, **k: [pred]  # type: ignore[method-assign]

    heads = inst.detect_head_bboxes_wholebody49(torch.zeros(3, 200, 100))

    assert heads.shape == (1, 5)
    # Normalised coords scale by the ORIGINAL frame size (W=100, H=200), not 640.
    np.testing.assert_allclose(heads[0, :4], (20.0, 20.0, 40.0, 60.0), atol=1e-4)
    np.testing.assert_allclose(heads[0, 4], 0.90, atol=1e-6)


def test_head_detection_returns_absolute_boxes_when_the_graph_asks_for_frame_size():
    """Exports carrying an orig_target_sizes input emit pixels, not [0, 1]."""
    inst = _detectors(
        {"DEIMv2Wholebody49Head": _fake_session(extra_inputs=("orig_target_sizes",))}
    )
    pred = _label_xyxy_score([(HEAD_CLASS_ID, 20.0, 20.0, 40.0, 60.0, 0.90)])
    inst._run_onnx_binding = lambda *a, **k: [pred]  # type: ignore[method-assign]

    heads = inst.detect_head_bboxes_wholebody49(torch.zeros(3, 200, 100))

    np.testing.assert_allclose(heads[0, :4], (20.0, 20.0, 40.0, 60.0), atol=1e-4)


def test_head_detection_sorts_by_score_and_drops_slivers():
    inst = _detectors({"DEIMv2Wholebody49Head": _fake_session()})
    pred = _label_xyxy_score(
        [
            (HEAD_CLASS_ID, 0.10, 0.10, 0.30, 0.30, 0.60),
            (HEAD_CLASS_ID, 0.50, 0.50, 0.70, 0.70, 0.95),
            # 1 px wide after scaling: useless as a crop source.
            (HEAD_CLASS_ID, 0.80, 0.10, 0.81, 0.40, 0.99),
        ]
    )
    inst._run_onnx_binding = lambda *a, **k: [pred]  # type: ignore[method-assign]

    heads = inst.detect_head_bboxes_wholebody49(torch.zeros(3, 100, 100))

    assert heads.shape == (2, 5)
    assert heads[0, 4] > heads[1, 4]


def test_head_detection_returns_an_empty_array_when_the_model_is_missing():
    """The caller falls back to an estimated head box, so this must not be None."""
    inst = _detectors()
    inst.models_processor.load_model = lambda name: None  # type: ignore[assignment]

    heads = inst.detect_head_bboxes_wholebody49(torch.zeros(3, 100, 100))

    assert isinstance(heads, np.ndarray)
    assert heads.shape == (0, 5)


def test_head_detection_registers_the_dependency_for_unloading():
    """unload_models() iterates active_landmark_models; a dependency left out of it
    would stay resident in VRAM after landmark detection is switched off."""
    inst = _detectors()
    inst._run_onnx_binding = lambda *a, **k: [  # type: ignore[method-assign]
        _label_xyxy_score([(HEAD_CLASS_ID, 0.1, 0.1, 0.3, 0.3, 0.9)])
    ]

    def load(name):
        inst.models_processor.models[name] = _fake_session()
        return inst.models_processor.models[name]

    inst.models_processor.load_model = load  # type: ignore[assignment]

    inst.detect_head_bboxes_wholebody49(torch.zeros(3, 100, 100))

    assert "DEIMv2Wholebody49Head" in inst.active_landmark_models


# --------------------------------------------------------------------------------
# Head box -> face box matching
# --------------------------------------------------------------------------------
def test_the_containing_head_box_wins_even_though_its_iou_is_poor():
    """A correct head box swallows the face box while being much larger, so its IoU is
    low. Ranking by IoU would prefer the wrong, similarly-sized box."""
    face = np.array([40.0, 50.0, 60.0, 80.0])
    heads = np.array(
        [
            # Correct head: contains the face fully, IoU only 600/2500 = 0.24.
            [30.0, 30.0, 80.0, 80.0, 0.9],
            # Same size as the face and heavily overlapping, IoU 0.47 -- but it clips
            # the face box, so containment is only 0.5.
            [50.0, 50.0, 70.0, 80.0, 0.9],
        ],
        dtype=np.float32,
    )

    picked = FaceLandmarkDetectors._head_bbox_for_face(heads, face)

    np.testing.assert_allclose(picked, (30.0, 30.0, 80.0, 80.0))


def test_a_head_box_belonging_to_another_face_is_rejected():
    face = np.array([40.0, 50.0, 60.0, 80.0])
    heads = np.array([[300.0, 300.0, 400.0, 400.0, 0.99]], dtype=np.float32)

    picked = FaceLandmarkDetectors._head_bbox_for_face(heads, face)

    # Falls through to the estimate, which stays centred on the face.
    assert picked[0] < face[0] and picked[2] > face[2]
    assert 300.0 not in picked


@pytest.mark.parametrize("heads", [None, np.empty((0, 5), dtype=np.float32)])
def test_the_estimated_head_box_is_a_square_shifted_up(heads):
    """The documented degraded path: 1.5 * max(w, h) about the face centre, shifted up
    by 0.08 * h because a head reaches much further above a face box than below it."""
    face = np.array([40.0, 50.0, 60.0, 100.0])  # w=20, h=50 -> side 75

    picked = FaceLandmarkDetectors._head_bbox_for_face(heads, face)

    width = picked[2] - picked[0]
    height = picked[3] - picked[1]
    np.testing.assert_allclose(width, 75.0, atol=1e-4)
    np.testing.assert_allclose(height, 75.0, atol=1e-4)
    np.testing.assert_allclose((picked[0] + picked[2]) / 2, 50.0, atol=1e-4)
    # Face centre y is 75; the crop centre sits 0.08 * 50 = 4 px above it.
    np.testing.assert_allclose((picked[1] + picked[3]) / 2, 71.0, atol=1e-4)


def test_a_degenerate_face_box_still_produces_a_head_box():
    """A zero-area bbox makes containment undefined; the estimate must not divide by
    zero or return an empty box."""
    picked = FaceLandmarkDetectors._head_bbox_for_face(
        np.array([[0.0, 0.0, 100.0, 100.0, 0.9]], dtype=np.float32),
        np.array([50.0, 50.0, 50.0, 50.0]),
    )

    assert np.all(np.isfinite(picked))


# --------------------------------------------------------------------------------
# HRFFA
# --------------------------------------------------------------------------------
def test_hrffa_scales_its_normalised_output_to_crop_pixels():
    inst = _detectors()
    calls: list = []
    _stub_crop(inst, calls)

    # Graph emits normalised coords; 0.25 -> 64 px, 0.5 -> 128 px at 256.
    normalised = np.full((1, 68, 2), 0.25, dtype=np.float32)
    normalised[0, 30] = (0.5, 0.5)  # ibug68 nose tip
    inst._run_onnx_binding = lambda name, inputs, outputs: [normalised]  # type: ignore[method-assign]

    kpss_5, kpss, scores = inst.detect_face_landmark_hrffa(
        torch.zeros(3, 512, 512),
        np.array([0, 0, 100, 100]),
        None,
        head_bboxes=np.empty((0, 5), dtype=np.float32),
    )

    assert kpss.shape == (68, 2)
    np.testing.assert_allclose(kpss[0], (64.0, 64.0), atol=1e-4)
    np.testing.assert_allclose(kpss[30], (128.0, 128.0), atol=1e-4)
    # ibug68 nose tip feeds slot 2 of the 5-point set.
    np.testing.assert_allclose(kpss_5[2], (128.0, 128.0), atol=1e-4)
    # No confidence head is surfaced, so it must pass through unthresholded.
    assert len(scores) == 0


def test_hrffa_only_binds_the_points_output():
    """vis_logits is left unbound so ORT prunes that branch."""
    inst = _detectors()
    _stub_crop(inst, [])
    seen: list = []

    def binding(name, inputs, outputs):
        seen.append((name, sorted(inputs), list(outputs)))
        return [np.zeros((1, 68, 2), dtype=np.float32)]

    inst._run_onnx_binding = binding  # type: ignore[method-assign]

    inst.detect_face_landmark_hrffa(
        torch.zeros(3, 512, 512),
        np.array([0, 0, 100, 100]),
        None,
        head_bboxes=np.empty((0, 5), dtype=np.float32),
    )

    assert seen == [("FaceLandmarkHRFFA", ["images"], ["points"])]


def test_hrffa_crops_the_head_box_axis_aligned_at_1_1_and_ignores_from_points():
    """scale 1.1 is the training pad of 0.05 on each side. det_kpss must arrive as None
    so _prepare_crop pins the roll angle to 0 -- HRFFA is trained on axis-aligned head
    crops and is robust through a full 360 deg of roll on its own."""
    inst = _detectors()
    calls: list = []
    _stub_crop(inst, calls)
    inst._run_onnx_binding = lambda *a, **k: [np.zeros((1, 68, 2), dtype=np.float32)]  # type: ignore[method-assign]
    head = np.array([[10.0, 20.0, 110.0, 140.0, 0.9]], dtype=np.float32)

    inst.detect_face_landmark_hrffa(
        torch.zeros(3, 512, 512),
        np.array([30, 50, 90, 110]),
        np.array([[40, 60], [80, 70], [60, 80], [45, 95], [75, 95]]),
        from_points=True,
        head_bboxes=head,
    )

    assert len(calls) == 1
    assert calls[0]["det_kpss"] is None
    assert calls[0]["from_points"] is False
    assert calls[0]["target_size"] == 256
    assert calls[0]["scale"] == 1.1
    assert calls[0]["warp_mode"] is None
    # It crops the HEAD box, not the face box it was handed.
    np.testing.assert_allclose(calls[0]["bbox"], (10.0, 20.0, 110.0, 140.0))


def test_hrffa_runs_the_head_detector_itself_when_none_is_supplied():
    """Callers without a per-face loop do not precompute head boxes."""
    inst = _detectors()
    _stub_crop(inst, [])
    inst._run_onnx_binding = lambda *a, **k: [np.zeros((1, 68, 2), dtype=np.float32)]  # type: ignore[method-assign]
    calls: list = []

    def fake_heads(img, score_threshold=None):
        calls.append(score_threshold)
        return np.array([[0.0, 0.0, 200.0, 200.0, 0.9]], dtype=np.float32)

    inst.detect_head_bboxes_wholebody49 = fake_heads  # type: ignore[method-assign]

    inst.detect_face_landmark_hrffa(
        torch.zeros(3, 512, 512), np.array([50, 50, 150, 150]), None
    )

    assert calls == [None]


def test_hrffa_survives_a_failed_crop():
    inst = _detectors()
    inst._prepare_crop = lambda *a, **k: (None, None, None)  # type: ignore[method-assign]

    result = inst.detect_face_landmark_hrffa(
        torch.zeros(3, 512, 512),
        np.array([0, 0, 100, 100]),
        None,
        head_bboxes=np.empty((0, 5), dtype=np.float32),
    )

    assert result == ([], [], [])
