import threading
from collections import OrderedDict, defaultdict
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from torchvision.transforms import v2

from app.processors.face_detailer import FaceDetailer
from app.processors.face_restorers import FaceRestorers
from app.processors.models_processor import ModelsProcessor


def _setup(boxes=([48, 48, 80, 80],)):
    worker = MagicMock()
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    points = np.array(
        [
            [
                [x0 + 0.3 * (x1 - x0), y0 + 0.35 * (y1 - y0)],
                [x0 + 0.7 * (x1 - x0), y0 + 0.35 * (y1 - y0)],
                [(x0 + x1) / 2, (y0 + y1) / 2],
                [x0 + 0.35 * (x1 - x0), y0 + 0.75 * (y1 - y0)],
                [x0 + 0.65 * (x1 - x0), y0 + 0.75 * (y1 - y0)],
            ]
            for x0, y0, x1, y1 in boxes
        ],
        dtype=np.float32,
    ).reshape(-1, 5, 2)
    worker.run_detect.return_value = (boxes, points, None)
    worker.apply_facerestorer.side_effect = lambda image, *a, **kw: torch.full_like(
        image, 200
    )
    controls = {
        "FaceDetailerEnableToggle": True,
        "FaceDetailerCanvasSizeSlider": 256,
        "FaceDetailerFeatherSlider": 0,
        "FaceDetailerMaskDilationSlider": 0,
        "FaceDetailerColorMatchToggle": False,
    }
    return FaceDetailer(MagicMock(), worker), worker, controls


@pytest.mark.parametrize(
    "override",
    [
        {"FaceDetailerEnableToggle": False},
        {"FaceDetailerBlendSlider": 0},
        {"FaceDetailerMaxFacesSlider": 0},
        {"FaceDetailerMinFaceSizeSlider": 200, "FaceDetailerMaxFaceSizeSlider": 100},
    ],
)
def test_inert_settings_do_not_run_inference(override):
    detailer, worker, control = _setup()
    control.update(override)
    frame = torch.zeros(3, 128, 128, dtype=torch.uint8)
    assert detailer.apply(frame, control) is frame
    worker.run_detect.assert_not_called()
    worker.apply_facerestorer.assert_not_called()


@pytest.mark.parametrize(
    "dtype", [torch.uint8, torch.float16, torch.float32, torch.float64]
)
def test_refinement_preserves_input_and_background(dtype):
    detailer, worker, control = _setup()
    frame = torch.full((3, 128, 128), 50, dtype=dtype)
    original = frame.clone()
    out = detailer.apply(frame, control)
    assert (
        out.shape == frame.shape and out.dtype == dtype and out.device == frame.device
    )
    assert out.is_contiguous()
    torch.testing.assert_close(frame, original)
    assert out[:, 64, 64].min() == 200
    torch.testing.assert_close(out[:, :40], original[:, :40])
    torch.testing.assert_close(out[:, 88:], original[:, 88:])
    kwargs = worker.run_detect.call_args.kwargs
    assert kwargs["max_num"] == 0
    assert kwargs["bypass_bytetrack"] is True
    assert kwargs["control_override"] is control
    args = worker.apply_facerestorer.call_args.args
    assert args[0].shape == (3, 256, 256)
    assert args[1] == "Reference"
    assert worker.apply_facerestorer.call_args.kwargs["slot_id"] == 3
    np.testing.assert_allclose(args[6][2], [128, 128])


def test_limit_applies_to_smallest_eligible_face_not_largest_detection():
    detailer, worker, control = _setup(
        [
            [0, 0, 160, 160],
            [160, 160, 224, 224],
            [48, 48, 80, 80],
        ]
    )
    control["FaceDetailerMaxFacesSlider"] = 1
    out = detailer.apply(torch.zeros(3, 256, 256, dtype=torch.uint8), control)
    assert out[:, 64, 64].min() == 200
    assert out[:, 192, 192].max() == 0
    worker.apply_facerestorer.assert_called_once()


@pytest.mark.parametrize("result", ["missing", "failure", "wrong_shape", "nan"])
def test_failed_restoration_leaves_frame_unchanged(result):
    detailer, worker, control = _setup()
    if result == "missing":
        worker.apply_facerestorer.side_effect = lambda image, *a, **kw: image
    elif result == "failure":
        worker.apply_facerestorer.side_effect = RuntimeError("unavailable")
    elif result == "wrong_shape":
        worker.apply_facerestorer.side_effect = lambda *a, **kw: torch.zeros(
            3, 512, 512
        )
    else:
        worker.apply_facerestorer.side_effect = lambda image, *a, **kw: torch.full_like(
            image, float("nan")
        )
    frame = torch.full((3, 128, 128), 50, dtype=torch.uint8)
    assert detailer.apply(frame, control) is frame
    assert torch.all(frame == 50)


def test_nonfinite_boxes_and_landmarks_are_skipped():
    detailer, worker, control = _setup()
    worker.run_detect.return_value = (
        np.array([[0, 0, np.nan, 40], [48, 48, 80, 80]]),
        np.full((2, 5, 2), np.nan),
        None,
    )
    frame = torch.zeros(3, 128, 128)
    assert detailer.apply(frame, control) is frame
    worker.apply_facerestorer.assert_not_called()


def test_detection_failure_preserves_frame():
    detailer, worker, control = _setup()
    worker.run_detect.side_effect = RuntimeError("unavailable")
    frame = torch.zeros(3, 128, 128)
    assert detailer.apply(frame, control) is frame


def test_face_failure_does_not_prevent_next_face_refinement():
    detailer, worker, control = _setup([[16, 16, 48, 48], [80, 80, 112, 112]])

    def restore(image, *a, **kw):
        if worker.apply_facerestorer.call_count == 1:
            raise RuntimeError("failed first face")
        return torch.full_like(image, 200)

    worker.apply_facerestorer.side_effect = restore
    frame = torch.full((3, 128, 128), 50, dtype=torch.uint8)
    out = detailer.apply(frame, control)
    assert torch.all(out[:, 32, 32] == 50)
    assert torch.all(out[:, 96, 96] == 200)
    assert torch.all(frame == 50)


def test_corner_crop_and_blend_use_source_coordinates():
    detailer, worker, control = _setup([[0, 0, 32, 32]])
    control["FaceDetailerBlendSlider"] = 25
    frame = torch.full((3, 128, 128), 50, dtype=torch.float32)
    out = detailer.apply(frame, control)
    torch.testing.assert_close(out[:, 16, 16], torch.full((3,), 87.5))
    torch.testing.assert_close(out[:, 100, 100], frame[:, 100, 100])
    np.testing.assert_allclose(
        worker.apply_facerestorer.call_args.args[6][2], [51.2, 51.2]
    )


def test_color_matching_matches_region_statistics():
    original = torch.arange(3 * 20 * 20, dtype=torch.float32).reshape(3, 20, 20) / 10
    refined = original * 1.5 + 25
    matched = FaceDetailer._colour_match_region(refined, original, (2, 2, 18, 18))
    torch.testing.assert_close(matched, original, atol=1e-4, rtol=1e-5)


def test_zero_feather_is_hard_mask_and_feather_is_bounded():
    hard = FaceDetailer._build_face_mask(
        128, (48, 48, 80, 80), dilation_canvas=0, feather_canvas=0, device="cpu"
    )
    soft = FaceDetailer._build_face_mask(
        128, (48, 48, 80, 80), dilation_canvas=5, feather_canvas=12, device="cpu"
    )
    assert set(hard.unique().tolist()) == {0, 1}
    assert soft.min() >= 0 and soft.max() <= 1
    assert ((soft > 0) & (soft < 1)).any()
    assert soft[64, 85] > hard[64, 85]


def _restorers():
    processor = MagicMock()
    processor.device = "cpu"
    processor.main_window.control = {}
    processor.is_model_active_in_ui.return_value = False
    return FaceRestorers(processor, MagicMock()), processor


def _select(restorers, name, slot):
    frame = torch.zeros(3, 64, 64)
    # Missing reference points skip image processing but exercise real ownership.
    assert (
        restorers.apply_facerestorer(
            frame, "Reference", name, 100, 0.9, 0.5, slot_id=slot
        )
        is frame
    )


def test_auxiliary_slot_preserves_shared_models_in_both_directions():
    restorers, processor = _restorers()
    _select(restorers, "GPEN-1024", 1)
    _select(restorers, "GPEN-1024", 3)
    _select(restorers, "CodeFormer", 1)
    processor.unload_model.assert_not_called()
    _select(restorers, "VQFR-v2", 3)
    processor.unload_model.assert_called_once_with("GPENBFR1024")
    assert restorers.active_model_slot1 == "CodeFormer"
    assert restorers.active_model_slot2 is None
    processor.unload_model.reset_mock()
    _select(restorers, "CodeFormer", 3)
    processor.unload_model.assert_called_once_with("VQFRv2")
    processor.unload_model.reset_mock()
    restorers.release_extra_slot(3)
    processor.unload_model.assert_not_called()


def test_canonical_names_set_by_ui_are_shared_and_extra_slot_is_released():
    restorers, processor = _restorers()
    restorers.active_model_slot1 = "GPENBFR1024"
    _select(restorers, "GPEN-1024", 3)
    _select(restorers, "CodeFormer", 3)
    processor.unload_model.assert_not_called()
    restorers.release_extra_slot(3)
    processor.unload_model.assert_called_once_with("CodeFormer")
    assert not restorers.is_model_used_by_extra_slots("CodeFormer")


def test_ui_disable_of_pipeline_slot_preserves_detailer_model():
    # Follow the application import order: UI action/layout modules are cyclic.
    from app.ui.main_ui import control_actions

    restorers, processor = _restorers()
    _select(restorers, "GPEN-1024", 1)
    _select(restorers, "GPEN-1024", 3)
    window = SimpleNamespace(
        current_widget_parameters={"FaceRestorerTypeSelection": "GPEN-1024"},
        function_worker=SimpleNamespace(face_restorers=restorers),
        models_processor=processor,
    )
    control_actions.handle_restorer_state_change(
        window, False, "FaceRestorerEnableToggle"
    )
    processor.unload_model.assert_not_called()
    assert restorers.active_model_slot1 is None
    restorers.release_extra_slot(3)
    processor.unload_model.assert_called_once_with("GPENBFR1024")


def test_ui_detailer_disable_updates_liveness_before_release():
    # Follow the application import order: UI action/layout modules are cyclic.
    from app.ui.main_ui import control_actions

    restorers, processor = _restorers()
    window = SimpleNamespace(
        control={"FaceDetailerEnableToggle": True},
        function_worker=SimpleNamespace(face_restorers=restorers),
        models_processor=processor,
        default_parameters={},
        parameters={},
    )
    processor.main_window = window
    processor.is_model_active_in_ui.side_effect = lambda model: (
        ModelsProcessor.is_model_active_in_ui(processor, model)
    )
    _select(restorers, "GPEN-1024", 3)
    control_actions.handle_face_detailer_change(
        window, False, "FaceDetailerEnableToggle"
    )
    assert window.control["FaceDetailerEnableToggle"] is False
    processor.unload_model.assert_called_once_with("GPENBFR1024")
    processor.purge_unused_restorers.assert_called_once_with()


def test_detailer_controls_are_global_settings():
    from app.ui.main_ui import COMMON_LAYOUT_DATA, SETTINGS_LAYOUT_DATA

    assert "Face Detailer" in SETTINGS_LAYOUT_DATA
    assert "Face Detailer" not in COMMON_LAYOUT_DATA
    assert (
        SETTINGS_LAYOUT_DATA["Face Detailer"]["FaceDetailerEnableToggle"]["default"]
        is False
    )


@pytest.mark.parametrize("model", ["GPENBFR1024", "OSDFace", "OSDFaceUNet"])
def test_model_liveness_includes_global_detailer(model):
    processor = ModelsProcessor.__new__(ModelsProcessor)
    processor.main_window = SimpleNamespace(
        control={
            "FaceDetailerEnableToggle": True,
            "FaceDetailerRestorerTypeSelection": "GPEN-1024"
            if model == "GPENBFR1024"
            else "OSDFace",
        },
        parameters={},
        default_parameters={},
    )
    assert processor.is_model_active_in_ui(model)
    processor.main_window.control["FaceDetailerEnableToggle"] = False
    assert not processor.is_model_active_in_ui(model)


@pytest.mark.parametrize(
    "name",
    [
        "GFPGAN-v1.4",
        "GFPGAN-1024",
        "GPEN-256",
        "GPEN-512",
        "GPEN-1024",
        "GPEN-2048",
        "CodeFormer",
        "RestoreFormer++",
        "VQFR-v2",
        "OSDFace",
    ],
)
def test_missing_weights_return_input_instead_of_unwritten_output(name):
    restorers, processor = _restorers()
    processor.load_model.return_value = None
    restorers.run_OSDFace = MagicMock(return_value=False)
    frame = torch.full((3, 512, 512), 75, dtype=torch.uint8)
    result = restorers.apply_facerestorer(frame, "None", name, 100, 0.9, 0.5, slot_id=3)
    assert result is frame
    assert torch.all(frame == 75)


def test_reference_restoration_returns_magnified_canvas_coordinates():
    restorers, processor = _restorers()
    processor.FFHQ_kps = np.array(
        [[192, 192], [320, 192], [256, 256], [208, 320], [304, 320]], dtype=np.float64
    )
    # A 2x transform between reference points and the 512 model crop.
    reference = processor.FFHQ_kps * 2
    restorers.run_GFPGAN = lambda image, output: output.copy_(image)
    frame = torch.full((3, 1024, 1024), 100.0)
    out = restorers.apply_facerestorer(
        frame, "Reference", "GFPGAN-v1.4", 100, 0.9, 0.5, reference, slot_id=3
    )
    assert out.shape == frame.shape
    torch.testing.assert_close(
        out[:, 384:640, 384:640], frame[:, 384:640, 384:640], atol=0.01, rtol=0
    )


@pytest.mark.parametrize("view", ["normal", "mask", "compare"])
@pytest.mark.parametrize("angle", [0, 90])
def test_standard_postpass_uses_display_coordinates_before_overlays(
    monkeypatch, view, angle
):
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
    worker.is_view_face_mask = view == "mask"
    worker.is_view_face_compare = view == "compare"
    worker._find_best_target_match.return_value = (None, {}, 0.0)
    worker.function_worker.run_detect.return_value = (
        np.array([[100, 120, 220, 260]], dtype=np.float32),
        np.array(
            [[[130, 150], [190, 150], [160, 185], [140, 225], [180, 225]]],
            dtype=np.float32,
        ),
        np.zeros((1, 68, 2), dtype=np.float32),
    )
    worker.function_worker.run_recognize_direct.return_value = (np.ones(512), None)
    calls = []

    def detail(image, controls):
        assert image.shape == (3, 256, 384)
        assert not torch.any(image)
        calls.append("detailer")
        return image + 20

    def overlay(image, *a, **kw):
        if view == "normal":
            assert torch.all(image == 20)
        calls.append("overlay")
        return image

    worker.function_worker.apply_face_detailer.side_effect = detail
    monkeypatch.setattr(standard, "draw_bounding_boxes_on_detected_faces", overlay)
    controls = defaultdict(
        bool,
        {
            "FaceDetailerEnableToggle": True,
            "ShowAllDetectedFacesBBoxToggle": True,
            "ManualRotationEnableToggle": bool(angle),
            "ManualRotationAngleSlider": angle,
        },
    )
    frame = torch.zeros(3, 256, 384, dtype=torch.uint8)
    out = standard.StandardProcessor(worker).process_standard_frame(
        frame, controls, threading.Event()
    )
    assert out.shape == frame.shape
    assert calls == (["detailer", "overlay"] if view == "normal" else ["overlay"])
