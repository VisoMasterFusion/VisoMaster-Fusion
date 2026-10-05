import ctypes
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from app.processors.face_mattings import FaceMattings
from app.processors.models_processor import ModelsProcessor


def setup():
    return FaceMattings(
        SimpleNamespace(
            device="cpu",
            device_type="cpu",
            binding_device_id=0,
            models={},
            model_lock=threading.RLock(),
        ),
        MagicMock(),
    )


@pytest.mark.parametrize("mode", ["Protect Target Hair", "Soften Hairline"])
@pytest.mark.parametrize("feather", [0, 6])
def test_hair_inside_opaque_coverage_and_exclusions(mode, feather):
    fm = setup()
    mask = torch.ones(1, 64, 64)
    mask[:, :4] = 0
    mask[:, 20:25, 20:25] = 0
    labels = torch.zeros(64, 64, dtype=torch.long)
    labels[5:32] = 17
    labels[40:] = 1
    fm.run_modnet = lambda _: torch.ones_like(mask)
    fm.run_hair_labels = lambda _: labels
    original = mask.clone()
    out = fm.apply_hair_matting(
        mask,
        torch.zeros(3, 64, 64),
        {
            "HairMattingModeSelection": mode,
            "HairMattingFaceDilationSlider": 0,
            "HairMattingFeatherSlider": feather,
        },
    )
    assert torch.equal(mask, original)
    assert torch.all(out <= mask)
    assert out[0, 48, 48] == 1
    if mode == "Protect Target Hair":
        assert out[0, 10, 10] < 0.05
    else:
        assert out[0, 20, 19] < 1
    assert torch.equal(out[mask == 0], mask[mask == 0])


def test_semantic_face_dilation():
    labels = torch.full((64, 64), 17)
    labels[25:40, 25:40] = 1
    matte = torch.ones(1, 64, 64)
    small = FaceMattings.compute_hair_region(matte, labels, face_dilation_px=0)
    large = FaceMattings.compute_hair_region(matte, labels, face_dilation_px=5)
    assert large.sum() < small.sum()
    assert large[0, 5, 5] == 1


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_invalid_matte_noop(bad):
    fm = setup()
    mask = torch.ones(1, 64, 64)
    fm.run_modnet = lambda _: torch.full_like(mask, bad)
    fm.run_hair_labels = MagicMock()
    assert fm.apply_hair_matting(mask, torch.zeros(3, 64, 64), {}) is mask
    fm.run_hair_labels.assert_not_called()


def test_zero_strength_skips_inference():
    fm = setup()
    fm.run_modnet = MagicMock()
    mask = torch.ones(1, 64, 64)
    assert fm.apply_hair_matting(mask, None, {"HairMattingStrengthSlider": 0}) is mask
    fm.run_modnet.assert_not_called()


def test_missing_model_and_bad_parameters_noop():
    fm = setup()
    mask = torch.ones(1, 64, 64)
    crop = torch.zeros(3, 64, 64)
    fm.models_processor.load_model = lambda _: None
    assert fm.apply_hair_matting(mask, crop, {}) is mask
    assert (
        fm.apply_hair_matting(mask, crop, {"HairMattingStrengthSlider": "bad"}) is mask
    )


def test_empty_hair_region_preserves_coverage():
    fm = setup()
    mask = torch.rand(1, 64, 64)
    fm.run_modnet = lambda _: torch.ones_like(mask)
    fm.run_hair_labels = lambda _: torch.ones(512, 512, dtype=torch.long)
    out = fm.apply_hair_matting(mask, torch.zeros(3, 128, 128), {})
    assert torch.equal(mask, out)


@pytest.mark.parametrize("dtype", [torch.float32, torch.uint8])
@pytest.mark.parametrize("size", [512, 128])
@pytest.mark.parametrize("failure", [False, True])
def test_inference_preserves_shared_source(dtype, size, failure):
    fm = setup()
    backing = torch.full((3, size, size * 2), 128, dtype=dtype)
    source = backing[:, :, ::2]
    original = backing.clone()
    session = MagicMock()
    fm.models_processor.models["MODNet"] = session

    def output(**kw):
        array = (ctypes.c_float * (512 * 512)).from_address(kw["buffer_ptr"])
        for i in range(len(array)):
            array[i] = 0.75

    session.io_binding.return_value.bind_output.side_effect = output

    def run(*_):
        if failure:
            raise RuntimeError("injected")

    fm._run_model_with_lazy_build_check = run
    result = fm.run_modnet(source)
    assert torch.equal(backing, original)
    assert (result is None) == failure
    if not failure:
        assert result.shape == (1, size, size)
        assert torch.allclose(result, torch.full_like(result, 0.75))


def test_shared_model_liveness():
    mp = ModelsProcessor.__new__(ModelsProcessor)
    mp.main_window = SimpleNamespace(
        default_parameters={},
        control={},
        parameters={
            "a": {"HairMattingEnableToggle": True},
            "b": {"HairMattingEnableToggle": False},
        },
    )
    assert mp.is_model_active_in_ui("MODNet") and mp.is_model_active_in_ui("FaceParser")
    mp.main_window.parameters["a"]["HairMattingEnableToggle"] = False
    assert not mp.is_model_active_in_ui("MODNet")
    mp.main_window.parameters["b"]["FaceParserEnableToggle"] = True
    assert mp.is_model_active_in_ui("FaceParser")
    mp.main_window.parameters["a"]["HairMattingEnableToggle"] = True
    assert mp.is_model_active_in_ui("MODNet")


def test_clear_reload_tracking():
    fm = setup()
    fm.models_processor.load_model = lambda _: MagicMock()
    fm.models_processor.unload_model = MagicMock()
    fm._run_model_with_lazy_build_check = MagicMock(
        side_effect=RuntimeError("injected")
    )
    fm.run_modnet(torch.zeros(3, 512, 512))
    assert fm.active_models == {"MODNet"}
    fm.unload_models()
    assert not fm.active_models
    fm.run_modnet(torch.zeros(3, 512, 512))
    assert fm.active_models == {"MODNet"}
