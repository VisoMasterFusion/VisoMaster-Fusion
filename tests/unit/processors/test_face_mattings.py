"""Unit tests for app.processors.face_mattings.

Covers the pure-tensor mask math (compute_hair_region, _soft_blur) and the
no-op guarantees of apply_hair_matting: missing model, zero strength, and
empty hair region must all return the input mask unchanged.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from app.processors.face_mattings import FaceMattings


def _make_mattings(matte: torch.Tensor | None = None) -> FaceMattings:
    """FaceMattings with a stubbed processor and run_modnet replaced."""
    mp = MagicMock()
    mp.device = torch.device("cpu")
    mp.models = {}
    fm = FaceMattings(mp, MagicMock())
    fm.run_modnet = lambda _img: matte  # noqa: E731
    return fm


def _ring_mask(size: int = 512) -> torch.Tensor:
    """(1, size, size) mask that is 1 inside a centered ellipse, 0 outside."""
    yy, xx = torch.meshgrid(
        torch.arange(size, dtype=torch.float32),
        torch.arange(size, dtype=torch.float32),
        indexing="ij",
    )
    c = (size - 1) / 2
    r = ((yy - c) / (0.35 * size)) ** 2 + ((xx - c) / (0.30 * size)) ** 2
    return (r <= 1.0).float().unsqueeze(0)


# ---------------------------------------------------------------------------
# compute_hair_region
# ---------------------------------------------------------------------------


def test_hair_region_is_matte_minus_dilated_face() -> None:
    size = 128
    matte = torch.ones(1, size, size)  # whole portrait is subject
    face = _ring_mask(size)
    hair = FaceMattings.compute_hair_region(
        matte, face, threshold=0.4, face_dilation_px=4
    )
    assert hair.shape == (1, size, size)
    assert hair.min() >= 0.0 and hair.max() <= 1.0
    # Center of the face must never be classified as hair.
    assert hair[0, size // 2, size // 2] == 0.0
    # Corners (matte=1, far from face) must be hair.
    assert hair[0, 2, 2] > 0.9


def test_hair_region_zero_when_matte_below_threshold() -> None:
    size = 64
    matte = torch.full((1, size, size), 0.3)
    face = torch.zeros(1, size, size)
    hair = FaceMattings.compute_hair_region(
        matte, face, threshold=0.4, face_dilation_px=0
    )
    assert float(hair.max()) == 0.0


def test_hair_region_accepts_mismatched_mask_size() -> None:
    matte = torch.ones(1, 100, 100)
    face = torch.zeros(1, 512, 512)
    hair = FaceMattings.compute_hair_region(
        matte, face, threshold=0.0, face_dilation_px=0
    )
    assert hair.shape == (1, 100, 100)


def test_face_dilation_shrinks_hair_region() -> None:
    size = 128
    matte = torch.ones(1, size, size)
    face = _ring_mask(size)
    small = FaceMattings.compute_hair_region(
        matte, face, threshold=0.0, face_dilation_px=2
    )
    large = FaceMattings.compute_hair_region(
        matte, face, threshold=0.0, face_dilation_px=16
    )
    assert float(large.sum()) < float(small.sum())


# ---------------------------------------------------------------------------
# _soft_blur
# ---------------------------------------------------------------------------


def test_soft_blur_preserves_shape_and_range() -> None:
    mask = _ring_mask(64)
    out = FaceMattings._soft_blur(mask, 6)
    assert out.shape == mask.shape
    assert out.min() >= 0.0 and out.max() <= 1.0
    # Blur must soften the hard edge: intermediate values appear.
    assert ((out > 0.05) & (out < 0.95)).any()


def test_soft_blur_zero_feather_is_identity() -> None:
    mask = _ring_mask(64)
    out = FaceMattings._soft_blur(mask, 0)
    assert torch.equal(out, mask)


# ---------------------------------------------------------------------------
# apply_hair_matting no-op guarantees
# ---------------------------------------------------------------------------


def test_apply_returns_input_when_model_missing() -> None:
    fm = _make_mattings(matte=None)
    swap_mask = _ring_mask()
    crop = torch.zeros(3, 512, 512)
    out = fm.apply_hair_matting(swap_mask, crop, {"HairMattingStrengthSlider": 100})
    assert out is swap_mask


def test_apply_returns_input_when_strength_zero() -> None:
    fm = _make_mattings(matte=torch.ones(1, 512, 512))
    swap_mask = _ring_mask()
    crop = torch.zeros(3, 512, 512)
    out = fm.apply_hair_matting(swap_mask, crop, {"HairMattingStrengthSlider": 0})
    assert out is swap_mask


def test_apply_returns_input_when_no_hair_found() -> None:
    # Matte everywhere zero -> hair region empty -> unchanged mask.
    fm = _make_mattings(matte=torch.zeros(1, 512, 512))
    swap_mask = _ring_mask()
    crop = torch.zeros(3, 512, 512)
    out = fm.apply_hair_matting(
        swap_mask, crop, {"HairMattingStrengthSlider": 100}
    )
    assert out is swap_mask


def test_apply_never_raises_on_garbage_parameters() -> None:
    fm = _make_mattings(matte=torch.ones(1, 512, 512))
    swap_mask = _ring_mask()
    crop = torch.zeros(3, 512, 512)
    out = fm.apply_hair_matting(swap_mask, crop, {"HairMattingStrengthSlider": "x"})
    assert out.shape == swap_mask.shape


# ---------------------------------------------------------------------------
# apply_hair_matting modes
# ---------------------------------------------------------------------------


def _full_matte_setup():
    """Matte = 1 everywhere, face = ellipse -> hair = everything but face."""
    fm = _make_mattings(matte=torch.ones(1, 512, 512))
    swap_mask = _ring_mask()
    crop = torch.zeros(3, 512, 512)
    params = {
        "HairMattingStrengthSlider": 100,
        "HairMattingThresholdSlider": 0,
        "HairMattingFaceDilationSlider": 12,
        "HairMattingFeatherSlider": 6,
    }
    return fm, swap_mask, crop, params


def test_protect_mode_carves_hair_out_of_mask() -> None:
    fm, swap_mask, crop, params = _full_matte_setup()
    params["HairMattingModeSelection"] = "Protect Target Hair"
    out = fm.apply_hair_matting(swap_mask, crop, params)
    assert out.shape == swap_mask.shape
    # Far outside the face the original mask was 0; the hair ring around the
    # face (mask ~1 edge outside dilation) must have been pushed toward 0.
    # Compare band just outside the face edge at the top center.
    assert float(out.sum()) < float(swap_mask.sum())


def test_protect_mode_full_strength_zeroes_hair_ring() -> None:
    fm, swap_mask, crop, params = _full_matte_setup()
    params["HairMattingModeSelection"] = "Protect Target Hair"
    params["HairMattingFeatherSlider"] = 0
    out = fm.apply_hair_matting(swap_mask, crop, params)
    # With matte=1 everywhere and strength=1, out = mask * (1 - hair).
    # Just outside the dilated face, hair ~1 -> out ~0 where mask was ~0..1.
    hair = FaceMattings.compute_hair_region(
        torch.ones(1, 512, 512), swap_mask, threshold=0.0, face_dilation_px=12
    )
    ring = (hair > 0.9)
    assert ring.any()
    assert float(out[ring].max()) < 0.05


def test_soften_mode_preserves_coverage() -> None:
    fm, swap_mask, crop, params = _full_matte_setup()
    params["HairMattingModeSelection"] = "Soften Hairline"
    out = fm.apply_hair_matting(swap_mask, crop, params)
    assert out.shape == swap_mask.shape
    # Coverage barely changes: soften only blurs, so total mass stays close.
    assert abs(float(out.sum()) - float(swap_mask.sum())) < 0.05 * float(
        swap_mask.sum()
    )
    # But the mask did change at the hairline.
    assert not torch.allclose(out, swap_mask)


def test_apply_resizes_mismatched_crop_matte() -> None:
    # Crop at a non-512 size (face-scale adjustment) must still return a
    # 512-space mask.
    fm = _make_mattings(matte=torch.ones(1, 538, 538))
    swap_mask = _ring_mask()
    crop = torch.zeros(3, 538, 538)
    out = fm.apply_hair_matting(
        swap_mask,
        crop,
        {
            "HairMattingStrengthSlider": 100,
            "HairMattingModeSelection": "Protect Target Hair",
        },
    )
    assert out.shape == swap_mask.shape
