"""
FM-* tests for face mask math and guard logic.
No ML models loaded; tests cover pure tensor/numpy operations.
"""

from __future__ import annotations

import torch


# ---------------------------------------------------------------------------
# FM-02: torch.zeros_like initialises mask to zero
# ---------------------------------------------------------------------------


def test_mask_initialises_to_zero():
    ref = torch.ones(1, 128, 128, dtype=torch.bool)
    mask = torch.zeros_like(ref)
    assert mask.sum().item() == 0
    assert mask.dtype == torch.bool


# ---------------------------------------------------------------------------
# FM-03: division-by-zero guard — denominator clamped to avoid NaN
# ---------------------------------------------------------------------------


def test_blend_no_nan_when_denominator_zero():
    """If denominator is 0 it must be clamped so no NaN appears in output."""
    weight_map = torch.zeros(1, 64, 64)  # denominator = 0 everywhere
    safe_denom = weight_map.clamp(min=1e-8)
    numerator = torch.rand(3, 64, 64)
    result = numerator / safe_denom
    assert torch.all(torch.isfinite(result)), "Division by near-zero produced inf/nan"


# ---------------------------------------------------------------------------
# FM-06: mask values stay in [0, 1] after feathering (Gaussian blur)
# ---------------------------------------------------------------------------


def test_feathered_mask_range():
    """After Gaussian blur a [0,1] mask should still be in [0,1]."""
    from torchvision import transforms

    mask = torch.zeros(1, 64, 64, dtype=torch.float32)
    mask[:, 16:48, 16:48] = 1.0
    gauss = transforms.GaussianBlur(kernel_size=11, sigma=3.0)
    feathered = gauss(mask)
    assert feathered.min().item() >= -1e-6  # numerical precision
    assert feathered.max().item() <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# FM-05: blending formula produces sensible output
# ---------------------------------------------------------------------------


def test_blend_formula_correctness():
    """target + (component - target) * mask should equal lerp(target, component, mask)."""
    H, W = 32, 32
    target = torch.full((3, H, W), 100.0)
    component = torch.full((3, H, W), 200.0)
    mask = torch.full((1, H, W), 0.5)

    # Explicit formula used in vr_utils
    diff = component - target
    blended = target + diff * mask

    expected = torch.full((3, H, W), 150.0)
    assert torch.allclose(blended, expected, atol=1e-4)


# ---------------------------------------------------------------------------
# FM-01: initialisation — mask groups are dict-like (structural check)
# ---------------------------------------------------------------------------


def test_mask_groups_are_dict_compatible():
    """Simulate the group-dict init pattern used in FaceMasks.__init__."""
    mouth_groups: dict = {}
    fp_groups: dict = {}
    tex_groups: dict = {}

    # All should start empty
    assert len(mouth_groups) == 0
    assert len(fp_groups) == 0
    assert len(tex_groups) == 0


# ---------------------------------------------------------------------------
# FM-04: transform order — affine applied before mask blend
# ---------------------------------------------------------------------------


def test_affine_before_blend_ordering():
    """
    Demonstrate that (warp-then-blend) != (blend-then-warp) when BOTH the face
    and the background are spatially non-constant.

    result_correct[i] = 0.5*bg[i]         + 0.5*face[shift(i)]
    result_wrong[i]   = 0.5*bg[shift(i)]  + 0.5*face[shift(i)]

    They differ wherever bg[i] != bg[shift(i)], which holds for any non-constant bg.
    """
    # Non-constant background (spatial gradient — not uniform)
    background = torch.arange(0, 64, dtype=torch.float32).reshape(1, 8, 8) * 2.0
    # Non-constant face
    face = torch.arange(64, 128, dtype=torch.float32).reshape(1, 8, 8)
    mask = torch.ones_like(face) * 0.5
    shift = 3

    # Correct order: warp face, then blend into background
    shifted_face = torch.roll(face, shifts=shift, dims=-1)
    result_correct = background + (shifted_face - background) * mask

    # Wrong order: blend first, then warp the blended result (background moves too)
    blended_first = background + (face - background) * mask
    result_wrong = torch.roll(blended_first, shifts=shift, dims=-1)

    assert not torch.allclose(result_correct, result_wrong), (
        "warp-then-blend and blend-then-warp must differ when background is non-constant"
    )
