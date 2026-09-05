"""
FM-* tests for face mask math and guard logic.
No ML models loaded; tests cover pure tensor/numpy operations.
"""

from __future__ import annotations

import pytest
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


# ---------------------------------------------------------------------------
# FM-08: Occluder / XSeg profile-collapse bypass
# ---------------------------------------------------------------------------


class MockModelsProcessor:
    def __init__(self):
        self.device = torch.device("cpu")
        self.models = {}


def _make_face_masks():
    from unittest.mock import MagicMock

    from app.processors.face_masks import FaceMasks

    return FaceMasks(MockModelsProcessor(), MagicMock())


def _occluder_returning(fill):
    """Build a run_occluder stub whose output keeps `fill` fraction of the crop."""

    rows = int(round(256 * fill))

    def _stub(img, output):
        output.zero_()
        output.view(256, 256)[:rows, :] = 1.0

    return _stub


def _xseg_returning(fill):
    rows = int(round(256 * fill))

    def _stub(img, output):
        output.zero_()
        output.view(256, 256)[:rows, :] = 1.0

    return _stub


PROFILE_YAW = 55.0
FRONTAL_YAW = 10.0


def test_occluder_bypassed_on_profile_when_mask_collapses():
    """Extreme yaw + collapsed mask == model failure -> bypass, swap still applied."""
    fm = _make_face_masks()
    fm.run_occluder = _occluder_returning(0.05)

    mask = fm.apply_occlusion(torch.zeros((3, 256, 256)), amount=0, yaw_deg=PROFILE_YAW)

    assert mask.shape == (1, 256, 256)
    assert mask.min().item() == 1.0, "bypass must return a fully clear mask"


def test_occluder_respected_when_frontal_face_is_heavily_occluded():
    """
    The regression guard: a hand covering most of a FRONTAL face produces the same
    low face ratio as a profile collapse, but it is a correct prediction. The
    occluder must be honoured, not bypassed.
    """
    fm = _make_face_masks()
    fm.run_occluder = _occluder_returning(0.05)

    mask = fm.apply_occlusion(torch.zeros((3, 256, 256)), amount=0, yaw_deg=FRONTAL_YAW)

    assert mask.min().item() == 0.0, "occlusion must survive on a frontal face"
    # Tolerance covers row quantization in the stub (1 row == 1/256).
    assert abs(mask.mean().item() - 0.05) < 0.005


def test_occluder_respected_on_profile_when_mask_is_plausible():
    """Extreme yaw alone must not trigger a bypass if the mask looks sane."""
    fm = _make_face_masks()
    fm.run_occluder = _occluder_returning(0.75)

    mask = fm.apply_occlusion(torch.zeros((3, 256, 256)), amount=0, yaw_deg=PROFILE_YAW)

    assert abs(mask.mean().item() - 0.75) < 1e-6


def test_occluder_not_bypassed_when_yaw_unknown():
    """Callers that cannot supply a head angle keep the model authoritative."""
    fm = _make_face_masks()
    fm.run_occluder = _occluder_returning(0.05)

    mask = fm.apply_occlusion(torch.zeros((3, 256, 256)), amount=0)

    assert mask.min().item() == 0.0


def test_xseg_bypassed_on_profile_returns_four_independent_zero_masks():
    """
    XSeg returns inverted masks (0 = Face), so an all-zero bypass makes every
    consumer's (1 - mask) a no-op. The four results must not alias each other:
    the pipeline applies in-place mask math to them.
    """
    fm = _make_face_masks()
    fm.run_dfl_xseg = _xseg_returning(0.05)

    masks = fm.apply_dfl_xseg(
        torch.zeros((3, 256, 256)),
        amount=0,
        mouth=torch.zeros((1, 256, 256)),
        parameters={},
        yaw_deg=PROFILE_YAW,
    )

    assert len(masks) == 4
    for m in masks:
        assert m.shape == (1, 256, 256)
        assert m.max().item() == 0.0

    # Aliasing check: mutating one must not disturb the others.
    masks[0].add_(1.0)
    assert [m.max().item() for m in masks[1:]] == [0.0, 0.0, 0.0]


def test_xseg_respected_when_frontal_face_is_heavily_occluded():
    fm = _make_face_masks()
    fm.run_dfl_xseg = _xseg_returning(0.05)

    mask, _, _, _ = fm.apply_dfl_xseg(
        torch.zeros((3, 256, 256)),
        amount=0,
        mouth=torch.zeros((1, 256, 256)),
        parameters={},
        yaw_deg=FRONTAL_YAW,
    )

    # Inverted convention: the 95% non-face region must still be excluded (1 = BG).
    assert mask.max().item() == 1.0
    assert abs(mask.mean().item() - 0.95) < 0.005


def test_xseg_ratio_uses_binarized_area_not_soft_probability_mass():
    """
    The guard must measure thresholded AREA, not soft probability mass. A mask
    covering 45% of the crop at p=0.6 has a soft mean of 0.27 -- under the 0.30
    threshold, so a mean-based check would wrongly call it a collapse -- while its
    binarized area is 0.45 and correctly passes.
    """
    fm = _make_face_masks()

    def _soft(img, output):
        output.zero_()
        # 45% of the crop at p=0.6 -> soft mean 0.27 (would trip a mean-based
        # check) but thresholded area 0.45 (correctly passes).
        output.view(256, 256)[: int(256 * 0.45), :] = 0.6

    fm.run_dfl_xseg = _soft

    mask, _, _, _ = fm.apply_dfl_xseg(
        torch.zeros((3, 256, 256)),
        amount=0,
        mouth=torch.zeros((1, 256, 256)),
        parameters={},
        yaw_deg=PROFILE_YAW,
    )

    # Not bypassed: the face region is inverted to ~0.4, background stays 1.0.
    assert mask.max().item() == 1.0


def test_clip_mask_is_not_silently_overridden():
    """
    ClipText is user-typed intent. A prompt that legitimately matches almost the
    whole crop must still mask it -- no degradation fallback on this path.
    """
    import inspect

    from app.processors.face_masks import FaceMasks

    src = inspect.getsource(FaceMasks.run_CLIPs)
    assert "0.05" not in src, "run_CLIPs must not re-introduce a silent mask override"


def test_soft_oval_mask_zero_radius_protection():
    """Zero radius or feather must not produce inf/NaN (0 * inf at the boundary)."""
    fm = _make_face_masks()

    mask = fm.soft_oval_mask(
        height=64, width=64, center=(32, 32), radius_x=0, radius_y=0
    )
    assert torch.isfinite(mask).all()

    mask_zero_feather = fm.soft_oval_mask(
        height=64, width=64, center=(32, 32), radius_x=10, radius_y=10, feather_radius=0
    )
    assert torch.isfinite(mask_zero_feather).all()


# ---------------------------------------------------------------------------
# FM-08b: UI-tunable profile-safeguard thresholds
# ---------------------------------------------------------------------------


def _occluder_mask(params, yaw=55.0, fill=0.05):
    """Run apply_occlusion with a collapsed mask and the given UI parameters."""
    fm = _make_face_masks()
    rows = int(round(256 * fill))

    def _stub(img, output):
        output.zero_()
        output.view(256, 256)[:rows, :] = 1.0

    fm.run_occluder = _stub
    return fm.apply_occlusion(
        torch.zeros((3, 256, 256)), amount=0, parameters=params, yaw_deg=yaw
    )


def test_seg_guard_defaults_match_module_constants():
    """
    Workspaces saved before these controls existed have no keys, and internal callers
    pass none at all. Both must keep the previous always-on behaviour.
    """
    from app.processors.face_masks import (
        _SEG_GUARD_MIN_ABS_YAW,
        _SEG_GUARD_MIN_FACE_RATIO,
        FaceMasks,
    )

    for params in (None, {}):
        enabled, min_yaw, min_ratio = FaceMasks._seg_guard_settings(params)
        assert enabled is True
        assert min_yaw == _SEG_GUARD_MIN_ABS_YAW
        assert min_ratio == pytest.approx(_SEG_GUARD_MIN_FACE_RATIO)


def test_seg_guard_converts_ui_percentage_to_fraction():
    """The slider is a percentage; the mask mean it is compared against is 0..1."""
    from app.processors.face_masks import FaceMasks

    _, _, min_ratio = FaceMasks._seg_guard_settings({"SegGuardMinFaceRatioSlider": 45})
    assert min_ratio == pytest.approx(0.45)


def test_seg_guard_toggle_off_restores_unguarded_behaviour():
    """With the safeguard off, a collapsed mask must survive as-is."""
    mask = _occluder_mask({"SegGuardEnableToggle": False})
    assert mask.min().item() == 0.0, "occlusion must not be bypassed when off"


def test_seg_guard_min_face_area_zero_disables():
    """A mean can never be below 0, so 0% must behave as off."""
    mask = _occluder_mask({"SegGuardMinFaceRatioSlider": 0})
    assert mask.min().item() == 0.0


def test_seg_guard_respects_raised_min_angle():
    """At 55 deg of yaw, a 90 deg minimum must not fire."""
    assert _occluder_mask({"SegGuardMinYawSlider": 90}, yaw=55.0).min().item() == 0.0
    # ...but the same frame fires once the minimum drops below the actual angle.
    assert _occluder_mask({"SegGuardMinYawSlider": 30}, yaw=55.0).min().item() == 1.0


def test_seg_guard_respects_lowered_min_face_area():
    """
    A mask keeping 20% of the crop is a collapse at the 30% default but plausible at
    a 10% setting, so the lower setting must leave the occluder alone.
    """
    assert (
        _occluder_mask({"SegGuardMinFaceRatioSlider": 10}, fill=0.20).min().item()
        == 0.0
    )
    assert (
        _occluder_mask({"SegGuardMinFaceRatioSlider": 30}, fill=0.20).min().item()
        == 1.0
    )


def test_seg_guard_settings_are_exposed_in_the_ui():
    """
    These constants are uncalibrated, so they must stay reachable from the UI for
    users to tune. Guards against a later refactor quietly dropping the widgets.
    """
    import ast
    import io

    src = io.open("app/ui/widgets/swapper_layout_data.py", encoding="utf-8").read()
    keys = {
        k.value
        for node in ast.walk(ast.parse(src))
        if isinstance(node, ast.Dict)
        for k in node.keys
        if isinstance(k, ast.Constant) and isinstance(k.value, str)
    }
    assert {
        "SegGuardEnableToggle",
        "SegGuardMinYawSlider",
        "SegGuardMinFaceRatioSlider",
    } <= keys
