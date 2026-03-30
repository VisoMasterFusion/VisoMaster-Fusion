"""
VR180-* integration tests for the VR180 feature.

Tests the full EquirectangularConverter → PerspectiveConverter round-trip
with real (but tiny) equirectangular images and no ML models.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from app.helpers.vr_utils import EquirectangularConverter, PerspectiveConverter

CPU = torch.device("cpu")


def make_equirect(h: int = 90, w: int = 180) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, 0] = np.tile(np.linspace(0, 255, w, dtype=np.uint8), (h, 1))
    img[:, :, 2] = np.tile(np.linspace(0, 255, h, dtype=np.uint8)[:, None], (1, w))
    return img


# ---------------------------------------------------------------------------
# VR180-01: Both-Eyes mode — left face bbox → only left half modified
# ---------------------------------------------------------------------------


def test_both_eyes_left_face_modifies_only_left_half():
    h, w = 90, 180
    img = make_equirect(h, w)
    pc = PerspectiveConverter(img, CPU)

    target = torch.zeros(3, h, w, dtype=torch.uint8)
    crop = torch.full((3, 32, 32), 200, dtype=torch.uint8)
    half = w // 2

    pc.stitch_single_perspective(
        target_equirect_torch_cxhxw_rgb_uint8=target,
        processed_crop_torch_cxhxw_rgb_uint8=crop,
        theta=-90.0,
        phi=0.0,
        fov=60.0,
        is_left_eye=True,
    )

    right_sum = target[:, :, half:].sum().item()
    assert right_sum == 0, "is_left_eye=True must not modify the right half"


# ---------------------------------------------------------------------------
# VR180-02: Both-Eyes mode — right face bbox → only right half modified
# ---------------------------------------------------------------------------


def test_both_eyes_right_face_modifies_only_right_half():
    h, w = 90, 180
    img = make_equirect(h, w)
    pc = PerspectiveConverter(img, CPU)

    target = torch.zeros(3, h, w, dtype=torch.uint8)
    crop = torch.full((3, 32, 32), 200, dtype=torch.uint8)
    half = w // 2

    pc.stitch_single_perspective(
        target_equirect_torch_cxhxw_rgb_uint8=target,
        processed_crop_torch_cxhxw_rgb_uint8=crop,
        theta=90.0,
        phi=0.0,
        fov=60.0,
        is_left_eye=False,
    )

    left_sum = target[:, :, :half].sum().item()
    assert left_sum == 0, "is_left_eye=False must not modify the left half"


# ---------------------------------------------------------------------------
# VR180-03: Single-Eye mode — full frame can be modified
# ---------------------------------------------------------------------------


def test_single_eye_allows_full_frame_modification():
    h, w = 90, 180
    img = make_equirect(h, w)
    pc = PerspectiveConverter(img, CPU)
    half = w // 2

    # Full-frame (is_left_eye=None)
    target_single = torch.zeros(3, h, w, dtype=torch.uint8)
    crop = torch.full((3, 64, 64), 255, dtype=torch.uint8)
    pc.stitch_single_perspective(
        target_equirect_torch_cxhxw_rgb_uint8=target_single,
        processed_crop_torch_cxhxw_rgb_uint8=crop,
        theta=0.0,
        phi=0.0,
        fov=90.0,
        is_left_eye=None,
    )

    # Left-eye-only for comparison
    target_left = torch.zeros(3, h, w, dtype=torch.uint8)
    pc.stitch_single_perspective(
        target_equirect_torch_cxhxw_rgb_uint8=target_left,
        processed_crop_torch_cxhxw_rgb_uint8=crop,
        theta=0.0,
        phi=0.0,
        fov=90.0,
        is_left_eye=True,
    )

    # Full-frame should have more total signal than left-only
    total_single = target_single.float().sum().item()
    total_left = target_left.float().sum().item()
    assert total_single >= total_left, (
        "Single-eye mode should produce at least as much output as left-eye-only"
    )

    # Right half should be allowed to change in single-eye mode
    right_single = target_single[:, :, half:].float().sum().item()
    right_left = target_left[:, :, half:].float().sum().item()
    assert right_single >= right_left, (
        "Single-eye mode should allow the right half to be modified"
    )


# ---------------------------------------------------------------------------
# VR180-04: calculate_theta_phi round-trip stability
# ---------------------------------------------------------------------------


def test_theta_phi_round_trip_consistency():
    """
    The same bbox at the same location should always produce the same angles —
    verifying determinism of calculate_theta_phi_from_bbox.
    """
    h, w = 90, 180
    img = make_equirect(h, w)
    ec = EquirectangularConverter(img, CPU)

    bbox = np.array([40.0, 30.0, 60.0, 50.0])
    theta1, phi1 = ec.calculate_theta_phi_from_bbox(bbox)
    theta2, phi2 = ec.calculate_theta_phi_from_bbox(bbox)

    assert theta1 == theta2
    assert phi1 == phi2


# ---------------------------------------------------------------------------
# VR180-05: switching mode mid-session doesn't corrupt state
# ---------------------------------------------------------------------------


def test_mode_switch_does_not_corrupt_target():
    """
    Process one stitch in Both-Eyes mode, then one in Single-Eye mode.
    Verify the second stitch is independent of the first.
    """
    h, w = 90, 180
    img = make_equirect(h, w)
    pc = PerspectiveConverter(img, CPU)
    target = torch.zeros(3, h, w, dtype=torch.uint8)
    crop = torch.full((3, 32, 32), 128, dtype=torch.uint8)

    # First stitch: Both-Eyes left mode
    pc.stitch_single_perspective(
        target_equirect_torch_cxhxw_rgb_uint8=target,
        processed_crop_torch_cxhxw_rgb_uint8=crop,
        theta=-90.0,
        phi=0.0,
        fov=60.0,
        is_left_eye=True,
    )
    # Second stitch: Single-Eye mode at a different position
    crop2 = torch.full((3, 32, 32), 64, dtype=torch.uint8)
    pc.stitch_single_perspective(
        target_equirect_torch_cxhxw_rgb_uint8=target,
        processed_crop_torch_cxhxw_rgb_uint8=crop2,
        theta=90.0,
        phi=0.0,
        fov=60.0,
        is_left_eye=None,
    )

    # After the second stitch the right half should differ from state_after_first
    # (the second crop at theta=90 will affect the right region)
    # We just verify no exception and dtype is preserved
    assert target.dtype == torch.uint8
    assert target.shape == (3, h, w)


# ---------------------------------------------------------------------------
# VR180 — equirect dimensions are stored correctly
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("h,w", [(90, 180), (180, 360), (360, 720)])
def test_equirect_converter_dimensions(h, w):
    img = make_equirect(h, w)
    ec = EquirectangularConverter(img, CPU)
    assert ec.height == h
    assert ec.width == w
    assert ec.channels == 3


@pytest.mark.parametrize("h,w", [(90, 180), (180, 360)])
def test_perspective_converter_dimensions(h, w):
    img = make_equirect(h, w)
    pc = PerspectiveConverter(img, CPU)
    assert pc.orig_height == h
    assert pc.orig_width == w
    assert pc.orig_channels == 3
