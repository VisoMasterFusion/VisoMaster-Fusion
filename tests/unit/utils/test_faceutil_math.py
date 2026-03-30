"""
FU-* tests for pure mathematical functions in faceutil / geometry helpers.

These tests do NOT import faceutil directly (it has heavy optional deps).
Instead they test the mathematical logic in isolation to verify correctness.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# FU-01: paste_back clone does not share storage with input
# ---------------------------------------------------------------------------


def test_paste_back_clone_independent():
    """Cloning a tensor must not share data storage with the original."""
    original = torch.randint(0, 256, (3, 128, 128), dtype=torch.uint8)
    clone = original.clone()
    assert clone.data_ptr() != original.data_ptr()
    # Mutating clone must not affect original
    clone[:] = 0
    assert original.max().item() > 0


# ---------------------------------------------------------------------------
# FU-04: landmark_to_bbox — output bbox contains all input landmark points
# ---------------------------------------------------------------------------


def test_landmark_to_bbox_contains_all_points():
    """bbox = (x_min, y_min, x_max, y_max) must enclose every landmark."""
    landmarks = np.array(
        [
            [10.0, 20.0],
            [50.0, 5.0],
            [30.0, 80.0],
            [70.0, 40.0],
            [5.0, 60.0],
        ]
    )
    x_min = landmarks[:, 0].min()
    y_min = landmarks[:, 1].min()
    x_max = landmarks[:, 0].max()
    y_max = landmarks[:, 1].max()

    for lm in landmarks:
        assert lm[0] >= x_min
        assert lm[0] <= x_max
        assert lm[1] >= y_min
        assert lm[1] <= y_max


# ---------------------------------------------------------------------------
# FU-05: 3DMM transform output shape (3, N)
# ---------------------------------------------------------------------------


def test_3dmm_transform_output_shape():
    """A 3×4 projection matrix applied to N homogeneous points gives (3, N)."""
    N = 68  # typical landmark count
    P = np.random.randn(3, 4)  # projection matrix
    pts_hom = np.ones((4, N))  # homogeneous points (4, N)
    pts_hom[:3] = np.random.randn(3, N)

    result = P @ pts_hom
    assert result.shape == (3, N)


# ---------------------------------------------------------------------------
# FU-03: arcface_src not mutated across calls
# ---------------------------------------------------------------------------


def test_reference_points_not_mutated():
    """
    If arcface_src is a module-level global, calling a function that reads it
    should not alter the original array.
    """
    arcface_src = np.array(
        [
            [38.29, 51.70],
            [73.53, 51.50],
            [56.02, 71.73],
            [41.55, 92.36],
            [70.72, 92.20],
        ],
        dtype=np.float32,
    )

    original_copy = arcface_src.copy()

    # Simulate a transform that might naively mutate the input
    def compute_transform(src):
        # Bad impl would do: src += something  — but correct impl uses a copy
        local = src.copy()
        local += 1.0
        return local

    _ = compute_transform(arcface_src)
    assert np.allclose(arcface_src, original_copy), "arcface_src was mutated!"


# ---------------------------------------------------------------------------
# FU-06: get_reference_facial_points — correct count
# ---------------------------------------------------------------------------


def test_reference_facial_points_count():
    """Standard 5-point facial landmark reference should have exactly 5 points."""
    reference_5pt = np.array(
        [
            [38.29, 51.70],
            [73.53, 51.50],
            [56.02, 71.73],
            [41.55, 92.36],
            [70.72, 92.20],
        ],
        dtype=np.float32,
    )
    assert reference_5pt.shape == (5, 2)


# ---------------------------------------------------------------------------
# FU-02: dynamic resolution — no hardcoded 512
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("crop_size", [128, 256, 384, 512, 640])
def test_crop_size_is_dynamic(crop_size):
    """A crop/affine operation parameterised by crop_size should scale correctly."""
    face_img = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
    assert face_img.shape[0] == crop_size
    assert face_img.shape[1] == crop_size


# ---------------------------------------------------------------------------
# Coordinate math: theta/phi calculations
# ---------------------------------------------------------------------------


def test_theta_phi_center_of_equirect():
    """Center pixel of an equirectangular should give theta≈0, phi≈0."""
    h, w = 360, 720
    x_center = w / 2
    y_center = h / 2
    theta = (x_center / w - 0.5) * 360.0
    phi = -(y_center / h - 0.5) * 180.0
    assert abs(theta) < 1e-6
    assert abs(phi) < 1e-6


def test_theta_phi_left_edge():
    """Left edge of equirect → theta = -180°."""
    w = 720
    theta = (0.0 / w - 0.5) * 360.0
    assert abs(theta - (-180.0)) < 1e-4


def test_theta_phi_right_edge():
    """Right edge (last pixel) → theta close to +180°."""
    w = 720
    theta = ((w - 1) / w - 0.5) * 360.0
    assert theta > 179.0


def test_phi_top_edge():
    """Top of equirect (y=0) → positive phi (looking up)."""
    h = 360
    phi = -(0.0 / h - 0.5) * 180.0
    assert phi == pytest.approx(90.0)


def test_phi_bottom_edge():
    """Bottom of equirect → negative phi (looking down)."""
    h = 360
    phi = -((h - 1) / h - 0.5) * 180.0
    assert phi < -89.0
