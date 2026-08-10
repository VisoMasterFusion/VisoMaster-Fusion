"""VRPROJ-* integration tests for configurable VR coverage and projection.

These exercise the real EquirectangularConverter → PerspectiveConverter path with
tiny synthetic frames and no ML models.

The property under test is *localisation*: a face at a known frame position must
land in the centre of the extracted perspective crop, and the swapped crop must be
stitched back onto that same position.  If the forward and inverse projections
disagree, swapped faces appear offset — which is the failure mode a hardcoded
180°/equirectangular assumption produces on 200° or fisheye footage.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from app.helpers.vr_geometry import (
    PROJECTION_EQUIRECTANGULAR,
    PROJECTION_FISHEYE,
    VRGeometry,
)
from app.helpers.vr_utils import EquirectangularConverter, PerspectiveConverter

CPU = torch.device("cpu")

# Large enough that a few pixels of tolerance is a tight bound, small enough to
# stay fast on CPU.
FRAME_H, FRAME_W = 256, 512
CROP = 64

PROJECTIONS = [PROJECTION_EQUIRECTANGULAR, PROJECTION_FISHEYE]
COVERAGES = [130.0, 180.0, 200.0]


def geometry_for(projection: str, coverage: float, both_eyes: bool = True):
    return VRGeometry(
        frame_height=FRAME_H,
        frame_width=FRAME_W,
        both_eyes=both_eyes,
        projection=projection,
        coverage_deg=coverage,
    )


def frame_with_marker(x: int, y: int, radius: int = 5) -> np.ndarray:
    """A mid-grey frame with one bright square, so the marker is unambiguous."""
    img = np.full((FRAME_H, FRAME_W, 3), 40, dtype=np.uint8)
    img[y - radius : y + radius + 1, x - radius : x + radius + 1] = 255
    return img


def bright_centroid(
    chw_uint8: torch.Tensor, threshold: float = 160.0
) -> tuple[float, float]:
    """(x, y) centroid of the bright pixels of a CHW tensor.

    A centroid rather than argmax: the marker is a saturated plateau, so argmax
    would return its top-left corner and report a spurious offset.
    """
    luma = chw_uint8.float().mean(dim=0)
    rows, columns = torch.nonzero(luma > threshold, as_tuple=True)
    assert len(rows) > 0, "no bright pixels found at all"
    return float(columns.float().mean()), float(rows.float().mean())


def marker_positions(geometry: VRGeometry) -> list[tuple[int, int, bool]]:
    """Well-inside-the-view sample points for each eye, as (x, y, is_left_eye).

    Kept away from the edges so a 40° crop around the point stays inside the
    view — the test is about localisation, not edge clamping.
    """
    points: list[tuple[int, int, bool]] = []
    for is_left_eye in (True, False):
        cx = geometry.eye_pixel_center(is_left_eye)
        cy = (FRAME_H - 1) / 2.0
        span = geometry.eye_pixel_span(is_left_eye)
        for dx_fraction, dy_fraction in [(0.0, 0.0), (0.25, -0.2), (-0.3, 0.25)]:
            points.append(
                (
                    round(cx + dx_fraction * span / 2.0),
                    round(cy + dy_fraction * (FRAME_H - 1) / 2.0),
                    is_left_eye,
                )
            )
    return points


# ---------------------------------------------------------------------------
# VRPROJ-01: a marker at (x, y) lands at the centre of the crop aimed at it
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("projection", PROJECTIONS)
@pytest.mark.parametrize("coverage", COVERAGES)
def test_crop_is_centred_on_the_direction_it_was_asked_for(projection, coverage):
    geometry = geometry_for(projection, coverage)
    converter = EquirectangularConverter(
        frame_with_marker(FRAME_W // 4, FRAME_H // 2), CPU, geometry
    )

    for x, y, is_left_eye in marker_positions(geometry):
        img = frame_with_marker(x, y)
        converter.equirect_tensor_cxhxw_rgb_uint8.copy_(
            torch.from_numpy(img).permute(2, 0, 1)
        )
        converter.e2p_instance._img_float = None

        theta, phi = converter.calculate_theta_phi_from_bbox(
            np.array([x - 5, y - 5, x + 5, y + 5], dtype=np.float32)
        )
        assert geometry.eye_of_pixel(x) is is_left_eye, "test point in the wrong eye"

        crop = converter.get_perspective_crop(
            40.0, theta, phi, CROP, CROP, is_left_eye=is_left_eye
        )
        found_x, found_y = bright_centroid(crop)
        center = (CROP - 1) / 2.0
        assert abs(found_x - center) <= 3, (
            f"{projection} @{coverage}° marker at ({x},{y}) landed at x={found_x}, "
            f"expected ~{center}"
        )
        assert abs(found_y - center) <= 3, (
            f"{projection} @{coverage}° marker at ({x},{y}) landed at y={found_y}, "
            f"expected ~{center}"
        )


# ---------------------------------------------------------------------------
# VRPROJ-02: the swapped crop is stitched back onto the position it came from
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("projection", PROJECTIONS)
@pytest.mark.parametrize("coverage", COVERAGES)
def test_stitch_lands_on_the_direction_it_came_from(projection, coverage):
    geometry = geometry_for(projection, coverage)
    stitcher = PerspectiveConverter(
        np.zeros((FRAME_H, FRAME_W, 3), np.uint8), CPU, geometry
    )

    for x, y, is_left_eye in marker_positions(geometry):
        theta, phi = geometry.pixel_to_theta_phi(x, y)
        target = torch.zeros(3, FRAME_H, FRAME_W, dtype=torch.uint8)
        stitcher.stitch_single_perspective(
            target_equirect_torch_cxhxw_rgb_uint8=target,
            processed_crop_torch_cxhxw_rgb_uint8=torch.full(
                (3, CROP, CROP), 255, dtype=torch.uint8
            ),
            theta=theta,
            phi=phi,
            fov=40.0,
            is_left_eye=is_left_eye,
        )

        written = target.float().mean(dim=0)
        assert written[y, x] > 128, (
            f"{projection} @{coverage}°: nothing stitched at ({x},{y})"
        )

        # The written region must be centred on the requested position, not merely
        # touch it — an offset projection would still clip the point.
        rows, columns = torch.nonzero(written > 32, as_tuple=True)
        assert len(rows) > 0
        assert abs(float(columns.float().mean()) - x) <= 4
        assert abs(float(rows.float().mean()) - y) <= 4


# ---------------------------------------------------------------------------
# VRPROJ-03: stereo isolation — one eye's swap never touches the other eye
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("projection", PROJECTIONS)
@pytest.mark.parametrize("coverage", COVERAGES)
def test_stitch_never_crosses_into_the_other_eye(projection, coverage):
    geometry = geometry_for(projection, coverage)
    stitcher = PerspectiveConverter(
        np.zeros((FRAME_H, FRAME_W, 3), np.uint8), CPU, geometry
    )
    half = FRAME_W // 2

    for is_left_eye in (True, False):
        # Deliberately near the eye's inner edge, where bleed would show up.
        cx = geometry.eye_pixel_center(is_left_eye)
        span = geometry.eye_pixel_span(is_left_eye)
        x = round(cx + (0.8 if is_left_eye else -0.8) * span / 2.0)
        theta, phi = geometry.pixel_to_theta_phi(x, FRAME_H // 2)

        target = torch.zeros(3, FRAME_H, FRAME_W, dtype=torch.uint8)
        stitcher.stitch_single_perspective(
            target_equirect_torch_cxhxw_rgb_uint8=target,
            processed_crop_torch_cxhxw_rgb_uint8=torch.full(
                (3, CROP, CROP), 255, dtype=torch.uint8
            ),
            theta=theta,
            phi=phi,
            fov=60.0,
            is_left_eye=is_left_eye,
        )
        other_half = target[:, :, half:] if is_left_eye else target[:, :, :half]
        assert int(other_half.sum().item()) == 0, (
            f"{projection} @{coverage}°: {'left' if is_left_eye else 'right'} eye "
            "swap bled into the other eye"
        )


# ---------------------------------------------------------------------------
# VRPROJ-04: an extract → stitch round trip restores the original pixels
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("projection", PROJECTIONS)
@pytest.mark.parametrize("coverage", COVERAGES)
def test_extract_then_stitch_preserves_the_marker_position(projection, coverage):
    """Pull a crop out, put it straight back, and the marker must not move.

    This is the end-to-end version of the property: it fails if the forward and
    inverse projections disagree even slightly.
    """
    geometry = geometry_for(projection, coverage)
    x, y, is_left_eye = marker_positions(geometry)[0]
    img = frame_with_marker(x, y, radius=8)

    extractor = EquirectangularConverter(img, CPU, geometry)
    stitcher = PerspectiveConverter(img, CPU, geometry)

    theta, phi = geometry.pixel_to_theta_phi(x, y)
    # A larger crop resolution keeps resampling loss from dominating.
    crop = extractor.get_perspective_crop(
        40.0, theta, phi, 128, 128, is_left_eye=is_left_eye
    )

    target = torch.zeros(3, FRAME_H, FRAME_W, dtype=torch.uint8)
    stitcher.stitch_single_perspective(
        target_equirect_torch_cxhxw_rgb_uint8=target,
        processed_crop_torch_cxhxw_rgb_uint8=crop,
        theta=theta,
        phi=phi,
        fov=40.0,
        is_left_eye=is_left_eye,
    )

    restored_x, restored_y = bright_centroid(target)
    assert abs(restored_x - x) <= 3, (
        f"{projection} @{coverage}°: x moved from {x} to {restored_x:.1f}"
    )
    assert abs(restored_y - y) <= 3, (
        f"{projection} @{coverage}°: y moved from {y} to {restored_y:.1f}"
    )


# ---------------------------------------------------------------------------
# VRPROJ-05: the default settings reproduce the previous behaviour
# ---------------------------------------------------------------------------


def test_default_geometry_matches_the_implicit_legacy_geometry():
    """Passing no geometry must behave exactly like the VR180 default."""
    img = frame_with_marker(FRAME_W // 4, FRAME_H // 2, radius=8)
    theta, phi = -90.0, 0.0

    legacy = EquirectangularConverter(img, CPU)  # geometry=None
    explicit = EquirectangularConverter(
        img, CPU, geometry_for(PROJECTION_EQUIRECTANGULAR, 180.0)
    )

    legacy_crop = legacy.get_perspective_crop(60.0, theta, phi, CROP, CROP)
    explicit_crop = explicit.get_perspective_crop(
        60.0, theta, phi, CROP, CROP, is_left_eye=True
    )
    assert torch.equal(legacy_crop, explicit_crop)


def test_single_eye_stitch_still_covers_the_whole_frame():
    """is_left_eye=None keeps meaning 'the frame is one view', as before."""
    geometry = geometry_for(PROJECTION_EQUIRECTANGULAR, 180.0)
    stitcher = PerspectiveConverter(
        np.zeros((FRAME_H, FRAME_W, 3), np.uint8), CPU, geometry
    )
    half = FRAME_W // 2

    single = torch.zeros(3, FRAME_H, FRAME_W, dtype=torch.uint8)
    stitcher.stitch_single_perspective(
        target_equirect_torch_cxhxw_rgb_uint8=single,
        processed_crop_torch_cxhxw_rgb_uint8=torch.full(
            (3, CROP, CROP), 255, dtype=torch.uint8
        ),
        theta=0.0,
        phi=0.0,
        fov=90.0,
        is_left_eye=None,
    )
    # theta=0 is the seam between the eyes when the frame is one 360° view, so a
    # 90° crop there must write to both halves.
    assert int(single[:, :, :half].sum().item()) > 0
    assert int(single[:, :, half:].sum().item()) > 0


# ---------------------------------------------------------------------------
# VRPROJ-06: coverage actually changes the mapping
# ---------------------------------------------------------------------------


def test_coverage_changes_where_a_pixel_points():
    """Otherwise the slider would be silently inert."""
    at_180 = geometry_for(PROJECTION_EQUIRECTANGULAR, 180.0)
    at_200 = geometry_for(PROJECTION_EQUIRECTANGULAR, 200.0)
    x, y = FRAME_W // 8, FRAME_H // 2
    theta_180, _ = at_180.pixel_to_theta_phi(x, y)
    theta_200, _ = at_200.pixel_to_theta_phi(x, y)
    assert theta_180 != pytest.approx(theta_200)


def test_projection_changes_where_a_pixel_points():
    equirect = geometry_for(PROJECTION_EQUIRECTANGULAR, 200.0)
    fisheye = geometry_for(PROJECTION_FISHEYE, 200.0)
    # Off-axis and off-centre, where the two projections genuinely diverge.
    x = int(equirect.eye_pixel_center(True) + 0.6 * equirect.eye_pixel_span(True) / 2)
    y = FRAME_H // 4
    assert equirect.pixel_to_theta_phi(x, y) != pytest.approx(
        fisheye.pixel_to_theta_phi(x, y)
    )
