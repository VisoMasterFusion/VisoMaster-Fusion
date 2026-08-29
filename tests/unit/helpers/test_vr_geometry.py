"""VRGEO-* tests for VRGeometry — the pixel↔direction mapping for VR frames.

The critical properties are:

* the default geometry reproduces the old hardcoded VR180 formulas exactly, so
  existing projects keep rendering the way they did, and
* pixel → direction → pixel is a round trip for every projection and coverage,
  because the forward mapping picks a crop's centre and the inverse mapping
  decides where the swapped crop is pasted back.  If they disagree, faces land
  in the wrong place.
"""

from __future__ import annotations

import math

import pytest
import torch

from app.helpers.vr_geometry import (
    COVERAGE_DEFAULT_DEG,
    COVERAGE_MAX_DEG,
    COVERAGE_MIN_DEG,
    LAT_SPAN_MAX_DEG,
    PROJECTION_EQUIRECTANGULAR,
    PROJECTION_FISHEYE,
    VRGeometry,
    frame_pixel_rays,
    rays_to_frame_pixels,
)

CPU = torch.device("cpu")

# A standard VR180 side-by-side frame: 2:1 overall, so each eye is square.
SBS_H, SBS_W = 180, 360


def sbs(**kwargs) -> VRGeometry:
    return VRGeometry(frame_height=SBS_H, frame_width=SBS_W, **kwargs)


# ---------------------------------------------------------------------------
# VRGEO-01: the default geometry is the old VR180 special case
# ---------------------------------------------------------------------------


def test_default_geometry_is_legacy_vr180():
    geometry = sbs()
    assert geometry.both_eyes is True
    assert geometry.projection == PROJECTION_EQUIRECTANGULAR
    assert geometry.coverage_deg == COVERAGE_DEFAULT_DEG
    assert geometry.reproduces_legacy_vr180 is True
    # The whole frame is a full 360°x180° equirect — the coincidence the original
    # implementation relied on.
    assert geometry.lon_span_deg == pytest.approx(360.0)
    assert geometry.lat_span_deg == pytest.approx(180.0)
    assert geometry.eye_width == SBS_W // 2


def test_default_pixel_to_theta_phi_tracks_the_original_formula():
    """The old bbox formula was theta = (x/W - 0.5)*360, phi = -(y/H - 0.5)*180.

    It normalised by W and H while the projection code it fed normalised by W-1
    and H-1, so the original code's two halves already disagreed with each other.
    The geometry is internally consistent instead (see the round-trip test), which
    necessarily means it cannot reproduce both.  It agrees with the old bbox
    formula to within a pixel — far below detector precision, and the projection
    side, which decides actual sampling, is reproduced exactly.
    """
    geometry = sbs()
    deg_per_px = geometry.deg_per_pixel
    for x, y in [(0, 0), (37, 12), (180, 90), (359, 179), (SBS_W / 2, SBS_H / 2)]:
        theta, phi = geometry.pixel_to_theta_phi(x, y)
        legacy_theta = (x / SBS_W - 0.5) * 360.0
        legacy_phi = -(y / SBS_H - 0.5) * 180.0
        assert abs(theta - legacy_theta) < deg_per_px, f"theta drift at x={x}"
        assert abs(phi - legacy_phi) < deg_per_px, f"phi drift at y={y}"


def test_default_rays_to_pixels_matches_original_formula_exactly():
    """E2P used lon_px = (lon/pi)*cx + cx and lat_px = (-lat/(pi/2))*cy + cy.

    This is the mapping that decides which source pixels a crop samples, so the
    default geometry must reproduce it exactly or existing projects would
    re-render differently.
    """
    geometry = sbs()
    rays = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.6, -0.5, 0.62449980],
        ],
        dtype=torch.float32,
    )
    rays = rays / rays.norm(dim=-1, keepdim=True)
    # is_left_eye=None is how single-view (whole-frame) stitching has always been
    # signalled, and is the case the old formula described.
    x_px, y_px = rays_to_frame_pixels(rays, geometry, is_left_eye=None)

    center_x = (SBS_W - 1) / 2.0
    center_y = (SBS_H - 1) / 2.0
    expected_x = []
    expected_y = []
    for ray in rays:
        lon = math.atan2(float(ray[1]), float(ray[0]))
        lat = math.asin(max(-1.0, min(1.0, float(ray[2]))))
        expected_x.append((lon / math.pi) * center_x + center_x)
        expected_y.append((-lat / (math.pi / 2.0)) * center_y + center_y)

    assert x_px.tolist() == pytest.approx(expected_x, abs=1e-4)
    assert y_px.tolist() == pytest.approx(expected_y, abs=1e-4)


def test_default_both_eyes_pixel_mapping_matches_original_eye_placement():
    """Each eye's 180° maps onto (W-1)/2 px, centred where the old code put it.

    The original code addressed both eyes through one 360° formula over W-1
    pixels, which placed the eye axes at (W-1)/4 and 3(W-1)/4 rather than at the
    centres of the two integer halves.  Keeping that placement is what makes the
    default path render identically.
    """
    geometry = sbs()
    assert geometry.angular_pixel_span == pytest.approx((SBS_W - 1) / 2.0)
    assert geometry.eye_pixel_center(True) == pytest.approx((SBS_W - 1) / 4.0)
    assert geometry.eye_pixel_center(False) == pytest.approx(3 * (SBS_W - 1) / 4.0)

    on_axis = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
    for is_left_eye in (True, False):
        x_px, y_px = rays_to_frame_pixels(on_axis, geometry, is_left_eye)
        # The eye's optical axis is its own longitude, which the old whole-frame
        # formula placed at exactly these columns.
        expected = (SBS_W - 1) / 4.0 if is_left_eye else 3 * (SBS_W - 1) / 4.0
        assert float(x_px[0]) == pytest.approx(expected, abs=1e-4)
        assert float(y_px[0]) == pytest.approx((SBS_H - 1) / 2.0, abs=1e-4)


# ---------------------------------------------------------------------------
# VRGEO-02: spans scale with coverage and eye mode
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "coverage,expected_lon_span",
    [(90.0, 180.0), (130.0, 260.0), (180.0, 360.0), (200.0, 400.0)],
)
def test_both_eyes_lon_span_is_twice_coverage(coverage, expected_lon_span):
    geometry = sbs(coverage_deg=coverage)
    assert geometry.lon_span_deg == pytest.approx(expected_lon_span)


@pytest.mark.parametrize("coverage", [90.0, 130.0, 180.0, 200.0, 360.0])
def test_single_eye_lon_span_equals_coverage(coverage):
    geometry = VRGeometry(
        frame_height=SBS_H, frame_width=SBS_H, both_eyes=False, coverage_deg=coverage
    )
    assert geometry.lon_span_deg == pytest.approx(coverage)
    assert geometry.eye_width == SBS_H


def test_lat_span_is_capped_for_equirectangular():
    """A square eye at 200° coverage would need ±100° latitude, which lat/lon
    cannot express, so the vertical span saturates at 180°."""
    geometry = sbs(coverage_deg=200.0)
    assert geometry.lat_span_deg == pytest.approx(LAT_SPAN_MAX_DEG)


def test_lat_span_follows_aspect_ratio():
    """A frame half as tall as its eyes are wide covers half the vertical angle."""
    geometry = VRGeometry(frame_height=90, frame_width=360, coverage_deg=180.0)
    assert geometry.eye_width == 180
    assert geometry.lat_span_deg == pytest.approx(90.0)


def test_only_a_full_circle_wraps_longitude():
    """A view has to span the whole circle before running off its edge can wrap.

    A 180° eye must clamp instead: the pixels past its inner edge belong to the
    other eye, and sampling them was how the old whole-frame wrap could bleed one
    eye's image into the other's crop.
    """
    assert sbs().wraps_longitude is False
    assert sbs(coverage_deg=200.0).wraps_longitude is False
    assert sbs(projection=PROJECTION_FISHEYE).wraps_longitude is False
    assert (
        VRGeometry(
            frame_height=SBS_H, frame_width=SBS_W, both_eyes=False, coverage_deg=360.0
        ).wraps_longitude
        is True
    )


# ---------------------------------------------------------------------------
# VRGEO-03: eye assignment is consistent between pixel space and angle space
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("coverage", [90.0, 130.0, 180.0, 200.0])
@pytest.mark.parametrize("projection", [PROJECTION_EQUIRECTANGULAR, PROJECTION_FISHEYE])
def test_eye_of_pixel_and_eye_of_theta_agree(coverage, projection):
    geometry = sbs(coverage_deg=coverage, projection=projection)
    for x in (5, 40, 179, 180, 200, 355):
        y = SBS_H // 2
        by_pixel = geometry.eye_of_pixel(x)
        theta, _ = geometry.pixel_to_theta_phi(x, y)
        assert geometry.eye_of_theta(theta) is by_pixel, (
            f"x={x} theta={theta} disagrees for {projection} at {coverage}°"
        )


def test_single_eye_has_no_eye_split():
    geometry = sbs(both_eyes=False)
    assert geometry.eye_of_pixel(10) is None
    assert geometry.eye_of_theta(-42.0) is None
    assert geometry.eye_x_offset(None) == 0
    assert geometry.eye_center_theta(None) == 0.0
    assert geometry.partner_theta(17.0, None) == 17.0
    assert geometry.rotation_theta(17.0, None) == 17.0


@pytest.mark.parametrize("coverage", [90.0, 130.0, 180.0, 200.0])
def test_partner_theta_is_one_coverage_away(coverage):
    geometry = sbs(coverage_deg=coverage)
    left_theta = -coverage / 2.0
    right_theta = coverage / 2.0
    assert geometry.partner_theta(left_theta, True) == pytest.approx(right_theta)
    assert geometry.partner_theta(right_theta, False) == pytest.approx(left_theta)
    # Round trip back to the original eye.
    assert geometry.partner_theta(
        geometry.partner_theta(left_theta, True), False
    ) == pytest.approx(left_theta)


def test_eye_bounds_partition_the_frame():
    geometry = sbs()
    left_min, left_max = geometry.eye_x_bounds(True)
    right_min, right_max = geometry.eye_x_bounds(False)
    assert (left_min, left_max) == (0.0, SBS_W / 2 - 1)
    assert (right_min, right_max) == (SBS_W / 2, SBS_W - 1)


# ---------------------------------------------------------------------------
# VRGEO-04: rotation frame differs per projection
# ---------------------------------------------------------------------------


def test_rotation_is_relative_to_the_eyes_optical_axis():
    """Both projections rotate in the eye's own camera frame.

    Doing this in a widened frame-level longitude range instead would break above
    180° coverage, where the eyes' ranges pass ±180° and atan2 folds them onto
    each other.
    """
    for projection in (PROJECTION_EQUIRECTANGULAR, PROJECTION_FISHEYE):
        geometry = sbs(projection=projection)
        # An on-axis face rotates by 0 regardless of which eye it is in.
        assert geometry.rotation_theta(-90.0, True) == pytest.approx(0.0)
        assert geometry.rotation_theta(90.0, False) == pytest.approx(0.0)
        assert geometry.rotation_theta(-70.0, True) == pytest.approx(20.0)


@pytest.mark.parametrize("coverage", [90.0, 130.0, 180.0, 200.0])
@pytest.mark.parametrize("projection", [PROJECTION_EQUIRECTANGULAR, PROJECTION_FISHEYE])
def test_rotation_theta_stays_representable(coverage, projection):
    """Eye-relative longitude must stay inside the ±180° a direction vector can
    represent, at every coverage.  This is the invariant a widened frame-level
    longitude range violated above 180° coverage."""
    geometry = sbs(coverage_deg=coverage, projection=projection)
    for x in range(0, SBS_W, 7):
        for y in (0, SBS_H // 2, SBS_H - 1):
            theta, _phi = geometry.pixel_to_theta_phi(x, y)
            local = geometry.rotation_theta(theta, geometry.eye_of_pixel(x))
            assert abs(local) < 180.0, f"unrepresentable at x={x} y={y}: {local}"


@pytest.mark.parametrize("coverage", [90.0, 130.0, 180.0, 200.0])
def test_equirect_rotation_theta_stays_within_half_coverage(coverage):
    """For a lat/lon grid the eye's columns map exactly onto ±coverage/2.

    (A fisheye's corner pixels legitimately sit further off-axis than that — they
    are outside the lens circle, and marked invalid rather than clamped.)
    """
    geometry = sbs(coverage_deg=coverage)
    for x in range(0, SBS_W, 7):
        theta, _phi = geometry.pixel_to_theta_phi(x, SBS_H // 2)
        local = geometry.rotation_theta(theta, geometry.eye_of_pixel(x))
        assert abs(local) <= coverage / 2.0 + 1e-6


# ---------------------------------------------------------------------------
# VRGEO-05: fisheye mapping specifics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("coverage", [180.0, 200.0])
@pytest.mark.parametrize("projection", [PROJECTION_EQUIRECTANGULAR, PROJECTION_FISHEYE])
def test_eye_centre_pixel_looks_along_the_optical_axis(coverage, projection):
    geometry = sbs(projection=projection, coverage_deg=coverage)
    for is_left_eye in (True, False):
        cx = geometry.eye_pixel_center(is_left_eye)
        cy = (SBS_H - 1) / 2.0
        theta, phi = geometry.pixel_to_theta_phi(cx, cy)
        assert theta == pytest.approx(geometry.eye_center_theta(is_left_eye))
        assert phi == pytest.approx(0.0)


@pytest.mark.parametrize("coverage", [130.0, 180.0, 200.0])
@pytest.mark.parametrize("projection", [PROJECTION_EQUIRECTANGULAR, PROJECTION_FISHEYE])
def test_eye_edge_is_half_the_coverage_off_axis(coverage, projection):
    """At the horizontal edge of an eye the ray is coverage/2 off its axis."""
    geometry = sbs(projection=projection, coverage_deg=coverage)
    cy = (SBS_H - 1) / 2.0
    cx = geometry.eye_pixel_center(True)
    theta, phi = geometry.pixel_to_theta_phi(
        cx + geometry.eye_pixel_span(True) / 2.0, cy
    )
    off_axis = theta - geometry.eye_center_theta(True)
    assert off_axis == pytest.approx(coverage / 2.0, abs=1e-4)
    assert phi == pytest.approx(0.0)


def test_fisheye_scale_is_linear_in_angle():
    """Equidistant means radius ∝ angle, so half the radius is half the angle."""
    geometry = sbs(projection=PROJECTION_FISHEYE, coverage_deg=200.0)
    cy = (SBS_H - 1) / 2.0
    cx = geometry.eye_pixel_center(True)
    span = geometry.eye_pixel_span(True)
    quarter, _ = geometry.pixel_to_theta_phi(cx + span / 4.0, cy)
    half, _ = geometry.pixel_to_theta_phi(cx + span / 2.0, cy)
    center = geometry.eye_center_theta(True)
    assert (quarter - center) == pytest.approx((half - center) / 2.0, abs=1e-4)


def test_fisheye_marks_outside_the_lens_circle_invalid():
    geometry = sbs(projection=PROJECTION_FISHEYE, coverage_deg=180.0)
    _rays, valid = frame_pixel_rays(geometry, is_left_eye=True, device=CPU)
    assert valid is not None
    # The other eye's half is never writable from this eye's crop.
    assert not bool(valid[:, geometry.eye_width :].any())
    # The eye's own centre is inside the circle.
    assert bool(valid[SBS_H // 2, geometry.eye_width // 2])
    # A 180° circle inscribed in a square eye leaves the corners outside it.
    assert not bool(valid[0, 0])


def test_equirectangular_single_view_marks_every_pixel_usable():
    geometry = sbs(both_eyes=False)
    rays, valid = frame_pixel_rays(geometry, is_left_eye=None, device=CPU)
    assert valid is None
    assert rays.shape == (SBS_H, SBS_W, 3)
    assert torch.allclose(rays.norm(dim=-1), torch.ones(SBS_H, SBS_W), atol=1e-5)


def test_equirectangular_both_eyes_restricts_each_eye_to_its_own_half():
    """The other eye's columns look the same way through a different optical
    centre, so an eye's crop must never be written there."""
    geometry = sbs()
    half = SBS_W // 2
    for is_left_eye in (True, False):
        rays, valid = frame_pixel_rays(geometry, is_left_eye, CPU)
        assert valid is not None
        assert torch.allclose(rays.norm(dim=-1), torch.ones(SBS_H, SBS_W), atol=1e-5)
        own = valid[:, :half] if is_left_eye else valid[:, half:]
        other = valid[:, half:] if is_left_eye else valid[:, :half]
        assert bool(own.all()), "an eye must own every column of its own half"
        assert not bool(other.any()), "an eye must own no column of the other half"


# ---------------------------------------------------------------------------
# VRGEO-06: pixel → ray → pixel round trip, the property that keeps faces
# from being pasted back in the wrong place
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("coverage", [90.0, 130.0, 180.0, 200.0])
@pytest.mark.parametrize("projection", [PROJECTION_EQUIRECTANGULAR, PROJECTION_FISHEYE])
@pytest.mark.parametrize("is_left_eye", [True, False])
def test_pixel_ray_pixel_round_trip(coverage, projection, is_left_eye):
    geometry = sbs(coverage_deg=coverage, projection=projection)
    rays, valid = frame_pixel_rays(geometry, is_left_eye, CPU)
    x_px, y_px = rays_to_frame_pixels(rays, geometry, is_left_eye)

    columns = torch.arange(SBS_W, dtype=torch.float32).unsqueeze(0).expand(SBS_H, SBS_W)
    rows = torch.arange(SBS_H, dtype=torch.float32).unsqueeze(1).expand(SBS_H, SBS_W)

    keep = valid if valid is not None else torch.ones_like(columns, dtype=torch.bool)
    if not geometry.is_fisheye:
        # Skip the exact poles, where longitude is degenerate: every longitude maps
        # to the same point, so the inverse cannot recover the original column.
        keep = keep & (rows > 0) & (rows < SBS_H - 1)

    assert bool(keep.any()), "test would be vacuous — no pixels to check"
    assert torch.allclose(x_px[keep], columns[keep], atol=0.02), (
        f"x round trip drifted for {projection} at {coverage}°"
    )
    assert torch.allclose(y_px[keep], rows[keep], atol=0.02), (
        f"y round trip drifted for {projection} at {coverage}°"
    )


# ---------------------------------------------------------------------------
# VRGEO-07: bounding-box angular size
# ---------------------------------------------------------------------------


def test_equirect_bbox_width_is_corrected_for_latitude():
    geometry = sbs()
    bbox = (100.0, 20.0, 120.0, 40.0)
    at_equator, _ = geometry.bbox_angular_size_deg(bbox, phi_deg=0.0)
    at_latitude, _ = geometry.bbox_angular_size_deg(bbox, phi_deg=60.0)
    # cos(60°) = 0.5, so the correction is 2× — which is also the secondary cap.
    assert at_latitude == pytest.approx(at_equator * 2.0)


def test_equirect_bbox_latitude_correction_is_capped_at_2x():
    geometry = sbs()
    bbox = (100.0, 20.0, 120.0, 40.0)
    at_equator, _ = geometry.bbox_angular_size_deg(bbox, phi_deg=0.0)
    near_pole, _ = geometry.bbox_angular_size_deg(bbox, phi_deg=89.0)
    assert near_pole == pytest.approx(at_equator * 2.0)


def test_fisheye_bbox_size_needs_no_latitude_correction():
    """A fisheye has a uniform angular scale, so elevation must not change the size."""
    geometry = sbs(projection=PROJECTION_FISHEYE, coverage_deg=200.0)
    bbox = (100.0, 20.0, 120.0, 40.0)
    at_equator = geometry.bbox_angular_size_deg(bbox, phi_deg=0.0)
    high_up = geometry.bbox_angular_size_deg(bbox, phi_deg=70.0)
    assert at_equator == high_up
    # 20 px at 200° spread over the eye's angular pixel span.
    assert at_equator[0] == pytest.approx(20.0 * 200.0 / geometry.angular_pixel_span)


def test_bbox_size_scales_with_coverage():
    bbox = (100.0, 20.0, 120.0, 40.0)
    at_180, _ = sbs(coverage_deg=180.0).bbox_angular_size_deg(bbox, 0.0)
    at_200, _ = sbs(coverage_deg=200.0).bbox_angular_size_deg(bbox, 0.0)
    assert at_200 == pytest.approx(at_180 * 200.0 / 180.0)


# ---------------------------------------------------------------------------
# VRGEO-08: tiled-detection grid
# ---------------------------------------------------------------------------


def test_tile_grid_matches_the_previous_hardcoded_count_at_defaults():
    grid = sbs().tile_grid()
    assert len(grid) == 24
    thetas_at_equator = sorted(t for t, p in grid if p == 0.0)
    assert thetas_at_equator == pytest.approx([-150.0, -90.0, -30.0, 30.0, 90.0, 150.0])
    assert sorted({p for _t, p in grid}) == pytest.approx(
        [-70.0, -40.0, 0.0, 40.0, 70.0]
    )


@pytest.mark.parametrize("coverage", [90.0, 130.0, 180.0, 200.0])
@pytest.mark.parametrize("both_eyes", [True, False])
def test_tile_grid_stays_inside_the_covered_sphere(coverage, both_eyes):
    geometry = VRGeometry(
        frame_height=SBS_H,
        frame_width=SBS_W,
        both_eyes=both_eyes,
        coverage_deg=coverage,
    )
    grid = geometry.tile_grid()
    assert grid
    for theta, phi in grid:
        assert abs(theta) <= geometry.lon_span_deg / 2.0 + 1e-6
        assert abs(phi) <= geometry.lat_span_deg / 2.0 + 1e-6


def test_tile_grid_covers_both_eyes():
    geometry = sbs(coverage_deg=200.0)
    grid = geometry.tile_grid()
    assert any(geometry.eye_of_theta(t) is True for t, _p in grid)
    assert any(geometry.eye_of_theta(t) is False for t, _p in grid)


# ---------------------------------------------------------------------------
# VRGEO-09: building the geometry from the UI control dict
# ---------------------------------------------------------------------------


def test_from_control_defaults_to_legacy_vr180_when_keys_are_absent():
    geometry = VRGeometry.from_control({}, frame_height=SBS_H, frame_width=SBS_W)
    assert geometry == sbs()
    assert geometry.reproduces_legacy_vr180 is True


def test_from_control_reads_projection_coverage_and_eye_mode():
    geometry = VRGeometry.from_control(
        {
            "VRProjectionSelection": PROJECTION_FISHEYE,
            "VRCoverageSlider": 200,
            "VR180EyeModeSelection": "Single Eye",
        },
        frame_height=SBS_H,
        frame_width=SBS_W,
    )
    assert geometry.projection == PROJECTION_FISHEYE
    assert geometry.coverage_deg == pytest.approx(200.0)
    assert geometry.both_eyes is False


@pytest.mark.parametrize(
    "raw,expected",
    [
        (0, COVERAGE_MIN_DEG),
        (10, COVERAGE_MIN_DEG),
        (95, 95.0),
        (500, COVERAGE_MAX_DEG),
        ("200", 200.0),
        (False, COVERAGE_MIN_DEG),  # transient value a mid-edit slider can hold
        ("nonsense", COVERAGE_DEFAULT_DEG),
        (None, COVERAGE_DEFAULT_DEG),
    ],
)
def test_from_control_clamps_coverage(raw, expected):
    geometry = VRGeometry.from_control(
        {"VRCoverageSlider": raw}, frame_height=SBS_H, frame_width=SBS_W
    )
    assert geometry.coverage_deg == pytest.approx(expected)


def test_from_control_rejects_unknown_projection():
    geometry = VRGeometry.from_control(
        {"VRProjectionSelection": "Cube Map"},
        frame_height=SBS_H,
        frame_width=SBS_W,
    )
    assert geometry.projection == PROJECTION_EQUIRECTANGULAR


def test_geometry_is_hashable_so_it_can_key_the_projection_caches():
    # The converters use it as an lru_cache / dict key.
    assert hash(sbs()) == hash(sbs())
    assert hash(sbs(coverage_deg=200.0)) != hash(sbs())
    assert {sbs(): 1}[sbs()] == 1
