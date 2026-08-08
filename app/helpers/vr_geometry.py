"""Frame geometry description for the VR face-swap path.

The VR pipeline works in four steps: turn a bounding box in the flat video frame
into a *direction* on the viewing sphere, render an undistorted perspective crop
around that direction, swap the face inside the crop, then project the crop back
into the frame.  Three of those four steps need to agree on exactly how a pixel
in the frame maps to a direction.

That mapping used to be hardcoded for VR180 side-by-side equirectangular
content, where each eye covers 180° horizontally and 180° vertically.  That is a
convenient special case: it makes the *whole* stereo frame a valid 360°×180°
equirect, with the left eye occupying longitudes -180…0° and the right eye
0…180°, which is why the projection code could treat a stereo pair as a single
sphere.  Nothing else worked.

``VRGeometry`` makes the mapping explicit so that

* per-eye coverage is configurable (:data:`COVERAGE_MIN_DEG` …
  :data:`COVERAGE_MAX_DEG`, with 180° reproducing the old behaviour), and
* fisheye-encoded content (200° lenses and friends) works alongside
  equirectangular.

Both projections share one angle convention, so the rest of the pipeline is
unaffected by which one is in use:

``theta``
    Frame-level longitude in degrees, spanning ``±lon_span_deg / 2``.  In
    Both-Eyes mode the left eye occupies the negative half and the right eye the
    positive half, so a ``theta`` names one eye's view unambiguously.
``phi``
    Elevation in degrees, positive up.

Note that ``theta`` is *bookkeeping* only.  The eye is tracked separately, as an
explicit ``is_left_eye``, all the way down into the projection maths — it cannot
be recovered from a direction, because both eyes look the same way and therefore
produce identical rays.  (An earlier design gave each eye its own slice of a
widened longitude range; that silently breaks above 180° coverage, where the
slices run past ±180° and ``atan2`` folds them back on top of each other.)  All
ray maths is consequently done in the eye's own camera frame: ``x`` = optical
axis, ``y`` = right, ``z`` = up, matching the existing converters.

Pixel positions use the endpoint-inclusive ``size - 1`` convention that the
original converters used, so the default geometry reproduces their output.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

PROJECTION_EQUIRECTANGULAR = "Equirectangular"
PROJECTION_FISHEYE = "Fisheye (equidistant)"
PROJECTIONS = (PROJECTION_EQUIRECTANGULAR, PROJECTION_FISHEYE)

EYE_MODE_BOTH = "Both Eyes"
EYE_MODE_SINGLE = "Single Eye"

COVERAGE_MIN_DEG = 90.0
COVERAGE_MAX_DEG = 360.0
COVERAGE_DEFAULT_DEG = 180.0

# A latitude/longitude parameterisation of the sphere stops being injective past
# ±90° elevation, so the vertical span an *equirectangular* frame can describe is
# capped at 180°.  Content whose vertical coverage genuinely exceeds that has to
# be stored as fisheye, which has no poles.
LAT_SPAN_MAX_DEG = 180.0

# Control keys this module reads.  Kept here so the worker and the UI layout
# cannot drift apart.
COVERAGE_CONTROL_KEY = "VRCoverageSlider"
PROJECTION_CONTROL_KEY = "VRProjectionSelection"
EYE_MODE_CONTROL_KEY = "VR180EyeModeSelection"


@dataclass(frozen=True)
class VRGeometry:
    """How pixels in a VR video frame map to directions on the viewing sphere.

    Frozen (and therefore hashable) so it can be used directly as an
    ``lru_cache`` key by the projection converters.
    """

    frame_height: int
    frame_width: int
    both_eyes: bool = True
    projection: str = PROJECTION_EQUIRECTANGULAR
    coverage_deg: float = COVERAGE_DEFAULT_DEG

    @classmethod
    def from_control(
        cls, control: dict, frame_height: int, frame_width: int
    ) -> VRGeometry:
        """Build the geometry described by the UI control dict."""
        try:
            coverage = float(control.get(COVERAGE_CONTROL_KEY, COVERAGE_DEFAULT_DEG))
        except (TypeError, ValueError):
            # A slider mid-edit can briefly hold a non-numeric value.
            coverage = COVERAGE_DEFAULT_DEG
        coverage = min(max(coverage, COVERAGE_MIN_DEG), COVERAGE_MAX_DEG)

        projection = str(
            control.get(PROJECTION_CONTROL_KEY, PROJECTION_EQUIRECTANGULAR)
        )
        if projection not in PROJECTIONS:
            projection = PROJECTION_EQUIRECTANGULAR

        return cls(
            frame_height=int(frame_height),
            frame_width=int(frame_width),
            both_eyes=(
                control.get(EYE_MODE_CONTROL_KEY, EYE_MODE_BOTH) != EYE_MODE_SINGLE
            ),
            projection=projection,
            coverage_deg=coverage,
        )

    # ------------------------------------------------------------------
    # Derived sizes and spans
    # ------------------------------------------------------------------

    @property
    def is_fisheye(self) -> bool:
        return self.projection == PROJECTION_FISHEYE

    @property
    def eye_width(self) -> int:
        """Pixel width of a single eye's view."""
        return self.frame_width // 2 if self.both_eyes else self.frame_width

    @property
    def lon_span_deg(self) -> float:
        """Longitude range spanned by the *whole frame*."""
        return self.coverage_deg * (2.0 if self.both_eyes else 1.0)

    @property
    def lat_span_deg(self) -> float:
        """Latitude range spanned by the frame.

        Derived from the horizontal coverage on the assumption of square angular
        pixels, which is what both an equirectangular remap and an equidistant
        fisheye produce.  A standard 2:1 VR180 frame has square eyes, so this
        gives back 180°.
        """
        span = self.coverage_deg * self.frame_height / max(self.eye_width, 1)
        return min(span, LAT_SPAN_MAX_DEG)

    @property
    def angular_pixel_span(self) -> float:
        """Pixel width one eye's angular coverage maps onto.

        Uses the endpoint-inclusive ``frame_width - 1`` convention of the original
        converters, so a standard 360-wide stereo frame gives each eye 179.5 px
        for its 180° — exactly reproducing the previous sampling.
        """
        return (self.frame_width - 1) / (2.0 if self.both_eyes else 1.0)

    @property
    def deg_per_pixel(self) -> float:
        """Angular size of one pixel.

        Exact everywhere for an equidistant fisheye; for equirectangular it is
        the horizontal scale at the equator only (longitude compresses towards
        the poles — see :meth:`bbox_angular_size_deg`).
        """
        return self.coverage_deg / max(self.angular_pixel_span, 1e-9)

    @property
    def reproduces_legacy_vr180(self) -> bool:
        """True when this geometry is the old hardcoded VR180 special case.

        Used to keep the longitude seam wrapping (and therefore bit-identical
        output) on the default settings path.
        """
        return (
            not self.is_fisheye
            and self.both_eyes
            and abs(self.coverage_deg - COVERAGE_DEFAULT_DEG) < 1e-6
        )

    @property
    def wraps_longitude(self) -> bool:
        """Whether a single view closes into a full circle.

        Only then may a sample that runs off one horizontal edge be wrapped to
        the other; otherwise it has to be clamped, because the pixels there
        belong to a different eye (or to nothing at all).
        """
        return not self.is_fisheye and abs(self.coverage_deg - 360.0) < 1e-6

    # ------------------------------------------------------------------
    # Eyes
    # ------------------------------------------------------------------

    def eye_of_pixel(self, x: float) -> bool | None:
        """Which eye a frame x-coordinate belongs to.

        Returns True for the left eye, False for the right, and None when the
        frame holds a single view.
        """
        if not self.both_eyes:
            return None
        return x < self.eye_width

    def eye_of_theta(self, theta: float) -> bool | None:
        """Which eye a frame-level longitude belongs to.

        The left eye owns the negative half of the frame's longitude range, which
        is also the left half of its pixels, so this agrees with
        :meth:`eye_of_pixel`.
        """
        if not self.both_eyes:
            return None
        return theta < 0.0

    def eye_x_offset(self, is_left_eye: bool | None) -> int:
        """First frame column belonging to an eye."""
        if not self.both_eyes or is_left_eye is None:
            return 0
        return 0 if is_left_eye else self.eye_width

    def eye_x_bounds(self, is_left_eye: bool | None) -> tuple[float, float]:
        """Inclusive first/last frame column belonging to an eye."""
        if not self.both_eyes or is_left_eye is None:
            return 0.0, float(self.frame_width - 1)
        x0 = self.eye_x_offset(is_left_eye)
        return float(x0), float(x0 + self.eye_width - 1)

    # ``is_left_eye=None`` means "treat the frame as a single view", which is how
    # the stitcher has always signalled single-eye mode.  It therefore overrides
    # the eye layout rather than being merely unknown.

    def eye_coverage_deg(self, is_left_eye: bool | None) -> float:
        """Horizontal degrees the addressed view spans.

        ``None`` collapses the frame to a single view, so it spans the frame's
        whole longitude range rather than one eye's coverage.  On a stereo frame
        at default settings that is 360°, which is how the original code treated
        single-eye stitching.
        """
        return self.lon_span_deg if is_left_eye is None else self.coverage_deg

    def eye_pixel_span(self, is_left_eye: bool | None) -> float:
        """Pixel width the addressed view's angular coverage maps onto."""
        if is_left_eye is None:
            return float(self.frame_width - 1)
        return self.angular_pixel_span

    def eye_pixel_center(self, is_left_eye: bool | None) -> float:
        """Frame column of an eye's optical axis."""
        span = self.eye_pixel_span(is_left_eye)
        if not self.both_eyes or is_left_eye is None:
            return span / 2.0
        return span / 2.0 if is_left_eye else span * 1.5

    def eye_center_theta(self, is_left_eye: bool | None) -> float:
        """Frame-level longitude of an eye's optical axis."""
        if not self.both_eyes or is_left_eye is None:
            return 0.0
        return -self.coverage_deg / 2.0 if is_left_eye else self.coverage_deg / 2.0

    def fisheye_pixels_per_radian(self, is_left_eye: bool | None) -> float:
        """Equidistant fisheye scale factor.

        Radius from the optical axis is proportional to the angle away from it,
        with the full coverage circle's diameter spanning the eye's pixel width.
        """
        half_angle_rad = math.radians(self.eye_coverage_deg(is_left_eye) / 2.0)
        return (self.eye_pixel_span(is_left_eye) / 2.0) / max(half_angle_rad, 1e-9)

    def partner_theta(self, theta: float, is_left_eye: bool | None) -> float:
        """The same direction as seen by the other eye.

        Both eyes look the same way, so the partner sits at the identical
        eye-relative angle — one ``coverage_deg`` away in frame-level longitude.
        """
        if not self.both_eyes or is_left_eye is None:
            return theta
        return theta + self.coverage_deg if is_left_eye else theta - self.coverage_deg

    def rotation_theta(self, theta: float, is_left_eye: bool | None) -> float:
        """Longitude to build a crop's rotation matrix from.

        All projections are handled in the eye's own camera frame, so the eye's
        optical axis is subtracted from the frame-level longitude.  For a
        single-view frame the axis is 0° and this is the identity.
        """
        return theta - self.eye_center_theta(is_left_eye)

    # ------------------------------------------------------------------
    # Bounding boxes
    # ------------------------------------------------------------------

    def pixel_to_theta_phi(self, x: float, y: float) -> tuple[float, float]:
        """Direction of a single frame pixel, as (theta, phi) in degrees."""
        is_left_eye = self.eye_of_pixel(x)
        cx = self.eye_pixel_center(is_left_eye)
        cy = (self.frame_height - 1) / 2.0
        du = x - cx
        dv = y - cy

        if not self.is_fisheye:
            # Latitude/longitude grid, measured from this eye's optical axis.
            theta_local = (
                du
                / max(self.eye_pixel_span(is_left_eye) / 2.0, 1e-9)
                * (self.eye_coverage_deg(is_left_eye) / 2.0)
            )
            phi = (
                -dv
                / max((self.frame_height - 1) / 2.0, 1e-9)
                * (self.lat_span_deg / 2.0)
            )
            return theta_local + self.eye_center_theta(is_left_eye), phi

        radius = math.hypot(du, dv)
        if radius < 1e-9:
            theta_local, phi = 0.0, 0.0
        else:
            alpha = radius / self.fisheye_pixels_per_radian(is_left_eye)
            sin_a = math.sin(alpha)
            ray_x = math.cos(alpha)
            ray_y = sin_a * (du / radius)
            ray_z = sin_a * (-dv / radius)
            theta_local = math.degrees(math.atan2(ray_y, ray_x))
            phi = math.degrees(math.asin(max(-1.0, min(1.0, ray_z))))
        return theta_local + self.eye_center_theta(is_left_eye), phi

    def bbox_to_theta_phi(self, bbox) -> tuple[float, float]:
        """Direction of a bounding box's centre, as (theta, phi) in degrees."""
        x_center = (float(bbox[0]) + float(bbox[2])) / 2.0
        y_center = (float(bbox[1]) + float(bbox[3])) / 2.0
        return self.pixel_to_theta_phi(x_center, y_center)

    def bbox_angular_size_deg(self, bbox, phi_deg: float) -> tuple[float, float]:
        """Angular width and height a bounding box subtends, in degrees.

        Used to pick a per-face crop FOV, so it wants the *true* angular size
        rather than a naive pixel-to-degree scaling.
        """
        width_px = float(bbox[2]) - float(bbox[0])
        height_px = float(bbox[3]) - float(bbox[1])

        if self.is_fisheye:
            # An equidistant fisheye has a uniform angular scale in every
            # direction, so no latitude correction applies.
            deg_per_px = self.deg_per_pixel
            return width_px * deg_per_px, height_px * deg_per_px

        width_deg = width_px * self.deg_per_pixel
        height_deg = height_px / max(self.frame_height - 1, 1) * self.lat_span_deg

        # Equirectangular projection compresses longitude towards the poles, so
        # a face at high elevation spans more degrees than its pixel width
        # suggests: true angular width ≈ pixel width / cos(latitude).
        #
        # cos_phi is clamped to 0.35 (≈70° latitude) to cap the correction at
        # ~2.9×.  An earlier clamp of 0.1 allowed up to 10× amplification near
        # the poles, which produced extreme FOVs and garbage landmarks ("eye
        # sideways, mouth near ear") for faces at the top or bottom of frame.
        # The secondary cap of 2× the raw width keeps the corrected value
        # continuous with the uncorrected one for near-pole detections.
        cos_phi = math.cos(math.radians(phi_deg))
        width_deg_corrected = width_deg / max(cos_phi, 0.35)
        width_deg = min(width_deg_corrected, width_deg * 2.0)
        return width_deg, height_deg

    # ------------------------------------------------------------------
    # Tiled detection
    # ------------------------------------------------------------------

    def tile_grid(self, tile_fov_deg: float = 90.0) -> list[tuple[float, float]]:
        """(theta, phi) centres for the tiled-detection sweep.

        Tiles are spaced at two thirds of their FOV so neighbours overlap by a
        third, which is enough that a face never falls only on a seam.  Bands
        are placed at the equator, mid-latitude, and near the vertical extremes;
        the outermost bands get fewer tiles because a band of given angular
        radius needs proportionally fewer views to cover it.

        On the default VR180 geometry this yields the same 24 tiles the previous
        hardcoded grid used (the near-pole tiles are spaced slightly differently,
        which the generous 90° tile FOV absorbs).
        """
        lon_span = self.lon_span_deg
        spacing = max(tile_fov_deg * 2.0 / 3.0, 1.0)

        def _band_thetas(count: int) -> list[float]:
            step = lon_span / count
            return [-lon_span / 2.0 + step * (i + 0.5) for i in range(count)]

        main_count = max(1, math.ceil(lon_span / spacing))
        main_thetas = _band_thetas(main_count)
        outer_thetas = _band_thetas(max(1, main_count // 2))

        lat_half = self.lat_span_deg / 2.0
        grid: list[tuple[float, float]] = []
        for phi_fraction in (0.0, 40.0 / 90.0, -40.0 / 90.0):
            phi = lat_half * phi_fraction
            grid.extend((theta, phi) for theta in main_thetas)
        for phi_fraction in (70.0 / 90.0, -70.0 / 90.0):
            phi = lat_half * phi_fraction
            grid.extend((theta, phi) for theta in outer_thetas)
        return grid


# ----------------------------------------------------------------------
# Torch mappings shared by the forward (equirect→perspective) and inverse
# (perspective→equirect) converters, so the two can never disagree.
# ----------------------------------------------------------------------


def rays_to_frame_pixels(
    rays: torch.Tensor, geometry: VRGeometry, is_left_eye: bool | None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map unit direction vectors to frame pixel coordinates.

    ``rays`` is ``(..., 3)``, expressed in the eye's own camera frame — i.e. the
    frame :meth:`VRGeometry.rotation_theta` rotates into.

    Returns ``(x_px, y_px)`` in frame pixel space.  Coordinates outside the eye's
    columns are possible and left for the caller to wrap or clamp.
    """
    eye_center_x = geometry.eye_pixel_center(is_left_eye)
    half_eye_px = geometry.eye_pixel_span(is_left_eye) / 2.0
    center_y = (geometry.frame_height - 1) / 2.0

    if not geometry.is_fisheye:
        longitude = torch.atan2(rays[..., 1], rays[..., 0])
        # Clamp before asin: float rounding at the poles can push the component
        # a hair outside [-1, 1] and produce NaN.
        latitude = torch.asin(torch.clamp(rays[..., 2], -1.0, 1.0))
        half_lon_rad = math.radians(geometry.eye_coverage_deg(is_left_eye) / 2.0)
        half_lat_rad = math.radians(geometry.lat_span_deg / 2.0)
        x_px = (longitude / half_lon_rad) * half_eye_px + eye_center_x
        y_px = (-latitude / half_lat_rad) * center_y + center_y
        return x_px, y_px

    # Equidistant fisheye: distance from the optical axis in the image is
    # proportional to the ray's angle away from that axis.
    alpha = torch.acos(torch.clamp(rays[..., 0], -1.0, 1.0))
    radius = alpha * geometry.fisheye_pixels_per_radian(is_left_eye)
    # Length of the ray's projection onto the image plane, used to get its
    # direction there.  Floored so a ray exactly along the axis (where the
    # direction is undefined but the radius is 0) does not divide by zero.
    in_plane = torch.sqrt(rays[..., 1] ** 2 + rays[..., 2] ** 2).clamp(min=1e-12)
    x_px = eye_center_x + radius * (rays[..., 1] / in_plane)
    y_px = center_y - radius * (rays[..., 2] / in_plane)
    return x_px, y_px


def frame_pixel_rays(
    geometry: VRGeometry,
    is_left_eye: bool | None,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Unit direction for every pixel of the frame — the inverse mapping.

    Returns ``(rays, valid)`` where ``rays`` is ``(H, W, 3)`` and ``valid`` is
    ``(H, W)`` bool, or None when every pixel of the frame carries a usable
    direction for this eye.  Directions are expressed in the eye's own camera
    frame, i.e. the frame :meth:`VRGeometry.rotation_theta` rotates out of.
    """
    height, width = geometry.frame_height, geometry.frame_width

    columns = torch.arange(width, dtype=torch.float32).unsqueeze(0)
    rows = torch.arange(height, dtype=torch.float32).unsqueeze(1)
    du = columns - geometry.eye_pixel_center(is_left_eye)
    dv = rows - (height - 1) / 2.0
    half_eye_px = geometry.eye_pixel_span(is_left_eye) / 2.0
    half_frame_px = max((height - 1) / 2.0, 1e-9)

    eye_coverage = geometry.eye_coverage_deg(is_left_eye)

    if not geometry.is_fisheye:
        lon_rad = torch.deg2rad(du / half_eye_px * (eye_coverage / 2.0))
        lat_rad = torch.deg2rad(-dv / half_frame_px * (geometry.lat_span_deg / 2.0))
        cos_lat = torch.cos(lat_rad)
        rays = torch.stack(
            torch.broadcast_tensors(
                cos_lat * torch.cos(lon_rad),
                cos_lat * torch.sin(lon_rad),
                torch.broadcast_to(torch.sin(lat_rad), (height, width)),
            ),
            dim=2,
        )
        if is_left_eye is None or not geometry.both_eyes:
            # One view spanning the whole frame — every pixel belongs to it.
            return rays.to(device), None
        # The other eye's columns look the same way but through a different
        # optical centre, so they must never be written from this eye's crop.
        # A hair of tolerance keeps the boundary column itself included.
        valid = (du.abs() <= half_eye_px + 1e-4).expand(height, width)
        return rays.to(device), valid.to(device)

    radius = torch.sqrt(du**2 + dv**2)
    alpha = radius / geometry.fisheye_pixels_per_radian(is_left_eye)

    safe_radius = radius.clamp(min=1e-12)
    sin_a = torch.sin(alpha)
    rays = torch.stack(
        torch.broadcast_tensors(
            torch.cos(alpha),
            sin_a * (du / safe_radius),
            sin_a * (-dv / safe_radius),
        ),
        dim=2,
    )

    # Outside the lens circle there is no image data, and in Both-Eyes mode the
    # other eye's half belongs to a different optical axis entirely.  Both are
    # marked invalid so the caller can reject them rather than sampling garbage.
    valid = alpha <= math.radians(eye_coverage / 2.0)
    if geometry.both_eyes and is_left_eye is not None:
        x_min, x_max = geometry.eye_x_bounds(is_left_eye)
        valid = valid & (columns >= x_min) & (columns <= x_max)
    valid = valid.expand(height, width)

    return rays.to(device), valid.to(device)
