import threading
from collections import OrderedDict
from functools import lru_cache

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from app.helpers.vr_geometry import VRGeometry, rays_to_frame_pixels

# E2P-CACHE-01: perspective plane grid cache — (FOV, height, width) → (persp_xx, persp_yy, w_len, h_len).
# Stores CPU tensors; callers move to device.  Thread-safe: _PERSP_GRID_CACHE_LOCK guards
# all read-modify-write sequences so concurrent pool workers cannot race on eviction.
_PERSP_GRID_CACHE: OrderedDict = OrderedDict()
_PERSP_GRID_CACHE_MAX = 16
_PERSP_GRID_CACHE_LOCK = threading.Lock()


def clear_persp_grid_cache() -> None:
    """Release all cached perspective sampling grids (CPU tensors only).

    Cheap to clear and rebuild — the meshgrid generation is fast.
    """
    with _PERSP_GRID_CACHE_LOCK:
        _PERSP_GRID_CACHE.clear()


def clear_rotation_matrix_cache() -> None:
    """Release all cached rotation matrices (GPU tensors).

    Heavier to rebuild than the perspective grids because each entry triggers
    cv2.Rodrigues + numpy → GPU transfers, but holds device memory across jobs.
    """
    _get_e2p_rotation_matrices_cached.cache_clear()


def clear_persp_cache() -> None:
    """Backwards-compatible umbrella: clear both grid and rotation caches.

    Called by VideoProcessor.join_and_clear_threads() between jobs to prevent
    accumulation across multiple recording sessions. Callers that only need
    to drop one cache should use the dedicated clear_*_cache helpers.
    """
    clear_persp_grid_cache()
    clear_rotation_matrix_cache()


# E2P-CACHE-02: rotation matrix cache — same pattern as P2E's _get_rotation_matrices_cached.
# Avoids 2× cv2.Rodrigues + 2× numpy→GPU transfers on EVERY GetPerspective call.
# With 24 tiles × multiple faces, this was 48+ redundant Rodrigues calls per detection frame.
@lru_cache(maxsize=1024)
def _get_e2p_rotation_matrices_cached(THETA_deg: float, PHI_deg: float, device_str: str):
    device = torch.device(device_str)
    y_axis_np = np.array([0.0, 1.0, 0.0], np.float32)
    z_axis_np = np.array([0.0, 0.0, 1.0], np.float32)
    R1_np, _ = cv2.Rodrigues(z_axis_np * np.radians(THETA_deg))
    R2_np, _ = cv2.Rodrigues(np.dot(R1_np, y_axis_np) * np.radians(-PHI_deg))
    R1_torch = torch.from_numpy(R1_np).float().to(device)
    R2_torch = torch.from_numpy(R2_np).float().to(device)
    return R1_torch, R2_torch


class Equirectangular:
    def __init__(self, img_tensor_cxhxw_rgb_uint8: torch.Tensor):
        """
        Initializes with an equirectangular image tensor.
        :param img_tensor_cxhxw_rgb_uint8: Torch tensor (C, H, W) in RGB, uint8 format, on GPU.
        """
        if not isinstance(img_tensor_cxhxw_rgb_uint8, torch.Tensor):
            raise ValueError("Input must be a PyTorch tensor.")
        if img_tensor_cxhxw_rgb_uint8.ndim != 3:
            raise ValueError("Input tensor must be 3-dimensional (C, H, W).")

        # VR-MEM-01: store uint8 (1/4 the size of float32) and convert on-the-fly in
        # GetPerspective.  With 6-8 FrameWorkers, storing a full-frame float32 tensor
        # persistently per worker consumed 216 MiB × N workers, causing CUDA OOM.
        self._img_tensor_cxhxw_rgb_uint8 = img_tensor_cxhxw_rgb_uint8
        self.device = img_tensor_cxhxw_rgb_uint8.device
        self._channels, self._height, self._width = self._img_tensor_cxhxw_rgb_uint8.shape
        # VR-PERF-11: per-frame float32 cache — computed once when the uint8 tensor is
        # updated (via copy_() in frame_worker), reused across all GetPerspective calls
        # for that frame, then freed by setting to None at the next copy_() invalidation.
        # This avoids 24+ redundant uint8→float32 conversions per frame (one per tile/crop)
        # that were the dominant VR processing bottleneck after the VR-MEM-01 change.
        self._img_float: torch.Tensor | None = None

    def GetPerspective(self, FOV: float, THETA: float, PHI: float, height: int, width: int,
                       geometry: "VRGeometry | None" = None,
                       is_left_eye: "bool | None" = None) -> torch.Tensor:
        #
        # THETA is left/right angle, PHI is up/down angle, both in degree
        #
        # geometry describes how the source frame's pixels map to directions, and
        # is_left_eye says which eye THETA addresses (None = the frame is one view).
        # Passing no geometry keeps the historical assumption that the whole frame is
        # a 360°x180° equirect, i.e. VR180 side-by-side.
        #
        # Returns: Perspective crop as Torch tensor (C, H, W) in RGB, uint8 format, on GPU.

        equ_h = self._height
        equ_w = self._width

        if geometry is None:
            geometry = VRGeometry(frame_height=equ_h, frame_width=equ_w)

        # E2P-CACHE-01: perspective plane grid — depends only on FOV and output size,
        # NOT on THETA/PHI. Cached AS GPU TENSORS keyed by (FOV, h, w, device): a
        # CPU-resident cache forced a host→device upload of the grid (W*H*float32
        # ≈ a few MB) on every perspective extraction, multiplied by every face
        # tile every frame on every pool worker — cumulative PCIe traffic was
        # in the GB/s range and each transfer's sync was spin-waiting on CUDA 13.
        # Thread-safe via _PERSP_GRID_CACHE_LOCK.
        cache_key = (FOV, height, width, str(self.device))
        with _PERSP_GRID_CACHE_LOCK:
            _cached_grid = _PERSP_GRID_CACHE.get(cache_key)
            if _cached_grid is not None:
                _PERSP_GRID_CACHE.move_to_end(cache_key)

        if _cached_grid is not None:
            persp_xx, persp_yy = _cached_grid
        else:
            wFOV = FOV
            hFOV = float(height) / float(width) * wFOV
            w_len = torch.tan(torch.deg2rad(torch.tensor(wFOV / 2.0, device=self.device)))
            h_len = torch.tan(torch.deg2rad(torch.tensor(hFOV / 2.0, device=self.device)))

            persp_x_coords = torch.linspace(-w_len, w_len, width, device=self.device, dtype=torch.float32)
            persp_y_coords = torch.linspace(-h_len, h_len, height, device=self.device, dtype=torch.float32)
            persp_yy, persp_xx = torch.meshgrid(persp_y_coords, persp_x_coords, indexing='ij')

            with _PERSP_GRID_CACHE_LOCK:
                if cache_key not in _PERSP_GRID_CACHE:
                    if len(_PERSP_GRID_CACHE) >= _PERSP_GRID_CACHE_MAX:
                        _PERSP_GRID_CACHE.popitem(last=False)
                    _PERSP_GRID_CACHE[cache_key] = (persp_xx, persp_yy)

        # Points in 3D space on the perspective image plane (camera looking along X-axis)
        x_3d = torch.ones_like(persp_xx)
        y_3d = persp_xx
        z_3d = -persp_yy  # Negative because image y is top-to-bottom, z is up in 3D

        # Normalize to unit vectors
        D = torch.sqrt(x_3d**2 + y_3d**2 + z_3d**2)
        xyz_persp_norm = torch.stack((x_3d/D, y_3d/D, z_3d/D), dim=2)  # H, W, 3

        # E2P-CACHE-02: rotation matrices cached by (THETA, PHI, device).
        # Avoids 2× cv2.Rodrigues + numpy→GPU on every call (was 48+ calls per tile-detect frame).
        # Rotation happens in the addressed eye's own camera frame, so THETA has the
        # eye's optical axis subtracted from it.  Doing this in a frame-level sphere
        # instead would break above 180° coverage, where the two eyes' longitude
        # ranges pass ±180° and atan2 folds them onto each other.
        R1_torch, R2_torch = _get_e2p_rotation_matrices_cached(
            float(geometry.rotation_theta(float(THETA), is_left_eye)),
            float(PHI),
            str(self.device),
        )

        # Rotate the 3D points
        # (H, W, 3) -> (H*W, 3) -> (3, H*W) for matmul
        xyz_flat = xyz_persp_norm.reshape(-1, 3).T
        # Apply rotations: R = R2 @ R1
        # Rotated_xyz = R @ xyz_persp_norm (if xyz_persp_norm is column vectors)
        # Here, we transform points from perspective camera space to world space, then to equirectangular.
        # The original code implies rotations to align the perspective view within the equirectangular sphere.
        # So, we rotate the perspective rays.
        rotated_xyz_flat = R2_torch @ R1_torch @ xyz_flat
        rotated_xyz = rotated_xyz_flat.T.reshape(height, width, 3) # H, W, 3

        # Project each ray onto the source frame.  Which formula applies depends on
        # the projection, so it lives in vr_geometry alongside its exact inverse
        # (used by Perspec2Equirec) to keep the two from drifting apart.
        lon_px, lat_px = rays_to_frame_pixels(rotated_xyz, geometry, is_left_eye)

        # Pre-sanitise pixel coords before normalising for grid_sample.
        if geometry.wraps_longitude:
            # This view closes into a full circle, so wrap circularly and the
            # 0°/360° seam maps cleanly.  fmod also handles atan2 returning
            # exactly ±π, which would give lon_px == equ_w.
            lon_px = torch.fmod(lon_px, equ_w)
        else:
            # Partial coverage: pixels past the edge belong to the other eye (or to
            # nothing), so clamp within this eye's columns instead of wrapping.
            eye_x_min, eye_x_max = geometry.eye_x_bounds(is_left_eye)
            lon_px = torch.clamp(lon_px, eye_x_min, eye_x_max)
        # Latitude: clamp at the poles so float rounding near ±90° never exceeds image bounds.
        lat_px = torch.clamp(lat_px, 0.0, equ_h - 1.0)

        # Create grid for grid_sample (expects N, H_out, W_out, 2) with (x, y) in [-1, 1]
        grid_x = (lon_px / (equ_w - 1)) * 2.0 - 1.0
        grid_y = (lat_px / (equ_h - 1)) * 2.0 - 1.0
        grid = torch.stack((grid_x, grid_y), dim=2).unsqueeze(0) # 1, H_out, W_out, 2

        # VR-PERF-11: use per-frame float cache to avoid re-converting the full equirect
        # frame for every tile/crop GetPerspective call.  Cache is invalidated (set to None)
        # by the caller (frame_worker) via copy_() + _img_float = None each new frame.
        if self._img_float is None:
            self._img_float = self._img_tensor_cxhxw_rgb_uint8.float() * (1.0 / 255.0)
        img_float = self._img_float

        # Sample from the equirectangular image.
        # padding_mode='border' replicates edge pixels for any residual out-of-bound coords
        # after the fmod/clamp above, avoiding the black-pixel artefacts that 'zeros' caused.
        persp_float = F.grid_sample(img_float.unsqueeze(0), grid,
                                    mode='bilinear', padding_mode='border', align_corners=True)

        # Use round_() before byte() for nearest-integer rounding (matches P2E convention).
        # Plain .byte() truncates toward zero, causing a systematic -0.5 LSB bias on
        # pixel values that would otherwise round up (e.g. 127.9 → 127 instead of 128).
        persp_uint8 = torch.clamp(persp_float.squeeze(0) * 255.0, 0, 255).round_().byte()
        return persp_uint8

    def get_width(self):
        return self._width

    def get_height(self):
        return self._height
