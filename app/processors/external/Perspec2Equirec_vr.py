import threading
from collections import OrderedDict
from functools import lru_cache

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from app.helpers.vr_geometry import VRGeometry, frame_pixel_rays

# P2E-CACHE-01: module-level (grid, mask_out) cache — purely geometric, so the same
# (theta, phi, fov, crop_h, crop_w, eq_h, eq_w) always produces the same tensors.
# Cache hit skips the expensive H×W matrix-multiply, projection, and mask creation,
# leaving only the image-dependent F.grid_sample call (unavoidable per frame).
# CPU RAM cost: grid=(1,H,W,2) float32 + mask=(1,H,W) bool ≈ 17 MB at 1080p, 64 MB at 4K.
# 4 entries covers 2 faces × 2 eyes in a typical VR recording session with stable positions.
# Kept at 4 (not 8) to limit CPU RAM pressure: 4×64 MB = 256 MB vs 8×64 MB = 512 MB at 4K.
# Thread-safe: _P2E_GRID_MASK_CACHE_LOCK guards all read-modify-write sequences so
# concurrent pool workers cannot race on eviction (KeyError on dict access).
# NOTE: .cpu() transfers happen OUTSIDE the lock to avoid holding the lock during slow
# GPU→CPU copies (64 MB per entry) which would block all 8 pool workers sequentially.
_P2E_GRID_MASK_CACHE: OrderedDict = OrderedDict()
_P2E_GRID_MASK_CACHE_MAX = 4
_P2E_GRID_MASK_CACHE_LOCK = threading.Lock()


# calculates the 3D coordinate grid for the output frame.
# It is decorated with @lru_cache to ensure it only runs once for a given
# geometry, caching the result for all subsequent calls.
# Always stored on CPU — callers move to device as needed, keeping GPU memory free.
# maxsize=8: caps the cache at 8 unique geometries (~88 MB each at 4K) to prevent
# unbounded CPU RAM growth when multiple video resolutions are processed in a session.
# Raised from 4 because a Both-Eyes fisheye needs one entry per eye (each eye has
# its own optical axis), where equirectangular needs only one for the whole frame.
@lru_cache(maxsize=8)
def _get_frame_xyz_grid_cached(
    geometry: VRGeometry, is_left_eye: "bool | None"
) -> tuple[torch.Tensor, "torch.Tensor | None"]:
    """
    Generates and caches a grid of 3D Cartesian unit vectors corresponding to
    pixels in the source frame, plus a validity mask (None when every pixel
    carries a usable direction). Cached on CPU to avoid GPU fragmentation.
    """
    print(
        f"[VR Grid Cache] Generating new {geometry.projection} XYZ grid for "
        f"{geometry.frame_width}x{geometry.frame_height} "
        f"at {geometry.coverage_deg:g}° coverage on cpu..."
    )
    return frame_pixel_rays(geometry, is_left_eye, torch.device("cpu"))


def clear_p2e_caches() -> None:
    """Release the cached pixel→ray grids and (grid, mask) pairs.

    Called between jobs so entries built for one video's resolution and VR
    geometry do not sit on CPU and GPU memory while the next one runs.
    """
    _get_frame_xyz_grid_cached.cache_clear()
    with _P2E_GRID_MASK_CACHE_LOCK:
        _P2E_GRID_MASK_CACHE.clear()


# This function should be at the module level
@lru_cache(maxsize=1024)  # Bounded: prevents unbounded GPU tensor accumulation for long videos
def _get_rotation_matrices_cached(THETA_deg: float, PHI_deg: float, device_str: str):
    """
    Calculates and caches rotation matrices.
    THETA_deg, PHI_deg are in degrees.
    device_str is the string representation of the torch device.
    """
    device = torch.device(device_str)
    y_axis_np = np.array([0.0, 1.0, 0.0], np.float32)
    z_axis_np = np.array([0.0, 0.0, 1.0], np.float32)

    # Convert degrees to radians for Rodrigues
    theta_rad_val = np.radians(THETA_deg)
    phi_rad_val = np.radians(PHI_deg)

    # 1. Yaw rotation
    R1_np, _ = cv2.Rodrigues(z_axis_np * theta_rad_val)

    # 2. Pitch rotation axis and matrix
    rotated_y_axis_np = np.dot(R1_np, y_axis_np)
    # PHI is up/down angle. Negative PHI in Rodrigues often means rotating "upwards" from XY plane around the new Y.
    R2_np, _ = cv2.Rodrigues(rotated_y_axis_np * -phi_rad_val)

    R1_inv_torch = torch.from_numpy(np.linalg.inv(R1_np)).float().to(device)
    R2_inv_torch = torch.from_numpy(np.linalg.inv(R2_np)).float().to(device)

    return R1_inv_torch, R2_inv_torch

class Perspective:
    def __init__(self, img_tensor_cxhxw_rgb_uint8: torch.Tensor, FOV: float, THETA: float, PHI: float,
                 geometry: "VRGeometry | None" = None, is_left_eye: "bool | None" = None):
        """
        Initializes with a perspective image tensor.
        :param img_tensor_cxhxw_rgb_uint8: Torch tensor (C, H, W) in RGB, uint8 format, on GPU.
        :param geometry: how the target frame's pixels map to directions.  None keeps
            the historical assumption of a 360°x180° equirect frame (VR180 SBS); the
            frame size is then filled in by GetEquirec, which is where it is known.
        :param is_left_eye: which eye this crop belongs to, or None for a single-view frame.
        """
        if not isinstance(img_tensor_cxhxw_rgb_uint8, torch.Tensor):
            raise ValueError("Input must be a PyTorch tensor.")
        if img_tensor_cxhxw_rgb_uint8.ndim != 3:
            raise ValueError("Input tensor must be 3-dimensional (C, H, W).")

        self._img_tensor_cxhxw_rgb_float = img_tensor_cxhxw_rgb_uint8.float() / 255.0 # Normalize to [0,1]
        self.device = img_tensor_cxhxw_rgb_uint8.device
        self._channels, self._height, self._width = self._img_tensor_cxhxw_rgb_float.shape

        self.geometry = geometry
        self.is_left_eye = is_left_eye

        # Store original THETA, PHI degrees and device string for caching rotation matrices
        self.THETA_deg_for_cache = THETA
        self.PHI_deg_for_cache = PHI
        self.device_str_for_cache = str(self.device)

        self._init_params(FOV, THETA, PHI)

    def _init_params(self, FOV, THETA, PHI):
        self.wFOV = FOV
        self.THETA_rad = torch.deg2rad(torch.tensor(THETA, device=self.device, dtype=torch.float32))
        self.PHI_rad = torch.deg2rad(torch.tensor(PHI, device=self.device, dtype=torch.float32))
        self.hFOV = float(self._height) / float(self._width) * FOV
        self.w_len = torch.tan(torch.deg2rad(torch.tensor(self.wFOV / 2.0, device=self.device)))
        self.h_len = torch.tan(torch.deg2rad(torch.tensor(self.hFOV / 2.0, device=self.device)))

        # Rotation happens in whichever frame the pixel→ray grid is expressed in:
        # the frame-level sphere for equirectangular, the eye's own camera frame for
        # a fisheye.  rotation_theta() picks the matching longitude.
        self.rotation_theta_deg = (
            THETA if self.geometry is None
            else self.geometry.rotation_theta(THETA, self.is_left_eye)
        )

        # Call the new module-level cached function
        self.R1, self.R2 = _get_rotation_matrices_cached(
            self.rotation_theta_deg,
            self.PHI_deg_for_cache,
            self.device_str_for_cache
        )

    def SetParameters(self, FOV, THETA, PHI):
        self._init_params(FOV, THETA, PHI)

    def GetEquirec(self, height: int, width: int) -> tuple[torch.Tensor, torch.Tensor]:
        # P2E-CACHE-01: (grid, mask_out) are purely geometric — same theta/phi/fov/size
        # gives identical results every frame.  Cache to skip the expensive H×W matrix-
        # multiply, projection, and mask creation on every stitch call for stable faces.
        # Only F.grid_sample (image-dependent) is executed on every call.
        # Cache key now includes the device so GPU-resident tensors are reused
        # without a host→device round-trip on every stitch call. The previous
        # CPU-cached variant was uploading ~56MiB grid + ~7MiB mask per cache
        # hit, every frame, on every pool worker — and each upload's sync was
        # spin-waiting on CUDA 13/Windows. The "GPU fragmentation" the original
        # comment worried about is bounded by _P2E_GRID_MASK_CACHE_MAX entries.
        geometry = self.geometry
        if geometry is None:
            geometry = VRGeometry(frame_height=height, frame_width=width)
        elif (geometry.frame_height, geometry.frame_width) != (height, width):
            raise ValueError(
                f"geometry describes a {geometry.frame_width}x{geometry.frame_height} "
                f"frame but GetEquirec was asked for {width}x{height}"
            )

        # The grid also depends on the projection, coverage and eye, so the geometry
        # is part of the key — otherwise switching coverage mid-session would keep
        # reusing a grid built for the previous one.
        _cache_key = (self.THETA_deg_for_cache, self.PHI_deg_for_cache,
                      self.wFOV, self._height, self._width, height, width,
                      geometry, self.is_left_eye,
                      str(self.device))
        # Thread-safe cache lookup — hold lock only for the dict read.
        with _P2E_GRID_MASK_CACHE_LOCK:
            _cached = _P2E_GRID_MASK_CACHE.get(_cache_key)

        if _cached is not None:
            grid, mask_out = _cached
        else:
            # Call the cached function to get the 3D coordinate grid (stored on CPU).
            # Move to device for the matrix multiply and projection.
            xyz_equ_norm, pixel_valid = _get_frame_xyz_grid_cached(
                geometry, self.is_left_eye
            )
            xyz_equ_norm = xyz_equ_norm.to(self.device)
            if pixel_valid is not None:
                pixel_valid = pixel_valid.to(self.device)

            # Rotate these 3D points (from equirect space to perspective camera's view space)
            xyz_flat = xyz_equ_norm.reshape(-1, 3).T  # (3, H*W)
            # R1, R2 are inverse rotations from _calc_rotation_matrices
            rotated_xyz_flat = self.R1 @ self.R2 @ xyz_flat
            rotated_xyz_persp_view = rotated_xyz_flat.T.reshape(height, width, 3)  # H, W, 3

            # Perspective projection: u = x'/z', v = y'/z'
            depth_val = rotated_xyz_persp_view[..., 0]
            is_in_front = depth_val > 1e-5  # Points in front of the camera

            # Normalized screen coordinates (relative to camera's principal axis)
            u_norm = torch.full_like(depth_val, float('inf'))
            v_norm = torch.full_like(depth_val, float('inf'))

            safe_depth_divisor = torch.where(is_in_front, depth_val, torch.tensor(1.0, device=self.device))
            u_norm = torch.where(is_in_front, rotated_xyz_persp_view[..., 1] / safe_depth_divisor, u_norm)
            v_norm = torch.where(is_in_front, rotated_xyz_persp_view[..., 2] / safe_depth_divisor, v_norm)

            # Check FOV conditions
            fov_conditions = (u_norm >= -self.w_len) & (u_norm <= self.w_len) & \
                             (v_norm >= -self.h_len) & (v_norm <= self.h_len)

            mask = is_in_front & fov_conditions  # H, W boolean tensor
            if pixel_valid is not None:
                # Fisheye: pixels outside the lens circle, or belonging to the other
                # eye, hold no direction at all and must never be written to.
                mask = mask & pixel_valid

            grid_x_persp = u_norm / self.w_len
            grid_y_persp = -(v_norm / self.h_len)  # Invert Y-axis for grid_sample convention

            # Bug 4 fix: clamp out-of-FOV coords to ±1.0 (boundary) and use padding_mode='border'.
            grid_x_persp = torch.where(mask, grid_x_persp, torch.clamp(grid_x_persp, -1.0, 1.0))
            grid_y_persp = torch.where(mask, grid_y_persp, torch.clamp(grid_y_persp, -1.0, 1.0))

            grid = torch.stack((grid_x_persp, grid_y_persp), dim=2).unsqueeze(0)  # 1, H_out, W_out, 2
            mask_out = mask.unsqueeze(0)  # 1, H, W

            # Cache GPU tensors directly. The earlier CPU-storage variant was
            # paying a host→device upload (~63MiB total) on every cache hit
            # which, multiplied by every face × every frame × every worker,
            # dominated PCIe traffic and caused per-transfer sync spin.
            with _P2E_GRID_MASK_CACHE_LOCK:
                if _cache_key not in _P2E_GRID_MASK_CACHE:
                    if len(_P2E_GRID_MASK_CACHE) >= _P2E_GRID_MASK_CACHE_MAX:
                        _P2E_GRID_MASK_CACHE.popitem(last=False)
                    _P2E_GRID_MASK_CACHE[_cache_key] = (grid, mask_out)

            # Free intermediate GPU tensors that are not part of the cached
            # outputs. grid and mask_out are still needed below; all other
            # intermediates are no longer referenced.
            del xyz_equ_norm, xyz_flat, rotated_xyz_flat, rotated_xyz_persp_view
            del depth_val, is_in_front, safe_depth_divisor, u_norm, v_norm
            del fov_conditions, mask, grid_x_persp, grid_y_persp, pixel_valid

        # Image-dependent sampling — always executed (image changes every frame)
        equirect_component_float = F.grid_sample(self._img_tensor_cxhxw_rgb_float.unsqueeze(0), grid,
                                                 mode='bilinear', padding_mode='border', align_corners=True)

        # P2E-MEM-02: convert to uint8 in-place to avoid two ~108 MiB float32 temporaries
        # that torch.clamp(tensor * 255.0, ...) would otherwise allocate.
        # squeeze_(0): in-place view reshape — removes batch dim, no copy.
        # mul_(255.0), clamp_(0, 255): in-place — reuse equirect_component_float storage.
        # .byte(): only the final uint8 tensor is a new allocation (~27 MiB vs ~243 MiB before).
        equirect_component_uint8 = equirect_component_float.squeeze_(0).mul_(255.0).clamp_(0, 255).byte()
        del equirect_component_float

        return equirect_component_uint8, mask_out
