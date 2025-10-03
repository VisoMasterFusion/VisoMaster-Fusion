import cv2
import numpy as np
import torch
import torch.nn.functional as F
from functools import lru_cache


# calculates the 3D coordinate grid for an equirectangular output.
# It is decorated with @lru_cache to ensure it only runs once for a given
# height, width, and device, caching the result for all subsequent calls.
@lru_cache(maxsize=None)
def _get_equirect_xyz_grid_cached(height: int, width: int, device_str: str) -> torch.Tensor:
    """
    Generates and caches a grid of 3D Cartesian unit vectors corresponding to
    pixels in an equirectangular projection.
    """
    print(f"[VR Grid Cache] Generating new equirectangular XYZ grid for {width}x{height} on {device_str}...")
    device = torch.device(device_str)

    # Create equirectangular grid
    equ_lon_coords = torch.linspace(-180, 180, width, device=device, dtype=torch.float32)
    equ_lat_coords = torch.linspace(90, -90, height, device=device, dtype=torch.float32)
    equ_lon_grid, equ_lat_grid = torch.meshgrid(equ_lon_coords, equ_lat_coords, indexing='xy')

    # Convert equirectangular (lon, lat) to 3D Cartesian unit vectors
    lon_rad = torch.deg2rad(equ_lon_grid)
    lat_rad = torch.deg2rad(equ_lat_grid)

    x_3d = torch.cos(lat_rad) * torch.cos(lon_rad)
    y_3d = torch.cos(lat_rad) * torch.sin(lon_rad)
    z_3d = torch.sin(lat_rad)
    xyz_equ_norm = torch.stack((x_3d, y_3d, z_3d), dim=2)  # Shape: H, W, 3

    return xyz_equ_norm


# This function should be at the module level
@lru_cache(maxsize=None) # Cache based on THETA_deg, PHI_deg, device_str
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
    def __init__(self, img_tensor_cxhxw_rgb_uint8: torch.Tensor, FOV: float, THETA: float, PHI: float):
        """
        Initializes with a perspective image tensor.
        :param img_tensor_cxhxw_rgb_uint8: Torch tensor (C, H, W) in RGB, uint8 format, on GPU.
        """
        if not isinstance(img_tensor_cxhxw_rgb_uint8, torch.Tensor):
            raise ValueError("Input must be a PyTorch tensor.")
        if img_tensor_cxhxw_rgb_uint8.ndim != 3:
            raise ValueError("Input tensor must be 3-dimensional (C, H, W).")

        self._img_tensor_cxhxw_rgb_float = img_tensor_cxhxw_rgb_uint8.float() / 255.0 # Normalize to [0,1]
        self.device = img_tensor_cxhxw_rgb_uint8.device
        self._channels, self._height, self._width = self._img_tensor_cxhxw_rgb_float.shape

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

        # Call the new module-level cached function
        self.R1, self.R2 = _get_rotation_matrices_cached(
            self.THETA_deg_for_cache,
            self.PHI_deg_for_cache,
            self.device_str_for_cache
        )

    def SetParameters(self, FOV, THETA, PHI):
        self._init_params(FOV, THETA, PHI)

    def GetEquirec(self, height: int, width: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Call the cached function to get the 3D coordinate grid.
        # This grid is now computed only once for this resolution and device.
        xyz_equ_norm = _get_equirect_xyz_grid_cached(height, width, str(self.device))

        # Rotate these 3D points (from equirect space to perspective camera's view space)
        xyz_flat = xyz_equ_norm.reshape(-1, 3).T # (3, H*W)
        # R1, R2 are inverse rotations from _calc_rotation_matrices
        rotated_xyz_flat = self.R1 @ self.R2 @ xyz_flat
        rotated_xyz_persp_view = rotated_xyz_flat.T.reshape(height, width, 3) # H, W, 3

        # Perspective projection: u = x'/z', v = y'/z'
        # rotated_xyz_persp_view[..., 0] is depth (along camera's X-axis)
        # rotated_xyz_persp_view[..., 1] is horizontal screen coord
        # rotated_xyz_persp_view[..., 2] is vertical screen coord
        depth_val = rotated_xyz_persp_view[..., 0]
        is_in_front = depth_val > 1e-5 # Points in front of the camera

        # Normalized screen coordinates (relative to camera's principal axis)
        # Initialize with out-of-FOV values
        u_norm = torch.full_like(depth_val, float('inf'))
        v_norm = torch.full_like(depth_val, float('inf'))

        safe_depth_divisor = torch.where(is_in_front, depth_val, torch.tensor(1.0, device=self.device))
        u_norm = torch.where(is_in_front, rotated_xyz_persp_view[..., 1] / safe_depth_divisor, u_norm)
        v_norm = torch.where(is_in_front, rotated_xyz_persp_view[..., 2] / safe_depth_divisor, v_norm)

        # Check FOV conditions
        fov_conditions = (u_norm >= -self.w_len) & (u_norm <= self.w_len) & \
                         (v_norm >= -self.h_len) & (v_norm <= self.h_len)
        
        mask = is_in_front & fov_conditions # H, W boolean tensor

        # Map normalized screen coordinates to pixel coordinates in the perspective image
        # For grid_sample, these need to be in [-1, 1] range.
        # u_norm maps to x in perspective image, v_norm maps to y
        # Perspective image: x from -w_len to w_len, y from -h_len to h_len (center is 0,0)
        # grid_sample x: -1 (left) to 1 (right)
        # grid_sample y: -1 (top) to 1 (bottom)
        
        grid_x_persp = u_norm / self.w_len  # Maps to [-1, 1]
        grid_y_persp = - (v_norm / self.h_len)  # Invert Y-axis for grid_sample convention

        # Where mask is False, set grid coordinates to something outside [-1,1] to be handled by padding_mode
        grid_x_persp = torch.where(mask, grid_x_persp, torch.tensor(2.0, device=self.device)) # Value > 1
        grid_y_persp = torch.where(mask, grid_y_persp, torch.tensor(2.0, device=self.device)) # Value > 1

        grid = torch.stack((grid_x_persp, grid_y_persp), dim=2).unsqueeze(0) # 1, H_out, W_out, 2

        # Sample from the perspective image
        # self._img_tensor_cxhxw_rgb_float is (C, H_persp, W_persp)
        equirect_component_float = F.grid_sample(self._img_tensor_cxhxw_rgb_float.unsqueeze(0), grid,
                                                 mode='bilinear', padding_mode='zeros', align_corners=True)
        
        equirect_component_uint8 = (torch.clamp(equirect_component_float.squeeze(0) * 255.0, 0, 255)).byte()
        
        # Mask should be (H_out, W_out, 1) or (1, H_out, W_out) for broadcasting
        mask_out = mask.unsqueeze(0) # 1, H, W

        return equirect_component_uint8, mask_out
