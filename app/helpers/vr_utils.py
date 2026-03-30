import threading
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from collections import OrderedDict
from typing import Optional


# Assuming Equirec2Perspec_vr and Perspec2Equirec_vr are in app.processors.external
from app.processors.external.Equirec2Perspec_vr import (
    Equirectangular as E2P_Equirectangular,
)
from app.processors.external.Perspec2Equirec_vr import Perspective as P2E_Perspective

# P2E-CACHE-02: module-level feathered-mask cache — persists across worker instances so
# both pool workers (recording) and single-frame workers (scrubbing) benefit.
# Cache hit skips _apply_feathering (max_pool2d + GaussianBlur on the full H×W equirect).
# Key: (theta, phi, fov, is_left_eye, orig_height, orig_width) — purely geometric.
# CPU RAM cost: (1, H, W) float32 ≈ 8 MB at 1080p, 30 MB at 4K.  Max 8 entries.
# Kept at 8 (not 16) to limit CPU RAM pressure: 8×30 MB = 240 MB vs 16×30 MB = 480 MB at 4K.
# Thread-safe: _FEATHERED_MASK_CACHE_LOCK guards all read-modify-write sequences so
# concurrent pool workers cannot race on eviction (KeyError on move_to_end).
# NOTE: .cpu() transfer happens OUTSIDE the lock to avoid holding the lock during the
# slow GPU→CPU copy (~30 MB per entry) which would block all 8 pool workers.
_FEATHERED_MASK_CACHE: OrderedDict = OrderedDict()
_FEATHERED_MASK_CACHE_MAX = 8
_FEATHERED_MASK_CACHE_LOCK = threading.Lock()


# Define Sobel kernels once at the module level
_SOBEL_X_KERNEL = torch.tensor(
    [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=torch.float32
).reshape(1, 1, 3, 3)
_SOBEL_Y_KERNEL = torch.tensor(
    [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], dtype=torch.float32
).reshape(1, 1, 3, 3)


def _get_sobel_kernels(device):
    """Moves the pre-defined Sobel kernels to the specified device."""
    return _SOBEL_X_KERNEL.to(device), _SOBEL_Y_KERNEL.to(device)


class EquirectangularConverter:
    def __init__(self, equirect_image_data_rgb_uint8: np.ndarray, device: torch.device):
        """
        Initializes with equirectangular image data.
        :param equirect_image_data_rgb_uint8: NumPy array (H, W, C) in RGB, uint8 format.
        :param device: PyTorch device to use.
        """

        self.device = device
        # Convert NumPy HWC RGB to Torch CHW RGB tensor on GPU
        self.equirect_tensor_cxhxw_rgb_uint8 = (
            torch.from_numpy(equirect_image_data_rgb_uint8)
            .permute(2, 0, 1)
            .to(self.device)
        )

        self.channels, self.height, self.width = (
            self.equirect_tensor_cxhxw_rgb_uint8.shape
        )
        self.e2p_instance = E2P_Equirectangular(self.equirect_tensor_cxhxw_rgb_uint8)

    def calculate_theta_phi_from_bbox(self, bbox_np: np.ndarray):
        x1, y1, x2, y2 = map(int, bbox_np)
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2

        theta = (x_center / self.width - 0.5) * 360.0
        phi = (
            -(y_center / self.height - 0.5) * 180.0
        )  # Negative because image y is top-to-bottom

        return theta, phi

    def get_perspective_crop(
        self, FOV: float, THETA: float, PHI: float, height: int, width: int
    ) -> torch.Tensor:
        """
        Returns a perspective crop as a Torch tensor (C, H, W) in RGB, uint8 format, on GPU.
        """
        # E2P_Equirectangular.GetPerspective now returns a Torch tensor (CHW, RGB, uint8)
        persp_torch_cxhxw_rgb_uint8 = self.e2p_instance.GetPerspective(
            FOV, THETA, PHI, height, width
        )
        return persp_torch_cxhxw_rgb_uint8


class PerspectiveConverter:
    def __init__(
        self, base_equirect_image_data_rgb_uint8: np.ndarray, device: torch.device
    ):
        """
        Initializes with the base equirectangular image data (used for dimensions and as background).
        :param base_equirect_image_data_rgb_uint8: NumPy array (H, W, C) in RGB, uint8 format.
        :param device: PyTorch device to use.
        """

        self.device = device
        # Only store dimensions — the full pixel data is never used after __init__.
        # Previously a full GPU uint8 tensor was kept alive permanently, wasting
        # ~5–25 MB of VRAM per cached PerspectiveConverter instance.
        h, w, c = base_equirect_image_data_rgb_uint8.shape
        self.orig_height: int = h
        self.orig_width: int = w
        self.orig_channels: int = c
        self.sobel_x_kernel, self.sobel_y_kernel = _get_sobel_kernels(self.device)
        # Bounded LRU cache for GaussianBlur instances (keyed by kernel_size, sigma).
        # A plain dict would grow unbounded across frames with varying face sizes.
        self._blur_cache: OrderedDict[tuple, torch.nn.Module] = OrderedDict()
        self._blur_cache_max = 32

    def _apply_feathering(
        self,
        mask_torch: torch.Tensor,
        feather_radius: int = 15,
        blur_sigma_factor: float = 0.5,
        erosion_kernel_size: int = 5,
    ) -> torch.Tensor:
        """Applies feathering to a Torch mask using a memory-efficient erosion followed by Gaussian blur.
        :param mask_torch: Torch tensor (1, H, W) or (H, W), boolean or float, on GPU.
        :param feather_radius: The approximate radius of the feathering effect (in pixels).
        :param blur_sigma_factor: Factor to determine sigma from feather_radius.
        :param erosion_kernel_size: Size of the kernel for the erosion step. Must be an odd integer.
        :return: Feathered mask as Torch tensor (1, H, W), float, on GPU.
        """
        mask_float_torch = mask_torch.float()
        if mask_float_torch.ndim == 2:  # HW
            mask_float_torch = mask_float_torch.unsqueeze(0)  # 1HW
        if mask_float_torch.ndim == 3 and mask_float_torch.shape[0] != 1:
            mask_float_torch = mask_float_torch[0:1, :, :]

        # --- EROSION STEP (Memory-Efficient using max_pool2d) ---
        eroded_mask = mask_float_torch
        if erosion_kernel_size > 1:
            kernel_size_er = (
                erosion_kernel_size
                if erosion_kernel_size % 2 != 0
                else erosion_kernel_size + 1
            )
            padding = kernel_size_er // 2

            # Erosion = 1 - Dilation(1 - mask). Dilation of a binary mask is equivalent to max_pool2d.
            inverted_mask = 1.0 - mask_float_torch
            # Add batch dimension for max_pool2d, which expects (N, C, H, W)
            dilated_inverted = F.max_pool2d(
                inverted_mask.unsqueeze(0),
                kernel_size=kernel_size_er,
                stride=1,
                padding=padding,
            )
            eroded_mask = 1.0 - dilated_inverted.squeeze(0)

        # --- BLUR STEP for feathering ---
        kernel_size_blur = max(3, 2 * feather_radius + 1)
        sigma = max(1.0, float(feather_radius) * blur_sigma_factor)

        blur_key = (kernel_size_blur, sigma)
        if blur_key not in self._blur_cache:
            if len(self._blur_cache) >= self._blur_cache_max:
                self._blur_cache.popitem(last=False)  # evict oldest
            self._blur_cache[blur_key] = transforms.GaussianBlur(
                kernel_size_blur, sigma
            )
        else:
            # Move to end (most-recently-used) for LRU eviction ordering
            self._blur_cache.move_to_end(blur_key)

        gauss = self._blur_cache[blur_key]
        feathered_mask = gauss(eroded_mask)

        feathered_mask = torch.clamp(feathered_mask, 0.0, 1.0)

        return feathered_mask  # Returns 1HW float mask

    def stitch_single_perspective(
        self,
        target_equirect_torch_cxhxw_rgb_uint8: torch.Tensor,
        processed_crop_torch_cxhxw_rgb_uint8: torch.Tensor,
        theta: float,
        phi: float,
        fov: float,
        is_left_eye: Optional[bool],  # None = single-eye (full-frame) mode
    ):
        """
        Stitches a single processed perspective crop back into the target equirectangular image.
        Modifies target_equirect_torch_cxhxw_rgb_uint8 in place.
        """
        if (
            processed_crop_torch_cxhxw_rgb_uint8 is None
            or processed_crop_torch_cxhxw_rgb_uint8.numel() == 0
        ):
            print(
                f"[WARN] stitch_single_perspective: processed_crop is None or empty. Skipping stitch for theta={theta}, phi={phi}."
            )
            return

        p2e_instance = P2E_Perspective(
            processed_crop_torch_cxhxw_rgb_uint8, FOV=fov, THETA=theta, PHI=phi
        )
        equirect_component_torch, mask_torch_original_shape = p2e_instance.GetEquirec(
            self.orig_height, self.orig_width
        )

        # P2E-CACHE-02: feathered-mask cache (module-level) — depends only on geometry
        # (theta/phi/fov/eye/size), not on image content.  Module-level so both pool workers
        # (recording) and single-frame workers (scrubbing) share the same cached masks.
        # Cache hit skips eye-mask derivation, feather-radius estimation, max_pool2d erosion,
        # and Gaussian blur — all of which operate on the full H×W equirect frame.
        _fmask_key = (
            round(theta, 3),
            round(phi, 3),
            round(fov, 3),
            is_left_eye,
            self.orig_height,
            self.orig_width,
        )
        # Thread-safe cache lookup: hold the lock only for the dict read so concurrent
        # pool workers cannot race between `key in cache` and `move_to_end(key)`.
        _mask_device = mask_torch_original_shape.device
        with _FEATHERED_MASK_CACHE_LOCK:
            feathered_mask_torch_float_1hw = _FEATHERED_MASK_CACHE.get(_fmask_key)
            if feathered_mask_torch_float_1hw is not None:
                _FEATHERED_MASK_CACHE.move_to_end(_fmask_key)

        if feathered_mask_torch_float_1hw is not None:
            # Cache stores CPU tensors — move to device here (avoids persistent GPU fragmentation).
            # (~36 MiB per entry at 4K; 16 entries = ~576 MiB saved on GPU)
            feathered_mask_torch_float_1hw = feathered_mask_torch_float_1hw.to(
                _mask_device
            )

        if feathered_mask_torch_float_1hw is None:
            eye_region_mask = torch.zeros_like(
                mask_torch_original_shape, dtype=torch.bool
            )
            if is_left_eye is None:
                # Single-eye mode: stitch covers the full frame
                eye_region_mask[:] = True
            else:
                half_width = self.orig_width // 2
                if is_left_eye:
                    eye_region_mask[:, :, :half_width] = True
                else:
                    eye_region_mask[:, :, half_width:] = True

            eye_specific_mask_torch_original_shape = (
                mask_torch_original_shape & eye_region_mask
            )

            # Fixed feather parameters — small kernels keep max_pool2d + GaussianBlur fast
            # even on 4K equirects (1920×3840).  Dynamic scaling (VR-PERF-12) was replaced
            # because it produced erosion_k=41 on 4K frames, which caused severe GPU stalls.
            feather_radius_val = 12
            feathered_mask_torch_float_1hw = self._apply_feathering(
                eye_specific_mask_torch_original_shape,
                feather_radius=feather_radius_val,
                blur_sigma_factor=0.5,
                erosion_kernel_size=(2 * feather_radius_val + 1),  # = 25
            )

            # Store result on CPU — avoids holding large float masks in GPU memory permanently.
            # .cpu() transfer happens BEFORE acquiring the lock to avoid holding the lock
            # during the ~30 MB GPU→CPU copy, which would serialize all 8 pool workers.
            _fmask_cpu = feathered_mask_torch_float_1hw.cpu()
            with _FEATHERED_MASK_CACHE_LOCK:
                if _fmask_key not in _FEATHERED_MASK_CACHE:
                    if len(_FEATHERED_MASK_CACHE) >= _FEATHERED_MASK_CACHE_MAX:
                        _FEATHERED_MASK_CACHE.popitem(last=False)
                    _FEATHERED_MASK_CACHE[_fmask_key] = _fmask_cpu

            del eye_region_mask, eye_specific_mask_torch_original_shape

        # VR-PERF-10b: removed the `feathered_mask.max() < 1e-4` guard (VR-14).
        # That check called `.max()` on a (1, H, W) GPU tensor inside an `if`, which
        # forced an implicit .item() → CUDA stream sync (waits for ALL workers' pending
        # GPU ops on the default stream).  With 8 workers × 4 stitch calls per frame,
        # 32 such syncs per frame-batch were causing cascading stalls.
        # Safety: blending with a zero mask is a no-op — `target += (component - target) * 0`
        # leaves the target unchanged, so removing the guard is correct.

        # Memory-efficient blending
        # uses in-place operations to reduce peak memory usage.
        # The logic is equivalent to: target = target + (component - target) * mask

        # 1. Convert tensors to float for calculation.
        target_float = target_equirect_torch_cxhxw_rgb_uint8.float()
        component_float = equirect_component_torch.float()

        # 2. Calculate the difference between component and target.
        #    This can be done in-place on component_float to save one allocation.
        component_float.sub_(target_float)

        # 3. Multiply the difference by the mask (in-place).
        component_float.mul_(feathered_mask_torch_float_1hw)

        # 4. Add the weighted difference to the target (in-place).
        target_float.add_(component_float)

        # 5. Clamp and convert back to uint8, writing back to the original tensor.
        # Q-BUG-02: use round_() before byte() to avoid systematic truncation bias
        # (.byte() truncates toward zero; .round_() gives nearest integer).
        target_equirect_torch_cxhxw_rgb_uint8[:] = (
            target_float.clamp_(0, 255).round_().byte()
        )

        del p2e_instance, equirect_component_torch, mask_torch_original_shape
        del feathered_mask_torch_float_1hw
        # Manually clear large temporary float tensors to help the garbage collector
        del target_float, component_float
