import math
import re
from math import ceil
from collections import OrderedDict, deque
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import v2
import kornia.enhance as ke
import kornia.color as kc
import kornia.geometry.transform as kgm

from app.processors.utils import faceutil

if TYPE_CHECKING:
    # Forward reference to the main FrameWorker orchestrator
    from .frame_worker import FrameWorker


class PipelineProcessor:
    """
    Handles the heavy tensor operations, model inference, and VRAM management.
    Operates strictly within the CUDA stream and thread context of its parent FrameWorker.
    """

    def __init__(self, worker: "FrameWorker"):
        self.worker = worker

        # Q-QUAL-03: EMA alpha for AutoColor reference statistics
        self._COLOR_EMA_ALPHA: float = 0.30

        # FW-MEM-01: Gabor kernel cache as strictly typed LRU-bounded OrderedDict
        self._gabor_kernels_expanded_cache: OrderedDict[tuple, torch.Tensor] = (
            OrderedDict()
        )
        self._gabor_kernels_cache: OrderedDict[tuple, torch.Tensor] = OrderedDict()

        # Q-QUAL-03: EMA over per-face AutoColor reference statistics to reduce flicker.
        self._color_stats_ema: OrderedDict[bytes, dict[str, torch.Tensor]] = (
            OrderedDict()
        )
        self._COLOR_STATS_EMA_MAX: int = 32

        # OPTIMIZATION: Cached Convolution Kernels (VRAM)
        self._kernel_lap: torch.Tensor | None = None
        self._kernel_sobel_x: torch.Tensor | None = None
        self._kernel_sobel_y: torch.Tensor | None = None

        self._mouth_action_score: float = 0.0

    @property
    def kernel_lap(self) -> torch.Tensor:
        """Lazy initialization of the Laplacian kernel to ensure Thread-Safe CUDA allocation."""
        if self._kernel_lap is None:
            device = self.worker.models_processor.device
            self._kernel_lap = torch.tensor(
                [[0, 1, 0], [1, -4, 1], [0, 1, 0]], device=device, dtype=torch.float32
            ).view(1, 1, 3, 3)
        return self._kernel_lap

    @property
    def kernel_sobel_x(self) -> torch.Tensor:
        """Lazy initialization of the Sobel X kernel to ensure Thread-Safe CUDA allocation."""
        if self._kernel_sobel_x is None:
            device = self.worker.models_processor.device
            self._kernel_sobel_x = torch.tensor(
                [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=device, dtype=torch.float32
            ).view(1, 1, 3, 3)
        return self._kernel_sobel_x

    @property
    def kernel_sobel_y(self) -> torch.Tensor:
        """Lazy initialization of the Sobel Y kernel to ensure Thread-Safe CUDA allocation."""
        if self._kernel_sobel_y is None:
            self._kernel_sobel_y = self.kernel_sobel_x.transpose(2, 3)
        return self._kernel_sobel_y

    def _apply_denoiser_pass(
        self,
        image_tensor_cxhxw_uint8: torch.Tensor,
        control: dict[str, Any],
        pass_suffix: str,
        kv_map: dict | None,
        color_mask: torch.Tensor | None = None,
        blend_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Helper to run the diffusion-based denoiser (Ref-LDM).

        FW-QUAL-13: pass_suffix convention:
          - "Before"     → DenoiserUNetEnableBeforeRestorersToggle (before Restoration 1)
          - "AfterFirst" → DenoiserAfterFirstRestorerToggle (between Restoration 1 and 2)
          - "After"      → DenoiserAfterRestorersToggle (after Restoration 2)
        """
        use_exclusive_path = control.get("UseReferenceExclusivePathToggle", False)
        denoiser_seed_from_slider_val = int(control.get("DenoiserBaseSeedSlider", 1))
        denoiser_mode_val = control.get(
            f"DenoiserModeSelection{pass_suffix}", "Single Step (Fast)"
        )
        ddim_steps_val = int(control.get(f"DenoiserDDIMStepsSlider{pass_suffix}", 20))
        cfg_scale_val = float(
            control.get(f"DenoiserCFGScaleDecimalSlider{pass_suffix}", 1.0)
        )
        single_step_t_val = int(
            control.get(f"DenoiserSingleStepTimestepSlider{pass_suffix}", 1)
        )
        sharpen_val = float(
            control.get(f"DenoiserLatentSharpeningDecimalSlider{pass_suffix}", 0.0)
        )

        # Global Color Correction toggle (applied equally to all active passes)
        enable_color = control.get("DenoiserColorCorrectionToggle", True)

        if not kv_map and use_exclusive_path:
            return image_tensor_cxhxw_uint8

        denoised_image = self.worker.function_worker.apply_denoiser_unet(
            image_tensor_cxhxw_uint8,
            reference_kv_map=kv_map,
            use_reference_exclusive_path=use_exclusive_path,
            denoiser_mode=denoiser_mode_val,
            base_seed=denoiser_seed_from_slider_val,
            denoiser_single_step_t=single_step_t_val,
            denoiser_ddim_steps=ddim_steps_val,
            denoiser_cfg_scale=cfg_scale_val,
            latent_sharpening_strength=sharpen_val,
            enable_color_correction=enable_color,
            color_mask=color_mask,
        )

        denoised_image = torch.clamp(denoised_image, 0, 255)

        # Restrict Denoiser to the active Swap Mask.
        if blend_mask is not None:
            if (
                blend_mask.shape[-1] != denoised_image.shape[-1]
                or blend_mask.shape[-2] != denoised_image.shape[-2]
            ):
                blend_mask = v2.functional.resize(
                    blend_mask,
                    [denoised_image.shape[-2], denoised_image.shape[-1]],
                    interpolation=v2.InterpolationMode.BILINEAR,
                    antialias=False,
                )

            denoised_image = torch.lerp(
                image_tensor_cxhxw_uint8.float(), denoised_image.float(), blend_mask
            ).to(image_tensor_cxhxw_uint8.dtype)

        return denoised_image

    @staticmethod
    def _apply_likeness(
        source_latent: torch.Tensor, target_latent: torch.Tensor, params: dict
    ) -> torch.Tensor:
        """FW-QUAL-09: Identity Boost (Face Likeness) via SLERP / LERP on ArcFace embeddings.
        Now includes Latent Overdrive (Magnitude Scaling) via FaceLikenessStrengthDecimalSlider
        to increase identity strength without geometric warping.
        """
        if not params.get("FaceLikenessEnableToggle", False):
            return source_latent

        factor: float = float(params.get("FaceLikenessFactorDecimalSlider", 0.0))

        # Extract the new volume multiplier. Default to 1.0 (baseline energy) if missing.
        strength_multiplier: float = float(
            params.get("FaceLikenessStrengthDecimalSlider", 1.0)
        )

        # Early return only if both sliders are at their passive baseline
        if factor == 0.0 and strength_multiplier == 1.0:
            return source_latent

        # 1. Capture original energy (Norms are generally constant in ArcFace)
        s_norm: torch.Tensor = torch.norm(source_latent)
        t_norm: torch.Tensor = torch.norm(target_latent)

        if s_norm < 1e-6 or t_norm < 1e-6:
            return source_latent

        # 2. Normalize to get directional vectors on the hypersphere
        s_dir: torch.Tensor = source_latent / s_norm
        t_dir: torch.Tensor = target_latent / t_norm

        # Declare the variable once for mypy inference
        blended_dir: torch.Tensor

        if factor < 0.0:
            # --- INTERPOLATION (SLERP) ---
            # Move naturally towards the target face along the sphere
            t: float = 1.0 + factor

            cos_theta: torch.Tensor = torch.sum(s_dir * t_dir)
            cos_theta = torch.clamp(cos_theta, -0.9999, 0.9999)
            theta: torch.Tensor = torch.acos(cos_theta)
            sin_theta: torch.Tensor = torch.sin(theta)

            if sin_theta < 1e-3:
                # Removed redundant type hint here
                blended_dir = (1.0 - t) * t_dir + t * s_dir
            else:
                weight_t: torch.Tensor = torch.sin((1.0 - t) * theta) / sin_theta
                weight_s: torch.Tensor = torch.sin(t * theta) / sin_theta
                # Removed redundant type hint here
                blended_dir = weight_t * t_dir + weight_s * s_dir

        elif factor > 0.0:
            # --- EXTRAPOLATION (LERP) ---
            # Push the vector away from the target to exaggerate the source identity
            difference_vector: torch.Tensor = s_dir - t_dir
            # Removed redundant type hint here
            blended_dir = s_dir + (factor * difference_vector)

        else:
            # Factor is 0.0, but strength_multiplier is active. Keep original direction.
            # Removed redundant type hint here
            blended_dir = s_dir

        # 3. Apply Directional Normalization and Latent Overdrive (Volume)
        blended_dir = blended_dir / torch.norm(blended_dir)

        # Multiply the original source energy by the new slider's volume
        final_latent: torch.Tensor = blended_dir * (s_norm * strength_multiplier)

        return final_latent

    def get_affined_face_dim_and_swapping_latents(
        self,
        original_faces: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        swapper_model: str,
        dfm_model_name: str | None,
        s_e: np.ndarray | None,
        t_e: np.ndarray | None,
        parameters: dict[str, Any],
        cmddebug: bool,
        tform: Any,
    ) -> tuple[torch.Tensor | None, Any | None, int, torch.Tensor | list | None]:
        """
        Selects the correct input face resolution and computes the swapping latent vector
        for the active swapper model.

        Args:
            original_faces: Tuple ``(face_512, face_384, face_256, face_128)`` of CHW tensors.
            swapper_model:  Active swapper name (e.g. ``"Inswapper128"``).
            dfm_model_name: DFM model filename; used only when swapper_model is ``"DeepFaceLive (DFM)"``.
            s_e:            Source ArcFace embedding (numpy array) or ``None`` for DFM.
            t_e:            Target ArcFace embedding (numpy array) or ``None``.
            parameters:     Per-face parameter dict.
            cmddebug:       Whether command-line debug output is enabled.
            tform:          Similarity transform (used by Inswapper auto-resolution).

        Returns:
            Tuple ``(input_face_affined, dfm_model_instance, dim, latent)`` where
            *input_face_affined* is ``None`` on failure, *dim* is the resolution multiplier
            (1=128, 2=256, 3=384, 4=512), and *latent* is the model-specific embedding tensor.
        """
        original_face_512, original_face_384, original_face_256, original_face_128 = (
            original_faces
        )

        dfm_model_instance = None
        input_face_affined = None
        dim = 1
        latent = None

        # FW-QUAL-09: apply_likeness_with_norm_preservation promoted to @staticmethod.
        # Use self._apply_likeness(...) everywhere below.

        # --- Inswapper128 Logic ---
        if swapper_model == "Inswapper128":
            # FS-ROBUST-01: calc_inswapper_latent may return None on emap failure
            _s_latent_np = self.worker.function_worker.calc_inswapper_latent(s_e)
            _t_latent_np = self.worker.function_worker.calc_inswapper_latent(t_e)
            if _s_latent_np is None or _t_latent_np is None:
                print(
                    "[ERROR] calc_inswapper_latent returned None (emap unavailable). Skipping swap."
                )
                return input_face_affined, dfm_model_instance, dim, latent

            latent = (
                torch.from_numpy(_s_latent_np)
                .float()
                .to(self.worker.models_processor.device)
            )
            dst_latent = (
                torch.from_numpy(_t_latent_np)
                .float()
                .to(self.worker.models_processor.device)
            )

            latent = self._apply_likeness(latent, dst_latent, parameters)

            dim = 1
            if parameters["SwapperResAutoSelectEnableToggle"]:
                if tform.scale <= 1.00:
                    dim = 4
                    input_face_affined = original_face_512
                elif tform.scale <= 1.75:
                    dim = 3
                    input_face_affined = original_face_384
                elif tform.scale <= 2:
                    dim = 2
                    input_face_affined = original_face_256
                else:
                    dim = 1
                    input_face_affined = original_face_128
            else:
                if parameters["SwapperResSelection"] == "128":
                    dim = 1
                    input_face_affined = original_face_128
                elif parameters["SwapperResSelection"] == "256":
                    dim = 2
                    input_face_affined = original_face_256
                elif parameters["SwapperResSelection"] == "384":
                    dim = 3
                    input_face_affined = original_face_384
                elif parameters["SwapperResSelection"] == "512":
                    dim = 4
                    input_face_affined = original_face_512

        # --- InStyleSwapper Logic ---
        elif swapper_model in (
            "InStyleSwapper256 Version A",
            "InStyleSwapper256 Version B",
            "InStyleSwapper256 Version C",
        ):
            version = swapper_model[-1]
            latent = (
                torch.from_numpy(
                    self.worker.function_worker.calc_swapper_latent_iss(s_e, version)
                )
                .float()
                .to(self.worker.models_processor.device)
            )
            dst_latent = (
                torch.from_numpy(
                    self.worker.function_worker.calc_swapper_latent_iss(t_e, version)
                )
                .float()
                .to(self.worker.models_processor.device)
            )

            latent = self._apply_likeness(latent, dst_latent, parameters)

            if (
                (
                    parameters["SwapModelSelection"] == "InStyleSwapper256 Version A"
                    and parameters["InStyleResAEnableToggle"]
                )
                or (
                    parameters["SwapModelSelection"] == "InStyleSwapper256 Version B"
                    and parameters["InStyleResBEnableToggle"]
                )
                or (
                    parameters["SwapModelSelection"] == "InStyleSwapper256 Version C"
                    and parameters["InStyleResCEnableToggle"]
                )
            ):
                dim = 4
                input_face_affined = original_face_512
            else:
                dim = 2
                input_face_affined = original_face_256

        # --- SimSwap Logic ---
        elif swapper_model == "SimSwap512":
            latent = (
                torch.from_numpy(
                    self.worker.function_worker.calc_swapper_latent_simswap512(s_e)
                )
                .float()
                .to(self.worker.models_processor.device)
            )
            dst_latent = (
                torch.from_numpy(
                    self.worker.function_worker.calc_swapper_latent_simswap512(t_e)
                )
                .float()
                .to(self.worker.models_processor.device)
            )

            latent = self._apply_likeness(latent, dst_latent, parameters)
            dim = 4
            input_face_affined = original_face_512

        # --- GhostFace Logic ---
        # FW-QUAL-10: use GHOSTFACE_MODELS frozenset
        elif swapper_model in self.worker.GHOSTFACE_MODELS:
            latent = (
                torch.from_numpy(
                    self.worker.function_worker.calc_swapper_latent_ghost(s_e)
                )
                .float()
                .to(self.worker.models_processor.device)
            )
            dst_latent = (
                torch.from_numpy(
                    self.worker.function_worker.calc_swapper_latent_ghost(t_e)
                )
                .float()
                .to(self.worker.models_processor.device)
            )

            latent = self._apply_likeness(latent, dst_latent, parameters)
            dim = 2
            input_face_affined = original_face_256

        # --- CSCS Logic ---
        elif swapper_model == "CSCS":
            latent = (
                torch.from_numpy(
                    self.worker.function_worker.calc_swapper_latent_cscs(s_e)
                )
                .float()
                .to(self.worker.models_processor.device)
            )
            dst_latent = (
                torch.from_numpy(
                    self.worker.function_worker.calc_swapper_latent_cscs(t_e)
                )
                .float()
                .to(self.worker.models_processor.device)
            )

            latent = self._apply_likeness(latent, dst_latent, parameters)
            dim = 2
            input_face_affined = original_face_256

        # --- DFM Logic ---
        elif swapper_model == "DeepFaceLive (DFM)" and dfm_model_name:
            # Explicitly notify FaceSwappers to unload the previous ONNX model
            # This prevents VRAM leaks when switching to heavy DFM models.
            self.worker.function_worker.manage_dfm_swapper_model_state(
                "DeepFaceLive (DFM)"
            )

            dfm_model_instance = self.worker.models_processor.load_dfm_model(
                dfm_model_name
            )
            # FIX: Attach the filename to the instance so we can robustly extract its resolution later
            if dfm_model_instance is not None:
                dfm_model_instance._dfm_filename_fallback = dfm_model_name

            latent = []
            input_face_affined = original_face_512
            dim = 4

        return input_face_affined, dfm_model_instance, dim, latent

    def _fix_drift_and_texture(
        self,
        current_face: torch.Tensor,
        first_face: torch.Tensor,
    ) -> torch.Tensor:
        """
        Corrects spatial warping and restores skin texture without ghosting.

        OPTIMIZED:
        1. Thread-safe grid caching to prevent CUDA memory fragmentation.
        2. Masked LAB AdaIN to prevent background color pollution.
        3. Hardcoded 5.0 pixel limit (tuned for 512x512) for micro-jitter correction.

        Args:
            current_face:  Current swapped face tensor [C, H, W] in range [0..1].
            first_face:    The uncorrupted first pass tensor [C, H, W].

        Returns:
            Corrected CHW float32 tensor in range [0..1].
        """
        if first_face is None:
            return current_face

        device = current_face.device
        C, H, W = current_face.shape

        # Initialize thread-safe cache for grids if it doesn't exist on this worker instance
        if not hasattr(self, "_drift_cache"):
            self._drift_cache: dict[tuple[int, int, str], dict[str, torch.Tensor]] = {}

        cache_key = (H, W, str(device))

        with torch.no_grad():
            curr_bchw = current_face.unsqueeze(0)
            first_bchw = first_face.unsqueeze(0)

            # --- 0. CACHE RETRIEVAL FOR ZERO-ALLOCATION GRIDS ---
            if cache_key not in self._drift_cache:
                y_coords = torch.arange(H, dtype=torch.float32, device=device)
                x_coords = torch.arange(W, dtype=torch.float32, device=device)
                Y_pix, X_pix = torch.meshgrid(y_coords, x_coords, indexing="ij")

                Y_norm = (Y_pix - H / 2) / (H / 2)
                X_norm = (X_pix - W / 2) / (W / 2)

                # Core mask for Center of Mass tracking
                dist_norm_core = torch.sqrt(X_norm**2 + (Y_norm / 1.2) ** 2)
                core_mask = (
                    torch.exp(-0.5 * (dist_norm_core / 0.4) ** 2)
                    .unsqueeze(0)
                    .unsqueeze(0)
                )

                # Spatial mask for peripheral anchoring
                dist_norm_spatial = torch.sqrt(X_norm**2 + (Y_norm / 0.5) ** 2)
                spatial_mask = torch.exp(
                    -0.5 * (dist_norm_spatial / 0.5) ** 2
                ).unsqueeze(0)
                spatial_mask = torch.clamp(spatial_mask, 0.0, 1.0)

                self._drift_cache[cache_key] = {
                    "Y_pix": Y_pix,
                    "X_pix": X_pix,
                    "core_mask": core_mask,
                    "spatial_mask": spatial_mask,
                }

            cache = self._drift_cache[cache_key]
            Y_pix = cache["Y_pix"]
            X_pix = cache["X_pix"]
            core_mask = cache["core_mask"]
            spatial_mask = cache["spatial_mask"]

            # --- 1. ROBUST SUB-PIXEL DRIFT CORRECTION ---
            # Fast grayscale conversion
            weights = torch.tensor([0.299, 0.587, 0.114], device=device).view(
                1, 3, 1, 1
            )
            curr_gray = (curr_bchw * weights).sum(dim=1, keepdim=True)
            first_gray = (first_bchw * weights).sum(dim=1, keepdim=True)

            blob_blur = v2.GaussianBlur(kernel_size=51, sigma=15.0)
            curr_blob = blob_blur(curr_gray)
            first_blob = blob_blur(first_gray)

            # Apply core mask to isolate face volume
            curr_blob_masked = curr_blob * core_mask
            first_blob_masked = first_blob * core_mask

            curr_mass = curr_blob_masked.sum() + 1e-8
            first_mass = first_blob_masked.sum() + 1e-8

            curr_cx = (curr_blob_masked * X_pix).sum() / curr_mass
            curr_cy = (curr_blob_masked * Y_pix).sum() / curr_mass

            first_cx = (first_blob_masked * X_pix).sum() / first_mass
            first_cy = (first_blob_masked * Y_pix).sum() / first_mass

            # Strict micro-jitter clamp (resolves high-iteration identity wash-out)
            max_drift = 5.0
            dx = torch.clamp(first_cx - curr_cx, -max_drift, max_drift)
            dy = torch.clamp(first_cy - curr_cy, -max_drift, max_drift)

            M = torch.eye(3, device=device)[:2, :].unsqueeze(0)
            M[0, 0, 2] = dx
            M[0, 1, 2] = dy

            curr_bchw = kgm.warp_affine(
                curr_bchw,
                M,
                dsize=(H, W),
                mode="bilinear",
                padding_mode="border",
                align_corners=False,
            )
            current_face = curr_bchw.squeeze(0)

            # --- 2. PERIPHERAL ANCHORING ---
            current_face = (current_face * spatial_mask) + (
                first_face * (1.0 - spatial_mask)
            )
            # Blend to dampen jitter
            current_face = (current_face * 0.85) + (first_face * 0.15)

            # --- 3. COLOR & LUMINANCE LOCK (Masked LAB AdaIN) ---
            curr_bchw_clamp = current_face.unsqueeze(0).clamp(0.0, 1.0)
            first_bchw_clamp = first_face.unsqueeze(0).clamp(0.0, 1.0)

            curr_lab = kc.rgb_to_lab(curr_bchw_clamp)
            first_lab = kc.rgb_to_lab(first_bchw_clamp)

            # Use the core mask to calculate statistics ONLY on the face, ignoring background
            # Expand mask to match LAB channels [1, 3, H, W]
            stat_mask = core_mask.expand_as(curr_lab)
            mask_sum = stat_mask.sum(dim=(2, 3), keepdim=True) + 1e-8

            # Masked Mean
            mean_curr = (curr_lab * stat_mask).sum(dim=(2, 3), keepdim=True) / mask_sum
            mean_first = (first_lab * stat_mask).sum(
                dim=(2, 3), keepdim=True
            ) / mask_sum

            # Masked Variance / Std
            var_curr = (((curr_lab - mean_curr) ** 2) * stat_mask).sum(
                dim=(2, 3), keepdim=True
            ) / mask_sum
            var_first = (((first_lab - mean_first) ** 2) * stat_mask).sum(
                dim=(2, 3), keepdim=True
            ) / mask_sum

            std_curr = torch.sqrt(var_curr) + 1e-6
            std_first = torch.sqrt(var_first) + 1e-6

            # Apply AdaIN
            matched_lab = (curr_lab - mean_curr) * (std_first / std_curr) + mean_first
            current_face = kc.lab_to_rgb(matched_lab).squeeze(0)

        return torch.clamp(current_face, 0.0, 1.0)

    def get_swapped_and_prev_face(
        self,
        output: torch.Tensor,
        input_face_affined: torch.Tensor,
        original_face_512: torch.Tensor,
        latent: torch.Tensor | list | None,
        itex: int,
        dim: int,
        swapper_model: str,
        dfm_model: Any | None,
        parameters: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Runs the swapper model inference and returns the swapped face tensor.

        Applies optional pre-swap sharpness, executes the swapper loop *itex* times
        (strength slider), and delegates to the architecture-specific branch
        (Inswapper, Ghost, SimSwap, InStyle, CSCS, or DFM).

        Args:
            output:             Pre-allocated output tensor (HWC float32, [0..1]).
            input_face_affined: Aligned face CHW tensor at the chosen resolution.
            original_face_512:  Unmodified 512-px face CHW uint8 tensor (used by DFM).
            latent:             Swapping latent computed by ``get_affined_face_dim_and_swapping_latents``.
            itex:               Number of inference iterations (from StrengthAmountSlider).
            dim:                Resolution multiplier (1=128, 2=256, 3=384, 4=512).
            swapper_model:      Active swapper name.
            dfm_model:          Loaded ``DFMModel`` instance, or ``None`` for non-DFM swappers.
            parameters:         Per-face parameter dict.

        Returns:
            Tuple ``(swap_chw_uint8, prev_face_hwc_float)``.

        Optimized to use reference swapping (Double Buffering) to minimize VRAM
        fragmentation during multi-iteration swaps (Strength/Iterations).
        """
        # Pre-process: Apply sharpness adjustment if requested
        if parameters["PreSwapSharpnessDecimalSlider"] != 1.0:
            input_face_affined = input_face_affined.permute(2, 0, 1)
            input_face_affined = v2.functional.adjust_sharpness(
                input_face_affined, parameters["PreSwapSharpnessDecimalSlider"]
            )
            input_face_affined = input_face_affined.permute(1, 2, 0)

        # Initialize tracking states using references to avoid unnecessary VRAM copies
        prev_face = input_face_affined
        first_pass_face = None
        # Strength mode 2 toggle
        use_mode_2 = parameters.get("StrengthMode2EnableToggle", False)

        # --- Inswapper128 Path ---
        if swapper_model == "Inswapper128":
            _use_batched = False

            for k in range(itex):
                # Double Buffering: Update previous state reference before current pass
                prev_face = input_face_affined

                if _use_batched:
                    # ------ BATCHED PATH (dim > 1) ------
                    tiles_list = []
                    tile_coords = []
                    for j in range(dim):
                        for i in range(dim):
                            tiles_list.append(
                                input_face_affined[j::dim, i::dim]
                                .permute(2, 0, 1)
                                .contiguous()
                            )
                            tile_coords.append((j, i))

                    batch_input = torch.stack(tiles_list, dim=0)  # [B, 3, 128, 128]
                    batch_output = torch.empty_like(batch_input)

                    self.worker.function_worker.run_inswapper_batched(
                        batch_input, latent, batch_output
                    )

                    if self.worker.models_processor.device_type == "cuda":
                        torch.cuda.current_stream().synchronize()

                    # Fallback for zero-output tiles
                    tile_sums = batch_output.abs().sum(dim=(1, 2, 3))
                    zero_mask = tile_sums < 1.0
                    if zero_mask.any():
                        batch_output[zero_mask] = batch_input[zero_mask]

                    # Reconstruction: Only one clone needed as a canvas per iteration
                    temp_output = input_face_affined.clone()
                    for idx, (j, i) in enumerate(tile_coords):
                        temp_output[j::dim, i::dim] = batch_output[idx].permute(1, 2, 0)

                    # --- MODE 2 ---
                    if use_mode_2:
                        curr_chw = temp_output.permute(2, 0, 1)
                        if k == 0:
                            first_pass_face = (
                                curr_chw.clone()
                            )  # Store first pass for drift correction
                        else:
                            curr_chw = self._fix_drift_and_texture(
                                curr_chw, first_pass_face
                            )
                            temp_output = curr_chw.permute(1, 2, 0)

                    # Update working reference and final output buffer
                    input_face_affined = temp_output
                    output = torch.clamp(temp_output * 255.0, 0, 255)

                else:
                    # ------ SEQUENTIAL PATH (ORT providers or dim==1) ------
                    # Allocate iteration canvas
                    temp_output = input_face_affined.clone()
                    tile_inputs = []
                    tile_outputs = []
                    tile_coords = []

                    for j in range(dim):
                        for i in range(dim):
                            tile = input_face_affined[j::dim, i::dim]
                            t_in = tile.permute(2, 0, 1).contiguous().unsqueeze(0)
                            t_out = torch.empty_like(t_in)
                            tile_inputs.append(t_in)
                            tile_outputs.append(t_out)
                            tile_coords.append((j, i))

                    with torch.no_grad():
                        for idx in range(len(tile_inputs)):
                            self.worker.function_worker.run_inswapper(
                                tile_inputs[idx], latent, tile_outputs[idx]
                            )

                    if self.worker.models_processor.device_type == "cuda":
                        torch.cuda.current_stream().synchronize()

                    for idx, (j, i) in enumerate(tile_coords):
                        res = (
                            tile_inputs[idx]
                            if tile_outputs[idx].sum() < 1.0
                            else tile_outputs[idx]
                        )
                        temp_output[j::dim, i::dim] = res.squeeze(0).permute(1, 2, 0)

                    # --- MODE 2 ---
                    if use_mode_2:
                        curr_chw = temp_output.permute(2, 0, 1)
                        if k == 0:
                            first_pass_face = curr_chw.clone()
                        else:
                            curr_chw = self._fix_drift_and_texture(
                                curr_chw, first_pass_face
                            )
                            temp_output = curr_chw.permute(1, 2, 0)

                    input_face_affined = temp_output
                    output = torch.clamp(temp_output * 255.0, 0, 255)

        # --- InStyleSwapper Path ---
        elif swapper_model in (
            "InStyleSwapper256 Version A",
            "InStyleSwapper256 Version B",
            "InStyleSwapper256 Version C",
        ):
            version = swapper_model[-1]
            dim_res = dim // 2
            for k in range(itex):
                prev_face = input_face_affined
                temp_output = input_face_affined.clone()
                tile_inputs = []
                tile_outputs = []
                tile_coords = []

                for j in range(dim_res):
                    for i in range(dim_res):
                        tile = input_face_affined[j::dim_res, i::dim_res]
                        t_in = tile.permute(2, 0, 1).contiguous().unsqueeze(0)
                        t_out = torch.empty_like(t_in)
                        tile_inputs.append(t_in)
                        tile_outputs.append(t_out)
                        tile_coords.append((j, i))

                with torch.no_grad():
                    for idx in range(len(tile_inputs)):
                        self.worker.function_worker.run_iss_swapper(
                            tile_inputs[idx], latent, tile_outputs[idx], version
                        )

                if self.worker.models_processor.device_type == "cuda":
                    torch.cuda.current_stream().synchronize()

                for idx, (j, i) in enumerate(tile_coords):
                    res = (
                        tile_inputs[idx]
                        if tile_outputs[idx].sum() < 1.0
                        else tile_outputs[idx]
                    )
                    temp_output[j::dim_res, i::dim_res] = res.squeeze(0).permute(
                        1, 2, 0
                    )

                # --- MODE 2 ---
                if use_mode_2:
                    curr_chw = temp_output.permute(2, 0, 1)
                    if k == 0:
                        first_pass_face = curr_chw.clone()
                    else:
                        curr_chw = self._fix_drift_and_texture(
                            curr_chw, first_pass_face
                        )
                        temp_output = curr_chw.permute(1, 2, 0)

                input_face_affined = temp_output
                output = torch.clamp(temp_output * 255.0, 0, 255)

        # --- SimSwap Path ---
        elif swapper_model == "SimSwap512":
            for k in range(itex):
                # Zero-clone optimization: Model generates a fresh tensor
                prev_face = input_face_affined
                input_face_disc = (
                    input_face_affined.permute(2, 0, 1).unsqueeze(0).contiguous()
                )
                swapper_output = torch.empty(
                    (1, 3, 512, 512),
                    dtype=torch.float32,
                    device=self.worker.models_processor.device,
                ).contiguous()

                self.worker.function_worker.run_swapper_simswap512(
                    input_face_disc, latent, swapper_output
                )
                if self.worker.models_processor.device_type == "cuda":
                    torch.cuda.current_stream().synchronize()

                # Robustness: Fallback to input if output is empty
                if swapper_output.abs().max() < 1e-4:
                    swapper_output = input_face_disc
                swapper_output = swapper_output.squeeze(0)

                # --- MODE 2 ---
                if use_mode_2:
                    if k == 0:
                        first_pass_face = swapper_output.clone()
                    else:
                        swapper_output = self._fix_drift_and_texture(
                            swapper_output, first_pass_face
                        )

                swapper_output_hwc = swapper_output.permute(1, 2, 0)
                input_face_affined = swapper_output_hwc
                output = torch.clamp(swapper_output_hwc * 255.0, 0, 255)

        # --- GhostFace Path ---
        elif swapper_model in self.worker.GHOSTFACE_MODELS:
            for k in range(itex):
                # Performance Optimization: Avoiding redundant VRAM allocations
                prev_face = input_face_affined

                # Model-specific preprocessing (Normalizing to [-1, 1])
                input_face_disc = torch.mul(input_face_affined, 255.0).permute(2, 0, 1)
                input_face_disc = torch.div(input_face_disc.float(), 127.5)
                input_face_disc = (
                    torch.sub(input_face_disc, 1).unsqueeze(0).contiguous()
                )

                swapper_output = torch.empty(
                    (1, 3, 256, 256),
                    dtype=torch.float32,
                    device=self.worker.models_processor.device,
                ).contiguous()

                self.worker.function_worker.run_swapper_ghostface(
                    input_face_disc, latent, swapper_output, swapper_model
                )
                if self.worker.models_processor.device_type == "cuda":
                    torch.cuda.current_stream().synchronize()

                swapper_output = swapper_output[0]
                # FW-BUG-11: use abs().mean() instead of sum() for zero-output heuristic
                if swapper_output.abs().mean() < 0.01:
                    # input_face_affined is [H,W,3] in [0,1]; convert to the
                    # GhostFace [-1,1] CHW range that swapper_output uses.
                    swapper_output = input_face_affined.permute(2, 0, 1) * 2.0 - 1.0

                # --- MODE 2 ---
                if use_mode_2:
                    # Post-processing and drift correction
                    swapper_output = torch.add(torch.mul(swapper_output, 127.5), 127.5)
                    curr_chw = torch.div(swapper_output, 255.0)

                    if k == 0:
                        first_pass_face = curr_chw.clone()
                    else:
                        curr_chw = self._fix_drift_and_texture(
                            curr_chw, first_pass_face
                        )

                    input_face_affined = curr_chw.permute(1, 2, 0)
                    output = torch.clamp(input_face_affined * 255.0, 0, 255)
                else:
                    swapper_output = swapper_output.permute(1, 2, 0)
                    swapper_output = torch.add(torch.mul(swapper_output, 127.5), 127.5)

                    input_face_affined = torch.div(swapper_output, 255.0)
                    output = torch.clamp(swapper_output, 0, 255)

        # --- CSCS Path ---
        elif swapper_model == "CSCS":
            for k in range(itex):
                prev_face = input_face_affined
                input_face_disc = input_face_affined.permute(2, 0, 1)
                input_face_disc = v2.functional.normalize(
                    input_face_disc, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=False
                )
                input_face_disc = input_face_disc.unsqueeze(0).contiguous()

                swapper_output = torch.empty(
                    (1, 3, 256, 256),
                    dtype=torch.float32,
                    device=self.worker.models_processor.device,
                ).contiguous()

                self.worker.function_worker.run_swapper_cscs(
                    input_face_disc, latent, swapper_output
                )
                if self.worker.models_processor.device_type == "cuda":
                    torch.cuda.current_stream().synchronize()

                swapper_output = swapper_output.squeeze(0)
                swapper_output = torch.add(torch.mul(swapper_output, 0.5), 0.5)

                # --- MODE 2 ---
                if use_mode_2:
                    if k == 0:
                        first_pass_face = swapper_output.clone()
                    else:
                        swapper_output = self._fix_drift_and_texture(
                            swapper_output, first_pass_face
                        )

                    input_face_affined = swapper_output.permute(1, 2, 0)
                    output = torch.clamp(input_face_affined * 255.0, 0, 255)
                else:
                    input_face_affined = swapper_output.permute(1, 2, 0)
                    output = torch.clamp(input_face_affined * 255.0, 0, 255)

        # --- DeepFaceLive (DFM) Path ---
        elif swapper_model == "DeepFaceLive (DFM)" and dfm_model:
            # Detect model resolution (Results are cached for performance)
            if hasattr(dfm_model, "_cached_dfm_res"):
                dfm_res = dfm_model._cached_dfm_res
            else:
                dfm_res = 256  # Safe default
                queue = deque([dfm_model])
                visited = set([id(dfm_model)])
                found_res = None
                while queue and not found_res:
                    current = queue.popleft()
                    if hasattr(current, "get_inputs") and callable(current.get_inputs):
                        try:
                            shape = current.get_inputs()[0].shape
                            for s in shape:
                                if isinstance(s, int) and s > 32 and s % 16 == 0:
                                    found_res = s
                                    break
                        except Exception:
                            pass
                    if not found_res and hasattr(current, "__dict__"):
                        for k, v in current.__dict__.items():
                            if id(v) not in visited and not k.startswith("__"):
                                visited.add(id(v))
                                queue.append(v)
                if found_res:
                    dfm_res = found_res
                elif hasattr(dfm_model, "_dfm_filename_fallback"):
                    match = re.search(
                        r"(128|192|224|256|320|384|448|512)",
                        dfm_model._dfm_filename_fallback,
                    )
                    if match:
                        dfm_res = int(match.group(1))
                # Cache it for all future frames
                dfm_model._cached_dfm_res = dfm_res

            # Prepare input for DFM
            if dfm_res != 512:
                dfm_input = v2.functional.resize(
                    original_face_512.float(),
                    [dfm_res, dfm_res],
                    interpolation=v2.InterpolationMode.BILINEAR,
                    antialias=True,
                ).to(original_face_512.dtype)
            else:
                dfm_input = original_face_512.clone()

            # Execute DFM inference with thread-safety lock
            with self.worker.models_processor.dfm_inference_lock:
                if self.worker.models_processor.device_type == "cuda":
                    torch.cuda.current_stream().synchronize()
                out_celeb, _, _ = dfm_model.convert(
                    dfm_input,
                    parameters["DFMAmpMorphSlider"] / 100,
                    rct=parameters["DFMRCTColorToggle"],
                )

            if isinstance(out_celeb, np.ndarray):
                out_celeb = torch.from_numpy(out_celeb).to(original_face_512.device)

            if getattr(out_celeb, "ndim", 0) != 3 or out_celeb.shape[2] != 3:
                print(
                    f"[WARN] DFM output shape unexpected: {out_celeb.shape}. Proceeding anyway."
                )

            # Standardize output scale
            if out_celeb.max() > 2.0:
                out_celeb_float = out_celeb.float() / 255.0
            else:
                out_celeb_float = out_celeb.float()

            # VRAM Optimization: Direct reference assignment
            input_face_affined = out_celeb_float
            prev_face = out_celeb_float
            output = out_celeb_float * 255.0

        # Quality check: Alert if output is abnormally dark (potential VRAM/Model issue)
        if output.abs().max() < 30.0:
            print(
                "[WARN] Swap model output near-zero for face — possible VRAM pressure"
            )

        # Prepare final CHW tensor and resize back to canonical template size (512)
        output = output.permute(2, 0, 1)
        assert self.worker.t512 is not None, (
            "t512 transform must be initialized before swapping"
        )
        swap = self.worker.t512(output)
        return swap, prev_face

    def get_border_mask(self, parameters):
        """Creates the border fade mask based on sliders."""
        border_mask = torch.ones(
            (128, 128), dtype=torch.float32, device=self.worker.models_processor.device
        )
        border_mask = torch.unsqueeze(border_mask, 0)

        if not parameters.get("BordermaskEnableToggle", False):
            return border_mask, border_mask.clone()

        top = parameters["BorderTopSlider"]
        left = parameters["BorderLeftSlider"]
        right = 128 - parameters["BorderRightSlider"]
        bottom = 128 - parameters["BorderBottomSlider"]

        # P3-02: clamp border values instead of assert (assert is disabled under -O)
        left = max(0, min(left, 128))
        right = max(left, min(right, 128))
        top = max(0, min(top, 128))
        bottom = max(top, min(bottom, 128))

        border_mask[:, :top, :] = 0
        border_mask[:, bottom:, :] = 0
        border_mask[:, :, :left] = 0
        border_mask[:, :, right:] = 0

        border_mask_calc = border_mask.clone()
        blur_amount = parameters["BorderBlurSlider"]
        blur_kernel_size = blur_amount * 2 + 1
        if blur_kernel_size > 1:
            sigma_val = max(blur_amount * 0.15 + 0.1, 1e-6)
            gauss = v2.GaussianBlur(blur_kernel_size, sigma=sigma_val)
            border_mask = gauss(border_mask)
        return border_mask, border_mask_calc

    def get_dynamic_side_mask(
        self, yaw_deg, pitch_deg, height, width, device, parameters, kps_5, tform
    ):
        """
        Smart Profile Masking:
        Instead of a blind gradient, this uses the projected eye positions to ensure
        we NEVER mask the eyes.
        """
        mask = torch.ones((1, height, width), dtype=torch.float32, device=device)
        if not parameters.get("ProfileAngleMaskEnableToggle", False):
            return mask

        start_angle = parameters.get("ProfileAngleMaskThresholdSlider", 20)
        max_strength = parameters.get("ProfileAngleMaskStrengthSlider", 100) / 100.0

        if tform is not None:
            kps_proj = tform(kps_5)
            le_x, re_x = kps_proj[0][0], kps_proj[1][0]
        else:
            le_x, re_x = width * 0.35, width * 0.65

        le_x_norm = np.clip(le_x / width, 0.0, 1.0)
        re_x_norm = np.clip(re_x / width, 0.0, 1.0)
        eye_safety_margin = 0.05

        abs_yaw = abs(yaw_deg)
        if abs_yaw > start_angle:
            angle_excess = max(0, abs_yaw - start_angle)
            strength_yaw = min(angle_excess / 45.0, 1.0) * max_strength
            linspace_x = torch.linspace(0, 1, width, device=device).view(1, 1, width)

            if yaw_deg > 0:
                # Looking Right -> Mask Left side
                fade_end = max(0.0, le_x_norm - eye_safety_margin)
                if fade_end > 0.05:
                    grad_yaw = torch.clamp(linspace_x / fade_end, 0, 1)
                    grad_yaw = 1.0 - (1.0 - grad_yaw) * strength_yaw
                    mask = mask * grad_yaw
            else:
                # Looking Left -> Mask Right side
                fade_start = min(1.0, re_x_norm + eye_safety_margin)
                if fade_start < 0.95:
                    grad_yaw = torch.clamp(
                        (linspace_x - fade_start) / (1.0 - fade_start), 0, 1
                    )
                    grad_yaw = 1.0 - grad_yaw
                    mask_r = torch.ones_like(linspace_x)
                    mask_r[linspace_x > fade_start] = 1.0 - (
                        (linspace_x[linspace_x > fade_start] - fade_start)
                        / (1.0 - fade_start)
                    )
                    grad_yaw = 1.0 - (1.0 - mask_r) * strength_yaw
                    mask = mask * grad_yaw
        return mask

    def _apply_restorer_with_auto(
        self,
        swap: torch.Tensor,
        swap2: torch.Tensor,
        swap_original2: torch.Tensor,
        original_face_512: torch.Tensor,
        mask_forcalc_512: torch.Tensor,
        parameters: dict,
        tform_scale: float,
        debug: bool,
        debug_info: dict,
        slot_id: int,
    ) -> torch.Tensor:
        """
        FW-QUAL-01/02: Shared helper for Restoration-2 auto-blend logic.

        Applies auto sharpness-driven alpha blending (or Gaussian blur fallback)
        between `swap_original2` (pre-restorer) and `swap2` (post-restorer).
        When auto mode is disabled, applies a simple weighted blend.

        Args:
            swap:             Current swap tensor (not used directly, but returned as output).
            swap2:            Restorer output tensor.
            swap_original2:   Snapshot of swap *before* restorer was applied.
            original_face_512: Original face tensor (used by face_restorer_auto).
            mask_forcalc_512: Mask tensor for sharpness calculation.
            parameters:       Per-face parameters dict.
            tform_scale:      Similarity-transform scale (used as scale_factor).
            debug:            Whether debug mode is active.
            debug_info:       Mutable dict for debug annotations; key is f'Restore2_{slot_id}'.
            slot_id:          Restorer slot identifier (2 for both callers).

        Returns:
            Updated swap tensor after blending.
        """
        debug_key = f"Restore2_{slot_id}"
        if parameters["FaceRestorerAutoEnable2Toggle"]:
            alpha_restorer2 = float(parameters["FaceRestorerBlend2Slider"]) / 100.0
            adjust_sharpness2 = float(parameters["FaceRestorerAutoSharpAdjust2Slider"])
            scale_factor2 = round(tform_scale, 2)
            automasktoggle2 = parameters["FaceRestorerAutoMask2EnableToggle"]
            automaskadjust2 = parameters[
                "FaceRestorerAutoSharpMask2AdjustDecimalSlider"
            ]
            automaskblur2 = 2
            restore_mask = mask_forcalc_512.clone()

            alpha_auto2, blur_value2 = self.face_restorer_auto(
                original_face_512.clone(),
                swap_original2.clone(),
                swap2,
                alpha_restorer2,
                adjust_sharpness2,
                scale_factor2,
                debug,
                restore_mask,
                automasktoggle2,
                automaskadjust2,
                automaskblur2,
            )

            if blur_value2 > 0:
                kernel_size = 2 * blur_value2 + 1
                sigma = blur_value2 * 0.1
                gaussian_blur = v2.GaussianBlur(kernel_size=kernel_size, sigma=sigma)
                swap = gaussian_blur(swap_original2)
                debug_info[debug_key] = f": {-blur_value2:.2f}"
            elif isinstance(alpha_auto2, torch.Tensor):
                swap = swap2 * alpha_auto2 + swap_original2 * (1 - alpha_auto2)
            elif alpha_auto2 != 0:
                swap = swap2 * alpha_auto2 + swap_original2 * (1 - alpha_auto2)
                if debug:
                    debug_info[debug_key] = f": {alpha_auto2 * 100:.2f}"
            else:
                swap = swap_original2
                if debug:
                    debug_info[debug_key] = f": {alpha_auto2 * 100:.2f}"
        else:
            alpha_restorer2 = float(parameters["FaceRestorerBlend2Slider"]) / 100.0
            swap = (
                torch.lerp(swap_original2.float(), swap2.float(), alpha_restorer2)
                .to(swap2.dtype)
                .contiguous()
            )
        return swap

    def _detect_mouth_action_score(
        self,
        face_bbox: "np.ndarray | None" = None,
        vr_crop_chw: "torch.Tensor | np.ndarray | None" = None,
    ) -> "float | None":
        """Run the mouth action detector on a face-region crop.

        For standard mode pass ``face_bbox`` ([x1,y1,x2,y2]) — the function crops
        ``self.frame`` at 2× scale around the bbox so the head and mouth surroundings
        are included without flooding the model with irrelevant background.

        For VR180 mode pass ``vr_crop_chw`` (the 512×512 perspective crop tensor) —
        that image is already a focused face region and is used directly.

        Falls back to the full frame when neither argument is supplied.

        Returns a positive float on detection, or None (no detection) so that the
        MouthOpennessState occlusion-timeout can bridge short missed-frame gaps.
        """
        from app.processors.mouth_action_detector import MouthActionDetector

        detector = MouthActionDetector.get()
        if not detector.available:
            if not getattr(self, "_mouth_action_detector_warned", False):
                err = detector.load_error or "unknown error"
                print(f"[WARN] Mouth action detector unavailable: {err}")
                self._mouth_action_detector_warned = True
            return None

        if vr_crop_chw is not None:
            # VR perspective crop — already CHW, convert to numpy if needed
            if isinstance(vr_crop_chw, torch.Tensor):
                img_chw_np = vr_crop_chw.cpu().numpy().astype(np.uint8)
            else:
                img_chw_np = np.asarray(vr_crop_chw, dtype=np.uint8)
        elif face_bbox is not None and self.worker.frame is not None:
            # Standard mode — crop self.frame (HWC) at 2× scale around the face bbox
            frame_hwc = self.worker.frame  # HWC uint8 RGB numpy
            fh, fw = frame_hwc.shape[:2]
            x1, y1, x2, y2 = (
                float(face_bbox[0]),
                float(face_bbox[1]),
                float(face_bbox[2]),
                float(face_bbox[3]),
            )

            # FW-MOUTH-1: For sub-512px input frames, _process_frame_standard
            # upscales the working tensor and scales the bboxes UP to match.
            # `self.frame` is the ORIGINAL (un-upscaled) numpy frame. Using
            # upscaled-space bbox coordinates against the original frame produces
            # an out-of-bounds crop → mouth-action detector returns None →
            # auto-mouth never fires for low-resolution videos.
            #
            # Detect the mismatch (bbox extending well past the frame's actual
            # dimensions) and rescale the bbox back to original-frame space.
            _bbox_max = max(x2, y2)
            if _bbox_max > max(fw, fh) + 8 and (fw < 512 or fh < 512):
                if fw <= fh:
                    _new_w, _new_h = 512, int(512 * fh / fw)
                else:
                    _new_h, _new_w = 512, int(512 * fw / fh)
                _ratio_w_down = fw / float(_new_w)
                _ratio_h_down = fh / float(_new_h)
                x1 *= _ratio_w_down
                x2 *= _ratio_w_down
                y1 *= _ratio_h_down
                y2 *= _ratio_h_down
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            half_w, half_h = x2 - x1, y2 - y1
            crop_x1, crop_y1 = max(0, int(cx - half_w)), max(0, int(cy - half_h))
            crop_x2, crop_y2 = min(fw, int(cx + half_w)), min(fh, int(cy + half_h))
            crop_hwc = frame_hwc[crop_y1:crop_y2, crop_x1:crop_x2]
            if crop_hwc.size == 0:
                return None
            img_chw_np = np.transpose(crop_hwc, (2, 0, 1))
        else:
            # Fallback — full frame
            img_chw_np = np.transpose(self.worker.frame, (2, 0, 1))

        raw_score = detector.score(img_chw_np)
        # Return None on no detection so the state machine uses the occlusion grace period
        return raw_score if raw_score > 0.0 else None

    def _apply_auto_mouth(
        self,
        params: dict,
        target_fb: Any,
        face_bbox: "np.ndarray | None" = None,
        vr_crop_chw: "torch.Tensor | np.ndarray | None" = None,
    ) -> dict:
        """Check auto-mouth state and, if active, return a modified params dict.

        'Expression Restorer' mode: takes precedence over the face expression
        restorer's lip settings while preserving the user's eye/brow/general
        restorer configuration. When auto-mouth deactivates, the original restorer
        settings resume automatically for lips.

        'Face Parser Only' mode: applies face-parser mouth masks only, no
        expression transfer — faster, no mouth-position update.

        Returns *params* unchanged (same object) when disabled or not triggered.
        """
        if not params.get("AutoMouthExpressionEnableToggle", False):
            return params

        from app.processors.mouth_openness import MouthOpennessState

        _alpha = params.get("AutoMouthEMAAlphaDecimalSlider", 0.65)
        _threshold = params.get("AutoMouthOpenThresholdDecimalSlider", 0.50)
        # None = no detection; triggers occlusion grace period in state machine.
        _ratio: "float | None" = self._detect_mouth_action_score(
            face_bbox=face_bbox, vr_crop_chw=vr_crop_chw
        )

        _state: "MouthOpennessState | None" = getattr(
            target_fb, "mouth_openness_state", None
        )
        if _state is None:
            target_fb.mouth_openness_state = MouthOpennessState()
            _state = target_fb.mouth_openness_state

        # FW-MOUTH-2: When the worker is servicing a stopped-frame preview
        # (single-frame mode), every refresh re-evaluates the SAME underlying
        # frame. None-ratios from transient bbox/setting toggles must not accrue
        # an occlusion penalty — otherwise toggling color transfer on a stopped
        # frame slowly decays the EMA past the deactivate threshold and the user
        # cannot re-enable auto-mouth without resuming playback.
        _single_frame_mode = bool(getattr(self.worker, "is_single_frame", False))
        _auto_active, _ema_value = _state.update(
            _ratio, _alpha, _threshold, single_frame_mode=_single_frame_mode
        )

        _exclude_upper_teeth = bool(
            params.get("AutoMouthExcludeUpperTeethToggle", False)
        )
        _was_auto_active = bool(getattr(target_fb, "_auto_mouth_prev_active", False))
        target_fb._auto_mouth_prev_active = _auto_active

        if not _auto_active:
            return params

        _base_strength = params.get("AutoMouthExpressionStrengthDecimalSlider", 1.00)
        _normalize = params.get("AutoMouthNormalizeLipsToggle", True)
        _restore_mode = params.get(
            "AutoMouthRestoreModeSelection", "Expression Restorer"
        )
        # Proportional strength ramp: smooth fade-in from threshold to threshold+ramp.
        _ramp_range = max(_threshold * 0.5, 0.04)
        _proportion = min(1.0, max(0.0, (_ema_value - _threshold) / _ramp_range))
        _strength = _base_strength * _proportion

        # Skip entirely when strength is negligible — avoids a full warp-decode cycle.
        # Still apply FaceParser overrides (teeth exclusion) even at low strength.
        if _strength < 0.01:
            if _exclude_upper_teeth:
                _p = dict(params)
                _p["FaceParserEnableToggle"] = True
                _p["AutoMouthUpperTeethExcludeActive"] = True
                return _p
            return params

        _p = dict(params)

        # --- Face-parser mouth override (applied in both modes) ---
        _mouth_val = int(params.get("AutoMouthMouthParserSlider", 1))
        _upper_val = int(params.get("AutoMouthUpperLipParserSlider", 3))
        _lower_val = int(params.get("AutoMouthLowerLipParserSlider", 17))
        if _mouth_val > 0 or _upper_val > 0 or _lower_val > 0 or _exclude_upper_teeth:
            _p["FaceParserEnableToggle"] = True
        _p["MouthParserSlider"] = _mouth_val
        _p["UpperLipParserSlider"] = _upper_val
        _p["LowerLipParserSlider"] = _lower_val
        _p["AutoMouthUpperTeethExcludeActive"] = _exclude_upper_teeth

        if _restore_mode == "Face Parser Only":
            # Mask-only mode: do not touch the expression restorer.
            return _p

        # --- Expression Restorer mode ---
        # Auto mouth takes precedence over restorer lip settings; eyes/brows/general
        # are preserved from the user's existing restorer configuration.
        _user_enabled = params.get("FaceExpressionEnableBothToggle", False)
        _user_mode = params.get("FaceExpressionModeSelection", "Advanced")
        _user_region = str(params.get("FaceExpressionAnimationRegionSelection", "all"))

        _p["FaceExpressionEnableBothToggle"] = True
        _p["FaceExpressionBeforeTypeSelection"] = "Beginning"

        if _user_enabled and _user_mode == "Advanced":
            # Keep user's Advanced-mode eye/brow/general settings; force lips only.
            _p["FaceExpressionModeSelection"] = "Advanced"
            _p["FaceExpressionLipsToggle"] = True
            _p["FaceExpressionFriendlyFactorLipsDecimalSlider"] = _strength
            _p["FaceExpressionRelativeLipsToggle"] = True
            _p["FaceExpressionRetargetingLipsBothEnableToggle"] = _normalize
            _p["FaceExpressionRetargetingLipsMultiplierBothDecimalSlider"] = 1.0
            # Eyes, brows, general keep user values from _p (shallow copy of params).

        elif _user_enabled and _user_mode == "Simple":
            # Convert Simple → Advanced, preserving eye motion if user had 'all' region.
            _had_eyes = "eyes" in _user_region or "all" in _user_region
            _user_factor = float(
                params.get("FaceExpressionFriendlyFactorDecimalSlider", 1.0)
            )
            _p["FaceExpressionModeSelection"] = "Advanced"
            _p["FaceExpressionLipsToggle"] = True
            _p["FaceExpressionFriendlyFactorLipsDecimalSlider"] = _strength
            _p["FaceExpressionRelativeLipsToggle"] = True
            _p["FaceExpressionRetargetingLipsBothEnableToggle"] = _normalize
            _p["FaceExpressionRetargetingLipsMultiplierBothDecimalSlider"] = 1.0
            # Carry user's Simple eye factor over if they had eye motion.
            _p["FaceExpressionEyesToggle"] = _had_eyes
            if _had_eyes:
                _p["FaceExpressionFriendlyFactorEyesDecimalSlider"] = _user_factor
                _p["FaceExpressionRelativeEyesToggle"] = True
            _p["FaceExpressionBrowsToggle"] = False
            _p["FaceExpressionGeneralToggle"] = False

        else:
            # Restorer was disabled: enable Simple mode for lips (or region from UI).
            _region = params.get("AutoMouthAnimationRegionSelection", "lips")
            _p["FaceExpressionModeSelection"] = "Simple"
            _p["FaceExpressionAnimationRegionSelection"] = _region
            _p["FaceExpressionFriendlyFactorDecimalSlider"] = _strength
            _p["FaceExpressionNormalizeLipsEnableToggle"] = _normalize
            _p["FaceExpressionNormalizeLipsThresholdDecimalSlider"] = 0.03
            _p["FaceExpressionNeutralDecimalSlider"] = 1.0
            _p["FaceExpressionLipsToggle"] = False
            _p["FaceExpressionEyesToggle"] = False
            _p["FaceExpressionBrowsToggle"] = False
            _p["FaceExpressionGeneralToggle"] = False

        return _p

    def swap_core(
        self,
        img: torch.Tensor,
        kps_5: np.ndarray,
        kps: np.ndarray | None = None,  # FW-ROBUST-06: changed default False -> None
        kps_203: np.ndarray | None = None,  # NEW: Dedicated 203 points
        s_e: np.ndarray | None = None,
        t_e: np.ndarray | None = None,
        parameters: dict[str, Any] | None = None,
        control: dict[str, Any] | None = None,
        dfm_model_name: str | None = None,
        is_perspective_crop: bool = False,
        kv_map: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """
        Core function for face swapping. Handles:
        1. Alignment and Scaling.
        2. Swapping (Model inference).
        3. Blending and Masking (XSeg, Occluder, Texture Transfer).
        4. Color Correction.
        5. Restoration (GFPGAN/CodeFormer).
        6. Reverse alignment (Untransform).
        """
        valid_s_e = s_e if isinstance(s_e, np.ndarray) else None
        valid_t_e = t_e if isinstance(t_e, np.ndarray) else None
        parameters = parameters if parameters is not None else {}
        control = control if control is not None else {}
        swapper_model = parameters["SwapModelSelection"]
        itex = 1  # FW-BUG-10: default before any branching to prevent NameError

        # OPTIMIZED: Lightweight functional resize wrapper to prevent VRAM fragmentation
        # and GC stutters caused by inline v2.Resize class instantiation.
        def _resize_func(
            tensor: torch.Tensor, target_shape: tuple[int, int], is_mask: bool = True
        ) -> torch.Tensor:
            return v2.functional.resize(
                tensor,
                [target_shape[0], target_shape[1]],
                interpolation=v2.InterpolationMode.BILINEAR,
                antialias=not is_mask,
            )

        debug = False
        debug_info: dict[str, str] = {}

        tform = self.worker.get_face_similarity_tform(swapper_model, kps_5)
        # OPTIMIZATION: Transform full-frame smoothed keypoints to the 512x512 crop space
        kps_all_crop = None
        if (
            kps_203 is not None and len(kps_203) == 203
        ):  # Use the dedicated 203 variable
            raw_kps_crop = tform(kps_203)
            kps_all_crop = np.array(raw_kps_crop, dtype=np.float32)

        # FW-PERF-5: use promoted instance-attribute transforms (initialized in
        # set_scaling_transforms) instead of constructing new objects each call
        t512_mask = self.worker.t512_mask
        t128_mask = self.worker.t128_mask
        assert t512_mask is not None, (
            "t512_mask must be initialized via set_scaling_transforms before swap_core"
        )
        assert t128_mask is not None, (
            "t128_mask must be initialized via set_scaling_transforms before swap_core"
        )

        _face_interp = (
            "bicubic"
            if parameters.get("FaceAlignmentInterpolation", "Bilinear") == "Bicubic"
            else "bilinear"
        )
        original_face_512, original_face_384, original_face_256, original_face_128 = (
            self.worker.get_transformed_and_scaled_faces(
                tform, img, interp_mode=_face_interp
            )
        )
        original_faces = (
            original_face_512,
            original_face_384,
            original_face_256,
            original_face_128,
        )
        swap = original_face_512
        # Initialise prev_face to the normalised original face so that the
        # StrengthEnableToggle blend at the end of swap_core always has a valid
        # tensor, even when get_swapped_and_prev_face is skipped (e.g. DFM
        # selected but no model file chosen, or input_face_affined is None).
        prev_face = torch.div(original_face_512.float(), 255.0).permute(1, 2, 0)

        # --- SWAPPING INFERENCE ---
        if valid_s_e is not None or (
            swapper_model == "DeepFaceLive (DFM)" and dfm_model_name
        ):
            input_face_affined, dfm_model_instance, dim, latent = (
                self.get_affined_face_dim_and_swapping_latents(
                    original_faces,
                    swapper_model,
                    dfm_model_name,
                    valid_s_e,
                    valid_t_e,
                    parameters,
                    debug,
                    tform,
                )
            )

            # FW-BUG-03: guard input_face_affined is None (latent computation failed)
            if input_face_affined is None:
                swap = original_face_512
            # skip to mask section — latent computation failed, use original face
            else:
                # Optional Face Scaling adjustment
                if parameters["FaceAdjEnableToggle"]:
                    scale_val = 1.0 + (parameters["FaceScaleAmountSlider"] / 100.0)
                    orig_dtype = input_face_affined.dtype

                    # OPTIMIZED: Replaced F.grid_sample with v2.resize (with Antialiasing) + crop/pad.
                    # Downscaling with grid_sample lacks an anti-aliasing filter, causing aliasing/blur.
                    # v2.functional.resize with antialias=True applies proper filtering before scaling.
                    if scale_val != 1.0:
                        channels, h, w = input_face_affined.shape
                        new_h = max(1, int(h * scale_val))
                        new_w = max(1, int(w * scale_val))

                        # Resize applying high-quality Bicubic interpolation and anti-aliasing
                        resized = v2.functional.resize(
                            input_face_affined,
                            [new_h, new_w],
                            interpolation=v2.InterpolationMode.BICUBIC,
                            antialias=True,
                        )

                        if scale_val < 1.0:
                            # Shrinking: Pad edges with 0 (black) to match original bounds
                            pad_left = (w - new_w) // 2
                            pad_right = w - new_w - pad_left
                            pad_top = (h - new_h) // 2
                            pad_bottom = h - new_h - pad_top
                            input_face_affined = F.pad(
                                resized,
                                (pad_left, pad_right, pad_top, pad_bottom),
                                mode="constant",
                                value=0,
                            )
                        else:
                            # Enlarging: Center crop back to original bounds
                            crop_top = (new_h - h) // 2
                            crop_left = (new_w - w) // 2
                            input_face_affined = v2.functional.crop(
                                resized, crop_top, crop_left, h, w
                            )

                        # Torchvision native bicubic respects uint8 bounds automatically,
                        # but cast back to ensure safety if float-converted downstream.
                        if orig_dtype == torch.uint8:
                            input_face_affined = input_face_affined.to(orig_dtype)

                itex = 1
                if parameters["StrengthEnableToggle"]:
                    itex = ceil(parameters["StrengthAmountSlider"] / 100.0)

                output_size = int(128 * dim)
                output = torch.zeros(
                    (output_size, output_size, 3),
                    dtype=torch.float32,
                    device=self.worker.models_processor.device,
                )
                input_face_affined = input_face_affined.permute(1, 2, 0).contiguous()
                input_face_affined = torch.div(input_face_affined, 255.0)

                swap, prev_face = self.get_swapped_and_prev_face(
                    output,
                    input_face_affined,
                    original_face_512,
                    latent,
                    itex,
                    dim,
                    swapper_model,
                    dfm_model_instance,
                    parameters,
                )
        else:
            swap = original_face_512
            if parameters["StrengthEnableToggle"]:
                itex = ceil(parameters["StrengthAmountSlider"] / 100.0)
                prev_face = torch.div(swap, 255.0)
                prev_face = prev_face.permute(1, 2, 0)

        if parameters["StrengthEnableToggle"]:
            if itex == 0:
                swap = original_face_512.clone()
            else:
                alpha = np.mod(parameters["StrengthAmountSlider"], 100) * 0.01
                if alpha == 0:
                    alpha = 1
                prev_face = torch.mul(prev_face, 255)
                prev_face = torch.clamp(prev_face, 0, 255)
                prev_face = prev_face.permute(2, 0, 1)
                if prev_face.shape[-1] != swap.shape[-1]:
                    # Using functional resize for RGB buffer (antialias=True is fine here)
                    prev_face = _resize_func(
                        prev_face, (swap.shape[-2], swap.shape[-1]), is_mask=False
                    )
                swap = (
                    torch.lerp(prev_face.float(), swap.float(), alpha)
                    .to(swap.dtype)
                    .contiguous()
                )

        # --- DYNAMIC MASKS INITIALIZATION ---
        current_swap_h, current_swap_w = swap.shape[1], swap.shape[2]
        yaw_deg, pitch_deg = faceutil.calc_face_yaw_pitch(kps_5)
        side_mask = self.get_dynamic_side_mask(
            yaw_deg,
            pitch_deg,
            current_swap_h,
            current_swap_w,
            self.worker.models_processor.device,
            parameters,
            kps_5,
            tform,
        )

        # FW-PERF-09: skip get_border_mask entirely when the toggle is off
        if parameters.get("BordermaskEnableToggle", False):
            border_mask, border_mask_calc = self.get_border_mask(parameters)
            if (
                border_mask.shape[1] != current_swap_h
                or border_mask.shape[2] != current_swap_w
            ):
                border_mask = _resize_func(
                    border_mask, (current_swap_h, current_swap_w), is_mask=True
                )
                border_mask_calc = _resize_func(
                    border_mask_calc, (current_swap_h, current_swap_w), is_mask=True
                )
            border_mask = border_mask * side_mask
            border_mask_calc = border_mask_calc * side_mask
        else:
            border_mask = side_mask
            border_mask_calc = side_mask

        # BLEND MASKS (Soft edges, Alpha compositing)
        swap_mask = torch.ones(
            (1, current_swap_h, current_swap_w),
            dtype=torch.float32,
            device=self.worker.models_processor.device,
        )
        swap_mask_noFP = border_mask.clone()
        # CORE STATS MASKS (Hard edges, inner face isolation only)
        core_stats_mask = torch.ones(
            (1, current_swap_h, current_swap_w),
            dtype=torch.float32,
            device=self.worker.models_processor.device,
        )

        # Removed shared tensor 'BgExclude' which was causing state bleeding
        # (Race Conditions) between unconnected features. Explicit allocation is safer.
        base_shape = (1, 512, 512)
        diff_mask = torch.ones(
            base_shape, dtype=torch.float32, device=self.worker.models_processor.device
        )
        texture_mask_view = torch.ones(
            base_shape, dtype=torch.float32, device=self.worker.models_processor.device
        )
        texture_exclude_512 = torch.ones(
            base_shape, dtype=torch.float32, device=self.worker.models_processor.device
        )

        # Legacy pointers mapped to our clean core mask to prevent crashing the rest of the pipeline
        calc_mask = core_stats_mask
        calc_mask_dill = core_stats_mask
        mask_forcalc_512 = core_stats_mask

        M_ref = cast(np.ndarray, tform.params)[0:2]
        ones_column_ref = np.ones((kps_5.shape[0], 1), dtype=np.float32)
        kps_ref = np.hstack([kps_5, ones_column_ref]) @ M_ref.T

        swap = torch.clamp(swap, 0.0, 255.0)

        # --- FACE EDITING (Beginning) ---
        # Expression Restorer beginning
        if (
            parameters["FaceExpressionEnableBothToggle"]
            and (
                parameters["FaceExpressionLipsToggle"]
                or parameters["FaceExpressionEyesToggle"]
                or parameters["FaceExpressionBrowsToggle"]
                or parameters["FaceExpressionGeneralToggle"]
                or parameters.get("FaceExpressionModeSelection", "Advanced")
                in ("Simple", "Recast")
            )
            and parameters["FaceExpressionBeforeTypeSelection"] == "Beginning"
        ):
            if parameters.get("FaceExpressionModeSelection", "Advanced") == "Recast":
                swap = self.worker.function_worker.apply_perform_recast(
                    original_face_512,
                    swap,
                    cast(dict, parameters),
                    cast(dict, control),
                    driving_kps=kps_all_crop,
                )
            else:
                swap = self.worker.function_worker.apply_face_expression_restorer(
                    original_face_512,
                    swap,
                    cast(dict, parameters),
                    cast(dict, control),
                    driving_kps=kps_all_crop,
                )

        # Face editor beginning
        if (
            parameters["FaceEditorEnableToggle"]
            and self.worker.local_control_state_from_feeder.get("edit_enabled", True)
            and parameters["FaceEditorBeforeTypeSelection"] == "Beginning"
        ):
            editor_mask = swap_mask.clone()
            swap = (
                torch.lerp(original_face_512.float(), swap.float(), editor_mask)
                .to(swap.dtype)
                .contiguous()
            )
            swap = self.worker.function_worker.swap_edit_face_core(
                swap, swap, parameters, control
            )

        # First Denoiser pass - Before Restorers
        if control.get("DenoiserUNetEnableBeforeRestorersToggle", False):
            swap = self._apply_denoiser_pass(
                swap,
                control,
                "Before",
                kv_map,
                color_mask=mask_forcalc_512,
                blend_mask=swap_mask,
            )

        # --- MOUTH ENHANCEMENT & ALIGNMENT (PRE-RESTORER) ---
        paste_after_restorer = parameters.get("MouthParserStretchAfterToggle", False)
        if not paste_after_restorer:
            mouth_overlay_pkg = None
            if hasattr(self.worker.function_worker, "get_mouth_overlay"):
                mouth_overlay_pkg = self.worker.function_worker.get_mouth_overlay(
                    swap, original_face_512, parameters
                )

            if mouth_overlay_pkg is not None:
                overlay_rgb, overlay_mask = mouth_overlay_pkg
                if overlay_rgb is not None and overlay_mask is not None:
                    if overlay_rgb.shape[-1] != swap.shape[-1]:
                        overlay_rgb = _resize_func(
                            overlay_rgb, (swap.shape[-2], swap.shape[-1]), is_mask=False
                        )
                        overlay_mask = _resize_func(
                            overlay_mask.unsqueeze(0),
                            (swap.shape[-2], swap.shape[-1]),
                            is_mask=True,
                        ).squeeze(0)
                    swap = swap * (1.0 - overlay_mask) + overlay_rgb * overlay_mask

        # --- RESTORATION 1 ---
        # FW-PERF-11: defer clone until we know it is needed (lazy snapshot)
        swap_original = None
        if parameters["FaceRestorerEnableToggle"]:
            # FW-PERF-11: clone only when the restorer will actually run
            swap_original = swap.clone()
            swap_restorecalc = self.worker.function_worker.apply_facerestorer(
                swap,
                parameters["FaceRestorerDetTypeSelection"],
                parameters["FaceRestorerTypeSelection"],
                parameters["FaceRestorerBlendSlider"],
                parameters["FaceFidelityWeightDecimalSlider"],
                control["DetectorScoreSlider"],
                kps_ref,
                slot_id=1,
            )
        else:
            swap_restorecalc = swap.clone()

        # Occluder
        if parameters["OccluderEnableToggle"]:
            mask = self.worker.function_worker.apply_occlusion(
                original_face_256,
                parameters["OccluderSizeSlider"],
                parameters=parameters,
                original_face_512=swap_restorecalc,
            )
            if mask.shape[-1] != swap_mask.shape[-1]:
                mask = _resize_func(
                    mask, (swap_mask.shape[-2], swap_mask.shape[-1]), is_mask=True
                )
            swap_mask.mul_(mask)

            gauss = v2.GaussianBlur(
                parameters["OccluderXSegBlurSlider"] * 2 + 1,
                (parameters["OccluderXSegBlurSlider"] + 1) * 0.2,
            )
            swap_mask = gauss(swap_mask)

            if swap_mask_noFP.shape[-1] != swap_mask.shape[-1]:
                swap_mask_noFP = _resize_func(
                    swap_mask_noFP,
                    (swap_mask.shape[-2], swap_mask.shape[-1]),
                    is_mask=True,
                )
            swap_mask_noFP.mul_(swap_mask)

        # --- MASKS (Parser / CLIPs / Restore) ---
        need_any_parser = (
            parameters.get("FaceParserEnableToggle", False)
            or (
                parameters.get("DFLXSegEnableToggle", False)
                and (
                    (
                        parameters.get("XSegMouthEnableToggle", False)
                        and parameters.get("DFLXSegSizeSlider", 0)
                        != parameters.get("DFLXSeg2SizeSlider", 0)
                    )
                    or parameters.get("XSegExcludeInnerMouthToggle", False)
                )
            )
            or (
                (
                    parameters.get("TransferTextureEnableToggle", False)
                    or parameters.get("DifferencingEnableToggle", False)
                )
                and parameters.get("ExcludeMaskEnableToggle", False)
            )
        )

        FaceParser_mask = None
        mouth_512 = None
        mouth_debug_512 = None
        mouth_debug_teeth_512 = None
        inner_mouth_protection_512 = None

        if need_any_parser:
            out = self.worker.function_worker.process_masks_and_masks(
                swap_restorecalc, original_face_512, parameters, control
            )
            if not parameters.get("FaceParserEndToggle", False):
                FaceParser_mask = out.get("FaceParser_mask", None)
            texture_exclude_512 = out.get("texture_mask", texture_exclude_512)
            mouth_512 = out.get("mouth", None)
            mouth_debug_512 = out.get("mouth_debug", None)
            mouth_debug_teeth_512 = out.get("mouth_debug_teeth", None)
            inner_mouth_protection_512 = out.get("inner_mouth_protection", None)

        if FaceParser_mask is not None:
            if FaceParser_mask.shape[-1] != swap_mask.shape[-1]:
                FaceParser_mask = _resize_func(
                    FaceParser_mask,
                    (swap_mask.shape[-2], swap_mask.shape[-1]),
                    is_mask=True,
                )
            swap_mask.mul_(FaceParser_mask)

        # CLIPs
        if parameters.get("ClipEnableToggle", False):
            mask_clip = self.worker.function_worker.run_CLIPs(
                original_face_512,
                parameters["ClipText"],
                parameters["ClipAmountSlider"],
            )
            if mask_clip.shape[-1] != swap_mask.shape[-1]:
                mask_clip = _resize_func(
                    mask_clip, (swap_mask.shape[-2], swap_mask.shape[-1]), is_mask=True
                )
            swap_mask.mul_(mask_clip)
            if swap_mask_noFP.shape[-1] != mask_clip.shape[-1]:
                swap_mask_noFP = _resize_func(
                    swap_mask_noFP,
                    (mask_clip.shape[-2], mask_clip.shape[-1]),
                    is_mask=True,
                )
            swap_mask_noFP.mul_(mask_clip)

        # Restore Eyes/Mouth
        if parameters.get("RestoreMouthEnableToggle", False) or parameters.get(
            "RestoreEyesEnableToggle", False
        ):
            M = cast(np.ndarray, tform.params)[0:2]
            ones_column = np.ones((kps_5.shape[0], 1), dtype=np.float32)
            dst_kps_5 = np.hstack([kps_5, ones_column]) @ M.T

            img_swap_mask = torch.ones(
                (1, 512, 512),
                dtype=torch.float32,
                device=self.worker.models_processor.device,
            )
            img_orig_mask = torch.zeros(
                (1, 512, 512),
                dtype=torch.float32,
                device=self.worker.models_processor.device,
            )

            if parameters.get("RestoreMouthEnableToggle", False):
                img_swap_mask = self.worker.function_worker.restore_mouth(
                    img_orig_mask,
                    img_swap_mask,
                    dst_kps_5,
                    parameters["RestoreMouthBlendAmountSlider"] / 100.0,
                    parameters["RestoreMouthFeatherBlendSlider"],
                    parameters["RestoreMouthSizeFactorSlider"] / 100.0,
                    parameters["RestoreXMouthRadiusFactorDecimalSlider"],
                    parameters["RestoreYMouthRadiusFactorDecimalSlider"],
                    parameters["RestoreXMouthOffsetSlider"],
                    parameters["RestoreYMouthOffsetSlider"],
                ).clamp(0, 1)

            if parameters.get("RestoreEyesEnableToggle", False):
                img_swap_mask = self.worker.function_worker.restore_eyes(
                    img_orig_mask,
                    img_swap_mask,
                    dst_kps_5,
                    parameters["RestoreEyesBlendAmountSlider"] / 100.0,
                    parameters["RestoreEyesFeatherBlendSlider"],
                    parameters["RestoreEyesSizeFactorDecimalSlider"],
                    parameters["RestoreXEyesRadiusFactorDecimalSlider"],
                    parameters["RestoreYEyesRadiusFactorDecimalSlider"],
                    parameters["RestoreXEyesOffsetSlider"],
                    parameters["RestoreYEyesOffsetSlider"],
                    parameters["RestoreEyesSpacingOffsetSlider"],
                ).clamp(0, 1)

            if parameters.get("RestoreEyesMouthBlurSlider", 0) > 0:
                b = parameters["RestoreEyesMouthBlurSlider"]
                gauss = v2.GaussianBlur(b * 2 + 1, (b + 1) * 0.2)
                img_swap_mask = gauss(img_swap_mask)

            if img_swap_mask.shape[-1] != swap_mask.shape[-1]:
                mask_resized = _resize_func(
                    img_swap_mask,
                    (swap_mask.shape[-2], swap_mask.shape[-1]),
                    is_mask=True,
                )
            else:
                mask_resized = img_swap_mask
            swap_mask = swap_mask * mask_resized

        # --- DFL XSeg ---
        # FW-PERF-5: use promoted instance-attribute transform
        t256_near = self.worker.t256_near
        assert t256_near is not None, (
            "t256_near must be initialized via set_scaling_transforms"
        )

        if parameters.get("DFLXSegEnableToggle", False):
            img_xseg_256 = t256_near(original_face_512)
            mouth_256 = None
            inner_mouth_protection_256 = None
            if (
                parameters.get("XSegMouthEnableToggle", False)
                and parameters.get("DFLXSegSizeSlider", 0)
                != parameters.get("DFLXSeg2SizeSlider", 0)
                and mouth_512 is not None
            ):
                mouth_256 = t256_near(mouth_512.unsqueeze(0))
            if (
                parameters.get("XSegExcludeInnerMouthToggle", False)
                and inner_mouth_protection_512 is not None
            ):
                inner_mouth_protection_256 = t256_near(
                    inner_mouth_protection_512.unsqueeze(0)
                ).squeeze(0)

            img_mask_256, mask_forcalc_256, mask_forcalc_dill_256, outpred_noFP_256 = (
                self.worker.function_worker.apply_dfl_xseg(
                    img_xseg_256,
                    -parameters["DFLXSegSizeSlider"],
                    mouth_256 if mouth_256 is not None else 0,
                    parameters,
                    inner_mouth_mask=inner_mouth_protection_256,
                )
            )

            # 1. Update Blend Masks (swap_mask)
            if img_mask_256.shape[-1] != swap_mask.shape[-1]:
                img_mask_res = _resize_func(
                    img_mask_256,
                    (swap_mask.shape[-2], swap_mask.shape[-1]),
                    is_mask=True,
                )
                outpred_noFP_res = _resize_func(
                    outpred_noFP_256,
                    (swap_mask.shape[-2], swap_mask.shape[-1]),
                    is_mask=True,
                )
            else:
                img_mask_res = img_mask_256
                outpred_noFP_res = outpred_noFP_256

            if swap_mask_noFP.shape[-1] != outpred_noFP_res.shape[-1]:
                swap_mask_noFP = _resize_func(
                    swap_mask_noFP,
                    (outpred_noFP_res.shape[-2], outpred_noFP_res.shape[-1]),
                    is_mask=True,
                )

            # The standard Occluder blurs the global swap_mask, softening the harsh
            # border_mask edges. DFLXSeg blurs its mask internally, leaving the base
            # swap_mask edges razor-sharp. We apply the blur here to soften the boundaries
            # BEFORE multiplying XSeg, avoiding double-blurring the XSeg internal edges.
            if not parameters.get("OccluderEnableToggle", False):
                blur_amount = parameters.get("OccluderXSegBlurSlider", 0)
                if blur_amount > 0:
                    kernel_size = blur_amount * 2 + 1
                    sigma = (blur_amount + 1) * 0.2
                    gauss_op = v2.GaussianBlur(kernel_size, sigma)
                    swap_mask = gauss_op(swap_mask)
                    swap_mask_noFP = gauss_op(swap_mask_noFP)

            # apply_dfl_xseg returns inverted masks (0=Face, 1=BG).
            # We multiply by (1 - mask) to carve out the background.
            swap_mask_noFP.mul_(1.0 - outpred_noFP_res)
            swap_mask.mul_(1.0 - img_mask_res)

            # 2. Update Core Stats Masks (Strictly for statistics/color/texture)
            mask_forcalc_512 = t512_mask(mask_forcalc_256)
            mask_forcalc_dill_512 = t512_mask(mask_forcalc_dill_256)

            # Re-invert XSeg core masks so 1 = Face, 0 = BG
            xseg_core = 1.0 - mask_forcalc_512
            xseg_core_dill = 1.0 - mask_forcalc_dill_512

            # INTERSECT XSeg with our global core_stats_mask.
            # This ensures we never expand beyond the valid face area.
            calc_mask = core_stats_mask * xseg_core
            calc_mask_dill = core_stats_mask * xseg_core_dill
            mask_forcalc_512 = calc_mask.clone()

        else:
            # If XSeg is off, DO NOT clone swap_mask (which has blurred edges).
            # Use the pure core_stats_mask created at initialization.
            calc_mask = core_stats_mask.clone()
            calc_mask_dill = core_stats_mask.clone()
            mask_forcalc_512 = core_stats_mask.clone()

        # Initialize AutoColor mask based on the pure calculation mask
        mask_autocolor = calc_mask.clone()
        mask_autocolor = (mask_autocolor > 0.5).float()

        # Auto Restore (First Pass)
        if (
            parameters["FaceRestorerEnableToggle"]
            and parameters["FaceRestorerAutoEnableToggle"]
        ):
            assert swap_original is not None, (
                "swap_original must be set when FaceRestorerEnableToggle is active"
            )
            alpha_restorer = float(parameters["FaceRestorerBlendSlider"]) / 100.0
            adjust_sharpness = float(parameters["FaceRestorerAutoSharpAdjustSlider"])
            scale_factor = round(tform.scale, 2)
            automasktoggle = parameters["FaceRestorerAutoMaskEnableToggle"]
            automaskadjust = parameters["FaceRestorerAutoSharpMaskAdjustDecimalSlider"]
            automaskblur = 2
            restore_mask = mask_forcalc_512

            alpha_auto, blur_value = self.face_restorer_auto(
                original_face_512,
                swap_original,
                swap_restorecalc,
                alpha_restorer,
                adjust_sharpness,
                scale_factor,
                debug,
                restore_mask,
                automasktoggle,
                automaskadjust,
                automaskblur,
            )

            if blur_value > 0:
                kernel_size = 2 * blur_value + 1
                sigma = blur_value * 0.1
                gaussian_blur = v2.GaussianBlur(kernel_size=kernel_size, sigma=sigma)
                swap = gaussian_blur(swap_original)
                debug_info["Restore1"] = f": {-blur_value:.2f}"
            elif isinstance(alpha_auto, torch.Tensor):
                swap = swap_restorecalc * alpha_auto + swap_original * (1 - alpha_auto)
            elif alpha_auto != 0:
                swap = swap_restorecalc * alpha_auto + swap_original * (1 - alpha_auto)
                if debug:
                    debug_info["Restore1"] = f": {alpha_auto * 100:.2f}"
            else:
                swap = swap_original
                if debug:
                    debug_info["Restore1"] = f": {alpha_auto * 100:.2f}"
        elif parameters["FaceRestorerEnableToggle"]:
            assert swap_original is not None, (
                "swap_original must be set when FaceRestorerEnableToggle is active"
            )
            alpha_restorer = float(parameters["FaceRestorerBlendSlider"]) / 100.0
            swap = (
                torch.lerp(
                    swap_original.float(), swap_restorecalc.float(), alpha_restorer
                )
                .to(swap_restorecalc.dtype)
                .contiguous()
            )

        # Expression Restorer (After First)
        if (
            parameters["FaceExpressionEnableBothToggle"]
            and (
                parameters["FaceExpressionLipsToggle"]
                or parameters["FaceExpressionEyesToggle"]
                or parameters["FaceExpressionBrowsToggle"]
                or parameters["FaceExpressionGeneralToggle"]
                or parameters.get("FaceExpressionModeSelection", "Advanced")
                in ("Simple", "Recast")
            )
            and parameters["FaceExpressionBeforeTypeSelection"]
            == "After First Restorer"
        ):
            if parameters.get("FaceExpressionModeSelection", "Advanced") == "Recast":
                swap = self.worker.function_worker.apply_perform_recast(
                    original_face_512,
                    swap,
                    cast(dict, parameters),
                    cast(dict, control),
                    driving_kps=kps_all_crop,
                )
            else:
                swap = self.worker.function_worker.apply_face_expression_restorer(
                    original_face_512,
                    swap,
                    cast(dict, parameters),
                    cast(dict, control),
                    driving_kps=kps_all_crop,
                )

        # Face Editor (After First)
        if (
            parameters["FaceEditorEnableToggle"]
            and self.worker.local_control_state_from_feeder.get("edit_enabled", True)
            and parameters["FaceEditorBeforeTypeSelection"] == "After First Restorer"
        ):
            editor_mask = swap_mask.clone()
            swap = (
                torch.lerp(original_face_512.float(), swap.float(), editor_mask)
                .to(swap.dtype)
                .contiguous()
            )
            swap = self.worker.function_worker.swap_edit_face_core(
                swap, swap_restorecalc, parameters, control
            )
            if swap_mask_noFP.shape[-1] != swap.shape[-1]:
                swap_mask = _resize_func(
                    swap_mask_noFP, (swap.shape[-2], swap.shape[-1]), is_mask=True
                )
            else:
                swap_mask = swap_mask_noFP

        # Second Denoiser pass - After First Restorer
        if control.get("DenoiserAfterFirstRestorerToggle", False):
            swap = self._apply_denoiser_pass(
                swap,
                control,
                "AfterFirst",
                kv_map,
                color_mask=mask_forcalc_512,
                blend_mask=swap_mask,
            )

        # --- RESTORATION 2 ---
        # FW-QUAL-01/02: duplicated ~60-line block extracted to _apply_restorer_with_auto
        if (
            parameters["FaceRestorerEnable2Toggle"]
            and not parameters["FaceRestorerEnable2EndToggle"]
        ):
            swap_original2 = swap.clone()
            swap2 = self.worker.function_worker.apply_facerestorer(
                swap,
                parameters["FaceRestorerDetType2Selection"],
                parameters["FaceRestorerType2Selection"],
                parameters["FaceRestorerBlend2Slider"],
                parameters["FaceFidelityWeight2DecimalSlider"],
                control["DetectorScoreSlider"],
                kps_ref,
                slot_id=2,
            )
            swap = self._apply_restorer_with_auto(
                swap,
                swap2,
                swap_original2,
                original_face_512,
                mask_forcalc_512,
                parameters,
                tform.scale,
                debug,
                debug_info,
                slot_id=2,
            )

        # Expression (After Second)
        if (
            parameters["FaceExpressionEnableBothToggle"]
            and (
                parameters["FaceExpressionLipsToggle"]
                or parameters["FaceExpressionEyesToggle"]
                or parameters["FaceExpressionBrowsToggle"]
                or parameters["FaceExpressionGeneralToggle"]
                or parameters.get("FaceExpressionModeSelection", "Advanced")
                in ("Simple", "Recast")
            )
            and parameters["FaceExpressionBeforeTypeSelection"]
            == "After Second Restorer"
        ):
            if parameters.get("FaceExpressionModeSelection", "Advanced") == "Recast":
                swap = self.worker.function_worker.apply_perform_recast(
                    original_face_512,
                    swap,
                    cast(dict, parameters),
                    cast(dict, control),
                    driving_kps=kps_all_crop,
                )
            else:
                swap = self.worker.function_worker.apply_face_expression_restorer(
                    original_face_512,
                    swap,
                    cast(dict, parameters),
                    cast(dict, control),
                    driving_kps=kps_all_crop,
                )

        # Editor (After Second)
        if (
            parameters["FaceEditorEnableToggle"]
            and self.worker.local_control_state_from_feeder.get("edit_enabled", True)
            and parameters["FaceEditorBeforeTypeSelection"] == "After Second Restorer"
        ):
            editor_mask = t512_mask(swap_mask).clone()
            swap = (
                torch.lerp(original_face_512.float(), swap.float(), editor_mask)
                .to(swap.dtype)
                .contiguous()
            )
            swap = self.worker.function_worker.swap_edit_face_core(
                swap, swap, parameters, control
            )
            if swap_mask_noFP.shape[-1] != swap.shape[-1]:
                swap_mask = _resize_func(
                    swap_mask_noFP, (swap.shape[-2], swap.shape[-1]), is_mask=True
                )
            else:
                swap_mask = swap_mask_noFP

        # --- AUTO COLOR (Mask 512) ---
        # FW-QUAL-12: AutoColorEnableToggle runs here — BEFORE FaceParser mask is applied
        # to the global swap_mask (the EndingColorTransfer runs at the end, AFTER the
        # FaceParser end-pass and the final swap_mask re-calculation, so it operates on a
        # tighter mask that excludes eyes/mouth/hairline etc.).

        # Q-QUAL-03: build a smoothed reference face for AutoColor to reduce per-frame flicker.
        # Key by target embedding bytes so each target face has its own EMA history.
        original_face_for_color = original_face_512
        if parameters.get("AutoColorEnableToggle", False) and valid_t_e is not None:
            _ema_key = valid_t_e.tobytes()
            _face_f = original_face_512.float()
            _curr_mean = _face_f.mean(dim=(1, 2), keepdim=True)
            _curr_std = _face_f.std(dim=(1, 2), keepdim=True) + 1e-6
            if _ema_key in self._color_stats_ema:
                self._color_stats_ema.move_to_end(_ema_key)
                _prev = self._color_stats_ema[_ema_key]
                _ema_mean = (
                    self._COLOR_EMA_ALPHA * _curr_mean
                    + (1.0 - self._COLOR_EMA_ALPHA) * _prev["mean"]
                )
                _ema_std = (
                    self._COLOR_EMA_ALPHA * _curr_std
                    + (1.0 - self._COLOR_EMA_ALPHA) * _prev["std"]
                )
            else:
                _ema_mean, _ema_std = _curr_mean, _curr_std
                # LRU eviction before inserting a new entry
                if len(self._color_stats_ema) >= self._COLOR_STATS_EMA_MAX:
                    self._color_stats_ema.popitem(last=False)
            self._color_stats_ema[_ema_key] = {
                "mean": _ema_mean.detach(),
                "std": _ema_std.detach(),
            }
            # Remap original_face to have smoothed colour statistics
            original_face_for_color = (
                ((_face_f - _curr_mean) / _curr_std * _ema_std + _ema_mean)
                .clamp(0, 255)
                .to(original_face_512.dtype)
            )

        if parameters.get("AutoColorEnableToggle", False):
            if parameters["AutoColorTransferTypeSelection"] == "CDF Histogram":
                swap = faceutil.histogram_matching(
                    original_face_for_color,
                    swap,
                    parameters["AutoColorBlendAmountSlider"],
                )
            elif (
                parameters["AutoColorTransferTypeSelection"] == "CDF Histogram (Masked)"
            ):
                swap = faceutil.histogram_matching_withmask(
                    original_face_for_color,
                    swap,
                    mask_autocolor,
                    parameters["AutoColorBlendAmountSlider"],
                )
            elif parameters["AutoColorTransferTypeSelection"] == "Reinhard Transfer":
                swap = faceutil.apply_reinhard_color_transfer(
                    original_face_for_color,
                    swap,
                    parameters["AutoColorBlendAmountSlider"],
                )
            elif (
                parameters["AutoColorTransferTypeSelection"]
                == "Reinhard Transfer (Masked)"
            ):
                swap = faceutil.apply_reinhard_color_transfer(
                    original_face_for_color,
                    swap,
                    parameters["AutoColorBlendAmountSlider"],
                    mask_autocolor,
                )
            elif parameters["AutoColorTransferTypeSelection"] == "AdaIN (Core Masked)":
                swap = faceutil.apply_adain_color_transfer(
                    swap,
                    original_face_for_color,
                    swap_mask,
                    parameters["AutoColorBlendAmountSlider"],
                    calc_mask=mask_autocolor,
                )

        # --- TRANSFER TEXTURE ---
        if parameters.get("TransferTextureEnableToggle", False):
            # 1. Ensure resolutions match target 512x512
            if swap.shape[-1] != 512:
                swap = t512_mask(swap)
                swap_mask = t512_mask(swap_mask)
                swap_mask_noFP = t512_mask(swap_mask_noFP)

            mask_input_vgg = t128_mask(calc_mask.clone())
            mask_vgg_512 = torch.ones(
                (1, 512, 512),
                dtype=torch.float32,
                device=self.worker.models_processor.device,
            )
            upper_thresh = parameters["TextureUpperLimitSlider"] / 100.0

            # 2. VGG Mask Processing
            if parameters.get("ExcludeOriginalVGGMaskEnableToggle", False):
                # Fetch threshold values from UI
                thr = (
                    parameters["VGGMaskThresholdSlider"]
                    if parameters.get("ExcludeVGGMaskEnableToggle", False)
                    else 0
                )
                # Retrieve BOTH the thresholded mask and the raw normalized difference (Size: 128x128)
                mask_vgg_raw, diff_norm_texture_raw = (
                    self.worker.function_worker.apply_vgg_mask_simple(
                        swap,
                        original_face_512,
                        mask_input_vgg,
                        center_pct=thr,
                        softness_pct=100,
                        feature_layer="combo_relu3_3_relu3_1",
                        mode="smooth",
                    )
                )
                # Upscale to 512x512 IMMEDIATELY to prevent tensor mismatch
                mask_vgg_512 = t512_mask(mask_vgg_raw).clamp(0.0, 1.0)
                diff_norm_texture_512 = t512_mask(diff_norm_texture_raw).clamp(0.0, 1.0)

                # Fallback to the raw difference texture if manipulation is disabled (Restoring old behavior)
                if not parameters.get("ExcludeVGGMaskEnableToggle", False):
                    mask_vgg_512 = diff_norm_texture_512.clone()

                # Optional VGG specific blur
                if parameters.get("TextureBlendAmountSlider", 0) > 0:
                    b = parameters["TextureBlendAmountSlider"]
                    gauss = v2.GaussianBlur(b * 2 + 1, (b + 1) * 0.2)
                    mask_vgg_512 = gauss(mask_vgg_512.float())

            # 3. Features Exclusion Logic (Eyes, Mouth, etc.)
            if parameters.get("ExcludeMaskEnableToggle", False):
                # texture_exclude_512: 1 means KEEP texture (skin), 0 means REMOVE texture (eyes/mouth)
                feature_mask = texture_exclude_512.clone().float()

                # This creates a smooth gradient transition instead of a harsh binary cut-off.
                if parameters.get("ExcludeOriginalVGGMaskEnableToggle", False):
                    blur_val = parameters.get("FaceParserBlurTextureSlider", 0)
                    if blur_val > 0:
                        kernel_size = int(blur_val * 2 + 1)
                        sigma = max((blur_val + 1) * 0.2, 1e-6)
                        blur_op = v2.GaussianBlur(kernel_size, sigma=sigma)
                        feature_mask = blur_op(feature_mask)

                # Combine VGG mask with the spatial FaceParser mask
                if parameters.get("ExcludeOriginalVGGMaskEnableToggle", False):
                    # Clamp upper limits to protect extreme highlights/differences
                    mask_vgg_512 = torch.where(
                        mask_vgg_512 >= upper_thresh, upper_thresh, mask_vgg_512
                    )

                mask_final_512 = (
                    torch.max(mask_vgg_512 * (1.0 - feature_mask), 1.0 - calc_mask_dill)
                ).clamp(0.0, 1.0)
            elif parameters.get("ExcludeOriginalVGGMaskEnableToggle", False):
                # Clamp upper limits to protect extreme highlights/differences
                mask_vgg_512 = torch.where(
                    mask_vgg_512 >= upper_thresh, upper_thresh, mask_vgg_512
                )
                # Protect background if no spatial exclusion is active
                mask_final_512 = torch.max(mask_vgg_512, 1.0 - calc_mask_dill).clamp(
                    0.0, 1.0
                )
            else:
                # Fallback to raw mask if everything is disabled
                mask_final_512 = (1.0 - mask_forcalc_512).clamp(0.0, 1.0)

            # FW-QUAL-03: dead-code block converted from triple-quoted string to comments.
            # 4. Final Mask Smoothing (Applied only ONCE at the end) — disabled / superseded
            # if parameters.get("FaceParserBlurTextureSlider", 0) > 0:
            #     orig_m = mask_final_512.clone()
            #     b_fp = parameters["FaceParserBlurTextureSlider"]
            #     kernel_size = int(b_fp * 2 + 1)
            #     gauss = v2.GaussianBlur(kernel_size, (b_fp + 1) * 0.2)
            #     mask_final_512 = gauss(mask_final_512.type(torch.float32))
            #     # Restore sharp inner boundaries while softening the gradients
            #     mask_final_512 = torch.max(mask_final_512, orig_m).clamp(0.0, 1.0)
            # 5. AutoColor Backup Logic
            if parameters.get("AutoColorEnableToggle", False):
                swap_texture_backup = swap.clone()
            else:
                swap_texture_backup = faceutil.apply_reinhard_color_transfer(
                    original_face_512, swap.clone(), 100, mask_autocolor
                )

            # 5. Gradient / Texture Generation Settings
            TransferTextureKernelSizeSlider = 12
            TransferTextureSigmaDecimalSlider = 4.00
            TransferTextureWeightSlider = 1
            TransferTexturePhiDecimalSlider = 9.7
            TransferTextureGammaDecimalSlider = 0.5

            if parameters.get("TransferTextureModeEnableToggle", False):
                TransferTextureLambdSlider = 8
                TransferTextureThetaSlider = 8
            else:
                TransferTextureLambdSlider = 2
                TransferTextureThetaSlider = 1

            clip_limit = (
                parameters["TransferTextureClipLimitDecimalSlider"]
                if parameters.get("TransferTextureClaheEnableToggle", False)
                else 0.0
            )
            alpha_clahe = parameters["TransferTextureAlphaClaheDecimalSlider"]
            grid_size = (4, 4)
            global_gamma = parameters["TransferTexturePreGammaDecimalSlider"]
            global_contrast = parameters["TransferTexturePreContrastDecimalSlider"]

            gradient_texture = self.gradient_magnitude(
                original_face_512,
                calc_mask_dill,
                TransferTextureKernelSizeSlider,
                TransferTextureWeightSlider,
                TransferTextureSigmaDecimalSlider,
                TransferTextureLambdSlider,
                TransferTextureGammaDecimalSlider,
                TransferTexturePhiDecimalSlider,
                TransferTextureThetaSlider,
                clip_limit,
                alpha_clahe,
                grid_size,
                global_gamma,
                global_contrast,
            )

            gradient_texture = faceutil.apply_reinhard_color_transfer(
                original_face_512, gradient_texture, 100, mask_autocolor
            )

            if parameters["FaceParserBlurTextureSlider"] > 0:
                orig = mask_final_512.clone()
                gauss = v2.GaussianBlur(
                    parameters["FaceParserBlurTextureSlider"] * 2 + 1,
                    (parameters["FaceParserBlurTextureSlider"] + 1) * 0.2,
                )
                mask_final_512 = gauss(mask_final_512.type(torch.float32))
                mask_final_512 = torch.max(mask_final_512, orig).clamp(0.0, 1.0)
            # 6. Final Blending
            # alpha_t modulates the overall strength, w determines the per-pixel application map
            alpha_t = parameters["TransferTextureBlendAmountSlider"] / 100.0
            w = alpha_t * (1.0 - mask_final_512)
            w = w.clamp(0.0, 1.0)

            swap = (swap_texture_backup * (1.0 - w) + gradient_texture * w).clamp(
                0, 255
            )
            texture_mask_view = (1.0 - mask_final_512).clone()

        # --- DIFFERENCING ---
        if parameters.get("DifferencingEnableToggle", False):
            if swap.shape[-1] != 512:
                swap = t512_mask(swap)
                swap_mask = t512_mask(swap_mask)
                swap_mask_noFP = t512_mask(swap_mask_noFP)

            diff_mask_128 = t128_mask(calc_mask.clone())
            swapped_face_resized = swap.clone()
            original_face_resized = original_face_512.clone()
            FeatureLayerTypeSelection = "combo_relu3_3_relu3_1"

            lower_thresh = parameters["DifferencingLowerLimitThreshSlider"] / 100.0
            upper_thresh = parameters["DifferencingUpperLimitThreshSlider"] / 100.0
            middle_value = parameters["DifferencingMiddleLimitValueSlider"] / 100.0
            upper_value = parameters["DifferencingUpperLimitValueSlider"] / 100.0

            mask_diff_128, diff_norm_texture = (
                self.worker.function_worker.apply_perceptual_diff_onnx(
                    swapped_face_resized,
                    original_face_resized,
                    diff_mask_128,
                    lower_thresh,
                    0,
                    upper_thresh,
                    upper_value,
                    middle_value,
                    FeatureLayerTypeSelection,
                    False,
                )
            )

            eps = 1e-6
            inv_lower = 1.0 / max(lower_thresh, eps)
            inv_mid = 1.0 / max((upper_thresh - lower_thresh), eps)
            inv_high = 1.0 / max((1.0 - upper_thresh), eps)

            res_low = diff_norm_texture * inv_lower * middle_value
            res_mid = middle_value + (diff_norm_texture - lower_thresh) * inv_mid * (
                upper_value - middle_value
            )
            res_high = upper_value + (diff_norm_texture - upper_thresh) * inv_high * (
                1.0 - upper_value
            )

            piece = torch.where(
                diff_norm_texture < lower_thresh,
                res_low,
                torch.where(diff_norm_texture > upper_thresh, res_high, res_mid),
            )
            mask512 = t512_mask(piece)
            if parameters.get("DifferencingBlendAmountSlider", 0) > 0:
                b = parameters["DifferencingBlendAmountSlider"]
                gauss = v2.GaussianBlur(b * 2 + 1, (b + 1) * 0.2)
                mask512 = gauss(mask512.float())

            mask512 = torch.max((mask512), 1 - calc_mask_dill).clamp_(0, 1)
            swap = (
                torch.lerp(original_face_512.float(), swap.float(), mask512)
                .clamp_(0, 255)
                .to(swap.dtype)
                .contiguous()
            )
            diff_mask = 1 - mask512.clone()

        # Face Editor (After Texture Transfer)
        if (
            parameters["FaceEditorEnableToggle"]
            and self.worker.local_control_state_from_feeder.get("edit_enabled", True)
            and parameters["FaceEditorBeforeTypeSelection"] == "After Texture Transfer"
        ):
            editor_mask = t512_mask(swap_mask).clone()
            if swap.shape[-1] != 512:
                swap = t512_mask(swap)

            swap = (
                torch.lerp(original_face_512.float(), swap.float(), editor_mask)
                .to(swap.dtype)
                .contiguous()
            )
            swap = self.worker.function_worker.swap_edit_face_core(
                swap, swap, parameters, control
            )
            if swap_mask_noFP.shape[-1] != swap.shape[-1]:
                swap_mask = _resize_func(
                    swap_mask_noFP, (swap.shape[-2], swap.shape[-1]), is_mask=True
                )
            else:
                swap_mask = swap_mask_noFP

        # --- COLOR CORRECTIONS ---
        if parameters["ColorEnableToggle"]:
            # 1. Save a backup of the original swap to preserve the background/padding
            swap_pre_color = swap.clone()

            # 2. Apply color transformations
            swap_adj = torch.unsqueeze(swap, 0).contiguous()
            swap_adj = v2.functional.adjust_gamma(
                swap_adj, parameters["ColorGammaDecimalSlider"], 1.0
            )
            swap_adj = torch.squeeze(swap_adj).permute(1, 2, 0).type(torch.float32)

            del_color = torch.tensor(
                [
                    parameters["ColorRedSlider"],
                    parameters["ColorGreenSlider"],
                    parameters["ColorBlueSlider"],
                ],
                device=self.worker.models_processor.device,
            )
            swap_adj += del_color
            swap_adj = (
                torch.clamp(swap_adj, min=0.0, max=255.0).permute(2, 0, 1) / 255.0
            )

            swap_adj = v2.functional.adjust_brightness(
                swap_adj, parameters["ColorBrightnessDecimalSlider"]
            )
            swap_adj = v2.functional.adjust_contrast(
                swap_adj, parameters["ColorContrastDecimalSlider"]
            )
            swap_adj = v2.functional.adjust_saturation(
                swap_adj, parameters["ColorSaturationDecimalSlider"]
            )
            swap_adj = v2.functional.adjust_sharpness(
                swap_adj, parameters["ColorSharpnessDecimalSlider"]
            )
            swap_adj = v2.functional.adjust_hue(
                swap_adj, parameters["ColorHueDecimalSlider"]
            )

            swap_adj = swap_adj * 255.0

            # 3. Blend back using a Soft Padding Mask.
            # Apply a strong blur to the boundary mask for a seamless transition
            color_bounds_mask = v2.functional.gaussian_blur(
                (swap_pre_color.sum(dim=0, keepdim=True) > 0.05).float(),
                kernel_size=15,
                sigma=5.0,
            )
            swap = (
                torch.lerp(swap_pre_color.float(), swap_adj.float(), color_bounds_mask)
                .to(swap.dtype)
                .contiguous()
            )

        # --- RESTORATION 2 (END) ---
        # FW-QUAL-01/02: duplicated ~60-line block extracted to _apply_restorer_with_auto
        if (
            parameters["FaceRestorerEnable2Toggle"]
            and parameters["FaceRestorerEnable2EndToggle"]
        ):
            swap_original2 = swap.clone()
            swap2 = self.worker.function_worker.apply_facerestorer(
                swap,
                parameters["FaceRestorerDetType2Selection"],
                parameters["FaceRestorerType2Selection"],
                parameters["FaceRestorerBlend2Slider"],
                parameters["FaceFidelityWeight2DecimalSlider"],
                control["DetectorScoreSlider"],
                kps_ref,
                slot_id=2,
            )
            swap = self._apply_restorer_with_auto(
                swap,
                swap2,
                swap_original2,
                original_face_512,
                mask_forcalc_512,
                parameters,
                tform.scale,
                debug,
                debug_info,
                slot_id=2,
            )

        # --- MOUTH ENHANCEMENT & ALIGNMENT (POST-RESTORER) ---
        if parameters.get("MouthParserStretchAfterToggle", False):
            mouth_overlay_pkg = None
            if hasattr(self.worker.function_worker, "get_mouth_overlay"):
                # 'swap' now contains the fully restored face
                mouth_overlay_pkg = self.worker.function_worker.get_mouth_overlay(
                    swap, original_face_512, parameters
                )
            if mouth_overlay_pkg is not None:
                overlay_rgb, overlay_mask = mouth_overlay_pkg
                if overlay_rgb is not None and overlay_mask is not None:
                    if overlay_rgb.shape[-1] != swap.shape[-1]:
                        overlay_rgb = _resize_func(
                            overlay_rgb, (swap.shape[-2], swap.shape[-1]), is_mask=False
                        )
                        overlay_mask = _resize_func(
                            overlay_mask.unsqueeze(0),
                            (swap.shape[-2], swap.shape[-1]),
                            is_mask=True,
                        ).squeeze(0)
                    swap = swap * (1.0 - overlay_mask) + overlay_rgb * overlay_mask

        # --- FACE PARSER (END) ---
        if parameters.get("FaceParserEnableToggle") and parameters.get(
            "FaceParserEndToggle"
        ):
            out = self.worker.function_worker.process_masks_and_masks(
                swap, original_face_512, parameters, control
            )
            FaceParser_mask = out.get("FaceParser_mask", None)
            mouth_debug_512 = out.get("mouth_debug", mouth_debug_512)
            mouth_debug_teeth_512 = out.get("mouth_debug_teeth", mouth_debug_teeth_512)
            if FaceParser_mask is not None:
                if FaceParser_mask.shape[-1] != swap_mask.shape[-1]:
                    FaceParser_mask = _resize_func(
                        FaceParser_mask, (swap.shape[-2], swap.shape[-1]), is_mask=True
                    )
                swap_mask.mul_(FaceParser_mask)

        # RECALCULATE FINAL CORE STATS MASK (For Ending Color Transfer)
        mask_autocolor_end = mask_forcalc_512.clone()
        # If FaceParser End pass was active, intersect it strictly to exclude eyes/mouth from stats
        if (
            parameters.get("FaceParserEnableToggle")
            and parameters.get("FaceParserEndToggle")
            and FaceParser_mask is not None
        ):
            # Create a hard binary mask from the soft parser mask to protect stats
            fp_core = (FaceParser_mask > 0.5).float()
            if fp_core.shape[-1] != mask_autocolor_end.shape[-1]:
                fp_core = _resize_func(
                    fp_core,
                    (mask_autocolor_end.shape[-2], mask_autocolor_end.shape[-1]),
                    is_mask=True,
                )
            mask_autocolor_end = mask_autocolor_end * fp_core
        # Ensure it is strictly binary
        mask_autocolor_end = (mask_autocolor_end > 0.5).float()

        # AutoColor End (EndingColorTransfer)
        if parameters.get("EndingColorTransferEnableToggle", False):
            if parameters["EndingColorTransferTypeSelection"] == "CDF Histogram":
                swap = faceutil.histogram_matching(
                    original_face_512, swap, parameters["EndingColorBlendAmountSlider"]
                )
            elif (
                parameters["EndingColorTransferTypeSelection"]
                == "CDF Histogram (Masked)"
            ):
                swap = faceutil.histogram_matching_withmask(
                    original_face_512,
                    swap,
                    mask_autocolor_end,
                    parameters["EndingColorBlendAmountSlider"],
                )
            elif parameters["EndingColorTransferTypeSelection"] == "Reinhard Transfer":
                swap = faceutil.apply_reinhard_color_transfer(
                    original_face_512, swap, parameters["EndingColorBlendAmountSlider"]
                )
            elif (
                parameters["EndingColorTransferTypeSelection"]
                == "Reinhard Transfer (Masked)"
            ):
                swap = faceutil.apply_reinhard_color_transfer(
                    original_face_512,
                    swap,
                    parameters["EndingColorBlendAmountSlider"],
                    mask_autocolor_end,
                )
            elif (
                parameters["EndingColorTransferTypeSelection"] == "AdaIN (Core Masked)"
            ):
                swap = faceutil.apply_adain_color_transfer(
                    swap,
                    original_face_512,
                    swap_mask,
                    parameters["EndingColorBlendAmountSlider"],
                    calc_mask=mask_autocolor_end,
                )

        # Third denoiser pass - After all restorations, colour corrections and ending colour transfer.
        if control.get("DenoiserAfterRestorersToggle", False):
            # We use mask_autocolor_end here because it is the most strict mask available at the end of the pipeline
            swap = self._apply_denoiser_pass(
                swap,
                control,
                "After",
                kv_map,
                color_mask=mask_autocolor_end,
                blend_mask=swap_mask,
            )

        # Final blending (Global - Whole Face)
        if (
            parameters.get("FinalBlendAdjEnableToggle", False)
            and parameters.get("FinalBlendAmountSlider", 0) > 0
        ):
            final_blur_strength = parameters["FinalBlendAmountSlider"]
            kernel_size = 2 * final_blur_strength + 1
            sigma = max(final_blur_strength * 0.1, 1e-6)
            gaussian_blur = v2.GaussianBlur(kernel_size=kernel_size, sigma=sigma)
            swap = gaussian_blur(swap)

        # Final blending (Border Seam Only via Morphological Gradient)
        if (
            parameters.get("FinalBorderBlendEnableToggle", False)
            and parameters.get("FinalBorderBlendAmountSlider", 0) > 0
        ):
            border_blur_strength = int(parameters["FinalBorderBlendAmountSlider"])

            # Ensure kernel size is odd for spatial pooling and blurring
            kernel_size = int(2 * border_blur_strength + 1)
            padding = border_blur_strength
            sigma = max(border_blur_strength * 0.2, 1e-6)

            # 1. Create the blurred version of the swap tensor
            gaussian_blur = v2.GaussianBlur(kernel_size=kernel_size, sigma=sigma)
            blurred_swap = gaussian_blur(swap)

            # 2. Calculate Morphological Gradient to perfectly isolate the mask's transition seam.
            # Dilation expands the mask outward; Erosion shrinks it inward.
            # We use F.max_pool2d for highly optimized CUDA morphological operations.
            dilation = F.max_pool2d(
                swap_mask, kernel_size=kernel_size, stride=1, padding=padding
            )
            erosion = -F.max_pool2d(
                -swap_mask, kernel_size=kernel_size, stride=1, padding=padding
            )

            # The difference is exactly the boundary edge, spanning inward and outward evenly.
            raw_border_band = dilation - erosion

            # 3. Soften the extracted border band so the blur effect fades smoothly
            smooth_border_mask = gaussian_blur(raw_border_band).clamp_(0.0, 1.0)

            # 4. Blend the blurred image strictly along the isolated morphological seam
            swap = (
                torch.lerp(swap.float(), blurred_swap.float(), smooth_border_mask)
                .to(swap.dtype)
                .contiguous()
            )

        # Artefacts: Jpeg
        if parameters["JPEGCompressionEnableToggle"]:
            jpeg_q = int(parameters["JPEGCompressionAmountSlider"])
            if jpeg_q != 100:
                s = float(tform.scale)
                gamma = 0.60
                strength = 0.80
                q_min = 14
                q_max = 100

                jpeg_q_eff = faceutil._map_jpeg_quality(
                    base_q=jpeg_q,
                    face_scale=s,
                    gamma=gamma,
                    strength=strength,
                    q_min=q_min,
                    q_max=q_max,
                )
                if debug:
                    debug_info["JPEG Quality"] = f"{jpeg_q_eff}"

                # FW-BUG-14: renamed swap2 -> swap_jpeg in JPEG block
                swap_jpeg = faceutil.jpegBlur(swap, jpeg_q_eff)
                blend = parameters["JPEGCompressionBlendSlider"] / 100.0
                swap = (
                    torch.lerp(swap.float(), swap_jpeg.float(), blend)
                    .to(swap.dtype)
                    .contiguous()
                )

        # Artefacts: BlockShift
        if parameters["BlockShiftEnableToggle"]:
            base_quality = parameters["BlockShiftAmountSlider"]
            max_px = parameters["BlockShiftMaxAmountSlider"]

            # FW-BUG-14: renamed swap2 -> swap_blockshift in BlockShift block
            swap_blockshift = self.apply_block_shift_gpu_jitter(
                swap,
                block_size=base_quality,
                max_amount_pixels=float(max_px),
                seed=1337,
            )

            block_shift_blend = parameters["BlockShiftBlendAmountSlider"] / 100.0
            swap = (
                torch.lerp(swap.float(), swap_blockshift.float(), block_shift_blend)
                .to(swap.dtype)
                .contiguous()
            )

        if parameters["ColorNoiseDecimalSlider"] > 0:
            swap = swap.to(torch.float32)
            noise = (
                (torch.rand_like(swap, dtype=torch.float32) - 0.5)
                * 2
                * parameters["ColorNoiseDecimalSlider"]
            )
            swap = torch.clamp(swap + noise, 0.0, 255.0)

        if is_perspective_crop:
            return t512_mask(swap), t512_mask(swap_mask), None

        # Mask Post-Processing (Final Blend)
        gauss = v2.GaussianBlur(
            parameters["OverallMaskBlendAmountSlider"] * 2 + 1,
            (parameters["OverallMaskBlendAmountSlider"] + 1) * 0.2,
        )
        swap_mask = gauss(swap_mask)

        if border_mask.shape[-1] != swap_mask.shape[-1]:
            border_mask = _resize_func(
                border_mask, (swap_mask.shape[-2], swap_mask.shape[-1]), is_mask=True
            )

        swap_mask = torch.mul(swap_mask, border_mask)

        if swap.shape[-1] != 512:
            swap = t512_mask(swap)
            swap_mask = t512_mask(swap_mask)

        swap = torch.mul(swap, swap_mask)

        # --- VIEW MODES ---
        original_face_512_clone = (
            original_face_512.clone().type(torch.uint8).permute(1, 2, 0)
            if self.worker.is_view_face_compare
            else None
        )

        swap_mask_clone = None
        if self.worker.is_view_face_mask:
            mask_show_type = parameters.get("MaskShowSelection", "swap_mask")
            if mask_show_type == "swap_mask":
                swap_mask_clone = (
                    torch.ones_like(swap_mask)
                    if (
                        parameters["FaceEditorEnableToggle"]
                        and self.worker.local_control_state_from_feeder.get(
                            "edit_enabled", True
                        )
                    )
                    else swap_mask.clone()
                )
            elif mask_show_type == "diff":
                swap_mask_clone = diff_mask.clone()
            elif mask_show_type == "texture":
                swap_mask_clone = texture_mask_view.clone()

            if swap_mask_clone is not None:
                if swap_mask_clone.shape[-1] != 512:
                    swap_mask_clone = t512_mask(swap_mask_clone)
                swap_mask_clone = torch.mul(
                    torch.cat((torch.sub(1, swap_mask_clone),) * 3, 0).permute(1, 2, 0),
                    255.0,
                ).type(torch.uint8)

        # --- OPTIMIZED UNTRANSFORM (PASTE BACK) USING KORNIA ---
        # Eliminates CPU bound calculations, manual slicing, and memory-heavy paddings.
        # Warps directly to the full frame resolution in one highly optimized GPU pass.

        M_inv = (
            torch.from_numpy(cast(np.ndarray, tform.inverse.params)[0:2])
            .float()
            .unsqueeze(0)
            .to(self.worker.models_processor.device)
        )
        dsize = (img.shape[1], img.shape[2])

        # Warp the 512x512 face and mask directly into the full frame space
        swap_full = kgm.warp_affine(
            swap.unsqueeze(0).float(),
            M_inv,
            dsize=dsize,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        ).squeeze(0)
        swap_mask_full = kgm.warp_affine(
            swap_mask.unsqueeze(0).float(),
            M_inv,
            dsize=dsize,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        ).squeeze(0)

        img = (
            (swap_full + (img.float() * (1.0 - swap_mask_full)))
            .clamp_(0, 255)
            .type(torch.uint8)
        )

        # --- DEBUG: Draw mouth-region contours on full-frame preview ---
        # Colors: mouth(11+12+13)=green, teeth-keep=cyan.
        if parameters.get("AutoMouthShowDebugOutlineToggle", False):
            for _mask_512, _color in [
                (mouth_debug_512, [0, 255, 0]),
                (mouth_debug_teeth_512, [0, 255, 255]),
            ]:
                if _mask_512 is not None:
                    _mf = (
                        kgm.warp_affine(
                            _mask_512.unsqueeze(0).unsqueeze(0).float(),
                            M_inv,
                            dsize=dsize,
                            mode="bilinear",
                            padding_mode="zeros",
                            align_corners=True,
                        )
                        .squeeze(0)
                        .squeeze(0)
                    )
                    _bin = (_mf > 0.5).float().unsqueeze(0).unsqueeze(0)
                    _edge = (
                        F.max_pool2d(_bin, kernel_size=3, stride=1, padding=1)
                        - (-F.max_pool2d(-_bin, kernel_size=3, stride=1, padding=1))
                    ).squeeze(0).squeeze(0) > 0.0
                    img[0][_edge], img[1][_edge], img[2][_edge] = (
                        _color[0],
                        _color[1],
                        _color[2],
                    )

        return img, original_face_512_clone, swap_mask_clone

    @torch.no_grad()
    def gradient_magnitude(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        kernel_size: int,
        weighting_strength: float,
        sigma: float,
        lambd: float,
        gamma: float,
        psi: float,
        theta_count: int,
        clip_limit: float,
        alpha_clahe: float,
        grid_size: tuple[int, int],
        global_gamma: float,
        global_contrast: float,
    ) -> torch.Tensor:
        """
        Calculates the weighted Gabor magnitude for texture transfer.

        Args:
            image: Tensor [C, H, W] in [0..255]
            mask:  Tensor [C, H, W] (0/1)
        Returns:
            Tensor [C, H, W] – weighted Gabor magnitude
        """

        C, H, W = image.shape
        image = image.float() / 255.0
        mask = mask.bool()

        # 1) Global Gamma & Contrast
        if global_gamma != 1.0:
            image = image.pow(global_gamma)
        if global_contrast != 1.0:
            m_gc = image.mean((1, 2), keepdim=True)
            image = (image - m_gc) * global_contrast + m_gc

        # 2) CLAHE in L-channel (with alpha_clahe blending)
        if clip_limit > 0.0:
            image = image.unsqueeze(0).clamp(0, 1)  # [1,3,H,W]
            mask_b3 = mask.unsqueeze(0)  # [1,3,H,W]

            lab = kc.rgb_to_lab(image)  # [1,3,H,W]
            L = lab[:, 0:1, :, :] / 100.0  # [1,1,H,W]

            mb = mask_b3[:, 0:1, :, :]  # [1,1,H,W]
            area_l = mb.sum((2, 3), keepdim=True).clamp(min=1)
            mean_l = (L * mb).sum((2, 3), keepdim=True) / area_l
            Lf = torch.where(mb, L, mean_l)
            Leq = ke.equalize_clahe(
                Lf,
                clip_limit=clip_limit,
                grid_size=grid_size,
                slow_and_differentiable=False,
            ).clamp(0, 1)
            L_blend = alpha_clahe * Leq + (1 - alpha_clahe) * L
            Lnew = torch.where(mb, L_blend, L)

            lab_eq = torch.cat([Lnew * 100.0, lab[:, 1:, :, :]], dim=1)  # [1,3,H,W]
            x_eq = kc.lab_to_rgb(lab_eq)
            image = x_eq.squeeze(0)

        # 3) Gabor Filter setup
        kernel_size = max(1, 2 * kernel_size - 1)
        if theta_count == 10:
            theta_values = torch.tensor([math.pi / 4], device=image.device)
        else:
            theta_values = torch.linspace(
                0, math.pi, theta_count + 1, device=image.device
            )[:-1]

        # 4) Single Gabor Filter call
        magnitude = self.apply_gabor_filter_torch(
            image, kernel_size, sigma, lambd, gamma, psi, theta_values
        )  # [C, H, W]

        # 5) Invert
        max_mv = magnitude.amax((1, 2), keepdim=True)
        inverted = max_mv - magnitude  # [C, H, W]

        # 6) Weighting
        if weighting_strength > 0:
            img_m = image * mask
            weighted = inverted * (
                (1 - weighting_strength) + weighting_strength * img_m
            )
        else:
            weighted = inverted

        return weighted * 255  # [C, H, W]

    def apply_gabor_filter_torch(
        self, image, kernel_size, sigma, lambd, gamma, psi, theta_values
    ):
        """
        Applies Gabor filter bank to image.

        Args:
            image: Tensor [C, H, W]
            theta_values: Tensor [N]
        Returns:
            Tensor [C, H, W]
        """
        C, H, W = image.shape
        image = image.unsqueeze(0)  # → [1, C, H, W]

        N = theta_values.shape[0]

        kernels = self.get_gabor_kernels(
            kernel_size, sigma, lambd, gamma, psi, theta_values, image.device
        )  # [N, 1, k, k]

        # FW-PERF-06: cache expanded kernels keyed by (shape, C)
        expand_cache_key = (*kernels.shape, C)

        if expand_cache_key not in self._gabor_kernels_expanded_cache:
            MAX_GABOR_EXPANDED_CACHE = 16
            if len(self._gabor_kernels_expanded_cache) >= MAX_GABOR_EXPANDED_CACHE:
                self._gabor_kernels_expanded_cache.popitem(last=False)
            self._gabor_kernels_expanded_cache[expand_cache_key] = (
                kernels.repeat_interleave(C, dim=0)
            )

        weight = self._gabor_kernels_expanded_cache[expand_cache_key]
        out = F.conv2d(
            image,  # [1, C, H, W]
            weight,
            padding=kernel_size // 2,
            groups=C,  # each channel group gets N filters
        )  # out: [1, N*C, H, W]

        # reshape to [N, C, H, W]:
        out = out.squeeze(0).view(N, C, H, W)
        magnitudes = out.amax(dim=0)
        return magnitudes

    def get_gabor_kernels(
        self, kernel_size, sigma, lambd, gamma, psi, theta_values, device
    ):
        """
        Returns: Tensor [N, 1, k, k]

        FW-QUAL-3: Kernels are cached by parameter tuple so they are only
        rebuilt when the parameters actually change.
        """
        N = theta_values.shape[0]
        cache_key = (
            int(kernel_size),
            float(sigma),
            float(lambd),
            float(gamma),
            float(psi),
            int(N),
            str(device),
        )
        cached = self._gabor_kernels_cache.get(cache_key)
        if cached is not None:
            return cached

        half = kernel_size // 2
        y, x = torch.meshgrid(
            torch.linspace(-half, half, kernel_size, device=device),
            torch.linspace(-half, half, kernel_size, device=device),
            indexing="ij",
        )

        kernels = []
        for theta in theta_values:
            x_theta = x * torch.cos(theta) + y * torch.sin(theta)
            y_theta = -x * torch.sin(theta) + y * torch.cos(theta)

            gb = torch.exp(-0.5 * (x_theta**2 + (gamma**2) * y_theta**2) / sigma**2)
            gb *= torch.cos(2 * math.pi * x_theta / lambd + psi)
            kernels.append(gb)

        result = torch.stack(kernels).unsqueeze(1)  # → [N, 1, k, k]
        # FW-MEM-01: evict oldest entry if cache exceeds limit
        MAX_GABOR_CACHE = 32
        if len(self._gabor_kernels_cache) >= MAX_GABOR_CACHE:
            self._gabor_kernels_cache.popitem(last=False)
        self._gabor_kernels_cache[cache_key] = result
        return result

    def face_restorer_auto(
        self,
        original_face_512,  # [3,H,W], float in [0..255]
        swap_original,  # [3,H,W]
        swap,  # [3,H,W]
        alpha,  # initial scalar alpha (ignored; we binary search below)
        adjust_sharpness,
        scale_factor,
        debug,
        swap_mask,
        alpha_map_enable: bool = False,
        alpha_map_strength: float = 0.5,
        alpha_map_blur: int = 7,
    ):
        """Auto-Restorer: Blends between restored and original image based on sharpness."""
        # Baseline sharpness of original
        scores_original = self.sharpness_score(original_face_512)
        score_new_original = (
            scores_original["combined"].item() * 100 + adjust_sharpness / 10.0
        )

        # Binary search for scalar alpha
        alpha = 1.0
        max_iterations = 7
        alpha_min, alpha_max = 0.0, 1.0
        tolerance = 0.5
        min_alpha_change = 0.05
        iteration = 0
        prev_alpha = alpha
        iteration_blur = 0

        while iteration < max_iterations:
            swap2 = swap * alpha + swap_original * (1 - alpha)
            swap2_masked = swap2.clone()

            scores_swap = self.sharpness_score(swap2_masked)
            score_new_swap = scores_swap["combined"].item() * 100
            sharpness_diff = score_new_swap - score_new_original

            if abs(sharpness_diff) < tolerance:
                break

            if sharpness_diff < 0:
                if alpha > 0.99:
                    prev_alpha = alpha
                    break
                alpha_min = alpha
                alpha = (alpha + alpha_max) / 2.0
            else:
                alpha_max = alpha
                alpha = (alpha + alpha_min) / 2.0

            # Very small alpha -> blur fallback on base
            if sharpness_diff >= 0 and alpha < 0.07:
                prev_alpha = 0.0
                base = swap_original
                max_blur_strength = 10
                # FW-PERF-10: precompute GaussianBlur objects outside the scoring loop
                blur_kernels_for_auto = [
                    (
                        v2.GaussianBlur(1, 1e-6)
                        if bs == 0
                        else v2.GaussianBlur(2 * bs + 1, max(bs, 1e-6))
                    )
                    for bs in range(0, max_blur_strength + 1)
                ]
                for bs, gaussian_blur in enumerate(blur_kernels_for_auto):
                    swap2_blurred = gaussian_blur(base)
                    scores_swap_b = self.sharpness_score(swap2_blurred)
                    score_new_swap_b = scores_swap_b["combined"].item() * 100.0
                    sharpness_diff_b = score_new_swap_b - score_new_original

                    if sharpness_diff_b < 0:
                        iteration_blur = 0 if bs == 0 else (bs - 1)
                        break
                    if abs(sharpness_diff_b) <= tolerance:
                        iteration_blur = bs
                        break
                    iteration_blur = bs
                break

            if abs(prev_alpha - alpha) < min_alpha_change:
                prev_alpha = (prev_alpha + alpha) / 2.0
                if abs(prev_alpha) <= 0.05:
                    prev_alpha = 0.0
                break

            prev_alpha = alpha
            iteration += 1

        # Per-pixel alpha map, derived from sharpness distribution
        if alpha_map_enable and (prev_alpha > 0.0):
            # Build the *final* composite (for a stable map), then sharpness map of it
            swap_final = swap * prev_alpha + swap_original * (1 - prev_alpha)

            s_map = self.sharpness_map(
                swap_final,
                mask=swap_mask,
                tenengrad_thresh=0.05,
                comb_weight=0.5,
                smooth_kernel=alpha_map_blur
                if (alpha_map_blur and alpha_map_blur % 2 == 1)
                else 0,
            )

            # Mean sharpness inside mask (or global)
            if swap_mask is not None:
                m = (
                    (swap_mask if swap_mask.dim() == 2 else swap_mask.squeeze(0))
                    .float()
                    .to(s_map.device)
                )
                denom = m.sum().clamp_min(1.0)
                mu = (s_map * m).sum() / denom
            else:
                mu = s_map.mean()

            # Deviation map around mean, scale around prev_alpha
            dev = (s_map - mu).clamp(-1.0, 1.0)
            alpha_map = prev_alpha * (1.0 + alpha_map_strength * dev)
            alpha_map = alpha_map.clamp(0.0, 1.0)

            # Keep outside-face area at scalar prev_alpha (if a mask is provided)
            if swap_mask is not None:
                m = (
                    (swap_mask if swap_mask.dim() == 2 else swap_mask.squeeze(0))
                    .float()
                    .to(alpha_map.device)
                )
                alpha_map = alpha_map * m + prev_alpha * (1.0 - m)

            return alpha_map.unsqueeze(0), iteration_blur

        # Fallback: scalar like before
        return prev_alpha, iteration_blur

    def sharpness_score(
        self,
        image: torch.Tensor,
        mask: torch.Tensor = None,
        tenengrad_thresh: float = 0.05,
        comb_weight: float = 0.5,
    ) -> dict:
        """
        Calculates three sharpness metrics on an RGB image:
          1) var_lap: Variance of Laplacian
          2) tten: Thresholded Tenengrad (Proportion of strong edges)
          3) combined: comb_weight*var_lap + (1-comb_weight)*tten

        Args:
            image: Tensor [3, H, W], float in [0..1]
            mask:  optional Tensor [H, W] or [1, H, W] with 1=valid, 0=ignore
            tenengrad_thresh: Threshold for Tenengrad (0..1)
            comb_weight: Weight for var_lap in combination (0..1)

        Returns:
            {
              "var_lap": float Tensor,
              "ttengrad": float Tensor,
              "combined": float Tensor
            }
        """
        image = image / 255.0

        # 1) Grayscale [1,1,H,W]
        gray = image.mean(dim=0, keepdim=True).unsqueeze(0)

        # 2) Optional Mask on [H,W]
        if mask is not None:
            m = mask.float()
            if m.dim() == 3:  # [1,H,W]
                m = m.squeeze(0)
        else:
            m = None

        def valid_count(t):
            return m.sum().clamp(min=1.0) if m is not None else t.numel()

        # --- Variance of Laplacian ---
        # OPTIMIZED: Use pre-allocated kernel from VRAM to avoid CPU micro-allocations
        L = F.conv2d(gray, self.kernel_lap, padding=1).squeeze()  # [H,W]
        L2 = L.pow(2)
        if m is not None:
            L = L * m
            L2 = L2 * m
        cnt = valid_count(L2)
        mean_L2 = L2.sum() / cnt
        mean_L = L.sum() / cnt
        var_lap = (mean_L2 - mean_L.pow(2)).clamp(min=0.0)

        # --- Thresholded Tenengrad ---
        # OPTIMIZED: Use pre-allocated kernels
        Gx = F.conv2d(gray, self.kernel_sobel_x, padding=1).squeeze()  # [H,W]
        Gy = F.conv2d(gray, self.kernel_sobel_y, padding=1).squeeze()
        G = (Gx.pow(2) + Gy.pow(2)).sqrt()
        if m is not None:
            G = G * m
        total = cnt
        strong = (G > tenengrad_thresh).float().sum()
        ttengrad = strong / total

        # --- Combined Score ---
        combined = comb_weight * var_lap + (1 - comb_weight) * ttengrad

        return {"var_lap": var_lap, "ttengrad": ttengrad, "combined": combined}

    def sharpness_map(
        self,
        image: torch.Tensor,  # [3,H,W], float in [0..255]
        mask: torch.Tensor | None = None,
        tenengrad_thresh: float = 0.05,
        comb_weight: float = 0.5,
        smooth_kernel: int = 5,  # odd; 0/1 = no blur
    ) -> torch.Tensor:
        """
        Returns a normalized per-pixel sharpness map in [0..1] with shape [H,W].
        Combines Laplacian energy + gradient magnitude (Tenengrad-like).
        """
        eps = 1e-8
        device = image.device

        # [3,H,W] -> [1,1,H,W] gray, range [0..1]
        gray = (image / 255.0).mean(dim=0, keepdim=True).unsqueeze(0)

        # OPTIMIZED: Convs using pre-allocated VRAM kernels
        lap = F.conv2d(gray, self.kernel_lap, padding=1).squeeze(0).squeeze(0)  # [H,W]
        gx = (
            F.conv2d(gray, self.kernel_sobel_x, padding=1).squeeze(0).squeeze(0)
        )  # [H,W]
        gy = F.conv2d(gray, self.kernel_sobel_y, padding=1).squeeze(0).squeeze(0)
        grad = (gx.pow(2) + gy.pow(2)).sqrt()  # [H,W]

        # Robust normalization via percentiles inside mask (if given)
        def robust_norm(x, msk):
            if msk is not None:
                sel = x[msk > 0]
                if sel.numel() < 16:  # fallback if mask tiny
                    sel = x.reshape(-1)
            else:
                sel = x.reshape(-1)
            p5 = (
                torch.quantile(sel, 0.05)
                if sel.numel() > 0
                else torch.tensor(0.0, device=device)
            )
            p95 = (
                torch.quantile(sel, 0.95)
                if sel.numel() > 0
                else torch.tensor(1.0, device=device)
            )
            y = (x - p5) / (p95 - p5 + eps)
            return y.clamp_(0, 1)

        m = None
        if mask is not None:
            m = (mask if mask.dim() == 2 else mask.squeeze(0)).float().to(device)

        lap_n = robust_norm(lap.abs(), m)
        grad_n = robust_norm(grad, m)

        smap = comb_weight * lap_n + (1.0 - comb_weight) * grad_n  # [H,W]

        # Optional smoothing to avoid noisy alpha
        # FW-ROBUST-09: ensure kernel size is odd and at least 3
        if smooth_kernel:
            if smooth_kernel % 2 == 0:
                smooth_kernel += 1
            smooth_kernel = max(3, smooth_kernel)
        if smooth_kernel and smooth_kernel >= 3:
            k = smooth_kernel
            smap3 = smap.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
            gb = v2.GaussianBlur(kernel_size=k, sigma=max(1, k // 2))
            smap = gb(smap3).squeeze(0).squeeze(0)

        return smap.clamp(0, 1)

    @torch.no_grad()
    def apply_block_shift_gpu_jitter(
        self,
        img: torch.Tensor,
        block_size: int,
        max_amount_pixels: float,
        *,
        seed: int = 1337,
        pad_mode: str = "replicate",
        align_corners: bool = True,
    ) -> torch.Tensor:
        """
        MPEG-like Block Jitter: shifts every BxB block field by a
        deterministic (bx, by)-dependent offset in pixels.

        Args:
            img: Tensor [C, H, W] (BGR/RGB agnostic). CPU or CUDA.
            block_size: Block size B (e.g. 8).
            max_amount_pixels: max |Offset| in pixels (applied to both axes).
            seed: global seed for deterministic offsets (frame-stable).
            pad_mode: Padding mode for border (replicate|reflect|zeros).
            align_corners: as in grid_sample.

        Returns:
            Tensor [C, H, W] – same Device/Dtype as input.
        """
        seed = seed + self.worker.frame_number * 17
        assert img.ndim == 3, "expected [C,H,W]"
        C, H, W = img.shape
        device = img.device
        dtype = img.dtype

        # calculate on float32 for grid_sample if necessary
        work = (
            img
            if img.dtype in (torch.float32, torch.float16, torch.bfloat16)
            else img.float()
        )

        # Pad to multiples of B (bottom/right), crop back later
        # FW-BUG-13: use block_size directly as B; old 2**block_size was exponential
        B = max(1, int(block_size))
        H_pad = (B - (H % B)) % B
        W_pad = (B - (W % B)) % B
        if H_pad or W_pad:
            pad = (0, W_pad, 0, H_pad)  # (left, right, top, bottom)
            mode = {
                "replicate": "replicate",
                "reflect": "reflect",
                "zeros": "constant",
            }[pad_mode]
            work = F.pad(work[None], pad=pad, mode=mode).squeeze(0)
        Hp, Wp = work.shape[-2:]

        # Number of blocks
        nby = Hp // B
        nbx = Wp // B

        # --- deterministic offsets per block in range [-max, +max] ---
        # Build block coordinate fields
        by_grid, bx_grid = torch.meshgrid(
            torch.arange(nby, device=device, dtype=torch.float32),
            torch.arange(nbx, device=device, dtype=torch.float32),
            indexing="ij",
        )
        # simple Hash -> [0,1)
        h = torch.sin((bx_grid * 12.9898 + by_grid * 78.233 + float(seed)) * 43758.5453)
        frac = torch.frac(h * 0.5 + 0.5)

        # derive two independent offsets from hash
        # The / 4 scales the slider range to produce subtle MPEG-like block artifacts
        # rather than extreme pixel shifts (slider values of 40-100 become 10-25 px).
        _scaled_max = float(max_amount_pixels) / 4.0
        dx_base = ((frac) * 2.0 - 1.0) * _scaled_max

        # second "source": just another linear combo
        h2 = torch.sin(
            (bx_grid * 96.233 + by_grid * 15.987 + (float(seed) + 101)) * 12345.6789
        )
        frac2 = torch.frac(h2 * 0.5 + 0.5)
        dy_base = ((frac2) * 2.0 - 1.0) * _scaled_max

        # upsample to pixel grid by tiling each block offset BxB
        dx = torch.repeat_interleave(
            torch.repeat_interleave(dx_base, B, dim=0), B, dim=1
        )  # [Hp,Wp]
        dy = torch.repeat_interleave(
            torch.repeat_interleave(dy_base, B, dim=0), B, dim=1
        )  # [Hp,Wp]

        # --- Build Flow-Field for grid_sample ---
        xs = torch.linspace(-1.0, 1.0, Wp, device=device)
        ys = torch.linspace(-1.0, 1.0, Hp, device=device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")  # [Hp,Wp]
        dx_norm = (2.0 * dx) / max(Wp - 1, 1)
        dy_norm = (2.0 * dy) / max(Hp - 1, 1)

        flow_x = grid_x + dx_norm
        flow_y = grid_y + dy_norm
        flow = torch.stack([flow_x, flow_y], dim=-1)  # [Hp,Wp,2]

        warped = F.grid_sample(
            work[None],
            flow[None],
            mode="bilinear",
            padding_mode="border",
            align_corners=align_corners,
        ).squeeze(0)

        # crop back to original size if padded
        if H_pad or W_pad:
            warped = warped[..., :H, :W]

        if warped.dtype != dtype:
            warped = warped.to(dtype)

        return warped

    def analyze_image(self, image):
        """
        Analyses a CHW uint8 image tensor and returns a dict of quality scores in [0, 1].

        Computed metrics:
          - ``jpeg_artifacts``: High-frequency energy (higher = more ringing/blocking).
          - ``salt_pepper_noise``: Fraction of pixels with sharp local outliers.
          - ``speckle_noise``: Mean local variance (higher = more speckle).
          - ``blur``: Inverted Laplacian edge strength (higher = blurrier).
          - ``low_contrast``: Inverted pixel standard deviation (higher = flatter).
        """
        image = image.float() / 255.0
        C, H, W = image.shape
        grayscale = torch.mean(image, dim=0, keepdim=True)
        analysis = {}
        fft = torch.fft.fft2(grayscale)
        high_freq_energy = torch.mean(torch.abs(fft))
        analysis["jpeg_artifacts"] = min(high_freq_energy.item() / 50, 1.0)
        median_filtered = F.avg_pool2d(grayscale, 3, stride=1, padding=1)
        noise_map = torch.abs(grayscale - median_filtered)
        sp_noise = torch.mean((noise_map > 0.1).float())
        analysis["salt_pepper_noise"] = min(sp_noise.item() * 10, 1.0)
        local_var = F.avg_pool2d(grayscale**2, 5, stride=1, padding=2) - (
            F.avg_pool2d(grayscale, 5, stride=1, padding=2) ** 2
        )
        speckle_noise = torch.mean(local_var)
        analysis["speckle_noise"] = min(speckle_noise.item() * 50, 1.0)

        # OPTIMIZED: Use pre-allocated Laplacian kernel
        laplace_edges = F.conv2d(grayscale.unsqueeze(0), self.kernel_lap, padding=1)
        edge_strength = torch.mean(torch.abs(laplace_edges))
        analysis["blur"] = 1.0 - min(edge_strength.item() * 5, 1.0)
        contrast = grayscale.std()
        analysis["low_contrast"] = 1.0 - min(contrast.item() * 10, 1.0)
        return analysis
