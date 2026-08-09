from typing import TYPE_CHECKING, Dict

import torch
import threading
import cv2
import numpy as np
from torchvision import transforms
from torchvision.transforms import v2
import torch.nn.functional as F

from app.processors.external.clipseg import CLIPDensePredT
from app.processors.models_data import models_dir

if TYPE_CHECKING:
    from app.processors.models_processor import ModelsProcessor
    from app.processors.workers.function_worker import FunctionWorker

_VGG_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_VGG_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class FaceMasks:
    """
    Manages mask generation and manipulation for face swapping.
    Handles FaceParser, Occluder, XSeg, CLIP, and VGG-based differencing.
    """

    def __init__(
        self,
        models_processor: "ModelsProcessor",
        function_worker: "FunctionWorker",
    ):
        self.models_processor = models_processor
        self.function_worker = function_worker

        # Caches for morphological operations to avoid re-allocating tensors
        self._morph_kernels: Dict[tuple, torch.Tensor] = {}
        self._kernel_cache: Dict[str, torch.Tensor] = {}
        self._meshgrid_cache: Dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}

        # --- Thread safety locks for caches and lazy loading ---
        self._morph_cache_lock = threading.Lock()
        self._kernel_cache_lock = threading.Lock()
        self._meshgrid_cache_lock = threading.Lock()
        self._clip_load_lock = threading.Lock()

        self.clip_model_loaded = False
        self.active_models: set[str] = set()

    def unload_models(self):
        """Unloads all models managed by this class via the processor."""
        with self.models_processor.model_lock:
            for model_name in list(self.active_models):
                self.models_processor.unload_model(model_name)
            self.active_models.clear()

    # --- Inference Helpers ---

    def _faceparser_labels(self, img_uint8_3x512x512: torch.Tensor) -> torch.Tensor:
        """
        Runs FaceParser on a 512x512 input.
        Returns: NATIVE 512x512 label tensor (Long).
        """
        model_name = "FaceParser"

        # Preprocessing: [0..255] -> [0..1] -> Normalize (shared by all paths)
        x = img_uint8_3x512x512.float().div(255.0)
        x = v2.functional.normalize(x, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        x = x.unsqueeze(0).contiguous()

        # ORT path
        ort_session = self.models_processor.models.get(model_name)
        if not ort_session:
            ort_session = self.models_processor.load_model(model_name)

        if not ort_session:
            # Fallback empty labels
            return torch.zeros(
                (512, 512), dtype=torch.long, device=img_uint8_3x512x512.device
            )

        # Binding I/O
        out = torch.empty((1, 19, 512, 512), device=self.models_processor.device)
        io = ort_session.io_binding()
        io.bind_input(
            "input",
            self.models_processor.device_type,
            self.models_processor.binding_device_id,
            np.float32,
            (1, 3, 512, 512),
            x.data_ptr(),
        )
        io.bind_output(
            "output",
            self.models_processor.device_type,
            self.models_processor.binding_device_id,
            np.float32,
            (1, 19, 512, 512),
            out.data_ptr(),
        )

        # Handle Lazy TensorRT Build
        is_lazy_build = self.models_processor.check_and_clear_pending_build(model_name)
        if is_lazy_build:
            self.models_processor.show_build_dialog.emit(
                "Finalizing TensorRT Build",
                f"Performing first-run inference for:\n{model_name}\n\nThis may take several minutes.",
            )

        try:
            self.function_worker.run_ort_with_iobinding(ort_session, io)
        finally:
            if is_lazy_build:
                self.models_processor.hide_build_dialog.emit()

        # Argmax to get class indices: (1, 19, 512, 512) -> (512, 512)
        labels_512 = out.argmax(dim=1).squeeze(0).to(torch.long)
        return labels_512

    # --- Mouth Processing Logic ---

    def _enhance_and_align_swapped_mouth(
        self,
        swap_img: torch.Tensor,
        labels_swap: torch.Tensor,
        parameters: dict,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """
        Isolates the inner mouth from the raw swap, applies Thresholded Edge-Aware
        Unsharp Masking (USM) if requested, and applies the user's Zoom (Scale)
        using Center of Mass for stable anchoring.
        """
        # Group labels for performance
        mouth_swap = (labels_swap == 11) | (labels_swap == 12) | (labels_swap == 13)
        inner_swap = labels_swap == 11
        lips_swap = (labels_swap == 12) | (labels_swap == 13)

        if inner_swap.sum() == 0:
            return None, None

        # FM-16: guard against empty mouth_swap before calling max()/min() on it
        if mouth_swap.sum() == 0:
            return None, None

        enhanced_swap_img = swap_img.clone()

        # --- Thresholded Edge-Aware Unsharp Masking (USM) ---
        sharpen_amount = parameters.get("MouthParserStretchSharpenDecimalSlider", 0.0)

        if sharpen_amount > 0.0:
            y_ind, x_ind = torch.where(mouth_swap)
            # Add padding for processing
            ymin, ymax = (
                max(0, y_ind.min().item() - 15),
                min(swap_img.shape[1], y_ind.max().item() + 15),
            )
            xmin, xmax = (
                max(0, x_ind.min().item() - 15),
                min(swap_img.shape[2], x_ind.max().item() + 15),
            )

            mouth_crop = swap_img[:, ymin:ymax, xmin:xmax].clone().float()

            # 1. Apply Gaussian Blur to find the base frequencies
            blurred_mouth = v2.functional.gaussian_blur(
                mouth_crop, kernel_size=5, sigma=1.0
            )

            # 2. Subtract blurred image from original to isolate edges
            high_freq_edges = mouth_crop - blurred_mouth

            # Threshold to protect smooth skin/noise
            # Only amplify edges if their magnitude is above 5.0 (out of 255)
            threshold = 3.0
            mask_edges = (torch.abs(high_freq_edges) > threshold).float()

            # 3. Add the amplified edges back to the original image ONLY where threshold is met
            sharpened_mouth = mouth_crop + (
                sharpen_amount * high_freq_edges * mask_edges
            )

            sharpened_mouth = torch.clamp(sharpened_mouth, 0.0, 255.0).to(
                swap_img.dtype
            )
            enhanced_swap_img[:, ymin:ymax, xmin:xmax] = sharpened_mouth

        # --- Alignment Logic (Statistical Anchoring) ---
        y_s_full, x_s_full = torch.where(mouth_swap)

        # X-axis centroid of the entire mouth (highly stable spatial reference)
        cx_s = x_s_full.float().mean()

        # FW-PERF-FIX: Pure GPU control flow (CUDA Sync Avoidance)
        # Using pure boolean tensor logic avoids CPU-GPU synchronization.
        mask_upper = labels_swap == 12
        has_upper = mask_upper.any()

        # Dynamically select the mask: if upper lip exists, use it. Otherwise, fallback to inner_swap.
        target_mask = mask_upper | (inner_swap & ~has_upper)
        y_s_target, _ = torch.where(target_mask)

        # FW-BUG-FIX: "Statistical Boundary" anchoring.
        # Calculates the teeth line by averaging all pixels of the target mask.
        mean_ys = y_s_target.float().mean()
        std_ys = y_s_target.float().std()

        # Gracefully handle NaN strictly on the GPU (e.g., if only 1 pixel is found, std is NaN)
        std_ys = torch.nan_to_num(std_ys, nan=0.0)

        # The center of the reference region + 1.5x its standard deviation = optimal teeth alignment line
        cy_s = mean_ys + 1.5 * std_ys

        mouthzoom = parameters.get("MouthParserStretchDecimalSlider", 1.05)

        # Affine transform parameters
        affine_kwargs = {
            "angle": 0.0,
            "translate": [0.0, 0.0],
            "scale": mouthzoom,
            "shear": [0.0, 0.0],
            "center": [cx_s.item(), cy_s.item()],
        }

        overlay = v2.functional.affine(
            enhanced_swap_img,
            interpolation=v2.InterpolationMode.BILINEAR,
            **affine_kwargs,
        )

        inner_swap_transformed = v2.functional.affine(
            inner_swap.unsqueeze(0).float(),
            interpolation=v2.InterpolationMode.NEAREST,
            **affine_kwargs,
        ).squeeze(0)

        lips_swap_transformed = v2.functional.affine(
            lips_swap.unsqueeze(0).float(),
            interpolation=v2.InterpolationMode.NEAREST,
            **affine_kwargs,
        ).squeeze(0)

        lips_swap_transformed = self._dilate_binary(
            lips_swap_transformed, 1, mode="conv"
        )

        # Combine masks
        overlay_mask = inner_swap.float()
        overlay_mask = overlay_mask * (1.0 - lips_swap_transformed)
        overlay_mask = torch.minimum(overlay_mask, inner_swap_transformed)
        overlay_mask = self._dilate_binary(overlay_mask, 1, mode="conv")

        # Dynamic blur size based on mouth width to prevent harsh seams
        mouth_width = x_s_full.max() - x_s_full.min()
        dynamic_kernel = (
            int(mouth_width.item() * 0.15) | 1
        )  # 15% of width, force odd number
        dynamic_kernel = max(5, min(31, dynamic_kernel))  # Clamp between 5 and 31

        overlay_mask = v2.functional.gaussian_blur(
            overlay_mask.unsqueeze(0),
            kernel_size=dynamic_kernel,
            sigma=dynamic_kernel / 3.0,
        ).squeeze(0)

        # Multiply by inner_swap to prevent blur from bleeding onto the lips
        final_mask = overlay_mask * inner_swap.float()

        return overlay, final_mask

    def _enhance_and_align_original_mouth(
        self,
        img_orig: torch.Tensor,
        img_swap: torch.Tensor,
        labels_orig: torch.Tensor,
        labels_swap: torch.Tensor,
        parameters: dict,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """
        Creates an aligned mouth overlay from the original image to the swap image.
        Strictly limited to the INNER mouth (class 11) to avoid overriding swapped lips.
        """
        # Cleared original mouth from obstacle
        clear_mask = self._get_obstacle_mask(img_orig, parameters)
        inner_orig = (labels_orig == 11) & clear_mask
        inner_swap = labels_swap == 11

        # We strictly require the inner mouth.
        # If the mouth is closed (or completely covered), we cannot restore original teeth/tongue.
        if inner_orig.sum() == 0 or inner_swap.sum() == 0:
            return None, None

        mouth_orig = (
            (labels_orig == 11) | (labels_orig == 12) | (labels_orig == 13)
        ) & clear_mask
        mouth_swap = (labels_swap == 11) | (labels_swap == 12) | (labels_swap == 13)

        enhanced_img_orig = img_orig.clone()

        # --- Thresholded Edge-Aware Unsharp Masking (USM) ---
        sharpen_amount = parameters.get("MouthParserStretchSharpenDecimalSlider", 0.0)

        if sharpen_amount > 0.0:
            y_ind, x_ind = torch.where(mouth_orig)
            ymin, ymax = (
                max(0, y_ind.min().item() - 15),
                min(img_orig.shape[1], y_ind.max().item() + 15),
            )
            xmin, xmax = (
                max(0, x_ind.min().item() - 15),
                min(img_orig.shape[2], x_ind.max().item() + 15),
            )

            mouth_crop = img_orig[:, ymin:ymax, xmin:xmax].clone().float()
            blurred_mouth = v2.functional.gaussian_blur(
                mouth_crop, kernel_size=5, sigma=1.0
            )
            high_freq_edges = mouth_crop - blurred_mouth

            threshold = 3.0
            mask_edges = (torch.abs(high_freq_edges) > threshold).float()

            sharpened_mouth = mouth_crop + (
                sharpen_amount * high_freq_edges * mask_edges
            )
            sharpened_mouth = torch.clamp(sharpened_mouth, 0.0, 255.0).to(
                img_orig.dtype
            )
            enhanced_img_orig[:, ymin:ymax, xmin:xmax] = sharpened_mouth

        # --- Alignment Logic (Statistical Anchoring) ---
        y_o_full, x_o_full = torch.where(mouth_orig)
        y_s_full, x_s_full = torch.where(mouth_swap)

        std_x_o = x_o_full.float().std()
        std_x_s = x_s_full.float().std()
        std_y_o = y_o_full.float().std()
        std_y_s = y_s_full.float().std()

        if (
            std_x_o <= 0.0
            or std_x_s <= 0.0
            or torch.isnan(std_x_o)
            or torch.isnan(std_x_s)
            or std_y_o <= 0.0
            or std_y_s <= 0.0
            or torch.isnan(std_y_o)
            or torch.isnan(std_y_s)
        ):
            return None, None

        mouthzoom = parameters.get("MouthParserStretchDecimalSlider", 1.05)
        scale_factor = (std_x_s / std_x_o) * mouthzoom

        # 2. X-AXIS CENTROID
        cx_o = x_o_full.float().mean()
        cx_s = x_s_full.float().mean()

        # 3. Y-AXIS ANCHORING (Statistical Boundary of the upper lip)
        # FW-PERF-FIX: Pure GPU control flow (CUDA Sync Avoidance)
        # Replaced CPU-side 'len()' checks with pure boolean tensor logic.

        # --- Original Mouth Anchoring ---
        mask_o_upper = labels_orig == 12
        has_o_upper = mask_o_upper.any()

        # Fallback to the entire original mouth if no upper lip is detected
        target_mask_o = mask_o_upper | (mouth_orig & ~has_o_upper)
        y_o_target, _ = torch.where(target_mask_o)

        mean_yo = y_o_target.float().mean()
        std_yo = y_o_target.float().std()
        std_yo = torch.nan_to_num(std_yo, nan=0.0)
        cy_o = mean_yo + 1.5 * std_yo

        # --- Swapped Mouth Anchoring ---
        mask_s_upper = labels_swap == 12
        has_s_upper = mask_s_upper.any()

        # Fallback to the entire swapped mouth if no upper lip is detected
        target_mask_s = mask_s_upper | (mouth_swap & ~has_s_upper)
        y_s_target, _ = torch.where(target_mask_s)

        mean_ys = y_s_target.float().mean()
        std_ys = y_s_target.float().std()
        std_ys = torch.nan_to_num(std_ys, nan=0.0)
        cy_s = mean_ys + 1.5 * std_ys

        translate_x = cx_s - cx_o
        translate_y = cy_s - cy_o

        affine_kwargs = {
            "angle": 0.0,
            "translate": [translate_x.item(), translate_y.item()],
            "scale": scale_factor.item(),
            "shear": [0.0, 0.0],
            "center": [cx_o.item(), cy_o.item()],
        }

        overlay = v2.functional.affine(
            enhanced_img_orig,
            interpolation=v2.InterpolationMode.BILINEAR,
            **affine_kwargs,
        )

        # Inner mask alignment
        inner_orig_transformed = v2.functional.affine(
            inner_orig.unsqueeze(0).float(),
            interpolation=v2.InterpolationMode.NEAREST,
            **affine_kwargs,
        ).squeeze(0)

        # 1. Isolate the real content we want to keep (Original teeth/tongue)
        content_mask = torch.minimum(inner_swap.float(), inner_orig_transformed)

        content_mask_blurred = v2.functional.gaussian_blur(
            content_mask.unsqueeze(0), kernel_size=5, sigma=1.0
        ).squeeze(0)

        w_s = (x_s_full.max() - x_s_full.min()).float()

        # 2. Destroy the fake teeth with blur (Controlled by UI Slider)
        cavity_blur_pct = parameters.get("MouthOriginalCavityBlurSlider", 15) / 100.0
        # Calculate dynamic kernel size based on mouth width and user slider
        blur_kernel = int(w_s.item() * cavity_blur_pct) | 1
        # Clamp to prevent PyTorch crash (min 3) and extreme VRAM usage (max 121)
        blur_kernel = max(3, min(121, blur_kernel))

        blurred_swap = v2.functional.gaussian_blur(
            img_swap.clone().float(), kernel_size=blur_kernel, sigma=blur_kernel / 3.0
        )

        # DARKENING: Non-linear Gamma Correction (Controlled by UI Slider)
        cavity_gamma = parameters.get("MouthOriginalCavityDarkenDecimalSlider", 1.5)
        blurred_swap_norm = blurred_swap / 255.0
        dark_cavity = torch.pow(blurred_swap_norm, cavity_gamma) * 255.0

        # --- Original Cavity Blending ---
        # Allow the user to inject the aligned original mouth back into the cavity void
        # 0.0 = Pure artificial darkness | 1.0 = Pure original aligned background
        cavity_blend = parameters.get("MouthOriginalCavityBlendingDecimalSlider", 0.0)
        cavity_background = dark_cavity * (1.0 - cavity_blend) + overlay * cavity_blend

        # 3. Composite the overlay
        cavity_layer = overlay * content_mask_blurred + cavity_background * (
            1.0 - content_mask_blurred
        )

        # 4. WIDEN THE MASK to catch persistent edge pixels
        overlay_mask = self._dilate_binary(content_mask, 3, mode="conv")
        dynamic_kernel = max(5, min(31, int(w_s.item() * 0.15) | 1))
        blurred_mask = v2.functional.gaussian_blur(
            overlay_mask.unsqueeze(0),
            kernel_size=dynamic_kernel,
            sigma=dynamic_kernel / 3.0,
        ).squeeze(0)

        # Restrict strictly to inner_swap to prevent bleeding onto the swapped lips
        # 1. Allow a maximum physical expansion of exactly 2 pixels onto the lips
        allowed_bleed_region = self._dilate_binary(content_mask, 2, mode="conv")

        # 2. Blur this strict boundary slightly so it acts as a smooth braking gradient
        soft_limit = v2.functional.gaussian_blur(
            allowed_bleed_region.unsqueeze(0), kernel_size=3, sigma=2.0
        ).squeeze(0)

        # 3. Force the absolute inner mouth to remain strictly untouched (100% opaque)
        soft_limit = torch.maximum(soft_limit, content_mask)

        # 4. Multiply the final mask by this soft limit to gently but rapidly fade out any excess blur
        final_mask = blurred_mask * soft_limit

        # PHASE 2: ALIGNED TONGUE OVERRIDE
        tongue_alpha = None
        if parameters.get("RestoreTongueToggle", True):
            candidate_orig = ((labels_orig == 11) | (labels_orig == 13)).float()
            candidate_orig_transformed = v2.functional.affine(
                candidate_orig.unsqueeze(0),
                interpolation=v2.InterpolationMode.NEAREST,
                **affine_kwargs,
            ).squeeze(0)

            y_inner_s, _ = torch.where(inner_swap)
            cavity_s_h = (
                (y_inner_s.max() - y_inner_s.min()).float()
                if len(y_inner_s) > 0
                else 0.0
            )

            # The physical mouth hole must be open to allow a tongue
            if (
                len(y_s_full) > 0
                and candidate_orig_transformed.sum() > 0
                and cavity_s_h > 4.0
            ):
                y_swap_bottom = y_s_full.max().float()
                mouth_s_w = (x_s_full.max() - x_s_full.min()).float() + 1e-5
                mouth_s_h = (y_s_full.max() - y_s_full.min()).float() + 1e-5

                # Create the Spill Zone (Below the swapped mouth + 2 pixel alignment buffer)
                y_coords = torch.arange(
                    candidate_orig_transformed.shape[0],
                    device=candidate_orig_transformed.device,
                ).view(-1, 1)
                spill_zone_mask = (y_coords > (y_swap_bottom + 2.0)).float()

                raw_spill = candidate_orig_transformed * spill_zone_mask
                y_spill, x_spill = torch.where(raw_spill > 0.5)

                # Ignore micro-noise (spill must be physically > 4 pixels deep)
                if len(y_spill) > 10 and (y_spill.max() - y_spill.min()).float() > 4.0:
                    spill_w = (x_spill.max() - x_spill.min()).float()
                    spill_depth = (y_spill.max() - y_spill.min()).float()

                    spill_w_ratio = spill_w / mouth_s_w
                    spill_depth_ratio = spill_depth / mouth_s_h

                    # SHAPE GUARDS
                    is_localized = (
                        spill_w_ratio < 0.70
                    )  # Narrow enough to be a standard tongue
                    is_deep = (
                        spill_depth_ratio > 0.12
                    )  # Deep enough to definitively be a protrusion
                    is_massive = (
                        spill_depth_ratio > 0.22
                    )  # "Deep Override": So deep it physically must be a tongue, ignoring width limits

                    if is_deep and (is_localized or is_massive):
                        # Adapts dynamically to x_spill. If the tongue is narrow, the shield is narrow.
                        # If the tongue is massive, the shield opens wide automatically.
                        padding = int(
                            mouth_s_w * 0.10
                        )  # 10% padding for smooth blending
                        x_min = max(0, x_spill.min().item() - padding)
                        x_max = min(
                            candidate_orig_transformed.shape[1],
                            x_spill.max().item() + padding,
                        )

                        horizontal_mask = torch.zeros_like(candidate_orig_transformed)
                        horizontal_mask[:, x_min:x_max] = 1.0

                        tongue_base = candidate_orig_transformed * horizontal_mask

                        # Shield the SWAPPED upper lip to protect lips
                        upper_lip_swap = (labels_swap == 12).float()
                        shield = self._dilate_binary(upper_lip_swap, 3, mode="pool")

                        raw_tongue = tongue_base * (1.0 - shield)
                        raw_tongue = self._dilate_binary(
                            raw_tongue, -2, mode="pool"
                        )  # Fill gaps

                        # TONGUE HOLE FILLING (OpenCV Contours)
                        tongue_np = (raw_tongue.cpu().numpy() * 255).astype(np.uint8)
                        contours, _ = cv2.findContours(
                            tongue_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                        )
                        if len(contours) > 0:
                            cv2.drawContours(
                                tongue_np, contours, -1, 255, thickness=cv2.FILLED
                            )

                        raw_tongue = (
                            torch.from_numpy(tongue_np).to(
                                device=raw_tongue.device, dtype=torch.float32
                            )
                            / 255.0
                        )

                        # Blur blending
                        blend_val = parameters.get("TongueBlurDecimalSlider", 0.40)
                        if blend_val > 1.0:
                            blend_val = blend_val / 100.0

                        blur_k = max(3, min(35, int(blend_val * 50) | 1))

                        tongue_alpha = v2.functional.gaussian_blur(
                            raw_tongue.unsqueeze(0),
                            kernel_size=blur_k,
                            sigma=blur_k / 3.0,
                        ).squeeze(0)

        # Composite the final overlay
        if tongue_alpha is not None:
            overlay = overlay * tongue_alpha + cavity_layer * (1.0 - tongue_alpha)
            final_mask = torch.maximum(final_mask, tongue_alpha)
        else:
            overlay = cavity_layer

        return overlay, final_mask

    def get_mouth_overlay(
        self, swap_img: torch.Tensor, original_img: torch.Tensor, parameters: dict
    ) -> tuple[torch.Tensor | None, torch.Tensor | None] | None:
        """
        Public helper to retrieve the mouth overlay based on UI parameters.
        Routes to original mouth alignment or swapped mouth enhancement & zoom.
        """
        if not parameters.get("MouthParserStretchToggle", False):
            return None

        labels_swap = self._faceparser_labels(swap_img)

        if parameters.get("MouthParserStretchOriginalToggle", False):
            labels_orig = self._faceparser_labels(original_img)

            return self._enhance_and_align_original_mouth(
                img_orig=original_img,
                img_swap=swap_img,
                labels_orig=labels_orig,
                labels_swap=labels_swap,
                parameters=parameters,
            )
        else:
            return self._enhance_and_align_swapped_mouth(
                swap_img, labels_swap, parameters
            )

    def _get_obstacle_mask(
        self, img_512: torch.Tensor, parameters: dict
    ) -> torch.Tensor:
        """
        Executes the Occluder and/or XSeg models on the original image to identify obstacles (e.g., hands, microphones).
        Returns a boolean mask (512x512) where True = Clear area (face), False = Obstacle.
        """
        device = img_512.device

        need_occluder = parameters.get("OccluderEnableToggle", False)
        need_xseg = parameters.get("DFLXSegEnableToggle", False)

        # Early exit: If no occlusion models are enabled, allocate and return a full clear mask.
        # This saves VRAM allocation and prevents unnecessary tensor creation when unused.
        if not need_occluder and not need_xseg:
            return torch.ones((512, 512), dtype=torch.bool, device=device)

        # Downscale once to 256x256 since all occlusion models operate at this resolution.
        img_256 = v2.functional.resize(img_512, [256, 256], antialias=True)

        # We will combine masks at 256x256 first.
        # Doing boolean math at 256x256 processes 4x fewer pixels than at 512x512.
        combined_mask_256 = None

        if need_occluder:
            # apply_occlusion returns 1 for Face, 0 for Obstacle. We keep areas > 0.5.
            occ_mask = self.apply_occlusion(img_256, amount=1, parameters=parameters)
            combined_mask_256 = (occ_mask > 0.5).squeeze(0)

        if need_xseg:
            # Dummy mouth in 3D [1, 256, 256] shape to prevent PyTorch dimension crashes
            dummy_mouth = torch.zeros((1, 256, 256), device=device)

            # apply_dfl_xseg INVERTS its mask internally (outpred = 1.0 - outpred).
            # It returns 1 for Obstacle/Background and 0 for Face. We keep areas < 0.5.
            xseg_mask, _, _, _ = self.apply_dfl_xseg(
                img_256, amount=1, mouth=dummy_mouth, parameters=parameters
            )
            xseg_bool_mask = (xseg_mask < 0.5).squeeze(0)

            # Combine with occluder mask if it exists, otherwise initialize it
            if combined_mask_256 is None:
                combined_mask_256 = xseg_bool_mask
            else:
                combined_mask_256 = combined_mask_256 & xseg_bool_mask
        assert combined_mask_256 is not None
        # Upscale the final combined boolean mask back to 512x512 in one single operation.
        # NEAREST interpolation is essential here to maintain boolean values (True/False).
        # We unsqueeze(0) to satisfy torchvision v2 requirements (expects C, H, W) and squeeze(0) after.
        final_mask_512 = v2.functional.resize(
            combined_mask_256.unsqueeze(0),
            [512, 512],
            interpolation=v2.InterpolationMode.NEAREST,
        ).squeeze(0)

        return final_mask_512

    # --- Main Mask Processing Pipeline ---

    def process_masks_and_masks(
        self,
        swap_restorecalc: torch.Tensor,
        original_face_512: torch.Tensor,
        parameters: dict,
        control: dict,
    ) -> dict:
        """
        Generates all necessary masks (FaceParser, Mouth, Texture) based on settings.

        Args:
            swap_restorecalc: The swapped/restored face tensor.
            original_face_512: The original face tensor.
            parameters: Global parameters.
            control: UI controls.

        Returns:
            Dictionary containing the generated masks.
        """
        device = self.models_processor.device
        mode = control.get("DilatationTypeSelection", "conv")
        result = {"swap_formask": swap_restorecalc}

        target_h, target_w = swap_restorecalc.shape[1], swap_restorecalc.shape[2]

        # OPTIMIZED: Replaced expensive class instantiation with a lightweight functional wrapper.
        # Removed antialias=True to prevent ringing artifacts (values shooting below 0.0 or above 1.0)
        # on hard edges, which caused VRAM fragmentation and unpredictable alpha blending.
        def resize_to_target(
            tensor: torch.Tensor,
            interp_mode: v2.InterpolationMode = v2.InterpolationMode.BILINEAR,
        ) -> torch.Tensor:
            return v2.functional.resize(
                tensor,
                [target_h, target_w],
                interpolation=interp_mode,
                antialias=False,
            )

        # --- Check Requirements ---
        need_mouth_stretch = parameters.get("MouthParserStretchToggle", False)
        need_xseg_mouth_protection = parameters.get(
            "XSegExcludeInnerMouthToggle", False
        )
        need_parser = parameters.get("FaceParserEnableToggle", False) or (
            (
                parameters.get("TransferTextureEnableToggle", False)
                or parameters.get("DifferencingEnableToggle", False)
            )
            and parameters.get("ExcludeMaskEnableToggle", False)
        )
        need_parser_mouth = (
            parameters.get("DFLXSegEnableToggle", False)
            and parameters.get("XSegMouthEnableToggle", False)
            and parameters.get("DFLXSegSizeSlider", 0)
            != parameters.get("DFLXSeg2SizeSlider", 0)
        )

        labels_swap = None
        labels_orig = None

        # Determine if we need to run FaceParser
        if (
            need_parser
            or need_parser_mouth
            or need_mouth_stretch
            or need_xseg_mouth_protection
        ):
            labels_swap = self._faceparser_labels(swap_restorecalc)

        # We need Original labels if Parser/MouthStretch/ExcludeMask is active
        should_get_orig_labels = need_mouth_stretch or (
            need_parser
            and (
                parameters.get("FaceParserEnableToggle", False)
                or parameters.get("ExcludeMaskEnableToggle", False)
            )
        )

        if should_get_orig_labels:
            labels_orig = self._faceparser_labels(original_face_512)

        # FM-12: removed dead code (was triple-quoted string acting as a comment)
        # ---------- 1. MOUTH FIT & ALIGN LOGIC ----------
        # if need_mouth_stretch and labels_swap is not None:
        #     overlay, overlay_mask = self._enhance_swapped_mouth(
        #         swap_restorecalc, labels_swap, parameters
        #     )
        #
        #     if overlay is not None:
        #         if overlay.shape[1] != target_h:
        #             overlay = resize_to_target(overlay)
        #             overlay_mask = v2.Resize(
        #                 (target_h, target_w), interpolation=v2.InterpolationMode.NEAREST
        #             )(overlay_mask.unsqueeze(0)).squeeze(0)
        #
        #         result["mouth_overlay_info"] = (overlay, overlay_mask)
        #         if control.get("CommandLineDebugEnableToggle", False):
        #             print("[INFO] Mouth Align: Applied Enhanced Swapped Mouth.")
        # ---------- 1.5 XSEG MOUTH PROTECTION (NEW) ----------
        if need_xseg_mouth_protection and labels_swap is not None:
            # Class 11: Inner Mouth, 12: Upper Lip, 13: Lower Lip
            # We include lips to ensure XSeg doesn't cut into the lip boundaries.
            m = self._mask_from_labels_lut(labels_swap, [11, 12, 13])
            m = self._dilate_binary(m, 2, mode="conv")
            result["inner_mouth_protection"] = (
                resize_to_target(m.unsqueeze(0)).clamp(0, 1).squeeze()
            )

        # Debug mouth-region mask (inner mouth + lips), used only for preview contour.
        if labels_swap is not None:
            mouth_debug = self._mask_from_labels_lut(labels_swap, [11, 12, 13])
            result["mouth_debug"] = (
                resize_to_target(mouth_debug.unsqueeze(0)).clamp(0, 1).squeeze()
            )

        # ---------- 2. MOUTH MASK (Grouped Optimization) ----------
        if need_parser_mouth:
            mouth = torch.zeros((512, 512), device=device, dtype=torch.float32)
            mouth_groups: dict = {}
            mouth_specs = {
                11: "XsegMouthParserSlider",
                12: "XsegUpperLipParserSlider",
                13: "XsegLowerLipParserSlider",
            }

            for cls, pname in mouth_specs.items():
                val = int(parameters.get(pname, 0))
                if val not in mouth_groups:
                    mouth_groups[val] = []
                mouth_groups[val].append(cls)

            for val, classes in mouth_groups.items():
                if val:
                    m = self._mask_from_labels_lut(labels_swap, classes)
                    m = self._dilate_binary(m, val, mode)
                    mouth = torch.maximum(mouth, m)

            result["mouth"] = resize_to_target(mouth.unsqueeze(0)).clamp(0, 1).squeeze()

        # ---------- 3. FACEPARSER MASK (Grouped Optimization) ----------
        if parameters.get("FaceParserEnableToggle", False):
            fp = torch.zeros((512, 512), device=device, dtype=torch.float32)
            fp_groups: dict = {}
            face_classes = {
                1: "FaceParserSlider",
                2: "LeftEyebrowParserSlider",
                3: "RightEyebrowParserSlider",
                4: "LeftEyeParserSlider",
                5: "RightEyeParserSlider",
                6: "EyeGlassesParserSlider",
                10: "NoseParserSlider",
                11: "MouthParserSlider",
                12: "UpperLipParserSlider",
                13: "LowerLipParserSlider",
                14: "NeckParserSlider",
                17: "HairParserSlider",
            }
            mouth_inside = parameters.get("MouthParserInsideToggle", False)
            upper_teeth_keep_mask = None
            if (
                parameters.get("AutoMouthUpperTeethExcludeActive", False)
                and labels_swap is not None
            ):
                upper_teeth_keep_mask = self._build_upper_teeth_keep_mask(
                    labels_swap, swap_restorecalc
                )

            if upper_teeth_keep_mask is not None:
                result["mouth_debug_teeth"] = (
                    resize_to_target(upper_teeth_keep_mask.unsqueeze(0))
                    .clamp(0, 1)
                    .squeeze()
                )

            for cls, pname in face_classes.items():
                val = int(parameters.get(pname, 0))
                if val == 0:
                    continue
                is_min = mouth_inside and cls == 11
                key = (val, is_min)
                if key not in fp_groups:
                    fp_groups[key] = []
                fp_groups[key].append(cls)

            for (val, is_min), classes in fp_groups.items():
                m1 = self._mask_from_labels_lut(labels_swap, classes)
                m1 = self._dilate_binary(m1, val, mode)

                if labels_orig is not None:
                    m2 = self._mask_from_labels_lut(labels_orig, classes)
                    if is_min:
                        comb = torch.minimum(m1, m2)
                    else:
                        comb = torch.maximum(m1, m2)
                else:
                    comb = m1

                if upper_teeth_keep_mask is not None and 11 in classes:
                    comb = comb * (1.0 - upper_teeth_keep_mask)

                fp = torch.maximum(fp, comb)

            # Enforce keep mask after all classes so dilated lips cannot
            # reclaim the teeth area.
            if upper_teeth_keep_mask is not None:
                fp = fp * (1.0 - upper_teeth_keep_mask)

            # Recompute mouth-only dilated mask to reflect actual slider values in debug outline.
            _fp_mouth = torch.zeros((512, 512), device=device, dtype=torch.float32)
            for _cls, _pname in [
                (11, "MouthParserSlider"),
                (12, "UpperLipParserSlider"),
                (13, "LowerLipParserSlider"),
            ]:
                _val = int(parameters.get(_pname, 0))
                if _val > 0:
                    _m = self._mask_from_labels_lut(labels_swap, [_cls])
                    _m = self._dilate_binary(_m, _val, mode)
                    _fp_mouth = torch.maximum(_fp_mouth, _m)
            if _fp_mouth.sum() > 0:
                result["mouth_debug"] = (
                    resize_to_target(_fp_mouth.unsqueeze(0)).clamp(0, 1).squeeze()
                )

            if parameters.get("FaceBlurParserSlider", 0) > 0:
                b = parameters["FaceBlurParserSlider"]
                gauss = transforms.GaussianBlur(b * 2 + 1, (b + 1) * 0.2)
                fp = gauss(fp.unsqueeze(0).unsqueeze(0)).squeeze()

            mask_high_res = (1.0 - fp).unsqueeze(0)
            mask_final = resize_to_target(mask_high_res)

            if parameters.get("FaceParserBlendSlider", 0) > 0:
                mask_final = (
                    mask_final + parameters["FaceParserBlendSlider"] / 100.0
                ).clamp(0, 1)
            result["FaceParser_mask"] = mask_final

        # ---------- 4. TEXTURE / DIFFERENCING EXCLUDE ----------
        if (
            parameters.get("TransferTextureEnableToggle", False)
            or parameters.get("DifferencingEnableToggle", False)
        ) and parameters.get("ExcludeMaskEnableToggle", False):
            tex = torch.zeros((512, 512), device=device, dtype=torch.float32)
            tex_o = torch.zeros((512, 512), device=device, dtype=torch.float32)

            tex_specs = {
                1: "FaceParserTextureSlider",
                2: "EyebrowParserTextureSlider",
                3: "EyebrowParserTextureSlider",
                4: "EyeParserTextureSlider",
                5: "EyeParserTextureSlider",
                10: "NoseParserTextureSlider",
                11: "MouthParserTextureSlider",
                12: "MouthParserTextureSlider",
                13: "MouthParserTextureSlider",
                14: "NeckParserTextureSlider",
            }

            face_val = int(parameters.get(tex_specs[1], 0))
            if face_val > 0:
                blend = parameters.get("FaceParserTextureSlider", 0) / 10.0
                m_s = self._mask_from_labels_lut(labels_swap, [1]) * blend
                tex = torch.maximum(tex, m_s)
                if labels_orig is not None:
                    m_o = self._mask_from_labels_lut(labels_orig, [1]) * blend
                    tex_o = torch.maximum(tex_o, m_o)

            tex_groups: dict = {}
            for cls, pname in tex_specs.items():
                if cls == 1:
                    continue
                val = int(parameters.get(pname, 0))
                if val == 0:
                    continue
                if val not in tex_groups:
                    tex_groups[val] = []
                tex_groups[val].append(cls)

            for d, classes in tex_groups.items():
                m_s = self._mask_from_labels_lut(labels_swap, classes)
                m_o = (
                    self._mask_from_labels_lut(labels_orig, classes)
                    if labels_orig is not None
                    else torch.zeros_like(m_s)
                )

                if d > 0:
                    m_s = self._dilate_binary(m_s, d, mode)
                    m_o = self._dilate_binary(m_o, d, mode)
                    if parameters.get("FaceParserBlendTextureSlider", 0):
                        bl = parameters["FaceParserBlendTextureSlider"] / 100.0
                        m_s = (m_s + bl).clamp(0, 1)
                        m_o = (m_o + bl).clamp(0, 1)
                    tex = torch.maximum(tex, m_s)
                    tex_o = torch.maximum(tex_o, m_o)

                elif d < 0:
                    d_abs = abs(d)
                    m_s = self._dilate_binary(m_s, -d_abs, mode)
                    m_o = self._dilate_binary(m_o, -d_abs, mode)
                    if parameters.get("FaceParserBlendTextureSlider", 0):
                        bl = parameters["FaceParserBlendTextureSlider"] / 100.0
                        m_s = (m_s + bl).clamp(0, 1)
                        m_o = (m_o + bl).clamp(0, 1)
                    sub = torch.maximum(m_s, m_o)
                    tex = (tex - sub).clamp_min(0)
                    tex_o = (tex_o - sub).clamp_min(0)

            comb = torch.minimum(1.0 - tex.clamp(0, 1), 1.0 - tex_o.clamp(0, 1))
            result["texture_mask"] = comb.unsqueeze(0).clamp(0, 1)

        return result

    # --- Morphological Helpers ---

    def _get_circle_kernel(self, r: int, device: str) -> torch.Tensor:
        key = (int(r), str(device))

        # Thread-safe read
        with self._morph_cache_lock:
            k = self._morph_kernels.get(key)
            if k is not None:
                return k

        rr = int(r)

        ys, xs = torch.meshgrid(
            torch.arange(-rr, rr + 1, device=device),
            torch.arange(-rr, rr + 1, device=device),
            indexing="ij",
        )
        kernel = ((xs * xs + ys * ys) <= rr * rr).float().unsqueeze(0).unsqueeze(0)

        # Thread-safe write
        with self._morph_cache_lock:
            self._morph_kernels[key] = kernel

        return kernel

    def _dilate_binary(
        self, m: torch.Tensor, r: int, mode: str = "conv"
    ) -> torch.Tensor:
        """Applies dilation (r > 0) or erosion (r < 0) to a binary mask."""
        if r == 0:
            return m
        squeeze_back = False
        if m.dim() == 2:
            m_in = m.unsqueeze(0).unsqueeze(0)
            squeeze_back = True
        elif m.dim() == 4:
            m_in = m
        else:
            raise ValueError(f"_dilate_binary: unsupported shape {m.shape}")

        rr = abs(int(r))

        if mode == "pool":
            out = F.max_pool2d(m_in, kernel_size=2 * rr + 1, stride=1, padding=rr)
            out = (out > 0).float()
        elif mode == "iter_pool":
            out = m_in
            for _ in range(rr):
                out = F.max_pool2d(out, kernel_size=3, stride=1, padding=1)
            out = (out > 0).float()
        else:
            kernel = self._get_circle_kernel(rr, m_in.device)
            hits = F.conv2d(m_in, kernel, padding=rr)
            out = (hits > 0).float()

        return out.squeeze(0).squeeze(0) if squeeze_back else out

    def _mask_from_labels_lut(
        self, labels: torch.Tensor, classes: list[int]
    ) -> torch.Tensor:
        """Fast binary mask generation from labels using a Lookup Table."""
        lut = torch.zeros(19, device=labels.device, dtype=torch.float32)
        if classes:
            lut[torch.tensor(classes, device=labels.device, dtype=torch.long)] = 1.0
        # FM-08: clamp labels to valid LUT range to prevent out-of-bounds indexing
        labels_safe = labels.clamp(0, 18)
        return lut[labels_safe]

    def _build_upper_teeth_keep_mask(
        self,
        labels_swap: torch.Tensor,
        swap_restorecalc: torch.Tensor,
    ) -> torch.Tensor | None:
        """Builds a soft keep-mask for upper teeth based on brightness."""
        inner_swap = (labels_swap == 11).float()
        if inner_swap.sum() < 16:
            return None

        y_idx, _ = torch.where(inner_swap > 0.5)
        if y_idx.numel() == 0:
            return None

        y_min = int(y_idx.min().item())
        y_max = int(y_idx.max().item())
        mouth_h = max(1, y_max - y_min + 1)

        H = labels_swap.shape[0]
        device = labels_swap.device
        y_grid = torch.arange(H, device=device).view(-1, 1)

        # Detect bright teeth core in top 35 % of inner-mouth.
        top_limit = y_min + int(max(2, mouth_h * 0.35))
        top_band = (y_grid <= top_limit).expand(H, labels_swap.shape[1])
        bright_map = swap_restorecalc.float().div(255.0).mean(dim=0)
        keep_core = ((inner_swap > 0.5) & top_band & (bright_map >= 0.50)).float()
        if keep_core.sum() == 0:
            return None

        # Find the widest row in keep_core; hard-fill inner_swap upward from there
        # to close the dark gap between teeth surface and the cavity boundary.
        widest_row = int(keep_core.sum(dim=1).argmax().item())
        gap_fill = (
            (inner_swap > 0.5) & (y_grid < widest_row).expand_as(inner_swap)
        ).float()
        keep = torch.maximum(keep_core, gap_fill)

        # Dilate vertically ±2 rows to cover dark edge pixels at the bottom of
        # the teeth, then clip to inner_swap so we never bleed into tongue rows.
        keep = (
            F.max_pool2d(
                keep.unsqueeze(0).unsqueeze(0),
                kernel_size=(5, 1),
                stride=1,
                padding=(2, 0),
            )
            .squeeze(0)
            .squeeze(0)
        )
        keep = torch.minimum(keep, inner_swap)

        return keep

    # --- Occluder & XSeg ---

    def apply_occlusion(self, img, amount, parameters=None, original_face_512=None):
        """
        Runs the Occluder model to mask out obstacles (hands, microphones, etc.).
        Includes logic to protect the inner mouth (tongue/teeth) from being occluded.
        """
        img = torch.div(img, 255)
        img = torch.unsqueeze(img, 0).contiguous()

        # Output initialisation
        outpred = torch.ones(
            (256, 256), dtype=torch.float32, device=self.models_processor.device
        ).contiguous()

        self.run_occluder(img, outpred)

        outpred = torch.squeeze(outpred)
        # Binarize: True(1) = Face, False(0) = Occlusion
        outpred = outpred > 0
        outpred = torch.unsqueeze(outpred, 0).type(torch.float32)

        # --- TONGUE PRIORITY LOGIC ---
        # Ensures that objects inside the mouth (tongue, smoke) are not masked out
        # if the user enabled "OccluderTonguePriority".
        protected_mouth_region = None

        if (
            parameters is not None
            and parameters.get("OccluderTonguePriorityToggle", False)
            and original_face_512 is not None
        ):
            # 1. Get FaceParser labels for original face
            labels = self._faceparser_labels(original_face_512)

            # 2. Extract Inner Mouth (Class 11)
            mouth_mask = self._mask_from_labels_lut(labels, [11])

            # 3. Resize to 256x256 to match occluder
            mouth_mask_input = mouth_mask.unsqueeze(0).unsqueeze(0)
            mouth_mask_256 = F.interpolate(
                mouth_mask_input, size=(256, 256), mode="nearest"
            ).squeeze(0)

            # 4. Identify "Obstacle in Mouth"
            protected_mouth_region = (outpred < 0.5) * (mouth_mask_256 > 0.5)

        # Standard Morphology (Size Slider)
        if amount > 0:
            with self._kernel_cache_lock:
                if "3x3" not in self._kernel_cache:
                    self._kernel_cache["3x3"] = torch.ones(
                        (1, 1, 3, 3),
                        dtype=torch.float32,
                        device=self.models_processor.device,
                    )
                kernel = self._kernel_cache["3x3"]

            for _ in range(int(amount)):
                outpred = torch.nn.functional.conv2d(outpred, kernel, padding=(1, 1))
                outpred = torch.clamp(outpred, 0, 1)

            outpred = torch.squeeze(outpred)

        if amount < 0:
            outpred = torch.neg(outpred)
            outpred = torch.add(outpred, 1)
            kernel = torch.ones(
                (1, 1, 3, 3), dtype=torch.float32, device=self.models_processor.device
            )

            for _ in range(int(-amount)):
                outpred = torch.nn.functional.conv2d(outpred, kernel, padding=(1, 1))
                outpred = torch.clamp(outpred, 0, 1)

            outpred = torch.squeeze(outpred)
            outpred = torch.neg(outpred)
            outpred = torch.add(outpred, 1)

        outpred = torch.reshape(outpred, (1, 256, 256))

        # --- RESTORE PROTECTED REGION ---
        if protected_mouth_region is not None:
            # Explicit float cast to avoid boolean subtraction error
            outpred = outpred * (1.0 - protected_mouth_region.float())

        return outpred

    def run_occluder(self, image, output):
        model_name = "Occluder"
        ort_session = self.models_processor.models.get(model_name)

        if not ort_session:
            ort_session = self.models_processor.load_model(model_name)
            if ort_session:
                self.active_models.add(model_name)

        # FM-01: guard against None model
        if not ort_session:
            return

        io_binding = ort_session.io_binding()
        io_binding.bind_input(
            name="img",
            device_type=self.models_processor.device_type,
            device_id=self.models_processor.binding_device_id,
            element_type=np.float32,
            shape=(1, 3, 256, 256),
            buffer_ptr=image.data_ptr(),
        )
        io_binding.bind_output(
            name="output",
            device_type=self.models_processor.device_type,
            device_id=self.models_processor.binding_device_id,
            element_type=np.float32,
            shape=(1, 1, 256, 256),
            buffer_ptr=output.data_ptr(),
        )

        is_lazy_build = self.models_processor.check_and_clear_pending_build(model_name)
        if is_lazy_build:
            self.models_processor.show_build_dialog.emit(
                "Finalizing TensorRT Build",
                f"Performing first-run inference for:\n{model_name}\n\nThis may take several minutes.",
            )

        try:
            self.function_worker.run_ort_with_iobinding(ort_session, io_binding)
        finally:
            if is_lazy_build:
                self.models_processor.hide_build_dialog.emit()

    def apply_dfl_xseg(
        self,
        img: torch.Tensor,
        amount: float,
        mouth: torch.Tensor,
        parameters: dict,
        inner_mouth_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        import cv2

        # FM-07: use .get() for all parameter accesses to avoid KeyError
        amount2 = -parameters.get("DFLXSeg2SizeSlider", 0)
        amount_calc = -parameters.get("BackgroundParserTextureSlider", 0)

        # Check if obstacle protection is enabled
        exclude_obstacles = parameters.get("XSegExcludeInnerObstaclesToggle", False)

        img = img.type(torch.float32)
        img = torch.div(img, 255)
        img = torch.unsqueeze(img, 0).contiguous()
        outpred = torch.ones(
            (256, 256), dtype=torch.float32, device=self.models_processor.device
        ).contiguous()

        self.run_dfl_xseg(img, outpred)

        outpred = torch.clamp(outpred, min=0.0, max=1.0)
        outpred[outpred < 0.1] = 0
        outpred_calc = outpred.clone()

        # Invert: Face becomes 0, Background/Obstacles become 1
        outpred = 1.0 - outpred
        outpred = torch.unsqueeze(outpred, 0).type(torch.float32)

        if amount2 != amount:
            outpred2 = outpred.clone()

        # --- HPC OPTIMIZATION: Hybrid Smart Face Expansion ---
        needs_smart_expand_1 = (amount < 0) and exclude_obstacles
        needs_smart_expand_2 = (amount2 != amount and amount2 < 0) and exclude_obstacles

        true_obstacles = None
        if needs_smart_expand_1 or needs_smart_expand_2:
            # Re-invert temporarily to find the solid face contour (Face=1)
            face_mask = 1.0 - outpred

            # HPC OPTIMIZATION: Cast to 8-bit integer BEFORE transferring over PCIe to the CPU.
            # This reduces the VRAM-to-RAM transfer payload by 75%.
            mask_np_uint8 = (face_mask.squeeze() * 255).to(torch.uint8).cpu().numpy()

            contours, _ = cv2.findContours(
                mask_np_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            hull_mask_np = mask_np_uint8.copy()
            if len(contours) > 0:
                largest_contour = max(contours, key=cv2.contourArea)
                hull = cv2.convexHull(largest_contour)
                cv2.drawContours(hull_mask_np, [hull], -1, 255, thickness=cv2.FILLED)

            # Obstacles are the gaps exactly inside the face hull
            raw_obstacles_np = cv2.bitwise_and(
                hull_mask_np, cv2.bitwise_not(mask_np_uint8)
            )

            # Use morphological opening to remove thin boundary slivers (1-3px)
            kernel_morph = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            true_obstacles_np = cv2.morphologyEx(
                raw_obstacles_np, cv2.MORPH_OPEN, kernel_morph
            )

            # HPC OPTIMIZATION: Use non_blocking=True for the return transfer to allow
            # the CUDA stream to continue queuing operations if possible.
            true_obstacles = (
                torch.from_numpy(true_obstacles_np).to(
                    device=outpred.device, dtype=torch.float32, non_blocking=True
                )
                / 255.0
            )
            true_obstacles = true_obstacles.unsqueeze(0)

        # ---------- AMOUNT PROCESSING ----------
        if amount > 0:
            # Slider is Negative -> Shrink Face (Expand Background=1)
            r = int(amount)
            k = 2 * r + 1
            outpred = F.max_pool2d(outpred, kernel_size=k, stride=1, padding=r)
            outpred = outpred.clamp(0.0, 1.0)

        elif amount < 0:
            # Slider is Positive -> Expand Face
            r = int(-amount)
            k = 2 * r + 1

            if exclude_obstacles and true_obstacles is not None:
                # Shield only grows at 50% of the face expansion rate.
                shield_r = max(1, r // 2)
                shield_k = 2 * shield_r + 1

                face_mask = 1.0 - outpred
                dilated_face = F.max_pool2d(
                    face_mask, kernel_size=k, stride=1, padding=r
                )
                shield = F.max_pool2d(
                    true_obstacles, kernel_size=shield_k, stride=1, padding=shield_r
                )

                expanded_safe = torch.clamp(dilated_face - shield, 0.0, 1.0)
                final_face = torch.maximum(face_mask, expanded_safe)

                outpred = 1.0 - final_face
                outpred = outpred.clamp(0.0, 1.0)
            else:
                outpred = 1.0 - outpred
                outpred = F.max_pool2d(outpred, kernel_size=k, stride=1, padding=r)
                outpred = 1.0 - outpred
                outpred = outpred.clamp(0.0, 1.0)

        # Derive calculation masks AFTER size processing, but BEFORE blur.
        # This ensures color stats never sample excluded background pixels.
        outpred_calc = torch.where(outpred < 0.5, 0.0, 1.0).float()
        outpred_calc_dill = outpred_calc.clone()
        blur_amount = parameters.get("OccluderXSegBlurSlider", 0)
        if blur_amount > 0:
            k_size = blur_amount * 2 + 1
            sigma = (blur_amount + 1) * 0.2
            outpred = v2.functional.gaussian_blur(
                outpred, [k_size, k_size], [sigma, sigma]
            )

        outpred_noFP = outpred.clone()

        # ---------- AMOUNT2 PROCESSING ----------
        if amount2 != amount:
            if amount2 > 0:
                # Slider2 is Negative -> Shrink Face
                r2 = int(amount2)
                k2 = 2 * r2 + 1
                outpred2 = F.max_pool2d(outpred2, kernel_size=k2, stride=1, padding=r2)
                outpred2 = outpred2.clamp(0.0, 1.0)

            elif amount2 < 0:
                # Slider2 is Positive -> Expand Face
                r2 = int(-amount2)
                k2 = 2 * r2 + 1

                if exclude_obstacles and true_obstacles is not None:
                    shield_r2 = max(1, r2 // 2)
                    shield_k2 = 2 * shield_r2 + 1

                    face_mask2 = 1.0 - outpred2
                    dilated_face2 = F.max_pool2d(
                        face_mask2, kernel_size=k2, stride=1, padding=r2
                    )
                    shield2 = F.max_pool2d(
                        true_obstacles,
                        kernel_size=shield_k2,
                        stride=1,
                        padding=shield_r2,
                    )

                    expanded_safe2 = torch.clamp(dilated_face2 - shield2, 0.0, 1.0)
                    final_face2 = torch.maximum(face_mask2, expanded_safe2)

                    outpred2 = 1.0 - final_face2
                    outpred2 = outpred2.clamp(0.0, 1.0)
                else:
                    outpred2 = 1.0 - outpred2
                    outpred2 = F.max_pool2d(
                        outpred2, kernel_size=k2, stride=1, padding=r2
                    )
                    outpred2 = 1.0 - outpred2
                    outpred2 = outpred2.clamp(0.0, 1.0)

            blur_amount2 = parameters.get("XSeg2BlurSlider", 0)
            if blur_amount2 > 0:
                # OPTIMIZED: Zero-overhead functional blur
                k_size2 = blur_amount2 * 2 + 1
                sigma2 = (blur_amount2 + 1) * 0.2
                outpred2 = v2.functional.gaussian_blur(
                    outpred2, [k_size2, k_size2], [sigma2, sigma2]
                )

            outpred[mouth > 0.01] = outpred2[mouth > 0.01]

        outpred = torch.reshape(outpred, (1, 256, 256))

        if inner_mouth_mask is not None:
            # Mouth mask protection
            outpred = outpred * (1.0 - inner_mouth_mask)
            if outpred_noFP.dim() == 2:
                outpred_noFP = outpred_noFP.unsqueeze(0)
            outpred_noFP = outpred_noFP * (1.0 - inner_mouth_mask)

        # Background exclusion
        if parameters.get("BgExcludeEnableToggle", False) and amount_calc != 0:
            if amount_calc > 0:
                r2 = int(amount_calc)
                k2 = 2 * r2 + 1
                outpred_calc_dill = F.max_pool2d(
                    outpred_calc_dill, kernel_size=k2, stride=1, padding=r2
                )
                outpred_calc_dill = outpred_calc_dill.clamp(0.0, 1.0)
                bg_blur = parameters.get("BGExcludeBlurAmountSlider", 0)
                if bg_blur > 0:
                    k_bg = bg_blur * 2 + 1
                    s_bg = (bg_blur + 1) * 0.2
                    outpred_calc_dill = v2.functional.gaussian_blur(
                        outpred_calc_dill.type(torch.float32),
                        [k_bg, k_bg],
                        [s_bg, s_bg],
                    )
                outpred_calc_dill = outpred_calc_dill.clamp(0.0, 1.0)
            elif amount_calc < 0:
                r2 = int(-amount_calc)
                k2 = 2 * r2 + 1
                outpred_calc_dill = 1.0 - outpred_calc_dill
                outpred_calc_dill = F.max_pool2d(
                    outpred_calc_dill, kernel_size=k2, stride=1, padding=r2
                )
                outpred_calc_dill = 1.0 - outpred_calc_dill
                bg_blur = parameters.get("BGExcludeBlurAmountSlider", 0)
                if bg_blur > 0:
                    orig = outpred_calc_dill.clone()
                    k_bg = bg_blur * 2 + 1
                    s_bg = (bg_blur + 1) * 0.2
                    outpred_calc_dill = v2.functional.gaussian_blur(
                        outpred_calc_dill.type(torch.float32),
                        [k_bg, k_bg],
                        [s_bg, s_bg],
                    )
                    outpred_calc_dill = torch.max(outpred_calc_dill, orig)
                outpred_calc_dill = outpred_calc_dill.clamp(0.0, 1.0)

        return outpred, outpred_calc, outpred_calc_dill, outpred_noFP

    def run_dfl_xseg(self, image, output):
        model_name = "XSeg"
        ort_session = self.models_processor.models.get(model_name)
        if not ort_session:
            ort_session = self.models_processor.load_model(model_name)

        if not ort_session:
            return

        io_binding = ort_session.io_binding()
        io_binding.bind_input(
            name="in_face:0",
            device_type=self.models_processor.device_type,
            device_id=self.models_processor.binding_device_id,
            element_type=np.float32,
            shape=image.size(),
            buffer_ptr=image.data_ptr(),
        )
        io_binding.bind_output(
            name="out_mask:0",
            device_type=self.models_processor.device_type,
            device_id=self.models_processor.binding_device_id,
            element_type=np.float32,
            shape=(1, 1, 256, 256),
            buffer_ptr=output.data_ptr(),
        )

        is_lazy_build = self.models_processor.check_and_clear_pending_build(model_name)
        if is_lazy_build:
            self.models_processor.show_build_dialog.emit(
                "Finalizing TensorRT Build",
                f"Performing first-run inference for:\n{model_name}\n\nThis may take several minutes.",
            )

        try:
            self.function_worker.run_ort_with_iobinding(ort_session, io_binding)
        finally:
            if is_lazy_build:
                self.models_processor.hide_build_dialog.emit()

    def run_onnx(
        self, image_tensor: torch.Tensor, output_tensor: torch.Tensor, model_key: str
    ) -> torch.Tensor:
        sess = self.models_processor.models.get(model_key)
        if sess is None:
            sess = self.models_processor.load_model(model_key)

        # FM-02: guard against None model
        if sess is None:
            return output_tensor

        image_tensor = image_tensor.contiguous()
        io_binding = sess.io_binding()

        # FM-02: use dynamic node names instead of hardcoded "input" / "features"
        input_name = sess.get_inputs()[0].name
        output_name = sess.get_outputs()[0].name

        io_binding.bind_input(
            name=input_name,
            device_type=self.models_processor.device_type,
            device_id=self.models_processor.binding_device_id,
            element_type=np.float32,
            shape=image_tensor.shape,
            buffer_ptr=image_tensor.data_ptr(),
        )
        io_binding.bind_output(
            name=output_name,
            device_type=self.models_processor.device_type,
            device_id=self.models_processor.binding_device_id,
            element_type=np.float32,
            shape=output_tensor.shape,
            buffer_ptr=output_tensor.data_ptr(),
        )

        is_lazy_build = self.models_processor.check_and_clear_pending_build(model_key)
        if is_lazy_build:
            self.models_processor.show_build_dialog.emit(
                "Finalizing TensorRT Build",
                f"Performing first-run inference for:\n{model_key}\n\nThis may take several minutes.",
            )

        try:
            self.function_worker.run_ort_with_iobinding(sess, io_binding)
        finally:
            if is_lazy_build:
                self.models_processor.hide_build_dialog.emit()

        return output_tensor

    def run_CLIPs(self, img, CLIPText, CLIPAmount):
        device = img.device
        # --- OPTIMIZATION: Double-Checked Locking Pattern ---
        # Ensures the model is only loaded ONCE in VRAM, preventing Memory Leaks.
        # The first 'if' avoids lock overhead for 99% of frames.
        if not self.models_processor.clip_session:
            with self._clip_load_lock:
                # Check again inside the lock in case another thread already loaded it
                if not self.models_processor.clip_session:
                    self.models_processor.clip_session = CLIPDensePredT(
                        version="ViT-B/16", reduce_dim=64, complex_trans_conv=True
                    )
                    self.models_processor.clip_session.eval()
                    self.models_processor.clip_session.load_state_dict(
                        torch.load(
                            f"{models_dir}/rd64-uni-refined.pth", weights_only=True
                        ),
                        strict=False,
                    )
                    self.models_processor.clip_session.to(device)

        clip_mask = torch.ones((352, 352), device=device)

        # OPTIMIZED: Direct functional tensor operations (no Compose CPU overhead)
        img_float = img.float() / 255.0
        CLIPimg = v2.functional.resize(img_float, [352, 352], antialias=True)
        CLIPimg = v2.functional.normalize(
            CLIPimg, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        CLIPimg = CLIPimg.unsqueeze(0).contiguous().to(device)

        if CLIPText != "":
            prompts = CLIPText.split(",")
            with torch.no_grad():
                preds = self.models_processor.clip_session(
                    CLIPimg.repeat(len(prompts), 1, 1, 1), prompts
                )[0]

            clip_mask = 1 - torch.sigmoid(preds[0][0])
            for i in range(len(prompts) - 1):
                clip_mask *= 1 - torch.sigmoid(preds[i + 1][0])

            thresh = CLIPAmount / 100.0
            clip_mask = (clip_mask > thresh).float()

        return clip_mask.unsqueeze(0)

    # --- Restoration Helpers (Eyes/Mouth) ---

    def soft_oval_mask(
        self,
        height,
        width,
        center,
        radius_x,
        radius_y,
        feather_radius=None,
        device=None,
    ):
        if feather_radius is None:
            feather_radius = max(radius_x, radius_y) // 2

        # FM-09: include device in the cache key and create tensors on the correct device
        _device = device if device is not None else self.models_processor.device
        cache_key = (height, width, str(_device))

        # Thread-safe read and write
        with self._meshgrid_cache_lock:
            if cache_key in self._meshgrid_cache:
                y, x = self._meshgrid_cache[cache_key]
            else:
                y, x = torch.meshgrid(
                    torch.arange(height, device=_device),
                    torch.arange(width, device=_device),
                    indexing="ij",
                )
                self._meshgrid_cache[cache_key] = (y, x)

        normalized_distance = torch.sqrt(
            ((x - center[0]) / radius_x) ** 2 + ((y - center[1]) / radius_y) ** 2
        )
        mask = torch.clamp(
            (1 - normalized_distance) * (radius_x / feather_radius), 0, 1
        )
        return mask

    def restore_mouth(
        self,
        img_orig,
        img_swap,
        kpss_orig,
        blend_alpha=0.5,
        feather_radius=10,
        size_factor=0.5,
        radius_factor_x=1.0,
        radius_factor_y=1.0,
        x_offset=0,
        y_offset=0,
    ):
        left_mouth = np.array([int(val) for val in kpss_orig[3]])
        right_mouth = np.array([int(val) for val in kpss_orig[4]])

        mouth_center = (left_mouth + right_mouth) // 2
        mouth_base_radius = int(np.linalg.norm(left_mouth - right_mouth) * size_factor)

        radius_x = int(mouth_base_radius * radius_factor_x)
        radius_y = int(mouth_base_radius * radius_factor_y)

        mouth_center[0] += x_offset
        mouth_center[1] += y_offset

        ymin = max(0, mouth_center[1] - radius_y)
        ymax = min(img_orig.size(1), mouth_center[1] + radius_y)
        xmin = max(0, mouth_center[0] - radius_x)
        xmax = min(img_orig.size(2), mouth_center[0] + radius_x)

        mouth_region_orig = img_orig[:, ymin:ymax, xmin:xmax]
        mouth_mask = self.soft_oval_mask(
            ymax - ymin,
            xmax - xmin,
            (radius_x, radius_y),
            radius_x,
            radius_y,
            feather_radius,
        ).to(img_orig.device)

        target_ymin = ymin
        target_ymax = ymin + mouth_region_orig.size(1)
        target_xmin = xmin
        target_xmax = xmin + mouth_region_orig.size(2)

        img_swap_mouth = img_swap[:, target_ymin:target_ymax, target_xmin:target_xmax]
        blended_mouth = (
            blend_alpha * img_swap_mouth + (1 - blend_alpha) * mouth_region_orig
        )

        img_swap[:, target_ymin:target_ymax, target_xmin:target_xmax] = (
            mouth_mask * blended_mouth + (1 - mouth_mask) * img_swap_mouth
        )
        return img_swap

    def restore_eyes(
        self,
        img_orig,
        img_swap,
        kpss_orig,
        blend_alpha=0.5,
        feather_radius=10,
        size_factor=3.5,
        radius_factor_x=1.0,
        radius_factor_y=1.0,
        x_offset=0,
        y_offset=0,
        eye_spacing_offset=0,
    ):
        left_eye = np.array([int(val) for val in kpss_orig[0]])
        right_eye = np.array([int(val) for val in kpss_orig[1]])

        left_eye[0] += x_offset
        right_eye[0] += x_offset
        left_eye[1] += y_offset
        right_eye[1] += y_offset

        eye_distance = np.linalg.norm(left_eye - right_eye)
        base_eye_radius = int(eye_distance / size_factor)

        radius_x = int(base_eye_radius * radius_factor_x)
        radius_y = int(base_eye_radius * radius_factor_y)

        left_eye[0] += eye_spacing_offset
        right_eye[0] -= eye_spacing_offset

        def extract_and_blend_eye(
            eye_center,
            radius_x,
            radius_y,
            img_orig,
            img_swap,
            blend_alpha,
            feather_radius,
        ):
            ymin = max(0, eye_center[1] - radius_y)
            ymax = min(img_orig.size(1), eye_center[1] + radius_y)
            xmin = max(0, eye_center[0] - radius_x)
            xmax = min(img_orig.size(2), eye_center[0] + radius_x)

            eye_region_orig = img_orig[:, ymin:ymax, xmin:xmax]
            eye_mask = self.soft_oval_mask(
                ymax - ymin,
                xmax - xmin,
                (radius_x, radius_y),
                radius_x,
                radius_y,
                feather_radius,
            ).to(img_orig.device)

            target_ymin = ymin
            target_ymax = ymin + eye_region_orig.size(1)
            target_xmin = xmin
            target_xmax = xmin + eye_region_orig.size(2)

            img_swap_eye = img_swap[:, target_ymin:target_ymax, target_xmin:target_xmax]
            blended_eye = (
                blend_alpha * img_swap_eye + (1 - blend_alpha) * eye_region_orig
            )

            img_swap[:, target_ymin:target_ymax, target_xmin:target_xmax] = (
                eye_mask * blended_eye + (1 - eye_mask) * img_swap_eye
            )

        extract_and_blend_eye(
            left_eye,
            radius_x,
            radius_y,
            img_orig,
            img_swap,
            blend_alpha,
            feather_radius,
        )
        extract_and_blend_eye(
            right_eye,
            radius_x,
            radius_y,
            img_orig,
            img_swap,
            blend_alpha,
            feather_radius,
        )

        return img_swap

    # --- Difference & Perceptual Loss ---

    def apply_vgg_mask_simple(
        self,
        swapped_face: torch.Tensor,  # [3,512,512] uint8
        original_face: torch.Tensor,  # [3,512,512] uint8
        swap_mask_128: torch.Tensor,  # [1,128,128] float mask (0..1)
        center_pct: float,  # 0..100, e.g. parameters['VGGMaskThresholdSlider']
        softness_pct: float,  # 0..100, e.g. parameters['VGGMaskSoftnessSlider']
        feature_layer: str = "combo_relu3_3_relu3_1",
        mode: str = "smooth",  # 'smooth' (smoothstep) or 'linear'
    ):
        """
        Returns:
          mask_vgg: [1,512,512] float 0..1 (soft difference mask)
          diff_norm_texture: [1,128,128] float 0..1 (normalized raw difference in 128 resolution)
        """
        # 1) Get raw difference via existing ONNX pipeline in 128x128 (without mapping).
        #    We use apply_perceptual_diff_onnx in pass-through mode (ExcludeVGGMaskEnableToggle=False),
        #    and ignore the complex threshold parameters (they are replaced below).
        dummy_lower = 0.0
        dummy_upper = 1.0
        dummy_upper_v = 1.0
        dummy_middle_v = 0.5

        diff_mapped_128, diff_norm_128 = self.apply_perceptual_diff_onnx(
            swapped_face,
            original_face,
            swap_mask_128,
            dummy_lower,
            0.0,
            dummy_upper,
            dummy_upper_v,
            dummy_middle_v,
            feature_layer,
            ExcludeVGGMaskEnableToggle=False,
        )
        # diff_norm_128: [1,128,128] in [0..1]
        d = diff_mapped_128.squeeze(0)  # [128,128]

        # 2) Two-slider-Mapping -> lower/upper threshold derived from (center, softness)
        center = float(center_pct) / 100.0  # 0..1
        softness = float(softness_pct) / 100.0  # 0..1

        # Width of the transition band (practical values):
        #  - Min width 0.04, Max width 0.40
        band = 0.04 + 0.36 * softness
        lo = max(0.0, center - band * 0.5)
        hi = min(1.0, center + band * 0.5)

        # 3) Curve Shape
        x = (d - lo) / max(1e-6, (hi - lo))
        x = x.clamp(0.0, 1.0)
        if mode == "smooth":
            # Smoothstep
            x = x * x * (3.0 - 2.0 * x)
        # else: 'linear' -> x remains linear

        # 4) Upscale to 512x512 (bilinear)
        x_512 = torch.nn.functional.interpolate(
            x.unsqueeze(0).unsqueeze(0),
            size=(512, 512),
            mode="bilinear",
            align_corners=True,
        ).squeeze(0)

        return x_512.clamp(0, 1), diff_norm_128

    def apply_perceptual_diff_onnx(
        self,
        swapped_face: torch.Tensor,
        original_face: torch.Tensor,
        swap_mask: torch.Tensor,
        lower_thresh: float,
        lower_value: float,
        upper_thresh: float,
        upper_value: float,
        middle_value: float,
        feature_layer: str,
        ExcludeVGGMaskEnableToggle: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates perceptual difference using VGG features (ONNX).
        Returns both the mapped mask and the raw normalized difference texture.
        """
        feature_shapes = {
            "combo_relu3_3_relu3_1": (1, 512, 128, 128),
        }

        model_key = feature_layer
        if model_key not in self.models_processor.models:
            self.models_processor.models[model_key] = self.models_processor.load_model(
                model_key
            )
            self.models_processor.active_models.add(model_key)

        def preprocess(img: torch.Tensor) -> torch.Tensor:
            img = img.clone().float() / 255.0
            mean = _VGG_MEAN.to(img.device)
            std = _VGG_STD.to(img.device)
            return ((img - mean) / std).unsqueeze(0).contiguous()

        swapped = preprocess(swapped_face)
        original = preprocess(original_face)

        shape = feature_shapes[feature_layer]
        outpred = torch.empty(shape, dtype=torch.float32, device=swapped.device)
        outpred2 = torch.empty_like(outpred)

        swapped_feat = self.run_onnx(swapped, outpred, model_key)
        original_feat = self.run_onnx(original, outpred2, model_key)

        diff_map = torch.abs(swapped_feat - original_feat).mean(dim=1)[0]
        diff_map = diff_map * swap_mask.squeeze(0)

        # OPTIMIZED: Deterministic strided slicing instead of random sampling.
        # Eliminates CUDA RNG initialization overhead and prevents temporal mask flickering.
        sample = diff_map.view(-1)[::10]
        diff_max = torch.quantile(sample, 0.99)
        diff_map = torch.clamp(diff_map, max=diff_max)

        diff_min, diff_max = diff_map.amin(), diff_map.amax()
        diff_norm = (diff_map - diff_min) / (diff_max - diff_min + 1e-6)

        diff_norm_texture = diff_norm.clone()

        if ExcludeVGGMaskEnableToggle:
            eps = 1e-6
            inv_lower = 1.0 / max(lower_thresh, eps)
            inv_mid = 1.0 / max((upper_thresh - lower_thresh), eps)
            inv_high = 1.0 / max((1.0 - upper_thresh), eps)

            res_low = lower_value + diff_norm * inv_lower * (middle_value - lower_value)
            res_mid = middle_value + (diff_norm - lower_thresh) * inv_mid * (
                upper_value - middle_value
            )
            res_high = upper_value + (diff_norm - upper_thresh) * inv_high * (
                1.0 - upper_value
            )

            result = torch.where(
                diff_norm < lower_thresh,
                res_low,
                torch.where(diff_norm > upper_thresh, res_high, res_mid),
            )
        else:
            result = diff_norm

        return result.unsqueeze(0), diff_norm_texture.unsqueeze(0)
