from typing import TYPE_CHECKING, Any, Dict, Optional

import torch
import threading
import numpy as np
from torchvision import transforms
from torchvision.transforms import v2
import torch.nn.functional as F

from app.processors.external.clipseg import CLIPDensePredT
from app.processors.models_data import models_dir

if TYPE_CHECKING:
    from app.processors.models_processor import ModelsProcessor

_VGG_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_VGG_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class FaceMasks:
    """
    Manages mask generation and manipulation for face swapping.
    Handles FaceParser, Occluder, XSeg, CLIP, and VGG-based differencing.
    """

    def __init__(self, models_processor: "ModelsProcessor"):
        self.models_processor = models_processor
        # Caches for morphological operations to avoid re-allocating tensors
        self._morph_kernels: Dict[tuple, torch.Tensor] = {}
        self._kernel_cache: Dict[str, torch.Tensor] = {}
        self._meshgrid_cache: Dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}

        # --- Thread safety locks for caches and lazy loading ---
        self._morph_cache_lock = threading.Lock()
        self._kernel_cache_lock = threading.Lock()
        self._meshgrid_cache_lock = threading.Lock()
        self._clip_load_lock = threading.Lock()
        self._custom_inference_lock = threading.Lock()
        self._runner_locks: Dict[Any, threading.Lock] = {}
        self._faceparser_init_lock = threading.Lock()
        self._xseg_init_lock = threading.Lock()
        self._occluder_init_lock = threading.Lock()
        self._vgg_init_lock = threading.Lock()

        self.clip_model_loaded = False
        self.active_models: set[str] = set()

        # Custom-kernel instances (lazy-loaded on first use)
        self._faceparser_torch: Optional[object] = None
        self._faceparser_runner: Optional[object] = None
        self._xseg_torch: Optional[object] = None
        self._xseg_runner: Optional[object] = None
        self._occluder_torch: Optional[object] = None
        self._occluder_runner: Optional[object] = None
        self._vgg_combo_torch: Optional[object] = None
        self._vgg_combo_runner: Optional[object] = None

    def _get_runner_lock(self, runner):
        with self._custom_inference_lock:
            r_id = id(runner)
            if r_id not in self._runner_locks:
                self._runner_locks[r_id] = threading.Lock()
            return self._runner_locks[r_id]

    def unload_models(self):
        """Unloads all models managed by this class via the processor."""
        with self.models_processor.model_lock:
            for model_name in list(self.active_models):
                self.models_processor.unload_model(model_name)
            self.active_models.clear()
        # Release Custom-kernel instances so they rebuild on next use.
        with self._faceparser_init_lock:
            self._faceparser_torch = None
            self._faceparser_runner = None
        with self._xseg_init_lock:
            self._xseg_torch = None
            self._xseg_runner = None
        with self._occluder_init_lock:
            self._occluder_torch = None
            self._occluder_runner = None
        with self._vgg_init_lock:
            self._vgg_combo_torch = None
            self._vgg_combo_runner = None

    # --- Inference Helpers ---

    def _get_faceparser_runner(self):
        """Lazy-load the FaceParserResnet34Torch Custom-kernel runner."""
        if self._faceparser_runner is not None:
            return self._faceparser_runner
        self.models_processor.show_build_dialog.emit(
            "Finalizing Custom Provider",
            "Compiling & capturing CUDA graph for FaceParser…\nFirst run only — future sessions load instantly from cache.",
        )
        try:
            with self._faceparser_init_lock:
                if self._faceparser_runner is not None:
                    return self._faceparser_runner
                if self._faceparser_torch is None:
                    try:
                        import pathlib
                        from custom_kernels.faceparser_resnet34.faceparser_resnet34_torch import (
                            FaceParserResnet34Torch,
                        )

                        onnx_path = str(
                            pathlib.Path(__file__).parent.parent.parent
                            / "model_assets"
                            / "faceparser_resnet34.onnx"
                        )
                        m = (
                            FaceParserResnet34Torch.from_onnx(onnx_path)
                            .to(self.models_processor.device)
                            .eval()
                        )
                        self._faceparser_torch = m
                    except Exception as e:
                        print(f"[Custom] faceparser_resnet34 load failed: {e}")
                        return None
                try:
                    from custom_kernels.faceparser_resnet34.faceparser_resnet34_torch import (
                        build_cuda_graph_runner,
                    )

                    with self.models_processor.cuda_graph_capture_lock:
                        self._faceparser_runner = build_cuda_graph_runner(
                            self._faceparser_torch, torch_compile=True
                        )
                except Exception as e:
                    print(
                        f"[Custom] faceparser_resnet34 graph runner failed, using eager: {e}"
                    )
                    self._faceparser_runner = self._faceparser_torch
        finally:
            self.models_processor.hide_build_dialog.emit()
        return self._faceparser_runner

    def _get_xseg_runner(self):
        """Lazy-load the XSegTorch Custom-kernel runner."""
        if self._xseg_runner is not None:
            return self._xseg_runner
        self.models_processor.show_build_dialog.emit(
            "Finalizing Custom Provider",
            "Compiling & capturing CUDA graph for XSeg mask model…\nFirst run only — future sessions load instantly from cache.",
        )
        try:
            with self._xseg_init_lock:
                if self._xseg_runner is not None:
                    return self._xseg_runner
                if self._xseg_torch is None:
                    try:
                        import pathlib
                        from custom_kernels.xseg.xseg_torch import XSegTorch

                        onnx_path = str(
                            pathlib.Path(__file__).parent.parent.parent
                            / "model_assets"
                            / "XSeg_model.onnx"
                        )
                        m = (
                            XSegTorch.from_onnx(onnx_path)
                            .to(self.models_processor.device)
                            .eval()
                        )
                        self._xseg_torch = m
                    except Exception as e:
                        print(f"[Custom] xseg load failed: {e}")
                        return None
                try:
                    from custom_kernels.xseg.xseg_torch import build_cuda_graph_runner

                    with self.models_processor.cuda_graph_capture_lock:
                        self._xseg_runner = build_cuda_graph_runner(
                            self._xseg_torch, torch_compile=True
                        )
                except Exception as e:
                    print(f"[Custom] xseg graph runner failed, using eager: {e}")
                    self._xseg_runner = self._xseg_torch
        finally:
            self.models_processor.hide_build_dialog.emit()
        return self._xseg_runner

    def _get_occluder_runner(self):
        """Lazy-load the OccluderTorch Custom-kernel runner."""
        if self._occluder_runner is not None:
            return self._occluder_runner
        self.models_processor.show_build_dialog.emit(
            "Finalizing Custom Provider",
            "Compiling & capturing CUDA graph for Occluder mask model…\nFirst run only — future sessions load instantly from cache.",
        )
        try:
            with self._occluder_init_lock:
                if self._occluder_runner is not None:
                    return self._occluder_runner
                if self._occluder_torch is None:
                    try:
                        import pathlib
                        from custom_kernels.occluder.occluder_torch import OccluderTorch

                        onnx_path = str(
                            pathlib.Path(__file__).parent.parent.parent
                            / "model_assets"
                            / "occluder.onnx"
                        )
                        m = (
                            OccluderTorch.from_onnx(onnx_path)
                            .to(self.models_processor.device)
                            .eval()
                        )
                        self._occluder_torch = m
                    except Exception as e:
                        print(f"[Custom] occluder load failed: {e}")
                        return None
                try:
                    from custom_kernels.occluder.occluder_torch import (
                        build_cuda_graph_runner,
                    )

                    with self.models_processor.cuda_graph_capture_lock:
                        self._occluder_runner = build_cuda_graph_runner(
                            self._occluder_torch, torch_compile=True
                        )
                except Exception as e:
                    print(f"[Custom] occluder graph runner failed, using eager: {e}")
                    self._occluder_runner = self._occluder_torch
        finally:
            self.models_processor.hide_build_dialog.emit()
        return self._occluder_runner

    def _get_vgg_combo_runner(self):
        """Lazy-load the VggComboTorch Custom-kernel runner."""
        if self._vgg_combo_runner is not None:
            return self._vgg_combo_runner
        self.models_processor.show_build_dialog.emit(
            "Finalizing Custom Provider",
            "Compiling & capturing CUDA graph for VGG face differencing model…\nFirst run only — future sessions load instantly from cache.",
        )
        try:
            with self._vgg_init_lock:
                if self._vgg_combo_runner is not None:
                    return self._vgg_combo_runner
                if self._vgg_combo_torch is None:
                    try:
                        import pathlib
                        from custom_kernels.vgg_combo.vgg_combo_torch import (
                            VggComboTorch,
                        )

                        onnx_path = str(
                            pathlib.Path(__file__).parent.parent.parent
                            / "model_assets"
                            / "vgg_combo_relu3_3_relu3_1.onnx"
                        )
                        m = (
                            VggComboTorch.from_onnx(onnx_path)
                            .to(self.models_processor.device)
                            .eval()
                        )
                        self._vgg_combo_torch = m
                    except Exception as e:
                        print(f"[Custom] vgg_combo load failed: {e}")
                        return None
                try:
                    from custom_kernels.vgg_combo.vgg_combo_torch import (
                        build_cuda_graph_runner,
                    )

                    with self.models_processor.cuda_graph_capture_lock:
                        self._vgg_combo_runner = build_cuda_graph_runner(
                            self._vgg_combo_torch, torch_compile=True
                        )
                except Exception as e:
                    print(f"[Custom] vgg_combo CUDA graph failed, using eager: {e}")
                    self._vgg_combo_runner = self._vgg_combo_torch
        finally:
            self.models_processor.hide_build_dialog.emit()
        return self._vgg_combo_runner

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

        # Custom path: bypass ORT entirely
        if self.models_processor.provider_name == "Custom":
            runner = self._get_faceparser_runner()
            if runner is not None:
                with torch.no_grad():
                    with self._get_runner_lock(runner):
                        out = runner(x)
                        labels_512 = out.argmax(dim=1).squeeze(0).to(torch.long)
                        if self.models_processor.device == "cuda":
                            torch.cuda.current_stream().synchronize()
                return labels_512
            # Custom runner unavailable — fall through to ORT

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
            self.models_processor.device,
            0,
            np.float32,
            (1, 3, 512, 512),
            x.data_ptr(),
        )
        io.bind_output(
            "output",
            self.models_processor.device,
            0,
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
            # PRE-INFERENCE SYNC: Ensure PyTorch memory is ready
            if self.models_processor.device == "cuda":
                torch.cuda.current_stream().synchronize()
            elif self.models_processor.device != "cpu":
                self.models_processor.syncvec.cpu()

            ort_session.run_with_iobinding(io)

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

        # --- Alignment Logic (Stable Center of Mass) ---
        y_s_full, x_s_full = torch.where(mouth_swap)
        y_s_inner, _ = torch.where(inner_swap)

        # Use Mean (Center of Mass) instead of min/max to avoid temporal jittering
        cx_s = x_s_full.float().mean()
        # Keep the top of the inner mouth as Y anchor (but average the top 5% to avoid 1-pixel noise)
        top_y_k = max(1, int(y_s_inner.shape[0] * 0.01))
        cy_s = torch.topk(y_s_inner.float(), k=top_y_k, largest=False).values.mean()

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
        inner_orig = labels_orig == 11
        inner_swap = labels_swap == 11

        # We strictly require the inner mouth.
        # If the mouth is closed, we cannot restore original teeth/tongue.
        if inner_orig.sum() == 0 or inner_swap.sum() == 0:
            return None, None

        mouth_orig = (labels_orig == 11) | (labels_orig == 12) | (labels_orig == 13)
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

        # --- Alignment Logic (Stable Center of Mass) ---
        y_o_full, x_o_full = torch.where(mouth_orig)
        y_s_full, x_s_full = torch.where(mouth_swap)

        w_o = (x_o_full.max() - x_o_full.min()).float()
        w_s = (x_s_full.max() - x_s_full.min()).float()

        if w_o <= 0.0 or w_s <= 0.0:
            return None, None

        cx_o = x_o_full.float().mean()
        cx_s = x_s_full.float().mean()

        y_anchor_orig = torch.where(inner_orig)[0]
        y_anchor_swap = torch.where(inner_swap)[0]

        top_o_k = max(1, int(y_anchor_orig.shape[0] * 0.01))
        top_s_k = max(1, int(y_anchor_swap.shape[0] * 0.01))
        cy_o = torch.topk(y_anchor_orig.float(), k=top_o_k, largest=False).values.mean()
        cy_s = torch.topk(y_anchor_swap.float(), k=top_s_k, largest=False).values.mean()

        mouthzoom = parameters.get("MouthParserStretchDecimalSlider", 1.05)
        scale_factor = (w_s / w_o) * mouthzoom

        if scale_factor <= 0.0:
            return None, None

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

        # 3. Composite the overlay
        overlay = overlay * content_mask_blurred + dark_cavity * (
            1.0 - content_mask_blurred
        )

        # 4. WIDEN THE MASK to catch persistent edge pixels
        overlay_mask = self._dilate_binary(inner_swap.float(), 3, mode="conv")

        dynamic_kernel = int(w_s.item() * 0.15) | 1
        dynamic_kernel = max(5, min(31, dynamic_kernel))

        blurred_mask = v2.functional.gaussian_blur(
            overlay_mask.unsqueeze(0),
            kernel_size=dynamic_kernel,
            sigma=dynamic_kernel / 3.0,
        ).squeeze(0)

        # Restrict strictly to inner_swap to prevent bleeding onto the swapped lips
        # 1. Allow a maximum physical expansion of exactly 2 pixels onto the lips
        base_inner = inner_swap.float()
        allowed_bleed_region = self._dilate_binary(base_inner, 2, mode="conv")

        # 2. Blur this strict boundary slightly so it acts as a smooth braking gradient
        soft_limit = v2.functional.gaussian_blur(
            allowed_bleed_region.unsqueeze(0), kernel_size=3, sigma=2.0
        ).squeeze(0)

        # 3. Force the absolute inner mouth to remain strictly untouched (100% opaque)
        soft_limit = torch.maximum(soft_limit, base_inner)

        # 4. Multiply the final mask by this soft limit to gently but rapidly fade out any excess blur
        final_mask = blurred_mask * soft_limit

        return overlay, final_mask

    def get_mouth_overlay(self, swap_img, original_img, parameters):
        """
        Public helper to retrieve the mouth overlay based on UI parameters.
        Routes to original mouth alignment or swapped mouth enhancement & zoom.
        """
        if not parameters.get("MouthParserStretchToggle", False):
            return None

        labels_swap = self._faceparser_labels(swap_img)

        if parameters.get("MouthParserStretchOriginalToggle", False):
            # The user wants the ORIGINAL mouth (with Alignment and Zoom)
            labels_orig = self._faceparser_labels(original_img)
            return self._enhance_and_align_original_mouth(
                original_img, swap_img, labels_orig, labels_swap, parameters
            )
        else:
            # The user wants the SWAPPED mouth (with Upscale and Zoom)
            return self._enhance_and_align_swapped_mouth(
                swap_img, labels_swap, parameters
            )

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

        # OPTIMIZED: Replaced expensive class instantiation with a lightweight functional wrapper
        def resize_to_target(tensor):
            return v2.functional.resize(
                tensor,
                [target_h, target_w],
                interpolation=v2.InterpolationMode.BILINEAR,
                antialias=True,
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
                fp = torch.maximum(fp, comb)

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
        # Custom provider: use PyTorch OccluderTorch
        if self.models_processor.provider_name == "Custom":
            runner = self._get_occluder_runner()
            if runner is not None:
                with torch.no_grad():
                    with self._get_runner_lock(runner):
                        result = runner(image)
                        output.copy_(result.squeeze())
                        if self.models_processor.device == "cuda":
                            torch.cuda.current_stream().synchronize()
                return
            # runner unavailable — fall through to ORT

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
            device_type=self.models_processor.device,
            device_id=0,
            element_type=np.float32,
            shape=(1, 3, 256, 256),
            buffer_ptr=image.data_ptr(),
        )
        io_binding.bind_output(
            name="output",
            device_type=self.models_processor.device,
            device_id=0,
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
            # PRE-INFERENCE SYNC
            if self.models_processor.device == "cuda":
                torch.cuda.current_stream().synchronize()
            elif self.models_processor.device != "cpu":
                self.models_processor.syncvec.cpu()

            ort_session.run_with_iobinding(io_binding)

        finally:
            if is_lazy_build:
                self.models_processor.hide_build_dialog.emit()

    def run_faceparser(self, img, out):
        """
        Robust runner for FaceParser that bypasses generic run_onnx
        to avoid 'features' NodeArg error.
        """
        model_key = "FaceParser"

        # 1. Try TensorRT Execution first (Preferred)
        if hasattr(self.models_processor, "models_trt"):
            trt_model = self.models_processor.models_trt.get(model_key)
            if trt_model is not None:
                trt_model.run(img, out)
                return

        # 2. Try ONNX Runtime Execution
        session = self.models_processor.models.get(model_key)

        if session is None:
            session = self.models_processor.load_model(model_key)

        if session is not None:
            try:
                output_names = [x.name for x in session.get_outputs()]
                input_name = session.get_inputs()[0].name

                if img.is_cuda:
                    img_np = img.cpu().numpy()
                else:
                    img_np = img.numpy()

                result = session.run(output_names, {input_name: img_np})[0]
                out.copy_(torch.from_numpy(result))
            except Exception as e:
                print(f"[ERROR] run_faceparser (ONNX) failed: {e}")
        else:
            print("[ERROR] FaceParser model not found or failed to load.")

    def apply_dfl_xseg(self, img, amount, mouth, parameters, inner_mouth_mask=None):
        # FM-07: use .get() for all parameter accesses to avoid KeyError
        amount2 = -parameters.get("DFLXSeg2SizeSlider", 0)
        amount_calc = -parameters.get("BackgroundParserTextureSlider", 0)

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

        outpred = 1.0 - outpred
        outpred = torch.unsqueeze(outpred, 0).type(torch.float32)

        outpred_calc = torch.where(outpred_calc < 0.1, 0, 1).float()
        outpred_calc = 1.0 - outpred_calc
        outpred_calc = torch.unsqueeze(outpred_calc, 0).type(torch.float32)

        outpred_calc_dill = outpred_calc.clone()

        if amount2 != amount:
            outpred2 = outpred.clone()

        if amount > 0:
            r = int(amount)
            k = 2 * r + 1
            outpred = F.max_pool2d(outpred, kernel_size=k, stride=1, padding=r)
            outpred = outpred.clamp(0, 1)

        elif amount < 0:
            r = int(-amount)
            k = 2 * r + 1
            outpred = 1 - outpred
            outpred = F.max_pool2d(outpred, kernel_size=k, stride=1, padding=r)
            outpred = 1 - outpred
            outpred = outpred.clamp(0, 1)

        blur_amount = parameters.get("OccluderXSegBlurSlider", 0)
        if blur_amount > 0:
            # OPTIMIZED: Zero-overhead functional blur (bypasses class instantiation and dict caching)
            k_size = blur_amount * 2 + 1
            sigma = (blur_amount + 1) * 0.2
            outpred = v2.functional.gaussian_blur(
                outpred, [k_size, k_size], [sigma, sigma]
            )

        outpred_noFP = outpred.clone()
        if amount2 != amount:
            if amount2 > 0:
                r2 = int(amount2)
                k2 = 2 * r2 + 1
                outpred2 = F.max_pool2d(outpred2, kernel_size=k2, stride=1, padding=r2)
                outpred2 = outpred2.clamp(0, 1)

            elif amount2 < 0:
                r2 = int(-amount2)
                k2 = 2 * r2 + 1
                outpred2 = 1 - outpred2
                outpred2 = F.max_pool2d(outpred2, kernel_size=k2, stride=1, padding=r2)
                outpred2 = 1 - outpred2
                outpred2 = outpred2.clamp(0, 1)

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

        # FM-07: use .get() for BgExcludeEnableToggle and BGExcludeBlurAmountSlider
        if parameters.get("BgExcludeEnableToggle", False) and amount_calc != 0:
            if amount_calc > 0:
                r2 = int(amount_calc)
                k2 = 2 * r2 + 1
                outpred_calc_dill = F.max_pool2d(
                    outpred_calc_dill, kernel_size=k2, stride=1, padding=r2
                )
                outpred_calc_dill = outpred_calc_dill.clamp(0, 1)
                bg_blur = parameters.get("BGExcludeBlurAmountSlider", 0)
                if bg_blur > 0:
                    k_bg = bg_blur * 2 + 1
                    s_bg = (bg_blur + 1) * 0.2
                    outpred_calc_dill = v2.functional.gaussian_blur(
                        outpred_calc_dill.type(torch.float32),
                        [k_bg, k_bg],
                        [s_bg, s_bg],
                    )
                outpred_calc_dill = outpred_calc_dill.clamp(0, 1)
            elif amount_calc < 0:
                r2 = int(-amount_calc)
                k2 = 2 * r2 + 1
                outpred_calc_dill = 1 - outpred_calc_dill
                outpred_calc_dill = F.max_pool2d(
                    outpred_calc_dill, kernel_size=k2, stride=1, padding=r2
                )
                outpred_calc_dill = 1 - outpred_calc_dill
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
                outpred_calc_dill = outpred_calc_dill.clamp(0, 1)
        return outpred, outpred_calc, outpred_calc_dill, outpred_noFP

    def run_dfl_xseg(self, image, output):
        # Custom provider: use PyTorch XSegTorch
        if self.models_processor.provider_name == "Custom":
            runner = self._get_xseg_runner()
            if runner is not None:
                with torch.no_grad():
                    with self._get_runner_lock(runner):
                        result = runner(image)
                        output.copy_(result.view(256, 256))
                        if self.models_processor.device == "cuda":
                            torch.cuda.current_stream().synchronize()
                return
            # runner unavailable — fall through to ORT

        model_name = "XSeg"
        ort_session = self.models_processor.models.get(model_name)
        if not ort_session:
            ort_session = self.models_processor.load_model(model_name)

        if not ort_session:
            return

        io_binding = ort_session.io_binding()
        io_binding.bind_input(
            name="in_face:0",
            device_type=self.models_processor.device,
            device_id=0,
            element_type=np.float32,
            shape=image.size(),
            buffer_ptr=image.data_ptr(),
        )
        io_binding.bind_output(
            name="out_mask:0",
            device_type=self.models_processor.device,
            device_id=0,
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
            # PRE-INFERENCE SYNC
            if self.models_processor.device == "cuda":
                torch.cuda.current_stream().synchronize()
            elif self.models_processor.device != "cpu":
                self.models_processor.syncvec.cpu()

            ort_session.run_with_iobinding(io_binding)

        finally:
            if is_lazy_build:
                self.models_processor.hide_build_dialog.emit()

    def run_onnx(self, image_tensor, output_tensor, model_key):
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
            device_type=self.models_processor.device,
            device_id=0,
            element_type=np.float32,
            shape=image_tensor.shape,
            buffer_ptr=image_tensor.data_ptr(),
        )
        io_binding.bind_output(
            name=output_name,
            device_type=self.models_processor.device,
            device_id=0,
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
            # PRE-INFERENCE SYNC
            if self.models_processor.device == "cuda":
                torch.cuda.current_stream().synchronize()
            elif self.models_processor.device != "cpu":
                self.models_processor.syncvec.cpu()

            sess.run_with_iobinding(io_binding)

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

    def apply_fake_diff(
        self,
        swapped_face,
        original_face,
        lower_thresh,
        lower_value,
        upper_thresh,
        upper_value,
        middle_value,
        parameters,
    ):
        """
        Calculates a fake difference map based on pixel-wise absolute difference.
        Used for VGG mask emulation when VGG is disabled.
        """
        diff = torch.abs(swapped_face - original_face)

        # OPTIMIZED: Deterministic strided slicing instead of random sampling.
        # Eliminates CUDA RNG initialization overhead and prevents temporal mask flickering.
        sample = diff.view(-1)[::10]
        diff_max = torch.quantile(sample, 0.99)
        diff = torch.clamp(diff, max=diff_max)

        diff_min = diff.min()
        diff_max = diff.max()
        # FM-03: guard against division by zero when diff_min == diff_max
        diff_norm = (diff - diff_min) / (diff_max - diff_min + 1e-6)

        diff_mean = diff_norm.mean(dim=0)
        scale = diff_mean / lower_thresh
        result = torch.where(
            diff_mean < lower_thresh,
            lower_value + scale * (middle_value - lower_value),
            # FM-04: use zeros_like instead of empty_like to avoid uninitialized memory
            torch.zeros_like(diff_mean),
        )

        middle_scale = (diff_mean - lower_thresh) / (upper_thresh - lower_thresh)
        result = torch.where(
            (diff_mean >= lower_thresh) & (diff_mean <= upper_thresh),
            middle_value + middle_scale * (upper_value - middle_value),
            result,
        )

        above_scale = (diff_mean - upper_thresh) / (1 - upper_thresh)
        result = torch.where(
            diff_mean > upper_thresh,
            upper_value + above_scale * (1 - upper_value),
            result,
        )

        return result.unsqueeze(0)

    def apply_perceptual_diff_onnx(
        self,
        swapped_face,
        original_face,
        swap_mask,
        lower_thresh,
        lower_value,
        upper_thresh,
        upper_value,
        middle_value,
        feature_layer,
        ExcludeVGGMaskEnableToggle,
    ):
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
            self.active_models.add(model_key)

        def preprocess(img):
            img = img.clone().float() / 255.0
            mean = _VGG_MEAN.to(img.device)
            std = _VGG_STD.to(img.device)
            return ((img - mean) / std).unsqueeze(0).contiguous()

        swapped = preprocess(swapped_face)
        original = preprocess(original_face)

        if (
            self.models_processor.provider_name == "Custom"
            and feature_layer == "combo_relu3_3_relu3_1"
        ):
            vgg_runner = self._get_vgg_combo_runner()
            if vgg_runner is not None:
                with torch.no_grad():
                    with self._get_runner_lock(vgg_runner):
                        swapped_feat = vgg_runner(swapped)
                        original_feat = vgg_runner(original)
            else:
                shape = feature_shapes[feature_layer]
                outpred = torch.empty(shape, dtype=torch.float32, device=swapped.device)
                outpred2 = torch.empty_like(outpred)
                swapped_feat = self.run_onnx(swapped, outpred, model_key)
                original_feat = self.run_onnx(original, outpred2, model_key)
        else:
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
