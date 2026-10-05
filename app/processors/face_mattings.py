"""Portrait matting (MODNet) for hairline-aware swap blending.

Uses MODNet ("Real-Time Trimap-Free Portrait Matting", AAAI 2022) to produce an
alpha matte of the *target* face crop, extracts the hair region as the part of
the matte that lies outside the swap mask, and adjusts the final swap mask so
the target's own hair survives the composite.

Two modes, both operating on the 512x512 aligned swap mask right before
paste-back:

- ``Protect Target Hair``: hair pixels are subtracted from the swap mask, so
  swapped content never covers the target's hairline / stray strands. This is
  the fix for the classic "helmet hairline" swap artifact.
- ``Soften Hairline``: the swap mask gets an extra gaussian blur only where
  hair is present, widening the transition band at the forehead edge instead
  of changing coverage.

The model is the community ONNX export of ``modnet_photographic_portrait_matting``
(512x512 fixed input, RGB, normalized to [-1, 1] with mean/scale 127.5).
"""

from typing import TYPE_CHECKING, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import v2

if TYPE_CHECKING:
    from app.processors.models_processor import ModelsProcessor
    from app.processors.workers.function_worker import FunctionWorker

MODNET_MODEL_NAME = "MODNet"
_MODNET_INPUT_SIZE = 512


class FaceMattings:
    """Runs MODNet and converts its matte into swap-mask adjustments."""

    def __init__(
        self,
        models_processor: "ModelsProcessor",
        function_worker: "FunctionWorker",
    ):
        self.models_processor = models_processor
        self.function_worker = function_worker
        self.active_models: set[str] = set()
        self._warned = False

    def unload_models(self) -> None:
        """Unloads MODNet via the central processor."""
        with self.models_processor.model_lock:
            for model_name in list(self.active_models):
                self.models_processor.unload_model(model_name)
            self.active_models.clear()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    @torch.no_grad()
    def run_modnet(self, img_chw: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Runs MODNet on a CHW RGB tensor (float or uint8, 0-255, any size).

        Returns:
            Alpha matte as (1, H, W) float32 in [0, 1] at the input's spatial
            size, or None if the model is unavailable or inference failed.
        """
        if img_chw is None or img_chw.dim() != 3:
            return None

        ort_session = self.models_processor.models.get(MODNET_MODEL_NAME)
        if not ort_session:
            ort_session = self.models_processor.load_model(MODNET_MODEL_NAME)
        if not ort_session:
            if not self._warned:
                print(
                    "[WARN] MODNet model not available. Run "
                    "'python download_models.py' to install "
                    "model_assets/modnet_photographic_portrait_matting.onnx"
                )
                self._warned = True
            return None
        self.active_models.add(MODNET_MODEL_NAME)

        _, in_h, in_w = img_chw.shape
        try:
            x = img_chw.float()
            if (in_h, in_w) != (_MODNET_INPUT_SIZE, _MODNET_INPUT_SIZE):
                x = v2.functional.resize(
                    x.unsqueeze(0),
                    [_MODNET_INPUT_SIZE, _MODNET_INPUT_SIZE],
                    interpolation=v2.InterpolationMode.BILINEAR,
                    antialias=True,
                ).squeeze(0)
            # Mean/scale 127.5 -> [-1, 1]
            x = x.sub_(127.5).div_(127.5).unsqueeze(0).contiguous()

            matte = torch.empty(
                (1, 1, _MODNET_INPUT_SIZE, _MODNET_INPUT_SIZE),
                dtype=torch.float32,
                device=self.models_processor.device,
            ).contiguous()

            io_binding = ort_session.io_binding()
            io_binding.clear_binding_inputs()
            io_binding.clear_binding_outputs()
            io_binding.bind_input(
                name="input",
                device_type=self.models_processor.device_type,
                device_id=self.models_processor.binding_device_id,
                element_type=np.float32,
                shape=tuple(x.shape),
                buffer_ptr=x.data_ptr(),
            )
            io_binding.bind_output(
                name="output",
                device_type=self.models_processor.device_type,
                device_id=self.models_processor.binding_device_id,
                element_type=np.float32,
                shape=tuple(matte.shape),
                buffer_ptr=matte.data_ptr(),
            )
            self._run_model_with_lazy_build_check(ort_session, io_binding)
        except Exception as e:
            print(f"[WARN] MODNet inference failed: {e}")
            return None

        matte = matte.clamp_(0.0, 1.0)
        if (in_h, in_w) != (_MODNET_INPUT_SIZE, _MODNET_INPUT_SIZE):
            matte = v2.functional.resize(
                matte,
                [in_h, in_w],
                interpolation=v2.InterpolationMode.BILINEAR,
                antialias=True,
            )
        return matte.squeeze(0)  # (1, H, W)

    def _run_model_with_lazy_build_check(self, ort_session, io_binding) -> None:
        """TensorRT lazy-build dialog handling, mirroring FaceRestorers."""
        is_lazy_build: bool = self.models_processor.check_and_clear_pending_build(
            MODNET_MODEL_NAME
        )
        if is_lazy_build:
            self.models_processor.show_build_dialog.emit(
                "Finalizing TensorRT Build",
                "Performing first-run inference for:\nMODNet\n\nThis may take several minutes.",
            )
        try:
            self.function_worker.run_ort_with_iobinding(ort_session, io_binding)
        finally:
            if is_lazy_build:
                self.models_processor.hide_build_dialog.emit()

    # ------------------------------------------------------------------
    # Hair region extraction and mask adjustment
    # ------------------------------------------------------------------
    @staticmethod
    def compute_hair_region(
        matte: torch.Tensor,
        face_mask: torch.Tensor,
        *,
        threshold: float = 0.4,
        face_dilation_px: int = 12,
    ) -> torch.Tensor:
        """
        Extracts the hair region from a portrait matte.

        Hair is defined as the matte alpha *outside* the (dilated) face mask:
        subtracting the dilated mask from the continuous matte leaves a soft
        gradient right at the hairline, which is exactly where blending wants
        to be gentle.

        Args:
            matte: (1, H, W) float in [0, 1], MODNet output.
            face_mask: (1, H, W) float in [0, 1], the current swap mask.
            threshold: matte alpha below this is treated as background.
            face_dilation_px: how far the face mask is grown before the
                subtraction, so the face edge itself is never called hair.

        Returns:
            (1, H, W) float in [0, 1] hair-region mask.
        """
        if matte.shape != face_mask.shape:
            face_mask = v2.functional.resize(
                face_mask.unsqueeze(0),
                [matte.shape[-2], matte.shape[-1]],
                interpolation=v2.InterpolationMode.BILINEAR,
                antialias=False,
            ).squeeze(0)

        m = matte.clamp(0.0, 1.0)
        if threshold > 0.0:
            m = (m - threshold).clamp_(min=0.0) / max(1.0 - threshold, 1e-6)

        fm = face_mask.clamp(0.0, 1.0)
        if face_dilation_px > 0:
            k = int(2 * face_dilation_px + 1)
            fm = F.max_pool2d(fm.unsqueeze(0), kernel_size=k, stride=1, padding=face_dilation_px).squeeze(0)

        return (m - fm).clamp_(0.0, 1.0)

    @staticmethod
    def _soft_blur(mask: torch.Tensor, feather_px: int) -> torch.Tensor:
        """Gaussian blur on a (1, H, W) mask; feather_px <= 0 returns input."""
        if feather_px <= 0:
            return mask
        k = int(2 * feather_px + 1)
        sigma = float(max(feather_px * 0.4, 1e-3))
        return v2.functional.gaussian_blur(
            mask.unsqueeze(0), kernel_size=[k, k], sigma=[sigma, sigma]
        ).squeeze(0)

    @torch.no_grad()
    def apply_hair_matting(
        self,
        swap_mask: torch.Tensor,
        target_face_crop: torch.Tensor,
        parameters: dict,
    ) -> torch.Tensor:
        """
        Adjusts a 512-space swap mask using the MODNet hair region of the
        aligned target crop.

        Args:
            swap_mask: (1, 512, 512) float swap mask at its final pipeline stage.
            target_face_crop: (3, 512, 512) aligned target face (0-255).
            parameters: per-face parameters dict; all keys read with defaults.

        Returns:
            The adjusted swap mask. On any failure the input is returned
            unchanged, so the pass can only ever be a no-op, never a crash.
        """
        try:
            matte = self.run_modnet(target_face_crop)
            if matte is None:
                return swap_mask

            # Face-scale adjustments may leave the crop at a size other than
            # the mask's; work strictly in swap-mask space from here on.
            if matte.shape[-2:] != swap_mask.shape[-2:]:
                matte = v2.functional.resize(
                    matte.unsqueeze(0),
                    [swap_mask.shape[-2], swap_mask.shape[-1]],
                    interpolation=v2.InterpolationMode.BILINEAR,
                    antialias=False,
                ).squeeze(0)

            strength = (
                float(parameters.get("HairMattingStrengthSlider", 100)) / 100.0
            )
            if strength <= 0.0:
                return swap_mask

            hair = self.compute_hair_region(
                matte,
                swap_mask,
                threshold=float(parameters.get("HairMattingThresholdSlider", 40))
                / 100.0,
                face_dilation_px=int(
                    parameters.get("HairMattingFaceDilationSlider", 12)
                ),
            )
            feather = int(parameters.get("HairMattingFeatherSlider", 6))
            if float(hair.max()) <= 1e-4:
                return swap_mask

            mode = str(
                parameters.get("HairMattingModeSelection", "Protect Target Hair")
            )
            if mode == "Soften Hairline":
                # Extra blur only where hair lives: widen the transition band
                # at the hairline without changing overall coverage. The mask's
                # own edge band (where blur changes anything) is included in the
                # weight, because face dilation carves exactly that strip out of
                # the hair region — without it, softening would be a no-op right
                # at the hairline where it matters.
                blurred = self._soft_blur(swap_mask, max(feather * 2, 3))
                edge_band = (blurred - swap_mask).abs().clamp_(0.0, 1.0)
                weight = (torch.maximum(hair, edge_band) * strength).clamp_(
                    0.0, 1.0
                )
                return torch.lerp(swap_mask, blurred, weight).clamp_(0.0, 1.0)

            # Default: "Protect Target Hair" — carve hair out of the swap mask
            # so the target's own hair survives the composite.
            protected = swap_mask * (1.0 - (hair * strength).clamp(0.0, 1.0))
            protected = self._soft_blur(protected, feather)
            return protected.clamp_(0.0, 1.0)
        except Exception as e:
            print(f"[WARN] Hair matting pass failed: {e}")
            return swap_mask
