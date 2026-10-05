"""Semantic target-hair protection using FaceParser labels and MODNet alpha."""

from typing import TYPE_CHECKING

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
    def run_modnet(self, img_chw: torch.Tensor) -> torch.Tensor | None:
        """
        Runs MODNet on a CHW RGB tensor (float or uint8, 0-255, any size).

        Returns:
            Alpha matte as (1, H, W) float32 in [0, 1] at the input's spatial
            size, or None if the model is unavailable or inference failed.
        """
        if (
            img_chw is None
            or img_chw.dim() != 3
            or img_chw.shape[0] != 3
            or min(img_chw.shape[-2:]) <= 0
        ):
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
        with self.models_processor.model_lock:
            self.active_models.add(MODNET_MODEL_NAME)

        _, in_h, in_w = img_chw.shape
        try:
            x = img_chw.to(device=self.models_processor.device, dtype=torch.float32)
            if (in_h, in_w) != (_MODNET_INPUT_SIZE, _MODNET_INPUT_SIZE):
                x = v2.functional.resize(
                    x.unsqueeze(0),
                    [_MODNET_INPUT_SIZE, _MODNET_INPUT_SIZE],
                    interpolation=v2.InterpolationMode.BILINEAR,
                    antialias=True,
                ).squeeze(0)
            # Mean/scale 127.5 -> [-1, 1]
            x = x.sub(127.5).div(127.5).unsqueeze(0).contiguous()

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
        except Exception as e:  # noqa: BLE001 -- optional provider failures must preserve the crop
            print(f"[WARN] MODNet inference failed: {e}")
            return None

        if not torch.isfinite(matte).all():
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
    def run_hair_labels(self, crop: torch.Tensor) -> torch.Tensor:
        crop = crop.to(device=self.models_processor.device, dtype=torch.float32)
        crop = F.interpolate(
            crop.unsqueeze(0), size=(512, 512), mode="bilinear", align_corners=False
        ).squeeze(0)
        labels = self.function_worker.face_masks._faceparser_labels(crop)
        with self.models_processor.model_lock:
            self.active_models.add("FaceParser")
        return labels

    @staticmethod
    def compute_hair_region(matte, labels, *, threshold=0.4, face_dilation_px=12):
        """Hair class 17, independently of swap coverage, gated by portrait alpha.

        Face dilation uses semantic skin/nose/eyes/brows/ears/mouth/neck labels;
        background and clothing must not erode hair.
        """
        labels = F.interpolate(
            labels.float().reshape(1, 1, *labels.shape[-2:]),
            size=matte.shape[-2:],
            mode="nearest",
        ).squeeze(0)
        hair = (labels == 17).float()
        face = ((labels >= 1) & (labels <= 14)).float()
        dilation = max(0, min(50, int(face_dilation_px)))
        if dilation:
            face = F.max_pool2d(
                face.unsqueeze(0), 2 * dilation + 1, 1, dilation
            ).squeeze(0)
        alpha = (matte.clamp(0, 1) - threshold).clamp(min=0) / max(1 - threshold, 1e-6)
        return hair * (1 - face) * alpha.clamp(0, 1)

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
            strength = float(parameters.get("HairMattingStrengthSlider", 100)) / 100
            if not np.isfinite(strength) or strength <= 0:
                return swap_mask
            if (
                swap_mask.ndim != 3
                or swap_mask.shape[0] != 1
                or not torch.isfinite(swap_mask).all()
            ):
                return swap_mask
            matte = self.run_modnet(target_face_crop)
            if matte is None or matte.ndim != 3 or matte.shape[0] != 1:
                return swap_mask
            if not torch.isfinite(matte).all():
                return swap_mask
            matte = matte.to(device=swap_mask.device, dtype=torch.float32)
            matte = F.interpolate(
                matte.unsqueeze(0),
                size=swap_mask.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            labels = self.run_hair_labels(target_face_crop)
            if labels.ndim != 2 or not torch.isfinite(labels).all():
                return swap_mask
            hair = self.compute_hair_region(
                matte,
                labels.to(swap_mask.device),
                threshold=max(
                    0,
                    min(
                        1, float(parameters.get("HairMattingThresholdSlider", 40)) / 100
                    ),
                ),
                face_dilation_px=int(
                    parameters.get("HairMattingFaceDilationSlider", 12)
                ),
            )
            feather = max(
                0, min(50, int(parameters.get("HairMattingFeatherSlider", 6)))
            )
            # Blur the protection field, never the allowed coverage. Earlier
            # occluder and border exclusions remain upper bounds in both modes.
            protection = self._soft_blur(hair, feather).clamp(0, 1)
            weight = (protection * min(strength, 1)).clamp(0, 1)
            if parameters.get("HairMattingModeSelection") == "Soften Hairline":
                blurred = self._soft_blur(swap_mask, max(feather * 2, 3))
                return torch.minimum(
                    swap_mask, torch.lerp(swap_mask, blurred, weight)
                ).clamp(0, 1)
            return (swap_mask * (1 - weight)).clamp(0, 1)
        except Exception as e:  # noqa: BLE001 -- optional pass must preserve earlier masks
            print(f"[WARN] Hair matting pass failed: {e}")
            return swap_mask
