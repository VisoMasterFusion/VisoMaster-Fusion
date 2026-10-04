"""
Face Detailer — small-face refine post-pass.

Inspired by ComfyUI-H3-FaceRefine (Carasibana) and Impact Pack's FaceDetailer
(ltdrdata): faces that occupy only a few dozen pixels in the source frame carry
almost no real detail for the swap/restoration stages to work with. This module
re-detects faces on the *finished* frame, crops each small face with generous
context, magnifies the crop onto a fixed canvas, runs a face restorer at that
high effective resolution, and stitches the refined face back with a dilated,
feathered face-shaped mask.

Design notes:
- Pure post-pass: it runs after the swap/edit pipeline and before debug
  overlays, so it works identically for swapped, edited, or untouched faces.
- Reuses the existing FaceDetectors and FaceRestorers sessions through the
  FunctionWorker facade — no new model files are required.
- Restorer alignment uses det_type="Reference": the crop is warped to the
  FFHQ-aligned 512 space using the detected 5-point landmarks (translated into
  crop coordinates), restored, and inverse-warped back — the same
  align/restore/un-align contract swap_core uses.
- The restorer is invoked with a dedicated slot_id (3) so its resident model
  does not fight the two pipeline restorer slots for VRAM bookkeeping.
- Paste-back composites only the face region (dilate-then-blur mask), never the
  whole context crop — context exists to give the restorer information, not to
  be pasted.
"""

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn.functional as nnF
import torchvision.transforms.v2.functional as F

if TYPE_CHECKING:
    from app.processors.models_processor import ModelsProcessor
    from app.processors.workers.function_worker import FunctionWorker

logger = logging.getLogger(__name__)

# Dedicated restorer slot for the detailer. Slots 1 and 2 belong to the
# swap pipeline's Restoration 1/2 stages; see FaceRestorers.apply_facerestorer.
DETAILER_RESTORER_SLOT_ID = 3

# Bound the temporary context canvas independently of the model's native resolution.
_CANVAS_MIN = 256
_CANVAS_MAX = 1536


class FaceDetailer:
    """Magnify-and-restore pass for faces that are small in the source frame."""

    def __init__(
        self,
        models_processor: "ModelsProcessor",
        function_worker: "FunctionWorker",
    ):
        self.models_processor = models_processor
        self.function_worker = function_worker

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    @torch.no_grad()
    def apply(
        self, img_chw_uint8: torch.Tensor, control: dict[str, Any]
    ) -> torch.Tensor:
        """
        Refine small faces in a finished frame.

        Args:
            img_chw_uint8: CHW RGB tensor (uint8 or float 0-255) on the
                processing device, at final display resolution.
            control: Global control dict. All detailer keys are read with
                .get() defaults so the pass is inert unless enabled.

        Returns:
            A tensor of the same shape/dtype/device with small faces refined.
            On any detection-level failure the input is returned unchanged.
        """
        if not control.get("FaceDetailerEnableToggle", False):
            return img_chw_uint8
        if (
            img_chw_uint8 is None
            or img_chw_uint8.dim() != 3
            or img_chw_uint8.shape[0] != 3
        ):
            return img_chw_uint8

        img_h, img_w = int(img_chw_uint8.shape[1]), int(img_chw_uint8.shape[2])
        if img_h < 64 or img_w < 64:
            return img_chw_uint8

        # --- Parameters (all optional; defaults keep the pass safe) ---
        min_face_px = float(control.get("FaceDetailerMinFaceSizeSlider", 24))
        max_face_px = float(control.get("FaceDetailerMaxFaceSizeSlider", 128))
        crop_factor = float(control.get("FaceDetailerCropFactorDecimalSlider", 2.5))
        crop_factor = min(max(crop_factor, 1.2), 5.0)
        canvas = int(control.get("FaceDetailerCanvasSizeSlider", 768))
        canvas = min(max(canvas, _CANVAS_MIN), _CANVAS_MAX)
        feather_px = float(control.get("FaceDetailerFeatherSlider", 12))
        dilate_px = float(control.get("FaceDetailerMaskDilationSlider", 10))
        blend = float(control.get("FaceDetailerBlendSlider", 100)) / 100.0
        blend = min(max(blend, 0.0), 1.0)
        max_faces = max(0, min(int(control.get("FaceDetailerMaxFacesSlider", 4)), 16))
        if blend == 0.0 or max_faces == 0 or min_face_px > max_face_px:
            return img_chw_uint8
        restorer_type = str(
            control.get("FaceDetailerRestorerTypeSelection", "GPEN-1024")
        )
        fidelity = float(control.get("FaceDetailerFidelityDecimalSlider", 0.9))
        detect_score = float(control.get("FaceDetailerDetectScoreSlider", 50)) / 100.0
        color_match = bool(control.get("FaceDetailerColorMatchToggle", True))
        # Reuse the selected detector; switching models would evict the pipeline's
        # resident detector. Keep detection state local to this frame snapshot.
        detect_mode = str(control.get("DetectorModelSelection", "RetinaFace"))
        osdface_timestep = int(control.get("FaceDetailerOSDFaceTimestepSlider", 399))
        osdface_latent = float(
            control.get("FaceDetailerOSDFaceLatentStrengthDecimalSlider", 1.0)
        )

        # --- Detect faces on the finished frame ---
        try:
            bboxes, kpss_5, _kpss_all = self.function_worker.run_detect(
                img_chw_uint8,
                detect_mode,
                # Filter by size before limiting: the detector's ranking favors
                # large faces, which can otherwise hide every eligible small face.
                max_num=0,
                score=detect_score,
                input_size=(512, 512),
                use_landmark_detection=False,
                rotation_angles=[0],
                bypass_bytetrack=True,
                control_override=control,
            )
        except Exception as e:  # noqa: BLE001 - keep optional inference from dropping a frame
            logger.warning("Face detailer detection failed: %s", e)
            return img_chw_uint8

        if bboxes is None or len(bboxes) == 0:
            return img_chw_uint8

        was_uint8 = img_chw_uint8.dtype == torch.uint8
        out = img_chw_uint8.to(dtype=torch.float32).clone()

        # Smallest faces first — they have the most to gain.
        valid_boxes = []
        for idx, value in enumerate(bboxes):
            bbox = np.asarray(value, dtype=np.float64)
            if bbox.shape != (4,) or not np.isfinite(bbox).all():
                continue
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                continue
            valid_boxes.append((idx, bbox))
        order = sorted(valid_boxes, key=lambda item: item[1][3] - item[1][1])

        refined_count = 0
        for idx, bbox in order:
            if refined_count >= max_faces:
                break
            face_h = float(bbox[3] - bbox[1])
            if face_h <= 0:
                continue
            # Only faces inside the configured size band are refined.
            if face_h < min_face_px or face_h > max_face_px:
                continue
            kps_5 = None
            if kpss_5 is not None and len(kpss_5) > idx and kpss_5[idx] is not None:
                kps_5 = np.asarray(kpss_5[idx], dtype=np.float64)
            if kps_5 is None or kps_5.shape != (5, 2) or not np.isfinite(kps_5).all():
                continue  # Reference alignment needs the 5-point landmarks

            try:
                out, did_refine = self._refine_one_face(
                    out,
                    bbox,
                    kps_5,
                    crop_factor=crop_factor,
                    canvas=canvas,
                    feather_px=feather_px,
                    dilate_px=dilate_px,
                    blend=blend,
                    restorer_type=restorer_type,
                    fidelity=fidelity,
                    detect_score=detect_score,
                    color_match=color_match,
                    osdface_timestep=osdface_timestep,
                    osdface_latent=osdface_latent,
                )
                if did_refine:
                    refined_count += 1
            except Exception as e:  # noqa: BLE001 - one failed face must not drop the frame
                # Per-face failures leave that face at its original pixels.
                logger.warning("Face detailer failed for one face: %s", e)
                continue

        if refined_count == 0:
            return img_chw_uint8

        out = out.clamp_(0.0, 255.0)
        if was_uint8:
            out = out.round_().to(torch.uint8)
        else:
            out = out.to(dtype=img_chw_uint8.dtype)
        return out.contiguous()

    # ------------------------------------------------------------------
    # Per-face pipeline
    # ------------------------------------------------------------------
    def _refine_one_face(
        self,
        frame: torch.Tensor,
        bbox: np.ndarray,
        kps_5: np.ndarray,
        *,
        crop_factor: float,
        canvas: int,
        feather_px: float,
        dilate_px: float,
        blend: float,
        restorer_type: str,
        fidelity: float,
        detect_score: float,
        color_match: bool,
        osdface_timestep: int,
        osdface_latent: float,
    ) -> tuple[torch.Tensor, bool]:
        img_h, img_w = int(frame.shape[1]), int(frame.shape[2])

        face_h = float(bbox[3] - bbox[1])
        cx = float((bbox[0] + bbox[2]) * 0.5)
        cy = float((bbox[1] + bbox[3]) * 0.5)

        # Square context crop centered on the face box.
        side = round(max(face_h, float(bbox[2] - bbox[0])) * crop_factor)
        side = min(side, img_w, img_h)
        if side < 2:
            return frame, False

        x0 = round(cx - side / 2.0)
        y0 = round(cy - side / 2.0)
        x0 = min(max(x0, 0), img_w - side)
        y0 = min(max(y0, 0), img_h - side)

        # Magnification guard: if the crop is already at/above canvas size,
        # the restorer would downscale it — nothing to gain, skip.
        magnification = canvas / float(side)
        if magnification < 1.05:
            return frame, False

        crop = frame[:, y0 : y0 + side, x0 : x0 + side]

        # Landmarks in crop coordinates, then in canvas coordinates.
        scale = canvas / float(side)
        kps_crop = (kps_5 - np.array([x0, y0], dtype=np.float64)) * scale

        crop_canvas = F.resize(
            crop.unsqueeze(0),
            [canvas, canvas],
            interpolation=F.InterpolationMode.BILINEAR,
            antialias=True,
        ).squeeze(0)

        # Align -> restore -> un-align through the existing restorer stack.
        refined_canvas = self.function_worker.apply_facerestorer(
            crop_canvas,
            "Reference",
            restorer_type,
            100.0,  # full-strength restore; user blend is applied below
            fidelity,
            detect_score,
            kps_crop,
            slot_id=DETAILER_RESTORER_SLOT_ID,
            osdface_timestep=osdface_timestep,
            osdface_latent_strength=osdface_latent,
        )
        if refined_canvas is None or refined_canvas is crop_canvas:
            return frame, False
        refined_canvas = refined_canvas.to(dtype=torch.float32)
        if refined_canvas.shape != crop_canvas.shape:
            raise ValueError("Reference restoration must return the context crop shape")
        if not torch.isfinite(refined_canvas).all().item():
            raise ValueError("Restoration returned nonfinite pixels")

        # Face box on the canvas, for masking and colour matching.
        fb = (bbox - np.array([x0, y0, x0, y0], dtype=np.float64)) * scale
        fx0, fy0 = max(float(fb[0]), 0.0), max(float(fb[1]), 0.0)
        fx1 = min(float(fb[2]), float(canvas))
        fy1 = min(float(fb[3]), float(canvas))
        if fx1 - fx0 < 4 or fy1 - fy0 < 4:
            return frame, False

        # Colour match: align the refined crop's per-channel statistics to the
        # region it replaces, so an independent restoration pass does not come
        # back subtly brighter and read as pasted on.
        if color_match:
            refined_canvas = self._colour_match_region(
                refined_canvas, crop_canvas, (fx0, fy0, fx1, fy1)
            )

        # User blend toward the original crop.
        if blend < 1.0:
            refined_canvas = torch.lerp(crop_canvas, refined_canvas, blend)

        # Build the mask at source resolution: feather/dilation sliders are in
        # source pixels. Avoid a huge Gaussian kernel on the magnified canvas.
        mask_crop = self._build_face_mask(
            side,
            (fx0 / scale, fy0 / scale, fx1 / scale, fy1 / scale),
            dilation_canvas=dilate_px,
            feather_canvas=feather_px,
            device=frame.device,
        )

        # Back to source-crop resolution and composite into the frame.
        refined_crop = F.resize(
            refined_canvas.unsqueeze(0),
            [side, side],
            interpolation=F.InterpolationMode.BILINEAR,
            antialias=True,
        ).squeeze(0)
        mask_crop = mask_crop.clamp_(0.0, 1.0).unsqueeze(0)  # 1,H,W

        region = frame[:, y0 : y0 + side, x0 : x0 + side]
        composited = refined_crop * mask_crop + region * (1.0 - mask_crop)
        frame[:, y0 : y0 + side, x0 : x0 + side] = composited
        return frame, True

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _build_face_mask(
        canvas: int,
        face_box: tuple[float, float, float, float],
        *,
        dilation_canvas: float,
        feather_canvas: float,
        device: torch.device,
    ) -> torch.Tensor:
        """Ellipse mask with separable Gaussian feathering at paste resolution."""
        fx0, fy0, fx1, fy1 = face_box
        grow = max(dilation_canvas, 0.0)
        fx0, fy0 = fx0 - grow, fy0 - grow
        fx1, fy1 = fx1 + grow, fy1 + grow

        cx, cy = (fx0 + fx1) / 2.0, (fy0 + fy1) / 2.0
        rx = max((fx1 - fx0) / 2.0, 1.0)
        ry = max((fy1 - fy0) / 2.0, 1.0)

        ys = torch.arange(canvas, device=device, dtype=torch.float32).view(-1, 1)
        xs = torch.arange(canvas, device=device, dtype=torch.float32).view(1, -1)
        dist = ((xs - cx) / rx) ** 2 + ((ys - cy) / ry) ** 2
        mask = (dist <= 1.0).to(torch.float32)

        if feather_canvas <= 0:
            return mask
        sigma = max(feather_canvas / 2.0, 0.5)
        k_size = int(2 * round(3 * sigma) + 1)
        max_k = canvas if canvas % 2 == 1 else canvas - 1  # odd, <= canvas
        k_size = min(k_size, max_k)
        if k_size >= 3:
            radius = k_size // 2
            coords = torch.arange(
                -radius, radius + 1, device=device, dtype=torch.float32
            )
            kernel = torch.exp(-0.5 * (coords / sigma).square())
            kernel /= kernel.sum()
            mask = mask[None, None]
            mask = nnF.conv2d(
                nnF.pad(mask, (radius, radius, 0, 0), mode="replicate"),
                kernel.view(1, 1, 1, -1),
            )
            mask = (
                nnF.conv2d(
                    nnF.pad(mask, (0, 0, radius, radius), mode="replicate"),
                    kernel.view(1, 1, -1, 1),
                )
                .squeeze(0)
                .squeeze(0)
            )
        return mask

    @staticmethod
    def _colour_match_region(
        refined: torch.Tensor,
        original: torch.Tensor,
        face_box: tuple[float, float, float, float],
    ) -> torch.Tensor:
        """Match per-channel mean/std of the refined face region to the original."""
        fx0, fy0, fx1, fy1 = (round(v) for v in face_box)
        ref_region = refined[:, fy0:fy1, fx0:fx1].float()
        orig_region = original[:, fy0:fy1, fx0:fx1].float()
        if ref_region.numel() == 0 or orig_region.numel() == 0:
            return refined

        ref_mean = ref_region.mean(dim=(1, 2), keepdim=True)
        ref_std = ref_region.std(dim=(1, 2), keepdim=True).clamp_(min=1.0)
        orig_mean = orig_region.mean(dim=(1, 2), keepdim=True)
        orig_std = orig_region.std(dim=(1, 2), keepdim=True).clamp_(min=1.0)

        adjusted = (refined.float() - ref_mean) * (orig_std / ref_std) + orig_mean
        return adjusted.clamp_(0.0, 255.0)
