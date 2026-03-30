"""Face re-aging processor using the face_reaging ONNX model.

The model accepts a (batch, 5, H, W) input tensor composed of:
  channels 0-2 : RGB face image normalised to [0, 1]
  channel  3   : source-age map  (scalar value = source_age / 100)
  channel  4   : target-age map  (scalar value = target_age / 100)

It returns a (batch, 3, H, W) residual that is added to the original
image to produce the aged output.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from app.processors.models_processor import ModelsProcessor


class FaceReaging:
    """Applies age transformation to an aligned face crop (512 x 512 CHW RGB)."""

    MODEL_NAME = "FaceReaging"

    def __init__(self, models_processor: "ModelsProcessor") -> None:
        self.models_processor = models_processor

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def apply_reaging(
        self,
        face_chw_uint8: torch.Tensor,
        source_age: int,
        target_age: int,
    ) -> torch.Tensor:
        """Apply face re-aging transformation.

        Args:
            face_chw_uint8: CHW uint8 RGB tensor.  Expected to be 512 x 512 but
                            other even sizes work too (the model runs in a single
                            pass without sliding-window for sizes <= 512).
            source_age:     Approximate current age of the face (0–100).
            target_age:     Desired target age after transformation (0–100).

        Returns:
            Aged face as a CHW uint8 RGB tensor of the same spatial size.
            On any error the original tensor is returned unchanged.
        """
        session = self.models_processor.load_model(self.MODEL_NAME)
        if session is None:
            print(
                f"[ERROR] FaceReaging: model '{self.MODEL_NAME}' could not be loaded."
            )
            return face_chw_uint8

        try:
            # BUG-09: normalise to [0,1] float regardless of caller's tensor dtype/scale.
            #   uint8        → divide by 255  (normal path)
            #   float [0,1]  → use as-is      (already normalised)
            #   float [0,255]→ divide by 255  (pipeline float tensors before paste step)
            _f = face_chw_uint8.cpu().float()
            if face_chw_uint8.dtype == torch.uint8 or _f.max() > 2.0:
                face_float = _f / 255.0
            else:
                face_float = _f  # already in [0,1]
            del _f
            _, h, w = face_float.shape

            src_ch = torch.full((1, h, w), source_age / 100.0, dtype=torch.float32)
            tgt_ch = torch.full((1, h, w), target_age / 100.0, dtype=torch.float32)
            inp = torch.cat([face_float, src_ch, tgt_ch], dim=0).unsqueeze(
                0
            )  # (1,5,H,W)

            inp_np = inp.numpy().astype(np.float32)
            input_name = session.get_inputs()[0].name
            output_name = session.get_outputs()[0].name

            out_np = session.run([output_name], {input_name: inp_np})[0]  # (1,3,H,W)

            delta = torch.from_numpy(out_np).squeeze(0)  # (3,H,W)
            aged = torch.clamp(face_float + delta, 0.0, 1.0)
            return (aged * 255).to(torch.uint8).cpu()  # P3-05: always return on CPU

        except Exception as exc:
            print(f"[ERROR] FaceReaging.apply_reaging: {exc}")
            return face_chw_uint8
