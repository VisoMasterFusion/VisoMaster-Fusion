"""Mouth action detector: ONNX Runtime-based scene-level detector.

Uses a small object detection model to score frames for mouth action activity.
The model returns confidence scores per detected class; this module surfaces
the highest confidence for the action label of interest (label index 1 in the
bundled labels file).

Designed for real-time use: a single shared ONNX Runtime session is reused
across frames; a threading.Lock serialises inference so the object is safe to
call from multiple FrameWorker threads.
"""

from __future__ import annotations

import gc
import logging
import os
import threading
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
_MODEL_DIR = os.path.join(_PROJECT_ROOT, "model_assets", "mouth_action_detector")
_MODEL_PATH = os.path.join(_MODEL_DIR, "model.onnx")

# Label index for the mouth action class of interest ("oral" in the source labels)
_TRIGGER_LABEL_INDEX: int = 1

# Detection input size expected by the model (width, height)
_DETECTION_INPUT_SIZE: tuple[int, int] = (320, 320)


class MouthActionDetector:
    """Singleton wrapper around the ONNX mouth-action detection model.

    Usage::

        detector = MouthActionDetector.get()
        if detector.available:
            confidence = detector.score(frame_chw_uint8_np)

    ``score()`` returns a float in [0.0, 1.0] representing the highest
    detection confidence for the trigger label on the given frame, or 0.0
    when the model is unavailable or no matching detections are found.
    """

    _instance: Optional["MouthActionDetector"] = None
    _class_lock = threading.Lock()

    # ------------------------------------------------------------------
    def __init__(self) -> None:
        self._session: Optional[Any] = None
        self._input_name: Optional[str] = None
        self._boxes_name: Optional[str] = None
        self._scores_name: Optional[str] = None
        self._classes_name: Optional[str] = None
        self._infer_lock = threading.Lock()
        self._load_error: Optional[str] = None

    # ------------------------------------------------------------------
    @classmethod
    def get(cls) -> "MouthActionDetector":
        """Return the shared singleton, creating and loading it on first call."""
        if cls._instance is None:
            with cls._class_lock:
                if cls._instance is None:
                    inst = cls()
                    inst._lazy_load()
                    cls._instance = inst
        return cls._instance

    # ------------------------------------------------------------------
    @staticmethod
    def _providers() -> list[Any]:
        """Return the best available ONNX Runtime provider list."""
        try:
            import onnxruntime as ort
        except ImportError:
            return []

        available = set(ort.get_available_providers())
        requested = os.environ.get("VISOMASTER_MOUTH_ACTION_PROVIDER", "").strip()

        def _fallbacks() -> list[Any]:
            providers: list[Any] = []
            if "CUDAExecutionProvider" in available:
                providers.append("CUDAExecutionProvider")
            providers.append("CPUExecutionProvider")
            return providers

        if requested:
            normalized = requested.lower()
            if (
                normalized in {"tensorrt", "trt"}
                and "TensorrtExecutionProvider" in available
            ):
                return ["TensorrtExecutionProvider", *_fallbacks()]
            if normalized == "cuda" and "CUDAExecutionProvider" in available:
                return ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if normalized == "cpu":
                return ["CPUExecutionProvider"]

        try:
            from app.processors.utils import platform_support

            default_provider = platform_support.default_execution_provider()
        except Exception:  # noqa: BLE001
            default_provider = "CPU"

        if (
            default_provider in {"TensorRT", "TensorRT-Engine"}
            and "TensorrtExecutionProvider" in available
        ):
            return ["TensorrtExecutionProvider", *_fallbacks()]
        if default_provider == "CUDA" and "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    # ------------------------------------------------------------------
    def _lazy_load(self) -> None:
        """Load the ONNX model and open a persistent inference session."""
        try:
            import onnxruntime as ort
        except ImportError:
            self._load_error = (
                "onnxruntime is not installed; mouth action detection disabled."
            )
            logger.warning(self._load_error)
            return

        if not os.path.isfile(_MODEL_PATH):
            self._load_error = (
                f"Mouth action model not found at {_MODEL_PATH}. Detection disabled."
            )
            logger.warning(self._load_error)
            return

        try:
            session_options = ort.SessionOptions()
            session_options.log_severity_level = 4
            providers = self._providers()
            if not providers:
                raise RuntimeError("no ONNX Runtime execution providers available")

            session = ort.InferenceSession(
                _MODEL_PATH,
                sess_options=session_options,
                providers=providers,
            )
            output_names = {output.name for output in session.get_outputs()}
            required_outputs = {
                "detected_boxes:0",
                "detected_scores:0",
                "detected_classes:0",
            }
            missing_outputs = sorted(required_outputs - output_names)
            if missing_outputs:
                raise RuntimeError(
                    f"mouth action ONNX model missing outputs: {missing_outputs}"
                )

            self._session = session
            self._input_name = session.get_inputs()[0].name
            self._boxes_name = "detected_boxes:0"
            self._scores_name = "detected_scores:0"
            self._classes_name = "detected_classes:0"
            logger.info(
                "Mouth action detector loaded from %s with providers %s",
                _MODEL_PATH,
                session.get_providers(),
            )

        except Exception as exc:  # noqa: BLE001
            self._load_error = f"Failed to load mouth action model: {exc}"
            logger.warning(self._load_error)

    # ------------------------------------------------------------------
    @classmethod
    def unload(cls) -> None:
        """Release the loaded model and clear the singleton."""
        with cls._class_lock:
            inst = cls._instance
            cls._instance = None

        if inst is None:
            return

        with inst._infer_lock:
            if inst._session is not None:
                try:
                    inst._session.close()
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Error closing mouth action detector session: %s", exc)
            inst._session = None
            inst._input_name = None
            inst._boxes_name = None
            inst._scores_name = None
            inst._classes_name = None

        gc.collect()
        logger.info("Mouth action detector unloaded.")

    # ------------------------------------------------------------------
    @property
    def available(self) -> bool:
        """True when the model loaded and the session is ready."""
        return self._session is not None

    @property
    def load_error(self) -> Optional[str]:
        """Human-readable reason why the model is unavailable, or None."""
        return self._load_error

    # ------------------------------------------------------------------
    def score(self, frame_chw_uint8: np.ndarray) -> float:
        """Return the highest detection confidence for the trigger label.

        Args:
            frame_chw_uint8: Frame as a ``(C, H, W)`` uint8 NumPy array (RGB).

        Returns:
            Float in ``[0.0, 1.0]``. Returns ``0.0`` when the model is
            unavailable, inference fails, or no trigger detections are found.
        """
        if not self.available:
            return 0.0

        try:
            import cv2

            # CHW RGB -> HWC BGR -> resize to model input size
            hwc_rgb = np.transpose(frame_chw_uint8, (1, 2, 0))
            hwc_bgr = hwc_rgb[..., ::-1]
            resized = cv2.resize(hwc_bgr, _DETECTION_INPUT_SIZE).astype(np.float32)
            batch = resized[np.newaxis, ...]  # (1, H, W, 3)

            assert self._session is not None
            assert self._input_name is not None
            assert self._boxes_name is not None
            assert self._scores_name is not None
            assert self._classes_name is not None
            with self._infer_lock:
                _, scores, classes = self._session.run(
                    [self._boxes_name, self._scores_name, self._classes_name],
                    {self._input_name: batch},
                )

        except Exception as exc:  # noqa: BLE001
            logger.debug("Mouth action inference error: %s", exc)
            return 0.0

        best: float = 0.0
        for s, c in zip(np.ravel(scores), np.ravel(classes)):
            if int(c) == _TRIGGER_LABEL_INDEX:
                best = max(best, float(s))
        return best
