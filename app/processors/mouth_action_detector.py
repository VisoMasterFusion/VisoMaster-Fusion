"""Mouth action detector: TensorFlow-based scene-level detector.

Uses a frozen-graph object detection model to score frames for mouth
action activity.  The model returns confidence scores per detected class;
this module surfaces the highest confidence for the action label of
interest (label index 1 in the bundled labels file).

Designed for real-time use: a single shared graph + persistent session are
reused across frames; a threading.Lock serialises inference so the object
is safe to call from multiple FrameWorker threads.
"""

from __future__ import annotations

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
_MODEL_PATH = os.path.join(_MODEL_DIR, "model.pb")

# Label index for the mouth action class of interest ("oral" in the source labels)
_TRIGGER_LABEL_INDEX: int = 1

# Detection input size expected by the model (width, height)
_DETECTION_INPUT_SIZE: tuple[int, int] = (320, 320)


class MouthActionDetector:
    """Singleton wrapper around the TF frozen-graph detection model.

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
        self._graph: Optional[Any] = None  # tf.Graph once loaded
        self._session: Optional[Any] = None  # tf.compat.v1.Session once loaded
        self._inp_tensor: Optional[Any] = None
        self._boxes_tensor: Optional[Any] = None
        self._scores_tensor: Optional[Any] = None
        self._classes_tensor: Optional[Any] = None
        self._infer_lock = threading.Lock()  # serialise concurrent inference calls
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
    def _lazy_load(self) -> None:
        """Load the frozen graph and open a persistent inference session."""
        # Suppress TF C++ and Python verbosity.  These env-vars must be set
        # before the tensorflow module is imported (they control the C library).
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

        try:
            import tensorflow as tf
        except ImportError:
            self._load_error = (
                "tensorflow is not installed — mouth action detection disabled. "
                "Run: pip install tensorflow"
            )
            logger.warning(self._load_error)
            return

        # Silence Python-level TF and absl loggers
        tf.get_logger().setLevel("ERROR")
        try:
            import absl.logging as _absl_log

            _absl_log.set_verbosity(_absl_log.ERROR)
        except Exception:  # noqa: BLE001
            pass

        if not os.path.isfile(_MODEL_PATH):
            self._load_error = (
                f"Mouth action model not found at {_MODEL_PATH}. Detection disabled."
            )
            logger.warning(self._load_error)
            return

        try:
            with tf.io.gfile.GFile(_MODEL_PATH, "rb") as f:
                graph_def = tf.compat.v1.GraphDef()
                graph_def.ParseFromString(f.read())

            graph = tf.Graph()
            with graph.as_default():
                tf.compat.v1.import_graph_def(graph_def, name="")

            cfg = tf.compat.v1.ConfigProto()
            cfg.gpu_options.allow_growth = True
            session = tf.compat.v1.Session(graph=graph, config=cfg)

            self._graph = graph
            self._session = session
            self._inp_tensor = graph.get_tensor_by_name("image_tensor:0")
            self._boxes_tensor = graph.get_tensor_by_name("detected_boxes:0")
            self._scores_tensor = graph.get_tensor_by_name("detected_scores:0")
            self._classes_tensor = graph.get_tensor_by_name("detected_classes:0")
            logger.info("Mouth action detector loaded from %s", _MODEL_PATH)

        except Exception as exc:  # noqa: BLE001
            self._load_error = f"Failed to load mouth action model: {exc}"
            logger.warning(self._load_error)

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
            Float in ``[0.0, 1.0]``.  Returns ``0.0`` when the model is
            unavailable, inference fails, or no trigger detections are found.
        """
        if not self.available:
            return 0.0

        try:
            import cv2

            # CHW RGB → HWC BGR → resize to model input size
            hwc_rgb = np.transpose(frame_chw_uint8, (1, 2, 0))
            hwc_bgr = hwc_rgb[..., ::-1]
            resized = cv2.resize(hwc_bgr, _DETECTION_INPUT_SIZE).astype(np.float32)
            batch = resized[np.newaxis, ...]  # (1, H, W, 3)

            assert self._session is not None  # guarded by self.available check above
            with self._infer_lock:
                _, scores, classes = self._session.run(
                    [self._boxes_tensor, self._scores_tensor, self._classes_tensor],
                    feed_dict={self._inp_tensor: batch},
                )

        except Exception as exc:  # noqa: BLE001
            logger.debug("Mouth action inference error: %s", exc)
            return 0.0

        # Find the highest confidence detection for the trigger label
        best: float = 0.0
        for s, c in zip(scores, classes):
            if int(c) == _TRIGGER_LABEL_INDEX:
                best = max(best, float(s))
        return best
