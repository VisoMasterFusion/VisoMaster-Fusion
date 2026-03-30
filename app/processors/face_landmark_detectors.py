import threading
from itertools import product as product
from typing import TYPE_CHECKING, List, Dict, Optional
import pickle

import torch
import numpy as np
from torchvision.transforms import v2

if TYPE_CHECKING:
    from app.processors.models_processor import ModelsProcessor
from app.processors.models_data import models_dir
from app.processors.utils import faceutil


class FaceLandmarkDetectors:
    """
    Manages and executes various face landmark detection models.
    This class acts as a dispatcher to select the appropriate detector and provides
    helper methods for image preparation and filtering of detection results.
    """

    # Class-level declarations so mypy knows these types before unload_models is processed
    _custom_init_lock: threading.Lock
    _landmark5_torch: Optional[object]
    _landmark5_runner: Optional[object]
    _1k3d68_torch: Optional[object]
    _1k3d68_runner: Optional[object]
    _landmark203_torch: Optional[object]
    _landmark203_runner: Optional[object]
    _fan2dfan4_torch: Optional[object]
    _fan2dfan4_runner: Optional[object]
    _landmark478_torch: Optional[object]
    _landmark478_runner: Optional[object]
    _det106_torch: Optional[object]
    _det106_runner: Optional[object]
    _peppapig98_torch: Optional[object]
    _peppapig98_runner: Optional[object]
    _blendshapes_torch: Optional[object]
    _blendshapes_runner: Optional[object]

    def unload_models(self, keep_essential: bool = False):
        """
        Unloads landmark models.
        If keep_essential is True, it will NOT unload 'FaceLandmark203'
        as it is required by other processors (FaceEditor, ExpressionRestorer).
        """
        MODEL_203_NAME = "FaceLandmark203"  # Essential model

        models_to_unload = list(self.active_landmark_models)

        for model_name in models_to_unload:
            if keep_essential and model_name == MODEL_203_NAME:
                # Do not unload the essential model
                continue

            self.models_processor.unload_model(model_name)
            # Also remove it from the active_landmark_models set
            if model_name in self.active_landmark_models:
                self.active_landmark_models.remove(model_name)

        # If keep_essential is False, active_landmark_models has already been fully
        # emptied by the per-model remove() calls in the loop above; no extra
        # .clear() is needed here.
        with self._custom_init_lock:
            self._landmark5_torch = None
            self._landmark5_runner = None
            self._1k3d68_torch = None
            self._1k3d68_runner = None
            self._landmark203_torch = None
            self._landmark203_runner = None
            self._fan2dfan4_torch = None
            self._fan2dfan4_runner = None
            self._landmark478_torch = None
            self._landmark478_runner = None
            self._det106_torch = None
            self._det106_runner = None
            self._peppapig98_torch = None
            self._peppapig98_runner = None
            self._blendshapes_torch = None
            self._blendshapes_runner = None

    def __init__(self, models_processor: "ModelsProcessor"):
        """
        Initializes the FaceLandmarkDetectors.

        Args:
            models_processor (ModelsProcessor): A reference to the main ModelsProcessor instance
                                                which handles model loading and device management.
        """
        self.models_processor = models_processor
        self.active_landmark_models: set[str] = set()
        self.current_landmark_model_name: Optional[str] = None
        # Caches for model-specific data to avoid re-computation.
        self.landmark_5_anchors: list = []
        self.landmark_5_scale1_cache: Dict[tuple, torch.Tensor] = {}
        self.landmark_5_priors = None
        self._landmark5_torch: Optional[object] = None  # Res50Torch
        self._landmark5_runner: Optional[object] = None  # Res50CUDAGraphRunner
        self._1k3d68_torch: Optional[object] = None  # Landmark1k3d68Torch
        self._1k3d68_runner: Optional[object] = None  # Landmark1k3d68CUDAGraphRunner
        self._landmark203_torch: Optional[object] = None  # Landmark203Torch
        self._landmark203_runner: Optional[object] = None  # Landmark203CUDAGraphRunner
        self._fan2dfan4_torch: Optional[object] = None  # FAN2dfan4
        self._fan2dfan4_runner: Optional[object] = None  # FAN2dfan4CUDAGraphRunner
        self._landmark478_torch: Optional[object] = None  # FaceLandmark478Torch
        self._landmark478_runner: Optional[object] = (
            None  # FaceLandmark478CUDAGraphRunner
        )
        self._det106_torch: Optional[object] = None  # Det106Torch
        self._det106_runner: Optional[object] = None  # Det106CUDAGraphRunner
        self._peppapig98_torch: Optional[object] = None  # PeppaPig98Torch
        self._peppapig98_runner: Optional[object] = None  # PeppaPig98CUDAGraphRunner
        self._blendshapes_torch: Optional[object] = None  # FaceBlendShapesTorch
        self._blendshapes_runner: Optional[object] = (
            None  # FaceBlendShapesCUDAGraphRunner
        )
        self._anchor_lock = threading.Lock()
        self._cache_lock = (
            threading.Lock()
        )  # Added lock to prevent dictionary Race Conditions
        self._custom_inference_lock = threading.Lock()
        self._runner_locks: Dict[int, threading.Lock] = {}
        self._custom_init_lock = threading.Lock()  # serialises Custom-kernel lazy inits

        # A dictionary to map a string identifier (e.g., '68') to the corresponding
        # model name and the specific function that processes its output.
        # This makes the class easily extensible with new landmark detectors.
        self.detector_map = {
            "5": {
                "model_name": "FaceLandmark5",
                "function": self.detect_face_landmark_5,
            },
            "68": {
                "model_name": "FaceLandmark68",
                "function": self.detect_face_landmark_68,
            },
            "3d68": {
                "model_name": "FaceLandmark3d68",
                "function": self.detect_face_landmark_3d68,
            },
            "98": {
                "model_name": "FaceLandmark98",
                "function": self.detect_face_landmark_98,
            },
            "106": {
                "model_name": "FaceLandmark106",
                "function": self.detect_face_landmark_106,
            },
            "203": {
                "model_name": "FaceLandmark203",
                "function": self.detect_face_landmark_203,
            },
            "478": {
                "model_name": "FaceLandmark478",
                "function": self.detect_face_landmark_478,
            },
        }

    def run_detect_landmark(
        self,
        img,
        bbox,
        det_kpss,
        detect_mode="203",
        score=0.5,
        from_points=False,
        **kwargs,
    ):
        """
        Main dispatcher function to run a specific landmark detector.
        It handles model loading, caching, and calling the correct processing function.
        Accepts **kwargs to pass optional parameters like 'use_mean_eyes' to detectors.
        """
        kpss_5, kpss, scores = [], [], []

        # Look up the detector information from the map.
        detector_info = self.detector_map.get(detect_mode)
        if not detector_info:
            print(f"[WARN] Landmark detector mode '{detect_mode}' not found.")
            return kpss_5, kpss, scores

        model_name = detector_info["model_name"]
        detection_function = detector_info["function"]

        # Load model if it is not already loaded.
        loaded_model_instance = self.models_processor.models.get(model_name)
        if not loaded_model_instance:
            loaded_model_instance = self.models_processor.load_model(model_name)
            if loaded_model_instance:
                self.active_landmark_models.add(model_name)

        # If model still not loaded (e.g., failed to load), print a warning and return empty
        if not loaded_model_instance:
            print(
                f"[WARN] Landmark model '{model_name}' failed to load or is not available. Skipping detection."
            )
            return kpss_5, kpss, scores

        # Handle special setup cases for certain models.
        if detect_mode == "5":
            self._ensure_landmark_5_anchors()

        # Call the specific detection function with kwargs
        kpss_5, kpss, scores = detection_function(
            img, bbox=bbox, det_kpss=det_kpss, from_points=from_points, **kwargs
        )

        # --- Filtering Logic ---
        # We check if detection produced a result.
        has_result = len(kpss_5) > 0
        # We check if the model provided confidence scores (Regression models like 203 do not).
        has_scores = len(scores) > 0

        if has_result:
            if has_scores:
                # If the model supports scoring (e.g., 5, 68, 98), we apply the threshold filter.
                if np.mean(scores) >= score:
                    return kpss_5, kpss, scores
                else:
                    # Filtered out due to low confidence
                    return [], [], []
            else:
                # If the model does NOT support scoring (e.g., 203, 106, 478),
                # we implicitly trust the Face Detector's result and pass this through.
                return kpss_5, kpss, scores

        return [], [], []

    def _ensure_landmark_5_anchors(self):
        """
        Initializes the anchors for the FaceLandmark5 model.
        This complex calculation is performed only once and the result is cached for efficiency.
        Uses double-checked locking to ensure thread-safe initialization.
        """
        if self.landmark_5_priors is not None:
            return

        with self._anchor_lock:
            # Second check inside the lock to prevent redundant initialization
            # by another thread that acquired the lock first.
            if self.landmark_5_priors is not None:
                return

            feature_maps, min_sizes, steps, image_size = (
                [[64, 64], [32, 32], [16, 16]],
                [[16, 32], [64, 128], [256, 512]],
                [8, 16, 32],
                512,
            )
            anchors = []
            for k, f in enumerate(feature_maps):
                for i, j in product(range(f[0]), range(f[1])):
                    for min_size in min_sizes[k]:
                        s_kx, s_ky = min_size / image_size, min_size / image_size
                        dense_cx, dense_cy = (
                            [x * steps[k] / image_size for x in [j + 0.5]],
                            [y * steps[k] / image_size for y in [i + 0.5]],
                        )
                        for cy, cx in product(dense_cy, dense_cx):
                            anchors.extend([cx, cy, s_kx, s_ky])

            self.landmark_5_anchors = anchors
            self.landmark_5_priors = (
                torch.tensor(self.landmark_5_anchors)
                .view(-1, 4)
                .to(self.models_processor.device)
            )

    def _get_runner_lock(self, runner):
        with self._custom_inference_lock:
            r_id = id(runner)
            if r_id not in self._runner_locks:
                self._runner_locks[r_id] = threading.Lock()
            return self._runner_locks[r_id]

    def _get_landmark5_runner(self):
        """Lazy-load the Res50Torch Custom-kernel runner for FaceLandmark5."""
        if self._landmark5_runner is not None:
            return self._landmark5_runner
        self.models_processor.show_build_dialog.emit(
            "Finalizing Custom Provider",
            "Compiling & capturing CUDA graph for Landmark 5-point detector…\nFirst run only — future sessions load instantly from cache.",
        )
        try:
            with self._custom_init_lock:
                if self._landmark5_runner is not None:
                    return self._landmark5_runner
                if self._landmark5_torch is None:
                    try:
                        import pathlib
                        from custom_kernels.res50.res50_torch import Res50Torch

                        onnx_path = str(
                            pathlib.Path(__file__).parent.parent.parent
                            / "model_assets"
                            / "res50.onnx"
                        )
                        m = (
                            Res50Torch.from_onnx(onnx_path)
                            .to(self.models_processor.device)
                            .eval()
                        )
                        self._landmark5_torch = m
                    except Exception as e:
                        print(f"[Custom] res50 load failed: {e}")
                        return None
                try:
                    from custom_kernels.res50.res50_torch import build_cuda_graph_runner

                    with self.models_processor.cuda_graph_capture_lock:
                        self._landmark5_runner = build_cuda_graph_runner(
                            self._landmark5_torch, torch_compile=True
                        )
                except Exception as e:
                    print(f"[Custom] res50 CUDA graph failed, using eager: {e}")
                    self._landmark5_runner = self._landmark5_torch
        finally:
            self.models_processor.hide_build_dialog.emit()
        return self._landmark5_runner

    def _get_1k3d68_runner(self):
        """Lazy-load the Landmark1k3d68Torch Custom-kernel runner for FaceLandmark3d68."""
        if self._1k3d68_runner is not None:
            return self._1k3d68_runner
        self.models_processor.show_build_dialog.emit(
            "Finalizing Custom Provider",
            "Compiling & capturing CUDA graph for 1K3D68 3D landmark detector…\nFirst run only — future sessions load instantly from cache.",
        )
        try:
            with self._custom_init_lock:
                if self._1k3d68_runner is not None:
                    return self._1k3d68_runner
                if self._1k3d68_torch is None:
                    try:
                        import pathlib
                        from custom_kernels.landmark_1k3d68.landmark_1k3d68_torch import (
                            Landmark1k3d68Torch,
                        )

                        onnx_path = str(
                            pathlib.Path(__file__).parent.parent.parent
                            / "model_assets"
                            / "1k3d68.onnx"
                        )
                        m = (
                            Landmark1k3d68Torch.from_onnx(onnx_path)
                            .to(self.models_processor.device)
                            .eval()
                        )
                        self._1k3d68_torch = m
                    except Exception as e:
                        print(f"[Custom] 1k3d68 load failed: {e}")
                        return None
                try:
                    from custom_kernels.landmark_1k3d68.landmark_1k3d68_torch import (
                        build_cuda_graph_runner,
                    )

                    with self.models_processor.cuda_graph_capture_lock:
                        self._1k3d68_runner = build_cuda_graph_runner(
                            self._1k3d68_torch, torch_compile=True
                        )
                except Exception as e:
                    print(f"[Custom] 1k3d68 CUDA graph failed, using eager: {e}")
                    self._1k3d68_runner = self._1k3d68_torch
        finally:
            self.models_processor.hide_build_dialog.emit()
        return self._1k3d68_runner

    def _get_landmark203_runner(self):
        """Lazy-load the Landmark203Torch Custom-kernel runner for FaceLandmark203."""
        if self._landmark203_runner is not None:
            return self._landmark203_runner
        self.models_processor.show_build_dialog.emit(
            "Finalizing Custom Provider",
            "Compiling & capturing CUDA graph for Landmark 203-point detector…\nFirst run only — future sessions load instantly from cache.",
        )
        try:
            with self._custom_init_lock:
                if self._landmark203_runner is not None:
                    return self._landmark203_runner
                if self._landmark203_torch is None:
                    try:
                        import pathlib
                        from custom_kernels.landmark_203.landmark_203_torch import (
                            Landmark203Torch,
                        )

                        onnx_path = str(
                            pathlib.Path(__file__).parent.parent.parent
                            / "model_assets"
                            / "landmark.onnx"
                        )
                        m = (
                            Landmark203Torch.from_onnx(onnx_path)
                            .to(self.models_processor.device)
                            .eval()
                        )
                        self._landmark203_torch = m
                    except Exception as e:
                        print(f"[Custom] landmark_203 load failed: {e}")
                        return None
                try:
                    from custom_kernels.landmark_203.landmark_203_torch import (
                        build_cuda_graph_runner,
                    )

                    with self.models_processor.cuda_graph_capture_lock:
                        self._landmark203_runner = build_cuda_graph_runner(
                            self._landmark203_torch, torch_compile=True
                        )
                except Exception as e:
                    print(f"[Custom] landmark_203 CUDA graph failed, using eager: {e}")
                    self._landmark203_runner = self._landmark203_torch
        finally:
            self.models_processor.hide_build_dialog.emit()
        return self._landmark203_runner

    def _get_fan2dfan4_runner(self):
        """Lazy-load the FAN2dfan4 Custom-kernel runner for FaceLandmark68."""
        if self._fan2dfan4_runner is not None:
            return self._fan2dfan4_runner
        self.models_processor.show_build_dialog.emit(
            "Finalizing Custom Provider",
            "Compiling & capturing CUDA graph for 2DFan4 face alignment…\nFirst run only — future sessions load instantly from cache.",
        )
        try:
            with self._custom_init_lock:
                if self._fan2dfan4_runner is not None:
                    return self._fan2dfan4_runner
                if self._fan2dfan4_torch is None:
                    try:
                        import pathlib
                        from custom_kernels.fan_2dfan4.fan_2dfan4_torch import FAN2dfan4

                        onnx_path = str(
                            pathlib.Path(__file__).parent.parent.parent
                            / "model_assets"
                            / "2dfan4.onnx"
                        )
                        m = (
                            FAN2dfan4.from_onnx(onnx_path)
                            .to(self.models_processor.device)
                            .eval()
                        )
                        self._fan2dfan4_torch = m
                    except Exception as e:
                        print(f"[Custom] fan_2dfan4 load failed: {e}")
                        return None
                try:
                    from custom_kernels.fan_2dfan4.fan_2dfan4_torch import (
                        build_cuda_graph_runner,
                    )

                    with self.models_processor.cuda_graph_capture_lock:
                        self._fan2dfan4_runner = build_cuda_graph_runner(
                            self._fan2dfan4_torch, torch_compile=True
                        )
                except Exception as e:
                    print(f"[Custom] fan_2dfan4 CUDA graph failed, using eager: {e}")
                    self._fan2dfan4_runner = self._fan2dfan4_torch
        finally:
            self.models_processor.hide_build_dialog.emit()
        return self._fan2dfan4_runner

    def _get_landmark478_runner(self):
        """Lazy-load the FaceLandmark478Torch Custom-kernel runner for FaceLandmark478."""
        if self._landmark478_runner is not None:
            return self._landmark478_runner
        self.models_processor.show_build_dialog.emit(
            "Finalizing Custom Provider",
            "Compiling & capturing CUDA graph for Face Landmark 478 (MediaPipe)…\nFirst run only — future sessions load instantly from cache.",
        )
        try:
            with self._custom_init_lock:
                if self._landmark478_runner is not None:
                    return self._landmark478_runner
                if self._landmark478_torch is None:
                    try:
                        import pathlib
                        from custom_kernels.face_landmark478.face_landmark478_torch import (
                            FaceLandmark478Torch,
                        )

                        onnx_path = str(
                            pathlib.Path(__file__).parent.parent.parent
                            / "model_assets"
                            / "face_landmarks_detector_Nx3x256x256.onnx"
                        )
                        m = (
                            FaceLandmark478Torch.from_onnx(onnx_path)
                            .to(self.models_processor.device)
                            .eval()
                        )
                        self._landmark478_torch = m
                    except Exception as e:
                        print(f"[Custom] face_landmark478 load failed: {e}")
                        return None
                try:
                    from custom_kernels.face_landmark478.face_landmark478_torch import (
                        build_cuda_graph_runner,
                    )

                    with self.models_processor.cuda_graph_capture_lock:
                        self._landmark478_runner = build_cuda_graph_runner(
                            self._landmark478_torch, torch_compile=True
                        )
                except Exception as e:
                    print(
                        f"[Custom] face_landmark478 CUDA graph failed, using eager: {e}"
                    )
                    self._landmark478_runner = self._landmark478_torch
        finally:
            self.models_processor.hide_build_dialog.emit()
        return self._landmark478_runner

    def _get_blendshapes_runner(self):
        """Lazy-load the FaceBlendShapesTorch Custom-kernel runner."""
        if self._blendshapes_runner is not None:
            return self._blendshapes_runner
        self.models_processor.show_build_dialog.emit(
            "Finalizing Custom Provider",
            "Compiling & capturing CUDA graph for Face Blendshapes…\nFirst run only — future sessions load instantly from cache.",
        )
        try:
            with self._custom_init_lock:
                if self._blendshapes_runner is not None:
                    return self._blendshapes_runner
                if self._blendshapes_torch is None:
                    try:
                        import pathlib
                        from custom_kernels.face_blendshapes.face_blendshapes_torch import (
                            FaceBlendShapesTorch,
                        )

                        onnx_path = str(
                            pathlib.Path(__file__).parent.parent.parent
                            / "model_assets"
                            / "face_blendshapes_Nx146x2.onnx"
                        )
                        m = (
                            FaceBlendShapesTorch.from_onnx(onnx_path)
                            .to(self.models_processor.device)
                            .eval()
                        )
                        self._blendshapes_torch = m
                    except Exception as e:
                        print(f"[Custom] face_blendshapes load failed: {e}")
                        return None
                try:
                    from custom_kernels.face_blendshapes.face_blendshapes_torch import (
                        build_cuda_graph_runner,
                    )

                    with self.models_processor.cuda_graph_capture_lock:
                        self._blendshapes_runner = build_cuda_graph_runner(
                            self._blendshapes_torch, torch_compile=True
                        )
                except Exception as e:
                    print(
                        f"[Custom] face_blendshapes CUDA graph failed, using eager: {e}"
                    )
                    self._blendshapes_runner = self._blendshapes_torch
        finally:
            self.models_processor.hide_build_dialog.emit()
        return self._blendshapes_runner

    def _get_det106_runner(self):
        """Lazy-load the Det106Torch Custom-kernel runner for FaceLandmark106."""
        if self._det106_runner is not None:
            return self._det106_runner
        self.models_processor.show_build_dialog.emit(
            "Finalizing Custom Provider",
            "Compiling & capturing CUDA graph for 106-point face detector…\nFirst run only — future sessions load instantly from cache.",
        )
        try:
            with self._custom_init_lock:
                if self._det106_runner is not None:
                    return self._det106_runner
                if self._det106_torch is None:
                    try:
                        import pathlib
                        from custom_kernels.det_106.det_106_torch import Det106Torch

                        onnx_path = str(
                            pathlib.Path(__file__).parent.parent.parent
                            / "model_assets"
                            / "2d106det.onnx"
                        )
                        m = (
                            Det106Torch.from_onnx(onnx_path)
                            .to(self.models_processor.device)
                            .eval()
                        )
                        self._det106_torch = m
                    except Exception as e:
                        print(f"[Custom] det_106 load failed: {e}")
                        return None
                try:
                    from custom_kernels.det_106.det_106_torch import (
                        build_cuda_graph_runner,
                    )

                    with self.models_processor.cuda_graph_capture_lock:
                        self._det106_runner = build_cuda_graph_runner(
                            self._det106_torch, torch_compile=True
                        )
                except Exception as e:
                    print(f"[Custom] det_106 CUDA graph failed, using eager: {e}")
                    self._det106_runner = self._det106_torch
        finally:
            self.models_processor.hide_build_dialog.emit()
        return self._det106_runner

    def _get_peppapig98_runner(self):
        """Lazy-load the PeppaPig98Torch Custom-kernel runner for FaceLandmark98."""
        if self._peppapig98_runner is not None:
            return self._peppapig98_runner
        self.models_processor.show_build_dialog.emit(
            "Finalizing Custom Provider",
            "Compiling & capturing CUDA graph for PeppaPig 98-point detector…\nFirst run only — future sessions load instantly from cache.",
        )
        try:
            with self._custom_init_lock:
                if self._peppapig98_runner is not None:
                    return self._peppapig98_runner
                if self._peppapig98_torch is None:
                    try:
                        import pathlib
                        from custom_kernels.peppapig_98.peppapig_98_torch import (
                            PeppaPig98Torch,
                        )

                        onnx_path = str(
                            pathlib.Path(__file__).parent.parent.parent
                            / "model_assets"
                            / "peppapig_teacher_Nx3x256x256.onnx"
                        )
                        m = (
                            PeppaPig98Torch(onnx_path)
                            .to(self.models_processor.device)
                            .eval()
                        )
                        self._peppapig98_torch = m
                    except Exception as e:
                        print(f"[Custom] peppapig_98 load failed: {e}")
                        return None
                try:
                    from custom_kernels.peppapig_98.peppapig_98_torch import (
                        build_cuda_graph_runner,
                    )

                    with self.models_processor.cuda_graph_capture_lock:
                        self._peppapig98_runner = build_cuda_graph_runner(
                            self._peppapig98_torch, torch_compile=True
                        )
                except Exception as e:
                    print(f"[Custom] peppapig_98 CUDA graph failed, using eager: {e}")
                    self._peppapig98_runner = self._peppapig98_torch
        finally:
            self.models_processor.hide_build_dialog.emit()
        return self._peppapig98_runner

    def _prepare_crop(
        self,
        img,
        bbox,
        det_kpss,
        from_points,
        target_size,
        warp_mode=None,
        scale=1.5,
        vy_ratio=0.0,
    ):
        """
        Prepares a cropped and warped face image for a landmark detector.
        This helper centralizes the repetitive pre-processing logic of aligning a face
        based on either a bounding box or existing keypoints.

        Returns:
            Tuple[torch.Tensor, np.ndarray, np.ndarray]: The cropped image, the forward transform matrix (M),
                                                          and the inverse transform matrix (IM).
        """
        if not from_points:
            # Align the face using the bounding box center and size.
            w, h = (bbox[2] - bbox[0]), (bbox[3] - bbox[1])
            center = (bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2
            _scale = target_size / (max(w, h) * scale)
            aimg, M = faceutil.transform(img, center, target_size, _scale, 0)
            IM = faceutil.invertAffineTransform(M)
        else:
            if det_kpss is None or len(det_kpss) == 0:
                return None, None, None
            # Align the face using provided keypoints. Different modes use different alignment templates.
            if warp_mode in ["arcface128", "arcfacemap"]:
                aimg, M = faceutil.warp_face_by_face_landmark_5(
                    img,
                    det_kpss,
                    image_size=target_size,
                    mode=warp_mode,
                    interpolation=v2.InterpolationMode.BILINEAR,
                )
                IM = faceutil.invertAffineTransform(M)
            else:  # Default for models like landmark_203 which use a more generic warp.
                aimg, M, IM = faceutil.warp_face_by_face_landmark_x(
                    img,
                    det_kpss,
                    dsize=target_size,
                    scale=scale,
                    vy_ratio=vy_ratio,
                    interpolation=v2.InterpolationMode.BILINEAR,
                )
        return aimg, M, IM

    def _run_onnx_binding(
        self,
        model_name: str,
        input_bindings: Dict[str, torch.Tensor],
        output_names: List[str],
    ) -> List[np.ndarray]:
        """
        A centralized helper function to execute an ONNX model using efficient I/O binding.
        This avoids data copies between CPU and GPU and includes critical synchronization
        steps for safe memory access.

        Args:
            model_name (str): The name of the model to execute.
            input_bindings (Dict): A dictionary mapping input names to their torch.Tensor data.
            output_names (List): A list of the names of the output nodes.

        Returns:
            List[np.ndarray]: A list of numpy arrays containing the model's output.
        """
        # Check the model cache first to avoid the overhead of load_model when
        # the model is already loaded. Fall back to load_model (which is thread-safe)
        # only when the model is not yet present, preventing a KeyError if another
        # thread unloads the model between the check in run_detect_landmark and here.
        # CRITICAL FIX: Restored the strict thread-safe load_model call to prevent race condition
        model = self.models_processor.load_model(model_name)

        # Failsafe: If load_model fails (e.g., file not found, TRT build fail),
        # model will be None. We must abort to prevent a crash.
        if model is None:
            print(f"[ERROR] Failed to get or load model '{model_name}'.")
            return []

        io_binding = model.io_binding()

        # Bind inputs to the model.
        for name, tensor in input_bindings.items():
            io_binding.bind_input(
                name=name,
                device_type=self.models_processor.device,
                device_id=0,
                element_type=np.float32,
                shape=tensor.size(),
                buffer_ptr=tensor.data_ptr(),
            )

        # Bind outputs. The device will allocate memory for them.
        for name in output_names:
            io_binding.bind_output(name, self.models_processor.device)

        # --- LAZY BUILD CHECK ---
        is_lazy_build = self.models_processor.check_and_clear_pending_build(model_name)
        if is_lazy_build:
            # Use the 'model_name' variable for a reliable dialog message
            self.models_processor.show_build_dialog.emit(
                "Finalizing TensorRT Build",
                f"Performing first-run inference for:\n{model_name}\n\nThis may take several minutes.",
            )

        try:
            # PRE-INFERENCE SYNC: Ensure PyTorch has finished preparing the memory
            # before ONNX Runtime starts reading from the IOBinding pointers.
            if self.models_processor.device == "cuda":
                torch.cuda.current_stream().synchronize()
            elif self.models_processor.device != "cpu":
                self.models_processor.syncvec.cpu()

            # Run inference
            model.run_with_iobinding(io_binding)

            # POST-INFERENCE SYNC : Ensure the GPU has completed all
            # calculations before ONNX Runtime attempts to copy the result back to CPU RAM.
            # Without this, copy_outputs_to_cpu() might grab an incomplete tensor.
            if self.models_processor.device == "cuda":
                torch.cuda.current_stream().synchronize()
            elif self.models_processor.device != "cpu":
                self.models_processor.syncvec.cpu()

            # Copy results back to CPU safely
            net_outs = io_binding.copy_outputs_to_cpu()

        finally:
            if is_lazy_build:
                self.models_processor.hide_build_dialog.emit()

        return net_outs

    def detect_face_landmark_5(self, img, bbox, det_kpss, from_points=False, **kwargs):
        if not from_points:
            w, h = (bbox[2] - bbox[0]), (bbox[3] - bbox[1])
            center = (bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2
            _scale = 512.0 / (max(w, h) * 1.5)
            image, M = faceutil.transform(img, center, 512, _scale, 0)
        else:
            image, M = faceutil.warp_face_by_face_landmark_5(
                img,
                det_kpss,
                512,
                mode="arcface128",
                interpolation=v2.InterpolationMode.BILINEAR,
            )

        # OPTIMIZATION: Bypassed multiple .permute() ping-pongs.
        # Broadcasting the mean subtraction directly on the (C, H, W) tensor saves VRAM operations.
        mean = torch.tensor(
            [104.0, 117.0, 123.0],
            dtype=torch.float32,
            device=self.models_processor.device,
        ).view(3, 1, 1)
        image = torch.sub(image.float(), mean).unsqueeze(0)

        # Prepare scaling factor for post-processing.
        height, width = 512, 512
        # CRITICAL FIX: Thread-safe cache access without destructive clear
        with self._cache_lock:
            if (width, height) not in self.landmark_5_scale1_cache:
                self.landmark_5_scale1_cache[(width, height)] = torch.tensor(
                    [width, height] * 5,
                    dtype=torch.float32,
                    device=self.models_processor.device,
                )
            scale1 = self.landmark_5_scale1_cache[(width, height)]

        # Run inference.
        if self.models_processor.provider_name == "Custom":
            runner = self._get_landmark5_runner()
            if runner is not None:
                with torch.no_grad():
                    with self._get_runner_lock(runner):
                        conf, landmarks = runner(image)
                # conf: (1,10752,2) float32 GPU — already softmax-applied
                # landmarks: (1,10752,10) float32 GPU
            else:
                # Fall back to ORT if Custom kernel unavailable
                net_outs = self._run_onnx_binding(
                    "FaceLandmark5", {"input": image}, ["conf", "landmarks"]
                )
                if not net_outs or len(net_outs) < 2:
                    return [], [], []
                conf = torch.from_numpy(net_outs[0]).to(self.models_processor.device)
                landmarks = torch.from_numpy(net_outs[1]).to(
                    self.models_processor.device
                )
        else:
            net_outs = self._run_onnx_binding(
                "FaceLandmark5", {"input": image}, ["conf", "landmarks"]
            )
            if not net_outs or len(net_outs) < 2:
                return [], [], []
            conf = torch.from_numpy(net_outs[0]).to(self.models_processor.device)
            landmarks = torch.from_numpy(net_outs[1]).to(self.models_processor.device)

        # Post-process the raw model output.
        scores = torch.squeeze(conf)[:, 1]
        priors, pre = self.landmark_5_priors, torch.squeeze(landmarks, 0)

        # OPTIMIZATION: Vectorized decoding on the GPU.
        # Replaces the slow Python list comprehension [priors... for i in range(0, 10, 2)]
        pre_reshaped = pre.view(-1, 5, 2)
        priors_xy = priors[:, :2].unsqueeze(1)
        priors_wh = priors[:, 2:].unsqueeze(1)

        landmarks = (priors_xy + pre_reshaped * 0.1 * priors_wh).view(-1, 10) * scale1

        # OPTIMIZATION: GPU-side filtering BEFORE CPU transfer.
        # Drastically reduces the Device-to-Host (D2H) PCIe bandwidth usage.
        mask = scores > 0.1
        scores = scores[mask]
        landmarks = landmarks[mask]

        if len(scores) > 0:
            # Sort directly on the GPU
            order = torch.argsort(scores, descending=True)

            # Transfer ONLY the best result to the CPU
            best_landmark = landmarks[order[0]].cpu().numpy()
            best_score = scores[order[0]].cpu().item()

            # Reshape to standard (5, 2) format
            best_landmark = np.array(
                [[best_landmark[i], best_landmark[i + 1]] for i in range(0, 10, 2)]
            )

            # Transform landmarks back to the original image's coordinate space.
            IM = faceutil.invertAffineTransform(M)
            best_landmark = faceutil.trans_points2d(best_landmark, IM)

            return best_landmark, best_landmark, np.array([best_score])

        return [], [], []

    def detect_face_landmark_68(self, img, bbox, det_kpss, from_points=False, **kwargs):
        # This model's warping function returns a specific `affine_matrix`, so it's handled separately.
        if not from_points:
            crop_image, affine_matrix = (
                faceutil.warp_face_by_bounding_box_for_landmark_68(
                    img, bbox, (256, 256)
                )
            )
        else:
            crop_image, affine_matrix = faceutil.warp_face_by_face_landmark_5(
                img,
                det_kpss,
                256,
                mode="arcface128",
                interpolation=v2.InterpolationMode.BILINEAR,
            )

        crop_image = (
            torch.div(crop_image.to(dtype=torch.float32), 255.0)
            .unsqueeze(0)
            .contiguous()
        )

        if self.models_processor.provider_name == "Custom":
            runner = self._get_fan2dfan4_runner()
            if runner is not None:
                with torch.no_grad():
                    with self._get_runner_lock(runner):
                        lmk_xyscore, heatmaps_t = runner(crop_image)
                face_landmark_68 = (
                    lmk_xyscore[:, :, :2][0].cpu().numpy() / 64.0
                ).reshape(1, -1, 2) * 256.0
                face_heatmap = heatmaps_t.cpu().numpy()  # (1, 68, 64, 64)
            else:
                # Custom kernel unavailable — fall back to ORT
                net_outs = self._run_onnx_binding(
                    "FaceLandmark68",
                    {"input": crop_image},
                    ["landmarks_xyscore", "heatmaps"],
                )
                if not net_outs or len(net_outs) < 2:
                    return [], [], []
                face_landmark_68 = (net_outs[0][:, :, :2][0] / 64.0).reshape(
                    1, -1, 2
                ) * 256.0
                face_heatmap = net_outs[1]
        else:
            net_outs = self._run_onnx_binding(
                "FaceLandmark68",
                {"input": crop_image},
                ["landmarks_xyscore", "heatmaps"],
            )
            if not net_outs or len(net_outs) < 2:
                return [], [], []
            face_landmark_68 = (net_outs[0][:, :, :2][0] / 64.0).reshape(
                1, -1, 2
            ) * 256.0
            face_heatmap = net_outs[1]

        # OPTIMIZATION: Bypassed heavy cv2 CPU instanciation.
        # Using internal faceutil math directly on the Numpy points.
        IM = faceutil.invertAffineTransform(affine_matrix)
        face_landmark_68 = faceutil.trans_points2d(face_landmark_68[0], IM)

        face_landmark_68_score = np.amax(face_heatmap, axis=(2, 3)).reshape(-1, 1)

        # Convert the 68 points to a standard 5-point format.
        face_landmark_68_5, face_landmark_68_score = (
            faceutil.convert_face_landmark_68_to_5(
                face_landmark_68, face_landmark_68_score
            )
        )
        return face_landmark_68_5, face_landmark_68, face_landmark_68_score

    def detect_face_landmark_3d68(
        self, img, bbox, det_kpss, from_points=False, **kwargs
    ):
        # Ensure the 'meanshape_68.pkl' dependency is loaded once
        if len(self.models_processor.mean_lmk) == 0:
            try:
                with open(f"{models_dir}/meanshape_68.pkl", "rb") as f:
                    self.models_processor.mean_lmk = pickle.load(f)
            except Exception as e:
                print(
                    f"[ERROR] Failed to load 'meanshape_68.pkl' for FaceLandmark3d68: {e}"
                )
                return [], [], []  # Cannot proceed without this

        aimg, _, IM = self._prepare_crop(
            img, bbox, det_kpss, from_points, target_size=192, warp_mode="arcface128"
        )
        if aimg is None:
            return [], [], []

        aimg = (
            self.models_processor.normalize(aimg.to(dtype=torch.float32))
            .unsqueeze(0)
            .contiguous()
        )
        if self.models_processor.provider_name == "Custom":
            runner = self._get_1k3d68_runner()
            if runner is not None:
                with torch.no_grad():
                    with self._get_runner_lock(runner):
                        out = runner(aimg)  # (1, 3309) float32
                pred = out[0].cpu().numpy()
            else:
                # Custom kernel unavailable — fall back to ORT
                net_outs_3d68 = self._run_onnx_binding(
                    "FaceLandmark3d68", {"data": aimg}, ["fc1"]
                )
                if not net_outs_3d68 or len(net_outs_3d68) < 1:
                    return [], [], []
                pred = net_outs_3d68[0][0]
        else:
            net_outs_3d68 = self._run_onnx_binding(
                "FaceLandmark3d68", {"data": aimg}, ["fc1"]
            )
            if not net_outs_3d68 or len(net_outs_3d68) < 1:
                return [], [], []
            pred = net_outs_3d68[0][0]

        # Post-process the 1D prediction array into 3D/2D coordinates.
        # 68 * 3 = 204 means the model returned (x, y, z) triples; otherwise (x, y) pairs.
        # CRITICAL FIX: Restored strict Tensor structure verification
        # The ONNX model outputs either a 3D dense mesh or flat 2D points with offsets.
        pred = pred.reshape((-1, 3)) if pred.shape[0] >= 3000 else pred.reshape((-1, 2))

        if 68 < pred.shape[0]:
            pred = pred[-68:]
        pred[:, 0:2] = (pred[:, 0:2] + 1) * 96.0  # Scale to image size (192/2)
        if pred.shape[1] == 3:
            pred[:, 2] *= 96.0

        # Transform points back to original image space.
        pred = faceutil.trans_points3d(pred, IM)
        landmark2d68 = np.array(pred[:, :2])
        landmark2d68_5, _ = faceutil.convert_face_landmark_68_to_5(landmark2d68, [])
        return landmark2d68_5, landmark2d68, []

    def detect_face_landmark_98(self, img, bbox, det_kpss, from_points=False, **kwargs):
        # This model's warping function also has a unique return value ('detail').
        h, w = 0, 0
        if not from_points:
            crop_image, detail = faceutil.warp_face_by_bounding_box_for_landmark_98(
                img, bbox, (256, 256)
            )
        else:
            crop_image, M = faceutil.warp_face_by_face_landmark_5(
                img,
                det_kpss,
                image_size=256,
                mode="arcface128",
                interpolation=v2.InterpolationMode.BILINEAR,
            )
            if crop_image is not None:
                h, w = crop_image.size(1), crop_image.size(2)

        if crop_image is None:
            return [], [], []

        crop_image = (
            torch.div(crop_image.to(dtype=torch.float32), 255.0)
            .unsqueeze(0)
            .contiguous()
        )
        if self.models_processor.provider_name == "Custom":
            runner = self._get_peppapig98_runner()
            if runner is not None:
                with torch.no_grad():
                    with self._get_runner_lock(runner):
                        out_t = runner(crop_image)  # (1, 98, 3)
                landmarks_xyscore = out_t.cpu()
            else:
                # Custom kernel unavailable — fall back to ORT
                net_outs_98 = self._run_onnx_binding(
                    "FaceLandmark98", {"input": crop_image}, ["landmarks_xyscore"]
                )
                if not net_outs_98 or len(net_outs_98) < 1:
                    return [], [], []
                landmarks_xyscore = net_outs_98[0]
        else:
            net_outs_98 = self._run_onnx_binding(
                "FaceLandmark98", {"input": crop_image}, ["landmarks_xyscore"]
            )
            if not net_outs_98 or len(net_outs_98) < 1:
                return [], [], []
            landmarks_xyscore = net_outs_98[0]

        if len(landmarks_xyscore) > 0:
            one_face_landmarks = landmarks_xyscore[0]
            landmark_score, landmark = (
                one_face_landmarks[:, 2],
                one_face_landmarks[:, :2],
            )

            # Transform landmarks back using either 'detail' or the inverse matrix 'M'.
            if not from_points:
                landmark[:, 0] = landmark[:, 0] * detail[1] + detail[3] - detail[4]
                landmark[:, 1] = landmark[:, 1] * detail[0] + detail[2] - detail[4]
            else:
                landmark[:, 0] *= w
                landmark[:, 1] *= h
                landmark = faceutil.trans_points2d(
                    landmark, faceutil.invertAffineTransform(M)
                )

            landmark_5, landmark_score = faceutil.convert_face_landmark_98_to_5(
                landmark, landmark_score
            )
            return landmark_5, landmark, landmark_score
        return [], [], []

    def detect_face_landmark_106(
        self, img, bbox, det_kpss, from_points=False, **kwargs
    ):
        aimg, _, IM = self._prepare_crop(
            img, bbox, det_kpss, from_points, target_size=192, warp_mode="arcface128"
        )
        if aimg is None:
            return [], [], []

        aimg = (
            self.models_processor.normalize(aimg.to(dtype=torch.float32))
            .unsqueeze(0)
            .contiguous()
        )
        if self.models_processor.provider_name == "Custom":
            runner = self._get_det106_runner()
            if runner is not None:
                with torch.no_grad():
                    with self._get_runner_lock(runner):
                        out_t = runner(aimg)  # (1, 212)
                pred = out_t[0].cpu().numpy().reshape((-1, 2))
            else:
                # Custom kernel unavailable — fall back to ORT
                net_outs_106 = self._run_onnx_binding(
                    "FaceLandmark106", {"data": aimg}, ["fc1"]
                )
                if not net_outs_106 or len(net_outs_106) < 1:
                    return [], [], []
                pred = net_outs_106[0][0]
                pred = (
                    pred.reshape((-1, 3))
                    if pred.shape[0] >= 3000
                    else pred.reshape((-1, 2))
                )
                if 106 < pred.shape[0]:
                    pred = pred[-106:]
        else:
            net_outs_106 = self._run_onnx_binding(
                "FaceLandmark106", {"data": aimg}, ["fc1"]
            )
            if not net_outs_106 or len(net_outs_106) < 1:
                return [], [], []
            pred = net_outs_106[0][0]
            # 106 * 3 = 318 means the model returned (x, y, z) triples; otherwise (x, y) pairs.
            # CRITICAL FIX: Restored strict Tensor structure verification
            pred = (
                pred.reshape((-1, 3))
                if pred.shape[0] >= 3000
                else pred.reshape((-1, 2))
            )
            if 106 < pred.shape[0]:
                pred = pred[-106:]

        pred[:, :2] = (pred[:, :2] + 1) * 96.0
        if pred.shape[1] == 3:
            pred[:, 2] *= 96.0

        pred = faceutil.trans_points(pred, IM)
        pred_5 = (
            faceutil.convert_face_landmark_106_to_5(pred) if pred is not None else []
        )
        return pred_5, pred, []

    def detect_face_landmark_203(
        self, img, bbox, det_kpss, from_points=False, **kwargs
    ):
        # Extract the 'use_mean_eyes' parameter from kwargs, default to False.
        use_mean_eyes = kwargs.get("use_mean_eyes", False)

        # Select warp mode based on the number of keypoints available.
        warp_mode = (
            None
            if (from_points and det_kpss is not None and det_kpss.shape[0] > 5)
            else "arcface128"
        )
        aimg, M, IM = self._prepare_crop(
            img,
            bbox,
            det_kpss,
            from_points,
            target_size=224,
            warp_mode=warp_mode,
            scale=1.5,
            vy_ratio=-0.1,
        )
        if aimg is None:
            return [], [], []
        if IM is None:
            IM = faceutil.invertAffineTransform(M)

        aimg = torch.div(aimg.to(dtype=torch.float32), 255.0).unsqueeze(0).contiguous()

        if self.models_processor.provider_name == "Custom":
            runner = self._get_landmark203_runner()
            if runner is not None:
                with torch.no_grad():
                    with self._get_runner_lock(runner):
                        _, _, out_pts_t = runner(aimg)  # (1, 406) float32
                out_pts = out_pts_t[0].cpu().numpy().reshape((-1, 2)) * 224.0
            else:
                # Custom kernel unavailable — fall back to ORT
                out_lst = self._run_onnx_binding(
                    "FaceLandmark203", {"input": aimg}, ["output", "853", "856"]
                )
                if not out_lst or len(out_lst) < 3:
                    return [], [], []
                out_pts = out_lst[2].reshape((-1, 2)) * 224.0
        else:
            out_lst = self._run_onnx_binding(
                "FaceLandmark203", {"input": aimg}, ["output", "853", "856"]
            )
            if not out_lst or len(out_lst) < 3:
                return [], [], []
            out_pts = (
                out_lst[2].reshape((-1, 2)) * 224.0
            )  # The third output contains the landmarks.

        out_pts = faceutil.trans_points(out_pts, IM)
        # Pass 'use_mean_eyes' to the converter.
        out_pts_5 = (
            faceutil.convert_face_landmark_203_to_5(
                out_pts, use_mean_eyes=use_mean_eyes
            )
            if out_pts is not None
            else []
        )
        return out_pts_5, out_pts, []

    def detect_face_landmark_478(
        self, img, bbox, det_kpss, from_points=False, **kwargs
    ):
        # Extract the 'use_mean_eyes' parameter from kwargs, default to False.
        use_mean_eyes = kwargs.get("use_mean_eyes", False)

        # Ensure the 'FaceBlendShapes' dependency is loaded before we proceed
        if not self.models_processor.models.get("FaceBlendShapes"):
            # We use load_model, which handles caching. If it fails, it will return None.
            if not self.models_processor.load_model("FaceBlendShapes"):
                print(
                    "[ERROR] Failed to load dependency 'FaceBlendShapes'. Aborting landmark detection."
                )
                return [], [], []  # Fail fast

        aimg, _, IM = self._prepare_crop(
            img,
            bbox,
            det_kpss,
            from_points,
            target_size=256,
            warp_mode="arcfacemap",
            scale=1.5,
        )
        if aimg is None:
            return [], [], []

        aimg = torch.div(aimg.to(dtype=torch.float32), 255.0).unsqueeze(0).contiguous()

        if self.models_processor.provider_name == "Custom":
            runner = self._get_landmark478_runner()
            if runner is not None:
                with torch.no_grad():
                    with self._get_runner_lock(runner):
                        lmk_t, _vis, _score = runner(aimg)  # (1,1,1,1434)
                landmarks = lmk_t.cpu().numpy().reshape((1, 478, 3))
            else:
                # Custom kernel unavailable — fall back to ORT
                net_outs = self._run_onnx_binding(
                    "FaceLandmark478",
                    {"input_12": aimg},
                    ["Identity", "Identity_1", "Identity_2"],
                )
                if not net_outs or len(net_outs) < 1:
                    return [], [], []
                landmarks = net_outs[0].reshape((1, 478, 3))
        else:
            net_outs = self._run_onnx_binding(
                "FaceLandmark478",
                {"input_12": aimg},
                ["Identity", "Identity_1", "Identity_2"],
            )
            if not net_outs or len(net_outs) < 1:
                return [], [], []
            landmarks = net_outs[0].reshape((1, 478, 3))

        if len(landmarks) > 0:
            landmark = faceutil.trans_points3d(landmarks[0], IM)[:, :2].reshape(-1, 2)

            # This model uses a second network ('FaceBlendShapes') to get scores.
            landmark_for_score = landmark[self.models_processor.LandmarksSubsetIdxs]
            landmark_for_score = torch.from_numpy(
                np.expand_dims(landmark_for_score, axis=0).astype(np.float32)
            ).to(self.models_processor.device)
            landmark_score = []
            if self.models_processor.provider_name == "Custom":
                bs_runner = self._get_blendshapes_runner()
                if bs_runner is not None:
                    with torch.no_grad():
                        with self._get_runner_lock(bs_runner):
                            bs_out = bs_runner(landmark_for_score)
                            landmark_score = bs_out.cpu().numpy().flatten()
                else:
                    net_outs = self._run_onnx_binding(
                        "FaceBlendShapes",
                        {"input_points": landmark_for_score},
                        ["output"],
                    )
                    if net_outs and len(net_outs) > 0:
                        landmark_score = net_outs[0].flatten()
            else:
                net_outs = self._run_onnx_binding(
                    "FaceBlendShapes", {"input_points": landmark_for_score}, ["output"]
                )
                if net_outs and len(net_outs) > 0:
                    landmark_score = net_outs[0].flatten()

            # Pass 'use_mean_eyes' to the converter.
            landmark_5 = faceutil.convert_face_landmark_478_to_5(
                landmark, use_mean_eyes=use_mean_eyes
            )
            return landmark_5, landmark, landmark_score
        return [], [], []
