import logging
import threading
from itertools import product as product
from typing import TYPE_CHECKING, List, Dict, Optional, Any, Callable
import pickle

import torch
import numpy as np
from torchvision.transforms import v2

if TYPE_CHECKING:
    from app.processors.models_processor import ModelsProcessor
    from app.processors.workers.function_worker import FunctionWorker
from app.processors.models_data import models_dir
from app.processors.utils import faceutil

logger = logging.getLogger(__name__)


def _kps5_is_degenerate(kps5: np.ndarray, detect_mode: str = "203") -> bool:
    """
    Detects if the 5 keypoints form a degenerate geometry (extreme profile or severe pitch)
    that would cause a 5-point affine warp matrix to collapse or produce distorted "monster" faces.

    EVOLUTION & VIDEO OPTIMIZATION:
    Earlier versions relied on rigid thresholds or triangle areas, which failed because AI models
    often "hallucinate" occluded eyes on profiles. This version uses a Continuous Elliptical
    Deformation Energy model based on the nose's deviation from the face's central axis.
    Tolerances are deliberately relaxed to prevent temporal flickering (jitter) during video
    playback. It only triggers a fallback to bounding-box on absolute worst-case scenarios.

    HOW TO TUNE TOLERANCES:
    Do NOT change the `> 1.0` cutoff (it represents the exact boundary of the safe 3D sphere).
    Instead, adjust the `tol_` variables:
      - tol_x (Yaw): Increase to allow more extreme side-profiles before fallback.
      - tol_y (Pitch UP, dev_y_raw < 0): Increase to allow the head to tilt further back
        (nose crossing the eye line) before fallback.
      - tol_y (Pitch DOWN, dev_y_raw >= 0): Increase to allow the head to tilt further forward
        (nose dropping below the mouth) before fallback.
    """
    if kps5 is None or len(kps5) < 5:
        return True

    # 1. Extract Keypoints
    le = kps5[0]  # Left eye
    re = kps5[1]  # Right eye
    nose = kps5[2]  # Nose
    mouth_l = kps5[3]  # Left mouth corner
    mouth_r = kps5[4]  # Right mouth corner

    # 2. Base Distances
    eye_dist = np.linalg.norm(le - re)

    eye_mid = (le + re) / 2.0
    mouth_mid = (mouth_l + mouth_r) / 2.0

    face_axis = mouth_mid - eye_mid
    axis_length = np.linalg.norm(face_axis)

    # Prevent division by zero on corrupted data
    if axis_length < 1e-5 or eye_dist < 1e-5:
        return True

    # 3. Extreme 2D Compression Failsafe
    # Triggers if eyes are mashed together (severe profile). Slightly stricter for 478.
    compression_limit = 0.30 if detect_mode == "478" else 0.25
    if (eye_dist / axis_length) < compression_limit:
        return True

    # 4. ELLIPTICAL DEFORMATION ENERGY
    face_axis_normalized = face_axis / axis_length
    nose_vector = nose - eye_mid

    # Y-Axis (Pitch): Projection along the face axis
    nose_proj_y = np.dot(nose_vector, face_axis_normalized)

    # Ratio: 0.0 = eye level, 1.0 = mouth level
    nose_vertical_ratio = nose_proj_y / axis_length

    # X-Axis (Yaw): Orthogonal distance from the face axis
    nose_perp = nose_vector - (nose_proj_y * face_axis_normalized)
    nose_offset = np.linalg.norm(nose_perp)

    # Raw Y-axis deviation (no absolute value, to determine pitch direction)
    # A standard human nose sits roughly at 55% (0.55) down the eye-mouth axis.
    dev_y_raw = nose_vertical_ratio - 0.55

    # --- ASYMMETRIC VERTICAL TOLERANCES (PITCH) ---
    if dev_y_raw < 0:
        # Nose moves UP (Head tilted back).
        # tol_y = 0.65 allows the nose to reach or slightly pass the eye line (ratio -0.10)
        # without breaking the affine matrix. Slightly stricter for 478.
        tol_y = 0.50 if detect_mode == "478" else 0.65
    else:
        # Nose moves DOWN (Head tilted forward). Geometry remains naturally robust.
        # tol_y = 0.70 allows the nose to drop WELL BELOW the mouth (up to a ratio of 1.25).
        # Slightly stricter for 478.
        tol_y = 0.55 if detect_mode == "478" else 0.70

    dev_y = abs(dev_y_raw)
    dev_x = nose_offset / eye_dist

    # --- HORIZONTAL TOLERANCE (YAW) ---
    # Kept relatively wide (0.70) by default to ensure smooth video tracking.
    # Model '478' is extremely sensitive to affine squashing on profiles.
    # We tighten its tolerance to force the BBox fallback earlier.
    tol_x = 0.40 if detect_mode == "478" else 0.65

    # Deformation Ellipse Equation
    deformation_energy = (dev_y / tol_y) ** 2 + (dev_x / tol_x) ** 2

    # If energy exceeds 1.0, the nose is outside the safe 3D sphere.
    if deformation_energy > 1.0:
        return True

    return False


class FaceLandmarkDetectors:
    """
    Manages and executes various face landmark detection models.
    This class acts as a dispatcher to select the appropriate detector and provides
    helper methods for image preparation and filtering of detection results.
    """

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

    def __init__(
        self,
        models_processor: "ModelsProcessor",
        function_worker: "FunctionWorker",
    ):
        """
        Initializes the FaceLandmarkDetectors.

        Args:
            models_processor (ModelsProcessor): A reference to the main ModelsProcessor instance
                                                which handles model loading and device management.
            function_worker (FunctionWorker): The central FunctionWorker instance. Passed to sub-processors
                                              so they can route calls through this Facade.
        """
        self.models_processor = models_processor
        self.function_worker = function_worker
        self.active_landmark_models: set[str] = set()
        self.current_landmark_model_name: Optional[str] = None
        # Caches for model-specific data to avoid re-computation.
        self.landmark_5_anchors: list = []
        self.landmark_5_scale1_cache: Dict[tuple, torch.Tensor] = {}
        self.landmark_5_priors = None
        self._anchor_lock = threading.Lock()
        self._cache_lock = (
            threading.Lock()
        )  # Added lock to prevent dictionary Race Conditions

        # A dictionary to map a string identifier (e.g., '68') to the corresponding
        # model name and the specific function that processes its output.
        # This makes the class easily extensible with new landmark detectors.
        self.detector_map: Dict[str, Dict[str, Any]] = {
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
            "tufa98": {
                "model_name": "FaceLandmarkTUFA98",
                "function": self.detect_face_landmark_tufa98,
            },
            "orformer98": {
                "model_name": "FaceLandmarkORFormer98",
                "function": self.detect_face_landmark_orformer98,
            },
            "tufa314": {
                "model_name": "FaceLandmarkTUFA314",
                "function": self.detect_face_landmark_tufa314,
            },
            "hrffa": {
                "model_name": "FaceLandmarkHRFFA",
                "function": self.detect_face_landmark_hrffa,
            },
        }

    @torch.no_grad()
    def run_detect_landmark(
        self,
        img: torch.Tensor,
        bbox: np.ndarray,
        det_kpss: np.ndarray | None,
        detect_mode: str = "203",
        score: float = 0.5,
        from_points: bool = False,
        **kwargs: dict,
    ) -> tuple[np.ndarray | list, np.ndarray | list, np.ndarray | list]:
        """
        Main dispatcher function to run a specific landmark detector.
        It handles model loading, caching, and calling the correct processing function.
        Accepts **kwargs to pass optional parameters like 'use_mean_eyes' to detectors.
        """
        kpss_5: list = []
        kpss: list = []
        scores: list = []

        # Look up the detector information from the map.
        detector_info = self.detector_map.get(detect_mode)
        if not detector_info:
            print(f"[WARN] Landmark detector mode '{detect_mode}' not found.")
            return kpss_5, kpss, scores

        model_name: str = str(detector_info.get("model_name", ""))
        detection_function: Callable = detector_info["function"]

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

        # FLD-DEGENERATE-1: For extreme-yaw faces (full side angle), the 5-point
        # similarity transform used by `from_points=True` collapses to a
        # near-singular matrix and produces wildly wrong landmarks ("eye sideways,
        # mouth near the ear" symptom from user reports). Detect the degenerate
        # geometry and force the bbox-based warp path instead — it is less
        # accurate on aligned faces but produces sane output for side angles
        # where the keypoint-based warp simply cannot work.
        is_degenerate = False
        if det_kpss is not None and len(det_kpss) >= 5:
            try:
                is_degenerate = _kps5_is_degenerate(det_kpss, detect_mode=detect_mode)
            except TypeError:
                is_degenerate = _kps5_is_degenerate(det_kpss)

        # Fallback to BBox if the face angle is too extreme for the template
        effective_from_points = from_points
        if from_points and is_degenerate:
            effective_from_points = False

        # Call the specific detection function with kwargs
        kpss_5, kpss, scores = detection_function(
            img,
            bbox=bbox,
            det_kpss=det_kpss,
            from_points=effective_from_points,
            **kwargs,
        )

        # --- Filtering Logic ---
        # We check if detection produced a result.
        has_result = len(kpss_5) > 0
        # We check if the model provided confidence scores (Regression models like 203 do not).
        has_scores = len(scores) > 0

        if has_result:
            # FW-BUG-FIX: Exclude '478' from the threshold filter because its 'scores'
            # are actually 52 BlendShape values (expressions), not a detection confidence!
            # 'orformer98' is excluded for the same class of reason: its scores are
            # per-point visibility derived from ORFormer's internal codebook blend
            # weight, which hovers near 0.5 even on a clean, fully visible face. Passing
            # it through the threshold would reject good faces at any slider above ~50.
            if has_scores and detect_mode not in ["478", "orformer98"]:
                # If the model supports scoring (e.g., 5, 68, 98), we apply the threshold filter.
                if np.mean(scores) >= score:
                    return kpss_5, kpss, scores
                else:
                    # Filtered out due to low confidence
                    return [], [], []
            else:
                # If the model does NOT support scoring (e.g., 203, 106),
                # OR if the scores are actually blendshapes (478),
                # we implicitly trust the Face Detector's result and pass this through.
                return kpss_5, kpss, scores

        return [], [], []

    def _ensure_landmark_5_anchors(self) -> None:
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

    def _prepare_crop(
        self,
        img: torch.Tensor,
        bbox: np.ndarray,
        det_kpss: np.ndarray | None,
        from_points: bool,
        target_size: int,
        warp_mode: str | None = None,
        scale: float = 1.5,
        vy_ratio: float = 0.0,
    ) -> tuple[torch.Tensor | None, np.ndarray | None, np.ndarray | None]:
        """
        Prepares a cropped and warped face image for a landmark detector.
        This helper centralizes the repetitive pre-processing logic of aligning a face
        based on either a bounding box or existing keypoints.
        Returns:
            Tuple[torch.Tensor, np.ndarray, np.ndarray]: The cropped image, the forward transform matrix (M),
                                                          and the inverse transform matrix (IM).
        """
        import math

        if not from_points:
            # Align the face using the bounding box center and size.
            w, h = (bbox[2] - bbox[0]), (bbox[3] - bbox[1])
            center = (bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2
            _scale = target_size / (max(w, h) * scale)

            # Correct math implementation to upright tilted faces in fallback mode.
            angle = 0.0
            if det_kpss is not None and len(det_kpss) >= 2:
                dx = det_kpss[1][0] - det_kpss[0][0]
                dy = det_kpss[1][1] - det_kpss[0][1]
                if math.hypot(dx, dy) > 1e-3:
                    angle = math.degrees(math.atan2(-dy, dx))

            aimg, M = faceutil.transform(img, center, target_size, _scale, angle)
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
                device_type=self.models_processor.device_type,
                device_id=self.models_processor.binding_device_id,
                element_type=np.float32,
                shape=tensor.size(),
                buffer_ptr=tensor.data_ptr(),
            )

        # Bind outputs. The device will allocate memory for them.
        for name in output_names:
            io_binding.bind_output(
                name,
                self.models_processor.device_type,
                self.models_processor.binding_device_id,
            )

        # --- LAZY BUILD CHECK ---
        is_lazy_build = self.models_processor.check_and_clear_pending_build(model_name)
        if is_lazy_build:
            # Use the 'model_name' variable for a reliable dialog message
            self.models_processor.show_build_dialog.emit(
                "Finalizing TensorRT Build",
                f"Performing first-run inference for:\n{model_name}\n\nThis may take several minutes.",
            )

        try:
            # Run inference
            self.function_worker.run_ort_with_iobinding(model, io_binding)

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

        net_outs = self._run_onnx_binding(
            "FaceLandmark68",
            {"input": crop_image},
            ["landmarks_xyscore", "heatmaps"],
        )
        if not net_outs or len(net_outs) < 2:
            return [], [], []
        face_landmark_68 = (net_outs[0][:, :, :2][0] / 64.0).reshape(1, -1, 2) * 256.0
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
        net_outs_106 = self._run_onnx_binding(
            "FaceLandmark106", {"data": aimg}, ["fc1"]
        )
        if not net_outs_106 or len(net_outs_106) < 1:
            return [], [], []
        pred = net_outs_106[0][0]
        # 106 * 3 = 318 means the model returned (x, y, z) triples; otherwise (x, y) pairs.
        # CRITICAL FIX: Restored strict Tensor structure verification
        pred = pred.reshape((-1, 3)) if pred.shape[0] >= 3000 else pred.reshape((-1, 2))
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
            else:
                self.active_landmark_models.add("FaceBlendShapes")

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

    def _prepare_upright_square_crop(
        self,
        img: torch.Tensor,
        bbox: np.ndarray,
        det_kpss: np.ndarray | None,
        crop_scale: float,
    ) -> tuple[torch.Tensor | None, np.ndarray | None]:
        """
        Crop for TUFA / ORFormer: an upright square of side ``crop_scale * max(w, h)``
        centred on the bbox centre, resized to 256.

        Both models were trained on exactly this shape of crop (TUFA:
        ``utils/Detector.py:crop_img``, 1.15x the detector box; ORFormer:
        ``Dataloader/heatmapDataset.py`` + ``cfg.WFLW.FRACTION``, 1.20x the landmark
        box). Neither has ever seen a 5-point similarity warp onto a frontal ArcFace
        template, so the ``from_points`` path is deliberately not used for them — the
        caller's flag is ignored rather than silently producing an out-of-distribution
        crop. See onnx-export-notes.md §4.

        Roll correction from the eye keypoints IS kept (``_prepare_crop`` does this for
        the bbox path): both models trained with +-15 deg rotation augmentation, so
        uprighting keeps the face inside that range, whereas a heavily rolled face
        would be far outside it.
        """
        aimg, _M, IM = self._prepare_crop(
            img,
            bbox,
            det_kpss,
            from_points=False,
            target_size=256,
            scale=crop_scale,
        )
        if aimg is None:
            return None, None
        # Both graphs fold ImageNet normalisation internally and expect RGB in [0, 1].
        aimg = torch.div(aimg.to(dtype=torch.float32), 255.0).unsqueeze(0).contiguous()
        return aimg, IM

    def detect_face_landmark_tufa98(
        self, img, bbox, det_kpss, from_points=False, **kwargs
    ):
        """TUFA, 98-point WFLW topology. Output is normalised, hence the * 256.0."""
        aimg, IM = self._prepare_upright_square_crop(img, bbox, det_kpss, 1.15)
        if aimg is None:
            return [], [], []

        net_outs = self._run_onnx_binding(
            "FaceLandmarkTUFA98", {"image": aimg}, ["landmarks"]
        )
        if not net_outs or len(net_outs) < 1:
            return [], [], []

        pred = net_outs[0].reshape((-1, 2)) * 256.0
        pred = faceutil.trans_points2d(pred, IM)

        # TUFA emits no per-point confidence, so scores stay empty and
        # run_detect_landmark passes the result through (same as 106 / 203). The
        # converter still requires a score array positionally, hence the zeros.
        landmark_5, _ = faceutil.convert_face_landmark_98_to_5(pred, np.zeros(98))
        return landmark_5, pred, []

    def detect_face_landmark_tufa314(
        self, img, bbox, det_kpss, from_points=False, **kwargs
    ):
        """
        TUFA, dense 314-point topology. Same weights and same crop as tufa98 — only the
        structure prompt baked into the graph differs — so the * 256.0 denormalisation,
        the 1.15 crop scale and the "no confidence head" behaviour all carry over.
        """
        aimg, IM = self._prepare_upright_square_crop(img, bbox, det_kpss, 1.15)
        if aimg is None:
            return [], [], []

        net_outs = self._run_onnx_binding(
            "FaceLandmarkTUFA314", {"image": aimg}, ["landmarks"]
        )
        if not net_outs or len(net_outs) < 1:
            return [], [], []

        pred = net_outs[0].reshape((-1, 2)) * 256.0
        pred = faceutil.trans_points2d(pred, IM)

        landmark_5, _ = faceutil.convert_face_landmark_314_to_5(pred, np.zeros(314))
        return landmark_5, pred, []

    # --- HRFFA -------------------------------------------------------------
    # Every other landmark model here crops a face box. HRFFA is trained on WHOLE-HEAD
    # crops, which is where its high-angle robustness comes from, so it needs a head
    # detector in front of it. DEIMv2-Wholebody49 (class 7 = head) is that detector,
    # force-loaded as a dependency exactly the way 478 force-loads FaceBlendShapes.
    #
    # Class id 7 is "head" in both the Wholebody49 and Wholebody34 vocabularies.
    HRFFA_HEAD_CLASS_ID = 7
    # Upstream's demo defaults to 0.50. We use a looser threshold on purpose: a head
    # box is only ever used if it actually contains the already-confirmed face box
    # (_head_bbox_for_face), so a false positive elsewhere in the frame costs nothing,
    # while a miss drops us onto the approximate fallback crop.
    HRFFA_HEAD_SCORE_THRESHOLD = 0.30
    # Square side = max(w, h) * (1 + 2 * pad) with the training value pad = 0.05.
    HRFFA_CROP_SCALE = 1.1

    def detect_head_bboxes_wholebody49(
        self,
        img: torch.Tensor,
        score_threshold: float | None = None,
    ) -> np.ndarray:
        """
        Whole-head boxes for a full frame, via DEIMv2-Wholebody49.

        Returns (N, 5) float32 of [x1, y1, x2, y2, score] in original-frame pixels,
        sorted by descending score. Returns an empty (0, 5) array -- never None -- when
        the model is unavailable or finds nothing, so callers can fall back without a
        null check.

        Not a face detector and deliberately not in FaceDetectors.detector_map: it
        emits heads and no keypoints, and routing it through FaceDetectors would evict
        the resident face detector (FaceDetectors.current_detector_model keeps only one
        alive at a time).
        """
        empty = np.empty((0, 5), dtype=np.float32)
        model_name = "DEIMv2Wholebody49Head"

        if not self.models_processor.models.get(model_name):
            if not self.models_processor.load_model(model_name):
                print(
                    f"[ERROR] Failed to load dependency '{model_name}'. HRFFA will "
                    "fall back to a head box estimated from the face box; run "
                    "download_models.py if the file is missing."
                )
                return empty

        # Registered on every call rather than only on the load transition: a deferred
        # unload ("Smart Unload" during playback) drops the name from
        # active_landmark_models while the session is still resident, and it has to go
        # back in or unload_models() will never free it.
        self.active_landmark_models.add(model_name)

        session = self.models_processor.models.get(model_name)
        if session is None:
            return empty

        inp = session.get_inputs()[0]
        shape = inp.shape
        # Fixed 640x640 in the published export; fall back if a dynamic one turns up.
        in_h = shape[2] if isinstance(shape[2], int) else 640
        in_w = shape[3] if isinstance(shape[3], int) else 640

        img_h, img_w = int(img.shape[1]), int(img.shape[2])

        # A DIRECT resize, not the aspect-preserving letterbox in
        # FaceDetectors._prepare_detection_image: the model was exported for a plain
        # squash to its input square, and the normalised boxes it returns are relative
        # to the ORIGINAL frame, so undoing a letterbox would be wrong anyway.
        det_img = v2.functional.resize(
            img,
            [in_h, in_w],
            interpolation=v2.InterpolationMode.BILINEAR,
            antialias=True,
        )
        # RGB in [0, 1]; this graph applies no mean/std of its own.
        det_img = (
            torch.div(det_img.to(dtype=torch.float32), 255.0).unsqueeze(0).contiguous()
        )

        feed: Dict[str, torch.Tensor] = {inp.name: det_img}
        input_names = {i.name for i in session.get_inputs()}
        absolute = "orig_target_sizes" in input_names
        if absolute:
            # Present on some exports; the boxes then come out in absolute pixels.
            feed["orig_target_sizes"] = torch.tensor(
                [[float(img_w), float(img_h)]],
                dtype=torch.float32,
                device=self.models_processor.device,
            ).contiguous()

        net_outs = self._run_onnx_binding(model_name, feed, ["label_xyxy_score"])
        if not net_outs or len(net_outs) < 1:
            return empty

        pred = np.asarray(net_outs[0]).reshape(-1, 6)
        if score_threshold is None:
            score_threshold = self.HRFFA_HEAD_SCORE_THRESHOLD

        # DETR-style one-to-one matching, so the output is already NMS-free.
        keep = (pred[:, 0].astype(np.int64) == self.HRFFA_HEAD_CLASS_ID) & (
            pred[:, 5] >= score_threshold
        )
        pred = pred[keep]
        if len(pred) == 0:
            return empty

        boxes = pred[:, 1:5].astype(np.float32)
        if not absolute:
            boxes[:, [0, 2]] *= img_w
            boxes[:, [1, 3]] *= img_h
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, img_w - 1)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, img_h - 1)

        # Degenerate slivers are useless as a crop source.
        valid = ((boxes[:, 2] - boxes[:, 0]) >= 2.0) & (
            (boxes[:, 3] - boxes[:, 1]) >= 2.0
        )
        boxes, scores = boxes[valid], pred[valid, 5].astype(np.float32)
        if len(boxes) == 0:
            return empty

        head_bboxes = np.concatenate([boxes, scores[:, None]], axis=1)
        return head_bboxes[np.argsort(-head_bboxes[:, 4])]

    @staticmethod
    def _head_bbox_for_face(
        head_bboxes: np.ndarray | list | None, face_bbox: np.ndarray
    ) -> np.ndarray:
        """
        Pick the detected head box belonging to face_bbox, or synthesise one.

        Matching is by CONTAINMENT (intersection / face area), not IoU: a correct head
        box swallows the face box almost entirely while being far larger, which drags
        IoU down to roughly 0.25-0.4 and would make an IoU ranking prefer a wrong but
        similarly-sized box. Ties break on centre distance.

        The fallback keeps this landmark mode usable when the head detector is missing,
        failed its TensorRT build, or simply missed: a square of 1.5 * max(w, h) about
        the face centre, shifted up by 0.08 * h because a head extends much further
        above a face box than below it. It only approximates the training crop, so
        landmarks are worse than with a real head box but still sane.
        """
        face_bbox = np.asarray(face_bbox, dtype=np.float32)
        fx1, fy1, fx2, fy2 = (float(v) for v in face_bbox[:4])
        face_area = max((fx2 - fx1), 0.0) * max((fy2 - fy1), 0.0)

        best: np.ndarray | None = None
        if head_bboxes is not None and len(head_bboxes) > 0 and face_area > 0.0:
            heads = np.asarray(head_bboxes, dtype=np.float32).reshape(-1, 5)
            ix1 = np.maximum(heads[:, 0], fx1)
            iy1 = np.maximum(heads[:, 1], fy1)
            ix2 = np.minimum(heads[:, 2], fx2)
            iy2 = np.minimum(heads[:, 3], fy2)
            inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
            containment = inter / face_area

            candidates = np.flatnonzero(containment >= 0.5)
            if len(candidates) > 0:
                fcx, fcy = (fx1 + fx2) / 2.0, (fy1 + fy2) / 2.0
                hcx = (heads[candidates, 0] + heads[candidates, 2]) / 2.0
                hcy = (heads[candidates, 1] + heads[candidates, 3]) / 2.0
                dist = np.hypot(hcx - fcx, hcy - fcy)
                # Highest containment wins; centre distance only breaks near-ties.
                order = np.lexsort((dist, -containment[candidates]))
                best = heads[candidates[order[0]], :4].copy()

        if best is not None:
            return best

        w, h = fx2 - fx1, fy2 - fy1
        cx, cy = (fx1 + fx2) / 2.0, (fy1 + fy2) / 2.0 - 0.08 * h
        half = 0.5 * 1.5 * max(w, h)
        return np.array([cx - half, cy - half, cx + half, cy + half], dtype=np.float32)

    # --- YawNet -------------------------------------------------------------
    # Full-circle head yaw. Unlike faceutil.calc_face_yaw_pitch (a 5-landmark nose-offset
    # ratio that saturates near +-90 deg and reads a rear view as roughly frontal), this
    # distinguishes "profile" from "facing away", and reports its own confidence.
    #
    # Returned angles use the ring convention documented on the YawNet entry in
    # models_data: 0 = facing camera, 90 = subject's right, 180 = facing away, 270 =
    # left. Callers that want "how far from frontal" should use yaw_from_frontal().

    # Minimum von Mises concentration to act on an angle. DEFAULT 0.0 = accept every
    # reading, which is deliberate and not laziness:
    #
    # kappa is softplus(logit).clamp(1e-3, 100) (scripts/yawnet.py), and the kappa head's
    # bias is initialised so softplus(1.85) ~= 2.0 "the same as the previous FIXED
    # kappa" -- i.e. 2.0 is the model's NEUTRAL PRIOR, not a confidence floor. A
    # threshold anywhere near it would reject ordinary predictions. Upstream's own demo
    # reads only cos_sin and ignores kappa entirely, so there is no reference value to
    # copy, and calibrating one needs real in-distribution footage we do not have here.
    # Rather than ship an invented constant, the gate is off and every reading is logged
    # at debug level so a threshold can be chosen from real material later.
    #
    # Leaving it off is safe for the occluder/XSeg bypass because that path already
    # requires a COLLAPSED MASK as well as an extreme angle, so a spurious yaw on its
    # own cannot trigger it.
    YAWNET_MIN_KAPPA = 0.0

    @staticmethod
    def yaw_from_frontal(orientation_deg: float) -> float:
        """
        Fold a 0..360 ring angle onto 0..180 "degrees away from facing the camera".

        0 = facing the camera, 90 = full profile (either side), 180 = facing away.
        Side is deliberately discarded: every consumer here is symmetric, and folding
        removes the wrap-around bug of comparing raw ring degrees against a threshold
        (350 deg is 10 deg off frontal, not 350).
        """
        d = abs(float(orientation_deg)) % 360.0
        return 360.0 - d if d > 180.0 else d

    def estimate_head_yaw_yawnet(
        self,
        img: torch.Tensor,
        bbox: np.ndarray,
        head_bboxes: np.ndarray | list | None = None,
        min_kappa: float | None = None,
    ) -> tuple[float, float] | None:
        """
        Full-circle head yaw for one face, via YawNet.

        img is the whole frame (3, H, W) uint8/float RGB; bbox is that face's box in
        frame pixels. head_bboxes is the frame's DEIMv2 head boxes when the caller
        already has them ('hrffa' landmark mode); pass None and one is detected here.
        min_kappa overrides YAWNET_MIN_KAPPA, normally from the UI.

        Returns (orientation_deg, kappa) in the ring convention, or None when the model
        is unavailable or the prediction is too uncertain to use. None means "no
        opinion" and callers must keep their previous behaviour -- never treat it as 0.
        """
        model_name = "YawNet"

        if not self.models_processor.models.get(model_name):
            if not self.models_processor.load_model(model_name):
                print(
                    f"[ERROR] Failed to load '{model_name}'. Head-yaw estimation is "
                    "disabled; run download_models.py if the file is missing."
                )
                return None

        # Re-registered every call for the same reason as DEIMv2Wholebody49Head: a
        # deferred unload drops the name while the session is still resident.
        self.active_landmark_models.add(model_name)

        session = self.models_processor.models.get(model_name)
        if session is None:
            return None

        if head_bboxes is None:
            head_bboxes = self.detect_head_bboxes_wholebody49(img)
        head_bbox = self._head_bbox_for_face(head_bboxes, bbox)

        inp = session.get_inputs()[0]
        shape = inp.shape
        size = shape[2] if isinstance(shape[2], int) else 128

        img_h, img_w = int(img.shape[1]), int(img.shape[2])
        x1 = int(np.floor(np.clip(head_bbox[0], 0, img_w - 1)))
        y1 = int(np.floor(np.clip(head_bbox[1], 0, img_h - 1)))
        x2 = int(np.ceil(np.clip(head_bbox[2], 0, img_w)))
        y2 = int(np.ceil(np.clip(head_bbox[3], 0, img_h)))
        if x2 - x1 < 2 or y2 - y1 < 2:
            return None

        crop = img[:, y1:y2, x1:x2]
        # A DIRECT squash to the square, aspect ratio NOT preserved -- this is what
        # upstream's orientation.ts feeds the model, so a letterbox would be OOD.
        crop = v2.functional.resize(
            crop,
            [size, size],
            interpolation=v2.InterpolationMode.BILINEAR,
            antialias=True,
        )
        # center05: (x/255 - 0.5) / 0.5, identical to the x/127.5 - 1 in the README.
        crop = crop.to(dtype=torch.float32).div_(255.0).sub_(0.5).div_(0.5)
        crop = crop.unsqueeze(0).contiguous()

        net_outs = self._run_onnx_binding(
            model_name, {inp.name: crop}, ["cos_sin", "kappa"]
        )
        if not net_outs or len(net_outs) < 2:
            return None

        cos_sin = np.asarray(net_outs[0], dtype=np.float32).reshape(-1)
        kappa = float(np.asarray(net_outs[1], dtype=np.float32).reshape(-1)[0])
        if cos_sin.size < 2:
            return None
        if not np.isfinite(cos_sin[:2]).all() or not np.isfinite(kappa):
            return None

        # atan2(sin, cos) -> 0..360, then MIRROR: YawNet emits the yawpose convention
        # (+90 = screen left) and the ring convention is its mirror. Dropping this step
        # silently flips left and right. See models_data's YawNet entry.
        deg = (np.degrees(np.arctan2(cos_sin[1], cos_sin[0])) + 360.0) % 360.0
        deg = (360.0 - deg) % 360.0

        threshold = self.YAWNET_MIN_KAPPA if min_kappa is None else float(min_kappa)
        # Logged before the gate, and unconditionally, so that raising the threshold
        # from the UI is an informed choice rather than a guess: the readings you need
        # in order to pick a value are the ones a gate would have thrown away.
        logger.debug(
            "YawNet: %.1f deg (kappa %.3f, min %.3f)%s",
            deg,
            kappa,
            threshold,
            " -> rejected" if threshold > 0.0 and kappa < threshold else "",
        )
        if threshold > 0.0 and kappa < threshold:
            return None

        return float(deg), kappa

    def detect_face_landmark_hrffa(
        self, img, bbox, det_kpss, from_points=False, **kwargs
    ):
        """
        HRFFA, 68-point ibug topology, predicted on a whole-head crop.

        det_kpss is passed to _prepare_crop as None on purpose. That helper derives its
        roll angle from the eye keypoints, and None pins the angle to 0, which is
        exactly the axis-aligned crop HRFFA trained on. TUFA / ORFormer do want the
        uprighting (they only saw +-15 deg of rotation augmentation); HRFFA is robust
        through a full 360 deg of roll, and at the poses this mode exists for the 5
        keypoints are the least trustworthy thing available.

        from_points is ignored for the same reason it is ignored for TUFA and ORFormer:
        the model has never seen a 5-point similarity warp onto a frontal ArcFace
        template.

        head_bboxes may be supplied via kwargs by callers that already ran the head
        detector once for the whole frame (see the per-face loops in
        sequential_detector and frame_worker_standard). Without it we run the detector
        ourselves, which is correct but costs one extra inference per face.
        """
        head_bboxes = kwargs.get("head_bboxes")
        if head_bboxes is None:
            head_bboxes = self.detect_head_bboxes_wholebody49(img)
        head_bbox = self._head_bbox_for_face(head_bboxes, bbox)

        aimg, _M, IM = self._prepare_crop(
            img,
            head_bbox,
            det_kpss=None,
            from_points=False,
            target_size=256,
            scale=self.HRFFA_CROP_SCALE,
        )
        if aimg is None:
            return [], [], []

        # center05: (x / 255 - 0.5) / 0.5, i.e. x / 127.5 - 1. Nothing is folded into
        # the graph, unlike TUFA / ORFormer.
        aimg = (
            aimg.to(dtype=torch.float32).div(127.5).sub(1.0).unsqueeze(0).contiguous()
        )

        # "vis_logits" is left unbound so ORT prunes it; visibility is not surfaced.
        net_outs = self._run_onnx_binding(
            "FaceLandmarkHRFFA", {"images": aimg}, ["points"]
        )
        if not net_outs or len(net_outs) < 1:
            return [], [], []

        # "points" is normalised to the crop, hence the * 256.0.
        pred = net_outs[0].reshape((-1, 2)) * 256.0
        pred = faceutil.trans_points2d(pred, IM)

        # ibug68 is the same topology FaceLandmark68 (2dfan4) emits. HRFFA exposes no
        # per-point confidence here, so scores stay empty and run_detect_landmark
        # passes the result through unfiltered (same as TUFA / 106 / 203); the
        # converter still needs a score array positionally, hence the zeros.
        landmark_5, _ = faceutil.convert_face_landmark_68_to_5(pred, np.zeros(68))
        return landmark_5, pred, []

    def detect_face_landmark_orformer98(
        self, img, bbox, det_kpss, from_points=False, **kwargs
    ):
        """
        ORFormer, 98-point WFLW topology. Output is already in 256-crop pixels.

        The second output is ORFormer's per-patch occlusion score on a 16x16 latent
        grid (each cell = 16x16 px of the crop). It is sampled at each landmark and
        returned as ``1 - occlusion`` in the scores slot, giving the per-point
        visibility signal none of the other landmark models provide.

        Caveat that drives the exclusions in run_detect_landmark and
        FaceDetectors._refine_landmarks: this value is ORFormer's internal codebook
        BLEND WEIGHT, not a calibrated detection confidence. It sits near 0.5 even on a
        clean, fully visible face, so it must never be compared against the user's
        score threshold or against a detector confidence — only against other points on
        the same face.
        """
        aimg, IM = self._prepare_upright_square_crop(img, bbox, det_kpss, 1.20)
        if aimg is None:
            return [], [], []

        net_outs = self._run_onnx_binding(
            "FaceLandmarkORFormer98", {"image": aimg}, ["landmarks", "occlusion"]
        )
        if not net_outs or len(net_outs) < 2:
            return [], [], []

        pred = net_outs[0].reshape((-1, 2))  # already crop pixels
        occlusion = net_outs[1].reshape(16, 16)

        # Sample visibility in CROP space, before mapping the points back.
        cell = 256.0 / 16.0
        gx = np.clip((pred[:, 0] / cell).astype(np.int32), 0, 15)
        gy = np.clip((pred[:, 1] / cell).astype(np.int32), 0, 15)
        visibility = (1.0 - occlusion[gy, gx]).astype(np.float32)

        pred = faceutil.trans_points2d(pred, IM)
        landmark_5, _ = faceutil.convert_face_landmark_98_to_5(pred, visibility)
        return landmark_5, pred, visibility
