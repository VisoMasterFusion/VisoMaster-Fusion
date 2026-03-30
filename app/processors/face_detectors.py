import threading
import math
from typing import TYPE_CHECKING, Dict, Any, Optional

import torch
from torchvision.transforms import v2
from torchvision.ops import nms
import numpy as np

if TYPE_CHECKING:
    from app.processors.models_processor import ModelsProcessor

from app.processors.utils import faceutil

try:
    from app.processors.external.yolox.tracker.byte_tracker import BYTETracker
except ImportError:
    BYTETracker = None  # type: ignore[assignment,misc]


class _BYTETRACK_ARGS:
    """BYTETracker parameters — populated from UI control values when available."""

    def __init__(self, control: dict | None = None):
        """Initialises BYTETracker parameters from the given UI control dict."""
        c = control or {}
        self.track_thresh = c.get("ByteTrackTrackThreshSlider", 40) / 100.0
        self.match_thresh = c.get("ByteTrackMatchThreshSlider", 80) / 100.0
        self.track_buffer = int(c.get("ByteTrackTrackBufferSlider", 30))
        self.mot20 = False


class FaceDetectors:
    """
    Manages and executes various face detection models.
    This class acts as a dispatcher to select the appropriate detector and provides
    helper methods for image preparation and filtering of detection results.
    """

    # Class-level declarations so mypy knows these types before unload_models is processed
    _custom_init_lock: threading.Lock
    _det10g_torch: Optional[object]
    _det10g_runner: Optional[object]
    _yolo_torch: Optional[object]
    _yolo_runner: Optional[object]

    def unload_models(self):
        """
        Unloads the currently active face detector model from memory.
        """
        if self.current_detector_model:
            self.models_processor.unload_model(self.current_detector_model)
            self.current_detector_model = None
        # Release Custom-kernel instances so they rebuild on next use.
        with self._custom_init_lock:
            self._det10g_torch = None
            self._det10g_runner = None
            self._yolo_torch = None
            self._yolo_runner = None

    def __init__(self, models_processor: "ModelsProcessor"):
        """
        Initialises the FaceDetectors instance.

        Args:
            models_processor: The parent ModelsProcessor that owns this helper.
                              Provides access to model sessions, device, and signals.
        """
        self.models_processor = models_processor
        self.center_cache: Dict[tuple, np.ndarray] = {}
        self.current_detector_model = None
        self._det10g_torch: Optional[object] = None  # Det10gTorch
        self._det10g_runner: Optional[object] = None  # Det10gGraphRunner
        self._yolo_torch: Optional[object] = None  # YoloFace8nTorch
        self._yolo_runner: Optional[object] = None  # _CapturedGraph or YoloFace8nTorch
        self._custom_init_lock = threading.Lock()  # serialises Custom-kernel lazy inits
        self._ort_model_lock = (
            threading.Lock()
        )  # serialises ONNX model switch (unload→load)
        self._custom_inference_lock = threading.Lock()
        self._runner_locks: Dict[int, threading.Lock] = {}

        # Tracking State
        self.tracker = None
        self.track_history: dict = {}  # {track_id: {'cum_score': float, 'last_seen': int}}
        self._track_history_lock = threading.Lock()
        # BT-06/BT-07: dedicated lock for BYTETracker instance and frame_id to prevent
        # concurrent workers from corrupting Kalman filter state
        self._tracker_lock = threading.Lock()
        self.lambda_s = 0.3  # Smoothing factor for cumulative scores

        # This map links a detector name (from the UI) to its model file and processing function.
        self.detector_map: Dict[str, Dict[str, Any]] = {
            "RetinaFace": {
                "model_name": "RetinaFace",
                "function": self.detect_retinaface,
            },
            "SCRFD": {"model_name": "SCRFD2.5g", "function": self.detect_scrdf},
            "Yolov8": {"model_name": "YoloFace8n", "function": self.detect_yoloface},
            "Yunet": {"model_name": "YunetN", "function": self.detect_yunet},
        }

    def _get_runner_lock(self, runner):
        with self._custom_inference_lock:
            r_id = id(runner)
            if r_id not in self._runner_locks:
                self._runner_locks[r_id] = threading.Lock()
            return self._runner_locks[r_id]

    def _prepare_detection_image(
        self, img: torch.Tensor, input_size: tuple, normalization_mode: str
    ) -> tuple[torch.Tensor, torch.Tensor, tuple]:
        """
        OPTIMIZED: Prepares an image for a face detection model by resizing and padding.
        Replaced explicit canvas creation, slicing, and permutations with native PyTorch padding
        to eliminate memory fragmentation and speed up VRAM operations.
        """
        if not isinstance(input_size, tuple):
            input_size = (input_size, input_size)

        # Calculate new dimensions to resize the image while maintaining aspect ratio.
        img_height, img_width = img.shape[1], img.shape[2]
        im_ratio = img_height / img_width
        model_ratio = input_size[1] / input_size[0]

        if im_ratio > model_ratio:
            new_height = input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = input_size[0]
            new_height = int(new_width * im_ratio)

        # Use float for det_scale calculation initially for precision
        det_scale = torch.tensor(
            new_height / float(img_height), device=self.models_processor.device
        )

        resize = v2.Resize((new_height, new_width), antialias=True)
        resized_img = resize(img)

        # --- OPTIMIZATION: Native Zero-Copy Padding ---
        # Calculate needed padding on the right and bottom
        pad_right = input_size[0] - new_width
        pad_bottom = input_size[1] - new_height

        # Determine target dtype
        canvas_dtype = (
            torch.float32
            if normalization_mode in ["retinaface", "scrfd"]
            else torch.uint8
        )

        # Cast BEFORE padding to avoid casting the newly padded pixels (saves VRAM bandwidth)
        if canvas_dtype == torch.uint8 and resized_img.dtype != torch.uint8:
            # Assuming resized_img might be float [0, 255] after resize
            resized_img_casted = resized_img.clamp(0, 255).byte()
        elif canvas_dtype == torch.float32 and resized_img.dtype == torch.uint8:
            resized_img_casted = resized_img.float()
        else:
            resized_img_casted = resized_img

        # Native PyTorch Pad: applies (left, right, top, bottom) to the last 2 dimensions (H, W)
        det_img = torch.nn.functional.pad(
            resized_img_casted, (0, pad_right, 0, pad_bottom), mode="constant", value=0
        )

        # Apply model-specific color space.
        if normalization_mode == "yunet":
            # RGB to BGR natively without permutations
            det_img = det_img[[2, 1, 0], :, :]

        # Apply normalization ONLY for RetinaFace/SCRFD here.
        if normalization_mode in ["retinaface", "scrfd"]:
            det_img = (det_img - 127.5) / 128.0  # Normalize to [-1.0, 1.0] range

        return det_img, det_scale, input_size

    def _infer_fixed_square_input_from_outputs(
        self, ort_session, first_stride: int = 8
    ) -> int | None:
        """
        Attempts to infer a fixed square model input size from the first detection head
        output shape. For RetinaFace/SCRFD this head typically has shape {N,1} where
        N = (input/stride)^2 * num_anchors and num_anchors is usually 2.
        """
        try:
            outputs = ort_session.get_outputs()
            if not outputs:
                return None

            shape = outputs[0].shape
            if not shape or len(shape) < 1:
                return None

            n = shape[0]
            if not isinstance(n, int) or n <= 0:
                return None

            # Try common anchor counts used by RetinaFace/SCRFD exports.
            for anchors in (2, 1):
                if n % anchors != 0:
                    continue
                cells = n // anchors
                side = int(round(math.sqrt(cells)))
                if side > 0 and side * side == cells:
                    return side * first_stride
        except Exception:
            return None

        return None

    def _resolve_detector_input_size(
        self, detect_mode: str, requested_input_size: tuple, ort_session
    ) -> tuple:
        """
        Resolves the effective detection input size.
        For fixed-shape RetinaFace/SCRFD exports, running at a mismatched size can spam
        ONNX Runtime VerifyOutputSizes warnings. If a fixed square size is detectable,
        prefer it over the requested value.

        Strategy (most reliable first):
        1. Read the model's declared input shape (NCHW → H/W at indices 2 and 3).
        2. Fall back to inferring from the first detection-head output shape.
        """
        if detect_mode not in ("RetinaFace", "SCRFD"):
            return requested_input_size

        # PRIMARY: read declared input spatial size directly from model metadata.
        model_input_size = None
        try:
            inputs = ort_session.get_inputs()
            if inputs:
                shape = inputs[0].shape  # NCHW: [N, C, H, W]
                if shape and len(shape) >= 4:
                    h, w = shape[2], shape[3]
                    if isinstance(h, int) and isinstance(w, int) and h > 0 and h == w:
                        model_input_size = h
        except Exception:
            pass

        # FALLBACK: infer from the first detection-head output shape.
        if model_input_size is None:
            model_input_size = self._infer_fixed_square_input_from_outputs(ort_session)

        if model_input_size is None:
            return requested_input_size

        requested_w, requested_h = requested_input_size
        if requested_w == model_input_size and requested_h == model_input_size:
            return requested_input_size

        return (model_input_size, model_input_size)

    def _filter_detections_gpu(
        self,
        scores_list,
        bboxes_list,
        kpss_list,
        img_height,
        img_width,
        det_scale,
        max_num,
        skip_nms=False,
    ):
        """
        Performs GPU-accelerated NMS, sorting, and filtering on raw detections from all angles.

        Args:
            scores_list (list): List of score arrays (np.ndarray) from each detection angle.
            bboxes_list (list): List of bounding box arrays (np.ndarray) from each detection angle.
            kpss_list (list): List of keypoint arrays (np.ndarray) from each detection angle.
            img_height (int): The *original* height of the source image.
            img_width (int): The *original* width of the source image.
            det_scale (torch.Tensor): The scaling factor used to resize the image (new_height / original_height).
            max_num (int): The maximum number of faces to return, sorted by size and centrality.
            skip_nms (bool): If True, skips the Non-Maximum Suppression step.

        Returns:
            tuple: (det, kpss_final, score_values)
                - det (np.ndarray): Final bounding boxes, scaled to original image size.
                - kpss_final (np.ndarray): Final keypoints, scaled to original image size.
                - score_values (np.ndarray): Scores for the final detections.
        """
        if not bboxes_list:
            return None, None, None

        # Convert all raw detection lists to single GPU tensors.
        scores_tensor = (
            torch.from_numpy(np.vstack(scores_list))
            .to(self.models_processor.device)
            .squeeze()
        )
        bboxes_tensor = torch.from_numpy(np.vstack(bboxes_list)).to(
            self.models_processor.device
        )
        kpss_tensor = torch.from_numpy(np.vstack(kpss_list)).to(
            self.models_processor.device
        )

        bboxes_tensor = torch.as_tensor(bboxes_tensor, dtype=torch.float32)
        scores_tensor = torch.as_tensor(scores_tensor, dtype=torch.float32).reshape(-1)

        # --- Validation Block to ensure tensors are well-formed before NMS ---
        if bboxes_tensor.numel() == 0:
            return None, None, None
        if bboxes_tensor.dim() == 1 and bboxes_tensor.numel() == 4:
            bboxes_tensor = bboxes_tensor.unsqueeze(0)
        if scores_tensor.dim() == 0:
            scores_tensor = scores_tensor.unsqueeze(0)
        if bboxes_tensor.size(0) != scores_tensor.size(0):
            # Mismatch in tensor sizes, aborting.
            return None, None, None

        # Ensure tensors are contiguous (optimizes NMS)
        bboxes_tensor = bboxes_tensor.contiguous()
        scores_tensor = scores_tensor.contiguous()

        if not skip_nms:
            # Perform Non-Maximum Suppression on the GPU to remove overlapping boxes.
            nms_thresh = 0.4
            keep_indices = nms(bboxes_tensor, scores_tensor, iou_threshold=nms_thresh)

            det_boxes, det_kpss, det_scores = (
                bboxes_tensor[keep_indices],
                kpss_tensor[keep_indices],
                scores_tensor[keep_indices],
            )
        else:
            det_boxes, det_kpss, det_scores = (
                bboxes_tensor,
                kpss_tensor,
                scores_tensor,
            )

        # Sort the remaining detections by their confidence score.
        sorted_indices = torch.argsort(det_scores, descending=True)
        det_boxes, det_kpss, det_scores = (
            det_boxes[sorted_indices],
            det_kpss[sorted_indices],
            det_scores[sorted_indices],
        )

        # If more faces are detected than max_num, select the best ones.
        if max_num > 0 and det_boxes.shape[0] > max_num:
            if det_boxes.shape[0] > 1:
                # Score faces based on a combination of their size and proximity to the image center.
                # This filtering happens on *unscaled* coordinates (relative to the padded detection image).
                area = (det_boxes[:, 2] - det_boxes[:, 0]) * (
                    det_boxes[:, 3] - det_boxes[:, 1]
                )
                # The old logic (img_height / det_scale) was mathematically incorrect and
                # produced extreme values for non-standard aspect ratios (like VR videos).
                # The correct logic is to find the center of the *active image area*
                # on the padded canvas.
                # new_height_on_canvas = img_height * det_scale
                # new_width_on_canvas = img_width * det_scale
                det_img_center_y = (img_height * det_scale) / 2.0
                det_img_center_x = (img_width * det_scale) / 2.0

                center_x = (det_boxes[:, 0] + det_boxes[:, 2]) / 2 - det_img_center_x
                center_y = (det_boxes[:, 1] + det_boxes[:, 3]) / 2 - det_img_center_y

                offset_dist_squared = center_x**2 + center_y**2
                # This score favors large faces (area) that are close to the center
                # (low offset_dist_squared).
                values = area - offset_dist_squared * 2.0
                bindex = torch.argsort(values, descending=True)[:max_num]
                det_boxes, det_kpss, det_scores = (
                    det_boxes[bindex],
                    det_kpss[bindex],
                    det_scores[bindex],
                )
            else:
                bindex = torch.arange(
                    det_boxes.shape[0], device=self.models_processor.device
                )[:max_num]
                det_boxes = det_boxes[bindex]
                det_kpss = det_kpss[bindex]
                det_scores = det_scores[bindex]

        # Transfer final results back to CPU and scale them to the original image dimensions.
        det_scale_val = det_scale.cpu().item()
        det = det_boxes.cpu().numpy() / det_scale_val
        kpss_final = det_kpss.cpu().numpy() / det_scale_val

        score_values = det_scores.cpu().numpy()

        return det, kpss_final, score_values

    def _refine_landmarks(
        self,
        img_landmark,
        det,
        kpss,
        score_values,
        use_landmark_detection,
        landmark_detect_mode,
        landmark_score,
        from_points,
        **kwargs,
    ):
        """
        Optionally runs a secondary, more detailed landmark detector on the detected faces
        to refine the keypoints.
        """
        kpss_5 = kpss.copy()
        if use_landmark_detection and len(kpss_5) > 0:
            # We need to filter kwargs to remove arguments that are already passed positionally
            # to run_detect_landmark to avoid "got multiple values for argument" error.
            # run_detect_landmark signature: (img, bbox, det_kpss, detect_mode, score, from_points, **kwargs)
            landmark_kwargs = kwargs.copy()
            for key in ["img", "score", "from_points"]:
                landmark_kwargs.pop(key, None)

            refined_kpss = []
            for i in range(kpss_5.shape[0]):
                landmark_kpss_5, landmark_kpss, landmark_scores = (
                    self.models_processor.run_detect_landmark(
                        img_landmark,
                        det[i],
                        kpss_5[i],
                        landmark_detect_mode,
                        landmark_score,
                        from_points,
                        **landmark_kwargs,
                    )
                )
                refined_kpss.append(
                    landmark_kpss if len(landmark_kpss) > 0 else kpss_5[i]
                )
                # If the new landmarks have a higher confidence, replace the old 5-point landmarks.
                if len(landmark_kpss_5) > 0 and (
                    len(landmark_scores) == 0
                    or np.mean(landmark_scores) > np.mean(score_values[i])
                ):
                    kpss_5[i] = landmark_kpss_5
            kpss = np.array(refined_kpss, dtype=object)
        return det, kpss_5, kpss, score_values

    def _get_det10g_runner(self):
        """Lazy-load the Det10gTorch Custom-kernel runner for RetinaFace."""
        if self._det10g_runner is not None:
            return self._det10g_runner
        self.models_processor.show_build_dialog.emit(
            "Finalizing Custom Provider",
            "Compiling & capturing CUDA graph for RetinaFace (det10g)…\nFirst run only — future sessions load instantly from cache.",
        )
        try:
            with self._custom_init_lock:
                if self._det10g_runner is not None:
                    return self._det10g_runner
                if self._det10g_torch is None:
                    try:
                        import pathlib
                        from custom_kernels.det_10g.det10g_torch import Det10gTorch

                        onnx_path = str(
                            pathlib.Path(__file__).parent.parent.parent
                            / "model_assets"
                            / "det_10g.onnx"
                        )
                        m = (
                            Det10gTorch.from_onnx(onnx_path)
                            .to(self.models_processor.device)
                            .eval()
                        )
                        self._det10g_torch = m
                    except Exception as e:
                        print(f"[Custom] det_10g load failed: {e}")
                        return None
                try:
                    from custom_kernels.det_10g.det10g_torch import (
                        build_cuda_graph_runner,
                    )

                    with self.models_processor.cuda_graph_capture_lock:
                        self._det10g_runner = build_cuda_graph_runner(
                            self._det10g_torch, torch_compile=True
                        )
                except Exception as e:
                    print(f"[Custom] det_10g graph runner failed, using eager: {e}")
                    self._det10g_runner = self._det10g_torch
        finally:
            self.models_processor.hide_build_dialog.emit()
        return self._det10g_runner

    def _get_yolo_runner(self):
        """Lazy-load the YoloFace8nTorch Custom-kernel runner for YOLOv8 face detector."""
        if self._yolo_runner is not None:
            return self._yolo_runner
        self.models_processor.show_build_dialog.emit(
            "Finalizing Custom Provider",
            "Compiling & capturing CUDA graph for YOLOv8n face detector…\nFirst run only — future sessions load instantly from cache.",
        )
        try:
            with self._custom_init_lock:
                if self._yolo_runner is not None:
                    return self._yolo_runner
                if self._yolo_torch is None:
                    try:
                        import pathlib
                        from custom_kernels.yoloface_8n.yoloface8n_torch import (
                            YoloFace8nTorch,
                        )

                        onnx_path = str(
                            pathlib.Path(__file__).parent.parent.parent
                            / "model_assets"
                            / "yoloface_8n.onnx"
                        )
                        m = (
                            YoloFace8nTorch.from_onnx(onnx_path)
                            .to(self.models_processor.device)
                            .eval()
                        )
                        self._yolo_torch = m
                    except Exception as e:
                        print(f"[Custom] yoloface_8n load failed: {e}")
                        return None
                try:
                    from custom_kernels.yoloface_8n.yoloface8n_torch import (
                        build_cuda_graph_runner,
                    )

                    with self.models_processor.cuda_graph_capture_lock:
                        self._yolo_runner = build_cuda_graph_runner(
                            self._yolo_torch, torch_compile=True
                        )
                except Exception as e:
                    print(f"[Custom] yoloface_8n graph runner failed, using eager: {e}")
                    self._yolo_runner = self._yolo_torch
        finally:
            self.models_processor.hide_build_dialog.emit()
        return self._yolo_runner

    def _run_model_with_lazy_build_check(
        self, model_name: str, ort_session, io_binding
    ) -> list:
        """
        Runs the ONNX session with IOBinding, handling TensorRT lazy build dialogs.
        This centralizes the try/finally logic for showing/hiding the build progress dialog
        and includes the critical CUDA synchronization steps (Pre and Post inference).

        Args:
            model_name (str): The name of the model being run.
            ort_session: The ONNX Runtime session instance.
            io_binding: The pre-configured IOBinding object.

        Returns:
            list: The network outputs from copy_outputs_to_cpu().
        """
        is_lazy_build = self.models_processor.check_and_clear_pending_build(model_name)
        if is_lazy_build:
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

            ort_session.run_with_iobinding(io_binding)

            # POST-INFERENCE SYNC : Ensure the GPU has completed all
            # calculations before ONNX Runtime attempts to copy the result back to CPU RAM.
            # Without this, copy_outputs_to_cpu() might grab an incomplete tensor.
            if self.models_processor.device == "cuda":
                torch.cuda.current_stream().synchronize()
            elif self.models_processor.device != "cpu":
                self.models_processor.syncvec.cpu()

            net_outs = io_binding.copy_outputs_to_cpu()

        finally:
            if is_lazy_build:
                self.models_processor.hide_build_dialog.emit()

        return net_outs

    def track_faces(self, img, previous_detections, **kwargs):
        """
        Attempts to track faces based on their previous positions using landmark detection directly.
        Returns: (det, kpss, scores) or (None, None, None) if tracking failed for any face.
        """
        if not previous_detections:
            return None, None, None

        tracked_det = []
        tracked_kpss = []
        tracked_scores = []

        img_height, img_width = img.shape[1], img.shape[2]

        # Parameters for tracking
        landmark_score_threshold = kwargs.get("landmark_score", 0.5)
        detect_mode = kwargs.get("landmark_detect_mode", "203")
        use_mean_eyes = kwargs.get("use_mean_eyes", False)

        for prev_face in previous_detections:
            # Previous bounding box
            bbox = prev_face["bbox"]

            # Expand the box slightly to account for movement
            expansion_factor = 0.2  # 20% expansion
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]

            expanded_bbox = np.array(
                [
                    max(0, bbox[0] - w * expansion_factor),
                    max(0, bbox[1] - h * expansion_factor),
                    min(img_width, bbox[2] + w * expansion_factor),
                    min(img_height, bbox[3] + h * expansion_factor),
                ]
            )

            # Run landmark detection directly on the expanded previous area
            # We assume kpss_5 is enough to verify presence
            kpss_5, kpss_all, scores = self.models_processor.run_detect_landmark(
                img,
                expanded_bbox,
                None,  # No initial keypoints known for this frame yet
                detect_mode=detect_mode,
                score=landmark_score_threshold,
                from_points=False,  # Must be False here as we only have a box
                use_mean_eyes=use_mean_eyes,
            )

            # Verification: If no landmarks found, tracking failed -> Full Redetect needed
            if len(kpss_5) == 0:
                return None, None, None

            # Determine which keypoints to use for bbox recalculation
            # Use dense landmarks if available (more precise), otherwise 5 points
            current_kpss = (
                kpss_all if (kpss_all is not None and len(kpss_all) > 0) else kpss_5
            )

            # Recalculate Bounding Box from the new landmarks
            # This allows the box to "move" and follow the face
            if current_kpss is not None and len(current_kpss) > 0:
                min_x, min_y = np.min(current_kpss, axis=0)
                max_x, max_y = np.max(current_kpss, axis=0)

                # Add a little padding to the new box so it doesn't shrink over time
                pad_w = (max_x - min_x) * 0.1
                pad_h = (max_y - min_y) * 0.1

                new_bbox = np.array(
                    [
                        max(0, min_x - pad_w),
                        max(0, min_y - pad_h),
                        min(img_width, max_x + pad_w),
                        min(img_height, max_y + pad_h),
                    ]
                )

                # Append results
                tracked_det.append(new_bbox)
                tracked_kpss.append(
                    kpss_5
                )  # We keep the 5-points format for consistency
                # BT-14: use a conservative fallback score (0.5) rather than 0.99,
                # so secondary landmark models can improve the result when confidence is uncertain
                tracked_scores.append(prev_face.get("score", 0.5))
            else:
                return None, None, None

        return (
            np.array(tracked_det, dtype=np.float32),
            np.array(tracked_kpss, dtype=np.float32),
            np.array(tracked_scores, dtype=np.float32),
        )

    def run_detect(
        self,
        img,
        detect_mode="RetinaFace",
        max_num=1,
        score=0.5,
        input_size=(512, 512),
        use_landmark_detection=False,
        landmark_detect_mode="203",
        landmark_score=0.5,
        from_points=False,
        rotation_angles=None,
        previous_detections=None,
        bypass_bytetrack=False,
        **kwargs,
    ):
        """
        Main dispatcher for running face detection. Selects and runs the appropriate model.
        Supports tracking via 'previous_detections'.
        """
        control = self.models_processor.main_window.control
        use_bytetrack = control.get("FaceTrackingEnableToggle", False)
        # bypass_bytetrack=True disables ByteTrack for this call only (e.g. per-eye VR
        # detection where the half-width coordinate space would corrupt tracker state)
        if bypass_bytetrack:
            use_bytetrack = False

        # ByteTrack skip-frame shortcut: when ByteTrack is active, the tracker has been
        # initialised, AND this is a detection-interval skip frame (indicated by non-empty
        # previous_detections), advance the Kalman filter with empty detections rather than
        # running the full detector.  This makes FaceDetectionIntervalSlider effective even
        # when ByteTrack is enabled (previously the slider was silently ignored).
        if (
            use_bytetrack
            and BYTETracker is not None
            and previous_detections is not None
            and len(previous_detections) > 0
        ):
            with self._tracker_lock:
                if self.tracker is not None:
                    img_hw = (int(img.shape[1]), int(img.shape[2]))
                    online_targets = self.tracker.update(
                        np.empty((0, 5)), img_hw, img_hw
                    )
                else:
                    online_targets = []

            # Build return arrays from coasted tracks (Kalman-predicted positions
            # with last known landmarks) — same logic as the main ByteTrack path below
            tracked_det: list = []
            tracked_kpss_5: list = []
            tracked_kpss_all: list = []
            tracked_scores: list = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                t_bbox = np.array(
                    [tlwh[0], tlwh[1], tlwh[0] + tlwh[2], tlwh[1] + tlwh[3]]
                )
                with self._track_history_lock:
                    hist = self.track_history.get(tid)
                if hist is not None and hist.get("kps") is not None:
                    tracked_det.append(t_bbox)
                    tracked_kpss_5.append(hist["kps"])
                    tracked_kpss_all.append(hist["kps"])
                    tracked_scores.append(hist["cum_score"])

            if tracked_det:
                return (
                    np.array(tracked_det, dtype=np.float32),
                    np.array(tracked_kpss_5, dtype=np.float32),
                    np.array(tracked_kpss_all, dtype=object),
                )
            # Tracker had no active tracks (e.g. first ever frame) → fall through to full detection
            # so the user sees faces immediately rather than waiting one extra frame.

        # TRACKING ATTEMPT (Simple Fallback)
        # If we have previous faces and tracking is requested (via implicit logic or kwargs)
        # We skip this if ByteTrack is enabled to prioritize the advanced tracker
        if (
            not use_bytetrack
            and previous_detections is not None
            and len(previous_detections) > 0
        ):
            # Try to track
            t_det, t_kpss, t_scores = self.track_faces(
                img,
                previous_detections,
                landmark_score=landmark_score,
                landmark_detect_mode=landmark_detect_mode,
                **kwargs,
            )

            # If tracking succeeded (returns are not None), skip heavy detection
            if t_det is not None:
                # Optionally refine landmarks (if the user wants detailed landmarks)
                if use_landmark_detection:
                    det_r, kpss_5_r, kpss_r, _ = self._refine_landmarks(
                        img,
                        t_det,
                        t_kpss,
                        t_scores,
                        use_landmark_detection,
                        landmark_detect_mode,
                        landmark_score,
                        from_points,
                        **kwargs,
                    )
                    return det_r, kpss_5_r, kpss_r
                return t_det, t_kpss, t_kpss

        # FULL DETECTION FALLBACK
        # If no previous detections or tracking failed, run the heavy model
        detector = self.detector_map.get(detect_mode)
        if not detector:
            return np.empty((0, 4)), np.empty((0, 5, 2)), np.empty((0, 5, 2))

        model_name = detector["model_name"]
        # FD-RACE-01: serialise model switch so concurrent workers don't interleave unload/load.
        with self._ort_model_lock:
            if (
                self.current_detector_model
                and self.current_detector_model != model_name
            ):
                self.models_processor.unload_model(self.current_detector_model)
            self.current_detector_model = model_name
        ort_session = self.models_processor.load_model(model_name)
        if not ort_session:
            print(
                f"[ERROR] {model_name} model failed to load or is not available. Skipping detection."
            )
            return np.empty((0, 4)), np.empty((0, 5, 2)), np.empty((0, 5, 2))

        detection_function = detector["function"]

        # BT-05: when ByteTrack is enabled use a recall-optimised threshold that
        # respects the user's DetectorScoreSlider (capped at 0.5 so low-confidence
        # detections still reach the tracker's second-association pass).
        # Previously this was always hardcoded to 0.3, ignoring user intent and
        # falling below ByteTracker's own track_thresh (0.4), causing excess FP tracks.
        if use_bytetrack:
            user_score = control.get("DetectorScoreSlider", 50) / 100.0
            effective_score = min(user_score, 0.5)
        else:
            effective_score = score

        args = {
            "img": img,
            "max_num": max_num,
            "score": effective_score,
            "use_landmark_detection": use_landmark_detection,
            "landmark_detect_mode": landmark_detect_mode,
            "landmark_score": landmark_score,
            "from_points": from_points,
            "rotation_angles": rotation_angles or [0],
            "ort_session": ort_session,
        }
        args.update(kwargs)

        if detect_mode in ["RetinaFace", "SCRFD"]:
            args["input_size"] = input_size

        # Run the detector — returns (det, kpss_5, kpss, det_scores)
        det, kpss_5, kpss, det_scores = detection_function(**args)

        # BT-02: guarantee det_scores is always a 1-D array so downstream boolean
        # indexing and math operations behave consistently (some detectors return (N,1))
        if hasattr(det_scores, "flatten"):
            det_scores = det_scores.flatten()

        # Initialize score_values so it is always bound before the ByteTrack block
        # (det_scores from detection_function may not be assigned in all paths).
        score_values = np.array([])

        # ByteTrack Advanced Tracking
        if use_bytetrack and BYTETracker is not None:
            # BT-06/BT-07: serialize all tracker access under a dedicated lock to
            # prevent concurrent pool workers from corrupting Kalman filter state
            with self._tracker_lock:
                if self.tracker is None:
                    self.tracker = BYTETracker(_BYTETRACK_ARGS(control))

                # Prepare detections for ByteTrack [x1, y1, x2, y2, score]
                img_hw = (int(img.shape[1]), int(img.shape[2]))
                # BT-13: measure active tracks BEFORE update so we have the correct
                # pre-update baseline for scene-cut detection (measuring after update
                # inflates the count by newly confirmed tracks, making cuts harder to detect)
                active_before = len(self.tracker.tracked_stracks)
                if len(det) > 0:
                    # Use actual detection scores; fall back to 0.9 if lengths mismatch
                    scores_for_tracker = (
                        det_scores
                        if len(det_scores) == len(det)
                        else np.full(len(det), 0.9)
                    )
                    tracker_input = np.column_stack([det, scores_for_tracker])
                    online_targets = self.tracker.update(tracker_input, img_hw, img_hw)
                else:
                    online_targets = self.tracker.update(
                        np.empty((0, 5)), img_hw, img_hw
                    )

                # BT-13: scene-cut detection — if fewer than 30% of active tracks matched,
                # the scene has likely changed; reset the tracker to avoid stale Kalman state
                matched_count = len(online_targets)
                if active_before > 0 and matched_count / active_before < 0.3:
                    # print("[ByteTrack] Scene cut detected — resetting tracker")
                    self.tracker = (
                        None  # will be re-created with frame_id=0 on next call
                    )
                    with self._track_history_lock:
                        self.track_history.clear()
                    online_targets = []

            tracked_det = []
            tracked_kpss_5 = []
            tracked_kpss_all = []
            tracked_scores = []

            current_frame_num = getattr(
                self.models_processor, "current_frame_number", 0
            )

            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                # Convert back to [x1, y1, x2, y2]
                t_bbox = np.array(
                    [tlwh[0], tlwh[1], tlwh[0] + tlwh[2], tlwh[1] + tlwh[3]]
                )

                # Match landmarks from current detections back to the tracked object via IoU
                best_iou = 0
                match_idx = -1
                for i, d in enumerate(det):
                    iou = self._calculate_iou(t_bbox, d)
                    if iou > best_iou:
                        best_iou = iou
                        match_idx = i

                if match_idx != -1:
                    # Update Cumulative Similarity Score for UI stability
                    with self._track_history_lock:
                        if tid in self.track_history:
                            prev_score = self.track_history[tid]["cum_score"]
                            cum_score = (
                                self.lambda_s * t.score
                                + (1 - self.lambda_s) * prev_score
                            )
                        else:
                            cum_score = t.score
                        # BT-08: record last_seen so stale entries can be evicted
                        self.track_history[tid] = {
                            "cum_score": cum_score,
                            "last_seen": current_frame_num,
                            "kps": kpss_5[match_idx],
                        }

                    tracked_det.append(t_bbox)
                    tracked_kpss_5.append(kpss_5[match_idx])
                    tracked_kpss_all.append(kpss[match_idx])
                    tracked_scores.append(cum_score)
                else:
                    # BT-04: coasted track — no matching raw detection; use Kalman-predicted
                    # position with last known landmarks so brief occlusions are handled
                    with self._track_history_lock:
                        hist = self.track_history.get(tid)
                    if hist is not None and hist.get("kps") is not None:
                        last_kps = hist["kps"]
                        tracked_det.append(t_bbox)
                        tracked_kpss_5.append(last_kps)
                        # No dense landmarks available for coasted tracks; reuse 5-pt kps
                        tracked_kpss_all.append(last_kps)
                        tracked_scores.append(hist["cum_score"])

            # BT-08: evict stale track_history entries (last seen > track_buffer frames ago)
            track_buffer = int(control.get("ByteTrackTrackBufferSlider", 30))
            with self._track_history_lock:
                stale_ids = [
                    tid
                    for tid, data in self.track_history.items()
                    if current_frame_num - data.get("last_seen", 0) > track_buffer
                ]
                for tid in stale_ids:
                    del self.track_history[tid]

            # Ensure numerical types for math operations
            if tracked_det:
                det = np.array(tracked_det, dtype=np.float32)
                kpss_5 = np.array(tracked_kpss_5, dtype=np.float32)
                # Dense landmarks 'kpss' can remain object if shapes vary,
                # but refine_landmarks logic handles it
                kpss = np.array(tracked_kpss_all, dtype=object)
                score_values = np.array(tracked_scores, dtype=np.float32).flatten()
            else:
                # BT-12 / BT-14: no confirmed tracks yet (first frame, scene cut, or
                # tracker reset after seek).  Fall back to raw detector output so that
                # swap/edit is applied immediately instead of waiting for ByteTrack to
                # confirm tracks on a subsequent frame.
                if len(det) > 0:
                    score_values = (
                        det_scores
                        if len(det_scores) == len(det)
                        else np.full(len(det), 0.9, dtype=np.float32)
                    )
                    # det, kpss_5, kpss already hold the raw detector output — keep them
                else:
                    det = np.empty((0, 4), dtype=np.float32)
                    kpss_5 = np.empty((0, 5, 2), dtype=np.float32)
                    kpss = np.empty((0,), dtype=object)
                    score_values = np.empty((0,), dtype=np.float32)

        # Optionally refine landmarks (if the user wants detailed landmarks)
        bytetrack_active = use_bytetrack and BYTETracker is not None
        if use_landmark_detection and len(det) > 0:
            # Use tracked scores when ByteTrack is active, otherwise actual detector scores
            current_scores = score_values if bytetrack_active else det_scores

            det_r, kpss_5_r, kpss_r, _ = self._refine_landmarks(
                img,
                det,
                kpss_5,
                current_scores,
                use_landmark_detection,
                landmark_detect_mode,
                landmark_score,
                from_points,
                **kwargs,
            )
            return det_r, kpss_5_r, kpss_r

        return det, kpss_5, kpss

    def reset_tracker(self):
        """Reset the ByteTracker and its history (call on video seek or toggle to discard stale state)."""
        with self._tracker_lock:
            self.tracker = None
        with self._track_history_lock:
            self.track_history.clear()
        # Reset the global track-ID counter so new sessions start from ID 1
        try:
            from app.processors.external.yolox.tracker.basetrack import BaseTrack

            BaseTrack._count = 0
        except ImportError:
            pass

    def _calculate_iou(self, boxA, boxB):
        """
        Computes the Intersection-over-Union (IoU) between two axis-aligned bounding boxes.

        Args:
            boxA: Sequence [x1, y1, x2, y2] for the first box.
            boxB: Sequence [x1, y1, x2, y2] for the second box.

        Returns:
            float: IoU in [0, 1]. Returns 0.0 when either box has zero area
                   (BT-01: guards against degenerate Kalman predictions).
        """
        # BT-01: guard against division by zero when Kalman prediction produces
        # a degenerate (zero-area) bounding box
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        denominator = float(boxAArea + boxBArea - interArea)
        if denominator <= 0:
            return 0.0
        return interArea / denominator

    def detect_retinaface(self, **kwargs):
        """Runs the RetinaFace detection pipeline."""
        model_name = "RetinaFace"
        ort_session = kwargs.get("ort_session")

        img, input_size, score, rotation_angles = (
            kwargs.get("img"),
            kwargs.get("input_size"),
            kwargs.get("score"),
            kwargs.get("rotation_angles"),
        )
        img_landmark = img.clone() if kwargs.get("use_landmark_detection") else None

        det_img, det_scale, final_input_size = self._prepare_detection_image(
            img, input_size, "retinaface"
        )

        scores_list, bboxes_list, kpss_list = [], [], []
        cx, cy = final_input_size[0] / 2, final_input_size[1] / 2
        do_rotation = len(rotation_angles) > 1

        for angle in rotation_angles:
            if angle != 0:
                aimg, M = faceutil.transform(
                    det_img, (cx, cy), max(final_input_size), 1.0, angle
                )
                IM = faceutil.invertAffineTransform(M)
                aimg = torch.unsqueeze(aimg, 0).contiguous()
            else:
                IM, aimg = None, torch.unsqueeze(det_img, 0).contiguous()

            if self.models_processor.provider_name == "Custom":
                runner = self._get_det10g_runner()
                if runner is not None:
                    with torch.no_grad():
                        with self._get_runner_lock(runner):
                            pt_outs = runner(aimg)
                            if self.models_processor.device == "cuda":
                                torch.cuda.current_stream().synchronize()
                    net_outs = [t.cpu().numpy() for t in pt_outs]
                else:
                    io_binding = ort_session.io_binding()
                    io_binding.bind_input(
                        name="input.1",
                        device_type=self.models_processor.device,
                        device_id=0,
                        element_type=np.float32,
                        shape=aimg.size(),
                        buffer_ptr=aimg.data_ptr(),
                    )
                    for i in [
                        "448",
                        "471",
                        "494",
                        "451",
                        "474",
                        "497",
                        "454",
                        "477",
                        "500",
                    ]:
                        io_binding.bind_output(i, self.models_processor.device)
                    net_outs = self._run_model_with_lazy_build_check(
                        model_name, ort_session, io_binding
                    )
            else:
                io_binding = ort_session.io_binding()
                io_binding.bind_input(
                    name="input.1",
                    device_type=self.models_processor.device,
                    device_id=0,
                    element_type=np.float32,
                    shape=aimg.size(),
                    buffer_ptr=aimg.data_ptr(),
                )
                for i in [
                    "448",
                    "471",
                    "494",
                    "451",
                    "474",
                    "497",
                    "454",
                    "477",
                    "500",
                ]:
                    io_binding.bind_output(i, self.models_processor.device)
                # Run the model with lazy build handling
                net_outs = self._run_model_with_lazy_build_check(
                    model_name, ort_session, io_binding
                )

            input_height, input_width = aimg.shape[2], aimg.shape[3]
            fmc = 3
            # Process outputs from each feature map stride (8, 16, 32)
            for idx, stride in enumerate([8, 16, 32]):
                # Get scores, bbox predictions, and keypoint predictions
                scores, bbox_preds, kps_preds = (
                    net_outs[idx],
                    net_outs[idx + fmc] * stride,
                    net_outs[idx + fmc * 2] * stride,
                )
                height, width = input_height // stride, input_width // stride

                # Generate anchor centers (cache them for efficiency)
                key = (height, width, stride)
                if key in self.center_cache:
                    anchor_centers = self.center_cache[key]
                else:
                    anchor_centers = np.stack(
                        np.mgrid[:height, :width][::-1], axis=-1
                    ).astype(np.float32)
                    anchor_centers = (anchor_centers * stride).reshape((-1, 2))
                    anchor_centers = np.stack([anchor_centers] * 2, axis=1).reshape(
                        (-1, 2)
                    )
                    if len(self.center_cache) < 100:
                        self.center_cache[key] = anchor_centers

                # Filter detections by score
                pos_inds = np.where(scores >= score)[0]

                # Calculate bounding boxes from anchor centers and predictions
                bboxes = np.stack(
                    [
                        anchor_centers[:, 0] - bbox_preds[:, 0],
                        anchor_centers[:, 1] - bbox_preds[:, 1],
                        anchor_centers[:, 0] + bbox_preds[:, 2],
                        anchor_centers[:, 1] + bbox_preds[:, 3],
                    ],
                    axis=-1,
                )
                pos_scores, pos_bboxes = scores[pos_inds], bboxes[pos_inds]

                # If rotated, transform bboxes back to original orientation
                if angle != 0 and len(pos_bboxes) > 0:
                    points1, points2 = (
                        faceutil.trans_points2d(pos_bboxes[:, :2], IM),
                        faceutil.trans_points2d(pos_bboxes[:, 2:], IM),
                    )
                    _x1, _y1, _x2, _y2 = (
                        points1[:, 0],
                        points1[:, 1],
                        points2[:, 0],
                        points2[:, 1],
                    )
                    if angle in (-270, 90):
                        points1, points2 = (
                            np.stack((_x1, _y2), axis=1),
                            np.stack((_x2, _y1), axis=1),
                        )
                    elif angle in (-180, 180):
                        points1, points2 = (
                            np.stack((_x2, _y2), axis=1),
                            np.stack((_x1, _y1), axis=1),
                        )
                    elif angle in (-90, 270):
                        points1, points2 = (
                            np.stack((_x2, _y1), axis=1),
                            np.stack((_x1, _y2), axis=1),
                        )
                    pos_bboxes = np.hstack((points1, points2))

                # Calculate keypoints from anchor centers and predictions
                preds = [
                    item
                    for i in range(0, kps_preds.shape[1], 2)
                    for item in (
                        anchor_centers[:, i % 2] + kps_preds[:, i],
                        anchor_centers[:, i % 2 + 1] + kps_preds[:, i + 1],
                    )
                ]
                kpss = np.stack(preds, axis=-1).reshape((-1, 5, 2))
                pos_kpss = kpss[pos_inds]

                # Handle multi-rotation logic (filtering by face orientation)
                if do_rotation:
                    for i in range(len(pos_kpss)):
                        face_size = max(
                            pos_bboxes[i][2] - pos_bboxes[i][0],
                            pos_bboxes[i][3] - pos_bboxes[i][1],
                        )
                        angle_deg_to_front = faceutil.get_face_orientation(
                            face_size, pos_kpss[i]
                        )
                        # Discard faces that are not facing front
                        if abs(angle_deg_to_front) > 50.00:
                            pos_scores[i] = 0.0
                        if angle != 0:
                            pos_kpss[i] = faceutil.trans_points2d(pos_kpss[i], IM)
                    # Re-filter based on the new scores
                    pos_inds = np.where(pos_scores >= score)[0]
                    pos_scores, pos_bboxes, pos_kpss = (
                        pos_scores[pos_inds],
                        pos_bboxes[pos_inds],
                        pos_kpss[pos_inds],
                    )
                kpss_list.append(pos_kpss)
                bboxes_list.append(pos_bboxes)
                scores_list.append(pos_scores)

        # Filter all collected detections (from all angles/strides) using GPU NMS
        det, kpss, score_values = self._filter_detections_gpu(
            scores_list,
            bboxes_list,
            kpss_list,
            img.shape[1],
            img.shape[2],
            det_scale,
            kwargs.get("max_num"),
        )
        if det is None:
            return (
                np.empty((0, 4)),
                np.empty((0, 5, 2)),
                np.empty((0, 5, 2)),
                np.empty((0,)),
            )

        # Optionally refine landmarks with a secondary model
        return self._refine_landmarks(img_landmark, det, kpss, score_values, **kwargs)

    def detect_scrdf(self, **kwargs):
        """Runs the SCRFD detection pipeline."""
        model_name = "SCRFD2.5g"
        ort_session = kwargs.get("ort_session")

        img, input_size, score, rotation_angles = (
            kwargs.get("img"),
            kwargs.get("input_size"),
            kwargs.get("score"),
            kwargs.get("rotation_angles"),
        )
        img_landmark = img.clone() if kwargs.get("use_landmark_detection") else None

        det_img, det_scale, final_input_size = self._prepare_detection_image(
            img, input_size, "scrfd"
        )

        scores_list, bboxes_list, kpss_list = [], [], []
        cx, cy = final_input_size[0] / 2, final_input_size[1] / 2
        do_rotation = len(rotation_angles) > 1
        input_name = ort_session.get_inputs()[0].name
        output_names = [o.name for o in ort_session.get_outputs()]

        for angle in rotation_angles:
            if angle != 0:
                aimg, M = faceutil.transform(
                    det_img, (cx, cy), max(final_input_size), 1.0, angle
                )
                IM = faceutil.invertAffineTransform(M)
                aimg = torch.unsqueeze(aimg, 0).contiguous()
            else:
                IM, aimg = None, torch.unsqueeze(det_img, 0).contiguous()

            io_binding = ort_session.io_binding()

            io_binding.bind_input(
                name=input_name,
                device_type=self.models_processor.device,
                device_id=0,
                element_type=np.float32,
                shape=aimg.size(),
                buffer_ptr=aimg.data_ptr(),
            )
            for name in output_names:
                io_binding.bind_output(name, self.models_processor.device)

            # Run the model with lazy build handling
            net_outs = self._run_model_with_lazy_build_check(
                model_name, ort_session, io_binding
            )
            input_height, input_width = aimg.shape[2], aimg.shape[3]
            fmc = 3
            # Process outputs from each feature map stride (8, 16, 32)
            for idx, stride in enumerate([8, 16, 32]):
                # Get scores, bbox predictions, and keypoint predictions
                scores, bbox_preds, kps_preds = (
                    net_outs[idx],
                    net_outs[idx + fmc] * stride,
                    net_outs[idx + fmc * 2] * stride,
                )
                height, width = input_height // stride, input_width // stride

                # Generate anchor centers (cache them for efficiency)
                key = (height, width, stride)
                if key in self.center_cache:
                    anchor_centers = self.center_cache[key]
                else:
                    anchor_centers = np.stack(
                        np.mgrid[:height, :width][::-1], axis=-1
                    ).astype(np.float32)
                    anchor_centers = (anchor_centers * stride).reshape((-1, 2))
                    anchor_centers = np.stack([anchor_centers] * 2, axis=1).reshape(
                        (-1, 2)
                    )
                    if len(self.center_cache) < 100:
                        self.center_cache[key] = anchor_centers

                # Filter detections by score
                pos_inds = np.where(scores >= score)[0]

                # Calculate bounding boxes from anchor centers and predictions
                bboxes = np.stack(
                    [
                        anchor_centers[:, 0] - bbox_preds[:, 0],
                        anchor_centers[:, 1] - bbox_preds[:, 1],
                        anchor_centers[:, 0] + bbox_preds[:, 2],
                        anchor_centers[:, 1] + bbox_preds[:, 3],
                    ],
                    axis=-1,
                )
                pos_scores, pos_bboxes = scores[pos_inds], bboxes[pos_inds]

                # If rotated, transform bboxes back to original orientation
                if angle != 0 and len(pos_bboxes) > 0:
                    points1, points2 = (
                        faceutil.trans_points2d(pos_bboxes[:, :2], IM),
                        faceutil.trans_points2d(pos_bboxes[:, 2:], IM),
                    )
                    _x1, _y1, _x2, _y2 = (
                        points1[:, 0],
                        points1[:, 1],
                        points2[:, 0],
                        points2[:, 1],
                    )
                    if angle in (-270, 90):
                        points1, points2 = (
                            np.stack((_x1, _y2), axis=1),
                            np.stack((_x2, _y1), axis=1),
                        )
                    elif angle in (-180, 180):
                        points1, points2 = (
                            np.stack((_x2, _y2), axis=1),
                            np.stack((_x1, _y1), axis=1),
                        )
                    elif angle in (-90, 270):
                        points1, points2 = (
                            np.stack((_x2, _y1), axis=1),
                            np.stack((_x1, _y2), axis=1),
                        )
                    pos_bboxes = np.hstack((points1, points2))

                # Calculate keypoints from anchor centers and predictions
                preds = [
                    item
                    for i in range(0, kps_preds.shape[1], 2)
                    for item in (
                        anchor_centers[:, i % 2] + kps_preds[:, i],
                        anchor_centers[:, i % 2 + 1] + kps_preds[:, i + 1],
                    )
                ]
                kpss = np.stack(preds, axis=-1).reshape((-1, 5, 2))
                pos_kpss = kpss[pos_inds]

                # Handle multi-rotation logic (filtering by face orientation)
                if do_rotation:
                    for i in range(len(pos_kpss)):
                        face_size = max(
                            pos_bboxes[i][2] - pos_bboxes[i][0],
                            pos_bboxes[i][3] - pos_bboxes[i][1],
                        )
                        angle_deg_to_front = faceutil.get_face_orientation(
                            face_size, pos_kpss[i]
                        )
                        # Discard faces that are not facing front
                        if abs(angle_deg_to_front) > 50.00:
                            pos_scores[i] = 0.0
                        if angle != 0:
                            pos_kpss[i] = faceutil.trans_points2d(pos_kpss[i], IM)
                    # Re-filter based on the new scores
                    pos_inds = np.where(pos_scores >= score)[0]
                    pos_scores, pos_bboxes, pos_kpss = (
                        pos_scores[pos_inds],
                        pos_bboxes[pos_inds],
                        pos_kpss[pos_inds],
                    )
                kpss_list.append(pos_kpss)
                bboxes_list.append(pos_bboxes)
                scores_list.append(pos_scores)

        # Filter all collected detections (from all angles/strides) using GPU NMS
        det, kpss, score_values = self._filter_detections_gpu(
            scores_list,
            bboxes_list,
            kpss_list,
            img.shape[1],
            img.shape[2],
            det_scale,
            kwargs.get("max_num"),
        )
        if det is None:
            return (
                np.empty((0, 4)),
                np.empty((0, 5, 2)),
                np.empty((0, 5, 2)),
                np.empty((0,)),
            )

        # Optionally refine landmarks with a secondary model
        return self._refine_landmarks(img_landmark, det, kpss, score_values, **kwargs)

    def detect_yoloface(self, **kwargs):
        """Runs the Yolov8-face detection pipeline."""
        model_name = "YoloFace8n"
        ort_session = kwargs.get("ort_session")

        img, score, rotation_angles = (
            kwargs.get("img"),
            kwargs.get("score"),
            kwargs.get("rotation_angles"),
        )
        img_landmark = img.clone() if kwargs.get("use_landmark_detection") else None

        input_size = (640, 640)
        # _prepare_detection_image returns uint8 CHW tensor for yolo mode
        det_img, det_scale, final_input_size = self._prepare_detection_image(
            img, input_size, "yolo"
        )

        scores_list, bboxes_list, kpss_list = [], [], []
        cx, cy = final_input_size[0] / 2, final_input_size[1] / 2
        do_rotation = len(rotation_angles) > 1

        for angle in rotation_angles:
            if angle != 0:
                aimg, M = faceutil.transform(
                    det_img, (cx, cy), 640, 1.0, angle
                )  # Rotates uint8
                IM = faceutil.invertAffineTransform(M)
            else:
                IM, aimg = None, det_img  # aimg is uint8 CHW

            # Note: Convert to float and normalize AFTER rotation, before binding
            aimg_prepared = aimg.to(torch.float32) / 255.0
            aimg_prepared = torch.unsqueeze(
                aimg_prepared, 0
            ).contiguous()  # Add batch dim

            if self.models_processor.provider_name == "Custom":
                runner = self._get_yolo_runner()
                if runner is not None:
                    with torch.no_grad():
                        with self._get_runner_lock(runner):
                            _yolo_out = runner(aimg_prepared)
                            if self.models_processor.device == "cuda":
                                torch.cuda.current_stream().synchronize()
                    net_outs = _yolo_out.cpu().numpy()
                else:
                    io_binding = ort_session.io_binding()
                    io_binding.bind_input(
                        name="images",
                        device_type=self.models_processor.device,
                        device_id=0,
                        element_type=np.float32,
                        shape=aimg_prepared.size(),
                        buffer_ptr=aimg_prepared.data_ptr(),
                    )
                    io_binding.bind_output("output0", self.models_processor.device)
                    net_outs = self._run_model_with_lazy_build_check(
                        model_name, ort_session, io_binding
                    )
            else:
                io_binding = ort_session.io_binding()
                io_binding.bind_input(
                    name="images",
                    device_type=self.models_processor.device,
                    device_id=0,
                    element_type=np.float32,
                    shape=aimg_prepared.size(),  # Use shape of prepared tensor
                    buffer_ptr=aimg_prepared.data_ptr(),  # Use data_ptr of prepared tensor
                )
                io_binding.bind_output("output0", self.models_processor.device)
                # Run the model with lazy build handling
                net_outs = self._run_model_with_lazy_build_check(
                    model_name, ort_session, io_binding
                )

            outputs = np.squeeze(net_outs).T
            bbox_raw, score_raw, kps_raw, *_ = np.split(outputs, [4, 5], axis=1)
            # Flatten score_raw before comparison
            score_raw_flat = score_raw.flatten()
            keep_indices = np.where(score_raw_flat > score)[0]

            if keep_indices.size > 0:  # Check size instead of any() for numpy arrays
                bbox_raw, kps_raw, score_raw = (
                    bbox_raw[keep_indices],
                    kps_raw[keep_indices],
                    score_raw[keep_indices],  # Keep score_raw as [N, 1] or similar
                )
                # Convert (center_x, center_y, w, h) to (x1, y1, x2, y2)
                bboxes_raw = np.stack(
                    (
                        bbox_raw[:, 0] - bbox_raw[:, 2] / 2,
                        bbox_raw[:, 1] - bbox_raw[:, 3] / 2,
                        bbox_raw[:, 0] + bbox_raw[:, 2] / 2,
                        bbox_raw[:, 1] + bbox_raw[:, 3] / 2,
                    ),
                    axis=-1,
                )
                if angle != 0 and len(bboxes_raw) > 0:
                    points1, points2 = (
                        faceutil.trans_points2d(bboxes_raw[:, :2], IM),
                        faceutil.trans_points2d(bboxes_raw[:, 2:], IM),
                    )
                    _x1, _y1, _x2, _y2 = (
                        points1[:, 0],
                        points1[:, 1],
                        points2[:, 0],
                        points2[:, 1],
                    )
                    if angle in (-270, 90):
                        points1, points2 = (
                            np.stack((_x1, _y2), axis=1),
                            np.stack((_x2, _y1), axis=1),
                        )
                    elif angle in (-180, 180):
                        points1, points2 = (
                            np.stack((_x2, _y2), axis=1),
                            np.stack((_x1, _y1), axis=1),
                        )
                    elif angle in (-90, 270):
                        points1, points2 = (
                            np.stack((_x2, _y1), axis=1),
                            np.stack((_x1, _y2), axis=1),
                        )
                    bboxes_raw = np.hstack((points1, points2))

                # Reshape keypoints
                kpss_raw = np.stack(
                    [
                        np.array([[kps[i], kps[i + 1]] for i in range(0, len(kps), 3)])
                        for kps in kps_raw
                    ]
                )

                # Handle multi-rotation logic (filtering by face orientation)
                if do_rotation:
                    score_raw_flat_filtered = (
                        score_raw.flatten()
                    )  # Flatten again after filtering
                    for i in range(len(kpss_raw)):
                        face_size = max(
                            bboxes_raw[i][2] - bboxes_raw[i][0],
                            bboxes_raw[i][3] - bboxes_raw[i][1],
                        )
                        angle_deg_to_front = faceutil.get_face_orientation(
                            face_size, kpss_raw[i]
                        )
                        if abs(angle_deg_to_front) > 50.00:
                            score_raw_flat_filtered[i] = (
                                0.0  # Modify the flattened copy
                            )
                        if angle != 0:
                            kpss_raw[i] = faceutil.trans_points2d(kpss_raw[i], IM)

                    # Filter again based on the modified scores
                    keep_indices_rot = np.where(score_raw_flat_filtered >= score)[0]
                    score_raw = score_raw[keep_indices_rot]
                    bboxes_raw = bboxes_raw[keep_indices_rot]
                    kpss_raw = kpss_raw[keep_indices_rot]

                # Ensure score_raw has the correct shape [N, 1] before appending
                if score_raw.ndim == 1:
                    score_raw = score_raw[:, np.newaxis]

                # Check if there are still detections after rotation filtering
                if score_raw.size > 0:
                    kpss_list.append(kpss_raw)
                    bboxes_list.append(bboxes_raw)
                    scores_list.append(score_raw)

        det, kpss, score_values = self._filter_detections_gpu(
            scores_list,
            bboxes_list,
            kpss_list,
            img.shape[1],
            img.shape[2],
            det_scale,
            kwargs.get("max_num"),
        )
        if det is None:
            return (
                np.empty((0, 4)),
                np.empty((0, 5, 2)),
                np.empty((0, 5, 2)),
                np.empty((0,)),
            )

        return self._refine_landmarks(img_landmark, det, kpss, score_values, **kwargs)

    def detect_yunet(self, **kwargs):
        """Runs the Yunet detection pipeline."""
        model_name = "YunetN"
        ort_session = kwargs.get("ort_session")

        img, score, rotation_angles = (
            kwargs.get("img"),
            kwargs.get("score"),
            kwargs.get("rotation_angles"),
        )
        img_landmark = img.clone() if kwargs.get("use_landmark_detection") else None

        input_size = (640, 640)
        # _prepare_detection_image returns uint8 CHW BGR tensor for yunet mode
        det_img, det_scale, final_input_size = self._prepare_detection_image(
            img, input_size, "yunet"
        )

        scores_list, bboxes_list, kpss_list = [], [], []
        cx, cy = final_input_size[0] / 2, final_input_size[1] / 2
        do_rotation = len(rotation_angles) > 1
        input_name = ort_session.get_inputs()[0].name
        output_names = [o.name for o in ort_session.get_outputs()]

        for angle in rotation_angles:
            if angle != 0:
                aimg, M = faceutil.transform(
                    det_img, (cx, cy), 640, 1.0, angle
                )  # Rotates uint8 BGR
                IM = faceutil.invertAffineTransform(M)
            else:
                IM, aimg = None, det_img  # aimg is uint8 CHW BGR

            # Note: Convert to float AFTER rotation, before binding
            aimg_prepared = aimg.to(dtype=torch.float32)
            aimg_prepared = torch.unsqueeze(
                aimg_prepared, 0
            ).contiguous()  # Add batch dim

            io_binding = ort_session.io_binding()

            io_binding.bind_input(
                name=input_name,
                device_type=self.models_processor.device,
                device_id=0,
                element_type=np.float32,
                shape=aimg_prepared.size(),  # Use shape of prepared tensor
                buffer_ptr=aimg_prepared.data_ptr(),  # Use data_ptr of prepared tensor
            )
            for name in output_names:
                io_binding.bind_output(name, self.models_processor.device)

            # Run the model with lazy build handling
            net_outs = self._run_model_with_lazy_build_check(
                model_name, ort_session, io_binding
            )
            strides = [8, 16, 32]
            for idx, stride in enumerate(strides):
                # Get predictions from the current stride
                cls_pred, obj_pred, reg_pred, kps_pred = (
                    net_outs[idx].reshape(-1, 1),
                    net_outs[idx + len(strides)].reshape(-1, 1),
                    net_outs[idx + len(strides) * 2].reshape(-1, 4),
                    net_outs[idx + len(strides) * 3].reshape(-1, 5 * 2),
                )

                # Generate/retrieve anchor centers
                key = (tuple(final_input_size), stride)
                if key in self.center_cache:
                    anchor_centers = self.center_cache[key]
                else:
                    anchor_centers = np.stack(
                        np.mgrid[
                            : (final_input_size[1] // stride),
                            : (final_input_size[0] // stride),
                        ][::-1],
                        axis=-1,
                    )
                    anchor_centers = (
                        (anchor_centers * stride).astype(np.float32).reshape(-1, 2)
                    )
                    if len(self.center_cache) < 100:  # Added limit to cache size
                        self.center_cache[key] = anchor_centers

                scores_val = cls_pred * obj_pred
                # Flatten scores_val before comparison
                scores_val_flat = scores_val.flatten()
                pos_inds = np.where(scores_val_flat >= score)[0]

                # Ensure pos_inds is not empty before proceeding
                if pos_inds.size == 0:
                    continue

                # Calculate bboxes for positive detections
                bbox_cxy = (
                    reg_pred[pos_inds, :2] * stride + anchor_centers[pos_inds, :]
                )  # Filter anchor_centers too
                bbox_wh = np.exp(np.clip(reg_pred[pos_inds, 2:], -10.0, 10.0)) * stride

                bboxes = np.stack(
                    [
                        (bbox_cxy[:, 0] - bbox_wh[:, 0] / 2.0),
                        (bbox_cxy[:, 1] - bbox_wh[:, 1] / 2.0),
                        (bbox_cxy[:, 0] + bbox_wh[:, 0] / 2.0),
                        (bbox_cxy[:, 1] + bbox_wh[:, 1] / 2.0),
                    ],
                    axis=-1,
                )
                pos_scores = scores_val[pos_inds]  # Filter scores
                pos_bboxes = bboxes  # bboxes is already filtered

                # If rotated, transform bboxes back
                if angle != 0 and len(pos_bboxes) > 0:
                    points1, points2 = (
                        faceutil.trans_points2d(pos_bboxes[:, :2], IM),
                        faceutil.trans_points2d(pos_bboxes[:, 2:], IM),
                    )
                    _x1, _y1, _x2, _y2 = (
                        points1[:, 0],
                        points1[:, 1],
                        points2[:, 0],
                        points2[:, 1],
                    )
                    if angle in (-270, 90):
                        points1, points2 = (
                            np.stack((_x1, _y2), axis=1),
                            np.stack((_x2, _y1), axis=1),
                        )
                    elif angle in (-180, 180):
                        points1, points2 = (
                            np.stack((_x2, _y2), axis=1),
                            np.stack((_x1, _y1), axis=1),
                        )
                    elif angle in (-90, 270):
                        points1, points2 = (
                            np.stack((_x2, _y1), axis=1),
                            np.stack((_x1, _y2), axis=1),
                        )
                    pos_bboxes = np.hstack((points1, points2))

                # Calculate keypoints for positive detections
                kps_pred_filtered = kps_pred[pos_inds]
                anchor_centers_filtered = anchor_centers[pos_inds]

                kpss = np.concatenate(
                    [
                        (
                            (kps_pred_filtered[:, [2 * i, 2 * i + 1]] * stride)
                            + anchor_centers_filtered
                        )
                        for i in range(5)
                    ],
                    axis=-1,
                )
                kpss = kpss.reshape(
                    (kpss.shape[0], -1, 2)
                )  # Reshape based on filtered count
                pos_kpss = kpss  # Already filtered

                # Handle multi-rotation logic (filtering by face orientation)
                if do_rotation:
                    pos_scores_flat_filtered = (
                        pos_scores.flatten()
                    )  # Flatten again after filtering
                    for i in range(len(pos_kpss)):
                        face_size = max(
                            pos_bboxes[i][2] - pos_bboxes[i][0],
                            pos_bboxes[i][3] - pos_bboxes[i][1],
                        )
                        angle_deg_to_front = faceutil.get_face_orientation(
                            face_size, pos_kpss[i]
                        )
                        if abs(angle_deg_to_front) > 50.00:
                            pos_scores_flat_filtered[i] = (
                                0.0  # Modify the flattened copy
                            )
                        if angle != 0:
                            pos_kpss[i] = faceutil.trans_points2d(pos_kpss[i], IM)

                    # Filter again based on the modified scores
                    pos_inds_rot = np.where(pos_scores_flat_filtered >= score)[0]
                    pos_scores = pos_scores[pos_inds_rot]
                    pos_bboxes = pos_bboxes[pos_inds_rot]
                    pos_kpss = pos_kpss[pos_inds_rot]

                # Ensure pos_scores has the correct shape [N, 1] before appending
                if pos_scores.ndim == 1:
                    pos_scores = pos_scores[:, np.newaxis]

                # Check if there are still detections after rotation filtering
                if pos_scores.size > 0:
                    kpss_list.append(pos_kpss)
                    bboxes_list.append(pos_bboxes)
                    scores_list.append(pos_scores)

        det, kpss, score_values = self._filter_detections_gpu(
            scores_list,
            bboxes_list,
            kpss_list,
            img.shape[1],
            img.shape[2],
            det_scale,
            kwargs.get("max_num"),
        )
        if det is None:
            return (
                np.empty((0, 4)),
                np.empty((0, 5, 2)),
                np.empty((0, 5, 2)),
                np.empty((0,)),
            )

        return self._refine_landmarks(img_landmark, det, kpss, score_values, **kwargs)
