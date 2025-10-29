from itertools import product as product
from typing import TYPE_CHECKING, List, Dict
import pickle

import torch
import cv2
import numpy as np
from torchvision.transforms import v2

if TYPE_CHECKING:
    from app.processors.models_processor import ModelsProcessor
from app.processors.models_data import models_dir
from app.processors.utils import faceutil


class FaceLandmarkDetectors:
    """
    Manages and executes various face landmark detection models.
    This class acts as a dispatcher to select the appropriate detector based on a given mode.
    It handles model loading, pre-processing (image warping), inference execution,
    and post-processing (transforming landmarks back to the original image space).
    """

    def unload_models(self):
        if self.current_landmark_model:
            self.models_processor.unload_model(self.current_landmark_model)
            self.current_landmark_model = None

    def __init__(self, models_processor: "ModelsProcessor"):
        """
        Initializes the FaceLandmarkDetectors.

        Args:
            models_processor (ModelsProcessor): A reference to the main ModelsProcessor instance
                                                which handles model loading and device management.
        """
        self.models_processor = models_processor
        self.current_landmark_model = None
        # Caches for model-specific data to avoid re-computation.
        self.landmark_5_anchors = []
        self.landmark_5_scale1_cache = {}
        self.landmark_5_priors = None

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
        self, img, bbox, det_kpss, detect_mode="203", score=0.5, from_points=False
    ):
        """
        Main dispatcher function to run a specific landmark detector.

        Args:
            img (torch.Tensor): The full source image.
            bbox (np.ndarray): The bounding box of the face. Used if 'from_points' is False.
            det_kpss (np.ndarray): The initial keypoints (e.g., from a face detector). Used if 'from_points' is True.
            detect_mode (str): The identifier for the desired landmark model (e.g., '68', '203').
            score (float): The minimum confidence score to accept the detection.
            from_points (bool): If True, use 'det_kpss' to align the face. If False, use 'bbox'.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
                - 5-point landmarks.
                - Full landmarks (e.g., 68 points).
                - Confidence scores for the landmarks.
        """
        kpss_5, kpss, scores = [], [], []

        # Look up the detector information from the map.
        detector_info = self.detector_map.get(detect_mode)
        if not detector_info:
            print(f"[WARN] Landmark detector mode '{detect_mode}' not found.")
            return kpss_5, kpss, scores

        model_name = detector_info["model_name"]
        detection_function = detector_info["function"]  # Moved this line here

        # Unload previous model if different
        if self.current_landmark_model and self.current_landmark_model != model_name:
            print(f"Successfully unloaded model: {self.current_landmark_model}")
            self.models_processor.unload_model(self.current_landmark_model)
            # Reset current_landmark_model immediately after unloading
            self.current_landmark_model = None

        # Lazily load the required ONNX model if it's not already in memory.
        loaded_model_instance = self.models_processor.models.get(model_name)
        if not loaded_model_instance:
            loaded_model_instance = self.models_processor.load_model(model_name)

        # If model still not loaded (e.g., failed to load), print a warning and return empty
        if not loaded_model_instance:
            print(
                f"WARNING: Landmark model '{model_name}' failed to load or is not available. Skipping detection."
            )
            return kpss_5, kpss, scores

        # ONLY update current_landmark_model if the new model was successfully loaded
        if (
            self.current_landmark_model != model_name
        ):  # Check if it's actually a new model or if it was just re-loaded
            print(f"Successfully loaded model: {model_name}")
            self.current_landmark_model = model_name

        # Handle special setup cases for certain models.
        if detect_mode == "3d68":
            with open(f"{models_dir}/meanshape_68.pkl", "rb") as f:
                self.models_processor.mean_lmk = pickle.load(f)
        elif detect_mode == "478":
            if not self.models_processor.models["FaceBlendShapes"]:
                self.models_processor.models["FaceBlendShapes"] = (
                    self.models_processor.load_model("FaceBlendShapes")
                )
        elif detect_mode == "5":
            # This model requires pre-calculated anchor points.
            self._ensure_landmark_5_anchors()

        # Call the specific detection function (e.g., detect_face_landmark_68).
        kpss_5, kpss, scores = detection_function(
            img, bbox=bbox, det_kpss=det_kpss, from_points=from_points
        )

        # Filter the final results based on the provided confidence score.
        if len(kpss_5) > 0 and len(scores) > 0:
            if np.mean(scores) >= score:
                return kpss_5, kpss, scores
        elif (
            len(kpss_5) > 0
        ):  # If no scores are returned by the model, pass through the landmarks.
            return kpss_5, kpss, scores

        return [], [], []

    def _ensure_landmark_5_anchors(self):
        """
        Initializes the anchors for the FaceLandmark5 model.
        This complex calculation is performed only once and the result is cached for efficiency.
        """
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
        This avoids data copies between CPU and GPU.

        Args:
            model_name (str): The name of the model to execute.
            input_bindings (Dict): A dictionary mapping input names to their torch.Tensor data.
            output_names (List): A list of the names of the output nodes.

        Returns:
            List[np.ndarray]: A list of numpy arrays containing the model's output.
        """
        model = self.models_processor.models[model_name]
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

        # Synchronize the CUDA stream before execution.
        if self.models_processor.device == "cuda":
            torch.cuda.synchronize()
        elif self.models_processor.device != "cpu":
            self.models_processor.syncvec.cpu()

        # Run inference and copy results back to CPU.
        model.run_with_iobinding(io_binding)
        return io_binding.copy_outputs_to_cpu()

    def detect_face_landmark_5(self, img, bbox, det_kpss, from_points=False):
        # This model's pre-processing is unique, so it doesn't use the `_prepare_crop` helper.
        if from_points == False:
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

        # Pre-process: subtract mean and reshape for the model.
        image = image.permute(1, 2, 0)
        mean = torch.tensor(
            [104, 117, 123], dtype=torch.float32, device=self.models_processor.device
        )
        image = torch.sub(image, mean).permute(2, 0, 1).reshape(1, 3, 512, 512)

        # Prepare scaling factor for post-processing.
        height, width = 512, 512
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
        conf, landmarks = (
            torch.from_numpy(net_outs[0]).to(self.models_processor.device),
            torch.from_numpy(net_outs[1]).to(self.models_processor.device),
        )

        # Post-process the raw model output.
        scores = torch.squeeze(conf)[:, 1]
        priors, pre = self.landmark_5_priors, torch.squeeze(landmarks, 0)

        # Decode landmarks from priors and predictions.
        landmarks = (
            torch.cat(
                [
                    priors[:, :2] + pre[:, i : i + 2] * 0.1 * priors[:, 2:]
                    for i in range(0, 10, 2)
                ],
                dim=1,
            )
            * scale1
        )

        landmarks, scores = landmarks.cpu().numpy(), scores.cpu().numpy()
        inds = np.where(scores > 0.1)[0]
        landmarks, scores = landmarks[inds], scores[inds]

        order = scores.argsort()[::-1]
        if len(order) > 0:
            landmarks = landmarks[order][0]
            scores = scores[order][0]
            landmarks = np.array(
                [[landmarks[i], landmarks[i + 1]] for i in range(0, 10, 2)]
            )
            # Transform landmarks back to the original image's coordinate space.
            IM = faceutil.invertAffineTransform(M)
            landmarks = faceutil.trans_points2d(landmarks, IM)
            return landmarks, landmarks, np.array([scores])
        return [], [], []

    def detect_face_landmark_68(self, img, bbox, det_kpss, from_points=False):
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
            "FaceLandmark68", {"input": crop_image}, ["landmarks_xyscore", "heatmaps"]
        )
        face_landmark_68, face_heatmap = net_outs[0], net_outs[1]

        # Post-process: scale, transform, and reshape landmarks.
        face_landmark_68 = (face_landmark_68[:, :, :2][0] / 64.0).reshape(
            1, -1, 2
        ) * 256.0
        face_landmark_68 = cv2.transform(
            face_landmark_68, cv2.invertAffineTransform(affine_matrix)
        ).reshape(-1, 2)
        face_landmark_68_score = np.amax(face_heatmap, axis=(2, 3)).reshape(-1, 1)

        # Convert the 68 points to a standard 5-point format.
        face_landmark_68_5, face_landmark_68_score = (
            faceutil.convert_face_landmark_68_to_5(
                face_landmark_68, face_landmark_68_score
            )
        )
        return face_landmark_68_5, face_landmark_68, face_landmark_68_score

    def detect_face_landmark_3d68(self, img, bbox, det_kpss, from_points=False):
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
        pred = self._run_onnx_binding("FaceLandmark3d68", {"data": aimg}, ["fc1"])[0][0]

        # Post-process the 1D prediction array into 3D/2D coordinates.
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

    def detect_face_landmark_98(self, img, bbox, det_kpss, from_points=False):
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
        landmarks_xyscore = self._run_onnx_binding(
            "FaceLandmark98", {"input": crop_image}, ["landmarks_xyscore"]
        )[0]

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

    def detect_face_landmark_106(self, img, bbox, det_kpss, from_points=False):
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
        pred = self._run_onnx_binding("FaceLandmark106", {"data": aimg}, ["fc1"])[0][0]

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

    def detect_face_landmark_203(self, img, bbox, det_kpss, from_points=False):
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
        out_pts = (
            out_lst[2].reshape((-1, 2)) * 224.0
        )  # The third output contains the landmarks.

        out_pts = faceutil.trans_points(out_pts, IM)
        out_pts_5 = (
            faceutil.convert_face_landmark_203_to_5(out_pts)
            if out_pts is not None
            else []
        )
        return out_pts_5, out_pts, []

    def detect_face_landmark_478(self, img, bbox, det_kpss, from_points=False):
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
        landmarks = net_outs[0].reshape((1, 478, 3))

        if len(landmarks) > 0:
            landmark = faceutil.trans_points3d(landmarks[0], IM)[:, :2].reshape(-1, 2)

            # This model uses a second network ('FaceBlendShapes') to get scores.
            landmark_for_score = landmark[self.models_processor.LandmarksSubsetIdxs]
            landmark_for_score = torch.from_numpy(
                np.expand_dims(landmark_for_score, axis=0).astype(np.float32)
            ).to(self.models_processor.device)
            self._run_onnx_binding(
                "FaceBlendShapes", {"input_points": landmark_for_score}, ["output"]
            )

            landmark_5 = faceutil.convert_face_landmark_478_to_5(landmark)
            return landmark_5, landmark, []
        return [], [], []
