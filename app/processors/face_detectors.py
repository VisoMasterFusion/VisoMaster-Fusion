from typing import TYPE_CHECKING, Dict, Any

import torch
from torchvision.transforms import v2
from torchvision.ops import nms
import numpy as np

if TYPE_CHECKING:
    from app.processors.models_processor import ModelsProcessor

from app.processors.utils import faceutil


class FaceDetectors:
    """
    Manages and executes various face detection models.
    This class acts as a dispatcher to select the appropriate detector and provides
    helper methods for image preparation and filtering of detection results.
    """

    def unload_models(self):
        if self.current_detector_model:
            self.models_processor.unload_model(self.current_detector_model)
            self.current_detector_model = None

    def __init__(self, models_processor: "ModelsProcessor"):
        self.models_processor = models_processor
        self.center_cache = {}
        self.current_detector_model = None

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

    def _prepare_detection_image(
        self, img: torch.Tensor, input_size: tuple, normalization_mode: str
    ) -> tuple[torch.Tensor, torch.Tensor, tuple]:
        """
        Prepares an image for a face detection model by resizing, padding, and handling color space.
        Normalization and dtype conversion for Yolo/Yunet are deferred to their respective functions.
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
        )  # Ensure float division
        resize = v2.Resize((new_height, new_width), antialias=True)
        resized_img = resize(img)

        # Create a blank canvas with the model's required dtype and paste the resized image.
        # Yolo and Yunet start with uint8 here. RetinaFace/SCRFD use float32.
        canvas_dtype = (
            torch.float32
            if normalization_mode in ["retinaface", "scrfd"]
            else torch.uint8
        )
        det_img_canvas = torch.zeros(
            (input_size[1], input_size[0], 3),
            dtype=canvas_dtype,
            device=self.models_processor.device,
        )
        # Ensure resized_img is compatible with canvas dtype before assignment
        if canvas_dtype == torch.uint8 and resized_img.dtype != torch.uint8:
            # Assuming resized_img might be float [0, 255] after resize
            resized_img_casted = resized_img.byte()
        elif canvas_dtype == torch.float32 and resized_img.dtype == torch.uint8:
            resized_img_casted = resized_img.float()  # Keep range [0, 255] for now
        else:
            resized_img_casted = resized_img

        det_img_canvas[:new_height, :new_width, :] = resized_img_casted.permute(1, 2, 0)

        # Apply model-specific color space.
        if normalization_mode == "yunet":
            det_img_canvas = det_img_canvas[:, :, [2, 1, 0]]  # RGB to BGR for Yunet

        det_img = det_img_canvas.permute(2, 0, 1)  # Back to CHW format

        # Apply normalization ONLY for RetinaFace/SCRFD here.
        if normalization_mode in ["retinaface", "scrfd"]:
            # If canvas was uint8 initially (unlikely here but safe check)
            if det_img.dtype == torch.uint8:
                det_img = det_img.float()
            det_img = (det_img - 127.5) / 128.0  # Normalize to [-1.0, 1.0] range

        # For Yolo/Yunet, det_img remains uint8 [0, 255] at this stage.

        return det_img, det_scale, input_size

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
            # Log mismatch?
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
                # --- END BUG FIX ---

                center_x = (det_boxes[:, 0] + det_boxes[:, 2]) / 2 - det_img_center_x
                center_y = (det_boxes[:, 1] + det_boxes[:, 3]) / 2 - det_img_center_y

                offset_dist_squared = center_x**2 + center_y**2
                # This score favors large faces (area) that are close to the center
                # (low offset_dist_squared).
                values = area - offset_dist_squared * 2.0
                bindex = torch.argsort(values, descending=True)[:max_num]
            else:
                bindex = torch.arange(
                    det_boxes.shape[0], device=self.models_processor.device
                )[:max_num]

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
        return det, kpss_5, kpss

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
    ):
        """
        Main dispatcher for running face detection. Selects and runs the appropriate model.
        """
        detector = self.detector_map.get(detect_mode)
        if not detector:
            return [], [], []

        model_name = detector["model_name"]
        if self.current_detector_model and self.current_detector_model != model_name:
            self.models_processor.unload_model(self.current_detector_model)
        self.current_detector_model = model_name

        # Ensure the required model is loaded into memory.
        if not self.models_processor.models.get(model_name):
            self.models_processor.models[model_name] = self.models_processor.load_model(
                model_name
            )

        detection_function = detector["function"]

        # Prepare arguments for the specific detector function.
        args = {
            "img": img,
            "max_num": max_num,
            "score": score,
            "use_landmark_detection": use_landmark_detection,
            "landmark_detect_mode": landmark_detect_mode,
            "landmark_score": landmark_score,
            "from_points": from_points,
            "rotation_angles": rotation_angles or [0],
        }
        # Some detectors have a parameterized input size, others are fixed.
        if detect_mode in ["RetinaFace", "SCRFD"]:
            args["input_size"] = input_size

        return detection_function(**args)

    def detect_retinaface(self, **kwargs):
        """Runs the RetinaFace detection pipeline."""
        model_name = "RetinaFace"
        ort_session = self.models_processor.models.get(model_name)
        if not ort_session:
            print(f"WARNING: {model_name} model not loaded. Skipping detection.")
            return [], [], []

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
                aimg, M = faceutil.transform(det_img, (cx, cy), 640, 1.0, angle)
                IM = faceutil.invertAffineTransform(M)
                aimg = torch.unsqueeze(aimg, 0).contiguous()
            else:
                IM, aimg = None, torch.unsqueeze(det_img, 0).contiguous()

            io_binding = ort_session.io_binding()

            io_binding = self.models_processor.models["RetinaFace"].io_binding()
            io_binding.bind_input(
                name="input.1",
                device_type=self.models_processor.device,
                device_id=0,
                element_type=np.float32,
                shape=aimg.size(),
                buffer_ptr=aimg.data_ptr(),
            )
            for i in ["448", "471", "494", "451", "474", "497", "454", "477", "500"]:
                io_binding.bind_output(i, self.models_processor.device)
            if self.models_processor.device == "cuda":
                torch.cuda.synchronize()
            self.models_processor.models["RetinaFace"].run_with_iobinding(io_binding)
            net_outs = io_binding.copy_outputs_to_cpu()

            input_height, input_width = aimg.shape[2], aimg.shape[3]
            fmc = 3
            for idx, stride in enumerate([8, 16, 32]):
                scores, bbox_preds, kps_preds = (
                    net_outs[idx],
                    net_outs[idx + fmc] * stride,
                    net_outs[idx + fmc * 2] * stride,
                )
                height, width = input_height // stride, input_width // stride
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
                pos_inds = np.where(scores >= score)[0]
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
                if do_rotation:
                    for i in range(len(pos_kpss)):
                        face_size = max(
                            pos_bboxes[i][2] - pos_bboxes[i][0],
                            pos_bboxes[i][3] - pos_bboxes[i][1],
                        )
                        angle_deg_to_front = faceutil.get_face_orientation(
                            face_size, pos_kpss[i]
                        )
                        if abs(angle_deg_to_front) > 50.00:
                            pos_scores[i] = 0.0
                        if angle != 0:
                            pos_kpss[i] = faceutil.trans_points2d(pos_kpss[i], IM)
                    pos_inds = np.where(pos_scores >= score)[0]
                    pos_scores, pos_bboxes, pos_kpss = (
                        pos_scores[pos_inds],
                        pos_bboxes[pos_inds],
                        pos_kpss[pos_inds],
                    )
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
            return [], [], []

        return self._refine_landmarks(img_landmark, det, kpss, score_values, **kwargs)

    def detect_scrdf(self, **kwargs):
        """Runs the SCRFD detection pipeline."""
        model_name = "SCRFD2.5g"
        ort_session = self.models_processor.models.get(model_name)
        if not ort_session:
            print(f"WARNING: {model_name} model not loaded. Skipping detection.")
            return [], [], []

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
                aimg, M = faceutil.transform(det_img, (cx, cy), 640, 1.0, angle)
                IM = faceutil.invertAffineTransform(M)
                aimg = torch.unsqueeze(aimg, 0).contiguous()
            else:
                IM, aimg = None, torch.unsqueeze(det_img, 0).contiguous()

            io_binding = ort_session.io_binding()

            io_binding = self.models_processor.models["SCRFD2.5g"].io_binding()
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
            if self.models_processor.device == "cuda":
                torch.cuda.synchronize()
            self.models_processor.models["SCRFD2.5g"].run_with_iobinding(io_binding)
            net_outs = io_binding.copy_outputs_to_cpu()
            input_height, input_width = aimg.shape[2], aimg.shape[3]
            fmc = 3
            for idx, stride in enumerate([8, 16, 32]):
                scores, bbox_preds, kps_preds = (
                    net_outs[idx],
                    net_outs[idx + fmc] * stride,
                    net_outs[idx + fmc * 2] * stride,
                )
                height, width = input_height // stride, input_width // stride
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
                pos_inds = np.where(scores >= score)[0]
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
                if do_rotation:
                    for i in range(len(pos_kpss)):
                        face_size = max(
                            pos_bboxes[i][2] - pos_bboxes[i][0],
                            pos_bboxes[i][3] - pos_bboxes[i][1],
                        )
                        angle_deg_to_front = faceutil.get_face_orientation(
                            face_size, pos_kpss[i]
                        )
                        if abs(angle_deg_to_front) > 50.00:
                            pos_scores[i] = 0.0
                        if angle != 0:
                            pos_kpss[i] = faceutil.trans_points2d(pos_kpss[i], IM)
                    pos_inds = np.where(pos_scores >= score)[0]
                    pos_scores, pos_bboxes, pos_kpss = (
                        pos_scores[pos_inds],
                        pos_bboxes[pos_inds],
                        pos_kpss[pos_inds],
                    )
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
            return [], [], []

        return self._refine_landmarks(img_landmark, det, kpss, score_values, **kwargs)

    def detect_yoloface(self, **kwargs):
        """Runs the Yolov8-face detection pipeline."""
        model_name = "YoloFace8n"
        ort_session = self.models_processor.models.get(model_name)
        if not ort_session:
            print(f"WARNING: {model_name} model not loaded. Skipping detection.")
            return [], [], []

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

            # *** CORRECTION: Convert to float and normalize AFTER rotation, before binding ***
            aimg_prepared = aimg.to(torch.float32) / 255.0
            aimg_prepared = torch.unsqueeze(
                aimg_prepared, 0
            ).contiguous()  # Add batch dim

            io_binding = ort_session.io_binding()

            io_binding = self.models_processor.models["YoloFace8n"].io_binding()
            io_binding.bind_input(
                name="images",
                device_type=self.models_processor.device,
                device_id=0,
                element_type=np.float32,
                shape=aimg_prepared.size(),  # Use shape of prepared tensor
                buffer_ptr=aimg_prepared.data_ptr(),  # Use data_ptr of prepared tensor
            )
            io_binding.bind_output("output0", self.models_processor.device)
            if self.models_processor.device == "cuda":
                torch.cuda.synchronize()
            self.models_processor.models["YoloFace8n"].run_with_iobinding(io_binding)
            net_outs = io_binding.copy_outputs_to_cpu()

            outputs = np.squeeze(net_outs).T
            bbox_raw, score_raw, kps_raw, *_ = np.split(outputs, [4, 5], axis=1)
            # Flatten score_raw before comparison
            score_raw_flat = score_raw.flatten()
            keep_indices = np.where(score_raw_flat > score)[0]

            if keep_indices.size > 0:  # Check size instead of any() for numpy arrays
                bbox_raw, kps_raw, score_raw = (
                    bbox_raw[keep_indices],
                    kps_raw[keep_indices],
                    score_raw[
                        keep_indices
                    ],  # Keep score_raw as [N, 1] or similar for consistency
                )
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
                kpss_raw = np.stack(
                    [
                        np.array([[kps[i], kps[i + 1]] for i in range(0, len(kps), 3)])
                        for kps in kps_raw
                    ]
                )
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
                    # Make sure score_raw is [N, 1] or similar before indexing
                    score_raw = score_raw[keep_indices_rot]
                    bboxes_raw = bboxes_raw[keep_indices_rot]
                    kpss_raw = kpss_raw[keep_indices_rot]

                # Ensure score_raw has the correct shape before appending
                if score_raw.ndim == 1:
                    score_raw = score_raw[
                        :, np.newaxis
                    ]  # Reshape to [N, 1] if it's flat

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
            return [], [], []

        return self._refine_landmarks(img_landmark, det, kpss, score_values, **kwargs)

    def detect_yunet(self, **kwargs):
        """Runs the Yunet detection pipeline."""
        model_name = "YunetN"
        ort_session = self.models_processor.models.get(model_name)
        if not ort_session:
            print(f"WARNING: {model_name} model not loaded. Skipping detection.")
            return [], [], []

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

            # *** CORRECTION: Convert to float AFTER rotation, before binding ***
            aimg_prepared = aimg.to(dtype=torch.float32)
            aimg_prepared = torch.unsqueeze(
                aimg_prepared, 0
            ).contiguous()  # Add batch dim

            io_binding = ort_session.io_binding()

            io_binding = self.models_processor.models["YunetN"].io_binding()
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
            if self.models_processor.device == "cuda":
                torch.cuda.synchronize()
            self.models_processor.models["YunetN"].run_with_iobinding(io_binding)
            net_outs = io_binding.copy_outputs_to_cpu()
            strides = [8, 16, 32]
            for idx, stride in enumerate(strides):
                cls_pred, obj_pred, reg_pred, kps_pred = (
                    net_outs[idx].reshape(-1, 1),
                    net_outs[idx + len(strides)].reshape(-1, 1),
                    net_outs[idx + len(strides) * 2].reshape(-1, 4),
                    net_outs[idx + len(strides) * 3].reshape(-1, 5 * 2),
                )
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

                bbox_cxy = (
                    reg_pred[pos_inds, :2] * stride + anchor_centers[pos_inds, :]
                )  # Filter anchor_centers too
                bbox_wh = np.exp(reg_pred[pos_inds, 2:]) * stride

                bboxes = np.stack(
                    [
                        (bbox_cxy[:, 0] - bbox_wh[:, 0] / 2.0),
                        (bbox_cxy[:, 1] - bbox_wh[:, 1] / 2.0),
                        (bbox_cxy[:, 0] + bbox_wh[:, 0] / 2.0),
                        (bbox_cxy[:, 1] + bbox_wh[:, 1] / 2.0),
                    ],
                    axis=-1,
                )
                # Filter scores before assignment
                pos_scores = scores_val[pos_inds]
                pos_bboxes = bboxes  # bboxes is already filtered

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

                # Filter kps_pred and anchor_centers before calculating kpss
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

                # Reshape based on the number of filtered keypoints
                kpss = kpss.reshape((kpss.shape[0], -1, 2))
                pos_kpss = kpss  # Already filtered

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
                    # Make sure pos_scores is [N, 1] or similar before indexing
                    pos_scores = pos_scores[pos_inds_rot]
                    pos_bboxes = pos_bboxes[pos_inds_rot]
                    pos_kpss = pos_kpss[pos_inds_rot]

                # Ensure pos_scores has the correct shape before appending
                if pos_scores.ndim == 1:
                    pos_scores = pos_scores[
                        :, np.newaxis
                    ]  # Reshape to [N, 1] if it's flat

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
            return [], [], []

        return self._refine_landmarks(img_landmark, det, kpss, score_values, **kwargs)
