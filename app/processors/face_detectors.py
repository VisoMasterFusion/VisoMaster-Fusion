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

    def __init__(self, models_processor: "ModelsProcessor"):
        self.models_processor = models_processor
        self.center_cache = {}

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
        Prepares an image for a face detection model by resizing, padding, and normalizing it.
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

        det_scale = torch.tensor(new_height / img_height)
        resize = v2.Resize((new_height, new_width), antialias=True)
        resized_img = resize(img)

        # Create a blank canvas with the model's required dtype and paste the resized image.
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
        det_img_canvas[:new_height, :new_width, :] = resized_img.permute(1, 2, 0)

        # Apply model-specific color space and normalization.
        if normalization_mode == "yunet":
            det_img_canvas = det_img_canvas[:, :, [2, 1, 0]]  # RGB to BGR for Yunet

        det_img = det_img_canvas.permute(2, 0, 1)  # Back to CHW format

        if normalization_mode in ["retinaface", "scrfd"]:
            det_img = (det_img - 127.5) / 128.0  # Normalize to [-1.0, 1.0] range

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
    ):
        """
        Performs GPU-accelerated NMS, sorting, and filtering on raw detections from all angles.
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
            return None, None, None

        # Perform Non-Maximum Suppression on the GPU to remove overlapping boxes.
        nms_thresh = 0.4
        keep_indices = nms(bboxes_tensor, scores_tensor, iou_threshold=nms_thresh)

        det_boxes, det_kpss, det_scores = (
            bboxes_tensor[keep_indices],
            kpss_tensor[keep_indices],
            scores_tensor[keep_indices],
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
                area = (det_boxes[:, 2] - det_boxes[:, 0]) * (
                    det_boxes[:, 3] - det_boxes[:, 1]
                )
                det_img_center_y, det_img_center_x = (
                    img_height / det_scale,
                    img_width / det_scale,
                )
                center_x = (det_boxes[:, 0] + det_boxes[:, 2]) / 2 - det_img_center_x
                center_y = (det_boxes[:, 1] + det_boxes[:, 3]) / 2 - det_img_center_y
                offset_dist_squared = center_x**2 + center_y**2
                values = area - offset_dist_squared * 2.0
                bindex = torch.argsort(values, descending=True)[:max_num]
            else:
                bindex = torch.arange(
                    det_boxes.shape[0], device=self.models_processor.device
                )[:max_num]
            det_boxes, det_kpss, det_scores = (
                det_boxes[bindex],
                det_kpss[bindex],
                det_scores[bindex],
            )

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

        # Ensure the required model is loaded into memory.
        model_name = detector["model_name"]
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
        input_name = self.models_processor.models["SCRFD2.5g"].get_inputs()[0].name
        output_names = [
            o.name for o in self.models_processor.models["SCRFD2.5g"].get_outputs()
        ]

        for angle in rotation_angles:
            if angle != 0:
                aimg, M = faceutil.transform(det_img, (cx, cy), 640, 1.0, angle)
                IM = faceutil.invertAffineTransform(M)
                aimg = torch.unsqueeze(aimg, 0).contiguous()
            else:
                IM, aimg = None, torch.unsqueeze(det_img, 0).contiguous()
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
        img, score, rotation_angles = (
            kwargs.get("img"),
            kwargs.get("score"),
            kwargs.get("rotation_angles"),
        )
        img_landmark = img.clone() if kwargs.get("use_landmark_detection") else None

        input_size = (640, 640)
        det_img, det_scale, final_input_size = self._prepare_detection_image(
            img, input_size, "yolo"
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

            # Yolo normalization to [0,1] happens after potential rotation
            aimg = aimg.to(torch.float32) / 255.0

            io_binding = self.models_processor.models["YoloFace8n"].io_binding()
            io_binding.bind_input(
                name="images",
                device_type=self.models_processor.device,
                device_id=0,
                element_type=np.float32,
                shape=aimg.size(),
                buffer_ptr=aimg.data_ptr(),
            )
            io_binding.bind_output("output0", self.models_processor.device)
            if self.models_processor.device == "cuda":
                torch.cuda.synchronize()
            self.models_processor.models["YoloFace8n"].run_with_iobinding(io_binding)
            net_outs = io_binding.copy_outputs_to_cpu()

            outputs = np.squeeze(net_outs).T
            bbox_raw, score_raw, kps_raw, *_ = np.split(outputs, [4, 5], axis=1)
            keep_indices = np.where(score_raw.flatten() > score)[0]

            if keep_indices.any():
                bbox_raw, kps_raw, score_raw = (
                    bbox_raw[keep_indices],
                    kps_raw[keep_indices],
                    score_raw[keep_indices],
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
                    for i in range(len(kpss_raw)):
                        face_size = max(
                            bboxes_raw[i][2] - bboxes_raw[i][0],
                            bboxes_raw[i][3] - bboxes_raw[i][1],
                        )
                        angle_deg_to_front = faceutil.get_face_orientation(
                            face_size, kpss_raw[i]
                        )
                        if abs(angle_deg_to_front) > 50.00:
                            score_raw[i] = 0.0
                        if angle != 0:
                            kpss_raw[i] = faceutil.trans_points2d(kpss_raw[i], IM)
                    keep_indices = np.where(score_raw.flatten() >= score)[0]
                    score_raw, bboxes_raw, kpss_raw = (
                        score_raw[keep_indices],
                        bboxes_raw[keep_indices],
                        kpss_raw[keep_indices],
                    )
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
        img, score, rotation_angles = (
            kwargs.get("img"),
            kwargs.get("score"),
            kwargs.get("rotation_angles"),
        )
        img_landmark = img.clone() if kwargs.get("use_landmark_detection") else None

        input_size = (640, 640)
        det_img, det_scale, final_input_size = self._prepare_detection_image(
            img, input_size, "yunet"
        )

        scores_list, bboxes_list, kpss_list = [], [], []
        cx, cy = final_input_size[0] / 2, final_input_size[1] / 2
        do_rotation = len(rotation_angles) > 1
        input_name = self.models_processor.models["YunetN"].get_inputs()[0].name
        output_names = [
            o.name for o in self.models_processor.models["YunetN"].get_outputs()
        ]

        for angle in rotation_angles:
            if angle != 0:
                aimg, M = faceutil.transform(det_img, (cx, cy), 640, 1.0, angle)
                IM = faceutil.invertAffineTransform(M)
                aimg = torch.unsqueeze(aimg, 0).contiguous()
            else:
                IM, aimg = None, torch.unsqueeze(det_img, 0).contiguous()

            aimg = aimg.to(dtype=torch.float32)  # Yunet expects float32 input

            io_binding = self.models_processor.models["YunetN"].io_binding()
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
                    self.center_cache[key] = anchor_centers
                scores_val = cls_pred * obj_pred
                pos_inds = np.where(scores_val.flatten() >= score)[0]
                bbox_cxy = reg_pred[:, :2] * stride + anchor_centers[:]
                bbox_wh = np.exp(reg_pred[:, 2:]) * stride
                bboxes = np.stack(
                    [
                        (bbox_cxy[:, 0] - bbox_wh[:, 0] / 2.0),
                        (bbox_cxy[:, 1] - bbox_wh[:, 1] / 2.0),
                        (bbox_cxy[:, 0] + bbox_wh[:, 0] / 2.0),
                        (bbox_cxy[:, 1] + bbox_wh[:, 1] / 2.0),
                    ],
                    axis=-1,
                )
                pos_scores, pos_bboxes = scores_val[pos_inds], bboxes[pos_inds]
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
                kpss = np.concatenate(
                    [
                        ((kps_pred[:, [2 * i, 2 * i + 1]] * stride) + anchor_centers)
                        for i in range(5)
                    ],
                    axis=-1,
                )
                kpss = kpss.reshape((kpss.shape[0], -1, 2))
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
                    pos_inds = np.where(pos_scores.flatten() >= score)[0]
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
