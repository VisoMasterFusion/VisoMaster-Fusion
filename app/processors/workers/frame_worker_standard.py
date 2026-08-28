import math
import threading
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
from torchvision.transforms import v2
import kornia.geometry.transform as kgm

from app.processors.models_data import landmark_point_counts
from app.helpers.miscellaneous import (
    ParametersDict,
    draw_bounding_boxes_on_detected_faces,
    paint_landmarks_on_image,
    is_detected_face_eligible_for_matching,
    keypoints_adjustments,
)

if TYPE_CHECKING:
    # Forward reference to the main FrameWorker orchestrator
    from .frame_worker import FrameWorker


class StandardProcessor:
    """
    Handles all standard (flat) video and image processing operations.
    Delegates back to the main FrameWorker for pipeline execution (swap_core),
    matching, and shared VRAM caches to guarantee thread safety.
    """

    def __init__(self, worker: "FrameWorker"):
        self.worker = worker

    @torch.no_grad()
    def process_standard_frame(
        self,
        processed_tensor_rgb_uint8: torch.Tensor,
        control: dict,
        stop_event: threading.Event,
    ) -> torch.Tensor:
        """
        Handles the standard (flat) processing logic:
        - Rotation
        - Detection
        - Swapping/Editing
        - Overlays (BBox, Landmarks, Comparison)
        """
        # FW-RACE-02: read button state from snapshotted feeder dict, not live Qt buttons
        swap_button_is_checked_global = (
            self.worker.main_window.swapfacesButton.isChecked()
        )
        edit_button_is_checked_global = (
            self.worker.main_window.editFacesButton.isChecked()
        )

        # FW-RACE-01: snapshot target_faces under lock so the worker thread never
        # iterates the live dict while the UI thread may be modifying it.
        with self.worker.lock:
            target_faces_snapshot = dict(self.worker.main_window.target_faces)

        det_faces_data_for_display = []

        img = processed_tensor_rgb_uint8
        # FW-QUAL-11: rename img_x/img_y to img_w/img_h for clarity
        img_w, img_h = img.size(2), img.size(1)
        scale_applied = False

        # FW-BUG-06: Upscale small frames so the shorter side is at least 512px before detection
        if img_w < 512 or img_h < 512:
            if img_w <= img_h:
                new_h, new_w = int(512 * img_h / img_w), 512
            else:
                new_h, new_w = 512, int(512 * img_w / img_h)

            # FW-ROBUST-2: respect the user's interpolation setting rather than
            # always using the default (NEAREST / BILINEAR depending on build)
            # FW-PERF-08 / FW-MEM-02: LRU-bounded cache of v2.Resize objects to avoid
            # re-constructing the same transform on every frame (max 16 entries).
            _up_key = (new_h, new_w, self.worker.interpolation_scaleback, False)
            if _up_key in self.worker._resize_cache:
                self.worker._resize_cache.move_to_end(_up_key)
            else:
                if len(self.worker._resize_cache) >= self.worker._RESIZE_CACHE_MAX:
                    self.worker._resize_cache.popitem(last=False)
                self.worker._resize_cache[_up_key] = v2.Resize(
                    (new_h, new_w),
                    interpolation=self.worker.interpolation_scaleback,
                    antialias=False,
                )
            img = self.worker._resize_cache[_up_key](img)
            scale_applied = True

            # Upscale Feeder KPS if necessary
            if self.worker.precomputed_bboxes is not None:
                ratio_w = new_w / img_w
                ratio_h = new_h / img_h

                self.worker.precomputed_bboxes = [
                    b.copy() for b in self.worker.precomputed_bboxes
                ]
                for b in self.worker.precomputed_bboxes:
                    b[0] *= ratio_w
                    b[2] *= ratio_w
                    b[1] *= ratio_h
                    b[3] *= ratio_h

                if self.worker.precomputed_kpss_5 is not None:
                    self.worker.precomputed_kpss_5 = [
                        k.copy() for k in self.worker.precomputed_kpss_5
                    ]
                    for k in self.worker.precomputed_kpss_5:
                        k[:, 0] *= ratio_w
                        k[:, 1] *= ratio_h

                if self.worker.precomputed_kpss is not None:
                    self.worker.precomputed_kpss = [
                        k.copy() if k is not None else None
                        for k in self.worker.precomputed_kpss
                    ]
                    for k in self.worker.precomputed_kpss:
                        if k is not None:
                            k[:, 0] *= ratio_w
                            k[:, 1] *= ratio_h

                if self.worker.precomputed_kpss_203 is not None:
                    self.worker.precomputed_kpss_203 = [
                        k.copy() if k is not None else None
                        for k in self.worker.precomputed_kpss_203
                    ]
                    for k in self.worker.precomputed_kpss_203:
                        if k is not None:
                            k[:, 0] *= ratio_w
                            k[:, 1] *= ratio_h

        # Manual Rotation
        # FW-BUG-FIX: Store pre- and post-rotation dimensions to accurately
        # reverse the affine transform later without accumulating black borders.
        pre_rot_h, pre_rot_w = img.shape[1], img.shape[2]

        if control["ManualRotationEnableToggle"]:
            img = v2.functional.rotate(
                img,
                angle=control["ManualRotationAngleSlider"],
                interpolation=v2.InterpolationMode.BILINEAR,
                expand=True,
            )

        post_rot_h, post_rot_w = img.shape[1], img.shape[2]

        # --- DETECTION PHASE ---
        # The workers are now "Stateless Render Engines". They no longer track time or state.
        # They consume perfectly sequenced and EMA-smoothed detections from the Feeder thread.

        # FW-ARCH-FIX: Flag to know if faces were vetted by the Sequential Detector
        is_sequentially_tracked = not self.worker.is_single_frame

        if is_sequentially_tracked:
            # 1. Primary Path (Video/Webcam): Use the sequentially precomputed detections
            bboxes = self.worker.precomputed_bboxes
            kpss_5 = self.worker.precomputed_kpss_5
            kpss = self.worker.precomputed_kpss
            kpss_203 = self.worker.precomputed_kpss_203
        else:
            # 2. Fallback Path (Single-Frame Mode / Static Images)
            use_landmark, landmark_mode, from_points = (
                control.get("LandmarkDetectToggle", True),
                control.get("LandmarkDetectModelSelection", "203"),
                control.get("DetectFromPointsToggle", False),
            )

            # Identify if 203 points are strictly required by an advanced feature
            requires_203 = False
            from collections.abc import Mapping

            for face_params in self.worker.parameters.values():
                if isinstance(face_params, Mapping):
                    is_face_editor_active = (
                        edit_button_is_checked_global
                        and face_params.get("FaceEditorEnableToggle", False)
                    )
                    is_expression_active = face_params.get(
                        "FaceExpressionEnableBothToggle", False
                    )
                    is_auto_mouth_active = face_params.get(
                        "AutoMouthExpressionEnableToggle", False
                    )
                    is_makeup_active = edit_button_is_checked_global and (
                        face_params.get("FaceMakeupEnableToggle", False)
                        or face_params.get("HairMakeupEnableToggle", False)
                        or face_params.get("EyeBrowsMakeupEnableToggle", False)
                        or face_params.get("LipsMakeupEnableToggle", False)
                    )

                    if (
                        is_face_editor_active
                        or is_expression_active
                        or is_auto_mouth_active
                        or is_makeup_active
                    ):
                        requires_203 = True
                        break

            # --- Strict Early Exit (No Targets in Single-Frame) ---
            show_bboxes = control.get(
                "ShowAllDetectedFacesBBoxToggle", False
            ) or control.get("ShowByteTrackBBoxToggle", False)

            if len(target_faces_snapshot) == 0 and not show_bboxes:
                bboxes = np.empty((0, 4), dtype=np.float32)
                kpss_5 = np.empty((0, 5, 2), dtype=np.float32)
                kpss = np.empty((0, 68, 2), dtype=np.float32)
                kpss_203 = np.empty((0, 203, 2), dtype=np.float32)
                has_valid_dense_kpss = False
            else:
                # --- STEP 1: Standard detection (Respect UI Toggle) ---
                # We pass 'use_landmark_detection=use_landmark' to allow run_detect
                # to attempt an initial extraction. If Auto-Rotation is enabled,
                # run_detect may bypass dense landmark extraction if the angle is too extreme.
                bboxes, kpss_5, kpss = self.worker.function_worker.run_detect(
                    img,
                    control.get("DetectorModelSelection", "RetinaFace"),
                    max_num=control.get("MaxFacesToDetectSlider", 1),
                    score=control.get("DetectorScoreSlider", 50) / 100.0,
                    input_size=(512, 512),
                    use_landmark_detection=use_landmark,
                    landmark_detect_mode=landmark_mode,
                    landmark_score=control.get("LandmarkDetectScoreSlider", 50) / 100.0,
                    from_points=from_points,
                    rotation_angles=[0]
                    if not control.get("AutoRotationToggle", False)
                    else [0, 90, 180, 270],
                    use_mean_eyes=control.get("LandmarkMeanEyesToggle", False),
                    previous_detections=None,
                    bypass_bytetrack=True,
                )

                # FW-LOGIC-FIX: Validate if Step 1 returned actual dense landmarks (> 5 points).
                # If run_detect aborted dense extraction due to an extreme rotation angle,
                # 'kpss' will merely contain a fallback copy of the sparse 'kpss_5'.
                has_valid_dense_kpss = (
                    kpss is not None
                    and len(kpss) == len(bboxes)
                    and len(bboxes) > 0
                    and kpss[0].shape[0] > 5
                )

            # --- STEP 2: 203-Landmark Extraction (Expression Restorer / Face Editor) ---
            kpss_203 = None
            if requires_203:
                kpss_203_list = []
                # LOGIC FIX: We ONLY reuse Step 1 landmarks if they are already in the 203
                # format AND the user enabled 'from_points'. Otherwise, we MUST re-extract
                # with 'from_points=True' to ensure geometric alignment for the Expression Restorer.
                can_reuse_step1_203 = (
                    has_valid_dense_kpss and landmark_mode == "203" and from_points
                )

                if bboxes is not None and len(bboxes) > 0:
                    for idx in range(len(bboxes)):
                        if can_reuse_step1_203:
                            kps_203_local = kpss[idx].copy()
                        else:
                            # FORCE from_points=True: Strictly required for the geometric
                            # alignment of advanced tools (FaceEditor, Makeup, Expressions).
                            lm_203_5, lm_203, _ = (
                                self.worker.function_worker.run_detect_landmark(
                                    img,
                                    bboxes[idx],
                                    kpss_5[idx],
                                    detect_mode="203",
                                    score=0.5,
                                    use_mean_eyes=control.get(
                                        "LandmarkMeanEyesToggle", False
                                    ),
                                    from_points=True,
                                )
                            )
                            kps_203_local = (
                                lm_203
                                if len(lm_203) > 0
                                else np.zeros((203, 2), dtype=np.float32)
                            )

                            # If Step 1 failed and user actually requested 203 for the swap,
                            # ensure we update the Swapper's 5 points right now.
                            if (
                                landmark_mode == "203"
                                and not has_valid_dense_kpss
                                and len(lm_203_5) > 0
                            ):
                                kpss_5[idx] = lm_203_5

                        kpss_203_list.append(kps_203_local)
                kpss_203 = np.array(kpss_203_list, dtype=object)

            # --- STEP 3: Fallback for standard landmarks (UI Display & Swap Alignment) ---
            if use_landmark and not has_valid_dense_kpss and len(bboxes) > 0:
                kpss_list = []
                for idx in range(len(bboxes)):
                    # Smart reuse: If Step 2 just computed 203 landmarks, use them
                    # to avoid a redundant neural network forward pass.
                    if (
                        landmark_mode == "203"
                        and kpss_203 is not None
                        and len(kpss_203) > idx
                    ):
                        kpss_list.append(kpss_203[idx])
                    else:
                        # Respects the UI toggle state for standard UI landmarks
                        lm_std_5, lm_std, _ = (
                            self.worker.function_worker.run_detect_landmark(
                                img,
                                bboxes[idx],
                                kpss_5[idx],
                                detect_mode=landmark_mode,
                                score=control.get("LandmarkDetectScoreSlider", 50)
                                / 100.0,
                                use_mean_eyes=control.get(
                                    "LandmarkMeanEyesToggle", False
                                ),
                                from_points=from_points,
                            )
                        )

                        if len(lm_std) > 0:
                            kpss_list.append(lm_std)
                            if len(lm_std_5) > 0:
                                kpss_5[idx] = lm_std_5
                        else:
                            # int(landmark_mode) used to work because every mode string
                            # was its own point count. The named modes ('3d68',
                            # 'tufa98', 'tufa314', 'orformer98') are not numeric, so the
                            # count comes from the table instead of a ValueError.
                            kpss_list.append(
                                np.zeros(
                                    (landmark_point_counts.get(landmark_mode, 5), 2),
                                    dtype=np.float32,
                                )
                            )
                kpss = np.array(kpss_list, dtype=object)

        if (
            isinstance(kpss_5, np.ndarray)
            and kpss_5.shape[0] > 0
            or isinstance(kpss_5, list)
            and len(kpss_5) > 0
        ):
            safe_bboxes = cast(Any, bboxes)

            _kpss_list = cast(list, kpss) if kpss is not None else []
            _kpss_203_list = cast(list, kpss_203) if kpss_203 is not None else []

            for i in range(len(kpss_5)):
                _bbox_i = safe_bboxes[i]
                if not is_detected_face_eligible_for_matching(
                    kpss_5[i], _bbox_i, self.worker._MIN_FACE_PIXELS
                ):
                    continue

                similarity_type = str("Auto")
                face_emb, _ = self.worker.function_worker.run_recognize_direct(
                    img,
                    kpss_5[i],
                    similarity_type,
                    control["RecognitionModelSelection"],
                )

                kps_all_i = _kpss_list[i] if i < len(_kpss_list) else None
                # Extract the 203 points specifically
                kps_203_i = _kpss_203_list[i] if i < len(_kpss_203_list) else None

                det_faces_data_for_display.append(
                    {
                        "kps_5": kpss_5[i].copy() if kpss_5[i] is not None else None,
                        "kps_all": kps_all_i.copy() if kps_all_i is not None else None,
                        "kps_203": kps_203_i.copy() if kps_203_i is not None else None,
                        "embedding": face_emb,
                        "bbox": safe_bboxes[i].copy()
                        if safe_bboxes[i] is not None
                        else None,
                        "original_face": None,
                        "swap_mask": None,
                        "matched_target": None,
                    }
                )

        # Swapping / Editing Loop
        if det_faces_data_for_display:
            if control["SwapOnlyBestMatchEnableToggle"]:
                # --- Branch: Swap Only Best Match ---
                # FW-RACE-01: iterate the snapshot, not the live dict
                for _, target_face in target_faces_snapshot.items():
                    if stop_event.is_set():
                        break

                    # FW-ROBUST-04: use .get() with default_parameters as fallback
                    with self.worker.lock:
                        _default_params = dict(
                            self.worker.main_window.default_parameters.data
                        )
                    # FIX: Force face_id as string to prevent silent parameter fallback
                    face_id_str = str(target_face.face_id)
                    params = ParametersDict(
                        self.worker.parameters.get(face_id_str, _default_params),
                        _default_params,
                    )
                    best_fface, best_score = None, -1.0

                    for fface in det_faces_data_for_display:
                        # FW-BUG-09: pass snapshot; cache result on fface for downstream reuse
                        tgt, tgt_params, score = self.worker._find_best_target_match(
                            fface["embedding"],
                            control,
                            target_faces_snapshot,
                            is_sequentially_tracked=is_sequentially_tracked,
                        )
                        fface["matched_target"] = tgt
                        if tgt and tgt.face_id == target_face.face_id:
                            # FW-ARCH-FIX: Bypass strict UI threshold locally if rescued temporally
                            effective_threshold = (
                                tgt_params["SimilarityThresholdSlider"]
                                if not is_sequentially_tracked
                                else 0.0
                            )
                            if score >= effective_threshold and score > best_score:
                                best_score = score
                                best_fface = fface

                    if best_fface is not None and (
                        swap_button_is_checked_global or edit_button_is_checked_global
                    ):
                        denoiser_on = (
                            control.get(
                                "DenoiserUNetEnableBeforeRestorersToggle", False
                            )
                            or control.get("DenoiserAfterFirstRestorerToggle", False)
                            or control.get("DenoiserAfterRestorersToggle", False)
                        )
                        if (
                            denoiser_on
                            and target_face.assigned_kv_map is None
                            and target_face.assigned_input_faces
                        ):
                            with self.worker.models_processor.model_lock:
                                if target_face.assigned_kv_map is None:
                                    target_face.calculate_assigned_input_embedding()

                        # --- MORPHING: Swap Only Best Match ---
                        source_kps = None
                        if target_face:
                            if target_face.assigned_input_faces:
                                first_input_id = list(
                                    target_face.assigned_input_faces.keys()
                                )[0]
                                store = target_face.assigned_input_faces[first_input_id]
                                source_kps = store.get("kps_5")
                            elif (
                                hasattr(target_face, "assigned_merged_embeddings")
                                and target_face.assigned_merged_embeddings
                            ):
                                first_embed_id = list(
                                    target_face.assigned_merged_embeddings.keys()
                                )[0]
                                store = target_face.assigned_merged_embeddings[
                                    first_embed_id
                                ]
                                source_kps = store.get("kps_5")

                        best_fface["kps_5"] = keypoints_adjustments(
                            best_fface["kps_5"],
                            cast(dict, params),
                            source_kps=source_kps,
                        )

                        s_e = None
                        arcface_model = self.worker.function_worker.get_arcface_model(
                            params["SwapModelSelection"]
                        )
                        if (
                            swap_button_is_checked_global
                            and params["SwapModelSelection"] != "DeepFaceLive (DFM)"
                        ):
                            _reaging_on = params.get("FaceReagingEnableToggle", False)
                            _aged_emb = getattr(target_face, "aged_input_embedding", {})
                            if _reaging_on and _aged_emb:
                                s_e = _aged_emb.get(arcface_model)
                            else:
                                s_e = target_face.assigned_input_embedding.get(
                                    arcface_model
                                )
                            if s_e is not None and np.isnan(s_e).any():
                                s_e = None

                        # --- Early Exit for No-Op Swaps ---
                        # Respect the user's explicit ForceSwapToggle to process blank targets
                        _is_dfm = params["SwapModelSelection"] == "DeepFaceLive (DFM)"
                        _has_input = (s_e is not None) or _is_dfm
                        _force_swap = control.get("ForceSwapToggle", True)
                        if (
                            not _has_input
                            and not edit_button_is_checked_global
                            and not _force_swap
                        ):
                            continue

                        _aged_kv = getattr(target_face, "aged_kv_map", None)
                        _reaging_kv = (
                            _aged_kv
                            if params.get("FaceReagingEnableToggle", False)
                            and _aged_kv is not None
                            else target_face.assigned_kv_map
                        )
                        _params_for_swap_a = self.worker._apply_auto_mouth(
                            cast(dict, params),
                            target_face,
                            face_bbox=best_fface["bbox"],
                        )
                        try:
                            (
                                img,
                                best_fface["original_face"],
                                best_fface["swap_mask"],
                            ) = self.worker.swap_core(
                                img,
                                best_fface["kps_5"],
                                best_fface["kps_all"],
                                kps_203=best_fface.get("kps_203"),
                                s_e=s_e,
                                t_e=target_face.get_embedding(arcface_model),
                                parameters=_params_for_swap_a,
                                control=control,
                                dfm_model_name=params["DFMModelSelection"],
                                kv_map=_reaging_kv,
                            )
                            if edit_button_is_checked_global and any(
                                params[f]
                                for f in (
                                    "FaceMakeupEnableToggle",
                                    "HairMakeupEnableToggle",
                                    "EyeBrowsMakeupEnableToggle",
                                    "LipsMakeupEnableToggle",
                                )
                            ):
                                kps_for_makeup_best = (
                                    best_fface.get("kps_203")
                                    if best_fface.get("kps_203") is not None
                                    else best_fface.get("kps_5")
                                )
                                img = self.worker.function_worker.swap_edit_face_core_makeup(
                                    img, kps_for_makeup_best, params.data, control
                                )
                        except Exception as e:
                            print(
                                f"[ERROR] Standard mode swap_core failed for best face: {e}"
                            )
                            continue  # Ignore this face but save the frame

            else:
                # --- Branch: Swap All Matches ---
                for fface in det_faces_data_for_display:
                    if stop_event.is_set():
                        break
                    # FW-BUG-09: pass the target_faces snapshot to avoid re-iterating live dict
                    best_target, params, _ = self.worker._find_best_target_match(
                        fface["embedding"],
                        control,
                        target_faces_snapshot,
                        is_sequentially_tracked=is_sequentially_tracked,
                    )
                    # FW-BUG-09: cache matched target so downstream helpers can reuse it
                    fface["matched_target"] = best_target

                    if best_target and (
                        swap_button_is_checked_global or edit_button_is_checked_global
                    ):
                        denoiser_on = (
                            control.get(
                                "DenoiserUNetEnableBeforeRestorersToggle", False
                            )
                            or control.get("DenoiserAfterFirstRestorerToggle", False)
                            or control.get("DenoiserAfterRestorersToggle", False)
                        )
                        if (
                            denoiser_on
                            and best_target.assigned_kv_map is None
                            and best_target.assigned_input_faces
                        ):
                            with self.worker.models_processor.model_lock:
                                if (
                                    best_target.assigned_kv_map is None
                                ):  # Double Check inside lock
                                    best_target.calculate_assigned_input_embedding()

                        # --- MORPHING: Branch Swap All Matches ---
                        source_kps = None
                        if best_target:
                            if best_target.assigned_input_faces:
                                first_input_id = list(
                                    best_target.assigned_input_faces.keys()
                                )[0]
                                store = best_target.assigned_input_faces[first_input_id]
                                source_kps = store.get("kps_5")
                            elif (
                                hasattr(best_target, "assigned_merged_embeddings")
                                and best_target.assigned_merged_embeddings
                            ):
                                first_embed_id = list(
                                    best_target.assigned_merged_embeddings.keys()
                                )[0]
                                store = best_target.assigned_merged_embeddings[
                                    first_embed_id
                                ]
                                source_kps = store.get("kps_5")

                        fface["kps_5"] = keypoints_adjustments(
                            fface["kps_5"], params, source_kps=source_kps
                        )

                        arcface_model = self.worker.function_worker.get_arcface_model(
                            params["SwapModelSelection"]
                        )
                        s_e = None
                        if (
                            swap_button_is_checked_global
                            and params["SwapModelSelection"] != "DeepFaceLive (DFM)"
                        ):
                            _reaging_on = params.get("FaceReagingEnableToggle", False)
                            _aged_emb_bt = getattr(
                                best_target, "aged_input_embedding", {}
                            )
                            if _reaging_on and _aged_emb_bt:
                                s_e = _aged_emb_bt.get(arcface_model)
                            else:
                                s_e = best_target.assigned_input_embedding.get(
                                    arcface_model
                                )
                            if s_e is not None and np.isnan(s_e).any():
                                s_e = None

                        # --- Early Exit for No-Op Swaps ---
                        # Respect the user's explicit ForceSwapToggle to process blank targets
                        _is_dfm = params["SwapModelSelection"] == "DeepFaceLive (DFM)"
                        _has_input = (s_e is not None) or _is_dfm
                        _force_swap = control.get("ForceSwapToggle", True)

                        if (
                            not _has_input
                            and not edit_button_is_checked_global
                            and not _force_swap
                        ):
                            continue

                        _aged_kv_bt = getattr(best_target, "aged_kv_map", None)
                        _reaging_kv = (
                            _aged_kv_bt
                            if params.get("FaceReagingEnableToggle", False)
                            and _aged_kv_bt is not None
                            else best_target.assigned_kv_map
                        )
                        _params_for_swap_b = self.worker._apply_auto_mouth(
                            cast(dict, params),
                            best_target,
                            face_bbox=fface["bbox"],
                        )
                        try:
                            img, fface["original_face"], fface["swap_mask"] = (
                                self.worker.swap_core(
                                    img,
                                    fface["kps_5"],
                                    fface["kps_all"],
                                    kps_203=fface.get("kps_203"),
                                    s_e=s_e,
                                    t_e=best_target.get_embedding(arcface_model),
                                    parameters=_params_for_swap_b,
                                    control=control,
                                    dfm_model_name=params["DFMModelSelection"],
                                    kv_map=_reaging_kv,
                                )
                            )
                            if edit_button_is_checked_global and any(
                                params[f]
                                for f in (
                                    "FaceMakeupEnableToggle",
                                    "HairMakeupEnableToggle",
                                    "EyeBrowsMakeupEnableToggle",
                                    "LipsMakeupEnableToggle",
                                )
                            ):
                                kps_for_makeup = (
                                    fface.get("kps_203")
                                    if fface.get("kps_203") is not None
                                    else fface.get("kps_5")
                                )
                                img = self.worker.function_worker.swap_edit_face_core_makeup(
                                    img, kps_for_makeup, params.data, control
                                )
                        except Exception as e:
                            print(
                                f"[ERROR] Standard mode swap_core failed for a face: {e}"
                            )
                            continue

        # Undo Rotation / Scaling
        if control["ManualRotationEnableToggle"]:
            angle = control["ManualRotationAngleSlider"]

            # FW-BUG-FIX 1: Reverse rotation WITH expand=True so the canvas can physically
            # accommodate the restored dimensions (fixes the 90/270 degree crop).
            # Then center crop to strictly restore the original tensor shape.
            img = v2.functional.rotate(
                img,
                angle=-angle,
                interpolation=v2.InterpolationMode.BILINEAR,
                expand=True,  # CRITICAL: Was False, causing the image to be truncated
            )
            img = v2.functional.center_crop(img, (pre_rot_h, pre_rot_w))

            # FW-MATH: Reverse the affine transform on all spatial coordinates.
            rad = math.radians(angle)
            cos_a, sin_a = math.cos(rad), math.sin(rad)

            cx_orig, cy_orig = pre_rot_w / 2.0, pre_rot_h / 2.0
            cx_rot, cy_rot = post_rot_w / 2.0, post_rot_h / 2.0

            for fface in det_faces_data_for_display:
                if fface.get("_rotation_rescaled"):
                    continue

                def unrotate_pts(pts):
                    if pts is None or len(pts) == 0:
                        return pts
                    # 1. Move origin to the rotated center
                    x0 = pts[:, 0] - cx_rot
                    y0 = pts[:, 1] - cy_rot

                    # 2. Apply 2D inverse rotation matrix (Clockwise in Y-down coordinate system)
                    # BUG FIX: The previous signs were inverted, causing points to rotate further away!
                    x1 = x0 * cos_a - y0 * sin_a
                    y1 = x0 * sin_a + y0 * cos_a

                    # 3. Move back to the original image center
                    pts[:, 0] = x1 + cx_orig
                    pts[:, 1] = y1 + cy_orig
                    return pts

                # Bounding box requires converting to 4 corners, un-rotating, and getting min/max bounds
                if fface.get("bbox") is not None:
                    x1, y1, x2, y2 = fface["bbox"]
                    corners = np.array(
                        [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32
                    )
                    unrot_corners = unrotate_pts(corners)
                    fface["bbox"][0] = np.min(unrot_corners[:, 0])
                    fface["bbox"][1] = np.min(unrot_corners[:, 1])
                    fface["bbox"][2] = np.max(unrot_corners[:, 0])
                    fface["bbox"][3] = np.max(unrot_corners[:, 1])

                # Un-rotate all Keypoint Arrays
                if fface.get("kps_5") is not None:
                    fface["kps_5"] = unrotate_pts(fface["kps_5"])
                if fface.get("kps_all") is not None:
                    fface["kps_all"] = unrotate_pts(fface["kps_all"])
                if fface.get("kps_203") is not None:
                    fface["kps_203"] = unrotate_pts(fface["kps_203"])

                fface["_rotation_rescaled"] = True

        if scale_applied:
            # FW-QUAL-11: use renamed img_h/img_w variables
            # FW-PERF-08 / FW-MEM-02: LRU-bounded cache for the scale-back transform.
            _down_key = (img_h, img_w, self.worker.interpolation_scaleback, False)
            if _down_key in self.worker._resize_cache:
                self.worker._resize_cache.move_to_end(_down_key)
            else:
                if len(self.worker._resize_cache) >= self.worker._RESIZE_CACHE_MAX:
                    self.worker._resize_cache.popitem(last=False)
                self.worker._resize_cache[_down_key] = v2.Resize(
                    (img_h, img_w),
                    interpolation=self.worker.interpolation_scaleback,
                    antialias=False,
                )
            img = self.worker._resize_cache[_down_key](img)

            ratio_w_down = img_w / new_w
            ratio_h_down = img_h / new_h

            for fface in det_faces_data_for_display:
                # Idempotency guard: the same fface dict can survive into a follow-up
                # refresh-frame call (e.g., when the user toggles a setting that
                # triggers refresh_frame). Re-applying the downscale would
                # double-shrink the bbox/kps and produce out-of-frame coordinates
                # that broke auto-mouth detection on stopped frames.
                if fface.get("_displayspace_rescaled"):
                    continue
                if fface.get("bbox") is not None:
                    fface["bbox"][0] *= ratio_w_down
                    fface["bbox"][2] *= ratio_w_down
                    fface["bbox"][1] *= ratio_h_down
                    fface["bbox"][3] *= ratio_h_down
                if fface.get("kps_5") is not None:
                    fface["kps_5"][:, 0] *= ratio_w_down
                    fface["kps_5"][:, 1] *= ratio_h_down
                if fface.get("kps_all") is not None:
                    fface["kps_all"][:, 0] *= ratio_w_down
                    fface["kps_all"][:, 1] *= ratio_h_down
                if fface.get("kps_203") is not None:
                    fface["kps_203"][:, 0] *= ratio_w_down
                    fface["kps_203"][:, 1] *= ratio_h_down
                fface["_displayspace_rescaled"] = True

        processed_tensor_rgb_uint8 = img

        # --- Overlays ---
        if control["ShowAllDetectedFacesBBoxToggle"] and det_faces_data_for_display:
            processed_tensor_rgb_uint8 = draw_bounding_boxes_on_detected_faces(
                processed_tensor_rgb_uint8, det_faces_data_for_display
            )

        if control.get("ShowByteTrackBBoxToggle", False) and det_faces_data_for_display:
            processed_tensor_rgb_uint8 = draw_bounding_boxes_on_detected_faces(
                processed_tensor_rgb_uint8,
                det_faces_data_for_display,
                color_rgb=[255, 165, 0],
            )

        if control["ShowLandmarksEnableToggle"] and det_faces_data_for_display:
            landmarks_data = self._resolve_landmarks_to_draw(
                det_faces_data_for_display, control
            )
            if landmarks_data:
                temp_permuted = processed_tensor_rgb_uint8.permute(1, 2, 0)
                temp_permuted = paint_landmarks_on_image(temp_permuted, landmarks_data)
                processed_tensor_rgb_uint8 = temp_permuted.permute(2, 0, 1)

        compare_mode_active = (
            self.worker.is_view_face_mask or self.worker.is_view_face_compare
        )
        if compare_mode_active and det_faces_data_for_display:
            processed_tensor_rgb_uint8 = self.get_compare_faces_image(
                processed_tensor_rgb_uint8, det_faces_data_for_display, control
            )

        return processed_tensor_rgb_uint8

    def _resolve_landmarks_to_draw(self, det_faces_data: list, control: dict) -> list:
        """
        Helper to determine which landmarks to draw and in what color based on matches.
        """
        landmarks_to_draw = []
        for fface_data in det_faces_data:
            # FW-BUG-09: use cached matched_target if available to avoid re-running match
            cached_tgt = fface_data.get("matched_target")
            if cached_tgt is not None:
                with self.worker.lock:
                    _default_params = dict(
                        self.worker.main_window.default_parameters.data
                    )
                _face_id_1 = getattr(cached_tgt, "face_id", None)
                face_specific: dict = cast(
                    dict,
                    self.worker.parameters.get(str(_face_id_1), {})  # type: ignore[arg-type]
                    if _face_id_1 is not None
                    else {},
                )
                matched_params = ParametersDict(dict(face_specific), _default_params)
            else:
                _, matched_params, _ = self.worker._find_best_target_match(
                    fface_data["embedding"], control
                )
            if matched_params:
                use_adj = matched_params.get("LandmarksPositionAdjEnableToggle", False)

                kps_203 = fface_data.get("kps_203")
                kps_all = fface_data.get("kps_all")
                kps_5 = fface_data.get("kps_5")

                if use_adj:
                    # Manual Adjustment Mode: always force 5 points
                    keypoints = kps_5
                    kcolor = (255, 0, 0)  # BGR: Blue
                else:
                    # Smart Cascade: From most detailed to least detailed
                    if kps_all is not None and len(kps_all) > 5:
                        keypoints = kps_all
                        kcolor = (
                            0,
                            255,
                            255,
                        )  # BGR: Yellow (Standard dense model chosen in UI)
                    elif kps_203 is not None and len(kps_203) == 203:
                        keypoints = kps_203
                        kcolor = (
                            0,
                            165,
                            255,
                        )  # BGR: Orange (Indicates advanced tools triggered the 203 points)
                    else:
                        keypoints = kps_5
                        kcolor = (0, 255, 0)  # BGR: Green (Base 5 points model)

                if keypoints is not None:
                    landmarks_to_draw.append({"kps": keypoints, "color": kcolor})

        return landmarks_to_draw

    def get_compare_faces_image(
        self, img: torch.Tensor, det_faces_data: list, control: dict
    ) -> torch.Tensor:
        """
        Builds a side-by-side comparison image for all detected faces.

        For each detected face that has a matched target, creates a horizontal strip
        containing: [original_face | swapped_face | swap_mask (if available)].
        All strips are vertically stacked and returned as a single CHW tensor.
        Returns the original *img* unchanged if no matched faces are found.
        """
        imgs_to_vstack = []
        for _, fface in enumerate(det_faces_data):
            # FW-BUG-09 / FW-PERF-01/02/03: use cached match result when available
            cached_tgt = fface.get("matched_target")
            if cached_tgt is not None:
                with self.worker.lock:
                    _default_params = dict(
                        self.worker.main_window.default_parameters.data
                    )
                _face_id_2 = getattr(cached_tgt, "face_id", None)
                face_specific: dict = cast(
                    dict,
                    self.worker.parameters.get(str(_face_id_2), {})  # type: ignore[arg-type]
                    if _face_id_2 is not None
                    else {},
                )
                parameters_for_face = ParametersDict(
                    dict(face_specific), _default_params
                )
                best_target_for_compare = cached_tgt
            else:
                best_target_for_compare, parameters_for_face, _ = (
                    self.worker._find_best_target_match(fface["embedding"], control)
                )
            if best_target_for_compare and parameters_for_face:
                modified_face = self.get_cropped_face_using_kps(
                    img, fface["kps_5"], cast(dict, parameters_for_face)
                )
                # FW-PERF-03: skip enhance_core in compare view — diagnostic only,
                # enhancement is too expensive to run twice per face.
                imgs_to_cat_horizontally = []
                original_face_from_swap_core = fface.get("original_face")
                if original_face_from_swap_core is not None:
                    imgs_to_cat_horizontally.append(
                        original_face_from_swap_core.permute(2, 0, 1)
                    )
                imgs_to_cat_horizontally.append(modified_face)
                swap_mask_from_swap_core = fface.get("swap_mask")
                if swap_mask_from_swap_core is not None:
                    mask_chw = swap_mask_from_swap_core.permute(2, 0, 1)
                    if mask_chw.shape[0] == 1:
                        mask_chw = mask_chw.repeat(3, 1, 1)
                    imgs_to_cat_horizontally.append(mask_chw)
                if imgs_to_cat_horizontally:
                    min_h = min(t.shape[1] for t in imgs_to_cat_horizontally)
                    resized_imgs_to_cat = []
                    for t_img in imgs_to_cat_horizontally:
                        if t_img.shape[1] != min_h:
                            aspect_ratio = t_img.shape[2] / t_img.shape[1]
                            new_w = (
                                int(min_h * aspect_ratio)
                                if aspect_ratio > 0
                                else t_img.shape[2]
                            )
                            resized_imgs_to_cat.append(
                                v2.Resize((min_h, new_w), antialias=True)(t_img)
                            )
                        else:
                            resized_imgs_to_cat.append(t_img)
                    imgs_to_vstack.append(torch.cat(resized_imgs_to_cat, dim=2))

        if imgs_to_vstack:
            max_width_for_vstack = max(
                img_strip.size(2) for img_strip in imgs_to_vstack
            )
            padded_strips_for_vstack = [
                torch.nn.functional.pad(
                    img_strip, (0, max_width_for_vstack - img_strip.size(2), 0, 0)
                )
                for img_strip in imgs_to_vstack
            ]
            return torch.cat(padded_strips_for_vstack, dim=1)
        return img

    @torch.no_grad()
    def get_cropped_face_using_kps(
        self,
        img: torch.Tensor,
        kps_5: np.ndarray,
        parameters: dict,
        interp_mode: str = "bilinear",
    ) -> torch.Tensor:
        """
        Aligns and crops a 512x512 face patch from *img* using the 5-point keypoints.
        OPTIMIZED: Uses Kornia to warp directly on the GPU, avoiding CPU decomposition.

        Args:
            img:         Full-frame CHW uint8 tensor.
            kps_5:       5-point facial keypoints (numpy array, shape [5, 2]).
            parameters:  Per-face parameter dict containing at least ``SwapModelSelection``.
            interp_mode: Interpolation mode for warp_affine (e.g. "bilinear" or "bicubic").

        Returns:
            CHW uint8 tensor of shape [3, 512, 512].
        """
        tform = self.worker.get_face_similarity_tform(
            parameters["SwapModelSelection"], kps_5
        )

        M_tensor = (
            torch.from_numpy(cast(np.ndarray, tform.params)[0:2])
            .float()
            .unsqueeze(0)
            .to(img.device)
        )

        # Cast to float32 for Kornia
        img_b = img.unsqueeze(0) if img.dim() == 3 else img
        img_b_float = img_b.float()

        face_512_aligned = kgm.warp_affine(
            img_b_float,
            M_tensor,
            dsize=(512, 512),
            mode=interp_mode,
            align_corners=True,
        ).squeeze(0)

        # Convert back to original dtype (uint8)
        return face_512_aligned.to(img.dtype)
