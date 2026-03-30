from typing import TYPE_CHECKING

import torch
import numpy as np
from torchvision.transforms import v2
import kornia.geometry.transform as kgm

from app.processors.utils import faceutil

if TYPE_CHECKING:
    from app.processors.models_processor import ModelsProcessor


class FrameEdits:
    """
    Manages Face Editing operations (Expression restoration, LivePortrait editing, Makeup).
    Moves high-level processing logic out of FrameWorker.

    This class handles the 'LivePortrait' pipeline, including:
    - Motion Extraction (Pose, Expression)
    - Temporal Smoothing (SmartSmoother & OneEuroFilter)
    - Feature Retargeting (Eyes, Lips)
    - Image Warping and Pasting
    """

    def __init__(self, models_processor: "ModelsProcessor"):
        """
        Initializes the FrameEdits class.

        Args:
            models_processor: Reference to the central model manager (provides models & device).
        """
        self.models_processor = models_processor

        # Transforms will be updated per frame/settings via set_transforms
        self.t256_face: v2.Resize = v2.Resize(
            (256, 256),
            interpolation=v2.InterpolationMode.BILINEAR,
        )
        self.interpolation_expression_faceeditor_back = None

    def set_transforms(self, t256_face, interpolation_expression_faceeditor_back):
        """
        Updates the scaling transforms and interpolation modes based on current control settings.
        Called from FrameWorker.set_scaling_transforms.

        Always sets self.t256_face so callers never fall back to a lazy init that
        ignores the requested interpolation mode.
        """
        if t256_face is not None:
            self.t256_face = t256_face
        else:
            self.t256_face = v2.Resize(
                (256, 256),
                interpolation=v2.InterpolationMode.BILINEAR,
                antialias=False,
            )
        self.interpolation_expression_faceeditor_back = (
            interpolation_expression_faceeditor_back
        )

    def _apply_kornia_warp(
        self, out: torch.Tensor, M_c2o: np.ndarray, dsize: tuple
    ) -> torch.Tensor:
        """
        Internal helper to warp and paste the processed face back into the original frame using Kornia.
        Optimized to reduce code duplication and improve PCIe transfer latency.

        Args:
            out: The processed face tensor (C, H, W).
            M_c2o: Transformation matrix (Crop to Original).
            dsize: Destination size (Height, Width) of the full frame.

        Returns:
            torch.Tensor: The warped tensor ready for blending.
        """
        out = faceutil.pad_image_by_size(out, dsize)

        # OPTIMIZED: non_blocking=True prevents the CPU thread from stalling while waiting
        # for the RAM-to-VRAM transfer to complete via the PCIe bus.
        M_c2o_tensor = (
            torch.from_numpy(M_c2o)
            .float()
            .unsqueeze(0)
            .to(out.device, non_blocking=True)
        )

        out_b = out.unsqueeze(0) if out.dim() == 3 else out
        out = kgm.warp_affine(
            out_b,
            M_c2o_tensor,
            dsize=(dsize[0], dsize[1]),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        ).squeeze(0)

        return out

    def apply_face_expression_restorer(
        self,
        driving: torch.Tensor,
        target: torch.Tensor,
        parameters: dict,
        driving_kps: np.ndarray = None,
        target_kps: np.ndarray = None,
    ) -> torch.Tensor:
        """
        Restores the expression of the face using the LivePortrait model pipeline.

        Features:
        - Target Smoothing to reduce global jitter.
        - Ratio Smoothing (Eyes/Lips) to reduce high-frequency jitter on cilia/mouth.
        - Smart Dynamic Boost for Micro-Expressions: Amplifies subtle motions while
          protecting strong expressions from distortion.

        Args:
            driving: The original face tensor (source of expression).
            target: The swapped face tensor (destination of expression).
            parameters: Dictionary containing UI parameters.

        Returns:
            torch.Tensor: The expression-restored face image.
        """
        # SETUP THE ASYNCHRONOUS CONTEXT
        import contextlib

        # VRAM Leak: Creating a new cuda.Stream() per frame fragments the Caching Allocator
        # and explodes VRAM. We use the current stream (inherited from FrameWorker) instead.
        local_stream = (
            torch.cuda.current_stream() if torch.cuda.is_available() else None
        )
        stream_context = (
            torch.cuda.stream(local_stream)
            if local_stream
            else contextlib.nullcontext()
        )

        with stream_context:
            # --- CONFIGURATION ---
            use_mean_eyes = parameters.get("LandmarkMeanEyesToggle", False)
            # Sanitized Mode Selection
            mode_raw = parameters.get("FaceExpressionModeSelection", "Advanced")
            mode = mode_raw.strip() if isinstance(mode_raw, str) else "Advanced"

            # PARAMETER: Micro-Expression Strength
            # 1.0 = Realistic Transfer. < 1.0 = Dampened. > 1.0 = Boosted.
            micro_expression_boost = parameters.get(
                "FaceExpressionMicroExpressionBoostDecimalSlider", 0.50
            )

            # PARAMETER: Neutral Expression Factor (Anti-Surenchère)
            neutral_factor = parameters.get("FaceExpressionNeutralDecimalSlider", 1.0)

            # --- DRIVING FACE PROCESSING ---
            if driving_kps is not None:
                driving_lmk_crop = driving_kps
            else:
                _, driving_lmk_crop, _ = self.models_processor.run_detect_landmark(
                    driving,
                    bbox=np.array([0, 0, 512, 512]),
                    det_kpss=[],
                    detect_mode="203",
                    score=0.5,
                    from_points=False,
                    use_mean_eyes=use_mean_eyes,
                )

            if driving_lmk_crop is None or (
                hasattr(driving_lmk_crop, "__len__") and len(driving_lmk_crop) == 0
            ):
                return target

            interp_mode = (
                self.interpolation_expression_faceeditor_back
                if self.interpolation_expression_faceeditor_back is not None
                else v2.InterpolationMode.BILINEAR
            )

            # Warp driving face
            driving_face_512, _, _ = faceutil.warp_face_by_face_landmark_x(
                driving,
                driving_lmk_crop,
                dsize=512,
                scale=parameters.get("FaceExpressionCropScaleBothDecimalSlider", 2.3),
                vy_ratio=parameters.get(
                    "FaceExpressionVYRatioBothDecimalSlider", -0.125
                ),
                interpolation=interp_mode,
            )

            driving_face_256 = self.t256_face(driving_face_512)

            # Calculate Raw Ratios (Eyes/Lips openness)
            c_d_eyes_lst = faceutil.calc_eye_close_ratio(driving_lmk_crop[None])
            c_d_lip_lst = faceutil.calc_lip_close_ratio(driving_lmk_crop[None])

            # Extract Motion from Driving Face
            x_d_i_info = self.models_processor.lp_motion_extractor(
                driving_face_256, "Human-Face"
            )

            # --- TARGET FACE ---
            target = target.clamp(0, 255).type(torch.uint8)

            if target_kps is not None:
                source_lmk = target_kps
            else:
                _, source_lmk, _ = self.models_processor.run_detect_landmark(
                    target,
                    bbox=np.array([0, 0, 512, 512]),
                    det_kpss=[],
                    detect_mode="203",
                    score=0.5,
                    from_points=False,
                    use_mean_eyes=use_mean_eyes,
                )

            if source_lmk is None or (
                hasattr(source_lmk, "__len__") and len(source_lmk) == 0
            ):
                return target

            target_face_512, M_o2c, M_c2o = faceutil.warp_face_by_face_landmark_x(
                target,
                source_lmk,
                dsize=512,
                scale=parameters.get("FaceExpressionCropScaleBothDecimalSlider", 2.3),
                vy_ratio=parameters.get(
                    "FaceExpressionVYRatioBothDecimalSlider", -0.125
                ),
                interpolation=interp_mode,
            )
            target_face_256 = self.t256_face(target_face_512)
            x_s_info = self.models_processor.lp_motion_extractor(
                target_face_256, "Human-Face"
            )

            # Prepare Target Features
            x_c_s = x_s_info["kp"]
            R_s = faceutil.get_rotation_matrix(
                x_s_info["pitch"], x_s_info["yaw"], x_s_info["roll"]
            )
            f_s = self.models_processor.lp_appearance_feature_extractor(
                target_face_256, "Human-Face"
            )
            x_s = faceutil.transform_keypoint(x_s_info)

            face_editor_type = parameters.get("FaceEditorTypeSelection", "Human-Face")

            # --- ZERO-TRANSLATION PRE-CALCULATION ---
            default_delta_raw = self.models_processor.lp_stitch(
                x_s, x_s, face_editor_type
            )
            default_delta_exp = default_delta_raw[..., :-2].reshape(x_s.shape[0], 21, 3)

            # --- INDICES DEFINITION ---
            brow_indices = [1, 2]  # Eyebrows (elevation, frowning)
            eye_indices = [11, 13, 15, 16, 18]  # Eyes (blinking, gaze direction)
            lip_indices = [3, 6, 12, 14, 17, 19, 20]  # Mouth (lips, smiling, opening)

            # --- Granular breakdown of "General" structural indices ---
            nose_indices = [5]  # Nose tip and bridge
            jaw_indices = [7]  # Lower jaw / chin (critical for talking)
            cheek_indices = [0, 4]  # Lower cheeks (squish when smiling)
            contour_indices = [8, 9]  # Side face contours / Jawline
            head_top_indices = [10]  # Upper forehead / head stability

            # Recombining them into a customizable General list.
            general_indices = []
            if parameters.get("FaceExpressionGeneralNoseToggle", True):
                general_indices.extend(nose_indices)
            if parameters.get("FaceExpressionGeneralJawToggle", True):
                general_indices.extend(jaw_indices)
            if parameters.get("FaceExpressionGeneralCheekToggle", True):
                general_indices.extend(cheek_indices)
            if parameters.get("FaceExpressionGeneralContourToggle", True):
                general_indices.extend(contour_indices)
            if parameters.get("FaceExpressionGeneralHeadToggle", True):
                general_indices.extend(head_top_indices)

            # Anchor
            R_anchor = R_s
            t_anchor = x_s_info["t"].clone()
            t_anchor[..., 2].fill_(0)
            scale_anchor = x_s_info["scale"]

            # Load Lip Array (Neutral reference for lips)
            lp_lip_array = torch.from_numpy(self.models_processor.lp_lip_array).to(
                dtype=torch.float32, device=self.models_processor.device
            )

            # --- SHARED HELPER FUNCTION ---
            def get_component_motion(
                indices,
                driving_exp,
                multiplier,
                extra_delta=0,
                is_relative=False,
                neutral_ref=None,
                use_boost=False,
            ):
                """
                Helper to calculate motion with 'Smart Dynamic Boost' and 'Neutral Factor'.
                """
                delta_local = x_s_info["exp"].clone()

                if is_relative:
                    # Relative Motion Calculation
                    ref = neutral_ref if neutral_ref is not None else 0
                    if isinstance(ref, torch.Tensor) and ref.shape[-2] == 21:
                        ref_part = ref[..., indices, :]
                    else:
                        ref_part = ref

                    # Calculate the raw difference (motion intent)
                    raw_diff = driving_exp[:, indices, :] - ref_part

                    # --- SMART DYNAMIC BOOST ---
                    boost_val = micro_expression_boost if use_boost else 1.0

                    if use_boost and boost_val > 1.0:
                        magnitude = torch.abs(raw_diff)
                        decay = torch.exp(-10.0 * magnitude)
                        dynamic_scale = 1.0 + (boost_val - 1.0) * decay
                        diff = raw_diff * dynamic_scale
                    else:
                        diff = raw_diff * boost_val

                    # --- NEUTRAL FACTOR (Anti-Surenchère) ---
                    diff = diff * neutral_factor

                    delta_local[:, indices, :] = x_s_info["exp"][:, indices, :] + diff
                else:
                    # Absolute Motion
                    target_exp = driving_exp[:, indices, :]
                    current_exp = x_s_info["exp"][:, indices, :]

                    delta_local[:, indices, :] = (
                        current_exp * (1 - neutral_factor) + target_exp * neutral_factor
                    )

                # Projection & Refinement
                x_proj = scale_anchor * (x_c_s @ R_anchor + delta_local) + t_anchor
                raw_delta = self.models_processor.lp_stitch(
                    x_s, x_proj, face_editor_type
                )
                refinement_exp = raw_delta[..., :-2].reshape(x_s.shape[0], 21, 3)

                x_target = x_proj + (refinement_exp - default_delta_exp) + extra_delta
                return (x_target - x_s) * multiplier

            def merge_eye_motion_candidates(
                relative_motion, absolute_motion, normalize_eyes_enabled=False
            ):
                """
                Relative Lids + Retargeted Gaze eye merge:
                - keep horizontal gaze direction from the absolute + retargeted eye motion
                - keep eyelid/blink shape from the relative eye motion
                - blend back some retargeted vertical lid motion so Normalize Eyes
                  and Eyes Multiplier still have visible influence
                """
                merged_motion = relative_motion.clone()

                # Landmark 11/15 X is the clearest eyeball-direction signal.
                # Keep it fully from the absolute + retargeted branch for better gaze stability.
                merged_motion[:, 11, 0] = absolute_motion[:, 11, 0]
                merged_motion[:, 15, 0] = absolute_motion[:, 15, 0]

                # Vertical eye motion carries both lid state and some gaze drift.
                # Blend a limited amount of the retargeted branch back in so the
                # shipped mode still benefits from Normalize Eyes / Eyes Multiplier
                # without losing the relative eyelid feel that made it useful.
                eyelid_blend = 0.45 if normalize_eyes_enabled else 0.30
                eye_center_blend = 0.35 if normalize_eyes_enabled else 0.20

                for idx in (11, 15):
                    merged_motion[:, idx, 1] = torch.lerp(
                        relative_motion[:, idx, 1],
                        absolute_motion[:, idx, 1],
                        eye_center_blend,
                    )

                for idx in (13, 16):
                    merged_motion[:, idx, 1] = torch.lerp(
                        relative_motion[:, idx, 1],
                        absolute_motion[:, idx, 1],
                        eyelid_blend,
                    )

                return merged_motion

            accumulated_motion = torch.zeros_like(x_s)

            # --- MODE PROCESSING ---
            if mode == "Simple":
                # SIMPLE MODE
                driving_multiplier = parameters.get(
                    "FaceExpressionFriendlyFactorDecimalSlider", 1.0
                )

                animation_region = parameters.get(
                    "FaceExpressionAnimationRegionSelection", "all"
                )
                if not animation_region:
                    animation_region = "all"

                has_eyes = "eyes" in animation_region or "all" in animation_region
                has_lips = "lips" in animation_region or "all" in animation_region

                # Lip Normalization Logic (Optional)
                flag_normalize_lip = parameters.get(
                    "FaceExpressionNormalizeLipsEnableToggle", True
                )
                lip_normalize_threshold = parameters.get(
                    "FaceExpressionNormalizeLipsThresholdDecimalSlider", 0.03
                )
                lips_retarget_delta = 0
                if flag_normalize_lip and source_lmk is not None:
                    combined_lip_ratio = faceutil.calc_combined_lip_ratio(
                        c_d_lip_lst, source_lmk, device=self.models_processor.device
                    )
                    if combined_lip_ratio[0][0] >= lip_normalize_threshold:
                        lips_retarget_delta = self.models_processor.lp_retarget_lip(
                            x_s, combined_lip_ratio
                        )

                if has_eyes:
                    accumulated_motion += get_component_motion(
                        eye_indices,
                        x_d_i_info["exp"],
                        driving_multiplier,
                        is_relative=True,
                        use_boost=False,
                    )

                if has_lips:
                    accumulated_motion += get_component_motion(
                        lip_indices,
                        x_d_i_info["exp"],
                        driving_multiplier,
                        extra_delta=lips_retarget_delta,
                        is_relative=True,
                        neutral_ref=lp_lip_array,
                        use_boost=False,
                    )

            else:
                # ADVANCED MODE
                driving_multiplier_eyes = parameters.get(
                    "FaceExpressionFriendlyFactorEyesDecimalSlider", 1.0
                )
                driving_multiplier_lips = parameters.get(
                    "FaceExpressionFriendlyFactorLipsDecimalSlider", 1.0
                )
                driving_multiplier_brows = parameters.get(
                    "FaceExpressionFriendlyFactorBrowsDecimalSlider", 1.0
                )
                driving_multiplier_general = parameters.get(
                    "FaceExpressionFriendlyFactorGeneralDecimalSlider", 1.0
                )

                flag_activate_eyes = parameters.get("FaceExpressionEyesToggle", False)
                flag_activate_lips = parameters.get("FaceExpressionLipsToggle", False)
                flag_activate_brows = parameters.get("FaceExpressionBrowsToggle", False)
                flag_activate_general = parameters.get(
                    "FaceExpressionGeneralToggle", False
                )

                flag_relative_eyes = parameters.get(
                    "FaceExpressionRelativeEyesToggle", False
                )
                flag_retarget_eyes = parameters.get(
                    "FaceExpressionRetargetingEyesBothEnableToggle", False
                )
                flag_stable_gaze_eyes = parameters.get(
                    "FaceExpressionStableGazeEyesToggle", False
                )
                flag_relative_lips = parameters.get(
                    "FaceExpressionRelativeLipsToggle", False
                )
                flag_relative_brows = parameters.get(
                    "FaceExpressionRelativeBrowsToggle", False
                )
                flag_relative_general = parameters.get(
                    "FaceExpressionRelativeGeneralToggle", False
                )

                flag_normalize_eyes = parameters.get(
                    "FaceExpressionNormalizeEyesBothEnableToggle", True
                )
                eyes_normalize_threshold = parameters.get(
                    "FaceExpressionNormalizeEyesThresholdBothDecimalSlider", 0.40
                )
                eyes_normalize_max = parameters.get(
                    "FaceExpressionNormalizeEyesMaxBothDecimalSlider", 0.50
                )
                combined_eyes_ratio_normalize = None

                if flag_normalize_eyes and source_lmk is not None:
                    c_d_eyes_normalize = c_d_eyes_lst
                    eyes_ratio = np.array([c_d_eyes_normalize[0][0]], dtype=np.float32)
                    eyes_ratio_normalize = max(eyes_ratio, 0.10)
                    eyes_ratio_l = min(c_d_eyes_normalize[0][0], eyes_normalize_max)
                    eyes_ratio_r = min(c_d_eyes_normalize[0][1], eyes_normalize_max)
                    eyes_ratio_max = np.array(
                        [[eyes_ratio_l, eyes_ratio_r]], dtype=np.float32
                    )

                    if eyes_ratio_normalize > eyes_normalize_threshold:
                        combined_eyes_ratio_normalize = (
                            faceutil.calc_combined_eye_ratio_norm(
                                eyes_ratio_max,
                                source_lmk,
                                device=self.models_processor.device,
                            )
                        )
                    else:
                        combined_eyes_ratio_normalize = (
                            faceutil.calc_combined_eye_ratio(
                                eyes_ratio_max,
                                source_lmk,
                                device=self.models_processor.device,
                            )
                        )

                if flag_activate_eyes:
                    eyes_retarget_delta = 0
                    if flag_retarget_eyes:
                        eye_mult = parameters.get(
                            "FaceExpressionRetargetingEyesMultiplierBothDecimalSlider",
                            1.0,
                        )

                        if (
                            flag_normalize_eyes
                            and combined_eyes_ratio_normalize is not None
                        ):
                            target_eye_ratio = combined_eyes_ratio_normalize
                        else:
                            target_eye_ratio = faceutil.calc_combined_eye_ratio(
                                c_d_eyes_lst,
                                source_lmk,
                                device=self.models_processor.device,
                            )

                        eyes_retarget_delta = self.models_processor.lp_retarget_eye(
                            x_s, target_eye_ratio * eye_mult, face_editor_type
                        )

                    if (
                        flag_stable_gaze_eyes
                        and flag_relative_eyes
                        and flag_retarget_eyes
                    ):
                        relative_eye_motion = get_component_motion(
                            eye_indices,
                            x_d_i_info["exp"],
                            1.0,
                            is_relative=True,
                            neutral_ref=0,
                            use_boost=True,
                        )
                        absolute_retarget_eye_motion = get_component_motion(
                            eye_indices,
                            x_d_i_info["exp"],
                            1.0,
                            extra_delta=eyes_retarget_delta,
                            is_relative=False,
                            neutral_ref=0,
                            use_boost=True,
                        )
                        accumulated_motion += (
                            merge_eye_motion_candidates(
                                relative_eye_motion,
                                absolute_retarget_eye_motion,
                                normalize_eyes_enabled=flag_normalize_eyes,
                            )
                            * driving_multiplier_eyes
                        )
                    else:
                        accumulated_motion += get_component_motion(
                            eye_indices,
                            x_d_i_info["exp"],
                            driving_multiplier_eyes,
                            extra_delta=eyes_retarget_delta,
                            is_relative=flag_relative_eyes,
                            neutral_ref=0,
                            use_boost=True,
                        )

                if flag_activate_lips:
                    lips_retarget_delta = 0
                    if parameters.get(
                        "FaceExpressionRetargetingLipsBothEnableToggle", False
                    ):
                        lip_mult = parameters.get(
                            "FaceExpressionRetargetingLipsMultiplierBothDecimalSlider",
                            1.0,
                        )
                        c_d_lip = faceutil.calc_combined_lip_ratio(
                            c_d_lip_lst, source_lmk, device=self.models_processor.device
                        )
                        lips_retarget_delta = self.models_processor.lp_retarget_lip(
                            x_s, c_d_lip * lip_mult, face_editor_type
                        )

                    accumulated_motion += get_component_motion(
                        lip_indices,
                        x_d_i_info["exp"],
                        driving_multiplier_lips,
                        extra_delta=lips_retarget_delta,
                        is_relative=flag_relative_lips,
                        neutral_ref=lp_lip_array,
                        use_boost=True,
                    )

                if flag_activate_brows:
                    accumulated_motion += get_component_motion(
                        brow_indices,
                        x_d_i_info["exp"],
                        driving_multiplier_brows,
                        is_relative=flag_relative_brows,
                        neutral_ref=0,
                        use_boost=True,
                    )

                if flag_activate_general and len(general_indices) > 0:
                    accumulated_motion += get_component_motion(
                        general_indices,
                        x_d_i_info["exp"],
                        driving_multiplier_general,
                        is_relative=flag_relative_general,
                        neutral_ref=0,
                        use_boost=True,
                    )

            # --- GENERATE FINAL IMAGE ---
            x_d_i_new = x_s + accumulated_motion

            out = self.models_processor.lp_warp_decode(
                f_s, x_s, x_d_i_new, face_editor_type
            )
            out = torch.squeeze(out).clamp_(0, 1)

            # --- PASTE BACK ---
            dsize = (target.shape[1], target.shape[2])
            out = self._apply_kornia_warp(out, M_c2o, dsize)
            out = out.mul_(255.0).clamp_(0, 255)

        # Sync the stream safely
        if local_stream:
            local_stream.synchronize()

        return out.type(torch.float32)

    def swap_edit_face_core(
        self,
        img: torch.Tensor,
        swap_restorecalc: torch.Tensor,
        parameters: dict,
        control: dict,
        kps_crop: np.ndarray = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Applies Face Editor manipulations (Pose, Gaze, Expression) to the face via manual sliders.
        Optimized: Centralized GPU warp pasting.

        Args:
            img: The original image/frame.
            swap_restorecalc: The reference face for detection.
            parameters: Global parameters dictionary.
            control: UI control dictionary.

        Returns:
            torch.Tensor: The manipulated face image.
        """
        use_mean_eyes = parameters.get("LandmarkMeanEyesToggle", False)
        interp_mode = (
            self.interpolation_expression_faceeditor_back
            if self.interpolation_expression_faceeditor_back is not None
            else v2.InterpolationMode.BILINEAR
        )

        if parameters["FaceEditorEnableToggle"]:
            # SETUP THE ASYNCHRONOUS CONTEXT
            import contextlib

            local_stream = (
                torch.cuda.current_stream() if torch.cuda.is_available() else None
            )
            stream_context = (
                torch.cuda.stream(local_stream)
                if local_stream
                else contextlib.nullcontext()
            )

            with stream_context:
                init_source_eye_ratio = 0.0
                init_source_lip_ratio = 0.0

                # Detection
                if kps_crop is not None:
                    lmk_crop = kps_crop
                else:
                    _, lmk_crop, _ = self.models_processor.run_detect_landmark(
                        swap_restorecalc,
                        bbox=np.array([0, 0, 512, 512]),
                        det_kpss=[],
                        detect_mode="203",
                        score=0.5,
                        from_points=False,
                        use_mean_eyes=use_mean_eyes,
                    )
                source_eye_ratio = faceutil.calc_eye_close_ratio(lmk_crop[None])
                source_lip_ratio = faceutil.calc_lip_close_ratio(lmk_crop[None])
                init_source_eye_ratio = round(float(source_eye_ratio.mean()), 2)
                init_source_lip_ratio = round(float(source_lip_ratio[0][0]), 2)

                # Prepare Image
                original_face_512, M_o2c, M_c2o = faceutil.warp_face_by_face_landmark_x(
                    img,
                    lmk_crop,
                    dsize=512,
                    scale=parameters["FaceEditorCropScaleDecimalSlider"],
                    vy_ratio=parameters["FaceEditorVYRatioDecimalSlider"],
                    interpolation=interp_mode,
                )

                original_face_256 = self.t256_face(original_face_512)

                # Extract features
                x_s_info = self.models_processor.lp_motion_extractor(
                    original_face_256, parameters["FaceEditorTypeSelection"]
                )

                # --- OPTIMIZED 3D ROTATION COMPOSITION ---
                # 1. Get the original 3D rotation matrix of the target face
                R_s_original = faceutil.get_rotation_matrix(
                    x_s_info["pitch"], x_s_info["yaw"], x_s_info["roll"]
                )

                # 2. Extract slider deltas as tensors
                device = self.models_processor.device
                delta_pitch = torch.tensor(
                    [[parameters["HeadPitchSlider"]]],
                    dtype=torch.float32,
                    device=device,
                )
                delta_yaw = torch.tensor(
                    [[parameters["HeadYawSlider"]]], dtype=torch.float32, device=device
                )
                delta_roll = torch.tensor(
                    [[parameters["HeadRollSlider"]]], dtype=torch.float32, device=device
                )

                # 3. Create a rotation matrix strictly for the user's manual adjustments
                R_sliders = faceutil.get_rotation_matrix(
                    delta_pitch, delta_yaw, delta_roll
                )

                # 4. Compose rotations via Matrix Multiplication (R_new = R_sliders @ R_original)
                # This completely eliminates Euler addition distortion (Gimbal Lock) on extreme angles.
                R_d_new = R_sliders @ R_s_original

                f_s_user = self.models_processor.lp_appearance_feature_extractor(
                    original_face_256, parameters["FaceEditorTypeSelection"]
                )
                x_s_user = faceutil.transform_keypoint(x_s_info)

                # --- Create Tensors from Manual Sliders ---
                mov_x = torch.tensor(parameters["XAxisMovementDecimalSlider"]).to(
                    device
                )
                mov_y = torch.tensor(parameters["YAxisMovementDecimalSlider"]).to(
                    device
                )
                mov_z = torch.tensor(parameters["ZAxisMovementDecimalSlider"]).to(
                    device
                )

                eyeball_direction_x = torch.tensor(
                    parameters["EyeGazeHorizontalDecimalSlider"]
                ).to(device)
                eyeball_direction_y = torch.tensor(
                    parameters["EyeGazeVerticalDecimalSlider"]
                ).to(device)
                wink = torch.tensor(parameters["EyeWinkDecimalSlider"]).to(device)
                eyebrow = torch.tensor(parameters["EyeBrowsDirectionDecimalSlider"]).to(
                    device
                )

                smile = torch.tensor(parameters["MouthSmileDecimalSlider"]).to(device)
                lip_variation_zero = torch.tensor(
                    parameters["MouthPoutingDecimalSlider"]
                ).to(device)
                lip_variation_one = torch.tensor(
                    parameters["MouthPursingDecimalSlider"]
                ).to(device)
                lip_variation_two = torch.tensor(
                    parameters["MouthGrinDecimalSlider"]
                ).to(device)
                lip_variation_three = torch.tensor(
                    parameters["LipsCloseOpenSlider"]
                ).to(device)

                x_c_s = x_s_info["kp"]
                delta_new = x_s_info["exp"]
                scale_new = x_s_info["scale"]
                t_new = x_s_info["t"]

                # Calculate New Rotation Matrix
                # R_d_new = (R_d_user @ R_s_user.permute(0, 2, 1)) @ R_s_user

                # --- Apply Modifications to Expression Delta ---
                if eyeball_direction_x != 0 or eyeball_direction_y != 0:
                    delta_new = faceutil.update_delta_new_eyeball_direction(
                        eyeball_direction_x, eyeball_direction_y, delta_new
                    )
                if smile != 0:
                    delta_new = faceutil.update_delta_new_smile(smile, delta_new)
                if wink != 0:
                    delta_new = faceutil.update_delta_new_wink(wink, delta_new)
                if eyebrow != 0:
                    delta_new = faceutil.update_delta_new_eyebrow(eyebrow, delta_new)
                if lip_variation_zero != 0:
                    delta_new = faceutil.update_delta_new_lip_variation_zero(
                        lip_variation_zero, delta_new
                    )
                if lip_variation_one != 0:
                    delta_new = faceutil.update_delta_new_lip_variation_one(
                        lip_variation_one, delta_new
                    )
                if lip_variation_two != 0:
                    delta_new = faceutil.update_delta_new_lip_variation_two(
                        lip_variation_two, delta_new
                    )
                if lip_variation_three != 0:
                    delta_new = faceutil.update_delta_new_lip_variation_three(
                        lip_variation_three, delta_new
                    )
                if mov_x != 0:
                    delta_new = faceutil.update_delta_new_mov_x(-mov_x, delta_new)
                if mov_y != 0:
                    delta_new = faceutil.update_delta_new_mov_y(mov_y, delta_new)

                # Calculate final driving keypoints
                x_d_new = mov_z * scale_new * (x_c_s @ R_d_new + delta_new) + t_new
                eyes_delta, lip_delta = None, None

                # --- Retargeting Sliders (Opening/Closing) ---
                input_eye_ratio = max(
                    min(
                        init_source_eye_ratio
                        + parameters["EyesOpenRatioDecimalSlider"],
                        0.80,
                    ),
                    0.00,
                )
                if input_eye_ratio != init_source_eye_ratio:
                    combined_eye_ratio_tensor = faceutil.calc_combined_eye_ratio(
                        [[float(input_eye_ratio)]],
                        lmk_crop,
                        device=self.models_processor.device,
                    )
                    eyes_delta = self.models_processor.lp_retarget_eye(
                        x_s_user,
                        combined_eye_ratio_tensor,
                        parameters["FaceEditorTypeSelection"],
                    )

                input_lip_ratio = max(
                    min(
                        init_source_lip_ratio
                        + parameters["LipsOpenRatioDecimalSlider"],
                        0.80,
                    ),
                    0.00,
                )
                if input_lip_ratio != init_source_lip_ratio:
                    combined_lip_ratio_tensor = faceutil.calc_combined_lip_ratio(
                        [[float(input_lip_ratio)]],
                        lmk_crop,
                        device=self.models_processor.device,
                    )
                    lip_delta = self.models_processor.lp_retarget_lip(
                        x_s_user,
                        combined_lip_ratio_tensor,
                        parameters["FaceEditorTypeSelection"],
                    )

                # Add retargeting deltas to the main motion
                x_d_new = (
                    x_d_new
                    + (eyes_delta if eyes_delta is not None else 0)
                    + (lip_delta if lip_delta is not None else 0)
                )

                # Optional Stitching
                flag_stitching_retargeting_input: bool = kwargs.get(
                    "flag_stitching_retargeting_input", True
                )
                if flag_stitching_retargeting_input:
                    x_d_new = self.models_processor.lp_stitching(
                        x_s_user, x_d_new, parameters["FaceEditorTypeSelection"]
                    )

                # Generate Image
                out = self.models_processor.lp_warp_decode(
                    f_s_user, x_s_user, x_d_new, parameters["FaceEditorTypeSelection"]
                )
                out = torch.squeeze(out)
                out = out.clamp_(0, 1)

                # --- POST-PROCESSING (Paste Back) ---
                dsize = (img.shape[1], img.shape[2])
                out = self._apply_kornia_warp(out, M_c2o, dsize)

                img = out
                img = img.mul_(255.0).clamp_(0, 255).type(torch.float32)

            # Sync the stream safely
            if local_stream:
                local_stream.synchronize()

        return img

    def swap_edit_face_core_makeup(
        self,
        img: torch.Tensor,
        kps: np.ndarray,
        parameters: dict,
        control: dict,
        **kwargs,
    ) -> torch.Tensor:
        """
        Applies digital makeup to the face using face parser masks.

        Args:
            img: The original image tensor.
            kps: Keypoints of the face.
            parameters: Global parameters dictionary.
            control: Control settings dictionary.

        Returns:
            torch.Tensor: Image with makeup applied.
        """
        use_mean_eyes = parameters.get("LandmarkMeanEyesToggle", False)

        if (
            parameters["FaceMakeupEnableToggle"]
            or parameters["HairMakeupEnableToggle"]
            or parameters["EyeBrowsMakeupEnableToggle"]
            or parameters["LipsMakeupEnableToggle"]
        ):
            if kps is not None and len(kps) > 5:
                lmk_crop = kps
            else:
                _, lmk_crop, _ = self.models_processor.run_detect_landmark(
                    img,
                    bbox=[],
                    det_kpss=kps,
                    detect_mode="203",
                    score=0.5,
                    from_points=False,
                    use_mean_eyes=use_mean_eyes,
                )

            # Use the interpolation mode passed from FrameWorker, or default to BILINEAR
            interp_mode = (
                self.interpolation_expression_faceeditor_back
                if self.interpolation_expression_faceeditor_back is not None
                else v2.InterpolationMode.BILINEAR
            )

            # Prepare Image
            original_face_512, M_o2c, M_c2o = faceutil.warp_face_by_face_landmark_x(
                img,
                lmk_crop,
                dsize=512,
                scale=parameters["FaceEditorCropScaleDecimalSlider"],
                vy_ratio=parameters["FaceEditorVYRatioDecimalSlider"],
                interpolation=interp_mode,
            )

            out, mask_out = self.models_processor.apply_face_makeup(
                original_face_512, parameters
            )

            # Gaussian blur for soft blending of the crop mask
            gauss = v2.GaussianBlur(kernel_size=5 * 2 + 1, sigma=(5 + 1) * 0.2)
            out = torch.clamp(torch.div(out, 255.0), 0, 1).type(torch.float32)
            mask_crop = gauss(self.models_processor.lp_mask_crop)

            # Note: We keep faceutil.paste_back_adv here instead of _apply_kornia_warp
            # because this specific paste operation requires complex alpha blending
            # based on the generated makeup mask_crop, which the generic helper doesn't support yet.
            img = faceutil.paste_back_adv(out, M_c2o, img, mask_crop)

        return img
