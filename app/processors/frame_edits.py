from typing import TYPE_CHECKING, Any

import torch
import numpy as np
from torchvision.transforms import v2
import kornia.geometry.transform as kgm

from app.processors.utils import faceutil

if TYPE_CHECKING:
    from app.processors.models_processor import ModelsProcessor
    from app.processors.workers.function_worker import FunctionWorker


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

    def __init__(
        self,
        models_processor: "ModelsProcessor",
        function_worker: "FunctionWorker",
    ):
        """
        Initializes the FrameEdits class.

        Args:
            models_processor: Reference to the central model manager (provides models & device).
            function_worker: The central FunctionWorker instance. Passed to sub-processors
                              so they can route calls through this Facade.
        """
        self.models_processor = models_processor
        self.function_worker = function_worker

        # Transforms will be updated per frame/settings via set_transforms
        self.t256_face: v2.Resize = v2.Resize(
            (256, 256),
            interpolation=v2.InterpolationMode.BILINEAR,
        )
        self.interpolation_expression_faceeditor_back = None

        # Persistent VRAM cache for Recast feather masks to avoid per-frame allocation
        self._recast_feather_masks: dict = {}

        # Cache for Face Shaping GPU to prevent recreating meshgrids
        self._shaping_cache: dict[str, Any] | None = None

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
        self, out: torch.Tensor, M_c2o: np.ndarray, dsize: tuple[int, int]
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

    @torch.no_grad()
    def apply_face_expression_restorer(
        self,
        driving: torch.Tensor,
        target: torch.Tensor,
        parameters: dict,
        control: dict,
        driving_kps: np.ndarray | None = None,
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

        with stream_context, torch.inference_mode():
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

            # --- DRIVING FACE PROCESSING ---
            if driving_kps is not None and not np.all(driving_kps == 0):
                driving_lmk_crop = driving_kps
            else:
                _, driving_lmk_crop, _ = self.function_worker.run_detect_landmark(
                    driving,
                    bbox=np.array([0, 0, 512, 512]),
                    det_kpss=[],
                    detect_mode="203",
                    score=0.5,
                    from_points=False,
                    use_mean_eyes=use_mean_eyes,
                )
                if not control.get("VR180ModeEnableToggle", False):
                    print(
                        "[WARN] Could not get kps_203, running separate detection on driving face."
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
            x_d_i_info = self.function_worker.lp_motion_extractor(
                driving_face_256, "Human-Face"
            )

            # --- TARGET FACE ---
            target = target.clamp(0, 255).type(torch.uint8)

            # Always run detection on the target face to get the landmarks for warping
            _, source_lmk, _ = self.function_worker.run_detect_landmark(
                target,
                bbox=np.array([0, 0, 512, 512], dtype=np.float32),
                det_kpss=None,
                detect_mode="203",
                score=0.5,
                from_points=False,
                use_mean_eyes=use_mean_eyes,
            )

            if source_lmk is None or (
                hasattr(source_lmk, "__len__") and len(source_lmk) == 0
            ):
                return target.type(torch.float32)

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
            x_s_info = self.function_worker.lp_motion_extractor(
                target_face_256, "Human-Face"
            )

            # Prepare Target Features
            x_c_s = x_s_info["kp"]
            R_s = faceutil.get_rotation_matrix(
                x_s_info["pitch"], x_s_info["yaw"], x_s_info["roll"]
            )
            f_s = self.function_worker.lp_appearance_feature_extractor(
                target_face_256, "Human-Face"
            )
            x_s = faceutil.transform_keypoint(x_s_info)

            face_editor_type = parameters.get("FaceEditorTypeSelection", "Human-Face")

            # --- ZERO-TRANSLATION PRE-CALCULATION ---
            default_delta_raw = self.function_worker.lp_stitch(
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

            # Only send to GPU once by checking if it's already a tensor in the central processor.
            if not hasattr(self, "_cached_lp_lip_tensor"):
                self._cached_lp_lip_tensor = torch.from_numpy(
                    self.function_worker.face_editors.lp_lip_array
                ).to(dtype=torch.float32, device=self.models_processor.device)
            lp_lip_array = self._cached_lp_lip_tensor

            # --- SHARED HELPER FUNCTION ---
            def get_component_motion(
                indices: list[int],
                driving_exp: torch.Tensor,
                multiplier: float,
                extra_delta: torch.Tensor | int | float = 0,
                is_relative: bool = False,
                neutral_ref: torch.Tensor | int | None = None,
                use_boost: bool = False,
                neutral_factor: float = 0.3,
            ) -> torch.Tensor:
                """
                Helper to calculate motion with 'Smart Dynamic Boost' and 'Neutral Factor'.

                - Openness Gate: Ignores closed eyes.
                - Yaw Falloff (Cosine Squared): Boosts strength (+25%) when facing forward,
                  and smoothly attenuates strength on profiles (>45 deg) to prevent fisheye limits.
                """
                delta_local = x_s_info["exp"].clone()
                force_camera_gaze = parameters.get(
                    "FaceExpressionCameraGazeToggle", False
                )

                # If we dampen the expression towards neutral, the retargeting offsets (extra_delta)
                # must be strictly dampened by the same ratio to prevent decoupled floating features.
                if isinstance(extra_delta, torch.Tensor) or extra_delta != 0:
                    extra_delta = extra_delta * neutral_factor

                # 1. COMPUTE BASE EXPRESSION DELTA (Relative or Absolute)
                if is_relative:
                    ref = neutral_ref if neutral_ref is not None else 0
                    if isinstance(ref, torch.Tensor) and ref.shape[-2] == 21:
                        ref_part = ref[..., indices, :]
                    else:
                        ref_part = ref

                    raw_diff = driving_exp[:, indices, :] - ref_part

                    boost_val = micro_expression_boost if use_boost else 1.0
                    if use_boost and boost_val > 1.0:
                        magnitude = torch.abs(raw_diff)
                        decay = torch.exp(-10.0 * magnitude)
                        noise_gate = torch.clamp(magnitude / 0.005, 0.0, 1.0)
                        dynamic_scale = 1.0 + (boost_val - 1.0) * decay * noise_gate
                        diff = raw_diff * dynamic_scale
                    else:
                        diff = raw_diff * boost_val

                    diff = diff * neutral_factor
                    delta_local[:, indices, :] = x_s_info["exp"][:, indices, :] + diff

                else:
                    target_exp = driving_exp[:, indices, :]
                    current_exp = x_s_info["exp"][:, indices, :]
                    delta_local[:, indices, :] = (
                        current_exp * (1 - neutral_factor) + target_exp * neutral_factor
                    )

                # 2. HYBRID GAZE STABILIZATION PIPELINE
                if 11 in indices and 15 in indices:
                    idx_11, idx_15 = indices.index(11), indices.index(15)

                    if force_camera_gaze:
                        import math

                        # --- A. OPENNESS GATE (Blink protection) ---
                        current_eye_openness = max(
                            c_d_eyes_lst[0][0], c_d_eyes_lst[0][1]
                        )
                        openness_gate = max(
                            0.0, min(1.0, (current_eye_openness - 0.15) / 0.07)
                        )

                        # --- B. YAW FALLOFF (Profile Attenuation & Frontal Boost) ---
                        # Extract the real-time horizontal rotation (Yaw) of the face
                        head_yaw_deg = faceutil.headpose_pred_to_degree(
                            x_s_info["yaw"]
                        ).item()
                        abs_yaw = abs(head_yaw_deg)

                        # Quadratic Cosine Falloff:
                        # 0° = 1.25x | 30° = ~0.93x | 45° = 0.62x | 60° = 0.31x
                        yaw_gate = 1.25 * (
                            math.cos(math.radians(min(abs_yaw, 90.0))) ** 2
                        )

                        # --- C. FETCH UI PARAMETERS & CALCULATE FINAL STRENGTH ---
                        ui_strength = parameters.get(
                            "FaceExpressionCameraGazeStrengthDecimalSlider", 0.50
                        )
                        vertical_offset_ui = parameters.get(
                            "FaceExpressionCameraGazeVerticalOffsetDecimalSlider",
                            0.0,
                        )

                        # Multiply constraints and STRICTLY CLAMP at 1.0 for the lerp function
                        active_strength = ui_strength * openness_gate * yaw_gate
                        active_strength = max(0.0, min(1.0, active_strength))

                        # Only process heavy math if the gaze lock has a tangible effect
                        if active_strength > 0.01:
                            # D. 3D Z-Axis Projection
                            cam_world = torch.tensor(
                                [0.0, 0.0, 1.0],
                                dtype=torch.float32,
                                device=delta_local.device,
                            )
                            R_inv = R_anchor.squeeze(0).transpose(0, 1)
                            cam_local = torch.matmul(R_inv, cam_world)

                            # E. DYNAMIC PERCEPTUAL COMPENSATION (Pitch)
                            head_pitch_deg = faceutil.headpose_pred_to_degree(
                                x_s_info["pitch"]
                            ).item()
                            auto_perceptual_offset = head_pitch_deg * -0.0003

                            # F. CALCULATE IDEAL ABSOLUTE LATENTS
                            ideal_gaze_x = cam_local[0].item() * 0.035
                            ideal_gaze_y = (
                                (cam_local[1].item() * 0.020)
                                + auto_perceptual_offset
                                + (vertical_offset_ui * 0.015)
                            )

                            # G. STRICT SOFT CLAMPING (Prevents mesh tearing)
                            safe_ideal_x = (
                                0.040 * math.tanh(ideal_gaze_x / 0.040)
                                if ideal_gaze_x != 0
                                else 0.0
                            )
                            safe_ideal_y = (
                                0.020 * math.tanh(ideal_gaze_y / 0.020)
                                if ideal_gaze_y != 0
                                else 0.0
                            )

                            safe_ideal_x_t = torch.tensor(
                                safe_ideal_x,
                                dtype=torch.float32,
                                device=delta_local.device,
                            )
                            safe_ideal_y_t = torch.tensor(
                                safe_ideal_y,
                                dtype=torch.float32,
                                device=delta_local.device,
                            )

                            # H. SAVE INTENDED Y FOR EYELID CALCULATION
                            intended_y_11 = delta_local[:, 11, 1].clone()
                            intended_y_15 = delta_local[:, 15, 1].clone()

                            # I. PURE INTERPOLATION (Using the modulated strength)
                            delta_local[:, 11, 0] = torch.lerp(
                                delta_local[:, 11, 0], safe_ideal_x_t, active_strength
                            )
                            delta_local[:, 15, 0] = torch.lerp(
                                delta_local[:, 15, 0], safe_ideal_x_t, active_strength
                            )

                            delta_local[:, 11, 1] = torch.lerp(
                                delta_local[:, 11, 1], safe_ideal_y_t, active_strength
                            )
                            delta_local[:, 15, 1] = torch.lerp(
                                delta_local[:, 15, 1], safe_ideal_y_t, active_strength
                            )

                            # J. SOFT EYELID COMPENSATION
                            if 13 in indices and 16 in indices:
                                shift_y_11 = delta_local[:, 11, 1] - intended_y_11
                                shift_y_15 = delta_local[:, 15, 1] - intended_y_15

                                delta_local[:, 13, 1] += shift_y_11 * 0.35
                                delta_local[:, 16, 1] += shift_y_15 * 0.35

                    else:
                        # --- FALLBACK: Standard Dampening ---
                        if is_relative:
                            gaze_dampening = 0.50
                            delta_local[:, 11, 0] = x_s_info["exp"][:, 11, 0] + (
                                diff[:, idx_11, 0] * gaze_dampening
                            )
                            delta_local[:, 15, 0] = x_s_info["exp"][:, 15, 0] + (
                                diff[:, idx_15, 0] * gaze_dampening
                            )
                        else:
                            gaze_x_blend = 0.60
                            delta_local[:, 11, 0] = torch.lerp(
                                current_exp[:, idx_11, 0],
                                target_exp[:, idx_11, 0],
                                gaze_x_blend,
                            )
                            delta_local[:, 15, 0] = torch.lerp(
                                current_exp[:, idx_15, 0],
                                target_exp[:, idx_15, 0],
                                gaze_x_blend,
                            )

                # 3. PROJECTION & REFINEMENT
                x_proj = scale_anchor * (x_c_s @ R_anchor + delta_local) + t_anchor
                raw_delta = self.function_worker.lp_stitch(
                    x_s, x_proj, face_editor_type
                )
                refinement_exp = raw_delta[..., :-2].reshape(x_s.shape[0], 21, 3)

                x_target = x_proj + (refinement_exp - default_delta_exp) + extra_delta
                return (x_target - x_s) * multiplier

            def merge_eye_motion_candidates(
                relative_motion: torch.Tensor,
                absolute_motion: torch.Tensor,
                normalize_eyes_enabled: bool = False,
            ) -> torch.Tensor:
                """
                Relative Lids + Retargeted Gaze eye merge:
                - keep horizontal gaze direction from the absolute + retargeted eye motion
                - keep eyelid/blink shape from the relative eye motion
                - blend back some retargeted vertical lid motion so Normalize Eyes
                  and Eyes Multiplier still have visible influence
                """
                merged_motion = relative_motion.clone()

                # --- GAZE PRECISION (X-Axis) ---
                # 50% Absolute provides enough authority to direct the iris precisely,
                # while leaving 50% Relative to prevent IPD tearing on profile angles.
                gaze_blend = 0.50
                merged_motion[:, 11, 0] = torch.lerp(
                    relative_motion[:, 11, 0], absolute_motion[:, 11, 0], gaze_blend
                )
                merged_motion[:, 15, 0] = torch.lerp(
                    relative_motion[:, 15, 0], absolute_motion[:, 15, 0], gaze_blend
                )

                # Vertical eye motion carries both lid state and some gaze drift.
                # Blend a limited amount of the retargeted branch back in so the
                # shipped mode still benefits from Normalize Eyes / Eyes Multiplier
                # without losing the relative eyelid feel that made it useful.
                eyelid_blend = 0.45 if normalize_eyes_enabled else 0.30
                eye_center_blend = 0.35 if normalize_eyes_enabled else 0.20

                # 11, 15: Eye Centers (Iris vertical position & depth)
                for idx in (11, 15):
                    merged_motion[:, idx, 1] = torch.lerp(
                        relative_motion[:, idx, 1],
                        absolute_motion[:, idx, 1],
                        eye_center_blend,
                    )
                    merged_motion[:, idx, 2] = torch.lerp(
                        relative_motion[:, idx, 2],
                        absolute_motion[:, idx, 2],
                        eye_center_blend,
                    )

                # 13, 16: Eyelids (Blink & Squint)
                for idx in (13, 16):
                    merged_motion[:, idx, 1] = torch.lerp(
                        relative_motion[:, idx, 1],
                        absolute_motion[:, idx, 1],
                        eyelid_blend,
                    )
                    merged_motion[:, idx, 2] = torch.lerp(
                        relative_motion[:, idx, 2],
                        absolute_motion[:, idx, 2],
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

                # PARAMETER: Neutral Expression Factor
                neutral_factor = parameters.get(
                    "FaceExpressionNeutralDecimalSlider", 1.0
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
                        lips_retarget_delta = self.function_worker.lp_retarget_lip(
                            x_s, combined_lip_ratio
                        )

                if has_eyes:
                    accumulated_motion += get_component_motion(
                        eye_indices,
                        x_d_i_info["exp"],
                        driving_multiplier,
                        is_relative=True,
                        use_boost=False,
                        neutral_factor=neutral_factor,
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
                        neutral_factor=neutral_factor,
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

                # PARAMETER: Neutral Expression Factor
                neutral_factor_eyes = parameters.get(
                    "FaceExpressionNeutral_EyesDecimalSlider", 0.3
                )
                neutral_factor_lips = parameters.get(
                    "FaceExpressionNeutral_LipsDecimalSlider", 0.3
                )
                neutral_factor_brows = parameters.get(
                    "FaceExpressionNeutral_BrowsDecimalSlider", 0.3
                )
                neutral_factor_general = parameters.get(
                    "FaceExpressionNeutral_GeneralDecimalSlider", 0.3
                )

                # --- EYE NORMALIZATION PRE-PROCESSING ---
                # Default baseline is the raw driving ratio
                eyes_target_array = c_d_eyes_lst

                if flag_normalize_eyes and source_lmk is not None:
                    # Check if the overall eye openness exceeds the user's threshold
                    current_max_openness = max(c_d_eyes_lst[0][0], c_d_eyes_lst[0][1])

                    if current_max_openness > eyes_normalize_threshold:
                        # Clamp both eyes independently to the max allowed value
                        # This prevents the "surprised" look while preserving winks
                        eyes_target_array = np.array(
                            [
                                [
                                    min(c_d_eyes_lst[0][0], eyes_normalize_max),
                                    min(c_d_eyes_lst[0][1], eyes_normalize_max),
                                ]
                            ],
                            dtype=np.float32,
                        )

                if flag_activate_eyes:
                    eyes_retarget_delta = 0
                    if flag_retarget_eyes:
                        eye_mult = parameters.get(
                            "FaceExpressionRetargetingEyesMultiplierBothDecimalSlider",
                            1.0,
                        )

                        # 1. Get Independent Tensors for each eye to feed into the MLPs
                        ratio_left, ratio_right = faceutil.calc_independent_eye_ratios(
                            eyes_target_array,
                            source_lmk,
                            device=self.models_processor.device,
                        )

                        # 2. Double MLP Inference
                        delta_left_sym = self.function_worker.lp_retarget_eye(
                            x_s, ratio_left * eye_mult, face_editor_type
                        )
                        delta_right_sym = self.function_worker.lp_retarget_eye(
                            x_s, ratio_right * eye_mult, face_editor_type
                        )

                        # 3. Latent Splicing: Stitch Left and Right expressions
                        # Indices: 15 (Right pupil/center), 16 (Right eyelid)
                        eyes_retarget_delta = delta_left_sym.clone()
                        eyes_retarget_delta[:, [15, 16], :] = delta_right_sym[
                            :, [15, 16], :
                        ]

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
                            neutral_factor=neutral_factor_eyes,
                        )
                        absolute_retarget_eye_motion = get_component_motion(
                            eye_indices,
                            x_d_i_info["exp"],
                            1.0,
                            extra_delta=eyes_retarget_delta,
                            is_relative=False,
                            neutral_ref=0,
                            use_boost=True,
                            neutral_factor=neutral_factor_eyes,
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
                            neutral_factor=neutral_factor_eyes,
                        )

                if flag_activate_lips:
                    lips_retarget_delta = 0
                    flag_retarget_lips = parameters.get(
                        "FaceExpressionRetargetingLipsBothEnableToggle", False
                    )

                    if flag_retarget_lips:
                        lip_mult = parameters.get(
                            "FaceExpressionRetargetingLipsMultiplierBothDecimalSlider",
                            1.0,
                        )
                        c_d_lip = faceutil.calc_combined_lip_ratio(
                            c_d_lip_lst, source_lmk, device=self.models_processor.device
                        )
                        lips_retarget_delta = self.function_worker.lp_retarget_lip(
                            x_s, c_d_lip * lip_mult, face_editor_type
                        )

                    if flag_relative_lips and flag_retarget_lips:
                        # 1. Pure Relative Branch: Captures shape (smirk, width, pout) on X-axis
                        relative_lip_motion = get_component_motion(
                            lip_indices,
                            x_d_i_info["exp"],
                            1.0,
                            extra_delta=0,  # No retargeting here
                            is_relative=True,
                            neutral_ref=lp_lip_array,
                            use_boost=True,
                            neutral_factor=neutral_factor_lips,
                        )

                        # 2. Pure Absolute Branch: Captures precise jaw drop and mouth opening on Y/Z-axis
                        absolute_retarget_lip_motion = get_component_motion(
                            lip_indices,
                            x_d_i_info["exp"],
                            1.0,
                            extra_delta=lips_retarget_delta,
                            is_relative=False,
                            neutral_ref=lp_lip_array,
                            use_boost=True,
                            neutral_factor=neutral_factor_lips,
                        )

                        # 3. Structural Decoupling Merge (Softened)
                        # We use Lerp to blend Relative and Absolute on the Y axis.
                        # 0.5 means 50% relative influence, 50% retargeting influence.
                        merged_lip_motion = relative_lip_motion.clone()
                        blend_factor = 0.50

                        for idx in lip_indices:
                            merged_lip_motion[:, idx, 1] = torch.lerp(
                                relative_lip_motion[:, idx, 1],
                                absolute_retarget_lip_motion[:, idx, 1],
                                blend_factor,
                            )
                            merged_lip_motion[:, idx, 2] = absolute_retarget_lip_motion[
                                :, idx, 2
                            ]  # Depth stays absolute

                        accumulated_motion += (
                            merged_lip_motion * driving_multiplier_lips
                        )
                    else:
                        # Standard behavior if only one mode (or neither) is used
                        accumulated_motion += get_component_motion(
                            lip_indices,
                            x_d_i_info["exp"],
                            driving_multiplier_lips,
                            extra_delta=lips_retarget_delta,
                            is_relative=flag_relative_lips,
                            neutral_ref=lp_lip_array,
                            use_boost=True,
                            neutral_factor=neutral_factor_lips,
                        )

                if flag_activate_brows:
                    accumulated_motion += get_component_motion(
                        brow_indices,
                        x_d_i_info["exp"],
                        driving_multiplier_brows,
                        is_relative=flag_relative_brows,
                        neutral_ref=0,
                        use_boost=True,
                        neutral_factor=neutral_factor_brows,
                    )

                if flag_activate_general and len(general_indices) > 0:
                    accumulated_motion += get_component_motion(
                        general_indices,
                        x_d_i_info["exp"],
                        driving_multiplier_general,
                        is_relative=flag_relative_general,
                        neutral_ref=0,
                        use_boost=True,
                        neutral_factor=neutral_factor_general,
                    )

            # --- GENERATE FINAL IMAGE ---
            x_d_i_new = x_s + accumulated_motion

            out = self.function_worker.lp_warp_decode(
                f_s, x_s, x_d_i_new, face_editor_type
            )
            out = torch.squeeze(out).clamp_(0, 1)

            # --- PASTE BACK ---
            dsize = (target.shape[1], target.shape[2])

            # 1. Create a tracking mask, shaving 2 pixels off the outer edge to
            # prevent Kornia's bilinear boundary interpolation from creating a seam.
            warp_mask = torch.zeros(
                (1, out.shape[1], out.shape[2]), dtype=out.dtype, device=out.device
            )
            warp_mask[:, 2:-2, 2:-2] = 1.0
            warp_mask = self._apply_kornia_warp(warp_mask, M_c2o, dsize)

            # 2. Warp the edited face back to the target's coordinate space
            out = self._apply_kornia_warp(out, M_c2o, dsize)
            out = out.mul_(255.0).clamp_(0, 255)

            # 3. Composite over the original target to preserve the outer background pixels!
            target_float = target.type(torch.float32)
            out = out * warp_mask + target_float * (1.0 - warp_mask)

        return out.type(torch.float32)

    @torch.no_grad()
    def apply_perform_recast(
        self,
        driving: torch.Tensor,
        target: torch.Tensor,
        parameters: dict,
        control: dict,
        driving_kps: np.ndarray | None = None,
    ) -> torch.Tensor:
        """Transfer expression onto the swapped face using PerformRecast.

        This is the "Recast" mode of the Face Expression Restorer. Unlike the
        LivePortrait path it uses the PerformRecast 49-keypoint sub-networks
        (appearance F, motion M, warping W, SPADE generator G) and composes the
        expression with either the Enhancement (additive delta) or Replacement
        (absolute driving expression with source-side identity blending) mode.

        Args:
            driving: The original pre-swap face tensor (source of expression).
            target: The swapped face tensor (identity/pose to preserve).
            parameters: Dictionary containing UI parameters.
            control: UI control dictionary.
            driving_kps: Optional pre-computed 203-point driving landmarks.

        Returns:
            torch.Tensor: The expression-recast face image (float32, [0,255]).
        """
        import contextlib

        # Use the current stream (see apply_face_expression_restorer for why a
        # per-frame cuda.Stream() fragments the allocator).
        local_stream = (
            torch.cuda.current_stream() if torch.cuda.is_available() else None
        )
        stream_context = (
            torch.cuda.stream(local_stream)
            if local_stream
            else contextlib.nullcontext()
        )

        # Eagerly load (and build TensorRT engines for) all four sub-networks so
        # the build dialog is shown for every PerformRecast model, not just the
        # ones that happen to need a lazy first-run build.
        self.function_worker.perform_recast_prewarm()

        with stream_context, torch.inference_mode():
            use_mean_eyes = parameters.get("LandmarkMeanEyesToggle", False)

            mode = parameters.get("RecastModeSelection", "Enhancement")
            if isinstance(mode, str):
                mode = mode.strip()
            factor = float(parameters.get("RecastExpressionFactorDecimalSlider", 1.0))
            region = parameters.get("RecastAnimationRegionSelection", "all")
            eye_weight = float(
                parameters.get("RecastEyeDrivingWeightDecimalSlider", 1.0)
            )
            lip_weight = float(
                parameters.get("RecastLipDrivingWeightDecimalSlider", 1.0)
            )
            brows_weight = float(
                parameters.get("RecastBrowsDrivingWeightDecimalSlider", 1.0)
            )
            cheeks_weight = float(
                parameters.get("RecastCheeksDrivingWeightDecimalSlider", 0.20)
            )
            jaw_weight = float(
                parameters.get("RecastJawDrivingWeightDecimalSlider", 0.15)
            )
            feather_amount = float(
                parameters.get("RecastPasteBackFeatherDecimalSlider", 0.0)
            )
            structural_blend = float(
                parameters.get("RecastRelativeStructuralBlendDecimalSlider", 0.50)
            )
            crop_scale = float(
                parameters.get("FaceExpressionCropScaleBothDecimalSlider", 1.5)
            )
            vy_ratio = float(
                parameters.get("FaceExpressionVYRatioBothDecimalSlider", -0.125)
            )
            interp_mode = (
                self.interpolation_expression_faceeditor_back
                if self.interpolation_expression_faceeditor_back is not None
                else v2.InterpolationMode.BILINEAR
            )

            # Detection bboxes are derived from the actual tensor size rather
            # than hardcoded to 512. In VR180 mode the face tensor handed to
            # Recast is not guaranteed to be exactly 512x512, and a mismatched
            # bbox produced degenerate landmarks / crops.
            d_h, d_w = int(driving.shape[-2]), int(driving.shape[-1])

            # --- DRIVING FACE (source of expression) ---
            if driving_kps is not None and not np.all(driving_kps == 0):
                driving_lmk_crop = driving_kps
            else:
                _, driving_lmk_crop, _ = self.function_worker.run_detect_landmark(
                    driving,
                    bbox=np.array([0, 0, d_w, d_h], dtype=np.float32),
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

            driving_face_512, _, _ = faceutil.warp_face_by_face_landmark_x(
                driving,
                driving_lmk_crop,
                dsize=512,
                scale=crop_scale,
                vy_ratio=vy_ratio,
                interpolation=interp_mode,
                # Replicate edge pixels instead of filling out-of-bounds samples
                # with black — prevents all-black face crops when the crop window
                # extends past the image (the VR180 black-crop bug).
                padding_mode="border",
            )
            driving_face_256 = self.t256_face(driving_face_512)

            x_d_info = self.function_worker.perform_recast_motion(driving_face_256)
            exp_d = x_d_info["exp"]

            # --- TARGET FACE (identity / pose to preserve) ---
            target = target.clamp(0, 255).type(torch.uint8)
            t_h, t_w = int(target.shape[-2]), int(target.shape[-1])
            _, source_lmk, _ = self.function_worker.run_detect_landmark(
                target,
                bbox=np.array([0, 0, t_w, t_h], dtype=np.float32),
                det_kpss=None,
                detect_mode="203",
                score=0.5,
                from_points=False,
                use_mean_eyes=use_mean_eyes,
            )
            if source_lmk is None or (
                hasattr(source_lmk, "__len__") and len(source_lmk) == 0
            ):
                return target.type(torch.float32)

            target_face_512, M_o2c, M_c2o = faceutil.warp_face_by_face_landmark_x(
                target,
                source_lmk,
                dsize=512,
                scale=crop_scale,
                vy_ratio=vy_ratio,
                interpolation=interp_mode,
                padding_mode="zeros",
            )
            target_face_256 = self.t256_face(target_face_512)

            # Source motion + appearance. Appearance (F) takes the 512 crop.
            x_s_info = self.function_worker.perform_recast_motion(target_face_256)
            source_info = self.function_worker.perform_recast_build_source_info(
                x_s_info
            )
            f_s = self.function_worker.perform_recast_extract_appearance(
                target_face_512
            )

            # --- COMPOSE + GENERATE ---
            x_d_i = self.function_worker.perform_recast_compose_driven_keypoints(
                source_info,
                exp_d,
                mode=mode,
                factor=factor,
                region=region,
                eye_driving_weight=eye_weight,
                lip_driving_weight=lip_weight,
                brows_driving_weight=brows_weight,
                cheeks_driving_weight=cheeks_weight,
                jaw_driving_weight=jaw_weight,
                structural_blend=structural_blend,
            )
            out = self.function_worker.perform_recast_warp_decode(
                f_s, source_info["x_s"], x_d_i
            )
            out = torch.squeeze(out)

            # --- DEGENERATE-OUTPUT GUARD ---
            # The PerformRecast generator can emit an all-black (or NaN/Inf)
            # frame when it is fed an out-of-distribution crop (e.g. a too-tight
            # crop scale, or an extreme keypoint pose that overflows the fp16
            # warp). Rather than paste a black face into the result, fall back to
            # the un-recast swap for this frame so the output stays clean.
            if not torch.isfinite(out).all() or ((out < 0.02).float().mean() > 0.9):
                return target.type(torch.float32)
            out = out.clamp_(0, 1)

            # --- PASTE BACK ---
            # Optionally feather the crop border so the recast face blends into
            # the surrounding swap instead of showing a hard 512-crop seam. The
            # border falls back to the underlying swapped-face crop (not black),
            # so this only affects the edge transition, never identity.
            if feather_amount > 0.0:
                fade = max(1, int(round(feather_amount * 256)))

                # 1. Fetch or Create Cached Mask
                if fade not in self._recast_feather_masks:
                    # The SPADE generator output is always strictly 512x512
                    self._recast_feather_masks[fade] = faceutil.create_faded_inner_mask(
                        (512, 512),
                        border_thickness=0,
                        fade_thickness=fade,
                        blur_radius=3,
                        device=out.device,
                    ).unsqueeze(0)

                paste_mask = self._recast_feather_masks[fade]
                tgt01 = (target_face_512.to(out.dtype) / 255.0).clamp_(0, 1)

                # 2. Fused Hardware Lerp (Eliminates intermediate temporary tensors)
                out = torch.lerp(tgt01, out, paste_mask)

            # --- PASTE BACK ---
            dsize = (target.shape[1], target.shape[2])

            # 1. OPTIMIZED: Cache the base 512x512 tracking mask.
            # Avoids running torch.zeros() and slicing every single frame.
            if getattr(self, "_recast_base_warp_mask", None) is None:
                base_mask = torch.zeros(
                    (1, 512, 512), dtype=torch.float32, device=out.device
                )
                base_mask[:, 2:-2, 2:-2] = 1.0
                self._recast_base_warp_mask = base_mask

            warp_mask = self._apply_kornia_warp(
                self._recast_base_warp_mask, M_c2o, dsize
            )

            # 2. Warp the edited face back to the target's coordinate space
            out = self._apply_kornia_warp(out, M_c2o, dsize)
            out = out.mul_(255.0).clamp_(0, 255)

            # 3. OPTIMIZED: Hardware-Fused Lerp Compositing
            # Eliminates 3 full-frame tensor allocations (no `1.0 - mask`, no multiple math tensors).
            # This directly frees up the CUDA memory bus for the worker threads.
            target_float = target.to(dtype=torch.float32, non_blocking=True)
            out = torch.lerp(target_float, out, warp_mask)

        return out

    @torch.no_grad()
    def swap_edit_face_core(
        self,
        img: torch.Tensor,
        swap_restorecalc: torch.Tensor,
        parameters: dict,
        control: dict,
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

            with stream_context, torch.inference_mode():
                init_source_eye_ratio = 0.0
                init_source_lip_ratio = 0.0

                # Detection
                _, lmk_crop, _ = self.function_worker.run_detect_landmark(
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
                x_s_info = self.function_worker.lp_motion_extractor(
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

                f_s_user = self.function_worker.lp_appearance_feature_extractor(
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
                    eyes_delta = self.function_worker.lp_retarget_eye(
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
                    lip_delta = self.function_worker.lp_retarget_lip(
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
                    x_d_new = self.function_worker.lp_stitching(
                        x_s_user, x_d_new, parameters["FaceEditorTypeSelection"]
                    )

                # Generate Image
                out = self.function_worker.lp_warp_decode(
                    f_s_user, x_s_user, x_d_new, parameters["FaceEditorTypeSelection"]
                )
                out = torch.squeeze(out)
                out = out.clamp_(0, 1)

                # --- POST-PROCESSING (Paste Back) ---
                dsize = (img.shape[1], img.shape[2])

                # 1. Create a tracking mask, shaving 2 pixels off the outer edge to
                # prevent Kornia's bilinear boundary interpolation from creating a seam.
                warp_mask = torch.zeros(
                    (1, out.shape[1], out.shape[2]), dtype=out.dtype, device=out.device
                )
                warp_mask[:, 2:-2, 2:-2] = 1.0
                warp_mask = self._apply_kornia_warp(warp_mask, M_c2o, dsize)

                # 2. Warp the manipulated face back to the original frame dimensions
                out = self._apply_kornia_warp(out, M_c2o, dsize)
                out = out.mul_(255.0).clamp_(0, 255)

                # 3. Composite over the original img to preserve the background
                img_float = img.type(torch.float32)
                img = out * warp_mask + img_float * (1.0 - warp_mask)

        return img

    @torch.no_grad()
    def swap_edit_face_core_makeup(
        self,
        img: torch.Tensor,
        kps: np.ndarray | None,
        parameters: dict,
        control: dict,
        **kwargs: Any,
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
        if (
            parameters["FaceMakeupEnableToggle"]
            or parameters["HairMakeupEnableToggle"]
            or parameters["EyeBrowsMakeupEnableToggle"]
            or parameters["LipsMakeupEnableToggle"]
        ):
            if kps is not None and len(kps) >= 5:
                lmk_crop = kps
            else:
                return img

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

            # 1. The call generates both the makeup image AND the exact mask from FaceParser
            out, mask_out = self.function_worker.apply_face_makeup(
                original_face_512, parameters
            )

            # 2. Failsafe: Ensure the output tensor remains exactly 512x512
            if out.shape[-1] != 512:
                out = v2.Resize((512, 512), antialias=True)(out)

            # 3. Quality & Safety Fix: We use the TRUE makeup mask (mask_out)
            # instead of the generic LivePortrait square mask to preserve original skin quality.
            if mask_out is None:
                mask_out = torch.ones(
                    (1, 512, 512), dtype=torch.float32, device=out.device
                )
            elif mask_out.shape[-1] != 512:
                mask_out = v2.Resize((512, 512), antialias=True)(mask_out)

            # 4. Soften the image and blur the mask edges for smooth blending
            out = torch.clamp(torch.div(out, 255.0), 0, 1).type(torch.float32)
            gauss = v2.GaussianBlur(kernel_size=5 * 2 + 1, sigma=(5 + 1) * 0.2)
            mask_crop = gauss(mask_out)

            # 5. Final pasting: Only the makeup pixels are pasted, keeping the original skin intact
            img = faceutil.paste_back_adv(out, M_c2o, img, mask_crop)

        return img

    @torch.no_grad()
    def apply_face_shaping_gpu(
        self, img: torch.Tensor, kps: np.ndarray, parameters: dict
    ) -> torch.Tensor:
        """
        Pure PyTorch implementation of 2D Mesh Deformation for Face Shaping.
        Uses F.grid_sample to warp the tensor based on morphological coordinate offsets.
        Runs entirely on the GPU to prevent host-to-device bottlenecks.

        Args:
            img: 512x512 aligned swap tensor [C, H, W]
            kps: 5-point landmarks in the 512x512 space
            parameters: Dictionary containing the slider values

        Returns:
            torch.Tensor: Warped face tensor.
        """
        if not parameters.get("FaceShapingEnableToggle", False):
            return img

        device = img.device
        C, H, W = img.shape

        # 1. Thread-safe Coordinate Grid Caching (Massive FPS Optimization)
        if self._shaping_cache is None or self._shaping_cache["shape"] != (H, W):
            y, x = torch.meshgrid(
                torch.arange(H, device=device, dtype=torch.float32),
                torch.arange(W, device=device, dtype=torch.float32),
                indexing="ij",
            )

            face_center_x = float(W / 2.0)
            face_center_y = float(H / 2.0)

            # Static distances from center
            x_diff = x - face_center_x
            y_diff = y - face_center_y

            # Edge Falloff Mask: Pre-computed 10% bounding box margin.
            # Protects the 512x512 borders for a seamless paste-back while freeing the forehead/chin.
            margin_x = float(W * 0.10)
            margin_y = float(H * 0.10)
            x_dist = torch.min(x, W - 1.0 - x) / margin_x
            y_dist = torch.min(y, H - 1.0 - y) / margin_y
            edge_mask = torch.clamp(x_dist, 0.0, 1.0) * torch.clamp(y_dist, 0.0, 1.0)
            edge_mask = edge_mask * edge_mask * (3.0 - 2.0 * edge_mask)  # Smoothstep

            self._shaping_cache = {
                "shape": (H, W),
                "x": x,
                "y": y,
                "x_diff": x_diff,
                "y_diff": y_diff,
                "edge_mask": edge_mask,
            }

        # Initialize mutable maps and static references
        static_x = self._shaping_cache["x"]
        static_y = self._shaping_cache["y"]
        x_diff = self._shaping_cache["x_diff"]
        y_diff = self._shaping_cache["y_diff"]
        edge_mask = self._shaping_cache["edge_mask"]

        map_x = static_x.clone()
        map_y = static_y.clone()

        # 2. Dynamic Feature Coordinates (Derived from 5-point ArcFace Canonical KPS)
        eyes_y = float((kps[0, 1] + kps[1, 1]) / 2.0)
        nose_y = float(kps[2, 1])
        mouth_y = float((kps[3, 1] + kps[4, 1]) / 2.0)

        eye_dist = float(np.linalg.norm(kps[0] - kps[1]))
        face_width = eye_dist * 2.5
        face_height = face_width * 1.3

        # 3. Feature Protection Masks (Uses static_y to prevent masking interference)
        eyes_prot_range = face_height * 0.12
        nose_prot_range = face_height * 0.15
        mouth_prot_range = face_height * 0.12

        eyes_mask = torch.exp(-((static_y - eyes_y) ** 2) / (2 * eyes_prot_range**2))
        nose_mask = torch.exp(-((static_y - nose_y) ** 2) / (2 * nose_prot_range**2))
        mouth_mask = torch.exp(-((static_y - mouth_y) ** 2) / (2 * mouth_prot_range**2))

        features_x_range = face_width * 0.25
        features_x_mask = torch.exp(-(x_diff**2) / (2 * features_x_range**2))
        features_protection = (eyes_mask + nose_mask + mouth_mask) * features_x_mask

        deformation_mask = 1.0 - torch.clamp(features_protection, 0.0, 1.0)
        final_mask = deformation_mask * edge_mask

        # 4. Deformation Morphologies
        # Note: Slider < 50 (Shrink) yields negative power. Slider > 50 (Enlarge) yields positive power.
        # map_coord -= diff * power correctly maps the F.grid_sample destination space for both.

        # Face Slimming (Targets upper cheeks)
        val_face_slim = float(parameters.get("FaceShapingFaceSlimSlider", 50))
        if abs(val_face_slim - 50.0) > 1:
            slim_power = ((val_face_slim - 50.0) / 50.0) * 0.4
            vertical_range = face_height * 0.35
            y_weight = torch.exp(-((static_y - nose_y) ** 2) / (2 * vertical_range**2))
            optimal_dist = face_width * 0.35
            x_weight = torch.exp(
                -((torch.abs(x_diff) - optimal_dist) ** 2)
                / (2 * (face_width * 0.15) ** 2)
            )
            map_x -= x_diff * slim_power * (y_weight * x_weight * final_mask)

        # Chin Slimming (Targets lower jaw contour)
        val_chin_slim = float(parameters.get("FaceShapingChinSlimSlider", 50))
        if abs(val_chin_slim - 50.0) > 1:
            chin_power = ((val_chin_slim - 50.0) / 50.0) * 0.5
            cheek_center_y = mouth_y + mouth_prot_range * 0.5
            vertical_range = face_height * 0.25
            optimal_dist = face_width * 0.25
            y_weight = torch.exp(
                -((static_y - cheek_center_y) ** 2) / (2 * vertical_range**2)
            )
            x_weight = torch.exp(
                -((torch.abs(x_diff) - optimal_dist) ** 2)
                / (2 * (face_width * 0.12) ** 2)
            )
            map_x -= x_diff * chin_power * (y_weight * x_weight * final_mask)

        # Forehead Height
        val_forehead = float(parameters.get("FaceShapingForeheadSlider", 50))
        if abs(val_forehead - 50.0) > 1:
            forehead_power = ((val_forehead - 50.0) / 50.0) * 0.4
            forehead_end = eyes_y - eyes_prot_range
            y_ratio = (forehead_end - static_y) / max(forehead_end, 1.0)
            y_gradient = torch.clamp(y_ratio, 0.0, 1.0)
            y_gradient = y_gradient * y_gradient * (3.0 - 2.0 * y_gradient)
            x_weight = torch.exp(-(x_diff**2) / (2 * (face_width * 0.45) ** 2))
            forehead_mask = (
                (static_y < forehead_end).float() * y_gradient * x_weight * final_mask
            )
            max_deform = face_height * 0.2
            map_y -= torch.clamp(
                y_diff * forehead_power * forehead_mask, -max_deform, max_deform
            )

        # Chin Length
        val_chin_length = float(parameters.get("FaceShapingChinLengthSlider", 50))
        if abs(val_chin_length - 50.0) > 1:
            length_power = ((val_chin_length - 50.0) / 50.0) * 0.3
            chin_start = mouth_y + mouth_prot_range
            chin_region = (static_y > chin_start).float()
            y_ratio = (static_y - chin_start) / max(H - chin_start, 1.0)
            y_gradient = torch.clamp(y_ratio, 0.0, 1.0)
            y_gradient = y_gradient * y_gradient * (3.0 - 2.0 * y_gradient)
            x_weight = torch.exp(-(x_diff**2) / (2 * (face_width * 0.4) ** 2))
            chin_mask = (
                chin_region * y_gradient * x_weight * final_mask * (1.0 - mouth_mask)
            )
            max_deform = face_height * 0.15
            map_y -= torch.clamp(
                y_diff * length_power * chin_mask, -max_deform, max_deform
            )

        # Face Length
        val_face_length = float(parameters.get("FaceShapingFaceLengthSlider", 50))
        if abs(val_face_length - 50.0) > 1:
            length_power = ((val_face_length - 50.0) / 50.0) * 0.3
            top_area = (static_y < (eyes_y - eyes_prot_range)).float()
            bottom_area = (static_y > (mouth_y + mouth_prot_range)).float()

            top_ratio = (eyes_y - eyes_prot_range - static_y) / max(eyes_y, 1.0)
            top_mask = top_area * torch.clamp(top_ratio, 0.0, 1.0) * final_mask

            bottom_ratio = (static_y - (mouth_y + mouth_prot_range)) / max(
                H - mouth_y, 1.0
            )
            bottom_mask = bottom_area * torch.clamp(bottom_ratio, 0.0, 1.0) * final_mask

            x_weight = torch.exp(-(x_diff**2) / (2 * (face_width * 0.4) ** 2))
            y_offset = y_diff * length_power * x_weight * (top_mask + bottom_mask)
            map_y -= torch.clamp(y_offset, -face_height * 0.15, face_height * 0.15)

        # 5. Normalize Coordinates for F.grid_sample (-1.0 to 1.0)
        grid_x_norm = (map_x / (W - 1)) * 2.0 - 1.0
        grid_y_norm = (map_y / (H - 1)) * 2.0 - 1.0

        # Combine into expected [1, H, W, 2] shape
        grid = torch.stack((grid_x_norm, grid_y_norm), dim=-1).unsqueeze(0)

        # 6. Apply Warp natively on GPU (Reflection padding mirrors cv2.BORDER_REFLECT_101)
        import torch.nn.functional as F

        img_warped = F.grid_sample(
            img.unsqueeze(0),
            grid,
            mode="bilinear",
            padding_mode="reflection",
            align_corners=True,
        ).squeeze(0)

        return img_warped
