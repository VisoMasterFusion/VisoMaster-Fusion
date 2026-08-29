import gc
import math
import threading
import traceback
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
import torchvision
from torchvision.transforms import v2

from app.helpers.miscellaneous import (
    ParametersDict,
    draw_bounding_boxes_on_detected_faces,
    get_grid_for_pasting,
    keypoints_adjustments,
)
from app.helpers.vr_geometry import VRGeometry, rays_to_frame_pixels
from app.helpers.vr_utils import EquirectangularConverter, PerspectiveConverter
from app.processors.utils import platform_support

if TYPE_CHECKING:
    from app.ui.widgets import widget_components

    # Forward reference to the main FrameWorker orchestrator
    from .frame_worker import FrameWorker


class VRProcessor:
    """
    Handles all VR specific operations.
    Delegates back to the main FrameWorker for standard pipeline operations
    (swap_core, recognition, UI state) to guarantee thread safety.

    The mapping from frame pixels to directions on the viewing sphere is described
    by a VRGeometry built from the UI controls, so per-eye coverage (90°–360°) and
    the storage projection (equirectangular or fisheye) are both configurable.
    """

    def __init__(self, worker: "FrameWorker"):
        self.worker = worker

        # VR specific constants
        self.VR_PERSPECTIVE_RENDER_SIZE: int = 512
        self.VR_FOV_SCALE_FACTOR: float = 1.5

        # VR converter caches (VR-08 & Improvement K).  Keyed on the geometry, which
        # carries the frame size as well as the projection and coverage, so a change
        # to any of them rebuilds the converter.
        self._vr_converter: EquirectangularConverter | None = None
        self._vr_geometry: VRGeometry | None = None

        self._vr_p2e_converter: PerspectiveConverter | None = None
        self._vr_p2e_geometry: VRGeometry | None = None

        # VR tracking state
        self.last_detected_faces_vr: list[dict[str, Any]] = []
        self.last_processed_frame_number_vr: int = -1

        # VRAM Management: periodic CUDA allocator flush counter
        self._vr_processed_count: int = 0

        # Dirty-check cache for scaling transforms to prevent redundant recreations
        self._last_vr_scaling_control: dict[str, Any] | None = None

    @staticmethod
    def _rodrigues_np(axis_angle: np.ndarray) -> np.ndarray:
        """Compute a 3×3 rotation matrix from an axis-angle vector (Rodrigues formula).
        Pure numpy — no cv2 dependency needed for the VR bbox projection helper."""
        angle = float(np.linalg.norm(axis_angle))
        if angle < 1e-10:
            return np.eye(3, dtype=np.float64)
        k = axis_angle / angle
        K = np.array(
            [[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]],
            dtype=np.float64,
        )
        return np.eye(3) + math.sin(angle) * K + (1.0 - math.cos(angle)) * (K @ K)

    @staticmethod
    def _project_crop_bbox_to_frame(
        bbox_crop: np.ndarray,
        theta: float,
        phi: float,
        fov: float,
        crop_size: int,
        geometry: VRGeometry,
        is_left_eye: "bool | None",
    ) -> "np.ndarray | None":
        """Project a crop-space bounding box back to frame pixel coordinates.

        Samples 9 points (4 corners + 4 edge midpoints + centre) through the same
        forward rotation used by E2P, then takes the axis-aligned bounding box of
        the projected points.

        Returns (x1, y1, x2, y2) float32 in frame pixel space, or None if the
        resulting box is degenerate (<2 px on either side).
        """
        y_axis = np.array([0.0, 1.0, 0.0], np.float64)
        z_axis = np.array([0.0, 0.0, 1.0], np.float64)
        # Same longitude convention as E2P: frame-level for equirectangular,
        # eye-relative for a fisheye.
        rot_theta = geometry.rotation_theta(theta, is_left_eye)
        R1 = VRProcessor._rodrigues_np(z_axis * math.radians(rot_theta))
        R2 = VRProcessor._rodrigues_np(np.dot(R1, y_axis) * math.radians(-phi))
        R = R2 @ R1

        w_len = math.tan(math.radians(fov / 2.0))

        x1_c, y1_c, x2_c, y2_c = (
            float(bbox_crop[0]),
            float(bbox_crop[1]),
            float(bbox_crop[2]),
            float(bbox_crop[3]),
        )
        mx_c = (x1_c + x2_c) / 2.0
        my_c = (y1_c + y2_c) / 2.0

        sample_pts = [
            (x1_c, y1_c),
            (mx_c, y1_c),
            (x2_c, y1_c),
            (x1_c, my_c),
            (mx_c, my_c),
            (x2_c, my_c),
            (x1_c, y2_c),
            (mx_c, y2_c),
            (x2_c, y2_c),
        ]

        rays: np.ndarray = np.empty((len(sample_pts), 3), dtype=np.float32)
        for i, (u, v) in enumerate(sample_pts):
            px_norm = 2.0 * u / (crop_size - 1) - 1.0
            py_norm = 2.0 * v / (crop_size - 1) - 1.0
            d = np.array([1.0, w_len * px_norm, -w_len * py_norm], np.float64)
            rays[i] = R @ (d / np.linalg.norm(d))

        # Reuse the shared projection so the tiled detector's bboxes land exactly
        # where the crop was actually sampled from.
        x_px_t, y_px_t = rays_to_frame_pixels(
            torch.from_numpy(rays), geometry, is_left_eye
        )
        x_pxs = x_px_t.numpy()
        y_pxs = y_px_t.numpy()

        eq_w, eq_h = geometry.frame_width, geometry.frame_height
        x1_eq = float(np.clip(x_pxs.min(), 0.0, eq_w - 1))
        y1_eq = float(np.clip(y_pxs.min(), 0.0, eq_h - 1))
        x2_eq = float(np.clip(x_pxs.max(), 0.0, eq_w - 1))
        y2_eq = float(np.clip(y_pxs.max(), 0.0, eq_h - 1))

        if x2_eq - x1_eq < 2.0 or y2_eq - y1_eq < 2.0:
            return None
        return np.array([x1_eq, y1_eq, x2_eq, y2_eq], dtype=np.float32)

    def _detect_faces_vr_tiled(
        self,
        equirect_converter: EquirectangularConverter,
        control: dict,
        geometry: VRGeometry,
        crop_size: int,
        stop_event: threading.Event | None = None,
    ) -> list:
        """Detect faces in a grid of undistorted perspective crops.

        Catches faces that are invisible to the frame-domain detector because of
        projection distortion (high elevation, near an eye's edge, or very large
        near-camera faces).

        Returns a list of float32 numpy arrays, each (4,) or (5,) in frame pixel
        coordinates.
        """
        tile_fov = 90.0
        # The grid is derived from the geometry so it sweeps exactly the sphere the
        # frame actually covers, rather than a hardcoded full 360°×180°.
        tile_grid = geometry.tile_grid(tile_fov_deg=tile_fov)
        det_score = control["DetectorScoreSlider"] / 100.0
        found: list = []

        for theta, phi in tile_grid:
            if stop_event is not None and stop_event.is_set():
                break
            is_left_eye = geometry.eye_of_theta(theta)
            tile_crop = equirect_converter.get_perspective_crop(
                tile_fov, theta, phi, crop_size, crop_size, is_left_eye=is_left_eye
            )
            if tile_crop is None or tile_crop.numel() == 0:
                continue

            bboxes_tile, _, _ = self.worker.function_worker.run_detect(
                tile_crop,
                control["DetectorModelSelection"],
                max_num=control["MaxFacesToDetectSlider"],
                score=det_score,
                use_landmark_detection=False,
                landmark_detect_mode=control["LandmarkDetectModelSelection"],
                landmark_score=control["LandmarkDetectScoreSlider"] / 100.0,
                from_points=False,
                rotation_angles=[0],
                use_mean_eyes=False,
                previous_detections=None,
            )

            if not isinstance(bboxes_tile, np.ndarray):
                bboxes_tile = np.array(bboxes_tile)
            if bboxes_tile.ndim == 1 and bboxes_tile.shape[0] in (4, 5):
                bboxes_tile = bboxes_tile.reshape(1, -1)
            if bboxes_tile.ndim != 2 or bboxes_tile.shape[0] == 0:
                continue

            for bbox_tile in bboxes_tile:
                eq_bbox = self._project_crop_bbox_to_frame(
                    bbox_tile[:4],
                    theta,
                    phi,
                    tile_fov,
                    crop_size,
                    geometry,
                    is_left_eye,
                )
                if eq_bbox is None:
                    continue
                if bboxes_tile.shape[1] >= 5:
                    found.append(
                        np.append(eq_bbox, float(bbox_tile[4])).astype(np.float32)
                    )
                else:
                    found.append(eq_bbox)

        return found

    def _process_single_vr_perspective_crop_multi(
        self,
        perspective_crop_torch_rgb_uint8: torch.Tensor,
        target_face_button: "widget_components.TargetFaceCardButton | None",
        parameters_for_face: ParametersDict,
        control_global: dict[str, Any],
        kps_5_on_crop_param: np.ndarray,
        kps_all_on_crop_param: np.ndarray | None,
        swap_button_is_checked_global: bool,
        edit_button_is_checked_global: bool,
        eye_side_for_debug: str = "",
        kv_map_for_swap: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Processes a single perspective crop extracted from a VR frame.

        Returns:
            (processed_crop, swap_mask_1x512x512_float_or_None)
            The mask is only populated when self.is_view_face_mask is True.
        """
        # VR-12: assert crop is square with the configured render size so downstream
        # swap_core assumptions hold.  Improvement F allows 512/768/1024 via UI.
        _expected_crop_size = self.VR_PERSPECTIVE_RENDER_SIZE
        assert perspective_crop_torch_rgb_uint8.shape[-2:] == (
            _expected_crop_size,
            _expected_crop_size,
        ), (
            f"VR perspective crop must be {_expected_crop_size}×{_expected_crop_size}, "
            f"got {perspective_crop_torch_rgb_uint8.shape}"
        )
        processed_crop_torch_rgb_uint8 = perspective_crop_torch_rgb_uint8.clone()
        if kps_5_on_crop_param is None or kps_5_on_crop_param.size == 0:
            return processed_crop_torch_rgb_uint8, None

        if not (swap_button_is_checked_global or edit_button_is_checked_global):
            return processed_crop_torch_rgb_uint8, None

        arcface_model_for_swap = self.worker.function_worker.get_arcface_model(
            parameters_for_face["SwapModelSelection"]
        )
        s_e_for_swap_np = None
        if swap_button_is_checked_global and target_face_button is not None:
            _vr_reaging_on = parameters_for_face.get("FaceReagingEnableToggle", False)
            _vr_aged_emb = getattr(target_face_button, "aged_input_embedding", {})
            if _vr_reaging_on and _vr_aged_emb:
                s_e_for_swap_np = _vr_aged_emb.get(arcface_model_for_swap)
            else:
                s_e_for_swap_np = target_face_button.assigned_input_embedding.get(
                    arcface_model_for_swap
                )
            if (
                s_e_for_swap_np is None
                or not isinstance(s_e_for_swap_np, np.ndarray)
                or s_e_for_swap_np.size == 0
                or np.isnan(s_e_for_swap_np).any()
                or np.isinf(s_e_for_swap_np).any()
            ):
                s_e_for_swap_np = None

        t_e_for_swap_np = (
            target_face_button.get_embedding(arcface_model_for_swap)
            if target_face_button
            else None
        )
        dfm_model_instance_local = None
        if parameters_for_face["SwapModelSelection"] == "DeepFaceLive (DFM)":
            dfm_model_name = parameters_for_face["DFMModelSelection"]
            if dfm_model_name:
                dfm_model_instance_local = self.worker.models_processor.load_dfm_model(
                    dfm_model_name
                )

        s_e_for_swap_core = s_e_for_swap_np if swap_button_is_checked_global else None
        vr_swap_mask_for_compare: torch.Tensor | None = None

        # --- Early Exit for No-Op Swaps ---
        _is_dfm = parameters_for_face["SwapModelSelection"] == "DeepFaceLive (DFM)"
        _has_input = (s_e_for_swap_core is not None) or (
            _is_dfm and dfm_model_instance_local is not None
        )
        _force_swap = control_global.get("ForceSwapToggle", True)

        _should_run_pipeline = (
            swap_button_is_checked_global and (_has_input or _force_swap)
        ) or edit_button_is_checked_global

        if _should_run_pipeline:
            source_kps = None
            if target_face_button:
                # 1. Prioritize assigned Input Faces
                if target_face_button.assigned_input_faces:
                    first_input_id = list(
                        target_face_button.assigned_input_faces.keys()
                    )[0]
                    store = target_face_button.assigned_input_faces[first_input_id]
                    source_kps = store.get("kps_5")
                # 2. Fallback to assigned Merged Embeddings
                elif (
                    hasattr(target_face_button, "assigned_merged_embeddings")
                    and target_face_button.assigned_merged_embeddings
                ):
                    first_embed_id = list(
                        target_face_button.assigned_merged_embeddings.keys()
                    )[0]
                    store = target_face_button.assigned_merged_embeddings[
                        first_embed_id
                    ]
                    source_kps = store.get("kps_5")

            kps_5_on_crop_param = keypoints_adjustments(
                kps_5_on_crop_param,
                cast(dict, parameters_for_face),
                source_kps=source_kps,  # type: ignore[arg-type]
            )
            _params_for_swap_vr = self.worker._apply_auto_mouth(
                parameters_for_face.data,  # plain dict — VR convention
                target_face_button,
                vr_crop_chw=perspective_crop_torch_rgb_uint8,  # already a focused face region
            )
            try:
                (
                    swapped_face_512_torch_rgb_uint8,
                    comprehensive_mask_1x512x512_from_swap_core,
                    _,
                ) = self.worker.swap_core(
                    perspective_crop_torch_rgb_uint8,
                    kps_5_on_crop_param,
                    kps=kps_all_on_crop_param,
                    s_e=s_e_for_swap_core,
                    t_e=t_e_for_swap_np,
                    parameters=_params_for_swap_vr,
                    control=control_global,
                    dfm_model_name=parameters_for_face["DFMModelSelection"],
                    is_perspective_crop=True,
                    kv_map=kv_map_for_swap,
                )
            except Exception as e_swap_core:
                print(
                    f"[ERROR] Error in swap_core for VR crop {eye_side_for_debug}: {e_swap_core}"
                )
                traceback.print_exc()
                swapped_face_512_torch_rgb_uint8 = cast(v2.Resize, self.worker.t512)(
                    perspective_crop_torch_rgb_uint8
                )
                comprehensive_mask_1x512x512_from_swap_core = torch.zeros(
                    (1, 512, 512),
                    dtype=torch.float32,
                    device=perspective_crop_torch_rgb_uint8.device,
                )

            tform_persp_to_512template = self.worker.get_face_similarity_tform(
                parameters_for_face["SwapModelSelection"], kps_5_on_crop_param
            )

            # FW-PERF-5: use promoted instance-attribute transform for masks
            t512_mask = self.worker.t512_mask
            assert t512_mask is not None, (
                "t512_mask must be initialized via set_scaling_transforms"
            )

            if (
                comprehensive_mask_1x512x512_from_swap_core is None
                or comprehensive_mask_1x512x512_from_swap_core.numel() == 0
            ):
                # Fallback mask logic
                persp_final_combined_mask_1x512x512_float_for_paste = (
                    t512_mask(
                        self.worker.get_border_mask(parameters_for_face.data)[0]
                    ).float()
                    if swap_button_is_checked_global
                    else torch.zeros(
                        (1, 512, 512),
                        dtype=torch.float32,
                        device=perspective_crop_torch_rgb_uint8.device,
                    )
                )
                vr_swap_mask_for_compare = None
            else:
                # Primary path
                persp_final_combined_mask_1x512x512_float_for_paste = (
                    comprehensive_mask_1x512x512_from_swap_core.float()
                )
                vr_swap_mask_for_compare = (
                    comprehensive_mask_1x512x512_from_swap_core
                    if self.worker.is_view_face_mask
                    else None
                )

                if parameters_for_face.get("BordermaskEnableToggle", False):
                    border_mask_128, _ = self.worker.get_border_mask(
                        parameters_for_face.data
                    )
                    border_mask_512 = t512_mask(border_mask_128)
                    persp_final_combined_mask_1x512x512_float_for_paste *= (
                        border_mask_512
                    )

            persp_final_combined_mask_3x512x512_float_for_paste = (
                persp_final_combined_mask_1x512x512_float_for_paste.repeat(3, 1, 1)
            )
            masked_swapped_face_to_paste_float = (
                swapped_face_512_torch_rgb_uint8.float()
                * persp_final_combined_mask_3x512x512_float_for_paste
            )

            crop_h, crop_w = (
                perspective_crop_torch_rgb_uint8.shape[1],
                perspective_crop_torch_rgb_uint8.shape[2],
            )
            _, source_grid_normalized_xy_persp = get_grid_for_pasting(
                tform_persp_to_512template,
                crop_h,
                crop_w,
                512,
                512,
                perspective_crop_torch_rgb_uint8.device,
            )

            # Bug 5 fix: use align_corners=True to match the E2P/P2E projection convention.
            # Previously align_corners=False introduced a sub-pixel shift (~0.2% for 512px)
            # at the face-paste step, causing slight misalignment at crop boundaries.
            pasted_face_on_persp_float = torch.nn.functional.grid_sample(
                masked_swapped_face_to_paste_float.unsqueeze(0),
                source_grid_normalized_xy_persp,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            ).squeeze(0)
            transformed_mask_on_persp_float = torch.nn.functional.grid_sample(
                persp_final_combined_mask_3x512x512_float_for_paste.unsqueeze(0),
                source_grid_normalized_xy_persp,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            ).squeeze(0)
            blended_persp_crop_float = (
                pasted_face_on_persp_float
                + perspective_crop_torch_rgb_uint8.float()
                * (1.0 - transformed_mask_on_persp_float)
            )
            processed_crop_torch_rgb_uint8 = torch.clamp(
                blended_persp_crop_float, 0, 255
            ).byte()

            if edit_button_is_checked_global:
                _, _, kps_all_for_editor_list = self.worker.function_worker.run_detect(
                    processed_crop_torch_rgb_uint8,
                    control_global["DetectorModelSelection"],
                    max_num=1,
                    score=control_global["DetectorScoreSlider"] / 100.0,
                    input_size=(
                        processed_crop_torch_rgb_uint8.shape[1],
                        processed_crop_torch_rgb_uint8.shape[2],
                    ),
                    use_landmark_detection=True,
                    landmark_detect_mode="203",
                    landmark_score=control_global["LandmarkDetectScoreSlider"] / 100.0,
                    from_points=True,
                    rotation_angles=[0],
                    use_mean_eyes=control_global.get("LandmarkMeanEyesToggle", False),
                )
                kps_all_for_editor_on_crop = (
                    kps_all_for_editor_list[0]
                    if len(kps_all_for_editor_list) > 0
                    else None
                )
                if (
                    kps_all_for_editor_on_crop is not None
                    and kps_all_for_editor_on_crop.size > 0
                ):
                    processed_crop_torch_rgb_uint8 = (
                        self.worker.function_worker.swap_edit_face_core(
                            processed_crop_torch_rgb_uint8,
                            processed_crop_torch_rgb_uint8,
                            parameters_for_face.data,
                            control_global,
                        )
                    )
                    if any(
                        parameters_for_face.get(f, False)
                        for f in (
                            "FaceMakeupEnableToggle",
                            "HairMakeupEnableToggle",
                            "EyeBrowsMakeupEnableToggle",
                            "LipsMakeupEnableToggle",
                        )
                    ):
                        assert (
                            kps_all_on_crop_param is not None
                        )  # guarded by outer landmark check
                        processed_crop_torch_rgb_uint8 = (
                            self.worker.function_worker.swap_edit_face_core_makeup(
                                processed_crop_torch_rgb_uint8,
                                kps_all_on_crop_param,
                                parameters_for_face.data,
                                control_global,
                            )
                        )

        return processed_crop_torch_rgb_uint8, vr_swap_mask_for_compare

    @torch.no_grad()
    def process_frame_vr180(
        self,
        original_equirect_tensor_for_vr: torch.Tensor,
        img_numpy_rgb_uint8: np.ndarray,
        control: dict,
        stop_event: threading.Event,
    ) -> torch.Tensor:
        """
        Handles the specific logic for VR frames:
        - Whole-frame detection
        - Perspective cropping
        - Processing per crop
        - Stitching back
        """
        # FW-RACE-02: read from snapshotted feeder state instead of live Qt button
        swap_button_is_checked_global = (
            self.worker.main_window.swapfacesButton.isChecked()
        )
        edit_button_is_checked_global = (
            self.worker.main_window.editFacesButton.isChecked()
        )

        # How this frame's pixels map to directions: eye layout, storage projection
        # and per-eye coverage.  Everything downstream asks the geometry rather than
        # assuming the VR180 360°×180° special case.
        geometry = VRGeometry.from_control(
            control,
            frame_height=img_numpy_rgb_uint8.shape[0],
            frame_width=img_numpy_rgb_uint8.shape[1],
        )

        # VR-08: cache the EquirectangularConverter instance at worker level.
        # When the geometry is unchanged (common for video), reuse the cached
        # instance and update its image data in-place to avoid redundant Python-side
        # object allocation.
        if self._vr_converter is None or self._vr_geometry != geometry:
            self._vr_converter = EquirectangularConverter(
                img_numpy_rgb_uint8,
                device=self.worker.models_processor.device,
                geometry=geometry,
            )
            self._vr_geometry = geometry
        else:
            # VR-MEM-01: update in-place to avoid new GPU allocation per frame.
            self._vr_converter.equirect_tensor_cxhxw_rgb_uint8.copy_(
                torch.from_numpy(img_numpy_rgb_uint8).permute(2, 0, 1)
            )
            # VR-PERF-11: invalidate the per-frame float cache so GetPerspective
            # recomputes it once from the fresh uint8 data (instead of 24+ times).
            self._vr_converter.e2p_instance._img_float = None
        equirect_converter = self._vr_converter
        assert equirect_converter is not None, (
            "VR converter must be initialized before use"
        )

        # VR-11: always use rotation_angles=[0] in VR mode — non-zero angles produce
        # geometrically invalid spherical coordinates.
        vr_rotation_angles = [0]

        # Improvement F: read perspective crop resolution from UI (512/768/1024).
        self.VR_PERSPECTIVE_RENDER_SIZE = int(
            control.get("VR180CropResolutionSelection", "512")
        )
        # VR-15: ensure scaling transforms are set for this resolution.
        # FW-PERF-07b: mirror the standard-path dirty-check so set_scaling_transforms
        # is not called on every VR frame when settings have not changed.
        _vr_scaling_keys = {
            k: control.get(k)
            for k in (
                "get_cropped_face_kpsTypeSelection",
                "original_face_128_384TypeSelection",
                "original_face_512TypeSelection",
                "UntransformTypeSelection",
                "ScalebackFrameTypeSelection",
                "expression_faceeditor_t256TypeSelection",
                "expression_faceeditor_backTypeSelection",
                "block_shiftTypeSelection",
                "AntialiasTypeSelection",
            )
        }
        _vr_scaling_keys["VR180CropResolutionSelection"] = (
            self.VR_PERSPECTIVE_RENDER_SIZE
        )
        if self._last_vr_scaling_control != _vr_scaling_keys:
            self.worker.set_scaling_transforms(control)
            self._last_vr_scaling_control = _vr_scaling_keys

        # Improvement D: in Both-Eyes mode, detect on each eye-half separately.
        # A standard VR180 SBS equirect is 2:1 (W=2H), so each half is square (H×H).
        # Detecting on each 1:1 half gives the detector 4× more pixels per face vs
        # letterboxing the full 2:1 equirect to 512×256.
        _eq_h = geometry.frame_height
        _eq_w = geometry.frame_width
        _is_both_eyes = geometry.both_eyes
        _aspect = _eq_w / max(_eq_h, 1)
        _do_per_eye_detect = _is_both_eyes and _aspect >= 1.8

        _det_score = control["DetectorScoreSlider"] / 100.0
        _det_mode = control["DetectorModelSelection"]
        _det_max = control["MaxFacesToDetectSlider"]
        _lm_mode = control["LandmarkDetectModelSelection"]
        _lm_score = control["LandmarkDetectScoreSlider"] / 100.0
        _use_mean_eyes = control.get("LandmarkMeanEyesToggle", False)

        def _run_det(tensor, bypass_bytetrack=False):
            return self.worker.function_worker.run_detect(
                tensor,
                _det_mode,
                max_num=_det_max,
                score=_det_score,
                use_landmark_detection=False,
                landmark_detect_mode=_lm_mode,
                landmark_score=_lm_score,
                from_points=False,
                rotation_angles=vr_rotation_angles,
                use_mean_eyes=_use_mean_eyes,
                bypass_bytetrack=bypass_bytetrack,
            )

        def _norm_bboxes(arr):
            if not isinstance(arr, np.ndarray):
                arr = np.array(arr)
            if arr.ndim == 1 and arr.shape[0] in (4, 5):
                arr = arr.reshape(1, -1)
            return arr

        # --- Strict Early Exit (No Targets in VR) ---
        with self.worker.lock:
            target_faces_snapshot = dict(self.worker.main_window.target_faces)

        show_bboxes = control.get(
            "ShowAllDetectedFacesBBoxToggle", False
        ) or control.get("ShowByteTrackBBoxToggle", False)

        # The entire detection phase (including tiles) must be sealed inside the 'else' branch.
        if len(target_faces_snapshot) == 0 and not show_bboxes:
            bboxes_eq_np = np.array([])
        else:
            if _do_per_eye_detect:
                _half_w = original_equirect_tensor_for_vr.shape[2] // 2
                _left_tensor = original_equirect_tensor_for_vr[:, :, :_half_w]
                _right_tensor = original_equirect_tensor_for_vr[:, :, _half_w:]

                # bypass_bytetrack=True: per-eye detection uses half-width coordinate spaces,
                # which would corrupt the tracker's Kalman state if ByteTrack ran on both halves.
                _bboxes_left = _norm_bboxes(
                    _run_det(_left_tensor, bypass_bytetrack=True)[0]
                )
                _bboxes_right = _norm_bboxes(
                    _run_det(_right_tensor, bypass_bytetrack=True)[0]
                )

                # Offset right-half x-coordinates into full-equirect space
                if _bboxes_right.ndim == 2 and _bboxes_right.shape[0] > 0:
                    _bboxes_right = _bboxes_right.copy()
                    _bboxes_right[:, 0] += _half_w
                    _bboxes_right[:, 2] += _half_w

                _pieces = []
                if _bboxes_left.ndim == 2 and _bboxes_left.shape[0] > 0:
                    _pieces.append(_bboxes_left)
                if _bboxes_right.ndim == 2 and _bboxes_right.shape[0] > 0:
                    _pieces.append(_bboxes_right)
                bboxes_eq_np = np.vstack(_pieces) if _pieces else np.array([])
            else:
                # Single Eye or non-2:1 equirect — detect on the full frame as before.
                # Use the standard (512, 512) input size — TRT engines are compiled for this shape.
                bboxes_eq_np = _norm_bboxes(
                    _run_det(original_equirect_tensor_for_vr)[0]
                )

            if not isinstance(bboxes_eq_np, np.ndarray):
                bboxes_eq_np = np.array(bboxes_eq_np)

            # FW-ROBUST-07: reshape 1-D bbox (e.g. single face returned as shape (4,) or (5,))
            if bboxes_eq_np.ndim == 1 and bboxes_eq_np.shape[0] in (4, 5):
                bboxes_eq_np = bboxes_eq_np.reshape(1, -1)

            # Improvement A: tiled perspective detection — catch faces missed by equirect
            if control.get("VR180TileDetectionToggle", False):
                _tile_bboxes = self._detect_faces_vr_tiled(
                    equirect_converter,
                    control,
                    geometry,
                    self.VR_PERSPECTIVE_RENDER_SIZE,
                    stop_event=stop_event,
                )
                if _tile_bboxes:
                    _tile_arr = np.array(_tile_bboxes, dtype=np.float32)
                    if _tile_arr.ndim == 1 and _tile_arr.shape[0] in (4, 5):
                        _tile_arr = _tile_arr.reshape(1, -1)
                    if _tile_arr.ndim == 2 and _tile_arr.shape[0] > 0:
                        _dev = self.worker.models_processor.device
                        _tile_boxes_t = torch.from_numpy(
                            _tile_arr[:, :4].astype(np.float32)
                        ).to(_dev)

                        # Step 1 — intra-tile NMS: the same face is often detected by
                        # multiple overlapping tiles (90° FOV with 60° spacing gives 30°
                        # of overlap).  Run a tight NMS to merge these near-duplicates
                        # before comparing against the equirect detections.
                        if _tile_arr.shape[0] > 1:
                            _t_scores = (
                                torch.from_numpy(_tile_arr[:, 4].astype(np.float32))
                                .to(_dev)
                                .clamp(min=1e-6)
                                if _tile_arr.shape[1] >= 5
                                else (
                                    (_tile_boxes_t[:, 2] - _tile_boxes_t[:, 0])
                                    * (_tile_boxes_t[:, 3] - _tile_boxes_t[:, 1])
                                ).clamp(min=0.0)
                            )
                            _t_keep = torchvision.ops.nms(
                                _tile_boxes_t, _t_scores, iou_threshold=0.3
                            )
                            _tile_arr = _tile_arr[_t_keep.cpu().numpy()]
                            _tile_boxes_t = torch.from_numpy(
                                _tile_arr[:, :4].astype(np.float32)
                            ).to(_dev)

                        # Step 2 — suppress tile detections that already have a
                        # corresponding equirect detection.  If a tile bbox overlaps any
                        # equirect bbox by IoU ≥ 0.3, it represents the same face and
                        # adding it would cause double-processing (two swaps stitched on
                        # top of each other → artifacts, wrong size, colour shift).
                        # Only genuinely NEW detections (IoU < 0.3 with all equirect
                        # bboxes) are merged into the candidate list.
                        if bboxes_eq_np.ndim == 2 and bboxes_eq_np.shape[0] > 0:
                            _eq_boxes_t = torch.from_numpy(
                                bboxes_eq_np[:, :4].astype(np.float32)
                            ).to(_dev)
                            # pairwise IoU matrix: shape (N_tile, N_equirect)
                            _iou_tile_vs_eq = torchvision.ops.box_iou(
                                _tile_boxes_t, _eq_boxes_t
                            )
                            _novel_mask = (
                                (_iou_tile_vs_eq.max(dim=1).values < 0.3).cpu().numpy()
                            )
                            _novel_tile = _tile_arr[_novel_mask]
                            if _novel_tile.shape[0] > 0:
                                # Pad score column with 0.9 default where missing
                                _n_cols = max(
                                    bboxes_eq_np.shape[1], _novel_tile.shape[1]
                                )
                                if bboxes_eq_np.shape[1] < _n_cols:
                                    bboxes_eq_np = np.hstack(
                                        [
                                            bboxes_eq_np,
                                            np.full(
                                                (
                                                    bboxes_eq_np.shape[0],
                                                    _n_cols - bboxes_eq_np.shape[1],
                                                ),
                                                0.9,
                                                dtype=np.float32,
                                            ),
                                        ]
                                    )
                                if _novel_tile.shape[1] < _n_cols:
                                    _novel_tile = np.hstack(
                                        [
                                            _novel_tile,
                                            np.full(
                                                (
                                                    _novel_tile.shape[0],
                                                    _n_cols - _novel_tile.shape[1],
                                                ),
                                                0.9,
                                                dtype=np.float32,
                                            ),
                                        ]
                                    )
                                bboxes_eq_np = np.vstack([bboxes_eq_np, _novel_tile])
                        else:
                            # No equirect detections at all — use all (intra-NMS'd) tile bboxes
                            bboxes_eq_np = _tile_arr

        # VR-03 / Bug 2 fix: IoU-NMS using detector confidence score when available,
        # falling back to bbox area only when scores are not present.
        if bboxes_eq_np.ndim == 2 and bboxes_eq_np.shape[0] > 1:
            _boxes_t = torch.from_numpy(bboxes_eq_np[:, :4].astype(np.float32)).to(
                self.worker.models_processor.device
            )
            if bboxes_eq_np.shape[1] >= 5:
                # Use detector confidence score — keeps the highest-confidence detection
                _scores_t = (
                    torch.from_numpy(bboxes_eq_np[:, 4].astype(np.float32))
                    .to(self.worker.models_processor.device)
                    .clamp(min=1e-6)
                )
            else:
                # Fall back to area when no confidence score column is present
                _scores_t = (
                    (_boxes_t[:, 2] - _boxes_t[:, 0])
                    * (_boxes_t[:, 3] - _boxes_t[:, 1])
                ).clamp(min=0.0)
            _keep = torchvision.ops.nms(_boxes_t, _scores_t, iou_threshold=0.5)
            bboxes_eq_np = bboxes_eq_np[_keep.cpu().numpy()]

        # Update VR tracking state for the next frame. Done here — post-NMS so the
        # stored bboxes are clean, pre-VR-MIRROR so synthetic bboxes are not tracked.
        _vr_state_for_next: list = (
            [
                {"bbox": row[:4], "score": float(row[4]) if len(row) >= 5 else 1.0}
                for row in bboxes_eq_np
            ]
            if bboxes_eq_np.ndim == 2 and bboxes_eq_np.shape[0] > 0
            else []
        )
        with self.worker.lock:
            self.last_detected_faces_vr = _vr_state_for_next
            self.last_processed_frame_number_vr = self.worker.frame_number

        if len(bboxes_eq_np) == 0:
            return original_equirect_tensor_for_vr

        # VR-MIRROR: in Both-Eyes mode, if a face is detected on one eye side but not
        # the other, synthesize a mirrored bbox on the missing side using the same
        # relative position and size. This prevents one-eye-only swaps when the detector
        # fires on one half but not the other due to marginal confidence or rendering
        # differences between the two stereo views.
        # Only applies to Both-Eyes mode (not to a single-view frame).
        if geometry.both_eyes and bboxes_eq_np.ndim == 2 and bboxes_eq_np.shape[0] > 0:
            _half_w = float(geometry.eye_width)
            _cx = (bboxes_eq_np[:, 0] + bboxes_eq_np[:, 2]) / 2.0
            _is_left = _cx < _half_w
            _mirrored_bboxes: list[np.ndarray] = []
            for _i in range(len(bboxes_eq_np)):
                _b = bboxes_eq_np[_i]
                _tol = max(_b[2] - _b[0], _b[3] - _b[1]) * 0.5
                if _is_left[_i]:
                    _mx1, _mx2 = _b[0] + _half_w, _b[2] + _half_w
                    _other_mask = ~_is_left
                else:
                    _mx1, _mx2 = _b[0] - _half_w, _b[2] - _half_w
                    _other_mask = _is_left
                _mcx = (_mx1 + _mx2) / 2.0
                _mcy = (_b[1] + _b[3]) / 2.0
                _found = False
                for _j in np.where(_other_mask)[0]:
                    _ocx = (bboxes_eq_np[_j, 0] + bboxes_eq_np[_j, 2]) / 2.0
                    _ocy = (bboxes_eq_np[_j, 1] + bboxes_eq_np[_j, 3]) / 2.0
                    if abs(_ocx - _mcx) < _tol and abs(_ocy - _mcy) < _tol:
                        _found = True
                        break
                if not _found:
                    _mb = _b.copy()
                    _mb[0] = _mx1
                    _mb[2] = _mx2
                    _mirrored_bboxes.append(_mb)
            if _mirrored_bboxes:
                bboxes_eq_np = np.vstack([bboxes_eq_np, np.array(_mirrored_bboxes)])

        # VR-07: use list of namedtuple-style tuples instead of a string-keyed dict
        # to avoid string allocation for each face and simplify downstream iteration
        processed_perspective_crops_details = []  # each entry: (eye_side, tensor, theta, phi, fov)
        analyzed_faces_for_vr = []
        # Crops for detected faces that had no matching target — shown as-is in compare/mask mode
        unmatched_compare_crops: list[torch.Tensor] = []
        compare_mode_vr_early = (
            self.worker.is_view_face_mask or self.worker.is_view_face_compare
        )

        # Phase 1: extract all perspective crops and run landmark detection.
        # Failures are kept (kps_on_crop=None) so Phase 2 can attempt to fill them
        # from the partner eye view before discarding.
        _crop_landmark_results: list[dict] = []
        for bbox_eq_single in bboxes_eq_np:
            if stop_event.is_set():
                break

            theta, phi = equirect_converter.calculate_theta_phi_from_bbox(
                bbox_eq_single
            )
            _is_left_eye = geometry.eye_of_pixel(
                (float(bbox_eq_single[0]) + float(bbox_eq_single[2])) / 2.0
            )
            # None = the frame holds a single view, so there is no side to name.
            original_eye_side = (
                "single" if _is_left_eye is None else ("L" if _is_left_eye else "R")
            )

            # VR-13: use renamed VR_FOV_SCALE_FACTOR (was VR_DYNAMIC_FOV_PADDING_FACTOR)
            # The angular size (including the equirectangular latitude-compression
            # correction, which a fisheye does not need) is computed by the geometry.
            angular_width_deg, angular_height_deg = geometry.bbox_angular_size_deg(
                bbox_eq_single, phi
            )
            # Improvement C: use VR180MaxFOVSlider instead of the previous hard-cap of
            # 100°.  Near-camera faces can subtend >100° and were being clipped.
            _vr_max_fov = float(control.get("VR180MaxFOVSlider", 120))
            dynamic_fov_for_crop = float(
                np.clip(
                    max(angular_width_deg, angular_height_deg)
                    * self.VR_FOV_SCALE_FACTOR,
                    15.0,
                    _vr_max_fov,
                )
            )

            face_crop_tensor = equirect_converter.get_perspective_crop(
                dynamic_fov_for_crop,
                theta,
                phi,
                self.VR_PERSPECTIVE_RENDER_SIZE,
                self.VR_PERSPECTIVE_RENDER_SIZE,
                is_left_eye=_is_left_eye,
            )
            if face_crop_tensor is None or face_crop_tensor.numel() == 0:
                continue

            # Landmark detection on the crop
            crop_size = self.VR_PERSPECTIVE_RENDER_SIZE
            padding = int(crop_size * 0.025)
            dummy_bbox_on_crop = np.array(
                [padding, padding, crop_size - padding, crop_size - padding]
            )

            kpss_5_crop_list, kpss_crop_list, _ = (
                self.worker.function_worker.run_detect_landmark(
                    img=face_crop_tensor,
                    bbox=dummy_bbox_on_crop,
                    det_kpss=[],
                    detect_mode=control["LandmarkDetectModelSelection"],
                    score=control["LandmarkDetectScoreSlider"] / 100.0,
                    from_points=False,
                    use_mean_eyes=control.get("LandmarkMeanEyesToggle", False),
                )
            )

            kpss_5_crop = [kpss_5_crop_list] if len(kpss_5_crop_list) > 0 else []
            kpss_crop = [kpss_crop_list] if len(kpss_crop_list) > 0 else []

            # FW-BUG-07: fixed check — kpss_crop is a list not ndarray; use truthiness
            landmark_ok = (
                isinstance(kpss_5_crop, np.ndarray)
                and kpss_5_crop.shape[0] > 0
                or isinstance(kpss_5_crop, list)
                and len(kpss_5_crop) > 0
            )
            _crop_landmark_results.append(
                {
                    "theta": theta,
                    "phi": phi,
                    "original_eye_side": original_eye_side,
                    "is_left_eye": _is_left_eye,
                    "face_crop_tensor": face_crop_tensor,
                    "fov_used_for_crop": dynamic_fov_for_crop,
                    "kps_on_crop": kpss_5_crop[0] if landmark_ok else None,
                    "kps_all_on_crop": (
                        kpss_crop[0] if (landmark_ok and kpss_crop) else None
                    ),
                }
            )

        # Improvement I / VR-LANDMARK-MIRROR:
        # Step 1: for faces that failed landmark detection, retry with a reduced
        # score threshold (half of user setting) before falling back to stereo copy.
        # This avoids importing wrong-eye keypoints when the face is clearly present
        # in the crop but the detector is marginally below threshold.
        _lm_score_orig = control["LandmarkDetectScoreSlider"] / 100.0
        _lm_score_retry = max(0.05, _lm_score_orig * 0.5)
        _lm_detect_mode = control["LandmarkDetectModelSelection"]
        _lm_use_mean = control.get("LandmarkMeanEyesToggle", False)
        for _fd in _crop_landmark_results:
            if _fd["kps_on_crop"] is not None:
                continue
            _cs = _fd["face_crop_tensor"].shape[-1]
            _pad = int(_cs * 0.025)
            _dummy_bb = np.array([_pad, _pad, _cs - _pad, _cs - _pad])
            _r5, _rall, _ = self.worker.function_worker.run_detect_landmark(
                img=_fd["face_crop_tensor"],
                bbox=_dummy_bb,
                det_kpss=[],
                detect_mode=_lm_detect_mode,
                score=_lm_score_retry,
                from_points=False,
                use_mean_eyes=_lm_use_mean,
            )
            if len(_r5) > 0:
                _fd["kps_on_crop"] = _r5
                _fd["kps_all_on_crop"] = _rall if len(_rall) > 0 else None

        # Step 2: VR-LANDMARK-MIRROR — if a face crop still has no landmarks but the
        # partner eye view (same face, other stereo half) succeeded, reuse those
        # keypoints as a last resort.  Both eyes look the same way, so the partner
        # sits at the identical eye-relative angle — one coverage width away in
        # frame-level theta (180° on standard VR180).
        if geometry.both_eyes:
            _kps_cache: dict[tuple[float, float], tuple] = {}
            for _fd in _crop_landmark_results:
                if _fd["kps_on_crop"] is not None:
                    _kps_cache[(_fd["theta"], _fd["phi"])] = (
                        _fd["kps_on_crop"],
                        _fd["kps_all_on_crop"],
                    )
            for _fd in _crop_landmark_results:
                if _fd["kps_on_crop"] is None:
                    _mir_theta = geometry.partner_theta(
                        _fd["theta"], _fd["is_left_eye"]
                    )
                    for (_ct, _cp), _ckps in _kps_cache.items():
                        if abs(_ct - _mir_theta) < 1.0 and abs(_cp - _fd["phi"]) < 0.5:
                            _fd["kps_on_crop"], _fd["kps_all_on_crop"] = _ckps
                            break

        # Phase 2: recognition and target matching for all faces with valid landmarks.
        for _fd in _crop_landmark_results:
            if stop_event.is_set():
                break

            if _fd["kps_on_crop"] is None:
                # Improvement H: log faces that are detected but cannot be swapped
                # (landmark detection failed even after retry and stereo fallback).
                print(
                    f"[VR] Skipping face at theta={_fd['theta']:.1f}° phi={_fd['phi']:.1f}° "
                    f"({_fd['original_eye_side']}) — landmark detection failed."
                )
                del _fd["face_crop_tensor"]
                continue

            kps_on_crop = _fd["kps_on_crop"]
            kps_all_on_crop = _fd["kps_all_on_crop"]
            face_crop_tensor = _fd["face_crop_tensor"]
            theta = _fd["theta"]
            phi = _fd["phi"]
            original_eye_side = _fd["original_eye_side"]
            dynamic_fov_for_crop = _fd["fov_used_for_crop"]
            similarity_type = "Auto"

            face_emb_crop, _ = self.worker.function_worker.run_recognize_direct(
                face_crop_tensor,
                kps_on_crop,
                similarity_type,
                control["RecognitionModelSelection"],
            )

            best_target_button_vr, best_params_for_target_vr, _ = (
                self.worker._find_best_target_match(face_emb_crop, control)
            )

            # Force Swap Toggle validation
            _force_swap = control.get("ForceSwapToggle", True)

            if best_target_button_vr or _force_swap:
                # If no target but force swap is on, we ensure default dummy parameters are loaded
                if not best_target_button_vr:
                    with self.worker.lock:
                        _default_params = dict(
                            self.worker.main_window.default_parameters.data
                        )
                    best_params_for_target_vr = ParametersDict(
                        _default_params, _default_params
                    )

                denoiser_on = (
                    control.get("DenoiserUNetEnableBeforeRestorersToggle", False)
                    or control.get("DenoiserAfterFirstRestorerToggle", False)
                    or control.get("DenoiserAfterRestorersToggle", False)
                )
                if (
                    denoiser_on
                    and best_target_button_vr
                    and best_target_button_vr.assigned_kv_map is None
                    and best_target_button_vr.assigned_input_faces
                ):
                    with self.worker.models_processor.model_lock:
                        if (
                            best_target_button_vr.assigned_kv_map is None
                        ):  # Double Check inside lock
                            best_target_button_vr.calculate_assigned_input_embedding()

                analyzed_faces_for_vr.append(
                    {
                        "theta": theta,
                        "phi": phi,
                        "original_eye_side": original_eye_side,
                        "is_left_eye": _fd["is_left_eye"],
                        "face_crop_tensor": face_crop_tensor,
                        "kps_on_crop": kps_on_crop,
                        "kps_all_on_crop": kps_all_on_crop,
                        "target_button": best_target_button_vr,
                        "params": best_params_for_target_vr,
                        "fov_used_for_crop": dynamic_fov_for_crop,
                    }
                )
            else:
                if compare_mode_vr_early:
                    # No target match — save the raw crop for compare display (shown as-is)
                    unmatched_compare_crops.append(face_crop_tensor)
                else:
                    del face_crop_tensor

        # Process collected faces
        compare_mode_active_vr = (
            self.worker.is_view_face_mask or self.worker.is_view_face_compare
        )
        vr_compare_crops: list[
            tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]
        ] = []
        # Include unmatched detected faces in compare view (original shown on both sides, no mask)
        for _unmatched in unmatched_compare_crops:
            vr_compare_crops.append((_unmatched, _unmatched, None))

        for item_data in analyzed_faces_for_vr:
            original_crop_for_compare = (
                item_data["face_crop_tensor"].clone()
                if compare_mode_active_vr
                else None
            )
            if swap_button_is_checked_global or edit_button_is_checked_global:
                processed_crop_for_stitching, vr_crop_swap_mask = (
                    self._process_single_vr_perspective_crop_multi(
                        item_data["face_crop_tensor"],
                        item_data["target_button"],
                        item_data["params"],
                        control,
                        kps_5_on_crop_param=item_data["kps_on_crop"],
                        kps_all_on_crop_param=item_data["kps_all_on_crop"],
                        swap_button_is_checked_global=swap_button_is_checked_global,
                        edit_button_is_checked_global=edit_button_is_checked_global,
                        eye_side_for_debug=item_data["original_eye_side"],
                        kv_map_for_swap=(
                            getattr(item_data["target_button"], "aged_kv_map", None)
                            if item_data["params"].get("FaceReagingEnableToggle", False)
                            and getattr(item_data["target_button"], "aged_kv_map", None)
                            is not None
                            else getattr(
                                item_data["target_button"], "assigned_kv_map", None
                            )
                        ),
                    )
                )
            else:
                processed_crop_for_stitching = item_data["face_crop_tensor"]
                vr_crop_swap_mask = None

            if compare_mode_active_vr and original_crop_for_compare is not None:
                vr_compare_crops.append(
                    (
                        original_crop_for_compare,
                        processed_crop_for_stitching,
                        vr_crop_swap_mask,
                    )
                )

            # VR-07: append (is_left_eye, tensor, theta, phi, fov) tuple instead of dict entry
            processed_perspective_crops_details.append(
                (
                    item_data["is_left_eye"],
                    processed_crop_for_stitching,
                    item_data["theta"],
                    item_data["phi"],
                    item_data["fov_used_for_crop"],
                )
            )
            del item_data["face_crop_tensor"]

        # Stitch back
        # FW-PERF-12: skip PerspectiveConverter instantiation when there are no crops to stitch
        if not processed_perspective_crops_details:
            _result_no_stitch = original_equirect_tensor_for_vr
            _det_faces_eq = [{"bbox": row[:4]} for row in bboxes_eq_np]
            if control.get("ShowAllDetectedFacesBBoxToggle", False) and _det_faces_eq:
                _result_no_stitch = draw_bounding_boxes_on_detected_faces(
                    _result_no_stitch, _det_faces_eq
                )
            if control.get("ShowByteTrackBBoxToggle", False) and _det_faces_eq:
                _result_no_stitch = draw_bounding_boxes_on_detected_faces(
                    _result_no_stitch, _det_faces_eq, color_rgb=[255, 165, 0]
                )
            return _result_no_stitch

        final_equirect_torch_cxhxw_rgb_uint8 = original_equirect_tensor_for_vr.clone()

        # Improvement K: cache PerspectiveConverter across frames (like E2P converter).
        # PerspectiveConverter only uses img_numpy_rgb_uint8 for its dimensions; the
        # per-frame image content is not stored in the converter itself (stitch_single_perspective
        # receives the target tensor directly).  We can therefore reuse the cached instance
        # whenever the geometry is unchanged, avoiding repeated kernel allocation.
        # NOTE: deliberately uses _vr_p2e_geometry (NOT _vr_geometry) for this check.
        # _vr_geometry is updated by the E2P branch above, so by the time we reach this
        # point it always equals geometry.  A separate tracking attribute ensures the
        # P2E converter is properly recreated when the geometry changes.
        if self._vr_p2e_converter is None or self._vr_p2e_geometry != geometry:
            self._vr_p2e_converter = PerspectiveConverter(
                img_numpy_rgb_uint8,
                device=self.worker.models_processor.device,
                geometry=geometry,
            )
            self._vr_p2e_geometry = geometry
        p2e_converter = self._vr_p2e_converter

        # VR-15: check stop_event in stitching loop so we can abort early
        # VR-07: iterate over list of (is_left_eye, tensor, theta, phi, fov) tuples
        for (
            _is_left_eye,
            _crop_tensor,
            _theta,
            _phi,
            _fov,
        ) in processed_perspective_crops_details:
            if stop_event is not None and stop_event.is_set():
                break
            p2e_converter.stitch_single_perspective(
                target_equirect_torch_cxhxw_rgb_uint8=final_equirect_torch_cxhxw_rgb_uint8,
                processed_crop_torch_cxhxw_rgb_uint8=_crop_tensor,
                theta=_theta,
                phi=_phi,
                fov=_fov,
                is_left_eye=_is_left_eye,
            )
            # FW-MEM-02: release the crop tensor immediately after stitching
            del _crop_tensor

        processed_tensor_rgb_uint8 = final_equirect_torch_cxhxw_rgb_uint8

        # --- Bounding box overlays on the stitched equirectangular image ---
        _det_faces_eq = [{"bbox": row[:4]} for row in bboxes_eq_np]
        if control.get("ShowAllDetectedFacesBBoxToggle", False) and _det_faces_eq:
            processed_tensor_rgb_uint8 = draw_bounding_boxes_on_detected_faces(
                processed_tensor_rgb_uint8, _det_faces_eq
            )
        if control.get("ShowByteTrackBBoxToggle", False) and _det_faces_eq:
            processed_tensor_rgb_uint8 = draw_bounding_boxes_on_detected_faces(
                processed_tensor_rgb_uint8, _det_faces_eq, color_rgb=[255, 165, 0]
            )

        # Cleanup
        # VR-08: equirect_converter is now self._vr_converter (cached); do not del it
        # Improvement K: p2e_converter is now self._vr_p2e_converter (cached); do not del it
        del processed_perspective_crops_details, analyzed_faces_for_vr
        # FW-MEM-1: removed torch.cuda.empty_cache() — calling it per-frame
        # defeats the CUDA caching allocator and causes unnecessary overhead.

        # VR-MEM-03: periodic CUDA allocator flush to prevent VRAM fragmentation
        # over long recording sessions. Per-worker counter
        # staggers flushes across 8 pool workers so they don't all flush simultaneously.
        # empty_cache() returns fragmented free blocks to CUDA; gc.collect() clears
        # Python cyclic garbage that may hold tensor references.
        self._vr_processed_count += 1
        if self._vr_processed_count % 300 == 0:
            platform_support.empty_cache()
            gc.collect()

        # --- VR Compare/Mask view ---
        # Build side-by-side crop strips when Face Compare or Face Mask is active.
        if compare_mode_active_vr and vr_compare_crops:
            imgs_to_vstack = []
            for original_chw, processed_chw, mask_1chw_float in vr_compare_crops:
                imgs_to_cat: list[torch.Tensor] = []
                if self.worker.is_view_face_compare:
                    imgs_to_cat.append(original_chw)
                    imgs_to_cat.append(processed_chw)
                elif not self.worker.is_view_face_mask:
                    # Neither compare nor mask — shouldn't reach here, but show processed
                    imgs_to_cat.append(processed_chw)
                if self.worker.is_view_face_mask and mask_1chw_float is not None:
                    mask_display = torch.sub(1, mask_1chw_float).repeat(3, 1, 1)
                    mask_display = torch.mul(mask_display, 255.0).type(torch.uint8)
                    imgs_to_cat.append(mask_display)
                if imgs_to_cat:
                    imgs_to_vstack.append(torch.cat(imgs_to_cat, dim=2))
            if imgs_to_vstack:
                max_width = max(t.size(2) for t in imgs_to_vstack)
                padded = [
                    torch.nn.functional.pad(t, (0, max_width - t.size(2), 0, 0))
                    for t in imgs_to_vstack
                ]
                return torch.cat(padded, dim=1)

        return processed_tensor_rgb_uint8
