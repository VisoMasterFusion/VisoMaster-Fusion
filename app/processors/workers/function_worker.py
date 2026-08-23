from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import threading


# --- Internal Sub-Processor Imports ---
from app.processors.face_detectors import FaceDetectors
from app.processors.face_landmark_detectors import FaceLandmarkDetectors
from app.processors.face_masks import FaceMasks
from app.processors.face_restorers import FaceRestorers
from app.processors.face_swappers import FaceSwappers
from app.processors.frame_enhancers import FrameEnhancers
from app.processors.face_editors import FaceEditors
from app.processors.face_reaging import FaceReaging
from app.processors.perform_recast import PerformRecast
from app.processors.face_denoiser import FaceDenoiser
from app.processors.frame_edits import FrameEdits
from app.processors.models_data import arcface_mapping_model_dict
from app.processors.utils import platform_support

if TYPE_CHECKING:
    from app.processors.models_processor import ModelsProcessor


class FunctionWorker:
    """
    Centralized dispatcher for all AI model function calls.
    Acts as a Thread-Safe Facade pattern.

    By instantiating sub-processors here, we ensure that:
    1. Worker threads do not duplicate sub-processor caches (saving VRAM).
    2. ModelsProcessor is decoupled from inference routing.
    """

    def __init__(self, models_processor: "ModelsProcessor") -> None:
        """
        Initializes the FunctionWorker and all sub-processors.

        Args:
            models_processor: The central Model resource manager. Passed to sub-processors
                              so they can request model loading/unloading safely.
        """
        self.mp = models_processor

        # --- Granular Locking ---
        self._session_locks: dict[int, threading.RLock] = {}
        self._session_locks_mutex = threading.Lock()

        # Initialize Sub-Processors centrally.
        # We pass `self.mp` for VRAM management, and `self` so they can route calls through this Facade.
        self.face_detectors = FaceDetectors(self.mp, self)
        self.face_landmark_detectors = FaceLandmarkDetectors(self.mp, self)
        self.face_masks = FaceMasks(self.mp, self)
        self.face_restorers = FaceRestorers(self.mp, self)
        self.face_swappers = FaceSwappers(self.mp, self)
        self.frame_enhancers = FrameEnhancers(self.mp, self)
        self.face_editors = FaceEditors(self.mp, self)
        self.face_reaging = FaceReaging(self.mp, self)
        self.perform_recast = PerformRecast(self.mp, self)
        self.face_denoiser = FaceDenoiser(self.mp, self)
        self.frame_edits = FrameEdits(self.mp, self)

    def _get_session_lock(self, session: Any) -> threading.RLock:
        """Retrieves or creates a dedicated lock for a specific ORT Session."""
        session_id = id(session)
        with self._session_locks_mutex:
            if session_id not in self._session_locks:
                self._session_locks[session_id] = threading.RLock()
            return self._session_locks[session_id]

    def run_ort_with_iobinding(self, session: Any, io_binding: Any) -> None:
        """Serialize shared ONNX/TensorRT session execution ACROSS THE SAME MODEL only."""
        session_lock = self._get_session_lock(session)

        # 1. PRE-INFERENCE SYNC (Outside the lock)
        # Ensure PyTorch has finished writing inputs to the GPU on the isolated worker_stream.
        # This occurs outside the lock so other threads aren't blocked during tensor prep.
        if self.mp.device_type == "cuda":
            platform_support.blocking_stream_sync()
        elif self.mp.device_type != "cpu":
            self.mp.syncvec.cpu()

        # 2. INFERENCE AND CORRECT POST-SYNC (Inside the lock)
        with session_lock:
            # Execute ONNX Runtime / TensorRT.
            session.run_with_iobinding(io_binding)

            # 3. POST-INFERENCE SYNC (MUST BE INSIDE THE LOCK)
            # We use ORT's native synchronization instead of PyTorch's stream sync.
            # PyTorch stream sync does not order ORT's internal stream.
            # This guarantees ORT is completely finished writing the pre-allocated output tensors
            # before we release the lock to the next worker thread.
            io_binding.synchronize_outputs()

    # --- Models Unloaders ---

    def unload_face_detector_models(self) -> None:
        """Unloads the active face detector model under the model lock."""
        with self.mp.model_lock:
            self.face_detectors.unload_models()

    def unload_face_landmark_detector_models(self) -> None:
        """Unloads the active face landmark detector model under the model lock."""
        with self.mp.model_lock:
            self.face_landmark_detectors.unload_models()

    def unload_face_editor_models(self) -> None:
        """Unloads all loaded face editor models under the model lock."""
        with self.mp.model_lock:
            self.face_editors.unload_models()

    def unload_perform_recast_models(self) -> None:
        """Unloads all loaded PerformRecast (Recast mode) models under the lock."""
        with self.mp.model_lock:
            self.perform_recast.unload_models()

    def unload_face_mask_models(self) -> None:
        """Unloads all loaded face mask models under the model lock."""
        with self.mp.model_lock:
            self.face_masks.unload_models()

    def unload_frame_enhancer_models(self) -> None:
        """Unloads all loaded frame enhancer models under the model lock."""
        with self.mp.model_lock:
            self.frame_enhancers.unload_models()

    def unload_face_restorer_models(self) -> None:
        """Unloads all loaded face restorer models under the model lock."""
        with self.mp.model_lock:
            self.face_restorers.unload_models()

    def unload_face_swapper_models(self) -> None:
        """Unloads all loaded face swapper models under the model lock."""
        with self.mp.model_lock:
            self.face_swappers.unload_models()

    # HARDWARE ORCHESTRATION

    def clear_gpu_memory(self) -> None:
        """
        Force-unloads every loaded model (ONNX, DFM, KV Extractor, CLIP) and
        releases all GPU memory safely across all sub-processors.
        """
        print("[INFO] Clearing GPU Memory: Unloading all models...")
        # 1. Stop processing to ensure no workers try to access models during unload
        self.mp.main_window.video_processor.stop_processing()

        # --- Clear Global Target Face KV Caches ---
        # The Denoiser generates massive PyTorch KV cache tensors per face.
        # Because they are attached to the global UI target_faces dictionary,
        # they permanently survive "Clear VRAM" unless explicitly deleted here.
        try:
            for target_face in self.mp.main_window.target_faces.values():
                target_face.assigned_kv_map = None
                if hasattr(target_face, "aged_kv_map"):
                    target_face.aged_kv_map = None
        except Exception as e:
            print(f"[WARN] Could not clear target face KV caches: {e}")

        # 2. Bypass KeepModelsAliveToggle
        self.mp.force_unload_in_progress = True
        try:
            self.unload_face_detector_models()
            self.unload_face_landmark_detector_models()
            self.unload_face_mask_models()
            self.unload_face_restorer_models()
            self.unload_face_swapper_models()
            self.unload_frame_enhancer_models()
            self.unload_face_editor_models()
            self.unload_perform_recast_models()

            self.mp.delete_models()
            self.mp.delete_models_dfm()

            self.face_denoiser.unload_kv_extractor()
            if self.mp.clip_session:
                del self.mp.clip_session
                self.mp.clip_session = []
        finally:
            self.mp.force_unload_in_progress = False

        # 3. Clean up VRAM
        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("[INFO] GPU Memory Cleared.")

    def switch_providers_priority(self, provider_name: str) -> str:
        """
        Reconfigures the ONNX Runtime provider list and the active device.
        Safely unloads existing sessions and migrates persistent tensors to the new device.
        """
        # 1. Release existing ORT sessions from the old provider to free VRAM
        self.unload_face_detector_models()
        self.unload_face_mask_models()  # Use local wrapper
        self.unload_face_swapper_models()
        self.unload_face_landmark_detector_models()
        self.unload_face_restorer_models()

        # 2. Delegate the raw hardware string/dict configuration to the Resource Manager
        new_provider = self.mp.update_provider_configuration(provider_name)

        # 3. Migrate persistent tensors to the new device to prevent device-mismatch crashes
        if hasattr(self.mp, "lp_mask_crop_latent"):
            self.mp.lp_mask_crop_latent = self.mp.lp_mask_crop_latent.to(self.mp.device)

        if hasattr(self.face_editors, "lp_mask_crop"):
            self.face_editors.lp_mask_crop = self.face_editors.lp_mask_crop.to(
                self.mp.device
            )

        if hasattr(self.face_denoiser, "alphas_cumprod_torch"):
            self.face_denoiser.alphas_cumprod_torch = (
                self.face_denoiser.alphas_cumprod_torch.to(self.mp.device)
            )

        return new_provider

    # --- Face Detector & Landmark Detector Wrappers ---

    def run_detect(
        self,
        img: np.ndarray,
        detect_mode: str = "RetinaFace",
        max_num: int = 1,
        score: float = 0.5,
        input_size: Tuple[int, int] = (512, 512),
        use_landmark_detection: bool = False,
        landmark_detect_mode: str = "203",
        landmark_score: float = 0.5,
        from_points: bool = False,
        rotation_angles: Optional[List[int]] = None,
        **kwargs: Any,
    ) -> Any:
        rotation_angles = rotation_angles or [0]
        return self.face_detectors.run_detect(
            img,
            detect_mode,
            max_num,
            score,
            input_size,
            use_landmark_detection,
            landmark_detect_mode,
            landmark_score,
            from_points,
            rotation_angles,
            **kwargs,
        )

    def run_detect_landmark(
        self,
        img: np.ndarray,
        bbox: np.ndarray,
        det_kpss: np.ndarray,
        detect_mode: str = "203",
        score: float = 0.5,
        from_points: bool = False,
        **kwargs: Any,
    ) -> Any:
        return self.face_landmark_detectors.run_detect_landmark(
            img, bbox, det_kpss, detect_mode, score, from_points, **kwargs
        )

    def get_arcface_model(self, face_swapper_model: str) -> str:
        if face_swapper_model in arcface_mapping_model_dict:
            return arcface_mapping_model_dict[face_swapper_model]
        else:
            raise ValueError(f"Face swapper model {face_swapper_model} not found.")

    def run_recognize_direct(
        self,
        img: np.ndarray,
        kps: np.ndarray,
        similarity_type: str = "Auto",
        arcface_model: str = "Inswapper128ArcFace",
    ) -> Any:
        return self.face_swappers.run_recognize_direct(
            img, kps, similarity_type, arcface_model
        )

    def findCosineDistance(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        vector1 = vector1.ravel()
        vector2 = vector2.ravel()
        cos_dist = 1 - np.dot(vector1, vector2) / (
            np.linalg.norm(vector1) * np.linalg.norm(vector2)
        )
        return 100 - cos_dist * 50

    def reset_face_tracker(self) -> None:
        """Thread-safe facade to reset the face tracking state."""
        self.face_detectors.reset_tracker()

    # --- Frame Enhancer & Utility Wrappers ---

    def enhance_core(self, img: torch.Tensor, control: dict) -> torch.Tensor:
        """Thread-safe facade for frame enhancement."""
        return self.frame_enhancers.enhance_core(img, control=control)

    def set_frame_edits_transforms(
        self, t256_face: Any, interpolation_expression_faceeditor_back: Any
    ) -> None:
        """Thread-safe facade for configuring FrameEdits interpolations."""
        self.frame_edits.set_transforms(
            t256_face, interpolation_expression_faceeditor_back
        )

    # --- Face Swapper Wrappers ---

    def calc_inswapper_latent(self, source_embedding: np.ndarray) -> np.ndarray:
        return self.face_swappers.calc_inswapper_latent(source_embedding)

    def run_inswapper(
        self, image: np.ndarray, embedding: np.ndarray, output: Any
    ) -> None:
        self.face_swappers.run_inswapper(image, embedding, output)

    def run_inswapper_batched(
        self, images: List[np.ndarray], embedding: np.ndarray, output: Any
    ) -> None:
        self.face_swappers.run_inswapper_batched(images, embedding, output)

    def calc_swapper_latent_alphaface(
        self, source_embedding: np.ndarray
    ) -> np.ndarray | None:
        return self.face_swappers.calc_swapper_latent_alphaface(source_embedding)

    def run_swapper_alphaface(
        self, image: torch.Tensor, embedding: torch.Tensor, output: torch.Tensor
    ) -> None:
        self.face_swappers.run_swapper_alphaface(image, embedding, output)

    def run_swapper_alphaface_batched(
        self, images: torch.Tensor, embedding: torch.Tensor, output: torch.Tensor
    ) -> None:
        self.face_swappers.run_swapper_alphaface_batched(images, embedding, output)

    def calc_swapper_latent_iss(
        self, source_embedding: np.ndarray, version: str = "A"
    ) -> np.ndarray:
        return self.face_swappers.calc_swapper_latent_iss(source_embedding, version)

    def run_iss_swapper(
        self, image: np.ndarray, embedding: np.ndarray, output: Any, version: str = "A"
    ) -> None:
        self.face_swappers.run_iss_swapper(image, embedding, output, version)

    def calc_swapper_latent_simswap512(
        self, source_embedding: np.ndarray
    ) -> np.ndarray:
        return self.face_swappers.calc_swapper_latent_simswap512(source_embedding)

    def run_swapper_simswap512(
        self, image: np.ndarray, embedding: np.ndarray, output: Any
    ) -> None:
        self.face_swappers.run_swapper_simswap512(image, embedding, output)

    def calc_swapper_latent_ghost(self, source_embedding: np.ndarray) -> np.ndarray:
        return self.face_swappers.calc_swapper_latent_ghost(source_embedding)

    def run_swapper_ghostface(
        self,
        image: np.ndarray,
        embedding: np.ndarray,
        output: Any,
        swapper_model: str = "GhostFace-v2",
    ) -> None:
        self.face_swappers.run_swapper_ghostface(
            image, embedding, output, swapper_model
        )

    def calc_swapper_latent_cscs(self, source_embedding: np.ndarray) -> np.ndarray:
        return self.face_swappers.calc_swapper_latent_cscs(source_embedding)

    def run_swapper_cscs(
        self, image: np.ndarray, embedding: np.ndarray, output: Any
    ) -> None:
        self.face_swappers.run_swapper_cscs(image, embedding, output)

    def manage_dfm_swapper_model_state(self, swapper_model: str) -> None:
        """Thread-safe state management for model switching."""
        with self.mp.model_lock:
            self.face_swappers._manage_model(swapper_model)
            self.face_swappers.current_swapper_model = swapper_model  # type: ignore[assignment]

    # --- Face Editor Wrappers ---

    def apply_face_expression_restorer(
        self,
        original_face_512: torch.Tensor,
        swap: torch.Tensor,
        parameters: dict,
        control: dict,
        driving_kps: Any = None,
    ) -> torch.Tensor:
        return self.frame_edits.apply_face_expression_restorer(
            original_face_512, swap, parameters, control, driving_kps=driving_kps
        )

    def swap_edit_face_core(
        self,
        swap: torch.Tensor,
        original_face: torch.Tensor,
        parameters: dict,
        control: dict,
    ) -> torch.Tensor:
        return self.frame_edits.swap_edit_face_core(
            swap, original_face, parameters, control
        )

    def swap_edit_face_core_makeup(
        self, img: torch.Tensor, kps: np.ndarray, parameters: dict, control: dict
    ) -> torch.Tensor:
        """Thread-safe facade for applying face makeup edits."""
        return self.frame_edits.swap_edit_face_core_makeup(
            img, kps, parameters, control
        )

    def apply_face_makeup(self, img: np.ndarray, parameters: dict) -> np.ndarray:
        return self.face_editors.apply_face_makeup(img, parameters)

    def lp_motion_extractor(
        self, img: np.ndarray, face_editor_type: str = "Human-Face", **kwargs: Any
    ) -> dict:
        return self.face_editors.lp_motion_extractor(img, face_editor_type, **kwargs)

    def lp_appearance_feature_extractor(
        self, img: np.ndarray, face_editor_type: str = "Human-Face"
    ) -> Any:
        return self.face_editors.lp_appearance_feature_extractor(img, face_editor_type)

    def lp_retarget_eye(
        self,
        kp_source: torch.Tensor,
        eye_close_ratio: torch.Tensor,
        face_editor_type: str = "Human-Face",
    ) -> torch.Tensor:
        return self.face_editors.lp_retarget_eye(
            kp_source, eye_close_ratio, face_editor_type
        )

    def lp_retarget_lip(
        self,
        kp_source: torch.Tensor,
        lip_close_ratio: torch.Tensor,
        face_editor_type: str = "Human-Face",
    ) -> torch.Tensor:
        return self.face_editors.lp_retarget_lip(
            kp_source, lip_close_ratio, face_editor_type
        )

    def lp_stitch(
        self,
        kp_source: torch.Tensor,
        kp_driving: torch.Tensor,
        face_editor_type: str = "Human-Face",
    ) -> torch.Tensor:
        return self.face_editors.lp_stitch(kp_source, kp_driving, face_editor_type)

    def lp_stitching(
        self,
        kp_source: torch.Tensor,
        kp_driving: torch.Tensor,
        face_editor_type: str = "Human-Face",
    ) -> torch.Tensor:
        return self.face_editors.lp_stitching(kp_source, kp_driving, face_editor_type)

    def lp_warp_decode(
        self,
        feature_3d: torch.Tensor,
        kp_source: torch.Tensor,
        kp_driving: torch.Tensor,
        face_editor_type: str = "Human-Face",
    ) -> torch.Tensor:
        return self.face_editors.lp_warp_decode(
            feature_3d, kp_source, kp_driving, face_editor_type
        )

    def apply_face_shaping_gpu(
        self, img: torch.Tensor, kps: np.ndarray, parameters: dict
    ) -> torch.Tensor:
        return self.frame_edits.apply_face_shaping_gpu(img, kps, parameters)

    # --- PerformRecast Wrappers ---

    def perform_recast_prewarm(self) -> None:
        self.perform_recast.prewarm()

    def perform_recast_motion(self, img: torch.Tensor) -> dict:
        return self.perform_recast.motion(img)

    def perform_recast_build_source_info(self, x_s_info: dict) -> dict:
        return self.perform_recast.build_source_info(x_s_info)

    def perform_recast_extract_appearance(self, img: torch.Tensor) -> torch.Tensor:
        return self.perform_recast.extract_appearance(img)

    def perform_recast_compose_driven_keypoints(
        self, source_info: dict, exp_d: torch.Tensor, **kwargs: Any
    ) -> torch.Tensor:
        return self.perform_recast.compose_driven_keypoints(
            source_info, exp_d, **kwargs
        )

    def perform_recast_warp_decode(
        self, f_s: torch.Tensor, x_s: torch.Tensor, x_d_i: torch.Tensor
    ) -> torch.Tensor:
        return self.perform_recast.warp_decode(f_s, x_s, x_d_i)

    def apply_perform_recast(
        self,
        original_face_512: torch.Tensor,
        swap: torch.Tensor,
        parameters: dict,
        control: dict,
        driving_kps: Any = None,
    ) -> torch.Tensor:
        return self.frame_edits.apply_perform_recast(
            original_face_512, swap, parameters, control, driving_kps=driving_kps
        )

    # --- Face Mask Wrappers ---

    def get_faceparser_labels(self, img_uint8_3x512x512: torch.Tensor) -> torch.Tensor:
        return self.face_masks._faceparser_labels(img_uint8_3x512x512)

    def run_CLIPs(
        self, img: np.ndarray, CLIPText: str, CLIPAmount: float
    ) -> np.ndarray:
        return self.face_masks.run_CLIPs(img, CLIPText, CLIPAmount)

    def apply_occlusion(
        self,
        img: torch.Tensor | np.ndarray,
        amount: float,
        parameters: Optional[dict] = None,
        original_face_512: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | np.ndarray:
        return self.face_masks.apply_occlusion(
            img, amount, parameters=parameters, original_face_512=original_face_512
        )

    def apply_dfl_xseg(
        self,
        img: np.ndarray,
        amount: float,
        mouth: Any,
        parameters: dict,
        inner_mouth_mask: Any,
    ) -> np.ndarray:
        return self.face_masks.apply_dfl_xseg(
            img, amount, mouth, parameters, inner_mouth_mask
        )

    def process_masks_and_masks(
        self,
        swap_restorecalc: np.ndarray,
        original_face_512: np.ndarray,
        parameters: dict,
        control: dict,
    ) -> Any:
        return self.face_masks.process_masks_and_masks(
            swap_restorecalc, original_face_512, parameters, control
        )

    def get_mouth_overlay(
        self, swap: torch.Tensor, original_face_512: torch.Tensor, parameters: dict
    ) -> Optional[Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]]:
        return self.face_masks.get_mouth_overlay(swap, original_face_512, parameters)

    def apply_vgg_mask_simple(
        self,
        swap: torch.Tensor,
        original_face_512: torch.Tensor,
        mask_input_vgg: torch.Tensor,
        center_pct: float,
        softness_pct: float,
        feature_layer: str,
        mode: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.face_masks.apply_vgg_mask_simple(
            swap,
            original_face_512,
            mask_input_vgg,
            center_pct=center_pct,
            softness_pct=softness_pct,
            feature_layer=feature_layer,
            mode=mode,
        )

    def apply_perceptual_diff_onnx(
        self,
        swapped_face_resized: torch.Tensor,
        original_face_resized: torch.Tensor,
        diff_mask_128: torch.Tensor,
        lower_thresh: float,
        placeholder: int,
        upper_thresh: float,
        upper_value: float,
        middle_value: float,
        feature_layer_type: str,
        flag: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.face_masks.apply_perceptual_diff_onnx(
            swapped_face_resized,
            original_face_resized,
            diff_mask_128,
            lower_thresh,
            placeholder,
            upper_thresh,
            upper_value,
            middle_value,
            feature_layer_type,
            flag,
        )

    def restore_mouth(
        self,
        img_orig: np.ndarray,
        img_swap: np.ndarray,
        kpss_orig: np.ndarray,
        blend_alpha: float = 0.5,
        feather_radius: int = 10,
        size_factor: float = 0.5,
        radius_factor_x: float = 1.0,
        radius_factor_y: float = 1.0,
        x_offset: int = 0,
        y_offset: int = 0,
    ) -> np.ndarray:
        return self.face_masks.restore_mouth(
            img_orig,
            img_swap,
            kpss_orig,
            blend_alpha,
            feather_radius,
            size_factor,
            radius_factor_x,
            radius_factor_y,
            x_offset,
            y_offset,
        )

    def restore_eyes(
        self,
        img_orig: np.ndarray,
        img_swap: np.ndarray,
        kpss_orig: np.ndarray,
        blend_alpha: float = 0.5,
        feather_radius: int = 10,
        size_factor: float = 3.5,
        radius_factor_x: float = 1.0,
        radius_factor_y: float = 1.0,
        x_offset: int = 0,
        y_offset: int = 0,
        eye_spacing_offset: int = 0,
    ) -> np.ndarray:
        return self.face_masks.restore_eyes(
            img_orig,
            img_swap,
            kpss_orig,
            blend_alpha,
            feather_radius,
            size_factor,
            radius_factor_x,
            radius_factor_y,
            x_offset,
            y_offset,
            eye_spacing_offset,
        )

    # --- Face Denoiser & Ref-LDM Wrappers ---

    def get_kv_map_for_face(
        self, image: Any, unload_after: bool = True
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        return self.face_denoiser.get_kv_map_for_face(image, unload_after)

    def apply_denoiser_unet(
        self,
        image_cxhxw_uint8: torch.Tensor,
        reference_kv_map: Dict | None,
        use_reference_exclusive_path: bool,
        denoiser_mode: str = "Single Step (Fast)",
        denoiser_single_step_t: int = 1,
        denoiser_ddim_steps: int = 20,
        denoiser_cfg_scale: float = 1.0,
        denoiser_ddim_eta: float = 0.0,
        base_seed: int = 220,
        latent_sharpening_strength: float = 0.0,
        enable_color_correction: bool = True,
        color_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.face_denoiser.apply_denoiser_unet(
            image_cxhxw_uint8,
            reference_kv_map,
            use_reference_exclusive_path,
            denoiser_mode,
            denoiser_single_step_t,
            denoiser_ddim_steps,
            denoiser_cfg_scale,
            denoiser_ddim_eta,
            base_seed,
            latent_sharpening_strength,
            enable_color_correction,
            color_mask,
        )

    def run_vae_encoder(
        self,
        image_srgb_float_minus1_1_batched: torch.Tensor,
        encoded_latent_direct_vae_out_bchw: torch.Tensor,
    ) -> None:
        self.face_restorers.run_vae_encoder(
            image_srgb_float_minus1_1_batched, encoded_latent_direct_vae_out_bchw
        )

    def run_ref_ldm_unet(
        self,
        x_noisy_plus_lq_latent: torch.Tensor,
        timesteps_tensor: torch.Tensor,
        is_ref_flag_tensor: torch.Tensor,
        use_reference_exclusive_path_globally_tensor: torch.Tensor,
        kv_tensor_map: Dict | None,
        output_unet_tensor: torch.Tensor,
    ) -> None:
        self.face_restorers.run_ref_ldm_unet(
            x_noisy_plus_lq_latent,
            timesteps_tensor,
            is_ref_flag_tensor,
            use_reference_exclusive_path_globally_tensor,
            kv_tensor_map,
            output_unet_tensor,
        )

    def run_vae_decoder(
        self,
        latent_for_vae_decoder: torch.Tensor,
        decoded_image_normalized_bchw: torch.Tensor,
    ) -> None:
        self.face_restorers.run_vae_decoder(
            latent_for_vae_decoder, decoded_image_normalized_bchw
        )

    def ensure_denoiser_models_loaded(self) -> None:
        self.face_denoiser.ensure_denoiser_models_loaded()

    @property
    def denoiser_kv_extraction_lock(self):
        """Exposes the KV extraction lock safely for batch processing."""
        return self.face_denoiser.kv_extraction_lock

    def unload_denoiser_models(self) -> None:
        """Thread-safe unload of denoiser UNet and VAEs."""
        with self.mp.model_lock:
            self.face_denoiser.unload_models()

    def unload_denoiser_kv_extractor(self) -> None:
        """Thread-safe unload of the KV extractor."""
        with self.mp.model_lock:
            self.face_denoiser.unload_kv_extractor()

    # --- Face Restorer Wrappers ---

    def apply_facerestorer(
        self,
        swapped_face_upscaled: np.ndarray,
        restorer_det_type: str,
        restorer_type: str,
        restorer_blend: float,
        fidelity_weight: float,
        detect_score: float,
        target_kps: np.ndarray,
        slot_id: int = 1,
    ) -> np.ndarray:
        return self.face_restorers.apply_facerestorer(
            swapped_face_upscaled,
            restorer_det_type,
            restorer_type,
            restorer_blend,
            fidelity_weight,
            detect_score,
            target_kps,
            slot_id=slot_id,
        )

    def apply_reaging(
        self,
        face_chw_uint8: torch.Tensor,
        source_age: int,
        target_age: int,
    ) -> torch.Tensor:
        return self.face_reaging.apply_reaging(face_chw_uint8, source_age, target_age)
