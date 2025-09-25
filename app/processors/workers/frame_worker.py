import traceback
from typing import TYPE_CHECKING, Dict
import threading
import math
from math import floor, ceil

from PIL import Image
from app.ui.widgets import widget_components
import torch
from skimage import transform as trans
import kornia.enhance as ke
import kornia.color as kc

import os

from torchvision.transforms import v2
import torchvision
from torchvision import transforms
from torchvision import models
from torchvision.transforms.functional import normalize

import numpy as np
import cv2
import torch.nn.functional as F

from app.processors.models_data import models_dir # For UNet model existence check
from app.processors.utils import faceutil
import app.ui.widgets.actions.common_actions as common_widget_actions
from app.ui.widgets.actions import video_control_actions
from app.helpers.miscellaneous import ParametersDict, get_scaling_transforms
from app.helpers.vr_utils import EquirectangularConverter, PerspectiveConverter

if TYPE_CHECKING:
    from app.ui.main_ui import MainWindow

torchvision.disable_beta_transforms_warning()

# These will be dynamically set per frame based on control parameters
t512, t384, t256, t128, interpolation_get_cropped_face_kps, interpolation_original_face_128_384, interpolation_original_face_512, interpolation_Untransform, interpolation_scaleback, t256_face, interpolation_expression_faceeditor_back, interpolation_block_shift = None, None, None, None, None, None, None, None, None, None, None, None

class FrameWorker(threading.Thread):
    def __init__(self, frame, main_window: 'MainWindow', frame_number, frame_queue, is_single_frame=False):
        super().__init__()
        self.frame_queue = frame_queue
        self.frame = frame # Expected to be HxWxC RGB uint8 NumPy array
        self.main_window = main_window
        self.frame_number = frame_number
        self.models_processor = main_window.models_processor
        self.video_processor = main_window.video_processor
        self.is_single_frame = is_single_frame
        self.parameters = {} # Will be populated from main_window.parameters
        # VR specific constants
        self.VR_PERSPECTIVE_RENDER_SIZE = 512 # Pixels, for rendering perspective crops
        self.VR_DYNAMIC_FOV_PADDING_FACTOR = 1.0 # Padding factor for dynamic FOV calculation
        self.is_view_face_compare: bool = False
        self.is_view_face_mask: bool = False
        self.lock = threading.Lock()

    def set_scaling_transforms(self, control_params):
        global t512, t384, t256, t128, interpolation_get_cropped_face_kps, interpolation_original_face_128_384, interpolation_original_face_512, interpolation_Untransform, interpolation_scaleback, t256_face, interpolation_expression_faceeditor_back, interpolation_block_shift
        t512, t384, t256, t128, interpolation_get_cropped_face_kps, interpolation_original_face_128_384, interpolation_original_face_512, interpolation_Untransform, interpolation_scaleback, t256_face, interpolation_expression_faceeditor_back, interpolation_block_shift = get_scaling_transforms(control_params)

    def run(self):
        try:
            # Update parameters from markers (if exists)
            with self.main_window.models_processor.model_lock: # Ensure thread safety for UI data access
                video_control_actions.update_parameters_and_control_from_marker(self.main_window, self.frame_number)
                self.parameters = self.main_window.parameters.copy() 
                current_control_state = self.main_window.control.copy()

            self.is_view_face_compare = self.main_window.faceCompareCheckBox.isChecked()
            self.is_view_face_mask = self.main_window.faceMaskCheckBox.isChecked()

            # Determine if any processing is needed
            needs_processing = (
                self.main_window.swapfacesButton.isChecked() or
                self.main_window.editFacesButton.isChecked() or
                current_control_state.get('FrameEnhancerEnableToggle', False) or
                current_control_state.get('VR180ModeEnableToggle', False) # VR180 always processes
            )

            if needs_processing:
                if not self.frame.flags['C_CONTIGUOUS']: # Ensure input frame is C-contiguous
                    self.frame = np.ascontiguousarray(self.frame)
                # process_frame returns BGR, uint8
                processed_frame_bgr_np_uint8 = self.process_frame(current_control_state)
                # Ensure output is C-contiguous for Qt display
                self.frame = np.ascontiguousarray(processed_frame_bgr_np_uint8)
            else:
                # If no processing, just convert RGB to BGR for display
                self.frame = self.frame[..., ::-1]
                self.frame = np.ascontiguousarray(self.frame)

            # self.frame is now consistently BGR. It can be used for both pixmap creation and signal emission.
            pixmap = common_widget_actions.get_pixmap_from_frame(self.main_window, self.frame)

            if self.video_processor.file_type == 'webcam' and not self.is_single_frame:
                self.video_processor.webcam_frame_processed_signal.emit(pixmap, self.frame)
            elif not self.is_single_frame:
                self.video_processor.frame_processed_signal.emit(self.frame_number, pixmap, self.frame)
            else:  # Single frame processing (image or paused video)
                self.video_processor.single_frame_processed_signal.emit(self.frame_number, pixmap, self.frame)

            self.video_processor.frame_queue.get()
            self.video_processor.frame_queue.task_done()

            if self.video_processor.frame_queue.empty() and \
               not self.video_processor.processing and \
               self.video_processor.next_frame_to_display >= self.video_processor.max_frame_number:
                self.video_processor.stop_processing()

        except Exception as e:
            print(f"Error in FrameWorker for frame {self.frame_number}: {e}")
            traceback.print_exc()

    def tensor_to_pil(self, tensor):
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        if tensor.dim() == 3 and tensor.shape[0] == 1:
            tensor = tensor.repeat(3, 1, 1)
        if tensor.dtype == torch.float32 or tensor.dtype == torch.float64:
            tensor = (tensor * 255).clamp(0, 255).byte()
        tensor = tensor.permute(1, 2, 0).cpu().numpy()
        return Image.fromarray(tensor)
    
    def _apply_denoiser_pass(self, image_tensor_cxhxw_uint8: torch.Tensor, control: dict, pass_suffix: str, kv_map: Dict | None) -> torch.Tensor:
        use_exclusive_path = control.get('UseReferenceExclusivePathToggle', False)
        denoiser_seed_from_slider_val = int(control.get('DenoiserBaseSeedSlider', 1))
        denoiser_mode_key = f'DenoiserModeSelection{pass_suffix}'
        denoiser_mode_val = control.get(denoiser_mode_key, "Single Step (Fast)")
        ddim_steps_key = f'DenoiserDDIMStepsSlider{pass_suffix}'
        ddim_steps_val = int(control.get(ddim_steps_key, 20))
        cfg_scale_key = f'DenoiserCFGScaleDecimalSlider{pass_suffix}'
        cfg_scale_val = float(control.get(cfg_scale_key, 1.0))
        single_step_t_key = f'DenoiserSingleStepTimestepSlider{pass_suffix}'
        single_step_t_val = int(control.get(single_step_t_key, 1))

        if not kv_map:
            if use_exclusive_path:
                print(f"Denoiser {pass_suffix}: No source face for K/V, but 'Exclusive Reference Path' is ON. Skipping.")
                return image_tensor_cxhxw_uint8

        denoised_image = self.models_processor.apply_denoiser_unet(
            image_tensor_cxhxw_uint8,
            reference_kv_map=kv_map, 
            use_reference_exclusive_path=use_exclusive_path,
            denoiser_mode=denoiser_mode_val, 
            base_seed=denoiser_seed_from_slider_val,
            denoiser_single_step_t=single_step_t_val,
            denoiser_ddim_steps=ddim_steps_val,
            denoiser_cfg_scale=cfg_scale_val
        )
        return denoised_image

    def _find_best_target_match(self, detected_embedding_np, control_global):
        best_target_button = None
        best_params_pd = None
        highest_sim = -1.0

        for target_id, target_button_widget in list(self.main_window.target_faces.items()):
            face_specific_params_dict = self.parameters.get(target_id, {})
            default_params_dict = dict(self.main_window.default_parameters) if isinstance(self.main_window.default_parameters, ParametersDict) else self.main_window.default_parameters
            
            current_params_pd = ParametersDict(dict(face_specific_params_dict), default_params_dict)
            target_embedding_np = target_button_widget.get_embedding(control_global['RecognitionModelSelection'])
            if target_embedding_np is None: continue
            sim = self.models_processor.findCosineDistance(detected_embedding_np, target_embedding_np)
            
            if sim >= current_params_pd['SimilarityThresholdSlider'] and sim > highest_sim:
                highest_sim = sim
                best_target_button = target_button_widget
                best_params_pd = current_params_pd
        return best_target_button, best_params_pd, highest_sim

    def _process_single_vr_perspective_crop_multi(self,
                                 perspective_crop_torch_rgb_uint8: torch.Tensor,
                                 target_face_button: 'widget_components.TargetFaceCardButton',
                                 parameters_for_face: ParametersDict,
                                 control_global: dict,
                                 kps_5_on_crop_param: np.ndarray,
                                 kps_all_on_crop_param: np.ndarray | None,
                                 swap_button_is_checked_global: bool,
                                 edit_button_is_checked_global: bool,
                                 eye_side_for_debug: str = "",
                                 kv_map_for_swap: Dict | None = None
                                 ) -> torch.Tensor:
        processed_crop_torch_rgb_uint8 = perspective_crop_torch_rgb_uint8.clone()
        if kps_5_on_crop_param is None or kps_5_on_crop_param.size == 0:
            return processed_crop_torch_rgb_uint8

        if not (swap_button_is_checked_global or edit_button_is_checked_global):
            return processed_crop_torch_rgb_uint8

        arcface_model_for_swap = self.models_processor.get_arcface_model(parameters_for_face['SwapModelSelection'])
        s_e_for_swap_np = None
        if swap_button_is_checked_global:
            s_e_for_swap_np = target_face_button.assigned_input_embedding.get(arcface_model_for_swap)
            if s_e_for_swap_np is None or not isinstance(s_e_for_swap_np, np.ndarray) or s_e_for_swap_np.size == 0 or np.isnan(s_e_for_swap_np).any() or np.isinf(s_e_for_swap_np).any():
                s_e_for_swap_np = None

        t_e_for_swap_np = target_face_button.get_embedding(arcface_model_for_swap)
        dfm_model_instance_local = None
        if parameters_for_face['SwapModelSelection'] == 'DeepFaceLive (DFM)':
            dfm_model_name = parameters_for_face['DFMModelSelection']
            if dfm_model_name:
                dfm_model_instance_local = self.models_processor.load_dfm_model(dfm_model_name)
        
        s_e_for_swap_core = s_e_for_swap_np if swap_button_is_checked_global else None

        if (swap_button_is_checked_global and (s_e_for_swap_core is not None or (parameters_for_face['SwapModelSelection'] == 'DeepFaceLive (DFM)' and dfm_model_instance_local is not None))) or edit_button_is_checked_global:
            try:
                swapped_face_512_torch_rgb_uint8, comprehensive_mask_1x512x512_from_swap_core, _ = self.swap_core(
                    perspective_crop_torch_rgb_uint8, kps_5_on_crop_param,
                    kps=kps_all_on_crop_param, s_e=s_e_for_swap_core, t_e=t_e_for_swap_np,
                    parameters=parameters_for_face, control=control_global, dfm_model_name=parameters_for_face['DFMModelSelection'],
                    is_perspective_crop=True,
                    kv_map=kv_map_for_swap
                )
            except Exception as e_swap_core:
                print(f"Error in swap_core for VR crop {eye_side_for_debug}: {e_swap_core}")
                traceback.print_exc()
                swapped_face_512_torch_rgb_uint8 = t512(perspective_crop_torch_rgb_uint8)
                comprehensive_mask_1x512x512_from_swap_core = torch.zeros((1, 512, 512), dtype=torch.float32, device=perspective_crop_torch_rgb_uint8.device)

            tform_persp_to_512template = self.get_face_similarity_tform(parameters_for_face['SwapModelSelection'], kps_5_on_crop_param)
            
            if comprehensive_mask_1x512x512_from_swap_core is None or comprehensive_mask_1x512x512_from_swap_core.numel() == 0:
                persp_final_combined_mask_1x512x512_float_for_paste = t512(self.get_border_mask(parameters_for_face)[0]).float() if swap_button_is_checked_global else torch.zeros((1, 512, 512), dtype=torch.float32, device=perspective_crop_torch_rgb_uint8.device)
            else: 
                persp_final_combined_mask_1x512x512_float_for_paste = comprehensive_mask_1x512x512_from_swap_core.float()

            persp_final_combined_mask_3x512x512_float_for_paste = persp_final_combined_mask_1x512x512_float_for_paste.repeat(3,1,1)
            masked_swapped_face_to_paste_float = swapped_face_512_torch_rgb_uint8.float() * persp_final_combined_mask_3x512x512_float_for_paste

            crop_h, crop_w = perspective_crop_torch_rgb_uint8.shape[1], perspective_crop_torch_rgb_uint8.shape[2]
            _, source_grid_normalized_xy_persp = self.get_grid_for_pasting(
                tform_persp_to_512template, crop_h, crop_w, 512, 512, perspective_crop_torch_rgb_uint8.device
            )
            pasted_face_on_persp_float = torch.nn.functional.grid_sample(
                masked_swapped_face_to_paste_float.unsqueeze(0),
                source_grid_normalized_xy_persp,
                mode='bilinear', padding_mode='border', align_corners=False
            ).squeeze(0)
            transformed_mask_on_persp_float = torch.nn.functional.grid_sample(
                persp_final_combined_mask_3x512x512_float_for_paste.unsqueeze(0),
                source_grid_normalized_xy_persp,
                mode='bilinear', padding_mode='zeros', align_corners=False
            ).squeeze(0)
            blended_persp_crop_float = pasted_face_on_persp_float + perspective_crop_torch_rgb_uint8.float() * (1.0 - transformed_mask_on_persp_float)
            processed_crop_torch_rgb_uint8 = torch.clamp(blended_persp_crop_float, 0, 255).byte()
            
            if edit_button_is_checked_global:
                _, _, kps_all_for_editor_list = self.models_processor.run_detect(
                    processed_crop_torch_rgb_uint8, control_global['DetectorModelSelection'], max_num=1,
                    score=control_global['DetectorScoreSlider']/100.0,
                    input_size=(processed_crop_torch_rgb_uint8.shape[1], processed_crop_torch_rgb_uint8.shape[2]),
                    use_landmark_detection=True, landmark_detect_mode="203", 
                    landmark_score=control_global["LandmarkDetectScoreSlider"]/100.0,
                    from_points=True, rotation_angles=[0]
                )
                kps_all_for_editor_on_crop = kps_all_for_editor_list[0] if kps_all_for_editor_list.shape[0] > 0 else None
                if kps_all_for_editor_on_crop is not None and kps_all_for_editor_on_crop.size > 0:
                    processed_crop_torch_rgb_uint8 = self.swap_edit_face_core(processed_crop_torch_rgb_uint8, kps_all_for_editor_on_crop, parameters_for_face, control_global)
                    if any(parameters_for_face.get(f, False) for f in ('FaceMakeupEnableToggle', 'HairMakeupEnableToggle', 'EyeBrowsMakeupEnableToggle', 'LipsMakeupEnableToggle')): #, 'EyesMakeupEnableToggle')):
                        processed_crop_torch_rgb_uint8 = self.swap_edit_face_core_makeup(
                            processed_crop_torch_rgb_uint8,
                            kps_all_for_editor_on_crop,
                            parameters_for_face,
                            control_global
                        )

        return processed_crop_torch_rgb_uint8

    def process_frame(self, control: dict):
            self.set_scaling_transforms(control)
            img_numpy_rgb_uint8 = self.frame
            swap_button_is_checked_global = self.main_window.swapfacesButton.isChecked()
            edit_button_is_checked_global = self.main_window.editFacesButton.isChecked()
            processed_tensor_rgb_uint8 = torch.from_numpy(img_numpy_rgb_uint8).to(self.models_processor.device).permute(2, 0, 1)
            det_faces_data_for_display = []

            if control.get('VR180ModeEnableToggle', False):
                # --- VR180 Path ---
                original_equirect_tensor_for_vr = processed_tensor_rgb_uint8.clone()
                equirect_converter = EquirectangularConverter(img_numpy_rgb_uint8, device=self.models_processor.device)
                bboxes_eq_np, kpss_5_eq_np, kpss_all_eq_np = self.models_processor.run_detect(
                    original_equirect_tensor_for_vr, control['DetectorModelSelection'], max_num=control['MaxFacesToDetectSlider'],
                    score=control['DetectorScoreSlider']/100.0, input_size=(512, 512),
                    use_landmark_detection=control['LandmarkDetectToggle'], landmark_detect_mode=control['LandmarkDetectModelSelection'],
                    landmark_score=control["LandmarkDetectScoreSlider"]/100.0, from_points=control["DetectFromPointsToggle"],
                    rotation_angles=[0] if not control["AutoRotationToggle"] else [0, 90, 180, 270]
                )
                processed_perspective_crops_details = {}
                analyzed_faces_for_vr = []
                for i, bbox_eq_single in enumerate(bboxes_eq_np):
                    theta, phi = equirect_converter.calculate_theta_phi_from_bbox(bbox_eq_single)
                    original_eye_side = "L" if (bbox_eq_single[0] + bbox_eq_single[2]) / 2 < equirect_converter.width / 2 else "R"
                    angular_width_deg = (bbox_eq_single[2] - bbox_eq_single[0]) / equirect_converter.width * 360.0
                    angular_height_deg = (bbox_eq_single[3] - bbox_eq_single[1]) / equirect_converter.height * 180.0
                    dynamic_fov_for_crop = np.clip(max(angular_width_deg, angular_height_deg) * self.VR_DYNAMIC_FOV_PADDING_FACTOR, 15.0, 100.0)
                    face_crop_tensor = equirect_converter.get_perspective_crop(dynamic_fov_for_crop, theta, phi, self.VR_PERSPECTIVE_RENDER_SIZE, self.VR_PERSPECTIVE_RENDER_SIZE)
                    if face_crop_tensor is None or face_crop_tensor.numel() == 0: continue
                    
                    _, kps_on_crop_list, kps_all_on_crop_list = self.models_processor.run_detect(
                        face_crop_tensor, control['DetectorModelSelection'], max_num=1, score=control['DetectorScoreSlider']/100.0,
                        input_size=(self.VR_PERSPECTIVE_RENDER_SIZE, self.VR_PERSPECTIVE_RENDER_SIZE), use_landmark_detection=True,
                        landmark_detect_mode="203", landmark_score=control["LandmarkDetectScoreSlider"]/100.0,
                        from_points=False, rotation_angles=[0]
                    )

                    if not (isinstance(kps_on_crop_list, np.ndarray) and kps_on_crop_list.any()) and not kps_on_crop_list:
                        del face_crop_tensor; continue

                    kps_on_crop = kps_on_crop_list[0]
                    kps_all_on_crop = kps_all_on_crop_list[0] if isinstance(kps_all_on_crop_list, np.ndarray) and kps_all_on_crop_list.size > 0 else None
                    face_emb_crop, _ = self.models_processor.run_recognize_direct(face_crop_tensor, kps_on_crop, control['SimilarityTypeSelection'], control['RecognitionModelSelection'])
                    best_target_button_vr, best_params_for_target_vr, _ = self._find_best_target_match(face_emb_crop, control)
                    
                    if best_target_button_vr:
                        denoiser_on = control.get('DenoiserUNetEnableBeforeRestorersToggle', False) or \
                                      control.get('DenoiserAfterFirstRestorerToggle', False) or \
                                      control.get('DenoiserAfterRestorersToggle', False)
                        if denoiser_on and best_target_button_vr.assigned_kv_map is None and best_target_button_vr.assigned_input_faces:
                            best_target_button_vr.calculate_assigned_input_embedding()

                        analyzed_faces_for_vr.append({
                            'theta': theta, 'phi': phi, 'original_eye_side': original_eye_side, 
                            'face_crop_tensor': face_crop_tensor, 'kps_on_crop': kps_on_crop, 
                            'kps_all_on_crop': kps_all_on_crop, 'target_button': best_target_button_vr, 
                            'params': best_params_for_target_vr, 'fov_used_for_crop': dynamic_fov_for_crop
                        })
                    else: del face_crop_tensor

                for item_data in analyzed_faces_for_vr:
                    processed_crop_for_stitching = self._process_single_vr_perspective_crop_multi(
                        item_data['face_crop_tensor'], item_data['target_button'], item_data['params'], control,
                        kps_5_on_crop_param=item_data['kps_on_crop'], kps_all_on_crop_param=item_data['kps_all_on_crop'],
                        swap_button_is_checked_global=swap_button_is_checked_global, edit_button_is_checked_global=edit_button_is_checked_global,
                        eye_side_for_debug=item_data['original_eye_side'],
                        kv_map_for_swap=item_data['target_button'].assigned_kv_map
                    ) if swap_button_is_checked_global or edit_button_is_checked_global else item_data['face_crop_tensor']
                    
                    processed_perspective_crops_details[f"{item_data['original_eye_side']}_{item_data['theta']}_{item_data['phi']}"] = {
                        'tensor_rgb_uint8': processed_crop_for_stitching, 'theta': item_data['theta'], 'phi': item_data['phi'], 
                        'fov_used_for_crop': item_data['fov_used_for_crop']
                    }
                    del item_data['face_crop_tensor']
                
                final_equirect_torch_cxhxw_rgb_uint8 = original_equirect_tensor_for_vr.clone()
                p2e_converter = PerspectiveConverter(img_numpy_rgb_uint8, device=self.models_processor.device)
                for eye_side, data in processed_perspective_crops_details.items():
                    p2e_converter.stitch_single_perspective(
                        target_equirect_torch_cxhxw_rgb_uint8=final_equirect_torch_cxhxw_rgb_uint8,
                        processed_crop_torch_cxhxw_rgb_uint8=data['tensor_rgb_uint8'],
                        theta=data['theta'], phi=data['phi'], fov=data['fov_used_for_crop'],
                        is_left_eye=("L" in eye_side.split('_')[0])
                    )
                processed_tensor_rgb_uint8 = final_equirect_torch_cxhxw_rgb_uint8
                del equirect_converter, p2e_converter, original_equirect_tensor_for_vr, processed_perspective_crops_details, analyzed_faces_for_vr
                torch.cuda.empty_cache()
            else:
                # --- Standard Path ---
                img = processed_tensor_rgb_uint8
                img_x, img_y = img.size(2), img.size(1)
                scale_applied = False
                if img_x < 512 or img_y < 512:
                    if img_x <= img_y: new_h, new_w = int(512 * img_y / img_x), 512
                    else: new_h, new_w = 512, int(512 * img_x / img_y)
                    img = v2.Resize((new_h, new_w), antialias=False)(img)
                    scale_applied = True
                
                if control['ManualRotationEnableToggle']:
                    img = v2.functional.rotate(img, angle=control['ManualRotationAngleSlider'], interpolation=v2.InterpolationMode.BILINEAR, expand=True)

                use_landmark, landmark_mode, from_points = control['LandmarkDetectToggle'], control['LandmarkDetectModelSelection'], control["DetectFromPointsToggle"]
                if edit_button_is_checked_global:
                    use_landmark, landmark_mode, from_points = True, "203", True

                bboxes, kpss_5, kpss = self.models_processor.run_detect(
                    img, control['DetectorModelSelection'], max_num=control['MaxFacesToDetectSlider'], score=control['DetectorScoreSlider']/100.0, 
                    input_size=(512, 512), use_landmark_detection=use_landmark, landmark_detect_mode=landmark_mode, 
                    landmark_score=control["LandmarkDetectScoreSlider"]/100.0, from_points=from_points, 
                    rotation_angles=[0] if not control["AutoRotationToggle"] else [0, 90, 180, 270]
                )

                if kpss_5.shape[0] > 0:
                    for i in range(kpss_5.shape[0]):
                        face_emb, _ = self.models_processor.run_recognize_direct(img, kpss_5[i], control['SimilarityTypeSelection'], control['RecognitionModelSelection'])
                        det_faces_data_for_display.append({'kps_5': kpss_5[i], 'kps_all': kpss[i], 'embedding': face_emb, 'bbox': bboxes[i], 'original_face': None, 'swap_mask': None})

                if det_faces_data_for_display:
                    if control['SwapOnlyBestMatchEnableToggle']:
                        for _, target_face in self.main_window.target_faces.items():
                            params = ParametersDict(self.parameters[target_face.face_id], self.main_window.default_parameters)

                            best_fface, best_score = None, -1.0
                            for fface in det_faces_data_for_display:
                                # benutze die EXISTIERENDE Matching-Logik
                                tgt, tgt_params, score = self._find_best_target_match(fface['embedding'], control)
                                # nur wenn dieser Face-Detection auch wirklich zu genau diesem target_face gehört:
                                if tgt and tgt.face_id == target_face.face_id:
                                    if score >= tgt_params['SimilarityThresholdSlider'] and score > best_score:
                                        best_score = score
                                        best_fface = fface

                            if best_fface is not None:
                                denoiser_on = control.get('DenoiserUNetEnableBeforeRestorersToggle', False) or \
                                              control.get('DenoiserAfterFirstRestorerToggle', False) or \
                                              control.get('DenoiserAfterRestorersToggle', False)
                                if denoiser_on and target_face.assigned_kv_map is None and target_face.assigned_input_faces:
                                    target_face.calculate_assigned_input_embedding()

                                s_e = None
                                arcface_model = self.models_processor.get_arcface_model(params['SwapModelSelection'])
                                if swap_button_is_checked_global and params['SwapModelSelection'] != 'DeepFaceLive (DFM)':
                                    s_e = target_face.assigned_input_embedding.get(arcface_model)
                                    if s_e is not None and np.isnan(s_e).any():
                                        s_e = None

                                kv_map_for_swap = target_face.assigned_kv_map
                                img, best_fface['original_face'], best_fface['swap_mask'] = self.swap_core(
                                    img, best_fface['kps_5'], best_fface['kps_all'],
                                    s_e=s_e, t_e=target_face.get_embedding(arcface_model),
                                    parameters=params, control=control,
                                    dfm_model_name=params['DFMModelSelection'],
                                    kv_map=kv_map_for_swap
                                )
                                if edit_button_is_checked_global and any(params[f] for f in ('FaceMakeupEnableToggle','HairMakeupEnableToggle','EyeBrowsMakeupEnableToggle','LipsMakeupEnableToggle')):
                                    img = self.swap_edit_face_core_makeup(img, best_fface['kps_all'], params, control)
                    else:
                        for fface in det_faces_data_for_display:
                            best_target, params, _ = self._find_best_target_match(fface['embedding'], control)
                            if best_target:
                                denoiser_on = control.get('DenoiserUNetEnableBeforeRestorersToggle', False) or \
                                              control.get('DenoiserAfterFirstRestorerToggle', False) or \
                                              control.get('DenoiserAfterRestorersToggle', False)
                                if denoiser_on and best_target.assigned_kv_map is None and best_target.assigned_input_faces:
                                    best_target.calculate_assigned_input_embedding()

                                fface['kps_5'] = self.keypoints_adjustments(fface['kps_5'], params)
                                arcface_model = self.models_processor.get_arcface_model(params['SwapModelSelection'])
                                s_e = None
                                if swap_button_is_checked_global and params['SwapModelSelection'] != 'DeepFaceLive (DFM)':
                                    s_e = best_target.assigned_input_embedding.get(arcface_model)
                                    if s_e is not None and np.isnan(s_e).any(): s_e = None
                                
                                kv_map_for_swap = best_target.assigned_kv_map
                                img, fface['original_face'], fface['swap_mask'] = self.swap_core(img, fface['kps_5'], fface['kps_all'], s_e=s_e, t_e=best_target.get_embedding(arcface_model), parameters=params, control=control, dfm_model_name=params['DFMModelSelection'], kv_map=kv_map_for_swap)
                                if edit_button_is_checked_global and any(params[f] for f in ('FaceMakeupEnableToggle', 'HairMakeupEnableToggle', 'EyeBrowsMakeupEnableToggle', 'LipsMakeupEnableToggle')):
                                    img = self.swap_edit_face_core_makeup(img, fface['kps_all'], params, control)

                if control['ManualRotationEnableToggle']:
                    img = v2.functional.rotate(img, angle=-control['ManualRotationAngleSlider'], interpolation=v2.InterpolationMode.BILINEAR, expand=True)
                if scale_applied:
                    img = v2.Resize((img_y, img_x), interpolation=interpolation_scaleback, antialias=False)(img)
                processed_tensor_rgb_uint8 = img
            
            # --- Common Post-Processing ---
            if control['ShowAllDetectedFacesBBoxToggle'] and det_faces_data_for_display and not control.get('VR180ModeEnableToggle', False):
                processed_tensor_rgb_uint8 = self.draw_bounding_boxes_on_detected_faces(processed_tensor_rgb_uint8, det_faces_data_for_display, control)
            
            if control["ShowLandmarksEnableToggle"] and det_faces_data_for_display and not control.get('VR180ModeEnableToggle', False):
                temp_permuted = processed_tensor_rgb_uint8.permute(1,2,0)
                temp_permuted = self.paint_face_landmarks(temp_permuted, det_faces_data_for_display, control)
                processed_tensor_rgb_uint8 = temp_permuted.permute(2,0,1)
            
            compare_mode_active = self.is_view_face_mask or self.is_view_face_compare
            if compare_mode_active and det_faces_data_for_display and not control.get('VR180ModeEnableToggle', False):
                processed_tensor_rgb_uint8 = self.get_compare_faces_image(processed_tensor_rgb_uint8, det_faces_data_for_display, control)

            if control['FrameEnhancerEnableToggle'] and not compare_mode_active:
                processed_tensor_rgb_uint8 = self.enhance_core(processed_tensor_rgb_uint8, control=control)

            final_img_np_rgb_uint8 = processed_tensor_rgb_uint8.permute(1,2,0).cpu().numpy().astype(np.uint8)
            if not final_img_np_rgb_uint8.flags['C_CONTIGUOUS']:
                final_img_np_rgb_uint8 = np.ascontiguousarray(final_img_np_rgb_uint8)
            
            return final_img_np_rgb_uint8[..., ::-1]
    
    def keypoints_adjustments(self, kps_5: np.ndarray, parameters: dict) -> np.ndarray:
        kps_5_adj = kps_5.copy()
        # Change the ref points
        if parameters['FaceAdjEnableToggle']:
            kps_5_adj[:,0] += parameters['KpsXSlider']
            kps_5_adj[:,1] += parameters['KpsYSlider']
            kps_5_adj[:,0] -= 255 
            kps_5_adj[:,0] *= (1 + parameters['KpsScaleSlider'] / 100.0)
            kps_5_adj[:,0] += 255
            kps_5_adj[:,1] -= 255
            kps_5_adj[:,1] *= (1 + parameters['KpsScaleSlider'] / 100.0)
            kps_5_adj[:,1] += 255

        # Face Landmarks
        if parameters['LandmarksPositionAdjEnableToggle']:
            kps_5_adj[0][0] += parameters['EyeLeftXAmountSlider']
            kps_5_adj[0][1] += parameters['EyeLeftYAmountSlider']
            kps_5_adj[1][0] += parameters['EyeRightXAmountSlider']
            kps_5_adj[1][1] += parameters['EyeRightYAmountSlider']
            kps_5_adj[2][0] += parameters['NoseXAmountSlider']
            kps_5_adj[2][1] += parameters['NoseYAmountSlider']
            kps_5_adj[3][0] += parameters['MouthLeftXAmountSlider']
            kps_5_adj[3][1] += parameters['MouthLeftYAmountSlider']
            kps_5_adj[4][0] += parameters['MouthRightXAmountSlider']
            kps_5_adj[4][1] += parameters['MouthRightYAmountSlider']
        return kps_5_adj
    
    def paint_face_landmarks(self, img: torch.Tensor, det_faces_data: list, control: dict) -> torch.Tensor:
        img_out_hwc = img.clone()
        p = 2 
        
        for fface_data in det_faces_data:
            _, matched_params, _ = self._find_best_target_match(fface_data['embedding'], control)
            if matched_params:
                keypoints = fface_data.get('kps_5') if matched_params['LandmarksPositionAdjEnableToggle'] else fface_data.get('kps_all')
                kcolor = (255, 0, 0) if matched_params['LandmarksPositionAdjEnableToggle'] else (0, 255, 255)
                if keypoints is not None:
                    for kpoint in keypoints:
                        kx, ky = int(kpoint[0]), int(kpoint[1])
                        for i_offset in range(-p // 2, p // 2 + 1):
                            for j_offset in range(-p // 2, p // 2 + 1):
                                final_y, final_x = ky + i_offset, kx + j_offset
                                if 0 <= final_y < img_out_hwc.shape[0] and 0 <= final_x < img_out_hwc.shape[1]:
                                    img_out_hwc[final_y, final_x] = torch.tensor(kcolor, device=img.device, dtype=img.dtype)
        return img_out_hwc
    
    def draw_bounding_boxes_on_detected_faces(self, img: torch.Tensor, det_faces_data: list, control: dict):
        for i, fface in enumerate(det_faces_data):
            color_rgb = [0, 255, 0]
            bbox = fface['bbox']
            x_min, y_min, x_max, y_max = map(int, bbox)
            # Ensure bounding box is within the image dimensions
            _, h, w = img.shape
            x_min, y_min = max(0, x_min), max(0, y_min)
            x_max, y_max = min(w - 1, x_max), min(h - 1, y_max)
            # Dynamically compute thickness based on the image resolution
            max_dimension = max(img.shape[1], img.shape[2])  # Height and width of the image
            thickness = max(4, max_dimension // 400)  # Thickness is 1/200th of the largest dimension, minimum 1
            # Prepare the color tensor with the correct dimensions
            color_tensor_c11 = torch.tensor(color_rgb, dtype=img.dtype, device=img.device).view(-1, 1, 1)
            img[:, y_min:y_min + thickness, x_min:x_max + 1] = color_tensor_c11.expand(-1, thickness, x_max - x_min + 1)
            img[:, y_max - thickness + 1:y_max + 1, x_min:x_max + 1] = color_tensor_c11.expand(-1, thickness, x_max - x_min + 1)
            img[:, y_min:y_max + 1, x_min:x_min + thickness] = color_tensor_c11.expand(-1, y_max - y_min + 1, thickness)
            img[:, y_min:y_max + 1, x_max - thickness + 1:x_max + 1] = color_tensor_c11.expand(-1, y_max - y_min + 1, thickness)  
        return img

    def get_compare_faces_image(self, img: torch.Tensor, det_faces_data: dict, control: dict) -> torch.Tensor:
        imgs_to_vstack = []  # Renamed for vertical stacking
        for _, fface in enumerate(det_faces_data):
            best_target_for_compare, parameters_for_face, _ = self._find_best_target_match(fface['embedding'], control)
            if best_target_for_compare and parameters_for_face:
                modified_face = self.get_cropped_face_using_kps(img, fface['kps_5'], parameters_for_face)
                if control['FrameEnhancerEnableToggle']:
                    enhanced_version = self.enhance_core(modified_face.clone(), control=control)
                    if enhanced_version.shape[1:] != modified_face.shape[1:]:
                        enhanced_version = v2.Resize(modified_face.shape[1:], antialias=True)(enhanced_version)
                    modified_face = enhanced_version
                imgs_to_cat_horizontally = []
                original_face_from_swap_core = fface.get('original_face')
                if original_face_from_swap_core is not None: imgs_to_cat_horizontally.append(original_face_from_swap_core.permute(2,0,1))
                imgs_to_cat_horizontally.append(modified_face)
                swap_mask_from_swap_core = fface.get('swap_mask')
                if swap_mask_from_swap_core is not None:
                    mask_chw = swap_mask_from_swap_core.permute(2,0,1)
                    if mask_chw.shape[0] == 1: mask_chw = mask_chw.repeat(3,1,1)
                    imgs_to_cat_horizontally.append(mask_chw)
                if imgs_to_cat_horizontally:
                    min_h = min(t.shape[1] for t in imgs_to_cat_horizontally)
                    resized_imgs_to_cat = []
                    for t_img in imgs_to_cat_horizontally:
                        if t_img.shape[1] != min_h:
                            aspect_ratio = t_img.shape[2] / t_img.shape[1]
                            new_w = int(min_h * aspect_ratio) if aspect_ratio > 0 else t_img.shape[2]
                            resized_imgs_to_cat.append(v2.Resize((min_h, new_w), antialias=True)(t_img))
                        else:
                            resized_imgs_to_cat.append(t_img)
                    imgs_to_vstack.append(torch.cat(resized_imgs_to_cat, dim=2))
    
        if imgs_to_vstack:
            max_width_for_vstack = max(img_strip.size(2) for img_strip in imgs_to_vstack)
            padded_strips_for_vstack = [torch.nn.functional.pad(img_strip, (0, max_width_for_vstack - img_strip.size(2), 0, 0)) for img_strip in imgs_to_vstack]
            return torch.cat(padded_strips_for_vstack, dim=1)
        return img
        
    def get_cropped_face_using_kps(self, img: torch.Tensor, kps_5: np.ndarray, parameters: dict) -> torch.Tensor:
        tform = self.get_face_similarity_tform(parameters['SwapModelSelection'], kps_5)
        face_512_aligned = v2.functional.affine(img, angle=tform.rotation * 57.2958, translate=(tform.translation[0], tform.translation[1]), scale=tform.scale, shear=(0.0, 0.0), center=(0,0), interpolation=interpolation_get_cropped_face_kps)
        return v2.functional.crop(face_512_aligned, 0, 0, 512, 512)

    def get_face_similarity_tform(self, swapper_model: str, kps_5: np.ndarray) -> trans.SimilarityTransform:
        tform = trans.SimilarityTransform()
        if swapper_model != 'GhostFace-v1' and swapper_model != 'GhostFace-v2' and swapper_model != 'GhostFace-v3' and swapper_model != 'CSCS':
            dst = faceutil.get_arcface_template(image_size=512, mode='arcface128')
            dst = np.squeeze(dst)
            tform.estimate(kps_5, dst)
        elif swapper_model == "CSCS":
            dst = faceutil.get_arcface_template(image_size=512, mode='arcfacemap')
            tform.estimate(kps_5, self.models_processor.FFHQ_kps)
        else:
            dst = faceutil.get_arcface_template(image_size=512, mode='arcfacemap')
            M, _ = faceutil.estimate_norm_arcface_template(kps_5, src=dst)
            tform.params[0:2] = M
        return tform
      
    def get_transformed_and_scaled_faces(self, tform, img):
        original_face_512 = v2.functional.affine(img, tform.rotation*57.2958, (tform.translation[0], tform.translation[1]) , tform.scale, 0, center = (0,0), interpolation=interpolation_original_face_512)
        original_face_512 = v2.functional.crop(original_face_512, 0,0, 512, 512)
        original_face_384 = t384(original_face_512)
        original_face_256 = t256(original_face_512)
        original_face_128 = t128(original_face_256)
        return original_face_512, original_face_384, original_face_256, original_face_128
    
    def get_affined_face_dim_and_swapping_latents(self, original_faces: tuple, swapper_model, dfm_model_name, s_e, t_e, parameters, cmddebug, tform):
        original_face_512, original_face_384, original_face_256, original_face_128 = original_faces

        dfm_model_instance = None
        
        input_face_affined = None
        dim = 1
        latent = None

        if swapper_model == 'Inswapper128':
            self.models_processor.load_inswapper_iss_emap('Inswapper128')
            latent = torch.from_numpy(self.models_processor.calc_inswapper_latent(s_e)).float().to(self.models_processor.device)
            if parameters['FaceLikenessEnableToggle']:
                factor = parameters['FaceLikenessFactorDecimalSlider']
                dst_latent = torch.from_numpy(self.models_processor.calc_inswapper_latent(t_e)).float().to(self.models_processor.device)
                latent = latent - (factor * dst_latent)

            dim = 1
            if parameters['SwapperResAutoSelectEnableToggle']:
                if tform.scale <= 1.00:
                    dim = 4
                    input_face_affined = original_face_512
                elif tform.scale <= 1.75:
                    dim = 3
                    input_face_affined = original_face_384
                elif tform.scale <= 2:
                    dim = 2
                    input_face_affined = original_face_256
                else:
                    dim = 1
                    input_face_affined = original_face_128
            else:        
                if parameters['SwapperResSelection'] == '128':
                    dim = 1
                    input_face_affined = original_face_128
                elif parameters['SwapperResSelection'] == '256':
                    dim = 2
                    input_face_affined = original_face_256
                elif parameters['SwapperResSelection'] == '384':
                    dim = 3
                    input_face_affined = original_face_384
                elif parameters['SwapperResSelection'] == '512':
                    dim = 4
                    input_face_affined = original_face_512

        elif swapper_model in ('InStyleSwapper256 Version A', 'InStyleSwapper256 Version B', 'InStyleSwapper256 Version C'):
            version = swapper_model[-1]
            self.models_processor.load_inswapper_iss_emap(swapper_model)
            latent = torch.from_numpy(self.models_processor.calc_swapper_latent_iss(s_e, version)).float().to(self.models_processor.device)
            if parameters['FaceLikenessEnableToggle']:
                factor = parameters['FaceLikenessFactorDecimalSlider']
                dst_latent = torch.from_numpy(self.models_processor.calc_swapper_latent_iss(t_e, version)).float().to(self.models_processor.device)
                latent = latent - (factor * dst_latent)

            if (parameters['SwapModelSelection'] == 'InStyleSwapper256 Version A' and parameters['InStyleResAEnableToggle']) or (parameters['SwapModelSelection'] == 'InStyleSwapper256 Version B' and parameters['InStyleResBEnableToggle']) or (parameters['SwapModelSelection'] == 'InStyleSwapper256 Version C' and parameters['InStyleResCEnableToggle']):
                dim = 4
                input_face_affined = original_face_512
            else:
                dim = 2
                input_face_affined = original_face_256

        elif swapper_model == 'SimSwap512':
            latent = torch.from_numpy(self.models_processor.calc_swapper_latent_simswap512(s_e)).float().to(self.models_processor.device)
            if parameters['FaceLikenessEnableToggle']:
                factor = parameters['FaceLikenessFactorDecimalSlider']
                dst_latent = torch.from_numpy(self.models_processor.calc_swapper_latent_simswap512(t_e)).float().to(self.models_processor.device)
                latent = latent - (factor * dst_latent)

            dim = 4
            input_face_affined = original_face_512

        elif swapper_model == 'GhostFace-v1' or swapper_model == 'GhostFace-v2' or swapper_model == 'GhostFace-v3':
            latent = torch.from_numpy(self.models_processor.calc_swapper_latent_ghost(s_e)).float().to(self.models_processor.device)
            if parameters['FaceLikenessEnableToggle']:
                factor = parameters['FaceLikenessFactorDecimalSlider']
                dst_latent = torch.from_numpy(self.models_processor.calc_swapper_latent_ghost(t_e)).float().to(self.models_processor.device)
                latent = latent - (factor * dst_latent)

            dim = 2
            input_face_affined = original_face_256

        elif swapper_model == 'CanonSwap':
            latent = torch.from_numpy(self.models_processor.calc_swapper_latent_canonswap(s_e)).float().to(self.models_processor.device)
            dim = 2
            input_face_affined = original_face_256

        elif swapper_model == 'CSCS':
            latent = torch.from_numpy(self.models_processor.calc_swapper_latent_cscs(s_e)).float().to(self.models_processor.device)
            if parameters['FaceLikenessEnableToggle']:
                factor = parameters['FaceLikenessFactorDecimalSlider']
                dst_latent = torch.from_numpy(self.models_processor.calc_swapper_latent_cscs(t_e)).float().to(self.models_processor.device)
                latent = latent - (factor * dst_latent)

            dim = 2
            input_face_affined = original_face_256

        if swapper_model == 'DeepFaceLive (DFM)' and dfm_model_name:
            dfm_model_instance = self.models_processor.load_dfm_model(dfm_model_name)
            latent = []
            input_face_affined = original_face_512
            dim = 4

        return input_face_affined, dfm_model_instance, dim, latent
    
    def get_swapped_and_prev_face(self, output, input_face_affined, original_face_512, latent, itex, dim, swapper_model, dfm_model, parameters, ):
        # original_face_512, original_face_384, original_face_256, original_face_128 = original_faces
        if parameters['PreSwapSharpnessDecimalSlider'] != 1.0:
            input_face_affined = input_face_affined.permute(2, 0, 1)
            input_face_affined = v2.functional.adjust_sharpness(input_face_affined, parameters['PreSwapSharpnessDecimalSlider'])
            input_face_affined = input_face_affined.permute(1, 2, 0)
        prev_face = input_face_affined.clone()
        if swapper_model == 'Inswapper128':
            with torch.no_grad():  # Disabilita il calcolo del gradiente se è solo per inferenza
                for _ in range(itex):
                    for j in range(dim):
                        for i in range(dim):
                            input_face_disc = input_face_affined[j::dim,i::dim]
                            input_face_disc = input_face_disc.permute(2, 0, 1)
                            input_face_disc = torch.unsqueeze(input_face_disc, 0).contiguous()

                            swapper_output = torch.empty((1,3,128,128), dtype=torch.float32, device=self.models_processor.device).contiguous()
                            self.models_processor.run_inswapper(input_face_disc, latent, swapper_output)

                            swapper_output = torch.squeeze(swapper_output)
                            swapper_output = swapper_output.permute(1, 2, 0)

                            output[j::dim, i::dim] = swapper_output.clone()
                    prev_face = input_face_affined.clone()
                    input_face_affined = output.clone()
                    output = torch.mul(output, 255)
                    output = torch.clamp(output, 0, 255)

        elif swapper_model in ('InStyleSwapper256 Version A', 'InStyleSwapper256 Version B', 'InStyleSwapper256 Version C'):
            version = swapper_model[-1] #Version Name
            with torch.no_grad():  # Disabilita il calcolo del gradiente se è solo per inferenza
                dim_res = dim // 2
                for _ in range(itex):
                    for j in range(dim_res):
                        for i in range(dim_res):
                            input_face_disc = input_face_affined[j::dim_res,i::dim_res]
                            input_face_disc = input_face_disc.permute(2, 0, 1)
                            input_face_disc = torch.unsqueeze(input_face_disc, 0).contiguous()

                            swapper_output = torch.empty((1,3,256,256), dtype=torch.float32, device=self.models_processor.device).contiguous()
                            self.models_processor.run_iss_swapper(input_face_disc, latent, swapper_output, version)

                            swapper_output = torch.squeeze(swapper_output)
                            swapper_output = swapper_output.permute(1, 2, 0)

                            output[j::dim_res, i::dim_res] = swapper_output.clone()
                    
                    prev_face = input_face_affined.clone()
                    input_face_affined = output.clone()
                    output = torch.mul(output, 255)
                    output = torch.clamp(output, 0, 255)

        elif swapper_model == 'SimSwap512':
            for k in range(itex):
                input_face_disc = input_face_affined.permute(2, 0, 1)
                input_face_disc = torch.unsqueeze(input_face_disc, 0).contiguous()
                swapper_output = torch.empty((1,3,512,512), dtype=torch.float32, device=self.models_processor.device).contiguous()
                self.models_processor.run_swapper_simswap512(input_face_disc, latent, swapper_output)
                swapper_output = torch.squeeze(swapper_output)
                swapper_output = swapper_output.permute(1, 2, 0)
                prev_face = input_face_affined.clone()
                input_face_affined = swapper_output.clone()

                output = swapper_output.clone()
                output = torch.mul(output, 255)
                output = torch.clamp(output, 0, 255)

        elif swapper_model == 'GhostFace-v1' or swapper_model == 'GhostFace-v2' or swapper_model == 'GhostFace-v3':
            for k in range(itex):
                input_face_disc = torch.mul(input_face_affined, 255.0).permute(2, 0, 1)
                input_face_disc = torch.div(input_face_disc.float(), 127.5)
                input_face_disc = torch.sub(input_face_disc, 1)
                #input_face_disc = input_face_disc[[2, 1, 0], :, :] # Inverte i canali da BGR a RGB (assumendo che l'input sia BGR)
                input_face_disc = torch.unsqueeze(input_face_disc, 0).contiguous()
                swapper_output = torch.empty((1,3,256,256), dtype=torch.float32, device=self.models_processor.device).contiguous()
                self.models_processor.run_swapper_ghostface(input_face_disc, latent, swapper_output, swapper_model)
                swapper_output = swapper_output[0]
                swapper_output = swapper_output.permute(1, 2, 0)
                swapper_output = torch.mul(swapper_output, 127.5)
                swapper_output = torch.add(swapper_output, 127.5)
                #swapper_output = swapper_output[:, :, [2, 1, 0]] # Inverte i canali da RGB a BGR (assumendo che l'input sia RGB)
                prev_face = input_face_affined.clone()
                input_face_affined = swapper_output.clone()
                input_face_affined = torch.div(input_face_affined, 255)

                output = swapper_output.clone()
                output = torch.clamp(output, 0, 255)

        elif swapper_model == 'CSCS':
            for k in range(itex):
                input_face_disc = input_face_affined.permute(2, 0, 1)
                input_face_disc = v2.functional.normalize(input_face_disc, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=False)
                input_face_disc = torch.unsqueeze(input_face_disc, 0).contiguous()
                swapper_output = torch.empty((1,3,256,256), dtype=torch.float32, device=self.models_processor.device).contiguous()
                self.models_processor.run_swapper_cscs(input_face_disc, latent, swapper_output)
                swapper_output = torch.squeeze(swapper_output)
                swapper_output = torch.add(torch.mul(swapper_output, 0.5), 0.5)
                swapper_output = swapper_output.permute(1, 2, 0)
                prev_face = input_face_affined.clone()
                input_face_affined = swapper_output.clone()

                output = swapper_output.clone()
                output = torch.mul(output, 255)
                output = torch.clamp(output, 0, 255)

        elif swapper_model == 'CanonSwap':
            # input_face_affined is a HWC tensor in [0,1] range.
            # run_swapper_canonswap expects a BCHW tensor in [0,1] range.
            input_face_disc = input_face_affined.permute(2, 0, 1)  # HWC to CHW
            input_face_disc = torch.unsqueeze(input_face_disc, 0).contiguous() # CHW to BCHW

            # run_swapper_canonswap returns a CxHxW tensor with values in [0, 255]
            swapped_face_tensor = self.models_processor.run_swapper_canonswap(input_face_disc, latent, None)
            prev_face = input_face_affined.clone() # This is [0,1] HWC
            swap = t512(swapped_face_tensor)
            return swap, prev_face


        elif swapper_model == 'DeepFaceLive (DFM)' and dfm_model:
            out_celeb, _, _ = dfm_model.convert(original_face_512, parameters['DFMAmpMorphSlider']/100, rct=parameters['DFMRCTColorToggle'])
            prev_face = input_face_affined.clone()
            input_face_affined = out_celeb.clone()
            output = out_celeb.clone()

        output = output.permute(2, 0, 1)
        swap = t512(output)   
        return swap, prev_face
    
    def get_border_mask(self, parameters):
        # Create border mask
        border_mask = torch.ones((128, 128), dtype=torch.float32, device=self.models_processor.device)
        border_mask = torch.unsqueeze(border_mask,0)

        # if parameters['BorderState']:
        top = parameters['BorderTopSlider']
        left = parameters['BorderLeftSlider']
        right = 128 - parameters['BorderRightSlider']
        bottom = 128 - parameters['BorderBottomSlider']

        border_mask[:, :top, :] = 0
        border_mask[:, bottom:, :] = 0
        border_mask[:, :, :left] = 0
        border_mask[:, :, right:] = 0

        border_mask_calc = border_mask.clone()

        blur_amount = parameters['BorderBlurSlider']
        blur_kernel_size = blur_amount * 2 + 1
        if blur_kernel_size > 1:
            sigma_val = max(blur_amount * 0.15 + 0.1, 1e-6) 
            gauss = transforms.GaussianBlur(blur_kernel_size, sigma=sigma_val)
            border_mask = gauss(border_mask)
        return border_mask, border_mask_calc

    def get_grid_for_pasting(self, tform_target_to_source: trans.SimilarityTransform,
                                target_h: int, target_w: int,
                                source_h: int, source_w: int,
                                device: torch.device):
            # tform_target_to_source: maps points from target (e.g. full image) to source (e.g. 512x512 face)
            grid_y, grid_x = torch.meshgrid(
                torch.arange(target_h, device=device, dtype=torch.float32),
                torch.arange(target_w, device=device, dtype=torch.float32),
                indexing='ij'
            )
            target_grid_yx_pixels = torch.stack((grid_y, grid_x), dim=2).unsqueeze(0) # 1xTargetHxTargetWx2 (Y,X order)

            # Convert target grid pixel coordinates to homogeneous coordinates (X,Y,1)
            target_grid_xy_flat_pixels = target_grid_yx_pixels[..., [1,0]].reshape(-1, 2) # (N,2) in XY
            ones = torch.ones(target_grid_xy_flat_pixels.shape[0], 1, device=device, dtype=torch.float32)
            homogeneous_target_grid_xy_pixels = torch.cat((target_grid_xy_flat_pixels, ones), dim=1) # (N,3)

            # Transformation matrix from tform_target_to_source (2x3)
            M_target_to_source = torch.tensor(tform_target_to_source.params[0:2, :], dtype=torch.float32, device=device)

            # Transform target grid to source coordinates (pixels)
            # (N,3) @ (3,2) -> (N,2) in XY order
            source_coords_xy_flat_pixels = torch.matmul(homogeneous_target_grid_xy_pixels, M_target_to_source.T)

            # Reshape to grid format 1xTargetHxTargetWx2
            source_coords_xy_grid_pixels = source_coords_xy_flat_pixels.view(1, target_h, target_w, 2)

            # Normalize source coordinates for grid_sample (expects XY order, range [-1,1])
            source_grid_normalized_xy = torch.empty_like(source_coords_xy_grid_pixels)
            # Normalize X coordinates
            source_grid_normalized_xy[..., 0] = (source_coords_xy_grid_pixels[..., 0] / (source_w - 1.0)) * 2.0 - 1.0
            # Normalize Y coordinates
            source_grid_normalized_xy[..., 1] = (source_coords_xy_grid_pixels[..., 1] / (source_h - 1.0)) * 2.0 - 1.0
            
            # target_grid_yx is not strictly needed by grid_sample but returned for completeness if ever useful
            return target_grid_yx_pixels, source_grid_normalized_xy


    def swap_core(self, img: torch.Tensor, kps_5: np.ndarray,
                    kps: np.ndarray | bool = False, s_e: np.ndarray | None = None, t_e: np.ndarray | None = None,
                    parameters: dict | None = None, control: dict | None = None,
                    dfm_model_name: str | None = None,
                    is_perspective_crop: bool = False,
                    kv_map: Dict | None = None
                    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:

        valid_s_e = s_e if isinstance(s_e, np.ndarray) else None
        valid_t_e = t_e if isinstance(t_e, np.ndarray) else None
        parameters = parameters if parameters is not None else {}
        control = control if control is not None else {}
        swapper_model = parameters['SwapModelSelection']

        mask_calc_dill = None

        debug = control.get("CommandLineDebugEnableToggle", False)
        debug_info: dict[str, str] = {}  

        tform = self.get_face_similarity_tform(swapper_model, kps_5)
        t512_mask = v2.Resize((512, 512), interpolation=v2.InterpolationMode.BILINEAR, antialias=False)
        t128_mask = v2.Resize((128, 128), interpolation=v2.InterpolationMode.BILINEAR, antialias=False)

        # Grab 512 face from image and create 256 and 128 copys
        original_face_512, original_face_384, original_face_256, original_face_128 = self.get_transformed_and_scaled_faces(tform, img)
        original_faces = (original_face_512, original_face_384, original_face_256, original_face_128)
        swap = original_face_512
        prev_face = None

        if valid_s_e is not None or (swapper_model == 'DeepFaceLive (DFM)' and dfm_model_name):
            input_face_affined, dfm_model_instance, dim, latent = self.get_affined_face_dim_and_swapping_latents(
            original_faces, swapper_model, dfm_model_name, valid_s_e, valid_t_e, parameters, debug, tform
        )
            if debug and parameters['SwapperResAutoSelectEnableToggle']:
                debug_info["Resolution"] = 128*dim

            itex = 1
            if parameters['StrengthEnableToggle']:
                itex = ceil(parameters['StrengthAmountSlider'] / 100.)

            # Create empty output image and preprocess it for swapping
            output_size = int(128 * dim)
            output = torch.zeros((output_size, output_size, 3), dtype=torch.float32, device=self.models_processor.device)
            input_face_affined = input_face_affined.permute(1, 2, 0)
            input_face_affined = torch.div(input_face_affined, 255.0)

            swap, prev_face = self.get_swapped_and_prev_face(output, input_face_affined, original_face_512, latent, itex, dim, swapper_model, dfm_model_instance, parameters)
        
        else:
            swap = original_face_512
            if parameters['StrengthEnableToggle']:
                itex = ceil(parameters['StrengthAmountSlider'] / 100.)
                prev_face = torch.div(swap, 255.)
                prev_face = prev_face.permute(1, 2, 0)

        if parameters['StrengthEnableToggle']:
            if itex == 0:
                swap = original_face_512.clone()
            else:
                alpha = np.mod(parameters['StrengthAmountSlider'], 100)*0.01
                if alpha==0:
                    alpha=1

                # Blend the images
                prev_face = torch.mul(prev_face, 255)
                prev_face = torch.clamp(prev_face, 0, 255)
                prev_face = prev_face.permute(2, 0, 1)
                #if dim != 4: prev_face = t512(prev_face)
                prev_face = t512(prev_face)
                swap = torch.mul(swap, alpha)
                prev_face = torch.mul(prev_face, 1-alpha)
                swap = torch.add(swap, prev_face)

        # Create masks
        border_mask, border_mask_calc = self.get_border_mask(parameters)
        swap_mask = torch.ones((128, 128), dtype=torch.float32, device=self.models_processor.device)
        swap_mask = torch.unsqueeze(swap_mask,0)
        #calc_mask = torch.ones((256, 256), dtype=torch.float32, device=self.models_processor.device)
        #calc_mask = torch.unsqueeze(calc_mask,0)
        
        BgExclude = torch.ones((512, 512), dtype=torch.float32, device=self.models_processor.device)
        BgExclude = torch.unsqueeze(BgExclude,0)
        diff_mask = BgExclude.clone()
        texture_mask_view = BgExclude.clone()     
        restore_mask = BgExclude.clone()
        texture_mask = BgExclude.clone()
        mask_forcalc = BgExclude.clone()
        mask_calc_dill = BgExclude.clone()
        swap_mask_noFP = swap_mask.clone()  # unveränderte 128er Basismaske für Editor-End
        
        swap = torch.clamp(swap, 0.0, 255.0)   

        # Expression Restorer beginning
        if parameters['FaceExpressionEnableToggleBoth'] and (parameters['FaceExpressionLipsToggle'] or parameters['FaceExpressionEyesToggle']) and parameters['FaceExpressionBeforeTypeSelection'] == 'Beginning':
            swap = self.apply_face_expression_restorer(original_face_512, swap, parameters)

        # Face editor beginning
        if parameters['FaceEditorEnableToggle'] and self.main_window.editFacesButton.isChecked() and parameters['FaceEditorBeforeTypeSelection'] == 'Beginning':
            editor_mask = t512_mask(swap_mask).clone()
            swap = swap * editor_mask + original_face_512 * (1 - editor_mask)
            swap = self.swap_edit_face_core(swap, kps, parameters, control)

        # First Denoiser pass - Before Restorers
        if control.get('DenoiserUNetEnableBeforeRestorersToggle', False):
            swap = self._apply_denoiser_pass(swap, control, "Before", kv_map)

        # First Restorer
        swap_original = swap.clone()   
        
        if parameters["FaceRestorerEnableToggle"]:
            swap_restorecalc = self.models_processor.apply_facerestorer(swap, parameters['FaceRestorerDetTypeSelection'], parameters['FaceRestorerTypeSelection'], parameters["FaceRestorerBlendSlider"], parameters['FaceFidelityWeightDecimalSlider'], control['DetectorScoreSlider'])                                    
        else:
            swap_restorecalc = swap.clone()

        # Occluder
        if parameters["OccluderEnableToggle"]:
            mask = self.models_processor.apply_occlusion(original_face_256, parameters["OccluderSizeSlider"])
            mask = t128_mask(mask)
            swap_mask = torch.mul(swap_mask, mask)
            gauss = transforms.GaussianBlur(parameters['OccluderXSegBlurSlider']*2+1, (parameters['OccluderXSegBlurSlider']+1)*0.2)
            swap_mask = gauss(swap_mask)
            swap_mask_noFP *= swap_mask

        # -------------------------------
        # MASKEN: Parser / CLIPs / Restore
        # -------------------------------
        t512_mask = v2.Resize((512, 512), interpolation=v2.InterpolationMode.BILINEAR, antialias=False)
        t256_near = v2.Resize((256, 256), interpolation=v2.InterpolationMode.NEAREST,  antialias=False)
        t128_bi   = v2.Resize((128, 128), interpolation=v2.InterpolationMode.BILINEAR, antialias=False)

        need_any_parser = (
            parameters.get("FaceParserEnableToggle", False)
            or (parameters.get("DFLXSegEnableToggle", False)
                and parameters.get("DFLXSeg2EnableToggle", False)
                and parameters.get("XSegMouthEnableToggle", False)
                and parameters.get("DFLXSegSizeSlider", 0) != parameters.get("DFLXSeg2SizeSlider", 0))
            or ((parameters.get("TransferTextureEnableToggle", False)
                 or parameters.get("DifferencingEnableToggle", False))
                and parameters.get("ExcludeMaskEnableToggle", False))
        )

        FaceParser_mask_128 = None  # [1,128,128] float
        texture_exclude_512 = None  # [1,512,512] float (AUSSCHLUSS)
        mouth_512           = None  # [512,512] float

        if need_any_parser:
            out = self.models_processor.process_masks_and_masks(
                swap_restorecalc,    # aktueller Swap-Stand (uint8, 3x512x512)
                original_face_512,   # original (uint8, 3x512x512)
                parameters
            )
            FaceParser_mask_128 = out.get("FaceParser_mask", None)    # [1,128,128], 1=behalten
            texture_exclude_512 = out.get("texture_mask", None)       # [1,512,512], 1=ausschließen
            mouth_512           = out.get("mouth", None)              # [512,512]

        # FaceParser-Maske (128) auf swap_mask anwenden (wenn vorhanden)
        if FaceParser_mask_128 is not None:
            swap_mask = swap_mask * FaceParser_mask_128

        # ----- CLIPs (falls aktiv): liefert i. d. R. 512er-Maske -> auf 128 bringen und an 128er swap_mask anhängen
        if parameters.get("ClipEnableToggle", False):
            mask_clip_512 = self.models_processor.run_CLIPs(original_face_512, parameters["ClipText"], parameters["ClipAmountSlider"])
            mask_clip_128 = t128_bi(mask_clip_512)
            swap_mask *= mask_clip_128
            swap_mask_noFP *= mask_clip_128

        # ----- Restore Eyes/Mouth (Steuerung im 512er Raum, danach auf 128 mappen)
        if parameters.get('RestoreMouthEnableToggle', False) or parameters.get('RestoreEyesEnableToggle', False):
            M = tform.params[0:2]
            ones_column = np.ones((kps_5.shape[0], 1), dtype=np.float32)
            dst_kps_5 = np.hstack([kps_5, ones_column]) @ M.T

            img_swap_mask = torch.ones((1, 512, 512), dtype=torch.float32, device=self.models_processor.device)
            img_orig_mask = torch.zeros((1, 512, 512), dtype=torch.float32, device=self.models_processor.device)

            if parameters.get('RestoreMouthEnableToggle', False):
                img_swap_mask = self.models_processor.restore_mouth(
                    img_orig_mask, img_swap_mask, dst_kps_5,
                    parameters['RestoreMouthBlendAmountSlider']/100.0,
                    parameters['RestoreMouthFeatherBlendSlider'],
                    parameters['RestoreMouthSizeFactorSlider']/100.0,
                    parameters['RestoreXMouthRadiusFactorDecimalSlider'],
                    parameters['RestoreYMouthRadiusFactorDecimalSlider'],
                    parameters['RestoreXMouthOffsetSlider'],
                    parameters['RestoreYMouthOffsetSlider']
                ).clamp(0, 1)

            if parameters.get('RestoreEyesEnableToggle', False):
                img_swap_mask = self.models_processor.restore_eyes(
                    img_orig_mask, img_swap_mask, dst_kps_5,
                    parameters['RestoreEyesBlendAmountSlider']/100.0,
                    parameters['RestoreEyesFeatherBlendSlider'],
                    parameters['RestoreEyesSizeFactorDecimalSlider'],
                    parameters['RestoreXEyesRadiusFactorDecimalSlider'],
                    parameters['RestoreYEyesRadiusFactorDecimalSlider'],
                    parameters['RestoreXEyesOffsetSlider'],
                    parameters['RestoreYEyesOffsetSlider'],
                    parameters['RestoreEyesSpacingOffsetSlider']
                ).clamp(0, 1)

            if parameters.get('RestoreEyesMouthBlurSlider', 0) > 0:
                b = parameters['RestoreEyesMouthBlurSlider']
                gauss = transforms.GaussianBlur(b*2+1, (b+1)*0.2)
                img_swap_mask = gauss(img_swap_mask)

            mask_128 = t128_bi(img_swap_mask)
            #swap_mask_noFP = swap_mask_noFP * mask_128
            swap_mask      = swap_mask      * mask_128
        
        # -------------------------------
        # DFL XSeg
        # -------------------------------
        # calc_mask: 512er "Erlaubnis"-Maske; mask_calc_dill: 512er dilatierte Variante (für Restorer-Auto)
        calc_mask     = torch.ones((1, 512, 512), dtype=torch.float32, device=self.models_processor.device)
        mask_calc_dill= torch.zeros((1, 512, 512), dtype=torch.float32, device=self.models_processor.device)

        if parameters.get("DFLXSegEnableToggle", False):
            # Basisbild für XSeg
            img_xseg_256 = t256_near(original_face_512)

            # Mouth ggf. auf 256 mappen
            mouth_256 = None
            if parameters.get("DFLXSeg2EnableToggle", False) and parameters.get("XSegMouthEnableToggle", False) \
               and parameters.get("DFLXSegSizeSlider", 0) != parameters.get("DFLXSeg2SizeSlider", 0) \
               and mouth_512 is not None:
                mouth_256 = t256_near(mouth_512.unsqueeze(0))  # [1,256,256]

            # apply_dfl_xseg liefert: img_mask(256), mask_forcalc(256), mask_calc_dill(256)
            img_mask_256, mask_forcalc_256, mask_calc_dill_256, outpred_noFP_256 = self.models_processor.apply_dfl_xseg(
                img_xseg_256,
                -parameters["DFLXSegSizeSlider"],
                0,                               # BgExcludeOccluder war nicht genutzt
                mouth_256 if mouth_256 is not None else 0,
                parameters
            )

            # auf 512 bringen
            img_mask_512       = t512_mask(img_mask_256)
            mask_forcalc_512   = t512_mask(mask_forcalc_256)
            mask_calc_dill_512 = t512_mask(mask_calc_dill_256)
            outpred_noFP_128 = t128_bi(outpred_noFP_256)

            # Deine Logik: invertiert als Erlaubnis-/Calc-Masken
            calc_mask      = (1.0 - mask_forcalc_512).clamp(0,1)          # [1,512,512]
            mask_calc_dill = (1.0 - mask_calc_dill_512).clamp(0,1)

            # swap_mask reduzieren (128er)
            img_mask_128 = t128_bi(img_mask_512)
            swap_mask_noFP = swap_mask_noFP * (1.0 - outpred_noFP_128)
            swap_mask      = swap_mask      * (1.0 - img_mask_128)
        else:
            # kein XSeg -> calc_mask aus FaceParser/Basis
            calc_mask = t512_mask(swap_mask.clone()).clamp(0,1)
            
        calc_mask = torch.where(calc_mask > 0.1, 1, 0).float()
        mask_calc_dill = torch.where(mask_calc_dill > 0.1, 1, 0).float()
        
        # First Restorer and Auto Restore pass (after masks)
        if parameters["FaceRestorerEnableToggle"] and parameters["FaceRestorerAutoEnableToggle"]:

            original_face_512_autorestore = original_face_512.clone()
            swap_original_autorestore = swap_original.clone()
            alpha_restorer = float(parameters["FaceRestorerBlendSlider"])/100.0
            adjust_sharpness = float(parameters["FaceRestorerAutoSharpAdjustSlider"])
            scale_factor = round(tform.scale, 2)
            automasktoggle = parameters["FaceRestorerAutoMaskEnableToggle"]
            automaskadjust = parameters["FaceRestorerAutoSharpMaskAdjustDecimalSlider"]
            automaskblur = 2 #parameters["FaceRestorerAutoSharpMaskBlurSlider"]
            restore_mask = mask_calc_dill.clone()
            
            alpha_auto, blur_value = self.face_restorer_auto(original_face_512_autorestore, swap_original_autorestore, swap_restorecalc, alpha_restorer, adjust_sharpness, scale_factor, debug, restore_mask, automasktoggle, automaskadjust, automaskblur)#, parameters["FaceRestorerMaskSlider"], parameters["AutoRestorerTenengradTreshSlider"]/100, parameters["AutoRestorerCombWeightSlider"]/100)

            if blur_value > 0:
                kernel_size = 2 * blur_value + 1
                sigma       = blur_value * 0.1
                gaussian_blur = transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)
                swap = gaussian_blur(swap_original)
                debug_info["Restore1"] = f": {-blur_value:.2f}"
            elif isinstance(alpha_auto, torch.Tensor):
                swap = swap_restorecalc * alpha_auto + swap_original * (1 - alpha_auto)                
            elif alpha_auto != 0:
                swap = swap_restorecalc * alpha_auto + swap_original * (1 - alpha_auto)
                if debug:
                    debug_info["Restore1"] = f": {alpha_auto*100:.2f}"
            else:
                swap = swap_original 
                if debug:
                    debug_info["Restore1"] = f": {alpha_auto*100:.2f}"

        elif parameters["FaceRestorerEnableToggle"]:
            alpha_restorer = float(parameters["FaceRestorerBlendSlider"])/100.0
            swap = torch.add(torch.mul(swap_restorecalc, alpha_restorer), torch.mul(swap_original, 1 - alpha_restorer))                             

        # Expression Restorer After First Restorer
        if parameters['FaceExpressionEnableToggleBoth'] and (parameters['FaceExpressionLipsToggle'] or parameters['FaceExpressionEyesToggle']) and parameters['FaceExpressionBeforeTypeSelection'] == 'After First Restorer':
            swap = self.apply_face_expression_restorer(original_face_512, swap, parameters)

        # Face editor After First Restorer
        if parameters['FaceEditorEnableToggle'] and self.main_window.editFacesButton.isChecked() and parameters['FaceEditorBeforeTypeSelection'] == 'After First Restorer':
            editor_mask = t512_mask(swap_mask).clone()
            swap = swap * editor_mask + original_face_512 * (1 - editor_mask)
            swap = self.swap_edit_face_core(swap, kps, parameters, control)
            swap_mask = swap_mask_noFP

        # Second Denoiser pass - After First Restorer
        if control.get('DenoiserAfterFirstRestorerToggle', False):
            swap = self._apply_denoiser_pass(swap, control, "AfterFirst", kv_map)

        if parameters["FaceRestorerEnable2Toggle"] and not parameters["FaceRestorerEnable2EndToggle"]:
            swap_original2 = swap.clone()

            swap2 = self.models_processor.apply_facerestorer(swap, parameters['FaceRestorerDetType2Selection'], parameters['FaceRestorerType2Selection'], parameters["FaceRestorerBlend2Slider"], parameters['FaceFidelityWeight2DecimalSlider'], control['DetectorScoreSlider'])
         
            if parameters["FaceRestorerAutoEnable2Toggle"]:
                original_face_512_autorestore2 = original_face_512.clone()
                swap_original_autorestore2 = swap_original2.clone()
                alpha_restorer2 = float(parameters["FaceRestorerBlend2Slider"])/100.0
                adjust_sharpness2 = float(parameters["FaceRestorerAutoSharpAdjust2Slider"])
                scale_factor2 = round(tform.scale, 2)
                automasktoggle2 = parameters["FaceRestorerAutoMask2EnableToggle"]
                automaskadjust2 = parameters["FaceRestorerAutoSharpMask2AdjustDecimalSlider"]
                automaskblur2 = 2 #parameters["FaceRestorerAutoSharpMask2BlurSlider"]
                restore_mask = mask_calc_dill.clone()
                
                alpha_auto2, blur_value2 = self.face_restorer_auto(original_face_512_autorestore2, swap_original_autorestore2, swap2, alpha_restorer2, adjust_sharpness2, scale_factor2, debug, restore_mask, automasktoggle2, automaskadjust2, automaskblur2)#, parameters["FaceRestorerMaskSlider"], parameters["AutoRestorerTenengradTreshSlider"]/100, parameters["AutoRestorerCombWeightSlider"]/100)

                if blur_value2 > 0:
                    kernel_size = 2 * blur_value2 + 1
                    sigma       = blur_value2 * 0.1
                    gaussian_blur = transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)
                    swap = gaussian_blur(swap_original2)
                    debug_info["Restore2"] = f": {-blur_value2:.2f}"
                elif isinstance(alpha_auto2, torch.Tensor):
                    swap = swap2 * alpha_auto2 + swap_original2 * (1 - alpha_auto2)
                elif alpha_auto2 != 0:
                    swap = swap2 * alpha_auto2 + swap_original2 * (1 - alpha_auto2)
                    if debug:
                        debug_info["Restore2"] = f": {alpha_auto2*100:.2f}"
                else:
                    swap = swap_original2 
                    if debug:
                        debug_info["Restore2"] = f": {alpha_auto2*100:.2f}"
            else:
                alpha_restorer2 = float(parameters["FaceRestorerBlend2Slider"])/100.0
                swap = torch.add(torch.mul(swap2, alpha_restorer2), torch.mul(swap_original2, 1 - alpha_restorer2))

        # Expression Restorer After Second Restorer
        if parameters['FaceExpressionEnableToggleBoth'] and (parameters['FaceExpressionLipsToggle'] or parameters['FaceExpressionEyesToggle']) and parameters['FaceExpressionBeforeTypeSelection'] == 'After Second Restorer':
            swap = self.apply_face_expression_restorer(original_face_512, swap, parameters)

        # Face editor After Second Restorer
        if parameters['FaceEditorEnableToggle'] and self.main_window.editFacesButton.isChecked() and parameters['FaceEditorBeforeTypeSelection'] == 'After Second Restorer':
            editor_mask = t512_mask(swap_mask).clone()
            swap = swap * editor_mask + original_face_512 * (1 - editor_mask)
            swap = self.swap_edit_face_core(swap, kps, parameters, control)
            swap_mask = swap_mask_noFP

        # Textures and Color pass begins
        if parameters["TransferTextureEnableToggle"] or parameters["DifferencingEnableToggle"] or parameters["AutoColorEnableToggle"]:          
            mask_calc = mask_calc_dill.clone()
            mask_calc = 1 - mask_calc
            
            mask_calc = mask_calc# + (BgExclude)
            mask_calc = torch.where(
                mask_calc > 0.01, 
                1,
                0
            )
            
            if parameters['BGExcludeBlurAmountSlider'] > 0:
                orig = mask_calc.clone()
                gauss = transforms.GaussianBlur(parameters['BGExcludeBlurAmountSlider']*2+1, (parameters['BGExcludeBlurAmountSlider']+1)*0.2)
                mask_calc = gauss(mask_calc.type(torch.float32))
                mask_calc = torch.max(mask_calc, orig) 

        # -------------------------------
        # AutoColor (Maske 512)
        # -------------------------------
        if parameters.get("AutoColorEnableToggle", False):
            # calc_mask ist [1,512,512], 1=erlaubt
            mask_autocolor = calc_mask.clone()
            mask_autocolor = (mask_autocolor > 0.1)
            #swap_backup = swap.clone()
            

            if parameters['AutoColorTransferTypeSelection'] == 'Test':
                swap = faceutil.histogram_matching(original_face_512, swap, parameters["AutoColorBlendAmountSlider"])

            elif parameters['AutoColorTransferTypeSelection'] == 'Test_Mask':
                swap = faceutil.histogram_matching_withmask(original_face_512, swap, mask_autocolor, parameters["AutoColorBlendAmountSlider"])
                #if parameters.get("ExcludeMaskEnableToggle", False):
                #    swap_backup = faceutil.histogram_matching_withmask(original_face_512, swap_backup, mask_autocolor, parameters["AutoColorBlendAmountSlider"])

            elif parameters['AutoColorTransferTypeSelection'] == 'DFL_Test':
                swap = faceutil.histogram_matching_DFL_test(original_face_512, swap, parameters["AutoColorBlendAmountSlider"])

            elif parameters['AutoColorTransferTypeSelection'] == 'DFL_Orig':
                swap = faceutil.histogram_matching_DFL_Orig(original_face_512, swap, mask_autocolor, 100)
        
        # -------------------------------
        # TransferTexture
        # -------------------------------
        if parameters.get("TransferTextureEnableToggle", False):
            # Basis-Maske(n)
            # calc_mask: [1,512,512], 1=erlaubt
            # texture_exclude_512: [1,512,512], 1=ausschließen  (aus Parser)
            # Wir bauen mask_final (512): 1 = erlauben, 0 = sperren
            if parameters.get("ExcludeMaskEnableToggle", False) and texture_exclude_512 is not None:
                # Optionaler Blend der FaceParser-Texture-Maske
                m = texture_exclude_512.clone()
                if parameters.get('FaceParserBlurTextureSlider', 0) > 0:
                    b = parameters['FaceParserBlurTextureSlider']
                    gauss = transforms.GaussianBlur(b*2+1, (b+1)*0.2)
                    m = torch.max(gauss(m.float()), m)  # blur + preserve edges
                # Ausschluss invertieren -> Erlauben
                mask_final_512 = (1.0 - m).clamp(0,1) + (mask_calc)
                mask_final_512 = mask_final_512.clamp(0,1)
                #if parameters.get("FaceParserBlendTextureSlider", 0) != 0:
                #    mask_final_512 = (mask_final_512 + parameters["FaceParserBlendTextureSlider"]/100.0).clamp(0,1)
                #mask_128_for_vgg = v2.Resize((128,128), interpolation=v2.InterpolationMode.BILINEAR, antialias=False)(mask_final_512)
            elif parameters.get("ExcludeOriginalVGGMaskEnableToggle", False):
                mask_final_512 = calc_mask.clone() #mask_calc #torch.ones((512, 512), dtype=torch.uint8, device=self.models_processor.device)
            else:
                mask_final_512 = (1 - calc_mask.clone()) + mask_calc
                #mask_128_for_vgg = v2.Resize((128,128), interpolation=v2.InterpolationMode.BILINEAR, antialias=False)(mask_final_512)
                #mask_final_512 = 1-calc_mask.clone()
                

            # ggf. auf 128er für VGG-Diff (falls dein apply_perceptual_diff_onnx das erwartet)
            #mask_128_for_vgg = v2.Resize((128,128), interpolation=v2.InterpolationMode.BILINEAR, antialias=False)(mask_final_512)

            # Histogrammvor-Anpassungen (wie bei dir)
            swap_texture_backup = faceutil.histogram_matching_DFL_Orig(original_face_512, swap.clone(), calc_mask, 100)

            # Gradient (dein Shader)
            TransferTextureKernelSizeSlider   = 12
            TransferTextureSigmaDecimalSlider = 4.00
            TransferTextureWeightSlider       = 1
            TransferTexturePhiDecimalSlider   = 9.7
            TransferTextureGammaDecimalSlider = 0.5
            if parameters.get('TransferTextureModeEnableToggle', False):
                TransferTextureLambdSlider = 8
                TransferTextureThetaSlider = 8
            else:
                TransferTextureLambdSlider = 2
                TransferTextureThetaSlider = 1
            TextureFeatureLayerTypeSelection = 'combo_relu3_3_relu3_1'

            clip_limit     = parameters['TransferTextureClipLimitDecimalSlider'] if parameters.get('TransferTextureClaheEnableToggle', False) else 0.0
            alpha_clahe    = parameters['TransferTextureAlphaClaheDecimalSlider']
            grid_size      = (4,4)
            global_gamma   = parameters['TransferTexturePreGammaDecimalSlider']
            global_contrast= parameters['TransferTexturePreContrastDecimalSlider']

            gradient_texture = self.gradient_magnitude(
                original_face_512, calc_mask,
                TransferTextureKernelSizeSlider, TransferTextureWeightSlider,
                TransferTextureSigmaDecimalSlider, TransferTextureLambdSlider,
                TransferTextureGammaDecimalSlider, TransferTexturePhiDecimalSlider,
                TransferTextureThetaSlider, clip_limit, alpha_clahe, grid_size, global_gamma, global_contrast
            )
            gradient_texture = faceutil.histogram_matching_DFL_Orig(original_face_512, gradient_texture, calc_mask, 100)

            # Optional VGG-Exclude (du hast beides kombiniert: VGG + Exclude)
            if parameters.get("ExcludeOriginalVGGMaskEnableToggle", False):
                mask_input = t128_mask(calc_mask.clone())
                # NEU: zwei Slider
                thr = parameters['VGGMaskThresholdSlider']     # 0..100
                soft = 100 #parameters['VGGMaskSoftnessSlider']     # 0..100
                #gamma = parameters['VGGMaskGammaSlider']     # 0..100
                upper_thresh = parameters['TextureUpperLimitSlider']/100
                #if parameters['ExcludeVGGMaskSmoothEnableToggle']:
                #    mode='smooth'
                #else:
                #    mode='linear'
                mode='smooth'
                mask_vgg_512, diff_norm_texture = self.models_processor.apply_vgg_mask_simple(
                    swap, original_face_512, mask_input,
                    center_pct=thr, softness_pct=soft,
                    feature_layer=TextureFeatureLayerTypeSelection,
                    mode=mode  # oder 'linear'
                )
                # mask_vgg_512: [1,512,512] in [0..1]
                 #Wenn ExcludeVGGMaskEnableToggle==False -> mask_vgg_512 = diff_norm_texture
                if not parameters.get('ExcludeVGGMaskEnableToggle', False):
                    diff_norm_texture = torch.nn.functional.interpolate(
                        diff_norm_texture.unsqueeze(0),
                        size=(512, 512),
                        mode='bilinear',
                        align_corners=True
                    ).squeeze(0)
                    mask_vgg_512 = diff_norm_texture

                mask_vgg_512 = torch.where(mask_vgg_512 >= upper_thresh, upper_thresh, mask_vgg_512)
                mask_vgg_512 = mask_vgg_512 + mask_calc

                if parameters.get('TextureBlendAmountSlider', 0) > 0:
                    b = parameters['TextureBlendAmountSlider']
                    gauss = transforms.GaussianBlur(b*2+1, (b+1)*0.2)
                    mask_vgg_512 = gauss(mask_vgg_512.float())
                    
                # Boost (Gamma < 1 macht härter, >1 macht softer)
                #gamma = 0.9 + 0.6 * (1.0 - gamma/100.0)  # bei geringer Softness etwas härter
                #mask_vgg_512 = mask_vgg_512.clamp(1e-6,1).pow(gamma)
                
                # VGG-Maske als weiteres Limit: 512
                mask_vgg_512 = t512_mask(mask_vgg_512)
                # final erlauben = mask_final_512 * (1 - (1 - mask_vgg_512)) = mask_final_512 * mask_vgg
                mask_final_512 = (mask_final_512 * mask_vgg_512).clamp(0,1)
                mask_final_512 = (mask_final_512 + mask_calc).clamp(0,1)                

            # Mischen:  w = alpha*(1 - mask_final_512)
            alpha_t = parameters['TransferTextureBlendAmountSlider']/100.0
            w = alpha_t * (1.0 - mask_final_512)# + (calc_mask)
            w = w.clamp(0,1)
            swap = (swap_texture_backup * (1.0 - w) + gradient_texture * w).clamp(0,255)

            texture_mask_view = (1.0 - mask_final_512).clone()  # falls du für Debug/Anzeige brauchst

        # -------------------------------
        # Differencing
        # -------------------------------
        if parameters.get("DifferencingEnableToggle", False):
            # 128er Eingabemaske für VGG
            diff_mask_128 = v2.Resize((128,128), interpolation=v2.InterpolationMode.BILINEAR, antialias=False)(calc_mask.clone())

            swapped_face_resized  = swap.clone()
            original_face_resized = original_face_512.clone()
            FeatureLayerTypeSelection = 'combo_relu3_3_relu3_1'

            lower_thresh = parameters['DifferencingLowerLimitThreshSlider']/100.0
            upper_thresh = parameters['DifferencingUpperLimitThreshSlider']/100.0
            middle_value = parameters['DifferencingMiddleLimitValueSlider']/100.0
            upper_value  = parameters['DifferencingUpperLimitValueSlider']/100.0

            mask_diff_128, diff_norm_texture = self.models_processor.apply_perceptual_diff_onnx(
                swapped_face_resized, original_face_resized, diff_mask_128,
                lower_thresh, 0, upper_thresh, upper_value, middle_value,
                FeatureLayerTypeSelection, False
            )

            # piecewise auf diff_norm_texture (wie bei dir), dann 512 & Blur/Blend
            eps = 1e-6
            inv_lower = 1.0 / max(lower_thresh, eps)
            inv_mid   = 1.0 / max((upper_thresh - lower_thresh), eps)
            inv_high  = 1.0 / max((1.0 - upper_thresh), eps)

            res_low  = diff_norm_texture * inv_lower * middle_value
            res_mid  = middle_value + (diff_norm_texture - lower_thresh) * inv_mid * (upper_value - middle_value)
            res_high = upper_value  + (diff_norm_texture - upper_thresh) * inv_high * (1.0 - upper_value)

            piece = torch.where(
                diff_norm_texture < lower_thresh, res_low,
                torch.where(diff_norm_texture > upper_thresh, res_high, res_mid)
            )

            mask512 = t512_mask(piece)
            if parameters.get('DifferencingBlendAmountSlider', 0) > 0:
                b = parameters['DifferencingBlendAmountSlider']
                gauss = transforms.GaussianBlur(b*2+1, (b+1)*0.2)
                mask512 = gauss(mask512.float())

            # mit calc_mask kombinieren
            mask512 = (mask512 + mask_calc).clamp(0,1)

            swap = (swap * mask512 + original_face_512 * (1.0 - mask512)).clamp(0,255)
            diff_mask = 1-mask512.clone()  # falls du später "diff" anzeigen willst
        
        # Apply color corrections
        if parameters['ColorEnableToggle']:
            swap = torch.unsqueeze(swap,0).contiguous()
            swap = v2.functional.adjust_gamma(swap, parameters['ColorGammaDecimalSlider'], 1.0)
            swap = torch.squeeze(swap)
            swap = swap.permute(1, 2, 0).type(torch.float32)

            del_color = torch.tensor([parameters['ColorRedSlider'], parameters['ColorGreenSlider'], parameters['ColorBlueSlider']], device=self.models_processor.device)
            swap += del_color
            swap = torch.clamp(swap, min=0., max=255.)
            swap = swap.permute(2, 0, 1) / 255.0

            swap = v2.functional.adjust_brightness(swap, parameters['ColorBrightnessDecimalSlider'])
            swap = v2.functional.adjust_contrast(swap, parameters['ColorContrastDecimalSlider'])
            swap = v2.functional.adjust_saturation(swap, parameters['ColorSaturationDecimalSlider'])
            swap = v2.functional.adjust_sharpness(swap, parameters['ColorSharpnessDecimalSlider'])
            swap = v2.functional.adjust_hue(swap, parameters['ColorHueDecimalSlider'])

            swap = swap * 255.0

        # Face editor After Second Restorer
        if parameters['FaceEditorEnableToggle'] and self.main_window.editFacesButton.isChecked() and parameters['FaceEditorBeforeTypeSelection'] == 'After Texture Transfer':
            editor_mask = t512_mask(swap_mask).clone()
            swap = swap * editor_mask + original_face_512 * (1 - editor_mask)
            swap = self.swap_edit_face_core(swap, kps, parameters, control)
            swap_mask = swap_mask_noFP

        # Second Restorer - After Diff / Texture Transfer and AutoColor
        if parameters["FaceRestorerEnable2Toggle"] and parameters["FaceRestorerEnable2EndToggle"]:
            swap_original2 = swap.clone()
            swap2 = self.models_processor.apply_facerestorer(swap, parameters['FaceRestorerDetType2Selection'], parameters['FaceRestorerType2Selection'], parameters["FaceRestorerBlend2Slider"], parameters['FaceFidelityWeight2DecimalSlider'], control['DetectorScoreSlider'])
            
            if parameters["FaceRestorerAutoEnable2Toggle"]:
                original_face_512_autorestore2 = original_face_512.clone()
                swap_original_autorestore2 = swap_original2.clone()
                alpha_restorer2 = float(parameters["FaceRestorerBlend2Slider"])/100.0
                adjust_sharpness2 = float(parameters["FaceRestorerAutoSharpAdjust2Slider"])
                scale_factor2 = round(tform.scale, 2)                
                automasktoggle2 = parameters["FaceRestorerAutoMask2EnableToggle"]
                automaskadjust2 = parameters["FaceRestorerAutoSharpMask2AdjustDecimalSlider"]
                automaskblur2 = 2 #parameters["FaceRestorerAutoSharpMask2BlurSlider"]
                restore_mask = mask_calc_dill.clone()
                
                alpha_auto2, blur_value2 = self.face_restorer_auto(original_face_512_autorestore2, swap_original_autorestore2, swap2, alpha_restorer2, adjust_sharpness2, scale_factor2, debug, restore_mask, automasktoggle2, automaskadjust2, automaskblur2)#, parameters["FaceRestorerMaskSlider"], parameters["AutoRestorerTenengradTreshSlider"]/100, parameters["AutoRestorerCombWeightSlider"]/100)

                if blur_value2 > 0:
                    kernel_size = 2 * blur_value2 + 1                  # 3,5,7,...
                    sigma       = blur_value2 * 0.1
                    gaussian_blur = transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)
                    swap = gaussian_blur(swap_original2)
                    debug_info["Restore2"] = f": {-blur_value2:.2f}"
                elif isinstance(alpha_auto2, torch.Tensor):
                    swap = swap2 * alpha_auto2 + swap_original2 * (1 - alpha_auto2)
                elif alpha_auto2 != 0:
                    swap = swap2 * alpha_auto2 + swap_original2 * (1 - alpha_auto2)
                    if debug:
                        debug_info["Restore2"] = f": {alpha_auto2*100:.2f}"
                else:
                    swap = swap_original2 
                    if debug:
                        debug_info["Restore2"] = f": {alpha_auto2*100:.2f}"
            else:
                alpha_restorer2 = float(parameters["FaceRestorerBlend2Slider"])/100.0
                swap = torch.add(torch.mul(swap2, alpha_restorer2), torch.mul(swap_original2, 1 - alpha_restorer2))

        # Third denoiser pass - After restorers
        if control.get('DenoiserAfterRestorersToggle', False):
            swap = self._apply_denoiser_pass(swap, control, "After", kv_map)

        # -------------------------------
        # AutoColor (Maske 512) - second pass at the end (to color the restored and denoized faces after the first pipeline pass)
        # -------------------------------
        if parameters.get("AutoColorEnableToggle", False) and parameters.get("AutoColorEndEnableToggle", False):
            # calc_mask ist [1,512,512], 1=erlaubt
            mask_autocolor = calc_mask.clone()
            mask_autocolor = (mask_autocolor > 0.1)
            #swap_backup = swap.clone()
            
            if parameters['AutoColorTransferTypeSelection'] == 'Test':
                swap = faceutil.histogram_matching(original_face_512, swap, parameters["AutoColorBlendAmountSlider"])

            elif parameters['AutoColorTransferTypeSelection'] == 'Test_Mask':
                swap = faceutil.histogram_matching_withmask(original_face_512, swap, mask_autocolor, parameters["AutoColorBlendAmountSlider"])
                #if parameters.get("ExcludeMaskEnableToggle", False):
                #    swap_backup = faceutil.histogram_matching_withmask(original_face_512, swap_backup, mask_autocolor, parameters["AutoColorBlendAmountSlider"])

            elif parameters['AutoColorTransferTypeSelection'] == 'DFL_Test':
                swap = faceutil.histogram_matching_DFL_test(original_face_512, swap, parameters["AutoColorBlendAmountSlider"])

            elif parameters['AutoColorTransferTypeSelection'] == 'DFL_Orig':
                swap = faceutil.histogram_matching_DFL_Orig(original_face_512, swap, mask_autocolor, 100)
        
        # Final blending
        if parameters['FinalBlendAdjEnableToggle'] and parameters['FinalBlendAmountSlider'] > 0:
            final_blur_strength = parameters['FinalBlendAmountSlider']  # Ein Parameter steuert beides
            # Bestimme kernel_size und sigma basierend auf dem Parameter
            kernel_size = 2 * final_blur_strength + 1  # Ungerade Zahl, z.B. 3, 5, 7, ...
            sigma = final_blur_strength * 0.1  # Sigma proportional zur Stärke
            # Gaussian Blur anwenden
            gaussian_blur = transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)
            swap = gaussian_blur(swap)

        # Artefact pass - Jpeg and Blockshift
        if parameters['JPEGCompressionEnableToggle']:
            jpeg_q = int(parameters["JPEGCompressionAmountSlider"])
            if jpeg_q != 100:
                s = float(tform.scale)

                gamma     = 0.60 #parameters["JPEGCompressionGammaDecimalSlider"]
                strength  = 0.80 #parameters["JPEGCompressionAdjustSlider"]/100
                q_min     = 14 #parameters["JPEGCompressionQMinSlider"]
                q_max     = 100 #parameters["JPEGCompressionQMaxSlider"]

                jpeg_q_eff = faceutil._map_jpeg_quality(
                    base_q=jpeg_q, face_scale=s,
                    gamma=gamma, strength=strength,
                    q_min=q_min, q_max=q_max
                )

                if debug:
                    debug_info["JPEG Quality"] = f"{jpeg_q_eff}"

                swap2 = faceutil.jpegBlur(swap, jpeg_q_eff)
                blend = parameters['JPEGCompressionBlendSlider'] / 100.0
                swap  = torch.add(swap2 * blend, swap * (1.0 - blend))

        if parameters["BlockShiftEnableToggle"]:
            base_quality = parameters["BlockShiftAmountSlider"]     
                                                                                                      
            max_px = parameters["BlockShiftMaxAmountSlider"]
            
            swap2 = self.apply_block_shift_gpu_jitter(
                swap, 
                block_size=base_quality, 
                max_amount_pixels=float(max_px),
                seed=1337
            )
                                                                                             
            block_shift_blend = parameters["BlockShiftBlendAmountSlider"]/100.0
            swap = swap2 * block_shift_blend + swap * (1.0 - block_shift_blend)  

            swap = torch.add(torch.mul(swap2, block_shift_blend), torch.mul(swap, 1 - block_shift_blend))                          
            
        if parameters['ColorNoiseDecimalSlider'] > 0:
            noise = (torch.rand_like(swap) - 0.5) * 2 * parameters['ColorNoiseDecimalSlider']
            swap = torch.clamp(swap + noise, 0.0, 255.0)

        if control.get('AnalyseImageEnableToggle', False):
            image_analyse_swap = self.analyze_image(swap)
            if debug:
                debug_info["JS: "] = image_analyse_swap
        
        if debug and debug_info:
            one_liner = ", ".join(f"{key}={value}" for key, value in debug_info.items())
            print(f"[DEBUG] {one_liner}")

        if is_perspective_crop:
            return swap, t512_mask(swap_mask), None

        # Add blur to swap_mask results
        gauss = transforms.GaussianBlur(parameters['OverallMaskBlendAmountSlider'] * 2 + 1, (parameters['OverallMaskBlendAmountSlider'] + 1) * 0.2)
        swap_mask = gauss(swap_mask)

        # Combine border and swap mask, scale, and apply to swap
        swap_mask = torch.mul(swap_mask, border_mask)
        swap_mask = t512_mask(swap_mask)
        swap = torch.mul(swap, swap_mask)

        # For face comparing
        original_face_512_clone = None
        if self.is_view_face_compare:
            original_face_512_clone = original_face_512.clone()
            original_face_512_clone = original_face_512_clone.type(torch.uint8)
            original_face_512_clone = original_face_512_clone.permute(1, 2, 0)
        swap_mask_clone = None
        # Uninvert and create image from swap mask
        if self.is_view_face_mask:
            mask_show_type = parameters['MaskShowSelection'] 
            if mask_show_type == 'swap_mask':
                if parameters['FaceEditorEnableToggle'] and self.main_window.editFacesButton.isChecked():
                    swap_mask_clone = t512_mask(torch.ones_like(swap_mask)).clone()
                else:
                    swap_mask_clone = swap_mask.clone()
            elif mask_show_type == 'diff':
                swap_mask_clone = diff_mask.clone()                        
            elif mask_show_type == 'texture':
                swap_mask_clone = texture_mask_view.clone() 
            swap_mask_clone = torch.sub(1, swap_mask_clone)
            swap_mask_clone = torch.cat((swap_mask_clone,swap_mask_clone,swap_mask_clone),0)
            swap_mask_clone = swap_mask_clone.permute(1, 2, 0)
            swap_mask_clone = torch.mul(swap_mask_clone, 255.).type(torch.uint8)

        # Calculate the area to be mergerd back to the original frame
        IM512 = tform.inverse.params[0:2, :]
        corners = np.array([[0,0], [0,511], [511, 0], [511, 511]])

        x = (IM512[0][0]*corners[:,0] + IM512[0][1]*corners[:,1] + IM512[0][2])
        y = (IM512[1][0]*corners[:,0] + IM512[1][1]*corners[:,1] + IM512[1][2])

        left = floor(np.min(x))
        if left<0:
            left=0
        top = floor(np.min(y))
        if top<0:
            top=0
        right = ceil(np.max(x))
        if right>img.shape[2]:
            right=img.shape[2]
        bottom = ceil(np.max(y))
        if bottom>img.shape[1]:
            bottom=img.shape[1]

        # Untransform the swap
        swap = v2.functional.pad(swap, (0,0,img.shape[2]-512, img.shape[1]-512))
        swap = v2.functional.affine(swap, tform.inverse.rotation*57.2958, (tform.inverse.translation[0], tform.inverse.translation[1]), tform.inverse.scale, 0, interpolation=interpolation_Untransform, center = (0,0) )
        swap = swap[0:3, top:bottom, left:right]


        # Untransform the swap mask
        swap_mask = v2.functional.pad(swap_mask, (0,0,img.shape[2]-512, img.shape[1]-512))
        swap_mask = v2.functional.affine(swap_mask, tform.inverse.rotation*57.2958, (tform.inverse.translation[0], tform.inverse.translation[1]), tform.inverse.scale, 0, interpolation=v2.InterpolationMode.BILINEAR, center = (0,0) )
        swap_mask = swap_mask[0:1, top:bottom, left:right]
        swap_mask_minus = swap_mask.clone()
        swap_mask_minus = torch.sub(1, swap_mask)

        # Apply the mask to the original image areas
        img_crop = img[0:3, top:bottom, left:right]
        img_crop = torch.mul(swap_mask_minus,img_crop)
            
        #Add the cropped areas and place them back into the original image
        swap = torch.add(swap, img_crop)
        swap = swap.type(torch.uint8)
        swap = swap.clamp(0, 255)
                                              
        img[0:3, top:bottom, left:right] = swap

        return img, original_face_512_clone, swap_mask_clone

    def enhance_core(self, img, control):
        enhancer_type = control['FrameEnhancerTypeSelection']

        match enhancer_type:
            case 'RealEsrgan-x2-Plus' | 'RealEsrgan-x4-Plus' | 'BSRGan-x2' | 'BSRGan-x4' | 'UltraSharp-x4' | 'UltraMix-x4' | 'RealEsr-General-x4v3':
                tile_size = 512

                if enhancer_type == 'RealEsrgan-x2-Plus' or enhancer_type == 'BSRGan-x2':
                    scale = 2
                else:
                    scale = 4

                image = img.type(torch.float32)
                if torch.max(image) > 256:  # 16-bit image
                    max_range = 65535
                else:
                    max_range = 255

                image = torch.div(image, max_range)
                image = torch.unsqueeze(image, 0).contiguous()

                image = self.models_processor.run_enhance_frame_tile_process(image, enhancer_type, tile_size=tile_size, scale=scale)

                image = torch.squeeze(image)
                image = torch.clamp(image, 0, 1)
                image = torch.mul(image, max_range)

                # Blend
                alpha = float(control["FrameEnhancerBlendSlider"])/100.0

                t_scale = v2.Resize((img.shape[1] * scale, img.shape[2] * scale), interpolation=v2.InterpolationMode.BILINEAR, antialias=False)
                img = t_scale(img)
                img = torch.add(torch.mul(image, alpha), torch.mul(img, 1-alpha))
                if max_range == 255:
                    img = img.type(torch.uint8)
                else:
                    img = img.type(torch.uint16)

            case 'DeOldify-Artistic' | 'DeOldify-Stable' | 'DeOldify-Video':
                render_factor = 384 # 12 * 32 | highest quality = 20 * 32 == 640

                _, h, w = img.shape
                t_resize_i = v2.Resize((render_factor, render_factor), interpolation=v2.InterpolationMode.BILINEAR, antialias=False)
                image = t_resize_i(img)

                image = image.type(torch.float32)
                image = torch.unsqueeze(image, 0).contiguous()

                output = torch.empty((image.shape), dtype=torch.float32, device=self.models_processor.device).contiguous()

                match enhancer_type:
                    case 'DeOldify-Artistic':
                        self.models_processor.run_deoldify_artistic(image, output)
                    case 'DeOldify-Stable':
                        self.models_processor.run_deoldify_stable(image, output)
                    case 'DeOldify-Video':
                        self.models_processor.run_deoldify_video(image, output)

                output = torch.squeeze(output)
                t_resize_o = v2.Resize((h, w), interpolation=v2.InterpolationMode.BILINEAR, antialias=False)
                output = t_resize_o(output)

                output = faceutil.rgb_to_yuv(output, True)
                # do a black and white transform first to get better luminance values
                hires = faceutil.rgb_to_yuv(img, True)

                hires[1:3, :, :] = output[1:3, :, :]
                hires = faceutil.yuv_to_rgb(hires, True)

                # Blend
                alpha = float(control["FrameEnhancerBlendSlider"]) / 100.0
                img = torch.add(torch.mul(hires, alpha), torch.mul(img, 1-alpha))

                img = img.type(torch.uint8)

            case 'DDColor-Artistic' | 'DDColor':
                render_factor = 384 # 12 * 32 | highest quality = 20 * 32 == 640

                # Converti RGB a LAB
                #'''
                #orig_l = img.permute(1, 2, 0).cpu().numpy()
                #orig_l = cv2.cvtColor(orig_l, cv2.COLOR_RGB2Lab)
                #orig_l = torch.from_numpy(orig_l).to(self.models_processor.device)
                #orig_l = orig_l.permute(2, 0, 1)
                #'''
                orig_l = faceutil.rgb_to_lab(img, True)

                orig_l = orig_l[0:1, :, :]  # (1, h, w)

                # Resize per il modello
                t_resize_i = v2.Resize((render_factor, render_factor), interpolation=v2.InterpolationMode.BILINEAR, antialias=False)
                image = t_resize_i(img)

                # Converti RGB in LAB
                #'''
                #img_l = image.permute(1, 2, 0).cpu().numpy()
                #img_l = cv2.cvtColor(img_l, cv2.COLOR_RGB2Lab)
                #img_l = torch.from_numpy(img_l).to(self.models_processor.device)
                #img_l = img_l.permute(2, 0, 1)
                #'''
                img_l = faceutil.rgb_to_lab(image, True)

                img_l = img_l[0:1, :, :]  # (1, render_factor, render_factor)
                img_gray_lab = torch.cat((img_l, torch.zeros_like(img_l), torch.zeros_like(img_l)), dim=0)  # (3, render_factor, render_factor)

                # Converti LAB in RGB
                #'''
                #img_gray_lab = img_gray_lab.permute(1, 2, 0).cpu().numpy()
                #img_gray_rgb = cv2.cvtColor(img_gray_lab, cv2.COLOR_LAB2RGB)
                #img_gray_rgb = torch.from_numpy(img_gray_rgb).to(self.models_processor.device)
                #img_gray_rgb = img_gray_rgb.permute(2, 0, 1)
                #'''
                img_gray_rgb = faceutil.lab_to_rgb(img_gray_lab)

                tensor_gray_rgb = torch.unsqueeze(img_gray_rgb.type(torch.float32), 0).contiguous()

                # Prepara il tensore per il modello
                output_ab = torch.empty((1, 2, render_factor, render_factor), dtype=torch.float32, device=self.models_processor.device)

                # Esegui il modello
                match enhancer_type:
                    case 'DDColor-Artistic':
                        self.models_processor.run_ddcolor_artistic(tensor_gray_rgb, output_ab)
                    case 'DDColor':
                        self.models_processor.run_ddcolor(tensor_gray_rgb, output_ab)

                output_ab = output_ab.squeeze(0)  # (2, render_factor, render_factor)

                t_resize_o = v2.Resize((img.size(1), img.size(2)), interpolation=v2.InterpolationMode.BILINEAR, antialias=False)
                output_lab_resize = t_resize_o(output_ab)

                # Combina il canale L originale con il risultato del modello
                output_lab = torch.cat((orig_l, output_lab_resize), dim=0)  # (3, original_H, original_W)

                # Convert LAB to RGB
                #'''
                #output_rgb = output_lab.permute(1, 2, 0).cpu().numpy()
                #output_rgb = cv2.cvtColor(output_rgb, cv2.COLOR_Lab2RGB)
                #output_rgb = torch.from_numpy(output_rgb).to(self.models_processor.device)
                #output_rgb = output_rgb.permute(2, 0, 1)
                #'''
                output_rgb = faceutil.lab_to_rgb(output_lab, True)  # (3, original_H, original_W)

                # Miscela le immagini
                alpha = float(control["FrameEnhancerBlendSlider"]) / 100.0
                blended_img = torch.add(torch.mul(output_rgb, alpha), torch.mul(img, 1 - alpha))

                # Converti in uint8
                img = blended_img.type(torch.uint8)

        return img

    def apply_face_expression_restorer(self, driving, target, parameters):
        _, driving_lmk_crop, _ = self.models_processor.run_detect_landmark(driving, bbox=np.array([0, 0, 512, 512]), det_kpss=[], detect_mode='203', score=0.5, from_points=False)
        driving_face_512 = driving.clone()
        driving_face_256 = t256_face(driving_face_512)

        # Making motion templates: driving_template_dct
        c_d_eyes_lst = faceutil.calc_eye_close_ratio(driving_lmk_crop[None]) #c_d_eyes_lst
        c_d_lip_lst = faceutil.calc_lip_close_ratio(driving_lmk_crop[None]) #c_d_lip_lst
        x_d_i_info = self.models_processor.lp_motion_extractor(driving_face_256, 'Human-Face')
        R_d_i = faceutil.get_rotation_matrix(x_d_i_info['pitch'], x_d_i_info['yaw'], x_d_i_info['roll'])
        ##
        
        # R_d_0, x_d_0_info = None, None
        driving_multiplier_eyes=parameters['FaceExpressionFriendlyFactorDecimalSliderEyes'] # Eyes slider
        driving_multiplier_lips=parameters['FaceExpressionFriendlyFactorDecimalSliderLips'] # Lips slider

        flag_activate_eyes = parameters['FaceExpressionEyesToggle']
        flag_eye_retargeting = parameters['FaceExpressionRetargetingEyesEnableToggleBoth'] 
        eye_retargeting_multiplier = parameters['FaceExpressionRetargetingEyesMultiplierDecimalSliderBoth'] 
        flag_activate_lips = parameters['FaceExpressionLipsToggle']
        flag_normalize_lip = parameters['FaceExpressionNormalizeLipsEnableToggleBoth'] 
        lip_normalize_threshold = parameters['FaceExpressionNormalizeLipsThresholdDecimalSliderBoth'] 
        flag_normalize_eyes = parameters['FaceExpressionNormalizeEyesEnableToggleBoth'] 
        eyes_normalize_threshold = parameters['FaceExpressionNormalizeEyesThresholdDecimalSliderBoth'] 
        flag_lip_retargeting = parameters['FaceExpressionRetargetingLipsEnableToggleBoth'] 
        lip_retargeting_multiplier = parameters['FaceExpressionRetargetingLipsMultiplierDecimalSliderBoth']
        eyes_normalize_max = parameters['FaceExpressionNormalizeEyesMaxDecimalSliderBoth']
        flag_relative_motion_eyes = parameters['FaceExpressionRelativeEyesToggle']
        flag_relative_motion_lips = parameters['FaceExpressionRelativeLipsToggle']
        
        lip_delta_before_animation, eye_delta_before_animation = None, None

        target = torch.clamp(target, 0, 255).type(torch.uint8)

        _, source_lmk, _ = self.models_processor.run_detect_landmark(target, bbox=np.array([0, 0, 512, 512]), det_kpss=[], detect_mode='203', score=0.5, from_points=False)

        target_face_512, M_o2c, M_c2o = faceutil.warp_face_by_face_landmark_x(target, source_lmk, dsize=512, scale=parameters['FaceExpressionCropScaleDecimalSliderBoth'], vy_ratio=parameters['FaceExpressionVYRatioDecimalSliderBoth'], interpolation=v2.InterpolationMode.BILINEAR)

        target_face_256 = t256_face(target_face_512)

        x_s_info = self.models_processor.lp_motion_extractor(target_face_256, 'Human-Face')
        x_c_s = x_s_info['kp']
        R_s = faceutil.get_rotation_matrix(x_s_info['pitch'], x_s_info['yaw'], x_s_info['roll'])
        f_s = self.models_processor.lp_appearance_feature_extractor(target_face_256, 'Human-Face')
        x_s = faceutil.transform_keypoint(x_s_info)

        # let lip-open scalar to be 0 at first
        if flag_normalize_lip and source_lmk is not None:
            c_d_lip_before_animation = [0.]
            combined_lip_ratio_tensor_before_animation = faceutil.calc_combined_lip_ratio(c_d_lip_before_animation, source_lmk, device=self.models_processor.device)
            if combined_lip_ratio_tensor_before_animation[0][0] >= lip_normalize_threshold:
                lip_delta_before_animation = self.models_processor.lp_retarget_lip(x_s, combined_lip_ratio_tensor_before_animation)
                
        # Compare eyes close ratio to the normalize slider
        if flag_normalize_eyes and source_lmk is not None:
            combined_eyes_ratio_tensor_before_animation = None
            c_d_eyes_normalize = c_d_eyes_lst
            eyes_ratio = np.array([c_d_eyes_normalize[0][0]], dtype=np.float32)
            eyes_ratio_normalize = max(eyes_ratio, 0.10) # Closed eye is 0.1, inferior to that the eyelids overlap
            eyes_ratio_l = min(c_d_eyes_normalize[0][0], eyes_normalize_max)
            eyes_ratio_r = min(c_d_eyes_normalize[0][1], eyes_normalize_max)
            eyes_ratio_max = np.array([[eyes_ratio_l, eyes_ratio_r]], dtype=np.float32)
            if eyes_ratio_normalize > eyes_normalize_threshold: # If the close ratio is bigger than the threshold
                combined_eyes_ratio_normalize = faceutil.calc_combined_eye_ratio_norm(eyes_ratio_max, source_lmk, device=self.models_processor.device)
            else:
                combined_eyes_ratio_normalize = faceutil.calc_combined_eye_ratio(eyes_ratio_max, source_lmk, device=self.models_processor.device)

        delta_new_eyes = x_s_info['exp'].clone()
        delta_new_lips = x_s_info['exp'].clone()
        
        #Eyes
        if flag_activate_eyes:
            for eyes_idx in [11, 13, 15, 16, 18]:
                if flag_relative_motion_eyes:
                    delta_new_eyes[:, eyes_idx, :] = (x_s_info['exp'] + (x_d_i_info['exp'] - 0))[:, eyes_idx, :]
                else:
                    delta_new_eyes[:, eyes_idx, :] = x_d_i_info['exp'][:, eyes_idx, :]

            scale_new_eyes = x_s_info['scale']
            R_new_eyes = R_s
            t_new_eyes = x_s_info['t']

            t_new_eyes[..., 2].fill_(0)  # zero tz
            x_d_i_new_eyes = scale_new_eyes * (x_c_s @ R_new_eyes + delta_new_eyes) + t_new_eyes
        
        if flag_activate_eyes and not flag_eye_retargeting:
            x_d_i_new_eyes = self.models_processor.lp_stitching(x_s, x_d_i_new_eyes, parameters["FaceEditorTypeSelection"])

        elif flag_activate_eyes and flag_eye_retargeting:
            eyes_delta = None
            if flag_eye_retargeting and source_lmk is not None:
                c_d_eyes_i = c_d_eyes_lst
                combined_eye_ratio_tensor = faceutil.calc_combined_eye_ratio(c_d_eyes_i, source_lmk, device=self.models_processor.device)
                if flag_normalize_eyes and combined_eyes_ratio_normalize is not None: # Use the normalized eye ratio
                    combined_eye_ratio_tensor = combined_eyes_ratio_normalize * eye_retargeting_multiplier
                    eyes_delta = self.models_processor.lp_retarget_eye(x_s, combined_eye_ratio_tensor, parameters["FaceEditorTypeSelection"])
                else:
                    combined_eye_ratio_tensor = combined_eye_ratio_tensor * eye_retargeting_multiplier
                # ∆_eyes,i = R_eyes(x_s; c_s,eyes, c_d,eyes,i)                
                    eyes_delta = self.models_processor.lp_retarget_eye(x_s, combined_eye_ratio_tensor, parameters["FaceEditorTypeSelection"])
            if flag_relative_motion_eyes:
                x_d_i_new_eyes = x_s + (eyes_delta if eyes_delta is not None else 0)
            else:
                x_d_i_new_eyes = x_d_i_new_eyes + (eyes_delta if eyes_delta is not None else 0)
            x_d_i_new_eyes = self.models_processor.lp_stitching(x_s, x_d_i_new_eyes, parameters["FaceEditorTypeSelection"])
        
        if flag_activate_eyes:
            x_d_i_new_eyes = (x_d_i_new_eyes - x_s) * driving_multiplier_eyes

        #Lips
        if flag_activate_lips:
            for lip_idx in [6, 12, 14, 17, 19, 20]:
                if flag_relative_motion_lips:
                    delta_new_lips[:, lip_idx, :] = (x_s_info['exp'] + (x_d_i_info['exp'] - torch.from_numpy(self.models_processor.lp_lip_array).to(dtype=torch.float32, device=self.models_processor.device)))[:, lip_idx, :]
                else:
                    delta_new_lips[:, lip_idx, :] = x_d_i_info['exp'][:, lip_idx, :]
                
            scale_new_lips = x_s_info['scale']
            R_new_lips = R_s
            t_new_lips = x_s_info['t']

            t_new_lips[..., 2].fill_(0)  # zero tz
            x_d_i_new_lips = scale_new_lips * (x_c_s @ R_new_lips + delta_new_lips) + t_new_lips
        
        if flag_activate_lips and not flag_lip_retargeting:
            # without retargeting
            if flag_normalize_lip and lip_delta_before_animation is not None:
                x_d_i_new_lips = self.models_processor.lp_stitching(x_s, x_d_i_new_lips, parameters["FaceEditorTypeSelection"]) + lip_delta_before_animation
            else:
                x_d_i_new_lips = self.models_processor.lp_stitching(x_s, x_d_i_new_lips, parameters["FaceEditorTypeSelection"])

        elif flag_activate_lips and flag_lip_retargeting:
            lip_delta = None
            if flag_lip_retargeting and source_lmk is not None:
                c_d_lip_i = c_d_lip_lst
                combined_lip_ratio_tensor = faceutil.calc_combined_lip_ratio(c_d_lip_i, source_lmk, device=self.models_processor.device)
                combined_lip_ratio_tensor = combined_lip_ratio_tensor * lip_retargeting_multiplier
                # ∆_lip,i = R_lip(x_s; c_s,lip, c_d,lip,i)
                lip_delta = self.models_processor.lp_retarget_lip(x_s, combined_lip_ratio_tensor, parameters["FaceEditorTypeSelection"])
            if flag_relative_motion_lips:
                x_d_i_new_lips = x_s + (lip_delta if lip_delta is not None else 0)
            else:
                x_d_i_new_lips = x_d_i_new_lips + (lip_delta if lip_delta is not None else 0)
            x_d_i_new_lips = self.models_processor.lp_stitching(x_s, x_d_i_new_lips, parameters["FaceEditorTypeSelection"])
        
        if flag_activate_lips:
            x_d_i_new_lips = (x_d_i_new_lips - x_s) * driving_multiplier_lips

        if flag_activate_lips and flag_activate_eyes:
            x_d_i_new = x_s + x_d_i_new_eyes + x_d_i_new_lips
        elif flag_activate_eyes and not flag_activate_lips:
            x_d_i_new = x_s + x_d_i_new_eyes
        elif not flag_activate_eyes and flag_activate_lips:
            x_d_i_new = x_s + x_d_i_new_lips
 
        out = self.models_processor.lp_warp_decode(f_s, x_s, x_d_i_new, parameters["FaceEditorTypeSelection"])
        out = torch.squeeze(out)
        out = torch.clamp(out, 0, 1)  # Clip i valori tra 0 e 1

        # Applica la maschera
        #out = torch.mul(out, self.models_processor.lp_mask_crop)  # Applica la maschera
        with self.lock:
            t = trans.SimilarityTransform()
            t.params[0:2] = M_c2o
            dsize = (target.shape[1], target.shape[2])
            # pad image by image size
            out = faceutil.pad_image_by_size(out, dsize)
            out = v2.functional.affine(out, t.rotation*57.2958, translate=(t.translation[0], t.translation[1]), scale=t.scale, shear=(0.0, 0.0), interpolation=v2.InterpolationMode.BILINEAR, center=(0, 0))
            out = v2.functional.crop(out, 0,0, dsize[0], dsize[1]) # cols, rows

        out = torch.clamp(torch.mul(out, 255.0), 0, 255).type(torch.float32)

        return out

    def swap_edit_face_core(self, img, kps, parameters, control, **kwargs): # img = RGB
        # Grab 512 face from image and create 256 and 128 copys
        if parameters['FaceEditorEnableToggle']:
            # Scaling Transforms
            #t256 = v2.Resize((256, 256), interpolation=v2.InterpolationMode.BILINEAR, antialias=False)

            # initial eye_ratio and lip_ratio values
            init_source_eye_ratio = 0.0
            init_source_lip_ratio = 0.0

            _, lmk_crop, _ = self.models_processor.run_detect_landmark( img, bbox=np.array([0, 0, 512, 512]), det_kpss=[], detect_mode='203', score=0.5, from_points=False)
            source_eye_ratio = faceutil.calc_eye_close_ratio(lmk_crop[None])
            source_lip_ratio = faceutil.calc_lip_close_ratio(lmk_crop[None])
            init_source_eye_ratio = round(float(source_eye_ratio.mean()), 2)
            init_source_lip_ratio = round(float(source_lip_ratio[0][0]), 2)

            # prepare_retargeting_image
            original_face_512, M_o2c, M_c2o = faceutil.warp_face_by_face_landmark_x(img, lmk_crop, dsize=512, scale=parameters["FaceEditorCropScaleDecimalSlider"], vy_ratio=parameters['FaceEditorVYRatioDecimalSlider'], interpolation=interpolation_expression_faceeditor_back)
            original_face_256 = t256_face(original_face_512)

            x_s_info = self.models_processor.lp_motion_extractor(original_face_256, parameters["FaceEditorTypeSelection"])
            x_d_info_user_pitch = x_s_info['pitch'] + parameters['HeadPitchSlider'] #input_head_pitch_variation
            x_d_info_user_yaw = x_s_info['yaw'] + parameters['HeadYawSlider'] # input_head_yaw_variation
            x_d_info_user_roll = x_s_info['roll'] + parameters['HeadRollSlider'] #input_head_roll_variation
            R_s_user = faceutil.get_rotation_matrix(x_s_info['pitch'], x_s_info['yaw'], x_s_info['roll'])
            R_d_user = faceutil.get_rotation_matrix(x_d_info_user_pitch, x_d_info_user_yaw, x_d_info_user_roll)
            f_s_user = self.models_processor.lp_appearance_feature_extractor(original_face_256, parameters["FaceEditorTypeSelection"])
            x_s_user = faceutil.transform_keypoint(x_s_info)

            #execute_image_retargeting
            mov_x = torch.tensor(parameters['XAxisMovementDecimalSlider']).to(self.models_processor.device)
            mov_y = torch.tensor(parameters['YAxisMovementDecimalSlider']).to(self.models_processor.device)
            mov_z = torch.tensor(parameters['ZAxisMovementDecimalSlider']).to(self.models_processor.device)
            eyeball_direction_x = torch.tensor(parameters['EyeGazeHorizontalDecimalSlider']).to(self.models_processor.device)
            eyeball_direction_y = torch.tensor(parameters['EyeGazeVerticalDecimalSlider']).to(self.models_processor.device)
            smile = torch.tensor(parameters['MouthSmileDecimalSlider']).to(self.models_processor.device)
            wink = torch.tensor(parameters['EyeWinkDecimalSlider']).to(self.models_processor.device)
            eyebrow = torch.tensor(parameters['EyeBrowsDirectionDecimalSlider']).to(self.models_processor.device)
            lip_variation_zero = torch.tensor(parameters['MouthPoutingDecimalSlider']).to(self.models_processor.device)
            lip_variation_one = torch.tensor(parameters['MouthPursingDecimalSlider']).to(self.models_processor.device)
            lip_variation_two = torch.tensor(parameters['MouthGrinDecimalSlider']).to(self.models_processor.device)
            lip_variation_three = torch.tensor(parameters['LipsCloseOpenSlider']).to(self.models_processor.device)

            x_c_s = x_s_info['kp']
            delta_new = x_s_info['exp']
            scale_new = x_s_info['scale']
            t_new = x_s_info['t']
            R_d_new = (R_d_user @ R_s_user.permute(0, 2, 1)) @ R_s_user

            if eyeball_direction_x != 0 or eyeball_direction_y != 0:
                delta_new = faceutil.update_delta_new_eyeball_direction(eyeball_direction_x, eyeball_direction_y, delta_new)
            if smile != 0:
                delta_new = faceutil.update_delta_new_smile(smile, delta_new)
            if wink != 0:
                delta_new = faceutil.update_delta_new_wink(wink, delta_new)
            if eyebrow != 0:
                delta_new = faceutil.update_delta_new_eyebrow(eyebrow, delta_new)
            if lip_variation_zero != 0:
                delta_new = faceutil.update_delta_new_lip_variation_zero(lip_variation_zero, delta_new)
            if lip_variation_one !=  0:
                delta_new = faceutil.update_delta_new_lip_variation_one(lip_variation_one, delta_new)
            if lip_variation_two != 0:
                delta_new = faceutil.update_delta_new_lip_variation_two(lip_variation_two, delta_new)
            if lip_variation_three != 0:
                delta_new = faceutil.update_delta_new_lip_variation_three(lip_variation_three, delta_new)
            if mov_x != 0:
                delta_new = faceutil.update_delta_new_mov_x(-mov_x, delta_new)
            if mov_y !=0 :
                delta_new = faceutil.update_delta_new_mov_y(mov_y, delta_new)

            x_d_new = mov_z * scale_new * (x_c_s @ R_d_new + delta_new) + t_new
            eyes_delta, lip_delta = None, None

            input_eye_ratio = max(min(init_source_eye_ratio + parameters['EyesOpenRatioDecimalSlider'], 0.80), 0.00)
            if input_eye_ratio != init_source_eye_ratio:
                combined_eye_ratio_tensor = faceutil.calc_combined_eye_ratio([[float(input_eye_ratio)]], lmk_crop, device=self.models_processor.device)
                eyes_delta = self.models_processor.lp_retarget_eye(x_s_user, combined_eye_ratio_tensor, parameters["FaceEditorTypeSelection"])

            input_lip_ratio = max(min(init_source_lip_ratio + parameters['LipsOpenRatioDecimalSlider'], 0.80), 0.00)
            if input_lip_ratio != init_source_lip_ratio:
                combined_lip_ratio_tensor = faceutil.calc_combined_lip_ratio([[float(input_lip_ratio)]], lmk_crop, device=self.models_processor.device)
                lip_delta = self.models_processor.lp_retarget_lip(x_s_user, combined_lip_ratio_tensor, parameters["FaceEditorTypeSelection"])

            x_d_new = x_d_new + \
                    (eyes_delta if eyes_delta is not None else 0) + \
                    (lip_delta if lip_delta is not None else 0)

            flag_stitching_retargeting_input: bool = kwargs.get('flag_stitching_retargeting_input', True)
            if flag_stitching_retargeting_input:
                x_d_new = self.models_processor.lp_stitching(x_s_user, x_d_new, parameters["FaceEditorTypeSelection"])

            out = self.models_processor.lp_warp_decode(f_s_user, x_s_user, x_d_new, parameters["FaceEditorTypeSelection"])
            out = torch.squeeze(out)
            out = torch.clamp(out, 0, 1)  # clip to 0~1

            #flag_do_crop_input_retargeting_image = kwargs.get('flag_do_crop_input_retargeting_image', False)
            #if flag_do_crop_input_retargeting_image:
            #    gauss = transforms.GaussianBlur(parameters['FaceEditorBlurAmountSlider']*2+1, (parameters['FaceEditorBlurAmountSlider']+1)*0.2)
            #    mask_crop = gauss(self.models_processor.lp_mask_crop)
            #    img = faceutil.paste_back_adv(out, M_c2o, img, mask_crop)
            #else:
            t = trans.SimilarityTransform()
            t.params[0:2] = M_c2o
            dsize = (img.shape[1], img.shape[2])
            # pad image by image size
            out = faceutil.pad_image_by_size(out, dsize)
            out = v2.functional.affine(out, t.rotation*57.2958, translate=(t.translation[0], t.translation[1]), scale=t.scale, shear=(0.0, 0.0), interpolation=interpolation_expression_faceeditor_back, center=(0, 0))
            out = v2.functional.crop(out, 0,0, dsize[0], dsize[1]) # cols, rows

            img = out                
            img = torch.mul(img, 255.0)
            img = torch.clamp(img, 0, 255).type(torch.float32)
                                                          
        return img
        
    def swap_edit_face_core_makeup(self, img, kps, parameters, control, **kwargs): # img = RGB
        if parameters['FaceMakeupEnableToggle'] or parameters['HairMakeupEnableToggle'] or parameters['EyeBrowsMakeupEnableToggle'] or parameters['LipsMakeupEnableToggle']: #or parameters['EyesMakeupEnableToggle']:
        
            _, lmk_crop, _ = self.models_processor.run_detect_landmark( img, bbox=[], det_kpss=kps, detect_mode='203', score=0.5, from_points=True)

            # prepare_retargeting_image
            original_face_512, M_o2c, M_c2o = faceutil.warp_face_by_face_landmark_x(img, lmk_crop, dsize=512, scale=parameters['FaceEditorCropScaleDecimalSlider'], vy_ratio=parameters['FaceEditorVYRatioDecimalSlider'], interpolation=interpolation_expression_faceeditor_back)

            out, mask_out = self.models_processor.apply_face_makeup(original_face_512, parameters)
            if 1:
                gauss = transforms.GaussianBlur(5*2+1, (5+1)*0.2)
                out = torch.clamp(torch.div(out, 255.0), 0, 1).type(torch.float32)
                mask_crop = gauss(self.models_processor.lp_mask_crop)
                img = faceutil.paste_back_adv(out, M_c2o, img, mask_crop)

        return img

    @torch.no_grad()
    def gradient_magnitude(self,
                           image: torch.Tensor,
                           mask: torch.Tensor,
                           kernel_size: int,
                           weighting_strength: float,
                           sigma: float,
                           lambd: float,
                           gamma: float,
                           psi: float,
                           theta_count: int,
                           #hoch: float,
                           # CLAHE-Params
                           clip_limit: float,
                           alpha_clahe: float,
                           grid_size: tuple[int,int],
                           # Flags
                           global_gamma: float,
                           global_contrast: float,
                          ) -> torch.Tensor:
        """
        image: Tensor [C, H, W] in [0..255]
        mask:  Tensor [C, H, W] (0/1)
        Returns: Tensor [C, H, W] – gewichtete Gabor-Magnitude
        """

        C, H, W = image.shape
        eps = 1e-6
        image = image.float() / 255.0
        mask = mask.bool()
        
        # 1) Global Gamma & Kontrast
        if global_gamma != 1.0:
            image = image.pow(global_gamma)
        if global_contrast != 1.0:
            m_gc = image.mean((1,2), keepdim=True)
            image = (image - m_gc) * global_contrast + m_gc

        # 2) CLAHE im L-Kanal (mit alpha_clahe-Blending)
        if clip_limit > 0.0:
            image = image.unsqueeze(0).clamp(0,1)     # [1,3,H,W]
            mask_b3 = mask.unsqueeze(0)               # [1,3,H,W]

            lab = kc.rgb_to_lab(image)                       # [1,3,H,W]
            L   = lab[:,0:1,:,:] / 100.0                     # [1,1,H,W]

            mb = mask_b3[:,0:1,:,:]                          # [1,1,H,W]
            area_l   = mb.sum((2,3), keepdim=True).clamp(min=1)
            mean_l = (L * mb).sum((2,3), keepdim=True) / area_l
            Lf     = torch.where(mb, L, mean_l)
            Leq    = ke.equalize_clahe(
                        Lf, clip_limit=clip_limit, grid_size=grid_size,
                        slow_and_differentiable=False
                    ).clamp(0,1)
            L_blend = alpha_clahe * Leq + (1-alpha_clahe) * L
            Lnew    = torch.where(mb, L_blend, L)

            lab_eq = torch.cat([Lnew*100.0, lab[:,1:,:,:]], dim=1)  # [1,3,H,W]
            x_eq   = kc.lab_to_rgb(lab_eq)#.clamp(0,1)
            image  = x_eq.squeeze(0)

        # 3) Gabor-Filter setup
        kernel_size = max(1, 2*kernel_size - 1) #23
        if theta_count == 10:
            theta_values = torch.tensor([math.pi/4], device=image.device)
        else:
            theta_values = torch.linspace(0, math.pi, theta_count+1, device=image.device)[:-1] #torch.arange(8, device=image.device) * (math.pi/8) #
            #print("theta_values: ", theta_values)
        # 4) Einziger Gabor-Filter-Aufruf
        magnitude = self.apply_gabor_filter_torch(
            image, kernel_size,
            sigma, lambd,
            gamma, psi,
            theta_values
        )  # [C, H, W]

        # 5) Invertieren
        max_mv = magnitude.amax((1,2), keepdim=True)
        inverted = max_mv - magnitude                       # [C, H, W]

        # 6) Gewichtung
        if weighting_strength > 0:
            img_m = (image * mask)
            weighted = inverted * ((1-weighting_strength)
                                   + weighting_strength * img_m)
        else:
            weighted = inverted

        return weighted * 255  # [C, H, W]
        
    def apply_gabor_filter_torch(self, image, kernel_size, sigma, lambd, gamma, psi, theta_values):
        """
        image: Tensor [C, H, W]
        theta_values: Tensor [N]
        Rückgabe: Tensor [C, H, W]
        """
        C, H, W = image.shape
        image = image.unsqueeze(0)  # → [1, C, H, W]
        
        N = theta_values.shape[0]
        
        kernels = self.get_gabor_kernels(kernel_size, sigma, lambd, gamma, psi, theta_values, image.device)  # [N, 1, k, k]

        #responses = []
        
        # kernels: [N, 1, k, k]
        # erweitere auf alle Channels:
        weight = kernels.repeat_interleave(C, dim=0)       # → [N*C, 1, k, k]
        out = F.conv2d(
            image,              # [1, C, H, W]
            weight, 
            padding=kernel_size//2,
            groups=C                         # jede Channel-Gruppe bekommt N Filter
        )  # out: [1, N*C, H, W]
        # umformen in [N, C, H, W]:
        out = out.squeeze(0).view(N, C, H, W)
        magnitudes = out.amax(dim=0)   # oder .mean(dim=0)
        return magnitudes

    def get_gabor_kernels(self, kernel_size, sigma, lambd, gamma, psi, theta_values, device):
        """
        Rückgabe: Tensor [N, 1, k, k]
        """
        half = kernel_size // 2
        y, x = torch.meshgrid(
            torch.linspace(-half, half, kernel_size, device=device),
            torch.linspace(-half, half, kernel_size, device=device),
            indexing='ij'
        )

        kernels = []
        for theta in theta_values:
            x_theta = x * torch.cos(theta) + y * torch.sin(theta)
            y_theta = -x * torch.sin(theta) + y * torch.cos(theta)

            gb = torch.exp(-0.5 * (x_theta**2 + (gamma**2) * y_theta**2) / sigma**2)
            gb *= torch.cos(2 * math.pi * x_theta / lambd + psi)
            kernels.append(gb)

        return torch.stack(kernels).unsqueeze(1)  # → [N, 1, k, k]

    def face_restorer_auto(
        self,
        original_face_512,           # [3,H,W], float in [0..255]
        swap_original,               # [3,H,W]
        swap,                        # [3,H,W]
        alpha,                       # initial scalar alpha (ignored; we binary search below)
        adjust_sharpness,
        scale_factor,
        debug,
        swap_mask,
        alpha_map_enable: bool = False,          # <--- NEW: toggle
        alpha_map_strength: float = 0.5,         # 0..1: how much to deviate around prev_alpha
        alpha_map_blur: int = 7                  # odd; smoothness of the alpha map
        ):
        
        # Copies (like before)
        original_face_512_autorestore = original_face_512.clone().float()
                                                                                 
                                                          

        swap_restorecalc = swap.clone()
                                                               
                                

        swap_original_autorestore = swap_original.clone()
                                                                                 
                                                  

        # Baseline sharpness of original
        scores_original = self.sharpness_score(original_face_512)
        score_new_original = scores_original["combined"].item() * 100 + adjust_sharpness / 10.0

        # Binary search for scalar alpha (your existing loop)
        alpha = 1.0
        max_iterations = 7
        alpha_min, alpha_max = 0.0, 1.0
        tolerance = 0.5
        min_alpha_change = 0.05
        iteration = 0
        prev_alpha = alpha
        iteration_blur = 0

        while iteration < max_iterations:
            swap2 = swap * alpha + swap_original * (1 - alpha)
                                      
            swap2_masked = swap2.clone()  # you can multiply by mask if you want only-face analysis

            scores_swap = self.sharpness_score(swap2_masked)
            score_new_swap = scores_swap["combined"].item() * 100
            sharpness_diff = score_new_swap - score_new_original
            #if debug:
            #    print(prev_alpha * 100, sharpness_diff, score_new_swap, score_new_original)

            if abs(sharpness_diff) < tolerance:
                break

            if sharpness_diff < 0:
                if alpha > 0.99:
                    prev_alpha = alpha
                    break
                alpha_min = alpha
                alpha = (alpha + alpha_max) / 2.0
            else:
                alpha_max = alpha
                alpha = (alpha + alpha_min) / 2.0

            # Very small alpha → blur fallback on base (your original logic)
            if sharpness_diff >= 0 and alpha < 0.07:
                prev_alpha = 0.0
                base = swap_original
                max_blur_strength = 10
                for bs in range(0, max_blur_strength + 1):
                    if bs == 0:
                        kernel_size = 1
                        sigma = 1e-6
                    else:
                        kernel_size = 2 * bs + 1
                        sigma = max(bs, 1e-6)
                    gaussian_blur = transforms.GaussianBlur(kernel_size, sigma)
                    swap2_blurred = gaussian_blur(base)
                    scores_swap_b = self.sharpness_score(swap2_blurred)
                    score_new_swap_b = scores_swap_b["combined"].item() * 100.0
                    sharpness_diff_b = score_new_swap_b - score_new_original
                    #if debug:
                    #    print(bs, sharpness_diff_b, scores_swap_b, score_new_original)

                    if sharpness_diff_b < 0:
                        iteration_blur = 0 if bs == 0 else (bs - 1)
                        break
                    if abs(sharpness_diff_b) <= tolerance:
                        iteration_blur = bs
                        break
                    iteration_blur = bs
                break

            if abs(prev_alpha - alpha) < min_alpha_change:
                prev_alpha = (prev_alpha + alpha) / 2.0
                #if debug:
                #    print("< min_alpha_change", prev_alpha)
                if abs(prev_alpha) <= 0.05:
                    prev_alpha = 0.0
                    #if debug:
                    #    print("prev_alpha very small → 0")
                break

            prev_alpha = alpha
            iteration += 1

        # -------- NEW: Per-pixel alpha map, derived from sharpness distribution --------
        # Only if enabled AND we found a positive scalar alpha.
        if alpha_map_enable and (prev_alpha > 0.0):
            # Build the *final* composite (for a stable map), then sharpness map of it:
            swap_final = swap * prev_alpha + swap_original * (1 - prev_alpha)

            # Per-pixel sharpness [H,W] in [0..1], smoothed:
            s_map = self.sharpness_map(
                swap_final,
                mask=swap_mask,                 # restrict stats/normalization to face
                tenengrad_thresh=0.05,
                comb_weight=0.5,
                smooth_kernel=alpha_map_blur if (alpha_map_blur and alpha_map_blur % 2 == 1) else 0
            )  # [H,W] in [0..1]

            # Mean sharpness inside mask (or global)
            if swap_mask is not None:
                m = (swap_mask if swap_mask.dim() == 2 else swap_mask.squeeze(0)).float().to(s_map.device)
                denom = m.sum().clamp_min(1.0)
                mu = (s_map * m).sum() / denom
            else:
                mu = s_map.mean()

            # Deviation map around mean, scale around prev_alpha
            # alpha_map_strength in [0..1] controls deviation amount
            dev = (s_map - mu).clamp(-1.0, 1.0)                     # [-1..1]-ish
            alpha_map = prev_alpha * (1.0 + alpha_map_strength * dev)  # raise in sharper, lower in blurrier
            alpha_map = alpha_map.clamp(0.0, 1.0)                    # [H,W]

            # Keep outside-face area at scalar prev_alpha (if a mask is provided)
            if swap_mask is not None:
                m = (swap_mask if swap_mask.dim() == 2 else swap_mask.squeeze(0)).float().to(alpha_map.device)
                alpha_map = alpha_map * m + prev_alpha * (1.0 - m)

            # Return as [1,H,W] to broadcast with [3,H,W] later
            return alpha_map.unsqueeze(0), iteration_blur

        # Fallback: scalar like before
        return prev_alpha, iteration_blur
              
    def sharpness_score(
        self,
        image: torch.Tensor,
        mask: torch.Tensor = None,
        tenengrad_thresh: float = 0.05,
        comb_weight: float = 0.5
    ) -> dict:
        """
        Berechnet drei Sharpness‐Metriken auf einem RGB-Image:
          1) var_lap: Variance of Laplacian
          2) tten: Thresholded Tenengrad (Anteil starker Kanten)
          3) combined: comb_weight*var_lap + (1-comb_weight)*tten

        Args:
            image: Tensor [3, H, W], float in [0..1]
            mask:  optional Tensor [H, W] oder [1, H, W] mit 1=gültig, 0=ignorieren
            tenengrad_thresh: Schwellwert für Tenengrad (0..1)
            comb_weight: Gewicht für var_lap in der Kombi (0..1)

        Returns:
            {
              "var_lap": float Tensor,
              "ttengrad": float Tensor,
              "combined": float Tensor
            }
        """
        image = image / 255.0
        
        # 1) Graustufen [1,1,H,W]
        gray = image.mean(dim=0, keepdim=True).unsqueeze(0)

        # 2) Optional Mask auf [H,W]
        if mask is not None:
            m = mask.float()
            if m.dim() == 3:  # [1,H,W]
                m = m.squeeze(0)
        else:
            m = None
            #print("no mask")

        # Hilfs: Anzahl gültiger Pixel
        def valid_count(t):
            return m.sum().clamp(min=1.0) if m is not None else t.numel()

        # --- Variance of Laplacian ---
        lap = torch.tensor([[0,1,0],[1,-4,1],[0,1,0]],
                           device=image.device, dtype=torch.float32).view(1,1,3,3)
        L = F.conv2d(gray, lap, padding=1).squeeze()   # [H,W]
        L2 = L.pow(2)
        # Mask anwenden
        if m is not None:
            L  = L * m
            L2 = L2 * m
        cnt = valid_count(L2)
        mean_L2 = L2.sum() / cnt
        mean_L  = L.sum()  / cnt
        var_lap = (mean_L2 - mean_L.pow(2)).clamp(min=0.0)

        # --- Thresholded Tenengrad ---
        sobel_x = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]],
                               device=image.device, dtype=torch.float32).view(1,1,3,3)
        sobel_y = sobel_x.transpose(2,3)
        Gx = F.conv2d(gray, sobel_x, padding=1).squeeze()  # [H,W]
        Gy = F.conv2d(gray, sobel_y, padding=1).squeeze()
        G  = (Gx.pow(2) + Gy.pow(2)).sqrt()
        if m is not None:
            G = G * m
        total = cnt
        strong = (G > tenengrad_thresh).float().sum()
        ttengrad = strong / total

        # --- Kombinierter Score ---
        combined = comb_weight * var_lap + (1 - comb_weight) * ttengrad

        return {
            "var_lap":    var_lap,
            "ttengrad":   ttengrad,
            "combined":   combined
        }        

    def sharpness_map(
        self,
        image: torch.Tensor,            # [3,H,W], float in [0..255]
        mask: torch.Tensor | None = None,
        tenengrad_thresh: float = 0.05,
        comb_weight: float = 0.5,
        smooth_kernel: int = 5          # odd; 0/1 = no blur
    ) -> torch.Tensor:
        """
        Returns a normalized per-pixel sharpness map in [0..1] with shape [H,W].
        Combines Laplacian energy + gradient magnitude (Tenengrad-like).
        """
        eps = 1e-8
        device = image.device

        # [3,H,W] -> [1,1,H,W] gray, range [0..1]
        gray = (image / 255.0).mean(dim=0, keepdim=True).unsqueeze(0)

        # Convs
        lap_k = torch.tensor([[0, 1, 0],
                              [1,-4, 1],
                              [0, 1, 0]], device=device, dtype=torch.float32).view(1,1,3,3)
        sobel_x = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]],
                               device=device, dtype=torch.float32).view(1,1,3,3)
        sobel_y = sobel_x.transpose(2,3)

        lap = F.conv2d(gray, lap_k, padding=1).squeeze(0).squeeze(0)               # [H,W]
        gx  = F.conv2d(gray, sobel_x, padding=1).squeeze(0).squeeze(0)             # [H,W]
        gy  = F.conv2d(gray, sobel_y, padding=1).squeeze(0).squeeze(0)
        grad = (gx.pow(2) + gy.pow(2)).sqrt()                                      # [H,W]

        # Robust normalization via percentiles inside mask (if given)
        def robust_norm(x, msk):
            if msk is not None:
                sel = x[msk > 0]
                if sel.numel() < 16:  # fallback if mask tiny
                    sel = x.reshape(-1)
            else:
                sel = x.reshape(-1)
            p5  = torch.quantile(sel, 0.05) if sel.numel() > 0 else torch.tensor(0.0, device=device)
            p95 = torch.quantile(sel, 0.95) if sel.numel() > 0 else torch.tensor(1.0, device=device)
            y = (x - p5) / (p95 - p5 + eps)
            return y.clamp_(0, 1)

        m = None
        if mask is not None:
            m = (mask if mask.dim() == 2 else mask.squeeze(0)).float().to(device)

        lap_n  = robust_norm(lap.abs(), m)
        grad_n = robust_norm(grad, m)

        smap = comb_weight * lap_n + (1.0 - comb_weight) * grad_n                   # [H,W]

        # Optional smoothing to avoid noisy alpha
        if smooth_kernel and smooth_kernel >= 3 and smooth_kernel % 2 == 1:
            k = smooth_kernel
            # Gaussian blur with torchvision
            smap3 = smap.unsqueeze(0).unsqueeze(0)                                  # [1,1,H,W]
            gb = transforms.GaussianBlur(kernel_size=k, sigma=max(1, k//2))
            smap = gb(smap3).squeeze(0).squeeze(0)

        return smap.clamp(0, 1)
    @torch.no_grad()
    def apply_block_shift_gpu_jitter(
        self,
        img: torch.Tensor, 
        block_size: int,
        max_amount_pixels: float,
        *,
        seed: int = 1337,
        pad_mode: str = "replicate",
        align_corners: bool = True
    ) -> torch.Tensor:
        """
        MPEG-ähnlicher Block-Jitter: verschiebt jedes BxB-Block-Feld um einen
        deterministischen (bx, by)-abhängigen Offset in Pixeln.

        Args:
            img: Tensor [C, H, W] (BGR/RGB egal). CPU oder CUDA. dtype float/uint8 egal.
            block_size: Blockgröße B (z. B. 8).
            max_amount_pixels: max. |Offset| in Pixeln (wird auf beide Achsen angewendet).
            seed: globaler Seed für deterministische Offsets (frame-stabil).
            pad_mode: Padding-Modus für Rand (replicate|reflect|zeros).
            align_corners: wie in grid_sample (True ist für pixelgenaue Shifts meist stabiler).

        Returns:
            Tensor [C, H, W] – gleiche Device/Dtype wie Eingang.
        """
        seed = seed + self.frame_number*17
        assert img.ndim == 3, "expected [C,H,W]"
        C, H, W = img.shape
        device  = img.device
        dtype   = img.dtype

        # ggf. auf float32 für grid_sample rechnen (am Ende casten wir zurück)
        work = img if img.dtype in (torch.float32, torch.float16, torch.bfloat16) else img.float()

        # Auf vielfache von B padden (unten/rechts), danach wieder croppen
        B = int(2 ** block_size)
        H_pad = (B - (H % B)) % B
        W_pad = (B - (W % B)) % B
        if H_pad or W_pad:
            pad = (0, W_pad, 0, H_pad)  # (left, right, top, bottom)
            mode = {"replicate":"replicate", "reflect":"reflect", "zeros":"constant"}[pad_mode]
            work = F.pad(work[None], pad=pad, mode=mode).squeeze(0)
        Hp, Wp = work.shape[-2:]

        # Anzahl Blöcke
        nby = Hp // B
        nbx = Wp // B

        # --- deterministische Offsets pro Block im Bereich [-max, +max] ---
        # Baue Block-Koordinatenfelder
        by_grid, bx_grid = torch.meshgrid(
            torch.arange(nby, device=device, dtype=torch.float32),
            torch.arange(nbx, device=device, dtype=torch.float32),
            indexing="ij"
        )
        # einfacher Hash -> [0,1)
        # (sin-Hash: frame-stabil, abhängig nur von (bx,by) und seed)
        h = torch.sin( (bx_grid*12.9898 + by_grid*78.233 + float(seed)) * 43758.5453 )
        frac = torch.frac(h * 0.5 + 0.5)   # in [0,1)

        # zwei unabhängige Offsets aus dem Hash ableiten
        # dx_base, dy_base in [-1,1] -> * max_amount_pixels
        max_amount_pixels = max_amount_pixels/4
        dx_base = ( (frac) * 2.0 - 1.0 ) * float(max_amount_pixels)
        # zweite "Quelle": einfach andere lineare Kombi
        h2 = torch.sin( (bx_grid*96.233 + by_grid*15.987 + (float(seed)+101)) * 12345.6789 )
        frac2 = torch.frac(h2 * 0.5 + 0.5)
        dy_base = ( (frac2) * 2.0 - 1.0 ) * float(max_amount_pixels)

        # auf Pixelraster upsamplen, indem wir jeden Block-Offset auf BxB "kacheln"
        dx = torch.repeat_interleave(torch.repeat_interleave(dx_base, B, dim=0), B, dim=1)  # [Hp,Wp]
        dy = torch.repeat_interleave(torch.repeat_interleave(dy_base, B, dim=0), B, dim=1)  # [Hp,Wp]

        # --- Flow-Field für grid_sample bauen ---
        # grid_sample nimmt Normalized-Koordinaten in [-1,1]
        # Displacement in "normalized" Einheiten: dx_norm = 2*dx/(Wp-1), dy_norm = 2*dy/(Hp-1)
        # (x ~ width, y ~ height)
        xs = torch.linspace(-1.0, 1.0, Wp, device=device)
        ys = torch.linspace(-1.0, 1.0, Hp, device=device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")        # [Hp,Wp]
        dx_norm = (2.0 * dx) / max(Wp - 1, 1)
        dy_norm = (2.0 * dy) / max(Hp - 1, 1)

        flow_x = grid_x + dx_norm
        flow_y = grid_y + dy_norm
        flow = torch.stack([flow_x, flow_y], dim=-1)                   # [Hp,Wp,2]

        # auf [1,C,Hp,Wp] bringen
        warped = F.grid_sample(
            work[None], 
            flow[None],
            mode="bilinear",
            padding_mode="border",   # keine schwarzen Kanten
            align_corners=align_corners
        ).squeeze(0)

        # auf Originalgröße zurückcroppen, wenn gepaddet
        if H_pad or W_pad:
            warped = warped[..., :H, :W]

        # dtype zurück
        if warped.dtype != dtype:
            warped = warped.to(dtype)

        return warped
        
    def analyze_image(self, image):
        image = image.float() /255.0
        C, H, W = image.shape
        grayscale = torch.mean(image, dim=0, keepdim=True)
        analysis = {}
        fft = torch.fft.fft2(grayscale)
        high_freq_energy = torch.mean(torch.abs(fft))
        analysis["jpeg_artifacts"] = min(high_freq_energy.item() / 50, 1.0)
        median_filtered = F.avg_pool2d(grayscale, 3, stride=1, padding=1)
        noise_map = torch.abs(grayscale - median_filtered)
        sp_noise = torch.mean((noise_map > 0.1).float())
        analysis["salt_pepper_noise"] = min(sp_noise.item() * 10, 1.0)
        local_var = F.avg_pool2d(grayscale**2, 5, stride=1, padding=2) - (F.avg_pool2d(grayscale, 5, stride=1, padding=2) ** 2)
        speckle_noise = torch.mean(local_var)
        analysis["speckle_noise"] = min(speckle_noise.item() * 50, 1.0)
        laplace_kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32, device=image.device).unsqueeze(0).unsqueeze(0)
        laplace_edges = F.conv2d(grayscale.unsqueeze(0), laplace_kernel, padding=1)
        edge_strength = torch.mean(torch.abs(laplace_edges))
        analysis["blur"] = 1.0 - min(edge_strength.item() * 5, 1.0)
        contrast = grayscale.std()
        analysis["low_contrast"] = 1.0 - min(contrast.item() * 10, 1.0)
        return analysis
