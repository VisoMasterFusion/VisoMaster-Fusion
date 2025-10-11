import pickle
from typing import TYPE_CHECKING
import platform

import torch
import numpy as np
from torch.cuda import nvtx

import torch.nn.functional
from collections import defaultdict

from torchvision import transforms
from torchvision.transforms import v2

from app.processors.models_data import models_dir
from app.processors.utils import faceutil
if TYPE_CHECKING:
    from app.processors.models_processor import ModelsProcessor
    
SYSTEM_PLATFORM = platform.system()

class FaceEditors:
    def __init__(self, models_processor: 'ModelsProcessor'):
        self.models_processor = models_processor
        self.lp_mask_crop = faceutil.create_faded_inner_mask(size=(512, 512), border_thickness=5, fade_thickness=15, blur_radius=5, device=self.models_processor.device)
        self.lp_mask_crop = torch.unsqueeze(self.lp_mask_crop, 0)
        try:
            self.lp_lip_array = np.array(self.load_lip_array())
        except FileNotFoundError:
            self.lp_lip_array = None
    def load_lip_array(self):
        with open(f'{models_dir}/liveportrait_onnx/lip_array.pkl', 'rb') as f:
            return pickle.load(f)
        
    def lp_motion_extractor(self, img, face_editor_type='Human-Face', **kwargs) -> dict:
        kp_info = {}
        with torch.no_grad():
            # prepare_source
            I_s = torch.div(img.type(torch.float32), 255.)
            I_s = torch.clamp(I_s, 0, 1)  # clamp to 0~1
            I_s = torch.unsqueeze(I_s, 0).contiguous()

            if self.models_processor.provider_name in ["TensorRT", "TensorRT-Engine"]:
                if face_editor_type == 'Human-Face':
                    if not self.models_processor.models_trt['LivePortraitMotionExtractor']:
                        self.models_processor.models_trt['LivePortraitMotionExtractor'] = self.models_processor.load_model_trt('LivePortraitMotionExtractor', custom_plugin_path=None, precision="fp32")

                motion_extractor_model = self.models_processor.models_trt['LivePortraitMotionExtractor']

                nvtx.range_push("forward")

                feed_dict = {"img": I_s}
                current_stream = torch.cuda.current_stream()
                preds_dict = motion_extractor_model.predict_async(feed_dict, current_stream)
                #current_stream.synchronize()
                
                kp_info = {
                    'pitch': preds_dict["pitch"],
                    'yaw': preds_dict["yaw"],
                    'roll': preds_dict["roll"],
                    't': preds_dict["t"],
                    'exp': preds_dict["exp"],
                    'scale': preds_dict["scale"],
                    'kp': preds_dict["kp"]
                }

                nvtx.range_pop()

            else:
                if face_editor_type == 'Human-Face':
                    if not self.models_processor.models['LivePortraitMotionExtractor']:
                        self.models_processor.models['LivePortraitMotionExtractor'] = self.models_processor.load_model('LivePortraitMotionExtractor')

                motion_extractor_model = self.models_processor.models['LivePortraitMotionExtractor']

                pitch = torch.empty((1,66), dtype=torch.float32, device=self.models_processor.device).contiguous()
                yaw = torch.empty((1,66), dtype=torch.float32, device=self.models_processor.device).contiguous()
                roll = torch.empty((1,66), dtype=torch.float32, device=self.models_processor.device).contiguous()
                t = torch.empty((1,3), dtype=torch.float32, device=self.models_processor.device).contiguous()
                exp = torch.empty((1,63), dtype=torch.float32, device=self.models_processor.device).contiguous()
                scale = torch.empty((1,1), dtype=torch.float32, device=self.models_processor.device).contiguous()
                kp = torch.empty((1,63), dtype=torch.float32, device=self.models_processor.device).contiguous()

                io_binding = motion_extractor_model.io_binding()
                io_binding.bind_input(name='img', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=I_s.size(), buffer_ptr=I_s.data_ptr())
                io_binding.bind_output(name='pitch', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=pitch.size(), buffer_ptr=pitch.data_ptr())
                io_binding.bind_output(name='yaw', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=yaw.size(), buffer_ptr=yaw.data_ptr())
                io_binding.bind_output(name='roll', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=roll.size(), buffer_ptr=roll.data_ptr())
                io_binding.bind_output(name='t', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=t.size(), buffer_ptr=t.data_ptr())
                io_binding.bind_output(name='exp', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=exp.size(), buffer_ptr=exp.data_ptr())
                io_binding.bind_output(name='scale', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=scale.size(), buffer_ptr=scale.data_ptr())
                io_binding.bind_output(name='kp', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=kp.size(), buffer_ptr=kp.data_ptr())

                if self.models_processor.device == "cuda":
                    torch.cuda.synchronize()
                elif self.models_processor.device != "cpu":
                    self.models_processor.syncvec.cpu()
                motion_extractor_model.run_with_iobinding(io_binding)

                kp_info = {
                    'pitch': pitch,
                    'yaw': yaw,
                    'roll': roll,
                    't': t,
                    'exp': exp,
                    'scale': scale,
                    'kp': kp
                }

            flag_refine_info: bool = kwargs.get('flag_refine_info', True)
            if flag_refine_info:
                bs = kp_info['kp'].shape[0]
                kp_info['pitch'] = faceutil.headpose_pred_to_degree(kp_info['pitch'])[:, None]  # Bx1
                kp_info['yaw'] = faceutil.headpose_pred_to_degree(kp_info['yaw'])[:, None]  # Bx1
                kp_info['roll'] = faceutil.headpose_pred_to_degree(kp_info['roll'])[:, None]  # Bx1
                kp_info['kp'] = kp_info['kp'].reshape(bs, -1, 3)  # BxNx3
                kp_info['exp'] = kp_info['exp'].reshape(bs, -1, 3)  # BxNx3

        return kp_info

    def lp_appearance_feature_extractor(self, img, face_editor_type='Human-Face'):
        with torch.no_grad():
            # prepare_source
            I_s = torch.div(img.type(torch.float32), 255.)
            I_s = torch.clamp(I_s, 0, 1)  # clamp to 0~1
            I_s = torch.unsqueeze(I_s, 0).contiguous()
            
            if self.models_processor.provider_name in ["TensorRT", "TensorRT-Engine"]:
                if face_editor_type == 'Human-Face':
                    if not self.models_processor.models_trt['LivePortraitAppearanceFeatureExtractor']:
                        self.models_processor.models_trt['LivePortraitAppearanceFeatureExtractor'] = self.models_processor.load_model_trt('LivePortraitAppearanceFeatureExtractor', custom_plugin_path=None, precision="fp16")

                appearance_feature_extractor_model = self.models_processor.models_trt['LivePortraitAppearanceFeatureExtractor']

                nvtx.range_push("forward")

                feed_dict = {"img": I_s}
                current_stream = torch.cuda.current_stream()
                preds_dict = appearance_feature_extractor_model.predict_async(feed_dict, current_stream)
                #current_stream.synchronize()

                output = preds_dict["output"]

                nvtx.range_pop()

            else:
                if face_editor_type == 'Human-Face':
                    if not self.models_processor.models['LivePortraitAppearanceFeatureExtractor']:
                        self.models_processor.models['LivePortraitAppearanceFeatureExtractor'] = self.models_processor.load_model('LivePortraitAppearanceFeatureExtractor')

                appearance_feature_extractor_model = self.models_processor.models['LivePortraitAppearanceFeatureExtractor']

                output = torch.empty((1,32,16,64,64), dtype=torch.float32, device=self.models_processor.device).contiguous()

                io_binding = appearance_feature_extractor_model.io_binding()
                io_binding.bind_input(name='img', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=I_s.size(), buffer_ptr=I_s.data_ptr())
                io_binding.bind_output(name='output', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=output.size(), buffer_ptr=output.data_ptr())

                if self.models_processor.device == "cuda":
                    torch.cuda.synchronize()
                elif self.models_processor.device != "cpu":
                    self.models_processor.syncvec.cpu()
                appearance_feature_extractor_model.run_with_iobinding(io_binding)

        return output

    def lp_retarget_eye(self, kp_source: torch.Tensor, eye_close_ratio: torch.Tensor, face_editor_type='Human-Face') -> torch.Tensor:
        """
        kp_source: BxNx3
        eye_close_ratio: Bx3
        Return: Bx(3*num_kp)
        """
        with torch.no_grad():
            # prepare_source
            feat_eye = faceutil.concat_feat(kp_source, eye_close_ratio).contiguous()

            if self.models_processor.provider_name in ["TensorRT", "TensorRT-Engine"]:
                if face_editor_type == 'Human-Face':
                    if not self.models_processor.models_trt['LivePortraitStitchingEye']:
                        self.models_processor.models_trt['LivePortraitStitchingEye'] = self.models_processor.load_model_trt('LivePortraitStitchingEye', custom_plugin_path=None, precision="fp16")

                stitching_eye_model = self.models_processor.models_trt['LivePortraitStitchingEye']

                nvtx.range_push("forward")

                feed_dict = {"input": feat_eye}
                current_stream = torch.cuda.current_stream()
                preds_dict = stitching_eye_model.predict_async(feed_dict, current_stream)
                #current_stream.synchronize() 

                delta = preds_dict["output"]

                nvtx.range_pop()

            else:
                if face_editor_type == 'Human-Face':
                    if not self.models_processor.models['LivePortraitStitchingEye']:
                        self.models_processor.models['LivePortraitStitchingEye'] = self.models_processor.load_model('LivePortraitStitchingEye')

                stitching_eye_model = self.models_processor.models['LivePortraitStitchingEye']

                delta = torch.empty((1,63), dtype=torch.float32, device=self.models_processor.device).contiguous()

                io_binding = stitching_eye_model.io_binding()
                io_binding.bind_input(name='input', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=feat_eye.size(), buffer_ptr=feat_eye.data_ptr())
                io_binding.bind_output(name='output', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=delta.size(), buffer_ptr=delta.data_ptr())

                if self.models_processor.device == "cuda":
                    torch.cuda.synchronize()
                elif self.models_processor.device != "cpu":
                    self.models_processor.syncvec.cpu()
                
                stitching_eye_model.run_with_iobinding(io_binding)

        return delta.reshape(-1, kp_source.shape[1], 3)

    def lp_retarget_lip(self, kp_source: torch.Tensor, lip_close_ratio: torch.Tensor, face_editor_type='Human-Face') -> torch.Tensor:
        """
        kp_source: BxNx3
        lip_close_ratio: Bx2
        Return: Bx(3*num_kp)
        """
        with torch.no_grad():
            # Prepare the input feature vector (common to both backends)
            feat_lip = faceutil.concat_feat(kp_source, lip_close_ratio).contiguous()

            if self.models_processor.provider_name in ["TensorRT", "TensorRT-Engine"]:
                if face_editor_type == 'Human-Face':
                    if not self.models_processor.models_trt['LivePortraitStitchingLip']:
                        self.models_processor.models_trt['LivePortraitStitchingLip'] = self.models_processor.load_model_trt('LivePortraitStitchingLip', custom_plugin_path=None, precision="fp16")

                stitching_lip_model = self.models_processor.models_trt['LivePortraitStitchingLip']

                nvtx.range_push("forward")

                feed_dict = {"input": feat_lip}
                current_stream = torch.cuda.current_stream()
                preds_dict = stitching_lip_model.predict_async(feed_dict, current_stream)
                #current_stream.synchronize() 

                delta = preds_dict["output"]

                nvtx.range_pop()

            else:
                if face_editor_type == 'Human-Face':
                    if not self.models_processor.models['LivePortraitStitchingLip']:
                        self.models_processor.models['LivePortraitStitchingLip'] = self.models_processor.load_model('LivePortraitStitchingLip')

                stitching_lip_model = self.models_processor.models['LivePortraitStitchingLip']

                delta = torch.empty((1,63), dtype=torch.float32, device=self.models_processor.device).contiguous()

                io_binding = stitching_lip_model.io_binding()
                io_binding.bind_input(name='input', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=feat_lip.size(), buffer_ptr=feat_lip.data_ptr())
                io_binding.bind_output(name='output', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=delta.size(), buffer_ptr=delta.data_ptr())

                if self.models_processor.device == "cuda":
                    torch.cuda.synchronize()
                elif self.models_processor.device != "cpu":
                    self.models_processor.syncvec.cpu()
                
                stitching_lip_model.run_with_iobinding(io_binding)

        return delta.reshape(-1, kp_source.shape[1], 3)
    
    def lp_stitch(self, kp_source: torch.Tensor, kp_driving: torch.Tensor, face_editor_type='Human-Face') -> torch.Tensor:
        """
        kp_source: BxNx3
        kp_driving: BxNx3
        Return: Bx(3*num_kp+2)
        """
        with torch.no_grad():
            # Prepare the concatenated feature vector (keypoints from source and driving)
            feat_stiching = faceutil.concat_feat(kp_source, kp_driving).contiguous()

            # When using TensorRT / engine. 
            if self.models_processor.provider_name in ["TensorRT", "TensorRT-Engine"]:
                if face_editor_type == 'Human-Face':
                    # Lazy loading of the TensorRT model
                    if not self.models_processor.models_trt['LivePortraitStitching']:
                        self.models_processor.models_trt['LivePortraitStitching'] = self.models_processor.load_model_trt('LivePortraitStitching', custom_plugin_path=None, precision="fp16")

                stitching_model = self.models_processor.models_trt['LivePortraitStitching']

                nvtx.range_push("forward")

                feed_dict = {"input": feat_stiching}
                
                # --- ASYNC FLOW CORRECTION FOR MULTITHREADING (TRT) ---
                # Use the current thread's CUDA stream for asynchronous execution.
                current_stream = torch.cuda.current_stream()
                
                # Launch inference asynchronously on the current stream.
                preds_dict = stitching_model.predict_async(feed_dict, current_stream)
                
                # CRITICAL SYNCHRONIZATION: Wait for the inference on this stream to complete 
                # before reading the results (delta) on the CPU or continuing with dependent GPU work.
                #current_stream.synchronize() 

                delta = preds_dict["output"]

                nvtx.range_pop()

            else:
                # --- Alternative Backend (e.g., ONNX Runtime) ---
                if face_editor_type == 'Human-Face':
                    # Lazy loading of the non-TRT model
                    if not self.models_processor.models['LivePortraitStitching']:
                        self.models_processor.models['LivePortraitStitching'] = self.models_processor.load_model('LivePortraitStitching')

                stitching_model = self.models_processor.models['LivePortraitStitching']

                delta = torch.empty((1,65), dtype=torch.float32, device=self.models_processor.device).contiguous()

                # Manual I/O binding (used for high-performance non-TRT runtimes)
                io_binding = stitching_model.io_binding()
                io_binding.bind_input(name='input', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=feat_stiching.size(), buffer_ptr=feat_stiching.data_ptr())
                io_binding.bind_output(name='output', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=delta.size(), buffer_ptr=delta.data_ptr())

                # Explicit synchronization logic for the non-TRT run
                if self.models_processor.device == "cuda":
                    # Synchronize all CUDA operations if running on GPU
                    torch.cuda.synchronize()
                elif self.models_processor.device != "cpu":
                    self.models_processor.syncvec.cpu()
                
                stitching_model.run_with_iobinding(io_binding)

        return delta

    def lp_stitching(self, kp_source: torch.Tensor, kp_driving: torch.Tensor, face_editor_type='Human-Face') -> torch.Tensor:
        """ conduct the stitching
        kp_source: Bxnum_kpx3
        kp_driving: Bxnum_kpx3
        """
        bs, num_kp = kp_source.shape[:2]

        # calculate default delta from kp_source (using kp_source as default)
        kp_driving_default = kp_source.clone()

        default_delta = self.models_processor.lp_stitch(kp_source, kp_driving_default, face_editor_type=face_editor_type)

        # Clone default delta values for expression and translation/rotation
        default_delta_exp = default_delta[..., :3*num_kp].reshape(bs, num_kp, 3).clone()  # 1x20x3
        default_delta_tx_ty = default_delta[..., 3*num_kp:3*num_kp+2].reshape(bs, 1, 2).clone()  # 1x1x2

        # Debug: Print default delta values (should be close to zero)
        #print("default_delta_exp:", default_delta_exp)
        #print("default_delta_tx_ty:", default_delta_tx_ty)

        kp_driving_new = kp_driving.clone()

        # calculate new delta based on kp_driving
        delta = self.models_processor.lp_stitch(kp_source, kp_driving_new, face_editor_type=face_editor_type)

        # Clone new delta values for expression and translation/rotation
        delta_exp = delta[..., :3*num_kp].reshape(bs, num_kp, 3).clone()  # 1x20x3
        delta_tx_ty = delta[..., 3*num_kp:3*num_kp+2].reshape(bs, 1, 2).clone()  # 1x1x2

        # Debug: Print new delta values
        #print("delta_exp:", delta_exp)
        #print("delta_tx_ty:", delta_tx_ty)

        # Calculate the difference between new and default delta
        delta_exp_diff = delta_exp - default_delta_exp
        delta_tx_ty_diff = delta_tx_ty - default_delta_tx_ty

        # Debug: Print the delta differences
        #print("delta_exp_diff:", delta_exp_diff)
        #print("delta_tx_ty_diff:", delta_tx_ty_diff)

        # Apply delta differences to the keypoints only if significant differences are found
        kp_driving_new += delta_exp_diff
        kp_driving_new[..., :2] += delta_tx_ty_diff

        return kp_driving_new

    def lp_warp_decode(self, feature_3d: torch.Tensor, kp_source: torch.Tensor, kp_driving: torch.Tensor, face_editor_type='Human-Face') -> torch.Tensor:
        """ get the image after the warping of the implicit keypoints
        feature_3d: Bx32x16x64x64, feature volume
        kp_source: BxNx3
        kp_driving: BxNx3
        """
        # Ensure all inputs are contiguous before the conditional logic
        feature_3d = feature_3d.contiguous()
        kp_source = kp_source.contiguous()
        kp_driving = kp_driving.contiguous()

        with torch.no_grad():
            if self.models_processor.provider_name in ["TensorRT", "TensorRT-Engine"]:
                if face_editor_type == 'Human-Face':
                    # Lazy loading of the TensorRT model
                    if not self.models_processor.models_trt['LivePortraitWarpingSpadeFix']:
                        
                        # Logic to select the correct platform-specific plugin path
                        if SYSTEM_PLATFORM == 'Windows':
                            plugin_path = f'{models_dir}/grid_sample_3d_plugin.dll'
                        elif SYSTEM_PLATFORM == 'Linux':
                            plugin_path = f'{models_dir}/libgrid_sample_3d_plugin.so'
                        else:
                            raise ValueError("TensorRT-Engine is only supported on Windows and Linux systems!")

                        # Load model with the custom plugin path and FP16 precision
                        self.models_processor.models_trt['LivePortraitWarpingSpadeFix'] = self.models_processor.load_model_trt('LivePortraitWarpingSpadeFix', custom_plugin_path=plugin_path, precision="fp16")

                warping_spade_model = self.models_processor.models_trt['LivePortraitWarpingSpadeFix']

                nvtx.range_push("forward")

                feed_dict = {
                    "feature_3d": feature_3d,
                    "kp_source": kp_source,
                    "kp_driving": kp_driving
                }
                
                # --- ASYNC FLOW CORRECTION FOR MULTITHREADING (TRT) ---
                # 1. Use the current thread's CUDA stream for asynchronous execution.
                current_stream = torch.cuda.current_stream()
                
                # 2. Launch inference asynchronously.
                preds_dict = warping_spade_model.predict_async(feed_dict, current_stream)
                
                # 3. CRITICAL SYNCHRONIZATION: Wait for the inference on this stream to complete.
                #current_stream.synchronize() 

                out = preds_dict["out"]

                nvtx.range_pop()
            else:
                # --- Alternative Backend (e.g., ONNX Runtime) ---
                if face_editor_type == 'Human-Face':
                    # Lazy loading of the non-TRT model
                    if not self.models_processor.models['LivePortraitWarpingSpade']:
                        self.models_processor.models['LivePortraitWarpingSpade'] = self.models_processor.load_model('LivePortraitWarpingSpade')

                warping_spade_model = self.models_processor.models['LivePortraitWarpingSpade']

                # Allocate output buffer
                out = torch.empty((1,3,512,512), dtype=torch.float32, device=self.models_processor.device).contiguous()
                
                # Manual I/O binding
                io_binding = warping_spade_model.io_binding()
                io_binding.bind_input(name='feature_3d', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=feature_3d.size(), buffer_ptr=feature_3d.data_ptr())
                io_binding.bind_input(name='kp_driving', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=kp_driving.size(), buffer_ptr=kp_driving.data_ptr())
                io_binding.bind_input(name='kp_source', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=kp_source.size(), buffer_ptr=kp_source.data_ptr())
                io_binding.bind_output(name='out', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=out.size(), buffer_ptr=out.data_ptr())

                # Explicit synchronization logic for the non-TRT run
                if self.models_processor.device == "cuda":
                    torch.cuda.synchronize()
                elif self.models_processor.device != "cpu":
                    self.models_processor.syncvec.cpu()
                    
                warping_spade_model.run_with_iobinding(io_binding)

        return out
    
    def _get_faceparser_labels_via_facemasks(self, img_uint8_3x512x512: torch.Tensor) -> torch.Tensor:
        """
        img_uint8_3x512x512: [3,512,512] uint8 (0..255)
        returns labels: [512,512] long
        """
        fm = getattr(self.models_processor, "face_masks", None)
        if fm is None or not hasattr(fm, "_faceparser_labels"):
            raise RuntimeError("models_processor.face_masks._faceparser_labels nicht verfügbar.")
        return fm._faceparser_labels(img_uint8_3x512x512)


    # --- maskiertes RGB-Blending (fehlte in deinem Code) ---
    def face_parser_makeup_direct_rgb_masked(self, img: torch.Tensor, mask: torch.Tensor,
                                             color=None, blend_factor: float = 0.2) -> torch.Tensor:
        """
        img:  [3,H,W] uint8
        mask: [H,W] bool oder float (0..1)
        color: [R,G,B] 0..255
        """
        device = img.device
        color = color or [230, 50, 20]
        blend_factor = float(max(0.0, min(1.0, blend_factor)))

        # Farbe in [0,1]
        r, g, b = [c / 255.0 for c in color]
        tar_color = torch.tensor([r, g, b], dtype=torch.float32, device=device).view(3, 1, 1)

        # Maske in float [0,1]
        if mask.dtype == torch.bool:
            m = mask.float()
        else:
            m = mask.clamp(0.0, 1.0).float()
        m = m.unsqueeze(0)  # [1,H,W]

        img_f = img.float() / 255.0
        w = m * blend_factor
        
        t512_mask =  v2.Resize((512, 512), interpolation=v2.InterpolationMode.BILINEAR, antialias=False)
        w = t512_mask(w)
        w = w.clamp(0, 255)
        
        out = img_f * (1.0 - w) + tar_color * w
        out = (out * 255.0).clamp(0, 255).to(torch.uint8)
        return out


    def face_parser_makeup_direct_rgb(self, img, parsing, part=(17,), color=None, blend_factor=0.2):
        """
        img:     [3,H,W] uint8
        parsing: [H,W] Labels ODER [1,19,H,W] Logits (robust)
        """
        device = img.device
        color = color or [230, 50, 20]
        blend_factor = float(max(0.0, min(1.0, blend_factor)))

        # Parsing -> Labels [H,W]
        if parsing.dim() == 2:
            labels = parsing.to(torch.long)
        elif parsing.dim() == 4 and parsing.shape[0] == 1 and parsing.shape[1] == 19:
            labels = parsing.argmax(dim=1).squeeze(0).to(torch.long)
        else:
            raise ValueError(f"Unsupported parsing tensor shape: {tuple(parsing.shape)}")

        # Zielmaske (bool [H,W])
        if isinstance(part, tuple):
            m = torch.zeros_like(labels, dtype=torch.bool, device=device)
            for p in part:
                m |= (labels == int(p))
        else:
            m = (labels == int(part))

        return self.face_parser_makeup_direct_rgb_masked(
            img=img, mask=m, color=color, blend_factor=blend_factor
        )


    def apply_face_makeup(self, img, parameters):
        """
        img: [3,512,512] uint8
        returns: out_img [3,512,512] uint8, combined_mask [1,512,512] float
        """
        device = img.device

        # 1) Labels via face_masks (schnell + konsistent)
        labels = self._get_faceparser_labels_via_facemasks(img)  # [512,512] long

        # 2) Arbeitskopie
        out = img.clone()

        # 3) Je Bereich einfärben (optional)
        if parameters.get('FaceMakeupEnableToggle', False):
            color = [parameters['FaceMakeupRedSlider'], parameters['FaceMakeupGreenSlider'], parameters['FaceMakeupBlueSlider']]
            out = self.face_parser_makeup_direct_rgb(out, labels, part=(1, 7, 8, 10),
                                                     color=color,
                                                     blend_factor=parameters['FaceMakeupBlendAmountDecimalSlider'])
        '''
        if parameters.get('EyesMakeupEnableToggle', False):
            color = [parameters['EyesMakeupRedSlider'], parameters['EyesMakeupGreenSlider'], parameters['EyesMakeupBlueSlider']]
            # Augen (4 & 5)
            eye_mask = ((labels == 4) | (labels == 5)).float().unsqueeze(0).unsqueeze(0)  # [1,1,512,512]
            # „Pupille“ enger (Erosion via max_pool auf invertierter Maske)
            pupil_mask = 1.0 - torch.nn.functional.max_pool2d(1.0 - eye_mask, kernel_size=5, stride=1, padding=2)
            pupil_mask = pupil_mask.squeeze(0).squeeze(0)  # [512,512]
            # ggf. blur
            blur_k = int(parameters.get('EyesMakeupBlurSlider', 0)) * 2 + 1
            if blur_k > 1:
                blur = transforms.GaussianBlur(blur_k, sigma=(parameters['EyesMakeupBlurSlider'] + 1) * 0.2)
                pupil_mask = blur(pupil_mask.unsqueeze(0)).squeeze(0)
            out = self.face_parser_makeup_direct_rgb_masked(
                img=out, mask=pupil_mask, color=color,
                blend_factor=parameters['EyesMakeupBlendAmountDecimalSlider']
            )
        '''
        
        if parameters.get('HairMakeupEnableToggle', False):
            color = [parameters['HairMakeupRedSlider'], parameters['HairMakeupGreenSlider'], parameters['HairMakeupBlueSlider']]
            out = self.face_parser_makeup_direct_rgb(out, labels, part=17,
                                                     color=color,
                                                     blend_factor=parameters['HairMakeupBlendAmountDecimalSlider'])

        if parameters.get('EyeBrowsMakeupEnableToggle', False):
            color = [parameters['EyeBrowsMakeupRedSlider'], parameters['EyeBrowsMakeupGreenSlider'], parameters['EyeBrowsMakeupBlueSlider']]
            out = self.face_parser_makeup_direct_rgb(out, labels, part=(2, 3),
                                                     color=color,
                                                     blend_factor=parameters['EyeBrowsMakeupBlendAmountDecimalSlider'])

        if parameters.get('LipsMakeupEnableToggle', False):
            color = [parameters['LipsMakeupRedSlider'], parameters['LipsMakeupGreenSlider'], parameters['LipsMakeupBlueSlider']]
            out = self.face_parser_makeup_direct_rgb(out, labels, part=(12, 13),
                                                     color=color,
                                                     blend_factor=parameters['LipsMakeupBlendAmountDecimalSlider'])

        # 4) Kombinierte Maske (für Rückgabe/Debug)
        face_attributes = {
            1: parameters.get('FaceMakeupEnableToggle', False),
            2: parameters.get('EyeBrowsMakeupEnableToggle', False),
            3: parameters.get('EyeBrowsMakeupEnableToggle', False),
            4: parameters.get('EyesMakeupEnableToggle', False),
            5: parameters.get('EyesMakeupEnableToggle', False),
            7: parameters.get('FaceMakeupEnableToggle', False),
            8: parameters.get('FaceMakeupEnableToggle', False),
            10: parameters.get('FaceMakeupEnableToggle', False),
            12: parameters.get('LipsMakeupEnableToggle', False),
            13: parameters.get('LipsMakeupEnableToggle', False),
            17: parameters.get('HairMakeupEnableToggle', False),
        }

        combined_mask = torch.zeros((256, 256), dtype=torch.float32, device=device)
        for attr, enabled in face_attributes.items():
            if not enabled:
                continue
            
            combined_mask = torch.max(combined_mask, (labels == int(attr)).float())
        
        t512_mask =  v2.Resize((512, 512), interpolation=v2.InterpolationMode.BILINEAR, antialias=False)
        combined_mask = t512_mask(combined_mask.unsqueeze(0))
        combined_mask = combined_mask.clamp(0, 255)
        
        out_final = out.to(torch.uint8)
        return out_final, combined_mask