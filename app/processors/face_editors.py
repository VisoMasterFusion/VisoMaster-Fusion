import pickle
from typing import TYPE_CHECKING, Dict
import platform
import os

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
    """
    Manages face editing functionalities, primarily using the LivePortrait model pipeline.
    This class handles motion extraction, feature extraction, stitching, warping,
    and post-processing effects like makeup application.
    """
    def __init__(self, models_processor: 'ModelsProcessor'):
        self.models_processor = models_processor
        # Pre-create a faded mask for cropping operations.
        self.lp_mask_crop = faceutil.create_faded_inner_mask(size=(512, 512), border_thickness=5, fade_thickness=15, blur_radius=5, device=self.models_processor.device)
        self.lp_mask_crop = torch.unsqueeze(self.lp_mask_crop, 0)
        try:
            # Load pre-calculated lip array for lip retargeting.
            self.lp_lip_array = np.array(self.load_lip_array())
        except FileNotFoundError:
            self.lp_lip_array = None

    def load_lip_array(self):
        """Loads the lip array data from a pickle file."""
        # Use os.path.join for better cross-platform compatibility.
        lip_array_path = os.path.join(models_dir, 'liveportrait_onnx', 'lip_array.pkl')
        with open(lip_array_path, 'rb') as f:
            return pickle.load(f)

    def _run_onnx_io_binding(self, model_name: str, inputs: Dict[str, torch.Tensor], output_spec: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Private helper to run inference using ONNX Runtime with io_binding.
        This centralizes the verbose binding logic for the ONNX backend.

        Args:
            model_name (str): The name of the model to run.
            inputs (Dict[str, torch.Tensor]): A dictionary mapping input names to their tensors.
            output_spec (Dict[str, torch.Tensor]): A dictionary mapping output names to pre-allocated empty tensors.

        Returns:
            Dict[str, torch.Tensor]: The output_spec dictionary, now populated with the model's output.
        """
        model = self.models_processor.models[model_name]
        io_binding = model.io_binding()

        # Bind inputs
        for name, tensor in inputs.items():
            io_binding.bind_input(name=name, device_type=self.models_processor.device, device_id=0,
                                  element_type=np.float32, shape=tensor.size(), buffer_ptr=tensor.data_ptr())
        # Bind outputs
        for name, tensor in output_spec.items():
            io_binding.bind_output(name=name, device_type=self.models_processor.device, device_id=0,
                                   element_type=np.float32, shape=tensor.size(), buffer_ptr=tensor.data_ptr())

        # Synchronize and run
        if self.models_processor.device == "cuda":
            torch.cuda.synchronize()
        elif self.models_processor.device != "cpu":
            self.models_processor.syncvec.cpu()
        model.run_with_iobinding(io_binding)

        return output_spec
        
    def lp_motion_extractor(self, img, face_editor_type='Human-Face', **kwargs) -> dict:
        """
        Extracts motion-related keypoints and parameters (head pose, expression, etc.) from an image.
        """
        kp_info = {}
        with torch.no_grad():
            I_s = torch.div(img.type(torch.float32), 255.)
            I_s = torch.clamp(I_s, 0, 1)
            I_s = torch.unsqueeze(I_s, 0).contiguous()

            if self.models_processor.provider_name in ["TensorRT", "TensorRT-Engine"]:
                if face_editor_type == 'Human-Face':
                    if not self.models_processor.models_trt['LivePortraitMotionExtractor']:
                        self.models_processor.models_trt['LivePortraitMotionExtractor'] = self.models_processor.load_model_trt('LivePortraitMotionExtractor', custom_plugin_path=None, precision="fp32")

                motion_extractor_model = self.models_processor.models_trt['LivePortraitMotionExtractor']

                # 1. Pré-allouer les tenseurs de sortie
                pitch = torch.empty((1, 66), dtype=torch.float32, device=self.models_processor.device).contiguous()
                yaw = torch.empty((1, 66), dtype=torch.float32, device=self.models_processor.device).contiguous()
                roll = torch.empty((1, 66), dtype=torch.float32, device=self.models_processor.device).contiguous()
                t = torch.empty((1, 3), dtype=torch.float32, device=self.models_processor.device).contiguous()
                exp = torch.empty((1, 63), dtype=torch.float32, device=self.models_processor.device).contiguous()
                scale = torch.empty((1, 1), dtype=torch.float32, device=self.models_processor.device).contiguous()
                kp = torch.empty((1, 63), dtype=torch.float32, device=self.models_processor.device).contiguous()

                # 2. Créer le dictionnaire de bindings complet (entrées + sorties)
                bindings = {
                    "img": I_s,
                    "pitch": pitch, "yaw": yaw, "roll": roll,
                    "t": t, "exp": exp, "scale": scale, "kp": kp
                }

                current_stream = torch.cuda.current_stream()
                # 3. Appeler predict_async
                motion_extractor_model.predict_async(bindings, current_stream)
                
                # Les résultats sont maintenant directement dans les tenseurs pitch, yaw, etc.
                kp_info = {
                    'pitch': pitch, 'yaw': yaw, 'roll': roll,
                    't': t, 'exp': exp, 'scale': scale, 'kp': kp
                }
            else: # ONNX Path
                if face_editor_type == 'Human-Face':
                    if not self.models_processor.models['LivePortraitMotionExtractor']:
                        self.models_processor.models['LivePortraitMotionExtractor'] = self.models_processor.load_model('LivePortraitMotionExtractor')

                inputs = {'img': I_s}
                output_spec = {
                    'pitch': torch.empty((1, 66), dtype=torch.float32, device=self.models_processor.device).contiguous(),
                    'yaw': torch.empty((1, 66), dtype=torch.float32, device=self.models_processor.device).contiguous(),
                    'roll': torch.empty((1, 66), dtype=torch.float32, device=self.models_processor.device).contiguous(),
                    't': torch.empty((1, 3), dtype=torch.float32, device=self.models_processor.device).contiguous(),
                    'exp': torch.empty((1, 63), dtype=torch.float32, device=self.models_processor.device).contiguous(),
                    'scale': torch.empty((1, 1), dtype=torch.float32, device=self.models_processor.device).contiguous(),
                    'kp': torch.empty((1, 63), dtype=torch.float32, device=self.models_processor.device).contiguous()
                }
                kp_info = self._run_onnx_io_binding('LivePortraitMotionExtractor', inputs, output_spec)

            # Refine output shapes and convert head pose to degrees.
            if kwargs.get('flag_refine_info', True):
                bs = kp_info['kp'].shape[0]
                kp_info['pitch'] = faceutil.headpose_pred_to_degree(kp_info['pitch'])[:, None] # Bx1
                kp_info['yaw'] = faceutil.headpose_pred_to_degree(kp_info['yaw'])[:, None]   # Bx1
                kp_info['roll'] = faceutil.headpose_pred_to_degree(kp_info['roll'])[:, None] # Bx1
                kp_info['kp'] = kp_info['kp'].reshape(bs, -1, 3)   # BxNx3
                kp_info['exp'] = kp_info['exp'].reshape(bs, -1, 3) # BxNx3

        return kp_info

    def lp_appearance_feature_extractor(self, img, face_editor_type='Human-Face'):
        """
        Extracts the appearance feature volume from an image.
        """
        with torch.no_grad():
            I_s = torch.div(img.type(torch.float32), 255.)
            I_s = torch.clamp(I_s, 0, 1)
            I_s = torch.unsqueeze(I_s, 0).contiguous()
            
            if self.models_processor.provider_name in ["TensorRT", "TensorRT-Engine"]:
                if face_editor_type == 'Human-Face':
                    if not self.models_processor.models_trt['LivePortraitAppearanceFeatureExtractor']:
                        self.models_processor.models_trt['LivePortraitAppearanceFeatureExtractor'] = self.models_processor.load_model_trt('LivePortraitAppearanceFeatureExtractor', custom_plugin_path=None, precision="fp16")

                appearance_feature_extractor_model = self.models_processor.models_trt['LivePortraitAppearanceFeatureExtractor']

                output = torch.empty((1, 32, 16, 64, 64), dtype=torch.float32, device=self.models_processor.device).contiguous()
                bindings = {"img": I_s, "output": output}
                current_stream = torch.cuda.current_stream()
                appearance_feature_extractor_model.predict_async(bindings, current_stream)

            else:
                if face_editor_type == 'Human-Face':
                    if not self.models_processor.models['LivePortraitAppearanceFeatureExtractor']:
                        self.models_processor.models['LivePortraitAppearanceFeatureExtractor'] = self.models_processor.load_model('LivePortraitAppearanceFeatureExtractor')
                
                inputs = {'img': I_s}
                output_spec = {'output': torch.empty((1, 32, 16, 64, 64), dtype=torch.float32, device=self.models_processor.device).contiguous()}
                results = self._run_onnx_io_binding('LivePortraitAppearanceFeatureExtractor', inputs, output_spec)
                output = results['output']

        return output

    def lp_retarget_eye(self, kp_source: torch.Tensor, eye_close_ratio: torch.Tensor, face_editor_type='Human-Face') -> torch.Tensor:
        """
        Calculates the delta to adjust eye keypoints based on a target eye close ratio.
        Args:
            kp_source (torch.Tensor): BxNx3 source keypoints.
            eye_close_ratio (torch.Tensor): Bx3 target eye close ratio.
        Returns:
            torch.Tensor: BxNx3 delta to be added to keypoints.
        """
        with torch.no_grad():
            feat_eye = faceutil.concat_feat(kp_source, eye_close_ratio).contiguous()
            
            if self.models_processor.provider_name in ["TensorRT", "TensorRT-Engine"]:
                if face_editor_type == 'Human-Face':
                    if not self.models_processor.models_trt['LivePortraitStitchingEye']:
                        self.models_processor.models_trt['LivePortraitStitchingEye'] = self.models_processor.load_model_trt('LivePortraitStitchingEye', custom_plugin_path=None, precision="fp16")

                stitching_eye_model = self.models_processor.models_trt['LivePortraitStitchingEye']

                delta = torch.empty((1, 63), dtype=torch.float32, device=self.models_processor.device).contiguous()
                bindings = {"input": feat_eye, "output": delta}
                current_stream = torch.cuda.current_stream()
                stitching_eye_model.predict_async(bindings, current_stream)

            else:
                if face_editor_type == 'Human-Face':
                    if not self.models_processor.models['LivePortraitStitchingEye']:
                        self.models_processor.models['LivePortraitStitchingEye'] = self.models_processor.load_model('LivePortraitStitchingEye')
                
                inputs = {'input': feat_eye}
                output_spec = {'output': torch.empty((1, 63), dtype=torch.float32, device=self.models_processor.device).contiguous()}
                results = self._run_onnx_io_binding('LivePortraitStitchingEye', inputs, output_spec)
                delta = results['output']

        return delta.reshape(-1, kp_source.shape[1], 3)

    def lp_retarget_lip(self, kp_source: torch.Tensor, lip_close_ratio: torch.Tensor, face_editor_type='Human-Face') -> torch.Tensor:
        """
        Calculates the delta to adjust lip keypoints based on a target lip close ratio.
        Args:
            kp_source (torch.Tensor): BxNx3 source keypoints.
            lip_close_ratio (torch.Tensor): Bx2 target lip close ratio.
        Returns:
            torch.Tensor: BxNx3 delta to be added to keypoints.
        """
        with torch.no_grad():
            feat_lip = faceutil.concat_feat(kp_source, lip_close_ratio).contiguous()

            if self.models_processor.provider_name in ["TensorRT", "TensorRT-Engine"]:
                if face_editor_type == 'Human-Face':
                    if not self.models_processor.models_trt['LivePortraitStitchingLip']:
                        self.models_processor.models_trt['LivePortraitStitchingLip'] = self.models_processor.load_model_trt('LivePortraitStitchingLip', custom_plugin_path=None, precision="fp16")

                stitching_lip_model = self.models_processor.models_trt['LivePortraitStitchingLip']
                
                delta = torch.empty((1, 63), dtype=torch.float32, device=self.models_processor.device).contiguous()
                bindings = {"input": feat_lip, "output": delta}
                current_stream = torch.cuda.current_stream()
                stitching_lip_model.predict_async(bindings, current_stream)
                
            else:
                if face_editor_type == 'Human-Face':
                    if not self.models_processor.models['LivePortraitStitchingLip']:
                        self.models_processor.models['LivePortraitStitchingLip'] = self.models_processor.load_model('LivePortraitStitchingLip')

                inputs = {'input': feat_lip}
                output_spec = {'output': torch.empty((1, 63), dtype=torch.float32, device=self.models_processor.device).contiguous()}
                results = self._run_onnx_io_binding('LivePortraitStitchingLip', inputs, output_spec)
                delta = results['output']

        return delta.reshape(-1, kp_source.shape[1], 3)
    
    def lp_stitch(self, kp_source: torch.Tensor, kp_driving: torch.Tensor, face_editor_type='Human-Face') -> torch.Tensor:
        """
        Calculates the delta for stitching source and driving keypoints.
        Args:
            kp_source (torch.Tensor): BxNx3 source keypoints.
            kp_driving (torch.Tensor): BxNx3 driving keypoints.
        Returns:
            torch.Tensor: Bx(3*num_kp+2) delta tensor.
        """
        with torch.no_grad():
            feat_stiching = faceutil.concat_feat(kp_source, kp_driving).contiguous()

            if self.models_processor.provider_name in ["TensorRT", "TensorRT-Engine"]:
                if face_editor_type == 'Human-Face':
                    if not self.models_processor.models_trt['LivePortraitStitching']:
                        self.models_processor.models_trt['LivePortraitStitching'] = self.models_processor.load_model_trt('LivePortraitStitching', custom_plugin_path=None, precision="fp16")
                
                stitching_model = self.models_processor.models_trt['LivePortraitStitching']
                
                delta = torch.empty((1, 65), dtype=torch.float32, device=self.models_processor.device).contiguous()
                bindings = {"input": feat_stiching, "output": delta}
                current_stream = torch.cuda.current_stream()
                stitching_model.predict_async(bindings, current_stream)
            
            else:
                if face_editor_type == 'Human-Face':
                    if not self.models_processor.models['LivePortraitStitching']:
                        self.models_processor.models['LivePortraitStitching'] = self.models_processor.load_model('LivePortraitStitching')
                
                inputs = {'input': feat_stiching}
                output_spec = {'output': torch.empty((1, 65), dtype=torch.float32, device=self.models_processor.device).contiguous()}
                results = self._run_onnx_io_binding('LivePortraitStitching', inputs, output_spec)
                delta = results['output']

        return delta

    def lp_stitching(self, kp_source: torch.Tensor, kp_driving: torch.Tensor, face_editor_type='Human-Face') -> torch.Tensor:
        """ 
        Conducts the stitching process by calculating and applying deltas.
        Args:
            kp_source (torch.Tensor): BxNx3 source keypoints.
            kp_driving (torch.Tensor): BxNx3 driving keypoints.
        Returns:
            torch.Tensor: The new, stitched driving keypoints.
        """
        bs, num_kp = kp_source.shape[:2]
        
        # Calculate default delta from kp_source (using kp_source as its own driving signal).
        kp_driving_default = kp_source.clone()
        default_delta = self.models_processor.lp_stitch(kp_source, kp_driving_default, face_editor_type=face_editor_type)

        # Separate default delta into expression and translation/rotation components.
        default_delta_exp = default_delta[..., :3*num_kp].reshape(bs, num_kp, 3).clone()
        default_delta_tx_ty = default_delta[..., 3*num_kp:3*num_kp+2].reshape(bs, 1, 2).clone()

        # Calculate new delta based on the actual kp_driving.
        kp_driving_new = kp_driving.clone()
        delta = self.models_processor.lp_stitch(kp_source, kp_driving_new, face_editor_type=face_editor_type)

        # Separate new delta into components.
        delta_exp = delta[..., :3*num_kp].reshape(bs, num_kp, 3).clone()
        delta_tx_ty = delta[..., 3*num_kp:3*num_kp+2].reshape(bs, 1, 2).clone()

        # Calculate the difference to find the true motion delta.
        delta_exp_diff = delta_exp - default_delta_exp
        delta_tx_ty_diff = delta_tx_ty - default_delta_tx_ty

        # Apply the motion delta to the driving keypoints.
        kp_driving_new += delta_exp_diff
        kp_driving_new[..., :2] += delta_tx_ty_diff

        return kp_driving_new

    def lp_warp_decode(self, feature_3d: torch.Tensor, kp_source: torch.Tensor, kp_driving: torch.Tensor, face_editor_type='Human-Face') -> torch.Tensor:
        """ 
        Generates the final image after warping the implicit keypoints.
        Args:
            feature_3d (torch.Tensor): Bx32x16x64x64 feature volume.
            kp_source (torch.Tensor): BxNx3 source keypoints.
            kp_driving (torch.Tensor): BxNx3 driving keypoints.
        Returns:
            torch.Tensor: The final warped and decoded image.
        """
        with torch.no_grad():
            feature_3d = feature_3d.contiguous()
            kp_source = kp_source.contiguous()
            kp_driving = kp_driving.contiguous()

            if self.models_processor.provider_name in ["TensorRT", "TensorRT-Engine"]:
                model_name = 'LivePortraitWarpingSpadeFix'
                if face_editor_type == 'Human-Face':
                    if not self.models_processor.models_trt.get(model_name):
                        if SYSTEM_PLATFORM == 'Windows':
                            plugin_path = os.path.join(models_dir, 'grid_sample_3d_plugin.dll')
                        elif SYSTEM_PLATFORM == 'Linux':
                            plugin_path = os.path.join(models_dir, 'libgrid_sample_3d_plugin.so')
                        else:
                            raise ValueError("TensorRT-Engine is only supported on Windows and Linux systems!")
                        
                        self.models_processor.models_trt[model_name] = self.models_processor.load_model_trt(model_name, custom_plugin_path=plugin_path, precision="fp16")
                
                warping_spade_model = self.models_processor.models_trt[model_name]
                
                out = torch.empty((1, 3, 512, 512), dtype=torch.float32, device=self.models_processor.device).contiguous()
                bindings = {"feature_3d": feature_3d, "kp_source": kp_source, "kp_driving": kp_driving, "out": out}
                current_stream = torch.cuda.current_stream()
                warping_spade_model.predict_async(bindings, current_stream)

            else:
                model_name = 'LivePortraitWarpingSpade'
                if face_editor_type == 'Human-Face':
                    if not self.models_processor.models.get(model_name):
                        self.models_processor.models[model_name] = self.models_processor.load_model(model_name)
                
                inputs = {"feature_3d": feature_3d, "kp_driving": kp_driving, "kp_source": kp_source}
                output_spec = {'out': torch.empty((1, 3, 512, 512), dtype=torch.float32, device=self.models_processor.device).contiguous()}
                results = self._run_onnx_io_binding(model_name, inputs, output_spec)
                out = results['out']

        return out
    
    def _get_faceparser_labels_via_facemasks(self, img_uint8_3x512x512: torch.Tensor) -> torch.Tensor:
        """
        Takes [3,512,512] uint8, calls the 512 parser model,
        but returns [256,256] labels (long, 0..18).

        Args:
            img_uint8_3x512x512 (torch.Tensor): Input image tensor [3,512,512] uint8 (0..255).

        Returns:
            torch.Tensor: Label map tensor [256,256] of type long.
        """
        fm = getattr(self.models_processor, "face_masks", None)
        if fm is None or not hasattr(fm, "_faceparser_labels"):
            raise RuntimeError("models_processor.face_masks._faceparser_labels is not available.")
        return fm._faceparser_labels(img_uint8_3x512x512)


    def face_parser_makeup_direct_rgb_masked(self, img: torch.Tensor, mask: torch.Tensor,
                                             color=None, blend_factor: float = 0.2) -> torch.Tensor:
        """
        Applies masked RGB blending.

        Args:
            img (torch.Tensor): Image tensor [3,H,W] uint8.
            mask (torch.Tensor): Mask tensor [H,W] bool or float (0..1).
            color (list, optional): [R,G,B] color, 0..255. Defaults to None.
            blend_factor (float, optional): Blending factor. Defaults to 0.2.

        Returns:
            torch.Tensor: Blended image tensor.
        """
        device = img.device
        color = color or [230, 50, 20]
        blend_factor = float(max(0.0, min(1.0, blend_factor)))

        # Color to [0,1] range
        r, g, b = [c / 255.0 for c in color]
        tar_color = torch.tensor([r, g, b], dtype=torch.float32, device=device).view(3, 1, 1)

        # Mask to float [0,1]
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
        Applies makeup to specific parts of a face based on a parsing map.

        Args:
            img (torch.Tensor): Image tensor [3,H,W] uint8.
            parsing (torch.Tensor): Parsing map [H,W] Labels OR [1,19,H,W] Logits.
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

        # Target mask (bool [H,W])
        if isinstance(part, tuple):
            m = torch.zeros_like(labels, dtype=torch.bool, device=device)
            for p in part:
                m |= (labels == int(p))
        else:
            m = (labels == int(p))

        return self.face_parser_makeup_direct_rgb_masked(
            img=img, mask=m, color=color, blend_factor=blend_factor
        )


    def apply_face_makeup(self, img, parameters):
        """
        Applies various makeup effects to a face image based on parameters.

        Args:
            img (torch.Tensor): Input image [3,512,512] uint8.
            parameters (dict): Dictionary of makeup parameters.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - out_img (torch.Tensor): Output image [3,512,512] uint8.
                - combined_mask (torch.Tensor): Combined mask [1,512,512] float.
        """
        device = img.device

        # 1) Get labels via face_masks (fast and consistent)
        labels = self._get_faceparser_labels_via_facemasks(img)  # [256,256] long

        # 2) Create a working copy
        out = img.clone()

        # 3) Color each area (optional)
        if parameters.get('FaceMakeupEnableToggle', False):
            color = [parameters['FaceMakeupRedSlider'], parameters['FaceMakeupGreenSlider'], parameters['FaceMakeupBlueSlider']]
            out = self.face_parser_makeup_direct_rgb(out, labels, part=(1, 7, 8, 10),
                                                     color=color,
                                                     blend_factor=parameters['FaceMakeupBlendAmountDecimalSlider'])
        
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

        # 4) Combined mask (for return/debug)
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