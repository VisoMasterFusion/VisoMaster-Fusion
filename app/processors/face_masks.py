from typing import TYPE_CHECKING

import torch
import numpy as np
from torchvision import transforms
from torchvision.transforms import v2
import torch.nn.functional as F
import kornia.morphology as morph
from collections import defaultdict

from app.processors.external.clipseg import CLIPDensePredT
from app.processors.models_data import models_dir
if TYPE_CHECKING:
    from app.processors.models_processor import ModelsProcessor

class FaceMasks:
    def __init__(self, models_processor: 'ModelsProcessor'):
        self.models_processor = models_processor

    def apply_occlusion(self, img, amount):
        img = torch.div(img, 255)
        img = torch.unsqueeze(img, 0).contiguous()
        outpred = torch.ones((256,256), dtype=torch.float32, device=self.models_processor.device).contiguous()

        self.models_processor.run_occluder(img, outpred)

        outpred = torch.squeeze(outpred)
        outpred = (outpred > 0)
        outpred = torch.unsqueeze(outpred, 0).type(torch.float32)

        if amount >0:
            kernel = torch.ones((1,1,3,3), dtype=torch.float32, device=self.models_processor.device)

            for _ in range(int(amount)):
                outpred = torch.nn.functional.conv2d(outpred, kernel, padding=(1, 1))
                outpred = torch.clamp(outpred, 0, 1)

            outpred = torch.squeeze(outpred)

        if amount <0:
            outpred = torch.neg(outpred)
            outpred = torch.add(outpred, 1)
            kernel = torch.ones((1,1,3,3), dtype=torch.float32, device=self.models_processor.device)

            for _ in range(int(-amount)):
                outpred = torch.nn.functional.conv2d(outpred, kernel, padding=(1, 1))
                outpred = torch.clamp(outpred, 0, 1)

            outpred = torch.squeeze(outpred)
            outpred = torch.neg(outpred)
            outpred = torch.add(outpred, 1)

        outpred = torch.reshape(outpred, (1, 256, 256))
        return outpred

    def run_occluder(self, image, output):
        if not self.models_processor.models['Occluder']:
            self.models_processor.models['Occluder'] = self.models_processor.load_model('Occluder')

        io_binding = self.models_processor.models['Occluder'].io_binding()
        io_binding.bind_input(name='img', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=(1,3,256,256), buffer_ptr=image.data_ptr())
        io_binding.bind_output(name='output', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=(1,1,256,256), buffer_ptr=output.data_ptr())

        if self.models_processor.device == "cuda":
            torch.cuda.synchronize()
        elif self.models_processor.device != "cpu":
            self.models_processor.syncvec.cpu()
        self.models_processor.models['Occluder'].run_with_iobinding(io_binding)

    def apply_dfl_xseg(self, img, amount, background, mouth, parameters):
        amount2 = -parameters["DFLXSeg2SizeSlider"]
        amount_calc = -parameters["DFLXSeg3SizeSlider"]

        img = img.type(torch.float32)
        img = torch.div(img, 255)
        img = torch.unsqueeze(img, 0).contiguous()
        outpred = torch.ones((256,256), dtype=torch.float32, device=self.models_processor.device).contiguous()

        self.run_dfl_xseg(img, outpred)

        outpred = torch.clamp(outpred, min=0.0, max=1.0)
        outpred[outpred < 0.1] = 0
        outpred_calc = outpred.clone()
        
        # invert values to mask areas to keep
        outpred = 1.0 - outpred
        outpred = torch.unsqueeze(outpred, 0).type(torch.float32)
        
        outpred_calc = torch.where(outpred_calc < 0.1, 0, 1).float()
        outpred_calc = 1.0 - outpred_calc
        outpred_calc = torch.unsqueeze(outpred_calc, 0).type(torch.float32)
 
        outpred_calc_dill = outpred_calc.clone()
        
        if amount2 != amount:
            outpred2 = outpred.clone()

        if amount > 0:
            r = int(amount)
            k = 2*r + 1
            # einmalige Dilatation um Radius r
            outpred = F.max_pool2d(outpred, kernel_size=k, stride=1, padding=r)
            # falls nötig, wieder auf [0,1] clampen
            outpred = outpred.clamp(0,1)

        elif amount < 0:
            r = int(-amount)
            k = 2*r + 1
            # Erosion = invertieren → dilatieren → invertieren
            outpred = 1 - outpred
            outpred = F.max_pool2d(outpred, kernel_size=k, stride=1, padding=r)
            outpred = 1 - outpred
            outpred = outpred.clamp(0,1)

        gauss = transforms.GaussianBlur(parameters['OccluderXSegBlurSlider']*2+1, (parameters['OccluderXSegBlurSlider']+1)*0.2)
        outpred = gauss(outpred)  
        if amount2 != amount:
            if amount2 > 0:
                r2 = int(amount2)
                k2 = 2*r2 + 1
                # Dilatation um Radius r2
                outpred2 = F.max_pool2d(outpred2, kernel_size=k2, stride=1, padding=r2)
                outpred2 = outpred2.clamp(0,1)

            elif amount2 < 0:
                r2 = int(-amount2)
                k2 = 2*r2 + 1
                # Erosion = invertieren → dilatieren → invertieren
                outpred2 = 1 - outpred2
                outpred2 = F.max_pool2d(outpred2, kernel_size=k2, stride=1, padding=r2)
                outpred2 = 1 - outpred2
                outpred2 = outpred2.clamp(0,1)
            #outpred2_autocolor = outpred2.clone()
            
            gauss = transforms.GaussianBlur(parameters['XSeg2BlurSlider']*2+1, (parameters['XSeg2BlurSlider']+1)*0.2)
            outpred2 = gauss(outpred2) 
            
            #print("outpred, outpred2, mouth: ", outpred.shape, outpred2.shape, mouth.shape)
            #outpred2_autocolor = outpred2.clone()
            outpred[background > 0.01] = outpred2[background > 0.01]
            outpred[mouth > 0.01] = outpred2[mouth > 0.01]

            #outpred2_autocolor = torch.reshape(outpred2_autocolor, (1, 256, 256))
            #outpred_autocolor[mouth > 0.1] = outpred2_autocolor[mouth > 0.1]

        outpred = torch.reshape(outpred, (1, 256, 256))

        if parameters["BgExcludeEnableToggle"] and amount_calc != 0:
            if amount_calc > 0:
                r2 = int(amount_calc)
                k2 = 2*r2 + 1
                # Dilatation um Radius r2
                outpred_calc_dill = F.max_pool2d(outpred_calc_dill, kernel_size=k2, stride=1, padding=r2)
                outpred_calc_dill = outpred_calc_dill.clamp(0,1)

            elif amount_calc < 0:
                r2 = int(-amount_calc)
                k2 = 2*r2 + 1
                # Erosion = invertieren → dilatieren → invertieren
                outpred_calc_dill = 1 - outpred_calc_dill
                outpred_calc_dill = F.max_pool2d(outpred_calc_dill, kernel_size=k2, stride=1, padding=r2)
                outpred_calc_dill = 1 - outpred_calc_dill
                outpred_calc_dill = outpred_calc_dill.clamp(0,1)  
                
        return outpred, outpred_calc, outpred_calc_dill

    def run_dfl_xseg(self, image, output):
        if not self.models_processor.models['XSeg']:
            self.models_processor.models['XSeg'] = self.models_processor.load_model('XSeg')

        io_binding = self.models_processor.models['XSeg'].io_binding()
        io_binding.bind_input(name='in_face:0', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=image.size(), buffer_ptr=image.data_ptr())
        io_binding.bind_output(name='out_mask:0', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=(1,1,256,256), buffer_ptr=output.data_ptr())

        if self.models_processor.device == "cuda":
            torch.cuda.synchronize()
        elif self.models_processor.device != "cpu":
            self.models_processor.syncvec.cpu()
        self.models_processor.models['XSeg'].run_with_iobinding(io_binding)

    def process_masks_and_masks(
        self,
        swap_restorecalc: torch.Tensor,
        original_face_512: torch.Tensor,
        parameters: dict
    ) -> dict:
        """
        Kombiniert BG- und FaceParser-Logik mit vereinfachten Masken-OPs.
        Gibt ein dict mit Schlüsseln zurück: swap_formask, BgExclude, BgExcludeOccluder,
        FaceParser_mask, texture_mask, mouth, etc.
        """
        # 1) Resizer
        t512 = v2.Resize((512,512), interpolation=v2.InterpolationMode.NEAREST, antialias=False)
        t256 = v2.Resize((256,256), interpolation=v2.InterpolationMode.NEAREST, antialias=False)
        t128 = v2.Resize((128,128), interpolation=v2.InterpolationMode.NEAREST, antialias=False)

        # 2) Basis-Bild für Masken
        swap_formask = swap_restorecalc
        result = {"swap_formask": swap_formask}

        # 3) Entscheide, ob FaceParser auf swap/orig gebraucht wird
        need_bg      = parameters["DFLXSegEnableToggle"] and parameters["DFLXSegBGEnableToggle"]
        need_parser  = (
            parameters["FaceParserEnableToggle"]
            or (parameters["DFLXSegEnableToggle"]
                and parameters["DFLXSeg2EnableToggle"]
                and parameters["XSegMouthEnableToggle"]
                and parameters["DFLXSegSizeSlider"] != parameters["DFLXSeg2SizeSlider"])
            or ((parameters["TransferTextureEnableToggle"]
                 or parameters["DifferencingEnableToggle"])
                and parameters["ExcludeMaskEnableToggle"])
        )
        #need_bg_orig = parameters["BgExcludeEnableToggle"] and parameters["CalcMaskBGTextureSlider"] != 0
        # helper-Funktion: erzeugt Masken aus Label-Map
        def make_mask(labels, attributes, dil):
            m = torch.isin(labels, torch.tensor(attributes, device=labels.device)).float()
            if abs(dil) > 1:
                dil = abs(dil)
                # invertieren falls negative, dann dilatieren
                inv = (1-m) if dil<0 else m
                inv = inv.unsqueeze(0).unsqueeze(0)
                #for _ in range(abs(dil)):
                #    inv = morph.dilation(inv, kernel=torch.ones((3,3),device=inv.device))
                inv = F.max_pool2d(inv, kernel_size=2*abs(dil-1)+1, padding=abs(dil-1), stride=1)
                m = (1-inv) if dil<0 else inv
                m = m.squeeze(0).squeeze(0)
            return m
    
        # 4) FaceParser on swap_formask, falls nötig
        if need_bg or need_parser: # or need_bg_orig:
            img = swap_formask.float().div(255.0)
            img = v2.functional.normalize(img, (0.485,0.456,0.406), (0.229,0.224,0.225))
            img = img.unsqueeze(0)  # [1,3,512,512]
            out_swap = torch.empty((1,19,512,512), device=self.models_processor.device)
            self.run_faceparser(img, out_swap)
            labels_swap = out_swap.argmax(dim=1).squeeze(0)  # [512,512]

        # 5) FaceParser on original, falls nötig
        if need_parser and (parameters["FaceParserEnableToggle"] or parameters["ExcludeMaskEnableToggle"]): # or need_bg_orig
            img_o = original_face_512.float().div(255.0)
            img_o = v2.functional.normalize(img_o, (0.485,0.456,0.406), (0.229,0.224,0.225))
            img_o = img_o.unsqueeze(0)
            out_orig = torch.empty((1,19,512,512), device=self.models_processor.device)
            self.run_faceparser(img_o, out_orig)
            labels_orig = out_orig.argmax(dim=1).squeeze(0)

        # 6) BG-Exclusion
        '''
        if need_bg_orig:
            attrs = [0,14,15,16,17,18]
            bg_orig = make_mask(labels_swap, attrs,
                                dil=-parameters["CalcMaskBGTextureSlider"])
            result["BgExclude"] = bg_orig.unsqueeze(0)
        '''
        # 7) FaceParser-Logik
        if need_parser:
            # a) Mouth
            mouth = torch.zeros((512,512), device=labels_swap.device)
            mouth_attrs = {
                11: 'XsegMouthParserSlider',
                12: 'XsegUpperLipParserSlider',
                13: 'XsegLowerLipParserSlider'
            }
            for cls, param in mouth_attrs.items():
                d = parameters[param]
                if d:
                    mouth = torch.max(mouth, make_mask(labels_swap, [cls], dil=d))
            result["mouth"] = mouth.clamp(0,1)

            # b) FaceParser-Mask
            if parameters["FaceParserEnableToggle"]:
                face_attrs = {
                    1: 'FaceParserSlider',
                    2: 'LeftEyebrowParserSlider',
                    3: 'RightEyebrowParserSlider',
                    4: 'LeftEyeParserSlider',
                    5: 'RightEyeParserSlider',
                    6: 'EyeGlassesParserSlider',
                    10: 'NoseParserSlider',
                    11: 'MouthParserSlider',
                    12: 'UpperLipParserSlider',
                    13: 'LowerLipParserSlider',
                    14: 'NeckParserSlider',
                    17: 'HairParserSlider'
                }
                mask_fp = torch.zeros((512,512), device=labels_swap.device)
                for cls, param in face_attrs.items():
                    d = parameters[param]
                    if d:
                        # kombiniere swap und orig
                        m1 = make_mask(labels_swap, [cls], dil=d)
                        m2 = make_mask(labels_orig, [cls], dil=0)
                        if parameters['MouthParserInsideToggle'] and cls == 11: #for mouth to be put in the swaped face (to minimize the overlap when the mouth in not aligned)
                            mask_fp = torch.max(mask_fp, torch.min(m1, m2))
                        else:
                            mask_fp = torch.max(mask_fp, torch.max(m1, m2))
                # optional Gaussian-Blur
                if parameters['FaceBlurParserSlider'] > 0:
                    k = parameters['FaceBlurParserSlider']*2+1
                    gauss = transforms.GaussianBlur(k, (parameters['FaceBlurParserSlider']+1)*0.2)
                    mask_fp = gauss(mask_fp.unsqueeze(0).unsqueeze(0)).squeeze()
                mask128 = t128(1 - mask_fp.unsqueeze(0))
                if parameters['FaceParserBlendSlider'] > 0:
                    mask128 = mask128 + parameters['FaceParserBlendSlider']/100
                result["FaceParser_mask"] = mask128.clamp(0,1)

            # c) Texture/Differencing Mask
            if (parameters["TransferTextureEnableToggle"] or parameters["DifferencingEnableToggle"]) \
               and parameters["ExcludeMaskEnableToggle"]:
                tex_attrs = {
                    1: 'FaceParserTextureSlider',
                    2: 'EyebrowParserTextureSlider',
                    3: 'EyebrowParserTextureSlider',
                    4: 'EyeParserTextureSlider',
                    5: 'EyeParserTextureSlider',
                    10: 'NoseParserTextureSlider',
                    11: 'MouthParserTextureSlider',
                    12: 'MouthParserTextureSlider',
                    13: 'MouthParserTextureSlider',
                    14: 'NeckParserTextureSlider'
                }
                tex = torch.zeros((512,512), device=labels_swap.device)
                tex_o = torch.zeros_like(tex)
                for cls, param in tex_attrs.items():
                    d = parameters[param]
                    if cls == 1 and d > 0:
                        blend_fp_tex = parameters["FaceParserTextureSlider"]/10
                        m_swap = torch.isin(labels_swap, torch.tensor(cls, device=labels_swap.device)).float()
                        m_orig = torch.isin(labels_orig, torch.tensor(cls, device=labels_orig.device)).float()
                        m_swap = m_swap * blend_fp_tex
                        m_orig = m_orig * blend_fp_tex
                        tex   = torch.max(tex,   m_swap)
                        tex_o = torch.max(tex_o, m_orig)
                    else:
                        if d:
                            if d > 0:
                                tex   = torch.max(tex,   make_mask(labels_swap, [cls], dil=d))
                                tex_o = torch.max(tex_o, make_mask(labels_orig, [cls], dil=d))
                            if d <= 0:
                                tex_temp = make_mask(labels_swap, [cls], dil=d)
                                tex_o_temp = make_mask(labels_orig, [cls], dil=d)
                                tex_temp = torch.max(tex_temp,   tex_o_temp)
                                #tex_o_temp = torch.max(tex_temp,   tex_o_temp) #tex_temp
                                tex   = tex - tex_temp #torch.max(tex,   make_mask(labels_swap, [cls], dil=d))
                                tex_o = tex_o - tex_temp #torch.max(tex_o, make_mask(labels_orig, [cls], dil=d))
                # optional BG texture exclusion
                if parameters["BackgroundParserTextureSlider"] != 0:
                    tex_o = torch.max(
                        tex_o,
                        make_mask(labels_orig, [0,14,15,16,17,18],
                                  dil=-parameters["BackgroundParserTextureSlider"])
                    )
                comb = torch.min(1-tex.clamp(0,1), 1-tex_o.clamp(0,1))
                result["texture_mask"] = comb.unsqueeze(0).clamp(0,1)

        return result
    # https://github.com/yakhyo/face-parsing

    def run_faceparser(self, image, output):
        if not self.models_processor.models['FaceParser']:
            self.models_processor.models['FaceParser'] = self.models_processor.load_model('FaceParser')

        image = image.contiguous()
        io_binding = self.models_processor.models['FaceParser'].io_binding()
        io_binding.bind_input(name='input', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=(1,3,512,512), buffer_ptr=image.data_ptr())
        io_binding.bind_output(name='output', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=(1,19,512,512), buffer_ptr=output.data_ptr())

        if self.models_processor.device == "cuda":
            torch.cuda.synchronize()
        elif self.models_processor.device != "cpu":
            self.models_processor.syncvec.cpu()
        self.models_processor.models['FaceParser'].run_with_iobinding(io_binding)

    def run_onnx(self, image_tensor, output_tensor, model_key):
        # Modell ggf. laden
        sess = self.models_processor.models.get(model_key)
        if sess is None:
            sess = self.models_processor.load_model(model_key)
            self.models_processor.models[model_key] = sess
        
        image_tensor = image_tensor.contiguous()
        io_binding  = sess.io_binding()

        io_binding.bind_input(
            name        = 'input',
            device_type = self.models_processor.device,
            device_id   = 0,
            element_type= np.float32,
            shape       = image_tensor.shape,
            buffer_ptr  = image_tensor.data_ptr()
        )
        io_binding.bind_output(
            name        = 'features',
            device_type = self.models_processor.device,
            device_id   = 0,
            element_type= np.float32,
            shape       = output_tensor.shape,
            buffer_ptr  = output_tensor.data_ptr()
        )

        if self.models_processor.device == 'cuda':
            torch.cuda.synchronize()
        else:
            self.models_processor.syncvec.cpu()

        sess.run_with_iobinding(io_binding)
        return output_tensor

    def run_CLIPs(self, img, CLIPText, CLIPAmount):
        # Ottieni il dispositivo su cui si trova l'immagine
        device = img.device

        # Controllo se la sessione CLIP è già stata inizializzata
        if not self.models_processor.clip_session:
            self.models_processor.clip_session = CLIPDensePredT(version='ViT-B/16', reduce_dim=64, complex_trans_conv=True)
            self.models_processor.clip_session.eval()
            self.models_processor.clip_session.load_state_dict(torch.load(f'{models_dir}/rd64-uni-refined.pth', weights_only=True), strict=False)
            self.models_processor.clip_session.to(device)  # Sposta il modello sul dispositivo dell'immagine

        # Crea un mask tensor direttamente sul dispositivo dell'immagine
        clip_mask = torch.ones((352, 352), device=device)

        # L'immagine è già un tensore, quindi la converto a float32 e la normalizzo nel range [0, 1]
        img = img.float() / 255.0  # Conversione in float32 e normalizzazione

        # Rimuovi la parte ToTensor(), dato che img è già un tensore.
        transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Resize((352, 352))
        ])

        # Applica la trasformazione all'immagine
        CLIPimg = transform(img).unsqueeze(0).contiguous().to(device)

        # Se ci sono prompt CLIPText, esegui la predizione
        if CLIPText != "":
            prompts = CLIPText.split(',')

            with torch.no_grad():
                # Esegui la predizione sulla sessione CLIP
                preds = self.models_processor.clip_session(CLIPimg.repeat(len(prompts), 1, 1, 1), prompts)[0]

            # Calcola la maschera CLIP usando la sigmoid e tieni tutto sul dispositivo
            clip_mask = 1 - torch.sigmoid(preds[0][0])
            for i in range(len(prompts) - 1):
                clip_mask *= 1 - torch.sigmoid(preds[i + 1][0])

            # Applica la soglia sulla maschera
            thresh = CLIPAmount / 100.0
            clip_mask = (clip_mask > thresh).float()

        return clip_mask.unsqueeze(0)  # Ritorna il tensore torch direttamente

    def soft_oval_mask(self, height, width, center, radius_x, radius_y, feather_radius=None):
        """
        Create a soft oval mask with feathering effect using integer operations.

        Args:
            height (int): Height of the mask.
            width (int): Width of the mask.
            center (tuple): Center of the oval (x, y).
            radius_x (int): Radius of the oval along the x-axis.
            radius_y (int): Radius of the oval along the y-axis.
            feather_radius (int): Radius for feathering effect.

        Returns:
            torch.Tensor: Soft oval mask tensor of shape (H, W).
        """
        if feather_radius is None:
            feather_radius = max(radius_x, radius_y) // 2  # Integer division

        # Calculating the normalized distance from the center
        y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')

        # Calculating the normalized distance from the center
        normalized_distance = torch.sqrt(((x - center[0]) / radius_x) ** 2 + ((y - center[1]) / radius_y) ** 2)

        # Creating the oval mask with a feathering effect
        mask = torch.clamp((1 - normalized_distance) * (radius_x / feather_radius), 0, 1)

        return mask

    def restore_mouth(self, img_orig, img_swap, kpss_orig, blend_alpha=0.5, feather_radius=10, size_factor=0.5, radius_factor_x=1.0, radius_factor_y=1.0, x_offset=0, y_offset=0):
        """
        Extract mouth from img_orig using the provided keypoints and place it in img_swap.

        Args:
            img_orig (torch.Tensor): The original image tensor of shape (C, H, W) from which mouth is extracted.
            img_swap (torch.Tensor): The target image tensor of shape (C, H, W) where mouth is placed.
            kpss_orig (list): List of keypoints arrays for detected faces. Each keypoints array contains coordinates for 5 keypoints.
            radius_factor_x (float): Factor to scale the horizontal radius. 1.0 means circular, >1.0 means wider oval, <1.0 means narrower.
            radius_factor_y (float): Factor to scale the vertical radius. 1.0 means circular, >1.0 means taller oval, <1.0 means shorter.
            x_offset (int): Horizontal offset for shifting the mouth left (negative value) or right (positive value).
            y_offset (int): Vertical offset for shifting the mouth up (negative value) or down (positive value).

        Returns:
            torch.Tensor: The resulting image tensor with mouth from img_orig placed on img_swap.
        """
        left_mouth = np.array([int(val) for val in kpss_orig[3]])
        right_mouth = np.array([int(val) for val in kpss_orig[4]])

        mouth_center = (left_mouth + right_mouth) // 2
        mouth_base_radius = int(np.linalg.norm(left_mouth - right_mouth) * size_factor)

        # Calculate the scaled radii
        radius_x = int(mouth_base_radius * radius_factor_x)
        radius_y = int(mouth_base_radius * radius_factor_y)

        # Apply the x/y_offset to the mouth center
        mouth_center[0] += x_offset
        mouth_center[1] += y_offset

        # Calculate bounding box for mouth region
        ymin = max(0, mouth_center[1] - radius_y)
        ymax = min(img_orig.size(1), mouth_center[1] + radius_y)
        xmin = max(0, mouth_center[0] - radius_x)
        xmax = min(img_orig.size(2), mouth_center[0] + radius_x)

        mouth_region_orig = img_orig[:, ymin:ymax, xmin:xmax]
        mouth_mask = self.soft_oval_mask(ymax - ymin, xmax - xmin,
                                            (radius_x, radius_y),
                                            radius_x, radius_y,
                                            feather_radius).to(img_orig.device)

        target_ymin = ymin
        target_ymax = ymin + mouth_region_orig.size(1)
        target_xmin = xmin
        target_xmax = xmin + mouth_region_orig.size(2)

        img_swap_mouth = img_swap[:, target_ymin:target_ymax, target_xmin:target_xmax]
        blended_mouth = blend_alpha * img_swap_mouth + (1 - blend_alpha) * mouth_region_orig

        img_swap[:, target_ymin:target_ymax, target_xmin:target_xmax] = mouth_mask * blended_mouth + (1 - mouth_mask) * img_swap_mouth
        return img_swap

    def restore_eyes(self, img_orig, img_swap, kpss_orig, blend_alpha=0.5, feather_radius=10, size_factor=3.5, radius_factor_x=1.0, radius_factor_y=1.0, x_offset=0, y_offset=0, eye_spacing_offset=0):
        """
        Extract eyes from img_orig using the provided keypoints and place them in img_swap.

        Args:
            img_orig (torch.Tensor): The original image tensor of shape (C, H, W) from which eyes are extracted.
            img_swap (torch.Tensor): The target image tensor of shape (C, H, W) where eyes are placed.
            kpss_orig (list): List of keypoints arrays for detected faces. Each keypoints array contains coordinates for 5 keypoints.
            radius_factor_x (float): Factor to scale the horizontal radius. 1.0 means circular, >1.0 means wider oval, <1.0 means narrower.
            radius_factor_y (float): Factor to scale the vertical radius. 1.0 means circular, >1.0 means taller oval, <1.0 means shorter.
            x_offset (int): Horizontal offset for shifting the eyes left (negative value) or right (positive value).
            y_offset (int): Vertical offset for shifting the eyes up (negative value) or down (positive value).
            eye_spacing_offset (int): Horizontal offset to move eyes closer together (negative value) or farther apart (positive value).

        Returns:
            torch.Tensor: The resulting image tensor with eyes from img_orig placed on img_swap.
        """
        # Extract original keypoints for left and right eye
        left_eye = np.array([int(val) for val in kpss_orig[0]])
        right_eye = np.array([int(val) for val in kpss_orig[1]])

        # Apply horizontal offset (x-axis)
        left_eye[0] += x_offset
        right_eye[0] += x_offset

        # Apply vertical offset (y-axis)
        left_eye[1] += y_offset
        right_eye[1] += y_offset

        # Calculate eye distance and radii
        eye_distance = np.linalg.norm(left_eye - right_eye)
        base_eye_radius = int(eye_distance / size_factor)

        # Calculate the scaled radii
        radius_x = int(base_eye_radius * radius_factor_x)
        radius_y = int(base_eye_radius * radius_factor_y)

        # Adjust for eye spacing (horizontal movement)
        left_eye[0] += eye_spacing_offset
        right_eye[0] -= eye_spacing_offset

        def extract_and_blend_eye(eye_center, radius_x, radius_y, img_orig, img_swap, blend_alpha, feather_radius):
            ymin = max(0, eye_center[1] - radius_y)
            ymax = min(img_orig.size(1), eye_center[1] + radius_y)
            xmin = max(0, eye_center[0] - radius_x)
            xmax = min(img_orig.size(2), eye_center[0] + radius_x)

            eye_region_orig = img_orig[:, ymin:ymax, xmin:xmax]
            eye_mask = self.soft_oval_mask(ymax - ymin, xmax - xmin,
                                            (radius_x, radius_y),
                                            radius_x, radius_y,
                                            feather_radius).to(img_orig.device)

            target_ymin = ymin
            target_ymax = ymin + eye_region_orig.size(1)
            target_xmin = xmin
            target_xmax = xmin + eye_region_orig.size(2)

            img_swap_eye = img_swap[:, target_ymin:target_ymax, target_xmin:target_xmax]
            blended_eye = blend_alpha * img_swap_eye + (1 - blend_alpha) * eye_region_orig

            img_swap[:, target_ymin:target_ymax, target_xmin:target_xmax] = eye_mask * blended_eye + (1 - eye_mask) * img_swap_eye

        # Process both eyes with updated positions
        extract_and_blend_eye(left_eye, radius_x, radius_y, img_orig, img_swap, blend_alpha, feather_radius)
        extract_and_blend_eye(right_eye, radius_x, radius_y, img_orig, img_swap, blend_alpha, feather_radius)

        return img_swap

    def apply_fake_diff(self, swapped_face, original_face, lower_thresh, lower_value, upper_thresh, upper_value, middle_value, parameters):
        # Kein permute nötig → [3, H, W]
        
        diff = torch.abs(swapped_face - original_face)
                    
        # Quantile (auf allen Kanälen)
        sample = diff.reshape(-1)
        sample = sample[torch.randint(0, sample.numel(), (50_000,), device=diff.device)]
        diff_max = torch.quantile(sample, 0.99)
        diff = torch.clamp(diff, max=diff_max)

        diff_min = diff.min()
        diff_max = diff.max()
        diff_norm = (diff - diff_min) / (diff_max - diff_min)

        diff_mean = diff_norm.mean(dim=0)  # [H, W]
        # Direkt mit torch.where statt vielen Masken
        scale = diff_mean / lower_thresh
        result = torch.where(
            diff_mean < lower_thresh,
            lower_value + scale * (middle_value - lower_value),
            torch.empty_like(diff_mean)
        )

        middle_scale = (diff_mean - lower_thresh) / (upper_thresh - lower_thresh)
        result = torch.where(
            (diff_mean >= lower_thresh) & (diff_mean <= upper_thresh),
            middle_value + middle_scale * (upper_value - middle_value),
            result
        )

        above_scale = (diff_mean - upper_thresh) / (1 - upper_thresh)
        result = torch.where(
            diff_mean > upper_thresh,
            upper_value + above_scale * (1 - upper_value),
            result
        )

        return result.unsqueeze(0)  # (1, H, W)
    
    def apply_perceptual_diff_onnx(self,
            swapped_face, original_face, swap_mask,
            lower_thresh, lower_value,
            upper_thresh, upper_value, middle_value,
            feature_layer, ExcludeVGGMaskEnableToggle):
        # ### 1) Channels & Shape je Backbone/Layer definieren ###
        feature_shapes = {
            # VGG16
            'relu2_2':               (1, 128, 128, 128),
            'relu3_1':               (1, 256, 128, 128),
            'relu3_3':               (1, 256, 128, 128),
            'relu4_1':               (1, 512, 128, 128),
            'combo_relu3_3_relu2_2': (1, 384, 128, 128),
            'combo_relu3_3_relu3_1': (1, 512, 128, 128),
            # EfficientNet-B0 (Layer 2 = C=24, Layer 3 = C=40, Layer 4 = C=80)
            'efficientnetb0_layer2': (1, 24, 128, 128),
            'efficientnetb0_layer3': (1, 40, 128, 128),
            'efficientnetb0_layer4': (1, 80, 128, 128),
        }

        # ### 2) Modell laden (oder aus Cache ziehen) ###
        model_key = feature_layer
        if model_key not in self.models_processor.models:
            # load_model erwartet nun exakt den gleichen string wie in models_data
            self.models_processor.models[model_key] = self.models_processor.load_model(model_key)
        model = self.models_processor.models[model_key]

        # ### 3) Preprocessing (identisch für alle Backbones) ###
        def preprocess(img):
            img = img.clone().float() / 255.0
            mean = torch.tensor([0.485,0.456,0.406], device=img.device).view(3,1,1)
            std  = torch.tensor([0.229,0.224,0.225], device=img.device).view(3,1,1)
            return ((img - mean) / std).unsqueeze(0).contiguous()  # [1,3,H,W]

        swapped  = preprocess(swapped_face)
        original = preprocess(original_face)

        # ### 4) Ausgabe-Puffer in der richtigen Form anlegen ###
        shape   = feature_shapes[feature_layer]
        outpred = torch.empty(shape,   dtype=torch.float32, device=swapped.device)
        outpred2= torch.empty_like(outpred)

        # ### 5) Onnx-Inferenz ###
        swapped_feat  = self.run_onnx(swapped,  outpred,  model_key)
        original_feat = self.run_onnx(original, outpred2, model_key)

        # ### 6) Diff + Masking + Remapping wie gehabt ###
        diff_map = torch.abs(swapped_feat - original_feat).mean(dim=1)[0]   # [128,128]

        diff_map = diff_map * swap_mask.squeeze(0)

        # Quantile clipping
        sample   = diff_map.reshape(-1)
        sample   = sample[torch.randint(0, diff_map.numel(), (50_000,), device=diff_map.device)]
        diff_max = torch.quantile(sample, 0.99)
        diff_map = torch.clamp(diff_map, max=diff_max)

        # 1) Normalisierung
        diff_min, diff_max = diff_map.amin(), diff_map.amax()
        diff_norm = (diff_map - diff_min) / (diff_max - diff_min + 1e-6)
        # (falls du diff_norm_texture wirklich separat brauchst, klon hier einmal:)
        diff_norm_texture = diff_norm.clone()
        if ExcludeVGGMaskEnableToggle:        
            eps = 1e-6
            # 2) Precompute Inverse-Bereiche (vermeidet Divisions-Op in jedem Pixel)
            inv_lower = 1.0 / max(lower_thresh, eps)
            inv_mid   = 1.0 / max((upper_thresh - lower_thresh), eps)
            inv_high  = 1.0 / max((1.0 - upper_thresh), eps)


            # 3) die drei linearen Ausdrücke
            res_low  = lower_value  + diff_norm * inv_lower * (middle_value - lower_value)
            res_mid  = middle_value + (diff_norm - lower_thresh) * inv_mid   * (upper_value  - middle_value)
            res_high = upper_value  + (diff_norm - upper_thresh) * inv_high * (1.0        - upper_value)

            # 4) nur zwei where-Schritte statt drei
            result = torch.where(
                diff_norm < lower_thresh,
                res_low,
                torch.where(
                    diff_norm > upper_thresh,
                    res_high,
                    res_mid
                )
            )
        else:
            result = diff_norm

        return result.unsqueeze(0), diff_norm_texture.unsqueeze(0)