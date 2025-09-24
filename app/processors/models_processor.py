import threading
import os
import subprocess as sp
import gc
import traceback
from typing import Dict, TYPE_CHECKING, Optional

from packaging import version
import numpy as np
import onnxruntime
import torch
import onnx
from torchvision.transforms import v2

from app.processors.utils import faceutil

# --- NEW KORNIA IMPORT ---
try:
    import kornia.color as K
except ImportError:
    K = None # Fallback if Kornia is not installed, can add error handling or power-law
    print("Warning: Kornia library not found. Color space conversions will use power-law approximation.")
# --- END NEW KORNIA IMPORT ---
from PySide6 import QtCore
try:
    import tensorrt as trt
    TENSORRT_AVAILABLE = True
except ModuleNotFoundError:
    print("No TensorRT Found")
    TENSORRT_AVAILABLE = False

from app.processors.utils.engine_builder import onnx_to_trt as onnx2trt
from app.processors.utils.tensorrt_predictor import TensorRTPredictor
from app.processors.face_detectors import FaceDetectors
from app.processors.face_landmark_detectors import FaceLandmarkDetectors
from app.processors.face_masks import FaceMasks
from app.processors.face_restorers import FaceRestorers
from app.processors.face_swappers import FaceSwappers
from app.processors.frame_enhancers import FrameEnhancers
from app.processors.face_editors import FaceEditors
from app.processors.utils.dfm_model import DFMModel
from app.processors.models_data import models_list, arcface_mapping_model_dict, models_trt_list
from app.helpers.miscellaneous import is_file_exists
from app.helpers.downloader import download_file
from app.processors.utils.ref_ldm_kv_embedding import KVExtractor

if TYPE_CHECKING:
    from app.ui.main_ui import MainWindow

onnxruntime.set_default_logger_severity(4)
onnxruntime.log_verbosity_level = -1
lock = threading.Lock()

SRGB_GAMMA = 2.2 # More precise sRGB gamma handling is complex, this is an approximation

def gamma_encode_linear_rgb_to_srgb(linear_rgb: torch.Tensor, gamma=SRGB_GAMMA):
    # Assumes input is float in [0,1], linear RGB
    # linear_rgb -> srgb
    return torch.pow(linear_rgb.clamp(0.0, 1.0), 1.0 / gamma)

def gamma_decode_srgb_to_linear_rgb(srgb: torch.Tensor, gamma=SRGB_GAMMA):
    # Assumes input is float in [0,1], sRGB
    # srgb -> linear_rgb
    return torch.pow(srgb.clamp(0.0, 1.0), gamma)

class ModelsProcessor(QtCore.QObject):
    processing_complete = QtCore.Signal()
    model_loaded = QtCore.Signal()  # Signal emitted with Onnx InferenceSession

    @staticmethod
    def print_tensor_stats(tensor: torch.Tensor, name: str, enabled: bool = True):
        if not enabled:
            return
        if isinstance(tensor, torch.Tensor):
            # Cast to float for mean and std calculation if tensor is uint8
            if tensor.dtype == torch.uint8:
                tensor_float = tensor.float() / 255.0 # Normalize for meaningful stats
                print(f"DEBUG DENOISER STATS for {name}: shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device}, min={tensor.min().item():.4f}, max={tensor.max().item():.4f}, mean={tensor_float.mean().item():.4f}, std={tensor_float.std().item():.4f} (stats on [0,1] float)")
            elif tensor.dtype == torch.float16 or tensor.dtype == torch.float32:
                print(f"DEBUG DENOISER STATS for {name}: shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device}, min={tensor.min().item():.4f}, max={tensor.max().item():.4f}, mean={tensor.mean().item():.4f}, std={tensor.std().item():.4f}")
            else:
                 print(f"DEBUG DENOISER STATS for {name}: shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device} (stats not computed for this dtype)")
        else:
            print(f"DEBUG DENOISER STATS for {name}: Not a tensor, type is {type(tensor)}")

    # --- START: Functions integrated from ldm.modules.diffusionmodules.util ---
    @staticmethod
    def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3) -> np.ndarray:
        if schedule == "linear":
            betas = (
                    torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
            )
        elif schedule == "cosine":
            timesteps = (
                    torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
            )
            alphas = timesteps / (1 + cosine_s) * np.pi / 2 # type: ignore
            alphas = torch.cos(alphas).pow(2)
            alphas = alphas / alphas[0]
            betas = 1 - alphas[1:] / alphas[:-1]
            betas = np.clip(betas.numpy(), a_min=0, a_max=0.999) # type: ignore
        elif schedule == "sqrt_linear":
            betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64)
        elif schedule == "sqrt": # Not used by ref-ldm
            betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64) ** 0.5
        else:
            raise ValueError(f"schedule '{schedule}' unknown.")
        return betas.numpy() if isinstance(betas, torch.Tensor) else betas


    @staticmethod
    def make_ddim_timesteps(ddim_discr_method: str, num_ddim_timesteps: int, num_ddpm_timesteps: int, verbose: bool = True) -> np.ndarray:
        if ddim_discr_method == 'uniform':
            c = num_ddpm_timesteps // num_ddim_timesteps
            if c == 0: c = 1 # Avoid division by zero or c=0 for small num_ddpm_timesteps
            ddim_timesteps = np.asarray(list(range(0, num_ddpm_timesteps, c)))
        elif ddim_discr_method == 'uniform_trailing':
            c = num_ddpm_timesteps // num_ddim_timesteps
            if c == 0: c = 1
            ddim_timesteps = np.arange(num_ddpm_timesteps, 0, -c).astype(int)[::-1] - 2 # Match LDM util
            ddim_timesteps = np.clip(ddim_timesteps, 0, num_ddpm_timesteps - 1) # Ensure valid range
        elif ddim_discr_method == 'quad':
            ddim_timesteps = ((np.linspace(0, np.sqrt(num_ddpm_timesteps * .8), num_ddim_timesteps)) ** 2).astype(int)
        else:
            raise NotImplementedError(f'There is no ddim discretization method called "{ddim_discr_method}"')

        steps_out = np.unique(ddim_timesteps)
        steps_out.sort()

        if verbose:
            print(f'Selected DDPM timesteps for DDIM sampler (0-indexed): {steps_out}')
        return steps_out

    @staticmethod
    def make_ddim_sampling_parameters(alphacums: np.ndarray, ddim_timesteps: np.ndarray, eta: float, verbose: bool=True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        _prev_t = np.concatenate(([-1], ddim_timesteps[:-1])) # Use -1 to signify "before first step"
        _alphas_prev = np.array([alphacums[pt] if pt != -1 else 1.0 for pt in _prev_t])
        _alphas = alphacums[ddim_timesteps]
        sigmas = eta * np.sqrt((1 - _alphas_prev) / (1 - _alphas) * (1 - _alphas / _alphas_prev))
        sigmas = np.nan_to_num(sigmas, nan=0.0)
        return sigmas, _alphas, _alphas_prev

    def __init__(self, main_window: 'MainWindow', device='cuda'):
        super().__init__()
        self.main_window = main_window
        self.K = K # Assign the module-level K to an instance attribute
        self.provider_name = 'TensorRT'
        self.internal_deep_copied_kv_map: Dict[str, Dict[str, torch.Tensor]] | None = None
        self.internal_kv_map_source_filename: str | None = None
        self.kv_extractor: Optional[KVExtractor] = None
        self.device = device
        self.model_lock = threading.RLock()  # Reentrant lock for model access
        self.trt_ep_options = {
            # 'trt_max_workspace_size': 3 << 30,  # Dimensione massima dello spazio di lavoro in bytes
            'trt_engine_cache_enable': True,
            'trt_engine_cache_path': "tensorrt-engines",
            'trt_timing_cache_enable': True,
            'trt_timing_cache_path': "tensorrt-engines",
            'trt_dump_ep_context_model': True,
            'trt_ep_context_file_path': "tensorrt-engines",
            'trt_layer_norm_fp32_fallback': True,
            'trt_builder_optimization_level': 5,
        }
        self.providers = [
            ('CUDAExecutionProvider'),
            ('CPUExecutionProvider')
        ]       
        self.syncvec = torch.empty((1, 1), dtype=torch.float32, device=self.device)
        self.nThreads = 1

        # Initialize models and models_path
        self.models: Dict[str, onnxruntime.InferenceSession] = {}
        self.models_path = {}
        self.models_data = {}
        for model_data in models_list:
            model_name, model_path = model_data['model_name'], model_data['local_path']
            self.models[model_name] = None #Model Instance
            self.models_path[model_name] = model_path
            self.models_data[model_name] = {'local_path': model_data['local_path'], 'hash': model_data['hash'], 'url': model_data.get('url')}

        self.dfm_models: Dict[str, DFMModel] = {}

        if TENSORRT_AVAILABLE:
            # Initialize models_trt and models_trt_path
            self.models_trt = {}
            self.models_trt_path = {}
            for model_data in models_trt_list:
                model_name, model_path = model_data['model_name'], model_data['local_path']
                self.models_trt[model_name] = None #Model Instance
                self.models_trt_path[model_name] = model_path

        self.face_detectors = FaceDetectors(self)
        self.face_landmark_detectors = FaceLandmarkDetectors(self)
        self.face_masks = FaceMasks(self)
        self.face_restorers = FaceRestorers(self)
        self.face_swappers = FaceSwappers(self)
        self.frame_enhancers = FrameEnhancers(self)
        self.face_editors = FaceEditors(self)

        self.lp_mask_crop_latent = faceutil.create_faded_inner_mask(size=(64, 64), border_thickness=3, fade_thickness=8, blur_radius=3, device=self.device)
        self.lp_mask_crop_latent = torch.unsqueeze(self.lp_mask_crop_latent, 0) # Shape: [1, 64, 64]

        # Denoiser specific initializations (from VR180 feature, which is correct for the implementation)
        num_ddpm_timesteps = 1000
        linear_start_val = 0.0015 
        linear_end_val = 0.0155   
        self.betas_np = ModelsProcessor.make_beta_schedule(
            schedule="linear", 
            n_timestep=num_ddpm_timesteps,
            linear_start=linear_start_val, 
            linear_end=linear_end_val    
        )
        self.alphas_np = 1.0 - self.betas_np
        self.alphas_cumprod_np = np.cumprod(self.alphas_np, axis=0)
        self.alphas_cumprod_torch = torch.from_numpy(self.alphas_cumprod_np).float().to(self.device)
        self.vae_scale_factor = 1.0 # Confirmed by user

        self.clip_session = []
        self.arcface_dst = np.array( [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)
        self.FFHQ_kps = np.array([[ 192.98138, 239.94708 ], [ 318.90277, 240.1936 ], [ 256.63416, 314.01935 ], [ 201.26117, 371.41043 ], [ 313.08905, 371.15118 ] ])
        self.mean_lmk = []
        self.anchors  = []
        self.emap = []
        self.LandmarksSubsetIdxs = [
            0, 1, 4, 5, 6, 7, 8, 10, 13, 14, 17, 21, 33, 37, 39,
            40, 46, 52, 53, 54, 55, 58, 61, 63, 65, 66, 67, 70, 78, 80,
            81, 82, 84, 87, 88, 91, 93, 95, 103, 105, 107, 109, 127, 132, 133,
            136, 144, 145, 146, 148, 149, 150, 152, 153, 154, 155, 157, 158, 159, 160,
            161, 162, 163, 168, 172, 173, 176, 178, 181, 185, 191, 195, 197, 234, 246,
            249, 251, 263, 267, 269, 270, 276, 282, 283, 284, 285, 288, 291, 293, 295,
            296, 297, 300, 308, 310, 311, 312, 314, 317, 318, 321, 323, 324, 332, 334,
            336, 338, 356, 361, 362, 365, 373, 374, 375, 377, 378, 379, 380, 381, 382,
            384, 385, 386, 387, 388, 389, 390, 397, 398, 400, 402, 405, 409, 415, 454,
            466, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477
        ]

        self.normalize = v2.Normalize(mean = [ 0., 0., 0. ],
                                      std = [ 1/1.0, 1/1.0, 1/1.0 ])
        
        self.lp_mask_crop = self.face_editors.lp_mask_crop
        self.lp_lip_array = self.face_editors.lp_lip_array
        self.rgb_to_linear_rgb_converter = None
        self.linear_rgb_to_rgb_converter = None

    def load_model(self, model_name, session_options=None):
        with self.model_lock:
            if self.models.get(model_name):
                return self.models[model_name]

            self.main_window.model_loading_signal.emit()
            try:
                if session_options is None:
                    model_instance = onnxruntime.InferenceSession(self.models_path[model_name], providers=self.providers)
                else:
                    model_instance = onnxruntime.InferenceSession(self.models_path[model_name], sess_options=session_options, providers=self.providers)

                # Race condition check: another thread might have loaded it while this one was creating the instance
                if self.models.get(model_name):
                    del model_instance
                    gc.collect()
                    return self.models.get(model_name)
                
                self.models[model_name] = model_instance
                return model_instance
            finally:
                self.main_window.model_loaded_signal.emit()

    def load_dfm_model(self, dfm_model):
        with self.model_lock:
            if not self.dfm_models.get(dfm_model):
                self.main_window.model_loading_signal.emit()
                max_models_to_keep = self.main_window.control['MaxDFMModelsSlider']
                total_loaded_models = len(self.dfm_models)
                if total_loaded_models==max_models_to_keep:
                    print("Clearing DFM Model")
                    model_name, model_instance = list(self.dfm_models.items())[0]
                    del model_instance
                    self.dfm_models.pop(model_name)
                    gc.collect()
                try:
                    self.dfm_models[dfm_model] = DFMModel(self.main_window.dfm_models_data[dfm_model], self.providers, self.device)
                except:
                    traceback.print_exc()   
                    self.dfm_models[dfm_model] = None         
                self.main_window.model_loaded_signal.emit()
            return self.dfm_models[dfm_model]


    def load_model_trt(self, model_name, custom_plugin_path=None, precision='fp16', debug=False):
        # self.showModelLoadingProgressBar()
        #time.sleep(0.5)
        self.main_window.model_loading_signal.emit()

        if not os.path.exists(self.models_trt_path[model_name]):
            onnx2trt(onnx_model_path=self.models_path[model_name],
                     trt_model_path=self.models_trt_path[model_name],
                     precision=precision,
                     custom_plugin_path=custom_plugin_path,
                     verbose=False
                    )
        model_instance = TensorRTPredictor(model_path=self.models_trt_path[model_name], custom_plugin_path=custom_plugin_path, pool_size=self.nThreads, device=self.device, debug=debug)

        self.main_window.model_loaded_signal.emit()
        return model_instance

    def delete_models(self):
        for model_name, model_instance in self.models.items():
            del model_instance
            self.models[model_name] = None
        self.clip_session = []
        gc.collect()

    def delete_models_trt(self):
        if TENSORRT_AVAILABLE:
            for model_data in models_trt_list:
                model_name = model_data['model_name']
                if isinstance(self.models_trt[model_name], TensorRTPredictor):
                    # È un'istanza di TensorRTPredictor
                    self.models_trt[model_name].cleanup()
                    del self.models_trt[model_name]
                    self.models_trt[model_name] = None #Model Instance
            gc.collect()

    def delete_models_dfm(self):
        keys_to_remove = []
        for model_name, model_instance in self.dfm_models.items():
            del model_instance
            keys_to_remove.append(model_name)
        
        for model_name in keys_to_remove:
            self.dfm_models.pop(model_name)
        
        self.clip_session = []
        gc.collect()

    def unload_model(self, model_name_to_unload):
        with self.model_lock:
            if model_name_to_unload in self.models and self.models.get(model_name_to_unload) is not None:
                print(f"Unloading model: {model_name_to_unload}")
                del self.models[model_name_to_unload]
                self.models[model_name_to_unload] = None
                gc.collect()
                torch.cuda.empty_cache()
            else:
                print(f"Model '{model_name_to_unload}' not found or already unloaded.")

    def showModelLoadingProgressBar(self):
        self.main_window.model_load_dialog.show()

    def hideModelLoadProgressBar(self):
        if self.main_window.model_load_dialog:
            self.main_window.model_load_dialog.close()

    def switch_providers_priority(self, provider_name):
        match provider_name:
            case "TensorRT" | "TensorRT-Engine":
                providers = [
                                ('TensorrtExecutionProvider', self.trt_ep_options),
                                ('CUDAExecutionProvider'),
                                ('CPUExecutionProvider')
                            ]
                self.device = 'cuda'
                if version.parse(trt.__version__) < version.parse("10.2.0") and provider_name == "TensorRT-Engine":
                    print("TensorRT-Engine provider cannot be used when TensorRT version is lower than 10.2.0.")
                    provider_name = "TensorRT"

            case "CPU":
                providers = [
                                ('CPUExecutionProvider')
                            ]
                self.device = 'cpu'
            case "CUDA":
                providers = [
                                ('CUDAExecutionProvider'),
                                ('CPUExecutionProvider')
                            ]
                self.device = 'cuda'
            #case _:

        self.providers = providers
        self.provider_name = provider_name
        self.lp_mask_crop = self.lp_mask_crop.to(self.device)

        return self.provider_name

    def set_number_of_threads(self, value):
        self.nThreads = value
        self.delete_models_trt()

    def get_gpu_memory(self):
        command = "nvidia-smi --query-gpu=memory.total --format=csv"
        memory_total_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
        memory_total = [int(x.split()[0]) for i, x in enumerate(memory_total_info)]

        command = "nvidia-smi --query-gpu=memory.free --format=csv"
        memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
        memory_free = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]

        memory_used = memory_total[0] - memory_free[0]

        return memory_used, memory_total[0]
    
    def clear_gpu_memory(self):
        self.delete_models()
        self.delete_models_dfm()
        self.delete_models_trt()
        torch.cuda.empty_cache()

    def ensure_kv_extractor_loaded(self):
        if self.kv_extractor is None:
            try:
                print("Loading KV Extractor...")

                base_path = "model_assets/ref-ldm_embedding"
                configs_path = os.path.join(base_path, "configs")
                ckpts_path = os.path.join(base_path, "ckpts")
                os.makedirs(configs_path, exist_ok=True)
                os.makedirs(ckpts_path, exist_ok=True)

                ref_ldm_files = {
                    "configs/ldm.yaml": "https://raw.githubusercontent.com/Glat0s/ref-ldm-onnx/slim-fast/configs/ldm.yaml",
                    "configs/refldm.yaml": "https://raw.githubusercontent.com/Glat0s/ref-ldm-onnx/slim-fast/configs/refldm.yaml",
                    "configs/vqgan.yaml": "https://raw.githubusercontent.com/Glat0s/ref-ldm-onnx/slim-fast/configs/vqgan.yaml",
                    "ckpts/refldm.ckpt": "https://github.com/ChiWeiHsiao/ref-ldm/releases/download/1.0.0/refldm.ckpt",
                    "ckpts/vqgan.ckpt": "https://github.com/ChiWeiHsiao/ref-ldm/releases/download/1.0.0/vqgan.ckpt",
                }

                for rel_path, url in ref_ldm_files.items():
                    full_path = os.path.join(base_path, rel_path)
                    if not is_file_exists(full_path):
                        print(f"Downloading ReF-LDM file: {os.path.basename(full_path)}...")
                        download_file(os.path.basename(full_path), full_path, None, url)

                config_path = os.path.join(configs_path, "refldm.yaml")
                model_path = os.path.join(ckpts_path, "refldm.ckpt")
                vae_path = os.path.join(ckpts_path, "vqgan.ckpt")

                if not all(os.path.exists(p) for p in [config_path, model_path, vae_path]):
                    print("ReF-LDM model files not found even after download attempt. Cannot load KV Extractor.")
                    return

                self.kv_extractor = KVExtractor(
                    model_config_path=config_path,
                    model_ckpt_path=model_path,
                    vae_ckpt_path=vae_path,
                    device=self.device
                )
                print("KV Extractor loaded.")
            except Exception as e:
                print(f"Failed to load KV Extractor: {e}")
                traceback.print_exc()
                self.kv_extractor = None

    def ensure_denoiser_models_loaded(self):
        """Loads the UNet and VAE models if they are not already loaded."""
        with self.model_lock: # Ensure thread safety
            #print("Ensuring denoiser models (UNet, VAEs) are loaded...")
            unet_model_name = self.main_window.fixed_unet_model_name
            vae_encoder_name = 'RefLDMVAEEncoder'
            vae_decoder_name = 'RefLDMVAEDecoder'

            if not self.models.get(unet_model_name): # Use .get() for safety
                #print(f"  Loading UNet model: {unet_model_name}")
                self.models[unet_model_name] = self.load_model(unet_model_name)
            # else:
                # print(f"  UNet model '{unet_model_name}' already loaded.")

            if not self.models.get(vae_encoder_name):
                #print(f"  Loading VAE Encoder model: {vae_encoder_name}")
                self.models[vae_encoder_name] = self.load_model(vae_encoder_name)
            # else:
                # print(f"  VAE Encoder model '{vae_encoder_name}' already loaded.")

            if not self.models.get(vae_decoder_name):
                #print(f"  Loading VAE Decoder model: {vae_decoder_name}")
                self.models[vae_decoder_name] = self.load_model(vae_decoder_name)
            # else:
                # print(f"  VAE Decoder model '{vae_decoder_name}' already loaded.")
            #print("Denoiser models loading check complete.")

    def unload_denoiser_models(self):
        """Unloads the UNet and VAE models."""
        with self.model_lock: # Ensure thread safety
            print("Unloading denoiser models (UNet, VAEs)...")
            self.unload_model(self.main_window.fixed_unet_model_name)
            self.unload_model('RefLDMVAEEncoder')
            self.unload_model('RefLDMVAEDecoder')
            print("Denoiser models unloaded.")

    def unload_kv_extractor(self):
        """Unloads the KVExtractor model and clears associated memory."""
        with self.model_lock:
            if self.kv_extractor is not None:
                print("Unloading KV Extractor...")
                del self.kv_extractor
                self.kv_extractor = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print("KV Extractor unloaded.")

    def apply_vgg_mask_simple(
        self,
        swapped_face: torch.Tensor,          # [3,512,512] uint8
        original_face: torch.Tensor,         # [3,512,512] uint8
        swap_mask_128: torch.Tensor,         # [1,128,128] float mask (0..1)
        center_pct: float,                    # 0..100, z.B. parameters['VGGMaskThresholdSlider']
        softness_pct: float,                  # 0..100, z.B. parameters['VGGMaskSoftnessSlider']
        feature_layer: str = 'combo_relu3_3_relu3_1',
        mode: str = 'smooth'                  # 'smooth' (smoothstep) oder 'linear'
    ):
        """
        Liefert:
          mask_vgg: [1,512,512] float 0..1 (weiche Diff-Maske)
          diff_norm_texture: [1,128,128] float 0..1 (normalisierte Roh-Diff in 128er Auflösung)
        """
        # 1) Roh-Diff per bestehender ONNX-Pipeline in 128x128 holen (ohne Mapping):
        #    Wir nutzen apply_perceptual_diff_onnx mit „Durchreich“-Modus (ExcludeVGGMaskEnableToggle=False),
        #    und ignorieren die komplexen Threshold-Parameter (werden unten ersetzt).
        dummy_lower = 0.0
        dummy_upper = 1.0
        dummy_upper_v = 1.0
        dummy_middle_v = 0.5

        diff_mapped_128, diff_norm_128 = self.apply_perceptual_diff_onnx(
            swapped_face, original_face, swap_mask_128,
            dummy_lower, 0.0, dummy_upper, dummy_upper_v, dummy_middle_v,
            feature_layer, ExcludeVGGMaskEnableToggle=False
        )
        # diff_norm_128: [1,128,128] in [0..1]
        d = diff_mapped_128.squeeze(0)  # [128,128]

        # 2) Two-slider-Mapping -> untere/obere Schwelle aus (center,softness)
        center   = float(center_pct)   / 100.0   # 0..1
        softness = float(softness_pct) / 100.0   # 0..1

        # Breite des Übergangsbandes (angenehme Praxis-Werte):
        #  - Mindestbreite 0.04, Maxbreite 0.40 (gefühlt gut kontrollierbar)
        band = 0.04 + 0.36 * softness
        lo = max(0.0, center - band * 0.5)
        hi = min(1.0, center + band * 0.5)

        # 3) Kurvenform
        x = (d - lo) / max(1e-6, (hi - lo))
        x = x.clamp(0.0, 1.0)
        if mode == 'smooth':
            # Smoothstep
            x = x * x * (3.0 - 2.0 * x)
        # else: 'linear' -> x bleibt linear

        # 4) Auf 512x512 hochskalieren (bilinear) – du arbeitest später meist in 512
        x_512 = torch.nn.functional.interpolate(
            x.unsqueeze(0).unsqueeze(0),
            size=(512, 512),
            mode='bilinear',
            align_corners=True
        ).squeeze(0)

        return x_512.clamp(0,1), diff_norm_128  # ([1,512,512], [1,128,128])

    def load_inswapper_iss_emap(self, model_name):
        with self.model_lock:
            if not self.models[model_name]:
                self.main_window.model_loading_signal.emit()
                graph = onnx.load(self.models_path[model_name]).graph
                self.emap = onnx.numpy_helper.to_array(graph.initializer[-1])
                self.main_window.model_loaded_signal.emit()

    def run_detect(self, img, detect_mode='RetinaFace', max_num=1, score=0.5, input_size=(512, 512), use_landmark_detection=False, landmark_detect_mode='203', landmark_score=0.5, from_points=False, rotation_angles=None):
        rotation_angles = rotation_angles or [0]
        return self.face_detectors.run_detect(img, detect_mode, max_num, score, input_size, use_landmark_detection, landmark_detect_mode, landmark_score, from_points, rotation_angles)
    
    def run_detect_landmark(self, img, bbox, det_kpss, detect_mode='203', score=0.5, from_points=False):
        return self.face_landmark_detectors.run_detect_landmark(img, bbox, det_kpss, detect_mode, score, from_points)

    def get_arcface_model(self, face_swapper_model): 
        if face_swapper_model in arcface_mapping_model_dict:
            return arcface_mapping_model_dict[face_swapper_model]
        else:
            raise ValueError(f"Face swapper model {face_swapper_model} not found.")

    def run_recognize_direct(self, img, kps, similarity_type='Opal', arcface_model='Inswapper128ArcFace'):
        return self.face_swappers.run_recognize_direct(img, kps, similarity_type, arcface_model)

    def calc_inswapper_latent(self, source_embedding):
        return self.face_swappers.calc_inswapper_latent(source_embedding)

    def run_inswapper(self, image, embedding, output):
        self.face_swappers.run_inswapper(image, embedding, output)

    def calc_swapper_latent_iss(self, source_embedding, version="A"):
        return self.face_swappers.calc_swapper_latent_iss(source_embedding, version)

    def run_iss_swapper(self, image, embedding, output, version="A"):
        self.face_swappers.run_iss_swapper(image, embedding, output, version)

    def calc_swapper_latent_simswap512(self, source_embedding):
        return self.face_swappers.calc_swapper_latent_simswap512(source_embedding)

    def run_swapper_simswap512(self, image, embedding, output):
        self.face_swappers.run_swapper_simswap512(image, embedding, output)

    def calc_swapper_latent_ghost(self, source_embedding):
        return self.face_swappers.calc_swapper_latent_ghost(source_embedding)

    def run_swapper_ghostface(self, image, embedding, output, swapper_model='GhostFace-v2'):
        self.face_swappers.run_swapper_ghostface(image, embedding, output, swapper_model)

    def calc_swapper_latent_cscs(self, source_embedding):
        return self.face_swappers.calc_swapper_latent_cscs(source_embedding)

    def run_swapper_cscs(self, image, embedding, output):
        self.face_swappers.run_swapper_cscs(image, embedding, output)

    def calc_swapper_latent_canonswap(self, source_embedding):
        return self.face_swappers.calc_swapper_latent_canonswap(source_embedding)

    def run_swapper_canonswap(self, image, embedding, output):
        return self.face_swappers.run_canonswap(image, embedding, output)

    def run_enhance_frame_tile_process(self, img, enhancer_type, tile_size=256, scale=1):
        return self.frame_enhancers.run_enhance_frame_tile_process(img, enhancer_type, tile_size, scale)

    def run_deoldify_artistic(self, image, output):
        return self.frame_enhancers.run_deoldify_artistic(image, output)

    def run_deoldify_stable(self, image, output):
        return self.frame_enhancers.run_deoldify_artistic(image, output)
    
    def run_deoldify_video(self, image, output):
        return self.frame_enhancers.run_deoldify_video(image, output)
    
    def run_ddcolor_artistic(self, image, output):
        return self.frame_enhancers.run_ddcolor_artistic(image, output)

    def run_ddcolor(self, tensor_gray_rgb, output_ab):
        return self.frame_enhancers.run_ddcolor(tensor_gray_rgb, output_ab)

    def run_occluder(self, image, output):
        self.face_masks.run_occluder(image, output)

    def run_dfl_xseg(self, image, output):
        self.face_masks.run_dfl_xseg(image, output)

    def run_faceparser(self, image, output):
        self.face_masks.run_faceparser(image, output)

    def run_CLIPs(self, img, CLIPText, CLIPAmount):
        return self.face_masks.run_CLIPs(img, CLIPText, CLIPAmount)
    
    def lp_motion_extractor(self, img, face_editor_type='Human-Face', **kwargs) -> dict:
        return self.face_editors.lp_motion_extractor(img, face_editor_type, **kwargs)

    def lp_appearance_feature_extractor(self, img, face_editor_type='Human-Face'):
        return self.face_editors.lp_appearance_feature_extractor(img, face_editor_type)

    def lp_retarget_eye(self, kp_source: torch.Tensor, eye_close_ratio: torch.Tensor, face_editor_type='Human-Face') -> torch.Tensor:
        return self.face_editors.lp_retarget_eye(kp_source, eye_close_ratio, face_editor_type)

    def lp_retarget_lip(self, kp_source: torch.Tensor, lip_close_ratio: torch.Tensor, face_editor_type='Human-Face') -> torch.Tensor:
        return self.face_editors.lp_retarget_lip(kp_source, lip_close_ratio, face_editor_type)

    def lp_stitch(self, kp_source: torch.Tensor, kp_driving: torch.Tensor, face_editor_type='Human-Face') -> torch.Tensor:
        return self.face_editors.lp_stitch(kp_source, kp_driving, face_editor_type)

    def lp_stitching(self, kp_source: torch.Tensor, kp_driving: torch.Tensor, face_editor_type='Human-Face') -> torch.Tensor:
        return self.face_editors.lp_stitching(kp_source, kp_driving, face_editor_type)

    def lp_warp_decode(self, feature_3d: torch.Tensor, kp_source: torch.Tensor, kp_driving: torch.Tensor, face_editor_type='Human-Face') -> torch.Tensor:
        return self.face_editors.lp_warp_decode(feature_3d, kp_source, kp_driving, face_editor_type)

    def findCosineDistance(self, vector1, vector2):
        vector1 = vector1.ravel()
        vector2 = vector2.ravel()
        cos_dist = 1 - np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))  # 2..0
        return 100 - cos_dist * 50

    def apply_facerestorer(self, swapped_face_upscaled, restorer_det_type, restorer_type, restorer_blend, fidelity_weight, detect_score):
        return self.face_restorers.apply_facerestorer(swapped_face_upscaled, restorer_det_type, restorer_type, restorer_blend, fidelity_weight, detect_score)

    def apply_occlusion(self, img, amount):
        return self.face_masks.apply_occlusion(img, amount)
    
    def apply_dfl_xseg(self, img, amount, background, mouth, parameters):
        return self.face_masks.apply_dfl_xseg(img, amount, background, mouth, parameters)
    
    def process_masks_and_masks(self, swap_restorecalc, original_face_512, parameters):
        return self.face_masks.process_masks_and_masks(swap_restorecalc, original_face_512, parameters)
         
    def apply_face_makeup(self, img, parameters):
        return self.face_editors.apply_face_makeup(img, parameters)
    
    def restore_mouth(self, img_orig, img_swap, kpss_orig, blend_alpha=0.5, feather_radius=10, size_factor=0.5, radius_factor_x=1.0, radius_factor_y=1.0, x_offset=0, y_offset=0):
        return self.face_masks.restore_mouth(img_orig, img_swap, kpss_orig, blend_alpha, feather_radius, size_factor, radius_factor_x, radius_factor_y, x_offset, y_offset)

    def restore_eyes(self, img_orig, img_swap, kpss_orig, blend_alpha=0.5, feather_radius=10, size_factor=3.5, radius_factor_x=1.0, radius_factor_y=1.0, x_offset=0, y_offset=0, eye_spacing_offset=0):
        return self.face_masks.restore_eyes(img_orig, img_swap, kpss_orig, blend_alpha, feather_radius, size_factor, radius_factor_x, radius_factor_y, x_offset, y_offset, eye_spacing_offset)

    def apply_fake_diff(self, swapped_face, original_face, lower_limit_thresh, lower_value, upper_thresh, upper_value, middle_value, parameters):
        return self.face_masks.apply_fake_diff(swapped_face, original_face, lower_limit_thresh, lower_value, upper_thresh, upper_value, middle_value, parameters)

    def run_onnx(self, image, output, model_key):
        return self.face_masks.run_onnx(image, output, model_key)

    def apply_perceptual_diff_onnx(self, swapped_face, original_face, swap_mask, lower_limit_thresh, lower_value, upper_thresh, upper_value, middle_value, feature_layer, ExcludeVGGMaskEnableToggle):
        return self.face_masks.apply_perceptual_diff_onnx(swapped_face, original_face, swap_mask, lower_limit_thresh, lower_value, upper_thresh, upper_value, middle_value, feature_layer, ExcludeVGGMaskEnableToggle)

    @staticmethod 
    def extract_into_tensor_torch(a: torch.Tensor, t: torch.Tensor, x_shape: tuple) -> torch.Tensor:
        if t.ndim == 0: t = t.unsqueeze(0)
        b = t.shape[0]
        out = torch.gather(a, 0, t.long()) 
        return out.reshape(b, *((1,) * (len(x_shape) - 1)))

    def apply_denoiser_unet(self,
                            image_cxhxw_uint8: torch.Tensor,
                            reference_kv_map: Dict | None,
                            use_reference_exclusive_path: bool,
                            denoiser_mode: str = "Single Step (Fast)",
                            denoiser_single_step_t: int = 1,
                            denoiser_ddim_steps: int = 20,
                            denoiser_cfg_scale: float = 1.0,
                            denoiser_ddim_eta: float = 0.0,
                            base_seed: int = 220
                            #blur_sigma_before_sharpen: float = 0.5,
                            #sharpen_strength: float = 0.5
                            ) -> torch.Tensor:
        # This flag is already defined in the original file.
        DEBUG_DENOISER = False
        unet_model_name = self.main_window.fixed_unet_model_name
        vae_encoder_name = 'RefLDMVAEEncoder'
        vae_decoder_name = 'RefLDMVAEDecoder'

        if DEBUG_DENOISER:
            print(f"\n--- Denoiser Pass Start: Mode='{denoiser_mode}', CFG Scale={denoiser_cfg_scale}, VAE Scale Factor={self.vae_scale_factor} ---")
            ModelsProcessor.print_tensor_stats(image_cxhxw_uint8, "Initial input image_cxhxw_uint8", DEBUG_DENOISER)

        with self.model_lock:
            self.ensure_denoiser_models_loaded()
            if not (self.models.get(unet_model_name) and \
                    self.models.get(vae_encoder_name) and \
                    self.models.get(vae_decoder_name)):
                print("Denoiser: Critical models (UNet/VAEs) not loaded. Skipping.")
                return image_cxhxw_uint8

            kv_tensor_map_for_this_run: Dict[str, Dict[str, torch.Tensor]] | None = None
            if reference_kv_map:
                try:
                    # Deep copy to ensure tensors are on the correct device and to avoid side effects
                    kv_tensor_map_for_this_run = {
                        layer: {
                            'k': tens_dict['k'].clone().to(self.device),
                            'v': tens_dict['v'].clone().to(self.device)
                        }
                        for layer, tens_dict in reference_kv_map.items()
                        if tens_dict and isinstance(tens_dict.get('k'), torch.Tensor) and isinstance(tens_dict.get('v'), torch.Tensor)
                    }
                except Exception as e:
                    print(f"Denoiser: Error deep copying K/V map: {e}. Skipping denoiser pass.")
                    return image_cxhxw_uint8
            
            if denoiser_mode == "Full Restore (DDIM)" and use_reference_exclusive_path and not kv_tensor_map_for_this_run:
                print(f"Denoiser (Full Restore): Reference K/V tensor file selected for use, but K/V map is empty. Skipping.")
                return image_cxhxw_uint8
            if denoiser_mode == "Single Step (Fast)" and use_reference_exclusive_path and not kv_tensor_map_for_this_run:
                 print(f"Denoiser (Single Step): Reference K/V tensor file selected for use, but K/V map is empty. Skipping.")
                 return image_cxhxw_uint8

        target_proc_dim = 512 
        _, h_input, w_input = image_cxhxw_uint8.shape
        if h_input != target_proc_dim or w_input != target_proc_dim:
            image_to_process_cxhxw_uint8 = v2.Resize((target_proc_dim, target_proc_dim), 
                                                     interpolation=v2.InterpolationMode.BILINEAR, 
                                                     antialias=True)(image_cxhxw_uint8)
        else:
            image_to_process_cxhxw_uint8 = image_cxhxw_uint8

        h_proc, w_proc = image_to_process_cxhxw_uint8.shape[1], image_to_process_cxhxw_uint8.shape[2]

        # --- VAE Encoder Input Preparation ---
        # Convert sRGB uint8 [0,255] to sRGB float [-1,1] for VAE Encoder
        # This assumes the VAE expects sRGB-like data normalized to [-1,1].
        image_srgb_float_minus1_1 = (image_to_process_cxhxw_uint8.float() / 127.5) - 1.0
        image_srgb_float_minus1_1_batched = image_srgb_float_minus1_1.unsqueeze(0).contiguous()
        
        latent_h, latent_w = h_proc // 8, w_proc // 8
        encoded_latent_direct_vae_out_bchw = torch.empty((1, 8, latent_h, latent_w), dtype=torch.float32, device=self.device).contiguous()
        # Pass the sRGB [-1,1] tensor to the VAE encoder
        self.face_restorers.run_vae_encoder(image_srgb_float_minus1_1_batched, encoded_latent_direct_vae_out_bchw)

        lq_latent_x0_scaled_for_unet = encoded_latent_direct_vae_out_bchw * self.vae_scale_factor # self.vae_scale_factor is 1.0
        
        final_denoised_latent_x0_scaled = None 
        should_use_kv_in_unet = use_reference_exclusive_path and (kv_tensor_map_for_this_run is not None)
        # Both flags are now controlled by the 'use_reference_exclusive_path' UI toggle.
        # is_ref_flag_input: Tells the UNet it's operating in a mode that might involve external K/V (like encoding a reference or using one exclusively).
        # use_reference_exclusive_path_globally_input: Specifically tells the UNet it MUST use external K/V if provided.
        is_ref_flag_tensor_for_unet = torch.tensor([use_reference_exclusive_path], dtype=torch.bool, device=self.device).contiguous()
        actual_use_exclusive_path_tensor_for_unet = torch.tensor([use_reference_exclusive_path], dtype=torch.bool, device=self.device).contiguous()

        if denoiser_mode == "Single Step (Fast)":
            torch.manual_seed(base_seed + denoiser_single_step_t) 
            noise_sample = torch.randn_like(lq_latent_x0_scaled_for_unet)
            current_t_idx = min(max(0, denoiser_single_step_t), len(self.alphas_cumprod_np) - 1)
            alpha_t_bar_val = self.alphas_cumprod_np[current_t_idx]
            sqrt_alpha_bar_t_torch = torch.sqrt(torch.tensor(alpha_t_bar_val, device=self.device, dtype=torch.float32))
            sqrt_one_minus_alpha_bar_t_torch = torch.sqrt(1.0 - torch.tensor(alpha_t_bar_val, device=self.device, dtype=torch.float32))
            xt_noisy_scaled_8_channel = lq_latent_x0_scaled_for_unet * sqrt_alpha_bar_t_torch + noise_sample * sqrt_one_minus_alpha_bar_t_torch
            unet_input_16_channel = torch.cat((xt_noisy_scaled_8_channel, lq_latent_x0_scaled_for_unet), dim=1)
            timesteps_tensor_unet = torch.tensor([current_t_idx], dtype=torch.int64, device=self.device)
            predicted_noise_from_unet = torch.empty((1, 8, latent_h, latent_w), dtype=torch.float32, device=self.device).contiguous()            

            self.face_restorers.run_ref_ldm_unet(
                x_noisy_plus_lq_latent=unet_input_16_channel,
                timesteps_tensor=timesteps_tensor_unet,
                is_ref_flag_tensor=is_ref_flag_tensor_for_unet,
                use_reference_exclusive_path_globally_tensor=actual_use_exclusive_path_tensor_for_unet,
                kv_tensor_map=kv_tensor_map_for_this_run, # Pass directly, can be None
                output_unet_tensor=predicted_noise_from_unet
            )
            final_denoised_latent_x0_scaled = (xt_noisy_scaled_8_channel - sqrt_one_minus_alpha_bar_t_torch * predicted_noise_from_unet) / sqrt_alpha_bar_t_torch

        elif denoiser_mode == "Full Restore (DDIM)":
            with torch.cuda.stream(torch.cuda.current_stream()):
                torch.cuda.manual_seed_all(base_seed) # Seed once before the loop for initial x_T and subsequent noise in DDIM step
                
                num_ddpm_timesteps = self.alphas_cumprod_np.shape[0]
                _ddim_raw_ddpm_timesteps_np = ModelsProcessor.make_ddim_timesteps(
                    ddim_discr_method="uniform", 
                    num_ddim_timesteps=denoiser_ddim_steps,
                    num_ddpm_timesteps=num_ddpm_timesteps,
                    verbose=DEBUG_DENOISER
                )
                _ddim_sigmas_np, _ddim_alphas_np, _ddim_alphas_prev_np = ModelsProcessor.make_ddim_sampling_parameters(
                    alphacums=self.alphas_cumprod_np,
                    ddim_timesteps=_ddim_raw_ddpm_timesteps_np, 
                    eta=denoiser_ddim_eta,
                    verbose=DEBUG_DENOISER
                )
                ddim_sigmas = torch.from_numpy(_ddim_sigmas_np).float().to(self.device)
                ddim_alphas = torch.from_numpy(_ddim_alphas_np).float().to(self.device)
                ddim_alphas_prev = torch.from_numpy(_ddim_alphas_prev_np).float().to(self.device)
                ddim_sqrt_one_minus_alphas = torch.sqrt(torch.clamp(1. - ddim_alphas, min=0.0))
                current_latent_xt_scaled = torch.randn_like(lq_latent_x0_scaled_for_unet)

                time_range_ddpm_indices = np.flip(_ddim_raw_ddpm_timesteps_np)
                total_steps = len(time_range_ddpm_indices)
                pred_x0_scaled_current_step = torch.empty_like(lq_latent_x0_scaled_for_unet)

                for i, step_ddpm_idx in enumerate(time_range_ddpm_indices):
                    torch.cuda.manual_seed_all(base_seed)
                    index_for_schedules = total_steps - 1 - i 
                    ts_unet = torch.full((1,), step_ddpm_idx, device=self.device, dtype=torch.int64)
                    unet_input_cond = torch.cat([current_latent_xt_scaled, lq_latent_x0_scaled_for_unet], dim=1)
                    e_t_cond = torch.empty_like(lq_latent_x0_scaled_for_unet)

                    self.face_restorers.run_ref_ldm_unet(
                        x_noisy_plus_lq_latent=unet_input_cond,
                        timesteps_tensor=ts_unet,
                        is_ref_flag_tensor=is_ref_flag_tensor_for_unet,
                        use_reference_exclusive_path_globally_tensor=actual_use_exclusive_path_tensor_for_unet,
                        kv_tensor_map=kv_tensor_map_for_this_run, # Pass directly, can be None
                        output_unet_tensor=e_t_cond
                    )
                    e_t = e_t_cond

                    if denoiser_cfg_scale != 1.0:
                        unet_input_uncond = torch.cat([current_latent_xt_scaled, lq_latent_x0_scaled_for_unet], dim=1)
                        e_t_uncond = torch.empty_like(lq_latent_x0_scaled_for_unet)                    
                        # For uncond path, exclusive_path_globally is effectively False, and no K/V map is used.
                        self.face_restorers.run_ref_ldm_unet(
                            x_noisy_plus_lq_latent=unet_input_uncond,
                            timesteps_tensor=ts_unet,
                            is_ref_flag_tensor=is_ref_flag_tensor_for_unet,
                            use_reference_exclusive_path_globally_tensor=torch.tensor([False], dtype=torch.bool, device=self.device).contiguous(), 
                            kv_tensor_map=None, 
                            output_unet_tensor=e_t_uncond
                        )
                        e_t = e_t_uncond + denoiser_cfg_scale * (e_t_cond - e_t_uncond)
                    
                    schedule_idx_tensor = torch.tensor([index_for_schedules], device=self.device, dtype=torch.long)
                    a_t = ModelsProcessor.extract_into_tensor_torch(ddim_alphas, schedule_idx_tensor, current_latent_xt_scaled.shape)
                    a_prev = ModelsProcessor.extract_into_tensor_torch(ddim_alphas_prev, schedule_idx_tensor, current_latent_xt_scaled.shape)
                    sigma_t = ModelsProcessor.extract_into_tensor_torch(ddim_sigmas, schedule_idx_tensor, current_latent_xt_scaled.shape)
                    sqrt_one_minus_a_t = ModelsProcessor.extract_into_tensor_torch(ddim_sqrt_one_minus_alphas, schedule_idx_tensor, current_latent_xt_scaled.shape)
                    pred_x0_scaled_current_step = (current_latent_xt_scaled - sqrt_one_minus_a_t * e_t) / torch.sqrt(a_t).clamp(min=1e-8) # Clamp to avoid div by zero if a_t is 0
                    dir_xt = torch.sqrt(torch.clamp(1. - a_prev - sigma_t**2, min=1e-8)) * e_t # Clamp to avoid sqrt of negative
                    # torch.manual_seed(base_seed + step_ddpm_idx) # Seeding here per DDIM step is valid, but let's test seeding once before loop
                    noise_ddim = sigma_t * torch.randn_like(current_latent_xt_scaled)
                    current_latent_xt_scaled = torch.sqrt(a_prev) * pred_x0_scaled_current_step + dir_xt + noise_ddim
                final_denoised_latent_x0_scaled = pred_x0_scaled_current_step
        
        else: 
            print(f"Denoiser: Unknown mode '{denoiser_mode}'. Skipping denoiser pass.")
            return image_cxhxw_uint8

        if final_denoised_latent_x0_scaled is None:
            return image_cxhxw_uint8

        latent_for_vae_decoder = final_denoised_latent_x0_scaled / self.vae_scale_factor
        decoded_image_normalized_bchw = torch.empty((1, 3, h_proc, w_proc), dtype=torch.float32, device=self.device).contiguous()
        
        # Run VAE Decoder
        self.face_restorers.run_vae_decoder(latent_for_vae_decoder, decoded_image_normalized_bchw)
        
        # --- VAE Decoder Output Post-processing ---
        # Apply tanh to softly map the VAE output to strictly [-1, 1] range.
        # This helps prevent hard clipping if the VAE overshoots its nominal range.
        decoded_image_soft_clamped_bchw = torch.tanh(decoded_image_normalized_bchw)

        # Convert sRGB float [-1,1] to sRGB float [0,1].
        # A final clamp to [0,1] is good practice for safety, though tanh helps.
        image_after_postproc_float_0_1 = (decoded_image_soft_clamped_bchw.squeeze(0) + 1.0) / 2.0
        image_after_postproc_float_0_1 = torch.clamp(image_after_postproc_float_0_1, 0.0, 1.0)

        
        final_image_uint8 = (image_after_postproc_float_0_1 * 255.0).byte()

        if h_proc != h_input or w_proc != w_input:
            output_image_cxhxw_uint8 = v2.Resize((h_input, w_input),
                                                interpolation=v2.InterpolationMode.BILINEAR,
                                                antialias=True)(final_image_uint8)
        else:
            output_image_cxhxw_uint8 = final_image_uint8
        
        return output_image_cxhxw_uint8