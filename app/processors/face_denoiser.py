import os
import threading
import gc
import traceback
from typing import TYPE_CHECKING, Dict, Optional, Union
from collections import OrderedDict

import torch
import numpy as np
from torchvision.transforms import v2

if TYPE_CHECKING:
    from app.processors.models_processor import ModelsProcessor
    from app.processors.workers.function_worker import FunctionWorker
    from PIL import Image

from app.processors.utils import faceutil
from app.processors.utils import platform_support
from app.helpers.miscellaneous import is_file_exists
from app.helpers.downloader import download_file
from app.processors.utils.ref_ldm_kv_embedding import KVExtractor


class FaceDenoiser:
    """
    Handles Diffusion-based Denoiser/Restorer (ReF-LDM) operations.
    Manages DDIM/DDPM mathematical schedules and VAE latent processing.
    """

    def __init__(
        self,
        models_processor: "ModelsProcessor",
        function_worker: "FunctionWorker",
    ):
        self.models_processor = models_processor
        self.function_worker = function_worker

        # --- KV Extractor State ---
        self.kv_extractor: Optional[KVExtractor] = None
        self.kv_extraction_lock = threading.Lock()

        # Denoiser specific initializations (VR180 feature compatible)
        num_ddpm_timesteps = 1000
        linear_start_val = 0.0015
        linear_end_val = 0.0155
        self.betas_np = self.make_beta_schedule(
            schedule="linear",
            n_timestep=num_ddpm_timesteps,
            linear_start=linear_start_val,
            linear_end=linear_end_val,
        )
        self.alphas_np = 1.0 - self.betas_np
        self.alphas_cumprod_np = np.cumprod(self.alphas_np, axis=0)

        # We store the base tensor here. Note: If the device changes,
        # ModelsProcessor's switch_providers_priority will need to update this,
        # or we dynamically cast it during inference.
        self.alphas_cumprod_torch = (
            torch.from_numpy(self.alphas_cumprod_np)
            .float()
            .to(self.models_processor.device)
        )

        # NOTE: vae_scale_factor=1.0 is intentional for this model's specific VAE configuration
        self.vae_scale_factor = 1.0

        # Cache for DDIM schedule tensors, keyed by (ddim_steps, ddim_eta).
        self._ddim_schedule_cache: OrderedDict = OrderedDict()
        self._DDIM_CACHE_MAX = 20

    @staticmethod
    def print_tensor_stats(tensor: torch.Tensor, name: str, enabled: bool = True):
        if not enabled:
            return
        if isinstance(tensor, torch.Tensor):
            if tensor.dtype == torch.uint8:
                tensor_float = tensor.float() / 255.0
                print(
                    f"DEBUG DENOISER STATS for {name}: shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device}, min={tensor.min().item():.4f}, max={tensor.max().item():.4f}, mean={tensor_float.mean().item():.4f}, std={tensor_float.std().item():.4f} (stats on [0,1] float)"
                )
            elif tensor.dtype == torch.float16 or tensor.dtype == torch.float32:
                print(
                    f"DEBUG DENOISER STATS for {name}: shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device}, min={tensor.min().item():.4f}, max={tensor.max().item():.4f}, mean={tensor.mean().item():.4f}, std={tensor.std().item():.4f}"
                )
            else:
                print(
                    f"DEBUG DENOISER STATS for {name}: shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device} (stats not computed for this dtype)"
                )
        else:
            print(
                f"DEBUG DENOISER STATS for {name}: Not a tensor, type is {type(tensor)}"
            )

    @staticmethod
    def make_beta_schedule(
        schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3
    ) -> np.ndarray:
        if schedule == "linear":
            betas = (
                torch.linspace(
                    linear_start**0.5, linear_end**0.5, n_timestep, dtype=torch.float64
                )
                ** 2
            )
        elif schedule == "cosine":
            timesteps = (
                torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep
                + cosine_s
            )
            alphas = timesteps / (1 + cosine_s) * np.pi / 2  # type: ignore
            alphas = torch.cos(alphas).pow(2)
            alphas = alphas / alphas[0]
            betas = 1 - alphas[1:] / alphas[:-1]
            betas = np.clip(betas.numpy(), a_min=0, a_max=0.999)  # type: ignore
        elif schedule == "sqrt_linear":
            betas = torch.linspace(
                linear_start, linear_end, n_timestep, dtype=torch.float64
            )
        elif schedule == "sqrt":
            betas = (
                torch.linspace(
                    linear_start, linear_end, n_timestep, dtype=torch.float64
                )
                ** 0.5
            )
        else:
            raise ValueError(f"schedule '{schedule}' unknown.")
        return betas.numpy() if isinstance(betas, torch.Tensor) else betas

    @staticmethod
    def make_ddim_timesteps(
        ddim_discr_method: str,
        num_ddim_timesteps: int,
        num_ddpm_timesteps: int,
        verbose: bool = True,
    ) -> np.ndarray:
        if ddim_discr_method == "uniform":
            c = num_ddpm_timesteps // num_ddim_timesteps
            if c == 0:
                c = 1
            ddim_timesteps = np.asarray(list(range(0, num_ddpm_timesteps, c)))
        elif ddim_discr_method == "uniform_trailing":
            c = num_ddpm_timesteps // num_ddim_timesteps
            if c == 0:
                c = 1
            ddim_timesteps = np.arange(num_ddpm_timesteps, 0, -c).astype(int)[::-1] - 2
            ddim_timesteps = np.clip(ddim_timesteps, 0, num_ddpm_timesteps - 1)
        elif ddim_discr_method == "quad":
            ddim_timesteps = (
                (np.linspace(0, np.sqrt(num_ddpm_timesteps * 0.8), num_ddim_timesteps))
                ** 2
            ).astype(int)
        else:
            raise NotImplementedError(
                f'There is no ddim discretization method called "{ddim_discr_method}"'
            )

        steps_out = np.unique(ddim_timesteps)
        steps_out.sort()

        if verbose:
            print(f"Selected DDPM timesteps for DDIM sampler (0-indexed): {steps_out}")
        return steps_out

    @staticmethod
    def make_ddim_sampling_parameters(
        alphacums: np.ndarray,
        ddim_timesteps: np.ndarray,
        eta: float,
        verbose: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        _prev_t = np.concatenate(
            ([-1], ddim_timesteps[:-1])
        )  # Use -1 to signify "before first step"
        _alphas_prev = np.array([alphacums[pt] if pt != -1 else 1.0 for pt in _prev_t])
        _alphas = alphacums[ddim_timesteps]
        sigmas = eta * np.sqrt(
            (1 - _alphas_prev) / (1 - _alphas) * (1 - _alphas / _alphas_prev)
        )
        sigmas = np.nan_to_num(sigmas, nan=0.0)
        return sigmas, _alphas, _alphas_prev

    @staticmethod
    def extract_into_tensor_torch(
        a: torch.Tensor, t: torch.Tensor, x_shape: tuple
    ) -> torch.Tensor:
        # Direct indexing is significantly faster than torch.gather for 1D arrays.
        # .view() maps the memory without allocating a new tensor.
        if t.ndim == 0:
            t = t.unsqueeze(0)
        b = t.shape[0]
        out = a[t.long()]
        return out.view(b, *((1,) * (len(x_shape) - 1)))

    def ensure_denoiser_models_loaded(self):
        """Loads the UNet and VAE models if they are not already loaded."""
        with self.models_processor.model_lock:
            unet_model_name = self.models_processor.main_window.fixed_unet_model_name
            vae_encoder_name = "RefLDMVAEEncoder"
            vae_decoder_name = "RefLDMVAEDecoder"

            if not self.models_processor.models.get(unet_model_name):
                self.models_processor.models[unet_model_name] = (
                    self.models_processor.load_model(unet_model_name)
                )

            if not self.models_processor.models.get(vae_encoder_name):
                self.models_processor.models[vae_encoder_name] = (
                    self.models_processor.load_model(vae_encoder_name)
                )

            if not self.models_processor.models.get(vae_decoder_name):
                self.models_processor.models[vae_decoder_name] = (
                    self.models_processor.load_model(vae_decoder_name)
                )

    def unload_models(self):
        """Unloads the UNet and VAE models."""
        with self.models_processor.model_lock:
            print("[INFO] Unloading denoiser models (UNet, VAEs)...")
            self.models_processor.unload_model(
                self.models_processor.main_window.fixed_unet_model_name
            )
            self.models_processor.unload_model("RefLDMVAEEncoder")
            self.models_processor.unload_model("RefLDMVAEDecoder")

    @torch.no_grad()
    def get_kv_map_for_face(
        self,
        image: Union["Image.Image", torch.Tensor],
        unload_after: bool = True,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Loads the KV Extractor, extracts K/V maps for a single image, and unloads (if requested).
        Callers are responsible for holding kv_extraction_lock around this call.
        """
        kv_map = {}
        try:
            # 1. Load the extractor
            self.ensure_kv_extractor_loaded()

            if self.kv_extractor is None:
                raise RuntimeError("KV Extractor model failed to load.")

            # 2. Perform the extraction
            print("[INFO] Extracting K/V from reference image...")
            kv_map = self.kv_extractor.extract_kv(image)

            print(
                f"[INFO] Successfully extracted K/V for {len(kv_map)} attention layers."
            )

        except Exception as e:
            print(f"[ERROR] Failed the K/V extraction: {e}")
            import traceback

            traceback.print_exc()
            kv_map = {}  # Return empty map if failed

        finally:
            # 3. Unload the extractor safely based on batching context
            if unload_after:
                self.unload_kv_extractor()

        return kv_map

    def ensure_kv_extractor_loaded(self):
        """
        Guarantees that the KVExtractor (Ref-LDM) model is loaded and ready.
        Downloads the required config and checkpoint files on first use.
        """
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
                print(
                    f"[INFO] Downloading ReF-LDM file: {os.path.basename(full_path)}..."
                )
                download_file(os.path.basename(full_path), full_path, None, url)

        config_path = os.path.join(configs_path, "refldm.yaml")
        model_path = os.path.join(ckpts_path, "refldm.ckpt")
        vae_path = os.path.join(ckpts_path, "vqgan.ckpt")

        if not all(os.path.exists(p) for p in [config_path, model_path, vae_path]):
            print(
                "[ERROR] ReF-LDM model files not found even after download attempt. Cannot load KV Extractor."
            )
            return

        with self.models_processor.model_lock:
            if self.kv_extractor is not None:
                return  # Already loaded

            try:
                print("[INFO] Loading KV Extractor...")
                self.kv_extractor = KVExtractor(
                    model_config_path=config_path,
                    model_ckpt_path=model_path,
                    vae_ckpt_path=vae_path,
                    device=self.models_processor.device,
                )
                print("[INFO] KV Extractor loaded.")
            except Exception as e:
                print(f"[ERROR] Failed to load KV Extractor: {e}")
                traceback.print_exc()
                self.kv_extractor = None

    def unload_kv_extractor(self, force_immediate=False):
        """Unloads the KVExtractor model and clears associated memory."""
        if not self.models_processor.force_unload_in_progress:
            if self.models_processor.main_window.control.get(
                "KeepModelsAliveToggle", False
            ):
                return

        if not force_immediate and not self.models_processor.force_unload_in_progress:
            vp = getattr(self.models_processor.main_window, "video_processor", None)
            if vp and getattr(vp, "processing", False):
                target_frame = getattr(vp, "current_frame_number", 0) + 1
                with self.models_processor.model_lock:
                    self.models_processor.deferred_unloads["KVExtractor"] = {
                        "type": "kv",
                        "target_frame": target_frame,
                    }
                print(
                    f"[INFO] Smart Unload: Deferring KV Extractor unload after frame {target_frame}"
                )
                return

        with self.models_processor.model_lock:
            if self.kv_extractor is not None:
                print("[INFO] Unloading KV Extractor...")
                del self.kv_extractor
                self.kv_extractor = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    @torch.no_grad()
    def apply_denoiser_unet(
        self,
        image_cxhxw_uint8: torch.Tensor,
        reference_kv_map: Optional[Dict[str, Dict[str, torch.Tensor]]],
        use_reference_exclusive_path: bool,
        denoiser_mode: str = "Single Step (Fast)",
        denoiser_single_step_t: int = 1,
        denoiser_ddim_steps: int = 20,
        denoiser_cfg_scale: float = 1.0,
        denoiser_ddim_eta: float = 0.0,
        base_seed: int = 220,
        latent_sharpening_strength: float = 0.0,
        enable_color_correction: bool = True,
        color_mask: Optional[torch.Tensor] = None,
        coarse_grain_reduction: float = 0.0,
        micro_grain_strength: float = 0.0,
    ) -> torch.Tensor:
        """
        Runs the Diffusion-based Denoiser/Restorer (ReF-LDM).
        Supports 'Single Step' (Fast) and 'Full Restore' (DDIM) modes.
        Features NaN-safe DDIM scheduling and post-VAE hot-pixel blowout neutralization.
        """
        import math

        # --- CONFIGURATION ---
        ENABLE_PIXEL_SHARPENING: bool = latent_sharpening_strength > 0.0
        PIXEL_SHARPEN_STRENGTH: float = latent_sharpening_strength
        ENABLE_COLOR_MATCH: bool = enable_color_correction
        COARSE_GRAIN_LAMBDA: float = max(0.0, min(1.0, float(coarse_grain_reduction)))
        MICRO_GRAIN_STRENGTH: float = max(0.0, min(1.0, float(micro_grain_strength)))

        DEBUG_DENOISER: bool = os.environ.get("VISOMASTER_DEBUG_DENOISER", "0") == "1"
        unet_model_name: str = self.models_processor.main_window.fixed_unet_model_name
        vae_encoder_name: str = "RefLDMVAEEncoder"
        vae_decoder_name: str = "RefLDMVAEDecoder"

        if DEBUG_DENOISER:
            print(
                f"\n--- Denoiser Pass Start: Mode='{denoiser_mode}', CFG Scale={denoiser_cfg_scale}, "
                f"CoarseReduction={COARSE_GRAIN_LAMBDA}, MicroGrain={MICRO_GRAIN_STRENGTH} ---"
            )
            self.print_tensor_stats(
                image_cxhxw_uint8, "Initial input image_cxhxw_uint8", DEBUG_DENOISER
            )

        with self.models_processor.model_lock:
            self.ensure_denoiser_models_loaded()
            unet_session = self.models_processor.models.get(unet_model_name)
            vae_enc_session = self.models_processor.models.get(vae_encoder_name)
            vae_dec_session = self.models_processor.models.get(vae_decoder_name)

            if not (unet_session and vae_enc_session and vae_dec_session):
                return image_cxhxw_uint8

        kv_tensor_map_for_this_run: Optional[Dict[str, Dict[str, torch.Tensor]]] = None
        if reference_kv_map:
            try:
                kv_tensor_map_for_this_run = {
                    layer: {"k": tens_dict["k"], "v": tens_dict["v"]}
                    for layer, tens_dict in reference_kv_map.items()
                    if tens_dict
                    and isinstance(tens_dict.get("k"), torch.Tensor)
                    and isinstance(tens_dict.get("v"), torch.Tensor)
                }
            except Exception as e:
                print(f"[ERROR] Denoiser: Error copying K/V map: {e}. Skipping.")
                return image_cxhxw_uint8

        if (
            denoiser_mode == "Full Restore (DDIM)"
            and use_reference_exclusive_path
            and not kv_tensor_map_for_this_run
        ):
            print(
                "[ERROR] Denoiser (Full Restore): Reference K/V tensor file selected for use, but K/V map is empty. Skipping."
            )
            return image_cxhxw_uint8
        if (
            denoiser_mode == "Single Step (Fast)"
            and use_reference_exclusive_path
            and not kv_tensor_map_for_this_run
        ):
            print(
                "[ERROR] Denoiser (Single Step): Reference K/V tensor file selected for use, but K/V map is empty. Skipping."
            )
            return image_cxhxw_uint8

        target_proc_dim: int = 512
        _, h_input, w_input = image_cxhxw_uint8.shape
        if h_input != target_proc_dim or w_input != target_proc_dim:
            image_to_process_cxhxw_uint8: torch.Tensor = v2.functional.resize(
                image_cxhxw_uint8,
                [target_proc_dim, target_proc_dim],
                interpolation=v2.InterpolationMode.BILINEAR,
                antialias=True,
            )
        else:
            image_to_process_cxhxw_uint8 = image_cxhxw_uint8

        h_proc: int = image_to_process_cxhxw_uint8.shape[1]
        w_proc: int = image_to_process_cxhxw_uint8.shape[2]

        # Normalized input tensor in [-1.0, 1.0]
        input_srgb_float_minus1_1: torch.Tensor = (
            image_to_process_cxhxw_uint8.to(
                dtype=torch.float32, copy=True, non_blocking=True
            )
            .mul_(1.0 / 127.5)
            .sub_(1.0)
        )
        image_srgb_float_minus1_1_batched: torch.Tensor = (
            input_srgb_float_minus1_1.unsqueeze(0).contiguous()
        )

        latent_h: int = h_proc // 8
        latent_w: int = w_proc // 8
        encoded_latent_direct_vae_out_bchw: torch.Tensor = torch.empty(
            (1, 8, latent_h, latent_w),
            dtype=torch.float32,
            device=self.models_processor.device,
        ).contiguous()

        self.function_worker.run_vae_encoder(
            image_srgb_float_minus1_1_batched, encoded_latent_direct_vae_out_bchw
        )

        lq_latent_x0_scaled_for_unet: torch.Tensor = (
            encoded_latent_direct_vae_out_bchw * self.vae_scale_factor
        )
        del encoded_latent_direct_vae_out_bchw
        del image_srgb_float_minus1_1_batched
        final_denoised_latent_x0_scaled: Optional[torch.Tensor] = None

        if use_reference_exclusive_path:
            is_ref_flag_tensor_for_unet: torch.Tensor = torch.ones(
                1, dtype=torch.bool, device=self.models_processor.device
            )
        else:
            is_ref_flag_tensor_for_unet = torch.zeros(
                1, dtype=torch.bool, device=self.models_processor.device
            )

        actual_use_exclusive_path_tensor_for_unet: torch.Tensor = (
            is_ref_flag_tensor_for_unet
        )
        false_tensor_for_unet: torch.Tensor = torch.zeros(
            1, dtype=torch.bool, device=self.models_processor.device
        )

        rng: torch.Generator = torch.Generator(device=self.models_processor.device)
        rng.manual_seed(base_seed)

        # --- PROCESS: Single Step ---
        if denoiser_mode == "Single Step (Fast)":
            rng.manual_seed(base_seed + denoiser_single_step_t)
            noise_sample: torch.Tensor = torch.randn(
                lq_latent_x0_scaled_for_unet.shape,
                device=self.models_processor.device,
                dtype=lq_latent_x0_scaled_for_unet.dtype,
                generator=rng,
            )

            current_t_idx: int = min(
                max(0, denoiser_single_step_t), len(self.alphas_cumprod_np) - 1
            )
            alpha_t_bar_val: float = float(self.alphas_cumprod_np[current_t_idx])

            sqrt_a: float = math.sqrt(alpha_t_bar_val)
            sqrt_one_minus_a: float = math.sqrt(1.0 - alpha_t_bar_val)

            xt_noisy_scaled_8_channel: torch.Tensor = (
                lq_latent_x0_scaled_for_unet * sqrt_a + noise_sample * sqrt_one_minus_a
            )
            unet_input_16_channel: torch.Tensor = torch.cat(
                (xt_noisy_scaled_8_channel, lq_latent_x0_scaled_for_unet), dim=1
            )

            timesteps_tensor_unet: torch.Tensor = torch.full(
                (1,),
                current_t_idx,
                dtype=torch.int64,
                device=self.models_processor.device,
            )

            predicted_noise_from_unet: torch.Tensor = torch.empty(
                (1, 8, latent_h, latent_w),
                dtype=torch.float32,
                device=self.models_processor.device,
            ).contiguous()

            if torch.cuda.is_available():
                platform_support.blocking_stream_sync()

            self.function_worker.run_ref_ldm_unet(
                x_noisy_plus_lq_latent=unet_input_16_channel,
                timesteps_tensor=timesteps_tensor_unet,
                is_ref_flag_tensor=is_ref_flag_tensor_for_unet,
                use_reference_exclusive_path_globally_tensor=actual_use_exclusive_path_tensor_for_unet,
                kv_tensor_map=kv_tensor_map_for_this_run,
                output_unet_tensor=predicted_noise_from_unet,
            )

            predicted_noise_sanitized: torch.Tensor = torch.nan_to_num(
                predicted_noise_from_unet, nan=0.0, posinf=4.0, neginf=-4.0
            )

            raw_estimated_x0: torch.Tensor = (
                xt_noisy_scaled_8_channel - sqrt_one_minus_a * predicted_noise_sanitized
            ) / max(sqrt_a, 1e-6)

            if COARSE_GRAIN_LAMBDA > 0.0:
                latent_delta: torch.Tensor = (
                    raw_estimated_x0 - lq_latent_x0_scaled_for_unet
                )
                # Soft-clamp uncancelled residual noise spikes to suppress macro-grain
                clamped_delta: torch.Tensor = torch.clamp(latent_delta, -2.5, 2.5)
                effective_delta: torch.Tensor = torch.lerp(
                    latent_delta, clamped_delta, COARSE_GRAIN_LAMBDA
                )
                final_denoised_latent_x0_scaled = (
                    lq_latent_x0_scaled_for_unet + effective_delta
                )
            else:
                final_denoised_latent_x0_scaled = raw_estimated_x0

        # --- PROCESS: Full Restore (DDIM) ---
        elif denoiser_mode == "Full Restore (DDIM)":
            num_ddpm_timesteps: int = self.alphas_cumprod_np.shape[0]

            _ddim_raw_ddpm_timesteps_np: np.ndarray = self.make_ddim_timesteps(
                ddim_discr_method="uniform",
                num_ddim_timesteps=denoiser_ddim_steps,
                num_ddpm_timesteps=num_ddpm_timesteps,
                verbose=DEBUG_DENOISER,
            )
            _ddim_sigmas_np, _ddim_alphas_np, _ddim_alphas_prev_np = (
                self.make_ddim_sampling_parameters(
                    alphacums=self.alphas_cumprod_np,
                    ddim_timesteps=_ddim_raw_ddpm_timesteps_np,
                    eta=denoiser_ddim_eta,
                    verbose=DEBUG_DENOISER,
                )
            )

            ddim_sigmas: torch.Tensor = (
                torch.from_numpy(_ddim_sigmas_np)
                .float()
                .to(self.models_processor.device, non_blocking=True)
            )
            ddim_alphas: torch.Tensor = (
                torch.from_numpy(_ddim_alphas_np)
                .float()
                .to(self.models_processor.device, non_blocking=True)
            )
            ddim_alphas_prev: torch.Tensor = (
                torch.from_numpy(_ddim_alphas_prev_np)
                .float()
                .to(self.models_processor.device, non_blocking=True)
            )

            ddim_sqrt_one_minus_alphas: torch.Tensor = torch.sqrt(
                torch.clamp(1.0 - ddim_alphas, min=0.0)
            )

            # --- OPTIMIZATION: Pre-view schedule tensors for O(1) direct indexing ---
            # We reshape to (N, 1, 1, 1) so PyTorch automatically broadcasts to (1, 8, H, W)
            view_shape: tuple[int, int, int, int] = (-1, 1, 1, 1)
            ddim_alphas_view: torch.Tensor = ddim_alphas.view(view_shape)
            ddim_alphas_prev_view: torch.Tensor = ddim_alphas_prev.view(view_shape)
            ddim_sigmas_view: torch.Tensor = ddim_sigmas.view(view_shape)
            ddim_sqrt_one_minus_alphas_view: torch.Tensor = (
                ddim_sqrt_one_minus_alphas.view(view_shape)
            )

            time_range_ddpm_indices: np.ndarray = np.flip(
                _ddim_raw_ddpm_timesteps_np
            ).copy()
            total_steps: int = len(time_range_ddpm_indices)

            noise_init: torch.Tensor = torch.randn(
                lq_latent_x0_scaled_for_unet.shape,
                device=self.models_processor.device,
                dtype=lq_latent_x0_scaled_for_unet.dtype,
                generator=rng,
            )

            if COARSE_GRAIN_LAMBDA > 0.0:
                init_step_idx: int = int(time_range_ddpm_indices[0])
                init_alpha_val: float = float(self.alphas_cumprod_np[init_step_idx])
                init_sqrt_a: float = math.sqrt(init_alpha_val)
                init_sqrt_one_minus_a: float = math.sqrt(1.0 - init_alpha_val)
                sdedit_init: torch.Tensor = (
                    lq_latent_x0_scaled_for_unet * init_sqrt_a
                    + noise_init * init_sqrt_one_minus_a
                )
                current_latent_xt_scaled = torch.lerp(
                    noise_init, sdedit_init, COARSE_GRAIN_LAMBDA
                )
            else:
                current_latent_xt_scaled = noise_init

            pred_x0_scaled_current_step: torch.Tensor = torch.empty_like(
                lq_latent_x0_scaled_for_unet
            )
            ts_unet: torch.Tensor = torch.empty(
                (1,), dtype=torch.int64, device=self.models_processor.device
            )
            e_t_cond: torch.Tensor = torch.empty_like(lq_latent_x0_scaled_for_unet)
            e_t_uncond: Optional[torch.Tensor] = (
                torch.empty_like(lq_latent_x0_scaled_for_unet)
                if denoiser_cfg_scale != 1.0
                else None
            )
            noise_ddim_buffer: torch.Tensor = torch.empty_like(
                lq_latent_x0_scaled_for_unet
            )

            # Pre-allocate the 16-channel UNet input buffer once.
            ddim_unet_input_16_channel: torch.Tensor = torch.empty(
                (1, 16, latent_h, latent_w),
                dtype=torch.float32,
                device=self.models_processor.device,
            ).contiguous()

            # The condition (LQ image) remains static. Write it to channels 8-15 once.
            ddim_unet_input_16_channel[:, 8:16] = lq_latent_x0_scaled_for_unet

            # Biological latent dynamic envelope for VQGAN (codebook upper bound is ~2.8)
            LATENT_DYNAMIC_BOUND: float = 2.8

            for i, step_ddpm_idx in enumerate(time_range_ddpm_indices):
                index_for_schedules: int = total_steps - 1 - i
                ts_unet.fill_(step_ddpm_idx)
                # Update only the dynamic noisy channels (0-7) in-place.
                ddim_unet_input_16_channel[:, :8] = current_latent_xt_scaled

                if torch.cuda.is_available():
                    platform_support.blocking_stream_sync()

                self.function_worker.run_ref_ldm_unet(
                    x_noisy_plus_lq_latent=ddim_unet_input_16_channel,
                    timesteps_tensor=ts_unet,
                    is_ref_flag_tensor=is_ref_flag_tensor_for_unet,
                    use_reference_exclusive_path_globally_tensor=actual_use_exclusive_path_tensor_for_unet,
                    kv_tensor_map=kv_tensor_map_for_this_run,
                    output_unet_tensor=e_t_cond,
                )

                if denoiser_cfg_scale != 1.0 and e_t_uncond is not None:
                    if torch.cuda.is_available():
                        platform_support.blocking_stream_sync()

                    # We re-use ddim_unet_input_16_channel directly.
                    self.function_worker.run_ref_ldm_unet(
                        x_noisy_plus_lq_latent=ddim_unet_input_16_channel,
                        timesteps_tensor=ts_unet,
                        is_ref_flag_tensor=is_ref_flag_tensor_for_unet,
                        use_reference_exclusive_path_globally_tensor=false_tensor_for_unet,
                        kv_tensor_map=None,
                        output_unet_tensor=e_t_uncond,
                    )

                    # Sanitized CFG delta calculation to prevent contrast runaway
                    cond_clean = torch.nan_to_num(
                        e_t_cond, nan=0.0, posinf=3.5, neginf=-3.5
                    )
                    uncond_clean = torch.nan_to_num(
                        e_t_uncond, nan=0.0, posinf=3.5, neginf=-3.5
                    )
                    cfg_diff = torch.clamp(cond_clean - uncond_clean, -2.5, 2.5)
                    e_t = uncond_clean + cfg_diff * denoiser_cfg_scale
                else:
                    e_t = torch.nan_to_num(e_t_cond, nan=0.0, posinf=3.5, neginf=-3.5)

                # --- OPTIMIZATION: Direct indexing replaces extract_into_tensor_torch ---
                a_t = ddim_alphas_view[index_for_schedules]
                a_prev = ddim_alphas_prev_view[index_for_schedules]
                sigma_t = ddim_sigmas_view[index_for_schedules]
                sqrt_one_minus_a_t = ddim_sqrt_one_minus_alphas_view[
                    index_for_schedules
                ]

                sqrt_a_t: torch.Tensor = torch.sqrt(a_t).clamp(min=1e-8)
                pred_x0_raw: torch.Tensor = (
                    current_latent_xt_scaled - sqrt_one_minus_a_t * e_t
                ) / sqrt_a_t

                # Arrest early-step division explosion and sanitize NaNs
                pred_x0_scaled_current_step = torch.nan_to_num(
                    pred_x0_raw,
                    nan=0.0,
                    posinf=LATENT_DYNAMIC_BOUND,
                    neginf=-LATENT_DYNAMIC_BOUND,
                ).clamp_(-LATENT_DYNAMIC_BOUND, LATENT_DYNAMIC_BOUND)

                e_t_consistent: torch.Tensor = (
                    current_latent_xt_scaled - sqrt_a_t * pred_x0_scaled_current_step
                ) / sqrt_one_minus_a_t.clamp(min=1e-8)

                dir_xt = torch.sqrt(
                    torch.clamp(1.0 - a_prev - sigma_t**2, min=1e-8)
                ) * torch.nan_to_num(e_t_consistent, nan=0.0, posinf=3.5, neginf=-3.5)

                if denoiser_ddim_eta > 0.0:
                    noise_ddim_buffer.normal_(generator=rng)
                    noise_ddim = sigma_t * noise_ddim_buffer
                else:
                    noise_ddim = 0.0

                current_latent_xt_scaled = torch.nan_to_num(
                    torch.sqrt(a_prev) * pred_x0_scaled_current_step
                    + dir_xt
                    + noise_ddim,
                    nan=0.0,
                    posinf=LATENT_DYNAMIC_BOUND,
                    neginf=-LATENT_DYNAMIC_BOUND,
                )

            if COARSE_GRAIN_LAMBDA > 0.0:
                latent_delta_ddim: torch.Tensor = (
                    pred_x0_scaled_current_step - lq_latent_x0_scaled_for_unet
                )
                clamped_delta_ddim: torch.Tensor = torch.clamp(
                    latent_delta_ddim, -2.2, 2.2
                )
                effective_delta_ddim: torch.Tensor = torch.lerp(
                    latent_delta_ddim, clamped_delta_ddim, COARSE_GRAIN_LAMBDA
                )
                final_denoised_latent_x0_scaled = (
                    lq_latent_x0_scaled_for_unet + effective_delta_ddim
                )
            else:
                final_denoised_latent_x0_scaled = pred_x0_scaled_current_step
        else:
            print(
                f"[ERROR] Denoiser: Unknown mode '{denoiser_mode}'. Skipping denoiser pass."
            )
            return image_cxhxw_uint8

        if final_denoised_latent_x0_scaled is None:
            return image_cxhxw_uint8

        # Hard safety clamp on final latent prior to VAE decode
        final_latent_sanitized: torch.Tensor = torch.nan_to_num(
            final_denoised_latent_x0_scaled / self.vae_scale_factor,
            nan=0.0,
            posinf=2.8,
            neginf=-2.8,
        ).clamp_(-2.8, 2.8)

        del final_denoised_latent_x0_scaled

        decoded_image_normalized_bchw: torch.Tensor = torch.empty(
            (1, 3, h_proc, w_proc),
            dtype=torch.float32,
            device=self.models_processor.device,
        ).contiguous()

        self.function_worker.run_vae_decoder(
            final_latent_sanitized, decoded_image_normalized_bchw
        )
        del final_latent_sanitized

        # --- POST-DECODER HOT-PIXEL & NAN NEUTRALIZATION ---
        # Eliminate any raw NaN or Inf created by VAE FP16 attention/GroupNorm overflow
        decoded_image_normalized_bchw = torch.nan_to_num(
            decoded_image_normalized_bchw, nan=0.0, posinf=1.0, neginf=-1.0
        )

        decoded_image_normalized_bchw.tanh_()
        image_after_postproc_float_0_1: torch.Tensor = (
            decoded_image_normalized_bchw.squeeze(0)
            .add_(1.0)
            .mul_(0.5)
            .clamp_(0.0, 1.0)
        )

        # Cavity Hot-Spot Neutralization:
        # Replaces any runaway pixel blowout where output is bright (>0.75) but original input
        # is deep cavity shadow (<0.20), such as nostril interiors or dark tear ducts.
        input_ref_float_0_1: torch.Tensor = image_to_process_cxhxw_uint8.float() / 255.0
        input_luma: torch.Tensor = (
            0.299 * input_ref_float_0_1[0:1]
            + 0.587 * input_ref_float_0_1[1:2]
            + 0.114 * input_ref_float_0_1[2:3]
        )
        output_luma: torch.Tensor = (
            0.299 * image_after_postproc_float_0_1[0:1]
            + 0.587 * image_after_postproc_float_0_1[1:2]
            + 0.114 * image_after_postproc_float_0_1[2:3]
        )

        hot_spot_mask: torch.Tensor = (input_luma < 0.20) & (output_luma > 0.75)
        if hot_spot_mask.any():
            image_after_postproc_float_0_1 = torch.where(
                hot_spot_mask.expand_as(image_after_postproc_float_0_1),
                input_ref_float_0_1,
                image_after_postproc_float_0_1,
            )

        # --- COLOR MATCHING BLOCK ---
        if ENABLE_COLOR_MATCH:
            # We scale res_tensor to [0, 255] float32 as expected by faceutil modules
            ref_tensor: torch.Tensor = image_to_process_cxhxw_uint8
            res_tensor: torch.Tensor = image_after_postproc_float_0_1 * 255.0

            # Secure mask formatting and scaling alignment
            if color_mask is not None:
                mask: torch.Tensor = color_mask.clone()
                if mask.dim() == 2:
                    mask = mask.unsqueeze(0)
                if mask.shape[-1] != ref_tensor.shape[-1]:
                    mask = v2.functional.resize(
                        mask,
                        [ref_tensor.shape[-2], ref_tensor.shape[-1]],
                        antialias=True,
                    ).squeeze(0)
                else:
                    mask = mask.squeeze(0)
            else:
                mask = (ref_tensor.sum(dim=0) > 0).float()

            try:
                # Fast, lightweight Reinhard statistical transfer in LAB space.
                # Locked to 100% blend strength using the facial region mask.
                matched_result: torch.Tensor = faceutil.apply_reinhard_color_transfer(
                    ref_tensor, res_tensor, 100.0, mask
                )
                image_after_postproc_float_0_1 = matched_result / 255.0
            except Exception as e:
                print(f"[WARN] Denoiser Color matching execution failed: {e}")

        # --- PIXEL SHARPENING (With Controllable Amplitude Coring) ---
        if ENABLE_PIXEL_SHARPENING:
            blurred: torch.Tensor = v2.functional.gaussian_blur(
                image_after_postproc_float_0_1.unsqueeze(0), [3, 3], [0.8, 0.8]
            ).squeeze(0)
            raw_detail: torch.Tensor = image_after_postproc_float_0_1 - blurred

            if COARSE_GRAIN_LAMBDA > 0.0:
                coring_gate: torch.Tensor = torch.clamp(
                    (torch.abs(raw_detail) - 0.015) / 0.035, 0.0, 1.0
                )
                gated_detail: torch.Tensor = raw_detail * coring_gate
                effective_detail: torch.Tensor = torch.lerp(
                    raw_detail, gated_detail, COARSE_GRAIN_LAMBDA
                )
            else:
                effective_detail = raw_detail

            image_after_postproc_float_0_1 = (
                image_after_postproc_float_0_1
                + effective_detail * PIXEL_SHARPEN_STRENGTH
            ).clamp_(0.0, 1.0)

        # --- POST-VAE SUB-PIXEL MICRO-GRAIN SYNTHESIS FILTER ---
        if MICRO_GRAIN_STRENGTH > 0.0:
            # Monochromatic sub-pixel noise on GPU
            grain_noise: torch.Tensor = torch.randn(
                (1, 1, h_proc, w_proc),
                device=self.models_processor.device,
                dtype=torch.float32,
                generator=rng,
            )
            # Slight spatial shaping gives natural film/pore structure rather than raw salt-and-pepper noise
            grain_shaped: torch.Tensor = v2.functional.gaussian_blur(
                grain_noise, kernel_size=[3, 3], sigma=[0.5, 0.5]
            ).squeeze(0)

            # Rec.709 Luminance Response: grain naturally appears in midtones, fading out in crushed darks and highlights
            luma: torch.Tensor = (
                0.2126 * image_after_postproc_float_0_1[0:1]
                + 0.7152 * image_after_postproc_float_0_1[1:2]
                + 0.0722 * image_after_postproc_float_0_1[2:3]
            )
            luma_gate: torch.Tensor = torch.clamp(4.0 * luma * (1.0 - luma), 0.0, 1.0)

            grain_delta: torch.Tensor = (
                grain_shaped * (MICRO_GRAIN_STRENGTH * 0.08) * luma_gate
            )

            # If a facial mask is provided, confine grain strictly within the face boundary
            if color_mask is not None:
                g_mask: torch.Tensor = color_mask.clone()
                if g_mask.dim() == 2:
                    g_mask = g_mask.unsqueeze(0)
                if g_mask.shape[-1] != w_proc or g_mask.shape[-2] != h_proc:
                    g_mask = v2.functional.resize(
                        g_mask, [h_proc, w_proc], antialias=True
                    )
                if g_mask.dim() == 3 and g_mask.shape[0] != 1:
                    g_mask = g_mask[0:1]
                grain_delta = grain_delta * g_mask.clamp(0.0, 1.0)

            image_after_postproc_float_0_1 = (
                image_after_postproc_float_0_1 + grain_delta
            ).clamp_(0.0, 1.0)

        final_image_uint8: torch.Tensor = (
            image_after_postproc_float_0_1 * 255.0
        ).byte()

        if h_proc != h_input or w_proc != w_input:
            output_image_cxhxw_uint8: torch.Tensor = v2.functional.resize(
                final_image_uint8,
                [h_input, w_input],
                interpolation=v2.InterpolationMode.BILINEAR,
                antialias=True,
            )
        else:
            output_image_cxhxw_uint8 = final_image_uint8

        if kv_tensor_map_for_this_run is not None:
            del kv_tensor_map_for_this_run

        del image_after_postproc_float_0_1
        del final_image_uint8

        return output_image_cxhxw_uint8
