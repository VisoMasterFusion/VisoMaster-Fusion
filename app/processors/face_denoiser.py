import os
import threading
import gc
import traceback
from typing import TYPE_CHECKING, Dict, Optional
from collections import OrderedDict

import torch
import numpy as np
from torchvision.transforms import v2

if TYPE_CHECKING:
    from app.processors.models_processor import ModelsProcessor
    from PIL import Image

from app.processors.utils import faceutil
from app.helpers.miscellaneous import is_file_exists
from app.helpers.downloader import download_file
from app.processors.utils.ref_ldm_kv_embedding import KVExtractor


class FaceDenoiser:
    """
    Handles Diffusion-based Denoiser/Restorer (ReF-LDM) operations.
    Manages DDIM/DDPM mathematical schedules and VAE latent processing.
    """

    def __init__(self, models_processor: "ModelsProcessor"):
        self.models_processor = models_processor

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
        if t.ndim == 0:
            t = t.unsqueeze(0)
        b = t.shape[0]
        out = torch.gather(a, 0, t.long())
        return out.reshape(b, *((1,) * (len(x_shape) - 1)))

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

    def get_kv_map_for_face(
        self, input_face_image_pil: "Image.Image", unload_after: bool = True
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Loads the KV Extractor, extracts K/V maps, and unloads (if unload_after is True).
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
            kv_map = self.kv_extractor.extract_kv(input_face_image_pil)
            print(
                f"[INFO] Successfully extracted K/V for {len(kv_map)} attention layers."
            )

        except Exception as e:
            print(f"[ERROR] Failed the K/V extraction: {e}")
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
        color_transfer: int = 100,
        color_transfer_mode: str = "Reinhard Transfer (Masked)",
        color_mask: torch.Tensor | None = None,
        blend_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Runs the Diffusion-based Denoiser/Restorer (ReF-LDM).
        Supports 'Single Step' (Fast) and 'Full Restore' (DDIM) modes.
        Also handles pixel sharpening and histogram matching for color consistency.
        """
        # --- CONFIGURATION ---
        ENABLE_PIXEL_SHARPENING = latent_sharpening_strength > 0.0
        PIXEL_SHARPEN_STRENGTH = latent_sharpening_strength

        ENABLE_COLOR_MATCH = color_transfer > 0
        COLOR_STRENGTH = color_transfer

        # P2-04: enable debug output via env var: set VISOMASTER_DEBUG_DENOISER=1
        DEBUG_DENOISER = os.environ.get("VISOMASTER_DEBUG_DENOISER", "0") == "1"
        unet_model_name = self.models_processor.main_window.fixed_unet_model_name
        vae_encoder_name = "RefLDMVAEEncoder"
        vae_decoder_name = "RefLDMVAEDecoder"

        if DEBUG_DENOISER:
            print(
                f"\n--- Denoiser Pass Start: Mode='{denoiser_mode}', CFG Scale={denoiser_cfg_scale}, VAE Scale Factor={self.vae_scale_factor} ---"
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

        kv_tensor_map_for_this_run: Dict[str, Dict[str, torch.Tensor]] | None = None
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
                print(f"[ERROR] Denoiser: Error deep copying K/V map: {e}. Skipping.")
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

        target_proc_dim = 512
        _, h_input, w_input = image_cxhxw_uint8.shape
        if h_input != target_proc_dim or w_input != target_proc_dim:
            image_to_process_cxhxw_uint8 = v2.functional.resize(
                image_cxhxw_uint8,
                [target_proc_dim, target_proc_dim],
                interpolation=v2.InterpolationMode.BILINEAR,
                antialias=True,
            )
        else:
            image_to_process_cxhxw_uint8 = image_cxhxw_uint8

        h_proc, w_proc = (
            image_to_process_cxhxw_uint8.shape[1],
            image_to_process_cxhxw_uint8.shape[2],
        )

        image_srgb_float_minus1_1 = (image_to_process_cxhxw_uint8.float() / 127.5) - 1.0
        image_srgb_float_minus1_1_batched = image_srgb_float_minus1_1.unsqueeze(
            0
        ).contiguous()

        latent_h, latent_w = h_proc // 8, w_proc // 8
        encoded_latent_direct_vae_out_bchw = torch.empty(
            (1, 8, latent_h, latent_w),
            dtype=torch.float32,
            device=self.models_processor.device,
        ).contiguous()

        self.models_processor.face_restorers.run_vae_encoder(
            image_srgb_float_minus1_1_batched, encoded_latent_direct_vae_out_bchw
        )

        lq_latent_x0_scaled_for_unet = (
            encoded_latent_direct_vae_out_bchw * self.vae_scale_factor
        )
        del encoded_latent_direct_vae_out_bchw
        del image_srgb_float_minus1_1_batched
        final_denoised_latent_x0_scaled = None

        if use_reference_exclusive_path:
            is_ref_flag_tensor_for_unet = torch.ones(
                1, dtype=torch.bool, device=self.models_processor.device
            )
        else:
            is_ref_flag_tensor_for_unet = torch.zeros(
                1, dtype=torch.bool, device=self.models_processor.device
            )

        actual_use_exclusive_path_tensor_for_unet = is_ref_flag_tensor_for_unet
        false_tensor_for_unet = torch.zeros(
            1, dtype=torch.bool, device=self.models_processor.device
        )

        rng = torch.Generator(device=self.models_processor.device)
        rng.manual_seed(base_seed)

        # --- PROCESS: Single Step ---
        if denoiser_mode == "Single Step (Fast)":
            rng.manual_seed(base_seed + denoiser_single_step_t)
            noise_sample = torch.randn(
                lq_latent_x0_scaled_for_unet.shape,
                device=self.models_processor.device,
                dtype=lq_latent_x0_scaled_for_unet.dtype,
                generator=rng,
            )

            current_t_idx = min(
                max(0, denoiser_single_step_t), len(self.alphas_cumprod_np) - 1
            )
            alpha_t_bar_val = self.alphas_cumprod_np[current_t_idx]

            sqrt_alpha_bar_t_torch = torch.sqrt(
                torch.full(
                    (1,),
                    alpha_t_bar_val,
                    dtype=torch.float32,
                    device=self.models_processor.device,
                )
            )
            sqrt_one_minus_alpha_bar_t_torch = torch.sqrt(
                1.0
                - torch.full(
                    (1,),
                    alpha_t_bar_val,
                    dtype=torch.float32,
                    device=self.models_processor.device,
                )
            )

            xt_noisy_scaled_8_channel = (
                lq_latent_x0_scaled_for_unet * sqrt_alpha_bar_t_torch
                + noise_sample * sqrt_one_minus_alpha_bar_t_torch
            )
            unet_input_16_channel = torch.cat(
                (xt_noisy_scaled_8_channel, lq_latent_x0_scaled_for_unet), dim=1
            )

            timesteps_tensor_unet = torch.full(
                (1,),
                current_t_idx,
                dtype=torch.int64,
                device=self.models_processor.device,
            )

            predicted_noise_from_unet = torch.empty(
                (1, 8, latent_h, latent_w),
                dtype=torch.float32,
                device=self.models_processor.device,
            ).contiguous()

            if torch.cuda.is_available():
                torch.cuda.current_stream().synchronize()

            self.models_processor.face_restorers.run_ref_ldm_unet(
                x_noisy_plus_lq_latent=unet_input_16_channel,
                timesteps_tensor=timesteps_tensor_unet,
                is_ref_flag_tensor=is_ref_flag_tensor_for_unet,
                use_reference_exclusive_path_globally_tensor=actual_use_exclusive_path_tensor_for_unet,
                kv_tensor_map=kv_tensor_map_for_this_run,
                output_unet_tensor=predicted_noise_from_unet,
            )
            final_denoised_latent_x0_scaled = (
                xt_noisy_scaled_8_channel
                - sqrt_one_minus_alpha_bar_t_torch * predicted_noise_from_unet
            ) / sqrt_alpha_bar_t_torch

        # --- PROCESS: Full Restore (DDIM) ---
        elif denoiser_mode == "Full Restore (DDIM)":
            num_ddpm_timesteps = self.alphas_cumprod_np.shape[0]

            _ddim_raw_ddpm_timesteps_np = self.make_ddim_timesteps(
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

            ddim_sigmas = (
                torch.from_numpy(_ddim_sigmas_np)
                .float()
                .to(self.models_processor.device, non_blocking=True)
            )
            ddim_alphas = (
                torch.from_numpy(_ddim_alphas_np)
                .float()
                .to(self.models_processor.device, non_blocking=True)
            )
            ddim_alphas_prev = (
                torch.from_numpy(_ddim_alphas_prev_np)
                .float()
                .to(self.models_processor.device, non_blocking=True)
            )

            ddim_sqrt_one_minus_alphas = torch.sqrt(
                torch.clamp(1.0 - ddim_alphas, min=0.0)
            )

            current_latent_xt_scaled = torch.randn(
                lq_latent_x0_scaled_for_unet.shape,
                device=self.models_processor.device,
                dtype=lq_latent_x0_scaled_for_unet.dtype,
                generator=rng,
            )
            time_range_ddpm_indices = np.flip(_ddim_raw_ddpm_timesteps_np).copy()
            total_steps = len(time_range_ddpm_indices)

            pred_x0_scaled_current_step = torch.empty_like(lq_latent_x0_scaled_for_unet)

            ts_unet = torch.empty(
                (1,), dtype=torch.int64, device=self.models_processor.device
            )
            schedule_idx_tensor = torch.empty(
                (1,), dtype=torch.long, device=self.models_processor.device
            )
            e_t_cond = torch.empty_like(lq_latent_x0_scaled_for_unet)
            e_t_uncond = (
                torch.empty_like(lq_latent_x0_scaled_for_unet)
                if denoiser_cfg_scale != 1.0
                else None
            )
            noise_ddim_buffer = torch.empty_like(lq_latent_x0_scaled_for_unet)

            # Pre-allocate the 16-channel UNet input buffer once.
            unet_input_16_channel = torch.empty(
                (1, 16, latent_h, latent_w),
                dtype=torch.float32,
                device=self.models_processor.device,
            ).contiguous()

            # The condition (LQ image) remains static. Write it to channels 8-15 once.
            unet_input_16_channel[:, 8:16] = lq_latent_x0_scaled_for_unet

            for i, step_ddpm_idx in enumerate(time_range_ddpm_indices):
                index_for_schedules = total_steps - 1 - i

                ts_unet.fill_(step_ddpm_idx)
                schedule_idx_tensor.fill_(index_for_schedules)

                # Update only the dynamic noisy channels (0-7) in-place.
                # No more torch.cat memory fragmentation inside the loop.
                unet_input_16_channel[:, :8] = current_latent_xt_scaled

                if torch.cuda.is_available():
                    torch.cuda.current_stream().synchronize()

                self.models_processor.face_restorers.run_ref_ldm_unet(
                    x_noisy_plus_lq_latent=unet_input_16_channel,
                    timesteps_tensor=ts_unet,
                    is_ref_flag_tensor=is_ref_flag_tensor_for_unet,
                    use_reference_exclusive_path_globally_tensor=actual_use_exclusive_path_tensor_for_unet,
                    kv_tensor_map=kv_tensor_map_for_this_run,
                    output_unet_tensor=e_t_cond,
                )

                if denoiser_cfg_scale != 1.0:
                    if torch.cuda.is_available():
                        torch.cuda.current_stream().synchronize()

                    # We re-use unet_input_16_channel directly.
                    # It contains the exact same data needed for the uncond pass.
                    self.models_processor.face_restorers.run_ref_ldm_unet(
                        x_noisy_plus_lq_latent=unet_input_16_channel,
                        timesteps_tensor=ts_unet,
                        is_ref_flag_tensor=is_ref_flag_tensor_for_unet,
                        use_reference_exclusive_path_globally_tensor=false_tensor_for_unet,
                        kv_tensor_map=None,
                        output_unet_tensor=e_t_uncond,
                    )
                    # In-place CFG math to save memory overhead
                    # Formula: e_t = e_t_uncond + denoiser_cfg_scale * (e_t_cond - e_t_uncond)
                    e_t = (
                        e_t_cond.sub_(e_t_uncond)
                        .mul_(denoiser_cfg_scale)
                        .add_(e_t_uncond)
                    )
                else:
                    e_t = e_t_cond

                a_t = self.extract_into_tensor_torch(
                    ddim_alphas, schedule_idx_tensor, current_latent_xt_scaled.shape
                )
                a_prev = self.extract_into_tensor_torch(
                    ddim_alphas_prev,
                    schedule_idx_tensor,
                    current_latent_xt_scaled.shape,
                )
                sigma_t = self.extract_into_tensor_torch(
                    ddim_sigmas, schedule_idx_tensor, current_latent_xt_scaled.shape
                )
                sqrt_one_minus_a_t = self.extract_into_tensor_torch(
                    ddim_sqrt_one_minus_alphas,
                    schedule_idx_tensor,
                    current_latent_xt_scaled.shape,
                )

                pred_x0_scaled_current_step = (
                    current_latent_xt_scaled - sqrt_one_minus_a_t * e_t
                ) / torch.sqrt(a_t).clamp(min=1e-8)

                dir_xt = (
                    torch.sqrt(torch.clamp(1.0 - a_prev - sigma_t**2, min=1e-8)) * e_t
                )

                noise_ddim_buffer.normal_(generator=rng)
                noise_ddim = sigma_t * noise_ddim_buffer

                current_latent_xt_scaled = (
                    torch.sqrt(a_prev) * pred_x0_scaled_current_step
                    + dir_xt
                    + noise_ddim
                )

            final_denoised_latent_x0_scaled = pred_x0_scaled_current_step
        else:
            print(
                f"[ERROR] Denoiser: Unknown mode '{denoiser_mode}'. Skipping denoiser pass."
            )
            return image_cxhxw_uint8

        if final_denoised_latent_x0_scaled is None:
            return image_cxhxw_uint8

        latent_for_vae_decoder = final_denoised_latent_x0_scaled / self.vae_scale_factor
        del final_denoised_latent_x0_scaled
        decoded_image_normalized_bchw = torch.empty(
            (1, 3, h_proc, w_proc),
            dtype=torch.float32,
            device=self.models_processor.device,
        ).contiguous()

        self.models_processor.face_restorers.run_vae_decoder(
            latent_for_vae_decoder, decoded_image_normalized_bchw
        )
        del latent_for_vae_decoder

        decoded_image_soft_clamped_bchw = torch.tanh(decoded_image_normalized_bchw)
        del decoded_image_normalized_bchw
        image_after_postproc_float_0_1 = (
            decoded_image_soft_clamped_bchw.squeeze(0) + 1.0
        ) / 2.0
        image_after_postproc_float_0_1 = torch.clamp(
            image_after_postproc_float_0_1, 0.0, 1.0
        )

        # --- COLOR MATCHING BLOCK ---
        if ENABLE_COLOR_MATCH:
            # We scale res_tensor to [0, 255] float32 as expected by faceutil modules
            ref_tensor = image_to_process_cxhxw_uint8
            res_tensor = image_after_postproc_float_0_1 * 255.0

            # Secure mask formatting and scaling alignment
            if color_mask is not None:
                mask = color_mask.clone()
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
                if color_transfer_mode == "CDF Histogram":
                    matched_result = faceutil.histogram_matching(
                        ref_tensor, res_tensor, float(COLOR_STRENGTH)
                    )
                elif color_transfer_mode == "CDF Histogram (Masked)":
                    matched_result = faceutil.histogram_matching_withmask(
                        ref_tensor, res_tensor, mask, float(COLOR_STRENGTH)
                    )
                elif color_transfer_mode == "Reinhard Transfer":
                    matched_result = faceutil.apply_reinhard_color_transfer(
                        ref_tensor, res_tensor, float(COLOR_STRENGTH), mask=None
                    )
                elif color_transfer_mode == "Reinhard Transfer (Masked)":
                    matched_result = faceutil.apply_reinhard_color_transfer(
                        ref_tensor, res_tensor, float(COLOR_STRENGTH), mask
                    )
                elif color_transfer_mode == "AdaIN (Core Masked)":
                    # For AdaIN: source and target are swapped so the generated face (res_tensor)
                    # matches the statistics of the raw input face (ref_tensor)
                    matched_result = faceutil.apply_adain_color_transfer(
                        res_tensor,
                        ref_tensor,
                        mask if blend_mask is None else blend_mask,
                        blend_amount=float(COLOR_STRENGTH),
                        calc_mask=mask,
                    )
                else:
                    matched_result = res_tensor

                image_after_postproc_float_0_1 = matched_result / 255.0

            except Exception as e:
                print(f"[WARN] Denoiser Color matching execution failed: {e}")

        if ENABLE_PIXEL_SHARPENING:
            blurred = v2.functional.gaussian_blur(
                image_after_postproc_float_0_1.unsqueeze(0), [5, 5], [1.0, 1.0]
            ).squeeze(0)
            detail = image_after_postproc_float_0_1 - blurred
            image_after_postproc_float_0_1 = (
                image_after_postproc_float_0_1 + detail * PIXEL_SHARPEN_STRENGTH
            )
            image_after_postproc_float_0_1 = image_after_postproc_float_0_1.clamp(
                0.0, 1.0
            )

        final_image_uint8 = (image_after_postproc_float_0_1 * 255.0).byte()

        if h_proc != h_input or w_proc != w_input:
            output_image_cxhxw_uint8 = v2.functional.resize(
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
