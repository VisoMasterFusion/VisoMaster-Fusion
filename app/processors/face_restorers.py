import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import torch
import numpy as np
from torchvision.transforms import v2
from skimage import transform as trans
import kornia.geometry.transform as kgm

if TYPE_CHECKING:
    from app.processors.models_processor import ModelsProcessor
    from app.processors.workers.function_worker import FunctionWorker


class FaceRestorers:
    osdface_model_names = (
        "OSDFacePromptEncoder",
        "OSDFaceVAEEncoder",
        "OSDFaceUNet",
        "OSDFaceVAEDecoder",
    )

    def __init__(
        self,
        models_processor: "ModelsProcessor",
        function_worker: "FunctionWorker",
    ):
        self.models_processor = models_processor
        self.function_worker = function_worker
        self.active_model_slot1: Optional[str] = None
        self.active_model_slot2: Optional[str] = None
        self._warned_models: set[str] = set()
        self.model_map: Dict[str, str] = {
            "GFPGAN-v1.4": "GFPGANv1.4",
            "GFPGAN-1024": "GFPGAN1024",
            "CodeFormer": "CodeFormer",
            "GPEN-256": "GPENBFR256",
            "GPEN-512": "GPENBFR512",
            "GPEN-1024": "GPENBFR1024",
            "GPEN-2048": "GPENBFR2048",
            "RestoreFormer++": "RestoreFormerPlusPlus",
            "VQFR-v2": "VQFRv2",
            "OSDFace": "OSDFace",
        }
        self._osdface_alphas_cumprod: Optional[list[float]] = None
        self._osdface_timestep: Optional[int] = None
        self._osdface_alpha: Optional[float] = None

    def unload_models(self) -> None:
        """Unloads all restorer models from memory and resets cached schedule state."""
        for restorer_name in set(self.model_map.values()):
            if restorer_name == "OSDFace":
                for sub_name in self.osdface_model_names:
                    self.models_processor.unload_model(sub_name)
            else:
                self.models_processor.unload_model(restorer_name)

        self.active_model_slot1 = None
        self.active_model_slot2 = None
        self._osdface_timestep = None
        self._osdface_alpha = None
        self._osdface_alphas_cumprod = None

    def _get_model_session(self, model_name: str) -> Optional[Any]:
        """
        Gets the model session by calling the centralized, provider-aware loader
        in ModelsProcessor. This ensures correct logging, caching, and provider handling.
        """
        # All complex logic is now delegated to the main loader.
        ort_session = self.models_processor.load_model(model_name)
        if not ort_session:
            if model_name not in self._warned_models:
                print(
                    f"[WARN] Model '{model_name}' failed to load or is not available. This operation will be skipped."
                )
                self._warned_models.add(model_name)
            return None
        return ort_session

    def _run_model_with_lazy_build_check(
        self, model_name: str, ort_session: Any, io_binding: Any
    ) -> None:
        """
        Runs the ONNX session with IOBinding, handling TensorRT lazy build dialogs.
        This centralizes the try/finally logic for showing/hiding the build progress dialog
        and includes the critical synchronization step for CUDA or other devices.

        Args:
            model_name (str): The name of the model being run.
            ort_session: The ONNX Runtime session instance.
            io_binding: The pre-configured IOBinding object.
        """
        is_lazy_build: bool = self.models_processor.check_and_clear_pending_build(
            model_name
        )
        if is_lazy_build:
            self.models_processor.show_build_dialog.emit(
                "Finalizing TensorRT Build",
                f"Performing first-run inference for:\n{model_name}\n\nThis may take several minutes.",
            )

        try:
            self.function_worker.run_ort_with_iobinding(ort_session, io_binding)
        finally:
            if is_lazy_build:
                self.models_processor.hide_build_dialog.emit()

    def _ensure_osdface_scheduler_loaded(self) -> Optional[List[float]]:
        """Loads and caches the static alphas_cumprod schedule to prevent disk I/O during playback."""
        if self._osdface_alphas_cumprod is not None:
            return self._osdface_alphas_cumprod

        scheduler_path = self.models_processor.models_path.get("OSDFaceScheduler")
        if not scheduler_path:
            if "OSDFaceScheduler" not in self._warned_models:
                print("[WARN] OSDFace scheduler metadata path is not registered.")
                self._warned_models.add("OSDFaceScheduler")
            return None

        sched_file = Path(scheduler_path)
        if not sched_file.is_file():
            if "OSDFaceScheduler" not in self._warned_models:
                print(
                    f"[WARN] OSDFace scheduler metadata file not found at: {scheduler_path}"
                )
                self._warned_models.add("OSDFaceScheduler")
            return None

        try:
            scheduler_data = json.loads(sched_file.read_text(encoding="utf-8"))
            alphas_cumprod = scheduler_data.get("alphas_cumprod")
            if not isinstance(alphas_cumprod, list) or len(alphas_cumprod) == 0:
                print(
                    "[WARN] OSDFace scheduler metadata contains an invalid 'alphas_cumprod' schedule."
                )
                return None
            self._osdface_alphas_cumprod = [float(a) for a in alphas_cumprod]
            return self._osdface_alphas_cumprod
        except Exception as exc:
            print(f"[WARN] Failed to read OSDFace scheduler metadata: {exc}")
            return None

    def _get_osdface_alpha(self, timestep_value: int) -> Optional[Tuple[int, float]]:
        """Returns the bounded timestep index and precomputed alpha from memory."""
        if (
            self._osdface_timestep is not None
            and self._osdface_alpha is not None
            and self._osdface_timestep == timestep_value
        ):
            return self._osdface_timestep, self._osdface_alpha

        alphas_cumprod = self._ensure_osdface_scheduler_loaded()
        if alphas_cumprod is None:
            return None

        timestep: int = max(0, min(int(timestep_value), len(alphas_cumprod) - 1))
        alpha: float = alphas_cumprod[timestep]
        self._osdface_timestep = timestep
        self._osdface_alpha = alpha
        return timestep, alpha

    @torch.no_grad()
    def apply_facerestorer(
        self,
        swapped_face_upscaled: torch.Tensor,
        restorer_det_type: str,
        restorer_type: str,
        restorer_blend: float,
        fidelity_weight: float,
        detect_score: float,
        target_kps: Optional[np.ndarray] = None,
        slot_id: int = 1,
        osdface_timestep: int = 399,
        osdface_latent_strength: float = 1.0,
    ) -> torch.Tensor:
        model_name_to_load = self.model_map.get(restorer_type)
        if not model_name_to_load:
            return swapped_face_upscaled

        # --- Strict VRAM Lifecycle Management Across Slots ---
        current_active = (
            self.active_model_slot1 if slot_id == 1 else self.active_model_slot2
        )
        other_active = (
            self.active_model_slot2 if slot_id == 1 else self.active_model_slot1
        )

        if current_active is not None and current_active != restorer_type:
            if current_active != other_active:
                if current_active == "OSDFace":
                    for m_name in self.osdface_model_names:
                        self.models_processor.unload_model(m_name)
                else:
                    prev_model = self.model_map.get(current_active)
                    if prev_model and prev_model != "OSDFace":
                        self.models_processor.unload_model(prev_model)

        if slot_id == 1:
            self.active_model_slot1 = restorer_type
        else:
            self.active_model_slot2 = restorer_type

        # If using a separate detection mode
        if restorer_det_type in ["Blend", "Reference"]:
            if restorer_det_type == "Blend":
                # Set up Transformation
                dst = self.models_processor.arcface_dst * 4.0
                dst[:, 0] += 32.0

            elif restorer_det_type == "Reference":
                # Instead of re-detecting landmarks, use the target_kps passed to the function.
                if target_kps is None or len(target_kps) == 0:
                    print(
                        "[WARN] 'Reference' alignment selected, but no target landmarks (target_kps) were provided. Skipping restoration."
                    )
                    return swapped_face_upscaled
                dst = target_kps

            try:
                # Use from_estimate constructor instead of .estimate()
                if hasattr(trans.SimilarityTransform, "from_estimate"):
                    tform = trans.SimilarityTransform.from_estimate(
                        dst, self.models_processor.FFHQ_kps
                    )
                else:
                    tform = trans.SimilarityTransform()
                    tform.estimate(dst, self.models_processor.FFHQ_kps)
            except Exception:
                return swapped_face_upscaled

            # Push matrix to device with non_blocking=True to hide PCIe transfer latency
            M_tensor = (
                torch.from_numpy(tform.params[0:2])
                .to(
                    device=swapped_face_upscaled.device,
                    dtype=torch.float32,
                    non_blocking=True,
                )
                .unsqueeze(0)
            )
            img_b = (
                swapped_face_upscaled.unsqueeze(0)
                if swapped_face_upscaled.dim() == 3
                else swapped_face_upscaled
            )

            # Kornia allocates a new tensor here, so we own this memory space.
            temp = kgm.warp_affine(
                img_b.to(dtype=torch.float32, non_blocking=True),
                M_tensor,
                dsize=(512, 512),
                mode="bilinear",
                align_corners=True,
            ).squeeze(0)

            # Safe to perform in-place math since 'temp' is a brand new tensor from Kornia
            temp.mul_(1.0 / 255.0)

        else:
            # If we did not warp the image, we MUST clone the original tensor
            # copy=True safely detaches it for this thread; in-place math saves a VRAM allocation.
            temp = swapped_face_upscaled.to(
                dtype=torch.float32, copy=True, non_blocking=True
            ).mul_(1.0 / 255.0)

        # High-Fidelity Scaling BEFORE Normalization
        # Use Bicubic to preserve eyelashes/pores, and clamp to prevent GAN ringing artifacts.
        if restorer_type == "GPEN-1024":
            temp = v2.functional.resize(
                temp,
                [1024, 1024],
                interpolation=v2.InterpolationMode.BICUBIC,
                antialias=False,
            )
            temp.clamp_(0.0, 1.0)  # Kill bicubic overshoot
        elif restorer_type == "GPEN-2048":
            temp = v2.functional.resize(
                temp,
                [2048, 2048],
                interpolation=v2.InterpolationMode.BICUBIC,
                antialias=False,
            )
            temp.clamp_(0.0, 1.0)
        elif restorer_type == "GPEN-256":
            temp = v2.functional.resize(
                temp,
                [256, 256],
                interpolation=v2.InterpolationMode.BICUBIC,
                antialias=False,
            )
            temp.clamp_(0.0, 1.0)
        elif restorer_type == "OSDFace" and (temp.shape[-2], temp.shape[-1]) != (
            512,
            512,
        ):
            # OSDFace UNet strictly requires 64x64 latent -> 512x512 spatial input
            temp = v2.functional.resize(
                temp,
                [512, 512],
                interpolation=v2.InterpolationMode.BILINEAR,
                antialias=False,
            )
            temp.clamp_(0.0, 1.0)

        # In-place normalization [-1, 1]
        temp = v2.functional.normalize(
            temp, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True
        )
        temp = torch.unsqueeze(temp, 0).contiguous()

        # Bindings
        outpred: Optional[torch.Tensor] = None

        if restorer_type == "GFPGAN-v1.4":
            outpred = torch.empty(
                (1, 3, 512, 512),
                dtype=torch.float32,
                device=self.models_processor.device,
            ).contiguous()
            self.run_GFPGAN(temp, outpred)

        elif restorer_type == "GFPGAN-1024":
            outpred = torch.empty(
                (1, 3, 1024, 1024),
                dtype=torch.float32,
                device=self.models_processor.device,
            ).contiguous()
            self.run_GFPGAN1024(temp, outpred)

        elif restorer_type == "CodeFormer":
            outpred = torch.empty(
                (1, 3, 512, 512),
                dtype=torch.float32,
                device=self.models_processor.device,
            ).contiguous()
            self.run_codeformer(temp, outpred, fidelity_weight)

        elif restorer_type == "GPEN-256":
            outpred = torch.empty(
                (1, 3, 256, 256),
                dtype=torch.float32,
                device=self.models_processor.device,
            ).contiguous()
            self.run_GPEN_256(temp, outpred)

        elif restorer_type == "GPEN-512":
            outpred = torch.empty(
                (1, 3, 512, 512),
                dtype=torch.float32,
                device=self.models_processor.device,
            ).contiguous()
            self.run_GPEN_512(temp, outpred)

        elif restorer_type == "GPEN-1024":
            outpred = torch.empty(
                (1, 3, 1024, 1024),
                dtype=torch.float32,
                device=self.models_processor.device,
            ).contiguous()
            self.run_GPEN_1024(temp, outpred)

        elif restorer_type == "GPEN-2048":
            outpred = torch.empty(
                (1, 3, 2048, 2048),
                dtype=torch.float32,
                device=self.models_processor.device,
            ).contiguous()
            self.run_GPEN_2048(temp, outpred)

        elif restorer_type == "RestoreFormer++":
            outpred = torch.empty(
                (1, 3, 512, 512),
                dtype=torch.float32,
                device=self.models_processor.device,
            ).contiguous()
            self.run_RestoreFormerPlusPlus(temp, outpred)

        elif restorer_type == "VQFR-v2":
            outpred = torch.empty(
                (1, 3, 512, 512),
                dtype=torch.float32,
                device=self.models_processor.device,
            ).contiguous()
            self.run_VQFR_v2(temp, outpred, fidelity_weight)

        elif restorer_type == "OSDFace":
            outpred = torch.empty(
                (1, 3, 512, 512),
                dtype=torch.float32,
                device=self.models_processor.device,
            ).contiguous()
            success: bool = self.run_OSDFace(
                temp,
                outpred,
                timestep_value=osdface_timestep,
                latent_strength=osdface_latent_strength,
            )
            if not success:
                # Immediate non-corrupting fallback if inference fails
                return swapped_face_upscaled

        if outpred is None:
            return swapped_face_upscaled

        # Fused in-place math: ((x clamped [-1, 1]) + 1.0) * 127.5 -> [0.0, 255.0]
        outpred = outpred.squeeze(0).clamp_(-1.0, 1.0).add_(1.0).mul_(127.5)

        # High-Fidelity Downscaling
        if restorer_type in ["GPEN-256", "GPEN-1024", "GPEN-2048", "GFPGAN-1024"]:
            outpred = v2.functional.resize(
                outpred,
                [512, 512],
                interpolation=v2.InterpolationMode.BICUBIC,
                antialias=True,
            )
            outpred.clamp_(0.0, 255.0)  # Suppress ringing artifacts after downscale

        # Invert Transform
        if restorer_det_type in ["Blend", "Reference"]:
            # OPTIMIZED: Direct Inverse GPU Affine Warp with non_blocking=True
            M_inv_tensor = (
                torch.from_numpy(tform.inverse.params[0:2])
                .to(device=outpred.device, dtype=torch.float32, non_blocking=True)
                .unsqueeze(0)
            )
            # Correct 4D tensor unsqueeze for Kornia grid sample
            out_b = outpred.unsqueeze(0) if outpred.dim() == 3 else outpred
            dsize: Tuple[int, int] = (
                swapped_face_upscaled.shape[1],
                swapped_face_upscaled.shape[2],
            )

            outpred = kgm.warp_affine(
                out_b,
                M_inv_tensor,
                dsize=(dsize[0], dsize[1]),
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            ).squeeze(0)

        elif restorer_type == "OSDFace" and (outpred.shape[-2], outpred.shape[-1]) != (
            swapped_face_upscaled.shape[-2],
            swapped_face_upscaled.shape[-1],
        ):
            # Scale back to original frame crop bounds if not using affine warping
            outpred = v2.functional.resize(
                outpred,
                [swapped_face_upscaled.shape[-2], swapped_face_upscaled.shape[-1]],
                interpolation=v2.InterpolationMode.BILINEAR,
                antialias=True,
            )

        return outpred

    @torch.no_grad()
    def run_OSDFace(
        self,
        image: torch.Tensor,
        output: torch.Tensor,
        timestep_value: int = 399,
        latent_strength: float = 1.0,
    ) -> bool:
        """
        Executes the One-Step Diffusion Face restoration pipeline.
        Returns True if inference succeeded, or False on missing models / invalid metadata.
        """
        scheduler_state = self._get_osdface_alpha(int(timestep_value))
        if scheduler_state is None:
            return False
        timestep_idx, alpha_value = scheduler_state
        latent_strength_clamped: float = max(0.0, min(float(latent_strength), 1.0))

        bind_device = self.models_processor.device
        bind_device_type = self.models_processor.device_type
        bind_device_id = self.models_processor.binding_device_id

        # 1. Prompt Encoder (Maps image [-1, 1] to [0, 1])
        prompt_input = image.mul(0.5).add(0.5).clamp_(0.0, 1.0).contiguous()
        prompt_embeds = torch.empty(
            (1, 77, 1024), dtype=torch.float32, device=bind_device
        ).contiguous()

        prompt_session = self._get_model_session("OSDFacePromptEncoder")
        if prompt_session is None:
            return False
        io_binding_prompt = prompt_session.io_binding()
        io_binding_prompt.bind_input(
            name="lq_0_1",
            device_type=bind_device_type,
            device_id=bind_device_id,
            element_type=np.float32,
            shape=tuple(prompt_input.shape),
            buffer_ptr=prompt_input.data_ptr(),
        )
        io_binding_prompt.bind_output(
            name="prompt_embeds",
            device_type=bind_device_type,
            device_id=bind_device_id,
            element_type=np.float32,
            shape=tuple(prompt_embeds.shape),
            buffer_ptr=prompt_embeds.data_ptr(),
        )
        self._run_model_with_lazy_build_check(
            "OSDFacePromptEncoder", prompt_session, io_binding_prompt
        )

        # 2. VAE Encoder (Maps image [-1, 1] to latent space)
        latent = torch.empty(
            (1, 4, 64, 64), dtype=torch.float32, device=bind_device
        ).contiguous()
        vae_encoder_session = self._get_model_session("OSDFaceVAEEncoder")
        if vae_encoder_session is None:
            return False
        io_binding_enc = vae_encoder_session.io_binding()
        io_binding_enc.bind_input(
            name="lq_neg1_1",
            device_type=bind_device_type,
            device_id=bind_device_id,
            element_type=np.float32,
            shape=tuple(image.shape),
            buffer_ptr=image.data_ptr(),
        )
        io_binding_enc.bind_output(
            name="latent",
            device_type=bind_device_type,
            device_id=bind_device_id,
            element_type=np.float32,
            shape=tuple(latent.shape),
            buffer_ptr=latent.data_ptr(),
        )
        self._run_model_with_lazy_build_check(
            "OSDFaceVAEEncoder", vae_encoder_session, io_binding_enc
        )

        # 3. UNet (Predicts epsilon noise conditioned on prompt and timestep)
        noise_pred = torch.empty_like(latent).contiguous()
        timestep_tensor = torch.tensor(
            [timestep_idx], dtype=torch.int64, device=bind_device
        ).contiguous()
        unet_session = self._get_model_session("OSDFaceUNet")
        if unet_session is None:
            return False
        io_binding_unet = unet_session.io_binding()
        io_binding_unet.bind_input(
            name="latent",
            device_type=bind_device_type,
            device_id=bind_device_id,
            element_type=np.float32,
            shape=tuple(latent.shape),
            buffer_ptr=latent.data_ptr(),
        )
        io_binding_unet.bind_input(
            name="timestep",
            device_type=bind_device_type,
            device_id=bind_device_id,
            element_type=np.int64,
            shape=tuple(timestep_tensor.shape),
            buffer_ptr=timestep_tensor.data_ptr(),
        )
        io_binding_unet.bind_input(
            name="prompt_embeds",
            device_type=bind_device_type,
            device_id=bind_device_id,
            element_type=np.float32,
            shape=tuple(prompt_embeds.shape),
            buffer_ptr=prompt_embeds.data_ptr(),
        )
        io_binding_unet.bind_output(
            name="noise_pred",
            device_type=bind_device_type,
            device_id=bind_device_id,
            element_type=np.float32,
            shape=tuple(noise_pred.shape),
            buffer_ptr=noise_pred.data_ptr(),
        )
        self._run_model_with_lazy_build_check(
            "OSDFaceUNet", unet_session, io_binding_unet
        )

        # 4. Fused Tweedie Reconstruction: x0 = (latent - sqrt(1 - alpha) * noise_pred) / sqrt(alpha)
        sqrt_beta: float = math.sqrt(max(0.0, 1.0 - alpha_value))
        sqrt_alpha: float = math.sqrt(max(1e-8, alpha_value))
        x0_latent = torch.add(latent, noise_pred, alpha=-sqrt_beta).div_(sqrt_alpha)

        if latent_strength_clamped < 1.0:
            x0_latent = torch.lerp(
                latent, x0_latent, latent_strength_clamped
            ).contiguous()

        # 5. VAE Decoder (Decodes x0_latent back to image [0, 1])
        decoded = torch.empty(
            (1, 3, 512, 512), dtype=torch.float32, device=bind_device
        ).contiguous()
        vae_decoder_session = self._get_model_session("OSDFaceVAEDecoder")
        if vae_decoder_session is None:
            return False
        io_binding_dec = vae_decoder_session.io_binding()
        io_binding_dec.bind_input(
            name="x0_latent",
            device_type=bind_device_type,
            device_id=bind_device_id,
            element_type=np.float32,
            shape=tuple(x0_latent.shape),
            buffer_ptr=x0_latent.data_ptr(),
        )
        io_binding_dec.bind_output(
            name="image_0_1",
            device_type=bind_device_type,
            device_id=bind_device_id,
            element_type=np.float32,
            shape=tuple(decoded.shape),
            buffer_ptr=decoded.data_ptr(),
        )
        self._run_model_with_lazy_build_check(
            "OSDFaceVAEDecoder", vae_decoder_session, io_binding_dec
        )

        # 6. In-place conversion from [0, 1] into [-1, 1] directly on pre-allocated output buffer
        output.copy_(decoded).mul_(2.0).sub_(1.0)
        return True

    @torch.no_grad()
    def run_vae_encoder(
        self, image_input_tensor: torch.Tensor, output_latent_tensor: torch.Tensor
    ) -> None:
        """
        Runs the VAE encoder model.
        image_input_tensor: Batch x 3 x Height x Width, float32, normalized to [-1, 1]
        output_latent_tensor: Placeholder for Batch x 8 x LatentH x LatentW, float32
        """
        model_name = "RefLDMVAEEncoder"
        # FR-BUG-04: use .get() to avoid KeyError when model is not yet loaded
        ort_session = self.models_processor.models.get(model_name)
        if ort_session is None:
            # Lazy reload via unified facade in case clear_gpu_memory() cleared the session
            self.function_worker.ensure_denoiser_models_loaded()
            ort_session = self.models_processor.models.get(model_name)
        if ort_session is None:
            error_msg = f"[ERROR] VAE Encoder model '{model_name}' not loaded when run_vae_encoder was called."
            print(error_msg)
            raise RuntimeError(error_msg)

        input_name = (
            ort_session.get_inputs()[0].name
            if ort_session.get_inputs()
            else "image_input"
        )
        output_name = (
            ort_session.get_outputs()[0].name
            if ort_session.get_outputs()
            else "latent_pre_quant_unscaled"
        )

        io_binding = ort_session.io_binding()
        io_binding.bind_input(
            name=input_name,
            device_type=self.models_processor.device_type,
            device_id=self.models_processor.binding_device_id,
            element_type=np.float32,
            shape=tuple(image_input_tensor.shape),
            buffer_ptr=image_input_tensor.data_ptr(),
        )
        io_binding.bind_output(
            name=output_name,
            device_type=self.models_processor.device_type,
            device_id=self.models_processor.binding_device_id,
            element_type=np.float32,
            shape=tuple(output_latent_tensor.shape),
            buffer_ptr=output_latent_tensor.data_ptr(),
        )

        # Run the model with lazy build handling
        self._run_model_with_lazy_build_check(model_name, ort_session, io_binding)

    @torch.no_grad()
    def run_vae_decoder(
        self, latent_input_tensor: torch.Tensor, output_image_tensor: torch.Tensor
    ) -> None:
        """
        Runs the VAE decoder model.
        latent_input_tensor: Batch x 8 x LatentH x LatentW, float32
        output_image_tensor: Placeholder for Batch x 3 x H x W, float32, normalized to [-1, 1]
        """
        model_name = "RefLDMVAEDecoder"
        # FR-BUG-04: use .get() to avoid KeyError when model is not yet loaded
        ort_session = self.models_processor.models.get(model_name)
        if ort_session is None:
            # Lazy reload via unified facade in case clear_gpu_memory() cleared the session
            self.function_worker.ensure_denoiser_models_loaded()
            ort_session = self.models_processor.models.get(model_name)
        if ort_session is None:
            error_msg = f"[ERROR] VAE Decoder model '{model_name}' not loaded when run_vae_decoder was called."
            print(error_msg)
            raise RuntimeError(error_msg)

        input_name = (
            ort_session.get_inputs()[0].name
            if ort_session.get_inputs()
            else "scaled_latent_input"
        )
        output_name = (
            ort_session.get_outputs()[0].name
            if ort_session.get_outputs()
            else "image_output"
        )

        io_binding = ort_session.io_binding()
        io_binding.bind_input(
            name=input_name,
            device_type=self.models_processor.device_type,
            device_id=self.models_processor.binding_device_id,
            element_type=np.float32,
            shape=tuple(latent_input_tensor.shape),
            buffer_ptr=latent_input_tensor.data_ptr(),
        )
        io_binding.bind_output(
            name=output_name,
            device_type=self.models_processor.device_type,
            device_id=self.models_processor.binding_device_id,
            element_type=np.float32,
            shape=tuple(output_image_tensor.shape),
            buffer_ptr=output_image_tensor.data_ptr(),
        )

        # Run the model with lazy build handling
        self._run_model_with_lazy_build_check(model_name, ort_session, io_binding)

    @torch.no_grad()
    def run_ref_ldm_unet(
        self,
        x_noisy_plus_lq_latent: torch.Tensor,
        timesteps_tensor: torch.Tensor,
        is_ref_flag_tensor: torch.Tensor,
        use_reference_exclusive_path_globally_tensor: torch.Tensor,
        kv_tensor_map: Optional[Dict[str, Dict[str, torch.Tensor]]],
        output_unet_tensor: torch.Tensor,
    ) -> None:
        """Runs the UNet denoiser model with external K/V inputs."""
        model_name = self.models_processor.main_window.fixed_unet_model_name
        ort_session = self.models_processor.models.get(model_name)

        if not ort_session:
            print(
                f"[ERROR] UNet model '{model_name}' not loaded when run_ref_ldm_unet was called."
            )
            return

        onnx_output_name = "unet_output"
        io_binding = ort_session.io_binding()
        bind_device_type = self.models_processor.device_type
        bind_device = self.models_processor.device
        bind_device_id = self.models_processor.binding_device_id

        # Bind standard inputs
        io_binding.bind_input(
            name="x_noisy_plus_lq_latent",
            device_type=bind_device_type,
            device_id=bind_device_id,
            element_type=np.float32,
            shape=tuple(x_noisy_plus_lq_latent.shape),
            buffer_ptr=x_noisy_plus_lq_latent.data_ptr(),
        )
        io_binding.bind_input(
            name="timesteps",
            device_type=bind_device_type,
            device_id=bind_device_id,
            element_type=np.int64,
            shape=tuple(timesteps_tensor.shape),
            buffer_ptr=timesteps_tensor.data_ptr(),
        )
        io_binding.bind_input(
            name="is_ref_flag_input",
            device_type=bind_device_type,
            device_id=bind_device_id,
            element_type=np.bool_,
            shape=tuple(is_ref_flag_tensor.shape),
            buffer_ptr=is_ref_flag_tensor.data_ptr(),
        )
        io_binding.bind_input(
            name="use_reference_exclusive_path_globally_input",
            device_type=bind_device_type,
            device_id=bind_device_id,
            element_type=np.bool_,
            shape=tuple(use_reference_exclusive_path_globally_tensor.shape),
            buffer_ptr=use_reference_exclusive_path_globally_tensor.data_ptr(),
        )

        onnx_model_inputs = ort_session.get_inputs()
        onnx_kv_input_names_to_shape: Dict[str, Tuple[int, ...]] = {
            inp.name: tuple(
                dim if isinstance(dim, int) and dim > 0 else 1 for dim in inp.shape
            )
            for inp in onnx_model_inputs
            if inp.name.endswith("_k_ext") or inp.name.endswith("_v_ext")
        }

        actual_kv_tensors_for_binding: Dict[str, torch.Tensor] = {}
        if kv_tensor_map:
            for pt_module_name, kv_pair in kv_tensor_map.items():
                onnx_base_name = pt_module_name.replace(".", "_")
                k_name_onnx = f"{onnx_base_name}_k_ext"
                v_name_onnx = f"{onnx_base_name}_v_ext"

                k_tensor_original = kv_pair.get("k")
                v_tensor_original = kv_pair.get("v")

                if (
                    k_tensor_original is not None
                    and k_name_onnx in onnx_kv_input_names_to_shape
                ):
                    actual_kv_tensors_for_binding[k_name_onnx] = (
                        k_tensor_original.unsqueeze(0)
                        .to(device=bind_device, dtype=torch.float32)
                        .contiguous()
                    )

                if (
                    v_tensor_original is not None
                    and v_name_onnx in onnx_kv_input_names_to_shape
                ):
                    actual_kv_tensors_for_binding[v_name_onnx] = (
                        v_tensor_original.unsqueeze(0)
                        .to(device=bind_device, dtype=torch.float32)
                        .contiguous()
                    )

        # IMPORTANT: Keep references to temporary zero tensors to prevent GC
        keep_alive_tensors: List[torch.Tensor] = []
        # FS-MEM-01: also keep actual KV tensors alive to prevent premature GC
        keep_alive_tensors.extend(actual_kv_tensors_for_binding.values())

        for onnx_kv_name, expected_shape in onnx_kv_input_names_to_shape.items():
            tensor_to_bind = actual_kv_tensors_for_binding.get(onnx_kv_name)
            if tensor_to_bind is None:
                # Create a zero tensor for missing K/V inputs (e.g., unconditional pass)
                tensor_to_bind = torch.zeros(
                    expected_shape, dtype=torch.float32, device=bind_device
                ).contiguous()
                # We MUST store this tensor in a list that persists for the function scope
                # Otherwise, it might be garbage collected before .run() is called
                keep_alive_tensors.append(tensor_to_bind)

            io_binding.bind_input(
                name=onnx_kv_name,
                device_type=bind_device_type,
                device_id=bind_device_id,
                element_type=np.float32,
                shape=tuple(tensor_to_bind.shape),
                buffer_ptr=tensor_to_bind.data_ptr(),
            )

        io_binding.bind_output(
            name=onnx_output_name,
            device_type=bind_device_type,
            device_id=bind_device_id,
            element_type=np.float32,
            shape=tuple(output_unet_tensor.shape),
            buffer_ptr=output_unet_tensor.data_ptr(),
        )

        # Run the model with lazy build handling
        self._run_model_with_lazy_build_check(model_name, ort_session, io_binding)

    def run_GFPGAN(self, image: torch.Tensor, output: torch.Tensor) -> None:
        model_name = "GFPGANv1.4"
        ort_session = self._get_model_session(model_name)
        if not ort_session:
            return  # Silently skip if model failed to load

        io_binding = ort_session.io_binding()
        io_binding.bind_input(
            name="input",
            device_type=self.models_processor.device_type,
            device_id=self.models_processor.binding_device_id,
            element_type=np.float32,
            shape=(1, 3, 512, 512),
            buffer_ptr=image.data_ptr(),
        )
        io_binding.bind_output(
            name="output",
            device_type=self.models_processor.device_type,
            device_id=self.models_processor.binding_device_id,
            element_type=np.float32,
            shape=(1, 3, 512, 512),
            buffer_ptr=output.data_ptr(),
        )
        # Run the model with lazy build handling
        self._run_model_with_lazy_build_check(model_name, ort_session, io_binding)

    def run_GFPGAN1024(self, image: torch.Tensor, output: torch.Tensor) -> None:
        model_name = "GFPGAN1024"
        ort_session = self._get_model_session(model_name)
        if not ort_session:
            return  # Silently skip if model failed to load

        io_binding = ort_session.io_binding()
        io_binding.bind_input(
            name="input",
            device_type=self.models_processor.device_type,
            device_id=self.models_processor.binding_device_id,
            element_type=np.float32,
            shape=(1, 3, 512, 512),
            buffer_ptr=image.data_ptr(),
        )
        io_binding.bind_output(
            name="output",
            device_type=self.models_processor.device_type,
            device_id=self.models_processor.binding_device_id,
            element_type=np.float32,
            shape=(1, 3, 1024, 1024),
            buffer_ptr=output.data_ptr(),
        )
        # Run the model with lazy build handling
        self._run_model_with_lazy_build_check(model_name, ort_session, io_binding)

    def run_GPEN_256(self, image: torch.Tensor, output: torch.Tensor) -> None:
        model_name = "GPENBFR256"
        ort_session = self._get_model_session(model_name)
        if not ort_session:
            return  # Silently skip if model failed to load

        io_binding = ort_session.io_binding()
        io_binding.bind_input(
            name="input",
            device_type=self.models_processor.device_type,
            device_id=self.models_processor.binding_device_id,
            element_type=np.float32,
            shape=(1, 3, 256, 256),
            buffer_ptr=image.data_ptr(),
        )
        io_binding.bind_output(
            name="output",
            device_type=self.models_processor.device_type,
            device_id=self.models_processor.binding_device_id,
            element_type=np.float32,
            shape=(1, 3, 256, 256),
            buffer_ptr=output.data_ptr(),
        )
        # Run the model with lazy build handling
        self._run_model_with_lazy_build_check(model_name, ort_session, io_binding)

    def run_GPEN_512(self, image: torch.Tensor, output: torch.Tensor) -> None:
        model_name = "GPENBFR512"
        ort_session = self._get_model_session(model_name)
        if not ort_session:
            return  # Silently skip if model failed to load

        io_binding = ort_session.io_binding()
        io_binding.bind_input(
            name="input",
            device_type=self.models_processor.device_type,
            device_id=self.models_processor.binding_device_id,
            element_type=np.float32,
            shape=(1, 3, 512, 512),
            buffer_ptr=image.data_ptr(),
        )
        io_binding.bind_output(
            name="output",
            device_type=self.models_processor.device_type,
            device_id=self.models_processor.binding_device_id,
            element_type=np.float32,
            shape=(1, 3, 512, 512),
            buffer_ptr=output.data_ptr(),
        )
        # Run the model with lazy build handling
        self._run_model_with_lazy_build_check(model_name, ort_session, io_binding)

    def run_GPEN_1024(self, image: torch.Tensor, output: torch.Tensor) -> None:
        model_name = "GPENBFR1024"
        ort_session = self._get_model_session(model_name)
        if not ort_session:
            return  # Silently skip if model failed to load

        io_binding = ort_session.io_binding()
        io_binding.bind_input(
            name="input",
            device_type=self.models_processor.device_type,
            device_id=self.models_processor.binding_device_id,
            element_type=np.float32,
            shape=(1, 3, 1024, 1024),
            buffer_ptr=image.data_ptr(),
        )
        io_binding.bind_output(
            name="output",
            device_type=self.models_processor.device_type,
            device_id=self.models_processor.binding_device_id,
            element_type=np.float32,
            shape=(1, 3, 1024, 1024),
            buffer_ptr=output.data_ptr(),
        )
        # Run the model with lazy build handling
        self._run_model_with_lazy_build_check(model_name, ort_session, io_binding)

    def run_GPEN_2048(self, image: torch.Tensor, output: torch.Tensor) -> None:
        model_name = "GPENBFR2048"
        ort_session = self._get_model_session(model_name)
        if not ort_session:
            return  # Silently skip if model failed to load

        io_binding = ort_session.io_binding()
        io_binding.bind_input(
            name="input",
            device_type=self.models_processor.device_type,
            device_id=self.models_processor.binding_device_id,
            element_type=np.float32,
            shape=(1, 3, 2048, 2048),
            buffer_ptr=image.data_ptr(),
        )
        io_binding.bind_output(
            name="output",
            device_type=self.models_processor.device_type,
            device_id=self.models_processor.binding_device_id,
            element_type=np.float32,
            shape=(1, 3, 2048, 2048),
            buffer_ptr=output.data_ptr(),
        )
        # Run the model with lazy build handling
        self._run_model_with_lazy_build_check(model_name, ort_session, io_binding)

    def run_codeformer(
        self,
        image: torch.Tensor,
        output: torch.Tensor,
        fidelity_weight_value: float = 0.9,
    ) -> None:
        model_name = "CodeFormer"
        ort_session = self._get_model_session(model_name)
        if not ort_session:
            return  # Silently skip if model failed to load

        io_binding = ort_session.io_binding()
        io_binding.bind_input(
            name="x",
            device_type=self.models_processor.device_type,
            device_id=self.models_processor.binding_device_id,
            element_type=np.float32,
            shape=(1, 3, 512, 512),
            buffer_ptr=image.data_ptr(),
        )
        w = np.array([fidelity_weight_value], dtype=np.double)
        io_binding.bind_cpu_input("w", w)
        io_binding.bind_output(
            name="y",
            device_type=self.models_processor.device_type,
            device_id=self.models_processor.binding_device_id,
            element_type=np.float32,
            shape=(1, 3, 512, 512),
            buffer_ptr=output.data_ptr(),
        )
        # Run the model with lazy build handling
        self._run_model_with_lazy_build_check(model_name, ort_session, io_binding)

    def run_VQFR_v2(
        self, image: torch.Tensor, output: torch.Tensor, fidelity_ratio_value: float
    ) -> None:
        model_name = "VQFRv2"
        ort_session = self._get_model_session(model_name)
        if not ort_session:
            return  # Silently skip if model failed to load

        if not (0.0 <= fidelity_ratio_value <= 1.0):
            raise ValueError(
                f"fidelity_ratio_value must be in [0,1], got {fidelity_ratio_value}"
            )
        fidelity_ratio = torch.tensor(
            fidelity_ratio_value,
            dtype=torch.float32,
            device=self.models_processor.device,
        )

        io_binding = ort_session.io_binding()
        io_binding.bind_input(
            name="x_lq",
            device_type=self.models_processor.device_type,
            device_id=self.models_processor.binding_device_id,
            element_type=np.float32,
            shape=tuple(image.shape),
            buffer_ptr=image.data_ptr(),
        )
        io_binding.bind_input(
            name="fidelity_ratio",
            device_type=self.models_processor.device_type,
            device_id=self.models_processor.binding_device_id,
            element_type=np.float32,
            shape=tuple(fidelity_ratio.shape),
            buffer_ptr=fidelity_ratio.data_ptr(),
        )
        io_binding.bind_output(
            "enc_feat",
            self.models_processor.device_type,
            self.models_processor.binding_device_id,
        )
        io_binding.bind_output(
            "quant_logit",
            self.models_processor.device_type,
            self.models_processor.binding_device_id,
        )
        io_binding.bind_output(
            "texture_dec",
            self.models_processor.device_type,
            self.models_processor.binding_device_id,
        )
        io_binding.bind_output(
            name="main_dec",
            device_type=self.models_processor.device_type,
            device_id=self.models_processor.binding_device_id,
            element_type=np.float32,
            shape=(1, 3, 512, 512),
            buffer_ptr=output.data_ptr(),
        )
        # Run the model with lazy build handling
        self._run_model_with_lazy_build_check(model_name, ort_session, io_binding)

    def run_RestoreFormerPlusPlus(
        self, image: torch.Tensor, output: torch.Tensor
    ) -> None:
        model_name = "RestoreFormerPlusPlus"
        ort_session = self._get_model_session(model_name)
        if not ort_session:
            return  # Silently skip if model failed to load

        io_binding = ort_session.io_binding()
        io_binding.bind_input(
            name="input",
            device_type=self.models_processor.device_type,
            device_id=self.models_processor.binding_device_id,
            element_type=np.float32,
            shape=tuple(image.shape),
            buffer_ptr=image.data_ptr(),
        )
        io_binding.bind_output(
            name="2359",
            device_type=self.models_processor.device_type,
            device_id=self.models_processor.binding_device_id,
            element_type=np.float32,
            shape=tuple(output.shape),
            buffer_ptr=output.data_ptr(),
        )
        io_binding.bind_output(
            "1228",
            self.models_processor.device_type,
            self.models_processor.binding_device_id,
        )
        io_binding.bind_output(
            "1238",
            self.models_processor.device_type,
            self.models_processor.binding_device_id,
        )
        io_binding.bind_output(
            "onnx::MatMul_1198",
            self.models_processor.device_type,
            self.models_processor.binding_device_id,
        )
        io_binding.bind_output(
            "onnx::Shape_1184",
            self.models_processor.device_type,
            self.models_processor.binding_device_id,
        )
        io_binding.bind_output(
            "onnx::ArgMin_1182",
            self.models_processor.device_type,
            self.models_processor.binding_device_id,
        )
        io_binding.bind_output(
            "input.1",
            self.models_processor.device_type,
            self.models_processor.binding_device_id,
        )
        io_binding.bind_output(
            "x",
            self.models_processor.device_type,
            self.models_processor.binding_device_id,
        )
        io_binding.bind_output(
            "x.3",
            self.models_processor.device_type,
            self.models_processor.binding_device_id,
        )
        io_binding.bind_output(
            "x.7",
            self.models_processor.device_type,
            self.models_processor.binding_device_id,
        )
        io_binding.bind_output(
            "x.11",
            self.models_processor.device_type,
            self.models_processor.binding_device_id,
        )
        io_binding.bind_output(
            "x.15",
            self.models_processor.device_type,
            self.models_processor.binding_device_id,
        )
        io_binding.bind_output(
            "input.252",
            self.models_processor.device_type,
            self.models_processor.binding_device_id,
        )
        io_binding.bind_output(
            "input.280",
            self.models_processor.device_type,
            self.models_processor.binding_device_id,
        )
        io_binding.bind_output(
            "input.288",
            self.models_processor.device_type,
            self.models_processor.binding_device_id,
        )
        # Run the model with lazy build handling
        self._run_model_with_lazy_build_check(model_name, ort_session, io_binding)
