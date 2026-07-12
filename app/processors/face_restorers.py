from typing import TYPE_CHECKING, Dict, Optional

import torch
import numpy as np
from torchvision.transforms import v2
from skimage import transform as trans
import kornia.geometry.transform as kgm

if TYPE_CHECKING:
    from app.processors.models_processor import ModelsProcessor


class FaceRestorers:
    def __init__(self, models_processor: "ModelsProcessor"):
        self.models_processor = models_processor
        self.active_model_slot1: Optional[str] = None
        self.active_model_slot2: Optional[str] = None
        self._warned_models: set[str] = set()  # To track warnings
        self.model_map = {
            "GFPGAN-v1.4": "GFPGANv1.4",
            "GFPGAN-1024": "GFPGAN1024",
            "CodeFormer": "CodeFormer",
            "GPEN-256": "GPENBFR256",
            "GPEN-512": "GPENBFR512",
            "GPEN-1024": "GPENBFR1024",
            "GPEN-2048": "GPENBFR2048",
            "RestoreFormer++": "RestoreFormerPlusPlus",
            "VQFR-v2": "VQFRv2",
        }

    def unload_models(self):
        """Unloads the restorer models held in both slots and resets state."""
        if self.active_model_slot1:
            self.models_processor.unload_model(self.active_model_slot1)
            self.active_model_slot1 = None
        if self.active_model_slot2:
            self.models_processor.unload_model(self.active_model_slot2)
            self.active_model_slot2 = None

    def _get_model_session(self, model_name: str):
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
        self, model_name: str, ort_session, io_binding
    ):
        """
        Runs the ONNX session with IOBinding, handling TensorRT lazy build dialogs.
        This centralizes the try/finally logic for showing/hiding the build progress dialog
        and includes the critical synchronization step for CUDA or other devices.

        Args:
            model_name (str): The name of the model being run.
            ort_session: The ONNX Runtime session instance.
            io_binding: The pre-configured IOBinding object.
        """
        is_lazy_build = self.models_processor.check_and_clear_pending_build(model_name)
        if is_lazy_build:
            self.models_processor.show_build_dialog.emit(
                "Finalizing TensorRT Build",
                f"Performing first-run inference for:\n{model_name}\n\nThis may take several minutes.",
            )

        try:
            # ⚠️ This is a critical synchronization point.
            # PRE-INFERENCE SYNC
            if self.models_processor.device_type == "cuda":
                torch.cuda.current_stream().synchronize()
            elif self.models_processor.device_type != "cpu":
                # This handles synchronization for other execution providers (e.g., DirectML)
                # by synchronizing with a placeholder vector.
                self.models_processor.syncvec.cpu()

            ort_session.run_with_iobinding(io_binding)

        finally:
            if is_lazy_build:
                self.models_processor.hide_build_dialog.emit()

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
    ) -> torch.Tensor:
        model_name_to_load = self.model_map.get(restorer_type)
        if not model_name_to_load:
            return swapped_face_upscaled

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

            # OPTIMIZED: Direct GPU Affine Warp with Kornia, skipping torchvision crop/affine
            M_tensor = (
                torch.from_numpy(tform.params[0:2])
                .float()
                .unsqueeze(0)
                .to(swapped_face_upscaled.device)
            )
            img_b = (
                swapped_face_upscaled.unsqueeze(0)
                if swapped_face_upscaled.dim() == 3
                else swapped_face_upscaled
            )

            # Kornia allocates a new tensor here, so we own this memory space.
            temp = kgm.warp_affine(
                img_b.float(),
                M_tensor,
                dsize=(512, 512),
                mode="bilinear",
                align_corners=True,
            ).squeeze(0)
            # Safe to perform math operations since 'temp' is a brand new tensor
            temp = temp.float() / 255.0

        else:
            # If we did not warp the image, we MUST clone the original tensor
            # before applying division. Using .div_(255.0) on the original reference corrupts
            # memory for other threads (Race Condition).
            temp = swapped_face_upscaled.clone().float() / 255.0

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

        # Now safe to use inplace normalization since scaling math is clean
        temp = v2.functional.normalize(
            temp, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True
        )

        temp = torch.unsqueeze(temp, 0).contiguous()

        # Bindings
        outpred = None

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

        if outpred is None:
            return swapped_face_upscaled

        # OPTIMIZED: Fused in-place math operations to save VRAM allocations.
        # Math: ((x clamped [-1, 1]) + 1.0) * 127.5 is equivalent to /2 * 255.
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
            # OPTIMIZED: Direct Inverse GPU Affine Warp with Kornia
            M_inv_tensor = (
                torch.from_numpy(tform.inverse.params[0:2])
                .float()
                .unsqueeze(0)
                .to(outpred.device)
            )
            out_b = outpred.unsqueeze(0) if outpred.dim() == 3 else outpred
            dsize = (swapped_face_upscaled.shape[1], swapped_face_upscaled.shape[2])

            outpred = kgm.warp_affine(
                out_b,
                M_inv_tensor,
                dsize=(dsize[0], dsize[1]),
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            ).squeeze(0)

        # Blend (Disabled by default as in original code)
        # alpha = float(restorer_blend)/100.0
        # outpred = torch.add(torch.mul(outpred, alpha), torch.mul(swapped_face_upscaled, 1-alpha))

        # --- EXPLICIT CLEANUP ---
        # Explicitly delete local intermediate tensors to free VRAM immediately
        # before returning the final image. This keeps the VRAM peak perfectly flat.
        try:
            del temp
            if restorer_det_type in ["Blend", "Reference"]:
                del M_tensor
                del img_b
                del M_inv_tensor
                del out_b
        except Exception:
            pass

        return outpred

    def run_vae_encoder(
        self, image_input_tensor: torch.Tensor, output_latent_tensor: torch.Tensor
    ):
        """
        Runs the VAE encoder model.
        image_input_tensor: Batch x 3 x Height x Width, float32, normalized to [-1, 1]
        output_latent_tensor: Placeholder for Batch x 8 x LatentH x LatentW, float32
        """
        model_name = "RefLDMVAEEncoder"
        # FR-BUG-04: use .get() to avoid KeyError when model is not yet loaded
        ort_session = self.models_processor.models.get(model_name)
        if ort_session is None:
            # Lazy reload in case clear_gpu_memory() cleared the session after a provider switch.
            self.models_processor.face_denoiser.ensure_denoiser_models_loaded()
            ort_session = self.models_processor.models.get(model_name)
        if ort_session is None:
            error_msg = f"[ERROR] VAE Encoder model '{model_name}' not loaded when run_vae_encoder was called. This model should be loaded by ModelsProcessor.face_denoiser.ensure_denoiser_models_loaded()."
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

    def run_vae_decoder(
        self, latent_input_tensor: torch.Tensor, output_image_tensor: torch.Tensor
    ):
        """
        Runs the VAE decoder model.
        latent_input_tensor: Batch x 8 x LatentH x LatentW, float32
        output_image_tensor: Placeholder for Batch x 3 x H x W, float32, normalized to [-1, 1]
        """
        model_name = "RefLDMVAEDecoder"
        # FR-BUG-04: use .get() to avoid KeyError when model is not yet loaded
        ort_session = self.models_processor.models.get(model_name)
        if ort_session is None:
            # Lazy reload in case clear_gpu_memory() cleared the session after a provider switch.
            self.models_processor.face_denoiser.ensure_denoiser_models_loaded()
            ort_session = self.models_processor.models.get(model_name)
        if ort_session is None:
            error_msg = f"[ERROR] VAE Decoder model '{model_name}' not loaded when run_vae_decoder was called. This model should be loaded by ModelsProcessor.face_denoiser.ensure_denoiser_models_loaded()."
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

    def run_ref_ldm_unet(
        self,
        x_noisy_plus_lq_latent: torch.Tensor,
        timesteps_tensor: torch.Tensor,
        is_ref_flag_tensor: torch.Tensor,
        use_reference_exclusive_path_globally_tensor: torch.Tensor,
        kv_tensor_map: Optional[Dict[str, Dict[str, torch.Tensor]]],
        output_unet_tensor: torch.Tensor,
    ):
        """
        Runs the UNet denoiser model with external K/V inputs.
        """
        model_name = self.models_processor.main_window.fixed_unet_model_name
        ort_session = self.models_processor.models.get(model_name)

        if not ort_session:
            # Enhanced error reporting
            error_messages = [
                f"[ERROR] UNet model '{model_name}' not loaded when run_ref_ldm_unet was called.",
                "  This model should be loaded by ModelsProcessor.face_denoiser.apply_denoiser_unet or a similar setup routine.",
            ]
            print("\n".join(error_messages))
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
        onnx_kv_input_names_to_shape: Dict[str, tuple] = {
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
        keep_alive_tensors: list = []
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

    def run_GFPGAN(self, image, output):
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

    def run_GFPGAN1024(self, image, output):
        model_name = "GFPGAN1024"

        ort_session = self._get_model_session(model_name)
        if not ort_session:
            return  # Silently skip

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

    def run_GPEN_256(self, image, output):
        model_name = "GPENBFR256"

        ort_session = self._get_model_session(model_name)
        if not ort_session:
            return  # Silently skip

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

    def run_GPEN_512(self, image, output):
        model_name = "GPENBFR512"

        ort_session = self._get_model_session(model_name)
        if not ort_session:
            return  # Silently skip

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

    def run_GPEN_1024(self, image, output):
        model_name = "GPENBFR1024"

        ort_session = self._get_model_session(model_name)
        if not ort_session:
            return  # Silently skip

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

    def run_GPEN_2048(self, image, output):
        model_name = "GPENBFR2048"

        ort_session = self._get_model_session(model_name)
        if not ort_session:
            return  # Silently skip

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

    def run_codeformer(self, image, output, fidelity_weight_value=0.9):
        model_name = "CodeFormer"
        ort_session = self._get_model_session(model_name)
        if not ort_session:
            return  # Silently skip

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

    def run_VQFR_v2(self, image, output, fidelity_ratio_value):
        model_name = "VQFRv2"
        ort_session = self._get_model_session(model_name)
        if not ort_session:
            return  # Silently skip

        # FR-ROBUST-05: replace assert with an explicit ValueError so it is never silenced by -O flag
        if not (0.0 <= fidelity_ratio_value <= 1.0):
            raise ValueError(
                f"fidelity_ratio_value must be in [0,1], got {fidelity_ratio_value}"
            )
        fidelity_ratio = torch.tensor(fidelity_ratio_value, dtype=torch.float32).to(
            self.models_processor.device
        )

        io_binding = ort_session.io_binding()
        io_binding.bind_input(
            name="x_lq",
            device_type=self.models_processor.device_type,
            device_id=self.models_processor.binding_device_id,
            element_type=np.float32,
            shape=image.size(),
            buffer_ptr=image.data_ptr(),
        )
        io_binding.bind_input(
            name="fidelity_ratio",
            device_type=self.models_processor.device_type,
            device_id=self.models_processor.binding_device_id,
            element_type=np.float32,
            shape=fidelity_ratio.size(),
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

    def run_RestoreFormerPlusPlus(self, image, output):
        model_name = "RestoreFormerPlusPlus"
        ort_session = self._get_model_session(model_name)
        if not ort_session:
            return  # Silently skip

        io_binding = ort_session.io_binding()
        io_binding.bind_input(
            name="input",
            device_type=self.models_processor.device_type,
            device_id=self.models_processor.binding_device_id,
            element_type=np.float32,
            shape=image.size(),
            buffer_ptr=image.data_ptr(),
        )
        io_binding.bind_output(
            name="2359",
            device_type=self.models_processor.device_type,
            device_id=self.models_processor.binding_device_id,
            element_type=np.float32,
            shape=output.size(),
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
