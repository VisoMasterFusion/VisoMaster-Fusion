from typing import TYPE_CHECKING, Dict

import torch
import numpy as np
from torchvision.transforms import v2
from skimage import transform as trans

if TYPE_CHECKING:
    from app.processors.models_processor import ModelsProcessor


class FaceRestorers:
    def __init__(self, models_processor: "ModelsProcessor"):
        self.models_processor = models_processor
        self.resize_transforms = {
            512: v2.Resize((512, 512), antialias=True),
            256: v2.Resize((256, 256), antialias=False),
            1024: v2.Resize((1024, 1024), antialias=False),
            2048: v2.Resize((2048, 2048), antialias=False),
        }
        self._warned_models = set() # To track warnings

    def _get_model_session(self, model_name: str):
        """
        Helper to safely get a model session.
        Tries to load it if not available, and returns None if loading fails.
        """
        # Check if model is already loaded (even if it's None, meaning it failed)
        if model_name in self.models_processor.models:
            ort_session = self.models_processor.models[model_name]
        else:
            ort_session = None

        if not ort_session:
            # Model isn't loaded. Let's try to load it once.
            ort_session = self.models_processor.load_model(model_name)
            # load_model will set self.models_processor.models[model_name] to None if it fails
        
        if not ort_session:
            # Loading failed or it was already marked as failed (None).
            # Warn once and return None.
            if model_name not in self._warned_models:
                print(f"WARNING: Model '{model_name}' failed to load or is not available. This operation will be skipped. You could retry loading the model.")
                self._warned_models.add(model_name)
            return None
            
        return ort_session
        
    def apply_facerestorer(
        self,
        swapped_face_upscaled,
        restorer_det_type,
        restorer_type,
        restorer_blend,
        fidelity_weight,
        detect_score,
        target_kps=None,
    ):
        temp = swapped_face_upscaled
        t512 = self.resize_transforms[512]
        t256 = self.resize_transforms[256]
        t1024 = self.resize_transforms[1024]
        t2048 = self.resize_transforms[2048]

        # If using a separate detection mode
        if restorer_det_type == "Blend" or restorer_det_type == "Reference":
            if restorer_det_type == "Blend":
                # Set up Transformation
                dst = self.models_processor.arcface_dst * 4.0
                dst[:, 0] += 32.0

            elif restorer_det_type == "Reference":
                # FIX: Instead of re-detecting landmarks, use the target_kps passed to the function.
                if target_kps is None or len(target_kps) == 0:
                    print(
                        "[WARN] 'Reference' alignment selected, but no target landmarks (target_kps) were provided. Skipping restoration."
                    )
                    return swapped_face_upscaled
                dst = target_kps

            # Return non-enhanced face if keypoints are empty
            if not isinstance(dst, np.ndarray) or len(dst) == 0:
                return swapped_face_upscaled

            tform = trans.SimilarityTransform()
            try:
                tform.estimate(dst, self.models_processor.FFHQ_kps)
            except:
                return swapped_face_upscaled
            # Transform, scale, and normalize
            temp = v2.functional.affine(
                swapped_face_upscaled,
                tform.rotation * 57.2958,
                (tform.translation[0], tform.translation[1]),
                tform.scale,
                0,
                center=(0, 0),
            )
            temp = v2.functional.crop(temp, 0, 0, 512, 512)

        temp = torch.div(temp, 255)
        temp = v2.functional.normalize(
            temp, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=False
        )

        if restorer_type == "GPEN-256":
            temp = t256(temp)

        temp = torch.unsqueeze(temp, 0).contiguous()

        # Bindings
        outpred = torch.empty(
            (1, 3, 512, 512), dtype=torch.float32, device=self.models_processor.device
        ).contiguous()

        if restorer_type == "GFPGAN-v1.4":
            self.run_GFPGAN(temp, outpred)

        elif restorer_type == "GFPGAN-1024":
            outpred = torch.empty(
                (1, 3, 1024, 1024),
                dtype=torch.float32,
                device=self.models_processor.device,
            ).contiguous()
            self.run_GFPGAN1024(temp, outpred)

        elif restorer_type == "CodeFormer":
            self.run_codeformer(temp, outpred, fidelity_weight)

        elif restorer_type == "GPEN-256":
            outpred = torch.empty(
                (1, 3, 256, 256),
                dtype=torch.float32,
                device=self.models_processor.device,
            ).contiguous()
            self.run_GPEN_256(temp, outpred)

        elif restorer_type == "GPEN-512":
            self.run_GPEN_512(temp, outpred)

        elif restorer_type == "GPEN-1024":
            temp = t1024(temp)
            outpred = torch.empty(
                (1, 3, 1024, 1024),
                dtype=torch.float32,
                device=self.models_processor.device,
            ).contiguous()
            self.run_GPEN_1024(temp, outpred)

        elif restorer_type == "GPEN-2048":
            temp = t2048(temp)
            outpred = torch.empty(
                (1, 3, 2048, 2048),
                dtype=torch.float32,
                device=self.models_processor.device,
            ).contiguous()
            self.run_GPEN_2048(temp, outpred)

        elif restorer_type == "RestoreFormer++":
            self.run_RestoreFormerPlusPlus(temp, outpred)

        elif restorer_type == "VQFR-v2":
            self.run_VQFR_v2(temp, outpred, fidelity_weight)

        # Format back to cxHxW @ 255
        outpred = torch.squeeze(outpred)
        outpred = torch.clamp(outpred, -1, 1)
        outpred = torch.add(outpred, 1)
        outpred = torch.div(outpred, 2)
        outpred = torch.mul(outpred, 255)

        if (
            restorer_type == "GPEN-256"
            or restorer_type == "GPEN-1024"
            or restorer_type == "GPEN-2048"
            or restorer_type == "GFPGAN-1024"
        ):
            outpred = t512(outpred)

        # Invert Transform
        if restorer_det_type == "Blend" or restorer_det_type == "Reference":
            outpred = v2.functional.affine(
                outpred,
                tform.inverse.rotation * 57.2958,
                (tform.inverse.translation[0], tform.inverse.translation[1]),
                tform.inverse.scale,
                0,
                interpolation=v2.InterpolationMode.BILINEAR,
                center=(0, 0),
            )

        # Blend
        # alpha = float(restorer_blend)/100.0
        # outpred = torch.add(torch.mul(outpred, alpha), torch.mul(swapped_face_upscaled, 1-alpha))

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
        ort_session = self.models_processor.models[model_name]
        if not ort_session:
            error_msg = f"Error: VAE Encoder model '{model_name}' not loaded when run_vae_encoder was called. This model should be loaded by ModelsProcessor.ensure_denoiser_models_loaded()."
            print(error_msg)
            raise RuntimeError(
                error_msg
            )  # Or handle more gracefully depending on desired behavior

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
            device_type=self.models_processor.device,
            device_id=0,
            element_type=np.float32,
            shape=tuple(image_input_tensor.shape),
            buffer_ptr=image_input_tensor.data_ptr(),
        )
        io_binding.bind_output(
            name=output_name,
            device_type=self.models_processor.device,
            device_id=0,
            element_type=np.float32,
            shape=tuple(output_latent_tensor.shape),
            buffer_ptr=output_latent_tensor.data_ptr(),
        )

        if self.models_processor.device == "cuda":
            torch.cuda.synchronize()
        elif self.models_processor.device != "cpu":
            self.models_processor.syncvec.cpu()
        ort_session.run_with_iobinding(io_binding)

    def run_vae_decoder(
        self, latent_input_tensor: torch.Tensor, output_image_tensor: torch.Tensor
    ):
        """
        Runs the VAE decoder model.
        latent_input_tensor: Batch x 8 x LatentH x LatentW, float32
        output_image_tensor: Placeholder for Batch x 3 x H x W, float32, normalized to [-1, 1]
        """
        model_name = "RefLDMVAEDecoder"
        ort_session = self.models_processor.models[model_name]
        if not ort_session:
            error_msg = f"Error: VAE Decoder model '{model_name}' not loaded when run_vae_decoder was called. This model should be loaded by ModelsProcessor.ensure_denoiser_models_loaded()."
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
            device_type=self.models_processor.device,
            device_id=0,
            element_type=np.float32,
            shape=tuple(latent_input_tensor.shape),
            buffer_ptr=latent_input_tensor.data_ptr(),
        )
        io_binding.bind_output(
            name=output_name,
            device_type=self.models_processor.device,
            device_id=0,
            element_type=np.float32,
            shape=tuple(output_image_tensor.shape),
            buffer_ptr=output_image_tensor.data_ptr(),
        )

        if self.models_processor.device == "cuda":
            torch.cuda.synchronize()
        elif self.models_processor.device != "cpu":
            self.models_processor.syncvec.cpu()
        ort_session.run_with_iobinding(io_binding)

    def run_ref_ldm_unet(
        self,
        x_noisy_plus_lq_latent: torch.Tensor,
        timesteps_tensor: torch.Tensor,
        is_ref_flag_tensor: torch.Tensor,
        use_reference_exclusive_path_globally_tensor: torch.Tensor,
        kv_tensor_map: Dict[str, Dict[str, torch.Tensor]],
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
                f"Error: UNet model '{model_name}' not loaded when run_ref_ldm_unet was called."
            ]
            error_messages.append(
                "  This model should be loaded by ModelsProcessor.apply_denoiser_unet or a similar setup routine."
            )
            # ... (error reporting details omitted for brevity) ...
            print("\n".join(error_messages))
            return

        onnx_output_name = "unet_output"

        io_binding = ort_session.io_binding()
        bind_device_type = self.models_processor.device
        bind_device_id = 0

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
                        .to(device=bind_device_type, dtype=torch.float32)
                        .contiguous()
                    )

                if (
                    v_tensor_original is not None
                    and v_name_onnx in onnx_kv_input_names_to_shape
                ):
                    actual_kv_tensors_for_binding[v_name_onnx] = (
                        v_tensor_original.unsqueeze(0)
                        .to(device=bind_device_type, dtype=torch.float32)
                        .contiguous()
                    )

        for onnx_kv_name, expected_shape in onnx_kv_input_names_to_shape.items():
            tensor_to_bind = actual_kv_tensors_for_binding.get(onnx_kv_name)
            if tensor_to_bind is None:
                tensor_to_bind = torch.zeros(
                    expected_shape, dtype=torch.float32, device=bind_device_type
                ).contiguous()
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

        if self.models_processor.device == "cuda":
            torch.cuda.synchronize()
        elif self.models_processor.device != "cpu":
            self.models_processor.syncvec.cpu()
        ort_session.run_with_iobinding(io_binding)

    def run_GFPGAN(self, image, output):
        model_name = "GFPGANv1.4"
        ort_session = self._get_model_session(model_name)
        if not ort_session:
            return  # Silently skip if model failed to load

        io_binding = ort_session.io_binding()
        io_binding.bind_input(
            name="input",
            device_type=self.models_processor.device,
            device_id=0,
            element_type=np.float32,
            shape=(1, 3, 512, 512),
            buffer_ptr=image.data_ptr(),
        )
        io_binding.bind_output(
            name="output",
            device_type=self.models_processor.device,
            device_id=0,
            element_type=np.float32,
            shape=(1, 3, 512, 512),
            buffer_ptr=output.data_ptr(),
        )

        if self.models_processor.device == "cuda":
            torch.cuda.synchronize()
        elif self.models_processor.device != "cpu":
            self.models_processor.syncvec.cpu()
        ort_session.run_with_iobinding(io_binding)

    def run_GFPGAN1024(self, image, output):
        model_name = "GFPGAN1024"
        ort_session = self._get_model_session(model_name)
        if not ort_session:
            return  # Silently skip

        io_binding = ort_session.io_binding()
        io_binding.bind_input(
            name="input",
            device_type=self.models_processor.device,
            device_id=0,
            element_type=np.float32,
            shape=(1, 3, 512, 512),
            buffer_ptr=image.data_ptr(),
        )
        io_binding.bind_output(
            name="output",
            device_type=self.models_processor.device,
            device_id=0,
            element_type=np.float32,
            shape=(1, 3, 1024, 1024),
            buffer_ptr=output.data_ptr(),
        )

        if self.models_processor.device == "cuda":
            torch.cuda.synchronize()
        elif self.models_processor.device != "cpu":
            self.models_processor.syncvec.cpu()
        ort_session.run_with_iobinding(io_binding)

    def run_GPEN_256(self, image, output):
        model_name = "GPENBFR256"
        ort_session = self._get_model_session(model_name)
        if not ort_session:
            return  # Silently skip

        io_binding = ort_session.io_binding()
        io_binding.bind_input(
            name="input",
            device_type=self.models_processor.device,
            device_id=0,
            element_type=np.float32,
            shape=(1, 3, 256, 256),
            buffer_ptr=image.data_ptr(),
        )
        io_binding.bind_output(
            name="output",
            device_type=self.models_processor.device,
            device_id=0,
            element_type=np.float32,
            shape=(1, 3, 256, 256),
            buffer_ptr=output.data_ptr(),
        )

        if self.models_processor.device == "cuda":
            torch.cuda.synchronize()
        elif self.models_processor.device != "cpu":
            self.models_processor.syncvec.cpu()
        ort_session.run_with_iobinding(io_binding)

    def run_GPEN_512(self, image, output):
        model_name = "GPENBFR512"
        ort_session = self._get_model_session(model_name)
        if not ort_session:
            return  # Silently skip

        io_binding = ort_session.io_binding()
        io_binding.bind_input(
            name="input",
            device_type=self.models_processor.device,
            device_id=0,
            element_type=np.float32,
            shape=(1, 3, 512, 512),
            buffer_ptr=image.data_ptr(),
        )
        io_binding.bind_output(
            name="output",
            device_type=self.models_processor.device,
            device_id=0,
            element_type=np.float32,
            shape=(1, 3, 512, 512),
            buffer_ptr=output.data_ptr(),
        )

        if self.models_processor.device == "cuda":
            torch.cuda.synchronize()
        elif self.models_processor.device != "cpu":
            self.models_processor.syncvec.cpu()
        ort_session.run_with_iobinding(io_binding)

    def run_GPEN_1024(self, image, output):
        model_name = "GPENBFR1024"
        ort_session = self._get_model_session(model_name)
        if not ort_session:
            return  # Silently skip

        io_binding = ort_session.io_binding()
        io_binding.bind_input(
            name="input",
            device_type=self.models_processor.device,
            device_id=0,
            element_type=np.float32,
            shape=(1, 3, 1024, 1024),
            buffer_ptr=image.data_ptr(),
        )
        io_binding.bind_output(
            name="output",
            device_type=self.models_processor.device,
            device_id=0,
            element_type=np.float32,
            shape=(1, 3, 1024, 1024),
            buffer_ptr=output.data_ptr(),
        )

        if self.models_processor.device == "cuda":
            torch.cuda.synchronize()
        elif self.models_processor.device != "cpu":
            self.models_processor.syncvec.cpu()
        ort_session.run_with_iobinding(io_binding)

    def run_GPEN_2048(self, image, output):
        model_name = "GPENBFR2048"
        ort_session = self._get_model_session(model_name)
        if not ort_session:
            return  # Silently skip

        io_binding = ort_session.io_binding()
        io_binding.bind_input(
            name="input",
            device_type=self.models_processor.device,
            device_id=0,
            element_type=np.float32,
            shape=(1, 3, 2048, 2048),
            buffer_ptr=image.data_ptr(),
        )
        io_binding.bind_output(
            name="output",
            device_type=self.models_processor.device,
            device_id=0,
            element_type=np.float32,
            shape=(1, 3, 2048, 2048),
            buffer_ptr=output.data_ptr(),
        )

        if self.models_processor.device == "cuda":
            torch.cuda.synchronize()
        elif self.models_processor.device != "cpu":
            self.models_processor.syncvec.cpu()
        ort_session.run_with_iobinding(io_binding)

    def run_codeformer(self, image, output, fidelity_weight_value=0.9):
        model_name = "CodeFormer"
        ort_session = self._get_model_session(model_name)
        if not ort_session:
            return  # Silently skip

        io_binding = ort_session.io_binding()
        io_binding.bind_input(
            name="x",
            device_type=self.models_processor.device,
            device_id=0,
            element_type=np.float32,
            shape=(1, 3, 512, 512),
            buffer_ptr=image.data_ptr(),
        )
        w = np.array([fidelity_weight_value], dtype=np.double)
        io_binding.bind_cpu_input("w", w)
        io_binding.bind_output(
            name="y",
            device_type=self.models_processor.device,
            device_id=0,
            element_type=np.float32,
            shape=(1, 3, 512, 512),
            buffer_ptr=output.data_ptr(),
        )

        if self.models_processor.device == "cuda":
            torch.cuda.synchronize()
        elif self.models_processor.device != "cpu":
            self.models_processor.syncvec.cpu()
        ort_session.run_with_iobinding(io_binding)

    def run_VQFR_v2(self, image, output, fidelity_ratio_value):
        model_name = "VQFRv2"
        ort_session = self._get_model_session(model_name)
        if not ort_session:
            return  # Silently skip

        assert fidelity_ratio_value >= 0.0 and fidelity_ratio_value <= 1.0, (
            "fidelity_ratio must in range[0,1]"
        )
        fidelity_ratio = torch.tensor(fidelity_ratio_value).to(
            self.models_processor.device
        )

        io_binding = ort_session.io_binding()
        io_binding.bind_input(
            name="x_lq",
            device_type=self.models_processor.device,
            device_id=0,
            element_type=np.float32,
            shape=image.size(),
            buffer_ptr=image.data_ptr(),
        )
        io_binding.bind_input(
            name="fidelity_ratio",
            device_type=self.models_processor.device,
            device_id=0,
            element_type=np.float32,
            shape=fidelity_ratio.size(),
            buffer_ptr=fidelity_ratio.data_ptr(),
        )
        io_binding.bind_output("enc_feat", self.models_processor.device)
        io_binding.bind_output("quant_logit", self.models_processor.device)
        io_binding.bind_output("texture_dec", self.models_processor.device)
        io_binding.bind_output(
            name="main_dec",
            device_type=self.models_processor.device,
            device_id=0,
            element_type=np.float32,
            shape=(1, 3, 512, 512),
            buffer_ptr=output.data_ptr(),
        )

        if self.models_processor.device == "cuda":
            torch.cuda.synchronize()
        elif self.models_processor.device != "cpu":
            self.models_processor.syncvec.cpu()
        ort_session.run_with_iobinding(io_binding)

    def run_RestoreFormerPlusPlus(self, image, output):
        model_name = "RestoreFormerPlusPlus"
        ort_session = self._get_model_session(model_name)
        if not ort_session:
            return  # Silently skip

        io_binding = ort_session.io_binding()
        io_binding.bind_input(
            name="input",
            device_type=self.models_processor.device,
            device_id=0,
            element_type=np.float32,
            shape=image.size(),
            buffer_ptr=image.data_ptr(),
        )
        io_binding.bind_output(
            name="2359",
            device_type=self.models_processor.device,
            device_id=0,
            element_type=np.float32,
            shape=output.size(),
            buffer_ptr=output.data_ptr(),
        )
        io_binding.bind_output("1228", self.models_processor.device)
        io_binding.bind_output("1238", self.models_processor.device)
        io_binding.bind_output("onnx::MatMul_1198", self.models_processor.device)
        io_binding.bind_output("onnx::Shape_1184", self.models_processor.device)
        io_binding.bind_output("onnx::ArgMin_1182", self.models_processor.device)
        io_binding.bind_output("input.1", self.models_processor.device)
        io_binding.bind_output("x", self.models_processor.device)
        io_binding.bind_output("x.3", self.models_processor.device)
        io_binding.bind_output("x.7", self.models_processor.device)
        io_binding.bind_output("x.11", self.models_processor.device)
        io_binding.bind_output("x.15", self.models_processor.device)
        io_binding.bind_output("input.252", self.models_processor.device)
        io_binding.bind_output("input.280", self.models_processor.device)
        io_binding.bind_output("input.288", self.models_processor.device)

        if self.models_processor.device == "cuda":
            torch.cuda.synchronize()
        elif self.models_processor.device != "cpu":
            self.models_processor.syncvec.cpu()
        ort_session.run_with_iobinding(
            io_binding
        )
