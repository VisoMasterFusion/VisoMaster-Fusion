import math
from typing import TYPE_CHECKING

import torch
import numpy as np
from torchvision.transforms import v2
from app.processors.utils import faceutil

if TYPE_CHECKING:
    from app.processors.models_processor import ModelsProcessor


class FrameEnhancers:
    """
    Manages frame enhancement (upscaling, colorization) models and processes.

    This class handles model loading, tiling for large images, and execution
    of various ONNX models (RealESRGAN, BSRGan, Deoldify, DDColor, etc.).
    """

    def __init__(self, models_processor: "ModelsProcessor"):
        """
        Initializes the FrameEnhancers class.

        Args:
            models_processor (ModelsProcessor): The central processor that manages
                                               model loading, unloading, and execution.
        """
        self.models_processor = models_processor
        self.current_enhancer_model = None  # Tracks the currently active enhancer model
        self.model_map = {
            # Maps user-facing names to internal model keys (used in models_processor)
            "RealEsrgan-x2-Plus": "RealEsrganx2Plus",
            "RealEsrgan-x4-Plus": "RealEsrganx4Plus",
            "BSRGan-x2": "BSRGANx2",
            "BSRGan-x4": "BSRGANx4",
            "UltraSharp-x4": "UltraSharpx4",
            "UltraMix-x4": "UltraMixx4",
            "RealEsr-General-x4v3": "RealEsrx4v3",
            "DeOldify-Artistic": "DeoldifyArt",
            "DeOldify-Stable": "DeoldifyStable",
            "DeOldify-Video": "DeoldifyVideo",
            "DDColor-Artistic": "DDColorArt",
            "DDColor": "DDcolor",
        }

    def unload_models(self):
        """
        Unloads the currently active enhancer model to free up VRAM.
        This is thread-safe, using the model_lock from models_processor.
        """
        with self.models_processor.model_lock:
            if self.current_enhancer_model:
                self.models_processor.unload_model(self.current_enhancer_model)
                self.current_enhancer_model = None

    def _run_model_with_lazy_build_check(
        self, model_name: str, ort_session, io_binding
    ):
        """
        Runs the ONNX session with IOBinding, handling TensorRT lazy build dialogs.

        This centralizes the try/finally logic for showing/hiding the build progress
        dialog and includes the critical synchronization steps (Pre and Post inference)
        for CUDA or other devices.

        Args:
            model_name (str): The name of the model being run.
            ort_session: The ONNX Runtime session instance.
            io_binding: The pre-configured IOBinding object.
        """
        # Check if TensorRT is performing its one-time build for this model
        is_lazy_build = self.models_processor.check_and_clear_pending_build(model_name)
        if is_lazy_build:
            # Show a "please wait" dialog to the user
            self.models_processor.show_build_dialog.emit(
                "Finalizing TensorRT Build",
                f"Performing first-run inference for:\n{model_name}\n\nThis may take several minutes.",
            )

        try:
            ort_session.run_with_iobinding(io_binding)
        finally:
            # Always hide the dialog, even if the run fails
            if is_lazy_build:
                self.models_processor.hide_build_dialog.emit()

    def run_enhance_frame_tile_process(
        self, img, enhancer_type, tile_size=256, scale=1
    ):
        """
        Applies a selected enhancement model to an image using a tiling process.
        This is necessary for high-resolution images that don't fit into
        VRAM in one go.

        Args:
            img (torch.Tensor): The input image tensor (B, C, H, W).
            enhancer_type (str): The name of the enhancer to use (e.g., "RealEsrgan-x4-Plus").
            tile_size (int): The size of the square tiles to process.
            scale (int): The upscaling factor of the model (e.g., 2 for x2, 4 for x4).

        Returns:
            torch.Tensor: The enhanced (upscaled or colorized) image tensor.
        """
        # Model loading/unloading is now handled by control_actions.py (UI events).
        # We remove the redundant per-frame logic to prevent conflicts.
        # The 'current_enhancer_model' state is still set by control_actions.

        _, _, height, width = img.shape

        # --- 1. Calculate Tiling and Padding ---

        # Calculate the number of tiles needed
        tiles_x = math.ceil(width / tile_size)
        tiles_y = math.ceil(height / tile_size)

        # Calculate padding required to make the image dimensions divisible by tile_size
        pad_right = (tile_size - (width % tile_size)) % tile_size
        pad_bottom = (tile_size - (height % tile_size)) % tile_size

        # Apply padding to the image if necessary
        if pad_right != 0 or pad_bottom != 0:
            # Use 'constant' padding (black pixels)
            img = torch.nn.functional.pad(
                img, (0, pad_right, 0, pad_bottom), "constant", 0
            )

        # --- 2. Prepare Output Tensor and Select Model ---

        # Create an empty output tensor with the new scaled dimensions
        b, c, h, w = img.shape  # Get new padded dimensions
        output = torch.empty(
            (b, c, h * scale, w * scale),
            dtype=torch.float32,
            device=self.models_processor.device,
        ).contiguous()

        # Select the upscaling function based on the enhancer_type
        upscaler_functions = {
            "RealEsrgan-x2-Plus": self.run_realesrganx2,
            "RealEsrgan-x4-Plus": self.run_realesrganx4,
            "BSRGan-x2": self.run_bsrganx2,
            "BSRGan-x4": self.run_bsrganx4,
            "UltraSharp-x4": self.run_ultrasharpx4,
            "UltraMix-x4": self.run_ultramixx4,
            "RealEsr-General-x4v3": self.run_realesrx4v3,
        }

        fn_upscaler = upscaler_functions.get(enhancer_type)

        if not fn_upscaler:  # If the enhancer type is not a valid upscaler
            # Crop the original image back if padding was added, and return it
            if pad_right != 0 or pad_bottom != 0:
                img = v2.functional.crop(img, 0, 0, height, width)
            return img

        # --- 3. Process Tiles ---
        # Pre-allocate a single reusable output tile (all tiles have the same size
        # because the image was padded to an exact multiple of tile_size above).
        output_tile = torch.empty(
            (b, c, tile_size * scale, tile_size * scale),
            dtype=torch.float32,
            device=self.models_processor.device,
        ).contiguous()

        with torch.no_grad():  # Disable gradient calculation for inference
            # Process tiles
            for j in range(tiles_y):
                for i in range(tiles_x):
                    x_start, y_start = i * tile_size, j * tile_size
                    x_end, y_end = x_start + tile_size, y_start + tile_size

                    # Extract the input tile
                    input_tile = img[:, :, y_start:y_end, x_start:x_end].contiguous()

                    # Run the selected upscaler function into the pre-allocated tile
                    fn_upscaler(input_tile, output_tile)

                    # --- 4. Reassemble Output ---
                    # Calculate coordinates to place the output tile in the main output tensor
                    output_y_start, output_x_start = y_start * scale, x_start * scale
                    output_y_end, output_x_end = (
                        output_y_start + output_tile.shape[2],
                        output_x_start + output_tile.shape[3],
                    )
                    # Place the processed tile into the output tensor
                    output[
                        :, :, output_y_start:output_y_end, output_x_start:output_x_end
                    ] = output_tile

            # Crop the final output to remove the padding that was added
            if pad_right != 0 or pad_bottom != 0:
                output = v2.functional.crop(output, 0, 0, height * scale, width * scale)

        return output

    def _run_enhancer_model(
        self, model_name: str, image: torch.Tensor, output: torch.Tensor
    ):
        """
        Private helper to run any specified enhancer model.

        This function centralizes the logic for:
        1. Translating user-facing model names to internal keys via model_map.
        2. Lazy-loading the model.
        3. Handling model loading errors with a robust fallback.
        4. Setting up IOBinding for inputs and outputs.
        5. Calling the synchronized execution function.

        Args:
            model_name (str): Either a user-facing name (e.g., "RealEsrgan-x4-Plus") or
                              an internal key (e.g., "RealEsrganx4Plus"). model_map is
                              consulted first; if no mapping is found the name is used as-is.
            image (torch.Tensor): The input image (or tile) tensor.
            output (torch.Tensor): The pre-allocated output tensor to be filled.
        """
        # Translate user-facing name to internal model key if applicable
        model_name = self.model_map.get(model_name, model_name)

        # Lazy-load the model if it's not already in memory
        if not self.models_processor.models[model_name]:
            self.models_processor.models[model_name] = self.models_processor.load_model(
                model_name
            )

        ort_session = self.models_processor.models[model_name]

        if not ort_session:
            # This fix ensures the output tensor is correctly populated instead
            # of containing uninitialized data (garbage).
            print(f"[WARN] Model {model_name} not loaded, skipping enhancer.")

            if image.shape == output.shape:
                # For 1:1 models (like colorizers), just copy the input
                output.copy_(image)
            else:
                # For upscalers, use bilinear interpolation as a fallback
                resized_image = torch.nn.functional.interpolate(
                    image, size=output.shape[-2:], mode="bilinear", align_corners=False
                )
                output.copy_(resized_image)
            return

        # Bind inputs and outputs directly to GPU memory pointers
        io_binding = ort_session.io_binding()

        # Bind input tensor
        io_binding.bind_input(
            name="input",
            device_type=self.models_processor.device,
            device_id=0,
            element_type=np.float32,
            shape=image.size(),
            buffer_ptr=image.data_ptr(),
        )
        # Bind output tensor
        io_binding.bind_output(
            name="output",
            device_type=self.models_processor.device,
            device_id=0,
            element_type=np.float32,
            shape=output.size(),
            buffer_ptr=output.data_ptr(),
        )

        # Run the model with lazy build handling and synchronization
        self._run_model_with_lazy_build_check(model_name, ort_session, io_binding)

    def run_realesrganx2(self, image, output):
        """
        Runs the RealEsrganx2Plus model on a given image tensor.
        This function is typically called per-tile by run_enhance_frame_tile_process.

        Args:
            image (torch.Tensor): The input image (or tile) tensor.
            output (torch.Tensor): The pre-allocated output tensor to be filled.
        """
        self._run_enhancer_model("RealEsrganx2Plus", image, output)

    def run_realesrganx4(self, image, output):
        """
        Runs the RealEsrganx4Plus model on a given image tensor.
        This function is typically called per-tile by run_enhance_frame_tile_process.

        Args:
            image (torch.Tensor): The input image (or tile) tensor.
            output (torch.Tensor): The pre-allocated output tensor to be filled.
        """
        self._run_enhancer_model("RealEsrganx4Plus", image, output)

    def run_realesrx4v3(self, image, output):
        """
        Runs the RealEsrx4v3 model on a given image tensor.
        This function is typically called per-tile by run_enhance_frame_tile_process.

        Args:
            image (torch.Tensor): The input image (or tile) tensor.
            output (torch.Tensor): The pre-allocated output tensor to be filled.
        """
        self._run_enhancer_model("RealEsrx4v3", image, output)

    def run_bsrganx2(self, image, output):
        """
        Runs the BSRGANx2 model on a given image tensor.
        This function is typically called per-tile by run_enhance_frame_tile_process.

        Args:
            image (torch.Tensor): The input image (or tile) tensor.
            output (torch.Tensor): The pre-allocated output tensor to be filled.
        """
        self._run_enhancer_model("BSRGANx2", image, output)

    def run_bsrganx4(self, image, output):
        """
        Runs the BSRGANx4 model on a given image tensor.
        This function is typically called per-tile by run_enhance_frame_tile_process.

        Args:
            image (torch.Tensor): The input image (or tile) tensor.
            output (torch.Tensor): The pre-allocated output tensor to be filled.
        """
        self._run_enhancer_model("BSRGANx4", image, output)

    def run_ultrasharpx4(self, image, output):
        """
        Runs the UltraSharpx4 model on a given image tensor.
        This function is typically called per-tile by run_enhance_frame_tile_process.

        Args:
            image (torch.Tensor): The input image (or tile) tensor.
            output (torch.Tensor): The pre-allocated output tensor to be filled.
        """
        self._run_enhancer_model("UltraSharpx4", image, output)

    def run_ultramixx4(self, image, output):
        """
        Runs the UltraMixx4 model on a given image tensor.
        This function is typically called per-tile by run_enhance_frame_tile_process.

        Args:
            image (torch.Tensor): The input image (or tile) tensor.
            output (torch.Tensor): The pre-allocated output tensor to be filled.
        """
        self._run_enhancer_model("UltraMixx4", image, output)

    def run_deoldify_artistic(self, image, output):
        """
        Runs the DeoldifyArt (artistic colorization) model on a given image tensor.

        Args:
            image (torch.Tensor): The input image (or tile) tensor.
            output (torch.Tensor): The pre-allocated output tensor to be filled.
        """
        self._run_enhancer_model("DeoldifyArt", image, output)

    def run_deoldify_stable(self, image, output):
        """
        Runs the DeoldifyStable (stable colorization) model on a given image tensor.

        Args:
            image (torch.Tensor): The input image (or tile) tensor.
            output (torch.Tensor): The pre-allocated output tensor to be filled.
        """
        self._run_enhancer_model("DeoldifyStable", image, output)

    def run_deoldify_video(self, image, output):
        """
        Runs the DeoldifyVideo (video colorization) model on a given image tensor.

        Args:
            image (torch.Tensor): The input image (or tile) tensor.
            output (torch.Tensor): The pre-allocated output tensor to be filled.
        """
        self._run_enhancer_model("DeoldifyVideo", image, output)

    def run_ddcolor_artistic(self, image, output):
        """
        Runs the DDColorArt (artistic colorization) model on a given image tensor.

        Args:
            image (torch.Tensor): The input image (or tile) tensor.
            output (torch.Tensor): The pre-allocated output tensor to be filled.
        """
        self._run_enhancer_model("DDColorArt", image, output)

    def run_ddcolor(self, image, output):
        """
        Runs the DDcolor (general colorization) model on a given image tensor.

        Args:
            image (torch.Tensor): The input image (or tile) tensor.
            output (torch.Tensor): The pre-allocated output tensor to be filled.
        """
        self._run_enhancer_model("DDcolor", image, output)

    def enhance_core(self, img, control):
        enhancer_type = control["FrameEnhancerTypeSelection"]

        match enhancer_type:
            case (
                "RealEsrgan-x2-Plus"
                | "RealEsrgan-x4-Plus"
                | "BSRGan-x2"
                | "BSRGan-x4"
                | "UltraSharp-x4"
                | "UltraMix-x4"
                | "RealEsr-General-x4v3"
            ):
                tile_size = 512

                if (
                    enhancer_type == "RealEsrgan-x2-Plus"
                    or enhancer_type == "BSRGan-x2"
                ):
                    scale = 2
                else:
                    scale = 4

                image = img.type(torch.float32)
                if torch.max(image) > 255:  # 16-bit image
                    max_range = 65535
                else:
                    max_range = 255

                image = torch.div(image, max_range)
                image = torch.unsqueeze(image, 0).contiguous()

                image = self.models_processor.run_enhance_frame_tile_process(
                    image, enhancer_type, tile_size=tile_size, scale=scale
                )

                image = torch.squeeze(image)
                image = torch.clamp(image, 0, 1)
                image = torch.mul(image, max_range)

                # Blend
                alpha = float(control["FrameEnhancerBlendSlider"]) / 100.0

                t_scale = v2.Resize(
                    (img.shape[1] * scale, img.shape[2] * scale),
                    interpolation=v2.InterpolationMode.BILINEAR,
                    antialias=False,
                )
                img = t_scale(img)
                img = torch.add(torch.mul(image, alpha), torch.mul(img, 1 - alpha))
                if max_range == 255:
                    img = img.type(torch.uint8)
                else:
                    img = img.type(torch.uint16)

            case "DeOldify-Artistic" | "DeOldify-Stable" | "DeOldify-Video":
                render_factor = 384  # 12 * 32 | highest quality = 20 * 32 == 640

                _, h, w = img.shape
                t_resize_i = v2.Resize(
                    (render_factor, render_factor),
                    interpolation=v2.InterpolationMode.BILINEAR,
                    antialias=False,
                )
                image = t_resize_i(img)

                image = image.type(torch.float32)
                image = torch.unsqueeze(image, 0).contiguous()

                output = torch.empty(
                    (image.shape),
                    dtype=torch.float32,
                    device=self.models_processor.device,
                ).contiguous()

                match enhancer_type:
                    case "DeOldify-Artistic":
                        self.models_processor.run_deoldify_artistic(image, output)
                    case "DeOldify-Stable":
                        self.models_processor.run_deoldify_stable(image, output)
                    case "DeOldify-Video":
                        self.models_processor.run_deoldify_video(image, output)

                output = torch.squeeze(output)
                t_resize_o = v2.Resize(
                    (h, w), interpolation=v2.InterpolationMode.BILINEAR, antialias=False
                )
                output = t_resize_o(output)

                output_normalized = output / 255.0 if output.max() > 1.0 else output
                output_normalized = faceutil.rgb_to_yuv(output_normalized, True)
                # do a black and white transform first to get better luminance values
                img_normalized = img / 255.0 if img.max() > 1.0 else img
                hires = faceutil.rgb_to_yuv(img_normalized, True)

                hires[1:3, :, :] = output_normalized[1:3, :, :]
                hires = faceutil.yuv_to_rgb(hires, True)

                # Blend
                alpha = float(control["FrameEnhancerBlendSlider"]) / 100.0
                img = torch.add(
                    torch.mul(hires, alpha), torch.mul(img_normalized, 1 - alpha)
                )

                img = img.type(torch.uint8)

            case "DDColor-Artistic" | "DDColor":
                render_factor = 384  # 12 * 32 | highest quality = 20 * 32 == 640

                # Converti RGB a LAB
                #'''
                # orig_l = img.permute(1, 2, 0).cpu().numpy()
                # orig_l = cv2.cvtColor(orig_l, cv2.COLOR_RGB2Lab)
                # orig_l = torch.from_numpy(orig_l).to(self.models_processor.device)
                # orig_l = orig_l.permute(2, 0, 1)
                #'''
                img_normalized = img / 255.0 if img.max() > 1.0 else img
                orig_l = faceutil.rgb_to_lab(img_normalized, True)

                orig_l = orig_l[0:1, :, :]  # (1, h, w)

                # Resize per il modello
                t_resize_i = v2.Resize(
                    (render_factor, render_factor),
                    interpolation=v2.InterpolationMode.BILINEAR,
                    antialias=False,
                )
                image = t_resize_i(img)

                # Converti RGB in LAB
                #'''
                # img_l = image.permute(1, 2, 0).cpu().numpy()
                # img_l = cv2.cvtColor(img_l, cv2.COLOR_RGB2Lab)
                # img_l = torch.from_numpy(img_l).to(self.models_processor.device)
                # img_l = img_l.permute(2, 0, 1)
                #'''
                image_normalized = image / 255.0 if image.max() > 1.0 else image
                img_l = faceutil.rgb_to_lab(image_normalized, True)

                img_l = img_l[0:1, :, :]  # (1, render_factor, render_factor)
                img_gray_lab = torch.cat(
                    (img_l, torch.zeros_like(img_l), torch.zeros_like(img_l)), dim=0
                )  # (3, render_factor, render_factor)

                # Converti LAB in RGB
                #'''
                # img_gray_lab = img_gray_lab.permute(1, 2, 0).cpu().numpy()
                # img_gray_rgb = cv2.cvtColor(img_gray_lab, cv2.COLOR_LAB2RGB)
                # img_gray_rgb = torch.from_numpy(img_gray_rgb).to(self.models_processor.device)
                # img_gray_rgb = img_gray_rgb.permute(2, 0, 1)
                #'''
                img_gray_rgb = faceutil.lab_to_rgb(img_gray_lab)

                tensor_gray_rgb = torch.unsqueeze(
                    img_gray_rgb.type(torch.float32), 0
                ).contiguous()

                # Prepara il tensore per il modello
                output_ab = torch.empty(
                    (1, 2, render_factor, render_factor),
                    dtype=torch.float32,
                    device=self.models_processor.device,
                )

                # Esegui il modello
                match enhancer_type:
                    case "DDColor-Artistic":
                        self.models_processor.run_ddcolor_artistic(
                            tensor_gray_rgb, output_ab
                        )
                    case "DDColor":
                        self.models_processor.run_ddcolor(tensor_gray_rgb, output_ab)

                output_ab = output_ab.squeeze(0)  # (2, render_factor, render_factor)

                t_resize_o = v2.Resize(
                    (img.size(1), img.size(2)),
                    interpolation=v2.InterpolationMode.BILINEAR,
                    antialias=False,
                )
                output_lab_resize = t_resize_o(output_ab)

                # Combina il canale L originale con il risultato del modello
                output_lab = torch.cat(
                    (orig_l, output_lab_resize), dim=0
                )  # (3, original_H, original_W)

                # Convert LAB to RGB
                #'''
                # output_rgb = output_lab.permute(1, 2, 0).cpu().numpy()
                # output_rgb = cv2.cvtColor(output_rgb, cv2.COLOR_Lab2RGB)
                # output_rgb = torch.from_numpy(output_rgb).to(self.models_processor.device)
                # output_rgb = output_rgb.permute(2, 0, 1)
                #'''
                output_rgb = faceutil.lab_to_rgb(
                    output_lab, True
                )  # (3, original_H, original_W)

                # Miscela le immagini
                alpha = float(control["FrameEnhancerBlendSlider"]) / 100.0
                blended_img = torch.add(
                    torch.mul(output_rgb, alpha), torch.mul(img, 1 - alpha)
                )

                # Converti in uint8
                img = blended_img.type(torch.uint8)

        return img
