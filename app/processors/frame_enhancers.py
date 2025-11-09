import math
from typing import TYPE_CHECKING

import torch
import numpy as np
from torchvision.transforms import v2

if TYPE_CHECKING:
    from app.processors.models_processor import ModelsProcessor


class FrameEnhancers:
    def __init__(self, models_processor: "ModelsProcessor"):
        self.models_processor = models_processor
        self.current_enhancer_model = None
        self.model_map = {
            "RealEsrgan-x2-Plus": "RealEsrganx2Plus",
            "RealEsrgan-x4-Plus": "RealEsrganx4Plus",
            "BSRGan-x2": "BSRGANx2",
            "BSRGan-x4": "BSRGANx4",
            "UltraSharp-x4": "UltraSharpx4",
            "UltraMix-x4": "UltraMixx4",
            "RealEsr-General-x4v3": "RealEsrx4v3",
            "Deoldify-Artistic": "DeoldifyArt",
            "Deoldify-Stable": "DeoldifyStable",
            "Deoldify-Video": "DeoldifyVideo",
            "DDColor-Artistic": "DDColorArt",
            "DDColor": "DDcolor",
        }

    def unload_models(self):
        with self.models_processor.model_lock:
            if self.current_enhancer_model:
                self.models_processor.unload_model(self.current_enhancer_model)
                self.current_enhancer_model = None

    def _run_model_with_lazy_build_check(self, model_name: str, ort_session, io_binding):
        """
        Runs the ONNX session with IOBinding, handling TensorRT lazy build dialogs.
        This centralizes the try/finally logic for showing/hiding the build progress dialog
        and includes the critical synchronization step for CUDA or other devices.

        Args:
            model_name (str): The name of the model being run.
            ort_session: The ONNX Runtime session instance.
            io_binding: The pre-configured IOBinding object.
        """
        # --- START LAZY BUILD CHECK ---
        is_lazy_build = self.models_processor.check_and_clear_pending_build(
            model_name
        )
        if is_lazy_build:
            self.models_processor.show_build_dialog.emit(
                "Finalizing TensorRT Build",
                f"Performing first-run inference for:\n{model_name}\n\nThis may take several minutes.",
            )
        
        try:
            # ⚠️ This is a critical synchronization point.
            if self.models_processor.device == "cuda":
                torch.cuda.synchronize()
            elif self.models_processor.device != "cpu":
                # This handles synchronization for other execution providers (e.g., DirectML)
                self.models_processor.syncvec.cpu()
                
            ort_session.run_with_iobinding(io_binding)
            
        finally:
            if is_lazy_build:
                self.models_processor.hide_build_dialog.emit()
        # --- END LAZY BUILD CHECK ---

    def run_enhance_frame_tile_process(
        self, img, enhancer_type, tile_size=256, scale=1
    ):
        new_model_to_load = self.model_map.get(enhancer_type)

        if new_model_to_load and self.current_enhancer_model != new_model_to_load:
            if self.current_enhancer_model:
                self.models_processor.unload_model(self.current_enhancer_model)
            self.current_enhancer_model = new_model_to_load

        _, _, height, width = img.shape

        # Calcolo del numero di tile necessari
        tiles_x = math.ceil(width / tile_size)
        tiles_y = math.ceil(height / tile_size)

        # Calcolo del padding necessario per adattare l'immagine alle dimensioni dei tile
        pad_right = (tile_size - (width % tile_size)) % tile_size
        pad_bottom = (tile_size - (height % tile_size)) % tile_size

        # Padding dell'immagine se necessario
        if pad_right != 0 or pad_bottom != 0:
            img = torch.nn.functional.pad(
                img, (0, pad_right, 0, pad_bottom), "constant", 0
            )

        # Creazione di un output tensor vuoto
        b, c, h, w = img.shape
        output = torch.empty(
            (b, c, h * scale, w * scale),
            dtype=torch.float32,
            device=self.models_processor.device,
        ).contiguous()

        # Selezione della funzione di upscaling in base al tipo
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

        if not fn_upscaler:  # Se il tipo di enhancer non è valido
            if pad_right != 0 or pad_bottom != 0:
                img = v2.functional.crop(img, 0, 0, height, width)
            return img

        with torch.no_grad():  # Disabilita il calcolo del gradiente
            # Elaborazione dei tile
            for j in range(tiles_y):
                for i in range(tiles_x):
                    x_start, y_start = i * tile_size, j * tile_size
                    x_end, y_end = x_start + tile_size, y_start + tile_size

                    # Estrazione del tile di input
                    input_tile = img[:, :, y_start:y_end, x_start:x_end].contiguous()
                    output_tile = torch.empty(
                        (
                            input_tile.shape[0],
                            input_tile.shape[1],
                            input_tile.shape[2] * scale,
                            input_tile.shape[3] * scale,
                        ),
                        dtype=torch.float32,
                        device=self.models_processor.device,
                    ).contiguous()

                    # Upscaling del tile
                    fn_upscaler(input_tile, output_tile)

                    # Inserimento del tile upscalato nel tensor di output
                    output_y_start, output_x_start = y_start * scale, x_start * scale
                    output_y_end, output_x_end = (
                        output_y_start + output_tile.shape[2],
                        output_x_start + output_tile.shape[3],
                    )
                    output[
                        :, :, output_y_start:output_y_end, output_x_start:output_x_end
                    ] = output_tile

            # Ritaglio dell'output per rimuovere il padding aggiunto
            if pad_right != 0 or pad_bottom != 0:
                output = v2.functional.crop(output, 0, 0, height * scale, width * scale)

        return output

    def run_realesrganx2(self, image, output):
        model_name = "RealEsrganx2Plus" # Define model_name
        if not self.models_processor.models[model_name]:
            self.models_processor.models[model_name] = (
                self.models_processor.load_model(model_name)
            )
        
        ort_session = self.models_processor.models[model_name]
        if not ort_session:
            print(f"WARNING: Model {model_name} not loaded, skipping enhancer.")
            output = image # Return original image if model fails
            return

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
            name="output",
            device_type=self.models_processor.device,
            device_id=0,
            element_type=np.float32,
            shape=output.size(),
            buffer_ptr=output.data_ptr(),
        )

        # Run the model with lazy build handling
        self._run_model_with_lazy_build_check(model_name, ort_session, io_binding)

    def run_realesrganx4(self, image, output):
        model_name = "RealEsrganx4Plus" # Define model_name
        if not self.models_processor.models[model_name]:
            self.models_processor.models[model_name] = (
                self.models_processor.load_model(model_name)
            )

        ort_session = self.models_processor.models[model_name]
        if not ort_session:
            print(f"WARNING: Model {model_name} not loaded, skipping enhancer.")
            output = image # Return original image if model fails
            return

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
            name="output",
            device_type=self.models_processor.device,
            device_id=0,
            element_type=np.float32,
            shape=output.size(),
            buffer_ptr=output.data_ptr(),
        )

        # Run the model with lazy build handling
        self._run_model_with_lazy_build_check(model_name, ort_session, io_binding)

    def run_realesrx4v3(self, image, output):
        model_name = "RealEsrx4v3" # Define model_name
        if not self.models_processor.models[model_name]:
            self.models_processor.models[model_name] = (
                self.models_processor.load_model(model_name)
            )

        ort_session = self.models_processor.models[model_name]
        if not ort_session:
            print(f"WARNING: Model {model_name} not loaded, skipping enhancer.")
            output = image # Return original image if model fails
            return

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
            name="output",
            device_type=self.models_processor.device,
            device_id=0,
            element_type=np.float32,
            shape=output.size(),
            buffer_ptr=output.data_ptr(),
        )

        # Run the model with lazy build handling
        self._run_model_with_lazy_build_check(model_name, ort_session, io_binding)

    def run_bsrganx2(self, image, output):
        model_name = "BSRGANx2" # Define model_name
        if not self.models_processor.models[model_name]:
            self.models_processor.models[model_name] = self.models_processor.load_model(
                model_name
            )

        ort_session = self.models_processor.models[model_name]
        if not ort_session:
            print(f"WARNING: Model {model_name} not loaded, skipping enhancer.")
            output = image # Return original image if model fails
            return

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
            name="output",
            device_type=self.models_processor.device,
            device_id=0,
            element_type=np.float32,
            shape=output.size(),
            buffer_ptr=output.data_ptr(),
        )

        # Run the model with lazy build handling
        self._run_model_with_lazy_build_check(model_name, ort_session, io_binding)

    def run_bsrganx4(self, image, output):
        model_name = "BSRGANx4" # Define model_name
        if not self.models_processor.models[model_name]:
            self.models_processor.models[model_name] = self.models_processor.load_model(
                model_name
            )

        ort_session = self.models_processor.models[model_name]
        if not ort_session:
            print(f"WARNING: Model {model_name} not loaded, skipping enhancer.")
            output = image # Return original image if model fails
            return

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
            name="output",
            device_type=self.models_processor.device,
            device_id=0,
            element_type=np.float32,
            shape=output.size(),
            buffer_ptr=output.data_ptr(),
        )

        # Run the model with lazy build handling
        self._run_model_with_lazy_build_check(model_name, ort_session, io_binding)

    def run_ultrasharpx4(self, image, output):
        model_name = "UltraSharpx4" # Define model_name
        if not self.models_processor.models[model_name]:
            self.models_processor.models[model_name] = (
                self.models_processor.load_model(model_name)
            )

        ort_session = self.models_processor.models[model_name]
        if not ort_session:
            print(f"WARNING: Model {model_name} not loaded, skipping enhancer.")
            output = image # Return original image if model fails
            return

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
            name="output",
            device_type=self.models_processor.device,
            device_id=0,
            element_type=np.float32,
            shape=output.size(),
            buffer_ptr=output.data_ptr(),
        )

        # Run the model with lazy build handling
        self._run_model_with_lazy_build_check(model_name, ort_session, io_binding)

    def run_ultramixx4(self, image, output):
        model_name = "UltraMixx4" # Define model_name
        if not self.models_processor.models[model_name]:
            self.models_processor.models[model_name] = (
                self.models_processor.load_model(model_name)
            )

        ort_session = self.models_processor.models[model_name]
        if not ort_session:
            print(f"WARNING: Model {model_name} not loaded, skipping enhancer.")
            output = image # Return original image if model fails
            return

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
            name="output",
            device_type=self.models_processor.device,
            device_id=0,
            element_type=np.float32,
            shape=output.size(),
            buffer_ptr=output.data_ptr(),
        )

        # Run the model with lazy build handling
        self._run_model_with_lazy_build_check(model_name, ort_session, io_binding)

    def run_deoldify_artistic(self, image, output):
        model_name = "DeoldifyArt" # Define model_name
        if not self.models_processor.models[model_name]:
            self.models_processor.models[model_name] = (
                self.models_processor.load_model(model_name)
            )

        ort_session = self.models_processor.models[model_name]
        if not ort_session:
            print(f"WARNING: Model {model_name} not loaded, skipping enhancer.")
            output = image # Return original image if model fails
            return

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
            name="output",
            device_type=self.models_processor.device,
            device_id=0,
            element_type=np.float32,
            shape=output.size(),
            buffer_ptr=output.data_ptr(),
        )

        # Run the model with lazy build handling
        self._run_model_with_lazy_build_check(model_name, ort_session, io_binding)

    def run_deoldify_stable(self, image, output):
        model_name = "DeoldifyStable" # Define model_name
        if not self.models_processor.models[model_name]:
            self.models_processor.models[model_name] = (
                self.models_processor.load_model(model_name)
            )

        ort_session = self.models_processor.models[model_name]
        if not ort_session:
            print(f"WARNING: Model {model_name} not loaded, skipping enhancer.")
            output = image # Return original image if model fails
            return

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
            name="output",
            device_type=self.models_processor.device,
            device_id=0,
            element_type=np.float32,
            shape=output.size(),
            buffer_ptr=output.data_ptr(),
        )

        # Run the model with lazy build handling
        self._run_model_with_lazy_build_check(model_name, ort_session, io_binding)

    def run_deoldify_video(self, image, output):
        model_name = "DeoldifyVideo" # Define model_name
        if not self.models_processor.models[model_name]:
            self.models_processor.models[model_name] = (
                self.models_processor.load_model(model_name)
            )

        ort_session = self.models_processor.models[model_name]
        if not ort_session:
            print(f"WARNING: Model {model_name} not loaded, skipping enhancer.")
            output = image # Return original image if model fails
            return

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
            name="output",
            device_type=self.models_processor.device,
            device_id=0,
            element_type=np.float32,
            shape=output.size(),
            buffer_ptr=output.data_ptr(),
        )

        # Run the model with lazy build handling
        self._run_model_with_lazy_build_check(model_name, ort_session, io_binding)

    def run_ddcolor_artistic(self, image, output):
        model_name = "DDColorArt" # Define model_name
        if not self.models_processor.models[model_name]:
            self.models_processor.models[model_name] = (
                self.models_processor.load_model(model_name)
            )

        ort_session = self.models_processor.models[model_name]
        if not ort_session:
            print(f"WARNING: Model {model_name} not loaded, skipping enhancer.")
            output = image # Return original image if model fails
            return

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
            name="output",
            device_type=self.models_processor.device,
            device_id=0,
            element_type=np.float32,
            shape=output.size(),
            buffer_ptr=output.data_ptr(),
        )

        # Run the model with lazy build handling
        self._run_model_with_lazy_build_check(model_name, ort_session, io_binding)

    def run_ddcolor(self, image, output):
        model_name = "DDcolor" # Define model_name
        if not self.models_processor.models[model_name]:
            self.models_processor.models[model_name] = self.models_processor.load_model(
                model_name
            )

        ort_session = self.models_processor.models[model_name]
        if not ort_session:
            print(f"WARNING: Model {model_name} not loaded, skipping enhancer.")
            output = image # Return original image if model fails
            return

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
            name="output",
            device_type=self.models_processor.device,
            device_id=0,
            element_type=np.float32,
            shape=output.size(),
            buffer_ptr=output.data_ptr(),
        )

        # Run the model with lazy build handling
        self._run_model_with_lazy_build_check(model_name, ort_session, io_binding)
