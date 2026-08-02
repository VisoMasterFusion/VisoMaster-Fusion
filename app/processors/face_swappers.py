import torch
import threading
from skimage import transform as trans
from torchvision.transforms import v2
from app.processors.utils import faceutil
import numpy as np
from numpy.linalg import norm as l2norm
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.processors.models_processor import ModelsProcessor
    from app.processors.workers.function_worker import FunctionWorker


class FaceSwappers:
    def __init__(
        self,
        models_processor: "ModelsProcessor",
        function_worker: "FunctionWorker",
    ):
        self.models_processor = models_processor
        self.function_worker = function_worker
        self.current_swapper_model = None
        self.current_arcface_model = None
        self._session_io_name_cache: dict = {}  # FS-PERF-02: cache input/output names keyed by session id
        self._io_cache_lock = threading.Lock()
        self.resize_112 = v2.Resize(
            (112, 112), interpolation=v2.InterpolationMode.BILINEAR, antialias=False
        )
        self.swapper_models = [
            "Inswapper128",
            "InStyleSwapper256 Version A",
            "InStyleSwapper256 Version B",
            "InStyleSwapper256 Version C",
            "SimSwap512",
            "GhostFacev1",
            "GhostFacev2",
            "GhostFacev3",
            "CSCS",
        ]
        self.arcface_models = [
            "Inswapper128ArcFace",
            "SimSwapArcFace",
            "GhostArcFace",
            "CSCSArcFace",
            "CSCSIDArcFace",
        ]

    def unload_models(self):
        with self.models_processor.model_lock:
            for model_name in self.swapper_models:
                self.models_processor.unload_model(model_name)
            for model_name in self.arcface_models:
                self.models_processor.unload_model(model_name)

    def _manage_model(self, new_model_name):
        # FS-RACE-01: protect read-modify-write of current_swapper_model with lock
        with self.models_processor.model_lock:
            if (
                self.current_swapper_model
                and self.current_swapper_model != new_model_name
            ):
                self.models_processor.unload_model(self.current_swapper_model)
            # FS-BUG-07: current_swapper_model is committed only after load confirmation (see _load_swapper_model)

    def _load_swapper_model(self, model_name):
        """Handles loading and swapping of swapper models."""
        self._manage_model(model_name)
        model = self.models_processor.models.get(model_name)
        if not model:
            model = self.models_processor.load_model(model_name)
        # FS-BUG-07: only commit state after load is confirmed non-None
        if model is not None:
            with self.models_processor.model_lock:
                self.current_swapper_model = model_name
        return model

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
                self.models_processor.syncvec.cpu()

            ort_session.run_with_iobinding(io_binding)

        finally:
            if is_lazy_build:
                self.models_processor.hide_build_dialog.emit()

    def run_recognize_direct(
        self, img, kps, similarity_type="Auto", arcface_model="Inswapper128ArcFace"
    ):
        # FS-RACE-01: protect read-modify-write of current_arcface_model with lock
        with self.models_processor.model_lock:
            if (
                self.current_arcface_model
                and self.current_arcface_model != arcface_model
            ):
                self.models_processor.unload_model(self.current_arcface_model)
            self.current_arcface_model = arcface_model

        ort_session = self.models_processor.models.get(arcface_model)
        if not ort_session:
            ort_session = self.models_processor.load_model(arcface_model)

        if not ort_session:
            print(
                f"[WARN] ArcFace model '{arcface_model}' failed to load. Skipping recognition."
            )
            return None, None

        if arcface_model == "CSCSArcFace":
            embedding, cropped_image = self.recognize_cscs(img, kps)
        else:
            embedding, cropped_image = self.recognize(
                arcface_model, img, kps, similarity_type=similarity_type
            )

        return embedding, cropped_image

    def run_recognize(
        self, img, kps, similarity_type="Auto", face_swapper_model="Inswapper128"
    ):
        arcface_model = self.function_worker.get_arcface_model(face_swapper_model)
        return self.run_recognize_direct(img, kps, similarity_type, arcface_model)

    def recognize(self, arcface_model, img, face_kps, similarity_type=None):
        """
        Generates the face embedding using the specified ArcFace model and alignment strategy.
        Automatically handles pose-awareness, ignoring user UI choice to prevent corrupted embeddings.
        """
        ort_session = self.models_processor.models.get(arcface_model)
        if not ort_session:
            return None, None

        # --- FORCED DYNAMIC "AUTO" MODE (STATELESS) ---
        # We ignore 'similarity_type' from UI to protect ArcFace from bad alignments (like Pearl).
        yaw, pitch = faceutil.calc_face_yaw_pitch(face_kps)

        if abs(yaw) > 30.0 or abs(pitch) > 30.0:
            actual_mode = "Optimal"
        else:
            actual_mode = "Opal"

        # --- ALIGNMENT STRATEGIES (Delegated to faceutil) ---
        if actual_mode == "Optimal":
            # Pose-aware 7-templates alignment (arcfacemap) for side faces
            img, _ = faceutil.warp_face_by_face_landmark_5(
                img,
                face_kps,
                image_size=112,
                mode="arcfacemap",
                interpolation=v2.InterpolationMode.BILINEAR,
            )
        else:
            # Standard frontal alignment (arcface112) for front faces
            img, _ = faceutil.warp_face_by_face_landmark_5(
                img,
                face_kps,
                image_size=112,
                mode="arcface112",
                interpolation=v2.InterpolationMode.BILINEAR,
            )

        # --- NORMALIZATION & PRE-PROCESSING ---
        cropped_image = img.permute(1, 2, 0).clone()  # Store for display/debug (H,W,3)

        if img.dtype == torch.uint8:
            img = img.float()

        # CLONE: Prevent race conditions if Kornia passed a reference
        img = img.clone()

        # OPTIMIZED: In-Place math operations (.sub_ and .div_)
        if arcface_model == "Inswapper128ArcFace":
            if img.max() <= 1.0:
                img = img * 255.0
            img.sub_(127.5).div_(127.5)

        elif arcface_model == "SimSwapArcFace":
            img.div_(255.0)
            v2.functional.normalize(
                img, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), inplace=True
            )
        else:
            img.div_(127.5).sub_(1.0)

        # --- INFERENCE ---
        img = torch.unsqueeze(img, 0).contiguous()

        session_id = id(ort_session)
        with self._io_cache_lock:
            if session_id not in self._session_io_name_cache:
                self._session_io_name_cache[session_id] = {
                    "input": ort_session.get_inputs()[0].name,
                    "outputs": [o.name for o in ort_session.get_outputs()],
                }
            input_name = self._session_io_name_cache[session_id]["input"]
            output_names = self._session_io_name_cache[session_id]["outputs"]

        io_binding = ort_session.io_binding()
        io_binding.bind_input(
            name=input_name,
            device_type=self.models_processor.device_type,
            device_id=self.models_processor.binding_device_id,
            element_type=np.float32,
            shape=img.size(),
            buffer_ptr=img.data_ptr(),
        )

        for name in output_names:
            io_binding.bind_output(
                name,
                self.models_processor.device_type,
                self.models_processor.binding_device_id,
            )

        self._run_model_with_lazy_build_check(arcface_model, ort_session, io_binding)

        return np.array(io_binding.copy_outputs_to_cpu()).flatten(), cropped_image

    def preprocess_image_cscs(self, img, face_kps):
        """
        Preprocesses the image for the CSCS ArcFace models.
        OPTIMIZED: Uses torchvision v2 for fast GPU affine transformations.
        BUGFIX: Resolves skimage deprecation warning while keeping exact
        mathematical alignment required by CSCS.
        """
        # OPTIMIZED: Fix deprecation warning using from_estimate
        tform = trans.SimilarityTransform.from_estimate(
            face_kps, self.models_processor.FFHQ_kps
        )

        # GPU Accelerated Affine Transformation (img is already a GPU Tensor here)
        # We preserve the exact center=(0,0) geometry required by CSCS models.
        temp = v2.functional.affine(
            img,
            angle=tform.rotation * 57.2958,  # Rad to Deg
            translate=(tform.translation[0], tform.translation[1]),
            scale=tform.scale,
            shear=0.0,
            center=(0, 0),
        )

        # Fast GPU Crop and Resize
        temp = v2.functional.crop(temp, top=0, left=0, height=512, width=512)
        image = self.resize_112(temp)

        cropped_image = image.permute(1, 2, 0).clone()

        if image.dtype == torch.uint8:
            image = image.float()

        # CLONE: Prevent cross-thread race conditions before in-place math
        image = image.clone()

        # OPTIMIZED: In-place division and normalization for CSCS [-1.0, 1.0] standard
        image.div_(255.0)
        v2.functional.normalize(image, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)

        return torch.unsqueeze(image, 0).contiguous(), cropped_image

    def recognize_cscs(self, img, face_kps):
        img, cropped_image = self.preprocess_image_cscs(img, face_kps)

        model_name = "CSCSArcFace"
        model = self.models_processor.models.get(model_name)
        if not model:
            print("[ERROR] CSCSArcFace model not loaded in recognize_cscs.")
            return None, None

        io_binding = model.io_binding()

        # SAFETY: Clear bindings to prevent thread caching errors
        io_binding.clear_binding_inputs()
        io_binding.clear_binding_outputs()

        io_binding.bind_input(
            name="input",
            device_type=self.models_processor.device_type,
            device_id=self.models_processor.binding_device_id,
            element_type=np.float32,
            shape=img.size(),
            buffer_ptr=img.data_ptr(),
        )
        io_binding.bind_output(
            name="output",
            device_type=self.models_processor.device_type,
            device_id=self.models_processor.binding_device_id,
        )

        self._run_model_with_lazy_build_check(model_name, model, io_binding)

        output = io_binding.copy_outputs_to_cpu()[0]

        # Exact p=2 normalization math required by CSCS
        embedding = torch.from_numpy(output).to("cpu")
        embedding = torch.nn.functional.normalize(embedding, dim=-1, p=2)
        embedding = embedding.numpy().flatten()

        embedding_id = self.recognize_cscs_id_adapter(img, None)

        if embedding_id.size == embedding.size:
            embedding = embedding + embedding_id

        return embedding, cropped_image

    def recognize_cscs_id_adapter(self, img, face_kps):
        model_name = "CSCSIDArcFace"
        model = self.models_processor.models.get(model_name)
        if not model:
            model = self.models_processor.load_model(model_name)

        if not model:
            print(f"[WARN] {model_name} model not loaded.")
            return np.array([])

        if face_kps is not None:
            img, _ = self.preprocess_image_cscs(img, face_kps)

        io_binding = model.io_binding()

        # SAFETY: Clear bindings
        io_binding.clear_binding_inputs()
        io_binding.clear_binding_outputs()

        io_binding.bind_input(
            name="input",
            device_type=self.models_processor.device_type,
            device_id=self.models_processor.binding_device_id,
            element_type=np.float32,
            shape=img.size(),
            buffer_ptr=img.data_ptr(),
        )
        io_binding.bind_output(
            name="output",
            device_type=self.models_processor.device_type,
            device_id=self.models_processor.binding_device_id,
        )

        self._run_model_with_lazy_build_check(model_name, model, io_binding)

        output = io_binding.copy_outputs_to_cpu()[0]

        # Exact p=2 normalization math required by CSCS
        embedding_id = torch.from_numpy(output).to("cpu")
        embedding_id = torch.nn.functional.normalize(embedding_id, dim=-1, p=2)

        return embedding_id.numpy().flatten()

    def calc_swapper_latent_cscs(self, source_embedding: np.ndarray) -> np.ndarray:
        latent = source_embedding.reshape((1, -1))
        return latent

    def run_swapper_cscs(
        self, image: torch.Tensor, embedding: torch.Tensor, output: torch.Tensor
    ) -> None:
        model_name = "CSCS"
        model = self._load_swapper_model(model_name)
        if not model:
            print("[ERROR] CSCS model not loaded.")
            return

        # SAFETY: Contiguous memory blocks required by TensorRT
        if not image.is_contiguous():
            image = image.contiguous()
        if not embedding.is_contiguous():
            embedding = embedding.contiguous()
        if not output.is_contiguous():
            output = output.contiguous()

        io_binding = model.io_binding()

        # SAFETY: Clear bindings
        io_binding.clear_binding_inputs()
        io_binding.clear_binding_outputs()

        # Hardcoded IO names validated by standard CSCS export
        io_binding.bind_input(
            name="input_1",
            device_type=self.models_processor.device_type,
            device_id=self.models_processor.binding_device_id,
            element_type=np.float32,
            shape=(1, 3, 256, 256),
            buffer_ptr=image.data_ptr(),
        )
        io_binding.bind_input(
            name="input_2",
            device_type=self.models_processor.device_type,
            device_id=self.models_processor.binding_device_id,
            element_type=np.float32,
            shape=(1, 512),
            buffer_ptr=embedding.data_ptr(),
        )
        io_binding.bind_output(
            name="output",
            device_type=self.models_processor.device_type,
            device_id=self.models_processor.binding_device_id,
            element_type=np.float32,
            shape=(1, 3, 256, 256),
            buffer_ptr=output.data_ptr(),
        )

        self._run_model_with_lazy_build_check(model_name, model, io_binding)

    def _calc_emap_latent(self, source_embedding: np.ndarray) -> np.ndarray:
        """FS-PERF-05: shared emap-based latent computation extracted from
        calc_inswapper_latent and calc_swapper_latent_iss."""
        n_e = source_embedding / l2norm(source_embedding)
        latent = n_e.reshape((1, -1))
        latent = np.dot(latent, self.models_processor.emap)
        latent /= np.linalg.norm(latent)
        return latent

    def _ensure_emap(self) -> bool:
        """Ensures emap is loaded; returns True if available, False otherwise."""
        if (
            not hasattr(self.models_processor, "emap")
            or not isinstance(self.models_processor.emap, np.ndarray)
            or self.models_processor.emap.size == 0
        ):
            self.models_processor.load_model("Inswapper128")

        return (
            hasattr(self.models_processor, "emap")
            and isinstance(self.models_processor.emap, np.ndarray)
            and self.models_processor.emap.size > 0
        )

    def calc_inswapper_latent(self, source_embedding: np.ndarray) -> np.ndarray | None:
        if not self._ensure_emap():
            print("[ERROR] Emap could not be loaded for latent calculation.")
            # FS-ROBUST-01: return None so callers can detect and handle the failure
            return None

        return self._calc_emap_latent(source_embedding)

    def run_inswapper(
        self, image: torch.Tensor, embedding: torch.Tensor, output: torch.Tensor
    ) -> None:
        model_name = "Inswapper128"

        # ORT-based inference
        model = self._load_swapper_model(model_name)
        if not model:
            print("[ERROR] Inswapper128 model not loaded.")
            return

        # FORCE CONTIGUOUS: Essential safety check.
        # Ensures that the memory pointer passed to TensorRT is valid and linear.
        if not image.is_contiguous():
            image = image.contiguous()
        if not embedding.is_contiguous():
            embedding = embedding.contiguous()
        if not output.is_contiguous():
            output = output.contiguous()

        io_binding = model.io_binding()

        # Clear previous bindings to avoid pointer caching issues
        io_binding.clear_binding_inputs()
        io_binding.clear_binding_outputs()

        io_binding.bind_input(
            name="target",
            device_type=self.models_processor.device_type,
            device_id=self.models_processor.binding_device_id,
            element_type=np.float32,
            shape=(1, 3, 128, 128),
            buffer_ptr=image.data_ptr(),
        )
        io_binding.bind_input(
            name="source",
            device_type=self.models_processor.device_type,
            device_id=self.models_processor.binding_device_id,
            element_type=np.float32,
            shape=(1, 512),
            buffer_ptr=embedding.data_ptr(),
        )
        io_binding.bind_output(
            name="output",
            device_type=self.models_processor.device_type,
            device_id=self.models_processor.binding_device_id,
            element_type=np.float32,
            shape=(1, 3, 128, 128),
            buffer_ptr=output.data_ptr(),
        )

        # Run the model with lazy build handling
        self._run_model_with_lazy_build_check(model_name, model, io_binding)

    def run_inswapper_batched(
        self, images: torch.Tensor, embedding: torch.Tensor, output: torch.Tensor
    ) -> None:
        """Batched InSwapper inference for pixel-shift resolution mode.

        Optimized zero-copy tile execution using a persistent IOBinding instance
        and direct GPU pointer offsets into contiguous memory blocks.
        """
        model_name: str = "Inswapper128"
        model = self._load_swapper_model(model_name)
        if not model:
            print("[ERROR] Inswapper128 model not loaded.")
            return

        # Ensure memory contiguity once outside the tile loop
        if not images.is_contiguous():
            images = images.contiguous()
        if not embedding.is_contiguous():
            embedding = embedding.contiguous()
        if not output.is_contiguous():
            output = output.contiguous()

        batch_size: int = images.shape[0]
        device_type: str = self.models_processor.device_type
        device_id: int = self.models_processor.binding_device_id

        # Reuse a single IOBinding instance across tiles
        io_binding = model.io_binding()

        for idx in range(batch_size):
            img_tile: torch.Tensor = images[idx : idx + 1]
            out_tile: torch.Tensor = output[idx : idx + 1]

            io_binding.clear_binding_inputs()
            io_binding.clear_binding_outputs()

            io_binding.bind_input(
                name="target",
                device_type=device_type,
                device_id=device_id,
                element_type=np.float32,
                shape=(1, 3, 128, 128),
                buffer_ptr=img_tile.data_ptr(),
            )
            io_binding.bind_input(
                name="source",
                device_type=device_type,
                device_id=device_id,
                element_type=np.float32,
                shape=(1, 512),
                buffer_ptr=embedding.data_ptr(),
            )
            io_binding.bind_output(
                name="output",
                device_type=device_type,
                device_id=device_id,
                element_type=np.float32,
                shape=(1, 3, 128, 128),
                buffer_ptr=out_tile.data_ptr(),
            )

            self._run_model_with_lazy_build_check(model_name, model, io_binding)

    def calc_swapper_latent_ghost(self, source_embedding: np.ndarray) -> np.ndarray:
        latent = source_embedding.reshape((1, -1))

        return latent

    def calc_swapper_latent_iss(
        self, source_embedding: np.ndarray, version: str = "A"
    ) -> np.ndarray:
        # FS-PERF-05: reuse shared _ensure_emap / _calc_emap_latent helpers
        if not self._ensure_emap():
            print("[ERROR] Emap could not be loaded for latent calculation.")
            n_e = source_embedding / l2norm(source_embedding)
            return n_e.reshape((1, -1))

        return self._calc_emap_latent(source_embedding)

    def run_iss_swapper(
        self,
        image: torch.Tensor,
        embedding: torch.Tensor,
        output: torch.Tensor,
        version: str = "A",
    ) -> None:
        model_name = f"InStyleSwapper256 Version {version}"
        model = self._load_swapper_model(model_name)
        if not model:
            print(f"[ERROR] {model_name} model not loaded.")
            return

        io_binding = model.io_binding()
        io_binding.bind_input(
            name="target",
            device_type=self.models_processor.device_type,
            device_id=self.models_processor.binding_device_id,
            element_type=np.float32,
            shape=(1, 3, 256, 256),
            buffer_ptr=image.data_ptr(),
        )
        io_binding.bind_input(
            name="source",
            device_type=self.models_processor.device_type,
            device_id=self.models_processor.binding_device_id,
            element_type=np.float32,
            shape=(1, 512),
            buffer_ptr=embedding.data_ptr(),
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
        self._run_model_with_lazy_build_check(model_name, model, io_binding)

    def calc_swapper_latent_simswap512(
        self, source_embedding: np.ndarray
    ) -> np.ndarray:
        latent = source_embedding.reshape(1, -1)
        # latent /= np.linalg.norm(latent)
        latent = latent / np.linalg.norm(latent, axis=1, keepdims=True)
        return latent

    def run_swapper_simswap512(
        self, image: torch.Tensor, embedding: torch.Tensor, output: torch.Tensor
    ) -> None:
        model_name = "SimSwap512"
        model = self._load_swapper_model(model_name)
        if not model:
            print("[ERROR] SimSwap512 model not loaded.")
            return

        io_binding = model.io_binding()
        io_binding.bind_input(
            name="input",
            device_type=self.models_processor.device_type,
            device_id=self.models_processor.binding_device_id,
            element_type=np.float32,
            shape=(1, 3, 512, 512),
            buffer_ptr=image.data_ptr(),
        )
        io_binding.bind_input(
            name="onnx::Gemm_1",
            device_type=self.models_processor.device_type,
            device_id=self.models_processor.binding_device_id,
            element_type=np.float32,
            shape=(1, 512),
            buffer_ptr=embedding.data_ptr(),
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
        self._run_model_with_lazy_build_check(model_name, model, io_binding)

    def run_swapper_ghostface(
        self,
        image: torch.Tensor,
        embedding: torch.Tensor,
        output: torch.Tensor,
        swapper_model: str = "GhostFace-v2",
    ) -> None:
        model_name = None
        if swapper_model == "GhostFace-v1":
            model_name = "GhostFacev1"
        elif swapper_model == "GhostFace-v2":
            model_name = "GhostFacev2"
        elif swapper_model == "GhostFace-v3":
            model_name = "GhostFacev3"

        if not model_name:
            print(f"[ERROR] Unknown GhostFace model version: {swapper_model}")
            return

        ghostfaceswap_model = self._load_swapper_model(model_name)
        if not ghostfaceswap_model:
            print(f"[ERROR] {model_name} model not loaded.")
            return

        # FS-ROBUST-02: introspect output name dynamically instead of hardcoding node IDs
        session_id = id(ghostfaceswap_model)
        with self._io_cache_lock:
            if session_id not in self._session_io_name_cache:
                self._session_io_name_cache[session_id] = {
                    "input": ghostfaceswap_model.get_inputs()[0].name,
                    "outputs": [o.name for o in ghostfaceswap_model.get_outputs()],
                }
            output_name = self._session_io_name_cache[session_id]["outputs"][0]

        io_binding = ghostfaceswap_model.io_binding()
        io_binding.bind_input(
            name="target",
            device_type=self.models_processor.device_type,
            device_id=self.models_processor.binding_device_id,
            element_type=np.float32,
            shape=(1, 3, 256, 256),
            buffer_ptr=image.data_ptr(),
        )
        io_binding.bind_input(
            name="source",
            device_type=self.models_processor.device_type,
            device_id=self.models_processor.binding_device_id,
            element_type=np.float32,
            shape=(1, 512),
            buffer_ptr=embedding.data_ptr(),
        )
        io_binding.bind_output(
            name=output_name,
            device_type=self.models_processor.device_type,
            device_id=self.models_processor.binding_device_id,
            element_type=np.float32,
            shape=(1, 3, 256, 256),
            buffer_ptr=output.data_ptr(),
        )

        # Run the model with lazy build handling
        self._run_model_with_lazy_build_check(
            model_name, ghostfaceswap_model, io_binding
        )
