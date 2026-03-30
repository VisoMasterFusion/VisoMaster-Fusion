import torch
import threading
from skimage import transform as trans
from torchvision.transforms import v2
from app.processors.utils import faceutil
import numpy as np
from numpy.linalg import norm as l2norm
from typing import TYPE_CHECKING, Optional
import kornia.geometry.transform as kgm

if TYPE_CHECKING:
    from app.processors.models_processor import ModelsProcessor


class FaceSwappers:
    def __init__(self, models_processor: "ModelsProcessor"):
        self.models_processor = models_processor
        self.current_swapper_model = None
        self.current_arcface_model = None
        self._session_io_name_cache: dict = {}  # FS-PERF-02: cache input/output names keyed by session id
        self._io_cache_lock = threading.Lock()
        self._inswapper_init_lock = threading.Lock()
        self._w600k_lock = threading.Lock()
        self._inswapper_b1_lock = threading.Lock()
        self._inswapper_batched_lock = threading.Lock()
        self._inswapper_torch = None  # InSwapperTorch instance
        self._inswapper_runner_b1: Optional[object] = None  # CUDA graph runner for B=1
        self._w600k_torch: Optional[object] = None  # IResNet50Torch
        self._w600k_runner: Optional[object] = None  # _CapturedGraph or eager model
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
        with self._inswapper_init_lock:
            self._inswapper_torch = None
            self._inswapper_runner_b1 = None
            self._w600k_torch = None
            self._w600k_runner = None

    def _manage_model(self, new_model_name):
        # FS-RACE-01: protect read-modify-write of current_swapper_model with lock
        with self.models_processor.model_lock:
            if (
                self.current_swapper_model
                and self.current_swapper_model != new_model_name
            ):
                self.models_processor.unload_model(self.current_swapper_model)
                # Free InSwapperTorch when switching away from Inswapper128
                if self.current_swapper_model == "Inswapper128":
                    with self._inswapper_init_lock:
                        self._inswapper_torch = None
                        self._inswapper_runner_b1 = None
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
            if self.models_processor.device == "cuda":
                torch.cuda.current_stream().synchronize()
            elif self.models_processor.device != "cpu":
                # This handles synchronization for other execution providers (e.g., DirectML)
                self.models_processor.syncvec.cpu()

            ort_session.run_with_iobinding(io_binding)

        finally:
            if is_lazy_build:
                self.models_processor.hide_build_dialog.emit()

    def _get_w600k_runner(self):
        """Lazy-load IResNet50Torch + CUDA graph runner for Inswapper128ArcFace."""
        if self._w600k_runner is not None:
            return self._w600k_runner
        self.models_processor.show_build_dialog.emit(
            "Finalizing Custom Provider",
            "Compiling & capturing CUDA graph for ArcFace (w600k)…\nFirst run only — future sessions load instantly from cache.",
        )
        try:
            with self._inswapper_init_lock:
                if self._w600k_runner is not None:
                    return self._w600k_runner
                if self._w600k_torch is None:
                    try:
                        import pathlib
                        from custom_kernels.w600k_r50.w600k_r50_torch import (
                            IResNet50Torch,
                        )

                        onnx_path = str(
                            pathlib.Path(__file__).parent.parent.parent
                            / "model_assets"
                            / "w600k_r50.onnx"
                        )
                        m = (
                            IResNet50Torch.from_onnx(onnx_path)
                            .to(self.models_processor.device)
                            .eval()
                        )
                        self._w600k_torch = m
                    except Exception as e:
                        print(f"[Custom] w600k_r50 load failed: {e}")
                        return None
                try:
                    from custom_kernels.w600k_r50.w600k_r50_torch import (
                        build_cuda_graph_runner,
                    )

                    with self.models_processor.cuda_graph_capture_lock:
                        self._w600k_runner = build_cuda_graph_runner(
                            self._w600k_torch, torch_compile=False
                        )
                except Exception as e:
                    print(f"[Custom] w600k_r50 graph runner failed, using eager: {e}")
                    self._w600k_runner = self._w600k_torch
        finally:
            self.models_processor.hide_build_dialog.emit()
        return self._w600k_runner

    def run_recognize_direct(
        self, img, kps, similarity_type="Opal", arcface_model="Inswapper128ArcFace"
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
        self, img, kps, similarity_type="Opal", face_swapper_model="Inswapper128"
    ):
        arcface_model = self.models_processor.get_arcface_model(face_swapper_model)
        return self.run_recognize_direct(img, kps, similarity_type, arcface_model)

    def recognize(self, arcface_model, img, face_kps, similarity_type):
        """
        Generates the face embedding using the specified ArcFace model and alignment strategy.

        Args:
            arcface_model (str): Name of the model to use.
            img (torch.Tensor): Input image tensor (CHW).
            face_kps (np.ndarray): 5 facial landmarks.
            similarity_type (str): Alignment strategy ('Optimal', 'Pearl', 'Opal').

        Returns:
            tuple: (embedding numpy array, cropped_face tensor HWC)
        """
        ort_session = self.models_processor.models.get(arcface_model)
        if not ort_session:
            # This is a safety check; run_recognize_direct should prevent this.
            return None, None

        # --- ALIGNMENT STRATEGIES ---
        if similarity_type == "Optimal":
            img, _ = faceutil.warp_face_by_face_landmark_5(
                img,
                face_kps,
                mode="arcfacemap",
                interpolation=v2.InterpolationMode.BILINEAR,
            )

        elif similarity_type == "Pearl":
            dst = self.models_processor.arcface_dst.copy()
            dst[:, 0] += 8.0
            tform = trans.SimilarityTransform.from_estimate(face_kps, dst)

            # OPTIMIZED: Direct GPU Warp to 128x128 using Kornia
            M_tensor = (
                torch.from_numpy(tform.params[0:2]).float().unsqueeze(0).to(img.device)
            )
            img_b = img.unsqueeze(0) if img.dim() == 3 else img
            img = kgm.warp_affine(
                img_b.float(),
                M_tensor,
                dsize=(128, 128),
                mode="bilinear",
                align_corners=True,
            ).squeeze(0)

            # Fast resize to standard 112
            img = v2.functional.resize(img, [112, 112], antialias=True)

        else:
            # Mode 3: Opal (Standard / Default)
            tform = trans.SimilarityTransform.from_estimate(
                face_kps, self.models_processor.arcface_dst
            )

            # OPTIMIZED: Direct GPU Warp to 112x112 using Kornia (bypasses torchvision crop/affine)
            M_tensor = (
                torch.from_numpy(tform.params[0:2]).float().unsqueeze(0).to(img.device)
            )
            img_b = img.unsqueeze(0) if img.dim() == 3 else img
            img = kgm.warp_affine(
                img_b.float(),
                M_tensor,
                dsize=(112, 112),
                mode="bilinear",
                align_corners=True,
            ).squeeze(0)

        # --- NORMALIZATION & PRE-PROCESSING ---
        cropped_image = img.permute(1, 2, 0).clone()  # Store for display/debug (H,W,3)

        # Ensure float format
        if img.dtype == torch.uint8:
            img = img.float()

        # We MUST clone the image before doing in-place math if we are
        # not strictly sure that we own a brand new Kornia tensor.
        # "Optimal" mode might pass a reference, causing Race Conditions across threads.
        img = img.clone()

        # OPTIMIZED: In-Place math operations (.sub_ and .div_) to save VRAM fragmentation
        if arcface_model == "Inswapper128ArcFace":
            # FS-BUG-03: ensure input is in [0, 255] before normalizing
            if img.max() <= 1.0:
                img = img * 255.0
            img.sub_(127.5).div_(127.5)

        elif arcface_model == "SimSwapArcFace":
            img.div_(255.0)
            v2.functional.normalize(
                img, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), inplace=True
            )

        else:
            # GhostArcFace, CSCSArcFace, etc.
            img.div_(127.5).sub_(1.0)

        # --- INFERENCE ---
        # Prepare data (N, C, H, W)
        img = torch.unsqueeze(img, 0).contiguous()

        # Custom provider: use PyTorch IResNet50Torch for Inswapper128ArcFace
        if (
            self.models_processor.provider_name == "Custom"
            and arcface_model == "Inswapper128ArcFace"
        ):
            runner = self._get_w600k_runner()
            if runner is not None:
                with torch.no_grad():
                    with self._w600k_lock:
                        embedding = runner(img)
                        if self.models_processor.device == "cuda":
                            torch.cuda.current_stream().synchronize()
                    embedding_np = embedding.cpu().numpy().flatten()
                return embedding_np, cropped_image
            # runner unavailable — fall through to ORT

        # FS-PERF-02: cache input/output names by session id to avoid repeated ONNX introspection
        # Lock prevents 'dictionary changed size during iteration' crashes when multiple
        # workers encounter a new model ID simultaneously.
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
            device_type=self.models_processor.device,
            device_id=0,
            element_type=np.float32,
            shape=img.size(),
            buffer_ptr=img.data_ptr(),
        )

        for name in output_names:
            io_binding.bind_output(name, self.models_processor.device)

        # Run the model with lazy build handling (TensorRT safety)
        self._run_model_with_lazy_build_check(arcface_model, ort_session, io_binding)

        # Return embedding (flattened) and the cropped image for visualization
        return np.array(io_binding.copy_outputs_to_cpu()).flatten(), cropped_image

    def preprocess_image_cscs(self, img, face_kps):
        tform = trans.SimilarityTransform.from_estimate(
            face_kps, self.models_processor.FFHQ_kps
        )

        # OPTIMIZED: Direct GPU Warp to 512x512 using Kornia
        M_tensor = (
            torch.from_numpy(tform.params[0:2]).float().unsqueeze(0).to(img.device)
        )
        img_b = img.unsqueeze(0) if img.dim() == 3 else img
        temp = kgm.warp_affine(
            img_b.float(),
            M_tensor,
            dsize=(512, 512),
            mode="bilinear",
            align_corners=True,
        ).squeeze(0)

        # Fast resize to 112x112
        image = v2.functional.resize(temp, [112, 112], antialias=True)

        cropped_image = image.permute(1, 2, 0).clone()

        if image.dtype == torch.uint8:
            image = image.float()

        # OPTIMIZED: In-place division and normalization
        image.div_(255.0)
        v2.functional.normalize(image, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)

        return torch.unsqueeze(image, 0).contiguous(), cropped_image

    def recognize_cscs(self, img, face_kps):
        # Usa la funzione di preprocessamento
        img, cropped_image = self.preprocess_image_cscs(img, face_kps)

        model_name = "CSCSArcFace"  # Define model_name
        model = self.models_processor.models.get(model_name)
        if not model:
            print("[ERROR] CSCSArcFace model not loaded in recognize_cscs.")
            return None, None

        io_binding = model.io_binding()
        io_binding.bind_input(
            name="input",
            device_type=self.models_processor.device,
            device_id=0,
            element_type=np.float32,
            shape=img.size(),
            buffer_ptr=img.data_ptr(),
        )
        io_binding.bind_output(name="output", device_type=self.models_processor.device)

        # Run the model with lazy build handling
        self._run_model_with_lazy_build_check(model_name, model, io_binding)

        output = io_binding.copy_outputs_to_cpu()[0]
        embedding = torch.from_numpy(output).to("cpu")
        embedding = torch.nn.functional.normalize(embedding, dim=-1, p=2)
        embedding = embedding.numpy().flatten()

        embedding_id = self.recognize_cscs_id_adapter(img, None)
        # FS-BUG-02: guard against empty embedding_id before combining
        if embedding_id.size == embedding.size:
            # FS-BUG-01: normalize the combined embedding to avoid magnitude blow-up
            combined = embedding + embedding_id
            norm = np.linalg.norm(combined)
            embedding = combined / norm if norm > 1e-8 else combined

        return embedding, cropped_image

    def recognize_cscs_id_adapter(self, img, face_kps):
        model_name = "CSCSIDArcFace"
        model = self.models_processor.models.get(model_name)
        if not model:
            model = self.models_processor.load_model(model_name)

        if not model:
            print(f"[WARN] {model_name} model not loaded.")
            return np.array([])  # Return empty array on failure

        # Use preprocess_image_cscs when face_kps is not None. When it is None img is already preprocessed.
        if face_kps is not None:
            img, _ = self.preprocess_image_cscs(img, face_kps)

        io_binding = model.io_binding()
        io_binding.bind_input(
            name="input",
            device_type=self.models_processor.device,
            device_id=0,
            element_type=np.float32,
            shape=img.size(),
            buffer_ptr=img.data_ptr(),
        )
        io_binding.bind_output(name="output", device_type=self.models_processor.device)

        # Run the model with lazy build handling
        self._run_model_with_lazy_build_check(model_name, model, io_binding)

        output = io_binding.copy_outputs_to_cpu()[0]
        embedding_id = torch.from_numpy(output).to("cpu")
        embedding_id = torch.nn.functional.normalize(embedding_id, dim=-1, p=2)

        return embedding_id.numpy().flatten()

    def calc_swapper_latent_cscs(self, source_embedding):
        latent = source_embedding.reshape((1, -1))
        return latent

    def run_swapper_cscs(self, image, embedding, output):
        model_name = "CSCS"  # Use the name from the models_list
        model = self._load_swapper_model(model_name)
        if not model:
            print("[ERROR] CSCS model not loaded.")
            return

        io_binding = model.io_binding()
        io_binding.bind_input(
            name="input_1",
            device_type=self.models_processor.device,
            device_id=0,
            element_type=np.float32,
            shape=(1, 3, 256, 256),
            buffer_ptr=image.data_ptr(),
        )
        io_binding.bind_input(
            name="input_2",
            device_type=self.models_processor.device,
            device_id=0,
            element_type=np.float32,
            shape=(1, 512),
            buffer_ptr=embedding.data_ptr(),
        )
        io_binding.bind_output(
            name="output",
            device_type=self.models_processor.device,
            device_id=0,
            element_type=np.float32,
            shape=(1, 3, 256, 256),
            buffer_ptr=output.data_ptr(),
        )

        # Run the model with lazy build handling
        self._run_model_with_lazy_build_check(model_name, model, io_binding)

    def _calc_emap_latent(self, source_embedding):
        """FS-PERF-05: shared emap-based latent computation extracted from
        calc_inswapper_latent and calc_swapper_latent_iss."""
        n_e = source_embedding / l2norm(source_embedding)
        latent = n_e.reshape((1, -1))
        latent = np.dot(latent, self.models_processor.emap)
        latent /= np.linalg.norm(latent)
        return latent

    def _ensure_emap(self):
        """Ensures emap is loaded; returns True if available, False otherwise."""
        if (
            not hasattr(self.models_processor, "emap")
            or not isinstance(self.models_processor.emap, np.ndarray)
            or self.models_processor.emap.size == 0
        ):
            # Custom provider: extract emap directly from InSwapperTorch if loaded
            if self.models_processor.provider_name == "Custom":
                torch_model = self._get_inswapper_torch()
                if torch_model is not None and torch_model.emap is not None:
                    self.models_processor.emap = torch_model.emap
                    return True

            self.models_processor.load_model("Inswapper128")

        return (
            hasattr(self.models_processor, "emap")
            and isinstance(self.models_processor.emap, np.ndarray)
            and self.models_processor.emap.size > 0
        )

    def _get_inswapper_torch(self):
        """Lazily load InSwapperTorch in GEMM/cuBLASLt mode (Custom provider)."""
        if self._inswapper_torch is not None:
            return self._inswapper_torch
        with self._inswapper_init_lock:
            if self._inswapper_torch is None:
                from custom_kernels.inswapper_128.inswapper_torch import InSwapperTorch

                onnx_path = self.models_processor.models_path["Inswapper128"]
                print("[InSwapperTorch] Loading model...")
                m = InSwapperTorch(onnx_path).cuda().eval()
                m.to_gemm_mode()
                print("[InSwapperTorch] GEMM mode enabled.")
                try:
                    m.to_cublaslt_mode()
                    print("[InSwapperTorch] cuBLASLt mode enabled.")
                except Exception as e:
                    print(
                        f"[InSwapperTorch] cuBLASLt unavailable ({e}); using torch.mm GEMM."
                    )
                self._inswapper_torch = m
        return self._inswapper_torch

    def _get_inswapper_runner_b1(self):
        """Lazily build a CUDA graph runner for B=1 single-tile inference."""
        if self._inswapper_runner_b1 is not None:
            return self._inswapper_runner_b1
        self.models_processor.show_build_dialog.emit(
            "Finalizing Custom Provider",
            "Capturing CUDA graph for Inswapper128 (Batch=1).\nThis only happens once and improves performance.",
        )
        try:
            with self._inswapper_init_lock:
                if self._inswapper_runner_b1 is not None:
                    return self._inswapper_runner_b1
                from custom_kernels.inswapper_128.inswapper_torch import (
                    build_cuda_graph_runner,
                )

                if self._inswapper_torch is None:
                    from custom_kernels.inswapper_128.inswapper_torch import (
                        InSwapperTorch,
                    )

                    onnx_path = self.models_processor.models_path["Inswapper128"]
                    print("[InSwapperTorch] Loading model...")
                    m = InSwapperTorch(onnx_path).cuda().eval()
                    m.to_gemm_mode()
                    print("[InSwapperTorch] GEMM mode enabled.")
                    try:
                        m.to_cublaslt_mode()
                        print("[InSwapperTorch] cuBLASLt mode enabled.")
                    except Exception as e:
                        print(
                            f"[InSwapperTorch] cuBLASLt unavailable ({e}); using torch.mm GEMM."
                        )
                    self._inswapper_torch = m
                model = self._inswapper_torch
                target_ex = torch.zeros(
                    1, 3, 128, 128, device="cuda", dtype=torch.float32
                )
                source_ex = torch.zeros(1, 512, device="cuda", dtype=torch.float32)
                print("[InSwapperTorch] Capturing CUDA graph (B=1)...")
                try:
                    with self.models_processor.cuda_graph_capture_lock:
                        self._inswapper_runner_b1 = build_cuda_graph_runner(
                            model, target_ex, source_ex
                        )
                    print("[InSwapperTorch] CUDA graph ready.")
                except Exception as e:
                    print(
                        f"[InSwapperTorch] CUDA graph failed ({e}); using eager model."
                    )
                    _m = model
                    self._inswapper_runner_b1 = lambda t, s: _m(t, s)
        finally:
            self.models_processor.hide_build_dialog.emit()
        return self._inswapper_runner_b1

    def calc_inswapper_latent(self, source_embedding):
        if not self._ensure_emap():
            print("[ERROR] Emap could not be loaded for latent calculation.")
            # FS-ROBUST-01: return None so callers can detect and handle the failure
            return None

        return self._calc_emap_latent(source_embedding)

    def run_inswapper(self, image, embedding, output):
        model_name = "Inswapper128"

        # ---- Custom provider: PyTorch-native inference with CUDA graph runner ----
        if self.models_processor.provider_name == "Custom":
            if not self._ensure_emap():
                self._load_swapper_model(model_name)

            runner = self._get_inswapper_runner_b1()
            with torch.no_grad():
                with self._inswapper_b1_lock:
                    result = runner(image, embedding)  # [1, 3, 128, 128] float32
                    output.copy_(result)
                    if self.models_processor.device == "cuda":
                        torch.cuda.current_stream().synchronize()
            return

        # ---- All other providers: ORT-based inference ----
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
            device_type=self.models_processor.device,
            device_id=0,
            element_type=np.float32,
            shape=(1, 3, 128, 128),
            buffer_ptr=image.data_ptr(),
        )
        io_binding.bind_input(
            name="source",
            device_type=self.models_processor.device,
            device_id=0,
            element_type=np.float32,
            shape=(1, 512),
            buffer_ptr=embedding.data_ptr(),
        )
        io_binding.bind_output(
            name="output",
            device_type=self.models_processor.device,
            device_id=0,
            element_type=np.float32,
            shape=(1, 3, 128, 128),
            buffer_ptr=output.data_ptr(),
        )

        # Run the model with lazy build handling
        self._run_model_with_lazy_build_check(model_name, model, io_binding)

    def run_inswapper_batched(
        self, images: torch.Tensor, embedding: torch.Tensor, output: torch.Tensor
    ) -> None:
        """Batched Custom-provider InSwapper inference for pixel-shift resolution mode."""
        model_name = "Inswapper128"
        if not self._ensure_emap():
            self._load_swapper_model(model_name)

        torch_model = self._get_inswapper_torch()
        with torch.no_grad():
            result = torch_model(images, embedding)  # [B, 3, 128, 128] float32
        output.copy_(result)

    def calc_swapper_latent_ghost(self, source_embedding):
        latent = source_embedding.reshape((1, -1))

        return latent

    def calc_swapper_latent_iss(self, source_embedding, version="A"):
        # FS-PERF-05: reuse shared _ensure_emap / _calc_emap_latent helpers
        if not self._ensure_emap():
            print("[ERROR] Emap could not be loaded for latent calculation.")
            n_e = source_embedding / l2norm(source_embedding)
            return n_e.reshape((1, -1))

        return self._calc_emap_latent(source_embedding)

    def run_iss_swapper(self, image, embedding, output, version="A"):
        model_name = f"InStyleSwapper256 Version {version}"
        model = self._load_swapper_model(model_name)
        if not model:
            print(f"[ERROR] {model_name} model not loaded.")
            return

        io_binding = model.io_binding()
        io_binding.bind_input(
            name="target",
            device_type=self.models_processor.device,
            device_id=0,
            element_type=np.float32,
            shape=(1, 3, 256, 256),
            buffer_ptr=image.data_ptr(),
        )
        io_binding.bind_input(
            name="source",
            device_type=self.models_processor.device,
            device_id=0,
            element_type=np.float32,
            shape=(1, 512),
            buffer_ptr=embedding.data_ptr(),
        )
        io_binding.bind_output(
            name="output",
            device_type=self.models_processor.device,
            device_id=0,
            element_type=np.float32,
            shape=(1, 3, 256, 256),
            buffer_ptr=output.data_ptr(),
        )

        # Run the model with lazy build handling
        self._run_model_with_lazy_build_check(model_name, model, io_binding)

    def calc_swapper_latent_simswap512(self, source_embedding):
        latent = source_embedding.reshape(1, -1)
        # latent /= np.linalg.norm(latent)
        latent = latent / np.linalg.norm(latent, axis=1, keepdims=True)
        return latent

    def run_swapper_simswap512(self, image, embedding, output):
        model_name = "SimSwap512"
        model = self._load_swapper_model(model_name)
        if not model:
            print("[ERROR] SimSwap512 model not loaded.")
            return

        io_binding = model.io_binding()
        io_binding.bind_input(
            name="input",
            device_type=self.models_processor.device,
            device_id=0,
            element_type=np.float32,
            shape=(1, 3, 512, 512),
            buffer_ptr=image.data_ptr(),
        )
        io_binding.bind_input(
            name="onnx::Gemm_1",
            device_type=self.models_processor.device,
            device_id=0,
            element_type=np.float32,
            shape=(1, 512),
            buffer_ptr=embedding.data_ptr(),
        )
        io_binding.bind_output(
            name="output",
            device_type=self.models_processor.device,
            device_id=0,
            element_type=np.float32,
            shape=(1, 3, 512, 512),
            buffer_ptr=output.data_ptr(),
        )

        # Run the model with lazy build handling
        self._run_model_with_lazy_build_check(model_name, model, io_binding)

    def run_swapper_ghostface(
        self, image, embedding, output, swapper_model="GhostFace-v2"
    ):
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
            device_type=self.models_processor.device,
            device_id=0,
            element_type=np.float32,
            shape=(1, 3, 256, 256),
            buffer_ptr=image.data_ptr(),
        )
        io_binding.bind_input(
            name="source",
            device_type=self.models_processor.device,
            device_id=0,
            element_type=np.float32,
            shape=(1, 512),
            buffer_ptr=embedding.data_ptr(),
        )
        io_binding.bind_output(
            name=output_name,
            device_type=self.models_processor.device,
            device_id=0,
            element_type=np.float32,
            shape=(1, 3, 256, 256),
            buffer_ptr=output.data_ptr(),
        )

        # Run the model with lazy build handling
        self._run_model_with_lazy_build_check(
            model_name, ghostfaceswap_model, io_binding
        )
