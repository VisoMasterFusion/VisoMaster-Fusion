import threading
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
        self._gfpgan_torch: Optional[object] = None  # GFPGANTorch (v1.4)
        self._gfpgan1024_torch: Optional[object] = None  # GFPGANTorch (1024)
        self._gfpgan_runner: Optional[object] = None  # CUDA graph runner (v1.4)
        self._gfpgan1024_runner: Optional[object] = None  # CUDA graph runner (1024)
        self._gpen_torch: Dict[int, Optional[object]] = {}  # GPENTorch per size
        self._gpen_runner: Dict[
            int, Optional[object]
        ] = {}  # CUDA graph runner per size
        self._ref_ldm_encoder_torch: Optional[object] = None  # RefLDMEncoderTorch
        self._ref_ldm_encoder_runner: Optional[object] = None  # CUDA graph runner
        self._ref_ldm_decoder_torch: Optional[object] = None  # RefLDMDecoderTorch
        self._ref_ldm_decoder_runner: Optional[object] = None  # CUDA graph runner
        self._ref_ldm_unet_torch: Optional[object] = None  # RefLDMUNetTorch
        self._ref_ldm_unet_runner: dict = {}  # {use_exclusive: UNetCUDAGraphRunner}
        self._codeformer_torch: Optional[object] = None  # CodeFormerTorch
        self._codeformer_runner: Optional[object] = None  # CUDA graph runner
        self._codeformer_runner_w: Optional[float] = (
            None  # fidelity weight baked into runner
        )
        self._restoreformer_torch: Optional[object] = None  # RestoreFormerPlusPlusTorch
        self._restoreformer_runner: Optional[object] = None  # CUDA graph runner
        self._custom_inference_lock = threading.Lock()
        self._runner_locks: Dict[int, threading.Lock] = {}
        self._custom_init_lock = threading.Lock()  # serialises Custom-kernel lazy inits
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

    def _get_runner_lock(self, runner):
        with self._custom_inference_lock:
            r_id = id(runner)
            if r_id not in self._runner_locks:
                self._runner_locks[r_id] = threading.Lock()
            return self._runner_locks[r_id]

    def unload_models(self):
        """Unloads the restorer models held in both slots and resets state."""
        if self.active_model_slot1:
            self.models_processor.unload_model(self.active_model_slot1)
            self.active_model_slot1 = None
        if self.active_model_slot2:
            self.models_processor.unload_model(self.active_model_slot2)
            self.active_model_slot2 = None
        with self._custom_init_lock:
            self._gfpgan_torch = None
            self._gfpgan1024_torch = None
            self._gfpgan_runner = None
            self._gfpgan1024_runner = None
            self._gpen_torch = {}
            self._gpen_runner = {}
            self._ref_ldm_encoder_torch = None
            self._ref_ldm_encoder_runner = None
            self._ref_ldm_decoder_torch = None
            self._ref_ldm_decoder_runner = None
            self._ref_ldm_unet_torch = None
            self._ref_ldm_unet_runner = {}
            self._codeformer_torch = None
            self._codeformer_runner = None
            self._codeformer_runner_w = None
            self._restoreformer_torch = None
            self._restoreformer_runner = None

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

    def _get_gfpgan_runner(self, is_1024: bool = False):
        """Lazily load GFPGANTorch + CUDA graph runner (FP16 PyTorch kernel)."""
        model_attr = "_gfpgan1024_torch" if is_1024 else "_gfpgan_torch"
        runner_attr = "_gfpgan1024_runner" if is_1024 else "_gfpgan_runner"
        label = "GFPGAN-1024" if is_1024 else "GFPGAN v1.4"

        runner = getattr(self, runner_attr)
        if runner is not None:
            return runner
        self.models_processor.show_build_dialog.emit(
            "Finalizing Custom Provider",
            f"Compiling & capturing CUDA graph for {label}…\nFirst run only — future sessions load instantly from cache.",
        )
        try:
            with self._custom_init_lock:
                runner = getattr(self, runner_attr)
                if runner is not None:
                    return runner  # built by another thread while we waited
                model = getattr(self, model_attr)
                if model is None:
                    try:
                        import pathlib
                        from custom_kernels.gfpgan_v1_4.gfpgan_torch import GFPGANTorch

                        onnx_name = "gfpgan-1024.onnx" if is_1024 else "GFPGANv1.4.onnx"
                        onnx_path = str(
                            pathlib.Path(__file__).parent.parent.parent
                            / "model_assets"
                            / onnx_name
                        )
                        print(f"[GFPGANTorch] Loading PyTorch {label} model...")
                        m = GFPGANTorch.from_onnx(onnx_path).cuda().eval()
                        setattr(self, model_attr, m)
                        model = m
                    except Exception as e:
                        print(
                            f"[GFPGANTorch] Failed to load {label} custom kernel: {e}"
                        )
                        return None
                # Build CUDA graph runner
                try:
                    from custom_kernels.gfpgan_v1_4.gfpgan_torch import (
                        build_cuda_graph_runner,
                    )

                    with self.models_processor.cuda_graph_capture_lock:
                        r = build_cuda_graph_runner(
                            model, inp_shape=(1, 3, 512, 512), torch_compile=True
                        )
                    setattr(self, runner_attr, r)
                    runner = r
                except Exception as e:
                    print(
                        f"[GFPGANTorch] CUDA graph build failed for {label}, "
                        f"using direct inference: {e}"
                    )
                    setattr(self, runner_attr, model)
                    runner = model
        finally:
            self.models_processor.hide_build_dialog.emit()
        return runner

    def _get_gpen_runner(self, size: int):
        """Lazily load GPENTorch + CUDA graph runner for GPEN-BFR-{size}."""
        if size in self._gpen_runner:
            return self._gpen_runner.get(size)
        self.models_processor.show_build_dialog.emit(
            "Finalizing Custom Provider",
            f"Capturing CUDA graph for GPEN-BFR-{size}…",
        )
        try:
            with self._custom_init_lock:
                if size in self._gpen_runner:
                    return self._gpen_runner.get(size)
                model = self._gpen_torch.get(size)
                if model is None:
                    try:
                        from custom_kernels.gpen_bfr.gpen_torch import GPENTorch
                        import pathlib

                        onnx_path = str(
                            pathlib.Path(__file__).parent.parent.parent
                            / "model_assets"
                            / f"GPEN-BFR-{size}.onnx"
                        )
                        print(f"[GPENTorch] Loading GPEN-BFR-{size} model...")
                        model = (
                            GPENTorch.from_onnx(onnx_path, compute_dtype=torch.float16)
                            .cuda()
                            .eval()
                        )
                        self._gpen_torch[size] = model
                    except Exception as e:
                        print(f"[GPENTorch] Failed to load GPEN-BFR-{size}: {e}")
                        self._gpen_torch[size] = None
                        self._gpen_runner[size] = None
                        return None
                try:
                    from custom_kernels.gpen_bfr.gpen_torch import (
                        build_cuda_graph_runner,
                    )

                    inp_hw = model.in_size  # type: ignore[attr-defined]
                    with self.models_processor.cuda_graph_capture_lock:
                        runner = build_cuda_graph_runner(
                            model,  # type: ignore[arg-type]
                            inp_shape=(1, 3, inp_hw, inp_hw),
                        )
                    self._gpen_runner[size] = runner
                except Exception as e:
                    print(
                        f"[GPENTorch] CUDA graph build failed for GPEN-{size}, using direct inference: {e}"
                    )
                    self._gpen_runner[size] = model  # fallback: direct model call
        finally:
            self.models_processor.hide_build_dialog.emit()
        return self._gpen_runner.get(size)

    def _run_gpen_custom(self, size: int, image: torch.Tensor, output: torch.Tensor):
        """Run GPEN via Custom provider (GPENTorch + CUDA graph)."""
        runner = self._get_gpen_runner(size)
        if runner is None:
            return False
        with torch.no_grad():
            with self._custom_inference_lock:
                result = runner(image)
        output.copy_(result)
        return True

    def _get_ref_ldm_encoder_runner(self):
        """Lazily load RefLDMEncoderTorch + CUDA graph runner."""
        if self._ref_ldm_encoder_runner is not None:
            return self._ref_ldm_encoder_runner
        self.models_processor.show_build_dialog.emit(
            "Finalizing Custom Provider",
            "Compiling & capturing CUDA graph for RefLDM VAE Encoder…\nFirst run only — future sessions load instantly from cache.",
        )
        try:
            with self._custom_init_lock:
                if self._ref_ldm_encoder_runner is not None:
                    return self._ref_ldm_encoder_runner
                if self._ref_ldm_encoder_torch is None:
                    try:
                        import pathlib
                        from custom_kernels.ref_ldm.ref_ldm_torch import (
                            RefLDMEncoderTorch,
                        )

                        onnx_path = str(
                            pathlib.Path(__file__).parent.parent.parent
                            / "model_assets"
                            / "ref_ldm_vae_encoder.onnx"
                        )
                        print("[RefLDMTorch] Loading VAE encoder...")
                        m = RefLDMEncoderTorch.from_onnx(onnx_path).cuda().eval()
                        self._ref_ldm_encoder_torch = m
                    except Exception as e:
                        print(f"[RefLDMTorch] Failed to load VAE encoder: {e}")
                        return None
                try:
                    from custom_kernels.ref_ldm.ref_ldm_torch import (
                        build_cuda_graph_runner,
                    )

                    with self.models_processor.cuda_graph_capture_lock:
                        runner = build_cuda_graph_runner(
                            self._ref_ldm_encoder_torch,
                            inp_shape=(1, 3, 512, 512),
                            torch_compile=True,
                        )
                    self._ref_ldm_encoder_runner = runner
                except Exception as e:
                    print(
                        f"[RefLDMTorch] CUDA graph build failed for encoder, using direct inference: {e}"
                    )
                    self._ref_ldm_encoder_runner = self._ref_ldm_encoder_torch
        finally:
            self.models_processor.hide_build_dialog.emit()
        return self._ref_ldm_encoder_runner

    def _get_ref_ldm_decoder_runner(self):
        """Lazily load RefLDMDecoderTorch + CUDA graph runner."""
        if self._ref_ldm_decoder_runner is not None:
            return self._ref_ldm_decoder_runner
        self.models_processor.show_build_dialog.emit(
            "Finalizing Custom Provider",
            "Compiling & capturing CUDA graph for RefLDM VAE Decoder…\nFirst run only — future sessions load instantly from cache.",
        )
        try:
            with self._custom_init_lock:
                if self._ref_ldm_decoder_runner is not None:
                    return self._ref_ldm_decoder_runner
                if self._ref_ldm_decoder_torch is None:
                    try:
                        import pathlib
                        from custom_kernels.ref_ldm.ref_ldm_torch import (
                            RefLDMDecoderTorch,
                        )

                        onnx_path = str(
                            pathlib.Path(__file__).parent.parent.parent
                            / "model_assets"
                            / "ref_ldm_vae_decoder.onnx"
                        )
                        print("[RefLDMTorch] Loading VAE decoder...")
                        m = RefLDMDecoderTorch.from_onnx(onnx_path).cuda().eval()
                        self._ref_ldm_decoder_torch = m
                    except Exception as e:
                        print(f"[RefLDMTorch] Failed to load VAE decoder: {e}")
                        return None
                try:
                    from custom_kernels.ref_ldm.ref_ldm_torch import (
                        build_cuda_graph_runner,
                    )

                    with self.models_processor.cuda_graph_capture_lock:
                        runner = build_cuda_graph_runner(
                            self._ref_ldm_decoder_torch,
                            inp_shape=(1, 8, 64, 64),
                            torch_compile=True,
                        )
                    self._ref_ldm_decoder_runner = runner
                except Exception as e:
                    print(
                        f"[RefLDMTorch] CUDA graph build failed for decoder, using direct inference: {e}"
                    )
                    self._ref_ldm_decoder_runner = self._ref_ldm_decoder_torch
        finally:
            self.models_processor.hide_build_dialog.emit()
        return self._ref_ldm_decoder_runner

    def _get_ref_ldm_unet_torch(self):
        """Lazily load RefLDMUNetTorch."""
        if self._ref_ldm_unet_torch is not None:
            return self._ref_ldm_unet_torch
        with self._custom_init_lock:
            if self._ref_ldm_unet_torch is None:
                try:
                    import pathlib
                    from custom_kernels.ref_ldm.ref_ldm_torch import RefLDMUNetTorch

                    onnx_path = str(
                        pathlib.Path(__file__).parent.parent.parent
                        / "model_assets"
                        / "ref_ldm_unet_external_kv.onnx"
                    )
                    print("[RefLDMTorch] Loading UNet...")
                    m = RefLDMUNetTorch.from_onnx(onnx_path).cuda().eval()
                    self._ref_ldm_unet_torch = m
                except Exception as e:
                    print(f"[RefLDMTorch] Failed to load UNet: {e}")
        return self._ref_ldm_unet_torch

    def _get_restoreformer_runner(self):
        """Lazily load RestoreFormerPlusPlusTorch + CUDA graph runner."""
        if self._restoreformer_runner is not None:
            return self._restoreformer_runner
        self.models_processor.show_build_dialog.emit(
            "Finalizing Custom Provider",
            "Compiling & capturing CUDA graph for RestoreFormer++…\nFirst run only — future sessions load instantly from cache.",
        )
        try:
            with self._custom_init_lock:
                if self._restoreformer_runner is not None:
                    return self._restoreformer_runner
                if self._restoreformer_torch is None:
                    try:
                        import pathlib
                        from custom_kernels.restoreformer.restoreformer_torch import (
                            RestoreFormerPlusPlusTorch,
                        )

                        onnx_path = str(
                            pathlib.Path(__file__).parent.parent.parent
                            / "model_assets"
                            / "RestoreFormerPlusPlus.fp16.onnx"
                        )
                        print("[RFP++Torch] Loading RestoreFormerPlusPlus model...")
                        m = (
                            RestoreFormerPlusPlusTorch.from_onnx(onnx_path)
                            .to(self.models_processor.device)
                            .eval()
                        )
                        self._restoreformer_torch = m
                    except Exception as e:
                        print(f"[RFP++Torch] Failed to load model: {e}")
                        return None
                try:
                    from custom_kernels.restoreformer.restoreformer_torch import (
                        build_cuda_graph_runner,
                    )

                    with self.models_processor.cuda_graph_capture_lock:
                        runner = build_cuda_graph_runner(
                            self._restoreformer_torch,
                            inp_shape=(1, 3, 512, 512),
                            torch_compile=True,
                        )
                    self._restoreformer_runner = runner
                except Exception as e:
                    print(
                        f"[RFP++Torch] CUDA graph build failed, using direct inference: {e}"
                    )
                    self._restoreformer_runner = self._restoreformer_torch
        finally:
            self.models_processor.hide_build_dialog.emit()
        return self._restoreformer_runner

    def _get_codeformer_torch(self):
        """Lazily load CodeFormerTorch (FP16 PyTorch kernel)."""
        if self._codeformer_torch is not None:
            return self._codeformer_torch
        self.models_processor.show_build_dialog.emit(
            "Finalizing Custom Provider",
            "Loading CodeFormer model (Custom provider)…\nThis only happens once.",
        )
        try:
            with self._custom_init_lock:
                if self._codeformer_torch is not None:
                    return self._codeformer_torch
                try:
                    import pathlib
                    from custom_kernels.codeformer.codeformer_torch import (
                        CodeFormerTorch,
                    )

                    onnx_path = str(
                        pathlib.Path(__file__).parent.parent.parent
                        / "model_assets"
                        / "codeformer_fp16.onnx"
                    )
                    self._codeformer_torch = (
                        CodeFormerTorch.from_onnx(onnx_path)
                        .to(self.models_processor.device)
                        .eval()
                    )
                except Exception as e:
                    print(f"[Custom] CodeFormerTorch load failed: {e}")
        finally:
            self.models_processor.hide_build_dialog.emit()
        return self._codeformer_torch

    def _get_codeformer_runner(self, model, fidelity_weight: float):
        """
        Return a CUDA-graph runner for the given fidelity_weight.

        The fidelity weight is baked into the graph at capture time.
        If the weight changes, the old runner is invalidated and a new one is
        built on the next call (one direct-inference frame in between).
        """
        w = round(float(fidelity_weight), 3)  # quantise to 3dp to reduce rebuilds
        if (
            self._codeformer_runner is not None
            and self._codeformer_runner_w is not None
            and abs(self._codeformer_runner_w - w) < 0.005
        ):
            return self._codeformer_runner

        # Invalidate stale runner — caller will use direct inference this frame.
        self._codeformer_runner = None
        self._codeformer_runner_w = None

        self.models_processor.show_build_dialog.emit(
            "Finalizing Custom Provider",
            f"Compiling & capturing CUDA graph for CodeFormer (w={w:.2f})…\nFirst run only — future sessions load instantly from cache.",
        )
        try:
            with self._custom_init_lock:
                # Re-check under lock in case another thread just built it.
                if (
                    self._codeformer_runner is not None
                    and self._codeformer_runner_w is not None
                    and abs(self._codeformer_runner_w - w) < 0.005
                ):
                    return self._codeformer_runner
                try:
                    from custom_kernels.codeformer.codeformer_torch import (
                        build_cuda_graph_runner,
                    )

                    with self.models_processor.cuda_graph_capture_lock:
                        runner = build_cuda_graph_runner(
                            model,
                            inp_shape=(1, 3, 512, 512),
                            fidelity_weight=w,
                            torch_compile=True,
                        )
                    self._codeformer_runner = runner
                    self._codeformer_runner_w = w
                except Exception as e:
                    print(f"[Custom] CodeFormer CUDA graph build failed: {e}")
        finally:
            self.models_processor.hide_build_dialog.emit()
        return self._codeformer_runner

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
                # by synchronizing with a placeholder vector.
                self.models_processor.syncvec.cpu()

            ort_session.run_with_iobinding(io_binding)

        finally:
            if is_lazy_build:
                self.models_processor.hide_build_dialog.emit()

    def apply_facerestorer(
        self,
        swapped_face_upscaled,
        restorer_det_type,
        restorer_type,
        restorer_blend,
        fidelity_weight,
        detect_score,
        target_kps=None,
        slot_id: int = 1,
    ):
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

        # Now safe to use inplace normalization as we definitely own the 'temp' memory footprint
        temp = v2.functional.normalize(
            temp, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True
        )

        if restorer_type == "GPEN-256":
            temp = v2.functional.resize(temp, [256, 256], antialias=False)

        temp = torch.unsqueeze(temp, 0).contiguous()

        # Bindings
        # FR-ROBUST-04: removed default 512x512 pre-allocation; each branch allocates at correct size
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
            temp = v2.functional.resize(temp, [1024, 1024], antialias=False)
            outpred = torch.empty(
                (1, 3, 1024, 1024),
                dtype=torch.float32,
                device=self.models_processor.device,
            ).contiguous()
            self.run_GPEN_1024(temp, outpred)

        elif restorer_type == "GPEN-2048":
            temp = v2.functional.resize(temp, [2048, 2048], antialias=False)
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

        if restorer_type in ["GPEN-256", "GPEN-1024", "GPEN-2048", "GFPGAN-1024"]:
            outpred = v2.functional.resize(outpred, [512, 512], antialias=True)

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

        return outpred

    def run_vae_encoder(
        self, image_input_tensor: torch.Tensor, output_latent_tensor: torch.Tensor
    ):
        """
        Runs the VAE encoder model.
        image_input_tensor: Batch x 3 x Height x Width, float32, normalized to [-1, 1]
        output_latent_tensor: Placeholder for Batch x 8 x LatentH x LatentW, float32
        """
        if self.models_processor.provider_name == "Custom":
            runner = self._get_ref_ldm_encoder_runner()
            if runner is not None:
                with torch.no_grad():
                    with self._get_runner_lock(runner):
                        result = runner(image_input_tensor)
                output_latent_tensor.copy_(result)
                return
            # Custom runner failed to build — skip silently (denoiser not applied)
            print(
                "[WARN] run_vae_encoder: Custom runner unavailable, skipping encoder."
            )
            return

        model_name = "RefLDMVAEEncoder"
        # FR-BUG-04: use .get() to avoid KeyError when model is not yet loaded
        ort_session = self.models_processor.models.get(model_name)
        if ort_session is None:
            # Lazy reload in case clear_gpu_memory() cleared the session after a
            # provider switch (e.g. TRT→Custom→TRT without a UI toggle change).
            self.models_processor.ensure_denoiser_models_loaded()
            ort_session = self.models_processor.models.get(model_name)
        if ort_session is None:
            error_msg = f"[ERROR] VAE Encoder model '{model_name}' not loaded when run_vae_encoder was called. This model should be loaded by ModelsProcessor.ensure_denoiser_models_loaded()."
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
        if self.models_processor.provider_name == "Custom":
            runner = self._get_ref_ldm_decoder_runner()
            if runner is not None:
                with torch.no_grad():
                    with self._get_runner_lock(runner):
                        result = runner(latent_input_tensor)
                output_image_tensor.copy_(result)
                return
            # Custom runner failed to build — skip silently (denoiser not applied)
            print(
                "[WARN] run_vae_decoder: Custom runner unavailable, skipping decoder."
            )
            return

        model_name = "RefLDMVAEDecoder"
        # FR-BUG-04: use .get() to avoid KeyError when model is not yet loaded
        ort_session = self.models_processor.models.get(model_name)
        if ort_session is None:
            # Lazy reload in case clear_gpu_memory() cleared the session after a
            # provider switch (e.g. TRT→Custom→TRT without a UI toggle change).
            self.models_processor.ensure_denoiser_models_loaded()
            ort_session = self.models_processor.models.get(model_name)
        if ort_session is None:
            error_msg = f"[ERROR] VAE Decoder model '{model_name}' not loaded when run_vae_decoder was called. This model should be loaded by ModelsProcessor.ensure_denoiser_models_loaded()."
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
        if self.models_processor.provider_name == "Custom":
            unet = self._get_ref_ldm_unet_torch()
            if unet is not None:
                use_exclusive = bool(
                    use_reference_exclusive_path_globally_tensor.item()
                )
                # Lazily build a UNetCUDAGraphRunner on the first call that
                # has a real kv_map (K/V shapes are fixed by architecture so
                # the template only needs to be captured once per use_exclusive value).
                runner = self._ref_ldm_unet_runner.get(use_exclusive)
                if runner is None and kv_tensor_map is not None:
                    try:
                        from custom_kernels.ref_ldm.ref_ldm_torch import (
                            build_unet_cuda_graph_runner,
                        )

                        self.models_processor.show_build_dialog.emit(
                            "Finalizing Custom Provider",
                            f"Compiling & capturing CUDA graph for RefLDM UNet "
                            f"(use_exclusive={use_exclusive})…\n"
                            f"First run only — future sessions load instantly from cache.",
                        )
                        try:
                            with self.models_processor.cuda_graph_capture_lock:
                                runner = build_unet_cuda_graph_runner(
                                    unet,
                                    x_shape=tuple(x_noisy_plus_lq_latent.shape),
                                    ts_example=timesteps_tensor.clone(),
                                    kv_map_template=kv_tensor_map,
                                    use_exclusive=use_exclusive,
                                    torch_compile=False,
                                )
                            self._ref_ldm_unet_runner[use_exclusive] = runner
                            print(
                                f"[RefLDMTorch] UNet CUDA graph captured "
                                f"(use_exclusive={use_exclusive})."
                            )
                        finally:
                            self.models_processor.hide_build_dialog.emit()
                    except Exception as e:
                        print(
                            f"[RefLDMTorch] UNet CUDA graph build failed "
                            f"(use_exclusive={use_exclusive}): {e} — "
                            "falling back to eager mode."
                        )
                        self._ref_ldm_unet_runner[use_exclusive] = None

                if runner is not None:
                    with torch.no_grad():
                        with self._get_runner_lock(runner):
                            result = runner(
                                x_noisy_plus_lq_latent,
                                timesteps_tensor,
                                kv_tensor_map,
                                use_exclusive,
                            )
                    output_unet_tensor.copy_(result)
                    return
                # Fallback: eager mode (no kv_map yet, or CUDA graph build failed)
                with torch.no_grad():
                    with self._get_runner_lock(unet):
                        result = unet(
                            x_noisy_plus_lq_latent,
                            timesteps_tensor,
                            kv_map=kv_tensor_map,
                            use_exclusive=use_exclusive,
                        )
                output_unet_tensor.copy_(result)
                return
            # Custom runner failed to build — skip silently (denoiser not applied)
            print("[WARN] run_ref_ldm_unet: Custom runner unavailable, skipping unet.")
            return

        model_name = self.models_processor.main_window.fixed_unet_model_name
        ort_session = self.models_processor.models.get(model_name)

        if not ort_session:
            # Enhanced error reporting
            error_messages = [
                f"[ERROR] UNet model '{model_name}' not loaded when run_ref_ldm_unet was called.",
                "  This model should be loaded by ModelsProcessor.apply_denoiser_unet or a similar setup routine.",
            ]
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

        # IMPORTANT: Keep references to temporary zero tensors to prevent GC
        keep_alive_tensors: list = []
        # FS-MEM-01: also keep actual KV tensors alive to prevent premature GC
        keep_alive_tensors.extend(actual_kv_tensors_for_binding.values())

        for onnx_kv_name, expected_shape in onnx_kv_input_names_to_shape.items():
            tensor_to_bind = actual_kv_tensors_for_binding.get(onnx_kv_name)

            if tensor_to_bind is None:
                # Create a zero tensor for missing K/V inputs (e.g., unconditional pass)
                tensor_to_bind = torch.zeros(
                    expected_shape, dtype=torch.float32, device=bind_device_type
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

        if self.models_processor.provider_name == "Custom":
            runner = self._get_gfpgan_runner(is_1024=False)
            if runner is not None:
                with torch.no_grad():
                    with self._get_runner_lock(runner):
                        result = runner(image)
                output.copy_(result)
                return

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

        # Run the model with lazy build handling
        self._run_model_with_lazy_build_check(model_name, ort_session, io_binding)

    def run_GFPGAN1024(self, image, output):
        model_name = "GFPGAN1024"

        if self.models_processor.provider_name == "Custom":
            runner = self._get_gfpgan_runner(is_1024=True)
            if runner is not None:
                with torch.no_grad():
                    with self._get_runner_lock(runner):
                        result = runner(image)
                output.copy_(result)
                return

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

        # Run the model with lazy build handling
        self._run_model_with_lazy_build_check(model_name, ort_session, io_binding)

    def run_GPEN_256(self, image, output):
        model_name = "GPENBFR256"

        if self.models_processor.provider_name == "Custom":
            if self._run_gpen_custom(256, image, output):
                return

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

        # Run the model with lazy build handling
        self._run_model_with_lazy_build_check(model_name, ort_session, io_binding)

    def run_GPEN_512(self, image, output):
        model_name = "GPENBFR512"

        if self.models_processor.provider_name == "Custom":
            if self._run_gpen_custom(512, image, output):
                return

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

        # Run the model with lazy build handling
        self._run_model_with_lazy_build_check(model_name, ort_session, io_binding)

    def run_GPEN_1024(self, image, output):
        model_name = "GPENBFR1024"

        if self.models_processor.provider_name == "Custom":
            if self._run_gpen_custom(1024, image, output):
                return

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

        # Run the model with lazy build handling
        self._run_model_with_lazy_build_check(model_name, ort_session, io_binding)

    def run_GPEN_2048(self, image, output):
        model_name = "GPENBFR2048"

        if self.models_processor.provider_name == "Custom":
            if self._run_gpen_custom(2048, image, output):
                return

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

        # Run the model with lazy build handling
        self._run_model_with_lazy_build_check(model_name, ort_session, io_binding)

    def run_codeformer(self, image, output, fidelity_weight_value=0.9):
        if self.models_processor.provider_name == "Custom":
            model = self._get_codeformer_torch()
            if model is not None:
                w = float(fidelity_weight_value)
                runner = self._get_codeformer_runner(model, w)
                with torch.no_grad():
                    with self._get_runner_lock(model):
                        if runner is not None:
                            result = runner(image)
                        else:
                            # Runner not yet ready (first frame after weight change):
                            # fall back to direct FP16 inference.
                            result = model(image, fidelity_weight=w)
                output.copy_(result)
                return

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

        # Run the model with lazy build handling
        self._run_model_with_lazy_build_check(model_name, ort_session, io_binding)

    def run_RestoreFormerPlusPlus(self, image, output):
        if self.models_processor.provider_name == "Custom":
            runner = self._get_restoreformer_runner()
            if runner is not None:
                with torch.no_grad():
                    with self._get_runner_lock(runner):
                        result = runner(image)
                output.copy_(result)
                return

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

        # Run the model with lazy build handling
        self._run_model_with_lazy_build_check(model_name, ort_session, io_binding)
