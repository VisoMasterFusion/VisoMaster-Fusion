"""PerformRecast — expression-only "Recast" backend for the Face Expression Restorer.

PerformRecast (CVPR 2026) transfers the *expression* of a driving face onto a
source face while preserving the source's identity and head pose. Inside
VisoMaster the driving face is the original pre-swap face and the source is the
swapped face, mirroring the role of the LivePortrait-based expression restorer.

It uses four exported ONNX sub-networks (see ``models_data.py``):

    F  PerformRecastAppearanceFeatureExtractor  (B,3,512,512)[0,1] -> feature_3d (B,32,16,64,64)
    M  PerformRecastMotionExtractor             (B,3,256,256)[0,1] -> pitch,yaw,roll,t,exp,scale,kp
    W  PerformRecastWarpingModule               feature_3d,kp_driving,kp_source -> occlusion_map,deformation,out (B,256,64,64)
    G  PerformRecastSpadeGenerator              feature (B,256,64,64) -> out (B,3,512,512)[0,1]

The model uses 49 implicit keypoints (NUM_KP = 49) — not LivePortrait's 21 — and
a head-pose binning convention of ``softmax * 3 - 99`` (HopeNet), which differs
from LivePortrait's ``- 97.5``. The keypoint transform and the two composition
modes are ported verbatim from the reference ``src/pipeline.py`` /
``scripts/onnx_inference.py`` so behaviour matches the upstream model.

W and G have no fp16 kernel (5-D grid_sample / SPADE), so they always run fp32;
all graph I/O is float32 regardless of internal precision.
"""

from typing import TYPE_CHECKING, Dict

import numpy as np
import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from app.processors.models_processor import ModelsProcessor

# Implicit keypoint count of the PerformRecast motion model
# (performrecast_models.yaml -> common_params.num_kp).
NUM_KP = 49

# ONNX model registry keys for the four sub-networks.
APPEARANCE_MODEL = "PerformRecastAppearanceFeatureExtractor"
MOTION_MODEL = "PerformRecastMotionExtractor"
WARPING_MODEL = "PerformRecastWarpingModule"
SPADE_MODEL = "PerformRecastSpadeGenerator"

# Mapping of the user-facing mode label -> upstream inference_mode integer.
MODE_REPLACEMENT = "Replacement"  # inference_mode 1
MODE_ENHANCEMENT = "Enhancement"  # inference_mode 2
MODE_RELATIVE = "Relative"  # Relative mode


class PerformRecast:
    """onnxruntime-backed runner for the four PerformRecast sub-networks.

    Mirrors the structure of :class:`app.processors.face_editors.FaceEditors`:
    models are lazily loaded through ``models_processor.load_model`` on first
    use and can be released together via :meth:`unload_models`.
    """

    # All four ONNX models, grouped so the UI can load/unload them together.
    model_group = [APPEARANCE_MODEL, MOTION_MODEL, WARPING_MODEL, SPADE_MODEL]

    def __init__(self, models_processor: "ModelsProcessor"):
        self.models_processor = models_processor

    # ------------------------------------------------------------------ #
    # Low-level ONNX runner
    # ------------------------------------------------------------------ #
    def _session(self, model_name: str):
        """Return a loaded InferenceSession for ``model_name`` (lazy load)."""
        session = self.models_processor.models.get(model_name)
        if session is None:
            session = self.models_processor.load_model(model_name)
            self.models_processor.models[model_name] = session
        if session is None:
            raise RuntimeError(f"Model {model_name} could not be loaded")
        return session

    def _infer(
        self,
        model_name: str,
        inputs: Dict[str, "torch.Tensor"],
        output_specs: Dict[str, tuple],
    ) -> Dict[str, "torch.Tensor"]:
        """Run a model and return the requested outputs as torch tensors.

        On CUDA this uses zero-copy onnxruntime IO binding (matching
        ``FaceEditors._run_onnx_io_binding``) so the per-frame inputs/outputs
        stay on the GPU instead of round-tripping through host numpy buffers —
        the original ``session.run(None, feeds)`` path forced a full CPU<->GPU
        copy of every tensor each call (notably the 32x16x64x64 feature volume
        and the 256x64x64 warp output), which is why Recast was slower than the
        IO-bound LivePortrait expression restorer. On CPU (and under the mocked
        sessions used by the unit tests) it falls back to ``session.run``.

        Args:
            model_name: registry key of the sub-network.
            inputs: feed name -> torch tensor (any device/dtype; coerced to
                contiguous float32 on the models_processor device).
            output_specs: output name -> concrete shape for the outputs we
                actually consume. Other graph outputs are still produced but
                left for onnxruntime to allocate (we never read them).

        TensorRT "lazy build" models compile their engine on the first real
        inference (not during the isolated probe). We surface the build dialog
        around that first run so the UI does not appear frozen.
        """
        session = self._session(model_name)
        mp = self.models_processor

        is_lazy_build = False
        check_pending = getattr(mp, "check_and_clear_pending_build", None)
        if callable(check_pending):
            is_lazy_build = check_pending(model_name)
        if is_lazy_build:
            mp.show_build_dialog.emit(
                "Finalizing TensorRT Build",
                f"Performing first-run inference for:\n{model_name}\n\n"
                f"This may take several minutes.",
            )
        try:
            if getattr(mp, "device_type", "cpu") == "cuda":
                return self._infer_iobinding(session, inputs, output_specs)
            return self._infer_numpy(session, inputs, output_specs)
        finally:
            if is_lazy_build:
                mp.hide_build_dialog.emit()

    def _infer_iobinding(
        self, session, inputs: Dict[str, "torch.Tensor"], output_specs: Dict[str, tuple]
    ) -> Dict[str, "torch.Tensor"]:
        """Zero-copy CUDA inference path (see :meth:`_infer`)."""
        mp = self.models_processor
        io_binding = session.io_binding()

        # Keep references to the coerced input tensors alive until run() — the
        # binding only stores raw data pointers.
        kept = []
        for name, tensor in inputs.items():
            t = tensor.detach().to(mp.device, torch.float32).contiguous()
            kept.append(t)
            io_binding.bind_input(
                name=name,
                device_type=mp.device_type,
                device_id=mp.binding_device_id,
                element_type=np.float32,
                shape=tuple(t.shape),
                buffer_ptr=t.data_ptr(),
            )

        out_buffers: Dict[str, "torch.Tensor"] = {}
        for o in session.get_outputs():
            name = o.name
            if name in output_specs:
                buf = torch.empty(
                    output_specs[name], dtype=torch.float32, device=mp.device
                ).contiguous()
                out_buffers[name] = buf
                io_binding.bind_output(
                    name=name,
                    device_type=mp.device_type,
                    device_id=mp.binding_device_id,
                    element_type=np.float32,
                    shape=tuple(buf.shape),
                    buffer_ptr=buf.data_ptr(),
                )
            else:
                # Output we never read — let ORT allocate it on the device.
                io_binding.bind_output(
                    name=name,
                    device_type=mp.device_type,
                    device_id=mp.binding_device_id,
                )

        # Ensure PyTorch has finished writing the input buffers before ORT reads
        # from the bound pointers.
        if mp.device_type == "cuda":
            torch.cuda.current_stream().synchronize()
        session.run_with_iobinding(io_binding)
        return out_buffers

    def _infer_numpy(
        self, session, inputs: Dict[str, "torch.Tensor"], output_specs: Dict[str, tuple]
    ) -> Dict[str, "torch.Tensor"]:
        """Host (CPU / mocked-session) fallback path (see :meth:`_infer`)."""
        device = self.models_processor.device
        feeds = {}
        for name, tensor in inputs.items():
            if isinstance(tensor, torch.Tensor):
                arr = tensor.detach().cpu().numpy()
            else:
                arr = tensor
            feeds[name] = np.ascontiguousarray(arr, dtype=np.float32)

        names = [o.name for o in session.get_outputs()]
        outs = dict(zip(names, session.run(None, feeds)))
        return {
            name: torch.from_numpy(np.asarray(outs[name])).to(device)
            for name in output_specs
        }

    def prewarm(self):
        """Load all four sub-networks up front (no inference).

        Loading each model triggers its TensorRT engine build / cache probe (and
        the associated "Building TensorRT Cache" dialog) inside
        ``models_processor.load_model``. Doing this once, eagerly and in a fixed
        order at the start of a Recast frame — instead of lazily interleaved with
        inference — guarantees the build dialog is shown for *every* PerformRecast
        model so the user knows what the app is doing during the (multi-minute)
        engine builds. It is a cheap dict lookup once the models are loaded.
        """
        for model_name in self.model_group:
            self._session(model_name)

    def unload_models(self):
        """Release all four PerformRecast models from VRAM."""
        for model_name in self.model_group:
            self.models_processor.unload_model(model_name)

    # ------------------------------------------------------------------ #
    # Model-level API (mirrors scripts/onnx_inference.py)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _to_input(img: torch.Tensor) -> np.ndarray:
        """(C,H,W) uint8/float [0,255] -> (1,3,H,W) float32 [0,1] numpy."""
        x = img.detach().to(torch.float32) / 255.0
        x = torch.clamp(x, 0.0, 1.0)
        if x.dim() == 3:
            x = x.unsqueeze(0)
        return x.contiguous().cpu().numpy()

    def _to_input_t(self, img: torch.Tensor) -> torch.Tensor:
        """(C,H,W) uint8/float [0,255] -> (1,3,H,W) float32 [0,1] on device."""
        x = img.detach().to(self.models_processor.device, torch.float32) / 255.0
        x = torch.clamp(x, 0.0, 1.0)
        if x.dim() == 3:
            x = x.unsqueeze(0)
        return x.contiguous()

    def extract_appearance(self, img512: torch.Tensor) -> torch.Tensor:
        """F: source crop (C,512,512)[0,255] -> feature_3d (1,32,16,64,64)."""
        out = self._infer(
            APPEARANCE_MODEL,
            {"source_image": self._to_input_t(img512)},
            {"feature_3d": (1, 32, 16, 64, 64)},
        )
        return out["feature_3d"]

    def motion(self, img256: torch.Tensor) -> Dict[str, torch.Tensor]:
        """M: crop (C,256,256)[0,255] -> raw motion dict as torch tensors.

        Keys: pitch, yaw, roll (1,66); t (1,3); scale (1,3);
        exp, kp reshaped to (1, NUM_KP, 3).
        """
        kp_info = self._infer(
            MOTION_MODEL,
            {"image": self._to_input_t(img256)},
            {
                "pitch": (1, 66),
                "yaw": (1, 66),
                "roll": (1, 66),
                "t": (1, 3),
                "exp": (1, NUM_KP * 3),
                "scale": (1, 3),
                "kp": (1, NUM_KP * 3),
            },
        )
        kp_info["exp"] = kp_info["exp"].reshape(1, -1, 3)
        kp_info["kp"] = kp_info["kp"].reshape(1, -1, 3)
        return kp_info

    def warp_decode(
        self, feature_3d, kp_source: torch.Tensor, kp_driving: torch.Tensor
    ) -> torch.Tensor:
        """W + G: D(W(f_s; x_s, x_d)) -> RGB image (1,3,512,512) torch [0,1].

        ``feature_3d`` may be a torch tensor (IO-binding path) or a numpy array
        (legacy/fallback). Note the upstream input ordering: the warping module
        receives ``kp_driving`` first and ``kp_source`` second. The intermediate
        warp feature stays on the GPU between W and G on the CUDA path.
        """
        device = self.models_processor.device
        if not isinstance(feature_3d, torch.Tensor):
            feature_3d = torch.from_numpy(np.asarray(feature_3d)).to(device)
        warped = self._infer(
            WARPING_MODEL,
            {
                "feature_3d": feature_3d,
                "kp_driving": kp_driving,
                "kp_source": kp_source,
            },
            {"out": (1, 256, 64, 64)},
        )["out"]
        out = self._infer(SPADE_MODEL, {"feature": warped}, {"out": (1, 3, 512, 512)})
        return out["out"]

    # ------------------------------------------------------------------ #
    # Geometry — ported verbatim from src/pipeline.py (HopeNet -99 binning)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _headpose_to_degree(pred: torch.Tensor) -> torch.Tensor:
        """(bs,66) pose logits -> (bs,) degrees. PerformRecast uses *3 - 99."""
        idx = torch.arange(66, dtype=torch.float32, device=pred.device)
        pred = F.softmax(pred, dim=1)
        return torch.sum(pred * idx, dim=1) * 3 - 99

    @staticmethod
    def _rotation_matrix(
        yaw: torch.Tensor, pitch: torch.Tensor, roll: torch.Tensor
    ) -> torch.Tensor:
        """(bs,) degree angles -> (bs,3,3) rotation matrix (pitch·yaw·roll)."""
        deg2rad = 3.14 / 180  # match upstream's literal 3.14 constant exactly
        yaw = (yaw * deg2rad).unsqueeze(1)
        pitch = (pitch * deg2rad).unsqueeze(1)
        roll = (roll * deg2rad).unsqueeze(1)

        zeros = torch.zeros_like(pitch)
        ones = torch.ones_like(pitch)

        pitch_mat = torch.cat(
            [
                ones,
                zeros,
                zeros,
                zeros,
                torch.cos(pitch),
                -torch.sin(pitch),
                zeros,
                torch.sin(pitch),
                torch.cos(pitch),
            ],
            dim=1,
        ).view(-1, 3, 3)
        yaw_mat = torch.cat(
            [
                torch.cos(yaw),
                zeros,
                torch.sin(yaw),
                zeros,
                ones,
                zeros,
                -torch.sin(yaw),
                zeros,
                torch.cos(yaw),
            ],
            dim=1,
        ).view(-1, 3, 3)
        roll_mat = torch.cat(
            [
                torch.cos(roll),
                -torch.sin(roll),
                zeros,
                torch.sin(roll),
                torch.cos(roll),
                zeros,
                zeros,
                zeros,
                ones,
            ],
            dim=1,
        ).view(-1, 3, 3)

        return torch.einsum("bij,bjk,bkm->bim", pitch_mat, yaw_mat, roll_mat)

    def build_source_info(
        self, kp_info: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Compute the per-frame source descriptor used during animation.

        Returns a dict with the canonical keypoints ``kp`` (1,N,3), expression
        ``exp`` (1,N,3), rotation ``R`` (1,3,3), ``scale`` (1,1), translation
        ``t`` (1,3, tz zeroed) and the transformed source keypoints ``x_s``.
        """
        kp = kp_info["kp"]
        exp = kp_info["exp"]
        yaw = self._headpose_to_degree(kp_info["yaw"])
        pitch = self._headpose_to_degree(kp_info["pitch"])
        roll = self._headpose_to_degree(kp_info["roll"])
        R = self._rotation_matrix(yaw, pitch, roll)
        scale = kp_info["scale"]
        t = kp_info["t"].clone()
        t[..., 2] = 0  # zero tz

        kp_e = kp + exp
        kp_rotated = torch.einsum("bmp,bkp->bkm", R, kp_e)
        kp_rotated = kp_rotated * scale.unsqueeze(1)
        x_s = kp_rotated + t.unsqueeze(1)

        return {"kp": kp, "exp": exp, "R": R, "scale": scale, "t": t, "x_s": x_s}

    # ------------------------------------------------------------------ #
    # Expression composition — ported from src/pipeline.py animate loop
    # ------------------------------------------------------------------ #
    # Implicit-keypoint channel groups referenced by the upstream
    # "Replacement" modulation. Used here for optional region gating.
    EYE_INDICES = list(range(31, 39))  # 31..38 (eye channels)
    MOUTH_INDICES = list(range(44, 47))  # 44..46 (jaw / lip-contour channels)

    def compose_driven_keypoints(
        self,
        source_info: Dict[str, torch.Tensor],
        exp_d: torch.Tensor,
        mode: str = MODE_ENHANCEMENT,
        factor: float = 1.0,
        region: str = "all",
        eye_driving_weight: float = 0.7,
        lip_driving_weight: float = 0.8,
        structural_blend: float = 0.50,
    ) -> torch.Tensor:
        """Build the driven keypoints ``x_d_i`` fed to the warping module.

        Args:
            source_info: output of :meth:`build_source_info` (the swapped face).
            exp_d: driving expression (1,N,3) from the original face's motion.
            mode: ``"Replacement"`` (upstream mode 1) or ``"Enhancement"`` (mode 2) or ``"Relative"`` (mode 3).
            factor: expression strength. 0 keeps the source expression, 1 applies
                the full transfer; values >1 exaggerate it.
            region: ``"all"`` | ``"eyes"`` | ``"lips"`` — restrict where the
                driving expression is applied (source expression kept elsewhere).
            eye_driving_weight: in Replacement mode, how strongly the driver
                overrides the source eye-channel identity (0 keeps source eyes,
                1 fully follows the driver). Upstream default 0.7.
            lip_driving_weight: same, for the lip/jaw channels. Upstream
                default 0.8.

        Mode semantics:
          * Replacement — ``factor=1`` yields the driver's expression with the
            swapped face's eye/lip identity blended back (``eye/lip_driving_weight``).
            ``factor`` interpolates source -> that target.
          * Enhancement — adds the driver's expression on top of the swapped
            face's own expression (``exp_s + factor*exp_d``); ``factor=0`` keeps
            the source, higher values stack/boost the driver's expression.

        The upstream video pipeline uses the driving video's first frame as a
        neutral reference; VisoMaster is stateless per frame, so Enhancement
        treats the implicit keypoint ``exp_d`` (already a delta from the
        canonical keypoints) as the additive delta directly.
        """
        x_s_c = source_info["kp"]
        exp_s = source_info["exp"]
        R = source_info["R"]
        scale = source_info["scale"]
        t = source_info["t"]

        if mode == MODE_REPLACEMENT:
            # Start from the absolute driving expression and blend back
            # source-side eye / lip / jaw channels (identity micro-cues). The
            # blend weights default to the upstream 0.7 (eyes) / 0.8 (lips), but
            # are exposed so users can dial how strongly the driver overrides
            # the swapped face's own eye/lip identity (similarity preservation).
            ew = float(eye_driving_weight)
            lw = float(lip_driving_weight)
            modulated = exp_d.clone()
            modulated[:, 31:34, 2] = exp_s[:, 31:34, 2]
            modulated[:, 36:39, 2] = exp_s[:, 36:39, 2]
            modulated[:, 44:47, 2] = exp_s[:, 44:47, 2]
            modulated[:, 44:47, 0] = exp_s[:, 44:47, 0]
            modulated[:, 44:47, 1] = (
                exp_s[:, 44:47, 1] * (1.0 - lw) + exp_d[:, 44:47, 1] * lw
            )
            modulated[:, 31:34, :2] = (
                exp_s[:, 31:34, :2] * (1.0 - ew) + exp_d[:, 31:34, :2] * ew
            )
            modulated[:, 36:39, :2] = (
                exp_s[:, 36:39, :2] * (1.0 - ew) + exp_d[:, 36:39, :2] * ew
            )
            new_exp = exp_s + factor * (modulated - exp_s)

        elif mode == MODE_RELATIVE:
            ew = float(eye_driving_weight)
            lw = float(lip_driving_weight)

            # We build the new expression additively on top of the source
            new_exp = exp_s.clone()

            # --- 1. MOUTH ANIMATION (Indices 44-46) ---
            mouth_idx = list(range(44, 47))

            # Calculate centroids
            s_mouth_center = exp_s[:, mouth_idx, :].mean(dim=1, keepdim=True)
            d_mouth_center = exp_d[:, mouth_idx, :].mean(dim=1, keepdim=True)

            # Only align X (horizontal) and Z (depth) axes.
            # Aligning the Y axis cancels out the jaw drop
            pos_offset_mouth = s_mouth_center - d_mouth_center
            pos_offset_mouth[..., 1] = 0.0  # Zero out the Y-axis alignment

            aligned_d_mouth = exp_d[:, mouth_idx, :] + pos_offset_mouth
            mouth_delta = aligned_d_mouth - exp_s[:, mouth_idx, :]

            # Apply factor and lip weight
            new_exp[:, mouth_idx, :] = exp_s[:, mouth_idx, :] + (
                mouth_delta * lw * factor
            )

            # --- 2. EYE ANIMATION (Indices 31-38) ---
            eye_idx = list(range(31, 39))

            s_eye_center = exp_s[:, eye_idx, :].mean(dim=1, keepdim=True)
            d_eye_center = exp_d[:, eye_idx, :].mean(dim=1, keepdim=True)

            # Same fix for eyes: preserve vertical movement (brow raises, wide eyes)
            pos_offset_eye = s_eye_center - d_eye_center
            pos_offset_eye[..., 1] = 0.0

            aligned_d_eye = exp_d[:, eye_idx, :] + pos_offset_eye
            eye_delta = aligned_d_eye - exp_s[:, eye_idx, :]

            new_exp[:, eye_idx, :] = exp_s[:, eye_idx, :] + (eye_delta * ew * factor)

            # --- 3. STRUCTURAL BLEND (All remaining indices) ---
            struct_idx = [i for i in range(49) if i not in (mouth_idx + eye_idx)]

            s_head_center = exp_s.mean(dim=1, keepdim=True)
            d_head_center = exp_d.mean(dim=1, keepdim=True)

            # For the general head structure, we align all axes to prevent the whole head from migrating
            pos_offset_struct = s_head_center - d_head_center

            aligned_d_struct = exp_d[:, struct_idx, :] + pos_offset_struct
            struct_delta = aligned_d_struct - exp_s[:, struct_idx, :]

            # Apply structural blend (flexibility)
            blended_struct = exp_s[:, struct_idx, :] + (struct_delta * structural_blend)

            # Rigidly lock Z-axis (skull depth) for the structure to preserve likeness
            blended_struct[..., 2] = exp_s[:, struct_idx, 2]

            new_exp[:, struct_idx, :] = blended_struct

        elif mode == MODE_ENHANCEMENT:
            # ENHANCEMENT = keep the swapped face's own expression and ADD the
            # driving expression on top of it (scaled by `factor`). This is
            # genuinely additive, so it stays distinct from Replacement (which
            # *replaces* the expression with the driver's). The implicit keypoint
            # `exp` is already a delta from the canonical keypoints, so the
            # driver's `exp_d` acts as the additive expression delta directly —
            # no explicit neutral-reference frame is needed.
            #
            # (Previously this used `exp_s + factor*(exp_d - exp_s)`, i.e. the
            # source's own expression as the neutral reference. At factor=1 that
            # cancels `exp_s` and collapses to `exp_d` — making Enhancement
            # nearly identical to Replacement, which is why switching modes
            # appeared to do nothing.)
            new_exp = exp_s + factor * exp_d
        else:
            raise ValueError(f"Unsupported recast mode: {mode!r}")

        # Optional region gating: keep the source expression outside the region.
        if region != "all":
            indices = self.EYE_INDICES if region == "eyes" else self.MOUTH_INDICES
            gated = exp_s.clone()
            gated[:, indices, :] = new_exp[:, indices, :]
            new_exp = gated

        kp_e = x_s_c + new_exp
        kp_rotated = torch.einsum("bmp,bkp->bkm", R, kp_e)
        kp_rotated = kp_rotated * scale.unsqueeze(1)
        x_d_i = kp_rotated + t.unsqueeze(1)

        return x_d_i
