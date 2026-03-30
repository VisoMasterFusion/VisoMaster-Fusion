"""
Integration tests: custom kernel accuracy vs ORT CUDA EP.

Requires: CUDA GPU, model files in model_assets/.
Run with:
    pytest tests/integration/test_custom_kernels_accuracy.py -v

Each test class:
  1. Creates an ORT CUDA EP session (baseline).
  2. Loads the corresponding PyTorch custom kernel.
  3. Runs both on the same random input.
  4. Asserts max|diff| is within the tolerance for that model type.
"""

from __future__ import annotations

import numpy as np
import pytest

try:
    import torch
    import onnxruntime as ort

    _CUDA_AVAILABLE = torch.cuda.is_available()
    _ORT_CUDA_AVAILABLE = "CUDAExecutionProvider" in ort.get_available_providers()
except ImportError:
    _CUDA_AVAILABLE = False
    _ORT_CUDA_AVAILABLE = False

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ort_cuda_session(onnx_path: str | Path):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        str(onnx_path),
        so,
        providers=[
            ("CUDAExecutionProvider", {"device_id": "0"}),
            ("CPUExecutionProvider", {}),
        ],
    )


def _skip_no_cuda():
    if not _CUDA_AVAILABLE or not _ORT_CUDA_AVAILABLE:
        pytest.skip("CUDA or ORT CUDA EP not available")


def _skip_no_model(path: Path):
    if not path.exists():
        pytest.skip(f"model not found: {path}")


# ---------------------------------------------------------------------------
# det_10g — SCRFD-10G face detector
# ---------------------------------------------------------------------------


class TestDet10gAccuracy:
    ONNX = ROOT / "model_assets" / "det_10g.onnx"
    TOL = 0.02

    def test_fp16_cuda_graph_vs_ort(self):
        _skip_no_cuda()
        _skip_no_model(self.ONNX)
        from custom_kernels.det_10g.det10g_torch import (
            Det10gTorch,
            build_cuda_graph_runner,
        )

        sess = _ort_cuda_session(self.ONNX)
        in_name = sess.get_inputs()[0].name
        out_names = [o.name for o in sess.get_outputs()]

        inp = torch.randn(1, 3, 640, 640, dtype=torch.float32, device="cuda")
        inp_np = inp.cpu().numpy()

        model = (
            Det10gTorch.from_onnx(str(self.ONNX), compute_dtype=torch.float16)
            .cuda()
            .eval()
        )
        runner = build_cuda_graph_runner(model)

        ort_out = sess.run(out_names, {in_name: inp_np})
        with torch.no_grad():
            pt_out = runner(inp)

        for i, (a, b) in enumerate(zip(ort_out, pt_out)):
            diff = np.abs(np.array(a) - b.float().cpu().numpy()).max()
            assert diff < self.TOL, (
                f"output[{i}]: max|diff|={diff:.4e} > tol={self.TOL}"
            )


# ---------------------------------------------------------------------------
# yoloface_8n — YOLOv8n face detector
# ---------------------------------------------------------------------------


class TestYoloFace8nAccuracy:
    ONNX = ROOT / "model_assets" / "yoloface_8n.onnx"
    TOL = 0.02

    def test_fp16_cuda_graph_vs_ort(self):
        _skip_no_cuda()
        _skip_no_model(self.ONNX)
        from custom_kernels.yoloface_8n.yoloface8n_torch import (
            YoloFace8nTorch,
            build_cuda_graph_runner,
        )

        sess = _ort_cuda_session(self.ONNX)
        in_name = sess.get_inputs()[0].name

        inp = torch.rand(1, 3, 640, 640, dtype=torch.float32, device="cuda")
        inp_np = inp.cpu().numpy()

        model = (
            YoloFace8nTorch.from_onnx(str(self.ONNX), compute_dtype=torch.float16)
            .cuda()
            .eval()
        )
        runner = build_cuda_graph_runner(model)

        ort_out = sess.run(["output0"], {in_name: inp_np})[0]
        with torch.no_grad():
            pt_out = runner(inp)

        diff = np.abs(ort_out - pt_out.float().cpu().numpy()).max()
        assert diff < self.TOL, f"max|diff|={diff:.4e} > tol={self.TOL}"


# ---------------------------------------------------------------------------
# det_106 — 2D-106 landmark detector
# ---------------------------------------------------------------------------


class TestDet106Accuracy:
    ONNX = ROOT / "model_assets" / "2d106det.onnx"
    TOL = 0.02

    def test_fp16_cuda_graph_vs_ort(self):
        _skip_no_cuda()
        _skip_no_model(self.ONNX)
        from custom_kernels.det_106.det_106_torch import (
            Det106Torch,
            build_cuda_graph_runner,
        )

        sess = _ort_cuda_session(self.ONNX)
        in_name = sess.get_inputs()[0].name
        out_names = [o.name for o in sess.get_outputs()]

        inp = torch.randint(
            0, 256, (1, 3, 192, 192), dtype=torch.float32, device="cuda"
        )
        inp_np = inp.cpu().numpy()

        model = (
            Det106Torch.from_onnx(str(self.ONNX), compute_dtype=torch.float16)
            .cuda()
            .eval()
        )
        runner = build_cuda_graph_runner(model)

        ort_out = sess.run(out_names, {in_name: inp_np})[0]
        with torch.no_grad():
            pt_out = runner(inp)

        diff = np.abs(ort_out - pt_out.float().cpu().numpy()).max()
        assert diff < self.TOL, f"max|diff|={diff:.4e} > tol={self.TOL}"


# ---------------------------------------------------------------------------
# inswapper_128 — face swapper
# ---------------------------------------------------------------------------


class TestInswapper128Accuracy:
    ONNX = ROOT / "model_assets" / "inswapper_128.fp16.onnx"
    TOL = 0.05

    def test_triton_cuda_graph_vs_ort(self):
        _skip_no_cuda()
        _skip_no_model(self.ONNX)
        from custom_kernels.inswapper_128.inswapper_torch import (
            InSwapperTorch,
            build_cuda_graph_runner,
        )

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess = ort.InferenceSession(
            str(self.ONNX),
            so,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            provider_options=[{"device_id": "0"}, {}],
        )

        rng = np.random.default_rng(42)
        target_np = rng.random((1, 3, 128, 128)).astype(np.float32)
        src_raw = rng.standard_normal((1, 512)).astype(np.float32)
        source_np = (src_raw / np.linalg.norm(src_raw)).astype(np.float32)

        io = sess.io_binding()
        io.bind_cpu_input("target", target_np)
        io.bind_cpu_input("source", source_np)
        io.bind_output("output", "cuda")
        sess.run_with_iobinding(io)
        ort_out = np.array(io.copy_outputs_to_cpu()[0])

        target_gpu = torch.from_numpy(target_np).cuda()
        source_gpu = torch.from_numpy(source_np).cuda()

        model = InSwapperTorch(str(self.ONNX), use_custom_kernel=True).cuda().eval()
        runner = build_cuda_graph_runner(model, target_gpu, source_gpu)

        with torch.no_grad():
            pt_out = runner(target_gpu, source_gpu).cpu().numpy()

        diff = np.abs(ort_out - pt_out).max()
        assert diff < self.TOL, f"max|diff|={diff:.4e} > tol={self.TOL}"


# ---------------------------------------------------------------------------
# w600k_r50 — IResNet-50 ArcFace (face recognizer)
# ---------------------------------------------------------------------------


class TestW600kR50Accuracy:
    ONNX = ROOT / "model_assets" / "w600k_r50.onnx"
    COSINE_TOL = 0.999

    def test_fp16_cuda_graph_vs_ort(self):
        _skip_no_cuda()
        _skip_no_model(self.ONNX)
        from custom_kernels.w600k_r50.w600k_r50_torch import (
            IResNet50Torch,
            build_cuda_graph_runner,
        )

        sess = _ort_cuda_session(self.ONNX)
        in_name = sess.get_inputs()[0].name

        inp = torch.randn(1, 3, 112, 112, dtype=torch.float32, device="cuda")
        inp_np = inp.cpu().numpy()

        model = (
            IResNet50Torch.from_onnx(str(self.ONNX), compute_dtype=torch.float16)
            .cuda()
            .eval()
        )
        runner = build_cuda_graph_runner(model)

        ort_out = sess.run(None, {in_name: inp_np})[0].flatten()
        with torch.no_grad():
            pt_out = runner(inp).float().cpu().numpy().flatten()

        cos_sim = float(
            np.dot(ort_out, pt_out)
            / (np.linalg.norm(ort_out) * np.linalg.norm(pt_out) + 1e-8)
        )
        assert cos_sim > self.COSINE_TOL, (
            f"cosine_sim={cos_sim:.6f} < {self.COSINE_TOL}"
        )


# ---------------------------------------------------------------------------
# faceparser_resnet34 — BiSeNet face parser
# ---------------------------------------------------------------------------


class TestFaceparserResnet34Accuracy:
    ONNX = ROOT / "model_assets" / "faceparser_resnet34.onnx"
    TOL = 0.1

    def test_fp16_cuda_graph_vs_ort(self):
        _skip_no_cuda()
        _skip_no_model(self.ONNX)
        from custom_kernels.faceparser_resnet34.faceparser_resnet34_torch import (
            FaceParserResnet34Torch,
            build_cuda_graph_runner,
        )

        sess = _ort_cuda_session(self.ONNX)
        in_name = sess.get_inputs()[0].name

        inp = torch.randn(1, 3, 512, 512, dtype=torch.float32, device="cuda")
        inp_np = inp.cpu().numpy()

        model = (
            FaceParserResnet34Torch.from_onnx(
                str(self.ONNX), compute_dtype=torch.float16
            )
            .cuda()
            .eval()
        )
        runner = build_cuda_graph_runner(model)

        ort_out = sess.run(["output"], {in_name: inp_np})[0]
        with torch.no_grad():
            pt_out = runner(inp).float().cpu().numpy()

        diff = np.abs(ort_out - pt_out).max()
        assert diff < self.TOL, f"max|diff|={diff:.4e} > tol={self.TOL}"


# ---------------------------------------------------------------------------
# GFPGANv1.4 — face restorer
# ---------------------------------------------------------------------------


class TestGFPGANv14Accuracy:
    ONNX = ROOT / "model_assets" / "GFPGANv1.4.onnx"
    TOL = 0.1

    def test_fp16_cuda_graph_vs_ort(self):
        _skip_no_cuda()
        _skip_no_model(self.ONNX)
        from custom_kernels.gfpgan_v1_4.gfpgan_torch import (
            GFPGANTorch,
            build_cuda_graph_runner,
        )

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess = ort.InferenceSession(
            str(self.ONNX),
            so,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            provider_options=[{"device_id": "0"}, {}],
        )
        inp_np = np.random.default_rng(0).random((1, 3, 512, 512)).astype(np.float32)
        ort_out = sess.run(None, {"input": inp_np})[0]

        inp = torch.from_numpy(inp_np).cuda()
        model = (
            GFPGANTorch.from_onnx(str(self.ONNX), compute_dtype=torch.float16)
            .cuda()
            .eval()
        )
        runner = build_cuda_graph_runner(model, inp_shape=(1, 3, 512, 512))

        with torch.no_grad():
            pt_out = runner(inp).float().cpu().numpy()

        diff = np.abs(ort_out - pt_out).max()
        assert diff < self.TOL, f"max|diff|={diff:.4e} > tol={self.TOL}"


# ---------------------------------------------------------------------------
# GFPGAN-1024 — face restorer (1024 variant)
# ---------------------------------------------------------------------------


class TestGFPGAN1024Accuracy:
    ONNX = ROOT / "model_assets" / "gfpgan-1024.onnx"
    TOL = 0.1

    def test_fp16_cuda_graph_vs_ort(self):
        _skip_no_cuda()
        _skip_no_model(self.ONNX)
        from custom_kernels.gfpgan_v1_4.gfpgan_torch import (
            GFPGANTorch,
            build_cuda_graph_runner,
        )

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess = ort.InferenceSession(
            str(self.ONNX),
            so,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            provider_options=[{"device_id": "0"}, {}],
        )
        inp_np = np.random.default_rng(0).random((1, 3, 512, 512)).astype(np.float32)
        ort_out = sess.run(None, {"input": inp_np})[0]

        inp = torch.from_numpy(inp_np).cuda()
        model = (
            GFPGANTorch.from_onnx(str(self.ONNX), compute_dtype=torch.float16)
            .cuda()
            .eval()
        )
        runner = build_cuda_graph_runner(model, inp_shape=(1, 3, 512, 512))

        with torch.no_grad():
            pt_out = runner(inp).float().cpu().numpy()

        diff = np.abs(ort_out - pt_out).max()
        assert diff < self.TOL, f"max|diff|={diff:.4e} > tol={self.TOL}"


# ---------------------------------------------------------------------------
# GPEN-BFR-256 — face restorer
# ---------------------------------------------------------------------------


class TestGPENBFR256Accuracy:
    ONNX = ROOT / "model_assets" / "GPEN-BFR-256.onnx"
    TOL = 0.1

    def test_fp16_cuda_graph_vs_ort(self):
        _skip_no_cuda()
        _skip_no_model(self.ONNX)
        from custom_kernels.gpen_bfr.gpen_torch import (
            GPENTorch,
            build_cuda_graph_runner,
        )

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess = ort.InferenceSession(
            str(self.ONNX),
            so,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            provider_options=[{"device_id": "0"}, {}],
        )
        inp_name = sess.get_inputs()[0].name
        inp_np = np.random.default_rng(0).random((1, 3, 256, 256)).astype(np.float32)
        ort_out = sess.run(None, {inp_name: inp_np})[0]

        inp = torch.from_numpy(inp_np).cuda()
        model = (
            GPENTorch.from_onnx(str(self.ONNX), compute_dtype=torch.float16)
            .cuda()
            .eval()
        )
        runner = build_cuda_graph_runner(model, inp_shape=(1, 3, 256, 256))

        with torch.no_grad():
            pt_out = runner(inp).float().cpu().numpy()

        diff = np.abs(ort_out - pt_out).max()
        assert diff < self.TOL, f"max|diff|={diff:.4e} > tol={self.TOL}"


# ---------------------------------------------------------------------------
# CodeFormer — face restorer
# ---------------------------------------------------------------------------


class TestCodeFormerAccuracy:
    ONNX = ROOT / "model_assets" / "codeformer_fp16.onnx"
    TOL = 0.1

    def test_fp16_vs_ort(self):
        _skip_no_cuda()
        _skip_no_model(self.ONNX)
        from custom_kernels.codeformer.codeformer_torch import CodeFormerTorch

        sess = _ort_cuda_session(self.ONNX)
        in_name = sess.get_inputs()[0].name
        w_name = sess.get_inputs()[1].name
        out_name = sess.get_outputs()[0].name

        inp = torch.randn(1, 3, 512, 512, dtype=torch.float32, device="cuda")
        inp_np = inp.cpu().numpy()
        w_np = np.array([0.5], dtype=np.float64)

        ort_out = sess.run([out_name], {in_name: inp_np, w_name: w_np})[0]

        model = (
            CodeFormerTorch.from_onnx(str(self.ONNX), compute_dtype=torch.float16)
            .cuda()
            .eval()
        )
        with torch.no_grad():
            pt_out = model(inp, fidelity_weight=0.5).float().cpu().numpy()

        diff = np.abs(ort_out - pt_out).max()
        assert diff < self.TOL, f"max|diff|={diff:.4e} > tol={self.TOL}"


# ---------------------------------------------------------------------------
# RestoreFormerPlusPlus — face restorer
# ---------------------------------------------------------------------------


class TestRestoreFormerAccuracy:
    ONNX = ROOT / "model_assets" / "RestoreFormerPlusPlus.fp16.onnx"
    TOL = 0.1

    def test_fp16_cuda_graph_vs_ort(self):
        _skip_no_cuda()
        _skip_no_model(self.ONNX)
        from custom_kernels.restoreformer.restoreformer_torch import (
            RestoreFormerPlusPlusTorch,
            build_cuda_graph_runner,
        )

        sess = _ort_cuda_session(self.ONNX)
        in_name = sess.get_inputs()[0].name
        out_name = sess.get_outputs()[0].name

        inp = torch.randn(1, 3, 512, 512, dtype=torch.float32, device="cuda")
        inp_np = inp.cpu().numpy()

        ort_out = sess.run([out_name], {in_name: inp_np})[0]

        model = (
            RestoreFormerPlusPlusTorch.from_onnx(
                str(self.ONNX), compute_dtype=torch.float16
            )
            .cuda()
            .eval()
        )
        runner = build_cuda_graph_runner(model, inp_shape=(1, 3, 512, 512))

        with torch.no_grad():
            pt_out = runner(inp).float().cpu().numpy()

        diff = np.abs(ort_out - pt_out).max()
        assert diff < self.TOL, f"max|diff|={diff:.4e} > tol={self.TOL}"


# ---------------------------------------------------------------------------
# landmark_203 — 203-point face landmark detector
# ---------------------------------------------------------------------------


class TestLandmark203Accuracy:
    ONNX = ROOT / "model_assets" / "landmark.onnx"
    TOL = 0.5

    def test_fp16_cuda_graph_vs_ort(self):
        _skip_no_cuda()
        _skip_no_model(self.ONNX)
        from custom_kernels.landmark_203.landmark_203_torch import (
            Landmark203Torch,
            build_cuda_graph_runner,
        )

        sess = _ort_cuda_session(self.ONNX)
        in_name = sess.get_inputs()[0].name
        out_names = [o.name for o in sess.get_outputs()]

        inp = torch.randn(1, 3, 224, 224, dtype=torch.float32, device="cuda")
        inp_np = inp.cpu().numpy()

        model = (
            Landmark203Torch.from_onnx(str(self.ONNX), compute_dtype=torch.float16)
            .cuda()
            .eval()
        )
        runner = build_cuda_graph_runner(model)

        ref_outs = sess.run(out_names, {in_name: inp_np})
        ref_pts = ref_outs[2][0]  # (406,)
        with torch.no_grad():
            pt_outs = runner(inp)
        pt_pts = pt_outs[2][0].float().cpu().numpy()

        diff = np.abs(ref_pts - pt_pts).max()
        assert diff < self.TOL, f"max|diff|={diff:.4e} > tol={self.TOL}"


# ---------------------------------------------------------------------------
# landmark_1k3d68 — 1k3d68 3D face landmark detector
# ---------------------------------------------------------------------------


class TestLandmark1k3d68Accuracy:
    ONNX = ROOT / "model_assets" / "1k3d68.onnx"
    TOL = 0.5

    def test_fp16_cuda_graph_vs_ort(self):
        _skip_no_cuda()
        _skip_no_model(self.ONNX)
        from custom_kernels.landmark_1k3d68.landmark_1k3d68_torch import (
            Landmark1k3d68Torch,
            build_cuda_graph_runner,
        )

        sess = _ort_cuda_session(self.ONNX)
        in_name = sess.get_inputs()[0].name
        out_name = sess.get_outputs()[0].name

        inp = torch.randn(1, 3, 192, 192, dtype=torch.float32, device="cuda")
        inp_np = inp.cpu().numpy()

        model = (
            Landmark1k3d68Torch.from_onnx(str(self.ONNX), compute_dtype=torch.float16)
            .cuda()
            .eval()
        )
        runner = build_cuda_graph_runner(model)

        ref_np = sess.run([out_name], {in_name: inp_np})[0][0]  # (3309,)
        with torch.no_grad():
            pt_out = runner(inp)[0].float().cpu().numpy()

        diff = np.abs(ref_np - pt_out).max()
        assert diff < self.TOL, f"max|diff|={diff:.4e} > tol={self.TOL}"


# ---------------------------------------------------------------------------
# fan_2dfan4 — 2D FAN landmark detector
# ---------------------------------------------------------------------------


class TestFan2dfan4Accuracy:
    ONNX = ROOT / "model_assets" / "2dfan4.onnx"
    TOL = 0.5

    def test_fp16_cuda_graph_vs_ort(self):
        _skip_no_cuda()
        _skip_no_model(self.ONNX)
        from custom_kernels.fan_2dfan4.fan_2dfan4_torch import (
            FAN2dfan4,
            build_cuda_graph_runner,
        )

        sess = _ort_cuda_session(self.ONNX)
        in_name = sess.get_inputs()[0].name
        out_names = [o.name for o in sess.get_outputs()]

        inp = torch.randn(1, 3, 256, 256, dtype=torch.float32, device="cuda")
        inp_np = inp.cpu().numpy()

        model = (
            FAN2dfan4.from_onnx(str(self.ONNX), compute_dtype=torch.float16)
            .cuda()
            .eval()
        )
        runner = build_cuda_graph_runner(model)

        ref_outs = sess.run(out_names, {in_name: inp_np})
        ref_lmk = ref_outs[0][0, :, :2]  # (68, 2)
        with torch.no_grad():
            pt_outs = runner(inp)
        pt_lmk = pt_outs[0][0, :, :2].float().cpu().numpy()

        diff = np.abs(ref_lmk - pt_lmk).max()
        assert diff < self.TOL, f"max|diff|={diff:.4e} > tol={self.TOL}"


# ---------------------------------------------------------------------------
# face_landmark478 — MediaPipe 478-point face landmark detector
# ---------------------------------------------------------------------------


class TestFaceLandmark478Accuracy:
    ONNX = ROOT / "model_assets" / "face_landmarks_detector_Nx3x256x256.onnx"
    TOL = 0.5

    def test_fp16_cuda_graph_vs_ort(self):
        _skip_no_cuda()
        _skip_no_model(self.ONNX)
        from custom_kernels.face_landmark478.face_landmark478_torch import (
            FaceLandmark478Torch,
            build_cuda_graph_runner,
        )

        sess = _ort_cuda_session(self.ONNX)
        in_name = sess.get_inputs()[0].name
        out_names = [o.name for o in sess.get_outputs()]

        inp = torch.randn(1, 3, 256, 256, dtype=torch.float32, device="cuda")
        inp_np = inp.cpu().numpy()

        model = (
            FaceLandmark478Torch.from_onnx(str(self.ONNX), compute_dtype=torch.float16)
            .cuda()
            .eval()
        )
        runner = build_cuda_graph_runner(model)

        ref_outs = sess.run(out_names, {in_name: inp_np})
        ref_lmk = ref_outs[0].reshape(478, 3)
        with torch.no_grad():
            pt_outs = runner(inp)
        pt_lmk = pt_outs[0].float().cpu().numpy().reshape(478, 3)

        diff = np.abs(ref_lmk - pt_lmk).max()
        assert diff < self.TOL, f"max|diff|={diff:.4e} > tol={self.TOL}"


# ---------------------------------------------------------------------------
# face_blendshapes — MediaPipe face blendshapes
# ---------------------------------------------------------------------------


class TestFaceBlendshapesAccuracy:
    ONNX = ROOT / "model_assets" / "face_blendshapes_Nx146x2.onnx"
    TOL = 0.5

    def test_fp16_cuda_graph_vs_ort(self):
        _skip_no_cuda()
        _skip_no_model(self.ONNX)
        from custom_kernels.face_blendshapes.face_blendshapes_torch import (
            FaceBlendShapesTorch,
            build_cuda_graph_runner,
        )

        sess = _ort_cuda_session(self.ONNX)
        in_name = sess.get_inputs()[0].name
        out_names = [o.name for o in sess.get_outputs()]

        inp = torch.randn(1, 146, 2, dtype=torch.float32, device="cuda")
        inp_np = inp.cpu().numpy()

        model = (
            FaceBlendShapesTorch.from_onnx(str(self.ONNX), compute_dtype=torch.float16)
            .cuda()
            .eval()
        )
        runner = build_cuda_graph_runner(model)

        ref_out = sess.run(out_names, {in_name: inp_np})[0]
        with torch.no_grad():
            pt_out = runner(inp).float().cpu().numpy()

        diff = np.abs(ref_out - pt_out).max()
        assert diff < self.TOL, f"max|diff|={diff:.4e} > tol={self.TOL}"


# ---------------------------------------------------------------------------
# peppapig_98 — PeppaPig 98-point landmark detector
# ---------------------------------------------------------------------------


class TestPeppaPig98Accuracy:
    ONNX = ROOT / "model_assets" / "peppapig_teacher_Nx3x256x256.onnx"
    TOL = 0.5

    def test_fp16_cuda_graph_vs_ort(self):
        _skip_no_cuda()
        _skip_no_model(self.ONNX)
        from custom_kernels.peppapig_98.peppapig_98_torch import (
            PeppaPig98Torch,
            build_cuda_graph_runner,
        )

        sess = _ort_cuda_session(self.ONNX)
        in_name = sess.get_inputs()[0].name
        out_names = [o.name for o in sess.get_outputs()]

        inp = torch.rand(1, 3, 256, 256, dtype=torch.float32, device="cuda")
        inp_np = inp.cpu().numpy()

        model = (
            PeppaPig98Torch(str(self.ONNX), compute_dtype=torch.float16).cuda().eval()
        )
        runner = build_cuda_graph_runner(model)

        ref_outs = sess.run(out_names, {in_name: inp_np})
        ref_lmk = ref_outs[0][0]  # (98, 3)
        with torch.no_grad():
            pt_out = runner(inp)[0].float().cpu().numpy()

        diff = np.abs(ref_lmk[:, :2] - pt_out[:, :2]).max()
        assert diff < self.TOL, f"max|diff|={diff:.4e} > tol={self.TOL}"


# ---------------------------------------------------------------------------
# occluder — face occlusion segmentation
# ---------------------------------------------------------------------------


class TestOccluderAccuracy:
    ONNX = ROOT / "model_assets" / "occluder.onnx"
    TOL = 0.05

    def test_fp16_cuda_graph_vs_ort(self):
        _skip_no_cuda()
        _skip_no_model(self.ONNX)
        from custom_kernels.occluder.occluder_torch import (
            OccluderTorch,
            build_cuda_graph_runner,
        )

        sess = _ort_cuda_session(self.ONNX)
        in_name = sess.get_inputs()[0].name

        inp = torch.rand(1, 3, 256, 256, dtype=torch.float32, device="cuda")
        inp_np = inp.cpu().numpy()

        model = (
            OccluderTorch.from_onnx(str(self.ONNX), compute_dtype=torch.float16)
            .cuda()
            .eval()
        )
        runner = build_cuda_graph_runner(model)

        ort_out = sess.run(None, {in_name: inp_np})[0]
        with torch.no_grad():
            pt_out = runner(inp).float().cpu().numpy()

        diff = np.abs(ort_out - pt_out).max()
        assert diff < self.TOL, f"max|diff|={diff:.4e} > tol={self.TOL}"


# ---------------------------------------------------------------------------
# xseg — XSeg face segmentation
# ---------------------------------------------------------------------------


class TestXSegAccuracy:
    ONNX = ROOT / "model_assets" / "XSeg_model.onnx"
    TOL = 0.05

    def test_fp16_cuda_graph_vs_ort(self):
        _skip_no_cuda()
        _skip_no_model(self.ONNX)
        from custom_kernels.xseg.xseg_torch import XSegTorch, build_cuda_graph_runner

        sess = _ort_cuda_session(self.ONNX)
        in_name = sess.get_inputs()[0].name

        inp = torch.rand(1, 3, 256, 256, dtype=torch.float32, device="cuda")
        inp_np = inp.cpu().numpy()

        model = (
            XSegTorch.from_onnx(str(self.ONNX), compute_dtype=torch.float16)
            .cuda()
            .eval()
        )
        runner = build_cuda_graph_runner(model)

        ort_out = sess.run(None, {in_name: inp_np})[0]
        with torch.no_grad():
            pt_out = runner(inp).float().cpu().numpy()

        diff = np.abs(ort_out - pt_out).max()
        assert diff < self.TOL, f"max|diff|={diff:.4e} > tol={self.TOL}"


# ---------------------------------------------------------------------------
# res50 — RetinaFace/FaceLandmark5 detector
# ---------------------------------------------------------------------------


class TestRes50Accuracy:
    ONNX = ROOT / "model_assets" / "res50.onnx"
    TOL = 0.02

    def test_fp16_cuda_graph_vs_ort(self):
        _skip_no_cuda()
        _skip_no_model(self.ONNX)
        from custom_kernels.res50.res50_torch import Res50Torch, build_cuda_graph_runner

        sess = _ort_cuda_session(self.ONNX)
        in_name = sess.get_inputs()[0].name
        out_names = [o.name for o in sess.get_outputs()]

        inp = torch.randn(1, 3, 512, 512, dtype=torch.float32, device="cuda")
        inp_np = inp.cpu().numpy()

        model = (
            Res50Torch.from_onnx(str(self.ONNX), compute_dtype=torch.float16)
            .cuda()
            .eval()
        )
        runner = build_cuda_graph_runner(model)

        ort_out = sess.run(out_names, {in_name: inp_np})
        with torch.no_grad():
            pt_out = runner(inp)

        for i, (a, b) in enumerate(zip(ort_out, pt_out)):
            diff = np.abs(np.array(a) - b.float().cpu().numpy()).max()
            assert diff < self.TOL, (
                f"output[{i}]: max|diff|={diff:.4e} > tol={self.TOL}"
            )


# ---------------------------------------------------------------------------
# vgg_combo — VGG feature extractor (relu3_3 + relu3_1)
# ---------------------------------------------------------------------------


class TestVggComboAccuracy:
    ONNX = ROOT / "model_assets" / "vgg_combo_relu3_3_relu3_1.onnx"
    TOL = 0.1

    def test_fp16_cuda_graph_vs_ort(self):
        _skip_no_cuda()
        _skip_no_model(self.ONNX)
        from custom_kernels.vgg_combo.vgg_combo_torch import (
            VggComboTorch,
            build_cuda_graph_runner,
        )

        sess = _ort_cuda_session(self.ONNX)
        in_name = sess.get_inputs()[0].name
        out_names = [o.name for o in sess.get_outputs()]

        inp = torch.randn(1, 3, 512, 512, dtype=torch.float32, device="cuda")
        inp_np = inp.cpu().numpy()

        model = (
            VggComboTorch.from_onnx(str(self.ONNX), compute_dtype=torch.float16)
            .cuda()
            .eval()
        )
        runner = build_cuda_graph_runner(model)

        ort_out = sess.run(out_names, {in_name: inp_np})[0]
        with torch.no_grad():
            pt_out = runner(inp).float().cpu().numpy()

        diff = np.abs(ort_out - pt_out).max()
        assert diff < self.TOL, f"max|diff|={diff:.4e} > tol={self.TOL}"


# ---------------------------------------------------------------------------
# ref_ldm VAE Encoder
# ---------------------------------------------------------------------------


class TestRefLDMEncoderAccuracy:
    ONNX = ROOT / "model_assets" / "ref_ldm_vae_encoder.onnx"
    TOL = 0.05

    def test_fp16_cuda_graph_vs_ort(self):
        _skip_no_cuda()
        _skip_no_model(self.ONNX)
        from custom_kernels.ref_ldm.ref_ldm_torch import (
            RefLDMEncoderTorch,
            build_cuda_graph_runner,
        )

        sess = _ort_cuda_session(self.ONNX)
        in_name = sess.get_inputs()[0].name
        out_name = sess.get_outputs()[0].name

        inp = torch.randn(1, 3, 512, 512, dtype=torch.float32, device="cuda")
        inp_np = inp.cpu().numpy()

        model = (
            RefLDMEncoderTorch.from_onnx(str(self.ONNX), compute_dtype=torch.float16)
            .cuda()
            .eval()
        )
        runner = build_cuda_graph_runner(model, inp_shape=(1, 3, 512, 512))

        ort_out = sess.run([out_name], {in_name: inp_np})[0]
        with torch.no_grad():
            pt_out = runner(inp).float().cpu().numpy()

        diff = np.abs(ort_out - pt_out).max()
        assert diff < self.TOL, f"max|diff|={diff:.4e} > tol={self.TOL}"


# ---------------------------------------------------------------------------
# ref_ldm VAE Decoder
# ---------------------------------------------------------------------------


class TestRefLDMDecoderAccuracy:
    ONNX = ROOT / "model_assets" / "ref_ldm_vae_decoder.onnx"
    TOL = 0.05

    def test_fp16_cuda_graph_vs_ort(self):
        _skip_no_cuda()
        _skip_no_model(self.ONNX)
        from custom_kernels.ref_ldm.ref_ldm_torch import (
            RefLDMDecoderTorch,
            build_cuda_graph_runner,
        )

        sess = _ort_cuda_session(self.ONNX)
        in_name = sess.get_inputs()[0].name
        out_name = sess.get_outputs()[0].name

        lat = torch.randn(1, 8, 64, 64, dtype=torch.float32, device="cuda")
        lat_np = lat.cpu().numpy()

        model = (
            RefLDMDecoderTorch.from_onnx(str(self.ONNX), compute_dtype=torch.float16)
            .cuda()
            .eval()
        )
        runner = build_cuda_graph_runner(model, inp_shape=(1, 8, 64, 64))

        ort_out = sess.run([out_name], {in_name: lat_np})[0]
        with torch.no_grad():
            pt_out = runner(lat).float().cpu().numpy()

        diff = np.abs(ort_out - pt_out).max()
        assert diff < self.TOL, f"max|diff|={diff:.4e} > tol={self.TOL}"


# ---------------------------------------------------------------------------
# ref_ldm UNet
# ---------------------------------------------------------------------------


class TestRefLDMUNetAccuracy:
    ONNX = ROOT / "model_assets" / "ref_ldm_unet_external_kv.onnx"
    TOL = 0.05

    def test_fp16_vs_ort(self):
        _skip_no_cuda()
        _skip_no_model(self.ONNX)
        from custom_kernels.ref_ldm.ref_ldm_torch import RefLDMUNetTorch

        sess = _ort_cuda_session(self.ONNX)
        onnx_inputs = sess.get_inputs()
        out_name = sess.get_outputs()[0].name

        x = torch.randn(1, 16, 64, 64, dtype=torch.float32, device="cuda")
        ts = torch.tensor([500], dtype=torch.int64, device="cuda")

        feeds = {
            "x_noisy_plus_lq_latent": x.cpu().numpy(),
            "timesteps": ts.cpu().numpy(),
            "is_ref_flag_input": np.array([True], dtype=bool),
            "use_reference_exclusive_path_globally_input": np.array([True], dtype=bool),
        }
        for inp in onnx_inputs:
            if inp.name.endswith("_k_ext") or inp.name.endswith("_v_ext"):
                shape = tuple(
                    d if isinstance(d, int) and d > 0 else 1 for d in inp.shape
                )
                feeds[inp.name] = np.zeros(shape, dtype=np.float32)

        ort_out = sess.run([out_name], feeds)[0]

        model = (
            RefLDMUNetTorch.from_onnx(str(self.ONNX), compute_dtype=torch.float16)
            .cuda()
            .eval()
        )
        # Use empty kv_map (no reference)
        with torch.no_grad():
            pt_out = model(x, ts, kv_map={}, use_exclusive=False).float().cpu().numpy()

        diff = np.abs(ort_out - pt_out).max()
        assert diff < self.TOL, f"max|diff|={diff:.4e} > tol={self.TOL}"
