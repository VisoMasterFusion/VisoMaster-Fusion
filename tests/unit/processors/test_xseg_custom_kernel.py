"""Tests for the XSeg custom PyTorch kernel (xseg_torch.py).

Covers:
  - Architecture sanity: parameter counts, module counts, output shape.
  - Numerical accuracy: XSegTorch (fp32, CPU) output vs. OnnxRuntime reference.
    Requires model_assets/XSeg_model.onnx; skipped when absent.
  - CUDA graph correctness: different inputs must produce different outputs
    (regression for the non_blocking=True race condition bug).
    Runs only when CUDA is available.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

# Project root so custom_kernels is importable
_ROOT = Path(__file__).parent.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ---------------------------------------------------------------------------
# Stub only missing heavy deps that custom_kernels modules try to import.
# Do NOT stub onnx/onnxruntime — they are installed and needed for accuracy tests.
# ---------------------------------------------------------------------------
_stub_list: list[str] = []
for _m in _stub_list:  # nothing to stub for xseg_torch itself
    if _m not in sys.modules:
        sys.modules[_m] = MagicMock()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_ONNX_PATH = _ROOT / "model_assets" / "XSeg_model.onnx"

_have_onnx = False
_have_ort = False
try:
    import onnx as _onnx_pkg  # noqa: F401

    _have_onnx = hasattr(_onnx_pkg, "__version__")
except Exception:
    pass

try:
    import onnxruntime as _ort_pkg  # noqa: F401

    _have_ort = hasattr(_ort_pkg, "__version__")
except Exception:
    pass

_have_model_file = _ONNX_PATH.exists()

# ---------------------------------------------------------------------------
# Architecture tests (no model file needed)
# ---------------------------------------------------------------------------


class TestXSegTorchArchitecture:
    """Structural tests that do not require the ONNX model file."""

    def _make_model(self):
        from custom_kernels.xseg.xseg_torch import XSegTorch

        m = XSegTorch()
        m.eval()
        return m

    def test_output_shape(self):
        """Forward pass with random float32 input → (1,1,256,256) float32."""
        m = self._make_model()
        x = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            y = m(x)
        assert y.shape == (1, 1, 256, 256), f"Unexpected output shape {y.shape}"
        assert y.dtype == torch.float32

    def test_output_sigmoid_range(self):
        """Sigmoid output must be in [0, 1]."""
        m = self._make_model()
        x = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            y = m(x)
        assert float(y.min()) >= -1e-6
        assert float(y.max()) <= 1.0 + 1e-6

    def test_conv_module_count(self):
        """Exactly 43 Conv/ConvTranspose modules must be present."""
        from custom_kernels.xseg.xseg_torch import (
            XSegTorch,
            _conv_modules_in_forward_order,
        )

        m = XSegTorch()
        mods = _conv_modules_in_forward_order(m)
        assert len(mods) == 43, f"Expected 43 conv modules, got {len(mods)}"

    def test_norm_module_count(self):
        """Exactly 36 _RMSNormMax modules must be present."""
        from custom_kernels.xseg.xseg_torch import (
            XSegTorch,
            _rms_norm_mods_in_forward_order,
        )

        m = XSegTorch()
        mods = _rms_norm_mods_in_forward_order(m)
        assert len(mods) == 36, f"Expected 36 norm modules, got {len(mods)}"

    def test_ct_bias_count(self):
        """Exactly 6 ConvTranspose additive bias parameters."""
        from custom_kernels.xseg.xseg_torch import (
            XSegTorch,
            _ct_bias_params_in_forward_order,
        )

        m = XSegTorch()
        biases = _ct_bias_params_in_forward_order(m)
        assert len(biases) == 6, f"Expected 6 CT biases, got {len(biases)}"

    def test_all_encoder_skip_shapes(self):
        """Encoder skip features must have the expected spatial sizes."""
        from custom_kernels.xseg.xseg_torch import XSegTorch

        m = XSegTorch()
        m.eval()
        x = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            x0, s0 = m.enc0(x)
            x1, s1 = m.enc1(x0)
            x2, s2 = m.enc2(x1)
            x3, s3 = m.enc3(x2)
            x4, s4 = m.enc4(x3)
            x5, s5 = m.enc5(x4)

        assert s0.shape == (1, 32, 256, 256), f"s0 {s0.shape}"
        assert s1.shape == (1, 64, 128, 128), f"s1 {s1.shape}"
        assert s2.shape == (1, 128, 64, 64), f"s2 {s2.shape}"
        assert s3.shape == (1, 256, 32, 32), f"s3 {s3.shape}"
        assert s4.shape == (1, 256, 16, 16), f"s4 {s4.shape}"
        assert s5.shape == (1, 256, 8, 8), f"s5 {s5.shape}"
        assert x5.shape == (1, 256, 4, 4), f"bottleneck {x5.shape}"

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="fp16 Triton path requires CUDA"
    )
    def test_fp16_forward(self):
        """Model in fp16 compute mode produces fp32 output (always-float32 contract)."""
        from custom_kernels.xseg.xseg_torch import XSegTorch

        m = XSegTorch()
        m._compute_dtype = torch.float16
        m = m.half().cuda()
        m.eval()
        x = torch.randn(1, 3, 256, 256, device="cuda")
        with torch.no_grad():
            y = m(x)
        assert y.dtype == torch.float32, "Output must always be float32"
        assert y.shape == (1, 1, 256, 256)

    def test_rms_norm_max_pytorch_fallback(self):
        """_RMSNormMax PyTorch path: output shape and dtype preserved."""
        from custom_kernels.xseg.xseg_torch import _RMSNormMax

        norm = _RMSNormMax(32)
        norm.eps = 1e-5
        x = torch.randn(1, 32, 64, 64)
        with torch.no_grad():
            y = norm(x)
        assert y.shape == x.shape
        assert y.dtype == x.dtype

    def test_rms_norm_max_formula(self):
        """_RMSNormMax output matches manually computed RMS normalisation."""
        from custom_kernels.xseg.xseg_torch import _RMSNormMax

        torch.manual_seed(42)
        C, H = 8, 4
        norm = _RMSNormMax(C)
        norm.eps = 1e-3
        with torch.no_grad():
            norm.gamma.fill_(2.0)
            norm.beta.fill_(0.5)
            norm.max_val.fill_(-1.0)  # floor below any plausible output

        x = torch.randn(1, C, H, H)
        with torch.no_grad():
            y_model = norm(x)

        # Manual computation
        x_f32 = x.float()
        rms = torch.sqrt(torch.mean(x_f32 * x_f32, dim=[2, 3], keepdim=True) + norm.eps)
        y_ref = (x_f32 / rms) * norm.gamma.float() + norm.beta.float()
        y_ref = torch.maximum(y_ref, norm.max_val.float())

        assert torch.allclose(y_model, y_ref, atol=1e-5), (
            f"Max diff {(y_model - y_ref).abs().max().item():.2e}"
        )

    def test_decoder_up_block_output_size(self):
        """_UpBlock doubles spatial dimensions (ConvTranspose 3x3 stride-2)."""
        from custom_kernels.xseg.xseg_torch import _UpBlock

        up = _UpBlock(256, 128)
        up.eval()
        x = torch.randn(1, 256, 4, 4)
        with torch.no_grad():
            y = up(x)
        assert y.shape == (1, 128, 8, 8), f"Expected (1,128,8,8), got {y.shape}"


# ---------------------------------------------------------------------------
# Numerical accuracy vs ORT (requires model file + onnx + onnxruntime)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _have_model_file, reason="XSeg_model.onnx not found in model_assets/"
)
@pytest.mark.skipif(not _have_onnx, reason="onnx package not available")
@pytest.mark.skipif(not _have_ort, reason="onnxruntime not available")
class TestXSegTorchVsORT:
    """Compare XSegTorch (fp32, CPU) against OnnxRuntime for random inputs."""

    @pytest.fixture(scope="class")
    def ort_session(self):
        import onnxruntime as ort

        sess = ort.InferenceSession(str(_ONNX_PATH), providers=["CPUExecutionProvider"])
        return sess

    @pytest.fixture(scope="class")
    def torch_model(self):
        from custom_kernels.xseg.xseg_torch import XSegTorch

        model = XSegTorch.from_onnx(str(_ONNX_PATH), compute_dtype=torch.float32)
        model.eval()
        return model

    @pytest.mark.parametrize("seed", [0, 1, 2, 7, 42])
    def test_output_matches_ort(self, ort_session, torch_model, seed):
        """XSegTorch (fp32) and ORT must agree within fp32 tolerance."""
        input_name = ort_session.get_inputs()[0].name
        torch.manual_seed(seed)
        np.random.seed(seed)
        x_np = np.random.rand(1, 3, 256, 256).astype(np.float32)
        x_t = torch.from_numpy(x_np)

        ort_out = ort_session.run(None, {input_name: x_np})[0]  # (1,1,256,256)
        with torch.no_grad():
            torch_out = torch_model(x_t).numpy()

        max_diff = float(np.abs(ort_out - torch_out).max())
        mean_diff = float(np.abs(ort_out - torch_out).mean())

        assert max_diff < 1e-3, (
            f"seed={seed}: max_diff={max_diff:.2e} mean_diff={mean_diff:.2e} — "
            "XSegTorch output diverges from ORT reference"
        )

    def test_output_shape_matches_ort(self, ort_session, torch_model):
        """ORT and XSegTorch must produce the same output shape."""
        input_name = ort_session.get_inputs()[0].name
        x_np = np.zeros((1, 3, 256, 256), dtype=np.float32)
        ort_out = ort_session.run(None, {input_name: x_np})[0]
        x_t = torch.zeros(1, 3, 256, 256)
        with torch.no_grad():
            torch_out = torch_model(x_t).numpy()
        assert ort_out.shape == torch_out.shape, (
            f"Shape mismatch: ORT {ort_out.shape} vs Torch {torch_out.shape}"
        )

    def test_face_input_foreground_is_high(self, ort_session, torch_model):
        """Both models should agree on a bright-centre input."""
        input_name = ort_session.get_inputs()[0].name
        x_np = np.zeros((1, 3, 256, 256), dtype=np.float32)
        x_np[:, :, 64:192, 64:192] = 0.8
        x_t = torch.from_numpy(x_np.copy())
        ort_out = ort_session.run(None, {input_name: x_np})[0]
        with torch.no_grad():
            torch_out = torch_model(x_t).numpy()
        max_diff = float(np.abs(ort_out - torch_out).max())
        assert max_diff < 1e-3, f"Bright-centre input diverges: max_diff={max_diff:.2e}"


# ---------------------------------------------------------------------------
# CUDA graph correctness (requires CUDA + model file + onnx)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(
    not _have_model_file, reason="XSeg_model.onnx not found in model_assets/"
)
@pytest.mark.skipif(not _have_onnx, reason="onnx package not available")
class TestXSegCudaGraph:
    """CUDA-graph runner must produce input-dependent, consistent outputs.

    These tests catch the non_blocking=True race condition where the graph
    executes on stale self._inp data, causing every call to return the same
    (wrong) output regardless of the actual input.
    """

    @pytest.fixture(scope="class")
    def cuda_runner(self):
        from custom_kernels.xseg.xseg_torch import XSegTorch, build_cuda_graph_runner

        model = (
            XSegTorch.from_onnx(str(_ONNX_PATH), compute_dtype=torch.float16)
            .to("cuda")
            .eval()
        )
        runner = build_cuda_graph_runner(model, warmup=3)
        return runner

    def test_different_inputs_give_different_outputs(self, cuda_runner):
        """Two distinct inputs must produce distinct outputs.

        If non_blocking=True is present, both calls may run the graph on the
        same stale self._inp (e.g. zeros), producing identical outputs for
        all inputs — the regression this test catches.
        """
        torch.manual_seed(0)
        x1 = torch.randn(1, 3, 256, 256, device="cuda")
        x2 = torch.randn(1, 3, 256, 256, device="cuda") * 2.0 + 0.5

        out1 = cuda_runner(x1).cpu()
        out2 = cuda_runner(x2).cpu()

        max_diff = float((out1 - out2).abs().max())
        assert max_diff > 0.01, (
            f"Outputs for different inputs are identical (max_diff={max_diff:.2e}). "
            "Likely non_blocking=True race condition: graph runs on stale input."
        )

    def test_same_input_gives_same_output(self, cuda_runner):
        """The same input must always produce the same output (determinism)."""
        torch.manual_seed(123)
        x = torch.randn(1, 3, 256, 256, device="cuda")

        out_a = cuda_runner(x).cpu()
        out_b = cuda_runner(x).cpu()

        max_diff = float((out_a - out_b).abs().max())
        assert max_diff < 1e-5, (
            f"Same input gives different outputs (max_diff={max_diff:.2e}). "
            "Non-deterministic CUDA graph or output race condition."
        )

    def test_output_range(self, cuda_runner):
        """CUDA graph runner output must be in [0, 1] (sigmoid)."""
        x = torch.rand(1, 3, 256, 256, device="cuda")
        out = cuda_runner(x).cpu()
        assert float(out.min()) >= -1e-4, f"Output below 0: {out.min()}"
        assert float(out.max()) <= 1.0 + 1e-4, f"Output above 1: {out.max()}"

    @pytest.mark.skipif(not _have_ort, reason="onnxruntime not available")
    def test_cuda_graph_matches_eager_fp32(self, cuda_runner):
        """CUDA graph (fp16) output must be close to eager fp32 model."""
        from custom_kernels.xseg.xseg_torch import XSegTorch

        model_fp32 = (
            XSegTorch.from_onnx(str(_ONNX_PATH), compute_dtype=torch.float32)
            .to("cuda")
            .eval()
        )

        torch.manual_seed(7)
        x = torch.randn(1, 3, 256, 256, device="cuda")

        with torch.no_grad():
            out_fp32 = model_fp32(x).cpu()
        out_graph = cuda_runner(x).cpu()

        max_diff = float((out_fp32 - out_graph).abs().max())
        assert max_diff < 0.05, (
            f"CUDA graph (fp16) diverges too much from fp32 eager "
            f"(max_diff={max_diff:.3f}). Check weight loading or Triton kernels."
        )
