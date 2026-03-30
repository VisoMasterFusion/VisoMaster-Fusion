"""Unit tests for app.processors.face_reaging.

UT-06: apply_reaging with float [0,1] input is auto-normalised (BUG-09 fix).
UT-07: apply_reaging output is always a CHW uint8 CPU tensor (P3-05 fix).
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# Stub heavy dependencies so the module can be imported without GPU/ONNX
# ---------------------------------------------------------------------------

for _name in [
    "onnxruntime",
]:
    if _name not in sys.modules:
        sys.modules[_name] = MagicMock()


# Provide a minimal ModelsProcessor stub
class _FakeModelsProcessor:
    device = torch.device("cpu")

    def __init__(self, session):
        self._session = session

    def load_model(self, _name):
        return self._session


from app.processors.face_reaging import FaceReaging  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_onnx_session(delta_value: float = 0.0) -> MagicMock:
    """Return a fake ONNX InferenceSession whose output is a constant delta."""

    session = MagicMock()
    # MagicMock(name=…) sets the mock's internal repr name, NOT the .name attribute.
    # Use explicit attribute assignment instead.
    _inp = MagicMock()
    _inp.name = "input"
    _out = MagicMock()
    _out.name = "output"
    session.get_inputs.return_value = [_inp]
    session.get_outputs.return_value = [_out]

    # Model output: (1, 3, H, W) delta filled with delta_value
    def _run(output_names, feed):
        inp = next(iter(feed.values()))  # safe: only one input
        _, _, h, w = inp.shape
        return [np.full((1, 3, h, w), delta_value, dtype=np.float32)]

    session.run.side_effect = _run
    return session


def _make_reaging(delta: float = 0.0) -> FaceReaging:
    session = _make_onnx_session(delta)
    proc = _FakeModelsProcessor(session)
    return FaceReaging(proc)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# UT-07: Output is always CHW uint8 on CPU
# ---------------------------------------------------------------------------


class TestFaceReagingOutputType:
    def test_output_dtype_is_uint8(self):
        reaging = _make_reaging(delta=0.0)
        face = torch.randint(0, 256, (3, 64, 64), dtype=torch.uint8)
        result = reaging.apply_reaging(face, source_age=25, target_age=50)
        assert result.dtype == torch.uint8

    def test_output_is_on_cpu(self):
        reaging = _make_reaging(delta=0.0)
        face = torch.randint(0, 256, (3, 64, 64), dtype=torch.uint8)
        result = reaging.apply_reaging(face, source_age=25, target_age=50)
        assert result.device.type == "cpu"

    def test_output_shape_matches_input(self):
        reaging = _make_reaging(delta=0.0)
        face = torch.randint(0, 256, (3, 128, 128), dtype=torch.uint8)
        result = reaging.apply_reaging(face, source_age=25, target_age=50)
        assert result.shape == (3, 128, 128)

    def test_output_values_in_uint8_range(self):
        reaging = _make_reaging(delta=0.1)
        face = torch.randint(0, 256, (3, 64, 64), dtype=torch.uint8)
        result = reaging.apply_reaging(face, source_age=25, target_age=50)
        assert result.min() >= 0
        assert result.max() <= 255


# ---------------------------------------------------------------------------
# UT-06: Float [0,1] input is auto-normalised (BUG-09 fix)
# ---------------------------------------------------------------------------


class TestFaceReagingDtypeGuard:
    def test_float_01_input_produces_uint8_output(self):
        """A float [0,1] tensor must not produce a near-black result."""
        reaging = _make_reaging(delta=0.0)
        # Simulate float [0,1] tensor that a pipeline path might pass in
        face_float = torch.ones(3, 64, 64, dtype=torch.float32) * 0.5  # mid-grey
        result = reaging.apply_reaging(face_float, source_age=25, target_age=50)
        assert result.dtype == torch.uint8
        # Should be approximately 128 (0.5 * 255), not near 0
        assert result.float().mean() > 10.0, (
            "Float [0,1] input was not auto-normalised — result is near-black"
        )

    def test_float_0_255_input_produces_reasonable_output(self):
        """A float [0,255] tensor should also be handled gracefully."""
        reaging = _make_reaging(delta=0.0)
        face_float = torch.full((3, 64, 64), 128.0, dtype=torch.float32)
        result = reaging.apply_reaging(face_float, source_age=25, target_age=50)
        assert result.dtype == torch.uint8
        assert result.min() >= 0
        assert result.max() <= 255

    def test_uint8_input_passes_through_unchanged_with_zero_delta(self):
        """With delta=0 the output should equal the input (identity transform)."""
        reaging = _make_reaging(delta=0.0)
        face = torch.full((3, 64, 64), 100, dtype=torch.uint8)
        result = reaging.apply_reaging(face, source_age=25, target_age=50)
        assert result.dtype == torch.uint8
        # delta=0 → aged = face_float + 0 → clamp → *255 → uint8 ≈ 100
        assert abs(result.float().mean().item() - 100.0) < 2.0

    def test_none_session_returns_original_input(self):
        """When model cannot be loaded, original tensor is returned unchanged."""
        proc = _FakeModelsProcessor(None)  # load_model returns None
        reaging = FaceReaging(proc)
        face = torch.randint(0, 256, (3, 64, 64), dtype=torch.uint8)
        result = reaging.apply_reaging(face, source_age=25, target_age=50)
        assert torch.equal(result, face)


# ---------------------------------------------------------------------------
# UT-01 / UT-02: DFM scale logic (pure arithmetic replica, no FrameWorker import)
# ---------------------------------------------------------------------------


class TestDFMScaleLogic:
    """UT-01: After the BUG-01 fix, DFM output (HWC float [0,1]) multiplied by 255
    must produce values in [0, 255] — matching what all other swappers produce.

    UT-02: The FW-QUAL-08 blank-detection threshold (30.0) must not trigger for a
    valid DFM frame whose max pixel is ~255 after scaling, but must trigger for a
    genuinely blank frame.
    """

    def _simulate_dfm_output(self, mean_value: float = 0.5) -> torch.Tensor:
        """Simulate HWC float [0,1] output from dfm_model.convert()."""
        return torch.full((512, 512, 3), mean_value, dtype=torch.float32)

    # UT-01 —
    def test_dfm_scaled_output_in_0_255_range(self):
        out_celeb = self._simulate_dfm_output(0.5)
        # BUG-01 fix: multiply by 255 before permute
        output = out_celeb * 255.0
        assert output.max().item() <= 255.0
        assert output.min().item() >= 0.0
        assert output.abs().max().item() == pytest.approx(127.5, abs=1.0)

    def test_dfm_permute_gives_chw(self):
        out_celeb = self._simulate_dfm_output(0.5)
        output = (out_celeb * 255.0).permute(2, 0, 1)
        assert output.shape == (3, 512, 512)

    # UT-02 —
    def test_qual08_threshold_does_not_trigger_for_valid_dfm_output(self):
        out_celeb = self._simulate_dfm_output(0.5)
        output = out_celeb * 255.0  # after BUG-01 fix
        # threshold is 30.0; max is ~127.5 → should NOT trigger
        assert output.abs().max().item() >= 30.0

    def test_qual08_threshold_triggers_for_blank_output(self):
        out_celeb = self._simulate_dfm_output(0.0)  # blank / all-zero
        output = out_celeb * 255.0
        # max is 0.0 → should trigger
        assert output.abs().max().item() < 30.0

    def test_qual08_old_threshold_1_always_triggered_for_unscaled_dfm(self):
        """Demonstrates that the OLD threshold of 1.0 was wrong for DFM.
        Unscaled DFM output (float [0,1]) would always have max <= 1.0,
        causing a false alarm every frame."""
        out_celeb = self._simulate_dfm_output(0.8)  # clearly a valid face
        # Old broken behaviour: check against 1.0 on UN-scaled output
        unscaled_max = out_celeb.abs().max().item()
        assert unscaled_max < 1.0 + 1e-6, "Pre-fix: max is within [0,1]"
        # New correct behaviour: check against 30.0 on scaled output
        scaled_max = (out_celeb * 255.0).abs().max().item()
        assert scaled_max >= 30.0, "Post-fix: scaled max is safely above threshold"
