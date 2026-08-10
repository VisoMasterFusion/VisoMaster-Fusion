"""Unit tests for app.processors.perform_recast (PerformRecast "Recast" mode).

These exercise the pure logic — keypoint transform, the two composition modes,
region gating and the W->G warp_decode chaining — with fully mocked ONNX
sessions, so no GPU / real models are required.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

# perform_recast imports only numpy/torch (no onnxruntime at module level), so
# no stubbing is required to import it here.
from app.processors.perform_recast import (  # noqa: E402
    NUM_KP,
    MODE_ENHANCEMENT,
    MODE_REPLACEMENT,
    PerformRecast,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


def _fake_session(output_arrays: dict) -> MagicMock:
    """Build a fake InferenceSession returning the given {name: ndarray}.

    `get_outputs()` reports the names in insertion order; `run()` returns the
    arrays in the same order (computed lazily so per-call shapes can depend on
    feeds if a callable is supplied).
    """
    session = MagicMock()
    out_objs = []
    for name in output_arrays:
        o = MagicMock()
        o.name = name
        out_objs.append(o)
    session.get_outputs.return_value = out_objs

    def _run(_names, feeds):
        result = []
        for name, val in output_arrays.items():
            result.append(val(feeds) if callable(val) else val)
        return result

    session.run.side_effect = _run
    return session


class _FakeModelsProcessor:
    def __init__(self):
        self.device = torch.device("cpu")
        self.models: dict = {}
        self._sessions: dict = {}

    def register(self, name, session):
        self._sessions[name] = session

    def load_model(self, name):
        return self._sessions.get(name)

    def unload_model(self, name, force_immediate=False):
        self.models[name] = None


def _motion_outputs(seed: int) -> dict:
    """Deterministic motion-extractor outputs of the correct shapes."""
    rng = np.random.RandomState(seed)
    return {
        "pitch": rng.rand(1, 66).astype(np.float32),
        "yaw": rng.rand(1, 66).astype(np.float32),
        "roll": rng.rand(1, 66).astype(np.float32),
        "t": rng.rand(1, 3).astype(np.float32),
        "exp": rng.rand(1, NUM_KP * 3).astype(np.float32),
        "scale": (rng.rand(1, 3).astype(np.float32) + 0.5),
        "kp": rng.rand(1, NUM_KP * 3).astype(np.float32),
    }


def _make_recast() -> PerformRecast:
    mp = _FakeModelsProcessor()
    mp.register("PerformRecastMotionExtractor", _fake_session(_motion_outputs(0)))
    mp.register(
        "PerformRecastAppearanceFeatureExtractor",
        _fake_session({"feature_3d": np.zeros((1, 32, 16, 64, 64), dtype=np.float32)}),
    )
    mp.register(
        "PerformRecastWarpingModule",
        _fake_session(
            {
                "occlusion_map": np.zeros((1, 1, 64, 64), dtype=np.float32),
                "deformation": np.zeros((1, 16, 64, 64, 3), dtype=np.float32),
                "out": np.zeros((1, 256, 64, 64), dtype=np.float32),
            }
        ),
    )
    mp.register(
        "PerformRecastSpadeGenerator",
        _fake_session({"out": np.full((1, 3, 512, 512), 0.5, dtype=np.float32)}),
    )
    return PerformRecast(mp, None)  # type: ignore[arg-type]


def _torch_motion(seed: int) -> dict:
    out = _motion_outputs(seed)
    info = {k: torch.from_numpy(v) for k, v in out.items()}
    info["exp"] = info["exp"].reshape(1, -1, 3)
    info["kp"] = info["kp"].reshape(1, -1, 3)
    return info


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


class TestGeometry:
    def test_headpose_degree_formula(self):
        # All probability mass on bin 33 -> 33*3 - 99 == 0 degrees.
        logits = torch.full((1, 66), -1e4)
        logits[0, 33] = 1e4
        deg = PerformRecast._headpose_to_degree(logits)
        assert deg.shape == (1,)
        assert abs(deg.item() - 0.0) < 1e-3

    def test_headpose_uses_minus_99_not_975(self):
        # Mass on bin 0 -> 0*3 - 99 == -99 (PerformRecast convention).
        logits = torch.full((1, 66), -1e4)
        logits[0, 0] = 1e4
        deg = PerformRecast._headpose_to_degree(logits)
        assert abs(deg.item() - (-99.0)) < 1e-2

    def test_rotation_matrix_identity_at_zero(self):
        z = torch.zeros(1)
        R = PerformRecast._rotation_matrix(z, z, z)
        assert R.shape == (1, 3, 3)
        assert torch.allclose(R[0], torch.eye(3), atol=1e-5)

    def test_rotation_matrix_is_orthonormal(self):
        R = PerformRecast._rotation_matrix(
            torch.tensor([30.0]), torch.tensor([15.0]), torch.tensor([-20.0])
        )
        ident = R[0] @ R[0].T
        assert torch.allclose(ident, torch.eye(3), atol=1e-4)


# ---------------------------------------------------------------------------
# Source descriptor
# ---------------------------------------------------------------------------


class TestBuildSourceInfo:
    def test_shapes(self):
        recast = _make_recast()
        info = recast.build_source_info(_torch_motion(2))
        assert info["kp"].shape == (1, NUM_KP, 3)
        assert info["exp"].shape == (1, NUM_KP, 3)
        assert info["R"].shape == (1, 3, 3)
        assert info["x_s"].shape == (1, NUM_KP, 3)

    def test_tz_is_zeroed(self):
        recast = _make_recast()
        info = recast.build_source_info(_torch_motion(2))
        assert torch.all(info["t"][..., 2] == 0)


# ---------------------------------------------------------------------------
# Composition modes
# ---------------------------------------------------------------------------


class TestComposeDrivenKeypoints:
    def test_enhancement_factor_zero_equals_source(self):
        """factor=0 keeps the source expression -> driven kp == source x_s."""
        recast = _make_recast()
        info = recast.build_source_info(_torch_motion(3))
        exp_d = _torch_motion(99)["exp"]
        x_d = recast.compose_driven_keypoints(
            info, exp_d, mode=MODE_ENHANCEMENT, factor=0.0
        )
        assert torch.allclose(x_d, info["x_s"], atol=1e-5)

    def test_enhancement_factor_one_differs_from_source(self):
        recast = _make_recast()
        info = recast.build_source_info(_torch_motion(3))
        exp_d = _torch_motion(99)["exp"]
        x_d = recast.compose_driven_keypoints(
            info, exp_d, mode=MODE_ENHANCEMENT, factor=1.0
        )
        assert x_d.shape == (1, NUM_KP, 3)
        assert not torch.allclose(x_d, info["x_s"], atol=1e-4)

    def test_modes_produce_different_results(self):
        """Enhancement (additive) must differ from Replacement at factor=1 —
        the regression that made switching modes a no-op."""
        recast = _make_recast()
        info = recast.build_source_info(_torch_motion(13))
        exp_d = _torch_motion(77)["exp"]
        x_enh = recast.compose_driven_keypoints(
            info, exp_d, mode=MODE_ENHANCEMENT, factor=1.0
        )
        x_rep = recast.compose_driven_keypoints(
            info, exp_d, mode=MODE_REPLACEMENT, factor=1.0
        )
        assert not torch.allclose(x_enh, x_rep, atol=1e-4)

    def test_enhancement_is_additive(self):
        """Enhancement adds the driver's expression on top of the source."""
        recast = _make_recast()
        info = recast.build_source_info(_torch_motion(14))
        exp_d = _torch_motion(88)["exp"]
        x_d = recast.compose_driven_keypoints(
            info, exp_d, mode=MODE_ENHANCEMENT, factor=1.0
        )
        # Reconstruct expected: kp + (exp_s + exp_d) through the same geometry.
        kp_e = info["kp"] + (info["exp"] + exp_d)
        R, scale, t = info["R"], info["scale"], info["t"]
        kp_rot = torch.einsum("bmp,bkp->bkm", R, kp_e) * scale.unsqueeze(1)
        x_expected = kp_rot + t.unsqueeze(1)
        assert torch.allclose(x_d, x_expected, atol=1e-5)

    def test_replacement_returns_correct_shape(self):
        recast = _make_recast()
        info = recast.build_source_info(_torch_motion(4))
        exp_d = _torch_motion(7)["exp"]
        x_d = recast.compose_driven_keypoints(
            info, exp_d, mode=MODE_REPLACEMENT, factor=1.0
        )
        assert x_d.shape == (1, NUM_KP, 3)

    def test_unknown_mode_raises(self):
        recast = _make_recast()
        info = recast.build_source_info(_torch_motion(4))
        exp_d = _torch_motion(7)["exp"]
        with pytest.raises(ValueError):
            recast.compose_driven_keypoints(info, exp_d, mode="Bogus")

    def test_replacement_eye_lip_weight_zero_keeps_source(self):
        """eye/lip driving weight 0 -> the eye/lip planar channels stay at the
        source expression; weight 1 -> they follow the driver."""
        recast = _make_recast()
        info = recast.build_source_info(_torch_motion(8))
        exp_d = _torch_motion(123)["exp"]

        x_keep = recast.compose_driven_keypoints(
            info,
            exp_d,
            mode=MODE_REPLACEMENT,
            factor=1.0,
            eye_driving_weight=0.0,
            lip_driving_weight=0.0,
        )
        x_follow = recast.compose_driven_keypoints(
            info,
            exp_d,
            mode=MODE_REPLACEMENT,
            factor=1.0,
            eye_driving_weight=1.0,
            lip_driving_weight=1.0,
        )
        # With weight 0 the eye planar channels equal the source; with weight 1
        # they differ (driver), so the two compositions must differ on the eyes.
        assert not torch.allclose(x_keep[:, 31:34, :], x_follow[:, 31:34, :], atol=1e-4)

    def test_replacement_default_weights_match_legacy_constants(self):
        """Default weights (0.7 eyes / 0.8 lips) reproduce the original
        hardcoded 0.3/0.7 and 0.2/0.8 blend exactly."""
        recast = _make_recast()
        info = recast.build_source_info(_torch_motion(11))
        exp_s = info["exp"]
        exp_d = _torch_motion(222)["exp"]

        # Recompute the legacy modulated expression directly.
        modulated = exp_d.clone()
        modulated[:, 31:34, 2] = exp_s[:, 31:34, 2]
        modulated[:, 36:39, 2] = exp_s[:, 36:39, 2]
        modulated[:, 44:47, 2] = exp_s[:, 44:47, 2]
        modulated[:, 44:47, 0] = exp_s[:, 44:47, 0]
        modulated[:, 44:47, 1] = exp_s[:, 44:47, 1] * 0.2 + exp_d[:, 44:47, 1] * 0.8
        modulated[:, 31:34, :2] = exp_s[:, 31:34, :2] * 0.3 + exp_d[:, 31:34, :2] * 0.7
        modulated[:, 36:39, :2] = exp_s[:, 36:39, :2] * 0.3 + exp_d[:, 36:39, :2] * 0.7
        legacy_new_exp = exp_s + 1.0 * (modulated - exp_s)

        # The composed driven keypoints use the same transform, so compare the
        # internal new_exp by reconstructing it through a region='all' compose
        # against a hand-built reference transform.
        x_default = recast.compose_driven_keypoints(
            info, exp_d, mode=MODE_REPLACEMENT, factor=1.0
        )
        # Build expected x_d from legacy_new_exp using the same geometry.
        kp_e = info["kp"] + legacy_new_exp
        R, scale, t = info["R"], info["scale"], info["t"]
        kp_rot = torch.einsum("bmp,bkp->bkm", R, kp_e) * scale.unsqueeze(1)
        x_expected = kp_rot + t.unsqueeze(1)
        assert torch.allclose(x_default, x_expected, atol=1e-5)

    def test_region_eyes_keeps_non_eye_keypoints_at_source(self):
        """With region='eyes', keypoints outside the eye group are unchanged."""
        recast = _make_recast()
        info = recast.build_source_info(_torch_motion(5))
        exp_d = _torch_motion(50)["exp"]
        x_d = recast.compose_driven_keypoints(
            info, exp_d, mode=MODE_ENHANCEMENT, factor=1.0, region="eyes"
        )
        eye = recast.EYE_INDICES
        non_eye = [i for i in range(NUM_KP) if i not in eye]
        # Transform is applied per-keypoint independently, so untouched
        # expression channels map to the unchanged source keypoints.
        assert torch.allclose(x_d[:, non_eye, :], info["x_s"][:, non_eye, :], atol=1e-5)
        assert not torch.allclose(x_d[:, eye, :], info["x_s"][:, eye, :], atol=1e-4)


# ---------------------------------------------------------------------------
# Model-level API
# ---------------------------------------------------------------------------


class TestModelAPI:
    def test_motion_reshapes_exp_and_kp(self):
        recast = _make_recast()
        img = torch.randint(0, 256, (3, 256, 256), dtype=torch.uint8)
        info = recast.motion(img)
        assert info["exp"].shape == (1, NUM_KP, 3)
        assert info["kp"].shape == (1, NUM_KP, 3)
        assert info["scale"].shape == (1, 3)

    def test_extract_appearance_shape(self):
        recast = _make_recast()
        img = torch.randint(0, 256, (3, 512, 512), dtype=torch.uint8)
        feat = recast.extract_appearance(img)
        assert feat.shape == (1, 32, 16, 64, 64)

    def test_warp_decode_chains_w_then_g(self):
        recast = _make_recast()
        f_s = np.zeros((1, 32, 16, 64, 64), dtype=np.float32)
        x_s = torch.zeros(1, NUM_KP, 3)
        x_d = torch.zeros(1, NUM_KP, 3)
        out = recast.warp_decode(f_s, x_s, x_d)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (1, 3, 512, 512)

    def test_to_input_normalises_to_0_1_with_batch(self):
        arr = PerformRecast._to_input(torch.full((3, 8, 8), 255, dtype=torch.uint8))
        assert arr.shape == (1, 3, 8, 8)
        assert arr.dtype == np.float32
        assert pytest.approx(arr.max(), abs=1e-6) == 1.0

    def test_prewarm_loads_all_models(self):
        recast = _make_recast()
        assert all(
            recast.models_processor.models.get(n) is None for n in recast.model_group
        )
        recast.prewarm()
        for name in recast.model_group:
            assert recast.models_processor.models.get(name) is not None

    def test_unload_models_calls_unload_for_each(self):
        recast = _make_recast()
        # Pretend all four are loaded.
        for name in recast.model_group:
            recast.models_processor.models[name] = object()
        recast.unload_models()
        for name in recast.model_group:
            assert recast.models_processor.models[name] is None


# ---------------------------------------------------------------------------
# TensorRT readiness: the warping module needs shape inference (GridSample)
# ---------------------------------------------------------------------------

_WARPING_ONNX = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "..",
    "model_assets",
    "performrecast_onnx",
    "warping_module.onnx",
)


class TestTensorRTShapeInference:
    def test_warping_module_listed_for_shape_inference(self):
        from app.processors.models_data import tensorrt_shape_infer_models

        assert "PerformRecastWarpingModule" in tensorrt_shape_infer_models

    @pytest.mark.skipif(
        not os.path.exists(_WARPING_ONNX),
        reason="warping_module.onnx not downloaded",
    )
    def test_static_shape_inference_makes_gridsample_shapes_concrete(self):
        """The TRT 'has no shape specified' error is fixed once GridSample
        outputs gain concrete shapes after batch pinning + shape inference."""
        import onnx
        from onnxruntime.tools.onnx_model_utils import make_dim_param_fixed
        from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference

        model = onnx.load(_WARPING_ONNX)
        make_dim_param_fixed(model.graph, "batch", 1)
        model = SymbolicShapeInference.infer_shapes(
            model, auto_merge=True, guess_output_rank=True
        )
        grid_vis = [
            vi
            for vi in list(model.graph.value_info) + list(model.graph.output)
            if "GridSample" in vi.name
        ]
        assert grid_vis, "expected GridSample tensors in the warping module"
        for vi in grid_vis:
            dims = vi.type.tensor_type.shape.dim
            assert len(dims) > 0, f"{vi.name} still has no shape"
            # Every dim must be a concrete integer (no dim_param) for TRT.
            for d in dims:
                assert d.dim_value > 0 and not d.dim_param, (
                    f"{vi.name} has a non-static dim: {d}"
                )
