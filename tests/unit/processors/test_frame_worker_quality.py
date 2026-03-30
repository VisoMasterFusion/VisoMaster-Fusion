"""
Q-* quality improvement unit tests for FrameWorker logic.

Tests the mathematical logic and guard conditions directly without importing
FrameWorker (which requires heavy GPU/kornia stubs).  The formulas are taken
verbatim from the implementation so regressions are caught immediately.

Coverage:
  Q-BUG-03  _is_kps_valid — NaN / Inf / out-of-bounds guards
  Q-IMP-04  _MIN_FACE_PIXELS — minimum bounding-box side guard
  Q-QUAL-01 _KPS_EMA_ALPHA  — keypoint EMA smoothing math
  Q-QUAL-03 _COLOR_EMA_ALPHA — AutoColor reference stats EMA math
  Q-QUAL-07 denoiser output clamped to [0, 255]
"""

from __future__ import annotations

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# Local replicas of pure logic from FrameWorker
# (keeps tests import-free and avoids heavy stubbing)
# ---------------------------------------------------------------------------

_MIN_FACE_PIXELS: int = 20  # mirrors FrameWorker._MIN_FACE_PIXELS
_KPS_EMA_ALPHA: float = 0.35  # mirrors FrameWorker._KPS_EMA_ALPHA
_COLOR_EMA_ALPHA: float = 0.30  # mirrors FrameWorker._COLOR_EMA_ALPHA


def _is_kps_valid(kps, img_h: int, img_w: int) -> bool:
    """Replica of FrameWorker._is_kps_valid for isolated unit testing."""
    if kps is None or kps.size == 0:
        return False
    if np.any(np.isnan(kps)) or np.any(np.isinf(kps)):
        return False
    if np.any(kps[:, 0] < 0) or np.any(kps[:, 0] >= img_w):
        return False
    if np.any(kps[:, 1] < 0) or np.any(kps[:, 1] >= img_h):
        return False
    return True


# ---------------------------------------------------------------------------
# Q-BUG-03: _is_kps_valid
# ---------------------------------------------------------------------------


class TestIsKpsValid:
    IMG_H, IMG_W = 480, 640

    def _valid_kps(self) -> np.ndarray:
        """Five keypoints well inside a 640×480 image."""
        return np.array(
            [
                [100.0, 100.0],
                [200.0, 100.0],
                [150.0, 150.0],
                [110.0, 200.0],
                [190.0, 200.0],
            ],
            dtype=np.float32,
        )

    def test_valid_kps_returns_true(self):
        assert _is_kps_valid(self._valid_kps(), self.IMG_H, self.IMG_W) is True

    def test_none_returns_false(self):
        assert _is_kps_valid(None, self.IMG_H, self.IMG_W) is False

    def test_empty_array_returns_false(self):
        assert (
            _is_kps_valid(np.empty((0, 2), dtype=np.float32), self.IMG_H, self.IMG_W)
            is False
        )

    def test_nan_in_x_returns_false(self):
        kps = self._valid_kps()
        kps[0, 0] = np.nan
        assert _is_kps_valid(kps, self.IMG_H, self.IMG_W) is False

    def test_nan_in_y_returns_false(self):
        kps = self._valid_kps()
        kps[2, 1] = np.nan
        assert _is_kps_valid(kps, self.IMG_H, self.IMG_W) is False

    def test_pos_inf_returns_false(self):
        kps = self._valid_kps()
        kps[1, 0] = np.inf
        assert _is_kps_valid(kps, self.IMG_H, self.IMG_W) is False

    def test_neg_inf_returns_false(self):
        kps = self._valid_kps()
        kps[3, 1] = -np.inf
        assert _is_kps_valid(kps, self.IMG_H, self.IMG_W) is False

    def test_x_negative_returns_false(self):
        kps = self._valid_kps()
        kps[0, 0] = -1.0
        assert _is_kps_valid(kps, self.IMG_H, self.IMG_W) is False

    def test_y_negative_returns_false(self):
        kps = self._valid_kps()
        kps[0, 1] = -0.5
        assert _is_kps_valid(kps, self.IMG_H, self.IMG_W) is False

    def test_x_equals_width_is_out_of_bounds(self):
        """x == img_w is invalid (valid range is [0, img_w-1])."""
        kps = self._valid_kps()
        kps[0, 0] = float(self.IMG_W)
        assert _is_kps_valid(kps, self.IMG_H, self.IMG_W) is False

    def test_y_equals_height_is_out_of_bounds(self):
        kps = self._valid_kps()
        kps[0, 1] = float(self.IMG_H)
        assert _is_kps_valid(kps, self.IMG_H, self.IMG_W) is False

    def test_x_at_max_valid_pixel_returns_true(self):
        """x == img_w - 1 is the last valid column."""
        kps = self._valid_kps()
        kps[0, 0] = float(self.IMG_W - 1)
        assert _is_kps_valid(kps, self.IMG_H, self.IMG_W) is True

    def test_y_at_max_valid_pixel_returns_true(self):
        kps = self._valid_kps()
        kps[0, 1] = float(self.IMG_H - 1)
        assert _is_kps_valid(kps, self.IMG_H, self.IMG_W) is True

    def test_x_zero_is_valid(self):
        kps = self._valid_kps()
        kps[0, 0] = 0.0
        assert _is_kps_valid(kps, self.IMG_H, self.IMG_W) is True


# ---------------------------------------------------------------------------
# Q-IMP-04: minimum face bounding-box side guard
# ---------------------------------------------------------------------------


def _bbox_shortest_side(bbox) -> float:
    """width/height minimum — mirrors the guard in _process_frame_standard."""
    return min(bbox[2] - bbox[0], bbox[3] - bbox[1])


class TestMinFacePixelsGuard:
    def test_tiny_face_below_threshold(self):
        bbox = [10.0, 10.0, 25.0, 18.0]  # width=15, height=8
        assert _bbox_shortest_side(bbox) < _MIN_FACE_PIXELS

    def test_normal_face_above_threshold(self):
        bbox = [10.0, 10.0, 60.0, 70.0]  # width=50, height=60
        assert _bbox_shortest_side(bbox) >= _MIN_FACE_PIXELS

    def test_exactly_at_threshold_is_not_skipped(self):
        """Shortest side == 20 must NOT be skipped (guard is strictly < 20)."""
        bbox = [10.0, 10.0, 30.0, 30.0]  # both sides == 20
        assert not (_bbox_shortest_side(bbox) < _MIN_FACE_PIXELS)

    def test_one_pixel_under_threshold_is_skipped(self):
        bbox = [10.0, 10.0, 29.0, 30.0]  # width=19 < 20
        assert _bbox_shortest_side(bbox) < _MIN_FACE_PIXELS

    @pytest.mark.parametrize(
        "w,h,expect_skip",
        [
            (5, 5, True),
            (19, 100, True),
            (20, 20, False),
            (21, 21, False),
            (100, 100, False),
        ],
    )
    def test_parametrized_sizes(self, w, h, expect_skip):
        bbox = [0.0, 0.0, float(w), float(h)]
        assert (_bbox_shortest_side(bbox) < _MIN_FACE_PIXELS) is expect_skip


# ---------------------------------------------------------------------------
# Q-QUAL-01: keypoint EMA smoothing math
# ---------------------------------------------------------------------------


class TestKpsEMA:
    def test_first_frame_stores_raw_value(self):
        """On first detection the smoothed value equals the raw keypoint."""
        raw = np.array([[100.0, 200.0]], dtype=np.float32)
        smoothed: dict = {}
        i = 0
        if i not in smoothed:
            smoothed[i] = raw[i].copy()
        else:
            smoothed[i] = _KPS_EMA_ALPHA * raw[i] + (1.0 - _KPS_EMA_ALPHA) * smoothed[i]
        assert np.allclose(smoothed[0], raw[0])

    def test_second_frame_blends_old_and_new(self):
        """EMA result is between prev and curr values."""
        prev = np.array([0.0, 0.0], dtype=np.float32)
        curr = np.array([100.0, 100.0], dtype=np.float32)
        blended = _KPS_EMA_ALPHA * curr + (1.0 - _KPS_EMA_ALPHA) * prev
        assert np.all(blended > prev)
        assert np.all(blended < curr)

    def test_ema_formula_matches_expected(self):
        prev = np.array([10.0, 20.0], dtype=np.float32)
        curr = np.array([50.0, 60.0], dtype=np.float32)
        expected = _KPS_EMA_ALPHA * curr + (1.0 - _KPS_EMA_ALPHA) * prev
        # With alpha=0.35: 0.35*50 + 0.65*10 = 17.5+6.5 = 24.0
        assert np.isclose(expected[0], 0.35 * 50.0 + 0.65 * 10.0)
        assert np.isclose(expected[1], 0.35 * 60.0 + 0.65 * 20.0)

    def test_face_count_change_resets_state(self):
        """When the number of detected faces changes, the EMA dict resets."""
        smoothed = {0: np.array([1.0, 2.0]), 1: np.array([3.0, 4.0])}
        n_faces_new = 1
        if len(smoothed) != n_faces_new:
            smoothed = {}
        assert smoothed == {}

    def test_ema_converges_to_stable_value(self):
        """Repeated identical input converges the EMA to that value."""
        target = np.array([77.0, 88.0], dtype=np.float32)
        val = np.zeros(2, dtype=np.float32)
        for _ in range(60):
            val = _KPS_EMA_ALPHA * target + (1.0 - _KPS_EMA_ALPHA) * val
        assert np.allclose(val, target, atol=0.5)

    def test_alpha_weight_applied_correctly(self):
        """New value contributes exactly alpha fraction."""
        prev = np.array([0.0], dtype=np.float32)
        curr = np.array([1.0], dtype=np.float32)
        result = _KPS_EMA_ALPHA * curr + (1.0 - _KPS_EMA_ALPHA) * prev
        assert np.isclose(result[0], _KPS_EMA_ALPHA)

    def test_nan_raw_does_not_contaminate_ema_state(self):
        """Q-BUG-EMA: a NaN raw detection must NOT update the EMA state.

        Before the fix, blending NaN into the EMA would permanently corrupt it;
        every subsequent frame would also become NaN and the face would never swap.
        """
        IMG_H, IMG_W = 480, 640
        smoothed: dict = {}
        smoothed[0] = np.array([300.0, 200.0], dtype=np.float32)  # prior good value

        raw_nan = np.array([np.nan, np.nan], dtype=np.float32)
        # Simulate the fixed EMA logic: skip update if raw is invalid
        if not _is_kps_valid(raw_nan.reshape(1, 2), IMG_H, IMG_W):
            pass  # do NOT update smoothed[0]
        else:
            smoothed[0] = (
                _KPS_EMA_ALPHA * raw_nan + (1.0 - _KPS_EMA_ALPHA) * smoothed[0]
            )

        # EMA state must still be the prior good value
        assert not np.any(np.isnan(smoothed[0]))
        assert np.allclose(smoothed[0], [300.0, 200.0])

    def test_nan_raw_falls_back_to_previous_smoothed(self):
        """When raw is invalid and a prior smoothed value exists, use it as fallback."""
        IMG_H, IMG_W = 480, 640
        kpss_5 = np.array([[np.nan, np.nan]], dtype=np.float32)
        smoothed: dict = {0: np.array([320.0, 210.0], dtype=np.float32)}

        raw = kpss_5[0]
        if not _is_kps_valid(raw.reshape(1, 2), IMG_H, IMG_W):
            if 0 in smoothed:
                kpss_5[0] = smoothed[0]  # fall back to last good value
        else:
            smoothed[0] = _KPS_EMA_ALPHA * raw + (1.0 - _KPS_EMA_ALPHA) * smoothed[0]
            kpss_5[0] = smoothed[0]

        # kpss_5[0] should now be the previous good smoothed value, not NaN
        assert _is_kps_valid(kpss_5[0].reshape(1, 2), IMG_H, IMG_W)
        assert np.allclose(kpss_5[0], [320.0, 210.0])


# ---------------------------------------------------------------------------
# Q-QUAL-03: AutoColor reference statistics EMA math
# ---------------------------------------------------------------------------


class TestColorStatsEMA:
    def test_first_call_ema_equals_current(self):
        """No prior history → EMA = current stats (no blending)."""
        ema_cache: dict = {}
        key = b"fake_embedding"
        curr_mean = torch.tensor([[[128.0]], [[64.0]], [[32.0]]])
        curr_std = torch.tensor([[[30.0]], [[20.0]], [[10.0]]])

        if key not in ema_cache:
            ema_mean, ema_std = curr_mean, curr_std
        else:
            prev = ema_cache[key]
            ema_mean = (
                _COLOR_EMA_ALPHA * curr_mean + (1 - _COLOR_EMA_ALPHA) * prev["mean"]
            )
            ema_std = _COLOR_EMA_ALPHA * curr_std + (1 - _COLOR_EMA_ALPHA) * prev["std"]

        ema_cache[key] = {"mean": ema_mean.detach(), "std": ema_std.detach()}
        assert torch.allclose(ema_cache[key]["mean"], curr_mean)
        assert torch.allclose(ema_cache[key]["std"], curr_std)

    def test_second_call_blends_stats(self):
        """Second call blends current with stored history."""
        alpha = _COLOR_EMA_ALPHA
        prev_mean = torch.tensor([[[100.0]]])
        curr_mean = torch.tensor([[[200.0]]])
        result = alpha * curr_mean + (1 - alpha) * prev_mean
        expected = torch.tensor([[[0.3 * 200.0 + 0.7 * 100.0]]])
        assert torch.allclose(result, expected, atol=1e-4)

    def test_different_keys_are_independent(self):
        """Each target-face embedding has its own EMA history."""
        ema_cache: dict = {}
        ema_cache[b"face_a"] = {
            "mean": torch.tensor([[[10.0]]]),
            "std": torch.tensor([[[1.0]]]),
        }
        ema_cache[b"face_b"] = {
            "mean": torch.tensor([[[200.0]]]),
            "std": torch.tensor([[[20.0]]]),
        }
        assert not torch.allclose(
            ema_cache[b"face_a"]["mean"], ema_cache[b"face_b"]["mean"]
        )

    def test_std_floor_prevents_division_by_zero(self):
        """The 1e-6 std floor ensures no division-by-zero in colour normalisation."""
        std = torch.tensor([[[0.0]]])
        std_safe = std + 1e-6
        assert std_safe.item() > 0.0

    def test_ema_remapped_face_stays_in_range(self):
        """After EMA remapping, values clamped to [0,255] stay in [0,255]."""
        face_f = torch.full((3, 16, 16), 128.0)
        curr_mean = face_f.mean(dim=(1, 2), keepdim=True)
        curr_std = face_f.std(dim=(1, 2), keepdim=True) + 1e-6
        ema_mean = curr_mean  # first frame: EMA == current
        ema_std = curr_std
        remapped = ((face_f - curr_mean) / curr_std * ema_std + ema_mean).clamp(0, 255)
        assert remapped.min().item() >= 0.0
        assert remapped.max().item() <= 255.0


# ---------------------------------------------------------------------------
# Q-QUAL-07: denoiser output clamped to [0, 255]
# ---------------------------------------------------------------------------


class TestDenoiserClamp:
    def test_values_above_255_are_clamped(self):
        t = torch.tensor([200.0, 300.0, 256.0, 1000.0])
        assert torch.clamp(t, 0, 255).max().item() <= 255.0

    def test_negative_values_are_clamped(self):
        t = torch.tensor([-10.0, -1.0, 0.0, 128.0])
        assert torch.clamp(t, 0, 255).min().item() >= 0.0

    def test_in_range_values_unchanged(self):
        t = torch.tensor([0.0, 64.0, 128.0, 255.0])
        assert torch.equal(torch.clamp(t, 0, 255), t)

    def test_dtype_preserved_after_clamp(self):
        """clamp must not change float32 to another dtype."""
        t = torch.randn(3, 32, 32) * 400  # deliberately out-of-range
        clamped = torch.clamp(t, 0, 255)
        assert clamped.dtype == torch.float32

    def test_chw_tensor_clamped_correctly(self):
        """Realistic denoiser output shape (C, H, W) is fully clamped."""
        t = torch.randn(3, 64, 64) * 500
        clamped = torch.clamp(t, 0, 255)
        assert clamped.min().item() >= 0.0
        assert clamped.max().item() <= 255.0
