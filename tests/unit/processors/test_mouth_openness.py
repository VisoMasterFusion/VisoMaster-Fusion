"""Unit tests for app.processors.mouth_openness."""

from __future__ import annotations

import threading
import numpy as np
import pytest

from app.processors.mouth_openness import (
    MIN_MOUTH_SPAN_PX,
    OCCLUSION_TIMEOUT_FRAMES,
    MouthOpennessState,
    compute_lip_open_ratio_203,
    compute_lip_open_ratio_68,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_kps_203(
    n: int = 203, mouth_vert: float = 10.0, mouth_horiz: float = 50.0
) -> np.ndarray:
    """Build a dummy kps array for the 203-point case.

    203-pt relevant indices:
      48, 66 — horizontal mouth corners
      90, 102 — vertical lip points

    Note: kps[66] is the *horizontal* right-mouth-corner in the 203-pt model.
    """
    assert n >= 203
    kps = np.zeros((n, 2), dtype=np.float32)
    kps[48] = [0.0, 0.0]
    kps[66] = [mouth_horiz, 0.0]  # right mouth corner (horiz)
    kps[90] = [mouth_horiz / 2, 0.0]
    kps[102] = [mouth_horiz / 2, mouth_vert]
    return kps


def _make_kps_68(
    n: int = 68, mouth_vert: float = 10.0, mouth_horiz: float = 50.0
) -> np.ndarray:
    """Build a dummy kps array for the 68-point case.

    68-pt relevant indices:
      48, 54 — horizontal mouth corners
      62, 66 — vertical lip points
    """
    assert n >= 68
    kps = np.zeros((n, 2), dtype=np.float32)
    kps[48] = [0.0, 0.0]
    kps[54] = [mouth_horiz, 0.0]
    kps[62] = [mouth_horiz / 2, 0.0]
    kps[66] = [mouth_horiz / 2, mouth_vert]
    return kps


# ---------------------------------------------------------------------------
# compute_lip_open_ratio_203
# ---------------------------------------------------------------------------


class TestComputeLipOpenRatio203:
    def test_returns_none_when_kps_is_none(self):
        assert compute_lip_open_ratio_203(None) is None

    def test_returns_none_when_too_few_landmarks(self):
        kps = np.zeros((100, 2), dtype=np.float32)
        assert compute_lip_open_ratio_203(kps) is None

    def test_returns_none_when_exactly_202_landmarks(self):
        kps = np.zeros((202, 2), dtype=np.float32)
        assert compute_lip_open_ratio_203(kps) is None

    def test_returns_float_for_203_landmarks(self):
        kps = _make_kps_203(203, mouth_vert=10.0, mouth_horiz=50.0)
        result = compute_lip_open_ratio_203(kps)
        assert isinstance(result, float)

    def test_ratio_value_correct(self):
        # vert=10, horiz=50 → ratio ≈ 10/50 = 0.2
        kps = _make_kps_203(203, mouth_vert=10.0, mouth_horiz=50.0)
        result = compute_lip_open_ratio_203(kps)
        assert result == pytest.approx(10.0 / 50.0, rel=1e-4)

    def test_returns_none_when_mouth_span_too_small(self):
        kps = _make_kps_203(203, mouth_vert=2.0, mouth_horiz=MIN_MOUTH_SPAN_PX - 1.0)
        assert compute_lip_open_ratio_203(kps) is None

    def test_returns_value_when_span_exactly_at_min(self):
        # Guard is strict <, so span == MIN_MOUTH_SPAN_PX passes through and returns a value
        kps = _make_kps_203(203, mouth_horiz=MIN_MOUTH_SPAN_PX, mouth_vert=2.0)
        assert compute_lip_open_ratio_203(kps) is not None

    def test_returns_value_when_span_above_min(self):
        kps = _make_kps_203(203, mouth_horiz=MIN_MOUTH_SPAN_PX + 1.0, mouth_vert=5.0)
        result = compute_lip_open_ratio_203(kps)
        assert result is not None
        assert result > 0.0

    def test_closed_mouth_returns_low_ratio(self):
        # vert very small relative to horiz
        kps = _make_kps_203(203, mouth_vert=0.5, mouth_horiz=50.0)
        result = compute_lip_open_ratio_203(kps)
        assert result < 0.05

    def test_wide_open_mouth_returns_high_ratio(self):
        kps = _make_kps_203(203, mouth_vert=30.0, mouth_horiz=50.0)
        result = compute_lip_open_ratio_203(kps)
        assert result > 0.5

    def test_accepts_more_than_203_landmarks(self):
        kps = _make_kps_203(300, mouth_vert=10.0, mouth_horiz=50.0)
        result = compute_lip_open_ratio_203(kps)
        assert result is not None


# ---------------------------------------------------------------------------
# compute_lip_open_ratio_68
# ---------------------------------------------------------------------------


class TestComputeLipOpenRatio68:
    def test_returns_none_when_kps_is_none(self):
        assert compute_lip_open_ratio_68(None) is None

    def test_returns_none_when_too_few_landmarks(self):
        kps = np.zeros((30, 2), dtype=np.float32)
        assert compute_lip_open_ratio_68(kps) is None

    def test_returns_none_when_exactly_67_landmarks(self):
        kps = np.zeros((67, 2), dtype=np.float32)
        assert compute_lip_open_ratio_68(kps) is None

    def test_returns_float_for_68_landmarks(self):
        kps = _make_kps_68(68, mouth_vert=10.0, mouth_horiz=50.0)
        result = compute_lip_open_ratio_68(kps)
        assert isinstance(result, float)

    def test_ratio_value_correct(self):
        kps = _make_kps_68(68, mouth_vert=10.0, mouth_horiz=50.0)
        result = compute_lip_open_ratio_68(kps)
        assert result == pytest.approx(10.0 / 50.0, rel=1e-4)

    def test_returns_none_when_mouth_span_too_small(self):
        kps = _make_kps_68(68, mouth_horiz=MIN_MOUTH_SPAN_PX - 1.0)
        assert compute_lip_open_ratio_68(kps) is None

    def test_returns_value_when_span_above_min(self):
        kps = _make_kps_68(68, mouth_horiz=MIN_MOUTH_SPAN_PX + 1.0, mouth_vert=4.0)
        assert compute_lip_open_ratio_68(kps) is not None


# ---------------------------------------------------------------------------
# MouthOpennessState
# ---------------------------------------------------------------------------


class TestMouthOpennessState:
    # -- initial state --

    def test_initial_state_is_inactive(self):
        state = MouthOpennessState()
        assert state.active is False

    def test_initial_ema_is_zero(self):
        state = MouthOpennessState()
        assert state.ema == 0.0

    def test_initial_none_streak_is_zero(self):
        state = MouthOpennessState()
        assert state.none_streak == 0

    # -- update returns tuple --

    def test_update_returns_tuple_of_bool_and_float(self):
        state = MouthOpennessState()
        result = state.update(ratio=0.5, alpha=1.0, threshold=0.2)
        assert isinstance(result, tuple)
        assert len(result) == 2
        active, ema = result
        assert isinstance(active, bool)
        assert isinstance(ema, float)

    # -- immediate first-frame activation --

    def test_immediate_activation_on_first_trigger(self):
        """First valid ratio >= threshold activates instantly (no EMA ramp)."""
        state = MouthOpennessState()
        active, ema = state.update(ratio=0.30, alpha=0.40, threshold=0.12)
        assert active is True
        assert state.active is True
        # EMA is set directly to ratio (not alpha-blended from 0)
        assert ema == pytest.approx(0.30)

    def test_immediate_activation_sets_ema_to_ratio_not_blended(self):
        """Immediate activation bypasses the EMA formula for the first trigger."""
        state = MouthOpennessState()
        # With alpha=0.4 and ema_before=0, normal EMA would give 0.4*0.30 = 0.12
        state.update(ratio=0.30, alpha=0.40, threshold=0.12)
        assert state.ema == pytest.approx(0.30)  # not 0.12

    def test_immediate_activation_does_not_fire_when_already_active(self):
        """Subsequent frames go through normal EMA update, not immediate activation."""
        state = MouthOpennessState(active=True, ema=0.30)
        active, ema = state.update(ratio=0.30, alpha=0.40, threshold=0.12)
        # EMA blended: 0.4*0.30 + 0.6*0.30 = 0.30
        assert ema == pytest.approx(0.30)
        assert active is True

    # -- update: activation --

    def test_update_activates_when_ratio_above_threshold(self):
        state = MouthOpennessState()
        active, _ = state.update(ratio=0.5, alpha=1.0, threshold=0.2)
        assert active is True
        assert state.active is True

    def test_update_activates_when_ratio_equals_threshold(self):
        state = MouthOpennessState()
        active, _ = state.update(ratio=0.2, alpha=1.0, threshold=0.2)
        assert active is True

    # -- hysteresis: deactivation threshold is 75% of activation threshold --

    def test_deactivation_uses_hysteresis_threshold(self):
        """Active state deactivates at threshold*0.75, not threshold."""
        state = MouthOpennessState(active=True, ema=0.5)
        threshold = 0.20
        # deactivate_threshold = threshold * 0.75 = 0.15

        # Ratio that would deactivate without hysteresis but not with it
        # EMA after update: 1.0*0.16 = 0.16 → 0.16 >= 0.15 → stays active
        active, _ = state.update(ratio=0.16, alpha=1.0, threshold=threshold)
        assert active is True

    def test_deactivates_when_ema_falls_below_hysteresis_threshold(self):
        """Deactivation fires when EMA drops below threshold * 0.75."""
        state = MouthOpennessState(active=True, ema=0.5)
        threshold = 0.20
        # EMA after: 1.0*0.10 = 0.10 → 0.10 < 0.15 → deactivate
        active, _ = state.update(ratio=0.10, alpha=1.0, threshold=threshold)
        assert active is False
        assert state.active is False

    def test_deactivation_requires_multiple_frames_with_low_alpha(self):
        """With alpha=0.5 and high starting EMA, it takes multiple frames to cross
        the hysteresis threshold."""
        state = MouthOpennessState(active=True, ema=0.40)
        threshold = 0.20  # deactivate at 0.15
        # Frame 1: ema = 0.5*0.0 + 0.5*0.40 = 0.20 → still above deactivate_threshold (0.15)
        active, ema = state.update(ratio=0.0, alpha=0.5, threshold=threshold)
        assert active is True
        assert ema == pytest.approx(0.20)
        # Frame 2: ema = 0.5*0.0 + 0.5*0.20 = 0.10 → 0.10 < 0.15 → deactivate
        active, ema = state.update(ratio=0.0, alpha=0.5, threshold=threshold)
        assert active is False

    # -- update: rule 2 — ratio=None → stay --

    def test_update_stays_active_when_ratio_is_none(self):
        state = MouthOpennessState(active=True, ema=0.5)
        active, _ = state.update(ratio=None, alpha=0.4, threshold=0.2)
        assert active is True
        assert state.active is True

    def test_update_stays_inactive_when_ratio_is_none(self):
        state = MouthOpennessState(active=False, ema=0.0)
        active, _ = state.update(ratio=None, alpha=0.4, threshold=0.2)
        assert active is False
        assert state.active is False

    def test_update_none_does_not_change_ema_before_timeout(self):
        state = MouthOpennessState(active=True, ema=0.5)
        state.update(ratio=None, alpha=0.4, threshold=0.2)
        assert state.ema == 0.5

    def test_none_streak_increments_while_active_and_ratio_none(self):
        state = MouthOpennessState(active=True, ema=0.5)
        for i in range(3):
            state.update(ratio=None, alpha=0.4, threshold=0.2)
        assert state.none_streak == 3

    def test_none_streak_does_not_increment_when_inactive(self):
        state = MouthOpennessState(active=False, ema=0.0)
        for _ in range(5):
            state.update(ratio=None, alpha=0.4, threshold=0.2)
        assert state.none_streak == 0

    # -- occlusion timeout --

    def test_occlusion_timeout_decays_ema_after_n_frames(self):
        """After OCCLUSION_TIMEOUT_FRAMES of None while active, EMA starts decaying."""
        # Start with ema well above deactivate threshold to make it non-trivial
        state = MouthOpennessState(active=True, ema=0.5)
        threshold = 0.20

        # Feed exactly OCCLUSION_TIMEOUT_FRAMES None frames — no decay yet
        for _ in range(OCCLUSION_TIMEOUT_FRAMES):
            state.update(ratio=None, alpha=0.4, threshold=threshold)

        ema_at_timeout = state.ema
        assert ema_at_timeout == pytest.approx(0.5)  # no decay yet

        # One more frame past timeout — decay starts
        state.update(ratio=None, alpha=0.4, threshold=threshold)
        assert state.ema < ema_at_timeout

    def test_occlusion_timeout_eventually_deactivates(self):
        """Enough None frames eventually cause deactivation via EMA decay."""
        state = MouthOpennessState(active=True, ema=0.20)
        threshold = 0.20  # deactivate at 0.15

        # Feed many None frames past the timeout
        for _ in range(OCCLUSION_TIMEOUT_FRAMES + 100):
            state.update(ratio=None, alpha=0.4, threshold=threshold)

        assert state.active is False

    def test_none_streak_resets_when_valid_ratio_arrives(self):
        """A valid ratio resets the none_streak counter."""
        state = MouthOpennessState(active=True, ema=0.5)
        for _ in range(10):
            state.update(ratio=None, alpha=0.4, threshold=0.2)
        assert state.none_streak == 10

        # Valid ratio resets the streak
        state.update(ratio=0.3, alpha=0.4, threshold=0.2)
        assert state.none_streak == 0

    # -- EMA smoothing --

    def test_ema_smoothing_formula_when_already_active(self):
        """When already active, normal EMA formula is applied (no immediate activation)."""
        state = MouthOpennessState(active=True, ema=0.0)
        # alpha=0.5, ratio=0.8, ema_before=0.0 → 0.5*0.8 + 0.5*0.0 = 0.4
        state.update(ratio=0.8, alpha=0.5, threshold=0.2)
        assert state.ema == pytest.approx(0.4)

    def test_ema_smoothing_formula_when_ratio_below_threshold(self):
        """When ratio < threshold and inactive, EMA is still updated (for approach tracking)."""
        state = MouthOpennessState(active=False, ema=0.0)
        # ratio=0.08 < threshold=0.9 → no immediate activation, normal EMA
        # alpha=0.5: ema = 0.5*0.08 + 0.5*0.0 = 0.04
        state.update(ratio=0.08, alpha=0.5, threshold=0.9)
        assert state.ema == pytest.approx(0.04)

    def test_ema_alpha_zero_freezes_ema_when_active(self):
        """With alpha=0, EMA never changes (when already active, no immediate path)."""
        state = MouthOpennessState(active=True, ema=0.5)
        state.update(ratio=1.0, alpha=0.0, threshold=0.2)
        assert state.ema == pytest.approx(0.5)
        assert state.active is True

    def test_ema_accumulates_across_updates(self):
        state = MouthOpennessState()
        # First update: ratio=0.08 < threshold=0.9 → no immediate activation
        state.update(ratio=0.08, alpha=0.5, threshold=0.9)  # ema=0.04, below threshold
        assert state.active is False
        # Second update: ratio=0.8 >= threshold=0.5 → immediate activation
        state.update(ratio=0.8, alpha=0.5, threshold=0.5)
        assert state.active is True

    def test_ema_decays_toward_zero_when_active(self):
        state = MouthOpennessState(active=True, ema=0.9)
        state.update(ratio=0.0, alpha=0.5, threshold=0.2)
        assert state.ema == pytest.approx(0.45)

    # -- ema value returned from update --

    def test_update_returns_current_ema_value(self):
        state = MouthOpennessState(active=True, ema=0.4)
        _, ema = state.update(ratio=0.6, alpha=0.5, threshold=0.2)
        # ema = 0.5*0.6 + 0.5*0.4 = 0.5
        assert ema == pytest.approx(0.5)

    def test_update_returns_ema_unchanged_when_ratio_none(self):
        state = MouthOpennessState(active=True, ema=0.4)
        _, ema = state.update(ratio=None, alpha=0.5, threshold=0.2)
        assert ema == pytest.approx(0.4)

    # -- high threshold --

    def test_high_threshold_not_triggered_by_normal_ratio(self):
        state = MouthOpennessState()
        active, _ = state.update(ratio=0.15, alpha=1.0, threshold=0.50)
        assert active is False

    # -- reset --

    def test_reset_clears_active_ema_and_none_streak(self):
        state = MouthOpennessState(active=True, ema=0.8)
        state.none_streak = 15
        state.reset()
        assert state.active is False
        assert state.ema == 0.0
        assert state.none_streak == 0

    # -- proportional strength helper --

    def test_ema_enables_proportional_strength_calculation(self):
        """The returned EMA value enables the caller to compute proportional strength."""
        state = MouthOpennessState()
        threshold = 0.12
        ramp_range = max(threshold * 0.5, 0.04)  # = 0.06

        # EMA at threshold → proportion = 0 (just activated)
        active, ema = state.update(ratio=threshold, alpha=1.0, threshold=threshold)
        proportion = min(1.0, max(0.0, (ema - threshold) / ramp_range))
        assert proportion == pytest.approx(0.0)

        # EMA at threshold + ramp_range → proportion = 1.0 (full strength)
        state2 = MouthOpennessState(active=True, ema=threshold + ramp_range)
        active2, ema2 = state2.update(
            ratio=threshold + ramp_range, alpha=1.0, threshold=threshold
        )
        proportion2 = min(1.0, max(0.0, (ema2 - threshold) / ramp_range))
        assert proportion2 == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# UT-03: Occlusion decay is continuous — fires on every frame AFTER timeout
# ---------------------------------------------------------------------------


class TestOcclusionDecayContinuous:
    """UT-03: Verify that EMA decay is applied on every frame past the timeout
    (not just once at the boundary).  Continuous decay is the intended behaviour
    to produce a smooth fade-out rather than a single step.
    """

    def test_decay_applied_on_every_frame_after_timeout(self):
        state = MouthOpennessState(active=True, ema=0.5)
        threshold = 0.20

        # Advance exactly to the timeout boundary — decay should not fire yet
        for _ in range(OCCLUSION_TIMEOUT_FRAMES):
            state.update(ratio=None, alpha=0.4, threshold=threshold)
        assert state.ema == pytest.approx(0.5)

        # Each subsequent None frame should keep decaying
        ema_prev = state.ema
        for _ in range(5):
            state.update(ratio=None, alpha=0.4, threshold=threshold)
            assert state.ema < ema_prev, (
                "EMA should decrease on each frame past timeout"
            )
            ema_prev = state.ema

    def test_decay_not_applied_at_exact_timeout_boundary(self):
        """none_streak == OCCLUSION_TIMEOUT_FRAMES: condition is >, so no decay yet."""
        state = MouthOpennessState(active=True, ema=0.5)
        # Feed exactly OCCLUSION_TIMEOUT_FRAMES None frames
        for _ in range(OCCLUSION_TIMEOUT_FRAMES):
            state.update(ratio=None, alpha=0.4, threshold=0.2)
        # EMA must still be 0.5 — no decay at the boundary
        assert state.ema == pytest.approx(0.5)

    def test_ema_monotonically_decreases_after_timeout(self):
        state = MouthOpennessState(active=True, ema=0.5)
        emas = [0.5]
        for _ in range(OCCLUSION_TIMEOUT_FRAMES + 30):
            state.update(ratio=None, alpha=0.4, threshold=0.2)
            emas.append(state.ema)

        post_timeout = emas[OCCLUSION_TIMEOUT_FRAMES + 1 :]
        for i in range(1, len(post_timeout)):
            assert post_timeout[i] <= post_timeout[i - 1]


# ---------------------------------------------------------------------------
# UT-08: MouthOpennessState thread safety
# ---------------------------------------------------------------------------


class TestMouthOpennessStateThreadSafety:
    """UT-08: Concurrent update() calls from two threads must not corrupt state."""

    def test_concurrent_updates_do_not_raise(self):
        """Two threads calling update() simultaneously should not raise exceptions."""
        state = MouthOpennessState(active=True, ema=0.5)
        errors: list = []

        def worker(ratio_val):
            try:
                for _ in range(200):
                    state.update(ratio=ratio_val, alpha=0.4, threshold=0.2)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        t1 = threading.Thread(target=worker, args=(0.3,))
        t2 = threading.Thread(target=worker, args=(None,))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert errors == [], f"Thread-safety violation: {errors}"

    def test_concurrent_reset_and_update_do_not_raise(self):
        state = MouthOpennessState(active=True, ema=0.8)
        errors: list = []

        def updater():
            try:
                for _ in range(200):
                    state.update(ratio=0.3, alpha=0.4, threshold=0.2)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        def resetter():
            try:
                for _ in range(50):
                    state.reset()
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        t1 = threading.Thread(target=updater)
        t2 = threading.Thread(target=resetter)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert errors == [], f"Thread-safety violation during reset: {errors}"
