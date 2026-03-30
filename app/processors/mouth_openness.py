"""Auto-mouth: detect mouth openness from pipeline landmarks; maintain on/stay/off state."""

from __future__ import annotations
from dataclasses import dataclass, field
import threading
import numpy as np

# Minimum pixel span of the mouth (corners) to trust the ratio calculation.
# Lowered to 4.0 (was 8.0) to handle small/distant faces without false None returns.
MIN_MOUTH_SPAN_PX: float = 4.0

# Number of consecutive None-ratio frames while active before starting EMA decay.
# At 30 fps this is ~1.5 s of occlusion before the feature begins to fade out.
OCCLUSION_TIMEOUT_FRAMES: int = 45


def compute_lip_open_ratio_203(kps: np.ndarray | None) -> float | None:
    """Compute lip-open ratio using 203-point landmarks.

    Uses the same pairs as faceutil.calc_lip_close_ratio:
      vertical distance: kps[90] ↔ kps[102]
      horizontal span:   kps[48] ↔ kps[66]

    Returns None when landmarks are unavailable or the face is too small.
    """
    if kps is None or len(kps) < 203:
        return None
    span = float(np.linalg.norm(kps[48] - kps[66]))
    if span < MIN_MOUTH_SPAN_PX:
        return None
    vert = float(np.linalg.norm(kps[90] - kps[102]))
    return vert / (span + 1e-6)


def compute_lip_open_ratio_68(kps: np.ndarray | None) -> float | None:
    """Compute lip-open ratio using 68-point landmarks.

    vertical distance: kps[62] ↔ kps[66]
    horizontal span:   kps[48] ↔ kps[54]

    Returns None when landmarks are unavailable or the face is too small.
    """
    if kps is None or len(kps) < 68:
        return None
    span = float(np.linalg.norm(kps[48] - kps[54]))
    if span < MIN_MOUTH_SPAN_PX:
        return None
    vert = float(np.linalg.norm(kps[62] - kps[66]))
    return vert / (span + 1e-6)


@dataclass
class MouthOpennessState:
    """Per-face EMA state for the Auto Mouth Expression feature.

    Update rules:
      1. ratio >= threshold AND not yet active → activate immediately (no cold-start ramp)
      2. ratio is None → stay in current state; increment occlusion counter; after
         OCCLUSION_TIMEOUT_FRAMES decay EMA until it drops below the deactivate threshold
      3. ratio >= threshold (already active) → keep active; update EMA normally
      4. ratio < deactivate_threshold (= threshold * 0.75) → deactivate  (hysteresis band)

    Hysteresis prevents rapid on/off oscillation when the ratio hovers at the threshold.
    Occlusion timeout prevents the feature from staying stuck active after tracking is lost.
    """

    active: bool = False
    ema: float = 0.0
    none_streak: int = field(default=0, compare=False, repr=False)
    # P2-06: guards concurrent access from multiple FrameWorker threads
    _lock: threading.Lock = field(
        default_factory=threading.Lock, compare=False, repr=False, init=False
    )

    def update(
        self,
        ratio: float | None,
        alpha: float,
        threshold: float,
    ) -> tuple[bool, float]:
        """Update EMA and activation state.

        Returns:
            (active, ema) — current activation flag and smoothed EMA value.
            The EMA value is used by the caller for proportional strength scaling.
        """
        deactivate_threshold = threshold * 0.75

        with self._lock:
            if ratio is None:
                # Rule 2: stay; manage occlusion timeout
                if self.active:
                    self.none_streak += 1
                    if self.none_streak > OCCLUSION_TIMEOUT_FRAMES:
                        # Slow EMA decay so the feature fades out instead of snapping off
                        self.ema *= 0.92
                        if self.ema < deactivate_threshold:
                            self.active = False
                            self.none_streak = 0
                return self.active, self.ema

            # Valid ratio received — reset occlusion counter
            self.none_streak = 0

            # Rule 1: immediate first-frame activation (skip cold-start ramp)
            if not self.active and ratio >= threshold:
                self.ema = ratio
                self.active = True
                return True, self.ema

            # Normal EMA update
            self.ema = alpha * ratio + (1.0 - alpha) * self.ema

            # Hysteresis: activate at threshold, deactivate at threshold * 0.75
            # (25% band prevents rapid on/off toggling near the boundary)
            if not self.active and self.ema >= threshold:
                self.active = True
            elif self.active and self.ema < deactivate_threshold:
                self.active = False

            return self.active, self.ema

    def reset(self) -> None:
        """Reset state to inactive (call when switching target faces or disabling)."""
        with self._lock:
            self.active = False
            self.ema = 0.0
            self.none_streak = 0
