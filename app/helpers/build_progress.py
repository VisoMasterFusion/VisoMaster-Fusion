"""Qt-free helpers backing the TensorRT engine-build progress UI.

Everything in this module is pure Python so it can be unit-tested without a
Qt application instance. It is used by ``app.processors.models_processor``
(build-probe orchestration) and ``app.ui.widgets.trt_build_dialog`` (display).
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Optional

# --- Build stage detection ---------------------------------------------------
#
# The engine build happens entirely inside ONNX Runtime's TensorRT execution
# provider, so there is no callback API for true percentage progress. What we
# *can* do is watch the log lines the probe process prints and map known
# ONNX Runtime / TensorRT message fragments onto coarse human-readable
# phases. Matching is deliberately defensive (case-insensitive substring with
# several alternative fragments per phase): if a future TensorRT release
# rewords its logs, the dialog simply stays on the previous stage instead of
# breaking.

STAGE_DEFINITIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "Reading and parsing the ONNX graph",
        (
            "attempting to load",
            "parsing model",
            "parsing onnx",
            "parse onnx",
            "reading onnx",
            "loading onnx",
            "graph optimization",
            "graph transformer",
        ),
    ),
    (
        "Compiling and optimizing the TensorRT engine",
        (
            "building engine",
            "build engine",
            "building trt engine",
            "engine generation",
            "tactic",
            "autotun",
            "myelin",
            "timing cache",
            "optimization profile",
            "compiling",
        ),
    ),
    (
        "Writing the engine cache to disk",
        (
            "serializ",
            "writing engine",
            "saving engine",
            "engine cache",
            "dumping engine",
            "flushed",
        ),
    ),
    (
        "Build complete",
        (
            "load successful",
            "deserialize",
        ),
    ),
)


def parse_build_stage(line: str) -> Optional[str]:
    """Return the human-readable stage label for a probe log line.

    Returns ``None`` when the line matches no known build phase. Later stages
    are checked first so that a line matching several phases (e.g. the final
    "Load successful. TRT engine cache built and flushed." mentions both the
    cache and the success) maps to the most advanced one.
    """
    lowered = line.lower()
    for label, fragments in reversed(STAGE_DEFINITIONS):
        if any(fragment in lowered for fragment in fragments):
            return label
    return None


class StageTracker:
    """Feeds probe log lines through :func:`parse_build_stage`.

    Only ever advances *forward* through :data:`STAGE_DEFINITIONS`, so
    out-of-order or repeated log lines never move the dialog backwards.
    """

    def __init__(self) -> None:
        self._stage_index = -1

    @property
    def current_label(self) -> Optional[str]:
        """Label of the stage currently reached, or None before any match."""
        if self._stage_index < 0:
            return None
        return STAGE_DEFINITIONS[self._stage_index][0]

    def update(self, line: str) -> Optional[str]:
        """Process one log line; return the label when the stage advances."""
        label = parse_build_stage(line)
        if label is None:
            return None
        labels = [definition[0] for definition in STAGE_DEFINITIONS]
        new_index = labels.index(label)
        if new_index > self._stage_index:
            self._stage_index = new_index
            return label
        return None

    def reset(self) -> None:
        """Forget the current stage (e.g. before a retry attempt)."""
        self._stage_index = -1


def format_duration(seconds: float) -> str:
    """Format a duration in seconds as ``m:ss`` (``h:mm:ss`` past one hour)."""
    total = max(0, int(round(seconds)))
    minutes, secs = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


class BuildTimeStore:
    """Persists per-model TensorRT engine build durations as JSON.

    The progress dialog uses these to show an honest "usually takes about X"
    estimate once a model has been built before. The file lives next to the
    downloaded models (``model_assets/trt_build_times.json``) so it survives
    both app updates and TensorRT cache wipes — the two events that trigger
    rebuilds.
    """

    MAX_SAMPLES_PER_MODEL = 5
    SCHEMA_VERSION = 1

    def __init__(self, path: str) -> None:
        self.path = path
        self._times: dict[str, list[float]] = self._load()

    def _load(self) -> dict[str, list[float]]:
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, ValueError):
            # Missing or corrupt file: start fresh rather than breaking loads.
            return {}

        models = payload.get("models") if isinstance(payload, dict) else None
        if not isinstance(models, dict):
            return {}

        times: dict[str, list[float]] = {}
        for key, entry in models.items():
            values = entry.get("times") if isinstance(entry, dict) else None
            if not isinstance(values, list):
                continue
            clean = [
                float(value)
                for value in values
                if isinstance(value, (int, float)) and not isinstance(value, bool)
                and value > 0
            ]
            if clean:
                times[str(key)] = clean[-self.MAX_SAMPLES_PER_MODEL :]
        return times

    def expected_seconds(self, model_key: str) -> Optional[float]:
        """Average of the recorded build durations, or None if unknown."""
        times = self._times.get(model_key)
        if not times:
            return None
        return sum(times) / len(times)

    def sample_count(self, model_key: str) -> int:
        """How many build durations are recorded for the model."""
        return len(self._times.get(model_key, ()))

    def record(self, model_key: str, seconds: float) -> None:
        """Record one successful build duration and persist the store."""
        if seconds <= 0:
            return
        times = self._times.setdefault(model_key, [])
        times.append(float(seconds))
        self._times[model_key] = times[-self.MAX_SAMPLES_PER_MODEL :]
        self._save()

    def _save(self) -> None:
        payload = {
            "version": self.SCHEMA_VERSION,
            "models": {key: {"times": times} for key, times in self._times.items()},
        }
        directory = os.path.dirname(self.path)
        try:
            if directory:
                os.makedirs(directory, exist_ok=True)
            # Write-then-replace so a crash mid-write cannot corrupt the store.
            fd, tmp_path = tempfile.mkstemp(
                dir=directory or None,
                prefix=".trt_build_times.",
                suffix=".tmp",
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    json.dump(payload, handle, indent=2)
                os.replace(tmp_path, self.path)
            except BaseException:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
                raise
        except OSError as exc:
            # A read-only install must never break model loading.
            print(f"[WARN] Could not persist TensorRT build times: {exc}")
