"""Unit tests for app.helpers.build_progress (Qt-free build-progress logic)."""

import json

from app.helpers.build_progress import (
    BuildTimeStore,
    StageTracker,
    format_duration,
    parse_build_stage,
)


class TestParseBuildStage:
    def test_parse_phase_markers(self):
        assert (
            parse_build_stage("[ONNX Prober]: Attempting to load det_10g.onnx...")
            == "Reading and parsing the ONNX graph"
        )

    def test_build_phase_markers(self):
        assert (
            parse_build_stage("[TRT] Building engine with 1 optimization profile")
            == "Compiling and optimizing the TensorRT engine"
        )
        assert (
            parse_build_stage("[TRT] Running tactic selection for layer 42")
            == "Compiling and optimizing the TensorRT engine"
        )

    def test_serialize_phase_markers(self):
        assert (
            parse_build_stage("[TRT] Serializing engine to file")
            == "Writing the engine cache to disk"
        )

    def test_done_phase_markers(self):
        assert (
            parse_build_stage(
                "[ONNX Prober]: Load successful. TRT engine cache built and flushed."
            )
            == "Build complete"
        )

    def test_matching_is_case_insensitive(self):
        assert (
            parse_build_stage("ATTEMPTING TO LOAD MODEL")
            == "Reading and parsing the ONNX graph"
        )

    def test_unrelated_line_returns_none(self):
        assert parse_build_stage("[INFO] Loading model: Inswapper128") is None
        assert parse_build_stage("") is None


class TestStageTracker:
    def test_advances_and_reports_new_stage(self):
        tracker = StageTracker()
        assert tracker.current_label is None
        assert tracker.update("Attempting to load x.onnx") == (
            "Reading and parsing the ONNX graph"
        )
        assert tracker.update("some noise line") is None
        assert tracker.update("Building engine...") == (
            "Compiling and optimizing the TensorRT engine"
        )
        assert tracker.current_label == (
            "Compiling and optimizing the TensorRT engine"
        )

    def test_never_moves_backwards(self):
        tracker = StageTracker()
        tracker.update("Building engine...")
        # A later line matching an earlier phase must not move the stage back.
        assert tracker.update("Attempting to load x.onnx") is None
        assert tracker.current_label == (
            "Compiling and optimizing the TensorRT engine"
        )

    def test_repeated_stage_is_not_reported_twice(self):
        tracker = StageTracker()
        assert tracker.update("Building engine...") is not None
        assert tracker.update("Still building engine...") is None

    def test_reset(self):
        tracker = StageTracker()
        tracker.update("Building engine...")
        tracker.reset()
        assert tracker.current_label is None
        # After a reset (retry attempt), the earliest stage can be reported again.
        assert tracker.update("Attempting to load x.onnx") is not None


class TestFormatDuration:
    def test_seconds(self):
        assert format_duration(0) == "0:00"
        assert format_duration(5.4) == "0:05"
        assert format_duration(59.6) == "1:00"

    def test_minutes(self):
        assert format_duration(83) == "1:23"
        assert format_duration(600) == "10:00"

    def test_hours(self):
        assert format_duration(3600) == "1:00:00"
        assert format_duration(3661) == "1:01:01"

    def test_negative_clamps_to_zero(self):
        assert format_duration(-5) == "0:00"


class TestBuildTimeStore:
    def test_missing_file_starts_empty(self, tmp_path):
        store = BuildTimeStore(str(tmp_path / "nope.json"))
        assert store.expected_seconds("ModelA") is None
        assert store.sample_count("ModelA") == 0

    def test_record_and_expected_roundtrip(self, tmp_path):
        path = str(tmp_path / "times.json")
        store = BuildTimeStore(path)
        store.record("ModelA", 100.0)
        store.record("ModelA", 120.0)
        assert store.expected_seconds("ModelA") == 110.0

        # A fresh instance sees the persisted values.
        reloaded = BuildTimeStore(path)
        assert reloaded.expected_seconds("ModelA") == 110.0
        assert reloaded.sample_count("ModelA") == 2

    def test_keeps_only_recent_samples(self, tmp_path):
        store = BuildTimeStore(str(tmp_path / "times.json"))
        for index in range(BuildTimeStore.MAX_SAMPLES_PER_MODEL + 3):
            store.record("ModelA", float(index + 1))
        assert store.sample_count("ModelA") == BuildTimeStore.MAX_SAMPLES_PER_MODEL
        # Oldest samples were dropped: the average covers only the last five.
        assert store.expected_seconds("ModelA") == sum([4.0, 5.0, 6.0, 7.0, 8.0]) / 5

    def test_corrupt_file_is_tolerated(self, tmp_path):
        path = tmp_path / "times.json"
        path.write_text("{not valid json", encoding="utf-8")
        store = BuildTimeStore(str(path))
        assert store.expected_seconds("ModelA") is None

    def test_wrong_shape_is_tolerated(self, tmp_path):
        path = tmp_path / "times.json"
        path.write_text(json.dumps({"models": "oops"}), encoding="utf-8")
        assert BuildTimeStore(str(path)).expected_seconds("ModelA") is None

        path.write_text(
            json.dumps({"models": {"ModelA": {"times": ["x", -1, 90.0]}}}),
            encoding="utf-8",
        )
        store = BuildTimeStore(str(path))
        assert store.expected_seconds("ModelA") == 90.0

    def test_non_positive_durations_are_ignored(self, tmp_path):
        store = BuildTimeStore(str(tmp_path / "times.json"))
        store.record("ModelA", 0)
        store.record("ModelA", -3)
        assert store.expected_seconds("ModelA") is None

    def test_creates_missing_directory(self, tmp_path):
        path = str(tmp_path / "nested" / "dir" / "times.json")
        store = BuildTimeStore(path)
        store.record("ModelA", 42.0)
        assert BuildTimeStore(path).expected_seconds("ModelA") == 42.0
