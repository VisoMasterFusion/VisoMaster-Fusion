from __future__ import annotations

import queue
from pathlib import Path
from types import SimpleNamespace

from app.processors.video_processor import VideoProcessor


class _RunResult:
    def __init__(self, returncode: int = 0, stderr: str = "", stdout: str = ""):
        self.returncode = returncode
        self.stderr = stderr
        self.stdout = stdout


def test_mark_skipped_frame_tracks_reason_counts():
    dummy = SimpleNamespace(
        skipped_frames=set(),
        total_skipped_frames=0,
        manual_dropped_skip_count=0,
        read_error_skip_count=0,
    )

    VideoProcessor._mark_skipped_frame(dummy, 10, "manual_drop")
    VideoProcessor._mark_skipped_frame(dummy, 11, "read_error")

    assert dummy.skipped_frames == {10, 11}
    assert dummy.total_skipped_frames == 2
    assert dummy.manual_dropped_skip_count == 1
    assert dummy.read_error_skip_count == 1


def test_identify_frame_segments_without_skips_uses_processing_start_frame():
    dummy = SimpleNamespace(processing_start_frame=100, skipped_frames=set())

    segments = VideoProcessor._identify_frame_segments(dummy, 120)

    assert segments == [(100, 120)]


def test_identify_frame_segments_handles_boundary_and_pre_start_skips():
    dummy = SimpleNamespace(
        processing_start_frame=100,
        skipped_frames={95, 100, 101, 105, 112},
    )

    segments = VideoProcessor._identify_frame_segments(dummy, 112)

    assert segments == [(102, 104), (106, 111)]


def test_extract_audio_segments_always_normalizes_to_containerized_aac(
    tmp_path, monkeypatch
):
    calls: list[list[str]] = []
    validation_results = iter([False, True])

    def fake_run(args, **kwargs):
        calls.append(list(args))
        return _RunResult()

    dummy = SimpleNamespace(
        fps=30.0,
        media_path=str(tmp_path / "input.mkv"),
        _validate_audio_file=lambda _: next(validation_results),
    )

    monkeypatch.setattr("subprocess.run", fake_run)

    ok, audio_files = VideoProcessor._extract_audio_segments(
        dummy, [(0, 29)], str(tmp_path)
    )

    assert ok is True
    assert len(audio_files) == 1
    assert audio_files[0].endswith(".m4a")

    first_call = calls[0]
    retry_call = calls[1]

    assert first_call[-1].endswith(".m4a")
    assert first_call[first_call.index("-c:a") + 1] == "aac"
    assert first_call[first_call.index("-af") + 1] == "aresample=async=1:first_pts=0"
    assert first_call[first_call.index("-map") + 1] == "0:a:0?"

    assert retry_call[-1].endswith(".m4a")
    assert retry_call[retry_call.index("-c:a") + 1] == "aac"
    assert retry_call[retry_call.index("-af") + 1] == "aresample=async=1:first_pts=0"


def test_extract_audio_segments_returns_failure_after_double_validation_failure(
    tmp_path, monkeypatch
):
    calls: list[list[str]] = []
    removed_paths: list[str] = []

    def fake_run(args, **kwargs):
        calls.append(list(args))
        return _RunResult()

    dummy = SimpleNamespace(
        fps=30.0,
        media_path=str(tmp_path / "input.mkv"),
        _validate_audio_file=lambda _: False,
    )

    monkeypatch.setattr("subprocess.run", fake_run)
    monkeypatch.setattr("os.remove", lambda path: removed_paths.append(path))

    ok, audio_files = VideoProcessor._extract_audio_segments(
        dummy, [(0, 29)], str(tmp_path)
    )

    assert ok is False
    assert audio_files == []
    assert len(calls) == 2
    assert removed_paths == [str(tmp_path / "audio_segment_0000.m4a")]


def test_concatenate_audio_segments_reencodes_concat_output_to_m4a(
    tmp_path, monkeypatch
):
    calls: list[list[str]] = []

    def fake_run(args, **kwargs):
        calls.append(list(args))
        return _RunResult()

    audio_files = []
    for name in ("seg_a.m4a", "seg_b.m4a"):
        path = tmp_path / name
        path.write_bytes(b"stub")
        audio_files.append(str(path))

    monkeypatch.setattr("subprocess.run", fake_run)

    output_path = VideoProcessor._concatenate_audio_segments(
        SimpleNamespace(), audio_files, str(tmp_path)
    )

    assert output_path == str(tmp_path / "audio_concatenated.m4a")
    assert len(calls) == 1

    concat_call = calls[0]
    manifest_path = Path(tmp_path / "concat_manifest.txt")
    manifest_text = manifest_path.read_text(encoding="utf-8")

    assert "file '" in manifest_text
    assert concat_call[concat_call.index("-c:a") + 1] == "aac"
    assert concat_call[concat_call.index("-af") + 1] == "aresample=async=1:first_pts=0"
    assert concat_call[-1].endswith(".m4a")


def test_finalize_default_style_recording_uses_rebuilt_audio_when_frames_skipped(
    tmp_path, monkeypatch
):
    temp_file = tmp_path / "temp_output.mp4"
    temp_file.write_bytes(b"temp-video")
    rebuilt_audio = tmp_path / "rebuilt_audio.m4a"
    rebuilt_audio.write_bytes(b"temp-audio")
    final_output = tmp_path / "final_output.mp4"

    identify_calls: list[int] = []
    extract_calls: list[tuple[list[tuple[int, int]], str]] = []
    concat_calls: list[tuple[list[str], str]] = []
    ffmpeg_calls: list[list[str]] = []

    def fake_run(args, **kwargs):
        ffmpeg_calls.append(list(args))
        return _RunResult()

    q = queue.Queue()
    dummy = SimpleNamespace(
        feeder_thread=None,
        frames_to_display=[],
        frame_queue=q,
        join_and_clear_threads=lambda: None,
        gpu_memory_update_timer=SimpleNamespace(stop=lambda: None),
        preroll_timer=SimpleNamespace(stop=lambda: None),
        stop_live_sound=lambda: None,
        media_capture=None,
        recording_sp=None,
        recording=True,
        processing=True,
        is_processing_segments=False,
        next_frame_to_display=31,
        max_frame_number=100,
        frames_written=30,
        fps=30.0,
        play_start_time=0.0,
        play_end_time=0.0,
        total_skipped_frames=2,
        manual_dropped_skip_count=1,
        read_error_skip_count=1,
        temp_file=str(temp_file),
        media_path=str(tmp_path / "input.mkv"),
        stopped_by_error_limit=False,
        triggered_by_job_manager=False,
        processing_start_frame=10,
        last_displayed_frame=29,
        main_window=SimpleNamespace(
            control={
                "OutputMediaFolder": str(tmp_path),
                "AutoSaveWorkspaceToggle": False,
                "OpenOutputToggle": False,
            }
        ),
        _apply_job_timestamp_to_output_name=lambda *args: (None, None),
        _identify_frame_segments=lambda actual_end_frame: (
            identify_calls.append(actual_end_frame) or [(10, 14), (16, 29)]
        ),
        _extract_audio_segments=lambda segments, temp_audio_dir: (
            extract_calls.append((segments, temp_audio_dir))
            or (True, [str(tmp_path / "seg_0000.m4a"), str(tmp_path / "seg_0001.m4a")])
        ),
        _concatenate_audio_segments=lambda audio_files, temp_audio_dir: (
            concat_calls.append((audio_files, temp_audio_dir)) or str(rebuilt_audio)
        ),
        _write_video_only_output=lambda *args: True,
        _log_processing_summary=lambda *args: None,
        disable_virtualcam=lambda: None,
        processing_stopped_signal=SimpleNamespace(emit=lambda: None),
        file_type="image",
        start_time=0.0,
    )

    monkeypatch.setattr("subprocess.run", fake_run)
    monkeypatch.setattr(
        "app.processors.video_processor.misc_helpers.get_output_file_path",
        lambda *args, **kwargs: str(final_output),
    )
    monkeypatch.setattr(
        "app.processors.video_processor.layout_actions.enable_all_parameters_and_control_widget",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "app.processors.video_processor.video_control_actions.reset_media_buttons",
        lambda *args, **kwargs: None,
    )

    VideoProcessor._finalize_default_style_recording(dummy)

    assert identify_calls == [29]
    assert extract_calls and extract_calls[0][0] == [(10, 14), (16, 29)]
    assert concat_calls

    final_mux_call = ffmpeg_calls[-1]
    assert str(rebuilt_audio) in final_mux_call
    assert str(temp_file) in final_mux_call
    assert "-ss" not in final_mux_call
    assert dummy.temp_file == ""
