from types import SimpleNamespace

from app.ui.widgets.actions.video_control_actions import (
    _handle_issue_scan_cancelled,
    _handle_issue_scan_completed,
    _handle_issue_scan_failed,
    _handle_issue_scan_issue_found,
    _handle_issue_scan_progress,
    run_issue_scan,
    toggle_issue_scan,
)
from app.ui.widgets.ui_workers import IssueScanWorker


class _DummySignal:
    def __init__(self):
        self._callbacks = []

    def connect(self, callback):
        self._callbacks.append(callback)

    def emit(self, *args, **kwargs):
        for callback in list(self._callbacks):
            callback(*args, **kwargs)


class _DummyButton:
    def __init__(self, text="", checked=False):
        self.enabled = True
        self.text = text
        self.tooltip = ""
        self.checked = checked

    def setEnabled(self, value):
        self.enabled = bool(value)

    def setDisabled(self, value):
        self.enabled = not bool(value)

    def setText(self, text):
        self.text = text

    def setToolTip(self, tooltip):
        self.tooltip = tooltip

    def isChecked(self):
        return self.checked

    def setChecked(self, checked):
        self.checked = bool(checked)


class _DummySlider:
    def __init__(self, value=0):
        self._value = value
        self.block_calls = []
        self.updated = 0

    def value(self):
        return self._value

    def blockSignals(self, value):
        self.block_calls.append(bool(value))

    def setValue(self, value):
        self._value = int(value)

    def update(self):
        self.updated += 1


class _FakeIssueScanWorker:
    def __init__(self, main_window):
        self.main_window = main_window
        self._scan_scope_text = "Scanning full clip"
        self.progress = _DummySignal()
        self.completed = _DummySignal()
        self.cancelled = _DummySignal()
        self.failed = _DummySignal()
        self.issue_found = _DummySignal()
        self.started = False
        self.cancel_calls = 0
        self.deleted = False

    def start(self):
        self.started = True

    def cancel(self):
        self.cancel_calls += 1

    def deleteLater(self):
        self.deleted = True


def _make_worker_main_window():
    return SimpleNamespace(
        control={},
        parameters={},
        target_faces={},
        parameter_widgets={},
        videoSeekSlider=SimpleNamespace(value=lambda: 0),
        video_processor=SimpleNamespace(
            _get_issue_scan_ranges=lambda: [(0, 2)],
            describe_issue_scan_scope=lambda _ranges: "Scanning 1 marked range",
            _get_target_input_height=lambda: 256,
            prepare_issue_scan_target_faces_snapshot=lambda *_args, **_kwargs: {},
            scan_issue_frames=None,
        ),
    )


def _make_scan_main_window(keep_controls=False):
    slider = _DummySlider(24)
    main_window = SimpleNamespace(
        control={"KeepControlsToggle": keep_controls},
        parameters={},
        target_faces={"face_1": object()},
        parameter_widgets={},
        issue_frames_by_face={"face_1": {3, 5}, "face_2": {9}},
        issue_frames=set(),
        dropped_frames=set(),
        markers={},
        selected_target_face_id="face_1",
        scan_issue_worker=None,
        scan_issue_ui_state={},
        videoSeekSlider=slider,
        videoSeekLineEdit=SimpleNamespace(
            text=lambda: "24", setText=lambda _text: None
        ),
        graphicsViewFrame=SimpleNamespace(
            scene=lambda: SimpleNamespace(items=lambda: []),
            setSceneRect=lambda *_args: None,
        ),
        runScanButton=_DummyButton("Scan for Issues"),
        scanToolsToggleButton=_DummyButton("Scan Tools"),
        buttonMediaPlay=_DummyButton("Play"),
        buttonMediaRecord=_DummyButton("Record"),
        prevIssueButton=_DummyButton("Prev Issue"),
        nextIssueButton=_DummyButton("Next Issue"),
        dropFrameButton=_DummyButton("Drop Frame"),
        dropAllIssueFramesButton=_DummyButton("Drop Issue Frames"),
        clearScanResultsButton=_DummyButton("Clear Scan Results"),
        clearDroppedFramesButton=_DummyButton("Clear Dropped Frames"),
    )
    main_window.video_processor = SimpleNamespace(
        file_type="video",
        media_path="dummy.mp4",
        current_frame_number=24,
        current_frame=None,
        stop_processing=lambda: False,
        process_current_frame=lambda: None,
    )
    return main_window


def test_issue_scan_worker_progress_emits_live_fps(monkeypatch):
    main_window = _make_worker_main_window()

    def fake_scan_issue_frames(**kwargs):
        progress_callback = kwargs["progress_callback"]
        progress_callback(1, 3, 10)
        progress_callback(2, 3, 11)
        progress_callback(3, 3, 12)
        return {
            "issue_frames_by_face": {},
            "frames_scanned": 3,
            "faces_with_issues": 0,
            "cancelled": False,
        }

    main_window.video_processor.scan_issue_frames = fake_scan_issue_frames
    worker = IssueScanWorker(main_window)
    emitted = []
    completed = []
    monotonic_values = iter([10.0, 10.5, 11.0, 11.5, 12.0])

    monkeypatch.setattr(
        "app.ui.widgets.ui_workers.time.monotonic",
        lambda: next(monotonic_values),
    )

    worker.progress.connect(
        lambda processed, total, frame_number, scan_fps: emitted.append(
            (processed, total, frame_number, scan_fps)
        )
    )
    worker.completed.connect(
        lambda issue_frames_by_face, frames_scanned, faces_with_issues, scope_text, elapsed_seconds, cancelled: (
            completed.append(
                (
                    issue_frames_by_face,
                    frames_scanned,
                    faces_with_issues,
                    scope_text,
                    elapsed_seconds,
                    cancelled,
                )
            )
        )
    )

    worker.run()

    assert emitted == [
        (1, 3, 10, 2.0),
        (2, 3, 11, 2.0),
        (3, 3, 12, 2.0),
    ]
    assert completed == [({}, 3, 0, "Scanning 1 marked range", 2.0, False)]


def test_issue_scan_worker_passes_control_defaults_snapshot():
    control_widget = SimpleNamespace(default_value="default-control")
    main_window = _make_worker_main_window()
    main_window.control = {"ControlA": "live-control"}
    main_window.parameter_widgets = {
        "ControlA": control_widget,
        "IgnoredWidget": control_widget,
    }
    captured = {}

    def fake_scan_issue_frames(**kwargs):
        captured["control_defaults_snapshot"] = kwargs["control_defaults_snapshot"]
        return {
            "issue_frames_by_face": {},
            "frames_scanned": 1,
            "faces_with_issues": 0,
            "cancelled": False,
        }

    main_window.video_processor.scan_issue_frames = fake_scan_issue_frames

    worker = IssueScanWorker(main_window)
    worker.run()

    assert captured["control_defaults_snapshot"] == {"ControlA": "default-control"}


def test_issue_scan_worker_preserves_explicitly_empty_snapshots():
    main_window = _make_worker_main_window()
    captured = {}

    def fake_scan_issue_frames(**kwargs):
        captured["base_control"] = kwargs["base_control"]
        captured["base_params"] = kwargs["base_params"]
        captured["target_faces_snapshot"] = kwargs["target_faces_snapshot"]
        captured["control_defaults_snapshot"] = kwargs["control_defaults_snapshot"]
        return {
            "issue_frames_by_face": {},
            "frames_scanned": 1,
            "faces_with_issues": 0,
            "cancelled": False,
        }

    main_window.video_processor.scan_issue_frames = fake_scan_issue_frames

    worker = IssueScanWorker(main_window)
    worker.run()

    assert captured == {
        "base_control": {},
        "base_params": {},
        "target_faces_snapshot": {},
        "control_defaults_snapshot": {},
    }


def test_issue_scan_worker_passes_plain_target_face_snapshot_without_widget_methods():
    control_widget = SimpleNamespace(default_value="default-control")
    captured = {}

    class _TargetFaceWithoutEmbeddingAccess:
        def __init__(self):
            self.face_id = "face_1"
            self.cropped_face = None

        def get_embedding(self, _recognition_model):
            raise AssertionError(
                "IssueScanWorker should not call target-face widget methods"
            )

    def fake_prepare_issue_scan_target_faces_snapshot(*_args, **_kwargs):
        return {
            "face_1": {
                "face_id": "face_1",
                "embeddings_by_model": {
                    "arcface_128": {
                        "Opal": "prepared-embedding",
                    }
                },
            }
        }

    main_window = _make_worker_main_window()
    main_window.control = {"ControlA": "live-control"}
    main_window.target_faces = {"face_1": _TargetFaceWithoutEmbeddingAccess()}
    main_window.parameter_widgets = {"ControlA": control_widget}
    main_window.video_processor.prepare_issue_scan_target_faces_snapshot = (
        fake_prepare_issue_scan_target_faces_snapshot
    )

    def fake_scan_issue_frames(**kwargs):
        captured["target_faces_snapshot"] = kwargs["target_faces_snapshot"]
        return {
            "issue_frames_by_face": {},
            "frames_scanned": 1,
            "faces_with_issues": 0,
            "cancelled": False,
        }

    main_window.video_processor.scan_issue_frames = fake_scan_issue_frames

    worker = IssueScanWorker(main_window)
    worker.run()

    assert captured["target_faces_snapshot"] == {
        "face_1": {
            "face_id": "face_1",
            "embeddings_by_model": {
                "arcface_128": {
                    "Opal": "prepared-embedding",
                }
            },
        }
    }


def test_handle_issue_scan_progress_moves_slider_and_updates_abort_button(monkeypatch):
    main_window = _make_scan_main_window()
    main_window.runScanButton.tooltip = (
        "Scanning 2 marked ranges\nAbort the active issue scan."
    )
    monkeypatch.setattr(
        "app.ui.widgets.actions.video_control_actions.QtCore.QCoreApplication.processEvents",
        lambda: None,
    )

    _handle_issue_scan_progress(
        main_window,
        "Scanning 2 marked ranges",
        12,
        40,
        345,
        7.25,
    )

    assert main_window.videoSeekSlider.value() == 345
    assert main_window.videoSeekSlider.block_calls == [True, False]
    assert main_window.runScanButton.text == "Abort Scan (12/40)"
    assert main_window.runScanButton.tooltip == (
        "Scanning 2 marked ranges\nAbort the active issue scan."
    )


def test_handle_issue_scan_issue_found_merges_and_refreshes_selected_face(monkeypatch):
    main_window = _make_scan_main_window()
    main_window.issue_frames_by_face = {}
    refreshed = []
    monkeypatch.setattr(
        "app.ui.widgets.actions.video_control_actions.refresh_issue_frames_for_selected_face",
        lambda _main_window: refreshed.append(dict(_main_window.issue_frames_by_face)),
    )
    monkeypatch.setattr(
        "app.ui.widgets.actions.video_control_actions.QtCore.QCoreApplication.processEvents",
        lambda: None,
    )

    _handle_issue_scan_issue_found(main_window, "face_1", 12)
    _handle_issue_scan_issue_found(main_window, "face_1", 12)
    _handle_issue_scan_issue_found(main_window, "face_2", 8)

    assert main_window.issue_frames_by_face == {"face_1": {12}, "face_2": {8}}
    assert refreshed == [
        {"face_1": {12}},
    ]


def test_run_issue_scan_disables_controls_like_recording_when_keep_controls_off(
    monkeypatch,
):
    main_window = _make_scan_main_window(keep_controls=False)
    fake_worker = _FakeIssueScanWorker(main_window)
    disabled_calls = []

    monkeypatch.setattr(
        "app.ui.widgets.actions.video_control_actions.ui_workers.IssueScanWorker",
        lambda _main_window: fake_worker,
    )
    monkeypatch.setattr(
        "app.ui.widgets.actions.video_control_actions.Path.is_file",
        lambda self: True,
    )
    monkeypatch.setattr(
        "app.ui.widgets.actions.video_control_actions.layout_actions.disable_all_parameters_and_control_widget",
        lambda _main_window: disabled_calls.append("disabled"),
    )
    monkeypatch.setattr(
        "app.ui.widgets.actions.video_control_actions.QtCore.QCoreApplication.processEvents",
        lambda: None,
    )

    run_issue_scan(main_window)

    assert fake_worker.started is True
    assert disabled_calls == ["disabled"]
    assert main_window.scan_issue_worker is fake_worker
    assert main_window.runScanButton.enabled is True
    assert main_window.runScanButton.text == "Abort Scan"
    assert "processed" not in main_window.runScanButton.tooltip
    assert "FPS" not in main_window.runScanButton.tooltip
    assert main_window.issue_frames_by_face == {}
    assert main_window.buttonMediaPlay.enabled is False
    assert main_window.buttonMediaRecord.enabled is False
    assert main_window.scanToolsToggleButton.enabled is False
    assert main_window.prevIssueButton.enabled is False
    assert main_window.nextIssueButton.enabled is False
    assert main_window.dropFrameButton.enabled is False
    assert main_window.dropAllIssueFramesButton.enabled is False
    assert main_window.clearScanResultsButton.enabled is False
    assert main_window.clearDroppedFramesButton.enabled is False


def test_run_issue_scan_respects_keep_controls_toggle(monkeypatch):
    main_window = _make_scan_main_window(keep_controls=True)
    fake_worker = _FakeIssueScanWorker(main_window)
    disabled_calls = []

    monkeypatch.setattr(
        "app.ui.widgets.actions.video_control_actions.ui_workers.IssueScanWorker",
        lambda _main_window: fake_worker,
    )
    monkeypatch.setattr(
        "app.ui.widgets.actions.video_control_actions.Path.is_file",
        lambda self: True,
    )
    monkeypatch.setattr(
        "app.ui.widgets.actions.video_control_actions.layout_actions.disable_all_parameters_and_control_widget",
        lambda _main_window: disabled_calls.append("disabled"),
    )
    monkeypatch.setattr(
        "app.ui.widgets.actions.video_control_actions.QtCore.QCoreApplication.processEvents",
        lambda: None,
    )

    run_issue_scan(main_window)

    assert fake_worker.started is True
    assert disabled_calls == []
    assert main_window.runScanButton.enabled is True
    assert main_window.runScanButton.text == "Abort Scan"
    assert "processed" not in main_window.runScanButton.tooltip
    assert "FPS" not in main_window.runScanButton.tooltip
    assert main_window.issue_frames_by_face == {}
    assert main_window.prevIssueButton.enabled is False
    assert main_window.nextIssueButton.enabled is False
    assert main_window.dropFrameButton.enabled is False
    assert main_window.dropAllIssueFramesButton.enabled is False
    assert main_window.clearScanResultsButton.enabled is False
    assert main_window.clearDroppedFramesButton.enabled is False


def test_run_issue_scan_does_not_start_twice(monkeypatch):
    main_window = _make_scan_main_window()
    existing_worker = _FakeIssueScanWorker(main_window)
    main_window.scan_issue_worker = existing_worker
    worker_calls = []

    monkeypatch.setattr(
        "app.ui.widgets.actions.video_control_actions.ui_workers.IssueScanWorker",
        lambda _main_window: worker_calls.append("created"),
    )
    monkeypatch.setattr(
        "app.ui.widgets.actions.video_control_actions.Path.is_file",
        lambda self: True,
    )

    run_issue_scan(main_window)

    assert worker_calls == []
    assert main_window.scan_issue_worker is existing_worker


def test_toggle_issue_scan_cancels_active_worker():
    main_window = _make_scan_main_window()
    active_worker = _FakeIssueScanWorker(main_window)
    main_window.scan_issue_worker = active_worker

    toggle_issue_scan(main_window)

    assert active_worker.cancel_calls == 1


def test_issue_scan_completion_restores_slider_and_ui(monkeypatch):
    main_window = _make_scan_main_window(keep_controls=False)
    fake_worker = _FakeIssueScanWorker(main_window)
    enabled_calls = []
    toast_calls = []
    restore_calls = []

    monkeypatch.setattr(
        "app.ui.widgets.actions.video_control_actions.layout_actions.enable_all_parameters_and_control_widget",
        lambda _main_window: enabled_calls.append("enabled"),
    )
    monkeypatch.setattr(
        "app.ui.widgets.actions.video_control_actions.common_widget_actions.create_and_show_toast_message",
        lambda *_args, **_kwargs: toast_calls.append((_args, _kwargs)),
    )
    monkeypatch.setattr(
        "app.ui.widgets.actions.video_control_actions.QtCore.QCoreApplication.processEvents",
        lambda: None,
    )
    monkeypatch.setattr(
        "app.ui.widgets.actions.video_control_actions._restore_issue_scan_display",
        lambda _main_window: restore_calls.append("restored"),
    )

    main_window.scan_issue_worker = fake_worker
    main_window.scan_issue_ui_state = {
        "active": True,
        "start_frame": 24,
        "scope_text": "Scanning full clip",
        "keep_controls": False,
    }
    main_window.videoSeekSlider.setValue(90)
    main_window.video_processor.current_frame_number = 90

    _handle_issue_scan_completed(
        main_window,
        {"face_1": [1, 2]},
        50,
        1,
        "Scanning full clip",
        5.0,
        False,
    )

    assert enabled_calls == ["enabled"]
    assert main_window.videoSeekSlider.value() == 24
    assert main_window.video_processor.current_frame_number == 24
    assert main_window.scan_issue_worker is None
    assert fake_worker.deleted is True
    assert main_window.runScanButton.text == "Scan for Issues"
    assert main_window.buttonMediaPlay.enabled is True
    assert main_window.prevIssueButton.enabled is True
    assert main_window.nextIssueButton.enabled is True
    assert main_window.dropFrameButton.enabled is True
    assert main_window.dropAllIssueFramesButton.enabled is True
    assert main_window.clearScanResultsButton.enabled is True
    assert main_window.clearDroppedFramesButton.enabled is True
    assert restore_calls == ["restored"]
    assert toast_calls


def test_issue_scan_partial_completion_keeps_partial_results(monkeypatch):
    main_window = _make_scan_main_window()
    fake_worker = _FakeIssueScanWorker(main_window)
    toast_calls = []
    restore_calls = []

    monkeypatch.setattr(
        "app.ui.widgets.actions.video_control_actions.common_widget_actions.create_and_show_toast_message",
        lambda *_args, **_kwargs: toast_calls.append((_args, _kwargs)),
    )
    monkeypatch.setattr(
        "app.ui.widgets.actions.video_control_actions.QtCore.QCoreApplication.processEvents",
        lambda: None,
    )
    monkeypatch.setattr(
        "app.ui.widgets.actions.video_control_actions._restore_issue_scan_display",
        lambda _main_window: restore_calls.append("restored"),
    )

    main_window.scan_issue_worker = fake_worker
    main_window.scan_issue_ui_state = {
        "active": True,
        "start_frame": 24,
        "scope_text": "Scanning full clip",
        "keep_controls": True,
    }
    main_window.videoSeekSlider.setValue(91)

    _handle_issue_scan_completed(
        main_window,
        {"face_1": [8, 9]},
        12,
        1,
        "Scanning full clip",
        2.0,
        True,
    )

    assert main_window.videoSeekSlider.value() == 24
    assert main_window.issue_frames_by_face == {"face_1": {8, 9}}
    assert main_window.scan_issue_worker is None
    assert fake_worker.deleted is True
    assert main_window.prevIssueButton.enabled is True
    assert main_window.nextIssueButton.enabled is True
    assert main_window.dropFrameButton.enabled is True
    assert main_window.dropAllIssueFramesButton.enabled is True
    assert main_window.clearScanResultsButton.enabled is True
    assert main_window.clearDroppedFramesButton.enabled is True
    assert restore_calls == ["restored"]
    assert toast_calls[0][0][1] == "Scan Aborted"


def test_issue_scan_failed_restores_ui_and_shows_message(monkeypatch):
    main_window = _make_scan_main_window()
    fake_worker = _FakeIssueScanWorker(main_window)
    messagebox_calls = []
    restore_calls = []

    monkeypatch.setattr(
        "app.ui.widgets.actions.video_control_actions.common_widget_actions.create_and_show_messagebox",
        lambda *_args, **_kwargs: messagebox_calls.append((_args, _kwargs)),
    )
    monkeypatch.setattr(
        "app.ui.widgets.actions.video_control_actions.QtCore.QCoreApplication.processEvents",
        lambda: None,
    )
    monkeypatch.setattr(
        "app.ui.widgets.actions.video_control_actions._restore_issue_scan_display",
        lambda _main_window: restore_calls.append("restored"),
    )

    main_window.scan_issue_worker = fake_worker
    main_window.scan_issue_ui_state = {
        "active": True,
        "start_frame": 24,
        "scope_text": "Scanning full clip",
        "keep_controls": True,
    }
    main_window.videoSeekSlider.setValue(101)

    _handle_issue_scan_failed(main_window, "boom")

    assert main_window.videoSeekSlider.value() == 24
    assert main_window.scan_issue_worker is None
    assert fake_worker.deleted is True
    assert main_window.prevIssueButton.enabled is True
    assert main_window.nextIssueButton.enabled is True
    assert main_window.dropFrameButton.enabled is True
    assert main_window.dropAllIssueFramesButton.enabled is True
    assert main_window.clearScanResultsButton.enabled is True
    assert main_window.clearDroppedFramesButton.enabled is True
    assert restore_calls == ["restored"]
    assert messagebox_calls[0][0][1] == "Scan Failed"
    assert messagebox_calls[0][0][2] == "boom"


def test_issue_scan_cancelled_fallback_keeps_partial_results_message(monkeypatch):
    main_window = _make_scan_main_window()
    fake_worker = _FakeIssueScanWorker(main_window)
    toast_calls = []
    restore_calls = []

    monkeypatch.setattr(
        "app.ui.widgets.actions.video_control_actions.common_widget_actions.create_and_show_toast_message",
        lambda *_args, **_kwargs: toast_calls.append((_args, _kwargs)),
    )
    monkeypatch.setattr(
        "app.ui.widgets.actions.video_control_actions.QtCore.QCoreApplication.processEvents",
        lambda: None,
    )
    monkeypatch.setattr(
        "app.ui.widgets.actions.video_control_actions._restore_issue_scan_display",
        lambda _main_window: restore_calls.append("restored"),
    )

    main_window.issue_frames_by_face = {"face_1": {8}}
    main_window.scan_issue_worker = fake_worker
    main_window.scan_issue_ui_state = {
        "active": True,
        "start_frame": 24,
        "scope_text": "Scanning full clip",
        "keep_controls": True,
    }

    _handle_issue_scan_cancelled(main_window)

    assert main_window.scan_issue_worker is None
    assert fake_worker.deleted is True
    assert restore_calls == ["restored"]
    assert toast_calls[0][0][1] == "Scan Cancelled"
    assert "Kept any issue frames found so far." in toast_calls[0][0][2]
