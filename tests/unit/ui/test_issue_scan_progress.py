from types import SimpleNamespace

from app.ui.widgets.actions import common_actions
from app.ui.widgets.actions import card_actions, job_manager_actions, list_view_actions
from app.ui.widgets.actions import save_load_actions
from app.ui.widgets import event_filters
from app.processors.video_utils.issue_scanner import IssueScanner
from app.ui.widgets.actions.video_control_actions import (
    _handle_issue_scan_cancelled,
    _handle_issue_scan_completed,
    _handle_issue_scan_failed,
    _handle_issue_scan_issue_found,
    _handle_issue_scan_progress,
    add_video_slider_marker,
    block_if_issue_scan_active,
    is_issue_scan_active,
    remove_all_markers,
    run_issue_scan,
    toggle_issue_scan,
    update_scan_review_button_states,
)
from app.ui.widgets import widget_components
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
        self.block_calls = []

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

    def isEnabled(self):
        return self.enabled

    def blockSignals(self, value):
        self.block_calls.append(bool(value))


class _DummySlider:
    def __init__(self, value=0):
        self._value = value
        self.enabled = True
        self.block_calls = []
        self.updated = 0

    def value(self):
        return self._value

    def setEnabled(self, value):
        self.enabled = bool(value)

    def setDisabled(self, value):
        self.enabled = not bool(value)

    def isEnabled(self):
        return self.enabled

    def blockSignals(self, value):
        self.block_calls.append(bool(value))

    def setValue(self, value):
        self._value = int(value)

    def update(self):
        self.updated += 1


class _DummyLineEdit:
    def __init__(self, text=""):
        self._text = str(text)
        self.enabled = True
        self.block_calls = []

    def text(self):
        return self._text

    def setText(self, text):
        self._text = str(text)

    def setEnabled(self, value):
        self.enabled = bool(value)

    def setDisabled(self, value):
        self.enabled = not bool(value)

    def isEnabled(self):
        return self.enabled

    def blockSignals(self, value):
        self.block_calls.append(bool(value))


def _make_guarded_card(main_window, checked=False):
    card = _DummyButton(checked=checked)
    card.main_window = main_window
    card._restore_pre_click_checked_state = lambda: (
        widget_components.CardButton._restore_pre_click_checked_state(card)
    )
    return card


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


def _issue_scan_unavailable_reason(
    control, scan_ranges=None, markers=None, fallback_control=None
):
    """Mirror the VideoProcessor facade, which delegates to the IssueScanner.

    The VR180 check only reads the control/marker dictionaries handed to it, so a
    scanner without media state is enough.
    """
    scanner = IssueScanner(main_window=None, sequential_detector=None)  # type: ignore[arg-type]
    return scanner.get_issue_scan_unavailable_reason(
        control, scan_ranges, markers, fallback_control
    )


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
        target_faces={"face_1": _DummyButton("Face 1")},
        target_videos={"media_1": _DummyButton("Media 1")},
        input_faces={"input_1": _DummyButton("Input 1")},
        merged_embeddings={"embed_1": _DummyButton("Embedding 1")},
        parameter_widgets={},
        issue_frames_by_face={"face_1": {3, 5}, "face_2": {9}},
        issue_frames=set(),
        dropped_frames=set(),
        markers={},
        job_marker_pairs=[],
        selected_target_face_id="face_1",
        selected_video_button=SimpleNamespace(file_type="video"),
        scan_issue_worker=None,
        scan_issue_ui_state={},
        videoSeekSlider=slider,
        targetVideosList=_DummyButton("Target Videos"),
        inputFacesList=_DummyButton("Input Faces"),
        inputEmbeddingsList=_DummyButton("Input Embeddings"),
        jobQueueList=_DummyButton("Job Queue"),
        buttonTargetVideosPath=_DummyButton("Target Path"),
        buttonInputFacesPath=_DummyButton("Input Path"),
        targetVideosFilterMenuButton=_DummyButton("Target Media Filter Menu"),
        targetVideosFilterWebcamsCheckBox=_DummyButton("Filter Webcams"),
        findTargetFacesButton=_DummyButton("Find Faces"),
        clearTargetFacesButton=_DummyButton("Clear Faces"),
        addMarkerButton=_DummyButton("Add Marker"),
        removeMarkerButton=_DummyButton("Remove Marker"),
        frameAdvanceButton=_DummyButton("Frame Advance"),
        frameRewindButton=_DummyButton("Frame Rewind"),
        nextMarkerButton=_DummyButton("Next Marker"),
        previousMarkerButton=_DummyButton("Previous Marker"),
        swapfacesButton=_DummyButton("Swap Faces"),
        editFacesButton=_DummyButton("Edit Faces"),
        openEmbeddingButton=_DummyButton("Open Embeddings"),
        loadJobButton=_DummyButton("Load Job"),
        buttonProcessAll=_DummyButton("Process All"),
        buttonProcessSelected=_DummyButton("Process Selected"),
        actionLoad_SavedWorkspace=_DummyButton("Load Workspace"),
        actionOpen_Videos_Folder=_DummyButton("Open Videos Folder"),
        actionOpen_Video_Files=_DummyButton("Open Video Files"),
        actionLoad_Source_Image_Files=_DummyButton("Load Source Files"),
        actionLoad_Source_Images_Folder=_DummyButton("Load Source Folder"),
        actionLoad_Embeddings=_DummyButton("Load Embeddings"),
        videoSeekLineEdit=_DummyLineEdit("24"),
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
        placeholder_update_signal=_DummySignal(),
    )
    main_window.video_processor = SimpleNamespace(
        file_type="video",
        media_path="dummy.mp4",
        current_frame_number=24,
        current_frame=None,
        stop_processing=lambda: False,
        process_current_frame=lambda: None,
        _get_issue_scan_ranges=lambda: [(0, 24)],
        get_issue_scan_unavailable_reason=_issue_scan_unavailable_reason,
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


def test_issue_scan_worker_prepares_target_snapshot_during_construction():
    main_window = _make_worker_main_window()
    prep_calls = []
    captured = {}

    def fake_prepare_issue_scan_target_faces_snapshot(*_args, **_kwargs):
        prep_calls.append("prepared")
        return {"face_1": {"face_id": "face_1", "embeddings_by_model": {}}}

    def fake_scan_issue_frames(**kwargs):
        captured["target_faces_snapshot"] = kwargs["target_faces_snapshot"]
        return {
            "issue_frames_by_face": {},
            "frames_scanned": 1,
            "faces_with_issues": 0,
            "cancelled": False,
        }

    main_window.video_processor.prepare_issue_scan_target_faces_snapshot = (
        fake_prepare_issue_scan_target_faces_snapshot
    )
    main_window.video_processor.scan_issue_frames = fake_scan_issue_frames

    worker = IssueScanWorker(main_window)

    assert prep_calls == ["prepared"]

    worker.run()

    assert captured["target_faces_snapshot"] == {
        "face_1": {"face_id": "face_1", "embeddings_by_model": {}}
    }


def test_issue_scan_worker_passes_control_defaults_snapshot():
    control_widget = SimpleNamespace(default_value="default-control")
    main_window = _make_worker_main_window()
    main_window.control = {
        "DetectorModelSelection": "SCRFD",
        "IgnoredControl": "live-control",
    }
    main_window.parameter_widgets = {
        "DetectorModelSelection": control_widget,
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

    assert captured["control_defaults_snapshot"] == {
        "DetectorModelSelection": "default-control"
    }


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
    main_window.control = {
        "DetectorModelSelection": "SCRFD",
        "IgnoredControl": "live-control",
    }
    main_window.target_faces = {"face_1": _TargetFaceWithoutEmbeddingAccess()}
    main_window.parameter_widgets = {"DetectorModelSelection": control_widget}
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


def test_issue_scan_worker_filters_snapshot_control_and_params():
    main_window = _make_worker_main_window()
    main_window.control = {
        "DetectorScoreSlider": 42,
        "FaceTrackingEnableToggle": True,
        "IgnoredControl": "skip",
    }
    main_window.parameters = {
        "face_1": {
            "SimilarityThresholdSlider": 77,
            "FaceExpressionEnableBothToggle": True,
        },
        "face_2": {
            "SimilarityThresholdSlider": 61,
        },
    }
    main_window.target_faces = {"face_1": object()}
    captured = {}

    def fake_scan_issue_frames(**kwargs):
        captured["base_control"] = kwargs["base_control"]
        captured["base_params"] = kwargs["base_params"]
        return {
            "issue_frames_by_face": {},
            "frames_scanned": 1,
            "faces_with_issues": 0,
            "cancelled": False,
        }

    main_window.video_processor.scan_issue_frames = fake_scan_issue_frames

    worker = IssueScanWorker(main_window)
    worker.run()

    assert captured["base_control"] == {
        "DetectorScoreSlider": 42,
        "FaceTrackingEnableToggle": True,
    }
    assert captured["base_params"] == {
        "face_1": {"SimilarityThresholdSlider": 77},
    }


def test_issue_scan_worker_does_not_pass_fixed_target_height():
    main_window = _make_worker_main_window()
    captured = {}

    def fake_scan_issue_frames(**kwargs):
        captured.update(kwargs)
        return {
            "issue_frames_by_face": {},
            "frames_scanned": 1,
            "faces_with_issues": 0,
            "cancelled": False,
        }

    main_window.video_processor.scan_issue_frames = fake_scan_issue_frames

    worker = IssueScanWorker(main_window)
    worker.run()

    assert "target_height" not in captured


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


def test_handle_issue_scan_issue_found_keeps_review_controls_disabled_while_active(
    monkeypatch,
):
    main_window = _make_scan_main_window()
    main_window.issue_frames_by_face = {}
    main_window.scan_issue_worker = object()
    main_window.prevIssueButton.enabled = False
    main_window.nextIssueButton.enabled = False
    main_window.dropAllIssueFramesButton.enabled = False
    monkeypatch.setattr(
        "app.ui.widgets.actions.video_control_actions.QtCore.QCoreApplication.processEvents",
        lambda: None,
    )

    _handle_issue_scan_issue_found(main_window, "face_1", 12)

    assert main_window.issue_frames_by_face == {"face_1": {12}}
    assert main_window.prevIssueButton.enabled is False
    assert main_window.nextIssueButton.enabled is False
    assert main_window.dropAllIssueFramesButton.enabled is False


def test_handle_issue_scan_progress_records_processed_frames(monkeypatch):
    main_window = _make_scan_main_window()
    monkeypatch.setattr(
        "app.ui.widgets.actions.video_control_actions.QtCore.QCoreApplication.processEvents",
        lambda: None,
    )

    _handle_issue_scan_progress(
        main_window,
        "Scanning full clip",
        7,
        20,
        88,
        5.5,
    )

    assert main_window.scan_issue_ui_state["frames_scanned"] == 7


def test_update_scan_review_button_states_respects_active_scan_state():
    main_window = _make_scan_main_window()
    main_window.prevIssueButton.enabled = False
    main_window.nextIssueButton.enabled = False
    main_window.dropAllIssueFramesButton.enabled = False
    main_window.runScanButton.enabled = False

    update_scan_review_button_states(main_window)

    assert main_window.runScanButton.enabled is True
    assert main_window.prevIssueButton.enabled is True
    assert main_window.nextIssueButton.enabled is True
    assert main_window.dropAllIssueFramesButton.enabled is True

    main_window.scan_issue_worker = object()
    update_scan_review_button_states(main_window)

    assert main_window.runScanButton.enabled is True
    assert main_window.prevIssueButton.enabled is False
    assert main_window.nextIssueButton.enabled is False
    assert main_window.dropAllIssueFramesButton.enabled is False


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
    assert main_window.scan_issue_ui_state["frames_scanned"] == 0
    assert main_window.runScanButton.enabled is True
    assert main_window.runScanButton.text == "Abort Scan"
    assert "processed" not in main_window.runScanButton.tooltip
    assert "FPS" not in main_window.runScanButton.tooltip
    assert main_window.issue_frames_by_face == {}
    assert main_window.buttonMediaPlay.enabled is False
    assert main_window.buttonMediaRecord.enabled is False
    assert main_window.scanToolsToggleButton.enabled is False
    assert main_window.findTargetFacesButton.enabled is False
    assert main_window.clearTargetFacesButton.enabled is False
    assert main_window.targetVideosFilterMenuButton.enabled is False
    assert main_window.addMarkerButton.enabled is False
    assert main_window.removeMarkerButton.enabled is False
    assert main_window.videoSeekSlider.enabled is False
    assert main_window.videoSeekLineEdit.enabled is False
    assert main_window.frameAdvanceButton.enabled is False
    assert main_window.frameRewindButton.enabled is False
    assert main_window.nextMarkerButton.enabled is False
    assert main_window.previousMarkerButton.enabled is False
    assert main_window.swapfacesButton.enabled is False
    assert main_window.editFacesButton.enabled is False
    assert main_window.openEmbeddingButton.enabled is False
    assert main_window.buttonTargetVideosPath.enabled is False
    assert main_window.buttonInputFacesPath.enabled is False
    assert main_window.targetVideosList.enabled is False
    assert main_window.inputFacesList.enabled is False
    assert main_window.inputEmbeddingsList.enabled is False
    assert main_window.loadJobButton.enabled is False
    assert main_window.buttonProcessAll.enabled is False
    assert main_window.buttonProcessSelected.enabled is False
    assert main_window.actionLoad_SavedWorkspace.enabled is False
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
    main_window.frameAdvanceButton.enabled = False
    main_window.openEmbeddingButton.enabled = False
    main_window.targetVideosFilterMenuButton.enabled = False

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
    assert main_window.findTargetFacesButton.enabled is False
    assert main_window.clearTargetFacesButton.enabled is False
    assert main_window.targetVideosFilterMenuButton.enabled is False
    assert main_window.addMarkerButton.enabled is False
    assert main_window.removeMarkerButton.enabled is False
    assert main_window.videoSeekSlider.enabled is False
    assert main_window.videoSeekLineEdit.enabled is False
    assert main_window.frameAdvanceButton.enabled is False
    assert main_window.frameRewindButton.enabled is False
    assert main_window.nextMarkerButton.enabled is False
    assert main_window.previousMarkerButton.enabled is False
    assert main_window.swapfacesButton.enabled is False
    assert main_window.editFacesButton.enabled is False
    assert main_window.openEmbeddingButton.enabled is False
    assert main_window.buttonTargetVideosPath.enabled is False
    assert main_window.buttonInputFacesPath.enabled is False
    assert main_window.targetVideosList.enabled is False
    assert main_window.inputFacesList.enabled is False
    assert main_window.inputEmbeddingsList.enabled is False
    assert main_window.loadJobButton.enabled is False
    assert main_window.buttonProcessAll.enabled is False
    assert main_window.buttonProcessSelected.enabled is False
    assert main_window.actionLoad_SavedWorkspace.enabled is False
    assert main_window.target_faces["face_1"].enabled is True
    assert main_window.prevIssueButton.enabled is False
    assert main_window.nextIssueButton.enabled is False
    assert main_window.dropFrameButton.enabled is False
    assert main_window.dropAllIssueFramesButton.enabled is False
    assert main_window.clearScanResultsButton.enabled is False
    assert main_window.clearDroppedFramesButton.enabled is False


def test_seek_line_edit_event_filter_blocks_manual_seek_while_scan_active():
    main_window = _make_scan_main_window()
    main_window.scan_issue_worker = object()
    processed = []
    main_window.video_processor.process_current_frame = lambda: processed.append(
        "processed"
    )
    main_window.videoSeekLineEdit.setText("77")
    event_filter = event_filters.videoSeekSliderLineEditEventFilter(main_window)
    event = SimpleNamespace(
        type=lambda: event_filters.QtCore.QEvent.KeyPress,
        key=lambda: event_filters.QtCore.Qt.Key_Return,
    )

    handled = event_filter.eventFilter(main_window.videoSeekLineEdit, event)

    assert handled is True
    assert main_window.videoSeekSlider.value() == 24
    assert main_window.videoSeekLineEdit.text() == "77"
    assert processed == []


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


def test_run_issue_scan_blocks_vr180_mode(monkeypatch):
    main_window = _make_scan_main_window()
    main_window.control["VR180ModeEnableToggle"] = True
    messagebox_calls = []
    worker_calls = []

    monkeypatch.setattr(
        "app.ui.widgets.actions.video_control_actions.common_widget_actions.create_and_show_messagebox",
        lambda *_args, **_kwargs: messagebox_calls.append((_args, _kwargs)),
    )
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
    assert messagebox_calls[0][0][1] == "Scan Not Available"
    assert (
        messagebox_calls[0][0][2]
        == "Issue scans are not supported while VR180 mode is enabled."
    )


def test_run_issue_scan_blocks_marker_enabled_vr180_mode(monkeypatch):
    main_window = _make_scan_main_window()
    main_window.markers = {12: {"control": {"VR180ModeEnableToggle": True}}}
    main_window.video_processor._get_issue_scan_ranges = lambda: [(0, 24)]
    messagebox_calls = []
    worker_calls = []

    monkeypatch.setattr(
        "app.ui.widgets.actions.video_control_actions.common_widget_actions.create_and_show_messagebox",
        lambda *_args, **_kwargs: messagebox_calls.append((_args, _kwargs)),
    )
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
    assert messagebox_calls[0][0][1] == "Scan Not Available"
    assert (
        messagebox_calls[0][0][2]
        == "Issue scans are not supported while VR180 mode is enabled."
    )


def test_run_issue_scan_reports_construction_failures_without_clearing_results(
    monkeypatch,
):
    main_window = _make_scan_main_window()
    main_window.issue_frames_by_face = {"face_1": {8}}
    messagebox_calls = []

    def raise_on_snapshot(_main_window):
        raise RuntimeError("prep boom")

    monkeypatch.setattr(
        "app.ui.widgets.actions.video_control_actions.ui_workers.IssueScanWorker",
        raise_on_snapshot,
    )
    monkeypatch.setattr(
        "app.ui.widgets.actions.video_control_actions.Path.is_file",
        lambda self: True,
    )
    monkeypatch.setattr(
        "app.ui.widgets.actions.video_control_actions.common_widget_actions.create_and_show_messagebox",
        lambda *_args, **_kwargs: messagebox_calls.append((_args, _kwargs)),
    )

    run_issue_scan(main_window)

    assert main_window.issue_frames_by_face == {"face_1": {8}}
    assert main_window.scan_issue_worker is None
    assert main_window.scan_issue_ui_state == {}
    assert messagebox_calls[0][0][1] == "Scan Failed"
    assert messagebox_calls[0][0][2] == "prep boom"


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
        "frames_scanned": 50,
        "mutation_lock_enabled_states": [
            (main_window.findTargetFacesButton, True),
            (main_window.clearTargetFacesButton, True),
            (main_window.targetVideosFilterMenuButton, True),
            (main_window.videoSeekSlider, True),
            (main_window.videoSeekLineEdit, True),
            (main_window.frameAdvanceButton, False),
            (main_window.frameRewindButton, True),
            (main_window.nextMarkerButton, True),
            (main_window.previousMarkerButton, False),
            (main_window.swapfacesButton, True),
            (main_window.editFacesButton, False),
            (main_window.openEmbeddingButton, False),
            (main_window.loadJobButton, True),
        ],
    }
    main_window.findTargetFacesButton.enabled = False
    main_window.clearTargetFacesButton.enabled = False
    main_window.targetVideosFilterMenuButton.enabled = False
    main_window.videoSeekSlider.enabled = False
    main_window.videoSeekLineEdit.enabled = False
    main_window.frameAdvanceButton.enabled = False
    main_window.frameRewindButton.enabled = False
    main_window.nextMarkerButton.enabled = False
    main_window.previousMarkerButton.enabled = False
    main_window.swapfacesButton.enabled = False
    main_window.editFacesButton.enabled = False
    main_window.openEmbeddingButton.enabled = False
    main_window.loadJobButton.enabled = False
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
    assert main_window.findTargetFacesButton.enabled is True
    assert main_window.clearTargetFacesButton.enabled is True
    assert main_window.targetVideosFilterMenuButton.enabled is True
    assert main_window.videoSeekSlider.enabled is True
    assert main_window.videoSeekLineEdit.enabled is True
    assert main_window.frameAdvanceButton.enabled is False
    assert main_window.frameRewindButton.enabled is True
    assert main_window.nextMarkerButton.enabled is True
    assert main_window.previousMarkerButton.enabled is False
    assert main_window.swapfacesButton.enabled is True
    assert main_window.editFacesButton.enabled is False
    assert main_window.openEmbeddingButton.enabled is False
    assert main_window.loadJobButton.enabled is True
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
        "frames_scanned": 12,
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


def test_issue_scan_failed_without_progress_keeps_current_attempt_results_only(
    monkeypatch,
):
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
        "frames_scanned": 0,
        "mutation_lock_enabled_states": [
            (main_window.videoSeekSlider, True),
            (main_window.videoSeekLineEdit, True),
            (main_window.frameAdvanceButton, False),
            (main_window.targetVideosFilterMenuButton, True),
        ],
    }
    main_window.videoSeekSlider.setValue(101)
    main_window.issue_frames_by_face = {}
    main_window.videoSeekSlider.enabled = False
    main_window.videoSeekLineEdit.enabled = False
    main_window.frameAdvanceButton.enabled = False
    main_window.targetVideosFilterMenuButton.enabled = False

    _handle_issue_scan_failed(main_window, "boom")

    assert main_window.videoSeekSlider.value() == 24
    assert main_window.issue_frames_by_face == {}
    assert main_window.scan_issue_worker is None
    assert fake_worker.deleted is True
    assert main_window.videoSeekSlider.enabled is True
    assert main_window.videoSeekLineEdit.enabled is True
    assert main_window.frameAdvanceButton.enabled is False
    assert main_window.targetVideosFilterMenuButton.enabled is True
    assert main_window.prevIssueButton.enabled is True
    assert main_window.nextIssueButton.enabled is True
    assert main_window.dropFrameButton.enabled is True
    assert main_window.dropAllIssueFramesButton.enabled is True
    assert main_window.clearScanResultsButton.enabled is True
    assert main_window.clearDroppedFramesButton.enabled is True
    assert restore_calls == ["restored"]
    assert messagebox_calls[0][0][1] == "Scan Failed"
    assert "boom" in messagebox_calls[0][0][2]
    assert "Any previous issue findings were cleared" in messagebox_calls[0][0][2]


def test_issue_scan_failed_after_progress_keeps_current_results(monkeypatch):
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
        "frames_scanned": 4,
    }
    main_window.videoSeekSlider.setValue(101)
    main_window.issue_frames_by_face = {}

    _handle_issue_scan_failed(main_window, "boom")

    assert main_window.videoSeekSlider.value() == 24
    assert main_window.issue_frames_by_face == {}
    assert main_window.scan_issue_worker is None
    assert fake_worker.deleted is True
    assert restore_calls == ["restored"]
    assert messagebox_calls[0][0][1] == "Scan Failed"
    assert "boom" in messagebox_calls[0][0][2]


def test_issue_scan_failed_after_partial_hits_keeps_current_attempt_findings(
    monkeypatch,
):
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
        "frames_scanned": 4,
    }
    main_window.videoSeekSlider.setValue(101)
    main_window.issue_frames_by_face = {"face_1": {8}}

    _handle_issue_scan_failed(main_window, "boom")

    assert main_window.videoSeekSlider.value() == 24
    assert main_window.issue_frames_by_face == {"face_1": {8}}
    assert main_window.scan_issue_worker is None
    assert fake_worker.deleted is True
    assert restore_calls == ["restored"]
    assert messagebox_calls[0][0][1] == "Scan Failed"
    assert (
        "Only findings from the current scan attempt remain visible."
        in (messagebox_calls[0][0][2])
    )


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
        "frames_scanned": 12,
    }

    _handle_issue_scan_cancelled(main_window)

    assert main_window.scan_issue_worker is None
    assert fake_worker.deleted is True
    assert restore_calls == ["restored"]
    assert toast_calls[0][0][1] == "Scan Cancelled"
    assert "Kept 1 issue frames from this scan attempt." in toast_calls[0][0][2]


def test_issue_scan_cancelled_completion_after_progress_keeps_empty_results(
    monkeypatch,
):
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
    main_window.issue_frames_by_face = {}
    main_window.scan_issue_ui_state = {
        "active": True,
        "start_frame": 24,
        "scope_text": "Scanning full clip",
        "keep_controls": True,
        "frames_scanned": 12,
    }

    _handle_issue_scan_completed(
        main_window,
        {},
        12,
        0,
        "Scanning full clip",
        2.0,
        True,
    )

    assert main_window.issue_frames_by_face == {}
    assert main_window.scan_issue_worker is None
    assert fake_worker.deleted is True
    assert restore_calls == ["restored"]
    assert toast_calls[0][0][1] == "Scan Aborted"
    assert "Kept 0 issue frames from this scan attempt." in toast_calls[0][0][2]


def test_issue_scan_cancelled_completion_without_progress_keeps_empty_results(
    monkeypatch,
):
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
    main_window.issue_frames_by_face = {}
    main_window.scan_issue_ui_state = {
        "active": True,
        "start_frame": 24,
        "scope_text": "Scanning full clip",
        "keep_controls": True,
        "frames_scanned": 0,
    }

    _handle_issue_scan_completed(
        main_window,
        {},
        0,
        0,
        "Scanning full clip",
        2.0,
        True,
    )

    assert main_window.issue_frames_by_face == {}
    assert main_window.scan_issue_worker is None
    assert fake_worker.deleted is True
    assert restore_calls == ["restored"]
    assert toast_calls[0][0][1] == "Scan Aborted"
    assert "Kept 0 issue frames from this scan attempt." in toast_calls[0][0][2]


def test_issue_scan_cancelled_without_progress_keeps_empty_results(monkeypatch):
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
    main_window.issue_frames_by_face = {}
    main_window.scan_issue_ui_state = {
        "active": True,
        "start_frame": 24,
        "scope_text": "Scanning full clip",
        "keep_controls": True,
        "frames_scanned": 0,
        "mutation_lock_enabled_states": [
            (main_window.nextMarkerButton, True),
            (main_window.swapfacesButton, True),
            (main_window.targetVideosFilterMenuButton, False),
            (main_window.videoSeekLineEdit, True),
        ],
    }
    main_window.nextMarkerButton.enabled = False
    main_window.swapfacesButton.enabled = False
    main_window.targetVideosFilterMenuButton.enabled = False

    _handle_issue_scan_cancelled(main_window)

    assert main_window.issue_frames_by_face == {}
    assert main_window.scan_issue_worker is None
    assert fake_worker.deleted is True
    assert main_window.nextMarkerButton.enabled is True
    assert main_window.swapfacesButton.enabled is True
    assert main_window.targetVideosFilterMenuButton.enabled is False
    assert main_window.videoSeekLineEdit.enabled is True
    assert restore_calls == ["restored"]
    assert toast_calls[0][0][1] == "Scan Cancelled"
    assert "Kept 0 issue frames from this scan attempt." in toast_calls[0][0][2]


def test_issue_scan_cancelled_after_progress_keeps_current_results(monkeypatch):
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

    main_window.issue_frames_by_face = {}
    main_window.scan_issue_worker = fake_worker
    main_window.scan_issue_ui_state = {
        "active": True,
        "start_frame": 24,
        "scope_text": "Scanning full clip",
        "keep_controls": True,
        "frames_scanned": 6,
    }

    _handle_issue_scan_cancelled(main_window)

    assert main_window.issue_frames_by_face == {}
    assert main_window.scan_issue_worker is None
    assert fake_worker.deleted is True
    assert restore_calls == ["restored"]
    assert toast_calls[0][0][1] == "Scan Cancelled"
    assert "Kept 0 issue frames from this scan attempt." in toast_calls[0][0][2]


def test_filter_target_videos_defers_refresh_while_scan_is_active(monkeypatch):
    main_window = _make_scan_main_window()
    main_window.scan_issue_worker = object()
    calls = []

    monkeypatch.setattr(
        "app.ui.widgets.actions.list_view_actions.filter_actions.filter_target_videos",
        lambda _main_window: calls.append("filtered"),
    )
    monkeypatch.setattr(
        "app.ui.widgets.actions.list_view_actions.load_target_webcams",
        lambda _main_window: calls.append("webcams"),
    )

    list_view_actions.filter_target_videos(main_window)

    assert calls == []
    assert main_window.scan_issue_ui_state["pending_target_media_refresh"] is True


def test_issue_scan_completion_replays_deferred_target_media_refresh_once(monkeypatch):
    main_window = _make_scan_main_window(keep_controls=False)
    fake_worker = _FakeIssueScanWorker(main_window)
    enabled_calls = []
    restore_calls = []
    replay_calls = []
    toast_calls = []

    monkeypatch.setattr(
        "app.ui.widgets.actions.video_control_actions.layout_actions.enable_all_parameters_and_control_widget",
        lambda _main_window: enabled_calls.append("enabled"),
    )
    monkeypatch.setattr(
        "app.ui.widgets.actions.video_control_actions.QtCore.QCoreApplication.processEvents",
        lambda: None,
    )
    monkeypatch.setattr(
        "app.ui.widgets.actions.video_control_actions._restore_issue_scan_display",
        lambda _main_window: restore_calls.append("restored"),
    )
    monkeypatch.setattr(
        "app.ui.widgets.actions.list_view_actions.filter_actions.filter_target_videos",
        lambda _main_window: replay_calls.append("filtered"),
    )
    monkeypatch.setattr(
        "app.ui.widgets.actions.list_view_actions.load_target_webcams",
        lambda _main_window: replay_calls.append("webcams"),
    )
    monkeypatch.setattr(
        "app.ui.widgets.actions.video_control_actions.common_widget_actions.create_and_show_toast_message",
        lambda *_args, **_kwargs: toast_calls.append((_args, _kwargs)),
    )

    main_window.scan_issue_worker = fake_worker
    main_window.scan_issue_ui_state = {
        "active": True,
        "start_frame": 24,
        "scope_text": "Scanning full clip",
        "keep_controls": False,
        "frames_scanned": 50,
        "pending_target_media_refresh": True,
        "mutation_lock_enabled_states": [
            (main_window.findTargetFacesButton, True),
            (main_window.targetVideosFilterMenuButton, True),
            (main_window.videoSeekSlider, True),
        ],
    }
    main_window.findTargetFacesButton.enabled = False
    main_window.targetVideosFilterMenuButton.enabled = False
    main_window.videoSeekSlider.enabled = False
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
    assert replay_calls == ["filtered", "webcams"]
    assert main_window.scan_issue_ui_state == {}
    assert restore_calls == ["restored"]
    assert fake_worker.deleted is True
    assert toast_calls[0][0][1] == "Scan Complete"


def test_issue_scan_completion_replays_deferred_refresh_after_scan_is_inactive(
    monkeypatch,
):
    main_window = _make_scan_main_window(keep_controls=False)
    fake_worker = _FakeIssueScanWorker(main_window)
    replay_calls = []

    class _FakeTargetMediaLoaderWorker:
        def __init__(self, main_window, webcam_mode=False):
            replay_calls.append(("worker_init", webcam_mode))
            self.main_window = main_window
            self.webcam_mode = webcam_mode
            self.webcam_thumbnail_ready = _DummySignal()

        def start(self):
            replay_calls.append(("worker_start", self.webcam_mode))

    monkeypatch.setattr(
        "app.ui.widgets.actions.video_control_actions.layout_actions.enable_all_parameters_and_control_widget",
        lambda _main_window: None,
    )
    monkeypatch.setattr(
        "app.ui.widgets.actions.video_control_actions.QtCore.QCoreApplication.processEvents",
        lambda: None,
    )
    monkeypatch.setattr(
        "app.ui.widgets.actions.video_control_actions._restore_issue_scan_display",
        lambda _main_window: None,
    )
    monkeypatch.setattr(
        "app.ui.widgets.actions.list_view_actions.filter_actions.filter_target_videos",
        lambda _main_window: replay_calls.append(
            ("filter", is_issue_scan_active(_main_window))
        ),
    )
    monkeypatch.setattr(
        "app.ui.widgets.actions.list_view_actions.ui_workers.TargetMediaLoaderWorker",
        _FakeTargetMediaLoaderWorker,
    )
    monkeypatch.setattr(
        "app.ui.widgets.actions.video_control_actions.common_widget_actions.create_and_show_toast_message",
        lambda *_args, **_kwargs: None,
    )

    main_window.targetVideosFilterWebcamsCheckBox.setChecked(True)
    main_window.scan_issue_worker = fake_worker
    main_window.scan_issue_ui_state = {
        "active": True,
        "start_frame": 24,
        "scope_text": "Scanning full clip",
        "keep_controls": False,
        "frames_scanned": 50,
        "pending_target_media_refresh": True,
        "mutation_lock_enabled_states": [
            (main_window.findTargetFacesButton, True),
            (main_window.targetVideosFilterMenuButton, True),
            (main_window.videoSeekSlider, True),
        ],
    }

    _handle_issue_scan_completed(
        main_window,
        {"face_1": [1, 2]},
        50,
        1,
        "Scanning full clip",
        5.0,
        False,
    )

    assert replay_calls == [
        ("filter", False),
        ("worker_init", True),
        ("worker_start", True),
    ]
    assert main_window.scan_issue_worker is None
    assert main_window.scan_issue_ui_state == {}


def test_issue_scan_guard_helpers_block_structural_mutations(monkeypatch):
    main_window = _make_scan_main_window(keep_controls=True)
    main_window.scan_issue_worker = object()
    toast_calls = []

    monkeypatch.setattr(
        "app.ui.widgets.actions.video_control_actions.common_widget_actions.create_and_show_toast_message",
        lambda *_args, **_kwargs: toast_calls.append((_args, _kwargs)),
    )
    monkeypatch.setattr(
        "app.ui.widgets.actions.card_actions.common_widget_actions.create_and_show_toast_message",
        lambda *_args, **_kwargs: toast_calls.append((_args, _kwargs)),
    )
    monkeypatch.setattr(
        "app.ui.widgets.actions.job_manager_actions.common_widget_actions.create_and_show_toast_message",
        lambda *_args, **_kwargs: toast_calls.append((_args, _kwargs)),
    )

    assert is_issue_scan_active(main_window) is True
    assert block_if_issue_scan_active(main_window, "load a workspace") is True

    original_target_videos = dict(main_window.target_videos)
    original_input_faces = dict(main_window.input_faces)
    original_embeddings = dict(main_window.merged_embeddings)

    card_actions.find_target_faces(main_window)
    list_view_actions.select_target_medias(
        main_window, "files", files_list=["clip.mp4"]
    )
    save_load_actions.load_saved_workspace(main_window, "workspace.json")
    job_manager_actions.load_job(main_window)
    add_video_slider_marker(main_window)
    remove_all_markers(main_window)

    widget_components.TargetMediaCardButton.load_media(
        _make_guarded_card(main_window, checked=True)
    )
    widget_components.TargetFaceCardButton.remove_target_face_from_list(
        SimpleNamespace(main_window=main_window)
    )
    widget_components.InputFaceCardButton.load_input_face(
        _make_guarded_card(main_window, checked=True)
    )
    widget_components.EmbeddingCardButton.load_embedding(
        _make_guarded_card(main_window, checked=True)
    )

    assert main_window.target_videos == original_target_videos
    assert main_window.input_faces == original_input_faces
    assert main_window.merged_embeddings == original_embeddings
    assert toast_calls


def test_scan_guard_restores_target_media_card_checked_state(monkeypatch):
    main_window = _make_scan_main_window(keep_controls=True)
    main_window.scan_issue_worker = object()
    toast_calls = []

    monkeypatch.setattr(
        "app.ui.widgets.actions.video_control_actions.common_widget_actions.create_and_show_toast_message",
        lambda *_args, **_kwargs: toast_calls.append((_args, _kwargs)),
    )

    checked_card = _make_guarded_card(main_window, checked=False)
    widget_components.TargetMediaCardButton.load_media(checked_card)

    unchecked_card = _make_guarded_card(main_window, checked=True)
    widget_components.TargetMediaCardButton.load_media(unchecked_card)

    assert checked_card.checked is True
    assert unchecked_card.checked is False
    assert checked_card.block_calls == [True, False]
    assert unchecked_card.block_calls == [True, False]
    assert len(toast_calls) == 2


def test_target_media_load_clears_single_frame_preview_caches(monkeypatch):
    class _Capture:
        def __init__(self):
            self.released = False
            self.set_calls = []

        def isOpened(self):
            return True

        def set(self, prop, value):
            self.set_calls.append((prop, value))

        def get(self, prop):
            if prop == widget_components.cv2.CAP_PROP_FRAME_COUNT:
                return 5
            if prop == widget_components.cv2.CAP_PROP_FPS:
                return 24
            return 0

        def release(self):
            self.released = True

    refresh_calls = []
    clear_calls = []
    capture = _Capture()
    toggled = []

    main_window = SimpleNamespace(
        selected_video_button=SimpleNamespace(toggle=lambda: toggled.append(True)),
        selected_target_face_id=None,
        current_widget_parameters={},
        parameters={},
        control={
            "AutoSwapToggle": False,
            "SendVirtCamFramesEnableToggle": False,
        },
        video_processor=SimpleNamespace(
            stop_processing=lambda: False,
            _clear_single_frame_preview_caches=lambda: clear_calls.append(True),
            current_frame_number=99,
            media_path="old.mp4",
            current_frame=[],
            media_capture=None,
            media_rotation=0,
            fps=0,
            max_frame_number=0,
            next_frame_to_display=99,
            file_type=None,
        ),
        scene=SimpleNamespace(clear=lambda: None),
        graphicsViewFrame=SimpleNamespace(update=lambda: None),
        videoSeekSlider=SimpleNamespace(
            blockSignals=lambda *_args, **_kwargs: None,
            setMaximum=lambda *_args, **_kwargs: None,
            setValue=lambda *_args, **_kwargs: None,
            rangeChanged=SimpleNamespace(emit=lambda *_args: None),
        ),
        loading_new_media=False,
    )

    card = SimpleNamespace(
        main_window=main_window,
        file_type="video",
        media_path="new.mp4",
        media_capture=None,
        reset_related_widgets_and_values=lambda: None,
        _restore_pre_click_checked_state=lambda: None,
        _toggle_timeline_visibility=lambda *_args, **_kwargs: None,
    )

    monkeypatch.setattr(
        "app.ui.widgets.widget_components.get_video_rotation", lambda *_args: 0
    )
    monkeypatch.setattr(
        "app.ui.widgets.widget_components.misc_helpers.check_and_warn_vfr",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "app.ui.widgets.widget_components.cv2.VideoCapture", lambda *_args: capture
    )
    monkeypatch.setattr(
        "app.ui.widgets.widget_components.misc_helpers.read_frame",
        lambda *_args, **_kwargs: (True, "frame0"),
    )
    monkeypatch.setattr(
        "app.ui.widgets.widget_components.common_widget_actions.get_pixmap_from_frame",
        lambda *_args, **_kwargs: "pixmap",
    )
    monkeypatch.setattr(
        "app.ui.widgets.widget_components.graphics_view_actions.update_graphics_view",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "app.ui.widgets.widget_components.common_widget_actions.set_widgets_values_using_face_id_parameters",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "app.ui.widgets.widget_components.common_widget_actions.refresh_frame",
        lambda *_args, **kwargs: refresh_calls.append(kwargs),
    )

    widget_components.TargetMediaCardButton.load_media(card)

    assert toggled == [True]
    assert clear_calls == [True]
    assert refresh_calls == [{"synchronous": True}]
    assert main_window.video_processor.current_frame == "frame0"
    assert main_window.video_processor.media_capture is capture
    assert main_window.selected_video_button is card


def test_scan_guard_restores_input_face_card_checked_state(monkeypatch):
    main_window = _make_scan_main_window(keep_controls=True)
    main_window.scan_issue_worker = object()
    main_window.cur_selected_target_face_button = None
    toast_calls = []

    monkeypatch.setattr(
        "app.ui.widgets.actions.video_control_actions.common_widget_actions.create_and_show_toast_message",
        lambda *_args, **_kwargs: toast_calls.append((_args, _kwargs)),
    )

    checked_card = _make_guarded_card(main_window, checked=False)
    widget_components.InputFaceCardButton.load_input_face(checked_card)

    unchecked_card = _make_guarded_card(main_window, checked=True)
    widget_components.InputFaceCardButton.load_input_face(unchecked_card)

    assert checked_card.checked is True
    assert unchecked_card.checked is False
    assert checked_card.block_calls == [True, False]
    assert unchecked_card.block_calls == [True, False]
    assert len(toast_calls) == 2


def test_scan_guard_restores_embedding_card_checked_state(monkeypatch):
    main_window = _make_scan_main_window(keep_controls=True)
    main_window.scan_issue_worker = object()
    main_window.cur_selected_target_face_button = None
    toast_calls = []

    monkeypatch.setattr(
        "app.ui.widgets.actions.video_control_actions.common_widget_actions.create_and_show_toast_message",
        lambda *_args, **_kwargs: toast_calls.append((_args, _kwargs)),
    )

    checked_card = _make_guarded_card(main_window, checked=False)
    widget_components.EmbeddingCardButton.load_embedding(checked_card)

    unchecked_card = _make_guarded_card(main_window, checked=True)
    widget_components.EmbeddingCardButton.load_embedding(unchecked_card)

    assert checked_card.checked is True
    assert unchecked_card.checked is False
    assert checked_card.block_calls == [True, False]
    assert unchecked_card.block_calls == [True, False]
    assert len(toast_calls) == 2


def test_target_face_parameter_mutations_are_blocked_while_scan_is_active(
    monkeypatch,
):
    main_window = _make_scan_main_window(keep_controls=True)
    main_window.scan_issue_worker = object()
    main_window.parameters = {"face_1": {"mode": "original"}}
    main_window.copied_parameters = {"mode": "copied"}
    main_window.control["ExistingSetting"] = "before"
    main_window.selected_target_face_id = "face_1"
    toast_calls = []
    widget_value_updates = []
    control_value_updates = []
    refresh_calls = []

    monkeypatch.setattr(
        "app.ui.widgets.actions.video_control_actions.common_widget_actions.create_and_show_toast_message",
        lambda *_args, **_kwargs: toast_calls.append((_args, _kwargs)),
    )
    monkeypatch.setattr(
        "app.ui.widgets.actions.common_actions.set_widgets_values_using_face_id_parameters",
        lambda *_args, **_kwargs: widget_value_updates.append("updated"),
    )
    monkeypatch.setattr(
        "app.ui.widgets.actions.save_load_actions.common_widget_actions.set_control_widgets_values",
        lambda *_args, **_kwargs: control_value_updates.append("updated"),
    )
    monkeypatch.setattr(
        "app.ui.widgets.actions.save_load_actions.common_widget_actions.refresh_frame",
        lambda *_args, **_kwargs: refresh_calls.append("refreshed"),
    )

    assert common_actions.paste_selected_face_parameters(main_window, "face_1") is False
    assert save_load_actions.load_parameters_and_settings(main_window, "face_1") is None
    assert (
        save_load_actions.load_parameters_and_settings(main_window, "face_1", True)
        is None
    )

    assert main_window.parameters == {"face_1": {"mode": "original"}}
    assert main_window.control["ExistingSetting"] == "before"
    assert widget_value_updates == []
    assert control_value_updates == []
    assert refresh_calls == []
    assert len(toast_calls) == 3


def test_target_face_context_menu_disables_mutating_actions_while_scan_is_active():
    menu_exec_calls = []
    menu_state_snapshots = []
    target_face_button = SimpleNamespace(
        main_window=_make_scan_main_window(keep_controls=True),
        get_display_label=lambda: "Face 1",
        mapToGlobal=lambda point: point,
        popMenu=None,
    )

    def create_context_menu():
        target_face_button.face_header_action = _DummyButton("Face 1")
        target_face_button.parameters_copy_action = _DummyButton("Copy Parameters")
        target_face_button.parameters_paste_action = _DummyButton(
            "Apply Copied Parameters"
        )
        target_face_button.save_parameters_action = _DummyButton(
            "Save Current Parameters and Settings"
        )
        target_face_button.load_parameters_action = _DummyButton("Load Parameters")
        target_face_button.load_parameters_and_settings_action = _DummyButton(
            "Load Parameters and Settings"
        )
        target_face_button.small_thumbnails_action = _DummyButton("Small Thumbnails")
        target_face_button.large_thumbnails_action = _DummyButton("Large Thumbnails")
        target_face_button.remove_action = _DummyButton("Remove from List")

    def exec_context_menu(point):
        menu_exec_calls.append(point)
        menu_state_snapshots.append(
            {
                "parameters_copy": target_face_button.parameters_copy_action.enabled,
                "parameters_paste": target_face_button.parameters_paste_action.enabled,
                "save_parameters": target_face_button.save_parameters_action.enabled,
                "load_parameters": target_face_button.load_parameters_action.enabled,
                "load_parameters_and_settings": target_face_button.load_parameters_and_settings_action.enabled,
                "remove": target_face_button.remove_action.enabled,
            }
        )

    def release_context_menu(*action_attrs):
        for attr in action_attrs:
            setattr(target_face_button, attr, None)
        target_face_button.popMenu = None

    target_face_button.create_context_menu = create_context_menu
    target_face_button._exec_context_menu = exec_context_menu
    target_face_button._release_context_menu = release_context_menu
    target_face_button.main_window.scan_issue_worker = object()

    widget_components.TargetFaceCardButton.on_context_menu(target_face_button, 12)

    assert menu_state_snapshots[-1] == {
        "parameters_copy": True,
        "parameters_paste": False,
        "save_parameters": True,
        "load_parameters": False,
        "load_parameters_and_settings": False,
        "remove": False,
    }
    assert menu_exec_calls == [12]
    assert target_face_button.remove_action is None

    target_face_button.main_window.scan_issue_worker = None
    target_face_button.main_window.scan_issue_ui_state = {}

    widget_components.TargetFaceCardButton.on_context_menu(target_face_button, 34)

    assert menu_state_snapshots[-1] == {
        "parameters_copy": True,
        "parameters_paste": True,
        "save_parameters": True,
        "load_parameters": True,
        "load_parameters_and_settings": True,
        "remove": True,
    }
    assert menu_exec_calls == [12, 34]
    assert target_face_button.remove_action is None


def test_target_face_refresh_display_label_does_not_require_context_menu():
    display_label = _DummyButton("Face")
    target_face_button = SimpleNamespace(
        display_label=display_label,
        get_display_label=lambda: "Face 2",
    )

    widget_components.TargetFaceCardButton.refresh_display_label(target_face_button)

    assert display_label.text == "Face 2"


def test_target_face_refresh_display_label_updates_live_context_menu_header():
    display_label = _DummyButton("Face")
    face_header_action = _DummyButton("Face")
    target_face_button = SimpleNamespace(
        display_label=display_label,
        face_header_action=face_header_action,
        get_display_label=lambda: "Face 2",
    )

    widget_components.TargetFaceCardButton.refresh_display_label(target_face_button)

    assert display_label.text == "Face 2"
    assert face_header_action.text == "Face 2"
