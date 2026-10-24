from types import SimpleNamespace
from unittest.mock import MagicMock

from PySide6 import QtWidgets

from app.ui.widgets.actions import list_view_actions


class _DummySignal:
    def __init__(self):
        self.calls = []

    def emit(self, *args):
        self.calls.append(args)


class _DummyLineEdit:
    def __init__(self, text=""):
        self.value = text
        self.tooltip = text

    def setText(self, text):
        self.value = str(text)

    def setToolTip(self, tooltip):
        self.tooltip = str(tooltip)


class _DummyListWidget:
    def __init__(self):
        self.cleared = 0

    def clear(self):
        self.cleared += 1


class _DummyTargetFace:
    def __init__(self, input_faces=None, merged_embeddings=None):
        self.assigned_input_faces = dict(input_faces or {})
        self.assigned_merged_embeddings = dict(merged_embeddings or {})
        self.recalculated = 0

    def calculate_assigned_input_embedding(self):
        self.recalculated += 1


class _DummyTargetMediaButton:
    def __init__(self, main_window, media_id):
        self.main_window = main_window
        self.media_id = media_id
        self.media_path = f"E:/media/{media_id}.mp4"
        self.removed = 0

    def remove_target_media_from_list(self):
        self.removed += 1
        self.main_window.target_videos.pop(self.media_id, None)


class _DummyInputFaceButton:
    def __init__(self, main_window, face_id):
        self.main_window = main_window
        self.face_id = face_id
        self.kv_removed = 0
        self.deleted = 0

    def remove_kv_data_file(self):
        self.kv_removed += 1

    def _remove_face_from_lists(self):
        self.main_window.input_faces.pop(self.face_id, None)
        for target_face in self.main_window.target_faces.values():
            target_face.assigned_input_faces.pop(self.face_id, None)
            target_face.calculate_assigned_input_embedding()

    def deleteLater(self):
        self.deleted += 1


class _DummyEmbedButton:
    def __init__(self):
        self.deleted = 0

    def deleteLater(self):
        self.deleted += 1


def test_clear_all_target_media_cancel_leaves_state_unchanged(monkeypatch):
    placeholder_signal = _DummySignal()
    path_line_edit = _DummyLineEdit("E:/media")
    main_window = SimpleNamespace(
        control={},
        target_videos={},
        target_faces={"face_1": object()},
        selected_video_button=object(),
        targetVideosPathLineEdit=path_line_edit,
        last_target_media_folder_path="E:/media",
        targetVideosList=_DummyListWidget(),
        placeholder_update_signal=placeholder_signal,
        video_loader_worker=None,
        video_processor=MagicMock(),
        scene=MagicMock(),
        graphicsViewFrame=MagicMock(),
        videoSeekSlider=MagicMock(),
    )
    button = _DummyTargetMediaButton(main_window, "media_1")
    main_window.target_videos["media_1"] = button

    monkeypatch.setattr(
        "app.ui.widgets.actions.video_control_actions.block_if_issue_scan_active",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "question",
        lambda *_args, **_kwargs: QtWidgets.QMessageBox.No,
    )

    assert list_view_actions.clear_all_target_media(main_window) is False
    assert list(main_window.target_videos) == ["media_1"]
    assert main_window.selected_video_button is not None
    assert main_window.targetVideosPathLineEdit.value == "E:/media"
    assert main_window.last_target_media_folder_path == "E:/media"
    assert placeholder_signal.calls == []
    assert button.removed == 0


def test_clear_all_target_media_confirm_clears_state(monkeypatch):
    placeholder_signal = _DummySignal()
    path_line_edit = _DummyLineEdit("E:/media")
    clear_target_faces_calls = []
    main_window = SimpleNamespace(
        control={"AutoLoadTargetFolderToggle": True},
        target_videos={},
        target_faces={"face_1": object()},
        selected_video_button=object(),
        targetVideosPathLineEdit=path_line_edit,
        last_target_media_folder_path="E:/media",
        targetVideosList=_DummyListWidget(),
        placeholder_update_signal=placeholder_signal,
        video_loader_worker=None,
        video_processor=MagicMock(),
        scene=MagicMock(),
        graphicsViewFrame=MagicMock(),
        videoSeekSlider=MagicMock(),
    )
    button_a = _DummyTargetMediaButton(main_window, "media_1")
    button_b = _DummyTargetMediaButton(main_window, "media_2")
    main_window.target_videos = {
        "media_1": button_a,
        "media_2": button_b,
    }

    monkeypatch.setattr(
        "app.ui.widgets.actions.video_control_actions.block_if_issue_scan_active",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "question",
        lambda *_args, **_kwargs: QtWidgets.QMessageBox.Yes,
    )
    monkeypatch.setattr(
        "app.ui.widgets.actions.list_view_actions.card_actions.clear_target_faces",
        lambda mw, refresh_frame=False: clear_target_faces_calls.append(
            (mw, refresh_frame)
        ),
    )

    watch_calls = []
    monkeypatch.setattr(
        list_view_actions, "set_target_folder_auto_watch",
        lambda mw, enabled: watch_calls.append((mw, enabled)),
    )

    assert list_view_actions.clear_all_target_media(main_window) is True
    assert main_window.target_videos == {}
    assert main_window.selected_video_button is None
    assert main_window.targetVideosPathLineEdit.value == "E:/media"
    assert main_window.targetVideosPathLineEdit.tooltip == "E:/media"
    assert main_window.last_target_media_folder_path == "E:/media"
    main_window.video_processor.stop_processing.assert_called_once_with()
    assert main_window.video_processor.media_capture is None
    assert main_window.video_processor.media_path is None
    assert main_window.video_processor.current_frame is None
    main_window.scene.clear.assert_called_once_with()
    main_window.graphicsViewFrame.update.assert_called_once_with()
    main_window.videoSeekSlider.setMaximum.assert_called_once_with(1)
    main_window.videoSeekSlider.setValue.assert_called_once_with(0)
    assert placeholder_signal.calls == [(main_window.targetVideosList, False)]
    assert button_a.removed == 1
    assert button_b.removed == 1
    assert clear_target_faces_calls == [(main_window, False)]

    assert watch_calls == [(main_window, True)]
    expected_paths = {
        list_view_actions._normalize_media_path(button.media_path)
        for button in (button_a, button_b)
    }
    assert main_window._target_folder_seen_paths == expected_paths
    assert main_window._target_folder_ignored_paths == expected_paths


def test_clear_all_target_media_skips_confirmation_when_toggle_is_set(monkeypatch):
    main_window = SimpleNamespace(
        control={"SkipClearConfirmationToggle": True},
        target_videos={},
        target_faces={},
        selected_video_button=object(),
        targetVideosPathLineEdit=_DummyLineEdit("E:/media"),
        last_target_media_folder_path="E:/media",
        targetVideosList=_DummyListWidget(),
        placeholder_update_signal=_DummySignal(),
        video_loader_worker=None,
        video_processor=MagicMock(),
        scene=MagicMock(),
        graphicsViewFrame=MagicMock(),
        videoSeekSlider=MagicMock(),
    )
    main_window.target_videos["media_1"] = _DummyTargetMediaButton(
        main_window, "media_1"
    )

    monkeypatch.setattr(
        "app.ui.widgets.actions.video_control_actions.block_if_issue_scan_active",
        lambda *_args, **_kwargs: False,
    )

    def fail_question(*_args, **_kwargs):
        raise AssertionError("confirmation dialog must be skipped")

    monkeypatch.setattr(QtWidgets.QMessageBox, "question", fail_question)

    assert list_view_actions.clear_all_target_media(main_window) is True
    assert main_window.target_videos == {}


def test_clear_all_target_media_blocked_leaves_state_unchanged(monkeypatch):
    main_window = SimpleNamespace(
        target_videos={"media_1": object()},
        target_faces={},
        selected_video_button=object(),
        targetVideosPathLineEdit=_DummyLineEdit("E:/media"),
        last_target_media_folder_path="E:/media",
        targetVideosList=_DummyListWidget(),
        placeholder_update_signal=_DummySignal(),
        video_loader_worker=None,
        video_processor=MagicMock(),
        scene=MagicMock(),
        graphicsViewFrame=MagicMock(),
        videoSeekSlider=MagicMock(),
    )

    monkeypatch.setattr(
        "app.ui.widgets.actions.video_control_actions.block_if_issue_scan_active",
        lambda *_args, **_kwargs: True,
    )

    assert list_view_actions.clear_all_target_media(main_window) is False
    assert list(main_window.target_videos) == ["media_1"]
    assert main_window.selected_video_button is not None
    assert main_window.targetVideosPathLineEdit.value == "E:/media"
    assert main_window.last_target_media_folder_path == "E:/media"


def test_clear_all_input_faces_cancel_leaves_state_unchanged(monkeypatch):
    assigned_face = object()
    target_face = _DummyTargetFace({"face_1": assigned_face})
    path_line_edit = _DummyLineEdit("E:/faces")
    main_window = SimpleNamespace(
        control={},
        input_faces={},
        target_faces={"target_1": target_face},
        inputFacesPathLineEdit=path_line_edit,
        last_input_media_folder_path="E:/faces",
        inputFacesList=_DummyListWidget(),
        placeholder_update_signal=_DummySignal(),
        input_faces_loader_worker=None,
    )
    face_button = _DummyInputFaceButton(main_window, "face_1")
    main_window.input_faces["face_1"] = face_button

    monkeypatch.setattr(
        "app.ui.widgets.actions.video_control_actions.block_if_issue_scan_active",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "question",
        lambda *_args, **_kwargs: QtWidgets.QMessageBox.No,
    )
    refresh_calls = []
    monkeypatch.setattr(
        "app.ui.widgets.actions.list_view_actions.common_widget_actions.refresh_frame",
        lambda mw: refresh_calls.append(mw),
    )

    assert list_view_actions.clear_all_input_faces(main_window) is False
    assert list(main_window.input_faces) == ["face_1"]
    assert target_face.assigned_input_faces == {"face_1": assigned_face}
    assert main_window.inputFacesPathLineEdit.value == "E:/faces"
    assert main_window.last_input_media_folder_path == "E:/faces"
    assert refresh_calls == []


def test_clear_all_input_faces_confirm_clears_state(monkeypatch):
    placeholder_signal = _DummySignal()
    path_line_edit = _DummyLineEdit("E:/faces")
    target_face = _DummyTargetFace({"face_1": object(), "face_2": object()})
    main_window = SimpleNamespace(
        control={},
        input_faces={},
        target_faces={"target_1": target_face},
        inputFacesPathLineEdit=path_line_edit,
        last_input_media_folder_path="E:/faces",
        inputFacesList=_DummyListWidget(),
        placeholder_update_signal=placeholder_signal,
        input_faces_loader_worker=None,
    )
    face_one = _DummyInputFaceButton(main_window, "face_1")
    face_two = _DummyInputFaceButton(main_window, "face_2")
    main_window.input_faces = {"face_1": face_one, "face_2": face_two}

    monkeypatch.setattr(
        "app.ui.widgets.actions.video_control_actions.block_if_issue_scan_active",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "question",
        lambda *_args, **_kwargs: QtWidgets.QMessageBox.Yes,
    )
    refresh_calls = []
    monkeypatch.setattr(
        "app.ui.widgets.actions.list_view_actions.common_widget_actions.refresh_frame",
        lambda mw: refresh_calls.append(mw),
    )

    assert list_view_actions.clear_all_input_faces(main_window) is True
    assert main_window.input_faces == {}
    assert target_face.assigned_input_faces == {}
    assert target_face.recalculated == 2
    assert main_window.inputFacesPathLineEdit.value == ""
    assert main_window.inputFacesPathLineEdit.tooltip == ""
    assert main_window.last_input_media_folder_path == ""
    assert placeholder_signal.calls == [(main_window.inputFacesList, False)]
    assert refresh_calls == [main_window]
    assert face_one.kv_removed == 1
    assert face_two.kv_removed == 1
    assert face_one.deleted == 1
    assert face_two.deleted == 1


def test_clear_all_input_faces_blocked_leaves_state_unchanged(monkeypatch):
    main_window = SimpleNamespace(
        input_faces={"face_1": object()},
        target_faces={},
        inputFacesPathLineEdit=_DummyLineEdit("E:/faces"),
        last_input_media_folder_path="E:/faces",
        inputFacesList=_DummyListWidget(),
        placeholder_update_signal=_DummySignal(),
        input_faces_loader_worker=None,
    )

    monkeypatch.setattr(
        "app.ui.widgets.actions.video_control_actions.block_if_issue_scan_active",
        lambda *_args, **_kwargs: True,
    )

    assert list_view_actions.clear_all_input_faces(main_window) is False
    assert list(main_window.input_faces) == ["face_1"]
    assert main_window.inputFacesPathLineEdit.value == "E:/faces"
    assert main_window.last_input_media_folder_path == "E:/faces"


def test_clear_all_embeddings_cancel_leaves_state_unchanged(monkeypatch):
    assigned_embedding = object()
    target_face = _DummyTargetFace(merged_embeddings={"embed_1": assigned_embedding})
    embed_button = _DummyEmbedButton()
    main_window = SimpleNamespace(
        control={},
        merged_embeddings={"embed_1": embed_button},
        target_faces={"target_1": target_face},
        inputEmbeddingsList=_DummyListWidget(),
    )

    monkeypatch.setattr(
        "app.ui.widgets.actions.video_control_actions.block_if_issue_scan_active",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "question",
        lambda *_args, **_kwargs: QtWidgets.QMessageBox.No,
    )
    refresh_calls = []
    monkeypatch.setattr(
        "app.ui.widgets.actions.card_actions.common_widget_actions.refresh_frame",
        lambda *args, **kwargs: refresh_calls.append(
            kwargs.get("main_window", args[0] if args else None)
        ),
    )

    assert list_view_actions.clear_all_embeddings(main_window) is False
    assert list(main_window.merged_embeddings) == ["embed_1"]
    assert target_face.assigned_merged_embeddings == {"embed_1": assigned_embedding}
    assert refresh_calls == []


def test_clear_all_embeddings_confirm_clears_state(monkeypatch):
    target_face = _DummyTargetFace(merged_embeddings={"embed_1": object()})
    embed_button = _DummyEmbedButton()
    list_widget = _DummyListWidget()
    main_window = SimpleNamespace(
        control={},
        merged_embeddings={"embed_1": embed_button},
        target_faces={"target_1": target_face},
        inputEmbeddingsList=list_widget,
    )

    monkeypatch.setattr(
        "app.ui.widgets.actions.video_control_actions.block_if_issue_scan_active",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "question",
        lambda *_args, **_kwargs: QtWidgets.QMessageBox.Yes,
    )
    refresh_calls = []
    monkeypatch.setattr(
        "app.ui.widgets.actions.card_actions.common_widget_actions.refresh_frame",
        lambda *args, **kwargs: refresh_calls.append(
            kwargs.get("main_window", args[0] if args else None)
        ),
    )

    assert list_view_actions.clear_all_embeddings(main_window) is True
    assert main_window.merged_embeddings == {}
    assert target_face.assigned_merged_embeddings == {}
    assert target_face.recalculated == 1
    assert list_widget.cleared == 1
    assert embed_button.deleted == 1
    assert refresh_calls == [main_window]


def test_clear_all_embeddings_blocked_leaves_state_unchanged(monkeypatch):
    main_window = SimpleNamespace(
        merged_embeddings={"embed_1": object()},
        target_faces={},
        inputEmbeddingsList=_DummyListWidget(),
    )

    monkeypatch.setattr(
        "app.ui.widgets.actions.video_control_actions.block_if_issue_scan_active",
        lambda *_args, **_kwargs: True,
    )

    assert list_view_actions.clear_all_embeddings(main_window) is False
    assert list(main_window.merged_embeddings) == ["embed_1"]



class _FullDummyListWidget:
    def __init__(self, buttons=None):
        self._items = list(buttons or [])

    def setUpdatesEnabled(self, _enabled):
        pass

    def count(self):
        return len(self._items)

    def item(self, index):
        return self._items[index]

    def itemWidget(self, item):
        return item

    def takeItem(self, index):
        return self._items.pop(index)

    def clear(self):
        self._items.clear()


def test_set_target_folder_auto_watch_false_when_inactive_preserves_loaded_media(tmp_path):
    media_file = tmp_path / "clip.mp4"
    media_file.write_bytes(b"00")
    btn = SimpleNamespace(
        media_id="m1",
        media_path=str(media_file),
        is_webcam=False,
        deleteLater=MagicMock(),
    )
    list_widget = _FullDummyListWidget([btn])
    worker = MagicMock()
    main_window = SimpleNamespace(
        control={"AutoLoadTargetFolderToggle": False},
        target_videos={"m1": btn},
        target_faces={"f1": object()},
        parameters={"f1": {"SwapModelSelection": "Inswapper128"}},
        selected_video_button=btn,
        selected_target_face_id="f1",
        targetVideosPathLineEdit=SimpleNamespace(text=lambda: str(tmp_path)),
        last_target_media_folder_path=str(tmp_path),
        targetVideosList=list_widget,
        placeholder_update_signal=_DummySignal(),
        video_loader_worker=worker,
        video_processor=MagicMock(),
        scene=MagicMock(),
    )

    list_view_actions.set_target_folder_auto_watch(main_window, False)

    assert main_window.video_loader_worker is worker
    assert list_widget.count() == 1
    assert "m1" in main_window.target_videos
    assert main_window.selected_video_button is btn
    assert main_window.parameters == {"f1": {"SwapModelSelection": "Inswapper128"}}


def test_set_target_folder_auto_watch_false_after_active_clears_media(tmp_path, monkeypatch):
    media_file = tmp_path / "clip.mp4"
    media_file.write_bytes(b"00")
    btn = SimpleNamespace(
        media_id="m1",
        media_path=str(media_file),
        is_webcam=False,
        deleteLater=MagicMock(),
    )
    list_widget = _FullDummyListWidget([btn])
    main_window = SimpleNamespace(
        control={"AutoLoadTargetFolderToggle": True, "AutoLoadTargetFolderRecursiveToggle": False},
        target_videos={"m1": btn},
        target_faces={},
        parameters={"f1": {"SwapModelSelection": "Inswapper128"}},
        selected_video_button=btn,
        selected_target_face_id="f1",
        targetVideosPathLineEdit=SimpleNamespace(text=lambda: str(tmp_path)),
        last_target_media_folder_path=str(tmp_path),
        targetVideosList=list_widget,
        placeholder_update_signal=_DummySignal(),
        video_loader_worker=None,
        video_processor=MagicMock(),
        scene=MagicMock(),
    )
    monkeypatch.setattr(
        list_view_actions,
        "scan_and_append_new_target_media",
        lambda _mw: None,
    )

    list_view_actions.set_target_folder_auto_watch(main_window, True)
    assert main_window._target_folder_watch_active is True

    list_view_actions.set_target_folder_auto_watch(main_window, False)
    assert main_window._target_folder_watch_active is False
    assert list_widget.count() == 0
    assert main_window.target_videos == {}
    assert main_window.selected_video_button is None


def test_prune_missing_and_remove_subfolder_target_media(tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    root_file = tmp_path / "root.mp4"
    sub_file = sub / "nested.mp4"
    root_file.write_bytes(b"00")
    sub_file.write_bytes(b"00")

    btn_root = SimpleNamespace(
        media_id="root",
        media_path=str(root_file),
        is_webcam=False,
        deleteLater=MagicMock(),
    )
    btn_sub = SimpleNamespace(
        media_id="sub",
        media_path=str(sub_file),
        is_webcam=False,
        deleteLater=MagicMock(),
    )
    list_widget = _FullDummyListWidget([btn_root, btn_sub])
    capture = MagicMock()
    main_window = SimpleNamespace(
        target_videos={"root": btn_root, "sub": btn_sub},
        selected_video_button=btn_sub,
        selected_target_face_id="f1",
        cur_selected_target_face_button=object(),
        parameters={"f1": {}},
        targetVideosList=list_widget,
        placeholder_update_signal=_DummySignal(),
        video_processor=SimpleNamespace(stop_processing=MagicMock(), media_capture=capture),
        scene=MagicMock(),
    )

    list_view_actions._remove_subfolder_target_media(main_window, str(tmp_path))
    assert list_widget.count() == 1
    assert "sub" not in main_window.target_videos
    assert "root" in main_window.target_videos
    assert main_window.selected_video_button is None
    assert main_window.cur_selected_target_face_button is None
    capture.release.assert_called_once_with()

    # Now prune when root_file is no longer in on_disk
    main_window.selected_video_button = btn_root
    list_view_actions._prune_missing_auto_loaded_target_media(main_window, str(tmp_path), set())
    assert list_widget.count() == 0
    assert main_window.target_videos == {}
    assert main_window.selected_video_button is None
