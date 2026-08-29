from types import SimpleNamespace

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
        control={},
        target_videos={},
        target_faces={"face_1": object()},
        selected_video_button=object(),
        targetVideosPathLineEdit=path_line_edit,
        last_target_media_folder_path="E:/media",
        targetVideosList=_DummyListWidget(),
        placeholder_update_signal=placeholder_signal,
        video_loader_worker=None,
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

    assert list_view_actions.clear_all_target_media(main_window) is True
    assert main_window.target_videos == {}
    assert main_window.selected_video_button is None
    assert main_window.targetVideosPathLineEdit.value == ""
    assert main_window.targetVideosPathLineEdit.tooltip == ""
    assert main_window.last_target_media_folder_path == ""
    assert placeholder_signal.calls == [(main_window.targetVideosList, False)]
    assert button_a.removed == 1
    assert button_b.removed == 1
    assert clear_target_faces_calls == [(main_window, False)]


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
