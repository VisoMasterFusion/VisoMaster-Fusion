import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import inspect
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from PySide6 import QtWidgets

from app.ui import main_ui
from app.ui.widgets.actions import (
    card_actions,
    control_actions,
)
from app.ui.widgets.actions import (
    list_view_actions as lva,
)


@pytest.fixture
def window(monkeypatch):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    monkeypatch.setattr(lva.ui_workers, "FilterWorker", lambda **_: None)
    monkeypatch.setattr(lva.common_widget_actions, "refresh_frame", lambda **_: None)
    w = SimpleNamespace(
        embeddingTabs=QtWidgets.QTabWidget(),
        embedding_tab_states=[],
        merged_embeddings={},
        loaded_embedding_filename="",
        target_faces={},
        control={},
        video_processor=None,
    )
    w.embeddingTabs.currentChanged.connect(
        lambda i: lva._on_embedding_tab_changed(w, i)
    )
    w.embeddingTabs.tabBar().tabMoved.connect(
        lambda *_: lva._activate_embedding_tab(w, w.embeddingTabs.currentIndex())
    )
    lva.add_embedding_tab(w, list_widget=QtWidgets.QListWidget(), title="A")
    yield w
    w.embeddingTabs.deleteLater()
    app.processEvents()


def add_button(w, eid):
    button = QtWidgets.QPushButton(eid)
    button.embedding_id = eid
    button.embedding_name = eid
    button.embedding_store = {"ArcFace": np.array([1.0, 2.0])}
    item = QtWidgets.QListWidgetItem(w.inputEmbeddingsList)
    button.list_item = item
    w.inputEmbeddingsList.setItemWidget(item, button)
    w.merged_embeddings[eid] = button
    return button


def test_switch_drag_close_preserves_assignments(window):
    w = window
    a = add_button(w, "a")
    face = SimpleNamespace(
        assigned_merged_embeddings={"a": a.embedding_store},
        calculate_assigned_input_embedding=MagicMock(),
    )
    w.target_faces = {"face": face}
    lva.add_embedding_tab(w, list_widget=QtWidgets.QListWidget(), title="B")
    add_button(w, "b")
    assert "a" in face.assigned_merged_embeddings
    w.embeddingTabs.tabBar().moveTab(1, 0)
    assert w.inputEmbeddingsList is w.embeddingTabs.currentWidget()
    assert list(w.merged_embeddings) == ["b"]
    assert lva.get_embedding_tabs_state(w)["tabs"][0]["embedding_ids"] == ["b"]
    lva._on_embedding_tab_close(w, 0)
    assert "a" in face.assigned_merged_embeddings
    assert list(w.merged_embeddings) == ["a"]


def test_restore_all_tabs_without_external_files(window):
    w = window
    a = add_button(w, "a")
    b = add_button(w, "b")
    snapshot = {
        "tabs": [
            {"title": "A", "embedding_ids": ["a"]},
            {"title": "B", "filename": "missing.json", "embedding_ids": ["b"]},
        ],
        "active_index": 1,
    }
    lva.restore_embedding_tabs_state(w, snapshot)
    assert lva.get_all_merged_embeddings(w) == {"a": a, "b": b}
    assert w.embedding_tab_states[0]["embeddings"] == {"a": a}
    assert w.embedding_tab_states[1]["embeddings"] == {"b": b}
    assert w.merged_embeddings == {"b": b}
    assert np.array_equal(b.embedding_store["ArcFace"], [1, 2])
    assert w.loaded_embedding_filename == "missing.json"


def test_close_assigned_tab_removes_only_deleted_ids(window):
    w = window
    a = add_button(w, "a")
    lva.add_embedding_tab(w, list_widget=QtWidgets.QListWidget(), title="B")
    b = add_button(w, "b")
    face = SimpleNamespace(
        assigned_merged_embeddings={"a": a.embedding_store, "b": b.embedding_store},
        calculate_assigned_input_embedding=MagicMock(),
    )
    w.target_faces = {"face": face}
    lva._on_embedding_tab_close(w, 1)
    assert list(face.assigned_merged_embeddings) == ["a"]


def test_legacy_filename_only_preserves_inline(window):
    w = window
    a = add_button(w, "saved-id")
    lva.restore_embedding_tabs_state(
        w, {"tabs": [{"filename": "missing.json", "title": "Saved"}]}
    )
    assert w.merged_embeddings == {"saved-id": a}


def test_clear_active_only(window):
    w = window
    a = add_button(w, "a")
    lva.add_embedding_tab(w, list_widget=QtWidgets.QListWidget(), title="B")
    b = add_button(w, "b")
    face = SimpleNamespace(
        assigned_merged_embeddings={"a": a.embedding_store, "b": b.embedding_store},
        calculate_assigned_input_embedding=MagicMock(),
    )
    w.target_faces = {"face": face}
    card_actions.clear_merged_embeddings(w)
    assert list(face.assigned_merged_embeddings) == ["a"]
    assert "a" in lva.get_all_merged_embeddings(w)


def test_real_callbacks():
    assert inspect.isfunction(control_actions.change_theme)


def test_faces_panel_cycles(window):
    app = QtWidgets.QApplication.instance()
    w = QtWidgets.QWidget()
    w.gridLayout_2 = QtWidgets.QGridLayout(w)
    w.facesButtonsWidget = QtWidgets.QWidget(w)
    w.controlButtonsLayout = QtWidgets.QVBoxLayout(w.facesButtonsWidget)
    w.targetFacesList = QtWidgets.QListWidget(w)
    for name in (
        "findTargetFacesButton",
        "clearTargetFacesButton",
        "swapfacesButton",
        "editFacesButton",
        "saveImageButton",
    ):
        setattr(w, name, QtWidgets.QPushButton(name, w))
    w.verticalWidget = QtWidgets.QWidget(w)
    w.facesPanelGroupBox = QtWidgets.QGroupBox(w)
    w._configure_faces_panel_button_column = lambda: (
        main_ui.MainWindow._configure_faces_panel_button_column(w)
    )
    w._configure_faces_panel_button_column()
    top = w.controlButtonsLayout.contentsMargins().top()
    holder = QtWidgets.QWidget()
    right = QtWidgets.QVBoxLayout(holder)
    for _ in range(3):
        right.addWidget(w.targetFacesList)
        main_ui.MainWindow._restore_faces_strip_to_panel(w)
        app.processEvents()
        assert w._faces_list_offset_container.layout().indexOf(w.targetFacesList) >= 0
        assert w.controlButtonsLayout.contentsMargins().top() == top
