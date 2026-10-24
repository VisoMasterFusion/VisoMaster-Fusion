"""Run real application Qt workers in an isolated process without model inference."""

import os
import sys
import tempfile
import threading
import traceback
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
from PySide6 import QtCore, QtWidgets
from shiboken6 import isValid

from app.ui import main_ui
from app.ui.widgets import ui_workers
from app.ui.widgets.actions import (
    filter_actions,
    save_load_actions,
)
from app.ui.widgets.actions import (
    list_view_actions as lva,
)


def pump():
    app = QtWidgets.QApplication.instance()
    for _ in range(3):
        app.processEvents()
    QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)


def window():
    w = SimpleNamespace(
        embeddingTabs=QtWidgets.QTabWidget(),
        embedding_tab_states=[],
        inputEmbeddingsSearchBox=QtWidgets.QLineEdit(),
        merged_embeddings={},
        target_faces={},
        control={},
        video_processor=SimpleNamespace(processing=True, ui_state_is_dirty=False),
        targetVideosList=QtWidgets.QListWidget(),
        inputFacesList=QtWidgets.QListWidget(),
    )
    w.embeddingTabs.currentChanged.connect(
        lambda i: lva._on_embedding_tab_changed(w, i)
    )
    w.embeddingTabs.tabBar().tabMoved.connect(
        lambda *_: lva._activate_embedding_tab(w, w.embeddingTabs.currentIndex())
    )
    lva.add_embedding_tab(w, list_widget=QtWidgets.QListWidget(), title="A")
    return w


def add_card(w, name):
    lva.create_and_add_embed_button_to_list(
        w, name, {"ArcFace": np.array([1.0, 2.0])}, name
    )
    return w.merged_embeddings[name]


def active_worker(w):
    """Hold a real filter in flight until retire() waits for its completion."""
    lva.stop_embedding_filter_worker(w)
    entered = threading.Event()
    release = threading.Event()

    class HeldFilter(ui_workers.FilterWorker):
        def run(self):
            entered.set()
            assert release.wait(5), "filter retirement never joined"
            super().run()

    worker = HeldFilter(w, filter_list="merged_embeddings")
    w.merged_embeddings_filter_worker = worker
    worker.start()
    assert entered.wait(5)
    threading.Timer(0.03, release.set).start()
    return worker


def active_worker_tabs():
    w = window()
    for _ in range(10):
        before = active_worker(w)
        lva.add_embedding_tab(w, list_widget=QtWidgets.QListWidget(), title="B")
        assert not before.isRunning()
        before = active_worker(w)
        lva._on_embedding_tab_close(w, 1)
        assert not before.isRunning()
        before = active_worker(w)
        lva.reset_embedding_tabs(w)
        assert not before.isRunning()
        pump()
    lva.stop_filter_workers(w)
    pump()


def stale_results():
    w = window()
    add_card(w, "a")
    add_card(w, "b")
    lva.stop_embedding_filter_worker(w)
    for dispatch_signal_first in (False, True):
        worker = ui_workers.FilterWorker(
            w, search_text="a", filter_list="merged_embeddings"
        )
        worker.items_snapshot = [(0, "a"), (1, "b")]
        w.merged_embeddings_filter_worker = worker
        worker.start()
        assert worker.wait(5000)
        if dispatch_signal_first:
            QtWidgets.QApplication.instance().processEvents()
        lva.stop_embedding_filter_worker(w)
        pump()
        assert not any(w.inputEmbeddingsList.item(i).isHidden() for i in range(2))
    # A queued result from an earlier run of a reused worker must also be ignored.
    worker = filter_actions.ensure_embedding_filter_worker(w)
    worker.search_text = "a"
    worker.items_snapshot = [(0, "a"), (1, "b")]
    worker.start()
    assert worker.wait(5000)
    worker.stop_thread()
    worker.search_text = "b"
    worker.start()
    assert worker.wait(5000)
    pump()
    assert w.inputEmbeddingsList.item(0).isHidden()
    assert not w.inputEmbeddingsList.item(1).isHidden()
    # A deferred update must not access a deleted native list widget.
    listing = QtWidgets.QListWidget()
    filter_actions.update_filtered_list(w, listing, [])
    listing.deleteLater()
    QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
    assert not isValid(listing)
    pump()
    lva.stop_filter_workers(w)


def shutdown():
    w = window()
    before = active_worker(w)
    workers = [before]
    for attribute, kind in (
        ("target_videos_filter_worker", "target_videos"),
        ("input_faces_filter_worker", "input_faces"),
    ):
        worker = ui_workers.FilterWorker(w, filter_list=kind)
        setattr(w, attribute, worker)
        worker.start()
        workers.append(worker)
    lva.stop_filter_workers(w)
    pump()
    assert all(not worker.isRunning() for worker in workers)
    assert w.merged_embeddings_filter_worker is None


def workspace_roundtrip(populated):
    # Full window and save/load entrypoints; only skip the interactive startup
    # workspace/provider dialog. No worker or event-processing mocks.
    main_ui.MainWindow.load_last_workspace = lambda _: None
    w = main_ui.MainWindow()
    with tempfile.TemporaryDirectory() as folder:
        w.project_root_path = Path(folder)
        w.last_workspace_path = Path(folder) / "workspace.json"
        if populated:
            add_card(w, "a")
            lva.add_embedding_tab(w, title="B")
            b = add_card(w, "b")
            b.kv_map = {"cpu": __import__("torch").ones(1)}
        for _ in range(5):
            save_load_actions.save_current_workspace(w, str(w.last_workspace_path))
            save_load_actions.load_saved_workspace(w, str(w.last_workspace_path))
            pump()
            if populated:
                restored = lva.get_all_merged_embeddings(w)
                assert set(restored) == {"a", "b"}
                assert all(isValid(button) for button in restored.values())
                assert np.array_equal(restored["b"].embedding_store["ArcFace"], [1, 2])
                assert restored["b"].kv_map["cpu"].item() == 1
                assert list(w.embedding_tab_states[0]["embeddings"]) == ["a"]
                assert list(w.embedding_tab_states[1]["embeddings"]) == ["b"]
            else:
                assert not lva.get_all_merged_embeddings(w)
        w.quit_without_saving = True
        w.close()
        pump()
        assert w.merged_embeddings_filter_worker is None


if __name__ == "__main__":
    callback_errors = []

    def capture_exception(kind, value, tb):
        callback_errors.append(value)
        traceback.print_exception(kind, value, tb)

    sys.excepthook = capture_exception
    app = QtWidgets.QApplication([])
    scenario = sys.argv[1]
    if scenario == "empty_workspace":
        workspace_roundtrip(False)
    elif scenario == "populated_workspace":
        workspace_roundtrip(True)
    else:
        globals()[scenario]()
    assert not callback_errors, callback_errors
    print(f"PASS {scenario}", flush=True)
