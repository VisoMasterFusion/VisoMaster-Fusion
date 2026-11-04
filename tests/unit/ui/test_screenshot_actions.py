"""Tests for session screenshot capture, comparison, and saving."""

from __future__ import annotations

import os
from datetime import datetime
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtCore, QtGui, QtWidgets

from app.ui.widgets.actions import graphics_view_actions, screenshot_actions


@pytest.fixture(scope="module")
def qapp():
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


def _pixmap(color: QtGui.QColor, size: int = 12) -> QtGui.QPixmap:
    pixmap = QtGui.QPixmap(size, size)
    pixmap.fill(color)
    return pixmap


def _make_window(qapp, tmp_path):
    del qapp
    host = QtWidgets.QWidget()
    view = QtWidgets.QGraphicsView(host)
    scene = QtWidgets.QGraphicsScene(host)
    view.setScene(scene)
    base_item = QtWidgets.QGraphicsPixmapItem(_pixmap(QtGui.QColor("red")))
    scene.addItem(base_item)

    window = SimpleNamespace(
        _host=host,
        graphicsViewFrame=view,
        videoSeekSlider=QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal),
        videoSeekLineEdit=QtWidgets.QLineEdit(),
        videoTimeLineEdit=None,
        video_processor=SimpleNamespace(media_path="/media/clip.mp4", fps=25.0),
        control={"OutputMediaFolder": str(tmp_path)},
        session_screenshot=None,
        session_screenshot_source_path="",
        session_screenshot_frame_number=0,
        session_screenshot_saved_path="",
        _base_pixmap_item=base_item,
        _screenshot_overlay_item=None,
        showScreenshotButton=QtWidgets.QPushButton("Show SS"),
        saveScreenshotButton=QtWidgets.QPushButton("Save SS"),
        screenshotStatusIndicator=QtWidgets.QLabel("●"),
    )
    window.showScreenshotButton.setCheckable(True)
    window.videoSeekSlider.setRange(0, 100)
    window.videoSeekSlider.setValue(42)
    return window, base_item


def test_screenshot_path_uses_capture_metadata_and_avoids_collisions(tmp_path):
    timestamp = datetime(2026, 10, 25, 12, 30, 45)
    first = screenshot_actions.build_screenshot_path(
        str(tmp_path), "/media/my clip.mp4", 17, timestamp
    )
    first.touch()
    second = screenshot_actions.build_screenshot_path(
        str(tmp_path), "/media/my clip.mp4", 17, timestamp
    )

    assert first.name == "my clip_frame_000017_20261025_123045.png"
    assert second.name == "my clip_frame_000017_20261025_123045_01.png"


def test_capture_show_live_update_and_save(qapp, tmp_path, monkeypatch):
    window, base_item = _make_window(qapp, tmp_path)
    monkeypatch.setattr(
        screenshot_actions.common_widget_actions,
        "create_and_show_toast_message",
        lambda *args, **kwargs: None,
    )

    assert screenshot_actions.take_screenshot(window)
    base_item.setPixmap(_pixmap(QtGui.QColor("blue")))
    assert window.session_screenshot.toImage().pixelColor(2, 2) == QtGui.QColor("red")
    assert "55c978" in window.screenshotStatusIndicator.styleSheet()

    assert screenshot_actions.show_screenshot(window, True)
    overlay = graphics_view_actions.get_screenshot_overlay_item(window)
    assert overlay is not None and overlay.isVisible()
    assert overlay.pixmap().toImage().pixelColor(2, 2) == QtGui.QColor("red")

    graphics_view_actions.update_graphics_view(
        window, _pixmap(QtGui.QColor("green")), 43
    )
    assert base_item.pixmap().toImage().pixelColor(2, 2) == QtGui.QColor("green")
    assert graphics_view_actions.get_display_pixmap_item(window) is overlay
    assert overlay.pixmap().toImage().pixelColor(2, 2) == QtGui.QColor("red")

    window.videoSeekSlider.setValue(99)
    window.video_processor.media_path = "/media/different.mp4"
    saved_path = screenshot_actions.save_screenshot(window)
    assert saved_path is not None and saved_path.is_file()
    assert "clip_frame_000042_" in saved_path.name
    assert QtGui.QImage(str(saved_path)).pixelColor(2, 2) == QtGui.QColor("red")

    screenshot_actions.show_screenshot(window, False)
    assert not overlay.isVisible()
    assert graphics_view_actions.get_display_pixmap_item(window) is base_item


def test_session_screenshot_survives_scene_clear(qapp, tmp_path):
    window, _ = _make_window(qapp, tmp_path)
    assert screenshot_actions.take_screenshot(window)
    saved = window.session_screenshot

    screenshot_actions.show_screenshot(window, True)
    window.showScreenshotButton.setChecked(True)
    window.graphicsViewFrame.scene().clear()
    new_base = QtWidgets.QGraphicsPixmapItem(_pixmap(QtGui.QColor("blue")))
    window.graphicsViewFrame.scene().addItem(new_base)
    window._base_pixmap_item = new_base

    assert window.session_screenshot is saved
    graphics_view_actions.update_graphics_view(
        window, _pixmap(QtGui.QColor("green")), 43
    )
    overlay = graphics_view_actions.get_screenshot_overlay_item(window)
    assert overlay is not None and overlay.scene() is window.graphicsViewFrame.scene()
    assert overlay.isVisible()
    assert overlay.pixmap().toImage().pixelColor(2, 2) == QtGui.QColor("red")


def test_save_requires_output_folder(qapp, tmp_path, monkeypatch):
    window, _ = _make_window(qapp, tmp_path)
    window.control["OutputMediaFolder"] = ""
    assert screenshot_actions.take_screenshot(window)
    messages = []
    monkeypatch.setattr(
        screenshot_actions.common_widget_actions,
        "create_and_show_toast_message",
        lambda *args, **kwargs: messages.append(args[1:3]),
    )

    assert screenshot_actions.save_screenshot(window) is None
    assert messages == [("Output Folder Required", "Select an Output Folder in Settings before saving a screenshot.")]


def test_screenshot_controls_are_added_once(qapp):
    host = QtWidgets.QWidget()
    center = QtWidgets.QWidget(host)
    layout = QtWidgets.QHBoxLayout(center)
    window = SimpleNamespace(
        mediaControlsMainLayout=layout,
        mediaControlsCenterWidget=center,
        scanToolsToggleButton=QtWidgets.QPushButton("scan"),
        session_screenshot=None,
    )
    layout.addWidget(window.scanToolsToggleButton)

    screenshot_actions.add_screenshot_controls(window)
    screenshot_actions.add_screenshot_controls(window)

    assert not window.showScreenshotButton.isEnabled()
    assert not window.saveScreenshotButton.isEnabled()
    assert not hasattr(window, "previousFrameStepButton")
    assert not hasattr(window, "nextFrameStepButton")
    widgets = [
        layout.itemAt(index).widget()
        for index in range(layout.count())
        if layout.itemAt(index).widget() is not None
    ]
    assert widgets.count(window.takeScreenshotButton) == 1
    assert widgets.count(window.showScreenshotButton) == 1
    assert widgets.count(window.saveScreenshotButton) == 1
