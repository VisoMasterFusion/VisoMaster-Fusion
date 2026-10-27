"""Tests for dedicated single-frame navigation controls."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtWidgets

from app.ui.widgets.actions import frame_step_actions


@pytest.fixture(scope="module")
def qapp():
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


def test_navigation_buttons_step_exactly_one_frame(qapp, monkeypatch):
    host = QtWidgets.QWidget()
    center = QtWidgets.QWidget(host)
    layout = QtWidgets.QHBoxLayout(center)
    window = SimpleNamespace(
        mediaControlsMainLayout=layout,
        mediaControlsCenterWidget=center,
        frameRewindButton=QtWidgets.QPushButton("skip back"),
        buttonMediaRecord=QtWidgets.QPushButton("record"),
        buttonMediaPlay=QtWidgets.QPushButton("play"),
        frameAdvanceButton=QtWidgets.QPushButton("skip forward"),
        liveSoundButton=QtWidgets.QPushButton("sound"),
    )
    for button in (
        window.frameRewindButton,
        window.buttonMediaRecord,
        window.buttonMediaPlay,
        window.frameAdvanceButton,
        window.liveSoundButton,
    ):
        layout.addWidget(button)

    frame_step_actions.add_frame_step_controls(window)
    calls = []
    monkeypatch.setattr(
        frame_step_actions.video_control_actions,
        "rewind_video_slider_by_n_frames",
        lambda _window, n: calls.append(("back", n)),
    )
    monkeypatch.setattr(
        frame_step_actions.video_control_actions,
        "advance_video_slider_by_n_frames",
        lambda _window, n: calls.append(("forward", n)),
    )

    window.previousFrameStepButton.click()
    window.nextFrameStepButton.click()

    assert calls == [("back", 1), ("forward", 1)]
    assert layout.indexOf(window.previousFrameStepButton) < layout.indexOf(
        window.frameRewindButton
    )
    assert layout.indexOf(window.nextFrameStepButton) == layout.indexOf(
        window.frameAdvanceButton
    ) + 1


def test_frame_step_controls_are_not_added_twice(qapp):
    host = QtWidgets.QWidget()
    center = QtWidgets.QWidget(host)
    layout = QtWidgets.QHBoxLayout(center)
    window = SimpleNamespace(
        mediaControlsMainLayout=layout,
        mediaControlsCenterWidget=center,
        frameRewindButton=QtWidgets.QPushButton(),
        frameAdvanceButton=QtWidgets.QPushButton(),
    )
    layout.addWidget(window.frameRewindButton)
    layout.addWidget(window.frameAdvanceButton)

    frame_step_actions.add_frame_step_controls(window)
    frame_step_actions.add_frame_step_controls(window)

    buttons = [
        layout.itemAt(index).widget()
        for index in range(layout.count())
        if layout.itemAt(index).widget() is not None
    ]
    assert buttons.count(window.previousFrameStepButton) == 1
    assert buttons.count(window.nextFrameStepButton) == 1
