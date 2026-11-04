"""Dedicated single-frame navigation controls."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6 import QtCore, QtGui, QtWidgets

from app.ui.widgets.actions import video_control_actions

if TYPE_CHECKING:
    from app.ui.main_ui import MainWindow


def add_frame_step_controls(main_window: "MainWindow") -> None:
    """Add exact previous/next-frame buttons beside the configurable skip controls."""
    if hasattr(main_window, "previousFrameStepButton"):
        return

    layout = main_window.mediaControlsMainLayout
    parent = main_window.mediaControlsCenterWidget

    previous_button = QtWidgets.QPushButton("−1", parent)
    previous_button.setObjectName("previousFrameStepButton")
    previous_button.setIcon(QtGui.QIcon(":/media/media/tl_left_off.png"))
    previous_button.setToolTip("Previous frame (1 frame)")
    previous_button.setFlat(True)
    previous_button.clicked.connect(
        lambda: video_control_actions.rewind_video_slider_by_n_frames(main_window, 1)
    )
    main_window.previousFrameStepButton = previous_button
    rewind_index = layout.indexOf(main_window.frameRewindButton)
    layout.insertWidget(rewind_index, previous_button)

    next_button = QtWidgets.QPushButton("+1", parent)
    next_button.setObjectName("nextFrameStepButton")
    next_button.setIcon(QtGui.QIcon(":/media/media/tl_right_off.png"))
    next_button.setToolTip("Next frame (1 frame)")
    next_button.setFlat(True)
    next_button.clicked.connect(
        lambda: video_control_actions.advance_video_slider_by_n_frames(main_window, 1)
    )
    main_window.nextFrameStepButton = next_button
    advance_index = layout.indexOf(main_window.frameAdvanceButton)
    layout.insertWidget(advance_index + 1, next_button)

    if hasattr(main_window, "_queue_media_controls_balance"):
        main_window._queue_media_controls_balance()

    play_button = getattr(main_window, "buttonMediaPlay", None)
    if play_button is not None:
        target_height = play_button.sizeHint().height()
        for button in (previous_button, next_button):
            button.setMinimumHeight(target_height)
            button.setMaximumHeight(target_height)
            button.setMinimumWidth(40)
            button.setMaximumWidth(96)
            button.setIconSize(QtCore.QSize(22, 22))
