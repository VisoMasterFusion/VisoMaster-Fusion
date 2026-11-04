from __future__ import annotations

import importlib
import os
import sys
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest


def _stub(name: str) -> MagicMock:
    m = MagicMock()
    m.__name__ = name
    m.__spec__ = None
    return m


@pytest.fixture(scope="module")
def video_actions_env():
    stubbed_modules = {
        "PySide6": _stub("PySide6"),
        "PySide6.QtWidgets": _stub("PySide6.QtWidgets"),
        "PySide6.QtCore": _stub("PySide6.QtCore"),
        "PySide6.QtGui": _stub("PySide6.QtGui"),
        "cv2": _stub("cv2"),
        "numpy": _stub("numpy"),
        "PIL": _stub("PIL"),
        "PIL.Image": _stub("PIL.Image"),
        "app.helpers": _stub("app.helpers"),
        "app.helpers.typing_helper": _stub("app.helpers.typing_helper"),
        "app.helpers.miscellaneous": _stub("app.helpers.miscellaneous"),
        "app.ui.widgets.widget_components": _stub("app.ui.widgets.widget_components"),
        "app.ui.widgets.ui_workers": _stub("app.ui.widgets.ui_workers"),
        "app.ui.widgets.actions.common_actions": _stub(
            "app.ui.widgets.actions.common_actions"
        ),
        "app.ui.widgets.actions.card_actions": _stub(
            "app.ui.widgets.actions.card_actions"
        ),
        "app.ui.widgets.actions.graphics_view_actions": _stub(
            "app.ui.widgets.actions.graphics_view_actions"
        ),
        "app.ui.widgets.actions.layout_actions": _stub(
            "app.ui.widgets.actions.layout_actions"
        ),
    }
    saved_modules = {
        name: sys.modules.get(name)
        for name in [
            *stubbed_modules,
            "app.ui.widgets.actions.video_control_actions",
        ]
    }
    saved_package_attrs: dict[tuple[str, str], tuple[bool, object | None]] = {}

    for module_name in [
        *stubbed_modules,
        "app.ui.widgets.actions.video_control_actions",
    ]:
        parent_name, _, attr_name = module_name.rpartition(".")
        if not parent_name:
            continue
        parent_module = sys.modules.get(parent_name)
        had_attr = parent_module is not None and hasattr(parent_module, attr_name)
        saved_package_attrs[(parent_name, attr_name)] = (
            had_attr,
            getattr(parent_module, attr_name) if had_attr else None,
        )

    try:
        for name, module in stubbed_modules.items():
            sys.modules[name] = module

        stubbed_modules["PySide6"].QtWidgets = stubbed_modules["PySide6.QtWidgets"]
        stubbed_modules["PySide6"].QtCore = stubbed_modules["PySide6.QtCore"]
        stubbed_modules["PySide6"].QtGui = stubbed_modules["PySide6.QtGui"]
        stubbed_modules["PIL"].Image = stubbed_modules["PIL.Image"]
        stubbed_modules["app.helpers"].typing_helper = stubbed_modules[
            "app.helpers.typing_helper"
        ]
        stubbed_modules["app.helpers"].miscellaneous = stubbed_modules[
            "app.helpers.miscellaneous"
        ]
        for module_name, module in stubbed_modules.items():
            parent_name, _, attr_name = module_name.rpartition(".")
            parent_module = sys.modules.get(parent_name)
            if parent_module is not None and attr_name:
                setattr(parent_module, attr_name, module)

        sys.modules.pop("app.ui.widgets.actions.video_control_actions", None)

        common_widget_actions = importlib.import_module(
            "app.ui.widgets.actions.common_actions"
        )
        video_control_actions = importlib.import_module(
            "app.ui.widgets.actions.video_control_actions"
        )
        # These shared mocks are patched by several tests. Define their attributes
        # eagerly so monkeypatch restores them rather than deleting lazy children.
        video_control_actions.QtWidgets.QMessageBox = MagicMock()
        video_control_actions.QtWidgets.QMessageBox.question = MagicMock()
        video_control_actions.QtCore.QThread.msleep = MagicMock()
        video_control_actions.widget_components.ProgressDialog = MagicMock()
        video_control_actions.misc_helpers.read_image_file = MagicMock()
        video_control_actions.misc_helpers.read_frame = MagicMock()
        video_control_actions.misc_helpers.seek_frame = MagicMock()
        video_control_actions.misc_helpers.release_capture = MagicMock()
        video_control_actions.misc_helpers.get_output_file_path = MagicMock()
        video_control_actions.card_actions.clear_target_faces = MagicMock()
        video_control_actions.card_actions.find_target_faces = MagicMock()
        video_control_actions.cv2.VideoCapture = MagicMock()

        yield SimpleNamespace(
            module=video_control_actions,
            common_widget_actions=common_widget_actions,
            view_fullscreen=video_control_actions.view_fullscreen,
            toggle_theatre_mode=video_control_actions.toggle_theatre_mode,
            record_video=video_control_actions.record_video,
            process_batch_images=video_control_actions.process_batch_images,
            disable_compare_preview_modes_for_recording=(
                video_control_actions._disable_compare_preview_modes_for_recording
            ),
        )
    finally:
        for name, original_module in saved_modules.items():
            if original_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original_module

        for (parent_name, attr_name), (
            had_attr,
            original_value,
        ) in saved_package_attrs.items():
            parent_module = sys.modules.get(parent_name)
            if parent_module is None:
                continue
            if had_attr:
                setattr(parent_module, attr_name, original_value)
            elif hasattr(parent_module, attr_name):
                delattr(parent_module, attr_name)


def test_view_fullscreen_keeps_theatre_mode_active(video_actions_env):
    synced = []
    main_window = SimpleNamespace(
        is_theatre_mode=True,
        is_full_screen=False,
        _fullscreen_restore_was_maximized=False,
        _fullscreen_restore_geometry=None,
        showFullScreen=MagicMock(),
        showNormal=MagicMock(),
        isMaximized=lambda: False,
        isFullScreen=lambda: False,
        normalGeometry=lambda: "normal-geometry",
        geometry=lambda: "live-geometry",
        _sync_viewer_menu_actions=lambda: synced.append(True),
    )

    video_actions_env.view_fullscreen(main_window)

    main_window.showFullScreen.assert_called_once()
    main_window.showNormal.assert_not_called()
    assert main_window.is_theatre_mode is True
    assert main_window.is_full_screen is True
    assert main_window._fullscreen_restore_geometry == "live-geometry"
    assert synced == [True]


def test_view_fullscreen_uses_real_window_transition_outside_theatre(video_actions_env):
    synced = []
    main_window = SimpleNamespace(
        is_theatre_mode=False,
        is_full_screen=True,
        showFullScreen=MagicMock(),
        showNormal=MagicMock(),
        isMaximized=lambda: False,
        isFullScreen=lambda: False,
        normalGeometry=lambda: "normal-geometry",
        geometry=lambda: "live-geometry",
        _sync_viewer_menu_actions=lambda: synced.append(True),
    )

    video_actions_env.view_fullscreen(main_window)

    main_window.showFullScreen.assert_called_once()
    main_window.showNormal.assert_not_called()
    assert synced == [True]


class _FakeGeometry:
    def __init__(
        self,
        x: int = 100,
        y: int = 200,
        width: int = 900,
        height: int = 600,
    ):
        self._x = x
        self._y = y
        self._width = width
        self._height = height

    def x(self):
        return self._x

    def y(self):
        return self._y

    def width(self):
        return self._width

    def height(self):
        return self._height


class _StatefulFullscreenWindow:
    def __init__(
        self,
        *,
        state: str = "normal",
        geometry: _FakeGeometry | None = None,
        is_theatre_mode: bool = False,
        theatre_forced_fullscreen: bool = False,
    ):
        self._state = state
        self._geometry = geometry or _FakeGeometry()
        self._normal_geometry = self._geometry
        self.is_theatre_mode = is_theatre_mode
        self.is_full_screen = state == "fullscreen"
        self._fullscreen_restore_was_maximized = False
        self._fullscreen_restore_geometry = None
        self._was_maximized = state == "maximized"
        self._was_custom_fullscreen = False
        self._was_normal_geometry = self._normal_geometry
        self._theatre_forced_fullscreen = theatre_forced_fullscreen
        self._sync_calls: list[bool] = []
        self._theatre_snapshot_sync_calls = 0
        self.showFullScreen = MagicMock(side_effect=self._show_fullscreen)
        self.showNormal = MagicMock(side_effect=self._show_normal)
        self.showMaximized = MagicMock(side_effect=self._show_maximized)
        self.setGeometry = MagicMock(side_effect=self._set_geometry)

    def _show_fullscreen(self):
        self._state = "fullscreen"
        self.is_full_screen = True

    def _show_normal(self):
        self._state = "normal"
        self.is_full_screen = False

    def _show_maximized(self):
        self._state = "maximized"
        self.is_full_screen = False

    def _set_geometry(self, geometry):
        self._geometry = geometry
        self._normal_geometry = geometry

    def isFullScreen(self):
        return self._state == "fullscreen"

    def isMaximized(self):
        return self._state == "maximized"

    def normalGeometry(self):
        return self._normal_geometry

    def geometry(self):
        return self._geometry

    def _sync_viewer_menu_actions(self):
        self._sync_calls.append(True)

    def _sync_theatre_base_window_snapshot(self):
        self._theatre_snapshot_sync_calls += 1
        if not self.is_theatre_mode:
            return

        if self.isFullScreen():
            if self._theatre_forced_fullscreen:
                return
            self._was_custom_fullscreen = True
            self._was_maximized = False
            self._was_normal_geometry = (
                self._fullscreen_restore_geometry
                if self._fullscreen_restore_geometry is not None
                else self.normalGeometry()
            )
            return

        self._was_custom_fullscreen = False
        if self.isMaximized():
            self._was_maximized = True
            self._was_normal_geometry = self.normalGeometry()
        else:
            self._was_maximized = False
            self._was_normal_geometry = self.geometry()


def test_view_fullscreen_restores_saved_geometry_after_round_trip(video_actions_env):
    geometry = _FakeGeometry()
    main_window = _StatefulFullscreenWindow(state="normal", geometry=geometry)

    video_actions_env.view_fullscreen(main_window)
    video_actions_env.view_fullscreen(main_window)

    main_window.showFullScreen.assert_called_once()
    main_window.showNormal.assert_called_once()
    assert main_window.setGeometry.call_args_list[-1].args == (geometry,)
    assert main_window.isFullScreen() is False
    assert main_window.isMaximized() is False
    assert main_window._fullscreen_restore_was_maximized is False
    assert main_window._fullscreen_restore_geometry is None
    assert main_window._sync_calls == [True, True]


def test_view_fullscreen_restores_maximized_state_after_round_trip(video_actions_env):
    main_window = _StatefulFullscreenWindow(state="maximized")

    video_actions_env.view_fullscreen(main_window)
    video_actions_env.view_fullscreen(main_window)

    main_window.showFullScreen.assert_called_once()
    main_window.showMaximized.assert_called_once()
    main_window.showNormal.assert_not_called()
    main_window.setGeometry.assert_not_called()
    assert main_window.isMaximized() is True
    assert main_window.isFullScreen() is False
    assert main_window._fullscreen_restore_was_maximized is False
    assert main_window._fullscreen_restore_geometry is None
    assert main_window._sync_calls == [True, True]


def test_view_fullscreen_preserves_maximized_base_state_in_theatre(video_actions_env):
    main_window = _StatefulFullscreenWindow(state="maximized", is_theatre_mode=True)
    main_window._was_maximized = True
    main_window._was_custom_fullscreen = False
    main_window._was_normal_geometry = main_window.normalGeometry()

    video_actions_env.view_fullscreen(main_window)
    video_actions_env.view_fullscreen(main_window)

    assert main_window.is_theatre_mode is True
    assert main_window.isMaximized() is True
    assert main_window.isFullScreen() is False
    assert main_window.showMaximized.call_count == 1
    assert main_window._was_maximized is True
    assert main_window._was_custom_fullscreen is False
    assert main_window._was_normal_geometry == main_window.normalGeometry()
    assert main_window._fullscreen_restore_geometry is None
    assert main_window._theatre_snapshot_sync_calls == 2


def test_view_fullscreen_preserves_normal_geometry_in_theatre(video_actions_env):
    geometry = _FakeGeometry()
    main_window = _StatefulFullscreenWindow(
        state="normal",
        geometry=geometry,
        is_theatre_mode=True,
    )
    main_window._was_maximized = False
    main_window._was_custom_fullscreen = False
    main_window._was_normal_geometry = geometry

    video_actions_env.view_fullscreen(main_window)
    video_actions_env.view_fullscreen(main_window)

    assert main_window.is_theatre_mode is True
    assert main_window.isMaximized() is False
    assert main_window.isFullScreen() is False
    assert main_window.showNormal.call_count == 1
    assert main_window.setGeometry.call_args_list[-1].args == (geometry,)
    assert main_window._was_maximized is False
    assert main_window._was_custom_fullscreen is False
    assert main_window._was_normal_geometry is geometry
    assert main_window._theatre_snapshot_sync_calls == 2


def test_view_fullscreen_keeps_theatre_layout_active_during_round_trip(
    video_actions_env,
):
    main_window = _StatefulFullscreenWindow(state="normal", is_theatre_mode=True)
    base_geometry = main_window.geometry()
    main_window._was_maximized = False
    main_window._was_custom_fullscreen = False
    main_window._was_normal_geometry = base_geometry

    video_actions_env.view_fullscreen(main_window)

    assert main_window.is_theatre_mode is True
    assert main_window.isFullScreen() is True
    assert main_window._was_normal_geometry is base_geometry
    assert main_window._was_custom_fullscreen is True
    assert main_window._was_maximized is False

    video_actions_env.view_fullscreen(main_window)

    assert main_window.is_theatre_mode is True
    assert main_window.isFullScreen() is False
    assert main_window.geometry() is base_geometry
    assert main_window._theatre_snapshot_sync_calls == 2


class _FakeWidget:
    def __init__(self, visible: bool = True):
        self._visible = visible
        self._minimum_height = None

    def isVisible(self):
        return self._visible

    def hide(self):
        self._visible = False

    def show(self):
        self._visible = True

    def sizeHint(self):
        return SimpleNamespace(height=lambda: 42)

    def setMinimumHeight(self, value):
        self._minimum_height = value


class _FakeMenuBar(_FakeWidget):
    pass


class _FakeLayout:
    def count(self):
        return 0

    def itemAt(self, _index):
        raise IndexError

    def takeAt(self, _index):
        raise IndexError

    def setContentsMargins(self, *_args):
        return None

    def contentsMargins(self):
        return (0, 0, 0, 0)

    def setSpacing(self, *_args):
        return None

    def spacing(self):
        return 0

    def insertItem(self, *_args):
        return None

    def invalidate(self):
        return None


class _FakeGraphicsViewFrame:
    def frameShape(self):
        return "box"

    def setFrameShape(self, *_args):
        return None

    def setStyleSheet(self, *_args):
        return None

    def setVerticalScrollBarPolicy(self, *_args):
        return None

    def setHorizontalScrollBarPolicy(self, *_args):
        return None


def _make_theatre_entry_window(
    *,
    is_fullscreen: bool,
    is_maximized: bool = False,
    theatre_uses_fullscreen: bool = False,
):
    menu_bar = _FakeMenuBar()
    state = "fullscreen" if is_fullscreen else "maximized" if is_maximized else "normal"

    def _show_fullscreen():
        nonlocal state
        state = "fullscreen"

    def _show_normal():
        nonlocal state
        state = "normal"

    def _show_maximized():
        nonlocal state
        state = "maximized"

    return SimpleNamespace(
        is_theatre_mode=False,
        is_full_screen=False,
        _saved_window_state=None,
        _theatre_forced_fullscreen=False,
        input_Target_DockWidget=_FakeWidget(),
        input_Faces_DockWidget=_FakeWidget(),
        jobManagerDockWidget=_FakeWidget(),
        controlOptionsDockWidget=_FakeWidget(),
        facesPanelGroupBox=_FakeWidget(),
        menuBar=lambda: menu_bar,
        horizontalLayout=_FakeLayout(),
        verticalLayout=_FakeLayout(),
        verticalLayoutMediaControls=_FakeLayout(),
        panelVisibilityCheckBoxLayout=_FakeLayout(),
        graphicsViewFrame=_FakeGraphicsViewFrame(),
        saveState=lambda: "window-state",
        isMaximized=lambda: state == "maximized",
        isFullScreen=lambda: state == "fullscreen",
        normalGeometry=lambda: "normal-geometry",
        geometry=lambda: "live-geometry",
        control={"TheatreModeUsesFullscreenToggle": theatre_uses_fullscreen},
        setWindowState=MagicMock(),
        showFullScreen=MagicMock(side_effect=_show_fullscreen),
        showNormal=MagicMock(side_effect=_show_normal),
        showMaximized=MagicMock(side_effect=_show_maximized),
        setGeometry=MagicMock(),
    )


def test_toggle_theatre_mode_keeps_fullscreen_when_base_mode_is_fullscreen(
    monkeypatch, video_actions_env
):
    monkeypatch.setattr(
        video_actions_env.module, "_set_media_controls_visible", lambda *_args: None
    )
    video_actions_env.module.layout_actions.fit_image_to_view_onchange.reset_mock()
    main_window = _make_theatre_entry_window(is_fullscreen=True)

    video_actions_env.toggle_theatre_mode(main_window)

    assert main_window._was_custom_fullscreen is True
    assert main_window._was_normal_geometry == "normal-geometry"
    main_window.setWindowState.assert_called_once_with(
        video_actions_env.module.QtCore.Qt.WindowState.WindowFullScreen
    )
    main_window.showFullScreen.assert_called_once()
    assert main_window.is_full_screen is True


def test_toggle_theatre_mode_keeps_normal_window_when_base_mode_is_windowed(
    monkeypatch, video_actions_env
):
    monkeypatch.setattr(
        video_actions_env.module, "_set_media_controls_visible", lambda *_args: None
    )
    video_actions_env.module.layout_actions.fit_image_to_view_onchange.reset_mock()
    main_window = _make_theatre_entry_window(is_fullscreen=False, is_maximized=False)

    video_actions_env.toggle_theatre_mode(main_window)

    assert main_window._was_custom_fullscreen is False
    main_window.setWindowState.assert_not_called()
    main_window.showFullScreen.assert_not_called()
    assert main_window.is_full_screen is False


def test_toggle_theatre_mode_keeps_maximized_window_when_base_mode_is_maximized(
    monkeypatch, video_actions_env
):
    monkeypatch.setattr(
        video_actions_env.module, "_set_media_controls_visible", lambda *_args: None
    )
    video_actions_env.module.layout_actions.fit_image_to_view_onchange.reset_mock()
    main_window = _make_theatre_entry_window(is_fullscreen=False, is_maximized=True)

    video_actions_env.toggle_theatre_mode(main_window)

    assert main_window._was_custom_fullscreen is False
    assert main_window._was_maximized is True
    main_window.setWindowState.assert_not_called()
    main_window.showFullScreen.assert_not_called()
    assert main_window.is_full_screen is False


def test_toggle_theatre_mode_enters_fullscreen_from_windowed_state_when_enabled(
    monkeypatch, video_actions_env
):
    monkeypatch.setattr(
        video_actions_env.module, "_set_media_controls_visible", lambda *_args: None
    )
    video_actions_env.module.layout_actions.fit_image_to_view_onchange.reset_mock()
    main_window = _make_theatre_entry_window(
        is_fullscreen=False,
        is_maximized=False,
        theatre_uses_fullscreen=True,
    )

    video_actions_env.toggle_theatre_mode(main_window)

    assert main_window._was_custom_fullscreen is False
    assert main_window._theatre_forced_fullscreen is True
    assert main_window._was_maximized is False
    assert main_window._was_normal_geometry == "live-geometry"
    main_window.setWindowState.assert_called_once_with(
        video_actions_env.module.QtCore.Qt.WindowState.WindowFullScreen
    )
    main_window.showFullScreen.assert_called_once()
    assert main_window.is_full_screen is True


def test_toggle_theatre_mode_enters_fullscreen_from_maximized_state_when_enabled(
    monkeypatch, video_actions_env
):
    monkeypatch.setattr(
        video_actions_env.module, "_set_media_controls_visible", lambda *_args: None
    )
    video_actions_env.module.layout_actions.fit_image_to_view_onchange.reset_mock()
    main_window = _make_theatre_entry_window(
        is_fullscreen=False,
        is_maximized=True,
        theatre_uses_fullscreen=True,
    )

    video_actions_env.toggle_theatre_mode(main_window)

    assert main_window._was_custom_fullscreen is False
    assert main_window._theatre_forced_fullscreen is True
    assert main_window._was_maximized is True
    assert main_window._was_normal_geometry == "normal-geometry"
    main_window.setWindowState.assert_called_once_with(
        video_actions_env.module.QtCore.Qt.WindowState.WindowFullScreen
    )
    main_window.showFullScreen.assert_called_once()
    assert main_window.is_full_screen is True


def test_toggle_theatre_mode_keeps_existing_fullscreen_when_setting_enabled(
    monkeypatch, video_actions_env
):
    monkeypatch.setattr(
        video_actions_env.module, "_set_media_controls_visible", lambda *_args: None
    )
    video_actions_env.module.layout_actions.fit_image_to_view_onchange.reset_mock()
    main_window = _make_theatre_entry_window(
        is_fullscreen=True,
        theatre_uses_fullscreen=True,
    )

    video_actions_env.toggle_theatre_mode(main_window)

    assert main_window._was_custom_fullscreen is True
    assert main_window._theatre_forced_fullscreen is False
    assert main_window._was_normal_geometry == "normal-geometry"
    main_window.setWindowState.assert_called_once_with(
        video_actions_env.module.QtCore.Qt.WindowState.WindowFullScreen
    )
    main_window.showFullScreen.assert_called_once()
    assert main_window.is_full_screen is True


def test_toggle_theatre_mode_seeds_fullscreen_restore_geometry_when_forced(
    monkeypatch, video_actions_env
):
    monkeypatch.setattr(
        video_actions_env.module, "_set_media_controls_visible", lambda *_args: None
    )
    video_actions_env.module.layout_actions.fit_image_to_view_onchange.reset_mock()
    main_window = _make_theatre_entry_window(
        is_fullscreen=False,
        is_maximized=False,
        theatre_uses_fullscreen=True,
    )

    video_actions_env.toggle_theatre_mode(main_window)
    video_actions_env.view_fullscreen(main_window)

    assert main_window.is_theatre_mode is True
    assert main_window._fullscreen_restore_was_maximized is False
    assert main_window._fullscreen_restore_geometry is None
    main_window.showNormal.assert_called_once()
    assert main_window.setGeometry.call_args_list[-1].args == ("live-geometry",)


def test_toggle_theatre_mode_seeds_fullscreen_restore_maximized_when_forced(
    monkeypatch, video_actions_env
):
    monkeypatch.setattr(
        video_actions_env.module, "_set_media_controls_visible", lambda *_args: None
    )
    video_actions_env.module.layout_actions.fit_image_to_view_onchange.reset_mock()
    main_window = _make_theatre_entry_window(
        is_fullscreen=False,
        is_maximized=True,
        theatre_uses_fullscreen=True,
    )

    video_actions_env.toggle_theatre_mode(main_window)
    video_actions_env.view_fullscreen(main_window)

    assert main_window.is_theatre_mode is True
    assert main_window._fullscreen_restore_was_maximized is False
    assert main_window._fullscreen_restore_geometry is None
    main_window.showMaximized.assert_called_once()
    main_window.showNormal.assert_not_called()


def test_toggle_theatre_mode_restores_saved_normal_geometry_on_exit(
    monkeypatch, video_actions_env
):
    monkeypatch.setattr(
        video_actions_env.module, "_set_media_controls_visible", lambda *_args: None
    )
    video_actions_env.module.layout_actions.fit_image_to_view_onchange.reset_mock()

    menu_bar = _FakeMenuBar()
    saved_geometry = SimpleNamespace(
        x=lambda: 100,
        y=lambda: 200,
        width=lambda: 900,
        height=lambda: 600,
    )
    set_geometry_calls = []
    main_window = SimpleNamespace(
        is_theatre_mode=True,
        _was_custom_fullscreen=False,
        _theatre_forced_fullscreen=True,
        _was_maximized=False,
        _was_normal_geometry=saved_geometry,
        _saved_window_state="window-state",
        _saved_dock_states={},
        _saved_layout_props={},
        _main_v_spacers=[],
        _top_bar_spacers=[],
        _top_bar_widgets_state={},
        input_Target_DockWidget=_FakeWidget(False),
        input_Faces_DockWidget=_FakeWidget(False),
        jobManagerDockWidget=_FakeWidget(False),
        controlOptionsDockWidget=_FakeWidget(False),
        facesPanelGroupBox=_FakeWidget(False),
        menuBar=lambda: menu_bar,
        horizontalLayout=_FakeLayout(),
        verticalLayout=_FakeLayout(),
        verticalLayoutMediaControls=_FakeLayout(),
        panelVisibilityCheckBoxLayout=_FakeLayout(),
        graphicsViewFrame=_FakeGraphicsViewFrame(),
        isFullScreen=lambda: False,
        normalGeometry=lambda: saved_geometry,
        geometry=lambda: saved_geometry,
        showFullScreen=MagicMock(),
        showMaximized=MagicMock(),
        showNormal=MagicMock(),
        setGeometry=lambda geometry: set_geometry_calls.append(geometry),
        restoreState=MagicMock(),
        setUpdatesEnabled=MagicMock(),
    )

    video_actions_env.toggle_theatre_mode(main_window)

    main_window.showNormal.assert_called_once()
    main_window.showFullScreen.assert_not_called()
    main_window.showMaximized.assert_not_called()
    assert set_geometry_calls == [saved_geometry]
    main_window.restoreState.assert_called_once_with("window-state")
    assert [call.args for call in main_window.setUpdatesEnabled.call_args_list] == [
        (False,),
        (True,),
    ]
    assert main_window._theatre_forced_fullscreen is False
    assert main_window.is_full_screen is False


def test_toggle_theatre_mode_restores_maximized_state_on_exit(
    monkeypatch, video_actions_env
):
    monkeypatch.setattr(
        video_actions_env.module, "_set_media_controls_visible", lambda *_args: None
    )
    video_actions_env.module.layout_actions.fit_image_to_view_onchange.reset_mock()

    menu_bar = _FakeMenuBar()
    main_window = SimpleNamespace(
        is_theatre_mode=True,
        _was_custom_fullscreen=False,
        _theatre_forced_fullscreen=True,
        _was_maximized=True,
        _was_normal_geometry="normal-geometry",
        _saved_window_state="window-state",
        _saved_dock_states={},
        _saved_layout_props={},
        _main_v_spacers=[],
        _top_bar_spacers=[],
        _top_bar_widgets_state={},
        input_Target_DockWidget=_FakeWidget(False),
        input_Faces_DockWidget=_FakeWidget(False),
        jobManagerDockWidget=_FakeWidget(False),
        controlOptionsDockWidget=_FakeWidget(False),
        facesPanelGroupBox=_FakeWidget(False),
        menuBar=lambda: menu_bar,
        horizontalLayout=_FakeLayout(),
        verticalLayout=_FakeLayout(),
        verticalLayoutMediaControls=_FakeLayout(),
        panelVisibilityCheckBoxLayout=_FakeLayout(),
        graphicsViewFrame=_FakeGraphicsViewFrame(),
        isFullScreen=lambda: False,
        isMaximized=lambda: False,
        normalGeometry=lambda: "normal-geometry",
        geometry=lambda: "live-geometry",
        showFullScreen=MagicMock(),
        showMaximized=MagicMock(),
        showNormal=MagicMock(),
        setGeometry=MagicMock(),
        restoreState=MagicMock(),
        setUpdatesEnabled=MagicMock(),
    )

    video_actions_env.toggle_theatre_mode(main_window)

    main_window.showFullScreen.assert_not_called()
    main_window.showMaximized.assert_called_once()
    main_window.showNormal.assert_not_called()
    main_window.setGeometry.assert_not_called()
    main_window.restoreState.assert_called_once_with("window-state")
    assert [call.args for call in main_window.setUpdatesEnabled.call_args_list] == [
        (False,),
        (True,),
    ]
    assert main_window._theatre_forced_fullscreen is False
    assert main_window.is_full_screen is False


def test_toggle_theatre_mode_restores_existing_fullscreen_state_on_exit(
    monkeypatch, video_actions_env
):
    monkeypatch.setattr(
        video_actions_env.module, "_set_media_controls_visible", lambda *_args: None
    )
    video_actions_env.module.layout_actions.fit_image_to_view_onchange.reset_mock()

    menu_bar = _FakeMenuBar()
    main_window = SimpleNamespace(
        is_theatre_mode=True,
        _was_custom_fullscreen=True,
        _theatre_forced_fullscreen=False,
        _was_maximized=False,
        _was_normal_geometry="normal-geometry",
        _saved_window_state="window-state",
        _saved_dock_states={},
        _saved_layout_props={},
        _main_v_spacers=[],
        _top_bar_spacers=[],
        _top_bar_widgets_state={},
        input_Target_DockWidget=_FakeWidget(False),
        input_Faces_DockWidget=_FakeWidget(False),
        jobManagerDockWidget=_FakeWidget(False),
        controlOptionsDockWidget=_FakeWidget(False),
        facesPanelGroupBox=_FakeWidget(False),
        menuBar=lambda: menu_bar,
        horizontalLayout=_FakeLayout(),
        verticalLayout=_FakeLayout(),
        verticalLayoutMediaControls=_FakeLayout(),
        panelVisibilityCheckBoxLayout=_FakeLayout(),
        graphicsViewFrame=_FakeGraphicsViewFrame(),
        isFullScreen=lambda: True,
        isMaximized=lambda: False,
        normalGeometry=lambda: "normal-geometry",
        geometry=lambda: "live-geometry",
        showFullScreen=MagicMock(),
        showMaximized=MagicMock(),
        showNormal=MagicMock(),
        setWindowState=MagicMock(),
        setGeometry=MagicMock(),
        restoreState=MagicMock(),
        setUpdatesEnabled=MagicMock(),
    )

    video_actions_env.toggle_theatre_mode(main_window)

    main_window.setWindowState.assert_called_once_with(
        video_actions_env.module.QtCore.Qt.WindowState.WindowFullScreen
    )
    main_window.showFullScreen.assert_not_called()
    main_window.showMaximized.assert_not_called()
    main_window.showNormal.assert_not_called()
    main_window.restoreState.assert_called_once_with("window-state")
    assert main_window._theatre_forced_fullscreen is False
    assert main_window.is_full_screen is True


def test_disable_compare_preview_modes_for_recording_disables_both_and_toasts(
    video_actions_env,
):
    calls = []
    main_window = SimpleNamespace(
        view_face_compare_enabled=True,
        view_face_mask_enabled=True,
        _set_compare_mode=lambda mode, checked: calls.append((mode, checked)),
    )
    video_actions_env.common_widget_actions.create_and_show_toast_message.reset_mock()

    video_actions_env.disable_compare_preview_modes_for_recording(main_window)

    assert calls == [("compare", False), ("mask", False)]
    video_actions_env.common_widget_actions.create_and_show_toast_message.assert_called_once()


def test_disable_compare_preview_modes_for_recording_is_noop_when_already_off(
    video_actions_env,
):
    calls = []
    main_window = SimpleNamespace(
        view_face_compare_enabled=False,
        view_face_mask_enabled=False,
        _set_compare_mode=lambda mode, checked: calls.append((mode, checked)),
    )
    video_actions_env.common_widget_actions.create_and_show_toast_message.reset_mock()

    video_actions_env.disable_compare_preview_modes_for_recording(main_window)

    assert calls == []
    video_actions_env.common_widget_actions.create_and_show_toast_message.assert_not_called()


class _FakeRecordButton:
    def __init__(self):
        self.checked_states = []
        self.block_calls = []

    def blockSignals(self, value):
        self.block_calls.append(value)

    def setChecked(self, value):
        self.checked_states.append(value)

    def setIcon(self, *_args):
        return None

    def setToolTip(self, *_args):
        return None


class _FakePromptBox:
    Warning = "warning"
    Yes = 1
    No = 2
    next_result = Yes
    instances: list["_FakePromptBox"] = []

    def __init__(self, _parent):
        self.window_title = None
        self.text = None
        self.informative_text = None
        self.standard_buttons = None
        self.default_button = None
        _FakePromptBox.instances.append(self)

    def setIcon(self, _icon):
        return None

    def setWindowTitle(self, value):
        self.window_title = value

    def setText(self, value):
        self.text = value

    def setInformativeText(self, value):
        self.informative_text = value

    def setStandardButtons(self, value):
        self.standard_buttons = value

    def setDefaultButton(self, value):
        self.default_button = value

    def exec(self):
        return _FakePromptBox.next_result


def _make_record_stop_window(
    *,
    confirm_before_stop: bool = True,
    recording: bool = False,
    is_processing_segments: bool = False,
    job_manager_initiated_record: bool = False,
):
    return SimpleNamespace(
        control={"ConfirmBeforeStoppingRecordingToggle": confirm_before_stop},
        video_processor=SimpleNamespace(
            file_type="video",
            recording=recording,
            is_processing_segments=is_processing_segments,
            finalize_segment_concatenation=MagicMock(),
            _finalize_default_style_recording=MagicMock(),
        ),
        buttonMediaRecord=_FakeRecordButton(),
        buttonMediaPlay=SimpleNamespace(setEnabled=MagicMock()),
        job_manager_initiated_record=job_manager_initiated_record,
    )


def test_record_video_prompts_before_manual_stop_when_setting_enabled(
    monkeypatch, video_actions_env
):
    _FakePromptBox.instances = []
    _FakePromptBox.next_result = _FakePromptBox.Yes
    monkeypatch.setattr(
        video_actions_env.module.QtWidgets, "QMessageBox", _FakePromptBox
    )
    main_window = _make_record_stop_window(recording=True, confirm_before_stop=True)

    video_actions_env.record_video(main_window, checked=False)

    assert len(_FakePromptBox.instances) == 1
    prompt = _FakePromptBox.instances[0]
    assert prompt.window_title == "Confirm stop"
    assert prompt.text == "Stop recording?"
    assert (
        prompt.informative_text
        == "Recording will stop immediately. Output may be incomplete."
    )
    main_window.video_processor._finalize_default_style_recording.assert_called_once()


def test_record_video_skips_prompt_before_manual_stop_when_setting_disabled(
    monkeypatch, video_actions_env
):
    _FakePromptBox.instances = []
    monkeypatch.setattr(
        video_actions_env.module.QtWidgets, "QMessageBox", _FakePromptBox
    )
    main_window = _make_record_stop_window(recording=True, confirm_before_stop=False)

    video_actions_env.record_video(main_window, checked=False)

    assert _FakePromptBox.instances == []
    main_window.video_processor._finalize_default_style_recording.assert_called_once()


def test_record_video_skips_prompt_for_job_manager_stop_even_when_enabled(
    monkeypatch, video_actions_env
):
    _FakePromptBox.instances = []
    monkeypatch.setattr(
        video_actions_env.module.QtWidgets, "QMessageBox", _FakePromptBox
    )
    main_window = _make_record_stop_window(
        recording=True,
        confirm_before_stop=True,
        job_manager_initiated_record=True,
    )

    video_actions_env.record_video(main_window, checked=False)

    assert _FakePromptBox.instances == []
    main_window.video_processor._finalize_default_style_recording.assert_called_once()


def test_record_video_rearms_toggle_when_stop_is_cancelled(
    monkeypatch, video_actions_env
):
    _FakePromptBox.instances = []
    _FakePromptBox.next_result = _FakePromptBox.No
    monkeypatch.setattr(
        video_actions_env.module.QtWidgets, "QMessageBox", _FakePromptBox
    )
    main_window = _make_record_stop_window(recording=True, confirm_before_stop=True)

    video_actions_env.record_video(main_window, checked=False)

    assert len(_FakePromptBox.instances) == 1
    assert main_window.buttonMediaRecord.block_calls == [True, False]
    assert main_window.buttonMediaRecord.checked_states == [True]
    main_window.video_processor._finalize_default_style_recording.assert_not_called()
    main_window.video_processor.finalize_segment_concatenation.assert_not_called()


def test_record_video_finalizes_segment_recording_after_confirmation(
    monkeypatch, video_actions_env
):
    _FakePromptBox.instances = []
    _FakePromptBox.next_result = _FakePromptBox.Yes
    monkeypatch.setattr(
        video_actions_env.module.QtWidgets, "QMessageBox", _FakePromptBox
    )
    main_window = _make_record_stop_window(
        is_processing_segments=True,
        confirm_before_stop=True,
    )

    video_actions_env.record_video(main_window, checked=False)

    assert len(_FakePromptBox.instances) == 1
    main_window.video_processor.finalize_segment_concatenation.assert_called_once()
    main_window.video_processor._finalize_default_style_recording.assert_not_called()


class _FakeBatchFrame:
    size = 1

    def __getitem__(self, _key):
        return self


class _FakeBatchInputFace:
    def __init__(self, face_id: str = "face_1", checked: bool = True):
        self.face_id = face_id
        self.embedding_store = {"embedding": face_id}
        self._checked = checked

    def isChecked(self):
        return self._checked


class _FakeBatchTargetFace:
    def __init__(self, face_id: str = "target_1"):
        self.face_id = face_id
        self.assigned_input_faces: dict[str, Any] = {}
        self.assigned_merged_embeddings: dict[str, Any] = {}
        self.calculate_assigned_input_embedding = MagicMock()


class _FakeBatchMediaWidget:
    def __init__(self, file_type: str, media_path: str):
        self.file_type = file_type
        self.media_path = media_path


class _FakeBatchList:
    def __init__(self, widgets):
        self._widgets = list(widgets)

    def count(self):
        return len(self._widgets)

    def item(self, index):
        return index

    def itemWidget(self, item):
        return self._widgets[item]


class _FakeOutputLineEdit:
    def __init__(self, value: str):
        self._value = value

    def text(self):
        return self._value


class _FakeSlider:
    def __init__(self):
        self.maximum = None
        self.value_set = None
        self.block_calls = []

    def setMaximum(self, value):
        self.maximum = value

    def blockSignals(self, value):
        self.block_calls.append(value)

    def setValue(self, value):
        self.value_set = value

    def value(self):
        return 0


class _FakeCapture:
    def __init__(self):
        self.opened = True

    def isOpened(self):
        return self.opened

    def get(self, prop):
        if prop == "frame_count":
            return 10
        if prop == "fps":
            return 24
        return 0

    def set(self, *_args):
        return True


class _FakeProgressDialog:
    instances: list["_FakeProgressDialog"] = []
    confirmed_sequence = [False]

    def __init__(self, *_args, **_kwargs):
        self.value_calls = []
        self.labels = []
        self.closed_without_confirmation = 0
        self.shown = 0
        self._confirmed_calls = 0
        _FakeProgressDialog.instances.append(self)

    def setWindowModality(self, *_args):
        return None

    def setWindowTitle(self, *_args):
        return None

    def setValue(self, value):
        self.value_calls.append(value)

    def show(self):
        self.shown += 1

    def setLabelText(self, value):
        self.labels.append(value)

    def confirmedCanceled(self):
        index = min(self._confirmed_calls, len(self.confirmed_sequence) - 1)
        self._confirmed_calls += 1
        return self.confirmed_sequence[index]

    def close_without_confirmation(self):
        self.closed_without_confirmation += 1


class _FakeLineEditText:
    def __init__(self, value: str):
        self._value = value

    def text(self):
        return self._value


def test_resolve_output_folder_preserves_source_directory_structure(video_actions_env):
    main_window = SimpleNamespace(
        control={
            "OutputMediaFolder": "E:/output",
            "OutputToTargetLocationToggle": False,
            "PreserveOutputDirectoryStructureToggle": True,
            "ClusterOutputBySourceToggle": False,
        },
        targetVideosPathLineEdit=_FakeLineEditText("E:/targets"),
        last_target_media_folder_path="",
        merged_embeddings={},
        cur_selected_target_face_button=None,
    )

    output_folder = video_actions_env.module.resolve_output_folder(
        main_window, "E:/targets/set_a/sub_01/image_1.png"
    )

    assert os.path.normpath(output_folder) == os.path.normpath("E:/output/set_a/sub_01")


def test_resolve_output_folder_preserve_and_cluster(video_actions_env):
    main_window = SimpleNamespace(
        control={
            "OutputMediaFolder": "E:/output",
            "OutputToTargetLocationToggle": False,
            "PreserveOutputDirectoryStructureToggle": True,
            "ClusterOutputBySourceToggle": True,
        },
        targetVideosPathLineEdit=_FakeLineEditText("E:/targets"),
        last_target_media_folder_path="",
        merged_embeddings={7: SimpleNamespace(embedding_name="embedding_alice")},
        cur_selected_target_face_button=SimpleNamespace(
            assigned_merged_embeddings={7: object()}
        ),
    )

    output_folder = video_actions_env.module.resolve_output_folder(
        main_window, "E:/targets/set_a/sub_01/image_1.png"
    )

    assert os.path.normpath(output_folder) == os.path.normpath(
        "E:/output/set_a/sub_01/embedding_alice"
    )


def test_resolve_output_folder_target_location_overrides_preserve(video_actions_env):
    main_window = SimpleNamespace(
        control={
            "OutputMediaFolder": "E:/output",
            "OutputToTargetLocationToggle": True,
            "PreserveOutputDirectoryStructureToggle": True,
            "ClusterOutputBySourceToggle": False,
        },
        targetVideosPathLineEdit=_FakeLineEditText("E:/targets"),
        last_target_media_folder_path="",
        merged_embeddings={},
        cur_selected_target_face_button=None,
    )

    output_folder = video_actions_env.module.resolve_output_folder(
        main_window, "E:/targets/set_a/sub_01/image_1.png"
    )

    assert os.path.normpath(output_folder) == os.path.normpath(
        "E:/targets/set_a/sub_01"
    )


def _make_batch_main_window(*widgets):
    frame = _FakeBatchFrame()
    video_processor = SimpleNamespace(
        media_path=None,
        file_type=None,
        current_frame_number=0,
        media_capture=None,
        current_frame=frame,
        max_frame_number=0,
        fps=0,
        processing=False,
        is_processing_segments=False,
        process_current_frame=MagicMock(side_effect=lambda synchronous=True: None),
        stop_processing=MagicMock(side_effect=lambda: None),
    )
    return SimpleNamespace(
        outputFolderLineEdit=_FakeOutputLineEdit("E:/output"),
        targetVideosList=_FakeBatchList(widgets),
        current_widget_parameters={"Strength": 50},
        input_faces={"face_1": _FakeBatchInputFace()},
        merged_embeddings={},
        target_faces={"original_face": object()},
        parameters={},
        control={
            "ImageFormatToggle": False,
            "OutputMediaFolder": "E:/output",
            "OutputToTargetLocationToggle": False,
            "ClusterOutputBySourceToggle": False,
        },
        video_processor=video_processor,
        videoSeekSlider=_FakeSlider(),
        scene=SimpleNamespace(clear=MagicMock()),
        is_batch_processing=False,
    )


def test_process_batch_images_all_faces_uses_non_confirming_close(
    monkeypatch, video_actions_env
):
    _FakeProgressDialog.instances = []
    _FakeProgressDialog.confirmed_sequence = [False]
    main_window = _make_batch_main_window(
        _FakeBatchMediaWidget("image", "E:/media/image_1.png")
    )

    monkeypatch.setattr(
        video_actions_env.module.widget_components,
        "ProgressDialog",
        _FakeProgressDialog,
    )
    monkeypatch.setattr(
        video_actions_env.module.numpy,
        "ndarray",
        _FakeBatchFrame,
        raising=False,
    )
    monkeypatch.setattr(
        video_actions_env.module.QtWidgets.QMessageBox,
        "question",
        lambda *_args, **_kwargs: video_actions_env.module.QtWidgets.QMessageBox.Yes,
    )
    monkeypatch.setattr(
        video_actions_env.module.os, "makedirs", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        video_actions_env.module.misc_helpers,
        "read_image_file",
        lambda *_args, **_kwargs: _FakeBatchFrame(),
    )
    monkeypatch.setattr(
        video_actions_env.module.misc_helpers,
        "get_output_file_path",
        lambda *_args, **_kwargs: "E:/output/image_1.png",
    )
    monkeypatch.setattr(
        video_actions_env.module.card_actions,
        "clear_target_faces",
        lambda mw, refresh_frame=False: mw.target_faces.clear(),
    )
    monkeypatch.setattr(
        video_actions_env.module.card_actions,
        "find_target_faces",
        lambda mw: mw.target_faces.update({"target_1": _FakeBatchTargetFace()}),
    )

    video_actions_env.process_batch_images(main_window, process_all_faces=True)

    progress_dialog = _FakeProgressDialog.instances[0]
    assert progress_dialog.closed_without_confirmation == 1
    video_actions_env.common_widget_actions.create_and_show_messagebox.assert_called()
    message = (
        video_actions_env.common_widget_actions.create_and_show_messagebox.call_args[0][
            2
        ]
    )
    assert "Batch processing complete." in message
    assert "Successfully processed: 1" in message


def test_process_batch_images_mixed_batch_uses_non_confirming_close(
    monkeypatch, video_actions_env
):
    _FakeProgressDialog.instances = []
    _FakeProgressDialog.confirmed_sequence = [False]
    main_window = _make_batch_main_window(
        _FakeBatchMediaWidget("image", "E:/media/image_1.png"),
        _FakeBatchMediaWidget("video", "E:/media/video_1.mp4"),
    )

    monkeypatch.setattr(
        video_actions_env.module.widget_components,
        "ProgressDialog",
        _FakeProgressDialog,
    )
    monkeypatch.setattr(
        video_actions_env.module.numpy,
        "ndarray",
        _FakeBatchFrame,
        raising=False,
    )
    monkeypatch.setattr(
        video_actions_env.module.QtWidgets.QMessageBox,
        "question",
        lambda *_args, **_kwargs: video_actions_env.module.QtWidgets.QMessageBox.Yes,
    )
    monkeypatch.setattr(
        video_actions_env.module.os, "makedirs", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        video_actions_env.module.misc_helpers,
        "read_image_file",
        lambda *_args, **_kwargs: _FakeBatchFrame(),
    )
    monkeypatch.setattr(
        video_actions_env.module.misc_helpers,
        "get_output_file_path",
        lambda *_args, **_kwargs: "E:/output/image_1.png",
    )
    monkeypatch.setattr(
        video_actions_env.module, "get_video_rotation", lambda *_args: 0
    )
    monkeypatch.setattr(
        video_actions_env.module.cv2,
        "VideoCapture",
        lambda *_args, **_kwargs: _FakeCapture(),
    )
    monkeypatch.setattr(
        video_actions_env.module.cv2,
        "CAP_PROP_ORIENTATION_AUTO",
        "orientation_auto",
        raising=False,
    )
    monkeypatch.setattr(
        video_actions_env.module.cv2,
        "CAP_PROP_FRAME_COUNT",
        "frame_count",
        raising=False,
    )
    monkeypatch.setattr(
        video_actions_env.module.cv2,
        "CAP_PROP_FPS",
        "fps",
        raising=False,
    )
    monkeypatch.setattr(
        video_actions_env.module.misc_helpers,
        "read_frame",
        lambda *_args, **_kwargs: (True, _FakeBatchFrame()),
    )
    monkeypatch.setattr(
        video_actions_env.module.misc_helpers,
        "seek_frame",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        video_actions_env.module.misc_helpers,
        "release_capture",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        video_actions_env.module,
        "record_video",
        lambda mw, _checked: setattr(mw.video_processor, "processing", False),
    )
    monkeypatch.setattr(
        video_actions_env.module.QtCore.QThread, "msleep", lambda *_args: None
    )

    video_actions_env.process_batch_images(main_window, process_all_faces=False)

    progress_dialog = _FakeProgressDialog.instances[0]
    assert progress_dialog.closed_without_confirmation == 1
    message = (
        video_actions_env.common_widget_actions.create_and_show_messagebox.call_args[0][
            2
        ]
    )
    assert "Successfully processed: 2" in message


def test_process_batch_images_cancelled_video_is_not_counted_completed(
    monkeypatch, video_actions_env
):
    _FakeProgressDialog.instances = []
    _FakeProgressDialog.confirmed_sequence = [False, True, True]
    main_window = _make_batch_main_window(
        _FakeBatchMediaWidget("video", "E:/media/video_1.mp4")
    )

    def _stop_processing():
        main_window.video_processor.processing = False

    main_window.video_processor.stop_processing = MagicMock(
        side_effect=_stop_processing
    )

    monkeypatch.setattr(
        video_actions_env.module.widget_components,
        "ProgressDialog",
        _FakeProgressDialog,
    )
    monkeypatch.setattr(
        video_actions_env.module.numpy,
        "ndarray",
        _FakeBatchFrame,
        raising=False,
    )
    monkeypatch.setattr(
        video_actions_env.module.QtWidgets.QMessageBox,
        "question",
        lambda *_args, **_kwargs: video_actions_env.module.QtWidgets.QMessageBox.Yes,
    )
    monkeypatch.setattr(
        video_actions_env.module, "get_video_rotation", lambda *_args: 0
    )
    monkeypatch.setattr(
        video_actions_env.module.cv2,
        "VideoCapture",
        lambda *_args, **_kwargs: _FakeCapture(),
    )
    monkeypatch.setattr(
        video_actions_env.module.cv2,
        "CAP_PROP_ORIENTATION_AUTO",
        "orientation_auto",
        raising=False,
    )
    monkeypatch.setattr(
        video_actions_env.module.cv2,
        "CAP_PROP_FRAME_COUNT",
        "frame_count",
        raising=False,
    )
    monkeypatch.setattr(
        video_actions_env.module.cv2,
        "CAP_PROP_FPS",
        "fps",
        raising=False,
    )
    monkeypatch.setattr(
        video_actions_env.module.misc_helpers,
        "read_frame",
        lambda *_args, **_kwargs: (True, _FakeBatchFrame()),
    )
    monkeypatch.setattr(
        video_actions_env.module.misc_helpers,
        "seek_frame",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        video_actions_env.module.misc_helpers,
        "release_capture",
        lambda *_args, **_kwargs: None,
    )

    def _record_video(mw, _checked):
        mw.video_processor.processing = True
        mw.video_processor.is_processing_segments = False

    monkeypatch.setattr(video_actions_env.module, "record_video", _record_video)
    monkeypatch.setattr(
        video_actions_env.module.QtCore.QThread, "msleep", lambda *_args: None
    )

    video_actions_env.process_batch_images(main_window, process_all_faces=False)

    main_window.video_processor.stop_processing.assert_called_once()
    message = (
        video_actions_env.common_widget_actions.create_and_show_messagebox.call_args[0][
            2
        ]
    )
    assert "Batch processing cancelled." in message
    assert "Processed: 0" in message
