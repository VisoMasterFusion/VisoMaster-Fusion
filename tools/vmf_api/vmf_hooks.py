"""Import-time hooks for the VisoMaster launcher and app (no tracked file is modified).

A meta path finder waits for a given module to be imported, lets it execute
normally, then calls a patch function on it. If the expected class/method is
gone after an update, the patch logs a message and does nothing.
"""

from __future__ import annotations

import functools
import inspect
import os
import sys

from vmf_common import log, read_config

LAUNCHER_MODULE = "app.ui.launcher.launcher_window"
APP_MODULE = "app.ui.main_ui"


# ---------------------------------------------------------------------------
# Post-import finder
# ---------------------------------------------------------------------------


class _PostImportHook:
    """Meta path finder that runs callback(module) right after `fullname` executes."""

    def __init__(self, fullname, callback):
        self.fullname = fullname
        self.callback = callback

    def find_spec(self, fullname, path=None, target=None):
        if fullname != self.fullname:
            return None
        # Remove ourselves first: one-shot, and avoids recursion.
        try:
            sys.meta_path.remove(self)
        except ValueError:
            pass
        spec = None
        for finder in sys.meta_path:
            find = getattr(finder, "find_spec", None)
            if find is None:
                continue
            try:
                spec = find(fullname, path, target)
            except Exception:
                spec = None
            if spec is not None:
                break
        if spec is None or spec.loader is None or not hasattr(spec.loader, "exec_module"):
            log(f"cannot hook {fullname} (module not found)")
            return None
        loader = spec.loader
        original_exec = loader.exec_module
        callback = self.callback

        def exec_module(module):
            original_exec(module)
            try:
                callback(module)
            except Exception as e:  # never break the import
                log(f"patch of {fullname} skipped: {e!r}")

        loader.exec_module = exec_module
        return spec


def _on_import(fullname, callback):
    if fullname in sys.modules:  # already imported (should not happen)
        try:
            callback(sys.modules[fullname])
        except Exception as e:
            log(f"patch of {fullname} skipped: {e!r}")
        return
    sys.meta_path.insert(0, _PostImportHook(fullname, callback))


def _call_compatible(fn, self, args, kwargs):
    """Call fn(self, *args) but drop args a Qt signal may add (e.g. clicked's `checked`)."""
    try:
        inspect.signature(fn).bind(self, *args, **kwargs)
    except (TypeError, ValueError):
        return fn(self)
    return fn(self, *args, **kwargs)


# ---------------------------------------------------------------------------
# Launcher
# ---------------------------------------------------------------------------

CHECKBOX_LABEL = "Enable API / MCP"
# The checkbox state lives in this launcher process only: it starts unticked every time the launcher
# starts (the API is opt-in per session; vmf-cli launch starts VisoMaster with the API without it).
_LAUNCH = {"api": False}


def _set_api_ticked(checked: bool):
    _LAUNCH["api"] = bool(checked)
    log(f"API checkbox {'ticked' if checked else 'unticked'}")


def _apply_env_from_config():
    if _LAUNCH["api"]:
        port = read_config()["port"]
        os.environ["VMF_API"] = "1"
        os.environ["VMF_API_PORT"] = str(port)
        log(f"API enabled for this launch (127.0.0.1:{port})")
    else:
        os.environ.pop("VMF_API", None)
        os.environ.pop("VMF_API_PORT", None)


def _add_checkbox(window):
    from PySide6 import QtWidgets

    page = getattr(window, "page_home", None)
    layout = page.layout() if page is not None else None
    if layout is None:
        layout = window.layout()
    if layout is None:
        log("launcher: no layout found for the API checkbox")
        return
    # Rebuilt pages are fresh widgets, but never add twice to the same one.
    owner = layout.parentWidget() or window
    if owner.findChild(QtWidgets.QCheckBox, "vmfApiCheckBox") is not None:
        return

    cfg = read_config()
    box = QtWidgets.QCheckBox(CHECKBOX_LABEL)
    box.setObjectName("vmfApiCheckBox")
    box.setChecked(_LAUNCH["api"])  # a page rebuilt in the same launcher session keeps the tick
    box.setToolTip(
        f"Starts the local API (127.0.0.1:{cfg['port']}) when VisoMaster is launched.\n"
        "Unticked each time the launcher starts. Used by vmf-cli and the MCP server.\n"
        "Token in vmf_api\\runtime\\server.json."
    )
    box.toggled.connect(_set_api_ticked)

    # Just below the last action button of the home page; else at the end.
    insert_at = None
    for i in range(layout.count()):
        w = layout.itemAt(i).widget()
        if isinstance(w, QtWidgets.QPushButton):
            insert_at = i + 1
    if insert_at is not None and hasattr(layout, "insertWidget"):
        layout.insertWidget(insert_at, box)
    else:
        layout.addWidget(box)


def _patch_launcher(module):
    cls = getattr(module, "LauncherWindow", None)
    on_launch = getattr(cls, "on_launch", None) if cls is not None else None
    if cls is None or not callable(on_launch):
        log("launcher hook unavailable (LauncherWindow.on_launch not found): no API checkbox")
        return

    @functools.wraps(on_launch)
    def wrapped_on_launch(self, *args, **kwargs):
        try:
            _apply_env_from_config()
        except Exception as e:
            log(f"environment variables not set: {e!r}")
        return _call_compatible(on_launch, self, args, kwargs)

    # Exact (self) signature so Qt does not pass clicked's `checked` argument.
    def on_launch_entry(self):
        return wrapped_on_launch(self)

    functools.update_wrapper(on_launch_entry, on_launch)
    del on_launch_entry.__wrapped__  # keep Qt's view of the signature: (self)
    cls.on_launch = on_launch_entry

    def add_box_safely(self):
        try:
            _add_checkbox(self)
        except Exception as e:
            log(f"API checkbox not added: {e!r}")

    build_home = getattr(cls, "_build_home_page", None)
    if callable(build_home):

        @functools.wraps(build_home)
        def wrapped_build_home(self, *args, **kwargs):
            result = build_home(self, *args, **kwargs)
            add_box_safely(self)
            return result

        cls._build_home_page = wrapped_build_home
    else:
        init = cls.__init__

        @functools.wraps(init)
        def wrapped_init(self, *args, **kwargs):
            init(self, *args, **kwargs)
            add_box_safely(self)
            try:
                self.adjustSize()
            except Exception:
                pass

        cls.__init__ = wrapped_init
    log("launcher hook installed")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


def _patch_app(module):
    cls = getattr(module, "MainWindow", None)
    if cls is None:
        log("app hook unavailable (MainWindow not found): VisoMaster starts without the API")
        return
    init = cls.__init__

    @functools.wraps(init)
    def wrapped_init(self, *args, **kwargs):
        init(self, *args, **kwargs)
        try:
            from PySide6 import QtCore

            QtCore.QTimer.singleShot(0, lambda: _start_server(self))
        except Exception as e:
            log(f"API not started: {e!r}")

    cls.__init__ = wrapped_init

    close_event = getattr(cls, "closeEvent", None)
    if callable(close_event):

        @functools.wraps(close_event)
        def wrapped_close(self, event, *args, **kwargs):
            result = close_event(self, event, *args, **kwargs)
            try:
                if event.isAccepted():
                    _stop_server(self)
            except Exception as e:
                log(f"API shutdown: {e!r}")
            return result

        cls.closeEvent = wrapped_close
    log("app hook installed")


def _start_server(main_window):
    try:
        import server

        port = int(os.environ.get("VMF_API_PORT") or read_config()["port"])
        main_window._vmf_api_server = server.start(main_window, port)
    except Exception as e:
        log(f"API not started ({e!r}); VisoMaster runs normally")


def _stop_server(main_window):
    srv = getattr(main_window, "_vmf_api_server", None)
    if srv is not None:
        srv.stop()
        main_window._vmf_api_server = None


# ---------------------------------------------------------------------------


def install(kind, vmf_dir=None):
    if kind == "launcher":
        _on_import(LAUNCHER_MODULE, _patch_launcher)
    elif kind == "app":
        log("VMF_API=1: the API will start once the window is open")
        _on_import(APP_MODULE, _patch_app)
