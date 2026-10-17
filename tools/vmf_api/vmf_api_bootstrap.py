"""VisoMaster Fusion API bootstrap, loaded by vmf_api_hook.pth at interpreter start.

Copied into the portable Python's site-packages by vmf_api\\install.bat.
It is INERT for every Python process except two:
  * the launcher  : python -m app.ui.launcher  (or the launcher.py shim)
  * the app       : python <...>\\main.py  with VMF_API=1 in the environment
In those two cases it adds <BASE_DIR>\\vmf_api to sys.path and lets vmf_hooks
patch the right class when its module gets imported.

Must never raise: any error is printed once and ignored.
Only os/sys are used here (both already loaded by site).
"""

import os
import sys

_TAG = "[VMF-API]"


def _detect():
    """Return "launcher", "app" or None, from the raw command line.

    While .pth files run, sys.argv is not final yet (['-m'] for module mode),
    so sys.orig_argv is used.
    """
    argv = list(getattr(sys, "orig_argv", None) or [])
    if not argv:
        return None
    args = argv[1:]
    # python [-X utf8 ...] -m app.ui.launcher
    for i, a in enumerate(args):
        if a == "-m":
            if i + 1 < len(args) and args[i + 1] == "app.ui.launcher":
                return "launcher"
            return None
        if a == "-c":
            return None
    # First non-option argument = script path
    skip_next = False
    script = None
    for a in args:
        if skip_next:
            skip_next = False
            continue
        if a in ("-X", "-W", "-Q"):
            skip_next = True
            continue
        if a.startswith("-"):
            continue
        script = a
        break
    if not script:
        return None
    name = os.path.basename(script).lower()
    folder = os.path.dirname(os.path.abspath(script))
    if name == "launcher.py" and os.path.isfile(
        os.path.join(folder, "app", "ui", "launcher", "launcher_window.py")
    ):
        return "launcher"
    if name == "main.py" and os.path.isfile(os.path.join(folder, "app", "ui", "main_ui.py")):
        if os.environ.get("VMF_API") == "1":
            return "app"
    return None


def _app_dir():
    """The VisoMaster-Fusion checkout: the folder of the script (launcher.py / main.py), else the
    current folder (python -m app.ui.launcher is run from the checkout)."""
    args = list(getattr(sys, "orig_argv", None) or [])[1:]
    for a in args:
        if a.lower().endswith(".py") and os.path.isfile(a):
            return os.path.dirname(os.path.abspath(a))
    return os.getcwd()


def _vmf_dir(app_dir=None):
    """VMF_API_HOME, else <BASE_DIR>\\vmf_api next to the checkout (portable layout:
    <BASE_DIR>\\portable-files\\python\\python.exe), else <checkout>\\tools\\vmf_api."""
    override = os.environ.get("VMF_API_HOME")
    if override:
        return override
    exe = os.path.abspath(sys.executable or "")
    base = os.path.dirname(os.path.dirname(os.path.dirname(exe)))
    beside = os.path.join(base, "vmf_api")
    if app_dir and not os.path.isfile(os.path.join(beside, "vmf_hooks.py")):
        inside = os.path.join(app_dir, "tools", "vmf_api")
        if os.path.isfile(os.path.join(inside, "vmf_hooks.py")):
            return inside
    return beside


def _main():
    kind = _detect()
    if kind is None:
        return
    vmf_dir = _vmf_dir(_app_dir())
    if not os.path.isfile(os.path.join(vmf_dir, "vmf_hooks.py")):
        print(f"{_TAG} vmf_api folder not found ({vmf_dir}): API disabled.", flush=True)
        return
    if vmf_dir not in sys.path:
        sys.path.append(vmf_dir)
    import vmf_hooks  # noqa: E402  (from vmf_dir)

    vmf_hooks.install(kind, vmf_dir)


try:
    _main()
except Exception as _e:  # never break the interpreter
    try:
        print(f"{_TAG} bootstrap skipped after an error: {_e!r}", flush=True)
    except Exception:
        pass
