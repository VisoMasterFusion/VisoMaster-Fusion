# core.py
# ---------------------------------------------------------------------------
# Core Utilities for VisoMaster Fusion Launcher
# ---------------------------------------------------------------------------
# Provides:
#   • Centralized filesystem path resolution via PATHS
#   • Basic runtime validation (must_exist)
#   • Theme application (True-Dark QSS)
#   • Portable subprocess helpers for Python and UV operations
# ---------------------------------------------------------------------------

from pathlib import Path
import sys
import subprocess
from PySide6 import QtWidgets


# ---------- Path Resolution ----------


def resolve_paths():
    """Return all filesystem paths used by the launcher."""
    script_path = Path(__file__).resolve()
    ui_dir = script_path.parent.parent  # .../app/ui
    app_dir = ui_dir.parent  # .../app
    repo_dir = app_dir.parent  # .../VisoMaster-Fusion
    base_dir = repo_dir.parent  # .../VisoMaster
    portable_dir = base_dir / "portable-files"

    return {
        "BASE_DIR": base_dir,
        "PORTABLE_DIR": portable_dir,
        "APP_DIR": repo_dir,
        "VENV_DIR": portable_dir / "venv",
        "PYTHON_EXE": portable_dir / "venv" / "Scripts" / "python.exe",
        "UV_EXE": portable_dir / "uv" / "uv.exe",
        "GIT_EXE": portable_dir / "git" / "bin" / "git.exe",
        "STYLES_DIR": app_dir / "ui" / "styles",
        "LOGO_PNG": app_dir / "ui" / "core" / "media" / "visomaster_logo.png",
        "SMALL_ICON": app_dir / "ui" / "core" / "media" / "visomaster_small.png",
        "REQ_FILE": repo_dir / "requirements_cu129.txt",
        "MAIN_PY": repo_dir / "main.py",
        "DOWNLOAD_PY": repo_dir / "download_models.py",
        "PORTABLE_CFG": base_dir / "portable.cfg",
    }


PATHS = resolve_paths()


# ---------- Validation ----------


def must_exist(p: Path, what: str):
    """Exit early with a clear message if a required path is missing."""
    if not Path(p).exists():
        print(f"[Launcher] ERROR: Missing {what}: {p}")
        sys.exit(1)


# ---------- Theme Handling ----------


def apply_theme_to_app(app: QtWidgets.QApplication):
    """Apply the True-Dark QSS theme to the launcher."""
    qss_path = PATHS["STYLES_DIR"] / "true_dark.qss"
    if not qss_path.exists():
        print(f"[Launcher] Warning: true_dark.qss not found in {PATHS['STYLES_DIR']}")
        return
    try:
        app.setStyleSheet(qss_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[Launcher] Error applying theme: {e}")


# ---------- Subprocess Helpers ----------


def run_python(script_path: Path):
    """Run a Python script using the portable Python interpreter."""
    subprocess.run(
        [str(PATHS["PYTHON_EXE"]), str(script_path)], cwd=PATHS["APP_DIR"], shell=False
    )


def uv_pip_install():
    """Run dependency installation using the portable uv executable."""
    subprocess.run(
        [
            str(PATHS["UV_EXE"]),
            "pip",
            "install",
            "-r",
            str(PATHS["REQ_FILE"]),
            "--python",
            str(PATHS["PYTHON_EXE"]),
        ],
        cwd=PATHS["APP_DIR"],
        shell=False,
    )
