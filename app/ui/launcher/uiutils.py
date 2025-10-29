# uiutils.py
# ---------------------------------------------------------------------------
# UI Utilities for VisoMaster Fusion Launcher
# ---------------------------------------------------------------------------
# Lightweight helpers for time display and user notifications.
# ---------------------------------------------------------------------------

from datetime import datetime, timezone
from PySide6 import QtWidgets
import subprocess
import os


# ---------- Time Formatting ----------

def humanize_elapsed(dt_then: datetime, dt_now: datetime | None = None) -> str:
    """Return a short human-readable time difference string (e.g. '5m ago')."""
    if dt_now is None:
        dt_now = datetime.now(timezone.utc)
    delta = int((dt_now - dt_then).total_seconds())

    if delta < 10:
        return "just now"
    if delta < 60:
        return f"{delta}s ago"

    mins = delta // 60
    if mins < 60:
        return f"{mins}m ago"

    hours = mins // 60
    if hours < 24:
        return f"{hours}h ago"

    days = hours // 24
    return f"{days}d ago"


# ---------- Notifications ----------

def notify_backup_created(parent: QtWidgets.QWidget, zip_path: str):
    """Show a message box when a backup zip is created, with option to open folder."""
    m = QtWidgets.QMessageBox(parent)
    m.setIcon(QtWidgets.QMessageBox.Information)
    m.setWindowTitle("Backup created")
    m.setText("A safety backup of modified app files was created.")
    m.setInformativeText("You can restore files manually from the backup if needed.")
    m.setStandardButtons(QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Open)
    m.button(QtWidgets.QMessageBox.Ok).setText("OK")
    m.button(QtWidgets.QMessageBox.Open).setText("Open folder")

    res = m.exec()
    if res == QtWidgets.QMessageBox.Open:
        try:
            subprocess.Popen(["explorer", "/select,", os.fspath(zip_path)])
        except Exception as e:
            print(f"[Launcher] Failed to open backup folder: {e}")
