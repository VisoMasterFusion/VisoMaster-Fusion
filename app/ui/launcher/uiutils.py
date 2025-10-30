# uiutils.py
# ---------------------------------------------------------------------------
# UI Utilities for VisoMaster Fusion Launcher
# ---------------------------------------------------------------------------
# Provides small reusable helpers for time display and user feedback:
#   • humanize_elapsed(dt_then, dt_now=None) → human-readable time diff ("5m ago")
#   • notify_backup_created(parent, zip_path) → message box with optional folder open
#   • make_divider(color="#363636") → subtle horizontal line divider
#   • make_header_widget(title_text, logo_path=None, logo_width=160) → styled header
# ---------------------------------------------------------------------------

from datetime import datetime, timezone
from PySide6 import QtWidgets, QtGui, QtCore
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


# ---------- UI Elements ----------

def make_divider(color: str = "#363636") -> QtWidgets.QFrame:
    """Return a thin horizontal line divider."""
    divider = QtWidgets.QFrame()
    divider.setFrameShape(QtWidgets.QFrame.HLine)
    divider.setStyleSheet(f"color: {color}; background-color: {color};")
    return divider


def make_header_widget(title_text: str, logo_path: str | None = None, logo_width: int = 160) -> QtWidgets.QWidget:
    """Return a reusable header section with optional logo and horizontal line divider."""
    container = QtWidgets.QWidget()
    v = QtWidgets.QVBoxLayout(container)
    v.setContentsMargins(10, 10, 10, 10)
    v.setSpacing(6)

    if logo_path and os.path.exists(logo_path):
        logo_lbl = QtWidgets.QLabel()
        pix = QtGui.QPixmap(logo_path)
        if not pix.isNull():
            scaled = pix.scaledToWidth(logo_width, QtCore.Qt.SmoothTransformation)
            logo_lbl.setPixmap(scaled)
            logo_lbl.setAlignment(QtCore.Qt.AlignCenter)
            v.addWidget(logo_lbl)

    title = QtWidgets.QLabel(title_text)
    f = QtGui.QFont("Segoe UI Semibold", 11)
    title.setFont(f)
    title.setAlignment(QtCore.Qt.AlignCenter)
    v.addWidget(title)

    line = make_divider()
    v.addWidget(line)

    return container
