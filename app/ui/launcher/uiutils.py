# ---------------------------------------------------------------------------
# UI Utilities for VisoMaster Fusion Launcher
# ---------------------------------------------------------------------------
# Provides shared UI helpers:
#   • make_header_widget() / make_divider() for layout styling
#   • notify_backup_created() for user messages
#   • with_busy_state() context manager for temporary UI locking
# ---------------------------------------------------------------------------

from PySide6 import QtWidgets, QtGui, QtCore
import subprocess
import os
from contextlib import contextmanager


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


def make_header_widget(
    title_text: str, logo_path: str | None = None, logo_width: int = 160
) -> QtWidgets.QWidget:
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


# ---------- Busy State Management (Context Manager) ----------


@contextmanager
def with_busy_state(widget: QtWidgets.QWidget, busy: bool, text: str | None = None):
    """Context manager for setting a busy state in the UI."""
    # Disable buttons and show loading text during long operations
    for button in widget.findChildren(QtWidgets.QPushButton):
        button.setEnabled(not busy)

    # Set the window title to indicate the busy state
    if busy and text:
        widget.setWindowTitle(text)
    else:
        widget.setWindowTitle("VisoMaster Fusion Launcher")

    # Process events and make sure the UI is updated
    QtWidgets.QApplication.processEvents()

    try:
        yield
    finally:
        # Re-enable buttons and reset window title after operation
        for button in widget.findChildren(QtWidgets.QPushButton):
            button.setEnabled(True)

        widget.setWindowTitle("VisoMaster Fusion Launcher")
        QtWidgets.QApplication.processEvents()
