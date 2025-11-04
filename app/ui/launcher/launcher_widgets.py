# launcher_widgets.py
# ---------------------------------------------------------------------------
# UI Widgets for VisoMaster Fusion Launcher
# ---------------------------------------------------------------------------
# Defines reusable visual elements and custom controls for the launcher.
# Keeps styling and interaction logic separate from layout construction.
#
# Developer Notes:
#   • Add new custom widgets here — keep each widget class self-contained
#     and focused on a single UI element or behavior.
# ---------------------------------------------------------------------------

from PySide6 import QtWidgets, QtGui, QtCore

# ---------- Toggle Switch ----------


class ToggleSwitch(QtWidgets.QPushButton):
    """A minimalist animated toggle switch styled for dark UI themes."""

    def __init__(
        self,
        checked: bool = False,
        bg_color: str = "#2b2b2b",
        active_color: str = "#4090a3",
        circle_color: str = "#f0f0f0",
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)
        self.setCheckable(True)
        self.setChecked(checked)
        self.setCursor(QtCore.Qt.PointingHandCursor)
        self.setFixedSize(36, 18)

        self._bg_color = bg_color
        self._active_color = active_color
        self._circle_color = circle_color
        self._circle_position = self.width() - self.height() if checked else 1

        # Animation setup
        self._animation = QtCore.QPropertyAnimation(self, b"circle_position", self)
        self._animation.setDuration(200)
        self._animation.setEasingCurve(QtCore.QEasingCurve.OutCubic)

    def _get_circle_position(self) -> float:
        return self._circle_position

    def _set_circle_position(self, pos: float) -> None:
        self._circle_position = pos
        self.update()

    # Property for animation
    circle_position = QtCore.Property(float, _get_circle_position, _set_circle_position)

    def paintEvent(self, event):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.setPen(QtCore.Qt.NoPen)

        rect = self.rect()
        radius = rect.height() / 2

        # Background color
        color = QtGui.QColor(self._active_color if self.isChecked() else self._bg_color)
        p.setBrush(color)
        p.drawRoundedRect(rect, radius, radius)

        # Circle
        circle_diameter = rect.height() - 4
        circle_x = self._circle_position
        p.setBrush(QtGui.QColor(self._circle_color))
        p.drawEllipse(circle_x, 2, circle_diameter, circle_diameter)
        p.end()

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        self.start_animation()

    def start_animation(self):
        """Animate the circle sliding left (off) or right (on)."""
        start = self._circle_position
        end = self.width() - self.height() if self.isChecked() else 1
        self._animation.stop()
        self._animation.setStartValue(start)
        self._animation.setEndValue(end)
        self._animation.start()


# ---------- Status Pill Label ----------


class StatusPill(QtWidgets.QLabel):
    """A small non-interactive pill-style label for status indicators."""

    def __init__(self, text: str, color: str = "rgba(255,255,255,0.08)", parent=None):
        super().__init__(text, parent)
        self.setAlignment(QtCore.Qt.AlignCenter)

        font = QtGui.QFont("Segoe UI", 11)
        self.setFont(font)

        # Basic pill styling
        self.setStyleSheet(f"""
            QLabel {{
                padding: 4px 8px;
                color: rgba(255,255,255,0.75);
                background-color: {color};
                border: 1px solid rgba(255,255,255,0.10);
                border-radius: 6px;
            }}
        """)
