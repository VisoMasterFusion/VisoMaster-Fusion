"""Modal progress dialog shown while TensorRT engine caches are built.

Unlike the plain indeterminate ``QProgressDialog`` used before, it keeps the
user informed during builds that can take several minutes:

* the current build stage, parsed from the streamed probe log,
* elapsed time and, once the model has been built before, an estimate of the
  remaining time,
* a live tail of the build log behind an expandable "details" section,
* a Cancel button (with confirmation) that aborts the build.

A true percentage bar is not possible: the build runs inside ONNX Runtime's
TensorRT execution provider, which exposes no progress callback, so the bar
stays indeterminate by design.

All methods are only ever called in the GUI thread; ``ModelsProcessor`` talks
to the dialog purely through queued signals, so builds can be driven from any
worker thread.
"""

import time

from PySide6 import QtCore, QtGui, QtWidgets

from app.helpers.build_progress import format_duration


class TrtBuildDialog(QtWidgets.QDialog):
    """Rich, non-blocking progress dialog for TensorRT engine builds."""

    # Emitted when the user confirms cancellation; ModelsProcessor listens.
    cancel_requested = QtCore.Signal()

    _LOG_MAX_BLOCKS = 500

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Building TensorRT Cache")
        self.setWindowIcon(QtGui.QIcon(":/media/media/visomaster_small.png"))
        self.setWindowFlag(QtCore.Qt.WindowType.WindowCloseButtonHint, False)
        self.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        self.setMinimumWidth(560)

        self._started_monotonic = 0.0
        self._expected_seconds: float | None = None

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        self.model_label = QtWidgets.QLabel(self)
        model_font = self.model_label.font()
        model_font.setBold(True)
        self.model_label.setFont(model_font)
        self.model_label.setWordWrap(True)
        layout.addWidget(self.model_label)

        self.stage_label = QtWidgets.QLabel(self)
        self.stage_label.setWordWrap(True)
        layout.addWidget(self.stage_label)

        self.progress_bar = QtWidgets.QProgressBar(self)
        # Indeterminate (busy) mode: no real percentage is available.
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setTextVisible(False)
        layout.addWidget(self.progress_bar)

        info_row = QtWidgets.QHBoxLayout()
        self.time_label = QtWidgets.QLabel(self)
        info_row.addWidget(self.time_label, 1)
        self.count_label = QtWidgets.QLabel(self)
        self.count_label.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter
        )
        info_row.addWidget(self.count_label, 0)
        layout.addLayout(info_row)

        self.details_button = QtWidgets.QPushButton("Show details", self)
        self.details_button.setCheckable(True)
        self.details_button.setFlat(True)
        self.details_button.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        layout.addWidget(self.details_button, 0, QtCore.Qt.AlignmentFlag.AlignLeft)

        self.log_view = QtWidgets.QPlainTextEdit(self)
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(self._LOG_MAX_BLOCKS)
        self.log_view.setMinimumHeight(160)
        log_font = QtGui.QFont("Consolas")
        log_font.setStyleHint(QtGui.QFont.StyleHint.Monospace)
        self.log_view.setFont(log_font)
        self.log_view.setVisible(False)
        layout.addWidget(self.log_view)
        self.details_button.toggled.connect(self._on_details_toggled)

        button_row = QtWidgets.QHBoxLayout()
        button_row.addStretch(1)
        self.cancel_button = QtWidgets.QPushButton("Cancel", self)
        self.cancel_button.clicked.connect(self._confirm_cancel)
        button_row.addWidget(self.cancel_button)
        layout.addLayout(button_row)

        self._elapsed_timer = QtCore.QTimer(self)
        self._elapsed_timer.setInterval(500)
        self._elapsed_timer.timeout.connect(self._update_time_label)

    # -- API used by ModelsProcessor (GUI thread only) ------------------------

    def start_build(
        self,
        title: str,
        model_label: str,
        expected_seconds: float,
        build_number: int,
    ) -> None:
        """(Re)open the dialog for a new build.

        ``expected_seconds`` <= 0 hides the estimate. ``build_number`` <= 0
        means the work is not a cancellable engine build (e.g. a one-time
        shape-inference step), which hides the counter and disables Cancel.
        """
        self.setWindowTitle(title)
        self.model_label.setText(model_label)
        self._expected_seconds = expected_seconds if expected_seconds > 0 else None
        self._started_monotonic = time.monotonic()
        self.stage_label.setText("Starting the build worker...")
        self.log_view.clear()
        self.cancel_button.setEnabled(build_number > 0)
        if build_number > 0:
            self.count_label.setText(f"Engine build #{build_number} this session")
            self.count_label.setVisible(True)
        else:
            self.count_label.setVisible(False)
        self._update_time_label()
        self._elapsed_timer.start()
        if not self.isVisible():
            self.show()
        self.raise_()

    def append_log_line(self, line: str) -> None:
        """Append one streamed probe log line to the details view."""
        self.log_view.appendPlainText(line.rstrip())

    def set_stage(self, stage_label: str) -> None:
        """Update the human-readable build phase label."""
        self.stage_label.setText(stage_label)

    def finish(self) -> None:
        """Close the dialog after the build attempt ends."""
        self._elapsed_timer.stop()
        self.close()

    # -- internals ------------------------------------------------------------

    def _on_details_toggled(self, checked: bool) -> None:
        self.log_view.setVisible(checked)
        self.details_button.setText("Hide details" if checked else "Show details")
        self.adjustSize()

    def _update_time_label(self) -> None:
        elapsed = max(0.0, time.monotonic() - self._started_monotonic)
        elapsed_text = format_duration(elapsed)
        if self._expected_seconds:
            remaining = self._expected_seconds - elapsed
            if remaining > 0:
                text = (
                    f"Elapsed {elapsed_text} - roughly "
                    f"{format_duration(remaining)} remaining "
                    f"(usually takes ~{format_duration(self._expected_seconds)})"
                )
            else:
                text = (
                    f"Elapsed {elapsed_text} - taking longer than usual "
                    f"(~{format_duration(self._expected_seconds)} last time)"
                )
        else:
            text = f"Elapsed {elapsed_text}"
        self.time_label.setText(text)

    def _confirm_cancel(self) -> None:
        box = QtWidgets.QMessageBox(self)
        box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
        box.setWindowTitle("Cancel engine build")
        box.setText("Cancel the TensorRT engine build?")
        box.setInformativeText(
            "The model will not be loaded and its incomplete engine cache "
            "will be discarded. The build will start over next time."
        )
        box.setStandardButtons(
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.No
        )
        box.setDefaultButton(QtWidgets.QMessageBox.StandardButton.No)
        # Force on-top so the confirmation cannot hide behind the dialog.
        box.setWindowFlag(QtCore.Qt.WindowType.WindowStaysOnTopHint, True)
        if box.exec() == QtWidgets.QMessageBox.StandardButton.Yes:
            self.cancel_button.setEnabled(False)
            self.stage_label.setText("Cancelling - waiting for the build worker...")
            self.cancel_requested.emit()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        # The title-bar close button is disabled, but be explicit: closing the
        # window only ends the display; cancellation must go through Cancel.
        self._elapsed_timer.stop()
        super().closeEvent(event)
