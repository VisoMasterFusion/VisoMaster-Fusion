# launcher_window.py
# ---------------------------------------------------------------------------
# Main GUI Logic for VisoMaster Fusion Launcher
# ---------------------------------------------------------------------------
# Handles:
#   • Home, Maintenance, and Rollback pages
#   • Git update detection and refresh indicators
#   • Dependency and model checksum tracking (portable.cfg: DEPS_SHA, MODELS_SHA)
#   • StatusPill-based UI for component states (⚠️ / ✅)
#   • Window centering, dynamic resizing, and drag behavior
#
# Adding new actions/buttons:
#   1. Define a new entry in ACTIONS_HOME or ACTIONS_MAINT as:
#        ("Button Label", "method_name", "Tooltip text")
#   2. Implement the corresponding method (e.g., def on_new_action(self): ...)
#   3. The button will appear automatically on the respective page.
# ---------------------------------------------------------------------------

import sys
from datetime import datetime, timezone
from PySide6 import QtWidgets, QtGui, QtCore
from PySide6.QtWidgets import QMessageBox

from .core import PATHS, run_python, uv_pip_install
from app.processors.models_data import models_list
from .gittools import (
    run_git,
    fetch_commit_list,
    git_changed_files,
    backup_changed_files,
    get_current_short_commit,
    is_launcher_update_available,
    trigger_self_update_if_needed,
)
from .cfgtools import (
    read_portable_cfg,
    get_launcher_enabled_from_cfg,
    set_launcher_enabled_to_cfg,
    update_current_commit_in_cfg,
    update_last_updated_in_cfg,
    read_version_info,
    format_last_updated_local,
    read_checksum_state,
    write_checksum_state,
    compute_file_sha256,
    compute_models_sha256,
    check_models_presence,
    write_portable_cfg,
    get_branch_from_cfg,
)
from .uiutils import (
    notify_backup_created,
    make_header_widget,
    make_divider,
    with_busy_state,
)
from .launcher_widgets import ToggleSwitch, StatusPill


# ---------- Page Actions ----------
ACTIONS_HOME = [
    ("Launch VisoMaster Fusion", "on_launch", "Start normally"),
    ("Update / Maintenance", "_go_maint", "Open maintenance tools"),
    ("Quit", "close", "Exit launcher"),
]

ACTIONS_MAINT = [
    ("Update from Git", "on_update_git", "Fetch and apply updates from origin"),
    ("Repair Installation", "on_repair_installation", "Restore tracked files to HEAD"),
    ("Check / Update Dependencies", "on_update_deps", "Reinstall requirements via UV"),
    ("Check / Update Models", "on_update_models", "Run model downloader"),
    ("Revert to Previous Version", "_go_rollback", "Select and revert to older commit"),
    ("Back", "_go_home", "Return to home screen"),
]


# ---------- Update Check ----------


# ---------- Update Check ----------
def check_update_status():
    """Return 'behind', 'up_to_date', or 'offline' after a quick origin fetch."""
    branch = get_branch_from_cfg()
    print(f"Launcher: Checking for updates on branch '{branch}'...")
    try:
        r = run_git(["fetch", "origin", branch], capture=True)
        if r is None or r.returncode != 0:
            err = r.stderr.strip() if r and getattr(r, "stderr", None) else "unknown"
            print(f"Launcher: Git fetch failed (offline?): {err}")
            return "offline"
    except Exception as e:
        print(f"Launcher: Fetch exception (offline?): {e}")
        return "offline"

    head = run_git(["rev-parse", "HEAD"], capture=True)
    origin = run_git(["rev-parse", f"origin/{branch}"], capture=True)

    if head and origin and head.returncode == 0 and origin.returncode == 0:
        head_hash, origin_hash = head.stdout.strip(), origin.stdout.strip()
        if head_hash != origin_hash:
            print(
                f"Launcher: Local version behind origin (HEAD={head_hash[:7]} -> {origin_hash[:7]})"
            )
            return "behind"
        print(f"Launcher: Repository up to date (HEAD={head_hash[:7]})")
        return "up_to_date"

    print(f"Launcher: Unable to read HEAD or origin/{branch}; treating as offline.")
    return "offline"


# ---------------------------------------------------------------------------


class LauncherWindow(QtWidgets.QWidget):
    """Small, portable maintenance/launch UI for VisoMaster Fusion."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("VisoMaster Fusion Launcher")
        if PATHS["SMALL_ICON"].exists():
            self.setWindowIcon(QtGui.QIcon(str(PATHS["SMALL_ICON"])))

        # Safety defaults
        self.deps_changed = self.models_changed = self.files_changed = False
        self._user_moved = False
        self.last_checked_utc: datetime | None = None

        # Check for launcher script update
        self.launcher_update_available = is_launcher_update_available()
        self.update_status = self._check_and_log_update_status()

        update_current_commit_in_cfg()
        self.commits = fetch_commit_list(10)

        # --- Checksum state ---
        self._load_checksum_status()

        self._build_ui()
        self._resize_to_current_page()
        self._center_on_screen()
        print("[Launcher] Launcher ready.")

    # ---------- Helpers ----------

    def _register_action(self, label, callback, tooltip=None, icon=None):
        """Create a standardized launcher button."""
        btn = QtWidgets.QPushButton(label)
        btn.setMinimumHeight(34)
        if tooltip:
            btn.setToolTip(tooltip)

        # Highlight maintenance button if any issue is detected
        if "Update / Maintenance" in label:
            if (
                self.update_status == "behind"
                or self.launcher_update_available
                or getattr(self, "deps_changed", False)
                or getattr(self, "models_changed", False)
                or getattr(self, "files_changed", False)
            ):
                icon = self.style().standardIcon(QtWidgets.QStyle.SP_MessageBoxWarning)

        if "Update Launcher Script" in label and self.launcher_update_available:
            icon = self.style().standardIcon(QtWidgets.QStyle.SP_MessageBoxWarning)
        if "Update from Git" in label and self.update_status == "behind":
            icon = self.style().standardIcon(QtWidgets.QStyle.SP_MessageBoxWarning)
        if "Dependencies" in label and getattr(self, "deps_changed", False):
            icon = self.style().standardIcon(QtWidgets.QStyle.SP_MessageBoxWarning)
        if "Models" in label and getattr(self, "models_changed", False):
            icon = self.style().standardIcon(QtWidgets.QStyle.SP_MessageBoxWarning)
        if "Repair Installation" in label and getattr(self, "files_changed", False):
            icon = self.style().standardIcon(QtWidgets.QStyle.SP_MessageBoxWarning)

        if icon:
            btn.setIcon(icon)
            btn.setIconSize(QtCore.QSize(22, 22))

        btn.clicked.connect(callback)
        return btn

    def _go_home(self):
        self._navigate_to("page_home")

    def _go_maint(self):
        self._navigate_to("page_maint")

    def _go_rollback(self):
        self._navigate_to("page_rollback")

    # ---------- Navigation ----------

    def _navigate_to(self, page_name: str):
        page = getattr(self, page_name, None)
        if page and self.stack.indexOf(page) != -1:
            self.stack.setCurrentWidget(page)
            self._resize_to_current_page()
        else:
            print(
                f"[Launcher] Warning: Cannot navigate to '{page_name}' (not in stack)."
            )

    def _reset_page(self, page_widget: QtWidgets.QWidget):
        layout = page_widget.layout()
        if layout is not None:
            QtWidgets.QWidget().setLayout(layout)

    # ---------- Checksum Handling ----------

    def _load_checksum_status(self):
        """Compute and compare checksums for dependencies and models."""
        try:
            pass
        except Exception as e:
            print(f"[Launcher] Warning: Could not import models_list: {e}")

        # --- Dependency Check ---
        state = read_checksum_state()
        deps_now = compute_file_sha256(PATHS["REQ_FILE"])

        # --- Auto-initialize missing checksum keys on first run ---
        if not state.get("DEPS_SHA") and deps_now:
            print("[Launcher] Initializing missing DEPS_SHA in portable.cfg...")
            write_checksum_state(deps_sha=deps_now)

        if not state.get("MODELS_SHA") and models_list:
            print("[Launcher] Initializing missing MODELS_SHA in portable.cfg...")
            write_checksum_state(models_sha=compute_models_sha256(models_list))

        # Re-read state to update comparisons after writing new keys
        state = read_checksum_state()

        # --- Dependency Comparison ---
        self.deps_changed = deps_now != state.get("DEPS_SHA")
        if self.deps_changed:
            print("[Launcher] Warning: Dependencies checksum mismatch.")

        # --- Model Presence Check ---
        self.models_changed, self.missing_models = check_models_presence()
        if self.models_changed:
            print(f"[Launcher] Warning: {len(self.missing_models)} model(s) missing.")

        # --- Tracked Files Check ---
        try:
            changed_files = git_changed_files(True)
            self.files_changed = bool(changed_files)
            if self.files_changed:
                print("[Launcher] Warning: Tracked files changed.")
        except Exception as e:
            print(f"[Launcher] Warning: Could not check tracked files: {e}")
            self.files_changed = False

    # ---------- Meta Panel ----------

    def _build_meta_panel(self, curr_short, last_nice):
        panel = QtWidgets.QFrame()
        panel.setObjectName("MetaPanel")
        layout = QtWidgets.QGridLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setHorizontalSpacing(12)
        layout.setVerticalSpacing(4)

        def klabel(text):
            lbl = QtWidgets.QLabel(text)
            lbl.setStyleSheet(
                "color: rgba(255,255,255,0.75); font-size: 13px; font-weight: 600;"
            )
            return lbl

        def vlabel(text, mono=False, tooltip=None):
            lbl = QtWidgets.QLabel(text)
            if mono:
                fam = "Consolas, 'Cascadia Mono', 'Fira Code', monospace"
                lbl.setStyleSheet(
                    f"color: rgba(255,255,255,0.92); font-size: 13px; font-family: {fam};"
                )
            else:
                lbl.setStyleSheet("color: rgba(255,255,255,0.85); font-size: 13px;")
            if tooltip:
                lbl.setToolTip(tooltip)
            return lbl

        if curr_short:
            layout.addWidget(klabel("Current build"), 0, 0)
            layout.addWidget(
                vlabel(
                    curr_short, mono=True, tooltip="Full commit hash in portable.cfg"
                ),
                0,
                1,
            )
        if last_nice:
            layout.addWidget(klabel("Last updated"), 0, 2)
            layout.addWidget(vlabel(last_nice, tooltip="UTC ISO in portable.cfg"), 0, 3)

        panel.setStyleSheet(
            """
            QFrame#MetaPanel {
                background-color: rgba(255,255,255,0.05);
                border: 1px solid rgba(255,255,255,0.08);
                border-radius: 8px;
            }
        """
        )
        return panel

    # ---------- Build UI ----------

    def _build_ui(self):
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        self.stack = QtWidgets.QStackedWidget()
        root.addWidget(self.stack)

        self.page_home = QtWidgets.QWidget()
        self.page_maint = QtWidgets.QWidget()
        self.page_rollback = QtWidgets.QWidget()

        self._build_home_page()
        self._build_maint_page()
        self._build_rollback_page()

        self.stack.addWidget(self.page_home)
        self.stack.addWidget(self.page_maint)
        self.stack.addWidget(self.page_rollback)
        self.stack.setCurrentWidget(self.page_home)

    def _build_home_page(self):
        """Create the home screen with launcher options and status info."""
        self._reset_page(self.page_home)
        lay = QtWidgets.QVBoxLayout(self.page_home)
        lay.addWidget(
            make_header_widget("VisoMaster Fusion Launcher", str(PATHS["LOGO_PNG"]))
        )

        for text, method, *tip in ACTIONS_HOME:
            fn = getattr(self, method)
            lay.addWidget(self._register_action(text, fn, tip[0] if tip else None))

        # --- Status Pills Section ---
        lay.addSpacing(10)
        lay.addWidget(make_divider())
        lay.addSpacing(10)

        if self.launcher_update_available:
            lay.addWidget(StatusPill("⚠️ Launcher script update available"))

        if self.update_status == "behind":
            lay.addWidget(StatusPill("⚠️ Git updates available"))

        elif self.update_status == "offline":
            lay.addWidget(
                StatusPill(
                    "Offline — can’t check for updates", color="rgba(255,255,255,0.08)"
                )
            )

        if self.deps_changed or self.models_changed or self.files_changed:
            if self.deps_changed:
                lay.addWidget(StatusPill("⚠️ Dependencies changed"))
            if self.models_changed:
                lay.addWidget(StatusPill("⚠️ Models changed"))
            if self.files_changed:
                lay.addWidget(StatusPill("⚠️ Tracked files changed"))
        elif self.update_status != "behind":
            lay.addWidget(StatusPill("✅ All components up to date"))

        curr, last = read_version_info()
        curr_short = curr[:7] if curr else None
        if curr_short or last:
            meta = self._build_meta_panel(curr_short, last)
            meta.setContentsMargins(0, 0, 0, 0)
            lay.addWidget(meta)

        lay.addStretch(1)
        lay.addWidget(make_divider())

        footer = QtWidgets.QHBoxLayout()
        footer.setContentsMargins(6, 6, 10, 10)
        footer.addStretch(1)

        lbl_toggle = QtWidgets.QLabel("Use launcher on startup")
        lbl_toggle.setFont(QtGui.QFont("Segoe UI Semibold", 10))
        lbl_toggle.setStyleSheet("color: #f0f0f0; margin-right: 6px;")

        is_enabled = bool(get_launcher_enabled_from_cfg())
        self.launcher_toggle = ToggleSwitch(checked=is_enabled)
        self.launcher_toggle.toggled.connect(self._on_launcher_toggle_changed)
        lbl_toggle.mousePressEvent = lambda _e: self.launcher_toggle.click()

        footer.addWidget(lbl_toggle)
        footer.addWidget(self.launcher_toggle)
        lay.addLayout(footer)
        footer.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignBottom)

    def _build_maint_page(self):
        self._reset_page(self.page_maint)
        lay = QtWidgets.QVBoxLayout(self.page_maint)
        lay.addWidget(make_header_widget("VisoMaster Fusion — Maintenance"))

        # --- Branch Switcher ---
        branch_box = QtWidgets.QFrame()
        branch_box.setObjectName("BranchBox")
        branch_box.setStyleSheet(
            "QFrame#BranchBox { border: 1px solid #363636; border-radius: 6px; }"
        )
        branch_layout = QtWidgets.QHBoxLayout(branch_box)
        branch_layout.setContentsMargins(10, 5, 10, 5)

        branch_label = QtWidgets.QLabel("Active Branch:")
        branch_label.setStyleSheet("border: none; background: transparent;")

        self.branch_combo = QtWidgets.QComboBox()
        self.branch_combo.addItems(["main", "dev"])

        # Set the current branch from config without triggering the signal
        self.branch_combo.blockSignals(True)
        self.branch_combo.setCurrentText(get_branch_from_cfg())
        self.branch_combo.blockSignals(False)

        self.branch_combo.currentTextChanged.connect(self.on_switch_branch)

        branch_layout.addWidget(branch_label)
        branch_layout.addStretch()
        branch_layout.addWidget(self.branch_combo)
        lay.addWidget(branch_box)
        lay.addWidget(make_divider())

        for text, method, *tip in ACTIONS_MAINT:
            fn = getattr(self, method)
            lay.addWidget(self._register_action(text, fn, tip[0] if tip else None))

        cfg = read_portable_cfg()
        last = cfg.get("LAST_UPDATED")
        nice_last = format_last_updated_local(last) if last else "—"
        self.lbl_last_updated = QtWidgets.QLabel(f"Last updated: {nice_last}")
        self.lbl_last_updated.setAlignment(QtCore.Qt.AlignRight)
        self.lbl_last_updated.setStyleSheet(
            "color: rgba(255,255,255,0.58); font-size: 12px; padding: 2px;"
        )
        lay.addWidget(self.lbl_last_updated)
        lay.addStretch(1)

    def _build_rollback_page(self):
        self._reset_page(self.page_rollback)
        lay = QtWidgets.QVBoxLayout(self.page_rollback)
        lay.addWidget(make_header_widget("Revert to Previous Version"))
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(inner)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(6)
        current_short = get_current_short_commit()

        if not self.commits:
            v.addWidget(QtWidgets.QLabel("No commits found (or git unavailable)."))
        else:
            for c in self.commits:
                is_current = current_short and c["hash"] == current_short
                card = QtWidgets.QFrame()
                card_lay = QtWidgets.QVBoxLayout(card)
                card_lay.setContentsMargins(8, 6, 8, 6)

                title_html = f"<b>{c['hash']}</b>  —  <i>{c['date']}</i>"
                hash_label = QtWidgets.QLabel(title_html)
                msg_label = QtWidgets.QLabel(c["msg"])
                msg_label.setWordWrap(True)

                card_lay.addWidget(hash_label)
                card_lay.addWidget(msg_label)

                if is_current:
                    pill = StatusPill("This is the current version")
                    card_lay.addWidget(pill)
                else:
                    btn = QtWidgets.QPushButton("Revert to this version")
                    btn.setFixedHeight(26)
                    btn.setFocusPolicy(QtCore.Qt.NoFocus)
                    btn.clicked.connect(lambda _, h=c["hash"]: self.on_rollback(h))
                    card_lay.addWidget(btn)

                v.addWidget(card)

        v.addStretch(1)
        inner.setLayout(v)
        scroll.setWidget(inner)
        lay.addWidget(scroll)
        lay.addWidget(self._register_action("Back", self._go_maint))

    # ---------- Actions ----------

    def on_self_update(self):
        """Action to trigger the self-update mechanism."""
        trigger_self_update_if_needed(self)

    def on_launch(self):
        self.hide()
        QtWidgets.QApplication.processEvents()
        print("[Launcher] Launching VisoMaster Fusion...")
        import subprocess

        subprocess.run(
            [str(PATHS["PYTHON_EXE"]), str(PATHS["MAIN_PY"])],
            cwd=PATHS["APP_DIR"],
            shell=False,
        )
        QtWidgets.QApplication.quit()

    def on_switch_branch(self, new_branch: str):
        """Action to switch the git branch."""
        current_branch = get_branch_from_cfg()
        if new_branch == current_branch:
            return

        confirm = QtWidgets.QMessageBox.question(
            self,
            "Confirm Branch Switch",
            f"You are about to switch from '{current_branch}' to '{new_branch}'.\n\n"
            "This will discard any local changes and synchronize with the new branch.\n"
            "Are you sure you want to continue?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
        )

        if confirm != QtWidgets.QMessageBox.Yes:
            # Revert the combobox selection if the user cancels
            self.branch_combo.blockSignals(True)
            self.branch_combo.setCurrentText(current_branch)
            self.branch_combo.blockSignals(False)
            return

        print(f"Launcher: Switching branch to '{new_branch}'...")
        with with_busy_state(self, busy=True, text=f"Switching to {new_branch}..."):
            try:
                write_portable_cfg({"BRANCH": new_branch})
                run_git(["checkout", new_branch])
                run_git(["reset", "--hard", f"origin/{new_branch}"])

                # After switching, update config and force dependency check
                update_current_commit_in_cfg()
                update_last_updated_in_cfg()

                # Invalidate checksums to prompt for an update
                write_checksum_state(deps_sha="changed", models_sha="changed")

                self._refresh_update_indicators()

                QtWidgets.QMessageBox.information(
                    self,
                    "Branch Switched",
                    f"Successfully switched to the '{new_branch}' branch.\n\n"
                    "It is highly recommended to run 'Check / Update Dependencies' now.",
                )

            except Exception as e:
                print(f"Launcher: Error during branch switch: {e}")
                QtWidgets.QMessageBox.critical(
                    self, "Error", f"Failed to switch branch:\n{e}"
                )
                # Revert config if it failed
                write_portable_cfg({"BRANCH": current_branch})
                self._refresh_update_indicators()

    def on_update_git(self):
        print("Launcher: Checking for updates (git fetch/reset)...")
        with with_busy_state(self, busy=True, text="Updating..."):
            try:
                branch = get_branch_from_cfg()
                run_git(["fetch", "origin", branch])
                run_git(["reset", "--hard", f"origin/{branch}"])
                update_current_commit_in_cfg()
                update_last_updated_in_cfg()
                self.commits = fetch_commit_list(10)
                self._rebuild_page("page_rollback", self._build_rollback_page)
                self._refresh_update_indicators()
                print("Launcher: Update complete.")
                # After updating, check if the launcher script itself needs an update
                if is_launcher_update_available():
                    self._refresh_update_indicators()
                    QMessageBox.information(
                        self,
                        "Launcher Update Available",
                        "The launcher script (Start_Portable.bat) has an update.\nPlease close the launcher and run Start_Portable.bat again to apply it.",
                    )
            except Exception as e:
                print(f"Launcher: Error during update: {e}")

    def on_repair_installation(self):
        confirm = QtWidgets.QMessageBox.question(
            self,
            "Repair installation",
            "Restore official app files for this version?\n\n"
            "This will overwrite modified tracked files with the current version (HEAD).\n"
            "Your personal files are not touched.",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
        )
        if confirm != QtWidgets.QMessageBox.Yes:
            return

        changed = git_changed_files(True)
        if not changed:
            QtWidgets.QMessageBox.information(self, "Repair", "No files to restore.")
            print("[Launcher] Repair skipped — no changed files.")
            return

        print(f"[Launcher] Repairing installation ({len(changed)} file(s))...")
        with with_busy_state(self, busy=True, text="Repairing..."):
            try:
                backup_path = backup_changed_files(changed)
                if backup_path:
                    notify_backup_created(self, backup_path)
                # A more forceful repair: reset to HEAD
                print("[Launcher] Forcefully resetting all tracked files to HEAD...")
                run_git(["reset", "--hard", "HEAD"])
                update_current_commit_in_cfg()
                update_last_updated_in_cfg()
                write_checksum_state(
                    deps_sha=compute_file_sha256(PATHS["REQ_FILE"]),
                    models_sha=compute_models_sha256(models_list),
                )
                self._load_checksum_status()
                self._refresh_update_indicators()
                if is_launcher_update_available():
                    self._refresh_update_indicators()
                    QMessageBox.information(
                        self,
                        "Launcher Update",
                        "An update for the launcher script may be available after repair.",
                    )
                print("Launcher: Repair complete.")
            except Exception as e:
                print(f"[Launcher] Error during repair: {e}")

    def on_update_deps(self):
        print("[Launcher] Updating/checking dependencies via uv...")
        with with_busy_state(self, busy=True, text="Updating dependencies..."):
            uv_pip_install()
            write_checksum_state(deps_sha=compute_file_sha256(PATHS["REQ_FILE"]))
            self._load_checksum_status()
            self._refresh_update_indicators()
            print("[Launcher] Dependencies update complete.")

    def on_update_models(self):
        print("[Launcher] Running model downloader...")
        with with_busy_state(self, busy=True, text="Updating models..."):
            run_python(PATHS["DOWNLOAD_PY"])
            try:
                write_checksum_state(models_sha=compute_models_sha256(models_list))
            except Exception as e:
                print(f"[Launcher] Warning: Could not update model checksum: {e}")
            self._load_checksum_status()
            self._refresh_update_indicators()
            print("[Launcher] Model update complete.")

    def on_rollback(self, commit_hash: str):
        confirm = QtWidgets.QMessageBox.question(
            self,
            "Confirm Revert",
            f"Revert to commit {commit_hash}?\n\nThis will discard all local changes.",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
        )
        if confirm != QtWidgets.QMessageBox.Yes:
            return

        changed = git_changed_files(True)
        if changed:
            print(
                f"[Launcher] Backing up {len(changed)} changed file(s) before revert..."
            )
            backup_path = backup_changed_files(changed)
            if backup_path:
                notify_backup_created(self, backup_path)

        print(f"[Launcher] Reverting to commit {commit_hash[:7]}...")
        with with_busy_state(self, busy=True, text="Reverting..."):
            try:
                run_git(["reset", "--hard", commit_hash])
                update_current_commit_in_cfg()
                update_last_updated_in_cfg()
                print("[Launcher] Revert complete.")
                self.commits = fetch_commit_list(10)
                self._rebuild_page("page_rollback", self._build_rollback_page)
                self._refresh_update_indicators()
                if is_launcher_update_available():
                    self._refresh_update_indicators()
                    QMessageBox.information(
                        self,
                        "Launcher Update",
                        "An update for the launcher script may be available after reverting.",
                    )
            except Exception as e:
                print(f"[Launcher] Error during revert: {e}")

    # ---------- Utility (Rebuild & State) ----------

    def _rebuild_page(self, attr_name: str, builder_fn):
        current_before = self.stack.currentWidget()
        old_page = getattr(self, attr_name, None)
        was_current = current_before is old_page
        idx = self.stack.indexOf(old_page) if old_page else -1

        if old_page:
            self.stack.removeWidget(old_page)
            old_page.deleteLater()

        new_page = QtWidgets.QWidget()
        setattr(self, attr_name, new_page)
        builder_fn()
        if idx != -1:
            self.stack.insertWidget(idx, new_page)
        else:
            self.stack.addWidget(new_page)

        QtCore.QTimer.singleShot(
            0, lambda: self._safe_restore_page(current_before, new_page, was_current)
        )
        QtCore.QTimer.singleShot(50, self._resize_to_current_page)

    def _safe_restore_page(self, old_widget, new_widget, was_current):
        if was_current:
            self.stack.setCurrentWidget(new_widget)
        elif self.stack.indexOf(old_widget) != -1:
            self.stack.setCurrentWidget(old_widget)

    def _resize_to_current_page(self):
        """Resize dynamically; rollback opens mid-height but caps exactly at full list height."""
        current = self.stack.currentWidget()
        if not current:
            return

        BASE_WIDTH = 420
        MIN_HEIGHT = 320
        MAX_HEIGHT = 1200  # safety upper bound

        if current == getattr(self, "page_rollback", None):
            scroll = current.findChild(QtWidgets.QScrollArea)

            def finalize_resize():
                # After layout settles, measure the true content height
                if scroll and scroll.widget():
                    inner = scroll.widget()
                    inner.adjustSize()
                    content_height = (
                        inner.height() + scroll.horizontalScrollBar().height() + 80
                    )
                else:
                    content_height = current.sizeHint().height() + 40

                # Cap and compute starting height
                max_height = min(content_height, MAX_HEIGHT)
                initial_height = max(MIN_HEIGHT, min(max_height // 2 + 200, max_height))

                # Apply resize and vertical limits
                self.setMinimumSize(BASE_WIDTH, MIN_HEIGHT)
                self.setMaximumSize(BASE_WIDTH, max_height)
                self.resize(BASE_WIDTH, initial_height)
                self.setSizePolicy(
                    QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Expanding
                )

            QtCore.QTimer.singleShot(0, finalize_resize)

        else:
            hint = current.sizeHint()
            fixed_height = max(MIN_HEIGHT, hint.height() + 40)
            self.setFixedSize(BASE_WIDTH, fixed_height)

        if not self._user_moved:
            self._center_on_screen()

    def moveEvent(self, e):
        self._user_moved = True
        super().moveEvent(e)

    def _center_on_screen(self):
        """Center the window on the primary screen."""
        screen = QtWidgets.QApplication.primaryScreen()
        if not screen:
            return
        geo = screen.availableGeometry()
        x = (geo.width() - self.width()) // 2
        y = (geo.height() - self.height()) // 2
        self.move(geo.x() + x, geo.y() + y)

    def _check_and_log_update_status(self) -> str:
        """Check and log git update status with timestamp."""
        status = check_update_status()
        self.last_checked_utc = datetime.now(timezone.utc)
        print(f"[Launcher] Update status: {status} (checked just now)")
        sys.stdout.flush()
        return status

    def _refresh_update_indicators(self):
        """Recheck update status and rebuild visible pages accordingly."""
        self.update_status = self._check_and_log_update_status()
        self.launcher_update_available = is_launcher_update_available()
        self._load_checksum_status()

        current = self.stack.currentWidget()
        current_name = None
        if current == getattr(self, "page_home", None):
            current_name = "page_home"
        elif current == getattr(self, "page_maint", None):
            current_name = "page_maint"
        elif current == getattr(self, "page_rollback", None):
            current_name = "page_rollback"

        self._rebuild_page("page_home", self._build_home_page)
        if current_name == "page_maint":
            self._rebuild_page("page_maint", self._build_maint_page)
        elif current_name == "page_rollback":
            self._rebuild_page("page_rollback", self._build_rollback_page)

    def _on_launcher_toggle_changed(self, checked: bool):
        """Save toggle preference to config."""
        set_launcher_enabled_to_cfg(1 if checked else 0)
