# gittools.py
# ---------------------------------------------------------------------------
# Git Utilities for VisoMaster Fusion Launcher
# ---------------------------------------------------------------------------
# Provides simple Git integration for update checks, commit info, backups,
# and rollback operations. All commands are executed using the portable
# Git binary bundled with the app to ensure consistent behavior across systems.
# Also manages dependency/model checksum state (DEPS_SHA, MODELS_SHA) for maintenance tracking.
# ---------------------------------------------------------------------------

import os
import subprocess
from datetime import datetime
from pathlib import Path
from .core import PATHS
import filecmp
from PySide6.QtWidgets import QApplication, QMessageBox


# ---------- Git Command Wrapper ----------


def run_git(args: list[str], capture: bool = False, check: bool = False):
    """Run a Git command in the app repo context using the portable git.exe."""
    exe = str(PATHS["GIT_EXE"])
    repo = str(PATHS["APP_DIR"])
    git_dir = os.path.join(repo, ".git")

    if not Path(exe).exists():
        print(f"[Launcher] ERROR: git.exe not found at {exe}")
        return None

    env = os.environ.copy()
    env["PATH"] = f"{Path(exe).parent}{os.pathsep}{env['PATH']}"
    cmd = [exe, f"--git-dir={git_dir}", f"--work-tree={repo}"] + args

    try:
        return subprocess.run(
            cmd,
            cwd=repo,
            env=env,
            text=True,
            capture_output=capture,
            check=check,
            shell=False,
        )
    except FileNotFoundError:
        print(f"[Launcher] ERROR: Git executable missing or inaccessible: {exe}")
    except subprocess.SubprocessError as e:
        print(f"[Launcher] ERROR running Git command: {e}")
    except Exception as e:
        print(f"[Launcher] Unexpected Git error: {e}")
    return None


# ---------- Commit Utilities ----------


def fetch_commit_list(n: int = 10):
    """Return a list of recent commits (hash, date, message)."""
    try:
        result = run_git(
            ["log", "--pretty=format:%h|%ad|%s", "--date=short", "-n", str(n)],
            capture=True,
        )
        if not result or result.returncode != 0:
            msg = result.stderr.strip() if result else "No response from Git."
            print(f"[Launcher] Git log failed: {msg}")
            return []

        commits = []
        for ln in result.stdout.strip().splitlines():
            parts = ln.split("|", 2)
            if len(parts) == 3:
                commits.append({"hash": parts[0], "date": parts[1], "msg": parts[2]})
        return commits
    except Exception as e:
        print(f"[Launcher] Error fetching commit list: {e}")
        return []


def get_current_short_commit() -> str | None:
    """Return the current 7-character commit hash."""
    try:
        from .cfgtools import read_portable_cfg  # avoid circular import

        cfg = read_portable_cfg()
        curr = cfg.get("CURRENT_COMMIT")
        if curr and len(curr) >= 7:
            return curr[:7]
        r = run_git(["rev-parse", "HEAD"], capture=True)
        if r and r.returncode == 0:
            return r.stdout.strip()[:7]
    except Exception as e:
        print(f"[Launcher] Error reading current commit: {e}")
    return None


# ---------- Change Detection ----------


def git_changed_files(tracked_only: bool = True) -> list[str]:
    """Return a list of modified or changed tracked files."""
    try:
        args = ["status", "--porcelain"]
        if tracked_only:
            args.append("--untracked-files=no")
        r = run_git(args, capture=True)
        if not r or r.returncode != 0:
            return []
        files = []
        for line in r.stdout.splitlines():
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                files.append(parts[1])
        return files
    except Exception as e:
        print(f"[Launcher] Error detecting changed files: {e}")
        return []


# ---------- Backup ----------


def backup_changed_files(changed: list[str]) -> str | None:
    """Create a zip backup of modified tracked files and return its path."""
    if not changed:
        return None
    import zipfile

    repo_dir = PATHS["APP_DIR"]
    backups_dir = PATHS["PORTABLE_DIR"] / "backups"
    backups_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    zip_path = backups_dir / f"git-changes-backup-{stamp}.zip"

    try:
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for rel in changed:
                p = (repo_dir / rel).resolve()
                if not p.exists():
                    print(f"[Launcher] Skipping missing file: {rel}")
                    continue
                if repo_dir not in p.parents:
                    print(f"[Launcher] Skipping file outside repo: {rel}")
                    continue
                zf.write(p, arcname=rel)
        print(f"[Launcher] Backup created: {zip_path.name}")
        return str(zip_path)
    except PermissionError:
        print("[Launcher] ERROR: Permission denied while creating backup.")
    except zipfile.BadZipFile:
        print("[Launcher] ERROR: Backup archive creation failed (corrupt zip).")
    except Exception as e:
        print(f"[Launcher] Backup failed: {e}")
    return None


# ---------- Self-Update Trigger ----------


def is_launcher_update_available() -> bool:
    """Check if the root Start_Portable.bat differs from the one in the repo."""
    root_bat = PATHS["BASE_DIR"] / "Start_Portable.bat"
    repo_bat = PATHS["APP_DIR"] / "Start_Portable.bat"

    if not root_bat.exists() or not repo_bat.exists():
        return False

    # filecmp.cmp returns True if files are identical, so we return the opposite.
    return not filecmp.cmp(str(root_bat), str(repo_bat), shallow=False)


def trigger_self_update_if_needed(parent_widget=None):
    """
    Creates an updater script and quits the application to allow the update.
    This is now only called when the user explicitly clicks the update button.
    """
    root_bat = PATHS["BASE_DIR"] / "Start_Portable.bat"
    repo_bat = PATHS["APP_DIR"] / "Start_Portable.bat"

    print("Launcher: Start_Portable.bat has been updated. Relaunching...")
    QMessageBox.information(
        parent_widget,
        "Launcher Update Required",
        "The launcher script (Start_Portable.bat) has been updated.\n\n"
        "The application will now close and restart itself to apply the changes.",
    )

    updater_script_content = f"""
    @echo off
    echo.
    echo Waiting for launcher to exit...
    timeout /t 3 /nobreak >nul
    echo Replacing Start_Portable.bat...
    copy /y "{repo_bat.resolve()}" "{root_bat.resolve()}"
    echo.
    echo Update complete. Restarting launcher...
    start "" /d "{root_bat.parent.resolve()}" "Start_Portable.bat"
    exit
    """
    updater_path = PATHS["PORTABLE_DIR"] / "update_start_portable.bat"
    try:
        updater_path.write_text(updater_script_content, encoding="utf-8")
    except Exception as e:
        print(f"Launcher: ERROR: Could not write updater script: {e}")
        QMessageBox.critical(
            parent_widget, "Update Error", f"Could not create the updater script:\n{e}"
        )
        return

    # Use DETACHED_PROCESS to ensure the updater runs independently
    subprocess.Popen(
        f'cmd /c "{updater_path.resolve()}"',
        creationflags=subprocess.DETACHED_PROCESS,
        close_fds=True,
        shell=False,
    )

    # Quit the current application
    QApplication.quit()
