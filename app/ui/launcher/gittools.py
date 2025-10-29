# gittools.py
# ---------------------------------------------------------------------------
# Git Utilities for VisoMaster Fusion Launcher
# ---------------------------------------------------------------------------
# Provides simple Git integration for update checks, commit info, backups,
# and rollback operations. All commands are executed using the portable
# Git binary bundled with the app to ensure consistent behavior across systems.
# ---------------------------------------------------------------------------

import os
import subprocess
from datetime import datetime
from pathlib import Path
from .core import PATHS


# ---------- Git Command Wrapper ----------

def run_git(args: list[str], capture: bool = False, check: bool = False):
    """Run a Git command in the app repo context using the portable git.exe."""
    exe = str(PATHS["GIT_EXE"])
    repo = str(PATHS["APP_DIR"])
    git_dir = os.path.join(repo, ".git")
    env = os.environ.copy()
    env["PATH"] = f"{Path(exe).parent}{os.pathsep}{env['PATH']}"
    cmd = [exe, f"--git-dir={git_dir}", f"--work-tree={repo}"] + args
    return subprocess.run(
        cmd, cwd=repo, env=env, text=True,
        capture_output=capture, check=check, shell=False
    )


# ---------- Commit Utilities ----------

def fetch_commit_list(n: int = 10):
    """Return a list of recent commits (hash, date, message)."""
    try:
        result = run_git(
            ["log", "--pretty=format:%h|%ad|%s", "--date=short", "-n", str(n)],
            capture=True
        )
        if result.returncode != 0:
            print(f"[Launcher] Git log failed: {result.stderr.strip()}")
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
    from .cfgtools import read_portable_cfg  # avoid circular import
    cfg = read_portable_cfg()
    curr = cfg.get("CURRENT_COMMIT")
    if curr and len(curr) >= 7:
        return curr[:7]
    r = run_git(["rev-parse", "HEAD"], capture=True)
    if r and r.returncode == 0:
        return r.stdout.strip()[:7]
    return None


# ---------- Change Detection ----------

def git_changed_files(tracked_only: bool = True) -> list[str]:
    """Return a list of modified or changed tracked files."""
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
                if p.exists() and repo_dir in p.parents:
                    zf.write(p, arcname=rel)
        print(f"[Launcher] Backup created: {zip_path.name}")
        return str(zip_path)
    except Exception as e:
        print(f"[Launcher] Backup failed: {e}")
        return None
