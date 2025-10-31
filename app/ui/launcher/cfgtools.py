# cfgtools.py
# ---------------------------------------------------------------------------
# Configuration Management for VisoMaster Fusion Launcher
# ---------------------------------------------------------------------------
# Handles reading/writing of portable.cfg — a lightweight key=value config file
# used to store runtime metadata such as:
#   CURRENT_COMMIT, LAST_UPDATED, LAUNCHER_ENABLED, etc.
# Unknown keys are preserved to avoid overwriting user-defined fields.
# ---------------------------------------------------------------------------

from datetime import datetime, timezone
from .core import PATHS
import sys


# ---------- Core File I/O ----------

def read_portable_cfg() -> dict:
    """Read portable.cfg as a simple key=value dict."""
    cfg = {}
    p = PATHS["PORTABLE_CFG"]
    if not p.exists():
        return cfg
    try:
        for line in p.read_text(encoding="utf-8").splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                cfg[k.strip()] = v.strip()
    except Exception as e:
        print(f"[Launcher] Error reading portable.cfg: {e}")
    return cfg


def write_portable_cfg(updated: dict) -> bool:
    """Merge-write portable.cfg, preserving unknown keys and their order. Only updates the provided key-value pairs in 'updated'."""
    p = PATHS["PORTABLE_CFG"]
    lines, kv = [], {}

    # Load existing structure (preserving it) or create defaults
    if p.exists():
        raw = p.read_text(encoding="utf-8").splitlines()
        lines = raw[:]
        for i, line in enumerate(lines):
            if "=" in line:
                k, v = line.split("=", 1)
                kv[k.strip()] = (i, v.strip())
    else:
        lines = ["LAUNCHER_ENABLED=1"]
        kv = {"LAUNCHER_ENABLED": (0, "1")}

    changed = False

    # Update only the keys related to the launcher
    for k, v in updated.items():
        v_str = str(v)
        if k in kv:
            idx, old_v = kv[k]
            if old_v != v_str:
                lines[idx] = f"{k}={v_str}"
                kv[k] = (idx, v_str)
                changed = True
        else:
            lines.append(f"{k}={v_str}")
            kv[k] = (len(lines) - 1, v_str)
            changed = True

    if not changed and p.exists():
        return False

    try:
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return True
    except Exception as e:
        print(f"[Launcher] Error writing portable.cfg: {e}")
        return False


# ---------- Launcher Settings ----------

def get_launcher_enabled_from_cfg() -> int:
    """Return 1 if launcher should run on startup (based on 'LAUNCHER_ENABLED' in portable.cfg), else return 0."""
    cfg = read_portable_cfg()
    v = cfg.get("LAUNCHER_ENABLED")
    return 1 if v is None else (1 if str(v).strip() in ("1", "true", "True", "yes", "on") else 0)


def set_launcher_enabled_to_cfg(value: int):
    """Enable or disable the launcher in portable.cfg."""
    value = 1 if value else 0
    if write_portable_cfg({"LAUNCHER_ENABLED": value}):
        print(f"[Launcher] Config updated: LAUNCHER_ENABLED={value}")


# ---------- Version Tracking ----------

def update_current_commit_in_cfg():
    """Fetch the current Git commit hash and save it to portable.cfg under 'CURRENT_COMMIT'."""
    from .gittools import run_git
    r = run_git(["rev-parse", "HEAD"], capture=True)
    if r and r.returncode == 0:
        commit = r.stdout.strip()
        write_portable_cfg({"CURRENT_COMMIT": commit})


def update_last_updated_in_cfg():
    """Save the current UTC timestamp (as 'LAST_UPDATED') to portable.cfg."""
    iso_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    if write_portable_cfg({"LAST_UPDATED": iso_utc}):
        print(f"[Launcher] Last updated: {iso_utc}")


# ---------- Formatting / Read Utilities ----------

def format_last_updated_local(iso_str: str) -> str:
    """Convert a UTC ISO timestamp string to a local time string for display."""
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00")).astimezone()
        return dt.strftime("%d %b %Y, %H:%M")
    except Exception:
        return "Invalid date format"


def read_version_info():
    """Return (CURRENT_COMMIT, formatted LAST_UPDATED) from portable.cfg."""
    cfg = read_portable_cfg()
    curr = cfg.get("CURRENT_COMMIT")
    last = cfg.get("LAST_UPDATED")
    nice_last = format_last_updated_local(last) if last else None
    return curr, nice_last
