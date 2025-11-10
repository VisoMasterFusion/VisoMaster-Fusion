# cfgtools.py
# ---------------------------------------------------------------------------
# Configuration Management for VisoMaster Fusion Launcher
# ---------------------------------------------------------------------------
# Handles reading/writing of portable.cfg — a lightweight key=value config file
# used to store runtime metadata such as:
#   CURRENT_COMMIT, LAST_UPDATED, LAUNCHER_ENABLED, etc.
# Also manages dependency and model checksum state (DEPS_SHA, MODELS_SHA)
# for maintenance verification and repair operations.
#
# Developer Notes:
#   • Add new persistent keys via write_portable_cfg({"KEY": value})
#     — unknown keys are preserved automatically.
#   • Read existing values with read_portable_cfg() or helper functions.
# ---------------------------------------------------------------------------

import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

from .core import PATHS
from app.processors.models_data import models_list
from collections.abc import Sequence
from typing import Dict, Any


# ---------- Core File I/O ----------


def read_portable_cfg() -> dict[str, str | None]:
    """Read portable.cfg as a simple key=value dict."""
    cfg: dict[str, str | None] = {}
    p: Path = PATHS["PORTABLE_CFG"]
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


def write_portable_cfg(updated: dict[str, Any]) -> bool:
    """Merge-write portable.cfg, preserving unknown keys and their order.

    Only updates the provided key-value pairs in 'updated'.
    """
    p: Path = PATHS["PORTABLE_CFG"]
    lines: list[str] = []
    kv: dict[str, tuple[int, str]] = {}

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
        v_str: str = str(v)
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
    """Return 1 if launcher should run on startup (based on portable.cfg)."""
    cfg = read_portable_cfg()
    v = cfg.get("LAUNCHER_ENABLED")
    return (
        1
        if v is None
        else (1 if str(v).strip() in ("1", "true", "True", "yes", "on") else 0)
    )


def set_launcher_enabled_to_cfg(value: int) -> None:
    """Enable or disable the launcher in portable.cfg."""
    value = 1 if value else 0
    if write_portable_cfg({"LAUNCHER_ENABLED": value}):
        print(f"Launcher: Config updated: LAUNCHER_ENABLED={value}")


# ---------- Branch Management ----------
def get_branch_from_cfg() -> str:
    """Read the configured branch from portable.cfg, defaulting to 'main'."""
    cfg = read_portable_cfg()
    # Ensure there's always a fallback
    return cfg.get("BRANCH", "main").strip() or "main"


# ---------- Version Tracking ----------
def update_current_commit_in_cfg() -> None:
    """Fetch the current Git commit hash and save it to portable.cfg."""
    from .gittools import run_git  # local import to avoid circular deps

    r = run_git(["rev-parse", "HEAD"], capture=True)
    if r and r.returncode == 0:
        commit = r.stdout.strip()
        write_portable_cfg({"CURRENT_COMMIT": commit})


def update_last_updated_in_cfg() -> None:
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


def read_version_info() -> tuple[str | None, str | None]:
    """Return (CURRENT_COMMIT, formatted LAST_UPDATED) from portable.cfg."""
    cfg = read_portable_cfg()
    curr = cfg.get("CURRENT_COMMIT")
    last = cfg.get("LAST_UPDATED")
    nice_last = format_last_updated_local(last) if last else None
    return curr, nice_last


# ---------- Checksum Tracking ----------


def compute_file_sha256(path: Path) -> str | None:
    """Return SHA256 checksum of the given file, or None if not found."""
    try:
        if not path.exists():
            print(f"[Launcher] Warning: File missing for checksum: {path}")
            return None
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception as e:
        print(f"[Launcher] Error computing file checksum: {e}")
        return None


def compute_models_sha256(models_list: Sequence[dict[str, Any] | str]) -> str | None:
    """Return SHA256 of the serialized models list for consistency tracking."""
    try:
        # Normalize and sort models list for deterministic hashing
        normalized: list[Dict[str, Any] | str] = []
        for item in models_list:
            if isinstance(item, dict):
                normalized.append({k: item[k] for k in sorted(item.keys())})
        payload = json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        return hashlib.sha256(payload).hexdigest()
    except Exception as e:
        print(f"[Launcher] Error computing models checksum: {e}")
        return None


# ---------------------------------------------------------------------------
# Model Presence Check
# ---------------------------------------------------------------------------


def check_models_presence() -> tuple[bool, list[str]]:
    """Quickly check if any expected model files are missing (no hash check)."""
    missing: list[str] = []
    for model in models_list:
        local_path = model.get("local_path")
        if not local_path:
            continue
        model_path = Path(local_path)
        if not model_path.exists():
            missing.append(str(model_path))
            print(f"[Launcher] Warning: Missing model file → {model_path}")
    return bool(missing), missing


# ---------- Checksum State ----------


def read_checksum_state() -> dict[str, str | None]:
    """Read current checksum values from portable.cfg."""
    cfg = read_portable_cfg()
    return {
        "DEPS_SHA": cfg.get("DEPS_SHA"),
        "MODELS_SHA": cfg.get("MODELS_SHA"),
        "LAST_MAINT_TS": cfg.get("LAST_MAINT_TS"),
    }


def write_checksum_state(
    deps_sha: str | None = None, models_sha: str | None = None
) -> None:
    """Update checksum values in portable.cfg and record maintenance timestamp."""
    updated: dict[str, str | int] = {}
    if deps_sha:
        updated["DEPS_SHA"] = deps_sha
    if models_sha:
        updated["MODELS_SHA"] = models_sha
    if updated:
        iso_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        updated["LAST_MAINT_TS"] = iso_utc
        if write_portable_cfg(updated):
            print("[Launcher] Updated checksum state in config.")
