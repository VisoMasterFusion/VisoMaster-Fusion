"""Paths, config and runtime file shared by the hooks, the server and the CLI.

Layout (<vmf_api> = <BASE_DIR>\\vmf_api, or VisoMaster-Fusion\\tools\\vmf_api: see _layout):
  <vmf_api>\\config.json        {"port": 8765}  (the launcher checkbox is not saved: unticked at each start)
  <vmf_api>\\runtime\\server.json {"port", "token", "pid", ...} while the app runs
  <vmf_api>\\backups\\           config and workspace safety copies
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

API_VERSION = "1.0.0"
DEFAULT_PORT = 8765
TAG = "[VMF-API]"

# VMF_API_HOME: same override as vmf_api_bootstrap (used by the tests).
VMF_DIR = Path(os.environ.get("VMF_API_HOME") or Path(__file__).resolve().parent).resolve()


def _layout(vmf_dir: Path) -> tuple[Path, Path]:
    """(BASE_DIR, APP_DIR) for the two supported places of this folder:
    inside the repository   <APP_DIR>\\tools\\vmf_api      (APP_DIR = the VisoMaster-Fusion checkout)
    next to it (portable)   <BASE_DIR>\\vmf_api, with <BASE_DIR>\\VisoMaster-Fusion and portable-files."""
    repo = vmf_dir.parent.parent
    if vmf_dir.parent.name.lower() == "tools" and (repo / "main.py").is_file() and (repo / "app").is_dir():
        return repo.parent, repo
    return vmf_dir.parent, vmf_dir.parent / "VisoMaster-Fusion"


BASE_DIR, APP_DIR = _layout(VMF_DIR)
CONFIG_PATH = VMF_DIR / "config.json"
RUNTIME_DIR = VMF_DIR / "runtime"
RUNTIME_PATH = RUNTIME_DIR / "server.json"
BACKUP_DIR = VMF_DIR / "backups"
EXPORT_DIR = VMF_DIR / "exports"
SNAPSHOT_DIR = VMF_DIR / "snapshots"
PERSONAL_KEYS_PATH = VMF_DIR / "personal_keys.json"


def log(msg: str) -> None:
    try:
        print(f"{TAG} {msg}", file=sys.stderr, flush=True)  # stdout is the CLI's JSON / the MCP stdio stream
    except Exception:
        pass


def write_json_atomic(path: Path, data, indent: int = 4) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def read_config() -> dict:
    cfg = {"port": DEFAULT_PORT}  # an old "enabled" key is ignored: the checkbox is not saved
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            port = int(data.get("port", DEFAULT_PORT))
            if 1024 <= port <= 65535:
                cfg["port"] = port
    except FileNotFoundError:
        pass
    except Exception as e:
        log(f"config.json unreadable ({e}), using defaults")
    return cfg


def write_config(port: int) -> dict:
    cfg = read_config()
    port = int(port)
    if not 1024 <= port <= 65535:
        raise ValueError(f"port {port} out of range 1024-65535")
    cfg["port"] = port
    write_json_atomic(CONFIG_PATH, cfg)
    return cfg


def read_runtime() -> dict | None:
    try:
        with open(RUNTIME_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (FileNotFoundError, ValueError):
        return None
