"""vmf-cli: drive VisoMaster Fusion through its local API (and MCP server).

    python vmf_cli.py status
    python vmf_cli.py diff their_preset.json
    python vmf_cli.py apply-config their_preset.json --faces all
    python vmf_cli.py snapshot preview.png
    python vmf_cli.py apply-offline their_preset.json        (VisoMaster closed)
    python vmf_cli.py load-input-faces D:\\faces --wait      (returns once every face is loaded)
    python vmf_cli.py batch-input-faces                      (then: recording-wait)
    python vmf_cli.py mcp                                    (stdio MCP server)

Exit codes: 0 done, 1 refused / error, 2 bad command line, 3 confirmation code needed,
4 a wait gave up before the end (max wait reached, the work goes on).

Port + token read from <vmf_api>\\runtime\\server.json (vmf_api beside VisoMaster-Fusion, or in its tools folder).
Stdlib only, except the `mcp` subcommand (`mcp` SDK, venv vmf_api\\.venv).
"""

import argparse
import base64
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from vmf_common import (  # noqa: E402
    APP_DIR,
    BACKUP_DIR,
    PERSONAL_KEYS_PATH,
    RUNTIME_PATH,
    read_config,
    read_runtime,
    write_config,
)


class ApiError(Exception):
    pass


# ---------------------------------------------------------------------------
# HTTP client
# ---------------------------------------------------------------------------


# Never through a proxy: HTTP_PROXY or a Windows system proxy would otherwise receive the token and
# every argument (127.0.0.1 is not in a "<local>" bypass list), and could answer with made-up results.
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def call(command: str, args: dict | None = None, timeout: float = 700.0):
    rt = read_runtime()
    if not rt:
        raise ApiError(
            "VisoMaster is not running with the API (no runtime/server.json). "
            "Tick \"Enable API / MCP\" in the launcher, then launch VisoMaster."
        )
    if rt.get("pid") and _pid_gone(rt["pid"]):
        # Left over by a VisoMaster that crashed: another program may now listen on that port and would
        # receive the token and the arguments.
        raise ApiError(
            "VisoMaster is not running with the API (runtime/server.json is left over from a closed session). "
            "Tick \"Enable API / MCP\" in the launcher, then launch VisoMaster."
        )
    url = f"http://127.0.0.1:{int(rt['port'])}/api/{command}"
    body = json.dumps({k: v for k, v in (args or {}).items() if v is not None}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {rt['token']}"},
    )
    try:
        with _OPENER.open(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            payload = json.loads(e.read().decode("utf-8"))
        except Exception:
            raise ApiError(f"HTTP {e.code}") from None
    except (urllib.error.URLError, ConnectionError, socket.timeout) as e:
        raise ApiError(
            f"API unreachable on 127.0.0.1:{rt['port']} ({getattr(e, 'reason', e)}). "
            "Is VisoMaster running? (runtime/server.json may be left over from a closed session)"
        ) from None
    if not payload.get("ok"):
        raise ApiError(payload.get("error") or "unknown error")
    return payload.get("result")


def poll(command: str, args: dict | None = None, tries: int = 3):
    """call() for the wait loops: one slow or dropped status request (VisoMaster busy for a moment,
    e.g. while finalizing a video) is retried instead of ending the whole wait."""
    for i in range(tries):
        try:
            return call(command, args, timeout=60)
        except ApiError as e:
            msg = str(e)
            transient = (msg.startswith("API unreachable") or "did not process the command in time" in msg
                         or "still running in VisoMaster after the timeout" in msg)
            if i == tries - 1 or not transient:
                raise
            time.sleep(2)


def _abs(p: str | None) -> str | None:
    if not p:
        return None
    # Never resolved: merely checking a \\host\share path makes Windows send the user's credentials to the host.
    if str(p).replace("/", "\\").startswith("\\\\"):
        raise ApiError(f"refused: network or device path ({p}): use a local file or folder")
    return str(Path(p).expanduser().resolve())


def _trt_progress_line(st: dict) -> str:
    items = st.get("items") or []
    cur = next((i for i in items if i.get("status") == "running"), None)
    if "progress" not in st and not cur:
        return st.get("message") or "no build"
    line = f"[{st.get('progress', '?')}]"
    if cur:
        line += f" {cur['model']} running for {cur.get('elapsed_s', 0):.0f} s"
        if cur.get("attempt"):
            line += f" (attempt {cur['attempt']})"
        if cur.get("timeouts"):
            line += f", {cur['timeouts']} timeout(s)"
    c = st.get("counts") or {}
    if c.get("failed"):
        line += f" - {c['failed']} failed"
    return line


def wait_tensorrt(max_wait: float = 3600, interval: float = 5, on_progress=None) -> dict:
    """Block until the TensorRT build finishes (or max_wait), polling the API."""
    deadline = time.time() + max_wait
    while True:
        st = poll("tensorrt_build_status")
        if on_progress:
            on_progress(st)
        if not st.get("running"):
            st["wait"] = "finished"
            return st
        if time.time() >= deadline:
            st["wait"] = f"still running after waiting {max_wait:.0f} s (the build goes on)"
            return st
        time.sleep(interval)


def _rec_progress_line(rec: dict) -> str:
    if not rec or not rec.get("id"):
        return "no recording"
    batch = rec.get("kind") == "batch"
    res = rec.get("result")
    if res:
        if batch:
            line = f"{res['state']}: {res.get('processed')} processed, {res.get('failed')} failed"
            if res.get("processing_time_s") is not None:
                line += f" in {res['processing_time_s']} s ({res.get('average_fps')} per s)"
            return line + f" -> {res.get('output_path')}"
        if res.get("exists"):
            return f"{res['state']}: {res.get('output_path')} ({res.get('size_bytes') or 0} bytes)"
        return f"{res['state']}: {res.get('hint', 'no output file')}"
    p = rec.get("progress") or {}
    done, total = p.get("frames_written", 0), p.get("total_frames") or "?"
    line = f"batch {done}/{total} {p.get('unit') or 'items'}" if batch else f"recording {done}/{total} frames"
    if p.get("percent") is not None:
        line += f" ({p['percent']}%)"
    if p.get("processing_fps"):
        line += f", {p['processing_fps']} per s" if batch else f", {p['processing_fps']} fps"
    if p.get("eta_s") is not None:
        line += f", about {p['eta_s']} s left"
    if p.get("phase"):
        line += f" ({p['phase']})"
    return line


def _ready_line(st: dict) -> str:
    if st.get("ready"):
        return "ready"
    return "waiting: " + "; ".join(st.get("waiting_for") or [])


def wait_ready(max_wait: float = 1800, quiet: float = 2.0, interval: float = 0.5, on_progress=None) -> dict:
    """Block until VisoMaster is idle (no model loading, no preview frame) for `quiet` seconds."""
    deadline = time.time() + max_wait
    since = None
    while True:
        st = poll("ready_status")
        if on_progress:
            on_progress(st)
        now = time.time()
        since = (since or now) if st.get("ready") else None
        if since is not None and now - since >= quiet:
            st["wait"] = "ready"
            return st
        if now >= deadline:
            st["wait"] = f"still not ready after {max_wait:.0f} s"
            return st
        time.sleep(interval)


_LAST_READY = {"line": None}


def _print_ready(st: dict):
    line = _ready_line(st)
    if line != _LAST_READY["line"]:
        print(time.strftime("%H:%M:%S"), line, file=sys.stderr, flush=True)
        _LAST_READY["line"] = line


def start_recording_when_ready(from_start: bool = False, on_progress=None) -> dict:
    """Wait for VisoMaster to be idle, then start the recording (retried once after a seek)."""
    # What waiting would not end by itself: say so at once instead of waiting up to 30 min.
    s = call("status")
    if s.get("recording"):
        raise ApiError("a recording is already running: wait_recording, or stop_recording")
    if (s.get("batch_info") or {}).get("active"):
        raise ApiError("a batch is running: wait_recording, or stop_recording to cancel it")
    if s.get("scan_active") or s.get("job_queue_running"):
        raise ApiError("a scan or the job queue is running: wait for it to end")
    if s.get("processing"):
        raise ApiError("playback in progress: stop it first (run_action buttonMediaPlay)")
    for _ in range(3):
        st = wait_ready(on_progress=on_progress)
        if not st.get("ready"):
            raise ApiError("VisoMaster is not ready: " + _ready_line(st))
        try:
            res = call("start_recording", {"from_start": from_start})
        except ApiError as e:
            if str(e).startswith("not ready: "):
                continue
            raise
        loads = st.get("last_model_loads")
        if loads:
            res["models_loaded_before_start"] = loads
        return res
    raise ApiError("VisoMaster kept being busy: try again")


def call_when_ready(command: str, args: dict, on_progress=None) -> dict:
    """call(), and when VisoMaster answers "not ready: ..." (models loading, preview frame, media list
    still loading), wait until it is idle and try again: what a person would do."""
    for _ in range(3):
        try:
            return call(command, args)
        except ApiError as e:
            if not str(e).startswith("not ready: "):
                raise
        st = wait_ready(on_progress=on_progress)
        if not st.get("ready"):
            raise ApiError("VisoMaster is not ready: " + _ready_line(st))
    raise ApiError("VisoMaster kept being busy: try again")


def wait_recording(max_wait: float = 3 * 3600, interval: float = 5, on_progress=None) -> dict:
    """Block until the recording ends and its file is written (or max_wait), polling the API."""
    deadline = time.time() + max_wait
    while True:
        rec = poll("recording_status")
        if on_progress:
            on_progress(rec)
        if not rec.get("active"):
            rec["wait"] = "finished" if rec.get("result") else "no recording in progress"
            return rec
        if time.time() >= deadline:
            rec["wait"] = f"still recording after waiting {max_wait:.0f} s (the recording goes on)"
            return rec
        time.sleep(interval)


# ---------------------------------------------------------------------------
# Offline mode
# ---------------------------------------------------------------------------


def _port_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", int(port)), timeout=0.5):
            return True
    except OSError:
        return False


def visomaster_running() -> tuple[bool | None, str]:
    """(running?, detail). None = could not tell."""
    rt = read_runtime()
    if rt and _port_open(rt.get("port", 0)):
        return True, f"API open on port {rt['port']} (pid {rt.get('pid')})"
    main_py = str(APP_DIR / "main.py").lower()
    try:
        ps = (
            "Get-CimInstance Win32_Process -Filter \"Name like 'python%'\" | "
            "ForEach-Object { \"$($_.ProcessId)`t$($_.ExecutablePath)`t$($_.CommandLine)\" }"
        )
        # Full path: a bare "powershell" is looked up in the current folder first, where a planted
        # powershell.exe would run instead.
        powershell = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe"
        out = subprocess.run(
            [str(powershell), "-NoProfile", "-NonInteractive", "-Command", ps],
            capture_output=True, text=True, timeout=30,
        ).stdout
    except Exception as e:
        return None, f"cannot inspect processes: {e}"
    portable_py = str(PORTABLE_DIR / "python" / "python.exe").lower()
    for line in out.splitlines():
        pid, _, rest = line.partition("\t")
        exe, _, cmdline = rest.partition("\t")
        low = cmdline.lower().replace("/", "\\")
        runs_main = low.rstrip('" ').endswith("main.py")
        # Start_Portable.bat runs "<portable python>" "main.py" from VisoMaster's folder: a relative path.
        if main_py in low or (runs_main and (str(APP_DIR).lower() in low or exe.lower() == portable_py)):
            return True, f"VisoMaster process found (pid {pid})"
    return False, ""


# ---------------------------------------------------------------------------
# Launch / quit
# ---------------------------------------------------------------------------

PORTABLE_DIR = APP_DIR.parent / "portable-files"
LAUNCH_LOG = RUNTIME_PATH.parent / "visomaster.log"


def _pid_alive(pid) -> bool:
    try:
        import ctypes

        k = ctypes.windll.kernel32
        h = k.OpenProcess(0x1000, False, int(pid))  # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        code = ctypes.c_ulong()
        k.GetExitCodeProcess(h, ctypes.byref(code))
        k.CloseHandle(h)
        return code.value == 259  # STILL_ACTIVE
    except Exception:
        return False


def _pid_gone(pid) -> bool:
    """True only when Windows says this process does not exist or has exited (False when it cannot tell)."""
    try:
        import ctypes

        k = ctypes.WinDLL("kernel32", use_last_error=True)
        h = k.OpenProcess(0x1000, False, int(pid))  # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return ctypes.get_last_error() == 87  # ERROR_INVALID_PARAMETER: no such process
        code = ctypes.c_ulong()
        ok = k.GetExitCodeProcess(h, ctypes.byref(code))
        k.CloseHandle(h)
        return bool(ok) and code.value != 259  # STILL_ACTIVE
    except Exception:
        return False


def _python_exe() -> Path:
    override = os.environ.get("VMF_PYTHON")
    py = Path(override) if override else PORTABLE_DIR / "python" / "python.exe"
    if not py.is_file():
        raise ApiError(f"VisoMaster's Python not found: {py} (set VMF_PYTHON to its python.exe)")
    return py


def _launch_env(hidden: bool, port: int, extra: dict | None = None) -> dict:
    env = dict(os.environ)
    for k in ("PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV", "QT_QPA_PLATFORM"):
        env.pop(k, None)  # the MCP server runs in its own venv: never leak it into VisoMaster
    env.update(VMF_API="1", VMF_API_PORT=str(port), PYTHONUTF8="1", PYTHONUNBUFFERED="1")  # live log
    ffmpeg = sorted(PORTABLE_DIR.glob("ffmpeg-*/bin"))
    if ffmpeg:
        env["PATH"] = str(ffmpeg[-1]) + os.pathsep + env.get("PATH", "")
    if hidden:
        env["QT_QPA_PLATFORM"] = "offscreen"  # Qt renders off screen: no window at all
    bad = sorted(str(k) for k in (extra or {}) if not _env_allowed(str(k)))
    if bad:
        raise ApiError(f"environment variables not allowed: {bad} (only VISOMASTER_* and VMF_* switches, "
                       f"except {sorted(_ENV_RESERVED)}: others could make VisoMaster load code from elsewhere)")
    for k, v in (extra or {}).items():
        env[str(k)] = str(v)  # e.g. VisoMaster's own switches (VISOMASTER_CUDA_SPIN_SYNC=1)
    return env


# launch(env=...): only feature switches. PYTHONPATH, PATH, VMF_API_HOME... would let the caller make
# VisoMaster's Python import or load code from any folder.
_ENV_RESERVED = {"VMF_API", "VMF_API_PORT", "VMF_API_HOME", "VMF_PYTHON"}


def _env_allowed(name: str) -> bool:
    n = name.upper()
    return n.startswith(("VISOMASTER_", "VMF_")) and n not in _ENV_RESERVED


def launch(hidden: bool = False, port: int | None = None, max_wait: float = 300, on_progress=None,
           env: dict | None = None) -> dict:
    """Start VisoMaster with the API (hidden=True: without any window), wait until the API answers
    and VisoMaster is idle. env: extra environment variables for VisoMaster's process."""
    running, detail = visomaster_running()
    if running:
        rt = read_runtime() or {}
        if _port_open(rt.get("port", 0)):
            if env:
                raise ApiError("VisoMaster is already running, so env would not apply: quit_visomaster first, "
                               "then launch again with env")
            return {"already_running": True, "pid": rt.get("pid"), "port": rt.get("port"),
                    "note": "already running, launched earlier (hidden or not as it was then): quit_visomaster "
                            "first to relaunch it differently"}
        raise ApiError(f"VisoMaster is already running without the API ({detail}): close it first")
    port = int(port or read_config()["port"])
    if _port_open(port):  # VisoMaster would start WITHOUT its API (port taken) and stay running unseen
        raise ApiError(f"port {port} is already used by another program: free it, or choose another port "
                       f"(vmf-cli api --port N)")
    _launch_env(hidden, port, env)  # refuses a disallowed env before anything is started
    if hidden and not (APP_DIR / "last_workspace.json").is_file():
        # First start: VisoMaster asks for the execution provider in a dialog before its window opens.
        raise ApiError("first start of this VisoMaster (no last_workspace.json): it asks for the execution "
                       "provider in a dialog, which stays invisible without a window. Launch it once with its "
                       "window (without hidden), choose the provider, quit, then launch it hidden")
    LAUNCH_LOG.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(LAUNCH_LOG, "w", encoding="utf-8", errors="replace") as logf:
        proc = subprocess.Popen(
            [str(_python_exe()), "-X", "utf8", str(APP_DIR / "main.py")],
            cwd=str(APP_DIR), env=_launch_env(hidden, port, env), stdin=subprocess.DEVNULL, stdout=logf,
            stderr=subprocess.STDOUT,
            # CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP: a hidden console, inherited by VisoMaster's
            # ffmpeg / ffprobe children. Detached (no console), each child opened its own console window
            # and ran throttled: recordings took ~19 s instead of ~6.8 s, with a 6 s finalization.
            creationflags=0x08000000 | 0x00000200,
            close_fds=True,
        )
    while True:
        if proc.poll() is not None:
            tail = LAUNCH_LOG.read_text(encoding="utf-8", errors="replace")[-1500:]
            raise ApiError(f"VisoMaster exited during startup (code {proc.returncode}); end of {LAUNCH_LOG}:\n{tail}")
        rt = read_runtime() or {}
        if rt.get("pid") == proc.pid and _port_open(rt.get("port", 0)):
            break
        if time.time() - t0 > max_wait:
            if hidden:
                # Never leave a VisoMaster nobody can reach (hidden = no window to close it from).
                proc.kill()
                raise ApiError(f"the API did not answer within {max_wait:.0f} s: VisoMaster (pid {proc.pid}) "
                               f"was stopped (a dialog waiting for an answer before the window opens?); "
                               f"see {LAUNCH_LOG}")
            # With a window, a dialog may be waiting for the user (first start: execution provider).
            raise ApiError(f"the API did not answer within {max_wait:.0f} s: VisoMaster (pid {proc.pid}) is left "
                           f"running, a dialog may be waiting for an answer on screen; see {LAUNCH_LOG}")
        if on_progress:
            on_progress({"waiting_for": ["VisoMaster starting"]})
        time.sleep(1)
    ready = wait_ready(max_wait=max(30.0, max_wait - (time.time() - t0)), on_progress=on_progress)
    return {
        "launched": True, "pid": proc.pid, "port": port, "hidden": hidden,
        "startup_s": round(time.time() - t0, 1), "ready": ready.get("ready"), "log": str(LAUNCH_LOG),
        **({"env": {str(k): str(v) for k, v in env.items()}} if env else {}),
    }


def quit_visomaster(save: bool = True, max_wait: float = 120) -> dict:
    """Close VisoMaster and wait until its process is gone."""
    rt = read_runtime() or {}
    res = call("quit", {"save": save})
    pid, t0 = rt.get("pid"), time.time()
    while pid and _pid_alive(pid) and time.time() - t0 < max_wait:
        time.sleep(0.5)
    res["closed"] = not (pid and _pid_alive(pid))
    res["after_s"] = round(time.time() - t0, 1)
    return res


_quit_and_wait = quit_visomaster  # the MCP tool of the same name shadows it inside run_mcp


def apply_offline(path, workspace=None, faces="selected", no_control=False, no_params=False,
                  source_face=None, missing="keep", dry_run=False, force=False) -> dict:
    import merge
    import schema as schema_mod

    ws_path = Path(workspace) if workspace else APP_DIR / "last_workspace.json"
    if not ws_path.is_file():
        raise ApiError(f"workspace not found: {ws_path}")
    if ws_path.suffix.lower() != ".json":
        raise ApiError(f"the workspace must be a .json file: {ws_path}")
    if not dry_run:
        running, detail = visomaster_running()
        if running:
            raise ApiError(f"VisoMaster is running ({detail}): it would overwrite {ws_path.name} on exit. Close VisoMaster or use apply-config.")
        if running is None and not force:
            raise ApiError(f"{detail}; use --force if VisoMaster is really closed")
    schema = schema_mod.from_source(APP_DIR)
    rules = merge.PersonalRules.load(PERSONAL_KEYS_PATH)
    imported = merge.parse_config(path, schema, source_face=source_face)
    with open(ws_path, encoding="utf-8") as f:
        ws = json.load(f)
    if not isinstance(ws, dict) or not isinstance(ws.get("control"), dict):
        raise ApiError(f"not a VisoMaster workspace (no \"control\" settings): {ws_path}")
    new, report = merge.merge_into_workspace(ws, imported, schema, rules, faces, not no_control, not no_params, missing)
    report["workspace"] = str(ws_path)
    if dry_run:
        report["dry_run"] = True
        return report
    # Kept out of the repo folder: vmf_api\backups\last_workspace.json.bak-<date>
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    backup = BACKUP_DIR / (ws_path.name + ".bak-" + time.strftime("%Y%m%d-%H%M%S"))
    backup.write_bytes(ws_path.read_bytes())
    tmp = ws_path.with_name(ws_path.name + ".vmf-tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(new, f, indent=4)
    os.replace(tmp, ws_path)
    report["backup"] = str(backup)
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_value(text: str):
    """`set` value: sent as typed. The server types it from the schema (coerce_strings): "1" for a toggle,
    "2.50" for a selection and "true" for a text stay what the setting expects (JSON parsing broke them)."""
    return text


def _add_import_opts(p):
    p.add_argument("path")
    p.add_argument("--faces", default="selected", help="all | selected | <face_id> (default: selected)")
    p.add_argument("--no-control", action="store_true", help="do not touch global settings")
    p.add_argument("--no-params", action="store_true", help="do not touch per-face parameters")
    p.add_argument("--source-face", help="multi-face workspace: face to take the parameters from")
    p.add_argument("--missing", choices=["keep", "default"], default="keep",
                   help="keys missing from the file: keep mine (default) or reset to default")


def build_parser():
    ap = argparse.ArgumentParser(prog="vmf-cli", description="Drive VisoMaster Fusion (local API).")
    ap.add_argument("--compact", action="store_true", help="one-line JSON")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status")
    sub.add_parser("help", help="commands exposed by the API")
    p = sub.add_parser("get-config"); p.add_argument("--face-id"); p.add_argument("--keys", nargs="*")
    p = sub.add_parser("export-config"); p.add_argument("path", nargs="?"); p.add_argument("--face-id")
    p.add_argument("--include-personal", action="store_true")
    p = sub.add_parser("diff"); _add_import_opts(p)
    p = sub.add_parser("apply-config"); _add_import_opts(p); p.add_argument("--dry-run", action="store_true")
    p = sub.add_parser("press-parameter-button", help="click a button attached to a setting (e.g. Apply)")
    p.add_argument("key"); p.add_argument("--label")
    p.add_argument("--confirm", help="code returned by a first call (reset buttons)")
    sub.add_parser("list-backups")
    p = sub.add_parser("save-backup", help="save the current config under a name"); p.add_argument("name")
    p = sub.add_parser("restore-backup"); p.add_argument("name", nargs="?")
    p = sub.add_parser("set"); p.add_argument("key"); p.add_argument("value"); p.add_argument("--face-id")
    p = sub.add_parser("schema"); p.add_argument("key", nargs="?"); p.add_argument("--group")
    p = sub.add_parser("snapshot"); p.add_argument("path", nargs="?"); p.add_argument("--view", action="store_true")
    p = sub.add_parser("list-actions"); p.add_argument("--enabled-only", action="store_true")
    p = sub.add_parser("run-action"); p.add_argument("name")
    p.add_argument("--confirm", help="code returned by the warning of a destructive action")
    p = sub.add_parser("load-workspace"); p.add_argument("path")
    p = sub.add_parser("save-workspace"); p.add_argument("path", nargs="?")
    p = sub.add_parser("load-target-media", help="load a folder or files (--wait: until all are loaded)")
    p.add_argument("paths", nargs="+"); p.add_argument("--wait", action="store_true")
    p = sub.add_parser("load-input-faces", help="load a folder or images (--wait: until all are loaded)")
    p.add_argument("paths", nargs="+"); p.add_argument("--wait", action="store_true")
    p = sub.add_parser("select-target-media", help="no argument: list"); p.add_argument("name", nargs="?")
    p = sub.add_parser("select-target-face", help="no argument: list"); p.add_argument("face_id", nargs="?")
    p = sub.add_parser("list-tensorrt-models"); p.add_argument("--only", choices=["missing", "built", "active", "loaded"])
    p = sub.add_parser("build-tensorrt-cache", help='no argument: active models; "all"; "failed"; or names')
    p.add_argument("models", nargs="*")
    p.add_argument("--rebuild", action="store_true", help="delete and recompile existing engines (warning + code)")
    p.add_argument("--confirm", help="code returned by the --rebuild warning")
    p = sub.add_parser("tensorrt-build-status")
    p.add_argument("--detail", action="store_true", help="list every model (default: running and failed only)")
    p = sub.add_parser("tensorrt-wait", help="wait for the build to finish and print the final status")
    p.add_argument("--max-wait", type=float, default=3 * 3600, help="seconds (default: 3 h)")
    p.add_argument("--interval", type=float, default=5)
    sub.add_parser("cancel-tensorrt-build")
    sub.add_parser("recording-status")
    sub.add_parser("ready-status", help="is VisoMaster idle (no model loading, no preview frame)?")
    p = sub.add_parser("ready-wait", help="wait until VisoMaster is idle")
    p.add_argument("--max-wait", type=float, default=1800)
    p = sub.add_parser("recording-wait", help="wait until the recording ends and print the output file")
    p.add_argument("--max-wait", type=float, default=3 * 3600, help="seconds (default: 3 h)")
    p.add_argument("--interval", type=float, default=5)
    p = sub.add_parser("select-input-face", help="assign a loaded input face to a target face (no name: list)")
    p.add_argument("name", nargs="?"); p.add_argument("--face-id"); p.add_argument("--unselect", action="store_true")
    sub.add_parser("list-embeddings")
    p = sub.add_parser("select-embedding", help="embedding (id or name) for a target face's swap")
    p.add_argument("embedding"); p.add_argument("--face-id"); p.add_argument("--unselect", action="store_true")
    p = sub.add_parser("start-recording"); p.add_argument("--from-start", action="store_true", help="record from the first frame")
    sub.add_parser("stop-recording")
    sub.add_parser("list-jobs")
    p = sub.add_parser("process-jobs"); p.add_argument("jobs", nargs="*")
    p = sub.add_parser("batch-input-faces", help="swap the current frame once per input face (then recording-wait)")
    p.add_argument("input_faces", nargs="*"); p.add_argument("--face-id")
    p = sub.add_parser("apply-offline", help="merge directly into last_workspace.json (VisoMaster closed)")
    _add_import_opts(p)
    p.add_argument("--workspace", help="default: VisoMaster-Fusion\\last_workspace.json")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true", help="if process detection fails")
    p = sub.add_parser("api", help="API state and port (the launcher checkbox is unticked at each start)")
    p.add_argument("--port", type=_port_arg, help="port used at the next launch (1024-65535)")
    p = sub.add_parser("quit", help="close VisoMaster (saves the workspace) and wait for it to exit")
    p.add_argument("--no-save", action="store_true")
    p = sub.add_parser("launch", help="start VisoMaster with the API, wait until it is ready")
    p.add_argument("--hidden", action="store_true", help="no window at all (Qt offscreen)")
    p.add_argument("--port", type=_port_arg)
    p.add_argument("--env", action="append", default=[], metavar="KEY=VALUE", type=_env_arg,
                   help="extra environment variable for VisoMaster (repeatable)")
    sub.add_parser("mcp", help="MCP server (stdio)")
    return ap


def _port_arg(text: str) -> int:
    try:
        port = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a port number: {text}") from None
    if not 1024 <= port <= 65535:
        raise argparse.ArgumentTypeError(f"port out of range 1024-65535: {port}")
    return port


def _env_arg(text: str) -> tuple:
    key, sep, value = text.partition("=")
    if not sep or not key.strip():
        raise argparse.ArgumentTypeError(f"expected KEY=VALUE: {text}")
    return key.strip(), value


async def _progress(ctx, sent: dict, value, total, message: str) -> None:
    """MCP progress notification. The progress value must increase from one notification to the next:
    an unchanged value is not sent again."""
    if value is None or value <= sent.get("last", -1):
        return
    sent["last"] = value
    try:
        await ctx.report_progress(value, total, message)
    except Exception:
        pass  # client without progress support


def _loaded(res: dict, wait: bool, count_key: str) -> dict:
    """--wait of the load commands: block until VisoMaster has loaded every card, then give the count."""
    if not wait:
        return res
    st = wait_ready(on_progress=_print_ready)
    res["wait"] = st.get("wait")
    res["loaded"] = call("status").get(count_key)
    res.pop("note", None)
    return res


def _import_args(a) -> dict:
    return {
        "path": _abs(a.path), "faces": a.faces, "no_control": a.no_control,
        "no_params": a.no_params, "source_face": a.source_face, "missing": a.missing,
    }


def run_cli(a):
    c = a.cmd
    if c == "status":
        return call("status")
    if c == "help":
        return call("help")
    if c == "get-config":
        return call("get_config", {"face_id": a.face_id, "keys": a.keys})
    if c == "export-config":
        return call("export_config", {"path": _abs(a.path), "face_id": a.face_id, "include_personal": a.include_personal})
    if c == "diff":
        return call("diff", _import_args(a))
    if c == "apply-config":
        return call("apply_config", {**_import_args(a), "dry_run": a.dry_run})
    if c == "save-backup":
        return call("save_backup", {"name": a.name})
    if c == "press-parameter-button":
        return call("press_parameter_button", {"key": a.key, "label": a.label, "confirm": a.confirm})
    if c == "list-backups":
        return call("list_backups")
    if c == "restore-backup":
        return call("restore_backup", {"name": a.name})
    if c == "set":
        value = _parse_value(a.value)
        if a.key.endswith("Folder") and value and not Path(value).is_absolute():
            value = _abs(value)  # relative to the folder the command is typed in, not VisoMaster's
        return call("set", {"key": a.key, "value": value, "face_id": a.face_id})
    if c == "schema":
        return call("schema", {"key": a.key, "group": a.group})
    if c == "snapshot":
        out = _abs(a.path) or str(Path.cwd() / f"vmf-snapshot-{time.strftime('%Y%m%d-%H%M%S')}.png")
        res = call("snapshot", {"path": out, "view": a.view, "include_data": False})
        return res
    if c == "list-actions":
        return call("list_actions", {"include_disabled": not a.enabled_only})
    if c == "run-action":  # batches wait until VisoMaster is idle, like start-recording
        return call_when_ready("run_action", {"name": a.name, "confirm": a.confirm}, _print_ready)
    if c == "load-workspace":
        return call("load_workspace", {"path": _abs(a.path)})
    if c == "save-workspace":
        return call("save_workspace", {"path": _abs(a.path)})
    if c == "load-target-media":
        return _loaded(call("load_target_media", {"path": [_abs(p) for p in a.paths]}), a.wait, "target_media")
    if c == "load-input-faces":
        return _loaded(call("load_input_faces", {"path": [_abs(p) for p in a.paths]}), a.wait, "input_faces")
    if c == "select-target-media":
        return call("select_target_media", {"name": a.name})
    if c == "select-target-face":
        return call("select_target_face", {"face_id": a.face_id})
    if c == "list-tensorrt-models":
        return call("list_tensorrt_models", {"only": a.only})
    if c == "build-tensorrt-cache":
        sel = a.models
        keywords = (["active"], ["all"], ["missing"], ["failed"])  # sent as a string, like the MCP does
        models = "active" if not sel else (sel[0] if sel in keywords else sel)
        return call("build_tensorrt_cache", {"models": models, "rebuild": a.rebuild, "confirm": a.confirm})
    if c == "tensorrt-build-status":
        return call("tensorrt_build_status", {"detail": a.detail})
    if c == "tensorrt-wait":
        last = {"line": None}

        def show(st):
            line = _trt_progress_line(st)
            if line != last["line"]:
                print(time.strftime("%H:%M:%S"), line, file=sys.stderr, flush=True)
                last["line"] = line

        return wait_tensorrt(a.max_wait, a.interval, show)
    if c == "cancel-tensorrt-build":
        return call("cancel_tensorrt_build")
    if c == "recording-status":
        return call("recording_status")
    if c == "ready-status":
        return call("ready_status")
    if c == "ready-wait":
        return wait_ready(a.max_wait, on_progress=_print_ready)
    if c == "recording-wait":
        last = {"line": None}

        def show_rec(rec):
            line = _rec_progress_line(rec)
            if line != last["line"]:
                print(time.strftime("%H:%M:%S"), line, file=sys.stderr, flush=True)
                last["line"] = line

        return wait_recording(a.max_wait, a.interval, show_rec)
    if c == "select-input-face":
        return call("select_input_face", {"name": a.name, "face_id": a.face_id, "selected": not a.unselect})
    if c == "list-embeddings":
        return call("list_embeddings")
    if c == "select-embedding":
        return call("select_embedding", {"embedding": a.embedding, "face_id": a.face_id, "selected": not a.unselect})
    if c == "start-recording":
        return start_recording_when_ready(a.from_start, on_progress=_print_ready)
    if c == "stop-recording":
        return call("stop_recording")
    if c == "list-jobs":
        return call("list_jobs")
    if c == "process-jobs":
        return call("process_jobs", {"jobs": a.jobs or None})
    if c == "batch-input-faces":
        return call_when_ready("batch_input_faces", {"input_faces": a.input_faces or None, "face_id": a.face_id},
                               _print_ready)
    if c == "quit":
        return quit_visomaster(not a.no_save)
    if c == "launch":
        return launch(a.hidden, a.port, on_progress=_print_ready, env=dict(a.env))
    if c == "apply-offline":
        return apply_offline(
            _abs(a.path), _abs(a.workspace), a.faces, a.no_control, a.no_params,
            a.source_face, a.missing, a.dry_run, a.force,
        )
    if c == "api":
        if a.port:
            write_config(a.port)
        rt = read_runtime()
        return {"config": read_config(), "running": bool(rt and _port_open(rt.get("port", 0))),
                "runtime_file": str(RUNTIME_PATH),
                "launcher_checkbox": "unticked each time the launcher starts: tick it to start VisoMaster with "
                                     "the API (vmf-cli launch does it without the launcher)"}
    raise ApiError(f"unknown command: {c}")


# ---------------------------------------------------------------------------
# MCP server (stdio)
# ---------------------------------------------------------------------------


def run_mcp():
    build_mcp().run()


def build_mcp():
    """The MCP server with every tool registered (not started: run_mcp does that)."""
    try:  # mcp >= 2: FastMCP renamed MCPServer (same tool/Image API)
        from mcp.server.mcpserver import Context, Image, MCPServer as FastMCP
        from mcp.server.mcpserver.exceptions import ToolError
    except ImportError:
        try:
            from mcp.server.fastmcp import Context, FastMCP, Image
            from mcp.server.fastmcp.exceptions import ToolError
        except ImportError:
            FastMCP = None
    if FastMCP is None:
        sys.stderr.write(
            "The MCP SDK is not installed in this Python. Run vmf_api\\install.bat "
            "then use vmf_api\\.venv\\Scripts\\python.exe vmf_cli.py mcp\n"
        )
        sys.exit(2)

    mcp = FastMCP(
        "visomaster",
        instructions=(
            "Drives a running VisoMaster Fusion (face swap). Personal settings "
            "(paths, GPU, appearance, playback, webcam...) are never overwritten by apply_config. "
            "Always run diff before apply_config. Write commands are refused during "
            "a recording, a scan, a job queue or a TensorRT build. Embeddings: only list them and "
            "select one for the swap; creating, editing, deleting, loading or saving them is reserved "
            "to the user. "
            "Tool map (some clients load tool definitions on demand: search for these names): "
            "state = status (swap on/off, VRAM), snapshot; settings = schema (search by UI label or key; "
            "no argument = list of groups with their switches, the toggles that turn a feature on; "
            "depends_on tells which toggles must be on; buttons such as Apply -> press_parameter_button), "
            "get_config, set_value; "
            "import a config = diff then apply_config; baseline = save_backup(name) then restore_backup(name), "
            "list_backups; Clear VRAM = run_action clearMemoryButton (it turns the swap off: check status.swap); "
            "swap = load_target_media, select_target_media, list_actions / run_action (findTargetFacesButton), "
            "select_target_face, list_embeddings, select_embedding, load_input_faces, select_input_face; "
            "recording = start_recording then wait_recording (returns when the file is written, with "
            "progress), recording_status, stop_recording; batch of images = run_action batchallImageButton / "
            "batchImageButton (VisoMaster's batch processing, without its dialogs) then wait_recording; "
            "one frame x every input face = batch_input_faces then wait_recording; "
            "jobs = list_jobs, process_jobs; "
            "TensorRT = list_tensorrt_models, build_tensorrt_cache then wait_tensorrt_build, "
            "tensorrt_build_status, cancel_tensorrt_build; workspace = load_workspace, save_workspace; "
            "VisoMaster itself = launch_visomaster (hidden=true: no window), quit_visomaster. "
            "Models load lazily (first frame that needs them): wait_ready waits until VisoMaster is idle; "
            "start_recording does it itself. "
            "Never poll in a loop: use the wait_* tools. Do not send other commands while a recording runs "
            "(VisoMaster is sensitive to any action during processing): just call wait_recording."
        ),
    )

    def _call(_command, /, **kw):
        # Positional-only: tools pass arguments literally called "name" (run_action, restore_backup...).
        try:
            return call(_command, kw)
        except ApiError as e:
            raise ToolError(str(e)) from None
        except Exception as e:  # never let the SDK hide the reason behind "Error executing tool"
            raise ToolError(f"{type(e).__name__}: {e}") from None

    @mcp.tool()
    def status() -> dict:
        """VisoMaster version, provider, busy state, media, faces, swap state (enabled? does each target
        face have a source?), VRAM used/total (whole GPU), current or last recording (recording_info)."""
        return _call("status")

    @mcp.tool()
    def get_config(face_id: str | None = None, keys: list[str] | None = None) -> dict:
        """Global settings (control) + parameters of the selected face (or face_id). keys: filter."""
        return _call("get_config", face_id=face_id, keys=keys)

    @mcp.tool()
    def export_config(path: str | None = None, face_id: str | None = None, include_personal: bool = False) -> dict:
        """Export the config in preset format {parameters, control}, without personal keys (include_personal=true:
        with them, e.g. paths and GPU settings). path: a .json file outside VisoMaster's and vmf_api's folders
        (default: vmf_api/exports); an existing file is copied to backups/overwritten first."""
        return _call("export_config", path=_abs(path), face_id=face_id, include_personal=include_personal)

    @mcp.tool()
    def diff(path: str, faces: str = "selected", no_control: bool = False, no_params: bool = False,
             source_face: str | None = None, missing: str = "keep") -> dict:
        """What importing this file (preset or last_workspace.json) would change: changed, excluded (personal),
        unknown (obsolete), invalid, missing. faces: all | selected | <face_id>."""
        return _call("diff", path=_abs(path), faces=faces, no_control=no_control, no_params=no_params,
                     source_face=source_face, missing=missing)

    @mcp.tool()
    def apply_config(path: str, faces: str = "selected", no_control: bool = False, no_params: bool = False,
                     source_face: str | None = None, missing: str = "keep", dry_run: bool = False) -> dict:
        """Import someone else's preset/workspace without overwriting personal settings.
        Automatic backup first (restore_backup to undo)."""
        return _call("apply_config", path=_abs(path), faces=faces, no_control=no_control, no_params=no_params,
                     source_face=source_face, missing=missing, dry_run=dry_run)

    @mcp.tool()
    def list_backups() -> dict:
        """Available backups (most recent first)."""
        names = _call("list_backups")  # an empty list would reach the model as no content at all
        return {"backups": names, "count": len(names)}

    @mcp.tool()
    def press_parameter_button(key: str, label: str | None = None, confirm: str | None = None) -> dict:
        """Click a button attached to a setting, listed in schema as "buttons" (e.g. "Apply" of Face
        Re-Aging's Target Age: VisoMaster only uses new ages after it). Exactly like a click in the UI;
        the setting's depends_on conditions must be met. A reset button returns confirmation_required with
        a code: ask the user, then call again with confirm=<code> (a config backup is written first)."""
        return _call("press_parameter_button", key=key, label=label, confirm=confirm)

    @mcp.tool()
    def save_backup(name: str) -> dict:
        """Save the current config (settings + every face's parameters) under a name, e.g. a baseline to
        come back to between experiments with restore_backup(name). Changes nothing in VisoMaster."""
        return _call("save_backup", name=name)

    @mcp.tool()
    def restore_backup(name: str | None = None) -> dict:
        """Re-apply a backup: a name given to save_backup (its latest save) or a file from list_backups.
        Default: the most recent backup. The current state is backed up first."""
        return _call("restore_backup", name=name)

    @mcp.tool()
    def set_value(key: str, value: str | int | float | bool, face_id: str | None = None) -> dict:
        """Set one key (control or face parameter), validated against the schema (type, bounds, options).
        Applied like a change in VisoMaster's UI (widget + its function, e.g. model loading); a text field
        is committed as if Enter was pressed. If schema shows "buttons" for the key (e.g. Apply), call
        press_parameter_button afterwards. Works on personal keys too
        (only apply_config protects them): the result says personal=true and why. A relative path for a
        ...Folder key is made absolute from the MCP server's folder."""
        if key.endswith("Folder") and isinstance(value, str) and value:
            value = _abs(value)
        return _call("set", key=key, value=value, face_id=face_id)

    @mcp.tool()
    def schema(key: str | None = None, group: str | None = None) -> dict:
        """Settings schema. No argument: the list of groups with their key counts. key: exact name OR words
        of the UI label / key / group / tab, case-insensitive (e.g. "thread", "restore eyes", "pipeline
        position"; unknown -> suggestions). Each key has its UI label, help, tab and depends_on (the
        toggles/selections that must be set for it to apply).
        group: all keys of a group ("*" = everything). Each key: type, default, min/max, options,
        personal + personal_reason."""
        return _call("schema", key=key, group=group)

    @mcp.tool()
    def snapshot(view: bool = False, path: str | None = None, include_image: bool = True):
        """PNG capture of the preview (processed frame; view=true: the view as displayed).
        include_image=false: only source / width / height / size (and path if given), no image returned."""
        res = _call("snapshot", view=view, path=_abs(path), include_data=include_image)
        if not include_image:
            return res
        data = base64.b64decode(res.pop("png_base64"))
        return [Image(data=data, format="png"), res]

    @mcp.tool()
    def list_actions(include_disabled: bool = False) -> dict:
        """Menu actions and main window buttons, with counts. allowed=false: cannot be triggered (reason given);
        requires: "selected_face" and/or "confirmation" (destructive actions)."""
        return _call("list_actions", include_disabled=include_disabled)

    @mcp.tool()
    async def run_action(name: str, confirm: str | None = None) -> dict:
        """Trigger a dialog-free action/button (name from list_actions). Destructive actions return a WARNING and
        a code first: show it to the user and only call again with confirm=<code> after their explicit approval.
        batchallImageButton / batchImageButton run VisoMaster's batch processing of the target list without
        its dialogs and return at once (after waiting until VisoMaster is idle): then call wait_recording
        (stop_recording cancels)."""
        import asyncio

        try:
            return await asyncio.to_thread(call_when_ready, "run_action", {"name": name, "confirm": confirm})
        except ApiError as e:
            raise ToolError(str(e)) from None

    @mcp.tool()
    def load_workspace(path: str) -> dict:
        """Load a full workspace (replaces media, faces, settings). The current one is saved first."""
        return _call("load_workspace", path=_abs(path))

    @mcp.tool()
    def save_workspace(path: str | None = None) -> dict:
        """Save the workspace to a .json file (default: last_workspace.json, its previous content kept in
        backups). Refused while busy (a batch's temporary media and faces would be saved)."""
        return _call("save_workspace", path=_abs(path))

    @mcp.tool()
    def load_target_media(paths: list[str]) -> dict:
        """Load a folder (single path) or a list of files as target media. Asynchronous (cards arrive one
        by one): call wait_ready before using them; batches refuse to start while it loads."""
        return _call("load_target_media", path=[_abs(p) for p in paths])

    @mcp.tool()
    def load_input_faces(paths: list[str]) -> dict:
        """Load a folder (single path) or a list of images as input faces. Asynchronous (cards arrive one
        by one): call wait_ready before using them; batches refuse to start while it loads."""
        return _call("load_input_faces", path=[_abs(p) for p in paths])

    @mcp.tool()
    def select_target_media(name: str | None = None):
        """Select a loaded target media (file name, media_id or index). No name: list them."""
        return _call("select_target_media", name=name)

    @mcp.tool()
    def select_target_face(face_id: str | None = None) -> dict:
        """Select a detected target face. No id: list the faces."""
        return _call("select_target_face", face_id=face_id)

    @mcp.tool()
    def list_tensorrt_models(only: str | None = None) -> dict:
        """ONNX models and the state of their TensorRT engine (built / built on first use / absent, downloaded,
        loaded, used by the settings). only: missing | built | active | loaded."""
        return _call("list_tensorrt_models", only=only)

    @mcp.tool()
    def build_tensorrt_cache(models: list[str] | str = "active", rebuild: bool = False,
                             confirm: str | None = None) -> dict:
        """Build missing TensorRT engines, as on a model's first activation (several minutes each).
        models: list of names, "active", "all" or "failed". An existing engine is never recompiled, except
        with rebuild=true (e.g. after an NVIDIA driver update): the 1st call returns a WARNING and a code,
        to show to the user; only call again with confirm=<code> after their explicit approval.
        Models already known to be lazy (dynamic input shape) are skipped and listed in lazy_known: they cannot
        be pre-built, VisoMaster compiles them on first real use (rebuild=true probes them again).
        TensorRT must be the active provider. Asynchronous: use wait_tensorrt_build."""
        return _call("build_tensorrt_cache", models=models, rebuild=rebuild, confirm=confirm)

    @mcp.tool()
    def tensorrt_build_status(detail: bool = False) -> dict:
        """TensorRT build progress: counts, current model, running / failed models and, once finished, the
        summary. detail=true lists every model (pending / running for X s / built / lazy / failed / skipped)."""
        return _call("tensorrt_build_status", detail=detail)

    @mcp.tool()
    async def wait_tensorrt_build(ctx: Context, max_wait_s: int = 600, detail: bool = False) -> dict:
        """Wait for the TensorRT build to finish and return the final status. Its "summary" lists, per model,
        success (built / lazy) or failure (with the error) and build_time_s, plus total_build_time_s. Sends progress notifications while waiting. After max_wait_s, returns the current status
        (the build goes on): call again to keep waiting. detail=true lists every model."""
        import asyncio

        deadline = time.time() + max_wait_s
        sent: dict = {}
        while True:
            try:
                st = await asyncio.to_thread(poll, "tensorrt_build_status", {"detail": detail})
            except ApiError as e:
                raise ToolError(str(e)) from None
            c = st.get("counts") or {}
            done = sum(c.get(k, 0) for k in ("built", "lazy", "failed", "skipped"))
            # "done/total" from the server: items only lists running / failed models unless detail=true
            total = str(st.get("progress") or "").partition("/")[2]
            await _progress(ctx, sent, done, int(total) if total.isdigit() else None, _trt_progress_line(st))
            if not st.get("running"):
                st["wait"] = "finished"
                return st
            if time.time() >= deadline:
                st["wait"] = f"still running after {max_wait_s} s (the build goes on)"
                return st
            await asyncio.sleep(5)

    @mcp.tool()
    def cancel_tensorrt_build() -> dict:
        """Stop the build after the current model."""
        return _call("cancel_tensorrt_build")

    @mcp.tool()
    def list_embeddings() -> dict:
        """Available merged embeddings (all tabs) and their assignment to target faces.
        Creating, editing, deleting, loading or saving embeddings is reserved to the user."""
        return _call("list_embeddings")

    @mcp.tool()
    def select_input_face(name: str | None = None, face_id: str | None = None, selected: bool = True) -> dict:
        """Assign a loaded input face (file name or id, see load_input_faces) to a target face (default:
        selected face), like a click on its card: it becomes the face's only input face. selected=false:
        remove it. No name: list the input faces and where they are assigned. Some features need an
        input face, not only an embedding (e.g. Face Re-Aging's Apply)."""
        return _call("select_input_face", name=name, face_id=face_id, selected=selected)

    @mcp.tool()
    def select_embedding(embedding: str, face_id: str | None = None, selected: bool = True) -> dict:
        """Choose the embedding (id or exact name) used for a target face's swap (default: selected
        face), like a click in VisoMaster. selected=false: remove it from the face."""
        return _call("select_embedding", embedding=embedding, face_id=face_id, selected=selected)

    @mcp.tool()
    async def start_recording(ctx: Context, from_start: bool = False) -> dict:
        """Start recording the selected media to OutputMediaFolder, from the playhead (from_start=true:
        from the first frame; needed when the playhead is at the end, e.g. after a previous recording).
        First waits until VisoMaster is idle (models loaded, preview frame done), so the recording
        measures processing only. Then call wait_recording: it returns when the video file is written
        (output path, size, duration). stop_recording stops early."""
        import asyncio

        try:
            return await asyncio.to_thread(start_recording_when_ready, from_start)
        except ApiError as e:
            raise ToolError(str(e)) from None

    @mcp.tool()
    async def wait_ready(ctx: Context, max_wait_s: int = 1800) -> dict:
        """Wait until VisoMaster is idle and ready: no model loading (models load on the first frame that
        needs them, e.g. after Clear VRAM or turning the swap or a feature on; a missing TensorRT engine
        is built at that moment), no target media or input faces still loading, no preview frame or
        playback in progress. Sends progress while
        waiting. Returns last_model_loads (model, seconds)."""
        import asyncio

        deadline = time.time() + max_wait_s
        sent: dict = {}
        since, t_start = None, time.time()
        while True:
            try:
                st = await asyncio.to_thread(poll, "ready_status")
            except ApiError as e:
                raise ToolError(str(e)) from None
            now = time.time()
            since = (since or now) if st.get("ready") else None
            await _progress(ctx, sent, round(now - t_start, 1), None, _ready_line(st))  # seconds waited
            if since is not None and now - since >= 2.0:
                st["wait"] = "ready"
                return st
            if now >= deadline:
                st["wait"] = f"still not ready after {max_wait_s} s"
                return st
            await asyncio.sleep(0.5)

    @mcp.tool()
    def stop_recording() -> dict:
        """Stop the recording early (without the confirmation box), or cancel a running batch. Returns the
        result: output file written so far, size, frames."""
        return _call("stop_recording")

    @mcp.tool()
    def recording_status() -> dict:
        """Current or last recording (or batch, kind="batch"): progress (frames, %, processing fps, ETA), then
        the final result (state finished / stopped / failed, output_path, size_bytes, duration_s)."""
        return _call("recording_status")

    @mcp.tool()
    async def wait_recording(ctx: Context, max_wait_s: int = 1800) -> dict:
        """Wait until the recording ends and return its result: state (finished / stopped / failed),
        output_path, size_bytes, duration_s, processing_time_s and average_fps (VisoMaster's own
        measure, as printed in its console: "Processing completed in ..."), split into frames_time_s
        (frame processing) and finalize_s (encoder close, audio merge, saving), vram_mb (start / peak / end). Sends progress notifications (frames, %, ETA) while
        waiting, so no polling is needed. After max_wait_s, returns the progress (the recording goes
        on): call again to keep waiting. Also follows a batch started with run_action (kind="batch": items
        instead of frames; result has processed, failed, processing_time_s, average_fps = items/s)."""
        import asyncio

        deadline = time.time() + max_wait_s
        sent: dict = {}
        while True:
            try:
                rec = await asyncio.to_thread(poll, "recording_status")
            except ApiError as e:
                raise ToolError(str(e)) from None
            p = rec.get("progress") or {}
            await _progress(ctx, sent, p.get("frames_written", 0), p.get("total_frames"), _rec_progress_line(rec))
            if not rec.get("active"):
                rec["wait"] = "finished" if rec.get("result") else "no recording in progress"
                return rec
            if time.time() >= deadline:
                rec["wait"] = f"still recording after {max_wait_s} s (the recording goes on)"
                return rec
            await asyncio.sleep(5)

    @mcp.tool()
    def list_jobs() -> dict:
        """Saved jobs."""
        jobs = _call("list_jobs")  # an empty list would reach the model as no content at all
        return {"jobs": jobs, "count": len(jobs)}

    @mcp.tool()
    def process_jobs(jobs: list[str] | None = None) -> dict:
        """Start the job queue (all jobs, or the listed ones)."""
        return _call("process_jobs", jobs=jobs)

    @mcp.tool()
    async def batch_input_faces(input_faces: list[str] | None = None, face_id: str | None = None) -> dict:
        """Swap the current frame (image, or video frame at the playhead) once per loaded input face (all, or
        the listed file names / ids), each one alone on the target face (default: selected face), and save
        each result to OutputMediaFolder as <media>__<input face>.png (.jpg with ImageFormatToggle), like
        Save Image; an existing file is never overwritten (_2, _3... added). The target face must have no
        embedding assigned (it would be merged). Waits until VisoMaster is idle, starts, returns at once:
        then call wait_recording (progress, then processing_time_s, processed / failed); stop_recording
        cancels."""
        import asyncio

        try:
            return await asyncio.to_thread(call_when_ready, "batch_input_faces",
                                           {"input_faces": input_faces, "face_id": face_id})
        except ApiError as e:
            raise ToolError(str(e)) from None

    @mcp.tool()
    async def quit_visomaster(save: bool = True) -> dict:
        """Close VisoMaster (saves last_workspace.json unless save=false) and wait until it has exited.
        Useful before apply_offline or to relaunch it."""
        import asyncio

        try:
            return await asyncio.to_thread(_quit_and_wait, save)
        except ApiError as e:
            raise ToolError(str(e)) from None

    @mcp.tool()
    async def launch_visomaster(hidden: bool = False, env: dict[str, str] | None = None) -> dict:
        """Start VisoMaster with the API and wait until it is ready (models loaded, idle). hidden=true:
        no window at all; everything still works through these tools (a dialog VisoMaster would open
        is invisible then). env: extra environment variables for VisoMaster's process, only for this run
        (e.g. a VisoMaster switch to compare two behaviours). Returns pid, startup_s, log file. If it already
        runs with the API: returns already_running (refused with env, which would not apply)."""
        import asyncio

        try:
            return await asyncio.to_thread(launch, hidden, None, 300, None, env)
        except ApiError as e:
            raise ToolError(str(e)) from None

    @mcp.tool(name="apply_offline")
    def apply_offline_tool(path: str, workspace: str | None = None, faces: str = "selected", no_control: bool = False,
                      no_params: bool = False, source_face: str | None = None, missing: str = "keep",
                      dry_run: bool = False, force: bool = False) -> dict:
        """VisoMaster CLOSED: same merge directly into last_workspace.json (backup vmf_api/backups/last_workspace.json.bak-<date>).
        force=true only if VisoMaster is really closed but its process could not be checked."""
        try:
            return apply_offline(_abs(path), _abs(workspace), faces, no_control, no_params, source_face, missing,
                                 dry_run, force)
        except (ApiError, ValueError, OSError) as e:
            raise ToolError(str(e)) from None

    return mcp


def main(argv=None):
    # JSON output is UTF-8 even when piped (Windows would otherwise use the console code page).
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    a = build_parser().parse_args(argv)
    if a.cmd == "mcp":
        run_mcp()
        return 0
    try:
        res = run_cli(a)
    except ApiError as e:
        print(json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False, indent=None if a.compact else 2))
        return 1
    except Exception as e:  # stdout stays one JSON document, whatever went wrong
        print(json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"}, ensure_ascii=False))
        return 1
    print(json.dumps({"ok": True, "result": res}, ensure_ascii=False, indent=None if a.compact else 2, default=str))
    if a.cmd in ("quit", "launch") and isinstance(res, dict) and (res.get("closed") is False or res.get("ready") is False):
        return 4  # quit: still running after the wait; launch: started but not idle yet
    # Exit codes for scripts: 0 done, 1 refused / error, 2 bad command line (argparse),
    # 3 nothing done yet: confirmation code needed, 4 a *-wait (or --wait) gave up before the end.
    if isinstance(res, dict):
        if res.get("confirmation_required"):
            return 3
        if str(res.get("wait", "")).startswith("still"):
            return 4
        if a.cmd == "recording-wait" and (res.get("result") or {}).get("state") == "failed":
            return 1  # ended without a video file
        if a.cmd == "tensorrt-wait" and (res.get("summary") or {}).get("success") is False:
            return 1  # at least one model failed to build
    return 0


if __name__ == "__main__":
    sys.exit(main())
