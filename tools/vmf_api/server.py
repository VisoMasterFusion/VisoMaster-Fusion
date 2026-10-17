"""Local HTTP control port for VisoMaster Fusion.

* 127.0.0.1 only, random token per start (runtime\\server.json), required on
  every request (Authorization: Bearer <token>).
* HTTP runs in a background thread; every command runs on the Qt main thread
  through a queued signal, the HTTP thread waits on a Future (with timeout).

Protocol: POST /api/<command> with a JSON object of arguments.
Answer   : {"ok": bool, "result": ..., "error": "..."}
"""

from __future__ import annotations

import concurrent.futures
import hmac
import json
import os
import secrets
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from PySide6 import QtCore

import commands
from vmf_common import API_VERSION, RUNTIME_PATH, log, write_json_atomic

MAX_BODY = 20 * 1024 * 1024


class _Dispatcher(QtCore.QObject):
    """Lives in the Qt main thread; runs jobs posted from other threads."""

    _job = QtCore.Signal(object)

    def __init__(self):
        super().__init__()
        self._job.connect(self._run, QtCore.Qt.ConnectionType.QueuedConnection)

    def _run(self, job):
        fn, future = job
        if not future.set_running_or_notify_cancel():
            return  # caller already gave up
        try:
            future.set_result(fn())
        except BaseException as e:  # noqa: BLE001 - forwarded to the caller
            future.set_exception(e)

    def call(self, fn, timeout):
        future = concurrent.futures.Future()
        self._job.emit((fn, future))
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            if future.cancel():
                raise TimeoutError(
                    "VisoMaster did not process the command in time "
                    "(main thread busy or a dialog is open); command cancelled"
                ) from None
            raise TimeoutError(
                "command still running in VisoMaster after the timeout; "
                "it will complete but its result is lost"
            ) from None


class _ExclusiveHTTPServer(ThreadingHTTPServer):
    """Refuse a port already in use.

    ThreadingHTTPServer sets SO_REUSEADDR, which on Windows lets a second
    VisoMaster bind the same port silently (requests then reach the first
    instance with the second one's token). Use SO_EXCLUSIVEADDRUSE instead.
    """

    allow_reuse_address = False
    daemon_threads = True
    max_connections = 32  # one thread each: a local program opening thousands cannot exhaust VisoMaster

    def __init__(self, *a, **k):
        self._active = 0
        self._active_lock = threading.Lock()
        super().__init__(*a, **k)

    def process_request(self, request, client_address):
        with self._active_lock:
            full = self._active >= self.max_connections
            if not full:
                self._active += 1
        if full:
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except Exception:
            with self._active_lock:
                self._active -= 1
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            with self._active_lock:
                self._active -= 1

    def server_bind(self):
        import socket

        excl = getattr(socket, "SO_EXCLUSIVEADDRUSE", None)
        if excl is not None:
            self.socket.setsockopt(socket.SOL_SOCKET, excl, 1)
        super().server_bind()


class ApiServer:
    def __init__(self, main_window, port: int):
        self.main_window = main_window
        self.port = port
        self.token = secrets.token_urlsafe(32)
        self.dispatcher = _Dispatcher()
        self.ctx = commands.Context(main_window, api_port=port)
        self.httpd: ThreadingHTTPServer | None = None
        self.thread: threading.Thread | None = None

    # ---------- lifecycle ----------

    def start(self):
        handler = _make_handler(self)
        self.httpd = _ExclusiveHTTPServer(("127.0.0.1", self.port), handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, name="vmf-api-http", daemon=True)
        self.thread.start()
        write_json_atomic(
            RUNTIME_PATH,
            {
                "port": self.port,
                "token": self.token,
                "pid": os.getpid(),
                "api_version": API_VERSION,
                "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
        )
        log(f"API listening on http://127.0.0.1:{self.port} (token in {RUNTIME_PATH})")
        return self

    def stop(self):
        if getattr(self, "_stopped", False):
            return
        self._stopped = True
        if self.httpd is not None:
            try:
                self.httpd.shutdown()
                self.httpd.server_close()
            except Exception as e:
                log(f"server shutdown: {e!r}")
            self.httpd = None
        try:
            from vmf_common import read_runtime

            rt = read_runtime()
            if rt and rt.get("pid") == os.getpid():
                os.remove(RUNTIME_PATH)
        except OSError:
            pass
        log("API stopped")

    # ---------- dispatch ----------

    def handle(self, name: str, args: dict) -> dict:
        cmd = commands.COMMANDS.get(name)
        if cmd is None:
            return {"ok": False, "error": f"unknown command: {name}", "result": sorted(commands.COMMANDS)}
        try:
            if cmd.main_thread:
                result = self.dispatcher.call(lambda: cmd.run(self.ctx, args), cmd.timeout)
            else:  # read-only polling: never touches Qt, so it does not disturb VisoMaster's UI thread
                result = cmd.run(self.ctx, args)
            return {"ok": True, "result": result}
        except commands.CommandError as e:
            return {"ok": False, "error": str(e)}
        except TimeoutError as e:
            return {"ok": False, "error": str(e)}
        except Exception as e:  # unexpected: report, never crash VisoMaster
            log(f"error in {name}: {e!r}")
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def _make_handler(api: ApiServer):
    allowed_hosts = {f"127.0.0.1:{api.port}", f"localhost:{api.port}"}

    class Handler(BaseHTTPRequestHandler):
        server_version = "VMF-API/" + API_VERSION
        protocol_version = "HTTP/1.1"
        timeout = 60  # socket reads: an idle or stalled client cannot hold a thread forever

        def log_message(self, fmt, *args):  # keep the console quiet
            pass

        def _refuse(self, status: int, error: str):
            # Read the small body the client already sent before answering: closing a socket with unread
            # data makes Windows reset the connection, and the client sees an abort instead of the status.
            try:
                length = int(self.headers.get("Content-Length") or 0)
                if 0 < length <= 1024 * 1024:
                    self.rfile.read(length)
            except (ValueError, OSError):
                pass
            self.close_connection = True
            self._send(status, {"ok": False, "error": error})

        def _send(self, status: int, payload: dict):
            body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            if self.close_connection:
                self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body)

        def _foreign(self) -> bool:
            # Browsers (DNS rebinding / CSRF): wrong Host or any Origin -> refuse.
            return self.headers.get("Host", "") not in allowed_hosts or bool(self.headers.get("Origin"))

        def _authorized(self) -> bool:
            auth = self.headers.get("Authorization", "")
            token = auth[7:] if auth.startswith("Bearer ") else self.headers.get("X-VMF-Token", "")
            return hmac.compare_digest(token.encode(), api.token.encode())

        def do_GET(self):
            self._refuse(405, "use POST /api/<command>")

        def do_POST(self):
            if self._foreign():
                self._refuse(403, "refused: requests from a web page (Origin header) or "
                                  "through another host name are not accepted")
                return
            if not self._authorized():
                self._refuse(401, "missing or invalid token")
                return
            if not self.path.startswith("/api/"):
                self._refuse(404, "unknown path")
                return
            name = self.path[len("/api/") :].split("?", 1)[0].strip("/")
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                length = -1
            if length < 0 or length > MAX_BODY:
                self.close_connection = True  # body left unread: the connection cannot be reused
                self._send(413, {"ok": False, "error": "invalid or too large request body"})
                return
            raw = self.rfile.read(length) if length else b""
            try:
                args = json.loads(raw.decode("utf-8")) if raw else {}
                if not isinstance(args, dict):
                    raise ValueError("JSON object expected")
            except ValueError as e:
                self._send(400, {"ok": False, "error": f"invalid JSON: {e}"})
                return
            self._send(200, api.handle(name, args))

    return Handler


def start(main_window, port: int) -> ApiServer | None:
    import winqos

    # Same speed whether the window is in front or not (see winqos). VMF_ALLOW_POWER_THROTTLING=1 keeps the
    # Windows default (to compare).
    if os.environ.get("VMF_ALLOW_POWER_THROTTLING") == "1":
        log("power throttling left to Windows (VMF_ALLOW_POWER_THROTTLING=1)")
    else:
        qos = winqos.disable_power_throttling()
        log("power throttling disabled" if qos.get("applied") else f"power throttling unchanged: {qos.get('reason')}")
    gil_ms = os.environ.get("VMF_GIL_SWITCH_INTERVAL_MS")
    if gil_ms:
        # Experiment: how often Python hands the GIL to another waiting thread (default 5 ms).
        try:
            sys.setswitchinterval(float(gil_ms) / 1000.0)
            log(f"GIL switch interval set to {sys.getswitchinterval() * 1000:g} ms")
        except Exception as e:
            log(f"GIL switch interval unchanged: {e}")
    try:
        commands.install_hooks(main_window)
    except Exception as e:
        log(f"hooks not installed: {type(e).__name__}: {e}")
    try:
        srv = ApiServer(main_window, port).start()
    except OSError as e:
        log(
            f"port {port} already in use (another VisoMaster with the API?): "
            f"this instance runs WITHOUT the API ({e})"
        )
        return None
    app = QtCore.QCoreApplication.instance()
    if app is not None:
        app.aboutToQuit.connect(srv.stop)
    return srv
