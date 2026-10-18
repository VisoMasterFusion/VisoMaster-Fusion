"""Call EVERY MCP tool through the real stdio server, against a fake VisoMaster API.

Catches bugs in the MCP layer itself (argument mapping, result conversion) that the
CLI does not exercise. Needs vmf_api\\.venv (created by install.bat); skipped otherwise.
"""

import base64
import json
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

VMF = Path(__file__).resolve().parent.parent
VENV_PY = VMF / ".venv" / "Scripts" / "python.exe"

# 1x1 transparent PNG
PNG = base64.b64encode(
    bytes.fromhex(
        "89504e470d0a1a0a0000000d4948445200000001000000010806000000"
        "1f15c4890000000d49444154789c6360000002000100e221bc330000000049454e44ae426082"
    )
).decode()


class FakeApi(BaseHTTPRequestHandler):
    calls: list = []

    def log_message(self, *a):
        pass

    def do_POST(self):
        cmd = self.path.rsplit("/", 1)[-1]
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        args = json.loads(body or b"{}")
        FakeApi.calls.append((cmd, args))
        result = {"echo": cmd, "args": args}
        if cmd == "snapshot" and args.get("include_data"):
            result.update(png_base64=PNG, width=1, height=1)
        if cmd == "tensorrt_build_status":
            result = {"running": False, "items": [], "counts": {}, "progress": "0/0"}
        if cmd == "ready_status":
            result = {"ready": True}
        data = json.dumps({"ok": True, "result": result}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def dummy(schema: dict):
    t = schema.get("type")
    if t is None and "anyOf" in schema:
        t = next((s.get("type") for s in schema["anyOf"] if s.get("type") not in (None, "null")), "string")
    return {"string": "x", "integer": 1, "number": 1, "boolean": False, "array": ["x"]}.get(t, "x")


@unittest.skipUnless(VENV_PY.is_file(), "vmf_api\\.venv missing (run install.bat)")
class McpToolsTests(unittest.TestCase):
    def test_every_tool_reaches_the_api(self):
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), FakeApi)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        home = Path(tempfile.mkdtemp(prefix="vmf_mcp_"))
        p = None
        err_log = None
        try:
            (home / "runtime").mkdir()
            (home / "runtime" / "server.json").write_text(
                json.dumps({"port": httpd.server_address[1], "token": "t", "pid": 0}), encoding="utf-8"
            )
            env = dict(os.environ, VMF_API_HOME=str(home))
            # stderr to a file: an undrained pipe fills up with tracebacks and freezes the server.
            err_log = open(home / "mcp_stderr.log", "w", encoding="utf-8")
            p = subprocess.Popen(
                [str(VENV_PY), str(VMF / "vmf_cli.py"), "mcp"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=err_log, text=True, encoding="utf-8", env=env,
            )
            lines: queue.Queue = queue.Queue()
            threading.Thread(target=lambda: [lines.put(x) for x in iter(p.stdout.readline, "")] and None,
                             daemon=True).start()

            def send(m):
                p.stdin.write(json.dumps(m) + "\n")
                p.stdin.flush()

            def recv(i):
                while True:
                    try:
                        line = lines.get(timeout=30)
                    except queue.Empty:
                        self.fail(f"no answer from the MCP server for request {i} within 30 s")
                    d = json.loads(line)
                    if d.get("id") == i:
                        return d

            send({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
                "protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "0"}}})
            recv(1)
            send({"jsonrpc": "2.0", "method": "notifications/initialized"})
            send({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
            tools = recv(2)["result"]["tools"]
            self.assertGreaterEqual(len(tools), 30)
            for n, tool in enumerate(tools, start=10):
                schema = tool.get("inputSchema", {})
                args = {k: dummy(v) for k, v in schema.get("properties", {}).items() if k in schema.get("required", [])}
                with self.subTest(tool=tool["name"]):
                    send({"jsonrpc": "2.0", "id": n, "method": "tools/call",
                          "params": {"name": tool["name"], "arguments": args}})
                    res = recv(n)["result"]
                    text = " ".join(c.get("text", "") for c in res["content"] if c["type"] == "text")
                    if tool["name"] == "apply_offline":
                        # runs locally (no API): a clear error about the missing file is expected
                        self.assertTrue(res.get("isError"))
                        self.assertIn("not found", text)
                        continue
                    self.assertFalse(res.get("isError"), f"{tool['name']}: {text}")
            # arguments named "name" reach the API untouched (regression: _call(name, **kw) clash)
            sent = {c: a for c, a in FakeApi.calls}
            self.assertEqual(sent["run_action"].get("name"), "x")
            self.assertEqual(sent["restore_backup"].get("name", "absent"), "absent")  # optional, not sent
            # a refused path keeps its reason (regression: the SDK showed only "Error executing tool")
            send({"jsonrpc": "2.0", "id": 900, "method": "tools/call",
                  "params": {"name": "load_target_media", "arguments": {"paths": ["\\\\host\\share\\x.mp4"]}}})
            res = recv(900)["result"]
            text = " ".join(c.get("text", "") for c in res["content"] if c["type"] == "text")
            self.assertTrue(res.get("isError"))
            self.assertIn("network or device path", text)
        finally:
            # Always kill the MCP server: a leftover child keeps the pipes open and hangs the caller.
            if p is not None:
                for stream in (p.stdin, p.stdout, p.stderr):
                    try:
                        stream.close()
                    except Exception:
                        pass
                if p.poll() is None:
                    p.kill()
                p.wait(timeout=30)
            try:
                err_log.close()
            except Exception:
                pass
            httpd.shutdown()
            httpd.server_close()
            shutil.rmtree(home, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
