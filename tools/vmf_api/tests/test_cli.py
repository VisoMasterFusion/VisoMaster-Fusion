"""CLI output for humans and scripts: progress lines, exit codes (no VisoMaster needed)."""

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import vmf_cli  # noqa: E402


class LaunchClosedTests(unittest.TestCase):
    """VisoMaster closed while launch waits for it to be ready."""

    NO_API = "VisoMaster is not running with the API (no runtime/server.json). Tick \"Enable API / MCP\" ..."

    def _launch(self, proc):
        patches = [
            mock.patch.object(vmf_cli, "LAUNCH_LOG", Path(tempfile.mkdtemp()) / "visomaster.log"),
            mock.patch.object(vmf_cli, "visomaster_running", return_value=(False, "")),
            mock.patch.object(vmf_cli, "read_config", return_value={"port": 1}),
            mock.patch.object(vmf_cli, "_port_open", side_effect=[False, True]),
            mock.patch.object(vmf_cli, "_launch_env", return_value={}),
            mock.patch.object(vmf_cli, "_python_exe", return_value=Path("python.exe")),
            mock.patch.object(vmf_cli.subprocess, "Popen", return_value=proc),
            mock.patch.object(vmf_cli, "read_runtime", return_value={"pid": 42, "port": 1}),
            mock.patch.object(vmf_cli, "wait_ready", side_effect=vmf_cli.ApiError(self.NO_API)),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        with self.assertRaises(vmf_cli.ApiError) as cm:
            vmf_cli.launch()
        return str(cm.exception)

    def _proc(self, wait):
        proc = mock.Mock(pid=42)
        proc.poll.return_value = None
        proc.wait.side_effect = wait
        return proc

    def test_closed_says_so(self):
        msg = self._launch(self._proc(lambda timeout: 0))
        self.assertTrue(msg.startswith("VisoMaster was closed before it was ready (exit code 0)"), msg)
        self.assertNotIn("Tick", msg)

    def test_still_running_keeps_the_error(self):
        def wait(timeout):
            raise vmf_cli.subprocess.TimeoutExpired("python.exe", timeout)

        self.assertEqual(self._launch(self._proc(wait)), self.NO_API)


class ProgressLineTests(unittest.TestCase):
    def test_batch_progress_is_not_a_recording(self):
        rec = {"id": "b1", "kind": "batch", "active": True,
               "progress": {"frames_written": 19, "total_frames": 500, "unit": "input faces", "percent": 3.8,
                            "processing_fps": 3.7, "eta_s": 131}}
        self.assertEqual(vmf_cli._rec_progress_line(rec),
                         "batch 19/500 input faces (3.8%), 3.7 per s, about 131 s left")

    def test_batch_result_gives_counts_not_bytes(self):
        rec = {"id": "b1", "kind": "batch", "active": False,
               "result": {"state": "finished", "exists": True, "output_path": "D:/out", "processed": 500,
                          "failed": 0, "processing_time_s": 113.4, "average_fps": 4.41}}
        self.assertEqual(vmf_cli._rec_progress_line(rec),
                         "finished: 500 processed, 0 failed in 113.4 s (4.41 per s) -> D:/out")

    def test_recording_lines_unchanged(self):
        rec = {"id": 1, "active": True, "progress": {"frames_written": 10, "total_frames": 270, "percent": 3.7}}
        self.assertEqual(vmf_cli._rec_progress_line(rec), "recording 10/270 frames (3.7%)")
        rec = {"id": 1, "result": {"state": "finished", "exists": True, "output_path": "a.mp4", "size_bytes": 5}}
        self.assertEqual(vmf_cli._rec_progress_line(rec), "finished: a.mp4 (5 bytes)")

    def test_tensorrt_line_without_build(self):
        st = {"running": False, "message": "no build started since VisoMaster started"}
        self.assertEqual(vmf_cli._trt_progress_line(st), "no build started since VisoMaster started")


class ExitCodeTests(unittest.TestCase):
    def run_main(self, result=None, error=None):
        def fake(_a):
            if error:
                raise vmf_cli.ApiError(error)
            return result

        out = io.StringIO()
        with mock.patch.object(vmf_cli, "run_cli", fake), contextlib.redirect_stdout(out):
            return vmf_cli.main(["--compact", "status"])

    def test_codes(self):
        self.assertEqual(self.run_main({"ok": 1}), 0)
        self.assertEqual(self.run_main(error="refused"), 1)
        self.assertEqual(self.run_main({"triggered": False, "confirmation_required": True}), 3)
        self.assertEqual(self.run_main({"wait": "still recording after waiting 60 s (the recording goes on)"}), 4)
        self.assertEqual(self.run_main({"wait": "finished"}), 0)

    def test_bad_command_line(self):
        for argv in (["no-such-command"], ["launch", "--env", "NOEQUALS"], ["launch", "--port", "80"],
                     ["api", "--port", "70000"], ["launch", "--env", "=x"]):
            with self.assertRaises(SystemExit) as cm, contextlib.redirect_stderr(io.StringIO()):
                vmf_cli.main(argv)
            self.assertEqual(cm.exception.code, 2, argv)

    def test_env_and_port_parsed(self):
        a = vmf_cli.build_parser().parse_args(["launch", "--env", "VMF_X=a=b", "--port", "8800"])
        self.assertEqual(dict(a.env), {"VMF_X": "a=b"})
        self.assertEqual(a.port, 8800)

    def run_wait(self, cmd, result):
        out = io.StringIO()
        with mock.patch.object(vmf_cli, "run_cli", lambda _a: result), contextlib.redirect_stdout(out):
            return vmf_cli.main(["--compact", cmd])

    def test_failed_waits(self):
        self.assertEqual(self.run_wait("recording-wait", {"wait": "finished", "result": {"state": "failed"}}), 1)
        self.assertEqual(self.run_wait("recording-wait", {"wait": "finished", "result": {"state": "stopped"}}), 0)
        self.assertEqual(self.run_wait("tensorrt-wait", {"wait": "finished", "summary": {"success": False}}), 1)
        self.assertEqual(self.run_wait("tensorrt-wait", {"wait": "finished", "summary": {"success": True}}), 0)

    def test_quit_and_launch_not_done(self):
        self.assertEqual(self.run_wait("quit", {"closed": False}), 4)
        self.assertEqual(self.run_wait("launch", {"launched": True, "ready": False}), 4)


class PollTests(unittest.TestCase):
    def test_retries_busy_timeouts_only(self):
        calls = []

        def busy(command, args=None, timeout=0):
            calls.append(command)
            if len(calls) < 3:
                raise vmf_cli.ApiError("VisoMaster did not process the command in time (main thread busy)")
            return {"ok": True}

        with mock.patch.object(vmf_cli, "call", busy), mock.patch.object(vmf_cli.time, "sleep", lambda s: None):
            self.assertEqual(vmf_cli.poll("tensorrt_build_status"), {"ok": True})
        self.assertEqual(len(calls), 3)

        def refused(command, args=None, timeout=0):
            calls.append(command)
            raise vmf_cli.ApiError("refused: busy")

        calls.clear()
        with mock.patch.object(vmf_cli, "call", refused):
            with self.assertRaises(vmf_cli.ApiError):
                vmf_cli.poll("recording_status")
        self.assertEqual(len(calls), 1)



class SnapshotPathTests(unittest.TestCase):
    def test_current_folder_outside_visomaster(self):
        import tempfile
        cwd = Path(tempfile.gettempdir())
        out = vmf_cli._snapshot_path(None, cwd)
        self.assertEqual(Path(out).parent, cwd.resolve())
        self.assertTrue(Path(out).name.startswith("vmf-snapshot-"))

    def test_inside_visomaster_or_vmf_api_left_to_the_api(self):
        self.assertIsNone(vmf_cli._snapshot_path(None, vmf_cli.VMF_DIR))
        self.assertIsNone(vmf_cli._snapshot_path(None, vmf_cli.APP_DIR / "app"))

    def test_given_path_kept(self):
        self.assertEqual(vmf_cli._snapshot_path("C:/out/a.png", Path.cwd()), str(Path("C:/out/a.png").resolve()))


if __name__ == "__main__":
    unittest.main()
