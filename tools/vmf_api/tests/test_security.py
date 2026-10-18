"""Security and safety checks of the API layer (no VisoMaster needed)."""

import json
import socket
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import commands  # noqa: E402
import merge  # noqa: E402
import schema as schema_mod  # noqa: E402
import vmf_cli  # noqa: E402

try:
    import server  # needs PySide6 (portable Python)
except ImportError:  # pragma: no cover
    server = None


class LaunchEnvAllowlistTests(unittest.TestCase):
    def test_allowed_names(self):
        self.assertTrue(vmf_cli._env_allowed("VISOMASTER_CUDA_SPIN_SYNC"))
        self.assertTrue(vmf_cli._env_allowed("VMF_GIL_SWITCH_INTERVAL_MS"))
        for name in ("PYTHONSTARTUP", "PYTHONPATH", "PATH", "VMF_API", "VMF_API_PORT", "VMF_API_HOME", "VMF_PYTHON"):
            self.assertFalse(vmf_cli._env_allowed(name), name)

    def test_refused_at_launch(self):
        with self.assertRaises(vmf_cli.ApiError):
            vmf_cli._launch_env(False, 9999, {"PYTHONSTARTUP": "evil.py"})


class WorkspaceTests(unittest.TestCase):
    def _ws(self, data) -> Path:
        d = Path(tempfile.mkdtemp())
        p = d / "ws.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        return p

    def test_kv_map_outside_model_assets(self):
        inside = str(commands.APP_DIR / "model_assets" / "reference_kv_data" / "kv_1.pt")
        p = self._ws({"control": {}, "input_faces_data": {"a": {"media_path": "a.png", "kv_map": inside},
                                                         "b": {"media_path": "b.png", "kv_map": "C:/x/evil.pt"}}})
        with mock.patch("pathlib.Path.exists", lambda self: True):
            found = commands._foreign_kv_maps(p)
        self.assertEqual(found, ["C:/x/evil.pt"])

    def test_kv_map_network_path_refused_unchecked(self):
        # A network host could answer "missing" to the check, then serve the pickle to VisoMaster.
        unc = r"\\host\share\x.pt"
        p = self._ws({"control": {}, "input_faces_data": {"a": {"media_path": "a.png", "kv_map": unc},
                                                         "b": {"media_path": "b.png", "kv_map": "//host/s/y.pt"}}})
        real_resolve = Path.resolve

        def resolve(self, *a, **k):
            if "host" in str(self):
                raise AssertionError("network path resolved")
            return real_resolve(self, *a, **k)

        with mock.patch("pathlib.Path.resolve", resolve):
            found = commands._foreign_kv_maps(p)
        self.assertEqual(sorted(found), sorted([unc, "//host/s/y.pt"]))

    def test_not_a_workspace(self):
        for data in ([1, 2], {"control": {}}, {"parameters": {}, "control": {}}):
            with self.assertRaises(commands.CommandError):
                commands._foreign_kv_maps(self._ws(data))

    def test_plain_workspace_ok(self):
        self.assertEqual(commands._foreign_kv_maps(self._ws({"control": {}, "target_medias_data": []})), [])

    def test_damaged_workspace(self):
        for data in ({"control": {}, "target_medias_data": [{"media_id": "1"}]},
                     {"control": {}, "target_faces_data": {"0": {"cropped_face": []}}},
                     {"control": {}, "input_faces_data": {"a": "not an object"}},
                     {"control": {}, "target_medias_data": [], "job_marker_pairs": "x"}):
            with self.assertRaisesRegex(commands.CommandError, "damaged workspace"):
                commands._foreign_kv_maps(self._ws(data))

    def test_camera_turned_on_refused(self):
        data = {"control": {"SendVirtCamFramesEnableToggle": True}, "target_medias_data": []}
        with self.assertRaisesRegex(commands.CommandError, "privacy"):
            commands._foreign_kv_maps(self._ws(data))
        data["control"]["SendVirtCamFramesEnableToggle"] = False
        self.assertEqual(commands._foreign_kv_maps(self._ws(data)), [])


class OutputPathTests(unittest.TestCase):
    def test_suffix(self):
        self.assertIsNone(commands._out_path(None, ".json"))
        self.assertEqual(commands._out_path("C:/tmp/a.JSON", ".json").name, "a.JSON")
        for bad in ("C:/tmp/a.bat", "C:/tmp/a.json.exe", "C:/tmp/a"):
            with self.assertRaises(commands.CommandError):
                commands._out_path(bad, ".json")

    def test_own_folders_refused(self):
        for root in (commands.APP_DIR, commands.VMF_DIR):
            with self.assertRaisesRegex(commands.CommandError, "inside"):
                commands._out_path(str(root / "presets" / "x.json"), ".json")
        last = commands.APP_DIR / "last_workspace.json"
        self.assertEqual(commands._out_path(str(last), ".json", allowed=(last,)), last.resolve())

    def test_network_paths_refused(self):
        for bad in (r"\\host\share\a.json", "//host/share/a.json", r"\\?\UNC\host\a.json"):
            with self.assertRaisesRegex(commands.CommandError, "network"):
                commands._abs(bad)
            with self.assertRaisesRegex(vmf_cli.ApiError, "network"):
                vmf_cli._abs(bad)

    def test_existing_foreign_file_not_overwritten(self):
        d = Path(tempfile.mkdtemp())
        other = d / "settings.json"
        other.write_text(json.dumps({"editor.fontSize": 14}))
        with self.assertRaisesRegex(commands.CommandError, "already exists"):
            commands._out_path(str(other), ".json")
        image = d / "photo.png"
        image.write_bytes(b"x")
        with self.assertRaisesRegex(commands.CommandError, "already exists"):
            commands._out_path(str(image), ".png")
        mine = d / "preset.json"
        mine.write_text(json.dumps({"control": {}, "parameters": {}}))
        self.assertEqual(commands._out_path(str(mine), ".json"), mine.resolve())

    def test_recent_overwritten_copies_kept(self):
        d = Path(tempfile.mkdtemp())
        target = d / "out.json"
        target.write_text("first")
        with mock.patch.object(commands, "OVERWRITTEN_DIR", d / "overwritten"):
            first = commands._keep_previous(target)
            for i in range(25):
                target.write_text(str(i))
                commands._keep_previous(target)
        self.assertTrue(Path(first).is_file())  # younger than 30 days: never pruned by a burst

    def test_previous_file_kept(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        d = Path(tmp.name)
        target = d / "out.json"
        with mock.patch.object(commands, "OVERWRITTEN_DIR", d / "overwritten"):
            self.assertIsNone(commands._keep_previous(target))
            target.write_text("old")
            kept = commands._keep_previous(target)
        self.assertEqual(Path(kept).read_text(), "old")

    def test_free_path_never_overwrites(self):
        d = tempfile.mkdtemp()
        first = commands._free_path(d, "img__a", ".png")
        Path(first).write_bytes(b"x")
        second = commands._free_path(d, "img__a", ".png")
        self.assertNotEqual(first, second)
        self.assertTrue(second.endswith("img__a_2.png"))

    def test_backup_names_unique_and_ordered(self):
        d = Path(tempfile.mkdtemp())
        with mock.patch.object(commands, "BACKUP_DIR", d), mock.patch.object(commands, "_stamp", lambda: "20261017-120000"):
            a = commands._backup_path("before-apply")
            a.write_text("{}")
            b = commands._backup_path("before-apply")
        self.assertNotEqual(a, b)
        self.assertEqual(sorted([a.name, b.name]), [a.name, b.name])  # list_backups order stays by time


class RestoreBackupTests(unittest.TestCase):
    def test_default_skips_workspace_backups(self):
        names = ["20261017-130000-workspace-before-load.json", "20261017-120000-before-apply.json"]
        ctx = SimpleNamespace(mw=SimpleNamespace(control={}, parameters={}))
        d = Path(tempfile.mkdtemp())
        (d / names[0]).write_text("{}")
        with mock.patch.object(commands, "cmd_list_backups", lambda ctx: names), \
                mock.patch.object(commands, "BACKUP_DIR", d):
            with self.assertRaises(FileNotFoundError):  # it went for the config backup (not on disk here)
                commands.cmd_restore_backup(ctx)
            with self.assertRaisesRegex(commands.CommandError, "whole workspace"):
                commands.cmd_restore_backup(ctx, names[0])


class FindBackupTests(unittest.TestCase):
    def test_exact_label(self):
        names = ["20261017-120500-before-apply.json", "20261017-120000-apply.json", "20261017-110000.2-apply.json"]
        self.assertEqual(commands._find_backup(names, "apply"), "20261017-120000-apply.json")
        self.assertEqual(commands._find_backup(names, "before-apply"), "20261017-120500-before-apply.json")
        self.assertIsNone(commands._find_backup(names, "ply"))

    def test_list_most_recent_first(self):
        import os

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        d = Path(tmp.name)
        for i, name in enumerate(["20261017-120000.10-a.json", "20261017-120000.9-a.json", "20261017-120000-a.json"]):
            (d / name).write_text("{}")
            os.utime(d / name, (1000 + i, 1000 + i))
        with mock.patch.object(commands, "BACKUP_DIR", d):
            self.assertEqual(commands.cmd_list_backups(None)[0], "20261017-120000-a.json")


class StartRecordingGuardTests(unittest.TestCase):
    """Cases where VisoMaster would open a modal message box (the API would hang): refused first."""

    def _ctx(self, file_type="video", max_frame=100, markers=None):
        vp = SimpleNamespace(file_type=file_type, processing=False, is_processing_segments=False,
                             max_frame_number=max_frame)
        mw = SimpleNamespace(video_processor=vp, control={"OutputMediaFolder": "C:/out"}, job_marker_pairs=markers or [])
        return SimpleNamespace(mw=mw)

    def test_refused(self):
        for ctx, msg in ((self._ctx("image"), "image"), (self._ctx(max_frame=0), "length"),
                         (self._ctx(markers=[(10, None)]), "no end marker"),
                         (self._ctx(markers=[(50, 20)]), "start must be before")):
            with self.assertRaisesRegex(commands.CommandError, msg):
                commands.cmd_start_recording(ctx)


class SetUnchangedTests(unittest.TestCase):
    def test_same_value_not_reapplied(self):
        validate = lambda key, value, coerce_strings=False: (True, "CUDA", "")  # noqa: E731
        ctx = SimpleNamespace(
            mw=SimpleNamespace(control={"ProvidersPrioritySelection": "CUDA"}),
            schema=SimpleNamespace(validate=validate, scope_of=lambda k: "control"),
            rules=SimpleNamespace(reason=lambda k, s: None))
        with mock.patch.object(commands, "_apply_control_updates", side_effect=AssertionError("re-applied")):
            res = commands.cmd_set(ctx, "ProvidersPrioritySelection", "CUDA")
        self.assertTrue(res["unchanged"])


class JobsTests(unittest.TestCase):
    def test_empty_list_refused(self):
        with self.assertRaisesRegex(commands.CommandError, "empty job list"):
            commands.cmd_process_jobs(SimpleNamespace(mw=None), [])


class PrivacyTests(unittest.TestCase):
    def _ctx(self):
        validate = lambda key, value, coerce_strings=False: (True, True, "")  # noqa: E731
        return SimpleNamespace(mw=SimpleNamespace(control={}), schema=SimpleNamespace(validate=validate))

    def test_virtual_camera_refused(self):
        for key in ("SendVirtCamFramesEnableToggle", "WebcamEnableToggle", "VirtualCamToggle"):
            with self.assertRaisesRegex(commands.CommandError, "privacy"):
                commands.cmd_set(self._ctx(), key, True)

    def test_camera_gaze_not_blocked(self):
        self.assertIsNone(commands._CAMERA_ON_KEYS.search("FaceExpressionCameraGazeToggle"))

    def test_webcam_card_refused(self):
        card = SimpleNamespace(file_type="webcam", media_id="w0")
        mw = SimpleNamespace(target_videos={"w0": card}, control={})
        with mock.patch.object(commands, "_card_name", lambda c: "Webcam 0"):
            with self.assertRaisesRegex(commands.CommandError, "webcam"):
                commands.cmd_select_target_media(SimpleNamespace(mw=mw), "Webcam 0")


class SchemaNumberTests(unittest.TestCase):
    def test_nan_inf_rejected(self):
        for v in ("nan", "inf", "-inf", float("nan"), 1e400):
            self.assertIsNone(schema_mod._num(v), v)
        self.assertEqual(schema_mod._num("2.5"), 2.5)


class MergeUnknownTests(unittest.TestCase):
    def test_unknown_names_only(self):
        sp = merge.SectionPlan(scope="control", unknown={"OldKey": "secret value"})
        self.assertEqual(sp.to_dict()["unknown"], ["OldKey"])


class RecordingStatusBatchTests(unittest.TestCase):
    def tearDown(self):
        commands._BATCH.clear()

    def test_batch_reported_while_its_videos_record(self):
        t0 = time.time() - 10
        commands._BATCH.clear()
        commands._BATCH.update({"id": "batch-x", "action": "batchImageButton", "active": True, "mode": "m",
                                "started_at": t0, "total": 3, "done": 1, "output_folder": "C:/out"})
        video = {"id": 1, "active": False, "result": {"state": "finished"}}
        with mock.patch.object(commands, "_rec_update", lambda mw, install_hooks=True: video), \
                mock.patch.dict(commands._REC, {"started_at": t0 + 5}):
            res = commands.cmd_recording_status(SimpleNamespace(mw=None))
        self.assertEqual(res["kind"], "batch")
        self.assertTrue(res["active"])  # wait_recording keeps waiting for the batch, not the first video

    def test_late_seen_recording_start(self):
        vp = SimpleNamespace(start_time=time.perf_counter() - 30, fps=30.0)
        self.assertAlmostEqual(commands._rec_start_time(vp, 300), time.time() - 30, delta=1)
        vp = SimpleNamespace(start_time=0.0 - 10 * 86400, fps=30.0)  # not usable: falls back to the fps estimate
        self.assertAlmostEqual(commands._rec_start_time(vp, 300), time.time() - 10, delta=1)


@unittest.skipIf(server is None, "PySide6 missing: run with the portable Python")
class HttpGuardTests(unittest.TestCase):
    def setUp(self):
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        self.port = s.getsockname()[1]
        s.close()
        api = SimpleNamespace(port=self.port, token="secret-token", handle=lambda name, args: {"ok": True, "result": name})
        self.httpd = server._ExclusiveHTTPServer(("127.0.0.1", self.port), server._make_handler(api))
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def _post(self, headers):
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}/api/status", data=b"{}", method="POST",
                                     headers={"Content-Type": "application/json", **headers})
        try:
            with urllib.request.urlopen(req, timeout=5) as r:
                return r.status
        except urllib.error.HTTPError as e:
            return e.code

    def test_guards(self):
        auth = {"Authorization": "Bearer secret-token"}
        self.assertEqual(self._post(auth), 200)
        self.assertEqual(self._post({}), 401)
        self.assertEqual(self._post({"Authorization": "Bearer wrong"}), 401)
        self.assertEqual(self._post({**auth, "Origin": "http://evil.example"}), 403)
        self.assertEqual(self._post({**auth, "Host": "evil.example"}), 403)

    def test_connection_cap(self):
        self.httpd.max_connections = 0  # every new connection is over the cap: closed without an answer
        with self.assertRaises((urllib.error.URLError, ConnectionError, OSError)):
            self._post({"Authorization": "Bearer secret-token"})


if __name__ == "__main__":
    unittest.main()
