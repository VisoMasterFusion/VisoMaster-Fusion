"""Write commands are refused while recording / scanning / running jobs (no Qt needed)."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import commands  # noqa: E402


class FakeJobs:
    def __init__(self, running):
        self.running = running

    def isRunning(self):
        return self.running


def fake_mw(**vp):
    return SimpleNamespace(
        video_processor=SimpleNamespace(**{"recording": False, "is_processing_segments": False, "processing": False, **vp}),
        job_processor=None,
        scan_issue_worker=None,
        is_batch_processing=False,
        control={},
    )


class BusyTests(unittest.TestCase):
    def setUp(self):
        # Outside VisoMaster: no app package, _scan_active falls back to scan_issue_worker.
        self.calls = []
        commands.COMMANDS["_probe_write"] = commands.Command(lambda ctx: self.calls.append(1) or "done", write=True)
        commands.COMMANDS["_probe_read"] = commands.Command(lambda ctx: "read", write=False)

    def tearDown(self):
        commands.COMMANDS.pop("_probe_write", None)
        commands.COMMANDS.pop("_probe_read", None)

    def run_cmd(self, name, mw):
        return commands.COMMANDS[name].run(commands.Context(mw), {})

    def test_idle_allows_writes(self):
        self.assertEqual(self.run_cmd("_probe_write", fake_mw()), "done")

    def test_refusals(self):
        cases = {
            "enregistrement": fake_mw(recording=True),
            "segments": fake_mw(is_processing_segments=True),
        }
        scan = fake_mw()
        scan.scan_issue_worker = object()
        cases["scan"] = scan
        jobs = fake_mw()
        jobs.job_processor = FakeJobs(True)
        cases["jobs"] = jobs
        batch = fake_mw()
        batch.is_batch_processing = True
        cases["batch"] = batch
        for label, mw in cases.items():
            with self.subTest(label):
                with self.assertRaises(commands.CommandError) as cm:
                    self.run_cmd("_probe_write", mw)
                self.assertIn("refused", str(cm.exception))
                # reads still work
                self.assertEqual(self.run_cmd("_probe_read", mw), "read")
        self.assertEqual(self.calls, [])

    def test_media_loading_is_not_ready(self):
        # load_target_media / load_input_faces return at once: VisoMaster is not ready until its loaders end.
        mw = fake_mw()
        mw.input_faces = {i: object() for i in range(230)}
        mw.input_faces_loader_worker = FakeJobs(True)
        mw.video_loader_worker = FakeJobs(False)
        ready = commands._readiness(mw)
        self.assertFalse(ready["ready"])
        self.assertIn("loading input faces (230 so far)", ready["waiting_for"])
        mw.input_faces_loader_worker = FakeJobs(False)
        self.assertTrue(commands._readiness(mw)["ready"])

    def test_batch_refused_while_loading(self):
        # A batch started during the loading would process a partial list (seen: 50 of 500 images).
        mw = fake_mw(file_type="image", media_path="x.png")
        mw.control = {"OutputMediaFolder": "out"}
        mw.target_faces = {"f1": SimpleNamespace(assigned_merged_embeddings={})}
        mw.selected_target_face_id = "f1"
        mw.input_faces = {"a": object()}
        mw.input_faces_loader_worker = FakeJobs(True)
        with self.assertRaises(commands.CommandError) as cm:
            commands.cmd_batch_input_faces(commands.Context(mw))
        self.assertIn("loading input faces", str(cm.exception))
        self.assertIn("ready-wait", str(cm.exception))
        mw.video_loader_worker = FakeJobs(True)
        with self.assertRaises(commands.CommandError) as cm:
            commands._start_batch(mw, "batchallImageButton", True)
        self.assertIn("loading target media", str(cm.exception))

    def test_write_flags(self):
        for name in ("apply_config", "restore_backup", "set", "run_action", "load_workspace",
                     "load_target_media", "load_input_faces", "start_recording", "process_jobs", "quit",
                     "select_embedding"):
            self.assertTrue(commands.COMMANDS[name].write, name)
        for name in ("status", "get_config", "diff", "export_config", "snapshot", "list_actions", "stop_recording",
                     "list_embeddings"):
            self.assertFalse(commands.COMMANDS[name].write, name)


class TensorRTTests(unittest.TestCase):
    def tearDown(self):
        commands._TRT_JOB = None

    def test_running_build_makes_app_busy(self):
        mw = fake_mw()
        self.assertIsNone(commands.busy_reason(mw))
        commands._TRT_JOB = {"running": True, "current": "GPENBFR512"}
        self.assertIn("TensorRT", commands.busy_reason(mw))
        with self.assertRaises(commands.CommandError):
            commands.COMMANDS["apply_config"].run(commands.Context(mw), {"path": "x.json"})
        # status / cancel stay available
        self.assertFalse(commands.COMMANDS["tensorrt_build_status"].write)
        self.assertFalse(commands.COMMANDS["cancel_tensorrt_build"].write)

    def test_visible_build_dialog_makes_app_busy(self):
        mw = fake_mw()
        mw.build_progress_dialog = SimpleNamespace(isVisible=lambda: True)
        self.assertIn("TensorRT", commands.busy_reason(mw))

    def test_build_refused_without_tensorrt_provider(self):
        mw = fake_mw()
        mw.models_processor = SimpleNamespace(
            provider_name="CUDA",
            providers=[("CUDAExecutionProvider", {}), "CPUExecutionProvider"],
        )
        with self.assertRaises(commands.CommandError) as cm:
            commands.COMMANDS["build_tensorrt_cache"].run(commands.Context(mw), {})
        self.assertIn("provider", str(cm.exception))
        self.assertTrue(commands.COMMANDS["build_tensorrt_cache"].write)

    def _fake_trt_mw(self, tmpdir):
        import os

        files = {}
        for name in ("ModelA", "ModelB", "ModelC"):
            path = os.path.join(tmpdir, name + ".onnx")
            open(path, "wb").close()
            files[name] = path

        class FakeMP:
            provider_name = "TensorRT"
            providers = [("TensorrtExecutionProvider", {}), "CPUExecutionProvider"]
            trt_ep_options = {}
            models_pending_build = set()
            force_unload_in_progress = False

            def __init__(self):
                self.models_path = dict(files)
                self.models = {"ModelA": object(), "ModelB": None, "ModelC": None}
                self.engines = {"ModelA"}  # A already built
                self.log = []

            def _check_tensorrt_cache_state(self, name, path):
                return "EXPLICIT" if name in self.engines else None

            def is_model_active_in_ui(self, name):
                return name in ("ModelA", "ModelB")

            def load_model(self, name):
                self.log.append(("load", name))
                self.engines.add(name)
                self.models[name] = object()
                return self.models[name]

            def unload_model(self, name, force_immediate=False):
                self.log.append(("unload", name, self.force_unload_in_progress))
                self.models[name] = None

            def _clean_tensorrt_cache(self, path, opts):
                self.log.append(("clean", opts["trt_engine_cache_prefix"]))
                self.engines.discard(opts["trt_engine_cache_prefix"])

        mw = fake_mw()
        mw.models_processor = FakeMP()
        return mw

    def _wait_job(self):
        import time

        for _ in range(200):
            if not commands._TRT_JOB["running"]:
                return
            time.sleep(0.01)
        self.fail("build thread did not finish")

    def test_build_skips_existing_and_tracks_progress(self):
        import tempfile

        mw = self._fake_trt_mw(tempfile.mkdtemp())
        ctx = commands.Context(mw)
        listing = commands.COMMANDS["list_tensorrt_models"].run(ctx, {})
        engines = {r["model"]: r["engine"] for r in listing["models"]}
        self.assertEqual(engines, {"ModelA": "built", "ModelB": "absent", "ModelC": "absent"})
        res = commands.COMMANDS["build_tensorrt_cache"].run(ctx, {"models": "all"})
        self.assertEqual(res["queue"], ["ModelB", "ModelC"])  # A already built: never redone
        self._wait_job()
        status = commands.COMMANDS["tensorrt_build_status"].run(ctx, {})
        self.assertEqual(status["counts"]["built"], 2)
        self.assertEqual(status["progress"], "2/2")
        mp = mw.models_processor
        self.assertNotIn(("clean", "ModelA"), mp.log)
        # B and C were not loaded before: unloaded again, bypassing KeepModelsAlive
        self.assertIn(("unload", "ModelC", True), mp.log)
        self.assertIsNone(mp.models["ModelC"])
        # nothing left to build
        res = commands.COMMANDS["build_tensorrt_cache"].run(ctx, {"models": "all"})
        self.assertFalse(res["started"])

    def test_rebuild_needs_confirmation(self):
        import tempfile

        mw = self._fake_trt_mw(tempfile.mkdtemp())
        ctx = commands.Context(mw)
        mp = mw.models_processor
        first = commands.COMMANDS["build_tensorrt_cache"].run(ctx, {"models": ["ModelA"], "rebuild": True})
        self.assertFalse(first["started"])
        self.assertTrue(first["confirmation_required"])
        self.assertIn("DELETED", first["warning"])
        self.assertEqual(mp.log, [], "nothing may happen before confirmation")
        # wrong code or other model list -> new warning, still nothing done
        again = commands.COMMANDS["build_tensorrt_cache"].run(
            ctx, {"models": ["ModelA", "ModelB"], "rebuild": True, "confirm": first["confirm"]})
        self.assertTrue(again["confirmation_required"])
        self.assertEqual(mp.log, [])
        ok = commands.COMMANDS["build_tensorrt_cache"].run(
            ctx, {"models": ["ModelA"], "rebuild": True, "confirm": first["confirm"]})
        self.assertTrue(ok["started"])
        self._wait_job()
        self.assertEqual([e[0] for e in mp.log], ["unload", "clean", "load"])
        self.assertIsNotNone(mp.models["ModelA"], "A was loaded: it stays loaded after the rebuild")
        # a code is single-use
        reuse = commands.COMMANDS["build_tensorrt_cache"].run(
            ctx, {"models": ["ModelA"], "rebuild": True, "confirm": first["confirm"]})
        self.assertTrue(reuse["confirmation_required"])

    def test_known_lazy_models_are_skipped_and_announced(self):
        import tempfile

        mw = self._fake_trt_mw(tempfile.mkdtemp())
        mp = mw.models_processor
        mp.models_pending_build = {"ModelC"}  # probed earlier, dynamic shape, no engine
        ctx = commands.Context(mw)
        listing = commands.COMMANDS["list_tensorrt_models"].run(ctx, {})
        self.assertEqual({r["model"]: r["engine"] for r in listing["models"]}["ModelC"], commands.LAZY_LABEL)
        res = commands.COMMANDS["build_tensorrt_cache"].run(ctx, {"models": "all"})
        self.assertEqual(res["queue"], ["ModelB"])
        self.assertEqual(res["lazy_known"], ["ModelC"])
        self._wait_job()
        self.assertNotIn(("load", "ModelC"), mp.log, "a known lazy model is not probed again")
        # only lazy left: nothing started, but it is reported
        res = commands.COMMANDS["build_tensorrt_cache"].run(ctx, {"models": ["ModelC"]})
        self.assertFalse(res["started"])
        self.assertEqual(res["lazy_known"], ["ModelC"])
        self.assertIn("first use", res["message"])
        # rebuild still includes it (after a driver update its probe must run again)
        res = commands.COMMANDS["build_tensorrt_cache"].run(ctx, {"models": ["ModelC"], "rebuild": True})
        self.assertTrue(res["confirmation_required"])
        self.assertEqual(res["models"], ["ModelC"])

    def test_failure_with_timeout_is_reported_and_retryable(self):
        import tempfile

        mw = self._fake_trt_mw(tempfile.mkdtemp())
        mp = mw.models_processor

        def failing_load(name):
            # Same messages as ModelsProcessor.load_model when the probe times out.
            for attempt in (1, 2, 3):
                print(f"[INFO] Probe attempt {attempt} of 3 for {name}...")
                print(f"[ERROR] Probe process for {name} timed out! Terminating.")
                print("[ERROR] TensorRT Engine build timed out.")
                print(f"[WARN] Probe attempt {attempt} failed with exit code -15.")
            print(f"[ERROR] Isolated probe failed for {name}.")
            return None

        mp.load_model = failing_load
        ctx = commands.Context(mw)
        commands.COMMANDS["build_tensorrt_cache"].run(ctx, {"models": ["ModelB", "ModelC"]})
        self._wait_job()
        st = commands.COMMANDS["tensorrt_build_status"].run(ctx, {})
        self.assertEqual(st["counts"]["failed"], 2, "a failure does not stop the next models")
        item = st["items"][0]
        self.assertEqual(item["timeouts"], 3)
        self.assertEqual(item["attempt"], "3/3")
        self.assertEqual(item["exit_codes"], [-15, -15, -15])
        self.assertIn("timeout", item["error"])
        self.assertIn("failed", item["after_failure"])
        self.assertTrue(item["log_tail"])
        # retry only the failures, with the build now working
        mp.load_model = lambda name: (mp.engines.add(name), object())[1]
        res = commands.COMMANDS["build_tensorrt_cache"].run(ctx, {"models": "failed"})
        self.assertEqual(res["queue"], ["ModelB", "ModelC"])
        self._wait_job()
        self.assertEqual(commands.COMMANDS["tensorrt_build_status"].run(ctx, {})["counts"]["built"], 2)
        # console capture removed once the build thread is done
        self.assertNotIsInstance(sys.stdout, commands._ThreadTee)
        self.assertNotIsInstance(sys.stderr, commands._ThreadTee)

    def test_trt_provider_detection(self):
        self.assertTrue(commands._trt_active(SimpleNamespace(providers=[("TensorrtExecutionProvider", {}), "CPUExecutionProvider"])))
        self.assertFalse(commands._trt_active(SimpleNamespace(providers=["CPUExecutionProvider"])))


class DestructiveActionTests(unittest.TestCase):
    def tearDown(self):
        commands._ACTION_CONFIRM.clear()

    def test_detection(self):
        for name in ("clearTargetFacesButton", "dropAllIssueFramesButton", "actionEdit_ResetParameters"):
            self.assertTrue(commands._is_destructive(name, ""), name)
        self.assertTrue(commands._is_destructive("someFutureButton", "Delete all markers"))
        self.assertFalse(commands._is_destructive("clearMemoryButton", "Clear VRAM"))
        self.assertFalse(commands._is_destructive("swapfacesButton", "Swap Faces"))
        self.assertFalse(commands._is_destructive("buttonMediaPlay", ""))

    def test_no_false_positives(self):
        # whole words only
        self.assertFalse(commands._is_destructive("actionView_Help_Presets", "Presets"))
        self.assertFalse(commands._is_destructive("someDropdownButton", "Choose"))
        # search box reset, view toggles, marker menu
        self.assertFalse(commands._is_destructive("targetVideosSearchResetButton", ""))
        self.assertFalse(commands._is_destructive("actionViewToggle_faces", "Faces / Embeddings"))
        self.assertFalse(commands._is_destructive("addMarkerButton", ""))
        # still caught
        self.assertTrue(commands._is_destructive("clearDroppedFramesButton", "Restore Dropped"))
        self.assertTrue(commands._is_destructive("futureButton", "Removes all scan markers"))
        self.assertTrue(commands._is_destructive("x", "Cleared"))

    def test_modal_kinds(self):
        help_link = ("QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(p))\n"
                     "common_widget_actions.create_and_show_messagebox(main_window, 'Not Found')")
        self.assertEqual(commands._modal_kind(help_link), "external")
        self.assertEqual(commands._modal_kind("menu = QMenu(main_window)\nmenu.exec(pos)"), "menu")
        self.assertEqual(commands._modal_kind("QtWidgets.QMessageBox.question(main_window)"), "dialog")
        self.assertIsNone(commands._modal_kind("main_window.searchBox.clear()"))

    def test_spec_tuple_slots(self):
        src = '''(
            "clearScanResultsButton",
            "Clear Issues",
            "Removes the current scan issue markers.\\n"
            "Does not affect dropped frames.",
            lambda: clear_scan_results(main_window),
            "cleanup",
        ),'''
        self.assertEqual(commands._SPEC_SLOT_RE.findall(src), [("clearScanResultsButton", "clear_scan_results")])

    def test_labels_never_leak_paths(self):
        card = "clip_holiday.mp4\n1920x1080 - 245.3 MB\nD:\\Videos\\Perso\\clip_holiday.mp4"
        self.assertNotIn("Perso", commands._safe_label(card))
        self.assertNotIn("holiday", commands._safe_label(card))
        self.assertNotIn("home", commands._safe_label("Open /home/user/videos/x.mp4"))
        self.assertEqual(commands._safe_label("&Swap Faces"), "Swap Faces")
        self.assertEqual(commands._safe_label("Faces / Embeddings"), "Faces / Embeddings")

    def test_preset_apply_needs_confirmation(self):
        self.assertTrue(commands._is_destructive("applyPresetButton", "Apply"))
        self.assertIn("apply_config", commands.USE_INSTEAD["applyPresetButton"])

    def test_last_workspace_actions_point_to_api(self):
        self.assertEqual(commands.DIALOG_ACTIONS["actionSave_LastWorkspace"], "save_workspace")
        self.assertEqual(commands.DIALOG_ACTIONS["actionReset_to_LastWorkspace"], "load_workspace")
        self.assertIn("backup", commands.MANUAL_REASONS["actionSave_LastWorkspace"])

    def test_privacy_and_kv_reasons(self):
        self.assertTrue(commands._privacy_reserved("targetVideosFilterWebcamsCheckBox", "Webcams"))
        self.assertTrue(commands._privacy_reserved("someButton", "Start Virtual Camera"))
        self.assertFalse(commands._privacy_reserved("swapfacesButton", "Swap Faces"))
        self.assertIn("K/V maps", commands._reserved_reason("purgeKVMapsButton", "Purge"))
        self.assertIn("embeddings", commands._reserved_reason("openEmbeddingButton", ""))

    def test_labels_cosmetics(self):
        self.assertEqual(commands._split_label("Fullscreen\tF11"), ("Fullscreen", "F11"))
        self.assertEqual(commands._split_label("Swap Faces"), ("Swap Faces", ""))
        self.assertEqual(commands._label_from_name("frameAdvanceButton"), "Frame Advance")
        self.assertEqual(commands._label_from_name("sortTargetVideosDirectionButton"), "Sort Target Videos Direction")

    def test_media_check(self):
        from types import SimpleNamespace as NS
        vp = lambda **k: NS(video_processor=NS(**k))
        self.assertTrue(commands._media_ok(vp(media_path="x.mp4", file_type="video")))
        self.assertTrue(commands._media_ok(vp(media_path="x.png", file_type="image")))
        self.assertFalse(commands._media_ok(vp(media_path="0", file_type="webcam")))
        self.assertFalse(commands._media_ok(vp(media_path=None, file_type=None)))

    def test_confirmation_flow(self):
        first = commands._action_confirmation("clearTargetFacesButton", "Clear Target Faces", None)
        self.assertTrue(first["confirmation_required"])
        self.assertIn("erases data", first["warning"])
        code = first["confirm"]
        # a code is bound to its action
        other = commands._action_confirmation("dropAllIssueFramesButton", "", code)
        self.assertTrue(other["confirmation_required"])
        self.assertIsNone(commands._action_confirmation("clearTargetFacesButton", "", code))
        # single use
        self.assertTrue(commands._action_confirmation("clearTargetFacesButton", "", code)["confirmation_required"])


class EmbeddingPolicyTests(unittest.TestCase):
    def test_reserved_controls(self):
        for name in commands.EMBEDDING_RESERVED:
            self.assertTrue(commands._embedding_reserved(name, ""), name)
        # future controls caught by name or label
        self.assertTrue(commands._embedding_reserved("deleteEmbeddingButton", ""))
        self.assertTrue(commands._embedding_reserved("someButton", "Merge selected faces"))
        self.assertTrue(commands._embedding_reserved("kvMapButton", ""))
        self.assertFalse(commands._embedding_reserved("swapfacesButton", "Swap Faces"))
        self.assertFalse(commands._embedding_reserved("buttonMediaPlay", ""))
        # showing/hiding the "Faces / Embeddings" panel touches no embedding
        self.assertFalse(commands._embedding_reserved("actionViewToggle_faces", "Faces / Embeddings"))
        self.assertFalse(commands._embedding_reserved("facesPanelCheckBox", "Faces / Embeddings"))

    def test_no_embedding_mutation_commands(self):
        # Only list + select are exposed.
        embed_cmds = {n for n in commands.COMMANDS if "embed" in n}
        self.assertEqual(embed_cmds, {"list_embeddings", "select_embedding"})


class RecordingTrackingTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        commands._REC.clear()
        commands._OUTPUTS.clear()
        self.dir = Path(tempfile.mkdtemp(prefix="vmf_rec_"))
        self.vp = SimpleNamespace(recording=False, is_processing_segments=False, processing=False,
                                  frames_written=0, next_frame_to_display=0, max_frame_number=99,
                                  fps=25.0, media_path="C:/x/clip.mp4")
        self.mw = SimpleNamespace(video_processor=self.vp, control={"OutputMediaFolder": str(self.dir)})

    def tearDown(self):
        import shutil
        commands._REC.clear()
        commands._OUTPUTS.clear()
        shutil.rmtree(self.dir, ignore_errors=True)

    def _run(self, frames, write_file=True, name="clip_out.mp4"):
        vp = self.vp
        vp.recording = True
        commands._REC_STARTER["by"] = "api"
        rec = commands._rec_update(self.mw)
        self.assertTrue(rec["active"])
        self.assertEqual(rec["started_by"], "api")
        self.assertEqual(rec["total_frames"], 100)
        vp.frames_written, vp.next_frame_to_display = frames // 2, frames // 2
        prog = commands._rec_update(self.mw)["progress"]
        self.assertEqual(prog["frames_written"], frames // 2)
        self.assertAlmostEqual(prog["percent"], frames / 2, places=0)
        vp.frames_written = frames
        vp.start_time, vp.end_time = 100.0, 108.0
        out = self.dir / name
        commands._OUTPUTS.append((__import__("time").time(), str(out)))
        if write_file:
            out.write_bytes(b"x" * 1234)
        vp.recording = False
        return commands._rec_update(self.mw)

    def test_finished(self):
        rec = self._run(100)
        self.assertFalse(rec["active"])
        res = rec["result"]
        self.assertEqual(res["state"], "finished")
        self.assertTrue(res["exists"])
        self.assertEqual(res["size_bytes"], 1234)
        self.assertEqual(res["duration_s"], 4.0)
        self.assertEqual(res["processing_time_s"], 8.0)
        self.assertEqual(res["average_fps"], 12.5)
        self.assertTrue(res["output_path"].endswith("clip_out.mp4"))

    def test_stopped_early(self):
        self.assertEqual(self._run(40)["result"]["state"], "stopped")

    def test_failed_without_file(self):
        res = self._run(100, write_file=False)["result"]
        self.assertEqual(res["state"], "failed")
        self.assertIn("console", res["hint"])

    def test_incomplete_suffix(self):
        vp = self.vp
        vp.recording = True
        commands._rec_update(self.mw)
        vp.frames_written = 30
        planned = self.dir / "clip_out.mp4"
        commands._OUTPUTS.append((__import__("time").time(), str(planned)))
        (self.dir / "clip_out_incomplete.mp4").write_bytes(b"x")
        vp.recording = False
        res = commands._rec_update(self.mw)["result"]
        self.assertTrue(res["incomplete"])
        self.assertTrue(res["exists"])

    def test_recording_started_from_ui_is_tracked(self):
        self.vp.recording = True
        self.vp.frames_written, self.vp.next_frame_to_display = 10, 60  # started at frame 50
        rec = commands._rec_update(self.mw)
        self.assertEqual(rec["started_by"], "visomaster ui")
        self.assertEqual(rec["start_frame"], 50)
        self.assertEqual(rec["total_frames"], 50)

    def test_no_recording(self):
        self.assertIsNone(commands._rec_update(self.mw))


class SchemaSearchTests(unittest.TestCase):
    def setUp(self):
        E = commands.schema_mod.Entry
        self.schema = commands.schema_mod.Schema(entries={
            "nThreadsSlider": E("nThreadsSlider", "control", "SETTINGS_LAYOUT_DATA", "General", "slider"),
            "nStreamSlider": E("nStreamSlider", "control", "SETTINGS_LAYOUT_DATA", "General", "slider"),
            "FaceRestorerEnableToggle": E("FaceRestorerEnableToggle", "parameter", "COMMON", "Restorer", "toggle"),
            "HeadYawSlider": E("HeadYawSlider", "parameter", "FACE_EDITOR_LAYOUT_DATA", "", "slider"),
        })

    def test_find(self):
        self.assertEqual(commands._schema_find(self.schema, "nThreadsSlider")[0], "nThreadsSlider")
        exact, matches, _ = commands._schema_find(self.schema, "thread")
        self.assertIsNone(exact)
        self.assertEqual(matches, ["nThreadsSlider"])
        self.assertEqual(commands._schema_find(self.schema, "restorer enable")[1], ["FaceRestorerEnableToggle"])
        self.assertIn("nThreadsSlider", commands._schema_find(self.schema, "nThredsSlider")[2])

    def test_groups(self):
        g = commands._schema_groups(self.schema)
        self.assertEqual(g["General"], {"keys": 2, "scope": "control"})
        self.assertEqual(g["Restorer"]["keys"], 1)
        self.assertEqual(g["Face Editor"]["keys"], 1)  # layout without groups
        self.assertNotIn("", g)


class SchemaMetadataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.schema = commands.schema_mod.from_source(commands.APP_DIR)

    def test_depends_on(self):
        d = commands.schema_mod.depends_on
        self.assertEqual(d({"parentToggle": "A & B"}), {"all": [{"key": "A", "value": True}, {"key": "B", "value": True}]})
        self.assertEqual(d({"parentToggle": "A | B"}), {"any": [{"key": "A", "value": True}, {"key": "B", "value": True}]})
        self.assertEqual(d({"parentToggle": "A", "requiredToggleValue": False}), {"all": [{"key": "A", "value": False}]})
        self.assertEqual(d({"parentSelection": "S", "requiredSelectionValue": "X"}), {"all": [{"key": "S", "value": "X"}]})
        self.assertIsNone(d({}))

    def test_real_entries_have_ui_metadata(self):
        e = self.schema.get("FaceRestorerTypeSelection")
        self.assertTrue(e.label)
        self.assertEqual(e.tab, "Common")
        self.assertEqual(e.depends_on, {"all": [{"key": "FaceRestorerEnableToggle", "value": True}]})
        self.assertTrue(sum(1 for x in self.schema.entries.values() if x.label) > 300)

    def test_search_by_ui_label(self):
        _, m, _ = commands._schema_find(self.schema, "pipeline position")
        self.assertTrue(m)
        self.assertTrue(all("pipeline position" in (self.schema.get(k).label or "").lower() or "pipeline" in k.lower() for k in m))
        _, m, _ = commands._schema_find(self.schema, "restore eyes")
        self.assertTrue(any((self.schema.get(k).label or "") == "Restore Eyes" for k in m))
        self.assertIn("nThreadsSlider", commands._schema_find(self.schema, "thread")[1])


class SwapAndVramTests(unittest.TestCase):
    class Btn:
        def __init__(self, checked):
            self.c = checked

        def isChecked(self):
            return self.c

    def test_swap_state(self):
        face = SimpleNamespace(assigned_input_faces={}, assigned_merged_embeddings={"111": 1})
        mw = SimpleNamespace(swapfacesButton=self.Btn(False), target_faces={"f1": face})
        st = commands._swap_state(mw)
        self.assertFalse(st["enabled"])
        self.assertTrue(st["faces"]["f1"]["has_source"])
        self.assertIn("swapfacesButton", st["note"])
        mw.swapfacesButton = self.Btn(True)
        self.assertNotIn("note", commands._swap_state(mw))
        face.assigned_merged_embeddings = {}
        self.assertIn("no target face has a source", commands._swap_state(mw)["note"])

    def test_vram(self):
        mw = SimpleNamespace(models_processor=SimpleNamespace(get_gpu_memory=lambda: (6144, 24576)))
        self.assertEqual(commands._vram(mw), {"used_mb": 6144, "total_mb": 24576, "percent": 25.0})
        self.assertIsNone(commands._vram(SimpleNamespace(models_processor=SimpleNamespace(get_gpu_memory=lambda: (0, 0)))))
        self.assertIsNone(commands._vram(SimpleNamespace()))


class NamedBackupTests(unittest.TestCase):
    def test_label(self):
        self.assertEqual(commands._backup_label("my baseline"), "my_baseline")
        self.assertEqual(commands._backup_label("../evil"), "evil")
        with self.assertRaises(commands.CommandError):
            commands._backup_label("  ")

    def test_find_latest_by_name(self):
        names = ["20261017-120000-baseline.json", "20261016-120000-baseline.json", "20261016-110000-other.json"]
        self.assertEqual(commands._find_backup(names, "baseline"), "20261017-120000-baseline.json")
        self.assertIsNone(commands._find_backup(names, "missing"))


class ParameterExecTests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self._saved = commands._PARAM_SPECS
        commands._PARAM_SPECS = {
            "ClipEnableToggle": {"exec_function": lambda mw, v, *a: self.calls.append(("clip", v, a)),
                                 "exec_function_args": ["x"]},
            "BrokenToggle": {"exec_function": lambda mw, v: 1 / 0},
            "PlainSlider": {},
        }

    def tearDown(self):
        commands._PARAM_SPECS = self._saved

    def test_runs_only_changed_keys(self):
        errors = commands._run_param_exec(None, {"ClipEnableToggle": False, "PlainSlider": 1},
                                          {"ClipEnableToggle": True, "PlainSlider": 2})
        self.assertEqual(errors, [])
        self.assertEqual(self.calls, [("clip", True, ("x",))])
        self.calls.clear()
        commands._run_param_exec(None, {"ClipEnableToggle": True}, {"ClipEnableToggle": True})
        self.assertEqual(self.calls, [])

    def test_errors_are_reported(self):
        errors = commands._run_param_exec(None, {}, {"BrokenToggle": True})
        self.assertEqual(len(errors), 1)
        self.assertIn("BrokenToggle: ZeroDivisionError", errors[0])

    def test_unmet(self):
        mw = SimpleNamespace(control={"A": True, "B": False}, selected_target_face_id=None, target_faces={},
                             current_widget_parameters={"P": "X"})
        self.assertEqual(commands._unmet(mw, None), [])
        self.assertEqual(commands._unmet(mw, {"all": [{"key": "A", "value": True}]}), [])
        self.assertEqual(commands._unmet(mw, {"all": [{"key": "B", "value": True}]}), [{"key": "B", "value": True}])
        anyof = [{"key": "A", "value": False}, {"key": "B", "value": True}]
        self.assertEqual(commands._unmet(mw, {"any": anyof}), [{"any_of": anyof}])
        self.assertEqual(commands._unmet(mw, {"any": anyof + [{"key": "P", "value": "X"}]}), [])


class SchemaSwitchesTests(unittest.TestCase):
    def test_switches(self):
        E = commands.schema_mod.Entry
        on = lambda *keys: {"any": [{"key": k, "value": True} for k in keys]}  # noqa: E731
        schema = commands.schema_mod.Schema(entries={
            "DenoiserBeforeToggle": E("DenoiserBeforeToggle", "control", "L", "Denoiser", "toggle", label="Before"),
            "DenoiserAfterToggle": E("DenoiserAfterToggle", "control", "L", "Denoiser", "toggle", label="After"),
            "DenoiserSteps": E("DenoiserSteps", "control", "L", "Denoiser", "slider",
                               depends_on=on("DenoiserBeforeToggle", "DenoiserAfterToggle")),
            "DenoiserSub": E("DenoiserSub", "control", "L", "Denoiser", "toggle", depends_on=on("DenoiserBeforeToggle")),
            "Lonely": E("Lonely", "control", "L", "Other", "toggle"),
        })
        g = commands._schema_groups(schema)
        self.assertEqual(g["Denoiser"]["switches"], {"DenoiserAfterToggle": "After", "DenoiserBeforeToggle": "Before"})
        self.assertNotIn("switches", g["Other"])

    def test_real_denoiser_switches(self):
        g = commands._schema_groups(commands.schema_mod.from_source(commands.APP_DIR))
        sw = {k for v in g.values() for k in v.get("switches", {})}
        self.assertTrue(any("Denoiser" in k for k in sw), sw)

    def test_real_reaging_apply_button(self):
        schema = commands.schema_mod.from_source(commands.APP_DIR)
        withb = [e for e in schema.entries.values() if e.buttons and "Reaging" in e.key]
        self.assertTrue(withb)
        self.assertIn("Apply", [b["label"] for e in withb for b in e.buttons])

    def test_dynamic_options_and_group(self):
        from types import SimpleNamespace

        schema = commands.schema_mod.from_layouts([
            ("SWAPPER_LAYOUT_DATA", {"Swap": {"DFMModelSelection": {
                "label": "DFM Model", "options": lambda m: m.get_selection_values(),
                "default": lambda m: m.get_default_value()}}}, "parameter"),
            ("FACE_EDITOR_LAYOUT_DATA", {"": {"FaceEditorEnableToggle": {"label": "Enable"}}}, "parameter"),
        ])
        e = schema.get("DFMModelSelection")
        self.assertIsNone(e.default)
        self.assertTrue(e.options_note)

        class Combo:
            def count(self):
                return 2

            def itemText(self, i):
                return ["a.dfm", "b.dfm"][i]

        ctx = SimpleNamespace(schema=schema, rules=SimpleNamespace(reason=lambda k, s: None),
                              mw=SimpleNamespace(parameter_widgets={"DFMModelSelection": Combo()}))
        out = commands.cmd_schema(ctx, group="*")
        self.assertEqual(out["DFMModelSelection"]["options"], ["a.dfm", "b.dfm"])
        self.assertEqual(out["FaceEditorEnableToggle"]["group"], "Face Editor")


class PowerThrottlingTests(unittest.TestCase):
    def test_own_process(self):
        import sys

        import winqos

        res = winqos.disable_power_throttling()
        if sys.platform == "win32":
            self.assertTrue(res["applied"], res)
        else:
            self.assertFalse(res["applied"])

class ConsoleTapTests(unittest.TestCase):
    def test_keeps_warnings_only(self):
        import io as _io
        import time as _time

        buf = _io.StringIO()
        tap = commands._ConsoleTap(buf)
        t0 = _time.time() - 0.01
        tap.write("[INFO] fine\n")
        tap.write("[WARN] Tail-drain stalled for too long (8.0s). Forcing finalization.")
        tap.write("\n")
        self.assertIn("[INFO] fine", buf.getvalue())
        got = commands._console_since(t0)
        self.assertEqual(got[-1], "[WARN] Tail-drain stalled for too long (8.0s). Forcing finalization.")
        self.assertFalse(any("[INFO]" in g for g in got))


class DetectorTimingTests(unittest.TestCase):
    def test_wrap_and_stats(self):
        import time as _time

        session = object()
        fw = SimpleNamespace(run_ort_with_iobinding=lambda s, b: None)
        mw = SimpleNamespace(function_worker=fw, models_processor=SimpleNamespace(models={"RetinaFace": session}))

        def run(**k):
            mw.function_worker.run_ort_with_iobinding(session, None)
            return ("b", "k5", "k", "k203")

        det = SimpleNamespace(run=run)
        mw.video_processor = SimpleNamespace(sequential_detector=det)
        commands._DETECT.clear()
        commands._install_rec_hooks(mw)
        wrapped = det.run
        commands._install_rec_hooks(mw)  # no double wrap
        self.assertIs(det.run, wrapped)
        t0 = _time.perf_counter()
        self.assertEqual(det.run(frame_number=1), ("b", "k5", "k", "k203"))
        det.run(frame_number=2)
        t1 = _time.perf_counter()
        s = commands._detector_stats(t0, t1, 2.0)
        self.assertEqual(s["frames"], 2)
        self.assertIn("share_of_frames_time", s)
        self.assertEqual(s["models"]["RetinaFace"]["calls"], 2)
        self.assertIn("not_on_cpu", s["breakdown_ms_per_frame"])
        fw.run_ort_with_iobinding(session, None)  # outside the detector: not counted
        self.assertIsNone(commands._detector_stats(t1 + 1, t1 + 2, 2.0))


class TensorRTSummaryTests(unittest.TestCase):
    def test_summary(self):
        s = commands._trt_summary([
            {"model": "A", "status": "built", "seconds": 61.0},
            {"model": "B", "status": "failed", "seconds": 900.0, "error": "timeout", "timeouts": 3},
            {"model": "C", "status": "skipped", "reason": "already built"},
            {"model": "D", "status": "lazy"},
        ])
        self.assertEqual(s["built"], [{"model": "A", "build_time_s": 61.0}])
        self.assertEqual(s["failed"][0]["error"], "timeout")
        self.assertEqual(s["skipped"], [{"model": "C", "reason": "already built"}])
        self.assertEqual(s["lazy"][0]["model"], "D")
        self.assertEqual(s["total_build_time_s"], 961.0)
        self.assertFalse(s["success"])
        self.assertTrue(commands._trt_summary([{"model": "A", "status": "built", "seconds": 1}])["success"])


class ReadinessTests(unittest.TestCase):
    def setUp(self):
        commands._LOADS["inflight"].clear()
        commands._LOADS["recent"].clear()

    def mw(self, **wpm):
        mw = fake_mw()
        mw.video_processor.worker_pool_manager = SimpleNamespace(
            **{"pending_single_frame_request": None, "current_single_frame_worker": None, **wpm})
        return mw

    def test_idle_is_ready(self):
        self.assertEqual(commands._readiness(self.mw()), {"ready": True})

    def test_preview_and_playback(self):
        r = commands._readiness(self.mw(pending_single_frame_request={"frame_number": 0}))
        self.assertFalse(r["ready"])
        self.assertIn("computing the preview frame", r["waiting_for"])
        mw = self.mw()
        mw.video_processor.processing = True
        self.assertIn("playback in progress", commands._readiness(mw)["waiting_for"])

    def test_load_hook_tracks_real_loads_only(self):
        import threading

        gate = threading.Event()
        mp = SimpleNamespace(models={"Cached": object()}, dfm_models={})

        def load_model(name, session_options=None):
            if name != "Cached":
                gate.wait(5)
                mp.models[name] = object()
            return mp.models[name]

        mp.load_model = load_model
        mw = self.mw()
        mw.models_processor = mp
        commands._install_load_hooks(mw)
        commands._install_load_hooks(mw)  # idempotent
        self.assertTrue(mp.load_model._vmf_wrapped)
        self.assertFalse(hasattr(mp.load_model.__wrapped__, "_vmf_wrapped"))
        mp.load_model("Cached")
        self.assertEqual(commands._LOADS["recent"], [])
        t = threading.Thread(target=mp.load_model, args=("GFPGAN",))
        t.start()
        for _ in range(100):
            if commands._LOADS["inflight"]:
                break
            threading.Event().wait(0.01)
        r = commands._readiness(mw)
        self.assertFalse(r["ready"])
        self.assertEqual(r["loading_models"][0]["model"], "GFPGAN")
        threading.Event().wait(0.06)
        gate.set()
        t.join()
        r = commands._readiness(mw)
        self.assertTrue(r["ready"])
        self.assertEqual(r["last_model_loads"][0]["model"], "GFPGAN")
        self.assertTrue(commands._loads_between(time_now() - 60, time_now()))


class FinalizeTests(unittest.TestCase):
    def test_finalization_still_counts_as_recording(self):
        seen = []
        mw = fake_mw()
        wrapped = commands._wrap_finalize(lambda: seen.append(commands._recording(mw)))
        self.assertFalse(commands._recording(mw))
        wrapped()
        self.assertEqual(seen, [True])
        self.assertFalse(commands._recording(mw))
        self.assertTrue(wrapped._vmf_wrapped)


class LaunchEnvTests(unittest.TestCase):
    def test_env(self):
        import os

        import vmf_cli

        old = dict(os.environ)
        try:
            os.environ.update(VIRTUAL_ENV="x", PYTHONPATH="y", QT_QPA_PLATFORM="windows")
            env = vmf_cli._launch_env(True, 9999)
            self.assertEqual(env["QT_QPA_PLATFORM"], "offscreen")
            self.assertEqual((env["VMF_API"], env["VMF_API_PORT"]), ("1", "9999"))
            self.assertNotIn("VIRTUAL_ENV", env)
            self.assertNotIn("PYTHONPATH", env)
            self.assertNotIn("QT_QPA_PLATFORM", vmf_cli._launch_env(False, 9999))
            extra = vmf_cli._launch_env(False, 9999, {"VISOMASTER_CUDA_SPIN_SYNC": 1})
            self.assertEqual(extra["VISOMASTER_CUDA_SPIN_SYNC"], "1")
        finally:
            os.environ.clear()
            os.environ.update(old)


class LazyTensorRTTests(unittest.TestCase):
    def test_build_steps_are_timed(self):
        import time

        commands._BUILDS["running"].clear()
        commands._BUILDS["done"].clear()
        commands._LOADS["inflight"].clear()
        t0 = time.time()
        commands._build_started("Finalizing TensorRT Build", "Performing first-run inference for:\nYoloFace8n\n\nThis may take")
        mw = fake_mw()
        r = commands._readiness(mw)
        self.assertFalse(r["ready"])
        self.assertEqual(r["tensorrt_building"][0]["model"], "YoloFace8n")
        self.assertIn("TensorRT: Finalizing TensorRT Build (YoloFace8n", r["waiting_for"][0])
        commands._build_finished()
        commands._build_finished()  # unmatched hide: ignored
        b = commands._builds_between(t0 - 1, time.time())
        self.assertEqual([(x["step"], x["model"]) for x in b], [("Finalizing TensorRT Build", "YoloFace8n")])
        self.assertTrue(commands._readiness(mw)["ready"])
        commands._BUILDS["done"].clear()

    def test_pending_and_new_engines(self):
        import os
        import tempfile
        import time
        from pathlib import Path

        mw = SimpleNamespace(models_processor=SimpleNamespace(models_pending_build={"B", "A"}))
        self.assertEqual(commands._lazy_pending(mw), ["A", "B"])
        self.assertEqual(commands._lazy_pending(SimpleNamespace()), [])
        old = commands.APP_DIR
        with tempfile.TemporaryDirectory() as d:
            try:
                commands.APP_DIR = Path(d)
                eng = Path(d) / "tensorrt-engines" / "sub"
                eng.mkdir(parents=True)
                (eng / "old.engine").write_text("x")
                os.utime(eng / "old.engine", (1, 1))
                t0 = time.time() - 1
                (eng / "new.engine").write_text("x")
                (eng / "x.timing").write_text("x")
                self.assertEqual(commands._engines_since(t0), ["new.engine"])
            finally:
                commands.APP_DIR = old


def time_now():
    import time

    return time.time()


if __name__ == "__main__":
    unittest.main()
