"""Fusion tests: exclusions, patterns, sanitization, preset / workspace formats.

Schema comes from the real layout files of the VisoMaster-Fusion checkout (AST, no import).
Real JSON files are used when present:
  tests/fixtures/*.json                      (copied real files)
  VMF_TEST_WORKSPACE / VMF_TEST_PRESET       (paths to your own files)
"""

import copy
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
VMF = HERE.parent
sys.path.insert(0, str(VMF))

import merge  # noqa: E402
import schema as schema_mod  # noqa: E402
from vmf_common import APP_DIR  # noqa: E402  (next to vmf_api, or vmf_api is in its tools folder)
FIXTURES = HERE / "fixtures"


def load_schema():
    return schema_mod.from_source(APP_DIR)


SCHEMA = load_schema()
RULES = merge.PersonalRules.load(VMF / "personal_keys.json")
CTL_DEFAULTS = {}
PARAM_DEFAULTS = {}
for _k, _e in SCHEMA.entries.items():
    ok, v, _ = SCHEMA.validate(_k, _e.default, coerce_strings=True) if _e.default is not None else (False, None, "")
    if ok:
        (CTL_DEFAULTS if _e.scope == "control" else PARAM_DEFAULTS)[_k] = v


def write_json(tmp, name, data):
    p = Path(tmp) / name
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def fake_workspace(control, faces, selected=None):
    return {
        "control": control,
        "target_medias_data": [{"media_path": "D:/videos/a.mp4", "media_id": "m1"}],
        "selected_media_id": "m1",
        "selected_target_face_id": selected,
        "input_faces_data": {"i1": {"media_path": "D:/faces/x.png"}},
        "target_faces_data": {
            fid: {"parameters": p, "cropped_face": [], "embedding_store": {}, "assigned_input_faces": [],
                  "assigned_merged_embeddings": [], "assigned_input_embedding": {}}
            for fid, p in faces.items()
        },
        "current_widget_parameters": dict(PARAM_DEFAULTS),
        "markers": {"10": {"parameters": {}, "control": {}}},
        "window_state_data": {"x": 1},
    }


class SchemaTests(unittest.TestCase):
    def test_layouts_found(self):
        self.assertGreater(len(SCHEMA.keys("control")), 50)
        self.assertGreater(len(SCHEMA.keys("parameter")), 100)

    def test_common_layout_is_per_face(self):
        # Code check of the spec's open question: COMMON_LAYOUT_DATA (Face Restorer,
        # Face expressions) is added with data_type="parameter" -> per face.
        restorer = [k for k, e in SCHEMA.entries.items() if e.layout == "COMMON_LAYOUT_DATA"]
        self.assertTrue(restorer)
        self.assertTrue(all(SCHEMA.scope_of(k) == "parameter" for k in restorer))
        self.assertEqual(SCHEMA.scope_of("DenoiserModeSelectionBefore") or "control", "control")

    def test_default_types(self):
        e = SCHEMA.get("nThreadsSlider")
        self.assertEqual((e.default, e.min, e.max), (4, 1, 30))
        self.assertTrue(all(type(v) is int for v in (e.default, e.min, e.max)))
        self.assertIsInstance(SCHEMA.get("FaceDetailerCropFactorDecimalSlider").default, float)
        self.assertIs(SCHEMA.get("FaceDetailerEnableToggle").default, False)
        self.assertIsInstance(SCHEMA.get("FaceDetailerRestorerTypeSelection").default, str)

    def test_validate(self):
        e = SCHEMA.get("FaceDetailerCropFactorDecimalSlider")
        self.assertEqual(e.kind, "decimal")
        self.assertTrue(SCHEMA.validate(e.key, 2.0)[0])
        self.assertFalse(SCHEMA.validate(e.key, 99.0)[0])
        self.assertFalse(SCHEMA.validate(e.key, "2.0")[0])
        self.assertEqual(SCHEMA.validate(e.key, "2.0", coerce_strings=True)[1], 2.0)
        self.assertEqual(SCHEMA.validate("FaceDetailerEnableToggle", "on", coerce_strings=True)[1], True)
        self.assertFalse(SCHEMA.validate("FaceDetailerEnableToggle", 1)[0])
        sel = SCHEMA.get("FaceDetailerRestorerTypeSelection")
        self.assertTrue(SCHEMA.validate(sel.key, sel.options[0])[0])
        self.assertFalse(SCHEMA.validate(sel.key, "NoSuchModel")[0])
        self.assertFalse(SCHEMA.validate("FaceDetailerMinFaceSizeSlider", 10.5)[0])
        # upstream default below its own min: still valid (VisoMaster stores it)
        lips = "FaceExpressionNormalizeLipsThresholdDecimalSlider"
        if SCHEMA.get(lips) is not None:
            self.assertTrue(SCHEMA.validate(lips, float(SCHEMA.get(lips).default))[0])
            self.assertFalse(SCHEMA.validate(lips, 0.05)[0])


class PersonalRulesTests(unittest.TestCase):
    def r(self, key, value=None, current=None):
        return RULES.reason(key, SCHEMA, value, current)

    def test_groups(self):
        for g in ("Appearance", "Output Settings", "Video Playback Settings", "Webcam and Virtualcam Settings", "Misc"):
            keys = [k for k, e in SCHEMA.entries.items() if e.layout == "SETTINGS_LAYOUT_DATA" and e.group == g]
            self.assertTrue(keys, g)
            for k in keys:
                self.assertIsNotNone(self.r(k), k)

    def test_general_keys(self):
        for k in ("ProvidersPrioritySelection", "nThreadsSlider", "nStreamSlider", "FrameWorkerDelayDecimalSlider"):
            self.assertIsNotNone(self.r(k), k)
        # other General keys are imported
        self.assertIsNone(self.r("KeepControlsToggle"))
        self.assertIsNone(self.r("GlobalInputResizeToggle"))

    def test_isolated_keys(self):
        for k in ("ConfirmBeforeStoppingRecordingToggle", "OpenOutputToggle", "MaxDFMModelsSlider",
                  "KeepModelsAliveToggle", "ReferenceKVTensorsSelection"):
            self.assertIsNotNone(self.r(k), k)

    def test_patterns(self):
        for k in ("OutputMediaFolder", "SomeNewPathSetting", "FooDirectory", "UseGpuX", "GPUIndex",
                  "NewWebcamThing", "VirtCamFoo", "ThemeX", "ToastY"):
            self.assertIsNotNone(self.r(k), k)

    def test_path_values(self):
        self.assertIsNotNone(self.r("WhateverText", "C:\\foo\\bar.png"))
        self.assertIsNotNone(self.r("WhateverText", "D:/models/x.onnx"))
        self.assertIsNotNone(self.r("WhateverText", "x", "\\\\nas\\share"))
        self.assertIsNone(self.r("WhateverText", "GFPGAN-v1.4"))
        self.assertIsNone(self.r("WhateverText", "Single Step (Fast)"))
        self.assertIsNone(self.r("WhateverText", "128/384"))

    def test_force_include(self):
        self.assertIsNone(self.r("UseReferenceExclusivePathToggle"))

    def test_reasons_are_explicit(self):
        # each reason says why, and which rule of personal_keys.json to edit
        self.assertIn("GPU", self.r("ProvidersPrioritySelection"))
        self.assertIn("personal_keys.json 'keys'", self.r("ProvidersPrioritySelection"))
        self.assertIn("personal_keys.json 'patterns'", self.r("OutputMediaFolder"))
        self.assertIn("path", self.r("WhateverText", "C:\\foo\\bar.png"))
        g = next(k for k, e in SCHEMA.entries.items() if e.layout == "SETTINGS_LAYOUT_DATA" and e.group == "Appearance")
        self.assertIn("'Appearance'", self.r(g))
        # a plain list of keys still loads
        rules = merge.PersonalRules(keys=["X"])
        self.assertIn("personal_keys.json 'keys'", rules.reason("X"))

    def test_swap_settings_imported(self):
        swap = [k for k, e in SCHEMA.entries.items() if e.group == "Swap settings"]
        self.assertTrue(swap)
        self.assertTrue(any(self.r(k) is None for k in swap))
        self.assertTrue(all(self.r(k) is None for k in SCHEMA.keys("parameter") if "Path" not in k))


class PlanTests(unittest.TestCase):
    def test_plan_categories(self):
        current = dict(CTL_DEFAULTS)
        current["OutputMediaFolder"] = "C:/my/out"
        current["ProvidersPrioritySelection"] = "CUDA"  # computed default in the layout
        incoming = dict(current)
        incoming["FaceDetailerEnableToggle"] = not current["FaceDetailerEnableToggle"]  # changed
        incoming["ProvidersPrioritySelection"] = "CPU"  # personal key
        incoming["OutputMediaFolder"] = "D:/their/out"  # personal (pattern)
        incoming["ObsoleteOldKeyToggle"] = True  # unknown
        incoming["FaceDetailerCropFactorDecimalSlider"] = 42.0  # invalid (> max)
        missing_key = "FaceDetailerBlendSlider"
        del incoming[missing_key]
        plan = merge.plan_section(incoming, current, "control", SCHEMA, RULES, CTL_DEFAULTS)
        self.assertIn("FaceDetailerEnableToggle", plan.changed)
        self.assertIn("ProvidersPrioritySelection", plan.excluded)
        self.assertIn("OutputMediaFolder", plan.excluded)
        self.assertIn("ObsoleteOldKeyToggle", plan.unknown)
        self.assertIn("FaceDetailerCropFactorDecimalSlider", plan.invalid)
        self.assertIn(missing_key, plan.missing)
        self.assertEqual(set(plan.updates), {"FaceDetailerEnableToggle"})

    def test_missing_default_policy(self):
        current = dict(CTL_DEFAULTS)
        current["FaceDetailerBlendSlider"] = 50
        incoming = {k: v for k, v in current.items() if k != "FaceDetailerBlendSlider"}
        keep = merge.plan_section(incoming, current, "control", SCHEMA, RULES, CTL_DEFAULTS, "keep")
        self.assertEqual(keep.updates, {})
        dflt = merge.plan_section(incoming, current, "control", SCHEMA, RULES, CTL_DEFAULTS, "default")
        self.assertEqual(dflt.updates, {"FaceDetailerBlendSlider": CTL_DEFAULTS["FaceDetailerBlendSlider"]})
        # personal keys are never reset to defaults either
        current["nThreadsSlider"] = 7
        incoming.pop("nThreadsSlider", None)
        dflt = merge.plan_section(incoming, current, "control", SCHEMA, RULES, CTL_DEFAULTS, "default")
        self.assertNotIn("nThreadsSlider", dflt.updates)

    def test_numeric_equality(self):
        self.assertTrue(schema_mod.values_equal(1, 1.0))
        self.assertFalse(schema_mod.values_equal(True, 1))
        self.assertFalse(schema_mod.values_equal("1", "1.0"))


class FormatTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def test_preset_format(self):
        p = write_json(self.tmp, "p.json", {"parameters": {"SwapperResSelection": "256"}, "control": {"nThreadsSlider": 3}})
        cfg = merge.parse_config(p, SCHEMA)
        self.assertEqual(cfg.fmt, "preset")
        self.assertEqual(cfg.parameters, {"SwapperResSelection": "256"})
        self.assertEqual(cfg.control, {"nThreadsSlider": 3})

    def test_presets_folder_format(self):
        params = {k: v for k, v in list(PARAM_DEFAULTS.items())[:20]}
        p = write_json(self.tmp, "mine.json", params)
        write_json(self.tmp, "mine_ctl.json", {"FaceDetailerEnableToggle": True})
        cfg = merge.parse_config(p, SCHEMA)
        self.assertEqual(cfg.fmt, "preset_folder")
        self.assertEqual(cfg.control, {"FaceDetailerEnableToggle": True})

    def test_workspace_format(self):
        f1 = dict(PARAM_DEFAULTS)
        f2 = dict(PARAM_DEFAULTS)
        some = next(k for k in SCHEMA.keys("parameter") if SCHEMA.get(k).kind == "toggle" and k in f2)
        f2[some] = not f2[some]
        ws = fake_workspace(dict(CTL_DEFAULTS), {"a": f1, "b": f2}, selected="b")
        p = write_json(self.tmp, "last_workspace.json", ws)
        cfg = merge.parse_config(p, SCHEMA)
        self.assertEqual(cfg.fmt, "workspace")
        self.assertEqual(cfg.source_face, "b")
        self.assertEqual(cfg.parameters[some], f2[some])
        cfg = merge.parse_config(p, SCHEMA, source_face="a")
        self.assertEqual(cfg.parameters[some], f1[some])
        with self.assertRaises(ValueError):
            merge.parse_config(p, SCHEMA, source_face="zzz")

    def test_merge_into_workspace(self):
        mine_ctl = dict(CTL_DEFAULTS)
        mine_ctl["OutputMediaFolder"] = "C:/mine"
        mine_ctl["nThreadsSlider"] = 2
        some = next(k for k in SCHEMA.keys("parameter") if SCHEMA.get(k).kind == "toggle" and k in PARAM_DEFAULTS)
        mine = fake_workspace(mine_ctl, {"f1": dict(PARAM_DEFAULTS), "f2": dict(PARAM_DEFAULTS)}, selected="f1")
        theirs_ctl = dict(CTL_DEFAULTS)
        theirs_ctl["OutputMediaFolder"] = "E:/theirs"
        theirs_ctl["nThreadsSlider"] = 8
        theirs_ctl["FaceDetailerEnableToggle"] = not CTL_DEFAULTS["FaceDetailerEnableToggle"]
        tp = dict(PARAM_DEFAULTS)
        tp[some] = not tp[some]
        theirs = write_json(self.tmp, "theirs.json", fake_workspace(theirs_ctl, {"zz": tp}, selected="zz"))
        imported = merge.parse_config(theirs, SCHEMA)
        snapshot = copy.deepcopy(mine)
        new, report = merge.merge_into_workspace(mine, imported, SCHEMA, RULES, faces="selected")
        self.assertEqual(mine, snapshot, "input workspace must not be modified")
        self.assertEqual(new["control"]["OutputMediaFolder"], "C:/mine")
        self.assertEqual(new["control"]["nThreadsSlider"], 2)
        self.assertEqual(new["control"]["FaceDetailerEnableToggle"], theirs_ctl["FaceDetailerEnableToggle"])
        self.assertEqual(new["target_faces_data"]["f1"]["parameters"][some], tp[some])
        self.assertEqual(new["target_faces_data"]["f2"]["parameters"][some], PARAM_DEFAULTS[some])
        # media, faces, markers untouched
        for k in ("target_medias_data", "input_faces_data", "markers", "window_state_data", "selected_media_id"):
            self.assertEqual(new[k], mine[k])
        new_all, _ = merge.merge_into_workspace(mine, imported, SCHEMA, RULES, faces="all")
        self.assertEqual(new_all["target_faces_data"]["f2"]["parameters"][some], tp[some])
        new_nc, _ = merge.merge_into_workspace(mine, imported, SCHEMA, RULES, faces="all", do_control=False)
        self.assertEqual(new_nc["control"], mine["control"])


def _real_files():
    files = []
    if FIXTURES.is_dir():
        files += sorted(FIXTURES.glob("*.json"))
    for var in ("VMF_TEST_WORKSPACE", "VMF_TEST_PRESET"):
        if os.environ.get(var):
            files.append(Path(os.environ[var]))
    return files


class RealFilesTests(unittest.TestCase):
    """Runs on real VisoMaster JSON files when available."""

    def test_real_files(self):
        files = _real_files()
        if not files:
            self.skipTest("no real JSON (tests/fixtures or VMF_TEST_WORKSPACE / VMF_TEST_PRESET)")
        for f in files:
            with self.subTest(file=f.name):
                if f.name.endswith("_ctl.json"):
                    continue
                cfg = merge.parse_config(f, SCHEMA)
                self.assertIn(cfg.fmt, ("preset", "workspace", "preset_folder", "flat_params", "flat_control"))
                # Applying a file onto itself changes nothing and never touches personal keys.
                if cfg.control:
                    plan = merge.plan_section(cfg.control, dict(cfg.control), "control", SCHEMA, RULES)
                    self.assertEqual(plan.changed, {})
                    self.assertFalse(plan.unknown)
                    excluded_any = [k for k in cfg.control if RULES.reason(k, SCHEMA, cfg.control[k])]
                    self.assertTrue(excluded_any, "a real control always contains personal keys")
                    # their values vs. modified copy of mine: personal keys stay mine
                    mine = dict(cfg.control)
                    for k in excluded_any:
                        mine[k] = "__mine__"
                    plan = merge.plan_section(cfg.control, mine, "control", SCHEMA, RULES)
                    self.assertEqual(set(plan.excluded) & set(plan.changed), set())
                    self.assertTrue(set(excluded_any) <= set(plan.excluded))
                if cfg.parameters:
                    plan = merge.plan_section(cfg.parameters, dict(cfg.parameters), "parameter", SCHEMA, RULES)
                    self.assertEqual(plan.changed, {})
                if cfg.fmt == "workspace":
                    ws = json.loads(f.read_text(encoding="utf-8"))
                    new, _ = merge.merge_into_workspace(ws, cfg, SCHEMA, RULES, faces="all")
                    self.assertEqual(new["control"], ws["control"])


if __name__ == "__main__":
    unittest.main()
