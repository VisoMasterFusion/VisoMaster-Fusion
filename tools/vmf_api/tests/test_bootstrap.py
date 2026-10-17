"""Bootstrap tests: inert for any Python, active only for the launcher / main.py + VMF_API=1.

Each case runs a real interpreter (the portable one if present, else the current
one). The .pth is emulated with a sitecustomize.py on PYTHONPATH (same moment
of the start-up sequence: site initialisation, before sys.argv is final).
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from unittest import mock
from pathlib import Path

HERE = Path(__file__).resolve().parent
VMF = HERE.parent
sys.path.insert(0, str(VMF))
from vmf_common import BASE_DIR  # noqa: E402

PORTABLE_PY = BASE_DIR / "portable-files" / "python" / "python.exe"
PY = str(PORTABLE_PY) if PORTABLE_PY.is_file() else sys.executable

FAKE_HOOKS = textwrap.dedent(
    """
    import os
    def install(kind, vmf_dir):
        with open(os.environ["VMF_TEST_MARK"], "w") as f:
            f.write(kind)
    """
)


class BootstrapDetectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="vmf_boot_"))
        site = cls.tmp / "site"
        site.mkdir()
        shutil.copy(VMF / "vmf_api_bootstrap.py", site / "vmf_api_bootstrap.py")
        (site / "sitecustomize.py").write_text("import vmf_api_bootstrap\n", encoding="utf-8")
        cls.home = cls.tmp / "vmf_api"
        cls.home.mkdir()
        (cls.home / "vmf_hooks.py").write_text(FAKE_HOOKS, encoding="utf-8")
        # fake repo
        cls.repo = cls.tmp / "VisoMaster-Fusion"
        (cls.repo / "app" / "ui" / "launcher").mkdir(parents=True)
        for p in ("app/__init__.py", "app/ui/__init__.py", "app/ui/launcher/__init__.py",
                  "app/ui/launcher/launcher_window.py", "app/ui/main_ui.py"):
            (cls.repo / p).write_text("", encoding="utf-8")
        (cls.repo / "app" / "ui" / "launcher" / "__main__.py").write_text("print('launcher ran')\n", encoding="utf-8")
        (cls.repo / "main.py").write_text("print('main ran')\n", encoding="utf-8")
        (cls.repo / "launcher.py").write_text("print('shim ran')\n", encoding="utf-8")
        other = cls.tmp / "other"
        other.mkdir()
        (other / "main.py").write_text("print('other main')\n", encoding="utf-8")
        (cls.tmp / "download_models.py").write_text("print('dl')\n", encoding="utf-8")
        cls.site = site

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def run_py(self, args, vmf_api=None, cwd=None):
        mark = self.tmp / "mark.txt"
        if mark.exists():
            mark.unlink()
        env = {k: v for k, v in os.environ.items() if k not in ("VMF_API", "PYTHONPATH", "PYTHONHOME")}
        env["PYTHONPATH"] = str(self.site)
        env["VMF_API_HOME"] = str(self.home)
        env["VMF_TEST_MARK"] = str(mark)
        if vmf_api is not None:
            env["VMF_API"] = vmf_api
        r = subprocess.run([PY, *args], cwd=str(cwd or self.tmp), env=env, capture_output=True, text=True, timeout=120)
        self.assertEqual(r.returncode, 0, r.stderr)
        return (mark.read_text() if mark.exists() else None), r.stdout

    def test_plain_python_inert(self):
        self.assertIsNone(self.run_py(["-c", "pass"])[0])
        self.assertIsNone(self.run_py(["-c", "pass"], vmf_api="1")[0])

    def test_other_scripts_inert(self):
        self.assertIsNone(self.run_py([str(self.tmp / "download_models.py")], vmf_api="1")[0])
        self.assertIsNone(self.run_py([str(self.tmp / "other" / "main.py")], vmf_api="1")[0])
        self.assertIsNone(self.run_py(["-m", "pip", "--version"])[0])

    def test_launcher_module(self):
        kind, out = self.run_py(["-X", "utf8", "-m", "app.ui.launcher"], cwd=self.repo)
        self.assertEqual(kind, "launcher")
        self.assertIn("launcher ran", out)

    def test_launcher_shim(self):
        self.assertEqual(self.run_py([str(self.repo / "launcher.py")])[0], "launcher")

    def test_app_needs_flag(self):
        self.assertIsNone(self.run_py([str(self.repo / "main.py")], cwd=self.repo)[0])
        self.assertIsNone(self.run_py([str(self.repo / "main.py")], vmf_api="0", cwd=self.repo)[0])
        kind, out = self.run_py([str(self.repo / "main.py")], vmf_api="1", cwd=self.repo)
        self.assertEqual(kind, "app")
        self.assertIn("main ran", out)

    def test_inert_does_not_import(self):
        _, out = self.run_py(["-c", "import sys; print(sorted(m for m in sys.modules if m.startswith('vmf')))"])
        self.assertEqual(out.strip(), "['vmf_api_bootstrap']")

    def test_missing_vmf_dir_is_harmless(self):
        env_home = self.home
        try:
            self.__class__.home = self.tmp / "nope"
            kind, out = self.run_py([str(self.repo / "main.py")], vmf_api="1", cwd=self.repo)
            self.assertIsNone(kind)
            self.assertIn("main ran", out)
            self.assertIn("[VMF-API]", out)
        finally:
            self.__class__.home = env_home


class HookPatchTests(unittest.TestCase):
    """Real vmf_hooks against fake launcher/app modules (no Qt window)."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="vmf_hook_"))
        cls.home = cls.tmp / "vmf_api"
        cls.home.mkdir()
        for f in ("vmf_hooks.py", "vmf_common.py"):
            shutil.copy(VMF / f, cls.home / f)
        cls.repo = cls.tmp / "VisoMaster-Fusion"
        (cls.repo / "app" / "ui" / "launcher").mkdir(parents=True)
        for p in ("app/__init__.py", "app/ui/__init__.py", "app/ui/launcher/__init__.py"):
            (cls.repo / p).write_text("", encoding="utf-8")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def run_with(self, launcher_src, ticked, saved_enabled=True):
        # saved_enabled: an old config.json with "enabled": true must not tick the box any more.
        (self.repo / "app" / "ui" / "launcher" / "launcher_window.py").write_text(launcher_src, encoding="utf-8")
        (self.home / "config.json").write_text(json.dumps({"enabled": saved_enabled, "port": 8799}), encoding="utf-8")
        code = textwrap.dedent(
            f"""
            import sys
            sys.path.insert(0, {str(self.home)!r}); sys.path.insert(0, {str(self.repo)!r})
            import vmf_hooks; vmf_hooks.install("launcher")
            from app.ui.launcher import launcher_window as lw
            w = lw.LauncherWindow()
            if {ticked!r}:
                vmf_hooks._set_api_ticked(True)  # what a click on the checkbox does
            r = w.on_launch() if hasattr(w, "on_launch") else None
            print("RESULT", r)
            """
        )
        env = {k: v for k, v in os.environ.items() if k not in ("VMF_API", "VMF_API_PORT")}
        env["VMF_API_HOME"] = str(self.home)
        r = subprocess.run([PY, "-c", code], env=env, capture_output=True, text=True, timeout=120)
        self.assertEqual(r.returncode, 0, r.stderr)
        return r.stdout + r.stderr  # the hooks log to stderr

    LAUNCHER = textwrap.dedent(
        """
        import os
        class LauncherWindow:
            def __init__(self):
                self.built = 0
                self._build_home_page()
            def _build_home_page(self):
                self.built += 1
            def on_launch(self):
                return (os.environ.get("VMF_API"), os.environ.get("VMF_API_PORT"))
        """
    )

    def test_env_set_when_ticked(self):
        out = self.run_with(self.LAUNCHER, True)
        self.assertIn("RESULT ('1', '8799')", out)
        self.assertIn("launcher hook installed", out)

    def test_unticked_at_each_start(self):
        # Not ticked in this launcher session: no API, even if an old config.json says "enabled": true.
        self.assertIn("RESULT (None, None)", self.run_with(self.LAUNCHER, False, saved_enabled=True))
        self.assertIn("RESULT (None, None)", self.run_with(self.LAUNCHER, False, saved_enabled=False))

    def test_missing_hook_point(self):
        out = self.run_with("class LauncherWindow:\n    pass\n", True)
        self.assertIn("launcher hook unavailable", out)
        self.assertIn("RESULT None", out)


class LayoutTests(unittest.TestCase):
    """vmf_api next to the checkout (portable layout) or in its tools folder."""

    def _tree(self, beside: bool, inside: bool) -> Path:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        base = Path(tmp.name)
        repo = base / "VisoMaster-Fusion"
        (repo / "app").mkdir(parents=True)
        (repo / "main.py").write_text("")
        for folder, wanted in ((base / "vmf_api", beside), (repo / "tools" / "vmf_api", inside)):
            if wanted:
                folder.mkdir(parents=True)
                (folder / "vmf_hooks.py").write_text("")
        return base

    def _vmf_dir(self, base: Path) -> str:
        import importlib.util

        # This folder's file, not the copy the portable Python already imported from site-packages.
        spec = importlib.util.spec_from_file_location("vmf_api_bootstrap_under_test", VMF / "vmf_api_bootstrap.py")
        vmf_api_bootstrap = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(vmf_api_bootstrap)
        exe = str(base / "portable-files" / "python" / "python.exe")
        env = {k: v for k, v in os.environ.items() if k != "VMF_API_HOME"}
        with mock.patch.object(sys, "executable", exe), mock.patch.dict(os.environ, env, clear=True):
            return vmf_api_bootstrap._vmf_dir(str(base / "VisoMaster-Fusion"))

    def test_inside_repository(self):
        base = self._tree(beside=False, inside=True)
        self.assertEqual(Path(self._vmf_dir(base)), base / "VisoMaster-Fusion" / "tools" / "vmf_api")

    def test_beside_wins(self):
        base = self._tree(beside=True, inside=True)
        self.assertEqual(Path(self._vmf_dir(base)), base / "vmf_api")

    def test_common_layout(self):
        import vmf_common

        base = self._tree(beside=False, inside=True)
        repo = base / "VisoMaster-Fusion"
        self.assertEqual(vmf_common._layout(repo / "tools" / "vmf_api"), (base, repo))
        self.assertEqual(vmf_common._layout(base / "vmf_api"), (base, repo))


if __name__ == "__main__":
    unittest.main()
