"""Install / uninstall logic behind install.bat and uninstall.bat (idempotent).

Run with the portable Python:  portable-files\\python\\python.exe -I install_helper.py install|uninstall|check
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

VMF_DIR = Path(__file__).resolve().parent
# Next to the checkout (<BASE_DIR>\vmf_api) or inside it (<BASE_DIR>\VisoMaster-Fusion\tools\vmf_api),
# same rule as vmf_common._layout (not imported: this script runs with -I).
_REPO = VMF_DIR.parent.parent
BASE_DIR = _REPO.parent if VMF_DIR.parent.name.lower() == "tools" and (_REPO / "main.py").is_file() else VMF_DIR.parent
PORTABLE = BASE_DIR / "portable-files"
PY = PORTABLE / "python" / "python.exe"
UV = PORTABLE / "uv" / "uv.exe"
VENV = VMF_DIR / ".venv"
FILES = ("vmf_api_hook.pth", "vmf_api_bootstrap.py")


MCP_SPEC = "mcp>=2.3,<3"  # the version tested; a new major version may change the server API


def say(msg=""):
    print(msg, flush=True)


def probe(code: str) -> str:
    r = subprocess.run([str(PY), "-I", "-c", code], capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip() or f"code {r.returncode}")
    return r.stdout.strip()


def site_packages() -> Path:
    """site-packages of the portable Python, refusing if `import site` is disabled."""
    if not PY.is_file():
        raise SystemExit(f"[ERROR] portable Python not found: {PY}")
    has_site = probe("import sys; print('site' in sys.modules)")
    if has_site != "True":
        pth = sorted(p.name for p in PY.parent.glob("python*._pth"))
        raise SystemExit(
            "[ERROR] `import site` is disabled in the portable Python "
            f"({', '.join(pth) or '._pth file'} has no 'import site' line).\n"
            ".pth files would not be read: installation cancelled, nothing was changed.\n"
            "(Start_Portable.bat only uncomments python311._pth; with Python 3.12 the file is python312._pth.)"
        )
    cands = probe("import site; print('\\n'.join(site.getsitepackages()))").splitlines()
    for c in cands:
        if c.lower().rstrip("\\/").endswith("site-packages"):
            return Path(c)
    raise SystemExit(f"[ERROR] site-packages not found among {cands}")


def _venv_matches() -> bool:
    cfg = VENV / "pyvenv.cfg"
    try:
        for line in cfg.read_text(encoding="utf-8").splitlines():
            key, _, value = line.partition("=")
            if key.strip().lower() == "home":
                return Path(value.strip()).resolve() == PY.parent.resolve()
    except OSError:
        pass
    return False


def install():
    say(f"Base dir       : {BASE_DIR}")
    say(f"Portable Python: {PY}")
    say("Version        : " + probe("import sys; print(sys.version.split()[0])"))
    sp = site_packages()
    say(f"site-packages  : {sp}")
    for name in FILES:
        shutil.copy2(VMF_DIR / name, sp / name)
        say(f"  copied       : {name}")
    loaded = probe("import sys; print('vmf_api_bootstrap' in sys.modules)")
    if loaded != "True":
        raise SystemExit("[ERROR] the .pth file is not run at Python start-up (check site-packages).")
    say("  .pth active  : yes (inert outside launcher / main.py+VMF_API=1)")

    # venv for the MCP server (official `mcp` SDK), separate from VisoMaster's Python
    env = dict(os.environ)
    env["UV_CACHE_DIR"] = str(PORTABLE / "uv-cache")  # same as Start_Portable.bat
    venv_py = VENV / "Scripts" / "python.exe"
    if venv_py.is_file() and not _venv_matches():
        # vmf_api copied from another install: its venv points to another Python.
        say("MCP venv bound to another Python: recreating ...")
        shutil.rmtree(VENV)
    if not venv_py.is_file():
        say("Creating the MCP venv ...")
        if UV.is_file():
            subprocess.run([str(UV), "venv", "--python", str(PY), str(VENV)], check=True, env=env)
        else:
            subprocess.run([str(PY), "-m", "venv", str(VENV)], check=True)
    say("Installing the mcp SDK in the venv ...")
    if UV.is_file():
        subprocess.run([str(UV), "pip", "install", "--python", str(venv_py), "-q", MCP_SPEC], check=True, env=env)
    else:
        subprocess.run([str(venv_py), "-m", "pip", "install", "-q", MCP_SPEC], check=True)
    ver = subprocess.run(
        [str(venv_py), "-c", "import importlib.metadata as m; print(m.version('mcp'))"],
        capture_output=True, text=True,
    ).stdout.strip()
    say(f"  mcp          : {ver}")
    say()
    say("OK. In the launcher, tick \"Enable API / MCP\", then launch VisoMaster.")
    say()
    say("MCP server, for any MCP client (add a local / stdio server):")
    say(f"  command   : {venv_py}")
    say(f"  arguments : {VMF_DIR / 'vmf_cli.py'} mcp")
    say("JSON (\"mcpServers\" section of most clients):")
    cfg = {"mcpServers": {"visomaster": {"command": str(venv_py), "args": [str(VMF_DIR / "vmf_cli.py"), "mcp"]}}}
    say(json.dumps(cfg, indent=2))
    say("Other client formats: docs\\MCP.md")


def uninstall():
    if not PY.is_file():
        say(f"Portable Python not found ({PY}): nothing to remove.")
        return
    try:
        cands = probe("import site; print('\\n'.join(site.getsitepackages()))").splitlines()
    except Exception:
        cands = [str(PY.parent / "Lib" / "site-packages")]
    removed = 0
    for c in cands:
        for name in FILES:
            p = Path(c) / name
            if p.is_file():
                p.unlink()
                removed += 1
                say(f"  removed: {p}")
    say(f"{removed} file(s) removed. The vmf_api folder (config, backups, venv) is kept. Its backups\\ folder "
        f"holds workspace copies with your face crops and embeddings: delete the folder if you no longer need it.")


def check():
    sp = site_packages()
    for name in FILES:
        p = sp / name
        same = p.is_file() and p.read_bytes() == (VMF_DIR / name).read_bytes()
        say(f"{name}: {'up to date' if same else ('different' if p.is_file() else 'missing')}")


if __name__ == "__main__":
    action = sys.argv[1] if len(sys.argv) > 1 else "install"
    {"install": install, "uninstall": uninstall, "check": check}[action]()
