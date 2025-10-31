# __main__.py
# ---------------------------------------------------------------------------
# VisoMaster Fusion Launcher Entrypoint
# ---------------------------------------------------------------------------
# Runs the VisoMaster Fusion Launcher GUI via:
#   python -m app.ui.launcher
# ---------------------------------------------------------------------------

import sys

try:
    from .main import main
    print("[Launcher] Starting VisoMaster Fusion Launcher...")
    main()
except Exception as e:
    print(f"[Launcher] Failed to start the VisoMaster Fusion Launcher: {e}")
    sys.exit(1)
