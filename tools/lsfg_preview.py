"""Small Windows helper for using Lossless Scaling LSFG with VisoMaster Fusion.

This does not inject into LSFG. It launches Lossless Scaling and reports the
recommended base FPS for the requested monitor refresh/multiplier. LS captures
the VisoMaster window itself using its normal DXGI/WGC capture path.
"""

from __future__ import annotations

import argparse
import time

from app.helpers.lsfg_bridge import LosslessScalingBridge


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare VisoMaster for Lossless Scaling LSFG"
    )
    parser.add_argument(
        "--refresh", type=float, default=60.0, help="Display refresh rate in Hz"
    )
    parser.add_argument(
        "--multiplier", type=float, default=2.0, help="LSFG fixed multiplier"
    )
    parser.add_argument(
        "--exe", default=None, help="Optional path to LosslessScaling.exe"
    )
    parser.add_argument(
        "--wait", type=float, default=0.0, help="Seconds to wait before status check"
    )
    args = parser.parse_args()

    bridge = LosslessScalingBridge(args.exe)
    executable = bridge.launch()
    if executable is None:
        print("[LSFG] Lossless Scaling was not found on this Windows installation.")
        return 2

    if args.wait > 0:
        time.sleep(args.wait)

    status = bridge.status()
    base_fps = bridge.recommended_base_fps(args.refresh, args.multiplier)

    print(f"[LSFG] Lossless Scaling: {executable}")
    print(f"[LSFG] VisoMaster window found: {status.visomaster_window_found}")
    print(
        f"[LSFG] Recommended base FPS: {base_fps} for {args.refresh:g} Hz x{args.multiplier:g}"
    )
    print(
        "[LSFG] In Lossless Scaling: select VisoMaster's window, enable LSFG, then start scaling."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
