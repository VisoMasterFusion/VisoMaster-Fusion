"""Opt VisoMaster out of Windows power throttling (EcoQoS).

Windows 11 throttles a process whose window is in the background, minimized or covered: its
threads may run on efficiency settings and its timer-resolution requests are ignored, so every
short sleep / timed wait lasts ~15.6 ms instead of ~1 ms. VisoMaster's frame pipeline is full of
such waits, so a recording started by the API while the window is not in front ran ~40 % slower
than the same recording started by a click. Opting out keeps the speed independent of the window.
"""

from __future__ import annotations

import os
import sys

_PROCESS_POWER_THROTTLING = 4  # PROCESS_INFORMATION_CLASS.ProcessPowerThrottling
_EXECUTION_SPEED = 0x1
_IGNORE_TIMER_RESOLUTION = 0x4
_PROCESS_SET_INFORMATION = 0x0200
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000


def disable_power_throttling(pid: int | None = None) -> dict:
    """ControlMask = the two features, StateMask = 0: never throttled, timer requests honoured.
    pid=None: the current process. Returns {"applied": bool, ...}; never raises."""
    if sys.platform != "win32":
        return {"applied": False, "reason": "not Windows"}
    try:
        import ctypes
        from ctypes import wintypes

        class _State(ctypes.Structure):
            _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG), ("StateMask", wintypes.ULONG)]

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.OpenProcess.restype = wintypes.HANDLE
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        own = pid is None or pid == os.getpid()
        h = k32.GetCurrentProcess() if own else k32.OpenProcess(
            _PROCESS_SET_INFORMATION | _PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
        if not h:
            return {"applied": False, "reason": f"OpenProcess failed (error {ctypes.get_last_error()})"}
        try:
            mask = _EXECUTION_SPEED | _IGNORE_TIMER_RESOLUTION
            state = _State(1, mask, 0)
            ok = k32.SetProcessInformation(wintypes.HANDLE(h), _PROCESS_POWER_THROTTLING,
                                           ctypes.byref(state), ctypes.sizeof(state))
            if not ok:
                return {"applied": False, "reason": f"SetProcessInformation failed (error {ctypes.get_last_error()})"}
            return {"applied": True, "pid": pid or os.getpid()}
        finally:
            if not own:
                k32.CloseHandle(wintypes.HANDLE(h))
    except Exception as e:  # never break VisoMaster's startup
        return {"applied": False, "reason": f"{type(e).__name__}: {e}"}
