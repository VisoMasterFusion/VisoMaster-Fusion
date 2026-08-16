"""Optional Lossless Scaling / LSFG companion bridge.

This module deliberately does *not* reimplement or reverse-engineer LSFG. Lossless
Scaling owns its frame-generation engine and, as of the current public releases,
does not expose a public Python/SDK interface for feeding frames to LSFG.

The bridge provides the safe integration points Fusion can use today:
- discover a Steam-installed Lossless Scaling executable on Windows;
- start/stop the companion process;
- find the VisoMaster window so the user can select it in Lossless Scaling;
- calculate a sensible base-FPS target for a requested refresh rate/multiplier.

The actual capture/interpolation remains inside Lossless Scaling.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


LOSSLESS_SCALING_EXE = "LosslessScaling.exe"


@dataclass(frozen=True)
class LSFGStatus:
    supported_platform: bool
    executable: Path | None
    running: bool
    visomaster_window_found: bool


class LosslessScalingBridge:
    """Windows companion-process bridge for Lossless Scaling."""

    def __init__(self, executable: str | os.PathLike[str] | None = None) -> None:
        self._explicit_executable = Path(executable) if executable else None
        self._process: subprocess.Popen[bytes] | None = None

    @staticmethod
    def supported_platform() -> bool:
        return sys.platform == "win32"

    @staticmethod
    def _candidate_roots() -> Iterable[Path]:
        env_roots = (
            os.environ.get("ProgramFiles"),
            os.environ.get("ProgramFiles(x86)"),
            os.environ.get("LOCALAPPDATA"),
        )
        for root in env_roots:
            if root:
                yield Path(root)

        # Common Steam library locations. The recursive scan is intentionally
        # limited to the library root rather than the entire system drive.
        for root in LosslessScalingBridge._candidate_roots():
            yield root / "Steam" / "steamapps" / "common" / "Lossless Scaling"
            yield root / "Steam" / "steamapps" / "common"

    def find_executable(self) -> Path | None:
        if not self.supported_platform():
            return None

        if self._explicit_executable:
            path = self._explicit_executable.expanduser().resolve()
            return path if path.is_file() else None

        # Fast paths first.
        for root in self._candidate_roots():
            direct = root / LOSSLESS_SCALING_EXE
            if direct.is_file():
                return direct

        # Steam libraries are often installed on another drive. If Steam's
        # registry entry is available, inspect its configured library folders.
        try:
            import winreg  # type: ignore

            steam_install = None
            for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
                for key_name in (
                    r"Software\Valve\Steam",
                    r"Software\WOW6432Node\Valve\Steam",
                ):
                    try:
                        with winreg.OpenKey(hive, key_name) as key:
                            steam_install = winreg.QueryValueEx(key, "InstallPath")[0]
                            if steam_install:
                                break
                    except OSError:
                        continue
                if steam_install:
                    break

            if steam_install:
                steam_root = Path(steam_install)
                library_files = [
                    steam_root / "steamapps" / "libraryfolders.vdf",
                ]
                for library_file in library_files:
                    if not library_file.is_file():
                        continue
                    text = library_file.read_text(encoding="utf-8", errors="ignore")
                    for line in text.splitlines():
                        if '"path"' not in line.lower():
                            continue
                        value = line.split('"')[-2].strip()
                        if value:
                            candidate = Path(value) / "steamapps" / "common" / "Lossless Scaling" / LOSSLESS_SCALING_EXE
                            if candidate.is_file():
                                return candidate
        except Exception:
            # Discovery is best-effort; never break VisoMaster startup.
            pass

        return None

    @staticmethod
    def _process_running() -> bool:
        if not LosslessScalingBridge.supported_platform():
            return False
        try:
            import ctypes
            from ctypes import wintypes

            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            snapshot = ctypes.windll.kernel32.CreateToolhelp32Snapshot(0x00000002, 0)
            if snapshot in (0, -1):
                return False

            class PROCESSENTRY32W(ctypes.Structure):
                _fields_ = [
                    ("dwSize", wintypes.DWORD),
                    ("cntUsage", wintypes.DWORD),
                    ("th32ProcessID", wintypes.DWORD),
                    ("th32DefaultHeapID", ctypes.c_void_p),
                    ("th32ModuleID", wintypes.DWORD),
                    ("cntThreads", wintypes.DWORD),
                    ("th32ParentProcessID", wintypes.DWORD),
                    ("pcPriClassBase", ctypes.c_long),
                    ("dwFlags", wintypes.DWORD),
                    ("szExeFile", wintypes.WCHAR * 260),
                ]

            entry = PROCESSENTRY32W()
            entry.dwSize = ctypes.sizeof(PROCESSENTRY32W)
            first = ctypes.windll.kernel32.Process32FirstW(snapshot, ctypes.byref(entry))
            if not first:
                ctypes.windll.kernel32.CloseHandle(snapshot)
                return False
            try:
                while True:
                    if entry.szExeFile.lower() == LOSSLESS_SCALING_EXE.lower():
                        return True
                    if not ctypes.windll.kernel32.Process32NextW(snapshot, ctypes.byref(entry)):
                        break
            finally:
                ctypes.windll.kernel32.CloseHandle(snapshot)
        except Exception:
            return False
        return False

    @staticmethod
    def visomaster_window_found() -> bool:
        """Return True when a visible window containing 'VisoMaster' exists."""
        if not LosslessScalingBridge.supported_platform():
            return False
        try:
            import ctypes
            from ctypes import wintypes

            found = False

            @ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
            def enum_proc(hwnd: int, _lparam: int) -> bool:
                nonlocal found
                if not ctypes.windll.user32.IsWindowVisible(hwnd):
                    return True
                length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
                if length <= 0:
                    return True
                buffer = ctypes.create_unicode_buffer(length + 1)
                ctypes.windll.user32.GetWindowTextW(hwnd, buffer, length + 1)
                if "visomaster" in buffer.value.lower():
                    found = True
                    return False
                return True

            ctypes.windll.user32.EnumWindows(enum_proc, 0)
            return found
        except Exception:
            return False

    def launch(self) -> Path | None:
        """Start Lossless Scaling if it can be found; return its executable path."""
        executable = self.find_executable()
        if executable is None:
            return None
        if self._process is not None and self._process.poll() is None:
            return executable
        self._process = subprocess.Popen(
            [str(executable)],
            cwd=str(executable.parent),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return executable

    def status(self) -> LSFGStatus:
        return LSFGStatus(
            supported_platform=self.supported_platform(),
            executable=self.find_executable(),
            running=self._process_running(),
            visomaster_window_found=self.visomaster_window_found(),
        )

    @staticmethod
    def recommended_base_fps(refresh_rate: float, multiplier: float = 2.0) -> int:
        """Return a conservative integer base FPS for fixed-multiplier LSFG."""
        if refresh_rate <= 0 or multiplier <= 0:
            raise ValueError("refresh_rate and multiplier must be positive")
        return max(10, int(refresh_rate / multiplier))


def get_lsfg_status() -> LSFGStatus:
    return LosslessScalingBridge().status()
