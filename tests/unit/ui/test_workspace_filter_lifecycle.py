"""Native Qt failures must fail a child process, rather than abort pytest."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "scenario",
    [
        "empty_workspace",
        "populated_workspace",
        "active_worker_tabs",
        "stale_results",
        "shutdown",
    ],
)
def test_workspace_filter_lifecycle(scenario):
    root = Path(__file__).resolve().parents[3]
    child = Path(__file__).with_name("workspace_filter_scenarios.py")
    environment = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    result = subprocess.run(
        [sys.executable, "-u", str(child), scenario],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "QThread: Destroyed" not in result.stderr
    assert f"PASS {scenario}" in result.stdout
