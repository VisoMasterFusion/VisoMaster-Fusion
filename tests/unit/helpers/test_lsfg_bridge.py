from pathlib import Path

from app.helpers.lsfg_bridge import LosslessScalingBridge


def test_recommended_base_fps():
    assert LosslessScalingBridge.recommended_base_fps(120, 2) == 60
    assert LosslessScalingBridge.recommended_base_fps(144, 3) == 48


def test_recommended_base_fps_rejects_invalid_values():
    for refresh_rate, multiplier in ((0, 2), (120, 0), (-1, 2)):
        try:
            LosslessScalingBridge.recommended_base_fps(refresh_rate, multiplier)
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError")


def test_explicit_executable(tmp_path: Path):
    executable = tmp_path / "LosslessScaling.exe"
    executable.write_bytes(b"")

    bridge = LosslessScalingBridge(executable)
    if bridge.supported_platform():
        assert bridge.find_executable() == executable.resolve()
    else:
        assert bridge.find_executable() is None
