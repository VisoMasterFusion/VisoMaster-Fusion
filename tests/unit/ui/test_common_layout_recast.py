"""Schema tests for the PerformRecast "Recast" widgets in COMMON_LAYOUT_DATA.

Pure data-validation — no Qt/GPU. control_actions is stubbed so the module
imports without PySide6.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock


def _stub_module(name: str) -> MagicMock:
    mod = MagicMock()
    mod.__name__ = name
    mod.__spec__ = None
    return mod


for _mod_name in [
    "PySide6",
    "PySide6.QtWidgets",
    "PySide6.QtCore",
    "PySide6.QtGui",
    "app.ui.widgets.actions.control_actions",
]:
    if _mod_name not in sys.modules:
        sys.modules[_mod_name] = _stub_module(_mod_name)

try:
    import cv2  # noqa: F401
except ImportError:
    sys.modules["cv2"] = MagicMock()

# common_layout_data uses `import app.ui.widgets.actions.control_actions as ...`.
# With the leaf stubbed but the parent namespace package not yet imported, that
# dotted form fails to resolve `actions`. Import the real (cheap, __init__-less)
# namespace package first so the stubbed leaf binds correctly.
import app.ui.widgets.actions  # noqa: E402,F401

from app.ui.widgets.common_layout_data import COMMON_LAYOUT_DATA  # noqa: E402

FACE_EXPR = COMMON_LAYOUT_DATA["Face expressions"]
RECAST_WIDGETS = [
    "RecastModeSelection",
    "RecastExpressionFactorDecimalSlider",
    "RecastAnimationRegionSelection",
]


def test_recast_is_a_mode_option():
    assert "Recast" in FACE_EXPR["FaceExpressionModeSelection"]["options"]


def test_recast_widgets_exist():
    for name in RECAST_WIDGETS:
        assert name in FACE_EXPR, f"{name} missing from Face expressions"


def test_recast_widgets_gated_on_recast_selection():
    for name in RECAST_WIDGETS:
        entry = FACE_EXPR[name]
        assert entry.get("parentSelection") == "FaceExpressionModeSelection"
        assert entry.get("requiredSelectionValue") == "Recast"
        assert entry.get("parentToggle") == "FaceExpressionEnableBothToggle"
        assert entry.get("requiredToggleValue") is True


def test_recast_mode_options_and_default():
    entry = FACE_EXPR["RecastModeSelection"]
    assert entry["options"] == ["Enhancement", "Replacement"]
    assert entry["default"] in entry["options"]


def test_recast_region_options_and_default():
    entry = FACE_EXPR["RecastAnimationRegionSelection"]
    assert entry["options"] == ["all", "eyes", "lips"]
    assert entry["default"] == "all"


def test_recast_expression_factor_range():
    entry = FACE_EXPR["RecastExpressionFactorDecimalSlider"]
    assert float(entry["min_value"]) == 0.0
    # Range was raised to 3.0 to allow stylization/exaggeration.
    assert float(entry["max_value"]) == 3.0
    assert float(entry["default"]) == 1.0


def test_recast_crop_scale_defaults_to_safe_framing():
    entry = FACE_EXPR["RecastCropScaleDecimalSlider"]
    # Default matches the proven 2.3 framing (best viable similarity). Too-tight
    # crops drive the generator into black frames (handled by the guard); wider
    # is safer. Default must stay within range.
    assert float(entry["default"]) == 2.3
    assert (
        float(entry["min_value"])
        <= float(entry["default"])
        <= float(entry["max_value"])
    )


def test_recast_blend_weight_defaults_match_upstream():
    eye = FACE_EXPR["RecastEyeDrivingWeightDecimalSlider"]
    lip = FACE_EXPR["RecastLipDrivingWeightDecimalSlider"]
    assert float(eye["default"]) == 0.7
    assert float(lip["default"]) == 0.8
    for entry in (eye, lip):
        assert float(entry["min_value"]) == 0.0
        assert float(entry["max_value"]) == 1.0


def test_recast_smoothing_widgets_exist_and_default_off():
    toggle = FACE_EXPR["RecastExpressionSmoothToggle"]
    assert toggle["default"] is False
    strength = FACE_EXPR["RecastSmoothStrengthDecimalSlider"]
    # Strength is additionally gated on the smoothing toggle.
    assert "RecastExpressionSmoothToggle" in strength["parentToggle"]


def test_recast_paste_back_feather_defaults_off():
    entry = FACE_EXPR["RecastPasteBackFeatherDecimalSlider"]
    assert float(entry["default"]) == 0.0
