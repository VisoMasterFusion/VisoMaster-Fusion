"""
Shared pytest fixtures for VisoMaster-Fusion test suite.
All GPU tests are skipped by default; use -m gpu to run them explicitly.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# Make sure the project root is on sys.path so `app.*` imports resolve even
# when pytest is invoked from the project root without installing the package.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

FIXTURE_IMAGES = PROJECT_ROOT / "tests" / "fixtures" / "images"

# ---------------------------------------------------------------------------
# Synthetic image fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def face_rgb_np() -> np.ndarray:
    """512×512 synthetic face image as HWC uint8 RGB numpy array."""
    import cv2

    path = FIXTURE_IMAGES / "face_512.png"
    img_bgr = cv2.imread(str(path))
    assert img_bgr is not None, (
        f"Fixture image not found: {path}. Run tests/fixtures/generate_fixtures.py first."
    )
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


@pytest.fixture(scope="session")
def equirect_np() -> np.ndarray:
    """360×720 equirectangular image as HWC uint8 RGB numpy array."""
    import cv2

    path = FIXTURE_IMAGES / "equirect_360_720.png"
    img_bgr = cv2.imread(str(path))
    assert img_bgr is not None, (
        f"Fixture image not found: {path}. Run tests/fixtures/generate_fixtures.py first."
    )
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


@pytest.fixture(scope="session")
def equirect_half_np() -> np.ndarray:
    """180×360 single-eye equirectangular image as HWC uint8 RGB numpy array."""
    import cv2

    path = FIXTURE_IMAGES / "equirect_180_360.png"
    img_bgr = cv2.imread(str(path))
    assert img_bgr is not None, (
        f"Fixture image not found: {path}. Run tests/fixtures/generate_fixtures.py first."
    )
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


@pytest.fixture
def blank_rgb_np() -> np.ndarray:
    """256×256 blank uint8 RGB numpy array."""
    return np.zeros((256, 256, 3), dtype=np.uint8)


@pytest.fixture
def small_equirect_np() -> np.ndarray:
    """Small 90×180 equirectangular for fast unit tests (no fixture file needed)."""
    img = np.zeros((90, 180, 3), dtype=np.uint8)
    img[:, :, 0] = np.tile(np.linspace(0, 255, 180, dtype=np.uint8), (90, 1))
    img[:, :, 2] = np.tile(np.linspace(0, 255, 90, dtype=np.uint8)[:, None], (1, 180))
    return img


# ---------------------------------------------------------------------------
# CPU device fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def cpu_device() -> torch.device:
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Mock ModelsProcessor
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_models_processor(cpu_device):
    mp = MagicMock()
    mp.device = cpu_device
    mp.model_lock = threading.Lock()
    # Sensible defaults for commonly called methods
    mp.run_detect.return_value = (np.empty((0, 5), dtype=np.float32), None, None)
    mp.get_face_swapper_model.return_value = MagicMock()
    return mp


# ---------------------------------------------------------------------------
# Mock MainWindow
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_main_window(mock_models_processor):
    mw = MagicMock()
    mw.models_processor = mock_models_processor
    mw.video_processor = MagicMock()
    mw.control = {}
    mw.parameters = {}
    return mw


# ---------------------------------------------------------------------------
# Minimal control dict (safe defaults — all toggle-gated features off)
# ---------------------------------------------------------------------------


@pytest.fixture
def base_control() -> dict:
    return {
        "VR180ModeEnableToggle": False,
        "VR180EyeModeSelection": "Both Eyes",
        "DetectorModelSelection": "RetinaFace",
        "MaxFacesToDetectSlider": 10,
        "DetectorScoreSlider": 50,
        "LandmarkDetectModelSelection": "2dfan4",
        "LandmarkDetectScoreSlider": 50,
        "LandmarkMeanEyesToggle": False,
        "BordermaskEnableToggle": False,
        # Interpolation defaults
        "get_cropped_face_kpsTypeSelection": "BILINEAR",
        "original_face_128_384TypeSelection": "BILINEAR",
        "original_face_512TypeSelection": "BILINEAR",
        "UntransformTypeSelection": "BILINEAR",
        "ScalebackFrameTypeSelection": "BILINEAR",
        "expression_faceeditor_t256TypeSelection": "BILINEAR",
        "expression_faceeditor_backTypeSelection": "BILINEAR",
        "block_shiftTypeSelection": "NEAREST",
        "AntialiasTypeSelection": "False",
    }
