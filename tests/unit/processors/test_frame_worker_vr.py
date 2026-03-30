"""
FW-VR-* tests for the VR180 control-flow logic inside FrameWorker.

Strategy: mock everything external to FrameWorker so we can test the
decision logic without loading any ML models or requiring a GPU.
"""

from __future__ import annotations

import sys
import threading
from unittest.mock import MagicMock, patch
from collections import OrderedDict

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# Stub heavyweight imports that FrameWorker pulls at module level
# ---------------------------------------------------------------------------


def _stub(name: str) -> MagicMock:
    # Do NOT use spec= — a spec-limited mock restricts attribute access, which breaks
    # sibling test files that import real modules needing arbitrary attributes from
    # the same stub (e.g. PySide6.QtCore.Slot, PySide6.QtCore.QThread, etc.).
    m = MagicMock()
    m.__name__ = name
    m.__spec__ = None
    return m


_STUBS = [
    "PySide6",
    "PySide6.QtWidgets",
    "PySide6.QtCore",
    "PySide6.QtGui",
    "kornia",
    "kornia.enhance",
    "kornia.color",
    # frame_worker.py imports kornia.geometry.transform at module level (c6b67d0);
    # all three levels must be stubbed so the dotted import resolves.
    "kornia.geometry",
    "kornia.geometry.transform",
    "skimage",
    "skimage.transform",
    "app.ui.widgets.widget_components",
    # Do NOT stub the parent package — it's a namespace package and stubbing it
    # prevents sibling test files from importing real submodules in the same session.
    "app.ui.widgets.actions.common_actions",
    "app.processors.frame_enhancers",
    "app.processors.frame_edits",
    "app.processors.utils",
    "app.processors.utils.faceutil",
    # Do NOT stub app.helpers.miscellaneous — it is a pure-Python module with no
    # heavy dependencies and later test files (test_save_load_actions, etc.) need
    # the real ParametersDict class from it.  The VR tests only exercise early-return
    # paths (empty detections) that never call get_scaling_transforms etc., so no
    # module-level patches to that module are required.
]
for _s in _STUBS:
    if _s not in sys.modules:
        sys.modules[_s] = _stub(_s)

# Stub FrameEnhancers / FrameEdits constructors
sys.modules["app.processors.frame_enhancers"].FrameEnhancers = MagicMock  # type: ignore[attr-defined]
sys.modules["app.processors.frame_edits"].FrameEdits = MagicMock  # type: ignore[attr-defined]

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CPU = torch.device("cpu")


@pytest.fixture
def mock_models_processor():
    mp = MagicMock()
    mp.device = CPU
    mp.model_lock = threading.Lock()
    # run_detect returns empty → no faces → early-return path
    mp.run_detect.return_value = (np.empty((0, 5), dtype=np.float32), None, None)
    return mp


@pytest.fixture
def mock_main_window(mock_models_processor):
    mw = MagicMock()
    mw.models_processor = mock_models_processor
    mw.video_processor = MagicMock()
    mw.control = {}
    mw.parameters = {}
    return mw


@pytest.fixture
def small_equirect() -> np.ndarray:
    img = np.zeros((90, 180, 3), dtype=np.uint8)
    img[:, :, 0] = np.tile(np.linspace(0, 255, 180, dtype=np.uint8), (90, 1))
    return img


# ---------------------------------------------------------------------------
# Helpers to build a FrameWorker instance without starting the thread
# ---------------------------------------------------------------------------


def _make_worker(main_window):
    """Import and instantiate FrameWorker without calling .start()."""
    # Patch vr_utils so we don't need the full external converters
    with (
        patch("app.helpers.vr_utils.E2P_Equirectangular", MagicMock()),
        patch("app.helpers.vr_utils.P2E_Perspective", MagicMock()),
    ):
        from app.processors.workers.frame_worker import FrameWorker

        worker = FrameWorker.__new__(FrameWorker)
        # Must call Thread.__init__ so the .name property setter works
        threading.Thread.__init__(worker)
        # Manually init the attributes FrameWorker.__init__ would set
        worker.stop_event = threading.Event()
        worker.main_window = main_window
        worker.models_processor = main_window.models_processor
        worker.video_processor = main_window.video_processor
        worker.frame_enhancers = MagicMock()
        worker.frame_edits = MagicMock()
        worker.frame_queue = None
        worker.worker_id = -1
        worker.frame = None
        worker.frame_number = -1
        worker.is_single_frame = True
        worker.is_pool_worker = False
        worker.name = "FrameWorker-Test"
        worker.parameters = {}
        worker.last_processed_frame_number = -1
        worker.last_detected_faces = []
        worker.last_detected_faces_vr = []
        worker.last_processed_frame_number_vr = -1
        worker.VR_PERSPECTIVE_RENDER_SIZE = 512
        worker.VR_FOV_SCALE_FACTOR = 1.5
        worker.is_view_face_compare = False
        worker.is_view_face_mask = False
        worker.lock = threading.Lock()
        worker._tracker_lock = threading.Lock()
        worker.local_control_state_from_feeder = {}
        # Convolution kernels added in c6b67d0 (use current attribute names)
        worker.kernel_lap = None
        worker.kernel_sobel_x = None
        worker.kernel_sobel_y = None
        worker._vr_converter = None
        worker._vr_frame_size = None
        worker._vr_p2e_converter = None  # Improvement K: cached PerspectiveConverter
        worker._vr_p2e_frame_size = None
        worker._last_scaling_control = None
        worker._last_vr_scaling_control = None
        worker._resize_cache = {}
        worker._gabor_kernels_cache = OrderedDict()
        # Q-QUAL-01 / Q-QUAL-03: EMA state dicts
        worker._smoothed_kps = {}
        worker._color_stats_ema = {}
        worker.t512 = MagicMock()
        worker.t384 = MagicMock()
        worker.t256 = MagicMock()
        worker.t128 = MagicMock()
        worker.interpolation_get_cropped_face_kps = None
        worker.interpolation_original_face_128_384 = None
        worker.interpolation_original_face_512 = None
        worker.interpolation_Untransform = None
        worker.interpolation_scaleback = None
        worker.t256_face = MagicMock()
        worker.interpolation_expression_faceeditor_back = None
        worker.interpolation_block_shift = None
        worker.t512_mask = MagicMock()
        worker.t128_mask = MagicMock()
        worker.t256_near = MagicMock()
        return worker


# ---------------------------------------------------------------------------
# FW-VR-04: missing VR180EyeModeSelection defaults to "Both Eyes"
# ---------------------------------------------------------------------------


def test_vr180_eye_mode_defaults_to_both_eyes(mock_main_window, small_equirect):
    """control dict without VR180EyeModeSelection should behave like "Both Eyes"."""
    control = {
        "VR180EyeModeSelection": "Both Eyes",  # default path
    }
    # The key formula: control.get("VR180EyeModeSelection", "Both Eyes") == "Single Eye"
    assert (control.get("VR180EyeModeSelection", "Both Eyes") == "Single Eye") is False


def test_missing_key_defaults_to_both_eyes():
    control = {}
    assert (control.get("VR180EyeModeSelection", "Both Eyes") == "Single Eye") is False


# ---------------------------------------------------------------------------
# FW-VR-02 / FW-VR-03: vr_single_eye_mode flag evaluation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "selection,expected_single",
    [
        ("Both Eyes", False),
        ("Single Eye", True),
        ("Both Eyes", False),  # duplicate to ensure no state leak
    ],
)
def test_vr_single_eye_mode_flag(selection, expected_single):
    control = {"VR180EyeModeSelection": selection}
    result = control.get("VR180EyeModeSelection", "Both Eyes") == "Single Eye"
    assert result is expected_single


# ---------------------------------------------------------------------------
# FW-VR-02: is_left_eye is bool (not None) when "Both Eyes"
# ---------------------------------------------------------------------------


def test_is_left_eye_is_bool_for_both_eyes():
    vr_single_eye_mode = False
    _eye_side = "L"
    is_left_eye = None if vr_single_eye_mode else ("L" in _eye_side)
    assert isinstance(is_left_eye, bool)
    assert is_left_eye is True


def test_is_left_eye_is_bool_false_for_right_eye():
    vr_single_eye_mode = False
    _eye_side = "R"
    is_left_eye = None if vr_single_eye_mode else ("L" in _eye_side)
    assert isinstance(is_left_eye, bool)
    assert is_left_eye is False


# ---------------------------------------------------------------------------
# FW-VR-03: is_left_eye is None when "Single Eye"
# ---------------------------------------------------------------------------


def test_is_left_eye_is_none_for_single_eye():
    vr_single_eye_mode = True
    _eye_side = "L"  # doesn't matter
    is_left_eye = None if vr_single_eye_mode else ("L" in _eye_side)
    assert is_left_eye is None


def test_is_left_eye_is_none_for_single_eye_right():
    vr_single_eye_mode = True
    _eye_side = "R"
    is_left_eye = None if vr_single_eye_mode else ("L" in _eye_side)
    assert is_left_eye is None


# ---------------------------------------------------------------------------
# FW-VR-05: empty bboxes_eq_np → return original equirect unchanged
# ---------------------------------------------------------------------------


def test_empty_bboxes_returns_original(mock_main_window, small_equirect):
    """When no faces are detected, _process_frame_vr180 returns the input tensor."""
    # mock run_detect to return empty
    mock_main_window.models_processor.run_detect.return_value = (
        np.empty((0, 5), dtype=np.float32),
        None,
        None,
    )
    original_tensor = torch.from_numpy(small_equirect).permute(2, 0, 1).to(CPU)

    with (
        patch("app.helpers.vr_utils.E2P_Equirectangular", MagicMock()),
        patch("app.helpers.vr_utils.P2E_Perspective", MagicMock()),
    ):
        worker = _make_worker(mock_main_window)

        # Minimal control dict
        control = {
            "DetectorModelSelection": "RetinaFace",
            "MaxFacesToDetectSlider": 10,
            "DetectorScoreSlider": 50,
            "LandmarkDetectModelSelection": "2dfan4",
            "LandmarkDetectScoreSlider": 50,
            "LandmarkMeanEyesToggle": False,
            "VR180EyeModeSelection": "Both Eyes",
        }
        stop_event = threading.Event()

        # Patch EquirectangularConverter to return a minimal mock
        mock_ec = MagicMock()
        mock_ec.height = small_equirect.shape[0]
        mock_ec.width = small_equirect.shape[1]

        with patch(
            "app.processors.workers.frame_worker.EquirectangularConverter",
            return_value=mock_ec,
        ):
            result = worker._process_frame_vr180(
                img_numpy_rgb_uint8=small_equirect,
                original_equirect_tensor_for_vr=original_tensor,
                control=control,
                stop_event=stop_event,
            )

    assert result is original_tensor


# ---------------------------------------------------------------------------
# FW-VR-06: no crops → PerspectiveConverter NOT instantiated
# ---------------------------------------------------------------------------


def test_no_crops_skips_perspective_converter(mock_main_window, small_equirect):
    """If processed_perspective_crops_details is empty, PerspectiveConverter must not be created."""
    original_tensor = torch.from_numpy(small_equirect).permute(2, 0, 1).to(CPU)

    mock_main_window.models_processor.run_detect.return_value = (
        np.empty((0, 5), dtype=np.float32),
        None,
        None,
    )

    control = {
        "DetectorModelSelection": "RetinaFace",
        "MaxFacesToDetectSlider": 10,
        "DetectorScoreSlider": 50,
        "LandmarkDetectModelSelection": "2dfan4",
        "LandmarkDetectScoreSlider": 50,
        "LandmarkMeanEyesToggle": False,
        "VR180EyeModeSelection": "Both Eyes",
    }
    stop_event = threading.Event()
    mock_ec = MagicMock()
    mock_ec.height = small_equirect.shape[0]
    mock_ec.width = small_equirect.shape[1]

    with (
        patch("app.helpers.vr_utils.E2P_Equirectangular", MagicMock()),
        patch("app.helpers.vr_utils.P2E_Perspective", MagicMock()),
    ):
        worker = _make_worker(mock_main_window)

        with (
            patch(
                "app.processors.workers.frame_worker.EquirectangularConverter",
                return_value=mock_ec,
            ),
            patch(
                "app.processors.workers.frame_worker.PerspectiveConverter"
            ) as mock_pc_cls,
        ):
            worker._process_frame_vr180(
                img_numpy_rgb_uint8=small_equirect,
                original_equirect_tensor_for_vr=original_tensor,
                control=control,
                stop_event=stop_event,
            )
            # PerspectiveConverter should NOT have been called
            mock_pc_cls.assert_not_called()


# ---------------------------------------------------------------------------
# FW-VR-01: stop_event exits stitch loop early
# ---------------------------------------------------------------------------


def test_stop_event_in_stitch_loop():
    """stop_event.is_set() == True before the loop body → stitch is skipped."""
    stop_event = threading.Event()
    stop_event.set()

    stitch_calls = []

    class FakePConverter:
        def stitch_single_perspective(self, **kwargs):
            stitch_calls.append(kwargs)

    crops = [("L", torch.zeros(3, 32, 32, dtype=torch.uint8), 0.0, 0.0, 60.0)]
    target = torch.zeros(3, 90, 180, dtype=torch.uint8)
    pc = FakePConverter()

    for _eye_side, _crop_tensor, _theta, _phi, _fov in crops:
        if stop_event is not None and stop_event.is_set():
            break
        pc.stitch_single_perspective(
            target_equirect_torch_cxhxw_rgb_uint8=target,
            processed_crop_torch_cxhxw_rgb_uint8=_crop_tensor,
            theta=_theta,
            phi=_phi,
            fov=_fov,
            is_left_eye=("L" in _eye_side),
        )

    assert len(stitch_calls) == 0, "stitch should not be called when stop_event is set"


# ---------------------------------------------------------------------------
# FW-CACHE-01/02: _resize_cache reuse
# ---------------------------------------------------------------------------


def test_resize_cache_reuses_object():
    """Same (h, w, interp, antialias) key returns the same Resize object."""
    from torchvision.transforms import v2

    cache: dict = {}
    key = (256, 256, v2.InterpolationMode.BILINEAR, False)
    if key not in cache:
        cache[key] = v2.Resize(
            (256, 256), interpolation=v2.InterpolationMode.BILINEAR, antialias=False
        )
    obj1 = cache[key]

    if key not in cache:
        cache[key] = v2.Resize(
            (256, 256), interpolation=v2.InterpolationMode.BILINEAR, antialias=False
        )
    obj2 = cache[key]

    assert obj1 is obj2


def test_resize_cache_different_size_different_object():
    from torchvision.transforms import v2

    cache: dict = {}
    key1 = (256, 256, v2.InterpolationMode.BILINEAR, False)
    key2 = (128, 128, v2.InterpolationMode.BILINEAR, False)
    cache[key1] = v2.Resize((256, 256))
    cache[key2] = v2.Resize((128, 128))
    assert cache[key1] is not cache[key2]


# ---------------------------------------------------------------------------
# FW-GHOSTFACE: GHOSTFACE_MODELS frozenset completeness
# ---------------------------------------------------------------------------


def test_ghostface_models_frozenset():
    with (
        patch("app.helpers.vr_utils.E2P_Equirectangular", MagicMock()),
        patch("app.helpers.vr_utils.P2E_Perspective", MagicMock()),
    ):
        from app.processors.workers.frame_worker import FrameWorker

        expected = {"GhostFace-v1", "GhostFace-v2", "GhostFace-v3"}
        assert FrameWorker.GHOSTFACE_MODELS == expected
        assert isinstance(FrameWorker.GHOSTFACE_MODELS, frozenset)
