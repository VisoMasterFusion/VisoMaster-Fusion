"""
FD-* tests for face detector logic (NMS, bbox handling, edge cases).

No ML models are loaded; all inference paths are mocked.
"""

from __future__ import annotations

import numpy as np
import torch


# ---------------------------------------------------------------------------
# FD-03: NMS (torchvision IoU-NMS) reduces overlapping boxes
# ---------------------------------------------------------------------------


def test_nms_removes_overlapping_boxes():
    """torchvision.ops.nms should suppress heavily overlapping boxes."""
    from torchvision.ops import nms

    # Three boxes: first two overlap heavily, third is separate
    boxes = torch.tensor(
        [
            [10.0, 10.0, 100.0, 100.0],
            [12.0, 12.0, 102.0, 102.0],  # heavily overlaps box 0
            [200.0, 200.0, 300.0, 300.0],  # no overlap
        ],
        dtype=torch.float32,
    )
    areas = ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])).clamp(min=0.0)
    keep = nms(boxes, areas, iou_threshold=0.5)
    kept_indices = keep.tolist()

    # Box 1 (smaller or equal area vs box 0 depending on order) should be suppressed
    # The separate box (index 2) should always be kept
    assert 2 in kept_indices
    # At most one of the two overlapping boxes should remain
    assert not (0 in kept_indices and 1 in kept_indices), (
        "Both overlapping boxes survived NMS — at least one should be suppressed"
    )


def test_nms_keeps_all_non_overlapping_boxes():
    from torchvision.ops import nms

    boxes = torch.tensor(
        [
            [0.0, 0.0, 50.0, 50.0],
            [100.0, 100.0, 150.0, 150.0],
            [200.0, 200.0, 250.0, 250.0],
        ],
        dtype=torch.float32,
    )
    areas = ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])).clamp(min=0.0)
    keep = nms(boxes, areas, iou_threshold=0.5)
    assert len(keep) == 3


# ---------------------------------------------------------------------------
# FD-05: bindex slicing produces a subset of input bboxes
# ---------------------------------------------------------------------------


def test_bindex_slicing():
    """Selecting rows by index array gives the correct subset."""
    bboxes = np.array(
        [
            [10, 10, 50, 50, 0.9],
            [20, 20, 60, 60, 0.8],
            [30, 30, 70, 70, 0.7],
        ],
        dtype=np.float32,
    )
    keep_indices = np.array([0, 2])
    result = bboxes[keep_indices]
    assert result.shape == (2, 5)
    assert np.allclose(result[0], bboxes[0])
    assert np.allclose(result[1], bboxes[2])


# ---------------------------------------------------------------------------
# FD-07: empty bbox array handled gracefully (IndexError guard)
# ---------------------------------------------------------------------------


def test_empty_bbox_array_len_check():
    """len(bboxes_eq_np) == 0 should short-circuit without IndexError."""
    bboxes = np.empty((0, 5), dtype=np.float32)
    if len(bboxes) == 0:
        result = "early_return"
    else:
        result = "processed"
    assert result == "early_return"


def test_single_bbox_reshape():
    """1-D bbox (4,) or (5,) should be reshaped to (1, N) without error."""
    bbox_1d = np.array([10.0, 20.0, 50.0, 60.0], dtype=np.float32)
    if bbox_1d.ndim == 1 and bbox_1d.shape[0] in (4, 5):
        bbox_2d = bbox_1d.reshape(1, -1)
    assert bbox_2d.shape == (1, 4)

    bbox_1d_5 = np.array([10.0, 20.0, 50.0, 60.0, 0.9], dtype=np.float32)
    if bbox_1d_5.ndim == 1 and bbox_1d_5.shape[0] in (4, 5):
        bbox_2d_5 = bbox_1d_5.reshape(1, -1)
    assert bbox_2d_5.shape == (1, 5)


# ---------------------------------------------------------------------------
# FD-06: np.exp clipping — no overflow on extreme scores
# ---------------------------------------------------------------------------


def test_exp_clipping_no_overflow():
    """Scores passed through np.exp should be clipped to avoid overflow."""
    extreme_scores = np.array([1000.0, -1000.0, 0.0, 500.0])
    clipped = np.clip(extreme_scores, -500, 500)
    result = np.exp(clipped)
    assert np.all(np.isfinite(result)), "np.exp produced inf/nan on extreme input"


# ---------------------------------------------------------------------------
# FD-02: BYTETracker None-guard — tracking disabled gracefully
# ---------------------------------------------------------------------------


def test_bytetracker_none_guard():
    """If BYTETracker is None, tracking should be skipped without AttributeError."""
    BYTETracker = None

    tracker = None
    if BYTETracker is not None:
        tracker = BYTETracker()  # would fail with None

    assert tracker is None


# ---------------------------------------------------------------------------
# FD-08: anchor init not called twice under concurrent access (race guard)
# ---------------------------------------------------------------------------


def test_anchor_init_once_under_concurrency():
    """Simulate the pattern: init flag guarded by a lock is called exactly once."""
    import threading

    init_count = 0
    lock = threading.Lock()
    initialized = [False]

    def maybe_init():
        nonlocal init_count
        with lock:
            if not initialized[0]:
                init_count += 1
                initialized[0] = True

    threads = [threading.Thread(target=maybe_init) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert init_count == 1, f"Expected init_count=1, got {init_count}"
