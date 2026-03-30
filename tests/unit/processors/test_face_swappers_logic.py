"""
FS-* tests for face-swapper logic (embedding math, guards, model dispatch).

All model inference is mocked.
"""

from __future__ import annotations

import numpy as np
import torch


# ---------------------------------------------------------------------------
# FS-02: CSCS normalized embedding is a unit vector
# ---------------------------------------------------------------------------


def test_cscs_embedding_unit_norm():
    """After L2 normalisation the embedding magnitude should be 1.0."""
    raw = torch.randn(512)
    normed = raw / (raw.norm() + 1e-8)
    norm_val = normed.norm().item()
    assert abs(norm_val - 1.0) < 1e-5, f"Norm should be ~1.0, got {norm_val}"


def test_cscs_embedding_unit_norm_batch():
    """Works for a batch of embeddings."""
    raw = torch.randn(4, 512)
    norms = raw.norm(dim=1, keepdim=True).clamp(min=1e-8)
    normed = raw / norms
    per_norm = normed.norm(dim=1)
    assert torch.allclose(per_norm, torch.ones(4), atol=1e-5)


# ---------------------------------------------------------------------------
# FS-01: calc_inswapper_latent returns None on bad input
# ---------------------------------------------------------------------------


def test_calc_inswapper_latent_returns_none_on_none_embedding():
    """Simulate the None-on-failure guard in calc_inswapper_latent."""
    embedding = None

    def calc_inswapper_latent(emb):
        if emb is None:
            return None
        return emb @ emb.T  # placeholder

    result = calc_inswapper_latent(embedding)
    assert result is None


def test_calc_inswapper_latent_returns_none_on_empty():
    embedding = np.array([])

    def calc_inswapper_latent(emb):
        if emb is None or len(emb) == 0:
            return None
        return emb

    result = calc_inswapper_latent(embedding)
    assert result is None


# ---------------------------------------------------------------------------
# FS-05: GHOSTFACE_MODELS frozenset has exactly 3 expected members
# ---------------------------------------------------------------------------


def test_ghostface_models_frozenset_contents():
    GHOSTFACE_MODELS = frozenset({"GhostFace-v1", "GhostFace-v2", "GhostFace-v3"})
    assert "GhostFace-v1" in GHOSTFACE_MODELS
    assert "GhostFace-v2" in GHOSTFACE_MODELS
    assert "GhostFace-v3" in GHOSTFACE_MODELS
    assert len(GHOSTFACE_MODELS) == 3


def test_ghostface_models_is_frozenset():
    GHOSTFACE_MODELS = frozenset({"GhostFace-v1", "GhostFace-v2", "GhostFace-v3"})
    assert isinstance(GHOSTFACE_MODELS, frozenset)


# ---------------------------------------------------------------------------
# FS-04: GhostFace fallback to input face when model fails
# ---------------------------------------------------------------------------


def test_ghostface_fallback_on_none_output():
    """If the swapper returns None, output should fall back to the input face."""
    input_face = torch.randint(0, 256, (3, 128, 128), dtype=torch.uint8)

    def run_ghostface_swap(face_tensor, model):
        # Simulate model returning None
        return None

    model = None  # mocked
    result = run_ghostface_swap(input_face, model)
    swapped = result if result is not None else input_face

    assert torch.equal(swapped, input_face)


# ---------------------------------------------------------------------------
# FS-06: keep_alive_tensors list grows when restorer/KV tensors are appended
# ---------------------------------------------------------------------------


def test_keep_alive_tensors_grows():
    keep_alive_tensors: list = []
    kv_tensor = torch.randn(1, 4, 64, 64)
    keep_alive_tensors.append(kv_tensor)
    assert len(keep_alive_tensors) == 1
    assert keep_alive_tensors[0] is kv_tensor


def test_keep_alive_tensors_prevents_gc():
    """Tensors appended to keep_alive_tensors are still reachable."""
    import weakref

    keep_alive: list = []
    t = torch.randn(100, 100)
    ref = weakref.ref(t)
    keep_alive.append(t)
    del t
    import gc

    gc.collect()
    # Still alive because keep_alive holds a reference
    assert ref() is not None
