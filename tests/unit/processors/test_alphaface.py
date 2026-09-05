"""Focused tests for the AlphaFace swapper integration."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F
from torchvision.transforms import v2

from app.processors.alphaface.model import IdentityFeedingBlock, OperationUnit
from app.processors.face_swappers import FaceSwappers
from app.processors.models_data import (
    arcface_mapping_model_dict,
    fp16_safe_models_list,
    models_list,
    tensorrt_shape_infer_models,
)
from app.processors.utils import faceutil
from app.processors.workers.frame_worker import FrameWorker
from app.processors.workers.frame_worker_pipeline import PipelineProcessor


def test_alphaface_ships_one_model_and_reuses_the_shared_arcface() -> None:
    entries = [item for item in models_list if item["model_name"] == "AlphaFace"]

    assert len(entries) == 1
    assert entries[0]["local_path"].endswith("alphaface_swapper_fused_norm.onnx")
    assert entries[0]["url"].endswith("/alphaface_swapper_fused_norm.onnx")
    # No separate recognition model: AlphaFace re-projects the W600K embedding.
    assert arcface_mapping_model_dict["AlphaFace"] == "Inswapper128ArcFace"
    assert "AlphaFace" in fp16_safe_models_list


def test_alphaface_is_shape_inferred_before_the_tensorrt_build() -> None:
    """The shipped ONNX has no value_info and an output whose batch and height
    share one dim_param. Building a TensorRT engine from it that way crashes the
    driver, so the loader must route AlphaFace through the shape-inference
    sidecar (ModelsProcessor._ensure_trt_ready_onnx)."""
    assert "AlphaFace" in tensorrt_shape_infer_models


def test_alphaface_projection_is_matrix_multiply_then_l2_normalize() -> None:
    swapper = FaceSwappers.__new__(FaceSwappers)
    swapper._alphaface_emap = np.eye(512, dtype=np.float32) * 2.0
    embedding: np.ndarray = np.arange(1, 513, dtype=np.float32)

    latent = swapper.calc_swapper_latent_alphaface(embedding)

    assert latent is not None
    expected = embedding.reshape(1, -1) / np.linalg.norm(embedding)
    np.testing.assert_allclose(latent, expected, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(np.linalg.norm(latent), 1.0, atol=1e-6)


def test_alphaface_projection_rejects_wrong_embedding_length() -> None:
    swapper = FaceSwappers.__new__(FaceSwappers)
    swapper._alphaface_emap = np.eye(512, dtype=np.float32)

    assert swapper.calc_swapper_latent_alphaface(np.ones(256, dtype=np.float32)) is None


def test_alphaface_identity_block_matches_official_singleton_adain() -> None:
    """The 1x1 projected identity makes the official AdaIN collapse to the mean."""
    torch.manual_seed(7)
    block = IdentityFeedingBlock(output_dim=8, identity_dim=4)
    identity = torch.randn((1, 4))
    target = torch.randn((1, 4, 8, 8))

    projected = block.fc(identity).unsqueeze(2).unsqueeze(3)
    first, second = projected.chunk(2, dim=1)

    def official_adain(x: torch.Tensor) -> torch.Tensor:
        x_mean = torch.sum(x, (2, 3)) / (x.shape[2] * x.shape[3])
        centered = (x.permute(2, 3, 0, 1) - x_mean).permute(2, 3, 0, 1)
        x_std = torch.sqrt(
            (torch.sum(centered**2, (2, 3)) + 2.3e-8) / (x.shape[2] * x.shape[3])
        )
        target_mean = torch.sum(target, (2, 3)) / (target.shape[2] * target.shape[3])
        target_centered = (target.permute(2, 3, 0, 1) - target_mean).permute(2, 3, 0, 1)
        target_std = torch.sqrt(
            (torch.sum(target_centered**2, (2, 3)) + 2.3e-8)
            / (target.shape[2] * target.shape[3])
        )
        normalized = (x.permute(2, 3, 0, 1) - x_mean) / x_std
        return (target_std * normalized + target_mean).permute(2, 3, 0, 1)

    expected = (
        (first + official_adain(first)) / 2.0,
        (second + official_adain(second)) / 2.0,
    )
    actual = block(identity, target)

    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)


def test_alphaface_instance_norm_matches_the_official_manual_expansion() -> None:
    """F.instance_norm replaces the official ReduceMean/Mul/Sqrt/Div chain."""
    torch.manual_seed(11)
    unit = OperationUnit(4, 8, activate=False)
    features = torch.randn((1, 4, 16, 16))
    identity = torch.randn((1, 512))

    def official_forward(module: OperationUnit) -> torch.Tensor:
        scale, bias = module.IFF(identity, features)
        output = module.Conv1(F.pad(features, (1, 1, 1, 1), mode="reflect"))
        output = output - torch.mean(output, dim=(2, 3), keepdim=True)
        variance = torch.mean(torch.mul(output, output), (2, 3), keepdim=True)
        inverse_std = torch.div(1.0, torch.sqrt(torch.add(variance, 1.0e-8)))
        output = torch.mul(output, inverse_std)
        return torch.add(torch.mul(scale, output), bias)

    torch.testing.assert_close(
        unit(features, identity), official_forward(unit), rtol=2e-5, atol=2e-5
    )


def test_alphaface_selects_256px_face_and_projected_latent() -> None:
    source: np.ndarray = np.ones(512, dtype=np.float32)
    target: np.ndarray = np.full(512, 2.0, dtype=np.float32)

    class Functions:
        @staticmethod
        def calc_swapper_latent_alphaface(embedding: np.ndarray) -> np.ndarray:
            value = 3.0 if embedding is source else 4.0
            return np.full((1, 512), value, dtype=np.float32)

    worker = SimpleNamespace(
        function_worker=Functions(),
        models_processor=SimpleNamespace(device=torch.device("cpu")),
    )
    pipeline = PipelineProcessor(worker)  # type: ignore[arg-type]
    faces = tuple(torch.zeros((3, size, size)) for size in (512, 384, 256, 128))

    selected, dfm, dim, latent = pipeline.get_affined_face_dim_and_swapping_latents(
        faces, "AlphaFace", None, source, target, {}, False, None
    )

    assert selected is faces[2]
    assert dfm is None
    assert dim == 2
    assert torch.is_tensor(latent)
    assert torch.all(latent == 3.0)


def test_alphaface_skips_the_target_projection_when_likeness_is_off() -> None:
    source: np.ndarray = np.ones(512, dtype=np.float32)
    target: np.ndarray = np.full(512, 2.0, dtype=np.float32)

    class Functions:
        @staticmethod
        def calc_swapper_latent_alphaface(embedding: np.ndarray) -> np.ndarray:
            if embedding is target:
                raise AssertionError("target latent should not be projected")
            return np.ones((1, 512), dtype=np.float32)

    worker = SimpleNamespace(
        function_worker=Functions(),
        models_processor=SimpleNamespace(device=torch.device("cpu")),
    )
    pipeline = PipelineProcessor(worker)  # type: ignore[arg-type]
    faces = tuple(torch.zeros((3, size, size)) for size in (512, 384, 256, 128))

    selected, _dfm, _dim, latent = pipeline.get_affined_face_dim_and_swapping_latents(
        faces,
        "AlphaFace",
        None,
        source,
        target,
        {"FaceLikenessEnableToggle": False},
        False,
        None,
    )

    assert selected is faces[2]
    assert torch.is_tensor(latent)


def test_alphaface_failed_projection_skips_the_swap() -> None:
    class Functions:
        @staticmethod
        def calc_swapper_latent_alphaface(_embedding: np.ndarray) -> None:
            return None

    worker = SimpleNamespace(
        function_worker=Functions(),
        models_processor=SimpleNamespace(device=torch.device("cpu")),
    )
    pipeline = PipelineProcessor(worker)  # type: ignore[arg-type]
    faces = tuple(torch.zeros((3, size, size)) for size in (512, 384, 256, 128))

    selected, _dfm, _dim, latent = pipeline.get_affined_face_dim_and_swapping_latents(
        faces, "AlphaFace", None, np.ones(512, dtype=np.float32), None, {}, False, None
    )

    assert selected is None
    assert latent is None


def test_alphaface_lean_crop_path_skips_unused_resizes() -> None:
    worker = FrameWorker.__new__(FrameWorker)
    worker.t256 = v2.Resize((256, 256), antialias=True)
    worker.t384 = None
    worker.t128 = None
    image = torch.zeros((3, 512, 512), dtype=torch.uint8)
    transform = SimpleNamespace(params=np.eye(3, dtype=np.float32))

    face_512, face_384, face_256, face_128 = worker.get_transformed_and_scaled_faces(
        transform, image, interp_mode="bilinear", only_256=True
    )

    # The unused slots alias their larger sibling instead of allocating a resize.
    assert face_384.data_ptr() == face_512.data_ptr()
    assert face_128.data_ptr() == face_256.data_ptr()
    assert face_256.shape == (3, 256, 256)


def test_alphaface_uses_pose_aware_target_alignment() -> None:
    worker = FrameWorker.__new__(FrameWorker)
    profile_landmarks = faceutil.get_arcface_template(
        image_size=512, mode="arcfacemap"
    )[0]

    alphaface = worker.get_face_similarity_tform("AlphaFace", profile_landmarks)
    inswapper = worker.get_face_similarity_tform("Inswapper128", profile_landmarks)

    # Feeding a pose template back in must select that same template, so the
    # transform is exactly the AlphaFace crop adjustment: the 112-based map is
    # zoomed out to the 128-based Inswapper crop and re-centred horizontally.
    expected = profile_landmarks * (112.0 / 128.0)
    expected[:, 0] += (512.0 / 128.0) * 8.0
    np.testing.assert_allclose(alphaface(profile_landmarks), expected, atol=1e-3)

    assert not np.allclose(inswapper.params, alphaface.params, atol=1e-3)


def test_alphaface_excludes_scale_popping_pitch_templates() -> None:
    worker = FrameWorker.__new__(FrameWorker)
    pitch_up_landmarks = faceutil.get_arcface_template(
        image_size=512, mode="arcfacemap"
    )[5]

    alphaface = worker.get_face_similarity_tform("AlphaFace", pitch_up_landmarks)

    assert not np.allclose(alphaface.params, np.eye(3), atol=1e-3)


def test_alphaface_inference_path_preserves_unit_range_contract() -> None:
    class Functions:
        @staticmethod
        def run_swapper_alphaface(image, embedding, output) -> None:
            assert image.shape == (1, 3, 256, 256)
            assert embedding.shape == (1, 512)
            output.fill_(0.25)

    worker = SimpleNamespace(
        function_worker=Functions(),
        models_processor=SimpleNamespace(device=torch.device("cpu"), device_type="cpu"),
        t512=lambda tensor: tensor,
        GHOSTFACE_MODELS=frozenset(),
    )
    pipeline = PipelineProcessor(worker)  # type: ignore[arg-type]
    face = torch.full((256, 256, 3), 0.5)

    swap, previous = pipeline.get_swapped_and_prev_face(
        output=torch.empty_like(face),
        input_face_affined=face,
        original_face_512=torch.zeros((3, 512, 512)),
        latent=torch.ones((1, 512)),
        itex=1,
        dim=2,
        swapper_model="AlphaFace",
        dfm_model=None,
        parameters={"PreSwapSharpnessDecimalSlider": 1.0},
    )

    assert swap.shape == (3, 256, 256)
    assert torch.allclose(swap, torch.full_like(swap, 63.75))
    assert previous is face


def test_alphaface_nonfinite_output_falls_back_to_aligned_crop() -> None:
    class Functions:
        @staticmethod
        def run_swapper_alphaface(image, embedding, output) -> None:
            output.fill_(float("nan"))

    worker = SimpleNamespace(
        function_worker=Functions(),
        models_processor=SimpleNamespace(device=torch.device("cpu"), device_type="cpu"),
        t512=lambda tensor: tensor,
        GHOSTFACE_MODELS=frozenset(),
    )
    pipeline = PipelineProcessor(worker)  # type: ignore[arg-type]
    face = torch.full((256, 256, 3), 0.5)

    swap, _ = pipeline.get_swapped_and_prev_face(
        output=torch.empty_like(face),
        input_face_affined=face,
        original_face_512=torch.zeros((3, 512, 512)),
        latent=torch.ones((1, 512)),
        itex=1,
        dim=2,
        swapper_model="AlphaFace",
        dfm_model=None,
        parameters={"PreSwapSharpnessDecimalSlider": 1.0},
    )

    assert torch.isfinite(swap).all()
    assert torch.allclose(swap, torch.full_like(swap, 127.5))


def test_alphaface_empty_output_falls_back_to_aligned_crop() -> None:
    """An unloaded model zeroes the buffer; the crop must survive unswapped."""

    class Functions:
        @staticmethod
        def run_swapper_alphaface(image, embedding, output) -> None:
            output.zero_()

    worker = SimpleNamespace(
        function_worker=Functions(),
        models_processor=SimpleNamespace(device=torch.device("cpu"), device_type="cpu"),
        t512=lambda tensor: tensor,
        GHOSTFACE_MODELS=frozenset(),
    )
    pipeline = PipelineProcessor(worker)  # type: ignore[arg-type]
    face = torch.full((256, 256, 3), 0.5)

    swap, _ = pipeline.get_swapped_and_prev_face(
        output=torch.empty_like(face),
        input_face_affined=face,
        original_face_512=torch.zeros((3, 512, 512)),
        latent=torch.ones((1, 512)),
        itex=1,
        dim=2,
        swapper_model="AlphaFace",
        dfm_model=None,
        parameters={"PreSwapSharpnessDecimalSlider": 1.0},
    )

    assert torch.allclose(swap, torch.full_like(swap, 127.5))
