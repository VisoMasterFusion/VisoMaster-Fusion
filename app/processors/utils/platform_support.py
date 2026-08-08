"""Execution-provider capability detection.

VisoMaster Fusion targets Windows/Linux with NVIDIA hardware, where the
TensorRT and CUDA execution providers are always assumed present. macOS has
neither, so every code path that hard-codes ``"cuda"`` needs a way to ask what
this machine can actually do.

This module is the single source of truth for that question. It reports the
execution providers that are genuinely usable here (in priority order) and maps
each one to the torch device the rest of the pipeline should run tensors on.

Provider names are the user-facing strings stored in
``control["ProvidersPrioritySelection"]`` and matched in
``ModelsProcessor.update_provider_configuration``.
"""

import platform
import sys

import torch

SYSTEM_PLATFORM = platform.system()
IS_MACOS = sys.platform == "darwin"


def _onnxruntime_providers() -> list:
    """Available ONNX Runtime EPs, or an empty list if ORT cannot be queried."""
    try:
        import onnxruntime

        return list(onnxruntime.get_available_providers())
    except Exception:
        return []


def has_cuda() -> bool:
    """True when a usable CUDA device is present."""
    try:
        return torch.cuda.is_available()
    except Exception:
        return False


def has_mps() -> bool:
    """True when torch can run on Apple's Metal Performance Shaders backend."""
    try:
        return torch.backends.mps.is_available()
    except Exception:
        return False


def has_tensorrt() -> bool:
    """True when the TensorRT python bindings import."""
    try:
        import tensorrt  # noqa: F401

        return True
    except Exception:
        return False


def has_coreml() -> bool:
    """True when ONNX Runtime exposes the CoreML execution provider."""
    return "CoreMLExecutionProvider" in _onnxruntime_providers()


def has_neural_engine() -> bool:
    """True on Apple Silicon, which is the only Mac hardware with a Neural Engine.

    Intel Macs expose the CoreML EP too, but it can only fall back to their CPU
    and (via MPSGraph) an integrated/AMD GPU that CoreML drives poorly.
    """
    return IS_MACOS and platform.machine() == "arm64"


def available_execution_providers() -> list:
    """Execution providers usable on this machine, best first.

    Always ends with ``"CPU"``, which is available everywhere and is the
    guaranteed-working fallback.

    On Intel Macs CoreML is offered but deliberately ranked *below* CPU. Measured
    on an i7-9750H / Radeon Pro 5300M across the per-frame hot path
    (det_10g 62ms vs 65ms, w600k_r50 84ms vs 89ms, inswapper_128 1021ms vs
    1244ms), CoreML lost every time: the models carry dynamic dimensions, and
    RequireStaticInputShapes — which is mandatory for correct output, see
    ``ModelsProcessor.coreml_ep_options`` — hands those subgraphs back to the CPU
    EP anyway, leaving only partitioning overhead. Apple Silicon has the Neural
    Engine and a very different profile, so CoreML leads there.
    """
    providers = []
    if has_cuda():
        if has_tensorrt():
            providers.extend(["TensorRT", "TensorRT-Engine"])
        providers.append("CUDA")
    if has_coreml() and has_neural_engine():
        providers.append("CoreML")
    providers.append("CPU")
    if has_coreml() and not has_neural_engine():
        providers.append("CoreML")
    return providers


def default_execution_provider() -> str:
    """The provider to select when the user has not chosen one."""
    return available_execution_providers()[0]


def torch_device_for_provider(provider_name: str, gpu_id: int = 0) -> tuple:
    """Map a provider name to ``(device, device_type)`` for torch tensors.

    ``device`` is the string handed to ``torch.Tensor.to()``; ``device_type`` is
    the bare backend name used for ``torch.autocast`` and for the ``!= "cpu"``
    checks scattered through the pipeline.
    """
    if provider_name in ("TensorRT", "TensorRT-Engine", "CUDA"):
        return f"cuda:{gpu_id}", "cuda"
    if provider_name == "CoreML":
        # Torch stays on CPU even though MPS exists. The inference path binds
        # torch tensors straight into ONNX Runtime via io_binding
        # (device_type=..., buffer_ptr=tensor.data_ptr()). ORT has no "mps"
        # device and cannot dereference a Metal buffer pointer, so an MPS tensor
        # would be bound as garbage. CoreML still runs the model itself on the
        # GPU/Neural Engine — only the pre/post-processing is CPU-side.
        return "cpu", "cpu"
    return "cpu", "cpu"


def default_torch_device() -> str:
    """Torch device matching this machine's default execution provider.

    Deliberately derived from the provider rather than from raw hardware
    capability, so torch tensors always live somewhere the active ONNX Runtime
    provider can bind them.
    """
    return torch_device_for_provider(default_execution_provider())[0]


def supports_fp16(device_type: str) -> bool:
    """Whether half precision is worth using on this backend.

    CUDA has fast fp16 throughout. MPS supports fp16 but several ops used by the
    swapper pipeline fall back to (or outright fail on) fp32, so fp16 is only
    enabled for CUDA.
    """
    return device_type == "cuda"


def sync_device(device_type: str) -> None:
    """Block until queued work on ``device_type`` has completed.

    Replaces bare ``torch.cuda.current_stream().synchronize()`` calls, which
    raise on machines without CUDA.
    """
    if device_type == "cuda" and has_cuda():
        torch.cuda.current_stream().synchronize()
    elif device_type == "mps" and has_mps():
        torch.mps.synchronize()


def empty_cache(device_type: str = "") -> None:
    """Release cached allocator memory on whichever backend is active."""
    if has_cuda():
        torch.cuda.empty_cache()
    elif has_mps():
        torch.mps.empty_cache()
