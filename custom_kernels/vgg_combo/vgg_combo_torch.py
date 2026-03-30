"""
FP16 PyTorch reimplementation of
``model_assets/vgg_combo_relu3_3_relu3_1.onnx`` — a VGG feature extractor
that returns two sets of block-3 feature maps concatenated along the channel
dimension.

Architecture
------------
    Input   : (N, 3, 512, 512)  float32 — VGG-normalised image
    Output  : (N, 512, 128, 128)  float32

    VGG Block 1 (relu1_x):
        Conv(  3→ 64, 3×3, p=1) → ReLU
        Conv( 64→ 64, 3×3, p=1) → ReLU
        MaxPool(2×2, s=2)                          → (N,  64, 256, 256)

    VGG Block 2 (relu2_x):
        Conv( 64→128, 3×3, p=1) → ReLU
        Conv(128→128, 3×3, p=1) → ReLU
        MaxPool(2×2, s=2)                          → (N, 128, 128, 128)

    VGG Block 3 partial:
        Conv(128→256, 3×3, p=1)                    → pre_relu3_1  (N, 256, 128, 128)

        Branch A  (first 256 output channels):
            pre_relu3_1   ──────────────────────→  feat_A  (N, 256, 128, 128)
            (no activation — ONNX applies Resize to pre-relu tensor, which is
            a no-op for 512×512 input since features are already 128×128)

        Branch B  (second 256 output channels):
            ReLU(pre_relu3_1) = relu3_1
            Conv(256→256, 3×3, p=1) → ReLU = relu3_2
            Conv(256→256, 3×3, p=1)           = pre_relu3_3
            → feat_B  (N, 256, 128, 128)

        Output: Concat([feat_A, feat_B], dim=1)    → (N, 512, 128, 128)

Note on the ONNX resize
-----------------------
    The ONNX graph contains two Resize nodes that dynamically upsample the
    branch outputs to a fixed 128×128 spatial size.  For the standard 512×512
    input, both feature maps are already 128×128 after the two MaxPool layers
    so the resize is a no-op.  This PyTorch implementation targets the 512×512
    use-case and skips the resize entirely.

Weight loading
--------------
    All 14 ONNX initialisers have named weights (``model.N.weight / .bias``).
    Loaded by name:
        model.0  → conv1_1    model.2  → conv1_2
        model.5  → conv2_1    model.7  → conv2_2
        model.10 → conv3_1    model.12 → conv3_2    model.14 → conv3_3
"""

from __future__ import annotations

import pathlib
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class VggComboTorch(nn.Module):
    """
    FP16-ready VGG combo feature extractor.

    Instantiate via :meth:`from_onnx` to load pretrained weights.
    """

    def __init__(self, compute_dtype: torch.dtype = torch.float16):
        super().__init__()
        self.compute_dtype = compute_dtype

        # VGG Block 1
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=1, bias=True)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1, bias=True)

        # VGG Block 2
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1, bias=True)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1, bias=True)

        # VGG Block 3 (partial)
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1, bias=True)  # → pre_relu3_1
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1, bias=True)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1, bias=True)  # → pre_relu3_3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (N, 3, 512, 512) float32 (VGG-normalised)

        Returns:
            (N, 512, 128, 128) float32 — concat of pre_relu3_1 and pre_relu3_3
        """
        x = x.to(self.compute_dtype)

        # Block 1
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = F.max_pool2d(x, 2, 2)

        # Block 2
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = F.max_pool2d(x, 2, 2)

        # Block 3 — first conv, no ReLU yet
        pre_relu3_1 = self.conv3_1(x)  # (N, 256, 128, 128)

        # Branch B: relu3_1 → conv3_2 → relu3_2 → conv3_3
        h = F.relu(pre_relu3_1)
        h = F.relu(self.conv3_2(h))
        pre_relu3_3 = self.conv3_3(h)  # (N, 256, 128, 128)

        # Concat: [pre_relu3_1, pre_relu3_3] → (N, 512, 128, 128)
        out = torch.cat([pre_relu3_1, pre_relu3_3], dim=1)
        return out.float()  # always return float32

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_onnx(
        cls,
        onnx_path: Union[str, pathlib.Path],
        compute_dtype: torch.dtype = torch.float16,
    ) -> "VggComboTorch":
        """
        Construct a VggComboTorch and load weights from the ONNX model.

        All 7 Conv weight/bias pairs are loaded by their named initialisers
        (``model.N.weight`` / ``model.N.bias``).
        """
        import onnx
        from onnx import numpy_helper

        proto = onnx.load(str(onnx_path))
        g = proto.graph
        init_map = {
            init.name: numpy_helper.to_array(init).copy() for init in g.initializer
        }

        m = cls(compute_dtype=compute_dtype)

        def _load(layer: nn.Conv2d, w_name: str, b_name: str) -> None:
            layer.weight.data = torch.from_numpy(init_map[w_name]).to(compute_dtype)
            layer.bias.data = torch.from_numpy(init_map[b_name]).to(compute_dtype)

        _load(m.conv1_1, "model.0.weight", "model.0.bias")
        _load(m.conv1_2, "model.2.weight", "model.2.bias")
        _load(m.conv2_1, "model.5.weight", "model.5.bias")
        _load(m.conv2_2, "model.7.weight", "model.7.bias")
        _load(m.conv3_1, "model.10.weight", "model.10.bias")
        _load(m.conv3_2, "model.12.weight", "model.12.bias")
        _load(m.conv3_3, "model.14.weight", "model.14.bias")

        print("[vgg_combo] loaded: 7 Conv weight+bias tensors")
        m._visomaster_onnx_path = str(onnx_path)
        return m


# ---------------------------------------------------------------------------
# CUDA graph runner
# ---------------------------------------------------------------------------


class VggComboCUDAGraphRunner:
    """
    Wraps VggComboTorch in a CUDA graph for minimal kernel-launch overhead.

    A single (1, 3, 512, 512) graph is captured.  The runner is called twice
    sequentially (swapped then original) under a lock in face_masks.py; the
    static input buffer is rewritten and replayed for each call.
    """

    def __init__(
        self,
        model: VggComboTorch,
        input_shape: tuple = (1, 3, 512, 512),
    ):
        self.model = model
        self.device = next(model.parameters()).device

        self._x_buf = torch.zeros(input_shape, dtype=torch.float32, device=self.device)

        # Warm-up: cuDNN workspace allocation + auto-tuning
        with torch.no_grad():
            for _ in range(3):
                _ = model(self._x_buf)
        torch.cuda.synchronize()

        # Capture
        self._stream = torch.cuda.Stream()
        torch.cuda.synchronize()
        self._graph = torch.cuda.CUDAGraph()
        with torch.no_grad():
            with torch.cuda.graph(
                self._graph, stream=self._stream, capture_error_mode="relaxed"
            ):
                self._out = model(self._x_buf)  # (1, 512, 128, 128) float32
        torch.cuda.synchronize()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (1, 3, 512, 512)  float32 CUDA

        Returns:
            (1, 512, 128, 128)  float32
        """
        self._x_buf.copy_(x)
        self._graph.replay()
        return self._out.clone()


def build_cuda_graph_runner(
    model: VggComboTorch,
    input_shape: tuple = (1, 3, 512, 512),
    torch_compile: bool = False,
) -> VggComboCUDAGraphRunner:
    """Build and return a CUDA-graph-backed runner for VggComboTorch."""
    if torch_compile:
        try:
            from custom_kernels.compile_utils import apply_torch_compile
            device = next(model.parameters()).device
            example_inp = torch.zeros(input_shape, dtype=torch.float32, device=device)
            compiled = apply_torch_compile(model, example_inp)
            print("[VggComboTorch] torch.compile warmup done.")
            return compiled  # CUDA graph on top of torch.compile fails on Windows
        except Exception as e:
            print(f"[VggComboTorch] torch.compile failed ({e!s:.120}), falling back to CUDA graph.")
    return VggComboCUDAGraphRunner(model, input_shape=input_shape)
