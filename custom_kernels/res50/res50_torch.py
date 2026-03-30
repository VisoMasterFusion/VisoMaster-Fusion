"""
res50_torch.py — FP16 PyTorch reimplementation of res50.onnx (RetinaFace / FaceLandmark5)
==========================================================================================

Architecture (reverse-engineered from ONNX topological order):

  Input : (1, 3, 512, 512) float32
  Outputs: conf (1, 10752, 2) float32 [softmaxed], landmarks (1, 10752, 10) float32

  Backbone : ResNet-50 with BN folded into Conv (all Conv2d have bias=True, no BN layers)
  Neck     : FPN (Feature Pyramid Network) over C3/C4/C5
  Modules  : 3 × SSH (Single-Scale Head Module)
  Heads    : ClassHead / BboxHead / LandmarkHead (nn.ModuleList of 3 each)

Weight loading strategy
-----------------------
  * "Named" Conv nodes (head convolutions) are matched by initializer name.
  * "Anonymous" Conv nodes (backbone + FPN + SSH) are matched positionally via
    _conv_modules_in_forward_order(), which returns exactly 73 Conv2d modules in
    the same topological order the ONNX exporter produced them.

No Triton kernels are required — the FP16 speed-up comes entirely from cuDNN
TensorCore conv kernels when the model is cast to torch.float16.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Backbone helpers
# ---------------------------------------------------------------------------


class _Bottleneck(nn.Module):
    """ResNet bottleneck with BN folded into Conv (all biases present, no BN)."""

    def __init__(
        self,
        in_ch: int,
        mid_ch: int,
        out_ch: int,
        stride: int = 1,
        downsample: bool = False,
    ) -> None:
        super().__init__()
        # PyTorch forward order matches ONNX topological order:
        #   conv1 → conv2 → conv3, then downsample (if present)
        self.conv1 = nn.Conv2d(in_ch, mid_ch, 1, bias=True)
        self.conv2 = nn.Conv2d(mid_ch, mid_ch, 3, padding=1, stride=stride, bias=True)
        self.conv3 = nn.Conv2d(mid_ch, out_ch, 1, bias=True)
        self.downsample: Optional[nn.Conv2d] = (
            nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=True)
            if downsample
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = self.conv3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        return F.relu(out + identity)


def _make_layer(
    in_ch: int,
    mid_ch: int,
    out_ch: int,
    num_blocks: int,
    stride: int = 1,
) -> nn.Sequential:
    """Build a ResNet stage (Sequential of _Bottleneck blocks)."""
    blocks: List[_Bottleneck] = []
    # First block may downsample spatially and always projects channels
    blocks.append(_Bottleneck(in_ch, mid_ch, out_ch, stride=stride, downsample=True))
    for _ in range(1, num_blocks):
        blocks.append(_Bottleneck(out_ch, mid_ch, out_ch, stride=1, downsample=False))
    return nn.Sequential(*blocks)


class _ResNet50Body(nn.Module):
    """ResNet-50 backbone returning (C3, C4, C5) feature maps."""

    def __init__(self) -> None:
        super().__init__()
        # Stem
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        # Stages — stored as nn.Sequential so we can iterate blocks easily
        self.layer1 = _make_layer(64, 64, 256, num_blocks=3, stride=1)
        self.layer2 = _make_layer(256, 128, 512, num_blocks=4, stride=2)
        self.layer3 = _make_layer(512, 256, 1024, num_blocks=6, stride=2)
        self.layer4 = _make_layer(1024, 512, 2048, num_blocks=3, stride=2)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = F.relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.layer1(x)
        C3 = self.layer2(x)
        C4 = self.layer3(C3)
        C5 = self.layer4(C4)
        return C3, C4, C5


# ---------------------------------------------------------------------------
# FPN
# ---------------------------------------------------------------------------


class _FPN(nn.Module):
    """Feature Pyramid Network over C3/C4/C5.

    Conv order for weight loading (ONNX topological):
        output1, output2, output3, merge2, merge1
    """

    def __init__(self) -> None:
        super().__init__()
        self.output1 = nn.Conv2d(512, 256, 1, bias=True)  # lateral C3
        self.output2 = nn.Conv2d(1024, 256, 1, bias=True)  # lateral C4
        self.output3 = nn.Conv2d(2048, 256, 1, bias=True)  # lateral C5
        self.merge2 = nn.Conv2d(256, 256, 3, padding=1, bias=True)
        self.merge1 = nn.Conv2d(256, 256, 3, padding=1, bias=True)

    def forward(
        self,
        C3: torch.Tensor,
        C4: torch.Tensor,
        C5: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        p3 = F.leaky_relu(self.output1(C3), negative_slope=0.1)
        p4 = F.leaky_relu(self.output2(C4), negative_slope=0.1)
        p5 = F.leaky_relu(self.output3(C5), negative_slope=0.1)

        p4_up = F.interpolate(p5, size=p4.shape[-2:], mode="nearest")
        p4m = F.leaky_relu(self.merge2(p4 + p4_up), negative_slope=0.1)

        p3_up = F.interpolate(p4m, size=p3.shape[-2:], mode="nearest")
        p3m = F.leaky_relu(self.merge1(p3 + p3_up), negative_slope=0.1)

        return p3m, p4m, p5


# ---------------------------------------------------------------------------
# SSH module
# ---------------------------------------------------------------------------


class _SSH(nn.Module):
    """Single-Scale Head Module.

    Conv order for weight loading (ONNX topological):
        conv3X3, conv5X5_1, conv5X5_2, conv7X7_2, conv7x7_3
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv3X3 = nn.Conv2d(256, 128, 3, padding=1, bias=True)
        self.conv5X5_1 = nn.Conv2d(256, 64, 3, padding=1, bias=True)
        self.conv5X5_2 = nn.Conv2d(64, 64, 3, padding=1, bias=True)
        self.conv7X7_2 = nn.Conv2d(64, 64, 3, padding=1, bias=True)
        self.conv7x7_3 = nn.Conv2d(64, 64, 3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Branch A: 3×3 only (no activation before concat)
        a = self.conv3X3(x)

        # Branch B: 5×5 (two 3×3 convs), only first has LeakyReLU
        h = F.leaky_relu(self.conv5X5_1(x), negative_slope=0.1)
        b = self.conv5X5_2(h)  # no activation before concat

        # Branch C: 7×7 (three 3×3 convs), middle has LeakyReLU
        h2 = F.leaky_relu(self.conv7X7_2(h), negative_slope=0.1)
        c = self.conv7x7_3(h2)  # no activation before concat

        out = torch.cat([a, b, c], dim=1)  # 128 + 64 + 64 = 256 channels
        return F.relu(out)


# ---------------------------------------------------------------------------
# Detection heads
# ---------------------------------------------------------------------------


class _ClassHead(nn.Module):
    """Classification head: 256ch → 4ch → (1, N, 2)."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1x1 = nn.Conv2d(256, 4, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1x1(x)
        x = x.permute(0, 2, 3, 1)
        return x.reshape(x.shape[0], -1, 2)


class _BboxHead(nn.Module):
    """Bounding-box regression head: 256ch → 8ch → (1, N, 4).

    Instantiated and weights loaded, but the main forward does NOT include its
    output in the returned tuple (matches ONNX exported outputs).
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv1x1 = nn.Conv2d(256, 8, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1x1(x)
        x = x.permute(0, 2, 3, 1)
        return x.reshape(x.shape[0], -1, 4)


class _LandmarkHead(nn.Module):
    """Landmark regression head: 256ch → 20ch → (1, N, 10)."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1x1 = nn.Conv2d(256, 20, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1x1(x)
        x = x.permute(0, 2, 3, 1)
        return x.reshape(x.shape[0], -1, 10)


# ---------------------------------------------------------------------------
# Weight-loading helpers
# ---------------------------------------------------------------------------


def _conv_modules_in_forward_order(model: "Res50Torch") -> List[nn.Conv2d]:
    """Return the 73 anonymous Conv2d modules in ONNX topological order.

    Breakdown:
        1   stem conv
        10  layer1 (3 blocks × 3 convs + 1 downsample in block0)
        13  layer2 (4 blocks × 3 convs + 1 downsample in block0)
        19  layer3 (6 blocks × 3 convs + 1 downsample in block0)
        10  layer4 (3 blocks × 3 convs + 1 downsample in block0)
         5  FPN (output1, output2, output3, merge2, merge1)
        15  SSH ×3 (5 convs each)
        --
        73  total
    """
    mods: List[nn.Conv2d] = []

    # Stem
    mods.append(model.body.conv1)

    # ResNet stages
    for layer in (
        model.body.layer1,
        model.body.layer2,
        model.body.layer3,
        model.body.layer4,
    ):
        for block in layer:
            mods.append(block.conv1)
            mods.append(block.conv2)
            mods.append(block.conv3)
            if block.downsample is not None:
                mods.append(block.downsample)

    # FPN (order matches ONNX topological order)
    mods.append(model.fpn.output1)
    mods.append(model.fpn.output2)
    mods.append(model.fpn.output3)
    mods.append(model.fpn.merge2)
    mods.append(model.fpn.merge1)

    # SSH modules
    for ssh in (model.ssh1, model.ssh2, model.ssh3):
        mods.append(ssh.conv3X3)
        mods.append(ssh.conv5X5_1)
        mods.append(ssh.conv5X5_2)
        mods.append(ssh.conv7X7_2)
        mods.append(ssh.conv7x7_3)

    assert len(mods) == 73, f"Expected 73 anonymous Conv2d, got {len(mods)}"
    return mods


def _load_anonymous_conv_params(
    model: "Res50Torch",
    onnx_model,  # onnx.ModelProto
    dtype: torch.dtype,
) -> None:
    """Load backbone / FPN / SSH Conv2d weights from ONNX initializers positionally.

    We iterate ONNX graph nodes in topological order, identify Conv nodes whose
    weight initializer name starts with 'onnx::' (anonymous), and assign them
    sequentially to the modules returned by _conv_modules_in_forward_order.
    """
    import numpy as np

    # Build initializer name → numpy-array map
    init_map = {init.name: init for init in onnx_model.graph.initializer}

    modules = _conv_modules_in_forward_order(model)
    mod_idx = 0

    for node in onnx_model.graph.node:
        if node.op_type != "Conv":
            continue

        # node.input[1] is the weight initializer name
        weight_name = node.input[1]
        if not weight_name.startswith("onnx::"):
            # Named conv — handled by _load_named_params
            continue

        if mod_idx >= len(modules):
            raise RuntimeError(
                f"More anonymous Conv nodes than expected (already consumed {len(modules)})"
            )

        target = modules[mod_idx]
        mod_idx += 1

        # Weight
        w_np = np.frombuffer(init_map[weight_name].raw_data, dtype=np.float32).copy()
        w_np = w_np.reshape(list(init_map[weight_name].dims))
        with torch.no_grad():
            target.weight.copy_(torch.from_numpy(w_np).to(dtype))

        # Bias (optional — node.input[2] if present)
        if len(node.input) > 2 and node.input[2]:
            bias_name = node.input[2]
            b_np = np.frombuffer(init_map[bias_name].raw_data, dtype=np.float32).copy()
            with torch.no_grad():
                target.bias.copy_(torch.from_numpy(b_np).to(dtype))

    if mod_idx != len(modules):
        raise RuntimeError(
            f"Anonymous Conv count mismatch: expected {len(modules)}, found {mod_idx}"
        )


def _load_named_params(
    model: "Res50Torch",
    onnx_model,  # onnx.ModelProto
    dtype: torch.dtype,
) -> None:
    """Load detection-head Conv2d weights by matching ONNX initializer names.

    ONNX names like 'ClassHead.0.conv1x1.weight' map directly to PyTorch
    state_dict keys of the same form.
    """
    import numpy as np

    state = model.state_dict()
    init_map = {init.name: init for init in onnx_model.graph.initializer}

    for node in onnx_model.graph.node:
        if node.op_type != "Conv":
            continue

        weight_name = node.input[1]
        if weight_name.startswith("onnx::"):
            # Anonymous — handled by _load_anonymous_conv_params
            continue

        # weight_name looks like  'ClassHead.0.conv1x1.weight'
        # bias may be             'ClassHead.0.conv1x1.bias'
        if weight_name not in state:
            raise KeyError(
                f"Named Conv weight '{weight_name}' not found in model state_dict"
            )

        w_np = np.frombuffer(init_map[weight_name].raw_data, dtype=np.float32).copy()
        w_np = w_np.reshape(list(init_map[weight_name].dims))
        with torch.no_grad():
            state[weight_name].copy_(torch.from_numpy(w_np).to(dtype))

        if len(node.input) > 2 and node.input[2]:
            bias_name = node.input[2]
            if bias_name not in state:
                raise KeyError(
                    f"Named Conv bias '{bias_name}' not found in model state_dict"
                )
            b_np = np.frombuffer(init_map[bias_name].raw_data, dtype=np.float32).copy()
            with torch.no_grad():
                state[bias_name].copy_(torch.from_numpy(b_np).to(dtype))


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------


class Res50Torch(nn.Module):
    """FP16-capable PyTorch reimplementation of res50.onnx (RetinaFace / FaceLandmark5).

    Inputs:  (1, 3, 512, 512) float32
    Outputs: (conf, landmarks)
             conf      — (1, 10752, 2)  float32  [softmaxed class scores]
             landmarks — (1, 10752, 10) float32  [5-point facial landmarks]
    """

    def __init__(self) -> None:
        super().__init__()
        self.body = _ResNet50Body()
        self.fpn = _FPN()

        self.ssh1 = _SSH()
        self.ssh2 = _SSH()
        self.ssh3 = _SSH()

        self.ClassHead = nn.ModuleList([_ClassHead() for _ in range(3)])
        self.BboxHead = nn.ModuleList([_BboxHead() for _ in range(3)])
        self.LandmarkHead = nn.ModuleList([_LandmarkHead() for _ in range(3)])

        # Set after from_onnx; used to cast inputs in forward()
        self._compute_dtype: torch.dtype = torch.float32

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.to(self._compute_dtype)

        # Backbone
        C3, C4, C5 = self.body(x)

        # FPN
        f1, f2, f3 = self.fpn(C3, C4, C5)

        # SSH
        s1 = self.ssh1(f1)
        s2 = self.ssh2(f2)
        s3 = self.ssh3(f3)

        features = [s1, s2, s3]

        # Classification — softmax over last dim, concat along anchor dim
        cls_outs = [self.ClassHead[i](features[i]) for i in range(3)]
        conf = F.softmax(torch.cat(cls_outs, dim=1), dim=-1)

        # Bbox — computed but not returned (matches ONNX exported interface)
        _ = [self.BboxHead[i](features[i]) for i in range(3)]

        # Landmarks
        ldmk_outs = [self.LandmarkHead[i](features[i]) for i in range(3)]
        ldmk = torch.cat(ldmk_outs, dim=1)

        # Always return FP32 regardless of compute dtype
        return conf.float(), ldmk.float()

    # ------------------------------------------------------------------
    # Class-method constructor
    # ------------------------------------------------------------------

    @classmethod
    def from_onnx(
        cls,
        onnx_path: str | Path,
        compute_dtype: torch.dtype = torch.float16,
    ) -> "Res50Torch":
        """Build Res50Torch and load weights from an ONNX file.

        Args:
            onnx_path:     Path to res50.onnx.
            compute_dtype: dtype for internal computation (default: float16).

        Returns:
            Res50Torch instance with loaded weights, converted to compute_dtype.
        """
        import onnx  # type: ignore

        onnx_path = Path(onnx_path)
        print(f"[Res50Torch] Loading ONNX model from {onnx_path} …")
        onnx_model = onnx.load(str(onnx_path))

        model = cls()
        _load_anonymous_conv_params(model, onnx_model, dtype=torch.float32)
        _load_named_params(model, onnx_model, dtype=torch.float32)

        model._compute_dtype = compute_dtype
        model = model.to(compute_dtype)
        model.eval()

        total_params = sum(p.numel() for p in model.parameters())
        print(
            f"[Res50Torch] Loaded {total_params:,} parameters "
            f"| compute dtype: {compute_dtype}"
        )
        model._visomaster_onnx_path = str(onnx_path)
        return model


# ---------------------------------------------------------------------------
# CUDA graph runner
# ---------------------------------------------------------------------------


class Res50CUDAGraphRunner:
    """Wraps Res50Torch in a CUDA graph for lower-latency inference.

    The model is replayed on a static input buffer; outputs are cloned so
    callers receive independent tensors each call.
    """

    def __init__(self, model: Res50Torch, warmup: int = 3) -> None:
        device = next(model.parameters()).device
        self._inp = torch.zeros(1, 3, 512, 512, dtype=torch.float32, device=device)

        # Warm-up runs to prime cuDNN auto-tuner
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(self._inp)

        self._graph = torch.cuda.CUDAGraph()
        self._stream = torch.cuda.Stream()

        torch.cuda.synchronize()
        with (
            torch.no_grad(),
            torch.cuda.graph(
                self._graph, stream=self._stream, capture_error_mode="relaxed"
            ),
        ):
            self._out_conf, self._out_ldmk = model(self._inp)
        torch.cuda.synchronize()

    def __call__(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self._inp.copy_(x, non_blocking=True)
        self._graph.replay()
        return self._out_conf.clone(), self._out_ldmk.clone()


def build_cuda_graph_runner(
    model: Res50Torch, warmup: int = 3, torch_compile: bool = False
) -> Res50CUDAGraphRunner:
    """Convenience factory for Res50CUDAGraphRunner."""
    if torch_compile:
        try:
            from custom_kernels.compile_utils import apply_torch_compile
            device = next(model.parameters()).device
            example_inp = torch.zeros((1, 3, 512, 512), dtype=torch.float32, device=device)
            compiled = apply_torch_compile(model, example_inp)
            print("[Res50Torch] torch.compile warmup done.")
            return compiled  # CUDA graph on top of torch.compile fails on Windows
        except Exception as e:
            print(f"[Res50Torch] torch.compile failed ({e!s:.120}), falling back to CUDA graph.")
    return Res50CUDAGraphRunner(model, warmup)
