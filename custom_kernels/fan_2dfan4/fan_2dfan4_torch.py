"""
FP16 PyTorch reimplementation of ``model_assets/2dfan4.onnx`` — the 4-stacked
hourglass 68-point face landmark detector (FaceLandmark68).

Architecture
------------
    Input   : (1, 3, 256, 256)  float32
    Outputs : landmarks_xyscore (1, 68, 3)  — (x, y, score), x/y in [0.5..63.5]
              heatmaps           (1, 68, 64, 64)  float32

    Backbone: 4-stacked hourglass network (Newell et al. / Bulat 2-D FAN)
    Blocks  : pre-activation concatenation bottleneck (3 convs, outputs concat'd)
    Decoder : nearest-neighbour ×2 Resize + skip-connection Add
    Head    : per-stack ReLU(1×1 Conv) → l_i (68-ch heatmap); stacks 0-2 also
              project features back via bl_i + al_i for the next stack's input.

Post-processing (baked into forward())
    1. ReduceMax over spatial dims → per-landmark score.
    2. ArgMax → peak pixel (py, px).
    3. Build L2-distance mask: keep pixels within radius 6.4 of peak.
    4. Clip masked heatmap to ≥ 0, compute 2-D moment centroid with half-pixel
       offsets [0.5, 1.5, …, 63.5] — matches the ONNX x_indices / y_indices.

Weight loading
--------------
    Named   : 910 initializers matching PyTorch state_dict keys directly
              (all BN, Conv, head weights) → loaded via load_state_dict(strict=False).
    Anon    : 10 anonymous Conv initializers (stem + 4 inter-stack convs) collected
              in topological node order and assigned positionally.
    Shared  : conv2.downsample[0] and conv4.downsample[0] share running_mean/var
              with their respective bn1 — fixed up manually after the named load.
"""

from __future__ import annotations

import pathlib
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class _FanBlock(nn.Module):
    """
    Pre-activation concatenation bottleneck.

    Three convolutions are chained in sequence; their OUTPUTS are concatenated
    (not just the last one).  The shortcut is either an identity (when in_ch ==
    c1+c2+c3) or a BN→ReLU→Conv(1×1) projection.

    Unlike standard pre-activation ResNets, the BN+ReLU at the start of the
    block is shared between the main path (→ conv1) and the projection shortcut
    (→ downsample conv).  The shortcut BN uses the same running_mean/running_var
    as bn1 but its own affine parameters (downsample.0.weight/bias), which are
    handled in from_onnx() after the named-weight load.
    """

    def __init__(self, in_ch: int, c1: int, c2: int, c3: int, proj_out: int = 0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, c1, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c1)
        self.conv2 = nn.Conv2d(c1, c2, 3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(c2)
        self.conv3 = nn.Conv2d(c2, c3, 3, padding=1, bias=False)

        if proj_out > 0:
            # Shortcut: BN → ReLU → Conv(1×1), no bias on conv
            self.downsample = nn.Sequential(
                nn.BatchNorm2d(in_ch),  # [0]
                nn.ReLU(inplace=False),  # [1]
                nn.Conv2d(in_ch, proj_out, 1, bias=False),  # [2]
            )
        else:
            self.downsample = None  # identity

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.bn1(x), inplace=False)  # pre-activation
        c1 = self.conv1(h)
        c2 = self.conv2(F.relu(self.bn2(c1), inplace=False))
        c3 = self.conv3(F.relu(self.bn3(c2), inplace=False))
        concat = torch.cat([c1, c2, c3], dim=1)  # c1+c2+c3 channels
        skip = self.downsample(x) if self.downsample is not None else x
        return concat + skip


# ---------------------------------------------------------------------------
# Recursive 4-level hourglass module
# ---------------------------------------------------------------------------


class _HourGlass(nn.Module):
    """
    Recursive 4-level hourglass with pre-activation concatenation bottlenecks.

    All 13 residual blocks operate on 256 channels (identity shortcut):
        b1_4  b2_4  b1_3  b2_3  b1_2  b2_2  b1_1  b2_1   ← encoder
        b2_plus_1                                           ← base case
        b3_1  b3_2  b3_3  b3_4                             ← decoder

    Pooling : 2×2 AveragePool stride-2 (no padding — inputs are always powers
              of 2: 64 → 32 → 16 → 8 → 4).
    Upsampling : nearest-neighbour ×2.
    """

    BLOCK_ARGS = (256, 128, 64, 64)  # (in, c1, c2, c3) → output 256ch

    def __init__(self, depth: int = 4):
        super().__init__()
        self._depth = depth

        def B():
            return _FanBlock(*self.BLOCK_ARGS)

        for d in range(1, depth + 1):
            self.add_module(f"b1_{d}", B())
            self.add_module(f"b2_{d}", B())
        self.b2_plus_1 = B()
        for d in range(1, depth + 1):
            self.add_module(f"b3_{d}", B())

    def _step(self, level: int, x: torch.Tensor) -> torch.Tensor:
        b1 = getattr(self, f"b1_{level}")
        b2 = getattr(self, f"b2_{level}")
        b3 = getattr(self, f"b3_{level}")

        # Encoder
        skip = b1(x)
        low = F.avg_pool2d(x, kernel_size=2, stride=2)
        low = b2(low)

        # Recurse or base case
        inner = self._step(level - 1, low) if level > 1 else self.b2_plus_1(low)

        # Decoder — use skip's exact spatial size to handle odd-dimension inputs
        # (scale_factor=2 would give floor(h/2)*2 ≠ h when h is odd)
        up = F.interpolate(b3(inner), size=skip.shape[-2:], mode="nearest")
        return skip + up

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._step(self._depth, x)


# ---------------------------------------------------------------------------
# Full 2DFAN4 model
# ---------------------------------------------------------------------------


class FAN2dfan4(nn.Module):
    """
    4-stacked hourglass FAN (Face Alignment Network) for 68-point landmarks.

    Input  : (1, 3, 256, 256)  float32
    Outputs: (landmarks_xyscore (1,68,3), heatmaps (1,68,64,64))  float32
    """

    # Distance threshold for Gaussian centroid mask (from ONNX onnx::Greater_1889)
    _DIST_THRESH: float = 6.4
    # Min denominator for centroid (from ONNX onnx::Clip_1999)
    _M00_EPS: float = 1.1920929e-07

    def __init__(self, compute_dtype: torch.dtype = torch.float16):
        super().__init__()
        self.compute_dtype = compute_dtype

        # -- Stem convolution (anonymous weight: onnx::Conv_1946/1947) --------
        self.stem_conv = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=True)

        # -- Stem residual blocks ---------------------------------------------
        # conv2: 64  → (64,32,32) → 128,  shortcut 64→128
        self.conv2 = _FanBlock(64, 64, 32, 32, proj_out=128)
        # conv3: 128 → (64,32,32) → 128,  identity
        self.conv3 = _FanBlock(128, 64, 32, 32, proj_out=0)
        # conv4: 128 → (128,64,64) → 256, shortcut 128→256
        self.conv4 = _FanBlock(128, 128, 64, 64, proj_out=256)

        # -- 4 stacked hourglasses -------------------------------------------
        for i in range(4):
            self.add_module(f"m{i}", _HourGlass(depth=4))
            self.add_module(f"top_m_{i}", _FanBlock(*_HourGlass.BLOCK_ARGS))
            # Anonymous inter-stack conv (256→256, 1×1, bias=True)
            self.add_module(f"inter_conv{i}", nn.Conv2d(256, 256, 1, bias=True))
            # Named prediction head
            self.add_module(f"l{i}", nn.Conv2d(256, 68, 1, bias=True))
            if i < 3:
                # Named feature back-projection heads (stacks 0-2 only)
                self.add_module(f"bl{i}", nn.Conv2d(256, 256, 1, bias=True))
                self.add_module(f"al{i}", nn.Conv2d(68, 256, 1, bias=True))

    # ------------------------------------------------------------------

    def _decode(self, heatmaps: torch.Tensor) -> torch.Tensor:
        """
        Replicate ONNX post-processing to produce landmarks_xyscore (1,68,3).

        Args:
            heatmaps : (1, 68, 64, 64)  float32
        Returns:
            (1, 68, 3)  float32  — (x_centroid, y_centroid, max_score)
        """
        B, K, H, W = heatmaps.shape
        dev = heatmaps.device
        dtype = heatmaps.dtype

        # --- Score: max over spatial dims ------------------------------------
        scores = heatmaps.amax(dim=3).amax(dim=2)  # (1, 68)

        # --- Peak location (integer pixel indices) ---------------------------
        flat = heatmaps.reshape(B, K, -1)
        peak_idx = flat.argmax(dim=-1)  # (1, 68)
        peak_y = (peak_idx // W).float()  # row index   [0,63]
        peak_x = (peak_idx % W).float()  # col index   [0,63]

        # --- L2-distance mask (integer grid, threshold 6.4) ------------------
        y_grid = torch.arange(H, dtype=dtype, device=dev).view(1, 1, H, 1)
        x_grid = torch.arange(W, dtype=dtype, device=dev).view(1, 1, 1, W)
        dist = torch.sqrt(
            (y_grid - peak_y.view(B, K, 1, 1)) ** 2
            + (x_grid - peak_x.view(B, K, 1, 1)) ** 2
        )  # (1, 68, 64, 64)
        mask = (dist <= self._DIST_THRESH).to(dtype)

        # --- Masked heatmap clipped to ≥ 0 -----------------------------------
        masked = heatmaps.clamp(min=0.0) * mask  # (1, 68, 64, 64)

        # --- Moments ---------------------------------------------------------
        m00 = masked.sum(dim=(2, 3)).clamp(min=self._M00_EPS)  # (1, 68)

        # Half-pixel centroid offsets match ONNX x_indices/y_indices = [0.5..63.5]
        x_idx = torch.arange(W, dtype=dtype, device=dev) + 0.5  # (W,)
        y_idx = torch.arange(H, dtype=dtype, device=dev) + 0.5  # (H,)

        # x: sum over H first (axis 2), then weight by x_idx, sum over W
        x_marginal = masked.sum(dim=2)  # (1,68,W)
        x_coord = (x_marginal * x_idx).sum(dim=2) / m00  # (1,68)

        # y: sum over W first (axis 3), then weight by y_idx, sum over H
        y_marginal = masked.sum(dim=3)  # (1,68,H)
        y_coord = (y_marginal * y_idx).sum(dim=2) / m00  # (1,68)

        return torch.stack([x_coord, y_coord, scores], dim=-1)  # (1,68,3)

    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor):
        """
        Args:
            x : (1, 3, 256, 256)  float32 CUDA
        Returns:
            landmarks_xyscore : (1, 68, 3)   float32  (x,y in [0.5,63.5], score)
            heatmaps          : (1, 68, 64, 64) float32
        """
        h = x.to(self.compute_dtype)

        # Stem
        h = F.relu(self.stem_conv(h))  # (1, 64, 128, 128)

        # Stem residual blocks
        h = self.conv2(h)  # (1, 128, 128, 128)
        h = F.avg_pool2d(h, 2, stride=2)  # (1, 128, 64, 64)
        h = self.conv3(h)  # (1, 128, 64, 64)
        h = self.conv4(h)  # (1, 256, 64, 64)

        # 4 stacked hourglass passes
        heatmaps = None
        for i in range(4):
            x_in = h
            hg_out = getattr(self, f"m{i}")(h)
            tm_out = getattr(self, f"top_m_{i}")(hg_out)
            interim = F.relu(getattr(self, f"inter_conv{i}")(tm_out))
            heatmap = getattr(self, f"l{i}")(interim)  # (1,68,64,64)

            if i < 3:
                h = (
                    x_in
                    + getattr(self, f"bl{i}")(interim)
                    + getattr(self, f"al{i}")(heatmap)
                )
            else:
                heatmaps = heatmap.float()

        # Post-processing: decode heatmaps → (x, y, score) landmarks
        lmk_xyscore = self._decode(heatmaps)
        return lmk_xyscore, heatmaps

    # ------------------------------------------------------------------

    @classmethod
    def from_onnx(
        cls,
        onnx_path: Union[str, pathlib.Path],
        compute_dtype: torch.dtype = torch.float16,
    ) -> "FAN2dfan4":
        """
        Construct a FAN2dfan4 and load all weights from the ONNX model.

        Named weights (910+) : loaded via load_state_dict(strict=False).
        Anonymous Conv weights (10): stem + 4 inter-stack convs (weight+bias),
            collected in ONNX node topological order.
        Shared BN running stats: conv2.downsample[0] and conv4.downsample[0]
            use the same running_mean/var as their bn1 — copied after named load.
        """
        import onnx
        from onnx import numpy_helper

        proto = onnx.load(str(onnx_path))
        g = proto.graph
        _init_map = {init.name: numpy_helper.to_array(init) for init in g.initializer}

        # ---- Separate named vs anonymous initializers -----------------------
        named_inits: dict = {}
        anon_conv_map: dict = {}  # name → ndarray

        for init in g.initializer:
            arr = numpy_helper.to_array(init)
            name = init.name
            if name.startswith("onnx::Conv_"):
                anon_conv_map[name] = arr
            else:
                named_inits[name] = arr

        # ---- Build model & load named weights -------------------------------
        # load_state_dict is avoided because its internal copy_() coerces the
        # source dtype to match the destination (FP32 buffers), silently undoing
        # the FP16 cast.  Direct .data = assignment replaces the buffer.
        m = cls(compute_dtype=compute_dtype)

        # Collect all BN module paths so their params/buffers stay FP32.
        bn_prefixes: set = set()
        for mod_name, mod in m.named_modules():
            if isinstance(mod, nn.BatchNorm2d):
                bn_prefixes.add(mod_name)

        def _is_bn_tensor(key: str) -> bool:
            # key is e.g. "conv2.bn1.weight" — check any BN prefix matches
            for pfx in bn_prefixes:
                if key == pfx or key.startswith(pfx + "."):
                    return True
            return False

        all_params = dict(m.named_parameters())
        all_buffers = dict(m.named_buffers())
        all_tensors = {**all_params, **all_buffers}

        loaded, skipped = 0, []
        for k, arr in named_inits.items():
            if k in all_tensors:
                t = torch.tensor(arr, dtype=torch.float32)
                target = all_tensors[k]
                if t.shape == target.data.shape:
                    # BN weight/bias/running-stats stay FP32; everything else → compute_dtype
                    target.data = t if _is_bn_tensor(k) else t.to(compute_dtype)
                    loaded += 1
                else:
                    skipped.append((k, tuple(t.shape), tuple(target.data.shape)))

        if skipped:
            for k, got, exp in skipped:
                print(f"[fan_2dfan4] shape mismatch: {k}: got {got}, expected {exp}")

        # ---- Fix shared BN running stats in shortcut paths ------------------
        # conv2.downsample[0] shares running_mean/var with conv2.bn1
        m.conv2.downsample[0].running_mean.copy_(m.conv2.bn1.running_mean)
        m.conv2.downsample[0].running_var.copy_(m.conv2.bn1.running_var)
        # conv4.downsample[0] shares running_mean/var with conv4.bn1
        m.conv4.downsample[0].running_mean.copy_(m.conv4.bn1.running_mean)
        m.conv4.downsample[0].running_var.copy_(m.conv4.bn1.running_var)

        # ---- Load anonymous Conv weights in topological node order ----------
        # Expected order (by ONNX node position):
        #   [0] stem_conv.weight  [1] stem_conv.bias
        #   [2] inter_conv0.weight [3] inter_conv0.bias
        #   [4] inter_conv1.weight [5] inter_conv1.bias
        #   [6] inter_conv2.weight [7] inter_conv2.bias
        #   [8] inter_conv3.weight [9] inter_conv3.bias
        anon_seq = []
        seen: set = set()
        for node in g.node:
            if node.op_type == "Conv":
                for inp in node.input[
                    1:
                ]:  # skip tensor input [0], take weight [1] and bias [2]
                    if inp in anon_conv_map and inp not in seen:
                        anon_seq.append(anon_conv_map[inp])
                        seen.add(inp)

        if len(anon_seq) == 10:
            m.stem_conv.weight.data = torch.from_numpy(anon_seq[0].copy()).to(
                compute_dtype
            )
            m.stem_conv.bias.data = torch.from_numpy(anon_seq[1].copy()).to(
                compute_dtype
            )
            for i in range(4):
                ic = getattr(m, f"inter_conv{i}")
                ic.weight.data = torch.from_numpy(anon_seq[2 + i * 2].copy()).to(
                    compute_dtype
                )
                ic.bias.data = torch.from_numpy(anon_seq[3 + i * 2].copy()).to(
                    compute_dtype
                )
        else:
            print(
                f"[fan_2dfan4] WARNING: expected 10 anon Conv tensors, "
                f"got {len(anon_seq)}"
            )

        print(
            f"[fan_2dfan4] loaded: {loaded} named + {len(anon_seq)} anon-Conv "
            f"initializers"
        )
        m._visomaster_onnx_path = str(onnx_path)
        return m


# ---------------------------------------------------------------------------
# CUDA graph runner
# ---------------------------------------------------------------------------


class FAN2dfan4CUDAGraphRunner:
    """Wraps FAN2dfan4 in a CUDA graph for minimal kernel-launch overhead."""

    def __init__(self, model: FAN2dfan4, input_shape: tuple = (1, 3, 256, 256)):
        self.model = model
        self.device = next(model.parameters()).device

        self._x_buf = torch.zeros(input_shape, dtype=torch.float32, device=self.device)

        # Warm-up (cuDNN auto-tune, workspace allocation)
        with torch.no_grad():
            for _ in range(3):
                _ = model(self._x_buf)
        torch.cuda.synchronize()

        # Capture
        self._stream = torch.cuda.Stream()
        self._graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        with torch.no_grad():
            with torch.cuda.graph(
                self._graph, stream=self._stream, capture_error_mode="relaxed"
            ):
                self._out = model(self._x_buf)  # (lmk_xyscore, heatmaps)
        torch.cuda.synchronize()

    def __call__(self, x: torch.Tensor):
        """
        Args:
            x : (1, 3, 256, 256)  float32 CUDA
        Returns:
            (landmarks_xyscore (1,68,3), heatmaps (1,68,64,64))  fresh tensors
        """
        self._x_buf.copy_(x)
        self._graph.replay()
        return tuple(o.clone() for o in self._out)


def build_cuda_graph_runner(
    model: FAN2dfan4,
    input_shape: tuple = (1, 3, 256, 256),
    torch_compile: bool = False,
) -> FAN2dfan4CUDAGraphRunner:
    """
    Build and return a CUDA-graph-backed runner for FAN2dfan4.

    Args:
        model         : FAN2dfan4 on CUDA in eval() mode.
        input_shape   : fixed input shape (default (1, 3, 256, 256)).
        torch_compile : if True, apply torch.compile before building the CUDA graph.
    """
    if torch_compile:
        try:
            from custom_kernels.compile_utils import apply_torch_compile
            device = next(model.parameters()).device
            example_inp = torch.zeros(input_shape, dtype=torch.float32, device=device)
            compiled = apply_torch_compile(model, example_inp)
            print("[fan_2dfan4] torch.compile warmup done.")
            return compiled  # CUDA graph on top of torch.compile fails on Windows
        except Exception as e:
            print(f"[fan_2dfan4] torch.compile failed ({e!s:.120}), falling back to CUDA graph.")
    return FAN2dfan4CUDAGraphRunner(model, input_shape=input_shape)
