"""
xseg_torch.py — FP16 PyTorch reimplementation of XSeg_model.onnx
=================================================================

Architecture (reverse-engineered from ONNX topological order):

  Input : (1, 3, 256, 256) float32  — RGB face crop, values in [0, 1]
  Output: (1, 1, 256, 256) float32  — sigmoid face-segmentation mask, values in [0, 1]

  Backbone : SN256_XSeg — symmetric U-Net encoder-decoder
  Encoder  : 6 stages with 2–3 Conv + custom RMSNormMax blocks each, plus
             depthwise strided-Conv downsampling (no MaxPool)
  Bottleneck: Flatten → Linear(4096→512) → Linear(512→4096) → Reshape
  Decoder  : 6 stages; each stage upsamples with ConvTranspose, concatenates
             the matching encoder skip, then applies 2–3 Conv + RMSNormMax
  Output   : Conv(32→1, 3×3) + Sigmoid

Normalization (RMSNormMax — custom per-block):
  rms    = sqrt( mean(x²) + eps )          eps = per-block learned scalar (Abs_* init)
  x_norm = x / rms
  x_aff  = x_norm * gamma + beta           gamma, beta ∈ ℝ^{1×C×1×1}
  out    = max( x_aff, max_val )           max_val ∈ ℝ^{1×C×1×1}  (learned per-channel floor)

Weight loading strategy
-----------------------
  * 43 Conv/ConvTranspose — loaded POSITIONALLY in ONNX topological order.
    (Depthwise downsampling Convs have no bias; ConvTranspose has no built-in bias.)
  * 6 ConvTranspose additive biases — loaded POSITIONALLY by detecting Add nodes
    whose tensor input is produced by a ConvTranspose node.
  * 36 RMSNormMax blocks — eps/gamma/beta/max_val loaded POSITIONALLY by scanning
    the ONNX graph in topological order for the corresponding node types/inputs.
  * 2 Linear (FC bridge) — weights loaded by name, biases by shape pattern.

Optional Triton acceleration: when custom_kernels.triton_ops is importable and
Triton is available, _RMSNormMax.forward() uses triton_rmsnormmax — a fused
per-channel RMS-norm + affine + max-floor kernel that replaces 5 PyTorch ops
with 2 global-memory passes, further reducing inference latency for all 36
norm blocks in the encoder and decoder.  Falls back to pure PyTorch silently.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Optional Triton acceleration for RMSNormMax
# ---------------------------------------------------------------------------
_TRITON_RMSNORMMAX = None
try:
    from custom_kernels.triton_ops import TRITON_AVAILABLE as _TRITON_AVAIL

    if _TRITON_AVAIL:
        from custom_kernels.triton_ops import triton_rmsnormmax as _TRITON_RMSNORMMAX  # type: ignore[assignment]
except Exception:
    pass


# ---------------------------------------------------------------------------
# Custom normalization + activation
# ---------------------------------------------------------------------------


class _RMSNormMax(nn.Module):
    """Per-block RMS normalisation with learned affine and max-clamp activation.

    rms   = sqrt( mean(x²) + eps )     # eps: per-block scalar (Abs_* in ONNX)
    x_n   = x / rms
    x_aff = x_n * gamma + beta         # [1, C, 1, 1] learnable
    out   = max( x_aff, max_val )      # [1, C, 1, 1] learnable per-channel floor
    """

    def __init__(self, ch: int) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, ch, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, ch, 1, 1))
        self.max_val = nn.Parameter(torch.zeros(1, ch, 1, 1))
        # eps set by weight loader from the ONNX Abs_* initialiser (per-block scalar)
        self.eps: float = 1.0
        # Set to True by _CapturedGraph during CUDA graph capture to force the
        # PyTorch fallback path (Triton kernels may cause silent capture failures).
        self._triton_disabled: bool = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Triton fused path: fp16 only (2 memory passes vs 5 PyTorch ops).
        # Disabled during CUDA graph capture (see _CapturedGraph._triton_disabled).
        if (
            x.dtype == torch.float16
            and _TRITON_RMSNORMMAX is not None
            and not self._triton_disabled
        ):
            return _TRITON_RMSNORMMAX(x, self.gamma, self.beta, self.max_val, self.eps)
        # PyTorch fallback (fp32 or Triton unavailable)
        # Use FP32 for accumulation to avoid overflow (x*x) if activations > 256
        x_f32 = x.float()
        rms = torch.sqrt(torch.mean(x_f32 * x_f32, dim=[2, 3], keepdim=True) + self.eps)
        y = (x_f32 / rms) * self.gamma.float() + self.beta.float()
        y = torch.maximum(y, self.max_val.float())
        return y.to(x.dtype)


# ---------------------------------------------------------------------------
# Encoder components
# ---------------------------------------------------------------------------


class _EncBlock(nn.Module):
    """Encoder stage: n_convs × (Conv3×3 + RMSNormMax) + depthwise strided-Conv.

    Returns (downsampled_feature, skip_feature).
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        n_convs: int,
        ds_k: int,
        ds_pad: Tuple[int, int, int, int],  # (left, right, top, bottom) for F.pad
    ) -> None:
        super().__init__()
        self.convs: nn.ModuleList = nn.ModuleList()
        self.norms: nn.ModuleList = nn.ModuleList()
        ch = in_ch
        for _ in range(n_convs):
            self.convs.append(nn.Conv2d(ch, out_ch, 3, padding=1, bias=True))
            self.norms.append(_RMSNormMax(out_ch))
            ch = out_ch
        # Depthwise strided conv — no bias; uses explicit F.pad for asymmetric padding
        self.ds = nn.Conv2d(
            out_ch, out_ch, ds_k, stride=2, padding=0, groups=out_ch, bias=False
        )
        self._ds_pad = ds_pad

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        for conv, norm in zip(self.convs, self.norms):
            x = norm(conv(x))
        skip = x
        x = self.ds(F.pad(x, self._ds_pad))
        return x, skip


# ---------------------------------------------------------------------------
# Decoder components
# ---------------------------------------------------------------------------


class _UpBlock(nn.Module):
    """ConvTranspose upsample → additive bias → RMSNormMax.

    The ConvTranspose in the original ONNX has no built-in bias; instead a
    separate Add node with a [1, C, 1, 1] initialiser follows immediately.

    ONNX padding contract:
      All 6 ONNX ConvTranspose nodes use pads=[0, 0, 1, 1], which means the
      raw (2*H+1) × (2*W+1) output is trimmed by removing the LAST row and
      LAST column only — i.e. output = raw[:, :, :-1, :-1].

      PyTorch ConvTranspose2d(padding=1, output_padding=1) uses *symmetric*
      cropping (removes 1 from all four sides) then adds a zero row/col on
      one side.  This produces a 1-pixel spatial shift relative to the ONNX,
      causing misalignment between upsampled features and encoder skips —
      resulting in "half-face" mask cut-off artefacts.

      Fix: use ConvTranspose2d(padding=0) so the raw (2H+1)×(2W+1) output is
      produced without any cropping, then explicitly slice [:-1, :-1] to
      match the ONNX pads=[0,0,1,1] behaviour exactly.
    """

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        # padding=0: produces raw (2*H+1)×(2*W+1); we slice to (2H)×(2W) below.
        self.ct = nn.ConvTranspose2d(in_ch, out_ch, 3, stride=2, padding=0, bias=False)
        self.bias = nn.Parameter(torch.zeros(1, out_ch, 1, 1))
        self.norm = _RMSNormMax(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Slice :-1, :-1 replicates ONNX ConvTranspose pads=[0,0,1,1]:
        # keep rows [0:2H] and cols [0:2W] of the raw (2H+1)×(2W+1) output.
        return self.norm(self.ct(x)[:, :, :-1, :-1] + self.bias)


class _DecBlock(nn.Module):
    """Decoder stage: _UpBlock upsample + skip Concat + n_convs × (Conv3×3 + RMSNormMax)."""

    def __init__(
        self,
        in_ch: int,
        skip_ch: int,
        n_convs: int,
        first_out_ch: int,
        out_ch: int,
    ) -> None:
        super().__init__()
        self.up = _UpBlock(in_ch, in_ch // 2)
        concat_ch = in_ch // 2 + skip_ch
        self.convs: nn.ModuleList = nn.ModuleList()
        self.norms: nn.ModuleList = nn.ModuleList()
        ch = concat_ch
        for i in range(n_convs):
            o = first_out_ch if i == 0 else out_ch
            self.convs.append(nn.Conv2d(ch, o, 3, padding=1, bias=True))
            self.norms.append(_RMSNormMax(o))
            ch = o

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        for conv, norm in zip(self.convs, self.norms):
            x = norm(conv(x))
        return x


# ---------------------------------------------------------------------------
# Full XSeg U-Net
# ---------------------------------------------------------------------------


class XSegTorch(nn.Module):
    """FP16-capable PyTorch reimplementation of XSeg_model.onnx.

    Input : (1, 3, 256, 256) float32  — RGB face crop, values in [0, 1]
    Output: (1, 1, 256, 256) float32  — sigmoid segmentation mask

    The model is always returned as float32 regardless of compute dtype
    (matching the application's expectation).
    """

    def __init__(self) -> None:
        super().__init__()

        # ── Encoder ────────────────────────────────────────────────────────
        # ds_pad = (left, right, top, bottom) for F.pad before depthwise conv
        # Derived from ONNX pads=[top, left, bottom, right]:
        #   enc0 pads=[1,1,2,2]  → (1,2,1,2)
        #   enc1 pads=[1,1,1,1]  → (1,1,1,1)
        #   enc2–5 pads=[0,0,1,1]→ (0,1,0,1)
        self.enc0 = _EncBlock(3, 32, 2, ds_k=4, ds_pad=(1, 2, 1, 2))
        self.enc1 = _EncBlock(32, 64, 2, ds_k=3, ds_pad=(1, 1, 1, 1))
        self.enc2 = _EncBlock(64, 128, 2, ds_k=2, ds_pad=(0, 1, 0, 1))
        self.enc3 = _EncBlock(128, 256, 3, ds_k=2, ds_pad=(0, 1, 0, 1))
        self.enc4 = _EncBlock(256, 256, 3, ds_k=2, ds_pad=(0, 1, 0, 1))
        self.enc5 = _EncBlock(256, 256, 3, ds_k=2, ds_pad=(0, 1, 0, 1))

        # ── Bottleneck FC bridge ────────────────────────────────────────────
        # Spatial: 4×4×256 = 4096 → Linear(512) → Linear(4096) → reshape 256×4×4
        self.dense1 = nn.Linear(4096, 512, bias=True)
        self.dense2 = nn.Linear(512, 4096, bias=True)

        # ── Decoder (skip channels come from enc5→0 in order) ──────────────
        # dec5: in=256, skip=256ch(enc5), up→128, cat→384, 3×Conv(384→256→256→256)
        self.dec5 = _DecBlock(256, 256, 3, first_out_ch=256, out_ch=256)
        # dec4: in=256, skip=256ch(enc4), up→128, cat→384, 3×Conv(384→256→256→256)
        self.dec4 = _DecBlock(256, 256, 3, first_out_ch=256, out_ch=256)
        # dec3: in=256, skip=256ch(enc3), up→128, cat→384, 3×Conv(384→256→256→256)
        self.dec3 = _DecBlock(256, 256, 3, first_out_ch=256, out_ch=256)
        # dec2: in=256, skip=128ch(enc2), up→128, cat→256, 2×Conv(256→128→128)
        self.dec2 = _DecBlock(256, 128, 2, first_out_ch=128, out_ch=128)
        # dec1: in=128, skip=64ch(enc1),  up→64,  cat→128, 2×Conv(128→64→64)
        self.dec1 = _DecBlock(128, 64, 2, first_out_ch=64, out_ch=64)
        # dec0: in=64,  skip=32ch(enc0),  up→32,  cat→64,  2×Conv(64→32→32)
        self.dec0 = _DecBlock(64, 32, 2, first_out_ch=32, out_ch=32)

        # ── Output head ────────────────────────────────────────────────────
        self.out_conv = nn.Conv2d(32, 1, 3, padding=1, bias=True)

        self._compute_dtype: torch.dtype = torch.float32

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self._compute_dtype)

        # Encoder — save skip features before downsampling
        x, s0 = self.enc0(x)  # 256×256 → 128×128, s0: 32ch  @256×256
        x, s1 = self.enc1(x)  # 128×128 →  64×64,  s1: 64ch  @128×128
        x, s2 = self.enc2(x)  #  64×64  →  32×32,  s2: 128ch @ 64×64
        x, s3 = self.enc3(x)  #  32×32  →  16×16,  s3: 256ch @ 32×32
        x, s4 = self.enc4(x)  #  16×16  →   8×8,   s4: 256ch @ 16×16
        x, s5 = self.enc5(x)  #   8×8   →   4×4,   s5: 256ch @  8×8

        # Bottleneck FC bridge
        B, C, H, W = x.shape
        x = x.flatten(1)  # (B, 4096)
        x = self.dense1(x)  # (B, 512)
        x = self.dense2(x)  # (B, 4096)
        x = x.reshape(B, C, H, W)  # (B, 256, 4, 4)

        # Decoder — upsample and fuse with skips (deepest first)
        x = self.dec5(x, s5)  #  4×4  →   8×8
        x = self.dec4(x, s4)  #  8×8  →  16×16
        x = self.dec3(x, s3)  # 16×16 →  32×32
        x = self.dec2(x, s2)  # 32×32 →  64×64
        x = self.dec1(x, s1)  # 64×64 → 128×128
        x = self.dec0(x, s0)  # 128×128→ 256×256

        # Output
        x = torch.sigmoid(self.out_conv(x))
        return x.float()  # always float32 output

    # ------------------------------------------------------------------
    # Class-method constructor
    # ------------------------------------------------------------------

    @classmethod
    def from_onnx(
        cls,
        onnx_path: str | Path,
        compute_dtype: torch.dtype = torch.float16,
    ) -> "XSegTorch":
        """Build XSegTorch and load all weights from XSeg_model.onnx.

        Args:
            onnx_path:     Path to XSeg_model.onnx.
            compute_dtype: Internal compute dtype (default: float16).

        Returns:
            XSegTorch instance with weights loaded, converted to compute_dtype.
        """
        import onnx  # type: ignore

        onnx_path = Path(onnx_path)
        print(f"[XSegTorch] Loading ONNX model from {onnx_path} …")
        onnx_model = onnx.load(str(onnx_path))

        model = cls()
        _load_all_params(model, onnx_model)

        model._compute_dtype = compute_dtype
        model = model.to(compute_dtype)
        model.eval()

        total = sum(p.numel() for p in model.parameters())
        print(
            f"[XSegTorch] Loaded {total:,} parameters | compute dtype: {compute_dtype}"
        )
        model._visomaster_onnx_path = str(onnx_path)
        return model


# ---------------------------------------------------------------------------
# Weight-loading helpers
# ---------------------------------------------------------------------------


def _np(init):
    """Extract float32 numpy array from an ONNX TensorProto initialiser."""
    import numpy as np

    if init.raw_data:
        return np.frombuffer(init.raw_data, dtype=np.float32).copy()
    return np.array(init.float_data, dtype=np.float32)


def _conv_modules_in_forward_order(model: XSegTorch) -> List[nn.Conv2d]:
    """Return all 43 Conv2d / ConvTranspose2d in ONNX topological order.

    Breakdown:
         2  enc0  (conv01, conv02)
         1  enc0.ds
         2  enc1  (conv11, conv12)
         1  enc1.ds
         2  enc2  (conv21, conv22)
         1  enc2.ds
         3  enc3  (conv31–33)
         1  enc3.ds
         3  enc4  (conv41–43)
         1  enc4.ds
         3  enc5  (conv51–53)
         1  enc5.ds
         1  dec5.up.ct  (ConvTranspose)
         3  dec5 convs  (uconv53, uconv52, uconv51)
         1  dec4.up.ct
         3  dec4 convs
         1  dec3.up.ct
         3  dec3 convs
         1  dec2.up.ct
         2  dec2 convs
         1  dec1.up.ct
         2  dec1 convs
         1  dec0.up.ct
         2  dec0 convs
         1  out_conv
        --
        43  total
    """
    mods: List = []

    def _add_enc(enc: _EncBlock) -> None:
        for c in enc.convs:
            mods.append(c)
        mods.append(enc.ds)

    def _add_dec(dec: _DecBlock) -> None:
        mods.append(dec.up.ct)
        for c in dec.convs:
            mods.append(c)

    _add_enc(model.enc0)
    _add_enc(model.enc1)
    _add_enc(model.enc2)
    _add_enc(model.enc3)
    _add_enc(model.enc4)
    _add_enc(model.enc5)

    _add_dec(model.dec5)
    _add_dec(model.dec4)
    _add_dec(model.dec3)
    _add_dec(model.dec2)
    _add_dec(model.dec1)
    _add_dec(model.dec0)

    mods.append(model.out_conv)

    assert len(mods) == 43, f"Expected 43 conv-type modules, got {len(mods)}"
    return mods


def _rms_norm_mods_in_forward_order(model: XSegTorch) -> List[_RMSNormMax]:
    """Return all 36 _RMSNormMax modules in ONNX topological order.

    Order: enc0 norms → enc1 → enc2 → enc3 → enc4 → enc5
           → dec5 (up_norm + conv norms) → dec4 → dec3 → dec2 → dec1 → dec0
    """
    mods: List[_RMSNormMax] = []

    def _add_enc_norms(enc: _EncBlock) -> None:
        for n in enc.norms:
            mods.append(n)

    def _add_dec_norms(dec: _DecBlock) -> None:
        mods.append(dec.up.norm)
        for n in dec.norms:
            mods.append(n)

    _add_enc_norms(model.enc0)
    _add_enc_norms(model.enc1)
    _add_enc_norms(model.enc2)
    _add_enc_norms(model.enc3)
    _add_enc_norms(model.enc4)
    _add_enc_norms(model.enc5)

    _add_dec_norms(model.dec5)
    _add_dec_norms(model.dec4)
    _add_dec_norms(model.dec3)
    _add_dec_norms(model.dec2)
    _add_dec_norms(model.dec1)
    _add_dec_norms(model.dec0)

    assert len(mods) == 36, f"Expected 36 _RMSNormMax modules, got {len(mods)}"
    return mods


def _ct_bias_params_in_forward_order(model: XSegTorch) -> List[nn.Parameter]:
    """Return the 6 ConvTranspose additive-bias Parameters in decoder order."""
    return [
        model.dec5.up.bias,
        model.dec4.up.bias,
        model.dec3.up.bias,
        model.dec2.up.bias,
        model.dec1.up.bias,
        model.dec0.up.bias,
    ]


def _load_all_params(model: XSegTorch, onnx_model) -> None:
    """Load all weights from the ONNX graph into model (in float32; caller converts).

    Strategy:
      - Conv / ConvTranspose weights (and biases where present): positional,
        matched to _conv_modules_in_forward_order().
      - ConvTranspose additive biases (separate Add nodes): positional, matched
        to _ct_bias_params_in_forward_order() by detecting Add nodes whose tensor
        input was produced by a ConvTranspose node.
      - _RMSNormMax eps / gamma / beta / max_val: positional, extracted from
        the ONNX graph in topological order (eps from Add+Abs, gamma from Mul,
        beta from Add, max_val from Max nodes — all with [1,C,1,1] initialisers).
      - Linear weights / biases: matched by shape (dense1↔[4096,512], dense2↔[512,4096]).
    """

    init_map = {i.name: i for i in onnx_model.graph.initializer}

    # Build tensor-name → producing-node map (for ConvTranspose bias detection)
    output_to_node = {}
    for node in onnx_model.graph.node:
        for out in node.output:
            output_to_node[out] = node

    conv_mods = _conv_modules_in_forward_order(model)
    norm_mods = _rms_norm_mods_in_forward_order(model)
    ct_biases = _ct_bias_params_in_forward_order(model)

    conv_idx = 0
    norm_idx = 0
    ct_bias_idx = 0
    matmul_idx = 0

    for node in onnx_model.graph.node:
        # ── Conv / ConvTranspose weights & biases ─────────────────────────
        if node.op_type in ("Conv", "ConvTranspose"):
            if conv_idx >= len(conv_mods):
                raise RuntimeError(
                    f"More Conv/ConvTranspose nodes than expected ({len(conv_mods)})"
                )
            mod = conv_mods[conv_idx]
            conv_idx += 1

            w_init = init_map[node.input[1]]
            w = _np(w_init).reshape(list(w_init.dims))
            with torch.no_grad():
                mod.weight.copy_(torch.from_numpy(w))

            if len(node.input) > 2 and node.input[2]:
                b = _np(init_map[node.input[2]])
                with torch.no_grad():
                    mod.bias.copy_(torch.from_numpy(b))

        # ── FC weights ────────────────────────────────────────────────────
        elif node.op_type == "MatMul":
            w_init = init_map[node.input[1]]
            w = _np(w_init).reshape(list(w_init.dims))
            # ONNX MatMul: Y = X @ W  →  W_torch = W.T  (nn.Linear stores [out, in])
            with torch.no_grad():
                if matmul_idx == 0:
                    model.dense1.weight.copy_(torch.from_numpy(w.T))
                else:
                    model.dense2.weight.copy_(torch.from_numpy(w.T))
            matmul_idx += 1

        # ── Mul nodes: gamma [1,C,1,1] ────────────────────────────────────
        elif node.op_type == "Mul":
            for inp in node.input:
                init = init_map.get(inp)
                if init is None:
                    continue
                d = list(init.dims)
                if len(d) == 4 and d[0] == 1 and d[2] == 1 and d[3] == 1:
                    v = _np(init).reshape(d)
                    with torch.no_grad():
                        norm_mods[norm_idx].gamma.copy_(torch.from_numpy(v))

        # ── Add nodes: eps [1], beta [1,C,1,1], FC biases, ConvTranspose bias
        elif node.op_type == "Add":
            for inp in node.input:
                init = init_map.get(inp)
                if init is None:
                    continue
                d = list(init.dims)

                if d == [1]:
                    # norm eps (Abs_* initialiser) — set as float attr
                    norm_mods[norm_idx].eps = float(_np(init)[0])

                elif d == [1, 512]:
                    # dense1 bias
                    with torch.no_grad():
                        model.dense1.bias.copy_(torch.from_numpy(_np(init).flatten()))

                elif d == [1, 4096]:
                    # dense2 bias
                    with torch.no_grad():
                        model.dense2.bias.copy_(torch.from_numpy(_np(init).flatten()))

                elif len(d) == 4 and d[0] == 1 and d[2] == 1 and d[3] == 1:
                    # Could be ConvTranspose additive bias OR norm beta.
                    # Distinguish by checking whether the tensor input was
                    # produced by a ConvTranspose node.
                    tensor_inp = next(
                        (x for x in node.input if x not in init_map), None
                    )
                    is_ct_bias = (
                        tensor_inp is not None
                        and tensor_inp in output_to_node
                        and output_to_node[tensor_inp].op_type == "ConvTranspose"
                    )
                    v = _np(init).reshape(d)
                    if is_ct_bias:
                        with torch.no_grad():
                            ct_biases[ct_bias_idx].copy_(torch.from_numpy(v))
                        ct_bias_idx += 1
                    else:
                        with torch.no_grad():
                            norm_mods[norm_idx].beta.copy_(torch.from_numpy(v))

        # ── Max nodes: max_val [1,C,1,1] — last param of each norm block ──
        elif node.op_type == "Max":
            for inp in node.input:
                init = init_map.get(inp)
                if init is None:
                    continue
                d = list(init.dims)
                if len(d) == 4 and d[0] == 1 and d[2] == 1 and d[3] == 1:
                    v = _np(init).reshape(d)
                    with torch.no_grad():
                        norm_mods[norm_idx].max_val.copy_(torch.from_numpy(v))
                    norm_idx += 1  # max_val is the last norm param: advance block index

    # ── Sanity checks ─────────────────────────────────────────────────────
    if conv_idx != len(conv_mods):
        raise RuntimeError(
            f"Conv count mismatch: expected {len(conv_mods)}, loaded {conv_idx}"
        )
    if norm_idx != len(norm_mods):
        raise RuntimeError(
            f"Norm block count mismatch: expected {len(norm_mods)}, loaded {norm_idx}"
        )
    if ct_bias_idx != len(ct_biases):
        raise RuntimeError(
            f"ConvTranspose bias count mismatch: expected {len(ct_biases)}, loaded {ct_bias_idx}"
        )


# ---------------------------------------------------------------------------
# CUDA graph runner
# ---------------------------------------------------------------------------


class _CapturedGraph:
    """Single fixed-size CUDA graph for (1, 3, 256, 256) input.

    Capture strategy
    ----------------
    Triton-fused _RMSNormMax kernels may allocate internal CUDA scratch buffers
    via cudaMalloc (outside PyTorch's cudaMallocAsync pool) and are not always
    capturable.  The Triton path is disabled on every _RMSNormMax module during
    graph capture so that only standard PyTorch CUDA ops (always capturable) are
    recorded.

    Stream ordering
    ---------------
    All operations — copy_, replay, clone — run on the default CUDA stream.
    Intra-stream ordering ensures each step completes before the next begins,
    so no cross-stream races or explicit synchronisation events are needed.

    Fallback to eager mode
    ----------------------
    If graph capture raises (e.g., a non-capturable op remains in the environment
    even with Triton disabled), the runner transparently falls back to eager FP16
    execution.  Correctness is preserved; only the kernel-launch speedup is lost.
    """

    def __init__(self, model: XSegTorch, warmup: int = 3) -> None:
        device = next(model.parameters()).device
        self._inp = torch.zeros(1, 3, 256, 256, dtype=torch.float32, device=device)
        # Always keep a strong reference to the model.  The captured CUDA graph
        # references the model's weight tensors by their CUDA memory addresses.
        # If the model is garbage-collected, those addresses become dangling →
        # replay reads freed memory → NaN.  (In eager-fallback mode the reference
        # is also needed to call model.forward() on every inference.)
        self._model: XSegTorch = model
        self._eager = False

        # Disable Triton on every _RMSNormMax during BOTH warmup AND capture.
        #
        # Why disable during warmup?
        #   _rmsnormmax_fwd has 6 BLOCK-size variants (32/64/128/256/512/1024).
        #   When Triton is enabled and any variant has not yet been JIT-compiled,
        #   NVRTC allocates large temporary GPU compilation buffers (compiler
        #   workspace, PTX staging, cubin upload), one per variant.  On a GPU
        #   already loaded with other Custom-provider models this triggers a
        #   "spike to maximum VRAM" exactly at the moment the runner is first
        #   built (i.e. at recording start, when _get_xseg_runner() is called
        #   lazily for the first frame).
        #   Keeping Triton disabled during warmup prevents any JIT compilation
        #   here.  Triton compiles from cache on the first actual inference call,
        #   where the compiled kernels are loaded cheaply from the
        #   build_kernels.py cache without GPU compilation buffers.
        #
        # Why disable during capture?
        #   Triton kernels use cudaMalloc (not cudaMallocAsync) for scratch
        #   memory.  cudaMalloc is not capturable by CUDA graphs and would cause
        #   the capture to fail or produce a broken graph.
        norm_mods = [m for m in model.modules() if isinstance(m, _RMSNormMax)]
        for nm in norm_mods:
            nm._triton_disabled = True
        try:
            # Warmup with Triton disabled: settle cuDNN algorithm selection and
            # warm up PyTorch's caching allocator without NVRTC JIT spikes.
            with torch.no_grad():
                for _ in range(warmup):
                    _ = model(self._inp)

            # Return all freed warmup tensors to CUDA before creating the graph
            # pool.  This reduces memory fragmentation and lowers the initial
            # reservation size of the private graph pool.
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

            self._graph = torch.cuda.CUDAGraph()
            self._stream = torch.cuda.Stream()
            with (
                torch.no_grad(),
                torch.cuda.graph(
                    self._graph, stream=self._stream, capture_error_mode="relaxed"
                ),
            ):
                self._out = model(self._inp)
            torch.cuda.synchronize()
        except Exception as exc:
            print(
                f"[XSegTorch] CUDA graph capture failed ({exc}). "
                "Falling back to eager FP16 execution."
            )
            self._eager = True
            with torch.no_grad():
                self._out = model(self._inp)
        finally:
            for nm in norm_mods:
                nm._triton_disabled = False

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self._eager:
            with torch.no_grad():
                return self._model(x)  # type: ignore[union-attr]

        # copy_, replay(), and clone() are all on the default stream; intra-stream
        # ordering guarantees the input is fully copied before the graph reads it,
        # and the graph is fully finished before clone reads the output.
        self._inp.copy_(x)
        self._graph.replay()
        return self._out.clone()


def build_cuda_graph_runner(
    model: XSegTorch, warmup: int = 3, torch_compile: bool = False
) -> _CapturedGraph:
    """Capture a CUDA graph for model and return a callable runner."""
    if torch_compile:
        try:
            from custom_kernels.compile_utils import apply_torch_compile
            device = next(model.parameters()).device
            example_inp = torch.zeros((1, 3, 256, 256), dtype=torch.float32, device=device)
            # default avoids the Triton MLIR AV crash (0xC0000005 on Windows sm_89)
            # that mode='reduce-overhead' can trigger in the subprocess ptxas optimizer.
            compiled = apply_torch_compile(model, example_inp, compile_mode="default")
            print("[XSegTorch] torch.compile default done.")
            return compiled
        except Exception as e:
            print(f"[XSegTorch] torch.compile failed ({e!s:.120}), falling back to CUDA graph.")
            # Fall through to CUDA graph capture below
    return _CapturedGraph(model, warmup)
