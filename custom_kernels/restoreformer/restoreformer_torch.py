"""
RestoreFormerPlusPlus FP16 PyTorch reimplementation with Triton fused kernels.

Architecture: VQ-GAN encoder/decoder with encoder→decoder skip connections.
  Input : (1,3,512,512)f32  →  Output: (1,3,512,512)f32
  VQ codebook: 1024 codes × 256 dims
  87 GN affine pairs (81 InstanceNorm nodes; 3 nodes share InstanceNorm across encoder+decoder)

Config (reverse-engineered from RestoreFormerPlusPlus.fp16.onnx):
  ch=64, ch_mult=[1,2,2,4,4,8], num_res_blocks=2
  z_channels=256, n_embed=1024, embed_dim=256

Cross-attention skip connections (encoder features as Q, decoder features as K/V):
  decoder.mid.attn_1   : Q from enc.mid.block_2 output  (512ch, 16×16)
  decoder.up[5].attn[i]: Q from enc.mid.block_1 output  (512ch, 16×16)
  decoder.up[4].attn[i]: Q from enc.down[4] block output (256ch, 32×32)
"""

from __future__ import annotations

import pathlib
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Triton kernel (optional)
# ---------------------------------------------------------------------------
try:
    import sys as _sys

    _sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))
    from custom_kernels.triton_ops import triton_group_norm_silu as _tgns

    _TRITON_OK = True
except Exception:
    _TRITON_OK = False


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------


class _GN(nn.Module):
    """GroupNorm(32) with optional Triton-fused SiLU."""

    def __init__(self, num_channels: int):
        super().__init__()
        self.gn = nn.GroupNorm(32, num_channels, eps=1e-6, affine=True)

    def forward(self, x: torch.Tensor, fuse_silu: bool = False) -> torch.Tensor:
        if _TRITON_OK and x.is_cuda and x.dtype == torch.float16:
            return _tgns(x, self.gn.weight, self.gn.bias, fuse_silu=fuse_silu)
        h = self.gn(x.float() if x.dtype == torch.float16 else x)
        if x.dtype == torch.float16:
            h = h.half()
        if fuse_silu:
            h = F.silu(h)
        return h


class _ResBlock(nn.Module):
    """GN-SiLU-Conv × 2 with optional channel shortcut."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.norm1 = _GN(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = _GN(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.nin_shortcut: Optional[nn.Conv2d] = (
            nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x, fuse_silu=True)
        h = self.conv1(h)
        h = self.norm2(h, fuse_silu=True)
        h = self.conv2(h)
        if self.nin_shortcut is not None:
            x = self.nin_shortcut(x)
        return x + h


_ATTN_HEADS = 4  # all attention blocks use 4 heads


class _AttnBlock(nn.Module):
    """4-head spatial self-attention matching ONNX layout."""

    def __init__(self, in_ch: int):
        super().__init__()
        self.norm = _GN(in_ch)
        self.q = nn.Conv2d(in_ch, in_ch, 1)
        self.k = nn.Conv2d(in_ch, in_ch, 1)
        self.v = nn.Conv2d(in_ch, in_ch, 1)
        self.proj_out = nn.Conv2d(in_ch, in_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        B, C, H, W = h.shape
        nh = _ATTN_HEADS
        hd = C // nh  # head_dim; scale = 1/sqrt(hd) via SDPA
        # Reshape: (B,C,H,W) → (B,nh,hd,H*W) → transpose → (B,nh,H*W,hd)
        q = self.q(h).view(B, nh, hd, H * W).permute(0, 1, 3, 2)
        k = self.k(h).view(B, nh, hd, H * W).permute(0, 1, 3, 2)
        v = self.v(h).view(B, nh, hd, H * W).permute(0, 1, 3, 2)
        out = F.scaled_dot_product_attention(q, k, v)  # (B,nh,H*W,hd)
        # view(B,H,W,C) → permute(0,3,1,2) produces channels-last (NHWC) strides;
        # .contiguous() restores NCHW layout so downstream Triton GN is correct.
        out = (
            out.permute(0, 2, 1, 3)
            .contiguous()
            .view(B, H, W, C)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        return x + self.proj_out(out)


class _CrossAttnBlock(nn.Module):
    """
    4-head spatial cross-attention matching ONNX layout.

    Q  from enc_skip (normalised by norm2),
    K/V from current decoder features (normalised by norm1).
    Both enc_skip and x must have the same spatial size and channel count.
    """

    def __init__(self, in_ch: int, enc_ch: int):
        super().__init__()
        self.norm1 = _GN(in_ch)  # K, V — current decoder features
        self.norm2 = _GN(enc_ch)  # Q   — encoder skip features
        self.q = nn.Conv2d(enc_ch, in_ch, 1)
        self.k = nn.Conv2d(in_ch, in_ch, 1)
        self.v = nn.Conv2d(in_ch, in_ch, 1)
        self.proj_out = nn.Conv2d(in_ch, in_ch, 1)

    def forward(self, x: torch.Tensor, enc_skip: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        enc_n = self.norm2(enc_skip)
        B, C, H, W = h.shape
        nh = _ATTN_HEADS
        hd = C // nh
        # Q from enc_skip, K/V from decoder features
        q = self.q(enc_n).view(B, nh, hd, H * W).permute(0, 1, 3, 2)
        k = self.k(h).view(B, nh, hd, H * W).permute(0, 1, 3, 2)
        v = self.v(h).view(B, nh, hd, H * W).permute(0, 1, 3, 2)
        out = F.scaled_dot_product_attention(q, k, v)  # (B,nh,H*W,hd)
        # view(B,H,W,C) → permute(0,3,1,2) produces channels-last (NHWC) strides;
        # .contiguous() restores NCHW layout so downstream Triton GN is correct.
        out = (
            out.permute(0, 2, 1, 3)
            .contiguous()
            .view(B, H, W, C)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        return x + self.proj_out(out)


class _Downsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (0, 1, 0, 1)))


class _Upsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.interpolate(x, scale_factor=2.0, mode="nearest"))


# ---------------------------------------------------------------------------
# Encoder / Decoder stage blocks
# ---------------------------------------------------------------------------


class _DownBlock(nn.Module):
    """Encoder stage with interleaved ResBlocks + AttnBlocks."""

    def __init__(self, in_ch: int, out_ch: int, n_res: int, n_attn: int, has_ds: bool):
        super().__init__()
        self.block = nn.ModuleList(
            [_ResBlock(in_ch if i == 0 else out_ch, out_ch) for i in range(n_res)]
        )
        self.attn = nn.ModuleList([_AttnBlock(out_ch) for _ in range(n_attn)])
        self.downsample: Optional[_Downsample] = _Downsample(out_ch) if has_ds else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.attn:
            for b, a in zip(self.block, self.attn):
                x = b(x)
                x = a(x)
        else:
            for b in self.block:
                x = b(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class _UpBlock(nn.Module):
    """Decoder stage (no attention) — used for up.0–up.3."""

    def __init__(self, in_ch: int, out_ch: int, n_res: int, has_us: bool):
        super().__init__()
        self.block = nn.ModuleList(
            [_ResBlock(in_ch if i == 0 else out_ch, out_ch) for i in range(n_res)]
        )
        self.upsample: Optional[_Upsample] = _Upsample(out_ch) if has_us else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for b in self.block:
            x = b(x)
        if self.upsample is not None:
            x = self.upsample(x)
        return x


class _UpBlockCrossAttn(nn.Module):
    """
    Decoder stage with interleaved ResBlocks + CrossAttnBlocks.
    Used for decoder up.5 (self-attn recast as cross-attn from enc.mid.block_1 skip)
    and up.4 (cross-attn from enc.down.4 skip).
    """

    def __init__(self, in_ch: int, out_ch: int, n_res: int, enc_ch: int, has_us: bool):
        super().__init__()
        self._enc_ch = enc_ch  # used for skip-connection routing in _Decoder.forward
        self.block = nn.ModuleList(
            [_ResBlock(in_ch if i == 0 else out_ch, out_ch) for i in range(n_res)]
        )
        self.attn = nn.ModuleList(
            [_CrossAttnBlock(out_ch, enc_ch) for _ in range(n_res)]
        )
        self.upsample: Optional[_Upsample] = _Upsample(out_ch) if has_us else None

    def forward(self, x: torch.Tensor, enc_skip: torch.Tensor) -> torch.Tensor:
        for b, a in zip(self.block, self.attn):
            x = b(x)
            x = a(x, enc_skip)
        if self.upsample is not None:
            x = self.upsample(x)
        return x


class _EncoderMidBlock(nn.Module):
    """Encoder bottleneck: ResBlock + Self-AttnBlock + ResBlock."""

    def __init__(self, ch: int):
        super().__init__()
        self.block_1 = _ResBlock(ch, ch)
        self.attn_1 = _AttnBlock(ch)
        self.block_2 = _ResBlock(ch, ch)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (output, enc_mid_attn_skip, enc_norm_out_skip)."""
        x = self.block_1(x)
        enc_mid_attn_skip = x  # input to attn_1 — used by dec.up[5]
        x = self.attn_1(x)
        x = self.block_2(x)
        enc_norm_out_skip = x  # input to enc.norm_out — used by dec.mid
        return x, enc_mid_attn_skip, enc_norm_out_skip


class _DecoderMidBlock(nn.Module):
    """Decoder bottleneck: ResBlock + Cross-AttnBlock + ResBlock."""

    def __init__(self, ch: int):
        super().__init__()
        self.block_1 = _ResBlock(ch, ch)
        self.attn_1 = _CrossAttnBlock(ch, ch)
        self.block_2 = _ResBlock(ch, ch)

    def forward(self, x: torch.Tensor, enc_norm_out_skip: torch.Tensor) -> torch.Tensor:
        x = self.block_1(x)
        x = self.attn_1(x, enc_norm_out_skip)
        x = self.block_2(x)
        return x


class _VQLayer(nn.Module):
    """Nearest-neighbour VQ codebook lookup (FP32 distances for precision)."""

    def __init__(self, n_codes: int, dim: int):
        super().__init__()
        self.embedding = nn.Embedding(n_codes, dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B, C, H, W = z.shape
        z_f = z.permute(0, 2, 3, 1).reshape(-1, C).float()
        e = self.embedding.weight.float()
        d = (
            z_f.pow(2).sum(1, keepdim=True)
            - 2.0 * (z_f @ e.t())
            + e.pow(2).sum(1).unsqueeze(0)
        )
        idx = d.argmin(dim=1)
        # .contiguous() ensures NCHW layout — without it the permute produces a
        # channels-last (NHWC) view that breaks the Triton GroupNorm kernel.
        return (
            self.embedding(idx)
            .to(z.dtype)
            .reshape(B, H, W, C)
            .permute(0, 3, 1, 2)
            .contiguous()
        )


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------


class _Encoder(nn.Module):
    """
    Encoder: conv_in → 6 DownBlocks → mid → norm_out+SiLU → conv_out
    Returns: (z, enc_skip4, enc_mid_attn_skip, enc_norm_out_skip)
      enc_skip4        : down[4] block output before downsample (256ch, 32×32)
      enc_mid_attn_skip: mid.block_1 output = input to mid.attn_1 (512ch, 16×16)
      enc_norm_out_skip: mid.block_2 output = input to enc.norm_out (512ch, 16×16)
    """

    def __init__(self):
        super().__init__()
        self.conv_in = nn.Conv2d(3, 64, 3, padding=1)
        self.down = nn.ModuleList(
            [
                _DownBlock(64, 64, 2, 0, True),  # down.0  512→256
                _DownBlock(64, 128, 2, 0, True),  # down.1  256→128
                _DownBlock(128, 128, 2, 0, True),  # down.2  128→64
                _DownBlock(128, 256, 2, 0, True),  # down.3   64→32
                _DownBlock(256, 256, 2, 0, True),  # down.4   32→16 (skip captured here)
                _DownBlock(256, 512, 2, 2, False),  # down.5   16→16 (2 attn, no ds)
            ]
        )
        self.mid = _EncoderMidBlock(512)
        self.norm_out = _GN(512)
        self.conv_out = nn.Conv2d(512, 256, 3, padding=1)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.conv_in(x)
        enc_skip4: torch.Tensor = h  # will be overwritten
        for i, d in enumerate(self.down):
            if i == 4:
                # Capture block output before downsample
                for b in d.block:
                    h = b(h)
                enc_skip4 = h
                if d.downsample is not None:
                    h = d.downsample(h)
            else:
                h = d(h)
        h, enc_mid_attn_skip, enc_norm_out_skip = self.mid(h)
        h = self.norm_out(h, fuse_silu=True)
        return self.conv_out(h), enc_skip4, enc_mid_attn_skip, enc_norm_out_skip


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------


class _Decoder(nn.Module):
    """
    Decoder: conv_in → mid → 6 UpBlocks (5→0) → norm_out+SiLU → conv_out
    Requires encoder skip connections for cross-attention in mid, up[5], up[4].
    """

    def __init__(self):
        super().__init__()
        self.conv_in = nn.Conv2d(256, 512, 3, padding=1)
        self.mid = _DecoderMidBlock(512)
        self.up = nn.ModuleList(
            [
                _UpBlock(128, 64, 3, False),  # up.0  no upsample
                _UpBlock(128, 128, 3, True),  # up.1
                _UpBlock(256, 128, 3, True),  # up.2
                _UpBlock(256, 256, 3, True),  # up.3
                _UpBlockCrossAttn(512, 256, 3, enc_ch=256, has_us=True),  # up.4
                _UpBlockCrossAttn(512, 512, 3, enc_ch=512, has_us=True),  # up.5
            ]
        )
        self.norm_out = _GN(64)
        self.conv_out = nn.Conv2d(64, 3, 3, padding=1)

    def forward(
        self,
        z_q: torch.Tensor,
        enc_skip4: torch.Tensor,
        enc_mid_attn_skip: torch.Tensor,
        enc_norm_out_skip: torch.Tensor,
    ) -> torch.Tensor:
        h = self.mid(self.conv_in(z_q), enc_norm_out_skip)
        for up in reversed(self.up):  # up[5] → up[4] → ... → up[0]
            if isinstance(up, _UpBlockCrossAttn):
                # up[5] uses enc_mid_attn_skip, up[4] uses enc_skip4
                # Distinguish by output channel: up[5] out_ch=512, up[4] out_ch=256
                skip = enc_mid_attn_skip if up._enc_ch == 512 else enc_skip4
                h = up(h, skip)
            else:
                h = up(h)
        h = self.norm_out(h, fuse_silu=True)
        return self.conv_out(h)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------


class RestoreFormerPlusPlusTorch(nn.Module):
    """FP16 PyTorch reimplementation of RestoreFormerPlusPlus.fp16.onnx."""

    def __init__(self):
        super().__init__()
        self.encoder = _Encoder()
        self.quant_conv = nn.Conv2d(256, 256, 1)
        self.quantize = _VQLayer(1024, 256)
        self.post_quant_conv = nn.Conv2d(256, 256, 1)
        self.decoder = _Decoder()
        self._compute_dtype: torch.dtype = torch.float16

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype_in = x.dtype
        h = x.to(self._compute_dtype)
        z, enc_skip4, enc_mid_attn_skip, enc_norm_out_skip = self.encoder(h)
        z_e = self.quant_conv(z)
        z_q = self.quantize(z_e)
        dec_in = self.post_quant_conv(z_q)
        out = self.decoder(dec_in, enc_skip4, enc_mid_attn_skip, enc_norm_out_skip)
        return out.to(dtype_in)

    @classmethod
    def from_onnx(
        cls,
        onnx_path: str,
        compute_dtype: torch.dtype = torch.float16,
    ) -> "RestoreFormerPlusPlusTorch":
        import onnx

        model = cls()
        model._compute_dtype = compute_dtype
        onnx_model = onnx.load(onnx_path)
        n_named = _load_named_params(model, onnx_model, compute_dtype)
        n_gn = _load_gn_params(model, onnx_model, compute_dtype)
        print(
            f"[RFP++] Named: {n_named}, GN: {n_gn}  "
            f"(dtype={compute_dtype}, "
            f"triton={'yes' if _TRITON_OK else 'no (PyTorch fallback)'})"
        )
        model._visomaster_onnx_path = str(onnx_path)
        return model.to(compute_dtype)


# ---------------------------------------------------------------------------
# GN module ordering helper
# ---------------------------------------------------------------------------


def _gn_modules_in_forward_order(model: RestoreFormerPlusPlusTorch) -> List[_GN]:
    """
    Return 87 _GN instances in the ONNX GN-pair order.

    The ONNX lowers GN as: InstanceNorm → Reshape → Mul(scale) → Add(bias).
    Three InstanceNorm nodes are SHARED between encoder and decoder via skip
    connections. When we iterate InstanceNorm nodes in ONNX graph order and
    expand ALL their Mul→Add affine consumers, we get 87 pairs in this order:

      Pairs  0-19 : encoder down.0–4 (5 stages × 4 GN = 20)
      Pairs 20-25 : encoder down.5 interleaved (b0n1,b0n2,a0, b1n1,b1n2,a1)
      Pairs 26-27 : encoder mid.block_1 (n1, n2)
      Pair  28    : encoder mid.attn_1.norm  ← shared InstanceNorm, 4 affines total
      Pairs 29-31 : decoder up[5].attn[0,1,2].norm2  (from same InstanceNorm)
      Pairs 32-33 : encoder mid.block_2 (n1, n2)
      Pair  34    : encoder norm_out  ← shared InstanceNorm, 2 affines total
      Pair  35    : decoder mid.attn_1.norm2  (from same InstanceNorm)
      Pairs 36-37 : decoder mid.block_1 (n1, n2)
      Pair  38    : decoder mid.attn_1.norm1
      Pairs 39-40 : decoder mid.block_2 (n1, n2)
      Pairs 41-49 : decoder up[5] interleaved (b0n1,b0n2,a0.norm1, … ×3)
      Pairs 50-52 : decoder up[4].attn[0,1,2].norm2  ← separate InstanceNorm, 3 affines
      Pairs 53-61 : decoder up[4] interleaved (b0n1,b0n2,a0.norm1, … ×3)
      Pairs 62-67 : decoder up[3] blocks (3 × 2)
      Pairs 68-73 : decoder up[2] blocks
      Pairs 74-79 : decoder up[1] blocks
      Pairs 80-85 : decoder up[0] blocks
      Pair  86    : decoder norm_out
    """
    gns: List[_GN] = []
    enc = model.encoder
    dec = model.decoder

    # ── Encoder down.0–4 (no attn) ──────────────────────────────────────────
    for stage in list(enc.down)[:5]:
        for blk in stage.block:
            gns.extend([blk.norm1, blk.norm2])

    # ── Encoder down.5 (interleaved: 2 blocks + 2 attn) ─────────────────────
    d5 = enc.down[5]
    for b, a in zip(d5.block, d5.attn):
        gns.extend([b.norm1, b.norm2])
        gns.append(a.norm)

    # ── Encoder mid.block_1 ──────────────────────────────────────────────────
    gns.extend([enc.mid.block_1.norm1, enc.mid.block_1.norm2])

    # ── Encoder mid.attn_1.norm  +  decoder up[5] attn norm2 ─────────────────
    # (all from the same shared InstanceNorm in ONNX)
    gns.append(enc.mid.attn_1.norm)
    for ca in dec.up[5].attn:
        gns.append(ca.norm2)

    # ── Encoder mid.block_2  +  encoder norm_out ─────────────────────────────
    gns.extend([enc.mid.block_2.norm1, enc.mid.block_2.norm2])
    gns.append(enc.norm_out)

    # ── Decoder mid.attn_1.norm2  (from shared enc.norm_out InstanceNorm) ────
    gns.append(dec.mid.attn_1.norm2)

    # ── Decoder mid blocks + attn ─────────────────────────────────────────────
    gns.extend([dec.mid.block_1.norm1, dec.mid.block_1.norm2])
    gns.append(dec.mid.attn_1.norm1)
    gns.extend([dec.mid.block_2.norm1, dec.mid.block_2.norm2])

    # ── Decoder up[5] interleaved (cross-attn: norm2 already above) ──────────
    for b, a in zip(dec.up[5].block, dec.up[5].attn):
        gns.extend([b.norm1, b.norm2])
        gns.append(a.norm1)

    # ── Decoder up[4] norm2 (3 affines from shared InstanceNorm) ─────────────
    for ca in dec.up[4].attn:
        gns.append(ca.norm2)

    # ── Decoder up[4] interleaved ─────────────────────────────────────────────
    for b, a in zip(dec.up[4].block, dec.up[4].attn):
        gns.extend([b.norm1, b.norm2])
        gns.append(a.norm1)

    # ── Decoder up[3] – up[0] (no attn) ──────────────────────────────────────
    for stage in reversed(list(dec.up)[:4]):  # up[3], up[2], up[1], up[0]
        for blk in stage.block:
            gns.extend([blk.norm1, blk.norm2])

    # ── Decoder norm_out ──────────────────────────────────────────────────────
    gns.append(dec.norm_out)

    return gns


# ---------------------------------------------------------------------------
# Weight loaders
# ---------------------------------------------------------------------------


def _load_named_params(
    model: RestoreFormerPlusPlusTorch,
    onnx_model,
    dtype: torch.dtype,
) -> int:
    from onnx import numpy_helper

    init_map = {init.name: init for init in onnx_model.graph.initializer}
    state = model.state_dict()
    updates = {}
    loaded = 0
    for pt_name, pt_tensor in state.items():
        if pt_name not in init_map:
            continue
        np_val = numpy_helper.to_array(init_map[pt_name])
        t = torch.from_numpy(np_val.copy())
        if t.shape != pt_tensor.shape:
            if t.numel() == pt_tensor.numel():
                t = t.reshape(pt_tensor.shape)
            else:
                continue
        updates[pt_name] = t.to(
            dtype if pt_tensor.is_floating_point() else pt_tensor.dtype
        )
        loaded += 1
    state.update(updates)
    model.load_state_dict(state, strict=False)
    return loaded


def _load_gn_params(
    model: RestoreFormerPlusPlusTorch,
    onnx_model,
    dtype: torch.dtype,
) -> int:
    """
    Load anonymous GroupNorm affine params from ONNX.

    Iterates InstanceNorm nodes in graph order.  For each node, finds ALL
    Mul→Add affine chains from its Reshape output (some nodes are shared by
    encoder and decoder, producing multiple affine pairs per InstanceNorm).
    Total: 87 pairs matching 87 _GN modules from _gn_modules_in_forward_order().
    """
    from onnx import numpy_helper

    input_to_nodes: dict = {}
    for node in onnx_model.graph.node:
        for inp in node.input:
            input_to_nodes.setdefault(inp, []).append(node)

    init_map = {init.name: init for init in onnx_model.graph.initializer}

    def first_consumer(tensor: str, op_type: str):
        for n in input_to_nodes.get(tensor, []):
            if n.op_type == op_type:
                return n
        return None

    gn_pairs = []
    for node in onnx_model.graph.node:
        if node.op_type != "InstanceNormalization":
            continue
        reshape = first_consumer(node.output[0], "Reshape")
        if reshape is None:
            continue
        mul_nodes = [
            n for n in input_to_nodes.get(reshape.output[0], []) if n.op_type == "Mul"
        ]
        for mul_node in mul_nodes:
            add_node = first_consumer(mul_node.output[0], "Add")
            if add_node is None:
                continue
            scale_init = next(
                (init_map[i] for i in mul_node.input if i in init_map), None
            )
            bias_init = next(
                (init_map[i] for i in add_node.input if i in init_map), None
            )
            if scale_init is None or bias_init is None:
                continue
            scale = torch.from_numpy(numpy_helper.to_array(scale_init).copy()).squeeze()
            bias = torch.from_numpy(numpy_helper.to_array(bias_init).copy()).squeeze()
            gn_pairs.append((scale, bias))

    gn_modules = _gn_modules_in_forward_order(model)
    count = min(len(gn_pairs), len(gn_modules))
    if len(gn_pairs) != len(gn_modules):
        print(
            f"[RFP++] WARNING: {len(gn_pairs)} GN pairs in ONNX, "
            f"{len(gn_modules)} expected — loading {count}"
        )

    for gn_mod, (scale, bias) in zip(gn_modules[:count], gn_pairs[:count]):
        with torch.no_grad():
            gn_mod.gn.weight.copy_(scale.to(gn_mod.gn.weight.dtype))
            gn_mod.gn.bias.copy_(bias.to(gn_mod.gn.bias.dtype))

    return count


# ---------------------------------------------------------------------------
# CUDA graph runner
# ---------------------------------------------------------------------------


def build_cuda_graph_runner(
    model: "RestoreFormerPlusPlusTorch",
    inp_shape: Tuple[int, ...] = (1, 3, 512, 512),
    warmup: int = 3,
    torch_compile: bool = False,
) -> "CUDAGraphRunner":
    """
    Capture model into a CUDA graph.

    Args:
        torch_compile: If True, wrap the model with ``torch.compile`` before
                       capturing the CUDA graph.  Requires Triton; adds ~60 s
                       one-time compile overhead but gives ~1.16x speedup over
                       the uncompiled CUDA-graph baseline.
    """
    if torch_compile:
        try:
            from custom_kernels.compile_utils import apply_torch_compile
            device = next(model.parameters()).device
            example_inp = torch.zeros(inp_shape, dtype=torch.float32, device=device)
            compiled = apply_torch_compile(model, example_inp)
            print("[restoreformer] torch.compile warmup done.")
            # Return compiled model directly — CUDA graph on top of torch.compile
            # fails on Windows (64-bit kernel handles overflow 32-bit C long).
            return compiled
        except Exception as e:
            print(f"[restoreformer] torch.compile failed ({e!s:.120}), falling back to CUDA graph.")

    return CUDAGraphRunner(model, inp_shape, warmup)


class CUDAGraphRunner:
    def __init__(
        self,
        model: "RestoreFormerPlusPlusTorch",
        inp_shape: Tuple[int, ...],
        warmup: int = 3,
    ):
        self.model = model
        self.inp_shape = inp_shape
        device = next(model.parameters()).device
        self._inp = torch.zeros(inp_shape, dtype=torch.float32, device=device)
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
            self._out = model(self._inp)
        torch.cuda.synchronize()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        self._inp.copy_(x, non_blocking=True)
        self._graph.replay()
        return self._out.clone()
