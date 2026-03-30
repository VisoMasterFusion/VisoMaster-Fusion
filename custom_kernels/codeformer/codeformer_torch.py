"""
CodeFormerTorch — FP16 PyTorch implementation of CodeFormer face restoration.

Model: codeformer_fp16.onnx
Input : (1, 3, 512, 512) float32   — face image, BGR, [0,1] normalised
Output: (1, 3, 512, 512) float32   — restored face, same range

Architecture
------------
Encoder     : 25-block stack (Conv, ResBlocks, Downsample, AttnBlock, quant_conv)
VQ codebook : 1024 codes × 256 dims
Transformer : 9 × TransformerEncoderLayer(d_model=512, nhead=8, ffn=1024, GELU)
Generator   : 25-block stack (Conv, ResBlocks, Upsample, AttnBlock)
FuseBlocks  : SFT skip-connections at scales 32, 64, 128, 256

Speedup tiers vs ORT FP32 CUDA EP  (RTX 4090):
  Tier 0b — ORT TensorRT EP FP32 (target)               10.75ms   1.99x
  Tier 1  — PyTorch FP32                                 32.72ms   0.65x
  Tier 2  — PyTorch FP16 + Triton GroupNorm+SiLU         16.26ms   1.32x
  Tier 3  — FP16 + Triton + CUDA graph (best for dyn w) 13.57ms   1.58x
  Tier 4  — FP16 + Triton + SDPA4D + GEMM + CUDA graph  13.54ms   1.58x
  Tier 5  — + NHWC (regresses due to GN layout overhead) 18.40ms   1.16x
  Best app tier: Tier 3 (dynamic w) or Tier 4 (fixed w, marginal gain)

Usage:
    model = CodeFormerTorch.from_onnx("model_assets/codeformer_fp16.onnx").cuda().eval()
    runner = build_cuda_graph_runner(model, inp_shape=(1, 3, 512, 512))
    output = runner(face_image_f32_cuda)      # (1,3,512,512) float32
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Triton Kernels (preferred; no MSVC required)
# ---------------------------------------------------------------------------
try:
    from custom_kernels.triton_ops import (
        TRITON_AVAILABLE as _TRITON_AVAILABLE,
        triton_group_norm_silu as _triton_gn_silu,
        triton_vq_dist as _triton_vq_dist,
        triton_fused_gfpgan_act as _triton_sft,
    )
except Exception:
    _TRITON_AVAILABLE = False
    _triton_gn_silu = None  # type: ignore[assignment]
    _triton_vq_dist = None  # type: ignore[assignment]
    _triton_sft = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Fused GroupNorm(32) — channels-last safe
# ---------------------------------------------------------------------------
class _GN(nn.GroupNorm):
    """GroupNorm(32) with optional fused SiLU via Triton kernel.

    Channels-last safe: the Triton kernel assumes NCHW contiguous layout.
    If the input is channels-last (NHWC), we temporarily convert to NCHW,
    run the kernel, then restore channels-last so surrounding Conv2d ops
    can stay on the fast NHWC cuDNN path.
    """

    def __init__(self, num_channels: int, fuse_silu: bool = False):
        super().__init__(
            num_groups=32, num_channels=num_channels, eps=1e-6, affine=True
        )
        self.fuse_silu = fuse_silu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        was_cl = x.is_contiguous(memory_format=torch.channels_last)
        x_in = x.contiguous() if was_cl else x  # NCHW for Triton

        if (
            _TRITON_AVAILABLE
            and _triton_gn_silu is not None
            and x_in.dtype == torch.float16
        ):
            out = _triton_gn_silu(
                x_in,
                self.weight,
                self.bias,
                num_groups=32,
                eps=self.eps,
                fuse_silu=self.fuse_silu,
            )
        else:
            out = super().forward(x_in.float()).to(x_in.dtype)
            if self.fuse_silu:
                out = out * torch.sigmoid(out)

        return out.contiguous(memory_format=torch.channels_last) if was_cl else out


# ---------------------------------------------------------------------------
# _GemmConv2d — nn.Conv2d with optional im2col + cuBLAS GEMM path
# ---------------------------------------------------------------------------
class _GemmConv2d(nn.Conv2d):
    """
    Drop-in nn.Conv2d replacement with an optional im2col + cuBLAS GEMM path.

    State-dict keys are *identical* to nn.Conv2d (inherits .weight / .bias
    nn.Parameter attributes), so weight loading works without any changes.

    GEMM path is active when *all* hold:
      • enable_gemm_mode() has been called (registers pre-flattened _w_flat)
      • stride == 1  (stride-2 ops stay on cuDNN)
      • H * W <= _GEMM_THRESHOLD  (avoids massive im2col buffers at large spatial)

    For CodeFormer's bottleneck 3×3 convolutions:
      • 512ch @ 16×16  → GEMM [512, 4608] × [4608, 256]   (~85 % TC efficiency)
      • 256ch @ 32×32  → GEMM [256, 2304] × [2304, 1024]
      • 256ch @ 64×64  → GEMM [256, 2304] × [2304, 4096]
    cuDNN's heuristic-chosen algorithm achieves ~47 % for these shapes.
    """

    _GEMM_THRESHOLD = 4096  # HW ≤ 4096 (64×64) triggers GEMM

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._use_gemm: bool = False

    def enable_gemm_mode(self) -> None:
        """Pre-flatten weights to [C_out, C_in*kH*kW].  Idempotent."""
        # Always use NCHW-contiguous order regardless of channels-last state.
        w = self.weight.data.contiguous()  # [C_out, C_in, kH, kW] NCHW
        w_flat = w.reshape(w.shape[0], -1).contiguous()
        if hasattr(self, "_w_flat"):
            self._w_flat.copy_(w_flat)
        else:
            self.register_buffer("_w_flat", w_flat)
        self._use_gemm = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._use_gemm:
            return super().forward(x)
        B, C_in, H, W = x.shape
        s = (
            self.stride[0]
            if isinstance(self.stride, (list, tuple))
            else int(self.stride)
        )
        if s != 1 or H * W > self._GEMM_THRESHOLD:
            return super().forward(x)  # fallback: cuDNN for large spatial / stride-2

        k = self.kernel_size[0]
        p = (
            self.padding[0]
            if isinstance(self.padding, (list, tuple))
            else int(self.padding)
        )

        # im2col: [1, C_in*k*k, H*W] — x.contiguous() ensures NCHW for unfold
        x_col = F.unfold(x.contiguous(), kernel_size=k, padding=p).squeeze(
            0
        )  # [K_in, HW]
        # cuBLAS GEMM: [C_out, K_in] × [K_in, HW] → [C_out, HW]
        out = torch.mm(self._w_flat, x_col)
        if self.bias is not None:
            out = out + self.bias.unsqueeze(1)
        return out.reshape(B, -1, H, W)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


def _silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class _ResBlock(nn.Module):
    """ResnetBlock: GN+SiLU → Conv3×3 → GN+SiLU → Conv3×3 (+optional 1×1 shortcut)."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.norm1 = _GN(in_ch, fuse_silu=True)
        self.conv1 = _GemmConv2d(in_ch, out_ch, 3, 1, 1)
        self.norm2 = _GN(out_ch, fuse_silu=True)
        self.conv2 = _GemmConv2d(out_ch, out_ch, 3, 1, 1)
        # shortcut — only when channels change (matches ONNX weight name 'conv_out')
        if in_ch != out_ch:
            self.conv_out = nn.Conv2d(in_ch, out_ch, 1, bias=True)
        else:
            self.conv_out = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.norm1(x))
        h = self.conv2(self.norm2(h))
        skip = self.conv_out(x) if self.conv_out is not None else x
        return skip + h


class _AttnBlock(nn.Module):
    """Single-head spatial self-attention (1×1 Conv Q/K/V)."""

    def __init__(self, in_ch: int):
        super().__init__()
        self.norm = _GN(in_ch, fuse_silu=False)
        self.q = nn.Conv2d(in_ch, in_ch, 1)
        self.k = nn.Conv2d(in_ch, in_ch, 1)
        self.v = nn.Conv2d(in_ch, in_ch, 1)
        self.proj_out = nn.Conv2d(in_ch, in_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channels-last safety: attn reshape ops require NCHW-contiguous
        was_cl = x.is_contiguous(memory_format=torch.channels_last)
        x_nchw = x.contiguous() if was_cl else x

        h = self.norm(x_nchw)
        b, c, ht, wt = h.shape
        hw = ht * wt

        # SDPA 4D path (b, 1, hw, c) — triggers Flash/MemEff attention (not "math" fallback)
        q = self.q(h).reshape(b, c, hw).permute(0, 2, 1).unsqueeze(1)  # (b, 1, hw, c)
        k = self.k(h).reshape(b, c, hw).permute(0, 2, 1).unsqueeze(1)  # (b, 1, hw, c)
        v = self.v(h).reshape(b, c, hw).permute(0, 2, 1).unsqueeze(1)  # (b, 1, hw, c)
        out = F.scaled_dot_product_attention(q, k, v)  # (b, 1, hw, c)
        out = out.squeeze(1).permute(0, 2, 1).reshape(b, c, ht, wt)  # (b, c, ht, wt)

        result = x_nchw + self.proj_out(out)
        return (
            result.contiguous(memory_format=torch.channels_last) if was_cl else result
        )


class _Downsample(nn.Module):
    """Stride-2 conv with asymmetric padding (pad right+bottom)."""

    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (0, 1, 0, 1)))


class _Upsample(nn.Module):
    """2× nearest upsample + Conv3×3."""

    def __init__(self, ch: int):
        super().__init__()
        self.conv = _GemmConv2d(ch, ch, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.interpolate(x, scale_factor=2.0, mode="nearest"))


class _BlockList(nn.Module):
    """Thin wrapper so `self.encoder.blocks[i]` matches ONNX name `encoder.blocks.i.*`."""

    def __init__(self, blocks: list):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)

    def __getitem__(self, idx):
        return self.blocks[idx]

    def __len__(self):
        return len(self.blocks)

    def __iter__(self):
        return iter(self.blocks)


class _VQLayer(nn.Module):
    """VQ nearest-neighbour code lookup (inference-only, no gradients)."""

    def __init__(self, n_codes: int, code_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(n_codes, code_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (b, code_dim, h, w) → quantised (b, code_dim, h, w)."""
        b, c, h, w = z.shape
        emb = self.embedding.weight.float()  # (n_codes, c)

        if _TRITON_AVAILABLE and _triton_vq_dist is not None:
            # Fused Triton VQ distance: avoids [HW, 1024] distance matrix allocation
            idx = _triton_vq_dist(z, emb)  # (b, h, w)
            idx = idx.reshape(-1)
        else:
            # Flatten spatial: (b*h*w, c)
            zf = z.permute(0, 2, 3, 1).reshape(-1, c).float()
            # Squared distances: ||z - e||^2 = ||z||^2 - 2<z,e> + ||e||^2
            d = (
                zf.pow(2).sum(1, keepdim=True) - 2 * zf @ emb.T + emb.pow(2).sum(1)
            )  # (b*h*w, n_codes)
            idx = d.argmin(dim=1)  # (b*h*w,)

        zq = emb[idx].reshape(b, h, w, c).permute(0, 3, 1, 2)
        return zq.to(z.dtype)


class _FuseBlock(nn.Module):
    """
    SFT (Spatial Feature Transform) block used to inject encoder skips.

    forward(enc_feat, gen_feat, w):
        fused  = encode_enc(cat([enc_feat, gen_feat]))
        scale  = self.scale(fused)
        shift  = self.shift(fused)
        return gen_feat + w * (gen_feat * scale + shift)
    """

    def __init__(self, in_concat_ch: int, out_ch: int):
        super().__init__()
        # Uses ResBlock with GroupNorm (same as encoder/generator ResnetBlock)
        self.encode_enc = _ResBlock(in_concat_ch, out_ch)
        self.scale = nn.Sequential(
            _GemmConv2d(out_ch, out_ch, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            _GemmConv2d(out_ch, out_ch, 3, 1, 1),
        )
        self.shift = nn.Sequential(
            _GemmConv2d(out_ch, out_ch, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            _GemmConv2d(out_ch, out_ch, 3, 1, 1),
        )

    def forward(
        self,
        enc_feat: torch.Tensor,
        gen_feat: torch.Tensor,
        w: float,
    ) -> torch.Tensor:
        fuse_in = torch.cat([enc_feat, gen_feat], dim=1)
        fused = self.encode_enc(fuse_in)
        scale = self.scale(fused)
        shift = self.shift(fused)

        if (
            _TRITON_AVAILABLE
            and _triton_sft is not None
            and gen_feat.dtype == torch.float16
        ):
            # triton_fused_gfpgan_act implements (gen_feat * scale + shift) + leaky_relu
            # but we need gen_feat + w * (gen_feat * scale + shift).
            # We can still use it by passing noise=shift, bias=None, scale=scale.
            # However, triton_fused_gfpgan_act has a hardcoded leaky_relu.
            # Let's stick to pure PyTorch for SFT as it's not a simple activation.
            return gen_feat + w * (gen_feat * scale + shift)

        return gen_feat + w * (gen_feat * scale + shift)


# ---------------------------------------------------------------------------
# AdaIN helper
# ---------------------------------------------------------------------------


def _adain(soft: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """
    Normalise soft VQ features to match encoder z channel statistics.

    soft: (b, c, h, w)  — softmax-averaged codebook features
    z:    (b, c, h, w)  — encoder output latent

    Returns: soft features re-scaled to z's per-channel mean/std.
    """
    b, c, h, w = soft.shape
    # Channel-wise stats over H×W
    soft_flat = soft.reshape(b, c, -1).float()
    z_flat = z.reshape(b, c, -1).float()

    soft_mean = soft_flat.mean(dim=2, keepdim=True)
    soft_std = soft_flat.std(dim=2, keepdim=True).clamp(min=1e-5)

    z_mean = z_flat.mean(dim=2, keepdim=True)
    z_std = z_flat.std(dim=2, keepdim=True).clamp(min=1e-5)

    normalised = (soft_flat - soft_mean) / soft_std * z_std + z_mean
    return normalised.reshape(b, c, h, w).to(soft.dtype)


# ===========================================================================
# Transformer layer matching original CodeFormer's TransformerSALayer
# ===========================================================================


class _TransformerSALayer(nn.Module):
    """Pre-norm self-attention layer with positional embedding on Q and K.

    Matches the original CodeFormer TransformerSALayer exactly:
      - norm1 applied first (pre-norm)
      - query_pos added to Q and K (but NOT V) before attention
      - Pre-norm FFN with residuals
    """

    def __init__(self, embed_dim: int, nhead: int, dim_mlp: int) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim, nhead, dropout=0.0, batch_first=False
        )
        self.linear1 = nn.Linear(embed_dim, dim_mlp)
        self.linear2 = nn.Linear(dim_mlp, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.activation = nn.GELU()

    def forward(self, tgt: torch.Tensor, query_pos: torch.Tensor) -> torch.Tensor:
        # Pre-norm self-attention: pos_emb added to Q and K only
        tgt2 = self.norm1(tgt)
        q = k = tgt2 + query_pos  # pos_emb on Q and K
        tgt2, _ = self.self_attn(query=q, key=k, value=tgt2)
        tgt = tgt + tgt2  # residual to original tokens

        # Pre-norm FFN
        tgt2 = self.norm2(tgt)
        tgt2 = self.linear2(self.activation(self.linear1(tgt2)))
        tgt = tgt + tgt2
        return tgt


# ===========================================================================
# CodeFormer
# ===========================================================================


class CodeFormerTorch(nn.Module):
    """
    FP16 PyTorch reimplementation of CodeFormer (codeformer_fp16.onnx).

    Input : (1, 3, 512, 512) float32
    Output: (1, 3, 512, 512) float32

    fidelity_weight w:
        0.0 → pure VQ generation (no encoder skip corrections)
        1.0 → full encoder-skip SFT conditioning
    """

    def __init__(self):
        super().__init__()

        # ── Encoder (25 blocks, indexed to match ONNX names) ─────────────
        # Wrapped in _BlockList so PyTorch names them encoder.blocks.{i}.*
        self.encoder = _BlockList(
            [
                nn.Conv2d(3, 64, 3, 1, 1),  # 0
                _ResBlock(64, 64),  # 1
                _ResBlock(64, 64),  # 2
                _Downsample(64),  # 3
                _ResBlock(64, 128),  # 4
                _ResBlock(128, 128),  # 5  ← skip for fuse_256
                _Downsample(128),  # 6
                _ResBlock(128, 128),  # 7
                _ResBlock(128, 128),  # 8  ← skip for fuse_128
                _Downsample(128),  # 9
                _ResBlock(128, 256),  # 10
                _ResBlock(256, 256),  # 11 ← skip for fuse_64
                _Downsample(256),  # 12
                _ResBlock(256, 256),  # 13
                _ResBlock(256, 256),  # 14 ← skip for fuse_32
                _Downsample(256),  # 15
                _ResBlock(256, 512),  # 16
                _AttnBlock(512),  # 17
                _ResBlock(512, 512),  # 18
                _AttnBlock(512),  # 19
                _ResBlock(512, 512),  # 20
                _AttnBlock(512),  # 21
                _ResBlock(512, 512),  # 22
                _GN(
                    512, fuse_silu=False
                ),  # 23 standalone — no SiLU before quant_conv (matches ONNX)
                nn.Conv2d(512, 256, 3, 1, 1),  # 24 quant_conv
            ]
        )

        # Encoder skip indices (output after these blocks)
        self._enc_skip_idx = {
            "fuse_32": 14,
            "fuse_64": 11,
            "fuse_128": 8,
            "fuse_256": 5,
        }

        # ── VQ codebook ──────────────────────────────────────────────────
        self.quantize = _VQLayer(1024, 256)

        # ── Transformer code refinement ───────────────────────────────────
        # Operates on sequence of 256 code positions × 512 dims
        # seq_len first: (seq=256, batch, d_model=512)
        self.feat_emb = nn.Linear(256, 512, bias=True)
        self.ft_layers = nn.ModuleList(
            [
                _TransformerSALayer(embed_dim=512, nhead=8, dim_mlp=1024)
                for _ in range(9)
            ]
        )
        # Positional embedding (256, 1, 512) — loaded from ONNX /Expand_output_0
        # Added to Q and K (not V) inside each transformer layer
        self.register_buffer(
            "pos_emb",
            torch.zeros(256, 1, 512),
        )
        self.idx_pred_layer = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, 1024, bias=False),
        )

        # ── Generator (25 blocks) ─────────────────────────────────────────
        self.generator = _BlockList(
            [
                nn.Conv2d(256, 512, 3, 1, 1),  # 0
                _ResBlock(512, 512),  # 1
                _AttnBlock(512),  # 2
                _ResBlock(512, 512),  # 3
                _ResBlock(512, 512),  # 4
                _AttnBlock(512),  # 5
                _ResBlock(512, 512),  # 6
                _AttnBlock(512),  # 7
                _Upsample(512),  # 8
                _ResBlock(512, 256),  # 9  ← fuse_32 applied after
                _ResBlock(256, 256),  # 10
                _Upsample(256),  # 11
                _ResBlock(256, 256),  # 12 ← fuse_64 applied after
                _ResBlock(256, 256),  # 13
                _Upsample(256),  # 14
                _ResBlock(256, 128),  # 15 ← fuse_128 applied after
                _ResBlock(128, 128),  # 16
                _Upsample(128),  # 17
                _ResBlock(128, 128),  # 18 ← fuse_256 applied after
                _ResBlock(128, 128),  # 19
                _Upsample(128),  # 20
                _ResBlock(128, 64),  # 21
                _ResBlock(64, 64),  # 22
                _GN(64, fuse_silu=False),  # 23 standalone (no SiLU before output conv)
                nn.Conv2d(64, 3, 3, 1, 1),  # 24 output conv
            ]
        )

        # SFT fuse blocks (applied after generator blocks 9, 12, 15, 18)
        # Key = generator block index after which fuse is applied
        # Naming matches ONNX: fuse_convs_dict.{scale}.*
        self.fuse_convs_dict = nn.ModuleDict(
            {
                "32": _FuseBlock(512, 256),  # enc_skip(256) + gen_h(256) = 512 in
                "64": _FuseBlock(512, 256),
                "128": _FuseBlock(256, 128),  # enc_skip(128) + gen_h(128) = 256 in
                "256": _FuseBlock(256, 128),
            }
        )

        # gen block index → fuse key
        self._gen_fuse_at = {9: "32", 12: "64", 15: "128", 18: "256"}
        # fuse key → encoder skip key
        self._fuse_to_enc_skip = {
            "32": "fuse_32",
            "64": "fuse_64",
            "128": "fuse_128",
            "256": "fuse_256",
        }

        self._use_cl: bool = False

    # -----------------------------------------------------------------------
    # Optimization helpers
    # -----------------------------------------------------------------------

    def to_channels_last(self) -> "CodeFormerTorch":
        """Convert all Conv2d weights to channels-last (NHWC) format.

        Enables the cuDNN implicit-GEMM NHWC path for large-spatial convolutions
        that fall back from GEMM mode (128×128 / 256×256 encoder/generator blocks).
        The Triton GroupNorm kernel and _AttnBlock are both channels-last safe.
        """
        self._use_cl = True
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.to(memory_format=torch.channels_last)
        return self

    def to_gemm_mode(self) -> "CodeFormerTorch":
        """Enable im2col+cuBLAS GEMM for all _GemmConv2d layers.

        Applies to ResBlock conv1/conv2, FuseBlock scale/shift convs, and
        Upsample convs.  Stride-2 downsamples and large-spatial blocks
        (HW > 4096) fall back automatically to cuDNN at runtime.
        Must be called AFTER to_channels_last() if both are used, because
        enable_gemm_mode() calls .contiguous() to ensure NCHW weight order.
        """
        for m in self.modules():
            if isinstance(m, _GemmConv2d):
                m.enable_gemm_mode()
        return self

    # -----------------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        fidelity_weight: float = 0.5,
    ) -> torch.Tensor:
        """
        x: (1, 3, 512, 512) float32
        Returns: (1, 3, 512, 512) float32
        """
        dtype = next(self.parameters()).dtype
        x_in = x.to(dtype)
        if self._use_cl:
            x_in = x_in.contiguous(memory_format=torch.channels_last)

        # ── Encoder ──────────────────────────────────────────────────────
        h = self.encoder[0](x_in)
        enc_skips: Dict[str, torch.Tensor] = {}
        for i in range(1, 23):
            h = self.encoder[i](h)
            if i == self._enc_skip_idx["fuse_256"]:
                enc_skips["fuse_256"] = h
            if i == self._enc_skip_idx["fuse_128"]:
                enc_skips["fuse_128"] = h
            if i == self._enc_skip_idx["fuse_64"]:
                enc_skips["fuse_64"] = h
            if i == self._enc_skip_idx["fuse_32"]:
                enc_skips["fuse_32"] = h
        # blocks 23 (standalone GN+SiLU) and 24 (quant_conv)
        h = self.encoder[23](h)
        z = self.encoder[24](h)  # (b, 256, 16, 16)

        # ── VQ lookup ────────────────────────────────────────────────────
        _zq = self.quantize(z)  # nearest-neighbour, (b, 256, 16, 16)

        # ── Transformer code refinement ───────────────────────────────────
        b, c, hs, ws = z.shape  # b=1, c=256, hs=16, ws=16
        seq_len = hs * ws  # 256

        # Project encoder features to transformer dimension (seq, batch, dim)
        z_seq = z.reshape(b, c, seq_len).permute(2, 0, 1)  # (256, b, 256)
        tokens = self.feat_emb(z_seq)  # (256, b, 512) in dtype

        # Expand pos_emb for current batch size (256, 1, 512) → (256, b, 512)
        query_pos = self.pos_emb.expand(-1, b, -1).to(tokens.dtype)
        for layer in self.ft_layers:
            tokens = layer(tokens, query_pos)

        logits = self.idx_pred_layer(tokens)  # (256, b, 1024) in dtype

        # Hard VQ: ONNX uses TopK k=1 → one-hot ScatterElements → MatMul
        # which is equivalent to argmax codebook lookup (not soft weighted sum)
        best_idx = logits.float().argmax(dim=-1)  # (256, b)
        codebook = self.quantize.embedding.weight  # (1024, 256)
        soft_feat = codebook[best_idx]  # (256, b, 256)
        soft_feat = soft_feat.permute(1, 2, 0).reshape(b, c, hs, ws)

        # AdaIN: normalise soft features to match encoder z statistics
        z_gen = _adain(soft_feat, z)  # (b, 256, 16, 16)

        # ── Generator ────────────────────────────────────────────────────
        h = self.generator[0](z_gen)
        for i in range(1, 25):
            h = self.generator[i](h)
            if i in self._gen_fuse_at:
                key = self._gen_fuse_at[i]
                enc_key = self._fuse_to_enc_skip[key]
                enc_skip = enc_skips[enc_key]
                h = self.fuse_convs_dict[key](enc_skip, h, fidelity_weight)

        return h.float()

    # -----------------------------------------------------------------------
    # from_onnx factory
    # -----------------------------------------------------------------------

    @classmethod
    def from_onnx(
        cls,
        onnx_path: str,
        compute_dtype: torch.dtype = torch.float16,
        verbose: bool = False,
    ) -> "CodeFormerTorch":
        import onnx
        from onnx import numpy_helper

        onnx_model = onnx.load(onnx_path)
        init_map = {
            init.name: torch.from_numpy(numpy_helper.to_array(init).copy())
            for init in onnx_model.graph.initializer
        }

        model = cls()

        # 1. Standard named parameters (conv weights/biases, norms, etc.)
        _load_named_onnx_params(model, init_map, verbose=verbose)

        # 2. GroupNorm scale/bias (anonymous Mul_XXX / Add_XXX in ONNX)
        _load_gn_params_from_graph(model, onnx_model, init_map, verbose=verbose)

        # 3. Transformer anonymous weights (Q/K/V projections, FFN weights)
        _load_transformer_anon_weights(model, init_map, verbose=verbose)

        # 4. Positional embedding (256, 1, 512) — ONNX initializer /Expand_output_0
        pos_emb_tensor = init_map.get("/Expand_output_0")
        if pos_emb_tensor is not None:
            with torch.no_grad():
                model.pos_emb.copy_(pos_emb_tensor.float())
            if verbose:
                print("[CodeFormerTorch] pos_emb loaded from ONNX /Expand_output_0")
        elif verbose:
            print("[CodeFormerTorch] WARNING: pos_emb not found in ONNX")

        model._visomaster_onnx_path = str(onnx_path)
        return model.to(compute_dtype)


# ===========================================================================
# Weight loading helpers
# ===========================================================================


def _load_named_onnx_params(
    model: "CodeFormerTorch",
    init_map: Dict[str, torch.Tensor],
    verbose: bool = False,
) -> None:
    """
    Load initializers whose names match PyTorch parameter paths exactly.
    Also handles: feat_emb.weight (transposed MatMul_3608),
                  idx_pred_layer.1.weight (transposed MatMul_3834).
    """
    sd = dict(model.named_parameters())

    # Standard prefix-match loading
    mapped: Dict[str, torch.Tensor] = {}
    for pt_k, pt_v in sd.items():
        if pt_k in init_map and init_map[pt_k].shape == pt_v.shape:
            mapped[pt_k] = init_map[pt_k]

    # feat_emb.weight: ONNX stores as (256, 512) but Linear.weight is (512, 256)
    feat_emb_mm = init_map.get("onnx::MatMul_3608")
    if feat_emb_mm is not None:
        mapped["feat_emb.weight"] = feat_emb_mm.T  # (256,512) → (512,256)

    # idx_pred_layer.1.weight: Linear(512→1024), ONNX (512,1024) → (1024,512)
    pred_mm = init_map.get("onnx::MatMul_3834")
    if pred_mm is not None:
        mapped["idx_pred_layer.1.weight"] = pred_mm.T  # (512,1024) → (1024,512)

    missing, unexpected = model.load_state_dict(mapped, strict=False)
    loaded = len(sd) - len(missing)
    if verbose:
        print(
            f"[CodeFormerTorch] named params: {loaded}/{len(sd)} loaded, "
            f"{len(missing)} missing"
        )
        if missing:
            print(f"  Missing (first 10): {missing[:10]}")


def _load_gn_params_from_graph(
    model: "CodeFormerTorch",
    onnx_model,
    init_map: Dict[str, torch.Tensor],
    verbose: bool = False,
) -> None:
    """
    Extract GroupNorm scale/bias from anonymous ONNX Mul/Add initializers.

    In the ONNX export, GroupNorm(32) is lowered to:
      Reshape → InstanceNorm(identity affine) → Reshape → Mul(scale) → Add(bias)
    The Mul/Add initializers are named onnx::Mul_XXXX / onnx::Add_XXXX.

    We collect them in graph-node order (= forward execution order) and assign
    to model GroupNorm modules in the same order.
    """
    # Map each tensor name → list of nodes that CONSUME it (forward direction)
    input_to_nodes: Dict[str, List] = {}
    for node in onnx_model.graph.node:
        for inp in node.input:
            input_to_nodes.setdefault(inp, []).append(node)

    def _first_consumer(tensor_name: str, op_type: str):
        return next(
            (n for n in input_to_nodes.get(tensor_name, []) if n.op_type == op_type),
            None,
        )

    # Collect (scale, bias) pairs in graph execution order
    # Pattern: InstanceNorm.output[0] → Reshape → Mul(scale_init) → Add(bias_init)
    gn_pairs: List[Tuple[torch.Tensor, torch.Tensor]] = []

    for node in onnx_model.graph.node:
        if node.op_type != "InstanceNormalization":
            continue
        in_out = node.output[0]
        reshape_node = _first_consumer(in_out, "Reshape")
        if reshape_node is None:
            continue
        reshape_out = reshape_node.output[0]
        mul_node = _first_consumer(reshape_out, "Mul")
        if mul_node is None:
            continue
        scale_name = next(
            (inp for inp in mul_node.input if inp != reshape_out and inp in init_map),
            None,
        )
        if scale_name is None:
            continue
        mul_out = mul_node.output[0]
        add_node = _first_consumer(mul_out, "Add")
        if add_node is None:
            continue
        bias_name = next(
            (inp for inp in add_node.input if inp != mul_out and inp in init_map),
            None,
        )
        if bias_name is None:
            continue

        scale = init_map[scale_name].reshape(-1)
        bias = init_map[bias_name].reshape(-1)
        gn_pairs.append((scale, bias))

    # Collect GroupNorm modules in FORWARD execution order
    gn_mods = _gn_modules_in_forward_order(model)

    if len(gn_pairs) != len(gn_mods):
        print(
            f"[CodeFormerTorch] WARNING: {len(gn_pairs)} GN pairs in ONNX, "
            f"{len(gn_mods)} GN modules in model — skipping GN load."
        )
        return

    with torch.no_grad():
        for gm, (scale, bias) in zip(gn_mods, gn_pairs):
            gm.weight.copy_(scale)
            gm.bias.copy_(bias)

    if verbose:
        print(f"[CodeFormerTorch] loaded {len(gn_pairs)} GroupNorm scale/bias pairs")


def _gn_modules_in_forward_order(model: "CodeFormerTorch") -> List[nn.GroupNorm]:
    """
    Return GroupNorm modules in the order they are executed in forward().

    This matches the ONNX graph's InstanceNorm execution order:
      1. All encoder GN modules
      2. Generator GN interleaved with fuse encode_enc GN (at blocks 9,12,15,18)
    """
    mods: List[nn.GroupNorm] = []

    def _gn_from_block(blk: nn.Module) -> List[nn.GroupNorm]:
        if isinstance(blk, _ResBlock):
            return [blk.norm1, blk.norm2]
        if isinstance(blk, _AttnBlock):
            return [blk.norm]
        if isinstance(blk, _GN):
            return [blk]
        return []

    # Encoder blocks 1–23
    for i in range(1, 24):
        mods.extend(_gn_from_block(model.encoder.blocks[i]))

    # Generator blocks 1–23, with fuse encode_enc interleaved after 9/12/15/18
    gen_fuse_at = {9: "32", 12: "64", 15: "128", 18: "256"}
    for i in range(1, 24):
        mods.extend(_gn_from_block(model.generator.blocks[i]))
        if i in gen_fuse_at:
            key = gen_fuse_at[i]
            enc_enc = model.fuse_convs_dict[key].encode_enc
            mods.extend([enc_enc.norm1, enc_enc.norm2])

    return mods


def _load_transformer_anon_weights(
    model: "CodeFormerTorch",
    init_map: Dict[str, torch.Tensor],
    verbose: bool = False,
) -> None:
    """
    Load anonymous Q/K/V projection weights and FFN weights for the 9 transformer layers.

    Naming pattern (layer N, 0-indexed, stride 25):
      Q bias:    onnx::Add_{3624 + N*25}        (512,)
      K bias:    onnx::Add_{3626 + N*25}         (512,)
      V bias:    onnx::Add_{3628 + N*25}         (512,)
      Q weight:  onnx::MatMul_{3629 + N*25}    (512, 512) — ONNX row-major, needs .T
      K weight:  onnx::MatMul_{3630 + N*25}    (512, 512)
      V weight:  onnx::MatMul_{3631 + N*25}    (512, 512)
      FFN1 w:    onnx::MatMul_{3632 + N*25}    (512, 1024) → Linear(512,1024).weight (1024,512)
      FFN2 w:    onnx::MatMul_{3633 + N*25}    (1024, 512) → Linear(1024,512).weight (512,1024)
    """
    loaded_layers = 0
    for n, layer in enumerate(model.ft_layers):
        base_add = 3624 + n * 25
        base_mm = 3629 + n * 25

        q_b = init_map.get(f"onnx::Add_{base_add}")
        k_b = init_map.get(f"onnx::Add_{base_add + 2}")
        v_b = init_map.get(f"onnx::Add_{base_add + 4}")
        q_w = init_map.get(f"onnx::MatMul_{base_mm}")
        k_w = init_map.get(f"onnx::MatMul_{base_mm + 1}")
        v_w = init_map.get(f"onnx::MatMul_{base_mm + 2}")
        ff1w = init_map.get(f"onnx::MatMul_{base_mm + 3}")
        ff2w = init_map.get(f"onnx::MatMul_{base_mm + 4}")

        if any(t is None for t in [q_b, k_b, v_b, q_w, k_w, v_w, ff1w, ff2w]):
            if verbose:
                print(
                    f"[CodeFormerTorch] transformer layer {n}: some weights missing, skipped"
                )
            continue

        with torch.no_grad():
            # MultiheadAttention stores combined Q/K/V as in_proj_weight (1536, 512)
            # and in_proj_bias (1536,). ONNX MatMul weight (d_in, d_out) → transpose.
            in_proj_w = torch.cat([q_w.T, k_w.T, v_w.T], dim=0)  # type: ignore[union-attr]  # (1536, 512)
            in_proj_b = torch.cat([q_b, k_b, v_b], dim=0)  # (1536,)
            layer.self_attn.in_proj_weight.copy_(in_proj_w)
            layer.self_attn.in_proj_bias.copy_(in_proj_b)

            # FFN Linear: PyTorch stores (out, in), ONNX stores (in, out) → transpose
            layer.linear1.weight.copy_(ff1w.T)  # type: ignore[union-attr]  # (1024, 512)
            layer.linear2.weight.copy_(ff2w.T)  # type: ignore[union-attr]  # (512, 1024)

        loaded_layers += 1

    if verbose:
        print(
            f"[CodeFormerTorch] loaded transformer QKV+FFN for {loaded_layers}/9 layers"
        )


# ===========================================================================
# Optimized CUDA-graph runner (with torch.compile fusion)
# ===========================================================================


class CUDAGraphRunner:
    """Wraps a module in a CUDA graph for zero-overhead repeated inference."""

    def __init__(
        self,
        module: nn.Module,
        inp_shape: Tuple[int, ...],
        fidelity_weight: float = 0.5,
    ):
        self._module = module
        self._inp_shape = inp_shape
        self._fidelity_weight = fidelity_weight
        self._graph = None
        self._static_in: Optional[torch.Tensor] = None
        self._static_out: Optional[torch.Tensor] = None

    def _capture(self) -> None:
        device = next(self._module.parameters()).device
        self._static_in = torch.zeros(
            self._inp_shape, dtype=torch.float32, device=device
        )

        # Warm up before capture: initialises cuDNN algorithm caches and
        # triggers Triton JIT compilation.  These side-effects must complete
        # *before* the CUDA graph capture window opens, because the capture
        # window cannot contain CPU-GPU synchronisation or dynamic allocations.
        print("[CodeFormerTorch] Warming up (cuDNN/Triton JIT init)...")
        with torch.no_grad():
            for _ in range(3):
                _ = self._module(self._static_in, self._fidelity_weight)
        torch.cuda.synchronize(device)

        # Capture: all GPU kernel launches inside this context are recorded
        # into a CUDA graph and replayed verbatim on every subsequent call —
        # zero kernel-launch overhead.
        # NOTE: do NOT wrap a torch.compile(reduce-overhead) model here;
        # that mode already captures CUDA graphs internally and the two
        # contexts would conflict (double-capture = crash).
        print("[CodeFormerTorch] Capturing CUDA graph...")
        self._capture_stream = torch.cuda.Stream(device=device)
        self._graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize(device)
        with torch.no_grad():
            with torch.cuda.graph(
                self._graph,
                stream=self._capture_stream,
                capture_error_mode="relaxed",
            ):
                self._static_out = self._module(self._static_in, self._fidelity_weight)
        torch.cuda.synchronize(device)
        print("[CodeFormerTorch] CUDA graph captured.")

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self._graph is None:
            self._capture()
        self._static_in.copy_(x)  # type: ignore[union-attr]
        self._graph.replay()  # type: ignore[union-attr, attr-defined]
        return self._static_out.clone()  # type: ignore[union-attr]


def build_cuda_graph_runner(
    model: "CodeFormerTorch",
    inp_shape: Tuple[int, ...] = (1, 3, 512, 512),
    fidelity_weight: float = 0.5,
    torch_compile: bool = False,
) -> CUDAGraphRunner:
    """
    Capture model into a CUDA graph.  Call the returned runner like a function:

        runner = build_cuda_graph_runner(model, fidelity_weight=0.7)
        output = runner(face_image_f32_cuda)

    Args:
        torch_compile: If True, wrap the model with ``torch.compile`` before
                       capturing the CUDA graph.  Requires Triton; adds ~60 s
                       one-time compile overhead for this complex transformer but
                       gives ~1.17x speedup over the uncompiled CUDA-graph baseline.
    """
    if torch_compile:
        try:
            from custom_kernels.compile_utils import apply_torch_compile
            device = next(model.parameters()).device
            example_inp = torch.zeros(inp_shape, dtype=torch.float32, device=device)
            compiled = apply_torch_compile(
                model, example_inp,
                extra_kwargs={"fidelity_weight": fidelity_weight},
            )
            print("[codeformer] torch.compile warmup done.")
            # Return compiled model directly — CUDA graph on top of torch.compile
            # fails on Windows (64-bit kernel handles overflow 32-bit C long).
            # Wrap in a callable that matches the CUDAGraphRunner (x,) → tensor interface,
            # baking in fidelity_weight so callers don't need to pass it.
            _fw = fidelity_weight
            _compiled = compiled

            class _CompiledCodeFormerRunner:
                def __call__(self, x: "torch.Tensor") -> "torch.Tensor":
                    return _compiled(x, fidelity_weight=_fw)

            return _CompiledCodeFormerRunner()
        except Exception as e:
            print(f"[codeformer] torch.compile failed ({e!s:.120}), falling back to CUDA graph.")

    return CUDAGraphRunner(model, inp_shape, fidelity_weight)
