"""
RefLDMTorch — FP16 PyTorch implementation of the ReF-LDM denoiser models.

Three models:
  RefLDMEncoderTorch  — VAE encoder   (1,3,512,512)f32 → (1,8,64,64)f32
  RefLDMDecoderTorch  — VAE decoder   (1,8,64,64)f32  → (1,3,512,512)f32
  RefLDMUNetTorch     — Denoising UNet (1,16,64,64)f32 + K/V → (1,8,64,64)f32

Architecture from configs/refldm.yaml:
  VAE  : ch=128, ch_mult=[1,1,2,4], num_res_blocks=2, attn_resolutions=[]
  UNet : model_channels=160, channel_mult=[1,2,2,4], num_res_blocks=2,
         attention_resolutions=[2,4,8], num_head_channels=32,
         in_channels=16, out_channels=8

Speedup tiers vs ORT FP32 CUDA EP (RTX 4090):
  Tier 1 — PyTorch FP32                              ~1.0x
  Tier 2 — PyTorch FP16 + Triton GroupNorm+SiLU      ~2.0x (VAE), ~0.7x (UNet eager)
  Tier 3 — FP16 + Triton + CUDA graph                ~2.0x (enc), ~2.9x (dec), ~1.8x (unet)
  Tier 4 — FP16 + Triton + CUDA graph + NHWC         ~2.9x (dec, best), ~1.8x (unet)

Optimizations applied:
  - norm_out GroupNorm has fuse_silu=True — SiLU fused inside Triton kernel, eliminating
    a separate activation memory pass (most impactful for decoder's 512×512 output map)
  - UNet output GN also fuses SiLU (nn.Identity placeholder preserves out.2 weight index)
  - UNet supports CUDA graph via UNetCUDAGraphRunner with static K/V buffers

Usage:
    enc  = RefLDMEncoderTorch.from_onnx("model_assets/ref_ldm_vae_encoder.onnx").cuda().eval()
    dec  = RefLDMDecoderTorch.from_onnx("model_assets/ref_ldm_vae_decoder.onnx").cuda().eval()
    unet = RefLDMUNetTorch.from_onnx("model_assets/ref_ldm_unet_external_kv.onnx").cuda().eval()

    enc_runner  = build_cuda_graph_runner(enc,  inp_shape=(1, 3, 512, 512))
    dec_runner  = build_cuda_graph_runner(dec,  inp_shape=(1, 8, 64,  64 ))
    latent      = enc_runner(image_f32_cuda)          # → (1,8,64,64) f32
    image_out   = dec_runner(latent)                   # → (1,3,512,512) f32
    noise_pred  = unet(x_noisy_lq, timesteps, kv_map, use_exclusive_path=True)
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Triton GroupNorm+SiLU (preferred; no MSVC required)
# ---------------------------------------------------------------------------
try:
    from custom_kernels.triton_ops import (
        TRITON_AVAILABLE as _TRITON_AVAILABLE,
        triton_group_norm_silu as _triton_gn_silu,
    )
except Exception:
    _TRITON_AVAILABLE = False
    _triton_gn_silu = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Fused GroupNorm(32) module  — used everywhere in VAE and UNet
# ---------------------------------------------------------------------------
class _GN(nn.GroupNorm):
    """
    GroupNorm(32) with optional fused SiLU.

    When Triton is available and x is fp16, uses a single-pass Triton kernel
    that keeps FP32 accumulators internally — eliminates the GroupNorm32
    FP16→FP32→FP16 round-trip.

    When fuse_silu=True the activation is applied inside the same kernel
    (no extra memory read/write).

    Channels-last safe: if the input is in channels-last memory format the
    Triton kernel (which assumes NCHW contiguous layout) would read wrong
    memory.  We temporarily make the input NCHW-contiguous before invoking
    the kernel, then restore channels-last on output so that surrounding
    Conv2d ops can stay in the NHWC cuDNN path.
    """

    def __init__(self, num_channels: int, fuse_silu: bool = False):
        super().__init__(
            num_groups=32, num_channels=num_channels, eps=1e-6, affine=True
        )
        self.fuse_silu = fuse_silu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Triton GN kernel requires NCHW-contiguous layout.
        was_cl = x.is_contiguous(memory_format=torch.channels_last)
        x_in = x.contiguous() if was_cl else x  # NCHW for kernel

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
            # Fallback: standard GroupNorm with safe FP32 upcast
            out = F.group_norm(
                x_in.float(),
                self.num_groups,
                self.weight.float(),
                self.bias.float(),
                self.eps,
            ).to(x_in.dtype)
            if self.fuse_silu:
                out = out * torch.sigmoid(out)

        # Restore channels-last so surrounding Conv2d ops stay on the NHWC path.
        return out.contiguous(memory_format=torch.channels_last) if was_cl else out


# ===========================================================================
# VAE Building Blocks
# ===========================================================================


def _swish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class _VAEResBlock(nn.Module):
    """VAE residual block: GN+Swish → Conv → GN+Swish → Conv (+skip)."""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = _GN(in_ch, fuse_silu=True)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.norm2 = _GN(out_ch, fuse_silu=True)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, temb=None) -> torch.Tensor:
        h = self.conv1(self.norm1(x))
        h = self.conv2(self.drop(self.norm2(h)))
        return self.skip(x) + h


class _VAEAttnBlock(nn.Module):
    """Single-head spatial self-attention for VAE mid block."""

    def __init__(self, in_ch: int):
        super().__init__()
        self.norm = _GN(in_ch, fuse_silu=False)
        self.q = nn.Conv2d(in_ch, in_ch, 1)
        self.k = nn.Conv2d(in_ch, in_ch, 1)
        self.v = nn.Conv2d(in_ch, in_ch, 1)
        self.proj_out = nn.Conv2d(in_ch, in_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        was_cl = x.is_contiguous(memory_format=torch.channels_last)
        h = self.norm(x)
        b, c, ht, wt = h.shape

        # Conv2d outputs (b, c, ht, wt); must permute channels to last dim before SDPA
        # so that spatial positions are the "sequence" dimension, not channels.
        def _to_sdpa(t: torch.Tensor) -> torch.Tensor:
            return t.reshape(b, c, ht * wt).permute(0, 2, 1).unsqueeze(1)  # (b,1,hw,c)

        q = _to_sdpa(self.q(h))
        k = _to_sdpa(self.k(h))
        v = _to_sdpa(self.v(h))
        a = F.scaled_dot_product_attention(q, k, v)  # (b,1,hw,c)
        a = a.squeeze(1).permute(0, 2, 1).reshape(b, c, ht, wt)  # (b,c,ht,wt)
        out = x + self.proj_out(a)
        # Restore channels-last if the input was in that format — reshape/permute operations
        # above break the layout guarantee that cuDNN needs for surrounding conv ops.
        return out.contiguous(memory_format=torch.channels_last) if was_cl else out


class _VAEDownsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (0, 1, 0, 1)))


class _VAEUpsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.interpolate(x, scale_factor=2.0, mode="nearest"))


# ===========================================================================
# VAE Encoder
# ===========================================================================


class RefLDMEncoderTorch(nn.Module):
    """
    VAE encoder: image (1,3,512,512)f32 → latent (1,8,64,64)f32.
    Includes quant_conv (1×1) as exported in the ONNX model.
    Config: ch=128, ch_mult=[1,1,2,4], num_res_blocks=2, attn_resolutions=[],
            z_channels=8, double_z=False.
    """

    _CH = 128
    _CH_MULT = (1, 1, 2, 4)
    _NRB = 2
    _Z_CH = 8

    def __init__(self):
        super().__init__()
        ch, mult, nrb, z_ch = self._CH, self._CH_MULT, self._NRB, self._Z_CH
        num_res = len(mult)

        self.encoder = nn.Module()
        self.encoder.conv_in = nn.Conv2d(3, ch, 3, 1, 1)

        _in_ch_mult = (1,) + tuple(mult)
        self.encoder.down = nn.ModuleList()
        block_in = ch
        for i in range(num_res):
            block_out = ch * mult[i]
            blk = nn.ModuleList()
            for _ in range(nrb):
                blk.append(_VAEResBlock(block_in, block_out))
                block_in = block_out
            lvl = nn.Module()
            lvl.block = blk
            lvl.attn = nn.ModuleList()
            if i != num_res - 1:
                lvl.downsample = _VAEDownsample(block_in)
            self.encoder.down.append(lvl)

        mid = nn.Module()
        mid.block_1 = _VAEResBlock(block_in, block_in)
        mid.attn_1 = _VAEAttnBlock(block_in)
        mid.block_2 = _VAEResBlock(block_in, block_in)
        self.encoder.mid = mid

        self.encoder.norm_out = _GN(block_in, fuse_silu=True)
        self.encoder.conv_out = nn.Conv2d(block_in, z_ch, 3, 1, 1)

        self.quant_conv = nn.Conv2d(z_ch, z_ch, 1)
        self._use_cl: bool = False

    def to_channels_last(self) -> "RefLDMEncoderTorch":
        """Convert Conv2d weights to channels-last (NHWC) and flag input conversion.

        cuDNN automatically selects the faster NHWC implicit-GEMM path when both
        the Conv2d weight and the input tensor are in channels-last format.  The
        Triton GroupNorm kernel is NCHW-only so _GN.forward temporarily restores
        NCHW before calling it, then converts the output back to channels-last.
        Call *after* .cuda().eval() and weight loading.  Returns self.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data = m.weight.data.to(memory_format=torch.channels_last)
        self._use_cl = True
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accept fp32 input; run model at compute_dtype; return fp32
        x16 = x.to(self.encoder.conv_in.weight.dtype)
        if self._use_cl:
            x16 = x16.contiguous(memory_format=torch.channels_last)
        h = self.encoder.conv_in(x16)
        hs = [h]
        enc = self.encoder
        for i_lvl, lvl in enumerate(enc.down):
            for blk in lvl.block:
                h = blk(hs[-1])
                hs.append(h)
            if hasattr(lvl, "downsample"):
                hs.append(lvl.downsample(hs[-1]))
        h = hs[-1]
        h = enc.mid.block_1(h)
        h = enc.mid.attn_1(h)
        h = enc.mid.block_2(h)
        h = enc.norm_out(h)  # SiLU now fused into GN kernel
        h = enc.conv_out(h)
        return self.quant_conv(h).float()

    @classmethod
    def from_onnx(
        cls,
        onnx_path: str,
        compute_dtype: torch.dtype = torch.float16,
        verbose: bool = False,
    ) -> "RefLDMEncoderTorch":
        import onnx
        import re as _re
        from onnx import numpy_helper

        onnx_model = onnx.load(onnx_path)
        w = {
            init.name: torch.from_numpy(numpy_helper.to_array(init).copy())
            for init in onnx_model.graph.initializer
        }

        # 1. Rename nin_shortcut → skip (ONNX uses nin_shortcut, PT uses skip)
        for k in list(w.keys()):
            if "nin_shortcut" in k:
                w[k.replace("nin_shortcut", "skip")] = w.pop(k)

        # 2. Map anonymous onnx::Mul_XXX / onnx::Add_XXX tensors to GN param keys.
        #    These are stored positionally in forward-execution order.
        #    Shape (C,1,1): Mul → weight (gamma), Add → bias (beta).
        _enc_gn_keys: List[str] = []
        for _i in range(4):  # down.0 … down.3
            for _j in range(2):  # block.0, block.1
                _enc_gn_keys += [
                    f"encoder.down.{_i}.block.{_j}.norm1.weight",
                    f"encoder.down.{_i}.block.{_j}.norm1.bias",
                    f"encoder.down.{_i}.block.{_j}.norm2.weight",
                    f"encoder.down.{_i}.block.{_j}.norm2.bias",
                ]
        _enc_gn_keys += [
            "encoder.mid.block_1.norm1.weight",
            "encoder.mid.block_1.norm1.bias",
            "encoder.mid.block_1.norm2.weight",
            "encoder.mid.block_1.norm2.bias",
            "encoder.mid.attn_1.norm.weight",
            "encoder.mid.attn_1.norm.bias",
            "encoder.mid.block_2.norm1.weight",
            "encoder.mid.block_2.norm1.bias",
            "encoder.mid.block_2.norm2.weight",
            "encoder.mid.block_2.norm2.bias",
            "encoder.norm_out.weight",
            "encoder.norm_out.bias",
        ]
        _anon: List[Tuple[int, torch.Tensor]] = []
        for _name, _t in w.items():
            _m = _re.match(r"^onnx::(Mul|Add)_(\d+)$", _name)
            if _m and _t.ndim == 3 and _t.shape[1] == 1 and _t.shape[2] == 1:
                _anon.append((int(_m.group(2)), _t))
        _anon.sort(key=lambda x: x[0])
        if len(_anon) != len(_enc_gn_keys):
            print(
                f"[RefLDMEncoder] WARNING: {len(_anon)} anon GN tensors vs "
                f"{len(_enc_gn_keys)} expected keys — mapping truncated"
            )
        for (_idx, _t), _pt_k in zip(_anon, _enc_gn_keys):
            w[_pt_k] = _t.squeeze(-1).squeeze(-1)  # (C,1,1) → (C,)

        model = cls()
        _load_from_onnx_weights(
            model, w, prefixes=["", "encoder.", "first_stage_model."], verbose=verbose
        )
        model._visomaster_onnx_path = str(onnx_path)
        return model.to(compute_dtype)


# ===========================================================================
# VAE Decoder
# ===========================================================================


class RefLDMDecoderTorch(nn.Module):
    """
    VAE decoder: latent (1,8,64,64)f32 → image (1,3,512,512)f32.
    The exported ONNX decoder starts directly with post_quant_conv — VQ
    quantization is NOT included in the ONNX graph.
    Config: same VAE config as encoder.
    """

    _CH = 128
    _CH_MULT = (1, 1, 2, 4)
    _NRB = 2
    _Z_CH = 8

    def __init__(self):
        super().__init__()
        ch, mult, nrb, z_ch = self._CH, self._CH_MULT, self._NRB, self._Z_CH
        num_res = len(mult)

        self.post_quant_conv = nn.Conv2d(z_ch, z_ch, 1)

        # Decoder
        block_in = ch * mult[-1]
        dec = nn.Module()
        dec.conv_in = nn.Conv2d(z_ch, block_in, 3, 1, 1)

        mid = nn.Module()
        mid.block_1 = _VAEResBlock(block_in, block_in)
        mid.attn_1 = _VAEAttnBlock(block_in)
        mid.block_2 = _VAEResBlock(block_in, block_in)
        dec.mid = mid

        dec.up = nn.ModuleList()
        for i_lvl in reversed(range(num_res)):
            block_out = ch * mult[i_lvl]
            blk = nn.ModuleList()
            for _ in range(nrb + 1):
                blk.append(_VAEResBlock(block_in, block_out))
                block_in = block_out
            lvl = nn.Module()
            lvl.block = blk
            lvl.attn = nn.ModuleList()
            if i_lvl != 0:
                lvl.upsample = _VAEUpsample(block_in)
            dec.up.insert(0, lvl)

        dec.norm_out = _GN(block_in, fuse_silu=True)
        dec.conv_out = nn.Conv2d(block_in, 3, 3, 1, 1)
        self.decoder = dec
        self._use_cl: bool = False

    def to_channels_last(self) -> "RefLDMDecoderTorch":
        """Convert Conv2d weights to channels-last (NHWC) for faster cuDNN convolutions.
        Call *after* .cuda().eval() and weight loading.  Returns self."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data = m.weight.data.to(memory_format=torch.channels_last)
        self._use_cl = True
        return self

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # The exported ONNX decoder starts directly with post_quant_conv — VQ quantization
        # is NOT included in the ONNX model (the upstream pipeline provides pre-quantized
        # or raw latents directly).  Pass z through unchanged; just cast to compute dtype.
        z_q = z.to(self.post_quant_conv.weight.dtype)
        if self._use_cl:
            z_q = z_q.contiguous(memory_format=torch.channels_last)
        z_q = self.post_quant_conv(z_q)
        h = z_q
        dec = self.decoder
        h = dec.conv_in(h)
        h = dec.mid.block_1(h)
        h = dec.mid.attn_1(h)
        h = dec.mid.block_2(h)
        for i_lvl in reversed(range(len(dec.up))):
            for blk in dec.up[i_lvl].block:
                h = blk(h)
            if hasattr(dec.up[i_lvl], "upsample"):
                h = dec.up[i_lvl].upsample(h)
        h = dec.norm_out(h)  # SiLU now fused into GN kernel
        return dec.conv_out(h).float()

    @classmethod
    def from_onnx(
        cls,
        onnx_path: str,
        compute_dtype: torch.dtype = torch.float16,
        verbose: bool = False,
    ) -> "RefLDMDecoderTorch":
        import onnx
        import re as _re
        from onnx import numpy_helper

        onnx_model = onnx.load(onnx_path)
        w = {
            init.name: torch.from_numpy(numpy_helper.to_array(init).copy())
            for init in onnx_model.graph.initializer
        }

        # 1. Rename nin_shortcut → skip (ONNX uses nin_shortcut, PT uses skip)
        for k in list(w.keys()):
            if "nin_shortcut" in k:
                w[k.replace("nin_shortcut", "skip")] = w.pop(k)

        # 2. Map anonymous onnx::Mul_XXX / onnx::Add_XXX tensors to GN param keys.
        #    Forward order: mid → up[3] → up[2] → up[1] → up[0] → norm_out.
        _dec_gn_keys: List[str] = [
            "decoder.mid.block_1.norm1.weight",
            "decoder.mid.block_1.norm1.bias",
            "decoder.mid.block_1.norm2.weight",
            "decoder.mid.block_1.norm2.bias",
            "decoder.mid.attn_1.norm.weight",
            "decoder.mid.attn_1.norm.bias",
            "decoder.mid.block_2.norm1.weight",
            "decoder.mid.block_2.norm1.bias",
            "decoder.mid.block_2.norm2.weight",
            "decoder.mid.block_2.norm2.bias",
        ]
        for _i in [3, 2, 1, 0]:  # forward execution order (reversed levels)
            for _j in range(3):  # nrb + 1 = 3 blocks per level
                _dec_gn_keys += [
                    f"decoder.up.{_i}.block.{_j}.norm1.weight",
                    f"decoder.up.{_i}.block.{_j}.norm1.bias",
                    f"decoder.up.{_i}.block.{_j}.norm2.weight",
                    f"decoder.up.{_i}.block.{_j}.norm2.bias",
                ]
        _dec_gn_keys += ["decoder.norm_out.weight", "decoder.norm_out.bias"]

        _anon: List[Tuple[int, torch.Tensor]] = []
        for _name, _t in w.items():
            _m = _re.match(r"^onnx::(Mul|Add)_(\d+)$", _name)
            if _m and _t.ndim == 3 and _t.shape[1] == 1 and _t.shape[2] == 1:
                _anon.append((int(_m.group(2)), _t))
        _anon.sort(key=lambda x: x[0])
        if len(_anon) != len(_dec_gn_keys):
            print(
                f"[RefLDMDecoder] WARNING: {len(_anon)} anon GN tensors vs "
                f"{len(_dec_gn_keys)} expected keys — mapping truncated"
            )
        for (_idx, _t), _pt_k in zip(_anon, _dec_gn_keys):
            w[_pt_k] = _t.squeeze(-1).squeeze(-1)  # (C,1,1) → (C,)

        model = cls()
        _load_from_onnx_weights(
            model,
            w,
            prefixes=[
                "",
                "decoder.",
                "first_stage_model.",
                "first_stage_model.decoder.",
            ],
            verbose=verbose,
        )
        model._visomaster_onnx_path = str(onnx_path)
        return model.to(compute_dtype)


# ===========================================================================
# UNet Building Blocks
# ===========================================================================


def _zero_module(module: nn.Module) -> nn.Module:
    for p in module.parameters():
        p.detach().zero_()
    return module


def _timestep_embedding(
    timesteps: torch.Tensor, dim: int, max_period: int = 10000
) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(half, dtype=torch.float32, device=timesteps.device)
        / half
    )
    args = timesteps[:, None].float() * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class _UNetResBlock(nn.Module):
    """UNet residual block — attribute names match the ONNX initialiser keys exactly.

    in_layers  = [GroupNorm32+SiLU (index 0), SiLU skipped in Sequential, Conv2d (index 2)]
    emb_layers = [SiLU (index 0), Linear (index 1)]
    out_layers  = [GroupNorm32+SiLU (index 0), SiLU skipped, Dropout (index 2), Conv2d (index 3)]
    skip_connection = Conv2d (1×1) when channels change, else Identity
    """

    def __init__(
        self, ch: int, emb_ch: int, dropout: float, out_ch: Optional[int] = None
    ):
        super().__init__()
        out_ch = out_ch or ch

        # in_layers: indices 0 = GN+SiLU, 1 = SiLU (placeholder), 2 = Conv2d
        self.in_layers = nn.Sequential(
            _GN(ch, fuse_silu=True),
            nn.Identity(),  # index 1 placeholder (SiLU fused into GN)
            nn.Conv2d(ch, out_ch, 3, 1, 1),  # index 2 — "in_layers.2" in ONNX
        )
        # emb_layers: indices 0 = SiLU, 1 = Linear
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_ch, out_ch),  # index 1 — "emb_layers.1" in ONNX
        )
        # out_layers: indices 0 = GN+SiLU, 1 = SiLU placeholder, 2 = Dropout, 3 = Conv2d
        self.out_layers = nn.Sequential(
            _GN(out_ch, fuse_silu=True),
            nn.Identity(),  # index 1 placeholder
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            _zero_module(
                nn.Conv2d(out_ch, out_ch, 3, 1, 1)
            ),  # index 3 — "out_layers.3"
        )
        # skip_connection — "skip_connection" in ONNX (or absent when ch == out_ch)
        self.skip_connection: nn.Module = (
            nn.Conv2d(ch, out_ch, 1) if ch != out_ch else nn.Identity()
        )

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.in_layers(x)
        emb_o = self.emb_layers(emb).to(h.dtype)
        h = h + emb_o[:, :, None, None]
        h = self.out_layers(h)
        return self.skip_connection(x) + h


class _UNetDownsample(nn.Module):
    """Strided-Conv downsampler — matches ONNX 'input_blocks.N.0.op' weight."""

    def __init__(self, ch: int):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class _UNetUpsample(nn.Module):
    """Nearest-neighbour upsample + Conv — matches ONNX 'output_blocks.N.M.conv' weight."""

    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.interpolate(x, scale_factor=2.0, mode="nearest"))


class _UNetAttnBlock(nn.Module):
    """
    Multi-head self-attention for the UNet.
    During denoising supports external reference K/V (concat or exclusive).
    """

    def __init__(self, ch: int, num_head_channels: int = 32):
        super().__init__()
        self.n_heads = ch // num_head_channels
        self.norm = _GN(ch, fuse_silu=False)
        self.qkv = nn.Conv1d(ch, ch * 3, 1)
        self.proj_out = _zero_module(nn.Conv1d(ch, ch, 1))

    def forward(
        self,
        x: torch.Tensor,
        ext_k: Optional[torch.Tensor] = None,
        ext_v: Optional[torch.Tensor] = None,
        use_exclusive: bool = False,
    ) -> torch.Tensor:
        # Remember channels-last — reshape/permute below break the layout guarantee.
        was_cl = x.is_contiguous(memory_format=torch.channels_last)

        b, c, *spatial = x.shape
        hw = x.numel() // (b * c)
        r = x.reshape(b, c, hw)
        qkv = self.qkv(self.norm(r))  # (b, 3*c, hw)
        nh = self.n_heads
        ch = c // nh

        # Reshape to 4D (b, nh, ch, hw) then permute to (b, nh, hw, ch) for SDPA.
        # 4D input is required by PyTorch's fused Flash Attention / Memory-Efficient
        # Attention kernels; 3D input (b*nh, hw, ch) falls back to the slow "math"
        # implementation which is ~3× slower on Ampere+.
        q, k, v = qkv.reshape(b, nh, ch * 3, hw).split(
            ch, dim=2
        )  # each (b, nh, ch, hw)
        qt = q.permute(0, 1, 3, 2)  # (b, nh, hw, ch)  — Q
        kt = k.permute(0, 1, 3, 2)  # (b, nh, hw, ch)  — self K
        vt = v.permute(0, 1, 3, 2)  # (b, nh, hw, ch)  — self V

        # External K/V routing — always concatenate self + external K/V.
        # The ONNX model always concatenates regardless of use_exclusive;
        # use_exclusive only adds 0*constant=0 to Q in ONNX (a no-op).
        if ext_k is not None and ext_v is not None:
            # ext_k/v: (n_heads, ch, ref_seq) — from KVExtractor; already fp16 in
            # UNetCUDAGraphRunner static buffers so .to() is a cheap no-op.
            ek = ext_k.to(dtype=qt.dtype, device=qt.device)  # (nh, ch, ref_seq)
            ev = ext_v.to(dtype=qt.dtype, device=qt.device)
            # Unsqueeze batch dim and transpose to (1, nh, ref_seq, ch) for SDPA
            ek_t = ek.unsqueeze(0).permute(0, 1, 3, 2)  # (1, nh, ref_seq, ch)
            ev_t = ev.unsqueeze(0).permute(0, 1, 3, 2)
            k_use = torch.cat([kt, ek_t], dim=2)  # (b, nh, hw+ref_seq, ch)
            v_use = torch.cat([vt, ev_t], dim=2)
        else:
            k_use, v_use = kt, vt

        # 4D SDPA — dispatches to Flash Attention or Memory-Efficient Attention
        a = F.scaled_dot_product_attention(qt, k_use, v_use)  # (b, nh, hw, ch)
        h = a.permute(0, 1, 3, 2).reshape(b, c, hw)  # (b, c, hw)
        h = self.proj_out(h)
        out = (r + h).reshape(b, c, *spatial)
        return out.contiguous(memory_format=torch.channels_last) if was_cl else out


class _UNetTES(nn.Sequential):
    """
    TimestepEmbedSequential — routes timestep emb and external K/V to children.
    Mirrors UNet_TimestepEmbedSequential from ref_ldm_kv_embedding.py.
    """

    def forward(  # type: ignore[override]
        self,
        x: torch.Tensor,
        emb: torch.Tensor,
        ext_kv_map: Optional[Dict] = None,
        use_exclusive: bool = False,
        block_name: str = "",
    ) -> torch.Tensor:
        for i, layer in enumerate(self):
            if isinstance(layer, _UNetAttnBlock):
                path = f"{block_name}.{i}.attention"
                entry = ext_kv_map.get(path) if ext_kv_map else None
                ek = entry.get("k") if entry else None
                ev = entry.get("v") if entry else None
                x = layer(x, ext_k=ek, ext_v=ev, use_exclusive=use_exclusive)
            elif isinstance(layer, _UNetResBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


# ===========================================================================
# UNet
# ===========================================================================


class RefLDMUNetTorch(nn.Module):
    """
    Denoising UNet with external K/V reference conditioning.
    Config: model_channels=160, channel_mult=[1,2,2,4], num_res_blocks=2,
            attention_resolutions=[2,4,8], num_head_channels=32,
            in_channels=16, out_channels=8.
    """

    _MC = 160
    _MULT = (1, 2, 2, 4)
    _NRB = 2
    _AR = {2, 4, 8}
    _NHC = 32
    _IC = 16
    _OC = 8

    def __init__(self):
        super().__init__()
        mc, mult, nrb, ar, nhc = self._MC, self._MULT, self._NRB, self._AR, self._NHC
        ic, oc = self._IC, self._OC
        num_levels = len(mult)
        emb_dim = mc * 4

        # Timestep embedding
        self.time_embed = nn.Sequential(
            nn.Linear(mc, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )

        # Input blocks (down path)
        self.input_blocks: nn.ModuleList = nn.ModuleList(
            [_UNetTES(nn.Conv2d(ic, mc, 3, 1, 1))]
        )
        ch = mc
        ds = 1
        skips = [mc]

        for level, m in enumerate(mult):
            for _ in range(nrb):
                layers: List[nn.Module] = [
                    _UNetResBlock(ch, emb_dim, 0.0, out_ch=m * mc)
                ]
                ch = m * mc
                if ds in ar:
                    layers.append(_UNetAttnBlock(ch, nhc))
                self.input_blocks.append(_UNetTES(*layers))
                skips.append(ch)
            if level != num_levels - 1:
                self.input_blocks.append(_UNetTES(_UNetDownsample(ch)))
                skips.append(ch)
                ds *= 2

        # Middle block
        self.middle_block = _UNetTES(
            _UNetResBlock(ch, emb_dim, 0.0),
            _UNetAttnBlock(ch, nhc),
            _UNetResBlock(ch, emb_dim, 0.0),
        )

        # Output blocks (up path)
        self.output_blocks: nn.ModuleList = nn.ModuleList([])
        for level, m in list(enumerate(mult))[::-1]:
            for i in range(nrb + 1):
                skip_ch = skips.pop()
                layers = [_UNetResBlock(ch + skip_ch, emb_dim, 0.0, out_ch=m * mc)]
                ch = m * mc
                if ds in ar:
                    layers.append(_UNetAttnBlock(ch, nhc))
                if level and i == nrb:
                    layers.append(_UNetUpsample(ch))
                    ds //= 2
                self.output_blocks.append(_UNetTES(*layers))

        self.out = nn.Sequential(
            _GN(
                ch, fuse_silu=True
            ),  # SiLU fused — eliminates separate activation kernel
            nn.Identity(),  # placeholder (was nn.SiLU); index preserved for out.2 conv
            _zero_module(nn.Conv2d(ch, oc, 3, 1, 1)),
        )
        self._use_cl: bool = False

    def to_channels_last(self) -> "RefLDMUNetTorch":
        """Convert Conv2d weights to channels-last (NHWC) for faster cuDNN convolutions.
        Conv1d (used in attention QKV/proj_out) does not support channels_last and is
        left unchanged.  Call *after* .cuda().eval() and weight loading.  Returns self."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data = m.weight.data.to(memory_format=torch.channels_last)
        self._use_cl = True
        return self

    def forward(
        self,
        x: torch.Tensor,  # (1,16,64,64)  f32
        timesteps: torch.Tensor,  # (1,)  int64
        kv_map: Optional[Dict] = None,  # {path: {"k": tensor, "v": tensor}}
        use_exclusive: bool = False,
    ) -> torch.Tensor:
        x16 = x.to(self.input_blocks[0][0].weight.dtype)
        if self._use_cl:
            x16 = x16.contiguous(memory_format=torch.channels_last)
        t_emb = _timestep_embedding(timesteps, self.time_embed[0].in_features).to(
            x16.dtype
        )
        emb = self.time_embed(t_emb)

        hs: List[torch.Tensor] = []
        h = x16
        for i, blk in enumerate(self.input_blocks):
            h = blk(
                h,
                emb,
                ext_kv_map=kv_map,
                use_exclusive=use_exclusive,
                block_name=f"input_blocks.{i}",
            )
            hs.append(h)

        h = self.middle_block(
            h,
            emb,
            ext_kv_map=kv_map,
            use_exclusive=use_exclusive,
            block_name="middle_block",
        )

        for i, blk in enumerate(self.output_blocks):
            h = torch.cat([h, hs.pop()], dim=1)
            h = blk(
                h,
                emb,
                ext_kv_map=kv_map,
                use_exclusive=use_exclusive,
                block_name=f"output_blocks.{i}",
            )

        return self.out(h).float()

    @classmethod
    def from_onnx(
        cls,
        onnx_path: str,
        compute_dtype: torch.dtype = torch.float16,
        verbose: bool = False,
    ) -> "RefLDMUNetTorch":
        import onnx
        import re
        from onnx import numpy_helper

        onnx_model = onnx.load(onnx_path)
        w = {
            init.name: torch.from_numpy(numpy_helper.to_array(init).copy())
            for init in onnx_model.graph.initializer
        }

        # GroupNorm weights are stored as path-style initializers (pre-unsqueezed to (C,1,1)).
        # Extract them and add under their flat pt-style key ("…in_layers.0.weight/bias").
        #
        # Path pattern:  /unet_model/<blk_path>/in_layers/in_layers.0/Unsqueeze_output_0   → in_layers.0.weight
        #                /unet_model/<blk_path>/in_layers/in_layers.0/Unsqueeze_1_output_0  → in_layers.0.bias
        #                /unet_model/<blk_path>/out_layers/out_layers.0/Unsqueeze_output_0  → out_layers.0.weight
        #                /unet_model/<blk_path>/out_layers/out_layers.0/Unsqueeze_1_output_0 → out_layers.0.bias
        #                /unet_model/<blk_path>/norm/Unsqueeze_output_0                      → norm.weight
        #                /unet_model/<blk_path>/norm/Unsqueeze_1_output_0                    → norm.bias
        #                /unet_model/out/out.0/Unsqueeze_output_0                            → out.0.weight
        # Pattern for block-level GN (in_layers.0, out_layers.0, attn norm)
        _gn_pat = re.compile(
            r"^/unet_model/(?P<blk>.+?)"
            r"/(?P<sub>in_layers/in_layers\.0|out_layers/out_layers\.0|norm)"
            r"/Unsqueeze(?P<idx>_1)?_output_0$"
        )
        # Pattern for the final output GN (out.0)
        _out_gn_pat = re.compile(
            r"^/unet_model/out/out\.0/Unsqueeze(?P<idx>_1)?_output_0$"
        )
        for path_key, tensor in list(w.items()):
            # Check final output GN first
            m_out = _out_gn_pat.match(path_key)
            if m_out:
                suffix = "bias" if m_out.group("idx") else "weight"
                w[f"out.0.{suffix}"] = tensor.squeeze()
                continue

            m = _gn_pat.match(path_key)
            if m is None:
                continue
            # Flatten block path: "input_blocks.1/input_blocks.1.0" → "input_blocks.1.0"
            blk_raw = m.group("blk")
            blk_flat = blk_raw.split("/")[-1]  # keep deepest, e.g. "input_blocks.1.0"

            sub = m.group("sub")
            suffix = "bias" if m.group("idx") else "weight"

            if sub == "in_layers/in_layers.0":
                pt_key = f"{blk_flat}.in_layers.0.{suffix}"
            elif sub == "out_layers/out_layers.0":
                pt_key = f"{blk_flat}.out_layers.0.{suffix}"
            else:  # norm (attention)
                pt_key = f"{blk_flat}.norm.{suffix}"

            w[pt_key] = tensor.squeeze()  # (C,1,1) or (C,1) → (C,)

        model = cls()
        _load_from_onnx_weights(
            model,
            w,
            prefixes=["unet_model.", "", "model.diffusion_model.", "diffusion_model."],
            verbose=verbose,
        )
        model._visomaster_onnx_path = str(onnx_path)
        return model.to(compute_dtype)


# ===========================================================================
# Weight loading helper
# ===========================================================================


def _load_from_onnx_weights(
    model: nn.Module,
    onnx_w: Dict[str, torch.Tensor],
    prefixes: List[str] = ("",),  # type: ignore[assignment]
    verbose: bool = False,
) -> Tuple[List[str], List[str]]:
    """
    Map ONNX initialiser names → PyTorch state-dict keys.

    Strategy: try each prefix in turn; if the shape matches, use that tensor.
    Falls back to matching by suffix (parameter name without any prefix).
    """
    sd = model.state_dict()
    mapped = {}
    _suffix_map = {k.rsplit(".", 1)[-1]: k for k in sd}  # last component → full key

    for pt_k, pt_v in sd.items():
        found: Optional[torch.Tensor] = None

        # 1. Try prefixed ONNX names
        for pfx in prefixes:
            onnx_k = pfx + pt_k
            if onnx_k in onnx_w and onnx_w[onnx_k].shape == pt_v.shape:
                found = onnx_w[onnx_k]
                break

        # 2. Try stripping common wrapper prefixes from ONNX side
        if found is None:
            for onnx_k, onnx_v in onnx_w.items():
                bare = onnx_k
                for pfx in prefixes:
                    if bare.startswith(pfx):
                        bare = bare[len(pfx) :]
                        break
                if bare == pt_k and onnx_v.shape == pt_v.shape:
                    found = onnx_v
                    break

        if found is not None:
            mapped[pt_k] = found

    missing, unexpected = model.load_state_dict(mapped, strict=False)
    loaded = len(sd) - len(missing)
    print(
        f"[RefLDMTorch] loaded {loaded}/{len(sd)} parameters from ONNX "
        f"({len(missing)} missing, {len(unexpected)} unexpected)."
    )
    if verbose and missing:
        print(f"  Missing: {missing[:5]} {'...' if len(missing) > 5 else ''}")
    return missing, unexpected


# ===========================================================================
# CUDA Graph runner  (VAE encoder and decoder only — fixed shapes)
# ===========================================================================


def build_cuda_graph_runner(
    model: nn.Module,
    inp_shape: Tuple[int, ...],
    warmup: int = 3,
    torch_compile: bool = False,
) -> "CUDAGraphRunner":
    """
    Capture a CUDA graph for a fixed-shape model.
    Returns a CUDAGraphRunner callable with the same signature as model(x).

    Args:
        torch_compile: If True, wrap the model with ``torch.compile`` before
                       capturing the CUDA graph.  Requires Triton; adds ~30–60 s
                       one-time compile overhead.
    """
    if torch_compile:
        try:
            from custom_kernels.compile_utils import apply_torch_compile
            device = next(model.parameters()).device
            example_inp = torch.zeros(inp_shape, dtype=torch.float32, device=device)
            # default avoids the Triton MLIR AV crash (0xC0000005 on Windows sm_89)
            # that mode='reduce-overhead' can trigger in the subprocess ptxas optimizer.
            compiled = apply_torch_compile(model, example_inp, compile_mode="default")
            print("[ref_ldm] torch.compile default done.")
            return compiled
        except Exception as e:
            print(f"[ref_ldm] torch.compile failed ({e!s:.120}), falling back to CUDA graph.")

    return CUDAGraphRunner(model, inp_shape, warmup)


class CUDAGraphRunner:
    """Wraps a fixed-shape model in a CUDA graph for minimal kernel-launch overhead."""

    def __init__(self, model: nn.Module, inp_shape: Tuple[int, ...], warmup: int = 3):
        self.model = model
        self.inp_shape = inp_shape

        # Allocate static I/O buffers on the same device as model params
        device = next(model.parameters()).device
        self._inp = torch.zeros(inp_shape, dtype=torch.float32, device=device)

        # Warm-up passes (trigger JIT, Triton compilation, cuBLAS autotune)
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(self._inp)

        # Capture
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
        """Run the captured graph. x must have the same shape as inp_shape."""
        self._inp.copy_(x, non_blocking=True)
        self._graph.replay()
        return self._out.clone()


# ===========================================================================
# CUDA Graph runner for UNet  (handles dynamic K/V via static fp16 buffers)
# ===========================================================================


class UNetCUDAGraphRunner:
    """
    CUDA-graph-captured UNet runner.

    The UNet takes variable K/V attention maps per call (from the reference
    image KV extractor), which would normally prevent CUDA graph capture.
    This runner solves the problem by:

      1. Pre-allocating *static* fp16 K/V buffers (one per attention block),
         matching the dtype the attention kernel actually uses.
      2. Capturing the CUDA graph with those static buffers as K/V inputs.
         Because the model accesses them via Python dict lookup at capture
         time, the recorded CUDA kernels have their addresses baked in.
      3. Before each replay, ``copy_()`` the real K/V values into the static
         buffers — these copies are NOT part of the graph (they run on the
         default stream before the replay).

    Key requirement: ``kv_map_template`` must have the same paths and tensor
    shapes as every subsequent ``kv_map`` passed to ``__call__``.

    Typical usage::

        runner = UNetCUDAGraphRunner(
            unet_fp16, x_shape=(1, 16, 64, 64),
            ts_example=torch.tensor([500], dtype=torch.int64, device="cuda"),
            kv_map_template=kv_map,   # shapes extracted once from reference image
        )
        out = runner(x_noisy, timesteps, kv_map)
    """

    def __init__(
        self,
        model: "RefLDMUNetTorch",
        x_shape: Tuple[int, ...],
        ts_example: torch.Tensor,
        kv_map_template: Dict,
        use_exclusive: bool = True,
        warmup: int = 3,
    ) -> None:
        device = next(model.parameters()).device

        # --- Static input tensors (fixed GPU addresses) ---
        self._static_x = torch.zeros(x_shape, dtype=torch.float32, device=device)
        self._static_ts = ts_example.clone().to(device)

        # --- Static K/V buffers in fp16 (matching model compute dtype) ---
        # The attention blocks do ``ext_k.to(dtype=q.dtype)``; since q.dtype == float16
        # and our buffers are already float16, .to() is a no-op that returns the
        # same tensor object.  This ensures the CUDA graph records the correct
        # fixed pointer and copy_() updates propagate correctly on replay.
        self._static_kv: Dict = {}
        for path, entry in kv_map_template.items():
            sk = entry["k"].to(dtype=torch.float16, device=device).contiguous()
            sv = entry["v"].to(dtype=torch.float16, device=device).contiguous()
            self._static_kv[path] = {"k": sk, "v": sv}

        # --- Warm-up: trigger Triton JIT, cuDNN autotune, cudnn heuristics ---
        print("[UNetCUDAGraph] Warming up (Triton JIT / cuDNN autotune)...")
        with torch.no_grad():
            for _ in range(warmup):
                model(
                    self._static_x,
                    self._static_ts,
                    kv_map=self._static_kv,
                    use_exclusive=use_exclusive,
                )
        torch.cuda.synchronize(device)

        # --- Capture ---
        print("[UNetCUDAGraph] Capturing CUDA graph...")
        self._capture_stream = torch.cuda.Stream(device=device)
        self._graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize(device)
        with torch.no_grad():
            with torch.cuda.graph(
                self._graph,
                stream=self._capture_stream,
                capture_error_mode="relaxed",
            ):
                self._static_out = model(
                    self._static_x,
                    self._static_ts,
                    kv_map=self._static_kv,
                    use_exclusive=use_exclusive,
                )
        torch.cuda.synchronize(device)
        print("[UNetCUDAGraph] CUDA graph captured.")

    def __call__(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        kv_map: Dict,
        use_exclusive: bool = True,
    ) -> torch.Tensor:
        """Replay the captured CUDA graph with new inputs.

        ``kv_map`` must have the same paths and shapes as the template passed
        to the constructor.  The values are copied into the static buffers
        before replay; the ``use_exclusive`` flag is ignored at replay time
        (it was baked in at capture time).
        """
        self._static_x.copy_(x, non_blocking=True)
        self._static_ts.copy_(timesteps, non_blocking=True)
        for path, entry in kv_map.items():
            if path in self._static_kv:
                self._static_kv[path]["k"].copy_(entry["k"], non_blocking=True)
                self._static_kv[path]["v"].copy_(entry["v"], non_blocking=True)
        self._graph.replay()
        return self._static_out.clone()


def build_unet_cuda_graph_runner(
    model: "RefLDMUNetTorch",
    x_shape: Tuple[int, ...],
    ts_example: torch.Tensor,
    kv_map_template: Dict,
    use_exclusive: bool = True,
    warmup: int = 3,
    torch_compile: bool = False,
) -> UNetCUDAGraphRunner:
    """Convenience factory — see ``UNetCUDAGraphRunner`` for full documentation.

    Args:
        torch_compile: If True, wrap the model with ``torch.compile`` before
                       capturing the CUDA graph.  Requires Triton; complex UNet
                       adds ~60–120 s one-time compile overhead.
    """
    if torch_compile:
        try:
            from custom_kernels.compile_utils import setup_compile_env
            # TORCHINDUCTOR_USE_STATIC_CUDA_LAUNCHER=0 is set here, BEFORE torch.compile,
            # so compiled kernels use a 64-bit-safe launcher.  This prevents the
            # "Python int too large to convert to C long" overflow that occurs on Windows
            # when ctypes.c_long (32-bit) stores 64-bit CUDA function handles inside
            # _StaticCudaLauncher._launch_kernel during CUDA graph capture.
            setup_compile_env()
            _compiled = torch.compile(model, mode="default", fullgraph=False, dynamic=None)
            _device = next(model.parameters()).device
            _x_ex = torch.zeros(x_shape, dtype=torch.float32, device=_device)
            print("[ref_ldm UNet] Warming up torch.compile (this may take ~2 min)...")
            with torch.no_grad():
                for _ in range(warmup):
                    _compiled(_x_ex, ts_example, kv_map=kv_map_template, use_exclusive=use_exclusive)
            torch.cuda.synchronize()
            print("[ref_ldm UNet] torch.compile warmup done.")
            # Only assign to model after successful warmup, so the except clause below
            # always gets the original model for the fallback CUDA graph capture.
            model = _compiled
            # Fall through to UNetCUDAGraphRunner — the CUDA graph is captured over the
            # compiled kernels (same as the benchmark Tier 5).  Static K/V buffers mean
            # inference always replays the same fixed graph; dynamo never sees variable
            # K/V shapes so the recompile_limit is never hit during inference.
        except Exception as e:
            print(f"[ref_ldm UNet] torch.compile failed ({e!s:.120}), using CUDA graph only.")
            # model is still the original nn.Module — CUDA graph below works without compile

    return UNetCUDAGraphRunner(
        model,
        x_shape,
        ts_example,
        kv_map_template,
        use_exclusive=use_exclusive,
        warmup=warmup,
    )
