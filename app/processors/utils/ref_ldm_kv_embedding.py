"""
A standalone, self-contained module to extract K/V (Key/Value) tensor 
embeddings from a reference image using the ReF-LDM model.

This file contains all necessary code from the 'ldm' package inlined,
removing the need for the 'ldm' package to be in the Python path.

Prerequisites:
- PyTorch, OmegaConf, NumPy, Pillow, and einops must be installed.
  (pip install torch omegaconf numpy pillow einops)
- Model checkpoints ('refldm.ckpt', 'vqgan.ckpt') and the configuration
  file ('refldm.yaml') must be in their default locations ('ckpts/', 'configs/').
- The Taming Transformers library is an implicit dependency for the VAE, as
  referenced in the original project. If you encounter errors related to 'taming',
  you may need to install it.

Usage:
    from kv_extractor_standalone import KVExtractor
    from PIL import Image

    # Create a dummy 512x512 PIL image
    dummy_image = Image.new("RGB", (512, 512), color="red")

    # Initialize the extractor (loads the model once)
    extractor = KVExtractor()

    # Extract the K/V embedding from the image
    kv_embedding = extractor.extract_kv(dummy_image)

    # The result is a dictionary mapping layer names to their K/V tensors
    print(f"Extracted K/V for {len(kv_embedding)} layers.")
    first_layer_name = list(kv_embedding.keys())[0]
    k_tensor = kv_embedding[first_layer_name]['k']
    v_tensor = kv_embedding[first_layer_name]['v']
    print(f"First layer '{first_layer_name}' K tensor shape: {k_tensor.shape}")
    print(f"First layer '{first_layer_name}' V tensor shape: {v_tensor.shape}")
"""
import gc
import math
import os
from abc import abstractmethod
from collections import defaultdict
from functools import partial
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import v2
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image
from einops import repeat

# === Inlined from ldm/util.py ===

def get_obj_from_str(string, reload=False):
    """
    Modified to look up class names in the globals() of this module
    instead of using importlib, as all necessary classes are inlined.
    """
    # Original logic:
    # module, cls = string.rsplit(".", 1)
    # return getattr(importlib.import_module(module, package=None), cls)
    
    cls_name = string.rsplit(".", 1)[-1]
    if cls_name not in globals():
        raise ValueError(f"Class '{cls_name}' not found in the inlined module's globals. "
                         f"The config references a class that was not included.")
    return globals()[cls_name]

def instantiate_from_config(config):
    if OmegaConf.is_config(config):
        config = OmegaConf.to_object(config)
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode does not change anymore."""
    return self


# === Inlined from ldm/cache_kv.py ===

class CacheKVModule:
    """A namespace class to hold the cache_kv logic."""
    def __init__(self):
        self.mode = None
        self.k = defaultdict(list)
        self.v = defaultdict(list)

    def clear_cache(self):
        self.k.clear()
        self.v.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

cache_kv_module = CacheKVModule()


# === Inlined from ldm/modules/diffusionmodules/util.py ===

def checkpoint(func, inputs, params, flag):
    return func(*inputs)

def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module
    
def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    if not repeat_only:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    else:
        embedding = repeat(timesteps, 'b -> b d', d=dim)
    return embedding
    
class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)

def UNet_normalization(channels):
    return GroupNorm32(32, channels)

def conv_nd(dims, *args, **kwargs):
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")

def linear(*args, **kwargs):
    return nn.Linear(*args, **kwargs)

def avg_pool_nd(dims, *args, **kwargs):
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


# === Inlined from ldm/modules/attention.py ===

class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)

def VAE_nonlinearity(x):
    return x * torch.sigmoid(x)

def VAE_Normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)

# === Inlined from ldm/modules/diffusionmodules/model.py (VAE Components) ===

class VAE_Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x

class VAE_Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x

class VAE_ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False, dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.norm1 = VAE_Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        self.norm2 = VAE_Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = VAE_nonlinearity(h)
        h = self.conv1(h)
        if temb is not None:
            h = h + self.temb_proj(VAE_nonlinearity(temb))[:, :, None, None]
        h = self.norm2(h)
        h = VAE_nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)
        return x + h

class VAE_AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.norm = VAE_Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w).permute(0, 2, 1)
        k = k.reshape(b, c, h * w)
        w_ = torch.bmm(q, k)
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)
        h_ = torch.bmm(v, w_)
        h_ = h_.reshape(b, c, h, w)
        h_ = self.proj_out(h_)
        return x + h_

class VAE_Encoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1, 2, 4, 8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, double_z=True, **ignore_kwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.conv_in = torch.nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)
        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(VAE_ResnetBlock(in_channels=block_in, out_channels=block_out, temb_channels=self.temb_ch, dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(VAE_AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = VAE_Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)
        self.mid = nn.Module()
        self.mid.block_1 = VAE_ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout)
        self.mid.attn_1 = VAE_AttnBlock(block_in)
        self.mid.block_2 = VAE_ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout)
        self.norm_out = VAE_Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in, 2 * z_channels if double_z else z_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        temb = None
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)
        h = self.norm_out(h)
        h = VAE_nonlinearity(h)
        h = self.conv_out(h)
        return h

class VAE_Decoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1, 2, 4, 8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, give_pre_end=False, tanh_out=False, **ignorekwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2**(self.num_resolutions - 1)
        self.conv_in = torch.nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)
        self.mid = nn.Module()
        self.mid.block_1 = VAE_ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout)
        self.mid.attn_1 = VAE_AttnBlock(block_in)
        self.mid.block_2 = VAE_ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout)
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(VAE_ResnetBlock(in_channels=block_in, out_channels=block_out, temb_channels=self.temb_ch, dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(VAE_AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = VAE_Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)
        self.norm_out = VAE_Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        temb = None
        h = self.conv_in(z)
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)
        if self.give_pre_end:
            return h
        h = self.norm_out(h)
        h = VAE_nonlinearity(h)
        h = self.conv_out(h)
        if self.tanh_out:
            h = torch.tanh(h)
        return h

class VectorQuantizer2(nn.Module):
    """
    Improved version over VectorQuantizer, can be used as a drop-in replacement. Mostly
    avoids costly matrix multiplications and allows for post-hoc remapping of indices.
    """
    # NOTE: due to a bug the beta term was applied to the wrong term. for
    # backwards compatibility we use the buggy version by default, but you can
    # specify legacy=False to fix it.
    def __init__(self, n_e, e_dim, beta, remap=None, unknown_index="random",
                 sane_index_shape=False, legacy=True):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.legacy = legacy

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index # "random" or "extra" or integer
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed+1
            print(f"Remapping {self.n_e} indices to {self.re_embed} indices. "
                  f"Using {self.unknown_index} for unknown indices.")
        else:
            self.re_embed = n_e

        self.sane_index_shape = sane_index_shape

    def remap_to_used(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        match = (inds[:,:,None]==used[None,None,...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2)<1
        if self.unknown_index == "random":
            new[unknown]=torch.randint(0,self.re_embed,size=new[unknown].shape).to(device=new.device)
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        if self.re_embed > self.used.shape[0]: # extra token
            inds[inds>=self.used.shape[0]] = 0 # simply set to zero
        back=torch.gather(used[None,:][inds.shape[0]*[0],:], 1, inds)
        return back.reshape(ishape)

    def forward(self, z, temp=None, rescale_logits=False, return_logits=False):
        assert temp is None or temp==1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits==False, "Only for interface compatible with Gumbel"
        assert return_logits==False, "Only for interface compatible with Gumbel"
        # reshape z -> (batch, height, width, channel) and flatten
        z = rearrange(z, 'b c h w -> b h w c').contiguous()
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, rearrange(self.embedding.weight, 'n d -> d n'))

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)
        perplexity = None
        min_encodings = None

        # compute loss for embedding
        if not self.legacy:
            loss = self.beta * torch.mean((z_q.detach()-z)**2) + \
                   torch.mean((z_q - z.detach()) ** 2)
        else:
            loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
                   torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()

        if self.remap is not None:
            min_encoding_indices = min_encoding_indices.reshape(z.shape[0],-1) # add batch axis
            min_encoding_indices = self.remap_to_used(min_encoding_indices)
            min_encoding_indices = min_encoding_indices.reshape(-1,1) # flatten

        if self.sane_index_shape:
            min_encoding_indices = min_encoding_indices.reshape(
                z_q.shape[0], z_q.shape[2], z_q.shape[3])

        return z_q, loss, (perplexity, min_encodings, min_encoding_indices)

    def get_codebook_entry(self, indices, shape):
        # shape specifying (batch, height, width, channel)
        if self.remap is not None:
            indices = indices.reshape(shape[0],-1) # add batch axis
            indices = self.unmap_to_all(indices)
            indices = indices.reshape(-1) # flatten again

        # get quantized latent vectors
        z_q = self.embedding(indices)

        if shape is not None:
            z_q = z_q.view(shape)
            # reshape back to match original input shape
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q


# === Inlined from ldm/models/autoencoder.py ===

class VQModelInterface(nn.Module):
    def __init__(self, ddconfig, n_embed, embed_dim, ckpt_path=None, ignore_keys=None, image_key="image", colorize_nlabels=None, monitor=None, remap=None, sane_index_shape=False):
        super().__init__()

        self.embed_dim = embed_dim
        self.n_embed = n_embed
        self.encoder = VAE_Encoder(**ddconfig)
        self.decoder = VAE_Decoder(**ddconfig)
        self.quantize = VectorQuantizer2(n_embed, embed_dim, beta=0.25, remap=remap, sane_index_shape=sane_index_shape)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys or [])

    def init_from_ckpt(self, path, ignore_keys=None):
        sd = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print(f"Deleting key {k} from state_dict.")
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(f"Restored VQGAN from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def decode(self, h):
        quant, emb_loss, info = self.quantize(h)
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec


# === Inlined from ldm/modules/diffusionmodules/openaimodel.py (UNet Components) ===
# ... (All classes from openaimodel.py, including UNetModel, ResBlock, AttentionBlock, etc.)
class UNet_TimestepBlock(nn.Module):
    @abstractmethod
    def forward(self, x, emb):
        """Apply the module to `x` given `emb` timestep embeddings."""

class UNet_TimestepEmbedSequential(nn.Sequential, UNet_TimestepBlock):
    def forward(self, x, emb, context=None, external_kv_map=None, use_reference_exclusive_path_globally=None, is_ref_pass_for_attention=None, module_block_name=""):
        for i, layer in enumerate(self):
            if isinstance(layer, AttentionBlock):
                qkv_module_path = f"{module_block_name}.{i}.attention"
                kv_entry = external_kv_map.get(qkv_module_path) if external_kv_map else None
                current_external_kv = {'k': kv_entry.get('k'), 'v': kv_entry.get('v')} if kv_entry else None
                x = layer(x, external_kv_for_attention=current_external_kv, use_reference_exclusive_path_for_attention=use_reference_exclusive_path_globally, is_ref_pass_for_attention=is_ref_pass_for_attention)
            elif isinstance(layer, UNet_TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x

class UNet_Upsample(nn.Module):
    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=padding)

    def forward(self, x):
        assert x.shape[1] == self.channels
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x

class UNet_Downsample(nn.Module):
    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2
        if use_conv:
            self.op = conv_nd(dims, self.channels, self.out_channels, 3, stride=stride, padding=padding)
        else:
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        return self.op(x)

class UNet_ResBlock(UNet_TimestepBlock):
    def __init__(self, channels, emb_channels, dropout, out_channels=None, dims=2, use_checkpoint=False, use_scale_shift_norm=False, up=False, down=False):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        self.in_layers = nn.Sequential(UNet_normalization(channels), nn.SiLU(), conv_nd(dims, channels, self.out_channels, 3, padding=1))
        self.updown = up or down
        if up: self.h_upd = self.x_upd = UNet_Upsample(channels, False, dims)
        elif down: self.h_upd = self.x_upd = UNet_Downsample(channels, False, dims)
        else: self.h_upd = self.x_upd = nn.Identity()
        self.emb_layers = nn.Sequential(nn.SiLU(), linear(emb_channels, 2 * self.out_channels if use_scale_shift_norm else self.out_channels))
        self.out_layers = nn.Sequential(UNet_normalization(self.out_channels), nn.SiLU(), nn.Dropout(p=dropout), zero_module(conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)))
        self.skip_connection = nn.Identity() if self.out_channels == channels else conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        return checkpoint(self._forward, (x, emb), self.parameters(), self.use_checkpoint)

    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h

class QKVAttentionLegacy(nn.Module):
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads
    
    def forward(self, qkv, external_k=None, external_v=None, use_reference_exclusive_path=None, is_ref_pass=None):
        bs, width, length = qkv.shape
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        
        if is_ref_pass:
            if cache_kv_module.mode == 'save':
                cache_kv_module.k[id(self)].append(k.detach().clone())
                cache_kv_module.v[id(self)].append(v.detach().clone())
        
        scale = ch**-0.5
        weight = torch.bmm(q.transpose(1, 2), k) * scale
        weight = F.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.bmm(v, weight.transpose(1, 2))
        return a.reshape(bs, -1, length)

class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=1, num_head_channels=-1, use_checkpoint=False, use_new_attention_order=False):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1: self.num_heads = num_heads
        else: self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = UNet_normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        self.attention = QKVAttentionLegacy(self.num_heads)
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x, external_kv_for_attention=None, use_reference_exclusive_path_for_attention=None, is_ref_pass_for_attention=None):
        return checkpoint(self._forward, (x, external_kv_for_attention, use_reference_exclusive_path_for_attention, is_ref_pass_for_attention), self.parameters(), self.use_checkpoint)
        
    def _forward(self, x, external_kv_for_attention, use_reference_exclusive_path_for_attention, is_ref_pass_for_attention):
        b, c, *spatial = x.shape
        x_reshaped = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x_reshaped))
        h = self.attention(qkv, is_ref_pass=is_ref_pass_for_attention)
        h = self.proj_out(h)
        return (x_reshaped + h).reshape(b, c, *spatial)

class UNetModel(nn.Module):
    def __init__(self, image_size, in_channels, model_channels, out_channels, num_res_blocks,
                 attention_resolutions, dropout=0, channel_mult=(1, 2, 4, 8), conv_resample=True,
                 dims=2, num_classes=None, use_ref_emb=False, use_checkpoint=False, num_heads=-1,
                 num_head_channels=-1, use_scale_shift_norm=False, resblock_updown=False, **kwargs):
        super().__init__()
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.use_ref_emb = use_ref_emb

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(linear(model_channels, time_embed_dim), nn.SiLU(), linear(time_embed_dim, time_embed_dim))
        
        if self.use_ref_emb: self.ref_emb = nn.Parameter(torch.zeros(1, time_embed_dim))
        else: self.ref_emb = None

        self.input_blocks = nn.ModuleList([UNet_TimestepEmbedSequential(conv_nd(dims, in_channels, model_channels, 3, padding=1))])
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [UNet_ResBlock(ch, time_embed_dim, dropout, out_channels=mult * model_channels, dims=dims, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm)]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch, use_checkpoint=use_checkpoint, num_heads=num_heads, num_head_channels=num_head_channels))
                self.input_blocks.append(UNet_TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                self.input_blocks.append(UNet_TimestepEmbedSequential(UNet_Downsample(ch, conv_resample, dims=dims)))
                input_block_chans.append(ch)
                ds *= 2

        self.middle_block = UNet_TimestepEmbedSequential(
            UNet_ResBlock(ch, time_embed_dim, dropout, dims=dims, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm),
            AttentionBlock(ch, use_checkpoint=use_checkpoint, num_heads=num_heads, num_head_channels=num_head_channels),
            UNet_ResBlock(ch, time_embed_dim, dropout, dims=dims, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm)
        )

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                layers = [UNet_ResBlock(ch + input_block_chans.pop(), time_embed_dim, dropout, out_channels=model_channels * mult, dims=dims, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm)]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch, use_checkpoint=use_checkpoint, num_heads=num_heads, num_head_channels=num_head_channels))
                if level and i == num_res_blocks:
                    layers.append(UNet_Upsample(ch, conv_resample, dims=dims))
                    ds //= 2
                self.output_blocks.append(UNet_TimestepEmbedSequential(*layers))
        
        self.out = nn.Sequential(UNet_normalization(ch), nn.SiLU(), zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)))

    def forward(self, x, timesteps, context=None, y=None, is_ref=None, external_kv_map=None, use_reference_exclusive_path_globally=None, **kwargs):
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)
        if self.use_ref_emb and is_ref:
            emb = self.ref_emb.repeat(x.shape[0], 1)
        h = x
        for i, module in enumerate(self.input_blocks):
            h = module(h, emb, context, external_kv_map=external_kv_map, use_reference_exclusive_path_globally=use_reference_exclusive_path_globally, is_ref_pass_for_attention=is_ref, module_block_name=f"input_blocks.{i}")
            hs.append(h)
        h = self.middle_block(h, emb, context, external_kv_map=external_kv_map, use_reference_exclusive_path_globally=use_reference_exclusive_path_globally, is_ref_pass_for_attention=is_ref, module_block_name="middle_block")
        for i, module in enumerate(self.output_blocks):
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context, external_kv_map=external_kv_map, use_reference_exclusive_path_globally=use_reference_exclusive_path_globally, is_ref_pass_for_attention=is_ref, module_block_name=f"output_blocks.{i}")
        return self.out(h)

# === Inlined from ldm/models/diffusion/ddpm.py ===

class LatentDiffusion(nn.Module):
    def __init__(self, first_stage_config, cond_stage_config, num_timesteps_cond=None,
                 cond_stage_key="image", cond_stage_trainable=False, concat_mode=True,
                 cond_stage_forward=None, conditioning_key=None, scale_factor=1.0,
                 scale_by_std=False, unet_config=None, *args, **kwargs):
        super().__init__()
        # This is a highly simplified version for K/V extraction, removing all training
        # and sampling logic.
        self.scale_factor = scale_factor
        self.first_stage_model = instantiate_from_config(first_stage_config)
        self.model = DiffusionWrapper(unet_config, conditioning_key)

    @torch.no_grad()
    def encode_first_stage(self, x):
        return self.first_stage_model.encode(x)

    def state_dict(self):
        # Simplified to get state dicts of sub-modules for loading.
        return {
            **{f"first_stage_model.{k}": v for k, v in self.first_stage_model.state_dict().items()},
            **{f"model.diffusion_model.{k}": v for k, v in self.model.diffusion_model.state_dict().items()}
        }

    def load_state_dict(self, state_dict, strict=True):
        # Handle loading for sub-modules.
        first_stage_keys = {k: v for k, v in state_dict.items() if k.startswith("first_stage_model.")}
        model_keys = {k: v for k, v in state_dict.items() if k.startswith("model.diffusion_model.")}
        
        missing1, unexpected1 = self.first_stage_model.load_state_dict({k.replace("first_stage_model.", ""): v for k, v in first_stage_keys.items()}, strict=False)
        missing2, unexpected2 = self.model.diffusion_model.load_state_dict({k.replace("model.diffusion_model.", ""): v for k, v in model_keys.items()}, strict=False)
        
        return (missing1 + missing2, unexpected1 + unexpected2)


class DiffusionWrapper(nn.Module):
    def __init__(self, diff_model_config, conditioning_key):
        super().__init__()
        self.diffusion_model = instantiate_from_config(diff_model_config)
        self.conditioning_key = conditioning_key
        assert self.conditioning_key in [None, 'concat', 'crossattn', 'hybrid', 'adm']

class KVExtractor:
    """
    A class to load the ReF-LDM model and extract K/V tensor embeddings from
    reference images. The model is loaded once upon initialization for efficiency.
    """
    DEFAULT_MODEL_CONFIG_PATH = "configs/refldm.yaml"
    DEFAULT_MODEL_CKPT_PATH = "ckpts/refldm.ckpt"
    DEFAULT_VAE_CKPT_PATH = "ckpts/vqgan.ckpt"

    def __init__(self,
                 model_config_path: str = DEFAULT_MODEL_CONFIG_PATH,
                 model_ckpt_path: str = DEFAULT_MODEL_CKPT_PATH,
                 vae_ckpt_path: str = DEFAULT_VAE_CKPT_PATH,
                 device: str = 'cpu'):
        """
        Initializes the KVExtractor and loads the required models.

        Args:
            model_config_path (str): Path to the model configuration YAML file.
            model_ckpt_path (str): Path to the ReF-LDM UNet checkpoint.
            vae_ckpt_path (str): Path to the VQGAN (VAE) checkpoint.
            device (str): The device to run the model on ('cpu' or 'cuda').
        """
        self.device = torch.device(device)
        print(f"Initializing KVExtractor on device: {self.device}")
        
        self._validate_paths(model_config_path, model_ckpt_path, vae_ckpt_path)
        
        print("Loading ReF-LDM model configuration...")
        model_config_full = OmegaConf.load(model_config_path)
        model_config_full.model.params.first_stage_config.params.ckpt_path = vae_ckpt_path
        
        keys_to_pop = [
            'ckpt_path', 'perceptual_loss_config', 'val_loss_run_ddim_steps',
            'val_loss_run_ddim_cfg_scale', 'val_loss_run_ddim_cfg_key', 'scheduler_config'
        ]
        for k in keys_to_pop:
            model_config_full.model.params.pop(k, None)

        # The VQModelInterface (first_stage_model) doesn't accept a 'lossconfig' parameter,
        # which is present in some VQGAN model configs. We pop it here to avoid an error.
        model_config_full.model.params.first_stage_config.params.pop('lossconfig', None)

        print("Instantiating full ReF-LDM model (PyTorch)...")
        self.model: LatentDiffusion = instantiate_from_config(model_config_full.model)

        print(f"Loading ReF-LDM UNet weights from: {model_ckpt_path}")
        state_dict_container = torch.load(model_ckpt_path, map_location='cpu', weights_only=False)
        ldm_state_dict = state_dict_container['state_dict'] if 'state_dict' in state_dict_container else state_dict_container

        missing, unexpected = self.model.load_state_dict(ldm_state_dict, strict=False)
        print(f"Model loaded. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")

        self.model.to(self.device)
        self.model.eval()
        self.unet_model: UNetModel = self.model.model.diffusion_model

    @staticmethod
    def _validate_paths(*paths):
        for p in paths:
            if not os.path.exists(p):
                raise FileNotFoundError(f"Required file not found: {p}")
        print("All model files and configs found.")

    @staticmethod
    def _normalize_image(image) -> torch.Tensor:
        if isinstance(image, Image.Image):
            if image.size != (512, 512):
                image = image.resize((512, 512), Image.Resampling.LANCZOS)
            
            image_rgb = image.convert("RGB")
            img_np = np.array(image_rgb, dtype=np.float32)
            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)
        elif isinstance(image, torch.Tensor):
            if image.shape[1:] != (512, 512):
                img_tensor = v2.Resize((512, 512), antialias=True)(image)
            else:
                img_tensor = image.float()
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        image_tensor_norm = img_tensor / 127.5 - 1.0
        return image_tensor_norm.unsqueeze(0)

    @torch.no_grad()
    def extract_kv(self,
                    image,
                    scale_factor: float = 1.0,
                    color_match_image: Optional[torch.Tensor] = None
                    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Extracts the K/V embedding from a single 512x512 reference image,
        which can be a PIL Image or a PyTorch Tensor.
        """
        print("Extracting K/V from reference image...")

        if color_match_image is not None:
            from app.processors.utils import faceutil
            from torchvision.transforms import v2
            print("Applying color matching to reference image for K/V extraction.")
            if isinstance(image, Image.Image):
                image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1)
            else:
                image_tensor = image.clone()

            if image_tensor.dtype != torch.uint8:
                image_tensor = image_tensor.clamp(0, 255).byte()
            if color_match_image.dtype != torch.uint8:
                color_match_image = color_match_image.clamp(0, 255).byte()

            if color_match_image.shape[1:] != (512,512):
                color_match_image = v2.Resize((512,512), antialias=True)(color_match_image)

            # Match color of 'image' to 'color_match_image'
            image = faceutil.histogram_matching(color_match_image, image_tensor, 100)

        ref_tensor = self._normalize_image(image).to(self.device)

        cache_kv_module.mode = 'save'
        cache_kv_module.clear_cache()
        dummy_timesteps = torch.zeros(1, device=self.device).long()
        
        if self.unet_model.in_channels is None:
            raise ValueError("UNetModel.in_channels is None.")

        latent_unscaled = self.model.encode_first_stage(ref_tensor)
        latent_for_unet = latent_unscaled
        
        pad_channels = self.unet_model.in_channels - latent_for_unet.shape[1]
        if pad_channels < 0:
            raise ValueError(f"Ref latent channels > UNet input channels")
        elif pad_channels > 0:
            latent_for_unet = F.pad(latent_for_unet, (0, 0, 0, 0, 0, pad_channels))

        is_ref_tensor = torch.tensor(True, device=self.device, dtype=torch.bool)
        _ = self.unet_model(latent_for_unet, dummy_timesteps, context=None, is_ref=is_ref_tensor)

        extracted_kv_map = {}
        for name, module in self.unet_model.named_modules():
            if isinstance(module, QKVAttentionLegacy):
                k_list = cache_kv_module.k.get(id(module), [])
                v_list = cache_kv_module.v.get(id(module), [])

                if k_list and v_list:
                    final_k = k_list[0].clone().detach() * scale_factor
                    final_v = v_list[0].clone().detach() * scale_factor
                    
                    extracted_kv_map[name] = {'k': final_k.cpu(), 'v': final_v.cpu()}

        cache_kv_module.clear_cache()
        cache_kv_module.mode = None
        
        print(f"Successfully extracted K/V for {len(extracted_kv_map)} attention layers.")
        return extracted_kv_map


if __name__ == '__main__':
    print("--- Running KV Extractor Standalone ---")

    if not all(os.path.exists(p) for p in [
        KVExtractor.DEFAULT_MODEL_CKPT_PATH,
        KVExtractor.DEFAULT_VAE_CKPT_PATH,
        KVExtractor.DEFAULT_MODEL_CONFIG_PATH
    ]):
        print("\nERROR: Default model/config files not found.")
        print("Please ensure the following files are in your project directory:")
        print(f"  - {KVExtractor.DEFAULT_MODEL_CKPT_PATH}")
        print(f"  - {KVExtractor.DEFAULT_VAE_CKPT_PATH}")
        print(f"  - {KVExtractor.DEFAULT_MODEL_CONFIG_PATH}")
    else:
        try:
            print("\n1. Creating a dummy 512x512 PIL image...")
            dummy_image = Image.new("RGB", (512, 512), color="blue")
            print("   Image created successfully.")

            print("\n2. Initializing KVExtractor (this may take a moment)...")
            device_to_use = 'cuda' if torch.cuda.is_available() else 'cpu'
            extractor = KVExtractor(device=device_to_use)
            print("   Extractor initialized successfully.")

            print("\n3. Extracting K/V embedding from the dummy image...")
            kv_embedding = extractor.extract_kv(dummy_image, scale_factor=1.0)
            print("   Extraction complete.")

            print("\n4. Inspecting the resulting K/V embedding:")
            if kv_embedding:
                print(f"   - Extracted K/V for {len(kv_embedding)} layers.")
                # Find a layer name for demonstration
                example_layer_name = next((k for k in kv_embedding if 'attention' in k), None)
                if example_layer_name is None:
                    example_layer_name = list(kv_embedding.keys())[0]

                k_tensor = kv_embedding[example_layer_name]['k']
                v_tensor = kv_embedding[example_layer_name]['v']
                print(f"   - Example Layer: '{example_layer_name}'")
                print(f"   - K Tensor Shape: {k_tensor.shape}")
                print(f"   - V Tensor Shape: {v_tensor.shape}")
                print(f"   - K Tensor DType: {k_tensor.dtype}")
                print(f"   - K Tensor Device: {k_tensor.device}")
            else:
                print("   - No K/V embeddings were extracted.")

        except Exception as e:
            print(f"\nAn error occurred during the demo: {e}")
            import traceback
            traceback.print_exc()

    print("\n--- Finished ---")
