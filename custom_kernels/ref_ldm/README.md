# ReF-LDM Custom Kernel

FP16 PyTorch reimplementation of the three **ReF-LDM** denoiser ONNX models,
with Triton fused GroupNorm+SiLU kernels and CUDA graph capture for all three
components.  Used by VisoMaster-Fusion when the *Custom* execution provider is
selected.

## Models

| Model | Input | Output |
|-------|-------|--------|
| VAE Encoder | (1,3,512,512) f32 | (1,8,64,64) f32 |
| VAE Decoder | (1,8,64,64) f32 | (1,3,512,512) f32 |
| UNet denoiser | (1,16,64,64) f32 + external K/V | (1,8,64,64) f32 |

## Benchmark Results

**Hardware:** NVIDIA GeForce RTX 4090 · PyTorch 2.8.0+cu129 · CUDA 12.9 · Triton 3.4.0 · ORT 1.22.0
**Conditions:** 200 iterations, 50 warm-up passes

> **Note on benchmark isolation:** Building multiple ORT TRT engines in one process causes
> a native access violation (0xC0000005) in `torch_python.dll`. The benchmark runs each
> sub-benchmark in an isolated subprocess (`--encoder` / `--decoder` / `--unet` flags) so
> all three TRT engines build and measure correctly.

### VAE Encoder (1,3,512,512) → (1,8,64,64)

| Tier | Method | Latency | vs ORT CUDA EP |
|------|--------|--------:|:--------------:|
| 0    | ORT FP32 CUDA EP (baseline) | 20.98 ms | 1.00× |
| 0b   | ORT TRT EP FP32 | 12.35 ms | 1.70× |
| 1    | PyTorch FP32 pure ops | 21.95 ms | 0.96× |
| 2    | PyTorch FP16 + Triton GroupNorm+SiLU | 10.65 ms | 1.97× |
| **3** | **FP16 + Triton + CUDA graph (Custom)** | **10.43 ms** | **2.01×** |
| 4    | FP16 + Triton + CUDA graph + NHWC | 13.94 ms | 1.51× *(slower — NHWC overhead)* |
| **5** | **torch.compile + FP16 + Triton + CUDA graph** | **8.39 ms** | **2.50×** |
| 5b   | torch.compile reduce-overhead | — *(skipped by default; set `REFLDM_TORCH_COMPILE=1`)* | — |

> **Application uses Tier 3** (CUDA graph, NCHW — NHWC is slower for encoder).
> Pass `torch_compile=True` to `build_cuda_graph_runner` to activate Tier 5 (2.50×).

### VAE Decoder (1,8,64,64) → (1,3,512,512)

| Tier | Method | Latency | vs ORT CUDA EP |
|------|--------|--------:|:--------------:|
| 0    | ORT FP32 CUDA EP (baseline) | 96.06 ms | 1.00× |
| 0b   | ORT TRT EP FP32 | 22.27 ms | 4.31× |
| 1    | PyTorch FP32 pure ops | 36.04 ms | 2.67× |
| 2    | PyTorch FP16 + Triton GroupNorm+SiLU | 17.34 ms | 5.54× |
| **3** | **FP16 + Triton + CUDA graph (Custom)** | **16.97 ms** | **5.66×** |
| 4    | FP16 + Triton + CUDA graph + NHWC | 23.88 ms | 4.02× *(slower — NHWC overhead)* |
| **5** | **torch.compile + FP16 + Triton + CUDA graph** | **14.42 ms** | **6.66×** |
| 5b   | torch.compile reduce-overhead | — *(skipped by default; set `REFLDM_TORCH_COMPILE=1`)* | — |

> **ORT CUDA EP baseline is slow (96 ms)** because ORT inserts 6 memcpy nodes for the decoder
> (ConvTranspose asymmetric padding falls back to CPU path, requiring CPU↔GPU transfers).
> PyTorch runs the decoder fully on GPU. The `norm_out` GroupNorm fuses SiLU into the Triton
> kernel, saving ~0.5 ms per call. **Application uses Tier 3** (NCHW; NHWC regresses here).
> Pass `torch_compile=True` to `build_cuda_graph_runner` to activate Tier 5 (6.66×).

### UNet Denoiser (1,16,64,64) + K/V → (1,8,64,64)

| Tier | Method | Latency | vs ORT CUDA EP |
|------|--------|--------:|:--------------:|
| 0    | ORT FP32 CUDA EP — no K/V (baseline) | 9.63 ms | 1.00× |
| 0b   | ORT TRT EP FP32 | 6.68 ms | 1.44× |
| 1    | PyTorch FP32 pure ops (with K/V) | 27.69 ms | 0.35× |
| 2    | PyTorch FP16 + Triton GroupNorm+SiLU (with K/V) | 27.76 ms | 0.35× |
| **3** | **FP16 + Triton + CUDA graph (Custom)** | **5.25 ms** | **1.83×** |
| **4** | **FP16 + Triton + CUDA graph + NHWC** | **5.24 ms** | **1.84×** |
| **5** | **torch.compile + FP16 + Triton + CUDA graph** | **2.94 ms** | **3.27×** |
| 5b   | torch.compile reduce-overhead | — *(skipped by default; set `REFLDM_TORCH_COMPILE=1`)* | — |

> The UNet CUDA graph (Tier 3/4) works because K/V tensors are **static buffers** pre-allocated
> at capture time; real K/V values are copied in before each replay — the graph sees constant shapes.
> Tiers 1–2 run eager with real K/V (Python/kernel-launch overhead dominates).
> Once captured, the UNet runs at **5.25 ms** — **1.83× faster than ORT CUDA EP** (no-K/V baseline).
> NHWC (Tier 4) is neutral for the UNet; Tier 3 and Tier 4 are essentially equivalent.
>
> **Application uses Tier 4** (CUDA graph + NHWC). Pass `torch_compile=True` to
> `build_unet_cuda_graph_runner` to activate Tier 5 (3.27×).

> **Application uses:**
> - VAE Encoder → **Tier 3** (CUDA graph, NCHW — 2.01× vs ORT CUDA EP)
> - VAE Decoder → **Tier 3** (CUDA graph, NCHW — 5.66× vs ORT CUDA EP)
> - UNet → **Tier 4** (CUDA graph + NHWC — 1.84× vs ORT CUDA EP)

### Kernel Priority Chain

1. **Triton `triton_group_norm_silu`** — Windows-friendly, no MSVC required (preferred)
2. **Pure PyTorch** — automatic fallback (still FP16)

---

## Architecture

### VAE (VQModelInterface)

Config from `configs/refldm.yaml`:
```
ch=128, ch_mult=[1,1,2,4], num_res_blocks=2
attn_resolutions=[]   (NO attention in encoder/decoder blocks)
z_channels=8, double_z=False
```

**Encoder** path: `conv_in → 4× DownBlock[2×ResBlock + optional Downsample] → mid[ResBlock+Attn+ResBlock] → norm_out+SiLU(fused) → conv_out → quant_conv`

**Decoder** path: `post_quant_conv → conv_in → mid → 4× UpBlock[2×ResBlock + optional Upsample] → norm_out+SiLU(fused) → conv_out`

> The exported ONNX decoder does **not** include a VQ lookup step — the upstream pipeline
> passes latents directly to `post_quant_conv`.

**norm_out SiLU fusion**: Both encoder and decoder fuse the final GroupNorm and SiLU
activation into a single Triton kernel pass.  This eliminates one read + write of the
output feature map (most significant for the decoder where spatial resolution is 512×512).

### UNet Denoiser

Config from `configs/refldm.yaml`:
```
model_channels=160, channel_mult=[1,2,2,4], num_res_blocks=2
attention_resolutions=[2,4,8]   (spatial resolutions 32, 16, 8)
num_head_channels=32
in_channels=16, out_channels=8
use_spatial_transformer=False   (uses AttentionBlock, not CrossAttn)
```

**External K/V mechanism**: Each `AttentionBlock` at attention resolutions
receives reference K/V tensors extracted by `KVExtractor` from a reference image.

- When `use_exclusive=True`: attention uses **only** reference K/V (no self-attention)
- When `use_exclusive=False`: self-K/V concatenated with reference K/V along sequence dim

K/V tensors are keyed by PyTorch module path (e.g. `"input_blocks.4.1.attention"`)
with shape `(n_heads, ch_per_head, seq_len)`.

**Triton fused GroupNorm+SiLU kernel (`triton_group_norm_silu`):**
Each ResBlock in both VAE and UNet contains `GroupNorm(32) → SiLU` pairs.
The Triton kernel fuses these into a single two-pass operation:
- Pass 1: compute per-group mean and variance (FP32 accumulators)
- Pass 2: normalize, apply weight/bias, optional SiLU fusion
- Grid: `(N*G,)` programs — one per (batch, group)
- Eliminates the `GroupNorm32` FP16→FP32→FP16 round-trip

**Flash attention**: All attention blocks use `F.scaled_dot_product_attention`
(PyTorch 2.0+ automatically dispatches to FlashAttention on supported hardware).

---

## Files

| File | Purpose |
|------|---------|
| `ref_ldm_torch.py` | `RefLDMEncoderTorch`, `RefLDMDecoderTorch`, `RefLDMUNetTorch` + Triton kernels + CUDA graph runner |
| `benchmark_ref_ldm.py` | 5-tier latency benchmark vs ORT baseline (each sub-benchmark runs in an isolated subprocess) |
| `benchmark_results.txt` | Latest benchmark output |
| `__init__.py` | Package marker |

The Triton GroupNorm+SiLU kernel is defined in `custom_kernels/triton_ops.py`
as `triton_group_norm_silu` (Kernel 5).

---

## Application Integration

Select **"Custom"** in *Settings → General → Providers Priority*.

All three ReF-LDM models are then executed via `RefLDMEncoderTorch`,
`RefLDMDecoderTorch`, and `RefLDMUNetTorch` respectively.

```python
# Internal call path (face_restorers.py):
from custom_kernels.ref_ldm.ref_ldm_torch import (
    RefLDMEncoderTorch, RefLDMDecoderTorch, RefLDMUNetTorch,
    build_cuda_graph_runner, build_unet_cuda_graph_runner,
)
enc  = RefLDMEncoderTorch.from_onnx(onnx_path).cuda().eval()
dec  = RefLDMDecoderTorch.from_onnx(onnx_path).cuda().eval()
unet = RefLDMUNetTorch.from_onnx(onnx_path).cuda().eval()

enc_runner  = build_cuda_graph_runner(enc,  inp_shape=(1, 3, 512, 512))
dec_runner  = build_cuda_graph_runner(dec,  inp_shape=(1, 8, 64,  64 ))
unet_runner = build_unet_cuda_graph_runner(
    unet, x_shape=(1, 16, 64, 64), ts_example=timestep,
    kv_map_template=kv_map, use_exclusive=True,
)

latent     = enc_runner(image_f32_cuda)              # (1,3,512,512)f32 → (1,8,64,64)f32
image_out  = dec_runner(latent)                      # (1,8,64,64)f32 → (1,3,512,512)f32
noise_pred = unet_runner(x_noisy, timestep, kv_map)  # denoising step
```

VAE models are CUDA-graph-captured on first use (3 warm-up passes + capture).
UNet uses `UNetCUDAGraphRunner` which pre-allocates static K/V buffers so the
variable reference K/V tensors are compatible with CUDA graph replay.

---

## Weight Loading

Weights are loaded from the ONNX initializers.  The `_load_from_onnx_weights`
helper tries multiple prefix strategies:

- `RefLDMEncoderTorch`: prefixes `["", "encoder.", "first_stage_model."]`
- `RefLDMDecoderTorch`: prefixes `["", "decoder.", "first_stage_model."]`
- `RefLDMUNetTorch`: prefixes `["unet_model.", "", "model.diffusion_model.", "diffusion_model."]`

If a prefix match fails, falls back to suffix matching (parameter name without
any top-level namespace).

---

## Numerical Accuracy

FP16 GroupNorm with FP32 accumulators in the Triton kernel provides
accuracy equivalent to the `GroupNorm32` FP32 upcast used in the original
LDM code, while avoiding redundant dtype conversions.  The fused SiLU path
computes `out * sigmoid(out)` inside the same Triton pass — numerically
identical to a separate SiLU call.

Expected mean relative error vs ORT FP32: `< 2%` for both VAE components.
UNet: `< 2%` mean error (K/V attention paths may show slightly higher variance
at very low sequence lengths).

## Running the Benchmark

```bat
.venv/Scripts/python custom_kernels/ref_ldm/benchmark_ref_ldm.py
```

Optional env vars: `WARMUP` (default 50), `ITERS` (default 200), `ONNX_DIR`, `SKIP_TRT=1`

Run individual sub-benchmarks directly:
```bat
.venv/Scripts/python custom_kernels/ref_ldm/benchmark_ref_ldm.py --encoder
.venv/Scripts/python custom_kernels/ref_ldm/benchmark_ref_ldm.py --decoder
.venv/Scripts/python custom_kernels/ref_ldm/benchmark_ref_ldm.py --unet
```

> **Tier 5b (`torch.compile reduce-overhead`)** is skipped by default on Windows/sm_89.
> Set `REFLDM_TORCH_COMPILE=1` to attempt.
