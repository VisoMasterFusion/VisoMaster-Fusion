# RestoreFormer++ Custom Kernel

FP16 PyTorch reimplementation of the **RestoreFormer++** face restoration ONNX model,
with Triton fused GroupNorm+SiLU kernels and CUDA graph capture.
Used by VisoMaster-Fusion when the *Custom* execution provider is selected.

## Model

| Model | Input | Output |
|-------|-------|--------|
| RestoreFormerPlusPlus | (1,3,512,512) f32 | (1,3,512,512) f32 |

## Benchmark Results

**Environment:** NVIDIA GeForce RTX 4090 · PyTorch 2.8.0+cu129 · CUDA 12.9 · Triton 3.4.0 · ORT 1.22.0
**Method:** 50 iterations, 10 warm-up passes

| Tier | Method | Latency | vs ORT CUDA EP |
|------|--------|--------:|---------------:|
| 0  | ORT FP32 CUDA EP (baseline) | 19.47 ms | 1.00× |
| 0b | ORT TensorRT EP FP32 | 10.49 ms | 1.86× |
| 1  | PyTorch FP32 pure ops | 25.98 ms | 0.75× |
| 2  | PyTorch FP16 + Triton GroupNorm+SiLU | 24.77 ms | 0.79× |
| **3** | **FP16 + Triton + CUDA graph (Custom)** | **12.75 ms** | **1.53×** |
| **4** | **torch.compile default + FP16 + Triton + CUDA graph** | **10.57 ms** | **1.84×** |
| 4b | torch.compile reduce-overhead (no separate CUDA graph) | 10.50 ms | 1.85× |

Both compile modes work and achieve nearly identical speedup (~1.84–1.85×). `default+CUDA graph` is the recommended compile path since it is the application runtime.

> **Application uses Tier 3** (CUDA graph). Pass `torch_compile=True` to `build_cuda_graph_runner`
> to activate Tier 4 (default + CUDA graph, 1.21× faster than Tier 3).

> **Note on TRT EP accuracy:** ORT TRT EP max|diff| vs ORT CUDA EP = 3.11 (high),
> indicating numerical degradation with TRT for this model.

### Kernel Priority Chain

1. **Triton `triton_group_norm_silu`** — Windows-friendly, no MSVC required (preferred)
2. **Pure PyTorch** — automatic fallback (still FP16)

---

## Architecture

Config reverse-engineered from `model_assets/RestoreFormerPlusPlus.fp16.onnx`:

```
ch=64, ch_mult=[1,2,2,4,4,8], num_res_blocks=2
z_channels=256, n_embed=1024, embed_dim=256
attn at encoder.down.5 and decoder.up.5/up.4
```

### Encoder (32 GroupNorm layers)

```
conv_in:   Conv2d(3→64, 3×3)
down.0:    ResBlock(64,64) × 2 + Downsample      512→256
down.1:    ResBlock(64→128,128) × 2 + Downsample 256→128
down.2:    ResBlock(128,128) × 2 + Downsample    128→64
down.3:    ResBlock(128→256,256) × 2 + Downsample 64→32
down.4:    ResBlock(256,256) × 2 + Downsample     32→16
down.5:    ResBlock(256→512,512) × 2              16→16 (no downsample)
           AttnBlock(512) × 2
mid:       ResBlock(512,512) + AttnBlock(512) + ResBlock(512,512)
norm_out:  GroupNorm(32, 512) + SiLU
conv_out:  Conv2d(512→256, 3×3)
quant_conv: Conv2d(256→256, 1×1)
```

### VQ Codebook

Nearest-neighbour lookup over **1024 codes × 256 dims**.
Distances computed in FP32 for numerical stability.

```
post_quant_conv: Conv2d(256→256, 1×1)
```

### Decoder (48 GroupNorm layers)

```
conv_in:  Conv2d(256→512, 3×3)
mid:      ResBlock(512,512) + AttnBlock(512) + ResBlock(512,512)
up.5:     ResBlock(512,512) × 3 + AttnBlock(512) × 3 + Upsample   16→32
up.4:     ResBlock(512→256,256) × 3 + AttnBlock(256) × 3 + Upsample 32→64
up.3:     ResBlock(256,256) × 3 + Upsample                          64→128
up.2:     ResBlock(256→128,128) × 3 + Upsample                     128→256
up.1:     ResBlock(128,128) × 3 + Upsample                         256→512
up.0:     ResBlock(128→64,64) × 3                                   512→512 (no upsample)
norm_out: GroupNorm(32, 64) + SiLU
conv_out: Conv2d(64→3, 3×3)
```

Decoder processes up stages in reversed order: up[5] → up[4] → ... → up[0].

### GroupNorm+SiLU

All ResBlock and AttnBlock norms use `GroupNorm(32)`.  The Triton kernel:
- Pass 1: compute per-group mean/variance with FP32 accumulators
- Pass 2: normalise + apply scale/bias + optional SiLU fusion
- Eliminates the FP16→FP32→FP16 round-trip present in PyTorch's GroupNorm

Total GroupNorm layers: **80** (32 encoder + 48 decoder).

---

## Files

| File | Purpose |
|------|---------|
| `restoreformer_torch.py` | `RestoreFormerPlusPlusTorch` class + Triton kernels + CUDA graph runner |
| `benchmark_restoreformer.py` | 4-tier latency benchmark vs ORT baseline |
| `__init__.py` | Package marker |

The Triton GroupNorm+SiLU kernel is defined in `custom_kernels/triton_ops.py`
as `triton_group_norm_silu` (Kernel 5).

---

## Application Integration

Select **"Custom"** in *Settings → General → Providers Priority*.

RestoreFormer++ is then executed via `RestoreFormerPlusPlusTorch` in `run_RestoreFormerPlusPlus()`.

```python
# Internal call path (face_restorers.py):
from custom_kernels.restoreformer.restoreformer_torch import (
    RestoreFormerPlusPlusTorch, build_cuda_graph_runner
)

model  = RestoreFormerPlusPlusTorch.from_onnx(onnx_path).cuda().eval()
runner = build_cuda_graph_runner(model, inp_shape=(1, 3, 512, 512))
output = runner(face_image_f32_cuda)
# (1,3,512,512)f32 in → (1,3,512,512)f32 out
```

A CUDA graph is used in the application path because RestoreFormer++ has no dynamic
parameters (unlike CodeFormer's `fidelity_weight`).

---

## Weight Loading

Weights are loaded from the ONNX initializers.  Two strategies:

1. **Named params** — direct ONNX name → PyTorch parameter name matching.
   Covers all convolutions (`conv1`, `conv2`, `nin_shortcut`, `q`, `k`, `v`,
   `proj_out`, `conv_in`, `conv_out`, etc.) and the VQ embedding.

2. **GroupNorm scale/bias** — extracted by ONNX graph traversal
   (`InstanceNorm → Reshape → Mul(scale) → Add(bias)` pattern).
   All 80 GN pairs are anonymous in the ONNX initializers; assigned to model
   GroupNorm modules in forward-execution order.

---

## Numerical Accuracy

FP16 GroupNorm with FP32 accumulators in the Triton kernel provides accuracy
equivalent to the GroupNorm32 FP32 upcast used in the original LDM code.

VQ quantization distances are always computed in FP32 regardless of compute dtype.

The decoder output is correct for any given `post_quant_conv` input: isolated
decoder testing (ORT pqc + ORT encoder skips → PT decoder) gives `< 0.06%`
pixels with `|diff| > 0.1`.

The primary source of residual error vs ORT is **VQ code boundary sensitivity**:
~4/256 spatial positions in the 16×16 codebook grid may select a different
codebook entry due to epsilon-level differences in FP16 `quant_conv` output
between PyTorch cuDNN and ORT cuDNN. This affects ~0.5% of output pixels
with `MaxErr ≈ 4–5` at those patches. The same mismatch occurs with FP32
compute, confirming it is an inherent cuDNN implementation difference rather
than a precision issue. VQ code mismatches affect small localised patches
and are not visually noticeable in typical face restoration use cases.

## Running the Benchmark

```bat
.venv/Scripts/python custom_kernels/restoreformer/benchmark_restoreformer.py
```
