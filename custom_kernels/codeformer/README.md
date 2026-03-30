# CodeFormer Custom Kernel

FP16 PyTorch reimplementation of the **CodeFormer** face restoration ONNX model,
with Triton fused GroupNorm+SiLU kernels and optional CUDA graph acceleration.
Used by VisoMaster-Fusion when the *Custom* execution provider is selected.

## Model

| Model | Input | Output |
|-------|-------|--------|
| CodeFormer | (1,3,512,512) f32 | (1,3,512,512) f32 |

## Benchmark Results (RTX 4090, CUDA 12.9, PyTorch 2.8+cu129, ORT 1.22.0, Triton 3.4.0)

**Environment:** NVIDIA GeForce RTX 4090 · PyTorch 2.8.0+cu129 · CUDA 12.9 · Triton 3.4.0 · ORT 1.22.0
**Method:** 50 iterations, 10 warm-up passes · Input/Output: (1,3,512,512) float32

| Tier | Method | Latency | vs ORT CUDA EP |
|------|--------|--------:|---------------:|
| 0  | ORT FP32 CUDA EP (baseline) | 20.51 ms | 1.00× |
| 0b | ORT TRT EP (opt=5) | 10.30 ms | 1.99× |
| 1  | PyTorch FP32 pure ops | 31.74 ms | 0.65× |
| 2  | PyTorch FP16 + Triton GroupNorm+SiLU | 19.43 ms | 1.06× |
| **3** | **FP16 + Triton + CUDA graph (Custom)** | **13.30 ms** | **1.54×** |
| 4  | FP16 + Triton + SDPA4D + GEMM + CUDA graph | 13.24 ms | 1.55× |
| 5  | FP16 + Triton + SDPA4D + GEMM + NHWC + CUDA graph | 17.76 ms | 1.15× |
| **6** | **torch.compile default + FP16 + Triton + CUDA graph** | **11.25 ms** | **1.82×** |
| 4b | torch.compile reduce-overhead (no separate CUDA graph) | 11.76 ms | 1.74× |

The CUDA-graph path (Tier 4) is **1.55x faster** than ORT CUDA EP.
Tier 6 adds torch.compile(default) + CUDA graph for **1.82× vs ORT** (1.18× vs Tier 3).
Both `default+CUDA graph` and `reduce-overhead` work for CodeFormer; default is slightly faster.

> **Application uses Tier 3** (CUDA graph; CUDA graph is rebuilt when `fidelity_weight` changes).
> Pass `torch_compile=True` to `build_cuda_graph_runner` to activate Tier 6.

### Kernel Priority Chain

1. **Triton `triton_group_norm_silu`** — Windows-friendly, no MSVC required (preferred)
2. **Pure PyTorch** — automatic fallback (still FP16)

---

## Architecture

Config reverse-engineered from `model_assets/codeformer_fp16.onnx`:

### Encoder (25 blocks)

```
[0]:  Conv2d(3→64, 3×3)
[1]-[2]:   ResBlock(64,64)
[3]:  Downsample(64)               512→256
[4]-[5]:   ResBlock(64→128, 128)   [5] skip → fuse_256
[6]:  Downsample(128)              256→128
[7]-[8]:   ResBlock(128,128)       [8] skip → fuse_128
[9]:  Downsample(128)              128→64
[10]-[11]: ResBlock(128→256, 256)  [11] skip → fuse_64
[12]: Downsample(256)              64→32
[13]-[14]: ResBlock(256,256)       [14] skip → fuse_32
[15]: Downsample(256)              32→16
[16]: ResBlock(256→512)
[17,19,21]: AttnBlock(512)
[18,20,22]: ResBlock(512,512)
[23]: GroupNorm(512)+SiLU          standalone
[24]: Conv2d(512→256, 3×3)         quant_conv → (b,256,16,16)
```

### VQ Codebook

Nearest-neighbour lookup over **1024 codes × 256 dims**.

### Transformer (9 layers)

```
d_model=512, nhead=8, ffn_dim=1024, activation=GELU
Input: z reshaped to (seq=256, batch, 256) → Linear(256→512) → 9×TransformerEncoderLayer
Output: logits (256, batch, 1024) → softmax → weighted sum over codebook
```

AdaIN normalises soft-VQ features to match encoder z channel statistics before
passing to the generator.

### Generator (25 blocks)

```
[0]:  Conv2d(256→512, 3×3)
[1]-[7]:   ResBlock+AttnBlock stack (512ch, 16×16 spatial)
[8]:  Upsample(512)                16→32
[9]:  ResBlock(512→256)  → SFT fuse_32
[10]: ResBlock(256,256)
[11]: Upsample(256)                32→64
[12]: ResBlock(256,256)  → SFT fuse_64
[13]: ResBlock(256,256)
[14]: Upsample(256)                64→128
[15]: ResBlock(256→128)  → SFT fuse_128
[16]: ResBlock(128,128)
[17]: Upsample(128)                128→256
[18]: ResBlock(128,128)  → SFT fuse_256
[19]: ResBlock(128,128)
[20]: Upsample(128)                256→512
[21]-[22]: ResBlock(128→64, 64)
[23]: GroupNorm(64)+SiLU           standalone
[24]: Conv2d(64→3, 3×3)
```

### SFT Skip Connections (fuse_convs_dict)

Applied at scales 32, 64, 128, 256.  For each scale:

```python
fuse_in   = cat([enc_skip, gen_feat], dim=1)
enc_out   = ResBlock(fuse_in)        # with GroupNorm
scale     = CNN(enc_out)             # Conv→LeakyReLU→Conv
shift     = CNN(enc_out)
gen_feat  = gen_feat + w * (gen_feat * scale + shift)
```

`w` is the **fidelity_weight** slider (0 = pure generation, 1 = full encoder conditioning).

### GroupNorm+SiLU

All ResBlock and AttnBlock norms use `GroupNorm(32)`.  The Triton kernel:
- Pass 1: compute per-group mean/variance with FP32 accumulators
- Pass 2: normalise + apply scale/bias + optional SiLU fusion
- Eliminates the FP16→FP32→FP16 round-trip present in PyTorch's GroupNorm

Total GroupNorm layers: **72** (32 encoder + 40 generator including fuse encode_enc).

---

## Files

| File | Purpose |
|------|---------|
| `codeformer_torch.py` | `CodeFormerTorch` class + Triton kernels + CUDA graph runner |
| `benchmark_codeformer.py` | 4-tier latency benchmark vs ORT baseline |
| `__init__.py` | Package marker |

The Triton GroupNorm+SiLU kernel is defined in `custom_kernels/triton_ops.py`
as `triton_group_norm_silu` (Kernel 5).

---

## Application Integration

Select **"Custom"** in *Settings → General → Providers Priority*.

CodeFormer is then executed via `CodeFormerTorch` in `run_codeformer()`.

```python
# Internal call path (face_restorers.py):
from custom_kernels.codeformer.codeformer_torch import CodeFormerTorch

model = CodeFormerTorch.from_onnx(onnx_path).cuda().eval()
output = model(face_image_f32_cuda, fidelity_weight=0.5)
# (1,3,512,512)f32 in → (1,3,512,512)f32 out
```

No CUDA graph is used in the application path because `fidelity_weight` changes
dynamically per session.  For advanced single-`w` scenarios the
`build_cuda_graph_runner` helper is available but not wired by default.

---

## Weight Loading

Weights are loaded from the ONNX initializers.  Three strategies:

1. **Named params** — direct ONNX name → PyTorch parameter name matching (encoder
   convolutions, transformer norms, fuse_convs, etc.)

2. **GroupNorm scale/bias** — extracted by ONNX graph traversal (InstanceNorm →
   Reshape → `Mul(scale)` → `Add(bias)` pattern).  Assigned to model GroupNorm
   modules in forward-execution order.

3. **Transformer QKV + FFN weights** — stored as anonymous `onnx::MatMul_XXXX`
   initializers with stride-25 index pattern.  Q/K/V are recombined into
   `in_proj_weight` for `nn.MultiheadAttention`.

---

## Numerical Accuracy

FP16 GroupNorm with FP32 accumulators in the Triton kernel provides accuracy
equivalent to the GroupNorm32 FP32 upcast used in the original LDM code.

Multi-seed accuracy vs ORT FP16 (w=0.5):

| Seed | MaxErr | P99 | P99.9 | >0.1 pixels |
|------|-------:|----:|------:|------------:|
| 42 | 0.51 | 0.084 | 0.165 | 0.68% |
| 123 | 0.35 | 0.056 | 0.109 | 0.21% |
| 999 | 0.85 | 0.203 | 0.371 | 4.24% |

Seed 999 shows higher error due to gen[21–22] intermediate activations reaching
~12,000 in FP16 (inherent overflow boundary for this model at this input).
For typical face inputs this worst case is rare.

## Running the Benchmark

```bat
.venv/Scripts/python custom_kernels/codeformer/benchmark_codeformer.py
```
