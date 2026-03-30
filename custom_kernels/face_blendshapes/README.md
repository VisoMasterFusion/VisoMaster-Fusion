# face_blendshapes — Custom FP16 + CUDA-graph Kernel

FP16 PyTorch reimplementation of `model_assets/face_blendshapes_Nx146x2.onnx`, the
MLP-Mixer network that predicts 52 ARKit blendshape coefficients from 146 selected
2-D facial landmarks. Used by VisoMaster-Fusion when the *Custom* execution provider
is selected.

## Architecture

```
Input:  (N, 146, 2)  float32   ← 146 selected 2-D landmarks (x, y)
Output: (N, 52)      float32   ← blendshape weights ∈ (0, 1)
```

### Preprocessing

1. Subtract per-sample centroid (mean over 146 landmarks)
2. Divide by mean L2 norm of centred landmarks × 0.5
3. Reshape to NCHW `(N, 146, 1, 2)` for Conv2d

### Embedding

| Step | Op | Input → Output |
|------|----|---------------|
| 1 | Conv2d(146→96, 1×1) | `(N, 146, 1, 2)` → `(N, 96, 1, 2)` — landmark channel mix |
| 2 | Transpose [0,3,2,1] | `(N, 96, 1, 2)` → `(N, 2, 1, 96)` |
| 3 | Conv2d(2→64, 1×1)   | `(N, 2, 1, 96)` → `(N, 64, 1, 96)` — coord embedding |
| 4 | Prepend CLS token   | `(N, 64, 1, 96)` → `(N, 64, 1, 97)` — learnable class token |

### 4× MixerBlock

Working tensor: `(N, C=64, H=1, W=97)` throughout.

```
┌──────────────────────────────────────────┐
│  LayerNorm(C=64) per token               │  γ shape (64,), β = 0
│  Token mixing                            │
│    permute → (N, 97, 1, 64)             │
│    Conv2d(97→384, 1×1) → ReLU           │
│    Conv2d(384→97, 1×1)                  │
│    permute back → (N, 64, 1, 97)        │
│  + residual                              │
├──────────────────────────────────────────┤
│  LayerNorm(C=64) per token               │
│  Channel mixing                          │
│    Conv2d(64→256, 1×1) → ReLU           │
│    Conv2d(256→64, 1×1)                  │
│  + residual                              │
└──────────────────────────────────────────┘
```

### Head

```
x[:, :, :, 0:1]   ← CLS token at W=0  →  (N, 64, 1, 1)
Conv2d(64→52, 1×1) → Sigmoid → reshape  →  (N, 52)
```

### LayerNorm detail

The ONNX graph implements LN manually (TF export artefact):
- Normalise over channel dim C=64 for each of the 97 token positions
- Learnable scale γ (64,) per block; **no bias** (β = 0 everywhere)
- Epsilon = `1.013279e-06` (from `Transpose__290_0` initialiser)

## Parameters

| Component | Count |
|-----------|-------|
| Embedding Conv1 (146→96) | 96×146 + 96 = 14,112 |
| Embedding Conv2 (2→64)   | 64×2 + 64 = 192 |
| CLS token                | 64 |
| 4× token MLP (97→384→97) | 4×(97×384 + 384 + 384×97 + 97) = 298,048 |
| 4× channel MLP (64→256→64) | 4×(64×256 + 256 + 256×64 + 64) = 132,480 |
| 8× LN scale (64,)        | 512 |
| Head Conv (64→52)        | 64×52 + 52 = 3,380 |
| **Total**                | **~448 K** |

## Benchmark Results

**Hardware:** NVIDIA GeForce RTX 4090 · PyTorch 2.8.0+cu129 · CUDA 12.9 · ORT 1.22.0
**Conditions:** 500 iterations, 50 warm-up, batch=1

| Tier | Method | Time | vs ORT CUDA EP |
|------|--------|------|:--------------:|
| 0    | ORT FP32 CUDA EP (baseline) | 1.432 ms | 1.00× |
| 0b   | ORT TRT EP FP32 | 0.408 ms | 3.51× |
| 1    | PyTorch FP32 eager | 2.747 ms | 0.52× |
| 2    | PyTorch FP16 eager | 2.788 ms | 0.51× |
| **3** | **PyTorch FP16 + CUDA graph (Custom)** | **0.170 ms** | **8.41×** |
| **4** | **torch.compile default + FP16 + CUDA graph** | **0.110 ms** | **13.01×** |
| 4b   | torch.compile reduce-overhead | — *(skipped by default; set `BLENDSHAPES_TORCH_COMPILE=1`)* | — |

> **Application uses Tier 3** (CUDA graph). Pass `torch_compile=True` to `build_cuda_graph_runner()` to activate Tier 4 (13.01×).

Note: FaceBlendShapes is a tiny model (~448 K params, ~0.1 ms compute). The CUDA graph
eliminates Python/driver overhead which dominates at this scale.

## Accuracy

| Mode | Max |Δ| vs ORT FP32 | Mean |Δ| |
|------|------------------------|----------|
| FP32 eager | ~9.5e-7 (rounding only) | ~1.8e-7 |
| FP16 + CUDA graph | 0.00179 (blendshape coefficients in [0,1]) ✓ | 0.000281 |

## Files

| File | Purpose |
|------|---------|
| `face_blendshapes_torch.py` | `FaceBlendShapesTorch` model, `from_onnx()` loader, `build_cuda_graph_runner()` |
| `benchmark_face_blendshapes.py` | Five-tier benchmark vs ORT + numerical accuracy check |

## Weight Loading

All 59 ONNX initialisers are loaded:

- **19 Conv weight/bias pairs** — loaded positionally in ONNX node-traversal order (emb1, emb2, then 4 blocks × {tok_up, tok_down, ch_up, ch_down}, then head)
- **8 LN scale vectors** — loaded by name (`const_fold_opt__452`, `__396`, `__435`, `__434`, `__416`, `__415`, `__384`, `__386`)
- **1 CLS token** — loaded by name (`tile_Constant_4_output_0`)
- **31 shape/constant tensors** — skipped (reshape constants, LN epsilon, etc.)

## Running the Benchmark

```bash
# from repo root
.venv\Scripts\python custom_kernels\face_blendshapes\benchmark_face_blendshapes.py
```

Optional env vars: `ONNX_PATH`, `WARMUP` (default 50), `ITERS` (default 500).

## Usage

```python
from custom_kernels.face_blendshapes.face_blendshapes_torch import (
    FaceBlendShapesTorch,
    build_cuda_graph_runner,
)

model = (FaceBlendShapesTorch.from_onnx("model_assets/face_blendshapes_Nx146x2.onnx")
         .cuda().eval())
runner = build_cuda_graph_runner(model)   # captures CUDA graph once

# At inference:
landmarks_146x2 = ...  # torch.Tensor (1, 146, 2) float32 CUDA
blendshapes = runner(landmarks_146x2)    # (1, 52) float32
```

## Application Integration

When the inference provider is set to **Custom**, the blendshapes inference path
calls `_get_blendshapes_runner()`, which lazy-loads `FaceBlendShapesTorch` and
wraps it in a CUDA graph runner. The custom kernel is a pure drop-in replacement
for the ORT path.
