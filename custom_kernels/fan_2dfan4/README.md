# fan_2dfan4 — Custom FP16 + CUDA-graph Kernel

FP16 PyTorch reimplementation of `model_assets/2dfan4.onnx` — the
4-stacked hourglass 68-point face landmark detector used for `FaceLandmark68`.
Used by VisoMaster-Fusion when the *Custom* execution provider is selected.

## Architecture

| Component | Detail |
|-----------|--------|
| Input     | `(1, 3, 256, 256)` float32 — face crop in [0, 1] × 255 |
| Outputs   | `(1, 68, 3)` landmarks\_xyscore · `(1, 68, 64, 64)` heatmaps |
| Backbone  | 4-stacked hourglass network (Newell et al. / Bulat 2-D FAN) |
| Block     | Pre-activation concatenation bottleneck — 3 convs, outputs cat'd |
| Decoder   | Nearest-neighbour ×2 upsample + skip-connection add |
| Params    | ~98 M |

Only output index 0 — `(1, 68, 3)` — is consumed by `detect_face_landmark_68()`.
It contains 68 (x, y, score) landmark triples; x/y are in pixel-centre coordinates
`[0.5, 63.5]` within the 64×64 heatmap (i.e. must be divided by 64 and scaled by 256
to recover image-space pixel coordinates).

### Pre-activation concatenation bottleneck (_FanBlock)

```
BN → ReLU → Conv(3×3) → BN → ReLU → Conv(3×3) → BN → ReLU → Conv(3×3)
                 ↓ c1            ↓ c2            ↓ c3
                       cat([c1, c2, c3]) + shortcut
```

Shortcut is either identity (`in_ch == c1+c2+c3`) or `BN → ReLU → Conv(1×1)`.
The BN in the shortcut path (`downsample[0]`) **shares `running_mean/running_var`**
with `bn1` — fixed up in `from_onnx()` after the named-weight load.

### Recursive hourglass (_HourGlass, depth=4)

```
b1_4(x) ─────────────────────────────────────── skip4
  pool → b2_4 →
    b1_3(x) ──────────────────────── skip3
      pool → b2_3 →
        b1_2(x) ──────────────── skip2
          pool → b2_2 →
            b1_1(x) ────── skip1
              pool → b2_1 → b2_plus_1 → b3_1 → ×2
            skip1 + ×2 → b3_2 → ×2
          skip2 + ×2 → b3_3 → ×2
        skip3 + ×2 → b3_4 → ×2
      skip4 + ×2
```

### 4-stack inter-stack formula

```
interim = ReLU(inter_conv_i(top_m_i(hg_i(x))))
heatmap_i = l_i(interim)
x_next = x + bl_i(interim) + al_i(heatmap_i)   (stacks 0–2 only)
```

### Post-processing (baked into `forward()`)

```
1. scores = ReduceMax over H,W → (1, 68)
2. peak (py, px) = ArgMax over flattened spatial dim
3. L2-distance mask: keep pixels within radius 6.4 of peak (integer grid)
4. masked heatmap clipped to ≥ 0
5. weighted centroid: x_idx/y_idx = arange(64) + 0.5  (half-pixel centres)
6. output = stack(x_centroid, y_centroid, score) → (1, 68, 3)
```

## Benchmark Results

**Hardware:** NVIDIA GeForce RTX 4090 · PyTorch 2.8.0+cu129 · CUDA 12.9 · ORT 1.22.0
**Conditions:** 50 iterations, 10 warm-up, batch=1

| Tier | Method | Time | vs ORT CUDA EP |
|------|--------|------|:--------------:|
| 0    | ORT FP32 CUDA EP (baseline) | 10.293 ms | 1.00× |
| 0b   | ORT TensorRT EP FP32 | 4.638 ms | 2.22× |
| 1    | PyTorch FP32 eager | 15.594 ms | 0.66× |
| 2    | PyTorch FP16 eager | 17.478 ms | 0.59× |
| **3** | **PyTorch FP16 + CUDA graph (Custom)** | **3.521 ms** | **2.92×** |
| **4** | **torch.compile default + FP16 + CUDA graph** | **2.335 ms** | **4.41×** |
| 4b   | torch.compile reduce-overhead | — *(skipped by default; set `FAN_TORCH_COMPILE=1`)* | — |

> **Application uses Tier 3** (CUDA graph). Pass `torch_compile=True` to `build_cuda_graph_runner()` to activate Tier 4 (4.41×).

> **Tier 4 — torch.compile:** Pass `torch_compile=True` to `build_cuda_graph_runner()` to
> enable `torch.compile` before CUDA graph capture. This triggers an additional compilation
> warmup pass via `custom_kernels.compile_utils.apply_torch_compile`. Benchmark results
> will vary by GPU and PyTorch version.

### Speed-up breakdown (tier 3)

- FP16 TensorCore convolutions across the deep hourglass stacks (~2× on Ampere/Ada)
- CUDA graph elimination of per-kernel CPU launch overhead for this large 400+ op network

## Accuracy

| Mode | Max |Δ| xy vs ORT FP32 | Mean |Δ| |
|------|------------------------|----------|
| FP16 + CUDA graph | 0.2254 px (coordinate space [0.5..63.5]) | 0.0053 px |

> Coordinates are in [0.5, 63.5] heatmap-pixel space; max|Δ|=0.2254 is well sub-pixel
> in 256-px image space. Accuracy check: ✓ OK (< 1.0 pixel).

## Files

| File | Purpose |
|------|---------|
| `fan_2dfan4_torch.py` | `FAN2dfan4` model, `from_onnx()` loader, `build_cuda_graph_runner()` |
| `benchmark_fan_2dfan4.py` | Five-tier benchmark vs ORT + numerical accuracy check |

## Weight Loading

| Parameter type | Strategy |
|----------------|----------|
| ~910 named initializers (all BN, Conv, head weights) | **Named** — loaded via `load_state_dict(strict=False)` |
| 2 shared BN running stats (conv2/conv4 downsample) | **Copied** — `downsample[0].running_mean/var` ← `bn1.running_mean/var` after named load |
| 10 anonymous Conv weights (stem + 4×inter\_conv, weight+bias) | **Positional** — anonymous `onnx::Conv_NNN` in ONNX node topological order |

## Running the Benchmark

```bash
# from repo root
.venv\Scripts\python custom_kernels\fan_2dfan4\benchmark_fan_2dfan4.py

# override paths / iterations
ONNX_PATH=model_assets/2dfan4.onnx WARMUP=50 ITERS=500 \
    .venv\Scripts\python custom_kernels\fan_2dfan4\benchmark_fan_2dfan4.py
```

## Application Integration

When the inference provider is set to **Custom**, `detect_face_landmark_68()`
in `app/processors/face_landmark_detectors.py` calls `_get_fan2dfan4_runner()`,
which lazy-loads `FAN2dfan4` and wraps it in a CUDA graph runner.
The same pre-processing (face warp to 256 px, /255 normalisation) and
post-processing (x/y ÷ 64 × 256 → inverse affine → 5-point conversion) are applied
as in the ORT path — the custom kernel is a pure drop-in replacement.
