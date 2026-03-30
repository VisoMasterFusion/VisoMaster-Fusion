# face_landmark478 — Custom FP16 + CUDA-graph Kernel

FP16 PyTorch reimplementation of
`model_assets/face_landmarks_detector_Nx3x256x256.onnx` — the MobileNet-style
478-point face landmark detector used for `FaceLandmark478`.
Used by VisoMaster-Fusion when the *Custom* execution provider is selected.

## Architecture

| Component | Detail |
|-----------|--------|
| Input     | `(1, 3, 256, 256)` float32 |
| Outputs   | `Identity` `(1,1,1,1434)` — 478×(x,y,z) landmarks · `Identity_1` `(1,1,1,1)` — visibility · `Identity_2` `(1,1)` — face presence |
| Backbone  | 7-stage hierarchical depthwise-separable network |
| Stages    | 128→64→32→16→8→4→2 spatial; channels 16→32→64→128 (stages 4-7 stay at 128) |
| Params    | ~0.5 M (lightweight mobile model) |
| Activation| PReLU throughout (no BatchNorm — folded into conv weights at export) |

Only output index 0 — `Identity` `(1,1,1,1434)` — is used by
`detect_face_landmark_478()`, reshaped to `(1, 478, 3)` for (x, y, z) triples.

### Residual block types

**`_FirstBlock(C)`** — used only as block 1 of stage 1 (stem PReLU provides pre-activation):
```
PW_sq(C→C/2, 1×1) → PReLU(C/2) → DW(C/2, 3×3, groups=C/2) → PW_ex(C/2→C, 1×1) → Add(input)
```

**`_ResBlock(C)`** — used for all other residual blocks:
```
PReLU(C) → PW_sq(C→C/2, 1×1) → PReLU(C/2) → DW(C/2, 3×3, groups=C/2) → PW_ex(C/2→C, 1×1) → Add(PReLU_out)
```
The skip connection uses the leading PReLU output (pre-activation pattern).

### Stage transitions

**Stages 1→2, 2→3, 3→4** (channel doubling via zero-pad):
```
h_act  = stage_prelu(h)
main   = MaxPool(h_act, 2×2, stride=2)               # C channels
skip   = F.pad(Conv(h_act, C→C, 2×2, s=2), 0,C)     # zero-pad C→2C channels
# first block of next stage:
h      = PW_ex(DW(PReLU(main))) + skip               # expand C→2C
```

**Stages 4→5, 5→6, 6→7** (channels stay at 128):
```
skip   = MaxPool(h_act, 2×2, stride=2)               # 128 channels
conv   = Conv(h_act, 128→64, 2×2, s=2)
# first block of next stage:
h      = PW_ex(DW(PReLU(conv))) + skip               # expand 64→128
```

## Benchmark Results

**Hardware:** NVIDIA GeForce RTX 4090 · PyTorch 2.8.0+cu129 · CUDA 12.9 · ORT 1.22.0
**Conditions:** 500 iterations, 50 warm-up, batch=1

| Tier | Method | Time | vs ORT CUDA EP |
|------|--------|------|:--------------:|
| 0    | ORT FP32 CUDA EP (baseline) | 2.356 ms | 1.00× |
| 0b   | ORT TRT EP FP32 | 1.919 ms | 1.23× |
| 1    | PyTorch FP32 eager | 9.553 ms | 0.25× |
| 2    | PyTorch FP16 eager | 10.759 ms | 0.22× |
| **3** | **PyTorch FP16 + CUDA graph (Custom)** | **0.665 ms** | **3.54×** |
| **4** | **torch.compile default + FP16 + CUDA graph** | **0.509 ms** | **4.63×** |
| 4b   | torch.compile reduce-overhead | — *(skipped by default; set `LMK478_TORCH_COMPILE=1`)* | — |

> **Application uses Tier 3** (CUDA graph). Pass `torch_compile=True` to `build_cuda_graph_runner()` to activate Tier 4 (4.63×).

### Speed-up breakdown (tier 3)

- FP16 TensorCore depthwise + pointwise convolutions (~2× on Ampere/Ada)
- CUDA graph elimination of per-kernel CPU launch overhead for this deep 221-op network
- No BatchNorm overhead (already folded) — pure Conv+PReLU pipeline

## Accuracy

| Mode | Max |Δ| vs ORT FP32 | Mean |Δ| |
|------|------------------------|---------|
| FP16 + CUDA graph | 0.8682 (x/y/z in unnormalized landmark coords) ✓ | 0.1324 |

Output is visually correct on real faces — the difference is within the expected FP16
precision range for this coordinate scale. Accuracy check: ✓ OK (max error < 1.0).

## Files

| File | Purpose |
|------|---------|
| `face_landmark478_torch.py` | `FaceLandmark478Torch` model, `from_onnx()` loader, `build_cuda_graph_runner()` |
| `benchmark_face_landmark478.py` | Five-tier benchmark vs ORT + numerical accuracy check |

## Weight Loading

| Parameter type | Strategy |
|----------------|----------|
| All 256 initializers (named `const_fold_opt__NNNN`) | **Positional** — ONNX nodes traversed in topological order; Conv weight+bias and PReLU slope collected in encounter order and assigned sequentially to PyTorch parameters |
| No anonymous `onnx::` initializers | N/A |
| No BatchNorm | Already folded into conv weights/biases at TF→ONNX export |

## Running the Benchmark

```bash
# from repo root
.venv\Scripts\python custom_kernels\face_landmark478\benchmark_face_landmark478.py

# override paths / iterations
ONNX_PATH=model_assets/face_landmarks_detector_Nx3x256x256.onnx WARMUP=50 ITERS=500 \
    .venv\Scripts\python custom_kernels\face_landmark478\benchmark_face_landmark478.py
```

## Application Integration

When the inference provider is set to **Custom**, `detect_face_landmark_478()`
in `app/processors/face_landmark_detectors.py` calls `_get_landmark478_runner()`,
which lazy-loads `FaceLandmark478Torch` and wraps it in a CUDA graph runner.
The same pre-processing (face warp to 256 px, /255 normalisation) and
post-processing (reshape → `(1,478,3)` → inverse affine → 5-point conversion)
are applied as in the ORT path — the custom kernel is a pure drop-in replacement.
