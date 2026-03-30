# Custom Kernels — det_106 (106-Point Face Landmark Detector)

FP16 PyTorch reimplementation of `model_assets/2d106det.onnx` — the
MobileNetV1-style 106-point face landmark detector.  Used as the
**FaceLandmark106** detector in the application when the *Custom* execution
provider is selected.

## Models

| Model | Input | Output |
|-------|-------|--------|
| 2d106det | `(N,3,192,192)` float32 in **[0,255]** | `(N,212)` — 106 × (x, y) in model space |

Pre-processing `(x − 127.5) × 0.0078125` → **[−1, 1]** is baked into `forward`.

## Benchmark Results

**Hardware:** NVIDIA GeForce RTX 4090 · PyTorch 2.8.0+cu129 · CUDA 12.9 · ORT 1.22.0
**Conditions:** 500 iterations, 50 warm-up, input 192×192

| Tier | Method | ms | vs ORT CUDA EP |
|------|--------|----|:--------------:|
| 0    | ORT FP32 CUDA EP (baseline) | 0.800 ms | 1.00× |
| 0b   | ORT TensorRT EP FP32 | 0.442 ms | 1.81× |
| 1    | PyTorch FP32 | 1.274 ms | 0.63× |
| 2    | PyTorch FP16 | 1.287 ms | 0.62× |
| **3** | **PT FP16 + CUDA graph (Custom)** | **0.219 ms** | **3.65×** |
| **4** | **torch.compile default + FP16 + CUDA graph** | **0.181 ms** | **4.42×** |
| 4b   | torch.compile reduce-overhead | — *(skipped by default; set `DET106_TORCH_COMPILE=1`)* | — |

> **Application uses Tier 3** (FP16 + CUDA graph). Pass `torch_compile=True` to `build_cuda_graph_runner` to activate Tier 4 (4.42×).

**Accuracy (Tier 3 vs ORT FP32):** max|Δ| = 1.0e-03, mean|Δ| = 3.75e-04 (landmarks in model space [−1, 1]).

> **Application uses PT FP16 + CUDA graph** (single captured graph; fixed 192×192 input).
> If CUDA graph capture fails, falls back to FP16 eager.

### Speed-up Source

The large CUDA-graph gain (3.16x) on this small model reflects that kernel
launch overhead dominates for a fast MobileNetV1-style backbone.  Speed-up
comes from:

1. **FP16** — cuDNN dispatches depthwise and pointwise Conv2d to TensorCore
   kernels on Ampere/Ada GPUs.
2. **CUDA graph** — eliminates per-kernel CPU launch overhead, which is
   proportionally very large for this tiny backbone (~0.2 ms per pass).
3. **No BatchNorm overhead** — already folded into conv weights/biases at
   TF→ONNX export, leaving a pure Conv+PReLU pipeline.

No Triton kernels are used.

## Accuracy

FP16 CUDA graph vs ORT FP32 (192×192 input):

| Output | Max Absolute Error | Status |
|--------|--------------------|--------|
| Landmarks (x/y in model space [−1, 1]) | 5.0e-4 | Pass |

## Architecture Notes

Reverse-engineered from `model_assets/2d106det.onnx`:

```
MobileNetV1-style: DW+PW sequence → 192→96→48→24→12→12(×7)→6→6→3 spatial
Activations: PReLU throughout; no BatchNorm (folded into conv weights at export)
```

ONNX graph: 60 nodes — Sub(1) + Mul(1) + Conv(28) + PReLU(28) + Flatten(1) + Gemm(1).
All 88 initializers are named (`_v_XXX` for Conv, `conv_X_relu_gamma` /
`conv_X_dw_relu_gamma` for PReLU, `scalar_op1/2` for preprocessing,
`fc1_weight/bias` for Gemm).

### Layer sequence

```
Conv(3→16, 3×3, s=2, p=1)  + PReLU   → (N,  16, 96, 96)   stem
DW(16,  3×3, s=1, p=1, g=16) + PReLU   → (N,  16, 96, 96)
PW(16→32,  1×1)              + PReLU   → (N,  32, 96, 96)
DW(32,  3×3, s=2, p=1, g=32) + PReLU   → (N,  32, 48, 48)
PW(32→64,  1×1)              + PReLU   → (N,  64, 48, 48)
DW(64,  3×3, s=2, p=1, g=64) + PReLU   → (N,  64, 24, 24)
PW(64→128, 1×1)              + PReLU   → (N, 128, 24, 24)
DW(128, 3×3, s=2, p=1, g=128)+ PReLU   → (N, 128, 12, 12)
PW(128→256,1×1)              + PReLU   → (N, 256, 12, 12)
× 7:
  DW(256, 3×3, s=1, p=1, g=256)+ PReLU → (N, 256, 12, 12)
  PW(256→256,1×1)               + PReLU → (N, 256, 12, 12)
DW(256, 3×3, s=2, p=1, g=256)+ PReLU   → (N, 256,  6,  6)
PW(256→512,1×1)              + PReLU   → (N, 512,  6,  6)
DW(512, 3×3, s=1, p=1, g=512)+ PReLU   → (N, 512,  6,  6)
PW(512→512,1×1)              + PReLU   → (N, 512,  6,  6)
Conv(512→64, 3×3, s=2, p=1) + PReLU   → (N,  64,  3,  3)
Flatten                                 → (N, 576)
Linear(576→212)                         → (N, 212)
```

### Post-processing (in `detect_face_landmark_106`)

```python
pred = out[0].reshape(-1, 2)             # (106, 2)  x/y in model space [-1, 1]
pred[:, :2] = (pred[:, :2] + 1) * 96.0  # → pixel coords in 192×192 crop
pred = faceutil.trans_points(pred, IM)   # inverse-affine back to frame coords
```

### Weight Loading

| Parameter type | Strategy |
|----------------|----------|
| 28 Conv layers (weight + bias) | **Positional** — ONNX nodes visited in topological order; every `Conv` node collected and assigned sequentially |
| 28 PReLU layers (slope) | **Positional** — every `PRelu` node collected and assigned sequentially; ONNX shape `(C,1,1)` → reshaped to `(C,)` for PyTorch `nn.PReLU` |
| Gemm (fc) weight + bias | **By name** — inspected from the Gemm node's `input[1]` and `input[2]` |
| No BatchNorm | Already folded into conv weights/biases at TF→ONNX export |

**Preprocessing note:** The ORT path applies a no-op `v2.Normalize(mean=0, std=1)`
so the ONNX model receives raw float32 values in [0, 255].  `Det106Torch.forward()`
therefore applies `(x − 127.5) × 0.0078125` internally (mirroring ONNX nodes 0–1)
before the convolutional stages.

## Files

| File | Purpose |
|------|---------|
| `det_106_torch.py` | `Det106Torch` model, `from_onnx()` loader, `build_cuda_graph_runner()` |
| `benchmark_det_106.py` | 4-tier latency benchmark vs ORT + numerical accuracy check |

## Usage

```python
from custom_kernels.det_106.det_106_torch import Det106Torch, build_cuda_graph_runner
import torch

# Load model (one-time)
model  = Det106Torch.from_onnx("model_assets/2d106det.onnx").cuda().eval()
runner = build_cuda_graph_runner(model)   # single captured CUDA graph

# Per-frame call
with torch.no_grad():
    out = runner(face_tensor)  # (N,3,192,192) float32 → (N,212) float32
landmarks = out[0].reshape(-1, 2)
# Same post-processing as ORT path: (pred + 1) * 96.0 → inverse-affine
```

Select **"Custom"** in *Settings → General → Providers Priority* to activate.

The custom kernel is a pure drop-in replacement: the same pre-processing
(face warp to 192 px, float32 cast) and post-processing apply as in the ORT path.

## Running the Benchmark

```bat
.venv\Scripts\python custom_kernels\det_106\benchmark_det_106.py

REM Override paths / iterations:
set ONNX_PATH=model_assets/2d106det.onnx
set WARMUP=50
set ITERS=500
.venv\Scripts\python custom_kernels\det_106\benchmark_det_106.py
```
