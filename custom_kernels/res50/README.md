# Custom Kernels — res50 (FaceLandmark5 / RetinaFace)

FP16 PyTorch reimplementation of the `res50.onnx` face landmark detector with
CUDA graph capture.  Used as the **FaceLandmark5** (5-point facial landmark)
model in the application.

Model: **res50** `(1,3,512,512)f32 → conf(1,10752,2)f32 + landmarks(1,10752,10)f32`

---

## Benchmark Results

**Hardware:** NVIDIA GeForce RTX 4090 · PyTorch 2.8.0+cu129 · CUDA 12.9 · ORT 1.22.0
**Conditions:** 50 iterations, 10 warm-up, input 512×512

| Tier | Method | ms | vs ORT CUDA EP |
|------|--------|----|:--------------:|
| 0 | ORT FP32 CUDA EP (baseline) | 3.57 ms | 1.00× |
| 0b | ORT TRT EP FP32 | 2.23 ms | 1.60× |
| 1 | PyTorch FP32 | 3.80 ms | 0.94× |
| 2 | PyTorch FP16 | 4.50 ms | 0.79× |
| **3** | **PT FP16 + CUDA graph (Custom)** | **1.49 ms** | **2.39×** |
| **4** | **torch.compile default + FP16 + CUDA graph** | **0.98 ms** | **3.63×** |
| 4b | torch.compile reduce-overhead | — *(skipped by default; set `RES50_TORCH_COMPILE=1`)* | — |

> **Application uses Tier 3** (CUDA graph). Pass `torch_compile=True` to `build_cuda_graph_runner` to activate Tier 4 (3.63×).

Run `benchmark_res50.py` to measure on your hardware.

### Speed-up Source

Unlike the face-restoration models (CodeFormer, RestoreFormer++, GFPGAN)
which contain GroupNorm layers, `res50` is a **pure Conv2d architecture**
(BatchNorm folded into Conv at export). All speed-up comes from:

1. **FP16** — cuDNN dispatches Conv2d on FP16 weights to TensorCore GEMM
   kernels (~2× throughput vs FP32 on Ampere/Ada GPUs).
2. **CUDA graph** — eliminates Python/CUDA kernel-launch overhead
   (~15–30% additional gain on fast backbones).

No Triton kernels are used (no GroupNorm layers to fuse).

---

## Architecture

Reverse-engineered from `model_assets/res50.onnx`:

```
RetinaFace with ResNet-50 backbone (BN folded) + FPN + SSH heads
Input: 512 × 512 RGB, mean-subtracted [104, 117, 123]
```

### ResNet-50 Backbone (BN folded into Conv bias)

```
stem:   Conv(3→64, 7×7, s=2) + ReLU + MaxPool(3, s=2)
layer1: 3 × Bottleneck(64 → 64→256)           512→256 → 128×128
layer2: 4 × Bottleneck(256 → 128→512, s=2)    128×128 → 64×64   → C3
layer3: 6 × Bottleneck(512 → 256→1024, s=2)    64×64  → 32×32   → C4
layer4: 3 × Bottleneck(1024 → 512→2048, s=2)   32×32  → 16×16   → C5
```

Bottleneck block: `ReLU(conv1(1×1)) → ReLU(conv2(3×3,s)) → conv3(1×1)` + residual add + ReLU.
First block per layer uses a projection shortcut (`downsample`: 1×1 conv, same stride).

### Feature Pyramid Network (FPN)

```
output1: Conv(512→256, 1×1)  + LeakyReLU(0.1) on C3  → p3 (64×64, 256ch)
output2: Conv(1024→256, 1×1) + LeakyReLU(0.1) on C4  → p4 (32×32, 256ch)
output3: Conv(2048→256, 1×1) + LeakyReLU(0.1) on C5  → p5 (16×16, 256ch)

Top-down:
  merge2: (p4 + upsample(p5)) → Conv(256→256, 3×3) + LeakyReLU → p4m
  merge1: (p3 + upsample(p4m)) → Conv(256→256, 3×3) + LeakyReLU → p3m

SSH inputs: [p3m (64×64), p4m (32×32), p5 (16×16)]
```

### SSH Module (× 3, for each FPN level)

```
Input: 256ch
  Branch A: conv3X3  (256→128, 3×3)              → 128ch [no activation]
  Branch B: conv5X5_1 (256→64, 3×3)+LeakyReLU
            conv5X5_2 (64→64, 3×3)               →  64ch [no activation]
  Branch C: conv7X7_2 (64→64, 3×3)+LeakyReLU
            conv7x7_3 (64→64, 3×3)               →  64ch [no activation]
  Output:   ReLU(cat([A, B, C]))                 → 256ch
```

### Detection Heads (× 3 per type, one per FPN level)

```
ClassHead.{0,1,2}:    conv1x1(256→4, 1×1) → permute → reshape → softmax → conf
BboxHead.{0,1,2}:     conv1x1(256→8, 1×1) → permute → reshape            → loc  (not returned)
LandmarkHead.{0,1,2}: conv1x1(256→20,1×1) → permute → reshape            → landmarks
```

Anchor counts: 64×64×2 + 32×32×2 + 16×16×2 = 8192 + 2048 + 512 = **10752 anchors**.

---

## Files

| File | Purpose |
|------|---------|
| `res50_torch.py` | FP16 PyTorch RetinaFace + CUDA graph runner |
| `benchmark_res50.py` | 4-tier latency benchmark vs ORT baseline |
| `__init__.py` | Package marker |

---

## Application Integration

Select **"Custom"** in *Settings → General → Providers Priority*.

FaceLandmark5 is then executed via `Res50Torch` (CUDA graph) in `detect_face_landmark_5()`.

```python
# Internal call path (face_landmark_detectors.py):
from custom_kernels.res50.res50_torch import Res50Torch, build_cuda_graph_runner

model  = Res50Torch.from_onnx(onnx_path).cuda().eval()
runner = build_cuda_graph_runner(model)

conf, landmarks = runner(face_image_f32_cuda)
# conf:      (1,10752,2) float32 — softmax class scores
# landmarks: (1,10752,10) float32 — 5-point landmark regression
```

Post-processing (anchor decoding, threshold, best-match selection) is handled
entirely in the existing `detect_face_landmark_5()` code.

---

## Weight Loading

All weights are loaded from ONNX initializers.  Two strategies:

1. **Anonymous params** (73 Conv2d: backbone + FPN + SSH) — positional match.
   ONNX Conv nodes are iterated in topological order; their weight/bias
   initializers (named `onnx::Conv_XXXX`) are assigned sequentially to the
   73 PyTorch Conv2d modules in the same forward-execution order.

2. **Named params** (9 Conv2d: ClassHead/BboxHead/LandmarkHead × 3) — direct
   ONNX initializer name → PyTorch state-dict key matching.

---

## Numerical Accuracy

FP16 convolutions with cuDNN are numerically equivalent to FP32 for this
pure-conv architecture (no accumulated groupwise statistics as in GroupNorm).

Expected maximum absolute error vs ORT FP32: `< 1e-3` on typical inputs.
Landmark outputs (decoding happens in FP32 post-processing) are unaffected.

## Running the Benchmark

```bash
# from repo root
.venv/Scripts/python custom_kernels/res50/benchmark_res50.py
```
