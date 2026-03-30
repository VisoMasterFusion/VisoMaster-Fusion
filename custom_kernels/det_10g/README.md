# Custom Kernels — det_10g (SCRFD-10G Face Detector)

FP16 PyTorch reimplementation of the `det_10g.onnx` face detector with
per-shape CUDA graph caching.  Used as the **RetinaFace** face detector in
the application when the *Custom* execution provider is selected.

## Models

| Model | Input | Output |
|-------|-------|--------|
| SCRFD-10G | `(1,3,H,W)` float32 (dynamic spatial dims) | 9 tensors: scores ×3, bbox ×3, kps ×3 |

## Benchmark Results

**Hardware:** NVIDIA GeForce RTX 4090 · PyTorch 2.8.0+cu129 · CUDA 12.9 · ORT 1.22.0
**Conditions:** 50 iterations, 10 warm-up; input 640×640

| Tier | Method | Latency | vs ORT FP32 |
|------|--------|--------:|:-----------:|
| 0    | ORT FP32 CUDA EP (baseline) | 3.01 ms | 1.00× |
| 0b   | ORT TensorRT EP FP32 | 1.77 ms | 1.70× |
| 1    | PyTorch FP32 pure ops | 3.48 ms | 0.86× |
| 2    | PyTorch FP16 NCHW pure ops | 3.07 ms | 0.98× |
| **3** | **PyTorch FP16 NCHW + CUDA graph** | **1.22 ms** | **2.46×** |
| 4    | PyTorch FP16 NHWC (channels_last) | 3.00 ms | 1.00× |
| 5    | PyTorch FP16 NHWC + CUDA graph | 1.25 ms | 2.40× |
| **6** | **torch.compile default + FP16 NHWC + CUDA graph** | **1.04 ms** | **2.89×** |
| 4b   | torch.compile reduce-overhead (no separate CUDA graph) | 1.06 ms | 2.83× |

Both compile modes work. `default + CUDA graph` (Tier 6) is recommended — slightly faster (1.04 vs 1.06 ms).

> **Application uses Tier 5** (NHWC + CUDA graph). Pass `torch_compile=True` to `build_cuda_graph_runner`
> to activate Tier 6 (torch.compile default + NHWC + CUDA graph).
>
> On PyTorch 2.8 + cuDNN 9.x, cuDNN already auto-selects NHWC convolution kernels for FP16
> inputs, so Tier 4 vs 2 show identical eager latency.  The CUDA graph is the primary
> speedup — it eliminates Python + CUDA kernel-launch overhead (~2× gain over eager FP16).

### Speed-up Sources

SCRFD-10G is a **pure Conv2d architecture** (BatchNorm folded into Conv at export).
All speed-up comes from:

1. **FP16** — cuDNN dispatches Conv2d on FP16 weights to TensorCore GEMM
   kernels (~2× throughput vs FP32 on Ampere/Ada GPUs).
2. **CUDA graph** — eliminates Python/CUDA kernel-launch overhead
   (~2.44× total gain over ORT FP32).
3. **NHWC (channels_last)** — model weights converted to NHWC layout so cuDNN
   uses its native NHWC path without internal NCHW↔NHWC reformatting.
   On PyTorch 2.8 + cuDNN 9 the gain is already included in (1); on older
   cuDNN versions this step provides an additional ~30% speedup.

4. **torch.compile** (Tier 6) — fuses Conv/SiLU sequences into optimised Triton kernels;
   one-time ~30 s Triton JIT cost per unique input shape.

No Triton kernels are used in Tiers 1–5 (no GroupNorm layers to fuse).

### Dynamic Input Handling

Unlike `res50.onnx` (fixed 512×512), SCRFD-10G accepts any spatial input.
The `Det10gGraphRunner` maintains a `Dict[(H, W) → CUDAGraph]` cache.
The first inference with a new shape captures a graph; all subsequent
inferences with the same shape replay it.  In practice the application uses
one fixed detection resolution per session, so only one graph is captured.

## Accuracy

FP16 vs ORT FP32 (640×640 input):

| Output | Max Absolute Error | Status |
|--------|--------------------|--------|
| scores | < 8.8e-4 | Pass |
| bbox   | 4.81e-3 (target < 1e-2) | Pass |
| kps    | 4.09e-3 (target < 1e-2) | Pass |

FP16 convolutions with cuDNN are numerically very close to FP32 for this
pure-conv architecture (no accumulated groupwise statistics as in GroupNorm).
Detection results (bbox, kps decoded in FP32 post-processing) are unaffected.

## Architecture Notes

Reverse-engineered from `model_assets/det_10g.onnx`:

```
SCRFD-10G: 3-conv stem + 4 residual stages + PA-FPN + per-stride heads
Input: H × W RGB, no normalisation (raw float32)
```

### Backbone (BN folded into Conv bias)

```
stem:   Conv(3→28,3×3,s=2)+ReLU, Conv(28→28,3×3)+ReLU, Conv(28→56,3×3)+ReLU
        MaxPool(2×2, s=2)  →  stride 4, 56ch

stage1: 3 × BasicBlock(56→56,   stride=1)   identity shortcut  →  stride 4
stage2: 1 × BasicBlock(56→88,   stride=2)                      →  stride 8,  88ch  → C4
        3 × BasicBlock(88→88,   stride=1)
stage3: 1 × BasicBlock(88→88,   stride=2)                      →  stride 16, 88ch  → C5
        1 × BasicBlock(88→88,   stride=1)
stage4: 1 × BasicBlock(88→224,  stride=2)                      →  stride 32, 224ch → C6
        2 × BasicBlock(224→224, stride=1)
```

**Strided BasicBlock shortcut** (SCRFD style, not standard ResNet):
```
shortcut = Conv1×1(in→out, bias=True)(AvgPool2d(2×2, ceil_mode=True)(x))
```

### PA-FPN Neck (10 Conv2d, no activations)

```
Top-down:
  lat_c4 = lateral0(C4)                               56ch, stride 8
  lat_c5 = lateral1(C5)                               56ch, stride 16
  lat_c6 = lateral2(C6)                               56ch, stride 32
  merged_c5 = lat_c5 + nearest_up(lat_c6)             stride 16
  p4        = fpn0(lat_c4 + nearest_up(merged_c5))    stride 8  → OUTPUT

Bottom-up (verified against ONNX graph):
  p5_merged = fpn1(merged_c5) + ds0(p4)               ds0 stride 2
  p5_pa     = pafpn0(p5_merged)                       stride 16 → OUTPUT
  p6_merged = fpn2(lat_c6) + ds1(p5_merged)           ds1 feeds from p5_merged (!)
  p6_pa     = pafpn1(p6_merged)                       stride 32 → OUTPUT
```

> **Note:** `ds1` feeds from `p5_merged` (the pre-PA merge point), not from
> the `pafpn0` output — verified in ONNX topological order.

### Per-stride Detection Head (×3)

```
Input: 56ch PA-FPN feature map
  shared: Conv(56→80,3×3)+ReLU, Conv(80→80,3×3)+ReLU, Conv(80→80,3×3)+ReLU
  cls:    Conv(80→2, 3×3)  → permute(0,2,3,1) → reshape(-1,1) → Sigmoid
  reg:    Conv(80→8, 3×3)  → * learnable_scale → permute → reshape(-1,4)
  kps:    Conv(80→20,3×3)  → permute(0,2,3,1) → reshape(-1,10)
```

Anchors per location: 2 (2 anchors × {1 score, 4 bbox coords, 10 kps coords}).

| Stride | Feature map (640 input) | Anchors |
|--------|------------------------|---------|
| 8 | 80×80 | 12,800 |
| 16 | 40×40 | 3,200 |
| 32 | 20×20 | 800 |

### Weight Loading

All 58 Conv2d weights are loaded **positionally** (ONNX topological order →
PyTorch forward-execution order).  This avoids the bias naming inconsistency
in the exported ONNX where `fpn_convs.1.bias` and `fpn_convs.2.bias` are
stored under `downsample_convs.0.conv.bias` and `downsample_convs.1.conv.bias`
respectively.

Three additional `nn.Parameter` scalars (`bbox_head.scales.{0,1,2}.scale`)
are loaded by direct initializer name.

| Range | Source |
|-------|--------|
| [0–29] | Backbone (stem + 4 stages), all integer-named initializers |
| [30–39] | Neck (lateral + fpn + ds + pafpn), mixed named initializers |
| [40–57] | Detection heads (×3 strides × 6 convs), mixed named initializers |

## Files

| File | Purpose |
|------|---------|
| `det10g_torch.py` | FP16 PyTorch SCRFD-10G + NHWC + per-shape CUDA graph runner |
| `benchmark_det10g.py` | 6-tier latency benchmark vs ORT baseline (Tiers 0–5) |
| `__init__.py` | Package marker |

## Usage

```python
from custom_kernels.det_10g.det10g_torch import Det10gTorch, build_cuda_graph_runner
import torch

# Load model (one-time)
model  = Det10gTorch.from_onnx("model_assets/det_10g.onnx").cuda().eval()
runner = build_cuda_graph_runner(model)   # Det10gGraphRunner, per-shape cache

# Per-frame call
with torch.no_grad():
    outputs = runner(aimg)   # (1,3,H,W) float32 GPU tensor
# outputs: tuple of 9 float32 GPU tensors
net_outs = [t.cpu().numpy() for t in outputs]
# net_outs[0..2] = scores, net_outs[3..5] = bbox, net_outs[6..8] = kps
# Existing postprocessing (anchor decode, NMS) is unchanged.
```

Select **"Custom"** in *Settings → General → Providers Priority* to activate.

## Running the Benchmark

```bat
.venv\Scripts\python custom_kernels\det_10g\benchmark_det10g.py
```
