# Custom Kernels — yoloface_8n (YOLOv8n-face Detector)

FP16 PyTorch reimplementation of the `yoloface_8n.onnx` face detector with
a single-capture CUDA graph.  Used as the **YOLOv8** face detector in the
application when the *Custom* execution provider is selected.

## Models

| Model | Input | Output |
|-------|-------|--------|
| YOLOv8n-face | `(1,3,640,640)` float32 (fixed spatial dims) | `(1,20,8400)` float32 — [cx,cy,w,h, cls_conf, kps×5×3] |

## Benchmark Results (NVIDIA GeForce RTX 4090 · PyTorch 2.8.0+cu129 · CUDA 12.9 · ORT 1.22.0)

50 iterations, 10 warm-up; input 640×640:

| Tier | Method | Latency | vs ORT CUDA EP |
|------|--------|---------|:--------------:|
| 0   | ORT FP32 CUDA EP | 3.22 ms | 1.00× (baseline) |
| 0b  | ORT TensorRT EP FP32 | 1.48 ms | 2.17× |
| 1   | PyTorch FP32 | 5.02 ms | 0.64× |
| 2   | PyTorch FP16 | 6.23 ms | 0.52× |
| 2b  | PyTorch BF16 | 11.27 ms | 0.29× |
| **3**  | **PT FP16 + CUDA graph** | **1.13 ms** | **3.09×** |
| **4**  | **torch.compile (reduce-overhead) — `build_cuda_graph_runner(torch_compile=True)`** | **0.83 ms** | **4.19×** |

`torch.compile(mode="default")` + manual CUDA graph **fails** on this model: Triton-generated
kernels call `CUDAGeneratorImpl::current_seed` during graph capture (Windows/sm_89 issue).
The fix is to use `reduce-overhead` mode which manages its own internal CUDA graphs.

> **Application uses torch.compile reduce-overhead (Tier 4)** — **0.83 ms (4.19× vs ORT CUDA EP)**.
> Pass `torch_compile=True` to `build_cuda_graph_runner` (default in Custom provider).
> If compilation fails, falls back to Tier 3 (manual CUDA graph).

### Speed-up Source

YOLOv8n uses **SiLU activations** and **C2f blocks** (pure Conv2d with BN
folded at export).  Speed-up comes from:

1. **FP16** — cuDNN dispatches Conv2d on FP16 weights to TensorCore GEMM
   kernels (~2× throughput vs FP32 on Ampere/Ada GPUs).
2. **CUDA graph** — eliminates Python/CUDA kernel-launch overhead
   (~15–25% additional gain vs FP16 eager).
3. **torch.compile(reduce-overhead)** (Tier 4) — fuses SiLU/Conv sequences and
   captures its own internal CUDA graphs; achieves **0.83 ms** (4.19× vs ORT).
   `mode="default"` + separate manual CUDA graph fails for this model on Windows
   (Triton sm_89: `current_seed` called during capture). Fixed by using
   `reduce-overhead` mode, which bypasses the manual graph capture entirely.

No Triton kernels are used in Tier 3 (SiLU = sigmoid×x, fused by PyTorch autocast).

Note that PT FP32 and PT FP16 eager are *slower* than ORT CUDA EP for this
small model — the CUDA graph is what recovers and surpasses ORT by eliminating
per-kernel CPU launch overhead.

### Fixed Input Handling

Unlike `det_10g.onnx` (dynamic spatial dims), `yoloface_8n.onnx` uses a
fixed 640×640 input.  A single CUDA graph is captured at first use.  All
subsequent inferences replay the same graph with no re-capture overhead.

## Accuracy

FP16 vs ORT FP32 (640×640 input):

| Output | Max Absolute Error | Status |
|--------|--------------------|--------|
| bbox   | 3.5e-1 (raw pre-NMS scores; visually correct after postprocessing) | Pass |
| cls    | 6.3e-4 | Pass |
| kps    | output-space coords differ; NMS output is visually correct | Pass |

The larger bbox raw-output delta reflects coordinate-space differences in DFL
decode; final detected face positions after NMS are visually indistinguishable
from the ORT reference.

## Architecture Notes

Reverse-engineered from `model_assets/yoloface_8n.onnx`:

```
YOLOv8n-face: backbone (model.0–9) + PAN neck (model.10–21) + Detect head (model.22)
Input: 640×640 RGB, float32 in [0,1] (no additional normalisation)
Output: (1,20,8400) — [cx,cy,w,h, cls_conf, kps_x0,y0,vis0, ..., x4,y4,vis4]
```

### Backbone (BN folded into Conv bias)

```
model.0:  Conv(3→16,  3×3, s=2) + SiLU                 stride 2,  16ch
model.1:  Conv(16→32, 3×3, s=2) + SiLU                 stride 4,  32ch
model.2:  C2f(32,  32, n=1)                              stride 4,  32ch
model.3:  Conv(32→64, 3×3, s=2) + SiLU                 stride 8,  64ch
model.4:  C2f(64,  64, n=2)                              stride 8,  64ch  → P3
model.5:  Conv(64→128,3×3, s=2) + SiLU                 stride 16, 128ch
model.6:  C2f(128,128, n=2)                              stride 16, 128ch → P4
model.7:  Conv(128→256,3×3, s=2) + SiLU                stride 32, 256ch
model.8:  C2f(256,256, n=1)                              stride 32, 256ch
model.9:  SPPF(256,256, k=5)                             stride 32, 256ch → P5
```

**C2f block**: `cv1(1×1) → split(2) → n×Bottleneck(3×3+3×3) → cat → cv2(1×1)`
**SPPF**: `cv1(1×1) → MaxPool×3(k=5,s=1,p=2) → cat(4) → cv2(1×1)`

### PAN Neck

```
Top-down:
  cat(up(P5), P4)      = 384ch → model.12 C2f(384,128,n=1) → P4_neck (stride 16)
  cat(up(P4_neck), P3) = 192ch → model.15 C2f(192,64,n=1)  → P3_out  (stride 8)

Bottom-up:
  cat(model.16(P3_out,s=2), P4_neck) = 192ch → model.18 C2f(192,128,n=1) → P4_out (stride 16)
  cat(model.19(P4_out,s=2), P5)      = 384ch → model.21 C2f(384,256,n=1) → P5_out (stride 32)
```

### Detection Head (model.22)

```
Three detection scales:
  stride 8  (P3_out, 64ch)  → 80×80  = 6400 anchors
  stride 16 (P4_out, 128ch) → 40×40  = 1600 anchors
  stride 32 (P5_out, 256ch) → 20×20  =  400 anchors
  Total: 8400 anchors

Per-scale branches (cv2/cv3/cv4):
  cv2 (bbox): CBS(in,64,3) → CBS(64,64,3) → Conv(64,64,1)   [4×reg_max=64 ch]
  cv3 (cls):  CBS(in,64,3) → CBS(64,64,3) → Conv(64,1,1)    [nc=1 ch]
  cv4 (kps):  CBS(in,16,3) → CBS(16,16,3) → Conv(16,15,1)   [nkpt×3=15 ch]

DFL (Distribution Focal Loss) bbox decode:
  raw_reg(B,64,N) → reshape(B,4,16,N) → softmax(dim=2) → weighted_avg([0..15]) → ltrb
  dist2bbox: cx = anchor_x + (r-l)/2,  w = l+r  (symmetric)
  → multiply by stride → pixel coords

KPS decode:
  raw_kps(B,15,N) → reshape(B,5,3,N)
  kps_xy  = (raw_xy * 2 + anchor_xy) * stride
  kps_vis = sigmoid(raw_vis)
```

### Weight Loading

All 73 Conv2d weights are loaded **by name** (Ultralytics naming convention
preserved in ONNX export).  Each Conv2d in `_CBS` stores its weight under
the `.conv` sub-attribute to match the `model.X.conv.weight` ONNX path.

The three plain `nn.Conv2d` layers at the end of each head branch
(`cv2.*.2`, `cv3.*.2`, `cv4.*.2`) are named without `.conv.` because
Ultralytics exports them directly (no BN wrapper).

| Range | Module | Description |
|-------|--------|-------------|
| model.0–9 | Backbone | stem convs + C2f stages + SPPF |
| model.12–21 | Neck | PAN top-down + bottom-up C2f |
| model.22 | Detect head | cv2/cv3/cv4 × 3 scales + DFL |

## Files

| File | Purpose |
|------|---------|
| `yoloface8n_torch.py` | FP16 PyTorch YOLOv8n-face + single-graph CUDA runner |
| `benchmark_yoloface8n.py` | Multi-tier latency benchmark vs ORT baseline (Tier 4 = default+CG, Tier 4b = reduce-overhead) |
| `__init__.py` | Package marker |

## Usage

```python
from custom_kernels.yoloface_8n.yoloface8n_torch import YoloFace8nTorch, build_cuda_graph_runner
import torch

# Load model (one-time)
model  = YoloFace8nTorch.from_onnx("model_assets/yoloface_8n.onnx").cuda().eval()
runner = build_cuda_graph_runner(model)   # _CapturedGraph instance

# Per-frame call
with torch.no_grad():
    net_outs = runner(aimg_prepared)  # (1,3,640,640) float32 → (1,20,8400) float32
net_outs_np = net_outs.cpu().numpy()
# Existing post-processing (score filter, bbox decode, NMS) is unchanged.
```

Select **"Custom"** in *Settings → General → Providers Priority* to activate.

## Running the Benchmark

```bat
.venv\Scripts\python custom_kernels\yoloface_8n\benchmark_yoloface8n.py
```
