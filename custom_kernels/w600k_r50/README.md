# Custom Kernels — w600k_r50 (IResNet-50 / ArcFace Face Recognition)

FP16 PyTorch reimplementation of `w600k_r50.onnx`
(IResNet-50 trained on WebFace600K) with a single-capture CUDA graph.
Used as the **Inswapper128ArcFace** recognition model in the application.

Model: **IResNet-50 / ArcFace**  `(1,3,112,112)f32 → (1,512)f32`
(fixed spatial dims at inference; 512-dim ArcFace face embedding)

---

## Benchmark Results

**Hardware:** NVIDIA GeForce RTX 4090 · PyTorch 2.8.0+cu129 · CUDA 12.9 · ORT 1.22.0
**Conditions:** 30 iterations, 5 warm-up, input 112×112

| Tier | Method | ms | vs ORT CUDA EP |
|------|--------|---:|:--------------:|
| 0    | ORT FP32 CUDA EP (baseline) | 2.00 ms | 1.00× |
| 0b   | ORT TensorRT EP FP32 | 2.10 ms | 0.95× *(slower than ORT on this tiny model)* |
| 1    | PyTorch FP32 | 4.02 ms | 0.50× |
| 2    | PyTorch FP16 | 4.63 ms | 0.43× |
| **3** | **PT FP16 + CUDA graph (Custom)** | **1.13 ms** | **1.78×** |
| **4** | **torch.compile default + FP16 + CUDA graph** | **0.82 ms** | **2.44×** |
| 4b   | torch.compile reduce-overhead | — *(MLIR segfault on Windows/sm_89)* | — |

> **Application uses Tier 3** (single CUDA graph; fixed 112×112 input).
> Pass `torch_compile=True` to `build_cuda_graph_runner` to activate Tier 4 (default+CG, 2.44×).
> `reduce-overhead` crashes (MLIR segfault) on this model; set `W600K_TORCH_COMPILE=1` to attempt.

**Accuracy (Tier 4 vs ORT FP32):** MAE = 7.39e-04, MaxAbsErr = 3.03e-03;
Cosine similarity = **0.999998** — face recognition quality is unaffected.

### Speed-up Source

IResNet-50 is a **pure Conv2d/PReLU/BatchNorm architecture** with live BatchNorm
(not folded). Speed-up comes from:

1. **FP16** — cuDNN dispatches Conv2d on FP16 weights to TensorCore GEMM
   kernels (~2× throughput vs FP32 on Ampere/Ada GPUs).
2. **CUDA graph** — eliminates Python/CUDA kernel-launch overhead across the
   full backbone; the 112×112 fixed input makes single-capture practical.

BatchNorm runs in eval mode (uses stored running_mean/running_var),
so its behaviour is a simple affine transform — fully deterministic and
compatible with CUDA graphs.

---

## Architecture

Reverse-engineered from `model_assets/w600k_r50.onnx`:

```
IResNet-50 / InsightFace w600k_r50
Input: 112×112 RGB, float32, normalised ([0,255]-127.5)/127.5
Output: (1, 512) ArcFace embedding (raw; application L2-normalises)

130 nodes: 53 Conv + 26 BatchNorm + 25 PReLU + 24 Add + 1 Flatten + 1 Gemm
```

### Stem

```
Conv(3→64, 3×3, s=1, p=1, bias=True) → PReLU(64)
No BatchNorm, no MaxPool — spatial dim stays at 112×112
```

### IBasicBlock (single BN, pre-activation)

```
Input x →
  BN1(in_ch)                      ← pre-activation normalisation
  → Conv1(in→out, 3×3, s=1, p=1)
  → PReLU(out_ch)
  → Conv2(out→out, 3×3, stride, p=1)
  → Add( identity_or_downsample(x) )

Shortcut:
  identity     if in_ch==out_ch and stride==1
  Conv1×1(in→out, stride)  otherwise  (no BN on shortcut)
```

### Backbone Stages

```
layer1: 3 × IBasicBlock( 64→ 64, stride=2 at block0) → 64ch, 56×56  ← s=2
layer2: 4 × IBasicBlock( 64→128, stride=2 at block0) → 128ch, 28×28
layer3: 14× IBasicBlock(128→256, stride=2 at block0) → 256ch, 14×14
layer4: 3 × IBasicBlock(256→512, stride=2 at block0) → 512ch,  7×7
```

### Head

```
BN2(512, 2d) → Flatten → FC(25088→512) → BN_features(512, 1d) → (1,512)
```

---

## Files

| File | Purpose |
|------|---------|
| `w600k_r50_torch.py` | FP16 PyTorch IResNet-50 + CUDA graph runner |
| `benchmark_w600k_r50.py` | 4-tier latency benchmark vs ORT baseline |
| `__init__.py` | Package marker |

---

## Application Integration

Select **"Custom"** in *Settings → General → Providers Priority*.

The `recognize()` method in `face_swappers.py` is then routed through
`IResNet50Torch` instead of the ONNX Runtime session for `Inswapper128ArcFace`:

```python
# Internal call path (face_swappers.py):
from custom_kernels.w600k_r50.w600k_r50_torch import (
    IResNet50Torch, build_cuda_graph_runner,
)

model  = IResNet50Torch.from_onnx(onnx_path).cuda().eval()
runner = build_cuda_graph_runner(model)   # _CapturedGraph instance

# Per-call (img already normalised, shape (1,3,112,112)):
with torch.no_grad():
    embedding = runner(img)    # → (1, 512) float32
embedding_np = embedding.cpu().numpy().flatten()
```

---

## Weight Loading

**53 Conv2d** are loaded **positionally** by matching ONNX topological
Conv node order → PyTorch forward-execution order.
All Conv weight initialiser names are integers (`685`–`842`).

**25 PReLU** slopes are loaded **positionally** by matching ONNX PRelu node
order → PyTorch forward-execution order.  Slope initialiser names are
integers (`843`–`867`), each with shape `[C, 1, 1]` (per-channel).

**26 BatchNorm** are loaded **by name** — ONNX initialisers have
human-readable keys (`layer1.0.bn1.weight`, `bn2.bias`, `features.running_mean`, etc.)
that map directly to PyTorch state_dict keys.

**FC (Gemm)** and **BN_features** weights loaded by name
(`fc.weight [512,25088]`, `fc.bias [512]`, `features.*`).

| Range | Module | Conv nodes |
|-------|--------|-----------|
| [0]   | Stem conv1 | 1 |
| [1–3] | layer1 block0 (conv1, conv2, downsample) | 3 |
| [4–7] | layer1 blocks 1–2 (2×2) | 4 |
| [8–10] | layer2 block0 | 3 |
| [11–16] | layer2 blocks 1–3 | 6 |
| [17–19] | layer3 block0 | 3 |
| [20–45] | layer3 blocks 1–13 | 26 |
| [46–48] | layer4 block0 | 3 |
| [49–52] | layer4 blocks 1–2 | 4 |
