# Custom Kernels — occluder (ResNet-encoder U-Net Face Occluder)

FP16 PyTorch reimplementation of `occluder.onnx`
(ResNet-encoder U-Net — symmetric 256-px binary occluder network) with a
single-capture CUDA graph.  Used as the **Occluder** model in the application
(detects hands, microphones, and other face obstructions).

Model: **occluder**  `(1,3,256,256)f32 → (1,1,256,256)f32`
(fixed spatial dims at inference; raw logits — positive = face / not occluded)

---

## Benchmark Results

**Hardware:** NVIDIA GeForce RTX 4090 · PyTorch 2.8.0+cu129 · CUDA 12.9 · ORT 1.22.0
**Conditions:** 30 iterations, 5 warm-up, input 256×256

| Tier | Method | ms | vs ORT CUDA EP |
|------|--------|----|:--------------:|
| 0 | ORT FP32 CUDA EP (baseline) | 1.48 ms | 1.00× |
| 0b | ORT TRT EP FP32 | 1.04 ms | 1.42× |
| 1 | PyTorch FP32 | 2.57 ms | 0.57× |
| 2 | PyTorch FP16 | 3.02 ms | 0.49× |
| **3** | **PT FP16 + CUDA graph (Custom)** | **0.62 ms** | **2.37×** |
| **4** | **torch.compile default + FP16 + CUDA graph** | **0.41 ms** | **3.59×** |
| 4b | torch.compile reduce-overhead | — *(skipped by default; set `OCCLUDER_TORCH_COMPILE=1`)* | — |

> **Application uses Tier 3** (FP16 + CUDA graph). Pass `torch_compile=True` to `build_cuda_graph_runner` to activate Tier 4 (3.59×).

**Accuracy (Tier 3 vs ORT FP32):** MAE=3.31e-03, MaxAbsErr=1.18e-02; binary pixel agreement = 100.00%.

Run `benchmark_occluder.py` to measure on your hardware.

### Speed-up Source

The occluder is a **pure Conv2d / ReLU architecture** with no normalization.
Speed-up comes from:

1. **FP16** — cuDNN dispatches Conv2d on FP16 weights to TensorCore GEMM kernels
   (~1.8× throughput vs FP32 on Ampere/Ada GPUs).
2. **CUDA graph** — eliminates Python/CUDA kernel-launch overhead
   (~10–15% additional gain).

---

## Architecture

Reverse-engineered from `model_assets/occluder.onnx`:

```
Occluder ResNet-encoder U-Net
Input : 256×256 RGB, float32, values in [0, 1]
Output: (1, 1, 256, 256) raw logits  (positive = face, negative = occluded)

76 total nodes:
  31 Conv + 1 MaxPool + 27 Relu + 8 Add (residual) + 5 Resize + 4 Concat
```

### Encoder (ResNet-style, **no BatchNorm**)

All convolutions have an explicit learnable bias.  No normalization layer exists
anywhere in the model.

| Stage | Input shape | Blocks | Channels | Stride |
|-------|------------|--------|----------|--------|
| Stem  | (1,3,256,256) | Conv7×7 + ReLU + MaxPool | 3→64 | s=2, s=2 |
| layer1 | (1,64,64,64) | 2× BasicBlock | 64→64 | s=1 |
| layer2 | (1,64,64,64) | 2× BasicBlock | 64→128 | s=2 first block |
| layer3 | (1,128,32,32) | 2× BasicBlock | 128→256 | s=2 first block |
| layer4 | (1,256,16,16) | 2× BasicBlock | 256→512 | s=2 first block |
| bottleneck | (1,512,8,8) | — | — | — |

**BasicBlock** (no BN, forward order = ONNX node order):
```
conv1(in→out, 3×3, stride) → ReLU → conv2(out→out, 3×3) → Add(shortcut) → ReLU
shortcut: identity  OR  Conv(in→out, 1×1, stride)  [when shape changes]
```

Spatial progression: 256 → 128 → 64 → 32 → 16 → 8

### Decoder (nearest-neighbour upsampling with skip connections)

All decoder blocks: **Resize(×2, nearest) → Concat(encoder skip) → 2× Conv3×3+ReLU**

| Stage | Resize in | Skip source | Skip ch | Concat ch | Conv out |
|-------|-----------|-------------|---------|-----------|---------|
| dec0 | 512, 8×8 | layer3.1 relu | 256 | 768 | 256 |
| dec1 | 256, 16×16 | layer2.1 relu | 128 | 384 | 128 |
| dec2 | 128, 32×32 | layer1.1 relu | 64 | 192 | 64 |
| dec3 | 64, 64×64 | stem relu | 64 | 128 | 32 |
| dec4 | 32, 128×128 | *(none)* | 0 | 32 | 16 |

Spatial progression (decoder): 8 → 16 → 32 → 64 → 128 → 256

### Output Head

```
Conv(16→1, 3×3, bias=True) → (1, 1, 256, 256) raw logits
```

Binarization (done in application): `mask = logits > 0` — True = face.

---

## Files

| File | Purpose |
|------|---------|
| `occluder_torch.py` | FP16 PyTorch occluder U-Net + CUDA graph runner |
| `benchmark_occluder.py` | 4-tier latency benchmark vs ORT baseline |
| `__init__.py` | Package marker |

---

## Application Integration

Select **"Custom"** in *Settings → General → Providers Priority*.

The `run_occluder()` method in `face_masks.py` is then routed through
`OccluderTorch` instead of the ONNX Runtime session:

```python
# Internal call path (face_masks.py):
from custom_kernels.occluder.occluder_torch import OccluderTorch, build_cuda_graph_runner

model  = OccluderTorch.from_onnx(onnx_path).cuda().eval()
runner = build_cuda_graph_runner(model)   # _CapturedGraph instance

# Per-call (image is (1,3,256,256) float32, values in [0,1]):
with torch.no_grad():
    result = runner(image)   # (1, 1, 256, 256) float32 raw logits
output.copy_(result.squeeze())   # write to pre-allocated (256,256) output buffer
```

---

## Weight Loading

All 31 **Conv2d** modules are loaded **positionally** by matching
ONNX topological Conv node order → PyTorch forward-execution order.

| Position | Module | Weight shape |
|----------|--------|-------------|
| 0 | stem_conv | (64, 3, 7, 7) |
| 1–2 | layer1.0 conv1, conv2 | (64,64,3,3) × 2 |
| 3–4 | layer1.1 conv1, conv2 | (64,64,3,3) × 2 |
| 5–7 | layer2.0 conv1, conv2, downsample | (128,64,3,3), (128,128,3,3), (128,64,1,1) |
| 8–9 | layer2.1 conv1, conv2 | (128,128,3,3) × 2 |
| 10–12 | layer3.0 conv1, conv2, downsample | (256,128,3,3), (256,256,3,3), (256,128,1,1) |
| 13–14 | layer3.1 conv1, conv2 | (256,256,3,3) × 2 |
| 15–17 | layer4.0 conv1, conv2, downsample | (512,256,3,3), (512,512,3,3), (512,256,1,1) |
| 18–19 | layer4.1 conv1, conv2 | (512,512,3,3) × 2 |
| 20–21 | dec0 conv1, conv2 | (256,768,3,3), (256,256,3,3) |
| 22–23 | dec1 conv1, conv2 | (128,384,3,3), (128,128,3,3) |
| 24–25 | dec2 conv1, conv2 | (64,192,3,3), (64,64,3,3) |
| 26–27 | dec3 conv1, conv2 | (32,128,3,3), (32,32,3,3) |
| 28–29 | dec4 conv1, conv2 | (16,32,3,3), (16,16,3,3) |
| 30 | head | (1,16,3,3) |

Downsample Conv (positions 7, 12, 17) are 1×1 stride-2 and only appear in
blocks where `in_ch != out_ch`.  The ONNX order is `conv1 → conv2 → downsample`
(the downsample branch executes last in the residual add).

---

## Numerical Accuracy

Measured FP16 vs ORT FP32 (30 iterations, 5 warm-up):

| Metric | Value |
|--------|-------|
| MAE (raw logits) | 2.81e-03 |
| Max absolute error | 1.29e-02 |
| Binary pixel agreement (threshold 0) | 100.00% |

Face occlusion quality is unaffected at normal operating conditions.

## Running the Benchmark

```bash
# from repo root
.venv/Scripts/python custom_kernels/occluder/benchmark_occluder.py
```
