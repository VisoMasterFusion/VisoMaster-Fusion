# Custom Kernels — xseg (SN256_XSeg Face Segmentation U-Net)

FP16 PyTorch reimplementation of `XSeg_model.onnx`
(SN256_XSeg — symmetric 256-px U-Net face segmentation network) with a
single-capture CUDA graph.  Used as the **XSeg** model in the application
(DFL XSeg face occluder mask).

Model: **SN256_XSeg**  `(1,3,256,256)f32 → (1,1,256,256)f32`
(fixed spatial dims at inference; sigmoid binary segmentation mask)

---

## Benchmark Results

**Hardware:** NVIDIA GeForce RTX 4090 · PyTorch 2.8.0+cu129 · CUDA 12.9 · ORT 1.22.0
**Conditions:** 30 iterations, 5 warm-up, input 256×256

| Tier | Method | ms | vs ORT CUDA EP |
|------|--------|----|:--------------:|
| 0 | ORT FP32 CUDA EP (baseline) *(ConvTranspose on CPU)* | 11.83 ms | 1.00× |
| 0b | ORT TRT EP FP32 | 1.91 ms | 6.18× |
| 1 | PyTorch FP32 | 5.23 ms | 2.26× |
| 2 | PyTorch FP16 | 3.78 ms | 3.13× |
| **3** | **PT FP16 + CUDA graph (Custom)** | **1.89 ms** | **6.25×** |
| **4** | **PT FP16 + Triton RMSNormMax + CUDA graph (Custom)** | **1.89 ms** | **6.25×** |
| **5** | **torch.compile default + FP16 + CUDA graph** | **0.88 ms** | **13.44×** |
| 5b | torch.compile reduce-overhead | — *(skipped by default; set `XSEG_TORCH_COMPILE=1`)* | — |

> **Application uses Tier 4** when Triton is available. Pass `torch_compile=True` to `build_cuda_graph_runner` to activate Tier 5 (13.44× vs ORT CUDA EP, 2.17× vs TRT EP).
>
> **Note on ORT CUDA EP baseline:** ORT runs ConvTranspose nodes on CPU due to asymmetric padding — the baseline is artificially slow compared to fully GPU-resident PyTorch.

**Accuracy (Tier 4 vs ORT FP32):** MAE=8.20e-04, MaxAbsErr=1.91e-02; binary pixel agreement = 99.77%.

> **Tier 5 (torch.compile):** Pass `torch_compile=True` to `build_cuda_graph_runner`
> to enable `torch.compile(mode='default')` before CUDA graph capture.  Incurs a
> one-time ~30 s Triton JIT compile cost on first run.
>
> **Note on VRAM Leak Fix:** A multi-threading race condition caused by missing dedicated capture streams in the graph runner, which allowed CPU-side allocations to outpace GPU execution during high-FPS scenarios (like recording), has been permanently fixed. The runner now properly uses a dedicated stream and `torch.cuda.current_stream().synchronize()` inside the locks.
>
> **Note on ORT CUDA EP baseline:** ORT runs ConvTranspose nodes on CPU due to
> asymmetric padding, making the 15.07 ms baseline artificially slow.  The
> PyTorch FP32 result (5.50 ms) is a more realistic baseline for fair comparison
> — against which the Custom tier delivers **2.84x** speedup.
>
> **Note on Triton at this model size:** Triton RMSNormMax provides approximately
> the same speed as plain FP16 CUDA graph for this model size; the benefit
> appears at larger batch or spatial sizes.

Run `benchmark_xseg.py` to measure on your hardware.

### Speed-up Source

SN256_XSeg is a **Conv2d / ConvTranspose2d / custom-norm architecture**
with learned per-block RMS normalisation and Max activations.  Speed-up comes from:

1. **FP16** — cuDNN dispatches Conv2d and ConvTranspose2d on FP16 weights to
   TensorCore GEMM kernels (~1.8× throughput vs FP32 on Ampere/Ada GPUs).
2. **CUDA graph** — eliminates Python/CUDA kernel-launch overhead
   (~10–15% additional gain for this shallower but wide U-Net).
3. **Triton RMSNormMax** — fuses the 5-op PyTorch RMSNormMax sequence
   (`.pow(2)` → `.mean()` → `div` → `mul+add` → `torch.max`) into a single
   two-pass Triton kernel per channel (1 reduction pass + 1 write pass),
   reducing HBM round-trips across all 36 norm blocks in the forward pass.

---

## Architecture

Reverse-engineered from `model_assets/XSeg_model.onnx`:

```
SN256_XSeg U-Net
Input : 256×256 RGB, float32, values in [0, 1]
Output: (1, 1, 256, 256) sigmoid mask, values in [0, 1]

386 total nodes:
  37 Conv + 6 ConvTranspose + 36 GlobalAveragePool + 36 Sqrt + 36 Div
  108 Mul + 80 Add + 36 Max + 6 Concat + 2 MatMul + 2 Reshape + 1 Sigmoid
```

### Normalization (RMSNormMax — custom per-block)

Applied after every Conv/ConvTranspose except depthwise downsampling and the
final output Conv:

```
rms    = sqrt( mean(x²) + eps )         eps = learned scalar per block (Abs_* inits)
x_norm = x / rms
x_aff  = x_norm × γ + β                 γ, β ∈ ℝ^{1×C×1×1}  (Reshape_* inits)
out    = max( x_aff, max_val )          max_val ∈ ℝ^{1×C×1×1}  (learned per-channel floor)
```

### Encoder (downsampling path)

| Stage | In→Out | Convs (+ RMSNormMax) | Downsample kernel |
|-------|--------|---------------------|-------------------|
| enc0  | 3→32   | 2× Conv3×3          | DW 4×4 s=2, pad [1,1,2,2] |
| enc1  | 32→64  | 2× Conv3×3          | DW 3×3 s=2, pad=1 |
| enc2  | 64→128 | 2× Conv3×3          | DW 2×2 s=2, pad [0,0,1,1] |
| enc3  | 128→256| 3× Conv3×3          | DW 2×2 s=2, pad [0,0,1,1] |
| enc4  | 256→256| 3× Conv3×3          | DW 2×2 s=2, pad [0,0,1,1] |
| enc5  | 256→256| 3× Conv3×3          | DW 2×2 s=2, pad [0,0,1,1] |

Downsampling uses **grouped (depthwise) strided Conv** with asymmetric padding;
no bias on depthwise Conv.

Spatial progression: 256 → 128 → 64 → 32 → 16 → 8 → 4

### Bottleneck FC Bridge

```
4×4×256 = 4096 → Flatten → Linear(4096→512) → Linear(512→4096) → Reshape(256,4,4)
```

No activation between the two Linear layers.

### Decoder (upsampling path with skip connections)

| Stage | CT in→out | Skip source | Skip ch | Concat ch | Convs |
|-------|-----------|-------------|---------|-----------|-------|
| dec5  | 256→128   | enc5        | 256     | 384       | 3× (384→256→256→256) |
| dec4  | 256→128   | enc4        | 256     | 384       | 3× (384→256→256→256) |
| dec3  | 256→128   | enc3        | 256     | 384       | 3× (384→256→256→256) |
| dec2  | 256→128   | enc2        | 128     | 256       | 2× (256→128→128) |
| dec1  | 128→64    | enc1        | 64      | 128       | 2× (128→64→64) |
| dec0  | 64→32     | enc0        | 32      | 64        | 2× (64→32→32) |

Each ConvTranspose is followed by a **separate additive bias** `[1,C,1,1]`
(stored as an Add node in ONNX, not as a bias in the ConvTranspose itself),
then RMSNormMax.  All ConvTranspose use k=3, s=2, padding=1, output_padding=1.

### Output Head

```
Conv(32→1, 3×3, bias=True) → Sigmoid → (1, 1, 256, 256)
```

---

## Files

| File | Purpose |
|------|---------|
| `xseg_torch.py` | FP16 PyTorch SN256_XSeg U-Net + CUDA graph runner |
| `benchmark_xseg.py` | 4-tier latency benchmark vs ORT baseline |
| `__init__.py` | Package marker |

---

## Application Integration

Select **"Custom"** in *Settings → General → Providers Priority*.

The `run_dfl_xseg()` method in `face_masks.py` is then routed through
`XSegTorch` instead of the ONNX Runtime session:

```python
# Internal call path (face_masks.py):
from custom_kernels.xseg.xseg_torch import XSegTorch, build_cuda_graph_runner

model  = XSegTorch.from_onnx(onnx_path).cuda().eval()
runner = build_cuda_graph_runner(model)   # _CapturedGraph instance

# Per-call (image is (1,3,256,256) float32, values in [0,1]):
with torch.no_grad():
    result = runner(image)   # (1, 1, 256, 256) float32
output.copy_(result.view(256, 256))   # write to pre-allocated output buffer
```

---

## Weight Loading

All 43 **Conv / ConvTranspose** modules are loaded **positionally** by matching
ONNX topological Conv/ConvTranspose node order → PyTorch forward-execution order.

The 6 **ConvTranspose additive biases** (separate Add nodes with `[1,C,1,1]`
initialisers immediately after each ConvTranspose) are detected by checking
whether the tensor input of the Add was produced by a ConvTranspose node.

The 36 **RMSNormMax** blocks are loaded positionally by scanning ONNX nodes
in topological order:

| Node type | Initialiser shape | Destination |
|-----------|-------------------|-------------|
| `Add`     | `[1]`             | `norm.eps` (Python float attr) |
| `Mul`     | `[1, C, 1, 1]`    | `norm.gamma` |
| `Add`     | `[1, C, 1, 1]`    | `norm.beta` (if Add input ≠ ConvTranspose output) |
| `Max`     | `[1, C, 1, 1]`    | `norm.max_val`; then advance to next block |

**Linear (FC bridge)** weights are loaded by name
(`SN256_XSeg/dense1/weight/read__21 [4096,512]`,
`SN256_XSeg/dense2/weight/read__22 [512,4096]`), transposed for `nn.Linear`.
Biases are matched by Add-node initialiser shape (`[1,512]` and `[1,4096]`).

---

## Numerical Accuracy

Measured FP16 vs ORT FP32 (30 iterations, 5 warm-up):

| Metric | Value |
|--------|-------|
| MAE | 9.74e-04 |
| Max absolute error | 1.68e-02 |
| Binary pixel agreement (threshold 0.5) | 99.90% |

Face segmentation quality is unaffected at normal operating conditions.

## Running the Benchmark

```bash
# from repo root
.venv/Scripts/python custom_kernels/xseg/benchmark_xseg.py
```
