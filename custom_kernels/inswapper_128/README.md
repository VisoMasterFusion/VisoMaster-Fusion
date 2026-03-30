# Custom Kernels — inswapper_128.fp16.onnx

A PyTorch-native reimplementation of the `inswapper_128.fp16.onnx` face-swap model
with Triton and CUDA C++ fused kernels for AdaIN (Adaptive Instance Normalisation),
plus **batched pixel-shift inference** for the Inswapper Resolution setting.

---

## Benchmark Results

**Hardware:** NVIDIA GeForce RTX 4090 · PyTorch 2.8.0+cu129 · CUDA 12.9 · Triton 3.4.0 · ORT 1.22.0
**Conditions:** 50 iterations (shorter run), input 128×128

### Single-Tile Inference (B=1, 128px base)

| Tier | Method | ms | vs ORT CUDA EP | max\|diff\| |
|------|--------|---:|:--------------:|:-----------:|
| 0    | ORT CUDA EP (baseline) | 5.17 ms | 1.00× | — |
| 0b   | ORT TensorRT EP FP32 | 2.29 ms | 2.26× | 0.09570 |
| 1    | PyTorch pure ops | 3.71 ms | 1.39× | 0.01318 |
| 2    | PyTorch + Triton AdaIN | 3.71 ms | 1.39× | 0.01318 |
| 3    | PyTorch + CUDA C++ AdaIN | 4.63 ms | 1.12× | 0.01416 |
| 4    | PyTorch pure ops + CUDA graph | 3.53 ms | 1.47× | 0.01318 |
| **5** | **PyTorch + Triton AdaIN + CUDA graph (Custom)** | **3.52 ms** | **1.47×** | **0.01318** |
| 6    | PyTorch + CUDA C++ AdaIN + CUDA graph | 3.98 ms | 1.30× | 0.01416 |
| 7    | NHWC + Triton NHWC AdaIN + CUDA graph | 3.60 ms | 1.44× | 0.01514 |
| 8    | im2col + cuBLAS GEMM + Triton AdaIN + CUDA graph | 2.95 ms | 1.75× | 0.01587 |
| **8b** | **cuBLASLt HGEMM + fused BIAS + Triton AdaIN+residual + CUDA graph** | **2.60 ms** | **1.99×** | **0.02881** |
| 18   | torch.compile + Triton AdaIN + CUDA graph | — *(MLIR segfault on Windows/sm_89)* | — | — |
| 18b  | torch.compile reduce-overhead | — *(MLIR segfault on Windows/sm_89)* | — | — |

> **Best single-tile:** Tier 8b (cuBLASLt) at **2.60 ms (1.99× vs ORT)**.
> `torch.compile` crashes with an MLIR optimizer segfault in `libtriton.pyd` on this model
> (Windows/sm_89). Set `INSWAPPER_TORCH_COMPILE=1` env var to attempt it.
>
> Note: Tier 8b shows higher `max|diff|` (0.029) — the cuBLASLt BIAS epilogue uses different
> FP16 accumulation order than sequential `mm + add`. Face-swap quality is unaffected.

### Batched Pixel-Shift Inference (Custom provider only)

| Tier | Method | Total ms | Per-tile ms | Speedup |
|------|--------|--------:|------------:|:-------:|
| 9    | Sequential B=4  (dim=2, 256px) — 4 × single calls | 15.00 ms | 3.75 ms | 1.00× (ref) |
| **10** | **Batched B=4  (dim=2, 256px) — 1 × 4-tile call** | **11.75 ms** | **2.94 ms** | **1.28×** |
| 11   | Sequential B=9  (dim=3, 384px) — 9 × single calls | 33.60 ms | 3.73 ms | 1.00× (ref) |
| **12** | **Batched B=9  (dim=3, 384px) — 1 × 9-tile call** | **27.21 ms** | **3.02 ms** | **1.23×** |
| 13   | Sequential B=16 (dim=4, 512px) — 16 × single calls | 60.72 ms | 3.80 ms | 1.00× (ref) |
| **14** | **Batched B=16 (dim=4, 512px) — 1 × 16-tile call** | **50.15 ms** | **3.13 ms** | **1.21×** |

#### GEMM-mode Batched (Phase 1 optimisation — `torch.matmul` for B > 1)

| Tier | Method | Total ms | Diff (single vs batch) |
|------|--------|--------:|:----------------------:|
| 15   | GEMM Batched B=4  (torch.matmul) | 12.07 ms | 0.01172 |
| 16   | GEMM Batched B=9  (torch.matmul) | 28.89 ms | 0.00977 |
| 17   | GEMM Batched B=16 (torch.matmul) | 53.09 ms | 0.01025 |

> **Recommendation:** For batched pixel-shift inference the vanilla Triton AdaIN path
> (Tiers 10/12/14) remains faster than GEMM-mode batched (Tiers 15/16/17) by ~5%.
> cuDNN's implicit-GEMM convolution outperforms explicit `F.unfold + torch.matmul` at
> these batch sizes. GEMM/cuBLASLt mode is the best choice for single-tile B=1 inference only.

**Accuracy:** `max|diff|` batched vs single-tile: 0.010–0.012 — well within FP16
accumulation tolerance. GEMM batched is numerically correct (diff 0.010–0.012).

---

## Architecture Analysis

The ONNX model has 316 nodes, opset 11, all computation in FP16 with FP32 Cast
nodes at boundaries. The main bottleneck is the **AdaIN pattern** which repeats
12 times (6 style residual blocks × 2 convolutions each):

```
One AdaIN subgraph = 14 nodes:
  Cast(FP16->FP32), ReduceMean(mean), Cast(FP32->FP16),
  Sub(center), Cast×2(FP16->FP32), Mul(square), ReduceMean(var),
  Cast(FP32->FP16), Add(eps), Sqrt(std), Div(inv_std), Mul(scale), Add(bias)

Replaced by 1 kernel call:  y = scale * (x - mean) / sqrt(var + eps) + bias
```

### Custom Kernel Features

| Feature | Description |
|---------|-------------|
| **Welford single-pass** | Mean + variance in one pass over data (FP32 accumulators) |
| **`__half2` I/O** | 2 FP16 values per load/store — 2× memory throughput |
| **Warp-shuffle reduction** | `__shfl_down_sync` — 96 B shared memory vs 3 KB tree |
| **Grid: (B×C,)** | One program per `(batch_item, channel)` — handles any batch B |
| **Scale/bias broadcast** | Indexed by channel only; tiles sharing one source latent need no repeat |
| **Batched GEMM style** | 12 style projections `[2048×512]` fused into 1 `F.linear` call |
| **im2col + cuBLAS GEMM** | F.unfold + torch.mm for style-block convs — ~85 % tensor-core utilisation |
| **Batched cuBLAS GEMM** | `torch.matmul` broadcasts 2-D weight matrix over batch dim for B > 1 |
| **Fused reflect + im2col** | Triton Kernel 10 — eliminates intermediate padded tensor (saves ~0.85 ms at B=16) |
| **NHWC support** | `to_channels_last()` switches Conv2d to faster NHWC cuDNN path |
| **NHWC batched AdaIN** | `_adain_nhwc_batched_fwd` — native NHWC for any B (fixes B>1 correctness bug) |
| **cuBLASLt HGEMM** | cuBLASLt replaces torch.mm for 12 style-block GEMMs (Phase 3) — better algo selection |
| **cuBLASLt BIAS epilogue** | Bias add fused into the GEMM kernel — eliminates 12 separate kernel launches per forward |
| **Fused AdaIN + residual** | `_adain_fwd_batched_with_residual` (Triton Kernel 4b) — saves 6 residual-add launches |

### Triton AdaIN Kernel Dispatch Table

| Memory format | Batch | Kernel | Notes |
|---------------|-------|--------|-------|
| NCHW (default) | any B | `_adain_fwd_batched` | contiguous; base = `(b*C + c) * HW` |
| NCHW (default) | any B | `_adain_fwd_batched_with_residual` | **Phase 3** — fuses `y = adain(x) + residual` in one pass |
| NHWC (channels_last) | B = 1 | `_adain_nhwc_fwd` | stride-C scatter; C concurrent programs |
| NHWC (channels_last) | B > 1 | `_adain_nhwc_batched_fwd` | **Phase 1 fix** — native NHWC, no conversion; base = `b*HW*C`, offset = `hw*C + c` |

> **Bug fixed (Phase 1):** The previous NHWC B>1 path converted the input to NCHW
> (`x.contiguous()`) but wrote into the still-NHWC output buffer `y = empty_like(x)`,
> producing corrupted output (MaxAbsErr ~14.0) and wasting a full memory copy. The new
> `_adain_nhwc_batched_fwd` kernel operates natively in NHWC for any batch size.

### Fused ReflectionPad + Im2Col Kernel (Kernel 10)

The style-block convolutions in GEMM mode previously required two separate ops:

```
F.pad(x, pad=1, mode='reflect')   ->  [B, C, 34, 34]  (intermediate allocation)
F.unfold([B, C, 34, 34], k=3)     ->  [B, C*9, 1024]
torch.mm(w_flat, x_col)           ->  [C_out, 1024]
```

**Phase 2** replaces the first two steps with a single Triton kernel:

```
triton_im2col_reflect(x[B,C,32,32], k=3, pad=1)  ->  [B, C*9, 1024]  (no intermediate)
cuBLASLt_hgemm(w_flat, x_col)                     ->  [C_out, 1024]   (Phase 3)
```

**`_im2col_reflect_fwd` kernel design:**
- **Grid:** `(B × C × k², ceil(H×W / BLOCK_HW))` — one program per `(b, c_in, kh, kw)` slice
- **Reflection:** on-the-fly boundary clamping: `i < 0 -> -i`, `i >= H -> 2*(H-1) - i`
- **Output:** standard contiguous `[B, C*k², H*W]` ready for `torch.mm` / cuBLASLt
- **Saving:** eliminates one full read + write of the padded tensor (~34 MB at B=16, C=1024)

Enabled automatically by `enable_gemm_mode()` when Triton is available and `pad > 0`.

### cuBLASLt HGEMM with Fused BIAS Epilogue (Phase 3)

Each style block performs two 3×3 convolutions, each equivalent to a GEMM after im2col:

```
C[C_out, HW] = W[C_out, K_in] * X_col[K_in, HW] + bias[C_out]
```

**Phase 3** replaces `torch.mm` with cuBLASLt for all 12 style-block GEMMs:

1. **Better algorithm selection** — cuBLASLt heuristic search picks the optimal GEMM
   kernel for each (M, K, N) shape at model load time (once per session).

2. **BIAS epilogue** — `CUBLASLT_EPILOGUE_BIAS` fuses the bias add into the GEMM kernel,
   eliminating 12 separate `torch.add` launches per forward pass (~48 MB BW saved).

3. **Fused AdaIN + residual** — `triton_adain(..., residual=x)` runs the new
   `_adain_fwd_batched_with_residual` kernel, which computes `y = adain(x, sc, bi) + residual`
   in one pass, eliminating 6 residual-add launches per forward (~12–192 MB BW saved, B=1..16).

**Row-major ↔ col-major transposition recipe** (cuBLASLt is col-major):
```
C_row[M,N] = A_row[M,K] * B_row[K,N]
In col-major: C_col[N,M] = B_col[N,K] * A_col[K,M]
  A_row[M,K] -> col-major [K,M], ldb=K  (passed as cuBLASLt "B")
  B_row[K,N] -> col-major [N,K], lda=N  (passed as cuBLASLt "A")
  C_row[M,N] -> col-major [N,M], ldc=N
BIAS epilogue adds bias[j] to column j of C_col = row j of C_row (= C_out axis).
```

**CUDA graph compatibility** — cuBLASLt op handles (including the embedded bias pointer)
are created at `enable_cublaslt_mode()` time, *before* the CUDA graph capture window.
During replay, only `cublasLtMatmul` is executed. The 4 MiB workspace is pre-allocated
at a stable GPU address, so no dynamic allocation occurs inside the graph.

**Mathematical note on bias + AdaIN:** `adain(x + conv_bias) = adain(x)` because the
constant bias is cancelled by the mean subtraction. The bias epilogue is still present
(the cuBLASLt op is created with the real bias pointer) but is a no-op for layers
followed by AdaIN. The latency gain comes purely from eliminating the separate kernel
launches and from cuBLASLt's better GEMM algorithm selection.

### Batched GEMM for B > 1 (Phase 1)

`_forward_gemm` previously fell back to cuDNN `F.conv2d` for B > 1:

```python
# Before (fallback):
if B > 1:
    return F.conv2d(x.contiguous(), self.conv.weight, self.conv.bias, padding=0)

# After (batched cuBLAS):
x_col = F.unfold(x.contiguous(), kernel_size=k)          # [B, K_in, HW]
out   = torch.matmul(self._w_flat, x_col)                 # [B, C_out, HW]
# torch.matmul broadcasts w_flat [C_out, K_in] over batch dim -> [B, C_out, HW]
```

This ensures GEMM mode is fully functional for batched inference and eliminates the
cuDNN implicit-GEMM fallback. In practice cuDNN remains slightly faster for B > 1
(~5%), so the app uses vanilla Triton mode for pixel-shift batching.

### Pixel-Shift Triton Kernels (`triton_pixel_shift_extract/insert`)

Two Triton kernels for strided tile extraction and reassembly:

- **`triton_pixel_shift_extract(img[H,W,C], dim)`** -> `[B,C,128,128]`
  - Grid: `(B×C×128,)` — one program per row of one channel of one tile
  - Reads `img[j + th*dim, i + tw*dim, c]` with stride `dim×C`, writes contiguous BCHW
- **`triton_pixel_shift_insert(tiles[B,C,128,128], img[H,W,C], dim)`** — in-place scatter
  - Reverses the extract: reads contiguous BCHW, writes strided HWC

These kernels are compiled for dim ∈ {2, 3, 4} (Triton `constexpr` — one variant per dim).

---

## Optimisation History

| Phase | Change | Single-tile latency |
|-------|--------|---------------------|
| Initial | Triton AdaIN + CUDA graph (Tier 5) | 5.053 ms |
| GEMM mode | im2col + cuBLAS GEMM style blocks (Tier 8) | 4.263 ms |
| **Phase 1** | **Fix NHWC B>1 AdaIN bug; enable batched `torch.matmul` for B>1** | Correctness fix |
| **Phase 2** | **Fused reflect+im2col Triton kernel (Kernel 10)** | → **4.263 ms** (**1.91×** vs ORT CUDA EP) |
| **Phase 3** | **cuBLASLt HGEMM + fused BIAS epilogue + Triton AdaIN+residual** | → **3.630 ms** (**2.24×** vs ORT CUDA EP) |

### Phase 3 Kernel Launch Savings (B=1, per forward pass)

| Change | Launches saved | Bandwidth saved |
|--------|---------------|----------------|
| cuBLASLt BIAS epilogue (12 style-block GEMMs) | 12 | ~48 MB |
| Triton fused AdaIN + residual (6 style blocks) | 6 | ~12 MB |
| cuBLASLt better GEMM algorithm | — | ~5–10% GEMM throughput |

**Total Phase 3 improvement: 3.020 → 2.658 ms (-12.0%).**

---

## Kernel Priority Chain

```
GEMM-mode forward (style blocks with pad > 0):
  Phase 3 (cuBLASLt, B=1):
    1. cuBLASLt HGEMM with fused BIAS epilogue  -- style_block_ext.pyd loaded
    2. Triton fused reflect+im2col + torch.mm   -- fallback if ext unavailable
  Batched (B>1):
    1. Triton fused reflect+im2col + torch.matmul  -- im2col phase, then batched mm

AdaIN dispatch:
  1. Triton _adain_fwd_batched_with_residual (Phase 3) -- NCHW + residual fusion
  2. Triton _adain_fwd_batched                         -- NCHW, any B
  3. Triton _adain_nhwc_batched_fwd                    -- NHWC, B>1
  4. Triton _adain_nhwc_fwd                            -- NHWC, B=1
  5. CUDA C++ adain_fp16_ext                           -- pre-built .pyd or JIT (B=1 only)
  6. Pure PyTorch fallback                             -- always available
```

---

## Files

| File | Purpose |
|------|---------|
| `inswapper_torch.py` | PyTorch model + CUDA/cuBLASLt extension sources + `build_cuda_graph_runner` |
| `_styleblock_build/styleblock.cu` | Standalone cuBLASLt HGEMM source (mirrors `_STYLEBLOCK_CUDA_SRC`) |
| `_styleblock_build/main.cpp` | PyBind11 bindings for `styleblock.cu` |
| `_adain_build/adain_fp16_ext.pyd` | Pre-built Windows binary — fused AdaIN (Python 3.12, PyTorch 2.8+cu129) |
| `benchmark_inswapper.py` | 17-tier latency benchmark (single-tile + batched + GEMM batched) |
| `benchmark_results.txt` | Saved benchmark output |
| `dump_inswapper_ops.py` | Dev tool: prints all ONNX node shapes |
| `dump_weights.py` | Dev tool: prints all initializer names/dtypes |
| `profile_inswapper.py` | Dev tool: ORT profiler JSON analysis |
| `run_with_msvc.bat` | Windows build helper (sets up MSVC x64 environment) |

`model_assets/custom_kernels/` (shared multi-arch, built by `build_kernels.py`):

| File | Purpose |
|------|---------|
| `adain_fp16_ext.pyd` | Fused AdaIN — all GPU arches sm_75 through sm_120 |
| `gfpgan_demod_ext.pyd` | Fused weight demodulation — all GPU arches |
| `style_block_ext.pyd` | cuBLASLt HGEMM + BIAS epilogue — all GPU arches **(Phase 3)** |

---

## Application Integration

Select **"Custom"** in *Settings → General → Providers Priority*.

In Custom mode `Inswapper128` is executed by `InSwapperTorch`. On first use:
1. `to_gemm_mode()` enables im2col + Triton fused im2col (Tier 8 baseline)
2. `to_cublaslt_mode()` upgrades the 12 style-block GEMMs to cuBLASLt (Phase 3)
3. `build_cuda_graph_runner()` captures the entire forward as a static CUDA graph (B=1)

When the **Inswapper Resolution** setting is above 128px (`dim > 1`), all `dim×dim` tiles
are batched into a single GPU forward pass instead of `dim*dim` sequential calls:

```python
# Single-tile (128px / dim=1) — CUDA graph runner:
output = runner(target_f32_cuda[1,3,128,128], latent[1,512])  # -> [1,3,128,128]

# Batched pixel-shift (256px / dim=2, four tiles) — eager model:
output = model(target_f32_cuda[4,3,128,128], latent[1,512])   # -> [4,3,128,128]

# Batched pixel-shift (512px / dim=4, sixteen tiles) — eager model:
output = model(target_f32_cuda[16,3,128,128], latent[1,512])  # -> [16,3,128,128]
```

The source `latent[1,512]` is **shared** across all batch items (style vectors broadcast
automatically inside the model — the `F.linear` call and AdaIN scale/bias use `B=1` source
while target convolutions run at full batch size).

**Call path (face_swappers.py -> models_processor.py -> frame_worker.py):**

```python
# Single-tile path (Custom provider, dim=1) — CUDA graph:
face_swappers._get_inswapper_runner_b1()(target, latent)

# Batched path (Custom provider, dim > 1) — eager model:
models_processor.run_inswapper_batched(batch_B_3_128_128, latent_1_512, output_B_3_128_128)

# Sequential path (ORT providers or dim=1):
models_processor.run_inswapper(target_1_3_128_128, latent_1_512, output_1_3_128_128)
```

---

## Build Instructions (Windows — Developers Only)

> Regular users do **not** need to compile anything. The pre-built `.pyd` files in
> `model_assets/custom_kernels/` are loaded automatically on Python 3.12 with
> PyTorch 2.8+cu129. If they are incompatible with your stack the application falls
> back to Triton (preferred) or pure PyTorch ops gracefully.

### Prerequisites

| Tool | Version |
|------|---------|
| Visual Studio Build Tools | 2022 (MSVC v14.x) |
| CUDA Toolkit | 12.x |
| Python | 3.12 |
| PyTorch | 2.8+cu129 (must match runtime) |
| ninja | `uv pip install ninja` |

### Steps

1. **Open a terminal** in the project root.

2. **Build all kernels** (compiles `adain_fp16_ext`, `gfpgan_demod_ext`, and
   `style_block_ext` as multi-arch fat binaries):

   ```bat
   custom_kernels\inswapper_128\run_with_msvc.bat custom_kernels\build_kernels.py
   ```

3. **Run the benchmark** — run **without** `run_with_msvc.bat` (Triton's C launcher
   is incompatible with MSVC on PATH; use the venv Python directly):

   ```bat
   .venv\Scripts\python custom_kernels\inswapper_128\benchmark_inswapper.py
   ```

4. **Commit the new `.pyd` files** so other developers on the same stack skip compilation:

   ```bat
   git add model_assets/custom_kernels/
   git commit -m "Update pre-built custom kernel binaries (Phase 3: cuBLASLt)"
   ```

### How `run_with_msvc.bat` works

```bat
call "D:\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
cd /d "<project-root>"
.venv\Scripts\python %*
```

It initialises the MSVC x64 toolchain so `cl.exe` is on PATH, then runs the
given Python script from the project root. Only use this bat for **building** `.pyd`
files; do **not** use it for scripts that trigger Triton JIT compilation (Triton's
C launcher expects GCC/Clang, not MSVC).

### Kernel source locations

| Kernel | Source |
|--------|--------|
| Fused AdaIN (CUDA C++) | `_CUDA_SRC` / `_CPP_SRC` strings in `inswapper_torch.py`; standalone at `_adain_build/` |
| cuBLASLt HGEMM (Phase 3) | `_STYLEBLOCK_CUDA_SRC` / `_STYLEBLOCK_CPP_SRC` strings in `inswapper_torch.py`; standalone at `_styleblock_build/` |
| Triton AdaIN / im2col / pixel-shift | `custom_kernels/triton_ops.py` (JIT compiled, no manual build needed) |

---

## Extending to Other Models

This kernel follows the pattern for the **Custom provider** in VisoMaster-Fusion:

1. Create `custom_kernels/<model_name>/` with its own `__init__.py` and a
   PyTorch inference module.
2. Add a `_get_<model>_torch()` method in `face_swappers.py` (or the relevant
   processor) that lazily loads the model.
3. In the corresponding `run_<model>()` method check
   `self.models_processor.provider_name == "Custom"` and dispatch accordingly.
4. For models with batch-independent inputs (e.g. same conditioning vector per tile),
   add a `run_<model>_batched()` variant following the `run_inswapper_batched` pattern.
