# Custom Kernels — GFPGAN-1024

FP16 PyTorch reimplementation of the `gfpgan-1024.onnx` face-restoration model,
with CUDA-graph acceleration.  Shares the `GFPGANTorch` implementation with
`custom_kernels/gfpgan_v1_4/`.  Used by VisoMaster-Fusion when the *Custom*
execution provider is selected.

Input: 512×512  →  Output: 1024×1024 (super-resolution face restoration)

## Benchmark Results

**Hardware:** NVIDIA GeForce RTX 4090 · PyTorch 2.8.0+cu129 · CUDA 12.9 · ORT 1.22.0
**Conditions:** 50 iterations, 10 warm-up, input (1,3,512,512) → output (1,3,1024,1024)

| Tier | Method | ms | vs ORT CUDA EP | vs TRT EP |
|------|--------|----|---------------:|----------:|
| 0 | ORT FP32 CUDA EP (baseline) | 24.43 ms | 1.00× | — |
| 0b | ORT TensorRT EP FP32 | 17.62 ms | 1.39× | 1.00× |
| 1 | PyTorch FP32 | 21.14 ms | 1.16× | 0.83× |
| 2 | PyTorch FP16 + Triton demod | 17.74 ms | 1.38× | 0.99× |
| **3** | **FP16 + Triton + CUDA graph (Custom)** | **11.79 ms** | **2.07×** | **1.49×** |
| **4** | **torch.compile default + FP16 + CUDA graph** | **7.19 ms** | **3.40×** | **2.45×** |
| 4b | torch.compile reduce-overhead | — *(may crash MLIR segfault on Windows/sm_89)* | — | — |

> **Application uses Tier 3** (FP16 + CUDA graph). Pass `torch_compile=True` to `build_cuda_graph_runner` to activate Tier 4 (3.40× vs ORT CUDA EP, 2.45× vs TRT EP).
> `reduce-overhead` skipped by default; set `GFPGAN1024_TORCH_COMPILE=1` to attempt.

Numerical accuracy: comparable to GFPGANv1.4 (FP16 ~1.6% mean relative error vs ORT).

### Kernel Priority Chain

1. **Triton `triton_demod` + `triton_fused_gfpgan_act`** — Windows-friendly, no MSVC required (preferred)
2. **CUDA C++ `gfpgan_demod_ext`** — pre-built `.pyd` from `model_assets/custom_kernels/` or JIT via MSVC
3. **Pure PyTorch** — automatic fallback

---

## Architecture Differences vs GFPGANv1.4

| Feature | v1.4 (512) | GFPGAN-1024 |
|---------|-----------|-------------|
| Output size | 512×512 | 1024×1024 |
| Input size | 512×512 | 512×512 |
| Latent codes | 16 | 18 |
| `final_extend_linear` | absent | [4096→1024] |
| `final_body_up` | absent | ResBlock up |
| `final_scale/shift` | absent | SFT for 1024px |
| `final_conv1/conv2` | absent | modulated convs at 1024px |
| ONNX nodes | 513 | 575 |

Detection of the 1024-output variant is automatic: `GFPGANTorch.from_onnx()` checks
for the `final_extend_linear` key in the ONNX initializer set and enables the extra
decoder stages accordingly.

---

## Files

| File | Purpose |
|------|---------|
| `gfpgan1024_torch.py` | Thin wrapper re-exporting `GFPGANTorch` |
| `benchmark_gfpgan1024.py` | 5-tier benchmark for this model |
| `benchmark_results.txt` | Saved benchmark output |
| `run_with_msvc.bat` | Windows build helper (sets up MSVC environment) |

The actual implementation lives in `custom_kernels/gfpgan_v1_4/gfpgan_torch.py`.

Pre-built CUDA C++ binaries are stored in `model_assets/custom_kernels/gfpgan_demod_ext.pyd`
(multi-arch fat binary, sm_70–sm_90).  Triton JIT cache lives in
`model_assets/custom_kernels/triton_cache/`.

---

## Application Integration

Select **"Custom"** in *Settings → General → Providers Priority*.

Both `GFPGAN-v1.4` and `GFPGAN-1024` automatically use the custom FP16 + Triton +
CUDA-graph path.  All other ONNX models continue to use TensorRT EP.

```python
# Internal call path (face_restorers.py):
from custom_kernels.gfpgan_v1_4.gfpgan_torch import GFPGANTorch
model = GFPGANTorch.from_onnx(onnx_path).cuda().eval()
output = model(input_f32_cuda)  # [1,3,512,512] float32 in → [1,3,1024,1024] float32 out
```

---

## Running the Benchmark

```bat
custom_kernels/gfpgan_1024/run_with_msvc.bat custom_kernels/gfpgan_1024/benchmark_gfpgan1024.py
# or directly:
.venv/Scripts/python custom_kernels/gfpgan_1024/benchmark_gfpgan1024.py
```

---

## Build Instructions

See `custom_kernels/gfpgan_v1_4/README.md` for full build instructions and prerequisites.

To build multi-arch fat binaries for all supported GPUs (sm_70–sm_90):

```bat
.venv\Scripts\python custom_kernels\build_kernels.py
```
