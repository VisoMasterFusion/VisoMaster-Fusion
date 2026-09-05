# Running VisoMaster Fusion on macOS

The supported configuration for VisoMaster Fusion is Windows/Linux with an
NVIDIA GPU, using TensorRT or CUDA. macOS has neither, so this is a CPU/CoreML
port. It runs, and it produces correct output, but it is **much** slower than a
supported NVIDIA setup — see [Performance](#performance) before committing to it.

## Setup

```bash
brew install ffmpeg

uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements_mac.txt
.venv/bin/python download_models.py    # ~16 GB

./Start_mac.sh
```

`Start.bat` / `Start_Portable.bat` are Windows-only and assume a portable NVIDIA
runtime; `Start_mac.sh` replaces both.

## Why a separate requirements file

`requirements_cu13.txt` cannot be installed on macOS. It pins `torch==2.11.0+cu130`,
`onnxruntime-gpu`, `tensorrt-cu13`, and the `nvidia-*` CUDA runtime wheels, none
of which publish macOS builds.

`requirements_mac.txt` also has hard version ceilings that do not exist on other
platforms:

| Package | macOS x86_64 ceiling | Reason |
| --- | --- | --- |
| `torch` / `torchvision` | 2.2.2 / 0.17.2 | PyTorch stopped publishing x86_64 macOS wheels after 2.2.2 |
| `onnxruntime` | 1.23.2 | latest with macOS wheels; replaces `onnxruntime-gpu` |
| TensorRT | unavailable | NVIDIA-only |

The app already treats TensorRT as optional (`[WARN] No TensorRT Found`), so its
absence is not fatal.

`lightning` appears in `requirements_cu13.txt` but is not used by the macOS
runtime path, so it is omitted here.

## Execution providers

Provider choice is no longer a hardcoded list. `app/processors/utils/platform_support.py`
detects what the machine actually supports and the UI offers only those, so on a
Mac the "Providers Priority" dropdown shows `CPU` and `CoreML` rather than
CUDA/TensorRT options that would crash.

A workspace saved on an NVIDIA machine will request `TensorRT` or `CUDA`. Rather
than failing, that is downgraded to the best locally available provider with a
warning.

### CoreML

CoreML is wired up and selectable, but on **Intel** Macs it defaults to second
place behind CPU. Two findings drove that:

1. **Correctness.** CoreML must run with `RequireStaticInputShapes=1`. Several
   models here (RetinaFace, `det_10g`) have dynamic dimensions, and without that
   flag CoreML returns wrong-shaped outputs — inference dies with
   `Invalid shape for output feature`.
2. **Speed.** With that flag set, CoreML hands the dynamic subgraphs back to the
   CPU EP, so it contributes little and adds partitioning overhead. Measured on
   an i7-9750H / Radeon Pro 5300M:

   | Model | CPU | CoreML |
   | --- | --- | --- |
   | `det_10g` (detect) | 62 ms | 65 ms |
   | `w600k_r50` (recognize) | 84 ms | 89 ms |
   | `inswapper_128.fp16` (swap) | 1021 ms | 1244 ms |

   CoreML does win on fully static graphs (`vgg_combo_relu3_3_relu3_1`:
   108 ms vs 284 ms), which is why it stays available.

On **Apple Silicon** CoreML is ranked first, since the Neural Engine changes the
picture entirely. That path is untested — the port was developed and verified on
an Intel Mac.

## Performance

Roughly **1.2 s per face swap** on an i7-9750H. That is usable for single images
and fine for experimenting, but it makes video work impractical: a 30-second
30 fps clip is ~900 frames, or around 20 minutes of processing before restorers
and enhancers are enabled.

There is no way around this on Intel Mac hardware. Torch is capped at 2.2.2, there
is no CUDA, and MPS cannot be used for the model inference itself (see below).

## Why torch runs on CPU and not MPS

MPS *is* available on Metal 3 Macs, including Intel ones with an AMD GPU, and
plain torch ops do run on it. The pipeline still keeps tensors on the CPU because
inference binds torch tensors directly into ONNX Runtime:

```python
io_binding.bind_input(
    device_type=self.models_processor.device_type,
    buffer_ptr=image.data_ptr(),
    ...
)
```

There are ~74 such bindings. ONNX Runtime has no `mps` device and cannot
dereference a Metal buffer pointer, so an MPS tensor would be bound as garbage
rather than failing loudly. Moving torch to MPS would require staging every
bound tensor through the CPU each frame, which would cost more in transfers than
it saves.

`platform_support.torch_device_for_provider()` is the single place that decides
this, if someone wants to revisit it.

## Known limitations

- **Virtual camera** — `pyvirtualcam` needs OBS's virtual camera installed on
  macOS. The rest of the app works without it.
- **TensorRT features** — engine building, the TensorRT cache manager, and the
  "TensorRT-Engine" provider are all inert.
- **Unrelated test failures** — `tests/unit` reports 50 failures on macOS. These
  are pre-existing and reproduce identically on a clean checkout of `main`; they
  are not caused by this port.
