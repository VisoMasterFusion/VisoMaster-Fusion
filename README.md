# VisoMaster Fusion

VisoMaster Fusion is a powerful yet easy-to-use tool for face swapping and editing in images and videos. It uses AI to produce natural-looking results with minimal effort, making it useful for both casual users and professionals.

This version integrates major features developed by the community to create a single, enhanced application. It is built upon the incredible work of the original VisoMaster developers, **@argenspin** and **@Alucard24**.

---
<img src=".github/screenshot.png" height="auto"/>

## Quick Links

- [Quick Start Guide](./docs/quickstart.md)
- [User Manual](./docs/user_manual.md)
- [Join Discord](https://discord.gg/5rx4SQuDbp)

## Fusion Features - Changelog v1.0.0

VisoMaster Fusion includes all the great features of the original plus major enhancements from community mods. Many thanks to GlatOs, Hans, Raven, Tenka, UnloadingVirus, Nyny, forgotten others and of course the original authors, Argenspin and Alucard24.

- User Interface :
	- Version is now : VisoMaster - Fusion - 1.0.0
	- Separate widgets panels for 'Target Videos/Images', 'Input Faces', 'Jobs' with each a hide/show toggle. 'Faces' panel toggle puts primary function on the 'Control Options' tabs.
	- Keep track of filters for 'Target Videos/Images'.
	- Added Send to Trash and Open File Location to Media 'Target Videos/Images' and 'Input Faces'.
	- Added new Themes : True-Dark, Solarized-Dark, Solarized-Light, Dracula, Nord, Gruvbox.
	- Open Output folder button.
	- Improved Parameters and Controls sorting and placement in 'Control Options' tabs with added categories.
	- Added 'Help' Menu with two help panels.
	- Added complete 'Embedding Editor'.
	- Save and restore 'Main Window' and 'Widgets' states within the workspace .json.
	- Updated and complete Start_Portable.bat file with new Launcher functions.

- Models :
	- Added 512 resolution for InstyleSwapper256 model A, B and C.
	- Added VGG_Combo_relu3 model for Differencing and Texture Transfer.
	- Added GFPGAN-1024 restorer model.
	- Added REF-LDM model as Denoiser.

- Improvements :
	- Inswapper128 Auto resolution.
	- Pre Swap Sharpness slider.
	- Face parser option to run at Pipeline End and Mouth Inside toggle to parse only the inside area of the mouth.
	- New Differencing method.
	- Improved AutoColor Transfer with added function for a secondary pass at Pipeline end.
	- Face Editor Pipeline position selection for added control.
	- Face Restorers 'Auto Restore' function to automaticaly adjust Blend value and sharpness and Face Restorer 2 at end toggle.
	- Face Expression Restorer rework to separate the Eyes and Lips options for fine tuning, added Normalize Eyes (to prevent 'fish eyes'), added Relative position toggle and added Pipeline position selection.

- New Features :
	- VR video support.
	- Job Manager.
	- Embedding Editor.
	- Images Batch Processing.
	- Xseg Mouth masks for added masking control of mouth area.
	- Mask View selection for 'Swap Mask', 'Differencing' and 'Texture Transfer'.
	- Transfer Texture function for added realism with mask manipulation for finetuning.
	- MPEG compression artifacts simulation.
	- REF-LDM Denoiser with included K/V map extraction function, Single Step and Full Restore modes, three Pipeline passes options.
	- Keep Controls Active toggle during recording option.
	- Audio function during playback with Volume and Delay sliders.
	- Playback Buffering option.
	- Playback Loop option.
	- Video Recording options, Frame Resize to 1920*1080, Open Output folder after recording, HDR Encoding (CPU), FFMPEG controls.
	- Segment recording with added Start/Stop position Markers.
	- Swap Input Face only once function to prevent false face swapping with multiple similar faces on scene.
	- Auto Save warkspace.
	- Auto Load Last Workspace.
	- Experimental settings.
	- Complet Presets function to save and apply per face parameters and global controls.

- Performance :
	- Reviewed memory usage with new and improved model loading and unloading to only keep necessary models in memory.
	- Improved Expression Restorer / Face Editor with async cuda threads for multithread usage.
	- Optimized Detection method using GPU instead of CPU.
	- No recording speed limit.

- Fixes :
	- Corrected many typos.
	- Unified requirements.txt file.
	- Keep Markers visible after VRAM clear.
	- Fixed VM crashes from ONNXRUNTIME Engine Cache creation by using a separate ONNX PROBE process that allows multiple retrys if it fails.
	- Corrected and optimized python code with pre-commit, and AI passes for added and translated comments.
	- Lots of bug fixes during implementation of new features and changes.
	- Added more relevant Debugging lines in console for better understanding what is happening.
	- Improved checks for file paths or general function calls with try...except.

---

## Detailed Feature List

### Face Swap
- Supports multiple face swapper models
- Compatible with DeepFaceLab trained models (DFM)
- Advanced multi-face swapping with improved masking (Occlusion/XSeg integration for mouth and face)
- "Swap only best match" logic for cleaner results in multi-face scenes
- Works with all popular face detectors and landmark detectors
- Expression Restorer transfers original expressions to the swapped face

### Restoration & Enhancement
- **Face Restoration**: Supports popular upscaling models, including **GFPGAN-1024**
- **ReF-LDM Denoiser**: A reference-based U-Net denoiser to clean up and enhance face quality, with options to apply before or after other restorers
- **Advanced Texture Transfer**: Multiple modes for transferring texture details
- **AutoColor Transfer**: Improved color matching with a `Test_Mask` feature for more precise and stable results
- **Auto-Restore Blend**: Intelligently blends restored faces back into the original scene

### Job Manager & Batch Processing
- **Dockable UI**: Manage all your jobs from a simple, integrated widget
- **Save/Load Jobs**: Save your entire workspace state as a job file
- **Automated Batch Processing**: Queue up multiple jobs and process them all with a single click
- **Segmented Recording**: Set multiple start and end markers to render and combine sections of a video into one final output
- **Custom File Naming**: Optionally use the job name for the output video file

### Other Powerful Features
- **VR180 Mode**: Process and swap faces in hemispherical VR videos
- **Virtual Camera Streaming**: Send processed frames to OBS, Zoom, and similar apps
- **Live Playback**: Preview processed video in real time before saving
- **Face Embeddings**: Use multiple source faces for better accuracy and similarity
- **Live Swapping via Webcam**: Swap your face in real time
- **Improved User Interface**: Pan the preview window by holding the right mouse button, batch-select input faces with `Shift`, and choose from several themes
- **Video Markers**: Adjust settings per frame for precise results
- **TensorRT Support**: Leverages supported GPUs for faster processing

---

### Prerequisites
- Portable Version: No pre-requirements - minimum Nvidia driver version: `>=576.57`
- Non-Portable Version:
  - **Git** ([Download](https://git-scm.com/downloads))
  - **Miniconda** ([Download](https://www.anaconda.com/download))
  - or **uv** ([Installation choices](https://docs.astral.sh/uv/getting-started/installation/))

## Installation Guide

### Portable Version

Download only the `Start_Portable.bat` file from this repo using the link below and place it in a new directory where you want to run VisoMaster. Then execute the batch file to start VisoMaster. Portable dependencies will be installed on the first run into the portable files directory.

- [Download - Start_Portable.bat](Start_Portable.bat)

You do not need any of the non-portable steps below for the portable version. Always start VisoMaster with `Start_Portable.bat`.

### Non-Portable Installation Steps

**1. Clone the repository**

Open a terminal or command prompt and run:

```sh
git clone https://github.com/VisoMasterFusion/VisoMaster-Fusion
cd VisoMaster-Fusion
```

> Most users should use the `main` branch. The repository also has a `dev` branch for newer or in-progress changes.

**2. Create and activate a Python environment**

Skip this if you already have one.

#### Using Anaconda

```sh
conda create -n visomaster python=3.11 -y
conda activate visomaster
pip install uv
```

#### Using uv directly

```sh
uv venv --python 3.11
.venv\Scripts\activate
```

**3. Install requirements**

```sh
uv pip install -r requirements_cu129.txt
```

**4. Download required models**

```sh
python download_models.py
```

**5. Run the application**

Once everything is set up, start the application:

- Open `Start.bat` on Windows
- Or activate your conda or uv environment in a terminal inside the `VisoMaster-Fusion` directory and run:

```sh
# If you use Anaconda
conda activate visomaster

# If you use uv only
.venv\Scripts\activate

# Start VisoMaster
python main.py
```

**5.1 Update to the latest code state**

```sh
cd VisoMaster-Fusion
git pull
```

---

**6. Install ffmpeg**

On Windows, either:

- Run: `winget install -e --id Gyan.FFmpeg --version 7.1.1`
- Or:
  - Download: https://www.gyan.dev/ffmpeg/builds/packages/ffmpeg-7.1.1-essentials_build.zip
  - Unzip it somewhere
  - Add `\<unzipped ffmpeg path>\bin` to your Windows `PATH`

## More Documentation

- For a practical first-run guide, see [Quick Start Guide](./docs/quickstart.md).
- For detailed workflows, settings, and feature coverage, see [User Manual](./docs/user_manual.md).

## Development

### Custom Kernels (Provider: "Custom")

The "Custom" inference provider delivers faster PyTorch FP16 + CUDA graph runners for all key models. It also uses a small set of hand-written CUDA kernels for fused ops (AdaIN, weight demodulation, cuBLASLt HGEMM) and Triton JIT kernels for everything else (GroupNorm+SiLU, pixel-shift, im2col-reflect, etc.).

**Pre-built binaries** (committed to the repo under `model_assets/custom_kernels/`):

| File | Purpose |
|---|---|
| `adain_fp16_ext.pyd` | Fused Adaptive Instance Normalisation (InSwapper, FP16) |
| `gfpgan_demod_ext.pyd` | Fused weight demodulation (GFPGAN / GPEN) |
| `style_block_ext.pyd` | cuBLASLt HGEMM + fused BIAS (InSwapper style blocks) |

These are multi-arch fat binaries covering **sm_75 → sm_120** (RTX 2000 through RTX 5000). End users need no compiler — the binaries are loaded directly at runtime.

**Rebuilding** (required when CUDA kernel sources change):

```sh
# Requires Visual Studio 2019/2022 (C++ workload) + CUDA Toolkit 12.8+
python custom_kernels/build_kernels.py
```

Then commit the updated `.pyd` files. The Triton kernels in `triton_ops.py` do **not** need a manual build step — they JIT-compile at first use on the user's GPU and cache automatically under `model_assets/custom_kernels/triton_cache/`. The cache is versioned by Triton+CUDA+Python version and old entries are pruned automatically on startup.

- Please use pre-commit before `git add` and commit, and fix any issues it reports.

```sh
# Install pre-commit
uv pip install pre-commit

# Usage before commits - run twice if auto-fixes were applied
pre-commit run --all-files
```

### Unit Tests

The project has a test suite covering core pipeline logic such as VR math, face masks, face detectors, serialization, job validation, and widget logic. Tests run without a GPU and without Qt installed.

**Setup (one-time)**

```sh
# Create a lightweight test venv (separate from the main app venv)
uv venv --python 3.12 .venv-test
.venv-test\Scripts\activate

# Install test dependencies
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
uv pip install numpy scipy scikit-image opencv-python pillow pytest pytest-mock
```

> If you already have a `.venv` with the full `requirements_cu129.txt` installed, you can run tests directly in it - the GPU packages do not interfere.

**Running the tests**

```sh
# Activate your venv first, then:
python -m pytest                     # run the full suite (206 tests, ~2s)
python -m pytest tests/unit/         # unit tests only
python -m pytest tests/integration/  # integration tests only
python -m pytest -k "vr"             # filter by keyword
python -m pytest -v                  # verbose output
```

**Test structure**

```text
tests/
|-- unit/
|   |-- helpers/      # ParametersDict, miscellaneous utils, VR math, thumbnails
|   |-- processors/   # Face detectors, masks, swappers, frame worker VR flow
|   |-- ui/           # Settings layout schema, save/load actions, job manager, widget logic
|   `-- utils/        # faceutil math
`-- integration/      # VR180 pipeline end-to-end
```

---

## Troubleshooting

- If you face CUDA-related issues, ensure your GPU drivers are up to date.
- For missing models, double-check that all models are placed in the correct directories.

## Support The Project

This project was made possible by the combined efforts of **[@argenspin](https://github.com/argenspin)** and **[@Alucard24](https://github.com/alucard24)**, with support from many other members of the Discord community. If you would like to support the continued development of **VisoMaster**, you can donate to either of us.

### Mod Credits

VisoMaster-Fusion would not be possible without the incredible work of:

- **Job Manager Mod**: Axel (https://github.com/axel-devs/VisoMaster-Job-Manager)
- **Experimental Mod**: Hans (https://github.com/asdf31jsa/VisoMaster-Experimental)
- **VR180/Ref-ldm Mod**: Glat0s (https://github.com/Glat0s/VisoMaster/tree/dev-vr180)
- **Many Optimizations**: Nyny (https://github.com/Elricfae/VisoMaster---Modded)
- **Launcher**: Tenka (https://github.com/t3nka)

### argenspin

- [BuyMeACoffee](https://buymeacoffee.com/argenspin)
- BTC: bc1qe8y7z0lkjsw6ssnlyzsncw0f4swjgh58j9vrqm84gw2nscgvvs5s4fts8g
- ETH: 0x967a442FBd13617DE8d5fDC75234b2052122156B

### Alucard24

- [BuyMeACoffee](https://buymeacoffee.com/alucard_24)
- [PayPal](https://www.paypal.com/donate/?business=XJX2E5ZTMZUSQ&no_recurring=0&item_name=Support+us+with+a+donation!+Your+contribution+helps+us+continue+improving+and+providing+quality+content.+Thank+you!&currency_code=EUR)
- BTC: 15ny8vV3ChYsEuDta6VG3aKdT6Ra7duRAc

## Disclaimer

**VisoMaster** is a hobby project that we are making available to the community as a thank you to all of the contributors ahead of us. We've copied the disclaimer from Swap-Mukham here since it is well-written and applies 100% to this repo.

We would like to emphasize that our swapping software is intended for responsible and ethical use only. We must stress that users are solely responsible for their actions when using our software.

Intended Usage: This software is designed to assist users in creating realistic and entertaining content, such as movies, visual effects, virtual reality experiences, and other creative applications. We encourage users to explore these possibilities within the boundaries of legality, ethical considerations, and respect for others' privacy.

Ethical Guidelines: Users are expected to adhere to a set of ethical guidelines when using our software. These guidelines include, but are not limited to:

Not creating or sharing content that could harm, defame, or harass individuals. Obtaining proper consent and permissions from individuals featured in the content before using their likeness. Avoiding the use of this technology for deceptive purposes, including misinformation or malicious intent. Respecting and abiding by applicable laws, regulations, and copyright restrictions.

Privacy and Consent: Users are responsible for ensuring that they have the necessary permissions and consents from individuals whose likeness they intend to use in their creations. We strongly discourage the creation of content without explicit consent, particularly if it involves non-consensual or private content. It is essential to respect the privacy and dignity of all individuals involved.

Legal Considerations: Users must understand and comply with all relevant local, regional, and international laws pertaining to this technology. This includes laws related to privacy, defamation, intellectual property rights, and other relevant legislation. Users should consult legal professionals if they have any doubts regarding the legal implications of their creations.

Liability and Responsibility: We, as the creators and providers of the deep fake software, cannot be held responsible for the actions or consequences resulting from the usage of our software. Users assume full liability and responsibility for any misuse, unintended effects, or abusive behavior associated with the content they create.

By using this software, users acknowledge that they have read, understood, and agreed to abide by the above guidelines and disclaimers. We strongly encourage users to approach this technology with caution, integrity, and respect for the well-being and rights of others.

Remember, technology should be used to empower and inspire, not to harm or deceive. Let's strive for ethical and responsible use of deep fake technology for the betterment of society.
