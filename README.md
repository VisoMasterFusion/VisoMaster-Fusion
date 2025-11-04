# VisoMaster Fusion

### Visomaster Fusion is a powerful yet easy-to-use tool for face swapping and editing in images and videos. It utilizes AI to produce natural-looking results with minimal effort, making it ideal for both casual users and professionals.

This version integrates major features developed by the community to create a single, enhanced application. It is built upon the incredible work of the original VisoMaster developers, **@argenspin** and **@Alucard24**.

---
<img src=".github/screenshot.png" height="auto"/>

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

### 🔄 **Face Swap**
-   Supports multiple face swapper models
-   Compatible with DeepFaceLab trained models (DFM)
-   Advanced multi-face swapping with improved masking (Occlusion/XSeg integration for mouth and face)
-   "Swap only best match" logic for cleaner results in multi-face scenes
-   Works with all popular face detectors & landmark detectors
-   Expression Restorer: Transfers original expressions to the swapped face

### ✨ **Restoration & Enhancement**
-   **Face Restoration**: Supports popular upscaling models, including the newly added **GFPGAN-1024**.
-   **ReF-LDM Denoiser**: A powerful reference-based U-Net denoiser to clean up and enhance face quality, with options to apply before or after other restorers.
-   **Advanced Texture Transfer**: Multiple modes for transferring texture details.
-   **AutoColor Transfer**: Improved color matching with a "Test_Mask" feature for more precise and stable results.
-   **Auto-Restore Blend**: Intelligently blends restored faces back into the original scene.

### 🎬 **Job Manager & Batch Processing**
-   **Dockable UI**: Manage all your jobs from a simple, integrated widget.
-   **Save/Load Jobs**: Save your entire workspace state (models, settings, faces) as a job file.
-   **Automated Batch Processing**: Queue up multiple jobs and process them all with a single click.
-   **Segmented Recording**: Set multiple start and end markers to render and combine various sections of a video into one final output.
-   **Custom File Naming**: Optionally use the job name for the output video file.

### 🚀 **Other Powerful Features**
-   **VR180 Mode**: Process and swap faces in hemispherical VR videos.
-   **Virtual Camera Streaming**: Send processed frames to OBS, Zoom, etc.
-   **Live Playback**: See processed video in real-time before saving.
-   **Face Embeddings**: Use multiple source faces for better accuracy & similarity.
-   **Live Swapping via Webcam**: Swap your face in real-time.
-   **Improved User Interface**: Pan the preview window by holding the right mouse button, batch select input faces with the Shift key, and choose from several new themes.
-   **Video Markers**: Adjust settings per frame for precise results.
-   **TensorRT Support**: Leverages supported GPUs for ultra-fast processing.

---

### **Prerequisites**
- Portable Version: No pre-requirements - Minimum Nvidia driver version: >=576.57
- Non-Portable Version:
    -   **Git** ([Download](https://git-scm.com/downloads))
    -   **Miniconda** ([Download](https://www.anaconda.com/download))
        <br> or
    -   **uv** ([Installation choices])(https://docs.astral.sh/uv/getting-started/installation/)

## **Installation Guide (VisoMaster-Fusion)**

### **Portable version**

Download only the Run_Portable.bat file from this repo (you don't need to clone the whole repo) from link below and put it in a new directory were you want to run VisoMaster from. Then just execute the bat file to run VisoMaster. Portable dependencies will be installed on the first run to portable-files directory.
- [Download - Start_Portable.bat](Start_Portable.bat)

You don't need any other steps from below for the portable version. Always start visomaster with Start_Portable.bat

### **Non-Portable - Installation Steps**

**1. Clone the Repository**
Open a terminal or command prompt and run:
```sh
git clone <URL_TO_YOUR_VISOMASTER_FUSION_REPO>
```
```sh
cd VisoMaster
```
```sh
git checkout fusion
```

**2. Create and Activate a python Environment (Skip if you already have one)**


#### In case you like to use "anaconda"

```sh
conda create -n visomaster python=3.11 -y
```
```sh
conda activate visomaster
```
```sh
pip install uv
```

### In case you like to use "uv" directly

```sh
uv venv --python 3.11
```
```sh
.venv\Scripts\activate
```

**3. Install requirements**
```
uv pip install -r requirements_cu129.txt
```

**4. Download required models**
```sh
python download_models.py
```

**5. Run the Application**

Once everything is set up, start the application:
- by opening the **Start.bat** file (for Windows)
or
Activate conda or uv environment in a terminal in the visomaster directory:

```
# If you use Anaconda
conda activate visomaster

# If you use uv only
.venv\Scripts\activate

# Start visomaster
python main.py
```


**5.1 Update to latest code state**
```sh
cd VisoMaster
git pull
```

---

**6. Install ffmpeg**

In Windows - Either via:

- powershell command: "winget install -e --id Gyan.FFmpeg --version 7.1.1"

<br>or

- Download ffmpeg zip: https://www.gyan.dev/ffmpeg/builds/packages/ffmpeg-7.1.1-essentials_build.zip
- Unzip it somewhere
- Add "\<unzipped ffmpeg path>\bin" folder to your Windows environment PATH variable

## How to use the Job Manager
1.  Set up your workspace as you normally would before recording (select source/target faces, adjust settings, etc.).
2.  In the Job Manager widget, click the **"Save Job"** button.
3.  Give your job a name. You can also choose whether to use this name for the final output file.
4.  The job will appear in the list. Set up more jobs if you wish.
5.  To process, select one or more jobs and click **"Process Selected"**, or click **"Process All"** to run the entire queue.
6.  Processing will begin automatically. A pop-up will notify you when all jobs are complete.

## Development

- Please use pre-commit before doing git add & commit. And fix found issues before the commit.

  ```
  # Install pre-commit
  uv pip install pre-commit
  
  # Usage before commits - (run twice in case of found auto corrections)
  pre-commit run --all-files
  ```

---

## [Join Discord](https://discord.gg/5rx4SQuDbp)

## Support The Project
This project was made possible by the combined efforts of the original developers and the modding community. If you appreciate this work, please consider supporting them.

### **Mod Credits**
VisoMaster-Fusion would not be possible without the incredible work of:
-   **Job Manager Mod**: Axel (https://github.com/axel-devs/VisoMaster-Job-Manager)
-   **Experimental Mod**: Hans (https://github.com/asdf31jsa/VisoMaster-Experimental)
-   **VR180/Ref-ldm Mod**: Glat0s (https://github.com/Glat0s/VisoMaster/tree/dev-vr180)
-   **Many Optimizations**: Nyny (https://github.com/Elricfae/VisoMaster---Modded)

## **Troubleshooting**
- If you face CUDA-related issues, ensure your GPU drivers are up to date.
- For missing models, double-check that all models are placed in the correct directories.

## [Join Discord](https://discord.gg/5rx4SQuDbp)

## Support The Project ##
This project was made possible by the combined efforts of **[@argenspin](https://github.com/argenspin)** and **[@Alucard24](https://github.com/alucard24)** with the support of countless other members in our Discord community. If you wish to support us for the continued development of **Visomaster**, you can donate to either of us (or Both if you're double Awesome :smiley: )

### **argenspin** ###
- [BuyMeACoffee](https://buymeacoffee.com/argenspin)
- BTC: bc1qe8y7z0lkjsw6ssnlyzsncw0f4swjgh58j9vrqm84gw2nscgvvs5s4fts8g
- ETH: 0x967a442FBd13617DE8d5fDC75234b2052122156B
### **Alucard24** ###
- [BuyMeACoffee](https://buymeacoffee.com/alucard_24)
- [PayPal](https://www.paypal.com/donate/?business=XJX2E5ZTMZUSQ&no_recurring=0&item_name=Support+us+with+a+donation!+Your+contribution+helps+us+continue+improving+and+providing+quality+content.+Thank+you!&currency_code=EUR)
- BTC: 15ny8vV3ChYsEuDta6VG3aKdT6Ra7duRAc


## Disclaimer: ##
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
