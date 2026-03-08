# User Manual

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Getting Started](#2-getting-started)
3. [Face Swap Tab](#3-face-swap-tab)
4. [Face Restoration](#4-face-restoration)
5. [Denoiser](#5-denoiser)
6. [Face Expression Restorer](#6-face-expression-restorer)
7. [Face Pose / Expression Editor](#7-face-pose--expression-editor)
8. [Frame Enhancers](#8-frame-enhancers)
9. [Face Detection & Tracking](#9-face-detection--tracking)
10. [Job Manager](#10-job-manager)
11. [Presets](#11-presets)
12. [Video Timeline & Markers](#12-video-timeline--markers)
13. [Recording & Output](#13-recording--output)
14. [Settings](#14-settings)
15. [Model Optimiser](#15-model-optimiser)
16. [Tips & Best Practices](#16-tips--best-practices)
17. [Glossary](#17-glossary)

---

## 1. Introduction

VisoMaster Fusion is a desktop application for AI-powered face swapping, enhancement, and editing on images, videos, and live webcam feeds. It provides a full pipeline of tools — from face detection through swapping, masking, restoration, and expression editing — all controlled through a graphical interface built with PySide6.

The application supports multiple AI inference backends (CPU, CUDA, TensorRT, TensorRT-Engine) and includes a batch job manager for processing multiple files unattended.

---

## 2. Getting Started

### 2.1 Launcher

When you first open VisoMaster Fusion, a launcher window appears. The home screen shows the current build commit, last update timestamp, and live status pills that flag any detected issues — pending git updates, missing models, dependency changes, or modified tracked files.

From the home screen you can launch the application directly or open the **Update / Maintenance** menu. A toggle at the bottom of the home screen controls whether the launcher appears on each startup.

The **Update / Maintenance** menu contains the following actions:

| Action | Description |
|---|---|
| **Update from Git** | Fetches and applies the latest commits from the remote repository. |
| **Repair Installation** | Restores all tracked application files to the current HEAD, backing up any local modifications first. |
| **Check / Update Dependencies** | Reinstalls Python dependencies via UV to match the current `requirements` file. |
| **Check / Update Models** | Runs the model downloader to fetch any missing or updated model files. |
| **Optimize Models (onnxsim)** | Runs ONNX simplification and symbolic shape inference on eligible model files for faster inference. Originals are backed up automatically. |
| **Revert to Original Models** | Deletes any optimized model files and re-downloads the originals from source. |
| **Update Launcher Script** | Applies any available update to the launcher batch script itself. |
| **Revert to Previous Version** | Opens a scrollable list of recent commits and lets you hard-reset the installation to a selected version. |

A **branch selector** at the top of the maintenance menu lets you switch between the `main` and `dev` branches. Switching branches discards local changes and re-synchronizes with the chosen remote branch; a dependency update is recommended afterward.

### 2.2 Main Window Layout

The main window is divided into several key areas:

- **Left panel** — source face input and face cards for assigning reference faces
- **Centre** — media preview with video playback controls and a timeline
- **Right panel** — tabbed settings panels (Face Swap, Face Editor, Denoiser, Settings)
- **Top toolbar** — file open/save, recording controls, and preset management

### 2.3 Supported Media Types

- **Images** — JPG, PNG, and other common still formats
- **Videos** — MP4 and most container formats supported by FFmpeg
- **Webcam / Live input** — real-time processing from a connected camera

### 2.4 Loading Media

Click the Open File button (or drag-and-drop) to load a target image or video. Source faces — the faces you want to paste in — are loaded separately via the face input panel on the left. Each source face becomes a face card that can be assigned to one or more detected faces in the target.

---

## 3. Face Swap Tab

The Face Swap tab contains the core swapping controls. Settings here apply per face card, so different people in the same clip can use entirely different configurations.

### 3.1 Swapper Model

Choose the AI model used to perform the face transfer. Each model has different characteristics:

| Model | Description |
|---|---|
| **Inswapper128** | The default model. Fast, versatile, and works well at multiple resolutions. Good starting point for most use cases. |
| **InStyleSwapper256 A / B / C** | Three style-aware variants operating at 256 px (optionally 512 px). Tend to preserve skin tone and style cues from the target. |
| **SimSwap512** | Operates natively at 512 px. Good identity preservation and fine detail. |
| **GhostFace v1 / v2 / v3** | A family of lightweight swappers. v2 and v3 generally outperform v1 in sharpness and identity fidelity. |
| **CSCS** | Combines two embeddings (appearance + identity) for stronger likeness. Best for challenging angles. |
| **DeepFaceLive (DFM)** | Uses custom pretrained DFM model files placed in the `onnxmodels/dfm_models` folder. Supports AMP Morph Factor and RCT colour transfer. |

### 3.2 Swapper Resolution

Available when using Inswapper128. Sets the internal resolution of the face crop during swapping (128, 256, 384, or 512 px). Higher values give more detail but are slower. Enable **Auto Resolution** to let the app pick based on the detected face size in the target frame.

### 3.3 Similarity Threshold

A filter (1–100) that controls how closely a detected face in the target must match your reference face card before the swap is applied. Higher values are stricter — only near-identical faces get swapped. Useful when multiple people share the screen and you only want to swap one of them.

### 3.4 Swap Strength & Likeness

| Feature | Description |
|---|---|
| **Strength** | Runs additional swap iterations on the result to deepen the effect. The Amount slider goes up to 500% (5× passes). 200% is a common sweet spot. Setting it to 0% disables swapping entirely but still allows the rest of the pipeline to run on the original image. |
| **Mode 2 (Anti-Drift & Texture)** | An advanced iteration mode using phase correlation and frequency separation. Reduces drift across many passes and better preserves skin texture. |
| **Face Likeness** | Directly adjusts how much the result resembles the source face versus the target. Negative values lean toward the target; values above 1.0 push harder toward the source. The range is −1.0 to 3.0. |
| **Face Keypoints Replacer** | Transfers the spatial landmark layout of the target face onto the source before swapping, helping the result fit the target's head pose and geometry. An Amount slider (0.00–1.00) controls how strongly the target keypoints are applied. |
| **Pre-Swap Sharpness** | Sharpens the original face before it enters the swap model. Can improve edge detail but may interfere with Auto Face Restorer. |

### 3.5 Masks

Masks control which pixels from the swapped face are blended back into the original frame. Multiple mask types can be enabled simultaneously — their results are composited together before the final blend. Each mask type addresses a different compositing problem, so they are generally used in combination rather than in isolation.

#### 3.5.1 Border Mask

A rectangular mask with adjustable Top, Bottom, Left, Right, and Blur sliders. Anything outside the mask boundary fades back to the original image. Useful for hiding stray pixels at the hairline or chin.

#### 3.5.2 Profile Angle Mask

A separate mask that automatically fades the far side of the face when the head is turned in profile view, hiding distortions at the edge of the swap. Controls include an **Angle Threshold** (0–90°, the head turn angle at which fading begins — lower values trigger it sooner) and a **Fade Strength** slider (0–100) that controls the intensity of the gradient.

#### 3.5.3 Occluder Mask

Detects objects covering the face — a hand, glasses, a microphone — and cuts them out of the swap so they appear naturally in the final composite. An **Occluder Size** slider (−100 to 100) grows or shrinks the detected region. The **Tongue/Mouth Priority** toggle uses FaceParser to prevent the tongue or inner mouth from being accidentally erased when growing the mask. A shared **Occluder/DFL XSeg Blur** slider controls edge softness for both the Occluder and XSeg masks when either is active.

#### 3.5.4 XSeg Mask

A second occlusion method using a dedicated XSeg segmentation model. Provides an independent mask channel that can be blended with the Occluder. The **Mouth/Lips Protection** toggle prevents XSeg from masking out the inner mouth area (useful for open mouths). A **Size** slider (−20 to 20) grows or shrinks the region. Includes a **XSeg Mouth** sub-option that applies a second XSeg pass specifically to the mouth region, with its own Size, Blur, and FaceParser-based grow sliders for the Mouth, Upper Lip, and Lower Lip.

#### 3.5.5 CLIP Text Mask

Uses the CLIP vision-language model to identify objects described in plain English (e.g. "glasses", "hat", "hand") and cuts them from the swap. Type one or more comma-separated terms into the text box and press Enter. Increase the **Amount** slider to make the segmentation more aggressive.

#### 3.5.6 Mouth Fit & Align

Repositions and scales the original mouth region to fit cleanly inside the swapped face without distorting its shape. Can be used independently of FaceParser. Options include: **Use Original Mouth** (uses the original face's mouth as the reference rather than the swap), **Paste After Restorer** (applies the mouth mask after the restorers rather than before), **Smart Sharpen (USM)** (edge-aware unsharp masking to sharpen teeth and lip edges without adding noise to surrounding skin), and a **Mouth Zoom** slider (0.90–1.20).

#### 3.5.7 Face Parser Mask

Uses a semantic face parsing model to produce a pixel-accurate mask over the face region. Each parsed area has an independent grow slider (0–30) that controls how much of that region is included in the swap. Available regions are: **Background**, **Face**, **Left/Right Eyebrow**, **Left/Right Eye**, **Eyeglasses**, **Nose**, **Mouth** (inner mouth and tongue), **Upper Lip**, **Lower Lip**, **Neck**, and **Hair**. Additional controls include **Parse at Pipeline End** (runs the mask after the full pipeline rather than before restorers), **Mouth Inside** (keeps the parse inside the mouth boundary), **Background Blur**, **Face Blur**, and **Face Blend** sliders for edge softening and blending.

#### 3.5.8 Restore Eyes

Blends the original eyes back into the swapped face using a configurable elliptical mask. Controls include **Eyes Blend Amount** (balance between original and swapped eyes), **Eyes Feather Blend** (edge softness of the blend), **Eyes Size Factor** (overall mask size — reduce for smaller/distant faces), **X/Y Eyes Radius Factor** (shape the mask from circular to oval), and **X/Y Eyes Offset** and **Eyes Spacing Offset** sliders for precise positioning.

#### 3.5.9 Restore Mouth

Blends the original mouth back into the swapped face, similar in structure to Restore Eyes. Includes a **Mouth Blend Amount** slider for controlling the mix between original and swapped mouth regions.

#### 3.5.10 Background Mask

When enabled, the unprocessed background from the original image shows through in the final composite rather than being replaced by the swapped result. Useful when you want the swap to affect only the face region.

---

## 4. Face Restoration

After swapping, one or two face restorer models can be applied to sharpen details and correct AI artefacts. Restorers operate on the aligned face crop and blend the result back.

### 4.1 Restorer Models

| Model | Description |
|---|---|
| **GFPGAN v1.4** | Fast and versatile all-round restorer. Good default choice. |
| **GFPGAN-1024** | Higher-resolution variant. More detail at the cost of speed. |
| **CodeFormer** | Quality-focused restorer with a Fidelity Weight slider (0–1). Lower fidelity = more creative but less faithful; higher = closer to the original. |
| **GPEN-256 / 512 / 1024 / 2048** | GPEN at different internal resolutions. Higher = more detail, slower. |
| **RestoreFormer++** | Attention-based restorer. Tends to produce very natural skin. |
| **VQFR-v2** | Vector-quantised restorer. Supports a Fidelity Weight similar to CodeFormer. |

### 4.2 Restorer Controls

| Control | Description |
|---|---|
| **Alignment** | How the face crop is positioned for restoration. **Original** restores directly on the existing swap crop — the default and fastest option. **Blend** re-warps the crop to a standard ArcFace-aligned position before restoring, which can improve results on faces that are not well-centred. **Reference** aligns to the detected target face landmarks instead, useful when the swap geometry differs significantly from the source. |
| **Blend** | The mix ratio (0–100%) between the restored face and the raw swap output. 100% uses only the restorer result. |
| **Auto Restore** | Automatically adjusts the blend amount per frame based on a sharpness analysis. Useful when face size or motion varies across a video. Includes an **Adjust Sharpness** slider to offset the sharpness threshold used for the calculation. |
| **Sharpness Mask** | Within Auto Restore, uses a per-pixel sharpness map to apply stronger blending only where the image is soft. |
| **Second Restorer** | A second, independent restorer pass with its own model, alignment, and blend settings. Can be set to run at the end of the full pipeline — after the Face Editor — to recover sharpness lost by later processing steps. |

---

## 5. Denoiser

The Denoiser uses a UNet-based latent diffusion model (Ref-LDM) to reduce noise and reconstruct fine texture on the aligned face. It can be independently enabled at three points in the pipeline: **before the restorers**, **after the first restorer**, and **after all restorers**. Each pass has its own mode and parameter controls.

**Global controls** (shared across all passes):

| Setting | Description |
|---|---|
| **Exclusive Reference Path** | Forces the UNet to attend only to reference key/value features, maximising focus on the source face style. Enabled by default. |
| **Base Seed** | Fixed random seed (1–999) for reproducible noise patterns across all frames and all denoiser passes. |

**Per-pass controls** (available for each of the three pipeline positions):

| Setting | Description |
|---|---|
| **Denoiser Mode** | **Single Step (Fast)** adds and removes a controlled amount of noise in one pass — fast and subtle. **Full Restore (DDIM)** runs full iterative diffusion over multiple steps for more detail at greater cost. |
| **Single Step Timestep (t)** | Available in Single Step mode. Controls how much noise is injected and therefore removed. Lower values are more conservative (range: 1–500). |
| **DDIM Steps** | Available in Full Restore mode. Number of denoising iterations — more steps produce a more refined result (range: 1–300). |
| **CFG Scale** | Available in Full Restore mode. How strongly the denoiser follows the reference features. Higher values increase adherence to the source appearance (range: 0.0–10.0). |
| **Latent Sharpening** | Applies sharpening directly inside the latent space before decoding. A value around 0.15 is a reasonable starting point (range: 0.0–2.0). |

---

## 6. Face Expression Restorer

The Face Expression Restorer transfers the expression, eye movement, and head pose from the original (driving) face onto the swapped face using the LivePortrait model pipeline.

### 6.1 Pipeline Position

Choose where in the processing chain the expression restorer runs: **Beginning**, **After First Restorer**, **After Texture Transfer**, or **After Second Restorer**. Running it later can compensate for any stiffening introduced by restoration steps.

### 6.2 Mode

The restorer operates in two modes selected via the **Face Expression Mode** dropdown:

- **Simple** — A streamlined mode with a single unified expression factor slider and an animation region selector. Suitable for most use cases.
- **Advanced** — Exposes per-region controls for Eyes, Brows, Lips, and General Face Features independently, along with retargeting, normalization, and relative position toggles for each region.

### 6.3 Core Controls (both modes)

| Control | Description |
|---|---|
| **Neutral Factor** | The percentage of expression to restore (0.0–1.0, default 0.30). Because the swapped face already carries some expression from the swap model, applying a full value of 1.0 is not recommended — the slider tooltip advises keeping it below 1.0. |
| **Expression Factor** | Controls the overall intensity of expression transfer between the driving face and the swapped result. |
| **Micro-Expression Boost** | Amplifies subtle movements that might otherwise be lost during the swap, while protecting strong expressions from distortion. Default is 0.5. |

### 6.4 Simple Mode Controls

| Control | Description |
|---|---|
| **Animation Region** | Selects which facial regions are involved in expression restoration. Options include `all`, `eyes`, `lips`, and others. |
| **Normalize Lips** | Normalises the lip open/close ratio using a threshold, preventing extreme values. Includes a configurable **Threshold** slider. |

### 6.5 Advanced Mode Per-Region Controls

Each region (Eyes, Brows, Lips, General Face Features) has independent enable toggles and factor sliders, plus these additional options:

| Control | Description |
|---|---|
| **Relative Position** | Makes the animation relative to the initial pose of the source image, reducing geometric distortions on face shape. Available per region. |
| **Retargeting (Eyes / Lips)** | Adjusts the open/close ratio of eyes or lips to match the driving face more precisely. A **Multiplier** slider controls the intensity. |
| **Normalize (Eyes / Lips)** | Normalises the open/close ratio using a configurable threshold so extreme values are clamped to a sensible range. A **Max Open Ratio** cap is also available for eyes. |
| **Include Nose / Jaw / Cheek / Contour / Head Top** | Under General Face Features — adds the corresponding landmark group to the general expression region. Each has its own factor slider. |



---

## 7. Face Pose / Expression Editor

The Face Pose/Expression Editor lets you directly manipulate the swapped face's pose and expression using sliders, without needing a driving video. It uses the LivePortrait model pipeline under the hood.

### 7.1 Head Pose

| Control | Description |
|---|---|
| **Head Pitch** | Tilts the face up or down (nodding motion). |
| **Head Yaw** | Rotates the face left or right (turning motion). |
| **Head Roll** | Tilts the head sideways. |
| **X / Y / Z-Axis Movement** | Translates the face along the horizontal, vertical, or depth axis. |

### 7.2 Eye & Brow Controls

| Control | Description |
|---|---|
| **Eyes Open/Close Ratio** | Opens or closes the eyes on a continuous scale. |
| **Eye Wink** | Triggers a wink on one eye. |
| **EyeBrows Direction** | Raises or lowers the eyebrows. |
| **EyeGaze Horizontal / Vertical** | Redirects the gaze direction without moving the head. |

### 7.3 Mouth & Lip Controls

| Control | Description |
|---|---|
| **Lips Open/Close Ratio** | Opens or closes the mouth. |
| **Mouth Pouting** | Pushes the lips forward into a pout. |
| **Mouth Pursing** | Tightens and narrows the lips. |
| **Mouth Grin** | Widens the mouth into a grin. |
| **Mouth Smile** | Curves the corners of the mouth into a smile. |

### 7.4 Makeup

AI-powered makeup is applied using the FaceParser model to identify facial regions, then colour-blended on top of the image. Each area has independent Red/Green/Blue colour sliders and a Blend Amount (0 = original colour, 1 = full target colour).

| Area | Description |
|---|---|
| **Face Makeup** | Colours the skin on the face — cheeks, forehead, nose bridge — excluding hair, eyebrows, eyes, and lips. |
| **Hair Makeup** | Colours the hair region. |
| **EyeBrows Makeup** | Colours the eyebrows. |
| **Lips Makeup** | Colours the lips. |

---

## 8. Frame Enhancers

Frame Enhancers improve the quality of the entire output frame, not just the face region. They are applied as a post-processing step.

### 8.1 Upscaling Models

| Model | Description |
|---|---|
| **RealESRGAN x2 / x4 Plus** | AI super-resolution at 2× or 4× scale. Excellent general-purpose upscaler for photos and videos. |
| **BSRGan x2 / x4** | Blind super-resolution model. Good at recovering fine detail on compressed or blurry inputs. |
| **UltraSharp x4** | Optimised for sharpness and edge clarity at 4× scale. |
| **UltraMix x4** | A blended upscaler model balancing sharpness and naturalness. |
| **RealESR-General x4v3** | A general-purpose variant of RealESRGAN tuned for a wide range of degradation types. |

### 8.2 Colourisation Models

| Model | Description |
|---|---|
| **DeOldify Artistic** | Colourises black-and-white footage with a painterly, vibrant style. |
| **DeOldify Stable** | Colourises with a more conservative, consistent style suited to historical photos. |
| **DeOldify Video** | A temporal-aware variant of DeOldify optimised for video to reduce colour flickering. |
| **DDColor Artistic** | Modern deep-learning colouriser with rich, saturated colours. |
| **DDColor** | Standard DDColor model offering natural-looking colourisation. |

---

## 9. Face Detection & Tracking

### 9.1 Face Detector Models

The app uses ONNX-based detectors to locate faces in each frame before swapping. The active model is selected in the Settings tab.

| Model | Description |
|---|---|
| **RetinaFace** | A single-stage face detector from the InsightFace project (CVPR 2020). High accuracy across a wide range of face sizes and orientations. The default and generally recommended choice. |
| **SCRFD** | Sample and Computation Redistribution for Face Detection (ICLR 2022, InsightFace). Designed for an efficient accuracy-to-compute trade-off. The variant used here (SCRFD-2.5G) targets a 2.5 GFlop budget — faster than RetinaFace with competitive accuracy. |
| **Yolov8** | A YOLOv8-based face detector (YoloFace8n). Fastest of the four options. Good choice for real-time or webcam use where speed matters more than peak accuracy. |
| **Yunet** | A lightweight millisecond-level face detector developed by Shiqi Yu and distributed via the OpenCV Model Zoo. Very low compute footprint; well-suited to CPU inference and resource-constrained scenarios. Notable for handling side-on and partially occluded faces well. |

| Setting | Description |
|---|---|
| **Detect Score** | Minimum confidence threshold for a detection to be accepted. Lower values catch more faces but may produce false positives. |
| **Max Faces to Detect** | Limits how many faces are processed per frame. Useful for performance when only one or two faces are relevant. |
| **Detection Interval** | Runs face detection only every N frames and reuses the result in between. Reduces CPU/GPU load on high-frame-rate video. |
| **Auto Rotation** | Rotates the input frame to the detected face's upright orientation before processing. |

### 9.2 ByteTrack Face Tracking

When enabled, ByteTrack assigns a consistent ID to each face across frames. This allows the app to apply the correct face card settings to the right person even when faces briefly leave frame or overlap.

| Setting | Description |
|---|---|
| **Track Threshold** | Minimum detection score for a new track to be initialised. |
| **Match Threshold** | How closely a detection must match an existing track to be linked to it. |
| **Track Buffer (Frames)** | How many frames a track is kept alive after the face disappears before it is discarded. |

---

## 10. Job Manager

The Job Manager allows you to queue multiple target files for batch processing, each with its own saved configuration. Jobs run sequentially in the background so you can queue a series of videos and let them run unattended.

| Feature | Description |
|---|---|
| **Add Job** | Adds the current target file and all current control settings as a new job entry. |
| **Save / Load Jobs** | Jobs are stored as JSON files in the `jobs/` folder and can be reloaded across sessions. |
| **Auto Swap** | When enabled, swapping begins automatically as soon as a target file is loaded. |
| **Keep Selected Input Faces** | Retains the loaded source face embeddings between jobs so you don't need to re-select them for each file. |
| **Swap Input Face Only Once** | Processes each source face's embedding only once per job rather than re-computing it for every frame. Speeds up processing on long videos. |

---

## 11. Presets

Presets save and restore all control panel settings as named JSON files stored in the `presets/` folder. They let you quickly switch between configurations — for example, a preset optimised for portrait photos versus one for action video.

- Save a preset by clicking the Save button and entering a name
- Apply a preset by double-clicking its name in the preset list
- The **Control Preset** toggle enables or disables automatically applying preset settings when a face card is selected
- Presets can be renamed or deleted via the right-click context menu

---

## 12. Video Timeline & Markers

The video timeline supports time-coded markers that let you apply different face card settings at different points in a video. Useful for scenes where the camera angle, lighting, or cast changes.

- Click **Add Marker** to insert a marker at the current playback position
- Each marker stores the currently active control settings
- **Previous / Next Marker** buttons jump between markers for quick navigation
- **Marker Save** commits the current settings to the selected marker
- **Track Markers on Video Seek** automatically moves to the nearest marker when you scrub the timeline

---

## 13. Recording & Output

### 13.1 Recording Controls

The recording toolbar contains Start, Stop, and Pause buttons for capturing the processed output. Output files are saved in the `outputs/` folder by default.

### 13.2 FFmpeg Output Options

| Option | Description |
|---|---|
| **Presets SDR / HDR** | FFmpeg encoding presets for standard- and high-dynamic-range output. Use the HDR preset only on HDR source material. |
| **Quality** | CRF-equivalent quality setting. Lower values produce larger, higher-quality files. |
| **Spatial AQ / Temporal AQ** | Adaptive quantisation options available with NVENC. Improve perceptual quality in flat areas and across time. |
| **Frame resize to 1920×1080** | Forces the output to 1080p resolution regardless of the source dimensions. |
| **Open Output Folder After Recording** | Automatically opens the output directory in your file explorer when recording stops. |

### 13.3 Playback Settings

| Setting | Description |
|---|---|
| **Playback FPS Override** | Sets a custom playback frame rate instead of reading it from the video file. |
| **Playback Buffering** | Enables frame buffering to smooth out playback on slower systems. |
| **Playback Loop** | Loops video playback continuously. |
| **Audio Playback Volume** | Controls the volume of audio during preview playback. |
| **Audio Start Delay** | Introduces a delay (in seconds) before audio begins, useful to compensate for sync issues. |

---

## 14. Settings

### 14.1 Performance

**Providers Priority** — selects the inference backend. The four options are:

**CUDA** — Runs models via ONNX Runtime on the Nvidia GPU using the CUDAExecutionProvider. Straightforward and broadly compatible. A good choice if TensorRT is not installed or if you want simpler setup.

**TensorRT** — Also uses ONNX Runtime but with the TensorrtExecutionProvider. On first use of each model, it builds an optimised engine cache in the `tensorrt-engines/` folder; subsequent runs load from cache and are noticeably faster than plain CUDA. The build step happens automatically and shows a progress dialog when it runs.

**TensorRT-Engine** — Bypasses ONNX Runtime entirely for supported models and loads pre-built `.trt` engine files produced by the Model Optimiser tool. Delivers the highest throughput of the three GPU options. Requires running the Optimiser first (via the launcher maintenance menu); if a pre-built engine isn't available for a given model, it falls back to ONNX Runtime automatically. Requires TensorRT 10.2.0 or later — if the installed version is older, the app will fall back to regular TensorRT automatically.

**CPU** — Runs without a GPU using only the CPUExecutionProvider. Works on any hardware but is significantly slower than the GPU options.

| Setting | Description |
|---|---|
| **Number of Threads** | Number of execution threads used during playback and recording. Reduce to 1 if you encounter VRAM issues or crashes. |
| **Resize Input Source** | Downscales the input resolution before processing to trade output quality for speed. |
| **Input Resolution Target** | The target resolution when Resize Input Source is enabled (540p, 720p, or 1080p). Aspect ratio is preserved. |

### 14.2 Face Recognition

| Setting | Description |
|---|---|
| **Recognition Model** | The ArcFace-based embedding model used to generate face identity vectors. This setting controls two distinct things. During **face detection and matching** — identifying which detected face in the frame corresponds to which face card — the model selected here is used directly. During the **swap itself**, the app automatically selects the correct ArcFace model based on the active swapper (Inswapper128, InStyleSwapper256, and DFM use Inswapper128ArcFace; SimSwap512 uses SimSwapArcFace; GhostFace variants use GhostArcFace; CSCS uses CSCSArcFace) regardless of what is selected here. In most cases the default is fine; changing this may affect how well face cards are matched to detected faces when using the Similarity Threshold. |
| **Swapping Similarity Type** | The alignment strategy used when computing face embeddings: **Optimal** (full warp via arcfacemap — default), **Pearl** (offset alignment), or **Opal** (standard similarity transform). Affects how closely the embedding captures the face geometry. |
| **Embedding Merge Method** | When multiple source images are combined into a single face card embedding, controls how their individual vectors are merged: **Mean** (average of all vectors) or **Median** (more robust to outlier images). |

### 14.3 Appearance

The **Theme** selector lets you choose from a set of built-in UI colour schemes: True-Dark, Dark, Dark-Blue, Light, Solarized-Dark, Solarized-Light, Dracula, Nord, and Gruvbox. Themes are applied immediately without restarting.

### 14.4 VR / 360° Mode

When working with VR180 or equirectangular 360° video, enable **VR180 Mode**. The app will unproject perspective crops for each face, process them, and stitch them back into the equirectangular image. The **VR180 Eye Mode** setting controls whether processing targets the left eye, right eye, or both eyes of a side-by-side VR180 frame.

### 14.5 Embedding Manager

The **Advanced Embedding Editor** (accessible from the main UI) is a tool for loading, organising, and managing saved face embedding files. It loads `.json` embedding files produced by the face card system, displays each stored identity as a named card, and lets you select, reorder, merge, rename, and save embeddings between sessions. Operations include loading files additively or as a replacement set, saving selected embeddings to a new file, drag-and-drop reordering, multi-select with Select All / Deselect All, sorting (Manual, Original, A-Z, Z-A), and a search filter. Undo/redo is supported via Ctrl+Z / Ctrl+Shift+Z. This is useful for maintaining libraries of reference embeddings across multiple projects.

---

## 15. Model Optimiser

VisoMaster Fusion has two separate model optimisation processes, which are often confused:

**ONNX Simplification** (`app/tools/optimize_models.py`) simplifies eligible ONNX model files using onnxsim (constant folding, dead node removal) and symbolic shape inference. It replaces the original `.onnx` files in `model_assets/` with leaner versions, backing up the originals to `model_assets/unopt-backup/`. This produces optimised ONNX files — not TensorRT engines. It can be run via the **Optimize Models (onnxsim)** action in the launcher's maintenance menu, which calls this script directly. The **Revert to Original Models** launcher action deletes the optimised files and re-downloads the originals from source.

**TensorRT Engine Building** is a separate process that converts ONNX models into hardware-specific `.trt` engine files (`app/processors/utils/engine_builder.py`). This happens automatically the first time a model is used with the **TensorRT-Engine** provider — it is not triggered by the ONNX Simplifier. The resulting engine files are stored in `tensorrt-engines/` and reused on subsequent runs. A progress dialog is shown during the build.

> **Note:** TensorRT engines are hardware-specific. An engine built on one GPU will not work on a different GPU model and must be rebuilt. See §14.1 for details on when each provider uses these files.

---

## 16. Tips & Best Practices

- Start with the default **Inswapper128** model and **Auto Resolution** to verify your setup before trying higher-quality but slower options
- Enable **Face Restorer** (GFPGAN v1.4) as a first step — it corrects most visible artefacts with minimal configuration
- Use the **Similarity Threshold** to target a specific person in a crowd. Set it high (80+) if only one face should be swapped
- For video, enable **ByteTrack face tracking** so the app keeps the correct source assigned to the correct person across cuts and occlusions
- When using the **Face Expression Restorer**, keep the Neutral Factor below 1.0 — the default of 0.30 is a reasonable starting point since the swapped face already carries some expression from the swap model itself
- If the swapped face looks blurry, try increasing Swapper Resolution or enabling Strength at 200%
- Save frequently used configurations as **Presets** so you can switch quickly between different target subjects or content types
- For batch processing, load all jobs into the **Job Manager** and let them run unattended
- For difficult multi-face scenes, process one face at a time — remove all other detected faces, record, then run the output back through for the next face

---

## 17. Glossary

| Term | Definition |
|---|---|
| **ArcFace** | A deep learning model that encodes a face image into a fixed-length identity vector (embedding). VisoMaster Fusion uses several ArcFace variants paired to specific swapper models: Inswapper128ArcFace, SimSwapArcFace, GhostArcFace, and CSCSArcFace. The correct variant is selected automatically during swapping; the UI setting controls the face matching pass. |
| **ByteTrack** | A multi-object tracking algorithm that assigns a consistent ID to each detected face across video frames. Ensures the correct face card settings follow the correct person through motion and occlusion. |
| **CFG Scale** | Classifier-Free Guidance scale. Used by the Ref-LDM Denoiser in DDIM mode to control how strongly the output adheres to the reference face features. Higher values increase adherence. |
| **CLIP** | OpenAI's vision-language model. Used by the Text Masking feature (§3.5.5) to identify and segment objects described in plain English — such as "glasses" or "hat" — from the face region. |
| **CSCS** | A face swap model that uses two separate embeddings (appearance and identity) for stronger likeness transfer on challenging angles. Uses its own CSCSArcFace recognition model. |
| **CUDA** | Nvidia's GPU compute platform. One of four inference backends in VisoMaster Fusion (§14.1). Runs models via the ONNX Runtime CUDAExecutionProvider. |
| **DDIM** | Denoising Diffusion Implicit Models. The iterative denoising mode used by the Ref-LDM Denoiser (§5). Produces more refined output than Single Step mode at greater processing cost. |
| **DFM** | DeepFaceLive Model. A custom pretrained face swap model format from the DeepFaceLive project. DFM files are placed in `onnxmodels/dfm_models/` and selected via the DeepFaceLive (DFM) swapper option. Uses Inswapper128ArcFace for recognition. |
| **Embedding** | A fixed-length numerical vector encoding a face's identity, produced by an ArcFace model. Multiple source images can be merged into a single embedding using Mean or Median merge (§14.2) for more consistent swap results. |
| **FaceParser** | A semantic segmentation model that labels each pixel of a face crop into anatomical classes: background, skin, hair, lips, eyes, nose, neck, etc. Used by the Face Parser Mask (§3.5.7), Mouth Fit & Align (§3.5.6), Tongue/Mouth Priority in the Occluder (§3.5.3), and the XSeg Mouth sub-option (§3.5.4). |
| **GFPGAN** | A GAN-based face restoration model. Available in standard (v1.4) and high-resolution (1024) variants as a restorer option (§4). Repairs compression artefacts, blurring, and detail loss on the swapped face crop. |
| **GhostFace** | A family of lightweight face swap models (v1, v2, v3) available in VisoMaster Fusion. All variants use GhostArcFace for recognition. |
| **InStyleSwapper** | Style-aware swap models (variants A, B, C) operating at 256 px, with an optional 512 px mode. Use Inswapper128ArcFace for recognition. |
| **Inswapper128** | The default face swap model. Fast and versatile, with configurable internal resolution (128–512 px via Swapper Resolution or Auto Resolution). Uses Inswapper128ArcFace. |
| **Landmark** | A keypoint detected on the face (e.g. corner of an eye, tip of the nose). Used for face alignment, crop warping, and expression transfer. VisoMaster Fusion supports landmark models detecting 5, 68, 3D-68, 98, 106, 203, or 478 points. |
| **LivePortrait** | A neural animation pipeline used by both the Face Expression Restorer (§6) and the Face Pose/Expression Editor (§7). Extracts motion keypoints from a driving face and applies them to the target. |
| **ONNX** | Open Neural Network Exchange. The model format used throughout VisoMaster Fusion to run AI models across different hardware backends (CUDA, TensorRT, CPU) without recompiling for each. |
| **Occluder** | The occlusion mask model (§3.5.3) that detects foreground objects covering the face — hands, glasses, microphones — so they are preserved in the final composite rather than replaced by the swap. |
| **Ref-LDM** | Reference Latent Diffusion Model. The UNet-based model used by the Denoiser (§5). Uses the source face as a reference to denoise and reconstruct texture on the swapped face. Supports Single Step and DDIM modes. |
| **RetinaFace** | A single-stage face detector from the InsightFace project. The default detector in VisoMaster Fusion; high accuracy across a wide range of face sizes and orientations. |
| **SCRFD** | Sample and Computation Redistribution for Face Detection. A compact InsightFace detector (SCRFD-2.5G variant) offering faster inference than RetinaFace with competitive accuracy. |
| **SimSwap** | A face swap model operating natively at 512 px. Uses SimSwapArcFace for recognition. |
| **TensorRT** | Nvidia's inference optimisation library. When selected as the provider (§14.1), ONNX Runtime uses the TensorrtExecutionProvider and automatically builds an engine cache in `tensorrt-engines/` on first use. |
| **TensorRT-Engine** | Pre-built `.trt` engine files produced by the TRT engine building process (triggered on first use of the TensorRT-Engine provider). Bypasses ONNX Runtime entirely for supported models. Requires TensorRT 10.2.0 or later; distinct from the ONNX Simplifier in the Model Optimiser (§15). |
| **VR180** | A 180-degree equirectangular video format for VR headsets. VisoMaster Fusion can process VR180 content by unprojecting perspective crops per eye, applying the swap, and stitching back into the equirectangular frame. |
| **XSeg** | An occlusion segmentation model trained to identify foreground objects covering the face. Used by the DFL XSeg Mask (§3.5.4) as an alternative to the Occluder. |
| **YOLOv8** | A fast object detection architecture. Used here as the YoloFace8n face detector — the fastest of the four available detector options, suited to real-time and webcam use. |
| **YuNet** | A lightweight face detector from the OpenCV Model Zoo. Very low compute footprint; suited to CPU inference and resource-constrained scenarios. |
