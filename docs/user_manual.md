# VisoMaster Fusion - User Manual

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [System Requirements](#2-system-requirements)
3. [Getting Started](#3-getting-started)
4. [Face Swap Tab](#4-face-swap-tab)
5. [Face Restoration](#5-face-restoration)
6. [Denoiser](#6-denoiser)
7. [Face Expression Restorer](#7-face-expression-restorer)
8. [Face Pose / Expression Editor](#8-face-pose--expression-editor)
9. [Frame Enhancers](#9-frame-enhancers)
10. [Face Detection & Tracking](#10-face-detection--tracking)
11. [Job Manager](#11-job-manager)
12. [Presets](#12-presets)
13. [Video Timeline & Markers](#13-video-timeline--markers)
14. [Recording & Output](#14-recording--output)
15. [Settings](#15-settings)
16. [Model Optimiser](#16-model-optimiser)
17. [Tips & Best Practices](#17-tips--best-practices)
18. [Glossary](#18-glossary)

---

## 1. Introduction

VisoMaster Fusion is a desktop application for AI-powered face swapping, enhancement, and editing on images, videos, and live webcam feeds. It provides a full pipeline for face detection, swapping, masking, restoration, and expression editing through a graphical interface built with PySide6.

The application supports multiple AI inference backends (CPU, CUDA, TensorRT, TensorRT-Engine) and includes a batch job manager for processing multiple files unattended.

---

## 2. System Requirements

**Operating System:** Windows 10 or Windows 11 (64-bit).

**Nvidia GPU:** At least 6 GB of VRAM for basic use; 8-12 GB or more is recommended when running multiple models simultaneously (swapper + restorer + denoiser). The app can run on CPU but is significantly slower.

**Nvidia GPU Driver:** A recent driver supporting CUDA 12.9. If you see provider errors at launch, update your driver from the Nvidia website.

**Internet connection:** Required on first run only to download tools, dependencies, and model files. After setup, the app runs fully offline.

**Disk space:** 20-30 GB free space. The dependency install (PyTorch, TensorRT, cuDNN, etc.) is several gigabytes, and AI model files add several more.

### 2.1 Installation (Portable)

Execute `Start_Portable.bat` to begin. On first run it automatically downloads and installs everything needed: Python 3.11, Git, FFmpeg, and all AI libraries (PyTorch 2.8 + CUDA 12.9, TensorRT 10.9, cuDNN 9.13, ONNX Runtime).

No existing software on your system is modified. Everything is installed inside the application folder and is self-contained. This means you can move the entire folder to another drive or run it from a USB drive without reinstalling anything.

After the dependency install completes, the model downloader runs automatically to fetch the AI model files. On subsequent launches, `Start_Portable.bat` skips straight to the launcher.

---

## 3. Getting Started

### 3.1 Launcher

When you first open VisoMaster Fusion, a launcher window appears. The home screen shows the current build commit, last update timestamp, and live status pills that flag any detected issues, such as pending git updates, missing models, dependency changes, or modified tracked files.

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

### 3.2 Main Window Layout

The main window is divided into these main areas:

- **Left panel** - source face input and face cards for assigning reference faces
- **Centre** - media preview with video playback controls and a timeline
- **Right panel** - tabbed settings panels (Face Swap, Face Editor, Denoiser, Settings, Presets)
- **Top toolbar** - file open/save, recording controls, and preset management
- **Jobs area** - saved jobs for queued batch processing

### 3.3 Supported Media Types

- **Images** - JPG, PNG, and other common formats
- **Videos** - MP4 and most container formats supported by FFmpeg
- **Webcam / Live input** - real-time processing from a connected camera

### 3.4 Loading Media

Load target media from a folder, individual files, or available webcams. Load source faces separately through the Input Faces area on the left. Saved embeddings can also be loaded separately. Each source face becomes a face card that can be assigned to one or more detected faces in the target media.

---

### 3.5 Processing Pipeline Overview

VisoMaster Fusion processes each frame through a fixed sequence of stages. Understanding this order helps explain the **Pipeline Position** controls shown in several tabs:

1. **Face Detection** - Detects and tracks faces in the input frame (Section 10)
2. **Face Swap** - Applies the selected swap model to each detected face (Section 4)
3. **Masks** - Composites the swapped face back into the frame using the configured masks (Section 4.5)
4. **Denoiser** (optional, position configurable) - Reduces noise and reconstructs texture (Section 6)
5. **Face Restorer 1** (optional) - Sharpens and corrects artifacts on the face crop (Section 5)
6. **Denoiser** (optional, after first restorer position) - Second denoiser pass if configured (Section 6)
7. **Face Expression Restorer** (optional, position configurable) - Transfers expression and pose from the driving face (Section 7)
8. **Face Restorer 2** (optional, can run at end of pipeline) - Second restoration pass (Section 5)
9. **Denoiser** (optional, after all restorers position) - Third denoiser pass if configured (Section 6)
10. **Face Editor / Makeup** - Manual pose and expression adjustments (Section 8)
11. **Frame Enhancer** (optional) - Full-frame upscaling or colorization (Section 9)

The Denoiser, Face Expression Restorer, Second Restorer, and Face Pose/Expression Editor each expose a **Pipeline Position** control that lets you choose where in this chain they run. These are covered in detail in their respective sections.

---

## 4. Face Swap Tab

The Face Swap tab contains the core swapping controls. These settings apply per face card, so different people in the same clip can use different configurations.

### 4.1 Swapper Model

Choose the AI model used to perform the face transfer. Each model has different characteristics:

| Model | Description |
|---|---|
| **Inswapper128** | The default model. Fast, versatile, and works well at multiple resolutions. Recommended for most use cases. |
| **InStyleSwapper256 (A / B / C)** | Higher-resolution swappers based on the Inswapper architecture and trained using a custom technique. Operate at 256 px and tend to preserve skin tone, lighting, and style cues from the target scene. Each variant produces slightly different results. |
| **SimSwap512** | Operates natively at 512 px. Good identity preservation and fine detail. |
| **GhostFace-v1 / v2 / v3** | A family of lightweight swappers. v2 and v3 generally outperform v1 in sharpness and identity fidelity. |
| **CSCS** | Combines two embeddings (appearance + identity) for stronger likeness. Best for challenging angles. |
| **DeepFaceLive (DFM)** | Uses custom pretrained DFM model files placed in the `onnxmodels/dfm_models` folder. Supports AMP Morph Factor (1-100, default 50) and RCT color transfer. The **Maximum DFM Models to use** setting (1-5, in Settings) controls how many DFM models are held in VRAM simultaneously - increase it if you are switching between multiple DFM files frequently and have the VRAM to spare. |

### 4.2 Swapper Resolution

Available when using Inswapper128. This sets the internal face-crop resolution used during swapping (128, 256, 384, or 512 px). Higher values usually give more detail but are slower. Enable **Auto Resolution** to let the app choose based on the detected face size.

When using **InStyleSwapper256 Version A**, **InStyleSwapper256 Version B**, or **InStyleSwapper256 Version C**, a separate **512 Resolution** toggle is available for that selected variant.

### 4.3 Similarity Threshold

A filter (1-100, default 60) that controls how closely a detected face in the target must match your source face card before the swap is applied. Higher values are stricter, so only very close matches are swapped. This is useful when multiple people appear on screen and you only want to swap one of them.

### 4.4 Swap Strength & Likeness

| Feature | Description |
|---|---|
| **Strength** | Runs additional swap iterations on the result to deepen the effect. The Amount slider goes up to 500% (5x passes). 200% is a common sweet spot. Setting it to 0% disables swapping entirely but still allows the rest of the pipeline to run on the original image. |
| **Mode 2 (Anti-Drift & Texture)** | An advanced iteration mode using phase correlation and frequency separation. Reduces drift across many passes and better preserves skin texture. |
| **Face Likeness** | Directly adjusts how much the result resembles the source face versus the target. Negative values lean toward the target; values above 1.0 push harder toward the source. The range is -1.0 to 3.0. |
| **Face Keypoints Replacer** | Transfers the spatial landmark layout of the target face onto the source before swapping, helping the result fit the target's head pose and geometry. An Amount slider (0.00-1.00) controls how strongly the target keypoints are applied. |
| **Pre-Swap Sharpness** | Sharpens the original face before it enters the swap model (range 0.0-2.0, default 1.0). Can improve edge detail but may interfere with Auto Face Restorer. |

### 4.5 Masks

Masks control which pixels from the swapped face are blended back into the original frame. Multiple mask types can be enabled at the same time, and their results are combined before the final blend.

The **Mask View Selection** dropdown (`swap_mask`, `diff`, `texture`) lets you inspect the active composite mask in the preview while working. This is useful for diagnosing blending issues before committing to a full render.

The **Mask Dilatation Type** setting (in Settings) controls the algorithm used when growing mask regions. The available options are `conv`, `pool`, and `iter_pool`.

#### 4.5.1 Border Mask

A rectangular mask with adjustable Top, Bottom, Left, Right, and Blur sliders. Anything outside the mask boundary fades back to the original image. Useful for hiding stray pixels at the hairline or chin.

#### 4.5.2 Profile Angle Mask

A separate mask that automatically fades the far side of the face when the head is turned in profile view, hiding distortions at the edge of the swap. Controls include an **Angle Threshold** (0-90 degrees, the head turn angle at which fading begins - lower values trigger it sooner) and a **Fade Strength** slider (0-100) that controls the intensity of the gradient.

#### 4.5.3 Occlusion Mask

Detects objects covering the face - such as a hand, glasses, or a microphone - and cuts them out of the swap so they appear naturally in the final composite. An **Occluder Size** slider (-100 to 100) grows or shrinks the detected region. The **Tongue/Mouth Priority** toggle uses FaceParser to prevent the tongue or inner mouth from being accidentally erased when growing the mask. A shared **Occluder/DFL XSeg Blur** slider controls edge softness for both the Occluder and XSeg masks when either is active.

#### 4.5.4 XSeg Mask

A second occlusion method using a dedicated XSeg segmentation model. Provides an independent mask channel that can be blended with the Occluder. The **Mouth/Lips Protection** toggle prevents XSeg from masking out the inner mouth area (useful for open mouths). A **Size** slider (-20 to 20) grows or shrinks the region. Includes a **XSeg Mouth** sub-option that applies a second XSeg pass specifically to the mouth region, with its own Size, Blur, and FaceParser-based grow sliders for the Mouth, Upper Lip, and Lower Lip.

#### 4.5.5 Text Masking

Uses the CLIP vision-language model to identify objects described in plain English (e.g. "glasses", "hat", "hand") and cuts them from the swap. Enable **Text Masking**, type one or more comma-separated terms into **Text Masking Entry**, and press Enter. Increase **Amount** to make the segmentation more aggressive.

#### 4.5.6 Mouth Fit & Align

Repositions and scales the original mouth region to fit cleanly inside the swapped face without distorting its shape. Can be used independently of FaceParser. Options include: **Use Original Mouth** (uses the original face's mouth as the reference rather than the swap), **Paste After Restorer** (applies the mouth mask after the restorers rather than before), **Smart Sharpen (USM)** (edge-aware unsharp masking to sharpen teeth and lip edges without adding noise to surrounding skin), and a **Mouth Zoom** slider (0.90-1.20).

#### 4.5.7 Face Parser Mask

Uses a semantic face parsing model to produce a pixel-accurate mask over the face region. Each parsed area has an independent grow slider (0-30) that controls how much of that region is included in the swap. Available regions are: **Background**, **Face**, **Left/Right Eyebrow**, **Left/Right Eye**, **Eyeglasses**, **Nose**, **Mouth** (inner mouth and tongue), **Upper Lip**, **Lower Lip**, **Neck**, and **Hair**. Additional controls include **Parse at Pipeline End** (runs the mask after the full pipeline rather than before restorers), **Mouth Inside** (keeps the parse inside the mouth boundary), **Background Blur**, **Face Blur**, and **Face Blend** sliders for edge softening and blending.

#### 4.5.8 Restore Eyes

Blends the original eyes back into the swapped face using a configurable elliptical mask. Controls include **Eyes Blend Amount** (balance between original and swapped eyes), **Eyes Feather Blend** (edge softness of the blend), **Eyes Size Factor** (overall mask size - reduce for smaller or distant faces), **X/Y Eyes Radius Factor** (shape the mask from circular to oval), and **X/Y Eyes Offset** and **Eyes Spacing Offset** sliders for precise positioning.

#### 4.5.9 Restore Mouth

Blends the original mouth back into the swapped face, similar in structure to Restore Eyes. Includes a **Mouth Blend Amount** slider for controlling the mix between original and swapped mouth regions.

### 4.6 Textures and Colors

The Face Swap tab also includes a **Textures and Colors** group for refining how the swapped face blends with the original footage or image.

| Feature | Description |
|---|---|
| **Differencing** | Builds a transfer mask from the difference between the swap and the original face. Includes controls such as **Difference Lower Limit**, **Difference Upper Limit**, **Lower Strength**, **Upper Strength**, and **Mask Blur Amount**. |
| **Transfer Texture** | Transfers texture detail from the original face into the swap. Includes controls for **Texture Strength Amount**, **Texture Gamma adjust**, **Texture Contrast adjust**, **CLAHE**, feature-exclusion masks, and background exclusion. |
| **AutoColor Transfer** | Applies an automatic color transfer pass to better match tones between the source and target. |
| **Enable Ending Color Transfer** | Applies a final color transfer pass later in the pipeline. |
| **Color Adjustments** | Manual controls for **Red**, **Green**, **Blue**, **Brightness**, **Contrast**, **Saturation**, **Sharpness**, **Hue**, and **Gamma**. |
| **Noise** | Adds noise to the swapped face. |
| **JPEG Compression** | Applies controlled JPEG-style compression to the swapped face and blends it back in. |
| **MPEG Compression** | Applies a block-shift style compression effect with controls such as **Block Size**, **Shift Maximum**, and **Blend Amount**. |

### 4.7 Face Landmarks Correction

The **Face Landmarks Correction** group lets you manually adjust the detected keypoints used during alignment and blending.

| Feature | Description |
|---|---|
| **Face Adjustments** | Enables coarse landmark adjustment controls such as **Keypoints X-Axis**, **Keypoints Y-Axis**, **Keypoints Scale**, and **Face Scale Amount**. |
| **5 - Keypoints Adjustments** | Enables independent adjustment of the five main points: left eye, right eye, nose, left mouth, and right mouth. |

### 4.8 Blend Adjustments

The **Blend Adjustments** group contains final blending controls:

| Feature | Description |
|---|---|
| **Final Blend** | Enables a final blend pass at the end of the pipeline. |
| **Final Blend Amount** | Controls the strength of that final blend pass. |
| **Overall Mask Blend Amount** | Adjusts the combined blending distance for the active masks, excluding the border mask. |

### 4.9 Face Re-Aging

The **Face Re-Aging** group changes the apparent age of the assigned source face before swapping.

| Control | Description |
|---|---|
| **Enable Face Re-Aging** | Turns on re-aging for the assigned source face. |
| **Source Age** | Approximate current age of the source face. |
| **Target Age** | The age to transform the source face toward. |
| **Apply** | Recomputes the transformed face data after changing the age sliders. |

---

## 5. Face Restoration

After swapping, one or two face restorer models can be applied to sharpen detail and correct AI artefacts. Restorers work on the aligned face crop and blend the result back into the frame at a configurable ratio.

### 5.1 Restorer Models

| Model | Description |
|---|---|
| **GFPGAN-v1.4** | Fast and versatile all-round restorer. A good default choice for general face enhancement. |
| **GFPGAN-1024** | Higher-resolution GFPGAN variant designed for more detailed restoration, typically at a higher processing cost. |
| **CodeFormer** | Quality-focused restorer with a **Fidelity Weight** slider (`0-1`). Lower values tend to produce higher-quality, more heavily restored results, while higher values stay closer to the original face. |
| **GPEN-256 / 512 / 1024 / 2048** | GPEN blind face restoration models at different internal resolutions. Higher-resolution variants can recover more detail, but are generally slower and more demanding. |
| **RestoreFormer++** | Transformer-based face restorer designed to balance fidelity and realism, using spatial attention to model facial context and reconstruction priors. |
| **VQFR-v2** | Vector-quantised face restorer focused on texture recovery, with user-controlled balance between restoration quality and fidelity. |

### 5.2 Restorer Controls

| Control | Description |
|---|---|
| **Alignment** | How the face crop is positioned for restoration. **Original** restores directly on the existing swap crop - the default and fastest option. **Blend** re-warps the crop to a standard ArcFace-aligned position before restoring, which can improve results on faces that are not well-centered. **Reference** aligns to the detected target face landmarks instead, useful when the swap geometry differs significantly from the source. |
| **Blend** | The mix ratio (0-100%) between the restored face and the raw swap output. 100% uses only the restorer result. |
| **Auto Restore** | Automatically adjusts the blend amount per frame based on a sharpness analysis. Useful when face size or motion varies across a video. Includes an **Adjust Sharpness** slider (-60 to 60) to offset the sharpness threshold used for the calculation. |
| **Sharpness Mask** | Within Auto Restore, uses a per-pixel sharpness map to apply stronger blending only where the image is soft. Includes a **Mask Adjust** slider (0.00-1.00, default 1.00) that scales the min/max blend range around the base auto-blend value - lower values narrow the adjustment range for subtler automatic variation. |
| **Second Restorer** | A second, independent restorer pass with its own model, alignment, and blend settings. The **Apply at End** toggle makes it run at the very end of the full pipeline - after the Face Editor - to recover sharpness lost by later processing steps. Also supports its own Auto Restore and Sharpness Mask sub-controls. |

---

## 6. Denoiser

The Denoiser uses a reference-guided latent diffusion model (Ref-LDM) to reduce noise and reconstruct fine skin texture on the face crop. Unlike the face restorers (Section 5), which sharpen and clean up a completed swap, the Denoiser works inside the latent space of a diffusion model, using the source face as a reference to guide texture reconstruction. It can be enabled at up to three points in the pipeline independently - **before the restorers**, **after the first restorer**, and **after all restorers** - each with its own settings.

**Global controls** (shared across all passes):

| Setting | Description |
|---|---|
| **Exclusive Reference Path** | Forces the UNet to attend only to reference key/value features, maximising focus on the source face style. Enabled by default. |
| **Base Seed** | Fixed random seed (1-999) for reproducible noise patterns across all frames and all denoiser passes. |

**Per-pass controls** (available for each of the three pipeline positions):

| Setting | Description |
|---|---|
| **Denoiser Mode** | **Single Step (Fast)** adds and removes a controlled amount of noise in one pass - fast and subtle. **Full Restore (DDIM)** runs full iterative diffusion over multiple steps for more detail at greater cost. |
| **Single Step Timestep (t)** | Available in Single Step mode. Controls how much noise is injected and therefore removed. Lower values are more conservative (range: 1-500). |
| **DDIM Steps** | Available in Full Restore mode. Number of denoising iterations - more steps produce a more refined result (range: 1-300). |
| **CFG Scale** | Available in Full Restore mode. How strongly the denoiser follows the reference features. Higher values increase adherence to the source appearance (range: 0.0-10.0). |
| **Latent Sharpening** | Applies sharpening directly inside the latent space before decoding. A value around 0.15 is a reasonable starting point (range: 0.0-2.0). |

---

## 7. Face Expression Restorer

The Face Expression Restorer adds movement back into a swapped face. Without it, the result can look stiff because the swap mainly reflects the expression of the source face you loaded, not the ongoing expression changes of the person in the target media.

This feature uses the original target face as the *driving face* and transfers its movement onto the swapped result. In practice, that means blinks, gaze changes, mouth movement, and other expression changes can follow the target more closely instead of staying frozen.

It is best to think of this as a correction pass, not a full rebuild of the face. The swap already carries some expression information on its own, and the restorer adjusts that result to make it feel more natural. That is why **Neutral Factor** defaults to 0.30 instead of 1.0. The restorer uses the LivePortrait pipeline internally and is most useful on video, though it can also help on images when the target expression matters.

### 7.1 Pipeline Position

Because the restorer changes face geometry, its pipeline position affects the final sharpness. The **Pipeline Position** dropdown controls this:

- **Beginning** - Runs on the raw swap output, before any face restorers. Both restorers then run afterward, which can sharpen the result but may soften the expression slightly.
- **After First Restorer** - A good default. The face is cleaned up first, then expression is applied, and a second restorer (if enabled) can recover any sharpness lost during the warp.
- **After Second Restorer** - Runs last, giving the expression model the sharpest possible input. No sharpening follows, so this is best used when you are not running a second restorer.

As a general rule, running it later in the chain tends to produce the most natural-looking results.

### 7.2 Shared Controls (Both Modes)

These controls are always visible when the restorer is enabled.

| Control | Description |
|---|---|
| **Mode** | Choose between **Simple** and **Advanced**. Simple animates the eyes and lips together with one expression slider. Advanced breaks the face into four independent regions with their own toggles, intensity controls, and sub-options. |
| **Crop Scale** | Controls how wide a crop of the face is passed to the LivePortrait model. If you see distortion at the edges of the face crop, increase this slightly to give the model more context. |
| **VY Ratio** | Fine-tunes the vertical centering of the crop window. This usually does not need adjustment, but it can help if the face sits unusually high or low in frame. |
| **Neutral Factor** | Controls how much expression to restore. Lower values are subtler. Higher values apply more of the driving face's expression, but can start to look exaggerated if pushed too far. |

### 7.3 Simple Mode

Simple mode animates the eyes and lips together using a single **Expression Factor** slider. Motion is calculated relative to the source face's starting pose, which helps keep the result stable and reduces drift.

**Expression Factor** (0.0-3.0, default 1.0) - The overall strength of expression transfer. 1.0 applies the driving face's expressions at full intensity. Lower values are more subtle; values above 1.5-2.0 can start to look exaggerated.

**Animation Region** - Controls which facial regions are animated:
- `all` - Both eyes and lips are animated. This is the typical choice for most footage.
- `eyes` - Only the eye region (blinking, squinting, gaze direction). Useful when the mouth already looks natural but the eyes feel flat.
- `lips` - Only the lip and mouth region. Useful for speech-driven content where the eyes are already fine.

> Note: Simple mode does not animate brows or general face features (jaw, cheeks, contour). For those, use Advanced mode.

**Normalize Lips** (toggle, default on) - Guards against extreme mouth-open ratios from the driving video producing distorted results. If the source video has someone opening their mouth very wide, this prevents that extreme ratio from being applied literally to the swapped face. The **Normalize Lips Threshold** (0.10-1.00, default 0.03) sets the lip-open ratio at which clamping activates - a lower value means it kicks in sooner.

### 7.4 Advanced Mode

Advanced mode gives independent control over four facial regions. Each region has its own enable toggle, expression factor, and optional sub-controls. Use Advanced mode when Simple mode gives uneven results or when you want brow and general face-feature animation.

**Micro-Expression Boost** (0.80-1.50, default 1.00) - Applies only in Advanced mode when **Relative Position** is active on any region. Subtle expressions - slight squints, faint smirks, small brow raises - can get compressed during the swap and normalization steps. This slider amplifies those small movements before they are applied, using a smart scaling that boosts only small signals while leaving strong expressions largely unchanged. Values around 1.10-1.20 are a good starting point for recovering lost subtlety without looking exaggerated.

#### 7.4.1 Restore The Eyes

Restores eye movement from the driving face, including blinking, squinting, and gaze direction. The eye keypoints used by the model explicitly encode both lid state and eyeball direction, so enabling this region will cause the swapped face's eyes to follow where the original person was looking.

**Relative Position** - Computes the eye animation relative to the initial pose of your source reference image, rather than in absolute coordinates. This reduces geometric distortion on faces that aren't perfectly frontal. Recommended for most use cases.

**Eyes Expression Factor** (0.0-3.0, default 1.0) - How strongly the driving face's full eye expression (blink, squint, gaze shift) is applied to the swapped result. Lower values blend more toward the swap's existing eye state.

**Retargeting Eyes** (default off) - A secondary precision pass that measures the actual eye open/close ratio of the driving face and applies it directly, rather than relying on the motion keypoints alone. Useful when you need the eye openness to match the driver more exactly.
- **Eyes Multiplier** (0.0-2.0, default 1.0) - Scales the strength of the retargeting correction. Below 1.0 is gentler; above 1.0 exaggerates it.
- **Normalize Eyes** (default on, under Retargeting Eyes) - Prevents the retargeting from producing unnaturally wide-open or clamped-shut eyes. **Eyes Threshold** (0.10-1.00, default 0.40) is the open ratio above which a stricter normalization function is used. **Max Open Ratio** (0.00-1.00, default 0.45) hard-caps the maximum eye openness - values below 0.5 tend to look the most natural on swapped faces.

#### 7.4.2 Restore Brows

Animates eyebrow movement (raise, furrow) from the driving face, independently of the eyes. Brows contribute significantly to conveying emotion, and are only available in Advanced mode.

**Relative Position** - Anchors the brow animation to the source face's initial pose to reduce geometric drift.

**Brows Expression Factor** (0.0-3.0, default 1.0) - Strength of brow movement transfer. Because the swapped face's brow geometry often differs from the driver's, values slightly below 1.0 (for example, 0.7-0.8) frequently look more natural.

#### 7.4.3 Restore The Lips

Animates lip and mouth movement from the driving face - speech, smiling, and mouth open/close.

**Relative Position** - Anchors the lip animation to the source face's initial pose. Particularly useful when the driving video has significant head movement, as it prevents the mouth from appearing to drift or slide around.

**Lips Expression Factor** (0.0-3.0, default 1.0) - Scales the strength of lip movement transfer.

**Retargeting Lips** (default off) - A precision pass that directly matches the lip open/close ratio of the driving face, similar to Retargeting Eyes but for the mouth.
- **Lips Multiplier** (0.0-2.0, default 1.0) - Strength of the lip retargeting correction.
- **Normalize Lips** (default off in Advanced mode) - Clamps extreme lip ratios. **Lips Threshold** (0.00-0.20, default 0.03) sets the point at which clamping activates.

#### 7.4.4 Restore General Face Features

Handles the broader face geometry not covered by eyes, brows, or lips - jaw movement, cheek shape, face contour, and head scale. This region is important for making overall head motion and large expressions (wide smiles, jaw drops) look natural, and is only available in Advanced mode.

**Relative Position** - Strongly recommended here. Absolute animation of the outer face shape and contour tends to produce visible warping, especially on non-frontal faces.

**General Expression Factor** (0.0-3.0, default 1.0) - Overall strength of general face feature animation.

The toggles below control which landmark groups are included in this region. All are enabled by default when General is turned on - you can disable individual ones to remove any region that looks problematic on a specific face.

- **Include Nose** - Subtle nose tip and bridge movement (nose flare).
- **Include Jaw** - Lower jaw and chin movement. Important for natural-looking speech and large mouth openings.
- **Include Cheek** - Lower cheek shape changes. Contributes to the puffing and widening that happens during smiles.
- **Include Contour** - The outer face silhouette. Drives head-shape changes and profile-view compression.
- **Include Head Top** - Upper forehead and top-of-head indices. Contributes to overall head scale and forehead movement.

### 7.5 Auto Mouth Expression

**Enable Auto Mouth Expression** automatically activates mouth-expression controls when mouth movement is detected in the scene.

| Control | Description |
|---|---|
| **Confidence Threshold** | Minimum detection confidence required before auto-mouth activation begins. |
| **EMA Smoothing** | Smooths the detection confidence over time to reduce rapid on/off changes. |
| **Expression Strength** | Base strength applied when auto-mouth is active. |
| **Animation Region** | Chooses whether auto-mouth affects `lips` only or `all`. |
| **Normalize Lips** | Enables lip-ratio normalization for the auto-mouth workflow. |
| **Mouth Parser / Upper Lip Parser / Lower Lip Parser** | Override the Face Swap parser dilation values for the mouth region while auto-mouth is active. |

---

## 8. Face Pose / Expression Editor

The Face Pose/Expression Editor lets you directly adjust the swapped face's pose and expression using sliders, without needing a driving video. It uses the LivePortrait pipeline internally.

### 8.1 Editor Setup

Before using the pose and expression sliders, you can configure how the editor runs:

| Control | Description |
|---|---|
| **Pipeline Position** | Chooses where the editor runs in the pipeline: **Beginning**, **After First Restorer**, **After Texture Transfer**, or **After Second Restorer**. |
| **Crop Scale** | Controls how much surrounding face area is included in the editor crop. |
| **VY Ratio** | Adjusts the vertical positioning of that crop. |
| **Blur Amount** | Adds blur to the edited result. |
| **Enable Face Pose/Expression Editor** | Turns the editor on or off. |
| **Face Editor Type** | Currently supports **Human-Face**. |

### 8.2 Head Pose

| Control | Description |
|---|---|
| **Head Pitch** | Tilts the face up or down (nodding motion). |
| **Head Yaw** | Rotates the face left or right (turning motion). |
| **Head Roll** | Tilts the head sideways. |
| **X / Y / Z-Axis Movement** | Translates the face along the horizontal, vertical, or depth axis. |

### 8.3 Eye & Brow Controls

| Control | Description |
|---|---|
| **Eyes Open/Close Ratio** | Opens or closes the eyes on a continuous scale. |
| **Eye Wink** | Triggers a wink on one eye. |
| **EyeBrows Direction** | Raises or lowers the eyebrows. |
| **EyeGaze Horizontal / Vertical** | Redirects the gaze direction without moving the head. |

### 8.4 Mouth & Lip Controls

| Control | Description |
|---|---|
| **Lips Open/Close Ratio** | Opens or closes the mouth. |
| **Mouth Pouting** | Pushes the lips forward into a pout. |
| **Mouth Pursing** | Tightens and narrows the lips. |
| **Mouth Grin** | Widens the mouth into a grin. |
| **Mouth Smile** | Curves the corners of the mouth into a smile. |

### 8.5 Makeup

AI-powered makeup is applied using the FaceParser model to identify facial regions, then colour-blended on top of the image. Each area has independent Red/Green/Blue colour sliders and a Blend Amount (0 = original colour, 1 = full target colour).

| Area | Description |
|---|---|
| **Face Makeup** | Colors the skin on the face - cheeks, forehead, and nose bridge - excluding hair, eyebrows, eyes, and lips. |
| **Hair Makeup** | Colours the hair region. |
| **EyeBrows Makeup** | Colours the eyebrows. |
| **Lips Makeup** | Colours the lips. |

---

## 9. Frame Enhancers

Frame Enhancers improve the quality of the entire output frame, not just the face region. They are applied as a post-processing step.

### 9.1 Upscaling Models

| Model | Description |
|---|---|
| **RealEsrgan-x2-Plus / RealEsrgan-x4-Plus** | AI super-resolution at 2x or 4x scale. Excellent general-purpose upscalers for photos and videos. |
| **BSRGan-x2 / BSRGan-x4** | Blind super-resolution models. Good at recovering fine detail on compressed or blurry inputs. |
| **UltraSharp-x4** | Optimized for sharpness and edge clarity at 4x scale. |
| **UltraMix-x4** | A blended upscaler model balancing sharpness and naturalness. |
| **RealEsr-General-x4v3** | A general-purpose variant of RealESRGAN tuned for a wide range of degradation types. |

### 9.2 Colourisation Models

| Model | Description |
|---|---|
| **DeOldify-Artistic** | Colourises black-and-white footage with a painterly, vibrant style. |
| **DeOldify-Stable** | Colourises with a more conservative, consistent style suited to historical photos. |
| **DeOldify-Video** | A temporal-aware variant of DeOldify optimised for video to reduce colour flickering. |
| **DDColor-Artistic** | Modern deep-learning colouriser with rich, saturated colours. |
| **DDColor** | Standard DDColor model offering natural-looking colourisation. |

---

## 10. Face Detection & Tracking

### 10.1 Face Detector Models

The app uses ONNX-based detectors to locate faces in each frame before swapping. The active model is selected in the Settings tab.

| Model | Description |
|---|---|
| **RetinaFace** | A single-stage face detector from the InsightFace project (CVPR 2020). High accuracy across a wide range of face sizes and orientations. The default and generally recommended choice. |
| **SCRFD** | Sample and Computation Redistribution for Face Detection (ICLR 2022, InsightFace). Designed for an efficient accuracy-to-compute trade-off. The variant used here (SCRFD-2.5G) targets a 2.5 GFlop budget - faster than RetinaFace with competitive accuracy. |
| **Yolov8** | A YOLOv8-based face detector (YoloFace8n). Fastest of the four options. Good choice for real-time or webcam use where speed matters more than peak accuracy. |
| **Yunet** | A lightweight millisecond-level face detector developed by Shiqi Yu and distributed via the OpenCV Model Zoo. Very low compute footprint; well-suited to CPU inference and resource-constrained scenarios. Notable for handling side-on and partially occluded faces well. |

| Setting | Description |
|---|---|
| **Detect Score** | Minimum confidence threshold for a detection to be accepted. Lower values catch more faces but may produce false positives. |
| **Max Faces to Detect** | Limits how many faces are processed per frame. Useful for performance when only one or two faces are relevant. |
| **Detection Interval** | Runs face detection only every N frames and reuses the result in between. Reduces CPU/GPU load on high-frame-rate video. |
| **Auto Rotation** | Rotates the input frame to the detected face's upright orientation before processing. |
| **Manual Rotation** | Enables a fixed detector rotation instead of relying only on auto rotation. |
| **Rotation Angle** | Sets the fixed rotation angle used when Manual Rotation is enabled. |
| **Enable KPS Smoothing** | Applies temporal smoothing to facial keypoints to reduce jitter. |

### 10.2 ByteTrack Face Tracking

When enabled, ByteTrack assigns a consistent ID to each face across frames. This allows the app to apply the correct face card settings to the right person even when faces briefly leave frame or overlap.

| Setting | Description |
|---|---|
| **Track Threshold** | Minimum detection score for a new track to be initialised. |
| **Match Threshold** | How closely a detection must match an existing track to be linked to it. |
| **Track Buffer (Frames)** | How many frames a track is kept alive after the face disappears before it is discarded. |
| **Show ByteTrack Bounding Boxes** | Shows the tracked ByteTrack boxes separately from the raw detector boxes. |

---

## 11. Job Manager

The Job Manager lets you save the current workspace as a job, reload saved jobs, and batch-process multiple jobs in sequence.

| Feature | Description |
|---|---|
| **Save Job** | Saves the current workspace as a job entry. |
| **Load Job** | Loads a saved job back into the main workspace. |
| **Delete Job** | Removes selected jobs by moving them to the Recycle Bin. |
| **Process All** | Processes every saved job in the queue. |
| **Process Selected** | Processes only the selected jobs. |
| **Use job name for output file name** | When saving a job, uses the job name as the output filename. |
| **Output File Name** | Lets you choose a custom output filename when the job-name option is turned off. |

### 11.1 Basic Workflow

Use the Job Manager when you want to prepare several jobs first and process them later as a batch.

1. Set up your workspace as you normally would before recording. Choose the target media, assign source faces, and adjust the settings you want to save with the job.
2. In the Job Manager, click **Save Job**.
3. Enter a job name. You can also choose whether to use that name for the final output file.
4. Repeat this for any additional jobs you want to queue.
5. When you are ready to process, select one or more jobs and click **Process Selected**, or click **Process All** to run the full queue.
6. Processing starts automatically. When the queue finishes, the app shows a completion message.

---

## 12. Presets

Presets save and restore all control panel settings as named JSON files stored in the `presets/` folder. They let you quickly switch between configurations - for example, a preset optimized for portrait photos versus one for action video.

- Click **Save Current as Preset** to save a preset and enter a name.
- Double-click a preset name to apply it.
- Use **Apply Settings** if you want the preset to also apply saved settings/control values.
- Presets can be renamed or deleted from the right-click context menu.

---

## 13. Video Timeline & Markers

The video timeline supports markers that let you apply different face-card settings at different points in a video. This is useful when camera angle, lighting, or cast changes during a clip.

- Click **Add Marker** to open the marker menu.
- Use **Add Standard Marker** to insert a marker at the current playback position.
- Each standard marker stores the current parameter and control settings.
- Use **Add Record Start Marker** and **Add Record End Marker** to define a recording segment on the timeline.
- Use **Previous Marker** and **Next Marker** to move between markers.
- Enable **Track Markers on Video Seek** if you want settings to update when you seek to a marked position.

---

## 14. Recording & Output

### 14.1 Recording Controls

Playback and recording use toggle-style buttons. Related actions in the main window also include **Save Image** and the batch-processing buttons.

### 14.2 FFmpeg Output Options

| Option | Description |
|---|---|
| **Presets SDR** | HEVC_NVENC encoding presets for standard-dynamic-range output (p1-p7). p1 is the fastest but lowest quality; p7 is the slowest but highest quality. The default is p5, which offers a good balance. |
| **Presets HDR** | Encoding presets for high-dynamic-range output: ultrafast, superfast, veryfast, faster, fast, medium, slow. **Important:** HDR encoding bypasses the GPU encoder and uses CPU-based libx265, which is significantly slower. Use only on genuine HDR source material. |
| **Quality** | CRF-equivalent quality setting (0-51). Lower values produce larger, higher-quality files. Default is 18. |
| **Spatial AQ / Temporal AQ** | Adaptive quantisation options available with NVENC. Spatial AQ allocates more bits to detailed areas within a frame; Temporal AQ maintains consistent quality across frames over time. Both improve perceptual quality at similar file sizes. |
| **Frame resize to 1920x1080** | Forces the output to 1080p resolution regardless of the source dimensions. Only effective on 16:9 content. |
| **Open Output Folder After Recording** | Automatically opens the output directory in your file explorer when recording stops. |

### 14.3 Playback Settings

| Setting | Description |
|---|---|
| **Set Custom Video Playback FPS** | Enables manual playback FPS control instead of using the file's normal playback speed. |
| **Video Playback FPS** | Sets the manual playback frame rate when custom FPS is enabled. |
| **Playback Buffering** | Enables frame buffering to smooth out playback on slower systems. |
| **Playback Loop** | Loops video playback continuously. |
| **Audio Playback Volume** | Controls the volume of audio during preview playback. |
| **Audio Start Delay** | Introduces a delay (in seconds) before audio begins, useful to compensate for sync issues. |

---

## 15. Settings

### 15.1 Performance

**Providers Priority** - selects the inference backend. The default is **TensorRT**. The four options are:

**CUDA** - Runs models via ONNX Runtime using the GPU's CUDA cores. Straightforward and compatible with any Nvidia GPU that has CUDA installed. A good fallback if you encounter TensorRT issues.

**TensorRT** - Uses ONNX Runtime with the TensorRT execution provider. On first use of each model it builds an optimized engine cache in `tensorrt-engines/`; subsequent runs load from cache and are noticeably faster than plain CUDA. The build step is automatic and a progress dialog is shown while it runs. This is the default provider.

**TensorRT-Engine** - Bypasses ONNX Runtime entirely for supported models and loads pre-built `.trt` engine files. Delivers the highest throughput of the three GPU options. Engine files are built automatically on first use of this provider (requires TensorRT 10.2.0 or later, which is satisfied by the bundled install). If a pre-built engine is not available for a given model it falls back to ONNX Runtime automatically.

**CPU** - Runs without GPU acceleration. Works on any hardware but is significantly slower than all GPU options.

| Setting | Description |
|---|---|
| **Number of Threads** | Number of execution threads used during playback and recording. Reduce to 1 if you encounter VRAM issues or crashes. |
| **Keep Controls Active** | When enabled, UI controls remain interactive during recording, allowing you to make live adjustments. When disabled, controls are locked while recording is in progress. |
| **Track Markers on Video Seek** | Updates parameters and controls when seeking to a marked position on the timeline. |
| **Keep Loaded Models in Memory** | Prevents models from automatically unloading when you change settings. Models remain in VRAM until you press the Clear GPU button explicitly. Useful when rapidly iterating on settings to avoid repeated load delays. |
| **Resize Input Source (Performance/Output)** | Downscales the input resolution before processing to trade output quality for speed. |
| **Input Resolution Target** | The target resolution when Resize Input Source is enabled (540p, 720p, or 1080p). Aspect ratio is preserved. |

### 15.2 Face Recognition

| Setting | Description |
|---|---|
| **Recognition Model** | The ArcFace-based embedding model used to generate face identity vectors. This setting controls two distinct things. During **face detection and matching** - identifying which detected face in the frame corresponds to which face card - the model selected here is used directly. During the **swap itself**, the app automatically selects the correct ArcFace model based on the active swapper (Inswapper128, InStyleSwapper256, and DFM use Inswapper128ArcFace; SimSwap512 uses SimSwapArcFace; GhostFace-v1/v2/v3 use GhostArcFace; CSCS uses CSCSArcFace) regardless of what is selected here. In most cases the default is fine; changing this may affect how well face cards are matched to detected faces when using the Similarity Threshold. |
| **Swapping Similarity Type** | The alignment strategy used when computing face embeddings: **Optimal** (full warp via arcfacemap - default), **Pearl** (offset alignment), or **Opal** (standard similarity transform). Affects how closely the embedding captures the face geometry. |
| **Embedding Merge Method** | When multiple source images are combined into a single face card embedding, controls how their individual vectors are merged: **Mean** (average of all vectors) or **Median** (more robust to outlier images). |

### 15.3 Face Detection & Tracking Settings

These settings are also related to Section 10 but are configured in the Settings tab.

| Setting | Description |
|---|---|
| **Face Detect Model** | Selects the active face detector: RetinaFace, SCRFD, Yolov8, or Yunet. See Section 10.1 for descriptions of each. |
| **Detect Score** | Minimum confidence threshold for a detection to be accepted. |
| **Detection Interval (Frames)** | Runs detection only every N frames; reuses results in between for performance. |
| **Max No of Faces to Detect** | Limits how many faces are processed per frame. |
| **Auto Rotation** | Rotates the input frame to the detected face's upright orientation before processing. |
| **Manual Rotation** | Overrides auto rotation with a fixed angle. |
| **Rotation Angle** | The angle used when Manual Rotation is enabled. |
| **Enable Face Tracking (ByteTrack)** | Toggles the ByteTrack multi-face tracker. |
| **Track Threshold** | Minimum detection score to initialise a new track. |
| **Match Threshold** | How closely a detection must match an existing track to be linked to it. |
| **Track Buffer (Frames)** | How many frames a track is kept alive after a face disappears before being discarded. |
| **Show Bounding Boxes** | Displays face detection bounding boxes on the preview. Useful for diagnosing missed or incorrect detections. |
| **Show ByteTrack Bounding Boxes** | Displays ByteTrack-assigned bounding boxes separately from the raw detector output, helping distinguish between detection and tracking results. |
| **Enable KPS Smoothing** | Smooths facial keypoints over time to reduce jitter. |

### 15.4 Landmark Detection

VisoMaster Fusion includes a configurable landmark detector that can be used to improve face crop accuracy and alignment.

| Setting | Description |
|---|---|
| **Enable Landmark Detection** | Activates landmark detection alongside face detection. |
| **Landmark Detect Model** | Selects the landmark model by number of points: **5**, **68**, **3d68**, **98**, **106**, **203**, or **478**. More points provide finer landmark coverage at a greater computational cost. The 5-point model is the fastest and covers the key facial anchor positions. The 478-point model provides the most detailed mesh. |
| **Landmark Detect Score** | Minimum confidence threshold for landmark detections to be accepted. |
| **Detect From Points** | Uses the detected landmarks as the face crop reference rather than the bounding box from the face detector. |
| **Use Mean Eyes** | Averages the eye landmark positions to produce a more stable eye-centre estimate, reducing jitter in per-frame alignment. |
| **Show Landmarks** | Overlays the detected landmark points on the preview frame. Useful for verifying detection accuracy. |

### 15.5 Appearance

The **Theme** selector lets you choose from a set of built-in UI colour schemes: True-Dark, Dark, Dark-Blue, Light, Solarized-Dark, Solarized-Light, Dracula, Nord, and Gruvbox. Themes are applied immediately without restarting.

### 15.6 VR / 360-Degree Mode

When working with VR180 or equirectangular 360-degree video, enable **VR180 Mode**. The app will unproject perspective crops for each face, process them, and stitch them back into the equirectangular image.

| Setting | Description |
|---|---|
| **VR180 Eye Mode** | Controls whether the input is treated as **Both Eyes** or **Single Eye**. |
| **VR Tiled Face Detection** | Runs face detection on multiple perspective crops to catch faces that may be missed in the equirectangular view. |
| **VR Max Crop FOV** | Sets the maximum field of view used for per-face perspective crops. |
| **VR Crop Resolution** | Sets the resolution of the perspective crops used during swapping. |

### 15.7 Webcam & Virtual Camera

| Setting | Description |
|---|---|
| **Webcam Max No** | The maximum webcam device index to scan when enumerating available cameras. |
| **Webcam Backend** | The backend used to access the webcam (e.g. DirectShow, V4L2). Choose based on your OS and camera hardware. |
| **Webcam Resolution** | Sets the capture resolution requested from the webcam. |
| **Webcam FPS** | Sets the frame rate requested from the webcam. |
| **Send Frames to Virtual Camera** | Routes the processed output to a virtual camera device (e.g. OBS Virtual Camera, v4l2loopback). This lets other applications - video calls and streaming software - use VisoMaster Fusion's output as a live camera source. |
| **Virtual Camera Backend** | Selects the virtual camera driver to use. Choose the option that matches your installed virtual camera software. |

### 15.8 Workspace & File Management

| Setting | Description |
|---|---|
| **Auto Save Workspace** | Automatically saves the current workspace (all settings and face card assignments) as a `.json` file in the output folder when recording finishes. Only the state at the end of recording is saved. |
| **Auto Load Last Workspace** | Skips the "load last workspace?" prompt at startup and always loads the most recent workspace automatically. |
| **Target Media Include Subfolders** | When selecting a target media folder for batch processing, includes files from all subfolders. |
| **Input Faces Include Subfolders** | When selecting an input faces folder, includes face images from all subfolders. |
| **Save Output Image in JPG Format** | Saves output images (from Save Image or batch processing) in JPG format instead of the default PNG. |
| **Commandline Infos** | Outputs restore strength metrics and iteration counts to the console, along with JPEG/MPEG quality information. Useful for diagnosing performance and quality tradeoffs. |

### 15.9 Swap Settings

These global swap-behavior settings live in the Settings tab and affect how source faces and target media are handled.

| Setting | Description |
|---|---|
| **Auto Swap** | Automatically swaps faces using the selected source faces or embeddings when a target image or video is loaded. |
| **Keep Selected Input Faces / Embeddings** | Keeps the current source-face or embedding selection when switching target media. |
| **Swap Input Face only once** | Limits swapping to the best match for each input face instead of swapping every match above the threshold. |
| **Maximum DFM Models to use** | Sets how many DFM models can remain loaded in memory at once. |
| **Embedding Merge Method** | Controls whether merged embeddings use **Mean** or **Median**. |

### 15.10 Embedding Manager

The **Advanced Embedding Editor** is available from the main UI. It loads `.json` embedding files produced by the face-card system, displays each stored identity as a named card, and lets you select, reorder, rename, and save embeddings between sessions. Main actions include **Load File(s)**, **Load Additive**, **Save As**, **Save Selected**, drag-and-drop reordering, multi-select with **Select All** / **Deselect All**, sorting (**Manual**, **Original**, **A-Z**, **Z-A**), and a search filter. Undo/redo is supported through Ctrl+Z / Ctrl+Shift+Z.

### 15.11 Experimental Settings

The Settings tab also contains **Experimental settings (very experimental, better don't touch)**. These controls are intended for testing and troubleshooting rather than normal day-to-day use.

---

## 16. Model Optimiser

VisoMaster Fusion has two separate model optimisation processes:

**ONNX Simplification** (`app/tools/optimize_models.py`) simplifies eligible ONNX model files using onnxsim (constant folding, dead node removal) and symbolic shape inference. It replaces the original `.onnx` files in `model_assets/` with leaner versions, backing up the originals to `model_assets/unopt-backup/`. This produces optimized ONNX files - not TensorRT engines. It can be run via the **Optimize Models (onnxsim)** action in the launcher's maintenance menu, which calls this script directly. The **Revert to Original Models** launcher action deletes the optimized files and re-downloads the originals from source.

**TensorRT Engine Building** is a separate process that converts ONNX models into hardware-specific `.trt` engine files (`app/processors/utils/engine_builder.py`). This happens automatically the first time a model is used with the **TensorRT-Engine** provider - it is not triggered by the ONNX Simplifier. The resulting engine files are stored in `tensorrt-engines/` and reused on subsequent runs. A progress dialog is shown during the build.

> **Note:** TensorRT engines are hardware-specific. An engine built on one GPU will not work on a different GPU model and must be rebuilt. This happens automatically the first time you run with the new hardware.

---

## 17. Tips & Best Practices

- Start with the default **Inswapper128** model and **Auto Resolution** to verify your setup before trying higher-quality but slower options
- Enable **Face Restorer** (GFPGAN-v1.4) as a first step - it corrects most visible artifacts with minimal configuration.
- Use the **Similarity Threshold** to target a specific person in a crowd. Set it high (80+) if only one face should be swapped
- For video, enable **ByteTrack face tracking** so the app keeps the correct source assigned to the correct person across cuts and occlusions
- When using the **Face Expression Restorer**, keep the Neutral Factor below 1.0 - the default of 0.30 is a reasonable starting point since the swapped face already carries some expression from the swap model itself. Use Simple mode first; only switch to Advanced if you need to suppress or amplify specific facial regions independently.
- For expression restoration on tricky angles, enable **Relative Position** in each region (Advanced mode) - this anchors the animation to the source face's starting pose and significantly reduces warping.
- If the swapped face looks blurry, try increasing Swapper Resolution or enabling Strength at 200%
- The **Denoiser** works best as a targeted pass rather than being enabled at all three positions simultaneously. Start with a single pass **After First Restorer** using Single Step mode and a low timestep (t = 100-200) before experimenting further.
- Use the **Show Bounding Boxes** and **Show Landmarks** overlays in Settings to diagnose face detection issues before committing to a long render
- Save frequently used configurations as **Presets** so you can switch quickly between different target subjects or content types
- For batch processing, load all jobs into the **Job Manager** and let them run unattended
- For difficult multi-face scenes, process one face at a time - remove all other detected faces, record, then run the output back through for the next face.
- If you use multiple DFM models in a session, increase **Maximum DFM Models to use** in Settings to avoid repeated model loading delays when switching between them (adjust based on available VRAM)
- Use **Face Re-Aging** when the source face needs to look noticeably older or younger before swapping
- On VR180 footage, enable **VR Tiled Face Detection** if edge or near-camera faces are being missed
- Use **Auto Save Workspace** to automatically preserve your full settings after each recording session, making it easy to resume exactly where you left off
- If processing slows down or crashes during long renders, reduce **Number of Threads** in Settings and disable **Keep Loaded Models in Memory** to free VRAM between model uses

---

## 18. Glossary

| Term | Definition |
|---|---|
| **ArcFace** | A deep learning model that encodes a face image into a fixed-length identity vector (embedding). VisoMaster Fusion uses several ArcFace variants paired to specific swapper models: Inswapper128ArcFace, SimSwapArcFace, GhostArcFace, and CSCSArcFace. The correct variant is selected automatically during swapping; the UI setting controls the face matching pass. |
| **Auto Mouth Expression** | A feature in the Face Expression Restorer that automatically activates mouth-expression transfer when mouth action is detected in the scene. |
| **ByteTrack** | A multi-object tracking algorithm that assigns a consistent ID to each detected face across video frames. Ensures the correct face card settings follow the correct person through motion and occlusion. |
| **CFG Scale** | Classifier-Free Guidance scale. Used by the Ref-LDM Denoiser in DDIM mode to control how strongly the output adheres to the reference face features. Higher values increase adherence. |
| **CLIP** | OpenAI's vision-language model. Used by the Text Masking feature (Section 4.5.5) to identify and segment objects described in plain English, such as "glasses" or "hat", from the face region. |
| **Crop Scale** | A Face Expression Restorer parameter (Section 7.3) controlling the size of the face crop passed to the LivePortrait model. Higher values capture the face from a greater distance. |
| **CSCS** | A face swap model that uses two separate embeddings (appearance and identity) for stronger likeness transfer on challenging angles. Uses its own CSCSArcFace recognition model. |
| **CUDA** | Nvidia's GPU compute platform. One of four inference backends in VisoMaster Fusion (Section 15.1). Runs models via the ONNX Runtime CUDAExecutionProvider. |
| **DDIM** | Denoising Diffusion Implicit Models. The iterative denoising mode used by the Ref-LDM Denoiser (Section 6). Produces more refined output than Single Step mode at greater processing cost. |
| **DFM** | DeepFaceLive Model. A custom pretrained face swap model format from the DeepFaceLive project. DFM files are placed in `onnxmodels/dfm_models/` and selected via the DeepFaceLive (DFM) swapper option. Uses Inswapper128ArcFace for recognition. |
| **Embedding** | A fixed-length numerical vector encoding a face's identity, produced by an ArcFace model. Multiple source images can be merged into a single embedding using Mean or Median merge (Section 15.2) for more consistent swap results. |
| **Face Re-Aging** | A Face Swap feature that transforms the assigned source face toward a target age before swapping. |
| **FaceParser** | A semantic segmentation model that labels each pixel of a face crop into anatomical classes such as background, skin, hair, lips, eyes, nose, and neck. Used by the Face Parser Mask (Section 4.5.7), Mouth Fit & Align (Section 4.5.6), Tongue/Mouth Priority in the Occluder (Section 4.5.3), and the XSeg Mouth sub-option (Section 4.5.4). |
| **GFPGAN** | A GAN-based face restoration model. Available in standard (v1.4) and high-resolution (1024) variants as a restorer option (Section 5). Repairs compression artifacts, blurring, and detail loss on the swapped face crop. |
| **GhostFace** | A family of lightweight face swap models (GhostFace-v1, GhostFace-v2, GhostFace-v3) available in VisoMaster Fusion. All variants use GhostArcFace for recognition. |
| **InStyleSwapper** | A set of 256 px face swap models (variants A, B, C) derived from the Inswapper architecture and trained using a custom technique. Uses Inswapper128ArcFace for recognition. |
| **Inswapper128** | The default face swap model. Fast and versatile, with configurable internal resolution (128-512 px via Swapper Resolution or Auto Resolution). Uses Inswapper128ArcFace. |
| **Landmark** | A keypoint detected on the face, such as the corner of an eye or the tip of the nose. Used for face alignment, crop warping, and expression transfer. VisoMaster Fusion supports landmark models detecting 5, 68, 3D-68, 98, 106, 203, or 478 points. |
| **LivePortrait** | A neural animation pipeline used by both the Face Expression Restorer (Section 7) and the Face Pose/Expression Editor (Section 8). Extracts motion keypoints from a driving face and applies them to the target. |
| **Mask Dilatation** | The process of growing a face mask region outward by a set number of pixels to ensure clean compositing at the edges of the swap. Controlled by the **Mask Dilatation Type** setting (`conv`, `pool`, or `iter_pool`). |
| **Mask view selection** | A Face Swap preview control that changes which mask visualization is shown: `swap_mask`, `diff`, or `texture`. |
| **Micro-Expression Boost** | A multiplier in the Face Expression Restorer's Advanced mode (Section 7.5) that amplifies subtle facial movements, such as small squints, slight smirks, and minor brow furrows, that may be compressed or lost during swapping and normalization. Operates when Relative Position is active for any region. |
| **Occluder** | The occlusion mask model (Section 4.5.3) that detects foreground objects covering the face, such as hands, glasses, or microphones, so they are preserved in the final composite rather than replaced by the swap. |
| **ONNX** | Open Neural Network Exchange. The model format used throughout VisoMaster Fusion to run AI models across different hardware backends (CUDA, TensorRT, CPU) without recompiling for each. |
| **Ref-LDM** | Reference Latent Diffusion Model. The UNet-based model used by the Denoiser (Section 6). Uses the source face as a reference to denoise and reconstruct texture on the swapped face. Supports Single Step and DDIM modes. |
| **RetinaFace** | A single-stage face detector from the InsightFace project. The default detector in VisoMaster Fusion, with high accuracy across a wide range of face sizes and orientations. |
| **SCRFD** | Sample and Computation Redistribution for Face Detection (ICLR 2022, InsightFace). Designed for an efficient accuracy-to-compute trade-off. The variant used here (SCRFD-2.5G) targets a 2.5 GFlop budget - faster than RetinaFace with competitive accuracy. |
| **SimSwap** | A face swap model operating natively at 512 px. Uses SimSwapArcFace for recognition. |
| **TensorRT** | Nvidia's inference optimization library. When selected as the provider (Section 15.1), ONNX Runtime uses the TensorrtExecutionProvider and automatically builds an engine cache in `tensorrt-engines/` on first use. |
| **TensorRT-Engine** | Pre-built `.trt` engine files built automatically on first use of the TensorRT-Engine provider. Bypasses ONNX Runtime entirely for supported models. Distinct from the ONNX Simplifier in the Model Optimiser (Section 16). |
| **Virtual Camera** | A software device that presents processed video output as if it were a physical webcam. VisoMaster Fusion can route its output to a virtual camera, such as OBS Virtual Camera or v4l2loopback, allowing the swapped result to appear as a live camera source in video call or streaming applications. |
| **VR180** | A 180-degree equirectangular video format for VR headsets. VisoMaster Fusion can process VR180 content by unprojecting perspective crops per eye, applying the swap, and stitching back into the equirectangular frame. |
| **Workspace** | A saved `.json` file capturing the full state of the application, including control settings, face card assignments, markers, and presets, at a given point in time. Workspaces can be saved manually or automatically at the end of each recording session through Auto Save Workspace. |
| **XSeg** | An occlusion segmentation model trained to identify foreground objects covering the face. Used by the DFL XSeg Mask (Section 4.5.4) as an alternative to the Occluder. |
| **YOLOv8** | A fast object detection architecture. Used here as the YoloFace8n face detector, the fastest of the four available detector options, suited to real-time and webcam use. |
| **YuNet** | A lightweight face detector from the OpenCV Model Zoo. Very low compute footprint, suited to CPU inference and resource-constrained scenarios. |
