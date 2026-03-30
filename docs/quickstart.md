# Quick Start Guide

> Your first successful face swap in a few minutes.

---

## Before You Begin

You will need:

- VisoMaster Fusion installed and running
- A target image or video
- At least one clear source face image
- A reasonably capable Nvidia GPU

For best results, use a clear source face with good lighting, a mostly visible face, and minimal heavy shadows or obstructions.

> **Tip:** One good source image is enough for a first test. If you have several photos of the same person, you can combine them into an embedding later for more consistent results.

---

## Step 1 - Load Your Target Media

1. Open a target image or video in the main window.
2. Confirm it appears in the center preview.
3. If you loaded a video, scrub through a few frames to make sure it opened correctly.

> **Tip:** You can also drag and drop target files directly into the target media area.

---

## Step 2 - Load Your Source Face

1. Add one or more source face images in the **Input Faces** area on the left.
2. Each source image becomes a face card.
3. Use a clear front-facing image for the fastest first success.

> **Tip:** You can also drag and drop source face images directly into the **Input Faces** area.

---

## Step 3 - Detect and Assign the Face

1. Detect the face or faces in the target media.
2. Select the target face you want to swap.
3. Assign your source face card to that target face.

> **Tip:** If the scene contains several people and you only want to swap one of them, remove or ignore the other detected faces before recording.

---

## Step 4 - Use Safe Starter Settings

For a reliable first result, start with these settings:

- In **Face Swap**, set **Swapper Model** to **Inswapper128**
- Turn on **Enable Auto Resolution**
- Leave **Similarity Threshold** at its default value for the first test
- In **Face Restorer**, turn on **Enable Face Restorer**
- Set **Restorer Type** to **GFPGAN-v1.4**

This gives most users a good starting point without needing advanced tuning.

> **Tip:** For a first pass, do not worry about markers, denoiser passes, or advanced masks unless you already know you need them.

---

## Step 5 - Set Your Output Folder

Before saving or recording, make sure an output folder is selected.

Set this in the **Settings** tab before you use **Save Image** or **Record**.

- Use **Save Image** for images
- Use **Record** for videos

> **Tip:** If saving or recording does not work, one of the first things to check is whether the output folder has been set.

---

## Step 6 - Preview and Record

1. Preview the result in the main window.
2. If you are working with video, play a short section first.
3. When you are happy with the result, use **Save Image** or **Record**.

---

## If Something Looks Wrong

- Wrong person is being swapped: raise **Similarity Threshold**
- Result looks soft or blurry: keep **Inswapper128**, leave **Enable Auto Resolution** on, and enable **Face Restorer**
- Face identity jumps between people in video: enable **Enable Face Tracking (ByteTrack)**
- Saving or recording does not work: check that an output folder is selected

---

## Going Further

For full explanations of settings, workflows, masks, restorers, markers, jobs, and advanced tools, see the [User Manual](./user_manual.md).
