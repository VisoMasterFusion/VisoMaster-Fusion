# HRFFA

HRFFA (High-Angle Robust Fast FaceAlignment) is a 68-point landmark model. Select
**hrffa** from the Landmark Detect Model list in *Settings → Detectors*; there is
nothing else to configure.

It is the only landmark model in VisoMaster that predicts from a crop of the whole
**head** rather than the **face**. That is the entire reason it exists: it stays
accurate through a full 360° of in-plane rotation, pitch beyond ±85°, and yaw out to
±90° — the poses where every face-cropped model here, `tufa98` included, comes apart.
The cost is that it needs a head box, which no face detector provides, so the mode
runs a second network to get one.

Its 68 points use the standard ibug layout, the same one the existing **68** option
(2dfan4) produces, so the two are interchangeable everywhere downstream.

## Models

The model downloader installs two ONNX files:

```
model_assets/hrffa_vitt_ibug68_1x3x256x256.onnx                       (~34 MiB)
model_assets/deimv2_hgnetv2_n_wholebody49_boxes_only_webgpu.onnx      (~15 MiB)
```

`hrffa_vitt_ibug68` is the ViT-T/16 student. The author also publishes a
PP-HGNetV2-B0 student (5× smaller, a little less accurate), 96×96 variants of both,
and the DINOv3 ViT-L teacher — the teacher is 1.2 GiB and carries Meta's DINOv3
licence, so it is not used here.

`deimv2_hgnetv2_n_wholebody49_boxes_only` is the head detector. Of the three
boxes-only exports it is both the smallest and the fastest, and its PP-HGNetV2
backbone keeps the whole dependency chain on Apache-2.0 — the two `dinov3` variants
would pull the DINOv3 licence into a GPL-3 project.

Neither model is on the FP16 allowlist (`fp16_safe_models_list`), and neither should be
added without measuring first. HRFFA-vitt is a ViT regression head, which is the shape
that fails **silently** under `trt_fp16_enable` here — TUFA builds a working FP16
engine that emits ~69 px of error on a 256 px crop. DEIMv2 is a DETR-style decoder,
which is the shape whose FP16 build killed ORFormer outright with an access violation.
Both are fast enough in FP32 that guessing is not worth a silent failure.

If either file is missing, run `python download_models.py`. A missing HRFFA graph
disables the mode; a missing head detector logs one error and falls back (see below).

## Implementation notes

Everything lives in `app/processors/face_landmark_detectors.py`. The head detector is
driven from there rather than from `face_detectors.py`, following the pattern **478**
already uses to force-load `FaceBlendShapes`. Two reasons: `_run_onnx_binding` is
model-agnostic, and `FaceDetectors` keeps only one detector resident at a time
(`current_detector_model`), so routing a second detector through it would evict
RetinaFace on every frame.

- **Head detection.** DEIMv2-Wholebody49 takes RGB `/255` with no mean/std, produced by
  a *direct* resize to 640×640 — not the aspect-preserving letterbox
  `FaceDetectors._prepare_detection_image` applies, because the boxes it returns are
  normalised against the original frame. Its single output `label_xyxy_score`
  `(1, 1240, 6)` is `(class, x1, y1, x2, y2, score)`; class 7 is `head` in both the
  Wholebody49 and Wholebody34 vocabularies. DETR-style one-to-one matching means the
  output is already NMS-free, so decoding is just a class filter and a threshold.
- **Score threshold.** 0.30, looser than upstream's 0.50. A head box is only used if it
  actually contains the already-confirmed face box, so a false positive elsewhere in
  the frame costs nothing, while a miss drops onto the fallback crop.
- **Matching a head to a face.** By containment (intersection ÷ face area), not IoU. A
  correct head box swallows the face box while being far larger, which drags IoU down
  to roughly 0.25–0.4 — an IoU ranking would prefer a wrong but similarly-sized
  box.
- **Fallback head box.** If the head detector is unavailable or missed, the mode
  synthesises a square of `1.5 × max(w, h)` about the face centre, shifted up by
  `0.08 × h` because a head reaches much further above a face box than below it. It
  only approximates the training crop, so landmarks are worse than with a real head
  box, but the mode keeps working.
- **Crop.** An axis-aligned square of `1.1 × max(w, h)` of the head box, resized to
  256. The `1.1` is upstream's training pad of 0.05 per side.
- **No roll correction.** TUFA and ORFormer upright their crop from the eye keypoints
  because they only saw ±15° of rotation augmentation. HRFFA does not: it is robust
  through a full 360° of roll on its own, and at the poses this mode exists for the 5
  keypoints are the least trustworthy thing available. `_prepare_crop` is therefore
  called with `det_kpss=None`, which pins its angle to 0.
- **Normalisation.** `center05`, i.e. `(x/255 − 0.5)/0.5`. Nothing is folded into the
  graph, unlike TUFA and ORFormer.
- **Output.** `points` `(1, 68, 2)` is normalised to the crop, hence the `× 256`. The
  graph's second output `vis_logits` `(1, 68, 3)` — per-point visibility as
  outside-image / occluded / visible — is left unbound so ONNX Runtime prunes that
  branch. Nothing in VisoMaster consumes per-point visibility today.
- **One head pass per frame.** The head box belongs to the frame, not to a face, so the
  per-face loops in `sequential_detector.py` and `frame_worker_standard.py` call
  `FunctionWorker.run_detect_head_bboxes` once and pass the result down as
  `head_bboxes=`. Callers that do not (the "Find Faces" button, the VR worker) let the
  landmark function run the detector itself, which is correct but costs one inference
  per face. Caching on the frame tensor's address would be wrong — Torch's allocator
  reuses addresses across frames.
- **`from_points` is ignored,** for the same reason it is ignored by TUFA and ORFormer:
  the model has never seen a 5-point similarity warp onto a frontal ArcFace template.
- **The score slider is ignored.** HRFFA surfaces no per-point confidence, so it
  returns empty scores and `run_detect_landmark` passes the result through unfiltered,
  exactly as 106, 203 and TUFA do.

## Performance and accuracy

Measured on an RTX 4090, FP32, through VisoMaster's own call path:

| | TensorRT | CUDA |
|---|---|---|
| Landmark pass | 3.8 ms/face | 7.3 ms/face |
| Head detection pass | 3.4 ms/frame | 7.0 ms/frame |

Both graphs build a TensorRT engine cleanly on the first run -- roughly 50 s for
HRFFA and 130 s for DEIMv2 -- and TensorRT is about twice as fast as CUDA for both,
so there is no reason to avoid it here. Handed an identical head box, the TensorRT
and CUDA landmark outputs agree to 0.03 px mean / 0.07 px max on a 256 px crop, so
there is no silent divergence of the kind FP16 causes with TUFA.

On a real frontal face, HRFFA's five swap-driving points land 1.6 px (4.9% of
interocular distance) from `tufa98`'s. Two models with different topologies,
different crops and different training data agreeing that closely is the check that
the crop geometry, the coordinate round-trip and the 68-to-5 conversion here are all
correct.

It is not especially fussy about head-box framing: resizing the box by 15% moves the
landmarks 2.7-4.0% of interocular distance, in the same range as `orformer98`. The
estimated fallback box lands within 1.7% of what the real head detector produces on
a frontal face, which is why the degraded path is worth having -- expect it to be
worse than that at the extreme angles the model is actually for, where a face box
says much less about where the head is.

## Credits

Both ONNX files are downloaded directly from the author's
[release](https://github.com/PINTO0309/High-Angle_Robust_Fast_FaceAlignment/releases/tag/weights)
rather than re-hosted, unlike the TUFA and ORFormer assets. HRFFA's licence note asks
distributors to check the derived-work terms of the DINOv3 teacher and the Apache-2.0
DEIMv2 backbone before redistributing the weights; linking upstream means VisoMaster
never redistributes them, so that question does not arise.

HRFFA is by Katsuya Hyodo, MIT-licensed, at commit
`1155c7f7b3f07c649c64f45516750f86ca0e7015`. The decode geometry and normalisation here
follow `demo/demo_hrffa_onnx.py` and `demo/web/src/hrffa/` in that repository.
DOI: [10.5281/zenodo.22161811](https://doi.org/10.5281/zenodo.22161811).

The head detector is DEIMv2-Wholebody49, also by Katsuya Hyodo, trained on
[DEIMv2](https://github.com/Intellindust-AI-Lab/DEIMv2) (Apache-2.0) with a
PP-HGNetV2 backbone.
