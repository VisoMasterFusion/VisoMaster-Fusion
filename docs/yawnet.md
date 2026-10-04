# YawNet

YawNet estimates **head yaw over the full 360°**. Enable **Head Yaw (YawNet)** under
the Landmark Detect Model list in *Settings → Detectors*, with **Show Head Yaw Ring**
as an optional preview overlay beneath it.

VisoMaster already had a yaw estimate: `faceutil.calc_face_yaw_pitch` compares the
horizontal gap from the nose to each eye. That is free, since the 5 keypoints are
already there, but it has two limits. It saturates near ±90°, and — because it only
ever sees a face's own landmarks — it reads a head turned *fully away* as roughly
frontal. YawNet tells those apart, which is what makes it useful here.

Like `hrffa`, it predicts from a crop of the whole **head**, so it shares the same
DEIMv2-Wholebody49 head detector. In `hrffa` landmark mode that detector already runs
and YawNet is nearly free; in any other mode enabling it adds one head-detection pass
per frame. YawNet itself is ~0.3 ms.

## Model

The downloader installs one ONNX file:

| File | Input | Params | MAAE | Acc@30° |
|---|---|---|---|---|
| `yawnet_distill_128_unified_v6u_kappa_1x3x128x128.onnx` | 128×128 | 0.77 M | 12.62° | 92.6% |

The author also publishes 64 px and 96 px students. They are the same size and differ
by under 0.4° MAAE, so the most accurate one is pinned; the 320 px DINOv3 teacher
(0.97° MAAE) is 1.1 GB and carries Meta's DINOv3 licence, so it is not offered.

The `_kappa` export adds a confidence output at no measurable cost. See *Confidence*
below for why nothing currently gates on it.

## Implementation notes

**Input.** RGB float32, NCHW `1×3×128×128`, center05 normalised (`(x/255 − 0.5)/0.5`,
identical to the `x/127.5 − 1` in the upstream README). The head box is squashed to the
square **without preserving aspect ratio** — upstream's `orientation.ts` does the same,
and a letterbox would be out of distribution.

**Output.** `cos_sin` `(1,2)`, a unit biternion, plus `kappa` `(1)`.

**The angle convention is easy to get wrong.** YawNet emits the `yawpose` convention
and needs a mirror to become the ring convention:

```python
deg = (degrees(atan2(sin, cos)) + 360) % 360
deg = (360 - deg) % 360          # <- mirror; omitting this swaps left and right
```

giving **0 = facing the camera, 90 = subject's right, 180 = facing away, 270 = left**.
`FaceLandmarkDetectors.yaw_from_frontal()` folds that onto 0–180° "degrees away from
frontal", which is what consumers should compare against a threshold — comparing raw
ring degrees would read 350° (10° off frontal) as extreme.

**Not fp16-safe.** Untested in fp16, and deliberately excluded for the same reason as
HRFFA: the product of this model is an *angle* derived from a unit vector, so small
fp16 drift in `(cos, sin)` is a silently wrong pose rather than a visible artefact.

## What uses it

The **occluder / XSeg profile safeguard**. Those two segmentation models are trained on
mostly-frontal faces and can collapse on extreme profiles, marking the whole crop as
occluded, which zeroes the blend mask and makes the swap silently vanish for those
frames. The safeguard bypasses the failing model, but only when the head really is at
an extreme angle *and* the mask has collapsed — a low face ratio on its own is also
exactly what a correct prediction looks like when a hand covers most of the face.

YawNet supplies the angle half of that test. Without it the safeguard falls back to the
landmark estimate, which is least reliable at precisely the angles being judged.

It is deliberately **not** substituted into the yaw that feeds `get_dynamic_side_mask`.
That path's `ProfileAngleMaskThresholdSlider` is calibrated in the landmark estimate's
pseudo-degrees, and changing the scale underneath it would silently change what every
saved user setting means.

## Confidence

`kappa` is `softplus(logit).clamp(1e-3, 100)`. The gate defaults to **0 — off**:

- The kappa head's bias is initialised so `softplus(1.85) ≈ 2.0`, described upstream as
  "the same as the previous fixed kappa". **2.0 is the model's neutral prior, not a
  confidence floor**, so a threshold near it would reject ordinary predictions.
- Upstream's own demo reads only `cos_sin` and ignores `kappa`, so there is no
  reference value to copy.

Leaving it off is safe for the safeguard, which also requires a collapsed mask, so a
spurious angle alone cannot trigger anything.

**Min Confidence (kappa)** under the YawNet toggle raises it. Every reading is logged
at debug level *before* the gate is applied, and unconditionally, so the readings you
need in order to choose a threshold are not the ones a gate has already discarded.

## Calibration status

Three values are **plausible guesses, not measured ones**. All are now exposed in the
UI specifically so they can be tuned on real footage and better defaults chosen later:

| Setting | Where | Default | Constant |
|---|---|---|---|
| Profile Safeguard | Swapper, under the mask toggles | on | — |
| Min Head Angle | same | 40° | `_SEG_GUARD_MIN_ABS_YAW` |
| Min Face Area | same | 30% | `_SEG_GUARD_MIN_FACE_RATIO` |
| Min Confidence (kappa) | Settings, under YawNet | 0 (off) | `YAWNET_MIN_KAPPA` |

The module constants remain the fallbacks, so workspaces saved before these controls
existed — and internal callers that pass no parameters — keep the original behaviour.

Either safeguard slider can switch it off: **Min Face Area 0** (a mask mean can never
fall below zero) or **Min Head Angle 180**. The explicit toggle is clearer, but a
slider at those extremes will not misfire.

Useful directions when tuning:

- Profile swaps still vanish → *raise* Min Face Area, or *lower* Min Head Angle.
- Real occlusion (a hand, a microphone) is being ignored → *lower* Min Face Area, or
  *raise* Min Head Angle.

## Credits

The ONNX file is downloaded directly from the author's
[release](https://github.com/PINTO0309/YawNet/releases/tag/resources) rather than
re-hosted, matching how the HRFFA assets are handled.

YawNet is by Katsuya Hyodo, MIT-licensed. It is trained purely on a synthetic 42k-image
dataset of fictitious people (CC BY 4.0), so unlike the HRFFA students it inherits no
teacher-licence question. The preprocessing and angle convention here follow
`demo/web/src/hrffa/orientation.ts` in the
[HRFFA repository](https://github.com/PINTO0309/High-Angle_Robust_Fast_FaceAlignment),
and the `kappa` semantics follow `scripts/yawnet.py` in the YawNet repository.

Note that upstream uses YawNet only for visualisation — its README states the angle
"doesn't control HRFFA landmark prediction". Feeding it into the segmentation safeguard
is specific to VisoMaster.
