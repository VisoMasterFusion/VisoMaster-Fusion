# Face Detailer

Enable **Settings → Face Detailer → Enable Face Detailer** to restore small
faces after the normal swap/edit pipeline. This is a global setting: it also
applies to detected faces that were not selected as swap targets. It defaults
to off and uses the existing detector and restoration models.

The pass runs at final frame resolution, after working-frame scaling and
rotation have been undone and before bounding boxes, landmarks, and yaw rings
are drawn. Mask and comparison views skip it. The separate VR processor does
not run this pass.

For each eligible face, the processor crops context, resizes that crop to a
canvas, aligns the five detected landmarks for restoration, inverse-warps the
result, and pastes it through a dilated, feathered ellipse. The source frame
is preserved; a failed detection or restoration leaves the affected pixels
unchanged. Detection bypasses ByteTrack so this second pass cannot update the
video tracker.

Defaults:

| Setting | Default | Meaning |
| --- | --- | --- |
| Restorer | GPEN-1024 | Existing restoration model; requires its weights |
| Face height | 24–128 px | Height in the final frame, before magnification |
| Crop factor | 2.5 | Context side relative to the larger face-box dimension |
| Canvas | 768 px | Temporary context canvas; does not change native model resolution |
| Blend | 100% | Strength of the restored result |
| Feather / dilation | 12 / 10 px | Mask widths in final-frame pixels |
| Maximum faces | 4 | Smallest eligible faces first; applied after size filtering |
| Detection score | 50% | Confidence floor for the second detection pass |
| Color match | On | Match face-region channel means and standard deviations |

CodeFormer and VQFR-v2 use the fidelity control. OSDFace uses its timestep and
latent-strength controls. A dedicated third restorer slot shares resident
models with the two pipeline slots; changing or disabling the detailer releases
models that are no longer needed.

This adds one detection pass plus restoration per eligible face. A different
restorer from the pipeline can increase VRAM use. Larger canvases resample the
context but do not create new source detail or increase the network's native
input resolution. Visual improvement and temporal consistency depend on the
footage and model; compare results with the feature disabled when choosing
settings. CPU regression tests cover integration and failure handling;
representative video quality and CUDA/TensorRT execution have not been validated.
