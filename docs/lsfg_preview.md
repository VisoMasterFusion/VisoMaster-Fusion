# Lossless Scaling / LSFG preview bridge

VisoMaster Fusion can use Lossless Scaling as an external frame-generation layer for the live preview. The integration point is `app.helpers.lsfg_bridge.LosslessScalingBridge`.

## Important limitation

LSFG itself remains proprietary to Lossless Scaling. The public Lossless Scaling release notes document its capture engine, but do not provide a public SDK for injecting Fusion frames into LSFG. Therefore this bridge does **not** copy, reverse-engineer, or reimplement LSFG.

The bridge is intentionally limited to:

- locating `LosslessScaling.exe` on Windows/Steam installations;
- starting Lossless Scaling from Fusion tooling;
- detecting whether a VisoMaster window is present;
- calculating a sensible fixed-multiplier base FPS.

After Lossless Scaling is running, select the VisoMaster window in Lossless Scaling and enable LSFG there.

## Recommended preview setup

For LSFG 3, use a stable base framerate. For a 120 Hz display and X2 generation, target about 60 FPS. For a 144 Hz display and X3 generation, target about 48 FPS. Lossless Scaling's own guidance recommends borderless/windowed capture rather than exclusive fullscreen and recommends a stable base framerate for good frame pacing.

The bridge does not alter recording/export FPS; it is intended as a companion layer for the preview window.
