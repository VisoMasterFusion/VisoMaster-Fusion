# LSFG preview integration

VisoMaster Fusion can be used with Lossless Scaling (LSFG) as a preview-only frame-generation layer.

## Important limitation

LSFG is proprietary. Fusion does not feed frames directly into the LSFG engine. Lossless Scaling captures the visible VisoMaster window through its own DXGI/WGC capture path and performs frame generation itself.

Official LSFG documentation recommends a stable base framerate and notes that 30 FPS is a minimum, 40+ FPS is preferred, and 60 FPS is ideal at 1080p. LSFG 3.1 also provides Performance Mode for reducing GPU load.

## Helper

Run from the Fusion project directory on Windows:

```text
python tools/lsfg_preview.py --refresh 60 --multiplier 2 --wait 2
```

For a 120 Hz monitor:

```text
python tools/lsfg_preview.py --refresh 120 --multiplier 2
```

For 144 Hz with X3:

```text
python tools/lsfg_preview.py --refresh 144 --multiplier 3
```

The helper starts Lossless Scaling, checks whether a VisoMaster window exists, and calculates the recommended Fusion base FPS. It does not modify exported video FPS.

## Recommended Fusion profiles

- 60 Hz: 30 FPS -> LSFG X2 -> 60 FPS
- 120 Hz: 60 FPS -> LSFG X2 -> 120 FPS
- 144 Hz: 48 FPS -> LSFG X3 -> 144 FPS
- 240 Hz: 60 FPS -> LSFG X4 -> 240 FPS

For 1440p/4K, use LSFG's Flow Scale/Resolution Scale when GPU load is too high.

## Next step

A true one-click Fusion UI integration would require wiring the bridge into `MainWindow.initialize_widgets()` and the settings layout. The bridge is deliberately kept independent so it cannot destabilize the AI/rendering pipeline.
