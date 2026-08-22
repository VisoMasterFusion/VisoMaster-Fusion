# AlphaFace

AlphaFace is a 256 x 256 face swapper. Select **AlphaFace** from the Swapper
Model list in the Face Swap panel; there is nothing else to configure.

It reuses VisoMaster's existing W600K ArcFace encoder, so no extra recognition
model is loaded. The 512x512 matrix that re-projects that embedding into
AlphaFace's identity space ships in the repository as
`model_assets/alphaface/emp.npy` and comes from the AlphaFace authors.

## Model

The model downloader installs a single ONNX file:

```
model_assets/alphaface/alphaface_swapper_fused_norm.onnx   (~529 MiB)
```

It is an FP32 graph, but `AlphaFace` is on the FP16 allowlist
(`fp16_safe_models_list`), so the TensorRT provider runs it in FP16. The first
run on the TensorRT backend pauses while the engine is built and cached.

If the file is missing, the swapper logs an error and leaves the face
unswapped — run `python download_models.py` to fetch it.

## Implementation notes

- **Alignment.** AlphaFace uses the pose-aware `arcfacemap` template (the same
  family GhostFace uses) rather than the fixed `arcface128` crop, restricted to
  the five yaw templates. The two trailing pitch templates can pick a
  noticeably different crop scale on near-profile faces, which pops between
  frames.
- **Instance normalisation.** The official implementation spells instance norm
  out as `ReduceMean`/`Mul`/`Sqrt`/`Div`. The exported graph uses a single
  `InstanceNormalization` node instead: numerically equivalent to within ~2e-5,
  and something TensorRT can execute in FP16.
- **Identity injection.** The projected identity vector is spatially 1x1, so its
  centred value is exactly zero and the official AdaIN expression collapses to
  the target channel mean. The exported graph uses the reduced form, which is
  bit-identical and much cheaper.

## Re-exporting the model

Download `alphaface_demo.pt` from the
[official repository](https://github.com/andrewyu90/Alphaface_Official) and run:

```powershell
.venv\Scripts\python.exe app\tools\export_alphaface_onnx.py C:\path\to\alphaface_demo.pt
```

The result is byte-identical to the downloaded asset, so the hash in
`models_data.py` stays valid.

## TensorRT

The shipped ONNX cannot be handed to TensorRT as-is. `torch.onnx.export`
annotated none of its 709 tensors and emitted the output as
`[Divoutput_dim_0, 3, Divoutput_dim_0, Divoutput_dim_3]` — one `dim_param`
covering both the batch and the height axis, which in ONNX declares those axes
equal. The reflect-pad calls exporting as runtime `Shape`/`Gather` arithmetic
are what defeat ordinary inference here.

`AlphaFace` is therefore listed in `tensorrt_shape_infer_models`, so the loader
runs ONNX Runtime's symbolic shape inference once and caches the result beside
the model as `alphaface_swapper_fused_norm.trtshape.onnx`. That resolves the
output to a static `[1, 3, 256, 256]` and annotates all 709 tensors; the FP16
engine then builds in roughly 35 s using about 2.4 GiB of VRAM.

Keep that list entry. Without it the engine build fails, retries, and can hang
hard enough to take the display driver down with it.

## Credits

The inference code is based on the authors' MIT-licensed
[implementation](https://github.com/andrewyu90/Alphaface_Official) at commit
`d41fbd4974ed3a68d9a48b79019f9be297726c30` (see
`app/processors/alphaface/LICENSE`). Training-only modules are omitted. For
model details, see the [AlphaFace paper](https://arxiv.org/abs/2601.16429).
