import os
from pathlib import Path
from typing import Any

import numpy as np

models_dir = Path(__file__).resolve().parent.parent.parent / "model_assets"
# ensure ref-ldm paths exist
refldm_ckpts_path = models_dir / "ref-ldm_embedding/ckpts"
os.makedirs(refldm_ckpts_path, exist_ok=True)

# Ensure the grouped ONNX model subfolders exist up front. On a fresh/portable
# install (and during portable app updates) these subfolders are otherwise only
# created on first download — but the PerformRecast models were missing the
# folder, so the portable updater had nowhere to place the freshly downloaded
# ONNX files. Creating them here makes the destinations exist regardless of how
# the models arrive (download vs. copy-in). The downloader also makes parent
# dirs on demand, so this is belt-and-suspenders.
for _subfolder in ("alphaface", "liveportrait_onnx", "performrecast_onnx", "osdface"):
    os.makedirs(models_dir / _subfolder, exist_ok=True)

assets_repo = "https://github.com/visomaster/visomaster-assets/releases/download"
alphaface_repo = (
    "https://github.com/kodek4/VisoMaster-Fusion/releases/download/alphaface-model-v1"
)
tufa_repo = "https://github.com/Glat0s/TUFA-onnx/releases/download/v0.0.1"
orformer_repo = "https://github.com/Glat0s/ORFormer-onnx/releases/download/v0.0.1"
osdface_repo = "https://github.com/Glat0s/OSDFace-onnx/releases/download/v0.0.1"
# HRFFA and its DEIMv2-Wholebody49 head detector are linked straight at the
# author's release instead of being re-hosted like tufa_repo / orformer_repo.
# HRFFA ships the students under MIT but asks distributors to check the
# derived-work terms of the DINOv3 teacher first; linking upstream means we never
# redistribute the weights, so that question does not arise. See docs/hrffa.md.
hrffa_repo = (
    "https://github.com/PINTO0309/High-Angle_Robust_Fast_FaceAlignment"
    "/releases/download/weights"
)

ARCFACE_DST = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)

FFHQ_KPS = np.array(
    [
        [192.98138, 239.94708],
        [318.90277, 240.1936],
        [256.63416, 314.01935],
        [201.26117, 371.41043],
        [313.08905, 371.15118],
    ],
    dtype=np.float32,
)

LANDMARKS_SUBSET_IDXS = [
    0,
    1,
    4,
    5,
    6,
    7,
    8,
    10,
    13,
    14,
    17,
    21,
    33,
    37,
    39,
    40,
    46,
    52,
    53,
    54,
    55,
    58,
    61,
    63,
    65,
    66,
    67,
    70,
    78,
    80,
    81,
    82,
    84,
    87,
    88,
    91,
    93,
    95,
    103,
    105,
    107,
    109,
    127,
    132,
    133,
    136,
    144,
    145,
    146,
    148,
    149,
    150,
    152,
    153,
    154,
    155,
    157,
    158,
    159,
    160,
    161,
    162,
    163,
    168,
    172,
    173,
    176,
    178,
    181,
    185,
    191,
    195,
    197,
    234,
    246,
    249,
    251,
    263,
    267,
    269,
    270,
    276,
    282,
    283,
    284,
    285,
    288,
    291,
    293,
    295,
    296,
    297,
    300,
    308,
    310,
    311,
    312,
    314,
    317,
    318,
    321,
    323,
    324,
    332,
    334,
    336,
    338,
    356,
    361,
    362,
    365,
    373,
    374,
    375,
    377,
    378,
    379,
    380,
    381,
    382,
    384,
    385,
    386,
    387,
    388,
    389,
    390,
    397,
    398,
    400,
    402,
    405,
    409,
    415,
    454,
    466,
    468,
    469,
    470,
    471,
    472,
    473,
    474,
    475,
    476,
    477,
]

arcface_mapping_model_dict = {
    "Inswapper128": "Inswapper128ArcFace",
    "AlphaFace": "Inswapper128ArcFace",
    "InStyleSwapper256 Version A": "Inswapper128ArcFace",
    "InStyleSwapper256 Version B": "Inswapper128ArcFace",
    "InStyleSwapper256 Version C": "Inswapper128ArcFace",
    "DeepFaceLive (DFM)": "Inswapper128ArcFace",
    "SimSwap512": "SimSwapArcFace",
    "GhostFace-v1": "GhostArcFace",
    "GhostFace-v2": "GhostArcFace",
    "GhostFace-v3": "GhostArcFace",
    "CSCS": "CSCSArcFace",
}

detection_model_mapping = {
    "RetinaFace": "RetinaFace",
    "SCRFD": "SCRFD2.5g",
    "Yolov8": "YoloFace8n",
    "Yunet": "YunetN",
}

landmark_model_mapping = {
    "5": "FaceLandmark5",
    "68": "FaceLandmark68",
    "3d68": "FaceLandmark3d68",
    "98": "FaceLandmark98",
    "106": "FaceLandmark106",
    "203": "FaceLandmark203",
    "478": "FaceLandmark478",
    # Both emit the 98-point WFLW topology, so they reuse
    # faceutil.convert_face_landmark_98_to_5 unchanged.
    "tufa98": "FaceLandmarkTUFA98",
    "orformer98": "FaceLandmarkORFormer98",
    # TUFA's own dense 314-point set — not a dataset topology, hence its own
    # converter (faceutil.convert_face_landmark_314_to_5).
    "tufa314": "FaceLandmarkTUFA314",
    # ibug68, the same topology as FaceLandmark68 (2dfan4), so it reuses
    # faceutil.convert_face_landmark_68_to_5. This mode also drives a second graph,
    # DEIMv2Wholebody49Head, for the whole-head box it crops from; that dependency is
    # force-loaded by FaceLandmarkDetectors the way 478 force-loads FaceBlendShapes,
    # and is deliberately NOT in this table (it is not a landmark model and must not
    # become the value control_actions loads/unloads for the mode).
    "hrffa": "FaceLandmarkHRFFA",
}

# Point count per landmark_model_mapping key. The mode string used to double as the
# count ("98", "203", ...); the named modes broke that, so callers that need a count
# (e.g. building a zero placeholder for a failed detection) look it up here.
landmark_point_counts = {
    "5": 5,
    "68": 68,
    "3d68": 68,
    "98": 98,
    "106": 106,
    "203": 203,
    "478": 478,
    "tufa98": 98,
    "orformer98": 98,
    "tufa314": 314,
    "hrffa": 68,
}

# Mapping of composite pipelines to their constituent ONNX session identifiers.
# Enables ModelsProcessor to recursively unload multi-network pipelines cleanly.
compound_models_mapping: dict[str, tuple[str, ...]] = {
    "OSDFace": (
        "OSDFacePromptEncoder",
        "OSDFaceVAEEncoder",
        "OSDFaceUNet",
        "OSDFaceVAEDecoder",
    ),
}

# Authoritative mapping between UI restorer selections and models_data identifiers.
restorer_model_mapping: dict[str, str] = {
    "GFPGAN-v1.4": "GFPGANv1.4",
    "GFPGAN-1024": "GFPGAN1024",
    "CodeFormer": "CodeFormer",
    "GPEN-256": "GPENBFR256",
    "GPEN-512": "GPENBFR512",
    "GPEN-1024": "GPENBFR1024",
    "GPEN-2048": "GPENBFR2048",
    "RestoreFormer++": "RestoreFormerPlusPlus",
    "VQFR-v2": "VQFRv2",
    "OSDFace": "OSDFace",
}

# Models listed here get trt_fp16_enable=True on the TensorRT EP.
#
# DO NOT add FaceLandmarkTUFA98, FaceLandmarkTUFA314 or FaceLandmarkORFormer98. All
# were measured under the exact options in ModelsProcessor.trt_ep_options and all fail
# in fp16:
#   * TUFA fails SILENTLY — the fp16 engine builds and runs 1.9x faster (2.09 vs
#     3.89 ms) while emitting ~69 px of error on a 256 px crop. That is garbage, not
#     precision loss; reproduced twice with byte-identical output.
#     trt_layer_norm_fp32_fallback is already on and does not help. The 314-point
#     export is the same graph with a longer prompt constant, so it inherits this.
#   * ORFormer fails LOUDLY — the fp16 build never produces an engine. All three
#     isolated probe attempts died natively (0xC0000005 access violation x2,
#     0xC000041D x1). Its fp32 build succeeds first try in ~97 s.
# In fp32 both are fast enough (3.9 ms and 5.9 ms per face on an RTX 4090).
# See onnx-export-notes.md in the repo root for the full measurements.
#
# FaceLandmarkHRFFA and DEIMv2Wholebody49Head are absent for the same reason, but
# unmeasured rather than measured: HRFFA-vitt is a ViT regression head (the shape that
# fails silently for TUFA) and DEIMv2 is a DETR-style decoder (the shape whose fp16
# build killed ORFormer). Both are cheap enough in fp32 that guessing is not worth the
# risk of a silent ~70 px error. Measure before adding either.
fp16_safe_models_list = [
    # --- LivePortrait ---
    "LivePortraitAppearanceFeatureExtractor",
    "LivePortraitStitchingEye",
    "LivePortraitStitchingLip",
    "LivePortraitStitching",
    "LivePortraitWarpingSpade",
    # --- Detectors ---
    "RetinaFace",
    "SCRFD2.5g",
    "YoloFace8n",
    "YunetN",
    # --- Masking ---
    "FaceParser",
    "Occluder",
    # --- Upscaling ---
    "RealEsrganx2Plus",
    "RealEsrganx4Plus",
    "RealEsrx4v3",
    "BSRGANx2",
    "BSRGANx4",
    "UltraSharpx4",
    "UltraMixx4",
    # --- Colors ---
    "DeoldifyArt",
    "DeoldifyStable",
    "DeoldifyVideo",
    "DDColorArt",
    "DDcolor",
    # --- Landmarks ---
    "FaceLandmark5",
    "FaceLandmark68",
    "FaceLandmark98",
    "FaceLandmark106",
    "FaceLandmark203",
    "FaceLandmark478",
    "FaceBlendShapes",
    # --- Restorers ---
    "GPENBFR256",
    "GPENBFR512",
    "CodeFormer",
    "VQFRv2",
    "RestoreFormerPlusPlus",
    "OSDFacePromptEncoder",
    "OSDFaceVAEEncoder",
    # --- Recognition ---
    "Inswapper128ArcFace",
    "SimSwapArcFace",
    "GhostArcFace",
    "CSCSArcFace",
    "CSCSIDArcFace",
    # --- Denoiser ---
    "RefLDMVAEEncoder",
    "RefLDMVAEDecoder",
    "RefLDM_UNET_EXTERNAL_KV",
    # --- Texture ---
    "combo_relu3_3_relu3_1",
    # --- Swappers ---
    "AlphaFace",
    "InStyleSwapper256 Version A",
    "InStyleSwapper256 Version B",
    "InStyleSwapper256 Version C",
    "GhostFacev1",
    "GhostFacev2",
    "GhostFacev3",
]

# Models whose ONNX graph must be shape-inferred (with a static batch=1) before
# the TensorRT EP can build an engine. The PerformRecast warping module contains
# 5-D GridSample nodes whose outputs have no static shape, which makes the TRT EP
# abort with "has no shape specified. Please run shape inference on the onnx
# model first." The loader transparently builds a cached, shape-inferred sidecar
# (``*.trtshape.onnx``) for these models. See ModelsProcessor._ensure_trt_ready_onnx.
#
# AlphaFace needs it for a different reason. Its graph ships with no value_info
# at all (0 of 709 tensors annotated), and torch.onnx.export emitted the output
# as ``[Divoutput_dim_0, 3, Divoutput_dim_0, Divoutput_dim_3]`` — reusing a
# single dim_param for both the batch axis and the 256px height axis, which in
# ONNX declares those two axes equal. Without shape inference the engine build
# hangs/crashes hard enough to take the display driver with it; with it the
# output resolves to a static [1, 3, 256, 256], all 709 tensors get shapes, and
# the FP16 engine builds in ~35 s using ~2.4 GiB. The reflect-pad Shape/Gather
# chains PyTorch generates are what defeat plain inference here, so the symbolic
# pass is required.
tensorrt_shape_infer_models = [
    "PerformRecastWarpingModule",
    "AlphaFace",
]

models_list: list[dict[str, Any]] = [
    {
        "model_name": "Inswapper128",
        "local_path": f"{models_dir}/inswapper_128.fp16.onnx",
        "hash": "6d51a9278a1f650cffefc18ba53f38bf2769bf4bbff89267822cf72945f8a38b",
        "url": f"{assets_repo}/v0.1.0/inswapper_128.fp16.onnx",
    },
    {
        # FP32 graph; the TensorRT EP casts it to FP16 via fp16_safe_models_list.
        "model_name": "AlphaFace",
        "local_path": f"{models_dir}/alphaface/alphaface_swapper_fused_norm.onnx",
        "hash": "5514d967ab6cc27e1b0edc092e05ee97d235adccb4da68574a9b1a1e221a4c6a",
        "url": f"{alphaface_repo}/alphaface_swapper_fused_norm.onnx",
    },
    {
        "model_name": "InStyleSwapper256 Version A",
        "local_path": f"{models_dir}/InStyleSwapper256_Version_A.fp16.onnx",
        "url": f"{assets_repo}/v0.1.0/InStyleSwapper256_Version_A.fp16.onnx",
        "hash": "0e0ef024b935abca69fd367a385200ed46b83a3cc618287ffe89440e2cc646da",
    },
    {
        "model_name": "InStyleSwapper256 Version B",
        "local_path": f"{models_dir}/InStyleSwapper256_Version_B.fp16.onnx",
        "url": f"{assets_repo}/v0.1.0/InStyleSwapper256_Version_B.fp16.onnx",
        "hash": "0870b6c75eaea239bdd72b6c6d0910cb285310736e356c17a2cd67a961738116",
    },
    {
        "model_name": "InStyleSwapper256 Version C",
        "local_path": f"{models_dir}/InStyleSwapper256_Version_C.fp16.onnx",
        "url": f"{assets_repo}/v0.1.0/InStyleSwapper256_Version_C.fp16.onnx",
        "hash": "6eaefc04cfb1461222ab72a814ad5b5673ab1af4267f7eb9054e308797567cde",
    },
    {
        "model_name": "SimSwap512",
        "local_path": f"{models_dir}/simswap_512_unoff.onnx",
        "hash": "08c6ca9c0a65eff119bea42686a4574337141de304b9d26e2f9d11e78d9e8e86",
        "url": f"{assets_repo}/v0.1.0/simswap_512_unoff.onnx",
    },
    {
        "model_name": "GhostFacev1",
        "local_path": f"{models_dir}/ghost_unet_1_block.onnx",
        "hash": "304a86bccb325e7fcf5ab4f4f84ba5172e319bccc9de15d299bb436746e2e024",
        "url": f"{assets_repo}/v0.1.0/ghost_unet_1_block.onnx",
    },
    {
        "model_name": "GhostFacev2",
        "local_path": f"{models_dir}/ghost_unet_2_block.onnx",
        "hash": "25b72c107aabe27fc65ac5bf5377e58eda0929872d4dd3de5d5a9edefc49fa9f",
        "url": f"{assets_repo}/v0.1.0/ghost_unet_2_block.onnx",
    },
    {
        "model_name": "GhostFacev3",
        "local_path": f"{models_dir}/ghost_unet_3_block.onnx",
        "hash": "f471d4f322903da2bca360aa0d7ab9922e3b0001d683f825ca6b15d865382935",
        "url": f"{assets_repo}/v0.1.0/ghost_unet_3_block.onnx",
    },
    {
        "model_name": "CSCS",
        "local_path": f"{models_dir}/cscs_256.onnx",
        "hash": "664f8f7cab655b825fe8cf57ab90bfbcbb0acf75eab8e7771c824f18bdb28b67",
        "url": f"{assets_repo}/v0.1.0/cscs_256.onnx",
    },
    {
        "model_name": "RetinaFace",
        "local_path": f"{models_dir}/det_10g.onnx",
        "hash": "5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91",
        "url": f"{assets_repo}/v0.1.0/det_10g.onnx",
    },
    {
        "model_name": "SCRFD2.5g",
        "local_path": f"{models_dir}/scrfd_2.5g_bnkps.onnx",
        "hash": "bc24bb349491481c3ca793cf89306723162c280cb284c5a5e49df3760bf5c2ce",
        "url": f"{assets_repo}/v0.1.0/scrfd_2.5g_bnkps.onnx",
    },
    {
        "model_name": "YoloFace8n",
        "local_path": f"{models_dir}/yoloface_8n.onnx",
        "hash": "84d5bb985b0ea75fc851d7454483897b1494c71c211759b4fec3a22ac196d206",
        "url": f"{assets_repo}/v0.1.0/yoloface_8n.onnx",
    },
    {
        "model_name": "YunetN",
        "local_path": f"{models_dir}/yunet_n_640_640.onnx",
        "hash": "9e65c0213faef0173a3d2e05156b4bf44a45cde598bdabb69203da4a6b7ad61e",
        "url": f"{assets_repo}/v0.1.0/yunet_n_640_640.onnx",
    },
    {
        # Yolov11-face trained at 960×960 for VR180 face detection. Unlike the
        # other detectors it outputs no facial keypoints (output0: [1,5,N] =
        # cx,cy,w,h,score); landmarks are synthesised from the bbox downstream.
        "model_name": "YoloFace11nVR180",
        "local_path": f"{models_dir}/yoloface_11n-vr180.onnx",
        "hash": "7572f27c2930ff83d24f95fb4b6321ea0d7f5883cb40ed647fa80889e8d4e4d8",
        "url": "https://github.com/Glat0s/yolo-face/releases/download/vr/yoloface_11n-vr180.onnx",
    },
    {
        # Yolov12-face trained at 640×640 for VR180 face detection. Unlike the
        # other detectors it outputs no facial keypoints (output0: [1,5,N] =
        # cx,cy,w,h,score); landmarks are synthesised from the bbox downstream.
        "model_name": "YoloFace12nVR180",
        "local_path": f"{models_dir}/yoloface_12n-vr180.onnx",
        "hash": "8d5e9187fe5aa8f95be7a3267802639299fb39df969407d22bc379eed9aaf1e1",
        "url": "https://github.com/Glat0s/yolo-face/releases/download/vr/yoloface_12n-vr180.onnx",
    },
    {
        "model_name": "FaceLandmark5",
        "local_path": f"{models_dir}/res50.onnx",
        "hash": "025db4efa3f7bef9911adc8eb92663608c682696a843cc7e1116d90c223354b5",
        "url": f"{assets_repo}/v0.1.0/res50.onnx",
    },
    {
        "model_name": "FaceLandmark68",
        "local_path": f"{models_dir}/2dfan4.onnx",
        "hash": "1ceedb108439c7d7b3f92cfa2b25bdc69a1f5f6c8b41da228cb283ca98d4181d",
        "url": f"{assets_repo}/v0.1.0/2dfan4.onnx",
    },
    {
        "model_name": "FaceLandmark3d68",
        "local_path": f"{models_dir}/1k3d68.onnx",
        "hash": "df5c06b8a0c12e422b2ed8947b8869faa4105387f199c477af038aa01f9a45cc",
        "url": f"{assets_repo}/v0.1.0/1k3d68.onnx",
    },
    {
        "model_name": "FaceLandmark98",
        "local_path": f"{models_dir}/peppapig_teacher_Nx3x256x256.onnx",
        "hash": "d4aa6dbd0081763a6eef04bf51484175b6a133ed12999bdc83b681a03f3f87d2",
        "url": f"{assets_repo}/v0.1.0/peppapig_teacher_Nx3x256x256.onnx",
    },
    {
        "model_name": "FaceLandmark106",
        "local_path": f"{models_dir}/2d106det.onnx",
        "hash": "f001b856447c413801ef5c42091ed0cd516fcd21f2d6b79635b1e733a7109dbf",
        "url": f"{assets_repo}/v0.1.0/2d106det.onnx",
    },
    {
        "model_name": "FaceLandmark203",
        "local_path": f"{models_dir}/landmark.onnx",
        "hash": "31d22a5041326c31f19b78886939a634a5aedcaa5ab8b9b951a1167595d147db",
        "url": f"{assets_repo}/v0.1.0/landmark.onnx",
    },
    {
        "model_name": "FaceLandmark478",
        "local_path": f"{models_dir}/face_landmarks_detector_Nx3x256x256.onnx",
        "hash": "6d7932bdefc38871f57dd915b8c723d855e599f29cf4cdf19616fb35d0ed572e",
        "url": f"{assets_repo}/v0.1.0/face_landmarks_detector_Nx3x256x256.onnx",
    },
    {
        # TUFA (IJCV 2025), 98-point WFLW topology. ViT-S/8 + DETR-style decoder with
        # the structure prompt baked in as a constant. Best published WFLW pose-subset
        # NME of the models evaluated (6.48 vs STAR's 6.79).
        # Input: RGB float32 [0,1], NCHW 1x3x256x256 (ImageNet normalisation is inside
        # the graph). Output "landmarks" (1,98,2) is NORMALISED — multiply by 256.
        # NOT fp16-safe: see the note above fp16_safe_models_list.
        "model_name": "FaceLandmarkTUFA98",
        "local_path": f"{models_dir}/tufa_vits8_256_98pt.onnx",
        "hash": "cf8fab1d1e748b3a4b9f7e8421620659b0219d4c6a69792438086c6d610e52cc",
        "url": f"{tufa_repo}/tufa_vits8_256_98pt.onnx",
    },
    {
        # Same TUFA weights as above, exported with the 314-point structure prompt
        # (Prompt/shape_314.npz in the export fork) frozen into the graph. TUFA queries
        # a point by its anchor position in a canonical mean face, so one checkpoint
        # serves any topology; the ViT-S/8 encoder dominates the cost and the extra
        # decoder queries are essentially free (measured identical to the 98-point
        # graph, CUDA EP fp32).
        # This 314-point set is TUFA's own dense definition, NOT a dataset topology:
        # anchors 0..311 are a dense sampling of the facial regions ordered by x in the
        # canonical face, with the two pupil anchors appended at 312/313. 20 of the 98
        # WFLW anchors appear in it verbatim, which is where the 5-point indices in
        # faceutil.convert_face_landmark_314_to_5 come from.
        # Input: RGB float32 [0,1], NCHW 1x3x256x256 (ImageNet normalisation is inside
        # the graph). Output "landmarks" (1,314,2) is NORMALISED — multiply by 256.
        # NOT fp16-safe: see the note above fp16_safe_models_list.
        "model_name": "FaceLandmarkTUFA314",
        "local_path": f"{models_dir}/tufa_vits8_256_314pt.onnx",
        "hash": "0e848e4e1a8ce18404f9c42bc83ecd08c048ffda16d061dd7f21e98c967f971c",
        "url": f"{tufa_repo}/tufa_vits8_256_314pt.onnx",
    },
    {
        # ORFormer (WACV 2025 oral), 98-point WFLW topology. The upstream two-stage
        # pipeline (VQ-VAE+ORFormer heatmap generator at 64px, then StackedHGNet at
        # 256px) is fused into one graph, so the 64px downscale happens internally.
        # Input: RGB float32 [0,1], NCHW 1x3x256x256.
        # Outputs: "landmarks" (1,98,2) already in 256-crop PIXELS, and "occlusion"
        # (1,1,16,16) — a per-patch non-visibility score no other landmark model here
        # provides. NOT fp16-safe: see the note above fp16_safe_models_list.
        "model_name": "FaceLandmarkORFormer98",
        "local_path": f"{models_dir}/orformer_hgnet_wflw_98pt_256.onnx",
        "hash": "219835e107a44cebf73ce3b8d592b0ed8e2f25400bee918e8fccab36fbb43f1b",
        "url": f"{orformer_repo}/orformer_hgnet_wflw_98pt_256.onnx",
    },
    {
        # HRFFA (High-Angle Robust Fast FaceAlignment), 68-point ibug topology — the
        # same layout FaceLandmark68 (2dfan4) emits, so it reuses
        # faceutil.convert_face_landmark_68_to_5.
        # This is the `vitt` student (ViT-T/16, DINOv3-ViT-L teacher distilled away):
        # the most accurate model the author publishes as ONNX apart from the 1.2 GB
        # teacher, which additionally carries Meta's DINOv3 licence.
        # Unlike every other landmark model here it is trained on WHOLE-HEAD crops,
        # not face crops — hence the DEIMv2Wholebody49Head dependency below. Feeding
        # it a face-tight crop is out of distribution and throws away the high-angle
        # robustness that is the entire point of the model.
        # Input "images": RGB float32, NCHW 1x3x256x256, center05 normalised
        # ((x/255 - 0.5) / 0.5); the graph folds nothing.
        # Outputs: "points" (1,68,2) NORMALISED to the crop — multiply by 256 — and
        # "vis_logits" (1,68,3) per-point visibility (0=outside image / 1=occluded /
        # 2=visible), which we leave unbound so ORT prunes it.
        # NOT fp16-safe: see the note above fp16_safe_models_list.
        "model_name": "FaceLandmarkHRFFA",
        "local_path": f"{models_dir}/hrffa_vitt_ibug68_1x3x256x256.onnx",
        "hash": "f849af432427ae6d2e9fc1ce2d9bd27134af6e2e3a00060d0d7d312ccfc1d602",
        "url": f"{hrffa_repo}/hrffa_vitt_ibug68_1x3x256x256.onnx",
    },
    {
        # DEIMv2-Wholebody49 head detector — not a face detector and deliberately
        # absent from FaceDetectors.detector_map. It exists only to give
        # FaceLandmarkHRFFA the whole-head box it crops from, and is force-loaded by
        # FaceLandmarkDetectors the way 478 force-loads FaceBlendShapes.
        # This is the hgnetv2_n variant: PP-HGNetV2 backbone, so the lineage is
        # Apache-2.0 with no DINOv3 licence attached, and it is both the smallest and
        # the fastest of the three boxes-only exports.
        # Input "images": RGB float32 /255 (no mean/std), NCHW with a fixed H/W read
        # off the graph, produced by a DIRECT resize — the aspect-preserving letterbox
        # in FaceDetectors._prepare_detection_image is not what this model expects.
        # Some exports add an "orig_target_sizes" (1,2) float32 = [W, H] input, in
        # which case the boxes come out in absolute pixels; without it they are
        # normalised to [0,1]. The reader introspects the session and handles both.
        # Output "label_xyxy_score" (1,Q,6) = (class, x1, y1, x2, y2, score). DETR-
        # style one-to-one matching, so it is already NMS-free. Class 7 = head, in
        # both the Wholebody49 and Wholebody34 vocabularies.
        # NOT fp16-safe: see the note above fp16_safe_models_list.
        "model_name": "DEIMv2Wholebody49Head",
        "local_path": (
            f"{models_dir}/deimv2_hgnetv2_n_wholebody49_boxes_only_webgpu.onnx"
        ),
        "hash": "f24d8fa18583c75ce8d802e3d983af9d13a07fdc294267968a206dc79df85d36",
        "url": (f"{hrffa_repo}/deimv2_hgnetv2_n_wholebody49_boxes_only_webgpu.onnx"),
    },
    {
        "model_name": "FaceBlendShapes",
        "local_path": f"{models_dir}/face_blendshapes_Nx146x2.onnx",
        "hash": "79065a18016da3b95f71247ff9ade3fe09b9124903a26a1af85af6d9e2a4faf3",
        "url": f"{assets_repo}/v0.1.0/face_blendshapes_Nx146x2.onnx",
    },
    {
        "model_name": "Inswapper128ArcFace",
        "local_path": f"{models_dir}/w600k_r50.onnx",
        "hash": "4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43",
        "url": f"{assets_repo}/v0.1.0/w600k_r50.onnx",
    },
    {
        "model_name": "SimSwapArcFace",
        "local_path": f"{models_dir}/simswap_arcface_model.onnx",
        "hash": "58949c864ab4a89012aaefc117f1ab8548c5f470bbc3889474bca13a412fc843",
        "url": f"{assets_repo}/v0.1.0/simswap_arcface_model.onnx",
    },
    {
        "model_name": "GhostArcFace",
        "local_path": f"{models_dir}/ghost_arcface_backbone.onnx",
        "hash": "18bb8057d1cd3ca39411b8a4dde485fa55783e08ceecaf2352f551ca39cd1357",
        "url": f"{assets_repo}/v0.1.0/ghost_arcface_backbone.onnx",
    },
    {
        "model_name": "CSCSArcFace",
        "local_path": f"{models_dir}/cscs_arcface_model.onnx",
        "hash": "cd81a1745a736402d100d32c362918aee46d9a3f58c9c5ecbf0e415cf2df9dc0",
        "url": f"{assets_repo}/v0.1.0/cscs_arcface_model.onnx",
    },
    {
        "model_name": "CSCSIDArcFace",
        "local_path": f"{models_dir}/cscs_id_adapter.onnx",
        "hash": "288ee88fa208e64846261f9c16f19362db000074b2f4c9000ea49b2311a6d55b",
        "url": f"{assets_repo}/v0.1.0/cscs_id_adapter.onnx",
    },
    {
        "model_name": "GFPGANv1.4",
        "local_path": f"{models_dir}/GFPGANv1.4.onnx",
        "hash": "6548e54cbcf248af385248f0c1193b359c37a0f98b836282b09cf48af4fd2b73",
        "url": f"{assets_repo}/v0.1.0/GFPGANv1.4.onnx",
    },
    {
        "model_name": "GFPGAN1024",
        "local_path": f"{models_dir}/gfpgan-1024.onnx",
        "hash": "ee8dd6415e388b3a410689d5d9395a2bf50b5973b588421ebfa57bc266f19e24",
        "url": "https://github.com/Glat0s/GFPGAN-1024-onnx/releases/download/v0.0.1/gfpgan-1024.onnx",
    },
    {
        "model_name": "GPENBFR256",
        "local_path": f"{models_dir}/GPEN-BFR-256.onnx",
        "hash": "aa5bd3ab238640a378c59e4a560f7a7150627944cf2129e6311ae4720e833271",
        "url": f"{assets_repo}/v0.1.0/GPEN-BFR-256.onnx",
    },
    {
        "model_name": "GPENBFR512",
        "local_path": f"{models_dir}/GPEN-BFR-512.onnx",
        "hash": "0960f836488735444d508b588e44fb5dfd19c68fde9163ad7878aa24d1d5115e",
        "url": f"{assets_repo}/v0.1.0/GPEN-BFR-512.onnx",
    },
    {
        "model_name": "GPENBFR1024",
        "local_path": f"{models_dir}/GPEN-BFR-1024.onnx",
        "hash": "cec8892093d7b99828acde97bf231fb0964d3fb11b43f3b0951e36ef1e192a3e",
        "url": f"{assets_repo}/v0.1.0/GPEN-BFR-1024.onnx",
    },
    {
        "model_name": "GPENBFR2048",
        "local_path": f"{models_dir}/GPEN-BFR-2048.onnx",
        "hash": "d0229ff43f979c360bd19daa9cd0ce893722d59f41a41822b9223ebbe4f89b3e",
        "url": f"{assets_repo}/v0.1.0/GPEN-BFR-2048.onnx",
    },
    {
        "model_name": "CodeFormer",
        "local_path": f"{models_dir}/codeformer_fp16.onnx",
        "hash": "9c3ae2ce2de616815815628f966cdef5d9466722434a1be00c0785ec92e2a94f",
        "url": f"{assets_repo}/v0.1.0/codeformer_fp16.onnx",
    },
    {
        "model_name": "VQFRv2",
        "local_path": f"{models_dir}/VQFRv2.fp16.onnx",
        "hash": "30c3d854c8e5c8abaf9c83c00d2466b7c3f64865d7b3b8596f56714a717ffd6f",
        "url": f"{assets_repo}/v0.1.0/VQFRv2.fp16.onnx",
    },
    {
        "model_name": "RestoreFormerPlusPlus",
        "local_path": f"{models_dir}/RestoreFormerPlusPlus.fp16.onnx",
        "hash": "e5df99ed4f501be2009ed8e708f407dd26ac400c55a43a01d8c8c157bc475b3f",
        "url": f"{assets_repo}/v0.1.0/RestoreFormerPlusPlus.fp16.onnx",
    },
    {
        "model_name": "OSDFacePromptEncoder",
        "local_path": f"{models_dir}/osdface/prompt_encoder.onnx",
        "hash": "83187cb142963151ff8abb7454e119e0a7e248e17c03c8deaa9a14bd6ba8f2a9",
        "url": f"{osdface_repo}/prompt_encoder.onnx",
    },
    {
        "model_name": "OSDFaceVAEEncoder",
        "local_path": f"{models_dir}/osdface/vae_encoder.onnx",
        "hash": "95f6d278737a864b02f99e51ac8cd00bdfb6c0b515b6d157470689fe0257dfa7",
        "url": f"{osdface_repo}/vae_encoder.onnx",
    },
    {
        "model_name": "OSDFaceUNet",
        "local_path": f"{models_dir}/osdface/unet.onnx",
        "hash": "b14bdaa36274da7f80f1a628aecbf7a9029239dad6223918432a16430e15c9e4",
        "url": f"{osdface_repo}/unet.onnx",
    },
    {
        "model_name": "OSDFaceUNetData",
        "local_path": f"{models_dir}/osdface/unet.onnx.data",
        "hash": "89a8ed18e13a5254874d567f6fc3607363af0103b249679951f167b6580bcc2c",
        "multipart_zip": {
            "member": "unet.onnx.data",
            "hash": "ced4b8a667a54f92b1ac149f8f5006c7b7eb4770d8301f82aaea34ec281b51cf",
            "parts": [
                {
                    "model_name": "OSDFaceUNetDataZip001",
                    "local_path": f"{models_dir}/osdface/unet.onnx.data.zip.001",
                    "hash": "e23d1a35c93ede05d00a78359381b5804368b7951596244a502dd92856e9eea3",
                    "url": f"{osdface_repo}/unet.onnx.data.zip.001",
                },
                {
                    "model_name": "OSDFaceUNetDataZip002",
                    "local_path": f"{models_dir}/osdface/unet.onnx.data.zip.002",
                    "hash": "7a404682246d84ed3d5f0e3ab8b24f062bc3d723c79e701bcec440473fb4b9d1",
                    "url": f"{osdface_repo}/unet.onnx.data.zip.002",
                },
            ],
        },
    },
    {
        "model_name": "OSDFaceVAEDecoder",
        "local_path": f"{models_dir}/osdface/vae_decoder.onnx",
        "hash": "d3ac89733f86e445b3d106e801860455cb5f44f289c12cc4673c0dfccff4e051",
        "url": f"{osdface_repo}/vae_decoder.onnx",
    },
    {
        "model_name": "OSDFaceScheduler",
        "local_path": f"{models_dir}/osdface/scheduler.json",
        "hash": "fa7586cae137df656b274b2fdbfb551e95bc9fe42334354007c2ba13db6fd0c7",
        "url": f"{osdface_repo}/scheduler.json",
    },
    {
        "model_name": "RealEsrganx2Plus",
        "local_path": f"{models_dir}/RealESRGAN_x2plus.fp16.onnx",
        "hash": "0b1770bcb31b3a9021d4251b538da4eb47c84f42706504d44a76d17e8c267606",
        "url": f"{assets_repo}/v0.1.0/RealESRGAN_x2plus.fp16.onnx",
    },
    {
        "model_name": "RealEsrganx4Plus",
        "local_path": f"{models_dir}/RealESRGAN_x4plus.fp16.onnx",
        "hash": "0a06c68f463a14bf5563b78d77d61ba4394024e148383c4308d6d3783eac2dc5",
        "url": f"{assets_repo}/v0.1.0/RealESRGAN_x4plus.fp16.onnx",
    },
    {
        "model_name": "RealEsrx4v3",
        "local_path": f"{models_dir}/realesr-general-x4v3.onnx",
        "hash": "09b757accd747d7e423c1d352b3e8f23e77cc5742d04bae958d4eb8082b76fa4",
        "url": f"{assets_repo}/v0.1.0/realesr-general-x4v3.onnx",
    },
    {
        "model_name": "BSRGANx2",
        "local_path": f"{models_dir}/BSRGANx2.fp16.onnx",
        "hash": "ba3a43613f5d2434c853201411b87e75c25ccb5b5918f38af504e4cf3bd4df9a",
        "url": f"{assets_repo}/v0.1.0/BSRGANx2.fp16.onnx",
    },
    {
        "model_name": "BSRGANx4",
        "local_path": f"{models_dir}/BSRGANx4.fp16.onnx",
        "hash": "e1467fbe60d2846919480f55a12ddbd5c516e343685bcdeac50ddcfa1dde2f46",
        "url": f"{assets_repo}/v0.1.0/BSRGANx4.fp16.onnx",
    },
    {
        "model_name": "UltraSharpx4",
        "local_path": f"{models_dir}/4x-UltraSharp.fp16.onnx",
        "hash": "d801b7f6081746e0b2cccef407c7a8acdb95e284c89298684582a8f2b35ad0f9",
        "url": f"{assets_repo}/v0.1.0/4x-UltraSharp.fp16.onnx",
    },
    {
        "model_name": "UltraMixx4",
        "local_path": f"{models_dir}/4x-UltraMix_Smooth.fp16.onnx",
        "hash": "3b96d63c239121b1ad5992e42a2089d6b4e1185c493c6440adfeafc0a20591eb",
        "url": f"{assets_repo}/v0.1.0/4x-UltraMix_Smooth.fp16.onnx",
    },
    {
        "model_name": "DeoldifyArt",
        "local_path": f"{models_dir}/ColorizeArtistic.fp16.onnx",
        "hash": "c8ad5c54b1b333361e959fdc6591828931b731f6652055f891d6118532cad081",
        "url": f"{assets_repo}/v0.1.0/ColorizeArtistic.fp16.onnx",
    },
    {
        "model_name": "DeoldifyStable",
        "local_path": f"{models_dir}/ColorizeStable.fp16.onnx",
        "hash": "666811485bfd37b236fdef695dbf50de7d3a430b10dbf5a3001d1609de06ad88",
        "url": f"{assets_repo}/v0.1.0/ColorizeStable.fp16.onnx",
    },
    {
        "model_name": "DeoldifyVideo",
        "local_path": f"{models_dir}/ColorizeVideo.fp16.onnx",
        "hash": "4d93b3cca8aa514bdf18a0ed00b25e36de5a9cc70b7aec7e60132632f6feced3",
        "url": f"{assets_repo}/v0.1.0/ColorizeVideo.fp16.onnx",
    },
    {
        "model_name": "DDColorArt",
        "local_path": f"{models_dir}/ddcolor_artistic.onnx",
        "hash": "2f2510323e59995051eeac4f1ef8c267130eabf6187535defa55c11929b2b31c",
        "url": f"{assets_repo}/v0.1.0/ddcolor_artistic.onnx",
    },
    {
        "model_name": "DDcolor",
        "local_path": f"{models_dir}/ddcolor.onnx",
        "hash": "4e8b8a8d7c346ea7df08fc0bc985d30c67f5835cd1b81b6728f6bbe8b7658ae1",
        "url": f"{assets_repo}/v0.1.0/ddcolor.onnx",
    },
    {
        "model_name": "Occluder",
        "local_path": f"{models_dir}/occluder.onnx",
        "hash": "79f5c2edf10b83458693d122dd51488b210fb80c059c5d56347a047710d44a78",
        "url": f"{assets_repo}/v0.1.0/occluder.onnx",
    },
    {
        "model_name": "XSeg",
        "local_path": f"{models_dir}/XSeg_model.onnx",
        "hash": "4381395dcbec1eef469fa71cfb381f00ac8aadc3e5decb4c29c36b6eb1f38ad9",
        "url": f"{assets_repo}/v0.1.0/XSeg_model.onnx",
    },
    {
        "model_name": "FaceParser",
        "local_path": f"{models_dir}/faceparser_resnet34.onnx",
        "hash": "5b805bba7b5660ab7070b5a381dcf75e5b3e04199f1e9387232a77a00095102e",
        "url": f"{assets_repo}/v0.1.0/faceparser_resnet34.onnx",
    },
    {
        "model_name": "combo_relu3_3_relu3_1",
        "local_path": f"{models_dir}/vgg_combo_relu3_3_relu3_1.onnx",
        "hash": "1068ee41e3c67dcfbbeccbc93e539eb06f89bba08618951bb33e9be2c1fbc986",
        "url": "https://github.com/asdf31jsa/VisoMaster-Experimental/raw/refs/heads/ALL_Working/model_assets/vgg_combo_relu3_3_relu3_1.onnx",
    },
    {
        "model_name": "RD64ClipText",
        "local_path": f"{models_dir}/rd64-uni-refined.pth",
        "hash": "a4956f9a7978a75630b08c9d6ec075b7c51cf43b4751b686e3a011d4012ddc9d",
        "url": f"{assets_repo}/v0.1.0/rd64-uni-refined.pth",
    },
    {
        "model_name": "LivePortraitMotionExtractor",
        "local_path": f"{models_dir}/liveportrait_onnx/motion_extractor.onnx",
        "hash": "99d4b3c9dd3fd301910de9415a29560e38c0afaa702da51398281376cc36fdd3",
        "url": f"{assets_repo}/v0.1.0_lp/motion_extractor.onnx",
    },
    {
        "model_name": "LivePortraitAppearanceFeatureExtractor",
        "local_path": f"{models_dir}/liveportrait_onnx/appearance_feature_extractor.onnx",
        "hash": "dbbbb44e4bba12302d7137bdee6a0f249b45fb6dd879509fd5baa27d70c40e32",
        "url": f"{assets_repo}/v0.1.0_lp/appearance_feature_extractor.onnx",
    },
    {
        "model_name": "LivePortraitStitchingEye",
        "local_path": f"{models_dir}/liveportrait_onnx/stitching_eye.onnx",
        "hash": "251004fe4a994c57c8cd9f2c50f3d89feb289fb42e6bc3af74470a3a9fa7d83b",
        "url": f"{assets_repo}/v0.1.0_lp/stitching_eye.onnx",
    },
    {
        "model_name": "LivePortraitStitchingLip",
        "local_path": f"{models_dir}/liveportrait_onnx/stitching_lip.onnx",
        "hash": "1ca793eac4b0dc5464f1716cdaa62e595c2c2272c9971a444e39c164578dc34b",
        "url": f"{assets_repo}/v0.1.0_lp/stitching_lip.onnx",
    },
    {
        "model_name": "LivePortraitStitching",
        "local_path": f"{models_dir}/liveportrait_onnx/stitching.onnx",
        "hash": "43598e9747a19f4c55d8e1604fb7d7fa70ab22377d129cb7d1fe38c9a737cc79",
        "url": f"{assets_repo}/v0.1.0_lp/stitching.onnx",
    },
    {
        "model_name": "LivePortraitWarpingSpade",
        "local_path": f"{models_dir}/liveportrait_onnx/warping_spade.onnx",
        "hash": "d6ee9af4352b47e88e0521eba6b774c48204afddc8d91c671a5f7b8a0dfb4971",
        "url": f"{assets_repo}/v0.1.0_lp/warping_spade.onnx",
    },
    # --- PerformRecast (expression-only "Recast" mode of the Face Expression Restorer) ---
    {
        "model_name": "PerformRecastAppearanceFeatureExtractor",
        "local_path": f"{models_dir}/performrecast_onnx/appearance_feature_extractor.onnx",
        "hash": "208e4f848b430cbfa71a36dab7ec25a5b345882f846dbb27288abcfb2ae89a96",
        "url": "https://github.com/Glat0s/PerformRecast-onnx/releases/download/onnx-v1/appearance_feature_extractor.onnx",
    },
    {
        "model_name": "PerformRecastMotionExtractor",
        "local_path": f"{models_dir}/performrecast_onnx/motion_extractor.onnx",
        "hash": "b1b26c1b6d7520eb8020175050f0381c4f402ccaa6afbaebee259da4ff9dcb6c",
        "url": "https://github.com/Glat0s/PerformRecast-onnx/releases/download/onnx-v1/motion_extractor.onnx",
    },
    {
        "model_name": "PerformRecastWarpingModule",
        "local_path": f"{models_dir}/performrecast_onnx/warping_module.onnx",
        "hash": "92d8a1414a31a4117237bbfb667be02e71831a085af6697c9c3465200228a0ce",
        "url": "https://github.com/Glat0s/PerformRecast-onnx/releases/download/onnx-v1/warping_module.onnx",
    },
    {
        "model_name": "PerformRecastSpadeGenerator",
        "local_path": f"{models_dir}/performrecast_onnx/spade_generator.onnx",
        "hash": "4d8127313b1c2f6b53320b65208dafc22db260550f690cfa114dec646c7a8f5f",
        "url": "https://github.com/Glat0s/PerformRecast-onnx/releases/download/onnx-v1/spade_generator.onnx",
    },
    {
        "model_name": "RefLDMVAEEncoder",
        "local_path": f"{models_dir}/ref_ldm_vae_encoder.onnx",
        "hash": "b88d18e79bb0dc2a0d2763e4fd806d6ce7f885a6503a828ab862a7c284d456fc",
        "url": "https://github.com/Glat0s/ref-ldm-onnx/releases/download/v0.0.1/ref_ldm_vae_encoder.onnx",
    },
    {
        "model_name": "RefLDMVAEDecoder",
        "local_path": f"{models_dir}/ref_ldm_vae_decoder.onnx",
        "hash": "eca3065e6a40f4f73a0a14bc810769d07563a351964a0830ad59a481aa00b4f5",
        "url": "https://github.com/Glat0s/ref-ldm-onnx/releases/download/v0.0.1/ref_ldm_vae_decoder.onnx",
    },
    {
        "model_name": "RefLDM_UNET_EXTERNAL_KV",
        "local_path": f"{models_dir}/ref_ldm_unet_external_kv.onnx",
        "hash": "56edbea2aaf0361607645bbe0f35ce07ff8ddce80ee0ef617af305d50d251154",
        "url": "https://github.com/Glat0s/ref-ldm-onnx/releases/download/v0.0.1/ref_ldm_unet_external_kv.onnx",
    },
    {
        "model_name": "LivePortraitWarpingSpadeFix",
        "local_path": f"{models_dir}/liveportrait_onnx/warping_spade-fix.onnx",
        "hash": "a6164debbf1e851c3dcefa622111c42a78afd9bb8f1540e7d01172ddf642c3b5",
        "url": f"{assets_repo}/v0.1.0_lp/warping_spade-fix.onnx",
    },
    {
        "model_name": "RefLdm",
        "local_path": f"{models_dir}/ref-ldm_embedding/ckpts/refldm.ckpt",
        "hash": "ad953ba72b52ed32dd280232ff0070bc6cb097a71dce250318730c884e38b778",
        "url": "https://github.com/ChiWeiHsiao/ref-ldm/releases/download/1.0.0/refldm.ckpt",
    },
    {
        "model_name": "VQGAN",
        "local_path": f"{models_dir}/ref-ldm_embedding/ckpts/vqgan.ckpt",
        "hash": "7b08407b454f5328aaaf1eda35418a5a53dcc68caaf3bcf12ab88b8f21ec1a5d",
        "url": "https://github.com/ChiWeiHsiao/ref-ldm/releases/download/1.0.0/vqgan.ckpt",
    },
    {
        "model_name": "FaceReaging",
        "local_path": f"{models_dir}/face_reaging.onnx",
        "hash": "62c62598a71067cf12680c8421230556d08069d172f1dc645be2a5ebe815fb1f",
        "url": "https://github.com/VisoMasterFusion/VisoMaster-Fusion/releases/download/v1.0.0/face_reaging.onnx",
    },
]
