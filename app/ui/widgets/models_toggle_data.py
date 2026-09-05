"""
Mapping registry linking AI models (ONNX/PyTorch) to their respective UI toggles.
Used by the ModelsProcessor.
"""

from typing import List, Dict
from enum import Enum
from dataclasses import dataclass


class ToggleScope(Enum):
    GLOBAL = "global"
    PER_FACE = "per_face"


@dataclass
class ToggleDef:
    key: str
    scope: ToggleScope


# Centralized dictionary mapping exact model_name (from models_data.py) to UI toggle definitions.
# If ANY of the mapped toggles is True in its respective scope, the model is allowed to run.
# Models missing from this list (like Core Detectors, Swappers, ArcFace or facial landmarks)
# are considered always active if requested, bypassing the UI state check.
MODELS_TOGGLE_MAP: Dict[str, List[ToggleDef]] = {
    # --- DENOISERS (ReF-LDM UNet & VAEs) ---
    "RefLDMVAEEncoder": [
        ToggleDef("DenoiserUNetEnableBeforeRestorersToggle", ToggleScope.GLOBAL),
        ToggleDef("DenoiserAfterFirstRestorerToggle", ToggleScope.GLOBAL),
        ToggleDef("DenoiserAfterRestorersToggle", ToggleScope.GLOBAL),
    ],
    "RefLDMVAEDecoder": [
        ToggleDef("DenoiserUNetEnableBeforeRestorersToggle", ToggleScope.GLOBAL),
        ToggleDef("DenoiserAfterFirstRestorerToggle", ToggleScope.GLOBAL),
        ToggleDef("DenoiserAfterRestorersToggle", ToggleScope.GLOBAL),
    ],
    "RefLDM_UNET_EXTERNAL_KV": [
        ToggleDef("DenoiserUNetEnableBeforeRestorersToggle", ToggleScope.GLOBAL),
        ToggleDef("DenoiserAfterFirstRestorerToggle", ToggleScope.GLOBAL),
        ToggleDef("DenoiserAfterRestorersToggle", ToggleScope.GLOBAL),
    ],
    "KVExtractor": [
        ToggleDef("DenoiserUNetEnableBeforeRestorersToggle", ToggleScope.GLOBAL),
        ToggleDef("DenoiserAfterFirstRestorerToggle", ToggleScope.GLOBAL),
        ToggleDef("DenoiserAfterRestorersToggle", ToggleScope.GLOBAL),
    ],
    # --- FRAME ENHANCERS (Background / Full Image) ---
    "RealEsrganx2Plus": [ToggleDef("FrameEnhancerEnableToggle", ToggleScope.GLOBAL)],
    "RealEsrganx4Plus": [ToggleDef("FrameEnhancerEnableToggle", ToggleScope.GLOBAL)],
    "RealEsrx4v3": [ToggleDef("FrameEnhancerEnableToggle", ToggleScope.GLOBAL)],
    "BSRGANx2": [ToggleDef("FrameEnhancerEnableToggle", ToggleScope.GLOBAL)],
    "BSRGANx4": [ToggleDef("FrameEnhancerEnableToggle", ToggleScope.GLOBAL)],
    "UltraSharpx4": [ToggleDef("FrameEnhancerEnableToggle", ToggleScope.GLOBAL)],
    "UltraMixx4": [ToggleDef("FrameEnhancerEnableToggle", ToggleScope.GLOBAL)],
    "DeoldifyArt": [ToggleDef("FrameEnhancerEnableToggle", ToggleScope.GLOBAL)],
    "DeoldifyStable": [ToggleDef("FrameEnhancerEnableToggle", ToggleScope.GLOBAL)],
    "DeoldifyVideo": [ToggleDef("FrameEnhancerEnableToggle", ToggleScope.GLOBAL)],
    "DDColorArt": [ToggleDef("FrameEnhancerEnableToggle", ToggleScope.GLOBAL)],
    "DDcolor": [ToggleDef("FrameEnhancerEnableToggle", ToggleScope.GLOBAL)],
    # --- FACE RESTORERS ---
    "GFPGANv1.4": [
        ToggleDef("FaceRestorerEnableToggle", ToggleScope.PER_FACE),
        ToggleDef("FaceRestorerEnable2Toggle", ToggleScope.PER_FACE),
    ],
    "GFPGAN1024": [
        ToggleDef("FaceRestorerEnableToggle", ToggleScope.PER_FACE),
        ToggleDef("FaceRestorerEnable2Toggle", ToggleScope.PER_FACE),
    ],
    "CodeFormer": [
        ToggleDef("FaceRestorerEnableToggle", ToggleScope.PER_FACE),
        ToggleDef("FaceRestorerEnable2Toggle", ToggleScope.PER_FACE),
    ],
    "GPENBFR256": [
        ToggleDef("FaceRestorerEnableToggle", ToggleScope.PER_FACE),
        ToggleDef("FaceRestorerEnable2Toggle", ToggleScope.PER_FACE),
    ],
    "GPENBFR512": [
        ToggleDef("FaceRestorerEnableToggle", ToggleScope.PER_FACE),
        ToggleDef("FaceRestorerEnable2Toggle", ToggleScope.PER_FACE),
    ],
    "GPENBFR1024": [
        ToggleDef("FaceRestorerEnableToggle", ToggleScope.PER_FACE),
        ToggleDef("FaceRestorerEnable2Toggle", ToggleScope.PER_FACE),
    ],
    "GPENBFR2048": [
        ToggleDef("FaceRestorerEnableToggle", ToggleScope.PER_FACE),
        ToggleDef("FaceRestorerEnable2Toggle", ToggleScope.PER_FACE),
    ],
    "RestoreFormerPlusPlus": [
        ToggleDef("FaceRestorerEnableToggle", ToggleScope.PER_FACE),
        ToggleDef("FaceRestorerEnable2Toggle", ToggleScope.PER_FACE),
    ],
    "VQFRv2": [
        ToggleDef("FaceRestorerEnableToggle", ToggleScope.PER_FACE),
        ToggleDef("FaceRestorerEnable2Toggle", ToggleScope.PER_FACE),
    ],
    "OSDFacePromptEncoder": [
        ToggleDef("FaceRestorerEnableToggle", ToggleScope.PER_FACE),
        ToggleDef("FaceRestorerEnable2Toggle", ToggleScope.PER_FACE),
    ],
    # --- RE-AGING ---
    "FaceReaging": [ToggleDef("FaceReagingEnableToggle", ToggleScope.PER_FACE)],
    # --- FACE MASKS / OCCLUDERS / TEXTURES ---
    "Occluder": [ToggleDef("OccluderEnableToggle", ToggleScope.PER_FACE)],
    "XSeg": [ToggleDef("DFLXSegEnableToggle", ToggleScope.PER_FACE)],
    "RD64ClipText": [ToggleDef("ClipEnableToggle", ToggleScope.PER_FACE)],
    "FaceParser": [
        ToggleDef("FaceParserEnableToggle", ToggleScope.PER_FACE),
        ToggleDef("RestoreEyesEnableToggle", ToggleScope.PER_FACE),
        ToggleDef("RestoreMouthEnableToggle", ToggleScope.PER_FACE),
        ToggleDef("MouthParserStretchToggle", ToggleScope.PER_FACE),
        ToggleDef("TransferTextureEnableToggle", ToggleScope.PER_FACE),
        ToggleDef("DifferencingEnableToggle", ToggleScope.PER_FACE),
        ToggleDef("DFLXSegEnableToggle", ToggleScope.PER_FACE),  # Pour le XSegMouth
    ],
    "combo_relu3_3_relu3_1": [
        ToggleDef("TransferTextureEnableToggle", ToggleScope.PER_FACE),
        ToggleDef("DifferencingEnableToggle", ToggleScope.PER_FACE),
    ],
    # --- FACE EDITORS & EXPRESSIONS (LivePortrait) ---
    "LivePortraitMotionExtractor": [
        ToggleDef("FaceExpressionEnableBothToggle", ToggleScope.PER_FACE),
        ToggleDef("AutoMouthExpressionEnableToggle", ToggleScope.PER_FACE),
        ToggleDef("FaceEditorEnableToggle", ToggleScope.PER_FACE),
        ToggleDef("FaceMakeupEnableToggle", ToggleScope.PER_FACE),
        ToggleDef("HairMakeupEnableToggle", ToggleScope.PER_FACE),
        ToggleDef("EyeBrowsMakeupEnableToggle", ToggleScope.PER_FACE),
        ToggleDef("LipsMakeupEnableToggle", ToggleScope.PER_FACE),
    ],
    "LivePortraitAppearanceFeatureExtractor": [
        ToggleDef("FaceExpressionEnableBothToggle", ToggleScope.PER_FACE),
        ToggleDef("AutoMouthExpressionEnableToggle", ToggleScope.PER_FACE),
        ToggleDef("FaceEditorEnableToggle", ToggleScope.PER_FACE),
        ToggleDef("FaceMakeupEnableToggle", ToggleScope.PER_FACE),
        ToggleDef("HairMakeupEnableToggle", ToggleScope.PER_FACE),
        ToggleDef("EyeBrowsMakeupEnableToggle", ToggleScope.PER_FACE),
        ToggleDef("LipsMakeupEnableToggle", ToggleScope.PER_FACE),
    ],
    "LivePortraitStitching": [
        ToggleDef("FaceExpressionEnableBothToggle", ToggleScope.PER_FACE),
        ToggleDef("AutoMouthExpressionEnableToggle", ToggleScope.PER_FACE),
        ToggleDef("FaceEditorEnableToggle", ToggleScope.PER_FACE),
        ToggleDef("FaceMakeupEnableToggle", ToggleScope.PER_FACE),
        ToggleDef("HairMakeupEnableToggle", ToggleScope.PER_FACE),
        ToggleDef("EyeBrowsMakeupEnableToggle", ToggleScope.PER_FACE),
        ToggleDef("LipsMakeupEnableToggle", ToggleScope.PER_FACE),
    ],
    "LivePortraitStitchingEye": [
        ToggleDef("FaceExpressionEnableBothToggle", ToggleScope.PER_FACE),
        ToggleDef("AutoMouthExpressionEnableToggle", ToggleScope.PER_FACE),
        ToggleDef("FaceEditorEnableToggle", ToggleScope.PER_FACE),
        ToggleDef("FaceMakeupEnableToggle", ToggleScope.PER_FACE),
        ToggleDef("HairMakeupEnableToggle", ToggleScope.PER_FACE),
        ToggleDef("EyeBrowsMakeupEnableToggle", ToggleScope.PER_FACE),
        ToggleDef("LipsMakeupEnableToggle", ToggleScope.PER_FACE),
        ToggleDef(
            "FaceExpressionRetargetingEyesBothEnableToggle", ToggleScope.PER_FACE
        ),
    ],
    "LivePortraitStitchingLip": [
        ToggleDef("FaceExpressionEnableBothToggle", ToggleScope.PER_FACE),
        ToggleDef("AutoMouthExpressionEnableToggle", ToggleScope.PER_FACE),
        ToggleDef("FaceEditorEnableToggle", ToggleScope.PER_FACE),
        ToggleDef("FaceMakeupEnableToggle", ToggleScope.PER_FACE),
        ToggleDef("HairMakeupEnableToggle", ToggleScope.PER_FACE),
        ToggleDef("EyeBrowsMakeupEnableToggle", ToggleScope.PER_FACE),
        ToggleDef("LipsMakeupEnableToggle", ToggleScope.PER_FACE),
        ToggleDef(
            "FaceExpressionRetargetingLipsBothEnableToggle", ToggleScope.PER_FACE
        ),
    ],
    "LivePortraitWarpingSpade": [
        ToggleDef("FaceExpressionEnableBothToggle", ToggleScope.PER_FACE),
        ToggleDef("AutoMouthExpressionEnableToggle", ToggleScope.PER_FACE),
        ToggleDef("FaceEditorEnableToggle", ToggleScope.PER_FACE),
        ToggleDef("FaceMakeupEnableToggle", ToggleScope.PER_FACE),
        ToggleDef("HairMakeupEnableToggle", ToggleScope.PER_FACE),
        ToggleDef("EyeBrowsMakeupEnableToggle", ToggleScope.PER_FACE),
        ToggleDef("LipsMakeupEnableToggle", ToggleScope.PER_FACE),
    ],
    "LivePortraitWarpingSpadeFix": [
        ToggleDef("FaceExpressionEnableBothToggle", ToggleScope.PER_FACE),
        ToggleDef("AutoMouthExpressionEnableToggle", ToggleScope.PER_FACE),
        ToggleDef("FaceEditorEnableToggle", ToggleScope.PER_FACE),
        ToggleDef("FaceMakeupEnableToggle", ToggleScope.PER_FACE),
        ToggleDef("HairMakeupEnableToggle", ToggleScope.PER_FACE),
        ToggleDef("EyeBrowsMakeupEnableToggle", ToggleScope.PER_FACE),
        ToggleDef("LipsMakeupEnableToggle", ToggleScope.PER_FACE),
    ],
    "PerformRecastAppearanceFeatureExtractor": [
        ToggleDef("FaceExpressionEnableBothToggle", ToggleScope.PER_FACE)
    ],
    "PerformRecastMotionExtractor": [
        ToggleDef("FaceExpressionEnableBothToggle", ToggleScope.PER_FACE)
    ],
    "PerformRecastWarpingModule": [
        ToggleDef("FaceExpressionEnableBothToggle", ToggleScope.PER_FACE)
    ],
    "PerformRecastSpadeGenerator": [
        ToggleDef("FaceExpressionEnableBothToggle", ToggleScope.PER_FACE)
    ],
}


def get_toggles_for_model(model_name: str) -> List[ToggleDef]:
    """
    Returns the list of UI toggle definitions associated with a specific model.
    """
    return MODELS_TOGGLE_MAP.get(model_name, [])
