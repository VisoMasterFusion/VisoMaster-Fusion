from app.helpers import miscellaneous as misc_helpers
from app.ui.widgets.actions import layout_actions
from app.helpers.typing_helper import LayoutDictTypes

# Widgets in Face Swap tab are created from this Layout
SWAPPER_LAYOUT_DATA: LayoutDictTypes = {
    'Swapper': {
        'SwapModelSelection': {
            'level': 1,
            'label': 'Swapper Model',
            'options': ['Inswapper128', 'InStyleSwapper256 Version A', 'InStyleSwapper256 Version B', 'InStyleSwapper256 Version C', 'DeepFaceLive (DFM)', 'SimSwap512', 'GhostFace-v1', 'GhostFace-v2', 'GhostFace-v3', 'CSCS', 'CanonSwap'],
            'default': 'Inswapper128',
            'help': 'Choose which swapper model to use for face swapping.'
        },
        'SwapperResSelection': {
            'level': 2,
            'label': 'Swapper Resolution',
            'options': ['128', '256', '384', '512'],
            'default': '512',
            'parentSelection': 'SwapModelSelection',
            'requiredSelectionValue': 'Inswapper128',
            'help': 'Select the resolution for the swapped face in pixels. Higher values offer better quality but are slower to process.'
        },
        'SwapperResAutoSelectEnableToggle': {
            'level': 2,
            'label': 'Enable Auto Resolution',
            'default': False,
            'parentSelection': 'SwapModelSelection',
            'requiredSelectionValue': 'Inswapper128',
            'help': 'Autoselect Swapper Resolution based on original Face Size (only for Inswapper).'
        },
        'InStyleResAEnableToggle': {
            'level': 2,
            'label': '512 Resolution',
            'default': False,            
            'parentSelection': 'SwapModelSelection',
            'requiredSelectionValue': 'InStyleSwapper256 Version A',
            'help': 'Like the inswapper Resolution (512) for InStyleSwappers. i dont know to hide it with 3 selections possible :(.'
        }, 
        'InStyleResBEnableToggle': {
            'level': 2,
            'label': '512 Resolution',
            'default': False,            
            'parentSelection': 'SwapModelSelection',
            'requiredSelectionValue': 'InStyleSwapper256 Version B',
            'help': 'Like the inswapper Resolution (512) for InStyleSwappers. i dont know to hide it with 3 selections possible :(.'
        }, 
        'InStyleResCEnableToggle': {
            'level': 2,
            'label': '512 Resolution',
            'default': False,            
            'parentSelection': 'SwapModelSelection',
            'requiredSelectionValue': 'InStyleSwapper256 Version C',
            'help': 'Like the inswapper Resolution (512) for InStyleSwappers. i dont know to hide it with 3 selections possible :(.'
        },
        'DFMModelSelection': {
            'level': 2,
            'label': 'DFM Model',
            'options': lambda manager: manager.get_selection_values(),
            'default': lambda manager: manager.get_default_value(),
            'parentSelection': 'SwapModelSelection',
            'requiredSelectionValue': 'DeepFaceLive (DFM)',
            'help': 'Select which pretrained DeepFaceLive (DFM) Model to use for swapping.'
        },
        'DFMAmpMorphSlider': {
            'level': 2,
            'label': 'AMP Morph Factor',
            'min_value': '1',
            'max_value': '100',
            'default': '50',
            'step': 1,
            'parentSelection': 'SwapModelSelection',
            'requiredSelectionValue': 'DeepFaceLive (DFM)',
            'help': 'AMP Morph Factor for DFM AMP Models',
        },
        'DFMRCTColorToggle': {
            'level': 2,
            'label': 'RCT Color Transfer',
            'default': False,
            'parentSelection': 'SwapModelSelection',
            'requiredSelectionValue': 'DeepFaceLive (DFM)',
            'help': 'RCT Color Transfer for DFM Models',
        },
        'SimilarityThresholdSlider': {
            'level': 1,
            'label': 'Similarity Threshold',
            'min_value': '1',
            'max_value': '100',
            'default': '60',
            'step': 1,
            'help': 'Set the similarity threshold to control how similar the detected face should be to the reference (target) face.'
        },
        'PreSwapSharpnessDecimalSlider': {
            'level': 1,
            'label': 'Pre Swap Sharpness (1.0)',
            'min_value': '0.0',
            'max_value': '2.0',
            'default': '1.0',
            'step': 0.1,
            'decimals': 1,
            'help': 'Sharpens the original face befor swapping. can sometimes be usefull. care it can tamper with "Auto Face Restorer"!'
        }
    },
    'Swap strenght and likeness': {
        'StrengthEnableToggle': {
            'level': 1,
            'label': 'Strength',
            'default': False,
            'help': 'Apply additional swapping iterations to increase the strength of the result, which may increase likeness.'
        },
        'StrengthAmountSlider': {
            'level': 2,
            'label': 'Amount',
            'min_value': '0',
            'max_value': '500',
            'default': '100',
            'step': 25,
            'parentToggle': 'StrengthEnableToggle',
            'requiredToggleValue': True,
            'help': 'Increase up to 5x additional swaps (500%). 200% is generally a good result. Set to 0 to turn off swapping but allow the rest of the pipeline to apply to the original image.'
        },
        'FaceLikenessEnableToggle': {
            'level': 1,
            'label': 'Face Likeness',
            'default': False,
            'help': 'This is a feature to perform direct adjustments to likeness of faces.'
        },
        'FaceLikenessFactorDecimalSlider': {
            'level': 2,
            'label': 'Amount',
            'min_value': '-1.00',
            'max_value': '1.00',
            'default': '0.00',
            'decimals': 2,
            'step': 0.05,
            'parentToggle': 'FaceLikenessEnableToggle',
            'requiredToggleValue': True,
            'help': 'Determines the factor of likeness between the source and assigned faces.'
        }
    },
    'Masks': {
        'MaskShowSelection':{
            'level': 1,
            'label': 'Mask view selection',
            'options': ['swap_mask', 'diff', 'texture'],
            'default': 'swap_mask',
            'help': 'select what mask is shown in "view face mask".'
        },
        'BordermaskEnableToggle': {
            'level': 1,
            'label': 'Border mask options',
            'default': False,
            'help': 'A rectangle with adjustable bottom, left, right, top, and sides that masks the swapped face result back into the original image.'
        },
        'BorderBottomSlider':{
            'level': 2,
            'label': 'Bottom Border',
            'min_value': '0',
            'max_value': '100',
            'default': '10',
            'step': 1,
            'parentToggle': 'BordermaskEnableToggle',
            'requiredToggleValue': True,
            'help': 'A rectangle with adjustable bottom, left, right, top, and sides that masks the swapped face result back into the original image.'
        },
        'BorderLeftSlider':{
            'level': 2,
            'label': 'Left Border',
            'min_value': '0',
            'max_value': '100',
            'default': '10',
            'step': 1,
            'parentToggle': 'BordermaskEnableToggle',
            'requiredToggleValue': True,
            'help': 'A rectangle with adjustable bottom, left, right, top, and sides that masks the swapped face result back into the original image.'
        },
        'BorderRightSlider':{
            'level': 2,
            'label': 'Right Border',
            'min_value': '0',
            'max_value': '100',
            'default': '10',
            'step': 1,
            'parentToggle': 'BordermaskEnableToggle',
            'requiredToggleValue': True,
            'help': 'A rectangle with adjustable bottom, left, right, top, and sides that masks the swapped face result back into the original image.'
        },
        'BorderTopSlider':{
            'level': 2,
            'label': 'Top Border',
            'min_value': '0',
            'max_value': '100',
            'default': '10',
            'step': 1,
            'parentToggle': 'BordermaskEnableToggle',
            'requiredToggleValue': True,
            'help': 'A rectangle with adjustable bottom, left, right, top, and sides that masks the swapped face result back into the original image.'
        },
        'BorderBlurSlider':{
            'level': 2,
            'label': 'Border Blur',
            'min_value': '0',
            'max_value': '100',
            'default': '10',
            'step': 1,
            'parentToggle': 'BordermaskEnableToggle',
            'requiredToggleValue': True,
            'help': 'Border mask blending distance.'
        },
        'OccluderEnableToggle': {
            'level': 1,
            'label': 'Occlusion Mask',
            'default': False,
            'help': 'Allow objects occluding the face to show up in the swapped image.'
        },
        'OccluderSizeSlider': {
            'level': 2,
            'label': 'Size',
            'min_value': '-100',
            'max_value': '100',
            'default': '0',
            'step': 1,
            'parentToggle': 'OccluderEnableToggle',
            'requiredToggleValue': True,
            'help': 'Grows or shrinks the occluded region'
        },
        'DFLXSegEnableToggle': {
            'level': 1,
            'label': 'DFL XSeg Mask',
            'default': False,
            'help': 'Allow objects occluding the face to show up in the swapped image.'
        },
        'DFLXSegSizeSlider': {
            'level': 2,
            'label': 'Size',
            'min_value': '-20',
            'max_value': '20',
            'default': '0',
            'step': 1,
            'parentToggle': 'DFLXSegEnableToggle',
            'requiredToggleValue': True,
            'help': 'Grows or shrinks the occluded region.'
        },
        'OccluderXSegBlurSlider': {
            'level': 1,
            'label': 'Occluder/DFL XSeg Blur',
            'min_value': '0',
            'max_value': '100',
            'default': '0',
            'step': 1,
            'parentToggle': 'OccluderEnableToggle | DFLXSegEnableToggle',
            'requiredToggleValue': True,
            'help': 'Blend value for Occluder and XSeg.'
        },
        'XSegMouthEnableToggle': {
            'level': 2,
            'label': 'Xseg Mouth',
            'default': False,
            'parentToggle': 'DFLXSegEnableToggle',
            'requiredToggleValue': True,
            'help': 'Use second Xseg Mask on mouth region.'
        },
        'DFLXSeg2SizeSlider': {
            'level': 3,
            'label': 'Size',
            'min_value': '-50',
            'max_value': '50',
            'default': '-1',
            'step': 1,
            'parentToggle': 'DFLXSegEnableToggle & XSegMouthEnableToggle',
            'requiredToggleValue': True,
            'help': 'Grows or shrinks the occluded mouth region Xseg Mask.'
        },
        'XSeg2BlurSlider': {
            'level': 3,
            'label': 'Blur',
            'min_value': '0',
            'max_value': '100',
            'default': '4',
            'step': 1,
            'parentToggle': 'DFLXSegEnableToggle & XSegMouthEnableToggle',
            'requiredToggleValue': True,
            'help': 'Blur for mouth XSeg Mask'
        },
        'XsegMouthParserSlider': {
            'level': 3,
            'label': 'Mouth',
            'min_value': '0',
            'max_value': '30',
            'default': '2',
            'step': 1,
            'parentToggle': 'DFLXSegEnableToggle & XSegMouthEnableToggle',
            'requiredToggleValue': True,
            'help': 'Grow Mouth Region (uses Faceparser on swap)'
        },
        'XsegUpperLipParserSlider': {
            'level': 3,
            'label': 'Upper Lip',
            'min_value': '0',
            'max_value': '30',
            'default': '2',
            'step': 1,
            'parentToggle': 'DFLXSegEnableToggle & XSegMouthEnableToggle',
            'requiredToggleValue': True,
            'help': 'Grow Upper Lip Region (uses Faceparser on swap)'
        },
        'XsegLowerLipParserSlider': {
            'level': 3,
            'label': 'Lower Lip',
            'min_value': '0',
            'max_value': '30',
            'default': '10',
            'step': 1,
            'parentToggle': 'DFLXSegEnableToggle & XSegMouthEnableToggle',
            'requiredToggleValue': True,
            'help': 'Grow Lower Lip Region (uses Faceparser on swap)'
        },
        'ClipEnableToggle': {
            'level': 1,
            'label': 'Text Masking',
            'default': False,
            'help': 'Use descriptions to identify objects that will be present in the final swapped image.'
        },
        'ClipText': {
            'level': 2,
            'label': 'Text Masking Entry',
            'min_value': '0',
            'max_value': '1000',
            'default': '',
            'width': 130,
            'parentToggle': 'ClipEnableToggle',
            'requiredToggleValue': True,
            'help': 'To use, type a word(s) in the box separated by commas and press <enter>.'
        },
        'ClipAmountSlider': {
            'level': 2,
            'label': 'Amount',
            'min_value': '0',
            'max_value': '100',
            'default': '50',
            'step': 1,
            'parentToggle': 'ClipEnableToggle',
            'requiredToggleValue': True,
            'help': 'Increase to strengthen the effect.'
        }
    },
    'Original Face Parsers': {
        'FaceParserEnableToggle': {
            'level': 1,
            'label': 'Face Parser Mask',
            'default': False,
            'help': 'Allow the unprocessed background from the orginal image to show in the final swap.'
        },
        'MouthParserInsideToggle': {
            'level': 2,
            'label': 'Mouth Inside toggle',
            'default': True,
            'parentToggle': 'FaceParserEnableToggle',
            'requiredToggleValue': True,
            'help': 'Keep the parse inside the mouth (value over 1 will slowly overlap the swapped mouth).'
        },
        'BackgroundParserSlider': {
            'level': 2,
            'label': 'Background',
            'min_value': '-50',
            'max_value': '50',
            'default': '0',
            'step': 1,
            'parentToggle': 'FaceParserEnableToggle',
            'requiredToggleValue': True,
            'help': 'Negative/Positive values shrink and grow the mask.'
        },
        'FaceParserSlider': {
            'level': 2,
            'label': 'Face',
            'min_value': '0',
            'max_value': '30',
            'default': '0',
            'step': 1,
            'parentToggle': 'FaceParserEnableToggle',
            'requiredToggleValue': True,
            'help': 'Adjust the size of the Mask. Mast the entire face.'
        },
        'LeftEyebrowParserSlider': {
            'level': 2,
            'label': 'Left Eyebrow',
            'min_value': '0',
            'max_value': '30',
            'default': '0',
            'step': 1,
            'parentToggle': 'FaceParserEnableToggle',
            'requiredToggleValue': True,
            'help': 'Adjust the size of the Mask. Mast the left eyebrow.'
        },
        'RightEyebrowParserSlider': {
            'level': 2,
            'label': 'Right Eyebrow',
            'min_value': '0',
            'max_value': '30',
            'default': '0',
            'step': 1,
            'parentToggle': 'FaceParserEnableToggle',
            'requiredToggleValue': True,
            'help': 'Adjust the size of the Mask. Mast the right eyebrow.'
        },
        'LeftEyeParserSlider': {
            'level': 2,
            'label': 'Left Eye',
            'min_value': '0',
            'max_value': '30',
            'default': '0',
            'step': 1,
            'parentToggle': 'FaceParserEnableToggle',
            'requiredToggleValue': True,
            'help': 'Adjust the size of the Mask. Mast the left eye.'
        },
        'RightEyeParserSlider': {
            'level': 2,
            'label': 'Right Eye',
            'min_value': '0',
            'max_value': '30',
            'default': '0',
            'step': 1,
            'parentToggle': 'FaceParserEnableToggle',
            'requiredToggleValue': True,
            'help': 'Adjust the size of the Mask. Mast the right eye.'
        },
        'EyeGlassesParserSlider': {
            'level': 2,
            'label': 'EyeGlasses',
            'min_value': '0',
            'max_value': '30',
            'default': '0',
            'step': 1,
            'parentToggle': 'FaceParserEnableToggle',
            'requiredToggleValue': True,
            'help': 'Adjust the size of the Mask. Mast the eyeglasses.'
        },
        'NoseParserSlider': {
            'level': 2,
            'label': 'Nose',
            'min_value': '0',
            'max_value': '30',
            'default': '0',
            'step': 1,
            'parentToggle': 'FaceParserEnableToggle',
            'requiredToggleValue': True,
            'help': 'Adjust the size of the Mask. Mast the nose.'
        },
        'MouthParserSlider': {
            'level': 2,
            'label': 'Mouth',
            'min_value': '0',
            'max_value': '30',
            'default': '0',
            'step': 1,
            'parentToggle': 'FaceParserEnableToggle',
            'requiredToggleValue': True,
            'help': 'Adjust the size of the Mask. Mast the inside of the mouth, including the tongue.'
        },
        'UpperLipParserSlider': {
            'level': 2,
            'label': 'Upper Lip',
            'min_value': '0',
            'max_value': '30',
            'default': '0',
            'step': 1,
            'parentToggle': 'FaceParserEnableToggle',
            'requiredToggleValue': True,
            'help': 'Adjust the size of the Mask. Mast the upper lip.'
        },
        'LowerLipParserSlider': {
            'level': 2,
            'label': 'Lower Lip',
            'min_value': '0',
            'max_value': '30',
            'default': '0',
            'step': 1,
            'parentToggle': 'FaceParserEnableToggle',
            'requiredToggleValue': True,
            'help': 'Adjust the size of the Mask. Mast the lower lip.'
        },
        'NeckParserSlider': {
            'level': 2,
            'label': 'Neck',
            'min_value': '0',
            'max_value': '30',
            'default': '0',
            'step': 1,
            'parentToggle': 'FaceParserEnableToggle',
            'requiredToggleValue': True,
            'help': 'Adjust the size of the Mask. Mast the neck.'
        },
        'HairParserSlider': {
            'level': 2,
            'label': 'Hair',
            'min_value': '0',
            'max_value': '30',
            'default': '0',
            'step': 1,
            'parentToggle': 'FaceParserEnableToggle',
            'requiredToggleValue': True,
            'help': 'Adjust the size of the Mask. Mast the hair.'
        },
        'BackgroundBlurParserSlider': {
            'level': 2,
            'label': 'Background Blur',
            'min_value': '0',
            'max_value': '100',
            'default': '5',
            'step': 1,
            'parentToggle': 'FaceParserEnableToggle',
            'requiredToggleValue': True,
            'help': 'Blend the value for Background Parser'
        },
        'FaceBlurParserSlider': {
            'level': 2,
            'label': 'Face Blur',
            'min_value': '0',
            'max_value': '100',
            'default': '5',
            'step': 1,
            'parentToggle': 'FaceParserEnableToggle',
            'requiredToggleValue': True,
            'help': 'Blend the value for Face Parser'
        },
        'FaceParserBlendSlider': {
            'level': 2,
            'label': 'Face Blend',
            'min_value': '0',
            'max_value': '100',
            'default': '0',
            'step': 1,
            'parentToggle': 'FaceParserEnableToggle',
            'requiredToggleValue': True,
            'help': 'Blend the value for Face Parser'
        },
        'RestoreEyesEnableToggle': {
            'level': 1,
            'label': 'Restore Eyes',
            'default': False,
            'help': 'Restore eyes from the original face.'
        },
        'RestoreEyesBlendAmountSlider': {
            'level': 2,
            'label': 'Eyes Blend Amount',
            'min_value': '1',
            'max_value': '100',
            'default': '50',
            'step': 1,
            'parentToggle': 'RestoreEyesEnableToggle',
            'requiredToggleValue': True,
            'help': 'Increase this to show more of the swapped eyes. Decrease it to show more of the original eyes.'
        },
        'RestoreEyesSizeFactorDecimalSlider': {
            'level': 2,
            'label': 'Eyes Size Factor',
            'min_value': '2.0',
            'max_value': '4.0',
            'default': '3.0',
            'decimals': 1,
            'step': 0.5,
            'parentToggle': 'RestoreEyesEnableToggle',
            'requiredToggleValue': True,
            'help': 'Reduce this when swapping faces zoomed out of the frame.'
        },
        'RestoreEyesFeatherBlendSlider': {
            'level': 2,
            'label': 'Eyes Feather Blend',
            'min_value': '1',
            'max_value': '100',
            'default': '10',
            'step': 1,
            'parentToggle': 'RestoreEyesEnableToggle',
            'requiredToggleValue': True,
            'help': 'Adjust the blending of eyes border. Increase this to show more of the original eyes. Decrease this to show more of the swapped eyes.'
        },
        'RestoreXEyesRadiusFactorDecimalSlider': {
            'level': 2,
            'label': 'X Eyes Radius Factor',
            'min_value': '0.3',
            'max_value': '3.0',
            'default': '1.0',
            'decimals': 1,
            'step': 0.1,
            'parentToggle': 'RestoreEyesEnableToggle',
            'requiredToggleValue': True,
            'help': 'These parameters determine the shape of the mask. If both are equal to 1.0, the mask will be circular. If either one is greater or less than 1.0, the mask will become oval, stretching or shrinking along the corresponding direction.'
        },
        'RestoreYEyesRadiusFactorDecimalSlider': {
            'level': 2,
            'label': 'Y Eyes Radius Factor',
            'min_value': '0.3',
            'max_value': '3.0',
            'default': '1.0',
            'decimals': 1,
            'step': 0.1,
            'parentToggle': 'RestoreEyesEnableToggle',
            'requiredToggleValue': True,
            'help': 'These parameters determine the shape of the mask. If both are equal to 1.0, the mask will be circular. If either one is greater or less than 1.0, the mask will become oval, stretching or shrinking along the corresponding direction.'
        },
        'RestoreXEyesOffsetSlider': {
            'level': 2,
            'label': 'X Eyes Offset',
            'min_value': '-300',
            'max_value': '300',
            'default': '0',
            'step': 1,
            'parentToggle': 'RestoreEyesEnableToggle',
            'requiredToggleValue': True,
            'help': 'Moves the Eyes Mask on the X Axis.'
        },
        'RestoreYEyesOffsetSlider': {
            'level': 2,
            'label': 'Y Eyes Offset',
            'min_value': '-300',
            'max_value': '300',
            'default': '0',
            'step': 1,
            'parentToggle': 'RestoreEyesEnableToggle',
            'requiredToggleValue': True,
            'help': 'Moves the Eyes Mask on the Y Axis.'
        },
        'RestoreEyesSpacingOffsetSlider': {
            'level': 2,
            'label': 'Eyes Spacing Offset',
            'min_value': '-200',
            'max_value': '200',
            'default': '0',
            'step': 1,
            'parentToggle': 'RestoreEyesEnableToggle',
            'requiredToggleValue': True,
            'help': 'Change the Eyes Spacing distance.'
        },
        'RestoreMouthEnableToggle': {
            'level': 1,
            'label': 'Restore Mouth',
            'default': False,
            'help': 'Restore mouth from the original face.'
        },
        'RestoreMouthBlendAmountSlider': {
            'level': 2,
            'label': 'Mouth Blend Amount',
            'min_value': '1',
            'max_value': '100',
            'default': '50',
            'step': 1,
            'parentToggle': 'RestoreMouthEnableToggle',
            'requiredToggleValue': True,
            'help': 'Increase this to show more of the swapped Mouth. Decrease it to show more of the original Mouth.'
        },       
        'RestoreMouthSizeFactorSlider': {
            'level': 2,
            'label': 'Mouth Size Factor',
            'min_value': '5',
            'max_value': '60',
            'default': '25',
            'step': 5,
            'parentToggle': 'RestoreMouthEnableToggle',
            'requiredToggleValue': True,
            'help': 'Increase this when swapping faces zoomed out of the frame.'
        },
        'RestoreMouthFeatherBlendSlider': {
            'level': 2,
            'label': 'Mouth Feather Blend',
            'min_value': '1',
            'max_value': '100',
            'default': '10',
            'step': 1,
            'parentToggle': 'RestoreMouthEnableToggle',
            'requiredToggleValue': True,
            'help': 'Adjust the border of Mouth blending. Increase this to show more of the original Mouth. Decrease this to show more of the swapped Mouth.'
        },
        'RestoreXMouthRadiusFactorDecimalSlider': {
            'level': 2,
            'label': 'X Mouth Radius Factor',
            'min_value': '0.3',
            'max_value': '3.0',
            'default': '1.0',
            'decimals': 1,
            'step': 0.1,
            'parentToggle': 'RestoreMouthEnableToggle',
            'requiredToggleValue': True,
            'help': 'These parameters determine the shape of the mask. If both are equal to 1.0, the mask will be circular. If either one is greater or less than 1.0, the mask will become oval, stretching or shrinking along the corresponding direction.'
        },
        'RestoreYMouthRadiusFactorDecimalSlider': {
            'level': 2,
            'label': 'Y Mouth Radius Factor',
            'min_value': '0.3',
            'max_value': '3.0',
            'default': '1.0',
            'decimals': 1,
            'step': 0.1,
            'parentToggle': 'RestoreMouthEnableToggle',
            'requiredToggleValue': True,
            'help': 'These parameters determine the shape of the mask. If both are equal to 1.0, the mask will be circular. If either one is greater or less than 1.0, the mask will become oval, stretching or shrinking along the corresponding direction.'
        },
        'RestoreXMouthOffsetSlider': {
            'level': 2,
            'label': 'X Mouth Offset',
            'min_value': '-300',
            'max_value': '300',
            'default': '0',
            'step': 1,
            'parentToggle': 'RestoreMouthEnableToggle',
            'requiredToggleValue': True,
            'help': 'Moves the Mouth Mask on the X Axis.'
        },
        'RestoreYMouthOffsetSlider': {
            'level': 2,
            'label': 'Y Mouth Offset',
            'min_value': '-300',
            'max_value': '300',
            'default': '0',
            'step': 1,
            'parentToggle': 'RestoreMouthEnableToggle',
            'requiredToggleValue': True,
            'help': 'Moves the Mouth Mask on the Y Axis.'
        },
        'RestoreEyesMouthBlurSlider': {
            'level': 1,
            'label': 'Eyes/Mouth Blur',
            'min_value': '0',
            'max_value': '50',
            'default': '0',
            'step': 1,
            'parentToggle': 'RestoreEyesEnableToggle | RestoreMouthEnableToggle',
            'requiredToggleValue': True,
            'help': 'Adjust the blur of mask border.'
        }
    },
    'Textures and colors': {
        'DifferencingEnableToggle': {
            'level': 1,
            'label': 'Differencing',
            'default': False,
            'help': 'Diffing between swap and original. advanced diffing model, which weights the difference between pixel.'
        },
        'DifferencingLowerLimitThreshSlider': {
            'level': 2,
            'label': 'Difference Lower Limit',
            'min_value': '0',
            'max_value': '100',
            'default': '30',
            'step': 1,
            'parentToggle': 'DifferencingEnableToggle',
            'requiredToggleValue': True,
            'help': 'Defines lower Limit for diffing mask. Check "View Face Mask" with "mask show -> diff" to see effect.'
        },   
        'DifferencingUpperLimitThreshSlider': {
            'level': 2,
            'label': 'Difference Upper Limit',
            'min_value': '0',
            'max_value': '100',
            'default': '50',
            'step': 1,
            'parentToggle': 'DifferencingEnableToggle',
            'requiredToggleValue': True,
            'help': 'Defines upper Limit for diffing mask. Check "View Face Mask" with "mask show -> diff" to see effect.'
        },                                     
        'DifferencingMiddleLimitValueSlider': {
            'level': 2,
            'label': 'Lower Strength',
            'min_value': '0',
            'max_value': '100',
            'default': '50',
            'step': 1,
            'parentToggle': 'DifferencingEnableToggle',
            'requiredToggleValue': True,
            'help': 'Defines the blending between swap and original for pixels under "Difference Lower Limit" in transfer mask. Check "View Face Mask" with "mask show -> diff" to see effect.'
        },       
        'DifferencingUpperLimitValueSlider': {
            'level': 2,
            'label': 'Upper Strength',
            'min_value': '0',
            'max_value': '100',
            'default': '100',
            'step': 1,
            'parentToggle': 'DifferencingEnableToggle',
            'requiredToggleValue': True,
            'help': 'Defines the blending between swap and original for pixels over "Difference Upper Limit" in transfer mask. Check "View Face Mask" with "mask show -> diff" to see effect.'
        },         
        'DifferencingBlendAmountSlider': {
            'level': 2,
            'label': 'Mask Blur Amount',
            'min_value': '0',
            'max_value': '100',
            'default': '16',
            'step': 1,
            'parentToggle': 'DifferencingEnableToggle',
            'requiredToggleValue': True,
            'help': 'Mask Blur for smoother transitions'
        },   
        'TransferTextureEnableToggle': {
            'level': 1,
            'label': 'Transfer Texture',
            'default': False,
            'help': 'Enable Texture Transfer'
        },   
        'TransferTextureModeEnableToggle': {
            'level': 2,
            'label': 'Mode 2',
            'default': False,
            'parentToggle': 'TransferTextureEnableToggle',
            'requiredToggleValue': True,
            'help': 'Different values for gradient magnitude calculation, sometimes better but sometimes too much stripes in face'
        },
        'TransferTextureBlendAmountSlider': {
            'level': 2,
            'label': 'Texture Strength Amount',
            'min_value': '0',
            'max_value': '100',
            'default': '30',
            'step': 1,
            'parentToggle': 'TransferTextureEnableToggle',
            'requiredToggleValue': True,
            'help': 'Define how strong texture Transfer is applied'
        },               
        'TransferTexturePreGammaDecimalSlider': {
            'level': 2,
            'label': 'Texture Gamma adjust',
            'min_value': '0.00',
            'max_value': '2.00',
            'default': '1.00',
            'decimals': 2,
            'step': 0.05,
            'parentToggle': 'TransferTextureEnableToggle',
            'requiredToggleValue': True,
            'help': 'adjusting can sometimes be good. Gamma adjust of original Face for Texture Transfer'
        },            
        'TransferTexturePreContrastDecimalSlider': {
            'level': 2,
            'label': 'Texture Contrast adjust',
            'min_value': '0.00',
            'max_value': '2.00',
            'default': '1.00',
            'decimals': 2,
            'step': 0.05,
            'parentToggle': 'TransferTextureEnableToggle',
            'requiredToggleValue': True,
            'help': 'adjusting can sometimes be good. Contrast adjust of original Face for Texture Transfer'
        }, 
        'TransferTextureClaheEnableToggle': {
            'level': 2,
            'label': 'CLAHE',
            'default': False,
            'parentToggle': 'TransferTextureEnableToggle',
            'requiredToggleValue': True,
            'help': 'for some extra wums. (Contrast Limited Adaptive Histogram Equalization). enhancing local contrast while preventing noise overamplification. handle with care, changes color dynamics.'
        },                   
        'TransferTextureClipLimitDecimalSlider': {
            'level': 3,
            'label': 'CLAHE Limit',
            'min_value': '0.0',
            'max_value': '5.0',
            'default': '1.0',
            'decimals': 1,
            'step': 0.1,
            'parentToggle': 'TransferTextureEnableToggle & TransferTextureClaheEnableToggle',
            'requiredToggleValue': True,
            'help': ''
        },             
        'TransferTextureAlphaClaheDecimalSlider': {
            'level': 3,
            'label': 'CLAHE Blend',
            'min_value': '0.00',
            'max_value': '1.00',
            'default': '0.40',
            'decimals': 2,
            'step': 0.05,
            'parentToggle': 'TransferTextureEnableToggle & TransferTextureClaheEnableToggle',
            'requiredToggleValue': True,
            'help': 'lower blend for additional texture strength without making colors too extreme'
        },        
        'ExcludeOriginalVGGMaskEnableToggle': {
            'level': 2,
            'label': 'VGG Mask Exclude',
            'default': False,
            'parentToggle': 'TransferTextureEnableToggle',
            'requiredToggleValue': True,            
            'help': 'Original Exclude Mask with VGG Model (without lower limit / upper limit / strenght  manipulation)'
        },       
        'TextureUpperLimitSlider': {
            'level': 3,
            'label': 'Upper Limit',
            'min_value': '0',
            'max_value': '100',
            'default': '100',
            'step': 1,
            'parentToggle': 'TransferTextureEnableToggle & ExcludeOriginalVGGMaskEnableToggle',
            'requiredToggleValue': True,
            'help': 'Defines the strenght of parts over "Face Features Upper Limit" in transfer mask. Check "View Face Mask" with "mask show -> texture" to see effect.'
        },    
        'TextureBlendAmountSlider': {
            'level': 3,
            'label': 'VGG Mask Blur Amount',
            'min_value': '0',
            'max_value': '20',
            'default': '2',
            'step': 1,
            'parentToggle': 'TransferTextureEnableToggle & ExcludeOriginalVGGMaskEnableToggle',
            'requiredToggleValue': True,
            'help': 'Blur the VGG Mask for smoother transitions.'
        },
        'ExcludeVGGMaskEnableToggle': {
            'level': 3,
            'label': 'VGG Mask Manipulation',
            'default': False,
            'parentToggle': 'TransferTextureEnableToggle & ExcludeOriginalVGGMaskEnableToggle',
            'requiredToggleValue': True,            
            'help': 'Exclude Mask with VGG Model and lower limit / upper limit / strenght'
        }, 
        #'ExcludeVGGMaskSmoothEnableToggle': {
        #    'level': 3,
        #    'label': 'VGG Mask Smooth Mode',
        #    'default': False,
        #    'parentToggle': 'TransferTextureEnableToggle & ExcludeOriginalVGGMaskEnableToggle',
        #    'requiredToggleValue': True,            
        #    'help': '.'
        #}, 
        'VGGMaskThresholdSlider': {
            'level': 3,
            'label': 'Threshold',
            'min_value': '0',
            'max_value': '100',
            'default': '35',
            'step': 1,
            'parentToggle': 'TransferTextureEnableToggle & ExcludeOriginalVGGMaskEnableToggle & ExcludeVGGMaskEnableToggle',
            'requiredToggleValue': True,
            'help': 'Defines Limit for vgg transfer mask. Check "View Face Mask" with "mask show -> texture" to see effect.'
        },   
        #'VGGMaskSoftnessSlider': {
        #    'level': 3,
        #    'label': 'Softness',
        #    'min_value': '0',
        #    'max_value': '100',
        #    'default': '50',
        #    'step': 1,
        #    'parentToggle': 'TransferTextureEnableToggle & ExcludeOriginalVGGMaskEnableToggle & ExcludeVGGMaskEnableToggle',
        #    'requiredToggleValue': True,
        #    'help': 'Defines upper Limit for transfer mask. Check "View Face Mask" with "mask show -> texture" to see effect.'
        #},                                     
        #'VGGMaskGammaSlider': {
        #    'level': 3,
        #    'label': 'Gamma',
        #    'min_value': '0',
        #    'max_value': '100',
        #    'default': '0',
        #    'step': 1,
        #    'parentToggle': 'TransferTextureEnableToggle & ExcludeOriginalVGGMaskEnableToggle & ExcludeVGGMaskEnableToggle',
        #    'requiredToggleValue': True,
        #    'help': 'Defines the strenght of parts under "Face Features Lower Limit" in transfer mask. Check "View Face Mask" with "mask show -> texture" to see effect.'
        #},
        'ExcludeMaskEnableToggle': {
            'level': 2,
            'label': 'Mask Features Exclude',
            'default': False,
            'parentToggle': 'TransferTextureEnableToggle',
            'requiredToggleValue': True,            
            'help': 'Exclude Faceparts from Texture Transfere and uses the original Swap there. Combineable with VGG Mask'
        },        
        'FaceParserTextureSlider': {
            'level': 3,
            'label': 'Face Blend',
            'min_value': '0',
            'max_value': '10',
            'default': '0',
            'step': 1,
            'parentToggle': 'TransferTextureEnableToggle & ExcludeMaskEnableToggle',
            'requiredToggleValue': True,
            'help': 'Blend amount of rest of Face (without features)'
        },           
        'FaceParserBlendTextureSlider': {
            'level': 3,
            'label': 'Excluded Features Blend',
            'min_value': '-100',
            'max_value': '0',
            'default': '0',
            'step': 1,
            'parentToggle': 'TransferTextureEnableToggle & ExcludeMaskEnableToggle',
            'requiredToggleValue': True,
            'help': 'Blend Amount of Excluded Feature Areas'
        },       
        'EyebrowParserTextureSlider': {
            'level': 3,
            'label': 'Eyebrows',
            'min_value': '-20',
            'max_value': '20',
            'default': '1',
            'step': 1,
            'parentToggle': 'TransferTextureEnableToggle & ExcludeMaskEnableToggle',
            'requiredToggleValue': True,
            'help': 'Exclude Faceparts (Eyes, Eyebrows, Nose, Mouth, Lips, Neck), 0=whole face is used, 1= Parts not included, 1+ = increase Parts size. Most of the time should be 1/1+. try 0 on low quality/artefacted targets'
        },         
        'EyeParserTextureSlider': {
            'level': 3,
            'label': 'Eyes',
            'min_value': '-20',
            'max_value': '20',
            'default': '1',
            'step': 1,
            'parentToggle': 'TransferTextureEnableToggle & ExcludeMaskEnableToggle',
            'requiredToggleValue': True,
            'help': 'Exclude Faceparts (Eyes, Eyebrows, Nose, Mouth, Lips, Neck), 0=whole face is used, 1= Parts not included, 1+ = increase Parts size. Most of the time should be 1/1+. try 0 on low quality/artefacted targets'
        },        
        'NoseParserTextureSlider': {
            'level': 3,
            'label': 'Nose',
            'min_value': '-20',
            'max_value': '20',
            'default': '1',
            'step': 1,
            'parentToggle': 'TransferTextureEnableToggle & ExcludeMaskEnableToggle',
            'requiredToggleValue': True,
            'help': 'Exclude Faceparts (Eyes, Eyebrows, Nose, Mouth, Lips, Neck), 0=whole face is used, 1= Parts not included, 1+ = increase Parts size. Most of the time should be 1/1+. try 0 on low quality/artefacted targets'
        },        
        'MouthParserTextureSlider': {
            'level': 3,
            'label': 'Mouth',
            'min_value': '-20',
            'max_value': '20',
            'default': '1',
            'step': 1,
            'parentToggle': 'TransferTextureEnableToggle & ExcludeMaskEnableToggle',
            'requiredToggleValue': True,
            'help': 'Exclude Faceparts (Eyes, Eyebrows, Nose, Mouth, Lips, Neck), 0=whole face is used, 1= Parts not included, 1+ = increase Parts size. Most of the time should be 1/1+. try 0 on low quality/artefacted targets'
        },        
        'NeckParserTextureSlider': {
            'level': 3,
            'label': 'Neck',
            'min_value': '-20',
            'max_value': '20',
            'default': '0',
            'step': 1,
            'parentToggle': 'TransferTextureEnableToggle & ExcludeMaskEnableToggle',
            'requiredToggleValue': True,
            'help': 'Exclude Faceparts (Eyes, Eyebrows, Nose, Mouth, Lips, Neck), 0=whole face is used, 1= Parts not included, 1+ = increase Parts size. Most of the time should be 1/1+. try 0 on low quality/artefacted targets'
        },   
        'FaceParserBlurTextureSlider': {
            'level': 2,
            'label': 'Texture Mask Blur',
            'min_value': '0',
            'max_value': '20',
            'default': '4',
            'step': 1,
            'parentToggle': 'TransferTextureEnableToggle',
            'requiredToggleValue': True,
            'help': 'Mask Blur on excluded Area Edges.'
        },      
        'BgExcludeEnableToggle': {
            'level': 2,
            'label': 'Background Exclude',
            'default': False,
            'parentToggle': 'TransferTextureEnableToggle',
            'requiredToggleValue': True,   
            'help': 'Background reduce for Texture Transfer Mask, usefull if xseg > 0'
        },    
        'BackgroundParserTextureSlider': {
            'level': 3,
            'label': 'Background',
            'min_value': '-20',
            'max_value': '0',
            'default': '0',
            'step': 1,
            'parentToggle': 'TransferTextureEnableToggle & BgExcludeEnableToggle',
            'requiredToggleValue': True,
            'help': 'Background reduce based on XSEG model on original face and faceparser background on swap face'
        },  
        'BGExcludeBlurAmountSlider': {
            'level': 3,
            'label': 'BG Blur',
            'min_value': '0',
            'max_value': '50',
            'default': '0',
            'step': 1,
            'parentToggle': 'TransferTextureEnableToggle & BgExcludeEnableToggle',
            'requiredToggleValue': True,
            'help': 'Background reduce based on faceparser model on original face'
        },
        'AutoColorEnableToggle': {
            'level': 1,
            'label': 'AutoColor Transfer',
            'default': False,
            'help': 'Enable AutoColor Transfer: 1. Hans Test without mask, 2. Hans Test with mask, 3. DFL Method without mask, 4. DFL Original Method.'
        },
        'AutoColorTransferTypeSelection':{
            'level': 2,
            'label': 'Transfer Type',
            'options': ['Test', 'Test_Mask', 'DFL_Test', 'DFL_Orig'],
            'default': 'Test',
            'parentToggle': 'AutoColorEnableToggle',
            'requiredToggleValue': True,
            'help': 'Select the AutoColor transfer method type. Hans Method could have some artefacts sometimes.'
        },
        'AutoColorBlendAmountSlider': {
            'level': 2,
            'label': 'Blend Amount',
            'min_value': '0',
            'max_value': '100',
            'default': '80',
            'step': 5,
            'parentToggle': 'AutoColorEnableToggle',
            'requiredToggleValue': True,
            'help': 'Adjust the blend value.'
        },
        'AutoColorEndEnableToggle': {
            'level': 2,
            'label': 'Ending AutoColor Pass',
            'default': False,
            'parentToggle': 'AutoColorEnableToggle',
            'requiredToggleValue': True,
            'help': 'Make a second AutoColor pass at the end.'
        },
        'ColorEnableToggle': {
            'level': 1,
            'label': 'Color Adjustments',
            'default': False,
            'help': 'Fine-tune the RGB color values of the swap.'
        },
        'ColorRedSlider': {
            'level': 2,
            'label': 'Red',
            'min_value': '-100',
            'max_value': '100',
            'default': '0',
            'step': 1,
            'parentToggle': 'ColorEnableToggle',
            'requiredToggleValue': True,
            'help': 'Red color adjustments'
        },
        'ColorGreenSlider': {
            'level': 2,
            'label': 'Green',
            'min_value': '-100',
            'max_value': '100',
            'default': '0',
            'step': 1,
            'parentToggle': 'ColorEnableToggle',
            'requiredToggleValue': True,
            'help': 'Green color adjustments'
        },
        'ColorBlueSlider': {
            'level': 2,
            'label': 'Blue',
            'min_value': '-100',
            'max_value': '100',
            'default': '0',
            'step': 1,
            'parentToggle': 'ColorEnableToggle',
            'requiredToggleValue': True,
            'help': 'Blue color adjustments'
        },
        'ColorBrightnessDecimalSlider': {
            'level': 2,
            'label': 'Brightness',
            'min_value': '0.00',
            'max_value': '2.00',
            'default': '1.00',
            'step': 0.01,
            'decimals': 2,
            'parentToggle': 'ColorEnableToggle',
            'requiredToggleValue': True,
            'help': 'Changes the Brightness.'
        },
        'ColorContrastDecimalSlider': {
            'level': 2,
            'label': 'Contrast',
            'min_value': '0.00',
            'max_value': '2.00',
            'default': '1.00',
            'step': 0.01,
            'decimals': 2,
            'parentToggle': 'ColorEnableToggle',
            'requiredToggleValue': True,
            'help': 'Changes the Contrast.'
        },
        'ColorSaturationDecimalSlider': {
            'level': 2,
            'label': 'Saturation',
            'min_value': '0.00',
            'max_value': '2.00',
            'default': '1.00',
            'step': 0.01,
            'decimals': 2,
            'parentToggle': 'ColorEnableToggle',
            'requiredToggleValue': True,
            'help': 'Changes the Saturation.'
        },
        'ColorSharpnessDecimalSlider': {
            'level': 2,
            'label': 'Sharpness',
            'min_value': '0.0',
            'max_value': '2.0',
            'default': '1.0',
            'step': 0.1,
            'decimals': 1,
            'parentToggle': 'ColorEnableToggle',
            'requiredToggleValue': True,
            'help': 'Changes the Sharpness.'
        },
        'ColorHueDecimalSlider': {
            'level': 2,
            'label': 'Hue',
            'min_value': '-0.50',
            'max_value': '0.50',
            'default': '0.00',
            'step': 0.01,
            'decimals': 2,
            'parentToggle': 'ColorEnableToggle',
            'requiredToggleValue': True,
            'help': 'Changes the Hue.'
        },
        'ColorGammaDecimalSlider': {
            'level': 2,
            'label': 'Gamma',
            'min_value': '0.00',
            'max_value': '2.00',
            'default': '1.00',
            'step': 0.01,
            'decimals': 2,
            'parentToggle': 'ColorEnableToggle',
            'requiredToggleValue': True,
            'help': 'Changes the Gamma.'
        },
        'ColorNoiseDecimalSlider': {
            'level': 2,
            'label': 'Noise',
            'min_value': '0.00',
            'max_value': '20.00',
            'default': '0.00',
            'step': 1.00,
            'decimals': 2,
            'parentToggle': 'ColorEnableToggle',
            'help': 'Add noise to swapped face.'
        },
        'JPEGCompressionEnableToggle': {
            'level': 1,
            'label': 'JPEG Compression',
            'default': False,
            'help': 'Apply JPEG Compression to the swapped face to make output more realistic',
        },
        'JPEGCompressionAmountSlider': {
            'level': 2,
            'label': 'Compression',
            'min_value': '1',
            'max_value': '100',
            'default': '60',
            'step': 1,
            'parentToggle': 'JPEGCompressionEnableToggle',
            'requiredToggleValue': True,
            'help': 'Adjust the JPEG Compression amount'
        },
        'JPEGCompressionBlendSlider': {
            'level': 2,
            'label': 'Blend',
            'min_value': '1',
            'max_value': '100',
            'default': '100',
            'step': 1,
            'parentToggle': 'JPEGCompressionEnableToggle',
            'requiredToggleValue': True,
            'help': 'Blend Amount for Jpeg Compression'
        },       
        'BlockShiftEnableToggle': {
            'level': 1,
            'label': 'MPEG Compression',
            'default': False,
            'help': 'Apply MPEG Compression like Block Shift Effect',
        },
        'BlockShiftAmountSlider': {
            'level': 2,
            'label': 'Block Size',
            'min_value': '1',
            'max_value': '8',
            'default': '4',
            'step': 1,
            'parentToggle': 'BlockShiftEnableToggle',
            'requiredToggleValue': True,
            'help': 'Size of (Pixel) Blocks'
        },        
        'BlockShiftMaxAmountSlider': {
            'level': 2,
            'label': 'Shift Maximum',
            'min_value': '1',
            'max_value': '8',
            'default': '2',
            'step': 1,
            'parentToggle': 'BlockShiftEnableToggle',
            'requiredToggleValue': True,
            'help': 'Maximum Shift of each Block'
        },        
        'BlockShiftBlendAmountSlider': {
            'level': 2,
            'label': 'Blend Amount',
            'min_value': '1',
            'max_value': '100',
            'default': '50',
            'step': 1,
            'parentToggle': 'BlockShiftEnableToggle',
            'requiredToggleValue': True,
            'help': 'Blend Amount'
        }
    },
    'Face Landmarks Correction': {
        'FaceAdjEnableToggle': {
            'level': 1,
            'label': 'Face Adjustments',
            'default': False,
            'help': 'This is an experimental feature to perform direct adjustments to the face landmarks found by the detector. There is also an option to adjust the scale of the swapped face.'
        },
        'KpsXSlider': {
            'level': 2,
            'label': 'Keypoints X-Axis',
            'min_value': '-100',
            'max_value': '100',
            'default': '0',
            'step': 1,
            'parentToggle': 'FaceAdjEnableToggle',
            'requiredToggleValue': True,
            'help': 'Shifts the detection points left and right.'
        },
        'KpsYSlider': {
            'level': 2,
            'label': 'Keypoints Y-Axis',
            'min_value': '-100',
            'max_value': '100',
            'default': '0',
            'step': 1,
            'parentToggle': 'FaceAdjEnableToggle',
            'requiredToggleValue': True,
            'help': 'Shifts the detection points up and down.'
        },
        'KpsScaleSlider': {
            'level': 2,
            'label': 'Keypoints Scale',
            'min_value': '-100',
            'max_value': '100',
            'default': '0',
            'step': 1,
            'parentToggle': 'FaceAdjEnableToggle',
            'requiredToggleValue': True,
            'help': 'Grows and shrinks the detection point distances.'
        },
        'FaceScaleAmountSlider': {
            'level': 2,
            'label': 'Face Scale Amount',
            'min_value': '-20',
            'max_value': '20',
            'default': '0',
            'step': 1,
            'parentToggle': 'FaceAdjEnableToggle',
            'requiredToggleValue': True,
            'help': 'Grows and shrinks the entire face.'
        },
        'LandmarksPositionAdjEnableToggle': {
            'level': 1,
            'label': '5 - Keypoints Adjustments',
            'default': False,
            'help': 'This is an experimental feature to perform direct adjustments to the position of face landmarks found by the detector.'
        },
        'EyeLeftXAmountSlider': {
            'level': 2,
            'label': 'Left Eye:   X',
            'min_value': '-100',
            'max_value': '100',
            'default': '0',
            'step': 1,
            'parentToggle': 'LandmarksPositionAdjEnableToggle',
            'requiredToggleValue': True,
            'help': 'Shifts the eye left detection point left and right.'
        },
        'EyeLeftYAmountSlider': {
            'level': 2,
            'label': 'Left Eye:   Y',
            'min_value': '-100',
            'max_value': '100',
            'default': '0',
            'step': 1,
            'parentToggle': 'LandmarksPositionAdjEnableToggle',
            'requiredToggleValue': True,
            'help': 'Shifts the eye left detection point up and down.'
        },
        'EyeRightXAmountSlider': {
            'level': 2,
            'label': 'Right Eye:   X',
            'min_value': '-100',
            'max_value': '100',
            'default': '0',
            'step': 1,
            'parentToggle': 'LandmarksPositionAdjEnableToggle',
            'requiredToggleValue': True,
            'help': 'Shifts the eye right detection point left and right.'
        },
        'EyeRightYAmountSlider': {
            'level': 2,
            'label': 'Right Eye:   Y',
            'min_value': '-100',
            'max_value': '100',
            'default': '0',
            'step': 1,
            'parentToggle': 'LandmarksPositionAdjEnableToggle',
            'requiredToggleValue': True,
            'help': 'Shifts the eye right detection point up and down.'
        },
        'NoseXAmountSlider': {
            'level': 2,
            'label': 'Nose:   X',
            'min_value': '-100',
            'max_value': '100',
            'default': '0',
            'step': 1,
            'parentToggle': 'LandmarksPositionAdjEnableToggle',
            'requiredToggleValue': True,
            'help': 'Shifts the nose detection point left and right.'
        },
        'NoseYAmountSlider': {
            'level': 2,
            'label': 'Nose:   Y',
            'min_value': '-100',
            'max_value': '100',
            'default': '0',
            'step': 1,
            'parentToggle': 'LandmarksPositionAdjEnableToggle',
            'requiredToggleValue': True,
            'help': 'Shifts the nose detection point up and down.'
        },
        'MouthLeftXAmountSlider': {
            'level': 2,
            'label': 'Left Mouth:   X',
            'min_value': '-100',
            'max_value': '100',
            'default': '0',
            'step': 1,
            'parentToggle': 'LandmarksPositionAdjEnableToggle',
            'requiredToggleValue': True,
            'help': 'Shifts the mouth left detection point left and right.'
        },
        'MouthLeftYAmountSlider': {
            'level': 2,
            'label': 'Left Mouth:   Y',
            'min_value': '-100',
            'max_value': '100',
            'default': '0',
            'step': 1,
            'parentToggle': 'LandmarksPositionAdjEnableToggle',
            'requiredToggleValue': True,
            'help': 'Shifts the mouth left detection point up and down.'
        },
        'MouthRightXAmountSlider': {
            'level': 2,
            'label': 'Right Mouth:   X',
            'min_value': '-100',
            'max_value': '100',
            'default': '0',
            'step': 1,
            'parentToggle': 'LandmarksPositionAdjEnableToggle',
            'requiredToggleValue': True,
            'help': 'Shifts the mouth Right detection point left and right.'
        },
        'MouthRightYAmountSlider': {
            'level': 2,
            'label': 'Right Mouth:   Y',
            'min_value': '-100',
            'max_value': '100',
            'default': '0',
            'step': 1,
            'parentToggle': 'LandmarksPositionAdjEnableToggle',
            'requiredToggleValue': True,
            'help': 'Shifts the mouth Right detection point up and down.'
        }
    },
    'Blend Adjustments':{
        'FinalBlendAdjEnableToggle': {
            'level': 1,
            'label': 'Final Blend',
            'default': False,
            'help': 'Blend at the end of pipeline.'
        },
        'FinalBlendAmountSlider': {
            'level': 2,
            'label': 'Final Blend Amount',
            'min_value': '1',
            'max_value': '50',
            'default': '1',
            'step': 1,
            'parentToggle': 'FinalBlendAdjEnableToggle',
            'requiredToggleValue': True,
            'help': 'Adjust the final blend value.'
        },
        'OverallMaskBlendAmountSlider': {
            'level': 1,
            'label': 'Overall Mask Blend Amount',
            'min_value': '0',
            'max_value': '100',
            'default': '0',
            'step': 1,
            'help': 'Combined masks blending distance. It is not applied to the border masks.'
        }
    }
}