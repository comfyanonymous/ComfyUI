"""
@author: Dr.Lt.Data
@title: Impact Pack
@nickname: Impact Pack
@description: This extension offers various detector nodes and detailer nodes that allow you to configure a workflow that automatically enhances facial details. And provide iterative upscaler.
"""

import folder_paths
import os
import sys
import logging

comfy_path = os.path.dirname(folder_paths.__file__)
impact_path = os.path.join(os.path.dirname(__file__))
modules_path = os.path.join(os.path.dirname(__file__), "modules")

sys.path.append(modules_path)

import impact.config
logging.info(f"### Loading: ComfyUI-Impact-Pack ({impact.config.version})")

# Core
# recheck dependencies for colab
try:
    import folder_paths
    import torch                  # noqa: F401
    import cv2                    # noqa: F401
    from cv2 import setNumThreads # noqa: F401
    import numpy as np            # noqa: F401
    import comfy.samplers
    import comfy.sd               # noqa: F401
    from PIL import Image, ImageFilter             # noqa: F401
    from skimage.measure import label, regionprops # noqa: F401
    from collections import namedtuple             # noqa: F401
    import piexif                                  # noqa: F401
    import nodes
except Exception as e:
    import logging
    logging.error("[Impact Pack] Failed to import due to several dependencies are missing!!!!")
    raise e


import impact.impact_server  # to load server api

from .modules.impact.impact_pack import *       # noqa: F403
from .modules.impact.detectors import *         # noqa: F403
from .modules.impact.pipe import *              # noqa: F403
from .modules.impact.logics import *            # noqa: F403
from .modules.impact.util_nodes import *        # noqa: F403
from .modules.impact.segs_nodes import *        # noqa: F403
from .modules.impact.special_samplers import *  # noqa: F403
from .modules.impact.hf_nodes import *          # noqa: F403
from .modules.impact.bridge_nodes import *      # noqa: F403
from .modules.impact.hook_nodes import *        # noqa: F403
from .modules.impact.animatediff_nodes import * # noqa: F403
from .modules.impact.segs_upscaler import *     # noqa: F403

import threading


threading.Thread(target=impact.wildcards.wildcard_load).start()


NODE_CLASS_MAPPINGS = {
    "SAMLoader": SAMLoader, # noqa: F405
    "CLIPSegDetectorProvider": CLIPSegDetectorProvider, # noqa: F405
    "ONNXDetectorProvider": ONNXDetectorProvider, # noqa: F405

    "BitwiseAndMaskForEach": BitwiseAndMaskForEach, # noqa: F405
    "SubtractMaskForEach": SubtractMaskForEach, # noqa: F405

    "DetailerForEach": DetailerForEach, # noqa: F405
    "DetailerForEachAutoRetry": DetailerForEachAutoRetry, # noqa: F405
    "DetailerForEachDebug": DetailerForEachTest, # noqa: F405
    "DetailerForEachPipe": DetailerForEachPipe, # noqa: F405
    "DetailerForEachDebugPipe": DetailerForEachTestPipe, # noqa: F405
    "DetailerForEachPipeForAnimateDiff": DetailerForEachPipeForAnimateDiff, # noqa: F405

    "SAMDetectorCombined": SAMDetectorCombined, # noqa: F405
    "SAMDetectorSegmented": SAMDetectorSegmented, # noqa: F405

    "FaceDetailer": FaceDetailer, # noqa: F405
    "FaceDetailerPipe": FaceDetailerPipe, # noqa: F405
    "MaskDetailerPipe": MaskDetailerPipe, # noqa: F405

    "ToDetailerPipe": ToDetailerPipe, # noqa: F405
    "ToDetailerPipeSDXL": ToDetailerPipeSDXL, # noqa: F405
    "FromDetailerPipe": FromDetailerPipe, # noqa: F405
    "FromDetailerPipe_v2": FromDetailerPipe_v2, # noqa: F405
    "FromDetailerPipeSDXL": FromDetailerPipe_SDXL, # noqa: F405
    "AnyPipeToBasic": AnyPipeToBasic, # noqa: F405
    "ToBasicPipe": ToBasicPipe, # noqa: F405
    "FromBasicPipe": FromBasicPipe, # noqa: F405
    "FromBasicPipe_v2": FromBasicPipe_v2, # noqa: F405
    "BasicPipeToDetailerPipe": BasicPipeToDetailerPipe, # noqa: F405
    "BasicPipeToDetailerPipeSDXL": BasicPipeToDetailerPipeSDXL, # noqa: F405
    "DetailerPipeToBasicPipe": DetailerPipeToBasicPipe, # noqa: F405
    "EditBasicPipe": EditBasicPipe, # noqa: F405
    "EditDetailerPipe": EditDetailerPipe, # noqa: F405
    "EditDetailerPipeSDXL": EditDetailerPipeSDXL, # noqa: F405

    "LatentPixelScale": LatentPixelScale, # noqa: F405
    "PixelKSampleUpscalerProvider": PixelKSampleUpscalerProvider, # noqa: F405
    "PixelKSampleUpscalerProviderPipe": PixelKSampleUpscalerProviderPipe, # noqa: F405
    "IterativeLatentUpscale": IterativeLatentUpscale, # noqa: F405
    "IterativeImageUpscale": IterativeImageUpscale, # noqa: F405
    "PixelTiledKSampleUpscalerProvider": PixelTiledKSampleUpscalerProvider, # noqa: F405
    "PixelTiledKSampleUpscalerProviderPipe": PixelTiledKSampleUpscalerProviderPipe, # noqa: F405
    "TwoSamplersForMaskUpscalerProvider": TwoSamplersForMaskUpscalerProvider, # noqa: F405
    "TwoSamplersForMaskUpscalerProviderPipe": TwoSamplersForMaskUpscalerProviderPipe, # noqa: F405

    "PixelKSampleHookCombine": PixelKSampleHookCombine, # noqa: F405
    "DenoiseScheduleHookProvider": DenoiseScheduleHookProvider, # noqa: F405
    "StepsScheduleHookProvider": StepsScheduleHookProvider, # noqa: F405
    "CfgScheduleHookProvider": CfgScheduleHookProvider, # noqa: F405
    "NoiseInjectionHookProvider": NoiseInjectionHookProvider, # noqa: F405
    "UnsamplerHookProvider": UnsamplerHookProvider, # noqa: F405
    "CoreMLDetailerHookProvider": CoreMLDetailerHookProvider, # noqa: F405
    "PreviewDetailerHookProvider": PreviewDetailerHookProvider, # noqa: F405
    "BlackPatchRetryHookProvider": BlackPatchRetryHookProvider,
    "CustomSamplerDetailerHookProvider": CustomSamplerDetailerHookProvider, # noqa: F405
    "LamaRemoverDetailerHookProvider": LamaRemoverDetailerHookProvider, # noqa: F405

    "DetailerHookCombine": DetailerHookCombine, # noqa: F405
    "NoiseInjectionDetailerHookProvider": NoiseInjectionDetailerHookProvider, # noqa: F405
    "UnsamplerDetailerHookProvider": UnsamplerDetailerHookProvider, # noqa: F405
    "DenoiseSchedulerDetailerHookProvider": DenoiseSchedulerDetailerHookProvider, # noqa: F405
    "SEGSOrderedFilterDetailerHookProvider": SEGSOrderedFilterDetailerHookProvider, # noqa: F405
    "SEGSRangeFilterDetailerHookProvider": SEGSRangeFilterDetailerHookProvider, # noqa: F405
    "SEGSLabelFilterDetailerHookProvider": SEGSLabelFilterDetailerHookProvider, # noqa: F405
    "VariationNoiseDetailerHookProvider": VariationNoiseDetailerHookProvider, # noqa: F405
    # "CustomNoiseDetailerHookProvider": CustomNoiseDetailerHookProvider,

    "BitwiseAndMask": BitwiseAndMask, # noqa: F405
    "SubtractMask": SubtractMask, # noqa: F405
    "AddMask": AddMask, # noqa: F405
    "MaskRectArea": MaskRectArea, # noqa: F405
    "MaskRectAreaAdvanced": MaskRectAreaAdvanced, # noqa: F405
    "ImpactSegsAndMask": SegsBitwiseAndMask, # noqa: F405
    "ImpactSegsAndMaskForEach": SegsBitwiseAndMaskForEach, # noqa: F405
    "EmptySegs": EmptySEGS, # noqa: F405
    "ImpactFlattenMask": FlattenMask, # noqa: F405

    "MediaPipeFaceMeshToSEGS": MediaPipeFaceMeshToSEGS, # noqa: F405
    "MaskToSEGS": MaskToSEGS, # noqa: F405
    "MaskToSEGS_for_AnimateDiff": MaskToSEGS_for_AnimateDiff, # noqa: F405
    "ToBinaryMask": ToBinaryMask, # noqa: F405
    "MasksToMaskList": MasksToMaskList, # noqa: F405
    "MaskListToMaskBatch": MaskListToMaskBatch, # noqa: F405
    "ImageListToImageBatch": ImageListToImageBatch, # noqa: F405
    "SetDefaultImageForSEGS": DefaultImageForSEGS, # noqa: F405
    "RemoveImageFromSEGS": RemoveImageFromSEGS, # noqa: F405

    "BboxDetectorSEGS": BboxDetectorForEach, # noqa: F405
    "SegmDetectorSEGS": SegmDetectorForEach, # noqa: F405
    "ONNXDetectorSEGS": BboxDetectorForEach, # noqa: F405
    "ImpactSimpleDetectorSEGS_for_AD": SimpleDetectorForAnimateDiff, # noqa: F405
    "ImpactSAM2VideoDetectorSEGS": SAM2VideoDetectorSEGS, # noqa: F405
    "ImpactSimpleDetectorSEGS": SimpleDetectorForEach, # noqa: F405
    "ImpactSimpleDetectorSEGSPipe": SimpleDetectorForEachPipe, # noqa: F405
    "ImpactControlNetApplySEGS": ControlNetApplySEGS, # noqa: F405
    "ImpactControlNetApplyAdvancedSEGS": ControlNetApplyAdvancedSEGS, # noqa: F405
    "ImpactControlNetClearSEGS": ControlNetClearSEGS, # noqa: F405
    "ImpactIPAdapterApplySEGS": IPAdapterApplySEGS, # noqa: F405

    "ImpactDecomposeSEGS": DecomposeSEGS, # noqa: F405
    "ImpactAssembleSEGS": AssembleSEGS, # noqa: F405
    "ImpactFrom_SEG_ELT": From_SEG_ELT, # noqa: F405
    "ImpactEdit_SEG_ELT": Edit_SEG_ELT, # noqa: F405
    "ImpactDilate_Mask_SEG_ELT": Dilate_SEG_ELT, # noqa: F405
    "ImpactDilateMask": DilateMask, # noqa: F405
    "ImpactGaussianBlurMask": GaussianBlurMask, # noqa: F405
    "ImpactDilateMaskInSEGS": DilateMaskInSEGS, # noqa: F405
    "ImpactGaussianBlurMaskInSEGS": GaussianBlurMaskInSEGS, # noqa: F405
    "ImpactScaleBy_BBOX_SEG_ELT": SEG_ELT_BBOX_ScaleBy, # noqa: F405
    "ImpactFrom_SEG_ELT_bbox": From_SEG_ELT_bbox, # noqa: F405
    "ImpactFrom_SEG_ELT_crop_region": From_SEG_ELT_crop_region, # noqa: F405
    "ImpactCount_Elts_in_SEGS": Count_Elts_in_SEGS, # noqa: F405

    "BboxDetectorCombined_v2": BboxDetectorCombined, # noqa: F405
    "SegmDetectorCombined_v2": SegmDetectorCombined, # noqa: F405
    "SegsToCombinedMask": SegsToCombinedMask, # noqa: F405

    "KSamplerProvider": KSamplerProvider, # noqa: F405
    "TwoSamplersForMask": TwoSamplersForMask, # noqa: F405
    "TiledKSamplerProvider": TiledKSamplerProvider, # noqa: F405

    "KSamplerAdvancedProvider": KSamplerAdvancedProvider, # noqa: F405
    "TwoAdvancedSamplersForMask": TwoAdvancedSamplersForMask, # noqa: F405

    "ImpactNegativeConditioningPlaceholder": NegativeConditioningPlaceholder, # noqa: F405

    "PreviewBridge": PreviewBridge, # noqa: F405
    "PreviewBridgeLatent": PreviewBridgeLatent, # noqa: F405
    "ImageSender": ImageSender, # noqa: F405
    "ImageReceiver": ImageReceiver, # noqa: F405
    "LatentSender": LatentSender, # noqa: F405
    "LatentReceiver": LatentReceiver, # noqa: F405
    "ImageMaskSwitch": ImageMaskSwitch, # noqa: F405
    "LatentSwitch": GeneralSwitch, # noqa: F405
    "SEGSSwitch": GeneralSwitch, # noqa: F405
    "ImpactSwitch": GeneralSwitch, # noqa: F405
    "ImpactInversedSwitch": GeneralInversedSwitch, # noqa: F405

    "ImpactWildcardProcessor": ImpactWildcardProcessor, # noqa: F405
    "ImpactWildcardEncode": ImpactWildcardEncode, # noqa: F405

    "SEGSUpscaler": SEGSUpscaler, # noqa: F405
    "SEGSUpscalerPipe": SEGSUpscalerPipe, # noqa: F405
    "SEGSDetailer": SEGSDetailer, # noqa: F405
    "SEGSPaste": SEGSPaste, # noqa: F405
    "SEGSPreview": SEGSPreview, # noqa: F405
    "SEGSPreviewCNet": SEGSPreviewCNet, # noqa: F405
    "SEGSToImageList": SEGSToImageList, # noqa: F405
    "ImpactSEGSToMaskList": SEGSToMaskList, # noqa: F405
    "ImpactSEGSToMaskBatch": SEGSToMaskBatch, # noqa: F405
    "ImpactSEGSConcat": SEGSConcat, # noqa: F405
    "ImpactSEGSPicker": SEGSPicker, # noqa: F405
    "ImpactMakeTileSEGS": MakeTileSEGS, # noqa: F405
    "ImpactSEGSMerge": SEGSMerge, # noqa: F405

    "SEGSDetailerForAnimateDiff": SEGSDetailerForAnimateDiff, # noqa: F405

    "ImpactKSamplerBasicPipe": KSamplerBasicPipe, # noqa: F405
    "ImpactKSamplerAdvancedBasicPipe": KSamplerAdvancedBasicPipe, # noqa: F405

    "ReencodeLatent": ReencodeLatent, # noqa: F405
    "ReencodeLatentPipe": ReencodeLatentPipe, # noqa: F405

    "ImpactImageBatchToImageList": ImageBatchToImageList, # noqa: F405
    "ImpactMakeImageList": MakeImageList, # noqa: F405
    "ImpactMakeImageBatch": MakeImageBatch, # noqa: F405
    "ImpactMakeAnyList": MakeAnyList, # noqa: F405
    "ImpactMakeMaskList": MakeMaskList, # noqa: F405
    "ImpactMakeMaskBatch": MakeMaskBatch, # noqa: F405
    "ImpactSelectNthItemOfAnyList": NthItemOfAnyList, # noqa: F405

    "RegionalSampler": RegionalSampler, # noqa: F405
    "RegionalSamplerAdvanced": RegionalSamplerAdvanced, # noqa: F405
    "CombineRegionalPrompts": CombineRegionalPrompts, # noqa: F405
    "RegionalPrompt": RegionalPrompt, # noqa: F405

    "ImpactCombineConditionings": CombineConditionings, # noqa: F405
    "ImpactConcatConditionings": ConcatConditionings, # noqa: F405

    "ImpactSEGSLabelAssign": SEGSLabelAssign, # noqa: F405
    "ImpactSEGSLabelFilter": SEGSLabelFilter, # noqa: F405
    "ImpactSEGSRangeFilter": SEGSRangeFilter, # noqa: F405
    "ImpactSEGSOrderedFilter": SEGSOrderedFilter, # noqa: F405
    "ImpactSEGSIntersectionFilter": SEGSIntersectionFilter, # noqa: F405
    "ImpactSEGSNMSFilter": SEGSNMSFilter, # noqa: F405

    "ImpactCompare": ImpactCompare, # noqa: F405
    "ImpactConditionalBranch": ImpactConditionalBranch, # noqa: F405
    "ImpactConditionalBranchSelMode": ImpactConditionalBranchSelMode, # noqa: F405
    "ImpactIfNone": ImpactIfNone, # noqa: F405
    "ImpactConvertDataType": ImpactConvertDataType, # noqa: F405
    "ImpactLogicalOperators": ImpactLogicalOperators, # noqa: F405
    "ImpactInt": ImpactInt, # noqa: F405
    "ImpactFloat": ImpactFloat, # noqa: F405
    "ImpactBoolean": ImpactBoolean, # noqa: F405
    "ImpactValueSender": ImpactValueSender, # noqa: F405
    "ImpactValueReceiver": ImpactValueReceiver, # noqa: F405
    "ImpactImageInfo": ImpactImageInfo, # noqa: F405
    "ImpactLatentInfo": ImpactLatentInfo, # noqa: F405
    "ImpactMinMax": ImpactMinMax, # noqa: F405
    "ImpactNeg": ImpactNeg, # noqa: F405
    "ImpactConditionalStopIteration": ImpactConditionalStopIteration, # noqa: F405
    "ImpactStringSelector": StringSelector, # noqa: F405
    "StringListToString": StringListToString, # noqa: F405
    "WildcardPromptFromString": WildcardPromptFromString, # noqa: F405
    "ImpactExecutionOrderController": ImpactExecutionOrderController, # noqa: F405
    "ImpactListBridge": ImpactListBridge, # noqa: F405

    "RemoveNoiseMask": RemoveNoiseMask, # noqa: F405

    "ImpactLogger": ImpactLogger, # noqa: F405
    "ImpactDummyInput": ImpactDummyInput, # noqa: F405

    "ImpactQueueTrigger": ImpactQueueTrigger, # noqa: F405
    "ImpactQueueTriggerCountdown": ImpactQueueTriggerCountdown, # noqa: F405
    "ImpactSetWidgetValue": ImpactSetWidgetValue, # noqa: F405
    "ImpactNodeSetMuteState": ImpactNodeSetMuteState, # noqa: F405
    "ImpactControlBridge": ImpactControlBridge, # noqa: F405
    "ImpactIsNotEmptySEGS": ImpactNotEmptySEGS, # noqa: F405
    "ImpactSleep": ImpactSleep, # noqa: F405
    "ImpactRemoteBoolean": ImpactRemoteBoolean, # noqa: F405
    "ImpactRemoteInt": ImpactRemoteInt, # noqa: F405

    "ImpactHFTransformersClassifierProvider": HF_TransformersClassifierProvider, # noqa: F405
    "ImpactSEGSClassify": SEGS_Classify, # noqa: F405

    "ImpactSchedulerAdapter": ImpactSchedulerAdapter, # noqa: F405
    "GITSSchedulerFuncProvider": GITSSchedulerFuncProvider # noqa: F405
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "SAMLoader": "SAMLoader (Impact)",

    "BboxDetectorSEGS": "BBOX Detector (SEGS)",
    "SegmDetectorSEGS": "SEGM Detector (SEGS)",
    "ONNXDetectorSEGS": "ONNX Detector (SEGS/legacy) - use BBOXDetector",
    "ImpactSimpleDetectorSEGS_for_AD": "Simple Detector for Video (SEGS)",
    "ImpactSAM2VideoDetectorSEGS": "SAM2 Video Detector (SEGS)",
    "ImpactSimpleDetectorSEGS": "Simple Detector (SEGS)",
    "ImpactSimpleDetectorSEGSPipe": "Simple Detector (SEGS/pipe)",
    "ImpactControlNetApplySEGS": "ControlNetApply (SEGS) - DEPRECATED",
    "ImpactControlNetApplyAdvancedSEGS": "ControlNetApply (SEGS)",
    "ImpactIPAdapterApplySEGS": "IPAdapterApply (SEGS)",

    "BboxDetectorCombined_v2": "BBOX Detector (combined)",
    "SegmDetectorCombined_v2": "SEGM Detector (combined)",
    "SegsToCombinedMask": "SEGS to MASK (combined)",
    "MediaPipeFaceMeshToSEGS": "MediaPipe FaceMesh to SEGS",
    "MaskToSEGS": "MASK to SEGS",
    "MaskToSEGS_for_AnimateDiff": "MASK to SEGS for Video",
    "BitwiseAndMaskForEach": "Pixelwise(SEGS & SEGS)",
    "SubtractMaskForEach": "Pixelwise(SEGS - SEGS)",
    "ImpactSegsAndMask": "Pixelwise(SEGS & MASK)",
    "ImpactSegsAndMaskForEach": "Pixelwise(SEGS & MASKS ForEach)",
    "BitwiseAndMask": "Pixelwise(MASK & MASK)",
    "SubtractMask": "Pixelwise(MASK - MASK)",
    "AddMask": "Pixelwise(MASK + MASK)",
    "MaskRectArea": "Mask Rect Area",
    "MaskRectAreaAdvanced": "Mask Rect Area (Advanced)",
    "ImpactFlattenMask": "Flatten Mask Batch",
    "DetailerForEach": "Detailer (SEGS)",
    "DetailerForEachAutoRetry": "Detailer (SEGS) with auto retry",
    "DetailerForEachPipe": "Detailer (SEGS/pipe)",
    "DetailerForEachDebug": "DetailerDebug (SEGS)",
    "DetailerForEachDebugPipe": "DetailerDebug (SEGS/pipe)",
    "SEGSDetailerForAnimateDiff": "SEGSDetailer For Video (SEGS/pipe)",
    "DetailerForEachPipeForAnimateDiff": "Detailer For Video (SEGS/pipe)",
    "SEGSUpscaler": "Upscaler (SEGS)",
    "SEGSUpscalerPipe": "Upscaler (SEGS/pipe)",

    "SAMDetectorCombined": "SAMDetector (combined)",
    "SAMDetectorSegmented": "SAMDetector (segmented)",
    "FaceDetailerPipe": "FaceDetailer (pipe)",
    "MaskDetailerPipe": "MaskDetailer (pipe)",

    "FromDetailerPipeSDXL": "FromDetailer (SDXL/pipe)",
    "BasicPipeToDetailerPipeSDXL": "BasicPipe -> DetailerPipe (SDXL)",
    "EditDetailerPipeSDXL": "Edit DetailerPipe (SDXL)",

    "BasicPipeToDetailerPipe": "BasicPipe -> DetailerPipe",
    "DetailerPipeToBasicPipe": "DetailerPipe -> BasicPipe",
    "EditBasicPipe": "Edit BasicPipe",
    "EditDetailerPipe": "Edit DetailerPipe",
    "AnyPipeToBasic": "Any PIPE -> BasicPipe",

    "LatentPixelScale": "Latent Scale (on Pixel Space)",
    "IterativeLatentUpscale": "Iterative Upscale (Latent/on Pixel Space)",
    "IterativeImageUpscale": "Iterative Upscale (Image)",

    "TwoSamplersForMaskUpscalerProvider": "TwoSamplersForMask Upscaler Provider",
    "TwoSamplersForMaskUpscalerProviderPipe": "TwoSamplersForMask Upscaler Provider (pipe)",

    "ReencodeLatent": "Reencode Latent",
    "ReencodeLatentPipe": "Reencode Latent (pipe)",

    "ImpactKSamplerBasicPipe": "KSampler (pipe)",
    "ImpactKSamplerAdvancedBasicPipe": "KSampler (Advanced/pipe)",
    "ImpactSEGSLabelAssign": "SEGS Assign (label)",
    "ImpactSEGSLabelFilter": "SEGS Filter (label)",
    "ImpactSEGSRangeFilter": "SEGS Filter (range)",
    "ImpactSEGSOrderedFilter": "SEGS Filter (ordered)",
    "ImpactSEGSIntersectionFilter": "SEGS Filter (intersection)",
    "ImpactSEGSNMSFilter": "SEGS Filter (non max suppression)",
    "ImpactSEGSConcat": "SEGS Concat",
    "ImpactSEGSToMaskList": "SEGS to Mask List",
    "ImpactSEGSToMaskBatch": "SEGS to Mask Batch",
    "ImpactSEGSPicker": "Picker (SEGS)",
    "ImpactMakeTileSEGS": "Make Tile SEGS",
    "ImpactSEGSMerge": "SEGS Merge",

    "ImpactDecomposeSEGS": "Decompose (SEGS)",
    "ImpactAssembleSEGS": "Assemble (SEGS)",
    "ImpactFrom_SEG_ELT": "From SEG_ELT",
    "ImpactEdit_SEG_ELT": "Edit SEG_ELT",
    "ImpactFrom_SEG_ELT_bbox": "From SEG_ELT bbox",
    "ImpactFrom_SEG_ELT_crop_region": "From SEG_ELT crop_region",
    "ImpactDilate_Mask_SEG_ELT": "Dilate Mask (SEG_ELT)",
    "ImpactScaleBy_BBOX_SEG_ELT": "ScaleBy BBOX (SEG_ELT)",
    "ImpactCount_Elts_in_SEGS": "Count Elts in SEGS",
    "ImpactDilateMask": "Dilate Mask",
    "ImpactGaussianBlurMask": "Gaussian Blur Mask",
    "ImpactDilateMaskInSEGS": "Dilate Mask (SEGS)",
    "ImpactGaussianBlurMaskInSEGS": "Gaussian Blur Mask (SEGS)",

    "PreviewBridge": "Preview Bridge (Image)",
    "PreviewBridgeLatent": "Preview Bridge (Latent)",
    "ImageSender": "Image Sender",
    "ImageReceiver": "Image Receiver",
    "ImageMaskSwitch": "Switch (images, mask)",
    "ImpactSwitch": "Switch (Any)",
    "ImpactInversedSwitch": "Inversed Switch (Any)",
    "ImpactExecutionOrderController": "Execution Order Controller",
    "ImpactListBridge": "List Bridge",

    "MasksToMaskList": "Mask Batch to Mask List",
    "MaskListToMaskBatch": "Mask List to Mask Batch",
    "ImpactImageBatchToImageList": "Image Batch to Image List",
    "ImageListToImageBatch": "Image List to Image Batch",

    "ImpactMakeImageList": "Make Image List",
    "ImpactMakeImageBatch": "Make Image Batch",
    "ImpactMakeMaskList": "Make Mask List",
    "ImpactMakeMaskBatch": "Make Mask Batch",
    "ImpactMakeAnyList": "Make List (Any)",
    "ImpactSelectNthItemOfAnyList": "Select Nth Item (Any list)",

    "ImpactStringSelector": "String Selector",
    "StringListToString": "String List to String",
    "WildcardPromptFromString": "Wildcard Prompt from String",
    "ImpactIsNotEmptySEGS": "SEGS isn't Empty",
    "SetDefaultImageForSEGS": "Set Default Image for SEGS",
    "RemoveImageFromSEGS": "Remove Image from SEGS",

    "RemoveNoiseMask": "Remove Noise Mask",

    "ImpactCombineConditionings": "Combine Conditionings",
    "ImpactConcatConditionings": "Concat Conditionings",

    "ImpactQueueTrigger": "Queue Trigger",
    "ImpactQueueTriggerCountdown": "Queue Trigger (Countdown)",
    "ImpactSetWidgetValue": "Set Widget Value",
    "ImpactNodeSetMuteState": "Set Mute State",
    "ImpactControlBridge": "Control Bridge",
    "ImpactSleep": "Sleep",
    "ImpactRemoteBoolean": "Remote Boolean (on prompt)",
    "ImpactRemoteInt": "Remote Int (on prompt)",

    "ImpactHFTransformersClassifierProvider": "HF Transformers Classifier Provider",
    "ImpactSEGSClassify": "SEGS Classify",

    "LatentSwitch": "Switch (latent/legacy)",
    "SEGSSwitch": "Switch (SEGS/legacy)",

    "SEGSPreviewCNet": "SEGSPreview (CNET Image)",

    "ImpactSchedulerAdapter": "Impact Scheduler Adapter",
    "GITSSchedulerFuncProvider": "GITSScheduler Func Provider",
    "ImpactNegativeConditioningPlaceholder": "Negative Cond Placeholder"
}


# NOTE:  Inject directly into EXTENSION_WEB_DIRS instead of WEB_DIRECTORY
#        Provide the js path fixed as ComfyUI-Impact-Pack instead of the path name, making it available for external use

# WEB_DIRECTORY = "js"  -- deprecated method
nodes.EXTENSION_WEB_DIRS["ComfyUI-Impact-Pack"] = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'js')


__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
