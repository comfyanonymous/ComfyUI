from .nodes.nodes import *
from .nodes.curve_nodes import *
from .nodes.batchcrop_nodes import *
from .nodes.audioscheduler_nodes import *
from .nodes.image_nodes import *
from .nodes.intrinsic_lora_nodes import *
from .nodes.mask_nodes import *
from .nodes.model_optimization_nodes import *
NODE_CONFIG = {
    #constants
    "BOOLConstant": {"class": BOOLConstant, "name": "BOOL Constant"},
    "INTConstant": {"class": INTConstant, "name": "INT Constant"},
    "FloatConstant": {"class": FloatConstant, "name": "Float Constant"},
    "StringConstant": {"class": StringConstant, "name": "String Constant"},
    "StringConstantMultiline": {"class": StringConstantMultiline, "name": "String Constant Multiline"},
    #conditioning
    "ConditioningMultiCombine": {"class": ConditioningMultiCombine, "name": "Conditioning Multi Combine"},
    "ConditioningSetMaskAndCombine": {"class": ConditioningSetMaskAndCombine, "name": "ConditioningSetMaskAndCombine"},
    "ConditioningSetMaskAndCombine3": {"class": ConditioningSetMaskAndCombine3, "name": "ConditioningSetMaskAndCombine3"},
    "ConditioningSetMaskAndCombine4": {"class": ConditioningSetMaskAndCombine4, "name": "ConditioningSetMaskAndCombine4"},
    "ConditioningSetMaskAndCombine5": {"class": ConditioningSetMaskAndCombine5, "name": "ConditioningSetMaskAndCombine5"},
    "CondPassThrough": {"class": CondPassThrough},
    #masking
    "DownloadAndLoadCLIPSeg": {"class": DownloadAndLoadCLIPSeg, "name": "(Down)load CLIPSeg"},
    "BatchCLIPSeg": {"class": BatchCLIPSeg, "name": "Batch CLIPSeg"},
    "ColorToMask": {"class": ColorToMask, "name": "Color To Mask"},
    "CreateGradientMask": {"class": CreateGradientMask, "name": "Create Gradient Mask"},
    "CreateTextMask": {"class": CreateTextMask, "name": "Create Text Mask"},
    "CreateAudioMask": {"class": CreateAudioMask, "name": "Create Audio Mask"},
    "CreateFadeMask": {"class": CreateFadeMask, "name": "Create Fade Mask"},
    "CreateFadeMaskAdvanced": {"class": CreateFadeMaskAdvanced, "name": "Create Fade Mask Advanced"},
    "CreateFluidMask": {"class": CreateFluidMask, "name": "Create Fluid Mask"},
    "CreateShapeMask": {"class": CreateShapeMask, "name": "Create Shape Mask"},
    "CreateVoronoiMask": {"class": CreateVoronoiMask, "name": "Create Voronoi Mask"},
    "CreateMagicMask": {"class": CreateMagicMask, "name": "Create Magic Mask"},
    "GetMaskSizeAndCount": {"class": GetMaskSizeAndCount, "name": "Get Mask Size & Count"},
    "GrowMaskWithBlur": {"class": GrowMaskWithBlur, "name": "Grow Mask With Blur"},
    "MaskBatchMulti": {"class": MaskBatchMulti, "name": "Mask Batch Multi"},
    "OffsetMask": {"class": OffsetMask, "name": "Offset Mask"},
    "RemapMaskRange": {"class": RemapMaskRange, "name": "Remap Mask Range"},
    "ResizeMask": {"class": ResizeMask, "name": "Resize Mask"},
    "RoundMask": {"class": RoundMask, "name": "Round Mask"},
    "SeparateMasks": {"class": SeparateMasks, "name": "Separate Masks"},
    #images
    "AddLabel": {"class": AddLabel, "name": "Add Label"},
    "ColorMatch": {"class": ColorMatch, "name": "Color Match"},
    "ImageTensorList": {"class": ImageTensorList, "name": "Image Tensor List"},
    "CrossFadeImages": {"class": CrossFadeImages, "name": "Cross Fade Images"},
    "CrossFadeImagesMulti": {"class": CrossFadeImagesMulti, "name": "Cross Fade Images Multi"},
    "GetImagesFromBatchIndexed": {"class": GetImagesFromBatchIndexed, "name": "Get Images From Batch Indexed"},
    "GetImageRangeFromBatch": {"class": GetImageRangeFromBatch, "name": "Get Image or Mask Range From Batch"},
    "GetLatentRangeFromBatch": {"class": GetLatentRangeFromBatch, "name": "Get Latent Range From Batch"},
    "GetImageSizeAndCount": {"class": GetImageSizeAndCount, "name": "Get Image Size & Count"},
    "FastPreview": {"class": FastPreview, "name": "Fast Preview"},
    "ImageBatchFilter": {"class": ImageBatchFilter, "name": "Image Batch Filter"},
    "ImageAndMaskPreview": {"class": ImageAndMaskPreview},
    "ImageAddMulti": {"class": ImageAddMulti, "name": "Image Add Multi"},
    "ImageBatchMulti": {"class": ImageBatchMulti, "name": "Image Batch Multi"},
    "ImageBatchRepeatInterleaving": {"class": ImageBatchRepeatInterleaving},
    "ImageBatchTestPattern": {"class": ImageBatchTestPattern, "name": "Image Batch Test Pattern"},
    "ImageConcanate": {"class": ImageConcanate, "name": "Image Concatenate"},
    "ImageConcatFromBatch": {"class": ImageConcatFromBatch, "name": "Image Concatenate From Batch"},
    "ImageConcatMulti": {"class": ImageConcatMulti, "name": "Image Concatenate Multi"},
    "ImageCropByMask": {"class": ImageCropByMask, "name": "Image Crop By Mask"},
    "ImageCropByMaskAndResize": {"class": ImageCropByMaskAndResize, "name": "Image Crop By Mask And Resize"},
    "ImageCropByMaskBatch": {"class": ImageCropByMaskBatch, "name": "Image Crop By Mask Batch"},
    "ImageUncropByMask": {"class": ImageUncropByMask, "name": "Image Uncrop By Mask"},
    "ImageGrabPIL": {"class": ImageGrabPIL, "name": "Image Grab PIL"},
    "ImageGridComposite2x2": {"class": ImageGridComposite2x2, "name": "Image Grid Composite 2x2"},
    "ImageGridComposite3x3": {"class": ImageGridComposite3x3, "name": "Image Grid Composite 3x3"},
    "ImageGridtoBatch": {"class": ImageGridtoBatch, "name": "Image Grid To Batch"},
    "ImageNoiseAugmentation": {"class": ImageNoiseAugmentation, "name": "Image Noise Augmentation"},
    "ImageNormalize_Neg1_To_1": {"class": ImageNormalize_Neg1_To_1, "name": "Image Normalize -1 to 1"},
    "ImagePass": {"class": ImagePass},
    "ImagePadKJ": {"class": ImagePadKJ, "name": "ImagePad KJ"},
    "ImagePadForOutpaintMasked": {"class": ImagePadForOutpaintMasked, "name": "Image Pad For Outpaint Masked"},
    "ImagePadForOutpaintTargetSize": {"class": ImagePadForOutpaintTargetSize, "name": "Image Pad For Outpaint Target Size"},
    "ImagePrepForICLora": {"class": ImagePrepForICLora, "name": "Image Prep For ICLora"},
    "ImageResizeKJ": {"class": ImageResizeKJ, "name": "Resize Image (deprecated)"},
    "ImageResizeKJv2": {"class": ImageResizeKJv2, "name": "Resize Image v2"},
    "ImageUpscaleWithModelBatched": {"class": ImageUpscaleWithModelBatched, "name": "Image Upscale With Model Batched"},
    "InsertImagesToBatchIndexed": {"class": InsertImagesToBatchIndexed, "name": "Insert Images To Batch Indexed"},
    "InsertLatentToIndexed": {"class": InsertLatentToIndex, "name": "Insert Latent To Index"},
    "LoadAndResizeImage": {"class": LoadAndResizeImage, "name": "Load & Resize Image"},
    "LoadImagesFromFolderKJ": {"class": LoadImagesFromFolderKJ, "name": "Load Images From Folder (KJ)"},
    "MergeImageChannels": {"class": MergeImageChannels, "name": "Merge Image Channels"},
    "PadImageBatchInterleaved": {"class": PadImageBatchInterleaved, "name": "Pad Image Batch Interleaved"},
    "PreviewAnimation": {"class": PreviewAnimation, "name": "Preview Animation"},
    "RemapImageRange": {"class": RemapImageRange, "name": "Remap Image Range"},
    "ReverseImageBatch": {"class": ReverseImageBatch, "name": "Reverse Image Batch"},
    "ReplaceImagesInBatch": {"class": ReplaceImagesInBatch, "name": "Replace Images In Batch"},
    "SaveImageWithAlpha": {"class": SaveImageWithAlpha, "name": "Save Image With Alpha"},
    "SaveImageKJ": {"class": SaveImageKJ, "name": "Save Image KJ"},
    "ShuffleImageBatch": {"class": ShuffleImageBatch, "name": "Shuffle Image Batch"},
    "SplitImageChannels": {"class": SplitImageChannels, "name": "Split Image Channels"},
    "TransitionImagesMulti": {"class": TransitionImagesMulti, "name": "Transition Images Multi"},
    "TransitionImagesInBatch": {"class": TransitionImagesInBatch, "name": "Transition Images In Batch"},
    #batch cropping
    "BatchCropFromMask": {"class": BatchCropFromMask, "name": "Batch Crop From Mask"},
    "BatchCropFromMaskAdvanced": {"class": BatchCropFromMaskAdvanced, "name": "Batch Crop From Mask Advanced"},
    "FilterZeroMasksAndCorrespondingImages": {"class": FilterZeroMasksAndCorrespondingImages},
    "InsertImageBatchByIndexes": {"class": InsertImageBatchByIndexes, "name": "Insert Image Batch By Indexes"},
    "BatchUncrop": {"class": BatchUncrop, "name": "Batch Uncrop"},
    "BatchUncropAdvanced": {"class": BatchUncropAdvanced, "name": "Batch Uncrop Advanced"},
    "SplitBboxes": {"class": SplitBboxes, "name": "Split Bboxes"},
    "BboxToInt": {"class": BboxToInt, "name": "Bbox To Int"},
    "BboxVisualize": {"class": BboxVisualize, "name": "Bbox Visualize"},
    #noise
    "GenerateNoise": {"class": GenerateNoise, "name": "Generate Noise"},
    "FlipSigmasAdjusted": {"class": FlipSigmasAdjusted, "name": "Flip Sigmas Adjusted"},
    "InjectNoiseToLatent": {"class": InjectNoiseToLatent, "name": "Inject Noise To Latent"},
    "CustomSigmas": {"class": CustomSigmas, "name": "Custom Sigmas"},
    #utility
    "StringToFloatList": {"class": StringToFloatList, "name": "String to Float List"},
    "WidgetToString": {"class": WidgetToString, "name": "Widget To String"},
    "SaveStringKJ": {"class": SaveStringKJ, "name": "Save String KJ"},
    "DummyOut": {"class": DummyOut, "name": "Dummy Out"},
    "GetLatentsFromBatchIndexed": {"class": GetLatentsFromBatchIndexed, "name": "Get Latents From Batch Indexed"},
    "ScaleBatchPromptSchedule": {"class": ScaleBatchPromptSchedule, "name": "Scale Batch Prompt Schedule"},
    "CameraPoseVisualizer": {"class": CameraPoseVisualizer, "name": "Camera Pose Visualizer"},
    "AppendStringsToList": {"class": AppendStringsToList, "name": "Append Strings To List"},
    "JoinStrings": {"class": JoinStrings, "name": "Join Strings"},
    "JoinStringMulti": {"class": JoinStringMulti, "name": "Join String Multi"},
    "SomethingToString": {"class": SomethingToString, "name": "Something To String"},
    "Sleep": {"class": Sleep, "name": "Sleep"},
    "VRAM_Debug": {"class": VRAM_Debug, "name": "VRAM Debug"},
    "SomethingToString": {"class": SomethingToString, "name": "Something To String"},
    "EmptyLatentImagePresets": {"class": EmptyLatentImagePresets, "name": "Empty Latent Image Presets"},
    "EmptyLatentImageCustomPresets": {"class": EmptyLatentImageCustomPresets, "name": "Empty Latent Image Custom Presets"},
    "ModelPassThrough": {"class": ModelPassThrough, "name": "ModelPass"},
    "ModelSaveKJ": {"class": ModelSaveKJ, "name": "Model Save KJ"},
    "SetShakkerLabsUnionControlNetType": {"class": SetShakkerLabsUnionControlNetType, "name": "Set Shakker Labs Union ControlNet Type"},
    "StyleModelApplyAdvanced": {"class": StyleModelApplyAdvanced, "name": "Style Model Apply Advanced"},
    #audioscheduler stuff
    "NormalizedAmplitudeToMask": {"class": NormalizedAmplitudeToMask},
    "NormalizedAmplitudeToFloatList": {"class": NormalizedAmplitudeToFloatList},
    "OffsetMaskByNormalizedAmplitude": {"class": OffsetMaskByNormalizedAmplitude},
    "ImageTransformByNormalizedAmplitude": {"class": ImageTransformByNormalizedAmplitude},
    "AudioConcatenate": {"class": AudioConcatenate},
    #curve nodes
    "SplineEditor": {"class": SplineEditor, "name": "Spline Editor"},
    "CreateShapeImageOnPath": {"class": CreateShapeImageOnPath, "name": "Create Shape Image On Path"},
    "CreateShapeMaskOnPath": {"class": CreateShapeMaskOnPath, "name": "Create Shape Mask On Path"},
    "CreateTextOnPath": {"class": CreateTextOnPath, "name": "Create Text On Path"},
    "CreateGradientFromCoords": {"class": CreateGradientFromCoords, "name": "Create Gradient From Coords"},
    "CutAndDragOnPath": {"class": CutAndDragOnPath, "name": "Cut And Drag On Path"},
    "GradientToFloat": {"class": GradientToFloat, "name": "Gradient To Float"},
    "WeightScheduleExtend": {"class": WeightScheduleExtend, "name": "Weight Schedule Extend"},
    "MaskOrImageToWeight": {"class": MaskOrImageToWeight, "name": "Mask Or Image To Weight"},
    "WeightScheduleConvert": {"class": WeightScheduleConvert, "name": "Weight Schedule Convert"},
    "FloatToMask": {"class": FloatToMask, "name": "Float To Mask"},
    "FloatToSigmas": {"class": FloatToSigmas, "name": "Float To Sigmas"},
    "SigmasToFloat": {"class": SigmasToFloat, "name": "Sigmas To Float"},
    "PlotCoordinates": {"class": PlotCoordinates, "name": "Plot Coordinates"},
    "InterpolateCoords": {"class": InterpolateCoords, "name": "Interpolate Coords"},
    "PointsEditor": {"class": PointsEditor, "name": "Points Editor"},
    #experimental
    "SoundReactive": {"class": SoundReactive, "name": "Sound Reactive"},
    "StableZero123_BatchSchedule": {"class": StableZero123_BatchSchedule, "name": "Stable Zero123 Batch Schedule"},
    "SV3D_BatchSchedule": {"class": SV3D_BatchSchedule, "name": "SV3D Batch Schedule"},
    "LoadResAdapterNormalization": {"class": LoadResAdapterNormalization},
    "Superprompt": {"class": Superprompt, "name": "Superprompt"},
    "GLIGENTextBoxApplyBatchCoords": {"class": GLIGENTextBoxApplyBatchCoords},
    "Intrinsic_lora_sampling": {"class": Intrinsic_lora_sampling, "name": "Intrinsic Lora Sampling"},
    "CheckpointPerturbWeights": {"class": CheckpointPerturbWeights, "name": "CheckpointPerturbWeights"},
    "Screencap_mss": {"class": Screencap_mss, "name": "Screencap mss"},
    "WebcamCaptureCV2": {"class": WebcamCaptureCV2, "name": "Webcam Capture CV2"},
    "DifferentialDiffusionAdvanced": {"class": DifferentialDiffusionAdvanced, "name": "Differential Diffusion Advanced"},
    "DiTBlockLoraLoader": {"class": DiTBlockLoraLoader, "name": "DiT Block Lora Loader"},
    "FluxBlockLoraSelect": {"class": FluxBlockLoraSelect, "name": "Flux Block Lora Select"},
    "HunyuanVideoBlockLoraSelect": {"class": HunyuanVideoBlockLoraSelect, "name": "Hunyuan Video Block Lora Select"},
    "Wan21BlockLoraSelect": {"class": Wan21BlockLoraSelect, "name": "Wan21 Block Lora Select"},
    "CustomControlNetWeightsFluxFromList": {"class": CustomControlNetWeightsFluxFromList, "name": "Custom ControlNet Weights Flux From List"},
    "CheckpointLoaderKJ": {"class": CheckpointLoaderKJ, "name": "CheckpointLoaderKJ"},
    "DiffusionModelLoaderKJ": {"class": DiffusionModelLoaderKJ, "name": "Diffusion Model Loader KJ"},
    "TorchCompileModelFluxAdvanced": {"class": TorchCompileModelFluxAdvanced, "name": "TorchCompileModelFluxAdvanced"},
    "TorchCompileModelFluxAdvancedV2": {"class": TorchCompileModelFluxAdvancedV2, "name": "TorchCompileModelFluxAdvancedV2"},
    "TorchCompileModelHyVideo": {"class": TorchCompileModelHyVideo, "name": "TorchCompileModelHyVideo"},
    "TorchCompileVAE": {"class": TorchCompileVAE, "name": "TorchCompileVAE"},
    "TorchCompileControlNet": {"class": TorchCompileControlNet, "name": "TorchCompileControlNet"},
    "PatchModelPatcherOrder": {"class": PatchModelPatcherOrder, "name": "Patch Model Patcher Order"},
    "TorchCompileLTXModel": {"class": TorchCompileLTXModel, "name": "TorchCompileLTXModel"},
    "TorchCompileCosmosModel": {"class": TorchCompileCosmosModel, "name": "TorchCompileCosmosModel"},
    "TorchCompileModelWanVideo": {"class": TorchCompileModelWanVideo, "name": "TorchCompileModelWanVideo"},
    "TorchCompileModelWanVideoV2": {"class": TorchCompileModelWanVideoV2, "name": "TorchCompileModelWanVideoV2"},
    "PathchSageAttentionKJ": {"class": PathchSageAttentionKJ, "name": "Patch Sage Attention KJ"},
    "LeapfusionHunyuanI2VPatcher": {"class": LeapfusionHunyuanI2V, "name": "Leapfusion Hunyuan I2V Patcher"},
    "VAELoaderKJ": {"class": VAELoaderKJ, "name": "VAELoader KJ"},
    "ScheduledCFGGuidance": {"class": ScheduledCFGGuidance, "name": "Scheduled CFG Guidance"},
    "ApplyRifleXRoPE_HunuyanVideo": {"class": ApplyRifleXRoPE_HunuyanVideo, "name": "Apply RifleXRoPE HunuyanVideo"},
    "ApplyRifleXRoPE_WanVideo": {"class": ApplyRifleXRoPE_WanVideo, "name": "Apply RifleXRoPE WanVideo"},
    "WanVideoTeaCacheKJ": {"class": WanVideoTeaCacheKJ, "name": "WanVideo Tea Cache (native)"},
    "WanVideoEnhanceAVideoKJ": {"class": WanVideoEnhanceAVideoKJ, "name": "WanVideo Enhance A Video (native)"},
    "SkipLayerGuidanceWanVideo": {"class": SkipLayerGuidanceWanVideo, "name": "Skip Layer Guidance WanVideo"},
    "TimerNodeKJ": {"class": TimerNodeKJ, "name": "Timer Node KJ"},
    "HunyuanVideoEncodeKeyframesToCond": {"class": HunyuanVideoEncodeKeyframesToCond, "name": "HunyuanVideo Encode Keyframes To Cond"},
    "CFGZeroStarAndInit": {"class": CFGZeroStarAndInit, "name": "CFG Zero Star/Init"},
    "ModelPatchTorchSettings": {"class": ModelPatchTorchSettings, "name": "Model Patch Torch Settings"},
    "WanVideoNAG": {"class": WanVideoNAG, "name": "WanVideoNAG"},

    #instance diffusion
    "CreateInstanceDiffusionTracking": {"class": CreateInstanceDiffusionTracking},
    "AppendInstanceDiffusionTracking": {"class": AppendInstanceDiffusionTracking},
    "DrawInstanceDiffusionTracking": {"class": DrawInstanceDiffusionTracking},
}

def generate_node_mappings(node_config):
    node_class_mappings = {}
    node_display_name_mappings = {}

    for node_name, node_info in node_config.items():
        node_class_mappings[node_name] = node_info["class"]
        node_display_name_mappings[node_name] = node_info.get("name", node_info["class"].__name__)

    return node_class_mappings, node_display_name_mappings

NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS = generate_node_mappings(NODE_CONFIG)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

WEB_DIRECTORY = "./web"

from aiohttp import web
from server import PromptServer
from pathlib import Path

if hasattr(PromptServer, "instance"):
    try:
        # NOTE: we add an extra static path to avoid comfy mechanism
        # that loads every script in web.
        PromptServer.instance.app.add_routes(
            [web.static("/kjweb_async", (Path(__file__).parent.absolute() / "kjweb_async").as_posix())]
        )
    except:
        pass