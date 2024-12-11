from .nodes_preprocessing import MD_LoadImageFromUrl, MD_CompressAdjustNode, MD_ImageToMotionPrompt
from .nodes_model import MD_LoadVideoModel, MD_ImgToVideo, MD_VideoSampler
from .nodes_output import MD_SaveAnimatedWEBP, MD_SaveMP4
NODE_CLASS_MAPPINGS = {
    # PREPROCESSING
    "Memedeck_ImageToMotionPrompt": MD_ImageToMotionPrompt,
    "Memedeck_CompressAdjustNode": MD_CompressAdjustNode,
    "Memedeck_LoadImageFromUrl": MD_LoadImageFromUrl,
    # MODEL NODES
    "Memedeck_LoadVideoModel": MD_LoadVideoModel,
    "Memedeck_ImgToVideo": MD_ImgToVideo,
    "Memedeck_VideoSampler": MD_VideoSampler,
    # POSTPROCESSING
    "Memedeck_SaveMP4": MD_SaveMP4,
    "Memedeck_SaveAnimatedWEBP": MD_SaveAnimatedWEBP
    # "Memedeck_SaveAnimatedGIF": MD_SaveAnimatedGIF
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # PREPROCESSING
    "Memedeck_ImageToMotionPrompt": "MemeDeck: Image To Motion Prompt",
    "Memedeck_CompressAdjustNode": "MemeDeck: Compression Detector & Adjuster",
    "Memedeck_LoadImageFromUrl": "MemeDeck: Load Image From URL",
    # MODEL NODES
    "Memedeck_LoadVideoModel": "MemeDeck: Load Video Model",
    "Memedeck_VideoScheduler": "MemeDeck: Video Scheduler",
    "Memedeck_ImgToVideo": "MemeDeck: Image To Video",
    "Memedeck_VideoSampler": "MemeDeck: Video Sampler",
    # POSTPROCESSING
    "Memedeck_SaveMP4": "MemeDeck: Save MP4"
    # "Memedeck_SaveAnimatedGIF": "MemeDeck: Save Animated GIF"
}
