import canny, hed, midas, mlsd, openpose, uniformer
from util import HWC3
import torch
import numpy as np

def img_np_to_tensor(img_np):
    return torch.from_numpy(img_np.astype(np.float32) / 255.0)[None,]
def img_tensor_to_np(img_tensor):
    img_tensor = img_tensor.clone()
    img_tensor = img_tensor * 255.0
    return img_tensor.squeeze(0).numpy().astype(np.uint8)
    #Thanks ChatGPT


class CannyPreprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "image": ("IMAGE", ) ,
                              "low_threshold": ("INT", {"default": 100, "min": 0, "max": 255, "step": 1}),
                              "high_threshold": ("INT", {"default": 100, "min": 0, "max": 255, "step": 1}),
                              "l2gradient": (["disable", "enable"], )
                              }}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "detect_edge"

    CATEGORY = "preprocessor"

    def detect_edge(self, image, low_threshold, high_threshold, l2gradient):
        apply_canny = canny.CannyDetector()
        image = apply_canny(img_tensor_to_np(image), low_threshold, high_threshold, l2gradient == "enable")
        image = img_np_to_tensor(HWC3(image))
        return (image,)

class HEDPreprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "image": ("IMAGE",) }}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "detect_edge"

    CATEGORY = "preprocessor"

    def detect_edge(self, image):
        apply_hed = hed.HEDdetector()
        image = apply_hed(img_tensor_to_np(image))
        image = img_np_to_tensor(HWC3(image))
        return (image,)

class MIDASPreprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "image": ("IMAGE", ) ,
                              "a": ("FLOAT", {"default": np.pi * 2.0, "min": 0.0, "max": np.pi * 5.0, "step": 0.1}),
                              "bg_threshold": ("FLOAT", {"default": 0.1, "min": 0, "max": 1, "step": 0.1})
                              }}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "estimate_depth"

    CATEGORY = "preprocessor"

    def estimate_depth(self, image, a, bg_threshold):
        model_midas = midas.MidasDetector()
        image, _ = model_midas(img_tensor_to_np(image), a, bg_threshold)
        image = img_np_to_tensor(HWC3(image))
        return (image,)

class MLSDPreprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "image": ("IMAGE",) ,
                              #Idk what should be the max value here since idk much about ML
                              "score_threshold": ("FLOAT", {"default": np.pi * 2.0, "min": 0.0, "max": np.pi * 2.0, "step": 0.1}), 
                              "dist_threshold": ("FLOAT", {"default": 0.1, "min": 0, "max": 1, "step": 0.1})
                              }}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "detect_edge"

    CATEGORY = "preprocessor"

    def detect_edge(self, image, score_threshold, dist_threshold):
        model_mlsd = mlsd.MLSDdetector()
        image = model_mlsd(img_tensor_to_np(image), score_threshold, dist_threshold)
        image = img_np_to_tensor(HWC3(image))
        return (image,)

class OpenPosePreprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "image": ("IMAGE", ),
                              "detect_hand": (["disable", "enable"],)
                              }}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "estimate_pose"

    CATEGORY = "preprocessor"

    def estimate_pose(self, image, detect_hand):
        model_openpose = openpose.OpenposeDetector()
        image, _ = model_openpose(img_tensor_to_np(image), detect_hand == "enable")
        image = img_np_to_tensor(HWC3(image))
        return (image,)

class UniformerPreprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "image": ("IMAGE", )
                              }}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "semantic_segmentate"

    CATEGORY = "preprocessor"

    def semantic_segmentate(self, image):
        model_uniformer = uniformer.UniformerDetector()
        image = model_uniformer(img_np_to_tensor(image))
        image = img_np_to_tensor(HWC3(image))
        return (image,)

NODE_CLASS_MAPPING = {
    "CannyPreprocessor": CannyPreprocessor,
    "HEDPreprocessor": HEDPreprocessor,
    "DepthPreprocessor": MIDASPreprocessor,
    "MLSDPreprocessor": MLSDPreprocessor,
    "OpenPosePreprocessor": OpenPosePreprocessor,
}