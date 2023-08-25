import os
import random
import sys
import hashlib
import base64

from clip_interrogator import Interrogator, Config
from torch import Tensor
import torchvision.transforms as T
from PIL import Image

class ClipInterrogator:
    MODEL_NAME = ["ViT-L-14/openai"]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "clip": ("CLIP",),
                "model_name": (ClipInterrogator.MODEL_NAME,),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "clip_interrogate"
    OUTPUT_NODE = True

    CATEGORY = "inflamously"

    VALUE = ""

    @classmethod
    def IS_CHANGED(s, image, clip, model_name):
        # TODO: Why does this not cache immidiately
        return hashlib.md5(str(bytearray(image.numpy())).encode("utf-8")).hexdigest()

    def clip_interrogate(self, image, clip, model_name):
        img_tensor = image[0]
        # define a transform to convert a tensor to PIL image
        transform = T.ToPILImage()
        h, w, c = img_tensor.size()
        # print(h,w,c)
        # convert the tensor to PIL image using above transform
        img = transform(image[0].reshape(c, h, w)) # Reshape since Tensor is using Height, Width, Color but Image needs C, H, W
        config = Config(clip_model_name=model_name)
        config.apply_low_vram_defaults()
        ci = Interrogator(config)
        ClipInterrogator.VALUE = ci.interrogate(img)
        print("Image:", ClipInterrogator.VALUE)
        tokens = clip.tokenize(ClipInterrogator.VALUE)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        return ([[cond, {"pooled_output": pooled}]], )


NODE_CLASS_MAPPINGS = {
    "ClipInterrogator": ClipInterrogator
}