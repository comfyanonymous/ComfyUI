import torch
from urllib import request
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import numpy as np

class LoadImageUrl:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "url": ("STRING", { "multiline": False, })
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "load_image"
    CATEGORY = "image"

    def load_image(self, url):
        with request.urlopen(url) as r:
            i = Image.open(r)
        image = i.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        if 'A' in i.getbands():
            mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
            mask = 1. - torch.from_numpy(mask)
        else:
            mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
        return (image, mask)

NODE_CLASS_MAPPINGS = {
    "LoadImageUrl": LoadImageUrl,
}
