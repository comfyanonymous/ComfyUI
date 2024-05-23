

import numpy as np
import pyheif
import cv2
import whatimage
import time
from PIL import Image
import requests
import torch

def image2format(image: bytes):

    fmt = whatimage.identify_image(image)
    if fmt in ["heic", "avif"]:
        i = pyheif.read_heif(image)
        image = Image.frombytes(mode=i.mode, size=i.size, data=i.data)
        image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    else:
        image = np.asarray(bytearray(image), dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    return image

class image_url_node:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_url": ("STRING", {"default": "null"}),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "download_image"
    CATEGORY = "image"

    def download_image(self, image_url):
        res = requests.get(image_url, timeout=3).content
        image = image2format(res)

        image = image.astype(np.float32) / 255.
        image = torch.from_numpy(image)
        image = image.unsqueeze(0)

        return (image,)

NODE_CLASS_MAPPINGS = {
    "image_url_node": image_url_node
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "image_url_node": "image_url_node"
}