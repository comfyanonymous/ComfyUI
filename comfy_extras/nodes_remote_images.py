import torch
import os
import json
from io import BytesIO
from base64 import b64encode
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

class SaveImageUrl:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", ),
                "url": ("STRING", { "multiline": False, }),
                "filename_prefix": ("STRING", {"default": "ComfyUI"}),
                "data_format": (["HTML_image", "Raw_data"],)
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "save_images"
    CATEGORY = "image"
    
    def save_images(self, images, url, data_format, filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None):
        def compute_vars(input):
            input = input.replace("%width%", str(images[0].shape[1]))
            input = input.replace("%height%", str(images[0].shape[0]))
            return input

        filename = os.path.basename(os.path.normpath(filename_prefix))

        counter = 1
        files = dict()
        for image in images:
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            metadata = PngInfo()
            if prompt is not None:
                metadata.add_text("prompt", json.dumps(prompt))
            if extra_pnginfo is not None:
                for x in extra_pnginfo:
                    metadata.add_text(x, json.dumps(extra_pnginfo[x]))
        
            file = f"{filename}_{counter:05}.png"

            buffer = BytesIO()
            img.save(buffer, "png", pnginfo=metadata, compress_level=4)
            buffer.seek(0)
            encoded = b64encode(buffer.read()).decode('utf-8')
            files[file] = f"data:image/png;base64,{encoded}" if data_format == "HTML_image" else encoded
            counter += 1

        data=bytes(json.dumps(files), encoding="utf-8")
        r = request.Request(url, data=data, method="POST")
        request.urlopen(r)
        return ()

NODE_CLASS_MAPPINGS = {
    "LoadImageUrl": LoadImageUrl,
    "SaveImageUrl": SaveImageUrl,
}
