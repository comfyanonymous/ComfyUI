import base64
from PIL import Image, ImageOps
from io import BytesIO
import numpy as np
import requests
import struct
import comfy.utils
import time

#You can use this node to save full size images through the websocket, the
#images will be sent in exactly the same format as the image previews: as
#binary images on the websocket with a 8 byte header indicating the type
#of binary message (first 4 bytes) and the image format (next 4 bytes).

#Note that no metadata will be put in the images saved with this node.

class SaveImageWebsocket:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"images": ("IMAGE", ),}
                }

    RETURN_TYPES = ()
    FUNCTION = "save_images"

    OUTPUT_NODE = True

    CATEGORY = "api/image"

    def save_images(self, images):
        pbar = comfy.utils.ProgressBar(images.shape[0])
        step = 0
        for image in images:
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            pbar.update_absolute(step, images.shape[0], ("PNG", img, None))
            step += 1

        return {}

    def IS_CHANGED(s, images):
        return time.time()


class SaveImageHttp:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"images": ("IMAGE",),
                             "callback_url": ("STRING", {"default": "null"})},
                }

    RETURN_TYPES = ()
    FUNCTION = "save_images"

    OUTPUT_NODE = True

    CATEGORY = "api/image"

    def save_images(self, images, callback_url):
        pbar = comfy.utils.ProgressBar(images.shape[0])
        step = 0
        for image in images:
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

            buffer = BytesIO()
            img.save(buffer, format="PNG")
            img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")

            pbar.update_absolute(step, images.shape[0], ("PNG", img, None))
            step += 1

        req = {
            "images": img_str,
        }

        requests.post(url=callback_url, json=req)

        return {}

    def IS_CHANGED(s, images):
        return time.time()


NODE_CLASS_MAPPINGS = {
    "SaveImageWebsocket": SaveImageWebsocket,
    "SaveImageHttp": SaveImageHttp,
}