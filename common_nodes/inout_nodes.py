
import os
import hashlib
import json
import uuid

from PIL import Image, ImageOps
from PIL.PngImagePlugin import PngInfo
import numpy as np
import folder_paths
import torch

from comfy.cli_args import args
from framework.model.object_storage import ResourceMgr
from config.config import CONFIG
from framework.app_log import AppLog


class IntInput:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
            "name": ("STRING", {"default": ""}),
            "data": ("INT", {"default": 0})
            }}
    
    
    RETURN_TYPES = ("INT", )
    FUNCTION = "execute"

    CATEGORY = "flow"
    
    INPUT_NODE = True
    INPUT_NODE_TYPE = "INT"
    INPUT_NODE_DATA = "data"
    
    
    def execute(self, name, data):
     
        return (data, ) 
    
    
    
    
class FloatInput:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
            "name": ("STRING", {"default": ""}),
            "data": ("FLOAT", {"default": 0})
            }}
    
    
    RETURN_TYPES = ("FLOAT", )
    FUNCTION = "execute"

    CATEGORY = "flow"
    
    INPUT_NODE = True
    INPUT_NODE_TYPE = "FLOAT"
    INPUT_NODE_DATA = "data"
    
    
    
    def execute(self, name, data):
     
        return (data, ) 
    
       
    
    
class StringInput:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
            "name": ("STRING", {"default": ""}),
            "data": ("STRING", {"default": "", "multiline": True})
            }}
    
    
    RETURN_TYPES = ("STRING", )
    FUNCTION = "execute"

    CATEGORY = "flow"
    
    INPUT_NODE = True
    INPUT_NODE_TYPE = "STRING"
    INPUT_NODE_DATA = "data"
    
    
    
    def execute(self, name, data):
        return (data, ) 
    
    
    
class BoolInput:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
            "name": ("STRING", {"default": ""}),
            "data": ("BOOLEAN", {"default": False})
            }}
    
    
    RETURN_TYPES = ("BOOLEAN", )
    FUNCTION = "execute"

    CATEGORY = "flow"
    
    INPUT_NODE = True
    INPUT_NODE_TYPE = "BOOLEAN"
    INPUT_NODE_DATA = "data"
    
    
    
    def execute(self, name, data):
        return (data, ) 
    
    
    
class ImageInput:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {"required":
                    {"name": ("STRING", {"default": ""}),
                        "image": (sorted(files), {"image_upload": True})},
                }

    CATEGORY = "flow"
    
    INPUT_NODE = True
    INPUT_NODE_TYPE = "IMAGE"
    INPUT_NODE_DATA = "image"
    

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "load_image"
    def load_image(self, name, image):
        AppLog.info(f"[ImageInput] load_image, image: {image}")
        image_path, i = ResourceMgr.instance.get_image(image)
        AppLog.info(f'[ImageInput] load_image, img path: {image_path}')
        i = ImageOps.exif_transpose(i)
        image = i.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        if 'A' in i.getbands():
            mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
            mask = 1. - torch.from_numpy(mask)
        else:
            mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
        return (image, mask.unsqueeze(0))

    @classmethod
    def IS_CHANGED(s, name, image):
        AppLog.info(f"[ImageInput] load_image, image: {image}")
        image_path, _ = ResourceMgr.instance.get_image(image, open=False)
        AppLog.info(f'[ImageInput] load_image, img path: {image_path}')
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, name, image):
        if not ResourceMgr.instance.exist_image(image):
            return "Invalid image file: {}".format(image)

        return True
    
    



class ImageOutput:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "name": ("STRING", {"default": ""}),
            "images": ("IMAGE", ),
            "filename_prefix": ("STRING", {"default": "ComfyUI"})},
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
                }

    RETURN_TYPES = ("IMAGE", )
    FUNCTION = "save_images"

    OUTPUT_NODE = True
    OUTPUT_NODE_TYPE = "IMAGE"

    CATEGORY = "flow"

    def save_images(self, name, images, filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None):
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])
        results = list()
        for image in images:
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            metadata = None
            if not args.disable_metadata:
                metadata = PngInfo()
                if prompt is not None:
                    metadata.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo is not None:
                    for x in extra_pnginfo:
                        metadata.add_text(x, json.dumps(extra_pnginfo[x]))

            file = f"{str(uuid.uuid4())}.png" #f"{filename}_{counter:05}_.png"
            full_output_folder = CONFIG["resource"]["out_img_path_local"]
            local_filepath = os.path.join(full_output_folder, file)
            img.save(local_filepath, pnginfo=metadata, compress_level=self.compress_level)
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })
            counter += 1
            
            final_path = ResourceMgr.instance.after_save_image_to_local(local_filepath)
            
        return { "ui": { "images": results }, "result": (final_path,) }



NODE_CLASS_MAPPINGS = {
    
    "IntInput": IntInput,
    "FloatInput": FloatInput,
    "StringInput": StringInput,
    "BoolInput": BoolInput,
    "ImageInput": ImageInput,
    "ImageOutput": ImageOutput
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IntInput": "Int Input",
    "FloatInput": "Float Input",
    "StringInput": "String Input",
    "BoolInput": "Bool Input",
    "ImageInput": "Image Input",
    "ImageOutput": "Image Output"
} 
