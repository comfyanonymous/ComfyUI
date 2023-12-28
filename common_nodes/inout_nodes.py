
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
from framework.model import object_storage
from config.config import CONFIG


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
    INPUT_NODE_TYPE = "IMAGE"
    
    
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
    INPUT_NODE_TYPE = "IMAGE"
    
    
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
    INPUT_NODE_TYPE = "IMAGE"
    
    
    def execute(self, name, data):
        return (data, ) 
    
    
    
class ImageInput:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {"required":
                    {"name": ("STRING", {"default": ""}),
                        "data": (sorted(files), {"image_upload": True})},
                }

    CATEGORY = "flow"
    
    INPUT_NODE = True
    INPUT_NODE_TYPE = "IMAGE"

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "load_image"
    def load_image(self, name, data):
        # image_path = folder_paths.get_annotated_filepath(image)
        print(f"image data: {data}")
        image_path = object_storage.MinIOConnection().fget_object(data)
        print(f'img path: {image_path}')
        i = Image.open(image_path)
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
    def IS_CHANGED(s, name, data):
        print(f"image data: {data}")
        image_path = object_storage.MinIOConnection().fget_object(data)
        print(f'img path: {image_path}')
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, name, data):
        if not folder_paths.exists_annotated_filepath(data):
            return "Invalid image file: {}".format(data)

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

    RETURN_TYPES = ("STRING", )
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
            local_filepath = os.path.join(full_output_folder, file)
            img.save(local_filepath, pnginfo=metadata, compress_level=self.compress_level)
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })
            counter += 1
            
            remote_dir = CONFIG["resource"]["out_img_path_cloud"]
            remote_path = f"{remote_dir}/{file}"
            object_storage.MinIOConnection().fput_object(remote_path, local_filepath)
            print(f"[ImageOutput] remote path: {remote_path}")

        return { "ui": { "images": results }, "result": (remote_path,) }



NODE_CLASS_MAPPINGS = {
    
    "IntInput": IntInput,
    "FloatInput": FloatInput,
    "StringInput": StringInput,
    "ImageInput": ImageInput,
    "ImageOutput": ImageOutput
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IntInput": "Int Input",
    "FloatInput": "Float Input",
    "StringInput": "String Input",
    "ImageInput": "Image Input",
    "ImageOutput": "Image Output"
} 
