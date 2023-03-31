import datetime

import torch

import os
import sys
import json
import hashlib
import copy
import traceback

from PIL import Image
from PIL.PngImagePlugin import PngInfo
import numpy as np
import comfy.samplers
import comfy.sd
import comfy.utils
import comfy_extras.clip_vision
import model_management
import importlib
import folder_paths


class Note:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"text": ("STRING", {"multiline": True})}}

    RETURN_TYPES = ()
    FUNCTION = "Note"

    OUTPUT_NODE = False

    CATEGORY = "silver_custom"


class SaveImageList:
    def __init__(self):
        current_dir = os.path.abspath(os.getcwd())
        self.output_dir = os.path.join(current_dir, "output")
        self.type = "output"

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"images": ("IMAGE",),
                     "filename_prefix": ("STRING", {"default": "ComfyUI"})},
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
                }

    RETURN_TYPES = ()
    FUNCTION = "save_images_list"

    OUTPUT_NODE = True

    CATEGORY = "silver_custom"

    def save_images_list(self, images, filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None):
        def map_filename(filename):
            prefix_len = len(os.path.basename(filename_prefix))
            prefix = filename[:prefix_len + 1]
            try:
                digits = int(filename[prefix_len + 1:].split('_')[0])
            except:
                digits = 0
            return (digits, prefix)

        subfolder = os.path.dirname(os.path.normpath(filename_prefix))
        filename = os.path.basename(os.path.normpath(filename_prefix))

        full_output_folder = os.path.join(self.output_dir, subfolder)

        if os.path.commonpath((self.output_dir, os.path.realpath(full_output_folder))) != self.output_dir:
            print("Saving image outside the output folder is not allowed.")
            return {}

        try:
            counter = max(filter(lambda a: a[1][:-1] == filename and a[1][-1] == "_",
                                 map(map_filename, os.listdir(full_output_folder))))[0] + 1
        except ValueError:
            counter = 1
        except FileNotFoundError:
            os.makedirs(full_output_folder, exist_ok=True)
            counter = 1

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        results = list()
        for image in images:
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            metadata = PngInfo()
            if prompt is not None:
                metadata.add_text("prompt", json.dumps(prompt))
            if extra_pnginfo is not None:
                for x in extra_pnginfo:
                    metadata.add_text(x, json.dumps(extra_pnginfo[x]))

            now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            file = f"{filename}-{now}_{counter:05}_.png"
            img.save(os.path.join(full_output_folder, file), pnginfo=metadata, optimize=True)
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })
            counter += 1

        return self.get_all_files()

    def get_all_files(self):
        results = []
        for root, dirs, files in os.walk(self.output_dir):
            for file in files:
                subfolder = os.path.relpath(root, self.output_dir)
                results.append({
                    "filename": file,
                    "subfolder": subfolder,
                    "type": self.type
                })
        sorted_results = sorted(results, key=lambda x: x["filename"])
        return {"ui": {"images": sorted_results}}


NODE_CLASS_MAPPINGS = {
    "Note": Note,
    "SaveImageList": SaveImageList,
}
