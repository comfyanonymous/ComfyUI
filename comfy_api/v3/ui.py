from __future__ import annotations
from abc import ABC, abstractmethod

from comfy_api.v3.io import Image, Mask, FolderType, _UIOutput, ComfyNodeV3
# used for image preview
from comfy.cli_args import args
import folder_paths
import random
from PIL import Image as PILImage
from PIL.PngImagePlugin import PngInfo
import os
import json
import numpy as np


class SavedResult:
    def __init__(self, filename: str, subfolder: str, type: FolderType):
        self.filename = filename
        self.subfolder = subfolder
        self.type = type
    
    def as_dict(self):
        return {
            "filename": self.filename,
            "subfolder": self.subfolder,
            "type": self.type
        }

class PreviewImage(_UIOutput):
    def __init__(self, image: Image.Type, animated: bool=False, cls: ComfyNodeV3=None, **kwargs):
        output_dir = folder_paths.get_temp_directory()
        type = "temp"
        prefix_append = "_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))
        compress_level = 1
        filename_prefix = "ComfyUI"

        filename_prefix += prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, output_dir, image[0].shape[1], image[0].shape[0])
        results = list()
        for (batch_number, image) in enumerate(image):
            i = 255. * image.cpu().numpy()
            img = PILImage.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            metadata = None
            if not args.disable_metadata and cls is not None:
                metadata = PngInfo()
                if cls.hidden.prompt is not None:
                    metadata.add_text("prompt", json.dumps(cls.hidden.prompt))
                if cls.hidden.extra_pnginfo is not None:
                    for x in cls.hidden.extra_pnginfo:
                        metadata.add_text(x, json.dumps(cls.hidden.extra_pnginfo[x]))

            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            file = f"{filename_with_batch_num}_{counter:05}_.png"
            img.save(os.path.join(full_output_folder, file), pnginfo=metadata, compress_level=compress_level)
            results.append(SavedResult(file, subfolder, type))
            counter += 1
        
        self.values = results
        self.animated = animated
    
    def as_dict(self):
        values = [x.as_dict() if isinstance(x, SavedResult) else x for x in self.values]
        return {
            "images": values,
            "animated": (self.animated,)
        }

class PreviewMask(PreviewImage):
    def __init__(self, mask: PreviewMask.Type, animated: bool=False, cls: ComfyNodeV3=None, **kwargs):
        preview = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3)
        super().__init__(preview, animated, cls, **kwargs)

# class UILatent(_UIOutput):
#     def __init__(self, values: list[SavedResult | dict], **kwargs):
#         output_dir = folder_paths.get_temp_directory()
#         type = "temp"
#         prefix_append = "_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))
#         compress_level = 1
#         filename_prefix = "ComfyUI"


#         full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir)

#         # support save metadata for latent sharing
#         prompt_info = ""
#         if prompt is not None:
#             prompt_info = json.dumps(prompt)

#         metadata = None
#         if not args.disable_metadata:
#             metadata = {"prompt": prompt_info}
#             if extra_pnginfo is not None:
#                 for x in extra_pnginfo:
#                     metadata[x] = json.dumps(extra_pnginfo[x])

#         file = f"{filename}_{counter:05}_.latent"

#         results: list[FileLocator] = []
#         results.append({
#             "filename": file,
#             "subfolder": subfolder,
#             "type": "output"
#         })

#         file = os.path.join(full_output_folder, file)

#         output = {}
#         output["latent_tensor"] = samples["samples"].contiguous()
#         output["latent_format_version_0"] = torch.tensor([])

#         comfy.utils.save_torch_file(output, file, metadata=metadata)

#         self.values = values
    
#     def as_dict(self):
#         values = [x.as_dict() if isinstance(x, SavedResult) else x for x in self.values]
#         return {
#             "latents": values,
#         }

class PreviewAudio(_UIOutput):
    def __init__(self, values: list[SavedResult | dict], **kwargs):
        self.values = values
    
    def as_dict(self):
        values = [x.as_dict() if isinstance(x, SavedResult) else x for x in self.values]
        return {
            "audio": values,
        }

class PreviewUI3D(_UIOutput):
    def __init__(self, values: list[SavedResult | dict], **kwargs):
        self.values = values
    
    def as_dict(self):
        values = [x.as_dict() if isinstance(x, SavedResult) else x for x in self.values]
        return {
            "3d": values,
        }

class PreviewText(_UIOutput):
    def __init__(self, value: str, **kwargs):
        self.value = value
    
    def as_dict(self):
        return {"text": (self.value,)}
