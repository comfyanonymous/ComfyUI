from __future__ import annotations

from comfy_api.v3.io import Image, FolderType, _UIOutput, ComfyNodeV3
# used for image preview
from comfy.cli_args import args
import folder_paths
import random
from PIL import Image as PILImage
from PIL.PngImagePlugin import PngInfo
import os
import json
import numpy as np


class SavedResult(dict):
    def __init__(self, filename: str, subfolder: str, type: FolderType):
        super().__init__(filename=filename, subfolder=subfolder,type=type.value)

    @property
    def filename(self) -> str:
        return self["filename"]

    @property
    def subfolder(self) -> str:
        return self["subfolder"]

    @property
    def type(self) -> FolderType:
        return FolderType(self["type"])


class PreviewImage(_UIOutput):
    def __init__(self, image: Image.Type, animated: bool=False, cls: ComfyNodeV3=None, **kwargs):
        output_dir = folder_paths.get_temp_directory()
        prefix_append = "_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))
        filename_prefix = "ComfyUI" + prefix_append

        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, output_dir, image[0].shape[1], image[0].shape[0])
        results = list()
        for (batch_number, image) in enumerate(image):
            img = PILImage.fromarray(np.clip(255. * image.cpu().numpy(), 0, 255).astype(np.uint8))
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
            img.save(os.path.join(full_output_folder, file), pnginfo=metadata, compress_level=1)
            results.append(SavedResult(file, subfolder, FolderType.temp))
            counter += 1

        self.values = results
        self.animated = animated

    def as_dict(self):
        return {
            "images": self.values,
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
#         return {
#             "latents": self.values,
#         }

class PreviewAudio(_UIOutput):
    def __init__(self, values: list[SavedResult | dict], **kwargs):
        self.values = values

    def as_dict(self):
        return {"audio": self.values}

class PreviewUI3D(_UIOutput):
    def __init__(self, values: list[SavedResult | dict], **kwargs):
        self.values = values

    def as_dict(self):
        return {"3d": self.values}

class PreviewText(_UIOutput):
    def __init__(self, value: str, **kwargs):
        self.value = value

    def as_dict(self):
        return {"text": (self.value,)}
