import os
import numpy as np
import folder_paths
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from comfy.cli_args import args
import json

class GetWorkflowMetadata:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "hidden": {
                "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }
    RETURN_TYPES = ("METADATA",)
    FUNCTION = "metaget"
    CATEGORY = "image"
    DESCRIPTION = "Gets the workflow metadata from the current workflow."
    def metaget(self, extra_pnginfo):
        metadata = PngInfo()
        if not args.disable_metadata:
            if extra_pnginfo is not None:
                for x in extra_pnginfo:
                    metadata.add_text(x, json.dumps(extra_pnginfo[x]))
        return (metadata,)

class EmptyMetadata:
    @classmethod
    def INPUT_TYPES(s):
        return {
        }
    RETURN_TYPES = ("METADATA",)
    FUNCTION = "metaget"
    CATEGORY = "image"
    DESCRIPTION = "Create a blank / empty metadata to add upon."
    def metaget(self):
        metadata = PngInfo()
        return (metadata,)

class AddMetadataValue:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "key": ("STRING", {"tooltip": "Key to save the value at."}),
                "value": ("STRING", {"tooltip": "What to add to the metadata."}),
                "metadata": ("METADATA", {"tooltip": "Metadata to add to."})
            }
        }
    RETURN_TYPES = ("METADATA",)
    FUNCTION = "metaadd"
    CATEGORY = "image"
    DESCRIPTION = "Add an arbitrary value to the metadata."
    def metaadd(self, key, value, metadata):
        metadata.add_text(key, json.dumps(value))
        return (metadata,)



class SaveImageCustomMetadata:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The images to save."}),
                "filename_prefix": ("STRING", {"default": "ComfyUI", "tooltip": "The prefix for the file to save. This may include formatting information such as %date:yyyy-MM-dd% or %Empty Latent Image.width% to include values from nodes."}),
                "metadata": ("METADATA", {"tooltip": "Metadata to save with the image"})
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"

    OUTPUT_NODE = True

    CATEGORY = "image"
    DESCRIPTION = "Saves the input images to your ComfyUI output directory with custom metadata."

    def save_images(self, images, filename_prefix="ComfyUI", metadata=None):
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])
        results = list()
        for (batch_number, image) in enumerate(images):
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            file = f"{filename_with_batch_num}_{counter:05}_.png"
            img.save(os.path.join(full_output_folder, file), pnginfo=metadata, compress_level=self.compress_level)
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })
            counter += 1

        return { "ui": { "images": results } }
    


NODE_CLASS_MAPPINGS = {
    "GetWorkflowMetadata": GetWorkflowMetadata,
    "EmptyMetadata": EmptyMetadata,
    "AddMetadataValue": AddMetadataValue,
    "SaveImageCustomMetadata": SaveImageCustomMetadata
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Get Workflow Metadata": "GetWorkflowMetadata",
    "Empty Metadata": "EmptyMetadata",
    "Add Metadata Value": "AddMetadataValue",
    "Save Image with custom metadata": "SaveImageCustomMetadata"
}