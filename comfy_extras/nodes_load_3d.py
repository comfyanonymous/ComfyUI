import nodes
import folder_paths
import os

from comfy.comfy_types import IO
from comfy_api.input_impl import VideoFromFile

from pathlib import Path

from PIL import Image
import numpy as np

import uuid

def normalize_path(path):
    return path.replace('\\', '/')

class Load3D():
    @classmethod
    def INPUT_TYPES(s):
        input_dir = os.path.join(folder_paths.get_input_directory(), "3d")

        os.makedirs(input_dir, exist_ok=True)

        input_path = Path(input_dir)
        base_path = Path(folder_paths.get_input_directory())

        files = [
            normalize_path(str(file_path.relative_to(base_path)))
            for file_path in input_path.rglob("*")
            if file_path.suffix.lower() in {'.gltf', '.glb', '.obj', '.fbx', '.stl'}
        ]

        return {"required": {
            "model_file": (sorted(files), {"file_upload": True}),
            "image": ("LOAD_3D", {}),
            "width": ("INT", {"default": 1024, "min": 1, "max": 4096, "step": 1}),
            "height": ("INT", {"default": 1024, "min": 1, "max": 4096, "step": 1}),
        }}

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "IMAGE", "LOAD3D_CAMERA", IO.VIDEO)
    RETURN_NAMES = ("image", "mask", "mesh_path", "normal", "camera_info", "recording_video")

    FUNCTION = "process"
    EXPERIMENTAL = True

    CATEGORY = "3d"

    def process(self, model_file, image, **kwargs):
        image_path = folder_paths.get_annotated_filepath(image['image'])
        mask_path = folder_paths.get_annotated_filepath(image['mask'])
        normal_path = folder_paths.get_annotated_filepath(image['normal'])

        load_image_node = nodes.LoadImage()
        output_image, ignore_mask = load_image_node.load_image(image=image_path)
        ignore_image, output_mask = load_image_node.load_image(image=mask_path)
        normal_image, ignore_mask2 = load_image_node.load_image(image=normal_path)

        video = None

        if image['recording'] != "":
            recording_video_path = folder_paths.get_annotated_filepath(image['recording'])

            video = VideoFromFile(recording_video_path)

        return output_image, output_mask, model_file, normal_image, image['camera_info'], video

class Preview3D():
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model_file": ("STRING", {"default": "", "multiline": False}),
        },
        "optional": {
            "camera_info": ("LOAD3D_CAMERA", {}),
            "bg_image": ("IMAGE", {})
        }}

    OUTPUT_NODE = True
    RETURN_TYPES = ()

    CATEGORY = "3d"

    FUNCTION = "process"
    EXPERIMENTAL = True

    def process(self, model_file, **kwargs):
        camera_info = kwargs.get("camera_info", None)
        bg_image = kwargs.get("bg_image", None)

        bg_image_path = None
        if bg_image is not None:

            img_array = (bg_image[0].cpu().numpy() * 255).astype(np.uint8)
            img = Image.fromarray(img_array)

            temp_dir = folder_paths.get_temp_directory()
            filename = f"bg_{uuid.uuid4().hex}.png"
            bg_image_path = os.path.join(temp_dir, filename)
            img.save(bg_image_path, compress_level=1)

            bg_image_path = f"temp/{filename}"

        return {
            "ui": {
                "result": [model_file, camera_info, bg_image_path]
            }
        }

NODE_CLASS_MAPPINGS = {
    "Load3D": Load3D,
    "Preview3D": Preview3D,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Load3D": "Load 3D & Animation",
    "Preview3D": "Preview 3D & Animation",
}
