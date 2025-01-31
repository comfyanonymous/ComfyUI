import nodes
import folder_paths
import os

def normalize_path(path):
    return path.replace('\\', '/')

class Load3D():
    @classmethod
    def INPUT_TYPES(s):
        input_dir = os.path.join(folder_paths.get_input_directory(), "3d")

        os.makedirs(input_dir, exist_ok=True)

        files = [normalize_path(os.path.join("3d", f)) for f in os.listdir(input_dir) if f.endswith(('.gltf', '.glb', '.obj', '.mtl', '.fbx', '.stl'))]

        return {"required": {
            "model_file": (sorted(files), {"file_upload": True}),
            "image": ("LOAD_3D", {}),
            "width": ("INT", {"default": 1024, "min": 1, "max": 4096, "step": 1}),
            "height": ("INT", {"default": 1024, "min": 1, "max": 4096, "step": 1}),
            "material": (["original", "normal", "wireframe", "depth"],),
            "light_intensity": ("INT", {"default": 10, "min": 1, "max": 20, "step": 1}),
            "up_direction": (["original", "-x", "+x", "-y", "+y", "-z", "+z"],),
            "fov": ("INT", {"default": 75, "min": 10, "max": 150, "step": 1}),
        }}

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("image", "mask", "mesh_path")

    FUNCTION = "process"
    EXPERIMENTAL = True

    CATEGORY = "3d"

    def process(self, model_file, image, **kwargs):
        if isinstance(image, dict):
            image_path = folder_paths.get_annotated_filepath(image['image'])
            mask_path = folder_paths.get_annotated_filepath(image['mask'])

            load_image_node = nodes.LoadImage()
            output_image, ignore_mask = load_image_node.load_image(image=image_path)
            ignore_image, output_mask = load_image_node.load_image(image=mask_path)

            return output_image, output_mask, model_file,
        else:
            # to avoid the format is not dict which will happen the FE code is not compatibility to core,
            # we need to this to double-check, it can be removed after merged FE into the core
            image_path = folder_paths.get_annotated_filepath(image)
            load_image_node = nodes.LoadImage()
            output_image, output_mask = load_image_node.load_image(image=image_path)
            return output_image, output_mask, model_file,

class Load3DAnimation():
    @classmethod
    def INPUT_TYPES(s):
        input_dir = os.path.join(folder_paths.get_input_directory(), "3d")

        os.makedirs(input_dir, exist_ok=True)

        files = [normalize_path(os.path.join("3d", f)) for f in os.listdir(input_dir) if f.endswith(('.gltf', '.glb', '.fbx'))]

        return {"required": {
            "model_file": (sorted(files), {"file_upload": True}),
            "image": ("LOAD_3D_ANIMATION", {}),
            "width": ("INT", {"default": 1024, "min": 1, "max": 4096, "step": 1}),
            "height": ("INT", {"default": 1024, "min": 1, "max": 4096, "step": 1}),
            "material": (["original", "normal", "wireframe", "depth"],),
            "light_intensity": ("INT", {"default": 10, "min": 1, "max": 20, "step": 1}),
            "up_direction": (["original", "-x", "+x", "-y", "+y", "-z", "+z"],),
            "fov": ("INT", {"default": 75, "min": 10, "max": 150, "step": 1}),
        }}

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("image", "mask", "mesh_path")

    FUNCTION = "process"
    EXPERIMENTAL = True

    CATEGORY = "3d"

    def process(self, model_file, image, **kwargs):
        if isinstance(image, dict):
            image_path = folder_paths.get_annotated_filepath(image['image'])
            mask_path = folder_paths.get_annotated_filepath(image['mask'])

            load_image_node = nodes.LoadImage()
            output_image, ignore_mask = load_image_node.load_image(image=image_path)
            ignore_image, output_mask = load_image_node.load_image(image=mask_path)

            return output_image, output_mask, model_file,
        else:
            image_path = folder_paths.get_annotated_filepath(image)
            load_image_node = nodes.LoadImage()
            output_image, output_mask = load_image_node.load_image(image=image_path)
            return output_image, output_mask, model_file,

class Preview3D():
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model_file": ("STRING", {"default": "", "multiline": False}),
            "material": (["original", "normal", "wireframe", "depth"],),
            "light_intensity": ("INT", {"default": 10, "min": 1, "max": 20, "step": 1}),
            "up_direction": (["original", "-x", "+x", "-y", "+y", "-z", "+z"],),
            "fov": ("INT", {"default": 75, "min": 10, "max": 150, "step": 1}),
        }}

    OUTPUT_NODE = True
    RETURN_TYPES = ()

    CATEGORY = "3d"

    FUNCTION = "process"
    EXPERIMENTAL = True

    def process(self, model_file, **kwargs):
        return {"ui": {"model_file": [model_file]}, "result": ()}

class Preview3DAnimation():
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model_file": ("STRING", {"default": "", "multiline": False}),
            "material": (["original", "normal", "wireframe", "depth"],),
            "light_intensity": ("INT", {"default": 10, "min": 1, "max": 20, "step": 1}),
            "up_direction": (["original", "-x", "+x", "-y", "+y", "-z", "+z"],),
            "fov": ("INT", {"default": 75, "min": 10, "max": 150, "step": 1}),
        }}

    OUTPUT_NODE = True
    RETURN_TYPES = ()

    CATEGORY = "3d"

    FUNCTION = "process"
    EXPERIMENTAL = True

    def process(self, model_file, **kwargs):
        return {"ui": {"model_file": [model_file]}, "result": ()}

NODE_CLASS_MAPPINGS = {
    "Load3D": Load3D,
    "Load3DAnimation": Load3DAnimation,
    "Preview3D": Preview3D,
    "Preview3DAnimation": Preview3DAnimation
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Load3D": "Load 3D",
    "Load3DAnimation": "Load 3D - Animation",
    "Preview3D": "Preview 3D",
    "Preview3DAnimation": "Preview 3D - Animation"
}
