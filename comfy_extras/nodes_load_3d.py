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
            "show_grid": ([True, False],),
            "camera_type": (["perspective", "orthographic"],),
            "view": (["front", "right", "top", "isometric"],),
            "material": (["original", "normal", "wireframe", "depth"],),
            "bg_color": ("STRING", {"default": "#000000", "multiline": False}),
            "light_intensity": ("INT", {"default": 10, "min": 1, "max": 20, "step": 1}),
            "up_direction": (["original", "-x", "+x", "-y", "+y", "-z", "+z"],),
        }}

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("image", "mask", "mesh_path")

    FUNCTION = "process"
    EXPERIMENTAL = True

    CATEGORY = "3d"

    def process(self, model_file, image, **kwargs):
        imagepath = folder_paths.get_annotated_filepath(image)

        load_image_node = nodes.LoadImage()

        output_image, output_mask = load_image_node.load_image(image=imagepath)

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
            "show_grid": ([True, False],),
            "camera_type": (["perspective", "orthographic"],),
            "view": (["front", "right", "top", "isometric"],),
            "material": (["original", "normal", "wireframe", "depth"],),
            "bg_color": ("STRING", {"default": "#000000", "multiline": False}),
            "light_intensity": ("INT", {"default": 10, "min": 1, "max": 20, "step": 1}),
            "up_direction": (["original", "-x", "+x", "-y", "+y", "-z", "+z"],),
            "animation_speed": (["0.1", "0.5", "1", "1.5", "2"], {"default": "1"}),
        }}

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("image", "mask", "mesh_path")

    FUNCTION = "process"
    EXPERIMENTAL = True

    CATEGORY = "3d"

    def process(self, model_file, image, **kwargs):
        imagepath = folder_paths.get_annotated_filepath(image)

        load_image_node = nodes.LoadImage()

        output_image, output_mask = load_image_node.load_image(image=imagepath)

        return output_image, output_mask, model_file,

class Preview3D():
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model_file": ("STRING", {"default": "", "multiline": False}),
            "show_grid": ([True, False],),
            "camera_type": (["perspective", "orthographic"],),
            "view": (["front", "right", "top", "isometric"],),
            "material": (["original", "normal", "wireframe", "depth"],),
            "bg_color": ("STRING", {"default": "#000000", "multiline": False}),
            "light_intensity": ("INT", {"default": 10, "min": 1, "max": 20, "step": 1}),
            "up_direction": (["original", "-x", "+x", "-y", "+y", "-z", "+z"],),
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
    "Preview3D": Preview3D
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Load3D": "Load 3D",
    "Load3DAnimation": "Load 3D - Animation",
    "Preview3D": "Preview 3D"
}