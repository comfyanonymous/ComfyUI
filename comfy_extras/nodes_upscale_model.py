import os
from comfy_extras.chainner_models import model_loading
from comfy.sd import load_torch_file
import model_management
from nodes import filter_files_extensions, recursive_search, supported_ckpt_extensions
import torch
import comfy.utils

class UpscaleModelLoader:
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "models")
    upscale_model_dir = os.path.join(models_dir, "upscale_models")

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model_name": (filter_files_extensions(recursive_search(s.upscale_model_dir), supported_ckpt_extensions), ),
                             }}
    RETURN_TYPES = ("UPSCALE_MODEL",)
    FUNCTION = "load_model"

    CATEGORY = "loaders"

    def load_model(self, model_name):
        model_path = os.path.join(self.upscale_model_dir, model_name)
        sd = load_torch_file(model_path)
        out = model_loading.load_state_dict(sd).eval()
        return (out, )


class ImageUpscaleWithModel:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "upscale_model": ("UPSCALE_MODEL",),
                              "image": ("IMAGE",),
                              }}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"

    CATEGORY = "image/upscaling"

    def upscale(self, upscale_model, image):
        device = model_management.get_torch_device()
        upscale_model.to(device)
        in_img = image.movedim(-1,-3).to(device)
        s = comfy.utils.tiled_scale(in_img, lambda a: upscale_model(a), tile_x=128 + 64, tile_y=128 + 64, overlap = 8, upscale_amount=upscale_model.scale)
        upscale_model.cpu()
        s = torch.clamp(s.movedim(-3,-1), min=0, max=1.0)
        return (s,)

NODE_CLASS_MAPPINGS = {
    "UpscaleModelLoader": UpscaleModelLoader,
    "ImageUpscaleWithModel": ImageUpscaleWithModel
}
