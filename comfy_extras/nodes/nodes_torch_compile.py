import logging

import torch

from comfy.model_patcher import ModelPatcher

DIFFUSION_MODEL = "diffusion_model"


class TorchCompileModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
            },
            "optional": {
                "object_patch": ("STRING", {"default": DIFFUSION_MODEL}),
                "fullgraph": ("BOOLEAN", {"default": False}),
                "dynamic": ("BOOLEAN", {"default": False}),
                "backend": ("STRING", {"default": "inductor"}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "_for_testing"
    EXPERIMENTAL = True

    def patch(self, model: ModelPatcher, object_patch: str | None = DIFFUSION_MODEL, fullgraph: bool = False, dynamic: bool = False, backend: str = "inductor"):
        if object_patch is None:
            object_patch = DIFFUSION_MODEL
        compile_kwargs = {
            "fullgraph": fullgraph,
            "dynamic": dynamic,
            "backend": backend
        }
        if isinstance(model, ModelPatcher):
            m = model.clone()
            m.add_object_patch(object_patch, torch.compile(model=m.get_model_object(object_patch), **compile_kwargs))
            return (m,)
        elif isinstance(model, torch.nn.Module):
            return torch.compile(model=model, **compile_kwargs),
        else:
            logging.warning("Encountered a model that cannot be compiled")
            return model,


NODE_CLASS_MAPPINGS = {
    "TorchCompileModel": TorchCompileModel,
}
