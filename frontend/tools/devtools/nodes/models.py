from __future__ import annotations

import os

import torch

import comfy.utils as utils
from comfy.model_patcher import ModelPatcher
import nodes
import folder_paths


class DummyPatch(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, dummy_float: float = 0.0):
        super().__init__()
        self.module = module
        self.dummy_float = dummy_float

    def forward(self, *args, **kwargs):
        if isinstance(self.module, DummyPatch):
            raise Exception(f"Calling nested dummy patch! {self.dummy_float}")

        return self.module(*args, **kwargs)


class ObjectPatchNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "target_module": ("STRING", {"multiline": True}),
            },
            "optional": {
                "dummy_float": ("FLOAT", {"default": 0.0}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_patch"
    CATEGORY = "DevTools"
    DESCRIPTION = "A node that applies an object patch"

    def apply_patch(
        self, model: ModelPatcher, target_module: str, dummy_float: float = 0.0
    ) -> ModelPatcher:
        module = utils.get_attr(model.model, target_module)
        work_model = model.clone()
        work_model.add_object_patch(target_module, DummyPatch(module, dummy_float))
        return (work_model,)


class LoadAnimatedImageTest(nodes.LoadImage):
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [
            f
            for f in os.listdir(input_dir)
            if os.path.isfile(os.path.join(input_dir, f)) and f.endswith(".webp")
        ]
        files = folder_paths.filter_files_content_types(files, ["image"])
        return {
            "required": {"image": (sorted(files), {"animated_image_upload": True})},
        }


NODE_CLASS_MAPPINGS = {
    "DevToolsObjectPatchNode": ObjectPatchNode,
    "DevToolsLoadAnimatedImageTest": LoadAnimatedImageTest,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DevToolsObjectPatchNode": "Object Patch Node",
    "DevToolsLoadAnimatedImageTest": "Load Animated Image",
}

__all__ = [
    "DummyPatch",
    "ObjectPatchNode",
    "LoadAnimatedImageTest",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
