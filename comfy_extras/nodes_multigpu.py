from __future__ import annotations
from inspect import cleandoc

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from comfy.model_patcher import ModelPatcher
import comfy.multigpu


class MultiGPUWorkUnitsNode:
    """
    Prepares model to have sampling accelerated via splitting work units.

    Should be placed after nodes that modify the model object itself, such as compile or attention-switch nodes.

    Other than those exceptions, this node can be placed in any order.
    """

    NodeId = "MultiGPU_WorkUnits"
    NodeName = "MultiGPU Work Units"
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "max_gpus" : ("INT", {"default": 8, "min": 1, "step": 1}),
            },
            "optional": {
                "gpu_options": ("GPU_OPTIONS",)
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "init_multigpu"
    CATEGORY = "advanced/multigpu"
    DESCRIPTION = cleandoc(__doc__)

    def init_multigpu(self, model: ModelPatcher, max_gpus: int, gpu_options: comfy.multigpu.GPUOptionsGroup=None):
        model = comfy.multigpu.create_multigpu_deepclones(model, max_gpus, gpu_options, reuse_loaded=True)
        return (model,)

class MultiGPUOptionsNode:
    """
    Select the relative speed of GPUs in the special case they have significantly different performance from one another.
    """

    NodeId = "MultiGPU_Options"
    NodeName = "MultiGPU Options"
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "device_index": ("INT", {"default": 0, "min": 0, "max": 64}),
                "relative_speed": ("FLOAT", {"default": 1.0, "min": 0.0, "step": 0.01})
            },
            "optional": {
                "gpu_options": ("GPU_OPTIONS",)
            }
        }

    RETURN_TYPES = ("GPU_OPTIONS",)
    FUNCTION = "create_gpu_options"
    CATEGORY = "advanced/multigpu"
    DESCRIPTION = cleandoc(__doc__)

    def create_gpu_options(self, device_index: int, relative_speed: float, gpu_options: comfy.multigpu.GPUOptionsGroup=None):
        if not gpu_options:
            gpu_options = comfy.multigpu.GPUOptionsGroup()
        gpu_options.clone()

        opt = comfy.multigpu.GPUOptions(device_index=device_index, relative_speed=relative_speed)
        gpu_options.add(opt)

        return (gpu_options,)


node_list = [
    MultiGPUWorkUnitsNode,
    MultiGPUOptionsNode
]
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

for node in node_list:
    NODE_CLASS_MAPPINGS[node.NodeId] = node
    NODE_DISPLAY_NAME_MAPPINGS[node.NodeId] = node.NodeName
