from __future__ import annotations

from comfy.model_patcher import ModelPatcher
import comfy.utils
import comfy.patcher_extension
import comfy.model_management
import comfy.multigpu


class MultiGPUInitialize:
    NodeId = "MultiGPU_Initialize"
    NodeName = "MultiGPU Initialize"
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

    def init_multigpu(self, model: ModelPatcher, max_gpus: int, gpu_options: comfy.multigpu.GPUOptionsGroup=None):
        extra_devices = comfy.model_management.get_all_torch_devices(exclude_current=True)
        extra_devices = extra_devices[:max_gpus-1]
        if len(extra_devices) > 0:
            model = model.clone()
            comfy.model_management.unload_model_and_clones(model)
            for device in extra_devices:
                device_patcher = model.multigpu_deepclone(new_load_device=device)
                device_patcher.is_multigpu_clone = True
                multigpu_models = model.get_additional_models_with_key("multigpu")
                multigpu_models.append(device_patcher)
                model.set_additional_models("multigpu", multigpu_models)
            if gpu_options is None:
                gpu_options = comfy.multigpu.GPUOptionsGroup()
            gpu_options.register(model)
        return (model,)

class MultiGPUOptionsNode:
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

    def create_gpu_options(self, device_index: int, relative_speed: float, gpu_options: comfy.multigpu.GPUOptionsGroup=None):
        if not gpu_options:
            gpu_options = comfy.multigpu.GPUOptionsGroup()
        gpu_options.clone()

        opt = comfy.multigpu.GPUOptions(device_index=device_index, relative_speed=relative_speed)
        gpu_options.add(opt)

        return (gpu_options,)


node_list = [
    MultiGPUInitialize,
    MultiGPUOptionsNode
]
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

for node in node_list:
    NODE_CLASS_MAPPINGS[node.NodeId] = node
    NODE_DISPLAY_NAME_MAPPINGS[node.NodeId] = node.NodeName

# TODO: remove
NODE_CLASS_MAPPINGS["test_multigpuinit"] = MultiGPUInitialize
