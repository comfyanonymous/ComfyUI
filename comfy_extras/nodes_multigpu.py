from __future__ import annotations
import torch

from comfy.model_patcher import ModelPatcher
import comfy.utils
import comfy.patcher_extension
import comfy.model_management


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

    def init_multigpu(self, model: ModelPatcher, max_gpus: int, gpu_options: GPUOptionsGroup=None):
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
                gpu_options = GPUOptionsGroup()
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

    def create_gpu_options(self, device_index: int, relative_speed: float, gpu_options: GPUOptionsGroup=None):
        if not gpu_options:
            gpu_options = GPUOptionsGroup()
        gpu_options.clone()

        opt = GPUOptions(device_index=device_index, relative_speed=relative_speed)
        gpu_options.add(opt)

        return (gpu_options,)


class GPUOptions:
    def __init__(self, device_index: int, relative_speed: float):
        self.device_index = device_index
        self.relative_speed = relative_speed

    def clone(self):
        return GPUOptions(self.device_index, self.relative_speed)
    
    def create_dict(self):
        return {
            "relative_speed": self.relative_speed
        }

class GPUOptionsGroup:
    def __init__(self):
        self.options: dict[int, GPUOptions] = {}

    def add(self, info: GPUOptions):
        self.options[info.device_index] = info

    def clone(self):
        c = GPUOptionsGroup()
        for opt in self.options.values():
            c.add(opt)
        return c

    def register(self, model: ModelPatcher):
        opts_dict = {}
        # get devices that are valid for this model
        devices: list[torch.device] = [model.load_device]
        for extra_model in model.get_additional_models_with_key("multigpu"):
            extra_model: ModelPatcher
            devices.append(extra_model.load_device)
        # create dictionary with actual device mapped to its GPUOptions
        device_opts_list: list[GPUOptions] = []
        for device in devices:
            device_opts = self.options.get(device.index, GPUOptions(device_index=device.index, relative_speed=1.0))
            opts_dict[device] = device_opts.create_dict()
            device_opts_list.append(device_opts)
        # make relative_speed relative to 1.0
        max_speed = max([x.relative_speed for x in device_opts_list])
        for value in opts_dict.values():
            value["relative_speed"] /= max_speed
        model.model_options["multigpu_options"] = opts_dict


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
