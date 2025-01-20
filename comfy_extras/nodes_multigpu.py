from comfy.model_patcher import ModelPatcher
import comfy.utils
import comfy.patcher_extension
import comfy.model_management
import copy


class MultiGPUInitialize:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
            },
            "optional": {
                "max_gpus" : ("INT", {"default": 8, "min": 1, "step": 1}),
            }
        }
    
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "init_multigpu"
    CATEGORY = "DevTools"

    def init_multigpu(self, model: ModelPatcher, max_gpus: int):
        extra_devices = comfy.model_management.get_all_torch_devices(exclude_current=True)
        extra_devices = extra_devices[:max_gpus-1]
        if len(extra_devices) > 0:
            model = model.clone()
            comfy.model_management.unload_all_models()
            for device in extra_devices:
                device_patcher = model.multigpu_clone(new_load_device=device)
                device_patcher.is_multigpu_clone = True
                multigpu_models = model.get_additional_models_with_key("multigpu")
                multigpu_models.append(device_patcher)
                model.set_additional_models("multigpu", multigpu_models)
        return (model,)
    

NODE_CLASS_MAPPINGS = {
    "test_multigpuinit": MultiGPUInitialize,
}