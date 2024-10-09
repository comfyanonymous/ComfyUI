import logging

import torch
from torch.nn import LayerNorm

from comfy import model_management
from comfy.model_patcher import ModelPatcher
from comfy.nodes.package_typing import CustomNode, InputTypes

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


class QuantizeModel(CustomNode):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypes:
        return {
            "required": {
                "model": ("MODEL", {}),
                "strategy": (["torchao", "quanto"], {"default": "torchao"})
            }
        }

    FUNCTION = "execute"
    CATEGORY = "_for_testing"
    EXPERIMENTAL = True

    RETURN_TYPES = ("MODEL",)

    def execute(self, model: ModelPatcher, strategy: str = "torchao"):
        logging.warning(f"Quantizing {model} this way quantizes it in place, making it insuitable for cloning. All uses of this model will be quantized.")
        logging.warning(f"Quantizing {model} will produce poor results due to Optimum's limitations")
        model = model.clone()
        unet = model.get_model_object("diffusion_model")
        # todo: quantize quantizes in place, which is not desired

        # default exclusions
        _unused_exclusions = {
            "time_embedding.",
            "add_embedding.",
            "time_in.",
            "txt_in.",
            "vector_in.",
            "img_in.",
            "guidance_in.",
            "final_layer.",
        }
        if strategy == "quanto":
            from optimum.quanto import quantize, qint8
            exclusion_list = [
                name for name, module in unet.named_modules() if isinstance(module, LayerNorm) and module.weight is None
            ]
            quantize(unet, weights=qint8, activations=qint8, exclude=exclusion_list)
            _in_place_fixme = unet
        elif strategy == "torchao":
            from torchao.quantization import quantize_, int8_dynamic_activation_int8_weight
            model = model.clone()
            unet = model.get_model_object("diffusion_model")
            # todo: quantize quantizes in place, which is not desired

            # def filter_fn(module: torch.nn.Module, name: str):
            #     return any("weight" in name for name, _ in (module.named_parameters())) and all(exclusion not in name for exclusion in exclusions)
            quantize_(unet, int8_dynamic_activation_int8_weight(), device=model_management.get_torch_device())
            _in_place_fixme = unet
        else:
            raise ValueError(f"unknown strategy {strategy}")

        model.add_object_patch("diffusion_model", _in_place_fixme)
        return model,


NODE_CLASS_MAPPINGS = {
    "TorchCompileModel": TorchCompileModel,
    "QuantizeModel": QuantizeModel,
}
