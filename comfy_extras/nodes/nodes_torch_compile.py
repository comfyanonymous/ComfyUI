import logging
import os
from pathlib import Path
from typing import Union, Callable

import torch
import torch._inductor.codecache
from torch.nn import LayerNorm

from comfy import model_management
from comfy.language.transformers_model_management import TransformersManagedModel
from comfy.model_patcher import ModelPatcher
from comfy.nodes.package_typing import CustomNode, InputTypes
from comfy.sd import VAE
from comfy_api.torch_helpers import set_torch_compile_wrapper

logger = logging.getLogger(__name__)

DIFFUSION_MODEL = "diffusion_model"
TORCH_COMPILE_BACKENDS = [
    "inductor",
    "torch_tensorrt",
    "onnxrt",
    "cudagraphs",
    "openxla",
    "tvm"
]

TORCH_COMPILE_MODES = [
    "default",
    "reduce-overhead",
    "max-autotune",
    "max-autotune-no-cudagraphs"
]

# fix torch bug on windows
_old_write_atomic = torch._inductor.codecache.write_atomic


def write_atomic(
        path: str, content: Union[str, bytes], make_dirs: bool = False
) -> None:
    if Path(path).exists():
        os.remove(path)
    _old_write_atomic(path, content, make_dirs=make_dirs)


torch._inductor.codecache.write_atomic = write_atomic
# torch._inductor.utils.is_big_gpu = lambda *args: True


class TorchCompileModel(CustomNode):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL,VAE",),
            },
            "optional": {
                "object_patch": ("STRING", {"default": ""}),
                "fullgraph": ("BOOLEAN", {"default": False}),
                "dynamic": ("BOOLEAN", {"default": False}),
                "backend": (TORCH_COMPILE_BACKENDS, {"default": "inductor"}),
                "mode": (TORCH_COMPILE_MODES, {"default": "max-autotune"}),
                "torch_tensorrt_optimization_level": ("INT", {"default": 3, "min": 1, "max": 5})
            }
        }

    RETURN_TYPES = ("MODEL,VAE",)
    FUNCTION = "patch"
    RETURN_NAMES = ("model or vae",)

    CATEGORY = "_for_testing"
    EXPERIMENTAL = True

    def patch(self, model: ModelPatcher | VAE | torch.nn.Module, object_patch: str | None = "", fullgraph: bool = False, dynamic: bool = False, backend: str = "inductor", mode: str = "max-autotune", torch_tensorrt_optimization_level: int = 3) -> tuple[Callable]:
        compile_kwargs = {
            "fullgraph": fullgraph,
            "dynamic": dynamic,
            "backend": backend,
            "mode": mode,
        }
        try:
            if backend == "torch_tensorrt":
                try:
                    import torch_tensorrt
                except (ImportError, ModuleNotFoundError):
                    logger.error(f"Install torch-tensorrt and modelopt")
                    raise
                compile_kwargs["options"] = {
                    # https://pytorch.org/TensorRT/dynamo/torch_compile.html
                    # Quantization/INT8 support is slated for a future release; currently, we support FP16 and FP32 precision layers.
                    "enabled_precisions": {torch.float, torch.half, torch.float8_e5m2},
                    "optimization_level": torch_tensorrt_optimization_level,
                    "cache_built_engines": True,
                    "reuse_cached_engines": True,
                    "enable_weight_streaming": True,
                    "make_refittable": True,
                }
                del compile_kwargs["mode"]
            if isinstance(model, (ModelPatcher, TransformersManagedModel, VAE)):
                to_return = model.clone()
                object_patches = [p.strip() for p in object_patch.split(",")]
                patcher: ModelPatcher
                if isinstance(to_return, VAE):
                    patcher = to_return.patcher
                    object_patches = ["encoder", "decoder"]
                else:
                    patcher = to_return
                if object_patch is None or len(object_patches) == 0 or len(object_patches) == 1 and object_patches[0].strip() == "":
                    object_patches = [DIFFUSION_MODEL]
                set_torch_compile_wrapper(patcher, keys=object_patches, **compile_kwargs)
                return to_return,
            elif isinstance(model, torch.nn.Module):
                model_management.unload_all_models()
                model.to(device=model_management.get_torch_device())
                res = torch.compile(model=model, **compile_kwargs),
                model.to(device=model_management.unet_offload_device())
                return res,
            else:
                logger.warning(f"Encountered a model {model} that cannot be compiled")
                return model,
        except OSError as os_error:
            try:
                torch._inductor.utils.clear_inductor_caches()  # pylint: disable=no-member
            except Exception:
                pass
            raise os_error
        except Exception as exc_info:
            try:
                torch._inductor.utils.clear_inductor_caches()  # pylint: disable=no-member
            except Exception:
                pass
            logger.error(f"An exception occurred while trying to compile {str(model)}, gracefully skipping compilation", exc_info=exc_info)
            return model,


_QUANTIZATION_STRATEGIES = [
    "torchao",
    "torchao-autoquant",
    "quanto",
]


class QuantizeModel(CustomNode):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypes:
        return {
            "required": {
                "model": ("MODEL", {}),
                "strategy": (_QUANTIZATION_STRATEGIES, {"default": _QUANTIZATION_STRATEGIES[0]})
            }
        }

    FUNCTION = "execute"
    CATEGORY = "_for_testing"
    EXPERIMENTAL = True

    RETURN_TYPES = ("MODEL",)

    def warn_in_place(self, model: ModelPatcher):
        logger.warning(f"Quantizing {model} this way quantizes it in place, making it insuitable for cloning. All uses of this model will be quantized.")

    def execute(self, model: ModelPatcher, strategy: str = _QUANTIZATION_STRATEGIES[0]) -> tuple[ModelPatcher]:
        model = model.clone()
        model.patch_model(force_patch_weights=True)
        unet = model.diffusion_model
        # todo: quantize quantizes in place, which is not desired

        # default exclusions
        always_exclude_these = {
            "time_embedding.",
            "add_embedding.",
            "time_in.in",
            "txt_in",
            "vector_in.in",
            "img_in",
            "guidance_in.in",
            "final_layer",
        }
        if strategy == "quanto":
            logger.warning(f"Quantizing {model} will produce poor results due to Optimum's limitations")
            self.warn_in_place(model)
            from optimum.quanto import quantize, qint8  # pylint: disable=import-error
            exclusion_list = [
                name for name, module in unet.named_modules() if isinstance(module, LayerNorm) and module.weight is None
            ]
            quantize(unet, weights=qint8, activations=qint8, exclude=exclusion_list)
            _in_place_fixme = unet
        elif "torchao" in strategy:
            from torchao.quantization import quantize_, int8_dynamic_activation_int8_weight, autoquant  # pylint: disable=import-error
            from torchao.utils import unwrap_tensor_subclass  # pylint: disable=import-error
            self.warn_in_place(model)
            model_management.load_models_gpu([model])

            def filter(module: torch.nn.Module, fqn: str) -> bool:
                return isinstance(module, torch.nn.Linear) and not any(prefix in fqn for prefix in always_exclude_these)

            if "autoquant" in strategy:
                _in_place_fixme = autoquant(unet, error_on_unseen=False)
            else:
                quantize_(unet, int8_dynamic_activation_int8_weight(), device=model_management.get_torch_device(), filter_fn=filter)
                _in_place_fixme = unet
            unwrap_tensor_subclass(_in_place_fixme)
        else:
            raise ValueError(f"unknown strategy {strategy}")

        model.add_object_patch("diffusion_model", _in_place_fixme)
        return model,


NODE_CLASS_MAPPINGS = {
    "TorchCompileModel": TorchCompileModel,
    "QuantizeModel": QuantizeModel,
}
