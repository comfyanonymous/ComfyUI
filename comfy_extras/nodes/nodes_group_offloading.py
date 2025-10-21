import torch
import logging
from diffusers import HookRegistry
from diffusers.hooks import apply_group_offloading, apply_layerwise_casting, ModelHook

from comfy.language.transformers_model_management import TransformersManagedModel
from comfy.model_management import vram_state, VRAMState, unload_all_models, get_free_memory, get_torch_device
from comfy.model_management_types import HooksSupport, ModelManageable
from comfy.model_patcher import ModelPatcher
from comfy.node_helpers import export_custom_nodes
from comfy.nodes.package_typing import CustomNode
from comfy.ops import manual_cast
from comfy.patcher_extension import WrappersMP
from comfy.rmsnorm import RMSNorm

_DISABLE_COMFYUI_CASTING_HOOK = "disable_comfyui_casting_hook"

logger = logging.getLogger(__name__)


class DisableComfyWeightCast(ModelHook):
    r"""
    A hook that casts the weights of a module to a high precision dtype for computation, and to a low precision dtype
    for storage. This process may lead to quality loss in the output, but can significantly reduce the memory
    footprint.
    """

    _is_stateful = False

    def __init__(self) -> None:
        super().__init__()

    def initialize_hook(self, module: torch.nn.Module):
        if hasattr(module, "comfy_cast_weights"):
            module.comfy_cast_weights = False
        return module

    def deinitalize_hook(self, module: torch.nn.Module):
        if hasattr(module, "comfy_cast_weights"):
            module.comfy_cast_weights = True
        return module


def disable_comfyui_weight_casting_hook(module: torch.nn.Module):
    registry = HookRegistry.check_if_exists_or_initialize(module)
    hook = DisableComfyWeightCast()
    registry.register_hook(hook, _DISABLE_COMFYUI_CASTING_HOOK)


def disable_comfyui_weight_casting(module: torch.nn.Module):
    types = [
        torch.nn.Linear,
        torch.nn.Conv1d,
        torch.nn.Conv2d,
        torch.nn.Conv3d,
        torch.nn.GroupNorm,
        torch.nn.LayerNorm,
        RMSNorm,
        torch.nn.ConvTranspose2d,
        torch.nn.ConvTranspose1d,
        torch.nn.Embedding
    ]
    try:
        from torch.nn import RMSNorm as TorchRMSNorm  # pylint: disable=no-member
        types.append(TorchRMSNorm)
    except (ImportError, ModuleNotFoundError):
        pass

    if isinstance(module, tuple(types)):
        disable_comfyui_weight_casting_hook(module)
        return

    for name, submodule in module.named_children():
        disable_comfyui_weight_casting(submodule)


def prepare_group_offloading_factory(load_device: torch.device, offload_device: torch.device):
    def wrapper(executor, model: ModelPatcher, *args, **kwargs):
        # this model will now just be loaded to CPU, since diffusers will manage moving to gpu
        model.load_device = offload_device

        # we'll have to unload everything to use pinning better, this includes trimming
        unload_all_models()

        # loads the model, prepares everything
        inner_model, conds, models = executor(model, *args, **kwargs)

        # we will need layer casting from diffusers in this situation
        if model.model.operations == manual_cast and model.diffusion_model.dtype != model.model.manual_cast_dtype:
            raise ValueError("manual casting operations, where the model is loaded in different weights than inference will occur, is not supported")

        # weights are patched, ready to go, inner model will be correctly deleted at the end of sampling
        model_size = model.model_size()

        model_too_large = model_size * 2 > get_free_memory(torch.cpu)
        low_vram_state = vram_state in (VRAMState.LOW_VRAM,)
        is_cuda_device = load_device.type == 'cuda'

        if model_too_large or low_vram_state:
            logger.error(f"group offloading did not use memory pinning because model_too_large={model_too_large} low_vram_state={low_vram_state}")
        if not is_cuda_device:
            logger.error(f"group offloading did not use stream because load_device.type={load_device.type} != \"cuda\"")
        apply_group_offloading(
            inner_model.diffusion_model,
            load_device,
            offload_device,
            use_stream=is_cuda_device,
            record_stream=is_cuda_device,
            low_cpu_mem_usage=low_vram_state or model_too_large,
            num_blocks_per_group=1
        )
        # then the inputs will be ready on the correct device due to the wrapper factory
        model.load_device = load_device
        return inner_model, conds, models

    return wrapper


def prepare_layerwise_casting_factory(dtype: torch.dtype):
    def wrapper(executor, model: ModelPatcher, *args, **kwargs):
        disable_comfyui_weight_casting(model.diffusion_model)
        apply_layerwise_casting(model.diffusion_model,
                                dtype,
                                model.diffusion_model.dtype,
                                non_blocking=True)
        inner_model, conds, models = executor(model, *args, **kwargs)

        return inner_model, conds, models

    return wrapper


class GroupOffload(CustomNode):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {})
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "execute"

    def execute(self, model: ModelManageable | HooksSupport | TransformersManagedModel) -> tuple[ModelPatcher,]:
        if isinstance(model, ModelManageable):
            model = model.clone()
        if isinstance(model, TransformersManagedModel):
            apply_group_offloading(
                model.model,
                model.load_device,
                model.offload_device,
                use_stream=True,
                record_stream=True,
                low_cpu_mem_usage=vram_state in (VRAMState.LOW_VRAM,),
                num_blocks_per_group=1
            )
        elif isinstance(model, HooksSupport) and isinstance(model, ModelManageable):
            model.add_wrapper_with_key(WrappersMP.PREPARE_SAMPLING, "group_offload", prepare_group_offloading_factory(model.load_device, model.offload_device))
        return model,


class LayerwiseCast(CustomNode):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {}),
                "dtype": (["float8_e4m3fn", "float8_e5m2"], {})
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "execute"

    def execute(self, model: ModelPatcher, dtype: str) -> tuple[ModelPatcher,]:
        model = model.clone()
        if dtype == "float8_e4m3fn":
            dtype = torch.float8_e4m3fn
        elif dtype == "float8_e5m2":
            dtype = torch.float8_e5m2

        model.add_wrapper(WrappersMP.PREPARE_SAMPLING, prepare_layerwise_casting_factory(dtype))
        return model,


export_custom_nodes()
