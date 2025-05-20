from __future__ import annotations
import torch

from comfy.patcher_extension import WrappersMP, WrapperExecutor
from typing import TYPE_CHECKING, Callable, Optional
if TYPE_CHECKING:
    from comfy.model_patcher import ModelPatcher


COMPILE_KEY = "torch.compile"


def apply_torch_compile_factory(compiled_diffusion_model: Callable) -> Callable:
    def apply_torch_compile_wrapper(executor: WrapperExecutor, *args, **kwargs):
        try:
            orig_diffusion_model = executor.class_obj.diffusion_model
            executor.class_obj.diffusion_model = compiled_diffusion_model
            return executor(*args, **kwargs)
        finally:
            executor.class_obj.diffusion_model = orig_diffusion_model
    return apply_torch_compile_wrapper


def set_torch_compile_wrapper(model: ModelPatcher, backend: str, options: Optional[dict[str,str]]=None, *args, **kwargs):
    # clear out any other torch.compile wrappers
    model.remove_wrappers_with_key(WrappersMP.APPLY_MODEL, COMPILE_KEY)
    # add torch.compile wrapper
    wrapper_func = apply_torch_compile_factory(
        torch.compile(
                model=model.get_model_object("diffusion_model"),
                backend=backend,
                options=options,
            ),
        )
    model.add_wrapper_with_key(WrappersMP.APPLY_MODEL, COMPILE_KEY, wrapper_func)


class TorchCompileModel:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                             "backend": (["inductor", "cudagraphs"],),
                              }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "_for_testing"
    EXPERIMENTAL = True

    def patch(self, model, backend):
        m = model.clone()
        set_torch_compile_wrapper(model=m, backend=backend)
        return (m, )

NODE_CLASS_MAPPINGS = {
    "TorchCompileModel": TorchCompileModel,
}
