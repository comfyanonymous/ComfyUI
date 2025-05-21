from __future__ import annotations
import torch

from comfy.patcher_extension import WrappersMP
from typing import TYPE_CHECKING, Callable, Optional
if TYPE_CHECKING:
    from comfy.model_patcher import ModelPatcher
    from comfy.patcher_extension import WrapperExecutor


COMPILE_KEY = "torch.compile"


def apply_torch_compile_factory(compiled_diffusion_model: Callable) -> Callable:
    '''
    Create a wrapper that will refer to the compiled_diffusion_model
    '''
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
