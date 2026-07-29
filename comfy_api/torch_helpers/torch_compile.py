from __future__ import annotations
import logging
import sys
import torch

import comfy.utils
from comfy.patcher_extension import WrappersMP
from typing import TYPE_CHECKING, Any, Callable, Optional
if TYPE_CHECKING:
    from comfy.model_patcher import ModelPatcher
    from comfy.patcher_extension import WrapperExecutor


COMPILE_KEY = "torch.compile"
TORCH_COMPILE_KWARGS = "torch_compile_kwargs"
WINDOWS_ROCM_INDUCTOR_OPTIONS = {
    "triton.cudagraphs": False,
    "triton.cudagraph_trees": False,
}


def _is_windows_rocm_inductor(backend: Optional[str]) -> bool:
    return backend == "inductor" and sys.platform == "win32" and getattr(torch.version, "hip", None) is not None


def normalize_torch_compile_kwargs(compile_kwargs: dict[str, Any]) -> dict[str, Any]:
    compile_kwargs = dict(compile_kwargs)
    if _is_windows_rocm_inductor(compile_kwargs.get("backend")) and compile_kwargs.get("mode") in (None, "", "default"):
        options = dict(compile_kwargs.get("options") or {})
        if set(options) <= {"guard_filter_fn"}:
            compile_kwargs["mode"] = None
            compile_kwargs["options"] = None
            logging.info("torch.compile: using default mode for Windows ROCm inductor.")
        else:
            changed = False
            for key, value in WINDOWS_ROCM_INDUCTOR_OPTIONS.items():
                if options.get(key) is not value:
                    options[key] = value
                    changed = True
            compile_kwargs["options"] = options
            compile_kwargs["mode"] = None
            if changed:
                logging.info("torch.compile: disabled inductor cudagraphs for Windows ROCm.")
    return compile_kwargs


def apply_torch_compile_factory(compiled_module_dict: dict[str, Callable]) -> Callable:
    '''
    Create a wrapper that will refer to the compiled_diffusion_model.
    '''
    def apply_torch_compile_wrapper(executor: WrapperExecutor, *args, **kwargs):
        try:
            orig_modules = {}
            for key, value in compiled_module_dict.items():
                orig_modules[key] = comfy.utils.get_attr(executor.class_obj, key)
                comfy.utils.set_attr(executor.class_obj, key, value)
            return executor(*args, **kwargs)
        finally:
            for key, value in orig_modules.items():
                comfy.utils.set_attr(executor.class_obj, key, value)
    return apply_torch_compile_wrapper


def set_torch_compile_wrapper(model: ModelPatcher, backend: str, options: Optional[dict[str, Any]]=None,
                              mode: Optional[str]=None, fullgraph=False, dynamic: Optional[bool]=None,
                              keys: list[str]=["diffusion_model"], *args, **kwargs):
    '''
    Perform torch.compile that will be applied at sample time for either the whole model or specific params of the BaseModel instance.

    When keys is None, it will default to using ["diffusion_model"], compiling the whole diffusion_model.
    When a list of keys is provided, it will perform torch.compile on only the selected modules.
    '''
    # clear out any other torch.compile wrappers
    model.remove_wrappers_with_key(WrappersMP.APPLY_MODEL, COMPILE_KEY)
    # if no keys, default to 'diffusion_model'
    if not keys:
        keys = ["diffusion_model"]
    # create kwargs dict that can be referenced later
    compile_kwargs = {
        "backend": backend,
        "options": options,
        "mode": mode,
        "fullgraph": fullgraph,
        "dynamic": dynamic,
    }
    compile_kwargs = normalize_torch_compile_kwargs(compile_kwargs)
    # get a dict of compiled keys
    compiled_modules = {}
    for key in keys:
        compiled_modules[key] = torch.compile(
                model=model.get_model_object(key),
                **compile_kwargs,
            )
    # add torch.compile wrapper
    wrapper_func = apply_torch_compile_factory(
        compiled_module_dict=compiled_modules,
    )
    # store wrapper to run on BaseModel's apply_model function
    model.add_wrapper_with_key(WrappersMP.APPLY_MODEL, COMPILE_KEY, wrapper_func)
    # keep compile kwargs for reference
    model.model_options[TORCH_COMPILE_KWARGS] = compile_kwargs
