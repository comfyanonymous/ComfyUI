from __future__ import annotations

import importlib
import importlib.util
import inspect
import logging
import threading
from pathlib import Path
from types import ModuleType
from typing import Callable, Dict, Iterable, Tuple

logger = logging.getLogger("baymax-zimage")

_PATCH_LOCK = threading.Lock()
_ORIGINAL_FUNCTIONS: Dict[Tuple[str, str], Callable] = {}
_PACKAGE_DIR = Path(__file__).resolve().parent
_USER_IMPL_PATH = _PACKAGE_DIR / "user_impl.py"


def _available_targets() -> Tuple[Tuple[ModuleType, str], ...]:
    targets = []
    for module, attribute_name in _target_modules():
        if hasattr(module, attribute_name):
            targets.append((module, attribute_name))
        else:
            logger.warning(
                "[baymax-zimage] Skipping %s.%s because it does not exist",
                module.__name__,
                attribute_name,
            )
    return tuple(targets)


def _target_modules() -> Iterable[Tuple[ModuleType, str]]:
    return (
        (importlib.import_module("comfy.ldm.lumina.model"), "apply_rope"),
        (importlib.import_module("comfy.ldm.lumina.controlnet"), "apply_rope"),
    )


def _load_user_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("baymax_zimage_user_impl", _USER_IMPL_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load user implementation from {_USER_IMPL_PATH}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _call_user_impl(user_function: Callable, original_function: Callable, xq, xk, freqs_cis):
    signature = inspect.signature(user_function)
    parameters = signature.parameters.values()
    positional_slots = sum(
        1
        for parameter in parameters
        if parameter.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    )
    has_varargs = any(parameter.kind == inspect.Parameter.VAR_POSITIONAL for parameter in signature.parameters.values())
    has_varkw = any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values())

    if "original_apply_rope" in signature.parameters or has_varkw:
        return user_function(xq, xk, freqs_cis, original_apply_rope=original_function)

    if positional_slots >= 4 or has_varargs:
        return user_function(xq, xk, freqs_cis, original_function)

    return user_function(xq, xk, freqs_cis)


def _make_wrapper(module_name: str, user_function: Callable) -> Callable:
    original_function = _ORIGINAL_FUNCTIONS[(module_name, "apply_rope")]

    def patched_apply_rope(xq, xk, freqs_cis):
        return _call_user_impl(user_function, original_function, xq, xk, freqs_cis)

    patched_apply_rope.__name__ = "apply_rope"
    patched_apply_rope.__module__ = __name__
    return patched_apply_rope


def _patch_rotary(enable: bool, reload_user_impl: bool) -> str:
    del reload_user_impl  # The user module is reloaded on every execution.

    with _PATCH_LOCK:
        targets = list(_available_targets())
        if not targets:
            raise RuntimeError("No z-Image rotary targets with apply_rope were found")

        for module, attribute_name in targets:
            key = (module.__name__, attribute_name)
            _ORIGINAL_FUNCTIONS.setdefault(key, getattr(module, attribute_name))

        if not enable:
            for module, attribute_name in targets:
                setattr(module, attribute_name, _ORIGINAL_FUNCTIONS[(module.__name__, attribute_name)])
            logger.info("[baymax-zimage] Restored the default z-Image rotary implementation")
            return "restored"

        user_module = _load_user_module()
        user_function = getattr(user_module, "apply_rotary_emb", None)
        if not callable(user_function):
            raise RuntimeError(f"{_USER_IMPL_PATH} must define a callable apply_rotary_emb function")

        for module, attribute_name in targets:
            setattr(module, attribute_name, _make_wrapper(module.__name__, user_function))

        logger.info("[baymax-zimage] Installed user rotary implementation from %s", _USER_IMPL_PATH)
        return "patched"


class BaymaxZImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "enable": ("BOOLEAN", {"default": True}),
                "reload_user_impl": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch_model"
    CATEGORY = "baymax"
    DESCRIPTION = (
        "Patch the z-Image NextDiT rotary function with the implementation in "
        "custom_nodes/baymax-zimage/user_impl.py."
    )

    def patch_model(self, model, enable=True, reload_user_impl=True):
        diffusion_model = getattr(getattr(model, "model", None), "diffusion_model", None)
        if diffusion_model is not None and diffusion_model.__class__.__name__ != "NextDiTPixelSpace":
            logger.warning(
                "[baymax-zimage] Received model type %s; patch still applies globally to z-Image rotary code paths",
                diffusion_model.__class__.__name__,
            )

        patched_model = model.clone()
        status = _patch_rotary(enable=enable, reload_user_impl=reload_user_impl)

        transformer_options = patched_model.model_options.setdefault("transformer_options", {})
        transformer_options["baymax_zimage"] = {
            "status": status,
            "user_impl": str(_USER_IMPL_PATH),
        }
        return (patched_model,)


NODE_CLASS_MAPPINGS = {
    "BaymaxZImage": BaymaxZImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BaymaxZImage": "baymax-zimage",
}