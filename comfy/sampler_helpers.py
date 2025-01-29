from __future__ import annotations
import uuid
import comfy.model_management
import comfy.conds
import comfy.utils
import comfy.hooks
import comfy.patcher_extension
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from comfy.model_patcher import ModelPatcher
    from comfy.model_base import BaseModel
    from comfy.controlnet import ControlBase

def prepare_mask(noise_mask, shape, device):
    return comfy.utils.reshape_mask(noise_mask, shape).to(device)

def get_models_from_cond(cond, model_type):
    models = []
    for c in cond:
        if model_type in c:
            if isinstance(c[model_type], list):
                models += c[model_type]
            else:
                models += [c[model_type]]
    return models

def get_hooks_from_cond(cond, full_hooks: comfy.hooks.HookGroup):
    # get hooks from conds, and collect cnets so they can be checked for extra_hooks
    cnets: list[ControlBase] = []
    for c in cond:
        if 'hooks' in c:
            for hook in c['hooks'].hooks:
                full_hooks.add(hook)
        if 'control' in c:
            cnets.append(c['control'])

    def get_extra_hooks_from_cnet(cnet: ControlBase, _list: list):
        if cnet.extra_hooks is not None:
            _list.append(cnet.extra_hooks)
        if cnet.previous_controlnet is None:
            return _list
        return get_extra_hooks_from_cnet(cnet.previous_controlnet, _list)

    hooks_list = []
    cnets = set(cnets)
    for base_cnet in cnets:
        get_extra_hooks_from_cnet(base_cnet, hooks_list)
    extra_hooks = comfy.hooks.HookGroup.combine_all_hooks(hooks_list)
    if extra_hooks is not None:
        for hook in extra_hooks.hooks:
            full_hooks.add(hook)

    return full_hooks

def convert_cond(cond):
    out = []
    for c in cond:
        temp = c[1].copy()
        model_conds = temp.get("model_conds", {})
        if c[0] is not None:
            temp["cross_attn"] = c[0]
        temp["model_conds"] = model_conds
        temp["uuid"] = uuid.uuid4()
        out.append(temp)
    return out

def get_additional_models(conds, dtype):
    """loads additional models in conditioning"""
    cnets: list[ControlBase] = []
    gligen = []
    add_models = []

    for k in conds:
        cnets += get_models_from_cond(conds[k], "control")
        gligen += get_models_from_cond(conds[k], "gligen")
        add_models += get_models_from_cond(conds[k], "additional_models")

    control_nets = set(cnets)

    inference_memory = 0
    control_models = []
    for m in control_nets:
        control_models += m.get_models()
        inference_memory += m.inference_memory_requirements(dtype)

    gligen = [x[1] for x in gligen]
    models = control_models + gligen + add_models

    return models, inference_memory

def get_additional_models_from_model_options(model_options: dict[str]=None):
    """loads additional models from registered AddModels hooks"""
    models = []
    if model_options is not None and "registered_hooks" in model_options:
        registered: comfy.hooks.HookGroup = model_options["registered_hooks"]
        for hook in registered.get_type(comfy.hooks.EnumHookType.AdditionalModels):
            hook: comfy.hooks.AdditionalModelsHook
            models.extend(hook.models)
    return models

def cleanup_additional_models(models):
    """cleanup additional models that were loaded"""
    for m in models:
        if hasattr(m, 'cleanup'):
            m.cleanup()


def prepare_sampling(model: ModelPatcher, noise_shape, conds, model_options=None):
    real_model: BaseModel = None
    models, inference_memory = get_additional_models(conds, model.model_dtype())
    models += get_additional_models_from_model_options(model_options)
    models += model.get_nested_additional_models()  # TODO: does this require inference_memory update?
    memory_required = model.memory_required([noise_shape[0] * 2] + list(noise_shape[1:])) + inference_memory
    minimum_memory_required = model.memory_required([noise_shape[0]] + list(noise_shape[1:])) + inference_memory
    comfy.model_management.load_models_gpu([model] + models, memory_required=memory_required, minimum_memory_required=minimum_memory_required)
    real_model = model.model

    return real_model, conds, models

def cleanup_models(conds, models):
    cleanup_additional_models(models)

    control_cleanup = []
    for k in conds:
        control_cleanup += get_models_from_cond(conds[k], "control")

    cleanup_additional_models(set(control_cleanup))

def prepare_model_patcher(model: 'ModelPatcher', conds, model_options: dict):
    '''
    Registers hooks from conds.
    '''
    # check for hooks in conds - if not registered, see if can be applied
    hooks = comfy.hooks.HookGroup()
    for k in conds:
        get_hooks_from_cond(conds[k], hooks)
    # add wrappers and callbacks from ModelPatcher to transformer_options
    model_options["transformer_options"]["wrappers"] = comfy.patcher_extension.copy_nested_dicts(model.wrappers)
    model_options["transformer_options"]["callbacks"] = comfy.patcher_extension.copy_nested_dicts(model.callbacks)
    # begin registering hooks
    registered = comfy.hooks.HookGroup()
    target_dict = comfy.hooks.create_target_dict(comfy.hooks.EnumWeightTarget.Model)
    # handle all TransformerOptionsHooks
    for hook in hooks.get_type(comfy.hooks.EnumHookType.TransformerOptions):
        hook: comfy.hooks.TransformerOptionsHook
        hook.add_hook_patches(model, model_options, target_dict, registered)
    # handle all AddModelsHooks
    for hook in hooks.get_type(comfy.hooks.EnumHookType.AdditionalModels):
        hook: comfy.hooks.AdditionalModelsHook
        hook.add_hook_patches(model, model_options, target_dict, registered)
    # handle all WeightHooks by registering on ModelPatcher
    model.register_all_hook_patches(hooks, target_dict, model_options, registered)
    # add registered_hooks onto model_options for further reference
    if len(registered) > 0:
        model_options["registered_hooks"] = registered
    # merge original wrappers and callbacks with hooked wrappers and callbacks
    to_load_options: dict[str] = model_options.setdefault("to_load_options", {})
    for wc_name in ["wrappers", "callbacks"]:
        comfy.patcher_extension.merge_nested_dicts(to_load_options.setdefault(wc_name, {}), model_options["transformer_options"][wc_name],
                                                    copy_dict1=False)
    return to_load_options
