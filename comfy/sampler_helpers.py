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

def get_hooks_from_cond(cond, hooks_dict: dict[comfy.hooks.EnumHookType, dict[comfy.hooks.Hook, None]]):
    # get hooks from conds, and collect cnets so they can be checked for extra_hooks
    cnets: list[ControlBase] = []
    for c in cond:
        if 'hooks' in c:
            for hook in c['hooks'].hooks:
                hook: comfy.hooks.Hook
                with_type = hooks_dict.setdefault(hook.hook_type, {})
                with_type[hook] = None
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
            with_type = hooks_dict.setdefault(hook.hook_type, {})
            with_type[hook] = None

    return hooks_dict

def convert_cond(cond):
    out = []
    for c in cond:
        temp = c[1].copy()
        model_conds = temp.get("model_conds", {})
        if c[0] is not None:
            model_conds["c_crossattn"] = comfy.conds.CONDCrossAttn(c[0]) #TODO: remove
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
    hooks: dict[comfy.hooks.EnumHookType, dict[comfy.hooks.Hook, None]] = {}

    for k in conds:
        cnets += get_models_from_cond(conds[k], "control")
        gligen += get_models_from_cond(conds[k], "gligen")
        add_models += get_models_from_cond(conds[k], "additional_models")
        get_hooks_from_cond(conds[k], hooks)

    control_nets = set(cnets)

    inference_memory = 0
    control_models = []
    for m in control_nets:
        control_models += m.get_models()
        inference_memory += m.inference_memory_requirements(dtype)

    gligen = [x[1] for x in gligen]
    hook_models = [x.model for x in hooks.get(comfy.hooks.EnumHookType.AddModels, {}).keys()]
    models = control_models + gligen + add_models + hook_models

    return models, inference_memory

def cleanup_additional_models(models):
    """cleanup additional models that were loaded"""
    for m in models:
        if hasattr(m, 'cleanup'):
            m.cleanup()


def prepare_sampling(model: 'ModelPatcher', noise_shape, conds):
    real_model: 'BaseModel' = None
    models, inference_memory = get_additional_models(conds, model.model_dtype())
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
    # check for hooks in conds - if not registered, see if can be applied
    hooks = {}
    for k in conds:
        get_hooks_from_cond(conds[k], hooks)
    # add wrappers and callbacks from ModelPatcher to transformer_options
    model_options["transformer_options"]["wrappers"] = comfy.patcher_extension.copy_nested_dicts(model.wrappers)
    model_options["transformer_options"]["callbacks"] = comfy.patcher_extension.copy_nested_dicts(model.callbacks)
    # register hooks on model/model_options
    model.register_all_hook_patches(hooks, comfy.hooks.EnumWeightTarget.Model, model_options)
