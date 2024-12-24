from __future__ import annotations
from typing import TYPE_CHECKING, Callable
import enum
import math
import torch
import numpy as np
import itertools
import logging

if TYPE_CHECKING:
    from comfy.model_patcher import ModelPatcher, PatcherInjection
    from comfy.model_base import BaseModel
    from comfy.sd import CLIP
import comfy.lora
import comfy.model_management
import comfy.patcher_extension
from node_helpers import conditioning_set_values

class EnumHookMode(enum.Enum):
    MinVram = "minvram"
    MaxSpeed = "maxspeed"

class EnumHookType(enum.Enum):
    Weight = "weight"
    Patch = "patch"
    ObjectPatch = "object_patch"
    AddModels = "add_models"
    Callbacks = "callbacks"
    Wrappers = "wrappers"
    SetInjections = "add_injections"

class EnumWeightTarget(enum.Enum):
    Model = "model"
    Clip = "clip"

class _HookRef:
    pass

# NOTE: this is an example of how the should_register function should look
def default_should_register(hook: 'Hook', model: 'ModelPatcher', model_options: dict, target: EnumWeightTarget, registered: list[Hook]):
    return True


class Hook:
    def __init__(self, hook_type: EnumHookType=None, hook_ref: _HookRef=None, hook_id: str=None,
                 hook_keyframe: 'HookKeyframeGroup'=None):
        self.hook_type = hook_type
        self.hook_ref = hook_ref if hook_ref else _HookRef()
        self.hook_id = hook_id
        self.hook_keyframe = hook_keyframe if hook_keyframe else HookKeyframeGroup()
        self.custom_should_register = default_should_register
        self.auto_apply_to_nonpositive = False

    @property
    def strength(self):
        return self.hook_keyframe.strength

    def initialize_timesteps(self, model: 'BaseModel'):
        self.reset()
        self.hook_keyframe.initialize_timesteps(model)

    def reset(self):
        self.hook_keyframe.reset()

    def clone(self, subtype: Callable=None):
        if subtype is None:
            subtype = type(self)
        c: Hook = subtype()
        c.hook_type = self.hook_type
        c.hook_ref = self.hook_ref
        c.hook_id = self.hook_id
        c.hook_keyframe = self.hook_keyframe
        c.custom_should_register = self.custom_should_register
        # TODO: make this do something
        c.auto_apply_to_nonpositive = self.auto_apply_to_nonpositive
        return c

    def should_register(self, model: 'ModelPatcher', model_options: dict, target: EnumWeightTarget, registered: list[Hook]):
        return self.custom_should_register(self, model, model_options, target, registered)

    def add_hook_patches(self, model: 'ModelPatcher', model_options: dict, target: EnumWeightTarget, registered: list[Hook]):
        raise NotImplementedError("add_hook_patches should be defined for Hook subclasses")

    def on_apply(self, model: 'ModelPatcher', transformer_options: dict[str]):
        pass

    def on_unapply(self, model: 'ModelPatcher', transformer_options: dict[str]):
        pass

    def __eq__(self, other: 'Hook'):
        return self.__class__ == other.__class__ and self.hook_ref == other.hook_ref

    def __hash__(self):
        return hash(self.hook_ref)

class WeightHook(Hook):
    def __init__(self, strength_model=1.0, strength_clip=1.0):
        super().__init__(hook_type=EnumHookType.Weight)
        self.weights: dict = None
        self.weights_clip: dict = None
        self.need_weight_init = True
        self._strength_model = strength_model
        self._strength_clip = strength_clip
    
    @property
    def strength_model(self):
        return self._strength_model * self.strength
    
    @property
    def strength_clip(self):
        return self._strength_clip * self.strength

    def add_hook_patches(self, model: 'ModelPatcher', model_options: dict, target: EnumWeightTarget, registered: list[Hook]):
        if not self.should_register(model, model_options, target, registered):
            return False
        weights = None
        if target == EnumWeightTarget.Model:
            strength = self._strength_model
        else:
            strength = self._strength_clip
        
        if self.need_weight_init:
            key_map = {}
            if target == EnumWeightTarget.Model:
                key_map = comfy.lora.model_lora_keys_unet(model.model, key_map)
            else:
                key_map = comfy.lora.model_lora_keys_clip(model.model, key_map)
            weights = comfy.lora.load_lora(self.weights, key_map, log_missing=False)
        else:
            if target == EnumWeightTarget.Model:
                weights = self.weights
            else:
                weights = self.weights_clip
        model.add_hook_patches(hook=self, patches=weights, strength_patch=strength)
        registered.append(self)
        return True
        # TODO: add logs about any keys that were not applied

    def clone(self, subtype: Callable=None):
        if subtype is None:
            subtype = type(self)
        c: WeightHook = super().clone(subtype)
        c.weights = self.weights
        c.weights_clip = self.weights_clip
        c.need_weight_init = self.need_weight_init
        c._strength_model = self._strength_model
        c._strength_clip = self._strength_clip
        return c

class PatchHook(Hook):
    def __init__(self):
        super().__init__(hook_type=EnumHookType.Patch)
        self.patches: dict = None
    
    def clone(self, subtype: Callable=None):
        if subtype is None:
            subtype = type(self)
        c: PatchHook = super().clone(subtype)
        c.patches = self.patches
        return c
    # TODO: add functionality

class ObjectPatchHook(Hook):
    def __init__(self):
        super().__init__(hook_type=EnumHookType.ObjectPatch)
        self.object_patches: dict = None
    
    def clone(self, subtype: Callable=None):
        if subtype is None:
            subtype = type(self)
        c: ObjectPatchHook = super().clone(subtype)
        c.object_patches = self.object_patches
        return c
    # TODO: add functionality

class AddModelsHook(Hook):
    def __init__(self, key: str=None, models: list['ModelPatcher']=None):
        super().__init__(hook_type=EnumHookType.AddModels)
        self.key = key
        self.models = models
        self.append_when_same = True
    
    def clone(self, subtype: Callable=None):
        if subtype is None:
            subtype = type(self)
        c: AddModelsHook = super().clone(subtype)
        c.key = self.key
        c.models = self.models.copy() if self.models else self.models
        c.append_when_same = self.append_when_same
        return c
    # TODO: add functionality

class CallbackHook(Hook):
    def __init__(self, key: str=None, callback: Callable=None):
        super().__init__(hook_type=EnumHookType.Callbacks)
        self.key = key
        self.callback = callback

    def clone(self, subtype: Callable=None):
        if subtype is None:
            subtype = type(self)
        c: CallbackHook = super().clone(subtype)
        c.key = self.key
        c.callback = self.callback
        return c
    # TODO: add functionality

class WrapperHook(Hook):
    def __init__(self, wrappers_dict: dict[str, dict[str, dict[str, list[Callable]]]]=None):
        super().__init__(hook_type=EnumHookType.Wrappers)
        self.wrappers_dict = wrappers_dict

    def clone(self, subtype: Callable=None):
        if subtype is None:
            subtype = type(self)
        c: WrapperHook = super().clone(subtype)
        c.wrappers_dict = self.wrappers_dict
        return c
    
    def add_hook_patches(self, model: 'ModelPatcher', model_options: dict, target: EnumWeightTarget, registered: list[Hook]):
        if not self.should_register(model, model_options, target, registered):
            return False
        add_model_options = {"transformer_options": self.wrappers_dict}
        comfy.patcher_extension.merge_nested_dicts(model_options, add_model_options, copy_dict1=False)
        registered.append(self)
        return True

class SetInjectionsHook(Hook):
    def __init__(self, key: str=None, injections: list['PatcherInjection']=None):
        super().__init__(hook_type=EnumHookType.SetInjections)
        self.key = key
        self.injections = injections
    
    def clone(self, subtype: Callable=None):
        if subtype is None:
            subtype = type(self)
        c: SetInjectionsHook = super().clone(subtype)
        c.key = self.key
        c.injections = self.injections.copy() if self.injections else self.injections
        return c
    
    def add_hook_injections(self, model: 'ModelPatcher'):
        # TODO: add functionality
        pass

class HookGroup:
    def __init__(self):
        self.hooks: list[Hook] = []

    def add(self, hook: Hook):
        if hook not in self.hooks:
            self.hooks.append(hook)
    
    def contains(self, hook: Hook):
        return hook in self.hooks
    
    def clone(self):
        c = HookGroup()
        for hook in self.hooks:
            c.add(hook.clone())
        return c

    def clone_and_combine(self, other: 'HookGroup'):
        c = self.clone()
        if other is not None:
            for hook in other.hooks:
                c.add(hook.clone())
        return c
    
    def set_keyframes_on_hooks(self, hook_kf: 'HookKeyframeGroup'):
        if hook_kf is None:
            hook_kf = HookKeyframeGroup()
        else:
            hook_kf = hook_kf.clone()
        for hook in self.hooks:
            hook.hook_keyframe = hook_kf

    def get_dict_repr(self):
        d: dict[EnumHookType, dict[Hook, None]] = {}
        for hook in self.hooks:
            with_type = d.setdefault(hook.hook_type, {})
            with_type[hook] = None
        return d

    def get_hooks_for_clip_schedule(self):
        scheduled_hooks: dict[WeightHook, list[tuple[tuple[float,float], HookKeyframe]]] = {}
        for hook in self.hooks:
            # only care about WeightHooks, for now
            if hook.hook_type == EnumHookType.Weight:
                hook_schedule = []
                # if no hook keyframes, assign default value
                if len(hook.hook_keyframe.keyframes) == 0:
                    hook_schedule.append(((0.0, 1.0), None))
                    scheduled_hooks[hook] = hook_schedule
                    continue
                # find ranges of values
                prev_keyframe = hook.hook_keyframe.keyframes[0]
                for keyframe in hook.hook_keyframe.keyframes:
                    if keyframe.start_percent > prev_keyframe.start_percent and not math.isclose(keyframe.strength, prev_keyframe.strength):
                        hook_schedule.append(((prev_keyframe.start_percent, keyframe.start_percent), prev_keyframe))
                        prev_keyframe = keyframe
                    elif keyframe.start_percent == prev_keyframe.start_percent:
                        prev_keyframe = keyframe
                # create final range, assuming last start_percent was not 1.0
                if not math.isclose(prev_keyframe.start_percent, 1.0):
                    hook_schedule.append(((prev_keyframe.start_percent, 1.0), prev_keyframe))
                scheduled_hooks[hook] = hook_schedule
        # hooks should not have their schedules in a list of tuples
        all_ranges: list[tuple[float, float]] = []
        for range_kfs in scheduled_hooks.values():
            for t_range, keyframe in range_kfs:
                all_ranges.append(t_range)
        # turn list of ranges into boundaries
        boundaries_set = set(itertools.chain.from_iterable(all_ranges))
        boundaries_set.add(0.0)
        boundaries = sorted(boundaries_set)
        real_ranges = [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]
        # with real ranges defined, give appropriate hooks w/ keyframes for each range
        scheduled_keyframes: list[tuple[tuple[float,float], list[tuple[WeightHook, HookKeyframe]]]] = []
        for t_range in real_ranges:
            hooks_schedule = []
            for hook, val in scheduled_hooks.items():
                keyframe = None
                # check if is a keyframe that works for the current t_range
                for stored_range, stored_kf in val:
                    # if stored start is less than current end, then fits - give it assigned keyframe
                    if stored_range[0] < t_range[1] and stored_range[1] > t_range[0]:
                        keyframe = stored_kf
                        break
                hooks_schedule.append((hook, keyframe))
            scheduled_keyframes.append((t_range, hooks_schedule))
        return scheduled_keyframes

    def reset(self):
        for hook in self.hooks:
            hook.reset()

    @staticmethod
    def combine_all_hooks(hooks_list: list['HookGroup'], require_count=0) -> 'HookGroup':
        actual: list[HookGroup] = []
        for group in hooks_list:
            if group is not None:
                actual.append(group)
        if len(actual) < require_count:
            raise Exception(f"Need at least {require_count} hooks to combine, but only had {len(actual)}.")
        # if no hooks, then return None
        if len(actual) == 0:
            return None
        # if only 1 hook, just return itself without cloning
        elif len(actual) == 1:
            return actual[0]
        final_hook: HookGroup = None
        for hook in actual:
            if final_hook is None:
                final_hook = hook.clone()
            else:
                final_hook = final_hook.clone_and_combine(hook)
        return final_hook


class HookKeyframe:
    def __init__(self, strength: float, start_percent=0.0, guarantee_steps=1):
        self.strength = strength
        # scheduling
        self.start_percent = float(start_percent)
        self.start_t = 999999999.9
        self.guarantee_steps = guarantee_steps
    
    def clone(self):
        c = HookKeyframe(strength=self.strength,
                                start_percent=self.start_percent, guarantee_steps=self.guarantee_steps)
        c.start_t = self.start_t
        return c

class HookKeyframeGroup:
    def __init__(self):
        self.keyframes: list[HookKeyframe] = []
        self._current_keyframe: HookKeyframe = None
        self._current_used_steps = 0
        self._current_index = 0
        self._current_strength = None
        self._curr_t = -1.

    # properties shadow those of HookWeightsKeyframe
    @property
    def strength(self):
        if self._current_keyframe is not None:
            return self._current_keyframe.strength
        return 1.0

    def reset(self):
        self._current_keyframe = None
        self._current_used_steps = 0
        self._current_index = 0
        self._current_strength = None
        self.curr_t = -1.
        self._set_first_as_current()
    
    def add(self, keyframe: HookKeyframe):
        # add to end of list, then sort
        self.keyframes.append(keyframe)
        self.keyframes = get_sorted_list_via_attr(self.keyframes, "start_percent")
        self._set_first_as_current()

    def _set_first_as_current(self):
        if len(self.keyframes) > 0:
            self._current_keyframe = self.keyframes[0]
        else:
            self._current_keyframe = None
    
    def has_index(self, index: int):
        return index >= 0 and index < len(self.keyframes)

    def is_empty(self):
        return len(self.keyframes) == 0
    
    def clone(self):
        c = HookKeyframeGroup()
        for keyframe in self.keyframes:
            c.keyframes.append(keyframe.clone())
        c._set_first_as_current()
        return c
    
    def initialize_timesteps(self, model: 'BaseModel'):
        for keyframe in self.keyframes:
            keyframe.start_t = model.model_sampling.percent_to_sigma(keyframe.start_percent)

    def prepare_current_keyframe(self, curr_t: float) -> bool:
        if self.is_empty():
            return False
        if curr_t == self._curr_t:
            return False
        prev_index = self._current_index
        prev_strength = self._current_strength
        # if met guaranteed steps, look for next keyframe in case need to switch
        if self._current_used_steps >= self._current_keyframe.guarantee_steps:
            # if has next index, loop through and see if need to switch
            if self.has_index(self._current_index+1):
                for i in range(self._current_index+1, len(self.keyframes)):
                    eval_c = self.keyframes[i]
                    # check if start_t is greater or equal to curr_t
                    # NOTE: t is in terms of sigmas, not percent, so bigger number = earlier step in sampling
                    if eval_c.start_t >= curr_t:
                        self._current_index = i
                        self._current_strength = eval_c.strength
                        self._current_keyframe = eval_c
                        self._current_used_steps = 0
                        # if guarantee_steps greater than zero, stop searching for other keyframes
                        if self._current_keyframe.guarantee_steps > 0:
                            break
                    # if eval_c is outside the percent range, stop looking further
                    else: break
        # update steps current context is used
        self._current_used_steps += 1
        # update current timestep this was performed on
        self._curr_t = curr_t
        # return True if keyframe changed, False if no change
        return prev_index != self._current_index and prev_strength != self._current_strength


class InterpolationMethod:
    LINEAR = "linear"
    EASE_IN = "ease_in"
    EASE_OUT = "ease_out"
    EASE_IN_OUT = "ease_in_out"

    _LIST = [LINEAR, EASE_IN, EASE_OUT, EASE_IN_OUT]

    @classmethod
    def get_weights(cls, num_from: float, num_to: float, length: int, method: str, reverse=False):
        diff = num_to - num_from
        if method == cls.LINEAR:
            weights = torch.linspace(num_from, num_to, length)
        elif method == cls.EASE_IN:
            index = torch.linspace(0, 1, length)
            weights = diff * np.power(index, 2) + num_from
        elif method == cls.EASE_OUT:
            index = torch.linspace(0, 1, length)
            weights = diff * (1 - np.power(1 - index, 2)) + num_from
        elif method == cls.EASE_IN_OUT:
            index = torch.linspace(0, 1, length)
            weights = diff * ((1 - np.cos(index * np.pi)) / 2) + num_from
        else:
            raise ValueError(f"Unrecognized interpolation method '{method}'.")
        if reverse:
            weights = weights.flip(dims=(0,))
        return weights

def get_sorted_list_via_attr(objects: list, attr: str) -> list:
    if not objects:
        return objects
    elif len(objects) <= 1:
        return [x for x in objects]
    # now that we know we have to sort, do it following these rules:
    # a) if objects have same value of attribute, maintain their relative order
    # b) perform sorting of the groups of objects with same attributes
    unique_attrs = {}
    for o in objects:
        val_attr = getattr(o, attr)
        attr_list: list = unique_attrs.get(val_attr, list())
        attr_list.append(o)
        if val_attr not in unique_attrs:
            unique_attrs[val_attr] = attr_list
    # now that we have the unique attr values grouped together in relative order, sort them by key
    sorted_attrs = dict(sorted(unique_attrs.items()))
    # now flatten out the dict into a list to return
    sorted_list = []
    for object_list in sorted_attrs.values():
        sorted_list.extend(object_list)
    return sorted_list

def create_hook_lora(lora: dict[str, torch.Tensor], strength_model: float, strength_clip: float):
    hook_group = HookGroup()
    hook = WeightHook(strength_model=strength_model, strength_clip=strength_clip)
    hook_group.add(hook)
    hook.weights = lora
    return hook_group

def create_hook_model_as_lora(weights_model, weights_clip, strength_model: float, strength_clip: float):
    hook_group = HookGroup()
    hook = WeightHook(strength_model=strength_model, strength_clip=strength_clip)
    hook_group.add(hook)
    patches_model = None
    patches_clip = None
    if weights_model is not None:
        patches_model = {}
        for key in weights_model:
            patches_model[key] = ("model_as_lora", (weights_model[key],))
    if weights_clip is not None:
        patches_clip = {}
        for key in weights_clip:
            patches_clip[key] = ("model_as_lora", (weights_clip[key],))
    hook.weights = patches_model
    hook.weights_clip = patches_clip
    hook.need_weight_init = False
    return hook_group

def get_patch_weights_from_model(model: 'ModelPatcher', discard_model_sampling=True):
    if model is None:
        return None
    patches_model: dict[str, torch.Tensor] = model.model.state_dict()
    if discard_model_sampling:
        # do not include ANY model_sampling components of the model that should act as a patch
        for key in list(patches_model.keys()):
            if key.startswith("model_sampling"):
                patches_model.pop(key, None)
    return patches_model

# NOTE: this function shows how to register weight hooks directly on the ModelPatchers
def load_hook_lora_for_models(model: 'ModelPatcher', clip: 'CLIP', lora: dict[str, torch.Tensor],
                              strength_model: float, strength_clip: float):
    key_map = {}
    if model is not None:
        key_map = comfy.lora.model_lora_keys_unet(model.model, key_map)
    if clip is not None:
        key_map = comfy.lora.model_lora_keys_clip(clip.cond_stage_model, key_map)

    hook_group = HookGroup()
    hook = WeightHook()
    hook_group.add(hook)
    loaded: dict[str] = comfy.lora.load_lora(lora, key_map)
    if model is not None:
        new_modelpatcher = model.clone()
        k = new_modelpatcher.add_hook_patches(hook=hook, patches=loaded, strength_patch=strength_model)
    else:
        k = ()
        new_modelpatcher = None
    
    if clip is not None:
        new_clip = clip.clone()
        k1 = new_clip.patcher.add_hook_patches(hook=hook, patches=loaded, strength_patch=strength_clip)
    else:
        k1 = ()
        new_clip = None
    k = set(k)
    k1 = set(k1)
    for x in loaded:
        if (x not in k) and (x not in k1):
            logging.warning(f"NOT LOADED {x}")
    return (new_modelpatcher, new_clip, hook_group)

def _combine_hooks_from_values(c_dict: dict[str, HookGroup], values: dict[str, HookGroup], cache: dict[tuple[HookGroup, HookGroup], HookGroup]):
    hooks_key = 'hooks'
    # if hooks only exist in one dict, do what's needed so that it ends up in c_dict
    if hooks_key not in values:
        return
    if hooks_key not in c_dict:
        hooks_value = values.get(hooks_key, None)
        if hooks_value is not None:
            c_dict[hooks_key] = hooks_value
        return
    # otherwise, need to combine with minimum duplication via cache
    hooks_tuple = (c_dict[hooks_key], values[hooks_key])
    cached_hooks = cache.get(hooks_tuple, None)
    if cached_hooks is None:
        new_hooks = hooks_tuple[0].clone_and_combine(hooks_tuple[1])
        cache[hooks_tuple] = new_hooks
        c_dict[hooks_key] = new_hooks
    else:
        c_dict[hooks_key] = cache[hooks_tuple]

def conditioning_set_values_with_hooks(conditioning, values={}, append_hooks=True):
    c = []
    hooks_combine_cache: dict[tuple[HookGroup, HookGroup], HookGroup] = {}
    for t in conditioning:
        n = [t[0], t[1].copy()]
        for k in values:
            if append_hooks and k == 'hooks':
                _combine_hooks_from_values(n[1], values, hooks_combine_cache)
            else:
                n[1][k] = values[k]
        c.append(n)

    return c

def set_hooks_for_conditioning(cond, hooks: HookGroup, append_hooks=True):
    if hooks is None:
        return cond
    return conditioning_set_values_with_hooks(cond, {'hooks': hooks}, append_hooks=append_hooks)

def set_timesteps_for_conditioning(cond, timestep_range: tuple[float,float]):
    if timestep_range is None:
        return cond
    return conditioning_set_values(cond, {"start_percent": timestep_range[0],
                                          "end_percent": timestep_range[1]})

def set_mask_for_conditioning(cond, mask: torch.Tensor, set_cond_area: str, strength: float):
    if mask is None:
        return cond
    set_area_to_bounds = False
    if set_cond_area != 'default':
        set_area_to_bounds = True
    if len(mask.shape) < 3:
        mask = mask.unsqueeze(0)
    return conditioning_set_values(cond, {'mask': mask,
                                          'set_area_to_bounds': set_area_to_bounds,
                                          'mask_strength': strength})

def combine_conditioning(conds: list):
    combined_conds = []
    for cond in conds:
        combined_conds.extend(cond)
    return combined_conds

def combine_with_new_conds(conds: list, new_conds: list):
    combined_conds = []
    for c, new_c in zip(conds, new_conds):
        combined_conds.append(combine_conditioning([c, new_c]))
    return combined_conds

def set_conds_props(conds: list, strength: float, set_cond_area: str,
                   mask: torch.Tensor=None, hooks: HookGroup=None, timesteps_range: tuple[float,float]=None, append_hooks=True):
    final_conds = []
    for c in conds:
        # first, apply lora_hook to conditioning, if provided
        c = set_hooks_for_conditioning(c, hooks, append_hooks=append_hooks)
        # next, apply mask to conditioning
        c = set_mask_for_conditioning(cond=c, mask=mask, strength=strength, set_cond_area=set_cond_area)
        # apply timesteps, if present
        c = set_timesteps_for_conditioning(cond=c, timestep_range=timesteps_range)
        # finally, apply mask to conditioning and store
        final_conds.append(c)
    return final_conds

def set_conds_props_and_combine(conds: list, new_conds: list, strength: float=1.0, set_cond_area: str="default",
                               mask: torch.Tensor=None, hooks: HookGroup=None, timesteps_range: tuple[float,float]=None, append_hooks=True):
    combined_conds = []
    for c, masked_c in zip(conds, new_conds):
        # first, apply lora_hook to new conditioning, if provided
        masked_c = set_hooks_for_conditioning(masked_c, hooks, append_hooks=append_hooks)
        # next, apply mask to new conditioning, if provided
        masked_c = set_mask_for_conditioning(cond=masked_c, mask=mask, set_cond_area=set_cond_area, strength=strength)
        # apply timesteps, if present
        masked_c = set_timesteps_for_conditioning(cond=masked_c, timestep_range=timesteps_range)
        # finally, combine with existing conditioning and store
        combined_conds.append(combine_conditioning([c, masked_c]))
    return combined_conds

def set_default_conds_and_combine(conds: list, new_conds: list,
                                   hooks: HookGroup=None, timesteps_range: tuple[float,float]=None, append_hooks=True):
    combined_conds = []
    for c, new_c in zip(conds, new_conds):
        # first, apply lora_hook to new conditioning, if provided
        new_c = set_hooks_for_conditioning(new_c, hooks, append_hooks=append_hooks)
        # next, add default_cond key to cond so that during sampling, it can be identified
        new_c = conditioning_set_values(new_c, {'default': True})
        # apply timesteps, if present
        new_c = set_timesteps_for_conditioning(cond=new_c, timestep_range=timesteps_range)
        # finally, combine with existing conditioning and store
        combined_conds.append(combine_conditioning([c, new_c]))
    return combined_conds
