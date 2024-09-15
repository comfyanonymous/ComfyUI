from typing import TYPE_CHECKING, List, Dict, Tuple
import enum
import torch
import numpy as np

if TYPE_CHECKING:
    from comfy.model_patcher import ModelPatcher
    from comfy.model_base import BaseModel
    from comfy.sd import CLIP
import comfy.lora
import comfy.model_management
from node_helpers import conditioning_set_values

class EnumHookMode(enum.Enum):
    MinVram = "minvram"
    MaxSpeed = "maxspeed"

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

class HookRef:
    pass

class Hook:
    def __init__(self):
        self.hook_ref = HookRef()
        self.hook_keyframe = HookKeyframeGroup()

    @property
    def strength(self):
        return self.hook_keyframe.strength

    def initialize_timesteps(self, model: 'BaseModel'):
        self.reset()
        self.hook_keyframe.initalize_timesteps(model)

    def reset(self):
        self.hook_keyframe.reset()

    def clone(self):
        c = Hook()
        c.hook_ref = self.hook_ref
        c.hook_keyframe = self.hook_keyframe
        return c
    
    def __eq__(self, other: 'Hook'):
        return self.__class__ == other.__class__ and self.hook_ref == other.hook_ref

    def __hash__(self):
        return hash(self.hook_ref)

class HookGroup:
    def __init__(self):
        self.hooks: List[Hook] = []

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
        for hook in other.hooks:
            c.add(hook.clone())
        return c
    
    def set_keyframes_on_hooks(self, hook_kf: 'HookKeyframeGroup'):
        hook_kf = hook_kf.clone()
        for hook in self.hooks:
            hook.hook_keyframe = hook_kf

    @staticmethod
    def combine_all_hooks(hooks_list: List['HookGroup'], require_count=1) -> 'HookGroup':
        actual: List[HookGroup] = []
        for group in hooks_list:
            if group is not None:
                actual.append(group)
        if len(actual) < require_count:
            raise Exception(f"Need at least {require_count} hooks to combine, but only had {len(actual)}.")
        # if only 1 hook, just reutnr itself without cloning
        if len(actual) == 1:
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
        self.keyframes: List[HookKeyframe] = []
        self._current_keyframe: HookKeyframe = None
        self._current_used_steps = 0
        self._current_index = 0
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
            c.keyframes.append(keyframe)
        c._set_first_as_current()
        return c
    
    def initalize_timesteps(self, model: 'BaseModel'):
        for keyframe in self.keyframes:
            keyframe.start_t = model.model_sampling.percent_to_sigma(keyframe.start_percent)

    def prepare_current_keyframe(self, curr_t: float) -> bool:
        if self.is_empty():
            return False
        if curr_t == self._curr_t:
            return False
        prev_index = self._current_index
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
        return prev_index != self._current_index

def get_sorted_list_via_attr(objects: List, attr: str) -> List:
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
        attr_list: List = unique_attrs.get(val_attr, list())
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


def load_hook_lora_for_models(model: 'ModelPatcher', clip: 'CLIP', lora: Dict[str, torch.Tensor],
                              hook: Hook, strength_model: float, strength_clip: float):
    key_map = {}
    if model is not None:
        key_map = comfy.lora.model_lora_keys_unet(model.model, key_map)
    if clip is not None:
        key_map = comfy.lora.model_lora_keys_clip(clip.cond_stage_model, key_map)

    loaded: Dict[str] = comfy.lora.load_lora(lora, key_map)
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
            print(f"NOT LOADED {x}")
    return (new_modelpatcher, new_clip)

def load_hook_model_as_lora_for_models(model: 'ModelPatcher', clip: 'CLIP',
                                       model_loaded: 'ModelPatcher', clip_loaded: 'CLIP',
                                       hook: Hook, strength_model: float, strength_clip: float):
    if model is not None and model_loaded is not None:
        new_modelpatcher = model.clone()
        expected_model_keys = set(model_loaded.model.state_dict().keys())
        patches_model: Dict[str, torch.Tensor] = model_loaded.model.state_dict()
        # do not include ANY model_sampling components of the model that should act as a patch
        for key in list(patches_model.keys()):
            if key.startswith("model_sampling"):
                expected_model_keys.discard(key)
                patches_model.pop(key, None)
        k = new_modelpatcher.add_hook_patches(hook=hook, patches=patches_model, strength_patch=strength_model, is_diff=True)
    else:
        k = ()
        new_modelpatcher = None
    
    if clip is not None and clip_loaded is not None:
        new_clip = clip.clone()
        comfy.model_management.unload_model_clones(new_clip.patcher)
        expected_clip_keys = clip_loaded.patcher.model.state_dict().copy()
        patches_clip: Dict[str, torch.Tensor] = clip_loaded.cond_stage_model.state_dict()
        k1 = new_clip.patcher.add_hook_patches(hook=hook, patches=patches_clip, strength_patch=strength_clip, is_diff=True)
    else:
        k1 = ()
        new_clip = None
    
    k = set(k)
    k1 = set(k1)
    if model is not None and model_loaded is not None:
        for key in expected_model_keys:
            if key not in k:
                print(f"MODEL-AS-LORA NOT LOADED {key}")
    if clip is not None and clip_loaded is not None:
        for key in expected_clip_keys:
            if key not in k1:
                print(f"CLIP-AS-LORA NOT LOADED {key}")
    
    return (new_modelpatcher, new_clip)

def set_hooks_for_conditioning(cond, hooks: HookGroup):
    if hooks is None:
        return cond
    return conditioning_set_values(cond, {'hooks': hooks})

def set_timesteps_for_conditioning(cond, timestep_range: Tuple[float,float]):
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

def combine_conditioning(conds: List):
    combined_conds = []
    for cond in conds:
        combined_conds.extend(cond)
    return combined_conds

def set_mask_conds(conds: List, strength: float, set_cond_area: str,
                   opt_mask: torch.Tensor=None, opt_hooks: HookGroup=None, opt_timestep_range: Tuple[float,float]=None):
    masked_conds = []
    for c in conds:
        # first, apply lora_hook to conditioning, if provided
        c = set_hooks_for_conditioning(c, opt_hooks)
        # next, apply mask to conditioning
        c = set_mask_for_conditioning(cond=c, mask=opt_mask, strength=strength, set_cond_area=set_cond_area)
        # apply timesteps, if present
        c = set_timesteps_for_conditioning(cond=c, timestep_range=opt_timestep_range)
        # finally, apply mask to conditioning and store
        masked_conds.append(c)
    return masked_conds

def set_mask_and_combine_conds(conds: List, new_conds: List, strength: float=1.0, set_cond_area: str="default",
                               opt_mask: torch.Tensor=None, opt_hooks: HookGroup=None, opt_timestep_range: Tuple[float,float]=None):
    combined_conds = []
    for c, masked_c in zip(conds, new_conds):
        # first, apply lora_hook to new conditioning, if provided
        masked_c = set_hooks_for_conditioning(masked_c, opt_hooks)
        # next, apply mask to new conditioning, if provided
        masked_c = set_mask_for_conditioning(cond=masked_c, mask=opt_mask, set_cond_area=set_cond_area, strength=strength)
        # apply timesteps, if present
        masked_c = set_timesteps_for_conditioning(cond=masked_c, timestep_range=opt_timestep_range)
        # finally, combine with existing conditioning and store
        combined_conds.append(combine_conditioning([c, masked_c]))
    return combined_conds

def set_default_and_combine_conds(conds: list, new_conds: list,
                                   opt_hooks: HookGroup=None, opt_timestep_range: Tuple[float,float]=None):
    combined_conds = []
    for c, new_c in zip(conds, new_conds):
        # first, apply lora_hook to new conditioning, if provided
        new_c = set_hooks_for_conditioning(new_c, opt_hooks)
        # next, add default_cond key to cond so that during sampling, it can be identified
        new_c = conditioning_set_values(new_c, {'default': True})
        # apply timesteps, if present
        new_c = set_timesteps_for_conditioning(cond=new_c, timestep_range=opt_timestep_range)
        # finally, combine with existing conditioning and store
        combined_conds.append(combine_conditioning([c, new_c]))
    return combined_conds
