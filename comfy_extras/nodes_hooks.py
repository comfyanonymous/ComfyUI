from __future__ import annotations
from typing import TYPE_CHECKING, Union
import torch
from collections.abc import Iterable

if TYPE_CHECKING:
    from comfy.sd import CLIP

import comfy.hooks
import comfy.sd
import comfy.utils
import folder_paths

###########################################
# Mask, Combine, and Hook Conditioning
#------------------------------------------
class PairConditioningSetProperties:
    NodeId = 'PairConditioningSetProperties'
    NodeName = 'Cond Pair Set Props'
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "positive_NEW": ("CONDITIONING", ),
                "negative_NEW": ("CONDITIONING", ),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "set_cond_area": (["default", "mask bounds"],),
            },
            "optional": {
                "mask": ("MASK", ),
                "hooks": ("HOOKS",),
                "timesteps": ("TIMESTEPS_RANGE",),
            }
        }
    
    EXPERIMENTAL = True
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    CATEGORY = "advanced/hooks/cond pair"
    FUNCTION = "set_properties"

    def set_properties(self, positive_NEW, negative_NEW,
                       strength: float, set_cond_area: str,
                       mask: torch.Tensor=None, hooks: comfy.hooks.HookGroup=None, timesteps: tuple=None):
        final_positive, final_negative = comfy.hooks.set_conds_props(conds=[positive_NEW, negative_NEW],
                                                                    strength=strength, set_cond_area=set_cond_area,
                                                                    mask=mask, hooks=hooks, timesteps_range=timesteps)
        return (final_positive, final_negative)
    
class PairConditioningSetPropertiesAndCombine:
    NodeId = 'PairConditioningSetPropertiesAndCombine'
    NodeName = 'Cond Pair Set Props Combine'
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "positive": ("CONDITIONING", ),
                "negative": ("CONDITIONING", ),
                "positive_NEW": ("CONDITIONING", ),
                "negative_NEW": ("CONDITIONING", ),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "set_cond_area": (["default", "mask bounds"],),
            },
            "optional": {
                "mask": ("MASK", ),
                "hooks": ("HOOKS",),
                "timesteps": ("TIMESTEPS_RANGE",),
            }
        }
    
    EXPERIMENTAL = True
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    CATEGORY = "advanced/hooks/cond pair"
    FUNCTION = "set_properties"

    def set_properties(self, positive, negative, positive_NEW, negative_NEW,
                       strength: float, set_cond_area: str,
                       mask: torch.Tensor=None, hooks: comfy.hooks.HookGroup=None, timesteps: tuple=None):
        final_positive, final_negative = comfy.hooks.set_conds_props_and_combine(conds=[positive, negative], new_conds=[positive_NEW, negative_NEW],
                                                                                strength=strength, set_cond_area=set_cond_area,
                                                                                mask=mask, hooks=hooks, timesteps_range=timesteps)
        return (final_positive, final_negative)

class ConditioningSetProperties:
    NodeId = 'ConditioningSetProperties'
    NodeName = 'Cond Set Props'
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "cond_NEW": ("CONDITIONING", ),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "set_cond_area": (["default", "mask bounds"],),
            },
            "optional": {
                "mask": ("MASK", ),
                "hooks": ("HOOKS",),
                "timesteps": ("TIMESTEPS_RANGE",),
            }
        }

    EXPERIMENTAL = True
    RETURN_TYPES = ("CONDITIONING",)
    CATEGORY = "advanced/hooks/cond single"
    FUNCTION = "set_properties"

    def set_properties(self, cond_NEW,
                       strength: float, set_cond_area: str,
                       mask: torch.Tensor=None, hooks: comfy.hooks.HookGroup=None, timesteps: tuple=None):
        (final_cond,) = comfy.hooks.set_conds_props(conds=[cond_NEW],
                                                   strength=strength, set_cond_area=set_cond_area,
                                                   mask=mask, hooks=hooks, timesteps_range=timesteps)
        return (final_cond,)

class ConditioningSetPropertiesAndCombine:
    NodeId = 'ConditioningSetPropertiesAndCombine'
    NodeName = 'Cond Set Props Combine'
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "cond": ("CONDITIONING", ),
                "cond_NEW": ("CONDITIONING", ),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "set_cond_area": (["default", "mask bounds"],),
            },
            "optional": {
                "mask": ("MASK", ),
                "hooks": ("HOOKS",),
                "timesteps": ("TIMESTEPS_RANGE",),
            }
        }

    EXPERIMENTAL = True
    RETURN_TYPES = ("CONDITIONING",)
    CATEGORY = "advanced/hooks/cond single"
    FUNCTION = "set_properties"

    def set_properties(self, cond, cond_NEW,
                       strength: float, set_cond_area: str,
                       mask: torch.Tensor=None, hooks: comfy.hooks.HookGroup=None, timesteps: tuple=None):
        (final_cond,) = comfy.hooks.set_conds_props_and_combine(conds=[cond], new_conds=[cond_NEW],
                                                               strength=strength, set_cond_area=set_cond_area,
                                                               mask=mask, hooks=hooks, timesteps_range=timesteps)
        return (final_cond,)

class PairConditioningCombine:
    NodeId = 'PairConditioningCombine'
    NodeName = 'Cond Pair Combine'
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "positive_A": ("CONDITIONING",),
                "negative_A": ("CONDITIONING",),
                "positive_B": ("CONDITIONING",),
                "negative_B": ("CONDITIONING",),
            },
        }
    
    EXPERIMENTAL = True
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    CATEGORY = "advanced/hooks/cond pair"
    FUNCTION = "combine"

    def combine(self, positive_A, negative_A, positive_B, negative_B):
        final_positive, final_negative = comfy.hooks.set_conds_props_and_combine(conds=[positive_A, negative_A], new_conds=[positive_B, negative_B],)
        return (final_positive, final_negative,)

class PairConditioningSetDefaultAndCombine:
    NodeId = 'PairConditioningSetDefaultCombine'
    NodeName = 'Cond Pair Set Default Combine'
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "positive_DEFAULT": ("CONDITIONING",),
                "negative_DEFAULT": ("CONDITIONING",),
            },
            "optional": {
                "hooks": ("HOOKS",),
            }
        }
    
    EXPERIMENTAL = True
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    CATEGORY = "advanced/hooks/cond pair"
    FUNCTION = "set_default_and_combine"

    def set_default_and_combine(self, positive, negative, positive_DEFAULT, negative_DEFAULT,
                                hooks: comfy.hooks.HookGroup=None):
        final_positive, final_negative = comfy.hooks.set_default_conds_and_combine(conds=[positive, negative], new_conds=[positive_DEFAULT, negative_DEFAULT],
                                                                                   hooks=hooks)
        return (final_positive, final_negative)
    
class ConditioningSetDefaultAndCombine:
    NodeId = 'ConditioningSetDefaultCombine'
    NodeName = 'Cond Set Default Combine'
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "cond": ("CONDITIONING",),
                "cond_DEFAULT": ("CONDITIONING",),
            },
            "optional": {
                "hooks": ("HOOKS",),
            }
        }

    EXPERIMENTAL = True
    RETURN_TYPES = ("CONDITIONING",)
    CATEGORY = "advanced/hooks/cond single"
    FUNCTION = "set_default_and_combine"

    def set_default_and_combine(self, cond, cond_DEFAULT,
                                hooks: comfy.hooks.HookGroup=None):
        (final_conditioning,) = comfy.hooks.set_default_conds_and_combine(conds=[cond], new_conds=[cond_DEFAULT],
                                                                        hooks=hooks)
        return (final_conditioning,)
    
class SetClipHooks:
    NodeId = 'SetClipHooks'
    NodeName = 'Set CLIP Hooks'
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP",),
                "apply_to_conds": ("BOOLEAN", {"default": True}),
                "schedule_clip": ("BOOLEAN", {"default": False})
            },
            "optional": {
                "hooks": ("HOOKS",)
            }
        }
    
    EXPERIMENTAL = True
    RETURN_TYPES = ("CLIP",)
    CATEGORY = "advanced/hooks/clip"
    FUNCTION = "apply_hooks"

    def apply_hooks(self, clip: 'CLIP', schedule_clip: bool, apply_to_conds: bool, hooks: comfy.hooks.HookGroup=None):
        if hooks is not None:
            clip = clip.clone()
            if apply_to_conds:
                clip.apply_hooks_to_conds = hooks
            clip.patcher.forced_hooks = hooks.clone()
            clip.use_clip_schedule = schedule_clip
            if not clip.use_clip_schedule:
                clip.patcher.forced_hooks.set_keyframes_on_hooks(None)
            clip.patcher.register_all_hook_patches(hooks.get_dict_repr(), comfy.hooks.EnumWeightTarget.Clip)
        return (clip,)

class ConditioningTimestepsRange:
    NodeId = 'ConditioningTimestepsRange'
    NodeName = 'Timesteps Range'
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001})
            },
        }
    
    EXPERIMENTAL = True
    RETURN_TYPES = ("TIMESTEPS_RANGE", "TIMESTEPS_RANGE", "TIMESTEPS_RANGE")
    RETURN_NAMES = ("TIMESTEPS_RANGE", "BEFORE_RANGE", "AFTER_RANGE")
    CATEGORY = "advanced/hooks"
    FUNCTION = "create_range"

    def create_range(self, start_percent: float, end_percent: float):
        return ((start_percent, end_percent), (0.0, start_percent), (end_percent, 1.0))
#------------------------------------------
###########################################


###########################################
# Create Hooks
#------------------------------------------
class CreateHookLora:
    NodeId = 'CreateHookLora'
    NodeName = 'Create Hook LoRA'
    def __init__(self):
        self.loaded_lora = None
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "lora_name": (folder_paths.get_filename_list("loras"), ),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01}),
                "strength_clip": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01}),
            },
            "optional": {
                "prev_hooks": ("HOOKS",)
            }
        }
    
    EXPERIMENTAL = True
    RETURN_TYPES = ("HOOKS",)
    CATEGORY = "advanced/hooks/create"
    FUNCTION = "create_hook"

    def create_hook(self, lora_name: str, strength_model: float, strength_clip: float, prev_hooks: comfy.hooks.HookGroup=None):
        if prev_hooks is None:
            prev_hooks = comfy.hooks.HookGroup()
        prev_hooks.clone()

        if strength_model == 0 and strength_clip == 0:
            return (prev_hooks,)
        
        lora_path = folder_paths.get_full_path("loras", lora_name)
        lora = None
        if self.loaded_lora is not None:
            if self.loaded_lora[0] == lora_path:
                lora = self.loaded_lora[1]
            else:
                temp = self.loaded_lora
                self.loaded_lora = None
                del temp
        
        if lora is None:
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
            self.loaded_lora = (lora_path, lora)

        hooks = comfy.hooks.create_hook_lora(lora=lora, strength_model=strength_model, strength_clip=strength_clip)
        return (prev_hooks.clone_and_combine(hooks),)

class CreateHookLoraModelOnly(CreateHookLora):
    NodeId = 'CreateHookLoraModelOnly'
    NodeName = 'Create Hook LoRA (MO)'
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "lora_name": (folder_paths.get_filename_list("loras"), ),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01}),
            },
            "optional": {
                "prev_hooks": ("HOOKS",)
            }
        }
    
    EXPERIMENTAL = True
    RETURN_TYPES = ("HOOKS",)
    CATEGORY = "advanced/hooks/create"
    FUNCTION = "create_hook_model_only"

    def create_hook_model_only(self, lora_name: str, strength_model: float, prev_hooks: comfy.hooks.HookGroup=None):
        return self.create_hook(lora_name=lora_name, strength_model=strength_model, strength_clip=0, prev_hooks=prev_hooks)

class CreateHookModelAsLora:
    NodeId = 'CreateHookModelAsLora'
    NodeName = 'Create Hook Model as LoRA'

    def __init__(self):
        # when not None, will be in following format:
        # (ckpt_path: str, weights_model: dict, weights_clip: dict)
        self.loaded_weights = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01}),
                "strength_clip": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01}),
            },
            "optional": {
                "prev_hooks": ("HOOKS",)
            }
        }
    
    EXPERIMENTAL = True
    RETURN_TYPES = ("HOOKS",)
    CATEGORY = "advanced/hooks/create"
    FUNCTION = "create_hook"

    def create_hook(self, ckpt_name: str, strength_model: float, strength_clip: float,
                    prev_hooks: comfy.hooks.HookGroup=None):
        if prev_hooks is None:
            prev_hooks = comfy.hooks.HookGroup()
        prev_hooks.clone()

        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        weights_model = None
        weights_clip = None
        if self.loaded_weights is not None:
            if self.loaded_weights[0] == ckpt_path:
                weights_model = self.loaded_weights[1]
                weights_clip = self.loaded_weights[2]
            else:
                temp = self.loaded_weights
                self.loaded_weights = None
                del temp
        
        if weights_model is None:
            out = comfy.sd.load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True, embedding_directory=folder_paths.get_folder_paths("embeddings"))
            weights_model = comfy.hooks.get_patch_weights_from_model(out[0])
            weights_clip = comfy.hooks.get_patch_weights_from_model(out[1].patcher if out[1] else out[1])
            self.loaded_weights = (ckpt_path, weights_model, weights_clip)

        hooks = comfy.hooks.create_hook_model_as_lora(weights_model=weights_model, weights_clip=weights_clip,
                                                      strength_model=strength_model, strength_clip=strength_clip)
        return (prev_hooks.clone_and_combine(hooks),)

class CreateHookModelAsLoraModelOnly(CreateHookModelAsLora):
    NodeId = 'CreateHookModelAsLoraModelOnly'
    NodeName = 'Create Hook Model as LoRA (MO)'
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01}),
            },
            "optional": {
                "prev_hooks": ("HOOKS",)
            }
        }
    
    EXPERIMENTAL = True
    RETURN_TYPES = ("HOOKS",)
    CATEGORY = "advanced/hooks/create"
    FUNCTION = "create_hook_model_only"

    def create_hook_model_only(self, ckpt_name: str, strength_model: float,
                               prev_hooks: comfy.hooks.HookGroup=None):
        return self.create_hook(ckpt_name=ckpt_name, strength_model=strength_model, strength_clip=0.0, prev_hooks=prev_hooks)
#------------------------------------------
###########################################


###########################################
# Schedule Hooks
#------------------------------------------
class SetHookKeyframes:
    NodeId = 'SetHookKeyframes'
    NodeName = 'Set Hook Keyframes'
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "hooks": ("HOOKS",),
            },
            "optional": {
                "hook_kf": ("HOOK_KEYFRAMES",),
            }
        }
    
    EXPERIMENTAL = True
    RETURN_TYPES = ("HOOKS",)
    CATEGORY = "advanced/hooks/scheduling"
    FUNCTION = "set_hook_keyframes"

    def set_hook_keyframes(self, hooks: comfy.hooks.HookGroup, hook_kf: comfy.hooks.HookKeyframeGroup=None):
        if hook_kf is not None:
            hooks = hooks.clone()
            hooks.set_keyframes_on_hooks(hook_kf=hook_kf)
        return (hooks,)

class CreateHookKeyframe:
    NodeId = 'CreateHookKeyframe'
    NodeName = 'Create Hook Keyframe'
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "strength_mult": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
            },
            "optional": {
                "prev_hook_kf": ("HOOK_KEYFRAMES",),
            }
        }
    
    EXPERIMENTAL = True
    RETURN_TYPES = ("HOOK_KEYFRAMES",)
    RETURN_NAMES = ("HOOK_KF",)
    CATEGORY = "advanced/hooks/scheduling"
    FUNCTION = "create_hook_keyframe"

    def create_hook_keyframe(self, strength_mult: float, start_percent: float, prev_hook_kf: comfy.hooks.HookKeyframeGroup=None):
        if prev_hook_kf is None:
            prev_hook_kf = comfy.hooks.HookKeyframeGroup()
        prev_hook_kf = prev_hook_kf.clone()
        keyframe = comfy.hooks.HookKeyframe(strength=strength_mult, start_percent=start_percent)
        prev_hook_kf.add(keyframe)
        return (prev_hook_kf,)

class CreateHookKeyframesInterpolated:
    NodeId = 'CreateHookKeyframesInterpolated'
    NodeName = 'Create Hook Keyframes Interp.'
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "strength_start": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "strength_end": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "interpolation": (comfy.hooks.InterpolationMethod._LIST, ),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "keyframes_count": ("INT", {"default": 5, "min": 2, "max": 100, "step": 1}),
                "print_keyframes": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "prev_hook_kf": ("HOOK_KEYFRAMES",),
            },
        }
    
    EXPERIMENTAL = True
    RETURN_TYPES = ("HOOK_KEYFRAMES",)
    RETURN_NAMES = ("HOOK_KF",)
    CATEGORY = "advanced/hooks/scheduling"
    FUNCTION = "create_hook_keyframes"

    def create_hook_keyframes(self, strength_start: float, strength_end: float, interpolation: str,
                              start_percent: float, end_percent: float, keyframes_count: int,
                              print_keyframes=False, prev_hook_kf: comfy.hooks.HookKeyframeGroup=None):
        if prev_hook_kf is None:
            prev_hook_kf = comfy.hooks.HookKeyframeGroup()
        prev_hook_kf = prev_hook_kf.clone()
        percents = comfy.hooks.InterpolationMethod.get_weights(num_from=start_percent, num_to=end_percent, length=keyframes_count,
                                                               method=comfy.hooks.InterpolationMethod.LINEAR)
        strengths = comfy.hooks.InterpolationMethod.get_weights(num_from=strength_start, num_to=strength_end, length=keyframes_count, method=interpolation)

        is_first = True
        for percent, strength in zip(percents, strengths):
            guarantee_steps = 0
            if is_first:
                guarantee_steps = 1
                is_first = False
            prev_hook_kf.add(comfy.hooks.HookKeyframe(strength=strength, start_percent=percent, guarantee_steps=guarantee_steps))
            if print_keyframes:
                print(f"Hook Keyframe - start_percent:{percent} = {strength}")
        return (prev_hook_kf,)

class CreateHookKeyframesFromFloats:
    NodeId = 'CreateHookKeyframesFromFloats'
    NodeName = 'Create Hook Keyframes From Floats'
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "floats_strength": ("FLOATS", {"default": -1, "min": -1, "step": 0.001, "forceInput": True}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "print_keyframes": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "prev_hook_kf": ("HOOK_KEYFRAMES",),
            }
        }
    
    EXPERIMENTAL = True
    RETURN_TYPES = ("HOOK_KEYFRAMES",)
    RETURN_NAMES = ("HOOK_KF",)
    CATEGORY = "advanced/hooks/scheduling"
    FUNCTION = "create_hook_keyframes"

    def create_hook_keyframes(self, floats_strength: Union[float, list[float]],
                              start_percent: float, end_percent: float,
                              prev_hook_kf: comfy.hooks.HookKeyframeGroup=None, print_keyframes=False):
        if prev_hook_kf is None:
            prev_hook_kf = comfy.hooks.HookKeyframeGroup()
        prev_hook_kf = prev_hook_kf.clone()
        if type(floats_strength) in (float, int):
            floats_strength = [float(floats_strength)]
        elif isinstance(floats_strength, Iterable):
            pass
        else:
            raise Exception(f"floats_strength must be either an iterable input or a float, but was{type(floats_strength).__repr__}.")
        percents = comfy.hooks.InterpolationMethod.get_weights(num_from=start_percent, num_to=end_percent, length=len(floats_strength),
                                                               method=comfy.hooks.InterpolationMethod.LINEAR)
        
        is_first = True
        for percent, strength in zip(percents, floats_strength):
            guarantee_steps = 0
            if is_first:
                guarantee_steps = 1
                is_first = False
            prev_hook_kf.add(comfy.hooks.HookKeyframe(strength=strength, start_percent=percent, guarantee_steps=guarantee_steps))
            if print_keyframes:
                print(f"Hook Keyframe - start_percent:{percent} = {strength}")
        return (prev_hook_kf,)
#------------------------------------------
###########################################


class SetModelHooksOnCond:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "hooks": ("HOOKS",),
            },
        }
    
    EXPERIMENTAL = True
    RETURN_TYPES = ("CONDITIONING",)
    CATEGORY = "advanced/hooks/manual"
    FUNCTION = "attach_hook"

    def attach_hook(self, conditioning, hooks: comfy.hooks.HookGroup):
        return (comfy.hooks.set_hooks_for_conditioning(conditioning, hooks),)


###########################################
# Combine Hooks
#------------------------------------------
class CombineHooks:
    NodeId = 'CombineHooks2'
    NodeName = 'Combine Hooks [2]'
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
            },
            "optional": {
                "hooks_A": ("HOOKS",),
                "hooks_B": ("HOOKS",),
            }
        }
    
    EXPERIMENTAL = True
    RETURN_TYPES = ("HOOKS",)
    CATEGORY = "advanced/hooks/combine"
    FUNCTION = "combine_hooks"

    def combine_hooks(self,
                      hooks_A: comfy.hooks.HookGroup=None,
                      hooks_B: comfy.hooks.HookGroup=None):
        candidates = [hooks_A, hooks_B]
        return (comfy.hooks.HookGroup.combine_all_hooks(candidates),)

class CombineHooksFour:
    NodeId = 'CombineHooks4'
    NodeName = 'Combine Hooks [4]'
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
            },
            "optional": {
                "hooks_A": ("HOOKS",),
                "hooks_B": ("HOOKS",),
                "hooks_C": ("HOOKS",),
                "hooks_D": ("HOOKS",),
            }
        }
    
    EXPERIMENTAL = True
    RETURN_TYPES = ("HOOKS",)
    CATEGORY = "advanced/hooks/combine"
    FUNCTION = "combine_hooks"

    def combine_hooks(self,
                      hooks_A: comfy.hooks.HookGroup=None,
                      hooks_B: comfy.hooks.HookGroup=None,
                      hooks_C: comfy.hooks.HookGroup=None,
                      hooks_D: comfy.hooks.HookGroup=None):
        candidates = [hooks_A, hooks_B, hooks_C, hooks_D]
        return (comfy.hooks.HookGroup.combine_all_hooks(candidates),)

class CombineHooksEight:
    NodeId = 'CombineHooks8'
    NodeName = 'Combine Hooks [8]'
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
            },
            "optional": {
                "hooks_A": ("HOOKS",),
                "hooks_B": ("HOOKS",),
                "hooks_C": ("HOOKS",),
                "hooks_D": ("HOOKS",),
                "hooks_E": ("HOOKS",),
                "hooks_F": ("HOOKS",),
                "hooks_G": ("HOOKS",),
                "hooks_H": ("HOOKS",),
            }
        }
    
    EXPERIMENTAL = True
    RETURN_TYPES = ("HOOKS",)
    CATEGORY = "advanced/hooks/combine"
    FUNCTION = "combine_hooks"

    def combine_hooks(self,
                      hooks_A: comfy.hooks.HookGroup=None,
                      hooks_B: comfy.hooks.HookGroup=None,
                      hooks_C: comfy.hooks.HookGroup=None,
                      hooks_D: comfy.hooks.HookGroup=None,
                      hooks_E: comfy.hooks.HookGroup=None,
                      hooks_F: comfy.hooks.HookGroup=None,
                      hooks_G: comfy.hooks.HookGroup=None,
                      hooks_H: comfy.hooks.HookGroup=None):
        candidates = [hooks_A, hooks_B, hooks_C, hooks_D, hooks_E, hooks_F, hooks_G, hooks_H]
        return (comfy.hooks.HookGroup.combine_all_hooks(candidates),)
#------------------------------------------
###########################################

node_list = [
    # Create
    CreateHookLora,
    CreateHookLoraModelOnly,
    CreateHookModelAsLora,
    CreateHookModelAsLoraModelOnly,
    # Scheduling
    SetHookKeyframes,
    CreateHookKeyframe,
    CreateHookKeyframesInterpolated,
    CreateHookKeyframesFromFloats,
    # Combine
    CombineHooks,
    CombineHooksFour,
    CombineHooksEight,
    # Attach
    ConditioningSetProperties,
    ConditioningSetPropertiesAndCombine,
    PairConditioningSetProperties,
    PairConditioningSetPropertiesAndCombine,
    ConditioningSetDefaultAndCombine,
    PairConditioningSetDefaultAndCombine,
    PairConditioningCombine,
    SetClipHooks,
    # Other
    ConditioningTimestepsRange,
]
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

for node in node_list:
    NODE_CLASS_MAPPINGS[node.NodeId] = node
    NODE_DISPLAY_NAME_MAPPINGS[node.NodeId] = node.NodeName
