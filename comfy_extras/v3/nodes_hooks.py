from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import TYPE_CHECKING, Union

import torch

from comfy_api.latest import io, resources

if TYPE_CHECKING:
    from comfy.sd import CLIP

import comfy.hooks
import comfy.sd
import comfy.utils
import folder_paths


###########################################
# Mask, Combine, and Hook Conditioning
#------------------------------------------
class PairConditioningSetProperties(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PairConditioningSetProperties_V3",
            display_name="Cond Pair Set Props _V3",
            category="advanced/hooks/cond pair",
            is_experimental=True,
            inputs=[
                io.Conditioning.Input("positive_NEW"),
                io.Conditioning.Input("negative_NEW"),
                io.Float.Input("strength", default=1.0, min=0.0, max=10.0, step=0.01),
                io.Combo.Input("set_cond_area", options=["default", "mask bounds"]),
                io.Mask.Input("mask", optional=True),
                io.Hooks.Input("hooks", optional=True),
                io.TimestepsRange.Input("timesteps", optional=True),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
            ]
        )

    @classmethod
    def execute(
        cls, positive_NEW, negative_NEW, strength: float, set_cond_area: str, mask: torch.Tensor=None, hooks: comfy.hooks.HookGroup=None, timesteps: tuple=None
    ):
        final_positive, final_negative = comfy.hooks.set_conds_props(conds=[positive_NEW, negative_NEW],
                                                                    strength=strength, set_cond_area=set_cond_area,
                                                                    mask=mask, hooks=hooks, timesteps_range=timesteps)
        return io.NodeOutput(final_positive, final_negative)


class PairConditioningSetPropertiesAndCombine(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PairConditioningSetPropertiesAndCombine_V3",
            display_name="Cond Pair Set Props Combine _V3",
            category="advanced/hooks/cond pair",
            is_experimental=True,
            inputs=[
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Conditioning.Input("positive_NEW"),
                io.Conditioning.Input("negative_NEW"),
                io.Float.Input("strength", default=1.0, min=0.0, max=10.0, step=0.01),
                io.Combo.Input("set_cond_area", options=["default", "mask bounds"]),
                io.Mask.Input("mask", optional=True),
                io.Hooks.Input("hooks", optional=True),
                io.TimestepsRange.Input("timesteps", optional=True),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
            ]
        )

    @classmethod
    def execute(
        cls, positive, negative, positive_NEW, negative_NEW, strength: float, set_cond_area: str, mask: torch.Tensor=None, hooks: comfy.hooks.HookGroup=None, timesteps: tuple=None
    ):
        final_positive, final_negative = comfy.hooks.set_conds_props_and_combine(conds=[positive, negative], new_conds=[positive_NEW, negative_NEW],
                                                                                strength=strength, set_cond_area=set_cond_area,
                                                                                mask=mask, hooks=hooks, timesteps_range=timesteps)
        return io.NodeOutput(final_positive, final_negative)


class ConditioningSetProperties(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ConditioningSetProperties_V3",
            display_name="Cond Set Props _V3",
            category="advanced/hooks/cond single",
            is_experimental=True,
            inputs=[
                io.Conditioning.Input("cond_NEW"),
                io.Float.Input("strength", default=1.0, min=0.0, max=10.0, step=0.01),
                io.Combo.Input("set_cond_area", options=["default", "mask bounds"]),
                io.Mask.Input("mask", optional=True),
                io.Hooks.Input("hooks", optional=True),
                io.TimestepsRange.Input("timesteps", optional=True),
            ],
            outputs=[
                io.Conditioning.Output(),
            ]
        )

    @classmethod
    def execute(
        cls, cond_NEW, strength: float, set_cond_area: str, mask: torch.Tensor=None, hooks: comfy.hooks.HookGroup=None, timesteps: tuple=None
    ):
        (final_cond,) = comfy.hooks.set_conds_props(conds=[cond_NEW],
                                                   strength=strength, set_cond_area=set_cond_area,
                                                   mask=mask, hooks=hooks, timesteps_range=timesteps)
        return io.NodeOutput(final_cond)


class ConditioningSetPropertiesAndCombine(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ConditioningSetPropertiesAndCombine_V3",
            display_name="Cond Set Props Combine _V3",
            category="advanced/hooks/cond single",
            is_experimental=True,
            inputs=[
                io.Conditioning.Input("cond"),
                io.Conditioning.Input("cond_NEW"),
                io.Float.Input("strength", default=1.0, min=0.0, max=10.0, step=0.01),
                io.Combo.Input("set_cond_area", options=["default", "mask bounds"]),
                io.Mask.Input("mask", optional=True),
                io.Hooks.Input("hooks", optional=True),
                io.TimestepsRange.Input("timesteps", optional=True),
            ],
            outputs=[
                io.Conditioning.Output(),
            ]
        )

    @classmethod
    def execute(
        cls, cond, cond_NEW, strength: float, set_cond_area: str, mask: torch.Tensor=None, hooks: comfy.hooks.HookGroup=None, timesteps: tuple=None
    ):
        (final_cond,) = comfy.hooks.set_conds_props_and_combine(conds=[cond], new_conds=[cond_NEW],
                                                               strength=strength, set_cond_area=set_cond_area,
                                                               mask=mask, hooks=hooks, timesteps_range=timesteps)
        return io.NodeOutput(final_cond)


class PairConditioningCombine(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PairConditioningCombine_V3",
            display_name="Cond Pair Combine _V3",
            category="advanced/hooks/cond pair",
            is_experimental=True,
            inputs=[
                io.Conditioning.Input("positive_A"),
                io.Conditioning.Input("negative_A"),
                io.Conditioning.Input("positive_B"),
                io.Conditioning.Input("negative_B"),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
            ]
        )

    @classmethod
    def execute(cls, positive_A, negative_A, positive_B, negative_B):
        final_positive, final_negative = comfy.hooks.set_conds_props_and_combine(conds=[positive_A, negative_A], new_conds=[positive_B, negative_B],)
        return io.NodeOutput(final_positive, final_negative)


class PairConditioningSetDefaultAndCombine(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PairConditioningSetDefaultCombine_V3",
            display_name="Cond Pair Set Default Combine _V3",
            category="advanced/hooks/cond pair",
            is_experimental=True,
            inputs=[
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Conditioning.Input("positive_DEFAULT"),
                io.Conditioning.Input("negative_DEFAULT"),
                io.Hooks.Input("hooks", optional=True),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
            ]
        )

    @classmethod
    def execute(cls, positive, negative, positive_DEFAULT, negative_DEFAULT, hooks: comfy.hooks.HookGroup=None):
        final_positive, final_negative = comfy.hooks.set_default_conds_and_combine(conds=[positive, negative], new_conds=[positive_DEFAULT, negative_DEFAULT],
                                                                                   hooks=hooks)
        return io.NodeOutput(final_positive, final_negative)


class ConditioningSetDefaultAndCombine(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ConditioningSetDefaultCombine_V3",
            display_name="Cond Set Default Combine _V3",
            category="advanced/hooks/cond single",
            is_experimental=True,
            inputs=[
                io.Conditioning.Input("cond"),
                io.Conditioning.Input("cond_DEFAULT"),
                io.Hooks.Input("hooks", optional=True),
            ],
            outputs=[
                io.Conditioning.Output(),
            ]
        )

    @classmethod
    def execute(cls, cond, cond_DEFAULT, hooks: comfy.hooks.HookGroup=None):
        (final_conditioning,) = comfy.hooks.set_default_conds_and_combine(conds=[cond], new_conds=[cond_DEFAULT],
                                                                        hooks=hooks)
        return io.NodeOutput(final_conditioning)


class SetClipHooks(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SetClipHooks_V3",
            display_name="Set CLIP Hooks _V3",
            category="advanced/hooks/clip",
            is_experimental=True,
            inputs=[
                io.Clip.Input("clip"),
                io.Boolean.Input("apply_to_conds", default=True),
                io.Boolean.Input("schedule_clip", default=False),
                io.Hooks.Input("hooks", optional=True),
            ],
            outputs=[
                io.Clip.Output(),
            ]
        )

    @classmethod
    def execute(cls, clip: CLIP, schedule_clip: bool, apply_to_conds: bool, hooks: comfy.hooks.HookGroup=None):
        if hooks is not None:
            clip = clip.clone()
            if apply_to_conds:
                clip.apply_hooks_to_conds = hooks
            clip.patcher.forced_hooks = hooks.clone()
            clip.use_clip_schedule = schedule_clip
            if not clip.use_clip_schedule:
                clip.patcher.forced_hooks.set_keyframes_on_hooks(None)
            clip.patcher.register_all_hook_patches(hooks, comfy.hooks.create_target_dict(comfy.hooks.EnumWeightTarget.Clip))
        return io.NodeOutput(clip)


class ConditioningTimestepsRange(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ConditioningTimestepsRange_V3",
            display_name="Timesteps Range _V3",
            category="advanced/hooks",
            is_experimental=True,
            inputs=[
                io.Float.Input("start_percent", default=0.0, min=0.0, max=1.0, step=0.001),
                io.Float.Input("end_percent", default=1.0, min=0.0, max=1.0, step=0.001),
            ],
            outputs=[
                io.TimestepsRange.Output(display_name="TIMESTEPS_RANGE"),
                io.TimestepsRange.Output(display_name="BEFORE_RANGE"),
                io.TimestepsRange.Output(display_name="AFTER_RANGE"),
            ]
        )

    @classmethod
    def execute(cls, start_percent: float, end_percent: float):
        return io.NodeOutput((start_percent, end_percent), (0.0, start_percent), (end_percent, 1.0))
#------------------------------------------
###########################################


###########################################
# Create Hooks
#------------------------------------------
class CreateHookLora(io.ComfyNode):
    # LOADED_LORA = None

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CreateHookLora_V3",
            display_name="Create Hook LoRA _V3",
            category="advanced/hooks/create",
            is_experimental=True,
            inputs=[
                io.Combo.Input("lora_name", options=folder_paths.get_filename_list("loras")),
                io.Float.Input("strength_model", default=1.0, min=-20.0, max=20.0, step=0.01),
                io.Float.Input("strength_clip", default=1.0, min=-20.0, max=20.0, step=0.01),
                io.Hooks.Input("prev_hooks", optional=True),
            ],
            outputs=[
                io.Hooks.Output(),
            ]
        )

    @classmethod
    def execute(cls, lora_name: str, strength_model: float, strength_clip: float, prev_hooks: comfy.hooks.HookGroup=None):
        if prev_hooks is None:
            prev_hooks = comfy.hooks.HookGroup()
        prev_hooks.clone()

        if strength_model == 0 and strength_clip == 0:
            return io.NodeOutput(prev_hooks)

        lora = cls.resources.get(resources.TorchDictFolderFilename("loras", lora_name))
        # TODO: remove commented code code
        # lora_path = folder_paths.get_full_path("loras", lora_name)
        # lora = None
        # if cls.LOADED_LORA is not None:
        #     if cls.LOADED_LORA[0] == lora_path:
        #         lora = cls.LOADED_LORA[1]
        #     else:
        #         temp = cls.LOADED_LORA
        #         cls.LOADED_LORA = None
        #         del temp
        #
        # if lora is None:
        #     lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
        #     cls.LOADED_LORA = (lora_path, lora)

        hooks = comfy.hooks.create_hook_lora(lora=lora, strength_model=strength_model, strength_clip=strength_clip)
        return io.NodeOutput(prev_hooks.clone_and_combine(hooks))


class CreateHookLoraModelOnly(CreateHookLora):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CreateHookLoraModelOnly_V3",
            display_name="Create Hook LoRA (MO) _V3",
            category="advanced/hooks/create",
            is_experimental=True,
            inputs=[
                io.Combo.Input("lora_name", options=folder_paths.get_filename_list("loras")),
                io.Float.Input("strength_model", default=1.0, min=-20.0, max=20.0, step=0.01),
                io.Hooks.Input("prev_hooks", optional=True),
            ],
            outputs=[
                io.Hooks.Output(),
            ]
        )

    @classmethod
    def execute(cls, lora_name: str, strength_model: float, prev_hooks: comfy.hooks.HookGroup=None):
        return super().execute(lora_name=lora_name, strength_model=strength_model, strength_clip=0, prev_hooks=prev_hooks)


class CreateHookModelAsLora(io.ComfyNode):
    # when not None, will be in following format:
    # (ckpt_path: str, weights_model: dict, weights_clip: dict)
    LOADED_WEIGHTS = None

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CreateHookModelAsLora_V3",
            display_name="Create Hook Model as LoRA _V3",
            category="advanced/hooks/create",
            is_experimental=True,
            inputs=[
                io.Combo.Input("ckpt_name", options=folder_paths.get_filename_list("checkpoints")),
                io.Float.Input("strength_model", default=1.0, min=-20.0, max=20.0, step=0.01),
                io.Float.Input("strength_clip", default=1.0, min=-20.0, max=20.0, step=0.01),
                io.Hooks.Input("prev_hooks", optional=True),
            ],
            outputs=[
                io.Hooks.Output(),
            ]
        )

    @classmethod
    def execute(cls, ckpt_name: str, strength_model: float, strength_clip: float, prev_hooks: comfy.hooks.HookGroup=None):
        if prev_hooks is None:
            prev_hooks = comfy.hooks.HookGroup()
        prev_hooks.clone()

        # TODO: can we store using the resource custom key-pairs? Should we add support for that?
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        weights_model = None
        weights_clip = None
        if cls.LOADED_WEIGHTS is not None:
            if cls.LOADED_WEIGHTS[0] == ckpt_path:
                weights_model = cls.LOADED_WEIGHTS[1]
                weights_clip = cls.LOADED_WEIGHTS[2]
            else:
                temp = cls.LOADED_WEIGHTS
                cls.LOADED_WEIGHTS = None
                del temp

        if weights_model is None:
            out = comfy.sd.load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True, embedding_directory=folder_paths.get_folder_paths("embeddings"))
            weights_model = comfy.hooks.get_patch_weights_from_model(out[0])
            weights_clip = comfy.hooks.get_patch_weights_from_model(out[1].patcher if out[1] else out[1])
            cls.LOADED_WEIGHTS = (ckpt_path, weights_model, weights_clip)

        hooks = comfy.hooks.create_hook_model_as_lora(weights_model=weights_model, weights_clip=weights_clip,
                                                      strength_model=strength_model, strength_clip=strength_clip)
        return io.NodeOutput(prev_hooks.clone_and_combine(hooks))


class CreateHookModelAsLoraModelOnly(CreateHookModelAsLora):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CreateHookModelAsLoraModelOnly_V3",
            display_name="Create Hook Model as LoRA (MO) _V3",
            category="advanced/hooks/create",
            is_experimental=True,
            inputs=[
                io.Combo.Input("ckpt_name", options=folder_paths.get_filename_list("checkpoints")),
                io.Float.Input("strength_model", default=1.0, min=-20.0, max=20.0, step=0.01),
                io.Hooks.Input("prev_hooks", optional=True),
            ],
            outputs=[
                io.Hooks.Output(),
            ]
        )

    @classmethod
    def execute(cls, ckpt_name: str, strength_model: float, prev_hooks: comfy.hooks.HookGroup=None):
        return super().execute(ckpt_name=ckpt_name, strength_model=strength_model, strength_clip=0.0, prev_hooks=prev_hooks)
#------------------------------------------
###########################################


###########################################
# Schedule Hooks
#------------------------------------------

class SetHookKeyframes(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SetHookKeyframes_V3",
            display_name="Set Hook Keyframes _V3",
            category="advanced/hooks/scheduling",
            is_experimental=True,
            inputs=[
                io.Hooks.Input("hooks"),
                io.HookKeyframes.Input("hook_kf", optional=True),
            ],
            outputs=[
                io.Hooks.Output(),
            ]
        )

    @classmethod
    def execute(cls, hooks: comfy.hooks.HookGroup, hook_kf: comfy.hooks.HookKeyframeGroup=None):
        if hook_kf is not None:
            hooks = hooks.clone()
            hooks.set_keyframes_on_hooks(hook_kf=hook_kf)
        return io.NodeOutput(hooks)


class CreateHookKeyframe(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CreateHookKeyframe_V3",
            display_name="Create Hook Keyframe _V3",
            category="advanced/hooks/scheduling",
            is_experimental=True,
            inputs=[
                io.Float.Input("strength_mult", default=1.0, min=-20.0, max=20.0, step=0.01),
                io.Float.Input("start_percent", default=0.0, min=0.0, max=1.0, step=0.001),
                io.HookKeyframes.Input("prev_hook_kf", optional=True),
            ],
            outputs=[
                io.HookKeyframes.Output(display_name="HOOK_KF"),
            ]
        )

    @classmethod
    def execute(cls, strength_mult: float, start_percent: float, prev_hook_kf: comfy.hooks.HookKeyframeGroup=None):
        if prev_hook_kf is None:
            prev_hook_kf = comfy.hooks.HookKeyframeGroup()
        prev_hook_kf = prev_hook_kf.clone()
        keyframe = comfy.hooks.HookKeyframe(strength=strength_mult, start_percent=start_percent)
        prev_hook_kf.add(keyframe)
        return io.NodeOutput(prev_hook_kf)


class CreateHookKeyframesInterpolated(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CreateHookKeyframesInterpolated_V3",
            display_name="Create Hook Keyframes Interp. _V3",
            category="advanced/hooks/scheduling",
            is_experimental=True,
            inputs=[
                io.Float.Input("strength_start", default=1.0, min=0.0, max=10.0, step=0.001),
                io.Float.Input("strength_end", default=1.0, min=0.0, max=10.0, step=0.001),
                io.Combo.Input("interpolation", options=comfy.hooks.InterpolationMethod._LIST),
                io.Float.Input("start_percent", default=0.0, min=0.0, max=1.0, step=0.001),
                io.Float.Input("end_percent", default=1.0, min=0.0, max=1.0, step=0.001),
                io.Int.Input("keyframes_count", default=5, min=2, max=100, step=1),
                io.Boolean.Input("print_keyframes", default=False),
                io.HookKeyframes.Input("prev_hook_kf", optional=True),
            ],
            outputs=[
                io.HookKeyframes.Output(display_name="HOOK_KF"),
            ]
        )

    @classmethod
    def execute(cls, strength_start: float, strength_end: float, interpolation: str,
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
                logging.info(f"Hook Keyframe - start_percent:{percent} = {strength}")
        return io.NodeOutput(prev_hook_kf)


class CreateHookKeyframesFromFloats(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CreateHookKeyframesFromFloats_V3",
            display_name="Create Hook Keyframes From Floats _V3",
            category="advanced/hooks/scheduling",
            is_experimental=True,
            inputs=[
                io.Float.Input("floats_strength", default=-1, min=-1, step=0.001, force_input=True),
                io.Float.Input("start_percent", default=0.0, min=0.0, max=1.0, step=0.001),
                io.Float.Input("end_percent", default=1.0, min=0.0, max=1.0, step=0.001),
                io.Boolean.Input("print_keyframes", default=False),
                io.HookKeyframes.Input("prev_hook_kf", optional=True),
            ],
            outputs=[
                io.HookKeyframes.Output(display_name="HOOK_KF"),
            ]
        )

    @classmethod
    def execute(cls, floats_strength: Union[float, list[float]],
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
                logging.info(f"Hook Keyframe - start_percent:{percent} = {strength}")
        return io.NodeOutput(prev_hook_kf)

#------------------------------------------
###########################################


class SetModelHooksOnCond(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SetModelHooksOnCond_V3",
            category="advanced/hooks/manual",
            is_experimental=True,
            inputs=[
                io.Conditioning.Input("conditioning"),
                io.Hooks.Input("hooks"),
            ],
            outputs=[
                io.Conditioning.Output(),
            ]
        )

    @classmethod
    def execute(cls, conditioning, hooks: comfy.hooks.HookGroup):
        return io.NodeOutput(comfy.hooks.set_hooks_for_conditioning(conditioning, hooks))


###########################################
# Combine Hooks
#------------------------------------------
class CombineHooks(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CombineHooks2_V3",
            display_name="Combine Hooks [2] _V3",
            category="advanced/hooks/combine",
            is_experimental=True,
            inputs=[
                io.Hooks.Input("hooks_A"),
                io.Hooks.Input("hooks_B"),
            ],
            outputs=[
                io.Hooks.Output(),
            ],
            hidden=[]
        )

    @classmethod
    def execute(
        cls,
        hooks_A: comfy.hooks.HookGroup=None,
        hooks_B: comfy.hooks.HookGroup=None,
    ):
        candidates = [hooks_A, hooks_B]
        return io.NodeOutput(comfy.hooks.HookGroup.combine_all_hooks(candidates))


class CombineHooksFour(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CombineHooks4_V3",
            display_name="Combine Hooks [4] _V3",
            category="advanced/hooks/combine",
            is_experimental=True,
            inputs=[
                io.Hooks.Input("hooks_A"),
                io.Hooks.Input("hooks_B"),
                io.Hooks.Input("hooks_C"),
                io.Hooks.Input("hooks_D"),
            ],
            outputs=[
                io.Hooks.Output(),
            ],
            hidden=[]
        )

    @classmethod
    def execute(
        cls,
        hooks_A: comfy.hooks.HookGroup=None,
        hooks_B: comfy.hooks.HookGroup=None,
        hooks_C: comfy.hooks.HookGroup=None,
        hooks_D: comfy.hooks.HookGroup=None,
    ):
        candidates = [hooks_A, hooks_B, hooks_C, hooks_D]
        return io.NodeOutput(comfy.hooks.HookGroup.combine_all_hooks(candidates))


class CombineHooksEight(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CombineHooks8_V3",
            display_name="Combine Hooks [8] _V3",
            category="advanced/hooks/combine",
            is_experimental=True,
            inputs=[
                io.Hooks.Input("hooks_A"),
                io.Hooks.Input("hooks_B"),
                io.Hooks.Input("hooks_C"),
                io.Hooks.Input("hooks_D"),
                io.Hooks.Input("hooks_E"),
                io.Hooks.Input("hooks_F"),
                io.Hooks.Input("hooks_G"),
                io.Hooks.Input("hooks_H"),
            ],
            outputs=[
                io.Hooks.Output(),
            ],
            hidden=[]
        )

    @classmethod
    def execute(
        cls,
        hooks_A: comfy.hooks.HookGroup=None,
        hooks_B: comfy.hooks.HookGroup=None,
        hooks_C: comfy.hooks.HookGroup=None,
        hooks_D: comfy.hooks.HookGroup=None,
        hooks_E: comfy.hooks.HookGroup=None,
        hooks_F: comfy.hooks.HookGroup=None,
        hooks_G: comfy.hooks.HookGroup=None,
        hooks_H: comfy.hooks.HookGroup=None,
    ):
        candidates = [hooks_A, hooks_B, hooks_C, hooks_D, hooks_E, hooks_F, hooks_G, hooks_H]
        return io.NodeOutput(comfy.hooks.HookGroup.combine_all_hooks(candidates))


NODES_LIST = [
    CombineHooks,
    CombineHooksFour,
    CombineHooksEight,
    ConditioningSetDefaultAndCombine,
    ConditioningSetProperties,
    ConditioningSetPropertiesAndCombine,
    ConditioningTimestepsRange,
    CreateHookKeyframe,
    CreateHookKeyframesFromFloats,
    CreateHookKeyframesInterpolated,
    CreateHookLora,
    CreateHookLoraModelOnly,
    CreateHookModelAsLora,
    CreateHookModelAsLoraModelOnly,
    PairConditioningCombine,
    PairConditioningSetDefaultAndCombine,
    PairConditioningSetProperties,
    PairConditioningSetPropertiesAndCombine,
    SetClipHooks,
    SetHookKeyframes,
    SetModelHooksOnCond,
]
