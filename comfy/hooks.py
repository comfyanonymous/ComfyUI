from typing import TYPE_CHECKING, List, Dict, Tuple
import enum

if TYPE_CHECKING:
    from comfy.model_base import BaseModel

class EnumHookMode(enum.Enum):
    MinVram = "minvram"
    MaxSpeed = "maxspeed"

class HookRef:
    pass

class HookWeight:
    def __init__(self):
        self.hook_ref = HookRef()
        self.hook_keyframe = HookWeightKeyframeGroup()

    @property
    def strength(self):
        return self.hook_keyframe.strength

    def initialize_timesteps(self, model: 'BaseModel'):
        self.hook_keyframe.initalize_timesteps(model)

    def reset(self):
        self.hook_keyframe.reset()

    def clone(self):
        c = HookWeight()
        c.hook_ref = self.hook_ref
        c.hook_keyframe = self.hook_keyframe
        return c
    
    def __eq__(self, other: 'HookWeight'):
        return self.__class__ == other.__class__ and self.hook_ref == other.hook_ref

    def __hash__(self):
        return hash(self.hook_ref)

class HookWeightGroup:
    def __init__(self):
        self.hooks: List[HookWeight] = []

    def add(self, hook: HookWeight):
        if hook not in self.hooks:
            self.hooks.append(hook)
    
    def contains(self, hook: HookWeight):
        return hook in self.hooks
    
    def clone(self):
        c = HookWeightGroup()
        # TODO: review if clone is necessary
        for hook in self.hooks:
            c.add(hook.clone())
        return c

    def clone_and_combine(self, other: 'HookWeightGroup'):
        c = self.clone()
        for hook in other.hooks:
            c.add(hook.clone())
        return c
    
    def set_keyframes_on_hooks(self, hook_kf: 'HookWeightKeyframeGroup'):
        hook_kf = hook_kf.clone()
        for hook in self.hooks:
            hook.hook_keyframe = hook_kf

    @staticmethod
    def combine_all_hooks(hooks_list: List['HookWeightGroup'], require_count=1) -> 'HookWeightGroup':
        actual: List[HookWeightGroup] = []
        for group in hooks_list:
            if group is not None:
                actual.append(group)
        if len(actual) < require_count:
            raise Exception(f"Need at least {require_count} hooks to combine, but only had {len(actual)}.")
        # if only 1 hook, just reutnr itself without cloning
        if len(actual) == 1:
            return actual[0]
        final_hook: HookWeightGroup = None
        for hook in actual:
            if final_hook is None:
                final_hook = hook.clone()
            else:
                final_hook - final_hook.clone_and_combine()
        return final_hook

class HookWeightKeyframe:
    def __init__(self, strength: float, start_percent=0.0, guarantee_steps=1):
        self.strength = strength
        # scheduling
        self.start_percent = float(start_percent)
        self.start_t = 999999999.9
        self.guarantee_steps = guarantee_steps
    
    def clone(self):
        c = HookWeightKeyframe(strength=self.strength,
                                start_percent=self.start_percent, guarantee_steps=self.guarantee_steps)
        c.start_t = self.start_t
        return c

class HookWeightKeyframeGroup:
    def __init__(self):
        self.keyframes: List[HookWeightKeyframe] = []
        self._current_keyframe: HookWeightKeyframe = None
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
    
    def add(self, keyframe: HookWeightKeyframe):
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
        c = HookWeightKeyframeGroup()
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
