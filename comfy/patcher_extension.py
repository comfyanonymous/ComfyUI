from __future__ import annotations
from typing import Callable

class CallbacksMP:
    ON_CLONE = "on_clone"
    ON_LOAD = "on_load_after"
    ON_CLEANUP = "on_cleanup"
    ON_PRE_RUN = "on_pre_run"
    ON_PREPARE_STATE = "on_prepare_state"
    ON_APPLY_HOOKS = "on_apply_hooks"
    ON_REGISTER_ALL_HOOK_PATCHES = "on_register_all_hook_patches"
    ON_INJECT_MODEL = "on_inject_model"
    ON_EJECT_MODEL = "on_eject_model"

    @classmethod
    def init_callbacks(cls):
        return {
            cls.ON_CLONE: {None: []},
            cls.ON_LOAD: {None: []},
            cls.ON_CLEANUP: {None: []},
            cls.ON_PRE_RUN: {None: []},
            cls.ON_PREPARE_STATE: {None: []},
            cls.ON_APPLY_HOOKS: {None: []},
            cls.ON_REGISTER_ALL_HOOK_PATCHES: {None: []},
            cls.ON_INJECT_MODEL: {None: []},
            cls.ON_EJECT_MODEL: {None: []},
        }

class WrappersMP:
    OUTER_SAMPLE = "outer_sample"
    SAMPLER_SAMPLE = "sampler_sample"
    CALC_COND_BATCH = "calc_cond_batch"
    APPLY_MODEL = "apply_model"
    DIFFUSION_MODEL = "diffusion_model"

    @classmethod
    def init_wrappers(cls):
        return {
            cls.OUTER_SAMPLE: {None: []},
            cls.SAMPLER_SAMPLE: {None: []},
            cls.CALC_COND_BATCH: {None: []},
            cls.APPLY_MODEL: {None: []},
            cls.DIFFUSION_MODEL: {None: []},
        }

class WrapperExecutor:
    """Handles call stack of wrappers around a function in an ordered manner."""
    def __init__(self, original: Callable, class_obj: object, wrappers: list[Callable], idx: int):
        self.original = original
        self.class_obj = class_obj
        self.wrappers = wrappers.copy()
        self.idx = idx
        self.is_last = idx == len(wrappers)
    
    def __call__(self, *args, **kwargs):
        """Calls the next wrapper in line or original function, whichever is appropriate."""
        new_executor = self._create_next_executor()
        return new_executor.execute(*args, **kwargs)
    
    def execute(self, *args, **kwargs):
        """Used to initiate executor internally - DO NOT use this if you received executor in wrapper."""
        args = list(args)
        kwargs = dict(kwargs)
        if self.is_last:
            return self.original(*args, **kwargs)
        return self.wrappers[self.idx](self, *args, **kwargs)

    def _create_next_executor(self) -> 'WrapperExecutor':
        new_idx = self.idx + 1
        if new_idx > len(self.wrappers):
            raise Exception(f"Wrapper idx exceeded available wrappers; something went very wrong.")
        if self.class_obj is None:
            return WrapperExecutor.new_executor(self.original, self.wrappers, new_idx)
        return WrapperExecutor.new_class_executor(self.original, self.class_obj, self.wrappers, new_idx)

    @classmethod
    def new_executor(cls, original: Callable, wrappers: list[Callable], idx=0):
        return cls(original, class_obj=None, wrappers=wrappers, idx=idx)
    
    @classmethod
    def new_class_executor(cls, original: Callable, class_obj: object, wrappers: list[Callable], idx=0):
        return cls(original, class_obj, wrappers, idx=idx)

class PatcherInjection:
    def __init__(self, inject: Callable, eject: Callable):
        self.inject = inject
        self.eject = eject
