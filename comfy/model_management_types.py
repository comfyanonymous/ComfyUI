from __future__ import annotations

import copy
import dataclasses
from abc import ABCMeta, abstractmethod
from typing import Protocol, Optional, TypeVar, runtime_checkable, Callable, Any, NamedTuple, TYPE_CHECKING

import torch
import torch.nn
from typing_extensions import TypedDict, NotRequired, override

from .comfy_types import UnetWrapperFunction
from .latent_formats import LatentFormat

if TYPE_CHECKING:
    from .hooks import EnumHookMode

ModelManageableT = TypeVar('ModelManageableT', bound='ModelManageable')
LatentFormatT = TypeVar('LatentFormatT', bound=LatentFormat)


@runtime_checkable
class DeviceSettable(Protocol):
    device: torch.device


@runtime_checkable
class HooksSupport(Protocol):
    wrappers: dict[str, dict[str, list[Callable]]]
    callbacks: dict[str, dict[str, list[Callable]]]
    hook_mode: "EnumHookMode"

    def prepare_hook_patches_current_keyframe(self, t, hook_group, model_options): ...

    def model_patches_models(self) -> list[ModelManageableT]: ...

    def restore_hook_patches(self): ...

    def cleanup(self): ...

    def pre_run(self): ...

    def prepare_state(self, *args, **kwargs): ...

    def register_all_hook_patches(self, a, b, c, d): ...

    def get_nested_additional_models(self): ...

    def apply_hooks(self, *args, **kwargs): ...

    def add_wrapper(self, wrapper_type: str, wrapper: Callable): ...

    def add_wrapper_with_key(self, wrapper_type: str, key: str, wrapper: Callable): ...


class HooksSupportStub(HooksSupport, metaclass=ABCMeta):
    def prepare_hook_patches_current_keyframe(self, t, hook_group, model_options):
        return

    def model_patches_models(self) -> list[ModelManageableT]:
        """
        Used to implement Qwen DiffSynth Controlnets (?)
        :return:
        """
        return []

    @property
    def hook_mode(self):
        from .hooks import EnumHookMode
        if not hasattr(self, "_hook_mode"):
            setattr(self, "_hook_mode", EnumHookMode.MaxSpeed)
        return getattr(self, "_hook_mode")

    @hook_mode.setter
    def hook_mode(self, value):
        setattr(self, "_hook_mode", value)

    def restore_hook_patches(self):
        return

    @property
    def wrappers(self):
        if not hasattr(self, "_wrappers"):
            setattr(self, "_wrappers", {})
        return getattr(self, "_wrappers")

    @wrappers.setter
    def wrappers(self, value):
        setattr(self, "_wrappers", value)

    @property
    def callbacks(self) -> dict:
        if not hasattr(self, "_callbacks"):
            setattr(self, "_callbacks", {})
        return getattr(self, "_callbacks")

    @callbacks.setter
    def callbacks(self, value):
        setattr(self, "_callbacks", value)

    def cleanup(self):
        pass

    def pre_run(self):
        if hasattr(self, "model"):
            model = getattr(self, "model")
            from .model_base import BaseModel

            if isinstance(model, BaseModel) or hasattr(model, "current_patcher") and isinstance(self, ModelManageable):
                model.current_patcher = self

    def prepare_state(self, *args, **kwargs):
        pass

    def register_all_hook_patches(self, a, b, c, d):
        pass

    def get_nested_additional_models(self):
        return []

    def apply_hooks(self, *args, **kwargs):
        return {}

    def add_wrapper(self, wrapper_type: str, wrapper: Callable):
        self.add_wrapper_with_key(wrapper_type, None, wrapper)

    def add_wrapper_with_key(self, wrapper_type: str, key: str, wrapper: Callable):
        w = self.wrappers.setdefault(wrapper_type, {}).setdefault(key, [])
        w.append(wrapper)


@runtime_checkable
class TrainingSupport(Protocol):
    def set_model_compute_dtype(self, dtype: torch.dtype): ...

    def add_weight_wrapper(self, name, function): ...


class TrainingSupportStub(TrainingSupport, metaclass=ABCMeta):
    def set_model_compute_dtype(self, dtype: torch.dtype):
        return

    def add_weight_wrapper(self, name, function):
        return


@runtime_checkable
class ModelManageable(HooksSupport, TrainingSupport, Protocol):
    """
    Objects which implement this protocol can be managed by

    >>> from comfy.model_management import load_models_gpu
    >>> class ModelWrapper(ModelManageable):
    >>>     ...
    >>>
    >>> some_model = ModelWrapper()
    >>> load_models_gpu([some_model])

    The minimum required
    """
    load_device: torch.device
    offload_device: torch.device
    model: torch.nn.Module

    @property
    def current_device(self) -> torch.device: ...

    def is_clone(self, other: ModelManageableT) -> bool: ...

    def clone_has_same_weights(self, clone: ModelManageableT) -> bool: ...

    def model_size(self) -> int: ...

    def model_patches_to(self, arg: torch.device | torch.dtype): ...

    def model_dtype(self) -> torch.dtype: ...

    def lowvram_patch_counter(self) -> int: ...

    def partially_load(self, device_to: torch.device, extra_memory: int = 0, force_patch_weights: bool = False) -> int:  ...

    def partially_unload(self, device_to: torch.device, memory_to_free: int = 0) -> int: ...

    def memory_required(self, input_shape: torch.Size) -> int: ...

    def loaded_size(self) -> int: ...

    def current_loaded_device(self) -> torch.device: ...

    def get_model_object(self, name: str) -> torch.nn.Module: ...

    @property
    def model_options(self) -> ModelOptions: ...

    @model_options.setter
    def model_options(self, value): ...

    def __del__(self): ...

    @property
    def parent(self) -> ModelManageableT | None: ...

    def detach(self, unpatch_all: bool = True): ...

    def clone(self) -> ModelManageableT: ...


class ModelManageableStub(HooksSupportStub, TrainingSupportStub, ModelManageable, metaclass=ABCMeta):
    """
    The bare minimum that must be implemented to support model management when inheriting from ModelManageable

    Attributes:
        load_device (torch.device): the device that this model's weights will be loaded onto for inference, typically the GPU
        offload_device (torch.device): the device that this model's weights will be offloaded onto when not being used for inference or when performing CPU offloading, typically the CPU
        model (torch.nn.Module): in principle this can be any callable, but it should be a torch model to work with the rest of the machinery
    :see: ModelManageable
    :see: PatchSupport
    """

    @abstractmethod
    def patch_model(self, device_to: torch.device | None = None, lowvram_model_memory: int = 0, load_weights: bool = True,
                    force_patch_weights: bool = False) -> torch.nn.Module:
        """
        Called by ModelManageable

        An implementation of this method should
        (1) Loads the model by moving it to the target device
        (2) Fusing the LoRA weights ("patches", if applicable)

        :param device_to:
        :param lowvram_model_memory:
        :param load_weights:
        :param force_patch_weights:
        :return:
        """
        ...

    @abstractmethod
    def unpatch_model(self, device_to: torch.device | None = None, unpatch_weights: Optional[bool] = False) -> torch.nn.Module:
        """
        Called by ModelManageable

        Unloads the model by:
        (1) Unfusing the LoRA weights ("unpatching", if applicable)
        (1) Moving the weights to the provided device
        :param device_to:
        :param unpatch_weights:
        :return:
        """
        ...

    @property
    @override
    def current_device(self) -> torch.device:
        """
        Only needed in Hidden Switch, does not need to be overridden
        :return:
        """
        return next(self.model.parameters()).device

    def is_clone(self, other: ModelManageableT) -> bool:
        return other.model is self.model

    def clone_has_same_weights(self, clone: ModelManageableT) -> bool:
        return clone.model is self.model

    def model_size(self) -> int:
        from .model_management import module_size
        return module_size(self.model)

    def model_patches_to(self, arg: torch.device | torch.dtype):
        pass

    def model_dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype

    def lowvram_patch_counter(self) -> int:
        """
        Returns a counter related to low VRAM patching, used to decide if a reload is necessary.
        """
        return 0

    def partially_load(self, device_to: torch.device, extra_memory: int = 0, force_patch_weights: bool = False):
        self.patch_model(device_to=device_to)
        return self.model_size()

    def partially_unload(self, device_to: torch.device, memory_to_free: int = 0):
        self.unpatch_model(device_to)
        return self.model_size()

    def memory_required(self, input_shape: torch.Size) -> int:
        from .model_base import BaseModel

        if isinstance(self.model, BaseModel):
            return self.model.memory_required(input_shape=input_shape)
        else:
            # todo: we need a real implementation of this
            return 0

    def loaded_size(self) -> int:
        if self.current_loaded_device() == self.load_device:
            return self.model_size()
        return 0

    def current_loaded_device(self) -> torch.device:
        return self.current_device

    def get_model_object(self, name: str) -> torch.nn.Module:
        from . import utils
        return utils.get_attr(self.model, name)

    @property
    def model_options(self) -> ModelOptions:
        if not hasattr(self, "_model_options"):
            setattr(self, "_model_options", {"transformer_options": {}})
        return getattr(self, "_model_options")

    @model_options.setter
    def model_options(self, value):
        setattr(self, "_model_options", value)

    def __del__(self):
        if hasattr(self.model, "__del__"):
            del self.model

    @property
    def parent(self) -> ModelManageableT | None:
        """
        Used for tracking a parent model from which this was cloned
        :return:
        """
        return None

    def detach(self, unpatch_all: bool = True):
        """
        Unloads the model
        :param unpatch_all:
        :return:
        """
        self.unpatch_model(self.offload_device, unpatch_weights=unpatch_all)
        return self.model

    def clone(self) -> ModelManageableT:
        return copy.copy(self)


@dataclasses.dataclass
class MemoryMeasurements:
    model: torch.nn.Module | DeviceSettable
    model_loaded_weight_memory: int = 0
    lowvram_patch_counter: int = 0
    model_lowvram: bool = False
    current_weight_patches_uuid: Any = None
    _device: torch.device | None = None

    @property
    def device(self) -> torch.device:
        if isinstance(self.model, DeviceSettable):
            return self.model.device
        elif hasattr(self.model, "device"):
            return self.model.device
        else:
            return self._device

    @device.setter
    def device(self, value: torch.device):
        if isinstance(self.model, DeviceSettable):
            # todo: is this correct?
            self.model.device = value
        # todo: we don't want to `to` anything anymore here
        self._device = value


class HasModels(Protocol):
    """A protocol for any object that has a .models() method returning a list."""

    def models(self) -> list:
        ...


class HasTo(Protocol):
    def to(self, device: torch.device):
        ...


class TransformerOptions(TypedDict, total=False):
    cond_or_uncond: NotRequired[list]
    patches: NotRequired[dict[str, list[HasModels]]]
    sigmas: NotRequired[torch.Tensor]
    patches_replace: NotRequired[dict[str, dict[Any, HasModels]]]


class ModelOptions(TypedDict, total=False):
    transformer_options: NotRequired[dict]
    # signature of BaseModel.apply_model
    model_function_wrapper: NotRequired[Callable | UnetWrapperFunction | HasModels | HasTo]
    sampler_cfg_function: NotRequired[Callable]
    sampler_post_cfg_function: NotRequired[list[Callable]]
    disable_cfg1_optimization: NotRequired[bool]
    denoise_mask_function: NotRequired[Callable]
    patches: NotRequired[dict[str, list]]


class LoadingListItem(NamedTuple):
    module_size: int
    name: str
    module: torch.nn.Module
    params: list[str]
