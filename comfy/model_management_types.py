from __future__ import annotations

import dataclasses
from abc import ABCMeta, abstractmethod
from typing import Protocol, Optional, TypeVar, runtime_checkable, Callable, Any, NamedTuple

import torch
import torch.nn
from typing_extensions import TypedDict, NotRequired

from .latent_formats import LatentFormat

ModelManageableT = TypeVar('ModelManageableT', bound='ModelManageable')
LatentFormatT = TypeVar('LatentFormatT', bound=LatentFormat)


@runtime_checkable
class DeviceSettable(Protocol):
    @property
    def device(self) -> torch.device:
        ...

    @device.setter
    def device(self, value: torch.device):
        ...


class ModelManageable(Protocol, metaclass=ABCMeta):
    """
    Objects which implement this protocol can be managed by

    >>> from comfy.model_management import load_models_gpu
    >>> class ModelWrapper(ModelManageable):
    >>>     ...
    >>>
    >>> some_model = ModelWrapper()
    >>> load_models_gpu([some_model])
    """
    load_device: torch.device
    offload_device: torch.device
    model: torch.nn.Module

    @property
    def current_device(self) -> torch.device:
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

    def patch_model(self, device_to: torch.device | None = None, lowvram_model_memory: int = 0, load_weights: bool = True, force_patch_weights: bool = False) -> torch.nn.Module:
        ...

    def unpatch_model(self, device_to: torch.device | None = None, unpatch_weights: Optional[bool] = False) -> torch.nn.Module:
        """
        Unloads the model by moving it to the offload device
        :param device_to:
        :param unpatch_weights:
        :return:
        """
        ...

    def lowvram_patch_counter(self) -> int:
        return 0

    def partially_load(self, device_to: torch.device, extra_memory: int = 0, force_patch_weights: bool = False):
        self.patch_model(device_to=device_to)
        return self.model_size()

    def partially_unload(self, device_to: torch.device, memory_to_free: int = 0):
        self.unpatch_model(device_to)
        return self.model_size()

    def memory_required(self, input_shape) -> int:
        from .model_base import BaseModel

        if isinstance(self.model, BaseModel):
            return self.model.memory_required(input_shape=input_shape)
        else:
            # todo: why isn't this true?
            return self.model_size()

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
        return None

    def detach(self, unpatch_all: bool = True):
        self.model_patches_to(self.offload_device)
        if unpatch_all:
            self.unpatch_model(self.offload_device, unpatch_weights=unpatch_all)
        return self.model

    def set_model_compute_dtype(self, dtype: torch.dtype):
        pass

    def add_weight_wrapper(self, name, function):
        pass

    @property
    def force_cast_weights(self) -> bool:
        return False

    def prepare_hook_patches_current_keyframe(self, t, hook_group, model_options):
        pass


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
            self.model.device = value
        elif hasattr(self.model, "to"):
            self.model.to(value)
        self._device = value


class TransformerOptions(TypedDict, total=False):
    cond_or_uncond: NotRequired[list]
    patches: NotRequired[dict]
    sigmas: NotRequired[torch.Tensor]


class ModelOptions(TypedDict, total=False):
    transformer_options: NotRequired[dict]
    # signature of BaseModel.apply_model
    model_function_wrapper: NotRequired[Callable]
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