from __future__ import annotations

from typing import Protocol, Optional, Any

import torch


class ModelManageable(Protocol):
    """
    Objects which implement this protocol can be managed by

    >>> import comfy.model_management
    >>> class SomeObj("ModelManageable"):
    >>>     ...
    >>>
    >>> comfy.model_management.load_model_gpu(SomeObj())
    """
    load_device: torch.device
    offload_device: torch.device
    model: torch.nn.Module

    @property
    def current_device(self) -> torch.device:
        ...

    def is_clone(self, other: Any) -> bool:
        ...

    def clone_has_same_weights(self, clone: torch.nn.Module) -> bool:
        ...

    def model_size(self) -> int:
        ...

    def model_patches_to(self, arg: torch.device | torch.dtype):
        ...

    def model_dtype(self) -> torch.dtype:
        ...

    def patch_model_lowvram(self, device_to: torch.device, lowvram_model_memory: int, force_patch_weights: Optional[bool] = False) -> torch.nn.Module:
        ...

    def patch_model(self, device_to: torch.device, patch_weights: bool) -> torch.nn.Module:
        ...

    def unpatch_model(self, offload_device: torch.device, unpatch_weights: Optional[bool] = False) -> torch.nn.Module:
        ...

    @property
    def lowvram_patch_counter(self) -> int:
        ...