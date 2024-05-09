from __future__ import annotations

from typing import Protocol, Optional

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
    current_device: torch.device

    @property
    def dtype(self) -> torch.dtype:
        ...

    def is_clone(self, other: torch.nn.Module) -> bool:
        pass

    def clone_has_same_weights(self, clone: torch.nn.Module) -> bool:
        pass

    def model_size(self) -> int:
        pass

    def model_patches_to(self, arg: torch.device | torch.dtype):
        pass

    def model_dtype(self) -> torch.dtype:
        pass

    def patch_model_lowvram(self, device_to: torch.device, lowvram_model_memory: int) -> torch.nn.Module:
        pass

    def patch_model(self, device_to: torch.device, patch_weights: bool) -> torch.nn.Module:
        pass

    def unpatch_model(self, offload_device: torch.device, unpatch_weights: Optional[bool] = False) -> torch.nn.Module:
        pass
