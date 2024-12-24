import torch
from typing import Callable, Protocol, TypedDict, Optional, List
from .node_typing import IO, InputTypeDict, ComfyNodeABC, CheckLazyMixin


class UnetApplyFunction(Protocol):
    """Function signature protocol on comfy.model_base.BaseModel.apply_model"""

    def __call__(self, x: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        pass


class UnetApplyConds(TypedDict):
    """Optional conditions for unet apply function."""

    c_concat: Optional[torch.Tensor]
    c_crossattn: Optional[torch.Tensor]
    control: Optional[torch.Tensor]
    transformer_options: Optional[dict]


class UnetParams(TypedDict):
    # Tensor of shape [B, C, H, W]
    input: torch.Tensor
    # Tensor of shape [B]
    timestep: torch.Tensor
    c: UnetApplyConds
    # List of [0, 1], [0], [1], ...
    # 0 means conditional, 1 means conditional unconditional
    cond_or_uncond: List[int]


UnetWrapperFunction = Callable[[UnetApplyFunction, UnetParams], torch.Tensor]


__all__ = [
    "UnetWrapperFunction",
    UnetApplyConds.__name__,
    UnetParams.__name__,
    UnetApplyFunction.__name__,
    IO.__name__,
    InputTypeDict.__name__,
    ComfyNodeABC.__name__,
    CheckLazyMixin.__name__,
]
