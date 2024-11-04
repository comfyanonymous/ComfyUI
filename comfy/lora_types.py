from __future__ import annotations

from typing import Literal, Any, NamedTuple, Protocol, Callable

import torch

PatchOffset = tuple[int, int, int]
PatchFunction = Any
PatchDictKey = str | tuple[str, PatchOffset] | tuple[str, PatchOffset, PatchFunction]
PatchType = Literal["lora", "loha", "lokr", "glora", "diff", ""]
PatchDictValue = tuple[PatchType, tuple]
PatchDict = dict[PatchDictKey, PatchDictValue]


class PatchConversionFunction(Protocol):
    def __call__(self, tensor: torch.Tensor, **kwargs) -> torch.Tensor:
        ...


class PatchWeightTuple(NamedTuple):
    weight: torch.Tensor
    convert_func: PatchConversionFunction | Callable[[torch.Tensor], torch.Tensor]


class PatchTuple(NamedTuple):
    strength_patch: float
    patch: PatchDictValue
    strength_model: float
    offset: PatchOffset
    function: PatchFunction


ModelPatchesDictValue = list[PatchTuple | PatchWeightTuple]
