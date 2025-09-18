from __future__ import annotations
from typing import Protocol, List, Dict, Optional, NamedTuple, Callable, Literal, Any, TypeAlias, Union, runtime_checkable
import torch

PatchOffset = tuple[int, int, int]
PatchFunction = Any
PatchDictKey = str | tuple[str, PatchOffset] | tuple[str, PatchOffset, PatchFunction]
PatchType = Literal["lora", "loha", "lokr", "glora", "diff", "set", ""]
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


ModelPatchesDictValue: TypeAlias = list[Union[PatchTuple, PatchWeightTuple]]


@runtime_checkable
class PatchSupport(Protocol):
    """
    Defines the interface for a model that supports LoRA patching.
    """

    def add_patches(
            self,
            patches: PatchDict,
            strength_patch: float = 1.0,
            strength_model: float = 1.0
    ) -> List[PatchDictKey]:
        """
        Applies a set of patches (like LoRA weights) to the model.

        Args:
            patches (PatchDict): A dictionary containing the patch weights and metadata.
            strength_patch (float): The strength multiplier for the patch itself.
            strength_model (float): The strength multiplier for the original model weights.

        Returns:
            List[PatchDictKey]: A list of keys for the weights that were successfully patched.
        """
        ...

    def get_key_patches(
            self,
            filter_prefix: Optional[str] = None
    ) -> Dict[str, ModelPatchesDictValue]:
        """
        Retrieves all active patches, optionally filtered by a key prefix.

        The returned dictionary maps a model weight's key to a list. The first
        element in the list is a tuple containing the original weight, and subsequent
        elements are the applied patch tuples.

        Args:
            filter_prefix (Optional[str]): A prefix to filter which weight patches are returned.

        Returns:
            Dict[str, ModelPatchesDictValue]: A dictionary of the model's patched weights.
        """
        ...
