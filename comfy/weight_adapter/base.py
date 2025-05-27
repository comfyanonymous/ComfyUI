from typing import Optional

import torch
import torch.nn as nn

import comfy.model_management


class WeightAdapterBase:
    name: str
    loaded_keys: set[str]
    weights: list[torch.Tensor]

    @classmethod
    def load(cls, x: str, lora: dict[str, torch.Tensor]) -> Optional["WeightAdapterBase"]:
        raise NotImplementedError

    def to_train(self) -> "WeightAdapterTrainBase":
        raise NotImplementedError

    def calculate_weight(
        self,
        weight,
        key,
        strength,
        strength_model,
        offset,
        function,
        intermediate_dtype=torch.float32,
        original_weight=None,
    ):
        raise NotImplementedError


class WeightAdapterTrainBase(nn.Module):
    def __init__(self):
        super().__init__()

    # [TODO] Collaborate with LoRA training PR #7032


def weight_decompose(dora_scale, weight, lora_diff, alpha, strength, intermediate_dtype, function):
    dora_scale = comfy.model_management.cast_to_device(dora_scale, weight.device, intermediate_dtype)
    lora_diff *= alpha
    weight_calc = weight + function(lora_diff).type(weight.dtype)

    wd_on_output_axis = dora_scale.shape[0] == weight_calc.shape[0]
    if wd_on_output_axis:
        weight_norm = (
            weight.reshape(weight.shape[0], -1)
            .norm(dim=1, keepdim=True)
            .reshape(weight.shape[0], *[1] * (weight.dim() - 1))
        )
    else:
        weight_norm = (
            weight_calc.transpose(0, 1)
            .reshape(weight_calc.shape[1], -1)
            .norm(dim=1, keepdim=True)
            .reshape(weight_calc.shape[1], *[1] * (weight_calc.dim() - 1))
            .transpose(0, 1)
        )
    weight_norm = weight_norm + torch.finfo(weight.dtype).eps

    weight_calc *= (dora_scale / weight_norm).type(weight.dtype)
    if strength != 1.0:
        weight_calc -= weight
        weight += strength * (weight_calc)
    else:
        weight[:] = weight_calc
    return weight


def pad_tensor_to_shape(tensor: torch.Tensor, new_shape: list[int]) -> torch.Tensor:
    """
    Pad a tensor to a new shape with zeros.

    Args:
        tensor (torch.Tensor): The original tensor to be padded.
        new_shape (List[int]): The desired shape of the padded tensor.

    Returns:
        torch.Tensor: A new tensor padded with zeros to the specified shape.

    Note:
        If the new shape is smaller than the original tensor in any dimension,
        the original tensor will be truncated in that dimension.
    """
    if any([new_shape[i] < tensor.shape[i] for i in range(len(new_shape))]):
        raise ValueError("The new shape must be larger than the original tensor in all dimensions")

    if len(new_shape) != len(tensor.shape):
        raise ValueError("The new shape must have the same number of dimensions as the original tensor")

    # Create a new tensor filled with zeros
    padded_tensor = torch.zeros(new_shape, dtype=tensor.dtype, device=tensor.device)

    # Create slicing tuples for both tensors
    orig_slices = tuple(slice(0, dim) for dim in tensor.shape)
    new_slices = tuple(slice(0, dim) for dim in tensor.shape)

    # Copy the original tensor into the new tensor
    padded_tensor[new_slices] = tensor[orig_slices]

    return padded_tensor
