from typing import Optional

import torch
import torch.nn as nn

import comfy.model_management


class WeightAdapterBase:
    name: str
    loaded_keys: set[str]
    weights: list[torch.Tensor]

    @classmethod
    def load(cls, x: str, lora: dict[str, torch.Tensor], alpha: float, dora_scale: torch.Tensor) -> Optional["WeightAdapterBase"]:
        raise NotImplementedError

    def to_train(self) -> "WeightAdapterTrainBase":
        raise NotImplementedError

    @classmethod
    def create_train(cls, weight, *args) -> "WeightAdapterTrainBase":
        """
        weight: The original weight tensor to be modified.
        *args: Additional arguments for configuration, such as rank, alpha etc.
        """
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
    # We follow the scheme of PR #7032
    def __init__(self):
        super().__init__()

    def __call__(self, w):
        """
        w: The original weight tensor to be modified.
        """
        raise NotImplementedError

    def passive_memory_usage(self):
        raise NotImplementedError("passive_memory_usage is not implemented")

    def move_to(self, device):
        self.to(device)
        return self.passive_memory_usage()


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


def tucker_weight_from_conv(up, down, mid):
    up = up.reshape(up.size(0), up.size(1))
    down = down.reshape(down.size(0), down.size(1))
    return torch.einsum("m n ..., i m, n j -> i j ...", mid, up, down)


def tucker_weight(wa, wb, t):
    temp = torch.einsum("i j ..., j r -> i r ...", t, wb)
    return torch.einsum("i j ..., i r -> r j ...", temp, wa)


def factorization(dimension: int, factor: int = -1) -> tuple[int, int]:
    """
    return a tuple of two value of input dimension decomposed by the number closest to factor
    second value is higher or equal than first value.

    examples)
    factor
        -1               2                4               8               16               ...
    127 -> 1, 127   127 -> 1, 127    127 -> 1, 127   127 -> 1, 127   127 -> 1, 127
    128 -> 8, 16    128 -> 2, 64     128 -> 4, 32    128 -> 8, 16    128 -> 8, 16
    250 -> 10, 25   250 -> 2, 125    250 -> 2, 125   250 -> 5, 50    250 -> 10, 25
    360 -> 8, 45    360 -> 2, 180    360 -> 4, 90    360 -> 8, 45    360 -> 12, 30
    512 -> 16, 32   512 -> 2, 256    512 -> 4, 128   512 -> 8, 64    512 -> 16, 32
    1024 -> 32, 32  1024 -> 2, 512   1024 -> 4, 256  1024 -> 8, 128  1024 -> 16, 64
    """

    if factor > 0 and (dimension % factor) == 0 and dimension >= factor**2:
        m = factor
        n = dimension // factor
        if m > n:
            n, m = m, n
        return m, n
    if factor < 0:
        factor = dimension
    m, n = 1, dimension
    length = m + n
    while m < n:
        new_m = m + 1
        while dimension % new_m != 0:
            new_m += 1
        new_n = dimension // new_m
        if new_m + new_n > length or new_m > factor:
            break
        else:
            m, n = new_m, new_n
    if m > n:
        n, m = m, n
    return m, n
