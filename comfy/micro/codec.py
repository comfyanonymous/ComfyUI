from io import BytesIO

import numpy as np
import torch


SUPPORTED_TYPES = frozenset({"IMAGE", "MASK"})


def _tensor_to_bytes(arr: torch.Tensor) -> bytes:
    buffer = BytesIO()
    np.save(buffer, arr.detach().cpu().numpy(), allow_pickle=False)
    return buffer.getvalue()


def _bytes_to_tensor(data: bytes) -> torch.Tensor:
    buffer = BytesIO(data)
    arr = np.load(buffer, allow_pickle=False)
    return torch.from_numpy(arr)


def encode_image(arr: torch.Tensor) -> bytes:
    """Encode a ComfyUI IMAGE tensor [N,H,W,3] fp16/fp32 to bytes."""

    return _tensor_to_bytes(arr)


def decode_image(data: bytes) -> torch.Tensor:
    """Inverse of encode_image."""

    return _bytes_to_tensor(data)


def encode_mask(arr: torch.Tensor) -> bytes:
    """Encode a ComfyUI MASK tensor to bytes."""

    return _tensor_to_bytes(arr)


def decode_mask(data: bytes) -> torch.Tensor:
    """Inverse of encode_mask."""

    return _bytes_to_tensor(data)


def supports_type(type_name: str) -> bool:
    return type_name in SUPPORTED_TYPES


def encode(type_name: str, value) -> bytes:
    if type_name == "IMAGE":
        return encode_image(value)
    if type_name == "MASK":
        return encode_mask(value)
    raise ValueError(f"unsupported micro type {type_name!r}")


def decode(type_name: str, data: bytes):
    if type_name == "IMAGE":
        return decode_image(data)
    if type_name == "MASK":
        return decode_mask(data)
    raise ValueError(f"unsupported micro type {type_name!r}")
