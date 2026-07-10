"""
TorchAO INT4 quantization helpers for ComfyUI.

- quantize_model: in-place INT4 quantization via torchao
- reconstruct_int4_state_dict: rebuild TINT4Linear layers from safetensors
"""

import torch
import torch.nn as nn
from torchao.quantization import Int4WeightOnlyConfig, quantize_

from .linear import TINT4Linear


def quantize_model(
    model: nn.Module,
    group_size: int = 128,
    filter_fn=None,
) -> nn.Module:
    """
    Quantize all nn.Linear layers in-place via torchao INT4 weight-only.

    Args:
        model: PyTorch model to quantize.
        group_size: Quantization group size (32/64/128/256).
        filter_fn: Optional callable(module, name) → bool.
                   Return True to skip that module.

    Returns:
        Same model (quantized in-place).
    """
    config = Int4WeightOnlyConfig(
        group_size=group_size,
        int4_packing_format="plain_int32",
    )
    quantize_(model, config, filter_fn=filter_fn)
    return model


# torchao plain_int32 safetensors suffixes
_QUANT_SUFFIXES = (
    ".weight_scale", ".weight_zp",
    ".weight_b0", ".weight_b1",
    ".weight_sh0", ".weight_sh1",
    ".comfy_quant",
)


def _is_quantized_weight(key: str, sd: dict) -> bool:
    """Check whether a safetensors key is a torchao-quantized weight tensor."""
    if not key.endswith(".weight"):
        return False
    base = key[:-len(".weight")]
    return f"{base}.weight_scale" in sd and f"{base}.weight_zp" in sd


def reconstruct_int4_state_dict(
    sd: dict,
    device: torch.device = torch.device("cpu"),
) -> dict[str, TINT4Linear]:
    """
    Extract TINT4Linear layers from a torchao-quantized safetensors dict.

    Pops quantized weight tensors from sd and replaces them with
    fully-constructed TINT4Linear modules keyed by base layer name.

    Caller is responsible for placing the modules into the target model.

    Args:
        sd: Safetensors state dict with torchao int4 packed weights.
        device: Target device for the reconstructed layers.

    Returns:
        Dict mapping base key (e.g. "blocks.0.attn.qkv") → TINT4Linear.
    """
    replaced: dict[str, TINT4Linear] = {}

    for key in list(sd.keys()):
        if not _is_quantized_weight(key, sd):
            continue

        base = key[:-len(".weight")]
        qdata = sd.pop(key)
        scale = sd.pop(f"{base}.weight_scale")
        zp = sd.pop(f"{base}.weight_zp")
        b0 = sd.pop(f"{base}.weight_b0")
        b1 = sd.pop(f"{base}.weight_b1")

        for suffix in _QUANT_SUFFIXES:
            sd.pop(f"{base}{suffix}", None)

        in_f = qdata.shape[1] * 8
        out_f = qdata.shape[0]

        layer = TINT4Linear(
            in_features=in_f,
            out_features=out_f,
            qdata=qdata.to(torch.int32),
            scale=scale.to(torch.float16),
            zp=zp.to(torch.int8),
            block_size=(b0.item(), b1.item()),
        )

        if device.type != "cpu":
            layer._qdata = layer._qdata.to(device)
            layer._scale = layer._scale.to(device)
            layer._zp = layer._zp.to(device)

        replaced[base] = layer

    return replaced
