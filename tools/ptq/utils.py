import torch
import logging
from typing import Dict, Optional

import comfy.ops
from modelopt.torch.quantization.nn import QuantModuleRegistry, TensorQuantizer


# FP8 E4M3 configuration
FP8_CFG = {
    "quant_cfg": {
        "*weight_quantizer": {"num_bits": (4, 3), "axis": None},
        "*input_quantizer": {"num_bits": (4, 3), "axis": None},
        "default": {"enable": False},
    },
    "algorithm": "max",
}


def register_comfy_ops():
    """Register ComfyUI operations with ModelOptimizer."""
    op = comfy.ops.disable_weight_init.Linear
    op_name = op.__name__

    if op in QuantModuleRegistry:
        logging.debug("ComfyUI Linear already registered with ModelOptimizer")
        return

    # Register ComfyUI Linear using the same handler as torch.nn.Linear
    QuantModuleRegistry.register(
        {op: f"comfy.{op_name}"}
    )(QuantModuleRegistry._registry[getattr(torch.nn, op_name)])

    logging.info("Registered ComfyUI Linear with ModelOptimizer")

def log_quant_summary(model: torch.nn.Module, log_level=logging.INFO):
    count = 0
    for name, mod in model.named_modules():
        if isinstance(mod, TensorQuantizer):
            logging.log(log_level, f"{name:80} {mod}")
            count += 1
    logging.log(log_level, f"{count} TensorQuantizers found in model")

def extract_amax_values(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    amax_dict = {}

    for name, module in model.named_modules():
        if not isinstance(module, TensorQuantizer):
            continue
        if not module.is_enabled:
            continue
        if hasattr(module, '_amax') and module._amax is not None:
            amax = module._amax
            if not isinstance(amax, torch.Tensor):
                amax = torch.tensor(amax, dtype=torch.float32)

            amax_dict[name] = amax.clone().cpu()
            logging.debug(f"Extracted amax from {name}: {amax.item():.6f}")

    logging.info(f"Extracted amax values from {len(amax_dict)} quantizers")
    return amax_dict


def save_amax_dict(amax_dict: Dict[str, torch.Tensor], output_path: str, metadata: Optional[Dict] = None):
    import json
    from datetime import datetime

    logging.info(f"Saving {len(amax_dict)} amax values to {output_path}")

    # Convert tensors to Python floats for JSON serialization
    amax_values = {}
    for key, value in amax_dict.items():
        if isinstance(value, torch.Tensor):
            # Convert to float (scalar) or list
            if value.numel() == 1:
                amax_values[key] = float(value.item())
            else:
                amax_values[key] = value.cpu().numpy().tolist()
        else:
            amax_values[key] = float(value)

    # Build output with metadata
    output_dict = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "num_layers": len(amax_values),
            **(metadata or {})
        },
        "amax_values": amax_values
    }

    # Save as formatted JSON for easy inspection
    with open(output_path, 'w') as f:
        json.dump(output_dict, f, indent=2, sort_keys=True)

    logging.info(f"✓ Amax values saved to {output_path}")
