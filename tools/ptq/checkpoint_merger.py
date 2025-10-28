import argparse
import logging
import sys
import yaml
import re
from typing import Dict, Tuple
import torch
from safetensors.torch import save_file
import json

# Add comfyui to path if needed
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import comfy.utils
from comfy.ops import QUANT_FORMAT_MIXINS
from comfy.quant_ops import F8_E4M3_MAX, F4_E2M1_MAX

class QuantizationConfig:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Compile disable list patterns
        self.disable_patterns = []
        for pattern in self.config.get('disable_list', []):
            # Convert glob-style patterns to regex
            regex_pattern = pattern.replace('*', '.*')
            self.disable_patterns.append(re.compile(regex_pattern))

        # Parse per-layer dtype config
        self.per_layer_dtype = self.config.get('per_layer_dtype', {})
        self.dtype_patterns = []
        for pattern, dtype in self.per_layer_dtype.items():
            regex_pattern = pattern.replace('*', '.*')
            self.dtype_patterns.append((re.compile(regex_pattern), dtype))

        logging.info(f"Loaded config with {len(self.disable_patterns)} disable patterns")
        logging.info(f"Per-layer dtype rules: {self.per_layer_dtype}")

    def should_quantize(self, layer_name: str) -> bool:
        for pattern in self.disable_patterns:
            if pattern.match(layer_name):
                logging.debug(f"Layer {layer_name} disabled by pattern {pattern.pattern}")
                return False
        return True

    def get_dtype(self, layer_name: str) -> str:
        for pattern, dtype in self.dtype_patterns:
            if pattern.match(layer_name):
                return dtype
        return None

def load_amax_artefact(artefact_path: str) -> Dict:
    logging.info(f"Loading amax artefact from {artefact_path}")

    with open(artefact_path, 'r') as f:
        data = json.load(f)

    if 'amax_values' not in data:
        raise ValueError("Invalid artefact format: missing 'amax_values' key")

    metadata = data.get('metadata', {})
    amax_values = data['amax_values']

    logging.info(f"Loaded {len(amax_values)} amax values from artefact")
    logging.info(f"Artefact metadata: {metadata}")

    return data

def get_scale_fp8(amax: float, dtype: torch.dtype) -> torch.Tensor:
    scale = amax / torch.finfo(dtype).max
    scale_tensor = torch.tensor(scale, dtype=torch.float32)
    return scale_tensor

def get_scale_nvfp4(amax: float, dtype: torch.dtype) -> torch.Tensor:
    scale = amax / (F8_E4M3_MAX * F4_E2M1_MAX)
    scale_tensor = torch.tensor(scale, dtype=torch.float32)
    return scale_tensor

def get_scale(amax: float, dtype: torch.dtype):
    if dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
        return get_scale_fp8(amax, dtype)
    elif dtype in [torch.float4_e2m1fn_x2]:
        return get_scale_nvfp4(amax, dtype)
    else:
        raise ValueError(f"Unsupported dtype {dtype} ")

def apply_quantization(
    checkpoint: Dict,
    amax_values: Dict[str, float],
    config: QuantizationConfig
) -> Tuple[Dict, Dict]:
    quantized_dict = {}
    layer_metadata = {}

    for key, amax in amax_values.items():
        if key.endswith(".input_quantizer"):
            continue

        layer_name = ".".join(key.split(".")[:-1])

        if not config.should_quantize(layer_name):
            logging.debug(f"Layer {layer_name} disabled by config")
            continue

        dtype_str = config.get_dtype(layer_name)
        dtype = getattr(torch, dtype_str)
        device = torch.device("cuda") # Required for NVFP4

        weight = checkpoint.pop(f"{layer_name}.weight").to(device)
        scale_tensor = get_scale(amax, dtype)

        input_amax = amax_values.get(f"{layer_name}.input_quantizer", None)
        if input_amax is not None:
            input_scale = get_scale(input_amax, dtype)
            quantized_dict[f"{layer_name}.input_scale"] = input_scale.clone()

        # logging.info(f"Quantizing {layer_name}: amax={amax}, scale={scale_tensor:.6f}")
        tensor_layout = QUANT_FORMAT_MIXINS[dtype_str]["layout_type"]
        quantized_weight, layout_params = tensor_layout.quantize(
            weight,
            scale=scale_tensor,
            dtype=dtype
        )
        quantized_dict[f"{layer_name}.weight_scale"] = scale_tensor.clone()
        quantized_dict[f"{layer_name}.weight"] = quantized_weight.clone()

        if "block_scale" in layout_params:
            quantized_dict[f"{layer_name}.weight_block_scale"] = layout_params["block_scale"].clone()

        # Build metadata
        layer_metadata[layer_name] = {
            "format": dtype_str,
            "params": {}
        }

    logging.info(f"Quantized {len(layer_metadata)} layers")

    quantized_dict = quantized_dict | checkpoint

    metadata_dict = {
        "_quantization_metadata": json.dumps({
            "format_version": "1.0",
            "layers": layer_metadata
        })
    }
    return quantized_dict, metadata_dict


def main():
    """Main entry point for checkpoint merger."""

    parser = argparse.ArgumentParser(
        description="Merge calibration artifacts with checkpoint to create quantized model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--artefact",
        required=True,
        help="Path to calibration artefact JSON file (amax values)"
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to original checkpoint to quantize"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML quantization config file"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output path for quantized checkpoint"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )

    args = parser.parse_args()

    # Configure logging
    if args.debug:
        logging.basicConfig(
            level=logging.DEBUG,
            format='[%(levelname)s] %(name)s: %(message)s'
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            format='[%(levelname)s] %(message)s'
        )

    # Print header

    # Step 1: Load calibration artefact
    logging.info("[1/5] Loading calibration artefact...")
    try:
        artefact_data = load_amax_artefact(args.artefact)
        amax_values = artefact_data['amax_values']
    except Exception as e:
        logging.error(f"Failed to load artefact: {e}")
        sys.exit(1)

    # Step 2: Load quantization config
    logging.info("[2/5] Loading quantization config...")
    try:
        config = QuantizationConfig(args.config)
    except Exception as e:
        logging.error(f"Failed to load config: {e}")
        sys.exit(1)

    # Step 3: Load checkpoint
    logging.info("[3/5] Loading checkpoint...")
    try:
        checkpoint = comfy.utils.load_torch_file(args.checkpoint)
        logging.info(f"Loaded checkpoint with {len(checkpoint)} keys")
    except Exception as e:
        logging.error(f"Failed to load checkpoint: {e}")
        sys.exit(1)

    # Step 4: Apply quantization
    logging.info("[4/5] Applying quantization...")
    try:
        quantized_dict, metadata_json = apply_quantization(
            checkpoint,
            amax_values,
            config
        )
    except Exception as e:
        logging.error(f"Failed to apply quantization: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Step 5: Export quantized checkpoint
    logging.info("[5/5] Exporting quantized checkpoint...")
    try:
        save_file(quantized_dict, args.output, metadata=metadata_json)

    except Exception as e:
        logging.error(f"Failed to export checkpoint: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

