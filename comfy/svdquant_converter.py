import json
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from safetensors import safe_open
from safetensors.torch import save_file


# Note: Fused layer splitting is no longer used


@dataclass
class ConvertedState:
    tensors: Dict[str, torch.Tensor]
    quant_layers: Dict[str, str]


def _is_svd_prefix(keys: set[str], prefix: str) -> bool:
    return (
        f"{prefix}.qweight" in keys
        and f"{prefix}.smooth_factor" in keys
        and f"{prefix}.proj_down" in keys
        and f"{prefix}.proj_up" in keys
    )


def _is_awq_prefix(keys: set[str], prefix: str) -> bool:
    return (
        f"{prefix}.qweight" in keys
        and f"{prefix}.wscales" in keys
        and f"{prefix}.wzeros" in keys
        and f"{prefix}.smooth_factor" not in keys  # Distinguish from SVDQuant
    )


def _detect_svd_prefixes(state_dict: Dict[str, torch.Tensor]) -> List[str]:
    prefixes = set()
    keys = set(state_dict.keys())
    for key in keys:
        if not key.endswith(".qweight"):
            continue
        prefix = key[: -len(".qweight")]
        if _is_svd_prefix(keys, prefix):
            prefixes.add(prefix)
    return sorted(prefixes)


def _detect_awq_prefixes(state_dict: Dict[str, torch.Tensor]) -> List[str]:
    prefixes = set()
    keys = set(state_dict.keys())
    for key in keys:
        if not key.endswith(".qweight"):
            continue
        prefix = key[: -len(".qweight")]
        if _is_awq_prefix(keys, prefix):
            prefixes.add(prefix)
    return sorted(prefixes)


def _detect_format(wscales: torch.Tensor) -> str:
    if wscales.dtype == torch.float8_e4m3fn:
        return "svdquant_nvfp4"
    return "svdquant_int4"


class _SVDQuantConverter:
    def __init__(self, state_dict: Dict[str, torch.Tensor]) -> None:
        self.src = dict(state_dict)
        self.dst: Dict[str, torch.Tensor] = {}
        self.quant_layers: Dict[str, str] = {}

    def convert(self) -> ConvertedState:
        prefixes = _detect_svd_prefixes(self.src)
        for prefix in prefixes:
            self._convert_single(prefix)

        for key, tensor in self.src.items():
            if key not in self.dst:
                self.dst[key] = tensor

        return ConvertedState(self.dst, self.quant_layers)

    def _pop_tensor(self, key: str) -> torch.Tensor:
        try:
            return self.src.pop(key)
        except KeyError as exc:
            raise KeyError(f"Missing key '{key}' in SVDQuant checkpoint") from exc

    def _pop_optional(self, key: str) -> torch.Tensor | None:
        return self.src.pop(key, None)

    def _convert_single(self, prefix: str) -> None:
        # Ensure all tensors are contiguous to avoid CUDA alignment issues
        self.dst[f"{prefix}.weight"] = self._pop_tensor(f"{prefix}.qweight").contiguous()
        wscales = self._pop_tensor(f"{prefix}.wscales").contiguous()
        self.dst[f"{prefix}.wscales"] = wscales
        format_name = _detect_format(wscales)

        self.dst[f"{prefix}.smooth_factor"] = self._pop_tensor(f"{prefix}.smooth_factor").contiguous()
        self.dst[f"{prefix}.smooth_factor_orig"] = self._pop_tensor(
            f"{prefix}.smooth_factor_orig"
        ).contiguous()
        self.dst[f"{prefix}.proj_down"] = self._pop_tensor(f"{prefix}.proj_down").contiguous()
        self.dst[f"{prefix}.proj_up"] = self._pop_tensor(f"{prefix}.proj_up").contiguous()

        bias = self._pop_optional(f"{prefix}.bias")
        if bias is not None:
            self.dst[f"{prefix}.bias"] = bias.contiguous()

        wtscale = self._pop_optional(f"{prefix}.wtscale")
        if wtscale is not None:
            self.dst[f"{prefix}.wtscale"] = wtscale.contiguous() if isinstance(wtscale, torch.Tensor) else wtscale

        wcscales = self._pop_optional(f"{prefix}.wcscales")
        if wcscales is not None:
            self.dst[f"{prefix}.wcscales"] = wcscales.contiguous()

        self.quant_layers[prefix] = format_name


class _AWQConverter:
    def __init__(self, state_dict: Dict[str, torch.Tensor]) -> None:
        self.src = dict(state_dict)
        self.dst: Dict[str, torch.Tensor] = {}
        self.quant_layers: Dict[str, str] = {}

    def convert(self) -> ConvertedState:
        prefixes = _detect_awq_prefixes(self.src)
        for prefix in prefixes:
            self._convert_single(prefix)

        for key, tensor in self.src.items():
            if key not in self.dst:
                self.dst[key] = tensor

        return ConvertedState(self.dst, self.quant_layers)

    def _pop_tensor(self, key: str) -> torch.Tensor:
        try:
            return self.src.pop(key)
        except KeyError as exc:
            raise KeyError(f"Missing key '{key}' in AWQ checkpoint") from exc

    def _pop_optional(self, key: str) -> torch.Tensor | None:
        return self.src.pop(key, None)

    def _convert_single(self, prefix: str) -> None:
        # Ensure all tensors are contiguous to avoid CUDA alignment issues
        self.dst[f"{prefix}.weight"] = self._pop_tensor(f"{prefix}.qweight").contiguous()
        self.dst[f"{prefix}.wscales"] = self._pop_tensor(f"{prefix}.wscales").contiguous()
        self.dst[f"{prefix}.wzeros"] = self._pop_tensor(f"{prefix}.wzeros").contiguous()

        bias = self._pop_optional(f"{prefix}.bias")
        if bias is not None:
            self.dst[f"{prefix}.bias"] = bias.contiguous()

        self.quant_layers[prefix] = "awq_int4"


def convert_svdquant_state_dict(state_dict: Dict[str, torch.Tensor]) -> ConvertedState:
    return _SVDQuantConverter(state_dict).convert()


def convert_awq_state_dict(state_dict: Dict[str, torch.Tensor]) -> ConvertedState:
    return _AWQConverter(state_dict).convert()


def detect_quantization_formats(state_dict: Dict[str, torch.Tensor]) -> Dict[str, List[str]]:
    """
    Detect quantization formats present in a state dict.
    
    Parameters
    ----------
    state_dict : Dict[str, torch.Tensor]
        State dictionary to analyze
    
    Returns
    -------
    Dict[str, List[str]]
        Dictionary mapping format names to lists of layer prefixes
        Example: {
            "svdquant_int4": ["layer1.attn.qkv", "layer2.mlp.up"],
            "svdquant_nvfp4": ["layer3.attn.qkv"],
            "awq_int4": ["layer1.mlp.down", "layer4.attn.qkv"]
        }
    """
    result = {}
    
    # Detect SVDQuant layers
    svd_prefixes = _detect_svd_prefixes(state_dict)
    if svd_prefixes:
        # Determine if int4 or nvfp4 based on wscales dtype
        for prefix in svd_prefixes:
            wscales_key = f"{prefix}.wscales"
            if wscales_key in state_dict:
                format_name = _detect_format(state_dict[wscales_key])
                if format_name not in result:
                    result[format_name] = []
                result[format_name].append(prefix)
    
    # Detect AWQ layers
    awq_prefixes = _detect_awq_prefixes(state_dict)
    if awq_prefixes:
        result["awq_int4"] = awq_prefixes
    
    return result


def convert_awq_file(
    input_path: str,
    output_path: str,
    format_version: str = "1.0",
) -> Tuple[int, Dict[str, str]]:
    with safe_open(input_path, framework="pt", device="cpu") as f:
        tensors = {key: f.get_tensor(key) for key in f.keys()}
        metadata = dict(f.metadata())

    converted = convert_awq_state_dict(tensors)
    
    # Convert layer format dict to expected metadata format
    # From: {"layer": "awq_int4"}
    # To: {"layer": {"format": "awq_int4"}}
    layers_metadata = {k: {"format": v} for k, v in converted.quant_layers.items()}
    
    metadata["_quantization_metadata"] = json.dumps(
        {"format_version": format_version, "layers": layers_metadata}, sort_keys=True
    )

    save_file(converted.tensors, output_path, metadata=metadata)
    return len(converted.quant_layers), converted.quant_layers


def convert_svdquant_file(
    input_path: str,
    output_path: str,
    format_version: str = "1.0",
) -> Tuple[int, Dict[str, str]]:
    with safe_open(input_path, framework="pt", device="cpu") as f:
        tensors = {key: f.get_tensor(key) for key in f.keys()}
        metadata = dict(f.metadata())

    converted = convert_svdquant_state_dict(tensors)
    
    # Convert layer format dict to expected metadata format
    # From: {"layer": "svdquant_int4"}
    # To: {"layer": {"format": "svdquant_int4"}}
    layers_metadata = {k: {"format": v} for k, v in converted.quant_layers.items()}
    
    metadata["_quantization_metadata"] = json.dumps(
        {"format_version": format_version, "layers": layers_metadata}, sort_keys=True
    )
    metadata["model_class"] = "QwenImageTransformer2DModel"

    save_file(converted.tensors, output_path, metadata=metadata)
    return len(converted.quant_layers), converted.quant_layers


def convert_quantized_file(
    input_path: str,
    output_path: str,
    format_version: str = "1.0",
    quant_format: str = "auto",
) -> Tuple[int, Dict[str, str]]:
    """
    Auto-detect and convert quantized checkpoint to ComfyUI format.
    
    Supports mixed-format models where some layers are SVDQuant and others are AWQ.
    Each layer is independently detected and converted to the appropriate format.
    
    Parameters
    ----------
    input_path : str
        Path to input checkpoint file
    output_path : str
        Path to output checkpoint file
    format_version : str, optional
        Quantization metadata format version (default: "1.0")
    quant_format : str, optional
        Quantization format: "auto", "svdquant", or "awq" (default: "auto")
    
    Returns
    -------
    Tuple[int, Dict[str, str]]
        Number of quantized layers and mapping of layer names to formats
    """
    with safe_open(input_path, framework="pt", device="cpu") as f:
        tensors = {key: f.get_tensor(key) for key in f.keys()}
        metadata = dict(f.metadata())

    # Auto-detect format if needed
    if quant_format == "auto":
        svd_prefixes = _detect_svd_prefixes(tensors)
        awq_prefixes = _detect_awq_prefixes(tensors)
        
        if svd_prefixes and awq_prefixes:
            # Mixed format - partition tensors by format and convert separately
            
            # Build sets of all quantized prefixes
            all_svd_prefixes = set(svd_prefixes)
            all_awq_prefixes = set(awq_prefixes)
            
            # Helper to check if a key belongs to a specific quantized layer
            def belongs_to_prefix(key, prefix):
                """Check if key belongs to a specific layer prefix."""
                return key == prefix or key.startswith(f"{prefix}.")
            
            def is_svd_key(key):
                """Check if key belongs to any SVDQuant layer."""
                return any(belongs_to_prefix(key, prefix) for prefix in all_svd_prefixes)
            
            def is_awq_key(key):
                """Check if key belongs to any AWQ layer."""
                return any(belongs_to_prefix(key, prefix) for prefix in all_awq_prefixes)
            
            # Partition tensors by format
            svd_tensors = {}
            awq_tensors = {}
            other_tensors = {}
            
            for key, tensor in tensors.items():
                if is_svd_key(key):
                    svd_tensors[key] = tensor
                elif is_awq_key(key):
                    awq_tensors[key] = tensor
                else:
                    other_tensors[key] = tensor
            
            # Convert each format separately with only its relevant tensors
            svd_converted = _SVDQuantConverter(svd_tensors).convert()
            awq_converted = _AWQConverter(awq_tensors).convert()
            
            # Merge results - each converter only has its own layer tensors
            converted_tensors = {}
            
            # Add SVDQuant converted tensors
            converted_tensors.update(svd_converted.tensors)
            
            # Add AWQ converted tensors
            converted_tensors.update(awq_converted.tensors)
            
            # Add non-quantized tensors
            converted_tensors.update(other_tensors)
            
            # Merge quantization layer metadata
            quant_layers = {}
            quant_layers.update(svd_converted.quant_layers)
            quant_layers.update(awq_converted.quant_layers)
            
            converted = ConvertedState(converted_tensors, quant_layers)
        elif svd_prefixes:
            converted = convert_svdquant_state_dict(tensors)
        elif awq_prefixes:
            converted = convert_awq_state_dict(tensors)
        else:
            raise ValueError("No quantized layers detected in checkpoint")
    elif quant_format == "svdquant":
        converted = convert_svdquant_state_dict(tensors)
    elif quant_format == "awq":
        converted = convert_awq_state_dict(tensors)
    else:
        raise ValueError(f"Unknown quantization format: {quant_format}")

    # Convert layer format dict to expected metadata format
    # From: {"layer": "awq_int4"}
    # To: {"layer": {"format": "awq_int4"}}
    layers_metadata = {k: {"format": v} for k, v in converted.quant_layers.items()}
    
    metadata["_quantization_metadata"] = json.dumps(
        {"format_version": format_version, "layers": layers_metadata}, sort_keys=True
    )
    metadata["model_class"] = "QwenImageTransformer2DModel"

    save_file(converted.tensors, output_path, metadata=metadata)
    return len(converted.quant_layers), converted.quant_layers


