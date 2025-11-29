#!/usr/bin/env python3
"""
Convert quantized checkpoints (SVDQuant, AWQ, or mixed) into the ComfyUI quantization format.
"""

import argparse
from pathlib import Path
from safetensors import safe_open

from comfy.svdquant_converter import (
    convert_quantized_file,
    convert_svdquant_file,
    convert_awq_file,
    detect_quantization_formats,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert quantized .safetensors files (SVDQuant, AWQ, or mixed) "
        "into the ComfyUI format with per-layer metadata for MixedPrecisionOps."
    )
    parser.add_argument("input", type=Path, help="Path to the source quantized .safetensors file.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Destination path for the converted checkpoint. "
        "Defaults to <input_name>_comfy.safetensors in the same directory.",
    )
    parser.add_argument(
        "--format-version",
        default="1.0",
        help="Format version to store inside _quantization_metadata (default: 1.0).",
    )
    parser.add_argument(
        "--format",
        choices=["auto", "svdquant", "awq"],
        default="auto",
        help="Quantization format (default: auto-detect).",
    )
    parser.add_argument(
        "--detect-only",
        action="store_true",
        help="Only detect and report quantization formats without converting.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    input_path = args.input.expanduser().resolve()
    
    # Detect formats if requested
    if args.detect_only:
        print(f"[Quantization Detector] Analyzing: {input_path}")
        with safe_open(str(input_path), framework="pt", device="cpu") as f:
            tensors = {key: f.get_tensor(key) for key in f.keys()}
        
        formats = detect_quantization_formats(tensors)
        
        if not formats:
            print("[Quantization Detector] No quantized layers detected.")
            return
        
        print(f"[Quantization Detector] Detected formats:")
        total_layers = 0
        for format_name, layer_prefixes in sorted(formats.items()):
            print(f"\n  {format_name}: {len(layer_prefixes)} layers")
            for prefix in sorted(layer_prefixes)[:5]:  # Show first 5
                print(f"    - {prefix}")
            if len(layer_prefixes) > 5:
                print(f"    ... and {len(layer_prefixes) - 5} more")
            total_layers += len(layer_prefixes)
        
        print(f"\n[Quantization Detector] Total: {total_layers} quantized layers")
        print(f"[Quantization Detector] Use without --detect-only to convert.")
        return
    
    # Convert checkpoint
    if args.output is None:
        output_path = input_path.with_name(f"{input_path.stem}_comfy.safetensors")
    else:
        output_path = args.output.expanduser().resolve()

    layer_count, quant_layers = convert_quantized_file(
        str(input_path),
        str(output_path),
        format_version=args.format_version,
        quant_format=args.format,
    )
    
    # Group layers by format for display
    format_groups = {}
    for layer_name, fmt in quant_layers.items():
        if fmt not in format_groups:
            format_groups[fmt] = []
        format_groups[fmt].append(layer_name)
    
    print(f"[Quantization Converter] Converted {layer_count} layers.")
    print(f"[Quantization Converter] Output saved to: {output_path}")
    print(f"\n[Quantization Converter] Quantized layers by format:")
    
    for fmt, layers in sorted(format_groups.items()):
        print(f"\n  {fmt}: {len(layers)} layers")
        for layer_name in sorted(layers)[:5]:  # Show first 5
            print(f"    - {layer_name}")
        if len(layers) > 5:
            print(f"    ... and {len(layers) - 5} more")


if __name__ == "__main__":
    main()

