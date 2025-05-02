#!/usr/bin/env python3
"""
ComfyUI Command Line Interface
"""

import os
import sys
import argparse
from pathlib import Path


def add_comfyui_to_path():
    """Add the ComfyUI package directory to the Python path."""
    # In development mode, running from source directory
    if os.path.exists(os.path.join(os.path.dirname(__file__), "main.py")):
        comfyui_path = os.path.dirname(__file__)
    else:
        # In installed mode, comfyui module path
        comfyui_path = os.path.dirname(os.path.abspath(__file__))
        
    if comfyui_path not in sys.path:
        sys.path.insert(0, comfyui_path)


def get_parser():
    """Create and return the argument parser for ComfyUI."""
    parser = argparse.ArgumentParser(
        description="ComfyUI - A Stable Diffusion GUI with a graph/nodes interface"
    )
    
    parser.add_argument(
        "--host", type=str, default="127.0.0.1",
        help="The host to listen on (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port", type=int, default=8188,
        help="The port to listen on (default: 8188)"
    )
    parser.add_argument(
        "--enable-cors-header", type=str, default=None,
        help="Enable CORS by setting the Access-Control-Allow-Origin header (default: disabled)"
    )
    parser.add_argument(
        "--cuda-device", type=int, default=None,
        help="The CUDA device to use for inference (default: auto)"
    )
    parser.add_argument(
        "--disable-cuda-malloc", action="store_true",
        help="Force PyTorch to use standard memory allocator instead of cudaMallocAsync"
    )
    parser.add_argument(
        "--auto-launch", action="store_true",
        help="Open the UI in default browser at startup"
    )
    parser.add_argument(
        "--output-directory", type=str, default=None,
        help="Override the default output directory"
    )
    parser.add_argument(
        "--input-directory", type=str, default=None,
        help="Override the default input directory"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable verbose logging"
    )
    
    return parser


def main():
    """Main entry point for the ComfyUI application."""
    add_comfyui_to_path()
    
    parser = get_parser()
    args, unknown = parser.parse_known_args()
    
    # Pass arguments to the original ComfyUI entry point
    sys.argv = [sys.argv[0]] + unknown
    
    # Set environment variables based on arguments
    if args.cuda_device is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_device)
        os.environ['HIP_VISIBLE_DEVICES'] = str(args.cuda_device)
    
    if args.disable_cuda_malloc:
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'backend:cudaMalloc'
    
    # Import and run the main ComfyUI entry point
    import main
    
    # Override command line arguments with the ones from our parser
    import comfy.cli_args
    comfy_args = comfy.cli_args.args
    
    # Set args from our parser to the ComfyUI args object
    comfy_args.listen = args.host
    comfy_args.port = args.port
    comfy_args.enable_cors_header = args.enable_cors_header
    comfy_args.cuda_device = args.cuda_device
    comfy_args.disable_cuda_malloc = args.disable_cuda_malloc
    comfy_args.auto_launch = args.auto_launch
    
    if args.output_directory:
        comfy_args.output_directory = args.output_directory
    if args.input_directory:
        comfy_args.input_directory = args.input_directory
    comfy_args.verbose = args.verbose
    
    # Start ComfyUI
    asyncio_loop, _, start_all_func = main.start_comfyui()
    try:
        x = start_all_func()
        asyncio_loop.run_until_complete(x)
    except KeyboardInterrupt:
        print("\nStopped server")
    
    main.cleanup_temp()


if __name__ == "__main__":
    main()