# Installation Guide

This document provides detailed instructions for installing ComfyUI using different methods.

## Table of Contents
- [System Requirements](#system-requirements)
- [Installation Methods](#installation-methods)
  - [Option 1: Install as a Package (Recommended)](#option-1-install-as-a-package-recommended)
  - [Option 2: Install from Source](#option-2-install-from-source)
  - [Option 3: Docker Installation](#option-3-docker-installation)
- [GPU Setup](#gpu-setup)
- [Troubleshooting](#troubleshooting)

## System Requirements

### Minimum Requirements
- Python 3.9 or newer (Python 3.11 recommended)
- NVIDIA GPU with at least 4GB VRAM
- 8GB RAM
- 2GB free disk space

### Recommended Specifications
- Python 3.11
- NVIDIA GPU with 8GB+ VRAM (RTX series recommended)
- 16GB RAM
- SSD with 10GB+ free space

## Installation Methods

### Option 1: Install as a Package (Recommended)

The package installation method is the simplest way to get started with ComfyUI.

#### Install with UV (Fastest)

[UV](https://github.com/astral-sh/uv) is a fast, reliable package manager for Python. We recommend using it for the best installation experience:

```bash
# Install UV if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create and activate a virtual environment
uv venv .venv
source .venv/bin/activate  # On Linux/macOS
# OR
.venv\Scripts\activate     # On Windows

# Install ComfyUI with GPU support
uv pip install "comfyui[gpu]"

# Run ComfyUI
comfyui
```

#### Install with Pip

If you prefer using standard pip:

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Linux/macOS
# OR
venv\Scripts\activate     # On Windows

# Install ComfyUI
pip install "comfyui[gpu]"

# Run ComfyUI
comfyui
```

### Option 2: Install from Source

Installing from source gives you the latest development version and more control over the installation process.

```bash
# Clone the repository
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI

# Method A: Use the setup script (recommended)
python uv_setup.py --gpu --advanced

# Method B: Manual installation
# Create virtual environment
uv venv .venv
source .venv/bin/activate  # On Linux/macOS
# OR
.venv\Scripts\activate     # On Windows

# Install dependencies
uv pip install -r requirements.txt

# Run ComfyUI
python main.py
```

### Option 3: Docker Installation

For users who prefer containerized applications:

```bash
# Pull and run the ComfyUI Docker image
docker pull comfyanonymous/comfyui:latest
docker run -p 8188:8188 -v /path/to/models:/app/models comfyanonymous/comfyui:latest
```

## GPU Setup

### NVIDIA GPUs

ComfyUI works best with NVIDIA GPUs using CUDA. To leverage GPU acceleration:

1. Ensure you have the latest NVIDIA drivers installed
2. Install the CUDA-enabled version of ComfyUI:
   ```bash
   uv pip install "comfyui[gpu]"
   ```

### AMD GPUs

For AMD GPUs, follow these additional steps:

1. Install ROCm (for compatible AMD GPUs)
2. Install PyTorch with ROCm support
3. Configure ComfyUI for AMD GPU usage:
   ```bash
   python main.py --use-rocm
   ```

## Troubleshooting

### Common Issues

#### CUDA Out of Memory
If you encounter CUDA out of memory errors:
- Try using `--disable-cuda-malloc` flag
- Lower model precision using the `--fp16` flag
- Use smaller-sized models

#### Missing Dependencies
If you encounter missing dependencies:
```bash
uv pip install --upgrade -r requirements.txt
```

#### Package Conflicts
If you have conflicting packages:
```bash
uv pip install --force-reinstall "comfyui[gpu]"
```

### Getting Help

If you encounter issues not covered here:
- Check the [Discord server](https://comfy.org/discord) for community support
- Search for similar issues in the [GitHub repository](https://github.com/comfyanonymous/ComfyUI/issues)
- Run with `--verbose` flag for detailed logs:
  ```bash
  comfyui --verbose
  ```