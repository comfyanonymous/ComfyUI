#!/bin/bash
# Start ComfyUI with AMD Radeon 7900 XTX (ROCm)

# Set ROCm architecture for Radeon 7900 XTX
echo "Setting ROCm environment for Radeon 7900 XTX (gfx1100)"
export PYTORCH_ROCM_ARCH=gfx1100
# Uncomment the next line if you encounter issues with ROCm version detection
export HSA_OVERRIDE_GFX_VERSION=11.0.0

# Optionally, activate your Python virtual environment here
source ../venv/bin/activate

# Start ComfyUI
python3 main.py --lowvram --use-split-cross-attention --fp16-unet --listen 0.0.0.0 "$@" 
