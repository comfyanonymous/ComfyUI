# RunPod Serverless ComfyUI Worker
FROM runpod/pytorch:2.2.0-py3.11-cuda12.1.1-devel-ubuntu22.04

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    unzip \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy ComfyUI source code
COPY . /workspace/ComfyUI

# Set ComfyUI as working directory
WORKDIR /workspace/ComfyUI

# Install Python dependencies
RUN pip install --no-cache-dir \
    runpod \
    requests \
    pillow \
    numpy \
    torch \
    torchvision \
    torchaudio \
    xformers \
    accelerate \
    transformers \
    diffusers \
    opencv-python \
    scipy \
    scikit-image \
    safetensors \
    bitsandbytes \
    optimum

# Install ComfyUI requirements if exists
RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi

# Create necessary directories
RUN mkdir -p /workspace/ComfyUI/models/checkpoints \
    /workspace/ComfyUI/models/vae \
    /workspace/ComfyUI/models/loras \
    /workspace/ComfyUI/models/controlnet \
    /workspace/ComfyUI/models/clip_vision \
    /workspace/ComfyUI/models/upscale_models \
    /workspace/ComfyUI/input \
    /workspace/ComfyUI/output \
    /tmp/inputs \
    /tmp/outputs \
    /tmp/comfyui

# Set environment variables
ENV PYTHONPATH="/workspace/ComfyUI:${PYTHONPATH}"
ENV COMFYUI_SERVERLESS=true
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Set environment variables
ENV PYTHONPATH="/workspace/ComfyUI:${PYTHONPATH}"
ENV COMFYUI_SERVERLESS=true

# Expose port
EXPOSE 8000

# Start command
CMD ["python3", "runpod_handler.py"]