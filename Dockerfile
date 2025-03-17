# Nvidia GPU Base Images
# For NVIDIA GPU with stable CUDA version
# FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04 AS base

# For NVIDIA GPU with latest CUDA version
# FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04 AS base 

# AMD GPU Base Images
# For AMD GPU with stable ROCm version
# FROM rocm/dev-ubuntu-24.04:6.2.4-complete AS base 

# For AMD GPU with latest ROCm version
# FROM rocm/dev-ubuntu-24.04:6.3.4-complete AS base 

# Environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install necessary dependencies and Python 3.12
RUN apt-get update \
    && apt-get install -y \
    git \
    software-properties-common \
    curl \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    python3-setuptools \
    wget \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Clone the ComfyUI repository and set up virtual environment
RUN git clone https://github.com/comfyanonymous/ComfyUI.git /app/comfyui \
    && python3.12 -m venv /app/venv \
    && /app/venv/bin/pip install --upgrade pip \
    && /app/venv/bin/pip install pyyaml \
    && /app/venv/bin/pip install -r /app/comfyui/requirements.txt

# Clone ComfyUI-Manager and install its dependencies
RUN git clone https://github.com/ltdrdata/ComfyUI-Manager.git /app/temp/ComfyUI-Manager \
    && mv /app/temp/* /app/comfyui/custom_nodes/ \
    && rm -rf /app/temp \
    && /app/venv/bin/pip install -r /app/comfyui/custom_nodes/ComfyUI-Manager/requirements.txt

    # NVIDIA GPU PyTorch Installation

    # Install PyTorch with CUDA 12.6 support (stable version)
# RUN /app/venv/bin/pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu126

    # Install PyTorch with CUDA 12.8 support (latest version)
# RUN /app/venv/bin/pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

    # AMD GPU PyTorch Installation

    # Install PyTorch with ROCm 6.2 support (stable version)
# RUN /app/venv/bin/pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2.4

    # Install PyTorch with ROCm 6.3 support (latest version)
# RUN /app/venv/bin/pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm6.3


# Expose the backend port
EXPOSE 8188

# Set the entrypoint to run the app
CMD ["/bin/bash", "-c", "source /app/venv/bin/activate && python3 /app/comfyui/main.py --listen 0.0.0.0 --port 8188"]
