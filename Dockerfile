FROM nvidia/cuda:12.1.0-base-ubuntu20.04 AS base
ENV DEBIAN_FRONTEND=noninteractive

# Install necessary dependencies and Python 3.12
RUN apt-get update \
    && apt-get install -y \
    git \
    software-properties-common \
    curl \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y \
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

# Install pip for Python 3.12 explicitly
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py \
    && python3.12 get-pip.py \
    && rm get-pip.py

# Set Python 3.12 as the default python3 and pip
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/local/bin/pip3 1

# Set the working directory
WORKDIR /app

# Clone the ComfyUI repository into the working directory
RUN git clone https://github.com/comfyanonymous/ComfyUI.git /app/comfyui

# Install backend dependencies
RUN python3 -m venv /app/venv
RUN . venv/bin/activate && pip install --upgrade pip
RUN . venv/bin/activate && pip install pyyaml
RUN . venv/bin/activate && pip install -r /app/comfyui/requirements.txt

# Install PyTorch with CUDA 12.1 support
RUN . venv/bin/activate && pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121

# Clone the ComfyUI-manager repository into a temporary directory, move it, and clean up
RUN git clone https://github.com/ltdrdata/ComfyUI-Manager.git /app/temp/ComfyUI-Manager && \
    mv /app/temp/* /app/comfyui/custom_nodes/ && \
    rm -rf /app/temp

# Install ComfyUI-manager dependencies
RUN . venv/bin/activate && pip install -r /app/comfyui/custom_nodes/ComfyUI-Manager/requirements.txt

# Expose the backend port
EXPOSE 8188

# Set the entrypoint command to activate the virtual environment and run the script
CMD ["/bin/bash", "-c", "source /app/venv/bin/activate && python3 /app/comfyui/main.py --listen 0.0.0.0 --port 8188"]
