FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

#  System packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git ca-certificates curl python3 python3-venv python3-pip && \
    rm -rf /var/lib/apt/lists/*

#  Fetch ComfyUI source
WORKDIR /opt/ComfyUI
COPY . /opt/ComfyUI

#  Python venv + pinned deps
RUN python3 -m venv /venv && \
    . /venv/bin/activate && \
    pip install --upgrade pip && \
    pip install --no-cache-dir "numpy<2" && \
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 \
        torch==2.2.2+cu121 torchvision==0.17.2+cu121 torchaudio==2.2.2+cu121 && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu121 \
        xformers==0.0.25.post1 triton==2.2.0 || true

ENV PATH="/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    COMFYUI_ROOT=/opt/ComfyUI

# Allow CLI flags via env
ENV CLI_ARGS=""

EXPOSE 8188

ENTRYPOINT ["bash", "-c", "python3 main.py --listen 0.0.0.0 --port 8188 $CLI_ARGS"]