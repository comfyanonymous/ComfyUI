FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ffmpeg \
    git \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt ./
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url ${TORCH_INDEX_URL} \
    && grep -vE '^(torch|torchvision|torchaudio)($|[<>=~ ])' requirements.txt > /tmp/requirements-no-torch.txt \
    && pip install --no-cache-dir -r /tmp/requirements-no-torch.txt

COPY manager_requirements.txt ./
RUN pip install --no-cache-dir -r manager_requirements.txt

COPY . .
RUN find custom_nodes -mindepth 2 -maxdepth 2 -name requirements.txt -print -exec sh -c \
    'grep -vE "^(torch|torchvision|torchaudio|onnxruntime-gpu)($|[<>=~ ])" "$1" > /tmp/custom-node-requirements.txt && if [ -s /tmp/custom-node-requirements.txt ]; then pip install --no-cache-dir -r /tmp/custom-node-requirements.txt; fi' sh {} \;

EXPOSE 8188
CMD ["python", "main.py", "--listen", "0.0.0.0", "--port", "8188", "--enable-manager"]
