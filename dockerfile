# Already installed nvidia cuda drivers to leverage GPU
FROM --platform=linux/amd64 nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# environment variables
ENV DEBIAN_FRONTEND=noninteractive \
                        TZ=Etc/UTC

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    python3 \
    python3-pip \
    python3-dev \
    wget \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/ComfyUI
COPY . .
RUN pip3 install --upgrade pip
RUN pip3 install --no-cache-dir -r requirements.txt

EXPOSE 8188

# Map volume in docker-compose file (so no need to map explicitly)
# RUN mkdir -p /workspace/ComfyUI/models
# VOLUME ["/workspace/ComfyUI/models"]

# python3 main.py --dont-print-server --listen 0.0.0.0 --port 8188
# Override Base command of container
ENTRYPOINT ["python3", "main.py", "--dont-print-server"]
# Pass extra args
CMD ["--listen", "0.0.0.0", "--port", "8188"]