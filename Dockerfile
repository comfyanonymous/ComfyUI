# Build argument for base image selection
ARG BASE_IMAGE=nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

# ----------------------
# Stage: Base Runtime
# ----------------------
FROM ${BASE_IMAGE} AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_PREFER_BINARY=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.12 python3.12-venv git git-lfs wget \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    ffmpeg \
    espeak-ng libespeak-ng1 \
    build-essential \
    && git lfs install \
    && ln -sf /usr/bin/python3.12 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip \
    && apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Install OpenTelemetry packages for optional instrumentation
RUN pip install --no-cache-dir opentelemetry-distro opentelemetry-exporter-otlp

WORKDIR /app/ComfyUI

# Copy entrypoint and helper scripts
COPY scripts/docker-entrypoint.sh /app/ComfyUI/scripts/docker-entrypoint.sh
COPY scripts/comfy-node-install.sh /usr/local/bin/comfy-node-install
RUN chmod +x /app/ComfyUI/scripts/docker-entrypoint.sh && \
    chmod +x /usr/local/bin/comfy-node-install

# Set entrypoint
ENTRYPOINT ["/app/ComfyUI/scripts/docker-entrypoint.sh"]

