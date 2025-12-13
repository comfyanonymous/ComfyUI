#!/bin/bash
# Docker entrypoint script for ComfyUI
# Handles virtual environment setup, dependency installation, custom nodes, and optional OTEL instrumentation

set -e

COMFYUI_DIR="/app/ComfyUI"
VENV_DIR="/app/venv"
WORKDIR="${COMFYUI_DIR}"

cd "${WORKDIR}"

# Create virtual environment if it doesn't exist
if [ ! -d "${VENV_DIR}" ]; then
    echo "Creating virtual environment..."
    python -m venv "${VENV_DIR}"
fi

# Activate virtual environment
source "${VENV_DIR}/bin/activate"

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 12.6 support
echo "Installing PyTorch with CUDA 12.6..."
pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Install ComfyUI requirements (upgrade ensures packages match requirements.txt)
echo "Installing ComfyUI requirements..."
pip install --upgrade --no-cache-dir -r requirements.txt

# Install OpenTelemetry packages if OTEL endpoint is configured
if [ -n "${OTEL_EXPORTER_OTLP_ENDPOINT}" ]; then
    echo "Installing OpenTelemetry packages..."
    pip install --no-cache-dir opentelemetry-distro opentelemetry-exporter-otlp
fi

# Explicitly ensure frontend package matches requirements.txt version
echo "Verifying frontend package version..."
REQUIRED_FRONTEND_VERSION=$(grep "^comfyui-frontend-package==" requirements.txt | cut -d'=' -f3)
if [ -n "$REQUIRED_FRONTEND_VERSION" ]; then
    echo "Installing frontend package version: $REQUIRED_FRONTEND_VERSION"
    pip install --upgrade --force-reinstall --no-cache-dir "comfyui-frontend-package==${REQUIRED_FRONTEND_VERSION}"
else
    echo "Warning: Could not determine required frontend version from requirements.txt"
fi

# Optional: Install curated custom nodes (can be enabled via environment variable)
if [ "${INSTALL_CURATED_NODES:-false}" = "true" ]; then
    echo "Installing curated custom nodes..."
    export COMFYUI_DIR="${COMFYUI_DIR}"
    comfy-node-install \
        https://github.com/city96/ComfyUI-GGUF \
        https://github.com/rgthree/rgthree-comfy \
        https://github.com/ClownsharkBatwing/RES4LYF \
        https://github.com/giriss/comfy-image-saver || echo "Warning: Some custom nodes failed to install"
fi

# Check if OpenTelemetry endpoint is configured
if [ -n "${OTEL_EXPORTER_OTLP_ENDPOINT}" ]; then
    echo "OpenTelemetry endpoint detected, enabling instrumentation..."
    exec opentelemetry-instrument \
        --traces_exporter otlp \
        --metrics_exporter otlp \
        --logs_exporter otlp \
        python main.py --listen 0.0.0.0 --port 8188 "$@"
else
    echo "Starting ComfyUI without OpenTelemetry instrumentation..."
    exec python main.py --listen 0.0.0.0 --port 8188 "$@"
fi

