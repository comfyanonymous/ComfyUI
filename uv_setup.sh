#!/bin/bash
# ComfyUI UV Environment Setup Script
# Created: May 2, 2025

# Set UV environment variables for better performance
export UV_CACHE_DIR="${HOME}/.cache/uv"
export UV_SYSTEM_PYTHON=false
export UV_DEFAULT_PYTHON=3.11
export UV_THREADS=auto
export UV_VERBOSITY=1

# Create directory structure if needed
mkdir -p "${UV_CACHE_DIR}"

echo "UV environment variables set:"
echo "  UV_CACHE_DIR: ${UV_CACHE_DIR}"
echo "  UV_DEFAULT_PYTHON: ${UV_DEFAULT_PYTHON}"
echo "  UV_THREADS: ${UV_THREADS}"

# Check if UV is installed
if ! command -v uv >/dev/null 2>&1; then
    echo "UV not found. Installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    
    # Add to PATH for this session
    source $HOME/.local/bin/env
fi

# Check for ComfyUI virtual environment
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    uv venv
    echo "Virtual environment created at .venv"
fi

echo ""
echo "To activate the virtual environment, run:"
echo "  source .venv/bin/activate     # On Linux/macOS"
echo "  .venv\\Scripts\\activate.bat  # On Windows"
echo ""
echo "To install/update dependencies:"
echo "  ./update_dependencies_uv.py --advanced --cuda --lock"
echo ""
echo "To start ComfyUI:"
echo "  python main.py"