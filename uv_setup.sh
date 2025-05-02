#!/bin/bash
# ComfyUI UV Setup Script

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is required but not found. Please install Python 3.9+ first."
    exit 1
fi

# Make Python script executable
chmod +x ./uv_setup.py

# Run the Python setup script with passed arguments
python3 ./uv_setup.py "$@"