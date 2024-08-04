#!/bin/bash

# Check for git
if command -v git >/dev/null 2>&1; then
    echo "Git is already installed."
else
    echo "Git is not installed. Installing..."

    sudo apt update
    sudo apt install -y git

    echo "Git has been installed."
fi

# Check for Python3.8
if command -v python3 >/dev/null 2>&1; then
    echo "Python3.8 is already installed."
else
    echo "Installing Python3.8..."

    sudo apt-get update
    sudo apt-get install -y python3.8

    echo "Python3.8 has been installed."
fi

# Check for NVIDIA drivers
if nvidia-smi >/dev/null 2>&1; then
    echo "NVIDIA drivers are already installed."
else
    echo "Installing NVIDIA driver..."

    sudo apt update
    sudo apt install -y nvidia-driver-535

    echo "NVIDIA driver has been installed."
fi

# Check for CUDA
if command -v nvcc >/dev/null 2>&1; then
    echo "CUDA is already installed."
else
    echo "Installing CUDA..."

    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
    sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
    wget https://developer.download.nvidia.com/compute/cuda/12.6.0/local_installers/cuda-repo-ubuntu2204-12-6-local_12.6.0-560.28.03-1_amd64.deb
    sudo dpkg -i cuda-repo-ubuntu2204-12-6-local_12.6.0-560.28.03-1_amd64.deb
    sudo cp /var/cuda-repo-ubuntu2204-12-6-local/cuda-*-keyring.gpg /usr/share/keyrings/
    sudo apt-get update
    sudo apt-get -y install cuda-toolkit-12-6

    echo "CUDA installed."
fi
