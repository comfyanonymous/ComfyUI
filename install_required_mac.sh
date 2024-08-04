#!/bin/bash

# Check for Homebrew
if command -v brew >/dev/null 2>&1; then
    echo "Homebrew is already installed."
else
    echo "Installing Homebrew..."

    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

    echo "Homebrew has been installed."
fi

# Check for git
if command -v git >/dev/null 2>&1; then
    echo "Git is already installed."
else
    echo "Installing GIT..."
    
    brew install git

    echo "Git has been installed."
fi

# Check for Python3.8
if command -v python3 >/dev/null 2>&1; then
    echo "Python3.8 is already installed."
else
    echo "Installing Python3.8..."

    brew install pyenv
    pyenv install 3.8
    python3 -m ensurepip --upgrade

    echo "Python3.8 has been installed."
fi

# Check for CUDA
if command -v nvcc >/dev/null 2>&1; then
    echo "CUDA is already installed."
else
    echo "Installing CUDA..."

    CUDA_VERSION="11.8"
    CUDA_INSTALLER="cuda_${CUDA_VERSION}_mac.dmg"
    CUDA_URL="https://developer.download.nvidia.com/compute/cuda/${CUDA_VERSION}/local_installers/${CUDA_INSTALLER}"
    curl -LO ${CUDA_URL}
    hdiutil mount ${CUDA_INSTALLER}
    sudo installer -pkg /Volumes/CUDA\ ${CUDA_VERSION}/CUDA.pkg -target /
    hdiutil unmount /Volumes/CUDA\ ${CUDA_VERSION}
    rm ${CUDA_INSTALLER}

    echo "CUDA has been installed."
fi