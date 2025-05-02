#!/usr/bin/env python3
"""
ComfyUI Installation Script using UV
"""

import os
import platform
import subprocess
import sys
import shutil
from pathlib import Path
import argparse
from datetime import datetime

def check_python_version():
    """Check if Python version is supported."""
    if sys.version_info < (3, 8):
        print("Python 3.8 or higher is required.")
        sys.exit(1)

def install_uv():
    """Install UV if not already installed."""
    if shutil.which("uv") is None:
        print("Installing UV package manager...")
        try:
            if platform.system() == "Windows":
                # For Windows, download and run the installer
                subprocess.check_call(
                    ["powershell", "-Command", 
                     "Invoke-WebRequest -Uri https://astral.sh/uv/install.ps1 -OutFile install.ps1; ./install.ps1"],
                    stdout=sys.stdout, stderr=sys.stderr
                )
            else:
                # For macOS and Linux
                subprocess.check_call(
                    ["bash", "-c", "curl -LsSf https://astral.sh/uv/install.sh | sh"],
                    stdout=sys.stdout, stderr=sys.stderr
                )
            print("UV installed successfully.")
        except subprocess.CalledProcessError:
            print("Failed to install UV. Please install it manually from https://github.com/astral-sh/uv")
            sys.exit(1)
    else:
        print("UV is already installed.")

def create_venv():
    """Create a virtual environment using UV."""
    print("Creating virtual environment...")
    try:
        subprocess.check_call(
            ["uv", "venv", ".venv"],
            stdout=sys.stdout, stderr=sys.stderr
        )
        print("Virtual environment created successfully.")
    except subprocess.CalledProcessError:
        print("Failed to create virtual environment.")
        sys.exit(1)

def activate_venv():
    """Return the activation command for the virtual environment."""
    if platform.system() == "Windows":
        return str(Path(".venv/Scripts/activate"))
    return f"source {str(Path('.venv/bin/activate'))}"

def install_requirements(args):
    """Install ComfyUI requirements using UV."""
    print("Installing requirements...")
    
    # Basic requirements
    try:
        subprocess.check_call(
            ["uv", "pip", "install", "-r", "requirements.txt"],
            stdout=sys.stdout, stderr=sys.stderr
        )
        print("Basic requirements installed successfully.")
    except subprocess.CalledProcessError:
        print("Failed to install basic requirements.")
        sys.exit(1)
    
    # Advanced requirements if specified
    if args.advanced:
        try:
            if os.path.exists("requirements_advanced.txt"):
                subprocess.check_call(
                    ["uv", "pip", "install", "-r", "requirements_advanced.txt"],
                    stdout=sys.stdout, stderr=sys.stderr
                )
                print("Advanced requirements installed successfully.")
            else:
                print("Advanced requirements file not found.")
        except subprocess.CalledProcessError:
            print("Failed to install advanced requirements.")
            sys.exit(1)
    
    # GPU requirements if specified
    if args.gpu:
        print("Installing PyTorch with CUDA support...")
        try:
            subprocess.check_call(
                ["uv", "pip", "install", "torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cu118"],
                stdout=sys.stdout, stderr=sys.stderr
            )
            print("PyTorch with CUDA installed successfully.")
        except subprocess.CalledProcessError:
            print("Failed to install PyTorch with CUDA.")
            sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Install ComfyUI with UV package manager")
    parser.add_argument("--advanced", action="store_true", help="Install advanced requirements")
    parser.add_argument("--gpu", action="store_true", help="Install PyTorch with CUDA support")
    parser.add_argument("--no-venv", action="store_true", help="Skip virtual environment creation")
    args = parser.parse_args()
    
    print(f"ComfyUI installation with UV started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    check_python_version()
    install_uv()
    
    if not args.no_venv:
        create_venv()
        print(f"To activate the virtual environment, run: {activate_venv()}")
    
    install_requirements(args)
    
    print("\nComfyUI installation completed successfully!")
    if not args.no_venv:
        print(f"Remember to activate the virtual environment with: {activate_venv()}")
    print("To start ComfyUI, run: python main.py")

if __name__ == "__main__":
    main()
