#!/usr/bin/env python3
"""
UV Setup Script for ComfyUI
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def check_python_version():
    """Check Python version compatibility."""
    if sys.version_info < (3, 9):
        print("ERROR: Python 3.9 or higher is required for ComfyUI.")
        sys.exit(1)

def is_uv_installed():
    """Check if UV is installed."""
    try:
        subprocess.run(["uv", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False

def install_uv():
    """Install UV package manager."""
    print("Installing UV package manager...")
    try:
        if sys.platform == "win32":
            subprocess.run(
                ["powershell", "-Command", "Invoke-WebRequest -Uri https://astral.sh/uv/install.ps1 -OutFile install.ps1; ./install.ps1"],
                check=True
            )
        else:  # macOS or Linux
            subprocess.run(
                ["bash", "-c", "curl -LsSf https://astral.sh/uv/install.sh | sh"],
                check=True
            )
        
        # Update PATH for this session
        if sys.platform != "win32":
            os.environ["PATH"] = f"{os.path.expanduser('~/.local/bin')}:{os.environ['PATH']}"
        
        print("✓ UV installed successfully.")
        return True
    except subprocess.SubprocessError:
        print("ERROR: UV installation failed.")
        return False

def create_venv(venv_path=".venv"):
    """Create a new virtual environment using UV."""
    print(f"Creating virtual environment at {venv_path}...")
    try:
        subprocess.run(["uv", "venv", venv_path], check=True)
        print(f"✓ Virtual environment created at {venv_path}.")
        return True
    except subprocess.SubprocessError:
        print("ERROR: Failed to create virtual environment.")
        return False

def install_comfyui(dev_mode=False, gpu=False, advanced=False, lock=False):
    """Install ComfyUI using UV."""
    print("Installing ComfyUI...")
    
    cmd = ["uv", "pip", "install"]
    
    if dev_mode:
        cmd.append("-e")
    
    # Build extras string
    extras = []
    if gpu:
        extras.append("gpu")
    if advanced:
        extras.append("advanced")
    if dev_mode:
        extras.append("dev")
    
    if extras:
        cmd.append(f".{{{','.join(extras)}}}")
    else:
        cmd.append(".")
    
    if lock:
        cmd.append("--lock")
    
    try:
        subprocess.run(cmd, check=True)
        print("✓ ComfyUI installed successfully.")
        return True
    except subprocess.SubprocessError:
        print("ERROR: Failed to install ComfyUI.")
        return False

def print_activation_instructions(venv_path=".venv"):
    """Print instructions for activating the virtual environment."""
    print("\nTo activate the virtual environment:")
    if sys.platform == "win32":
        print(f"    {venv_path}\\Scripts\\activate")
    else:
        print(f"    source {venv_path}/bin/activate")

def print_run_instructions():
    """Print instructions for running ComfyUI."""
    print("\nTo run ComfyUI:")
    print("    comfyui")
    print("\nOr with options:")
    print("    comfyui --host 0.0.0.0 --port 8188 --auto-launch")

def main():
    parser = argparse.ArgumentParser(description="Set up and install ComfyUI with UV")
    parser.add_argument("--no-venv", action="store_true", help="Skip creating a virtual environment")
    parser.add_argument("--venv-path", default=".venv", help="Path for the virtual environment (default: .venv)")
    parser.add_argument("--dev", action="store_true", help="Install in development mode")
    parser.add_argument("--gpu", action="store_true", help="Install with GPU/CUDA support")
    parser.add_argument("--advanced", action="store_true", help="Install with advanced dependencies")
    parser.add_argument("--lock", action="store_true", help="Generate lockfile during installation")
    
    args = parser.parse_args()
    
    # Checks
    check_python_version()
    
    # Install UV if needed
    if not is_uv_installed():
        if not install_uv():
            sys.exit(1)
    
    # Create virtual environment if requested
    venv_created = False
    if not args.no_venv:
        venv_created = create_venv(args.venv_path)
    
    # Install ComfyUI
    install_success = install_comfyui(
        dev_mode=args.dev,
        gpu=args.gpu,
        advanced=args.advanced,
        lock=args.lock
    )
    
    if install_success:
        print("\n✓ ComfyUI setup completed successfully!")
        
        if venv_created:
            print_activation_instructions(args.venv_path)
        
        print_run_instructions()

if __name__ == "__main__":
    main()