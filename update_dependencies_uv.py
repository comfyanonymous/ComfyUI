#!/usr/bin/env python3
"""
Update dependencies for ComfyUI using UV
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def check_uv_installed():
    """Check if UV is installed and install it if not."""
    try:
        subprocess.run(["uv", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        print("UV not found. Installing UV...")
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
            
            print("UV installed successfully.")
            return True
        except subprocess.SubprocessError:
            print("Failed to install UV. Please install manually from https://github.com/astral-sh/uv")
            return False

def update_dependencies(advanced=False, cuda=False, lock=False):
    """Update dependencies using UV."""
    if not check_uv_installed():
        return False
    
    print("Updating dependencies...")
    
    # Base dependencies
    try:
        cmd = ["uv", "pip", "install", "--upgrade", "-r", "requirements.txt"]
        if lock:
            cmd.append("--lock")
        
        subprocess.run(cmd, check=True)
        print("✓ Base dependencies updated successfully.")
    except subprocess.SubprocessError:
        print("Failed to update base dependencies.")
        return False
    
    # Advanced dependencies if requested
    if advanced:
        try:
            if os.path.exists("requirements_advanced.txt"):
                cmd = ["uv", "pip", "install", "--upgrade", "-r", "requirements_advanced.txt"]
                if lock:
                    cmd.append("--lock")
                
                subprocess.run(cmd, check=True)
                print("✓ Advanced dependencies updated successfully.")
            else:
                print("Advanced requirements file not found.")
        except subprocess.SubprocessError:
            print("Failed to update advanced dependencies.")
            return False
    
    # CUDA dependencies if requested
    if cuda:
        print("Updating CUDA dependencies...")
        extras = ["gpu"]
        if advanced:
            extras.append("advanced")
        
        try:
            cmd = ["uv", "pip", "install", "--upgrade", f".{{{','.join(extras)}}}"]
            if lock:
                cmd.append("--lock")
            
            subprocess.run(cmd, check=True)
            print("✓ CUDA dependencies updated successfully.")
        except subprocess.SubprocessError:
            print("Failed to update CUDA dependencies.")
            return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Update ComfyUI dependencies with UV")
    parser.add_argument("--advanced", action="store_true", help="Update advanced dependencies")
    parser.add_argument("--cuda", action="store_true", help="Update CUDA dependencies")
    parser.add_argument("--lock", action="store_true", help="Generate lockfile during update")
    
    args = parser.parse_args()
    
    if update_dependencies(args.advanced, args.cuda, args.lock):
        print("\n✓ All dependencies updated successfully!")
    else:
        print("\n❌ Failed to update some dependencies.")
        sys.exit(1)

if __name__ == "__main__":
    main()