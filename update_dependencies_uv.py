#!/usr/bin/env python3
"""
ComfyUI Dependency Update Script using UV
Created: May 2, 2025
"""

import argparse
import os
import platform
import subprocess
import sys

def check_uv_installed():
    """Check if UV is installed and install if missing."""
    try:
        subprocess.run(["uv", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        print("✅ UV is installed")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠️ UV not found. Installing...")
        
        if platform.system() == "Windows":
            subprocess.run(
                ["powershell", "-Command", "Invoke-WebRequest -Uri https://astral.sh/uv/install.ps1 -OutFile install.ps1; ./install.ps1"],
                check=True
            )
        else:
            subprocess.run(
                ["bash", "-c", "curl -LsSf https://astral.sh/uv/install.sh | sh"],
                check=True
            )
        
        print("✅ UV installed successfully")
        return True

def update_dependencies(args):
    """Update dependencies using UV."""
    # Ensure we're in a virtual environment
    if not os.environ.get("VIRTUAL_ENV") and not args.no_venv_check:
        print("⚠️ Not running in a virtual environment. Activate one before updating dependencies.")
        print("💡 Run: uv venv && source .venv/bin/activate (or .venv\\Scripts\\activate on Windows)")
        if not args.force:
            return False
    
    print("📦 Updating dependencies using UV...")
    
    # Update base requirements
    subprocess.run(["uv", "pip", "install", "--upgrade", "-r", "requirements.txt"], check=True)
    print("✅ Base dependencies updated")
    
    # Update advanced requirements if requested
    if args.advanced:
        subprocess.run(["uv", "pip", "install", "--upgrade", "-r", "requirements_advanced.txt"], check=True)
        print("✅ Advanced dependencies updated")
    
    # Update PyTorch with CUDA if requested
    if args.cuda:
        torch_version = "2.4.0" if args.latest else "2.3.0"
        cuda_version = "12.1" if args.latest else "11.8"
        
        print(f"🔄 Installing PyTorch {torch_version} with CUDA {cuda_version}...")
        subprocess.run([
            "uv", "pip", "install", "--upgrade", 
            f"torch=={torch_version}", 
            f"torchvision>={torch_version}", 
            f"torchaudio>={torch_version}", 
            "--index-url", f"https://download.pytorch.org/whl/cu{cuda_version.replace('.', '')}"
        ], check=True)
        print("✅ PyTorch updated with CUDA support")
    
    # Generate lockfile for reproducibility
    if args.lock:
        print("🔒 Generating lockfile...")
        subprocess.run(["uv", "pip", "freeze", ">", "UV-req-lock.txt"], shell=True, check=True)
        print("✅ Lockfile generated: UV-req-lock.txt")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Update ComfyUI dependencies using UV")
    parser.add_argument("--advanced", action="store_true", help="Update advanced dependencies")
    parser.add_argument("--cuda", action="store_true", help="Update PyTorch with CUDA support")
    parser.add_argument("--latest", action="store_true", help="Use latest versions (may be less stable)")
    parser.add_argument("--lock", action="store_true", help="Generate lockfile after update")
    parser.add_argument("--force", action="store_true", help="Force update even if not in venv")
    parser.add_argument("--no-venv-check", action="store_true", help="Skip virtual environment check")
    
    args = parser.parse_args()
    
    print("ComfyUI Dependency Update")
    print("========================")
    
    if not check_uv_installed():
        sys.exit(1)
    
    if update_dependencies(args):
        print("\n✨ Dependencies updated successfully!")
        print("\n💡 To start ComfyUI, run: python main.py")
    else:
        print("\n❌ Dependency update failed or was skipped")
        sys.exit(1)

if __name__ == "__main__":
    main()