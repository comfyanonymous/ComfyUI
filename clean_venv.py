#!/usr/bin/env python3
"""
ComfyUI Virtual Environment Cleanup Script
Created: May 2, 2025
"""

import argparse
import os
import shutil
import sys
import platform
from pathlib import Path

def deactivate_venv():
    """Attempt to deactivate any active virtual environment."""
    if "VIRTUAL_ENV" in os.environ:
        print(f"Active virtual environment detected: {os.environ['VIRTUAL_ENV']}")
        print("Attempting to deactivate...")
        
        # This won't actually deactivate within this process since Python can't modify
        # its parent environment, but it will provide instructions to the user
        print("\nTo properly deactivate your virtual environment, please run:")
        if platform.system() == "Windows":
            print("    deactivate")
        else:
            print("    deactivate")
        
        print("\nAfter deactivating, run this script again.")
        return True
    return False

def remove_venv(venv_path, force=False):
    """Remove a virtual environment directory if it exists."""
    path = Path(venv_path).expanduser().resolve()
    
    if not path.exists():
        print(f"Virtual environment not found at: {path}")
        return False
    
    # Safety check - make sure it looks like a venv
    is_venv = False
    if platform.system() == "Windows":
        is_venv = (path / "Scripts" / "python.exe").exists()
    else:
        is_venv = (path / "bin" / "python").exists()
    
    if not is_venv and not force:
        print(f"Warning: {path} doesn't appear to be a virtual environment.")
        print("Use --force to remove it anyway.")
        return False
    
    try:
        print(f"Removing virtual environment at: {path}")
        shutil.rmtree(path)
        print("✅ Virtual environment removed successfully.")
        return True
    except Exception as e:
        print(f"❌ Error removing virtual environment: {e}")
        if platform.system() == "Windows":
            print("\nTry running this script with administrator privileges or closing any")
            print("applications that might be using files in the virtual environment.")
        else:
            print("\nTry running this script with sudo if you have permission issues:")
            print(f"    sudo rm -rf {path}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Remove ComfyUI virtual environments")
    parser.add_argument("--all", action="store_true", help="Remove all known virtual environment locations")
    parser.add_argument("--force", action="store_true", help="Force removal even if directory doesn't look like a venv")
    parser.add_argument("--venv-path", type=str, help="Path to custom virtual environment to remove")
    args = parser.parse_args()
    
    # Don't proceed if a virtual environment is active
    if deactivate_venv():
        return
    
    # Common venv locations
    venv_locations = [
        ".venv",                  # Default UV venv name
        "venv",                   # Common alternative name
        Path.home() / ".venvs" / "comfyui",  # User-level venvs
    ]
    
    if args.venv_path:
        # Just remove the specified venv
        remove_venv(args.venv_path, args.force)
    elif args.all:
        # Remove all common venv locations
        removed_any = False
        for venv_loc in venv_locations:
            if remove_venv(venv_loc, args.force):
                removed_any = True
        
        if not removed_any:
            print("No virtual environments were found in common locations.")
    else:
        # Default: just remove .venv
        if not remove_venv(".venv", args.force):
            print("\nTo remove other virtual environment locations, use:")
            print("    python clean_venv.py --all")
            print("    python clean_venv.py --venv-path /path/to/venv")
    
    print("\n🚀 Ready to create a new UV-based environment!")
    print("Run the following commands to set up with UV:")
    print("    ./uv_setup.sh")
    print("    source .venv/bin/activate  # On Linux/macOS")
    print("    ./update_dependencies_uv.py --advanced --cuda --lock")

if __name__ == "__main__":
    main()