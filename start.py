#!/usr/bin/env python3
"""
Simple startup script for ComfyUI authentication module
"""

import uvicorn
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from comfyui_auth import create_app

app = create_app()

if __name__ == "__main__":
    print("Starting ComfyUI Authentication Module...")
    print("API endpoints will be available at http://localhost:8188")
    print("Login page: http://localhost:8188/web/login.html")
    print("Register page: http://localhost:8188/web/register.html")
    print("\nPress Ctrl+C to stop the server")
    
    uvicorn.run(app, host="0.0.0.0", port=8188, log_level="info")