@echo off
title ComfyUI - CPU Mode
echo ============================================
echo ComfyUI - CPU Mode (Stable)
echo ============================================
echo.
echo GPU: DISABLED (using CPU only)
echo Speed: SLOWER (5-10x slower than GPU)
echo Stability: HIGH (no Blackwell bugs)
echo.
echo Press Ctrl+C to stop ComfyUI
echo ============================================
echo.

REM Force CPU mode - multiple methods
set CUDA_VISIBLE_DEVICES=
set CUDA_DEVICE_ORDER=PCI_BUS_ID
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:0

REM Disable CUDA in PyTorch
set PYTORCH_ENABLE_MPS_FALLBACK=1

REM Navigate to ComfyUI
cd /d D:\AI\ComfyUI

REM Start ComfyUI forcing CPU
.\venv\Scripts\python.exe main.py --cpu --disable-cuda-malloc

pause
