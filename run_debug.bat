@echo off
echo ============================================
echo ComfyUI - CPU MODE
echo ============================================
echo.
echo Running on CPU (no GPU)
echo This will be SLOWER but STABLE
echo.

REM Hide GPU from PyTorch (force CPU mode)
set CUDA_VISIBLE_DEVICES=-1

REM Navigate to ComfyUI directory
cd /d D:\AI\ComfyUI

REM Run ComfyUI with CPU flag
echo Starting ComfyUI in CPU mode...
echo.
.\venv\Scripts\python.exe main.py --cpu

pause
