@echo off
cd /d D:\AI\ComfyUI

REM Optymalizacje CUDA dla unikniecia bledow illegal memory access
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
set CUDA_LAUNCH_BLOCKING=0

echo Starting ComfyUI with CUDA optimizations...
echo.
echo Server will be available at: http://127.0.0.1:8188
echo Press Ctrl+C to stop the server
echo.
echo CUDA optimizations enabled:
echo - expandable_segments: True
echo - max_split_size_mb: 512
echo.
venv\Scripts\python main.py
pause
