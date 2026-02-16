@echo off
cd /d D:\AI\ComfyUI

REM Optymalizacje CUDA i lowvram mode dla systemow z ograniczona pamiecia GPU
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256
set CUDA_LAUNCH_BLOCKING=0

echo Starting ComfyUI with LOWVRAM mode...
echo.
echo Server will be available at: http://127.0.0.1:8188
echo Press Ctrl+C to stop the server
echo.
echo CUDA optimizations enabled:
echo - expandable_segments: True
echo - max_split_size_mb: 256
echo - lowvram mode enabled
echo.
venv\Scripts\python main.py --lowvram
pause
