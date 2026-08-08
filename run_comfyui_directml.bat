@echo off
REM ComfyUI launcher for AMD Radeon RX 6800 XT via DirectML (device 0).
REM The integrated Radeon Graphics iGPU is device 1 and MUST be avoided.
REM PYTHONPATH is cleared so the local .venv (not any other venv) wins on imports.
cd /d "%~dp0"
set PYTHONPATH=
call .venv\Scripts\activate.bat
python main.py --directml 0 --listen 127.0.0.1 --port 8188
pause
