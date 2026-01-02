@echo off
cd /d C:\Users\spoko\www\ai\ComfyUI
echo Starting ComfyUI...
echo.
echo Server will be available at: http://127.0.0.1:8188
echo Press Ctrl+C to stop the server
echo.
venv\Scripts\python main.py
pause
