@echo off

:: If omitted, use the python installed in the system:
set PYTHON=

:: If omitted, use the default "venv" subfolder:
set VENV_DIR=

call comfyui_venv_and_launch.bat %*
