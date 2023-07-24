@echo off

:: If omitted, use the python installed in the system.
:: It's used only once, to create a venv (or if you explicitly tell the script to use it instead, see below).
set PYTHON=

:: If omitted, use the default "venv" subfolder.
:: Set to just a dash (set VENV_DIR=-) to make the launcher use a system python.
:: However, if you do, you might get into issues due to lack of write permissions.
:: (so set it to "-" only if you actually know what you do and why)
set VENV_DIR=

call comfyui_venv_and_launch.bat %*
