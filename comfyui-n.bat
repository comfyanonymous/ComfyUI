@echo off

SET MIOPEN_FIND_MODE=2
set PYTHON="%~dp0/venv/Scripts/python.exe"
set GIT=
set VENV_DIR=./venv

set COMMANDLINE_ARGS=--auto-launch --use-pytorch-cross-attention --reserve-vram 0.9

set ZLUDA_COMGR_LOG_LEVEL=1

echo *** Checking and updating to new version if possible 
git pull
echo.
.\zluda\zluda.exe -- %PYTHON% main.py %COMMANDLINE_ARGS%
pause
