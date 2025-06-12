@echo off

set PYTHON="%~dp0/venv/Scripts/python.exe"
set GIT=
set VENV_DIR=./venv

set COMMANDLINE_ARGS=--auto-launch --use-quad-cross-attention --reserve-vram 0.9

set ZLUDA_COMGR_LOG_LEVEL=1

echo *** Checking and updating to new version if possible 

copy comfy\customzluda\zluda-default.py comfy\zluda.py /y >NUL
git pull
copy comfy\customzluda\zluda-default.py comfy\zluda.py /y >NUL

echo.
.\zluda\zluda.exe -- %PYTHON% main.py %COMMANDLINE_ARGS%
pause
