@echo off

set PYTHON=%~dp0/venv/Scripts/python.exe
set GIT=
set VENV_DIR=./venv
set COMMANDLINE_ARGS=--auto-launch

echo *** Checking and updating to new version if possible 
git pull
echo.
.\zluda\zluda.exe -- %PYTHON% main.py %COMMANDLINE_ARGS%
