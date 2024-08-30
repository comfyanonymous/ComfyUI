@echo off

set PYTHON=%~dp0/venv/Scripts/python.exe
set GIT=
set VENV_DIR=./venv
set COMMANDLINE_ARGS=--auto-launch --fast --novram

echo *** Checking and updating to new version if possible 
git pull
echo.
echo *** Pytorch tunableOP enabled with --novram to avoid OOM as much as possible. 
echo *** (This is important for big models such as flux)
echo *** When you are done , please exit with CTRL-C on cmd window to make sure tunableop csv file properly written.
echo.
set PYTORCH_TUNABLEOP_ENABLED=1
set PYTORCH_TUNABLEOP_VERBOSE=1
set PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED=0
echo.

.\zluda\zluda.exe -- %PYTHON% main.py %COMMANDLINE_ARGS%
