@echo off

:: Set default port and read from .env if exists
set "port=8188"
for /f "tokens=2 delims==" %%i in ('findstr /r "^PORT=" .env') do set "port=%%i"

:: Set default venv path and read from .env if exists
set "venv_path="
for /f "tokens=2 delims==" %%i in ('findstr /r "^VENV_PATH=" .env') do set "venv_path=%%i"

:: Activate the virtual environment if specified
if not "%venv_path%"=="" (
    call "%venv_path%\Scripts\activate"
    echo Activated virtual environment: %venv_path%
) else (
    echo No virtual environment specified. Using system Python.
)

:: Pull updates and handle local changes
:pull
set "cmdOutput=cmd_output.txt"
git pull > "%cmdOutput%" 2>&1

:: Handle potential conflicts or changes
findstr /C:"error: Your local changes to the following files would be overwritten by merge:" "%cmdOutput%" > nul && (
    echo Pull conflicts detected. Stashing changes...
    git stash
    git pull
)

findstr /C:"Already up to date." "%cmdOutput%" > nul || goto rebuild

del "%cmdOutput%"

echo No changes detected. Starting ComfyUI
pip install -r requirements.txt

echo Opening localhost:%port% in your default browser...
start http://localhost:%port%

python main.py
