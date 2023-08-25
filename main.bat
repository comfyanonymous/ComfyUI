@echo off

:: Deactivate the virtual environment
call .\venv\Scripts\deactivate.bat

:: Activate the virtual environment
call .\venv\Scripts\activate.bat
set PATH=%PATH%;%~dp0venv\Lib\site-packages\torch\lib

:: If the exit code is 0, run the kohya_gui.py script with the command-line arguments
if %errorlevel% equ 0 (
    REM Check if the batch was started via double-click
    IF /i "%comspec% /c %~0 " equ "%cmdcmdline:"=%" (
        REM echo This script was started by double clicking.
        cmd /k python.exe main.py --auto-launch %*
    ) ELSE (
        REM echo This script was started from a command prompt.
        python.exe main.py --auto-launch %*
    )
)