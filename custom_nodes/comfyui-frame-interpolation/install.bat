@echo off

set "requirements_txt=%~dp0\requirements-no-cupy.txt"
set "python_exec=..\..\..\python_embeded\python.exe"

echo Installing ComfyUI Frame Interpolation..

if exist "%python_exec%" (
    echo Installing with ComfyUI Portable
    %python_exec% -s install.py
) else (
    echo Installing with system Python
    python install.py
)

pause