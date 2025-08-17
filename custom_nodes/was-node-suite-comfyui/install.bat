@echo off

set "requirements_txt=%~dp0\requirements.txt"
set "python_exec=..\..\..\python_embeded\python.exe"

echo Installing WAS-NS ...

if exist "%python_exec%" (
    echo Installing with ComfyUI Portable
    "%python_exec%" -s -m pip install -r "%requirements_txt%"
) else (
    echo Installing with system Python
    pip install -r "%requirements_txt%"
)

pause