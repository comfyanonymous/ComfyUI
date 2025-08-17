@echo off
echo Installing Taichi lang backend...

if exist "%python_exec%" (
    %python_exec% -s -m pip install taichi
) else (
    echo Installing with system Python
    pip install taichi
)

pause