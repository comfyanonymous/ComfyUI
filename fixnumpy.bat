@echo off
:: Install numpy silently
venv\Scripts\pip install --force-reinstall "numpy<2" --quiet

:: Check if installation was successful
if %ERRORLEVEL% neq 0 (
    echo Error: Failed to install numpy
    exit /b 1
)

:: Display version information
venv\Scripts\python -c "import numpy; print(f'numpy {numpy.__version__} reinstalled.')"
pause