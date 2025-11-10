@echo off
cd /d "%~dp0"

REM Check Python availability
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not found in PATH.
    echo Please ensure Python is installed and added to your PATH.
    pause
    exit /b 1
)

REM Get Python environment information
python -c "import sys, os; venv = os.environ.get('VIRTUAL_ENV', ''); is_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix); env_type = 'VENV_DETECTED' if (venv or is_venv) else 'SYSTEM_PYTHON'; print(env_type); print('PYTHON_PATH=' + sys.executable)" > env_info.tmp
for /f "tokens=1,* delims==" %%a in (env_info.tmp) do (
    if "%%a"=="VENV_DETECTED" set ENV_TYPE=VENV_DETECTED
    if "%%a"=="SYSTEM_PYTHON" set ENV_TYPE=SYSTEM_PYTHON
    if "%%a"=="PYTHON_PATH" set PYTHON_PATH=%%b
)
del env_info.tmp

REM Check for missing critical dependencies
python -c "import importlib.util; missing = []; deps = {'yaml': 'yaml', 'torch': 'torch', 'torchvision': 'torchvision', 'torchaudio': 'torchaudio', 'numpy': 'numpy', 'einops': 'einops', 'transformers': 'transformers', 'tokenizers': 'tokenizers', 'sentencepiece': 'sentencepiece', 'safetensors': 'safetensors', 'aiohttp': 'aiohttp', 'yarl': 'yarl', 'PIL': 'Pillow', 'scipy': 'scipy', 'tqdm': 'tqdm', 'psutil': 'psutil', 'alembic': 'alembic', 'sqlalchemy': 'sqlalchemy', 'av': 'av', 'comfyui_frontend': 'comfyui_frontend', 'comfyui_workflow_templates': 'comfyui_workflow_templates', 'comfyui_embedded_docs': 'comfyui_embedded_docs'}; [missing.append(k) for k, v in deps.items() if not importlib.util.find_spec(v)]; print(','.join(missing) if missing else 'ALL_OK')" > deps_check.tmp
set /p MISSING_DEPS=<deps_check.tmp
del deps_check.tmp

if "%MISSING_DEPS%"=="ALL_OK" goto :start_comfyui

REM Dependencies are missing - show warnings and prompt user
echo.
echo ========================================
echo MISSING DEPENDENCIES DETECTED
echo ========================================
echo.
echo The following critical dependencies are missing:
echo %MISSING_DEPS%
echo.

REM Display environment warnings
if "%ENV_TYPE%"=="VENV_DETECTED" (
    echo [INFO] You are running in a virtual environment.
    echo [INFO] Installing packages here is safe and recommended.
    echo.
) else (
    echo [WARNING] You are using system Python or user site-packages.
    echo [WARNING] Installing packages here may affect other applications.
    echo [WARNING] Consider using a virtual environment for better isolation.
    echo.
)

echo Python executable: %PYTHON_PATH%
echo.
echo ========================================
echo INSTALLATION WARNING
echo ========================================
echo.
echo This will install packages using: %PYTHON_PATH%
echo.
echo Potential risks:
echo   - If using system Python, packages may conflict with other applications
echo   - If using user site-packages, packages are installed per-user
echo   - Virtual environments are recommended for isolation
echo.
echo If you are unsure, you can:
echo   1. Create a virtual environment: python -m venv venv
echo   2. Activate it: venv\Scripts\activate
echo   3. Then run this script again
echo.
set /p INSTALL_CHOICE="Do you want to install missing dependencies now? (Y/N): "

if /i not "%INSTALL_CHOICE%"=="Y" (
    echo.
    echo Installation cancelled.
    echo.
    echo To install dependencies manually, run:
    echo   python -m pip install -r requirements.txt
    echo.
    pause
    exit /b 0
)

echo.
echo Installing dependencies...
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo ERROR: Failed to install dependencies.
    echo Please check the error messages above and try installing manually:
    echo   python -m pip install -r requirements.txt
    pause
    exit /b 1
)

echo.
echo Dependencies installed successfully.
echo.

:start_comfyui
python main.py
pause

