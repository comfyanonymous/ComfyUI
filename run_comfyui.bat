@echo off
chcp 65001 >nul 2>&1
cd /d "%~dp0"

REM Display ComfyUI 8-bit header
echo.
echo  ╔═══════════════════════════════════════════════════════════╗
echo  ║                                                           ║
echo  ║     ██████╗ ██████╗ ███╗   ███╗███████╗██╗   ██╗██╗      ║
echo  ║    ██╔════╝██╔═══██╗████╗ ████║██╔════╝╚██╗ ██╔╝██║      ║
echo  ║    ██║     ██║   ██║██╔████╔██║█████╗   ╚████╔╝ ██║      ║
echo  ║    ██║     ██║   ██║██║╚██╔╝██║██╔══╝    ╚██╔╝  ██║      ║
echo  ║    ╚██████╗╚██████╔╝██║ ╚═╝ ██║██║        ██║   ███████╗ ║
echo  ║     ╚═════╝ ╚═════╝ ╚═╝     ╚═╝╚═╝        ╚═╝   ╚══════╝ ║
echo  ║                                                           ║
echo  ║         The most powerful open source node-based          ║
echo  ║         application for generative AI                    ║
echo  ║                                                           ║
echo  ╚═══════════════════════════════════════════════════════════╝
echo.

REM Check Python availability
python --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo  ╔═══════════════════════════════════════════════════════════╗
    echo  ║                    Python Not Found                       ║
    echo  ╚═══════════════════════════════════════════════════════════╝
    echo.
    echo  ▓ ComfyUI needs Python to run, but we couldn't find it on your computer.
    echo.
    echo  ▓ What to do:
    echo    1. Download Python from: https://www.python.org/downloads/
    echo    2. During installation, make sure to check "Add Python to PATH"
    echo    3. Restart your computer after installing
    echo    4. Try running this script again
    echo.
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

REM Check for missing dependencies - separate critical vs optional
python -c "import importlib.util; critical = []; optional = []; critical_deps = {'yaml': 'yaml', 'torch': 'torch', 'torchvision': 'torchvision', 'torchaudio': 'torchaudio', 'numpy': 'numpy', 'einops': 'einops', 'transformers': 'transformers', 'tokenizers': 'tokenizers', 'sentencepiece': 'sentencepiece', 'safetensors': 'safetensors', 'aiohttp': 'aiohttp', 'yarl': 'yarl', 'PIL': 'PIL', 'scipy': 'scipy', 'tqdm': 'tqdm', 'psutil': 'psutil', 'alembic': 'alembic', 'sqlalchemy': 'sqlalchemy', 'av': 'av', 'comfyui_frontend': 'comfyui_frontend_package'}; optional_deps = {'comfyui_workflow_templates': 'comfyui_workflow_templates', 'comfyui_embedded_docs': 'comfyui_embedded_docs'}; [critical.append(k) for k, v in critical_deps.items() if not importlib.util.find_spec(v)]; [optional.append(k) for k, v in optional_deps.items() if not importlib.util.find_spec(v)]; print('CRITICAL:' + (','.join(critical) if critical else 'NONE')); print('OPTIONAL:' + (','.join(optional) if optional else 'NONE'))" > deps_check.tmp
for /f "tokens=1,* delims=:" %%a in (deps_check.tmp) do (
    if "%%a"=="CRITICAL" set MISSING_CRITICAL=%%b
    if "%%a"=="OPTIONAL" set MISSING_OPTIONAL=%%b
)
del deps_check.tmp

REM Check if we can launch without optional dependencies
if "%MISSING_CRITICAL%"=="NONE" (
    if not "%MISSING_OPTIONAL%"=="NONE" (
        echo.
        echo  ╔═══════════════════════════════════════════════════════════╗
        echo  ║              Optional Packages Available                   ║
        echo  ╚═══════════════════════════════════════════════════════════╝
        echo.
        echo  ▓ The following optional packages are missing:
        echo    %MISSING_OPTIONAL%
        echo.
        echo  ▓ These packages add extra features but aren't required to run ComfyUI.
        echo    ComfyUI will launch without them, but some features may be unavailable.
        echo.
        set /p INSTALL_OPTIONAL="Would you like to install optional packages? (Y/N/S=Skip for now): "
        if /i "%INSTALL_OPTIONAL%"=="Y" (
            echo.
            echo  ▓ Installing optional packages...
            python -m pip install comfyui-workflow-templates comfyui-embedded-docs >nul 2>&1
            echo  ▓ Optional packages installed.
            echo.
        ) else if /i "%INSTALL_OPTIONAL%"=="S" (
            echo.
            echo  ▓ Skipping optional packages. ComfyUI will launch with limited features.
            echo.
        ) else (
            echo.
            echo  ▓ Skipping optional packages.
            echo.
        )
        goto :check_pytorch
    )
    goto :check_pytorch
)

REM Critical dependencies are missing
if not "%MISSING_CRITICAL%"=="NONE" (
    echo.
    echo  ╔═══════════════════════════════════════════════════════════╗
    echo  ║            Missing Required Packages                      ║
    echo  ╚═══════════════════════════════════════════════════════════╝
    echo.
    echo  ▓ ComfyUI needs some additional software to run.
    echo    The following critical packages are missing:
    echo    %MISSING_CRITICAL%
    echo.
    if not "%MISSING_OPTIONAL%"=="NONE" (
        echo  ▓ Optional packages also missing: %MISSING_OPTIONAL%
        echo.
    )
    echo  ▓ These are like plugins that ComfyUI needs to work properly.
    echo.

    REM Display environment warnings
    if "%ENV_TYPE%"=="VENV_DETECTED" (
        echo  ▓ [Good News] You're using a virtual environment.
        echo    This means installing packages here won't affect other programs.
        echo.
    ) else (
        echo  ▓ [Heads Up] You're using your main Python installation.
        echo    Installing packages here might affect other programs that use Python.
        echo.
        echo  ▓ Tip: For better safety, you can create a separate environment:
        echo     1. Create it: python -m venv venv
        echo     2. Activate it: venv\Scripts\activate
        echo     3. Run this script again
        echo.
    )

    echo  ▓ We'll install packages using: %PYTHON_PATH%
    echo.
    echo  ╔═══════════════════════════════════════════════════════════╗
    echo  ║                    Installation Options                   ║
    echo  ╚═══════════════════════════════════════════════════════════╝
    echo.
    echo    [I] Install all missing packages (recommended)
    echo    [C] Install only critical packages
    echo    [N] Cancel and exit
    echo.
    set /p INSTALL_CHOICE="Choose an option (I/C/N): "

    if /i "%INSTALL_CHOICE%"=="I" (
        echo.
        echo  ▓ Installing all required packages...
        echo    This may take several minutes. Please wait...
        echo.
        python -m pip install --progress-bar on -r requirements.txt
        if errorlevel 1 (
            echo.
            echo  ╔═══════════════════════════════════════════════════════════╗
            echo  ║                  Installation Failed                       ║
            echo  ╚═══════════════════════════════════════════════════════════╝
            echo.
            echo  ▓ Something went wrong while installing the packages.
            echo.
            echo  ▓ Common problems and fixes:
            echo    - Internet connection issues: Check your internet and try again
            echo    - Permission errors: Try right-clicking and "Run as Administrator"
            echo    - Package conflicts: Try creating a virtual environment (see above)
            echo.
            echo  ▓ To try installing manually, open a terminal here and run:
            echo    python -m pip install -r requirements.txt
            echo.
            pause
            exit /b 1
        )
        echo.
        echo  ▓ Great! All packages installed successfully.
        echo.
    ) else if /i "%INSTALL_CHOICE%"=="C" (
        echo.
        echo  ▓ Installing critical packages only...
        echo.
        python -m pip install --progress-bar on torch torchvision torchaudio numpy einops transformers tokenizers sentencepiece safetensors aiohttp yarl pyyaml Pillow scipy tqdm psutil alembic SQLAlchemy av comfyui-frontend-package
        if errorlevel 1 (
            echo.
            echo  ╔═══════════════════════════════════════════════════════════╗
            echo  ║                  Installation Failed                       ║
            echo  ╚═══════════════════════════════════════════════════════════╝
            echo.
            echo  ▓ Something went wrong while installing the packages.
            echo    Please check the error messages above.
            echo.
            pause
            exit /b 1
        )
        echo.
        echo  ▓ Critical packages installed. ComfyUI should now launch.
        echo.
    ) else (
        echo.
        echo  ▓ Installation cancelled.
        echo.
        echo  ▓ If you want to install them later, open a terminal here and run:
        echo    python -m pip install -r requirements.txt
        echo.
        pause
        exit /b 0
    )
)

:check_pytorch
REM Check if PyTorch has CUDA support (for NVIDIA GPUs)
python -c "import torch; cuda_available = torch.cuda.is_available(); cuda_version = torch.version.cuda if cuda_available else None; pytorch_version = torch.__version__; print('CUDA_AVAILABLE:' + str(cuda_available)); print('CUDA_VERSION:' + (cuda_version if cuda_version else 'NONE')); print('PYTORCH_VERSION:' + pytorch_version)" > pytorch_check.tmp 2>&1
if errorlevel 1 (
    echo.
    echo  ╔═══════════════════════════════════════════════════════════╗
    echo  ║              Could Not Check GPU Support                 ║
    echo  ╚═══════════════════════════════════════════════════════════╝
    echo.
    echo  ▓ We couldn't check if your GPU will work with ComfyUI.
    echo    ComfyUI will try to start anyway, but it might run slowly on your CPU.
    echo.
    goto :start_comfyui
)

for /f "tokens=1,* delims=:" %%a in (pytorch_check.tmp) do (
    if "%%a"=="CUDA_AVAILABLE" set CUDA_AVAILABLE=%%b
    if "%%a"=="CUDA_VERSION" set CUDA_VERSION=%%b
    if "%%a"=="PYTORCH_VERSION" set PYTORCH_VERSION=%%b
)
del pytorch_check.tmp

REM Check if PyTorch version contains "+cpu" indicating CPU-only build
echo %PYTORCH_VERSION% | findstr /C:"+cpu" >nul
if not errorlevel 1 (
    echo.
    echo  ╔═══════════════════════════════════════════════════════════╗
    echo  ║     CPU-Only PyTorch Detected - CUDA Version Required     ║
    echo  ╚═══════════════════════════════════════════════════════════╝
    echo.
    echo  ▓ Your PyTorch installation doesn't support GPU acceleration.
    echo    ComfyUI requires CUDA-enabled PyTorch to run properly.
    echo.
    echo  ▓ We can automatically install the CUDA-enabled version for you.
    echo    This will:
    echo      1. Remove the current CPU-only version
    echo      2. Install the CUDA-enabled version (this will take several minutes)
    echo      3. Continue to launch ComfyUI automatically
    echo.
    echo  ▓ Note: This requires an NVIDIA graphics card with CUDA support.
    echo.
    set /p INSTALL_CUDA="Would you like to install CUDA-enabled PyTorch now? (Y/N): "
    if /i "%INSTALL_CUDA%"=="Y" (
        echo.
        echo  ▓ Uninstalling CPU-only PyTorch...
        python -m pip uninstall -y torch torchvision torchaudio
        if errorlevel 1 (
            echo.
            echo  ╔═══════════════════════════════════════════════════════════╗
            echo  ║              Uninstallation Failed                       ║
            echo  ╚═══════════════════════════════════════════════════════════╝
            echo.
            echo  ▓ Failed to uninstall CPU-only PyTorch.
            echo    Please try running as Administrator or uninstall manually.
            echo.
            pause
            exit /b 1
        )
        echo.
        echo  ▓ Installing CUDA-enabled PyTorch...
        echo    This may take several minutes. Please wait...
        echo.
        python -m pip install --progress-bar on torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu130
        if errorlevel 1 (
            echo.
            echo  ╔═══════════════════════════════════════════════════════════╗
            echo  ║              Installation Failed                           ║
            echo  ╚═══════════════════════════════════════════════════════════╝
            echo.
            echo  ▓ Failed to install CUDA-enabled PyTorch.
            echo    Please check your internet connection and try again.
            echo.
            echo  ▓ To install manually, run:
            echo    python -m pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu130
            echo.
            pause
            exit /b 1
        )
        echo.
        echo  ▓ CUDA-enabled PyTorch installed successfully!
        echo    Verifying installation...
        echo.
        REM Verify the installation
        python -c "import torch; print('CUDA_AVAILABLE:' + str(torch.cuda.is_available())); print('PYTORCH_VERSION:' + torch.__version__)" > pytorch_verify.tmp 2>&1
        if errorlevel 1 (
            echo  ▓ Warning: Could not verify PyTorch installation.
            echo    Continuing anyway...
            echo.
            REM Continue to launch ComfyUI even if verification failed
            goto :start_comfyui
        ) else (
            for /f "tokens=1,* delims=:" %%a in (pytorch_verify.tmp) do (
                if "%%a"=="CUDA_AVAILABLE" set CUDA_VERIFY=%%b
                if "%%a"=="PYTORCH_VERSION" set PYTORCH_VERIFY=%%b
            )
            del pytorch_verify.tmp
            
            REM Update CUDA_AVAILABLE and PYTORCH_VERSION with the new values
            set CUDA_AVAILABLE=%CUDA_VERIFY%
            set PYTORCH_VERSION=%PYTORCH_VERIFY%
            
            echo %PYTORCH_VERIFY% | findstr /C:"+cpu" >nul
            if not errorlevel 1 (
                echo  ▓ Warning: PyTorch still appears to be CPU-only.
                echo    The installation may have failed. Please check manually.
                echo.
                REM Still continue - let ComfyUI try to run
                goto :start_comfyui
            ) else (
                echo  ▓ Verification successful! CUDA-enabled PyTorch is ready.
                echo.
                REM Continue to launch ComfyUI
                goto :start_comfyui
            )
        )
        REM If verification failed but installation succeeded, continue anyway
        goto :start_comfyui
    ) else (
        echo.
        echo  ▓ Skipping CUDA PyTorch installation.
        echo    ComfyUI will not be able to run with CPU-only PyTorch.
        echo    Please install CUDA-enabled PyTorch manually and try again.
        echo.
        pause
        exit /b 0
    )
)

REM Check if CUDA is not available but PyTorch doesn't have "+cpu" (might be CUDA build but no GPU)
if "%CUDA_AVAILABLE%"=="False" (
    echo %PYTORCH_VERSION% | findstr /C:"+cpu" >nul
    if errorlevel 1 (
        echo.
        echo  ╔═══════════════════════════════════════════════════════════╗
        echo  ║                    GPU Not Detected                       ║
        echo  ╚═══════════════════════════════════════════════════════════╝
        echo.
        echo  ▓ PyTorch has GPU support installed, but we couldn't find your graphics card.
        echo.
        echo  ▓ This could mean:
        echo    - You don't have an NVIDIA graphics card
        echo    - Your graphics card drivers need to be updated
        echo    - Your graphics card isn't properly connected
        echo.
        echo  ▓ ComfyUI will run on your CPU instead, which will be slower.
        echo.
        set /p CONTINUE_CHOICE="Continue anyway? (Y/N): "
        if /i not "%CONTINUE_CHOICE%"=="Y" (
            echo.
            echo  ▓ Exiting. Check your graphics card setup and try again.
            pause
            exit /b 0
        )
    )
)

:start_comfyui
echo.
echo  ╔═══════════════════════════════════════════════════════════╗
echo  ║                  Starting ComfyUI...                      ║
echo  ╚═══════════════════════════════════════════════════════════╝
echo.
python main.py
if errorlevel 1 (
    echo.
    echo  ╔═══════════════════════════════════════════════════════════╗
    echo  ║                    ComfyUI Crashed                        ║
    echo  ╚═══════════════════════════════════════════════════════════╝
    echo.
    echo  ▓ ComfyUI encountered an error and stopped. Here's what might help:
    echo.
    echo  ▓ Error: "Torch not compiled with CUDA enabled"
    echo    Solution: You need to install the GPU version of PyTorch (see instructions above)
    echo.
    echo  ▓ Error: "ModuleNotFoundError" or "No module named"
    echo    Solution: Run this script again to install missing packages
    echo.
    echo  ▓ Error: "CUDA out of memory" or "OOM"
    echo    Solution: Your graphics card doesn't have enough memory. Try using smaller models.
    echo.
    echo  ▓ For other errors, check the error message above for clues.
    echo    You can also visit: https://github.com/comfyanonymous/ComfyUI/issues
    echo.
    echo  ▓ The full error details are shown above.
    echo.
)
pause



