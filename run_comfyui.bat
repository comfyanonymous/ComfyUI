@echo off
chcp 65001 >nul 2>&1
cd /d "%~dp0"

REM Display ComfyUI 8-bit header
echo.
echo  ╔═══════════════════════════════════════════════════════════╗
echo  ║                                                           ║
echo  ║          ██████╗ ██████╗ ███╗   ███╗███████╗██╗   ██╗     ║
echo  ║         ██╔════╝██╔═══██╗████╗ ████║██╔════╝╚██╗ ██╔╝     ║
echo  ║         ██║     ██║   ██║██╔████╔██║█████╗   ╚████╔╝      ║
echo  ║         ██║     ██║   ██║██║╚██╔╝██║██╔══╝    ╚██╔╝       ║
echo  ║         ╚██████╗╚██████╔╝██║ ╚═╝ ██║██║        ██║        ║
echo  ║          ╚═════╝ ╚═════╝ ╚═╝     ╚═╝╚═╝        ╚═╝        ║
echo  ║                                                           ║
echo  ║         The most powerful open source node-based          ║
echo  ║         application for generative AI                     ║
echo  ║                                                           ║
echo  ╚═══════════════════════════════════════════════════════════╝
echo.

echo  ╔═══════════════════════════════════════════════════════════╗
echo  ║                    Preflight Check                        ║
echo  ╚═══════════════════════════════════════════════════════════╝
echo.
echo  ▓ Taking a quick look around your rig... checking prereqs.
echo    This will only take a moment.
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

REM ---------------------------------------------------------------
REM Weekly full check logic (skip optional prompts for faster launch)
REM Force with: run_comfyui.bat --full-check
REM ---------------------------------------------------------------
set STATE_DIR=%LOCALAPPDATA%\ComfyUI\state
if not exist "%STATE_DIR%" mkdir "%STATE_DIR%" >nul 2>&1
set FULL_STAMP=%STATE_DIR%\last_full_check.stamp

set NEED_FULL=
for %%A in (%*) do (
    if /i "%%~A"=="--full-check" set NEED_FULL=1
)

if not defined NEED_FULL (
    if not exist "%FULL_STAMP%" (
        set NEED_FULL=1
    ) else (
        forfiles /P "%STATE_DIR%" /M "last_full_check.stamp" /D -7 >nul 2>&1
        if errorlevel 1 set NEED_FULL=
        if not errorlevel 1 set NEED_FULL=1
    )
)

REM Check for missing dependencies - separate critical vs optional
if not defined NEED_FULL goto :check_pytorch
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
        choice /C YNS /N /D S /T 10 /M "Install optional packages? (Y=Yes / N=No / S=Skip for now, default S in 10s): "
        if errorlevel 3 (
            echo.
            echo  ▓ Skipping optional packages. ComfyUI will launch with limited features.
            echo.
        ) else if errorlevel 2 (
            echo.
            echo  ▓ Skipping optional packages.
            echo.
        ) else (
            echo.
            echo  ▓ Installing optional packages...
            python -m pip install --disable-pip-version-check comfyui-workflow-templates comfyui-embedded-docs >nul 2>&1
            echo  ▓ Optional packages installed.
            echo.
        )
        type nul > "%FULL_STAMP%"
        goto :check_pytorch
    )
    type nul > "%FULL_STAMP%"
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
        python -m pip install --progress-bar on --disable-pip-version-check -r requirements.txt
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
        type nul > "%FULL_STAMP%"
    ) else if /i "%INSTALL_CHOICE%"=="C" (
        echo.
        echo  ▓ Installing critical packages only...
        echo.
        python -m pip install --progress-bar on --disable-pip-version-check torch torchvision torchaudio numpy einops transformers tokenizers sentencepiece safetensors aiohttp yarl pyyaml Pillow scipy tqdm psutil alembic SQLAlchemy av comfyui-frontend-package
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
        type nul > "%FULL_STAMP%"
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
REM Fast path: read torch version without importing (import is slow)
python -c "import sys; from importlib import util, metadata; s=util.find_spec('torch'); print('HAS_TORCH:' + ('1' if s else '0')); print('PYTORCH_VERSION:' + (metadata.version('torch') if s else 'NONE'))" > torch_meta.tmp 2>nul
set HAS_TORCH=
set PYTORCH_VERSION=NONE
for /f "tokens=1,* delims=:" %%a in (torch_meta.tmp) do (
    if "%%a"=="HAS_TORCH" set HAS_TORCH=%%b
    if "%%a"=="PYTORCH_VERSION" set PYTORCH_VERSION=%%b
)
del torch_meta.tmp 2>nul

REM Default CUDA vars
set CUDA_AVAILABLE=False
set CUDA_VERSION=NONE

REM Only import torch to check CUDA if present and not CPU build
if "%HAS_TORCH%"=="1" (
    echo %PYTORCH_VERSION% | findstr /C:"+cpu" >nul
    if errorlevel 1 (
        python -c "import torch; print('CUDA_AVAILABLE:' + str(torch.cuda.is_available())); print('CUDA_VERSION:' + (torch.version.cuda or 'NONE'))" > pytorch_check.tmp 2>nul
        if not errorlevel 1 (
            for /f "tokens=1,* delims=:" %%a in (pytorch_check.tmp) do (
                if "%%a"=="CUDA_AVAILABLE" set CUDA_AVAILABLE=%%b
                if "%%a"=="CUDA_VERSION" set CUDA_VERSION=%%b
            )
        )
        del pytorch_check.tmp 2>nul
    )
)

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
    choice /C YN /N /D N /T 15 /M "Install CUDA-enabled PyTorch now? (Y/N, default N in 15s): "
    if errorlevel 2 (
        echo.
        echo  ▓ Skipping CUDA PyTorch installation.
        echo    ComfyUI will not be able to run with CPU-only PyTorch.
        echo    Please install CUDA-enabled PyTorch manually and try again.
        echo.
        pause
        exit /b 0
    ) else (
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
        python -m pip install --progress-bar on --disable-pip-version-check torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu130
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
            REM Continue to launch (offer updates) even if verification failed
            goto :maybe_update_torch
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
                REM Continue to launch (offer updates)
                goto :maybe_update_torch
            )
        )
        REM If verification failed but installation succeeded, continue anyway
        goto :maybe_update_torch
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

REM If CUDA is available after checks, offer optional updates then show all-clear banner
if /i "%CUDA_AVAILABLE%"=="True" goto :maybe_update_torch

REM Otherwise go straight to launch (CPU fallback accepted)
goto :check_port

:maybe_update_torch
REM Quick connectivity probe - skip updates if offline
powershell -NoProfile -Command "try{(Invoke-WebRequest -Uri 'https://pypi.org' -Method Head -TimeoutSec 3)>$null; exit 0}catch{exit 1}"
if errorlevel 1 (
    echo.
    echo  ▓ Looks like we're offline. Skipping update checks.
    goto :all_clear_banner
)

set OUTDATED_TORCH=
python -m pip list --disable-pip-version-check --outdated --format=freeze 2>nul | findstr /i "^torch==" > outdated_torch.tmp
for /f %%i in (outdated_torch.tmp) do set OUTDATED_TORCH=1
del outdated_torch.tmp 2>nul

if defined OUTDATED_TORCH (
    echo.
    echo  ╔═══════════════════════════════════════════════════════════╗
    echo  ║               PyTorch Updates Available                   ║
    echo  ╚═══════════════════════════════════════════════════════════╝
    echo.
    echo  ▓ A newer version of PyTorch packages is available.
    echo  ▓ You can update now or skip and launch immediately.
    echo.
    choice /C YN /N /D N /T 10 /M "Update now? (Y/N, default N in 10s): "
    if errorlevel 2 (
        echo.
        echo  ▓ Skipping updates for now.
        echo.
    ) else (
        echo.
        echo  ▓ Updating PyTorch packages...
        python -m pip install --progress-bar on --disable-pip-version-check --upgrade torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu130
        echo.
    )
)

:all_clear_banner
echo.
echo  ╔═══════════════════════════════════════════════════════════╗
echo  ║                      You're All Set!                      ║
echo  ╚═══════════════════════════════════════════════════════════╝
echo.
echo  ▓ CUDA-enabled PyTorch is ready to go!
echo    Your GPU is configured and ready for ComfyUI.
echo.
echo  ▓ Launching ComfyUI in 3 seconds...
timeout /t 3 /nobreak >nul
echo.
goto :check_port

:check_port
if "%COMFY_PORT%"=="" set COMFY_PORT=8188
netstat -ano | findstr /r /c:":%COMFY_PORT% .*LISTENING" >nul
if errorlevel 1 (
    goto :port_ok
) else (
    for /l %%P in (8189,1,8199) do (
        netstat -ano | findstr /r /c:":%%P .*LISTENING" >nul
        if errorlevel 1 (
            set COMFY_PORT=%%P
            echo.
            echo  ▓ Port 8188 is busy. Rolling to free port %COMFY_PORT% in 5 seconds...
            timeout /t 5 /nobreak >nul
            goto :port_ok
        )
    )
    echo.
    echo  ▓ All fallback ports 8189-8199 appear busy. Please free a port and try again.
    echo.
    pause
    exit /b 1
)

:port_ok
goto :start_comfyui

:start_comfyui
echo.
echo  ╔═══════════════════════════════════════════════════════════╗
echo  ║                  Starting ComfyUI...                      ║
echo  ╚═══════════════════════════════════════════════════════════╝
echo.
set GUI_URL=http://127.0.0.1:%COMFY_PORT%
REM Spawn a background helper that opens the browser when the server is ready
start "" cmd /c "for /l %%i in (1,1,20) do (powershell -NoProfile -Command \"try{(Invoke-WebRequest -Uri '%GUI_URL%' -Method Head -TimeoutSec 1)>$null; exit 0}catch{exit 1}\" ^& if not errorlevel 1 goto open ^& timeout /t 1 ^>nul) ^& :open ^& start \"\" \"%GUI_URL%\""
python main.py --port %COMFY_PORT%
if errorlevel 1 (
    echo.
    echo  ╔═══════════════════════════════════════════════════════════╗
    echo  ║                    ComfyUI Crashed                        ║
    echo  ╚═══════════════════════════════════════════════════════════╝
    echo.
    echo  ▓ ComfyUI encountered an error and stopped. Here's what might help:
    echo.
    echo  ▓ Error: "Port already in use"
    echo    Solution: Close other ComfyUI instances or let this script auto-select a free port.
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



