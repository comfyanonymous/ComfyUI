@echo off

set "FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE"
set "FLASH_ATTENTION_TRITON_AMD_AUTOTUNE=TRUE"

set "MIOPEN_FIND_MODE=2"
set "MIOPEN_LOG_LEVEL=3"

set "PYTHON=%~dp0venv\Scripts\python.exe"
set "GIT="
set "VENV_DIR=.\venv"

set "COMMANDLINE_ARGS=--auto-launch --use-quad-cross-attention --reserve-vram 0.9"

:: in the comfyui-user.bat remove the dots on the line below and change the gfx1030 to your gpu's specific code. 
:: you can find out about yours here, https://llvm.org/docs/AMDGPUUsage.html#processors

:: set "TRITON_OVERRIDE_ARCH=gfx1030"

set "ZLUDA_COMGR_LOG_LEVEL=1"

:: Check Git version and installed commit hash
where git >NUL 2>&1
if errorlevel 1 (
    echo [FAIL] Git is not installed or not found in the system PATH.
    echo        Please install Git from https://git-scm.com and ensure it's added to your PATH during installation.
) else (
    for /f "tokens=2,*" %%a in ('git --version') do (
        echo [INFO] Detected Git version: %%b
    )
    for /f "usebackq delims=" %%h in (`git rev-parse --short HEAD`) do (
        echo [INFO] Current ComfyUI-Zluda commit hash: %%h
    )
)

echo [INFO] Checking and updating to a new version if possible...
copy comfy\customzluda\zluda-default.py comfy\zluda.py /y >NUL
git pull
copy comfy\customzluda\zluda.py comfy\zluda.py /y >NUL

:: Check for zluda.exe and nccl.dll inside the zluda folder
echo.
echo [INFO] Checking ZLUDA installation...
pushd .\zluda
if exist zluda.exe (
    if not exist nccl.dll (
        for /f "tokens=2 delims= " %%v in ('zluda.exe --version 2^>^&1') do (
            echo [FAIL] Detected version [%%v], but nccl.dll is missing. Likely blocked by AV as false positive.
        )
    ) else (
        for /f "tokens=2 delims= " %%v in ('zluda.exe --version 2^>^&1') do (
            echo [PASS] Detected version: %%v
        )
    )
) else (
    echo [FAIL] Can't detect zluda.exe inside .\zluda directory.
)
popd

echo [INFO] Launching application via ZLUDA...
echo.
.\zluda\zluda.exe -- %PYTHON% main.py %COMMANDLINE_ARGS%
pause
