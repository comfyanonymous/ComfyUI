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

:: Check for Build Tools via vswhere.exe (assuming default path)
setlocal enabledelayedexpansion
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" (
    echo [ERROR] vswhere.exe not found. Cannot detect Visual Studio installations.
)
for /f "tokens=*" %%v in ('"%VSWHERE%" -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property catalog_productDisplayVersion -latest') do (
    set "VSVer=%%v"
)
if defined VSVer (
    echo [INFO] Detected Visual Studio Build Tools version: !VSVer!
) else (
    echo [ERROR] Visual Studio Build Tools with MSVC not found.
	echo         Please install from https://aka.ms/vs/17/release/vs_BuildTools.exe and ensure it's added to your PATH.
)
endlocal

:: Check Git version
where git >NUL 2>&1
if errorlevel 1 (
    echo [ERROR] Git is not installed or not found in the system PATH.
    echo         Please install Git from https://git-scm.com and ensure it's added to your PATH during installation.
) else (
    for /f "tokens=3" %%v in ('git --version') do (
        echo [INFO] Detected Git version: %%v
    )
)

:: Get ComfyUI full path, if inside Users folder censor username
setlocal enabledelayedexpansion
set "fullpath=%~dp0"
echo %fullpath% | findstr /i "\\Users\\" >nul
if errorlevel 1 (
    echo [INFO] ComfyUI-Zluda current path: %fullpath%
) else (
    for /f "tokens=1,* delims=\ " %%a in ("%fullpath%") do (
        set "drive=%%a"
        set "rest=%%b"
    )
    for /f "tokens=1,2,* delims=\" %%a in ("!rest!") do (
        if /i "%%a"=="Users" (
            set "username=%%b"
            set "afterUser=%%c"
        )
    )
    set "censoredUser=***"
    if defined afterUser (
        set "finalPath=!drive!\Users\!censoredUser!\!afterUser!"
    ) else (
        set "finalPath=!drive!\Users\!censoredUser!\"
    )
    echo [INFO] ComfyUI-Zluda current path: !finalPath!
)
endlocal

:: Check current git branch, commit hash and date (Git required)
setlocal enabledelayedexpansion
for /f "delims=" %%b in ('git rev-parse --abbrev-ref HEAD') do set "BRANCH=%%b"
for /f "delims=" %%c in ('git rev-parse HEAD') do set "COMMIT=%%c"
for /f "delims=" %%d in ('git log -1 --format^="%%ad" --date=iso-strict') do set "FULL_DATE=%%d"
for /f "tokens=1 delims=T" %%x in ("!FULL_DATE!") do set "COMMIT_DATE=%%x"
for /f "tokens=1 delims=+" %%y in ("!FULL_DATE!") do set "COMMIT_TIME=%%y"
set "COMMIT_TIME=!COMMIT_TIME:*T=!"
echo [INFO] ComfyUI-Zluda current version: !COMMIT_DATE! !COMMIT_TIME! hash: !COMMIT:~0,8! branch: !BRANCH!
endlocal

echo [INFO] Checking and updating to a new version if possible...
copy comfy\customzluda\zluda-default.py comfy\zluda.py /y >NUL
git fetch >NUL
for /f "delims=" %%L in ('git rev-parse @') do set LOCAL=%%L
for /f "delims=" %%R in ('git rev-parse @{u}') do set REMOTE=%%R
if NOT "%LOCAL%"=="%REMOTE%" (
    echo.
    echo [INFO] Update available. New commits:
    git log --oneline %LOCAL%..%REMOTE%
    echo.
    echo [INFO] Pulling updates...
    git pull
) else (
    echo [INFO] Already up to date.
)
copy comfy\customzluda\zluda.py comfy\zluda.py /y >NUL

echo.
:: Get driver version via WMIC
setlocal enabledelayedexpansion
for /f "tokens=2 delims==" %%D in ('wmic path win32_videocontroller get driverversion /value ^| find "="') do (
    set "driverVer=%%D"
)
:: TEMP mapping driver to adrenalin, TODO registry detect or external list
set "adrenalinVer=Unknown"
if "%driverVer%"=="32.0.21025.15011" set "adrenalinVer=25.9.2" & goto :GotVer
if "%driverVer%"=="32.0.21025.10016" set "adrenalinVer=25.9.1" & goto :GotVer
if "%driverVer%"=="32.0.21025.1024" set "adrenalinVer=25.8.1" & goto :GotVer
if "%driverVer%"=="31.0.21923.11000" set "adrenalinVer=25.8.1_GCN" & goto :GotVer
if "%driverVer%"=="32.0.21013.9005" set "adrenalinVer=25.6.3" & goto :GotVer
if "%driverVer%"=="32.0.21013.4013" set "adrenalinVer=25.6.2" & goto :GotVer
if "%driverVer%"=="32.0.21013.1000" set "adrenalinVer=25.6.1" & goto :GotVer
if "%driverVer%"=="32.0.21010.10" set "adrenalinVer=25.Q2" & goto :GotVer
if "%driverVer%"=="32.0.21001.9024" set "adrenalinVer=25.5.1" & goto :GotVer
if "%driverVer%"=="31.0.21923.1000" set "adrenalinVer=25.5.1_GCN" & goto :GotVer
if "%driverVer%"=="32.0.13031.8021" set "adrenalinVer=25.4.1" & goto :GotVer
if "%driverVer%"=="32.0.13031.4034" set "adrenalinVer=25.3.2" & goto :GotVer
if "%driverVer%"=="32.0.13031.3015" set "adrenalinVer=25.3.1" & goto :GotVer
if "%driverVer%"=="32.0.12033.5029" set "adrenalinVer=25.2.1" & goto :GotVer
if "%driverVer%"=="32.0.12033.5010" set "adrenalinVer=25.1.1" & goto :GotVer
if "%driverVer%"=="32.0.12033.1030" set "adrenalinVer=24.12.1" & goto :GotVer
if "%driverVer%"=="32.0.12030.9" set "adrenalinVer=24.Q4" & goto :GotVer
if "%driverVer%"=="32.0.12019.1028" set "adrenalinVer=24.10.1" & goto :GotVer
if "%driverVer%"=="32.0.12011.1036" set "adrenalinVer=24.9.1" & goto :GotVer
if "%driverVer%"=="31.0.21921.1000" set "adrenalinVer=24.9.1_GCN" & goto :GotVer
:GotVer
if "%adrenalinVer%"=="Unknown" (
    echo [INFO] Driver version: %driverVer%
) else (
    echo [INFO] Driver version: %driverVer% [AMD Software: %adrenalinVer%]
)
endlocal

:: Check for zluda.exe and nccl.dll inside the zluda folder, pull version info from exe and build info from nvcuda.dll (via py script)
setlocal enabledelayedexpansion
pushd .\zluda
set "nightlyFlag=[unknown build]"
if exist zluda.exe (
    for /f "tokens=2 delims= " %%v in ('zluda.exe --version 2^>^&1') do set "zludaVer=%%v"
    for /f "delims=" %%f in ('python "..\comfy\customzluda\nvcuda.zluda_get_nightly_flag.py" 2^>nul') do set "nightlyFlag=%%f"
    if not exist nccl.dll (
        echo [ERROR] Detected ZLUDA version: !zludaVer! !nightlyFlag!, but nccl.dll is missing. Likely blocked by AV as false positive.
    ) else (
        echo [INFO] ZLUDA version: !zludaVer! !nightlyFlag!
    )
) else (
    echo [ERROR] Can't detect zluda.exe inside .\zluda directory.
)
popd
endlocal

echo [INFO] Launching application via ZLUDA...
echo.
.\zluda\zluda.exe -- %PYTHON% main.py %COMMANDLINE_ARGS%
pause
