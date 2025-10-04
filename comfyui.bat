@echo off

set "PYTHON=%~dp0venv\Scripts\python.exe"
set "GIT="
set "VENV_DIR=.\venv"

set "COMMANDLINE_ARGS=--auto-launch --use-quad-cross-attention --reserve-vram 0.9"

set "ZLUDA_COMGR_LOG_LEVEL=1"

:: echo Checking HIP environment variables...
if defined HIP_PATH (
    echo HIP_PATH = %HIP_PATH%
) else (
    echo HIP_PATH = Not set
)
if defined HIP_PATH_57 (
    echo HIP_PATH_57 = %HIP_PATH_57%
)
if defined HIP_PATH_61 (
    echo HIP_PATH_61 = %HIP_PATH_61%
)
if defined HIP_PATH_62 (
    echo HIP_PATH_62 = %HIP_PATH_62%
)
if defined HIP_PATH_64 (
    echo HIP_PATH_64 = %HIP_PATH_64%
)
if defined HIP_PATH_70 (
    echo HIP_PATH_70 = %HIP_PATH_70%
)
echo.

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
    git --no-pager log --oneline %LOCAL%..%REMOTE%
    echo.
    echo [INFO] Pulling updates...
    git --no-pager pull
) else (
    echo [INFO] Already up to date.
)
copy comfy\customzluda\zluda-default.py comfy\zluda.py /y >NUL
echo.

:: Get AMD Software version via registry (stop on first match)
setlocal enabledelayedexpansion
for %%K in (
  "HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"
  "HKLM\SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall"
) do (
  for /f "tokens=*" %%I in ('reg query %%K 2^>nul') do (
    for /f "skip=2 tokens=2,*" %%A in ('reg query "%%I" /v DisplayName 2^>nul') do (
      set "dispName=%%B"
      echo !dispName! | findstr /i "AMD" >nul
      if !errorlevel! == 0 (
        echo !dispName! | findstr /i "Software" >nul
        if !errorlevel! == 0 (
          for /f "skip=2 tokens=2,*" %%V in ('reg query "%%I" /v DisplayVersion 2^>nul') do (
            set "dispVer=%%W"
            if defined dispVer (
              echo [INFO] !dispName! version: !dispVer!
			  goto :match
            )
          )
        )
      )
    )
  )
)
:match
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

