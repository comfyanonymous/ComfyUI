@echo off
setlocal enabledelayedexpansion

echo ============================================
echo Cache Directory Cleanup Script
echo ============================================
echo.
echo This script will remove the following directories:
echo - ZLUDA ComputeCache
echo - MIOpen cache
echo - Triton cache
echo - TorchInductor temp files
echo - Torch/Triton/MIOpen/ZLUDA related cache subdirectories
echo - ComfyUI Triton and Inductor directories
echo.

REM Auto-detect COMFYUI_DIR if not set
if not defined COMFYUI_DIR (
    echo COMFYUI_DIR not set, attempting auto-detection...
    
    REM Check current directory first
    if exist "comfyui-n.bat" (
        set "COMFYUI_DIR=%CD%"
        echo   Found comfyui-n.bat in current directory: !COMFYUI_DIR!
    ) else (
        REM Check script directory
        set "SCRIPT_DIR=%~dp0"
        if exist "!SCRIPT_DIR!comfyui-n.bat" (
            set "COMFYUI_DIR=!SCRIPT_DIR!"
            REM Remove trailing backslash if present
            if "!COMFYUI_DIR:~-1!"=="\" set "COMFYUI_DIR=!COMFYUI_DIR:~0,-1!"
            echo   Found comfyui-n.bat in script directory: !COMFYUI_DIR!
        ) else (
            echo   comfyui-n.bat not found in current or script directory
        )
    )
)

if defined COMFYUI_DIR (
    echo Using COMFYUI_DIR: !COMFYUI_DIR!
) else (
    echo COMFYUI_DIR not detected - ComfyUI-specific directories will be skipped
)

echo.
pause

echo.
echo Starting cleanup...
echo.

REM ZLUDA ComputeCache
set "ZLUDA_CACHE=%USERPROFILE%\AppData\Local\ZLUDA\ComputeCache"
if exist "!ZLUDA_CACHE!" (
    echo Removing ZLUDA ComputeCache...
    rd /s /q "!ZLUDA_CACHE!" 2>nul
    if exist "!ZLUDA_CACHE!" (
        echo   Warning: Could not remove !ZLUDA_CACHE!
    ) else (
        echo   Successfully removed !ZLUDA_CACHE!
    )
) else (
    echo   ZLUDA ComputeCache not found: !ZLUDA_CACHE!
)

REM MIOpen cache
set "MIOPEN_CACHE=%USERPROFILE%\.miopen"
if exist "!MIOPEN_CACHE!" (
    echo Removing MIOpen cache...
    rd /s /q "!MIOPEN_CACHE!" 2>nul
    if exist "!MIOPEN_CACHE!" (
        echo   Warning: Could not remove !MIOPEN_CACHE!
    ) else (
        echo   Successfully removed !MIOPEN_CACHE!
    )
) else (
    echo   MIOpen cache not found: !MIOPEN_CACHE!
)

REM Triton cache
set "TRITON_CACHE=%USERPROFILE%\.triton"
if exist "!TRITON_CACHE!" (
    echo Removing Triton cache...
    rd /s /q "!TRITON_CACHE!" 2>nul
    if exist "!TRITON_CACHE!" (
        echo   Warning: Could not remove !TRITON_CACHE!
    ) else (
        echo   Successfully removed !TRITON_CACHE!
    )
) else (
    echo   Triton cache not found: !TRITON_CACHE!
)

REM TorchInductor temp files
set "TORCH_TEMP=%USERPROFILE%\AppData\Local\Temp"
echo Removing TorchInductor temp files...
for /d %%i in ("!TORCH_TEMP!\torchinductor_*") do (
    echo   Removing: %%i
    rd /s /q "%%i" 2>nul
)

REM Cache subdirectories related to torch, triton, miopen, zluda
set "USER_CACHE=%USERPROFILE%\.cache"
if exist "!USER_CACHE!" (
    echo Removing cache subdirectories related to torch, triton, miopen, zluda...
    for /d %%i in ("!USER_CACHE!\*torch*" "!USER_CACHE!\*triton*" "!USER_CACHE!\*miopen*" "!USER_CACHE!\*zluda*") do (
        if exist "%%i" (
            echo   Removing: %%i
            rd /s /q "%%i" 2>nul
        )
    )
) else (
    echo   User cache directory not found: !USER_CACHE!
)

REM ComfyUI directories (if COMFYUI_DIR is set)
if defined COMFYUI_DIR (
    set "COMFYUI_TRITON=!COMFYUI_DIR!\.triton"
    if exist "!COMFYUI_TRITON!" (
        echo Removing ComfyUI Triton directory...
        rd /s /q "!COMFYUI_TRITON!" 2>nul
        if exist "!COMFYUI_TRITON!" (
            echo   Warning: Could not remove !COMFYUI_TRITON!
        ) else (
            echo   Successfully removed !COMFYUI_TRITON!
        )
    ) else (
        echo   ComfyUI Triton directory not found: !COMFYUI_TRITON!
    )
    
    set "COMFYUI_INDUCTOR=!COMFYUI_DIR!\.inductor"
    if exist "!COMFYUI_INDUCTOR!" (
        echo Removing ComfyUI Inductor directory...
        rd /s /q "!COMFYUI_INDUCTOR!" 2>nul
        if exist "!COMFYUI_INDUCTOR!" (
            echo   Warning: Could not remove !COMFYUI_INDUCTOR!
        ) else (
            echo   Successfully removed !COMFYUI_INDUCTOR!
        )
    ) else (
        echo   ComfyUI Inductor directory not found: !COMFYUI_INDUCTOR!
    )
) else (
    echo   COMFYUI_DIR not available, skipping ComfyUI-specific directories
)

echo.
echo ============================================
echo Cleanup completed!
echo ============================================
echo.
echo If you encountered any warnings above, you may need to:
echo - Close any running applications that might be using these directories
echo - Run this script as Administrator
echo - Manually delete the directories that couldn't be removed
echo.

pause
