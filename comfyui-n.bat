@echo off

set FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
set FLASH_ATTENTION_TRITON_AMD_AUTOTUNE=TRUE

set MIOPEN_FIND_MODE=2
set MIOPEN_LOG_LEVEL=3

set PYTHON="%~dp0/venv/Scripts/python.exe"
set GIT=
set VENV_DIR=./venv

set COMMANDLINE_ARGS=--auto-launch --use-quad-cross-attention --reserve-vram 0.9

:: in the comfyui-user.bat remove the dots on the line below and change the gfx1030 to your gpu's specific code. 
:: you can find out about yours here, https://llvm.org/docs/AMDGPUUsage.html#processors

:: Set TRITON_OVERRIDE_ARCH=gfx1030

set ZLUDA_COMGR_LOG_LEVEL=1

echo *** Checking and updating to new version if possible 

copy comfy\customzluda\zluda-default.py comfy\zluda.py /y >NUL
git pull
copy comfy\customzluda\zluda.py comfy\zluda.py /y >NUL

echo.
.\zluda\zluda.exe -- %PYTHON% main.py %COMMANDLINE_ARGS%
pause
