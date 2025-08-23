@echo off
setlocal

set "ROOT=D:\4090_ComfyUI"
pushd "%ROOT%"

:: Activate venv, then list packages
for %%A in ("venv\Scripts\activate.bat" ".venv\Scripts\activate.bat") do (
  if exist "%%~A" (
    call "%%~A"
    goto after_venv
  )
)
:after_venv
pip list

:: Ensure custom_nodes exists
set "CNDIR=%ROOT%\custom_nodes"
if not exist "%CNDIR%\" mkdir "%CNDIR%"

:: Check for ComfyUI-Manager and clone if missing
set "TARGET=%CNDIR%\ComfyUI-Manager"
if exist "%TARGET%\NUL" (
  echo %TARGET%
) else (
  echo ComfyUI-Manager folder not found.
  echo Cloning into "%TARGET%" ...
  git clone --depth 1 https://github.com/Comfy-Org/ComfyUI-Manager "%TARGET%"
  if errorlevel 1 (
    echo git clone failed with errorlevel %errorlevel%.
  ) else (
    echo Clone complete.
  )
)

pause