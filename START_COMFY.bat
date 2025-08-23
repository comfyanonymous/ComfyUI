@echo off
setlocal
cd /d "%~dp0"

REM Ensure venv exists
if not exist "venv\Scripts\activate.bat" (
  echo ERROR: Virtual environment not found. Run 1_INSTALL.bat first.
  pause
  exit /b 1
)

REM Activate venv
call "venv\Scripts\activate.bat"

REM Ensure main.py exists
if not exist "main.py" (
  echo ERROR: main.py not found in: %cd%
  pause
  exit /b 1
)

set "LOG=%TEMP%\main_run_%RANDOM%.log"

REM Run and capture output
python "main.py" > "%LOG%" 2>&1
set "RC=%ERRORLEVEL%"

type "%LOG%"

REM If Torch is CPU-only, repair for CUDA 12.9
findstr /I /C:"AssertionError: Torch not compiled with CUDA enabled" "%LOG%" >nul
if %ERRORLEVEL%==0 (
  echo Torch is CPU-only; repairing for CUDA 12.9...

  REM Check Torch-reported CUDA version
  set "TORCH_CUDA="
  for /f "delims=" %%V in ('python -c "import torch,sys; print(getattr(torch.version,'cuda',None) or '')" 2^>NUL') do set "TORCH_CUDA=%%V"
  echo Detected torch.version.cuda="%TORCH_CUDA%"

  REM If not 12.9 (or blank), install CUDA 12.9 runtime via pip
  if /I not "%TORCH_CUDA%"=="12.9" (
    echo Installing NVIDIA CUDA 12.9 runtime via pip...
    python -m pip install --upgrade pip
    python -m pip install nvidia-pyindex
    python -m pip install -U nvidia-cuda-runtime-cu129
  )

  REM Reinstall PyTorch for cu129
  echo Reinstalling PyTorch CUDA 12.9 wheels...
  python -m pip uninstall -y torch torchaudio
  python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129

  echo Re-running main.py...
  python "main.py"
  set "RC=%ERRORLEVEL%"
)

del "%LOG%" >nul 2>&1

if not "%RC%"=="0" (
  echo main.py exited with code %RC%.
) else (
  echo Done.
)

pause
exit /b %RC%