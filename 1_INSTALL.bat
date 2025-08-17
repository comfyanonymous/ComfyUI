@echo off
setlocal

REM Run from this script's directory
cd /d "%~dp0"

set "PYTHON_EXE=C:\Users\james\AppData\Local\Programs\Python\Python312\python.exe"

if not exist "%PYTHON_EXE%" (
  echo ERROR: Python not found at "%PYTHON_EXE%".
  exit /b 1
)

REM Create venv if missing
if exist venv (
  echo Found existing virtual environment.
) else (
  echo Creating virtual environment...
  "%PYTHON_EXE%" -m venv venv
  if errorlevel 1 (
    echo ERROR: Failed to create virtual environment.
    exit /b 1
  )
)

REM Activate venv
call "venv\Scripts\activate.bat"

REM Install requirements
if exist requirements.txt (
  echo Upgrading pip...
  python -m pip install --upgrade pip
  echo Installing packages from requirements.txt...
  python -m pip install -r requirements.txt
) else (
  echo ERROR: requirements.txt not found.
  exit /b 1
)

echo Done.
endlocal
pause