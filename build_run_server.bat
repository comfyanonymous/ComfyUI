@echo off
setlocal

set VENV_DIR=venv
set REQUIREMENTS=requirements.txt

echo Setting up virtual environment...
python -m venv %VENV_DIR%
call %VENV_DIR%\Scripts\activate.bat

pip3 show torch > nul
if %errorlevel% neq 0 (
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
)

echo Installing requirements...
pip install -r %REQUIREMENTS%

echo Setup complete.

echo Running the application...
call %VENV_DIR%\Scripts\activate.bat


python aiyo_server_main.py --config "local"