@echo off

:: TODO: In the future, ALL of this (except for the initial python existance check)
:: should be done within a wrapping python script, to improve platform-independency,
:: readability and ease of maintenance.
:: But the contributor who added this (@Lex-DRL) lacks in experience
:: with launching one python interpreter from another, while also keeping
:: their output within the same console window (to avoid annoying users).
:: So, for now, as a temporary QoL improvement, at least this bat script.

:: Based on webui.bat from A11
:: X>Y syntax (1>NUL, 2>NUL) in CMD is stream output redirection.
:: 1st stream is stdout (same as giving no stream number), 2nd stream is stderr.

:: %ERRORLEVEL% contains exit code of the last called executable.

if not defined PYTHON (set PYTHON=python)
set ACTIVE_PYTHON_LABEL=global python installation

if not defined VENV_DIR (set "VENV_DIR=%~dp0%venv")
set VENV_PYTHON_PATH=%VENV_DIR%\Scripts\python.exe

if not defined MAIN_SCRIPT (set MAIN_SCRIPT=main.py)

:: We can currently use only NVIDIA under Windows, so no conditional logic here.
:: But commands are separated for future and consistency with repo instructions:
set PIP_INSTALL_ARGS_TORCH_NVIDIA=torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118 xformers
set PIP_INSTALL_ARGS_COMMON=-r requirements.txt

if not defined TMP_DIR (set TMP_DIR=tmp)
set TMP_STDOUT="%TMP_DIR%\stdout.txt"
set TMP_STDERR="%TMP_DIR%\stderr.txt"

:: Make tmp dir if not present, supress error output by redirecting to global NULL
mkdir tmp 2>NUL

if ["%SKIP_VENV%"] == ["1"] goto :venv_ready
if ["%VENV_DIR%"] == ["-"] goto :require_os_python

:: First and foremost, check if we might already have a venv.
:: If so, we don't even need a system-wide python.
"%VENV_PYTHON_PATH%" -c "" >%TMP_STDOUT% 2>%TMP_STDERR%
if %ERRORLEVEL% == 0 goto :activate_venv

echo venv doesn't exist. Locating a system-wide python...

:require_os_python
%PYTHON% -c "" >%TMP_STDOUT% 2>%TMP_STDERR%
if not %ERRORLEVEL% == 0 goto :error_no_python
if ["%VENV_DIR%"] == ["-"] goto :verify_pip_in_venv
goto :create_new_venv

:error_no_python
echo Couldn't launch python
goto :show_stdout_stderr

:create_new_venv
:: We should only get here if venv is required and doesn't exist yet.
for /f "delims=" %%i in ('CALL %PYTHON% -c "import sys; print(sys.executable)"') do set OS_PYTHON_PATH="%%i"
echo Creating a new venv in directory %VENV_DIR% using python %OS_PYTHON_PATH%
%OS_PYTHON_PATH% -m venv "%VENV_DIR%" >%TMP_STDOUT% 2>%TMP_STDERR%
if %ERRORLEVEL% == 0 goto :activate_venv
echo Unable to create venv in directory: %VENV_DIR%
goto :show_stdout_stderr

:activate_venv
set PYTHON="%VENV_PYTHON_PATH%"
set ACTIVE_PYTHON_LABEL=venv
echo venv %PYTHON%
:: ↓

:verify_pip_in_venv
%PYTHON% -m pip --help >%TMP_STDOUT% 2>%TMP_STDERR%
if %ERRORLEVEL% == 0 goto :verify_venv

:: We wasn't successfull trying to call pip from the active python.
:: Let's attempt installing pip.
echo pip not found. Attempting to add it into %ACTIVE_PYTHON_LABEL%.
%PYTHON% -m ensurepip --upgrade >%TMP_STDOUT% 2>%TMP_STDERR%
if %ERRORLEVEL% == 0 goto :verify_venv

if "%PIP_INSTALLER_LOCATION%" == "" goto :show_stdout_stderr
%PYTHON% "%PIP_INSTALLER_LOCATION%" >%TMP_STDOUT% 2>%TMP_STDERR%
if %ERRORLEVEL% == 0 goto :verify_venv
echo Couldn't install pip
goto :show_stdout_stderr

:verify_venv
:: TODO: add checks for the rest of required packages
%PYTHON% -c "import aiohttp, tqdm" >%TMP_STDOUT% 2>%TMP_STDERR%
if not %ERRORLEVEL% == 0 goto :install_venv_dependencies

%PYTHON% -c "import torch; assert torch.cuda.is_available()" >%TMP_STDOUT% 2>%TMP_STDERR%
if %ERRORLEVEL% == 0 goto :venv_ready
echo Torch with cuda support isn't available.
:: ↓

:install_venv_dependencies
echo.
echo Installing all pip dependencies into venv: %VENV_DIR%
:: Intentionally don't suppress stdout when installing with pip, to make it show progress.

:: First, update pip itself, just in case:
%PYTHON% -m pip install -U pip 2>%TMP_STDERR%
if not %ERRORLEVEL% == 0 goto :show_stdout_stderr
:: Now, torch with cuda:
%PYTHON% -m pip install -U %PIP_INSTALL_ARGS_TORCH_NVIDIA% 2>%TMP_STDERR%
if not %ERRORLEVEL% == 0 goto :show_stdout_stderr
:: Finally, the rest of requirements:
%PYTHON% -m pip install -U %PIP_INSTALL_ARGS_COMMON% 2>%TMP_STDERR%
if not %ERRORLEVEL% == 0 goto :show_stdout_stderr

echo All the dependencies are installed into venv: %VENV_DIR%
echo.
echo --------------------------------------------------
echo.
:: ↓

:venv_ready
if [%ACCELERATE%] == ["True"] goto :accelerate
goto :launch

:accelerate
echo Checking for accelerate
set ACCELERATE="%VENV_DIR%\Scripts\accelerate.exe"
if EXIST %ACCELERATE% goto :accelerate_launch

:launch
%PYTHON% %MAIN_SCRIPT% %*
if EXIST tmp/restart goto :venv_ready
pause
exit /b

:accelerate_launch
echo Accelerating
%ACCELERATE% launch --num_cpu_threads_per_process=6 %MAIN_SCRIPT%
if EXIST tmp/restart goto :venv_ready
pause
exit /b

:show_stdout_stderr

echo.
echo Exit code: %ERRORLEVEL%

for /f %%i in ("%TMP_STDOUT%") do set size=%%~zi
if %size% equ 0 goto :show_stderr
echo.
echo stdout:
type %TMP_STDOUT%

:show_stderr
for /f %%i in ("%TMP_STDERR%") do set size=%%~zi
if %size% equ 0 goto :show_stderr
echo.
echo stderr:
type %TMP_STDERR%

:endofscript
echo.
echo Launch unsuccessful. Exiting.
pause
