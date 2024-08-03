@echo off
setlocal

REM Function to check if Git is installed
:CheckGit
where git >nul 2>&1
if %ERRORLEVEL% == 0 (
    echo Git is already installed.
) else (
    echo Git is not installed. Installing...
    
    set "gitInstaller=Git-2.42.0-64-bit.exe"
    set "gitUrl=https://github.com/git-for-windows/git/releases/download/v2.42.0.windows.1/%gitInstaller%"
    powershell -Command "Invoke-WebRequest -Uri '%gitUrl%' -OutFile '%gitInstaller%'"
    
    start /wait %gitInstaller% /VERYSILENT
    
    del %gitInstaller%
    
    echo Git has been installed.
)
goto :eof

REM Check if Python is installed
python --version >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo Python3.8 is already installed.
) else (
    echo Python3.8 is not installed. Installing...
    set PYTHON_INSTALLER_URL=https://www.python.org/ftp/python/3.8.0/python-3.8.0a1-amd64.exe
    set PYTHON_INSTALLER=python_installer.exe
    powershell -Command "Invoke-WebRequest -Uri %PYTHON_INSTALLER_URL% -OutFile %PYTHON_INSTALLER%"
    %PYTHON_INSTALLER% /quiet InstallAllUsers=1 PrependPath=1
    del %PYTHON_INSTALLER%
    echo Python3.8 has been installed.
)

REM Function to check if NVIDIA drivers are installed
:CheckNvidiaDriver
wmic path Win32_PnPSignedDriver get DeviceName | find /i "NVIDIA" >nul
if %ERRORLEVEL% == 0 (
    echo NVIDIA driver is already installed.
) else (
    echo NVIDIA driver is not installed. Installing...
    
    set "nvidiaDriverInstaller= NVIDIA-Driver-531.41-Win10-Win11-64Bit-English.exe"
    set "nvidiaDriverUrl=https://download.nvidia.com/Windows/531.41/%nvidiaDriverInstaller%"
    powershell -Command "Invoke-WebRequest -Uri '%nvidiaDriverUrl%' -OutFile '%nvidiaDriverInstaller%'"
    
    start /wait %nvidiaDriverInstaller% -s
    
    del %nvidiaDriverInstaller%
    
    echo NVIDIA driver has been installed.
)
goto :eof

REM Function to check if CUDA is installed
:CheckCUDA
if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin\nvcc.exe" (
    echo CUDA is already installed.
) else (
    echo CUDA is not installed. Installing...
    
    set "cudaInstaller=cuda_11.8.0_windows.exe"
    set "cudaUrl=https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/%cudaInstaller%"
    powershell -Command "Invoke-WebRequest -Uri '%cudaUrl%' -OutFile '%cudaInstaller%'"
    
    start /wait %cudaInstaller% /S
    
    del %cudaInstaller%
    
    echo CUDA has been installed.
)
goto :eof

call :CheckGit
call :CheckNvidiaDriver
call :CheckCUDA

endlocal
pause
