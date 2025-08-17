@echo off
setlocal enabledelayedexpansion

:: Try to use embedded python first
if exist ..\..\..\python_embeded\python.exe (
    :: Use the embedded python
    set PYTHON=..\..\..\python_embeded\python.exe
) else (
    :: Embedded python not found, check for python in the PATH
    for /f "tokens=* USEBACKQ" %%F in (`python --version 2^>^&1`) do (
        set PYTHON_VERSION=%%F
    )
    if errorlevel 1 (
        echo I couldn't find an embedded version of Python, nor one in the Windows PATH. Please install manually.
        pause
        exit /b 1
    ) else (
        :: Use python from the PATH (if it's the right version and the user agrees)
        echo I couldn't find an embedded version of Python, but I did find !PYTHON_VERSION! in your Windows PATH.
        echo Would you like to proceed with the install using that version? (Y/N^)
        set /p USE_PYTHON=
        if /i "!USE_PYTHON!"=="Y" (
            set PYTHON=python
        ) else (
            echo Okay. Please install manually.
            pause
            exit /b 1
        )
    )
)

:: Install the package
echo Installing...
%PYTHON% install.py
echo Done^!

@pause