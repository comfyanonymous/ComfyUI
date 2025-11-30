@echo off

rem Check if any arguments were provided
if "%~1"=="" (
    echo No folders specified. Please provide folder names as arguments.
    echo Usage: %~nx0 folder1 folder2 folder3 ...
    exit /b 1
)

rem Process each argument as a folder
:process_folders
if "%~1"=="" goto :done
    if not exist "%~1" (
        echo Creating directory: %~1
        mkdir "%~1"
    )
    echo icacls "%~1" /setintegritylevel "(OI)(CI)Low"
    icacls "%~1" /setintegritylevel "(OI)(CI)Low" || goto :errorexit
    shift
    goto :process_folders

:done
echo Permissions set up successfully
exit /b 0

:errorexit
echo Sandbox permission setup script failed
rem Wait for a key to be pressed if unsuccessful so user can read the error
rem before the command window closes.
pause
exit /b 1
