@echo off

:: Set default port and read from .env if exists
set "port=8188"
for /f "tokens=2 delims==" %%i in ('findstr /r "^PORT=" .env') do set "port=%%i"

:: Check if NVIDIA GPU is available and set profile accordingly
for /f "tokens=*" %%i in ('wmic path win32_videocontroller get name ^| find /i "NVIDIA"') do set "profile=cuda" && goto pull
set "profile=cpu"

:: Pull updates and handle local changes
:pull
set "cmdOutput=cmd_output.txt"
git pull > "%cmdOutput%" 2>&1

:: Handle potential conflicts or changes
findstr /C:"error: Your local changes to the following files would be overwritten by merge:" "%cmdOutput%" > nul && (
    echo Pull conflicts detected. Stashing changes...
    git stash
    git pull
    goto rebuild
)

findstr /C:"Already up to date." "%cmdOutput%" > nul || goto rebuild

echo No changes detected. Starting Docker container...
docker compose --profile %profile% up -d
goto open_browser

:: Rebuild section
:rebuild
echo Changes detected. Rebuilding Docker image...
docker compose --profile %profile% up --build -d

:: Open the browser to localhost:%port%
:open_browser
echo Opening localhost:%port% in your default browser...
start http://localhost:%port%

:: Cleanup section
:cleanup
del "%cmdOutput%"
exit /f 0
