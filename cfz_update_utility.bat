@echo off
title ComfyUI-Zluda Update Utility v0.1
setlocal enabledelayedexpansion

set "PYTHON=%~dp0venv\Scripts\python.exe"
set "GIT="
set "VENV_DIR=.\venv"

:get_version_info
for /f "delims=" %%b in ('git rev-parse --abbrev-ref HEAD') do set "BRANCH=%%b"
for /f "delims=" %%c in ('git rev-parse HEAD') do set "COMMIT=%%c"
for /f "delims=" %%d in ('git log -1 --format^="%%ad" --date=iso-strict') do set "FULL_DATE=%%d"
for /f "tokens=1 delims=T" %%x in ("!FULL_DATE!") do set "COMMIT_DATE=%%x"
for /f "tokens=1 delims=+" %%y in ("!FULL_DATE!") do set "COMMIT_TIME=%%y"
set "COMMIT_TIME=!COMMIT_TIME:*T=!"

:get_version_info
:menu
cls
echo ======================================================================
echo * * *             ComfyUI-Zluda Update Utility v0.1              * * *
echo ----------------------------------------------------------------------
echo current version: !COMMIT_DATE! !COMMIT_TIME! hash: !COMMIT:~0,8! branch: !BRANCH!
echo ======================================================================
echo 1. Check for Updates          [Safe - check only]
echo 2. Check and Download Updates [Safe - pull latest commits]
echo 3. Fix Broken Update          [Force reset to origin/master]
echo 4. Force Reset Branch         [WARNING - hard reset]
echo 5. Info                       [Safe - local branch state]
echo.
echo 0. Exit
echo.
echo ======================================================================
REM (TODO: Auto-switch to master/main if not on correct branch, with a warning)
REM (TODO: Add check for staged/modified or uncommited changes when updating)
set choice=
set /p choice=Choose an option (0-5): 

if "%choice%"=="1" goto check_update
if "%choice%"=="2" goto regular_update
if "%choice%"=="3" goto fix_update
if "%choice%"=="4" goto force_reset
if "%choice%"=="5" goto status
if "%choice%"=="0" goto end

echo Invalid choice. Please try again...
pause
goto menu

:status
cls
echo ======================================================================
echo current version: !COMMIT_DATE! !COMMIT_TIME! hash: !COMMIT:~0,8! branch: !BRANCH!
echo ----------------------------------------------------------------------
git status --ignored
echo.
git status --porcelain
echo.
echo  M = Modified files  ?? = Untracked files
echo ======================================================================
pause
goto menu

:check_update
cls
echo ======================================================================
echo * * * Checking for available updates...
echo.

git fetch >NUL
for /f "delims=" %%L in ('git rev-parse @') do set LOCAL=%%L
for /f "delims=" %%R in ('git rev-parse @{u}') do set REMOTE=%%R
if NOT "%LOCAL%"=="%REMOTE%" (
	echo.
	echo [INFO] Update available. New commits:
	git --no-pager log --oneline %LOCAL%..%REMOTE%
	echo.
	echo * * * If your branch is "behind", you can run option 2 to update.
) else (
	echo.
    echo * * * Already up to date.
)

echo ======================================================================
pause
goto menu

:regular_update
cls
echo ======================================================================
echo * * * Checking and updating to a new version if possible...
echo.

copy comfy\customzluda\zluda-default.py comfy\zluda.py /y >NUL

git fetch >NUL
for /f "delims=" %%L in ('git rev-parse @') do set LOCAL=%%L
for /f "delims=" %%R in ('git rev-parse @{u}') do set REMOTE=%%R
if NOT "%LOCAL%"=="%REMOTE%" (
	echo.
	echo Update available. New commits:
	git --no-pager log --oneline %LOCAL%..%REMOTE%
	echo.
	echo Pulling updates...
    git --no-pager pull
	echo.
	echo * * * Update complete.
) else (
    echo.
	echo * * * Already up to date.
)

echo ======================================================================
:get_version_info
pause
goto menu

:fix_update
cls
echo ======================================================================
echo * * * Fixing broken Git state...
echo.

call "%VENV_DIR%\Scripts\activate" >NUL 2>&1

git fetch --all
git reset --hard origin/master
git --no-pager pull

echo.
echo * * * If you see a successful update now, it is done.
echo ======================================================================
:get_version_info
pause
goto menu

:force_reset
cls
echo ======================================================================
echo.
echo * * *     WARNING: This will completely reset all changes        * * *
echo.
echo ----------------------------------------------------------------------
echo.
echo               This will hard RESET to origin/master. 
echo        This includes deleting ALL untracked AND ignored files.
echo.
echo           --- Dry run: showing what would be deleted ---
git clean -n -f -d -x
echo ----------------------------------------------------------------------
echo.
set /p confirm=Are you sure? Type YES to proceed with full reset: 
if /I not "%confirm%"=="YES" goto menu

echo.
echo *** Proceeding with force reset ***
git fetch --all
git reset --hard origin/master
git clean -fdx

echo.
echo * * * Full force reset complete.
echo ======================================================================
:get_version_info
pause
goto menu

:end
cls
echo ======================================================================
echo.
echo Goodbye! ComfyUI-Zluda Update Utility has exited.
echo.
echo ======================================================================
endlocal
timeout /t 3
exit
