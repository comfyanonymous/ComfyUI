@echo off
title ComfyUI-Zluda Update Utility v0.4
setlocal enabledelayedexpansion

set "DEFAULT_BRANCH=master"

set "BRANCH="
set "COMMIT="
set "COMMIT_DATE="
set "COMMIT_TIME="
set "REMOTE_COMMIT="
set "REMOTE_COMMIT_DATE="
set "REMOTE_COMMIT_TIME="
set "AHEAD=0"
set "BEHIND=0"
	
git rev-parse --is-inside-work-tree >NUL 2>&1
if ERRORLEVEL 1 (
    echo This directory is not a Git repository.
    pause
    exit /b
)
goto show_menu

:manual_git
cls
echo :manual_git=====================================================================
echo [INFO] Type any git command below, e.g.:
echo         --version
echo         --help
echo         status
echo.
set /p gitcmd=git 

if "%gitcmd%"=="" (
    echo [INFO] No command entered.
    call :pausefix
    goto manual_git
)

echo.
echo %gitcmd% | findstr /R /I "\<reset\> \<clean\> \<rebase\>" >nul
if %errorlevel%==0 (
    echo [WARN] Destructive commands are not allowed here.
    call :pausefix
    goto show_menu
)

echo.
set /p confirm=Run this command? (Y/N): 
if /I not "%confirm%"=="Y" (
    echo [INFO] Command canceled.
    call :pausefix
    goto show_menu
)

echo [EXEC] git %gitcmd%
echo --------------------------------------------------------------------------------
git %gitcmd%
echo --------------------------------------------------------------------------------
call :pausefix
goto show_menu

:get_local_version_info
for /f "delims=" %%b in ('git rev-parse --abbrev-ref HEAD') do set "BRANCH=%%b"
for /f "delims=" %%c in ('git rev-parse HEAD') do set "COMMIT=%%c"
for /f "delims=" %%d in ('git log -1 --format^="%%ad" --date=iso-strict') do set "FULL_DATE=%%d"
for /f "tokens=1,2 delims=T" %%x in ("!FULL_DATE!") do (
    set "COMMIT_DATE=%%x"
    set "COMMIT_TIME=%%y"
)
set "COMMIT_TIME=!COMMIT_TIME:~0,8!"
exit /b

:get_remote_version_info
for /f "delims=" %%r in ('git rev-parse @{u}') do set "REMOTE_COMMIT=%%r"
for /f "delims=" %%d in ('git log -1 --format^="%%ad" --date=iso-strict @{u}') do set "REMOTE_FULL_DATE=%%d"
for /f "tokens=1,2 delims=T" %%x in ("!REMOTE_FULL_DATE!") do (
    set "REMOTE_COMMIT_DATE=%%x"
    set "REMOTE_COMMIT_TIME=%%y"
)
set "REMOTE_COMMIT_TIME=!REMOTE_COMMIT_TIME:~0,8!"
exit /b

:pausefix
echo %cmdcmdline% | findstr /i "\/c" >nul
if %errorlevel%==0 (
    pause
) else (
    pause
)
exit /b

:show_menu
call :get_local_version_info
call :get_remote_version_info
goto menu

:menu
cls
echo :menu===========================================================================
echo              ComfyUI-Zluda Update Utility v0.4                   branch: !BRANCH!
echo ================================================================================
echo  1. Show Updates         [Safe] - git fetch, git log
echo  2. Download Updates     [Safe] - git fetch, git log, git pull
echo  3. Fix Broken Update    [WARN] - git fetch, git reset             [Destructive]
echo  4. Force Reset Branch   [WARN] - git fetch, git reset, git clean  [Destructive]
echo  5. Branch Info          [Safe] - git status
echo  6. View Last 50 Commits [Safe] - git log --oneline
echo  7. Switch Branch        [Safe] - git checkout
echo  8. Git Command          [Adv.] - input a raw git command manually
echo.
echo  0. Exit
echo.
echo ---version----------------------------------------------------------------------
echo                    LOCAL: (!COMMIT:~0,8!) !COMMIT_DATE! !COMMIT_TIME!
echo                   REMOTE: (!REMOTE_COMMIT:~0,8!) !REMOTE_COMMIT_DATE! !REMOTE_COMMIT_TIME!
echo ---notes------------------------------------------------------------------------
echo  a): Option 5 shows which files will be removed by option 4.
echo      This includes: modified (not staged), untracked, and ignored files.
echo  b): Option 3 removes modified files only.
echo      Untracked and ignored files (per .gitignore) are preserved.
echo  c): Batch scripts can behave unpredictably; sometimes the menu will refresh
echo      without waiting. Pressing ENTER a few times before selection can help.
echo ================================================================================
echo.
set choice=
set /p choice=Choose an option (0-8): 
if "%choice%"=="1" goto check_update
if "%choice%"=="2" goto regular_update
if "%choice%"=="3" goto fix_update
if "%choice%"=="4" goto force_reset
if "%choice%"=="5" goto status
if "%choice%"=="6" goto view_commits
if "%choice%"=="7" goto switch_branch
if "%choice%"=="8" goto manual_git
if "%choice%"=="0" goto end
echo Invalid choice. Please try again...
pause
goto show_menu

:switch_branch
cls
echo :switch_branch==================================================================
echo [INFO] Available branches:
git branch -a
echo.
set /p target_branch=Enter branch name to switch to: 
if "%target_branch%"=="" (
    echo [INFO] No branch entered.
    call :pausefix
    goto show_menu
)
git checkout "%target_branch%"
call :pausefix
goto show_menu

:view_commits
cls
echo :view_commits===================================================================
echo [INFO] Last 50 commits:
git --no-pager log --oneline -n 50
call :pausefix
goto show_menu

:status
cls
echo :status=========================================================================
echo LOCAL:  !COMMIT_DATE! !COMMIT_TIME! hash: !COMMIT:~0,8!
echo REMOTE: !REMOTE_COMMIT_DATE! !REMOTE_COMMIT_TIME! hash: !REMOTE_COMMIT:~0,8!
echo --------------------------------------------------------------------------------

git rev-parse --abbrev-ref --symbolic-full-name @{u} >nul 2>&1
if errorlevel 1 (
    echo [WARN] No upstream tracking branch set.
    echo [INFO] You can set it with:
    echo         git branch --set-upstream-to=origin/%DEFAULT_BRANCH% %DEFAULT_BRANCH%
) else (
    for /f %%a in ('git rev-list --count HEAD..@{u}') do set "BEHIND=%%a"
    for /f %%a in ('git rev-list --count @{u}..HEAD') do set "AHEAD=%%a"
    if "!AHEAD!"=="0" if "!BEHIND!"=="0" (
        echo [INFO] Your branch is up to date with the remote.
    ) else (
        echo [INFO] Branch Status: Ahead by !AHEAD! / Behind by !BEHIND!
    )
)
echo --------------------------------------------------------------------------------
echo [INFO] Summary of changes (git diff --stat):
git --no-pager diff --stat 2>nul
echo --------------------------------------------------------------------------------
echo [INFO] Git status:
git status
echo --------------------------------------------------------------------------------
echo [INFO] Git status (porcelain):
git status --porcelain
echo --------------------------------------------------------------------------------
echo [INFO] Git status (ignored):
git status --ignored -s
echo ================================================================================
call :pausefix
goto show_menu

:check_update
cls
echo :check_update===================================================================
echo [INFO] Checking for available updates...
git fetch >NUL
for /f "delims=" %%L in ('git rev-parse @') do set LOCAL=%%L
for /f "delims=" %%R in ('git rev-parse @{u}') do set REMOTE=%%R
if NOT "%LOCAL%"=="%REMOTE%" (
	echo.
	echo [INFO] Update available. New commits:
	git --no-pager log --oneline %LOCAL%..%REMOTE%
	echo.
	echo [INFO] If your branch is "behind", you can run option 2 to update.
) else (
    echo [INFO] Already up to date.
)
echo.
echo [INFO] Checking for uncommitted changes...
for /f %%s in ('git status --porcelain') do (
    echo [WARN] You have uncommitted changes. Please commit or stash them before updating.
	call :pausefix
    goto show_menu
)

:regular_update
cls
echo :regular_update=================================================================
echo [INFO] Checking and updating to a new version if possible...
copy comfy\customzluda\zluda-default.py comfy\zluda.py /y >NUL
git fetch >NUL
for /f "delims=" %%L in ('git rev-parse @') do set LOCAL=%%L
for /f "delims=" %%R in ('git rev-parse @{u}') do set REMOTE=%%R
if NOT "%LOCAL%"=="%REMOTE%" (
	echo.
	echo [INFO] Update available. New commits:
	git --no-pager log --oneline %LOCAL%..%REMOTE%
	echo.
	echo [INFO] Pulling updates...
    git --no-pager pull
	echo.
	echo [DONE] Update complete.
) else (
	echo [INFO] Already up to date.
)
call :pausefix
goto show_menu

:fix_update
cls
echo :fix_update=====================================================================
echo [INFO] Fixing broken Git state...
echo.
git fetch --all
git reset --hard origin/%DEFAULT_BRANCH%
echo.
echo [INFO] If you see a successful update now, it is done.
call :pausefix
goto show_menu

:force_reset
cls
echo :force_reset====================================================================
echo.
echo           WARNING: This will completely reset all changes
echo.
echo --------------------------------------------------------------------------------
echo.
echo               This will hard RESET to origin/%DEFAULT_BRANCH%. 
echo        This includes deleting ALL untracked AND ignored files.
echo.
echo           --- Dry run: showing what would be deleted ---
git clean -n -f -d -x
echo --------------------------------------------------------------------------------
echo.
set /p confirm=Are you sure? Type YES to proceed with full reset: 
if /I not "%confirm%"=="YES" goto show_menu
echo.
echo [INFO] Proceeding with force reset...
git fetch --all
git reset --hard origin/%DEFAULT_BRANCH%
git clean -fdx
echo.
echo [INFO] Full force reset complete.
call :pausefix
goto show_menu

:end
endlocal
exit
