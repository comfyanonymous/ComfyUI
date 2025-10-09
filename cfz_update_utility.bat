@echo off
title ComfyUI-Zluda Update Utility v0.5
setlocal enabledelayedexpansion

set "LOG_FILE=cfz_update_utility_log.txt"
set "ENABLE_LOGGING=1"

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

for /f "tokens=2,3 delims= " %%a in ("%date% %time%") do (
    set "LOG_DATE=%%a"
    set "LOG_TIME=%%b"
)
set "LOG_TIME=!LOG_TIME:~0,8!"

>> "%LOG_FILE%" (
    echo.
    echo [START] ComfyUI-Zluda Update Utility Log
    echo [DATE]  !LOG_DATE! !LOG_TIME!
    echo ========================================================
)

goto show_menu

:manual_git
cls
echo :manual_git=====================================================================
call :log Entered manual_git
echo [INFO] Type any git command below, without "git" prefix, e.g.:
echo         --version
echo         --help
echo         status
echo.

set /p gitcmd=Enter git command: 

if /i "%gitcmd:~0,4%"=="git " (
    set "gitcmd=%gitcmd:~4%"
)

if "%gitcmd%"=="" (
    echo [INFO] No command entered.
    call :pausefix
    goto manual_git
)

echo %gitcmd% | findstr /R /I "\<reset\> \<clean\> \<rebase\>" >nul
if %errorlevel%==0 (
    echo [WARN] Destructive commands are not allowed here.
    call :pausefix
    goto show_menu
)

set "CHECK_HELP=%gitcmd:--help=%"
if not "%CHECK_HELP%"=="%gitcmd%" (
    echo [WARN] 'git --help' may open a separate window or fail to display in this console.
)

echo.
set /p confirm=Run this command? (Y/N): 
call :log User confirmation for git command: %confirm%
if /I not "%confirm%"=="Y" (
    echo [INFO] Command canceled.
	call :log User canceled the command
    call :pausefix
    goto show_menu
)

set "fullgitcmd=git %gitcmd%"
echo [EXEC] %fullgitcmd%
echo --------------------------------------------------------------------------------
set "TMP_FILE=%TEMP%\git_output.tmp"
%fullgitcmd% > "%TMP_FILE%" 2>&1
if exist "%TMP_FILE%" (
    type "%TMP_FILE%"
    type "%TMP_FILE%" >> "%LOG_FILE%"
    del "%TMP_FILE%"
) else (
    echo [WARN] Output file not found. Git may have launched an external viewer.
)
call :log Ran manual git command: %fullgitcmd%
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

:log
if "%ENABLE_LOGGING%"=="1" (
    setlocal enabledelayedexpansion
    for /f "tokens=2,3 delims= " %%a in ("%date% %time%") do (
        set "LOG_DATE=%%a"
        set "LOG_TIME=%%b"
    )
    set "LOG_TIME=!LOG_TIME:~0,8!"
    echo [!LOG_DATE! !LOG_TIME!] %* >> "%LOG_FILE%"
    endlocal
)
exit /b

:show_menu
call :get_local_version_info
call :get_remote_version_info
goto menu

:menu
cls
echo :menu===========================================================================
echo              ComfyUI-Zluda Update Utility v0.5                   branch: !BRANCH!
echo ================================================================================
call :log Entered menu
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
call :log Entered switch_branch
echo [INFO] Available branches:
git branch -a
echo.
set /p target_branch=Enter branch name to switch to: 
call :log User selected branch: %target_branch%
if "%target_branch%"=="" (
    echo [INFO] No branch entered.
	call :log User did not enter a branch to switch to
    call :pausefix
    goto show_menu
)
git checkout "%target_branch%"
call :log Switched to branch: %target_branch%
call :pausefix
goto show_menu

:view_commits
cls
echo :view_commits===================================================================
call :log Entered view_commits
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
call :log Entered status
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
call :log Entered check_update
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
call :log Entered regular_update
echo [INFO] Checking and updating to a new version if possible...
copy comfy\customzluda\zluda-default.py comfy\zluda.py /y >NUL
call :log Copied zluda-default.py to zluda.py
git fetch >NUL
call :log Starting regular update
for /f "delims=" %%L in ('git rev-parse @') do set LOCAL=%%L
for /f "delims=" %%R in ('git rev-parse @{u}') do set REMOTE=%%R
if NOT "%LOCAL%"=="%REMOTE%" (
	echo.
	echo [INFO] Update available. New commits:
	git --no-pager log --oneline %LOCAL%..%REMOTE%
	git --no-pager log --oneline %LOCAL%..%REMOTE% >> "%LOG_FILE%" 2>&1
    call :log New commits between %LOCAL% and %REMOTE%
	echo.
	echo [INFO] Pulling updates...
    git --no-pager pull >> "%LOG_FILE%" 2>&1
    call :log Pulled updates
	echo.
	echo [DONE] Update complete.
	call :log Update process completed
) else (
	echo [INFO] Already up to date.
	call :log No update necessary
)
call :pausefix
goto show_menu

:fix_update
cls
echo :fix_update=====================================================================
call :log Entered fix_update
echo [INFO] Fixing broken Git state...
echo.
git fetch --all >> "%LOG_FILE%" 2>&1
call :log Ran git fetch --all
git reset --hard origin/%DEFAULT_BRANCH% >> "%LOG_FILE%" 2>&1
call :log Ran git reset --hard origin/%DEFAULT_BRANCH%
echo.
echo [INFO] If you see a successful update now, it is done.
call :log fix_update complete
call :pausefix
goto show_menu

:force_reset
cls
echo :force_reset====================================================================
call :log Entered force_reset
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

if /I not "%confirm%"=="YES" (
    call :log User canceled force reset with input: %confirm%
    goto show_menu
)

call :log User confirmed force reset: %confirm%
echo.
echo [INFO] Proceeding with force reset...
git fetch --all >> "%LOG_FILE%" 2>&1
call :log Ran git fetch --all
git reset --hard origin/%DEFAULT_BRANCH% >> "%LOG_FILE%" 2>&1
call :log Ran git reset --hard origin/%DEFAULT_BRANCH%
git clean -fdx >> "%LOG_FILE%" 2>&1
call :log Ran git clean -fdx
echo.
echo [INFO] Full force reset complete.
call :pausefix
goto show_menu

:end
call :log Exiting...
endlocal
exit
