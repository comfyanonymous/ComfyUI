@echo off
title Fix for broken updates
cls
venv\scripts\activate
echo ..................................................... 
echo *** "fixing ..."
echo.
git fetch --all
git reset --hard origin/master
git pull
echo.
@echo  *If you see a successful update now, it is done. *
echo.
echo ..................................................... 
echo.
