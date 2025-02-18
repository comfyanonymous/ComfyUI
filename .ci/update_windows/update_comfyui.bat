@echo off
..\python_embedded\python.exe .\update.py ..\ComfyUI\
if exist update_new.py (
  move /y update_new.py update.py
  echo Running updater again since it got updated.
  ..\python_embedded\python.exe .\update.py ..\ComfyUI\ --skip_self_update
)
if "%~1"=="" pause
