@echo off
..\python_embeded\python.exe .\update.py ..\ComfyUI\ --stable
if exist update_new.py (
  move /y update_new.py update.py
  echo Running updater again since it got updated.
  ..\python_embeded\python.exe .\update.py ..\ComfyUI\ --skip_self_update --stable
)
if "%~1"=="" pause
