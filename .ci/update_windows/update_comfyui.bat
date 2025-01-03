@echo off
:: Set the http proxy here like `set proxy="http://127.0.0.1:888/"`. No spacebar allowed.
set proxy=""
..\python_embeded\python.exe .\update.py ..\ComfyUI\ --proxy %proxy%
if exist update_new.py (
  move /y update_new.py update.py
  echo Running updater again since it got updated.
  ..\python_embeded\python.exe .\update.py ..\ComfyUI\ --skip_self_update --proxy %proxy%
)
if "%~1"=="" pause
