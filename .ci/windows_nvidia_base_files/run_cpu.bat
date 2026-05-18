set "BASE=%~dp0"

set "PYTHONNOUSERSITE=1"
set "PYTHONPATH="
set "PYTHONHOME="

.\python_embeded\python.exe -s ComfyUI\main.py --cpu --windows-standalone-build
pause
