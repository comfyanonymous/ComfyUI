@echo off
call venv\scripts\activate
pip uninstall numpy -y --quiet
pip install numpy==1.26.4 --quiet
echo numpy 1.26.4 reinstalled.
pause
