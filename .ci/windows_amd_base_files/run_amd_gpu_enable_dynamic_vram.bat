set "BASE=%~dp0"

set PYTHONNOUSERSITE=1
set PYTHONPATH=
set PYTHONHOME=

set "USERPROFILE=%BASE%user_profile_cache"
set "HOME=%BASE%user_profile_cache"

set "HF_HOME=%BASE%user_profile_cache\huggingface"
set "TORCH_HOME=%BASE%user_profile_cache\torch"

.\python_embeded\python.exe -s ComfyUI\main.py --windows-standalone-build --enable-dynamic-vram
pause
