set "BASE=%~dp0"

set PYTHONNOUSERSITE=1
set PYTHONPATH=
set PYTHONHOME=

set "USERPROFILE=%BASE%user_profile_cache"
set "HOME=%BASE%user_profile_cache"

set "HF_HOME=%BASE%user_profile_cache\huggingface"
set "TORCH_HOME=%BASE%user_profile_cache\torch"

..\python_embeded\python.exe -s ..\ComfyUI\main.py --windows-standalone-build --disable-api-nodes
echo If you see this and ComfyUI did not start try updating your Nvidia Drivers to the latest. If you get a c10.dll error you need to install vc redist that you can find: https://aka.ms/vc14/vc_redist.x64.exe
pause
