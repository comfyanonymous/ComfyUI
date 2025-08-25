@echo off
title ComfyUI-Zluda Installer

setlocal EnableDelayedExpansion
set "startTime=%time: =0%"

cls
echo -------------------------------------------------------------
Echo ******************* COMFYUI-ZLUDA INSTALL *******************
echo -------------((( For older amd gpus and apus )))-------------
echo -------------------------------------------------------------
echo.
echo  ::  %time:~0,8%  ::  - Setting up the virtual enviroment
Set "VIRTUAL_ENV=venv"
If Not Exist "%VIRTUAL_ENV%\Scripts\activate.bat" (
    python.exe -m venv %VIRTUAL_ENV%
)

If Not Exist "%VIRTUAL_ENV%\Scripts\activate.bat" Exit /B 1

echo  ::  %time:~0,8%  ::  - Virtual enviroment activation
Call "%VIRTUAL_ENV%\Scripts\activate.bat"
echo  ::  %time:~0,8%  ::  - Updating the pip package 
python.exe -m pip install --upgrade pip --quiet
echo.
echo  ::  %time:~0,8%  ::  Beginning installation ...
echo.
echo  ::  %time:~0,8%  ::  - Installing required packages (it will be detailed so you can pinpoint any problems)
pip install -r requirements.txt
echo  ::  %time:~0,8%  ::  - Installing torch for AMD GPUs (it will be detailed so you can pinpoint any problems)
pip uninstall torch torchvision torchaudio -y --quiet
pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu118 --quiet
echo  ::  %time:~0,8%  ::  - Installing onnxruntime (required by some nodes)
pip install onnxruntime --quiet
echo  ::  %time:~0,8%  ::  - (temporary numpy fix)
pip install --force-reinstall numpy==1.26.4 --quiet
echo  ::  %time:~0,8%  ::  -  Installing older transformers & safetensors for older torch.
pip install --force-reinstall transformers==4.51.3 safetensors==0.5.3 --quiet
echo.
echo  ::  %time:~0,8%  ::  Custom node(s) installation ...
echo. 
echo  ::  %time:~0,8%  ::  - Installing Comfyui Manager
cd custom_nodes
git clone https://github.com/ltdrdata/ComfyUI-Manager.git --quiet
echo  ::  %time:~0,8%  ::  - Installing ComfyUI-deepcache
git clone https://github.com/styler00dollar/ComfyUI-deepcache.git --quiet
cd ..
echo. 
echo  ::  %time:~0,8%  ::  - Patching ZLUDA (Zluda 3.9.5 for HIP SDK 5.7.1)
%SystemRoot%\system32\curl.exe -sL --ssl-no-revoke https://github.com/lshqqytiger/ZLUDA/releases/download/rel.5e717459179dc272b7d7d23391f0fad66c7459cf/ZLUDA-windows-rocm5-amd64.zip > zluda.zip
%SystemRoot%\system32\tar.exe -xf zluda.zip
del zluda.zip
copy zluda\cublas.dll %VIRTUAL_ENV%\Lib\site-packages\torch\lib\cublas64_11.dll /y >NUL
copy zluda\cusparse.dll %VIRTUAL_ENV%\Lib\site-packages\torch\lib\cusparse64_11.dll /y >NUL
copy %VIRTUAL_ENV%\Lib\site-packages\torch\lib\nvrtc64_112_0.dll %VIRTUAL_ENV%\Lib\site-packages\torch\lib\nvrtc_cuda.dll /y >NUL
copy zluda\nvrtc.dll %VIRTUAL_ENV%\Lib\site-packages\torch\lib\nvrtc64_112_0.dll /y >NUL
@echo  ::  %time:~0,8%  ::  - ZLUDA is patched. (Zluda 3.9.5 for HIP 5.7.1)
echo. 
set "endTime=%time: =0%"
set "end=!endTime:%time:~8,1%=%%100)*100+1!"  &  set "start=!startTime:%time:~8,1%=%%100)*100+1!"
set /A "elap=((((10!end:%time:~2,1%=%%100)*60+1!%%100)-((((10!start:%time:~2,1%=%%100)*60+1!%%100), elap-=(elap>>31)*24*60*60*100"
set /A "cc=elap%%100+100,elap/=100,ss=elap%%60+100,elap/=60,mm=elap%%60+100,hh=elap/60+100"
copy comfyui.bat comfyui-user.bat /y >NUL
echo ..................................................... 
echo *** Installation is completed in %hh:~1%%time:~2,1%%mm:~1%%time:~2,1%%ss:~1%%time:~8,1%%cc:~1% . 
echo *** You can use "comfyui.bat" or "comfyui-user.bat" to start the app later. 
echo *** If you want to modify the launcher please use the "comfyui-user.bat" as it is not effected by the updates.
echo ..................................................... 
echo.
echo *** Starting the Comfyui-ZLUDA for the first time, please be patient...
echo.
.\zluda\zluda.exe -- python main.py --auto-launch --use-quad-cross-attention



