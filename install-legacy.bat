@echo off

title ComfyUI-Zluda Installer

set ZLUDA_COMGR_LOG_LEVEL=1
setlocal EnableDelayedExpansion
set "startTime=%time: =0%"

cls
echo -------------------------------------------------------------
Echo ******************* COMFYUI-ZLUDA INSTALL *******************
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
echo  ::  %time:~0,8%  ::  - Installing required packages
pip install -r requirements.txt --quiet
echo  ::  %time:~0,8%  ::  - Installing torch for AMD GPUs (First file is 2.7 GB, please be patient)
pip uninstall torch torchvision torchaudio -y --quiet
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu118 --quiet
echo  ::  %time:~0,8%  ::  - Installing onnxruntime (required by some nodes)
pip install onnxruntime --quiet
echo  ::  %time:~0,8%  ::  - (temporary numpy fix)
pip uninstall numpy -y --quiet
pip install numpy==1.26.4 --quiet
echo.
echo  ::  %time:~0,8%  ::  Custom node(s) installation ...
echo. 
echo :: %time:~0,8%  ::  - Installing CFZ Nodes (description in readme on github) 
xcopy /E /I /Y "cfz\nodes" "custom_nodes" >NUL
echo  ::  %time:~0,8%  ::  - Installing Comfyui Manager
cd custom_nodes
git clone https://github.com/ltdrdata/ComfyUI-Manager.git --quiet
echo  ::  %time:~0,8%  ::  - Installing ComfyUI-deepcache
git clone https://github.com/styler00dollar/ComfyUI-deepcache.git --quiet
cd ..
echo. 
echo  ::  %time:~0,8%  ::  - Patching ZLUDA

:: Try default install locations for HIP
if exist "C:\Program Files\AMD\ROCm\5.7\" (
    set "HIP_PATH=C:\Program Files\AMD\ROCm\5.7"
    set "HIP_VERSION=5.7"
) else if exist "%ProgramFiles%\AMD\ROCm\6.2\" (
    set "HIP_PATH=%ProgramFiles%\AMD\ROCm\6.2"
    set "HIP_VERSION=6.2"
) else (
    echo HIP SDK not found. Please install ROCm/HIP first.
    pause
    exit /b 1
)

echo  ::  %time:~0,8%  ::  - Detected HIP version: !HIP_VERSION!

:: Map HIP version to ZLUDA release
if "!HIP_VERSION!"=="5.7" (
    set "ZLUDA_HASH=5e717459179dc272b7d7d23391f0fad66c7459cf"
    set "ZLUDA_LABEL=rocm5"
) else if "!HIP_VERSION!"=="6.2" (
    set "ZLUDA_HASH=dba64c0966df2c71e82255e942c96e2e1cea3a2d"
    set "ZLUDA_LABEL=rocm6"
) else (
    echo Unsupported HIP version: !HIP_VERSION!
    echo Supported versions are 5.7 and 6.2
    pause
    exit /b 1
)

:: Download matching ZLUDA version
rmdir /S /Q zluda 2>nul
%SystemRoot%\system32\curl.exe -sL --ssl-no-revoke https://github.com/lshqqytiger/ZLUDA/releases/download/rel.!ZLUDA_HASH!/ZLUDA-windows-!ZLUDA_LABEL!-amd64.zip > zluda.zip

if not exist zluda.zip (
    echo Failed to download ZLUDA zip for HIP version !HIP_VERSION!
    pause
    exit /b 1
)

%SystemRoot%\system32\tar.exe -xf zluda.zip
del zluda.zip

:: Patch DLLs
copy zluda\cublas.dll %VIRTUAL_ENV%\Lib\site-packages\torch\lib\cublas64_11.dll /y >NUL
copy zluda\cusparse.dll %VIRTUAL_ENV%\Lib\site-packages\torch\lib\cusparse64_11.dll /y >NUL
copy %VIRTUAL_ENV%\Lib\site-packages\torch\lib\nvrtc64_112_0.dll %VIRTUAL_ENV%\Lib\site-packages\torch\lib\nvrtc_cuda.dll /y >NUL
copy zluda\nvrtc.dll %VIRTUAL_ENV%\Lib\site-packages\torch\lib\nvrtc64_112_0.dll /y >NUL

echo  ::  %time:~0,8%  ::  - ZLUDA patched for HIP SDK !HIP_VERSION!.
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



