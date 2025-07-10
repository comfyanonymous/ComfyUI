@echo off
title Fix for RX480 - 580 and variants

rem Activating venv to downgrade the proper torch installation
Set "VIRTUAL_ENV=venv"
If Not Exist "%VIRTUAL_ENV%\Scripts\activate.bat" Exit /B 1

echo Virtual enviroment activation
Call "%VIRTUAL_ENV%\Scripts\activate.bat"


cls
pip uninstall torch torchvision torchaudio -y
pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu118
echo ..................................................... 
echo *** "Patching ZLUDA Again"
echo.
rmdir /S /q zluda
%SystemRoot%\system32\curl.exe -s -L https://github.com/lshqqytiger/ZLUDA/releases/download/rel.5e717459179dc272b7d7d23391f0fad66c7459cf/ZLUDA-windows-rocm5-amd64.zip > zluda.zip
%SystemRoot%\system32\tar.exe -xf zluda.zip
del zluda.zip
copy zluda\cublas.dll %VIRTUAL_ENV%\Lib\site-packages\torch\lib\cublas64_11.dll /y >NUL
copy zluda\cusparse.dll %VIRTUAL_ENV%\Lib\site-packages\torch\lib\cusparse64_11.dll /y >NUL
copy %VIRTUAL_ENV%\Lib\site-packages\torch\lib\nvrtc64_112_0.dll %VIRTUAL_ENV%\Lib\site-packages\torch\lib\nvrtc_cuda.dll /y >NUL
copy zluda\nvrtc.dll %VIRTUAL_ENV%\Lib\site-packages\torch\lib\nvrtc64_112_0.dll /y >NUL
echo.
@echo  *ZLUDA 3.9.5 for hip 5.7.1 is patched. *
echo.
echo ..................................................... 
echo *** Installation is done. --
echo .....................................................
echo.
