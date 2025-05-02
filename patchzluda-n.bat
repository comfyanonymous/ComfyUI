@Echo off
cls
echo ======================================================================
echo Custom ZLUDA Patcher (Specifically for HIP 6.2.4 with MIOPEN - Triton)
echo ======================================================================
echo.
echo :: Make sure (1) you have HIP 6.2.4 installed, 
echo :: (2) HIP Addon (https://drive.google.com/file/d/1JSVDV9yKCJ_vldXb5hS7NEHUeDJGoG6M/view?usp=sharing) downloaded and extracted into "C:\Program Files\AMD\ROCm\6.2"
echo :: (3) Change the zluda.py inside comfy\ folder with the one under comfy\customzluda\.
echo :: * Don't forget if you want to update comfy,and if there is a change in the zluda.py, just delete the file for update to work, and after it completes copy it back from the comfy\customzluda\ folder.
echo.
rmdir /S /Q zluda 2>nul
mkdir zluda
cd zluda
%SystemRoot%\system32\curl.exe -sL --ssl-no-revoke https://github.com/lshqqytiger/ZLUDA/releases/download/rel.0d1513a017397bf9ebbac0b3c846160c8d4fc700/ZLUDA-nightly-windows-rocm6-amd64.zip > zluda.zip
%SystemRoot%\system32\tar.exe -xf zluda.zip
del zluda.zip
cd ..

:: Patch DLLs
copy zluda\cublas.dll venv\Lib\site-packages\torch\lib\cublas64_11.dll /y >NUL
copy zluda\cusparse.dll venv\Lib\site-packages\torch\lib\cusparse64_11.dll /y >NUL
copy zluda\nvrtc.dll venv\Lib\site-packages\torch\lib\nvrtc64_112_0.dll /y >NUL
copy zluda\cudnn.dll venv\Lib\site-packages\torch\lib\cudnn64_9.dll /y >NUL

echo  :: ZLUDA patched for HIP SDK 6.2.4 with miopen and triton-flash attention.
pause