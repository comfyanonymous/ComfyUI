@Echo off
cls
echo -----------------------------------------------------------------------
Echo     * ZLUDA Patcher (for HIP 6.2.4 / 6.4.2 with MIOPEN and Triton)*
echo -----------------------------------------------------------------------
echo.
echo :: Activating virtual environment
Call "venv\Scripts\activate.bat"
echo :: Uninstalling previous torch packages
pip uninstall torch torchvision torchaudio -y --quiet
rmdir /S /Q "venv\Lib\site-packages\torch" 2>nul
echo :: Installing torch 2.7 - torchaudio 2.7 and torchvision 0.22
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu118 --quiet
rmdir /S /Q zluda 2>nul
mkdir zluda
cd zluda
echo :: Downloading and patching zluda 3.9.5 nightly ...
%SystemRoot%\system32\curl.exe -sL --ssl-no-revoke https://github.com/lshqqytiger/ZLUDA/releases/download/rel.5e717459179dc272b7d7d23391f0fad66c7459cf/ZLUDA-nightly-windows-rocm6-amd64.zip > zluda.zip
%SystemRoot%\system32\tar.exe -xf zluda.zip
del zluda.zip
cd ..
:: Patch DLLs
copy zluda\cublas.dll venv\Lib\site-packages\torch\lib\cublas64_11.dll /y >NUL
copy zluda\cusparse.dll venv\Lib\site-packages\torch\lib\cusparse64_11.dll /y >NUL
copy venv\Lib\site-packages\torch\lib\nvrtc64_112_0.dll venv\Lib\site-packages\torch\lib\nvrtc_cuda.dll /y >NUL
copy zluda\nvrtc.dll venv\Lib\site-packages\torch\lib\nvrtc64_112_0.dll /y >NUL
:: removed cudnn.dll patching , check the results
:: copy zluda\cudnn.dll venv\Lib\site-packages\torch\lib\cudnn64_9.dll /y >NUL
copy zluda\cufft.dll venv\Lib\site-packages\torch\lib\cufft64_10.dll /y >NUL
copy zluda\cufftw.dll venv\Lib\site-packages\torch\lib\cufftw64_10.dll /y >NUL
copy comfy\customzluda\zluda.py comfy\zluda.py /y >NUL
echo.
echo  :: ZLUDA 3.9.5 nightly patched for HIP SDK 6.2.4 / 6.4.2 with miopen and triton-flash attention.
pause


