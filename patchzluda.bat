cls
echo -----------------------------------------------------------------------
Echo                   * ZLUDA Patcher (for HIP 5.7.1)*
echo -----------------------------------------------------------------------
echo.
echo :: Activating virtual environment
Call "venv\Scripts\activate.bat"
echo :: Uninstalling previous torch packages
pip uninstall torch torchvision torchaudio -y --quiet
rmdir /S /Q "venv\Lib\site-packages\torch" 2>nul
echo :: Installing torch 2.3 - torchaudio 2.3 and torchvision 0.18
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu118 --quiet
pip install numpy==1.26.4 --quiet
rmdir /S /Q zluda 2>nul
%SystemRoot%\system32\curl.exe -sL --ssl-no-revoke https://github.com/lshqqytiger/ZLUDA/releases/download/rel.5e717459179dc272b7d7d23391f0fad66c7459cf/ZLUDA-windows-rocm5-amd64.zip > zluda.zip
%SystemRoot%\system32\tar.exe -xf zluda.zip
del zluda.zip
copy zluda\cublas.dll venv\Lib\site-packages\torch\lib\cublas64_11.dll /y
copy zluda\cusparse.dll venv\Lib\site-packages\torch\lib\cusparse64_11.dll /y
copy zluda\nvrtc.dll venv\Lib\site-packages\torch\lib\nvrtc64_112_0.dll /y
echo.
echo * ZLUDA is patched. (Zluda 3.9.5 for HIP SDK 5.7.1)
