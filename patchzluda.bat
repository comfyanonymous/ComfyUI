@Echo off
rmdir /S /q zluda
curl -s -L https://github.com/lshqqytiger/ZLUDA/releases/download/rel.2930436a356baabafad9e66d49a5929ad2fc3eb9/ZLUDA-windows-rocm5-amd64.zip > zluda.zip
tar -xf zluda.zip
del zluda.zip
copy zluda\cublas.dll venv\Lib\site-packages\torch\lib\cublas64_11.dll /y
copy zluda\cusparse.dll venv\Lib\site-packages\torch\lib\cusparse64_11.dll /y
copy zluda\nvrtc.dll venv\Lib\site-packages\torch\lib\nvrtc64_112_0.dll /y
echo.
@echo * ZLUDA is patched. (Zluda 3.8.5 for HIP SDK 5.7.1)
