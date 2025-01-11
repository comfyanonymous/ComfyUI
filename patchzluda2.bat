@Echo off
cls
echo ===============================================
echo Custom ZLUDA Patcher (Specifically for HIP 6.x)
echo ===============================================
echo.
echo.
:: Prompt user for the ZLUDA URL
:input_url
echo Type or paste (right click on window to paste) the URL of ZLUDA version you want to download, then press ENTER:
echo Make sure it is a Windows build (e.g., ends with amd64.zip).
echo Example URL:
echo https://github.com/lshqqytiger/ZLUDA/releases/download/rel.d60bddbc870827566b3d2d417e00e1d2d8acc026/ZLUDA-windows-rocm6-amd64.zip
echo.
set /p zl="Enter URL: "
echo.

:: Validate the input
if "%zl%"=="" (
    echo Error: URL cannot be empty. Please try again.
    goto input_url
)

:: Check for required tools
where curl >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: curl is not installed or not in PATH.
    pause
    exit /b
)

where tar >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: tar is not installed or not in PATH.
    pause
    exit /b
)

:: Determine if URL contains "nightly"
echo Checking URL...
echo %zl% | find /i "nightly" >nul
if %errorlevel% equ 0 (
    echo "nightly" build detected. Using "zluda" directory...
    set target_dir=zluda
) else (
    echo "Normal" build detected. Using current directory...
    set target_dir=.
)

:: Prepare the target directory
if "%target_dir%"=="zluda" (
    rmdir /S /Q zluda >nul 2>&1
    mkdir zluda >nul 2>&1
    cd zluda
)

:: Download and extract the ZIP file
echo Downloading ZLUDA from: %zl%
curl -s -L "%zl%" -o zluda.zip
if %errorlevel% neq 0 (
    echo Error: Failed to download the file. Please check the URL and try again.
    pause
    exit /b
)

echo Extracting ZLUDA...
tar -xf zluda.zip >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Failed to extract the ZIP file.
    pause
    exit /b
)

del zluda.zip

:: Locate the Python environment dynamically
set torch_dir=%~dp0venv\Lib\site-packages\torch\lib
if not exist "%torch_dir%" (
    echo Error: Torch directory not found at %torch_dir%.
    echo Please update the script with the correct path.
    pause
    exit /b
)

:: Copy necessary files
echo Copying files to Torch library...
copy zluda\cublas.dll "%torch_dir%\cublas64_11.dll" /Y >nul 2>&1
copy zluda\cusparse.dll "%torch_dir%\cusparse64_11.dll" /Y >nul 2>&1
copy zluda\nvrtc.dll "%torch_dir%\nvrtc64_112_0.dll" /Y >nul 2>&1

if "%target_dir%"=="zluda" (
    cd ..
)

:: Final message
echo.
echo * ZLUDA has been successfully patched from the URL: %zl%
echo.
echo Press any key to close.
pause
exit /b