@echo off

set "requirements_txt=%~dp0\requirements.txt"
set "requirements_repair_txt=%~dp0\repair_dependency_list.txt"
set "python_exec=..\..\..\python_embeded\python.exe"
set "aki_python_exec=..\..\python\python.exe"

echo Installing EasyUse Requirements...

if exist "%python_exec%" (
    echo Installing with ComfyUI Portable
    "%python_exec%" -s -m pip install -r "%requirements_txt%"
)^
else if exist "%aki_python_exec%" (
    echo Installing with ComfyUI Aki
    "%aki_python_exec%" -s -m pip install -r "%requirements_txt%"
    for /f "delims=" %%i in (%requirements_repair_txt%) do (
        %aki_python_exec% -s -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple "%%i"
        )
)^
else (
    echo Installing with system Python
    pip install -r "%requirements_txt%"
)

pause