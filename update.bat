set PYTHON="%~dp0/venv/Scripts/python.exe"
set GIT=
set VENV_DIR=./venv
echo *** Checking and updating to new version if possible 
copy comfy\customzluda\zluda-default.py comfy\zluda.py /y >NUL
git pull
pause
