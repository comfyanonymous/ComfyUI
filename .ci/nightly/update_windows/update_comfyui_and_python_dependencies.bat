..\python_embeded\python.exe .\update.py ..\ComfyUI\
..\python_embeded\python.exe -s -m pip install --upgrade --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cu118 -r ../ComfyUI/requirements.txt pygit2
pause
