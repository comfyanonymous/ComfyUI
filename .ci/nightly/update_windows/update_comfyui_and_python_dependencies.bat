..\python_embeded\python.exe .\update.py ..\ComfyUI\
..\python_embeded\python.exe -s -m pip install --upgrade --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121 -r ../ComfyUI/requirements.txt pygit2
pause
