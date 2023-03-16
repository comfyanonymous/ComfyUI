..\python_embeded\python.exe .\update.py ..\ComfyUI\
..\python_embeded\python.exe -s -m pip install --upgrade torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118 xformers -r ../ComfyUI/requirements.txt pygit2
echo NOTE If you get an error with pip you can ignore it, it's pip being pip as usual, your ComfyUI should have updated anyways.
pause
