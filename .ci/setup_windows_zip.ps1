Invoke-WebRequest -URI https://www.python.org/ftp/python/3.10.9/python-3.10.9-embed-amd64.zip -O python_embeded.zip
Expand-Archive python_embeded.zip
cd python_embeded
Add-Content -Path .\python310._pth -Value 'import site'
Invoke-WebRequest -Uri https://bootstrap.pypa.io/get-pip.py -OutFile get-pip.py
.\python.exe get-pip.py
.\python.exe -s -m pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117 xformers -r ../ComfyUI/requirements.txt pygit2
"../ComfyUI`n" + (Get-Content .\python310._pth -Raw) | Set-Content .\python310._pth
cd ..


mkdir ComfyUI_windows_portable
mv python_embeded ComfyUI_windows_portable
mv ComfyUI_copy ComfyUI_windows_portable/ComfyUI

cd ComfyUI_windows_portable

mkdir update
cp ComfyUI/.ci/update_windows/* ./update/
cp ComfyUI/.ci/windows_base_files/* ./

cd ..

& "C:\Program Files\7-Zip\7z.exe" a -t7z -m0=lzma -mx=8 -mfb=64 -md=32m -ms=on ComfyUI_windows_portable.7z ComfyUI_windows_portable
mv ComfyUI_windows_portable.7z ComfyUI/ComfyUI_windows_portable_nvidia_or_cpu.7z
