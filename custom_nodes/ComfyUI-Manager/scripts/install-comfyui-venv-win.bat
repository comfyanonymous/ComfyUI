git clone https://github.com/comfyanonymous/ComfyUI
cd ComfyUI/custom_nodes
git clone https://github.com/ltdrdata/ComfyUI-Manager
cd ..
python -m venv venv
call venv/Scripts/activate
python -m pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python -m pip install -r custom_nodes/ComfyUI-Manager/requirements.txt
cd ..
echo "cd ComfyUI" >> run_gpu.bat
echo "call venv/Scripts/activate" >> run_gpu.bat
echo "python main.py" >> run_gpu.bat

echo "cd ComfyUI" >> run_cpu.bat
echo "call venv/Scripts/activate" >> run_cpu.bat
echo "python main.py --cpu" >> run_cpu.bat
