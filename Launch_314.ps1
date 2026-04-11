Write-Host "Initializing Blackwell Stack (Python 3.14.4 + CUDA 13.2)" -ForegroundColor Green

# start ComfyUI using .\Launch_314.ps1

# Activate the 3.14 venv
.\venv\Scripts\Activate.ps1

# GPU 1 is your 5070 Ti. 
# --highvram: Keeps the model weights in your 16GB VRAM for max speed.
# --use-pytorch-cross-attention: This is the native, stable path for Python 3.14.
python main.py --highvram --preview-method auto --use-pytorch-cross-attention