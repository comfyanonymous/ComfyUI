# ComfyUI

## The most powerful and modular stable diffusion GUI and backend.

## Clone this Repo

Open your terminal and type the following command:

```bash
git clone https://github.com/fablestudio/showrunner-batch-tools
cd showrunner-batch-tools
```

## Install Project Dependencies

### NVIDIA

Nvidia users should install stable pytorch using this command:

`pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121`

This is the command to install pytorch nightly instead which might have performance improvements:

`pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121`

#### Troubleshooting

If you get the "Torch not compiled with CUDA enabled" error, uninstall torch with:

`pip uninstall torch`

And install it again with the command above.

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

# Running

`python main.py`
