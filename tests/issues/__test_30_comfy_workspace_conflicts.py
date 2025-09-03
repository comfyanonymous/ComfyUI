"""
# get the workspace and set it up
git clone git@github.com:comfyanonymous/ComfyUI.git comfyanonymous_ComfyUI
cd comfyanonymous_ComfyUI
uv venv
source .venv/bin/activate
uv pip install --torch-backend=auto requirements.txt

# install livepeer with up to date comfyui lts
uv pip install --torch-backend=auto git+https://github.com/livepeer/comfystream.git --overrides=<(echo "comfyui@git+https://github.com/hiddenswitch/ComfyUI.git@fixes/issue-30")

# install the nodes
curl -L -o comfystream.zip https://github.com/livepeer/comfystream/archive/refs/heads/main.zip
mkdir -p custom_nodes
unzip comfystream.zip -d custom_nodes/
rm comfystream.zip

"""