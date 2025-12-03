#!/bin/bash
read -p "ComfyUI Path: " COMFYUI_PATH
COMFYUI_PATH=${COMFYUI_PATH}

DEFAULT_VENV="${COMFYUI_PATH}venv"
read -p "VEnv path (default: $DEFAULT_VENV): " VENV_PATH
VENV_PATH=${VENV_PATH:-$DEFAULT_VENV}

RUN_SCRIPT="comfyui-desktop/run.sh"
cat > "$RUN_SCRIPT" << EOL
#!/bin/bash
source "${VENV_PATH}/bin/activate"
cd "$COMFYUI_PATH"
python main.py
EOL

chmod +x "$RUN_SCRIPT"
if [ ! -d "/opt/comfyui" ]; then
    mkdir -p /opt/comfyui
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create /opt/comfyui. Please check permissions or run as root."
        exit 1
    fi
fi

cp 'comfyui-desktop/comfyui.png' /opt/comfyui/comfyui.png
cp 'comfyui-desktop/run.sh' /opt/comfyui/run.sh
cp 'comfyui-desktop/comfyui.desktop' /usr/share/applications/comfyui.desktop

echo "Setup complete! You can run ComfyUI using Start menu"
echo "comfyui.desktop copied to /usr/share/applications/comfyui.desktop"
