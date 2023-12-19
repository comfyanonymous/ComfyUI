#!/bin/bash

# Allow it to be executed
# Starts ComfyUI
# If you want to use a specific virtual environment, check out the comfyui-venv.sh.example file instead

# Set custom path to Python
python_path="path/to/python"

echo "Python path found, using the preferred Python version."
${python_path} -c "import sys; print(sys.executable)"
${python_path} "--version"

# If the specified Python path doesn't exist, use the environment's default 'python' command
if [ ! -x "${python_path}" ]; then
    python_path="python"
    
        echo "Python path not found, using the environment's default 'python' command instead."
    ${python_path} -c "import sys; print(sys.executable)"
    ${python_path} "--version"
fi

# Include user-specific command-line arguments if available
if [ -f "comfyui-user.conf" ]; then
    source "comfyui-user.conf"
    echo "comfyui-user.conf found."
    echo -n "Commands specified: "
    if [ -z "$COMMANDLINE_ARGS" ]; then
        echo "No commands found."
    else
        echo "${COMMANDLINE_ARGS}"
    fi
else
    echo "comfyui-user.conf not found."
fi

# Run main.py using the specified Python version and command-line arguments
${python_path} "$(dirname "$0")/main.py" ${COMMANDLINE_ARGS}
