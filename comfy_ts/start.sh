#!/bin/bash

set -e  # Exit the script if any statement returns a non-true return value

# ---------------------------------------------------------------------------- #
#                          Function Definitions                                #
# ---------------------------------------------------------------------------- #

# Start nginx service
start_nginx() {
    echo "Starting Nginx service..."
    service nginx start
}

# Setup ssh
setup_ssh() {
    if [[ $PUBLIC_KEY ]]; then
        echo "Setting up SSH..."
        mkdir -p ~/.ssh
        echo "$PUBLIC_KEY" >> ~/.ssh/authorized_keys
        chmod 700 -R ~/.ssh
        service ssh start
    fi
}

# Export env vars
export_env_vars() {
    echo "Exporting environment variables..."
    printenv | grep -E '^RUNPOD_|^PATH=|^_=' | awk -F = '{ print "export " $1 "=\"" $2 "\"" }' >> /etc/rp_environment
    echo 'source /etc/rp_environment' >> ~/.bashrc
}

# Create symbolic links so that ComfyUI custom-nodes can find their models
create_sym_links() {
    # Read the source-destination pairs from the symlinks file
    while read -r src dest; do
        # Check if both paths are non-empty
        if [[ -n "$src" && -n "$dest" ]]; then
            create_link "$src" "$dest"
        fi
    done < "/app/symlinks.txt"
}

create_link() {
    src=$1
    dest=$2

    # Check if the source directory exists
    if [ ! -d "$src" ]; then
        echo "Warning: $src does not exist. Skipping symbolic link creation."
        return
    fi
    
    # Get the parent directory of the destination
    parent_dir=$(dirname "$dest")
    
    # Check if the grandparent directory exists
    grandparent_dir=$(dirname "$parent_dir")
    if [ ! -d "$grandparent_dir" ]; then
        echo "Warning: $grandparent_dir does not exist. Skipping symbolic link creation for $dest."
        return
    fi

    # Create the destination parent directory if it does not exist
    mkdir -p "$(dirname "$dest")"

    # Remove the existing directory or symlink at the destination if it exists
    if [ -d "$dest" ] || [ -L "$dest" ]; then
        rm -rf "$dest"
    fi
    
    echo "Creating symbolic link from $src to $dest"
    ln -s "$src" "$dest"
}

# Start jupyter lab
start_jupyter() {
    if [[ $JUPYTER_PASSWORD ]]; then
        echo "Starting Jupyter Lab..."
        mkdir -p /runpod-volume && \
        cd / && \
        nohup jupyter lab --allow-root --no-browser --port=8888 --ip=* --FileContentsManager.delete_to_trash=False --ServerApp.terminado_settings='{"shell_command":["/bin/bash"]}' --ServerApp.token=$JUPYTER_PASSWORD --ServerApp.allow_origin=* --ServerApp.preferred_dir=/runpod-volume &> /jupyter.log &
        echo "Jupyter Lab started"
    fi
}

# Run ComfyUI
run_comfy() {
    echo "Running ComfyUI..."
    export PYTHONUNBUFFERED=1
    source /venv/bin/activate
    # rsync -au --remove-source-files /ComfyUI/ /app/ComfyUI/
    # ln -s /comfy-models/* /app/ComfyUI/models/checkpoints/

    cd /app/

    # Check if CUDA is available
    if python -c "import torch; print(torch.cuda.is_available())" | grep True; then
        echo "CUDA is available. Running ComfyUI with GPU support."
        python main.py --gpu-only --disable-metadata --listen --port 3000 &
    else
        echo "WARNING: CUDA unavailable. Running ComfyUI in CPU mode. DO NOT do this in production!"
        python main.py --cpu --disable-metadata --listen --port 3000 &
    fi

    # Capture the PID of the ComfyUI process, and wait to see if it starts
    COMFY_PID=$!
    sleep 3
    if ! kill -0 $COMFY_PID 2> /dev/null; then
        echo "Error: ComfyUI failed to start. See error message below:"
    fi
}

#Cache models in local storage
cache_models(){
if [ -d "/runpod-volume/models/" ] && [[ "1" -ne $NO_MODEL_CACHE ]] ; then
    echo "Copying models to tmpfs..."
    mkdir /usr/share/memory
    cp -r /runpod-volume/models /usr/share/memory
    echo "Copying config for local model cache"
    cp /usr/share/configs/extra_model_paths_cache.yaml /app/ComfyUI/extra_model_paths.yaml &
else
    echo "Copying config for loading models from NFS"
    cp /usr/share/configs/extra_model_paths_nfs.yaml /app/ComfyUI/extra_model_paths.yaml
fi
}

#Setting up catfs cache
catfs_mount_cache(){
    mkdir /var/cache/models
    /usr/local/bin/catfs /runpod-volume/models /var/cache/models /usr/share/memory &
}

# ---------------------------------------------------------------------------- #
#                               Main Program                                   #
# ---------------------------------------------------------------------------- #

start_nginx
#Commented out in favor of catfs
#Uncomented back as catfs doesn't worked on runpod.
cache_models
#catfs_mount_cache
create_sym_links
run_comfy

echo "Pod Started"

setup_ssh
start_jupyter
export_env_vars

echo "Start script finished; pod is ready to use."

sleep infinity  # This will keep the container running
