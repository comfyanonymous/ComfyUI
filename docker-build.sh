#!/bin/bash

# Backup the original Dockerfile
BACKUP_FILE="Dockerfile.bak"
cp Dockerfile "$BACKUP_FILE"

# Function to restore the Dockerfile
restore_dockerfile() {
    echo "Restoring Dockerfile to its original state..."
    mv "$BACKUP_FILE" Dockerfile
    echo "Dockerfile restored."
}

# Set up trap to restore Dockerfile on script exit (success or failure)
trap restore_dockerfile EXIT

# Function to display version information
display_version_info() {
    echo "==========================================================="
    echo "Stable Version:"
    echo "  - This is the latest stable version released by PyTorch."
    echo "  - It is thoroughly tested and recommended for deployment."
    echo "  - Pros: Reliable, well-tested, fewer bugs."
    echo "  - Cons: May not include the latest features or optimizations."
    echo ""
    echo "Latest Version:"
    echo "  - This is the latest development version of PyTorch."
    echo "  - It includes the newest features and optimizations but may have bugs."
    echo "  - Pros: Cutting-edge features, performance improvements."
    echo "  - Cons: Less stable, potential for encountering bugs."
    echo "==========================================================="
}

# Function to ask user for GPU type
ask_gpu_type() {
    echo "What GPU do you have?"
    select gpu in "NVIDIA" "AMD"; do
        case $gpu in
            NVIDIA)
                echo "You selected NVIDIA."
                break
                ;;
            AMD)
                echo "You selected AMD."
                break
                ;;
            *)
                echo "Invalid option. Please choose 1 or 2."
                ;;
        esac
    done
}

# Function to ask user for version preference
ask_version() {
    echo "Which version would you like to use?"
    select version in "Stable" "Latest"; do
        case $version in
            Stable)
                echo "You selected Stable."
                break
                ;;
            Latest)
                echo "You selected Latest."
                break
                ;;
            *)
                echo "Invalid option. Please choose 1 or 2."
                ;;
        esac
    done
}

# Display version information
display_version_info

# Ask user for GPU type and version
ask_gpu_type
ask_version

# Set base image and PyTorch installation command based on user input
if [[ "$gpu" == "NVIDIA" ]]; then
    if [[ "$version" == "Stable" ]]; then
        BASE_IMAGE="nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04"
        TORCH_INSTALL="pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu126"
        # Uncomment the stable NVIDIA FROM line
        sed -i '/# FROM nvidia\/cuda:12.6.3-cudnn-runtime-ubuntu24.04 AS base/s/^# //' Dockerfile
        # Uncomment the stable NVIDIA PyTorch installation line
        sed -i '/# RUN \/app\/venv\/bin\/pip install torch torchvision torchaudio --extra-index-url https:\/\/download.pytorch.org\/whl\/cu126/s/^# //' Dockerfile
    else
        BASE_IMAGE="nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04"
        TORCH_INSTALL="pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128"
        # Uncomment the latest NVIDIA FROM line
        sed -i '/# FROM nvidia\/cuda:12.8.1-cudnn-runtime-ubuntu24.04 AS base/s/^# //' Dockerfile
        # Uncomment the latest NVIDIA PyTorch installation line
        sed -i '/# RUN \/app\/venv\/bin\/pip install --pre torch torchvision torchaudio --index-url https:\/\/download.pytorch.org\/whl\/nightly\/cu128/s/^# //' Dockerfile
    fi
else
    if [[ "$version" == "Stable" ]]; then
        BASE_IMAGE="rocm/dev-ubuntu-24.04:6.2.4-complete"
        TORCH_INSTALL="pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2.4"
        # Uncomment the stable AMD FROM line
        sed -i '/# FROM rocm\/dev-ubuntu-24.04:6.2.4-complete AS base/s/^# //' Dockerfile
        # Uncomment the stable AMD PyTorch installation line
        sed -i '/# RUN \/app\/venv\/bin\/pip install torch torchvision torchaudio --index-url https:\/\/download.pytorch.org\/whl\/rocm6.2.4/s/^# //' Dockerfile
    else
        BASE_IMAGE="rocm/dev-ubuntu-24.04:6.3.4-complete"
        TORCH_INSTALL="pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm6.3"
        # Uncomment the latest AMD FROM line
        sed -i '/# FROM rocm\/dev-ubuntu-24.04:6.3.4-complete AS base/s/^# //' Dockerfile
        # Uncomment the latest AMD PyTorch installation line
        sed -i '/# RUN \/app\/venv\/bin\/pip3 install --pre torch torchvision torchaudio --index-url https:\/\/download.pytorch.org\/whl\/nightly\/rocm6.3/s/^# //' Dockerfile
    fi
fi

# Check if Docker and Docker Compose are installed
if ! command -v docker &> /dev/null || ! command -v docker-compose &> /dev/null; then
    echo "Docker or Docker Compose not found. Please install them before proceeding."
    exit 1
fi

# Step 1: Build the container without using cache
echo "Building the container to initialize the virtual environment..."
COMPOSE_BAKE=true docker-compose build --no-cache
if [ $? -eq 0 ]; then
    echo "Build completed successfully."
else
    echo "Build failed. Exiting."
    exit 1
fi

# Step 2: Start the container without mounting the volumes (venv, custom_nodes)
echo "Starting the container..."
COMPOSE_BAKE=true docker-compose up -d
if [ $? -eq 0 ]; then
    echo "Container started successfully."
else
    echo "Failed to start the container. Exiting."
    exit 1
fi

# Step 3: Stream Docker logs to the terminal
container_name="comfyui-red-container"
echo "Streaming Docker logs for container: $container_name..."
docker logs -f "$container_name" &
LOGS_PID=$!  # Save the PID of the background process

# Wait for the container logs to indicate it's ready (looking for the custom message)
echo "Waiting for the container to be fully started..."
while ! docker logs "$container_name" 2>&1 | grep -q "To see the GUI go to: http://0.0.0.0:8188"; do
    sleep 10
done

# Stop streaming logs (kill the background process)
kill $LOGS_PID
echo "Container is fully started."

# Step 4.1: Copy the 'venv' directory from the container to the host
echo "Checking if /app/venv exists in the container..."
if docker exec "$container_name" ls /app/venv; then
    echo "Copying the virtual environment from the container to the host..."
    if ! docker cp "$container_name:/app/venv" ./venv; then
        echo "Failed to copy the virtual environment. Exiting."
        exit 1
    else
        echo "Virtual environment copied successfully."
    fi
else
    echo "/app/venv does not exist in the container. Exiting."
    exit 1
fi

# Step 4.2: Copy the 'ComfyUI-Manager' directory from the container to the host
echo "Checking if /app/comfyui/custom_nodes/ComfyUI-Manager exists in the container..."
if docker exec "$container_name" ls /app/comfyui/custom_nodes/ComfyUI-Manager; then
    echo "Copying the ComfyUI-Manager from the container to the host..."
    if ! docker cp "$container_name:/app/comfyui/custom_nodes/ComfyUI-Manager" ./custom_nodes/ComfyUI-Manager; then
        echo "Failed to copy the ComfyUI-Manager. Exiting."
        exit 1
    else
        echo "ComfyUI-Manager copied successfully."
    fi
else
    echo "/app/comfyui/custom_nodes/ComfyUI-Manager does not exist in the container. Exiting."
    exit 1
fi

# Step 5: Stop the container
echo "Stopping the container..."
docker-compose down
if [ $? -eq 0 ]; then
    echo "Container stopped successfully."
else
    echo "Failed to stop the container. Exiting."
    exit 1
fi

# Step 6.1: Update the Docker Compose file to mount the venv volume
echo "Updating Docker Compose file to mount the virtual environment..."
sed -i '/# Mount the venv directory for persistence/a \ \ \ \ \ \ - ./venv:/app/venv' docker-compose.yml
if [ $? -eq 0 ]; then
    echo "Docker Compose file updated to include venv."
else
    echo "Failed to update Docker Compose file. Exiting."
    exit 1
fi

# Step 6.2: Update the Docker Compose file to mount the custom_nodes volume
echo "Updating Docker Compose file to mount the custom_nodes..."
sed -i '/# Mount the custom nodes directory directly inside/a \ \ \ \ \ \ - ./custom_nodes:/app/comfyui/custom_nodes' docker-compose.yml
if [ $? -eq 0 ]; then
    echo "Docker Compose file updated to include custom_nodes."
else
    echo "Failed to update Docker Compose file. Exiting."
    exit 1
fi

echo "======================================== SETUP COMPLETE ========================================"
echo "use 'docker-compose up' to start the container and 'docker-compose down' to stop the container."
echo "================================================================================================"