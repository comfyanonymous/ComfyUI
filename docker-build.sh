#!/bin/bash

# Check if Docker and Docker Compose are installed
if ! command -v docker &> /dev/null || ! command -v docker-compose &> /dev/null
then
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

# Step 2: Start the container without mounting the venv volume
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
    sleep 20
done

# Stop streaming logs (kill the background process)
kill $LOGS_PID
echo "Container is fully started."

# Step 4: Copy the 'venv' directory from the container to the host
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

# Step 4: Copy the 'ComfyUI-Manager' directory from the container to the host
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

# Step 6: Update the Docker Compose file to mount the venv volume
echo "Updating Docker Compose file to mount the virtual environment..."
sed -i '/# Mount the venv directory for persistence/a \ \ \ \ \ \ - ./venv:/app/venv' docker-compose.yml
if [ $? -eq 0 ]; then
    echo "Docker Compose file updated to include venv."
else
    echo "Failed to update Docker Compose file. Exiting."
    exit 1
fi

# Step 6: Update the Docker Compose file to mount the venv volume
echo "Updating Docker Compose file to mount the custom_nodes..."
sed -i '/# Mount the custom nodes directory directly inside/a \ \ \ \ \ \ - ./custom_nodes:/app/comfyui/custom_nodes' docker-compose.yml
if [ $? -eq 0 ]; then
    echo "Docker Compose file updated to include custom_nodes."
else
    echo "Failed to update Docker Compose file. Exiting."
    exit 1
fi

echo "Setup complete! you can use "docker-compose up" to start the container."