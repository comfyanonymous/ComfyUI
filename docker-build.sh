#!/bin/bash

# Check if Docker and Docker Compose are installed
if ! command -v docker &> /dev/null || ! command -v docker-compose &> /dev/null
then
    echo "Docker or Docker Compose not found. Please install them before proceeding."
    exit 1
fi

# Step 1: Build and start the container without mounting the venv volume
echo "Building and starting the container to initialize the virtual environment..."
docker-compose up --build -d

# Wait for the container logs to indicate it's ready (looking for the custom message)
container_name="comfyui-red-docker"
while ! docker logs "$container_name" 2>&1 | grep -q "Server started and ready to accept requests"; do
    echo "Waiting for the container to be fully started..."
    sleep 20
done

# Step 2: Copy the 'venv' directory from the container to the host
echo "Copying the virtual environment from the container to the host..."
docker cp "$container_name:/app/venv" ./venv

# Step 3: Stop the container
echo "Stopping the container..."
docker-compose down

# Step 4: Update the Docker Compose file to mount the venv volume
echo "Updating Docker Compose file to mount the virtual environment..."
sed -i '/# Mount the venv directory for persistence/a \ \ \ \ \ \ - ./venv:/app/venv' docker-compose.yml

# Step 5: Restart the container with the venv volume mounted
echo "Restarting the container with the virtual environment mounted..."
docker-compose up -d

echo "Setup complete! The container is running with the virtual environment persisted at ./venv."