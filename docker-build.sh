#!/bin/bash

# Check if Docker and Docker Compose are installed
if ! command -v docker &> /dev/null || ! command -v docker-compose &> /dev/null
then
    echo "Docker or Docker Compose not found. Please install them before proceeding."
    exit 1
fi

# Build and start the container
docker-compose up --build -d

# Wait for the container logs to indicate it's ready (looking for the custom message)
container_name="comfyui-v1"
while ! docker logs "$container_name" 2>&1 | grep -q "Server started and ready to accept requests"; do
    echo "Waiting for the container to be fully started..."
    sleep 1
done

# Copy the 'venv' directory from the container to the host
docker cp "$container_name:/app/venv" ./data/venv

# Optional: Stop the container after everything is set up
docker-compose down

echo "The container is set up, and the virtual environment is ready at ./venv."
