#!/bin/bash

# Start time
start_time=$(date +%s)

# Run Docker Compose
echo "Starting Docker Compose..."
docker-compose up -d

# Check if Docker Compose ran successfully
if [ $? -eq 0 ]; then
  echo "Docker Compose ran successfully."
else
  echo "Docker Compose failed to start."
  exit 1
fi

# End time
end_time=$(date +%s)

# Calculate time taken
time_taken=$((end_time - start_time))

# Print time taken in seconds
echo "Time taken: ${time_taken} seconds"
