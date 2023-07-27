#!/bin/bash

# Run the first Python script
python3 main.py &

# Wait for 20 seconds
sleep 20

# Run the second Python script
python3 comfy_runpod.py
