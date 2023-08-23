#!/bin/bash
cd ./custom_nodes/comfy_controlnet_preprocessors && python3.9 install.py && cd ../ &


# Run the first Python script
python3.9 main.py &

# Wait for 20 seconds
sleep 20

# Run the second Python script
python3.9 comfy_runpod.py
