#!/bin/bash

service="aiyo_server_main"
pids=$(ps -ef | grep "$service" | grep -v grep | awk '{print $2}')

if [ -n "$pids" ]; then
    for pid in $pids; do
        kill -9 "$pid"
    done
fi


VENV_DIR=venv
REQUIREMENTS=requirements_api_server.txt

echo Setting up virtual environment...
if [ ! -d "$VENV_DIR" ]; then
python3 -m venv "$VENV_DIR"
fi
. "$VENV_DIR/bin/activate"

echo Installing requirements...
pip3 install -r "$REQUIREMENTS"

echo Setup complete.

echo Running the application...
. "$VENV_DIR/bin/activate"
nohup python aiyo_api_server_main.py --config "debug_dp" --listen "0.0.0.0" > nohup.out 2>&1 &