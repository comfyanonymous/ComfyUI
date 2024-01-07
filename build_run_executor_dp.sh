#!/bin/bash

service="aiyo_executor_main"
pids=$(ps -ef | grep "$service" | grep -v grep | awk '{print $2}')

if [ -n "$pids" ]; then
    for pid in $pids; do
        kill -9 "$pid"
    done
fi


VENV_DIR=venv
REQUIREMENTS=requirements.txt

echo Setting up virtual environment...
if [ ! -d "$VENV_DIR" ]; then
python3 -m venv "$VENV_DIR"
fi
. "$VENV_DIR/bin/activate"

echo Installing requirements...
pip3 install -r "$REQUIREMENTS"

python aiyo_project_init.py

echo Setup complete.

echo Running the application...
. "$VENV_DIR/bin/activate"
nohup python aiyo_executor_main.py --config "debug_dp" > nohup.out 2>&1 &