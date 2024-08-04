#!/bin/bash

echo "Make sure you set up your branch name in colab_runner.ipynb"
if ! [ -d "./.venv" ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate && python3 -m pip install -r requirements.txt && python3 main.py --dont-print-server && deactivate