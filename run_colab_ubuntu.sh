#!/bin/bash

echo "Make sure you set up your branch name in colab_runner.ipynb"
python3 -m venv .venv && source .venv/bin/activate && python3 -m pip install pydrive && sudo python3 connect_to_colab.py && deactivate