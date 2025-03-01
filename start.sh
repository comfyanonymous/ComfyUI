#!/bin/bash

# Echo to indicate start of the program
echo "STARTING PROGRAM..."

# Activate the virtual environment
source venv/bin/activate

# Echo to indicate the start of the Python script
echo "*** BEGIN PROGRAM ***"

# Run the Python script
python main.py # main.py

# Echo to indicate the end of the Python script
echo "*** END PROGRAM ***"

# Deactivate the virtual environment
deactivate

# Echo to indicate program completion
echo "PROGRAM EXECUTION COMPLETE."
