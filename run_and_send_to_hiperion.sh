#!/bin/bash

# Run the Python script
/home/mrt/Projects/pix2pix/venv/bin/python /home/mrt/Projects/pix2pix/main.py

# Check if the Python script succeeded
if [ $? -eq 0 ]; then
    echo "Python script executed successfully. Proceeding with SCP."

    # Copy the directory to the remote machine
    scp -r /home/mrt/Projects/pix2pix/backstories hiperion:/home/mrt/Projects/pix2pix/
    
    if [ $? -eq 0 ]; then
        echo "Directory copied successfully."
    else
        echo "Error: Failed to copy the directory."
    fi
else
    echo "Error: Python script failed. Skipping SCP."
fi

