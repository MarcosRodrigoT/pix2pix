#!/bin/bash

# Run the first Python script
/home/mrt/Projects/pix2pix/venv/bin/python /home/mrt/Projects/pix2pix/main.py

# Check if the first Python script succeeded
if [ $? -eq 0 ]; then
    echo "Backstories generated successfully. Proceeding with the TTS model and SCP to Hiperion."

    # Run the second Python script
    /home/mrt/Projects/coqui-ai-TTS/venv/bin/python /home/mrt/Projects/coqui-ai-TTS/GTI/coffee_backstories_tts.py

    # Check if the second Python script succeeded
    if [ $? -eq 0 ]; then
        echo "TTS ran successfully. Proceeding with SCP to Hiperion."

        # Copy the directory to the remote machine
        scp -r /home/mrt/Projects/pix2pix/backstories hiperion:/home/mrt/Projects/pix2pix/

        if [ $? -eq 0 ]; then
            echo "Directory copied successfully."
        else
            echo "Error: Failed to copy the directory."
        fi
    else
        echo "Error: Second Python script failed. Skipping SCP."
    fi
else
    echo "Error: First Python script failed. Skipping subsequent steps."
fi
