#!/bin/bash

FLAGFILE_INSTALL="venv/.completed_installation"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Install requirements if not already installed
if [ ! -f $FLAGFILE_INSTALL ]; then
    echo "Installing dependencies..."
    venv/bin/python3 -m pip install --upgrade pip
    venv/bin/python3 -m pip install -U -r requirements.txt
    touch $FLAGFILE_INSTALL
fi

# Run the main script
echo "Running newspaper image analysis..."
venv/bin/python3 complete_workflow.py "$@"