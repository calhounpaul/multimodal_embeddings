#!/bin/bash

FLAGFILE_INSTALL="venv/.completed_installation"
FLAGFILE_TESSERACT="venv/.tesseract_installed"

# Check if tesseract is installed
check_tesseract() {
    if command -v tesseract > /dev/null 2>&1; then
        echo "Tesseract OCR is already installed."
        return 0
    else
        return 1
    fi
}

# Install tesseract if not available
install_tesseract() {
    echo "Installing Tesseract OCR..."
    sudo apt-get update
    sudo apt-get install -y tesseract-ocr tesseract-ocr-eng
    
    if check_tesseract; then
        echo "Tesseract OCR installed successfully."
        touch $FLAGFILE_TESSERACT
        return 0
    else
        echo "Failed to install Tesseract OCR. Please install it manually."
        return 1
    fi
}

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    mkdir -p venv
fi

# Check and install tesseract if needed
if [ ! -f $FLAGFILE_TESSERACT ]; then
    if ! check_tesseract; then
        echo "Tesseract OCR is not installed."
        install_tesseract
    else
        touch $FLAGFILE_TESSERACT
    fi
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