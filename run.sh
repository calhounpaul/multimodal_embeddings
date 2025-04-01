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

venv/bin/python3 0_orientation.py newspaper_images 0_oriented_images

venv/bin/python3 1_doclayout_bboxes.py --input_folder 0_oriented_images --output_folder 1_doclayout_parsed

venv/bin/python3 2_edge_box_filter.py --input_folder 1_doclayout_parsed --output_folder 2_edge_box_filtered

venv/bin/python3 3_combine_grids.py --input_folder 2_edge_box_filtered --output_folder 3_combined_bboxes

venv/bin/python3 4_extract_median_widths.py --input_folder 3_combined_bboxes/json --output_folder 4_medians_extracted

venv/bin/python3 5_detect_column_centers.py --input_folder 3_combined_bboxes/json --median_folder 4_medians_extracted/json --output_folder 5_column_detection