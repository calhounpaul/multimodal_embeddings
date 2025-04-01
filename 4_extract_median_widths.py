#!/usr/bin/env python3
"""
Median Width Extraction Script

This script analyzes the combined JSON results from the previous script to:
1. Extract the median width of all plain_text boxes in each file
2. Create visualizations with a horizontal red line representing the median width
3. Save each image's median width information to an individual JSON file

Requirements:
- OpenCV (cv2)
- NumPy
- tqdm (for progress bars)

Usage:
python extract_median_widths_individual.py --input_folder [path/to/combined/json] --output_folder [path]
"""

import os
import json
import argparse
import logging
import numpy as np
import cv2
from tqdm import tqdm
import glob
from collections import defaultdict

# Setup logging
def setup_logger():
    logger = logging.getLogger("MedianWidthExtractor")
    logger.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(console_handler)
    
    return logger

logger = setup_logger()

def bin_widths(widths, min_margin_percent, page_width):
    """
    Group widths into bins with a minimum margin.
    
    Args:
        widths (list): List of box widths
        min_margin_percent (float): Minimum margin as percentage of page width
        page_width (int): Width of the page
        
    Returns:
        dict: Dictionary mapping binned widths to counts
    """
    if not widths:
        return {}
    
    min_margin = page_width * (min_margin_percent / 100)
    binned_widths = defaultdict(int)
    
    for width in widths:
        # Find the appropriate bin
        assigned = False
        for bin_width in sorted(binned_widths.keys()):
            if abs(width - bin_width) <= min_margin:
                binned_widths[bin_width] += 1
                assigned = True
                break
        
        # If not assigned to any existing bin, create a new bin
        if not assigned:
            binned_widths[width] = 1
    
    return binned_widths

def calculate_median_width(widths_dict):
    """
    Calculate the median width from a dictionary of binned widths.
    
    Args:
        widths_dict (dict): Dictionary mapping widths to counts
        
    Returns:
        float: Median width
    """
    if not widths_dict:
        return 0
    
    # Create a flattened list of all widths (repeating each width by its count)
    all_widths = []
    for width, count in widths_dict.items():
        all_widths.extend([width] * count)
    
    # Return the median
    return np.median(all_widths)

def process_json_file(json_path, min_margin_percent=0.2):
    """
    Process a single JSON file to extract the median width of plain_text boxes.
    
    Args:
        json_path (str): Path to the JSON file
        min_margin_percent (float): Minimum margin as percentage of page width
        
    Returns:
        tuple: (image_path, median_width, page_width, page_height)
    """
    try:
        # Load the JSON file
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Get image path
        image_path = data.get('image_path', '')
        
        # Get image dimensions
        image_size = data.get('image_size', {})
        page_width = image_size.get('width', 0)
        page_height = image_size.get('height', 0)
        
        # Extract widths of plain_text boxes
        plain_text_widths = []
        
        # Get boxes and class names
        boxes = data.get('boxes', [])
        class_names = data.get('class_names', [])
        
        # Find all plain_text boxes
        for i, class_name in enumerate(class_names):
            if class_name == 'plain_text':
                if i < len(boxes):
                    box = boxes[i]
                    # Calculate width
                    width = box[2] - box[0]
                    plain_text_widths.append(width)
        
        # Bin the widths and calculate median
        binned_widths = bin_widths(plain_text_widths, min_margin_percent, page_width)
        median_width = calculate_median_width(binned_widths)
        
        return image_path, median_width, page_width, page_height
    
    except Exception as e:
        logger.error(f"Error processing {json_path}: {str(e)}")
        return None, 0, 0, 0

def create_line_visualization(image_path, median_width, output_path):
    """
    Create a visualization with a horizontal red line representing the median width.
    
    Args:
        image_path (str): Path to the original image
        median_width (float): Median width to visualize
        output_path (str): Path to save the visualization
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            return False
        
        # Get image dimensions
        height, width = image.shape[:2]
        
        # Calculate line position and dimensions
        line_y = int(height * 0.75)  # Position line in the lower middle (3/4 down)
        line_x_start = int((width - median_width) / 2)
        line_x_end = int(line_x_start + median_width)
        
        # Draw the red line
        line_thickness = max(3, int(height / 200))  # Scale line thickness with image height
        cv2.line(image, (line_x_start, line_y), (line_x_end, line_y), (0, 0, 255), line_thickness)
        
        # Add a label for the median width
        label = f"Median width: {median_width:.1f} px"
        font_scale = max(0.7, height / 2000)
        label_thickness = max(1, int(height / 500))
        
        # Calculate text size to position it properly
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, label_thickness
        )
        text_x = int((width - text_width) / 2)
        text_y = line_y - 20
        
        # Add a background for the text
        cv2.rectangle(
            image, 
            (text_x - 10, text_y - text_height - 10),
            (text_x + text_width + 10, text_y + 10),
            (255, 255, 255),
            -1
        )
        
        # Draw the text
        cv2.putText(
            image, 
            label, 
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 255),
            label_thickness
        )
        
        # Save the visualization
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, image)
        logger.info(f"Saved visualization to {output_path}")
        
        return True
    
    except Exception as e:
        logger.error(f"Error creating visualization for {image_path}: {str(e)}")
        return False

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Extract median width of plain_text boxes')
    parser.add_argument('--input_folder', required=True, help='Folder containing combined JSON results')
    parser.add_argument('--output_folder', required=True, help='Folder to save results')
    parser.add_argument('--min_margin_percent', type=float, default=0.2, 
                        help='Minimum margin as percentage of page width (default: 0.2%%)')
    
    args = parser.parse_args()
    
    # Create output folders
    output_json_folder = os.path.join(args.output_folder, 'json')
    output_viz_folder = os.path.join(args.output_folder, 'visualizations')
    os.makedirs(args.output_folder, exist_ok=True)
    os.makedirs(output_json_folder, exist_ok=True)
    os.makedirs(output_viz_folder, exist_ok=True)
    
    # Find all JSON files in the input folder
    json_folder = args.input_folder
    if not os.path.isdir(json_folder):
        json_folder = os.path.join(args.input_folder, 'json')
    
    if not os.path.exists(json_folder):
        logger.error(f"JSON folder not found: {json_folder}")
        return
    
    json_pattern = os.path.join(json_folder, '*.json')
    json_files = glob.glob(json_pattern)
    
    if not json_files:
        logger.error(f"No JSON files found in {json_folder}")
        return
    
    logger.info(f"Found {len(json_files)} JSON files")
    
    # Process each JSON file
    for json_path in tqdm(json_files, desc="Extracting median widths"):
        base_name = os.path.splitext(os.path.basename(json_path))[0]
        
        # Process the JSON file
        image_path, median_width, page_width, page_height = process_json_file(
            json_path=json_path,
            min_margin_percent=args.min_margin_percent
        )
        
        if image_path and os.path.exists(image_path):
            # Create individual result dictionary
            result = {
                'image_path': image_path,
                'median_width': median_width,
                'page_width': page_width,
                'page_height': page_height,
                'width_ratio': median_width / page_width if page_width > 0 else 0
            }
            
            # Save to individual JSON file
            output_json_path = os.path.join(output_json_folder, f"{base_name}_median_width.json")
            with open(output_json_path, 'w') as f:
                json.dump(result, f, indent=2)
            
            # Create visualization
            output_viz_path = os.path.join(output_viz_folder, f"{base_name}_median_width.jpg")
            create_line_visualization(image_path, median_width, output_viz_path)
    
    logger.info(f"Processing complete. Individual results saved to {output_json_folder}")
    logger.info(f"Visualizations saved to {output_viz_folder}")

if __name__ == "__main__":
    main()