#!/usr/bin/env python3
"""
Column Center Detection Script

This script analyzes document layout bounding boxes and median width data to:
1. Identify the centers of text columns on each page
2. Create visualizations with vertical lines showing column centers
3. Save the column center information to JSON files

Requirements:
- OpenCV (cv2)
- NumPy
- SciPy (for peak finding)
- tqdm (for progress bars)

Usage:
python detect_column_centers.py --input_folder [path/to/layout/json] --median_folder [path/to/median/json] --output_folder [path]
"""

import os
import json
import argparse
import logging
import numpy as np
import cv2
from tqdm import tqdm
import glob
from scipy.signal import find_peaks
from scipy.signal.windows import gaussian

# Custom JSON encoder for NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# Custom NumPy JSON serializer
class NumpyJSONEncoder(json.JSONEncoder):
    """JSON encoder that handles NumPy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyJSONEncoder, self).default(obj)

# Setup logging
def setup_logger():
    logger = logging.getLogger("ColumnCenterDetector")
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

def load_json_file(file_path):
    """
    Load a JSON file.
    
    Args:
        file_path (str): Path to the JSON file
        
    Returns:
        dict: Loaded JSON data, or None if loading fails
    """
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading JSON file {file_path}: {str(e)}")
        return None

def find_column_centers(boxes, class_names, scores, page_width, page_height, median_width, min_confidence=0.3, verbose=False):
    """
    Find the centers of text columns on a page.
    
    Args:
        boxes (list): List of bounding boxes [x1, y1, x2, y2]
        class_names (list): List of class names for each box
        scores (list): List of confidence scores for each box
        page_width (int): Width of the page
        page_height (int): Height of the page
        median_width (float): Median width of text boxes
        min_confidence (float): Minimum confidence score for boxes
        
    Returns:
        list: List of column center x-coordinates
        list: List of column width estimates
    """
    # Filter boxes to only include plain_text and title with sufficient confidence
    filtered_boxes = []
    for i, (box, class_name, score) in enumerate(zip(boxes, class_names, scores)):
        if (class_name in ["plain_text", "title"]) and score >= min_confidence:
            filtered_boxes.append(box)
    
    if not filtered_boxes:
        logger.warning("No text boxes found with sufficient confidence")
        return [], []
    
    # Create a density map for horizontal text positions
    # Use higher resolution for more accurate peaks
    resolution = max(1, int(page_width / 1000))  # Each bin represents 'resolution' pixels
    num_bins = page_width // resolution + 1
    density = np.zeros(num_bins)
    
    # Generate the density map based on box positions
    for box in filtered_boxes:
        x1, y1, x2, y2 = map(int, box)
        width = x2 - x1
        
        # Only consider boxes that are reasonable for text columns
        # Allow some flexibility: 0.33 to 2.0 times the median width
        if 0.33 * median_width <= width <= 2.0 * median_width:
            # Add weight to the density map for the entire box width
            left_bin = max(0, x1 // resolution)
            right_bin = min(num_bins - 1, x2 // resolution)
            
            # Add more weight to the center of the box
            center_bin = (x1 + x2) // (2 * resolution)
            
            # Add density across the box width, with higher weight in the center
            for bin_idx in range(left_bin, right_bin + 1):
                # Distance from center (normalized)
                dist_from_center = abs(bin_idx - center_bin) / ((right_bin - left_bin) / 2 + 1e-6)
                weight = 1.0 - 0.5 * min(1.0, dist_from_center)  # Higher weight near center
                density[bin_idx] += weight
    
    # Apply Gaussian smoothing to the density map
    window_size = max(5, int(median_width / (4 * resolution)))
    if window_size % 2 == 0:
        window_size += 1  # Ensure odd window size for Gaussian
    
    sigma = window_size / 6.0  # Standard deviation for Gaussian
    gaussian_window = gaussian(window_size, std=sigma)
    gaussian_window = gaussian_window / gaussian_window.sum()  # Normalize
    
    # Apply smoothing
    smoothed_density = np.convolve(density, gaussian_window, mode='same')
    
    # Define minimum peak height as a fraction of the maximum density
    min_peak_height = max(smoothed_density) * 0.2
    
    # Find peaks in the smoothed density map
    # Distance between peaks should be related to median width
    min_distance = max(1, int(median_width / (1.5 * resolution)))
    peaks, properties = find_peaks(
        smoothed_density, 
        height=min_peak_height, 
        distance=min_distance,
        prominence=max(smoothed_density) * 0.05  # Require a minimum prominence
    )
    
    if len(peaks) == 0:
        logger.warning("No peaks found in density map")
        return [], []
    
    # Convert peak indices back to page coordinates
    column_centers = [peak * resolution for peak in peaks]
    
    # Estimate column widths based on the peaks
    column_widths = []
    for i, peak in enumerate(peaks):
        # Find the left boundary (local minimum or half distance to previous peak)
        left_idx = peak
        if i > 0:
            prev_peak = peaks[i-1]
            for j in range(peak-1, prev_peak, -1):
                if j < 0 or j >= len(smoothed_density):
                    continue
                if smoothed_density[j] < smoothed_density[left_idx]:
                    left_idx = j
                if smoothed_density[j] < min_peak_height * 0.1:
                    break
            
            # If no clear minimum, use half distance to previous peak
            if left_idx == peak:
                left_idx = (peak + prev_peak) // 2
        
        # Find the right boundary (local minimum or half distance to next peak)
        right_idx = peak
        if i < len(peaks) - 1:
            next_peak = peaks[i+1]
            for j in range(peak+1, next_peak):
                if j < 0 or j >= len(smoothed_density):
                    continue
                if smoothed_density[j] < smoothed_density[right_idx]:
                    right_idx = j
                if smoothed_density[j] < min_peak_height * 0.1:
                    break
            
            # If no clear minimum, use half distance to next peak
            if right_idx == peak:
                right_idx = (peak + next_peak) // 2
        
        # Calculate column width
        width = (right_idx - left_idx) * resolution
        
        # Adjust width based on median if too narrow or too wide
        if width < 0.5 * median_width:
            width = median_width
        elif width > 2.5 * median_width:
            width = 2.0 * median_width
        
        column_widths.append(width)
    
    return column_centers, column_widths

def create_column_visualization(image_path, column_centers, column_widths, median_width, output_path, debug=False):
    """
    Create a visualization with vertical lines representing column centers.
    
    Args:
        image_path (str): Path to the original image
        column_centers (list): List of column center x-coordinates
        column_widths (list): List of column width estimates
        median_width (float): Median width of text boxes
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
        
        # Create output image
        # Copy the original image at reduced opacity
        # Create a copy of the original image
        output_image = image.copy()
        
        # Create a semi-transparent overlay
        overlay = image.copy()
        
        # If debug mode, use a darker background to better see the overlay
        if debug:
            alpha = 0.15  # More transparent background for original image
        else:
            alpha = 0.7   # Less transparent background
        
        # Blend the original image and a blank overlay
        cv2.addWeighted(image, alpha, np.zeros_like(image), 1 - alpha, 0, output_image)
        
        # Calculate line thickness based on image size
        line_thickness = max(2, int(width / 500))
        
        # Draw column boundaries with transparency
        for i, (center_x, col_width) in enumerate(zip(column_centers, column_widths)):
            center_x = int(center_x)
            
            # Draw column center line
            cv2.line(output_image, (center_x, 0), (center_x, height), (0, 0, 255), line_thickness * 2)
            
            # Draw column width boundaries
            half_width = col_width / 2
            left_bound = max(0, int(center_x - half_width))
            right_bound = min(width, int(center_x + half_width))
            
            # Draw column bounding box
            cv2.rectangle(output_image, (left_bound, 0), (right_bound, height), (0, 255, 0), line_thickness)
            
            # Add column number label at the top
            label = f"Column {i+1}"
            font_scale = max(0.7, height / 2000)
            label_thickness = max(1, int(height / 500))
            
            # Create a background for the text
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, label_thickness
            )
            cv2.rectangle(
                output_image, 
                (center_x - text_width // 2 - 5, 30 - text_height - 5),
                (center_x + text_width // 2 + 5, 30 + 5),
                (255, 255, 255),
                -1
            )
            
            # Draw the text
            cv2.putText(
                output_image,
                label,
                (center_x - text_width // 2, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 255),
                label_thickness
            )
        
        # Add a legend
        legend_text = f"Median Width: {median_width:.1f} px"
        legend_pos = (20, height - 30)
        cv2.putText(
            output_image,
            legend_text,
            legend_pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            max(0.7, height / 2000),
            (255, 255, 255),
            max(1, int(height / 500))
        )
        
        # Save the visualization
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, output_image)
        logger.info(f"Saved visualization to {output_path}")
        
        return True
    
    except Exception as e:
        logger.error(f"Error creating visualization for {image_path}: {str(e)}")
        return False

def process_page(layout_json_path, median_json_path, output_folder, min_confidence=0.3):
    """
    Process a single page to detect column centers.
    
    Args:
        layout_json_path (str): Path to the layout detection JSON file
        median_json_path (str): Path to the median width JSON file
        output_folder (str): Folder to save results
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Load the layout detection JSON
        layout_data = load_json_file(layout_json_path)
        if not layout_data:
            return False
        
        # Load the median width JSON
        median_data = load_json_file(median_json_path)
        if not median_data:
            return False
        
        # Get median width
        median_width = median_data.get("median_width", 0)
        if median_width <= 0:
            logger.error(f"Invalid median width: {median_width}")
            return False
        
        # Get image path and dimensions
        image_path = layout_data.get("image_path", "")
        
        # Some JSON files might have image_size in different formats
        image_size = layout_data.get("image_size", {})
        if isinstance(image_size, dict):
            page_width = image_size.get("width", 0)
            page_height = image_size.get("height", 0)
        elif isinstance(image_size, list) and len(image_size) >= 2:
            page_width = image_size[0]
            page_height = image_size[1]
        else:
            page_width = 0
            page_height = 0
        
        if not image_path or not os.path.exists(image_path) or page_width <= 0 or page_height <= 0:
            logger.error(f"Invalid image information: {image_path}, {page_width}x{page_height}")
            return False
        
        # Get boxes, classes, class names, and scores
        boxes = layout_data.get("boxes", [])
        class_names = layout_data.get("class_names", [])
        scores = layout_data.get("scores", [1.0] * len(boxes))  # Default to 1.0 if not available
        
        # Make sure we have valid page dimensions
        if page_width <= 0 or page_height <= 0:
            if 'image_path' in layout_data and os.path.exists(layout_data['image_path']):
                # Try to get dimensions from the image file
                try:
                    img = cv2.imread(layout_data['image_path'])
                    if img is not None:
                        page_height, page_width = img.shape[:2]
                        logger.info(f"Retrieved dimensions from image: {page_width}x{page_height}")
                except Exception as e:
                    logger.warning(f"Failed to read image dimensions: {str(e)}")
        
        # Find column centers and widths
        column_centers, column_widths = find_column_centers(
            boxes, class_names, scores, page_width, page_height, median_width,
            min_confidence=min_confidence, verbose=(True if logger.level <= logging.INFO else False)
        )
        
        # Convert NumPy types to native Python types
        column_centers = [float(x) for x in column_centers]
        column_widths = [float(x) for x in column_widths]
        
        if not column_centers:
            logger.warning(f"No column centers found for {os.path.basename(layout_json_path)}")
            return False
        
        # Create output folders
        output_json_folder = os.path.join(output_folder, "json")
        output_viz_folder = os.path.join(output_folder, "visualizations")
        os.makedirs(output_json_folder, exist_ok=True)
        os.makedirs(output_viz_folder, exist_ok=True)
        
        # Base filename without extension
        base_name = os.path.splitext(os.path.basename(layout_json_path))[0]
        
        # Create result dictionary
        result = {
            "image_path": image_path,
            "page_width": page_width,
            "page_height": page_height,
            "median_width": median_width,
            "column_centers": column_centers,
            "column_widths": column_widths,
            "num_columns": len(column_centers)
        }
        
        # Save to JSON file
        output_json_path = os.path.join(output_json_folder, f"{base_name}_columns.json")
        with open(output_json_path, 'w') as f:
            json.dump(result, f, indent=2, cls=NumpyJSONEncoder)
        
        # Create and save visualization
        output_viz_path = os.path.join(output_viz_folder, f"{base_name}_columns.jpg")
        create_column_visualization(image_path, column_centers, column_widths, median_width, output_viz_path, debug=False)
        
        # Also create a debug visualization with more transparent background
        output_debug_viz_folder = os.path.join(output_folder, "visualizations_debug")
        os.makedirs(output_debug_viz_folder, exist_ok=True)
        output_debug_viz_path = os.path.join(output_debug_viz_folder, f"{base_name}_columns_debug.jpg")
        create_column_visualization(image_path, column_centers, column_widths, median_width, output_debug_viz_path, debug=True)
        
        return True
    
    except Exception as e:
        logger.error(f"Error processing {os.path.basename(layout_json_path)}: {str(e)}")
        return False

def convert_numpy_types(obj):
    """
    Recursively convert numpy types to native Python types.
    
    Args:
        obj: Object to convert
        
    Returns:
        Object with numpy types converted to native Python types
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

def find_matching_median_json(layout_json_path, median_json_folder):
    """
    Find the matching median width JSON file for a layout JSON file.
    
    Args:
        layout_json_path (str): Path to the layout detection JSON file
        median_json_folder (str): Folder containing median width JSON files
        
    Returns:
        str: Path to the matching median width JSON file, or None if not found
    """
    # Get the base name without extension
    base_name = os.path.splitext(os.path.basename(layout_json_path))[0]
    
    # First try exact match
    exact_match = os.path.join(median_json_folder, f"{base_name}_median_width.json")
    if os.path.exists(exact_match):
        return exact_match
    
    # Try to match based on image name without the grid part
    # Example: if layout_json_path is "image_grid_2x2.json", try to find "image_median_width.json"
    if "_grid_" in base_name:
        non_grid_name = base_name.split("_grid_")[0]
        non_grid_match = os.path.join(median_json_folder, f"{non_grid_name}_median_width.json")
        if os.path.exists(non_grid_match):
            return non_grid_match
    
    # If still not found, look for any file that starts with the same prefix before '_grid_'
    if "_grid_" in base_name:
        prefix = base_name.split("_grid_")[0]
        for file_name in os.listdir(median_json_folder):
            if file_name.endswith("_median_width.json") and file_name.startswith(f"{prefix}_"):
                return os.path.join(median_json_folder, file_name)
    
    # If we still haven't found a match, look for any file that contains the page identifier
    # Common patterns are "page_XXXX" or "pageXXXX" in filenames
    parts = base_name.split("_")
    for part in parts:
        if part.lower().startswith("page") or (len(part) >= 4 and part.isdigit()):
            for file_name in os.listdir(median_json_folder):
                if part in file_name and file_name.endswith("_median_width.json"):
                    return os.path.join(median_json_folder, file_name)
    
    # Last resort: If the filename contains a recognizable pattern like 'page_0001', 
    # try to extract it and look for a matching median file
    import re
    page_pattern = re.compile(r'(page[_-]?\d+)', re.IGNORECASE)
    match = page_pattern.search(base_name)
    if match:
        page_id = match.group(1)
        for file_name in os.listdir(median_json_folder):
            if page_id in file_name and file_name.endswith("_median_width.json"):
                return os.path.join(median_json_folder, file_name)
    
    # If all else fails, check if there's only one median file in the folder
    median_files = [f for f in os.listdir(median_json_folder) if f.endswith("_median_width.json")]
    if len(median_files) == 1:
        return os.path.join(median_json_folder, median_files[0])
    
    return None

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Detect column centers in document pages')
    parser.add_argument('--input_folder', required=True, help='Folder containing layout detection JSON files')
    parser.add_argument('--median_folder', required=True, help='Folder containing median width JSON files')
    parser.add_argument('--output_folder', required=True, help='Folder to save results')
    parser.add_argument('--min_confidence', type=float, default=0.3, help='Minimum confidence score for bounding boxes')
    parser.add_argument('--verbose', action='store_true', help='Print more detailed information')
    
    args = parser.parse_args()
    
    # Create output folders
    os.makedirs(args.output_folder, exist_ok=True)
    
    # Find all layout JSON files
    json_pattern = os.path.join(args.input_folder, "*.json")
    layout_json_files = glob.glob(json_pattern)
    
    if not layout_json_files:
        logger.error(f"No JSON files found in {args.input_folder}")
        return
    
    logger.info(f"Found {len(layout_json_files)} layout JSON files")
    
    # Process each layout JSON file
    success_count = 0
    failure_count = 0
    
    for layout_json_path in tqdm(layout_json_files, desc="Detecting column centers"):
        # Find matching median width JSON
        median_json_path = find_matching_median_json(layout_json_path, args.median_folder)
        
        if not median_json_path:
            logger.warning(f"No matching median width JSON found for {os.path.basename(layout_json_path)}")
            failure_count += 1
            continue
        
        # Process the page
        success = process_page(layout_json_path, median_json_path, args.output_folder, 
                              min_confidence=args.min_confidence)
        
        if success:
            success_count += 1
        else:
            failure_count += 1
    
    logger.info(f"Processing complete. Successfully processed {success_count} pages, {failure_count} failures.")
    logger.info(f"Results saved to {args.output_folder}")

if __name__ == "__main__":
    main()