#!/usr/bin/env python3
"""
Grid Box Combination Script

This script combines bounding boxes from different grid patterns (2x2, 3x3, 4x4, etc.)
into a single combined output for each main image.

Requirements:
- OpenCV (cv2)
- NumPy
- tqdm (for progress bars)

Usage:
python combine_grid_boxes.py --input_folder [path] --output_folder [path] [--iou_threshold 0.5]
"""

import os
import json
import argparse
import logging
import numpy as np
import cv2
from tqdm import tqdm
import glob

# Setup logging
def setup_logger():
    logger = logging.getLogger("GridBoxCombiner")
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

def calculate_iou(box1, box2):
    """
    Calculate IoU (Intersection over Union) between two bounding boxes.
    
    Args:
        box1 (list): First bounding box [x_min, y_min, x_max, y_max]
        box2 (list): Second bounding box [x_min, y_min, x_max, y_max]
        
    Returns:
        float: IoU value
    """
    # Get coordinates of intersection rectangle
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])
    
    # Check if there is an intersection
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    # Calculate intersection area
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Calculate union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area
    
    # Calculate IoU
    iou = intersection_area / union_area if union_area > 0 else 0.0
    
    return iou

def apply_non_max_suppression(boxes, scores, classes, class_names, iou_threshold=0.5):
    """
    Apply non-maximum suppression to bounding boxes.
    
    Args:
        boxes (list): List of bounding boxes [x_min, y_min, x_max, y_max]
        scores (list): List of confidence scores
        classes (list): List of class IDs
        class_names (list): List of class names
        iou_threshold (float): IoU threshold for suppression
        
    Returns:
        tuple: (filtered_boxes, filtered_scores, filtered_classes, filtered_class_names)
    """
    if not boxes:
        return [], [], [], []
    
    # Create copies of the input lists
    boxes_copy = boxes.copy()
    scores_copy = scores.copy()
    classes_copy = classes.copy()
    class_names_copy = class_names.copy()
    
    # Lists to store the results
    filtered_boxes = []
    filtered_scores = []
    filtered_classes = []
    filtered_class_names = []
    
    # Continue until all boxes are processed
    while boxes_copy:
        # Find the box with the highest score
        max_score_idx = scores_copy.index(max(scores_copy))
        
        # Add the box with the highest score to the result
        filtered_boxes.append(boxes_copy[max_score_idx])
        filtered_scores.append(scores_copy[max_score_idx])
        filtered_classes.append(classes_copy[max_score_idx])
        filtered_class_names.append(class_names_copy[max_score_idx])
        
        # Remove the added box
        current_box = boxes_copy.pop(max_score_idx)
        current_class = classes_copy.pop(max_score_idx)
        scores_copy.pop(max_score_idx)
        class_names_copy.pop(max_score_idx)
        
        # Check remaining boxes
        i = 0
        while i < len(boxes_copy):
            # If IoU is higher than threshold and class is the same, remove the box
            if calculate_iou(current_box, boxes_copy[i]) > iou_threshold and classes_copy[i] == current_class:
                boxes_copy.pop(i)
                scores_copy.pop(i)
                classes_copy.pop(i)
                class_names_copy.pop(i)
            else:
                i += 1
    
    return filtered_boxes, filtered_scores, filtered_classes, filtered_class_names

def find_grid_jsons(input_folder):
    """
    Find all grid JSON files in the input folder and organize them by base image name.
    
    Args:
        input_folder (str): Path to the input folder
        
    Returns:
        dict: Dictionary mapping image base names to lists of grid JSON paths
    """
    grid_jsons = {}
    
    # Search for grid JSON files in the main JSON folder
    json_folder = os.path.join(input_folder, 'json')
    if os.path.exists(json_folder):
        # Find grid pattern JSONs
        grid_pattern = os.path.join(json_folder, '*_grid_*.json')
        for grid_json in glob.glob(grid_pattern):
            base_name = os.path.basename(grid_json).split('_grid_')[0]
            
            if base_name not in grid_jsons:
                grid_jsons[base_name] = []
            
            grid_jsons[base_name].append(grid_json)
        
        # Also search for standard JSON files (non-grid)
        standard_pattern = os.path.join(json_folder, '*.json')
        for json_file in glob.glob(standard_pattern):
            if '_grid_' not in json_file and '_combined' not in json_file:
                base_name = os.path.splitext(os.path.basename(json_file))[0]
                
                if base_name not in grid_jsons:
                    grid_jsons[base_name] = []
                    
                # Add standard JSON at the beginning
                grid_jsons[base_name].insert(0, json_file)
    
    # Also search for JSONs in grid folders
    for grid_folder in os.listdir(input_folder):
        if grid_folder.startswith('grid_') and os.path.isdir(os.path.join(input_folder, grid_folder)):
            grid_json_folder = os.path.join(input_folder, grid_folder, 'json')
            if os.path.exists(grid_json_folder):
                grid_json_pattern = os.path.join(grid_json_folder, '*.json')
                for grid_json in glob.glob(grid_json_pattern):
                    # Extract the base name from the grid cell filename
                    filename = os.path.basename(grid_json)
                    # Try to get base name with various patterns
                    if '_row' in filename and '_col' in filename:
                        base_name = filename.split('_row')[0]
                    else:
                        # Just use the name without extension
                        base_name = os.path.splitext(filename)[0]
                    
                    if base_name not in grid_jsons:
                        grid_jsons[base_name] = []
                    
                    grid_jsons[base_name].append(grid_json)
    
    return grid_jsons

def combine_boxes_for_image(image_base_name, json_paths, iou_threshold=0.5):
    """
    Combine bounding boxes from different grid patterns for a single image.
    
    Args:
        image_base_name (str): Base name of the image
        json_paths (list): List of paths to JSON files for this image
        iou_threshold (float): IoU threshold for non-maximum suppression
        
    Returns:
        dict: Combined regions dictionary
    """
    all_boxes = []
    all_scores = []
    all_classes = []
    all_class_names = []
    
    image_path = None
    image_size = None
    
    # First, collect all boxes from all grid patterns
    for json_path in json_paths:
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # Handle different JSON structures
            if 'cells' in data:  # This is a grid info JSON
                if not image_path and 'original_image_path' in data:
                    image_path = data['original_image_path']
                
                # Get boxes from each cell
                for cell in data['cells']:
                    if 'regions' in cell and 'boxes_original' in cell['regions']:
                        # Add boxes from this cell
                        boxes = cell['regions']['boxes_original']
                        scores = cell['regions']['scores']
                        classes = cell['regions']['classes']
                        names = cell['regions']['class_names']
                        
                        all_boxes.extend(boxes)
                        all_scores.extend(scores)
                        all_classes.extend(classes)
                        all_class_names.extend(names)
            
            elif 'boxes' in data:  # This is a standard JSON
                if not image_path and 'image_path' in data:
                    image_path = data['image_path']
                
                if not image_size and 'image_size' in data:
                    image_size = data['image_size']
                
                # Check if this is a grid cell JSON
                if 'boxes_original' in data:
                    # Use original coordinates
                    boxes = data['boxes_original']
                else:
                    # Use regular coordinates
                    boxes = data['boxes']
                
                scores = data['scores']
                classes = data['classes']
                names = data['class_names']
                
                all_boxes.extend(boxes)
                all_scores.extend(scores)
                all_classes.extend(classes)
                all_class_names.extend(names)
        
        except Exception as e:
            logger.error(f"Error reading {json_path}: {str(e)}")
    
    if not all_boxes:
        logger.warning(f"No boxes found for {image_base_name}")
        return None
    
    # Apply non-maximum suppression to remove duplicate detections
    filtered_boxes, filtered_scores, filtered_classes, filtered_class_names = apply_non_max_suppression(
        all_boxes, all_scores, all_classes, all_class_names, iou_threshold
    )
    
    # Create combined regions dictionary
    combined_regions = {
        'image_path': image_path,
        'image_size': image_size,
        'parameters': {'iou_threshold': iou_threshold},
        'boxes': filtered_boxes,
        'classes': filtered_classes,
        'scores': filtered_scores,
        'class_names': filtered_class_names,
        'source_jsons': json_paths
    }
    
    return combined_regions

def colormap(N=256, normalized=False):
    """
    Generate a color map for visualization.
    
    Args:
        N (int): Number of colors to generate
        normalized (bool): Whether to normalize values to [0,1]
        
    Returns:
        np.ndarray: Color map array of shape (N, 3)
    """
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)
        
    cmap = np.zeros((N, 3), dtype=np.uint8)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << (7 - j))
            g = g | (bitget(c, 1) << (7 - j))
            b = b | (bitget(c, 2) << (7 - j))
            c = c >> 3
        cmap[i] = np.array([r, g, b])
    
    if normalized:
        cmap = cmap.astype(np.float32) / 255.0
    return cmap

def visualize_combined_regions(image_path, regions, output_path, alpha=0.3):
    """
    Create a visualization of the combined regions.
    
    Args:
        image_path (str): Path to the original image
        regions (dict): Dictionary containing combined detection results
        output_path (str): Path to save the visualization
        alpha (float): Transparency of the region overlays
        
    Returns:
        None
    """
    if regions is None or len(regions['boxes']) == 0:
        logger.warning(f"No regions to visualize for {os.path.basename(image_path)}")
        return
    
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"Failed to load image for visualization: {image_path}")
        return
    
    overlay = image.copy()
    
    # Extract data from regions
    boxes = regions['boxes']
    classes = regions['classes']
    scores = regions['scores']
    class_names = regions['class_names']
    
    # Create colormap
    ID_TO_NAMES = {
        0: 'title',
        1: 'plain_text',
        2: 'abandon',
        3: 'figure',
        4: 'figure_caption',
        5: 'table',
        6: 'table_caption',
        7: 'table_footnote',
        8: 'isolate_formula',
        9: 'formula_caption'
    }
    cmap = colormap(N=len(ID_TO_NAMES), normalized=False)
    
    # Draw bounding boxes and labels
    for i, box in enumerate(boxes):
        x_min, y_min, x_max, y_max = map(int, box)
        class_id = int(classes[i])
        class_name = class_names[i]
        score = scores[i]
        
        text = f"{class_name}: {score:.3f}"
        
        # Get color for this class
        color = tuple(int(c) for c in cmap[class_id % len(cmap)])
        
        # Draw filled rectangle on overlay
        cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), color, -1)
        
        # Draw rectangle outline on image
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
        
        # Add class name with background
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(image, (x_min, y_min - text_height - baseline), 
                     (x_min + text_width, y_min), color, -1)
        cv2.putText(image, text, (x_min, y_min - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Blend the overlay with the original image
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    
    # Save the visualization
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, image)
    logger.info(f"Saved visualization to {output_path}")

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Combine bounding boxes from different grid patterns')
    parser.add_argument('--input_folder', required=True, help='Folder containing processed results')
    parser.add_argument('--output_folder', required=True, help='Folder to save combined results')
    parser.add_argument('--iou_threshold', type=float, default=0.5, help='IoU threshold for box merging')
    parser.add_argument('--viz_alpha', type=float, default=0.3, help='Transparency for visualizations')
    
    args = parser.parse_args()
    
    # Create output folders
    output_json_folder = os.path.join(args.output_folder, 'json')
    output_viz_folder = os.path.join(args.output_folder, 'visualizations')
    
    os.makedirs(output_json_folder, exist_ok=True)
    os.makedirs(output_viz_folder, exist_ok=True)
    
    # Find all grid JSON files
    logger.info(f"Searching for grid JSON files in {args.input_folder}")
    grid_jsons = find_grid_jsons(args.input_folder)
    
    if not grid_jsons:
        logger.error(f"No JSON files found in {args.input_folder}")
        return
    
    logger.info(f"Found JSONs for {len(grid_jsons)} images")
    
    # Process each image
    for image_base_name, json_paths in tqdm(grid_jsons.items(), desc="Combining bounding boxes"):
        # Combine boxes for this image
        combined_regions = combine_boxes_for_image(
            image_base_name=image_base_name,
            json_paths=json_paths,
            iou_threshold=args.iou_threshold
        )
        
        if combined_regions:
            # Save combined JSON
            combined_json_path = os.path.join(output_json_folder, f"{image_base_name}_combined.json")
            with open(combined_json_path, 'w') as f:
                json.dump(combined_regions, f, indent=2)
            
            # Create and save visualization
            image_path = combined_regions['image_path']
            if image_path and os.path.exists(image_path):
                viz_path = os.path.join(output_viz_folder, f"{image_base_name}_combined_viz.jpg")
                visualize_combined_regions(
                    image_path=image_path,
                    regions=combined_regions,
                    output_path=viz_path,
                    alpha=args.viz_alpha
                )
            else:
                logger.warning(f"Image path not found for {image_base_name}, skipping visualization")
    
    logger.info(f"Processing complete. Combined results saved to {args.output_folder}")

if __name__ == "__main__":
    main()