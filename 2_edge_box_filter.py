#!/usr/bin/env python3
"""
Edge Box Filter Script
This script processes document layout analysis results to filter out bounding boxes 
that touch or are close to internal edges created by grid division.
This version supports the extended folder structure with multiple grid configurations,
and only filters boxes touching internal grid edges.
Requirements:
- OpenCV (cv2)
- NumPy
- tqdm (for progress bars)
Usage:
python filter_edge_boxes.py --input_folder [path] --output_folder [path] 
    [--edge_threshold 10] [--viz_alpha 0.3] [--process_grids]
"""
import os
import json
import argparse
import logging
import numpy as np
import cv2
from tqdm import tqdm

# Setup logging
def setup_logger():
    logger = logging.getLogger("EdgeBoxFilter")
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

def is_box_touching_internal_edge(box, cell_coordinates, image_width, image_height, threshold=10):
    """
    Check if a bounding box is touching or close to an internal edge of a grid cell.
    Only considers edges that were created by the grid division.
    
    Args:
        box (list): Bounding box coordinates [x_min, y_min, x_max, y_max]
        cell_coordinates (list): Cell coordinates [x_min, y_min, x_max, y_max]
        image_width (int): Width of the original image
        image_height (int): Height of the original image
        threshold (int): Distance threshold from edge (in pixels)
        
    Returns:
        bool: True if the box is touching an internal edge, False otherwise
    """
    x_min, y_min, x_max, y_max = box
    
    # Handle cell_coordinates as a dict or list
    if isinstance(cell_coordinates, dict):
        cell_x_min = cell_coordinates.get('x_start', 0)
        cell_y_min = cell_coordinates.get('y_start', 0)
        cell_x_max = cell_coordinates.get('x_end', image_width)
        cell_y_max = cell_coordinates.get('y_end', image_height)
    else:
        cell_x_min, cell_y_min, cell_x_max, cell_y_max = cell_coordinates
    
    # Check against internal right edge (if not at the right edge of the original image)
    right_edge_internal = abs(cell_x_max - image_width) > threshold
    if right_edge_internal and x_max >= (cell_x_max - threshold):
        return True
    
    # Check against internal bottom edge (if not at the bottom edge of the original image)
    bottom_edge_internal = abs(cell_y_max - image_height) > threshold
    if bottom_edge_internal and y_max >= (cell_y_max - threshold):
        return True
    
    # Check against internal left edge (if not at the left edge of the original image)
    left_edge_internal = cell_x_min > threshold
    if left_edge_internal and x_min <= (cell_x_min + threshold):
        return True
    
    # Check against internal top edge (if not at the top edge of the original image)
    top_edge_internal = cell_y_min > threshold
    if top_edge_internal and y_min <= (cell_y_min + threshold):
        return True
    
    return False

def filter_edge_boxes(regions, threshold=10):
    """
    For non-grid images, don't filter any boxes.
    For grid cell images, filter out bounding boxes that touch internal edges.
    
    Args:
        regions (dict): Dictionary containing detection results
        threshold (int): Distance threshold from edge (in pixels)
        
    Returns:
        dict: New dictionary with filtered boxes
    """
    # If this is not a grid cell (no cell_coordinates), return the original regions
    if 'cell_coordinates' not in regions:
        logger.info("Non-grid image detected, not filtering any boxes")
        return regions
    
    # This is a grid cell, filter boxes touching internal edges
    image_width = regions['image_size']['width']
    image_height = regions['image_size']['height']
    cell_coordinates = regions['cell_coordinates']
    
    filtered_indices = []
    
    # Check each box
    for i, box in enumerate(regions['boxes']):
        # Keep boxes that don't touch internal edges
        if not is_box_touching_internal_edge(box, cell_coordinates, image_width, image_height, threshold):
            filtered_indices.append(i)
    
    # Create a new regions dictionary with filtered boxes
    filtered_regions = {
        'image_path': regions['image_path'],
        'image_size': regions['image_size'],
        'parameters': regions['parameters'],
        'boxes': [regions['boxes'][i] for i in filtered_indices],
        'classes': [regions['classes'][i] for i in filtered_indices],
        'scores': [regions['scores'][i] for i in filtered_indices],
        'class_names': [regions['class_names'][i] for i in filtered_indices]
    }
    
    # Add additional fields if they exist
    if 'boxes_original' in regions:
        filtered_regions['boxes_original'] = [regions['boxes_original'][i] for i in filtered_indices]
    
    if 'cell_coordinates' in regions:
        filtered_regions['cell_coordinates'] = regions['cell_coordinates']
    
    if 'original_image_path' in regions:
        filtered_regions['original_image_path'] = regions['original_image_path']
    
    if 'grid_info' in regions:
        filtered_regions['grid_info'] = regions['grid_info']
    
    return filtered_regions

def filter_grid_info(grid_info, threshold=10):
    """
    Filter boxes in a grid info JSON file, only removing boxes that touch internal grid edges.
    
    Args:
        grid_info (dict): Dictionary containing grid information
        threshold (int): Distance threshold from edge (in pixels)
        
    Returns:
        dict: New dictionary with filtered boxes in grid cells
    """
    # Create a new grid info dictionary
    filtered_grid_info = {
        'original_image_path': grid_info['original_image_path'],
        'cells': []
    }
    
    # Add grid_config if it exists
    if 'grid_config' in grid_info:
        filtered_grid_info['grid_config'] = grid_info['grid_config']
    
    # Get rows and cols from grid_info
    if 'grid_info' in grid_info and isinstance(grid_info['grid_info'], dict):
        rows = grid_info['grid_info'].get('rows', 1)
        cols = grid_info['grid_info'].get('cols', 1)
    else:
        # Try to extract from filename or assume default
        try:
            filename = os.path.basename(grid_info.get('image_path', ''))
            if 'grid_' in filename:
                grid_part = filename.split('grid_')[1].split('_')[0]
                if 'x' in grid_part:
                    rows, cols = map(int, grid_part.split('x'))
                else:
                    rows, cols = 1, 1
            else:
                rows, cols = 1, 1
        except:
            rows, cols = 1, 1
    
    # Get image dimensions
    if 'image_path' in grid_info and os.path.exists(grid_info['image_path']):
        image_path = grid_info['image_path']
    else:
        image_path = grid_info['original_image_path']
    
    if os.path.exists(image_path):
        image = cv2.imread(image_path)
        if image is not None:
            image_height, image_width = image.shape[:2]
        else:
            logger.warning(f"Could not read original image: {image_path}")
            return None
    else:
        logger.warning(f"Original image not found: {image_path}")
        return None
    
    # Filter boxes in each cell
    for cell in grid_info['cells']:
        cell_row = cell.get('row', 0)
        cell_col = cell.get('col', 0)
        cell_coords = cell['cell_coordinates']
        
        filtered_boxes_indices = []
        
        # Check each box in the original coordinates
        for i, box in enumerate(cell['regions']['boxes_original']):
            # Convert box to original image coordinates for filtering
            if not is_box_touching_internal_edge(box, cell_coords, image_width, image_height, threshold):
                filtered_boxes_indices.append(i)
        
        # Create filtered cell
        filtered_cell = {
            'cell_path': cell['cell_path'],
            'cell_json_path': cell['cell_json_path'],
            'cell_coordinates': cell['cell_coordinates'],
            'row': cell.get('row', 0),
            'col': cell.get('col', 0),
            'regions': {
                'boxes': [cell['regions']['boxes'][i] for i in filtered_boxes_indices],
                'boxes_original': [cell['regions']['boxes_original'][i] for i in filtered_boxes_indices],
                'classes': [cell['regions']['classes'][i] for i in filtered_boxes_indices],
                'scores': [cell['regions']['scores'][i] for i in filtered_boxes_indices],
                'class_names': [cell['regions']['class_names'][i] for i in filtered_boxes_indices]
            }
        }
        
        filtered_grid_info['cells'].append(filtered_cell)
    
    return filtered_grid_info

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

def visualize_regions(image_path, regions, output_path, alpha=0.3, use_original_coords=False):
    """
    Create a visualization of the detected regions.
    
    Args:
        image_path (str): Path to the input image
        regions (dict): Dictionary containing detection results
        output_path (str): Path to save the visualization
        alpha (float): Transparency of the region overlays
        use_original_coords (bool): Whether to use original coordinates from the parent image
        
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
    if use_original_coords and 'boxes_original' in regions:
        boxes = regions['boxes_original']
    else:
        boxes = regions['boxes']
        
    classes = regions['classes']
    scores = regions['scores']
    class_names = regions['class_names']
    
    # Create colormap
    # This should match the ID_TO_NAMES from the original script
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

def process_json_file(json_path, output_json_folder, output_viz_folder, edge_threshold, viz_alpha):
    """
    Process a single JSON file to filter edge boxes and create visualization.
    Always copies the JSON and creates visualizations, even if no boxes are filtered.
    
    Args:
        json_path (str): Path to the input JSON file
        output_json_folder (str): Folder to save the filtered JSON file
        output_viz_folder (str): Folder to save the visualization
        edge_threshold (int): Distance threshold from edge (in pixels)
        viz_alpha (float): Transparency for visualization
        
    Returns:
        bool: True if processing was successful, False otherwise
    """
    try:
        # Load the input JSON file
        with open(json_path, 'r') as f:
            regions = json.load(f)
        
        # Check if this is a grid info JSON file
        if 'cells' in regions and ('grid_config' in regions or 'grid_info' in regions):
            return process_grid_info_file(
                json_path, output_json_folder, output_viz_folder, 
                edge_threshold, viz_alpha
            )
        
        # Get the image path from the JSON
        image_path = regions.get('image_path')
        if not image_path or not os.path.exists(image_path):
            # Try to handle the case where the image path might be relative or incorrect
            # Check if this is a cell from grid processing
            if 'original_image_path' in regions:
                image_path = regions.get('original_image_path')
                if not os.path.exists(image_path):
                    logger.error(f"Image path not found: {image_path}")
                    return False
            else:
                # Try to locate the image relative to the JSON path
                base_dir = os.path.dirname(json_path)
                image_filename = os.path.basename(image_path) if image_path else None
                if image_filename:
                    # Try different possible locations
                    possible_paths = [
                        os.path.join(base_dir, '..', 'images', image_filename),
                        os.path.join(base_dir, '..', '..', image_filename),
                        os.path.join(base_dir, '..', '..', '..', image_filename)
                    ]
                    
                    for path in possible_paths:
                        if os.path.exists(path):
                            image_path = path
                            break
                    
                    if not os.path.exists(image_path):
                        logger.error(f"Image path not found: {image_path}")
                        return False
                else:
                    logger.error(f"Image path not found in JSON: {json_path}")
                    return False
        
        # Filter out boxes touching the internal edges (or keep all for non-grid images)
        filtered_regions = filter_edge_boxes(regions, edge_threshold)
        
        # Always save the JSON, even if no boxes were filtered
        output_json_path = os.path.join(output_json_folder, os.path.basename(json_path))
        with open(output_json_path, 'w') as f:
            json.dump(filtered_regions, f, indent=2)
        
        # Always create and save visualization
        basename = os.path.splitext(os.path.basename(json_path))[0]
        output_viz_path = os.path.join(output_viz_folder, f"{basename}_filtered_viz.jpg")
        
        # Determine whether to use original coordinates (for grid cells)
        use_original_coords = 'boxes_original' in filtered_regions
        vis_image_path = image_path
        
        # For grid cells, use the original image for visualization if available
        if use_original_coords and 'original_image_path' in filtered_regions:
            orig_path = filtered_regions['original_image_path']
            if os.path.exists(orig_path):
                vis_image_path = orig_path
        
        visualize_regions(
            image_path=vis_image_path,
            regions=filtered_regions,
            output_path=output_viz_path,
            alpha=viz_alpha,
            use_original_coords=use_original_coords
        )
        
        # Log the changes
        original_count = len(regions['boxes'])
        filtered_count = len(filtered_regions['boxes'])
        removed_count = original_count - filtered_count
        
        if removed_count == 0:
            logger.info(f"Processed {basename}: No boxes were filtered (copied without changes)")
        else:
            logger.info(f"Processed {basename}: removed {removed_count} of {original_count} boxes")
        
        return True
    
    except Exception as e:
        logger.error(f"Error processing {os.path.basename(json_path)}: {str(e)}")
        return False

def process_grid_info_file(json_path, output_json_folder, output_viz_folder, edge_threshold, viz_alpha):
    """
    Process a grid info JSON file.
    Always saves the JSON and creates visualizations, even if no boxes are filtered.
    
    Args:
        json_path (str): Path to the grid info JSON file
        output_json_folder (str): Folder to save the filtered JSON file
        output_viz_folder (str): Folder to save the visualization
        edge_threshold (int): Distance threshold from edge (in pixels)
        viz_alpha (float): Transparency for visualization
        
    Returns:
        bool: True if processing was successful, False otherwise
    """
    try:
        # Load the grid info JSON file
        with open(json_path, 'r') as f:
            grid_info = json.load(f)
        
        # Filter edge boxes in the grid info
        filtered_grid_info = filter_grid_info(grid_info, edge_threshold)
        
        if filtered_grid_info:
            # Always save the filtered grid info JSON
            output_json_path = os.path.join(output_json_folder, os.path.basename(json_path))
            with open(output_json_path, 'w') as f:
                json.dump(filtered_grid_info, f, indent=2)
            
            # Always create a visualization of the filtered grid info
            basename = os.path.splitext(os.path.basename(json_path))[0]
            output_viz_path = os.path.join(output_viz_folder, f"{basename}_filtered_viz.jpg")
            
            # Get the original image path
            image_path = filtered_grid_info['original_image_path']
            if not os.path.exists(image_path):
                logger.error(f"Original image not found: {image_path}")
                return False
            
            # Create a combined visualization
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Failed to load image for visualization: {image_path}")
                return False
            
            overlay = image.copy()
            
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
            
            # Draw all filtered boxes from all cells
            for cell in filtered_grid_info['cells']:
                for i, box in enumerate(cell['regions']['boxes_original']):
                    x_min, y_min, x_max, y_max = map(int, box)
                    class_id = int(cell['regions']['classes'][i])
                    class_name = cell['regions']['class_names'][i]
                    score = cell['regions']['scores'][i]
                    
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
            cv2.addWeighted(overlay, viz_alpha, image, 1 - viz_alpha, 0, image)
            
            # Save the visualization
            os.makedirs(os.path.dirname(output_viz_path), exist_ok=True)
            cv2.imwrite(output_viz_path, image)
            logger.info(f"Saved grid visualization to {output_viz_path}")
            
            # Count total boxes
            total_boxes = sum(len(cell['regions']['boxes_original']) for cell in filtered_grid_info['cells'])
            total_original_boxes = sum(len(cell['regions']['boxes']) for cell in grid_info['cells'])
            removed_count = total_original_boxes - total_boxes
            
            if removed_count == 0:
                logger.info(f"Processed grid {basename}: No boxes were filtered (copied without changes)")
            else:
                logger.info(f"Processed grid {basename}: removed {removed_count} of {total_original_boxes} boxes")
            
            return True
        else:
            # If filtering failed but we still want to copy the original
            output_json_path = os.path.join(output_json_folder, os.path.basename(json_path))
            with open(output_json_path, 'w') as f:
                json.dump(grid_info, f, indent=2)
            logger.info(f"Failed to filter grid info: {json_path}, copied original instead")
            return True
    
    except Exception as e:
        logger.error(f"Error processing grid info {os.path.basename(json_path)}: {str(e)}")
        return False

def process_grid_folder(grid_folder, output_folder, edge_threshold, viz_alpha):
    """
    Process a grid folder (e.g., grid_2x2).
    
    Args:
        grid_folder (str): Path to the grid folder
        output_folder (str): Base output folder
        edge_threshold (int): Distance threshold from edge (in pixels)
        viz_alpha (float): Transparency for visualization
        
    Returns:
        bool: True if processing was successful, False otherwise
    """
    try:
        # Get the grid configuration from the folder name
        grid_config = os.path.basename(grid_folder).replace('grid_', '')
        logger.info(f"Processing grid folder: {grid_config}")
        
        # Create output folders
        grid_output_folder = os.path.join(output_folder, os.path.basename(grid_folder))
        grid_json_output_folder = os.path.join(grid_output_folder, 'json')
        grid_viz_output_folder = os.path.join(grid_output_folder, 'visualizations')
        grid_viz_orig_output_folder = os.path.join(grid_output_folder, 'visualizations_original_coords')
        
        os.makedirs(grid_json_output_folder, exist_ok=True)
        os.makedirs(grid_viz_output_folder, exist_ok=True)
        os.makedirs(grid_viz_orig_output_folder, exist_ok=True)
        
        # Process JSON files in the grid folder
        grid_json_folder = os.path.join(grid_folder, 'json')
        if os.path.exists(grid_json_folder):
            json_files = [f for f in os.listdir(grid_json_folder) if f.endswith('.json')]
            
            for json_file in tqdm(json_files, desc=f"Processing {grid_config} JSON files"):
                json_path = os.path.join(grid_json_folder, json_file)
                process_json_file(
                    json_path=json_path,
                    output_json_folder=grid_json_output_folder,
                    output_viz_folder=grid_viz_output_folder,
                    edge_threshold=edge_threshold,
                    viz_alpha=viz_alpha
                )
        
        # Process original coords visualizations
        grid_viz_orig_folder = os.path.join(grid_folder, 'visualizations_original_coords')
        if os.path.exists(grid_viz_orig_folder):
            # Copy the original coordinate visualizations directory structure
            for root, dirs, files in os.walk(grid_viz_orig_folder):
                rel_path = os.path.relpath(root, grid_viz_orig_folder)
                if rel_path == '.':
                    rel_path = ''
                
                for file in files:
                    if file.endswith('.json'):
                        json_path = os.path.join(root, file)
                        dest_folder = os.path.join(grid_viz_orig_output_folder, rel_path)
                        os.makedirs(dest_folder, exist_ok=True)
                        
                        process_json_file(
                            json_path=json_path,
                            output_json_folder=dest_folder,
                            output_viz_folder=dest_folder,
                            edge_threshold=edge_threshold,
                            viz_alpha=viz_alpha
                        )
        
        return True
    
    except Exception as e:
        logger.error(f"Error processing grid folder {os.path.basename(grid_folder)}: {str(e)}")
        return False

def get_json_files(input_folder):
    """
    Get paths of all JSON files in the input folder.
    
    Args:
        input_folder (str): Path to the folder containing JSON files
        
    Returns:
        list: List of JSON file paths
    """
    json_paths = []
    
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith('.json'):
                json_paths.append(os.path.join(root, file))
    
    return sorted(json_paths)

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Filter bounding boxes that touch internal grid edges')
    parser.add_argument('--input_folder', required=True, help='Folder containing processed results')
    parser.add_argument('--output_folder', required=True, help='Folder to save filtered results')
    parser.add_argument('--edge_threshold', type=int, default=10, help='Distance threshold from edge (in pixels)')
    parser.add_argument('--viz_alpha', type=float, default=0.3, help='Transparency for visualizations')
    parser.add_argument('--skip_errors', action='store_true', help='Continue processing if a file fails')
    parser.add_argument('--process_grids', action='store_true', help='Process grid folders')
    
    args = parser.parse_args()
    
    # Create output folders
    output_json_folder = os.path.join(args.output_folder, 'json')
    output_viz_folder = os.path.join(args.output_folder, 'visualizations')
    
    os.makedirs(output_json_folder, exist_ok=True)
    os.makedirs(output_viz_folder, exist_ok=True)
    
    # Get JSON file paths from the main JSON folder
    json_folder = os.path.join(args.input_folder, 'json')
    if os.path.exists(json_folder):
        logger.info(f"Processing main JSON folder: {json_folder}")
        json_paths = get_json_files(json_folder)
        
        if not json_paths:
            logger.warning(f"No JSON files found in {json_folder}")
        else:
            logger.info(f"Found {len(json_paths)} JSON files in main folder")
            
            # Process each JSON file
            processed_count = 0
            error_count = 0
            
            for json_path in tqdm(json_paths, desc="Processing main JSON files"):
                try:
                    success = process_json_file(
                        json_path=json_path,
                        output_json_folder=output_json_folder,
                        output_viz_folder=output_viz_folder,
                        edge_threshold=args.edge_threshold,
                        viz_alpha=args.viz_alpha
                    )
                    
                    if success:
                        processed_count += 1
                    else:
                        error_count += 1
                        
                except Exception as e:
                    error_count += 1
                    logger.error(f"Error processing {os.path.basename(json_path)}: {str(e)}")
                    if not args.skip_errors:
                        logger.error("Stopping due to error. Use --skip_errors to continue despite errors.")
                        break
            
            logger.info(f"Main JSON processing complete. Successfully processed {processed_count} JSON files with {error_count} errors")
    else:
        logger.warning(f"Main JSON folder not found: {json_folder}")
    
    # Process grid folders if requested
    if args.process_grids:
        grid_processed_count = 0
        grid_error_count = 0
        
        # Find all grid folders
        grid_folders = []
        for item in os.listdir(args.input_folder):
            if item.startswith('grid_') and os.path.isdir(os.path.join(args.input_folder, item)):
                grid_folders.append(os.path.join(args.input_folder, item))
        
        logger.info(f"Found {len(grid_folders)} grid folders")
        
        for grid_folder in grid_folders:
            try:
                success = process_grid_folder(
                    grid_folder=grid_folder,
                    output_folder=args.output_folder,
                    edge_threshold=args.edge_threshold,
                    viz_alpha=args.viz_alpha
                )
                
                if success:
                    grid_processed_count += 1
                else:
                    grid_error_count += 1
                    
            except Exception as e:
                grid_error_count += 1
                logger.error(f"Error processing grid folder {os.path.basename(grid_folder)}: {str(e)}")
                if not args.skip_errors:
                    logger.error("Stopping grid processing due to error. Use --skip_errors to continue despite errors.")
                    break
        
        logger.info(f"Grid processing complete. Successfully processed {grid_processed_count} grid folders with {grid_error_count} errors")
    
    logger.info(f"All processing complete. Results saved to {args.output_folder}")

if __name__ == "__main__":
    main()