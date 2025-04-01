#!/usr/bin/env python3
"""
Document Layout Analysis Script

This standalone script analyzes document images using a YOLO-based document layout detection model.
It processes images from an input folder, detects document regions (titles, text, figures, tables, etc.),
and outputs both JSON files with the detection results and visualization images with bounding boxes.

The script can process images as a grid of cells (default 2x2, 3x3, and 4x4), with configurable number of grid configurations,
and with customizable overlap between adjacent cells (default 20% overlap).

Requirements:
- PyTorch
- OpenCV (cv2)
- PIL (Python Imaging Library)
- tqdm (for progress bars)
- huggingface_hub
- git (for cloning the repository)

Usage:
python document_layout_analyzer.py --input_folder [path] --output_folder [path] 
    [--conf_threshold 0.1] [--iou_threshold 0.45] [--grids "2x2,3x3,4x4"] [--overlap 20.0] [--disable_grid]

Notes:
- This script will clone the DocLayout-YOLO repository to get the custom YOLO implementation.
- The model will be downloaded from Hugging Face Hub on first run.
"""

import os
import json
import argparse
import logging
import numpy as np
import cv2
import torch
import torchvision
from PIL import Image
from tqdm import tqdm
from huggingface_hub import hf_hub_download, login

# Setup logging
def setup_logger():
    logger = logging.getLogger("DocLayoutAnalyzer")
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

class YOLODocumentLayoutDetector:
    """
    Wrapper for the YOLO-based document layout detection model.
    Detects regions like titles, text blocks, figures, tables, etc. in document images.
    """

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

    def __init__(self, conf_threshold=0.1, iou_threshold=0.45, device=None,
                 repo_id="juliozhao/DocLayout-YOLO-DocStructBench", 
                 model_filename="doclayout_yolo_docstructbench_imgsz1024.pt",
                 model_path=None):
        """
        Initialize the document layout detector.
        
        Args:
            conf_threshold (float): Confidence threshold for detection
            iou_threshold (float): IoU threshold for NMS
            device (str): Device to run inference on ('cpu' or 'cuda')
            repo_id (str): Hugging Face repo ID
            model_filename (str): Filename of the model in the repo
        """
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        logger.info(f"Initializing document layout detector on {self.device}")

        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.image_size = 1024  # Default image size for the model

        try:
            # Check for HF_TOKEN in environment variables
            token = os.environ.get("HF_TOKEN")
            if token:
                login(token=token)
                logger.info("Logged in to Hugging Face Hub using token from environment")
            
            # Use provided model path or download from Hugging Face
            if model_path:
                if not os.path.exists(model_path):
                    logger.error(f"Specified model path does not exist: {model_path}")
                    raise FileNotFoundError(f"Model file not found at {model_path}")
                logger.info(f"Using provided model path: {model_path}")
            else:
                # Download the model file from Hugging Face
                model_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=model_filename
                )
                logger.info(f"Downloaded model from Hugging Face: {model_path}")
            
            # Clone the custom YOLO implementation repository
            import subprocess
            import sys
            
            # Create a directory for the repo
            repo_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "doclayout_yolo_repo")
            os.makedirs(repo_dir, exist_ok=True)
            
            # Check if the required file exists in the repository directory
            if not os.path.exists(os.path.join(repo_dir, "doclayout_yolo.py")):
                logger.info("Required file 'doclayout_yolo.py' not found in existing directory.")
                
                # Check if directory exists but is not empty
                if os.path.exists(repo_dir) and os.listdir(repo_dir):
                    logger.info(f"Directory {repo_dir} already exists and is not empty.")
                    logger.info("Trying to pull latest changes instead of cloning...")
                    
                    try:
                        # Try to pull latest changes if it's a git repository
                        subprocess.check_call(["git", "-C", repo_dir, "pull"], stderr=subprocess.STDOUT)
                        logger.info("Successfully pulled latest changes.")
                    except subprocess.CalledProcessError:
                        logger.warning("Failed to pull changes. Directory might not be a git repository.")
                        logger.info("Checking for model file in the existing directory...")
                else:
                    # Directory doesn't exist or is empty, try to clone
                    logger.info("Cloning DocLayout-YOLO repository...")
                    try:
                        subprocess.check_call([
                            "git", "clone", "https://huggingface.co/juliozhao/DocLayout-YOLO-DocStructBench", 
                            repo_dir
                        ], stderr=subprocess.STDOUT)
                        logger.info("Successfully cloned repository.")
                    except subprocess.CalledProcessError as e:
                        logger.error(f"Failed to clone repository: {str(e)}")
                        # If clone fails, we'll try to directly download the required file
                        logger.info("Attempting to directly download the required file...")
                        try:
                            import requests
                            doclayout_url = "https://huggingface.co/juliozhao/DocLayout-YOLO-DocStructBench/raw/main/doclayout_yolo.py"
                            response = requests.get(doclayout_url)
                            with open(os.path.join(repo_dir, "doclayout_yolo.py"), "wb") as f:
                                f.write(response.content)
                            logger.info("Successfully downloaded doclayout_yolo.py file.")
                        except Exception as download_err:
                            logger.error(f"Failed to download required file: {str(download_err)}")
                            raise RuntimeError("Cannot obtain the required doclayout_yolo.py file. Please check your internet connection or manually download it.")
            
            # Add the repo directory to Python path
            sys.path.insert(0, repo_dir)
            
            # Import the custom YOLO implementation
            from doclayout_yolo import YOLOv10
            self.model = YOLOv10(model_path)
            
            # Set device
            if self.device == "cuda":
                self.model.to(self.device)
            
            logger.info(f"Model loaded successfully")

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def detect_regions(self, image_path):
        """
        Detect regions in an image.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            dict: Dictionary containing bounding boxes, classes, and scores for detected regions
        """
        try:
            image = Image.open(image_path)
            
            # Run detection using the custom YOLOv10 implementation
            det_res = self.model.predict(
                image,
                imgsz=self.image_size,
                conf=self.conf_threshold,
                device=self.device,
            )[0]
            
            # Extract detection results
            boxes = det_res.__dict__['boxes'].xyxy.cpu().numpy()
            classes = det_res.__dict__['boxes'].cls.cpu().numpy()
            scores = det_res.__dict__['boxes'].conf.cpu().numpy()

            # Apply NMS if needed
            if len(boxes) > 0:
                indices = torchvision.ops.nms(
                    boxes=torch.tensor(boxes),
                    scores=torch.tensor(scores),
                    iou_threshold=self.iou_threshold
                ).numpy()

                boxes, scores, classes = boxes[indices], scores[indices], classes[indices]

            regions = {
                'image_path': image_path,
                'image_size': {'width': image.width, 'height': image.height},
                'parameters': {'conf_threshold': self.conf_threshold, 'iou_threshold': self.iou_threshold},
                'boxes': boxes.tolist(),
                'classes': classes.tolist(),
                'scores': scores.tolist(),
                'class_names': [self.ID_TO_NAMES[int(cls)] for cls in classes]
            }

            logger.info(f"Detected {len(boxes)} regions in {os.path.basename(image_path)}")
            return regions

        except Exception as e:
            logger.error(f"Error detecting regions in {os.path.basename(image_path)}: {str(e)}")
            return None

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
    cmap = colormap(N=len(YOLODocumentLayoutDetector.ID_TO_NAMES), normalized=False)
    
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

def get_image_paths(input_folder):
    """
    Get paths of all images in the input folder.
    
    Args:
        input_folder (str): Path to the folder containing images
        
    Returns:
        list: List of image file paths
    """
    image_extensions = ['.jpg', '.jpeg', '.png', '.webp', '.tiff', '.tif', '.bmp']
    image_paths = []
    
    for root, _, files in os.walk(input_folder):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in image_extensions:
                image_paths.append(os.path.join(root, file))
    
    return sorted(image_paths)

def split_image_into_grid(image_path, rows, cols, overlap_percentage):
    """
    Split an image into a grid with the specified number of rows and columns,
    and with the specified overlap between adjacent cells.
    
    Args:
        image_path (str): Path to the input image
        rows (int): Number of rows in the grid
        cols (int): Number of columns in the grid
        overlap_percentage (float): Percentage of overlap between adjacent cells (0-100)
        
    Returns:
        list: List of dictionaries, each containing a grid cell image and its coordinates
    """
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"Failed to load image for grid splitting: {image_path}")
        return []
    
    height, width = image.shape[:2]
    
    # Calculate base cell dimensions (without overlap)
    base_cell_width = width / cols
    base_cell_height = height / rows
    
    # Calculate overlap in pixels (only for internal edges)
    overlap_x = base_cell_width * (overlap_percentage / 100)
    overlap_y = base_cell_height * (overlap_percentage / 100)
    
    grid_cells = []
    
    for row in range(rows):
        for col in range(cols):
            # Calculate cell coordinates with overlap
            x_start = col * base_cell_width
            if col > 0:  # Not the leftmost column, add left overlap
                x_start -= overlap_x
            
            y_start = row * base_cell_height
            if row > 0:  # Not the top row, add top overlap
                y_start -= overlap_y
            
            x_end = (col + 1) * base_cell_width
            if col < cols - 1:  # Not the rightmost column, add right overlap
                x_end += overlap_x
            
            y_end = (row + 1) * base_cell_height
            if row < rows - 1:  # Not the bottom row, add bottom overlap
                y_end += overlap_y
            
            # Ensure we stay within image boundaries
            x_start = max(0, x_start)
            y_start = max(0, y_start)
            x_end = min(width, x_end)
            y_end = min(height, y_end)
            
            # Convert to integers for image slicing
            x_start_int = int(x_start)
            y_start_int = int(y_start)
            x_end_int = int(x_end)
            y_end_int = int(y_end)
            
            # Extract the cell image
            cell_image = image[y_start_int:y_end_int, x_start_int:x_end_int]
            
            grid_cells.append({
                'image': cell_image,
                'coordinates': {
                    'x_start': x_start,
                    'y_start': y_start,
                    'x_end': x_end,
                    'y_end': y_end
                },
                'row': row + 1,  # 1-indexed for naming
                'col': col + 1   # 1-indexed for naming
            })
    
    return grid_cells

def process_image(detector, image_path, json_folder, viz_folder):
    """
    Process a single image (without grid).
    
    Args:
        detector: The document layout detector
        image_path (str): Path to the image
        json_folder (str): Folder to save JSON results
        viz_folder (str): Folder to save visualizations
        
    Returns:
        bool: True if processing was successful, False otherwise
    """
    try:
        # Detect regions
        regions = detector.detect_regions(image_path)
        
        if regions:
            # Save JSON results
            image_filename = os.path.basename(image_path)
            base_filename, _ = os.path.splitext(image_filename)
            
            json_path = os.path.join(json_folder, f"{base_filename}.json")
            with open(json_path, 'w') as f:
                json.dump(regions, f, indent=2)
            
            # Create and save visualization
            viz_path = os.path.join(viz_folder, f"{base_filename}_viz.jpg")
            visualize_regions(image_path, regions, viz_path)
            
            return True
        
        return False
        
    except Exception as e:
        logger.error(f"Error processing {os.path.basename(image_path)}: {str(e)}")
        return False

def translate_coordinates_to_original(boxes, cell_coordinates):
    """
    Translate bounding box coordinates from a grid cell to the original image.
    
    Args:
        boxes (list): List of bounding boxes in the grid cell
        cell_coordinates (dict): Coordinates of the grid cell in the original image
        
    Returns:
        list: List of translated bounding boxes
    """
    x_offset = cell_coordinates['x_start']
    y_offset = cell_coordinates['y_start']
    
    translated_boxes = []
    
    for box in boxes:
        x_min, y_min, x_max, y_max = box
        
        # Translate coordinates to the original image
        original_x_min = x_min + x_offset
        original_y_min = y_min + y_offset
        original_x_max = x_max + x_offset
        original_y_max = y_max + y_offset
        
        translated_boxes.append([original_x_min, original_y_min, original_x_max, original_y_max])
    
    return translated_boxes

def process_image_with_grid(detector, image_path, grid_folder, rows, cols, overlap_percentage, main_json_folder=None):
    """
    Process an image as a grid of cells.
    
    Args:
        detector: The document layout detector
        image_path (str): Path to the image
        grid_folder (str): Folder to save grid cell results
        rows (int): Number of rows in the grid
        cols (int): Number of columns in the grid
        overlap_percentage (float): Percentage of overlap between adjacent cells
        main_json_folder (str, optional): Folder to save JSON with all grid information
        
    Returns:
        bool: True if processing was successful, False otherwise
    """
    try:
        image_filename = os.path.basename(image_path)
        base_filename, ext = os.path.splitext(image_filename)
        
        # Create grid folders
        grid_images_folder = os.path.join(grid_folder, 'images')
        grid_json_folder = os.path.join(grid_folder, 'json')
        grid_viz_folder = os.path.join(grid_folder, 'visualizations')
        grid_viz_original_folder = os.path.join(grid_folder, 'visualizations_original_coords')
        
        os.makedirs(grid_images_folder, exist_ok=True)
        os.makedirs(grid_json_folder, exist_ok=True)
        os.makedirs(grid_viz_folder, exist_ok=True)
        os.makedirs(grid_viz_original_folder, exist_ok=True)
        
        # Split the image into a grid
        grid_cells = split_image_into_grid(image_path, rows, cols, overlap_percentage)
        
        if not grid_cells:
            logger.warning(f"Failed to split {image_filename} into a grid, skipping grid processing")
            return False
        
        # Dictionary to store all grid cell information for the main JSON
        all_grid_info = {
            'original_image_path': image_path,
            'grid_config': {
                'rows': rows,
                'cols': cols,
                'overlap_percentage': overlap_percentage
            },
            'cells': []
        }
        
        # Process each grid cell
        for cell in grid_cells:
            # Save the cell image
            cell_filename = f"{base_filename}_row{cell['row']}_col{cell['col']}{ext}"
            cell_filepath = os.path.join(grid_images_folder, cell_filename)
            
            cv2.imwrite(cell_filepath, cell['image'])
            
            try:
                # Detect regions in the cell
                cell_regions = detector.detect_regions(cell_filepath)
                
                if cell_regions:
                    # Translate bounding box coordinates back to the original image
                    original_boxes = translate_coordinates_to_original(
                        cell_regions['boxes'], cell['coordinates']
                    )
                    
                    # Add original image coordinates to the cell regions
                    cell_regions['cell_coordinates'] = cell['coordinates']
                    cell_regions['original_image_path'] = image_path
                    cell_regions['boxes_original'] = original_boxes
                    cell_regions['grid_info'] = {
                        'rows': rows,
                        'cols': cols,
                        'row': cell['row'],
                        'col': cell['col']
                    }
                    
                    # Save JSON results for the cell
                    cell_json_path = os.path.join(grid_json_folder, cell_filename.replace(ext, '.json'))
                    with open(cell_json_path, 'w') as f:
                        json.dump(cell_regions, f, indent=2)
                    
                    # Create and save visualization for the cell (with cell coordinates)
                    cell_viz_path = os.path.join(grid_viz_folder, cell_filename.replace(ext, '_viz.jpg'))
                    visualize_regions(cell_filepath, cell_regions, cell_viz_path, use_original_coords=False)
                    
                    # Create and save visualization with original coordinates
                    original_viz_path = os.path.join(grid_viz_original_folder, cell_filename.replace(ext, '_original_viz.jpg'))
                    try:
                        # Copy the original image for visualization
                        original_img = cv2.imread(image_path)
                        if original_img is not None:
                            # Create visualization on the original image
                            original_regions = cell_regions.copy()
                            original_regions['boxes'] = original_boxes  # Replace boxes with original coordinates
                            
                            # Save as a new image
                            tmp_img_path = os.path.join(grid_viz_original_folder, f"tmp_{cell_filename}")
                            cv2.imwrite(tmp_img_path, original_img)
                            
                            # Create visualization
                            visualize_regions(tmp_img_path, original_regions, original_viz_path)
                            
                            # Clean up temp file
                            if os.path.exists(tmp_img_path):
                                os.remove(tmp_img_path)
                    except Exception as viz_err:
                        logger.error(f"Error creating original coordinate visualization: {str(viz_err)}")
                    
                    # Add cell information to the all_grid_info dictionary
                    all_grid_info['cells'].append({
                        'cell_path': cell_filepath,
                        'cell_json_path': cell_json_path,
                        'cell_coordinates': cell['coordinates'],
                        'row': cell['row'],
                        'col': cell['col'],
                        'regions': {
                            'boxes': cell_regions['boxes'],
                            'boxes_original': original_boxes,
                            'classes': cell_regions['classes'],
                            'scores': cell_regions['scores'],
                            'class_names': cell_regions['class_names']
                        }
                    })
            
            except Exception as e:
                logger.error(f"Error processing grid cell {cell_filename}: {str(e)}")
        
        # Save all grid information to the main JSON folder if provided
        if main_json_folder and all_grid_info['cells']:
            grid_config_str = f"{rows}x{cols}"
            main_json_path = os.path.join(main_json_folder, f"{base_filename}_grid_{grid_config_str}.json")
            with open(main_json_path, 'w') as f:
                json.dump(all_grid_info, f, indent=2)
            logger.info(f"Saved grid information to {main_json_path}")
        
        return True
    
    except Exception as e:
        logger.error(f"Error processing grid for {os.path.basename(image_path)}: {str(e)}")
        return False

def parse_grid_configs(grid_str):
    """
    Parse a comma-separated string of grid configurations.
    
    Args:
        grid_str (str): String in the format "NxM,AxB,..."
        
    Returns:
        list: List of tuples (rows, cols)
    """
    grid_configs = []
    
    try:
        if grid_str:
            configs = grid_str.split(',')
            
            for config in configs:
                config = config.strip()
                if 'x' in config:
                    rows, cols = config.split('x')
                    grid_configs.append((int(rows), int(cols)))
    except ValueError as e:
        logger.error(f"Error parsing grid configuration: {str(e)}")
    
    return grid_configs

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Document Layout Analysis')
    parser.add_argument('--input_folder', required=True, help='Folder containing document images')
    parser.add_argument('--output_folder', required=True, help='Folder to save results')
    parser.add_argument('--conf_threshold', type=float, default=0.1, help='Confidence threshold for detection')
    parser.add_argument('--iou_threshold', type=float, default=0.45, help='IoU threshold for NMS')
    parser.add_argument('--device', choices=['cpu', 'cuda'], help='Device to use (default: auto-detect)')
    parser.add_argument('--model_path', help='Path to local model file (if not specified, will download from HF)')
    parser.add_argument('--skip_errors', action='store_true', help='Continue processing if an image fails')
    
    # Add grid processing arguments
    # Keep the old arguments for backward compatibility
    parser.add_argument('--rows', type=int, default=2, help='Number of rows for grid processing (default: 2)')
    parser.add_argument('--cols', type=int, default=2, help='Number of columns for grid processing (default: 2)')
    # Add new argument for multiple grid configurations
    parser.add_argument('--grids', type=str, default="2x2,3x3,4x4", help='Comma-separated grid configurations (e.g., "2x2,3x3,4x4")')
    parser.add_argument('--overlap', type=float, default=20.0, help='Percentage of overlap between grid cells (default: 20.0)')
    parser.add_argument('--disable_grid', action='store_true', help='Disable grid processing')
    
    args = parser.parse_args()
    
    # Create output folders
    json_folder = os.path.join(args.output_folder, 'json')
    viz_folder = os.path.join(args.output_folder, 'visualizations')
    
    os.makedirs(json_folder, exist_ok=True)
    os.makedirs(viz_folder, exist_ok=True)
    
    # Parse grid configurations
    grid_configs = []
    if not args.disable_grid:
        if args.grids:
            grid_configs = parse_grid_configs(args.grids)
            if not grid_configs:
                # Fallback to the old arguments if parsing failed
                grid_configs = [(args.rows, args.cols)]
                logger.info(f"Using fallback grid configuration: {args.rows}x{args.cols}")
            else:
                logger.info(f"Using grid configurations: {', '.join([f'{r}x{c}' for r, c in grid_configs])}")
        else:
            # Use the old arguments if --grids is not provided
            grid_configs = [(args.rows, args.cols)]
            logger.info(f"Using grid configuration: {args.rows}x{args.cols}")
    
    # Get image paths
    logger.info(f"Searching for images in {args.input_folder}")
    image_paths = get_image_paths(args.input_folder)
    
    if not image_paths:
        logger.error(f"No images found in {args.input_folder}")
        return
    
    logger.info(f"Found {len(image_paths)} images")
    
    # Initialize detector
    detector = YOLODocumentLayoutDetector(
        conf_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold,
        device=args.device,
        model_path=args.model_path
    )
    
    # Process each image
    processed_count = 0
    error_count = 0
    
    for image_path in tqdm(image_paths, desc="Processing images"):
        try:
            # Process the image without grid (original behavior)
            success = process_image(detector, image_path, json_folder, viz_folder)
            
            # Process with grid if enabled
            if grid_configs and success:
                for rows, cols in grid_configs:
                    grid_folder = os.path.join(args.output_folder, f'grid_{rows}x{cols}')
                    os.makedirs(grid_folder, exist_ok=True)
                    
                    grid_success = process_image_with_grid(
                        detector=detector,
                        image_path=image_path,
                        grid_folder=grid_folder,
                        rows=rows,
                        cols=cols,
                        overlap_percentage=args.overlap,
                        main_json_folder=json_folder  # Store grid info in the main JSON folder too
                    )
                    
                    if not grid_success:
                        logger.warning(f"Grid processing ({rows}x{cols}) failed for {os.path.basename(image_path)}")
            
            if success:
                processed_count += 1
            else:
                error_count += 1
                
        except Exception as e:
            error_count += 1
            logger.error(f"Error processing {os.path.basename(image_path)}: {str(e)}")
            if not args.skip_errors:
                logger.error("Stopping due to error. Use --skip_errors to continue despite errors.")
                break
    
    logger.info(f"Processing complete. Successfully processed {processed_count} images with {error_count} errors. Results saved to {args.output_folder}")

if __name__ == "__main__":
    main()