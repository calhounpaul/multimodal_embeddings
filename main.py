#!/usr/bin/env python3
"""
Main script for newspaper image analysis.
Coordinates the entire workflow: processing images, detecting regions,
creating cross-comparisons, and running queries.
"""

import os
import sys
import argparse
import torch
import numpy as np
from PIL import Image
import torchvision
import cv2
from huggingface_hub import snapshot_download
from tqdm import tqdm

from config import (
    IMAGE_FOLDER, OUTPUT_FOLDER, DB_FOLDER, TESTOUT_FOLDER, 
    CROSS_COMPARE_FOLDER, DEVICE, TEST_IMG, TEST_TEXT, CROSS_COMPARE_TOP_N,
    REGION_COMPARE_TOP_N, REGION_CACHE_FOLDER, REGION_OUTPUT_FOLDER,
    REGION_VISUALIZATION_FOLDER, REGION_COMPARISON_FOLDER,
    DOCLAYOUT_CONF_THRESHOLD, DOCLAYOUT_IOU_THRESHOLD, DOCLAYOUT_IMAGE_SIZE
)
from logger_setup import logger
from embedder import MmE5MllamaEmbedder
from image_utils import get_image_paths
from db_operations import initialize_db
from progress_tracker import load_progress
from image_processor import process_images
from cross_compare import create_cross_comparison
from demo_queries import run_demo_queries
from visualization import create_regions_visualization

def setup_model_paths():
    """Set up model paths and download models if needed."""
    # Create directory for downloaded model files
    model_cache_dir = os.path.join(os.path.dirname(__file__), "model_cache")
    os.makedirs(model_cache_dir, exist_ok=True)
    
    # Download DocLayout models using huggingface_hub
    logger.info("Downloading DocLayout model from Hugging Face Hub")
    try:
        model_dir = snapshot_download(
            'juliozhao/DocLayout-YOLO-DocStructBench', 
            local_dir=os.path.join(model_cache_dir, 'DocLayout-YOLO-DocStructBench')
        )
        logger.info(f"DocLayout model downloaded to {model_dir}")
        
        # Check for the specific model file
        model_path = os.path.join(model_dir, "doclayout_yolo_docstructbench_imgsz1024.pt")
        if os.path.exists(model_path):
            logger.info(f"Found model file at {model_path}")
        else:
            logger.error(f"Model file not found at {model_path}")
            
        return model_path
    except Exception as e:
        logger.error(f"Error downloading models: {e}")
        return None

class DocLayoutDetector:
    """
    Wrapper for the YOLO-based document layout detection model.
    Detects regions like titles, text blocks, figures, tables, etc. in document images.
    """

    # Class ID to region type mapping
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

    def __init__(self, model_path, conf_threshold=DOCLAYOUT_CONF_THRESHOLD, 
                 iou_threshold=DOCLAYOUT_IOU_THRESHOLD, device=None):
        """
        Initialize the document layout detector.
        
        Args:
            model_path: Path to the model file
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for non-maximum suppression
            device: Device to run the model on (cuda or cpu)
        """
        # Determine device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        logger.info(f"Initializing DocLayoutDetector on {self.device}")
        
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Create cache directory if it doesn't exist
        os.makedirs(REGION_CACHE_FOLDER, exist_ok=True)
        
        # Load the model
        try:
            from doclayout_yolo import YOLOv10
            self.model = YOLOv10(model_path)
            logger.info(f"DocLayout model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Error loading DocLayout model: {str(e)}")
            raise

    def detect_regions(self, image_path, force_recompute=False):
        """
        Detect regions in an image.
        
        Args:
            image_path: Path to the image file
            force_recompute: If True, recompute regions even if cached results exist
            
        Returns:
            dict: Dictionary containing bounding boxes, classes, and scores for detected regions
        """
        # Open the image
        try:
            image = Image.open(image_path)
            image_filename = os.path.basename(image_path)
            
            # Run detection
            logger.info(f"Detecting regions in {image_filename}")
            det_res = self.model.predict(
                image,
                imgsz=DOCLAYOUT_IMAGE_SIZE,
                conf=self.conf_threshold,
                device=self.device,
            )[0]
            
            # Extract detection results
            boxes = det_res.__dict__['boxes'].xyxy.cpu().numpy()
            classes = det_res.__dict__['boxes'].cls.cpu().numpy()
            scores = det_res.__dict__['boxes'].conf.cpu().numpy()
            
            # Apply non-maximum suppression
            tensor_boxes = torch.tensor(boxes)
            tensor_scores = torch.tensor(scores)
            
            if len(tensor_boxes) > 0:
                indices = torchvision.ops.nms(
                    boxes=tensor_boxes, 
                    scores=tensor_scores,
                    iou_threshold=self.iou_threshold
                ).numpy()
                
                boxes, scores, classes = boxes[indices], scores[indices], classes[indices]
                
                # Ensure boxes is 2D even if there's only one detection
                if len(boxes.shape) == 1:
                    boxes = np.expand_dims(boxes, 0)
                    scores = np.expand_dims(scores, 0)
                    classes = np.expand_dims(classes, 0)
            else:
                # No detections
                boxes = np.array([])
                scores = np.array([])
                classes = np.array([])
            
            # Convert numpy arrays to lists for JSON serialization
            regions = {
                'image_path': image_path,
                'image_size': {'width': image.width, 'height': image.height},
                'parameters': {
                    'conf_threshold': self.conf_threshold,
                    'iou_threshold': self.iou_threshold
                },
                'boxes': boxes.tolist(),
                'classes': classes.tolist(),
                'scores': scores.tolist(),
                'class_names': [self.ID_TO_NAMES[int(cls)] for cls in classes]
            }
            
            logger.info(f"Detected {len(boxes)} regions in {image_filename}")
            return regions
            
        except Exception as e:
            logger.error(f"Error detecting regions in {image_path}: {str(e)}")
            return None

    def get_region_image(self, image_path, box, padding=0):
        """
        Extract a region from an image using the bounding box.
        
        Args:
            image_path: Path to the image file
            box: Bounding box coordinates [x_min, y_min, x_max, y_max]
            padding: Optional padding to add around the region (in pixels)
            
        Returns:
            PIL.Image: The extracted region as a PIL Image
        """
        try:
            image = Image.open(image_path)
            x_min, y_min, x_max, y_max = map(int, box)
            
            # Add padding if specified
            if padding > 0:
                x_min = max(0, x_min - padding)
                y_min = max(0, y_min - padding)
                x_max = min(image.width, x_max + padding)
                y_max = min(image.height, y_max + padding)
            
            # Crop the region
            region_image = image.crop((x_min, y_min, x_max, y_max))
            return region_image
            
        except Exception as e:
            logger.error(f"Error extracting region from {image_path}: {str(e)}")
            return None

def process_image_regions(detector, image_path):
    """Process regions in a single image and visualize them."""
    # Detect regions
    regions = detector.detect_regions(image_path)
    
    if regions is None or not regions.get('boxes'):
        logger.warning(f"No regions detected in {os.path.basename(image_path)}")
        return
    
    # Create visualization output folder
    visualization_folder = os.path.join(OUTPUT_FOLDER, "region_visualizations")
    os.makedirs(visualization_folder, exist_ok=True)
    
    # Create visualization
    output_path = os.path.join(visualization_folder, f"{os.path.splitext(os.path.basename(image_path))[0]}_regions.jpg")
    visualization = create_regions_visualization(image_path, regions, output_path)
    
    logger.info(f"Created region visualization for {os.path.basename(image_path)} at {output_path}")
    
    return visualization

def main():
    """
    Main function that coordinates the newspaper image analysis workflow.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Newspaper Image Analysis')
    parser.add_argument('--test-image', default=TEST_IMG, help='Path to test image for regions')
    parser.add_argument('--conf-threshold', type=float, default=DOCLAYOUT_CONF_THRESHOLD, help='Confidence threshold for region detection')
    parser.add_argument('--iou-threshold', type=float, default=DOCLAYOUT_IOU_THRESHOLD, help='IoU threshold for NMS in region detection')
    parser.add_argument('--batch', action='store_true', help='Process all images in the folder')
    
    args = parser.parse_args()
    
    test_image_path = args.test_image
    conf_threshold = args.conf_threshold
    iou_threshold = args.iou_threshold
    batch_mode = args.batch
    
    logger.info("Starting newspaper image region detection")
    
    # Create necessary directories
    for folder in [
        OUTPUT_FOLDER, REGION_CACHE_FOLDER, REGION_OUTPUT_FOLDER, 
        REGION_VISUALIZATION_FOLDER
    ]:
        os.makedirs(folder, exist_ok=True)
    
    # Set up model paths and download if needed
    model_path = setup_model_paths()
    if not model_path:
        logger.error("Failed to set up model paths. Exiting.")
        return
    
    # Initialize document layout detector
    try:
        detector = DocLayoutDetector(
            model_path=model_path,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            device=DEVICE
        )
        logger.info("Document layout detector initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize document layout detector: {str(e)}")
        return
    
    if batch_mode:
        # Process all images in the folder
        logger.info(f"Processing all images in folder: {IMAGE_FOLDER}")
        
        if not os.path.exists(IMAGE_FOLDER):
            logger.error(f"Image folder '{IMAGE_FOLDER}' does not exist. Exiting.")
            return
            
        image_paths = get_image_paths(IMAGE_FOLDER)
        
        if not image_paths:
            logger.error("No image files found in the specified folder. Exiting.")
            return
        
        logger.info(f"Found {len(image_paths)} image files to process")
        
        # Process each image
        for idx, image_path in enumerate(tqdm(image_paths, desc="Processing images")):
            process_image_regions(detector, image_path)
            
            # Periodically log progress
            if (idx + 1) % 5 == 0 or idx == len(image_paths) - 1:
                logger.info(f"Progress: {idx + 1}/{len(image_paths)} images processed")
    else:
        # Process single test image
        if not os.path.exists(test_image_path):
            logger.error(f"Test image '{test_image_path}' does not exist. Exiting.")
            return
            
        logger.info(f"Processing single image: {test_image_path}")
        process_image_regions(detector, test_image_path)
    
    logger.info("Region detection completed successfully!")

if __name__ == "__main__":
    main()