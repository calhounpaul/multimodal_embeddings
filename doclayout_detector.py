#!/usr/bin/env python3
"""
Updated document layout detection script for newspaper image analysis.
Loads the YOLO-based DocLayout model using Hugging Face Hub with the correct method.
"""

import os
import json
import torch
import numpy as np
import torchvision
from PIL import Image
from huggingface_hub import login

from logger_setup import logger
from progress_tracker import (
    mark_region_detection_as_completed, is_region_detection_completed
)
from config import (
    REGION_CACHE_FOLDER, DOCLAYOUT_CONF_THRESHOLD,
    DOCLAYOUT_IOU_THRESHOLD, DOCLAYOUT_IMAGE_SIZE
)

# In doclayout_detector.py, modify the DocLayoutDetector class initialization

class DocLayoutDetector:
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

    def __init__(self, conf_threshold=DOCLAYOUT_CONF_THRESHOLD, 
                 iou_threshold=DOCLAYOUT_IOU_THRESHOLD, device=None,
                 repo_id="juliozhao/DocLayout-YOLO-DocStructBench", 
                 model_path=None):
        """
        Initialize the document layout detector.
        """
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        logger.info(f"Initializing DocLayoutDetector on {self.device}")

        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        os.makedirs(REGION_CACHE_FOLDER, exist_ok=True)

        try:
            token = os.environ.get("HF_TOKEN")
            if token:
                login(token=token)
                logger.info("Logged in to Hugging Face Hub using token from environment")

            from doclayout_yolo import YOLOv10
            
            # Use the actual model path if provided, otherwise load from repo
            if model_path:
                # Use a local model file
                script_dir = os.path.dirname(os.path.abspath(__file__))
                absolute_model_path = os.path.join(script_dir, model_path)
                logger.info(f"Loading DocLayout model from local path: {absolute_model_path}")
                
                if not os.path.exists(absolute_model_path):
                    logger.error(f"Model file not found at {absolute_model_path}")
                    fallback_path = os.path.join(script_dir, "model_cache/DocLayout-YOLO-DocStructBench/doclayout_yolo_docstructbench_imgsz1024.pt")
                    if os.path.exists(fallback_path):
                        logger.info(f"Using fallback model path: {fallback_path}")
                        absolute_model_path = fallback_path
                    else:
                        raise FileNotFoundError(f"Neither specified model path nor fallback path exists")
                        
                self.model = YOLOv10(absolute_model_path)
            else:
                # Use the Hugging Face hub model
                self.model = YOLOv10.from_pretrained(repo_id)
                
            logger.info(f"DocLayout model loaded successfully")

        except Exception as e:
            logger.error(f"Error loading DocLayout model: {str(e)}")
            raise

    def detect_regions(self, image_path, force_recompute=False):
        """
        Detect regions in an image and cache the results.
        
        Args:
            image_path: Path to the image file
            force_recompute: If True, recompute regions even if cached results exist
            
        Returns:
            dict: Dictionary containing bounding boxes, classes, and scores for detected regions
        """
        image_filename = os.path.basename(image_path)
        cache_filename = f"{os.path.splitext(image_filename)[0]}_conf{self.conf_threshold}_iou{self.iou_threshold}.json"
        cache_path = os.path.join(REGION_CACHE_FOLDER, cache_filename)

        if os.path.exists(cache_path) and not force_recompute:
            try:
                with open(cache_path, 'r') as f:
                    regions = json.load(f)
                logger.info(f"Loaded cached regions for {image_filename}")
                return regions
            except Exception as e:
                logger.warning(f"Error loading cached regions: {str(e)}. Recomputing.")

        try:
            image = Image.open(image_path)
            det_res = self.model.predict(
                image,
                imgsz=DOCLAYOUT_IMAGE_SIZE,
                conf=self.conf_threshold,
                device=self.device,
            )[0]

            boxes = det_res.__dict__['boxes'].xyxy.cpu().numpy()
            classes = det_res.__dict__['boxes'].cls.cpu().numpy()
            scores = det_res.__dict__['boxes'].conf.cpu().numpy()

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

            with open(cache_path, 'w') as f:
                json.dump(regions, f)

            logger.info(f"Detected {len(boxes)} regions in {image_filename}")
            return regions

        except Exception as e:
            logger.error(f"Error detecting regions in {image_filename}: {str(e)}")
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


def process_image_regions(detector, image_path, force_recompute=False):
    """
    Process an image to detect regions and return the results.
    """
    if is_region_detection_completed(image_path) and not force_recompute:
        logger.info(f"Skipping region detection for {image_path}, already completed")
        return

    regions = detector.detect_regions(image_path, force_recompute)

    if regions:
        mark_region_detection_as_completed(image_path)

    return regions
