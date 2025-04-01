#!/usr/bin/env python3
"""
Progress tracking functionality for newspaper image analysis.
Manages the tracking of processed images and cross-comparison progress.
"""

import os
import json
from config import (
    PROGRESS_FILE, CROSS_COMPARE_PROGRESS_FILE, 
    REGION_DETECTION_PROGRESS_FILE, REGION_EMBEDDING_PROGRESS_FILE,
    REGION_COMPARISON_PROGRESS_FILE
)
from logger_setup import logger

################################################################################
# Image Processing Progress Tracking
################################################################################

def load_progress():
    """Load the processing progress from a JSON file."""
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading progress file: {str(e)}")
            return {"processed_images": []}
    return {"processed_images": []}

def save_progress(progress_data):
    """Save the processing progress to a JSON file."""
    os.makedirs(os.path.dirname(PROGRESS_FILE), exist_ok=True)
    
    try:
        with open(PROGRESS_FILE, 'w') as f:
            json.dump(progress_data, f)
    except Exception as e:
        logger.error(f"Error saving progress file: {str(e)}")

def mark_image_as_processed(image_path):
    """Mark a specific image as processed in the progress tracker."""
    progress = load_progress()
    
    if image_path not in progress["processed_images"]:
        progress["processed_images"].append(image_path)
        save_progress(progress)

def is_image_processed(image_path):
    """Check if a specific image has already been processed."""
    progress = load_progress()
    return image_path in progress["processed_images"]

################################################################################
# Cross-Comparison Progress Tracking
################################################################################

def load_cross_compare_progress():
    """Load the cross-comparison progress from a JSON file."""
    if os.path.exists(CROSS_COMPARE_PROGRESS_FILE):
        try:
            with open(CROSS_COMPARE_PROGRESS_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading cross-compare progress file: {str(e)}")
            return {"completed_images": []}
    return {"completed_images": []}

def save_cross_compare_progress(progress_data):
    """Save the cross-comparison progress to a JSON file."""
    os.makedirs(os.path.dirname(CROSS_COMPARE_PROGRESS_FILE), exist_ok=True)
    
    try:
        with open(CROSS_COMPARE_PROGRESS_FILE, 'w') as f:
            json.dump(progress_data, f)
    except Exception as e:
        logger.error(f"Error saving cross-compare progress file: {str(e)}")

def mark_cross_compare_as_completed(image_path):
    """Mark a specific image as having completed cross-comparison."""
    progress = load_cross_compare_progress()
    
    if image_path not in progress["completed_images"]:
        progress["completed_images"].append(image_path)
        save_cross_compare_progress(progress)

def is_cross_compare_completed(image_path):
    """Check if cross-comparison has already been done for a specific image."""
    progress = load_cross_compare_progress()
    return image_path in progress["completed_images"]

################################################################################
# Region Detection Progress Tracking
################################################################################

def load_region_detection_progress():
    """Load the region detection progress from a JSON file."""
    if os.path.exists(REGION_DETECTION_PROGRESS_FILE):
        try:
            with open(REGION_DETECTION_PROGRESS_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading region detection progress file: {str(e)}")
            return {"completed_images": []}
    return {"completed_images": []}

def save_region_detection_progress(progress_data):
    """Save the region detection progress to a JSON file."""
    os.makedirs(os.path.dirname(REGION_DETECTION_PROGRESS_FILE), exist_ok=True)
    
    try:
        with open(REGION_DETECTION_PROGRESS_FILE, 'w') as f:
            json.dump(progress_data, f)
    except Exception as e:
        logger.error(f"Error saving region detection progress file: {str(e)}")

def mark_region_detection_as_completed(image_path):
    """Mark a specific image as having completed region detection."""
    progress = load_region_detection_progress()
    
    if image_path not in progress["completed_images"]:
        progress["completed_images"].append(image_path)
        save_region_detection_progress(progress)

def is_region_detection_completed(image_path):
    """Check if region detection has already been done for a specific image."""
    progress = load_region_detection_progress()
    return image_path in progress["completed_images"]

################################################################################
# Region Embedding Progress Tracking
################################################################################

def load_region_embedding_progress():
    """Load the region embedding progress from a JSON file."""
    if os.path.exists(REGION_EMBEDDING_PROGRESS_FILE):
        try:
            with open(REGION_EMBEDDING_PROGRESS_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading region embedding progress file: {str(e)}")
            return {"embedded_regions": []}
    return {"embedded_regions": []}

def save_region_embedding_progress(progress_data):
    """Save the region embedding progress to a JSON file."""
    os.makedirs(os.path.dirname(REGION_EMBEDDING_PROGRESS_FILE), exist_ok=True)
    
    try:
        with open(REGION_EMBEDDING_PROGRESS_FILE, 'w') as f:
            json.dump(progress_data, f)
    except Exception as e:
        logger.error(f"Error saving region embedding progress file: {str(e)}")

def mark_region_embedding_as_completed(region_id):
    """Mark a specific region as having completed embedding."""
    progress = load_region_embedding_progress()
    
    if region_id not in progress["embedded_regions"]:
        progress["embedded_regions"].append(region_id)
        save_region_embedding_progress(progress)

def is_region_embedding_completed(region_id):
    """Check if embedding has already been done for a specific region."""
    progress = load_region_embedding_progress()
    return region_id in progress["embedded_regions"]

################################################################################
# Region Comparison Progress Tracking
################################################################################

def load_region_comparison_progress():
    """Load the region comparison progress from a JSON file."""
    if os.path.exists(REGION_COMPARISON_PROGRESS_FILE):
        try:
            with open(REGION_COMPARISON_PROGRESS_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading region comparison progress file: {str(e)}")
            return {"completed_comparisons": []}
    return {"completed_comparisons": []}

def save_region_comparison_progress(progress_data):
    """Save the region comparison progress to a JSON file."""
    os.makedirs(os.path.dirname(REGION_COMPARISON_PROGRESS_FILE), exist_ok=True)
    
    try:
        with open(REGION_COMPARISON_PROGRESS_FILE, 'w') as f:
            json.dump(progress_data, f)
    except Exception as e:
        logger.error(f"Error saving region comparison progress file: {str(e)}")

def mark_region_comparison_as_completed(region_id):
    """Mark a specific region as having completed comparisons."""
    progress = load_region_comparison_progress()
    
    if region_id not in progress["completed_comparisons"]:
        progress["completed_comparisons"].append(region_id)
        save_region_comparison_progress(progress)

def is_region_comparison_completed(region_id):
    """Check if comparisons have already been done for a specific region."""
    progress = load_region_comparison_progress()
    return region_id in progress["completed_comparisons"]

################################################################################
# Orientation Correction Progress Tracking
################################################################################

def load_orientation_progress():
    """Load the orientation correction progress from a JSON file."""
    if os.path.exists(ORIENTATION_PROGRESS_FILE):
        try:
            with open(ORIENTATION_PROGRESS_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading orientation progress file: {str(e)}")
            return {"corrected_images": []}
    return {"corrected_images": []}

def save_orientation_progress(progress_data):
    """Save the orientation correction progress to a JSON file."""
    os.makedirs(os.path.dirname(ORIENTATION_PROGRESS_FILE), exist_ok=True)
    
    try:
        with open(ORIENTATION_PROGRESS_FILE, 'w') as f:
            json.dump(progress_data, f)
    except Exception as e:
        logger.error(f"Error saving orientation progress file: {str(e)}")

def mark_orientation_as_completed(image_path):
    """Mark a specific image as having completed orientation correction."""
    progress = load_orientation_progress()
    
    if image_path not in progress["corrected_images"]:
        progress["corrected_images"].append(image_path)
        save_orientation_progress(progress)

def is_orientation_completed(image_path):
    """Check if orientation correction has already been done for a specific image."""
    progress = load_orientation_progress()
    return image_path in progress["corrected_images"]