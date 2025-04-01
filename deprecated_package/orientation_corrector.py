#!/usr/bin/env python3
"""
Text orientation correction module for newspaper image analysis.
Uses advanced techniques to detect and correct subtle text orientation issues.
"""

import os
import cv2
import numpy as np
import pytesseract
from pytesseract import Output
import imutils
from PIL import Image
from tqdm import tqdm
import json

from logger_setup import logger
from image_utils import validate_image
from config import ORIENTATION_PROGRESS_FILE

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

class OrientationCorrector:
    """
    Handles text orientation detection and correction using multiple advanced techniques.
    """
    
    def __init__(self, output_folder=None, sensitivity_threshold=0.5, 
                 advanced_detection=True, debug=False):
        """
        Initialize the orientation corrector.
        
        Args:
            output_folder: Optional folder to save corrected images
            sensitivity_threshold: Minimum angle (in degrees) to trigger skew correction
            advanced_detection: Use advanced skew detection techniques
            debug: Enable debug logging and optional visualization
        """
        self.output_folder = output_folder
        self.sensitivity_threshold = sensitivity_threshold
        self.advanced_detection = advanced_detection
        self.debug = debug
        
        if output_folder:
            os.makedirs(output_folder, exist_ok=True)
            logger.info(f"Orientation corrector initialized with output folder: {output_folder}, "
                      f"sensitivity threshold: {sensitivity_threshold}°, "
                      f"advanced detection: {advanced_detection}")
        else:
            logger.info(f"Orientation corrector initialized (in-place correction), "
                      f"sensitivity threshold: {sensitivity_threshold}°")
    
    def detect_skew_tesseract(self, image_path):
        """
        Detect image skew using Tesseract's Orientation and Script Detection (OSD).
        
        Args:
            image_path: Path to the input image
            
        Returns:
            float: Detected rotation angle, or None if detection fails
        """
        try:
            # Read image and convert to RGB for Tesseract
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Failed to read image: {image_path}")
                return None
                
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Use Tesseract's OSD to detect orientation
            results = pytesseract.image_to_osd(rgb, output_type=Output.DICT)
            
            orientation = results["orientation"]
            rotation_needed = results["rotate"]
            
            logger.info(f"Tesseract OSD: Orientation {orientation}°, Rotation needed {rotation_needed}°")
            
            return rotation_needed
            
        except Exception as e:
            logger.error(f"Tesseract OSD error for {image_path}: {str(e)}")
            return None
    
    def detect_skew_opencv(self, image_path):
        """
        Detect text skew using OpenCV's advanced techniques with improved logic.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            float: Detected skew angle in degrees, or None if detection fails
        """
        try:
            # Read image in grayscale
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                logger.error(f"Failed to read image: {image_path}")
                return None
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(image, (5, 5), 0)
            
            # Apply adaptive thresholding to handle varying illumination
            thresh = cv2.adaptiveThreshold(
                blurred, 255, 
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 11, 2
            )
            
            # Detect edges
            edges = cv2.Canny(thresh, 50, 150, apertureSize=3)
            
            # Probabilistic Hough Line Transform with stricter parameters
            lines = cv2.HoughLinesP(
                edges, 1, np.pi/180, 
                threshold=100,  # Increased to reduce false positives
                minLineLength=min(image.shape[1]//2, 200),  # Minimum line length based on image width
                maxLineGap=10
            )
            
            if lines is None or len(lines) == 0:
                return None
                
            # Compute angle of detected lines
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                
                # Filter out extreme angles (likely false positives)
                if abs(angle) < 45:  # Only consider angles within ±45 degrees
                    angles.append(angle)
            
            if not angles:
                return None
                
            # Compute the most common angle using mode
            angles_array = np.array(angles)
            median_angle = np.median(angles_array)
            std_angle = np.std(angles_array)
            
            # Filter out inconsistent angles
            if std_angle > 10:  # If angle variation is too high, the detection is unreliable
                return None
                
            logger.info(f"OpenCV Skew Detection: Median Angle {median_angle}°, "
                      f"Angle Variation {std_angle}°")
            
            return median_angle
            
        except Exception as e:
            logger.error(f"OpenCV skew detection error for {image_path}: {str(e)}")
            return None
    
    def correct_orientation(self, image_path, save_corrected=True, force_reprocess=False):
        """
        Correct the orientation of an image using multiple detection techniques.
        
        Args:
            image_path: Path to the input image
            save_corrected: Whether to save the corrected image
            force_reprocess: Force reprocessing even if already completed
            
        Returns:
            str: Path to the corrected image
        """
        # Skip if already processed and not forced to reprocess
        if is_orientation_completed(image_path) and not force_reprocess:
            logger.info(f"Skipping orientation correction for {os.path.basename(image_path)}, already processed")
            
            # If we have an output folder, return the path to the corrected image
            if self.output_folder:
                corrected_path = os.path.join(self.output_folder, os.path.basename(image_path))
                if os.path.exists(corrected_path):
                    return corrected_path
            
            return image_path
        
        # Validate input image
        if not validate_image(image_path):
            logger.error(f"Invalid image for orientation correction: {image_path}")
            return image_path
        
        # Detect skew using multiple techniques
        tesseract_angle = self.detect_skew_tesseract(image_path)
        opencv_angle = self.detect_skew_opencv(image_path) if self.advanced_detection else None
        
        # Combine detection results with preference for OpenCV if both are available
        if opencv_angle is not None:
            detected_angle = opencv_angle
        else:
            detected_angle = tesseract_angle
        
        # No valid rotation detected
        if detected_angle is None:
            logger.info(f"No significant skew detected for {os.path.basename(image_path)}")
            mark_orientation_as_completed(image_path)
            return image_path
        
        # Apply sensitivity threshold
        if abs(detected_angle) < self.sensitivity_threshold:
            logger.info(f"Skew {detected_angle}° is below sensitivity threshold of {self.sensitivity_threshold}°")
            mark_orientation_as_completed(image_path)
            return image_path
        
        # Read and rotate the image
        try:
            image = cv2.imread(image_path)
            
            # Use imutils for bounded rotation to prevent cropping
            rotated = imutils.rotate_bound(image, angle=-detected_angle)
            
            # Save the rotated image
            if save_corrected:
                if self.output_folder:
                    # Save to output folder
                    output_path = os.path.join(self.output_folder, os.path.basename(image_path))
                    cv2.imwrite(output_path, rotated)
                    logger.info(f"Corrected {os.path.basename(image_path)} with {detected_angle}° rotation")
                    mark_orientation_as_completed(image_path)
                    return output_path
                else:
                    # Save in-place (overwrite original)
                    cv2.imwrite(image_path, rotated)
                    logger.info(f"Overwrote {image_path} with {detected_angle}° rotation")
                    mark_orientation_as_completed(image_path)
                    return image_path
            else:
                logger.info(f"Corrected {os.path.basename(image_path)} with {detected_angle}° rotation (not saved)")
                mark_orientation_as_completed(image_path)
                return image_path
            
        except Exception as e:
            logger.error(f"Error correcting orientation for {image_path}: {str(e)}")
            return image_path

def batch_correct_orientation(image_paths, output_folder, batch_size=8):
    """
    Correct orientation for a batch of images.
    
    Args:
        image_paths: List of paths to images
        output_folder: Folder to save corrected images
        batch_size: Number of images to process in parallel
        
    Returns:
        List of paths to corrected images
    """
    os.makedirs(output_folder, exist_ok=True)
    corrector = OrientationCorrector(output_folder=output_folder)
    
    corrected_paths = []
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Correcting orientation"):
        batch = image_paths[i:i+batch_size]
        for image_path in batch:
            try:
                corrected_path = corrector.correct_orientation(image_path)
                corrected_paths.append(corrected_path)
            except Exception as e:
                logger.error(f"Error correcting orientation for {image_path}: {str(e)}")
                corrected_paths.append(image_path)  # Use original if correction fails
    
    return corrected_paths