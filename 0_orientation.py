#!/usr/bin/env python3
"""
Newspaper Image Orientation Corrector

This script corrects the orientation of newspaper images in a folder and saves
the corrected versions to an output folder. It uses a combination of Tesseract OCR
and OpenCV for detecting and correcting the orientation.

Usage:
    python correct_orientation.py input_folder output_folder

Dependencies:
    - OpenCV (cv2)
    - NumPy
    - Pytesseract
    - PIL (Pillow)
    - imutils
    - tqdm
"""

import os
import sys
import cv2
import numpy as np
import pytesseract
from pytesseract import Output
import imutils
from PIL import Image
import glob
import argparse
import logging
import shutil
from tqdm import tqdm
import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def get_image_paths(image_folder):
    """Get all image paths from the specified folder."""
    # Accept common image formats
    image_extensions = ['.jpg', '.jpeg', '.png', '.webp', '.tiff', '.tif', '.bmp']
    
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(image_folder, f"*{ext}")))
        image_paths.extend(glob.glob(os.path.join(image_folder, f"*{ext.upper()}")))
    
    # Return sorted paths - important for consistency
    return sorted(image_paths)

def validate_image(image_path):
    """Validate if the image can be opened and is valid."""
    try:
        with Image.open(image_path) as img:
            # Force loading the image data
            img.verify()
            return True
    except Exception as e:
        logger.error(f"Invalid image file {image_path}: {str(e)}")
        return False

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
    
    def correct_orientation(self, image_path, save_corrected=True):
        """
        Correct the orientation of an image using multiple detection techniques.
        
        Args:
            image_path: Path to the input image
            save_corrected: Whether to save the corrected image
            
        Returns:
            str: Path to the corrected image
        """
        # Validate input image
        if not validate_image(image_path):
            logger.error(f"Invalid image for orientation correction: {image_path}")
            return image_path
        
        # Always define the output path
        output_path = image_path
        if save_corrected and self.output_folder:
            output_path = os.path.join(self.output_folder, os.path.basename(image_path))
        
        # Detect skew using multiple techniques
        tesseract_angle = self.detect_skew_tesseract(image_path)
        opencv_angle = self.detect_skew_opencv(image_path) if self.advanced_detection else None
        
        # Combine detection results with preference for OpenCV if both are available
        if opencv_angle is not None:
            detected_angle = opencv_angle
        else:
            detected_angle = tesseract_angle
        
        # No valid rotation detected or angle below threshold
        if detected_angle is None:
            logger.info(f"No significant skew detected for {os.path.basename(image_path)}")
            # Copy the original file to output if no rotation needed
            if save_corrected and self.output_folder:
                try:
                    shutil.copy2(image_path, output_path)
                    logger.info(f"Copied {os.path.basename(image_path)} to output (no rotation needed)")
                except Exception as e:
                    logger.error(f"Error copying {image_path}: {str(e)}")
            return output_path
        
        # Apply sensitivity threshold
        if abs(detected_angle) < self.sensitivity_threshold:
            logger.info(f"Skew {detected_angle}° is below sensitivity threshold of {self.sensitivity_threshold}°")
            # Copy the original file to output if angle below threshold
            if save_corrected and self.output_folder:
                try:
                    shutil.copy2(image_path, output_path)
                    logger.info(f"Copied {os.path.basename(image_path)} to output (angle below threshold)")
                except Exception as e:
                    logger.error(f"Error copying {image_path}: {str(e)}")
            return output_path
        
        # Read and rotate the image
        try:
            image = cv2.imread(image_path)
            
            # Use imutils for bounded rotation to prevent cropping
            rotated = imutils.rotate_bound(image, angle=-detected_angle)
            
            # Save the rotated image
            if save_corrected and self.output_folder:
                cv2.imwrite(output_path, rotated)
                logger.info(f"Corrected {os.path.basename(image_path)} with {detected_angle}° rotation")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error correcting orientation for {image_path}: {str(e)}")
            # Attempt to copy the original as fallback
            if save_corrected and self.output_folder:
                try:
                    shutil.copy2(image_path, output_path)
                    logger.info(f"Copied original {os.path.basename(image_path)} after rotation failure")
                except Exception as copy_error:
                    logger.error(f"Error copying {image_path} after rotation failure: {str(copy_error)}")
            return output_path

def batch_correct_orientation(image_paths, output_folder, batch_size=8, 
                              sensitivity_threshold=0.5, advanced_detection=True):
    """
    Correct orientation for a batch of images.
    
    Args:
        image_paths: List of paths to images
        output_folder: Folder to save corrected images
        batch_size: Number of images to process in parallel
        sensitivity_threshold: Minimum angle to trigger correction
        advanced_detection: Whether to use advanced skew detection
        
    Returns:
        List of paths to corrected images
    """
    os.makedirs(output_folder, exist_ok=True)
    corrector = OrientationCorrector(
        output_folder=output_folder,
        sensitivity_threshold=sensitivity_threshold,
        advanced_detection=advanced_detection
    )
    
    corrected_paths = []
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Correcting orientation"):
        batch = image_paths[i:i+batch_size]
        for image_path in batch:
            try:
                corrected_path = corrector.correct_orientation(image_path)
                corrected_paths.append(corrected_path)
            except Exception as e:
                logger.error(f"Error correcting orientation for {image_path}: {str(e)}")
                # Ensure file is copied even on exception
                output_path = os.path.join(output_folder, os.path.basename(image_path))
                try:
                    shutil.copy2(image_path, output_path)
                    logger.info(f"Copied original {os.path.basename(image_path)} after exception")
                    corrected_paths.append(output_path)
                except Exception as copy_error:
                    logger.error(f"Error copying {image_path} after exception: {str(copy_error)}")
                    corrected_paths.append(image_path)  # Use original if copy fails
    
    return corrected_paths

def main():
    """Main function to handle command line arguments and process images."""
    parser = argparse.ArgumentParser(
        description="Correct the orientation of newspaper images in a folder."
    )
    parser.add_argument("input_folder", help="Path to the folder containing input images")
    parser.add_argument("output_folder", help="Path to the folder where corrected images will be saved")
    parser.add_argument("--sensitivity", type=float, default=0.5, 
                       help="Minimum angle (in degrees) to trigger correction (default: 0.5)")
    parser.add_argument("--batch-size", type=int, default=8,
                       help="Number of images to process in parallel (default: 8)")
    parser.add_argument("--no-advanced", action="store_false", dest="advanced",
                       help="Disable advanced skew detection techniques")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode with additional logging")
    
    args = parser.parse_args()
    
    # Configure logging level based on debug flag
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    # Get input image paths
    input_folder = os.path.abspath(args.input_folder)
    output_folder = os.path.abspath(args.output_folder)
    
    if not os.path.exists(input_folder):
        logger.error(f"Input folder does not exist: {input_folder}")
        return 1
    
    logger.info(f"Processing images from: {input_folder}")
    logger.info(f"Saving all images to: {output_folder}")
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all image paths
    image_paths = get_image_paths(input_folder)
    
    if not image_paths:
        logger.warning(f"No images found in: {input_folder}")
        return 0
    
    logger.info(f"Found {len(image_paths)} images to process")
    
    # Process images in batches
    start_time = datetime.datetime.now()
    corrected_paths = batch_correct_orientation(
        image_paths=image_paths,
        output_folder=output_folder,
        batch_size=args.batch_size,
        sensitivity_threshold=args.sensitivity,
        advanced_detection=args.advanced
    )
    
    elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
    logger.info(f"Processed {len(image_paths)} images in {elapsed_time:.2f} seconds")
    
    # Count output files
    output_files = len([f for f in os.listdir(output_folder) if os.path.isfile(os.path.join(output_folder, f))])
    logger.info(f"Output folder contains {output_files} images")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())