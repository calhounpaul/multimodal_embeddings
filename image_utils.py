#!/usr/bin/env python3
"""
Image processing utilities for newspaper image analysis.
Provides functions for image validation, resizing, and path handling.
"""

import os
import glob
from PIL import Image

from logger_setup import logger

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

def resize_image_if_needed(image_path, output_path, max_size):
    """Resize image if it exceeds maximum dimensions, and save to output path."""
    try:
        image = Image.open(image_path)
        
        if image.size[0] > max_size or image.size[1] > max_size:
            scale_factor = max_size / max(image.size)
            new_size = (int(image.size[0] * scale_factor), int(image.size[1] * scale_factor))
            image = image.resize(new_size, Image.LANCZOS)
            logger.info(f"Resized image {image_path} to {new_size}")
        
        # Save the image (either resized or original)
        image.save(output_path)
        return True
    except Exception as e:
        logger.error(f"Error resizing/copying image {image_path}: {str(e)}")
        return False