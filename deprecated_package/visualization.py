#!/usr/bin/env python3
"""
Visualization utilities for newspaper image analysis.
Provides functions for visualizing document layout detection results.
"""

import os
import numpy as np
import cv2
from PIL import Image

def colormap(N=256, normalized=False):
    """
    Generate the color map.
    
    Args:
        N (int): Number of labels (default is 256).
        normalized (bool): If True, return colors normalized to [0, 1]. Otherwise, return [0, 255].
        
    Returns:
        np.ndarray: Color map array of shape (N, 3).
    """
    def bitget(byteval, idx):
        """
        Get the bit value at the specified index.
        
        Args:
            byteval (int): The byte value.
            idx (int): The index of the bit.
            
        Returns:
            int: The bit value (0 or 1).
        """
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

def visualize_bbox(image_path, bboxes, classes, scores, id_to_names, alpha=0.3, output_path=None):
    """
    Visualize layout detection results on an image.
    
    Args:
        image_path (str or PIL.Image): Path to the input image or a PIL Image object.
        bboxes (list): List of bounding boxes, each represented as [x_min, y_min, x_max, y_max].
        classes (list): List of class IDs corresponding to the bounding boxes.
        scores (list): List of confidence scores for each bounding box.
        id_to_names (dict): Dictionary mapping class IDs to class names.
        alpha (float): Transparency factor for the filled color (default is 0.3).
        output_path (str): Optional path to save the output image.
        
    Returns:
        np.ndarray: Image with visualized layout detection results.
    """
    # Check if image_path is a PIL.Image.Image object or numpy array
    if isinstance(image_path, Image.Image):
        image = np.array(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV
    elif isinstance(image_path, np.ndarray):
        image = image_path.copy()
        if image.shape[2] == 3:  # Check if RGB
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        image = cv2.imread(image_path)
    
    overlay = image.copy()
    
    cmap = colormap(N=len(id_to_names), normalized=False)
    
    # Iterate over each bounding box
    for i, bbox in enumerate(bboxes):
        x_min, y_min, x_max, y_max = map(int, bbox)
        class_id = int(classes[i])
        class_name = id_to_names[class_id]
        
        text = f"{class_name}: {scores[i]:.3f}"
        
        # Ensure class_id is within the bounds of cmap
        color = tuple(int(c) for c in cmap[class_id % len(cmap)])
        cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), color, -1)
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
        
        # Add the class name with a background rectangle
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.rectangle(image, (x_min, y_min - text_height - baseline), (x_min + text_width, y_min), color, -1)
        cv2.putText(image, text, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    
    # Blend the overlay with the original image
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    
    # Save the image if output_path is specified
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, image)
    
    return image

def create_regions_visualization(image_path, regions, output_path=None):
    """
    Create a visualization of detected regions in an image.
    
    Args:
        image_path (str): Path to the input image.
        regions (dict): Dictionary containing regions data (from DocLayoutDetector).
        output_path (str): Optional path to save the output image.
        
    Returns:
        np.ndarray: Image with visualized regions.
    """
    if regions is None:
        return None
        
    # Extract region data
    boxes = regions.get('boxes', [])
    classes = regions.get('classes', [])
    scores = regions.get('scores', [])
    
    # Get the ID to names mapping from the regions or use a default
    id_to_names = {}
    for i, class_name in enumerate(regions.get('class_names', [])):
        class_id = int(classes[i])
        id_to_names[class_id] = class_name
    
    # If id_to_names is empty, use the default mapping
    if not id_to_names:
        id_to_names = {
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
    
    # Visualize the regions
    return visualize_bbox(image_path, boxes, classes, scores, id_to_names, alpha=0.3, output_path=output_path)

def create_region_comparison_visualization(source_image, source_box, target_image, target_box, 
                                          similarity_score, output_path=None):
    """
    Create a visualization comparing two regions from different images.
    
    Args:
        source_image (str): Path to the source image.
        source_box (list or dict): Bounding box for the region in the source image.
        target_image (str): Path to the target image.
        target_box (list or dict): Bounding box for the region in the target image.
        similarity_score (float): Similarity score between the two regions.
        output_path (str): Optional path to save the output visualization.
        
    Returns:
        np.ndarray: Comparison visualization image.
    """
    # Load images
    try:
        src_img = cv2.imread(source_image)
        tgt_img = cv2.imread(target_image)
        
        if src_img is None or tgt_img is None:
            return None
            
        # Extract regions - handle different formats for boxes
        if isinstance(source_box, list) or isinstance(source_box, tuple):
            sx_min, sy_min, sx_max, sy_max = map(int, source_box)
        elif isinstance(source_box, dict):
            sx_min = int(source_box.get('box_x_min', 0))
            sy_min = int(source_box.get('box_y_min', 0))
            sx_max = int(source_box.get('box_x_max', 0))
            sy_max = int(source_box.get('box_y_max', 0))
        else:
            # Try to parse from string
            try:
                sx_min, sy_min, sx_max, sy_max = map(int, str(source_box).split(','))
            except:
                print(f"Couldn't parse source_box: {source_box}")
                return None
        
        if isinstance(target_box, list) or isinstance(target_box, tuple):
            tx_min, ty_min, tx_max, ty_max = map(int, target_box)
        elif isinstance(target_box, dict):
            tx_min = int(target_box.get('box_x_min', 0))
            ty_min = int(target_box.get('box_y_min', 0))
            tx_max = int(target_box.get('box_x_max', 0))
            ty_max = int(target_box.get('box_y_max', 0))
        else:
            # Try to parse from string
            try:
                tx_min, ty_min, tx_max, ty_max = map(int, str(target_box).split(','))
            except:
                print(f"Couldn't parse target_box: {target_box}")
                return None
        
        src_region = src_img[sy_min:sy_max, sx_min:sx_max]
        tgt_region = tgt_img[ty_min:ty_max, tx_min:tx_max]
        
        # Draw rectangle on source and target images
        cv2.rectangle(src_img, (sx_min, sy_min), (sx_max, sy_max), (0, 255, 0), 3)
        cv2.rectangle(tgt_img, (tx_min, ty_min), (tx_max, ty_max), (0, 255, 0), 3)
        
        # Resize images to have the same height
        height = 600  # Fixed height for display
        src_aspect = src_img.shape[1] / src_img.shape[0]
        tgt_aspect = tgt_img.shape[1] / tgt_img.shape[0]
        
        src_display = cv2.resize(src_img, (int(height * src_aspect), height))
        tgt_display = cv2.resize(tgt_img, (int(height * tgt_aspect), height))
        
        # Resize regions for inset display
        src_region_display = cv2.resize(src_region, (300, 300)) if src_region.size > 0 else np.zeros((300, 300, 3), dtype=np.uint8)
        tgt_region_display = cv2.resize(tgt_region, (300, 300)) if tgt_region.size > 0 else np.zeros((300, 300, 3), dtype=np.uint8)
        
        # Create a composite image
        total_width = src_display.shape[1] + tgt_display.shape[1]
        composite = np.zeros((height + 350, total_width, 3), dtype=np.uint8)
        
        # Add images to composite
        composite[:height, :src_display.shape[1]] = src_display
        composite[:height, src_display.shape[1]:] = tgt_display
        
        # Add region close-ups
        composite[height+25:height+325, (total_width//2)-325:total_width//2-25] = src_region_display
        composite[height+25:height+325, (total_width//2)+25:(total_width//2)+325] = tgt_region_display
        
        # Add similarity score
        text = f"Similarity Score: {similarity_score:.4f}"
        cv2.putText(composite, text, (total_width//2-150, height+345), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Add source and target labels
        cv2.putText(composite, "Source Image", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(composite, "Target Image", (src_display.shape[1] + 20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(composite, "Source Region", ((total_width//2)-325+20, height+50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(composite, "Target Region", ((total_width//2)+25+20, height+50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Save if output path is specified
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, composite)
            
        return composite
        
    except Exception as e:
        print(f"Error creating comparison visualization: {str(e)}")
        return None