#!/usr/bin/env python3
"""
Complete workflow for newspaper image analysis.
Handles the entire process: detecting regions, embedding them, and clustering.
"""

import os
import argparse
import torch
from tqdm import tqdm

from config import (
    IMAGE_FOLDER, OUTPUT_FOLDER, DB_FOLDER, DEVICE,
    REGION_CACHE_FOLDER, REGION_OUTPUT_FOLDER,
    DOCLAYOUT_CONF_THRESHOLD, DOCLAYOUT_IOU_THRESHOLD
)
from logger_setup import logger
from embedder import MmE5MllamaEmbedder
from image_utils import get_image_paths
from db_operations import initialize_db
from doclayout_detector import DocLayoutDetector, process_image_regions
from region_processor import RegionProcessor
from visualization import create_regions_visualization

def main():
    """
    Main function that coordinates the complete newspaper image analysis workflow.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Complete Newspaper Image Analysis Workflow')
    parser.add_argument('--conf-threshold', type=float, default=DOCLAYOUT_CONF_THRESHOLD, 
                        help='Confidence threshold for region detection')
    parser.add_argument('--iou-threshold', type=float, default=DOCLAYOUT_IOU_THRESHOLD, 
                        help='IoU threshold for NMS in region detection')
    parser.add_argument('--skip-clustering', action='store_true', 
                        help='Skip the final clustering step')
    
    args = parser.parse_args()
    
    conf_threshold = args.conf_threshold
    iou_threshold = args.iou_threshold
    run_clustering = not args.skip_clustering
    
    logger.info("Starting complete newspaper image analysis workflow")
    
    # Create necessary directories
    for folder in [
        OUTPUT_FOLDER, DB_FOLDER, REGION_CACHE_FOLDER, REGION_OUTPUT_FOLDER
    ]:
        os.makedirs(folder, exist_ok=True)
    
    # Step 1: Get image paths
    logger.info(f"Looking for images in {IMAGE_FOLDER}")
    image_paths = get_image_paths(IMAGE_FOLDER)
    
    if not image_paths:
        logger.error(f"No image files found in {IMAGE_FOLDER}. Exiting.")
        return
    
    logger.info(f"Found {len(image_paths)} newspaper images to analyze")
    
    # Step 2: Initialize document layout detector
    # In complete_workflow.py, in the main function

    # Initialize document layout detector
    logger.info("Initializing document layout detector")
    try:
        # Specify the path to the existing model file
        model_path = "model_cache/DocLayout-YOLO-DocStructBench/doclayout_yolo_docstructbench_imgsz1024.pt"
        
        detector = DocLayoutDetector(
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            device=DEVICE,
            model_path=model_path
        )
        logger.info("Document layout detector initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize document layout detector: {str(e)}")
        return
    
    # Step 3: Initialize the embedder
    logger.info("Initializing mmE5-mllama embedder")
    try:
        embedder = MmE5MllamaEmbedder(device=DEVICE)
        logger.info("Embedder initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize embedder: {str(e)}")
        return
    
    # Step 4: Initialize database connection
    logger.info("Initializing database")
    try:
        chroma_client, collection = initialize_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {str(e)}")
        return
    
    # Step 5: Process regions for all images
    logger.info("Processing regions for all images")
    # First detect regions for all images
    for idx, image_path in enumerate(tqdm(image_paths, desc="Detecting regions")):
        regions = process_image_regions(detector, image_path)
        
        # Create visualization (optional)
        visualization_folder = os.path.join(OUTPUT_FOLDER, "region_visualizations")
        os.makedirs(visualization_folder, exist_ok=True)
        output_path = os.path.join(visualization_folder, f"{os.path.splitext(os.path.basename(image_path))[0]}_regions.jpg")
        create_regions_visualization(image_path, regions, output_path)
        
        # Periodically log progress
        if (idx + 1) % 5 == 0 or idx == len(image_paths) - 1:
            logger.info(f"Region detection progress: {idx + 1}/{len(image_paths)} images")
    
    # Step 6: Initialize region processor and embed regions
    logger.info("Embedding and storing regions")
    region_processor = RegionProcessor(embedder, collection, detector)
    regions_processed = region_processor.process_regions(image_paths)
    logger.info(f"Processed {regions_processed} regions from {len(image_paths)} images")
    
    # Step 7: Run weighted region clustering if requested
    if run_clustering:
        logger.info("Running weighted region clustering")
        try:
            from weighted_region_clustering import (
                compute_image_similarity_matrix, cluster_images, 
                create_html_report, WEIGHTED_CLUSTERING_FOLDER
            )
            
            # Create output directory
            os.makedirs(WEIGHTED_CLUSTERING_FOLDER, exist_ok=True)
            
            # Compute similarity matrix
            similarity_matrix, image_names = compute_image_similarity_matrix(collection, image_paths)
            
            if similarity_matrix is None or image_names is None:
                logger.error("Failed to compute similarity matrix. Skipping clustering.")
            else:
                # Cluster images
                clustering_results = cluster_images(similarity_matrix, image_names)
                
                # Create HTML report
                html_dir = os.path.join(WEIGHTED_CLUSTERING_FOLDER, "html_report")
                report_path = create_html_report(
                    similarity_matrix, 
                    image_names, 
                    clustering_results, 
                    html_dir
                )
                logger.info(f"HTML clustering report created at {report_path}")
        except Exception as e:
            logger.error(f"Error during clustering: {str(e)}")
    
    logger.info("Complete newspaper image analysis workflow finished successfully!")

if __name__ == "__main__":
    main()