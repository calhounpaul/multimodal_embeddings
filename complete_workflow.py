#!/usr/bin/env python3
"""
Complete workflow for newspaper image analysis.
Handles the entire process: correcting orientation, detecting regions, embedding them, and clustering.
"""

import os
import argparse
import torch
import logging
import shutil
from tqdm import tqdm

from config import (
    IMAGE_FOLDER, OUTPUT_FOLDER, DB_FOLDER, DEVICE,
    REGION_CACHE_FOLDER, REGION_OUTPUT_FOLDER, PROGRESS_FILE,
    CROSS_COMPARE_PROGRESS_FILE, REGION_DETECTION_PROGRESS_FILE,
    REGION_EMBEDDING_PROGRESS_FILE, REGION_COMPARISON_PROGRESS_FILE,
    DOCLAYOUT_CONF_THRESHOLD, DOCLAYOUT_IOU_THRESHOLD,
    ORIENTATION_OUTPUT_FOLDER, ORIENTATION_CORRECTION_ENABLED
)
from logger_setup import logger
from embedder import MmE5MllamaEmbedder
from image_utils import get_image_paths
from db_operations import initialize_db
from doclayout_detector import DocLayoutDetector, process_image_regions
from region_processor import RegionProcessor
from visualization import create_regions_visualization
from orientation_corrector import OrientationCorrector, batch_correct_orientation

def check_regions_exist(collection):
    """Check if regions exist in the database."""
    try:
        # Query for a single region to check if any exist
        results = collection.get(
            where={"is_region": {"$eq": True}},
            limit=1
        )
        return results and len(results["ids"]) > 0
    except Exception as e:
        logger.error(f"Error checking for regions: {str(e)}")
        return False

def reset_workflow():
    """Reset all progress and clean up directories for a fresh start."""
    logger.info("Resetting all progress as requested")
    
    # Reset database
    if os.path.exists(DB_FOLDER):
        shutil.rmtree(DB_FOLDER)
        logger.info(f"Removed database folder: {DB_FOLDER}")
    
    # Reset progress files
    progress_files = [
        PROGRESS_FILE, 
        CROSS_COMPARE_PROGRESS_FILE, 
        REGION_DETECTION_PROGRESS_FILE,
        REGION_EMBEDDING_PROGRESS_FILE, 
        REGION_COMPARISON_PROGRESS_FILE
    ]
    
    for progress_file in progress_files:
        if os.path.exists(progress_file):
            os.remove(progress_file)
            logger.info(f"Removed progress file: {progress_file}")
    
    # Reset output directories while preserving the base folder
    if os.path.exists(OUTPUT_FOLDER):
        for item in os.listdir(OUTPUT_FOLDER):
            item_path = os.path.join(OUTPUT_FOLDER, item)
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
                logger.info(f"Removed output subfolder: {item_path}")
            elif os.path.isfile(item_path):
                os.remove(item_path)
                logger.info(f"Removed output file: {item_path}")
    
    logger.info("Reset completed successfully")

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
    parser.add_argument('--reset', action='store_true',
                        help='Reset all progress and start fresh')
    parser.add_argument('--stage', choices=['orient', 'detect', 'embed', 'cluster', 'all'], default='all',
                        help='Specific stage to run (default: all)')
    parser.add_argument('--diagnostic', action='store_true',
                        help='Run in diagnostic mode with detailed logging')
    parser.add_argument('--similarity-threshold', type=float, default=0.1, 
                        help='Minimum similarity threshold for clustering (default: 0.1)')
    parser.add_argument('--n-clusters', type=int, default=None,
                        help='Number of clusters to form (default: automatic)')
    parser.add_argument('--skip-orientation', action='store_true',
                        help='Skip the text orientation correction step')
    
    args = parser.parse_args()
    
    # Set up detailed logging if diagnostic mode is enabled
    if args.diagnostic:
        logger.setLevel(logging.DEBUG)
        for handler in logger.handlers:
            handler.setLevel(logging.DEBUG)
        logger.info("Running in diagnostic mode with detailed logging")
    
    # Handle reset if requested
    if args.reset:
        reset_workflow()
    
    conf_threshold = args.conf_threshold
    iou_threshold = args.iou_threshold
    run_clustering = not args.skip_clustering
    stage = args.stage
    similarity_threshold = args.similarity_threshold
    n_clusters = args.n_clusters
    run_orientation = not args.skip_orientation and (stage in ['all', 'orient'])
    
    logger.info("Starting complete newspaper image analysis workflow")
    logger.info(f"Selected stage: {stage}")
    
    # Create necessary directories
    for folder in [
        OUTPUT_FOLDER, DB_FOLDER, REGION_CACHE_FOLDER, REGION_OUTPUT_FOLDER,
        ORIENTATION_OUTPUT_FOLDER
    ]:
        os.makedirs(folder, exist_ok=True)
    
    # Step 1: Get image paths
    logger.info(f"Looking for images in {IMAGE_FOLDER}")
    image_paths = get_image_paths(IMAGE_FOLDER)
    
    if not image_paths:
        logger.error(f"No image files found in {IMAGE_FOLDER}. Exiting.")
        return
    
    logger.info(f"Found {len(image_paths)} newspaper images to analyze")
    
    # Step 2: Orientation correction if requested
    corrected_paths = image_paths
    if run_orientation and ORIENTATION_CORRECTION_ENABLED:
        logger.info("Running text orientation correction")
        try:
            corrected_paths = batch_correct_orientation(
                image_paths, 
                output_folder=ORIENTATION_OUTPUT_FOLDER,
                batch_size=8
            )
            logger.info(f"Orientation correction completed for {len(corrected_paths)} images")
        except Exception as e:
            logger.error(f"Orientation correction failed: {str(e)}")
            logger.info("Continuing with original images")
            corrected_paths = image_paths
    else:
        if not ORIENTATION_CORRECTION_ENABLED:
            logger.info("Orientation correction is disabled in config. Skipping.")
        else:
            logger.info("Orientation correction step skipped.")
    
    # Step 3: Initialize document layout detector
    # Only initialize if we're going to use it
    detector = None
    if stage in ['all', 'detect', 'embed']:
        logger.info("Initializing document layout detector")
        try:
            # Try to find the model file
            model_path = "model_cache/DocLayout-YOLO-DocStructBench/doclayout_yolo_docstructbench_imgsz1024.pt"
            if not os.path.exists(model_path):
                script_dir = os.path.dirname(os.path.abspath(__file__))
                model_path = os.path.join(script_dir, model_path)
                logger.info(f"Looking for model at {model_path}")
            
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
    
    # Step 4: Initialize the embedder
    # Only initialize if we're going to use it
    embedder = None
    if stage in ['all', 'embed']:
        logger.info("Initializing mmE5-mllama embedder")
        try:
            embedder = MmE5MllamaEmbedder(device=DEVICE)
            logger.info("Embedder initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize embedder: {str(e)}")
            return
    
    # Step 5: Initialize database connection
    logger.info("Initializing database")
    try:
        chroma_client, collection = initialize_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {str(e)}")
        return
    
    # Step 6: Process regions for all images (if requested)
    if stage in ['all', 'detect']:
        logger.info("Processing regions for all images")
        # First detect regions for all images
        for idx, image_path in enumerate(tqdm(corrected_paths, desc="Detecting regions")):
            regions = process_image_regions(detector, image_path)
            
            # Create visualization (optional)
            visualization_folder = os.path.join(OUTPUT_FOLDER, "region_visualizations")
            os.makedirs(visualization_folder, exist_ok=True)
            output_path = os.path.join(visualization_folder, f"{os.path.splitext(os.path.basename(image_path))[0]}_regions.jpg")
            create_regions_visualization(image_path, regions, output_path)
            
            # Periodically log progress
            if (idx + 1) % 5 == 0 or idx == len(corrected_paths) - 1:
                logger.info(f"Region detection progress: {idx + 1}/{len(corrected_paths)} images")
    
    # Step 7: Initialize region processor and embed regions (if requested)
    if stage in ['all', 'embed']:
        logger.info("Embedding and storing regions")
        region_processor = RegionProcessor(embedder, collection, detector)
        regions_processed = region_processor.process_regions(corrected_paths)
        logger.info(f"Processed {regions_processed} regions from {len(corrected_paths)} images")
    
    # Step 8: Run weighted region clustering if requested
    if run_clustering and stage in ['all', 'cluster']:
        # Check if regions exist before attempting clustering
        if not check_regions_exist(collection) and stage == 'cluster':
            logger.error("No regions found in the database. Please run the 'detect' and 'embed' stages first.")
            logger.error("You can do this with: python complete_workflow.py --stage all")
            return
        
        logger.info("Running weighted region clustering")
        try:
            from weighted_region_clustering import (
                compute_image_similarity_matrix, cluster_images, 
                create_html_report, WEIGHTED_CLUSTERING_FOLDER
            )
            
            # Create output directory
            os.makedirs(WEIGHTED_CLUSTERING_FOLDER, exist_ok=True)
            
            # Compute similarity matrix
            similarity_matrix, image_names = compute_image_similarity_matrix(
                collection, 
                corrected_paths,
                similarity_threshold=similarity_threshold
            )
            
            if similarity_matrix is None or image_names is None:
                logger.error("Failed to compute similarity matrix. Skipping clustering.")
            else:
                # Cluster images
                clustering_results = cluster_images(similarity_matrix, image_names, n_clusters=n_clusters)
                
                if clustering_results:
                    # Create HTML report
                    html_dir = os.path.join(WEIGHTED_CLUSTERING_FOLDER, "html_report")
                    report_path = create_html_report(
                        similarity_matrix, 
                        image_names, 
                        clustering_results, 
                        html_dir
                    )
                    logger.info(f"HTML clustering report created at {report_path}")
                else:
                    logger.error("Clustering failed. No results to display.")
        except Exception as e:
            logger.error(f"Error during clustering: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
    
    logger.info("Complete newspaper image analysis workflow finished successfully!")

if __name__ == "__main__":
    main()