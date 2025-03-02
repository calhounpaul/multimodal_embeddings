#!/usr/bin/env python3
"""
Configuration settings for the newspaper image analysis system.
Contains all constants, paths, and settings used across modules.
"""

import os
import torch

# Test constants
TEST_TEXT = "Hoosier. Hockey."
TEST_IMG = "./sciam.png"

# Disable telemetry
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# Image processing
MAX_IMAGE_HEIGHT_AND_WIDTH = 8000

# Result limits
TOP_RESULTS = 20
CROSS_COMPARE_TOP_N = 5  # Default number of similar images

# Folder paths
IMAGE_FOLDER = "newspaper_images"
OUTPUT_FOLDER = "output"
DB_FOLDER = "db"
TESTOUT_FOLDER = "testout"
CROSS_COMPARE_FOLDER = "cross_compare"
REGION_CACHE_FOLDER = os.path.join(OUTPUT_FOLDER, "region_cache")
REGION_OUTPUT_FOLDER = os.path.join(OUTPUT_FOLDER, "region_images")
REGION_VISUALIZATION_FOLDER = os.path.join(OUTPUT_FOLDER, "region_visualizations")
REGION_COMPARISON_FOLDER = os.path.join(OUTPUT_FOLDER, "region_comparisons")

# Load Hugging Face token from file if it exists
if os.path.exists("HF_TOKEN.txt"):
    os.environ["HF_TOKEN"] = open("HF_TOKEN.txt").read().strip()

# Progress tracking
PROGRESS_FILE = os.path.join(OUTPUT_FOLDER, "processing_progress.json")
CROSS_COMPARE_PROGRESS_FILE = os.path.join(OUTPUT_FOLDER, "cross_compare_progress.json")
REGION_DETECTION_PROGRESS_FILE = os.path.join(OUTPUT_FOLDER, "region_detection_progress.json")
REGION_EMBEDDING_PROGRESS_FILE = os.path.join(OUTPUT_FOLDER, "region_embedding_progress.json")
REGION_COMPARISON_PROGRESS_FILE = os.path.join(OUTPUT_FOLDER, "region_comparison_progress.json")

# Hardware
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Set to None to use all available GPUs, or specify a number
GPU_COUNT = None  # Using None will default to using all available GPUs
# Number of images to process in each batch
BATCH_SIZE = 8  # Adjust based on available GPU memory
REGION_BATCH_SIZE = 24  # Regions are typically smaller, so we can process more at once

# Database
COLLECTION_NAME = "newspaper_image_embeddings"

# Model settings
DEFAULT_MODEL_NAME = "intfloat/mmE5-mllama-11b-instruct"

# DocLayout settings
# Model will be downloaded from Hugging Face hub
DOCLAYOUT_CONF_THRESHOLD = 0.25  # Confidence threshold for region detection
DOCLAYOUT_IOU_THRESHOLD = 0.45   # IoU threshold for NMS
DOCLAYOUT_IMAGE_SIZE = 1024      # Input image size for the YOLO model

# Region processing settings
REGION_TYPES_TO_PROCESS = [
    'title',
    'plain_text',
    'figure',
    'figure_caption',
    'table',
    'table_caption'
]  # Types of regions to process and embed

# Region comparison settings
REGION_COMPARE_TOP_N = 10        # Number of similar regions to find for each region
REGION_SIMILARITY_THRESHOLD = 0.7 # Minimum similarity score for considering two regions similar
WEIGHT_BY_AREA = True            # Whether to weight similarity scores by region areas