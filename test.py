#!/usr/bin/env python3
"""
Script for:
1. Processing multiple newspaper images from a folder to generate multimodal embeddings using mmE5-mllama
2. Running demo queries with both image and text inputs
3. Generating cross-comparisons between images

Features:
- Resumable processing: can be stopped and restarted, continuing from where it left off
- Progress tracking for image processing
- Support for multiple image types (jpg, png, webp, etc.)
- Cross-comparison: creates a folder for each image with its N most similar images
"""

import asyncio
import os
import sys
import glob
import logging
import json
import shutil
import datetime
from pathlib import Path
import torch
from PIL import Image
import chromadb
from transformers import MllamaForConditionalGeneration, AutoProcessor
from chromadb.config import Settings

TEST_TEXT = "Hoosier. Hockey."
TEST_IMG = "./sciam.png"

os.environ["ANONYMIZED_TELEMETRY"] = "False"

# Open HF token file if it exists
try:
    with open("hf_token.txt", "r") as f:
        os.environ["HF_TOKEN"] = f.read().strip()
except Exception as e:
    pass  # It's okay if we don't have a token, transformers library will handle it

MAX_IMAGE_HEIGHT_AND_WIDTH = 8000

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("newspaper_process.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

TOP_RESULTS = 20
CROSS_COMPARE_TOP_N = 5  # Default number of similar images to include in cross-comparison

# Folder containing newspaper images
IMAGE_FOLDER = "newspaper_samples_consolidated"

# Global configuration
OUTPUT_FOLDER = "output"
PROGRESS_FILE = os.path.join(OUTPUT_FOLDER, "processing_progress.json")
DB_FOLDER = "db"
TESTOUT_FOLDER = "testout"
CROSS_COMPARE_FOLDER = "cross_compare"  # New folder for cross-comparison results
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COLLECTION_NAME = "newspaper_image_embeddings"

# File to track cross-comparison progress
CROSS_COMPARE_PROGRESS_FILE = os.path.join(OUTPUT_FOLDER, "cross_compare_progress.json")

################################################################################
# Helper functions/classes for mmE5-mllama
################################################################################

def last_pooling(last_hidden_state, attention_mask, normalize=True):
    """
    Pooling on the last token representation (similar to E5).

    If normalize is True, L2-normalizes the output vectors.
    """
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_state.shape[0]
    reps = last_hidden_state[torch.arange(batch_size, device=last_hidden_state.device), sequence_lengths]
    if normalize:
        reps = torch.nn.functional.normalize(reps, p=2, dim=-1)
    return reps

class MmE5MllamaEmbedder:
    """
    Wrapper around the mmE5-mllama-11b-instruct model to provide embeddings.
    """

    def __init__(self, model_name="intfloat/mmE5-mllama-11b-instruct", device="cuda"):
        logger.info(f"Loading mmE5-mllama model: {model_name}")
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = MllamaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
        ).to(device)
        self.model.eval()

    def get_image_embeddings(self, image_paths, is_query=False):
        """
        Given a list of image_paths, returns a list of embeddings (each a Python list of floats).
        """
        embeddings = []
        prompt_text = "<|image|><|begin_of_text|> Represent the given image."

        for img_path in image_paths:
            try:
                # Load the image
                image = Image.open(img_path)
                
                # Check for image dimensions and resize if needed
                if image.size[0] > MAX_IMAGE_HEIGHT_AND_WIDTH or image.size[1] > MAX_IMAGE_HEIGHT_AND_WIDTH:
                    scale_factor = MAX_IMAGE_HEIGHT_AND_WIDTH / max(image.size)
                    new_size = (int(image.size[0] * scale_factor), int(image.size[1] * scale_factor))
                    image = image.resize(new_size, Image.LANCZOS)
                    logger.info(f"Resized image {img_path} to {new_size}")

                # Preprocess and move to device
                inputs = self.processor(
                    text=prompt_text,
                    images=[image],
                    return_tensors="pt"
                ).to(self.device)

                # Forward pass
                with torch.no_grad():
                    outputs = self.model(**inputs, return_dict=True, output_hidden_states=True)
                    last_hidden_state = outputs.hidden_states[-1]  # final layer hidden states

                # Pooling
                rep = last_pooling(last_hidden_state, inputs['attention_mask'])  # shape [1, hidden_size]

                # Move to CPU and convert to list
                rep_list = rep.squeeze(0).cpu().tolist()
                embeddings.append(rep_list)
                
            except Exception as e:
                logger.error(f"Error processing image {img_path}: {str(e)}")
                # Return None for failed embeddings
                embeddings.append(None)

        return embeddings
    
    def get_text_embeddings(self, text):
        """
        Given a text string, return a single embedding (Python list of floats).
        """
        inputs = self.processor(
            text=text,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs, return_dict=True, output_hidden_states=True)
            last_hidden_state = outputs.hidden_states[-1]

        rep = last_pooling(last_hidden_state, inputs['attention_mask'])
        rep_list = rep.squeeze(0).cpu().tolist()
        return rep_list

################################################################################
# Progress Tracking
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
# Image Processing and Embedding Generation
################################################################################

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

def process_image(embedder, collection, image_path):
    """Process a single image and add its embedding to ChromaDB."""
    if not validate_image(image_path):
        logger.error(f"Skipping invalid image: {image_path}")
        return False
        
    image_filename = os.path.basename(image_path)
    image_id = f"image_{image_filename}"
    
    # Check if we've already processed this image in the database
    existing_items = collection.get(ids=[image_id], include=["embeddings", "metadatas", "documents"])
    
    # If already processed with valid embedding, skip
    if (
        len(existing_items["ids"]) > 0
        and "embeddings" in existing_items
        and len(existing_items["embeddings"]) > 0
        and existing_items["embeddings"][0] is not None
        and len(existing_items["embeddings"][0]) > 0
    ):
        logger.info(f" - Skipping {image_filename}, already has embedding in DB.")
        mark_image_as_processed(image_path)
        return True
    
    # Check our progress tracker
    if is_image_processed(image_path):
        # Double-check if it's actually in the DB with a valid embedding
        if len(existing_items["ids"]) == 0:
            logger.warning(f" - Image {image_filename} marked as processed but not in DB. Will reprocess.")
        else:
            logger.info(f" - Skipping {image_filename}, marked as processed in tracker.")
            return True
    
    # Generate embedding
    try:
        # Embed the image using mmE5-mllama
        embedding_results = embedder.get_image_embeddings([image_path], is_query=False)
        
        if not embedding_results or embedding_results[0] is None:
            logger.error(f"Failed to generate embedding for {image_filename}")
            return False
            
        embedding = embedding_results[0]
        
        # Create metadata
        metadata = {
            "image_name": image_filename,
            "image_path": os.path.abspath(image_path),  # Use absolute path for reliability
            "processed_time": str(datetime.datetime.now())
        }
        
        # Add (or update) to ChromaDB
        if len(existing_items["ids"]) > 0:
            # Update record if it exists
            collection.update(
                ids=[image_id],
                embeddings=[embedding],
                metadatas=[metadata],
                documents=[f"Image: {image_filename}"]
            )
            logger.info(f" + Updated embedding for {image_filename} in ChromaDB.")
        else:
            # Create new record
            collection.add(
                ids=[image_id],
                embeddings=[embedding],
                documents=[f"Image: {image_filename}"],
                metadatas=[metadata]
            )
            logger.info(f" + Embedded and stored {image_filename} in ChromaDB.")
        
        # Mark as processed
        mark_image_as_processed(image_path)
        return True
        
    except Exception as e:
        logger.error(f"Error embedding/storing {image_filename}: {str(e)}")
        return False

def process_images(image_paths, embedder, collection):
    """Process images and generate embeddings for each."""
    # Count total images and completed images
    total_images = len(image_paths)
    progress = load_progress()
    completed_images = len(progress["processed_images"])
    
    logger.info(f"Resuming processing from image {completed_images}/{total_images}")
    
    # Process each image
    for idx, image_path in enumerate(image_paths):
        # Skip already processed images based on progress tracker
        if is_image_processed(image_path):
            continue
            
        success = process_image(embedder, collection, image_path)
        
        # Update completion percentage periodically
        if (idx + 1) % 5 == 0 or idx == len(image_paths) - 1:
            new_completed = len(load_progress()["processed_images"])
            completion_pct = (new_completed / total_images) * 100
            logger.info(f"Processing: {new_completed}/{total_images} images ({completion_pct:.1f}%)")
            
        # Check for periodic saving
        if (idx + 1) % 20 == 0:
            logger.info(f"Checkpoint reached at image {idx+1}. Progress saved.")

################################################################################
# Cross-Comparison Functions
################################################################################

def resize_image_if_needed(image_path, output_path):
    """Resize image if it exceeds maximum dimensions, and save to output path."""
    try:
        image = Image.open(image_path)
        
        if image.size[0] > MAX_IMAGE_HEIGHT_AND_WIDTH or image.size[1] > MAX_IMAGE_HEIGHT_AND_WIDTH:
            scale_factor = MAX_IMAGE_HEIGHT_AND_WIDTH / max(image.size)
            new_size = (int(image.size[0] * scale_factor), int(image.size[1] * scale_factor))
            image = image.resize(new_size, Image.LANCZOS)
            logger.info(f"Resized image {image_path} to {new_size} for cross-comparison")
        
        # Save the image (either resized or original)
        image.save(output_path)
        return True
    except Exception as e:
        logger.error(f"Error resizing/copying image {image_path}: {str(e)}")
        return False

def get_embedding_from_db(collection, image_id):
    """
    Get an embedding from the database using image_id
    Returns (embedding, success)
    """
    try:
        result = collection.get(ids=[image_id], include=["embeddings"])
        
        if (
            len(result["ids"]) > 0
            and "embeddings" in result
            and len(result["embeddings"]) > 0
            and result["embeddings"][0] is not None
            and len(result["embeddings"][0]) > 0
        ):
            return result["embeddings"][0], True
        else:
            return None, False
    except Exception as e:
        logger.error(f"Error getting embedding for {image_id}: {str(e)}")
        return None, False

def create_cross_comparison(embedder, collection, image_paths, top_n=CROSS_COMPARE_TOP_N):
    """
    Create cross-comparison HTML pages for each image with its most similar images.
    
    For each image, create an HTML page containing:
    - The source image
    - The top N most similar images that differ in the first 20% of their filename
    - Similarity scores for all images
    - Clickable images that open the full resolution version
    """
    # Create HTML output folder
    html_output_folder = os.path.join(CROSS_COMPARE_FOLDER, "html_pages")
    os.makedirs(html_output_folder, exist_ok=True)
    
    # Reset cross-comparison progress if needed
    if not os.path.exists(CROSS_COMPARE_FOLDER):
        os.makedirs(CROSS_COMPARE_FOLDER, exist_ok=True)
        # If we're creating a fresh folder, reset the progress tracking
        save_cross_compare_progress({"completed_images": []})
    
    logger.info(f"Starting cross-comparison HTML page generation (top {top_n} similar images, excluding files with similar prefixes)")
    
    total_images = len(image_paths)
    cross_compare_progress = load_cross_compare_progress()
    completed_comparisons = len(cross_compare_progress["completed_images"])
    
    logger.info(f"Cross-comparison: {completed_comparisons}/{total_images} images already completed")
    
    # Create index page
    index_path = os.path.join(html_output_folder, "index.html")
    with open(index_path, "w") as f:
        f.write(f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Image Cross-Comparison Index</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
        h1 {{ color: #333; }}
        .description {{ margin-bottom: 20px; }}
        ul {{ list-style-type: none; padding: 0; }}
        li {{ margin-bottom: 8px; }}
        a {{ color: #0066cc; text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
    </style>
</head>
<body>
    <h1>Image Cross-Comparison Index</h1>
    <div class="description">
        <p>This index contains links to all image cross-comparison pages.</p>
        <p>Each page shows a source image and its most similar images that differ in the first 20% of their filename.</p>
    </div>
    <h2>All Comparisons:</h2>
    <ul>
""")
    
    # Process each image
    for idx, image_path in enumerate(image_paths):
        image_filename = os.path.basename(image_path)
        image_id = f"image_{image_filename}"
        
        # Skip if already completed
        if is_cross_compare_completed(image_path):
            logger.info(f" - Skipping cross-comparison for {image_filename}, already completed")
            continue
            
        # Create sanitized filename for the HTML page
        sanitized_name = os.path.splitext(image_filename)[0]  # Remove extension
        sanitized_name = sanitized_name.replace(" ", "_").replace(".", "_")  # Replace spaces and dots
        html_filename = f"{sanitized_name}.html"
        html_path = os.path.join(html_output_folder, html_filename)
        
        # Get embedding for this image
        source_embedding, success = get_embedding_from_db(collection, image_id)
        
        if not success:
            logger.warning(f"No valid embedding found for {image_filename} in DB.")
            
            # Try to process the image again
            logger.info(f"Attempting to regenerate embedding for {image_filename}...")
            if process_image(embedder, collection, image_path):
                source_embedding, success = get_embedding_from_db(collection, image_id)
                if not success:
                    logger.error(f"Failed to regenerate embedding for {image_filename}. Skipping.")
                    continue
            else:
                logger.error(f"Failed to process {image_filename}. Skipping.")
                continue
        
        # Calculate the prefix length (20% of the filename)
        prefix_length = max(1, int(len(image_filename) * 0.2))
        source_prefix = image_filename[:prefix_length]
        logger.info(f" - Source file prefix (first {prefix_length} chars): '{source_prefix}'")
        
        # Query for similar images
        try:
            # First, query for more results than we need, since we'll filter some out
            query_size = min(top_n * 5, 100)  # Request more results since we'll filter some out
            
            results = collection.query(
                query_embeddings=[source_embedding],
                n_results=query_size,
                include=["metadatas", "documents", "distances"]
            )
            
            # Check if results are valid
            if not results or "ids" not in results or len(results["ids"]) == 0 or len(results["ids"][0]) == 0:
                logger.warning(f"No results found for {image_filename}")
                continue
            
            # Start building the HTML content
            html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Cross-Comparison: {image_filename}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
        h1, h2 {{ color: #333; }}
        .source-info {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
        .source-image {{ margin-bottom: 30px; }}
        .similar-images {{ display: flex; flex-wrap: wrap; gap: 20px; }}
        .image-card {{ border: 1px solid #ddd; border-radius: 5px; padding: 15px; width: 300px; }}
        .image-container {{ margin-bottom: 10px; }}
        .image-container img {{ max-width: 100%; height: auto; cursor: pointer; }}
        .score {{ font-weight: bold; }}
        .prefix {{ color: #666; font-style: italic; }}
        a.back {{ display: inline-block; margin-top: 20px; padding: 10px 15px; background-color: #0066cc; color: white; text-decoration: none; border-radius: 4px; }}
        a.back:hover {{ background-color: #0052a3; }}
    </style>
</head>
<body>
    <h1>Cross-Comparison Results</h1>
    
    <div class="source-info">
        <h2>Source Image: {image_filename}</h2>
        <p>Source prefix (first {prefix_length} chars): <span class="prefix">'{source_prefix}'</span></p>
    </div>
    
    <div class="source-image">
        <h2>Source Image:</h2>
        <div class="image-container">
            <a href="../../{os.path.relpath(image_path)}" target="_blank">
                <img src="../../{os.path.relpath(image_path)}" alt="Source: {image_filename}" title="Click to open full image">
            </a>
        </div>
    </div>
    
    <h2>Similar Images (with different prefixes):</h2>
    <div class="similar-images">
"""
            
            # Process similar images
            similar_count = 0
            
            for i in range(len(results["ids"][0])):
                result_id = results["ids"][0][i]
                
                # Skip if it's the source image itself
                if result_id == image_id:
                    continue
                
                # Extract metadata for this result
                if "metadatas" in results and results["metadatas"] and len(results["metadatas"][0]) > i:
                    metadata = results["metadatas"][0][i]
                else:
                    logger.warning(f"Missing metadata for result {i} for {image_filename}")
                    continue
                
                if not metadata or not isinstance(metadata, dict):
                    logger.warning(f"Invalid metadata for result {i} for {image_filename}")
                    continue
                
                # Get the image name from metadata
                similar_path = metadata.get("image_path", "")
                if not similar_path or not os.path.exists(similar_path):
                    logger.warning(f"Similar image path not found: {similar_path}")
                    continue
                
                similar_filename = os.path.basename(similar_path)
                similar_prefix = similar_filename[:prefix_length] if len(similar_filename) >= prefix_length else similar_filename
                
                # Skip if the prefixes match (first 20% of characters)
                if similar_prefix == source_prefix:
                    logger.debug(f" - Skipping {similar_filename} - prefix '{similar_prefix}' matches source")
                    continue
                
                # Get the similarity score
                similarity_score = None
                if "distances" in results and results["distances"] and len(results["distances"][0]) > i:
                    similarity_score = results["distances"][0][i]
                    score_str = f"{similarity_score:.4f}" if similarity_score is not None else "N/A"
                else:
                    score_str = "N/A"
                
                # Add to HTML
                html_content += f"""
        <div class="image-card">
            <div class="image-container">
                <a href="../../{os.path.relpath(similar_path)}" target="_blank">
                    <img src="../../{os.path.relpath(similar_path)}" alt="Similar: {similar_filename}" title="Click to open full image">
                </a>
            </div>
            <p><strong>{similar_count + 1}.</strong> {similar_filename}</p>
            <p>Prefix: <span class="prefix">'{similar_prefix}'</span></p>
            <p>Similarity score: <span class="score">{score_str}</span></p>
        </div>
"""
                
                # Increment counter
                similar_count += 1
                
                # Stop if we've reached the desired number of similar images
                if similar_count >= top_n:
                    break
            
            # Complete the HTML content
            html_content += """
    </div>
    
    <a href="index.html" class="back">Back to Index</a>
    
    <script>
        // You can add JavaScript here if needed
    </script>
</body>
</html>
"""
            
            # Write the HTML file
            with open(html_path, 'w') as f:
                f.write(html_content)
            
            # Add to index page
            with open(index_path, 'a') as f:
                f.write(f'        <li><a href="{html_filename}">{image_filename}</a> - {similar_count} similar images</li>\n')
            
            # Mark as completed if we processed at least one similar image or if we checked all available results
            if similar_count > 0 or i >= len(results["ids"][0]) - 1:
                logger.info(f" + Created HTML page for {image_filename} with {similar_count} similar images (different prefixes)")
                mark_cross_compare_as_completed(image_path)
            else:
                logger.warning(f"No suitable similar images found for {image_filename}")
            
        except Exception as e:
            logger.error(f"Error in cross-comparison for {image_filename}: {str(e)}")
            continue
            
        # Update completion percentage periodically
        if (idx + 1) % 5 == 0 or idx == len(image_paths) - 1:
            new_completed = len(load_cross_compare_progress()["completed_images"])
            completion_pct = (new_completed / total_images) * 100
            logger.info(f"Cross-comparison progress: {new_completed}/{total_images} images ({completion_pct:.1f}%)")
    
    # Finish the index page
    with open(index_path, 'a') as f:
        f.write("""    </ul>
    <p><em>Generated on {}</em></p>
</body>
</html>""".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    
    logger.info(f"Cross-comparison HTML pages created in {html_output_folder}")
    logger.info(f"Index page created at {index_path}")

################################################################################
# Demo Query Functions
################################################################################

def run_demo_queries(embedder, collection, test_image_path=None, test_text_query="Newspaper headline"):
    """Run demo queries using both image and text inputs."""
    if not os.path.exists(TESTOUT_FOLDER):
        os.makedirs(TESTOUT_FOLDER)
    
    # Clear previous test results
    for f in glob.glob(os.path.join(TESTOUT_FOLDER, "*_img_result_*")):
        os.remove(f)
    for f in glob.glob(os.path.join(TESTOUT_FOLDER, "*_txt_result_*")):
        os.remove(f)
        
    # Create result information file
    results_info_path = os.path.join(TESTOUT_FOLDER, "query_results.txt")
    with open(results_info_path, "w") as f:
        f.write("QUERY RESULTS SUMMARY\n")
        f.write("====================\n\n")
    
    # Test A: Query by image
    if test_image_path and os.path.exists(test_image_path):
        logger.info(f"\n[TEST A] Query by IMAGE: {test_image_path}")
        
        with open(results_info_path, "a") as f:
            f.write(f"IMAGE QUERY: {test_image_path}\n")
            f.write("---------------------\n")
        
        try:
            query_embed = embedder.get_image_embeddings([test_image_path])[0]
            
            if query_embed is None:
                logger.error(f"Failed to generate embedding for test image {test_image_path}")
                return
                
            # Query top results
            results = collection.query(
                query_embeddings=[query_embed], 
                n_results=TOP_RESULTS,
                include=["embeddings", "metadatas", "documents", "distances"]
            )

            logger.info(f"Top {TOP_RESULTS} results (IMAGE query):")
            
            # Copy test image to results folder
            test_img_copy = os.path.join(TESTOUT_FOLDER, "test_image_query.png")
            try:
                shutil.copy(test_image_path, test_img_copy)
            except Exception as e:
                logger.error(f"Error copying test image: {e}")
            
            # Log and save the results
            for i in range(len(results["ids"][0])):
                distance = results["distances"][0][i] if "distances" in results and i < len(results["distances"][0]) else None
                distance_str = f" (distance: {distance:.4f})" if distance is not None else ""
                
                logger.info(f"  Rank {i+1}{distance_str}:")
                logger.info(f"    ID: {results['ids'][0][i]}")
                if "documents" in results and i < len(results["documents"][0]):
                    logger.info(f"    Document: {results['documents'][0][i]}")
                
                # Log to results file
                with open(results_info_path, "a") as f:
                    f.write(f"\nRank {i+1}{distance_str}:\n")
                    f.write(f"  ID: {results['ids'][0][i]}\n")
                    if "documents" in results and i < len(results["documents"][0]):
                        f.write(f"  Document: {results['documents'][0][i]}\n")
                
                # Try to copy the result image to results folder
                try:
                    if "metadatas" in results and i < len(results["metadatas"][0]):
                        metadata = results['metadatas'][0][i]
                        if metadata and isinstance(metadata, dict):
                            image_path = metadata.get("image_path")
                            
                            # If we have the image path, copy it
                            if image_path and os.path.exists(image_path):
                                output_path = os.path.join(
                                    TESTOUT_FOLDER, 
                                    f"{i+1:02d}_img_result_{os.path.basename(image_path)}"
                                )
                                shutil.copy(image_path, output_path)
                                logger.info(f"    Copied result image to {output_path}")
                                
                                # Add to results file
                                with open(results_info_path, "a") as f:
                                    f.write(f"  Image: {os.path.basename(image_path)}\n")
                except Exception as e:
                    logger.error(f"Error copying result image: {e}")
        except Exception as e:
            logger.error(f"Error while querying with image {test_image_path}: {e}")
    else:
        logger.warning(f"Test image not found or not specified. Skipping image query test.")

    # Test B: Query by text
    logger.info(f"\n[TEST B] Query by TEXT: '{test_text_query}'")
    
    with open(results_info_path, "a") as f:
        f.write(f"\n\nTEXT QUERY: '{test_text_query}'\n")
        f.write("---------------------\n")
    
    try:
        text_embed = embedder.get_text_embeddings(test_text_query)
        
        # Query top results
        results = collection.query(
            query_embeddings=[text_embed], 
            n_results=TOP_RESULTS,
            include=["embeddings", "metadatas", "documents", "distances"]
        )

        logger.info(f"Top {TOP_RESULTS} results (TEXT query):")
        
        # Log and save the results
        for i in range(len(results["ids"][0])):
            distance = results["distances"][0][i] if "distances" in results and i < len(results["distances"][0]) else None
            distance_str = f" (distance: {distance:.4f})" if distance is not None else ""
            
            logger.info(f"  Rank {i+1}{distance_str}:")
            logger.info(f"    ID: {results['ids'][0][i]}")
            if "documents" in results and i < len(results["documents"][0]):
                logger.info(f"    Document: {results['documents'][0][i]}")
            
            # Log to results file
            with open(results_info_path, "a") as f:
                f.write(f"\nRank {i+1}{distance_str}:\n")
                f.write(f"  ID: {results['ids'][0][i]}\n")
                if "documents" in results and i < len(results["documents"][0]):
                    f.write(f"  Document: {results['documents'][0][i]}\n")
            
            # Try to copy the result image to results folder
            try:
                if "metadatas" in results and i < len(results["metadatas"][0]):
                    metadata = results['metadatas'][0][i]
                    if metadata and isinstance(metadata, dict):
                        image_path = metadata.get("image_path")
                        
                        # If we have the image path, copy it
                        if image_path and os.path.exists(image_path):
                            output_path = os.path.join(
                                TESTOUT_FOLDER, 
                                f"{i+1:02d}_txt_result_{os.path.basename(image_path)}"
                            )
                            shutil.copy(image_path, output_path)
                            logger.info(f"    Copied result image to {output_path}")
                            
                            # Add to results file
                            with open(results_info_path, "a") as f:
                                f.write(f"  Image: {os.path.basename(image_path)}\n")
            except Exception as e:
                logger.error(f"Error copying result image: {e}")
                
    except Exception as e:
        logger.error(f"Error while querying with text '{test_text_query}': {e}")

################################################################################
# Main Function
################################################################################

async def main():
    """
    Main function that coordinates the entire newspaper image analysis workflow.
    This function:
    1. Processes command-line arguments
    2. Initializes the embedding model and database
    3. Processes images to generate embeddings
    4. Creates cross-comparisons between images
    5. Runs demo queries
    """
    import datetime  # Import here for datetime used in process_image
    
    # Get command-line arguments
    test_image_path = sys.argv[1] if len(sys.argv) > 1 else TEST_IMG
    test_text_query = sys.argv[2] if len(sys.argv) > 2 else TEST_TEXT
    cross_compare_n = int(sys.argv[3]) if len(sys.argv) > 3 else CROSS_COMPARE_TOP_N
    
    # Parse additional flags
    cross_compare_only = False  # Default mode
    for arg in sys.argv:
        if arg == "--cross-compare-only":
            cross_compare_only = True
            logger.info("Running in cross-compare only mode")
    
    logger.info("Starting newspaper image analysis process")
    
    # Create necessary directories
    for folder in [OUTPUT_FOLDER, DB_FOLDER, TESTOUT_FOLDER, CROSS_COMPARE_FOLDER]:
        os.makedirs(folder, exist_ok=True)
    
    # Step 1: Get paths to all images in the folder
    logger.info("Step 1: Finding newspaper images...")
    
    if not os.path.exists(IMAGE_FOLDER):
        logger.error(f"Image folder '{IMAGE_FOLDER}' does not exist. Exiting.")
        return
        
    image_paths = get_image_paths(IMAGE_FOLDER)
    
    if not image_paths:
        logger.error("No image files found in the specified folder. Exiting.")
        return
    
    logger.info(f"Found {len(image_paths)} image files to process")
    
    # Step 2: Initialize the embedding model and database
    logger.info("Step 2: Initializing embedding model and database...")
    embedder = MmE5MllamaEmbedder(model_name="intfloat/mmE5-mllama-11b-instruct", device=DEVICE)
    
    # Initialize ChromaDB
    logger.info("Initializing ChromaDB (persistent mode)...")
    chroma_client = chromadb.PersistentClient(path=DB_FOLDER, settings=Settings(anonymized_telemetry=False))
    
    try:
        collection = chroma_client.get_collection(name=COLLECTION_NAME)
        logger.info(f"Collection '{COLLECTION_NAME}' found in ChromaDB.")
    except Exception:
        logger.info(f"Creating collection '{COLLECTION_NAME}' in ChromaDB...")
        collection = chroma_client.create_collection(name=COLLECTION_NAME)
    
    # Step 3: Process images (unless we're in cross-compare-only mode)
    if not cross_compare_only:
        logger.info("Step 3: Processing images and generating embeddings...")
        
        # Check progress to report
        progress = load_progress()
        completed_images = len(progress["processed_images"])
        
        if completed_images > 0:
            logger.info(f"Resuming processing from image {completed_images}/{len(image_paths)}")
        
        # Process the images
        process_images(image_paths, embedder, collection)
    else:
        logger.info("Skipping embedding generation (cross-compare only mode)")
    
    # Step 4: Generate cross-comparisons
    logger.info("Step 4: Generating cross-comparisons...")
    create_cross_comparison(embedder, collection, image_paths, top_n=cross_compare_n)
    
    # Step 5: Run demo queries
    logger.info("Step 5: Running demo queries...")
    run_demo_queries(embedder, collection, test_image_path, test_text_query)
    
    logger.info("Newspaper image analysis completed successfully!")

if __name__ == "__main__":
    asyncio.run(main())
