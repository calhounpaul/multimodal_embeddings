#!/usr/bin/env python3
"""
Cross-comparison functionality for newspaper image analysis.
Creates HTML pages showing each image with its most similar images.
"""

import os
import datetime

from config import CROSS_COMPARE_FOLDER, CROSS_COMPARE_TOP_N
from logger_setup import logger
from progress_tracker import (
    load_cross_compare_progress, save_cross_compare_progress,
    mark_cross_compare_as_completed, is_cross_compare_completed
)
from db_operations import get_embedding_from_db
from image_processor import process_image

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