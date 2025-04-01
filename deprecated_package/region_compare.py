#!/usr/bin/env python3
"""
Region cross-comparison functionality for newspaper image analysis.
Creates HTML pages showing each region with its most similar regions from other images.
"""

import os
import json
import datetime
import numpy as np
from tqdm import tqdm

from config import (
    REGION_COMPARISON_FOLDER, REGION_COMPARE_TOP_N, REGION_SIMILARITY_THRESHOLD,
    WEIGHT_BY_AREA
)
from logger_setup import logger
from progress_tracker import (
    load_region_comparison_progress, save_region_comparison_progress,
    mark_region_comparison_as_completed, is_region_comparison_completed
)
from db_operations import get_embedding_from_db
from visualization import create_region_comparison_visualization

def create_region_cross_comparison(collection, top_n=REGION_COMPARE_TOP_N):
    """
    Create cross-comparison HTML pages for each region with its most similar regions from other images.
    
    For each region, create an HTML page containing:
    - The source region
    - The parent image with the region highlighted
    - The top N most similar regions from different parent images
    - Similarity scores for all regions
    - Region area percentages and weighted similarity scores
    """
    # Create output folders
    html_output_folder = os.path.join(REGION_COMPARISON_FOLDER, "html_pages")
    visualization_folder = os.path.join(REGION_COMPARISON_FOLDER, "visualizations")
    
    for folder in [REGION_COMPARISON_FOLDER, html_output_folder, visualization_folder]:
        os.makedirs(folder, exist_ok=True)
    
    # Reset progress if needed
    if not os.path.exists(REGION_COMPARISON_FOLDER):
        # If we're creating a fresh folder, reset the progress tracking
        save_region_comparison_progress({"completed_comparisons": []})
    
    logger.info(f"Starting region cross-comparison (top {top_n} similar regions from different images)")
    
    # Query collection for all region entries
    all_entries = collection.get(
        include=["metadatas", "documents"],
        where={"is_region": {"$eq": True}}  # Filter to only include region entries
    )
    
    if not all_entries or "ids" not in all_entries or not all_entries["ids"]:
        logger.warning("No regions found in the database. Make sure regions have been processed first.")
        return
    
    # Create list of all region IDs
    all_region_ids = []
    for i, document in enumerate(all_entries.get("documents", [])):
        if document and "Region:" in document:
            all_region_ids.append(all_entries["ids"][i])
    
    total_regions = len(all_region_ids)
    comparison_progress = load_region_comparison_progress()
    completed_comparisons = len(comparison_progress["completed_comparisons"])
    
    logger.info(f"Region cross-comparison: {completed_comparisons}/{total_regions} regions already completed")
    
    # Create index page
    index_path = os.path.join(html_output_folder, "index.html")
    with open(index_path, "w") as f:
        f.write(f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Region Cross-Comparison Index</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
        h1, h2 {{ color: #333; }}
        .description {{ margin-bottom: 20px; }}
        ul {{ list-style-type: none; padding: 0; }}
        li {{ margin-bottom: 8px; }}
        a {{ color: #0066cc; text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
        .region-type {{ display: inline-block; padding: 2px 6px; border-radius: 3px; margin-right: 8px; }}
        .title {{ background-color: #ffeeaa; }}
        .plain_text {{ background-color: #e0f7fa; }}
        .figure {{ background-color: #e8f5e9; }}
        .table {{ background-color: #f3e5f5; }}
        .caption {{ background-color: #fff3e0; }}
    </style>
</head>
<body>
    <h1>Region Cross-Comparison Index</h1>
    <div class="description">
        <p>This index contains links to all region cross-comparison pages.</p>
        <p>Each page shows a source region and its most similar regions from different parent images.</p>
    </div>
    <h2>All Comparisons:</h2>
    <ul>
""")
    
    # Process each region
    for idx, region_id in enumerate(tqdm(all_region_ids, desc="Comparing regions")):
        # Skip if already completed
        if is_region_comparison_completed(region_id):
            logger.debug(f"Skipping region comparison for {region_id}, already completed")
            continue
            
        # Get region metadata
        region_info = collection.get(
            ids=[region_id],
            include=["metadatas", "embeddings", "documents"]
        )
        
        if not region_info or not region_info["ids"]:
            logger.warning(f"No information found for region {region_id} in database")
            continue
        
        metadata = region_info["metadatas"][0]
        embedding = region_info["embeddings"][0]
        document = region_info["documents"][0]
        
        if not metadata or not embedding:
            logger.warning(f"Missing metadata or embedding for region {region_id}")
            continue
        
        parent_image = metadata.get("parent_image")
        region_type = metadata.get("region_type")
        
        # Create box from individual coordinates or extract from box_str if available
        if "box_str" in metadata and metadata["box_str"]:
            box = [float(x) for x in metadata["box_str"].split(",")]
        elif all(k in metadata for k in ["box_x_min", "box_y_min", "box_x_max", "box_y_max"]):
            box = [
                metadata["box_x_min"],
                metadata["box_y_min"],
                metadata["box_x_max"],
                metadata["box_y_max"]
            ]
        else:
            logger.warning(f"Missing box coordinates for region {region_id}")
            continue
        
        area_percentage = metadata.get("area_percentage", 0)
        
        if not parent_image or not region_type or not box:
            logger.warning(f"Missing essential metadata for region {region_id}")
            continue
        
        # Create sanitized filename for the HTML page
        sanitized_name = region_id.replace(" ", "_").replace(".", "_")
        html_filename = f"{sanitized_name}.html"
        html_path = os.path.join(html_output_folder, html_filename)
        
        # Query for similar regions
        try:
            # Request more results than we need since we'll filter some out
            query_size = min(top_n * 3, 100)
            
            results = collection.query(
                query_embeddings=[embedding],
                n_results=query_size,
                include=["metadatas", "documents", "distances"],
                where={"is_region": {"$eq": True}}  # Only consider other regions
            )
            
            # Check if results are valid
            if not results or "ids" not in results or not results["ids"][0]:
                logger.warning(f"No results found for region {region_id}")
                continue
            
            # Start building the HTML content
            html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Region Cross-Comparison: {region_id}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
        h1, h2, h3 {{ color: #333; }}
        .source-info {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
        .source-region {{ margin-bottom: 30px; }}
        .similar-regions {{ display: flex; flex-wrap: wrap; gap: 20px; }}
        .region-card {{ border: 1px solid #ddd; border-radius: 5px; padding: 15px; width: 300px; }}
        .image-container {{ margin-bottom: 10px; }}
        .image-container img {{ max-width: 100%; height: auto; cursor: pointer; }}
        .score {{ font-weight: bold; }}
        .region-type {{ display: inline-block; padding: 2px 6px; border-radius: 3px; margin-right: 8px; }}
        .title {{ background-color: #ffeeaa; }}
        .plain_text {{ background-color: #e0f7fa; }}
        .figure {{ background-color: #e8f5e9; }}
        .table {{ background-color: #f3e5f5; }}
        .caption {{ background-color: #fff3e0; }}
        a.back {{ display: inline-block; margin-top: 20px; padding: 10px 15px; background-color: #0066cc; color: white; text-decoration: none; border-radius: 4px; }}
        a.back:hover {{ background-color: #0052a3; }}
        a.visualization {{ display: inline-block; margin-top: 5px; padding: 5px 10px; background-color: #4caf50; color: white; text-decoration: none; border-radius: 4px; }}
        a.visualization:hover {{ background-color: #388e3c; }}
    </style>
</head>
<body>
    <h1>Region Cross-Comparison Results</h1>
    
    <div class="source-info">
        <h2>Source Region: {region_id}</h2>
        <p>Type: <span class="region-type {region_type.lower()}">{region_type}</span></p>
        <p>Parent Image: {os.path.basename(parent_image)}</p>
        <p>Area Percentage: {area_percentage:.2f}%</p>
    </div>
    
    <div class="source-region">
        <h2>Source Region:</h2>
        <div class="image-container">
            <a href="../../{os.path.relpath(os.path.join('output', 'region_images', f'{os.path.splitext(os.path.basename(parent_image))[0]}_region{metadata.get("region_index", "0")}_{region_type}.png'))}" target="_blank">
                <img src="../../{os.path.relpath(os.path.join('output', 'region_images', f'{os.path.splitext(os.path.basename(parent_image))[0]}_region{metadata.get("region_index", "0")}_{region_type}.png'))}" 
                     alt="Source Region" title="Click to open full image">
            </a>
        </div>
        <p>Parent Image:</p>
        <div class="image-container">
            <a href="../../{os.path.relpath(parent_image)}" target="_blank">
                <img src="../../{os.path.relpath(parent_image)}" alt="Parent Image" title="Click to open parent image" style="max-height: 300px;">
            </a>
        </div>
    </div>
    
    <h2>Similar Regions (from different images):</h2>
    <div class="similar-regions">
"""
            
            # Process similar regions
            similar_count = 0
            similar_regions = []
            
            for i in range(len(results["ids"][0])):
                result_id = results["ids"][0][i]
                
                # Skip if it's the source region itself
                if result_id == region_id:
                    continue
                
                # Extract metadata for this result
                if "metadatas" in results and results["metadatas"] and i < len(results["metadatas"][0]):
                    metadata = results["metadatas"][0][i]
                else:
                    continue
                
                if not metadata or not isinstance(metadata, dict):
                    continue
                
                # Get the parent image name from metadata
                similar_parent = metadata.get("parent_image", "")
                
                # Skip if from the same parent image
                if similar_parent == parent_image:
                    continue
                
                # Get similarity score and region properties
                similarity_score = None
                if "distances" in results and results["distances"] and i < len(results["distances"][0]):
                    similarity_score = results["distances"][0][i]
                    
                    # Skip if similarity score is below threshold
                    if similarity_score < REGION_SIMILARITY_THRESHOLD:
                        continue
                    
                    # Calculate weighted score if needed
                    if WEIGHT_BY_AREA:
                        source_area_pct = area_percentage
                        target_area_pct = metadata.get("area_percentage", 0)
                        
                        # Weighted score formula: basic_score * (source_area_pct/100) * (target_area_pct/100)
                        weighted_score = similarity_score * (source_area_pct/100) * (target_area_pct/100)
                    else:
                        weighted_score = similarity_score
                        
                    score_str = f"{similarity_score:.4f}"
                    weighted_score_str = f"{weighted_score:.6f}" if WEIGHT_BY_AREA else "N/A"
                else:
                    score_str = "N/A"
                    weighted_score_str = "N/A"
                
                similar_type = metadata.get("region_type", "unknown")
                
                # Get box coordinates
                if "box_str" in metadata and metadata["box_str"]:
                    similar_box = [float(x) for x in metadata["box_str"].split(",")]
                elif all(k in metadata for k in ["box_x_min", "box_y_min", "box_x_max", "box_y_max"]):
                    similar_box = [
                        metadata["box_x_min"],
                        metadata["box_y_min"],
                        metadata["box_x_max"],
                        metadata["box_y_max"]
                    ]
                else:
                    # Default empty box
                    similar_box = [0, 0, 0, 0]
                
                similar_area_pct = metadata.get("area_percentage", 0)
                similar_region_idx = metadata.get("region_index", 0)
                
                # Region image path
                similar_region_filename = f"{os.path.splitext(os.path.basename(similar_parent))[0]}_region{similar_region_idx}_{similar_type}.png"
                similar_region_path = os.path.join('output', 'region_images', similar_region_filename)
                
                # Create visualization path
                vis_filename = f"{sanitized_name}_vs_{result_id.replace(' ', '_').replace('.', '_')}.jpg"
                vis_path = os.path.join(visualization_folder, vis_filename)
                
                # Add to HTML
                html_content += f"""
        <div class="region-card">
            <div class="image-container">
                <a href="../../{os.path.relpath(similar_region_path)}" target="_blank">
                    <img src="../../{os.path.relpath(similar_region_path)}" alt="Similar Region" title="Click to open full image">
                </a>
            </div>
            <p><strong>{similar_count + 1}.</strong> Type: <span class="region-type {similar_type.lower()}">{similar_type}</span></p>
            <p>Parent: {os.path.basename(similar_parent)}</p>
            <p>Area: {similar_area_pct:.2f}%</p>
            <p>Similarity score: <span class="score">{score_str}</span></p>
            <p>Weighted score: <span class="score">{weighted_score_str}</span></p>
            <a href="../../{os.path.relpath(vis_path)}" class="visualization" target="_blank">View Comparison</a>
        </div>
"""
                
                # Create comparison visualization
                create_region_comparison_visualization(
                    parent_image, box, 
                    similar_parent, similar_box,
                    similarity_score, vis_path
                )
                
                # Add to list of similar regions for index
                similar_regions.append({
                    "id": result_id,
                    "score": similarity_score,
                    "weighted_score": weighted_score if WEIGHT_BY_AREA else similarity_score,
                    "parent_image": os.path.basename(similar_parent),
                    "type": similar_type
                })
                
                # Increment counter
                similar_count += 1
                
                # Stop if we've reached the desired number of similar regions
                if similar_count >= top_n:
                    break
            
            # Complete the HTML content
            html_content += """
    </div>
    
    <a href="index.html" class="back">Back to Index</a>
    
    <script>
        // JavaScript for interactive features can be added here
    </script>
</body>
</html>
"""
            
            # Write the HTML file
            with open(html_path, 'w') as f:
                f.write(html_content)
            
            # Add to index page
            region_type_class = region_type.lower()
            with open(index_path, 'a') as f:
                f.write(f'        <li><span class="region-type {region_type_class}">{region_type}</span> <a href="{html_filename}">{region_id}</a> - {similar_count} similar regions</li>\n')
            
            # Mark as completed
            if similar_count > 0:
                logger.info(f"Created HTML page for region {region_id} with {similar_count} similar regions")
            else:
                logger.warning(f"No suitable similar regions found for {region_id}")
                
            mark_region_comparison_as_completed(region_id)
            
        except Exception as e:
            logger.error(f"Error in region comparison for {region_id}: {str(e)}")
            continue
            
        # Update completion percentage periodically
        if (idx + 1) % 10 == 0 or idx == len(all_region_ids) - 1:
            new_completed = len(load_region_comparison_progress()["completed_comparisons"])
            completion_pct = (new_completed / total_regions) * 100
            logger.info(f"Region comparison progress: {new_completed}/{total_regions} regions ({completion_pct:.1f}%)")
    
    # Finish the index page
    with open(index_path, 'a') as f:
        f.write("""    </ul>
    <p><em>Generated on {}</em></p>
</body>
</html>""".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    
    logger.info(f"Region comparison HTML pages created in {html_output_folder}")
    logger.info(f"Region comparison visualizations saved in {visualization_folder}")
    logger.info(f"Index page created at {index_path}")
    
    return True