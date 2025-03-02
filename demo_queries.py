#!/usr/bin/env python3
"""
Demo query functionality for newspaper image analysis.
Runs demo queries using both image and text inputs, with optional region-specific search.
"""

import os
import glob
import shutil

from config import TESTOUT_FOLDER, TOP_RESULTS
from logger_setup import logger
from visualization import create_regions_visualization

def run_demo_queries(embedder, collection, test_image_path=None, test_text_query="Newspaper headline", include_regions=True):
    """
    Run demo queries using both image and text inputs.
    
    Args:
        embedder: Embedder instance for generating embeddings
        collection: ChromaDB collection for querying
        test_image_path: Path to the test image for image queries
        test_text_query: Text to use for text queries
        include_regions: Whether to include region-specific results
    """
    if not os.path.exists(TESTOUT_FOLDER):
        os.makedirs(TESTOUT_FOLDER)
    
    # Clear previous test results
    for f in glob.glob(os.path.join(TESTOUT_FOLDER, "*_img_result_*")):
        os.remove(f)
    for f in glob.glob(os.path.join(TESTOUT_FOLDER, "*_txt_result_*")):
        os.remove(f)
    for f in glob.glob(os.path.join(TESTOUT_FOLDER, "*_region_result_*")):
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
            # Use the improved embedder to get image embeddings
            query_embed = embedder.get_image_embeddings([test_image_path], is_query=True)[0]
            
            if query_embed is None:
                logger.error(f"Failed to generate embedding for test image {test_image_path}")
                return
                
            # Query top results - using a compatible where clause
            # Instead of using $contains operator, we'll filter based on 'metadata.is_region'
            results = collection.query(
                query_embeddings=[query_embed], 
                n_results=TOP_RESULTS,
                include=["embeddings", "metadatas", "documents", "distances"],
                where={"is_region": {"$eq": False}}  # Only include whole images
            )

            logger.info(f"Top {TOP_RESULTS} results (IMAGE query - whole images only):")
            
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
            
            # If we should include regions, do a separate query for just regions
            if include_regions:
                logger.info(f"\n[TEST A-2] Query by IMAGE (region results): {test_image_path}")
                
                with open(results_info_path, "a") as f:
                    f.write(f"\n\nIMAGE QUERY (region results): {test_image_path}\n")
                    f.write("---------------------\n")
                
                # Query top region results - using a compatible where clause
                region_results = collection.query(
                    query_embeddings=[query_embed], 
                    n_results=TOP_RESULTS,
                    include=["embeddings", "metadatas", "documents", "distances"],
                    where={"is_region": {"$eq": True}}  # Only include regions
                )
                
                if not region_results or "ids" not in region_results or not region_results["ids"][0]:
                    logger.info("No region results found for image query")
                    with open(results_info_path, "a") as f:
                        f.write("No region results found\n")
                else:
                    logger.info(f"Top {len(region_results['ids'][0])} region results (IMAGE query):")
                    
                    # Log and save the region results
                    for i in range(len(region_results["ids"][0])):
                        distance = region_results["distances"][0][i] if "distances" in region_results and i < len(region_results["distances"][0]) else None
                        distance_str = f" (distance: {distance:.4f})" if distance is not None else ""
                        
                        logger.info(f"  Rank {i+1}{distance_str}:")
                        logger.info(f"    ID: {region_results['ids'][0][i]}")
                        if "documents" in region_results and i < len(region_results["documents"][0]):
                            logger.info(f"    Document: {region_results['documents'][0][i]}")
                        
                        # Log to results file
                        with open(results_info_path, "a") as f:
                            f.write(f"\nRank {i+1}{distance_str}:\n")
                            f.write(f"  ID: {region_results['ids'][0][i]}\n")
                            if "documents" in region_results and i < len(region_results["documents"][0]):
                                f.write(f"  Document: {region_results['documents'][0][i]}\n")
                        
                        # Try to copy the result region to results folder
                        try:
                            if "metadatas" in region_results and i < len(region_results["metadatas"][0]):
                                metadata = region_results['metadatas'][0][i]
                                if metadata and isinstance(metadata, dict):
                                    parent_image = metadata.get("parent_image")
                                    region_type = metadata.get("region_type")
                                    region_index = metadata.get("region_index")
                                    box = metadata.get("box")
                                    
                                    if parent_image and os.path.exists(parent_image) and region_type and box:
                                        # Construct the region image path
                                        region_filename = f"{os.path.splitext(os.path.basename(parent_image))[0]}_region{region_index}_{region_type}.png"
                                        region_path = os.path.join("output", "region_images", region_filename)
                                        
                                        # Copy the region image if it exists
                                        if os.path.exists(region_path):
                                            output_path = os.path.join(
                                                TESTOUT_FOLDER, 
                                                f"{i+1:02d}_region_result_{os.path.basename(region_path)}"
                                            )
                                            shutil.copy(region_path, output_path)
                                            logger.info(f"    Copied result region to {output_path}")
                                            
                                            # Add to results file
                                            with open(results_info_path, "a") as f:
                                                f.write(f"  Region: {region_filename}\n")
                                                f.write(f"  Parent Image: {os.path.basename(parent_image)}\n")
                                                f.write(f"  Region Type: {region_type}\n")
                        except Exception as e:
                            logger.error(f"Error copying result region: {e}")
                
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
        # Use the improved embedder for text embeddings
        text_embed = embedder.get_text_embeddings(test_text_query)
        
        # Query top results for whole images - using a compatible where clause
        results = collection.query(
            query_embeddings=[text_embed], 
            n_results=TOP_RESULTS,
            include=["embeddings", "metadatas", "documents", "distances"],
            where={"is_region": {"$eq": False}}  # Only include whole images
        )

        logger.info(f"Top {TOP_RESULTS} results (TEXT query - whole images only):")
        
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
        
        # If we should include regions, do a separate query for just regions
        if include_regions:
            logger.info(f"\n[TEST B-2] Query by TEXT (region results): '{test_text_query}'")
            
            with open(results_info_path, "a") as f:
                f.write(f"\n\nTEXT QUERY (region results): '{test_text_query}'\n")
                f.write("---------------------\n")
            
            # Query top region results - using a compatible where clause
            region_results = collection.query(
                query_embeddings=[text_embed], 
                n_results=TOP_RESULTS,
                include=["embeddings", "metadatas", "documents", "distances"],
                where={"is_region": {"$eq": True}}  # Only include regions
            )
            
            if not region_results or "ids" not in region_results or not region_results["ids"][0]:
                logger.info("No region results found for text query")
                with open(results_info_path, "a") as f:
                    f.write("No region results found\n")
            else:
                logger.info(f"Top {len(region_results['ids'][0])} region results (TEXT query):")
                
                # Log and save the region results
                for i in range(len(region_results["ids"][0])):
                    distance = region_results["distances"][0][i] if "distances" in region_results and i < len(region_results["distances"][0]) else None
                    distance_str = f" (distance: {distance:.4f})" if distance is not None else ""
                    
                    logger.info(f"  Rank {i+1}{distance_str}:")
                    logger.info(f"    ID: {region_results['ids'][0][i]}")
                    if "documents" in region_results and i < len(region_results["documents"][0]):
                        logger.info(f"    Document: {region_results['documents'][0][i]}")
                    
                    # Log to results file
                    with open(results_info_path, "a") as f:
                        f.write(f"\nRank {i+1}{distance_str}:\n")
                        f.write(f"  ID: {region_results['ids'][0][i]}\n")
                        if "documents" in region_results and i < len(region_results["documents"][0]):
                            f.write(f"  Document: {region_results['documents'][0][i]}\n")
                    
                    # Try to copy the result region to results folder
                    try:
                        if "metadatas" in region_results and i < len(region_results["metadatas"][0]):
                            metadata = region_results['metadatas'][0][i]
                            if metadata and isinstance(metadata, dict):
                                parent_image = metadata.get("parent_image")
                                region_type = metadata.get("region_type")
                                region_index = metadata.get("region_index")
                                box = metadata.get("box")
                                
                                if parent_image and os.path.exists(parent_image) and region_type and box:
                                    # Construct the region image path
                                    region_filename = f"{os.path.splitext(os.path.basename(parent_image))[0]}_region{region_index}_{region_type}.png"
                                    region_path = os.path.join("output", "region_images", region_filename)
                                    
                                    # Copy the region image if it exists
                                    if os.path.exists(region_path):
                                        output_path = os.path.join(
                                            TESTOUT_FOLDER, 
                                            f"{i+1:02d}_txt_region_result_{os.path.basename(region_path)}"
                                        )
                                        shutil.copy(region_path, output_path)
                                        logger.info(f"    Copied result region to {output_path}")
                                        
                                        # Add to results file
                                        with open(results_info_path, "a") as f:
                                            f.write(f"  Region: {region_filename}\n")
                                            f.write(f"  Parent Image: {os.path.basename(parent_image)}\n")
                                            f.write(f"  Region Type: {region_type}\n")
                    except Exception as e:
                        logger.error(f"Error copying result region: {e}")
                
    except Exception as e:
        logger.error(f"Error while querying with text '{test_text_query}': {e}")