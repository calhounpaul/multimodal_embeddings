#!/usr/bin/env python3
"""
Core image processing functionality for newspaper image analysis.
Handles processing images and storing their embeddings in the database.
Enhanced with batch processing for better performance.
"""

import os
import datetime
import math
from tqdm import tqdm

from config import IMAGE_FOLDER, BATCH_SIZE
from logger_setup import logger
from progress_tracker import load_progress, is_image_processed, mark_image_as_processed
from image_utils import validate_image, get_image_paths

def process_image(embedder, collection, image_path):
    """
    Process a single image and add its embedding to ChromaDB.
    
    This function is maintained for backward compatibility with other modules
    like cross_compare.py that rely on single-image processing.
    """
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
        # Embed the image using the embedder
        embedding_results = embedder.get_image_embeddings([image_path], is_query=False)
        
        if not embedding_results or embedding_results[0] is None:
            logger.error(f"Failed to generate embedding for {image_filename}")
            return False
            
        embedding = embedding_results[0]
        
        # Create metadata
        metadata = {
            "image_name": image_filename,
            "image_path": os.path.abspath(image_path),  # Use absolute path for reliability
            "processed_time": str(datetime.datetime.now()),
            "is_region": False  # Indicate this is a whole image, not a region
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

def process_image_batch(embedder, collection, batch_paths):
    """Process a batch of images and add their embeddings to ChromaDB."""
    # Filter out already processed images and invalid images
    valid_paths = []
    image_ids = []
    for image_path in batch_paths:
        if is_image_processed(image_path):
            continue
        
        if not validate_image(image_path):
            logger.error(f"Skipping invalid image: {image_path}")
            continue
            
        valid_paths.append(image_path)
        image_filename = os.path.basename(image_path)
        image_ids.append(f"image_{image_filename}")
    
    if not valid_paths:
        return 0
    
    # Check which images are already in the database
    try:
        existing_items = collection.get(ids=image_ids, include=["embeddings"])
        existing_ids = set(existing_items["ids"])
    except Exception as e:
        logger.error(f"Error checking database for existing items: {str(e)}")
        existing_ids = set()
    
    # Filter out images already in database with valid embeddings
    to_process = []
    ids_to_process = []
    for i, image_path in enumerate(valid_paths):
        image_id = image_ids[i]
        image_filename = os.path.basename(image_path)
        
        if image_id in existing_ids:
            idx = existing_items["ids"].index(image_id)
            # If it exists with a valid embedding, skip it
            if (idx < len(existing_items["embeddings"]) and 
                existing_items["embeddings"][idx] is not None and 
                len(existing_items["embeddings"][idx]) > 0):
                logger.info(f" - Skipping {image_filename}, already has embedding in DB.")
                mark_image_as_processed(image_path)
                continue
        
        to_process.append(image_path)
        ids_to_process.append(image_id)
    
    if not to_process:
        return 0
    
    # Generate embeddings for the batch using the improved embedder
    try:
        embeddings = embedder.get_image_embeddings(to_process)
        
        # Prepare data for adding/updating to ChromaDB
        ids_list = []
        embeddings_list = []
        documents_list = []
        metadatas_list = []
        
        for i, (image_path, embedding) in enumerate(zip(to_process, embeddings)):
            if embedding is None:
                logger.error(f"Failed to generate embedding for {os.path.basename(image_path)}")
                continue
                
            image_id = ids_to_process[i]
            image_filename = os.path.basename(image_path)
            
            # Create metadata
            metadata = {
                "image_name": image_filename,
                "image_path": os.path.abspath(image_path),  # Use absolute path for reliability
                "processed_time": str(datetime.datetime.now())
            }
            
            ids_list.append(image_id)
            embeddings_list.append(embedding)
            documents_list.append(f"Image: {image_filename}")
            metadatas_list.append(metadata)
        
        if not ids_list:
            return 0
            
        # Add to ChromaDB - upsert handles both adding new and updating existing
        collection.upsert(
            ids=ids_list,
            embeddings=embeddings_list,
            documents=documents_list,
            metadatas=metadatas_list
        )
        
        logger.info(f" + Processed {len(ids_list)} images in batch")
        
        # Mark as processed
        for path in to_process:
            if path in [to_process[i] for i in range(len(ids_list))]:
                mark_image_as_processed(path)
            
        return len(ids_list)
        
    except Exception as e:
        logger.error(f"Error processing batch: {str(e)}")
        return 0

def process_images(image_paths, embedder, collection):
    """Process images and generate embeddings for each, using batch processing for efficiency."""
    # Count total images and completed images
    total_images = len(image_paths)
    progress = load_progress()
    completed_images = len(progress["processed_images"])
    
    logger.info(f"Resuming processing from image {completed_images}/{total_images}")
    
    # Create batches for processing
    batch_size = BATCH_SIZE  # From config
    num_batches = math.ceil(len(image_paths) / batch_size)
    
    # Process in batches with progress bar
    successfully_processed = 0
    with tqdm(total=total_images - completed_images, desc="Processing images") as pbar:
        for i in range(0, len(image_paths), batch_size):
            batch = image_paths[i:i+batch_size]
            
            # Count already processed in this batch
            already_processed = sum(1 for img in batch if is_image_processed(img))
            
            # Process batch
            newly_processed = process_image_batch(embedder, collection, batch)
            successfully_processed += newly_processed
            
            # Update progress bar
            pbar.update(newly_processed + already_processed)
            
            # Periodically log progress
            if i + batch_size >= len(image_paths) or (i // batch_size) % 5 == 0:
                total_processed = len(load_progress()["processed_images"])
                completion_pct = (total_processed / total_images) * 100
                logger.info(f"Processing: {total_processed}/{total_images} images ({completion_pct:.1f}%)")
    
    logger.info(f"Successfully processed {successfully_processed} new images")
    
    # Final progress report
    total_processed = len(load_progress()["processed_images"])
    logger.info(f"Total images processed: {total_processed}/{total_images} ({(total_processed/total_images)*100:.1f}%)")