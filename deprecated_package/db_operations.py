#!/usr/bin/env python3
"""
Database operations for newspaper image analysis.
Handles initialization and interaction with ChromaDB for storing and querying embeddings.
"""

import os
import datetime
import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

from config import DB_FOLDER, COLLECTION_NAME
from logger_setup import logger
from progress_tracker import mark_image_as_processed, is_image_processed

def initialize_db():
    """
    Initialize ChromaDB and return the client and collection.
    Configure HNSW parameters for better performance with large datasets.
    
    Returns:
        Tuple of (chroma_client, collection)
    """
    logger.info("Initializing ChromaDB (persistent mode)...")
    
    # Configure HNSW parameters as individual metadata fields
    hnsw_config = {
        "hnsw_space": "cosine",  # Use cosine similarity
        "hnsw_M": "32",          # Higher M for better recall
        "hnsw_ef_construction": "200",  # Higher ef_construction for better index quality
        "hnsw_ef": "200"        # Higher ef for better search quality
    }
    
    chroma_client = chromadb.PersistentClient(
        path=DB_FOLDER, 
        settings=Settings(
            anonymized_telemetry=False,
            allow_reset=True
        )
    )
    
    try:
        collection = chroma_client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=DefaultEmbeddingFunction()
        )
        
        # Update metadata with HNSW parameters if they don't exist
        current_metadata = collection.metadata or {}
        updated_metadata = {**current_metadata, **hnsw_config}
        collection.modify(metadata=updated_metadata)
        
        logger.info(f"Collection '{COLLECTION_NAME}' found in ChromaDB.")
    except Exception:
        logger.info(f"Creating collection '{COLLECTION_NAME}' in ChromaDB...")
        collection = chroma_client.create_collection(
            name=COLLECTION_NAME,
            embedding_function=DefaultEmbeddingFunction(),
            metadata=hnsw_config
        )
    
    return chroma_client, collection

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