#!/usr/bin/env python3
"""
Database operations for newspaper image analysis.
Handles initialization and interaction with ChromaDB for storing and querying embeddings.
"""

import os
import datetime
import chromadb
from chromadb.config import Settings

from config import DB_FOLDER, COLLECTION_NAME
from logger_setup import logger
from progress_tracker import mark_image_as_processed, is_image_processed

def initialize_db():
    """
    Initialize ChromaDB and return the client and collection.
    
    Returns:
        Tuple of (chroma_client, collection)
    """
    logger.info("Initializing ChromaDB (persistent mode)...")
    chroma_client = chromadb.PersistentClient(path=DB_FOLDER, settings=Settings(anonymized_telemetry=False))
    
    try:
        collection = chroma_client.get_collection(name=COLLECTION_NAME)
        logger.info(f"Collection '{COLLECTION_NAME}' found in ChromaDB.")
    except Exception:
        logger.info(f"Creating collection '{COLLECTION_NAME}' in ChromaDB...")
        collection = chroma_client.create_collection(name=COLLECTION_NAME)
    
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