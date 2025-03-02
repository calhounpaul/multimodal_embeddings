#!/usr/bin/env python3
"""
Region processing for newspaper image analysis.
Handles the extraction, embedding, and storage of image regions detected by the document layout detector.
"""

import os
import json
import uuid
from PIL import Image
import numpy as np
from tqdm import tqdm

from config import (
    REGION_OUTPUT_FOLDER, REGION_TYPES_TO_PROCESS, REGION_BATCH_SIZE
)
from logger_setup import logger
from doclayout_detector import DocLayoutDetector
from progress_tracker import (
    load_region_embedding_progress, save_region_embedding_progress,
    mark_region_embedding_as_completed, is_region_embedding_completed
)
from image_utils import validate_image

class RegionProcessor:
    """
    Handles the processing of image regions including extraction, embedding, and storage.
    """

    def __init__(self, embedder, collection, detector):
        self.embedder = embedder
        self.collection = collection
        self.detector = detector
        os.makedirs(REGION_OUTPUT_FOLDER, exist_ok=True)

    def process_regions(self, image_paths, force_recompute=False):
        total_images = len(image_paths)
        logger.info(f"Processing regions for {total_images} images")

        total_regions_processed = 0
        for idx, image_path in enumerate(tqdm(image_paths, desc="Detecting regions")):
            if not validate_image(image_path):
                logger.error(f"Skipping invalid image: {image_path}")
                continue

            image_filename = os.path.basename(image_path)
            regions = self.detector.detect_regions(image_path, force_recompute)

            if regions is None or not regions.get('boxes'):
                logger.warning(f"No regions detected in {image_filename}")
                continue

            regions_processed = self.process_image_regions(image_path, regions)
            total_regions_processed += regions_processed

            if (idx + 1) % 5 == 0 or idx == len(image_paths) - 1:
                logger.info(f"Region processing progress: {idx + 1}/{total_images} images")

        logger.info(f"Completed region processing: {total_regions_processed} regions processed in {total_images} images")
        return total_regions_processed

    def process_image_regions(self, image_path, regions):
        image_filename = os.path.basename(image_path)
        boxes = regions.get('boxes', [])
        classes = regions.get('classes', [])
        class_names = regions.get('class_names', [])
        scores = regions.get('scores', [])
        image_size = regions.get('image_size', {'width': 0, 'height': 0})

        if not boxes:
            return 0

        regions_to_embed, region_metadata, region_ids = [], [], []

        for i, (box, class_id, class_name, score) in enumerate(zip(boxes, classes, class_names, scores)):
            if class_name not in REGION_TYPES_TO_PROCESS:
                continue

            region_id = f"region_{os.path.splitext(image_filename)[0]}_{i}"
            if is_region_embedding_completed(region_id):
                logger.debug(f"Skipping region {region_id}, already embedded")
                continue

            region_image = self.detector.get_region_image(image_path, box)
            if region_image is None:
                logger.warning(f"Failed to extract region {i} from {image_filename}")
                continue

            x_min, y_min, x_max, y_max = map(int, box)
            region_width, region_height = x_max - x_min, y_max - y_min
            region_area = region_width * region_height
            total_area = image_size['width'] * image_size['height']
            area_percentage = (region_area / total_area) * 100 if total_area else 0

            metadata = {
                'parent_image': image_path,
                'parent_image_name': image_filename,
                'region_index': i,
                'region_type': class_name,
                'region_class_id': int(class_id),
                'region_score': float(score),
                'box': ','.join(map(str, box)),  # <-- Fixed metadata box to string
                'box_normalized': ','.join(map(str, [
                    x_min / image_size['width'],
                    y_min / image_size['height'],
                    x_max / image_size['width'],
                    y_max / image_size['height']
                ])),  # <-- Fixed normalized box as string
                'area_percentage': area_percentage,
                'width': region_width,
                'height': region_height,
                'is_region': True
            }

            region_filename = f"{os.path.splitext(image_filename)[0]}_region{i}_{class_name}.png"
            region_path = os.path.join(REGION_OUTPUT_FOLDER, region_filename)
            region_image.save(region_path)

            regions_to_embed.append(region_path)
            region_metadata.append(metadata)
            region_ids.append(region_id)

        embedded_count = 0
        for i in range(0, len(regions_to_embed), REGION_BATCH_SIZE):
            batch_paths = regions_to_embed[i:i+REGION_BATCH_SIZE]
            batch_metadata = region_metadata[i:i+REGION_BATCH_SIZE]
            batch_ids = region_ids[i:i+REGION_BATCH_SIZE]

            embeddings = self.embedder.get_image_embeddings(batch_paths)
            valid_embeddings, valid_metadata, valid_ids = [], [], []

            for embedding, metadata, region_id in zip(embeddings, batch_metadata, batch_ids):
                if embedding is not None:
                    valid_embeddings.append(embedding)
                    valid_metadata.append(metadata)
                    valid_ids.append(region_id)
                else:
                    logger.warning(f"Failed embedding: {region_id}")

            if valid_embeddings:
                documents = [f"Region: {m['region_type']} from {m['parent_image_name']}" for m in valid_metadata]

                try:
                    self.collection.upsert(
                        ids=valid_ids,
                        embeddings=valid_embeddings,
                        documents=documents,
                        metadatas=valid_metadata
                    )
                    for region_id in valid_ids:
                        mark_region_embedding_as_completed(region_id)
                        embedded_count += 1

                    logger.info(f"Embedded {len(valid_embeddings)} regions from {image_filename}")
                except Exception as e:
                    logger.error(f"DB Error: {e}")

        return embedded_count
