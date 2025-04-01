#!/usr/bin/env python3
"""
Embedding model wrapper and utilities for newspaper image analysis.
Provides interfaces for generating embeddings from images and text.
Enhanced with true parallel processing across multiple GPUs.
"""

import torch
import concurrent.futures
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor
from tqdm import tqdm

from config import DEFAULT_MODEL_NAME, DEVICE, MAX_IMAGE_HEIGHT_AND_WIDTH, BATCH_SIZE
from logger_setup import logger

def last_pooling(last_hidden_state, attention_mask, normalize=True):
    """
    Pooling on the last token representation (similar to E5).
    
    Args:
        last_hidden_state: Hidden state tensor
        attention_mask: Attention mask tensor
        normalize: Whether to L2-normalize the output vectors
        
    Returns:
        Pooled representation tensor
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
    Supports true parallel processing across multiple GPUs.
    """

    def __init__(self, model_name=DEFAULT_MODEL_NAME, device=DEVICE, gpu_count=None):
        """
        Initialize the embedder with the specified model.
        
        Args:
            model_name: Name of the model to load
            device: Device to use (cuda or cpu)
            gpu_count: Number of GPUs to use (None means use all available)
        """
        logger.info(f"Loading mmE5-mllama model: {model_name}")
        
        # Set up GPU usage based on availability
        if torch.cuda.is_available():
            available_gpus = torch.cuda.device_count()
            if gpu_count is None:
                self.gpu_count = available_gpus
            else:
                self.gpu_count = min(gpu_count, available_gpus)
            logger.info(f"Using {self.gpu_count} of {available_gpus} available GPUs")
            self.devices = [f"cuda:{i}" for i in range(self.gpu_count)]
        else:
            logger.info("No GPUs available, using CPU")
            self.gpu_count = 0
            self.devices = ["cpu"]
        
        self.model_name = model_name
        
        # Initialize processors and models for each device
        self.processors = []
        self.models = []
        
        for i, device in enumerate(self.devices):
            logger.info(f"Loading model on device {device}")
            processor = AutoProcessor.from_pretrained(model_name)
            model = MllamaForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
            ).to(device)
            model.eval()
            self.processors.append(processor)
            self.models.append(model)
        
        logger.info(f"Finished loading models on {len(self.devices)} devices")

    def _process_image_batch(self, batch_data):
        """
        Process a batch of images on a specific GPU.
        
        Args:
            batch_data: Tuple of (device_idx, [(image_idx, image_path), ...])
        
        Returns:
            List of (image_idx, embedding) tuples
        """
        device_idx, image_batch = batch_data
        device = self.devices[device_idx]
        processor = self.processors[device_idx]
        model = self.models[device_idx]
        
        results = []
        prompt_text = "<|image|><|begin_of_text|> Represent the given image."
        
        for image_idx, img_path in image_batch:
            try:
                # Load the image
                image = Image.open(img_path)
                
                # Check for image dimensions and resize if needed
                if image.size[0] > MAX_IMAGE_HEIGHT_AND_WIDTH or image.size[1] > MAX_IMAGE_HEIGHT_AND_WIDTH:
                    scale_factor = MAX_IMAGE_HEIGHT_AND_WIDTH / max(image.size)
                    new_size = (int(image.size[0] * scale_factor), int(image.size[1] * scale_factor))
                    image = image.resize(new_size, Image.LANCZOS)
                    logger.debug(f"Resized image {img_path} to {new_size}")

                # Preprocess and move to device
                inputs = processor(
                    text=prompt_text,
                    images=[image],
                    return_tensors="pt"
                ).to(device)

                # Forward pass
                with torch.no_grad():
                    outputs = model(**inputs, return_dict=True, output_hidden_states=True)
                    last_hidden_state = outputs.hidden_states[-1]

                # Pooling
                rep = last_pooling(last_hidden_state, inputs['attention_mask'])

                # Move to CPU and convert to list
                rep_list = rep.squeeze(0).cpu().tolist()
                results.append((image_idx, rep_list))
                
            except Exception as e:
                logger.error(f"Error processing image {img_path}: {str(e)}")
                results.append((image_idx, None))
        
        return results

    def get_image_embeddings(self, image_paths, is_query=False, batch_size=BATCH_SIZE):
        """
        Given a list of image_paths, returns a list of embeddings (each a Python list of floats).
        Distributes the work across available GPUs using true parallel processing.
        
        Args:
            image_paths: List of paths to images
            is_query: Whether this is a query operation
            batch_size: Number of images to process per GPU in each batch
            
        Returns:
            List of embeddings (each a list of floats)
        """
        if not image_paths:
            return []
        
        # For single image queries, don't use parallelism overhead
        if len(image_paths) == 1 and is_query:
            device = self.devices[0]
            processor = self.processors[0]
            model = self.models[0]
            
            try:
                image = Image.open(image_paths[0])
                if image.size[0] > MAX_IMAGE_HEIGHT_AND_WIDTH or image.size[1] > MAX_IMAGE_HEIGHT_AND_WIDTH:
                    scale_factor = MAX_IMAGE_HEIGHT_AND_WIDTH / max(image.size)
                    new_size = (int(image.size[0] * scale_factor), int(image.size[1] * scale_factor))
                    image = image.resize(new_size, Image.LANCZOS)
                
                prompt_text = "<|image|><|begin_of_text|> Represent the given image."
                inputs = processor(
                    text=prompt_text,
                    images=[image],
                    return_tensors="pt"
                ).to(device)
                
                with torch.no_grad():
                    outputs = model(**inputs, return_dict=True, output_hidden_states=True)
                    last_hidden_state = outputs.hidden_states[-1]
                
                rep = last_pooling(last_hidden_state, inputs['attention_mask'])
                return [rep.squeeze(0).cpu().tolist()]
            except Exception as e:
                logger.error(f"Error processing query image {image_paths[0]}: {str(e)}")
                return [None]
        
        # Initialize results array with None values
        embeddings = [None] * len(image_paths)
        
        # Create batches for each GPU
        batches = []
        for device_idx in range(len(self.devices)):
            # Collect images for this device
            device_images = []
            for i, img_path in enumerate(image_paths):
                if i % len(self.devices) == device_idx:
                    device_images.append((i, img_path))
            
            # Create batches of appropriate size
            for i in range(0, len(device_images), batch_size):
                batch = device_images[i:i+batch_size]
                if batch:  # Only add non-empty batches
                    batches.append((device_idx, batch))
        
        logger.info(f"Processing {len(image_paths)} images using {len(self.devices)} GPUs with {len(batches)} batches")
        
        # Process batches in parallel using ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.devices)) as executor:
            futures = [executor.submit(self._process_image_batch, batch) for batch in batches]
            
            # Use tqdm for progress tracking if many images
            if len(image_paths) > 10:
                futures_iter = tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Generating embeddings")
            else:
                futures_iter = concurrent.futures.as_completed(futures)
            
            # Collect results as they complete
            for future in futures_iter:
                try:
                    batch_results = future.result()
                    for image_idx, embedding in batch_results:
                        embeddings[image_idx] = embedding
                except Exception as e:
                    logger.error(f"Error in batch processing: {str(e)}")
        
        return embeddings
    
    def get_text_embeddings(self, text):
        """
        Given a text string, return a single embedding (Python list of floats).
        Uses the first model for text embeddings.
        
        Args:
            text: Text string to embed
            
        Returns:
            Embedding as a list of floats
        """
        device = self.devices[0]
        processor = self.processors[0]
        model = self.models[0]
        
        inputs = processor(
            text=text,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs, return_dict=True, output_hidden_states=True)
            last_hidden_state = outputs.hidden_states[-1]

        rep = last_pooling(last_hidden_state, inputs['attention_mask'])
        rep_list = rep.squeeze(0).cpu().tolist()
        return rep_list