#!/usr/bin/env python3
"""
Logging configuration for the newspaper image analysis system.
"""

import os
import logging

def setup_logger():
    """Set up and return the logger with file and console handlers."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("newspaper_process.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

# Create the logger instance
logger = setup_logger()