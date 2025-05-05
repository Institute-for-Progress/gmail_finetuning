"""Validation utilities for the project."""

import json
from typing import Dict, Any
from src.utils.logging import logger
from src.utils.config import Config

def validate_training_file(file_path: str) -> bool:
    """
    Validate the training file format.
    
    Args:
        file_path: Path to the training file
        
    Returns:
        bool: True if file is valid, False otherwise
    """
    try:
        with open(file_path, 'r') as f:
            for line in f:
                # Validate JSONL format
                data: Dict[str, Any] = json.loads(line)
                
                # Check required fields
                if 'messages' not in data:
                    logger.error("Missing 'messages' field in training data")
                    return False
                
                # Validate message structure
                if not isinstance(data['messages'], list):
                    logger.error("Messages must be a list")
                    return False
                
                # Check for required message types
                has_system = False
                has_user = False
                has_assistant = False
                
                for msg in data['messages']:
                    if not all(key in msg for key in ['role', 'content']):
                        logger.error("Invalid message format")
                        return False
                    
                    if msg['role'] == 'system':
                        has_system = True
                    elif msg['role'] == 'user':
                        has_user = True
                    elif msg['role'] == 'assistant':
                        has_assistant = True
                
                if not (has_system and has_user and has_assistant):
                    logger.error("Missing required message types (system, user, assistant)")
                    return False
        
        logger.info("Training file validation successful")
        return True
        
    except json.JSONDecodeError:
        logger.error("Invalid JSON format in training file")
        return False
    except Exception as e:
        logger.error(f"Error validating training file: {e}")
        return False

def validate_directory_structure() -> bool:
    """
    Validate that the project directory structure is correct.
    
    Returns:
        bool: True if structure is valid, False otherwise
    """
    required_dirs = [
        Config.CONFIG_DIR,
        Config.DATA_DIR,
        Config.TAKEOUT_DIR
    ]
    
    for directory in required_dirs:
        if not directory.exists():
            logger.error(f"Required directory not found: {directory}")
            return False
    
    if not Config.PARAMS_ENV.exists():
        logger.error(f"params.env file not found at: {Config.PARAMS_ENV}")
        return False
    
    logger.info("Directory structure validation successful")
    return True 