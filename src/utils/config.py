"""Configuration utilities for the project."""

import os
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional

# Load .env file for sensitive data like API keys
load_dotenv()

def get_param_from_file(file_path: Path, param_name: str) -> Optional[str]:
    """
    Get a parameter value from a file.
    
    Args:
        file_path: Path to the parameter file
        param_name: Name of the parameter to get
        
    Returns:
        Optional[str]: Parameter value if found, None otherwise
    """
    if not file_path.exists():
        return None
        
    try:
        with open(file_path, 'r') as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                if '=' not in line:
                    continue
                key, value = line.strip().split('=', 1)
                if key.strip() == param_name:
                    return value.strip().strip('"\'')
    except Exception:
        return None
    return None

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent

class Config:
    """Configuration class for model training and inference."""
    
    # Paths
    CONFIG_DIR = PROJECT_ROOT / "config"
    PARAMS_ENV = CONFIG_DIR / "params.env"
    DATA_DIR = PROJECT_ROOT / "data"
    TAKEOUT_DIR = DATA_DIR / "Takeout"
    TRAINING_FILE = DATA_DIR / "training_data.jsonl"
    
    # API Keys - loaded from .env
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not found in .env file")
    
    # Model configuration - loaded from params.env with fallbacks
    MODEL_NAME = get_param_from_file(PARAMS_ENV, "MODEL_NAME") or os.getenv("MODEL_NAME", "gpt-3.5-turbo")
    SUFFIX = get_param_from_file(PARAMS_ENV, "MODEL_SUFFIX") or os.getenv("MODEL_SUFFIX", "email_tuned")
    MAX_TOKENS = int(get_param_from_file(PARAMS_ENV, "MAX_TOKENS") or os.getenv("MAX_TOKENS", "300"))
    TEMPERATURE = float(get_param_from_file(PARAMS_ENV, "TEMPERATURE") or os.getenv("TEMPERATURE", "0.7"))
    
    # Training configuration - loaded from params.env with fallbacks
    MONITORING_INTERVAL = int(get_param_from_file(PARAMS_ENV, "MONITORING_INTERVAL") or os.getenv("MONITORING_INTERVAL", "60"))
    
    # Fine-tuning IDs - loaded from params.env
    TRAINING_FILE_ID = get_param_from_file(PARAMS_ENV, "TRAINING_FILE_ID") or os.getenv("TRAINING_FILE_ID", "")
    JOB_ID = get_param_from_file(PARAMS_ENV, "JOB_ID") or os.getenv("JOB_ID", "")
    TRAINED_MODEL_ID = get_param_from_file(PARAMS_ENV, "TRAINED_MODEL_ID") or os.getenv("TRAINED_MODEL_ID", "")
    
    # Prompts - loaded from params.env with fallbacks
    SYSTEM_PROMPT = get_param_from_file(PARAMS_ENV, "SYSTEM_PROMPT") or os.getenv("SYSTEM_PROMPT", 
        "You are an email assistant drafting replies in a specific user's tone."
    )
    TEST_EMAIL = get_param_from_file(PARAMS_ENV, "TEST_EMAIL") or os.getenv("TEST_EMAIL",
        "Subject: Quick test\n\nHi there,\nJust testing the new model.\n\nBest,\nTest User"
    ) 