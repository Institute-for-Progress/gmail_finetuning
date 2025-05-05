"""Model training functionality."""

from typing import Optional, Dict, Any
from openai import OpenAI, BadRequestError
from src.utils.logging import logger
from src.utils.config import Config
from src.utils.validation import validate_training_file
from src.inference.model import EmailModel

class ModelTrainer:
    """Handles the fine-tuning process for the email response model."""
    
    def __init__(self, client: Optional[OpenAI] = None):
        """Initialize the trainer with OpenAI client."""
        self.client = client or OpenAI(api_key=Config.OPENAI_API_KEY)
        self.training_file_id: Optional[str] = None
        self.job_id: Optional[str] = None
        self.model_id: Optional[str] = None
        self.model_helper = EmailModel(self.client)  # For parameter detection
    
    def upload_training_data(self, file_path: str = None) -> bool:
        """Upload training data to OpenAI."""
        logger.info("Uploading training data...")
        
        # Use the correct training file path
        if file_path is None:
            file_path = str(Config.TRAINING_FILE)
        try:
            if not validate_training_file(file_path):
                raise ValueError("Training file validation failed")
            
            logger.info(f"Uploading training file: {file_path}")
            with open(file_path, "rb") as f:
                response = self.client.files.create(
                    file=f,
                    purpose="fine-tune"
                )
            self.training_file_id = response.id
            logger.info(f"✓ File uploaded successfully. File ID: {self.training_file_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to upload training data: {e}")
            return False
    
    def start_finetuning(self) -> bool:
        """Start the fine-tuning job."""
        if not self.training_file_id:
            logger.error("No training file ID found. Please upload training data first.")
            return False
        
        logger.info("Starting fine-tuning job...")
        try:
            # Check model availability and get parameters
            is_available, message = self.model_helper.check_model_availability(Config.MODEL_NAME)
            if not is_available:
                logger.error(f"Cannot start fine-tuning: {message}")
                return False
            
            # Get model-specific parameters
            model_params = self.model_helper._detect_model_parameters(Config.MODEL_NAME)
            logger.info(f"Using model parameters: {model_params}")
            
            # Create fine-tuning job with appropriate parameters
            job_params = {
                'training_file': self.training_file_id,
                'model': Config.MODEL_NAME,
                'suffix': Config.SUFFIX
            }
            
            # Add model-specific parameters if supported and set
            if Config.TEMPERATURE is not None:
                job_params['hyperparameters'] = {
                    'temperature': Config.TEMPERATURE
                }
            
            response = self.client.fine_tuning.jobs.create(**job_params)
            
            self.job_id = response.id
            logger.info(f"✓ Fine-tuning job created successfully. Job ID: {self.job_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start fine-tuning job: {e}")
            return False 