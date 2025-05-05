"""Main script for orchestrating the fine-tuning process."""

from typing import List, Tuple, Callable, Dict, Any
import json
from openai import OpenAI
from src.utils.logging import logger
from src.utils.config import Config
from src.utils.validation import validate_training_file
from src.training.data_processor import EmailProcessor
from src.training.trainer import ModelTrainer
from src.training.monitor import TrainingMonitor
from src.inference.model import EmailModel

class FineTuningPipeline:
    """Orchestrates the complete fine-tuning process."""
    
    def __init__(self, test_mode: bool = False):
        """Initialize the pipeline with all required components."""
        self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
        self.processor = EmailProcessor()
        self.trainer = ModelTrainer(self.client)
        self.monitor = TrainingMonitor(self.client)
        self.model = EmailModel(self.client)
        self.test_mode = test_mode
    
    def run(self) -> bool:
        """Run the complete fine-tuning pipeline."""
        if self.test_mode:
            return self._run_test_mode()
        return self._run_full_mode()
    
    def _run_test_mode(self) -> bool:
        """Run in test mode, skipping actual training."""
        logger.info("Running in TEST MODE - skipping actual training")
        
        # Step 1: Validate directory structure
        logger.info("Step 1: Validating directory structure...")
        if not self._validate_directories():
            return False
        
        # Step 2: Test API connection
        logger.info("Step 2: Testing API connection...")
        if not self._test_api_connection():
            return False
        
        # Step 3: Process emails (create sample training data)
        logger.info("Step 3: Processing emails...")
        if not self.processor.process_emails():
            return False
        
        # Step 4: Validate training data
        logger.info("Step 4: Validating training data...")
        if not self._validate_training_data():
            return False
        
        # Step 5: Test base model with different email types
        logger.info("Step 5: Testing base model with different email types...")
        if not self._test_base_model():
            return False
        
        logger.info("✓ Test mode completed successfully!")
        logger.info("All components are working correctly.")
        logger.info("You can now run the full training process with test_mode=False")
        return True
    
    def _validate_directories(self) -> bool:
        """Validate that all required directories exist."""
        required_dirs = [
            Config.DATA_DIR,
            Config.TAKEOUT_DIR,
            Config.CONFIG_DIR
        ]
        
        for directory in required_dirs:
            if not directory.exists():
                logger.error(f"Required directory not found: {directory}")
                return False
        
        logger.info("✓ Directory structure validated")
        return True
    
    def _test_api_connection(self) -> bool:
        """Test the OpenAI API connection."""
        try:
            # Simple API call to verify connection
            self.client.models.list()
            logger.info("✓ API connection successful")
            return True
        except Exception as e:
            logger.error(f"API connection failed: {e}")
            return False
    
    def _validate_training_data(self) -> bool:
        """Validate the training data format and content."""
        if not validate_training_file(Config.TRAINING_FILE):
            return False
        
        # Additional validation: check data distribution
        try:
            with open(Config.TRAINING_FILE, 'r') as f:
                samples = [json.loads(line) for line in f]
            
            # Check sample count
            if len(samples) < 10:
                logger.warning(f"Low number of training samples: {len(samples)}")
            
            # Check message lengths
            avg_length = sum(len(msg['content']) for sample in samples 
                           for msg in sample['messages']) / len(samples)
            logger.info(f"Average message length: {avg_length:.1f} characters")
            
            logger.info("✓ Training data validated")
            return True
            
        except Exception as e:
            logger.error(f"Error analyzing training data: {e}")
            return False
    
    def _test_base_model(self) -> bool:
        """Test the base model with different email types."""
        # First check if the model is available
        is_available, message = self.model.check_model_availability()
        if not is_available:
            logger.error(f"Cannot run model tests: {message}")
            return False
        
        test_emails = {
            "short": "Hi, can we meet tomorrow?",
            "medium": """
Subject: Project Update Request
Hi team, could you please provide an update on the current project status? I need this for tomorrow's meeting.
Thanks, Manager""",
            "long": """
Subject: Quarterly Review Planning
Hi team, we need to schedule a quarterly review. Please prepare:
1. Project status
2. Key achievements
3. Challenges faced
Let me know your availability for next week.
Best regards, Director"""
        }
        
        success_count = 0
        total_tests = len(test_emails)
        
        try:
            for email_type, content in test_emails.items():
                logger.info(f"\nTesting {email_type} email...")
                reply = self.model.generate_reply(content)
                
                if reply:
                    logger.info(f"\n--- {email_type.capitalize()} Email Reply ---")
                    logger.info(reply)
                    logger.info("----------------")
                    
                    # Basic response validation
                    if self._validate_response(reply, email_type):
                        success_count += 1
                    else:
                        logger.warning(f"Response validation warning for {email_type} email")
                else:
                    logger.error(f"Failed to generate reply for {email_type} email")
            
            # Consider test successful if majority of emails were handled correctly
            if success_count >= (total_tests / 2):
                logger.info(f"✓ Base model testing completed ({success_count}/{total_tests} successful)")
                return True
            else:
                logger.error(f"Base model testing failed ({success_count}/{total_tests} successful)")
                return False
            
        except Exception as e:
            logger.error(f"Failed to test base model: {e}")
            return False
    
    def _validate_response(self, reply: str, email_type: str) -> bool:
        """Validate the model's response based on email type."""
        # Basic validation rules
        if not reply.strip():
            logger.error("Empty response")
            return False
        
        # Check for common email elements
        has_greeting = any(greeting in reply.lower() for greeting in ["hi", "hello", "dear"])
        has_signature = any(sig in reply.lower() for sig in ["best", "regards", "thanks"])
        
        # Type-specific validation
        if email_type == "short":
            if len(reply.split()) > 100:  # Should be concise
                logger.warning("Short email response might be too long")
        elif email_type == "long":
            if len(reply.split()) < 50:  # Should be detailed
                logger.warning("Long email response might be too short")
        
        return True
    
    def _run_full_mode(self) -> bool:
        """Run the complete fine-tuning process."""
        steps: List[Tuple[str, Callable[[], bool]]] = [
            ('Process emails', self.processor.process_emails),
            ('Upload training data', self.trainer.upload_training_data),
            ('Start fine-tuning', self.trainer.start_finetuning),
            ('Monitor training', lambda: self._monitor_training()),
            ('Test model', self.model.test_model)
        ]
        
        for step_name, step_func in steps:
            logger.info(f"Starting step: {step_name}")
            if not step_func():
                logger.error(f"Pipeline failed at step: {step_name}")
                return False
            logger.info(f"Completed step: {step_name}")
        
        logger.info("Fine-tuning process complete!")
        return True
    
    def _monitor_training(self) -> bool:
        """Monitor the training job and update model ID."""
        if not self.trainer.job_id:
            logger.error("No job ID found")
            return False
        
        model_id = self.monitor.monitor_job(self.trainer.job_id)
        if model_id:
            self.model.model_id = model_id
            return True
        return False

def main():
    """Main function to run the fine-tuning pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run the fine-tuning pipeline')
    parser.add_argument('--test', action='store_true', help='Run in test mode (skip actual training)')
    args = parser.parse_args()
    
    logger.info(f"Starting {'test' if args.test else 'full'} process...")
    
    pipeline = FineTuningPipeline(test_mode=args.test)
    if pipeline.run():
        logger.info("Process completed successfully!")
    else:
        logger.error("Process failed. Check the logs for details.")

if __name__ == "__main__":
    main() 