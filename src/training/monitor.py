"""Training job monitoring functionality."""

import time
from typing import Optional
from openai import OpenAI
from src.utils.logging import logger
from src.utils.config import Config

class TrainingMonitor:
    """Monitors the progress of a fine-tuning job."""
    
    def __init__(self, client: Optional[OpenAI] = None):
        """Initialize the monitor with OpenAI client."""
        self.client = client or OpenAI(api_key=Config.OPENAI_API_KEY)
        self.start_time: Optional[float] = None
    
    def monitor_job(self, job_id: str) -> Optional[str]:
        """
        Monitor a fine-tuning job until completion.
        
        Args:
            job_id: ID of the fine-tuning job to monitor
            
        Returns:
            Optional[str]: Model ID if successful, None if failed
        """
        self.start_time = time.time()
        
        try:
            while True:
                job_status = self.client.fine_tuning.jobs.retrieve(job_id)
                status = job_status.status
                
                # Calculate elapsed time
                elapsed = int(time.time() - self.start_time)
                elapsed_str = self._format_duration(elapsed)
                
                # Get progress information if available
                progress = getattr(job_status, 'progress', None)
                progress_str = f" - Progress: {progress}%" if progress else ""
                
                logger.info(f"Status: {status}{progress_str} (Elapsed: {elapsed_str})")
                
                if status in ["succeeded", "failed", "cancelled"]:
                    if status == "succeeded":
                        model_id = job_status.fine_tuned_model
                        logger.info(f"✓ Fine-tuning successful!")
                        logger.info(f"Fine-tuned model ID: {model_id}")
                        return model_id
                    else:
                        logger.error(f"Fine-tuning failed with status: {status}")
                        logger.error(f"Error details: {job_status.error}")
                        return None
                
                time.sleep(Config.MONITORING_INTERVAL)
                
        except Exception as e:
            logger.error(f"Error monitoring training job: {e}")
            return None
    
    @staticmethod
    def _format_duration(seconds: int) -> str:
        """Format duration in seconds to a human-readable string."""
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}" 