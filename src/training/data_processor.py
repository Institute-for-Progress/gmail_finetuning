"""Email data processing for training."""

import os
import json
from typing import List, Dict, Any
from src.utils.logging import logger
from src.utils.config import Config

class EmailProcessor:
    """Processes email data for training."""
    
    def __init__(self, takeout_dir: str = None):
        """Initialize the processor with the takeout directory."""
        self.takeout_dir = takeout_dir or str(Config.TAKEOUT_DIR)
        self.output_file = str(Config.TRAINING_FILE)
    
    def process_emails(self) -> bool:
        """
        Process emails from takeout directory.
        
        Returns:
            bool: True if processing successful, False otherwise
        """
        try:
            if not os.path.exists(self.takeout_dir):
                logger.info("Takeout directory not found, generating test data...")
                training_data = self._generate_test_data()
            else:
                # Process emails and create training data
                training_data = self._process_directory()
            
            # Write to JSONL file
            os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
            with open(self.output_file, 'w') as f:
                for item in training_data:
                    f.write(json.dumps(item) + '\n')
            
            logger.info(f"Training data written to {self.output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing emails: {e}")
            return False
            
    def _generate_test_data(self) -> List[Dict[str, Any]]:
        """Generate test data for development and testing."""
        test_data = [
            {
                "messages": [
                    {"role": "system", "content": Config.SYSTEM_PROMPT},
                    {"role": "user", "content": "Subject: Meeting Tomorrow\n\nHi team,\nCan we schedule a quick meeting tomorrow at 2 PM to discuss the project status?\n\nBest,\nJohn"},
                    {"role": "assistant", "content": "Hi John,\n\nThank you for reaching out. Yes, I'm available for a meeting tomorrow at 2 PM to discuss the project status. I'll add it to my calendar.\n\nLooking forward to our discussion.\n\nBest regards,\nSarah"}
                ]
            },
            {
                "messages": [
                    {"role": "system", "content": Config.SYSTEM_PROMPT},
                    {"role": "user", "content": "Subject: Quarterly Report Review\n\nDear team,\n\nPlease find attached the Q1 report for your review. Let me know if you have any questions or suggestions.\n\nRegards,\nMike"},
                    {"role": "assistant", "content": "Dear Mike,\n\nThank you for sharing the Q1 report. I'll review it thoroughly and provide my feedback by end of day tomorrow.\n\nBest regards,\nEmily"}
                ]
            },
            {
                "messages": [
                    {"role": "system", "content": Config.SYSTEM_PROMPT},
                    {"role": "user", "content": "Subject: Quick Question\n\nHey,\nDo you have the latest version of the presentation?\n\nThanks,\nAlex"},
                    {"role": "assistant", "content": "Hey Alex,\n\nYes, I have the latest version. I'll send it over right away.\n\nBest,\nJamie"}
                ]
            }
        ]
        
        logger.info(f"Generated test data with {len(test_data)} samples")
        return test_data
    
    def _process_directory(self) -> List[Dict[str, Any]]:
        """Process the takeout directory and return training data."""
        if not os.path.exists(self.takeout_dir):
            return self._generate_test_data()
            
        # TODO: Implement actual email processing from takeout
        # For now, return test data
        return self._generate_test_data() 