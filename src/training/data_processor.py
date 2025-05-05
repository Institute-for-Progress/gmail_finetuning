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
        """Process the takeout directory and return training data from MBOX files."""
        import mailbox
        import glob
        import email
        from email.header import decode_header

        training_data = []
        mbox_files = glob.glob(os.path.join(self.takeout_dir, '**', '*.mbox'), recursive=True)
        if not mbox_files:
            logger.warning("No MBOX files found in Takeout directory. Generating test data.")
            return self._generate_test_data()

        def decode_str(s):
            if not s:
                return ''
            parts = decode_header(s)
            return ''.join([
                (part.decode(enc or 'utf-8') if isinstance(part, bytes) else part)
                for part, enc in parts
            ])

        for mbox_path in mbox_files:
            logger.info(f"Processing MBOX file: {mbox_path}")
            mbox = mailbox.mbox(mbox_path)
            for msg in mbox:
                try:
                    subject = decode_str(msg.get('subject', ''))
                    from_ = decode_str(msg.get('from', ''))
                    to = decode_str(msg.get('to', ''))
                    date = decode_str(msg.get('date', ''))
                    # Get email body (plain text preferred)
                    body = ''
                    if msg.is_multipart():
                        for part in msg.walk():
                            ctype = part.get_content_type()
                            disp = str(part.get('Content-Disposition'))
                            if ctype == 'text/plain' and 'attachment' not in disp:
                                charset = part.get_content_charset() or 'utf-8'
                                body = part.get_payload(decode=True).decode(charset, errors='replace')
                                break
                    else:
                        charset = msg.get_content_charset() or 'utf-8'
                        body = msg.get_payload(decode=True)
                        if body:
                            body = body.decode(charset, errors='replace')
                        else:
                            body = ''
                    # Skip empty bodies
                    if not body.strip():
                        continue
                    # Format as OpenAI training example
                    training_data.append({
                        "messages": [
                            {"role": "system", "content": Config.SYSTEM_PROMPT},
                            {"role": "user", "content": f"Subject: {subject}\nFrom: {from_}\nTo: {to}\nDate: {date}\n\n{body.strip()}"},
                            {"role": "assistant", "content": "[PLACEHOLDER]"}
                        ]
                    })
                except Exception as e:
                    logger.warning(f"Failed to process an email: {e}")
                    continue
        logger.info(f"Processed {len(training_data)} emails from MBOX files.")
        return training_data 