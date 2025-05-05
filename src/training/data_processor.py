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
        self.takeout_dir = takeout_dir or str(Config.TAKEOUT_DIR / "Mail")
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
        """Process the takeout directory and return training data from MBOX files, pairing received and sent emails."""
        import mailbox
        import glob
        import os
        from email.header import decode_header

        def decode_str(s):
            if not s:
                return ''
            parts = decode_header(s)
            return ''.join([
                (part.decode(enc or 'utf-8') if isinstance(part, bytes) else part)
                for part, enc in parts
            ])

        def get_email_body(msg):
            body = None
            if msg.is_multipart():
                for part in msg.walk():
                    content_type = part.get_content_type()
                    content_disposition = str(part.get("Content-Disposition"))
                    if content_type == "text/plain" and "attachment" not in content_disposition:
                        try:
                            charset = part.get_content_charset() or 'utf-8'
                            body = part.get_payload(decode=True).decode(charset, errors='replace')
                            break
                        except Exception:
                            continue
            else:
                content_type = msg.get_content_type()
                if content_type == "text/plain":
                    try:
                        charset = msg.get_content_charset() or 'utf-8'
                        body = msg.get_payload(decode=True).decode(charset, errors='replace')
                    except Exception:
                        pass
            return body or ''

        # --- Index all received messages by Message-ID ---
        takeout_mail_dir = self.takeout_dir
        messages_by_id = {}
        all_mbox_files = [f for f in os.listdir(takeout_mail_dir) if f.endswith('.mbox')]
        mailboxes_to_index = [m for m in all_mbox_files if m != 'Sent.mbox']
        for mbox_file in mailboxes_to_index:
            mbox_path = os.path.join(takeout_mail_dir, mbox_file)
            try:
                mbox_obj = mailbox.mbox(mbox_path, factory=None)
                for msg in mbox_obj:
                    if msg is None:
                        continue
                    msg_id = msg.get('Message-ID')
                    if msg_id and msg_id not in messages_by_id:
                        body = get_email_body(msg)
                        messages_by_id[msg_id] = {
                            'body': body.strip() if body else '',
                            'subject': str(msg.get('Subject', '')),
                            'from': str(msg.get('From', ''))
                        }
            except Exception:
                continue
        logger.info(f"Indexed {len(messages_by_id)} received messages by Message-ID.")

        # --- Process Sent.mbox and create training data for matched pairs ---
        sent_mbox_path = os.path.join(takeout_mail_dir, "Sent.mbox")
        training_data = []
        your_email_address = getattr(Config, "YOUR_EMAIL_ADDRESS", None) or "violet@ifp.org"
        unique_from_addresses = set()
        sent_with_reply_header = 0
        total_sent = 0
        if os.path.exists(sent_mbox_path):
            sent_mbox_obj = mailbox.mbox(sent_mbox_path, factory=None)
            for sent_msg in sent_mbox_obj:
                if sent_msg is None:
                    continue
                total_sent += 1
                sender = sent_msg.get('From', '')
                unique_from_addresses.add(sender)
                in_reply_to_id = sent_msg.get('In-Reply-To')
                references = sent_msg.get('References')
                if in_reply_to_id or references:
                    sent_with_reply_header += 1
                if your_email_address not in str(sender).lower():
                    continue
                original_id_to_lookup = None
                if in_reply_to_id:
                    original_id_to_lookup = in_reply_to_id
                elif references:
                    ref_list = references.split()
                    if ref_list:
                        original_id_to_lookup = ref_list[-1]
                original_msg_data = None
                if original_id_to_lookup:
                    original_msg_data = messages_by_id.get(original_id_to_lookup)
                if original_msg_data:
                    original_body = original_msg_data.get('body', '')
                    sent_body_raw = get_email_body(sent_msg)
                    cleaned_sent_body = sent_body_raw.strip() if sent_body_raw else ''
                    if original_body and cleaned_sent_body:
                        training_data.append({
                            "messages": [
                                {"role": "system", "content": Config.SYSTEM_PROMPT},
                                {"role": "user", "content": original_body},
                                {"role": "assistant", "content": cleaned_sent_body}
                            ]
                        })
        logger.info(f"Processed {len(training_data)} matched email-reply pairs from MBOX files.")
        logger.info(f"Unique 'From' addresses in Sent.mbox: {unique_from_addresses}")
        logger.info(f"Total sent emails: {total_sent}")
        logger.info(f"Sent emails with In-Reply-To or References: {sent_with_reply_header}")
        return training_data 