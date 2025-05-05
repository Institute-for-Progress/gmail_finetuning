#!/usr/bin/env python3
"""Script to run the fine-tuned model for email generation."""

import sys
from src.inference.model import EmailModel
from src.utils.config import Config
from src.utils.logging import logger

def main():
    """Run the fine-tuned model."""
    try:
        # Initialize the model
        model = EmailModel()
        
        # Use fine-tuned model if available, otherwise use base model
        if Config.TRAINED_MODEL_ID:
            model.model_id = Config.TRAINED_MODEL_ID
            print("Using fine-tuned model:", Config.TRAINED_MODEL_ID)
        else:
            model.model_id = Config.MODEL_NAME
            print("Using base model:", Config.MODEL_NAME)
        
        # Check model availability
        available, message = model.check_model_availability()
        if not available:
            logger.error(f"Model not available: {message}")
            sys.exit(1)
        
        # Interactive mode
        print("\nFine-tuned Email Assistant")
        print("-------------------------")
        print("Type your email and press Enter twice to generate a reply.")
        print("Type 'quit' to exit.\n")
        
        while True:
            print("\nEnter email content:")
            lines = []
            while True:
                line = input()
                if not line:
                    break
                lines.append(line)
            
            email_content = "\n".join(lines)
            
            if email_content.lower() == "quit":
                break
            
            if not email_content.strip():
                continue
            
            # Generate reply
            reply = model.generate_reply(email_content)
            if reply:
                print("\nGenerated Reply:")
                print("---------------")
                print(reply)
            else:
                print("\nFailed to generate reply. Please try again.")
                
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 