"""Model inference functionality."""

from typing import Optional, Dict, Any, Tuple
from openai import OpenAI, BadRequestError
from src.utils.logging import logger
from src.utils.config import Config

class EmailModel:
    """Handles email response generation using the fine-tuned model."""
    
    def __init__(self, client: Optional[OpenAI] = None):
        """Initialize the model with OpenAI client."""
        self.client = client or OpenAI(api_key=Config.OPENAI_API_KEY)
        self.model_id: Optional[str] = None
        self._available_models = None
        self._model_data = {}
        self._model_params = {}
        
    def _get_base_model(self, model_name: str) -> str:
        """Get the base model name without fine-tuning suffix."""
        return model_name.split(':')[0]
        
    def _detect_model_parameters(self, model_name: str) -> Dict[str, Any]:
        """
        Detect required parameters for a model by testing API calls.
        
        Args:
            model_name: Name of the model to check
            
        Returns:
            Dict of supported parameters and their values
        """
        if model_name in self._model_params:
            return self._model_params[model_name]
            
        base_model = self._get_base_model(model_name)
        params = {'seed': 42}  # All models support seed
        
        # Set parameters based on model type
        if any(prefix in base_model for prefix in ['gpt-4', 'gpt-3.5-turbo']):
            params.update({
                'max_tokens': Config.MAX_TOKENS,
                'temperature': Config.TEMPERATURE
            })
        elif base_model.startswith(('o1-', 'o4-')):
            params.update({
                'max_completion_tokens': Config.MAX_TOKENS,
                'temperature': Config.TEMPERATURE  # o1/o4 models support temperature
            })
        else:
            # For unknown models, try parameters in order
            token_param = None
            
            # Test max_tokens
            try:
                self.client.chat.completions.create(
                    model=base_model,
                    messages=[{"role": "user", "content": "test"}],
                    max_tokens=1
                )
                token_param = 'max_tokens'
            except BadRequestError as e:
                error_msg = str(e).lower()
                if "max_completion_tokens" in error_msg:
                    token_param = 'max_completion_tokens'
                else:
                    logger.warning(f"Unexpected token parameter error: {error_msg}")
                    token_param = 'max_tokens'  # Safe default
            
            # Set the detected token parameter
            params[token_param] = Config.MAX_TOKENS
            
            # Test temperature support
            try:
                self.client.chat.completions.create(
                    model=base_model,
                    messages=[{"role": "user", "content": "test"}],
                    temperature=Config.TEMPERATURE,
                    **{token_param: 1}
                )
                params['temperature'] = Config.TEMPERATURE
            except BadRequestError as e:
                error_msg = str(e).lower()
                if "temperature" in error_msg:
                    logger.info(f"Model {base_model} does not support temperature parameter")
                else:
                    logger.warning(f"Unexpected temperature error: {error_msg}")
                    params['temperature'] = Config.TEMPERATURE  # Safe default
        
        logger.info(f"Detected parameters for {model_name}: {params}")
        self._model_params[model_name] = params
        return params
    
    def check_model_availability(self, model_name: Optional[str] = None) -> Tuple[bool, str]:
        """
        Check if the specified model is available and detect its parameters.
        
        Args:
            model_name: Name of the model to check, defaults to Config.MODEL_NAME
            
        Returns:
            Tuple[bool, str]: (is_available, message)
        """
        try:
            # Cache available models to avoid repeated API calls
            if self._available_models is None:
                logger.info("Fetching available models...")
                response = self.client.models.list()
                self._available_models = [model.id for model in response.data]
                self._model_data = {model.id: model for model in response.data}
                logger.debug(f"Found {len(self._available_models)} available models")
            
            target_model = model_name or self.model_id or Config.MODEL_NAME
            # IMPORTANT: Check for the full model ID (not just the base model)
            # This ensures fine-tuned models are recognized. For base models, the full ID is just the base name.
            # This logic works for both base and fine-tuned models.
            if target_model in self._available_models:
                model_data = self._model_data[target_model]
                logger.info(f"✓ Model '{target_model}' is available")
                logger.info("Model details:")
                logger.info(f"- Created: {getattr(model_data, 'created', 'N/A')}")
                logger.info(f"- Owned by: {getattr(model_data, 'owned_by', 'N/A')}")
                logger.info(f"- Object type: {getattr(model_data, 'object', 'N/A')}")
                
                # Detect parameters
                params = self._detect_model_parameters(target_model)
                logger.info("Supported parameters:")
                for param, value in params.items():
                    logger.info(f"- {param}: {value}")
                
                return True, f"Model '{target_model}' is available"
            else:
                # Find suitable alternative models
                chat_models = [m for m in self._available_models if any(
                    prefix in m for prefix in ['gpt-4', 'gpt-3.5-turbo', 'o1-', 'o4-']
                )]
                chat_models.sort()
                message = (
                    f"Model '{target_model}' is not available.\n"
                    f"Available chat models:\n"
                    f"{', '.join(chat_models)}\n"
                    f"Consider using one of these models instead."
                )
                logger.error(message)
                return False, message
                
        except Exception as e:
            message = f"Failed to check model availability: {str(e)}"
            logger.error(message)
            if hasattr(e, 'response'):
                logger.error(f"API Response: {e.response}")
            return False, message
    
    def generate_reply(self, email_content: str) -> Optional[str]:
        """
        Generate a reply to an email.
        
        Args:
            email_content: The email to reply to
            
        Returns:
            Optional[str]: Generated reply if successful, None if failed
        """
        try:
            model = self.model_id or Config.MODEL_NAME
            
            # Check model availability and get parameters
            is_available, message = self.check_model_availability(model)
            if not is_available:
                logger.error(f"Cannot generate reply: {message}")
                return None
            
            model_params = self._detect_model_parameters(model)
            logger.debug(f"Generating reply using model {model} with params: {model_params}")
            
            # Create messages list with simpler prompt
            messages = [
                {"role": "system", "content": "You are an email assistant. Write a reply to the following email."},
                {"role": "user", "content": f"Please write a reply to this email:\n\n{email_content}"}
            ]
            logger.debug(f"Input messages: {messages}")
            
            # Make API call
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                **model_params
            )
            
            # Log full response for debugging
            logger.debug(f"Full API Response: {response}")
            logger.debug(f"Response type: {type(response)}")
            logger.debug(f"Response dir: {dir(response)}")
            
            if not response:
                logger.error("No response from API")
                return None
                
            if not hasattr(response, 'choices') or not response.choices:
                logger.error("No choices in response")
                logger.debug(f"Response attributes: {dir(response)}")
                return None
                
            # Get the first choice
            choice = response.choices[0]
            logger.debug(f"First choice: {choice}")
            logger.debug(f"Choice type: {type(choice)}")
            logger.debug(f"Choice dir: {dir(choice)}")
            
            if not hasattr(choice, 'message'):
                logger.error("No message in choice")
                logger.debug(f"Choice attributes: {dir(choice)}")
                return None
                
            # Get the message
            message = choice.message
            logger.debug(f"Message object: {message}")
            logger.debug(f"Message type: {type(message)}")
            logger.debug(f"Message dir: {dir(message)}")
            
            if not hasattr(message, 'content'):
                logger.error("No content attribute in message")
                logger.debug(f"Message attributes: {dir(message)}")
                return None
                
            # Get the content
            content = message.content
            if not content:
                logger.error("Empty content in message")
                logger.debug(f"Content type: {type(content)}")
                return None
                
            # Clean up the content
            content = content.strip()
            if not content:
                logger.error("Empty content after stripping whitespace")
                return None
                
            logger.debug(f"Generated reply: {content}")
            return content
            
        except Exception as e:
            logger.error(f"Failed to generate reply: {str(e)}")
            if hasattr(e, 'response'):
                logger.error(f"API Response: {e.response}")
            return None
    
    def test_model(self) -> bool:
        """Test the model with a sample email."""
        logger.info("Testing model...")
        
        # First check if the model is available and get its parameters
        is_available, message = self.check_model_availability()
        if not is_available:
            logger.error(f"Cannot test model: {message}")
            return False
        
        try:
            reply = self.generate_reply(Config.TEST_EMAIL)
            if reply:
                logger.info("\n--- Test Reply ---")
                logger.info(reply)
                logger.info("----------------")
                logger.info("✓ Model test complete")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to test model: {e}")
            return False 