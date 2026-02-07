import logging
from groq import Groq

from src.config import Settings
logger = logging.getLogger(__name__)

class GROQ_Client:
    """Response the user query using the Groq API"""
    
    def __init__(self, settings: Settings):
        """Initialize the classifier model."""
        self.medium_model_name = settings.models.MEDIUM
        self.complex_model_name = settings.models.COMPLEX

        self.api_key = settings.groq.API_KEY

    def _build_messages_with_history(self, current_prompt: str, recent_logs: list = None) -> list:

        messages = []
        
        # Add conversation history if available
        if recent_logs:
            for log in recent_logs:
                # Add previous user message
                messages.append({
                    "role": "user",
                    "content": log.prompt
                })
                # Add previous assistant response
                messages.append({
                    "role": "assistant",
                    "content": log.llm_response
                })
            
            logger.info(f"Added {len(recent_logs)} previous exchanges to context")
        
        # Add current user message
        messages.append({
            "role": "user",
            "content": current_prompt
        })
        return messages
    
    
    def medium_task(self, user_prompt: str, recent_logs: list = None) -> str:
        """
        Make a response to the user query

        """
        
        try:
            # Use chat template 
            client = Groq(api_key=self.api_key)

            messages = self._build_messages_with_history(user_prompt, recent_logs)
            
            completion = client.chat.completions.create(
                model = self.medium_model_name,
                messages= messages
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error during executing respose from model assigned to medium task: {e}")
            raise

    def complex_task(self, user_prompt: str, recent_logs: list = None) -> str:
        """
        Make a response to the user query

        """
        
        try:
            # Use chat template 
            client = Groq(api_key=self.api_key)
            messages = self._build_messages_with_history(user_prompt, recent_logs)
            
            completion = client.chat.completions.create(
                model = self.complex_model_name,
                messages= messages
            )
            
            return completion.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error during executing respose from model assigned to complex task: {e}")
            raise


        