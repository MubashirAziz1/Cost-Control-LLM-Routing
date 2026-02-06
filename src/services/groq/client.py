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
    
    
    def medium_task(self, user_prompt: str) -> str:
        """
        Make a response to the user query

        """
        
        try:
            # Use chat template 
            client = Groq(api_key=self.api_key)
            
            completion = client.chat.completions.create(
                model = self.medium_model_name,
                messages=[
                {
                    "role": "user",
                    "content": f"{user_prompt}"
                }
                ]
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error during executing respose from model assigned to medium task: {e}")
            raise

    def complex_task(self, user_prompt: str) -> str:
        """
        Make a response to the user query

        """
        
        try:
            # Use chat template 
            client = Groq(api_key=self.api_key)
            
            completion = client.chat.completions.create(
                model = self.complex_model_name,
                messages=[
                {
                    "role": "user",
                    "content": f"{user_prompt}"
                }
                ]
            )
            
            return completion.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error during executing respose from model assigned to complex task: {e}")
            raise


        