
import logging
from typing import Any, Dict, List, Optional, Literal
from src.config import Settings
from src.services.ollama.prompt import PromptClassification
import ollama


logger = logging.getLogger(__name__)

DifficultyLevel = Literal["simple", "medium", "complex"]


class Ollama_Client:
    """Classifies prompts into simple, medium, or complex difficulty levels using Ollama."""
    
    def __init__(self, settings: Settings):
        """Initialize the Ollama client."""
        self.model_name = settings.models.CLASSIFIER  # Phi 3 Mini model
        self.prompt_builder = PromptClassification()
        self._initialized = True
    
    def _initialize_model(self):
        """Check if model is available locally."""
        if self._initialized:
            return
        
        try:
            logger.info(f"Checking Ollama model: {self.model_name}")
            
            # List available models
            models = ollama.list()
            model_names = [model['name'] for model in models.get('models', [])]
            
            if self.model_name not in model_names:
                logger.warning(f"Model {self.model_name} not found locally. Pulling model...")
                self._pull_model()
            
            self._initialized = True
            logger.info("Ollama client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Ollama client: {e}")
            raise
    
    def _pull_model(self):
        """Pull the Phi 3 Mini model if not available."""
        try:
            logger.info(f"Pulling model {self.model_name}...")
            ollama.pull(self.model_name)
            logger.info(f"Model {self.model_name} pulled successfully")
            
        except Exception as e:
            logger.error(f"Failed to pull model: {e}")
            raise
    
    def classify(self, user_prompt: str) -> str:
        """
        Classify the prompt into simple, medium, or complex.

        Args:
            user_prompt: User prompt to classify
            
        Returns:
            Difficulty level: "simple", "medium", or "complex"
        """
        self._initialize_model()
        
        # Create classification prompt
        prompt = self.prompt_builder.create_classifier_prompt(user_prompt)
        
        try:
            # Generate classification using Ollama
            response = ollama.chat(
                model=self.model_name,
                messages=prompt,
                options={
                    'temperature': 0.0,  # Deterministic output
                    'num_predict': 10,  # Max tokens
                }
            )
            
            generated_text = response['message']['content'].strip()
            return generated_text

        except Exception as e:
            logger.error(f"Error during classification: {e}")
            logger.warning("Defaulting to medium difficulty due to classification error")
            return "medium"

    def easy_task(self, user_prompt: str) -> str:
        """
        Make a response to the user query.

        Args:
            user_prompt: User prompt
            
        Returns:
            str: Response from the model.
        """
        self._initialize_model()

        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."}, 
            {"role": "user", "content": user_prompt}, 
        ]
        full_response = ""   # ✅ accumulator

        try:
            # Generate response using Ollama
            stream = ollama.chat(
                model=self.model_name,
                messages=messages,
                stream = True
            )
            for chunk in stream:
                token = chunk["message"]["content"]

                # Print live
                print(token, end="", flush=True)
                full_response += token

            return full_response

        except Exception as e:
            logger.error(f"Error during executing task from model assigned for easy task: {e}")
            raise