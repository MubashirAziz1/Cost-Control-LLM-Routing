"""Router module that orchestrates classification and generation."""
import logging
from typing import Tuple

from router.classifier import PromptClassifier, DifficultyLevel
from router.generator import ModelGenerator

logger = logging.getLogger(__name__)


class ModelRouter:
    """Routes prompts to appropriate models based on complexity classification."""
    
    def __init__(self):
        """Initialize the router with classifier and generator."""
        self.classifier = PromptClassifier()
        self.generator = ModelGenerator()
        logger.info("ModelRouter initialized")
    
    def route_and_generate(self, prompt: str) -> Tuple[str, str, DifficultyLevel]:
        """
        Classify prompt and route to appropriate model for generation.
        
        Args:
            prompt: User prompt to process
            
        Returns:
            Tuple of (response, model_name, difficulty)
        """
        logger.info(f"Processing prompt (length: {len(prompt)})")
        
        # Step 1: Classify the prompt
        difficulty = self.classifier.classify(prompt)
        logger.info(f"Prompt classified as: {difficulty}")
        
        # Step 2: Route to appropriate generator
        if difficulty == "simple":
            response = self.generator.generate_simple(prompt)
            model_name = self.generator.simple_model_name
        elif difficulty == "medium":
            response = self.generator.generate_medium(prompt)
            model_name = self.generator.medium_model_name
        else:  # complex
            response = self.generator.generate_complex(prompt)
            model_name = self.generator.complex_model_name
        
        logger.info(f"Generated response using model: {model_name}")
        return response, model_name, difficulty
