"""Classifier module for routing prompts based on complexity."""
import logging
from typing import Literal
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from config import CLASSIFIER_MODEL, HF_TOKEN

logger = logging.getLogger(__name__)

DifficultyLevel = Literal["simple", "medium", "complex"]


class PromptClassifier:
    """Classifies prompts into simple, medium, or complex difficulty levels."""
    
    def __init__(self):
        """Initialize the classifier model."""
        self.model_name = CLASSIFIER_MODEL
        self.tokenizer = None
        self.model = None
        self._initialized = False
    
    def _initialize_model(self):
        """Lazy initialization of the classifier model."""
        if self._initialized:
            return
        
        try:
            logger.info(f"Loading classifier model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                token=HF_TOKEN,
                trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                token=HF_TOKEN,
                trust_remote_code=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
            )
            if not torch.cuda.is_available():
                self.model = self.model.to("cpu")
            self.model.eval()
            self._initialized = True
            logger.info("Classifier model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load classifier model: {e}")
            raise
    
    def classify(self, prompt: str) -> DifficultyLevel:
        """
        Classify the prompt into simple, medium, or complex.
        
        Args:
            prompt: User prompt to classify
            
        Returns:
            Difficulty level: "simple", "medium", or "complex"
        """
        self._initialize_model()
        
        # Create classification prompt using chat template
        system_message = """You are a prompt classifier. Classify prompts into one of three categories: simple, medium, or complex.

Simple: Basic questions, simple tasks, straightforward requests.
Medium: Moderate complexity, requires reasoning, multi-step tasks.
Complex: High complexity, requires deep reasoning, creative tasks, or extensive analysis.

Respond with only one word: simple, medium, or complex."""
        
        user_message = f"Classify this prompt: {prompt}"
        
        try:
            # Use chat template if available, otherwise format manually
            if hasattr(self.tokenizer, 'apply_chat_template') and self.tokenizer.chat_template:
                messages = [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ]
                classification_prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                # Fallback format
                classification_prompt = f"<|system|>\n{system_message}<|end|>\n<|user|>\n{user_message}<|end|>\n<|assistant|>\n"
            
            # Tokenize and generate
            inputs = self.tokenizer(
                classification_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )
            
            if torch.cuda.is_available():
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False,
                    temperature=0.1,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract classification from response
            classification_text = response.split("Classification:")[-1].strip().lower()
            
            # Parse the classification
            if "simple" in classification_text:
                difficulty = "simple"
            elif "medium" in classification_text:
                difficulty = "medium"
            elif "complex" in classification_text:
                difficulty = "complex"
            else:
                # Default to medium if unclear
                logger.warning(f"Unclear classification, defaulting to medium. Response: {classification_text}")
                difficulty = "medium"
            
            logger.info(f"Classified prompt as: {difficulty}")
            return difficulty
            
        except Exception as e:
            logger.error(f"Error during classification: {e}")
            # Default to medium on error
            logger.warning("Defaulting to medium difficulty due to classification error")
            return "medium"
