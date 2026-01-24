import logging
from typing import Literal
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

from src.config import Settings
from src.services.hf.prompt import PromptClassification

logger = logging.getLogger(__name__)

DifficultyLevel = Literal["simple", "medium", "complex"]


class HF_Client:
    """Classifies prompts into simple, medium, or complex difficulty levels."""
    
    def __init__(self, settings: Settings):
        """Initialize the classifier model."""
        self.model_name = settings.models.CLASSIFIER
        self.hf_token = settings.huggingface.TOKEN
        self.tokenizer = None
        self.pipe = None
        self.prompt_builder = PromptClassification()
        self._initialized = False
    
    def _initialize_model(self):
        """Lazy initialization of the classifier model."""
        if self._initialized:
            return
        
        try:
            logger.info(f"Loading classifier model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                token=self.hf_token,
                trust_remote_code=True
            )

            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                token=self.hf_token,
                trust_remote_code=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
            )

            if not torch.cuda.is_available():
                self.model = model.to("cpu")

            model.eval()

            self.pipe = pipeline(
                "text-generation",
                model= model,
                tokenizer=self.tokenizer,
            )

            self._initialized = True

            logger.info("Classifier model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load classifier model: {e}")
            raise
    
    def classify(self, user_prompt: str) -> DifficultyLevel:
        """
        Classify the prompt into simple, medium, or complex.

        Args:
            prompt: User prompt to classify
            
        Returns:
            Difficulty level: "simple", "medium", or "complex"
        """

        self._initialize_model()
        
        # Create classification prompt using chat template
        prompt = self.prompt_builder.create_classifier_prompt(user_prompt)
        
        try:
            # Use chat template 
            classification_prompt = self.tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=True
            )
           
            
            generation_args = {
                "max_new_tokens": 500,
                "temperature": 0.0,
                "do_sample": False,
                "return_full_text": False,
            }

            output = self.pipe(classification_prompt, **generation_args)
            
            return output[0]["generated_text"]

        except Exception as e:
            logger.error(f"Error during classification: {e}")

            # Default to medium on error
            logger.warning("Defaulting to medium difficulty due to classification error")
            return "medium"


    def easy_task(self, user_prompt: str) -> str:
        """
        Make a response to the user query

        Args:
            prompt: User prompt
            
        Returns:
            str: Response from the model.
        """

        self._initialize_model()

        message = [ 
            {"role": "system", "content": "You are a helpful AI assistant."}, 
            {"role": "user", "content": f"{user_prompt}"}, 
        ] 
        
        try:
            # Use chat template 
            prompt = self.tokenizer.apply_chat_template(
                message,
                tokenize=False,
                add_generation_prompt=True
            )
           
            
            generation_args = {
                "max_new_tokens": 500,
                "temperature": 0.0,
                "do_sample": False,
                "return_full_text": False,
            }

            output = self.pipe(prompt, **generation_args)
            
            return output[0]["generated_text"]

        except Exception as e:
            logger.error(f"Error during executing task from model assigned for easy task: {e}")
            raise
        