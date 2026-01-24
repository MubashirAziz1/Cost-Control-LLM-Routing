"""Generator module for different LLM models."""
import logging
from typing import Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from openai import OpenAI

from config import SIMPLE_MODEL, MEDIUM_MODEL, COMPLEX_MODEL, OPENAI_API_KEY, OPENAI_BASE_URL, HF_TOKEN

logger = logging.getLogger(__name__)


class ModelGenerator:
    """Handles generation from different LLM models."""
    
    def __init__(self):
        """Initialize generators for all models."""
        self.simple_model_name = SIMPLE_MODEL
        self.medium_model_name = MEDIUM_MODEL
        self.complex_model_name = COMPLEX_MODEL
        
        # Lazy loading - models will be loaded when needed
        self.simple_tokenizer = None
        self.simple_model = None
        self.medium_tokenizer = None
        self.medium_model = None
        
        # OpenAI client for GPT-4o
        self.openai_client = None
        if OPENAI_API_KEY:
            client_kwargs = {"api_key": OPENAI_API_KEY}
            if OPENAI_BASE_URL:
                client_kwargs["base_url"] = OPENAI_BASE_URL
            self.openai_client = OpenAI(**client_kwargs)
        else:
            logger.warning("OpenAI API key not found. GPT-4o will not be available.")
    
    def _load_simple_model(self):
        """Lazy load the simple model (Phi-3 mini)."""
        if self.simple_model is not None:
            return
        
        try:
            logger.info(f"Loading simple model: {self.simple_model_name}")
            self.simple_tokenizer = AutoTokenizer.from_pretrained(
                self.simple_model_name,
                token=HF_TOKEN,
                trust_remote_code=True
            )
            self.simple_model = AutoModelForCausalLM.from_pretrained(
                self.simple_model_name,
                token=HF_TOKEN,
                trust_remote_code=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
            )
            if not torch.cuda.is_available():
                self.simple_model = self.simple_model.to("cpu")
            self.simple_model.eval()
            logger.info("Simple model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load simple model: {e}")
            raise
    
    def _load_medium_model(self):
        """Lazy load the medium model (Llama 3 70B)."""
        if self.medium_model is not None:
            return
        
        try:
            logger.info(f"Loading medium model: {self.medium_model_name}")
            self.medium_tokenizer = AutoTokenizer.from_pretrained(
                self.medium_model_name,
                token=HF_TOKEN,
                trust_remote_code=True
            )
            self.medium_model = AutoModelForCausalLM.from_pretrained(
                self.medium_model_name,
                token=HF_TOKEN,
                trust_remote_code=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
            )
            if not torch.cuda.is_available():
                self.medium_model = self.medium_model.to("cpu")
            self.medium_model.eval()
            logger.info("Medium model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load medium model: {e}")
            raise
    
    def _generate_with_hf_model(self, prompt: str, model, tokenizer, model_name: str) -> str:
        """Generate response using a Hugging Face model."""
        try:
            # Use chat template if available
            if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template:
                messages = [{"role": "user", "content": prompt}]
                formatted_prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            elif "phi" in model_name.lower():
                formatted_prompt = f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n"
            elif "llama" in model_name.lower():
                formatted_prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            else:
                formatted_prompt = prompt
            
            inputs = tokenizer(
                formatted_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            )
            
            if torch.cuda.is_available():
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the assistant's response
            # Remove the input prompt from the response
            if formatted_prompt in response:
                response = response.replace(formatted_prompt, "").strip()
            # Also check for common markers
            elif "<|assistant|>" in response:
                response = response.split("<|assistant|>")[-1].strip()
            elif "<|start_header_id|>assistant<|end_header_id|>" in response:
                response = response.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
            
            # Clean up any remaining special tokens
            response = response.replace("<|end|>", "").replace("<|eot_id|>", "").strip()
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating with {model_name}: {e}")
            raise
    
    def generate_simple(self, prompt: str) -> str:
        """Generate response using the simple model (Phi-3 mini)."""
        self._load_simple_model()
        logger.info("Generating response with simple model")
        return self._generate_with_hf_model(prompt, self.simple_model, self.simple_tokenizer, self.simple_model_name)
    
    def generate_medium(self, prompt: str) -> str:
        """Generate response using the medium model (Llama 3 70B)."""
        self._load_medium_model()
        logger.info("Generating response with medium model")
        return self._generate_with_hf_model(prompt, self.medium_model, self.medium_tokenizer, self.medium_model_name)
    
    def generate_complex(self, prompt: str) -> str:
        """Generate response using the complex model (GPT-4o)."""
        if not self.openai_client:
            raise ValueError("OpenAI API key not configured. Cannot use GPT-4o.")
        
        logger.info("Generating response with complex model (GPT-4o)")
        try:
            response = self.openai_client.chat.completions.create(
                model=self.complex_model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error generating with GPT-4o: {e}")
            raise
