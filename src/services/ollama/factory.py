from functools import lru_cache

from src.config import get_settings
from src.services.ollama.client import Ollama_Client


@lru_cache(maxsize=1)
def make_ollama_client() -> Ollama_Client:
    """
    Create and return a singleton Ollama client instance.

    Returns:
        Ollama_Client: Configured Ollam client
    """
    settings = get_settings()
    return Ollama_Client(settings)
