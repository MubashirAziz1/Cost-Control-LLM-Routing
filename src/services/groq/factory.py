from functools import lru_cache

from src.config import get_settings
from src.services.groq.client import GROQ_Client


@lru_cache(maxsize=1)
def make_groq_client() -> GROQ_Client:
    """
    Create and return a singleton GROQ client instance.

    Returns:
        GROQ_Client: Configured Groq client
    """
    
    settings = get_settings()
    return GROQ_Client(settings)
