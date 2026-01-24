from functools import lru_cache

from src.config import get_settings
from src.services.hf.client import HF_Client


@lru_cache(maxsize=1)
def make_huggingface_client() -> HF_Client:
    """
    Create and return a singleton Huggingface client instance.

    Returns:
        HF_Client: Configured Huggingface client
    """
    settings = get_settings()
    return HF_Client(settings)
