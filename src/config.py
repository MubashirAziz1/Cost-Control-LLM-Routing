import os
from pathlib import Path
from typing import List, Literal, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).parent.parent
ENV_FILE_PATH = PROJECT_ROOT / ".env"


class BaseConfigSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=[".env", str(ENV_FILE_PATH)],
        extra="ignore",
        frozen=True,
        env_nested_delimiter="__",
        case_sensitive=False,
    )

class ModelSettings(BaseConfigSettings):
    model_config = SettingsConfigDict(
        env_file=[".env", str(ENV_FILE_PATH)],
        env_prefix="MODEL__",
        extra="ignore",
        frozen=True,
        case_sensitive=False,
    )

    CLASSIFIER: str = ""
    SIMPLE: str = ""
    MEDIUM: str = ""
    COMPLEX: str = ""

class HuggingFaceSettings(BaseConfigSettings):
    model_config = SettingsConfigDict(
        env_file=[".env", str(ENV_FILE_PATH)],
        env_prefix="HF__",
        extra="ignore",
        frozen=True,
        case_sensitive=False,
    )

    TOKEN: str = ""

class GroqSettings(BaseConfigSettings):
    model_config = SettingsConfigDict(
        env_file=[".env", str(ENV_FILE_PATH)],
        env_prefix="GROQ__",
        extra="ignore",
        frozen=True,
        case_sensitive=False,
    )

    API_KEY: str = ""

class OpenAISettings(BaseConfigSettings):
    model_config = SettingsConfigDict(
        env_file=[".env", str(ENV_FILE_PATH)],
        env_prefix="OPENAI__",
        extra="ignore",
        frozen=True,
        case_sensitive=False,
    )

    API_KEY: str = ""
    BASE_URL: str = ""


class Settings(BaseConfigSettings):
    app_version: str = "0.1.0"
    debug: bool = True
    environment: Literal["development", "staging", "production"] = "development"
    service_name: str = "Cost_Controlled_ChatBot"


    models: ModelSettings = Field(default_factory=ModelSettings)
    huggingface: HuggingFaceSettings = Field(default_factory=HuggingFaceSettings)
    groq: GroqSettings = Field(default_factory=GroqSettings)
    openai: OpenAISettings = Field(default_factory=OpenAISettings)



def get_settings() -> Settings:
    return Settings()

