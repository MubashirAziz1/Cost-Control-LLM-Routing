from pydantic import BaseModel, Field
from enum import Enum

class DifficultyLevel(str, Enum):
    simple = "simple"
    medium = "medium"
    complex = "complex"


class Request(BaseModel):
    """Request schema for the /generate endpoint."""
    prompt: str = Field(..., description="User prompt to process", min_length=1)


class Response(BaseModel):
    """Response schema for the /generate endpoint."""
    llm_response: str = Field(..., description="Generated response from the selected model")

    
class LogsCreate(BaseModel):
    """ Schema to store data in db."""
    model_name: str = Field(..., description="Name of the model that generated the response")
    difficulty: DifficultyLevel
    prompt: str = Field(..., description="User prompt to generate response for", min_length=1)
    llm_response: str = Field(..., description="LLM model response against user prompt", min_length=5)

