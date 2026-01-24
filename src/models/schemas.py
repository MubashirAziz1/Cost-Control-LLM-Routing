from pydantic import BaseModel, Field


class Request(BaseModel):
    """Request schema for the /generate endpoint."""
    prompt: str = Field(..., description="User prompt to generate response for", min_length=1)


class Response(BaseModel):
    """Response schema for the /generate endpoint."""
    response: str = Field(..., description="Generated response from the selected model")
    model_name: str = Field(..., description="Name of the model that generated the response")
    difficulty: str = Field(..., description="Classified difficulty level (simple/medium/complex)")
