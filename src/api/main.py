"""FastAPI application with the /generate endpoint."""
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from models.schemas import GenerateRequest, GenerateResponse
from router.router import ModelRouter
from config import LOG_LEVEL

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Cost Control Smart Model Router",
    description="Routes prompts to different LLMs based on complexity to optimize costs",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize router (singleton)
router_instance = ModelRouter()


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "Cost Control Smart Model Router",
        "version": "0.1.0",
        "endpoints": {
            "/generate": "POST - Generate response based on prompt complexity"
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest) -> GenerateResponse:
    """
    Generate a response by routing the prompt to an appropriate model.
    
    The service:
    1. Classifies the prompt as simple, medium, or complex
    2. Routes to the appropriate model:
       - simple -> Phi-3 mini
       - medium -> Llama 3 70B
       - complex -> GPT-4o
    3. Returns the generated response and model information
    """
    try:
        logger.info(f"Received generation request for prompt: {request.prompt[:100]}...")
        
        # Route and generate
        response, model_name, difficulty = router_instance.route_and_generate(request.prompt)
        
        logger.info(f"Successfully generated response using {model_name}")
        
        return GenerateResponse(
            response=response,
            model_name=model_name,
            difficulty=difficulty
        )
    
    except Exception as e:
        logger.error(f"Error during generation: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error generating response: {str(e)}"
        )
