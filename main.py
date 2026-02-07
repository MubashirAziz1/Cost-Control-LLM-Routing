import logging
import os
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.config import get_settings
from src.db.factory import make_database
from src.services.ollama.factory import make_ollama_client
from src.services.groq.factory import make_groq_client
from src.router import ping, route
from src.repositories.request_log import LogsRepository
from src.tasks.cleanup import start_cleanup_task  # ← ADD THIS IMPORT



# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename= 'app.log'
)
logger = logging.getLogger(__name__)



@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan for the API.
    """
    logger.info("Starting RAG API...")

    settings = get_settings()
    app.state.settings = settings

    database = make_database()
    app.state.database = database
    logger.info("Database connected")

    app.state.ollama_client = make_ollama_client
    app.state.groq_client = make_groq_client
    logging.info("Services initialized: Ollama , Groq API")

    # ===== START BACKGROUND CLEANUP TASK ===== ← ADD THIS SECTION
    start_cleanup_task(
        inactive_minutes=2,        # Delete sessions inactive for 30+ minutes
        check_interval_seconds=30  # Check every 5 minutes (300 seconds)
    )
    logger.info("Background cleanup task started")
    # =========================================

    logger.info("API ready")
    yield


    database.teardown()
    logger.info("API shutdown complete")


app = FastAPI(
    title="Cost Controlled LLM Router",
    description="Cost Controlled Router for the agencies.",
    version=os.getenv("APP_VERSION", "0.1.0"),
    lifespan=lifespan,
)

# IMPORTANT: Configure CORS middleware BEFORE including routers
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],  # List all possible frontend origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],  # Allow all headers
    expose_headers=["*"],  # Expose all headers to frontend
)

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Include routers
app.include_router(ping.router)
app.include_router(route.router)

# Serve index.html 
@app.get("/")
async def serve_index():
    return FileResponse("static/index.html")


if __name__ == "__main__":
    uvicorn.run(app, port=8000, host="0.0.0.0")