"""Simple interactive script that classifies each user message,
routes it to a model, and logs the conversation to the database."""

import logging
import os
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from src.config import get_settings
from src.db.factory import make_database
from src.router import ping, route


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
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

    logger.info("API ready")
    yield

    # Cleanup
    database.teardown()
    logger.info("API shutdown complete")


app = FastAPI(
    title="Cost Controlled LLM Router",
    description="Cost Controlled Router for the agencies.",
    version=os.getenv("APP_VERSION", "0.1.0"),
    lifespan=lifespan,
) 

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST"],
    allow_headers=["Content-Type"],
)


# Include routers
app.include_router(ping.router) 
app.include_router(route.router) 

from fastapi.staticfiles import StaticFiles

app.mount("/static", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run(app, port=8000, host="0.0.0.0")





