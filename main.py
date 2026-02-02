import logging
import os
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from src.config import get_settings
from src.db.factory import make_database
from src.dependencies import get_db_session
from src.router import ping, route
from src.repositories.request_log import LogsRepository


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

    # Cleanup - Delete all logs before shutting down
    logger.info("Cleaning up logs...")
    try:
        db = database.session_factory()
        try:
            repo = LogsRepository(db)
            deleted_count = repo.delete_all()
            logger.info(f"Deleted {deleted_count} log records")
        finally:
            db.close()
    except Exception as e:
        logger.error(f"Failed to cleanup logs: {e}")
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





