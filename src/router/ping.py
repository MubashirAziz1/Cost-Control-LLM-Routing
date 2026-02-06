from fastapi import APIRouter
from sqlalchemy import text

from src.dependencies import DatabaseDep, SettingsDep
from src.schemas.api_health import HealthResponse, ServiceStatus
from src.services.ollama.client import Ollama_Client

router = APIRouter()


@router.get("/ping", tags=["Health"])
async def ping():
    """Simple ping endpoint for basic connectivity tests."""
    return {"status": "ok", "message": "pong"}


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check the health and status of the API service including database connectivity.",
    response_description="Service health information",
    tags=["Health"],
)
def health_check(settings: SettingsDep, database: DatabaseDep) -> HealthResponse:
    """
    Comprehensive health check endpoint for monitoring.

    Returns:
        HealthResponse: Contains service status, version, environment, and service checks

    """
    services = {}
    overall_status = "ok"

    # Test database connectivity
    try:
        with database.get_session() as session:
            # Simple query to test connection
            session.execute(text("SELECT 1"))
            services["database"] = ServiceStatus(status="healthy", message="Connected successfully")
    except Exception as e:
        services["database"] = ServiceStatus(status="unhealthy", message=f"Connection failed: {str(e)}")
        overall_status = "degraded"

    return HealthResponse(
        status=overall_status,
        version=settings.app_version,
        environment=settings.environment,
        service_name=settings.service_name,
        services=services,
    )
