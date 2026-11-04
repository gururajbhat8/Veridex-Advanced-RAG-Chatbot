
from fastapi import APIRouter, status
from pydantic import BaseModel
from src.core.config import settings

router = APIRouter(tags=["Health"])

class HealthResponse(BaseModel):
    status: str
    app_name: str
    environment: str

@router.get(
    "/health",
    response_model=HealthResponse,
    status_code= status.HTTP_200_OK,
    summary= "service Health Check",
    description="Return the current operation status of the service. "
)

async def health_check() -> HealthResponse:

    return HealthResponse(
        status="healthy",
        app_name=settings.APP_NAME,
        environment=settings.APP_ENV
    )
