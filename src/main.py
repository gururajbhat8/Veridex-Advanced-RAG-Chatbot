from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.core.config import settings
from src.api.routes import health

app = FastAPI(
    title = settings.APP_NAME,
    description="Enterprise Multimodal RAG Chatbot Backend API",
    version="1.0.0",
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
)


app.add_middleware(
    
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials = True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(health.router, prefix= "/api/v1")
 
@app.get("/", tags=["Root"])
async def root():
    return {
        "message": f"Welcome to {settings.APP_NAME} API",
        "docs": "/docs" if settings.DEBUG else "Swagger docs disabled in production"
    }