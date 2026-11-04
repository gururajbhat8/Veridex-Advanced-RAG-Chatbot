from functools import lru_cache
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    APP_NAME: str = "Veridex-RAG"
    APP_ENV: str = "development"
    DEBUG: bool = True
    PORT: int = 8000
    HOST: str = "0.0.0.0"

    # --- AWS Configuration ---
    AWS_REGION: str = "us-east-1"
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    # --- AWS S3 (Document Store) ---
    S3_BUCKET_NAME: str = "veridex-documents-storage"
    # --- AWS DynamoDB (Chat Memory) ---
    DYNAMODB_TABLE_NAME: str = "veridex-chat-history"

    # --- AWS Bedrock ---
    BEDROCK_MODEL_ID: str = "anthropic.claude-3-5-sonnet-20240620-v1:0"
    # --- Free Fallback LLM (Groq) ---
    GROQ_API_KEY: Optional[str] = None
    GROQ_MODEL_NAME: str = "llama-3.1-8b-instant"
    # --- Vector DB (Pinecone) ---
    PINECONE_API_KEY: Optional[str] = None
    PINECONE_INDEX_NAME: str = "veridex-rag-index"
    PINECONE_ENVIRONMENT: str = "us-east-1"
    # --- Embedding Model (HuggingFace Local) ---
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    # --- AWS Cognito (Auth & RBAC) ---
    COGNITO_USER_POOL_ID: Optional[str] = None
    COGNITO_APP_CLIENT_ID: Optional[str] = None
    # --- Observability (Langfuse) ---
    LANGFUSE_PUBLIC_KEY: Optional[str] = None
    LANGFUSE_SECRET_KEY: Optional[str] = None
    LANGFUSE_HOST: str = "https://cloud.langfuse.com"
@lru_cache()
def get_settings() -> Settings:
    """
    Singleton factory function.
    @lru_cache guarantees that the .env file is read and parsed only ONCE.
    Subsequent calls return the cached Settings instance in memory.
    """
    return Settings()
# Export a global instance for quick imports
settings = get_settings()