from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class ChatRequest(BaseModel):
    """
    Incoming request payload for querying the RAG chatbot.
    """
    question: str = Field(..., min_length=1, description="The user question to ask the RAG engine")
    
    session_id: Optional[str] = Field(None, description="Unique conversation session identifier")
    
    role: str = Field(default="user", description="The role of the requesting user (e.g. user, admin, manager)")
    
    top_k: int = Field(default=4, ge=1, le=10, description="Number of document chunks to retrieve")


class SourceCitation(BaseModel):
    """
    Metadata representation of a retrieved document chunk used to ground the answer.
    """
    document_id: str = Field(..., description="Unique ID of the source document")
    source: str = Field(..., description="File name of the source document")
    page: int = Field(..., description="Page number where the chunk is located")
    content_snippet: str = Field(..., description="First 200 characters of the retrieved chunk")
    score: Optional[float] = Field(None, description="Similarity score from vector search")
    image_url: Optional[str] = Field(None, description="Optional S3 presigned URL if chunk is an image/diagram")


class ChatResponse(BaseModel):
    """
    Outgoing response payload returned to the frontend or API client.
    """
    answer: str = Field(..., description="Generated answer from the LLM grounded in context")
    sources: List[SourceCitation] = Field(default_factory=list, description="List of source citations used")
    session_id: Optional[str] = Field(None, description="Conversation session ID")

    