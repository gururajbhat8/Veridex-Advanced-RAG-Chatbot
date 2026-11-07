from fastapi import APIRouter, Depends, HTTPException, status
from src.models.chat import ChatRequest, ChatResponse
from src.services.rag.chain import RAGChainService, get_rag_chain_service

router = APIRouter(tags=["Chat"])


# Define a POST endpoint at "/chat"
# response_model=ChatResponse: Automatically validates the outgoing JSON structure
# status_code=200: Standard HTTP status for successful retrieval
@router.post(
    "/chat",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Query RAG Knowledge Base",
    description="Accepts a user question, retrieves relevant context from Pinecone, and returns a grounded answer with citations."
)
async def chat_endpoint(
    # request: The incoming JSON payload, automatically parsed and validated by Pydantic
    request: ChatRequest,
    # rag_service: Injected by FastAPI using get_rag_chain_service
    # This avoids recreating the chain on every request and allows easy testing
    rag_service: RAGChainService = Depends(get_rag_chain_service)
) -> ChatResponse:
    """
    Handles user chat queries against the indexed knowledge base.
    """
    try:
        # Call our RAG chain service using the fields provided in the request
        response: ChatResponse = rag_service.query(
            question=request.question,
            role=request.role,
            top_k=request.top_k,
            session_id=request.session_id
        )
        
        # Return the structured response back to the client
        return response

    except Exception as e:
        # Catch unexpected errors (e.g. Pinecone timeout, LLM API failure)
        # Raise an HTTP 500 error with a safe, descriptive message
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while processing your chat request: {str(e)}"
        )