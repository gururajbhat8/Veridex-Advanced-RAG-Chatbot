from fastapi import APIRouter, Depends, HTTPException, status
from src.models.chat import ChatRequest, ChatResponse
from src.services.rag.chain import RAGChainService, get_rag_chain_service

router = APIRouter(tags=["Chat"])

@router.post(
    "/chat",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Query RAG Knowledge Base",
    description="Accepts a user question, retrieves relevant context from Pinecone, and returns a grounded answer with citations."
)
async def chat_endpoint(
    request: ChatRequest,
    rag_service: RAGChainService = Depends(get_rag_chain_service)
) -> ChatResponse:
    """
    Handles user chat queries against the indexed knowledge base.
    """
    try:
        response: ChatResponse = rag_service.query(
            question=request.question,
            role=request.role,
            top_k=request.top_k,
            session_id=request.session_id
        )
        
        return response

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while processing your chat request: {str(e)}"
        )