from functools import lru_cache
from typing import List, Dict, Any, Tuple
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.services.ingestion.vector_store import get_pinecone_service, PineconeService
from src.services.rag.llm import get_default_llm
from src.models.chat import SourceCitation, ChatResponse


RAG_SYSTEM_PROMPT = """You are Veridex, an enterprise AI assistant for document intelligence.
Your task is to answer the user's question using ONLY the provided context documents below.

Rules you MUST strictly follow:
1. Ground your answer completely in the given context. Do not invent, speculate, or extrapolate facts.
2. If the context does not contain the answer, say exactly:
   "I cannot find the answer to this question in the provided documents."
3. Cite the source document and page number in your answer when referencing specific information.
4. Keep your answer professional, clear, and structured.

Context Documents:
{context}
"""


class RAGChainService:
    """
    Orchestrates the complete Retrieval-Augmented Generation (RAG) pipeline:
    1. Retrieves relevant chunks from Pinecone using vector similarity and role filters.
    2. Formats retrieved chunks with metadata for the LLM.
    3. Invokes the LLM with anti-hallucination system prompt.
    4. Packages the answer with source citations.
    """

    def __init__(self):
        self.pinecone_service: PineconeService = get_pinecone_service()
        
        self.llm = get_default_llm()
        
        self.output_parser = StrOutputParser()

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", RAG_SYSTEM_PROMPT),
            ("human", "{question}")
        ])

        self.generation_chain = self.prompt | self.llm | self.output_parser

    def _format_docs(self, docs: List[Document]) -> str:
        """
        Formats retrieved LangChain documents into a clean string context with source markers.
        """
        formatted_blocks = []
        for i, doc in enumerate(docs, start=1):
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "Unknown")
            formatted_blocks.append(
                f"--- Document {i} [Source: {source}, Page: {page}] ---\n{doc.page_content}"
            )
        return "\n\n".join(formatted_blocks)

    def _extract_citations(self, docs: List[Document]) -> List[SourceCitation]:
        """
        Extracts structured SourceCitation objects from retrieved documents.
        """
        citations = []
        for doc in docs:
            snippet = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            citations.append(
                SourceCitation(
                    document_id=str(doc.metadata.get("document_id", "unknown")),
                    source=str(doc.metadata.get("source", "unknown")),
                    page=int(doc.metadata.get("page", 1)),
                    content_snippet=snippet,
                    image_url=doc.metadata.get("image_url")
                )
            )
        return citations

    def query(
        self,
        question: str,
        role: str = "user",
        top_k: int = 4,
        session_id: str = None
    ) -> ChatResponse:
        """
        Executes the full RAG pipeline for a user question.
        
        Parameters:
        - question: The user's query
        - role: User role for RBAC metadata filtering
        - top_k: Number of chunks to retrieve
        - session_id: Optional conversation session identifier
        """
       
        role_filter = {"role_access": {"$in": [role, "public"]}}

        retrieved_docs: List[Document] = self.pinecone_service.similarity_search(
            query=question,
            k=top_k,
            filter_dict=role_filter
        )

        if not retrieved_docs:
            return ChatResponse(
                answer="No relevant documents were found matching your access permissions.",
                sources=[],
                session_id=session_id
            )

        context_str = self._format_docs(retrieved_docs)

        answer = self.generation_chain.invoke({
            "context": context_str,
            "question": question
        })

        citations = self._extract_citations(retrieved_docs)

        return ChatResponse(
            answer=answer,
            sources=citations,
            session_id=session_id
        )


@lru_cache()
def get_rag_chain_service() -> RAGChainService:
    """
    Singleton factory for RAGChainService.
    """
    return RAGChainService()