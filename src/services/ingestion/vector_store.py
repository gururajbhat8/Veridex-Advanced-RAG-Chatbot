from functools import lru_cache
from typing import List, Optional, Dict, Any
from pinecone import Pinecone, ServerlessSpec
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from src.core.config import settings
from src.services.ingestion.embedder import get_embedding_service


class PineconeService:
    """
    Manages the lifecycle of the Pinecone Vector Database:
    - Index creation & verification
    - Upserting (insert/update) document chunks with metadata
    - Performing similarity vector search with role-based metadata filters
    """

    def __init__(self):
        if not settings.PINECONE_API_KEY:
            raise ValueError(
                "PINECONE_API_KEY is not set in the environment or .env file. "
                "Please add your Pinecone API key to continue."
            )

        self.pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        
        self.index_name = settings.PINECONE_INDEX_NAME
        
        self.dimension = 384

        self.embedding_service = get_embedding_service()
        
        self.embeddings = self.embedding_service.get_embeddings_client()

        self._ensure_index_exists()

        self.index = self.pc.Index(self.index_name)

        self.vector_store = PineconeVectorStore(
            index=self.index,
            embedding=self.embeddings,
            text_key="text"  
        )

    def _ensure_index_exists(self) -> None:
        """
        Private helper method: Checks if the index exists on Pinecone cloud.
        If it does not exist, it creates it automatically with dimension 384 and cosine metric.
        """
        existing_indexes = [idx["name"] for idx in self.pc.list_indexes()]

        if self.index_name not in existing_indexes:
            print(f"[PineconeService] Index '{self.index_name}' not found. Creating serverless index...")
            
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,   
                metric="cosine",               
                spec=ServerlessSpec(
                    cloud="aws",              
                    region="us-east-1"         
                )
            )
            print(f"[PineconeService] Index '{self.index_name}' created successfully.")
        else:
            print(f"[PineconeService] Index '{self.index_name}' already exists. Connecting...")

    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Embeds a list of document chunks and uploads them into Pinecone.
        Returns the list of generated vector IDs.
        """
        # add_documents automatically:
        # 1. Takes the text from each Document
        # 2. Runs our local HuggingFace model to get 384 numbers
        # 3. Sends the vector + text + metadata to Pinecone
        return self.vector_store.add_documents(documents=documents)

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Searches Pinecone for the top-k most semantically relevant chunks.
        
        Parameters:
        - query: The user question as a string
        - k: Number of chunks to retrieve (default: 5)
        - filter_dict: Optional metadata filter for Role-Based Access Control (RBAC),
                       e.g. {"role": {"$in": ["user", "public"]}}
        """
        return self.vector_store.similarity_search(
            query=query,
            k=k,
            filter=filter_dict
        )

    def get_retriever(self, search_kwargs: Optional[Dict[str, Any]] = None):
        """
        Exposes the LangChain retriever interface, which will be connected
        directly to our RAG chain in Phase 4.
        """
        return self.vector_store.as_retriever(search_kwargs=search_kwargs or {"k": 4})


@lru_cache()
def get_pinecone_service() -> PineconeService:
    """
    Singleton factory: Ensures we maintain one authenticated Pinecone connection.
    """
    return PineconeService()