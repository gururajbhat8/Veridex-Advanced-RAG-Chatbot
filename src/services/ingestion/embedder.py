from functools import lru_cache
from typing import List
from langchain_community.embeddings import HuggingFaceEmbeddings
from src.core.config import settings


class EmbeddingService:
    """
    Service responsible for transforming raw text into dense semantic vector representations.
    Uses local HuggingFace sentence-transformers (zero API cost, runs on CPU/GPU).
    """

    def __init__(self, model_name: str = settings.EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        # model_kwargs: 'cpu' or 'cuda' (auto-detected, default to cpu for universal compatibility)
        # encode_kwargs: normalize_embeddings=True ensures cosine similarity is mathematically optimal
        self._embeddings = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )

    def get_embeddings_client(self) -> HuggingFaceEmbeddings:
        """
        Returns the underlying LangChain embeddings object for use in vector stores and retrievers.
        """
        return self._embeddings

    def embed_query(self, text: str) -> List[float]:
        """
        Generates a vector embedding for a single user query.
        """
        return self._embeddings.embed_query(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generates vector embeddings for a batch of text chunks.
        """
        return self._embeddings.embed_documents(texts)


@lru_cache()
def get_embedding_service() -> EmbeddingService:
    """
    Singleton factory: Ensures the embedding model is loaded into memory only once.
    """
    return EmbeddingService()