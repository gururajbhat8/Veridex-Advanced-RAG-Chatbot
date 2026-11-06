from functools import lru_cache
from typing import Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_groq import ChatGroq
from src.core.config import settings


class LLMFactory:
    """
    Factory class responsible for providing Chat Model instances.
    """

    @staticmethod
    def get_llm(
        temperature: float = 0.1,
        max_tokens: int = 1024,
        streaming: bool = True
    ) -> BaseChatModel:
        """
        Instantiates and returns an LLM client.
        
        Parameters:
        - temperature: Degree of randomness (0.0 to 1.0). 
                       Set to 0.1 for RAG to prevent hallucinations and enforce factual accuracy.
        - max_tokens: Maximum number of tokens the model can generate in a single response.
        - streaming: Whether to stream the response chunk-by-chunk for low latency UI updates.
        """
        if not settings.GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY is not set in your .env file. "
            )

        # Initialize ChatGroq with model settings
        llm = ChatGroq(
            api_key=settings.GROQ_API_KEY,
            model_name=settings.GROQ_MODEL_NAME,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=streaming
        )

        return llm


@lru_cache()
def get_default_llm() -> BaseChatModel:
    """
    Singleton factory: Returns a shared default LLM instance with standard RAG parameters.
    """
    return LLMFactory.get_llm(temperature=0.1, max_tokens=1024, streaming=False)