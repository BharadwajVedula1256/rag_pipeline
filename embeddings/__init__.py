"""Embeddings package for generating text embeddings."""

from .embedder import (
    Embedder,
    EmbeddingProvider,
    OpenAIEmbedding,
    GoogleEmbedding,
    HuggingFaceEmbedding,
    CohereEmbedding,
    OllamaEmbedding,
    CustomEmbedding
)

__all__ = [
    "Embedder",
    "EmbeddingProvider",
    "OpenAIEmbedding",
    "GoogleEmbedding",
    "HuggingFaceEmbedding",
    "CohereEmbedding",
    "OllamaEmbedding",
    "CustomEmbedding"
]
