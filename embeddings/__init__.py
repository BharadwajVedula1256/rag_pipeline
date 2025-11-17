"""Embeddings package for generating text embeddings."""

from .embedder import (
    Embedder,
    EmbeddingProvider,
    OpenAIEmbedding,
    GoogleEmbedding,
    HuggingFaceEmbedding,
    CohereEmbedding,
    OllamaEmbedding,
    NebiusEmbedding,
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
    "NebiusEmbedding",
    "CustomEmbedding"
]
