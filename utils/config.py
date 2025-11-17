"""
Configuration management for the RAG pipeline.

This module provides centralized configuration management with environment variable support,
allowing flexible configuration across different deployment environments.
"""

import os
from typing import Any, Dict, Optional
from dataclasses import dataclass, field
import json


@dataclass
class EmbeddingConfig:
    """Configuration for embedding providers."""

    provider: str = "openai"  # openai, google, huggingface, cohere, custom
    model_name: str = "text-embedding-3-large"
    api_key: Optional[str] = None
    batch_size: int = 100
    dimension: Optional[int] = None  # Auto-detect if None
    custom_endpoint: Optional[str] = None  # For custom embedding APIs
    extra_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Load API key from environment if not provided."""
        if self.api_key is None:
            if self.provider == "openai":
                self.api_key = os.getenv("OPENAI_API_KEY")
            elif self.provider == "google":
                self.api_key = os.getenv("GOOGLE_API_KEY")
            elif self.provider == "cohere":
                self.api_key = os.getenv("COHERE_API_KEY")
            elif self.provider == "huggingface":
                self.api_key = os.getenv("HUGGINGFACE_API_KEY")
            elif self.provider == "nebius":
                self.api_key = os.getenv("NEBIUS_API_KEY")


@dataclass
class LLMConfig:
    """Configuration for LLM providers."""

    provider: str = "openai"  # openai, anthropic, google, cohere, ollama, custom
    model_name: str = "gpt-4"
    api_key: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 2000
    custom_endpoint: Optional[str] = None  # For custom LLM APIs
    system_prompt: Optional[str] = None
    extra_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Load API key from environment if not provided."""
        if self.api_key is None:
            if self.provider == "openai":
                self.api_key = os.getenv("OPENAI_API_KEY")
            elif self.provider == "anthropic":
                self.api_key = os.getenv("ANTHROPIC_API_KEY")
            elif self.provider == "google":
                self.api_key = os.getenv("GOOGLE_API_KEY")
            elif self.provider == "cohere":
                self.api_key = os.getenv("COHERE_API_KEY")
            elif self.provider == "nebius":
                self.api_key = os.getenv("NEBIUS_API_KEY")


@dataclass
class ChunkingConfig:
    """Configuration for text chunking."""

    chunk_size: int = 800  # In tokens
    chunk_overlap: int = 100  # In tokens
    strategy: str = "token"  # token, sentence, paragraph, custom
    respect_sentence_boundaries: bool = True


@dataclass
class VectorStoreConfig:
    """Configuration for vector store."""

    provider: str = "chroma"  # chroma, faiss, pinecone, qdrant, weaviate, custom
    collection_name: str = "rag_documents"
    persist_directory: str = "./vectorstore_data"
    distance_metric: str = "cosine"  # cosine, l2, ip
    index_type: Optional[str] = None  # For FAISS: Flat, IVF, HNSW
    api_key: Optional[str] = None  # For cloud vector DBs
    custom_endpoint: Optional[str] = None


@dataclass
class RetrievalConfig:
    """Configuration for retrieval."""

    top_k: int = 5
    score_threshold: Optional[float] = None
    enable_reranking: bool = False
    reranker_model: Optional[str] = None
    reranker_top_k: int = 3
    search_type: str = "similarity"  # similarity, mmr, similarity_score_threshold


@dataclass
class IngestionConfig:
    """Configuration for document ingestion."""

    supported_formats: list = field(default_factory=lambda: ["pdf", "txt", "md", "html", "docx", "csv", "json"])
    extract_images: bool = False
    extract_tables: bool = True
    ocr_enabled: bool = False
    clean_text: bool = True


@dataclass
class RAGConfig:
    """Main configuration class for the RAG pipeline."""

    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    vectorstore: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    ingestion: IngestionConfig = field(default_factory=IngestionConfig)

    # Pipeline settings
    verbose: bool = True
    enable_caching: bool = True
    cache_dir: str = "./cache"

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "RAGConfig":
        """
        Create configuration from a dictionary.

        Args:
            config_dict: Configuration dictionary

        Returns:
            RAGConfig instance
        """
        embedding_config = EmbeddingConfig(**config_dict.get("embedding", {}))
        llm_config = LLMConfig(**config_dict.get("llm", {}))
        chunking_config = ChunkingConfig(**config_dict.get("chunking", {}))
        vectorstore_config = VectorStoreConfig(**config_dict.get("vectorstore", {}))
        retrieval_config = RetrievalConfig(**config_dict.get("retrieval", {}))
        ingestion_config = IngestionConfig(**config_dict.get("ingestion", {}))

        return cls(
            embedding=embedding_config,
            llm=llm_config,
            chunking=chunking_config,
            vectorstore=vectorstore_config,
            retrieval=retrieval_config,
            ingestion=ingestion_config,
            verbose=config_dict.get("verbose", True),
            enable_caching=config_dict.get("enable_caching", True),
            cache_dir=config_dict.get("cache_dir", "./cache")
        )

    @classmethod
    def from_json(cls, json_path: str) -> "RAGConfig":
        """
        Load configuration from a JSON file.

        Args:
            json_path: Path to JSON configuration file

        Returns:
            RAGConfig instance
        """
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.

        Returns:
            Configuration dictionary
        """
        return {
            "embedding": {
                "provider": self.embedding.provider,
                "model_name": self.embedding.model_name,
                "batch_size": self.embedding.batch_size,
                "dimension": self.embedding.dimension,
                "custom_endpoint": self.embedding.custom_endpoint,
                "extra_params": self.embedding.extra_params
            },
            "llm": {
                "provider": self.llm.provider,
                "model_name": self.llm.model_name,
                "temperature": self.llm.temperature,
                "max_tokens": self.llm.max_tokens,
                "custom_endpoint": self.llm.custom_endpoint,
                "system_prompt": self.llm.system_prompt,
                "extra_params": self.llm.extra_params
            },
            "chunking": {
                "chunk_size": self.chunking.chunk_size,
                "chunk_overlap": self.chunking.chunk_overlap,
                "strategy": self.chunking.strategy,
                "respect_sentence_boundaries": self.chunking.respect_sentence_boundaries
            },
            "vectorstore": {
                "provider": self.vectorstore.provider,
                "collection_name": self.vectorstore.collection_name,
                "persist_directory": self.vectorstore.persist_directory,
                "distance_metric": self.vectorstore.distance_metric,
                "index_type": self.vectorstore.index_type
            },
            "retrieval": {
                "top_k": self.retrieval.top_k,
                "score_threshold": self.retrieval.score_threshold,
                "enable_reranking": self.retrieval.enable_reranking,
                "reranker_model": self.retrieval.reranker_model,
                "reranker_top_k": self.retrieval.reranker_top_k,
                "search_type": self.retrieval.search_type
            },
            "ingestion": {
                "supported_formats": self.ingestion.supported_formats,
                "extract_images": self.ingestion.extract_images,
                "extract_tables": self.ingestion.extract_tables,
                "ocr_enabled": self.ingestion.ocr_enabled,
                "clean_text": self.ingestion.clean_text
            },
            "verbose": self.verbose,
            "enable_caching": self.enable_caching,
            "cache_dir": self.cache_dir
        }

    def save_to_json(self, json_path: str) -> None:
        """
        Save configuration to a JSON file.

        Args:
            json_path: Path to save JSON configuration
        """
        with open(json_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


def get_default_config() -> RAGConfig:
    """
    Get default RAG configuration.

    Returns:
        Default RAGConfig instance
    """
    return RAGConfig()
