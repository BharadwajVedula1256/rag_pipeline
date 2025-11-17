"""
Universal RAG Pipeline

A complete, production-ready RAG system supporting any embedding model or LLM.
"""

from .pipeline import RAGPipeline, create_pipeline
from .utils import RAGConfig, get_default_config

__version__ = "1.0.0"
__author__ = "Universal RAG Pipeline"

__all__ = [
    "RAGPipeline",
    "create_pipeline",
    "RAGConfig",
    "get_default_config",
]
