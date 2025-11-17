"""Ingest package for document loading and processing."""

from .loader import DocumentLoader, Document
from .extractor import UniversalExtractor
from .cleaner import TextCleaningPipeline, get_default_pipeline, get_aggressive_pipeline

__all__ = [
    "DocumentLoader",
    "Document",
    "UniversalExtractor",
    "TextCleaningPipeline",
    "get_default_pipeline",
    "get_aggressive_pipeline"
]
