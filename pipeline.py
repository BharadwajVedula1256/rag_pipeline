"""
Universal RAG Pipeline - Main orchestration module.

This module provides a high-level interface for the entire RAG pipeline,
from document ingestion to query answering.
"""

from typing import List, Dict, Any, Optional, Union
from pathlib import Path

from utils import RAGConfig, get_default_config, get_logger
from ingest import DocumentLoader, Document
from chunking import DocumentChunker, TokenChunker
from embeddings import Embedder
from vectorstore import VectorStoreManager
from retrieval import Retriever
from llm import Generator


class RAGPipeline:
    """
    Universal RAG Pipeline for document ingestion, indexing, and querying.

    This is the main class that orchestrates all RAG components.
    """

    def __init__(self, config: Optional[RAGConfig] = None):
        """
        Initialize RAG Pipeline.

        Args:
            config: RAGConfig instance (uses default if None)
        """
        self.config = config or get_default_config()
        self.logger = get_logger("RAGPipeline")

        # Initialize components
        self.loader: Optional[DocumentLoader] = None
        self.chunker: Optional[DocumentChunker] = None
        self.embedder: Optional[Embedder] = None
        self.vector_store: Optional[VectorStoreManager] = None
        self.retriever: Optional[Retriever] = None
        self.generator: Optional[Generator] = None

        # Track pipeline state
        self.is_initialized = False
        self.is_indexed = False

        if self.config.verbose:
            self.logger.info("RAG Pipeline created", config=self.config.to_dict())

    def initialize(self) -> None:
        """
        Initialize all pipeline components.

        This sets up the loader, chunker, embedder, vector store, retriever, and generator.
        """
        if self.is_initialized:
            self.logger.warning("Pipeline already initialized")
            return

        self.logger.info("Initializing RAG Pipeline components...")

        # Initialize document loader
        self.loader = DocumentLoader(clean_text=self.config.ingestion.clean_text)
        self.logger.info("Document loader initialized")

        # Initialize chunker
        chunking_strategy = TokenChunker(
            chunk_size=self.config.chunking.chunk_size,
            chunk_overlap=self.config.chunking.chunk_overlap,
            respect_sentence_boundaries=self.config.chunking.respect_sentence_boundaries
        )
        self.chunker = DocumentChunker(strategy=chunking_strategy)
        self.logger.info(
            "Chunker initialized",
            chunk_size=self.config.chunking.chunk_size,
            overlap=self.config.chunking.chunk_overlap
        )

        # Initialize embedder
        self.embedder = Embedder.create_from_config(self.config.embedding)
        embedding_dim = self.embedder.get_dimension()
        self.logger.info(
            "Embedder initialized",
            provider=self.config.embedding.provider,
            model=self.config.embedding.model_name,
            dimension=embedding_dim
        )

        # Initialize vector store
        self.vector_store = VectorStoreManager.create_from_config(
            self.config.vectorstore,
            embedding_dim
        )
        self.logger.info(
            "Vector store initialized",
            provider=self.config.vectorstore.provider,
            collection=self.config.vectorstore.collection_name
        )

        # Initialize retriever
        self.retriever = Retriever.create_from_config(
            self.config.retrieval,
            self.vector_store,
            self.embedder
        )
        self.logger.info(
            "Retriever initialized",
            search_type=self.config.retrieval.search_type,
            top_k=self.config.retrieval.top_k
        )

        # Initialize generator
        self.generator = Generator.create_from_config(self.config.llm)
        self.logger.info(
            "Generator initialized",
            provider=self.config.llm.provider,
            model=self.config.llm.model_name
        )

        self.is_initialized = True
        self.logger.info("RAG Pipeline initialization complete")

    def ingest_documents(
        self,
        sources: Union[str, List[str], List[Dict[str, Any]]],
        source_type: str = "auto"
    ) -> List[Document]:
        """
        Ingest documents from various sources.

        Args:
            sources: File path(s), URL(s), or list of source dictionaries
            source_type: Type of source ('file', 'url', 'directory', 'auto')

        Returns:
            List of loaded Document instances
        """
        if not self.is_initialized:
            self.initialize()

        self.logger.info("Starting document ingestion", source_type=source_type)

        documents = []

        # Handle single source
        if isinstance(sources, str):
            if source_type == "auto":
                # Auto-detect
                if sources.startswith("http://") or sources.startswith("https://"):
                    source_type = "url"
                elif Path(sources).is_dir():
                    source_type = "directory"
                else:
                    source_type = "file"

            if source_type == "file":
                documents = [self.loader.load_file(sources)]
            elif source_type == "url":
                documents = [self.loader.load_url(sources)]
            elif source_type == "directory":
                documents = self.loader.load_directory(sources)

        # Handle multiple sources
        elif isinstance(sources, list):
            documents = self.loader.load_multiple(sources)

        self.logger.info(f"Ingested {len(documents)} documents")
        return documents

    def index_documents(self, documents: List[Document]) -> int:
        """
        Index documents into the vector store.

        This includes chunking, embedding, and storing in the vector database.

        Args:
            documents: List of Document instances to index

        Returns:
            Number of chunks indexed
        """
        if not self.is_initialized:
            self.initialize()

        self.logger.info(f"Starting indexing for {len(documents)} documents")

        # Chunk documents
        self.logger.info("Chunking documents...")
        chunks = self.chunker.chunk_documents(documents)
        self.logger.info(f"Created {len(chunks)} chunks")

        # Embed chunks
        self.logger.info("Generating embeddings...")
        embedded_chunks = self.embedder.embed_chunks(chunks)
        self.logger.info(f"Generated {len(embedded_chunks)} embeddings")

        # Store in vector database
        self.logger.info("Storing in vector database...")
        self.vector_store.add_documents(embedded_chunks)
        self.logger.info("Indexing complete")

        # Persist to disk
        self.vector_store.save()
        self.logger.info("Vector store persisted to disk")

        self.is_indexed = True
        return len(chunks)

    def run_ingestion(
        self,
        sources: Union[str, List[str], List[Dict[str, Any]]],
        source_type: str = "auto"
    ) -> int:
        """
        Run complete ingestion pipeline (load + index).

        Args:
            sources: Document sources
            source_type: Type of source

        Returns:
            Number of chunks indexed
        """
        documents = self.ingest_documents(sources, source_type)
        num_chunks = self.index_documents(documents)
        return num_chunks

    def answer_question(
        self,
        query: str,
        top_k: Optional[int] = None,
        enable_reranking: Optional[bool] = None,
        include_sources: bool = True,
        return_raw_results: bool = False
    ) -> Dict[str, Any]:
        """
        Answer a question using the RAG pipeline.

        Args:
            query: User question
            top_k: Number of documents to retrieve (uses config default if None)
            enable_reranking: Whether to enable reranking (uses config default if None)
            include_sources: Whether to include source documents in response
            return_raw_results: Whether to include raw retrieval results

        Returns:
            Dictionary with answer and metadata
        """
        if not self.is_initialized:
            self.initialize()

        if not self.is_indexed:
            raise RuntimeError(
                "No documents indexed. Run run_ingestion() or index_documents() first."
            )

        self.logger.info("Processing query", query=query)

        # Retrieve relevant documents
        k = top_k or self.config.retrieval.top_k
        rerank = enable_reranking if enable_reranking is not None else self.config.retrieval.enable_reranking

        self.logger.info("Retrieving relevant documents", k=k, rerank=rerank)
        search_results = self.retriever.retrieve(
            query=query,
            k=k,
            rerank=rerank,
            rerank_top_k=self.config.retrieval.reranker_top_k if rerank else None
        )

        self.logger.info(f"Retrieved {len(search_results)} documents")

        # Extract context
        context = [result.text for result in search_results]

        # Generate answer
        self.logger.info("Generating answer...")
        result = self.generator.generate(
            query=query,
            context=context,
            include_sources=include_sources
        )

        # Add retrieval information
        result["num_retrieved"] = len(search_results)
        result["retrieval_scores"] = [r.score for r in search_results]

        if return_raw_results:
            result["raw_results"] = [r.to_dict() for r in search_results]

        if include_sources:
            result["source_documents"] = [
                {
                    "text": r.text,
                    "score": r.score,
                    "metadata": r.metadata
                }
                for r in search_results
            ]

        self.logger.info("Query processing complete")
        return result

    def query(self, query: str, **kwargs) -> str:
        """
        Simplified query method that returns just the answer.

        Args:
            query: User question
            **kwargs: Additional parameters for answer_question

        Returns:
            Generated answer string
        """
        result = self.answer_question(query, **kwargs)
        return result["answer"]

    def clear_index(self) -> None:
        """Clear all indexed documents from the vector store."""
        if self.vector_store:
            self.logger.info("Clearing vector store...")
            self.vector_store.clear()
            self.is_indexed = False
            self.logger.info("Vector store cleared")
        else:
            self.logger.warning("Vector store not initialized")

    def save(self) -> None:
        """Persist the vector store to disk."""
        if self.vector_store:
            self.logger.info("Saving vector store...")
            self.vector_store.save()
            self.logger.info("Vector store saved")
        else:
            self.logger.warning("Vector store not initialized")

    @classmethod
    def from_config_file(cls, config_path: str) -> "RAGPipeline":
        """
        Create pipeline from a JSON configuration file.

        Args:
            config_path: Path to JSON config file

        Returns:
            Configured RAGPipeline instance
        """
        config = RAGConfig.from_json(config_path)
        return cls(config)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get pipeline statistics.

        Returns:
            Dictionary with pipeline statistics
        """
        return {
            "is_initialized": self.is_initialized,
            "is_indexed": self.is_indexed,
            "config": self.config.to_dict(),
            "components": {
                "loader": self.loader is not None,
                "chunker": self.chunker is not None,
                "embedder": self.embedder is not None,
                "vector_store": self.vector_store is not None,
                "retriever": self.retriever is not None,
                "generator": self.generator is not None
            }
        }


def create_pipeline(
    embedding_provider: str = "openai",
    embedding_model: str = "text-embedding-3-large",
    llm_provider: str = "openai",
    llm_model: str = "gpt-4",
    vectorstore_provider: str = "chroma",
    **kwargs
) -> RAGPipeline:
    """
    Quick helper to create a RAG pipeline with common settings.

    Args:
        embedding_provider: Embedding provider name
        embedding_model: Embedding model name
        llm_provider: LLM provider name
        llm_model: LLM model name
        vectorstore_provider: Vector store provider
        **kwargs: Additional config parameters

    Returns:
        Configured RAGPipeline instance
    """
    config = get_default_config()
    config.embedding.provider = embedding_provider
    config.embedding.model_name = embedding_model
    config.llm.provider = llm_provider
    config.llm.model_name = llm_model
    config.vectorstore.provider = vectorstore_provider

    # Apply any additional kwargs to config
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return RAGPipeline(config)
