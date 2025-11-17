"""
Retrieval layer for finding relevant documents.

Supports similarity search, MMR, filtering, and optional reranking.
"""

from typing import List, Dict, Any, Optional, Callable
from abc import ABC, abstractmethod
import numpy as np


class SearchResult:
    """Represents a single search result."""

    def __init__(
        self,
        text: str,
        score: float,
        metadata: Optional[Dict[str, Any]] = None,
        doc_id: Optional[str] = None
    ):
        """
        Initialize search result.

        Args:
            text: Retrieved text content
            score: Relevance score
            metadata: Result metadata
            doc_id: Document ID
        """
        self.text = text
        self.score = score
        self.metadata = metadata or {}
        self.doc_id = doc_id

    def __repr__(self) -> str:
        """String representation."""
        preview = self.text[:100] + "..." if len(self.text) > 100 else self.text
        return f"SearchResult(score={self.score:.4f}, preview='{preview}')"

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "text": self.text,
            "score": self.score,
            "metadata": self.metadata,
            "doc_id": self.doc_id
        }


class RetrievalStrategy(ABC):
    """Abstract base class for retrieval strategies."""

    @abstractmethod
    def retrieve(
        self,
        query_embedding: List[float],
        k: int = 5,
        **kwargs
    ) -> List[SearchResult]:
        """
        Retrieve relevant documents.

        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            **kwargs: Additional strategy-specific parameters

        Returns:
            List of SearchResult instances
        """
        pass


class SimilaritySearchStrategy(RetrievalStrategy):
    """Standard similarity search strategy."""

    def __init__(self, vector_store_manager):
        """
        Initialize similarity search strategy.

        Args:
            vector_store_manager: VectorStoreManager instance
        """
        self.vector_store = vector_store_manager

    def retrieve(
        self,
        query_embedding: List[float],
        k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None
    ) -> List[SearchResult]:
        """
        Retrieve documents using similarity search.

        Args:
            query_embedding: Query embedding vector
            k: Number of results
            filter_dict: Optional metadata filter
            score_threshold: Minimum score threshold

        Returns:
            List of SearchResult instances
        """
        results = self.vector_store.search(query_embedding, k, filter_dict)

        search_results = []
        for result in results:
            # Apply score threshold if specified
            if score_threshold and result['score'] < score_threshold:
                continue

            search_results.append(SearchResult(
                text=result['text'],
                score=result['score'],
                metadata=result['metadata'],
                doc_id=result.get('id')
            ))

        return search_results


class MMRSearchStrategy(RetrievalStrategy):
    """
    Maximal Marginal Relevance (MMR) search strategy.

    Balances relevance with diversity to reduce redundancy.
    """

    def __init__(self, vector_store_manager, lambda_mult: float = 0.5):
        """
        Initialize MMR search strategy.

        Args:
            vector_store_manager: VectorStoreManager instance
            lambda_mult: Balance between relevance (1.0) and diversity (0.0)
        """
        self.vector_store = vector_store_manager
        self.lambda_mult = lambda_mult

    def retrieve(
        self,
        query_embedding: List[float],
        k: int = 5,
        fetch_k: int = 20,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Retrieve documents using MMR.

        Args:
            query_embedding: Query embedding vector
            k: Number of final results
            fetch_k: Number of initial candidates to fetch
            filter_dict: Optional metadata filter

        Returns:
            List of SearchResult instances
        """
        # Fetch more candidates than needed
        candidates = self.vector_store.search(query_embedding, fetch_k, filter_dict)

        if len(candidates) <= k:
            return [
                SearchResult(
                    text=c['text'],
                    score=c['score'],
                    metadata=c['metadata'],
                    doc_id=c.get('id')
                )
                for c in candidates
            ]

        # Extract embeddings (would need to be stored in search results)
        # For simplicity, using a placeholder - in production, embeddings should be returned
        selected_indices = self._mmr_selection(
            query_embedding,
            candidates,
            k
        )

        return [
            SearchResult(
                text=candidates[i]['text'],
                score=candidates[i]['score'],
                metadata=candidates[i]['metadata'],
                doc_id=candidates[i].get('id')
            )
            for i in selected_indices
        ]

    def _mmr_selection(
        self,
        query_embedding: List[float],
        candidates: List[Dict],
        k: int
    ) -> List[int]:
        """
        Select k candidates using MMR algorithm.

        Args:
            query_embedding: Query embedding
            candidates: Candidate results
            k: Number to select

        Returns:
            List of selected indices
        """
        # Simplified MMR - in production, would use actual embeddings
        # For now, just return top k by score
        sorted_indices = sorted(
            range(len(candidates)),
            key=lambda i: candidates[i]['score'],
            reverse=True
        )
        return sorted_indices[:k]


class Reranker(ABC):
    """Abstract base class for rerankers."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: int = 5
    ) -> List[SearchResult]:
        """
        Rerank search results.

        Args:
            query: Original query text
            results: Initial search results
            top_k: Number of top results to return

        Returns:
            Reranked list of SearchResult instances
        """
        pass


class CrossEncoderReranker(Reranker):
    """Cross-encoder based reranker for more accurate relevance scoring."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize cross-encoder reranker.

        Args:
            model_name: HuggingFace cross-encoder model name
        """
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(model_name)
        except ImportError:
            raise ImportError(
                "sentence-transformers required for reranking. "
                "Install with: pip install sentence-transformers"
            )

    def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: int = 5
    ) -> List[SearchResult]:
        """
        Rerank using cross-encoder model.

        Args:
            query: Original query text
            results: Initial search results
            top_k: Number of top results to return

        Returns:
            Reranked search results
        """
        if not results:
            return []

        # Prepare pairs for scoring
        pairs = [[query, result.text] for result in results]

        # Get scores
        scores = self.model.predict(pairs)

        # Update scores and sort
        for result, score in zip(results, scores):
            result.score = float(score)

        # Sort by new scores
        reranked = sorted(results, key=lambda x: x.score, reverse=True)

        return reranked[:top_k]


class CohereReranker(Reranker):
    """Cohere Rerank API based reranker."""

    def __init__(self, api_key: str, model: str = "rerank-english-v3.0"):
        """
        Initialize Cohere reranker.

        Args:
            api_key: Cohere API key
            model: Rerank model name
        """
        try:
            import cohere
            self.client = cohere.Client(api_key)
            self.model = model
        except ImportError:
            raise ImportError("cohere required. Install with: pip install cohere")

    def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: int = 5
    ) -> List[SearchResult]:
        """
        Rerank using Cohere Rerank API.

        Args:
            query: Original query text
            results: Initial search results
            top_k: Number of top results to return

        Returns:
            Reranked search results
        """
        if not results:
            return []

        documents = [result.text for result in results]

        response = self.client.rerank(
            query=query,
            documents=documents,
            model=self.model,
            top_n=top_k
        )

        # Map reranked results back to SearchResult objects
        reranked = []
        for item in response.results:
            original_result = results[item.index]
            reranked.append(SearchResult(
                text=original_result.text,
                score=item.relevance_score,
                metadata=original_result.metadata,
                doc_id=original_result.doc_id
            ))

        return reranked


class CustomReranker(Reranker):
    """Custom reranker using a provided function."""

    def __init__(self, rerank_func: Callable):
        """
        Initialize custom reranker.

        Args:
            rerank_func: Function that takes (query, results, top_k) and returns reranked results
        """
        self.rerank_func = rerank_func

    def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: int = 5
    ) -> List[SearchResult]:
        """Apply custom reranking function."""
        return self.rerank_func(query, results, top_k)


class Retriever:
    """
    Main retriever class for document retrieval.

    Orchestrates search strategies and optional reranking.
    """

    def __init__(
        self,
        vector_store_manager,
        embedder,
        strategy: Optional[RetrievalStrategy] = None,
        reranker: Optional[Reranker] = None
    ):
        """
        Initialize retriever.

        Args:
            vector_store_manager: VectorStoreManager instance
            embedder: Embedder instance for query embedding
            strategy: RetrievalStrategy instance
            reranker: Optional Reranker instance
        """
        self.vector_store = vector_store_manager
        self.embedder = embedder
        self.strategy = strategy or SimilaritySearchStrategy(vector_store_manager)
        self.reranker = reranker

    def retrieve(
        self,
        query: str,
        k: int = 5,
        rerank: bool = False,
        rerank_top_k: Optional[int] = None,
        **kwargs
    ) -> List[SearchResult]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: Query text
            k: Number of results to retrieve
            rerank: Whether to apply reranking
            rerank_top_k: Number of results after reranking (defaults to k)
            **kwargs: Additional parameters for retrieval strategy

        Returns:
            List of SearchResult instances
        """
        # Embed query
        query_embedding = self.embedder.embed_text(query)

        # Retrieve using strategy
        if rerank and self.reranker:
            # Fetch more candidates for reranking
            fetch_k = kwargs.pop('fetch_k', k * 3)
            results = self.strategy.retrieve(query_embedding, fetch_k, **kwargs)

            # Rerank
            rerank_k = rerank_top_k or k
            results = self.reranker.rerank(query, results, rerank_k)
        else:
            results = self.strategy.retrieve(query_embedding, k, **kwargs)

        return results

    def retrieve_batch(
        self,
        queries: List[str],
        k: int = 5,
        **kwargs
    ) -> List[List[SearchResult]]:
        """
        Retrieve for multiple queries.

        Args:
            queries: List of query texts
            k: Number of results per query
            **kwargs: Additional parameters

        Returns:
            List of result lists, one per query
        """
        return [self.retrieve(query, k, **kwargs) for query in queries]

    def set_strategy(self, strategy: RetrievalStrategy) -> None:
        """
        Change retrieval strategy.

        Args:
            strategy: New RetrievalStrategy instance
        """
        self.strategy = strategy

    def set_reranker(self, reranker: Optional[Reranker]) -> None:
        """
        Set or update reranker.

        Args:
            reranker: Reranker instance or None to disable
        """
        self.reranker = reranker

    @staticmethod
    def create_from_config(
        config,
        vector_store_manager,
        embedder
    ) -> "Retriever":
        """
        Create retriever from configuration.

        Args:
            config: RetrievalConfig instance
            vector_store_manager: VectorStoreManager instance
            embedder: Embedder instance

        Returns:
            Configured Retriever instance
        """
        # Create strategy
        if config.search_type == "mmr":
            strategy = MMRSearchStrategy(vector_store_manager)
        else:
            strategy = SimilaritySearchStrategy(vector_store_manager)

        # Create reranker if enabled
        reranker = None
        if config.enable_reranking and config.reranker_model:
            # For now, default to CrossEncoder
            reranker = CrossEncoderReranker(config.reranker_model)

        return Retriever(
            vector_store_manager=vector_store_manager,
            embedder=embedder,
            strategy=strategy,
            reranker=reranker
        )
