"""
Vector store implementations for storing and retrieving embeddings.

Supports Chroma, FAISS, Pinecone, Qdrant, Weaviate, and custom vector stores.
"""

from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod
import numpy as np
from pathlib import Path


class VectorStore(ABC):
    """Abstract base class for vector stores."""

    @abstractmethod
    def add_vectors(
        self,
        vectors: List[List[float]],
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> None:
        """
        Add vectors to the store.

        Args:
            vectors: List of embedding vectors
            texts: List of text content
            metadatas: List of metadata dictionaries
            ids: Optional list of IDs for the vectors
        """
        pass

    @abstractmethod
    def similarity_search(
        self,
        query_vector: List[float],
        k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors.

        Args:
            query_vector: Query embedding vector
            k: Number of results to return
            filter_dict: Optional metadata filter

        Returns:
            List of dictionaries with text, metadata, and score
        """
        pass

    @abstractmethod
    def delete_collection(self) -> None:
        """Delete the entire collection."""
        pass

    @abstractmethod
    def persist(self) -> None:
        """Persist the vector store to disk."""
        pass


class ChromaVectorStore(VectorStore):
    """ChromaDB vector store implementation."""

    def __init__(
        self,
        collection_name: str = "rag_documents",
        persist_directory: str = "./vectorstore_data",
        distance_metric: str = "cosine"
    ):
        """
        Initialize Chroma vector store.

        Args:
            collection_name: Name of the collection
            persist_directory: Directory to persist data
            distance_metric: Distance metric (cosine, l2, ip)
        """
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError:
            raise ImportError("chromadb required. Install with: pip install chromadb")

        self.collection_name = collection_name
        self.persist_directory = persist_directory

        # Create persist directory
        Path(persist_directory).mkdir(parents=True, exist_ok=True)

        # Initialize client
        self.client = chromadb.PersistentClient(path=persist_directory)

        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": distance_metric}
        )

    def add_vectors(
        self,
        vectors: List[List[float]],
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> None:
        """Add vectors to Chroma collection."""
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(vectors))]

        if metadatas is None:
            metadatas = [{} for _ in range(len(vectors))]

        # Chroma requires metadata values to be strings, ints, or floats
        cleaned_metadatas = []
        for metadata in metadatas:
            cleaned = {}
            for k, v in metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    cleaned[k] = v
                else:
                    cleaned[k] = str(v)
            cleaned_metadatas.append(cleaned)

        self.collection.add(
            embeddings=vectors,
            documents=texts,
            metadatas=cleaned_metadatas,
            ids=ids
        )

    def similarity_search(
        self,
        query_vector: List[float],
        k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors in Chroma."""
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=k,
            where=filter_dict
        )

        # Format results
        formatted_results = []
        for i in range(len(results['ids'][0])):
            formatted_results.append({
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'score': 1 - results['distances'][0][i] if results['distances'] else 1.0,
                'id': results['ids'][0][i]
            })

        return formatted_results

    def delete_collection(self) -> None:
        """Delete the Chroma collection."""
        self.client.delete_collection(name=self.collection_name)

    def persist(self) -> None:
        """Persist data (automatic with PersistentClient)."""
        pass  # Chroma PersistentClient auto-persists


class FAISSVectorStore(VectorStore):
    """FAISS vector store implementation."""

    def __init__(
        self,
        dimension: int,
        index_type: str = "Flat",
        persist_directory: str = "./vectorstore_data",
        distance_metric: str = "cosine"
    ):
        """
        Initialize FAISS vector store.

        Args:
            dimension: Embedding dimension
            index_type: FAISS index type (Flat, IVF, HNSW)
            persist_directory: Directory to persist data
            distance_metric: Distance metric (cosine, l2, ip)
        """
        try:
            import faiss
        except ImportError:
            raise ImportError("faiss-cpu or faiss-gpu required. Install with: pip install faiss-cpu")

        self.dimension = dimension
        self.index_type = index_type
        self.persist_directory = persist_directory
        self.distance_metric = distance_metric

        Path(persist_directory).mkdir(parents=True, exist_ok=True)

        # Create index based on type and metric
        if distance_metric == "cosine":
            self.index = faiss.IndexFlatIP(dimension)  # Inner product for normalized vectors
            self.normalize = True
        elif distance_metric == "l2":
            self.index = faiss.IndexFlatL2(dimension)
            self.normalize = False
        else:  # ip (inner product)
            self.index = faiss.IndexFlatIP(dimension)
            self.normalize = False

        # Storage for texts and metadata
        self.texts: List[str] = []
        self.metadatas: List[Dict[str, Any]] = []
        self.ids: List[str] = []

        # Try to load existing index
        self._load_if_exists()

    def _load_if_exists(self) -> None:
        """Load index from disk if it exists."""
        import faiss
        import pickle

        index_path = Path(self.persist_directory) / "index.faiss"
        metadata_path = Path(self.persist_directory) / "metadata.pkl"

        if index_path.exists() and metadata_path.exists():
            self.index = faiss.read_index(str(index_path))
            with open(metadata_path, 'rb') as f:
                data = pickle.load(f)
                self.texts = data['texts']
                self.metadatas = data['metadatas']
                self.ids = data['ids']

    def add_vectors(
        self,
        vectors: List[List[float]],
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> None:
        """Add vectors to FAISS index."""
        vectors_array = np.array(vectors).astype('float32')

        if self.normalize:
            # Normalize vectors for cosine similarity
            faiss.normalize_L2(vectors_array)

        self.index.add(vectors_array)

        self.texts.extend(texts)
        self.metadatas.extend(metadatas or [{} for _ in range(len(vectors))])

        if ids is None:
            start_idx = len(self.ids)
            ids = [f"doc_{start_idx + i}" for i in range(len(vectors))]
        self.ids.extend(ids)

    def similarity_search(
        self,
        query_vector: List[float],
        k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors in FAISS."""
        query_array = np.array([query_vector]).astype('float32')

        if self.normalize:
            faiss.normalize_L2(query_array)

        distances, indices = self.index.search(query_array, k)

        # Format results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.texts):
                metadata = self.metadatas[idx]

                # Apply metadata filter if provided
                if filter_dict:
                    if not all(metadata.get(k) == v for k, v in filter_dict.items()):
                        continue

                results.append({
                    'text': self.texts[idx],
                    'metadata': metadata,
                    'score': float(distances[0][i]),
                    'id': self.ids[idx]
                })

        return results

    def delete_collection(self) -> None:
        """Delete all data."""
        import faiss
        if self.distance_metric == "cosine" or self.distance_metric == "ip":
            self.index = faiss.IndexFlatIP(self.dimension)
        else:
            self.index = faiss.IndexFlatL2(self.dimension)

        self.texts = []
        self.metadatas = []
        self.ids = []

    def persist(self) -> None:
        """Persist FAISS index and metadata to disk."""
        import faiss
        import pickle

        index_path = Path(self.persist_directory) / "index.faiss"
        metadata_path = Path(self.persist_directory) / "metadata.pkl"

        faiss.write_index(self.index, str(index_path))

        with open(metadata_path, 'wb') as f:
            pickle.dump({
                'texts': self.texts,
                'metadatas': self.metadatas,
                'ids': self.ids
            }, f)


class CustomVectorStore(VectorStore):
    """Custom vector store using provided functions."""

    def __init__(
        self,
        add_func: callable,
        search_func: callable,
        delete_func: callable,
        persist_func: Optional[callable] = None
    ):
        """
        Initialize custom vector store.

        Args:
            add_func: Function to add vectors
            search_func: Function to search vectors
            delete_func: Function to delete collection
            persist_func: Optional function to persist data
        """
        self.add_func = add_func
        self.search_func = search_func
        self.delete_func = delete_func
        self.persist_func = persist_func

    def add_vectors(
        self,
        vectors: List[List[float]],
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> None:
        """Add vectors using custom function."""
        self.add_func(vectors, texts, metadatas, ids)

    def similarity_search(
        self,
        query_vector: List[float],
        k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search using custom function."""
        return self.search_func(query_vector, k, filter_dict)

    def delete_collection(self) -> None:
        """Delete using custom function."""
        self.delete_func()

    def persist(self) -> None:
        """Persist using custom function if provided."""
        if self.persist_func:
            self.persist_func()


class VectorStoreManager:
    """
    Manager for vector store operations.

    Provides high-level interface for vector storage and retrieval.
    """

    def __init__(self, store: VectorStore):
        """
        Initialize vector store manager.

        Args:
            store: VectorStore instance
        """
        self.store = store

    def add_documents(
        self,
        embedded_chunks: List[Dict[str, Any]]
    ) -> None:
        """
        Add embedded chunks to vector store.

        Args:
            embedded_chunks: List of dicts with 'text', 'embedding', 'metadata'
        """
        vectors = [chunk['embedding'] for chunk in embedded_chunks]
        texts = [chunk['text'] for chunk in embedded_chunks]
        metadatas = [chunk['metadata'] for chunk in embedded_chunks]

        self.store.add_vectors(vectors, texts, metadatas)

    def search(
        self,
        query_embedding: List[float],
        k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents.

        Args:
            query_embedding: Query embedding vector
            k: Number of results
            filter_dict: Optional metadata filter

        Returns:
            List of search results
        """
        return self.store.similarity_search(query_embedding, k, filter_dict)

    def clear(self) -> None:
        """Clear all data from vector store."""
        self.store.delete_collection()

    def save(self) -> None:
        """Persist vector store to disk."""
        self.store.persist()

    @staticmethod
    def create_from_config(config, embedding_dimension: int) -> "VectorStoreManager":
        """
        Create vector store manager from configuration.

        Args:
            config: VectorStoreConfig instance
            embedding_dimension: Dimension of embeddings

        Returns:
            Configured VectorStoreManager instance
        """
        if config.provider == "chroma":
            store = ChromaVectorStore(
                collection_name=config.collection_name,
                persist_directory=config.persist_directory,
                distance_metric=config.distance_metric
            )
        elif config.provider == "faiss":
            store = FAISSVectorStore(
                dimension=embedding_dimension,
                index_type=config.index_type or "Flat",
                persist_directory=config.persist_directory,
                distance_metric=config.distance_metric
            )
        else:
            raise ValueError(f"Unknown vector store provider: {config.provider}")

        return VectorStoreManager(store)
