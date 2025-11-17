"""
Embedding generation for text using various providers.

Supports OpenAI, Google, HuggingFace, Cohere, and custom embedding models
with a unified interface.
"""

from typing import List, Optional, Union, Dict, Any
from abc import ABC, abstractmethod
import numpy as np


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Input text

        Returns:
            Embedding vector as list of floats
        """
        pass

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of input texts

        Returns:
            List of embedding vectors
        """
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """
        Get the dimension of embeddings.

        Returns:
            Embedding dimension
        """
        pass


class OpenAIEmbedding(EmbeddingProvider):
    """OpenAI embedding provider."""

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-large",
        batch_size: int = 100
    ):
        """
        Initialize OpenAI embedding provider.

        Args:
            api_key: OpenAI API key
            model: Model name (text-embedding-3-large, text-embedding-3-small, text-embedding-ada-002)
            batch_size: Batch size for processing
        """
        self.api_key = api_key
        self.model = model
        self.batch_size = batch_size

        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
        except ImportError:
            raise ImportError("openai package required. Install with: pip install openai")

        # Set dimensions based on model
        self.dimensions = {
            "text-embedding-3-large": 3072,
            "text-embedding-3-small": 1536,
            "text-embedding-ada-002": 1536
        }

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for single text."""
        response = self.client.embeddings.create(
            input=text,
            model=self.model
        )
        return response.data[0].embedding

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts with batching."""
        all_embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            response = self.client.embeddings.create(
                input=batch,
                model=self.model
            )
            embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(embeddings)

        return all_embeddings

    def get_dimension(self) -> int:
        """Get embedding dimension."""
        return self.dimensions.get(self.model, 1536)


class GoogleEmbedding(EmbeddingProvider):
    """Google embedding provider (Vertex AI / Gemini)."""

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-004",
        batch_size: int = 100
    ):
        """
        Initialize Google embedding provider.

        Args:
            api_key: Google API key
            model: Model name
            batch_size: Batch size for processing
        """
        self.api_key = api_key
        self.model = model
        self.batch_size = batch_size

        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self.client = genai
        except ImportError:
            raise ImportError("google-generativeai package required. Install with: pip install google-generativeai")

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for single text."""
        result = self.client.embed_content(
            model=f"models/{self.model}",
            content=text,
            task_type="retrieval_document"
        )
        return result['embedding']

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        all_embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            for text in batch:
                embedding = self.embed_text(text)
                all_embeddings.append(embedding)

        return all_embeddings

    def get_dimension(self) -> int:
        """Get embedding dimension."""
        return 768  # Default for text-embedding-004


class HuggingFaceEmbedding(EmbeddingProvider):
    """HuggingFace embedding provider (local or API)."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-large-en-v1.5",
        use_api: bool = False,
        api_key: Optional[str] = None,
        device: str = "cpu"
    ):
        """
        Initialize HuggingFace embedding provider.

        Args:
            model_name: HuggingFace model name
            use_api: Whether to use HuggingFace API instead of local model
            api_key: HuggingFace API key (required if use_api=True)
            device: Device to run model on (cpu, cuda, mps)
        """
        self.model_name = model_name
        self.use_api = use_api
        self.device = device

        if use_api:
            if not api_key:
                raise ValueError("API key required when use_api=True")
            try:
                from huggingface_hub import InferenceClient
                self.client = InferenceClient(token=api_key)
            except ImportError:
                raise ImportError("huggingface_hub required. Install with: pip install huggingface_hub")
        else:
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(model_name, device=device)
            except ImportError:
                raise ImportError("sentence-transformers required. Install with: pip install sentence-transformers")

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for single text."""
        if self.use_api:
            embedding = self.client.feature_extraction(text, model=self.model_name)
            return embedding
        else:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding.tolist()

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        if self.use_api:
            return [self.embed_text(text) for text in texts]
        else:
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            return embeddings.tolist()

    def get_dimension(self) -> int:
        """Get embedding dimension."""
        if self.use_api:
            # Common dimensions, but should ideally query the model
            return 1024
        else:
            return self.model.get_sentence_embedding_dimension()


class CohereEmbedding(EmbeddingProvider):
    """Cohere embedding provider."""

    def __init__(
        self,
        api_key: str,
        model: str = "embed-english-v3.0",
        batch_size: int = 96
    ):
        """
        Initialize Cohere embedding provider.

        Args:
            api_key: Cohere API key
            model: Model name
            batch_size: Batch size for processing
        """
        self.api_key = api_key
        self.model = model
        self.batch_size = batch_size

        try:
            import cohere
            self.client = cohere.Client(api_key)
        except ImportError:
            raise ImportError("cohere package required. Install with: pip install cohere")

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for single text."""
        response = self.client.embed(
            texts=[text],
            model=self.model,
            input_type="search_document"
        )
        return response.embeddings[0]

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts with batching."""
        all_embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            response = self.client.embed(
                texts=batch,
                model=self.model,
                input_type="search_document"
            )
            all_embeddings.extend(response.embeddings)

        return all_embeddings

    def get_dimension(self) -> int:
        """Get embedding dimension."""
        return 1024  # Default for embed-english-v3.0


class OllamaEmbedding(EmbeddingProvider):
    """Ollama local embedding provider."""

    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434"
    ):
        """
        Initialize Ollama embedding provider.

        Args:
            model: Ollama model name
            base_url: Ollama server URL
        """
        self.model = model
        self.base_url = base_url

        try:
            import ollama
            self.client = ollama.Client(host=base_url)
        except ImportError:
            raise ImportError("ollama package required. Install with: pip install ollama")

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for single text."""
        response = self.client.embeddings(
            model=self.model,
            prompt=text
        )
        return response['embedding']

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        return [self.embed_text(text) for text in texts]

    def get_dimension(self) -> int:
        """Get embedding dimension."""
        return 768  # Default, varies by model


class NebiusEmbedding(EmbeddingProvider):
    """Nebius AI Studio embedding provider (OpenAI-compatible API)."""

    def __init__(
        self,
        api_key: str,
        model: str = "BAAI/bge-en-icl",
        base_url: str = "https://api.studio.nebius.ai/v1/",
        batch_size: int = 100
    ):
        """
        Initialize Nebius embedding provider.

        Args:
            api_key: Nebius API key
            model: Model name (BAAI/bge-en-icl, text-embedding-ada-002, etc.)
            base_url: Nebius API base URL
            batch_size: Batch size for processing
        """
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.batch_size = batch_size

        try:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=api_key,
                base_url=base_url
            )
        except ImportError:
            raise ImportError("openai package required. Install with: pip install openai")

        # Common Nebius embedding dimensions
        self.dimensions = {
            "BAAI/bge-en-icl": 1024,
            "text-embedding-ada-002": 1536,
        }

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for single text."""
        response = self.client.embeddings.create(
            input=text,
            model=self.model
        )
        return response.data[0].embedding

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts with batching."""
        all_embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            response = self.client.embeddings.create(
                input=batch,
                model=self.model
            )
            embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(embeddings)

        return all_embeddings

    def get_dimension(self) -> int:
        """Get embedding dimension."""
        return self.dimensions.get(self.model, 1024)


class CustomEmbedding(EmbeddingProvider):
    """Custom embedding provider using a callable function."""

    def __init__(
        self,
        embed_func: callable,
        dimension: int,
        batch_func: Optional[callable] = None
    ):
        """
        Initialize custom embedding provider.

        Args:
            embed_func: Function that takes text and returns embedding
            dimension: Embedding dimension
            batch_func: Optional function for batch processing
        """
        self.embed_func = embed_func
        self.dimension = dimension
        self.batch_func = batch_func

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for single text."""
        return self.embed_func(text)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        if self.batch_func:
            return self.batch_func(texts)
        return [self.embed_text(text) for text in texts]

    def get_dimension(self) -> int:
        """Get embedding dimension."""
        return self.dimension


class Embedder:
    """
    Main embedder class with provider management.

    This is the primary interface for generating embeddings.
    """

    def __init__(self, provider: EmbeddingProvider):
        """
        Initialize embedder.

        Args:
            provider: EmbeddingProvider instance
        """
        self.provider = provider

    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for single text.

        Args:
            text: Input text

        Returns:
            Embedding vector
        """
        return self.provider.embed_text(text)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of input texts

        Returns:
            List of embedding vectors
        """
        return self.provider.embed_texts(texts)

    def embed_chunks(self, chunks: List) -> List[Dict[str, Any]]:
        """
        Generate embeddings for text chunks.

        Args:
            chunks: List of TextChunk objects

        Returns:
            List of dictionaries with text, embedding, and metadata
        """
        texts = [chunk.text for chunk in chunks]
        embeddings = self.embed_texts(texts)

        return [
            {
                "text": chunk.text,
                "embedding": embedding,
                "metadata": chunk.metadata
            }
            for chunk, embedding in zip(chunks, embeddings)
        ]

    def get_dimension(self) -> int:
        """
        Get embedding dimension.

        Returns:
            Embedding dimension
        """
        return self.provider.get_dimension()

    @staticmethod
    def create_from_config(config) -> "Embedder":
        """
        Create embedder from configuration.

        Args:
            config: EmbeddingConfig instance

        Returns:
            Configured Embedder instance
        """
        provider_map = {
            "openai": lambda: OpenAIEmbedding(
                api_key=config.api_key,
                model=config.model_name,
                batch_size=config.batch_size
            ),
            "google": lambda: GoogleEmbedding(
                api_key=config.api_key,
                model=config.model_name,
                batch_size=config.batch_size
            ),
            "huggingface": lambda: HuggingFaceEmbedding(
                model_name=config.model_name,
                use_api=config.api_key is not None,
                api_key=config.api_key
            ),
            "cohere": lambda: CohereEmbedding(
                api_key=config.api_key,
                model=config.model_name,
                batch_size=config.batch_size
            ),
            "ollama": lambda: OllamaEmbedding(
                model=config.model_name,
                base_url=config.custom_endpoint or "http://localhost:11434"
            ),
            "nebius": lambda: NebiusEmbedding(
                api_key=config.api_key,
                model=config.model_name,
                base_url=config.custom_endpoint or "https://api.studio.nebius.ai/v1/",
                batch_size=config.batch_size
            ),
        }

        if config.provider not in provider_map:
            raise ValueError(f"Unknown provider: {config.provider}")

        provider = provider_map[config.provider]()
        return Embedder(provider)
