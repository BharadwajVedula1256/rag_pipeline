"""
Text chunking strategies for splitting documents into manageable pieces.

Supports token-based, sentence-based, and custom chunking strategies
with configurable overlap for better context preservation.
"""

from typing import List, Dict, Any, Optional, Callable
from abc import ABC, abstractmethod
import re


class TextChunk:
    """
    Represents a chunk of text with metadata.

    Maintains reference to source document and position information.
    """

    def __init__(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        chunk_index: int = 0
    ):
        """
        Initialize a text chunk.

        Args:
            text: Chunk text content
            metadata: Metadata dictionary
            chunk_index: Index of this chunk in the sequence
        """
        self.text = text
        self.metadata = metadata or {}
        self.chunk_index = chunk_index

    def __repr__(self) -> str:
        """String representation."""
        preview = self.text[:50] + "..." if len(self.text) > 50 else self.text
        return f"TextChunk(index={self.chunk_index}, length={len(self.text)}, preview='{preview}')"

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert chunk to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "text": self.text,
            "metadata": self.metadata,
            "chunk_index": self.chunk_index
        }


class ChunkingStrategy(ABC):
    """Abstract base class for chunking strategies."""

    @abstractmethod
    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[TextChunk]:
        """
        Split text into chunks.

        Args:
            text: Input text
            metadata: Metadata to attach to chunks

        Returns:
            List of TextChunk instances
        """
        pass


class TokenChunker(ChunkingStrategy):
    """
    Token-based chunking strategy.

    Splits text based on token count with configurable overlap.
    """

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
        encoding_name: str = "cl100k_base",
        respect_sentence_boundaries: bool = True
    ):
        """
        Initialize token-based chunker.

        Args:
            chunk_size: Target chunk size in tokens
            chunk_overlap: Number of overlapping tokens between chunks
            encoding_name: Tiktoken encoding name (cl100k_base for GPT-4, p50k_base for GPT-3)
            respect_sentence_boundaries: Try to split at sentence boundaries
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding_name = encoding_name
        self.respect_sentence_boundaries = respect_sentence_boundaries

        # Initialize tokenizer
        try:
            import tiktoken
            self.encoder = tiktoken.get_encoding(encoding_name)
        except ImportError:
            raise ImportError(
                "tiktoken is required for token-based chunking. "
                "Install with: pip install tiktoken"
            )

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.

        Args:
            text: Input text

        Returns:
            Number of tokens
        """
        return len(self.encoder.encode(text))

    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[TextChunk]:
        """
        Chunk text based on token count.

        Args:
            text: Input text
            metadata: Metadata to attach to chunks

        Returns:
            List of TextChunk instances
        """
        if not text.strip():
            return []

        # Split into sentences if respecting boundaries
        if self.respect_sentence_boundaries:
            sentences = self._split_into_sentences(text)
            return self._chunk_sentences(sentences, metadata)
        else:
            return self._chunk_tokens(text, metadata)

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.

        Args:
            text: Input text

        Returns:
            List of sentences
        """
        # Simple sentence splitting (can be improved with spaCy or NLTK)
        sentence_endings = r'[.!?]+[\s\n]+'
        sentences = re.split(sentence_endings, text)
        return [s.strip() for s in sentences if s.strip()]

    def _chunk_sentences(
        self,
        sentences: List[str],
        metadata: Optional[Dict[str, Any]]
    ) -> List[TextChunk]:
        """
        Chunk by combining sentences while respecting token limits.

        Args:
            sentences: List of sentences
            metadata: Metadata to attach to chunks

        Returns:
            List of TextChunk instances
        """
        chunks = []
        current_chunk = []
        current_tokens = 0
        chunk_index = 0

        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)

            # If single sentence exceeds chunk size, split it
            if sentence_tokens > self.chunk_size:
                # Save current chunk if not empty
                if current_chunk:
                    chunk_text = " ".join(current_chunk)
                    chunks.append(TextChunk(chunk_text, metadata.copy() if metadata else {}, chunk_index))
                    chunk_index += 1
                    current_chunk = []
                    current_tokens = 0

                # Split long sentence by tokens
                sentence_chunks = self._chunk_tokens(sentence, metadata, start_index=chunk_index)
                chunks.extend(sentence_chunks)
                chunk_index += len(sentence_chunks)
                continue

            # Check if adding sentence would exceed limit
            if current_tokens + sentence_tokens > self.chunk_size:
                # Save current chunk
                if current_chunk:
                    chunk_text = " ".join(current_chunk)
                    chunks.append(TextChunk(chunk_text, metadata.copy() if metadata else {}, chunk_index))
                    chunk_index += 1

                # Start new chunk with overlap
                if self.chunk_overlap > 0 and current_chunk:
                    # Keep sentences that fit in overlap
                    overlap_chunk = []
                    overlap_tokens = 0
                    for sent in reversed(current_chunk):
                        sent_tokens = self.count_tokens(sent)
                        if overlap_tokens + sent_tokens <= self.chunk_overlap:
                            overlap_chunk.insert(0, sent)
                            overlap_tokens += sent_tokens
                        else:
                            break
                    current_chunk = overlap_chunk
                    current_tokens = overlap_tokens
                else:
                    current_chunk = []
                    current_tokens = 0

            # Add sentence to current chunk
            current_chunk.append(sentence)
            current_tokens += sentence_tokens

        # Add final chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(TextChunk(chunk_text, metadata.copy() if metadata else {}, chunk_index))

        return chunks

    def _chunk_tokens(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]],
        start_index: int = 0
    ) -> List[TextChunk]:
        """
        Chunk text directly by tokens (no sentence boundary respect).

        Args:
            text: Input text
            metadata: Metadata to attach to chunks
            start_index: Starting chunk index

        Returns:
            List of TextChunk instances
        """
        tokens = self.encoder.encode(text)
        chunks = []
        chunk_index = start_index

        start = 0
        while start < len(tokens):
            end = start + self.chunk_size
            chunk_tokens = tokens[start:end]
            chunk_text = self.encoder.decode(chunk_tokens)

            chunks.append(TextChunk(chunk_text, metadata.copy() if metadata else {}, chunk_index))
            chunk_index += 1

            # Move start position with overlap
            start = end - self.chunk_overlap if self.chunk_overlap > 0 else end

        return chunks


class SentenceChunker(ChunkingStrategy):
    """
    Sentence-based chunking strategy.

    Combines sentences into chunks of approximately equal size.
    """

    def __init__(self, sentences_per_chunk: int = 5, overlap_sentences: int = 1):
        """
        Initialize sentence-based chunker.

        Args:
            sentences_per_chunk: Number of sentences per chunk
            overlap_sentences: Number of sentences to overlap
        """
        self.sentences_per_chunk = sentences_per_chunk
        self.overlap_sentences = overlap_sentences

    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[TextChunk]:
        """
        Chunk text by sentences.

        Args:
            text: Input text
            metadata: Metadata to attach to chunks

        Returns:
            List of TextChunk instances
        """
        sentences = self._split_into_sentences(text)
        chunks = []
        chunk_index = 0

        for i in range(0, len(sentences), self.sentences_per_chunk - self.overlap_sentences):
            chunk_sentences = sentences[i:i + self.sentences_per_chunk]
            chunk_text = " ".join(chunk_sentences)

            chunks.append(TextChunk(chunk_text, metadata.copy() if metadata else {}, chunk_index))
            chunk_index += 1

        return chunks

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        sentence_endings = r'[.!?]+[\s\n]+'
        sentences = re.split(sentence_endings, text)
        return [s.strip() for s in sentences if s.strip()]


class ParagraphChunker(ChunkingStrategy):
    """
    Paragraph-based chunking strategy.

    Splits text by paragraphs (double newlines).
    """

    def __init__(self, max_paragraphs_per_chunk: int = 3):
        """
        Initialize paragraph-based chunker.

        Args:
            max_paragraphs_per_chunk: Maximum paragraphs per chunk
        """
        self.max_paragraphs_per_chunk = max_paragraphs_per_chunk

    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[TextChunk]:
        """
        Chunk text by paragraphs.

        Args:
            text: Input text
            metadata: Metadata to attach to chunks

        Returns:
            List of TextChunk instances
        """
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        chunks = []
        chunk_index = 0

        for i in range(0, len(paragraphs), self.max_paragraphs_per_chunk):
            chunk_paragraphs = paragraphs[i:i + self.max_paragraphs_per_chunk]
            chunk_text = "\n\n".join(chunk_paragraphs)

            chunks.append(TextChunk(chunk_text, metadata.copy() if metadata else {}, chunk_index))
            chunk_index += 1

        return chunks


class CustomChunker(ChunkingStrategy):
    """Apply a custom chunking function."""

    def __init__(self, func: Callable[[str], List[str]]):
        """
        Initialize custom chunker.

        Args:
            func: Function that takes text and returns list of chunk strings
        """
        self.func = func

    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[TextChunk]:
        """
        Apply custom chunking function.

        Args:
            text: Input text
            metadata: Metadata to attach to chunks

        Returns:
            List of TextChunk instances
        """
        chunk_texts = self.func(text)
        return [
            TextChunk(chunk_text, metadata.copy() if metadata else {}, i)
            for i, chunk_text in enumerate(chunk_texts)
        ]


class DocumentChunker:
    """
    Main chunker class for splitting documents into chunks.

    Wraps chunking strategies and handles document-level operations.
    """

    def __init__(self, strategy: Optional[ChunkingStrategy] = None):
        """
        Initialize document chunker.

        Args:
            strategy: ChunkingStrategy instance (creates default TokenChunker if None)
        """
        self.strategy = strategy or TokenChunker()

    def chunk_document(self, document) -> List[TextChunk]:
        """
        Chunk a document.

        Args:
            document: Document instance (from ingest.loader)

        Returns:
            List of TextChunk instances
        """
        chunks = self.strategy.chunk(document.text, document.metadata)

        # Enrich chunk metadata with document source
        for chunk in chunks:
            if 'source' not in chunk.metadata and hasattr(document, 'source'):
                chunk.metadata['source'] = document.source

        return chunks

    def chunk_documents(self, documents: List) -> List[TextChunk]:
        """
        Chunk multiple documents.

        Args:
            documents: List of Document instances

        Returns:
            Flat list of TextChunk instances from all documents
        """
        all_chunks = []
        for doc in documents:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)
        return all_chunks

    def set_strategy(self, strategy: ChunkingStrategy) -> None:
        """
        Change the chunking strategy.

        Args:
            strategy: New ChunkingStrategy instance
        """
        self.strategy = strategy
