"""
Document loader for ingesting various file types.

Orchestrates extraction, cleaning, and metadata management for documents.
"""

from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import requests
from urllib.parse import urlparse

from .extractor import UniversalExtractor
from .cleaner import TextCleaningPipeline, get_default_pipeline


class Document:
    """
    Represents a loaded document with text and metadata.

    This is the core data structure passed through the RAG pipeline.
    """

    def __init__(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        source: Optional[str] = None
    ):
        """
        Initialize a document.

        Args:
            text: Document text content
            metadata: Document metadata dictionary
            source: Source path or URL
        """
        self.text = text
        self.metadata = metadata or {}
        self.source = source

        # Auto-add source to metadata if provided
        if source and 'source' not in self.metadata:
            self.metadata['source'] = source

    def __repr__(self) -> str:
        """String representation of document."""
        preview = self.text[:100] + "..." if len(self.text) > 100 else self.text
        return f"Document(source={self.source}, text_length={len(self.text)}, preview='{preview}')"

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert document to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "text": self.text,
            "metadata": self.metadata,
            "source": self.source
        }


class DocumentLoader:
    """
    Universal document loader for files and URLs.

    Handles extraction, cleaning, and document object creation.
    """

    def __init__(
        self,
        extractor: Optional[UniversalExtractor] = None,
        cleaner: Optional[TextCleaningPipeline] = None,
        clean_text: bool = True
    ):
        """
        Initialize document loader.

        Args:
            extractor: UniversalExtractor instance (creates default if None)
            cleaner: TextCleaningPipeline instance (creates default if None)
            clean_text: Whether to clean extracted text
        """
        self.extractor = extractor or UniversalExtractor()
        self.cleaner = cleaner or get_default_pipeline()
        self.clean_text = clean_text

    def load_file(self, file_path: str) -> Document:
        """
        Load a single file.

        Args:
            file_path: Path to file

        Returns:
            Document instance

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file type is not supported
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Extract text
        extracted = self.extractor.extract(str(path))

        # Clean text if enabled
        text = extracted['text']
        if self.clean_text:
            text = self.cleaner.clean(text)

        # Create document
        metadata = extracted.get('metadata', {})
        metadata['file_name'] = path.name
        metadata['file_path'] = str(path.absolute())

        # Add page information if available (for PDFs)
        if 'pages' in extracted:
            metadata['num_pages'] = len(extracted['pages'])

        return Document(
            text=text,
            metadata=metadata,
            source=str(path.absolute())
        )

    def load_directory(
        self,
        directory_path: str,
        recursive: bool = True,
        file_patterns: Optional[List[str]] = None
    ) -> List[Document]:
        """
        Load all supported files from a directory.

        Args:
            directory_path: Path to directory
            recursive: Whether to search recursively
            file_patterns: List of glob patterns to match (e.g., ['*.pdf', '*.txt'])

        Returns:
            List of Document instances

        Raises:
            NotADirectoryError: If path is not a directory
        """
        path = Path(directory_path)

        if not path.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory_path}")

        documents = []
        patterns = file_patterns or ['**/*'] if recursive else ['*']

        for pattern in patterns:
            for file_path in path.glob(pattern):
                if file_path.is_file():
                    try:
                        doc = self.load_file(str(file_path))
                        documents.append(doc)
                    except (ValueError, Exception) as e:
                        # Skip unsupported or problematic files
                        print(f"Skipping {file_path}: {e}")
                        continue

        return documents

    def load_url(self, url: str, timeout: int = 30) -> Document:
        """
        Load content from a URL.

        Args:
            url: URL to fetch
            timeout: Request timeout in seconds

        Returns:
            Document instance

        Raises:
            requests.RequestException: If request fails
        """
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()

        content_type = response.headers.get('Content-Type', '')

        # Handle HTML content
        if 'text/html' in content_type:
            text = self._extract_html_from_string(response.text)
        else:
            text = response.text

        # Clean text if enabled
        if self.clean_text:
            text = self.cleaner.clean(text)

        # Create metadata
        metadata = {
            'url': url,
            'content_type': content_type,
            'status_code': response.status_code,
            'domain': urlparse(url).netloc
        }

        return Document(
            text=text,
            metadata=metadata,
            source=url
        )

    def _extract_html_from_string(self, html_content: str) -> str:
        """
        Extract text from HTML string.

        Args:
            html_content: HTML content

        Returns:
            Extracted text
        """
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError("beautifulsoup4 is required for HTML extraction. Install with: pip install beautifulsoup4")

        soup = BeautifulSoup(html_content, 'html.parser')

        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()

        return soup.get_text()

    def load_urls(self, urls: List[str], timeout: int = 30) -> List[Document]:
        """
        Load content from multiple URLs.

        Args:
            urls: List of URLs to fetch
            timeout: Request timeout in seconds

        Returns:
            List of Document instances
        """
        documents = []
        for url in urls:
            try:
                doc = self.load_url(url, timeout)
                documents.append(doc)
            except Exception as e:
                print(f"Failed to load {url}: {e}")
                continue

        return documents

    def load_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> Document:
        """
        Load text directly (no extraction needed).

        Args:
            text: Text content
            metadata: Optional metadata dictionary

        Returns:
            Document instance
        """
        # Clean text if enabled
        if self.clean_text:
            text = self.cleaner.clean(text)

        return Document(
            text=text,
            metadata=metadata or {},
            source="direct_text"
        )

    def load_multiple(
        self,
        sources: List[Union[str, Dict[str, Any]]]
    ) -> List[Document]:
        """
        Load multiple documents from various sources.

        Args:
            sources: List of file paths, URLs, or dictionaries with 'type' and 'path'/'url'

        Returns:
            List of Document instances

        Examples:
            >>> loader.load_multiple([
            ...     'document.pdf',
            ...     'https://example.com',
            ...     {'type': 'file', 'path': 'report.docx'},
            ...     {'type': 'url', 'url': 'https://example.com/api/data'}
            ... ])
        """
        documents = []

        for source in sources:
            try:
                if isinstance(source, str):
                    # Auto-detect file vs URL
                    if source.startswith('http://') or source.startswith('https://'):
                        doc = self.load_url(source)
                    else:
                        doc = self.load_file(source)
                elif isinstance(source, dict):
                    if source.get('type') == 'file':
                        doc = self.load_file(source['path'])
                    elif source.get('type') == 'url':
                        doc = self.load_url(source['url'])
                    elif source.get('type') == 'text':
                        doc = self.load_text(source['text'], source.get('metadata'))
                    else:
                        print(f"Unknown source type: {source.get('type')}")
                        continue
                else:
                    print(f"Unsupported source format: {type(source)}")
                    continue

                documents.append(doc)

            except Exception as e:
                print(f"Failed to load source {source}: {e}")
                continue

        return documents
