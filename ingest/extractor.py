"""
Text extraction from various document formats.

Supports PDFs, HTML, DOCX, and other common formats with a unified interface.
"""

from typing import Dict, List, Optional, Any
from pathlib import Path
import re
from abc import ABC, abstractmethod


class TextExtractor(ABC):
    """Abstract base class for text extractors."""

    @abstractmethod
    def extract(self, file_path: str) -> Dict[str, Any]:
        """
        Extract text from a file.

        Args:
            file_path: Path to the file

        Returns:
            Dictionary containing extracted text and metadata
        """
        pass


class PDFExtractor(TextExtractor):
    """Extract text from PDF files."""

    def __init__(self, extract_images: bool = False, extract_tables: bool = True):
        """
        Initialize PDF extractor.

        Args:
            extract_images: Whether to extract text from images using OCR
            extract_tables: Whether to extract tables
        """
        self.extract_images = extract_images
        self.extract_tables = extract_tables

    def extract(self, file_path: str) -> Dict[str, Any]:
        """
        Extract text from PDF file.

        Args:
            file_path: Path to PDF file

        Returns:
            Dictionary with text, metadata, and page information
        """
        try:
            import pypdf
        except ImportError:
            raise ImportError("pypdf is required for PDF extraction. Install with: pip install pypdf")

        text_content = []
        metadata = {}

        with open(file_path, 'rb') as f:
            reader = pypdf.PdfReader(f)
            metadata = {
                "num_pages": len(reader.pages),
                "author": reader.metadata.get('/Author', '') if reader.metadata else '',
                "title": reader.metadata.get('/Title', '') if reader.metadata else '',
                "subject": reader.metadata.get('/Subject', '') if reader.metadata else '',
            }

            for page_num, page in enumerate(reader.pages, 1):
                page_text = page.extract_text()
                text_content.append({
                    "page": page_num,
                    "text": page_text
                })

        return {
            "text": "\n\n".join([p["text"] for p in text_content]),
            "pages": text_content,
            "metadata": metadata,
            "source": file_path
        }


class TextFileExtractor(TextExtractor):
    """Extract text from plain text files."""

    def __init__(self, encoding: str = 'utf-8'):
        """
        Initialize text file extractor.

        Args:
            encoding: File encoding (default: utf-8)
        """
        self.encoding = encoding

    def extract(self, file_path: str) -> Dict[str, Any]:
        """
        Extract text from text file.

        Args:
            file_path: Path to text file

        Returns:
            Dictionary with text and metadata
        """
        with open(file_path, 'r', encoding=self.encoding) as f:
            text = f.read()

        return {
            "text": text,
            "metadata": {
                "encoding": self.encoding,
                "size": len(text)
            },
            "source": file_path
        }


class HTMLExtractor(TextExtractor):
    """Extract text from HTML files."""

    def extract(self, file_path: str) -> Dict[str, Any]:
        """
        Extract text from HTML file.

        Args:
            file_path: Path to HTML file

        Returns:
            Dictionary with extracted text and metadata
        """
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError("beautifulsoup4 is required for HTML extraction. Install with: pip install beautifulsoup4")

        with open(file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()

        soup = BeautifulSoup(html_content, 'html.parser')

        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()

        # Get text
        text = soup.get_text()

        # Extract metadata
        metadata = {
            "title": soup.title.string if soup.title else "",
            "meta_description": ""
        }

        meta_desc = soup.find("meta", attrs={"name": "description"})
        if meta_desc and meta_desc.get("content"):
            metadata["meta_description"] = meta_desc["content"]

        return {
            "text": text,
            "metadata": metadata,
            "source": file_path
        }


class MarkdownExtractor(TextExtractor):
    """Extract text from Markdown files."""

    def extract(self, file_path: str) -> Dict[str, Any]:
        """
        Extract text from Markdown file.

        Args:
            file_path: Path to Markdown file

        Returns:
            Dictionary with text and metadata
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        # Extract title from first heading
        title_match = re.search(r'^#\s+(.+)$', text, re.MULTILINE)
        title = title_match.group(1) if title_match else ""

        return {
            "text": text,
            "metadata": {
                "title": title,
                "format": "markdown"
            },
            "source": file_path
        }


class DOCXExtractor(TextExtractor):
    """Extract text from DOCX files."""

    def extract(self, file_path: str) -> Dict[str, Any]:
        """
        Extract text from DOCX file.

        Args:
            file_path: Path to DOCX file

        Returns:
            Dictionary with text and metadata
        """
        try:
            import docx
        except ImportError:
            raise ImportError("python-docx is required for DOCX extraction. Install with: pip install python-docx")

        doc = docx.Document(file_path)

        # Extract paragraphs
        paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
        text = "\n\n".join(paragraphs)

        # Extract metadata
        core_properties = doc.core_properties
        metadata = {
            "title": core_properties.title or "",
            "author": core_properties.author or "",
            "subject": core_properties.subject or "",
            "num_paragraphs": len(paragraphs)
        }

        return {
            "text": text,
            "metadata": metadata,
            "source": file_path
        }


class CSVExtractor(TextExtractor):
    """Extract text from CSV files."""

    def extract(self, file_path: str) -> Dict[str, Any]:
        """
        Extract text from CSV file.

        Args:
            file_path: Path to CSV file

        Returns:
            Dictionary with text and metadata
        """
        import csv

        rows = []
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
            for row in reader:
                rows.append(row)

        # Convert to text format
        text_lines = []
        for row in rows:
            line = " | ".join([f"{k}: {v}" for k, v in row.items()])
            text_lines.append(line)

        text = "\n".join(text_lines)

        return {
            "text": text,
            "metadata": {
                "num_rows": len(rows),
                "columns": list(headers) if headers else [],
                "format": "csv"
            },
            "source": file_path
        }


class JSONExtractor(TextExtractor):
    """Extract text from JSON files."""

    def extract(self, file_path: str) -> Dict[str, Any]:
        """
        Extract text from JSON file.

        Args:
            file_path: Path to JSON file

        Returns:
            Dictionary with text and metadata
        """
        import json

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Convert JSON to readable text
        text = json.dumps(data, indent=2)

        return {
            "text": text,
            "metadata": {
                "format": "json",
                "keys": list(data.keys()) if isinstance(data, dict) else []
            },
            "source": file_path
        }


class UniversalExtractor:
    """
    Universal text extractor that routes to specific extractors based on file type.

    This is the main class users should interact with.
    """

    def __init__(
        self,
        extract_images: bool = False,
        extract_tables: bool = True
    ):
        """
        Initialize universal extractor.

        Args:
            extract_images: Whether to extract text from images
            extract_tables: Whether to extract tables
        """
        self.extractors = {
            '.pdf': PDFExtractor(extract_images, extract_tables),
            '.txt': TextFileExtractor(),
            '.md': MarkdownExtractor(),
            '.markdown': MarkdownExtractor(),
            '.html': HTMLExtractor(),
            '.htm': HTMLExtractor(),
            '.docx': DOCXExtractor(),
            '.csv': CSVExtractor(),
            '.json': JSONExtractor(),
        }

    def extract(self, file_path: str) -> Dict[str, Any]:
        """
        Extract text from file based on extension.

        Args:
            file_path: Path to file

        Returns:
            Dictionary with extracted text and metadata

        Raises:
            ValueError: If file type is not supported
        """
        path = Path(file_path)
        extension = path.suffix.lower()

        if extension not in self.extractors:
            raise ValueError(
                f"Unsupported file type: {extension}. "
                f"Supported types: {list(self.extractors.keys())}"
            )

        return self.extractors[extension].extract(file_path)

    def add_custom_extractor(self, extension: str, extractor: TextExtractor) -> None:
        """
        Add a custom extractor for a specific file type.

        Args:
            extension: File extension (e.g., '.xml')
            extractor: TextExtractor instance
        """
        self.extractors[extension.lower()] = extractor
