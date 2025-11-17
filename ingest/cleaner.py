"""
Text cleaning and normalization utilities.

Provides various text cleaning strategies to normalize and prepare text
for embedding and retrieval.
"""

import re
from typing import List, Optional, Callable
from abc import ABC, abstractmethod


class TextCleaner(ABC):
    """Abstract base class for text cleaners."""

    @abstractmethod
    def clean(self, text: str) -> str:
        """
        Clean the input text.

        Args:
            text: Input text

        Returns:
            Cleaned text
        """
        pass


class WhitespaceCleaner(TextCleaner):
    """Clean excessive whitespace from text."""

    def clean(self, text: str) -> str:
        """
        Remove excessive whitespace.

        Args:
            text: Input text

        Returns:
            Text with normalized whitespace
        """
        # Replace multiple spaces with single space
        text = re.sub(r' +', ' ', text)

        # Replace multiple newlines with double newline
        text = re.sub(r'\n\s*\n+', '\n\n', text)

        # Remove leading/trailing whitespace from each line
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)

        return text.strip()


class SpecialCharacterCleaner(TextCleaner):
    """Remove or normalize special characters."""

    def __init__(self, keep_punctuation: bool = True, keep_newlines: bool = True):
        """
        Initialize special character cleaner.

        Args:
            keep_punctuation: Whether to keep punctuation marks
            keep_newlines: Whether to keep newline characters
        """
        self.keep_punctuation = keep_punctuation
        self.keep_newlines = keep_newlines

    def clean(self, text: str) -> str:
        """
        Clean special characters.

        Args:
            text: Input text

        Returns:
            Text with special characters handled
        """
        if not self.keep_punctuation:
            # Remove punctuation except spaces and newlines
            if self.keep_newlines:
                text = re.sub(r'[^\w\s\n]', '', text)
            else:
                text = re.sub(r'[^\w\s]', '', text)

        # Remove control characters except newlines and tabs
        text = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', text)

        return text


class URLCleaner(TextCleaner):
    """Remove or normalize URLs from text."""

    def __init__(self, remove: bool = False, replace_with: str = "[URL]"):
        """
        Initialize URL cleaner.

        Args:
            remove: Whether to remove URLs entirely
            replace_with: String to replace URLs with (if not removing)
        """
        self.remove = remove
        self.replace_with = replace_with

    def clean(self, text: str) -> str:
        """
        Clean URLs from text.

        Args:
            text: Input text

        Returns:
            Text with URLs handled
        """
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

        if self.remove:
            text = re.sub(url_pattern, '', text)
        else:
            text = re.sub(url_pattern, self.replace_with, text)

        return text


class EmailCleaner(TextCleaner):
    """Remove or normalize email addresses."""

    def __init__(self, remove: bool = False, replace_with: str = "[EMAIL]"):
        """
        Initialize email cleaner.

        Args:
            remove: Whether to remove emails entirely
            replace_with: String to replace emails with (if not removing)
        """
        self.remove = remove
        self.replace_with = replace_with

    def clean(self, text: str) -> str:
        """
        Clean email addresses from text.

        Args:
            text: Input text

        Returns:
            Text with emails handled
        """
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'

        if self.remove:
            text = re.sub(email_pattern, '', text)
        else:
            text = re.sub(email_pattern, self.replace_with, text)

        return text


class NumberCleaner(TextCleaner):
    """Normalize numbers in text."""

    def __init__(self, normalize: bool = True, remove: bool = False):
        """
        Initialize number cleaner.

        Args:
            normalize: Whether to normalize numbers to a standard format
            remove: Whether to remove numbers entirely
        """
        self.normalize = normalize
        self.remove = remove

    def clean(self, text: str) -> str:
        """
        Clean numbers from text.

        Args:
            text: Input text

        Returns:
            Text with numbers handled
        """
        if self.remove:
            text = re.sub(r'\b\d+\.?\d*\b', '', text)
        elif self.normalize:
            # Normalize number formatting (remove commas, etc.)
            text = re.sub(r'(\d+),(\d+)', r'\1\2', text)

        return text


class HTMLCleaner(TextCleaner):
    """Remove HTML tags and entities."""

    def clean(self, text: str) -> str:
        """
        Remove HTML tags and decode entities.

        Args:
            text: Input text

        Returns:
            Text with HTML removed
        """
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)

        # Decode common HTML entities
        html_entities = {
            '&nbsp;': ' ',
            '&amp;': '&',
            '&lt;': '<',
            '&gt;': '>',
            '&quot;': '"',
            '&#39;': "'",
            '&mdash;': '—',
            '&ndash;': '–',
        }

        for entity, char in html_entities.items():
            text = text.replace(entity, char)

        return text


class UnicodeCleaner(TextCleaner):
    """Normalize Unicode characters."""

    def __init__(self, normalize_form: str = 'NFKC'):
        """
        Initialize Unicode cleaner.

        Args:
            normalize_form: Unicode normalization form (NFC, NFKC, NFD, NFKD)
        """
        self.normalize_form = normalize_form

    def clean(self, text: str) -> str:
        """
        Normalize Unicode characters.

        Args:
            text: Input text

        Returns:
            Text with normalized Unicode
        """
        import unicodedata
        return unicodedata.normalize(self.normalize_form, text)


class CustomCleaner(TextCleaner):
    """Apply a custom cleaning function."""

    def __init__(self, func: Callable[[str], str]):
        """
        Initialize custom cleaner.

        Args:
            func: Custom cleaning function
        """
        self.func = func

    def clean(self, text: str) -> str:
        """
        Apply custom cleaning function.

        Args:
            text: Input text

        Returns:
            Cleaned text
        """
        return self.func(text)


class TextCleaningPipeline:
    """
    Pipeline for applying multiple text cleaning operations.

    This is the main class users should interact with.
    """

    def __init__(self, cleaners: Optional[List[TextCleaner]] = None):
        """
        Initialize text cleaning pipeline.

        Args:
            cleaners: List of TextCleaner instances to apply in order
        """
        self.cleaners = cleaners or self._get_default_cleaners()

    def _get_default_cleaners(self) -> List[TextCleaner]:
        """
        Get default set of cleaners.

        Returns:
            List of default TextCleaner instances
        """
        return [
            HTMLCleaner(),
            UnicodeCleaner(),
            WhitespaceCleaner(),
        ]

    def clean(self, text: str) -> str:
        """
        Apply all cleaners in the pipeline.

        Args:
            text: Input text

        Returns:
            Fully cleaned text
        """
        for cleaner in self.cleaners:
            text = cleaner.clean(text)
        return text

    def add_cleaner(self, cleaner: TextCleaner) -> None:
        """
        Add a cleaner to the pipeline.

        Args:
            cleaner: TextCleaner instance to add
        """
        self.cleaners.append(cleaner)

    def remove_cleaner(self, cleaner_class: type) -> None:
        """
        Remove all cleaners of a specific class.

        Args:
            cleaner_class: Class of cleaner to remove
        """
        self.cleaners = [c for c in self.cleaners if not isinstance(c, cleaner_class)]


def get_default_pipeline() -> TextCleaningPipeline:
    """
    Get a default text cleaning pipeline.

    Returns:
        Configured TextCleaningPipeline
    """
    cleaners = [
        HTMLCleaner(),
        UnicodeCleaner(),
        URLCleaner(remove=False),
        EmailCleaner(remove=False),
        WhitespaceCleaner(),
    ]
    return TextCleaningPipeline(cleaners)


def get_aggressive_pipeline() -> TextCleaningPipeline:
    """
    Get an aggressive text cleaning pipeline.

    Removes more content for focused text analysis.

    Returns:
        Configured TextCleaningPipeline with aggressive settings
    """
    cleaners = [
        HTMLCleaner(),
        UnicodeCleaner(),
        URLCleaner(remove=True),
        EmailCleaner(remove=True),
        SpecialCharacterCleaner(keep_punctuation=False),
        WhitespaceCleaner(),
    ]
    return TextCleaningPipeline(cleaners)
