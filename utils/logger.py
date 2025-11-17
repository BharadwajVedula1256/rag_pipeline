"""
Logging utilities for the RAG pipeline.

Provides structured logging with different levels, file and console output,
and context-aware logging for debugging and monitoring.
"""

import logging
import sys
from typing import Optional
from pathlib import Path
from datetime import datetime


class RAGLogger:
    """
    Custom logger for the RAG pipeline with structured output.

    Supports both console and file logging with configurable levels.
    """

    def __init__(
        self,
        name: str = "RAGPipeline",
        level: int = logging.INFO,
        log_file: Optional[str] = None,
        log_to_console: bool = True
    ):
        """
        Initialize the RAG logger.

        Args:
            name: Logger name
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file: Path to log file (optional)
            log_to_console: Whether to log to console
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.handlers.clear()  # Clear existing handlers

        # Create formatter
        formatter = logging.Formatter(
            fmt='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Console handler
        if log_to_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

        # File handler
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def debug(self, message: str, **kwargs) -> None:
        """Log debug message."""
        self.logger.debug(self._format_message(message, kwargs))

    def info(self, message: str, **kwargs) -> None:
        """Log info message."""
        self.logger.info(self._format_message(message, kwargs))

    def warning(self, message: str, **kwargs) -> None:
        """Log warning message."""
        self.logger.warning(self._format_message(message, kwargs))

    def error(self, message: str, **kwargs) -> None:
        """Log error message."""
        self.logger.error(self._format_message(message, kwargs))

    def critical(self, message: str, **kwargs) -> None:
        """Log critical message."""
        self.logger.critical(self._format_message(message, kwargs))

    def exception(self, message: str, **kwargs) -> None:
        """Log exception with traceback."""
        self.logger.exception(self._format_message(message, kwargs))

    def _format_message(self, message: str, context: dict) -> str:
        """
        Format log message with context.

        Args:
            message: Base log message
            context: Additional context dictionary

        Returns:
            Formatted message string
        """
        if context:
            context_str = " | ".join(f"{k}={v}" for k, v in context.items())
            return f"{message} | {context_str}"
        return message


def setup_logger(
    name: str = "RAGPipeline",
    level: str = "INFO",
    log_file: Optional[str] = None,
    log_to_console: bool = True
) -> RAGLogger:
    """
    Setup and return a configured logger.

    Args:
        name: Logger name
        level: Logging level as string (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        log_to_console: Whether to log to console

    Returns:
        Configured RAGLogger instance
    """
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }

    log_level = level_map.get(level.upper(), logging.INFO)

    return RAGLogger(
        name=name,
        level=log_level,
        log_file=log_file,
        log_to_console=log_to_console
    )


def get_logger(name: str = "RAGPipeline") -> RAGLogger:
    """
    Get or create a logger instance.

    Args:
        name: Logger name

    Returns:
        RAGLogger instance
    """
    return setup_logger(name=name)
