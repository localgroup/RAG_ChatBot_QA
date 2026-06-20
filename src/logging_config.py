"""
Centralized logging configuration for the RAG Document Q&A application.

This module provides a unified logging setup with:
- File-based logging with rotation
- Console output for real-time monitoring
- Structured logging format with timestamps
- Configurable log levels and directories

Functions:
    setup_logging(): Initialize and configure the application logger.
    get_logger(): Get the configured logger instance.
"""

# ============================================================================
# Imports - Standard Library
# ============================================================================

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

# ============================================================================
# Imports - Configuration
# ============================================================================

from config import (
    LOGS_DIRECTORY,
    LOG_FILE_NAME,
    LOG_LEVEL,
    MAX_LOG_FILE_SIZE,
    LOG_BACKUP_COUNT,
    ENABLE_CONSOLE_LOGGING,
)


# ============================================================================
# Module-Level Logger Instance
# ============================================================================

# Global logger instance (initialized by setup_logging())
_logger: logging.Logger | None = None


# ============================================================================
# Logging Setup Functions
# ============================================================================


def setup_logging() -> logging.Logger:
    """
    Initialize and configure the application logger with file and console handlers.

    This function sets up a centralized logger that:
    1. Creates logs directory if it doesn't exist
    2. Adds a RotatingFileHandler for persistent logging
    3. Optionally adds a StreamHandler for console output
    4. Applies a consistent format with timestamps and log levels
    5. Sets the configured log level

    Logger Format:
        %(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s
        Example: 2026-04-14 10:30:45,123 - rag_app - INFO - build_vector_store:42 - Started building vector store

    File Handler:
        - Location: {LOGS_DIRECTORY}/rag_app.log
        - Max size: 10MB per file
        - Backup files: Keeps 5 rotated files (latest to .5)
        - Format: Detailed with function name and line number

    Console Handler (optional):
        - Enabled if ENABLE_CONSOLE_LOGGING is True
        - Output to stderr
        - Same format as file handler

    Returns:
        logging.Logger: Configured logger instance ready for use.
            Access methods: logger.debug(), logger.info(), logger.warning(),
                           logger.error(), logger.critical()

    Raises:
        OSError: If unable to create logs directory or write permission denied.

    Side Effects:
        - Creates LOGS_DIRECTORY if it doesn't exist
        - Creates or appends to log file (RotatingFileHandler)
        - Sets global _logger module variable

    Note:
        - Safe to call multiple times (idempotent due to check_logger)
        - Logger name is "rag_app" to identify messages from this application
        - Log level can be changed in config.py (LOG_LEVEL)

    Examples:
        >>> logger = setup_logging()
        >>> logger.info("Application started")
        >>> logger.error("An error occurred", exc_info=True)
        >>> # Log entries will appear in:
        >>> # - console (if ENABLE_CONSOLE_LOGGING=True)
        >>> # - logs/rag_app.log file
    """

    global _logger

    # ========================================================================
    # Create Logger Instance
    # ========================================================================
    # Use a specific logger name "rag_app" to identify logs from this application
    logger = logging.getLogger("rag_app")

    # Set the configured log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    logger.setLevel(getattr(logging, LOG_LEVEL.upper()))

    # ========================================================================
    # Create Logs Directory
    # ========================================================================
    # Ensure the logs directory exists; create if needed
    LOGS_DIRECTORY.mkdir(parents=True, exist_ok=True)

    # ========================================================================
    # Configure File Handler with Rotation
    # ========================================================================
    # RotatingFileHandler manages file rotation based on file size
    # When max size is reached, the file is rotated and backups are kept
    log_file_path = LOGS_DIRECTORY / LOG_FILE_NAME

    file_handler = RotatingFileHandler(
        filename=log_file_path,              # Where to write logs
        maxBytes=MAX_LOG_FILE_SIZE,          # Rotate when file reaches this size
        backupCount=LOG_BACKUP_COUNT,        # Keep this many backup files
    )

    # Set file handler log level (usually same as logger level)
    file_handler.setLevel(getattr(logging, LOG_LEVEL.upper()))

    # ========================================================================
    # Configure Logging Format
    # ========================================================================
    # Structured format with all relevant context for debugging
    log_format = logging.Formatter(
        fmt=(
            "%(asctime)s - %(name)s - %(levelname)s - "
            "%(funcName)s:%(lineno)d - %(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S",  # ISO format for timestamps
    )

    # Apply format to file handler
    file_handler.setFormatter(log_format)

    # Add file handler to logger
    logger.addHandler(file_handler)

    # ========================================================================
    # Optional Console Handler
    # ========================================================================
    # Optionally add a console handler for real-time monitoring
    if ENABLE_CONSOLE_LOGGING:
        # Create a StreamHandler for console output (stderr by default)
        console_handler = logging.StreamHandler()

        # Set console handler log level
        console_handler.setLevel(getattr(logging, LOG_LEVEL.upper()))

        # Apply same format to console output
        console_handler.setFormatter(log_format)

        # Add console handler to logger
        logger.addHandler(console_handler)

    # ========================================================================
    # Store Global Reference and Return
    # ========================================================================
    # Cache the logger in module-level variable for quick access
    _logger = logger

    return logger


# ============================================================================
# Logger Accessor Functions
# ============================================================================


def get_logger() -> logging.Logger:
    """
    Get the configured logger instance, initializing if needed.

    This function provides a convenient way to access the application logger
    throughout the codebase. It ensures the logger is initialized exactly once
    (idempotent), and subsequent calls return the cached instance.

    Usage Pattern:
        from src.logging_config import get_logger
        logger = get_logger()
        logger.info("Starting document processing")

    Returns:
        logging.Logger: The configured "rag_app" logger instance.
            The same instance is returned on subsequent calls.

    Note:
        - Safe to call multiple times from different modules
        - First call initializes logging; subsequent calls return cached logger
        - If setup_logging() was already called, returns that logger
        - Never raises exceptions; always returns a valid logger

    Examples:
        >>> from src.logging_config import get_logger
        >>> logger = get_logger()
        >>> logger.info("Processing started")
        >>> logger.warning("Resource usage high")
        >>> logger.error("Failed to load vector store")
    """

    # ========================================================================
    # Initialize Logger if Needed
    # ========================================================================
    # Check if logger has already been initialized
    if _logger is None:
        # First call: initialize logging
        return setup_logging()

    # ========================================================================
    # Return Cached Logger
    # ========================================================================
    # Subsequent calls: return the cached logger instance
    return _logger
