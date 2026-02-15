"""
VectorSearcher - Unified Logger Configuration

Provides a centralized logging setup for the entire application.
Each pipeline module (ingest, search) gets its own log file.

Usage:
    from app.logger import setup_logging, get_logger

    # At CLI entry point
    setup_logging(module="ingest")

    # In any module
    logger = get_logger(__name__)
"""

import logging
import os
import sys
from datetime import datetime, timezone

# Default log format
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Log directory (relative to project root)
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")

_console_initialized = False
_current_file_handler: logging.FileHandler | None = None


def setup_logging(
    module: str | None = None,
    level: int | None = None,
    log_to_file: bool = True,
    log_to_console: bool = True,
) -> None:
    """Configure the global logging system with module-specific log file.

    Can be called multiple times:
    - Console handler is set up once on the first call.
    - File handler is swapped each time a different module is specified.
    - Pass module=None or log_to_file=False for console-only logging.

    Args:
        module: Module name for log filename (e.g. "ingest", "search").
                Log file: logs/{module}_YYYYMMDD.log
                None = no file logging.
        level: The minimum log level (default: from config.LOG_LEVEL_INT).
        log_to_file: Whether to write logs to a file.
        log_to_console: Whether to output logs to stderr.
    """
    from app import config

    global _console_initialized, _current_file_handler

    effective_level = level if level is not None else config.LOG_LEVEL_INT

    root_logger = logging.getLogger()
    root_logger.setLevel(effective_level)

    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)

    # Console handler (set up once)
    if not _console_initialized and log_to_console:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(effective_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
        _console_initialized = True

    # File handler (swap per module)
    if _current_file_handler is not None:
        root_logger.removeHandler(_current_file_handler)
        _current_file_handler.close()
        _current_file_handler = None

    if log_to_file and module:
        os.makedirs(LOG_DIR, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d")
        log_file = os.path.join(LOG_DIR, f"{module}_{timestamp}.log")
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(effective_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        _current_file_handler = file_handler


def get_logger(name: str) -> logging.Logger:
    """Get a named logger instance.

    Args:
        name: The logger name, typically __name__.

    Returns:
        A configured logging.Logger instance.
    """
    return logging.getLogger(name)
