"""
VectorVault - Unified Logger Configuration

Provides a centralized logging setup for the entire application.
Each pipeline module (ingest, search) gets its own log file,
while sharing a common format and console handler.

Usage:
    from app.logger import setup_logging, get_logger

    # At CLI entry point (once per process)
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

_initialized = False


def setup_logging(
    module: str = "app",
    level: int = logging.INFO,
    log_to_file: bool = True,
    log_to_console: bool = True,
) -> None:
    """Configure the global logging system with module-specific log file.

    Should be called once at application startup. Subsequent calls are no-ops.

    Args:
        module: Module name used for log filename (e.g. "ingest", "search").
                Log file will be: logs/{module}_YYYYMMDD.log
        level: The minimum log level (default: INFO).
        log_to_file: Whether to write logs to a file in logs/ directory.
        log_to_console: Whether to output logs to stderr.
    """
    global _initialized
    if _initialized:
        return
    _initialized = True

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)

    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # Module-specific file handler
    if log_to_file:
        os.makedirs(LOG_DIR, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d")
        log_file = os.path.join(LOG_DIR, f"{module}_{timestamp}.log")
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Get a named logger instance.

    Args:
        name: The logger name, typically __name__.

    Returns:
        A configured logging.Logger instance.
    """
    return logging.getLogger(name)
