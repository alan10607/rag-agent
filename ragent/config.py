"""
Ragent - Centralized Configuration

All configurable parameters are managed here.
Values are loaded from environment variables (.env) with sensible defaults.
"""

import logging
import os

from dotenv import load_dotenv

load_dotenv()

# Qdrant settings
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "documents")

# Embedding model settings
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
VECTOR_SIZE = int(os.getenv("VECTOR_SIZE", "384"))

# Search defaults
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "5"))

# Text splitter settings
# For Chinese text: 300 chars ≈ 300-450 tokens (within model's 512 token limit)
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "600"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))
CHUNK_MIN_SIZE = int(os.getenv("CHUNK_MIN_SIZE", "50"))

# Ingestion settings
# Project root is one level up from ragent/
_PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.getenv("DATA_DIR", os.path.join(_PROJECT_ROOT, "data"))
LOG_DIR = os.getenv("LOG_DIR", os.path.join(_PROJECT_ROOT, "logs"))
SUPPORTED_EXTENSIONS = [".pdf", ".txt", ".md", ".log", ".docx"]
UPSERT_BATCH_SIZE = 100

# LLM Agent settings (Cursor Agent CLI)
CURSOR_CLI_CMD = os.getenv("CURSOR_CLI_CMD", "agent")
CURSOR_API_KEY = os.getenv("CURSOR_API_KEY", "")
AGENT_MODEL = os.getenv("AGENT_MODEL", "gemini-3-flash")
AGENT_TIMEOUT_SECONDS = int(os.getenv("AGENT_TIMEOUT_SECONDS", "120"))

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_LEVEL_INT = getattr(logging, LOG_LEVEL, logging.INFO)

