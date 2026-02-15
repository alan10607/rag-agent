"""
VectorVault - Centralized Configuration

All configurable parameters are managed here.
Values are loaded from environment variables (.env) with sensible defaults.
"""

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
DEFAULT_TOP_K = 5

# Text splitter settings
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Ingestion settings
# Project root is one level up from app/
_PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.getenv("DATA_DIR", os.path.join(_PROJECT_ROOT, "data"))
SUPPORTED_EXTENSIONS = [".pdf", ".txt"]
UPSERT_BATCH_SIZE = 100

