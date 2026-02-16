# VectorSearcher

A lightweight, modular vector search engine built with Python and Qdrant, designed for fast semantic search with clear separation of ingestion and retrieval.

## Features

- Ingest documents from a local folder (`data/`)  
- Supports **Markdown (.md)**, **Word (.docx)**, and plain text  
- Automatic text chunking with recursive splitting and sliding window  
- Embeddings with `intfloat/multilingual-e5-small` (384-dimensional) or others  
- Vector storage and search using Qdrant  
- CLI for interactive or direct subcommand search  
- Docker Compose one-command setup  
- LLM integration(by cursor agent cli) for advanced QA  

## Prerequisites

- Docker & Docker Compose  
- Python 3.14+ (for local development without Docker)

## Quick Start

### By `start.sh` script

```bash
# This builds the app
./start.sh build

# Launches CLI, and shuts down containers on exit.
./start.sh

# Remove containers and volumes
./start.sh down
```

### Manual commands
```bash
# Start Qdrant
docker compose up qdrant -d

# Local development: create venv, install dependencies, and launch CLI
python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt && python3 -m app
```


## CLI Commands
You can also run individual modules via command line.

### Main CLI
```bash
python -m app -h
# Available commands: config, ingest, search, agent
```

### Show Current Configuration
```bash
python -m app config
```

### Ingest Documents
```bash
python -m app ingest

# Optional: specify data folder
python -m app ingest --data_dir ./data
```

### Semantic Search
```bash
python -m app search "your query"

# Optional: limit results
python -m app search "your query" --top_k 5
```

### RAG + LLM Agent (Cursor CLI)
```bash
python -m app agent "your question"

# Optional: limit context chunks and select model
python -m app agent "your question" --top_k 5 --model gemini-3-flash
```


## Environments
Copy this file to .env and adjust values as needed:
```
# Qdrant vector database connection
QDRANT_HOST=localhost
QDRANT_PORT=6333
COLLECTION_NAME=documents

# Embedding model settings
# English only: all-MiniLM-L6-v2
# Multilingual (Chinese, etc.): intfloat/multilingual-e5-small
EMBEDDING_MODEL_NAME=intfloat/multilingual-e5-small
VECTOR_SIZE=384

# Text splitter settings
CHUNK_SIZE=600
CHUNK_OVERLAP=120
CHUNK_MIN_SIZE=50

# Search defaults
DEFAULT_TOP_K=5

# Data directory (relative or absolute path)
DATA_DIR=./data

# LLM Agent settings (Cursor Agent CLI)
CURSOR_CLI_CMD=agent
CURSOR_API_KEY=your_cursor_api_key_here
AGENT_MODEL=gemini-3-flash
AGENT_TIMEOUT_SECONDS=120

# Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
LOG_LEVEL=INFO
```