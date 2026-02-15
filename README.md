# VectorSearcher

A modular local vector search engine built with Qdrant and Python, designed for scalable semantic search with clear ingestion and retrieval separation.

## Features

- Process documents from a local folder (`data/`)
- Automatic text chunking with recursive character splitting
- Embedding generation using `all-MiniLM-L6-v2` (384-dimensional vectors)
- Vector storage and search via Qdrant
- CLI-based semantic search (interactive menu + subcommands)
- Idempotent ingestion (safe to re-run)
- Docker Compose for one-command setup
- No LLM dependency — pure vector search

## Prerequisites

- Docker & Docker Compose

For local development without Docker:
- Python 3.14+

## Quick Start (Docker)

### One-command launch

```bash
./start.sh
```

This will:
1. Build the app image
2. Start Qdrant in background
3. Launch an interactive CLI menu
4. Shut down all containers when you exit

### Manual Docker commands

```bash
# Start Qdrant only (background)
docker compose up qdrant -d

# Interactive mode
docker compose run --rm vector-searcher

# Direct ingest
docker compose run --rm vector-searcher python -m app ingest

# Direct search
docker compose run --rm vector-searcher python -m app search "What is vector search?"
docker compose run --rm vector-searcher python -m app search "How do embeddings work?" --top_k 3

# Shut down everything
docker compose down
```

### Qdrant Dashboard

Once Qdrant is running, open `http://localhost:6333/dashboard` to browse collections and data.

## Quick Start (Local Development)

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

```bash
# Start Qdrant separately
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant

# Interactive mode
python -m app

# Or use subcommands directly
python -m app ingest
python -m app search "What is vector search?"

# Or run modules directly
python -m app.ingest
python -m app.search "What is vector search?" --top_k 5
```

## CLI Usage

### Interactive menu (no arguments)

```bash
python -m app
# ==================================================
#   VectorSearcher CLI
# ==================================================
#
#   1. Ingest  - Import documents from data/
#   2. Search  - Semantic search
#   3. Exit
#
#   Select [1/2/3]:
```

### Subcommands

```bash
python -m app ingest                          # Ingest documents
python -m app search "query" --top_k 3        # Search with options
```

## Project Structure

```
VectorSearcher/
├── app/
│   ├── __init__.py
│   ├── __main__.py            # Unified CLI entry point
│   ├── config.py              # Centralized configuration
│   ├── logger.py              # Unified logger (module-specific log files)
│   ├── embedding.py           # Embedding model wrapper (shared)
│   ├── qdrant_client.py       # Qdrant client wrapper (shared)
│   ├── ingest/                # Ingestion pipeline module
│   │   ├── __init__.py
│   │   ├── __main__.py        # python -m app.ingest
│   │   ├── pipeline.py
│   │   └── splitter.py        # Recursive text splitter
│   └── search/                # Search pipeline module
│       ├── __init__.py
│       ├── __main__.py        # python -m app.search
│       └── pipeline.py
├── data/                      # Document storage (mounted as volume)
│   └── sample.txt
├── logs/                      # Auto-generated log files
│   ├── ingest_YYYYMMDD.log
│   └── search_YYYYMMDD.log
├── test/
│   ├── test_splitter.py
│   └── test_embedding.py
├── Dockerfile
├── docker-compose.yml
├── start.sh                   # One-command launch script
├── requirements.txt
├── .env
├── .env.example
├── .dockerignore
├── .gitignore
└── README.md
```

## Configuration

All settings are managed in `app/config.py` and can be overridden via environment variables or `.env`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `QDRANT_HOST` | `localhost` | Qdrant server host |
| `QDRANT_PORT` | `6333` | Qdrant server port |
| `COLLECTION_NAME` | `documents` | Qdrant collection name |
| `DATA_DIR` | `./data` | Document source directory |
| `CHUNK_SIZE` | `500` | Max characters per chunk |
| `CHUNK_OVERLAP` | `50` | Overlap between chunks |
| `DEFAULT_TOP_K` | `5` | Default search results count |

## Running Tests

```bash
pip install pytest
pytest test/ -v
```

## Technology Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.14 |
| Vector DB | Qdrant (Docker) |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| PDF Parsing | PyPDF |
| Text Splitting | Custom recursive character splitter |
| Container | Docker Compose |

## Future Roadmap

- Phase 2: FastAPI REST endpoint
- Phase 3: LLM QA layer
- Phase 4: Metadata filtering
- Phase 5: Multi-collection support
- Phase 6: Cloud deployment
