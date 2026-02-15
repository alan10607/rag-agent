# VectorVault

A modular local vector search engine built with Qdrant and Python, designed for scalable semantic search with clear ingestion and retrieval separation.

## Features

- Process documents from a local folder (`data/`)
- Automatic text chunking with recursive character splitting
- Embedding generation using `all-MiniLM-L6-v2` (384-dimensional vectors)
- Vector storage and search via Qdrant
- CLI-based semantic search
- Idempotent ingestion (safe to re-run)
- No LLM dependency — pure vector search

## Prerequisites

- Python 3.11+
- Docker (for Qdrant)

## Quick Start

### 1. Start Qdrant

```bash
docker run -d --name qdrant \
  -p 6333:6333 \
  -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  qdrant/qdrant
```

### 2. Set Up Python Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Add Documents

Place `.pdf` or `.txt` files into the `data/` directory. A sample file is included.

### 4. Run Ingestion

```bash
python -m app.ingest
```

This will scan `data/`, extract text, split into chunks, generate embeddings, and store them in Qdrant.

### 5. Search

```bash
python -m app.search "What is vector search?"
python -m app.search "How do embeddings work?" --top_k 3
```

## Project Structure

```
vectorvault/
├── app/
│   ├── __init__.py
│   ├── logger.py              # Unified logger (module-specific log files)
│   ├── embedding.py           # Embedding model wrapper (shared)
│   ├── splitter.py            # Recursive text splitter (shared)
│   ├── qdrant_client.py       # Qdrant client wrapper (shared)
│   ├── ingest/                # Ingestion pipeline module
│   │   ├── __init__.py
│   │   ├── __main__.py        # CLI entry: python -m app.ingest
│   │   └── pipeline.py        # Ingestion logic
│   └── search/                # Search pipeline module
│       ├── __init__.py
│       ├── __main__.py        # CLI entry: python -m app.search
│       └── pipeline.py        # Search logic
├── data/                      # Document storage
│   └── sample.txt
├── logs/                      # Auto-generated log files
│   ├── ingest_YYYYMMDD.log
│   └── search_YYYYMMDD.log
├── tests/
│   ├── test_splitter.py
│   └── test_embedding.py
├── config.py
├── requirements.txt
├── .env
├── .env.example
└── README.md
```

## Configuration

All settings are managed in `config.py` and can be overridden via `.env`:

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
pytest tests/ -v
```

Note: Embedding tests require the `all-MiniLM-L6-v2` model to be downloaded (happens automatically on first run).

## Technology Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.11+ |
| Vector DB | Qdrant (Docker) |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| PDF Parsing | PyPDF |
| Text Splitting | Custom recursive character splitter |

## Future Roadmap

- Phase 2: FastAPI REST endpoint
- Phase 3: LLM QA layer
- Phase 4: Metadata filtering
- Phase 5: Multi-collection support
- Phase 6: Cloud deployment
