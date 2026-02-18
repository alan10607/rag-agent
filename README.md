# Ragent

A lightweight, modular RAG (Retrieval-Augmented Generation) agent built with Python and Qdrant. It bridges the gap between your local documents and LLMs through high-performance vector search.

## Features
- **Smart Ingestion**: Process Markdown, Word, and Text from `data/` with recursive chunking.
- **Multilingual Support**: Defaulted to `multilingual-e5-small` for superior Chinese/English semantic understanding.
- **Vector Power**: Powered by Qdrant for blazing-fast similarity search.
- **Agentic Workflow**: Integrated with **Cursor Agent CLI** for intelligent QA based on retrieved context.
- **Docker Ready**: One-command setup for both database and application.

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
python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt && python3 -m ragent
```


## CLI Commands
You can also run individual modules via command line.

### Main CLI
```bash
python -m ragent -h
# Available commands: config, ingest, search, agent
```

### Show Current Configuration
```bash
python -m ragent config
```

### Ingest Documents
```bash
python -m ragent ingest

# Optional: specify data folder
python -m ragent ingest --data_dir ./data
```

### Semantic Search
```bash
python -m ragent search "your query"

# Optional: limit results
python -m ragent search "your query" --top_k 5
```

### RAG + LLM Agent (Cursor CLI)
```bash
python -m ragent agent "your question"

# Optional: limit context chunks and select model
python -m ragent agent "your question" --top_k 5 --model gemini-3-flash
```

## Project Structure
```text
.
├── ragent/           # Core Logic (Python Package)
│   ├── agent         # LLM & Prompt logic
│   ├── retriever     # Vector Search logic
│   ├── indexing      # Vector Ingest logic
│   └── mcp/          # MCP Service
├── data/             # Your Documents (Mounted Volume)
└── logs/             # System Logs
```