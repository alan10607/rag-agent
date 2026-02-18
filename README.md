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
# Launches CLI, and shuts down containers on exit.
./start.sh

# This docker builds the app
./start.sh build

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



## Interactive CLI Demo

### Menu
The central command hub of Ragent, allowing seamless switching between data processing and intelligent retrieval.
```text
==================================================
  Ragent CLI
==================================================


? Select an option: (Use arrow keys)
 » Agent   - RAG + LLM Q&A
   Search  - Pure Semantic Search
   Ingest  - Import Documents
   Config  - Show Current Configuration
   Exit
```


---

### Agent 
RAG-Driven Intelligent Assistant.  
Experience the full power of RAG where LLMs answer based on your specific knowledge base.
- Contextual Integrity: The agent retrieves the most relevant chunks from Qdrant before generating a response.
- Zero-Hallucination Mode: Instructions are set to only answer if the information exists in your local data.
- Source Citations: Clearly shows which document IDs were used to formulate the answer.

```text
? Select an option: Agent   - RAG + LLM Q&A
? Enter question (empty to return): What is HNSW skip-list capability?
? Enter model (default: auto):

============================================================
  LLM Agent Answer
============================================================

Searching for more specific information on HNSW and skip-list.

Checking the vector search document for any HNSW technical details:

Here’s the answer from the reference material:

---

## HNSW skip-list capability

**From the reference (Ragent Technical Documentation: Vector Retrieval and Indexing):**

The **HNSW skip-list capability** is the way the **top layers** of the HNSW (Hierarchical Navigable Small World) index are used for **fast global navigation**.

In the docs, HNSW is described as a **multi-layered graph** where:

- **Top layers** give a **“skip-list” capability** for **rapid global navigation** across the graph (jumping over many nodes to get close to the right region).
- **Lower layers** are used for **local, fine-grained search** (refining the result in that region).

So “skip-list capability” here means: the upper layers act like a skip list, letting the search take long-range jumps instead of walking the full graph, which keeps **latency in the sub-millisecond range** even for large, semantic queries.

**Source:** `sample01_vector_search_deep_dive.txt`, Section 3 — “Acceleration via HNSW Indexing”.

------------------------------------------------------------
  Model: Auto
  Duration: 27.3s
  MCP Calls: 4 time(s)
  Context: 20 chunk(s) fromsample01_vector_search_deep_dive.txt, sample02_rag_and_prompt_engineering.md
```


---

### Search Demo
Semantic Vector Discovery & Scoring.  
Go beyond keywords. Find information based on meaning and intent.
- Similarity Scoring: Real-time display of Cosine Similarity scores (e.g., 0.92) for every result.
- Pure Retrieval: View the raw content stored in your vector database without LLM re-phrasing.
- Top-K Filtering: Demonstrates how HNSW indexing retrieves the best matches in milliseconds.

```text
? Enter query (empty to return): HNSW skip-list capability
? Enter number of search chunks (default: 10):

============================================================
  Found 10 result(s)
============================================================

  [1] Score: 0.8512
      Source: sample01_vector_search_deep_dive.txt
      Chunk: #2
      Preview:
        ## 3. Acceleration via HNSW Indexing
Scaling to millions of documents requires more than a simple linear scan. Qdrant implements **Hierarchical Navigable Small World (HNSW)** indexing. This algorithm constructs a multi-layered graph where top layers provide a "skip-list" capability for rapid global navigation, while lower layers allow for local, granular refinements. This architecture ensures sub-millisecond latency for complex semantic queries.
  --------------------------------------------------------
  [2] Score: 0.8034
      Source: sample01_vector_search_deep_dive.txt
      Chunk: #6
      Preview:
        ## 7. Re-ranking: The Precision Layer
The initial retrieval (Candidate Generation) focuses on high recall. Ragent then introduces a **Cross-Encoder Re-ranker**:
1. **Candidate Retrieval**: Quickly pulls the Top-100 candidates via HNSW.
2. **Deep Scoring**: The Cross-Encoder examines the *interaction* between the query and each chunk simultaneously.
3. **Filtering**: Only the Top-5 "vetted" chunks are passed to the LLM, significantly reducing noise and token costs.

...
```


---

### Ingest Demo
Dynamic Knowledge Base Construction.  
Transform your static files into a living, searchable vector space.
- Smart Chunking: Uses recursive character splitting to ensure semantic fragments remain intact.
- Vectorization Pipeline: Leverages the multilingual model to generate 384-dimensional embeddings.
```text
? Select an option: Ingest  - Import Documents
? Enter data directory (default: ./data):

Ingestion complete. Total points: 20, success: 2, failed: 0
```