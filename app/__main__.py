"""
VectorSearcher - Unified CLI Entry Point

Supports two modes:
  1. Subcommand mode:  python -m app ingest
                       python -m app search "query" --top_k 3
                       python -m app agent "question" --model gemini-3-flash
  2. Interactive mode:  python -m app  (no arguments -> menu)
"""

import argparse
import questionary
import sys

from app import config
from app.logger import setup_logging


def _show_config() -> None:
    """Display current configuration from environment variables."""
    print()
    print("=" * 50)
    print("  VectorSearcher - Current Configuration")
    print("=" * 50)
    print()
    print(f"  {'LOG_LEVEL':<25} {config.LOG_LEVEL}")
    print()
    print(f"  {'QDRANT_HOST':<25} {config.QDRANT_HOST}")
    print(f"  {'QDRANT_PORT':<25} {config.QDRANT_PORT}")
    print(f"  {'COLLECTION_NAME':<25} {config.COLLECTION_NAME}")
    print()
    print(f"  {'EMBEDDING_MODEL_NAME':<25} {config.EMBEDDING_MODEL_NAME}")
    print(f"  {'VECTOR_SIZE':<25} {config.VECTOR_SIZE}")
    print()
    print(f"  {'DEFAULT_TOP_K':<25} {config.DEFAULT_TOP_K}")
    print(f"  {'CHUNK_SIZE':<25} {config.CHUNK_SIZE}")
    print(f"  {'CHUNK_OVERLAP':<25} {config.CHUNK_OVERLAP}")
    print(f"  {'CHUNK_MIN_SIZE':<25} {config.CHUNK_MIN_SIZE}")
    print()
    print(f"  {'DATA_DIR':<25} {config.DATA_DIR}")
    print(f"  {'SUPPORTED_EXTENSIONS':<25} {', '.join(config.SUPPORTED_EXTENSIONS)}")
    print(f"  {'UPSERT_BATCH_SIZE':<25} {config.UPSERT_BATCH_SIZE}")
    print()
    print(f"  {'CURSOR_CLI_CMD':<25} {config.CURSOR_CLI_CMD}")
    print(f"  {'CURSOR_API_KEY':<25} {'set' if config.CURSOR_API_KEY else '(not set)'}")
    print(f"  {'AGENT_MODEL':<25} {config.AGENT_MODEL}")
    print(f"  {'AGENT_TIMEOUT_SECONDS':<25} {config.AGENT_TIMEOUT_SECONDS}s")
    print()


def _run_ingest(data_dir: str | None = None) -> None:
    """Execute the ingestion pipeline from a given data directory."""
    setup_logging(module="ingest")
    from app.indexing.indexer import ingest

    total, success_count, failed_count = ingest(data_dir=data_dir)
    print(f"\nIngestion complete. Total points: {total}, success: {success_count}, failed: {failed_count}")


def _run_search(query: str, top_k: int | None = None) -> None:
    """Execute a semantic search query."""
    setup_logging(module="search")
    from app.retrieval.retriever import search, format_results

    results = search(query, top_k=top_k)
    print(format_results(results))


def _run_agent(query: str, *, top_k: int | None = None, model: str | None = None) -> None:
    """Execute a RAG-based LLM agent query."""
    setup_logging(module="agent")
    from app.agent.llm_agent import ask, format_answer

    print(f"\n  Retrieving relevant context from vector DB ...")
    print(f"  Calling LLM ({model or config.AGENT_MODEL}) to generate answer ...\n")

    result = ask(question=query, top_k=top_k, model=model)
    print(format_answer(result))


def run_ingest_interactive():
    data_dir = questionary.text(
        f"Enter data directory (default: {config.DATA_DIR}):"
    ).ask()
    _run_ingest(data_dir=data_dir)


def run_search_interactive():
    while True:
        query = questionary.text("Enter query (empty to return):").ask()
        if not query:
            print("\nReturning to main menu...")
            break

        input_top_k = questionary.text(
            f"Enter number of search chunks (default: {config.DEFAULT_TOP_K}):"
        ).ask()
        try:
            top_k = int(input_top_k) if input_top_k else config.DEFAULT_TOP_K
        except ValueError:
            print("Invalid input, using default value.")
            top_k = config.DEFAULT_TOP_K

        _run_search(query, top_k=top_k)


def run_agent_interactive():
    while True:
        question = questionary.text("Enter question (empty to return):").ask()
        if not question:
            print("\nReturning to main menu...")
            break

        model = questionary.text(f"Enter model (default: {config.AGENT_MODEL}):").ask()
        _run_agent(question, model=model)


def _interactive_menu() -> None:
    print("\n" + "=" * 50)
    print("  VectorSearcher CLI")
    print("=" * 50)

    while True:
        print("\n")
        try:
            choice = questionary.select(
                "Select an option:",
                choices=[
                    "Agent   - RAG + LLM Q&A",
                    "Search  - Semantic search",
                    "Ingest  - Import documents",
                    "Config  - Show current configuration",
                    "Exit",
                ],
            ).ask()

            if choice is None or choice == "Exit":
                print("\nBye!")
                break
            elif choice.startswith("Ingest"):
                run_ingest_interactive()
            elif choice.startswith("Search"):
                run_search_interactive()
            elif choice.startswith("Agent"):
                run_agent_interactive()
            elif choice.startswith("Config"):
                _show_config()

        except KeyboardInterrupt:
            print("\nBye!")
            break
        

def main() -> None:
    """Main entry point: subcommand dispatch or interactive menu."""
    # If no subcommand arguments, enter interactive mode
    if len(sys.argv) == 1:
        _interactive_menu()
        return

    parser = argparse.ArgumentParser(
        prog="python -m app",
        description="VectorSearcher - Modular Vector Search Engine",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Config subcommand
    subparsers.add_parser("config", help="Show current configuration")

    # Ingest subcommand
    ingest_parser = subparsers.add_parser("ingest", help="Import documents into Qdrant")
    ingest_parser.add_argument(
        "--data_dir", type=str, default=config.DATA_DIR, help=f"The path to the data directory. Default from config if not provided ({config.DATA_DIR}).",
    )


    # Search subcommand
    search_parser = subparsers.add_parser("search", help="Semantic search against Qdrant")
    search_parser.add_argument("query", type=str, help="The text query to search for")
    search_parser.add_argument(
        "--top_k", type=int, default=config.DEFAULT_TOP_K, help=f"Number of results (default: {config.DEFAULT_TOP_K})"
    )

    # Agent subcommand
    agent_parser = subparsers.add_parser("agent", help="RAG + LLM agent (Cursor CLI)")
    agent_parser.add_argument("query", type=str, help="The question to ask the agent")
    agent_parser.add_argument(
        "--top_k", type=int, default=config.DEFAULT_TOP_K, help=f"Number of context chunks (default: {config.DEFAULT_TOP_K})"
    )
    agent_parser.add_argument(
        "--model", type=str, default=None, help=f"LLM model name (default: {config.AGENT_MODEL})"
    )

    args = parser.parse_args()

    if args.command == "config":
        _show_config()

    elif args.command == "ingest":
        _run_ingest(data_dir=args.data_dir)

    elif args.command == "search":
        _run_search(args.query, top_k=args.top_k)

    elif args.command == "agent":
        _run_agent(args.query, top_k=args.top_k, model=args.model)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
