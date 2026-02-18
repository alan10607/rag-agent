"""
Ragent - LLM Agent (RAG)

Orchestrates the full RAG pipeline:
1. Retrieve relevant context from the Qdrant vector store.
2. Build a prompt combining context + user question.
3. Call the Cursor Agent CLI to generate an answer.
4. Return the answer.
"""

from __future__ import annotations

import argparse
from ragent import config
from ragent.agent import cli_runner, prompt_builder
from ragent.logger import get_logger, setup_logging

logger = get_logger(__name__)


def ask(
    question: str,
    *,
    model: str | None = None,
    timeout: int | None = None,
) -> cli_runner.AgentResult:
    """Answer a question using RAG over the vector store.

    Args:
        question: The user's natural-language question.
        model: Override the LLM model name.
        timeout: Override the CLI timeout in seconds.

    Returns:
        An ``AgentResult`` containing the LLM answer and metadata.
    """
    # ------------------------------------------------------------------
    # Step 1: Build prompt
    # ------------------------------------------------------------------
    prompt = prompt_builder.build_prompt(question)
    logger.info(f"Prompt for Agent:\n{'-'*30}\n{prompt}\n{'-'*30}")

    # ------------------------------------------------------------------
    # Step 2: Call Cursor Agent CLI
    # ------------------------------------------------------------------
    if not cli_runner.check_mcp_status():
        logger.error("MCP is not ready, will not be able to use the MCP tools")

    logger.info("Sending prompt to Cursor Agent CLI (model=%s)", model or config.AGENT_MODEL)
    result = cli_runner.run(prompt, model=model, timeout=timeout)

    if result.success:
        logger.info(
            "Agent answered in %dms (answer_length=%d)",
            result.duration_ms,
            len(result.answer_text),
        )
    else:
        logger.error("Agent failed: %s", result.error)

    return result


def format_answer(result: cli_runner.AgentResult) -> str:
    """Format an ``AgentResult`` for CLI display.

    Args:
        result: The result from ``ask()``.

    Returns:
        A human-readable string.
    """
    lines: list[str] = []
    lines.append("")
    lines.append("=" * 60)
    lines.append("  LLM Agent Answer")
    lines.append("=" * 60)
    lines.append("")

    if result.success:
        lines.append(result.answer_text.strip())
        lines.append("")
        lines.append("-" * 60)

        # Duration in human-readable m/s format
        total_sec = result.duration_ms / 1000
        minutes, seconds = divmod(total_sec, 60)
        if minutes >= 1:
            duration_str = f"{int(minutes)}m {seconds:.1f}s"
        else:
            duration_str = f"{seconds:.1f}s"

        lines.append(f"  Model: {result.model or 'unknown'}")
        lines.append(f"  Duration: {duration_str}")

        if result.mcp_results:
            lines.append(f"  MCP Calls: {len(result.mcp_results)} time(s)")

        if result.vector_search_results:
            sources = {r.source for r in result.vector_search_results}
            lines.append(f"  Context: {len(result.vector_search_results)} chunk(s) from{', '.join(sorted(sources))}")


    else:
        lines.append(f"  Error: {result.error}")
        lines.append("")
        error = result.error or ""
        if "Vector DB" in error:
            lines.append("  Hint: Make sure Qdrant is running.")
            lines.append(f"  Expected at {config.QDRANT_HOST}:{config.QDRANT_PORT}")
            lines.append("  Start: docker compose up -d qdrant")
        elif "not authenticated" in error.lower():
            lines.append("  Hint: Cursor CLI is not authenticated.")
            lines.append("  Option 1: agent login")
            lines.append("  Option 2: set CURSOR_API_KEY in .env")
        else:
            lines.append("  Hint: Make sure Cursor CLI is installed and authenticated.")
            lines.append("  Install: curl https://cursor.com/install -fsS | bash")
            lines.append("  Auth: agent login  OR  set CURSOR_API_KEY in .env")

    lines.append("")
    return "\n".join(lines)

def main() -> None:
    """CLI entry point for LLM Agent."""
    parser = argparse.ArgumentParser(
        description="Ragent LLM Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "question",
        type=str,
        help="The text question to answer.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=config.AGENT_MODEL,
        help=f"The model to use (default: {config.AGENT_MODEL}).",
    )

    args = parser.parse_args()

    setup_logging(module="agent")

    results = ask(args.query, model=args.model)
    print(format_answer(results))


if __name__ == "__main__":
    main()