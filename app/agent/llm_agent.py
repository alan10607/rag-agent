"""
VectorSearcher - LLM Agent (RAG)

Orchestrates the full RAG pipeline:
1. Retrieve relevant context from the Qdrant vector store.
2. Build a prompt combining context + user question.
3. Call the Cursor Agent CLI to generate an answer.
4. Return the answer.
"""

from __future__ import annotations

from app import config
from app.agent import cli_runner, prompt_builder
from app.logger import get_logger

logger = get_logger(__name__)


def ask(
    question: str,
    *,
    top_k: int | None = None,
    model: str | None = None,
    timeout: int | None = None,
) -> cli_runner.AgentResult:
    """Answer a question using RAG over the vector store.

    Args:
        question: The user's natural-language question.
        top_k: Number of context chunks to retrieve (default from config).
        model: Override the LLM model name.
        timeout: Override the CLI timeout in seconds.

    Returns:
        An ``AgentResult`` containing the LLM answer and metadata.
    """
    k = top_k or config.DEFAULT_TOP_K

    # ------------------------------------------------------------------
    # Step 1: Retrieve context from vector DB
    # ------------------------------------------------------------------
    logger.info("RAG retrieval: question=%r, top_k=%d", question, k)

    from app.retrieval.retriever import search as vector_search

    try:
        context_chunks = vector_search(question, top_k=k)
        logger.info("Retrieved %d context chunks", len(context_chunks))
    except Exception as exc:
        err_msg = f"Vector DB retrieval failed: {exc}"
        logger.error(err_msg)
        result = cli_runner.AgentResult(error=err_msg)
        return result

    # ------------------------------------------------------------------
    # Step 2: Build prompt
    # ------------------------------------------------------------------
    prompt = prompt_builder.build_prompt(question, context_chunks)
    logger.info(f"Prompt for Agent: %s", prompt)

    # ------------------------------------------------------------------
    # Step 3: Call Cursor Agent CLI
    # ------------------------------------------------------------------
    logger.info("Sending prompt to Cursor Agent CLI (model=%s)", model or config.AGENT_MODEL)

    result = cli_runner.run(prompt, model=model, timeout=timeout)
    result.context_chunks = context_chunks

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
        lines.append(f"  Session: {result.session_id or 'N/A'}")

        if result.context_chunks:
            sources = {c.get("payload", {}).get("source", "unknown") for c in result.context_chunks}
            lines.append(f"  Context: {len(result.context_chunks)} chunk(s) from {', '.join(sorted(sources))}")
    else:
        lines.append(f"  Error: {result.error}")
        lines.append("")
        if "Vector DB" in (result.error or ""):
            lines.append("  Hint: Make sure Qdrant is running.")
            lines.append(f"  Expected at {config.QDRANT_HOST}:{config.QDRANT_PORT}")
            lines.append("  Start: ./start.sh start  OR  docker compose up -d qdrant")
        else:
            lines.append("  Hint: Make sure Cursor CLI is installed and authenticated.")
            lines.append("  Install: curl https://cursor.com/install -fsS | bash")
            lines.append("  Auth: agent login")

    lines.append("")
    return "\n".join(lines)
