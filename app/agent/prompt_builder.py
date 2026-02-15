"""
VectorSearcher - RAG Prompt Builder

Constructs prompts that combine retrieved context chunks with the user's question,
following a standard RAG (Retrieval-Augmented Generation) pattern.
"""

import re

from app.logger import get_logger

logger = get_logger(__name__)


def _detect_language(text: str) -> str:
    """Detect the dominant language of the text based on character distribution.

    Returns a human-readable language name (e.g. "Traditional Chinese",
    "English", "Japanese").
    """
    if not text:
        return "English"

    cjk = len(re.findall(r'[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]', text))
    jp_only = len(re.findall(r'[\u3040-\u309f\u30a0-\u30ff]', text))  # hiragana + katakana
    kr = len(re.findall(r'[\uac00-\ud7af\u1100-\u11ff]', text))

    total = len(text)
    if total == 0:
        return "English"

    if jp_only > 0 and (jp_only + cjk) / total > 0.1:
        return "Japanese"
    if kr / total > 0.1:
        return "Korean"
    if cjk / total > 0.1:
        return "Traditional Chinese"

    return "English"

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_SYSTEM_TEMPLATE = """\
You are a knowledgeable assistant. Answer the user's question based on the \
provided reference materials below.

Rules:
1. Prioritize information from the reference materials and cite the source.
2. If the reference materials are insufficient, clearly state which parts \
come from the references and which are your own supplementation.
3. **CRITICAL language rule**: You MUST reply in {lang_name} because the \
user's question is written in {lang_name}. Do NOT switch to any other \
language regardless of the language used in the reference materials.
4. Keep the answer concise and well-structured.
"""

_CONTEXT_TEMPLATE = """\
========== Reference Materials ==========
{context}
========== End of Reference Materials ==========
"""

_CHUNK_TEMPLATE = """\
[Source: {source}, chunk #{chunk_index}, similarity: {score:.4f}]
{text}
---
"""


def build_prompt(question: str, context_chunks: list[dict]) -> str:
    """Build a RAG prompt from the user question and retrieved context chunks.

    Args:
        question: The user's natural-language question.
        context_chunks: A list of result dicts returned by ``retriever.search()``.
            Each dict has keys: ``id``, ``score``, ``payload`` (with ``text``,
            ``source``, ``chunk_index``, etc.).

    Returns:
        A fully assembled prompt string ready to be sent to the LLM.
    """
    # Format each context chunk
    formatted_chunks: list[str] = []
    for chunk in context_chunks:
        payload = chunk.get("payload", {})
        formatted_chunks.append(
            _CHUNK_TEMPLATE.format(
                source=payload.get("source", "unknown"),
                chunk_index=payload.get("chunk_index", "?"),
                score=chunk.get("score", 0.0),
                text=payload.get("text", ""),
            )
        )

    context_block = _CONTEXT_TEMPLATE.format(
        context="\n".join(formatted_chunks) if formatted_chunks else "(No relevant reference materials found)"
    )

    lang_name = _detect_language(question)
    system_block = _SYSTEM_TEMPLATE.format(lang_name=lang_name)

    prompt = f"{system_block}\n{context_block}\nUser question: {question}\n"

    logger.info(
        "Built RAG prompt: question=%r, lang=%s, context_chunks=%d, prompt_length=%d",
        question,
        lang_name,
        len(context_chunks),
        len(prompt),
    )
    return prompt
