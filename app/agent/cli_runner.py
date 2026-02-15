"""
VectorSearcher - Cursor Agent CLI Runner

Manages the subprocess lifecycle for calling ``cursor agent`` in headless
(print) mode with ``--output-format stream-json``.

Responsibilities:
- Spawn the Cursor CLI subprocess, piping the prompt via stdin.
- Stream and parse NDJSON output line by line.
- Write **every** raw line to a log file so nothing is lost.
- Extract and accumulate assistant text from the event stream.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import IO

from app import config
from app.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class AgentResult:
    """Holds the aggregated result returned by Cursor Agent CLI."""

    answer_text: str = ""
    thinking_text: str = ""
    context_chunks: list[dict] = field(default_factory=list)
    raw_events: list[dict] = field(default_factory=list)
    duration_ms: int = 0
    model: str = ""
    session_id: str = ""
    success: bool = False
    error: str | None = None


# ---------------------------------------------------------------------------
# Log-file helpers
# ---------------------------------------------------------------------------

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
_AGENT_LOG_DIR = os.path.join(_PROJECT_ROOT, "logs", "agent")


def _open_raw_log(model_name: str) -> IO[str]:
    """Open a timestamped raw-output log file under ``logs/agent/``.

    Filename format: ``{model_name}_{YYYYMMDD}_{HHMMSS}.log``
    """
    os.makedirs(_AGENT_LOG_DIR, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    # Sanitise model name for use in filenames
    safe_model = model_name.replace("/", "_").replace(" ", "_")
    path = os.path.join(_AGENT_LOG_DIR, f"{safe_model}_{ts}.log")
    logger.info("Raw CLI output will be logged to %s", path)
    return open(path, "w", encoding="utf-8")  # noqa: SIM115


# ---------------------------------------------------------------------------
# NDJSON event parsing
# ---------------------------------------------------------------------------

def _parse_event(line: str) -> dict | None:
    """Try to parse a single NDJSON line; return *None* on failure."""
    line = line.strip()
    if not line:
        return None
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        logger.warning("Failed to parse NDJSON line: %s", line[:200])
        return None


def _extract_text_from_event(event: dict) -> str:
    """Extract assistant text content from a parsed event dict.

    Works for both full-message events and partial-delta events
    (``--stream-partial-output``).
    """
    if event.get("type") != "assistant":
        return ""

    message = event.get("message", {})
    content_list = message.get("content", [])
    parts: list[str] = []
    for item in content_list:
        if isinstance(item, dict) and item.get("type") == "text":
            parts.append(item.get("text", ""))
        elif isinstance(item, str):
            parts.append(item)
    return "".join(parts)


def _extract_thinking_from_event(event: dict) -> str:
    """Extract thinking / reasoning content from an assistant event.

    Some models emit a ``thinking`` content block alongside ``text``.
    """
    if event.get("type") != "assistant":
        return ""

    message = event.get("message", {})
    content_list = message.get("content", [])
    parts: list[str] = []
    for item in content_list:
        if isinstance(item, dict) and item.get("type") == "thinking":
            parts.append(item.get("thinking", "") or item.get("text", ""))
    return "".join(parts)


# ---------------------------------------------------------------------------
# CLI binary resolution
# ---------------------------------------------------------------------------

# Common install locations for the Cursor Agent CLI (beyond system PATH).
_EXTRA_SEARCH_PATHS: list[str] = [
    os.path.expanduser("~/.local/bin"),
    os.path.expanduser("~/.cursor/bin"),
    "/usr/local/bin",
]


def _resolve_cli_path(cmd_name: str) -> str:
    """Resolve the full path to the Cursor Agent CLI binary.

    Strategy:
    1. If ``cmd_name`` is already an absolute path and exists, use it directly.
    2. Try ``shutil.which()`` with the default system PATH.
    3. Try ``shutil.which()`` with extra search paths appended.
    4. Fall back to the original ``cmd_name`` (will raise FileNotFoundError
       in Popen if it truly doesn't exist).
    """
    # Already an absolute path
    if os.path.isabs(cmd_name):
        if os.path.isfile(cmd_name) and os.access(cmd_name, os.X_OK):
            return cmd_name
        return cmd_name  # let Popen raise FileNotFoundError

    # Try default PATH first
    found = shutil.which(cmd_name)
    if found:
        return found

    # Try with extra search paths
    extra_path = os.pathsep.join(_EXTRA_SEARCH_PATHS)
    system_path = os.environ.get("PATH", "")
    expanded_path = f"{extra_path}{os.pathsep}{system_path}"
    found = shutil.which(cmd_name, path=expanded_path)
    if found:
        logger.info("Resolved CLI binary via extra search paths: %s", found)
        return found

    # Last resort: return as-is
    return cmd_name


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_cli_command(model: str | None = None) -> list[str]:
    """Build the ``cursor agent`` command list.

    The command uses ``-`` (read from stdin) so we can pipe the prompt.
    """
    cmd_name = _resolve_cli_path(config.CURSOR_CLI_CMD)
    model_name = model or config.AGENT_MODEL

    cmd = [
        cmd_name,
        "-p",               # print (headless) mode
        "-f",               # force allow commands
        "--approve-mcps",   # auto-approve MCP calls
        "--output-format", "stream-json",
        "--stream-partial-output",
        "--model", model_name,
        "-",                # read prompt from stdin
    ]
    return cmd


def run(prompt: str, *, model: str | None = None, timeout: int | None = None) -> AgentResult:
    """Execute the Cursor Agent CLI with the given prompt and return the result.

    1. Spawns the CLI subprocess.
    2. Writes the prompt to stdin and closes it.
    3. Reads stdout line-by-line (NDJSON), logging every raw line.
    4. Accumulates assistant text into ``AgentResult.answer_text``.

    Args:
        prompt: The full prompt string (already assembled by ``prompt_builder``).
        model: Override the default model name.
        timeout: Maximum seconds to wait (``None`` = use config default).

    Returns:
        An ``AgentResult`` with the assembled answer and metadata.
    """
    effective_timeout = timeout if timeout is not None else config.AGENT_TIMEOUT_SECONDS
    model_name = model or config.AGENT_MODEL

    cmd = build_cli_command(model)
    logger.info("Launching Cursor CLI: %s", " ".join(cmd))

    result = AgentResult()
    raw_log: IO[str] | None = None

    try:
        raw_log = _open_raw_log(model_name)

        # Write a header to the raw log
        raw_log.write(f"# Cursor Agent CLI invocation\n")
        raw_log.write(f"# Time : {datetime.now(timezone.utc).isoformat()}\n")
        raw_log.write(f"# Cmd  : {' '.join(cmd)}\n")
        raw_log.write(f"# Prompt length: {len(prompt)} chars\n")
        raw_log.write(f"# {'=' * 60}\n\n")
        raw_log.write(f"# --- PROMPT ---\n{prompt}\n# --- END PROMPT ---\n\n")

        start = time.monotonic()

        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        )

        # Send prompt and close stdin
        assert proc.stdin is not None
        proc.stdin.write(prompt)
        proc.stdin.close()

        # Stream stdout line by line
        assert proc.stdout is not None
        text_parts: list[str] = []
        thinking_parts: list[str] = []

        for raw_line in proc.stdout:
            # Log every line unconditionally
            raw_log.write(raw_line)
            raw_log.flush()

            event = _parse_event(raw_line)
            if event is None:
                continue

            result.raw_events.append(event)

            # Capture session_id from any event
            if not result.session_id:
                result.session_id = event.get("session_id", "")

            evt_type = event.get("type", "")
            evt_subtype = event.get("subtype", "")

            # Extract model info from system/init event
            if evt_type == "system" and evt_subtype == "init":
                result.model = event.get("model", "")

            # Accumulate assistant text and thinking
            if evt_type == "assistant":
                chunk = _extract_text_from_event(event)
                if chunk:
                    text_parts.append(chunk)
                thinking = _extract_thinking_from_event(event)
                if thinking:
                    thinking_parts.append(thinking)

            # Capture final duration from result event
            if evt_type == "result":
                result.duration_ms = event.get("duration_ms", 0)

        # Wait for process to finish
        proc.wait(timeout=effective_timeout)

        elapsed_ms = int((time.monotonic() - start) * 1000)
        if result.duration_ms == 0:
            result.duration_ms = elapsed_ms

        # Capture stderr
        assert proc.stderr is not None
        stderr_output = proc.stderr.read()
        if stderr_output.strip():
            raw_log.write(f"\n# --- STDERR ---\n{stderr_output}\n# --- END STDERR ---\n")
            logger.warning("CLI stderr: %s", stderr_output.strip()[:500])

        result.answer_text = "".join(text_parts)
        result.thinking_text = "".join(thinking_parts)

        # Write human-readable summary at end of log
        raw_log.write(f"\n# {'=' * 60}\n")
        raw_log.write(f"# SUMMARY\n")
        raw_log.write(f"# {'=' * 60}\n")
        if result.thinking_text:
            raw_log.write(f"\n# --- THINKING ---\n{result.thinking_text}\n# --- END THINKING ---\n")
        raw_log.write(f"\n# --- ANSWER ---\n{result.answer_text}\n# --- END ANSWER ---\n")

        if proc.returncode == 0:
            result.success = True
            logger.info(
                "Cursor CLI completed successfully in %dms (answer_length=%d)",
                result.duration_ms,
                len(result.answer_text),
            )
        else:
            result.error = f"CLI exited with code {proc.returncode}: {stderr_output.strip()[:300]}"
            logger.error("Cursor CLI failed: %s", result.error)

    except FileNotFoundError:
        result.error = (
            f"Cursor CLI command not found: '{config.CURSOR_CLI_CMD}'. "
            "Please install it: curl https://cursor.com/install -fsS | bash"
        )
        logger.error(result.error)

    except subprocess.TimeoutExpired:
        result.error = f"Cursor CLI timed out after {effective_timeout}s"
        logger.error(result.error)
        if proc is not None:  # type: ignore[possibly-undefined]
            proc.kill()

    except Exception as exc:
        result.error = f"Unexpected error: {exc}"
        logger.exception("Unexpected error running Cursor CLI")

    finally:
        if raw_log is not None:
            raw_log.write(f"\n# --- DONE (success={result.success}) ---\n")
            raw_log.close()

    return result
