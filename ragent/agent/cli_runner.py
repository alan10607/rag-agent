"""
Ragent - Cursor Agent CLI Runner

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
import re
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import IO, Dict, Any, List, Union, Optional

from ragent import config
from ragent.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SemanticSearchResult:
    index: Union[int, str] = "?"
    score: float = 0.0
    source: str = "unknown"
    page: Optional[Union[int, str]] = None
    chunk_index: Union[int, str] = "?"
    text: str = "?"

@dataclass(frozen=True)
class MCPToolResult:
    tool_name: str
    tool_args: Dict[str, Any]
    success: bool
    timestamp_ms: int
    raw_text: str
    is_semantic_search: bool = False
    
    def __post_init__(self):
        if self.tool_name == "ragent-semantic_search":
            object.__setattr__(self, "is_semantic_search", True)

    @classmethod
    def from_event(cls, event: dict) -> "MCPToolResult":
        """Factory method to convert raw event dict to MCPToolResult object"""
        mcp_call = event.get("tool_call", {}).get("mcpToolCall", {})
        args_data = mcp_call.get("args", {})
        
        tool_name = args_data.get("name") or "unknown"
        tool_args = args_data.get("args", {})
        is_error = event.get("result", {}).get("success", {}).get("isError", False)
        
        result_data = mcp_call.get("result", {})
        content_list = result_data.get("success", {}).get("content", [])
        raw_text = ""
        if isinstance(content_list, list) and content_list:
            raw_text = content_list[0].get("text", {}).get("text", "")

        return cls(
            tool_name=tool_name,
            tool_args=tool_args,
            success=not is_error,
            timestamp_ms=event.get("timestamp_ms", 0),
            raw_text=raw_text
        )


    def to_semantic_results(self) -> List[SemanticSearchResult]:
        """Convert raw text to a list of SemanticSearchResult objects"""
        if not self.is_semantic_search or not self.raw_text:
            return []

        try:
            data = json.loads(self.raw_text)
            contents = data.get("content", []) if isinstance(data, dict) else []
            
            results = []
            for entry in contents:
                if entry.get("type") == "text" and isinstance(entry.get("text"), list):
                    for chunk in entry["text"]:
                        if isinstance(chunk, dict):
                            results.append(SemanticSearchResult(
                                index=chunk.get("index", "?"),
                                score=chunk.get("score", 0.0),
                                source=chunk.get("source", "unknown"),
                                page=chunk.get("page"),
                                chunk_index=chunk.get("chunk_index", "?"),
                                text=chunk.get("text", "?")
                            ))
            return results
        except (json.JSONDecodeError, TypeError):
            return []



@dataclass
class AgentResult:
    """Holds the aggregated result returned by Cursor Agent CLI."""

    answer_text: str = ""
    context_chunks: list[dict] = field(default_factory=list)
    raw_events: list[dict] = field(default_factory=list)
    mcp_results: list[MCPToolResult] = field(default_factory=list)
    vector_search_results: list[SemanticSearchResult] = field(default_factory=list)
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

            evt_type = event.get("type", "unknown")
            evt_subtype = event.get("subtype", "unknown")
            logger.info("Captured a event type: %s, subtype: %s", evt_type, evt_subtype)

            # Extract model info from system/init event
            if evt_type == "system" and evt_subtype == "init":
                result.model = event.get("model", "")

            # Process tool call event
            if evt_type == "tool_call" and evt_subtype == "completed":
                logger.info("Captured a tool call event with completed subtype, now parsing the result")
                mcp_result = MCPToolResult.from_event(event)
                logger.info(
                    "Parsed tool call: name: %s, args: %s, success: %s, timestamp: %s", 
                    mcp_result.tool_name, mcp_result.tool_args, mcp_result.success, mcp_result.timestamp_ms
                )
                result.mcp_results.append(mcp_result)

                if mcp_result.is_semantic_search and mcp_result.success:
                    logger.info("The result is a semantic sesarch, now parsing the semantic search results")
                    result.vector_search_results.extend(mcp_result.to_semantic_results())
                    logger.info("Parsed the semantic search results count: %d", len(result.vector_search_results))
                    for r in result.vector_search_results:
                        logger.info("Search result: index: %s, score: %f, source: %s, page: %s, chunk_index: %s", 
                            r.index, r.score, r.source, r.page, r.chunk_index)

            # Capture final duration from result event
            if evt_type == "result":
                result.duration_ms = event.get("duration_ms", 0)
                result.session_id = event.get("session_id", "")
                result.answer_text = event.get("result", "No result found in  Cursor Agent CLI event")
                logger.info(
                    "Captured the final answer text length: %d, model: %s, duration: %dms", 
                    len(result.answer_text), result.model, result.duration_ms
                )



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

        # Write human-readable summary at end of log
        raw_log.write(f"\n# {'=' * 60}\n")
        raw_log.write(f"# SUMMARY\n")
        raw_log.write(f"# {'=' * 60}\n")
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
        if result.mcp_results:
            # --- MCP TOOLS USED ---
            raw_log.write("\n# --- MCP TOOLS USED ---\n")
            for r in result.mcp_results:
                raw_log.write(f"# {r.tool_name}: args: {r.tool_args} success: {r.success} timestamp_ms: {r.timestamp_ms}\n")
            raw_log.write("# --- END MCP TOOLS USED ---\n")
            logger.info("MCP tools (%d) used in this session", len(result.mcp_results))

            # --- VECTOR SEARCH RESULTS ---
            raw_log.write("\n# --- VECTOR SEARCH RESULTS ---\n")
            for r in result.vector_search_results:
                raw_log.write(
                    f"# {r.index}, score: {r.score}, source: {r.source}, "
                    f"page: {r.page}, chunk_index: {r.chunk_index}\n"
                )
            raw_log.write("# --- END VECTOR SEARCH RESULTS ---\n")  
            logger.info("Vector search results (%d) found in this session", len(result.vector_search_results))

        if raw_log is not None:
            raw_log.write(f"\n# --- DONE (success={result.success}) ---\n")
            raw_log.close()

    return result


def check_mcp_status(mcp_name: str = "ragent") -> bool:
    """
    Check if the Cursor Agent MCP is ready.
    Returns True if the status is 'ready', False otherwise.
    """
    try:
        # Execute the command and capture stdout
        result = subprocess.run(
            ["agent", "mcp", "list"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True
        )
        
        # Get the output content and clean it (remove extra whitespace and newlines)
        output = result.stdout.strip()
        ansi_escape = re.compile(r'(?:\x1B[@-_]|[\x80-\x9F])[0-?]*[ -/]*[@-~]')
        output = ansi_escape.sub('', output)
        
        logger.info(f"Check MCP output: {output}")
        if "ready" in output:
            logger.info(f"MCP is ready.")
            return True
        else:
            logger.warning(f"MCP status abnormal")
            return False

    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to run 'agent mcp list'. Error: {e.stderr}")
        return False
    except FileNotFoundError:
        logger.error("'agent' command not found. Make sure Cursor Agent is installed.")
        return False