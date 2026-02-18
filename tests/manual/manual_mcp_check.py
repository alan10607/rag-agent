#!/usr/bin/env python3
"""
MCP Server STDIO Test Script

1. Starts MCP server as subprocess
2. Initializes protocol
3. Lists registered tools
4. Calls semantic_search tool
5. Prints all responses
"""

import json
import subprocess
import threading
import queue
import time

# -------------------------------
# Start MCP server
# -------------------------------
proc = subprocess.Popen(
    ["venv/bin/python3", "-u", "-m", "ragent.mcp.main"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    bufsize=1
)

# Queue to capture stdout lines
stdout_queue = queue.Queue()

# Thread to continuously read MCP stdout
def stdout_reader(pipe, queue):
    for line in pipe:
        if line.strip():
            queue.put(line.strip())

threading.Thread(target=stdout_reader, args=(proc.stdout, stdout_queue), daemon=True).start()

# Optional: capture stderr
def stderr_reader(pipe):
    for line in pipe:
        if line.strip():
            print("[SERVER STDERR]", line.strip())

threading.Thread(target=stderr_reader, args=(proc.stderr,), daemon=True).start()

# -------------------------------
# Helper to send JSON-RPC message
# -------------------------------
def send(msg, wait_response=True, timeout=5):
    """
    Send JSON-RPC message to MCP server and optionally wait for first response
    """
    try:
        proc.stdin.write(json.dumps(msg) + "\n")
        proc.stdin.flush()
    except BrokenPipeError:
        print("BrokenPipeError: MCP server stdin closed")
        return None

    if wait_response:
        start = time.time()
        while time.time() - start < timeout:
            try:
                line = stdout_queue.get(timeout=0.1)
                # JSON object line
                try:
                    parsed = json.loads(line)
                    return parsed
                except json.JSONDecodeError:
                    # Sometimes MCP prints partial JSON; ignore
                    continue
            except queue.Empty:
                continue
        print("No response received within timeout")
        return None
    return None

# -------------------------------
# 1. Initialize
# -------------------------------
init_request = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "clientInfo": {
            "name": "test-client",
            "version": "1.0"
        }
    }
}
resp = send(init_request)
print("1. Initialize response:", resp)

# -------------------------------
# 2. List tools
# -------------------------------
list_tools_request = {
    "jsonrpc": "2.0",
    "id": 2,
    "method": "tools/list",
    "params": {}
}
resp = send(list_tools_request)
print("2. List tools response:", resp)

# -------------------------------
# 3. Call semantic_search tool
# -------------------------------
call_tool_request = {
    "jsonrpc": "2.0",
    "id": 3,
    "method": "tools/call",
    "params": {
        "name": "semantic_search",
        "arguments": {
            "query": "test query",
            "top_k": 1
        }
    }
}
resp = send(call_tool_request, timeout=10)
print("3. Call semantic_search response:", resp)

# -------------------------------
# Shutdown server
# -------------------------------
proc.terminate()
proc.wait()
print("MCP server terminated")
