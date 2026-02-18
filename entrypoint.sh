#!/bin/bash

set -e
echo "Generating MCP approval key..."
python app/mcp/generate_mcp_approval.py --cwd /workspace

# Execute main command
exec "$@"