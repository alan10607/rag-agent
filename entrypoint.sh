#!/bin/bash

set -e
echo "Generating MCP approval key..."
python ragent/mcp/generate_mcp_approval.py --cwd /app

# Execute main command
exec "$@"