"""
Generate MCP approval key from mcp.json.

Usage:
    python generate_mcp_approval.py [--config PATH] [--cwd PATH] [--server NAME]
"""

import argparse
import hashlib
import json
from pathlib import Path
import os

# Default paths and server name
DEFAULT_CONFIG_PATH = "/root/.cursor/mcp.json"
DEFAULT_SERVER_NAME = "ragent"

def generate_approval_key(
    cwd: str,
    approval_dir: str,
    config_path: str,
    server_name: str,
) -> bool:
    """
    Generate MCP approval key from mcp.json config.

    Args:
        cwd: Working directory for hash
        approval_dir: Directory to write approval file
        config_path: Path to mcp.json
        server_name: MCP server name

    Returns:
        True if successful, False otherwise
    """
    try:
        # Read config file
        with open(config_path) as f:
            config = json.load(f)

        # Validate server
        servers = config.get("mcpServers")
        if not servers:
            print("No mcpServers found in config")
            return False

        if server_name not in servers:
            print(f"Server '{server_name}' not found in config")
            return False

        server_info = servers[server_name]

        # Generate hash (Node.js JSON.stringify equivalent)
        hash_input = {"path": cwd, "server": server_info}
        hash_str = json.dumps(hash_input, separators=(",", ":"), ensure_ascii=False)
        hash_value = hashlib.sha256(hash_str.encode()).hexdigest()[:16]
        approval_key = f"{server_name}-{hash_value}"

        # Write approval file
        approval_path = Path(approval_dir)
        approval_path.mkdir(parents=True, exist_ok=True)
        approval_file = approval_path / "mcp-approvals.json"
        with open(approval_file, "w") as f:
            json.dump([approval_key], f)

        print(f"MCP approval key: {approval_key} successfully generated")
        print(f"Written to: {approval_file}")
        return True

    except FileNotFoundError:
        print(f"Config file not found: {config_path}")
        return False
    except json.JSONDecodeError as e:
        print(f"Invalid JSON in config: {e}")
        return False
    except Exception as e:
        print(f"Failed to generate approval key: {e}")
        return False


def main():
    current_dir = os.getcwd()
    approval_dir = os.path.join("/root/.cursor/projects", current_dir.lstrip("/"))


    parser = argparse.ArgumentParser(description="Generate MCP approval key")
    parser.add_argument(
        "--cwd",
        default=current_dir,
        help=f"Working directory for hash (default=current working directory: {current_dir})"
    )
    parser.add_argument(
        "--config", "-c",
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to mcp.json (default: {DEFAULT_CONFIG_PATH})"
    )
    parser.add_argument(
        "--server", "-s",
        default=DEFAULT_SERVER_NAME,
        help=f"MCP server name (default: {DEFAULT_SERVER_NAME})"
    )

    args = parser.parse_args()

    print(
        f"Generate MCP approval key info: "
        f"cwd: {args.cwd}, approval_dir: {approval_dir}, config_path: {args.config}, server_name: {args.server}"
    )
    success = generate_approval_key(
        cwd=args.cwd,
        approval_dir=approval_dir,
        config_path=args.config,
        server_name=args.server,
    )

    exit(0 if success else 1)


if __name__ == "__main__":
    main()
