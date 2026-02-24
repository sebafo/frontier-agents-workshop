#!/usr/bin/env bash
# =============================================================================
# start-client.sh — Start the AG-UI console client
# =============================================================================
#
# Connects to the AG-UI server at http://localhost:8888/ and opens an
# interactive console chat with the weather agent.
#
# Prerequisites:
#   1. MCP servers must be running  (./src/mcp-server/start-mcp.sh)
#   2. AG-UI server must be running (./start-server.sh)
#
# Usage:
#   chmod +x start-client.sh
#   ./start-client.sh              # interactive mode
#   ./start-client.sh --scripted   # run predefined test queries
#
# Stop:
#   Type :q or quit, or press Ctrl+C
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Starting AG-UI Console Client ==="
echo ""

python "$SCRIPT_DIR/solution-client.py" "$@"
