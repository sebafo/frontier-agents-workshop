#!/usr/bin/env bash
# =============================================================================
# start-a2a-server.sh — Start the A2A weather agent server
# =============================================================================
#
#   Server             Port   Endpoint
#   ─────────────────────────────────────────────────────────
#   A2A Weather Agent  8080   http://localhost:8080/
#   Agent Card                http://localhost:8080/.well-known/agent.json
#
# Prerequisites:
#   MCP weather server must be running (./src/mcp-server/start-mcp.sh)
#
# Usage:
#   chmod +x start-a2a-server.sh
#   ./start-a2a-server.sh
#
# Stop:
#   Press Ctrl+C
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Starting A2A Weather Agent Server ==="
echo ""
echo "  A2A endpoint  → http://localhost:8080/"
echo "  Agent card    → http://localhost:8080/.well-known/agent.json"
echo "  Press Ctrl+C to stop."
echo ""

python "$SCRIPT_DIR/solution-a2a-server.py"
