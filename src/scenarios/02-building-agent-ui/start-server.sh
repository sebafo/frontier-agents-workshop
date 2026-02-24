#!/usr/bin/env bash
# =============================================================================
# start-server.sh — Start the AG-UI weather agent server
# =============================================================================
#
#   Server           Port   Endpoint
#   ─────────────────────────────────────────────
#   AG-UI Server     8888   http://localhost:8888/
#
# Prerequisites:
#   MCP servers must be running (./src/mcp-server/start-mcp.sh)
#
# Usage:
#   chmod +x start-server.sh
#   ./start-server.sh
#
# Stop:
#   Press Ctrl+C
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Starting AG-UI Weather Agent Server ==="
echo ""
echo "  AG-UI endpoint → http://localhost:8888/"
echo "  Press Ctrl+C to stop."
echo ""

python "$SCRIPT_DIR/solution-server.py"
