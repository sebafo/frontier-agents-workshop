#!/usr/bin/env bash
# =============================================================================
# start-travel-planner.sh — Start the travel planner client
# =============================================================================
#
# Connects to the A2A weather agent at http://localhost:8080/ and runs
# the travel planning agent that delegates weather checks via A2A.
#
# Prerequisites:
#   1. MCP weather server must be running  (./src/mcp-server/start-mcp.sh)
#   2. A2A weather server must be running  (./start-a2a-server.sh)
#
# Usage:
#   chmod +x start-travel-planner.sh
#   ./start-travel-planner.sh                # run predefined test queries
#   ./start-travel-planner.sh --interactive  # free-form interactive chat
#
# Stop:
#   Type quit/exit, or press Ctrl+C
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Starting Travel Planner Client ==="
echo ""

python "$SCRIPT_DIR/solution-travel-planner.py" "$@"
