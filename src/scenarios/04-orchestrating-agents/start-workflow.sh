#!/usr/bin/env bash
# =============================================================================
# start-workflow.sh — Start the Day Planner workflow (Scenario 04)
# =============================================================================
#
# Prerequisites:
#   MCP Weather server must be running on port 8001:
#     python src/mcp-server/04-weather-server/server-mcp-sse-weather.py
#   (or run ./src/mcp-server/start-mcp.sh to start all MCP servers)
#
# Usage:
#   chmod +x start-workflow.sh
#   ./start-workflow.sh                           # run default query
#   ./start-workflow.sh --interactive             # interactive mode
#   ./start-workflow.sh --devui                   # DevUI web interface
#   ./start-workflow.sh "Check weather in Tokyo"  # custom query
#
# Stop:
#   Press Ctrl+C
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

cd "$ROOT_DIR"

echo "=== Scenario 04 — Orchestrating Agents with Workflows ==="
echo ""

python "$SCRIPT_DIR/solution.py" "$@"
