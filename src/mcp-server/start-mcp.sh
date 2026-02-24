#!/usr/bin/env bash
# =============================================================================
# start-mcp.sh — Start all MCP servers
# =============================================================================
#
# Launches every MCP server in the mcp-server/ directory:
#
#   Server                Port   Endpoint
#   ─────────────────────────────────────────────
#   04-weather-server     8001   http://localhost:8001/mcp
#   02-user-server        8002   http://localhost:8002/mcp
#   01-customer-server    8003   http://localhost:8003/mcp
#   03-banking-server     8004   http://localhost:8004/mcp
#
# Usage:
#   chmod +x start-mcp.sh
#   ./start-mcp.sh
#
# Stop:
#   Press Ctrl+C — all servers will be terminated.
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIDS=()

echo "=== Starting all MCP servers (4) ==="
echo ""

# --- Weather server (port 8001) ---
echo "  [1/4] Weather server   → http://localhost:8001/mcp"
python "$SCRIPT_DIR/04-weather-server/server-mcp-sse-weather.py" &
PIDS+=($!)

# --- User server (port 8002) ---
echo "  [2/4] User server      → http://localhost:8002/mcp"
python "$SCRIPT_DIR/02-user-server/server-mcp-sse-user.py" &
PIDS+=($!)

# --- Customer server (port 8003) ---
echo "  [3/4] Customer server  → http://localhost:8003/mcp"
python "$SCRIPT_DIR/01-customer-server/server-mcp-sse-customers.py" &
PIDS+=($!)

# --- Banking server (port 8004) ---
echo "  [4/4] Banking server   → http://localhost:8004/mcp"
python "$SCRIPT_DIR/03-banking-server/server-mcp-sse-banking.py" &
PIDS+=($!)

echo ""
echo "  All servers running. PIDs: ${PIDS[*]}"
echo "  Press Ctrl+C to stop all."
echo ""

# Clean up all servers on exit
cleanup() {
    echo ""
    echo "  Stopping MCP servers..."
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait "${PIDS[@]}" 2>/dev/null || true
    echo "  Done."
}
trap cleanup EXIT INT TERM

# Wait for all background processes
wait
