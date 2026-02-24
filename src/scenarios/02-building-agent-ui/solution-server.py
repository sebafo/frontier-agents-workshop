# =============================================================================
# Scenario 2 – AG-UI Server: Weather Agent (SOLUTION)
# =============================================================================
#
# GOAL:
#   Host a weather-capable agent behind the AG-UI protocol so that any
#   AG-UI client (console, web, mobile) can connect and interact with it.
#
# ARCHITECTURE OVERVIEW:
#   ┌──────────────────┐         ┌────────────────────────┐
#   │  AG-UI Client    │◀──SSE──▶│  This FastAPI Server   │
#   │  (solution-      │  HTTP   │  (port 8888)           │
#   │   client.py)     │         │                        │
#   └──────────────────┘         │  ┌──────────────────┐  │
#                                │  │   ChatAgent      │  │
#                                │  │  + local time    │  │
#                                │  │  + MCP weather   │  │
#                                │  │  + MCP user      │  │
#                                │  └──────────────────┘  │
#                                └────────┬───────┬───────┘
#                                         │       │
#                           ┌─────────────┘       └──────────────┐
#                           ▼                                    ▼
#                   ┌───────────────┐                  ┌──────────────────┐
#                   │ MCP: Weather  │ (Port 8001)      │ MCP: User        │ (Port 8002)
#                   │ • weather     │                  │ • user / location│
#                   │ • locations   │                  │ • move           │
#                   └───────────────┘                  └──────────────────┘
#
# BEFORE RUNNING:
#   1. Make sure you have a valid .env file
#   2. Start the MCP servers:
#        python src/mcp-server/02-user-server/server-mcp-sse-user.py
#        python src/mcp-server/04-weather-server/server-mcp-sse-weather.py
#   3. Start this server:
#        python src/scenarios/02-building-agent-ui/solution-server.py
#   4. In a separate terminal, run the client:
#        python src/scenarios/02-building-agent-ui/solution-client.py
#
# =============================================================================

# ── Standard library imports ────────────────────────────────────────────────
import sys
from pathlib import Path
import os

# ── Ensure the project root is on sys.path ──────────────────────────────────
# Go up three levels: 02-building-agent-ui → scenarios → src → project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# ── Third-party / framework imports ────────────────────────────────────────
from dotenv import load_dotenv

# ── Microsoft Agent Framework imports ───────────────────────────────────────
from agent_framework import ChatAgent, MCPStreamableHTTPTool

# ── AG-UI integration ──────────────────────────────────────────────────────
# add_agent_framework_fastapi_endpoint wires a ChatAgent into a FastAPI app
# as an AG-UI-compatible endpoint, handling SSE streaming, tool execution,
# thread management, etc.
from agent_framework_ag_ui import add_agent_framework_fastapi_endpoint

from fastapi import FastAPI

# ── Shared model client helper ──────────────────────────────────────────────
from samples.shared.model_client import create_chat_client

# ── Load environment variables ──────────────────────────────────────────────
load_dotenv()


# =============================================================================
# 1. CONFIGURE THE LLM CLIENT
# =============================================================================
completion_model_name = os.environ.get("COMPLETION_DEPLOYMENT_NAME")
medium_model_name = os.environ.get("MEDIUM_DEPLOYMENT_MODEL_NAME")

completion_client = create_chat_client(completion_model_name)
medium_client = create_chat_client(medium_model_name)


# =============================================================================
# 2. CONFIGURE MCP SERVER URLS
# =============================================================================
USER_MCP_URL = os.environ.get("USER_MCP_URL", "http://localhost:8002/mcp")
WEATHER_MCP_URL = os.environ.get("WEATHER_MCP_URL", "http://localhost:8001/mcp")


# =============================================================================
# 3. AGENT SYSTEM PROMPT
# =============================================================================
AGENT_INSTRUCTIONS = """
You are a friendly, concise assistant for time and weather queries.

TOOLS YOU HAVE:
- get_current_time(location) — returns the current local time
  for an IANA timezone (e.g. "Europe/London", "Europe/Berlin").
- get_weather_at_location(location) — returns weather for a city name.
- list_supported_locations() — lists cities supported for weather.
- move(user, newlocation) — updates the user's stored location.
- get_current_user() — returns the logged-in username.
- get_current_location(user) — returns the user's stored location.

BEHAVIOUR:
- Remember the user's location from conversation history. Do NOT ask
  the user to repeat their location if they already told you.
- When the user says they moved to a new city or sets their location,
  call the move() tool to persist the change.
- Always use tools for real data — never make up times or weather.
- When reporting weather, include the local time too.
- If a location is not supported for weather, call
  list_supported_locations and tell the user which ones are available.
- When the user asks you to summarize past conversation, look through
  the conversation history and provide a concise summary.
"""


# =============================================================================
# 4. BUILD THE TOOLS LIST
# =============================================================================
# All tools come from the MCP servers — no local tools needed.
# The User MCP server already provides get_current_time which handles
# timezone lookups, so we don't need a separate local tool for that.
tools = [
    MCPStreamableHTTPTool(name="User Server", url=USER_MCP_URL),
    MCPStreamableHTTPTool(name="Weather Server", url=WEATHER_MCP_URL),
]


# =============================================================================
# 5. CREATE THE AGENT
# =============================================================================
agent = ChatAgent(
    chat_client=completion_client,
    name="TimeWeatherAgent",
    instructions=AGENT_INSTRUCTIONS,
    tools=tools,
)


# =============================================================================
# 6. CREATE FASTAPI APP AND REGISTER AG-UI ENDPOINT
# =============================================================================
# The AG-UI endpoint translates between the AG-UI wire protocol (SSE-based
# streaming, tool call forwarding, thread management) and the ChatAgent's
# run/run_stream interface. This single line is all we need to expose the
# full agent over AG-UI.
app = FastAPI(title="Scenario 2 — AG-UI Weather Agent Server")

add_agent_framework_fastapi_endpoint(app, agent, "/")


# =============================================================================
# 7. ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    import uvicorn

    print("=" * 70)
    print("  Scenario 2 — AG-UI Weather Agent Server")
    print("=" * 70)
    print()
    print("  Serving on http://127.0.0.1:8888/")
    print("  Press Ctrl+C to stop.")
    print()
    print("  In another terminal, run the client:")
    print("    python src/scenarios/02-building-agent-ui/solution-client.py")
    print()

    uvicorn.run(app, host="127.0.0.1", port=8888)
