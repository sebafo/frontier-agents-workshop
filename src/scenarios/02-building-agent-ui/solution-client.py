# =============================================================================
# Scenario 2 – AG-UI Client: Console Chat with Weather Agent (SOLUTION)
# =============================================================================
#
# GOAL:
#   Connect to the AG-UI weather agent server and interact with it through
#   a console-based interface. The client:
#     1. Sends user input to the server-hosted agent
#     2. Streams responses back in real time
#     3. Provides a client-side "set_location" command that tells the agent
#        where the user is, without the user having to phrase it as a
#        natural language request every time
#     4. Keeps conversation state consistent across turns via AG-UI threads
#
# WHY SEPARATE CLIENT AND SERVER?
#   In production, the agent logic (tools, LLM calls, MCP connections)
#   runs on a server. The UI (web, mobile, CLI) is a thin client that
#   only handles input/output. AG-UI provides the protocol boundary
#   between the two. This makes it easy to:
#     • Swap UIs without touching agent logic
#     • Scale the agent server independently
#     • Debug by inspecting the AG-UI message stream
#
# ARCHITECTURE:
#   ┌─────────────────────┐        ┌───────────────────────┐
#   │  This Console App   │──SSE──▶│  AG-UI Server         │
#   │                     │  HTTP  │  (solution-server.py)  │
#   │  • reads user input │        │  port 8888             │
#   │  • streams output   │        └───────────────────────┘
#   │  • /location cmd    │
#   └─────────────────────┘
#
# BEFORE RUNNING:
#   1. Start the MCP servers (weather + user)
#   2. Start the AG-UI server:
#        python src/scenarios/02-building-agent-ui/solution-server.py
#   3. Run this client:
#        python src/scenarios/02-building-agent-ui/solution-client.py
#      Or run in scripted mode (no manual input needed):
#        python src/scenarios/02-building-agent-ui/solution-client.py --scripted
#
# =============================================================================

import asyncio
import os

# ── Microsoft Agent Framework imports ───────────────────────────────────────
# ChatAgent:       Wraps the AGUIChatClient so we get thread/memory support
#                  on the client side as well.
# AGUIChatClient:  An AG-UI-aware chat client that connects to a remote
#                  AG-UI server endpoint via SSE/HTTP.
from agent_framework import ChatAgent
from agent_framework_ag_ui import AGUIChatClient

from dotenv import load_dotenv

load_dotenv()


# =============================================================================
# 1. CLIENT-SIDE LOCATION STATE
# =============================================================================
# We maintain the user's location on the client side. When the user types
# "/location <city>", we capture it and inject it into the next message
# so the server-side agent knows where the user is. This demonstrates how
# a client can contribute context to the agent without modifying the server.
user_location: str | None = None


def format_location_context(message: str) -> str:
    """
    If a user location is set, prepend it as context to the user's message.

    This is a simple approach: we prefix the message with a system-like
    note so the agent can read the location. The agent's instructions
    tell it to remember the user's location from conversation history,
    so after the first mention it will remember it for subsequent turns.
    """
    if user_location:
        return f"[User's current location is: {user_location}] {message}"
    return message


# =============================================================================
# 2. INTERACTIVE MODE
# =============================================================================
async def interactive_mode() -> None:
    """
    Run an interactive console chat loop connected to the AG-UI server.

    Commands:
      /location <city> — Set your location (e.g., /location Seattle)
      /help            — Show available commands
      :q  or  quit     — Exit

    Everything else is sent as a natural language message to the agent.
    """

    server_url = os.environ.get("AGUI_SERVER_URL", "http://127.0.0.1:8888/")

    print("=" * 70)
    print("  Scenario 2 — AG-UI Console Client")
    print("=" * 70)
    print(f"  Server: {server_url}")
    print()
    print("  Commands:")
    print("    /location <city>  — Set your current location")
    print("    /help             — Show this help")
    print("    :q  or  quit      — Exit")
    print()

    # ── Create the AG-UI chat client ────────────────────────────────────
    # AGUIChatClient speaks the AG-UI protocol (SSE streaming, tool
    # metadata forwarding, thread management) over HTTP.
    chat_client = AGUIChatClient(endpoint=server_url)

    # ── Wrap it in a ChatAgent for thread management ────────────────────
    # By wrapping the AGUIChatClient in a ChatAgent, we get automatic
    # thread (conversation history) management on the client side.
    # The instructions here are minimal because all the real reasoning
    # happens on the server-side agent.
    agent = ChatAgent(
        name="ConsoleClient",
        chat_client=chat_client,
        instructions="You are a client relaying messages to a remote agent.",
    )

    # ── Get a thread for conversation continuity ────────────────────────
    thread = agent.get_new_thread()

    global user_location

    try:
        while True:
            # ── Read user input ─────────────────────────────────────────
            try:
                raw_input = input("\nUser (:q to exit, /help for commands): ")
            except (EOFError, KeyboardInterrupt):
                print("\n\nGoodbye!")
                break

            message = raw_input.strip()
            if not message:
                print("  (empty input, try again)")
                continue

            # ── Handle commands ─────────────────────────────────────────
            if message.lower() in (":q", "quit", "exit"):
                print("Goodbye!")
                break

            if message.lower() == "/help":
                print("  /location <city>  — Set your current location")
                print("  /help             — Show this help")
                print("  :q  or  quit      — Exit")
                continue

            if message.lower().startswith("/location"):
                parts = message.split(maxsplit=1)
                if len(parts) < 2 or not parts[1].strip():
                    print("  Usage: /location <city>  (e.g., /location Seattle)")
                    continue
                user_location = parts[1].strip()
                print(f"  Location set to: {user_location}")
                # Also tell the agent about the location change
                message = f"Set my location to {user_location}"

            # ── Inject location context and send to agent ───────────────
            enriched = format_location_context(message)

            print("\nAssistant: ", end="", flush=True)

            # ── Stream the response ─────────────────────────────────────
            # agent.run_stream() sends the message to the AG-UI server,
            # which forwards it to the ChatAgent on the server side.
            # The server streams back SSE events containing text chunks,
            # tool call notifications, and other activities.
            async for update in agent.run_stream(enriched, thread=thread):
                if update.text:
                    print(f"\033[96m{update.text}\033[0m", end="", flush=True)

            print("\n")

    except Exception as e:
        print(f"\n\033[91mError: {e}\033[0m")
        import traceback
        traceback.print_exc()


# =============================================================================
# 3. SCRIPTED MODE
# =============================================================================
async def scripted_mode() -> None:
    """
    Run the predefined input queries from the README automatically.

    This is useful for testing and demonstration — no manual input needed.
    """

    server_url = os.environ.get("AGUI_SERVER_URL", "http://127.0.0.1:8888/")

    print("=" * 70)
    print("  Scenario 2 — AG-UI Console Client (Scripted Mode)")
    print("=" * 70)
    print(f"  Server: {server_url}")
    print()

    chat_client = AGUIChatClient(endpoint=server_url)

    agent = ChatAgent(
        name="ConsoleClient",
        chat_client=chat_client,
        instructions="You are a client relaying messages to a remote agent.",
    )

    thread = agent.get_new_thread()

    # ── Scripted test queries from the README ───────────────────────────
    # These exercise the key requirements:
    #   1. Set location + get weather → tests location setting & weather tool
    #   2. Follow-up weather question → tests conversation memory
    #   3. Summarize conversation → tests thread-based history recall
    #   4. Change location + forecast → tests location update & weather
    test_queries = [
        "Set my location to Seattle and tell me today's weather.",
        "What will the weather be like tomorrow here?",
        "Can you summarize the last three things I asked you?",
        "Change my location to Tokyo and give me a short forecast.",
    ]

    for i, query in enumerate(test_queries, start=1):
        print(f"{'─' * 60}")
        print(f"  Turn {i}")
        print(f"{'─' * 60}")
        print(f"  User: {query}")
        print(f"  Assistant: ", end="", flush=True)

        async for update in agent.run_stream(query, thread=thread):
            if update.text:
                print(f"\033[96m{update.text}\033[0m", end="", flush=True)

        print("\n")

    print("=" * 70)
    print("  All scripted queries completed!")
    print("=" * 70)
    print()
    print("  KEY TAKEAWAYS:")
    print("  • The AG-UI protocol cleanly separates UI from agent logic.")
    print("  • The server hosts all tools (MCP + local) and does reasoning.")
    print("  • The client only handles input/output and optional context.")
    print("  • Conversation history flows through the AG-UI thread.")
    print("  • Streaming gives real-time feedback in the console.")
    print()
    print("  NEXT STEPS:")
    print("  • Try --interactive mode to chat freely with the agent.")
    print("  • Inspect the AG-UI SSE stream to understand the wire protocol.")
    print("  • Swap the console client for a web frontend using AG-UI JS SDK.")


# =============================================================================
# 4. ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Scenario 2 — AG-UI Console Client"
    )
    parser.add_argument(
        "--scripted",
        action="store_true",
        help="Run predefined test queries instead of interactive mode",
    )
    args = parser.parse_args()

    if args.scripted:
        asyncio.run(scripted_mode())
    else:
        asyncio.run(interactive_mode())
