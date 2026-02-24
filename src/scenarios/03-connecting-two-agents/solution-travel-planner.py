# =============================================================================
# Scenario 3 – Travel Planning Agent with A2A Weather (SOLUTION)
# =============================================================================
#
# GOAL:
#   Build a travel planning agent that consults a remote weather agent via
#   the A2A protocol to plan trips only to locations with good weather.
#
# ARCHITECTURE OVERVIEW:
#   ┌────────────────────────────────────┐
#   │  This Travel Planner Agent         │
#   │  (ChatAgent + A2AAgent as tool)    │
#   │                                    │
#   │  Instructions:                     │
#   │  "Plan 5-day trips to places       │
#   │   with good weather"              │
#   │                                    │
#   │  Tools:                            │
#   │  • A2AAgent (weather check)        │
#   └──────────────┬─────────────────────┘
#                  │ A2A protocol
#                  ▼
#   ┌──────────────────────────────────┐
#   │  A2A Weather Agent Server        │
#   │  (solution-a2a-server.py)        │
#   │  Port 8080                       │
#   │  /.well-known/agent.json         │
#   └──────────────────────────────────┘
#
# BEFORE RUNNING:
#   1. Make sure you have a valid .env file
#   2. Start the Weather MCP server (port 8001):
#        python src/mcp-server/04-weather-server/server-mcp-sse-weather.py
#   3. Start the A2A Weather server (port 8080):
#        python src/scenarios/03-connecting-two-agents/solution-a2a-server.py
#   4. Run this travel planner:
#        python src/scenarios/03-connecting-two-agents/solution-travel-planner.py
#
# =============================================================================

# ── Standard library imports ────────────────────────────────────────────────
import sys
import os
import asyncio
from pathlib import Path

# ── Ensure the project root is on sys.path ──────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# ── Third-party / framework imports ────────────────────────────────────────
from dotenv import load_dotenv
import httpx

# ── A2A SDK imports ─────────────────────────────────────────────────────────
from a2a.client import A2ACardResolver

# ── Microsoft Agent Framework imports ───────────────────────────────────────
from agent_framework import ChatAgent, AgentThread
from agent_framework.a2a import A2AAgent

# ── Shared model client helper ──────────────────────────────────────────────
from samples.shared.model_client import create_chat_client

# ── Load environment variables ──────────────────────────────────────────────
load_dotenv()


# =============================================================================
# 1. CONFIGURE THE LLM CLIENT
# =============================================================================
completion_model_name = os.environ.get("COMPLETION_DEPLOYMENT_NAME")
completion_client = create_chat_client(completion_model_name)


# =============================================================================
# 2. A2A WEATHER AGENT URL
# =============================================================================
# The A2A weather agent server must be running at this address.
# Default: http://localhost:8080 (started by solution-a2a-server.py)
A2A_WEATHER_HOST = os.environ.get("A2A_AGENT_HOST", "http://localhost:8080")


# =============================================================================
# 3. TRAVEL PLANNER SYSTEM PROMPT
# =============================================================================
TRAVEL_PLANNER_INSTRUCTIONS = """
You are an expert travel planner specializing in European destinations.

YOUR CAPABILITIES:
- You can check weather conditions at any location by asking the weather
  agent (via the A2A tool available to you).
- You plan detailed 5-day trip itineraries based on weather conditions.
- You compare multiple destinations to find the best option.
- DON'T MAKE UP ANY WEATHER INFORMATION YOURSELF — ALWAYS use the weather agent for accurate data. If not available, say you can't check the weather

BEHAVIOUR:
- When asked to plan a trip, ALWAYS check the weather at candidate
  locations FIRST using your weather tool before making recommendations.
- Only recommend locations where the weather is "sunny" or "cloudy"
  (acceptable for travel). Avoid locations with "rainy" or "stormy"
  weather.
- If the weather is bad at the user's preferred location, proactively
  suggest alternative cities with better weather.
- When planning a 5-day itinerary, include:
  • Day-by-day activities and sightseeing recommendations
  • Local cuisine suggestions
  • Practical travel tips
- Always explain WHICH locations you checked and WHY you chose the
  final destination (transparency about your reasoning).
- If asked to shift dates, re-check weather conditions for the new
  dates before updating the itinerary.

EUROPEAN CITIES TO CONSIDER:
  Paris, London, Berlin, Barcelona, Rome, Amsterdam, Prague, Vienna,
  Lisbon, Madrid, Athens, Copenhagen, Stockholm, Dublin, Zurich,
  Budapest, Edinburgh, Florence, Nice, Dubrovnik

WEATHER INTERPRETATION:
  - "sunny" → Excellent for travel (outdoor activities, sightseeing)
  - "cloudy" → Good for travel (comfortable for walking tours)
  - "rainy" → Not ideal (suggest indoor alternatives or different city)
  - "stormy" → Bad for travel (strongly recommend alternative)
"""


# =============================================================================
# 4. MAIN FUNCTION
# =============================================================================
async def main() -> None:
    """
    Main entry point:
      1. Connects to the A2A weather agent server.
      2. Creates a travel planner ChatAgent that uses the weather agent
         as a tool (via A2A protocol).
      3. Runs a series of travel planning queries.
    """

    print("=" * 70)
    print("  Scenario 3 — Travel Planner with A2A Weather Agent")
    print("=" * 70)
    print()
    print(f"  Connecting to A2A weather agent at: {A2A_WEATHER_HOST}")
    print()

    # ── Discover the A2A weather agent ──────────────────────────────────
    # The A2ACardResolver retrieves the agent's capabilities from its
    # well-known endpoint (/.well-known/agent.json). This tells us what
    # the agent can do, its name, description, and supported modes.
    async with httpx.AsyncClient(timeout=60.0) as http_client:
        resolver = A2ACardResolver(
            httpx_client=http_client, base_url=A2A_WEATHER_HOST
        )
        agent_card = await resolver.get_agent_card()

    print(f"  Found weather agent: {agent_card.name}")
    print(f"  Description: {agent_card.description}")
    print()

    # ── Create A2AAgent wrapper ─────────────────────────────────────────
    # A2AAgent wraps an external A2A-compliant agent as a tool that our
    # ChatAgent can invoke. The travel planner sees it as just another
    # tool in its toolkit — but under the hood it sends A2A messages
    # to the remote weather agent.
    weather_a2a_agent = A2AAgent(
        name=agent_card.name,
        description=agent_card.description,
        agent_card=agent_card,
        url=A2A_WEATHER_HOST,
    )

    # ── Build the travel planner agent ──────────────────────────────────
    # The travel planner uses:
    #   - completion_client: The most capable LLM for complex reasoning
    #   - weather_a2a_agent: Remote weather agent accessible via A2A
    #   - TRAVEL_PLANNER_INSTRUCTIONS: Detailed planning persona
    #
    # ── IMPORTANT: Two ways to use an A2AAgent ──────────────────────────
    #
    # 1) DIRECT USAGE — calling the A2AAgent as the primary agent:
    #
    #       response = await weather_a2a_agent.run("What's the weather?")
    #
    #    This works because A2AAgent.run() sends the message over the
    #    A2A protocol, waits for the remote agent's response, and
    #    returns it directly. No wrapping needed.
    #    (See samples/a2a_communication/agent-client.py for an example.)
    #
    # 2) AS A TOOL FOR ANOTHER AGENT — the approach used here:
    #
    #       tools = [weather_a2a_agent.as_tool()]
    #       planner = ChatAgent(tools=tools, ...)
    #
    #    When an A2AAgent is passed as a tool to a parent ChatAgent,
    #    the LLM powering the parent agent needs to *decide when* to
    #    call it. For that, the LLM must see a function/tool definition
    #    in its API request. The OpenAI chat client only serialises
    #    FunctionTool instances into the API payload; any other
    #    ToolProtocol-compatible object (like a bare A2AAgent) is
    #    silently skipped — the LLM never knows the tool exists and
    #    cannot call it.
    #
    #    .as_tool() wraps the agent's .run() method inside a
    #    FunctionTool with a proper JSON schema definition, so the
    #    LLM can discover and invoke the weather agent on its own.
    #
    # TL;DR: Use the agent directly when it IS your agent.
    #        Use .as_tool() when it is a tool FOR another agent.
    tools = [weather_a2a_agent.as_tool()]

    async with ChatAgent(
        chat_client=completion_client,
        name="TravelPlannerAgent",
        instructions=TRAVEL_PLANNER_INSTRUCTIONS,
        tools=tools,
    ) as agent:

        # ── Create a persistent thread for multi-turn planning ──────────
        thread: AgentThread = agent.get_new_thread()

        # ── Define the test queries ─────────────────────────────────────
        # These match the "Input queries" from the README:
        #
        # Query 1: Plan a sunny European trip
        #   → Agent checks weather at several cities, picks the sunniest
        #
        # Query 2: Bad weather fallback
        #   → Agent checks London, finds bad weather, suggests alternative
        #
        # Query 3: Explain reasoning
        #   → Agent looks back at conversation history and explains its
        #     weather checks and decision process
        #
        # Query 4: Shift dates
        #   → Agent re-checks weather for shifted dates and updates plan
        test_queries = [
            "Plan a 5-day trip for me somewhere in Europe where the weather will be sunny next month.",
            "If the weather is bad in London on my dates, suggest an alternative city with better weather.",
            "Explain which locations you checked and why you chose this itinerary.",
            "What would the trip look like if I moved the dates by one week?",
        ]

        # ── Run each query ──────────────────────────────────────────────
        for i, query in enumerate(test_queries, start=1):
            print(f"{'─' * 60}")
            print(f"  Turn {i}")
            print(f"{'─' * 60}")
            print(f"  User: {query}")
            print()

            # The agent invokes the A2A weather agent as needed to check
            # weather at different locations, then plans accordingly.
            result = await agent.run(query, thread=thread)

            print(f"  Agent: {result.text}")
            print()

        # ── Summary ─────────────────────────────────────────────────────
        print("=" * 70)
        print("  All travel planning turns completed!")
        print("=" * 70)
        print()
        print("KEY TAKEAWAYS:")
        print("  • The travel planner delegated weather checks to the")
        print("    remote A2A weather agent instead of reasoning about")
        print("    weather itself — separation of concerns.")
        print("  • A2A protocol enabled agent-to-agent communication")
        print("    without sharing code or direct function calls.")
        print("  • The persistent thread gave the planner memory across")
        print("    turns for follow-up questions and plan adjustments.")
        print()


# =============================================================================
# 5. INTERACTIVE MODE
# =============================================================================
async def interactive_mode() -> None:
    """
    Run the travel planner in interactive mode for free-form queries.
    Type 'quit' or 'exit' to end the conversation.
    """

    print("=" * 70)
    print("  Scenario 3 — Interactive Travel Planner")
    print("=" * 70)
    print(f"  Connecting to A2A weather agent at: {A2A_WEATHER_HOST}")
    print("  Type your messages below. Type 'quit' or 'exit' to stop.")
    print()

    # ── Discover the weather agent ──────────────────────────────────────
    async with httpx.AsyncClient(timeout=60.0) as http_client:
        resolver = A2ACardResolver(
            httpx_client=http_client, base_url=A2A_WEATHER_HOST
        )
        agent_card = await resolver.get_agent_card()

    print(f"  Found: {agent_card.name} — {agent_card.description}")
    print()

    weather_a2a_agent = A2AAgent(
        name=agent_card.name,
        description=agent_card.description,
        agent_card=agent_card,
        url=A2A_WEATHER_HOST,
    )

    # Use .as_tool() because the A2AAgent is a tool FOR the ChatAgent,
    # not the primary agent. See the detailed explanation in main().
    async with ChatAgent(
        chat_client=completion_client,
        name="TravelPlannerAgent",
        instructions=TRAVEL_PLANNER_INSTRUCTIONS,
        tools=[weather_a2a_agent.as_tool()],
    ) as agent:

        thread = agent.get_new_thread()

        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if user_input.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break

            if not user_input:
                continue

            result = await agent.run(user_input, thread=thread)
            print(f"Agent: {result.text}")
            print()


# =============================================================================
# 6. ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Scenario 3 — Travel Planner with A2A Weather Agent"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive terminal chat mode",
    )
    args = parser.parse_args()

    if args.interactive:
        asyncio.run(interactive_mode())
    else:
        asyncio.run(main())
