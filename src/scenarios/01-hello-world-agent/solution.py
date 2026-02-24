# =============================================================================
# Scenario 1 – Your First Time & Weather Agent (SOLUTION)
# =============================================================================
#
# GOAL:
#   Build an agent using the Microsoft Agent Framework that can:
#     1. Remember the user's location from conversation history
#     2. Tell the user what time it is at their location
#     3. Provide weather information for the user's current location
#     4. Handle location changes mid-conversation
#     5. Recall previously mentioned information without asking again
#
# WHY THIS MATTERS:
#   Most real-world agents combine memory, tools, and external services
#   rather than answering a single prompt. This scenario teaches the
#   fundamental patterns: tool-calling, thread-based memory, and
#   connecting to external MCP servers.
#
# ARCHITECTURE OVERVIEW:
#   ┌──────────────┐       ┌────────────────────────┐
#   │  ChatAgent   │──────▶│  MCP: UserTimeLocation  │  (Port 8002)
#   │  (this file) │       │  • get_current_user      │
#   │              │       │  • get_current_location   │
#   │              │       │  • get_current_time       │
#   │              │       │  • move                   │
#   │              │       └────────────────────────┘
#   │              │
#   │              │       ┌────────────────────────┐
#   │              │──────▶│  MCP: WeatherTimeSpace  │  (Port 8001)
#   │              │       │  • get_weather_at_location│
#   │              │       │  • list_supported_locations│
#   │              │       └────────────────────────┘
#   │              │
#   │              │──────▶ Local tool: get_current_time_for_location()
#   └──────────────┘
#
# BEFORE RUNNING:
#   1. Make sure you have a valid .env file (copy from .env.example)
#   2. Start the User MCP server:
#        python src/mcp-server/02-user-server/server-mcp-sse-user.py
#   3. Start the Weather MCP server:
#        python src/mcp-server/04-weather-server/server-mcp-sse-weather.py
#   4. Run this solution:
#        python src/scenarios/01-hello-world-agent/solution.py
#
# =============================================================================

# ── Standard library imports ────────────────────────────────────────────────
import sys
from pathlib import Path
import os
import asyncio
from datetime import datetime
from typing import Annotated

# ── Ensure the project root is on sys.path ──────────────────────────────────
# This is necessary so we can import from `samples.shared.model_client`,
# which lives at the top level of the repository. We go up three levels
# from this file: 01-hello-world-agent → scenarios → src → project root.
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# ── Third-party / framework imports ────────────────────────────────────────
# Pydantic's Field is used to annotate tool parameters so the LLM knows
# what each argument means (description, type, etc.).
from pydantic import Field

# python-dotenv loads environment variables from a .env file into
# os.environ so we don't have to hard-code secrets.
from dotenv import load_dotenv

# pytz provides accurate timezone handling, which we need to convert
# a timezone name like "Europe/London" into a real local time.
import pytz

# ── Microsoft Agent Framework imports ───────────────────────────────────────
# ChatAgent:           The main agent class that wraps an LLM client with
#                      instructions, tools, and memory (threads).
# AgentThread:         Represents a conversation thread that stores message
#                      history so the agent can remember previous turns.
# MCPStreamableHTTPTool: Connects to a remote MCP server over HTTP using the
#                        Streamable HTTP transport. This lets us expose tools
#                        defined on separate servers as if they were local.
from agent_framework import ChatAgent, AgentThread, MCPStreamableHTTPTool

# ── Import the shared helper that creates an LLM chat client ───────────────
# `create_chat_client` reads environment variables (GITHUB_TOKEN or
# AZURE_OPENAI_ENDPOINT + key) and returns either an OpenAIChatClient
# or an AzureOpenAIChatClient depending on what is configured.
from samples.shared.model_client import create_chat_client

# ── Load environment variables from .env ────────────────────────────────────
# This call reads .env in the current working directory (or any parent)
# and sets the values as environment variables.
load_dotenv()


# =============================================================================
# 1. CONFIGURE THE LLM CLIENT
# =============================================================================
# We read the model deployment name from the environment.
# The .env.example ships these defaults:
#   COMPLETION_DEPLOYMENT_NAME = "openai/gpt-5-nano"
#   MEDIUM_DEPLOYMENT_MODEL_NAME = "openai/gpt-4.1-mini"
#   SMALL_DEPLOYMENT_MODEL_NAME = "openai/gpt-4.1-nano"
#
# For this scenario a small or medium model is sufficient because the
# tasks are straightforward tool-calling and summarisation.
completion_model_name = os.environ.get("COMPLETION_DEPLOYMENT_NAME")
medium_model_name = os.environ.get("MEDIUM_DEPLOYMENT_MODEL_NAME")
small_model_name = os.environ.get("SMALL_DEPLOYMENT_MODEL_NAME")

# Create chat client instances.
# We will use the `completion_client` (most capable model) for the main
# agent so it can reliably reason over multiple tools and conversation
# history. You could also try `medium_client` or `small_client` for
# faster / cheaper inference if the tasks are simple enough.
completion_client = create_chat_client(completion_model_name)
medium_client = create_chat_client(medium_model_name)
small_client = create_chat_client(small_model_name)


# =============================================================================
# 2. DEFINE A LOCAL TOOL: get_current_time_for_location
# =============================================================================
# This is a *local* Python function that the agent can call directly
# (without going through an MCP server). It demonstrates how to expose
# a simple function as a tool.
#
# HOW TOOL CALLING WORKS:
#   1. The LLM sees the function signature and docstring as part of its
#      system prompt (the framework serialises them automatically).
#   2. When the LLM decides it needs this tool, it emits a function_call
#      with the argument values (e.g., location="Europe/London").
#   3. The framework intercepts the function_call, executes the Python
#      function locally, and feeds the return value back to the LLM.
#   4. The LLM then uses the result to formulate its final answer.
#
# WHY Annotated + Field?
#   The `Annotated[str, Field(description=...)]` pattern lets us attach
#   metadata to function parameters. The Agent Framework reads these
#   annotations and tells the LLM what each parameter means, improving
#   the accuracy of tool calls.
def get_current_time_for_location(
    location: Annotated[
        str,
        Field(
            description=(
                "The IANA timezone name for the location, "
                "e.g. 'Europe/London', 'America/New_York', 'Asia/Tokyo'. "
                "Anything in Germany should use 'Europe/Berlin'."
            )
        ),
    ],
) -> str:
    """
    Get the current local time for the given timezone location.

    This tool accepts an IANA timezone string (e.g. 'Europe/Berlin')
    and returns a human-readable local time string.

    Returns:
        A string like "The current time in Europe/Berlin is 02:34:15 PM".
        If the timezone is invalid, returns an error message.
    """
    try:
        # ── Sanitise the input ──────────────────────────────────────────
        # LLMs sometimes add stray whitespace, quotes, or newlines to
        # arguments. We strip them to avoid pytz lookup failures.
        location = location.strip().replace('"', "").replace("\n", "")

        # ── Resolve the timezone ────────────────────────────────────────
        # pytz.timezone() takes an IANA timezone string and returns a
        # timezone object. If the string is invalid it raises
        # pytz.exceptions.UnknownTimeZoneError.
        timezone = pytz.timezone(location)

        # ── Get the current time in that timezone ───────────────────────
        # datetime.now(tz) returns the current UTC time converted to the
        # given timezone. strftime formats it in a 12-hour clock display.
        now = datetime.now(timezone)
        current_time = now.strftime("%I:%M:%S %p")

        return f"The current time in {location} is {current_time}."
    except Exception as e:
        # If something goes wrong (e.g. unknown timezone), return a
        # user-friendly error so the LLM can explain the issue.
        return f"Sorry, I couldn't find the timezone for '{location}': {e}"


# =============================================================================
# 3. CONFIGURE MCP SERVER URLS
# =============================================================================
# MCP (Model Context Protocol) servers expose tools, resources, and prompts
# over HTTP. Our agent connects to them using MCPStreamableHTTPTool, which
# discovers available tools at startup and makes them callable by the LLM.
#
# The URLs default to localhost because we expect the servers to be running
# locally during the workshop. In production you would point to deployed
# endpoints.

# The User MCP server provides:
#   • get_current_user()        → returns the logged-in username
#   • get_current_location(user) → returns the user's timezone
#   • get_current_time(location) → returns the current time
#   • move(user, newlocation)   → updates the user's location
USER_MCP_URL = os.environ.get("USER_MCP_URL", "http://localhost:8002/mcp")

# The Weather MCP server provides:
#   • list_supported_locations()          → returns the 6 supported cities
#   • get_weather_at_location(location)   → returns time-of-day weather
#   • get_weather_for_multiple_locations() → batch weather lookup
WEATHER_MCP_URL = os.environ.get("WEATHER_MCP_URL", "http://localhost:8001/mcp")


# =============================================================================
# 4. DEFINE THE AGENT SYSTEM PROMPT (INSTRUCTIONS)
# =============================================================================
# The system prompt tells the LLM what persona it should adopt and how it
# should behave. Good instructions are crucial for agent quality.
#
# KEY DESIGN CHOICES IN THIS PROMPT:
#   • We tell the agent it has access to three categories of tools so it
#     knows what actions are available.
#   • We instruct it to reuse information from conversation history
#     ("remember the user's location") so it doesn't ask again.
#   • We tell it to proactively use tools rather than guessing, which
#     prevents hallucinated weather or time data.
#   • We give it a friendly but concise personality.
AGENT_INSTRUCTIONS = """
You are a friendly assistant for time and weather queries.

LOCATION MEMORY:
- You remember the user's location from the conversation history.
  When the user says "I am in London", remember "London" / "Europe/London".
  When they say "I moved to Berlin", update to "Berlin" / "Europe/Berlin".
  You do NOT need to call any tool to remember this — the conversation
  history IS your memory.
- Do NOT ask the user to repeat their location if they already told you.

ANSWERING QUESTIONS:
- For time queries: call get_current_time_for_location with the IANA
  timezone (e.g. "Europe/London") from what the user already told you.
- For weather queries: call get_weather_at_location with the city name
  the user mentioned.
- Always use tools for real data — never make up times or weather.
- When reporting weather, include the local time too.
- If a location is not supported for weather, call
  list_supported_locations to tell the user which ones are available.
- Be concise and friendly.

CRITICAL — INTERPRETING TOOL RESULTS:
- Tool results are ALWAYS successful when they return data.
- Results may be wrapped in JSON like [{"type":"text","text":"..."}].
  The actual value is in the "text" field. Treat this as a SUCCESS.
- For example, if get_weather_at_location returns
  [{"type":"text","text":"Weather for London: Cool and clear"}],
  that means the weather IS "Cool and clear". Report it to the user.
- NEVER say "there was an error" or "I had trouble" when a tool
  returned data. If you received text back from a tool, it worked.
"""


# =============================================================================
# 5. MAIN FUNCTION: Run the interactive agent conversation
# =============================================================================
async def main() -> None:
    """
    Main entry point. Sets up the agent with:
      - An LLM client (configured via .env)
      - System instructions (the persona prompt above)
      - A local time tool (get_current_time_for_location)
      - Two remote MCP tool sources (user server + weather server)
      - A persistent conversation thread for multi-turn memory

    Then runs a series of test queries that exercise all the scenario
    requirements:
      1. User declares their location → agent remembers it.
      2. User asks about weather → agent uses the remembered location.
      3. User asks about time → agent uses the remembered location.
      4. User moves to a new city → agent updates and answers.
      5. User asks where they are → agent recalls from memory.
    """

    print("=" * 70)
    print("  Scenario 1 — Your First Time & Weather Agent")
    print("=" * 70)
    print()

    # ── Build the list of tools ─────────────────────────────────────────
    # We combine:
    #   (a) A local Python function (get_current_time_for_location)
    #   (b) An MCP tool pointing at the User server
    #   (c) An MCP tool pointing at the Weather server
    #
    # The framework will discover all tools from the MCP servers at
    # startup and merge them with the local tool. The LLM will see one
    # flat list of available functions.
    tools = [
        # ── Local tool ──────────────────────────────────────────────────
        # This Python function is directly callable by the agent.
        # It uses pytz to compute the current time for a given timezone.
        get_current_time_for_location,

        # ── Remote MCP tool: User server ────────────────────────────────
        # MCPStreamableHTTPTool wraps an MCP server. At agent startup it
        # issues a tools/list request to the server and registers each
        # returned tool so the LLM can call them.
        # The `name` is a human-readable label for logging / DevUI.
        MCPStreamableHTTPTool(
            name="User Server",
            url=USER_MCP_URL,
        ),

        # ── Remote MCP tool: Weather server ─────────────────────────────
        MCPStreamableHTTPTool(
            name="Weather Server",
            url=WEATHER_MCP_URL,
        ),
    ]

    # ── Create the agent ────────────────────────────────────────────────
    # `async with` is required because MCPStreamableHTTPTool holds HTTP
    # connections to the MCP servers that need to be cleaned up on exit.
    # The context manager ensures proper lifecycle management.
    async with ChatAgent(
        # The LLM client that will generate completions.
        chat_client=completion_client,

        # A human-readable name for this agent (shown in DevUI traces).
        name="TimeWeatherAgent",

        # The system prompt that defines the agent's behaviour.
        instructions=AGENT_INSTRUCTIONS,

        # The list of tools (local functions + MCP servers) the agent
        # can invoke during its reasoning loop.
        tools=tools,
    ) as agent:

        # ── Create a persistent conversation thread ─────────────────────
        # AgentThread stores the full message history (user messages,
        # assistant responses, tool calls, and tool results). By passing
        # the same thread to every `agent.run()` call, we give the agent
        # access to all previous turns — this is how it "remembers"
        # things like the user's location.
        #
        # WITHOUT a thread, each call would start fresh and the agent
        # would have no memory of prior interactions.
        thread: AgentThread = agent.get_new_thread()

        # ── Define the test queries ─────────────────────────────────────
        # These queries match the "Input queries" section in README.md.
        # Each one tests a different aspect of the agent:
        #
        # Query 1: "I am currently in London"
        #   → The agent should store "London" as the user's location.
        #     It should NOT call any weather/time tools yet — just
        #     acknowledge the information.
        #
        # Query 2: "What is the weather now here?"
        #   → The agent should recall that the user is in London
        #     (from the thread history) and call the weather MCP tool
        #     for London. It should NOT ask "where are you?".
        #
        # Query 3: "What time is it for me right now?"
        #   → The agent should recall London, map it to "Europe/London",
        #     and call get_current_time_for_location (or the MCP time
        #     tool) to return the actual local time.
        #
        # Query 4: "I moved to Berlin, what is the weather like today?"
        #   → The agent should update the user's location to Berlin,
        #     then immediately query weather for Berlin.
        #
        # Query 5: "Can you remind me where I said I am based?"
        #   → The agent should look back through conversation history
        #     and recall "Berlin" (the most recent location).
        test_queries = [
            "I am currently in London and what is the weather now here?",
            "What time is it for me right now?",
            "I moved to Berlin, what is the weather like today?",
            "Can you remind me where I said I am based?",
        ]

        # ── Run each query in sequence ──────────────────────────────────
        # We iterate through the test queries and send each one to the
        # agent. Because we pass the same `thread` every time, the
        # agent accumulates conversation history.
        for i, query in enumerate(test_queries, start=1):
            print(f"{'─' * 60}")
            print(f"  Turn {i}")
            print(f"{'─' * 60}")
            print(f"  User: {query}")

            # ── Call the agent ──────────────────────────────────────────
            # `agent.run()` does the following internally:
            #   1. Appends the user message to the thread's message store.
            #   2. Sends the full conversation history + system prompt +
            #      tool definitions to the LLM.
            #   3. If the LLM returns a tool call, the framework executes
            #      the tool (locally or via MCP HTTP) and feeds the result
            #      back to the LLM.
            #   4. Steps 2-3 repeat until the LLM produces a final text
            #      response (no more tool calls).
            #   5. The assistant's response is appended to the thread.
            #   6. Returns an AgentResponse with .text (the final answer).
            #
            # `thread=thread` ensures we use the persistent thread.
            result = await agent.run(query, thread=thread)

            # ── Print the agent's response ──────────────────────────────
            print(f"  Agent: {result.text}")
            print()

        # ── Final summary ───────────────────────────────────────────────
        print("=" * 70)
        print("  All turns completed!")
        print("=" * 70)
        print()
        print("KEY TAKEAWAYS:")
        print("  • The agent remembered 'London' and later 'Berlin'")
        print("    across turns thanks to the persistent AgentThread.")
        print("  • Weather data came from the Weather MCP server (port 8001).")
        print("  • Time data came from the local get_current_time_for_location tool.")
        print("  • The agent combined multiple tool results into coherent answers.")
        print()
        print("NEXT STEPS:")
        print("  • Start the Agent Framework Dev UI to inspect traces:")
        print("      pip install agent-framework-devui")
        print("      agent-framework-devui")
        print("  • Try adding more queries or changing the instructions.")
        print("  • Experiment with different model sizes (small / medium / large).")


# =============================================================================
# 6. INTERACTIVE MODE (OPTIONAL)
# =============================================================================
# In addition to the scripted test queries above, we provide an interactive
# chat loop so you can experiment freely with the agent.
async def interactive_mode() -> None:
    """
    Run the agent in interactive mode where you can type your own queries.

    This is useful for experimentation beyond the scripted test queries.
    Type 'quit' or 'exit' to end the conversation.
    """

    print("=" * 70)
    print("  Scenario 1 — Interactive Mode")
    print("=" * 70)
    print("  Type your messages below. Type 'quit' or 'exit' to stop.")
    print()

    # ── Set up tools (same as in main) ──────────────────────────────────
    tools = [
        get_current_time_for_location,
        MCPStreamableHTTPTool(name="User Server", url=USER_MCP_URL),
        MCPStreamableHTTPTool(name="Weather Server", url=WEATHER_MCP_URL),
    ]

    # ── Create the agent inside an async context manager ────────────────
    async with ChatAgent(
        chat_client=completion_client,
        name="TimeWeatherAgent",
        instructions=AGENT_INSTRUCTIONS,
        tools=tools,
    ) as agent:

        # ── Create a thread for the entire interactive session ──────────
        # All user inputs and agent responses accumulate here, giving
        # the agent persistent memory for the whole session.
        thread = agent.get_new_thread()

        # ── Read-eval-print loop ────────────────────────────────────────
        while True:
            # Read user input from the console.
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                # Handle Ctrl+C or piped input gracefully.
                print("\nGoodbye!")
                break

            # Check for exit commands.
            if user_input.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break

            # Skip empty inputs.
            if not user_input:
                continue

            # ── Send the message to the agent ───────────────────────────
            # Same pattern as the scripted mode: pass the thread for
            result = await agent.run(user_input, thread=thread)

            # ── Display the response ────────────────────────────────────
            print(f"Agent: {result.text}")
            print()


# =============================================================================
# 7. ENTRY POINT
# =============================================================================
# When this script is executed directly, we run the scripted test queries.
# To use interactive mode instead, change `main()` to `interactive_mode()`.
if __name__ == "__main__":
    # ── Choose which mode to run ────────────────────────────────────────
    # Uncomment the line you want:
    #   asyncio.run(main())              # Scripted test queries
    # asyncio.run(interactive_mode())  # Free-form interactive chat

    # Default: run the scripted test queries that match the scenario spec.
    asyncio.run(main())
