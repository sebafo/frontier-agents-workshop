# =============================================================================
# Scenario 4 – Orchestrating Agents Using Workflows (SOLUTION)
# =============================================================================
#
# GOAL:
#   Design and implement an explicit multi-stage workflow that coordinates
#   a Weather Agent and an Activity Planner Agent to answer composite
#   user requests like "Plan my day based on the weather."
#
# WHY THIS MATTERS:
#   Real-world applications often need predictable, auditable flows that
#   coordinate many capabilities — not just a single "black box" model call.
#   By using WorkflowBuilder, each stage is explicit, testable, and
#   independently modifiable. You can trace exactly which steps ran,
#   handle errors at each stage, and swap out individual agents without
#   affecting the rest of the pipeline.
#
# ARCHITECTURE OVERVIEW:
#
#   User Query
#       │
#       ▼
#   ┌──────────────────┐
#   │  1. Input Parser  │  @executor – extracts intent, locations, date
#   └────────┬─────────┘
#            │
#            ▼
#   ┌──────────────────┐       ┌────────────────────────┐
#   │  2. Weather Agent │──────▶│  MCP: WeatherTimeSpace  │  (Port 8001)
#   │     (ChatAgent)   │       │  • get_weather_at_location│
#   │                   │       │  • list_supported_locations│
#   └────────┬─────────┘       └────────────────────────┘
#            │
#            ▼
#   ┌──────────────────────┐
#   │  3. Activity Planner  │  ChatAgent – proposes activities
#   │     (ChatAgent)       │  based on weather conditions
#   └────────┬─────────────┘
#            │
#            ▼
#   ┌──────────────────┐
#   │  4. Summarizer    │  ChatAgent – creates final day plan
#   └────────┬─────────┘
#            │
#            ▼
#   ┌──────────────────┐
#   │  5. Output        │  @executor – formats and yields result
#   └──────────────────┘
#
# BEFORE RUNNING:
#   1. Make sure you have a valid .env file (copy from .env.example)
#   2. Start the Weather MCP server:
#        python src/mcp-server/04-weather-server/server-mcp-sse-weather.py
#      (or run ./src/mcp-server/start-mcp.sh to start all MCP servers)
#   3. Run this solution:
#        python src/scenarios/04-orchestrating-agents/solution.py "Plan my day"
#
# =============================================================================

# ── Standard library imports ────────────────────────────────────────────────
import sys
import os
import json
import asyncio
import logging
from pathlib import Path
from typing import Any

# ── Ensure the project root is on sys.path ──────────────────────────────────
# This is needed so we can import from `samples.shared.model_client`.
# We go up three levels: 04-orchestrating-agents → scenarios → src → root.
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# ── Third-party imports ────────────────────────────────────────────────────
from dotenv import load_dotenv
from pydantic import BaseModel

# ── Microsoft Agent Framework imports ───────────────────────────────────────
# WorkflowBuilder:       Constructs an explicit, auditable DAG of agent
#                        and executor steps with edges that define sequencing.
# executor:              Decorator that wraps a Python async function as a
#                        workflow step, so it can be wired into the graph.
# WorkflowContext:       Shared context object passed between workflow steps.
#                        Provides shared state (key/value store), message
#                        passing, and output yielding.
# AgentExecutorRequest:  Message type sent *to* a ChatAgent step.
# AgentExecutorResponse: Message type received *from* a ChatAgent step.
# ChatMessage / Role:    Standard message types for LLM conversations.
# ChatAgent:             An LLM-backed agent that can use tools and follow
#                        system instructions.
# MCPStreamableHTTPTool: Connects to a remote MCP server over HTTP, making
#                        its tools available to the agent.
from agent_framework import (
    WorkflowBuilder,
    executor,
    WorkflowContext,
    AgentExecutorRequest,
    AgentExecutorResponse,
    ChatMessage,
    Role,
    ChatAgent,
    MCPStreamableHTTPTool,
)

# ── Shared model client helper ──────────────────────────────────────────────
from samples.shared.model_client import create_chat_client

# ── Load environment variables from .env ────────────────────────────────────
load_dotenv()


# =============================================================================
# LOGGING SETUP
# =============================================================================
# We use Python's built-in logging so every workflow step is traceable.
# This is critical for debugging multi-step workflows: you can see exactly
# which stages ran, what data flowed between them, and where errors occurred.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("scenario-04-workflow")


# =============================================================================
# 1. CONFIGURE THE LLM CLIENTS
# =============================================================================
# We use different model sizes for different workflow stages:
#   - completion_client: most capable model, used for complex reasoning
#   - medium_client: balanced model, used for structured tasks
#   - small_client: fastest model, used for simple summarisation
completion_model_name = os.environ.get("COMPLETION_DEPLOYMENT_NAME")
medium_model_name = os.environ.get("MEDIUM_DEPLOYMENT_MODEL_NAME")
small_model_name = os.environ.get("SMALL_DEPLOYMENT_MODEL_NAME")

completion_client = create_chat_client(completion_model_name)
medium_client = create_chat_client(medium_model_name)
small_client = create_chat_client(small_model_name)


# =============================================================================
# 2. STRUCTURED OUTPUT MODELS (Pydantic)
# =============================================================================
# These models enforce schema-validated JSON responses from agents.
# Using structured outputs means the workflow can safely parse and
# route data between stages without fragile string parsing.

class ParsedInput(BaseModel):
    """Structured representation of the user's parsed request.

    The Input Parser agent extracts these fields so downstream stages
    know exactly what the user wants without re-interpreting free text.
    """
    locations: list[str]       # City names the user mentioned (or defaults)
    date_context: str          # e.g. "tomorrow", "weekend", "today"
    activity_type: str         # e.g. "day plan", "outdoor activities", "trip"
    original_query: str        # The raw user message for reference


class WeatherReport(BaseModel):
    """Structured weather results for one or more locations.

    Passed from the Weather Agent to the Activity Planner so it can
    reason about conditions when suggesting activities.
    """
    reports: list[str]         # One weather string per location
    locations_checked: list[str]  # Which locations we got data for
    any_failures: bool         # True if any location lookup failed
    failure_details: str       # Description of what went wrong (if anything)


# =============================================================================
# 3. MCP WEATHER SERVER URL
# =============================================================================
# The Weather MCP server exposes tools for looking up weather by city name.
# It must be running before this workflow is started.
WEATHER_MCP_URL = os.environ.get("WEATHER_MCP_URL", "http://localhost:8001/mcp")


# =============================================================================
# 4. CREATE THE AGENTS
# =============================================================================
# Each agent has a focused role in the workflow. By keeping their
# responsibilities narrow, we make each step testable and replaceable.

# ── Weather Agent ───────────────────────────────────────────────────────────
# This agent connects to the Weather MCP server and can call its tools
# (get_weather_at_location, list_supported_locations) to retrieve
# weather data. It returns a structured WeatherReport.
weather_agent = ChatAgent(
    name="WeatherAgent",
    instructions=(
        "You are a weather specialist agent. Your ONLY job is to check "
        "weather conditions for the requested locations.\n\n"
        "TOOLS AVAILABLE:\n"
        "- get_weather_at_location(location) — get weather for a city\n"
        "- list_supported_locations() — list valid city names\n"
        "- get_weather_for_multiple_locations(locations) — batch lookup\n\n"
        "RULES:\n"
        "- IMMEDIATELY call get_weather_at_location for EACH location "
        "mentioned in the user message. Do NOT ask for confirmation. "
        "Do NOT just list supported locations — actually fetch weather data.\n"
        "- Always use your tools to get real weather data. Never invent weather.\n"
        "- If a location is not supported, call list_supported_locations and "
        "report which cities ARE available.\n"
        "- Return your findings as a JSON object with:\n"
        "  reports: list of weather description strings\n"
        "  locations_checked: list of location names you checked\n"
        "  any_failures: true/false if any location was unsupported\n"
        "  failure_details: description of failures, or empty string if none"
    ),
    response_format=WeatherReport,
    chat_client=completion_client,
    tools=[
        MCPStreamableHTTPTool(
            name="WeatherServer",
            url=WEATHER_MCP_URL,
        ),
    ],
)

# ── Activity Planner Agent ──────────────────────────────────────────────────
# This agent receives weather conditions and proposes activities that
# match those conditions. It doesn't call any tools — it reasons purely
# from the weather data and the user's preferences.
activity_planner = ChatAgent(
    name="ActivityPlanner",
    instructions=(
        "You are a creative activity planner. You receive weather conditions "
        "for one or more locations and propose activities that fit.\n\n"
        "RULES:\n"
        "- Suggest 2-3 specific activities per location.\n"
        "- Each activity should be appropriate for the weather conditions.\n"
        "  (e.g., indoor activities for rainy weather, outdoor for sunny)\n"
        "- Include practical details: what to wear, what to bring, best "
        "time of day.\n"
        "- If the weather data mentions failures or unsupported locations, "
        "acknowledge that and suggest general activities instead.\n"
        "- Be specific and actionable, not generic."
    ),
    chat_client=medium_client,
)

# ── Summarizer Agent ───────────────────────────────────────────────────────
# This agent takes all the accumulated context (user request, weather data,
# activity proposals) and creates a polished, concise final summary.
summarizer = ChatAgent(
    name="Summarizer",
    instructions=(
        "You are a concise summarizer. You receive a user's original request, "
        "weather data, and activity proposals. Your job is to create a "
        "polished, friendly day plan summary.\n\n"
        "RULES:\n"
        "- Start with a brief weather overview.\n"
        "- Present the recommended activities in a clear, numbered list.\n"
        "- Include any warnings (e.g., bring an umbrella, wear sunscreen).\n"
        "- If any workflow step encountered errors, explain which step "
        "failed and what was attempted, in a user-friendly way.\n"
        "- End with a short, encouraging closing line.\n"
        "- Keep total length under 300 words."
    ),
    chat_client=small_client,
)


# =============================================================================
# 5. WORKFLOW EXECUTOR FUNCTIONS
# =============================================================================
# Executors are custom Python functions that run as workflow steps.
# They handle logic that doesn't need an LLM: parsing, bridging data
# between agents, error handling, and formatting final output.


@executor(id="start")
async def start(message: str, ctx: WorkflowContext) -> None:
    """Entry point — parse the user query and prepare for the Weather Agent.

    This executor:
    1. Stores the original user message in shared state for later steps.
    2. Analyses the query to determine which locations and date context
       the user is asking about.
    3. If no specific location is mentioned, defaults to popular cities.
    4. Forwards a structured request to the Weather Agent.
    """
    logger.info("=" * 60)
    logger.info("WORKFLOW START — Day Planner with Weather-Aware Activities")
    logger.info("=" * 60)
    logger.info(f"User query: {message}")
    print(f"\n{'=' * 60}")
    print(f"  DAY PLANNER WORKFLOW")
    print(f"{'=' * 60}")
    print(f"User query: {message}\n")

    # Store the original message so every downstream step can access it.
    await ctx.set_shared_state("original_query", message)

    # ── Simple keyword-based location extraction ────────────────────────
    # For a production system you'd use an NLP model or structured output
    # to extract entities. Here we do a quick keyword check against the
    # cities the Weather MCP server supports.
    supported_cities = ["Seattle", "New York", "London", "Berlin", "Tokyo", "Sydney"]
    query_lower = message.lower()

    # Find any mentioned cities in the user query.
    mentioned = [city for city in supported_cities if city.lower() in query_lower]

    # If the user didn't mention a specific city, provide sensible defaults
    # so the workflow can still produce useful output.
    if not mentioned:
        mentioned = ["Berlin", "London"]
        logger.info(
            f"No specific location found in query. "
            f"Defaulting to: {', '.join(mentioned)}"
        )
        print(f"  [Step 1] No specific location detected — "
              f"defaulting to {', '.join(mentioned)}")
    else:
        logger.info(f"Extracted locations: {', '.join(mentioned)}")
        print(f"  [Step 1] Extracted locations: {', '.join(mentioned)}")

    # Determine the date context from the query.
    if "weekend" in query_lower:
        date_context = "this weekend"
    elif "tomorrow" in query_lower:
        date_context = "tomorrow"
    else:
        date_context = "today"

    logger.info(f"Date context: {date_context}")
    print(f"  [Step 1] Date context: {date_context}")

    # Store parsed data in shared state for downstream executors.
    await ctx.set_shared_state("locations", mentioned)
    await ctx.set_shared_state("date_context", date_context)

    # ── Forward to the Weather Agent ────────────────────────────────────
    # We construct a clear, actionable prompt so the Weather Agent knows
    # exactly what to look up. This avoids ambiguity and reduces LLM
    # hallucination risk.
    weather_prompt = (
        f"Check the current weather for these locations: "
        f"{', '.join(mentioned)}. "
        f"The user is planning for {date_context}."
    )

    logger.info("Forwarding to WeatherAgent...")
    print(f"  [Step 1] → Forwarding to Weather Agent\n")

    await ctx.send_message(
        AgentExecutorRequest(
            messages=[ChatMessage(role=Role.USER, text=weather_prompt)],
            should_respond=True,
        )
    )


@executor(id="bridge_weather_to_planner")
async def bridge_weather_to_planner(
    resp: AgentExecutorResponse, ctx: WorkflowContext
) -> None:
    """Bridge step — takes Weather Agent output and forwards to Activity Planner.

    This executor:
    1. Reads the structured weather report from the Weather Agent.
    2. Stores it in shared state for traceability.
    3. Handles cases where the weather lookup partially or fully failed.
    4. Constructs a rich prompt for the Activity Planner that includes
       both weather data and the user's original intent.
    """
    logger.info("-" * 40)
    logger.info("BRIDGE: Weather → Activity Planner")

    # ── Parse the Weather Agent's response ──────────────────────────────
    weather_text = resp.agent_response.text
    logger.info(f"Weather Agent response: {weather_text[:200]}...")

    # Try to parse as structured JSON (WeatherReport). If the agent
    # returned free-form text instead, we still handle it gracefully.
    try:
        weather_data = json.loads(weather_text)
        any_failures = weather_data.get("any_failures", False)
        failure_details = weather_data.get("failure_details", "")
        reports = weather_data.get("reports", [])

        if any_failures:
            logger.warning(f"Weather lookup had failures: {failure_details}")
            print(f"  [Step 2] ⚠ Weather lookup issue: {failure_details}")
        else:
            print(f"  [Step 2] ✓ Weather data received for "
                  f"{len(reports)} location(s)")
    except (json.JSONDecodeError, KeyError) as e:
        # If parsing fails, treat the entire response as a text report.
        # The Activity Planner can still work with unstructured weather text.
        logger.warning(f"Could not parse structured weather response: {e}")
        print(f"  [Step 2] ⚠ Weather response was unstructured, proceeding anyway")
        any_failures = False
        failure_details = ""

    # Store weather data for the Summarizer to reference later.
    await ctx.set_shared_state("weather_response", weather_text)

    # ── Retrieve context from shared state ──────────────────────────────
    original_query = await ctx.get_shared_state("original_query")
    locations = await ctx.get_shared_state("locations")
    date_context = await ctx.get_shared_state("date_context")

    # ── Build the Activity Planner prompt ───────────────────────────────
    # We include all relevant context so the planner can make informed
    # suggestions without needing to call any tools itself.
    planner_prompt = (
        f"The user asked: \"{original_query}\"\n"
        f"Time frame: {date_context}\n"
        f"Locations: {', '.join(locations)}\n\n"
        f"Here is the current weather data:\n{weather_text}\n\n"
        f"Based on these weather conditions, propose 2-3 specific activities "
        f"per location that are appropriate for the conditions. "
        f"Include practical tips (what to wear, what to bring)."
    )

    if any_failures and failure_details:
        planner_prompt += (
            f"\n\nNote: Some weather lookups had issues: {failure_details}. "
            f"For those locations, suggest general indoor/flexible activities."
        )

    logger.info("Forwarding to ActivityPlanner...")
    print(f"  [Step 2] → Forwarding to Activity Planner\n")

    await ctx.send_message(
        AgentExecutorRequest(
            messages=[ChatMessage(role=Role.USER, text=planner_prompt)],
            should_respond=True,
        )
    )


@executor(id="bridge_planner_to_summarizer")
async def bridge_planner_to_summarizer(
    resp: AgentExecutorResponse, ctx: WorkflowContext
) -> None:
    """Bridge step — takes Activity Planner output and forwards to Summarizer.

    This executor aggregates all accumulated context (original query,
    weather data, activity proposals) into a single prompt for the
    Summarizer to create a polished final output.
    """
    logger.info("-" * 40)
    logger.info("BRIDGE: Activity Planner → Summarizer")

    activity_proposals = resp.agent_response.text
    logger.info(f"Activity Planner response: {activity_proposals[:200]}...")
    print(f"  [Step 3] ✓ Activity proposals received")

    # Store activity proposals for traceability.
    await ctx.set_shared_state("activity_proposals", activity_proposals)

    # ── Retrieve all accumulated context ────────────────────────────────
    original_query = await ctx.get_shared_state("original_query")
    weather_response = await ctx.get_shared_state("weather_response")
    date_context = await ctx.get_shared_state("date_context")

    # ── Build the Summarizer prompt ─────────────────────────────────────
    # The summarizer gets the full picture so it can create a coherent
    # day plan that ties everything together.
    summary_prompt = (
        f"Create a polished day plan summary based on the following:\n\n"
        f"ORIGINAL USER REQUEST:\n{original_query}\n\n"
        f"TIME FRAME: {date_context}\n\n"
        f"WEATHER CONDITIONS:\n{weather_response}\n\n"
        f"PROPOSED ACTIVITIES:\n{activity_proposals}\n\n"
        f"Combine this into a friendly, concise day plan. "
        f"Start with weather overview, list the recommended activities, "
        f"and end with a short encouraging note."
    )

    logger.info("Forwarding to Summarizer...")
    print(f"  [Step 3] → Forwarding to Summarizer\n")

    await ctx.send_message(
        AgentExecutorRequest(
            messages=[ChatMessage(role=Role.USER, text=summary_prompt)],
            should_respond=True,
        )
    )


@executor(id="output")
async def output(resp: AgentExecutorResponse, ctx: WorkflowContext) -> None:
    """Final step — display the Summarizer's polished day plan.

    This executor:
    1. Prints a workflow trace showing which steps executed successfully.
    2. Displays the final day plan to the user.
    3. Yields the output so callers of workflow.run() can capture it.
    """
    logger.info("-" * 40)
    logger.info("OUTPUT: Final day plan")

    final_plan = resp.agent_response.text

    # ── Print workflow trace ────────────────────────────────────────────
    # This trace is invaluable for debugging: you can see at a glance
    # which steps ran and whether any had issues.
    print(f"\n{'=' * 60}")
    print(f"  WORKFLOW TRACE")
    print(f"{'=' * 60}")
    print(f"  [Step 1] Input Parser      — ✓ completed")
    print(f"  [Step 2] Weather Agent      — ✓ completed")
    print(f"  [Step 3] Activity Planner   — ✓ completed")
    print(f"  [Step 4] Summarizer         — ✓ completed")
    print(f"{'=' * 60}\n")

    # ── Display the final result ────────────────────────────────────────
    print(f"{'=' * 60}")
    print(f"  YOUR DAY PLAN")
    print(f"{'=' * 60}")
    print(f"\n{final_plan}\n")
    print(f"{'=' * 60}\n")

    logger.info("Workflow completed successfully")

    # Yield the final output so it can be captured programmatically.
    await ctx.yield_output(final_plan)


# =============================================================================
# 6. BUILD THE WORKFLOW
# =============================================================================
# The WorkflowBuilder wires together all the steps into an explicit DAG
# (directed acyclic graph). Each add_edge() call defines a dependency:
# the source step must complete before the target step runs.
#
# Workflow graph:
#   start → weather_agent → bridge_weather_to_planner
#         → activity_planner → bridge_planner_to_summarizer
#         → summarizer → output
#
# This makes the flow:
#   1. Parse input (start executor)
#   2. Check weather (weather_agent ChatAgent with MCP tools)
#   3. Bridge weather data (bridge_weather_to_planner executor)
#   4. Plan activities (activity_planner ChatAgent)
#   5. Bridge activity data (bridge_planner_to_summarizer executor)
#   6. Summarize day plan (summarizer ChatAgent)
#   7. Output final result (output executor)
workflow = (
    WorkflowBuilder()
    .set_start_executor(start)
    # Step 1 → Step 2: Parse input, then check weather
    .add_edge(start, weather_agent)
    # Step 2 → Step 3: Weather results flow through the bridge
    .add_edge(weather_agent, bridge_weather_to_planner)
    # Step 3 → Step 4: Bridge forwards to the Activity Planner
    .add_edge(bridge_weather_to_planner, activity_planner)
    # Step 4 → Step 5: Activity proposals flow through the bridge
    .add_edge(activity_planner, bridge_planner_to_summarizer)
    # Step 5 → Step 6: Bridge forwards to the Summarizer
    .add_edge(bridge_planner_to_summarizer, summarizer)
    # Step 6 → Step 7: Summarizer output goes to final output
    .add_edge(summarizer, output)
    .build()
)


# =============================================================================
# 7. MAIN ENTRY POINT
# =============================================================================
async def main():
    """Run the Day Planner workflow.

    Usage:
        python solution.py "Plan my day tomorrow"
        python solution.py "Check the weather for the weekend and propose activities"
        python solution.py --interactive    (interactive mode)
        python solution.py --devui          (launch DevUI web interface)
    """
    # ── Handle --interactive mode ───────────────────────────────────────
    # Interactive mode lets you type queries one at a time and see
    # the full workflow execute for each one.
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        print("\n" + "=" * 60)
        print("  Day Planner Workflow — Interactive Mode")
        print("=" * 60)
        print("Type a request and press Enter. Type 'quit' or ':q' to exit.\n")
        print("Example queries:")
        print('  "Plan my day tomorrow including weather in Berlin"')
        print('  "Check the weather for the weekend and propose activities"')
        print('  "What should I do in Tokyo today?"')
        print()

        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if user_input.lower() in ("quit", ":q", "exit"):
                print("Goodbye!")
                break

            if not user_input:
                continue

            try:
                await workflow.run(user_input)
            except Exception as e:
                # ── Error handling ──────────────────────────────────────
                # If any workflow step fails, we catch the exception here
                # and report which step likely failed, based on the error.
                logger.error(f"Workflow error: {e}", exc_info=True)
                print(f"\n{'=' * 60}")
                print(f"  WORKFLOW ERROR")
                print(f"{'=' * 60}")
                print(f"  An error occurred during the workflow:")
                print(f"  {e}")
                print(f"\n  Troubleshooting tips:")
                print(f"  - Is the Weather MCP server running on port 8001?")
                print(f"  - Check your .env file for valid API credentials.")
                print(f"  - Review the logs above to see which step failed.")
                print(f"{'=' * 60}\n")

        return

    # ── Handle command-line argument mode ───────────────────────────────
    # If a message is passed as a CLI argument, run it once and exit.
    if len(sys.argv) > 1:
        message = " ".join(sys.argv[1:])
    else:
        # Default test query if nothing is provided.
        message = (
            "Plan my day tomorrow including where I should go "
            "based on the weather and give me a short summary."
        )
        print(f"No query provided. Using default: \"{message}\"\n")

    try:
        await workflow.run(message)
    except Exception as e:
        # ── Top-level error handler ────────────────────────────────────
        # Catches failures from any workflow step and provides a
        # user-friendly error report indicating which step failed.
        logger.error(f"Workflow error: {e}", exc_info=True)
        print(f"\n{'=' * 60}")
        print(f"  WORKFLOW ERROR")
        print(f"{'=' * 60}")
        print(f"  The workflow encountered an error:")
        print(f"  {e}")
        print()
        print(f"  This likely means one of these steps failed:")
        print(f"  1. Input parsing — check if the query is valid")
        print(f"  2. Weather lookup — is the MCP server running on port 8001?")
        print(f"  3. Activity planning — check LLM API credentials")
        print(f"  4. Summarization — check LLM API credentials")
        print(f"\n  Review the log output above to identify the exact step.")
        print(f"{'=' * 60}\n")


def _suppress_mcp_cleanup_errors():
    """Suppress the harmless MCP streamable_http_client cleanup error.

    The MCP client's async generator teardown can raise a RuntimeError
    ("Attempted to exit cancel scope in a different task") when the
    event loop shuts down. This is cosmetic — the workflow has already
    completed successfully — but it clutters the output. We install a
    custom exception handler on the event loop to silence it, and also
    filter the asyncio logger.
    """
    import asyncio

    def _handler(loop, context):
        exc = context.get("exception")
        msg = context.get("message", "")
        # Suppress only the known MCP cleanup error
        if "asynchronous generator" in msg or (
            exc and "cancel scope" in str(exc)
        ):
            return
        # Fall through to default handler for everything else
        loop.default_exception_handler(context)

    try:
        loop = asyncio.get_event_loop()
        loop.set_exception_handler(_handler)
    except RuntimeError:
        pass

    # Also suppress the asyncio logger's ERROR for this specific issue
    class _MCPCleanupFilter(logging.Filter):
        def filter(self, record):
            return "asynchronous generator" not in record.getMessage()

    logging.getLogger("asyncio").addFilter(_MCPCleanupFilter())


if __name__ == "__main__":
    # Handle --devui before entering asyncio.run(), because serve()
    # starts its own event loop internally.
    if len(sys.argv) > 1 and sys.argv[1] == "--devui":
        from agent_framework.devui import serve

        logger.info("Starting DevUI at http://localhost:8094")
        print("\nStarting DevUI at http://localhost:8094")
        print("Open the URL in your browser to interact with the workflow.\n")
        serve(entities=[workflow], port=8094, auto_open=True)
    else:
        _suppress_mcp_cleanup_errors()
        asyncio.run(main())
