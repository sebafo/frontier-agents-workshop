# =============================================================================
# Scenario 3 – A2A Weather Agent Server (SOLUTION)
# =============================================================================
#
# GOAL:
#   Expose a weather-capable agent over the A2A (Agent-to-Agent) protocol
#   so other agents (e.g. a travel planner) can invoke it as a remote
#   capability.
#
# ARCHITECTURE OVERVIEW:
#   ┌──────────────────────┐         ┌─────────────────────────┐
#   │  Travel Planner      │──A2A──▶│  This A2A Server         │
#   │  (solution-travel-   │  HTTP  │  (port 8080)             │
#   │   planner.py)        │        │                          │
#   └──────────────────────┘        │  ┌────────────────────┐  │
#                                   │  │  WeatherAgent      │  │
#                                   │  │  (ChatAgent)       │  │
#                                   │  │  + MCP weather     │  │
#                                   │  │  + local tool      │  │
#                                   │  └────────────────────┘  │
#                                   └──────────┬───────────────┘
#                                              │
#                                              ▼
#                                   ┌──────────────────┐
#                                   │ MCP: Weather     │ (Port 8001)
#                                   │ • weather        │
#                                   │ • locations      │
#                                   └──────────────────┘
#
# BEFORE RUNNING:
#   1. Make sure you have a valid .env file
#   2. Start the Weather MCP server:
#        python src/mcp-server/04-weather-server/server-mcp-sse-weather.py
#   3. Start this A2A server:
#        python src/scenarios/03-connecting-two-agents/solution-a2a-server.py
#   4. In a separate terminal, run the travel planner client:
#        python src/scenarios/03-connecting-two-agents/solution-travel-planner.py
#
# =============================================================================

# ── Standard library imports ────────────────────────────────────────────────
import sys
import os
import logging
from pathlib import Path
from random import randint
from typing import Annotated, override

# ── Ensure the project root is on sys.path ──────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# ── Third-party / framework imports ────────────────────────────────────────
from dotenv import load_dotenv
from pydantic import Field

import click
import uvicorn
from starlette.responses import JSONResponse
from starlette.routing import Route

# ── A2A SDK imports ─────────────────────────────────────────────────────────
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events import EventQueue
from a2a.server.request_handlers.default_request_handler import (
    DefaultRequestHandler,
)
from a2a.server.tasks.inmemory_task_store import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    GetTaskRequest,
    GetTaskResponse,
    SendMessageRequest,
    SendMessageResponse,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
)
from a2a.utils import new_agent_text_message, new_task, new_text_artifact

# ── Microsoft Agent Framework imports ───────────────────────────────────────
from agent_framework import ChatAgent, MCPStreamableHTTPTool

# ── Shared model client helper ──────────────────────────────────────────────
from samples.shared.model_client import create_chat_client

# ── Load environment variables ──────────────────────────────────────────────
load_dotenv()

# ── Configure logging ──────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("a2a-weather-server")


# =============================================================================
# 1. CONFIGURE THE LLM CLIENT
# =============================================================================
completion_model_name = os.environ.get("COMPLETION_DEPLOYMENT_NAME")
completion_client = create_chat_client(completion_model_name)


# =============================================================================
# 2. CONFIGURE MCP SERVER URL
# =============================================================================
WEATHER_MCP_URL = os.environ.get("WEATHER_MCP_URL", "http://localhost:8001/mcp")


# =============================================================================
# 3. LOCAL WEATHER TOOL (fallback for any location)
# =============================================================================
# This tool supplements the MCP weather server by providing a simple
# randomised weather response for any location (not just the 6 supported
# cities). The travel planner agent needs weather checks for many European
# cities, and this tool ensures we always return something useful.
def get_weather(
    location: Annotated[
        str, Field(description="The city or location to get the weather for.")
    ],
) -> str:
    """Get a weather report for a given location.

    Returns simulated weather conditions including temperature and a
    weather descriptor (sunny, cloudy, rainy, stormy). This tool works
    for any location worldwide.
    """
    conditions = ["sunny", "sunny", "cloudy", "rainy", "stormy", "sunny"]
    return (
        f"The weather in {location} is "
        f"{conditions[randint(0, len(conditions) - 1)]} "
        f"with a high of {randint(10, 30)}°C."
    )


# =============================================================================
# 4. AGENT INSTRUCTIONS
# =============================================================================
WEATHER_AGENT_INSTRUCTIONS = """
You are a weather expert agent. You answer questions about the weather at
specific locations. Use your tools to look up weather conditions.

TOOLS YOU HAVE:
- get_weather(location) — returns the current weather for any location.
- get_weather_at_location(location) — returns weather for supported locations
  (use list_supported_locations to see which cities are supported).
- list_supported_locations() — lists cities supported by the MCP weather server.

BEHAVIOUR:
- Always use tools to get real weather data — never make up weather.
- If asked about multiple locations, check each one individually.
- When asked whether weather is "good" or suitable for travel, interpret:
  - "sunny" = good for travel
  - "cloudy" = acceptable for travel
  - "rainy" = not ideal for travel
  - "stormy" = bad for travel
- Provide clear, concise answers about weather conditions.
- If asked about future dates, use the current weather as an approximation
  and mention this to the caller.
"""


# =============================================================================
# 5. AGENT CARD (A2A discovery metadata)
# =============================================================================
def weather_agent_card(url: str) -> AgentCard:
    """Define the A2A agent card for weather capability discovery."""
    skill = AgentSkill(
        id="check_weather",
        name="Check weather at locations",
        description=(
            "The agent can check weather conditions at any location and "
            "report whether the weather is good for travel. It can check "
            "multiple locations and compare conditions."
        ),
        tags=["weather", "travel", "forecast"],
        examples=[
            "What is the weather in Paris?",
            "Is the weather good for travel in Barcelona?",
            "Check the weather in London, Berlin, and Rome.",
            "Which of these cities has sunny weather: Paris, London, Tokyo?",
        ],
    )

    return AgentCard(
        name="Weather Expert Agent",
        description=(
            "A weather agent that can check weather conditions at any location "
            "and advise whether conditions are suitable for travel. Supports "
            "single and multi-location queries."
        ),
        url=url,
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(
            input_modes=["text"],
            output_modes=["text"],
            streaming=False,
        ),
        skills=[skill],
        examples=[
            "What is the weather in Amsterdam?",
            "Is the weather good for travel in Barcelona, Madrid, and Lisbon?",
        ],
    )


# =============================================================================
# 6. AGENT EXECUTOR (bridges A2A protocol ↔ ChatAgent)
# =============================================================================
class WeatherAgentExecutor(AgentExecutor):
    """
    A2A AgentExecutor that wraps a ChatAgent with weather tools.

    When an A2A message arrives, this executor:
      1. Extracts the user's query text.
      2. Passes it to the ChatAgent (which has access to MCP weather +
         local weather tools).
      3. Returns the agent's response as an A2A task artifact.
    """

    def __init__(self):
        self.tools = [
            get_weather,
            MCPStreamableHTTPTool(name="Weather Server", url=WEATHER_MCP_URL),
        ]
        self.agent = ChatAgent(
            chat_client=completion_client,
            name="WeatherExpertAgent",
            instructions=WEATHER_AGENT_INSTRUCTIONS,
            tools=self.tools,
        )
        logger.info("WeatherAgentExecutor initialized with model: %s", completion_model_name)

    @override
    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        task = context.current_task

        if not context.message:
            raise Exception("No message provided")

        if not task:
            task = new_task(context.message)
            await event_queue.enqueue_event(task)

        query = context.get_user_input()
        logger.info("Received A2A query: %s", query)

        # Use the ChatAgent with full tool support (MCP + local)
        async with self.agent:
            result = await self.agent.run(query)
            response_text = str(result.text)

        logger.info("Weather agent response: %s", response_text[:200])

        # Send the response back as an A2A task artifact
        await event_queue.enqueue_event(
            TaskArtifactUpdateEvent(
                append=False,
                context_id=task.context_id,
                task_id=task.id,
                last_chunk=True,
                artifact=new_text_artifact(
                    name="weather_result",
                    description="Weather information from the weather expert agent.",
                    text=response_text,
                ),
            )
        )
        await event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                status=TaskStatus(state=TaskState.completed),
                final=True,
                context_id=task.context_id,
                task_id=task.id,
            )
        )

    @override
    async def cancel(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        raise Exception("cancel not supported")


# =============================================================================
# 7. A2A REQUEST HANDLER
# =============================================================================
class A2ARequestHandler(DefaultRequestHandler):
    """A2A request handler that delegates to WeatherAgentExecutor."""

    def __init__(
        self, agent_executor: AgentExecutor, task_store: InMemoryTaskStore
    ):
        super().__init__(agent_executor, task_store)

    async def on_get_task(
        self, request: GetTaskRequest, *args, **kwargs
    ) -> GetTaskResponse:
        return await super().on_get_task(request, *args, **kwargs)

    async def on_message_send(
        self, request: SendMessageRequest, *args, **kwargs
    ) -> SendMessageResponse:
        return await super().on_message_send(request, *args, **kwargs)


# =============================================================================
# 8. ENTRY POINT
# =============================================================================
@click.command()
@click.option("--host", "host", default="0.0.0.0")
@click.option("--port", "port", default=8080)
def main(host: str, port: int):
    """Start the Weather Expert A2A agent server."""

    task_store = InMemoryTaskStore()
    request_handler = A2ARequestHandler(
        agent_executor=WeatherAgentExecutor(),
        task_store=task_store,
    )

    # Determine the public URL for the agent card
    if os.environ.get("CONTAINER_APP_NAME") and os.environ.get("CONTAINER_APP_ENV_DNS_SUFFIX"):
        url = f'https://{os.environ["CONTAINER_APP_NAME"]}.{os.environ["CONTAINER_APP_ENV_DNS_SUFFIX"]}'
    elif os.environ.get("A2A_AGENT_HOST"):
        url = os.environ["A2A_AGENT_HOST"]
    else:
        url = f"http://{host}:{port}/"

    server = A2AStarletteApplication(
        agent_card=weather_agent_card(url=url),
        http_handler=request_handler,
    )

    app = server.build()

    # Health check endpoint
    async def healthz(request):
        return JSONResponse({"status": "ok"})

    app.router.routes.append(Route("/_healthz", endpoint=healthz))

    print("=" * 70)
    print("  Scenario 3 — A2A Weather Agent Server")
    print("=" * 70)
    print()
    print(f"  Agent Card: {url}/.well-known/agent.json")
    print(f"  Serving on http://{host}:{port}/")
    print()
    print("  The weather agent is now discoverable via A2A.")
    print("  In another terminal, run the travel planner:")
    print("    python src/scenarios/03-connecting-two-agents/solution-travel-planner.py")
    print()
    print("  Press Ctrl+C to stop.")
    print()

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)
        sys.exit(0)
