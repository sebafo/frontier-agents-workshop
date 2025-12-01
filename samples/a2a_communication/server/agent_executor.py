import os
from random import randint
from typing import Annotated, override

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import (
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
)
from a2a.utils import new_agent_text_message, new_task, new_text_artifact
from agent_framework.openai import OpenAIChatClient
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import Field


load_dotenv()


def get_weather(
    location: Annotated[str, Field(description="The location to get the weather for.")],
) -> str:
    """Simple weather tool returning fake conditions for a location."""
    conditions = ["sunny", "cloudy", "rainy", "stormy"]
    return f"The weather in {location} is {conditions[randint(0, 3)]} with a high of {randint(10, 30)}°C."


def _create_openai_client() -> OpenAIChatClient:
    """Create an OpenAIChatClient similar to samples/simple-agents/basic-agent.py."""

    token: str
    endpoint: str
    model_name: str

    if os.environ.get("GITHUB_TOKEN") is not None:
        token = os.environ["GITHUB_TOKEN"]
        endpoint = "https://models.github.ai/inference"
        model_name = os.environ["COMPLETION_DEPLOYMENT_NAME"]
    elif os.environ.get("AZURE_OPENAI_API_KEY") is not None:
        token = os.environ["AZURE_OPENAI_API_KEY"]
        endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
        model_name = os.environ["COMPLETION_DEPLOYMENT_NAME"]
    else:
        raise RuntimeError(
            "No OpenAI credentials found. Set GITHUB_TOKEN or AZURE_OPENAI_API_KEY."
        )

    async_openai_client = AsyncOpenAI(
        base_url=endpoint,
        api_key=token,
    )

    return OpenAIChatClient(
        model_id=model_name,
        api_key=token,
        async_client=async_openai_client,
    )


class HelloWorldAgentExecutor(AgentExecutor):
    """Simple weather Q&A agent using Microsoft agent framework."""

    def __init__(self):
        # Reuse the same authentication logic as the basic agent sample
        self.agent = _create_openai_client()

    @override
    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        task = context.current_task

        if not context.message:
            raise Exception('No message provided')

        if not task:
            task = new_task(context.message)
            await event_queue.enqueue_event(task)

        query = "You are weather expert and know everything about weather. Try to help the user with their input using the tools you have available. " + context.get_user_input()
        # Use the Microsoft agent framework chat client with the weather tool.
        # We do a single-turn completion and treat the result as the final task artifact.
        query = context.get_user_input()

        # get_response may return a rich object; coerce to string for A2A
        response = await self.agent.get_response(query, tools=get_weather)

        # Ensure the artifact text is always a plain string
        response_text = str(response)

        await event_queue.enqueue_event(
            TaskArtifactUpdateEvent(
                append=False,
                context_id=task.context_id,
                task_id=task.id,
                last_chunk=True,
                artifact=new_text_artifact(
                    name='current_result',
                    description='Result of request to weather agent.',
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
        raise Exception('cancel not supported')