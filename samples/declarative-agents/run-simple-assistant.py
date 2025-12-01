# Copyright (c) Microsoft. All rights reserved.
import os
import asyncio
from pathlib import Path
from random import randint
from typing import Literal

from agent_framework.openai import OpenAIResponsesClient
from agent_framework_declarative import AgentFactory

from openai import AsyncOpenAI

from dotenv import load_dotenv

load_dotenv()

def get_weather(location: str, unit: Literal["celsius", "fahrenheit"] = "celsius") -> str:
    """A simple function tool to get weather information."""
    return f"The weather in {location} is {randint(-10, 30) if unit == 'celsius' else randint(30, 100)} degrees {unit}."


async def main():
    """Create an agent from a declarative yaml specification and run it."""
    # get the path
    current_path = Path(__file__).parent
    yaml_path = current_path / "weather-assistant.yaml"

    # load the yaml from the path
    with yaml_path.open("r") as f:
        yaml_str = f.read()

    if (os.environ.get("GITHUB_TOKEN") is not None):
        token = os.environ["GITHUB_TOKEN"]
        endpoint = "https://models.github.ai/inference"
        model_name = os.environ.get("SMALL_DEPLOYMENT_MODEL_NAME")
        print("Using GitHub Token for authentication")
    elif (os.environ.get("AZURE_OPENAI_API_KEY") is not None):
        token = os.environ["AZURE_OPENAI_API_KEY"]
        endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
        model_name = os.environ.get("SMALL_DEPLOYMENT_MODEL_NAME")
        print("Using Azure OpenAI Token for authentication")

    async_openai_client = AsyncOpenAI(
        base_url=endpoint,
        api_key=token
    )

    openai_client = OpenAIResponsesClient(
        model_id=model_name,
        api_key=token,
        async_client=async_openai_client,
    )

    # create the AgentFactory with a chat client and bindings
    agent_factory = AgentFactory(
        chat_client=openai_client,
        bindings={"get_weather": get_weather},
    )
    # create the agent from the yaml
    agent = agent_factory.create_agent_from_yaml(yaml_str)
    # use the agent
    response = await agent.run("What's the weather in Amsterdam, in celsius?")
    print("Agent response:", response.text)


if __name__ == "__main__":
    asyncio.run(main())