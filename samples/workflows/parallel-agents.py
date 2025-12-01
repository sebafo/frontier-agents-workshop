# Copyright (c) Microsoft. All rights reserved.

"""Agent Workflow - Content Review with Quality Routing.

This sample demonstrates:
- Using agents directly as executors
- Conditional routing based on structured outputs
- Quality-based workflow paths with convergence

Use case: Content creation with automated review.
Writer creates content, Reviewer evaluates quality:
  - High quality (score >= 80): → Publisher → Summarizer
  - Low quality (score < 80): → Editor → Publisher → Summarizer
Both paths converge at Summarizer for final report.
"""

import os
from typing import Any

from agent_framework import ChatMessage, ConcurrentBuilder
from agent_framework.openai import OpenAIChatClient
from pydantic import Field

from openai import AsyncOpenAI

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

if (os.environ.get("GITHUB_TOKEN") is not None):
    token = os.environ["GITHUB_TOKEN"]
    endpoint = "https://models.github.ai/inference"
    print("Using GitHub Token for authentication")
elif (os.environ.get("AZURE_OPENAI_API_KEY") is not None):
    token = os.environ["AZURE_OPENAI_API_KEY"]
    endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
    print("Using Azure OpenAI Token for authentication")

async_openai_client = AsyncOpenAI(
    base_url=endpoint,
    api_key=token
)

completion_model_name = os.environ.get("COMPLETION_DEPLOYMENT_NAME")
medium_model_name = os.environ.get("MEDIUM_DEPLOYMENT_MODEL_NAME")
small_model_name = os.environ.get("SMALL_DEPLOYMENT_MODEL_NAME")

completion_client=OpenAIChatClient(
    model_id = completion_model_name,
    api_key=token,
    async_client = async_openai_client
)

medium_client=OpenAIChatClient(
    model_id = medium_model_name,
    api_key=token,
    async_client = async_openai_client
)

small_client=OpenAIChatClient(
    model_id = small_model_name,
    api_key=token,
    async_client = async_openai_client
)


researcher = completion_client.create_agent(
    instructions=(
        "You're an expert market and product researcher. Given a prompt, provide concise, factual insights,"
        " opportunities, and risks."
    ),
    name="researcher",
)

marketer = small_client.create_agent(
    instructions=(
        "You're a creative marketing strategist. Craft compelling value propositions and target messaging"
        " aligned to the prompt."
    ),
    name="marketer",
)

legal = medium_client.create_agent(
    instructions=(
        "You're a cautious legal/compliance reviewer. Highlight constraints, disclaimers, and policy concerns"
        " based on the prompt."
    ),
    name="legal",
)

workflow = ConcurrentBuilder().participants([researcher, marketer, legal]).build()

def main():
    """Launch the branching workflow in DevUI."""
    import logging

    from agent_framework.devui import serve

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger(__name__)

    logger.info("Starting Agent Workflow (Content Review with Quality Routing)")
    logger.info("Available at: http://localhost:8093")
    logger.info("\nThis workflow demonstrates:")
    logger.info("- Conditional routing based on structured outputs")
    logger.info("- Path 1 (score >= 80): Reviewer → Publisher → Summarizer")
    logger.info("- Path 2 (score < 80): Reviewer → Editor → Publisher → Summarizer")
    logger.info("- Both paths converge at Summarizer for final report")

    serve(entities=[workflow], port=8093, auto_open=True)


if __name__ == "__main__":
    main()