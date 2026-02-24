import asyncio
import logging
import random

import uvicorn
from dotenv import load_dotenv
from fastmcp import FastMCP

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("BankingAPIs")

mcp = FastMCP("BankingAPIs")

# Use Streamable HTTP transport (recommended for web deployments)
streamable_http_app = mcp.http_app(path="/mcp", transport="streamable-http")


@mcp.resource("config://version")
def get_version() -> dict:
    return {
        "version": "1.0.0",
        "features": ["tools", "resources"],
    }


@mcp.tool()
def get_account_balance() -> float:
    """Retrieves the current account balance for the user in USD.
    This operation is read-only and does not require approval.
    """
    logger.info("Tool called: get_account_balance")
    balance = round(random.uniform(1000, 5000), 2)
    logger.info(f"Tool completed: get_account_balance | balance={balance}")
    return balance


@mcp.tool()
def submit_payment(amount: float, recipient: str, reference: str) -> str:
    """Submit a payment request.

    Args:
        amount: Payment amount in USD
        recipient: Recipient name or vendor ID
        reference: Short description for the payment reference
    """
    logger.info(f"Tool called: submit_payment | amount={amount}, recipient={recipient}, reference={reference}")
    result = (
        f"Payment of ${amount:.2f} to '{recipient}' has been submitted "
        f"with reference '{reference}'."
    )
    logger.info(f"Tool completed: submit_payment | result={result}")
    return result


async def check_mcp(mcp: FastMCP):
    tools = await mcp.get_tools()
    resources = await mcp.get_resources()
    templates = await mcp.get_resource_templates()

    print(f"{len(tools)} Tool(s): {', '.join([t.name for t in tools.values()])}")
    print(f"{len(resources)} Resource(s): {', '.join([r.name for r in resources.values()])}")
    print(f"{len(templates)} Resource Template(s): {', '.join([t.name for t in templates.values()])}")

    return mcp


if __name__ == "__main__":
    try:
        asyncio.run(check_mcp(mcp))
        uvicorn.run(streamable_http_app, host="0.0.0.0", port=8004)
    except KeyboardInterrupt:
        print("\nProgram interrupted by user. Cleaning up...")
    except Exception as e:
        print(f"An error occurred: {e}")
