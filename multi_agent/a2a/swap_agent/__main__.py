import asyncio
import logging
import sys

import httpx
import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore, InMemoryPushNotifier
from a2a.types import AgentCapabilities, AgentSkill, AgentCard
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient

from multi_agent.a2a.swap_agent.agent import SwapAgent
from multi_agent.a2a.swap_agent.agent_executor import SwapAgentExecutor

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MissingAPIKeyError(Exception):
    """Exception for missing API key."""


async def main(host, port):
    """Starts the Currency Agent server."""
    try:
        capabilities = AgentCapabilities(streaming=True, pushNotifications=True)
        skill = AgentSkill(
            id='swap_agent',
            name='Swap Agent',
            description='Helps with swap token',
            tags=['swap agent'],
            examples=['swap 1 USDT to ETH'],
        )
        agent_card = AgentCard(
            name='Swap Agent',
            description='Helps with swap token',
            url=f'http://{host}:{port}/',
            version='1.0.0',
            defaultInputModes=SwapAgent.SUPPORTED_CONTENT_TYPES,
            defaultOutputModes=SwapAgent.SUPPORTED_CONTENT_TYPES,
            capabilities=capabilities,
            skills=[skill],
        )

        # --8<-- [start:DefaultRequestHandler]
        tools = await MultiServerMCPClient(
            {
                "analysis": {
                    "url": "http://36.189.252.2:18133/mcp/swap",
                    "transport": "streamable_http",
                }
            }
        ).get_tools()
        httpx_client = httpx.AsyncClient()
        request_handler = DefaultRequestHandler(
            agent_executor=SwapAgentExecutor(tools),
            task_store=InMemoryTaskStore(),
            push_notifier=InMemoryPushNotifier(httpx_client),
        )
        server = A2AStarletteApplication(
            agent_card=agent_card, http_handler=request_handler
        )

        uvicorn.run(server.build(), host=host, port=port)
        # --8<-- [end:DefaultRequestHandler]

    except MissingAPIKeyError as e:
        logger.error(f'Error: {e}')
        sys.exit(1)
    except Exception as e:
        logger.error(f'An error occurred during server startup: {e}')
        sys.exit(1)


if __name__ == '__main__':
    asyncio.run(main('localhost', 10002))
