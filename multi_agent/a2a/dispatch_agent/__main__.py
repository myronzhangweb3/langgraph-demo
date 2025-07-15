import logging
import sys

import click
import httpx
import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore, InMemoryPushNotifier
from a2a.types import AgentCapabilities, AgentSkill, AgentCard
from dotenv import load_dotenv

from multi_agent.a2a.dispatch_agent.agent import DispatchAgent
from multi_agent.a2a.dispatch_agent.agent_executor import DispatchAgentExecutor

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MissingAPIKeyError(Exception):
    """Exception for missing API key."""


@click.command()
@click.option('--host', 'host', default='localhost')
@click.option('--port', 'port', default=10000)
def main(host, port):
    """Starts the Currency Agent server."""
    try:
        capabilities = AgentCapabilities(streaming=True, pushNotifications=True)
        skill = AgentSkill(
            id='dispatch_agent',
            name='Dispatch Agent',
            description='Helps with dispatch task to other agents',
            tags=['dispatch agent'],
            examples=['Please help me transfer 10 USDT to my friends'],
        )
        agent_card = AgentCard(
            name='Dispatch Agent',
            description='Helps with dispatch task to other agents',
            url=f'http://{host}:{port}/',
            version='1.0.0',
            defaultInputModes=DispatchAgent.SUPPORTED_CONTENT_TYPES,
            defaultOutputModes=DispatchAgent.SUPPORTED_CONTENT_TYPES,
            capabilities=capabilities,
            skills=[skill],
        )

        # --8<-- [start:DefaultRequestHandler]
        httpx_client = httpx.AsyncClient()
        request_handler = DefaultRequestHandler(
            agent_executor=DispatchAgentExecutor(),
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
    main()
