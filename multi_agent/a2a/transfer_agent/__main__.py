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

from multi_agent.a2a.transfer_agent.agent import TransferAgent
from multi_agent.a2a.transfer_agent.agent_executor import TransferAgentExecutor

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MissingAPIKeyError(Exception):
    """Exception for missing API key."""


@click.command()
@click.option('--host', 'host', default='localhost')
@click.option('--port', 'port', default=10001)
def main(host, port):
    """Starts the Currency Agent server."""
    try:
        capabilities = AgentCapabilities(streaming=True, pushNotifications=True)
        skill = AgentSkill(
            id='transfer_agent',
            name='Transfer Agent',
            description='Helps with transfer token',
            tags=['transfer agent'],
            examples=['What is exchange rate between USD and GBP?'],
        )
        agent_card = AgentCard(
            name='Transfer Agent',
            description='Helps with transfer token',
            url=f'http://{host}:{port}/',
            version='1.0.0',
            defaultInputModes=TransferAgent.SUPPORTED_CONTENT_TYPES,
            defaultOutputModes=TransferAgent.SUPPORTED_CONTENT_TYPES,
            capabilities=capabilities,
            skills=[skill],
        )

        # --8<-- [start:DefaultRequestHandler]
        httpx_client = httpx.AsyncClient()
        request_handler = DefaultRequestHandler(
            agent_executor=TransferAgentExecutor(),
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
