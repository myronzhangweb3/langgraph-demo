from collections.abc import Callable
from uuid import uuid4

import httpx

from a2a.client import A2AClient
from a2a.types import (
    AgentCard,
    JSONRPCErrorResponse,
    Message,
    MessageSendParams,
    SendMessageRequest,
    SendStreamingMessageRequest,
    Task,
    TaskArtifactUpdateEvent,
    TaskStatusUpdateEvent,
)
from a2a.utils import new_agent_text_message

TaskCallbackArg = Task | TaskStatusUpdateEvent | TaskArtifactUpdateEvent
TaskUpdateCallback = Callable[[TaskCallbackArg, AgentCard], Task]


class RemoteAgentConnections:
    """A class to hold the connections to the remote agents."""

    def __init__(self, client: httpx.AsyncClient, agent_card: AgentCard):
        self.agent_client = A2AClient(client, agent_card)
        self.card = agent_card
        self.pending_tasks = set()

    def get_agent(self) -> AgentCard:
        return self.card

    async def send_message(
        self,
        request: MessageSendParams,
        task_callback: TaskUpdateCallback | None,
    ) -> Task | Message | None:
        if self.card.capabilities.streaming:
            task = None
            async for response in self.agent_client.send_message_streaming(
                SendStreamingMessageRequest(id=str(uuid4()), params=request)
            ):
                if not response.root.result:
                    return response.root.error
                # In the case a message is returned, that is the end of the interaction.
                event = response.root.result
                if isinstance(event, Message):
                    return event
                if isinstance(event, TaskArtifactUpdateEvent):
                    return new_agent_text_message(text=event.artifact.parts[0].root.text, context_id=event.contextId, task_id=event.taskId)

                # Otherwise we are in the Task + TaskUpdate cycle.
                if task_callback and event:
                    task = task_callback(event, self.card)
                if hasattr(event, 'final') and event.final:
                    return new_agent_text_message(text=event.status.message.parts[0].root.text, context_id=event.contextId, task_id=event.taskId)
            return task
        # Non-streaming
        response = await self.agent_client.send_message(
            SendMessageRequest(id=str(uuid4()), params=request)
        )
        if isinstance(response.root, JSONRPCErrorResponse):
            return response.root.error
        if isinstance(response.root.result, Message):
            return response.root.result

        if task_callback:
            task_callback(response.root.result, self.card)
        return response.root.result
