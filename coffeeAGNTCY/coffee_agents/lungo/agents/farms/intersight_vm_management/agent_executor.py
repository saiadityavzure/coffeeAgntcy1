# Copyright AGNTCY Contributors (https://github.com/agntcy)
# SPDX-License-Identifier: Apache-2.0

import logging
from uuid import uuid4

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.utils.errors import ServerError
from a2a.types import (
    UnsupportedOperationError,
    JSONRPCResponse,
    ContentTypeNotSupportedError,
    InternalError,
    Message,
    Role,
    Part,
    TextPart,
    Task,
)
from a2a.utils import new_task

from agent import IntersightVMAgent
from card import AGENT_CARD

logger = logging.getLogger("coffee_agents.lungo.agents.farms.intersight_vm_management.agent_executor")


class IntersightVMAgentExecutor(AgentExecutor):
    """
    Executes the Intersight VM Management agent:
      - create_vm (tool call)
      - create_vm_snapshot (tool call)
      - generic VM Q&A (LLM response)
    """

    def __init__(self):
        self.agent = IntersightVMAgent()
        self.agent_card = AGENT_CARD.model_dump(mode="json", exclude_none=True)

    def _validate_request(self, context: RequestContext) -> JSONRPCResponse | None:
        """Validate the incoming request structure."""
        if not context or not context.message or not context.message.parts:
            logger.error("Invalid request parameters: %s", context)
            return JSONRPCResponse(error=ContentTypeNotSupportedError())
        return None

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """
        Handle a single agent invocation:
          1) Validate request
          2) Ensure a Task exists (create if missing)
          3) Invoke the VM agent with the user's text
          4) Emit the agent Message to the event queue
        """
        logger.debug("Received message request: %s", context.message)

        validation_error = self._validate_request(context)
        if validation_error:
            await event_queue.enqueue_event(validation_error)
            return

        prompt = context.get_user_input()
        task = context.current_task
        if not task:
            task = new_task(context.message)
            await event_queue.enqueue_event(task)

        try:
            output = await self.agent.ainvoke(prompt)

            message = Message(
                message_id=str(uuid4()),
                role=Role.agent,
                metadata={"name": self.agent_card["name"]},
                parts=[Part(TextPart(text=output))],
            )

            logger.info("agent output message: %s", message)
            await event_queue.enqueue_event(message)

        except Exception as e:
            logger.error("Error while executing VM agent: %s", e)
            raise ServerError(error=InternalError()) from e

    async def cancel(
        self, request: RequestContext, event_queue: EventQueue
    ) -> Task | None:
        """Cancel this agent's execution for the given request context."""
        raise ServerError(error=UnsupportedOperationError())
