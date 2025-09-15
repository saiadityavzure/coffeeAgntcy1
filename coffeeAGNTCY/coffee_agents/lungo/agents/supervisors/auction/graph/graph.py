# Copyright AGNTCY Contributors (https://github.com/agntcy)
# SPDX-License-Identifier: Apache-2.0

import logging
import uuid
from pydantic import BaseModel, Field

from langchain_core.prompts import PromptTemplate
from langchain_core.messages import AIMessage, SystemMessage

from langgraph.graph.state import CompiledStateGraph
from langgraph.graph import MessagesState
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from ioa_observe.sdk.decorators import agent, tool, graph

from common.llm import get_llm
from graph.tools import (
    get_farm_yield_inventory,
    get_all_farms_yield_inventory,
    create_order,
    get_order_details,
    tools_or_next,
    intersight_vm,  # NEW: Intersight VM pass-through tool
)

logger = logging.getLogger("lungo.supervisor.graph")


class NodeStates:
    SUPERVISOR = "exchange_supervisor"

    INVENTORY = "inventory_broker"
    INVENTORY_TOOLS = "inventory_tools"

    ORDERS = "orders_broker"
    ORDERS_TOOLS = "orders_tools"

    # NEW: Intersight VM routing
    VM = "vm_broker"
    VM_TOOLS = "vm_tools"

    REFLECTION = "reflection"
    GENERAL_INFO = "general"


class GraphState(MessagesState):
    """
    Represents the state of our graph, passed between nodes.
    """
    next_node: str


@agent(name="exchange_agent")
class ExchangeGraph:
    def __init__(self):
        self.graph = self.build_graph()

    @graph(name="exchange_graph")
    def build_graph(self) -> CompiledStateGraph:
        """
        Constructs and compiles a LangGraph instance.

        Agent Flow:

        supervisor_agent
            - converse with user and coordinate app flow

        inventory_agent
            - get inventory for a specific farm or broadcast to all farms

        orders_agent
            - initiate orders with a specific farm and retrieve order status

        vm_agent
            - forward Cisco Intersight VM requests (create VM, snapshot, Q&A) to the VM agent

        reflection_agent
            - determine if the user's request has been satisfied or if further action is needed
        """

        self.supervisor_llm = None
        self.reflection_llm = None
        self.inventory_llm = None
        self.orders_llm = None
        self.vm_llm = None  # NEW

        workflow = StateGraph(GraphState)

        # --- 1. Define Node States ---

        workflow.add_node(NodeStates.SUPERVISOR, self._supervisor_node)
        workflow.add_node(NodeStates.INVENTORY, self._inventory_node)
        workflow.add_node(
            NodeStates.INVENTORY_TOOLS,
            ToolNode([get_farm_yield_inventory, get_all_farms_yield_inventory]),
        )
        workflow.add_node(NodeStates.ORDERS, self._orders_node)
        workflow.add_node(
            NodeStates.ORDERS_TOOLS,
            ToolNode([create_order, get_order_details]),
        )

        # NEW: VM nodes
        workflow.add_node(NodeStates.VM, self._vm_node)
        workflow.add_node(NodeStates.VM_TOOLS, ToolNode([intersight_vm]))

        workflow.add_node(NodeStates.REFLECTION, self._reflection_node)
        workflow.add_node(NodeStates.GENERAL_INFO, self._general_response_node)

        # --- 2. Define the Agentic Workflow ---

        workflow.set_entry_point(NodeStates.SUPERVISOR)

        # Add conditional edges from the supervisor
        workflow.add_conditional_edges(
            NodeStates.SUPERVISOR,
            lambda state: state["next_node"],
            {
                NodeStates.INVENTORY: NodeStates.INVENTORY,
                NodeStates.ORDERS: NodeStates.ORDERS,
                NodeStates.VM: NodeStates.VM,  # NEW
                NodeStates.GENERAL_INFO: NodeStates.GENERAL_INFO,
            },
        )

        # Inventory tool routing
        workflow.add_conditional_edges(
            NodeStates.INVENTORY,
            tools_or_next(NodeStates.INVENTORY_TOOLS, NodeStates.REFLECTION),
        )
        workflow.add_edge(NodeStates.INVENTORY_TOOLS, NodeStates.INVENTORY)

        # Orders tool routing
        workflow.add_conditional_edges(
            NodeStates.ORDERS,
            tools_or_next(NodeStates.ORDERS_TOOLS, NodeStates.REFLECTION),
        )
        workflow.add_edge(NodeStates.ORDERS_TOOLS, NodeStates.ORDERS)

        # NEW: VM tool routing
        workflow.add_conditional_edges(
            NodeStates.VM,
            tools_or_next(NodeStates.VM_TOOLS, NodeStates.REFLECTION),
        )
        workflow.add_edge(NodeStates.VM_TOOLS, NodeStates.VM)

        workflow.add_edge(NodeStates.GENERAL_INFO, END)

        return workflow.compile()

    async def _supervisor_node(self, state: GraphState) -> dict:
        """
        Determines the intent of the user's message and routes to the appropriate node.
        """
        if not self.supervisor_llm:
            self.supervisor_llm = get_llm()

        user_message = state["messages"]

        prompt = PromptTemplate(
            template="""You are a global coffee exchange agent that can also route Cisco Intersight VM requests.
Classify the user's message as one of: 'inventory', 'orders', 'vm', or 'general'.

Respond with:
- 'inventory'  → checking yield/stock/product availability/regions of origin/specific item details
- 'orders'     → order status, placing an order, modifying an order
- 'vm'         → any Cisco Intersight / Virtual Machine request (e.g., create VM, snapshot VM, VM how-to)
- 'general'    → anything else

User message: {user_message}
""",
            input_variables=["user_message"],
        )

        chain = prompt | self.supervisor_llm
        response = chain.invoke({"user_message": user_message})
        intent = response.content.strip().lower()

        logger.info(f"Supervisor decided: {intent}")

        if "inventory" in intent:
            return {"next_node": NodeStates.INVENTORY, "messages": user_message}
        elif "orders" in intent:
            return {"next_node": NodeStates.ORDERS, "messages": user_message}
        elif "vm" in intent or "virtual machine" in str(user_message).lower() or "intersight" in str(user_message).lower():
            return {"next_node": NodeStates.VM, "messages": user_message}
        else:
            return {"next_node": NodeStates.GENERAL_INFO, "messages": user_message}

    async def _reflection_node(self, state: GraphState) -> dict:
        """
        Reflect on the conversation to determine if the user's query has been satisfied
        or if further action is needed.
        """
        if not self.reflection_llm:
            class ShouldContinue(BaseModel):
                should_continue: bool = Field(description="Whether to continue processing the request.")
                reason: str = Field(description="Reason for decision whether to continue the request.")

            # create a structured output LLM for reflection
            self.reflection_llm = get_llm().with_structured_output(ShouldContinue, strict=True)

        sys_msg_reflection = SystemMessage(
            content="""Decide whether the user query has been satisifed or if we need to continue.
Do not continue if the last message is a question or requires user input.
""",
            pretty_repr=True,
        )

        response = await self.reflection_llm.ainvoke(
            [sys_msg_reflection] + state["messages"]
        )
        logging.info(f"Reflection agent response: {response}")

        is_duplicate_message = (
            len(state["messages"]) > 2 and state["messages"][-1].content == state["messages"][-3].content
        )

        should_continue = response.should_continue and not is_duplicate_message
        next_node = NodeStates.SUPERVISOR if should_continue else END
        logging.info(f"Next node: {next_node}")

        return {
            "next_node": next_node,
            "messages": [SystemMessage(content=response.reason)],
        }

    async def _inventory_node(self, state: GraphState) -> dict:
        """
        Handles inventory-related queries using an LLM to formulate responses.
        """
        if not self.inventory_llm:
            self.inventory_llm = get_llm().bind_tools(
                [get_farm_yield_inventory, get_all_farms_yield_inventory],
                strict=True,
            )

        # get latest HumanMessage
        user_msg = next((m for m in reversed(state["messages"]) if m.type == "human"), None)
        # get latest ToolMessage
        tool_msg = next((m for m in reversed(state["messages"]) if m.type == "tool"), None)

        if tool_msg:
            context = f"Tool responded: {tool_msg.content}"
        else:
            context = "Tool has not yet responded"

        prompt = PromptTemplate(
            template="""You are an inventory broker for a global coffee exchange company.
Your task is to provide accurate and concise information about coffee yields and inventory based on user queries.

If the user asks about how much coffee we have, what the yield is or general coffee inventory, use the provided tools.
If no farm was specified, use the get_all_farms_yield_inventory tool to get the total yield across all farms.
If the user asks about a specific farm, use the get_farm_yield_inventory tool to get the yield for that farm.

If the user asks where we have coffee available, get the yield from all farms and respond with the total yield across all farms.

User question: {user_message}

{tool_context}
If the tool has answered, summarize it to the user. Otherwise ask again.
""",
            input_variables=["user_message", "tool_context"],
        )

        chain = prompt | self.inventory_llm

        llm_response = chain.invoke(
            {
                "user_message": user_msg,
                "tool_context": context,
            }
        )

        return {"messages": [llm_response]}

    async def _orders_node(self, state: GraphState) -> dict:
        if not self.orders_llm:
            self.orders_llm = get_llm().bind_tools([create_order, get_order_details])

        prompt = PromptTemplate(
            template="""You are an orders broker for a global coffee exchange company.
Your task is to handle user requests related to placing and checking orders with coffee farms.

If the issue is related to identity verification, respond with a short reply:
'The badge of this <current_farm> farm agent has not been found or could not be verified, and hence the order request failed.'
Do not ask further questions in this case.

If the user asks about placing an order, use the provided tools to create an order.
If the user asks about checking the status of an order, use the provided tools to retrieve order details.
If an order has been created, do not create a new order for the same request.
If further information is needed, ask the user for clarification.

User question: {user_message}
""",
            input_variables=["user_message"],
        )

        chain = prompt | self.orders_llm

        llm_response = chain.invoke(
            {
                "user_message": state["messages"],
            }
        )
        if llm_response.tool_calls:
            logger.info(f"Tool calls detected from orders_node: {llm_response.tool_calls}")
            logger.debug(f"Messages: {state['messages']}")
        return {"messages": [llm_response]}

    async def _vm_node(self, state: GraphState) -> dict:
        """
        Handles Cisco Intersight VM requests by invoking a single generic tool (intersight_vm)
        that forwards the user's natural-language command to the VM agent.
        """
        if not self.vm_llm:
            self.vm_llm = get_llm().bind_tools([intersight_vm], strict=True)

        # latest HumanMessage
        user_msg = next((m for m in reversed(state["messages"]) if m.type == "human"), None)
        # latest ToolMessage (if any)
        tool_msg = next((m for m in reversed(state["messages"]) if m.type == "tool"), None)

        context = f"Tool responded: {tool_msg.content}" if tool_msg else "Tool has not yet responded"

        prompt = PromptTemplate(
            template="""You are a broker for Cisco Intersight VM actions and Q&A.
Always call the `intersight_vm` tool with the user's full question/command
as the single `prompt` argument.

Examples of valid user asks:
- "Create VM web-01 with 4 CPU, 8 GB, network prod-net in DevCluster."
- "Snapshot web-01 as pre-patch 'before patching'."
- "How should I size CPU vs memory for a small web server?"

If the tool has already answered, summarize its result.
Otherwise, call the tool.

User question: {user_message}

{tool_context}
""",
            input_variables=["user_message", "tool_context"],
        )

        chain = prompt | self.vm_llm
        llm_response = chain.invoke(
            {
                "user_message": user_msg,
                "tool_context": context,
            }
        )

        return {"messages": [llm_response]}

    def _general_response_node(self, state: GraphState) -> dict:
        return {
            "next_node": END,
            "messages": [AIMessage(content="I'm not sure how to handle that. Could you please clarify?")],
        }

    async def serve(self, prompt: str):
        """
        Processes the input prompt and returns a response from the graph.
        Args:
            prompt (str): The input prompt to be processed by the graph.
        Returns:
            str: The response generated by the graph based on the input prompt.
        """
        try:
            logger.debug(f"Received prompt: {prompt}")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError("Prompt must be a non-empty string.")
            result = await self.graph.ainvoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                },
                {"configurable": {"thread_id": uuid.uuid4()}},
            )

            messages = result.get("messages", [])
            if not messages:
                raise RuntimeError("No messages found in the graph response.")

            # Find the last AIMessage with non-empty content
            for message in reversed(messages):
                if isinstance(message, AIMessage) and message.content.strip():
                    logger.debug(f"Valid AIMessage found: {message.content.strip()}")
                    return message.content.strip()

            raise RuntimeError("No valid AIMessage found in the graph response.")
        except ValueError as ve:
            logger.error(f"ValueError in serve method: {ve}")
            raise ValueError(str(ve))
        except Exception as e:
            logger.error(f"Error in serve method: {e}")
            raise Exception(str(e))
