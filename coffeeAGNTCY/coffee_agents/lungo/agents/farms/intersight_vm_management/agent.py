# Copyright AGNTCY Contributors (https://github.com/agntcy)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
"""
Intersight Virtual Machine Manager Agent (Create + Snapshot + VM Q&A)
====================================================================

Capabilities:
- create_vm { vm_name, cpu, memory_gb, network, cluster }            -> MCP tool
- create_vm_snapshot { vm_name, snapshot_name, description }          -> MCP tool
- vm_qa (generic VM questions: concepts, sizing, "how to", best practices) -> LLM reply

This file maps to:
  lungo.agents.farms.intersight_vm_management.agent
"""

import logging
from typing import Dict, Any

from langgraph.graph import MessagesState, StateGraph, END
from langchain_core.messages import AIMessage
from langchain_core.prompts import PromptTemplate

from common.llm import get_llm
from agntcy_app_sdk.factory import AgntcyFactory
from ioa_observe.sdk.decorators import agent, graph

from config.config import (
    DEFAULT_MESSAGE_TRANSPORT,
    TRANSPORT_SERVER_ENDPOINT,
)

# Match the folder path: coffee_agents.lungo.agents.farms.intersight_vm_management.agent
logger = logging.getLogger("coffee_agents.lungo.agents.farms.intersight_vm_management.agent")

# Initialize a multi-protocol, multi-transport agntcy factory.
factory = AgntcyFactory("lungo.farms.intersight_vm_management", enable_tracing=True)


# --- 1) Node Names ---
class NodeStates:
    SUPERVISOR = "supervisor"
    CREATE_VM = "create_vm_node"
    SNAPSHOT_VM = "snapshot_vm_node"
    VM_QA = "vm_qa_node"
    GENERAL = "general_response_node"


# --- 2) Graph State ---
class GraphState(MessagesState):
    """State object passed between nodes."""
    next_node: str
    params: Dict[str, Any]


# --- 3) Agent ---
@agent(name="intersight_vm_manager_agent")
class IntersightVMAgent:
    def __init__(self):
        self.supervisor_llm = None
        self.create_llm = None
        self.snapshot_llm = None
        self.qa_llm = None
        self.app = self._build_graph()

    # Utilities ---------------------------------------------------------------
    @staticmethod
    def _last_user_text(state: GraphState) -> str:
        """Return plain text from the last user message."""
        msgs = state.get("messages", [])
        if not msgs:
            return ""
        last = msgs[-1]
        if isinstance(last, str):
            return last
        # LangChain Message objects
        content = getattr(last, "content", None)
        return content if isinstance(content, str) else str(last)

    def _get_client(self):
        transport_instance = factory.create_transport(
            DEFAULT_MESSAGE_TRANSPORT,
            endpoint=TRANSPORT_SERVER_ENDPOINT,
            name="default/default/mcp_client",
        )
        return factory.create_client(
            "MCP",
            agent_topic="lungo_intersight_service",  # change if your topic differs
            transport=transport_instance,
        )

    async def _call_tool(self, name: str, arguments: Dict[str, Any]) -> str:
        """Call an MCP tool and return textual output, handling stream or non-stream."""
        try:
            async with self._get_client() as client:
                # Optional discovery (debug)
                try:
                    tools = await client.list_tools()
                    logger.debug("Available tools: %s", [t.name for t in getattr(tools, "tools", [])])
                except Exception as e:
                    logger.debug("list_tools failed (non-fatal): %s", e)

                result = await client.call_tool(name=name, arguments=arguments)

                out = ""
                if hasattr(result, "__aiter__"):
                    async for chunk in result:
                        delta = chunk.choices[0].delta
                        out += delta.content or ""
                else:
                    content_list = getattr(result, "content", None)
                    if isinstance(content_list, list) and content_list:
                        item = content_list[0]
                        out = getattr(item, "text", str(item))
                return out or "(No content returned by tool)"
        except Exception as e:
            logger.exception("Tool call '%s' failed", name)
            return f"Error invoking tool '{name}': {e}"

    # Nodes ------------------------------------------------------------------
    def _supervisor_node(self, state: GraphState) -> dict:
        """Route to create_vm, snapshot_vm, or vm_qa. Fallback -> vm_qa."""
        if not self.supervisor_llm:
            self.supervisor_llm = get_llm()

        user_text = self._last_user_text(state)

        prompt = PromptTemplate(
            template=(
                """
                You are a Cisco Intersight VM supervisor. Classify the user's request into one of:
                  - "create_vm"   (vm_name, cpu, memory_gb, network, cluster)
                  - "snapshot_vm" (vm_name, snapshot_name, description?)
                  - "vm_qa"       (generic VM questions: definitions, sizing, capabilities, how-to)
                Return a SINGLE-LINE JSON: {"intent": "...", "params": {...}}.
                Only include params you are confident about; do not invent values.

                User message: {user_message}
                """
            ),
            input_variables=["user_message"],
        )
        chain = prompt | self.supervisor_llm
        response = chain.invoke({"user_message": user_text})

        import json
        intent = NodeStates.VM_QA  # default to VM_QA so generic questions are answered
        params: Dict[str, Any] = {}
        try:
            data = json.loads(str(response.content).strip())
            intent_map = {
                "create_vm": NodeStates.CREATE_VM,
                "snapshot_vm": NodeStates.SNAPSHOT_VM,
                "vm_qa": NodeStates.VM_QA,
            }
            intent = intent_map.get(str(data.get("intent", "")).lower(), NodeStates.VM_QA)
            params = data.get("params", {}) or {}
        except Exception as e:
            logger.warning("Supervisor JSON parse failed: %s; raw=%r", e, response.content)

        logger.info("Supervisor routed intent=%s params=%s", intent, params)
        return {"next_node": intent, "messages": state["messages"], "params": params}

    async def _create_vm_node(self, state: GraphState) -> dict:
        if not self.create_llm:
            self.create_llm = get_llm()

        user_text = self._last_user_text(state)
        prompt = PromptTemplate(
            template=(
                """
                Extract VM creation parameters. Fill reasonable defaults ONLY if clearly implied.
                Return JSON: { "vm_name": str, "cpu": int, "memory_gb": int, "network": str, "cluster": str }.

                User message: {user_message}
                Existing params: {params}
                """
            ),
            input_variables=["user_message", "params"],
        )
        merged = dict(state.get("params", {}))
        chain = prompt | self.create_llm
        extra = chain.invoke({"user_message": user_text, "params": str(merged)}).content

        import json
        try:
            extracted = json.loads(extra)
            merged.update({k: v for k, v in extracted.items() if v not in (None, "")})
        except Exception:
            logger.debug("Create VM param JSON parse failed; using prior params only: %s", extra)

        required = ["vm_name", "cpu", "memory_gb", "network", "cluster"]
        missing = [k for k in required if k not in merged]
        if missing:
            return {
                "messages": [
                    AIMessage(
                        f"Missing required fields for VM creation: {', '.join(missing)}. "
                        f"Please provide them (e.g., 'Create vm web-01 with 4 CPU, 8 GB, network prod-net, cluster DevCluster')."
                    )
                ]
            }

        tool_out = await self._call_tool(
            "create_vm",
            {
                "vm_name": merged["vm_name"],
                "cpu": int(merged["cpu"]),
                "memory_gb": int(merged["memory_gb"]),
                "network": merged["network"],
                "cluster": merged["cluster"],
            },
        )
        return {"messages": [AIMessage(tool_out)]}

    async def _snapshot_vm_node(self, state: GraphState) -> dict:
        if not self.snapshot_llm:
            self.snapshot_llm = get_llm()

        user_text = self._last_user_text(state)
        prompt = PromptTemplate(
            template=(
                """
                Extract snapshot parameters. Return JSON: { "vm_name": str, "snapshot_name": str, "description": str }.

                User message: {user_message}
                Existing params: {params}
                """
            ),
            input_variables=["user_message", "params"],
        )
        merged = dict(state.get("params", {}))
        chain = prompt | self.snapshot_llm
        extra = chain.invoke({"user_message": user_text, "params": str(merged)}).content

        import json
        try:
            extracted = json.loads(extra)
            merged.update({k: v for k, v in extracted.items() if v not in (None, "")})
        except Exception:
            logger.debug("Snapshot param JSON parse failed; using prior params only: %s", extra)

        if not merged.get("vm_name") or not merged.get("snapshot_name"):
            return {"messages": [AIMessage("Please provide vm_name and snapshot_name for creating a snapshot.")]} 

        tool_out = await self._call_tool(
            "create_vm_snapshot",
            {
                "vm_name": merged["vm_name"],
                "snapshot_name": merged["snapshot_name"],
                "description": merged.get("description", ""),
            },
        )
        return {"messages": [AIMessage(tool_out)]}

    def _vm_qa_node(self, state: GraphState) -> dict:
        """Answer generic VM questions (concepts, sizing, 'how to'), scoped to Intersight."""
        if not self.qa_llm:
            self.qa_llm = get_llm()

        user_text = self._last_user_text(state)
        prompt = PromptTemplate(
            template=(
                """
                You are a Cisco Intersight Virtual Machine assistant.
                - Answer the user's VM question clearly and concisely.
                - If the question is about operations other than *creating a VM* or *creating a snapshot*,
                  give high-level steps for how to perform it in Intersight (UI and/or API), and explicitly state:
                  "This agent can currently execute only create_vm and create_vm_snapshot."
                - For sizing/best-practices questions, give 3–6 short bullet recommendations.
                - Do not fabricate exact product SKUs, costs, or environment-specific limits.

                User question: {user_message}
                """
            ),
            input_variables=["user_message"],
        )
        chain = prompt | self.qa_llm
        out = chain.invoke({"user_message": user_text}).content
        return {"messages": [AIMessage(out)]}

    def _general_response_node(self, state: GraphState) -> dict:
        response = (
            "I can create VMs and create VM snapshots in Cisco Intersight, and I can also answer general "
            "VM questions. Try: 'Create VM web-01 with 4 CPU, 8 GB, network prod-net in DevCluster', "
            "'Snapshot web-01 as pre-patch \"before patching\"', or ask a VM question like "
            "'How should I size CPU vs memory for a small web server?'."
        )
        return {"messages": [AIMessage(response)]}

    # Build graph ------------------------------------------------------------
    @graph(name="intersight_vm_manager_graph")
    def _build_graph(self):
        workflow = StateGraph(GraphState)

        workflow.add_node(NodeStates.SUPERVISOR, self._supervisor_node)
        workflow.add_node(NodeStates.CREATE_VM, self._create_vm_node)
        workflow.add_node(NodeStates.SNAPSHOT_VM, self._snapshot_vm_node)
        workflow.add_node(NodeStates.VM_QA, self._vm_qa_node)
        workflow.add_node(NodeStates.GENERAL, self._general_response_node)

        workflow.set_entry_point(NodeStates.SUPERVISOR)

        workflow.add_conditional_edges(
            NodeStates.SUPERVISOR,
            lambda state: state["next_node"],
            {
                NodeStates.CREATE_VM: NodeStates.CREATE_VM,
                NodeStates.SNAPSHOT_VM: NodeStates.SNAPSHOT_VM,
                NodeStates.VM_QA: NodeStates.VM_QA,
                NodeStates.GENERAL: NodeStates.GENERAL,
            },
        )

        workflow.add_edge(NodeStates.CREATE_VM, END)
        workflow.add_edge(NodeStates.SNAPSHOT_VM, END)
        workflow.add_edge(NodeStates.VM_QA, END)
        workflow.add_edge(NodeStates.GENERAL, END)

        return workflow.compile()

    # --- Public API ---------------------------------------------------------
    async def ainvoke(self, user_message: str) -> dict:
        inputs = {"messages": [user_message], "params": {}}
        result = await self.app.ainvoke(inputs)

        messages = result.get("messages", [])
        if not messages:
            raise RuntimeError("No messages found in the graph response.")

        for message in reversed(messages):
            if isinstance(message, AIMessage) and str(message.content).strip():
                logger.debug("Final AIMessage: %s", message.content)
                return message.content.strip()

        return messages[-1].content.strip() if messages else "No valid response generated."
