# Copyright AGNTCY Contributors (https://github.com/agntcy)
# SPDX-License-Identifier: Apache-2.0

from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)

CREATE_VM_SKILL = AgentSkill(
    id="create_vm",
    name="Create Virtual Machine",
    description=(
        "Creates a Virtual Machine in Cisco Intersight using provided parameters: "
        "vm_name, cpu, memory_gb, network, and cluster."
    ),
    tags=["intersight", "vm", "create", "ico"],
    examples=[
        "Create VM web-01 with 4 CPU, 8 GB, network prod-net in DevCluster.",
        "Provision a VM named app-02 with 8 CPU and 16 GB RAM on network backend-net in QACluster.",
        "Create a VM db-01: 16 CPU, 64 GB memory, network db-net, cluster ProdCluster.",
        "Spin up vm test-01 with 2 CPU, 4 GB, network lab-net in DevCluster.",
    ],
)

CREATE_VM_SNAPSHOT_SKILL = AgentSkill(
    id="create_vm_snapshot",
    name="Create VM Snapshot",
    description=(
        "Creates a snapshot for an existing VM in Cisco Intersight. "
        "Requires vm_name and snapshot_name; description is optional."
    ),
    tags=["intersight", "vm", "snapshot", "ico"],
    examples=[
        "Snapshot web-01 as pre-patch 'before patching'.",
        "Create snapshot nightly-2025-09-15 for vm app-02.",
        "Take a snapshot db-01 named backup-pre-upgrade with note 'prior to schema change'.",
        "Create VM snapshot test-01 as quick-save.",
    ],
)

VM_QA_SKILL = AgentSkill(
    id="vm_qa",
    name="Virtual Machine Q&A",
    description=(
        "Answers general VM questions (concepts, sizing, and how-to guidance) with a Cisco Intersight focus. "
        "For operations beyond create/snapshot, provides high-level steps rather than executing them."
    ),
    tags=["intersight", "vm", "qa", "guidance"],
    examples=[
        "How should I size CPU and memory for a small web server VM?",
        "What’s the difference between a VM restart and reset in Intersight?",
        "How do I power off a VM in Intersight?",
        "Best practices for naming VMs and snapshots?",
    ],
)

AGENT_CARD = AgentCard(
    name="Lungo Intersight VM Management",
    id="intersight-vm-manager",
    description=(
        "An AI agent for Cisco Intersight that can create virtual machines, create VM snapshots, "
        "and answer general VM questions (sizing and how-to)."
    ),
    url="",
    version="1.0.0",
    defaultInputModes=["text"],
    defaultOutputModes=["text"],
    capabilities=AgentCapabilities(streaming=True),
    skills=[CREATE_VM_SKILL, CREATE_VM_SNAPSHOT_SKILL, VM_QA_SKILL],
    supportsAuthenticatedExtendedCard=False,
)
