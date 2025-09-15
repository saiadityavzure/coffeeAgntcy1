# Copyright AGNTCY Contributors (https://github.com/agntcy)
# SPDX-License-Identifier: Apache-2.0

import logging

from services.identity_service_impl import IdentityServiceImpl

from config.config import (
    IDENTITY_INTERSIGHT_VM_AGENT_SERVICE_API_KEY,
    INTERSIGHT_VM_AGENT_URL,
    IDENTITY_API_KEY,
    IDENTITY_API_SERVER_URL,
)

logger = logging.getLogger("coffee_agents.lungo.agents.farms.intersight_vm_management.utils")


async def create_badge_for_intersight_vm_manager():
    """Create a badge for the Intersight VM Manager after the HTTP server starts and is ready."""
    try:
        identity_service = IdentityServiceImpl(
            api_key=IDENTITY_API_KEY,
            base_url=IDENTITY_API_SERVER_URL,
        )
        badge_output = await identity_service.create_badge(
            agent_url=INTERSIGHT_VM_AGENT_URL,
            svc_api_key=IDENTITY_INTERSIGHT_VM_AGENT_SERVICE_API_KEY,
        )
        logger.info("Creating badge for Intersight VM Manager: %s", badge_output)
    except Exception as e:
        logger.error("Failed to create badge for Intersight VM Manager: %s", e)
