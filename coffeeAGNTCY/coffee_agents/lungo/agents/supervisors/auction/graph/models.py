# Copyright AGNTCY Contributors (https://github.com/agntcy)
# SPDX-License-Identifier: Apache-2.0

from pydantic import BaseModel, Field
from typing import Literal

class InventoryArgs(BaseModel):
    """Arguments for the create_order tool."""
    prompt: str = Field(
        ...,
        description="The prompt to use for the broadcast. Must be a non-empty string."
    )
    farm : Literal["brazil", "colombia", "vietnam"] = Field(
        ...,
        description="The name of the farm. Must be one of 'brazil', 'colombia', or 'vietnam'."
    )

class CreateOrderArgs(BaseModel):
    """Arguments for the create_order tool."""
    farm: Literal["brazil", "colombia", "vietnam"] = Field(
        ...,
        description="The name of the farm. Must be one of 'brazil', 'colombia', or 'vietnam'."
    )
    quantity: int = Field(
        ...,
        description="The quantity of the order. Must be a positive integer."
    )
    price: float = Field(
        ...,
        description="The price of the order. Must be a positive float."
    )


# ---------------------------
# Intersight VM (generic pass-through)
# ---------------------------

class VMCommandArgs(BaseModel):
    """
    Generic VM command for the Intersight VM agent.

    Keep the supervisor decoupled: pass a natural-language prompt to the VM agent,
    which handles intent classification (create VM / snapshot VM / VM Q&A) and MCP calls.
    """
    prompt: str = Field(
        ..., min_length=1,
        description="Natural-language VM command/question for the Intersight VM agent."
    )