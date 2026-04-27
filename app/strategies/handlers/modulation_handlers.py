"""Modulation and control handlers.

Handles: SetSteeringMode
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from pydantic import BaseModel

from app.schemas import SetSteeringMode

if TYPE_CHECKING:
    from app.strategies.protocol import StrategyProtocol

logger = logging.getLogger("uvicorn")


def handle_modulation_message(
    strategy: "StrategyProtocol",
    message: BaseModel,
) -> Optional[dict]:
    """Handle modulation and control messages."""
    if isinstance(message, SetSteeringMode):
        return _handle_set_steering_mode(strategy, message)

    return None


def _handle_set_steering_mode(
    strategy: "StrategyProtocol",
    message: SetSteeringMode,
) -> None:
    """Set AUTO/MANUAL steering mode."""
    strategy._auto_mode = (message.mode == "auto")
    logger.info(
        f"SetSteeringMode: mode={message.mode}, "
        f"auto-config {'enabled' if strategy._auto_mode else 'disabled'}"
    )
    return None


def is_modulation_message(message: BaseModel) -> bool:
    """Check if message is a modulation message."""
    return isinstance(message, SetSteeringMode)
