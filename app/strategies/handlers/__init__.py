"""Message handlers for generation strategies.

Handlers are grouped by message type and dispatch pattern.
Each handler function takes the strategy instance and message,
keeping handlers stateless and testable.
"""

from .audio_handlers import handle_audio_message
from .slot_handlers import handle_slot_message
from .destination_handlers import handle_destination_message
from .modulation_handlers import handle_modulation_message

__all__ = [
    "handle_audio_message",
    "handle_slot_message",
    "handle_destination_message",
    "handle_modulation_message",
]
