"""Audio playback control handlers.

Handles: AudioPlay, AudioPause, AudioSeek, AudioTimeUpdate
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Optional

from pydantic import BaseModel

from app.schemas import AudioPlay, AudioPause, AudioSeek, AudioTimeUpdate
from hambajuba2ba.audio.focus_config import get_base_stem

if TYPE_CHECKING:
    from app.strategies.protocol import StrategyProtocol

logger = logging.getLogger("uvicorn")


def handle_audio_message(
    strategy: "StrategyProtocol",
    message: BaseModel,
) -> Optional[dict]:
    """Handle audio playback messages.

    Args:
        strategy: The SAE steering strategy instance
        message: One of AudioPlay, AudioPause, AudioSeek, AudioTimeUpdate

    Returns:
        Optional response dict (currently None for audio messages)
    """
    now = time.perf_counter()

    if isinstance(message, AudioTimeUpdate):
        strategy.clock.update(message.time, now)
        return None

    if isinstance(message, AudioPlay):
        strategy.clock.play(message.time, now)
        return None

    if isinstance(message, AudioPause):
        strategy.clock.pause()
        return None

    if isinstance(message, AudioSeek):
        _handle_seek(strategy, message, now)
        return None

    return None


def _handle_seek(
    strategy: "StrategyProtocol",
    message: AudioSeek,
    now: float,
) -> None:
    """Handle audio seek with physics reset.

    Resets physics simulations to match new audio position,
    eliminating lag after seeking.
    """
    strategy.clock.seek(message.time, now)

    # Reset physics for all enabled stems
    if strategy._physics is not None and strategy.audio_sampler is not None:
        block_configs = strategy._get_slot_configs()
        for config in block_configs.values():
            if not config.enabled:
                continue

            classification = None
            if strategy.stem_classifications:
                base = get_base_stem(config.link_target)
                if base:
                    classification = strategy.stem_classifications.get(base)

            effective = config.get_effective_config(classification)
            new_value = strategy.audio_sampler.sample_link_target(
                config.link_target, message.time, effective.layer
            )
            strategy._physics.reset_for_seek(config.link_target, new_value, velocity=0.0)


def is_audio_message(message: BaseModel) -> bool:
    """Check if message is an audio playback message."""
    return isinstance(message, (AudioTimeUpdate, AudioPlay, AudioPause, AudioSeek))
