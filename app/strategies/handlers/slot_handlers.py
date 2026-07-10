"""Slot configuration handlers.

Handles: UpdateSlotConfig (and its legacy alias UpdateBlockConfig).
One handler serves every backend: it applies the message's non-None
fields to strategy.slot_configs[slot], which the strategy base gives
every backend. Fields a backend doesn't use are stored and ignored.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from pydantic import BaseModel

from app.schemas import UpdateSlotConfig
from hambajuba2ba.config.slots import BlockLinkConfig

if TYPE_CHECKING:
    from app.strategies.protocol import StrategyProtocol

logger = logging.getLogger("uvicorn")


def handle_slot_message(
    strategy: "StrategyProtocol",
    message: BaseModel,
) -> Optional[dict]:
    """Handle slot configuration messages."""
    # isinstance covers the legacy UpdateBlockConfig subclass too
    if isinstance(message, UpdateSlotConfig):
        return _handle_update_slot_config(strategy, message)
    return None


def _handle_update_slot_config(
    strategy: "StrategyProtocol",
    message: UpdateSlotConfig,
) -> None:
    """Apply the message's non-None fields to the slot's config."""
    config = strategy.slot_configs.get(message.slot)
    if config is None:
        config = BlockLinkConfig(
            block=message.slot,
            feature_id=message.feature_id or 0,
        )
        strategy.slot_configs[message.slot] = config

    # Track if link_target/physics changed (may require physics re-init)
    link_target_before = config.link_target
    physics_before = config.physics_preset

    # Apply updates
    if message.link_target is not None:
        config.link_target = message.link_target
    if message.strength_min is not None:
        config.strength_min = message.strength_min
    if message.strength_max is not None:
        config.strength_max = message.strength_max
    if message.feature_id is not None:
        config.feature_id = message.feature_id
    if message.enabled is not None:
        config.enabled = message.enabled
    if message.auto_config is not None:
        config.auto_config = message.auto_config
    if "sae_rank" in message.model_fields_set:
        config.sae_rank = message.sae_rank
    if message.spatial_mode is not None:
        config.spatial_mode = message.spatial_mode
    if message.channel is not None:
        config.channel = message.channel
    if message.layer is not None:
        config.layer = message.layer
    if message.physics_preset is not None:
        preset = message.physics_preset
        if preset == "default":
            preset = "ambient"
        config.physics_preset = preset
    if message.spatial_mask is not None:
        config.spatial_mask = message.spatial_mask
        # Immediately upload to GPU for next frame
        if hasattr(strategy, '_spatial') and strategy._spatial is not None:
            strategy._spatial.set_block_mask(message.slot, message.spatial_mask)
    if message.stage_left is not None:
        config.stage_left = message.stage_left
    if message.stage_home is not None:
        config.stage_home = message.stage_home
    if message.stage_right is not None:
        config.stage_right = message.stage_right
    if message.position_source is not None:
        config.position_source = message.position_source
    if message.intensity_source is not None:
        config.intensity_source = message.intensity_source
    if message.position_smoothing_ms is not None:
        config.position_smoothing_ms = message.position_smoothing_ms
    if message.silence_behavior is not None:
        config.silence_behavior = message.silence_behavior
    if message.drift_ms is not None:
        config.drift_ms = message.drift_ms
    if message.intensity_curve is not None:
        config.intensity_curve = message.intensity_curve
    if message.intensity_gamma is not None:
        config.intensity_gamma = message.intensity_gamma

    # Auto defaults for dance model when auto_config is on and link target changes
    if message.link_target is not None and config.auto_config:
        classification = strategy.stem_classifications.get(config.link_target) if strategy.stem_classifications else None
        is_percussive = False
        if classification is not None:
            is_percussive = classification.percussive_confidence > 0.6
        else:
            is_percussive = (
                config.link_target.startswith("drums")
                or config.link_target.endswith("_percussive")
            )

        if is_percussive:
            config.intensity_source = "transient"
            config.intensity_curve = "linear"
        else:
            config.intensity_source = "energy_smooth"
            config.intensity_curve = "linear"

    # Reinitialize physics if link_target or physics preset changed
    if strategy._physics is not None and strategy.stem_features:
        if config.link_target != link_target_before or config.physics_preset != physics_before:
            try:
                strategy._physics.initialize(
                    strategy.slot_configs,
                    strategy.stem_features,
                    strategy.stem_classifications,
                )
            except Exception:
                logger.exception("Failed to reinitialize physics after slot update")

    logger.info(
        f"UpdateSlotConfig: slot={message.slot}, link_target={config.link_target}, "
        f"feature_id={config.feature_id}, enabled={config.enabled}, "
        f"auto_config={config.auto_config}, sae_rank={config.sae_rank}"
    )
    return None


def is_slot_message(message: BaseModel) -> bool:
    """Check if message is a slot configuration message."""
    return isinstance(message, UpdateSlotConfig)
