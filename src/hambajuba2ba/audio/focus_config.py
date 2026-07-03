"""Deprecated location — the slot contract moved to hambajuba2ba.config.slots.

This shim keeps old imports working for one release (the `rae` branch and
any external code import from here). New code imports from config.slots.
"""

from hambajuba2ba.config.slots import (
    DANCE_MODEL_DEFAULTS,
    DEFAULT_BLOCK_CONFIGS,
    VALID_RANKS,
    BlockLinkConfig,
    FocusConfig,
    SpatialModeType,
    derive_focus_config,
    get_base_stem,
)

__all__ = [
    "DANCE_MODEL_DEFAULTS",
    "DEFAULT_BLOCK_CONFIGS",
    "VALID_RANKS",
    "BlockLinkConfig",
    "FocusConfig",
    "SpatialModeType",
    "derive_focus_config",
    "get_base_stem",
]
