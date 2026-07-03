"""Protocol defining what handlers can access on a strategy.

PEP 544 structural subtyping: any class with these attributes
satisfies the protocol without explicit inheritance. This decouples
handlers from specific strategy implementations.
"""

from __future__ import annotations

from typing import Optional, Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from hambajuba2ba.audio.features import StemFeatures
    from hambajuba2ba.audio.classification import ComponentClassification
    from hambajuba2ba.config.slots import BlockLinkConfig
    from hambajuba2ba.audio.sampler import AudioSampler
    from hambajuba2ba.bridge.physics_manager import PhysicsManager
    from hambajuba2ba.bridge.spatial_manager import SpatialManager
    from hambajuba2ba.bridge.steering import SteeringComputation
    from hambajuba2ba.bridge.composition import CompositionEngine
    from hambajuba2ba.bridge.clock import AudioClock


class StrategyProtocol(Protocol):
    """What handlers are allowed to touch on a strategy.

    Audio, slot, and modulation handlers use this protocol.
    Destination handlers remain typed to SAESteeringStrategy
    (prompt SLERP is SAE-specific).
    """

    # State handlers read/write
    slot_configs: dict[str, BlockLinkConfig]
    stem_features: dict[str, StemFeatures]
    stem_classifications: dict[str, Optional[ComponentClassification]]
    clock: AudioClock
    audio_sampler: Optional[AudioSampler]
    _auto_mode: bool

    # Managers handlers interact with
    _physics: Optional[PhysicsManager]
    _spatial: Optional[SpatialManager]
    _steering: Optional[SteeringComputation]
    _composition: Optional[CompositionEngine]

    # Methods handlers call
    def _get_slot_configs(self) -> dict[str, BlockLinkConfig]: ...
